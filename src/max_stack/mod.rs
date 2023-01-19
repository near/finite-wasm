//! Analysis of the maximum amount of stack used by a function.
//!
//! This is a single pass linear-time algorithm.

pub use self::error::Error;
use prefix_sum_vec::PrefixSumVec;
use wasmparser::{BlockType, ValType, VisitOperator};

mod error;
mod instruction_visit;
#[cfg(test)]
mod test;

pub(crate) mod internal {
    #[derive(Default)]
    pub struct NotOverridable;
}

/// Configure size of various values that may end up on the stack.
pub trait SizeConfig {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8;
    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64;

    #[doc(hidden)]
    fn should_run(&self, _: internal::NotOverridable) -> bool {
        true
    }
}

impl<'a, C: SizeConfig> SizeConfig for &'a C {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(*self, locals)
    }
}

/// Stack sizes are not configured, meaning the maximum stack analysis will not run at all.
pub struct NoSizeConfig;
impl SizeConfig for NoSizeConfig {
    fn size_of_value(&self, _: wasmparser::ValType) -> u8 {
        std::process::abort()
    }

    fn size_of_function_activation(&self, _: &PrefixSumVec<ValType, u32>) -> u64 {
        std::process::abort()
    }

    fn should_run(&self, _: internal::NotOverridable) -> bool {
        false
    }
}

#[derive(Debug)]
pub(crate) struct Frame {
    /// Operand stack height at the time this frame was entered.
    ///
    /// This way no matter how the operand stack is modified during the execution of this frame, we
    /// can always reset the operand stack back to this specific height when the frame terminates.
    pub(crate) height: usize,

    /// Type of the block representing this frame.
    ///
    /// The parameters are below height and get popped when the frame terminates. Results get
    /// pushed back onto the operand stack.
    pub(crate) block_type: BlockType,

    /// Is the operand stack for the remainder of this frame considered polymorphic?
    ///
    /// Once the stack becomes polymorphic, the only way for it to stop being polymorphic is to pop
    /// the frames within which the stack is polymorphic.
    ///
    /// Note, that unlike validation, for the purposes of this analysis stack polymorphism is
    /// somewhat more lax. For example, validation algorithm will readly reject a function like
    /// `(func (unreachable) (i64.const 0) (i32.add))`, because at the time `i32.add` is evaluated
    /// the stack is `[t* i64]`, which does not unify with `[i32 i32] -> [i32]` expected by
    /// `i32.add`, despite being polymorphic. For the purposes of this analysis we do not keep
    /// track of the stack contents to that level of detail – all we care about is whether the
    /// stack is polymorphic at all. We then skip any tracking of stack operations for any
    /// instruction interacting with a polymorphic stack, as those instructions are effectively
    /// unreachable.
    ///
    /// See `stack-polymorphic` in the wasm-core specification for an extended explanation.
    pub(crate) stack_polymorphic: bool,
}

pub(crate) struct StackSizeVisitor<'a, Cfg> {
    pub(crate) offset: usize,

    pub(crate) config: &'a Cfg,
    pub(crate) functions: &'a [u32],
    pub(crate) types: &'a [wasmparser::Type],
    pub(crate) globals: &'a [wasmparser::ValType],
    pub(crate) tables: &'a [wasmparser::ValType],
    pub(crate) locals: &'a PrefixSumVec<wasmparser::ValType, u32>,

    /// Sizes of the operands currently pushed to the operand stack within this function.
    ///
    /// This is an unfortunate requirement for this analysis – some instructions are
    /// parametric (that is, they don’t specify the type of values they operate on), which
    /// means that any analysis must maintain a stack of operand information in order to be
    /// able to tell what types these instructions will operate on.
    ///
    /// A particular example here is a `drop` – given a code sequence such as…
    ///
    /// ```wast
    /// (i32.const 42)
    /// (i64.const 42)
    /// drop
    /// (f32.const 42.0)
    /// ```
    ///
    /// …it is impossible to tell what is going to be the effect of `drop` on the overall
    /// maximum size of the frame unless an accurate representation of the operand stack is
    /// maintained at all times.
    ///
    /// Fortunately, we don’t exactly need to maintain the _types_, only their sizes suffice.
    pub(crate) operands: &'a mut Vec<u8>,
    /// Sum of all values in the `operands` field above.
    pub(crate) size: u64,
    /// Maximum observed value for the `size` field above.
    pub(crate) max_size: u64,

    /// The stack of frames (as created by operations such as `block`).
    pub(crate) frames: &'a mut Vec<Frame>,
    /// The top-most frame.
    ///
    /// This aids quicker access at a cost of some special-casing for the very last `end` operator.
    pub(crate) current_frame: Frame,
}

type Output<'a, V> = <V as VisitOperator<'a>>::Output;

impl<'a, Cfg: SizeConfig> StackSizeVisitor<'a, Cfg> {
    fn function_type_index(&self, function_index: u32) -> Result<u32, Error> {
        let function_index_usize = usize::try_from(function_index)
            .map_err(|e| Error::FunctionIndexRange(function_index, e))?;
        self.functions
            .get(function_index_usize)
            .copied()
            .ok_or(Error::FunctionIndex(function_index))
    }

    fn type_params_results(&self, type_idx: u32) -> Result<(&'a [ValType], &'a [ValType]), Error> {
        let type_idx_usize =
            usize::try_from(type_idx).map_err(|e| Error::TypeIndexRange(type_idx, e))?;
        match self
            .types
            .get(type_idx_usize)
            .ok_or(Error::TypeIndex(type_idx))?
        {
            wasmparser::Type::Func(fnty) => Ok((fnty.params(), fnty.results())),
        }
    }

    fn with_block_types<F>(&mut self, block_type: BlockType, cb: F) -> Result<(), Error>
    where
        F: FnOnce(&mut Self, &[ValType], &[ValType]) -> Result<(), Error>,
    {
        match block_type {
            BlockType::Empty => cb(self, &[], &[]),
            BlockType::Type(result) => cb(self, &[], &[result]),
            BlockType::FuncType(type_index) => {
                let (params, results) = self.type_params_results(type_index)?;
                cb(self, params, results)
            }
        }
    }

    /// Create a new frame on the frame stack.
    ///
    /// `shift_operands` will take the provided number of operands off the operand stack and place
    /// them into the newly created frame. This is useful for various block instructions with
    /// parameters, where the naive approach would be to pop the parameters, push the frame, and
    /// push the same parameters that have been just pooped back onto the operand stack.
    fn new_frame(&mut self, block_type: BlockType, shift_operands: usize) -> Result<(), Error> {
        let stack_polymorphic = self.current_frame.stack_polymorphic;
        let height = self
            .operands
            .len()
            .checked_sub(shift_operands)
            .ok_or(Error::EmptyStack(self.offset))?;
        self.frames.push(std::mem::replace(
            &mut self.current_frame,
            Frame {
                height,
                block_type,
                stack_polymorphic,
            },
        ));
        Ok(())
    }

    /// Terminate the current frame.
    ///
    /// If the frame is not the function-level frame, it will be returned.
    ///
    /// As part of this procedure, the operand stack will be reset to the same height at which the
    /// frame was created.
    ///
    /// Callers are responsible for pushing the block results themselves.
    fn end_frame(&mut self) -> Result<Option<Frame>, Error> {
        if let Some(frame) = self.frames.pop() {
            let frame = std::mem::replace(&mut self.current_frame, frame);
            let to_pop = self
                .operands
                .len()
                .checked_sub(frame.height)
                .ok_or(Error::TruncatedOperandStack(self.offset))?;
            self.pop_many(to_pop)?;
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }

    /// Mark the current frame as polymorphic.
    fn make_polymorphic(&mut self) {
        self.current_frame.stack_polymorphic = true;
    }

    fn push(&mut self, t: ValType) {
        if !self.current_frame.stack_polymorphic {
            let value_size = self.config.size_of_value(t);
            self.operands.push(value_size);
            self.size += u64::from(value_size);
            self.max_size = std::cmp::max(self.size, self.max_size);
        }
    }

    fn pop(&mut self) -> Result<(), Error> {
        if !self.current_frame.stack_polymorphic {
            let operand_size =
                u64::from(self.operands.pop().ok_or(Error::EmptyStack(self.offset))?);
            self.size = self
                .size
                .checked_sub(operand_size)
                .expect("stack size is going negative");
        }
        Ok(())
    }

    fn pop_many(&mut self, count: usize) -> Result<(), Error> {
        if count == 0 || self.current_frame.stack_polymorphic {
            Ok(())
        } else {
            let operand_count = self.operands.len();
            let split_point = operand_count
                .checked_sub(count)
                .ok_or(Error::EmptyStack(self.offset))?;
            let size: u64 = self.operands.drain(split_point..).map(u64::from).sum();
            self.size = self
                .size
                .checked_sub(size)
                .expect("stack size is going negative");
            Ok(())
        }
    }

    fn visit_const(&mut self, t: ValType) -> Output<Self> {
        // [] → [t]
        self.push(t);
        Ok(None)
    }

    fn visit_unop(&mut self) -> Output<Self> {
        // [t] -> [t]

        // Function body intentionally left empty (pops and immediately pushes the same type back)
        Ok(None)
    }

    fn visit_binop(&mut self) -> Output<Self> {
        // [t t] -> [t]
        self.pop()?;
        Ok(None)
    }

    fn visit_testop(&mut self) -> Output<Self> {
        // [t] -> [i32]
        self.pop()?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_relop(&mut self) -> Output<Self> {
        // [t t] -> [i32]
        self.pop_many(2)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_cvtop(&mut self, result_ty: ValType) -> Output<Self> {
        // t2.cvtop_t1_sx? : [t1] -> [t2]
        self.pop()?;
        self.push(result_ty);
        Ok(None)
    }

    fn visit_vternop(&mut self) -> Output<Self> {
        // [v128 v128 v128] → [v128]
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_vrelop(&mut self) -> Output<Self> {
        // [v128 v128] -> [v128]
        self.visit_binop()
    }

    fn visit_vishiftop(&mut self) -> Output<Self> {
        // [v128 i32] -> [v128]
        self.pop()?;
        Ok(None)
    }

    fn visit_vinarrowop(&mut self) -> Output<Self> {
        // [v128 v128] -> [v128]
        self.pop()?;
        Ok(None)
    }

    fn visit_vbitmask(&mut self) -> Output<Self> {
        // [v128] -> [i32]
        self.pop()?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_splat(&mut self) -> Output<Self> {
        // [unpacked(t)] -> [v128]
        self.pop()?;
        self.push(ValType::V128);
        Ok(None)
    }

    fn visit_replace_lane(&mut self) -> Output<Self> {
        // shape.replace_lane laneidx : [v128 unpacked(shape)] → [v128]
        self.pop()?;
        Ok(None)
    }

    fn visit_extract_lane(&mut self, unpacked_shape: ValType) -> Output<Self> {
        // txN.extract_lane_sx ? laneidx : [v128] → [unpacked(shape)]
        self.pop()?;
        self.push(unpacked_shape);
        Ok(None)
    }

    fn visit_load(&mut self, t: ValType) -> Output<Self> {
        // t.load memarg           : [i32] → [t]
        // t.loadN_sx memarg       : [i32] → [t]
        // v128.loadNxM_sx memarg  : [i32] → [v128]
        // v128.loadN_splat memarg : [i32] → [v128]
        // v128.loadN_zero memarg  : [i32] → [v128]
        // t.atomic.load memarg    : [i32] → [t]
        // t.atomic.load memargN_u : [i32] → [t]
        self.pop()?;
        self.push(t);
        Ok(None)
    }

    fn visit_load_lane(&mut self) -> Output<Self> {
        // v128.loadN_lane memarg laneidx : [i32 v128] → [v128]
        self.pop_many(2)?;
        self.push(ValType::V128);
        Ok(None)
    }

    fn visit_store(&mut self) -> Output<Self> {
        // t.store memarg           : [i32 t] → []
        // t.storeN memarg          : [i32 t] → []
        // t.atomic.store memarg    : [i32 t] → []
        // t.atomic.store memargN_u : [i32 t] → []
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_store_lane(&mut self) -> Output<Self> {
        // v128.storeN_lane memarg laneidx : [i32 v128] → []
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_atomic_rmw(&mut self, t: ValType) -> Output<Self> {
        // t.atomic.rmw.atop memarg : [i32 t] → [t]
        // t.atomic.rmwN.atop_u memarg : [i32 t] → [t]
        self.pop_many(2)?;
        self.push(t);
        Ok(None)
    }

    fn visit_atomic_cmpxchg(&mut self, t: ValType) -> Output<Self> {
        // t.atomic.rmw.cmpxchg : [i32 t t] → [t]
        // t.atomic.rmwN.cmpxchg_u : [i32 t t] → [t]
        self.pop_many(3)?;
        self.push(t);
        Ok(None)
    }

    fn visit_function_call(&mut self, type_index: u32) -> Output<Self> {
        let (params, results) = self.type_params_results(type_index)?;
        self.pop_many(params.len())?;
        for result_ty in results {
            self.push(*result_ty);
        }
        Ok(None)
    }
}
