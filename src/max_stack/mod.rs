//! Analysis of the maximum amount of stack used by a function.
//!
//! This is a single pass linear-time algorithm.

pub use self::error::Error;
use crate::visitors;
use instruction_visit::Output;
use prefix_sum_vec::{PrefixSumVec, TryPushError};
use wasmparser::{BlockType, ValType};

mod error;
mod instruction_visit;
#[cfg(test)]
mod test;

/// The information about the whole-module necessary for the `max_stack` analysis.
///
/// This structure maintains the information gathered when parsing type, globals, tables and
/// function sections.
pub struct ModuleState {
    pub(crate) functions: Vec<u32>,
    pub(crate) types: Vec<wasmparser::Type>,
    pub(crate) globals: Vec<wasmparser::ValType>,
    pub(crate) tables: Vec<wasmparser::ValType>,
}

impl ModuleState {
    pub fn new() -> Self {
        Self {
            functions: vec![],
            types: vec![],
            globals: vec![],
            tables: vec![],
        }
    }
}

/// The per-function state used by the [`Visitor`].
///
/// This type maintains the state accumulated during the analysis of a single function in a module.
/// If the same instance of this `FunctionState` is used to analyze multiple functions, it will
/// result in re-use of the backing allocations, and thus an improved performance. However, make
/// sure to call [`FunctionState::clear`] between functions!
pub struct FunctionState {
    locals: PrefixSumVec<wasmparser::ValType, u32>,

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
    operands: Vec<u8>,
    /// Sum of all values in the `operands` field above.
    size: u64,
    /// Maximum observed value for the `size` field above.
    pub(crate) max_size: u64,

    /// The stack of frames (as created by operations such as `block`).
    frames: Vec<Frame>,
    /// The top-most frame.
    ///
    /// This aids quicker access at a cost of some special-casing for the very last `end` operator.
    current_frame: Frame,
}

impl FunctionState {
    pub fn new() -> Self {
        Self {
            locals: PrefixSumVec::new(),
            operands: vec![],
            size: 0,
            max_size: 0,
            frames: vec![],
            current_frame: Frame {
                height: 0,
                block_type: BlockType::Empty,
                stack_polymorphic: false,
            },
        }
    }

    pub fn clear(&mut self) {
        self.locals.clear();
        self.operands.clear();
        self.size = 0;
        self.max_size = 0;
        self.frames.clear();
        self.current_frame = Frame {
            height: 0,
            block_type: BlockType::Empty,
            stack_polymorphic: false,
        };
    }

    pub fn add_locals(&mut self, count: u32, ty: wasmparser::ValType) -> Result<(), TryPushError> {
        self.locals.try_push_more(count, ty)
    }
}

/// Configure size of various values that may end up on the stack.
pub trait SizeConfig {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8;
    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64;
}

/// The configuration for the stack analysis.
///
/// Note that this trait is not intended to implement directly. Implement [`SizeConfig`]
/// instead. Implementers of `SizeConfig` trait will also implement `max_stack::Config` by
/// definition.
pub trait Config {
    type StackVisitor<'b, 's>: visitors::VisitOperatorWithOffset<'b, Output = Output>
    where
        Self: 's;
    fn to_visitor<'b, 's>(
        &'s self,
        module_state: &'s ModuleState,
        function_state: &'s mut FunctionState,
    ) -> Self::StackVisitor<'b, 's>;
    fn frame_size(&self, function_state: &FunctionState) -> u64;
}

impl<S: SizeConfig> Config for S {
    type StackVisitor<'b, 's> = Visitor<'s, Self> where Self: 's;

    fn to_visitor<'b, 's>(
        &'s self,
        module_state: &'s ModuleState,
        function_state: &'s mut FunctionState,
    ) -> Self::StackVisitor<'b, 's> {
        Visitor {
            offset: 0,
            config: self,
            module_state,
            function_state,
        }
    }

    fn frame_size(&self, function_state: &FunctionState) -> u64 {
        self.size_of_function_activation(&function_state.locals)
    }
}

/// Disable the max stack analysis entirely.
pub struct NoConfig;
impl Config for NoConfig {
    type StackVisitor<'b, 's> = visitors::NoOpVisitor<Output>;

    fn to_visitor<'b, 's>(
        &'s self,
        _: &'s ModuleState,
        _: &'s mut FunctionState,
    ) -> Self::StackVisitor<'b, 's> {
        visitors::NoOpVisitor(Ok(()))
    }

    fn frame_size(&self, _: &FunctionState) -> u64 {
        0
    }
}

impl<'a, C: SizeConfig + ?Sized> SizeConfig for &'a C {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(*self, locals)
    }
}

impl<'a, C: SizeConfig + ?Sized> SizeConfig for &'a mut C {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(*self, locals)
    }
}

impl<'a, C: SizeConfig + ?Sized> SizeConfig for Box<C> {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(&*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(&*self, locals)
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

/// The core algorihtm of the `max_stack` analysis.
pub struct Visitor<'s, Cfg: ?Sized> {
    pub(crate) offset: usize,

    pub(crate) config: &'s Cfg,

    pub(crate) module_state: &'s ModuleState,
    pub(crate) function_state: &'s mut FunctionState,
}

impl<'b, 's, Cfg: SizeConfig + ?Sized> Visitor<'s, Cfg> {
    fn function_type_index(&self, function_index: u32) -> Result<u32, Error> {
        let function_index_usize = usize::try_from(function_index)
            .map_err(|e| Error::FunctionIndexRange(function_index, e))?;
        self.module_state
            .functions
            .get(function_index_usize)
            .copied()
            .ok_or(Error::FunctionIndex(function_index))
    }

    fn type_params_results(&self, type_idx: u32) -> Result<(&'s [ValType], &'s [ValType]), Error> {
        let type_idx_usize =
            usize::try_from(type_idx).map_err(|e| Error::TypeIndexRange(type_idx, e))?;
        match self
            .module_state
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
        let stack_polymorphic = self.function_state.current_frame.stack_polymorphic;
        let height = self
            .function_state
            .operands
            .len()
            .checked_sub(shift_operands)
            .ok_or(Error::EmptyStack(self.offset))?;
        self.function_state.frames.push(std::mem::replace(
            &mut self.function_state.current_frame,
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
        if let Some(frame) = self.function_state.frames.pop() {
            let frame = std::mem::replace(&mut self.function_state.current_frame, frame);
            let to_pop = self
                .function_state
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
        self.function_state.current_frame.stack_polymorphic = true;
    }

    fn push(&mut self, t: ValType) {
        if !self.function_state.current_frame.stack_polymorphic {
            let value_size = self.config.size_of_value(t);
            self.function_state.operands.push(value_size);
            self.function_state.size += u64::from(value_size);
            self.function_state.max_size =
                std::cmp::max(self.function_state.size, self.function_state.max_size);
        }
    }

    fn pop(&mut self) -> Result<(), Error> {
        if !self.function_state.current_frame.stack_polymorphic {
            let operand_size = u64::from(
                self.function_state
                    .operands
                    .pop()
                    .ok_or(Error::EmptyStack(self.offset))?,
            );
            self.function_state.size = self
                .function_state
                .size
                .checked_sub(operand_size)
                .expect("stack size is going negative");
        }
        Ok(())
    }

    fn pop_many(&mut self, count: usize) -> Result<(), Error> {
        if count == 0 || self.function_state.current_frame.stack_polymorphic {
            Ok(())
        } else {
            let operand_count = self.function_state.operands.len();
            let split_point = operand_count
                .checked_sub(count)
                .ok_or(Error::EmptyStack(self.offset))?;
            let size: u64 = self
                .function_state
                .operands
                .drain(split_point..)
                .map(u64::from)
                .sum();
            self.function_state.size = self
                .function_state
                .size
                .checked_sub(size)
                .expect("stack size is going negative");
            Ok(())
        }
    }

    fn visit_const(&mut self, t: ValType) -> Output {
        // [] → [t]
        self.push(t);
        Ok(())
    }

    fn visit_unop(&mut self) -> Output {
        // [t] -> [t]

        // Function body intentionally left empty (pops and immediately pushes the same type back)
        Ok(())
    }

    fn visit_binop(&mut self) -> Output {
        // [t t] -> [t]
        self.pop()?;
        Ok(())
    }

    fn visit_testop(&mut self) -> Output {
        // [t] -> [i32]
        self.pop()?;
        self.push(ValType::I32);
        Ok(())
    }

    fn visit_relop(&mut self) -> Output {
        // [t t] -> [i32]
        self.pop_many(2)?;
        self.push(ValType::I32);
        Ok(())
    }

    fn visit_cvtop(&mut self, result_ty: ValType) -> Output {
        // t2.cvtop_t1_sx? : [t1] -> [t2]
        self.pop()?;
        self.push(result_ty);
        Ok(())
    }

    fn visit_vternop(&mut self) -> Output {
        // [v128 v128 v128] → [v128]
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_vrelop(&mut self) -> Output {
        // [v128 v128] -> [v128]
        self.visit_binop()
    }

    fn visit_vishiftop(&mut self) -> Output {
        // [v128 i32] -> [v128]
        self.pop()?;
        Ok(())
    }

    fn visit_vinarrowop(&mut self) -> Output {
        // [v128 v128] -> [v128]
        self.pop()?;
        Ok(())
    }

    fn visit_vbitmask(&mut self) -> Output {
        // [v128] -> [i32]
        self.pop()?;
        self.push(ValType::I32);
        Ok(())
    }

    fn visit_splat(&mut self) -> Output {
        // [unpacked(t)] -> [v128]
        self.pop()?;
        self.push(ValType::V128);
        Ok(())
    }

    fn visit_replace_lane(&mut self) -> Output {
        // shape.replace_lane laneidx : [v128 unpacked(shape)] → [v128]
        self.pop()?;
        Ok(())
    }

    fn visit_extract_lane(&mut self, unpacked_shape: ValType) -> Output {
        // txN.extract_lane_sx ? laneidx : [v128] → [unpacked(shape)]
        self.pop()?;
        self.push(unpacked_shape);
        Ok(())
    }

    fn visit_load(&mut self, t: ValType) -> Output {
        // t.load memarg           : [i32] → [t]
        // t.loadN_sx memarg       : [i32] → [t]
        // v128.loadNxM_sx memarg  : [i32] → [v128]
        // v128.loadN_splat memarg : [i32] → [v128]
        // v128.loadN_zero memarg  : [i32] → [v128]
        // t.atomic.load memarg    : [i32] → [t]
        // t.atomic.load memargN_u : [i32] → [t]
        self.pop()?;
        self.push(t);
        Ok(())
    }

    fn visit_load_lane(&mut self) -> Output {
        // v128.loadN_lane memarg laneidx : [i32 v128] → [v128]
        self.pop_many(2)?;
        self.push(ValType::V128);
        Ok(())
    }

    fn visit_store(&mut self) -> Output {
        // t.store memarg           : [i32 t] → []
        // t.storeN memarg          : [i32 t] → []
        // t.atomic.store memarg    : [i32 t] → []
        // t.atomic.store memargN_u : [i32 t] → []
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_store_lane(&mut self) -> Output {
        // v128.storeN_lane memarg laneidx : [i32 v128] → []
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_atomic_rmw(&mut self, t: ValType) -> Output {
        // t.atomic.rmw.atop memarg : [i32 t] → [t]
        // t.atomic.rmwN.atop_u memarg : [i32 t] → [t]
        self.pop_many(2)?;
        self.push(t);
        Ok(())
    }

    fn visit_atomic_cmpxchg(&mut self, t: ValType) -> Output {
        // t.atomic.rmw.cmpxchg : [i32 t t] → [t]
        // t.atomic.rmwN.cmpxchg_u : [i32 t t] → [t]
        self.pop_many(3)?;
        self.push(t);
        Ok(())
    }

    fn visit_function_call(&mut self, type_index: u32) -> Output {
        let (params, results) = self.type_params_results(type_index)?;
        self.pop_many(params.len())?;
        for result_ty in results {
            self.push(*result_ty);
        }
        Ok(())
    }
}
