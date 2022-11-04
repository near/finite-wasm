//! Instrumentation-based implementation of the finite-wasm specification.
//!
//! The functionality provided by this module will transform a provided WebAssembly module in a way
//! that measures gas fees and stack depth without any special support by the runtime executing the
//! code in question.

pub use self::error::Error;
use prefix_sum_vec::PrefixSumVec;
use wasmparser::{BlockType, ValType, VisitOperator};

mod error;
mod instruction_visit;
#[cfg(test)]
mod test;

/// The results of parsing and analyzing the module.
///
/// This analysis collects information necessary to implement all of the transformations in one go,
/// so that re-parsing the module multiple times is not necessary.
pub struct Module {
    /// The sizes of the stack frame (the maximum amount of stack used at any given time during the
    /// execution of the function) for each function.
    pub function_stack_sizes: Vec<u64>,
}

impl Module {
    pub fn new(module: &[u8], configuration: &impl AnalysisConfig) -> Result<Self, Error> {
        let mut types = vec![];
        let mut functions = vec![];
        let mut function_stack_sizes = vec![];
        let mut globals = vec![];
        let mut tables = vec![];
        // Reused between functions for speeds.
        let mut locals = PrefixSumVec::new();
        let mut operand_stack = vec![];
        let mut frame_stack = vec![];

        let parser = wasmparser::Parser::new(0);
        for payload in parser.parse_all(module) {
            let payload = payload.map_err(Error::ParsePayload)?;
            match payload {
                wasmparser::Payload::ImportSection(reader) => {
                    for import in reader.into_iter() {
                        let import = import.map_err(Error::ParseImports)?;
                        match import.ty {
                            wasmparser::TypeRef::Func(f) => {
                                functions.push(f);
                                function_stack_sizes.push(0);
                            }
                            wasmparser::TypeRef::Global(g) => {
                                globals.push(g.content_type);
                            }
                            wasmparser::TypeRef::Table(t) => {
                                tables.push(t.element_type);
                            }
                            wasmparser::TypeRef::Memory(_) => continue,
                            wasmparser::TypeRef::Tag(_) => continue,
                        }
                    }
                }
                wasmparser::Payload::TypeSection(reader) => {
                    for ty in reader {
                        let ty = ty.map_err(Error::ParseTypes)?;
                        types.push(ty);
                    }
                }
                wasmparser::Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global.map_err(Error::ParseGlobals)?;
                        globals.push(global.ty.content_type);
                    }
                }
                wasmparser::Payload::TableSection(reader) => {
                    for tbl in reader.into_iter() {
                        let tbl = tbl.map_err(Error::ParseTable)?;
                        tables.push(tbl.element_type);
                    }
                }
                wasmparser::Payload::FunctionSection(reader) => {
                    for function in reader {
                        let function = function.map_err(Error::ParseFunctions)?;
                        functions.push(function);
                    }
                }
                wasmparser::Payload::CodeSectionEntry(function) => {
                    locals.clear();
                    operand_stack.clear();
                    frame_stack.clear();

                    // We use the length of `function_stack_sizes` to _also_ act as a counter for
                    // how many code section entries we have seen so far. This allows us to match
                    // up the function information with its type and such.
                    let function_id_usize = function_stack_sizes.len();
                    let function_id =
                        u32::try_from(function_id_usize).map_err(|_| Error::TooManyFunctions)?;
                    let type_id = *functions
                        .get(function_id_usize)
                        .ok_or(Error::FunctionIndex(function_id))?;
                    let type_id_usize =
                        usize::try_from(type_id).map_err(|e| Error::TypeIndexRange(type_id, e))?;
                    let fn_type = types.get(type_id_usize).ok_or(Error::TypeIndex(type_id))?;

                    match fn_type {
                        wasmparser::Type::Func(fnty) => {
                            for param in fnty.params() {
                                locals
                                    .try_push_more(1, *param)
                                    .map_err(|e| Error::TooManyLocals(e, function_id))?;
                            }
                        }
                    }
                    for local in function.get_locals_reader().map_err(Error::LocalsReader)? {
                        let local = local.map_err(Error::ParseLocals)?;
                        locals
                            .try_push_more(local.0, local.1)
                            .map_err(|e| Error::TooManyLocals(e, function_id))?;
                    }

                    // This includes accounting for any possible return pointer tracking,
                    // parameters and locals (which all are considered locals in wasm).
                    let activation_size = configuration.size_of_function_activation(&locals);
                    let mut visitor = StackSizeVisitor {
                        offset: 0,

                        config: configuration,
                        functions: &functions,
                        types: &types,
                        globals: &globals,
                        tables: &tables,
                        locals: &locals,

                        operands: &mut operand_stack,
                        size: 0,
                        max_size: 0,

                        // Future optimization opportunity: Struct-of-Arrays representation.
                        frames: &mut frame_stack,
                        top_frame: Frame {
                            height: 0,
                            block_type: BlockType::Empty,
                            stack_polymorphic: false,
                        },
                    };

                    let mut operators = function
                        .get_operators_reader()
                        .map_err(Error::OperatorReader)?;
                    loop {
                        visitor.offset = operators.original_position();
                        let result = operators
                            .visit_with_offset(&mut visitor)
                            .map_err(Error::VisitOperators)??;
                        if let Some(stack_size) = result {
                            function_stack_sizes.push(activation_size + stack_size);
                            break;
                        }
                    }
                }
                _ => (),
            }
        }
        Ok(Self {
            function_stack_sizes,
        })
    }
}

pub trait AnalysisConfig {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8;
    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64;
}

#[derive(Debug)]
struct Frame {
    /// Operand stack height at the time this frame was entered.
    ///
    /// This way no matter how the operand stack is modified during the execution of this frame, we
    /// can always reset the operand stack back to this specific height when the frame terminates.
    height: usize,

    /// Type of the block representing this frame.
    ///
    /// The parameters are below height and get popped when the frame terminates. Results get
    /// pushed back onto the operand stack.
    block_type: BlockType,

    /// Are the remaining instructions within this frame reachable?
    ///
    /// That is, is the operand stack part of this frame polymorphic? See `stack-polymorphic` in
    /// the wasm-core specification for explanation.
    stack_polymorphic: bool,
}

struct StackSizeVisitor<'a, Cfg> {
    offset: usize,

    config: &'a Cfg,
    functions: &'a [u32],
    types: &'a [wasmparser::Type],
    globals: &'a [wasmparser::ValType],
    tables: &'a [wasmparser::ValType],
    locals: &'a PrefixSumVec<wasmparser::ValType, u32>,

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
    operands: &'a mut Vec<u8>,
    /// Sum of all values in the `operands` field above.
    size: u64,
    /// Maximum observed value for the `size` field above.
    max_size: u64,

    /// The stack of frames (as created by operations such as `block`).
    frames: &'a mut Vec<Frame>,
    /// The top-most frame.
    ///
    /// This aids quicker access at a cost of some special-casing for the very last `end` operator.
    top_frame: Frame,
}

type Output<'a, V> = <V as VisitOperator<'a>>::Output;

impl<'a, Cfg: AnalysisConfig> StackSizeVisitor<'a, Cfg> {
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
        let stack_polymorphic = self.top_frame.stack_polymorphic;
        let height = self
            .operands
            .len()
            .checked_sub(shift_operands)
            .ok_or(Error::EmptyStack(self.offset))?;
        self.frames.push(std::mem::replace(
            &mut self.top_frame,
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
            let frame = std::mem::replace(&mut self.top_frame, frame);
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
    ///
    /// This means that any operand stack push and pop operations will do nothing and uncontionally
    /// succeed, while this frame is still active.
    fn stack_polymorphic(&mut self) {
        self.top_frame.stack_polymorphic = true;
    }

    fn push(&mut self, t: ValType) {
        if !self.top_frame.stack_polymorphic {
            let value_size = self.config.size_of_value(t);
            self.operands.push(value_size);
            self.size += u64::from(value_size);
            self.max_size = std::cmp::max(self.size, self.max_size);
        }
    }

    fn pop(&mut self) -> Result<(), Error> {
        if !self.top_frame.stack_polymorphic {
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
        if count == 0 {
            Ok(())
        } else if !self.top_frame.stack_polymorphic {
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
        } else {
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
