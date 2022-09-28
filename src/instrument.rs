//! Instrumentation-based implementation of the finite-wasm specification.
//!
//! The functionality provided by this module will transform a provided WebAssembly module in a way
//! that measures gas fees and stack depth without any special support by the runtime executing the
//! code in question.

use std::num::TryFromIntError;

use wasmparser::{BinaryReaderError, BlockType, BrTable, Ieee32, Ieee64, MemArg, ValType, V128};

use crate::partial_sum::PartialSumMap;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("could not parse a part of the WASM payload")]
    ParsePayload(#[source] BinaryReaderError),
    #[error("could not create a function locals’ reader")]
    LocalsReader(#[source] BinaryReaderError),
    #[error("could not create a function operators’ reader")]
    OperatorReader(#[source] BinaryReaderError),
    #[error("could not visit the function operators")]
    VisitOperators(#[source] BinaryReaderError),
    #[error("could not parse the type section entry")]
    ParseTypes(#[source] BinaryReaderError),
    #[error("could not parse the function section entry")]
    ParseFunctions(#[source] BinaryReaderError),
    #[error("could not parse the global section entry")]
    ParseGlobals(#[source] BinaryReaderError),
    #[error("could not parsse function locals")]
    ParseLocals(#[source] BinaryReaderError),
    #[error("could not parse the imports")]
    ParseImports(#[source] BinaryReaderError),
    #[error("too many functions in the module")]
    TooManyFunctions,
    #[error("could not parse the table section")]
    ParseTable(#[source] BinaryReaderError),

    // These are invariant violation bugs (e.g. validation has not been run)
    #[error("could not process locals for function ${1}")]
    TooManyLocals(#[source] crate::partial_sum::Error, u32),
    #[error("module is malformed at offset {0}, if..else..end is not correctly constructed")]
    MalformedIfElse(usize),
    #[error("empty stack, module should not have passed validation")]
    EmptyStack,
    #[error("type index is out of range")]
    TypeIndexRange(#[source] TryFromIntError),
    #[error("type index refers to non-existent type")]
    TypeIndex,
    #[error("function index is out of range")]
    FunctionIndexRange(#[source] TryFromIntError),
    #[error("function index refers to non-existent type")]
    FunctionIndex,
    #[error("global index is out of range")]
    GlobalIndexRange(#[source] TryFromIntError),
    #[error("global index refers to non-existent type")]
    GlobalIndex,
    #[error("table index is out of range")]
    TableIndexRange(#[source] TryFromIntError),
    #[error("table index refers to non-existent type")]
    TableIndex,
    #[error("local type could not be looked up for local {0}")]
    LocalIndex(u32),
    #[error("missing top-most frame, webassembly input is malformed")]
    CurrentFrameMissing,
}

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
        let mut locals = PartialSumMap::new();

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
                    // We use the length of `function_stack_sizes` to _also_ act as a counter for
                    // how many code section entries we have seen so far. This allows us to match
                    // up the function information with its type and such.
                    let function_id = function_stack_sizes.len();
                    let type_id = functions
                        .get(function_id)
                        .ok_or(Error::FunctionIndex)
                        .unwrap();
                    let function_id =
                        u32::try_from(function_id).map_err(|_| Error::TooManyFunctions)?;
                    let type_id = usize::try_from(*type_id).map_err(Error::TypeIndexRange)?;
                    let fn_type = types.get(type_id).ok_or(Error::TypeIndex)?;

                    locals.clear();
                    match fn_type {
                        wasmparser::Type::Func(fnty) => {
                            for param in fnty.params() {
                                locals
                                    .push(1, *param)
                                    .map_err(|e| Error::TooManyLocals(e, function_id))?;
                            }
                        }
                    }
                    for local in function.get_locals_reader().map_err(Error::LocalsReader)? {
                        let local = local.map_err(Error::ParseLocals)?;
                        locals
                            .push(local.0, local.1)
                            .map_err(|e| Error::TooManyLocals(e, function_id))?;
                    }

                    // FIXME: this needs to estimate parameters too… right?
                    let activation_size = configuration.size_of_function_activation(&locals);

                    let mut visitor = StackSizeVisitor {
                        config: configuration,
                        functions: &functions,
                        types: &types,
                        globals: &globals,
                        tables: &tables,
                        locals: &locals,
                        frames: vec![],
                    };
                    visitor.visit_function_entry(function_id)?;
                    let mut operators = function
                        .get_operators_reader()
                        .map_err(Error::OperatorReader)?;
                    loop {
                        let result = operators
                            .visit_with_offset(&mut visitor)
                            .map_err(Error::VisitOperators)??;
                        if let Some(stack_size) = result {
                            break function_stack_sizes.push(activation_size + stack_size);
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
    fn size_of_value(&self, ty: wasmparser::ValType) -> u64;
    fn size_of_label(&self) -> u64;
    fn size_of_function_activation(&self, locals: &PartialSumMap<u32, ValType>) -> u64;
}

/// Sizes of the operands currently pushed to the operand stack within this frame.
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
#[derive(Clone, Debug)]
struct Operands {
    stack: Vec<u64>,
    /// Sum of all values in the `stack` field above.
    size: u64,
    /// Maximum observed value for size.
    max_size: u64,
}

impl Operands {
    fn new() -> Self {
        Self {
            stack: vec![],
            size: 0,
            max_size: 0,
        }
    }

    fn push(&mut self, value_size: u64) -> &mut Self {
        self.stack.push(value_size);
        self.size += value_size;
        self.max_size = std::cmp::max(self.size, self.max_size);
        self
    }

    fn pop(&mut self) -> Result<&mut Self, Error> {
        self.size -= self.stack.pop().ok_or(Error::EmptyStack).unwrap();
        Ok(self)
    }

    fn pop_multiple(&mut self, count: usize) -> Result<&mut Self, Error> {
        let split_point = self
            .stack
            .len()
            .checked_sub(count)
            .ok_or(Error::EmptyStack)
            .unwrap();
        let size: u64 = self.stack.drain(split_point..).sum();
        self.size -= size;
        Ok(self)
    }
}

/// A regular frame produced by instructions such as `block`, `if` and such. One of these
/// frames are also implicitly created on an entry to a function.
#[derive(Debug)]
struct BlockInfo {
    ty: BlockType,
    operands: Operands,
    /// This block has been “terminated” by an unconditonal control flow instruction
    /// (`unreachable`, `br`, `br_table`, `return`, etc.)
    complete: bool,
}

#[derive(Debug)]
enum Frame {
    /// A frame produced by the `block` instruction.
    ///
    /// One of these frames are also implicitly created on an entry to a function.
    Block(BlockInfo),
    /// A frame produced by the `if` instruction.
    If(BlockInfo),
    /// A frame produced by the `if` instruction, in the `else` branch.
    Else {
        if_max_depth: u64,
        else_block: BlockInfo,
    },
    /// A frame produced by instructions such as `loop`. These differ from a regular block in that
    /// branches target the beginning of the frame (i.e. looping), rather than exiting the frame.
    Loop(BlockInfo),
}

impl Frame {
    fn max_depth(&self) -> u64 {
        match self {
            Frame::Block(bfi) => bfi.operands.max_size,
            // This is a malformed wasm construct, but the behaviour remains kinda reasonable if we
            // just return `max_size` as-is.
            Frame::If(bfi) => bfi.operands.max_size,
            Frame::Else {
                if_max_depth,
                else_block,
            } => std::cmp::max(*if_max_depth, else_block.operands.max_size),
            Frame::Loop(bfi) => bfi.operands.max_size,
        }
    }

    fn block_info(&self) -> &BlockInfo {
        match self {
            Frame::Block(bfi) => bfi,
            Frame::If(bfi) => bfi,
            Frame::Else { else_block, .. } => else_block,
            Frame::Loop(bfi) => bfi,
        }
    }

    fn block_info_mut(&mut self) -> &mut BlockInfo {
        match self {
            Frame::Block(bfi) => bfi,
            Frame::If(bfi) => bfi,
            Frame::Else { else_block, .. } => else_block,
            Frame::Loop(bfi) => bfi,
        }
    }
}

struct StackSizeVisitor<'a, Cfg> {
    config: &'a Cfg,
    functions: &'a [u32],
    types: &'a [wasmparser::Type],
    globals: &'a [wasmparser::ValType],
    tables: &'a [wasmparser::ValType],
    locals: &'a PartialSumMap<u32, wasmparser::ValType>,

    frames: Vec<Frame>,
}

impl<'a, Cfg: AnalysisConfig> StackSizeVisitor<'a, Cfg> {
    /// Get the frame at the specified depth.
    ///
    /// The frame at 0 depth is the “current” frame, the frame 1 levele deep is the current frame’s
    /// parent, etc.
    fn frame(&mut self, depth: usize) -> Option<&mut Frame> {
        let index = self.frames.len().checked_sub(1)?.checked_sub(depth)?;
        self.frames.get_mut(index)
    }

    fn current_frame(&mut self) -> Result<&mut Frame, Error> {
        self.frames.last_mut().ok_or(Error::CurrentFrameMissing)
    }

    fn ty(&self, type_index: u32) -> Result<&wasmparser::Type, Error> {
        let type_index = usize::try_from(type_index).map_err(Error::TypeIndexRange)?;
        self.types.get(type_index).ok_or(Error::TypeIndex)
    }

    fn function_type_index(&self, function_index: u32) -> Result<u32, Error> {
        let function_index = usize::try_from(function_index).map_err(Error::FunctionIndexRange)?;
        self.functions
            .get(function_index)
            .copied()
            .ok_or(Error::FunctionIndex)
    }

    fn push(&mut self, val: ValType) -> Result<Option<u64>, Error> {
        let Self { config, frames, .. } = self;
        let bfi = frames
            .last_mut()
            .ok_or(Error::CurrentFrameMissing)?
            .block_info_mut();
        if !bfi.complete {
            let size = config.size_of_value(val);
            bfi.operands.push(size);
        }
        Ok(None)
    }

    fn pop(&mut self) -> Result<Option<u64>, Error> {
        let bfi = self.current_frame()?.block_info_mut();
        if bfi.complete {
            Ok(None)
        } else {
            bfi.operands.pop().map(|_| None)
        }
    }

    fn pop_multiple(&mut self, len: usize) -> Result<Option<u64>, Error> {
        let bfi = self.current_frame()?.block_info_mut();
        if bfi.complete {
            Ok(None)
        } else {
            bfi.operands.pop_multiple(len).map(|_| None)
        }
    }

    fn pop2(&mut self) -> Result<Option<u64>, Error> {
        self.pop_multiple(2).map(|_| None)
    }

    fn pop3(&mut self) -> Result<Option<u64>, Error> {
        self.pop_multiple(3).map(|_| None)
    }

    fn binop(&mut self, ty: ValType) -> Result<Option<u64>, Error> {
        self.pop2()?;
        self.push(ty)
    }

    fn testop(&mut self) -> Result<Option<u64>, Error> {
        self.pop()?;
        self.push(ValType::I32) // Boolean integer result
    }

    fn relop(&mut self) -> Result<Option<u64>, Error> {
        self.pop2()?;
        self.push(ValType::I32) // Boolean integer result
    }

    fn cvtop(&mut self, result: ValType) -> Result<Option<u64>, Error> {
        self.pop()?;
        self.push(result)
    }

    fn vrelop(&mut self) -> Result<Option<u64>, Error> {
        self.pop() // [v128 v128] -> [v128]
    }

    fn vcvtop(&mut self) -> Result<Option<u64>, Error> {
        /* [v128] -> [v128] */
        Ok(None)
    }

    fn extract_lane(&mut self, ty: ValType) -> Result<Option<u64>, Error> {
        self.pop()?;
        self.push(ty)
    }

    fn splat(&mut self) -> Result<Option<u64>, Error> {
        self.pop()?;
        self.push(ValType::V128)
    }

    fn bitmask(&mut self) -> Result<Option<u64>, Error> {
        self.pop()?;
        self.push(ValType::I32)
    }

    fn atomic_rmw(&mut self, ty: ValType) -> Result<Option<u64>, Error> {
        self.pop2()?;
        self.push(ty)
    }

    fn atomic_cmpxchg(&mut self, ty: ValType) -> Result<Option<u64>, Error> {
        self.pop3()?;
        self.push(ty)
    }

    fn atomic_load(&mut self, ty: ValType) -> Result<Option<u64>, Error> {
        self.pop()?;
        self.push(ty)
    }

    fn call_typed_function(&mut self, type_index: u32) -> Result<Option<u64>, Error> {
        let type_index = usize::try_from(type_index).map_err(Error::TypeIndexRange)?;
        match self.types.get(type_index).ok_or(Error::TypeIndex)? {
            wasmparser::Type::Func(fnty) => {
                for _ in fnty.params() {
                    self.pop()?;
                }
                for result_ty in fnty.results() {
                    self.push(*result_ty)?;
                }
            }
        }
        Ok(None)
    }

    fn block_operands_from_ty(&self, ty: &BlockType) -> Result<Operands, Error> {
        match ty {
            // No input parameters, only a return.
            BlockType::Empty | BlockType::Type(_) => Ok(Operands::new()),
            BlockType::FuncType(fn_type_idx) => {
                // First, pop the appropriate number of operands from the current frame (inputs to
                // the block)
                match self.ty(*fn_type_idx)? {
                    wasmparser::Type::Func(fnty) => {
                        let stack: Vec<u64> = fnty
                            .params()
                            .into_iter()
                            .map(|vty| self.config.size_of_value(*vty))
                            .collect();
                        let size = stack.iter().sum();
                        Ok(Operands {
                            size,
                            stack,
                            max_size: size,
                        })
                    }
                }
            }
        }
    }

    fn visit_function_entry(&mut self, function_index: u32) -> Result<Option<u64>, Error> {
        let type_index = self.function_type_index(function_index)?;
        let ty = BlockType::FuncType(type_index);
        self.frames.push(Frame::Block(BlockInfo {
            ty,
            operands: Operands::new(),
            complete: false,
        }));
        Ok(None)
    }
}

impl<'a, 'cfg, Cfg: AnalysisConfig> wasmparser::VisitOperator<'a> for StackSizeVisitor<'cfg, Cfg> {
    type Output = Result<Option<u64>, Error>;

    // Special cases (parametrics)
    fn visit_nop(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_drop(&mut self, _: usize) -> Self::Output {
        self.pop()
    }
    fn visit_select(&mut self, _: usize) -> Self::Output {
        self.pop2()
    } // [t t i32] -> [t]
    fn visit_typed_select(&mut self, _: usize, _: ValType) -> Self::Output {
        self.pop2()
    }

    // t.const
    fn visit_i32_const(&mut self, _: usize, _: i32) -> Self::Output {
        self.push(ValType::I32)
    }
    fn visit_i64_const(&mut self, _: usize, _: i64) -> Self::Output {
        self.push(ValType::I64)
    }
    fn visit_f32_const(&mut self, _: usize, _: Ieee32) -> Self::Output {
        self.push(ValType::F32)
    }
    fn visit_f64_const(&mut self, _: usize, _: Ieee64) -> Self::Output {
        self.push(ValType::F64)
    }
    fn visit_v128_const(&mut self, _: usize, _: V128) -> Self::Output {
        self.push(ValType::V128)
    }

    // locals & globals
    fn visit_local_get(&mut self, _: usize, local_index: u32) -> Self::Output {
        self.push(
            *self
                .locals
                .find(local_index)
                .ok_or(Error::LocalIndex(local_index))?,
        )
    }
    fn visit_local_set(&mut self, _: usize, _: u32) -> Self::Output {
        self.pop()
    }
    fn visit_local_tee(&mut self, _: usize, _: u32) -> Self::Output {
        Ok(None)
    }

    fn visit_global_get(&mut self, _: usize, global_index: u32) -> Self::Output {
        let global_index = usize::try_from(global_index).map_err(Error::GlobalIndexRange)?;
        let global_ty = self.globals.get(global_index).ok_or(Error::GlobalIndex)?;
        self.push(*global_ty)
    }
    fn visit_global_set(&mut self, _: usize, _: u32) -> Self::Output {
        self.pop()
    }

    //  t.iunop | t.funop | t.vunop
    //  consume one operand and return an operand of the same type.
    fn visit_i32_clz(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64_clz(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_i32_ctz(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64_ctz(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_i32_popcnt(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64_popcnt(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i8x16_popcnt(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_f32_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f32x4_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64x2_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i16x8_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i32x4_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64x2_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i8x16_abs(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_f32_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f32x4_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64x2_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i8x16_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i16x8_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i32x4_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64x2_neg(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_v128_not(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_f32_sqrt(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64_sqrt(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f32x4_sqrt(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64x2_sqrt(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_f32_ceil(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64_ceil(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f32x4_ceil(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64x2_ceil(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_f32_floor(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64_floor(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f32x4_floor(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64x2_floor(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_f32_trunc(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64_trunc(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f32x4_trunc(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64x2_trunc(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_f32_nearest(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64_nearest(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f32x4_nearest(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_f64x2_nearest(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    fn visit_i32_extend8_s(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i32_extend16_s(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64_extend8_s(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64_extend16_s(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i64_extend32_s(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    // binop
    fn visit_i32_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }
    fn visit_f32_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F32)
    }
    fn visit_f64_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F64)
    }

    fn visit_i32_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }
    fn visit_f32_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F32)
    }
    fn visit_f64_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F64)
    }

    fn visit_i32_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }
    fn visit_f32_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F32)
    }
    fn visit_f64_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F64)
    }

    fn visit_i32_div_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_div_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_div_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_div_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_rem_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_rem_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_rem_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_rem_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_and(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_and(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_or(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_or(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_xor(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_xor(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_shl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_shl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_shr_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_shr_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_shr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_shr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_rotl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_rotl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_i32_rotr(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I32)
    }
    fn visit_i64_rotr(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::I64)
    }

    fn visit_f32_div(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F32)
    }
    fn visit_f64_div(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F64)
    }

    fn visit_f32_min(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F32)
    }
    fn visit_f64_min(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F64)
    }

    fn visit_f32_max(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F32)
    }
    fn visit_f64_max(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F64)
    }

    fn visit_f32_copysign(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F32)
    }
    fn visit_f64_copysign(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::F64)
    }

    // itestop
    fn visit_i32_eqz(&mut self, _: usize) -> Self::Output {
        self.testop()
    }
    fn visit_i64_eqz(&mut self, _: usize) -> Self::Output {
        self.testop()
    }
    fn visit_v128_any_true(&mut self, _: usize) -> Self::Output {
        self.testop()
    }
    fn visit_i8x16_all_true(&mut self, _: usize) -> Self::Output {
        self.testop()
    }
    fn visit_i16x8_all_true(&mut self, _: usize) -> Self::Output {
        self.testop()
    }
    fn visit_i32x4_all_true(&mut self, _: usize) -> Self::Output {
        self.testop()
    }
    fn visit_i64x2_all_true(&mut self, _: usize) -> Self::Output {
        self.testop()
    }

    // relop
    fn visit_i32_eq(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_eq(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f32_eq(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f64_eq(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_ne(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_ne(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f32_ne(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f64_ne(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_lt_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_lt_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_lt_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_lt_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_f32_lt(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f64_lt(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_gt_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_gt_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_gt_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_gt_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_f32_gt(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f64_gt(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_le_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_le_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_le_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_le_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_f32_le(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f64_le(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_ge_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_ge_s(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_i32_ge_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_i64_ge_u(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    fn visit_f32_ge(&mut self, _: usize) -> Self::Output {
        self.relop()
    }
    fn visit_f64_ge(&mut self, _: usize) -> Self::Output {
        self.relop()
    }

    // cvtop
    fn visit_i32_wrap_i64(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_f32s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_f32u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_f64s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_f64u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_reinterpret_f32(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_sat_f32_s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_sat_f32_u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_sat_f64_s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }
    fn visit_i32_trunc_sat_f64_u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I32)
    }

    fn visit_i64_extend_i32s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_extend_i32u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_f32s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_f32u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_f64s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_f64u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_reinterpret_f64(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_sat_f32_s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_sat_f32_u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_sat_f64_s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }
    fn visit_i64_trunc_sat_f64_u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::I64)
    }

    fn visit_f32_convert_i32s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F32)
    }
    fn visit_f32_convert_i32u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F32)
    }
    fn visit_f32_convert_i64s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F32)
    }
    fn visit_f32_convert_i64u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F32)
    }
    fn visit_f32_demote_f64(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F32)
    }
    fn visit_f32_reinterpret_i32(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F32)
    }

    fn visit_f64_convert_i32_s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F64)
    }
    fn visit_f64_convert_i32_u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F64)
    }
    fn visit_f64_convert_i64_s(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F64)
    }
    fn visit_f64_convert_i64_u(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F64)
    }
    fn visit_f64_promote_f32(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F64)
    }
    fn visit_f64_reinterpret_i64(&mut self, _: usize) -> Self::Output {
        self.cvtop(ValType::F64)
    }

    // vbinary_op
    fn visit_v128_and(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_v128_andnot(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_v128_or(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_v128_xor(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i8x16_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_add(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i8x16_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_sub(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_mul(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_div(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_div(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_min(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_min(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_max(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_max(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_pmin(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_pmin(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_pmax(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_pmax(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_add_sat_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i8x16_add_sat_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_add_sat_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_add_sat_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_sub_sat_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i8x16_sub_sat_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_sub_sat_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_sub_sat_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_min_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i8x16_min_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_min_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_min_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_min_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_min_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_max_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i8x16_max_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_max_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_max_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_max_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_max_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_relaxed_min(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f32x4_relaxed_max(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_relaxed_min(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_f64x2_relaxed_max(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_shl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_shl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_shl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_shl(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_shr_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i8x16_shr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_shr_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_shr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_shr_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_shr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_shr_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_shr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_f32x4_fma(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }
    fn visit_f32x4_fms(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }
    fn visit_f64x2_fma(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }
    fn visit_f64x2_fms(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }

    fn visit_i32x4_dot_i16x8_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i16x8_extmul_low_i8x16_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_extmul_high_i8x16_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_extmul_low_i8x16_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_extmul_high_i8x16_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_extmul_low_i16x8_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_extmul_high_i16x8_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_extmul_low_i16x8_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i32x4_extmul_high_i16x8_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_extmul_low_i32x4_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_extmul_high_i32x4_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_extmul_low_i32x4_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i64x2_extmul_high_i32x4_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_avgr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_avgr_u(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }
    fn visit_i16x8_q15mulr_sat_s(&mut self, _: usize) -> Self::Output {
        self.binop(ValType::V128)
    }

    fn visit_i8x16_laneselect(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }
    fn visit_i16x8_laneselect(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }
    fn visit_i32x4_laneselect(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }
    fn visit_i64x2_laneselect(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }

    fn visit_i8x16_eq(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_ne(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_lt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_lt_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_gt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_gt_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_le_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_le_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_ge_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i8x16_ge_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_eq(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_ne(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_lt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_lt_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_gt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_gt_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_le_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_le_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_ge_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i16x8_ge_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_eq(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_ne(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_lt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_lt_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_gt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_gt_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_le_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_le_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_ge_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i32x4_ge_u(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i64x2_eq(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i64x2_ne(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i64x2_lt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i64x2_gt_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i64x2_le_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_i64x2_ge_s(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f32x4_eq(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f32x4_ne(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f32x4_lt(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f32x4_gt(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f32x4_le(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f32x4_ge(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f64x2_eq(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f64x2_ne(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f64x2_lt(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f64x2_gt(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f64x2_le(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }
    fn visit_f64x2_ge(&mut self, _: usize) -> Self::Output {
        self.vrelop()
    }

    fn visit_i32x4_extend_low_i16x8_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_extend_high_i16x8_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_extend_low_i16x8_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_extend_high_i16x8_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i64x2_extend_low_i32x4_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i64x2_extend_high_i32x4_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i64x2_extend_low_i32x4_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i64x2_extend_high_i32x4_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i16x8_extend_low_i8x16_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i16x8_extend_high_i8x16_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i16x8_extend_low_i8x16_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i16x8_extend_high_i8x16_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_trunc_sat_f32x4_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_trunc_sat_f32x4_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_f32x4_convert_i32x4_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_f32x4_convert_i32x4_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_trunc_sat_f64x2_s_zero(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_trunc_sat_f64x2_u_zero(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_f64x2_convert_low_i32x4_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_f64x2_convert_low_i32x4_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_f32x4_demote_f64x2_zero(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_f64x2_promote_low_f32x4(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_relaxed_trunc_sat_f32x4_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_relaxed_trunc_sat_f32x4_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_relaxed_trunc_sat_f64x2_s_zero(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i32x4_relaxed_trunc_sat_f64x2_u_zero(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i8x16_narrow_i16x8_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i8x16_narrow_i16x8_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i16x8_narrow_i32x4_s(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }
    fn visit_i16x8_narrow_i32x4_u(&mut self, _: usize) -> Self::Output {
        self.vcvtop()
    }

    fn visit_i8x16_shuffle(&mut self, _: usize, _: [u8; 16]) -> Self::Output {
        self.pop()
    }

    fn visit_i8x16_extract_lane_s(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::I32)
    }
    fn visit_i8x16_extract_lane_u(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::I32)
    }
    fn visit_i16x8_extract_lane_s(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::I32)
    }
    fn visit_i16x8_extract_lane_u(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::I32)
    }
    fn visit_i32x4_extract_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::I32)
    }
    fn visit_i64x2_extract_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::I64)
    }
    fn visit_f32x4_extract_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::F32)
    }
    fn visit_f64x2_extract_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.extract_lane(ValType::F64)
    }

    fn visit_i8x16_replace_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_i16x8_replace_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_i32x4_replace_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_i64x2_replace_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_f32x4_replace_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_f64x2_replace_lane(&mut self, _: usize, _: u8) -> Self::Output {
        self.pop()
    }

    fn visit_i8x16_swizzle(&mut self, _: usize) -> Self::Output {
        self.pop()
    }
    fn visit_i8x16_relaxed_swizzle(&mut self, _: usize) -> Self::Output {
        self.pop()
    }

    fn visit_i8x16_splat(&mut self, _: usize) -> Self::Output {
        self.splat()
    }
    fn visit_i16x8_splat(&mut self, _: usize) -> Self::Output {
        self.splat()
    }
    fn visit_i32x4_splat(&mut self, _: usize) -> Self::Output {
        self.splat()
    }
    fn visit_i64x2_splat(&mut self, _: usize) -> Self::Output {
        self.splat()
    }
    fn visit_f32x4_splat(&mut self, _: usize) -> Self::Output {
        self.splat()
    }
    fn visit_f64x2_splat(&mut self, _: usize) -> Self::Output {
        self.splat()
    }

    fn visit_v128_bitselect(&mut self, _: usize) -> Self::Output {
        self.pop2()
    }

    fn visit_i8x16_bitmask(&mut self, _: usize) -> Self::Output {
        self.bitmask()
    }
    fn visit_i16x8_bitmask(&mut self, _: usize) -> Self::Output {
        self.bitmask()
    }
    fn visit_i32x4_bitmask(&mut self, _: usize) -> Self::Output {
        self.bitmask()
    }
    fn visit_i64x2_bitmask(&mut self, _: usize) -> Self::Output {
        self.bitmask()
    }

    fn visit_i16x8_extadd_pairwise_i8x16_s(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i16x8_extadd_pairwise_i8x16_u(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i32x4_extadd_pairwise_i16x8_s(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }
    fn visit_i32x4_extadd_pairwise_i16x8_u(&mut self, _: usize) -> Self::Output {
        Ok(None)
    }

    // memory
    // table
    fn visit_table_init(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        self.pop3()
    }
    fn visit_elem_drop(&mut self, _: usize, _: u32) -> Self::Output {
        Ok(None)
    }
    fn visit_table_copy(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        self.pop3()
    }
    fn visit_table_fill(&mut self, _: usize, _: u32) -> Self::Output {
        self.pop3()
    }
    fn visit_table_get(&mut self, _: usize, table: u32) -> Self::Output {
        let table = usize::try_from(table).map_err(Error::TableIndexRange)?;
        self.pop()?;
        self.push(*self.tables.get(table).ok_or(Error::TableIndex)?)
    }
    fn visit_table_set(&mut self, _: usize, _: u32) -> Self::Output {
        self.pop2()
    }
    fn visit_table_grow(&mut self, _: usize, _: u32) -> Self::Output {
        self.pop2()?;
        self.push(ValType::I32)
    }
    fn visit_table_size(&mut self, _: usize, _: u32) -> Self::Output {
        self.push(ValType::I32)
    }

    // references
    fn visit_ref_null(&mut self, _: usize, ty: ValType) -> Self::Output {
        self.push(ty)
    }
    fn visit_ref_func(&mut self, _: usize, _: u32) -> Self::Output {
        self.push(ValType::FuncRef)
    }
    fn visit_ref_is_null(&mut self, _: usize) -> Self::Output {
        self.pop()?;
        self.push(ValType::I32)
    }

    // memory
    fn visit_i32_load(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I32)
    }
    fn visit_i32_load8_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I32)
    }
    fn visit_i32_load8_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I32)
    }
    fn visit_i32_load16_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I32)
    }
    fn visit_i32_load16_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I32)
    }

    fn visit_i32_store(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_i32_store8(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_i32_store16(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }

    fn visit_i64_load(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I64)
    }
    fn visit_i64_load8_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I64)
    }
    fn visit_i64_load8_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I64)
    }
    fn visit_i64_load16_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I64)
    }
    fn visit_i64_load16_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I64)
    }
    fn visit_i64_load32_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I64)
    }
    fn visit_i64_load32_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::I64)
    }

    fn visit_i64_store(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_i64_store8(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_i64_store16(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_i64_store32(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }

    fn visit_f32_load(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::F32)
    }
    fn visit_f64_load(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::F64)
    }

    fn visit_f32_store(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_f64_store(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }

    fn visit_v128_load(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load8x8_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load8x8_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load16x4_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load16x4_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load32x2_s(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load32x2_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load8_splat(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load16_splat(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load32_splat(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load64_splat(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load32_zero(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load64_zero(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load8_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load16_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load32_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.push(ValType::V128)
    }
    fn visit_v128_load64_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.push(ValType::V128)
    }

    fn visit_v128_store(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_v128_store8_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_v128_store16_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_v128_store32_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.pop()
    }
    fn visit_v128_store64_lane(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
        self.pop()
    }

    fn visit_memory_size(&mut self, _: usize, _: u32, _: u8) -> Self::Output {
        self.push(ValType::I32)
    }
    fn visit_memory_grow(&mut self, _: usize, _: u32, _: u8) -> Self::Output {
        Ok(None)
    }
    fn visit_memory_init(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        self.pop3()
    }
    fn visit_data_drop(&mut self, _: usize, _: u32) -> Self::Output {
        Ok(None)
    }
    fn visit_memory_copy(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        self.pop3()
    }
    fn visit_memory_fill(&mut self, _: usize, _: u32) -> Self::Output {
        self.pop3()
    }

    // atomic
    fn visit_i32_atomic_load(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_load(ValType::I32)
    }
    fn visit_i32_atomic_load8_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_load(ValType::I32)
    }
    fn visit_i32_atomic_load16_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_load(ValType::I32)
    }
    fn visit_i64_atomic_load(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_load(ValType::I64)
    }
    fn visit_i64_atomic_load8_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_load(ValType::I64)
    }
    fn visit_i64_atomic_load16_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_load(ValType::I64)
    }
    fn visit_i64_atomic_load32_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_load(ValType::I64)
    }

    fn visit_i32_atomic_store(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_i32_atomic_store8(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_i32_atomic_store16(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_i64_atomic_store(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_i64_atomic_store8(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_i64_atomic_store16(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_i64_atomic_store32(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }

    fn visit_i32_atomic_rmw_cmpxchg(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_cmpxchg(ValType::I32)
    }
    fn visit_i32_atomic_rmw8_cmpxchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_cmpxchg(ValType::I32)
    }
    fn visit_i32_atomic_rmw16_cmpxchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_cmpxchg(ValType::I32)
    }
    fn visit_i64_atomic_rmw_cmpxchg(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_cmpxchg(ValType::I64)
    }
    fn visit_i64_atomic_rmw8_cmpxchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_cmpxchg(ValType::I64)
    }
    fn visit_i64_atomic_rmw16_cmpxchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_cmpxchg(ValType::I64)
    }
    fn visit_i64_atomic_rmw32_cmpxchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_cmpxchg(ValType::I64)
    }

    fn visit_i32_atomic_rmw_add(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw8_add_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw16_add_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw_sub(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw8_sub_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw16_sub_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw_and(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw8_and_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw16_and_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw_or(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw8_or_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw16_or_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw_xor(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw8_xor_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw16_xor_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw_xchg(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw8_xchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i32_atomic_rmw16_xchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I32)
    }
    fn visit_i64_atomic_rmw_add(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw8_add_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw16_add_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw32_add_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw_sub(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw8_sub_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw16_sub_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw32_sub_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw_and(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw8_and_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw16_and_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw32_and_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw_or(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw8_or_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw16_or_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw32_or_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw_xor(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw8_xor_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw16_xor_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw32_xor_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw_xchg(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw8_xchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw16_xchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }
    fn visit_i64_atomic_rmw32_xchg_u(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.atomic_rmw(ValType::I64)
    }

    fn visit_memory_atomic_notify(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop()
    }
    fn visit_memory_atomic_wait32(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_memory_atomic_wait64(&mut self, _: usize, _: MemArg) -> Self::Output {
        self.pop2()
    }
    fn visit_atomic_fence(&mut self, _: usize, _: u8) -> Self::Output {
        todo!("missing in spec?")
    }

    // Control-flow instructions
    fn visit_return(&mut self, offset: usize) -> Self::Output {
        // This behaves as-if a `br` to the outer-most block.
        self.visit_br(offset, self.frames.len() as u32)
    }
    fn visit_return_call(&mut self, offset: usize, _: u32) -> Self::Output {
        // `return_call` behaves as-if a regular `return` followed by the `call`. For the purposes
        // of modelling the frame size of the _current_ function, only the `return` portion of this
        // computation is relevant.
        self.visit_return(offset)
    }
    fn visit_return_call_indirect(&mut self, offset: usize, _: u32, _: u32) -> Self::Output {
        self.visit_return(offset)
    }
    fn visit_unreachable(&mut self, offset: usize) -> Self::Output {
        // TODO: Does this really behave as-if a return for the purposes of this analysis?
        self.visit_return(offset)
    }

    fn visit_call(&mut self, _: usize, function_index: u32) -> Self::Output {
        self.call_typed_function(self.function_type_index(function_index)?)
    }

    fn visit_call_indirect(&mut self, _: usize, type_index: u32, _: u32, _: u8) -> Self::Output {
        self.call_typed_function(type_index)
    }

    fn visit_block(&mut self, _: usize, ty: BlockType) -> Self::Output {
        let operands = self.block_operands_from_ty(&ty)?;
        self.pop_multiple(operands.stack.len())?;
        let complete = self.current_frame()?.block_info().complete;
        self.frames.push(Frame::Block(BlockInfo {
            ty,
            operands,
            complete,
        }));
        Ok(None)
    }

    fn visit_loop(&mut self, _: usize, ty: BlockType) -> Self::Output {
        let operands = self.block_operands_from_ty(&ty)?;
        self.pop_multiple(operands.stack.len())?;
        let complete = self.current_frame()?.block_info().complete;
        self.frames.push(Frame::Loop(BlockInfo {
            ty,
            operands,
            complete,
        }));
        Ok(None)
    }

    fn visit_if(&mut self, _: usize, ty: BlockType) -> Self::Output {
        let operands = self.block_operands_from_ty(&ty)?;
        // Block parameters and the condition.
        self.pop_multiple(operands.stack.len() + 1)?;
        let complete = self.current_frame()?.block_info().complete;
        self.frames.push(Frame::If(BlockInfo {
            ty,
            operands,
            complete,
        }));
        Ok(None)
    }

    fn visit_else(&mut self, offset: usize) -> Self::Output {
        let current_frame = self.frames.pop().expect("invalid use of analysis");
        match current_frame {
            Frame::If(bfi) => {
                let complete = self.current_frame()?.block_info().complete;
                self.frames.push(Frame::Else {
                    if_max_depth: bfi.operands.max_size,
                    else_block: BlockInfo {
                        ty: bfi.ty,
                        operands: self.block_operands_from_ty(&bfi.ty)?,
                        complete,
                    },
                });
            }
            // TODO: actually reachable in case of a malformed webassembly.
            Frame::Else { .. } | Frame::Loop(..) | Frame::Block(..) => {
                return Err(Error::MalformedIfElse(offset))
            }
        }
        Ok(None)
    }

    fn visit_end(&mut self, _: usize) -> Self::Output {
        let current_frame = self.frames.pop().expect("invalid use of analysis");
        let current_bfi = current_frame.block_info();
        let parent_frame = match self.frame(0) {
            // This is the end of the function and there are no more frames remaining to propagate
            // our computations into.
            None => return Ok(Some(current_bfi.operands.max_size)),
            Some(parent_frame) => parent_frame,
        };
        let parent_bfi = parent_frame.block_info_mut();
        if !parent_bfi.complete {
            let max_height = current_frame.max_depth();
            parent_bfi.operands.max_size = std::cmp::max(
                parent_bfi.operands.size + max_height,
                parent_bfi.operands.max_size,
            );
        }
        match current_bfi.ty {
            BlockType::Empty => Ok(None),
            BlockType::Type(ret_ty) => self.push(ret_ty),
            BlockType::FuncType(fn_type_idx) => {
                let result_types = match self.ty(fn_type_idx)? {
                    wasmparser::Type::Func(fnty) => fnty.results().to_vec(),
                };
                for result in result_types {
                    self.push(result)?;
                }
                Ok(None)
            }
        }
    }

    fn visit_br(&mut self, _: usize, _: u32) -> Self::Output {
        self.current_frame()?.block_info_mut().complete = true;
        Ok(None)
    }

    fn visit_br_table(&mut self, _: usize, _: BrTable<'a>) -> Self::Output {
        // br_table is an unconditional branch to one of the specified branch targets.
        //
        // This is somewhat more complicated than `br_if` but overall the similar line of thinking
        // as for a regular `br` applies. In particular we don’t need to worry about evaluating
        // rest of the operations within the frame.
        self.current_frame()?.block_info_mut().complete = true;
        Ok(None)
    }

    fn visit_br_if(&mut self, _: usize, _: u32) -> Self::Output {
        // There are two things that could happen here.
        //
        // First is when the condition operand is true. This instruction executed as-if it was a
        // plain `br` in this place. This won’t result in the stack size of this frame increasing
        // again. The continuation of the destination label `L` will have an arity of `n`. As part
        // of executing `br`, `n` operands are popped from the operand stack, Then a number of
        // labels/frames are popped from the stack, along with the values contained therein.
        // Finally `n` operands are pushed back onto the operand stack as the “return value” of the
        // block. As thus, executing a `(br_if (i32.const 1))` will _always_ result in a smaller
        // operand stack, and so it is uninteresting to explore this branch in isolation.
        //
        // Second is if the condition was actually false and the rest of this block is executed,
        // which can potentially later increase the size of this current frame. We’re largely
        // interested in this second case, so we don’t really need to do anything much more than…
        self.pop()
        // …the condition.
    }

    fn visit_delegate(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exceptions")
    }

    fn visit_try(&mut self, _: usize, _: BlockType) -> Self::Output {
        todo!("exceptions")
    }

    fn visit_catch(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exceptions")
    }

    fn visit_throw(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exceptions")
    }

    fn visit_rethrow(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exceptions")
    }

    fn visit_catch_all(&mut self, _: usize) -> Self::Output {
        todo!("exceptions")
    }
}
