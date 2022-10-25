//! Instrumentation-based implementation of the finite-wasm specification.
//!
//! The functionality provided by this module will transform a provided WebAssembly module in a way
//! that measures gas fees and stack depth without any special support by the runtime executing the
//! code in question.

use prefix_sum_vec::PrefixSumVec;
use std::num::TryFromIntError;
use wasmparser::{BinaryReaderError, BlockType, BrTable, Ieee32, Ieee64, MemArg, ValType, V128};

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

    // These codes are a result of a malformed input (e.g. validation has not been run)
    #[error("could not process locals for function ${1}")]
    TooManyLocals(#[source] prefix_sum_vec::TryPushError, u32),
    #[error("module is malformed at offset {0}: {}")]
    MalformedModule(usize, &'static str),
    #[error("empty stack at offset {0}")]
    EmptyStack(usize),
    #[error("type index {0} is out of range")]
    TypeIndexRange(u32, #[source] TryFromIntError),
    #[error("type index {0} refers to a non-existent type")]
    TypeIndex(u32),
    #[error("function index {0} is out of range")]
    FunctionIndexRange(u32, #[source] TryFromIntError),
    #[error("function index {0} refers to a non-existent type")]
    FunctionIndex(u32),
    #[error("global index {0} is out of range")]
    GlobalIndexRange(u32, #[source] TryFromIntError),
    #[error("global index {0} refers to a non-existent global")]
    GlobalIndex(u32),
    #[error("table index {0} is out of range")]
    TableIndexRange(u32, #[source] TryFromIntError),
    #[error("table index {0} refers to a non-existent table")]
    TableIndex(u32),
    #[error("type of local {0} could cannot be found")]
    LocalIndex(u32),
    #[error("too many frames in the function")]
    TooManyFrames,
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
        let mut locals = PrefixSumVec::new();
        // Reused between functions for speeds.
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

                    locals.clear();
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
                        config: configuration,
                        functions: &functions,
                        types: &types,
                        globals: &globals,
                        tables: &tables,
                        locals: &locals,

                        operands: std::mem::replace(&mut operand_stack, vec![]),
                        size: 0,
                        max_size: 0,

                        frames: std::mem::replace(&mut frame_stack, vec![]),
                        top_frame: Frame {
                            height: 0,
                            block_type: BlockType::FuncType(type_id),
                            stack_polymorphic: false,
                        },
                    };
                    visitor.push_block_params(visitor.top_frame.block_type)?;

                    let mut operators = function
                        .get_operators_reader()
                        .map_err(Error::OperatorReader)?;
                    loop {
                        let result = operators
                            .visit_with_offset(&mut visitor)
                            .map_err(Error::VisitOperators)??;
                        if let Some(stack_size) = result {
                            function_stack_sizes.push(activation_size + stack_size);
                            visitor.operands.clear();
                            visitor.frames.clear();
                            operand_stack = visitor.operands;
                            frame_stack = visitor.frames;
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
    fn size_of_value(&self, ty: wasmparser::ValType) -> u64;
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
    operands: Vec<u64>,
    /// Sum of all values in the `operands` field above.
    size: u64,
    /// Maximum observed value for the `size` field above.
    max_size: u64,

    /// The stack of frames (as created by operations such as `block`).
    frames: Vec<Frame>,
    /// The top-most frame.
    ///
    /// This aids quicker access at a cost of some special-casing for the very last `end` operator.
    top_frame: Frame,
}

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

    fn pop_block_params(&mut self, offset: usize, block_type: BlockType) -> Result<(), Error> {
        match block_type {
            BlockType::Empty => (),
            BlockType::Type(_) => (),
            BlockType::FuncType(type_index) => {
                let (params, _) = self.type_params_results(type_index)?;
                self.pop_many(offset, params.len())?;
            }
        }
        Ok(())
    }

    fn push_block_params(&mut self, block_type: BlockType) -> Result<(), Error> {
        match block_type {
            BlockType::Empty => (),
            BlockType::Type(_) => (),
            BlockType::FuncType(type_index) => {
                let (params, _) = self.type_params_results(type_index)?;
                for param in params {
                    self.push(*param);
                }
            }
        }
        Ok(())
    }

    fn push_block_results(&mut self, block_type: BlockType) -> Result<(), Error> {
        match block_type {
            BlockType::Empty => (),
            BlockType::Type(result) => {
                self.push(result);
            }
            BlockType::FuncType(type_index) => {
                let (_, results) = self.type_params_results(type_index)?;
                for result in results {
                    self.push(*result);
                }
            }
        }
        Ok(())
    }

    fn new_frame(&mut self, block_type: BlockType) -> Result<(), Error> {
        let stack_polymorphic = self.top_frame.stack_polymorphic;
        let height = self.operands.len();
        self.frames.push(std::mem::replace(
            &mut self.top_frame,
            Frame {
                height,
                block_type,
                stack_polymorphic,
            },
        ));
        self.push_block_params(block_type)
    }

    fn stack_polymorphic(&mut self) {
        self.top_frame.stack_polymorphic = true;
    }

    fn push(&mut self, t: ValType) {
        if !self.top_frame.stack_polymorphic {
            let value_size = self.config.size_of_value(t);
            self.operands.push(value_size);
            self.size += value_size;
            self.max_size = std::cmp::max(self.size, self.max_size);
        }
    }

    fn pop(&mut self, offset: usize) -> Result<(), Error> {
        if !self.top_frame.stack_polymorphic {
            self.size = self
                .size
                .checked_sub(self.operands.pop().ok_or(Error::EmptyStack(offset))?)
                .expect("stack size is going negative");
        }
        Ok(())
    }

    fn pop_many(&mut self, offset: usize, count: usize) -> Result<(), Error> {
        if count == 0 {
            Ok(())
        } else if !self.top_frame.stack_polymorphic {
            // TODO: would love a unit test for this.
            let operand_count = self.operands.len();
            let split_point = operand_count
                .checked_sub(count)
                .ok_or(Error::EmptyStack(offset))?;
            let size: u64 = self.operands.drain(split_point..).sum();
            self.size = self
                .size
                .checked_sub(size)
                .expect("stack size is going negative");
            Ok(())
        } else {
            Ok(())
        }
    }

    fn visit_const(&mut self, t: ValType) -> Result<Option<u64>, Error> {
        // [] → [t]
        self.push(t);
        Ok(None)
    }

    fn visit_unop(&mut self) -> Result<Option<u64>, Error> {
        // [t] -> [t]

        // Function body intentionally left empty (pops and immediately pushes the same type back)
        Ok(None)
    }

    fn visit_binop(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [t t] -> [t]
        self.pop(offset)?;
        Ok(None)
    }

    fn visit_testop(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [t] -> [i32]
        self.pop(offset)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_relop(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [t t] -> [i32]
        self.pop_many(offset, 2)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_cvtop(&mut self, offset: usize, result_ty: ValType) -> Result<Option<u64>, Error> {
        // t2.cvtop_t1_sx? : [t1] -> [t2]
        self.pop(offset)?;
        self.push(result_ty);
        Ok(None)
    }

    fn visit_vternop(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [v128 v128 v128] → [v128]
        self.pop_many(offset, 2)?;
        Ok(None)
    }

    fn visit_vrelop(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [v128 v128] -> [v128]
        self.visit_binop(offset)
    }

    fn visit_vishiftop(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [v128 i32] -> [v128]
        self.pop_many(offset, 2)?;
        self.push(ValType::V128);
        Ok(None)
    }

    fn visit_vinarrowop(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [v128 v128] -> [v128]
        self.pop(offset)?;
        Ok(None)
    }

    fn visit_vbitmask(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [v128] -> [i32]
        self.pop(offset)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_splat(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // [unpacked(t)] -> [v128]
        self.pop(offset)?;
        self.push(ValType::V128);
        Ok(None)
    }

    fn visit_replace_lane(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // shape.replace_lane laneidx : [v128 unpacked(shape)] → [v128]
        self.pop_many(offset, 2)?;
        self.push(ValType::V128);
        Ok(None)
    }

    fn visit_extract_lane(
        &mut self,
        offset: usize,
        unpacked_shape: ValType,
    ) -> Result<Option<u64>, Error> {
        // txN.extract_lane_sx ? laneidx : [v128] → [unpacked(shape)]
        self.pop(offset)?;
        self.push(unpacked_shape);
        Ok(None)
    }

    fn visit_load(&mut self, offset: usize, t: ValType) -> Result<Option<u64>, Error> {
        // t.load memarg           : [i32] → [t]
        // t.loadN_sx memarg       : [i32] → [t]
        // v128.loadNxM_sx memarg  : [i32] → [v128]
        // v128.loadN_splat memarg : [i32] → [v128]
        // v128.loadN_zero memarg  : [i32] → [v128]
        // t.atomic.load memarg    : [i32] → [t]
        // t.atomic.load memargN_u : [i32] → [t]
        self.pop(offset)?;
        self.push(t);
        Ok(None)
    }

    fn visit_load_lane(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // v128.loadN_lane memarg laneidx : [i32 v128] → [v128]
        self.pop_many(offset, 2)?;
        self.push(ValType::V128);
        Ok(None)
    }

    fn visit_store(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // t.store memarg           : [i32 t] → []
        // t.storeN memarg          : [i32 t] → []
        // t.atomic.store memarg    : [i32 t] → []
        // t.atomic.store memargN_u : [i32 t] → []
        self.pop_many(offset, 2)?;
        Ok(None)
    }

    fn visit_store_lane(&mut self, offset: usize) -> Result<Option<u64>, Error> {
        // v128.storeN_lane memarg laneidx : [i32 v128] → []
        self.pop_many(offset, 2)?;
        Ok(None)
    }

    fn visit_atomic_rmw(&mut self, offset: usize, t: ValType) -> Result<Option<u64>, Error> {
        // t.atomic.rmw.atop memarg : [i32 t] → [t]
        // t.atomic.rmwN.atop_u memarg : [i32 t] → [t]
        self.pop_many(offset, 2)?;
        self.push(t);
        Ok(None)
    }

    fn visit_atomic_cmpxchg(&mut self, offset: usize, t: ValType) -> Result<Option<u64>, Error> {
        // t.atomic.rmw.cmpxchg : [i32 t t] → [t]
        // t.atomic.rmwN.cmpxchg_u : [i32 t t] → [t]
        self.pop_many(offset, 3)?;
        self.push(t);
        Ok(None)
    }

    fn visit_function_call(
        &mut self,
        offset: usize,
        type_index: u32,
    ) -> Result<Option<u64>, Error> {
        let (params, results) = self.type_params_results(type_index)?;
        self.pop_many(offset, params.len())?;
        for result_ty in results {
            self.push(*result_ty);
        }
        Ok(None)
    }
}

macro_rules! instruction_category {
    ($($type:ident . const = $($insn:ident, $param: ty)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: $param) -> Self::Output {
            self.visit_const(ValType::$type)
        })*)*
    };
    ($($type:ident . unop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_unop()
        })*)*
    };
    ($($type:ident . binop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_binop(offset)
        })*)*
    };
    ($($type:ident . testop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_testop(offset)
        })*)*
    };

    ($($type:ident . relop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_relop(offset)
        })*)*
    };

    ($($type:ident . cvtop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_cvtop(offset, ValType::$type)
        })*)*
    };

    ($($type:ident . load = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: MemArg) -> Self::Output {
            self.visit_load(offset, ValType::$type)
        })*)*
    };

    ($($type:ident . store = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: MemArg) -> Self::Output {
            self.visit_store(offset)
        })*)*
    };

    ($($type:ident . loadlane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: MemArg, _: u8) -> Self::Output {
            self.visit_load_lane(offset)
        })*)*
    };

    ($($type:ident . storelane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: MemArg, _: u8) -> Self::Output {
            self.visit_store_lane(offset)
        })*)*
    };

    ($($type:ident . vternop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_vternop(offset)
        })*)*
    };

    ($($type:ident . vrelop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_vrelop(offset)
        })*)*
    };

    ($($type:ident . vishiftop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_vishiftop(offset)
        })*)*
    };

    ($($type:ident . vinarrowop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_vinarrowop(offset)
        })*)*
    };

    ($($type:ident . vbitmask = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_vbitmask(offset)
        })*)*
    };

    ($($type:ident . splat = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize) -> Self::Output {
            self.visit_splat(offset)
        })*)*
    };

    ($($type:ident . replacelane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: u8) -> Self::Output {
            self.visit_replace_lane(offset)
        })*)*
    };

    ($($type:ident . extractlane = $($insn:ident to $unpacked:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: u8) -> Self::Output {
            self.visit_extract_lane(offset, ValType::$unpacked)
        })*)*
    };

    ($($type:ident . atomic.rmw = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: MemArg) -> Self::Output {
            self.visit_atomic_rmw(offset, ValType::$type)
        })*)*
    };

    ($($type:ident . atomic.cmpxchg = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, offset: usize, _: MemArg) -> Self::Output {
            self.visit_atomic_cmpxchg(offset, ValType::$type)
        })*)*
    };
}

impl<'a, 'cfg, Cfg: AnalysisConfig> wasmparser::VisitOperator<'a> for StackSizeVisitor<'cfg, Cfg> {
    type Output = Result<Option<u64>, Error>;

    instruction_category! {
        I32.const = visit_i32_const, i32;
        I64.const = visit_i64_const, i64;
        F32.const = visit_f32_const, Ieee32;
        F64.const = visit_f64_const, Ieee64;
        V128.const = visit_v128_const, V128;
    }

    fn visit_ref_null(&mut self, _: usize, t: ValType) -> Self::Output {
        // [] -> [t]
        self.push(t);
        Ok(None)
    }

    fn visit_ref_func(&mut self, offset: usize, _: u32) -> Self::Output {
        self.visit_ref_null(offset, ValType::FuncRef)
    }

    instruction_category! {
        I32.unop = visit_i32_clz | visit_i32_ctz | visit_i32_popcnt;

        I64.unop = visit_i64_clz | visit_i64_ctz | visit_i64_popcnt;

        F32.unop = visit_f32_abs | visit_f32_neg | visit_f32_sqrt | visit_f32_ceil
                 | visit_f32_floor | visit_f32_trunc |visit_f32_nearest;

        F64.unop = visit_f64_abs | visit_f64_neg | visit_f64_sqrt | visit_f64_ceil
                 | visit_f64_floor | visit_f64_trunc |visit_f64_nearest;

        V128.unop = visit_v128_not;

        V128.unop = visit_i8x16_abs | visit_i8x16_neg | visit_i8x16_popcnt;
        V128.unop = visit_i16x8_abs | visit_i16x8_neg;
        V128.unop = visit_i32x4_abs | visit_i32x4_neg;
        V128.unop = visit_i64x2_abs | visit_i64x2_neg;

        V128.unop = visit_f32x4_abs | visit_f32x4_neg | visit_f32x4_sqrt | visit_f32x4_ceil
                  | visit_f32x4_floor | visit_f32x4_trunc | visit_f32x4_nearest;
        V128.unop = visit_f64x2_abs | visit_f64x2_neg | visit_f64x2_sqrt | visit_f64x2_ceil
                  | visit_f64x2_floor | visit_f64x2_trunc | visit_f64x2_nearest;

        // ishape1.extadd_pairwise_ishape2_sx : [v128] → [v128]
        V128.unop = visit_i16x8_extadd_pairwise_i8x16_s | visit_i16x8_extadd_pairwise_i8x16_u;
        V128.unop = visit_i32x4_extadd_pairwise_i16x8_s | visit_i32x4_extadd_pairwise_i16x8_u;
    }

    instruction_category! {
        I32.binop = visit_i32_add | visit_i32_sub | visit_i32_mul
                  | visit_i32_div_s | visit_i32_div_u
                  | visit_i32_rem_s | visit_i32_rem_u
                  | visit_i32_and | visit_i32_or | visit_i32_xor | visit_i32_shl
                  | visit_i32_shr_s | visit_i32_shr_u
                  | visit_i32_rotl | visit_i32_rotr;
        I64.binop = visit_i64_add | visit_i64_sub | visit_i64_mul
                  | visit_i64_div_s | visit_i64_div_u
                  | visit_i64_rem_s | visit_i64_rem_u
                  | visit_i64_and | visit_i64_or | visit_i64_xor | visit_i64_shl
                  | visit_i64_shr_s | visit_i64_shr_u
                  | visit_i64_rotl | visit_i64_rotr;
        F32.binop = visit_f32_add | visit_f32_sub | visit_f32_mul
                  | visit_f32_div | visit_f32_min | visit_f32_max | visit_f32_copysign;
        F64.binop = visit_f64_add | visit_f64_sub | visit_f64_mul
                  | visit_f64_div | visit_f64_min | visit_f64_max | visit_f64_copysign;

        V128.binop = visit_v128_and | visit_v128_andnot | visit_v128_or | visit_v128_xor;

        V128.binop = visit_i8x16_add | visit_i8x16_sub;
        V128.binop = visit_i16x8_add | visit_i16x8_sub | visit_i16x8_mul;
        V128.binop = visit_i32x4_add | visit_i32x4_sub | visit_i32x4_mul;
        V128.binop = visit_i64x2_add | visit_i64x2_sub | visit_i64x2_mul;

        V128.binop = visit_f32x4_add | visit_f32x4_sub | visit_f32x4_mul | visit_f32x4_div
                   | visit_f32x4_min | visit_f32x4_max | visit_f32x4_pmin | visit_f32x4_pmax
                   | visit_f32x4_relaxed_min | visit_f32x4_relaxed_max;
        V128.binop = visit_f64x2_add | visit_f64x2_sub | visit_f64x2_mul | visit_f64x2_div
                   | visit_f64x2_min | visit_f64x2_max | visit_f64x2_pmin | visit_f64x2_pmax
                   | visit_f64x2_relaxed_min | visit_f64x2_relaxed_max;

        V128.binop = visit_i8x16_min_s | visit_i8x16_min_u | visit_i8x16_max_s | visit_i8x16_max_u;
        V128.binop = visit_i16x8_min_s | visit_i16x8_min_u | visit_i16x8_max_s | visit_i16x8_max_u;
        V128.binop = visit_i32x4_min_s | visit_i32x4_min_u | visit_i32x4_max_s | visit_i32x4_max_u;

        V128.binop = visit_i8x16_add_sat_s | visit_i8x16_add_sat_u
                   | visit_i8x16_sub_sat_s | visit_i8x16_sub_sat_u;
        V128.binop = visit_i16x8_add_sat_s | visit_i16x8_add_sat_u
                   | visit_i16x8_sub_sat_s | visit_i16x8_sub_sat_u;

        V128.binop = visit_i8x16_avgr_u;
        V128.binop = visit_i16x8_avgr_u | visit_i16x8_q15mulr_sat_s | visit_i16x8_relaxed_q15mulr_s;

        // ishape1.dot_ishape2_s : [v128 v128] → [v128]
        V128.binop = visit_i32x4_dot_i16x8_s;
        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-integer-dot-product
        V128.binop = visit_i16x8_dot_i8x16_i7x16_s;


        // ishape1.extmul_half_ishape2_sx : [v128 v128] → [v128]
        V128.binop = visit_i16x8_extmul_low_i8x16_s | visit_i16x8_extmul_high_i8x16_s
                   | visit_i16x8_extmul_low_i8x16_u | visit_i16x8_extmul_high_i8x16_u;
        V128.binop = visit_i32x4_extmul_low_i16x8_s | visit_i32x4_extmul_high_i16x8_s
                   | visit_i32x4_extmul_low_i16x8_u | visit_i32x4_extmul_high_i16x8_u;
        V128.binop = visit_i64x2_extmul_low_i32x4_s | visit_i64x2_extmul_high_i32x4_s
                   | visit_i64x2_extmul_low_i32x4_u | visit_i64x2_extmul_high_i32x4_u;

        // i8x16.swizzle : [v128 v128] → [v128]
        // i8x16.relaxed_swizzle : [v128 v128] → [v128]
        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-swizzle
        V128.binop = visit_i8x16_swizzle | visit_i8x16_relaxed_swizzle;
    }

    instruction_category! {
        V128.vishiftop = visit_i8x16_shl | visit_i8x16_shr_s | visit_i8x16_shr_u;
        V128.vishiftop = visit_i16x8_shl | visit_i16x8_shr_s | visit_i16x8_shr_u;
        V128.vishiftop = visit_i32x4_shl | visit_i32x4_shr_s | visit_i32x4_shr_u;
        V128.vishiftop = visit_i64x2_shl | visit_i64x2_shr_s | visit_i64x2_shr_u;
    }

    instruction_category! {
        I32.testop = visit_i32_eqz;
        I64.testop = visit_i64_eqz;
        V128.testop = visit_v128_any_true
                    | visit_i8x16_all_true | visit_i16x8_all_true
                    | visit_i32x4_all_true | visit_i64x2_all_true;
        FuncRef.testop = visit_ref_is_null;
    }

    instruction_category! {
        I32.relop = visit_i32_eq | visit_i32_ne
                  | visit_i32_lt_s | visit_i32_lt_u | visit_i32_gt_s | visit_i32_gt_u
                  | visit_i32_le_s | visit_i32_le_u | visit_i32_ge_s | visit_i32_ge_u;
        I64.relop = visit_i64_eq | visit_i64_ne
                  | visit_i64_lt_s | visit_i64_lt_u | visit_i64_gt_s | visit_i64_gt_u
                  | visit_i64_le_s | visit_i64_le_u | visit_i64_ge_s | visit_i64_ge_u;
        F32.relop = visit_f32_eq | visit_f32_ne
                  | visit_f32_lt | visit_f32_gt | visit_f32_le | visit_f32_ge;
        F64.relop = visit_f64_eq | visit_f64_ne
                  | visit_f64_lt | visit_f64_gt | visit_f64_le | visit_f64_ge;
    }

    instruction_category! {
        I32.cvtop = visit_i32_wrap_i64
                  | visit_i32_extend8_s | visit_i32_extend16_s
                  | visit_i32_trunc_f32_s | visit_i32_trunc_f32_u
                  | visit_i32_trunc_f64_s | visit_i32_trunc_f64_u
                  | visit_i32_trunc_sat_f32_s | visit_i32_trunc_sat_f32_u
                  | visit_i32_trunc_sat_f64_s | visit_i32_trunc_sat_f64_u
                  | visit_i32_reinterpret_f32;

        I64.cvtop = visit_i64_extend8_s | visit_i64_extend16_s | visit_i64_extend32_s
                  | visit_i64_extend_i32_s | visit_i64_extend_i32_u
                  | visit_i64_trunc_f32_s | visit_i64_trunc_f32_u
                  | visit_i64_trunc_f64_s | visit_i64_trunc_f64_u
                  | visit_i64_trunc_sat_f32_s | visit_i64_trunc_sat_f32_u
                  | visit_i64_trunc_sat_f64_s | visit_i64_trunc_sat_f64_u
                  | visit_i64_reinterpret_f64;

        F32.cvtop = visit_f32_demote_f64
                  | visit_f32_convert_i32_s | visit_f32_convert_i32_u
                  | visit_f32_convert_i64_s | visit_f32_convert_i64_u
                  | visit_f32_reinterpret_i32;
        F64.cvtop = visit_f64_promote_f32
                  | visit_f64_convert_i32_s | visit_f64_convert_i32_u
                  | visit_f64_convert_i64_s | visit_f64_convert_i64_u
                  | visit_f64_reinterpret_i64;

        V128.cvtop = visit_i16x8_extend_low_i8x16_s | visit_i16x8_extend_high_i8x16_s
                   | visit_i16x8_extend_low_i8x16_u | visit_i16x8_extend_high_i8x16_u;

        V128.cvtop = visit_i32x4_extend_low_i16x8_s | visit_i32x4_extend_high_i16x8_s
                   | visit_i32x4_extend_low_i16x8_u | visit_i32x4_extend_high_i16x8_u
                   | visit_i32x4_trunc_sat_f32x4_s | visit_i32x4_trunc_sat_f32x4_u
                   | visit_i32x4_trunc_sat_f64x2_s_zero | visit_i32x4_trunc_sat_f64x2_u_zero
                   | visit_i32x4_relaxed_trunc_sat_f32x4_s
                   | visit_i32x4_relaxed_trunc_sat_f32x4_u
                   | visit_i32x4_relaxed_trunc_sat_f64x2_s_zero
                   | visit_i32x4_relaxed_trunc_sat_f64x2_u_zero;

        V128.cvtop = visit_i64x2_extend_low_i32x4_s | visit_i64x2_extend_high_i32x4_s
                   | visit_i64x2_extend_low_i32x4_u | visit_i64x2_extend_high_i32x4_u;

        V128.cvtop = visit_f32x4_demote_f64x2_zero
                   | visit_f32x4_convert_i32x4_s | visit_f32x4_convert_i32x4_u;

        V128.cvtop = visit_f64x2_promote_low_f32x4
                   | visit_f64x2_convert_low_i32x4_s | visit_f64x2_convert_low_i32x4_u;
    }

    instruction_category! {
        I32.load = visit_i32_load
                 | visit_i32_load8_s | visit_i32_load8_u
                 | visit_i32_load16_s | visit_i32_load16_u
                 | visit_i32_atomic_load
                 | visit_i32_atomic_load8_u
                 | visit_i32_atomic_load16_u;

        I64.load = visit_i64_load
                 | visit_i64_load8_s | visit_i64_load8_u
                 | visit_i64_load16_s | visit_i64_load16_u
                 | visit_i64_load32_s | visit_i64_load32_u
                 | visit_i64_atomic_load
                 | visit_i64_atomic_load8_u
                 | visit_i64_atomic_load16_u
                 | visit_i64_atomic_load32_u;

        F32.load = visit_f32_load;

        F64.load = visit_f64_load;

        V128.load = visit_v128_load
                  | visit_v128_load8x8_s | visit_v128_load8x8_u
                  | visit_v128_load16x4_s | visit_v128_load16x4_u
                  | visit_v128_load32x2_s | visit_v128_load32x2_u
                  | visit_v128_load32_zero | visit_v128_load64_zero
                  | visit_v128_load8_splat | visit_v128_load16_splat
                  | visit_v128_load32_splat | visit_v128_load64_splat;
    }

    instruction_category! {
        I32.store = visit_i32_store | visit_i32_store8 | visit_i32_store16
                  | visit_i32_atomic_store | visit_i32_atomic_store8 | visit_i32_atomic_store16;

        I64.store = visit_i64_store | visit_i64_store8 | visit_i64_store16 | visit_i64_store32
                  | visit_i64_atomic_store
                  | visit_i64_atomic_store8 | visit_i64_atomic_store16 | visit_i64_atomic_store32;

        F32.store = visit_f32_store;

        F64.store = visit_f64_store;

        V128.store = visit_v128_store;
    }

    instruction_category! {
        V128.loadlane = visit_v128_load8_lane | visit_v128_load16_lane
                       | visit_v128_load32_lane | visit_v128_load64_lane;
    }

    instruction_category! {
        V128.storelane = visit_v128_store8_lane | visit_v128_store16_lane
                        | visit_v128_store32_lane | visit_v128_store64_lane;
    }

    instruction_category! {
        V128.replacelane = visit_i8x16_replace_lane | visit_i16x8_replace_lane
                         | visit_i32x4_replace_lane | visit_i64x2_replace_lane
                         | visit_f32x4_replace_lane | visit_f64x2_replace_lane;
    }

    instruction_category! {
        V128.extractlane = visit_i8x16_extract_lane_s to I32 | visit_i8x16_extract_lane_u to I32
                         | visit_i16x8_extract_lane_s to I32 | visit_i16x8_extract_lane_u to I32
                         | visit_i32x4_extract_lane to I32
                         | visit_i64x2_extract_lane to I64
                         | visit_f32x4_extract_lane to F32
                         | visit_f64x2_extract_lane to F64;
    }

    instruction_category! {
        V128.vternop = visit_v128_bitselect;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-laneselect
        V128.vternop = visit_i8x16_relaxed_laneselect;
        V128.vternop = visit_i16x8_relaxed_laneselect;
        V128.vternop = visit_i32x4_relaxed_laneselect;
        V128.vternop = visit_i64x2_relaxed_laneselect;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-fused-multiply-add-and-fused-negative-multiply-add
        V128.vternop = visit_f32x4_relaxed_fma | visit_f32x4_relaxed_fnma;
        V128.vternop = visit_f64x2_relaxed_fma | visit_f64x2_relaxed_fnma;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-integer-dot-product
        V128.vternop = visit_i32x4_dot_i8x16_i7x16_add_s;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-bfloat16-dot-product
        V128.vternop = visit_f32x4_relaxed_dot_bf16x8_add_f32x4;
    }

    instruction_category! {
        V128.vrelop = visit_i8x16_eq | visit_i8x16_ne
                    | visit_i8x16_lt_s | visit_i8x16_lt_u | visit_i8x16_gt_s | visit_i8x16_gt_u
                    | visit_i8x16_le_s | visit_i8x16_le_u | visit_i8x16_ge_s | visit_i8x16_ge_u;
        V128.vrelop = visit_i16x8_eq | visit_i16x8_ne
                    | visit_i16x8_lt_s | visit_i16x8_lt_u | visit_i16x8_gt_s | visit_i16x8_gt_u
                    | visit_i16x8_le_s | visit_i16x8_le_u | visit_i16x8_ge_s | visit_i16x8_ge_u;
        V128.vrelop = visit_i32x4_eq | visit_i32x4_ne
                    | visit_i32x4_lt_s | visit_i32x4_lt_u | visit_i32x4_gt_s | visit_i32x4_gt_u
                    | visit_i32x4_le_s | visit_i32x4_le_u | visit_i32x4_ge_s | visit_i32x4_ge_u;
        V128.vrelop = visit_i64x2_eq | visit_i64x2_ne
                    | visit_i64x2_lt_s | visit_i64x2_gt_s | visit_i64x2_le_s | visit_i64x2_ge_s;
        V128.vrelop = visit_f32x4_eq | visit_f32x4_ne
                    | visit_f32x4_lt | visit_f32x4_gt | visit_f32x4_le | visit_f32x4_ge;
        V128.vrelop = visit_f64x2_eq | visit_f64x2_ne
                    | visit_f64x2_lt | visit_f64x2_gt | visit_f64x2_le | visit_f64x2_ge;
    }

    instruction_category! {
        V128.vinarrowop = visit_i8x16_narrow_i16x8_s | visit_i8x16_narrow_i16x8_u;
        V128.vinarrowop = visit_i16x8_narrow_i32x4_s | visit_i16x8_narrow_i32x4_u;
    }

    instruction_category! {
        V128.vbitmask = visit_i8x16_bitmask | visit_i16x8_bitmask
                      | visit_i32x4_bitmask | visit_i64x2_bitmask;
    }

    instruction_category! {
        V128.splat = visit_i8x16_splat | visit_i16x8_splat | visit_i32x4_splat | visit_i64x2_splat
                   | visit_f32x4_splat | visit_f64x2_splat;
    }

    fn visit_i8x16_shuffle(&mut self, offset: usize, _: [u8; 16]) -> Self::Output {
        // i8x16.shuffle laneidx^16 : [v128 v128] → [v128]
        self.pop(offset)?;
        Ok(None)
    }

    instruction_category! {
        I32.atomic.rmw = visit_i32_atomic_rmw_add | visit_i32_atomic_rmw_sub
                       | visit_i32_atomic_rmw_and | visit_i32_atomic_rmw_or
                       | visit_i32_atomic_rmw_xor | visit_i32_atomic_rmw_xchg
                       | visit_i32_atomic_rmw8_add_u | visit_i32_atomic_rmw16_add_u
                       | visit_i32_atomic_rmw8_sub_u | visit_i32_atomic_rmw16_sub_u
                       | visit_i32_atomic_rmw8_and_u | visit_i32_atomic_rmw16_and_u
                       | visit_i32_atomic_rmw8_or_u | visit_i32_atomic_rmw16_or_u
                       | visit_i32_atomic_rmw8_xor_u | visit_i32_atomic_rmw16_xor_u
                       | visit_i32_atomic_rmw8_xchg_u | visit_i32_atomic_rmw16_xchg_u;
        I64.atomic.rmw = visit_i64_atomic_rmw_add | visit_i64_atomic_rmw_sub
                       | visit_i64_atomic_rmw_and | visit_i64_atomic_rmw_or
                       | visit_i64_atomic_rmw_xor | visit_i64_atomic_rmw_xchg
                       | visit_i64_atomic_rmw8_add_u | visit_i64_atomic_rmw16_add_u
                       | visit_i64_atomic_rmw32_add_u
                       | visit_i64_atomic_rmw8_sub_u | visit_i64_atomic_rmw16_sub_u
                       | visit_i64_atomic_rmw32_sub_u
                       | visit_i64_atomic_rmw8_and_u | visit_i64_atomic_rmw16_and_u
                       | visit_i64_atomic_rmw32_and_u
                       | visit_i64_atomic_rmw8_or_u | visit_i64_atomic_rmw16_or_u
                       | visit_i64_atomic_rmw32_or_u
                       | visit_i64_atomic_rmw8_xor_u | visit_i64_atomic_rmw16_xor_u
                       | visit_i64_atomic_rmw32_xor_u
                       | visit_i64_atomic_rmw8_xchg_u | visit_i64_atomic_rmw16_xchg_u
                       | visit_i64_atomic_rmw32_xchg_u;
    }

    instruction_category! {
        I32.atomic.cmpxchg = visit_i32_atomic_rmw_cmpxchg | visit_i32_atomic_rmw8_cmpxchg_u
                           | visit_i32_atomic_rmw16_cmpxchg_u;
        I64.atomic.cmpxchg = visit_i64_atomic_rmw_cmpxchg | visit_i64_atomic_rmw8_cmpxchg_u
                           | visit_i64_atomic_rmw16_cmpxchg_u | visit_i64_atomic_rmw32_cmpxchg_u;
    }

    fn visit_memory_atomic_notify(&mut self, offset: usize, _: MemArg) -> Self::Output {
        // [i32 i32] -> [i32]
        self.pop(offset)?;
        Ok(None)
    }

    fn visit_memory_atomic_wait32(&mut self, offset: usize, _: MemArg) -> Self::Output {
        // [i32 i32 i64] -> [i32]
        self.pop_many(offset, 3)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_memory_atomic_wait64(&mut self, offset: usize, _: MemArg) -> Self::Output {
        // [i32 i64 i64] -> [i32]
        self.pop_many(offset, 3)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_atomic_fence(&mut self, _: usize) -> Self::Output {
        // https://github.com/WebAssembly/threads/blob/main/proposals/threads/Overview.md#fence-operator
        // [] -> []

        // Function body intentionally left empty
        Ok(None)
    }

    fn visit_local_get(&mut self, _: usize, local_index: u32) -> Self::Output {
        // [] → [t]
        let local_type = self
            .locals
            .get(&local_index)
            .ok_or(Error::LocalIndex(local_index))?;
        self.push(*local_type);
        Ok(None)
    }

    fn visit_local_set(&mut self, offset: usize, _: u32) -> Self::Output {
        // [] → [t]
        self.pop(offset)?;
        Ok(None)
    }

    fn visit_local_tee(&mut self, _: usize, _: u32) -> Self::Output {
        // [t] → [t]
        Ok(None)
    }

    fn visit_global_get(&mut self, _: usize, global: u32) -> Self::Output {
        // [] → [t]
        let global_usize =
            usize::try_from(global).map_err(|e| Error::GlobalIndexRange(global, e))?;
        let global_ty = self
            .globals
            .get(global_usize)
            .ok_or(Error::GlobalIndex(global))?;
        self.push(*global_ty);
        Ok(None)
    }

    fn visit_global_set(&mut self, offset: usize, _: u32) -> Self::Output {
        // [t] → []
        self.pop(offset)?;
        Ok(None)
    }

    fn visit_memory_size(&mut self, _: usize, _: u32, _: u8) -> Self::Output {
        // [] → [i32]
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_memory_grow(&mut self, _: usize, _: u32, _: u8) -> Self::Output {
        // [i32] → [i32]

        // Function body intentionally left empty.
        Ok(None)
    }

    fn visit_memory_fill(&mut self, offset: usize, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(offset, 3)?;
        Ok(None)
    }

    fn visit_memory_init(&mut self, offset: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(offset, 3)?;
        Ok(None)
    }

    fn visit_memory_copy(&mut self, offset: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(offset, 3)?;
        Ok(None)
    }

    fn visit_data_drop(&mut self, _: usize, _: u32) -> Self::Output {
        // [] → []
        Ok(None)
    }

    fn visit_table_get(&mut self, offset: usize, table: u32) -> Self::Output {
        // [i32] → [t]
        let table_usize = usize::try_from(table).map_err(|e| Error::TableIndexRange(table, e))?;
        let table_ty = *self
            .tables
            .get(table_usize)
            .ok_or(Error::TableIndex(table))?;
        self.pop(offset)?;
        self.push(table_ty);
        Ok(None)
    }

    fn visit_table_set(&mut self, offset: usize, _: u32) -> Self::Output {
        // [i32 t] → []
        self.pop_many(offset, 2)?;
        Ok(None)
    }

    fn visit_table_size(&mut self, _: usize, _: u32) -> Self::Output {
        // [] → [i32]
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_table_grow(&mut self, offset: usize, _: u32) -> Self::Output {
        // [t i32] → [i32]
        self.pop_many(offset, 2)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_table_fill(&mut self, offset: usize, _: u32) -> Self::Output {
        // [i32 t i32] → []
        self.pop_many(offset, 3)?;
        Ok(None)
    }

    fn visit_table_copy(&mut self, offset: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(offset, 3)?;
        Ok(None)
    }

    fn visit_table_init(&mut self, offset: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(offset, 3)?;
        Ok(None)
    }

    fn visit_elem_drop(&mut self, _: usize, _: u32) -> Self::Output {
        // [] → []
        Ok(None)
    }

    fn visit_select(&mut self, offset: usize) -> Self::Output {
        // [t t i32] -> [t]
        self.pop_many(offset, 2)?;
        Ok(None)
    }

    fn visit_typed_select(&mut self, offset: usize, t: ValType) -> Self::Output {
        // [t t i32] -> [t]
        self.pop_many(offset, 3)?;
        self.push(t);
        Ok(None)
    }

    fn visit_drop(&mut self, offset: usize) -> Self::Output {
        // [t] → []
        self.pop(offset)?;
        Ok(None)
    }

    fn visit_nop(&mut self, _: usize) -> Self::Output {
        // [] → []
        Ok(None)
    }

    fn visit_call(&mut self, offset: usize, function_index: u32) -> Self::Output {
        self.visit_function_call(offset, self.function_type_index(function_index)?)
    }

    fn visit_call_indirect(
        &mut self,
        offset: usize,
        type_index: u32,
        _: u32,
        _: u8,
    ) -> Self::Output {
        self.visit_function_call(offset, type_index)
    }

    fn visit_return_call(&mut self, offset: usize, _: u32) -> Self::Output {
        // `return_call` behaves as-if a regular `return` followed by the `call`. For the purposes
        // of modelling the frame size of the _current_ function, only the `return` portion of this
        // computation is relevant (as it makes the stack polymorphic)
        self.visit_return(offset)
    }

    fn visit_return_call_indirect(&mut self, offset: usize, _: u32, _: u32) -> Self::Output {
        self.visit_return(offset)
    }

    fn visit_unreachable(&mut self, _: usize) -> Self::Output {
        // [*] → [*]  (stack-polymorphic)
        self.stack_polymorphic();
        Ok(None)
    }

    fn visit_block(&mut self, _: usize, blockty: BlockType) -> Self::Output {
        // block blocktype instr* end : [t1*] → [t2*]
        self.new_frame(blockty)?;
        Ok(None)
    }

    fn visit_loop(&mut self, _: usize, blockty: BlockType) -> Self::Output {
        // loop blocktype instr* end : [t1*] → [t2*]
        self.new_frame(blockty)?;
        Ok(None)
    }

    fn visit_if(&mut self, offset: usize, blockty: BlockType) -> Self::Output {
        // if blocktype instr* else instr* end : [t1* i32] → [t2*]
        self.pop(offset)?;
        self.pop_block_params(offset, blockty)?;
        self.new_frame(blockty)?;
        Ok(None)
    }

    fn visit_else(&mut self, offset: usize) -> Self::Output {
        if let Some(frame) = self.frames.pop() {
            let frame = std::mem::replace(&mut self.top_frame, frame);
            self.operands.truncate(frame.height);
            if self.operands.len() != frame.height {
                return Err(Error::MalformedModule(offset, "operand stack is too short"));
            }
            self.new_frame(frame.block_type)?;
            Ok(None)
        } else {
            return Err(Error::MalformedModule(offset, "frame stack is too short"));
        }
    }

    fn visit_end(&mut self, offset: usize) -> Self::Output {
        if let Some(frame) = self.frames.pop() {
            let frame = std::mem::replace(&mut self.top_frame, frame);
            self.operands.truncate(frame.height);
            if self.operands.len() != frame.height {
                return Err(Error::MalformedModule(offset, "operand stack is too short"));
            }
            self.push_block_results(frame.block_type)?;
            Ok(None)
        } else {
            // Returning from the function. Malformed WASM modules may have trailing instructions,
            // but we do ignore processing them in the operand feed loop. For that reason,
            // replacing `top_stack` with some sentinel value would work okay.
            self.top_frame = Frame {
                height: !0,
                block_type: BlockType::Empty,
                stack_polymorphic: true,
            };
            self.operands.clear();
            Ok(Some(self.max_size))
        }
    }

    fn visit_br(&mut self, _: usize, _: u32) -> Self::Output {
        // [t1* t*] → [t2*]  (stack-polymorphic)
        self.stack_polymorphic();
        Ok(None)
    }

    fn visit_br_if(&mut self, offset: usize, _: u32) -> Self::Output {
        // [t* i32] → [t*]

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
        self.pop(offset)?;
        // …the condition.
        Ok(None)
    }

    fn visit_br_table(&mut self, offset: usize, _: BrTable) -> Self::Output {
        // [t1* t* i32] → [t2*]  (stack-polymorphic)
        self.pop(offset)?; // table index
        self.stack_polymorphic();
        Ok(None)
    }

    fn visit_return(&mut self, offset: usize) -> Self::Output {
        // This behaves as-if a `br` to the outer-most block.
        let branch_depth =
            u32::try_from(self.frames.len().saturating_sub(1)).map_err(|_| Error::TooManyFrames)?;
        self.visit_br(offset, branch_depth)
    }

    fn visit_try(&mut self, _: usize, _: BlockType) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_rethrow(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_throw(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_delegate(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_catch(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_catch_all(&mut self, _: usize) -> Self::Output {
        todo!("exception handling has not been implemented");
    }
}
