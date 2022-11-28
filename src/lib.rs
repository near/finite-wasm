use gas::InstructionKind;
use prefix_sum_vec::PrefixSumVec;
use std::num::TryFromIntError;
use visitors::VisitOperatorWithOffset;
use wasmparser::{BinaryReaderError, BlockType, VisitOperator};

pub mod gas;
mod instruction_categories;
pub mod max_stack;
mod visitors;

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
    #[error("could not process locals for function ${1}")]
    TooManyLocals(#[source] prefix_sum_vec::TryPushError, u32),
    #[error("type index {0} refers to a non-existent type")]
    TypeIndex(u32),
    #[error("type index {0} is out of range")]
    TypeIndexRange(u32, #[source] TryFromIntError),
    #[error("function index {0} refers to a non-existent type")]
    FunctionIndex(u32),

    #[error("could not process operator for max_stack analysis")]
    MaxStack(#[source] max_stack::Error),
    #[error("could not process operator for gas analysis")]
    Gas(#[source] gas::Error),
}

/// The results of parsing and analyzing the module.
///
/// This analysis collects information necessary to implement all of the transformations in one go,
/// so that re-parsing the module multiple times is not necessary.
pub struct Module {
    /// The sizes of the stack frame (the maximum amount of stack used at any given time during the
    /// execution of the function) for each function.
    pub function_stack_sizes: Vec<u64>,

    /// The table of offsets for gas instrumentation points
    pub gas_offsets: Vec<Box<[usize]>>,
    pub gas_costs: Vec<Box<[u64]>>,
    pub gas_kinds: Vec<Box<[InstructionKind]>>,
}

impl Module {
    pub fn new(
        module: &[u8],
        max_stack_cfg: Option<&impl max_stack::Config>,
        mut gas_cfg: Option<impl for<'a> VisitOperator<'a, Output = u64>>,
    ) -> Result<Self, Error> {
        let mut types = vec![];
        let mut functions = vec![];
        let mut function_stack_sizes = vec![];
        let mut globals = vec![];
        let mut tables = vec![];
        // Reused between functions for speeds.
        let mut locals = PrefixSumVec::new();
        let mut operand_stack = vec![];
        let mut max_stack_frame_stack = vec![];
        let mut gas_frame_stack = vec![];
        let mut offsets = vec![];
        let mut costs = vec![];
        let mut kinds = vec![];
        let mut gas_offsets = vec![];
        let mut gas_costs = vec![];
        let mut gas_kinds = vec![];

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
                    max_stack_frame_stack.clear();
                    offsets.clear();
                    kinds.clear();
                    costs.clear();

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

                    let mut gas_visitor = gas_cfg.as_mut().map(|config| gas::GasVisitor {
                        offset: 0,
                        model: config,
                        offsets: &mut offsets,
                        costs: &mut costs,
                        kinds: &mut kinds,
                        frame_stack: &mut gas_frame_stack,
                        current_frame: gas::Frame {
                            stack_polymorphic: false,
                            kind: gas::BranchTargetKind::UntakenForward,
                        },
                    });

                    let mut stack_visitor = max_stack_cfg.map(|config| {
                        // This includes accounting for any possible return pointer tracking,
                        // parameters and locals (which all are considered locals in wasm).
                        function_stack_sizes.push(config.size_of_function_activation(&locals));
                        max_stack::StackSizeVisitor {
                            offset: 0,

                            config,
                            functions: &functions,
                            types: &types,
                            globals: &globals,
                            tables: &tables,
                            locals: &locals,

                            operands: &mut operand_stack,
                            size: 0,
                            max_size: 0,

                            // Future optimization opportunity: Struct-of-Arrays representation.
                            frames: &mut max_stack_frame_stack,
                            current_frame: max_stack::Frame {
                                height: 0,
                                block_type: BlockType::Empty,
                                stack_polymorphic: false,
                            },
                        }
                    });
                    let mut nop_gas_visitor = visitors::NoOpVisitor(Ok(()));
                    let mut nop_stack_visitor = visitors::NoOpVisitor(Ok(None));

                    let mut combined_visitor = match (&mut gas_visitor, &mut stack_visitor) {
                        (Some(ref mut g), Some(ref mut s)) => visitors::JoinVisitor(
                            g as &mut dyn VisitOperatorWithOffset<Output = Result<(), gas::Error>>,
                            s as &mut dyn VisitOperatorWithOffset<
                                Output = Result<Option<u64>, max_stack::Error>,
                            >,
                        ),
                        (None, None) => visitors::JoinVisitor(
                            &mut nop_gas_visitor as &mut _,
                            &mut nop_stack_visitor as &mut _,
                        ),
                        (None, Some(ref mut s)) => {
                            visitors::JoinVisitor(&mut nop_gas_visitor as &mut _, s as &mut _)
                        }
                        (Some(ref mut g), None) => {
                            visitors::JoinVisitor(g as &mut _, &mut nop_stack_visitor as &mut _)
                        }
                    };
                    let mut operators = function
                        .get_operators_reader()
                        .map_err(Error::OperatorReader)?;
                    while !operators.eof() {
                        combined_visitor.set_offset(operators.original_position());
                        let (gas_result, stack_result) = operators
                            .visit_with_offset(&mut combined_visitor)
                            .map_err(Error::VisitOperators)?;
                        let stack_result = stack_result.map_err(Error::MaxStack)?;
                        gas_result.map_err(Error::Gas)?;
                        if let Some(stack_size) = stack_result {
                            let size = function_stack_sizes
                                .last_mut()
                                .expect("should have frame size already pushed if analyzing stack");
                            *size += stack_size;
                        }
                    }

                    if !kinds.is_empty() {
                        gas::optimize(&mut offsets, &mut costs, &mut kinds);
                        // FIXME(nagisa): this may have a much better representation… We might be
                        // able to avoid storing all the analysis results if we instrumented
                        // on-the-fly too...
                        gas_offsets.push(offsets.drain(..).collect());
                        gas_kinds.push(kinds.drain(..).collect());
                        gas_costs.push(costs.drain(..).collect());
                    }
                }
                _ => (),
            }
        }
        Ok(Self {
            function_stack_sizes,
            gas_costs,
            gas_kinds,
            gas_offsets,
        })
    }
}
