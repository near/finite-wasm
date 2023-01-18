#![doc = include_str!("../README.mkd")]

use gas::InstrumentationKind;
/// A re-export of the prefix_sum_vec crate. Use in implementing [`max_stack::SizeConfig`].
pub use prefix_sum_vec;
use prefix_sum_vec::PrefixSumVec;
use std::num::TryFromIntError;
use visitors::VisitOperatorWithOffset;
/// A re-export of the wasmparser crate. Use in implementing [`max_stack::SizeConfig`] and [`gas::CostModel`].
pub use wasmparser;
use wasmparser::{BinaryReaderError, BlockType};

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

/// The entry-point type to set-up your finite-wasm analysis.
///
/// This type allows running any number of analyses implemented by this crate. By default, none of
/// the analyses are run. Each can be enabled individually with methods such as
/// [`Self::with_stack`] or [`Self::with_gas`].
///
/// # Examples
///
/// See the [crate root](crate) for an example.
pub struct Analysis<StackConfig, GasCostModel> {
    max_stack_cfg: StackConfig,
    gas_cfg: GasCostModel,
}

impl Analysis<max_stack::NoSizeConfig, gas::NoCostModel> {
    pub fn new() -> Self {
        Self {
            max_stack_cfg: max_stack::NoSizeConfig,
            gas_cfg: gas::NoCostModel,
        }
    }
}

impl<SC, GC> Analysis<SC, GC> {
    pub fn with_stack<NewSC>(self, max_stack_cfg: NewSC) -> Analysis<NewSC, GC> {
        let Self { gas_cfg, .. } = self;
        Analysis {
            max_stack_cfg,
            gas_cfg,
        }
    }

    pub fn with_gas<NewGC>(self, gas_cfg: NewGC) -> Analysis<SC, NewGC> {
        let Self { max_stack_cfg, .. } = self;
        Analysis {
            max_stack_cfg,
            gas_cfg,
        }
    }
}

impl<SC: max_stack::SizeConfig, GC: for<'a> gas::CostModel<'a>> Analysis<SC, GC> {
    pub fn analyze(&mut self, module: &[u8]) -> Result<AnalysisOutcome, Error> {
        let mut types = vec![];
        let mut functions = vec![];
        let mut function_frame_sizes = vec![];
        let mut function_operand_stack_sizes = vec![];
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
        let mut current_fn_id = 0u32;

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
                                current_fn_id = current_fn_id
                                    .checked_add(1)
                                    .ok_or(Error::TooManyFunctions)?;
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

                    let function_id_usize = usize::try_from(current_fn_id)
                        .expect("failed converting from u32 to usize");
                    let type_id = *functions
                        .get(function_id_usize)
                        .ok_or(Error::FunctionIndex(current_fn_id))?;
                    let type_id_usize =
                        usize::try_from(type_id).map_err(|e| Error::TypeIndexRange(type_id, e))?;
                    let fn_type = types.get(type_id_usize).ok_or(Error::TypeIndex(type_id))?;

                    match fn_type {
                        wasmparser::Type::Func(fnty) => {
                            for param in fnty.params() {
                                locals
                                    .try_push_more(1, *param)
                                    .map_err(|e| Error::TooManyLocals(e, current_fn_id))?;
                            }
                        }
                    }
                    for local in function.get_locals_reader().map_err(Error::LocalsReader)? {
                        let local = local.map_err(Error::ParseLocals)?;
                        locals
                            .try_push_more(local.0, local.1)
                            .map_err(|e| Error::TooManyLocals(e, current_fn_id))?;
                    }

                    let mut gas_visitor;
                    let mut nop_gas_visitor = visitors::NoOpVisitor(Ok(()));
                    let gas_visitor = if let Some(model) = self.gas_cfg.to_visitor() {
                        gas_visitor = gas::GasVisitor {
                            offset: 0,
                            model,
                            offsets: &mut offsets,
                            costs: &mut costs,
                            kinds: &mut kinds,
                            frame_stack: &mut gas_frame_stack,
                            next_offset_cost: None,
                            current_frame: gas::Frame {
                                stack_polymorphic: false,
                                kind: gas::BranchTargetKind::UntakenForward,
                            },
                        };
                        &mut gas_visitor as &mut dyn VisitOperatorWithOffset<Output = Result<_, _>>
                    } else {
                        &mut nop_gas_visitor
                    };

                    let mut stack_visitor;
                    let mut nop_stack_visitor = visitors::NoOpVisitor(Ok(None));
                    let stack_visitor = if !self.max_stack_cfg.should_run(Default::default()) {
                        &mut nop_stack_visitor
                    } else {
                        // This includes accounting for any possible return pointer tracking,
                        // parameters and locals (which all are considered locals in wasm).
                        function_frame_sizes
                            .push(self.max_stack_cfg.size_of_function_activation(&locals));
                        stack_visitor = max_stack::StackSizeVisitor {
                            offset: 0,

                            config: &self.max_stack_cfg,
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
                        };
                        &mut stack_visitor
                            as &mut dyn VisitOperatorWithOffset<Output = Result<_, _>>
                    };

                    let mut combined_visitor = visitors::JoinVisitor(gas_visitor, stack_visitor);
                    let mut operators = function
                        .get_operators_reader()
                        .map_err(Error::OperatorReader)?;
                    while !operators.eof() {
                        combined_visitor.set_offset(operators.original_position());
                        let (gas_result, stack_result) = operators
                            .visit_operator(&mut combined_visitor)
                            .map_err(Error::VisitOperators)?;
                        let stack_result = stack_result.map_err(Error::MaxStack)?;
                        gas_result.map_err(Error::Gas)?;
                        if let Some(stack_size) = stack_result {
                            function_operand_stack_sizes.push(stack_size);
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
                    current_fn_id = current_fn_id
                        .checked_add(1)
                        .ok_or(Error::TooManyFunctions)?;
                }
                _ => (),
            }
        }
        Ok(AnalysisOutcome {
            function_frame_sizes,
            function_operand_stack_sizes,
            gas_costs,
            gas_kinds,
            gas_offsets,
        })
    }
}

/// The results of parsing and analyzing the module.
///
/// This analysis collects information necessary to implement all of the transformations in one go,
/// so that re-parsing the module multiple times is not necessary.
pub struct AnalysisOutcome {
    /// The sizes of the stack frame for each function in the module, *excluding* imports.
    ///
    /// This includes the things like the function label and the locals that are 0-initialized.
    pub function_frame_sizes: Vec<u64>,
    /// The maximum size of the operand stack for each function in the module, *excluding* imports.
    ///
    /// Throughout the execution the sum of sizes of the operands on the function’s operand stack
    /// will differ, but will never exceed the number here.
    pub function_operand_stack_sizes: Vec<u64>,

    /// The table of offsets for gas instrumentation points, *excluding* imports.
    pub gas_offsets: Vec<Box<[usize]>>,
    pub gas_costs: Vec<Box<[u64]>>,
    pub gas_kinds: Vec<Box<[InstrumentationKind]>>,
}
