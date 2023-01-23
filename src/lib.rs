#![doc = include_str!("../README.mkd")]

use gas::InstrumentationKind;
/// A re-export of the prefix_sum_vec crate. Use in implementing [`max_stack::SizeConfig`].
pub use prefix_sum_vec;
use std::num::TryFromIntError;
use visitors::VisitOperatorWithOffset;
/// A re-export of the wasmparser crate. Use in implementing [`max_stack::SizeConfig`] and
/// [`gas::Config`].
pub use wasmparser;
use wasmparser::BinaryReaderError;

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

impl Analysis<max_stack::NoConfig, gas::NoConfig> {
    pub fn new() -> Self {
        Self {
            max_stack_cfg: max_stack::NoConfig,
            gas_cfg: gas::NoConfig,
        }
    }
}

impl<SC, GC> Analysis<SC, GC> {
    /// Configure the stack analysis.
    ///
    /// You most likely want to pass in a type that implements the [`max_stack::SizeConfig`] trait.
    /// This can be either by value, by reference or as a dynamic object of some sort.
    pub fn with_stack<NewSC>(self, max_stack_cfg: NewSC) -> Analysis<NewSC, GC> {
        let Self { gas_cfg, .. } = self;
        Analysis {
            max_stack_cfg,
            gas_cfg,
        }
    }

    /// Configure the gas analysis.
    ///
    /// You most likely want to pass in a type that implements the [`wasmparser::VisitOperator`]
    /// trait. This can be either by value, by reference or as a dynamic object of some sort.
    /// Though do keep in mind, that using a dynamic object may incur a significant performance
    /// penality, as the configuration provided here is accessed for each instruction in the
    /// analyzed module.
    ///
    /// For more information see [`gas::Config`].
    pub fn with_gas<NewGC>(self, gas_cfg: NewGC) -> Analysis<SC, NewGC> {
        let Self { max_stack_cfg, .. } = self;
        Analysis {
            max_stack_cfg,
            gas_cfg,
        }
    }
}

impl<'b, SC: max_stack::Config<'b>, GC: gas::Config<'b>> Analysis<SC, GC> {
    /// Execute the analysis on the provided module.
    pub fn analyze(&mut self, module: &'b [u8]) -> Result<AnalysisOutcome, Error> {
        let mut current_fn_id = 0u32;
        let mut function_frame_sizes = vec![];
        let mut function_operand_stack_sizes = vec![];
        let mut gas_costs = vec![];
        let mut gas_kinds = vec![];
        let mut gas_offsets = vec![];
        // Reused between functions for speeds.
        let mut gas_state = gas::FunctionState::new();
        let mut stack_state = max_stack::FunctionState::new();
        let mut module_state = max_stack::ModuleState::new();

        let parser = wasmparser::Parser::new(0);
        for payload in parser.parse_all(module) {
            let payload = payload.map_err(Error::ParsePayload)?;
            match payload {
                wasmparser::Payload::ImportSection(reader) => {
                    for import in reader.into_iter() {
                        let import = import.map_err(Error::ParseImports)?;
                        match import.ty {
                            wasmparser::TypeRef::Func(f) => {
                                module_state.functions.push(f);
                                current_fn_id = current_fn_id
                                    .checked_add(1)
                                    .ok_or(Error::TooManyFunctions)?;
                            }
                            wasmparser::TypeRef::Global(g) => {
                                module_state.globals.push(g.content_type);
                            }
                            wasmparser::TypeRef::Table(t) => {
                                module_state.tables.push(t.element_type);
                            }
                            wasmparser::TypeRef::Memory(_) => continue,
                            wasmparser::TypeRef::Tag(_) => continue,
                        }
                    }
                }
                wasmparser::Payload::TypeSection(reader) => {
                    for ty in reader {
                        let ty = ty.map_err(Error::ParseTypes)?;
                        module_state.types.push(ty);
                    }
                }
                wasmparser::Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global.map_err(Error::ParseGlobals)?;
                        module_state.globals.push(global.ty.content_type);
                    }
                }
                wasmparser::Payload::TableSection(reader) => {
                    for tbl in reader.into_iter() {
                        let tbl = tbl.map_err(Error::ParseTable)?;
                        module_state.tables.push(tbl.element_type);
                    }
                }
                wasmparser::Payload::FunctionSection(reader) => {
                    for function in reader {
                        let function = function.map_err(Error::ParseFunctions)?;
                        module_state.functions.push(function);
                    }
                }
                wasmparser::Payload::CodeSectionEntry(function) => {
                    stack_state.clear();
                    let function_id_usize = usize::try_from(current_fn_id)
                        .expect("failed converting from u32 to usize");
                    let type_id = *module_state
                        .functions
                        .get(function_id_usize)
                        .ok_or(Error::FunctionIndex(current_fn_id))?;
                    let type_id_usize =
                        usize::try_from(type_id).map_err(|e| Error::TypeIndexRange(type_id, e))?;
                    let fn_type = module_state
                        .types
                        .get(type_id_usize)
                        .ok_or(Error::TypeIndex(type_id))?;

                    match fn_type {
                        wasmparser::Type::Func(fnty) => {
                            for param in fnty.params() {
                                stack_state
                                    .add_locals(1, *param)
                                    .map_err(|e| Error::TooManyLocals(e, current_fn_id))?;
                            }
                        }
                    }
                    for local in function.get_locals_reader().map_err(Error::LocalsReader)? {
                        let local = local.map_err(Error::ParseLocals)?;
                        stack_state
                            .add_locals(local.0, local.1)
                            .map_err(|e| Error::TooManyLocals(e, current_fn_id))?;
                    }

                    // Visit the function body.
                    let mut combined_visitor = visitors::JoinVisitor(
                        self.gas_cfg.to_visitor(&mut gas_state),
                        self.max_stack_cfg
                            .to_visitor(&module_state, &mut stack_state),
                    );
                    let mut operators = function
                        .get_operators_reader()
                        .map_err(Error::OperatorReader)?;
                    while !operators.eof() {
                        combined_visitor.set_offset(operators.original_position());
                        let (gas_result, stack_result) = operators
                            .visit_operator(&mut combined_visitor)
                            .map_err(Error::VisitOperators)?;
                        let () = gas_result.map_err(Error::Gas)?;
                        let () = stack_result.map_err(Error::MaxStack)?;
                    }
                    drop(combined_visitor);

                    function_frame_sizes.push(self.max_stack_cfg.frame_size(&stack_state));
                    function_operand_stack_sizes.push(stack_state.max_size);
                    gas_state.optimize();
                    let (offsets, costs, kinds) = gas_state.drain();
                    // FIXME(nagisa): this may have a much better representation… We might be
                    // able to avoid storing all the analysis results if we instrumented
                    // on-the-fly too...
                    gas_offsets.push(offsets);
                    gas_kinds.push(kinds);
                    gas_costs.push(costs);
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

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) struct SizeConfig {
        pub(crate) value_size: u8,
        pub(crate) local_size: u8,
    }

    impl<'a> crate::max_stack::SizeConfig for SizeConfig {
        fn size_of_value(&self, _: wasmparser::ValType) -> u8 {
            self.value_size
        }

        fn size_of_function_activation(
            &self,
            locals: &prefix_sum_vec::PrefixSumVec<wasmparser::ValType, u32>,
        ) -> u64 {
            let locals = locals.max_index().map(|&v| v + 1).unwrap_or(0);
            u64::from(locals) * u64::from(self.local_size)
        }
    }

    impl Default for SizeConfig {
        fn default() -> Self {
            SizeConfig {
                value_size: 9,
                local_size: 5,
            }
        }
    }

    macro_rules! define_fee {
        ($(@$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
            $(
                fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output { 1 }
            )*
        }
    }

    struct GasConfig;
    impl<'a> wasmparser::VisitOperator<'a> for GasConfig {
        type Output = u64;
        wasmparser::for_each_operator!(define_fee);
    }

    #[test]
    fn dynamic_dispatch_is_possible() {
        let dynamic_size_config = SizeConfig::default();
        let mut dynamic_gas_config = GasConfig;

        let _ = crate::Analysis::new()
            .with_stack(&dynamic_size_config as &dyn crate::max_stack::SizeConfig)
            .analyze(b"");

        let _ = crate::Analysis::new()
            .with_stack(Box::new(dynamic_size_config) as Box<dyn crate::max_stack::SizeConfig>)
            .analyze(b"");

        let _ = crate::Analysis::new()
            .with_gas(&mut dynamic_gas_config)
            .analyze(b"");

        let _ = crate::Analysis::new()
            .with_gas(&mut dynamic_gas_config as &mut dyn wasmparser::VisitOperator<Output = u64>)
            .analyze(b"");

        let _ = crate::Analysis::new()
            .with_gas(
                Box::new(dynamic_gas_config) as Box<dyn wasmparser::VisitOperator<Output = u64>>
            )
            .analyze(b"");
    }
}
