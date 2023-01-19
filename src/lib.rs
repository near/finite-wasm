#![doc = include_str!("../README.mkd")]
#![cfg_attr(finite_wasm_docs, feature(doc_auto_cfg))]

use gas::InstrumentationKind;
#[cfg(feature = "instrument")]
pub use instrument::Error as InstrumentError;
/// A re-export of the prefix_sum_vec crate. Use in implementing [`max_stack::SizeConfig`].
pub use prefix_sum_vec;
use visitors::VisitOperatorWithOffset;
/// A re-export of the wasmparser crate. Use in implementing [`max_stack::SizeConfig`] and
/// [`gas::Config`].
pub use wasmparser;
use wasmparser::BinaryReaderError;

pub mod gas;
mod instruction_categories;
#[cfg(feature = "instrument")]
mod instrument;
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
    #[error("could not process locals")]
    ProcessLocals(#[source] max_stack::Error),

    #[error("could not process operator for max_stack analysis")]
    MaxStack(#[source] max_stack::Error),
    #[error("could not process operator for gas analysis")]
    Gas(#[source] gas::Error),
}

/// No config is provided for the analysis, meaning the specific analysis will not run.
pub struct NoConfig;

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

impl Analysis<NoConfig, NoConfig> {
    pub fn new() -> Self {
        Self {
            max_stack_cfg: NoConfig,
            gas_cfg: NoConfig,
        }
    }
}

impl<StackConfig, GasCostModel> Analysis<StackConfig, GasCostModel> {
    /// Configure the stack analysis.
    ///
    /// You most likely want to pass in a type that implements the [`max_stack::SizeConfig`] trait.
    /// This can be either by value, by reference or as a dynamic object of some sort.
    pub fn with_stack<NewSC>(self, max_stack_cfg: NewSC) -> Analysis<NewSC, GasCostModel> {
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
    pub fn with_gas<NewGC>(self, gas_cfg: NewGC) -> Analysis<StackConfig, NewGC> {
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
        let mut outcome = AnalysisOutcome {
            function_frame_sizes: vec![],
            function_operand_stack_sizes: vec![],
            gas_offsets: vec![],
            gas_costs: vec![],
            gas_kinds: vec![],
        };
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
                                self.max_stack_cfg.add_function(&mut module_state, f);
                                current_fn_id = current_fn_id
                                    .checked_add(1)
                                    .ok_or(Error::TooManyFunctions)?;
                            }
                            wasmparser::TypeRef::Global(g) => {
                                self.max_stack_cfg
                                    .add_global(&mut module_state, g.content_type);
                            }
                            wasmparser::TypeRef::Table(t) => {
                                self.max_stack_cfg
                                    .add_table(&mut module_state, t.element_type);
                            }
                            wasmparser::TypeRef::Memory(_) => continue,
                            wasmparser::TypeRef::Tag(_) => continue,
                        }
                    }
                }
                wasmparser::Payload::TypeSection(reader) => {
                    for ty in reader {
                        let ty = ty.map_err(Error::ParseTypes)?;
                        self.max_stack_cfg.add_type(&mut module_state, ty);
                    }
                }
                wasmparser::Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global.map_err(Error::ParseGlobals)?;
                        self.max_stack_cfg
                            .add_global(&mut module_state, global.ty.content_type);
                    }
                }
                wasmparser::Payload::TableSection(reader) => {
                    for tbl in reader.into_iter() {
                        let tbl = tbl.map_err(Error::ParseTable)?;
                        self.max_stack_cfg
                            .add_table(&mut module_state, tbl.element_type);
                    }
                }
                wasmparser::Payload::FunctionSection(reader) => {
                    for function in reader {
                        let function = function.map_err(Error::ParseFunctions)?;
                        self.max_stack_cfg.add_function(&mut module_state, function);
                    }
                }
                wasmparser::Payload::CodeSectionEntry(function) => {
                    self.max_stack_cfg
                        .populate_locals(&module_state, &mut stack_state, current_fn_id)
                        .map_err(Error::ProcessLocals)?;
                    for local in function.get_locals_reader().map_err(Error::LocalsReader)? {
                        let local = local.map_err(Error::ParseLocals)?;
                        stack_state
                            .add_locals(local.0, local.1)
                            .map_err(Error::ProcessLocals)?;
                    }

                    // Visit the function body.
                    let mut combined_visitor = visitors::JoinVisitor(
                        self.gas_cfg.make_visitor(&mut gas_state),
                        self.max_stack_cfg
                            .make_visitor(&module_state, &mut stack_state),
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

                    self.max_stack_cfg
                        .save_outcomes(&mut stack_state, &mut outcome);
                    self.gas_cfg.save_outcomes(&mut gas_state, &mut outcome);
                    current_fn_id = current_fn_id
                        .checked_add(1)
                        .ok_or(Error::TooManyFunctions)?;
                }
                _ => (),
            }
        }
        Ok(outcome)
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

    /// The table of offsets for gas instrumentation points.
    ///
    /// This vector is indexed by entries in the code section (that is, it is indexed by the
    /// function index, *excluding* imports).
    pub gas_offsets: Vec<Box<[usize]>>,
    /// The table of gas costs for gas instrumentation points.
    ///
    /// This vector is indexed by entries in the code section (that is, it is indexed by the
    /// function index, *excluding* imports).
    pub gas_costs: Vec<Box<[u64]>>,
    /// The table of instrumentation kinds for gas instrumentation points.
    ///
    /// This vector is indexed by entries in the code section (that is, it is indexed by the
    /// function index, *excluding* imports).
    pub gas_kinds: Vec<Box<[InstrumentationKind]>>,
}

impl AnalysisOutcome {
    /// Modify the provided `wasm` module to enforce gas and stack limits.
    ///
    /// The instrumentation approach provided by this crate has been largely tailored for this
    /// crate’s own testing needs and may not be applicable to every use-case. However the code is
    /// reasonably high quality that it might be useful for development purposes.
    ///
    /// This function will modify the provided core wasm module to introduce two imports:
    ///
    /// * `{env}.finite_wasm_gas`: `(func (params i64))
    /// * `{env}.finite_wasm_stack`: `(func (params i64 i64))
    ///
    /// These functions must be provided by the embedder. The `finite_wasm_gas` should reduce the
    /// pool of remaining gas by the only argument supplied and trap the execution when the gas is
    /// exhausted. When the gas is exhausted the reamining gas pool must be set to 0, as per the
    /// specification.
    ///
    /// The `finite_wasm_stack` is called with two arguments. First is the size by which the
    /// operands stack increases. Second is the size of the stack reserved by the function frame.
    /// This host function must keep track of the current total stack height and raise a trap if
    /// the stack limit is exceeded.
    #[cfg(feature = "instrument")]
    pub fn instrument(&self, env: &str, wasm: &[u8]) -> Result<Vec<u8>, InstrumentError> {
        instrument::instrument(wasm, env, self)
    }
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
