fn instrument_finite_wasm(code: &[u8]) -> Result<Vec<u8>, ()> {
    let outcome = finite_wasm::Analysis::new()
        .with_stack(DefaultStackConfig)
        .with_gas(DefaultGasConfig)
        .analyze(code)
        .map_err(|_| ())?;
    outcome.instrument("env", code).map_err(|_| ())
}

fn instrument_wasm_instrument(code: &[u8]) -> Result<Vec<u8>, ()> {
    let parsed =
        wasm_instrument::parity_wasm::elements::Module::from_bytes(code).map_err(|_| ())?;
    let injector = wasm_instrument::gas_metering::host_function::Injector::new("env", "gas");
    let rules = wasm_instrument::gas_metering::ConstantCostRules::new(1, 1, 1);
    let gassed = wasm_instrument::gas_metering::inject(parsed, injector, &rules).map_err(|_| ())?;
    let stack_limited = wasm_instrument::inject_stack_limiter(gassed, 1024).map_err(|_| ())?;
    stack_limited.into_bytes().map_err(|_| ())
}

fn all_tests(c: &mut criterion::Criterion) {
    let current_directory = std::env::current_dir().expect("get current_dir");
    let tests_directory = current_directory.join("tests");
    let temp_directory = tests_directory.join("tmp");
    let snaps_directory = tests_directory.join("snaps");
    let mut group = c.benchmark_group("tests");
    for entry in walkdir::WalkDir::new(&tests_directory) {
        let entry = entry.expect("walkdir");
        let entry_path = entry.path();
        if entry_path.starts_with(&temp_directory) {
            continue;
        }
        if entry_path.starts_with(&snaps_directory) {
            continue;
        }
        if Some(std::ffi::OsStr::new("wast")) != entry_path.extension() {
            continue;
        }
        let test_name = entry_path
            .strip_prefix(&tests_directory)
            .unwrap_or(&entry_path)
            .display()
            .to_string();

        let test_contents = std::fs::read_to_string(entry_path).expect("read the test");
        let mut lexer = wast::lexer::Lexer::new(&test_contents);
        lexer.allow_confusing_unicode(true);
        let buf = wast::parser::ParseBuffer::new_with_lexer(lexer).expect("parse buffer");
        let wast: wast::Wast = wast::parser::parse(&buf).expect("parse wast");
        let mut modules = vec![];
        for directive in wast.directives {
            use wast::{QuoteWat as QW, WastDirective as WD, WastExecute as WE};
            match directive {
                WD::Wat(QW::Wat(wat))
                | WD::AssertTrap {
                    exec: WE::Wat(wat), ..
                }
                | WD::AssertMalformed {
                    module: QW::Wat(wat),
                    ..
                }
                | WD::AssertInvalid {
                    module: QW::Wat(wat),
                    ..
                }
                | WD::AssertReturn {
                    exec: WE::Wat(wat), ..
                }
                | WD::AssertException {
                    exec: WE::Wat(wat), ..
                } => match wat {
                    wast::Wat::Module(mut module) => {
                        let module_bytes = module.encode().expect("encode module");
                        modules.push(module_bytes);
                    }
                    wast::Wat::Component(_) => todo!("components"),
                },
                _ => {}
            }
        }

        // We must filter the list of modules to those valid for both approaches. This makes sure
        // both of the crates are doing the same amount of work, and not running faster because
        // e.g. they just failed to process the module entirely.
        modules.retain(|m| {
            instrument_finite_wasm(&m).is_ok() && instrument_wasm_instrument(&m).is_ok()
        });

        if !modules.is_empty() {
            group.bench_with_input(
                criterion::BenchmarkId::new("finite_wasm", &test_name),
                &modules,
                |b, i| {
                    b.iter(|| {
                        i.iter()
                            .map(|m| instrument_finite_wasm(m))
                            .collect::<Vec<_>>()
                    })
                },
            );
            group.bench_with_input(
                criterion::BenchmarkId::new("wasm_instrument", &test_name),
                &modules,
                |b, i| {
                    b.iter(|| {
                        i.iter()
                            .map(|m| instrument_wasm_instrument(m))
                            .collect::<Vec<_>>()
                    })
                },
            );
        }
    }
}

criterion::criterion_group!(benches, all_tests);
criterion::criterion_main!(benches);

struct DefaultStackConfig;
impl finite_wasm::max_stack::SizeConfig for DefaultStackConfig {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        use wasmparser::ValType::*;
        match ty {
            I32 => 4,
            I64 => 8,
            F32 => 4,
            F64 => 8,
            V128 => 16,
            FuncRef => 32,
            ExternRef => 32,
        }
    }

    fn size_of_function_activation(
        &self,
        locals: &prefix_sum_vec::PrefixSumVec<wasmparser::ValType, u32>,
    ) -> u64 {
        u64::from(locals.max_index().map(|&v| v + 1).unwrap_or(0))
    }
}

pub(crate) struct DefaultGasConfig;

macro_rules! gas_visit {
    (visit_end => $({ $($arg:ident: $argty:ty),* })?) => {};
    (visit_else => $({ $($arg:ident: $argty:ty),* })?) => {};
    ($visit:ident => $({ $($arg:ident: $argty:ty),* })?) => {
        fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output {
            1u64
        }
    };

    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
        $(gas_visit!{ $visit => $({ $($arg: $argty),* })? })*
    }
}

impl<'a> wasmparser::VisitOperator<'a> for DefaultGasConfig {
    type Output = u64;
    fn visit_end(&mut self) -> u64 {
        0
    }
    fn visit_else(&mut self) -> u64 {
        0
    }

    wasmparser::for_each_operator!(gas_visit);
}
