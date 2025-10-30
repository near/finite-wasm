use finite_wasm::gas::InstrumentationKind;
use finite_wasm::{max_stack, prefix_sum_vec, wasmparser, AnalysisOutcome, Error, Fee};
use std::fs;
use std::path::PathBuf;

fn analyze(wasm: &[u8]) -> Result<AnalysisOutcome, Error> {
    struct DefaultStackConfig;

    impl max_stack::SizeConfig for DefaultStackConfig {
        fn size_of_value(&self, _ty: wasmparser::ValType) -> u8 {
            u8::MAX
        }

        fn size_of_function_activation(
            &self,
            _locals: &prefix_sum_vec::PrefixSumVec<wasmparser::ValType, u32>,
        ) -> u64 {
            u64::MAX / 128
        }
    }

    struct DefaultGasConfig;

    macro_rules! gas_visit {
    (visit_end => $({ $($arg:ident: $argty:ty),* })?) => {};
    (visit_else => $({ $($arg:ident: $argty:ty),* })?) => {};
    ($visit:ident => $({ $($arg:ident: $argty:ty),* })?) => {
        fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output {
            Fee::constant(u64::MAX / 128) // allow for 128 operations before an overflow would occur.
        }
    };

    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident ($($ann:tt)*))*) => {
        $(gas_visit!{ $visit => $({ $($arg: $argty),* })? })*
    }
}

    impl<'a> wasmparser::VisitOperator<'a> for DefaultGasConfig {
        type Output = Fee;
        fn visit_end(&mut self) -> Fee {
            Fee::ZERO
        }
        fn visit_else(&mut self) -> Fee {
            Fee::ZERO
        }
        wasmparser::for_each_visit_operator!(gas_visit);
    }

    impl<'a> wasmparser::VisitSimdOperator<'a> for DefaultGasConfig {
        wasmparser::for_each_visit_simd_operator!(gas_visit);
    }

    let wasm = wat::parse_bytes(wasm).expect("failed to parse WAT");
    finite_wasm::Analysis::new()
        .with_stack(DefaultStackConfig)
        .with_gas(DefaultGasConfig)
        .analyze(&wasm)
}

#[test]
fn analyze_component() {
    const FUNC: &str = r#"(func (export "main")
    (i32.div_u
      (i32.const 0)
      (f32.gt
        (f32.copysign
          (f32.const 1.0)
          (f32.sqrt (f32.const -1.0)))
        (f32.const 0)))
    drop
  )"#;

    assert_eq!(
        analyze(
            format!(
                r#"
(module
   {FUNC}
)"#
            )
            .as_bytes()
        )
        .unwrap(),
        AnalysisOutcome {
            function_frame_sizes: vec![144115188075855871],
            function_operand_stack_sizes: vec![765],
            gas_offsets: vec![[33, 54].into()],
            gas_costs: vec![[
                Fee {
                    constant: 1152921504606846968,
                    linear: 0
                },
                Fee {
                    constant: 144115188075855871,
                    linear: 0
                }
            ]
            .into()],
            gas_kinds: vec![[
                InstrumentationKind::PreControlFlow,
                InstrumentationKind::PostControlFlow
            ]
            .into()]
        }
    );

    assert_eq!(
        analyze(
            format!(
                r#"
(component
  (core module
    {FUNC}
  )
)"#
            )
            .as_bytes()
        )
        .unwrap(),
        AnalysisOutcome {
            function_frame_sizes: vec![144115188075855871],
            function_operand_stack_sizes: vec![765],
            gas_offsets: vec![[43, 64].into()],
            gas_costs: vec![[
                Fee {
                    constant: 1152921504606846968,
                    linear: 0
                },
                Fee {
                    constant: 144115188075855871,
                    linear: 0
                }
            ]
            .into()],
            gas_kinds: vec![[
                InstrumentationKind::PreControlFlow,
                InstrumentationKind::PostControlFlow
            ]
            .into()]
        }
    );
}

#[test]
fn analyze_spec() {
    let tests = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("component-model")
        .join("test");
    let tests = fs::read_dir(tests).expect("failed to read component model spec test directory");
    let tests = tests
        .filter_map(|entry| {
            let entry = entry.unwrap();
            let ty = entry.file_type().unwrap();
            if !ty.is_dir() {
                return None;
            }
            let d = fs::read_dir(entry.path())
                .expect("failed to read component model spec test subdirectory");
            Some(d.filter_map(|entry| {
                let entry = entry.unwrap();
                let ty = entry.file_type().unwrap();
                if !ty.is_file() {
                    return None;
                }
                let path = entry.path();
                let buf = fs::read(&path).expect("failed to read test");
                Some((path, buf))
            }))
        })
        .flatten();
    for (_, buf) in tests {
        let buf = str::from_utf8(&buf).unwrap();

        let mut lexer = wast::lexer::Lexer::new(buf);
        lexer.allow_confusing_unicode(true);
        let buf = wast::parser::ParseBuffer::new_with_lexer(lexer).unwrap();
        let wast: wast::Wast = wast::parser::parse(&buf).unwrap();
        for directive in wast.directives {
            match directive {
                wast::WastDirective::Module(wast::QuoteWat::Wat(wast::Wat::Module(..))) => {
                    unreachable!("doesn’t actually occur in our test suite");
                }
                wast::WastDirective::Module(wast::QuoteWat::QuoteModule(_, _)) => {
                    unreachable!("doesn’t actually occur in our test suite");
                }
                wast::WastDirective::Module(wast::QuoteWat::Wat(wast::Wat::Component(mut c)))
                | wast::WastDirective::ModuleDefinition(wast::QuoteWat::Wat(
                    wast::Wat::Component(mut c),
                )) => {
                    let c = c.encode().unwrap();
                    let _outcome = analyze(&c).unwrap();
                }
                wast::WastDirective::Module(wast::QuoteWat::QuoteComponent(_, _)) => {
                    // Same
                    continue;
                }
                wast::WastDirective::ModuleDefinition(_) => {
                    unreachable!("doesn’t actually occur in our test suite");
                }

                // Ignore the “operations”, we only care about module analysis results.
                wast::WastDirective::ModuleInstance { .. } => continue,
                wast::WastDirective::Register { .. } => continue,
                wast::WastDirective::Invoke(..) => continue,
                wast::WastDirective::AssertTrap { .. } => continue,
                wast::WastDirective::AssertReturn { .. } => continue,
                wast::WastDirective::AssertExhaustion { .. } => continue,
                wast::WastDirective::AssertException { .. } => continue,
                // Do not attempt to process invalid modules.
                wast::WastDirective::AssertMalformed { .. } => continue,
                wast::WastDirective::AssertInvalid { .. } => continue,
                wast::WastDirective::AssertUnlinkable { .. } => continue,
                wast::WastDirective::AssertSuspension { .. } => {
                    unreachable!("doesn’t actually occur in our test suite");
                }
                wast::WastDirective::Thread(_) => {
                    unreachable!("doesn’t actually occur in our test suite");
                }
                wast::WastDirective::Wait { .. } => {
                    unreachable!("doesn’t actually occur in our test suite");
                }
            };
        }
    }
}
