use crate::{max_stack, prefix_sum_vec, wasmparser, Fee};

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

pub(crate) struct DefaultGasConfig;

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
        Fee::constant(0)
    }
    fn visit_else(&mut self) -> Fee {
        Fee::constant(0)
    }
    wasmparser::for_each_visit_operator!(gas_visit);

    fn simd_visitor(
        &mut self,
    ) -> Option<&mut dyn wasmparser::VisitSimdOperator<'a, Output = Self::Output>> {
        Some(self)
    }
}

impl<'a> wasmparser::VisitSimdOperator<'a> for DefaultGasConfig {
    wasmparser::for_each_visit_simd_operator!(gas_visit);
}

#[derive(Debug)]
struct WasmSmithModule {
    data: Vec<u8>,
}

impl<'a> arbitrary::Arbitrary<'a> for WasmSmithModule {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut config = wasm_smith::Config::arbitrary(u)?;
        config.allow_invalid_funcs = true;
        config.exceptions_enabled = false;
        config.gc_enabled = false;
        wasm_smith::Module::new(config, u).map(|m| Self { data: m.to_bytes() })
    }
    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        wasm_smith::Module::size_hint(depth)
    }
}

#[test]
fn fuzz() {
    bolero::check!()
        .with_arbitrary::<WasmSmithModule>()
        .for_each(|module| {
            let data = &module.data;
            let features = wasmparser::WasmFeatures::default()
                | wasmparser::WasmFeatures::MUTABLE_GLOBAL
                | wasmparser::WasmFeatures::WIDE_ARITHMETIC
                & !wasmparser::WasmFeatures::EXCEPTIONS
                & !wasmparser::WasmFeatures::LEGACY_EXCEPTIONS;
            // First, try to validate the data.
            let is_valid = wasmparser::Validator::new_with_features(features)
                .validate_all(data)
                .is_ok();
            let analysis_results = crate::Analysis::new()
                .with_stack(DefaultStackConfig)
                .with_gas(DefaultGasConfig)
                .analyze(data);
            let analysis_results = match analysis_results {
                Ok(res) => res,
                Err(e) if is_valid => {
                    let _ = std::fs::write("/tmp/input.wasm", data);
                    panic!("valid module didn't analyze successfully: {:?}!", e)
                }
                Err(_) => return,
            };
            match analysis_results.instrument("spectest", data, 2, 100) {
                // If the original input was valid, we want the instrumented module to be valid too!
                Ok(res) if is_valid => {
                    if let Err(e) =
                        wasmparser::Validator::new_with_features(features).validate_all(&res)
                    {
                        let _ = std::fs::write("/tmp/input.wasm", data);
                        let _ = std::fs::write("/tmp/instrumented.wasm", res);
                        panic!(
                            "valid module is no longer valid post-instrumentation: {:?}",
                            e
                        );
                    }
                }
                // We're happy that things did not explode, but we also want to ensure that the module
                // remains invalid if it was invalid initially.
                Ok(_res) => {
                    // This unfortunately does not work right now. In particular we might have an input
                    // along the lines of:
                    //
                    // 0x0 | 00 61 73 6d | version 22 (Component)
                    //     | 16 00 01 00
                    //
                    // which becomes a
                    //
                    // 0x0 | 00 61 73 6d | version 1 (Module)
                    //     | 01 00 00 00
                    // 0x8 | 01 01       | type section
                    // 0xa | 00          | 0 count
                    // 0xb | 02 01       | import section
                    // 0xd | 00          | 0 count
                    //
                    // after the instrumentation. This is due to a few factors, one of which is that
                    // wasm_encoder does not allow us to directly write out the `Payload::Version`,
                    // unfortunately. At least as things are right now.
                    //
                    // if let Ok(_) = wasmparser::validate(&res) {
                    //     let _ = std::fs::write("/tmp/input.wasm", data);
                    //     let _ = std::fs::write("/tmp/instrumented.wasm", res);
                    //     panic!("invalid module after instrumentation has become valid");
                    // }
                }
                Err(e) if is_valid => {
                    let _ = std::fs::write("/tmp/input.wasm", data);
                    panic!("valid module didn't instrument successfully: {:?}!", e)
                }
                Err(_) => return,
            };
        });
}
