#![no_main]
use finite_wasm::{max_stack, prefix_sum_vec, wasmparser};
use libfuzzer_sys::fuzz_target;

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
            u64::MAX / 128 // allow for 128 operations before an overflow would occur.
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

fuzz_target!(|module: wasm_smith::MaybeInvalidModule| {
    let data = module.to_bytes();
    let features = wasmparser::WasmFeatures {
        exceptions: false,
        ..Default::default()
    };
    let is_valid = wasmparser::Validator::new_with_features(features)
        .validate_all(&data)
        .is_ok();
    let analysis_results = finite_wasm::Analysis::new()
        .with_stack(DefaultStackConfig)
        .with_gas(DefaultGasConfig)
        .analyze(&data);
    let analysis_results = match analysis_results {
        Ok(res) => res,
        Err(e) if is_valid => {
            let _ = std::fs::write("/tmp/input.wasm", &data);
            panic!("valid module didn't analyze successfully: {:?}!", e)
        }
        Err(_) => return,
    };
    match analysis_results.instrument("spectest", &data) {
        // If the original input was valid, we want the instrumented module to be valid too!
        Ok(res) => {
            match (
                wasmparser::Validator::new_with_features(features)
                    .validate_all(&res)
                    .is_ok(),
                is_valid,
            ) {
                (true, true) | (false, false) => return,
                (result, _) => {
                    let _ = std::fs::write("/tmp/input.wasm", &data);
                    let _ = std::fs::write("/tmp/instrumented.wasm", res);
                    panic!(
                        "validity changed post-instrumentation, is_valid: {:?}, result: {:?}",
                        is_valid, result
                    );
                }
            }
        }
        Err(e) if is_valid => {
            let _ = std::fs::write("/tmp/input.wasm", data);
            panic!("valid module didn't instrument successfully: {:?}!", e)
        }
        Err(_) => return,
    };
});
