#![no_main]
use finite_wasm::{max_stack, prefix_sum_vec, wasmparser};
use libfuzzer_sys::fuzz_target;

struct DefaultStackConfig;
impl max_stack::SizeConfig for DefaultStackConfig {
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

fuzz_target!(|data: &[u8]| {
    // First, try to validate the data.
    let is_valid = wasmparser::validate(data).is_ok();
    let analysis_results = finite_wasm::Analysis::new()
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
    match analysis_results.instrument("spectest", data) {
        // If the original input was valid, we want the instrumented module to be valid too!
        Ok(res) if is_valid => {
            if let Err(e) = wasmparser::validate(&res) {
                let _ = std::fs::write("/tmp/input.wasm", data);
                let _ = std::fs::write("/tmp/instrumented.wasm", res);
                panic!(
                    "valid module after instrumentation is no longer valid: {:?}",
                    e
                );
            }
        }
        // Otherwise we're happy that things did not explode :)
        Ok(_) => return,
        Err(e) if is_valid => {
            let _ = std::fs::write("/tmp/input.wasm", data);
            panic!("valid module didn't instrument successfully: {:?}!", e)
        }
        Err(_) => return,
    };
});
