use finite_wasm::{PartialSumMap, instrument::AnalysisConfig};

struct Config;

impl AnalysisConfig for Config {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u64 {
        match ty {
            wasmparser::ValType::I32 => 4,
            wasmparser::ValType::I64 => 8,
            wasmparser::ValType::F32 => 4,
            wasmparser::ValType::F64 => 8,
            wasmparser::ValType::V128 => 16,
            wasmparser::ValType::FuncRef => 8,
            wasmparser::ValType::ExternRef => 8,
        }
    }
    fn size_of_label(&self) -> u64 {
        0
    }

    fn size_of_function_activation(&self, locals: &PartialSumMap<u32, wasmparser::ValType>) -> u64 {
        locals.into_iter()
            .map(|(cnt, l)| u64::from(*cnt) * self.size_of_value(*l))
            .sum()
    }
}

fn main() {
    let input = std::fs::read(std::env::args().nth(1).unwrap()).unwrap();
    let module = finite_wasm::instrument::Module::new(&input, &Config).unwrap();
    // dbg!(module.function_stack_sizes);
}
