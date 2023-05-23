#![no_main]

use libfuzzer_sys::fuzz_target;
use std::fmt::Write;
use std::path::PathBuf;

/// Finds all no-parameter exported functions
pub fn find_entry_points(contract: &[u8]) -> Vec<String> {
    let mut tys = Vec::new();
    let mut fns = Vec::new();
    let mut entries = Vec::new();
    for payload in wasmparser::Parser::default().parse_all(contract) {
        match payload {
            Ok(wasmparser::Payload::FunctionSection(rdr)) => fns.extend(rdr),
            Ok(wasmparser::Payload::TypeSection(rdr)) => tys.extend(rdr),
            Ok(wasmparser::Payload::ExportSection(rdr)) => {
                for export in rdr {
                    if let Ok(wasmparser::Export { name, kind: wasmparser::ExternalKind::Func, index }) = export {
                        if let Some(&Ok(ty_index)) = fns.get(index as usize) {
                            if let Some(Ok(wasmparser::Type::Func(func_type))) = tys.get(ty_index as usize) {
                                if func_type.params().is_empty() {
                                    entries.push(name.to_string());
                                }
                            }
                        }
                    }
                }
            }
            _ => (),
        }
    }
    entries
}

#[derive(Debug, arbitrary::Arbitrary)]
struct ModuleConfig;
impl wasm_smith::Config for ModuleConfig {
    fn max_imports(&self) -> usize {
        0
    }
    fn max_instructions(&self) -> usize {
        1000
    }
    fn allow_start_export(&self) -> bool {
        false // traps in start are not caught properly
    }
}

fuzz_target!(|data: wasm_smith::ConfiguredModule<ModuleConfig>| {
    let bytes = data.module.to_bytes();
    let exports = find_entry_points(&bytes);

    if exports.is_empty() { // TODO: try removing once tests usually pass
        return;
    }

    let mut wast_test = String::new();
    write!(wast_test, "(module binary \"").unwrap();
    for b in bytes.iter() {
        write!(wast_test, "\\{:02X}", b).unwrap();
    }
    write!(wast_test, "\")\n").unwrap();
    for e in exports {
        write!(wast_test, "(invoke {:?})\n", e).unwrap();
    }

    let mut f = tempfile::Builder::new()
        .prefix("fuzzed-module-")
        .tempfile()
        .expect("creating temp file");
    std::io::Write::write_all(&mut f, wast_test.as_bytes()).expect("writing wast test file");

    let mut t = finite_wasm::wast_tests::test::TestContext::new(
        "fuzz".to_owned(),
        f.path().to_owned(),
        PathBuf::from("/"),
        PathBuf::from("/tmp"),
    );
    t.run();
    if t.failed() {
        let _ = std::fs::write("/tmp/fuzz-crash.wasm", &bytes);
        panic!(
            "Test failed, module available in /tmp/fuzz-crash.wasm.
Module bytes were:
{:?}
Wast test was:
{}
Test output:
{}",
            bytes,
            wast_test,
            String::from_utf8_lossy(&t.output),
        );
    }
});
