#![no_main]

use libfuzzer_sys::fuzz_target;
use std::fmt::Write;
use std::path::PathBuf;

fuzz_target!(|data: wasm_smith::Module| {
    let bytes = data.to_bytes();

    let mut exports = Vec::new();
    for s in wasmparser::Parser::new(0).parse_all(&bytes) {
        let s = s.expect("wasm-smith-generated module is invalid");
        if let wasmparser::Payload::ExportSection(s) = s {
            for e in s.into_iter() {
                let e = e.expect("wasm-smith-generated module is invalid");
                if e.kind == wasmparser::ExternalKind::Func {
                    exports.push(e.name.to_owned());
                }
            }
        }
    }

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
