#![no_main]

use libfuzzer_sys::fuzz_target;
use std::fmt::Write;
use std::path::PathBuf;

fuzz_target!(|data: wasm_smith::Module| {
    let bytes = data.to_bytes();

    let module = wasmparser::validate(&bytes).expect("wasm-smith-generated module is invalid");
    if module.module_count() < 1 {
        return;
    }
    let exports = &module
        .module_at(0)
        .expect("wasm-smith-generated module does not have one module")
        .exports;

    let mut wast_test = String::new();
    write!(wast_test, "(module binary \"").unwrap();
    for b in bytes.iter() {
        write!(wast_test, "\\{:02}", b).unwrap();
    }
    write!(wast_test, "\")").unwrap();
    for e in exports {
        if matches!(e.1, wasmparser::types::EntityType::Func(_)) {
            write!(wast_test, "(invoke {:?})", e.0).unwrap();
        }
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
    assert!(
        !t.failed(),
        "Test failed.\nModule bytes were:\n{:?}\nWast test was:\n{}\n",
        bytes,
        wast_test,
    );
});
