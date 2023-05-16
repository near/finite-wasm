#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use std::path::PathBuf;

fuzz_target!(|data: wasm_smith::MaybeInvalidModule| {
    let mut f = tempfile::Builder::new()
        .prefix("fuzzed-module-")
        .tempfile()
        .expect("creating temp file");
    f.write_all(&data.to_bytes()).expect("writing module to file");
    let mut t = finite_wasm::wast_tests::test::TestContext::new(
        "fuzz".to_owned(),
        f.path().to_owned(),
        PathBuf::from("/"),
        PathBuf::from("/tmp"),
    );
    t.run();
    assert!(!t.failed(), "test failed. Module was:\n{:?}", data);
});
