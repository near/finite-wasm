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
                    if let Ok(wasmparser::Export {
                        name,
                        kind: wasmparser::ExternalKind::Func,
                        index,
                    }) = export
                    {
                        if name.chars().any(|c| !c.is_ascii_alphanumeric()) {
                            continue; // ignore non-ascii-alnum exports for convenience
                        }
                        if let Some(&Ok(ty_index)) = fns.get(index as usize) {
                            if let Some(Ok(wasmparser::Type::Func(func_type))) =
                                tys.get(ty_index as usize)
                            {
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

fn adjust_cfg(mut config: wasm_smith::Config) -> wasm_smith::Config {
    config.max_imports = 0;
    config.max_instructions = 1000;
    config.allow_start_export = false;
    config
}

#[derive(Debug)]
struct WasmSmithModule {
    data: Vec<u8>,
}

impl<'a> arbitrary::Arbitrary<'a> for WasmSmithModule {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let config = adjust_cfg(wasm_smith::Config::arbitrary(u)?);
        wasm_smith::Module::new(config, u).map(|m| Self {
            data: m.to_bytes(),
        })
    }
    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        let cfg_size = wasm_smith::Config::size_hint(depth);
        let module_size = wasm_smith::Module::size_hint(depth);
        (cfg_size.0 + module_size.0, None)
    }
}

#[test]
fn fuzz() {
    bolero::check!()
        .with_arbitrary::<WasmSmithModule>()
        .for_each(|module| {
            let bytes = &module.data;
            let exports = find_entry_points(&bytes);

            if exports.is_empty() {
                // It’s pointless looking more into this test as we won’t be running anything anyway.
                return;
            }

            let mut wast_test = String::new();
            write!(wast_test, "(module binary \"").unwrap();
            for b in bytes.iter() {
                write!(wast_test, "\\{:02X}", b).unwrap();
            }
            write!(wast_test, "\")\n").unwrap();
            for e in exports {
                // `(invoke ...)` will be turned into `(just_run ...)` by the test framework after wast parsing.
                // XREF:INVOKE-FOR-JUST-RUN
                write!(wast_test, "(invoke {:?})\n", e).unwrap();
            }

            let mut f = tempfile::Builder::new()
                .prefix("fuzzed-module-")
                .tempfile()
                .expect("creating temp file");
            std::io::Write::write_all(&mut f, wast_test.as_bytes())
                .expect("writing wast test file");

            let mut t = crate::wast_tests::test::TestContext::new(
                "fuzz".to_owned(),
                f.path().to_owned(),
                PathBuf::from("/"),
                PathBuf::from("/tmp"),
                false,
            );
            t.run();
            if t.failed() {
                let _ = std::fs::write("/tmp/fuzz-crash.wasm", &bytes);
                let output = String::from_utf8_lossy(&t.output);
                eprintln!(
                    "Test failed, module available in /tmp/fuzz-crash.wasm.
Module bytes were:
{:?}
Wast test was:
{}
Test output:
{}",
                    bytes, wast_test, output,
                );
                panic!("{}", output.lines().collect::<Vec<_>>().join("; "));
            }
        })
}
