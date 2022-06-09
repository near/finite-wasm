use std::error;
use std::ffi::OsStr;
use std::io::{self, Write};
use std::path::PathBuf;

#[path = "test.rs"]
mod test;

#[derive(thiserror::Error, Debug)]
enum Error {
    #[error("could not obtiain the current working directory")]
    CurrentDirectory(#[source] std::io::Error),
    #[error("could not walk the `tests` directory")]
    WalkDirEntry(#[source] walkdir::Error),
    #[error("could not write the test output to the standard error")]
    WriteTestOutput(#[source] std::io::Error),
    #[error("some tests failed")]
    TestsFailed,
}

struct Test {
    contents: String,
    path: PathBuf,
}

fn write_error(mut to: impl io::Write, error: &impl error::Error) -> std::io::Result<()> {
    writeln!(to, "error: {}", error)?;
    let mut source = error.source();
    while let Some(error) = source {
        writeln!(to, "caused by: {}", error)?;
        source = error.source();
    }
    Ok(())
}

struct TestContext<WastErrorBuilder> {
    test_name: String,
    test_path: PathBuf,
    output: Vec<u8>,
    failed: bool,
    error_builder: WastErrorBuilder,
}

impl<WastErrorBuilder> TestContext<WastErrorBuilder>
where
    WastErrorBuilder: Fn(wast::token::Span, String) -> wast::Error,
{
    fn new(test_name: String, test_path: PathBuf, error_builder: WastErrorBuilder) -> Self {
        Self {
            test_name,
            test_path,
            output: vec![],
            failed: false,
            error_builder,
        }
    }

    fn pass(&mut self) {
        self.output.extend_from_slice(b"[PASS] ");
        self.output.extend_from_slice(self.test_name.as_bytes());
        self.output.extend_from_slice(b"\n");
    }

    fn fail(&mut self, error: &impl error::Error) {
        self.output.extend_from_slice(b"[FAIL] ");
        self.output.extend_from_slice(self.test_name.as_bytes());
        self.output.extend_from_slice(b"\n");
        write_error(&mut self.output, error).expect("this should be infallible");
        self.failed = true;
    }

    fn fail_wast(&mut self, error: &mut wast::Error) {
        error.set_path(&self.test_path);
        self.fail(error)
    }

    fn fail_test_error(&mut self, error: &mut test::Error) {
        error.set_path(&self.test_path);
        self.fail(error)
    }

    fn fail_wast_msg(&mut self, span: wast::token::Span, message: impl Into<String>) {
        let mut error = (self.error_builder)(span, message.into());
        self.fail_wast(&mut error)
    }

    fn run(&mut self, parsed_waft: Result<&mut test::Waft, &mut test::Error>) {
        let test = match parsed_waft {
            Ok(waft) => waft,
            Err(error) => return self.fail_test_error(error),
        };

        // First, collect and encode to wasm binary encoding all the input modules.
        let mut modules = Vec::new();
        for directive in &mut test.directives {
            match directive {
                test::WaftDirective::Module(module) => match module.encode() {
                    Ok(encoded) => modules.push((module.id, encoded)),
                    Err(error) => return self.fail(&error),
                },
                test::WaftDirective::AssertInstrumentedWat { .. }
                | test::WaftDirective::AssertStackWat { .. }
                | test::WaftDirective::AssertGasWat { .. } => {}
            }
        }

        // Now, execute all the assertions
        for directive in &test.directives {
            let index = match directive {
                test::WaftDirective::Module(_) => continue,
                test::WaftDirective::AssertGasWat { index, .. } => index,
                test::WaftDirective::AssertStackWat { index, .. } => index,
                test::WaftDirective::AssertInstrumentedWat { index, .. } => index,
            };
            let module = match *index {
                None => modules.get(0).ok_or_else(|| {
                    let span = wast::token::Span::from_offset(0);
                    self.fail_wast_msg(span, "this file defines no input modules")
                }),
                Some(wast::token::Index::Num(num, span)) => {
                    modules.get(num as usize).ok_or_else(|| {
                        self.fail_wast_msg(span, format!("module {} is not defined", num))
                    })
                }
                Some(wast::token::Index::Id(id)) => {
                    modules.iter().find(|m| m.0 == Some(id)).ok_or_else(|| {
                        self.fail_wast_msg(
                            id.span(),
                            format!("module {} is not defined", id.name()),
                        )
                    })
                }
            }
            .map(|m| &m.1);
            let _module = match module {
                Ok(m) => m,
                Err(()) => return,
            };

            match directive {
                test::WaftDirective::Module(_) => continue,
                test::WaftDirective::AssertGasWat {
                    expected_result, ..
                } => {
                    drop(expected_result);
                }
                test::WaftDirective::AssertStackWat {
                    expected_result, ..
                } => {
                    drop(expected_result);
                }
                test::WaftDirective::AssertInstrumentedWat {
                    expected_result, ..
                } => {
                    drop(expected_result);
                }
            };
        }

        return self.pass();
    }
}

fn run() -> Result<(), Error> {
    let current_directory = std::env::current_dir().map_err(Error::CurrentDirectory)?;
    let tests_directory = current_directory.join("tests");
    let mut tests = Vec::new();
    for entry in walkdir::WalkDir::new(&tests_directory) {
        let entry = entry.map_err(Error::WalkDirEntry)?;
        if Some(OsStr::new("wast")) == entry.path().extension() {
            tests.push(Test {
                contents: String::new(),
                path: entry.path().into(),
            });
        }
    }

    let mut lexed_tests = Vec::with_capacity(tests.len());
    for test in &mut tests {
        let result = test::read(&mut test.contents, &test.path);
        lexed_tests.push((result.and_then(|_| test::lex(&test.contents)), &*test));
    }

    // Tricky: the `wast` library requires that both the `ParseBuffer` and the original input
    // string are live before and while the test is being handled. We want all the test cases in a
    // list for parallelization and reporting purposes, and that's why we need three distinct
    // vectors that are built separately and awkardly like that.
    let mut parsed_tests = Vec::with_capacity(lexed_tests.len());
    for (lex, test) in &mut lexed_tests {
        parsed_tests.push((lex.as_mut().map(|lex| test::parse(lex)), *test));
    }

    println!("running {} tests", parsed_tests.len());
    let mut failures = 0;
    for (parsed, test) in &mut parsed_tests {
        let test_path = test
            .path
            .strip_prefix(&current_directory)
            .unwrap_or(&test.path);
        let test_name = test
            .path
            .strip_prefix(&tests_directory)
            .unwrap_or(&test.path);
        let parse_result = match parsed.as_mut() {
            Err(e) => Err(&mut **e),
            Ok(Err(e)) => Err(&mut *e),
            Ok(Ok(m)) => Ok(m),
        };

        let mut context = TestContext::new(
            test_name.display().to_string(),
            test_path.into(),
            |span, message| {
                let mut error = wast::Error::new(span, message);
                error.set_text(&test.contents);
                error
            },
        );
        context.run(parse_result);
        if context.failed {
            failures += 1;
        }
        std::io::stderr().lock().write_all(&context.output).map_err(Error::WriteTestOutput)?;
    }

    if failures != 0 {
        Err(Error::TestsFailed)
    } else {
        Ok(())
    }
}

// Custom test harness
#[cfg(test)]
pub(crate) fn main() {
    std::process::exit(match run() {
        Ok(()) => 0,
        Err(error) => {
            write_error(std::io::stderr().lock(), &error).expect("failed writing out the error");
            1
        }
    })
}
