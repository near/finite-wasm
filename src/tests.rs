use std::ffi::OsStr;
use std::path::PathBuf;

#[path = "test.rs"]
mod test;

#[derive(thiserror::Error, Debug)]
enum Error {
    #[error("could not obtiain the current working directory")]
    CurrentDirectory(#[source] std::io::Error),
    #[error("could not walk the `tests` directory")]
    WalkDirEntry(#[source] walkdir::Error),
    #[error("some tests failed")]
    TestsFailed,
}

struct Test {
    contents: String,
    path: PathBuf,
}

fn print_error(error: &impl std::error::Error) {
    eprintln!("error: {}", error);
    let mut source = error.source();
    while let Some(error) = source {
        eprintln!("caused by: {}", error);
        source = error.source();
    }
}

fn run_test(
    test_desc: &impl std::fmt::Display,
    parsed_waft: Result<&mut test::Waft, &test::Error>,
    error_builder: &impl std::ops::Fn(wast::token::Span, String) -> wast::Error,
) -> bool {
    let test = match parsed_waft {
        Ok(waft) => waft,
        Err(error) => {
            eprintln!("[PARSE] {}", test_desc);
            print_error(&error);
            return false;
        }
    };

    // First, collect and encode to wasm binary encoding all the input modules.
    let mut modules = Vec::new();
    for directive in &mut test.directives {
        match directive {
            test::WaftDirective::Module(module) => match module.encode() {
                Ok(encoded) => modules.push((module.id, encoded)),
                Err(error) => {
                    eprintln!("[ERROR] {}", test_desc);
                    print_error(&error);
                    return false;
                }
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
                error_builder(span, "this file defines no input modules".into())
            }),
            Some(wast::token::Index::Num(num, span)) => modules
                .get(num as usize)
                .ok_or_else(|| error_builder(span, format!("module {} is not defined", num))),
            Some(wast::token::Index::Id(id)) => {
                modules.iter().find(|m| m.0 == Some(id)).ok_or_else(|| {
                    error_builder(id.span(), format!("module {} is not defined", id.name()))
                })
            }
        }
        .map(|m| &m.1);
        let _module = match module {
            Ok(m) => m,
            Err(error) => {
                eprintln!("[ERROR] {}", test_desc);
                print_error(&error);
                return false;
            }
        };

        match directive {
            test::WaftDirective::Module(_) => continue,
            test::WaftDirective::AssertGasWat {
                expected_result: _, ..
            } => {
                todo!()
            }
            test::WaftDirective::AssertStackWat {
                expected_result: _, ..
            } => {
                todo!()
            }
            test::WaftDirective::AssertInstrumentedWat {
                expected_result: _, ..
            } => {
                todo!()
            }
        };
    }

    eprintln!("[ PASS] {}", test_desc);
    true
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
            Err(e) => {
                e.set_path(test_path);
                Err(&**e)
            }
            Ok(Err(e)) => {
                e.set_path(test_path);
                Err(&*e)
            }
            Ok(Ok(m)) => Ok(m),
        };

        let error_builder = |span, message| {
            let mut error = wast::Error::new(span, message);
            error.set_text(&test.contents);
            error.set_path(
                test.path
                    .strip_prefix(&current_directory)
                    .unwrap_or(&test.path),
            );
            error
        };
        if !run_test(&test_name.display(), parse_result, &error_builder) {
            failures += 1;
        }
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
            print_error(&error);
            1
        }
    })
}
