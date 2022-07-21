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

fn write_error(mut to: impl io::Write, error: impl error::Error) -> std::io::Result<()> {
    writeln!(to, "error: {}", error)?;
    let mut source = error.source();
    while let Some(error) = source {
        writeln!(to, "caused by: {}", error)?;
        source = error.source();
    }
    Ok(())
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

        let mut context = test::TestContext::new(
            test_name.display().to_string(),
            test_path.into(),
            &test.contents,
        );
        context.run(parse_result);
        if context.failed() {
            failures += 1;
        }
        std::io::stderr()
            .lock()
            .write_all(&context.output)
            .map_err(Error::WriteTestOutput)?;
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
