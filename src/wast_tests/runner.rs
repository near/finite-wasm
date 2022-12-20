use std::error;
use std::ffi::OsStr;
use std::io::{self, Write};
use std::path::PathBuf;
use rayon::prelude::*;

mod test;
mod instrument;

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
    #[error("could not create a temporary directory for tests")]
    CreateTempDirectory(#[source] std::io::Error),
}

struct Test {
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
    let snaps_directory = tests_directory.join("snaps");
    let temp_directory = tests_directory.join("tmp");
    let mut tests = Vec::new();
    for entry in walkdir::WalkDir::new(&tests_directory) {
        let entry = entry.map_err(Error::WalkDirEntry)?;
        let entry_path = entry.path();
        if entry_path.starts_with(&temp_directory) {
            continue;
        }
        if Some(OsStr::new("wast")) == entry_path.extension() {
            tests.push(Test {
                path: entry.path().into(),
            });
        }
    }

    println!("running {} tests", tests.len());
    let mut failures = std::sync::atomic::AtomicU16::new(0);
    tests.par_iter_mut().try_for_each(|test| {
        let test_path = test
            .path
            .strip_prefix(&current_directory)
            .unwrap_or(&test.path);
        let test_name = test
            .path
            .strip_prefix(&tests_directory)
            .unwrap_or(&test.path);
        let mut context = test::TestContext::new(
            test_name.display().to_string(),
            test_path.into(),
            snaps_directory.clone(),
            temp_directory.clone(),
        );
        context.run();
        if context.failed() {
            failures.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        std::io::stderr()
            .lock()
            .write_all(&context.output)
            .map_err(Error::WriteTestOutput)?;
        Ok::<_, Error>(())
    })?;

    if failures.load(std::sync::atomic::Ordering::SeqCst) != 0 {
        Err(Error::TestsFailed)
    } else {
        Ok(())
    }
}

fn main() {
    std::process::exit(match run() {
        Ok(()) => 0,
        Err(error) => {
            write_error(std::io::stderr().lock(), &error).expect("failed writing out the error");
            1
        }
    })
}
