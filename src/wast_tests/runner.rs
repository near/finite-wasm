use finite_wasm::wast_tests::test;
use rayon::prelude::*;
use std::ffi::OsStr;
use std::io::Write;
use std::path::PathBuf;

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
    path: PathBuf,
}

fn run() -> Result<(), Error> {
    let current_directory = std::env::current_dir().map_err(Error::CurrentDirectory)?;
    let tests_directory = current_directory.join("tests");
    let snaps_directory = tests_directory.join("snaps");
    let temp_directory = tests_directory.join("tmp");
    let filter = std::env::args().nth(1);
    let mut tests = vec![Test {
        path: "!internal-self-test-interpreter".into(),
    }];
    for entry in walkdir::WalkDir::new(&tests_directory) {
        let entry = entry.map_err(Error::WalkDirEntry)?;
        let entry_path = entry.path();
        if entry_path.starts_with(&temp_directory) {
            continue;
        }
        let test_name = entry_path
            .strip_prefix(&tests_directory)
            .unwrap_or(&entry_path)
            .display()
            .to_string();
        if let Some(filter) = &filter {
            if !test_name.contains(filter) {
                continue;
            }
        }
        if Some(OsStr::new("wast")) == entry_path.extension() {
            tests.push(Test {
                path: entry_path.into(),
            });
        }
    }

    println!("running {} tests", tests.len());
    let failures = std::sync::atomic::AtomicUsize::new(0);
    tests.par_iter_mut().try_for_each(|test| {
        let test_path = test
            .path
            .strip_prefix(&current_directory)
            .unwrap_or(&test.path);
        let test_name = test
            .path
            .strip_prefix(&tests_directory)
            .unwrap_or(&test.path)
            .display()
            .to_string();
        let mut context = test::TestContext::new(
            test_name,
            test_path.into(),
            snaps_directory.clone(),
            temp_directory.clone(),
            true,
        );
        context.run();
        if context.failed() {
            failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        std::io::stderr()
            .lock()
            .write_all(&context.output)
            .map_err(Error::WriteTestOutput)?;
        Ok::<_, Error>(())
    })?;

    if failures.load(std::sync::atomic::Ordering::Relaxed) != 0 {
        Err(Error::TestsFailed)
    } else {
        Ok(())
    }
}

fn main() {
    std::process::exit(match run() {
        Ok(()) => 0,
        Err(error) => {
            test::write_error(std::io::stderr().lock(), &error)
                .expect("failed writing out the error");
            1
        }
    })
}
