use finite_wasm::{max_stack, Module};
use std::error;
use std::ffi::OsString;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

#[derive(Debug, PartialEq)]
enum Line {
    Equal(String),
    Delete(String),
    Insert(String),
}

#[derive(Debug)]
pub(crate) struct DiffError {
    diff: Vec<Line>,
    path: Option<PathBuf>,
}

pub enum InterpreterMode {
    GasTrace,
    StackTrace,
}

impl DiffError {
    fn diff(old: &str, new: &str) -> Option<Self> {
        if old == new {
            return None;
        }
        let result = dissimilar::diff(old, new);
        let diff = result
            .into_iter()
            .map(|s| match s {
                dissimilar::Chunk::Delete(s) => Line::Delete(s.into()),
                dissimilar::Chunk::Insert(s) => Line::Insert(s.into()),
                dissimilar::Chunk::Equal(s) => Line::Equal(s.into()),
            })
            .collect::<Vec<_>>();
        if diff.iter().all(|l| matches!(l, Line::Equal(_))) {
            None
        } else {
            Some(Self { diff, path: None })
        }
    }
    fn with_path(self, path: PathBuf) -> Self {
        Self {
            diff: self.diff,
            path: Some(path),
        }
    }
}
impl std::error::Error for DiffError {}
impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(path) = &self.path {
            f.write_fmt(format_args!(
                "non-empty differential against {}",
                path.display()
            ))?;
        } else {
            f.write_str("non-empty differential")?;
        }
        for line in self.diff.iter() {
            match line {
                Line::Delete(s) => {
                    f.write_str("\n      - ")?;
                    f.write_str(&s)?;
                }
                Line::Insert(s) => {
                    f.write_str("\n      + ")?;
                    f.write_str(&s)?;
                }
                Line::Equal(s) => {
                    f.write_str("\n        ")?;
                    f.write_str(&s)?;
                }
            }
        }
        f.write_str("\n")
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("could not read the test file `{1}`")]
    ReadTestContents(#[source] std::io::Error, PathBuf),
    #[error("could not interpret test file `{1}` as UTF-8")]
    FromUtf8(#[source] std::str::Utf8Error, PathBuf),
    #[error("could not create the parse buffer for the test file")]
    ParseBuffer(#[source] wast::Error),
    #[error("could not parse the test file")]
    ParseWast(#[source] wast::Error),
    #[error("could not encode the wast module `{1}`")]
    EncodeModule(#[source] wast::Error, String),

    #[error("could not open the snapshot file {1:?}")]
    OpenSnap(#[source] std::io::Error, PathBuf),
    #[error("could not read the snapshot file {1:?}")]
    ReadSnap(#[source] std::io::Error, PathBuf),
    #[error("could not truncate the snapshot file {1:?}")]
    TruncateSnap(#[source] std::io::Error, PathBuf),
    #[error("could not seek the snapshot file {1:?}")]
    SeekSnap(#[source] std::io::Error, PathBuf),
    #[error("could not write the snapshot file {1:?}")]
    WriteSnap(#[source] std::io::Error, PathBuf),
    #[error("snapshot comparison failed")]
    DiffSnap(#[source] DiffError),

    #[error("could not execute the interpreter with arguments: {1:?}")]
    InterpreterLaunch(#[source] std::io::Error, Vec<OsString>),
    #[error("could not wait on the interpreter")]
    InterpreterWait(#[source] std::io::Error),
    #[error("reference interpreter failed with exit code {1:?} (arguments: {2:?})")]
    InterpreterExit(
        #[source] Box<dyn std::error::Error + Send + Sync>,
        ExitStatus,
        Vec<OsString>,
    ),
    #[error("interpreter output wasn’t valid UTF-8")]
    InterpreterOutput(#[source] std::string::FromUtf8Error),
    #[error("could not analyze the module {1} at {2:?}")]
    AnalyseModule(#[source] finite_wasm::Error, String, PathBuf),
    #[error("could not analyze the module {0} at {1:?}, analysis panicked")]
    AnalyseModulePanic(String, PathBuf),
    #[error("could not instrument the test module")]
    Instrument(#[source] instrument::Error),
}

impl Error {
    pub(crate) fn set_path(&mut self, path: &Path) {
        match self {
            Error::ReadTestContents(_, _)
            | Error::FromUtf8(_, _)
            | Error::OpenSnap(_, _)
            | Error::SeekSnap(_, _)
            | Error::TruncateSnap(_, _)
            | Error::ReadSnap(_, _)
            | Error::WriteSnap(_, _)
            | Error::InterpreterLaunch(_, _)
            | Error::InterpreterWait(_)
            | Error::InterpreterExit(_, _, _)
            | Error::InterpreterOutput(_)
            | Error::DiffSnap(_)
            | Error::AnalyseModule(_, _, _)
            | Error::AnalyseModulePanic(_, _)
            | Error::Instrument(_) => {}
            Error::ParseBuffer(s) => set_wast_path(s, path),
            Error::ParseWast(s) => set_wast_path(s, path),
            Error::EncodeModule(s, _) => set_wast_path(s, path),
        }
    }
}

fn set_wast_path(error: &mut wast::Error, path: &Path) {
    error.set_path(path);
    if let Ok(content) = std::fs::read_to_string(path) {
        error.set_text(&content);
    }
}

pub(crate) fn read(storage: &mut String, path: &Path) -> Result<(), Error> {
    let test_contents = std::fs::read(path).map_err(|e| Error::ReadTestContents(e, path.into()))?;
    let test_string =
        std::str::from_utf8(&test_contents).map_err(|e| Error::FromUtf8(e, path.into()))?;
    storage.push_str(test_string);
    Ok(())
}

#[derive(Debug, PartialEq)]
enum Status {
    None,
    Passed,
    Failed,
}

#[derive(Debug)]
pub(crate) struct TestContext {
    test_name: String,
    test_path: PathBuf,
    snap_base: PathBuf,
    pub(crate) output: Vec<u8>,
    status: Status,
}

impl<'a> TestContext {
    pub(crate) fn new(test_name: String, test_path: PathBuf, snap_base: PathBuf) -> Self {
        Self {
            test_name,
            test_path,
            snap_base,
            output: vec![],
            status: Status::None,
        }
    }

    pub(crate) fn failed(&self) -> bool {
        self.status == Status::Failed
    }

    fn pass(&mut self) {
        assert!(self.status == Status::None);
        self.status = Status::Passed;
        self.output.extend_from_slice(b"[PASS] ");
        self.output.extend_from_slice(self.test_name.as_bytes());
        self.output.extend_from_slice(b"\n");
    }

    fn fail(&mut self, error: impl error::Error) {
        assert!([Status::None, Status::Failed].contains(&self.status));
        if let Status::None = self.status {
            self.output.extend_from_slice(b"[FAIL] ");
            self.output.extend_from_slice(self.test_name.as_bytes());
            self.output.extend_from_slice(b"\n");
            self.status = Status::Failed;
        }
        super::write_error(&mut self.output, error).expect("this should be infallible");
    }

    fn fail_test_error(&mut self, error: &mut Error) {
        error.set_path(&self.test_path);
        self.fail(&*error)
    }

    pub(crate) fn run(&mut self) {
        if let Err(mut e) = self.run_analysis() {
            return self.fail_test_error(&mut e);
        }

        // Run the interpreter here with the wast file in some sort of a tracing mode (needs to
        // be implemented inside the interpreter).
        //
        // The output is probably going to be extremely verbose, but hey, it doesn’t result in
        // excessive effort at least, does it?
        let output = match self.exec_interpreter(InterpreterMode::GasTrace) {
            Ok(o) => o,
            Err(mut e) => return self.fail_test_error(&mut e),
        };

        // NB: some of the reference snapshots end up at upwards of 250M in size. We can’t
        // reasonably store that in the repository, but we should still be good with storing
        // snapshots of the actual implementation and running the reference interpreter every time.
        //
        // if let Err(mut e) = self.compare_snapshot(output, "interpreter-gas") {
        //     return self.fail_test_error(&mut e);
        // }
        drop(output);

        self.pass();
    }

    fn run_analysis(&mut self) -> Result<(), Error> {
        let mut test_contents = String::new();
        read(&mut test_contents, &self.test_path)?;
        let mut lexer = wast::lexer::Lexer::new(&test_contents);
        lexer.allow_confusing_unicode(true);
        let buf = wast::parser::ParseBuffer::new_with_lexer(lexer).map_err(Error::ParseBuffer)?;
        let wast: wast::Wast = wast::parser::parse(&buf).map_err(Error::ParseWast)?;
        for (directive_index, directive) in wast.directives.into_iter().enumerate() {
            let (id, module) = match directive {
                wast::WastDirective::Wat(wast::QuoteWat::Wat(wast::Wat::Module(mut module))) => {
                    let id = module.id.map_or_else(
                        || format!("[directive {directive_index}]"),
                        |id| format!("{id:?}"),
                    );
                    let module = module
                        .encode()
                        .map_err(|e| Error::EncodeModule(e, id.clone()))?;
                    (id, module)
                }
                wast::WastDirective::Wat(wast::QuoteWat::QuoteModule(_, _)) => {
                    unreachable!("doesn’t actually occur in our test suite");
                }
                wast::WastDirective::Wat(wast::QuoteWat::Wat(wast::Wat::Component(_))) => {
                    // These are difficult and I would rather skip them for now...
                    continue;
                }
                wast::WastDirective::Wat(wast::QuoteWat::QuoteComponent(_, _)) => {
                    // Same
                    continue;
                }

                // Ignore the “operations”, we only care about module analysis results.
                wast::WastDirective::Register { .. } => continue,
                wast::WastDirective::Invoke(_) => continue,
                wast::WastDirective::AssertTrap { .. } => continue,
                wast::WastDirective::AssertReturn { .. } => continue,
                wast::WastDirective::AssertExhaustion { .. } => continue,
                wast::WastDirective::AssertException { .. } => continue,
                // Do not attempt to process invalid modules.
                wast::WastDirective::AssertMalformed { .. } => continue,
                wast::WastDirective::AssertInvalid { .. } => continue,
                wast::WastDirective::AssertUnlinkable { .. } => continue,
            };

            let results = std::panic::catch_unwind(|| {
                finite_wasm::Module::new(&module, Some(&DefaultStackConfig), Some(DefaultGasConfig))
            })
            .map_err(|_| Error::AnalyseModulePanic(id.clone(), self.test_path.clone()))?
            .map_err(|e| Error::AnalyseModule(e, id.clone(), self.test_path.clone()))?;
            let instrumented = module;
            // self.instrument(&module, results).map_err(Error::Instrument)?;
            let print = wasmprinter::print_bytes(&instrumented).expect("print");
            self.compare_snapshot(&print, &format!("instrumented.{id}"))?;
        }
        Ok(())
    }

    fn exec_interpreter(&mut self, mode: InterpreterMode) -> Result<String, Error> {
        let mut args = vec!["-i".into(), self.test_path.as_os_str().into()];
        match mode {
            InterpreterMode::GasTrace => args.push("-tg".into()),
            InterpreterMode::StackTrace => args.push("-ts".into()),
        };

        // TODO: basedir is this the project root, not cwd
        let process = std::process::Command::new("interpreter/wasm")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .args(&args)
            .spawn()
            .map_err(|e| Error::InterpreterLaunch(e, args.clone()))?;
        let output = process
            .wait_with_output()
            .map_err(|e| Error::InterpreterWait(e))?;
        if !output.status.success() {
            return Err(Error::InterpreterExit(
                String::from_utf8_lossy(&output.stderr).into(),
                output.status,
                args,
            ));
        }
        let output = String::from_utf8(output.stdout).map_err(Error::InterpreterOutput)?;
        Ok(output)
    }

    fn compare_snapshot(&mut self, directive_output: &str, index: &str) -> Result<(), Error> {
        let should_update = std::env::var_os("SNAP_UPDATE").is_some();
        let snap_filename = format!("{}@{}.snap", self.test_name, index);
        let snap_path = self.snap_base.join(snap_filename);
        if let Some(parent) = snap_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut snap_file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&snap_path)
            .map_err(|e| Error::OpenSnap(e, snap_path.clone()))?;
        let mut snap_contents = String::with_capacity(1024);
        snap_file
            .read_to_string(&mut snap_contents)
            .map_err(|e| Error::ReadSnap(e, snap_path.clone()))?;
        if let Some(error) = DiffError::diff(&snap_contents, &directive_output) {
            if !should_update {
                self.output.extend_from_slice(
                    "note: run with SNAP_UPDATE environment variable to update\n".as_bytes(),
                );
                Err(Error::DiffSnap(error.with_path(snap_path)))
            } else {
                snap_file
                    .set_len(0)
                    .map_err(|e| Error::TruncateSnap(e, snap_path.clone()))?;
                // TRICKY: If we don’t seek, the file will be filled with 0-bytes up to the
                // current position (we read the file’s contents just now!) before the data is
                // written...
                snap_file
                    .seek(SeekFrom::Start(0))
                    .map_err(|e| Error::SeekSnap(e, snap_path.clone()))?;
                snap_file
                    .write_all(directive_output.as_bytes())
                    .map_err(|e| Error::WriteSnap(e, snap_path))?;
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

struct DefaultStackConfig;
impl max_stack::Config for DefaultStackConfig {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        use wasmparser::ValType::*;
        match ty {
            I32 => 4,
            I64 => 8,
            F32 => 4,
            F64 => 8,
            V128 => 16,
            FuncRef => 32,
            ExternRef => 32,
        }
    }

    fn size_of_function_activation(
        &self,
        locals: &prefix_sum_vec::PrefixSumVec<wasmparser::ValType, u32>,
    ) -> u64 {
        // NB: These sizes have been chosen to make it easier to tell what contributes
        // to locals and what contributes to the operand stack.
        let locals = u64::from(locals.max_index().copied().unwrap_or(0)) * 1_000_000;
        1_000_000_000_000 + locals
    }
}

pub(crate) struct DefaultGasConfig;

macro_rules! gas_visit {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
        $(fn $visit(&mut self, _: usize $($(,$arg: $argty)*)?) -> Self::Output {
            1u64
        })*
    }
}

impl<'a> wasmparser::VisitOperator<'a> for DefaultGasConfig {
    type Output = u64;
    wasmparser::for_each_operator!(gas_visit);
}
