use std::error;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use wast::token::Span;

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
    // NB: these `wast::Error` are later augmented with a `Path` to the file by the caller.
    #[error("could not construct a parse buffer")]
    NewParseBuffer(#[source] wast::Error),
    #[error("test file is not a valid wasm-finite test file")]
    ParseWaft(#[source] wast::Error),
    #[error("input module is invalid {1}")]
    InvalidModule(#[source] wasmparser::BinaryReaderError, wast::Error),

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
    #[error("snapshot comparison failed {1}")]
    DiffSnap(#[source] DiffError, wast::Error),
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
            | Error::WriteSnap(_, _) => {}
            Error::NewParseBuffer(s) => s.set_path(path),
            Error::ParseWaft(s) => s.set_path(path),
            Error::InvalidModule(_, s) => s.set_path(path),
            Error::DiffSnap(_, s) => s.set_path(path),
        }
    }
}

pub(crate) fn read(storage: &mut String, path: &Path) -> Result<(), Error> {
    let test_contents = std::fs::read(path).map_err(|e| Error::ReadTestContents(e, path.into()))?;
    let test_string =
        std::str::from_utf8(&test_contents).map_err(|e| Error::FromUtf8(e, path.into()))?;
    storage.push_str(test_string);
    Ok(())
}

pub(crate) fn lex(storage: &str) -> Result<ParseBuffer, Error> {
    let mut lexer = wast::lexer::Lexer::new(storage);
    lexer.allow_confusing_unicode(true);
    parser::ParseBuffer::new_with_lexer(lexer).map_err(Error::NewParseBuffer)
}

pub(crate) fn parse<'a>(buffer: &'a ParseBuffer) -> Result<Waft<'a>, Error> {
    let parsed = parser::parse(buffer).map_err(Error::ParseWaft)?;
    Ok(parsed)
}

#[derive(Debug, PartialEq)]
enum Status {
    None,
    Passed,
    Failed,
}

#[derive(Debug)]
pub(crate) struct TestContext<'a> {
    test_name: String,
    test_path: PathBuf,
    wast_data: &'a str,
    pub(crate) output: Vec<u8>,
    status: Status,

    modules: Vec<(Option<wast::token::Id<'a>>, Span, Vec<u8>)>,
}

impl<'a> TestContext<'a> {
    pub(crate) fn new(test_name: String, test_path: PathBuf, wast_data: &'a str) -> Self {
        Self {
            test_name,
            test_path,
            wast_data,
            output: vec![],
            status: Status::None,
            modules: vec![],
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

    fn fail_wast(&mut self, error: &mut wast::Error) {
        error.set_path(&self.test_path);
        self.fail(&*error)
    }

    fn fail_test_error(&mut self, error: &mut Error) {
        error.set_path(&self.test_path);
        self.fail(&*error)
    }

    fn fail_wast_msg(&mut self, span: wast::token::Span, message: impl Into<String>) {
        self.fail_wast(&mut self.wast_error(span, message))
    }

    fn wast_error(&self, span: wast::token::Span, message: impl Into<String>) -> wast::Error {
        let mut error = wast::Error::new(span, message.into());
        error.set_text(self.wast_data);
        error
    }

    pub(crate) fn run(&mut self) {
        // Run the interpreter here with the wast file in some sort of a tracing mode (needs to
        // be implemented inside the interpreter). Use snapshots as a mechanism to store
        // information about execution.
        //
        // The output is probably going to be extremely verbose, but hey, it doesn’t result in
        // excessive effort at least, does it?
        self.pass()
    }

    fn compare_snapshot(
        &mut self,
        span: Span,
        index: usize,
        directive_output: String,
    ) -> Result<(), Error> {
        let should_update = std::env::var_os("SNAP_UPDATE").is_some();
        let snaps_dir = self.test_path.parent().unwrap().join("snaps");
        let snap_filename = format!("{}@{}.snap", self.test_name, index);
        let snap_path = snaps_dir.join(snap_filename);
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
                    "note: run with SNAP_UPDATE environment variable to update".as_bytes(),
                );
                Err(Error::DiffSnap(
                    error.with_path(snap_path),
                    self.wast_error(span, String::new()),
                ))
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
