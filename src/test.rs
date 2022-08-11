use std::error;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use wast::parser::{self, Parse, ParseBuffer, Parser};
use wast::token::Span;
use wast::{kw, token};

mod waft_kw {
    wast::custom_keyword!(assert_instrumented_gas);
    wast::custom_keyword!(assert_instrumented_stack);
    wast::custom_keyword!(assert_instrumented);

    wast::custom_keyword!(snapshot_indirected);
    wast::custom_keyword!(skip);
}

/// A wasm-finite-test description.
pub(crate) struct Waft<'a> {
    pub(crate) directives: Vec<WaftDirective<'a>>,
}

pub(crate) enum WaftDirective<'a> {
    /// Define a module and make it available for instrumentation.
    Module(wast::core::Module<'a>),
    /// Instrument a specified module with gas only and compare the result.
    AssertGasWat { index: Option<token::Index<'a>> },
    /// Instrument a specified module with stack only and compare the result.
    AssertStackWat { index: Option<token::Index<'a>> },
    /// Instrument a specified module with all instrumentations and compare the result.
    AssertInstrumentedWat { index: Option<token::Index<'a>> },
    /// Apply the indirection trasnformation and compare the output.
    AssertIndirectedWat {
        index: Option<token::Index<'a>>,
        span: Span,
    },
    /// Ignore this wast file
    Skip { span: Span },
}

impl<'a> Parse<'a> for Waft<'a> {
    fn parse(parser: Parser<'a>) -> wast::parser::Result<Self> {
        let mut directives = Vec::new();
        while !parser.is_empty() {
            directives.push(parser.parens(|p| p.parse())?);
        }
        Ok(Waft { directives })
    }
}

impl<'a> Parse<'a> for WaftDirective<'a> {
    fn parse(parser: Parser<'a>) -> wast::parser::Result<Self> {
        let mut l = parser.lookahead1();
        if l.peek::<kw::module>() {
            Ok(WaftDirective::Module(parser.parse()?))
        } else if l.peek::<waft_kw::skip>() {
            let kw = parser.parse::<waft_kw::skip>()?;
            let _reason = parser.parse::<String>()?;
            Ok(WaftDirective::Skip { span: kw.0 })
        } else if l.peek::<waft_kw::assert_instrumented_gas>() {
            parser.parse::<waft_kw::assert_instrumented_gas>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            Ok(WaftDirective::AssertGasWat { index })
        } else if l.peek::<waft_kw::assert_instrumented_stack>() {
            parser.parse::<waft_kw::assert_instrumented_stack>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            Ok(WaftDirective::AssertStackWat { index })
        } else if l.peek::<waft_kw::assert_instrumented>() {
            parser.parse::<waft_kw::assert_instrumented>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            Ok(WaftDirective::AssertInstrumentedWat { index })
        } else if l.peek::<waft_kw::snapshot_indirected>() {
            let kw = parser.parse::<waft_kw::snapshot_indirected>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            Ok(WaftDirective::AssertIndirectedWat { index, span: kw.0 })
        } else {
            Err(l.error())
        }
    }
}

impl WaftDirective<'_> {
    fn span(&self) -> Span {
        match self {
            WaftDirective::Module(m) => m.span,
            WaftDirective::Skip { span } => *span,
            WaftDirective::AssertGasWat { .. } => todo!(),
            WaftDirective::AssertStackWat { .. } => todo!(),
            WaftDirective::AssertInstrumentedWat { .. } => todo!(),
            WaftDirective::AssertIndirectedWat { span, .. } => *span,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DiffError {
    diff: Vec<diff::Result<String>>,
    path: Option<PathBuf>,
}

impl DiffError {
    fn diff(old: &str, new: &str) -> Option<Self> {
        if old == new {
            return None;
        }
        let result = diff::lines(old, new);
        if result.is_empty() {
            None
        } else {
            let diff = result
                .into_iter()
                .filter_map(|s| {
                    Some(match s {
                        diff::Result::Left(l) => diff::Result::Left(l.into()),
                        diff::Result::Right(r) => diff::Result::Right(r.into()),
                        diff::Result::Both(l, r) if l != r => {
                            diff::Result::Both(l.into(), r.into())
                        }
                        diff::Result::Both(_, _) => return None,
                    })
                })
                .collect();
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
        for line in &self.diff {
            match line {
                diff::Result::Left(l) => {
                    f.write_str("\n      - ")?;
                    f.write_str(&l)?;
                }
                diff::Result::Right(r) => {
                    f.write_str("\n      + ")?;
                    f.write_str(&r)?;
                }
                diff::Result::Both(l, r) => {
                    f.write_str("\n      - ")?;
                    f.write_str(&l)?;
                    f.write_str("\n      + ")?;
                    f.write_str(&r)?;
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
    #[error("output module is invalid {1}")]
    InvalidOutput(#[source] wasmparser::BinaryReaderError, wast::Error),

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
            Error::InvalidOutput(_, s) => s.set_path(path),
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
    parser::ParseBuffer::new(storage).map_err(Error::NewParseBuffer)
}

pub(crate) fn parse<'a>(buffer: &'a ParseBuffer) -> Result<Waft<'a>, Error> {
    let parsed = parser::parse(buffer).map_err(Error::ParseWaft)?;
    Ok(parsed)
}

#[derive(PartialEq)]
enum Status {
    None,
    Passed,
    Failed,
}

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

    pub(crate) fn run(&mut self, parsed_waft: Result<&mut Waft<'a>, &mut Error>) {
        let test = match parsed_waft {
            Ok(waft) => waft,
            Err(error) => return self.fail_test_error(error),
        };

        // Collect & encode to wasm binary encoding all the input modules.
        let mut errors = false;
        for directive in &mut test.directives {
            if let WaftDirective::Module(module) = directive {
                let id = module.id.clone();
                match module.encode() {
                    Ok(encoded) => self.modules.push((id, module.span, encoded)),
                    Err(error) => {
                        errors = true;
                        self.fail_wast_msg(error.span(), error.message());
                    }
                }
            }
        }

        // Ensure all invalid test cases are caught early and don't become red-herrings.
        if self.validate_modules().is_err() || errors {
            // Module indices may be all invalid now. Don’t actually execute any directives.
            return;
        }

        for (directive_num, directive) in test.directives.iter().enumerate() {
            self.evaluate_directive(directive_num, directive)
        }
        // Ensure we reported some sort of status.
        assert!(matches!(self.status, Status::Passed | Status::Failed));
    }

    fn validate_modules(&mut self) -> Result<(), ()> {
        let mut validation_errors = vec![];
        for (_, span, encoded) in &self.modules {
            if let Err(e) = wasmparser::validate(&encoded) {
                let wast = self.wast_error(*span, String::new());
                validation_errors.push(Error::InvalidModule(e, wast));
            }
        }
        if !validation_errors.is_empty() {
            for error in &mut validation_errors {
                self.fail_test_error(error);
            }
            return Err(());
        }
        Ok(())
    }

    #[allow(unreachable_code, unused_variables)]
    fn evaluate_directive(&mut self, directive_num: usize, directive: &WaftDirective) {
        let index = match directive {
            WaftDirective::Module(_) => return,
            WaftDirective::Skip { .. } => return self.pass(),
            WaftDirective::AssertGasWat { index, .. } => index,
            WaftDirective::AssertStackWat { index, .. } => index,
            WaftDirective::AssertInstrumentedWat { index, .. } => index,
            WaftDirective::AssertIndirectedWat { index, .. } => index,
        };
        let (_module_id, module_span, encoded) = match *index {
            None => match self.modules.get(0) {
                Some(m) => m,
                None => {
                    let span = wast::token::Span::from_offset(0);
                    return self.fail_wast_msg(span, "this file defines no input modules");
                }
            },
            Some(wast::token::Index::Num(num, span)) => match self.modules.get(num as usize) {
                Some(m) => m,
                None => return self.fail_wast_msg(span, format!("module {} is not defined", num)),
            },
            Some(wast::token::Index::Id(i)) => match self.modules.iter().find(|m| m.0 == Some(i)) {
                Some(m) => m,
                None => {
                    return self
                        .fail_wast_msg(i.span(), format!("module {} is not defined", i.name()));
                }
            },
        };

        let output = match directive {
            WaftDirective::Module(_) => return,
            WaftDirective::Skip { .. } => return self.pass(),
            WaftDirective::AssertGasWat { .. } => {
                todo!()
            }
            WaftDirective::AssertStackWat { .. } => {
                todo!()
            }
            WaftDirective::AssertInstrumentedWat { .. } => {
                todo!()
            }
            WaftDirective::AssertIndirectedWat { .. } => {
                todo!()
            }
        };

        match self.compare_snapshot(&directive, directive_num, output) {
            Ok(()) => self.pass(),
            Err(mut e) => self.fail_test_error(&mut e),
        }
    }

    fn compare_snapshot(
        &mut self,
        directive: &WaftDirective,
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
                    self.wast_error(directive.span(), String::new()),
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
