use std::error;
use std::path::{Path, PathBuf};
use wast::parser::{self, Parse, ParseBuffer, Parser, Peek};
use wast::token::Span;
use wast::{kw, token};

use crate::indirection::{self, Indirector};

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

    fn display(&self) -> &'static str {
        let text = match self {
            WaftDirective::Module(_) => kw::module::display(),
            WaftDirective::Skip { .. } => waft_kw::skip::display(),
            WaftDirective::AssertGasWat { .. } => waft_kw::assert_instrumented_gas::display(),
            WaftDirective::AssertStackWat { .. } => waft_kw::assert_instrumented_stack::display(),
            WaftDirective::AssertInstrumentedWat { .. } => waft_kw::assert_instrumented::display(),
            WaftDirective::AssertIndirectedWat { .. } => waft_kw::snapshot_indirected::display(),
        };
        text.strip_prefix("`")
            .unwrap_or(text)
            .strip_suffix("`")
            .unwrap_or(text)
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
    #[error("could not indirect the module {1}")]
    Indirect(#[source] indirection::Error, wast::Error),
    #[error("snapshot comparison failed")]
    Insta,
}

impl Error {
    pub(crate) fn set_path(&mut self, path: &Path) {
        match self {
            Error::ReadTestContents(_, _) => {}
            Error::FromUtf8(_, _) => {}
            Error::NewParseBuffer(s) => s.set_path(path),
            Error::ParseWaft(s) => s.set_path(path),
            Error::Indirect(_, s) => s.set_path(path),
            Error::InvalidModule(_, s) => s.set_path(path),
            Error::InvalidOutput(_, s) => s.set_path(path),
            Error::Insta => {}
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
}

impl<'a> TestContext<'a> {
    pub(crate) fn new(test_name: String, test_path: PathBuf, wast_data: &'a str) -> Self {
        Self {
            test_name,
            test_path,
            wast_data,
            output: vec![],
            status: Status::None,
        }
    }

    pub(crate) fn failed(&self) -> bool {
        matches!(self.status, Status::Failed)
    }

    fn pass(&mut self) {
        assert!(matches!(self.status, Status::None));
        self.status = Status::Passed;
        self.output.extend_from_slice(b"[PASS] ");
        self.output.extend_from_slice(self.test_name.as_bytes());
        self.output.extend_from_slice(b"\n");
    }

    fn fail(&mut self, error: impl error::Error) {
        assert!(matches!(self.status, Status::None | Status::Failed));
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

    pub(crate) fn run(&mut self, parsed_waft: Result<&mut Waft, &mut Error>) {
        let test = match parsed_waft {
            Ok(waft) => waft,
            Err(error) => return self.fail_test_error(error),
        };

        // First, collect, encode to wasm binary encoding all the input modules. Then validate all
        // inputs unconditionally (to ensure all invalid test cases are caught early.)
        let mut modules = Vec::new();
        let mut errors = false;
        for directive in &mut test.directives {
            if let WaftDirective::Module(module) = directive {
                let id = module.id.clone();
                match module.encode() {
                    Ok(encoded) => modules.push((id, module.span, encoded)),
                    Err(error) => {
                        errors = true;
                        self.fail_wast_msg(error.span(), error.message());
                    }
                }
            }
        }
        for (_, span, encoded) in &modules {
            if let Err(e) = wasmparser::validate(&encoded) {
                errors = true;
                let wast = self.wast_error(*span, String::new());
                self.fail_test_error(&mut Error::InvalidModule(e, wast))
            }
        }
        if errors {
            // Module indices are probably all invalid now. Don’t actually execute any directives.
            return;
        }

        // Now, execute all the assertions
        for directive in &test.directives {
            let index = match directive {
                WaftDirective::Module(_) => continue,
                WaftDirective::Skip { .. } => return self.pass(),
                WaftDirective::AssertGasWat { index, .. } => index,
                WaftDirective::AssertStackWat { index, .. } => index,
                WaftDirective::AssertInstrumentedWat { index, .. } => index,
                WaftDirective::AssertIndirectedWat { index, .. } => index,
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
            };
            let (_module_id, module_span, encoded) = match module {
                Ok(m) => m,
                Err(()) => continue,
            };

            let output = match directive {
                WaftDirective::Module(_) => continue,
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
                    match Indirector::analyze(&encoded).and_then(|i| i.indirect()) {
                        Ok(result) => {
                            if let Err(e) = wasmparser::validate(&result) {
                                let wast = self.wast_error(*module_span, String::new());
                                return self.fail_test_error(&mut Error::InvalidOutput(e, wast));
                            }
                            wasmprinter::print_bytes(result).unwrap()
                        }
                        Err(e) => {
                            return self.fail_test_error(&mut Error::Indirect(
                                e,
                                self.wast_error(*module_span, String::new()),
                            ))
                        }
                    }
                }
            };

            let mut insta = insta::Settings::new();
            insta.set_prepend_module_to_snapshot(false);
            insta.set_snapshot_path(
                self.test_path
                    .canonicalize()
                    .expect("unsable to canonicalize the test path")
                    .parent()
                    .expect("could not get the parent directory of the test path")
                    .join("snaps")
            );
            insta.set_snapshot_suffix(directive.display());
            let (line, _) = directive.span().linecol_in(self.wast_data);
            insta.bind(|| {
                let _hook = std::panic::set_hook(Box::new(|_| {}));
                let result = std::panic::catch_unwind(|| {
                    insta::_macro_support::assert_snapshot(
                        // Creates a ReferenceValue::Named variant
                        insta::_macro_support::ReferenceValue::Named(Some(
                            self.test_name.clone().into(),
                        )),
                        &output,
                        env!("CARGO_MANIFEST_DIR"),
                        "",
                        &self.test_name,
                        &self.test_path.display().to_string(),
                        line as u32 + 1, // lines are 0-indexed here
                        directive.display(),
                    )
                });
                let _ = std::panic::take_hook();
                match result {
                    Ok(Ok(_)) => self.pass(),
                    Ok(Err(error)) => return self.fail(&*error),
                    Err(_panic) => return self.fail(Error::Insta),
                }
            });
        }

        // Ensure we reported some sort of status.
        assert!(matches!(self.status, Status::Passed | Status::Failed));
    }
}
