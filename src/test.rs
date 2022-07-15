use std::error;
use std::path::{Path, PathBuf};
use wast::parser::{self, Parse, ParseBuffer, Parser};
use wast::token::Index;
use wast::{kw, token};

use crate::indirection::{self, indirect};

mod waft_kw {
    wast::custom_keyword!(assert_instrumented_gas);
    wast::custom_keyword!(assert_instrumented_stack);
    wast::custom_keyword!(assert_instrumented);
    wast::custom_keyword!(assert_indirected);
}

/// A wasm-finite-test description.
#[derive(Debug)]
pub(crate) struct Waft<'a> {
    pub(crate) directives: Vec<WaftDirective<'a>>,
}

#[derive(Debug)]
pub(crate) enum WaftDirective<'a> {
    /// Define a module and make it available for instrumentation.
    Module(wast::core::Module<'a>),
    /// Instrument a specified module with gas only and compare the result.
    AssertGasWat {
        index: Option<token::Index<'a>>,
        expected_result: wast::Wat<'a>,
    },
    /// Instrument a specified module with stack only and compare the result.
    AssertStackWat {
        index: Option<token::Index<'a>>,
        expected_result: wast::Wat<'a>,
    },
    /// Instrument a specified module with all instrumentations and compare the result.
    AssertInstrumentedWat {
        index: Option<token::Index<'a>>,
        expected_result: wast::Wat<'a>,
    },

    /// Apply the indirection trasnformation and compare the output.
    AssertIndirectedWat {
        index: Option<token::Index<'a>>,
        expected_result: wast::Wat<'a>,
    },
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
        } else if l.peek::<waft_kw::assert_instrumented_gas>() {
            parser.parse::<waft_kw::assert_instrumented_gas>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            let expected_result = parser.parse()?;
            Ok(WaftDirective::AssertGasWat {
                index,
                expected_result,
            })
        } else if l.peek::<waft_kw::assert_instrumented_stack>() {
            parser.parse::<waft_kw::assert_instrumented_stack>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            let expected_result = parser.parse()?;
            Ok(WaftDirective::AssertStackWat {
                index,
                expected_result,
            })
        } else if l.peek::<waft_kw::assert_instrumented>() {
            parser.parse::<waft_kw::assert_instrumented>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            let expected_result = parser.parse()?;
            Ok(WaftDirective::AssertInstrumentedWat {
                index,
                expected_result,
            })
        } else if l.peek::<waft_kw::assert_indirected>() {
            parser.parse::<waft_kw::assert_indirected>()?;
            let index = if parser.lookahead1().peek::<token::Index>() {
                Some(parser.parse::<token::Index>()?)
            } else {
                None
            };
            let expected_result = parser.parse()?;
            Ok(WaftDirective::AssertIndirectedWat {
                index,
                expected_result,
            })
        } else {
            Err(l.error())
        }
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
    #[error("could not indirect the module {1}")]
    Indirect(#[source] indirection::Error, wast::Error),
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

pub(crate) struct TestContext<WastErrorBuilder> {
    test_name: String,
    test_path: PathBuf,
    pub(crate) output: Vec<u8>,
    pub(crate) failed: bool,
    error_builder: WastErrorBuilder,
}

impl<WastErrorBuilder> TestContext<WastErrorBuilder>
where
    WastErrorBuilder: Fn(wast::token::Span, String) -> wast::Error,
{
    pub(crate) fn new(
        test_name: String,
        test_path: PathBuf,
        error_builder: WastErrorBuilder,
    ) -> Self {
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
        super::write_error(&mut self.output, error).expect("this should be infallible");
        self.failed = true;
    }

    fn fail_wast(&mut self, error: &mut wast::Error) {
        error.set_path(&self.test_path);
        self.fail(error)
    }

    fn fail_test_error(&mut self, error: &mut Error) {
        error.set_path(&self.test_path);
        self.fail(error)
    }

    fn fail_wast_msg(&mut self, span: wast::token::Span, message: impl Into<String>) {
        let mut error = (self.error_builder)(span, message.into());
        self.fail_wast(&mut error)
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
                let wast = (self.error_builder)(*span, String::new());
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

            match directive {
                WaftDirective::Module(_) => continue,
                WaftDirective::AssertGasWat {
                    expected_result, ..
                } => {
                    drop(expected_result);
                }
                WaftDirective::AssertStackWat {
                    expected_result, ..
                } => {
                    drop(expected_result);
                }
                WaftDirective::AssertInstrumentedWat {
                    expected_result, ..
                } => {
                    drop(expected_result);
                }

                WaftDirective::AssertIndirectedWat {
                    expected_result, ..
                } => match indirect(&encoded) {
                    Ok(_) => drop(expected_result),
                    Err(e) => self.fail_test_error(&mut Error::Indirect(
                        e,
                        (self.error_builder)(*module_span, String::new()),
                    )),
                },
            };
        }

        self.pass()
    }
}
