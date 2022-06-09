use std::path::{Path, PathBuf};
use wast::parser::{self, Parse, ParseBuffer, Parser};
use wast::{kw, token};

mod waft_kw {
    wast::custom_keyword!(assert_instrumented_gas);
    wast::custom_keyword!(assert_instrumented_stack);
    wast::custom_keyword!(assert_instrumented);
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
}

impl Error {
    pub(crate) fn set_path(&mut self, path: &Path) {
        match self {
            Error::ReadTestContents(_, _) => {},
            Error::FromUtf8(_, _) => {},
            Error::NewParseBuffer(s) => s.set_path(path),
            Error::ParseWaft(s) => s.set_path(path),
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

pub(crate) fn lex<'a>(storage: &'a str) -> Result<ParseBuffer<'a>, Error> {
    parser::ParseBuffer::new(storage).map_err(Error::NewParseBuffer)
}

pub(crate) fn parse<'a>(buffer: &'a ParseBuffer) -> Result<Waft<'a>, Error> {
    let parsed = parser::parse(buffer).map_err(Error::ParseWaft)?;
    Ok(parsed)
}
