use std::num::TryFromIntError;
use wasmparser::BinaryReaderError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("could not parse a part of the WASM payload")]
    ParsePayload(#[source] BinaryReaderError),
    #[error("could not create a function locals’ reader")]
    LocalsReader(#[source] BinaryReaderError),
    #[error("could not create a function operators’ reader")]
    OperatorReader(#[source] BinaryReaderError),
    #[error("could not visit the function operators")]
    VisitOperators(#[source] BinaryReaderError),
    #[error("could not parse the type section entry")]
    ParseTypes(#[source] BinaryReaderError),
    #[error("could not parse the function section entry")]
    ParseFunctions(#[source] BinaryReaderError),
    #[error("could not parse the global section entry")]
    ParseGlobals(#[source] BinaryReaderError),
    #[error("could not parsse function locals")]
    ParseLocals(#[source] BinaryReaderError),
    #[error("could not parse the imports")]
    ParseImports(#[source] BinaryReaderError),
    #[error("too many functions in the module")]
    TooManyFunctions,
    #[error("could not parse the table section")]
    ParseTable(#[source] BinaryReaderError),

    // These codes are a result of a malformed input (e.g. validation has not been run)
    #[error("could not process locals for function ${1}")]
    TooManyLocals(#[source] prefix_sum_vec::TryPushError, u32),
    #[error("frame stack is too short at offset {0}")]
    TruncatedFrameStack(usize),
    #[error("operand stack is too short at offset {0}")]
    TruncatedOperandStack(usize),
    #[error("empty stack at offset {0}")]
    EmptyStack(usize),
    #[error("type index {0} is out of range")]
    TypeIndexRange(u32, #[source] TryFromIntError),
    #[error("type index {0} refers to a non-existent type")]
    TypeIndex(u32),
    #[error("function index {0} is out of range")]
    FunctionIndexRange(u32, #[source] TryFromIntError),
    #[error("function index {0} refers to a non-existent type")]
    FunctionIndex(u32),
    #[error("global index {0} is out of range")]
    GlobalIndexRange(u32, #[source] TryFromIntError),
    #[error("global index {0} refers to a non-existent global")]
    GlobalIndex(u32),
    #[error("table index {0} is out of range")]
    TableIndexRange(u32, #[source] TryFromIntError),
    #[error("table index {0} refers to a non-existent table")]
    TableIndex(u32),
    #[error("type of local {0} could cannot be found")]
    LocalIndex(u32),
    #[error("too many frames in the function")]
    TooManyFrames,
}
