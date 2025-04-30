use std::num::TryFromIntError;

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    // These codes are a result of a malformed input (e.g. validation has not been run)
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
    #[error("could not process function locals")]
    TooManyLocals(#[source] prefix_sum_vec::TryPushError),
    #[error("the exceptions proposal is not supported (at offset {0})")]
    ExceptionsNotSupported(usize),
    #[error("the memory control proposal is not supported (at offset {0})")]
    MemoryControlNotSupported(usize),
    #[error("the garbage collection proposal is not supported (at offset {0})")]
    GcNotSupported(usize),
    #[error("the threads proposal is not supported (at offset {0})")]
    ThreadsNotSupported(usize),
    #[error("the stack switching proposal is not supported (at offset {0})")]
    StackSwitchingNotSupported(usize),
    #[error("the wide arithmetic proposal is not supported (at offset {0})")]
    WideArithmeticNotSupported(usize),
    #[error("type is too large (at offset {0})")]
    TypeTooLarge(usize),
}
