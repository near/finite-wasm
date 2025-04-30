#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("branch depth is too large at offset {0}")]
    BranchDepthTooLarge(usize),
    #[error("could not parse the brtable targets")]
    ParseBrTable(#[source] wasmparser::BinaryReaderError),
    #[error("the branch target is invalid at offset {0}")]
    InvalidBrTarget(usize),
    #[error("the exceptions proposal is not supported (at offset {0})")]
    ExceptionsNotSupported(usize),
    #[error("the memory control proposal is not supported (at offset {0})")]
    MemoryControlNotSupported(usize),
    #[error("the threads proposal is not supported (at offset {0})")]
    ThreadsNotSupported(usize),
    #[error("the stack switching proposal is not supported (at offset {0})")]
    StackSwitchingNotSupported(usize),
    #[error("the wide arithmetic proposal is not supported (at offset {0})")]
    WideArithmeticNotSupported(usize),
    #[error("the garbage collection proposal is not supported (at offset {0})")]
    GcNotSupported(usize),
}
