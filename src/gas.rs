use wasmparser::{BlockType, BrTable, VisitOperator};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("branch depth is too large at offset {0}")]
    BranchDepthTooLarge(usize),
    #[error("could not parse the brtable targets")]
    ParseBrTable(#[source] wasmparser::BinaryReaderError),
    #[error("the branch target is invalid at offset {0}")]
    InvalidBrTarget(usize),
}

pub(crate) enum InstructionKind {
    /// This instruction is largely uninteresting for the purposes of gas analysis, besides its
    /// inherent cost.
    ///
    /// These do not participate in branching, trapping or any other kind of data flow. Simple
    /// operations such as an addition or subtraction will typically fall under this kind.
    Pure,

    /// This instruction is a definite branch target.
    ///
    /// As a result the gas charge cannot be consolidated across this instruction.
    BranchTarget,

    /// This instruction is a branch.
    Branch,

    /// This instruction is a side effect.
    ///
    /// An example of such a side-effect could be a trap, or a host function call.
    ///
    /// At the time this instruction is executed the gas counter must be exact.
    SideEffect,

    /// This instruction is an aggregate operation.
    ///
    /// Instructions such as `memory.fill` fall here. The amount of work they do depends on the
    /// operands.
    ///
    /// TODO: a variant for each such instruction may be warranted?
    Aggregate,

    /// This instruction is unreachable (usually because the stack is polymorphic at this point.)
    ///
    /// We could remove these instructions entirely in our instrumentation code.
    Unreachable,
}

#[derive(Debug)]
pub(crate) enum BranchKind {
    // A branch instruction in this frame branches backwards in the instruction stream to the
    // instruction at the provided index.
    Backward(usize),
    // A branch instruction in this frame branches forwards, but no branch instruction doing so has
    // been encountered yet.
    //
    // If a frame is popped while in this state, the surrounding structure that created this frame
    // in the first place is dead code.
    UntakenForward,
    // There is a branch instruction within this frame that branches to the `End` (or `Else` in the
    // case of `if..else..end`) instruction.
    Forward,
}

#[derive(Debug)]
pub(crate) struct Frame {
    /// Are the remaining instructions within this frame reachable?
    ///
    /// That is, is the operand stack part of this frame polymorphic? See `stack-polymorphic` in
    /// the wasm-core specification for explanation.
    pub(crate) stack_polymorphic: bool,

    /// Index to the instruction which a backwards branch would target.
    ///
    /// This is _only_ relevant (and `Some`) for frames created by the `loop` instruction).
    pub(crate) kind: BranchKind,
}

pub(crate) struct GasVisitor<'a, CostModel> {
    pub(crate) offset: usize,

    /// A visitor that produces costs for instructions.
    pub(crate) model: &'a mut CostModel,

    /// Table of instruction ranges, and the total gas cost of executing the range.
    pub(crate) offsets: &'a mut Vec<usize>,
    pub(crate) costs: &'a mut Vec<u64>,
    pub(crate) types: &'a mut Vec<InstructionKind>,

    /// Information about the analyzed function’s frame stack.
    pub(crate) frame_stack: &'a mut Vec<Frame>,
    pub(crate) current_frame: Frame,
}

macro_rules! generate_visitor {
    // Special cases. For the purposes of readability, and resilience to future changes in
    // wasmparser those are implemented in the trait definition below.
    (visit_unreachable $($_:tt)*) => {};
    (visit_end $($_:tt)*) => {};
    (visit_loop $($_:tt)*) => {};
    (visit_if $($_:tt)*) => {};
    (visit_else $($_:tt)*) => {};
    (visit_block $($_:tt)*) => {};

    (visit_br $($_:tt)*) => {};
    (visit_br_if $($_:tt)*) => {};
    (visit_br_table $($_:tt)*) => {};
    (visit_return $($_:tt)*) => {};
    (visit_call $($_:tt)*) => {};
    (visit_call_indirect $($_:tt)*) => {};
    (visit_return_call $($_:tt)*) => {};
    (visit_return_call_indirect $($_:tt)*) => {};

    (visit_try $($_:tt)*) => {};
    (visit_rethrow $($_:tt)*) => {};
    (visit_throw $($_:tt)*) => {};
    (visit_delegate $($_:tt)*) => {};
    (visit_catch $($_:tt)*) => {};
    (visit_catch_all $($_:tt)*) => {};

    (visit_memory_copy $($_:tt)*) => {};
    (visit_memory_fill $($_:tt)*) => {};
    (visit_memory_grow $($_:tt)*) => {};
    (visit_memory_init $($_:tt)*) => {};
    (visit_data_drop $($_:tt)*) => {};

    (visit_table_copy $($_:tt)*) => {};
    (visit_table_fill $($_:tt)*) => {};
    (visit_table_grow $($_:tt)*) => {};
    (visit_table_init $($_:tt)*) => {};
    (visit_elem_drop $($_:tt)*) => {};

    ($visit:ident( $($arg:ident: $argty:ty),* )) => {
        fn $visit(&mut self, offset: usize $(,$arg: $argty)*) -> Self::Output {
            let cost = self.model.$visit(offset, $($arg),*);
            self.visit_instruction(InstructionKind::Pure, cost);
            Ok(())
        }
    };

    ($( @$_a:ident $_b:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
        $(generate_visitor!{ $visit(
            $($($arg: $argty),*)?
        ) })*
    }
}

impl<'a, CostModel> GasVisitor<'a, CostModel> {
    /// Visit an instruction, populating information about it within the internal tables.
    fn visit_instruction(&mut self, kind: InstructionKind, cost: u64) {
        self.offsets.push(self.offset);
        self.types.push(if self.current_frame.stack_polymorphic {
            InstructionKind::Unreachable
        } else {
            kind
        });
        self.costs.push(cost);
    }

    /// Create a new frame on the frame stack.
    ///
    /// The caller is responsible for specifying how the branches behve when branched to this
    /// frame.
    fn new_frame(&mut self, kind: BranchKind) {
        let stack_polymorphic = self.current_frame.stack_polymorphic;
        self.frame_stack.push(std::mem::replace(
            &mut self.current_frame,
            Frame {
                stack_polymorphic,
                kind,
            },
        ));
    }

    fn end_frame(&mut self) -> Option<Frame> {
        if let Some(frame) = self.frame_stack.pop() {
            let frame = std::mem::replace(&mut self.current_frame, frame);
            Some(frame)
        } else {
            None
        }
    }

    /// Mark the current frame as polymorphic.
    ///
    /// This means that any operand stack push and pop operations will do nothing and uncontionally
    /// succeed, while this frame is still active.
    fn stack_polymorphic(&mut self) {
        self.current_frame.stack_polymorphic = true;
    }

    fn branch(&mut self, depth: usize) -> Result<(), Error> {
        let frame = if let Some(frame_stack_index) = depth.checked_sub(1) {
            self.frame_stack
                .iter_mut()
                .nth_back(frame_stack_index)
                .ok_or(Error::InvalidBrTarget(self.offset))?
        } else {
            &mut self.current_frame
        };
        match frame.kind {
            BranchKind::Forward => (),
            BranchKind::UntakenForward => frame.kind = BranchKind::Forward,
            BranchKind::Backward(instruction_index) => {
                self.types[instruction_index] = InstructionKind::BranchTarget
            }
        }
        Ok(())
    }
}

impl<'a, CostModel: VisitOperator<'a, Output = u64>> VisitOperator<'a>
    for GasVisitor<'a, CostModel>
{
    type Output = Result<(), Error>;

    wasmparser::for_each_operator!(generate_visitor);

    fn visit_unreachable(&mut self, offset: usize) -> Self::Output {
        let cost = self.model.visit_unreachable(offset);
        self.visit_instruction(InstructionKind::SideEffect, cost);
        self.stack_polymorphic();
        Ok(())
    }

    fn visit_loop(&mut self, offset: usize, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_loop(offset, blockty);
        let insn_type_index = self.types.len();
        // For the time being this instruction is not a branch target, and therefore is pure.
        self.visit_instruction(InstructionKind::Pure, cost);
        // However, it will become a branch target if there is a branching instruction targettting
        // the frame created by this instruction. At that point we will make a point of adjusting
        // the instruction kind to a `InstructionKind::BranchTarget`.
        self.new_frame(BranchKind::Backward(insn_type_index));
        Ok(())
    }

    // Branch Target (for if, block, else), only if there is a `br`/`br_if`/`br_table` to exactly
    // the frame created by the matching insn.
    fn visit_end(&mut self, offset: usize) -> Self::Output {
        let cost = self.model.visit_end(offset);
        let kind = match self.current_frame.kind {
            BranchKind::Forward => InstructionKind::BranchTarget,
            BranchKind::Backward(_) => InstructionKind::Pure,
            BranchKind::UntakenForward => InstructionKind::Pure,
        };
        self.visit_instruction(kind, cost);
        self.end_frame();
        Ok(())
    }

    // Branch
    fn visit_if(&mut self, offset: usize, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_if(offset, blockty);
        // `if` is already a branch instruction, it can execute the instruction that follows, or it
        // could jump to the `else` or `end` associated with this frame.
        self.visit_instruction(InstructionKind::Branch, cost);
        self.new_frame(BranchKind::Forward);
        Ok(())
    }

    // Branch Target (unconditionally)
    fn visit_else(&mut self, offset: usize) -> Self::Output {
        let cost = self.model.visit_else(offset);
        // `else` is already a taken branch target from `if` (if the condition is false).
        self.end_frame();
        // `else` is both a branch and a branch target, depending on how it was reached.
        // If `else` ends up being reached by control flow naturally reaching this instruction
        // from the truthy `if` body, then this is logically a branch to the `end` instruction
        // associated with this frame.
        //
        // For some interpretations it can also be a branch target -- when the `if` condition is
        // false, `if` can be considered to be a branch to either this `else` or to the first
        // instruction of its body. Which interpretation makes sense here depends really on how
        // frames are handled.
        //
        // We’ll use the first interpretation, but in practice either would work equally well.
        self.visit_instruction(InstructionKind::Branch, cost);
        self.new_frame(BranchKind::Forward);
        Ok(())
    }

    fn visit_block(&mut self, offset: usize, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_block(offset, blockty);
        self.visit_instruction(InstructionKind::Pure, cost);
        self.new_frame(BranchKind::UntakenForward);
        Ok(())
    }

    fn visit_br(&mut self, offset: usize, relative_depth: u32) -> Self::Output {
        let frame_idx =
            usize::try_from(relative_depth).map_err(|_| Error::BranchDepthTooLarge(self.offset))?;
        self.branch(frame_idx)?;
        let cost = self.model.visit_br(offset, relative_depth);
        self.visit_instruction(InstructionKind::Branch, cost);
        Ok(())
    }

    fn visit_br_if(&mut self, offset: usize, relative_depth: u32) -> Self::Output {
        let frame_idx =
            usize::try_from(relative_depth).map_err(|_| Error::BranchDepthTooLarge(self.offset))?;
        self.branch(frame_idx)?;
        let cost = self.model.visit_br_if(offset, relative_depth);
        self.visit_instruction(InstructionKind::Branch, cost);
        Ok(())
    }

    fn visit_br_table(&mut self, offset: usize, targets: BrTable<'a>) -> Self::Output {
        for target in targets.targets() {
            let target = target.map_err(Error::ParseBrTable)?;
            let frame_idx =
                usize::try_from(target).map_err(|_| Error::BranchDepthTooLarge(self.offset))?;
            self.branch(frame_idx)?;
        }
        let cost = self.model.visit_br_table(offset, targets);
        self.visit_instruction(InstructionKind::Branch, cost);
        Ok(())
    }

    fn visit_return(&mut self, offset: usize) -> Self::Output {
        let cost = self.model.visit_return(offset);
        self.visit_instruction(InstructionKind::Branch, cost);
        let root_frame_index = self.frame_stack.len();
        self.branch(root_frame_index)?;
        Ok(())
    }

    fn visit_call(&mut self, offset: usize, function_index: u32) -> Self::Output {
        let cost = self.model.visit_call(offset, function_index);
        // Functions can inspect the remaining gas, or initiate other side effects (e.g. trap) so
        // we must be conservative with its handling. Inlining is a transformation which would
        // allow us to be less conservative, but it will already have been done during the
        // compilation from the source language to wasm, or wasm-opt, most of the time.
        self.visit_instruction(InstructionKind::SideEffect, cost);
        Ok(())
    }

    fn visit_call_indirect(
        &mut self,
        offset: usize,
        type_index: u32,
        table_index: u32,
        table_byte: u8,
    ) -> Self::Output {
        let cost = self
            .model
            .visit_call_indirect(offset, type_index, table_index, table_byte);
        self.visit_instruction(InstructionKind::SideEffect, cost);
        Ok(())
    }

    fn visit_return_call(&mut self, offset: usize, function_index: u32) -> Self::Output {
        let cost = self.model.visit_return_call(offset, function_index);
        // This is both a side-effect _and_ a branch. Side effects are more general, so we pick
        // that.
        self.visit_instruction(InstructionKind::SideEffect, cost);
        // NB: this implicitly is 1-less than the number of frames due to us maintaining a
        // `current_frame` field.
        let root_frame_index = self.frame_stack.len();
        self.branch(root_frame_index)?;
        Ok(())
    }

    fn visit_return_call_indirect(
        &mut self,
        offset: usize,
        type_index: u32,
        table_index: u32,
    ) -> Self::Output {
        let cost = self
            .model
            .visit_return_call_indirect(offset, type_index, table_index);
        self.visit_instruction(InstructionKind::SideEffect, cost);
        // NB: this implicitly is 1-less than the number of frames due to us maintaining a
        // `current_frame` field.
        let root_frame_index = self.frame_stack.len();
        self.branch(root_frame_index)?;
        Ok(())
    }

    fn visit_memory_grow(&mut self, offset: usize, mem: u32, mem_byte: u8) -> Self::Output {
        let cost = self.model.visit_memory_grow(offset, mem, mem_byte);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_memory_init(&mut self, offset: usize, data_index: u32, mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_init(offset, data_index, mem);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_data_drop(&mut self, offset: usize, data_index: u32) -> Self::Output {
        // TODO: [] -> []; does not interact with the operand stack, so isn’t really an aggregate
        // instruction. In practice, though, it may involve non-trivial amount of work in the
        // runtime anyway? Validate.
        let cost = self.model.visit_data_drop(offset, data_index);
        self.visit_instruction(InstructionKind::Pure, cost);
        Ok(())
    }

    fn visit_memory_copy(&mut self, offset: usize, dst_mem: u32, src_mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_copy(offset, dst_mem, src_mem);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_memory_fill(&mut self, offset: usize, mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_fill(offset, mem);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_table_init(&mut self, offset: usize, elem_index: u32, table: u32) -> Self::Output {
        let cost = self.model.visit_table_init(offset, elem_index, table);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_elem_drop(&mut self, offset: usize, elem_index: u32) -> Self::Output {
        // TODO: [] -> []; does not interact with the operand stack, so isn’t really an aggregate
        // instruction. In practice, though, it may involve non-trivial amount of work in the
        // runtime anyway? Validate.
        let cost = self.model.visit_elem_drop(offset, elem_index);
        self.visit_instruction(InstructionKind::Pure, cost);
        Ok(())
    }

    fn visit_table_copy(&mut self, offset: usize, dst_table: u32, src_table: u32) -> Self::Output {
        let cost = self.model.visit_table_copy(offset, dst_table, src_table);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_table_fill(&mut self, offset: usize, table: u32) -> Self::Output {
        let cost = self.model.visit_table_fill(offset, table);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_table_grow(&mut self, offset: usize, table: u32) -> Self::Output {
        let cost = self.model.visit_table_grow(offset, table);
        self.visit_instruction(InstructionKind::Aggregate, cost);
        Ok(())
    }

    fn visit_try(&mut self, _: usize, _: BlockType) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_catch(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_throw(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_rethrow(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_delegate(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_catch_all(&mut self, _: usize) -> Self::Output {
        todo!("exception handling extension")
    }
}
