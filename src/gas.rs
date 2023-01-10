//! # Gas analysis
//!
//! The gas analysis is a two-pass linear algorithm. This algorithm first constructs a table along
//! the lines of
//!
//! | Instructions | Cost | [Kind] |
//! | ============ | ==== | ====== |
//! | i32.const 0  | 1    | Pure   |
//! | i32.const 1  | 1    | Pure   |
//! | i32.add      | 1    | Pure   |
//!
//! [Kind]: InstrumentationKind
//!
//! In this table the instructions with certain instrumentation kind combinations can then be
//! coalesced in order to reduce the number of gas instrumentation points. For example all
//! instrumentation can be merged together across all instructions considered `Pure` to produce a
//! table like this:
//!
//! | Instructions                           | Cost | [Kind] |
//! | ====================================== | ==== | ====== |
//! | (i32.add (i32.const 0) (i32.const 1))  | 3    | Pure   |
//!
//! Instrumentation can then, instead of inserting a gas charge before each of the 3 instructions,
//! insert a charge of 3 gas before the entire sequence, all without any observable difference in
//! execution semantics.
//!
//! **Why two passes?** A short answer is – branching. As the algorithm goes through the function
//! code for the first time, it can mark certain instructions as being a branch or a branch target.
//! For example `end` in `block…end` can be either pure or a branch target. Table entries for the
//! instructions that participate in control flow cannot be merged together if an eventually
//! accurate gas count is desired.

use crate::instruction_categories as gen;
use wasmparser::{BlockType, BrTable, VisitOperator};

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("branch depth is too large at offset {0}")]
    BranchDepthTooLarge(usize),
    #[error("could not parse the brtable targets")]
    ParseBrTable(#[source] wasmparser::BinaryReaderError),
    #[error("the branch target is invalid at offset {0}")]
    InvalidBrTarget(usize),
}

#[derive(Clone, Copy, Debug)]
pub enum InstrumentationKind {
    /// This instrumentation point precedes an instruction that is largely uninteresting for the
    /// purposes of gas analysis, besides its inherent cost.
    ///
    /// These do not participate in branching, trapping or any other kind of data flow. Simple
    /// operations such as an addition or subtraction will typically fall under this kind.
    Pure,

    /// This instrumentation point is unreachable (usually because the stack is polymorphic at this
    /// point.)
    ///
    /// For example,
    ///
    /// ```wast
    /// block
    ///   br 0                                ;; the stack becomes polymorphic
    ///                                       ;; this instrumentation point is unreachable
    ///   i32.add (i32.const 0) (i32.const 1)
    ///                                       ;; this instrumentation point is still unreachable
    /// end
    ///                                       ;; this instrumentation point is reachable again
    /// ```
    Unreachable,

    /// This instrumentation point precedes a branch target, branch, side effect, potential trap or
    /// similar construct.
    ///
    /// As a result none of the succeeding instrumentation points may be merged up into this
    /// instrumentation point, as the gas may be observed by the instruction that is going to
    /// execute after this instruction.
    PreControlFlow,

    /// This instrumentation point succeeds a branch target, branch, side effect, potential trap or
    /// similar construct.
    ///
    /// As a result this instrumentation point may not be merged up into the preceding
    /// instrumentation points (but succeeding instrumentation points may still be merged into
    /// this).
    PostControlFlow,

    /// This instrumentation point precedes an aggregate operation.
    ///
    /// Instructions such as `memory.fill` cause this categorization. The amount of work they do
    /// depends on the operands.
    ///
    /// TODO: a variant for each such instruction may be warranted?
    Aggregate,
}

#[derive(Debug)]
pub(crate) enum BranchTargetKind {
    // A branch instruction in this frame branches backwards in the instruction stream to the
    // instruction at the provided index.
    Backward(usize),
    // A branch instruction in this frame branches forwards, but no branch instruction doing so has
    // been encountered yet.
    //
    // If a frame is popped while in this state, the surrounding structure that created this frame
    // in the first place is dead code.
    //
    // For example, `block (BODY…) end` without any branches to the frame created by `block`, can
    // be replaced with just `BODY…`.
    UntakenForward,
    // There is a branch instruction within this frame that branches to the `End` (or `Else` in the
    // case of `if…else…end`) instruction.
    Forward,
}

#[derive(Debug)]
pub(crate) struct Frame {
    /// Is the operand stack for the remainder of this frame considered polymorphic?
    ///
    /// Once the stack becomes polymorphic, the only way for it to stop being polymorphic is to pop
    /// the frames within which the stack is polymorphic.
    ///
    /// Note, that unlike validation, for the purposes of this analysis stack polymorphism is
    /// somewhat more lax. For example, validation algorithm will readly reject a function like
    /// `(func (unreachable) (i64.const 0) (i32.add))`, because at the time `i32.add` is evaluated
    /// the stack is `[t* i64]`, which does not unify with `[i32 i32] -> [i32]` expected by
    /// `i32.add`, despite being polymorphic. For the purposes of this analysis we do not keep
    /// track of the stack contents to that level of detail – all we care about is whether the
    /// stack is polymorphic at all.
    ///
    /// Search for `stack-polymorphic` in the wasm-core specification for further description and
    /// the list of instructions that make the stack polymorphic.
    pub(crate) stack_polymorphic: bool,

    /// How does a branch instruction behaves when targetting this frame?
    pub(crate) kind: BranchTargetKind,
}

pub(crate) struct GasVisitor<'a, CostModel> {
    pub(crate) offset: usize,

    /// A visitor that produces costs for instructions.
    pub(crate) model: &'a mut CostModel,

    /// Table of instruction ranges, and the total gas cost of executing the range.
    pub(crate) offsets: &'a mut Vec<usize>,
    pub(crate) costs: &'a mut Vec<u64>,
    pub(crate) kinds: &'a mut Vec<InstrumentationKind>,

    /// Is there a charge we want to introduce "after" the current offset?
    ///
    /// Note that the implementation here depends on the fact that all instructions invoke
    /// `charge_before` or `charge_after`, even if with a 0-cost so that there is an opportunity to
    /// merge this cost into the table.
    pub(crate) next_offset_cost: Option<(u64, InstrumentationKind)>,

    /// Information about the analyzed function’s frame stack.
    pub(crate) frame_stack: &'a mut Vec<Frame>,
    pub(crate) current_frame: Frame,
}

impl<'a, CostModel> GasVisitor<'a, CostModel> {
    /// Charge some gas before the currently analyzed instruction.
    fn charge_before(&mut self, kind: InstrumentationKind, cost: u64) {
        let next_offset_cost = self.next_offset_cost.take();
        let cost = cost + next_offset_cost.map(|v| v.0).unwrap_or(0);
        self.offsets.push(self.offset);
        self.kinds.push(if self.current_frame.stack_polymorphic {
            InstrumentationKind::Unreachable
        } else {
            kind
        });
        self.costs.push(cost);
    }

    /// Charge some gas after the currently analyzed instruction.
    ///
    /// Note that this method works by enqueueing a charge to be added to the tables at a next call
    /// of the `charge_before` or `charge_after` function.
    fn charge_after(&mut self, kind: InstrumentationKind, cost: u64) {
        if let Some((cost, kind)) = self.next_offset_cost.take() {
            self.charge_before(kind, cost);
        }
        assert!(self.next_offset_cost.replace((cost, kind)).is_none());
    }

    /// Create a new frame on the frame stack.
    ///
    /// The caller is responsible for specifying how the branches behave when branched to this
    /// frame.
    fn new_frame(&mut self, kind: BranchTargetKind) {
        let stack_polymorphic = self.current_frame.stack_polymorphic;
        self.frame_stack.push(std::mem::replace(
            &mut self.current_frame,
            Frame {
                stack_polymorphic,
                kind,
            },
        ));
    }

    /// Terminate the current top-most frame on the frame stack.
    ///
    /// When there is only one frame remaining this becomes a no-op.
    fn end_frame(&mut self) {
        if let Some(frame) = self.frame_stack.pop() {
            self.current_frame = frame;
        }
    }

    /// Mark the current frame as polymorphic.
    fn make_polymorphic(&mut self) {
        self.current_frame.stack_polymorphic = true;
    }

    /// The index of the root frame (that is the one representing the function entry.)
    fn root_frame_index(&self) -> usize {
        // NB: this implicitly is 1-less than the number of frames due to us maintaining a
        // `current_frame` field.
        self.frame_stack.len()
    }

    fn frame_index(&self, relative_depth: u32) -> Result<usize, Error> {
        usize::try_from(relative_depth).map_err(|_| Error::BranchDepthTooLarge(self.offset))
    }

    fn visit_branch(&mut self, depth: usize) -> Result<(), Error> {
        let frame = if let Some(frame_stack_index) = depth.checked_sub(1) {
            self.frame_stack
                .iter_mut()
                .nth_back(frame_stack_index)
                .ok_or(Error::InvalidBrTarget(self.offset))?
        } else {
            &mut self.current_frame
        };
        match frame.kind {
            BranchTargetKind::Forward => (),
            BranchTargetKind::UntakenForward => frame.kind = BranchTargetKind::Forward,
            BranchTargetKind::Backward(instruction_index) => {
                self.kinds[instruction_index] = InstrumentationKind::PostControlFlow
            }
        }
        Ok(())
    }

    fn visit_unconditional_branch(&mut self, depth: usize, cost: u64) -> Result<(), Error> {
        self.charge_before(InstrumentationKind::PreControlFlow, cost);
        self.visit_branch(depth)?;
        self.make_polymorphic();
        Ok(())
    }
}

macro_rules! trapping_insn {
    (fn $visit:ident( $($arg:ident: $ty:ty),* )) => {
        fn $visit(&mut self, $($arg: $ty),*) -> Self::Output {
            let cost = self.model.$visit($($arg),*);
            self.charge_before(InstrumentationKind::PreControlFlow, cost);
            Ok(())
        }
    };
    ($($_t:ident .
        $(atomic.rmw)?
        $(atomic.cmpxchg)?
        $(load)?
        $(store)?
    = $($insn:ident)|* ;)*) => {
        $($(trapping_insn!(fn $insn(mem: wasmparser::MemArg));)*)*
    };
    ($($_t:ident . $(loadlane)? $(storelane)? = $($insn:ident)|* ;)*) => {
        $($(trapping_insn!(fn $insn(mem: wasmparser::MemArg, lane: u8));)*)*
    };
    ($($_t:ident . $(binop)? $(cvtop)? = $($insn:ident)|* ;)*) => {
        $($(trapping_insn!(fn $insn());)*)*
    };
}

macro_rules! pure_insn {
    (fn $visit:ident( $($arg:ident: $ty:ty),* )) => {
        fn $visit(&mut self, $($arg: $ty),*) -> Self::Output {
            let cost = self.model.$visit($($arg),*);
            self.charge_before(InstrumentationKind::Pure, cost);
            Ok(())
        }
    };
    ($($_t:ident .
        // This sequence below "matches" any of these categories.
        $(unop)?
        $(binop)?
        $(cvtop)?
        $(relop)?
        $(testop)?
        $(vbitmask)?
        $(vinarrowop)?
        $(vrelop)?
        $(vternop)?
        $(vishiftop)?
        $(splat)?
    = $($insn:ident)|* ;)*) => {
        $($(pure_insn!(fn $insn());)*)*
    };
    ($($_t:ident . const = $($insn:ident, $param:ty)|* ;)*) => {
        $($(pure_insn!(fn $insn(val: $param));)*)*
    };
    ($($_t:ident . $(extractlane)? $(replacelane)? = $($insn:ident)|* ;)*) => {
        $($(pure_insn!(fn $insn(lane: u8));)*)*
    };
    ($($_t:ident . localsglobals = $($insn:ident)|* ;)*) => {
        $($(pure_insn!(fn $insn(index: u32));)*)*
    };
}

impl<'a, CostModel: VisitOperator<'a, Output = u64>> VisitOperator<'a>
    for GasVisitor<'a, CostModel>
{
    type Output = Result<(), Error>;

    gen::atomic_cmpxchg!(trapping_insn);
    gen::atomic_rmw!(trapping_insn);
    gen::load!(trapping_insn);
    gen::store!(trapping_insn);
    gen::loadlane!(trapping_insn);
    gen::storelane!(trapping_insn);
    gen::binop_partial!(trapping_insn);
    gen::cvtop_partial!(trapping_insn);

    // Functions can inspect the remaining gas, or initiate other side effects (e.g. trap) so
    // we must be conservative with its handling. Inlining is a transformation which would
    // allow us to be less conservative, but it will already have been done during the
    // compilation from the source language to wasm, or wasm-opt, most of the time.
    trapping_insn!(fn visit_call(index: u32));
    trapping_insn!(fn visit_call_indirect(ty_index: u32, table_index: u32, table_byte: u8));
    // TODO: double check if these may actually trap
    trapping_insn!(fn visit_memory_atomic_notify(mem: wasmparser::MemArg));
    trapping_insn!(fn visit_memory_atomic_wait32(mem: wasmparser::MemArg));
    trapping_insn!(fn visit_memory_atomic_wait64(mem: wasmparser::MemArg));
    trapping_insn!(fn visit_table_set(table: u32));
    trapping_insn!(fn visit_table_get(table: u32));

    fn visit_unreachable(&mut self) -> Self::Output {
        let cost = self.model.visit_unreachable();
        self.charge_before(InstrumentationKind::PreControlFlow, cost);
        self.make_polymorphic();
        Ok(())
    }

    gen::binop_complete!(pure_insn);
    gen::cvtop_complete!(pure_insn);
    gen::unop!(pure_insn);
    gen::relop!(pure_insn);
    gen::vrelop!(pure_insn);
    gen::vishiftop!(pure_insn);
    gen::vternop!(pure_insn);
    gen::vbitmask!(pure_insn);
    gen::vinarrowop!(pure_insn);
    gen::splat!(pure_insn);
    gen::r#const!(pure_insn);
    gen::extractlane!(pure_insn);
    gen::replacelane!(pure_insn);
    gen::testop!(pure_insn);

    pure_insn!(fn visit_ref_null(t: wasmparser::ValType));
    pure_insn!(fn visit_ref_func(index: u32));
    pure_insn!(fn visit_i8x16_shuffle(pattern: [u8; 16]));
    pure_insn!(fn visit_atomic_fence());
    pure_insn!(fn visit_select());
    pure_insn!(fn visit_typed_select(t: wasmparser::ValType));
    pure_insn!(fn visit_drop());
    pure_insn!(fn visit_nop());
    pure_insn!(fn visit_table_size(table: u32));
    pure_insn!(fn visit_memory_size(mem: u32, idk: u8));
    pure_insn!(fn visit_global_set(global: u32));
    pure_insn!(fn visit_global_get(global: u32));
    pure_insn!(fn visit_local_set(local: u32));
    pure_insn!(fn visit_local_get(local: u32));
    pure_insn!(fn visit_local_tee(local: u32));

    fn visit_loop(&mut self, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_loop(blockty);
        let insn_type_index = self.kinds.len();
        // For the time being this instruction is not a branch target, and therefore is pure.
        // However, we must charge for it _after_ it has been executed, just in case it becomes a
        // branch target later. That's because as per the WebAssembly specification, the `loop`
        // instruction is executed on every iteration.
        self.charge_after(InstrumentationKind::Pure, cost);
        // This instruction will become a branch target if there is a branching instruction
        // targetting the frame created by this instruction. At that point we will make a point of
        // adjusting the instruction kind to a `InstrumentationKind::PostControlFlow`.
        self.new_frame(BranchTargetKind::Backward(insn_type_index));
        Ok(())
    }

    // Branch Target (for if, block, else), only if there is a `br`/`br_if`/`br_table` to exactly
    // the frame created by the matching insn.
    fn visit_end(&mut self) -> Self::Output {
        let cost = self.model.visit_end();
        assert!(cost == 0, "the `end` instruction costs aren’t handled right, set it to 0");
        let kind = match self.current_frame.kind {
            BranchTargetKind::Forward => InstrumentationKind::PreControlFlow,
            BranchTargetKind::Backward(_) => InstrumentationKind::Pure,
            BranchTargetKind::UntakenForward => InstrumentationKind::Pure,
        };
        self.end_frame();
        // TODO: this needs to note if this is a `if..end` or `if..else..end` branch. In the case
        // of the former, the code consuming analysis results needs to know to generate the `else`
        // branch and propagate the gas cost in both branches.
        //
        // Note that we cannot `charge_after` here because `end` is not "executed" when a branch
        // within the frame is executed.
        self.charge_before(kind, cost);
        Ok(())
    }

    // Branch
    fn visit_if(&mut self, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_if(blockty);
        // `if` is already a branch instruction, it can execute the instruction that follows, or it
        // could jump to the `else` or `end` associated with this frame.
        self.charge_before(InstrumentationKind::PreControlFlow, cost);
        self.new_frame(BranchTargetKind::Forward);
        Ok(())
    }

    // Branch Target (unconditionally)
    fn visit_else(&mut self) -> Self::Output {
        let cost = self.model.visit_else();
        assert!(cost == 0, "the `else` instruction costs aren’t handled right, set it to 0");
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
        self.charge_before(InstrumentationKind::PreControlFlow, cost);
        self.new_frame(BranchTargetKind::Forward);
        Ok(())
    }

    fn visit_block(&mut self, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_block(blockty);
        self.charge_before(InstrumentationKind::Pure, cost);
        self.new_frame(BranchTargetKind::UntakenForward);
        Ok(())
    }

    fn visit_br(&mut self, relative_depth: u32) -> Self::Output {
        let frame_idx = self.frame_index(relative_depth)?;
        let cost = self.model.visit_br(relative_depth);
        self.visit_unconditional_branch(frame_idx, cost)
    }

    fn visit_br_if(&mut self, relative_depth: u32) -> Self::Output {
        let frame_idx = self.frame_index(relative_depth)?;
        let cost = self.model.visit_br_if(relative_depth);
        self.charge_before(InstrumentationKind::PreControlFlow, cost);
        self.visit_branch(frame_idx)
    }

    fn visit_br_table(&mut self, targets: BrTable<'a>) -> Self::Output {
        let cost = self.model.visit_br_table(targets.clone());
        self.charge_before(InstrumentationKind::PreControlFlow, cost);
        for target in targets.targets() {
            let target = target.map_err(Error::ParseBrTable)?;
            self.visit_branch(self.frame_index(target)?)?;
        }
        self.visit_branch(self.frame_index(targets.default())?)?;
        self.make_polymorphic();
        Ok(())
    }

    fn visit_return(&mut self) -> Self::Output {
        let cost = self.model.visit_return();
        self.visit_unconditional_branch(self.root_frame_index(), cost)
    }

    fn visit_return_call(&mut self, function_index: u32) -> Self::Output {
        let cost = self.model.visit_return_call(function_index);
        self.visit_unconditional_branch(self.root_frame_index(), cost)
    }

    fn visit_return_call_indirect(&mut self, type_index: u32, table_index: u32) -> Self::Output {
        let cost = self
            .model
            .visit_return_call_indirect(type_index, table_index);
        self.visit_unconditional_branch(self.root_frame_index(), cost)
    }

    fn visit_memory_grow(&mut self, mem: u32, mem_byte: u8) -> Self::Output {
        let cost = self.model.visit_memory_grow(mem, mem_byte);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_memory_init(&mut self, data_index: u32, mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_init(data_index, mem);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_data_drop(&mut self, data_index: u32) -> Self::Output {
        // TODO: [] -> []; does not interact with the operand stack, so isn’t really an aggregate
        // instruction. In practice, though, it may involve non-trivial amount of work in the
        // runtime anyway? Validate.
        let cost = self.model.visit_data_drop(data_index);
        self.charge_before(InstrumentationKind::Pure, cost);
        Ok(())
    }

    fn visit_memory_copy(&mut self, dst_mem: u32, src_mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_copy(dst_mem, src_mem);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_memory_fill(&mut self, mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_fill(mem);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_table_init(&mut self, elem_index: u32, table: u32) -> Self::Output {
        let cost = self.model.visit_table_init(elem_index, table);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_elem_drop(&mut self, elem_index: u32) -> Self::Output {
        // TODO: [] -> []; does not interact with the operand stack, so isn’t really an aggregate
        // instruction. In practice, though, it may involve non-trivial amount of work in the
        // runtime anyway? Validate.
        let cost = self.model.visit_elem_drop(elem_index);
        self.charge_before(InstrumentationKind::Pure, cost);
        Ok(())
    }

    fn visit_table_copy(&mut self, dst_table: u32, src_table: u32) -> Self::Output {
        let cost = self.model.visit_table_copy(dst_table, src_table);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_table_fill(&mut self, table: u32) -> Self::Output {
        let cost = self.model.visit_table_fill(table);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_table_grow(&mut self, table: u32) -> Self::Output {
        let cost = self.model.visit_table_grow(table);
        self.charge_before(InstrumentationKind::Aggregate, cost);
        Ok(())
    }

    fn visit_try(&mut self, _: BlockType) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_catch(&mut self, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_throw(&mut self, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_rethrow(&mut self, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_delegate(&mut self, _: u32) -> Self::Output {
        todo!("exception handling extension")
    }

    fn visit_catch_all(&mut self) -> Self::Output {
        todo!("exception handling extension")
    }
}

impl<'a, CostModel: VisitOperator<'a, Output = u64>> crate::visitors::VisitOperatorWithOffset<'a>
    for GasVisitor<'a, CostModel>
{
    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }
}

/// Optimize the instruction tables.
pub(crate) fn optimize(
    offsets: &mut Vec<usize>,
    costs: &mut Vec<u64>,
    kinds: &mut Vec<InstrumentationKind>,
) {
    let mut previous_kind = kinds
        .first()
        .copied()
        .unwrap_or(InstrumentationKind::PreControlFlow);
    let mut output_idx = 0;
    for input_idx in 1..kinds.len() {
        let kind = kinds[input_idx];
        let merge_to_previous = match (kind, previous_kind) {
            (InstrumentationKind::Pure, InstrumentationKind::Pure) => true,
            (InstrumentationKind::Unreachable, InstrumentationKind::Unreachable) => true,
            (InstrumentationKind::PreControlFlow, InstrumentationKind::Pure) => true,
            (InstrumentationKind::Pure, InstrumentationKind::PostControlFlow) => true,
            _ => false,
        };
        previous_kind = kind;
        if merge_to_previous {
            kinds[output_idx] = kind;
            costs[output_idx] += costs[input_idx];
        } else {
            output_idx += 1;
            kinds[output_idx] = kind;
            costs[output_idx] = costs[input_idx];
            offsets[output_idx] = offsets[input_idx];
        }
    }
    kinds.truncate(output_idx + 1);
    costs.truncate(output_idx + 1);
    offsets.truncate(output_idx + 1);
}
