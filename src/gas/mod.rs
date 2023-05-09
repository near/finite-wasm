//! Analysis of the amount of time (gas) a function takes to execute.
//!
//! The gas analysis is a two-pass linear-time algorithm. This algorithm first constructs a table
//! containing instrumentation offsets (represented as instructions in the example below), their
//! costs and kinds:
//!
//! <pre>
//! | Instructions | Cost | Kind |
//! | ============ | ==== | ==== |
//! | i32.const 0  | 1    | Pure |
//! | i32.const 1  | 1    | Pure |
//! | i32.add      | 1    | Pure |
//! </pre>
//!
//! In this table the instructions with certain instrumentation kind combinations can then be
//! coalesced in order to reduce the number of gas instrumentation points. For example all
//! instrumentation can be merged together across all instructions considered `Pure` to produce a
//! table like this:
//!
//! <pre>
//! | Instructions                           | Cost | Kind |
//! | ====================================== | ==== | ==== |
//! | (i32.add (i32.const 0) (i32.const 1))  | 3    | Pure |
//! </pre>
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
pub use config::Config;
pub use error::Error;
use wasmparser::{BlockType, BrTable, VisitOperator};

mod config;
mod error;
mod optimize;

/// The type of a particular instrumentation point (as denoted by its offset.)
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

    /// This instrumentation point is between two control flow instructions (see
    /// Pre/PostControlFlow).
    ///
    /// This is largely used as a bottom type in optimization, representing a case where no further
    /// optimizations involving this type can be made (important if optimization is run multiple
    /// times, for example.)
    BetweenControlFlow,

    /// This instrumentation point precedes an aggregate operation.
    ///
    /// Instructions such as `memory.fill` cause this categorization. The amount of work they do
    /// depends on the operands.
    ///
    // TODO: a variant for each such instruction may be warranted?
    Aggregate,
}

#[derive(Debug)]
pub(crate) enum BranchTargetKind {
    // A branch instruction in this frame branches backwards in the instruction stream to the
    // loop header. Instrumentation points for this loop instruction are before and after the
    // stored indices.
    Loop(usize, usize),
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

pub(crate) struct ScheduledInstrumentation {
    cost: u64,
    kind: InstrumentationKind,
}

/// The per-function state used by the [`Visitor`].
///
/// This type maintains the state accumulated during the analysis of a single function in a module.
/// If the same instance of this `FunctionState` is used to analyze multiple functions, it will
/// result in re-use of the backing allocations, and thus an improved performance.
pub struct FunctionState {
    /// Table of instruction ranges, and the total gas cost of executing the range.
    pub(crate) offsets: Vec<usize>,
    pub(crate) costs: Vec<u64>,
    pub(crate) kinds: Vec<InstrumentationKind>,

    /// Information about the analyzed function’s frame stack.
    pub(crate) frame_stack: Vec<Frame>,

    pub(crate) current_frame: Frame,

    /// Is there a charge we want to introduce "after" the current offset?
    ///
    /// Note that the implementation here depends on the fact that all instructions invoke
    /// `charge_before` or `charge_after`, even if with a 0-cost so that there is an opportunity to
    /// merge this cost into the table.
    pub(crate) scheduled_instrumentation: Option<ScheduledInstrumentation>,
}

impl FunctionState {
    /// Create a new state for the gas analysis.
    pub fn new() -> Self {
        Self {
            offsets: vec![],
            costs: vec![],
            kinds: vec![],
            frame_stack: vec![],
            current_frame: Frame {
                stack_polymorphic: false,
                kind: BranchTargetKind::UntakenForward,
            },
            scheduled_instrumentation: None,
        }
    }
}

/// The core algorihtm of the `gas` analysis.
pub struct Visitor<'s, CostModel> {
    pub(crate) offset: usize,

    /// A visitor that produces costs for instructions.
    pub(crate) model: &'s mut CostModel,

    /// Per-function visitor state.
    ///
    /// This state allocates data intermediate results during the function analysis and ultimately
    /// then drains it into summarized data. As thus, this state can be reused between functions
    /// for better performance.
    pub(crate) state: &'s mut FunctionState,
}

impl<'a, CostModel> Visitor<'a, CostModel> {
    /// Charge fees for a pure instruction.
    ///
    /// Pure instructions do not participate in control flow, have no side effects and execute in
    /// roughly a known amount of time (i.e. their execution time is largely independent of the
    /// inputs.)
    fn visit_pure_instruction(&mut self, cost: u64) {
        self.push_instrumentation_before(InstrumentationKind::Pure, cost)
    }

    /// Charge fees before executing a side-effectful instruction.
    ///
    /// Side effectful instructions are those that are known to execute in roughly a known amount
    /// of time, but may branch, call into a host function or execute some other side effect.
    /// Instructions that are potential branch targets are not applicable.
    fn visit_side_effect_instruction(&mut self, cost: u64) {
        self.push_instrumentation_before(InstrumentationKind::PreControlFlow, cost);
        self.push_instrumentation_after(InstrumentationKind::PostControlFlow, 0);
    }

    /// Charge fees before executing an aggregate instruction.
    ///
    /// Aggregate instructions are those, whose execution time is proportional to the amplitude or
    /// number of the inputs it consumes. These instructions may be side-effectful (see
    /// [`Self::visit_side_effect`].)
    fn visit_aggregate_instruction(&mut self, cost: u64) {
        self.push_instrumentation_before(InstrumentationKind::Aggregate, cost);
        self.push_instrumentation_after(InstrumentationKind::PostControlFlow, 0);
    }

    /// Charge some gas before the currently analyzed instruction.
    fn push_instrumentation_before(&mut self, kind: InstrumentationKind, cost: u64) {
        let kind = if self.state.current_frame.stack_polymorphic {
            InstrumentationKind::Unreachable
        } else {
            kind
        };
        self.state.offsets.push(self.offset);
        self.state.kinds.push(kind);
        self.state.costs.push(cost);
    }

    /// Charge some gas after the currently analyzed instruction.
    ///
    /// Note that this method works by enqueueing a charge to be added to the tables at a next call
    /// of the `charge_before` or `charge_after` function.
    fn push_instrumentation_after(&mut self, kind: InstrumentationKind, cost: u64) {
        assert!(self
            .state
            .scheduled_instrumentation
            .replace(ScheduledInstrumentation { cost, kind })
            .is_none());
    }

    /// Create a new frame on the frame stack.
    ///
    /// The caller is responsible for specifying how the branches behave when branched to this
    /// frame.
    fn new_frame(&mut self, kind: BranchTargetKind) {
        let stack_polymorphic = self.state.current_frame.stack_polymorphic;
        self.state.frame_stack.push(std::mem::replace(
            &mut self.state.current_frame,
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
        if let Some(frame) = self.state.frame_stack.pop() {
            self.state.current_frame = frame;
        }
    }

    /// Mark the current frame as polymorphic.
    fn make_polymorphic(&mut self) {
        self.state.current_frame.stack_polymorphic = true;
    }

    /// The index of the root frame (that is the one representing the function entry.)
    fn root_frame_index(&self) -> usize {
        // NB: this implicitly is 1-less than the number of frames due to us maintaining a
        // `current_frame` field.
        self.state.frame_stack.len()
    }

    fn frame_index(&self, relative_depth: u32) -> Result<usize, Error> {
        usize::try_from(relative_depth).map_err(|_| Error::BranchDepthTooLarge(self.offset))
    }

    fn adjust_branch_target(&mut self, frame_index: usize) -> Result<(), Error> {
        let frame = if let Some(frame_stack_index) = frame_index.checked_sub(1) {
            self.state
                .frame_stack
                .iter_mut()
                .nth_back(frame_stack_index)
                .ok_or(Error::InvalidBrTarget(self.offset))?
        } else {
            &mut self.state.current_frame
        };
        match frame.kind {
            BranchTargetKind::Forward => (),
            BranchTargetKind::UntakenForward => frame.kind = BranchTargetKind::Forward,
            BranchTargetKind::Loop(pre_index, post_index) => {
                self.state.kinds[post_index] = InstrumentationKind::PostControlFlow;
                self.state.kinds[pre_index] = InstrumentationKind::PreControlFlow;
            }
        }
        Ok(())
    }

    fn visit_conditional_branch(&mut self, frame_index: usize, cost: u64) -> Result<(), Error> {
        self.visit_side_effect_instruction(cost);
        self.adjust_branch_target(frame_index)?;
        Ok(())
    }

    fn visit_unconditional_branch(&mut self, frame_index: usize, cost: u64) -> Result<(), Error> {
        self.visit_conditional_branch(frame_index, cost)?;
        self.make_polymorphic();
        Ok(())
    }
}

macro_rules! trapping_insn {
    (fn $visit:ident( $($arg:ident: $ty:ty),* )) => {
        fn $visit(&mut self, $($arg: $ty),*) -> Self::Output {
            let cost = self.model.$visit($($arg),*);
            Ok(self.visit_side_effect_instruction(cost))
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
            Ok(self.visit_pure_instruction(cost))
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

impl<'a, 'b, CostModel: VisitOperator<'b, Output = u64>> VisitOperator<'b>
    for Visitor<'a, CostModel>
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
    trapping_insn!(fn visit_call_ref(ht: wasmparser::HeapType));
    trapping_insn!(fn visit_call_indirect(ty_index: u32, table_index: u32, table_byte: u8));
    // TODO: double check if these may actually trap
    trapping_insn!(fn visit_memory_atomic_notify(mem: wasmparser::MemArg));
    trapping_insn!(fn visit_memory_atomic_wait32(mem: wasmparser::MemArg));
    trapping_insn!(fn visit_memory_atomic_wait64(mem: wasmparser::MemArg));
    trapping_insn!(fn visit_table_set(table: u32));
    trapping_insn!(fn visit_table_get(table: u32));
    trapping_insn!(fn visit_ref_as_non_null());

    fn visit_unreachable(&mut self) -> Self::Output {
        let cost = self.model.visit_unreachable();
        self.push_instrumentation_before(InstrumentationKind::PreControlFlow, cost);
        self.push_instrumentation_after(InstrumentationKind::Unreachable, 0);
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

    pure_insn!(fn visit_ref_null(t: wasmparser::HeapType));
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
        // For the time being this instruction is not a branch target, and therefore is pure.
        // However, we must charge for it _after_ it has been executed, just in case it becomes a
        // branch target later. That's because as per the WebAssembly specification, the `loop`
        // instruction is executed on every iteration.
        let instrumentation_kind_index_pre = self.state.kinds.len();
        self.push_instrumentation_before(InstrumentationKind::Pure, 0);
        let instrumentation_kind_index_post = self.state.kinds.len();
        self.push_instrumentation_after(InstrumentationKind::Pure, cost);
        // This instruction will become a branch target if there is a branching instruction
        // targetting the frame created by this instruction. At that point we will make a point of
        // adjusting the instruction kind to a `InstrumentationKind::PostControlFlow`.
        self.new_frame(BranchTargetKind::Loop(
            instrumentation_kind_index_pre,
            instrumentation_kind_index_post,
        ));
        Ok(())
    }

    // Branch Target (for if, block, else), only if there is a `br`/`br_if`/`br_table` to exactly
    // the frame created by the matching insn.
    fn visit_end(&mut self) -> Self::Output {
        let cost = self.model.visit_end();
        assert!(
            cost == 0,
            "the `end` instruction costs aren’t handled right, set it to 0"
        );

        // TODO: this needs to note if this is a `if..end` or `if..else..end` branch. In the case
        // of the former, the code consuming analysis results needs to know to generate the `else`
        // branch and propagate the gas cost upwards to both branches.
        //
        // Fixing this would allow us to remove the `assert!` here and in the `visit_else`.
        //
        // Note that we cannot `charge_after` here because `end` is not "executed" when a branching
        // instruction within the frame is executed.
        match self.state.current_frame.kind {
            BranchTargetKind::Forward => self.visit_side_effect_instruction(cost),
            BranchTargetKind::Loop(_, _) => self.visit_pure_instruction(cost),
            BranchTargetKind::UntakenForward => self.visit_pure_instruction(cost),
        }
        self.end_frame();
        Ok(())
    }

    // Branch
    fn visit_if(&mut self, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_if(blockty);
        self.visit_side_effect_instruction(cost);
        // `if` is already a branch instruction, it can execute the instruction that follows (i.e.
        // acting just like a pure instruction), or it could jump to the `else` (or `end`)
        // instruction that terminates this frame.
        self.new_frame(BranchTargetKind::Forward);
        Ok(())
    }

    // Branch Target (unconditionally)
    fn visit_else(&mut self) -> Self::Output {
        let cost = self.model.visit_else();
        assert!(
            cost == 0,
            "the `else` instruction costs aren’t handled right, set it to 0"
        );
        // `else` is already a taken branch target from `if` (if the condition is false).
        self.end_frame();
        // `else` is both a branch and a branch target, depending on how it was reached.
        //
        // If `else` ends up being executed from the truthy body of the `if..else..end` block, then
        // this acts like an unconditional branch to the `end` instruction associated with this
        // frame.
        //
        // Whenever `if` condition is falsy, `else` is instead a branch target for the `if` to
        // branch to.
        self.new_frame(BranchTargetKind::Forward);
        self.visit_side_effect_instruction(cost);
        Ok(())
    }

    fn visit_block(&mut self, blockty: BlockType) -> Self::Output {
        let cost = self.model.visit_block(blockty);
        self.visit_pure_instruction(cost);
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
        self.visit_conditional_branch(frame_idx, cost)
    }

    fn visit_br_on_null(&mut self, relative_depth: u32) -> Self::Output {
        let frame_idx = self.frame_index(relative_depth)?;
        let cost = self.model.visit_br_on_null(relative_depth);
        self.visit_conditional_branch(frame_idx, cost)
    }

    fn visit_br_on_non_null(&mut self, relative_depth: u32) -> Self::Output {
        let frame_idx = self.frame_index(relative_depth)?;
        let cost = self.model.visit_br_on_non_null(relative_depth);
        self.visit_conditional_branch(frame_idx, cost)
    }

    fn visit_br_table(&mut self, targets: BrTable<'b>) -> Self::Output {
        let cost = self.model.visit_br_table(targets.clone());
        self.visit_side_effect_instruction(cost);
        for target in targets.targets() {
            let target = target.map_err(Error::ParseBrTable)?;
            self.adjust_branch_target(self.frame_index(target)?)?;
        }
        self.adjust_branch_target(self.frame_index(targets.default())?)?;
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

    fn visit_return_call_ref(&mut self, ht: wasmparser::HeapType) -> Self::Output {
        let cost = self.model.visit_return_call_ref(ht);
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
        Ok(self.visit_aggregate_instruction(cost))
    }

    fn visit_memory_init(&mut self, data_index: u32, mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_init(data_index, mem);
        Ok(self.visit_aggregate_instruction(cost))
    }

    fn visit_data_drop(&mut self, data_index: u32) -> Self::Output {
        // TODO: [] -> []; does not interact with the operand stack, so isn’t really an aggregate
        // instruction. In practice, though, it may involve non-trivial amount of work in the
        // runtime anyway? Validate.
        let cost = self.model.visit_data_drop(data_index);
        Ok(self.visit_pure_instruction(cost))
    }

    fn visit_memory_copy(&mut self, dst_mem: u32, src_mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_copy(dst_mem, src_mem);
        Ok(self.visit_aggregate_instruction(cost))
    }

    fn visit_memory_fill(&mut self, mem: u32) -> Self::Output {
        let cost = self.model.visit_memory_fill(mem);
        Ok(self.visit_aggregate_instruction(cost))
    }

    fn visit_table_init(&mut self, elem_index: u32, table: u32) -> Self::Output {
        let cost = self.model.visit_table_init(elem_index, table);
        Ok(self.visit_aggregate_instruction(cost))
    }

    fn visit_elem_drop(&mut self, elem_index: u32) -> Self::Output {
        // TODO: [] -> []; does not interact with the operand stack, so isn’t really an aggregate
        // instruction. In practice, though, it may involve non-trivial amount of work in the
        // runtime anyway? Validate.
        let cost = self.model.visit_elem_drop(elem_index);
        Ok(self.visit_pure_instruction(cost))
    }

    fn visit_table_copy(&mut self, dst_table: u32, src_table: u32) -> Self::Output {
        let cost = self.model.visit_table_copy(dst_table, src_table);
        Ok(self.visit_aggregate_instruction(cost))
    }

    fn visit_table_fill(&mut self, table: u32) -> Self::Output {
        let cost = self.model.visit_table_fill(table);
        Ok(self.visit_aggregate_instruction(cost))
    }

    fn visit_table_grow(&mut self, table: u32) -> Self::Output {
        let cost = self.model.visit_table_grow(table);
        Ok(self.visit_aggregate_instruction(cost))
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

    fn visit_memory_discard(&mut self, _: u32) -> Self::Output {
        todo!("memory control extension")
    }
}

impl<'a, 'b, CostModel: VisitOperator<'b, Output = u64>>
    crate::visitors::VisitOperatorWithOffset<'b> for Visitor<'a, CostModel>
{
    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
        if let Some(scheduled) = self.state.scheduled_instrumentation.take() {
            self.state.offsets.push(self.offset);
            self.state.kinds.push(scheduled.kind);
            self.state.costs.push(scheduled.cost);
        }
    }
}
