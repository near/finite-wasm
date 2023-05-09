//! Optimize the instrumentation tables to minimize number of instrumentation points.
//!
//! Each instrumentation point means slower execution as the VM needs to spend time executing gas
//! accounting instead of doing useful work. This module implements optimization of the instruction
//! tables produced by [`super::Visitor`] on a per-function basis.
//!
//! There are a few cases that we need to worry about:
//!
//! * There are multiple instrumentation points at the same offset.
//!
//!   This can happen when two side-effectful instructions follow each other. Tables then can end
//!   up with a situation like this:
//!
//!   ```text
//!   - PreSideEffect
//!   {side effect}
//!   - PostSideEffect
//!   - PreSideEffect
//!   {side effect}
//!   - PostSideEffect
//!   ```
//!
//!   The `PostSideEffect` and `PreSideEffect` that appear at the same offset can be merged
//!   together reducing the number of instrumentation points by one.
//!
//! * We can also sometimes merge multiple instrumentation points _across_ instructions that have
//!   certain properties. Of particular interest would be runs of pure instructions which have no
//!   observable side-effects. This means that however the execution proceeds, the user will not be
//!   able to observe if these instructions have executed or not.
//!
//!   That means we can move any gas instrumentation points within continuous runs of such pure
//!   instructions to the beginning of a block, charging gas only once.

use super::InstrumentationKind;

pub(crate) struct InstrumentationPoint {
    offset: usize,
    kind: InstrumentationKind,
    cost: u64,
}

impl InstrumentationKind {
    /// Instrumentation kind after merging two instrumentation points at the same offset.
    fn merge_same_point(self, other: Self) -> Option<Self> {
        use InstrumentationKind::*;
        // FIXME(#49): make this infallible to guarantee single instrumentation point per offset.
        Some(match (self, other) {
            (Unreachable, _) => Unreachable,
            (_, Unreachable) => Unreachable,
            (Aggregate, _) => return None,
            (_, Aggregate) => return None,
            (Pure, Pure | PreControlFlow | PostControlFlow | BetweenControlFlow) => other,
            (PreControlFlow, PreControlFlow | Pure) => PreControlFlow,
            (PreControlFlow, PostControlFlow | BetweenControlFlow) => BetweenControlFlow,
            (PostControlFlow, PostControlFlow | Pure) => PostControlFlow,
            (PostControlFlow, PreControlFlow | BetweenControlFlow) => BetweenControlFlow,
            (BetweenControlFlow, Pure | PreControlFlow | PostControlFlow | BetweenControlFlow) => {
                BetweenControlFlow
            }
        })
    }

    /// Instrumentation kind after merging two instrumentation ranges separaated by an instruction.
    fn merge_across_instructions(self, other: Self) -> Option<Self> {
        // Implementation note: this should pretty much never return a “weaker” kind than either of
        // the inputs. That is, merging `Pure` with any other kind should never return a `Pure`,
        // because that might make it possible to merge two kinds that otherwise would never be
        // mergeable.
        use InstrumentationKind::*;
        match (self, other) {
            // The changes to the remaining gas pool are not observable across a pure instruction.
            (Pure, Pure) => Some(Pure),
            // The unreachable code is never executed.
            (Unreachable, Unreachable) => Some(Unreachable),
            // Two control flow operators meet here.
            (PostControlFlow, PreControlFlow) => Some(BetweenControlFlow),
            // The next instruction is about to be a control-flow instruction. We can still merge
            // into this instrumentation point from a previous, pure instrumentation point.
            (Pure, PreControlFlow) => Some(PreControlFlow),
            // The previous instrumentation point comes after a control-flow instruction. The
            // current instruction is pure, though.
            (PostControlFlow, Pure) => Some(PostControlFlow),
            _ => None,
        }
    }
}

impl super::FunctionState {
    pub(crate) fn instrumentation_count(&self) -> usize {
        let Self {
            kinds,
            offsets,
            costs,
            ..
        } = self;
        debug_assert!(kinds.len() == offsets.len() && kinds.len() == costs.len());
        kinds.len()
    }

    pub(crate) fn instrumentation_at(&self, index: usize) -> Option<InstrumentationPoint> {
        Some(InstrumentationPoint {
            kind: *self.kinds.get(index)?,
            offset: *self.offsets.get(index)?,
            cost: *self.costs.get(index)?,
        })
    }

    pub(crate) fn set_instrumentation_point(&mut self, index: usize, point: InstrumentationPoint) {
        self.kinds[index] = point.kind;
        self.costs[index] = point.cost;
        self.offsets[index] = point.offset;
    }

    pub(crate) fn optimize_with(
        &mut self,
        merge_point: impl Fn(
            &InstrumentationPoint,
            &InstrumentationPoint,
        ) -> Option<InstrumentationPoint>,
    ) {
        // NB: If indexing turns out to be expensive for whatever reason, it could probably be
        // optimized by replacing indexing with pointer pairs.
        let mut output_idx = 0;
        let mut prev_point = match self.instrumentation_at(0) {
            Some(point) => point,
            None => return,
        };
        for input_idx in 1..self.instrumentation_count() {
            let current_point = self
                .instrumentation_at(input_idx)
                .expect("should always succeed");
            if let Some(merged_point) = merge_point(&prev_point, &current_point) {
                prev_point = merged_point;
            } else {
                self.set_instrumentation_point(output_idx, prev_point);
                prev_point = current_point;
                output_idx += 1;
            }
        }
        self.set_instrumentation_point(output_idx, prev_point);
        output_idx += 1;
        self.kinds.truncate(output_idx);
        self.costs.truncate(output_idx);
        self.offsets.truncate(output_idx);
    }

    /// Optimize the instrumentation tables.
    ///
    /// This reduces the number of entries in the supplied vectors while preserving the equivalence
    /// of observable (instrumented) program behaviour.
    pub(crate) fn optimize(&mut self) {
        // First: merge all instrumentation points at the same offset. This is always possible, and
        // may enable slightly better optimizations, as the less sophisticated merger that works
        // across instructions will only need to deal with one kind of instrumentation.
        self.optimize_with(|prev, next| {
            if prev.offset == next.offset {
                Some(InstrumentationPoint {
                    offset: prev.offset,
                    cost: prev.cost.checked_add(next.cost)?,
                    kind: prev.kind.merge_same_point(next.kind)?,
                })
            } else {
                None
            }
        });
        // Then, merge instrumentation points across instructions to construct conceptually ranges
        // of instructions covered by a single instrumentation point.
        self.optimize_with(|prev, next| {
            Some(InstrumentationPoint {
                offset: std::cmp::min(prev.offset, next.offset),
                kind: prev.kind.merge_across_instructions(next.kind)?,
                cost: prev.cost.checked_add(next.cost)?,
            })
        });
    }
}
