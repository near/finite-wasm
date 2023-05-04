use crate::{gas, visitors};

use super::{BranchTargetKind, Frame};

/// The configuration for the gas analysis.
///
/// Note that this trait is not intended to be implemented directly. Implement
/// [`finite_wasm::wasmparser::VisitOperator`](crate::wasmparser::VisitOperator) with `type Output
/// = u64`, where each of the `visit_*` methods return a gas cost for the specific instrution being
/// visited. Implementers of such trait will also implement `gas::Config` by definition.
pub trait Config<'b> {
    type GasVisitor<'s>: visitors::VisitOperatorWithOffset<'b, Output = Result<(), gas::Error>>
    where
        Self: 's;

    fn make_visitor<'s>(&'s mut self, state: &'s mut gas::FunctionState) -> Self::GasVisitor<'s>;
    fn save_outcomes(
        &self,
        state: &mut gas::FunctionState,
        destination: &mut crate::AnalysisOutcome,
    );
}

impl<'b> Config<'b> for crate::NoConfig {
    type GasVisitor<'s> = visitors::NoOpVisitor<Result<(), gas::Error>>;
    fn make_visitor<'s>(&'s mut self, _: &'s mut gas::FunctionState) -> Self::GasVisitor<'s> {
        visitors::NoOpVisitor(Ok(()))
    }

    fn save_outcomes(&self, _: &mut gas::FunctionState, _: &mut crate::AnalysisOutcome) {}
}

impl<'b, V: wasmparser::VisitOperator<'b, Output = u64>> Config<'b> for V {
    type GasVisitor<'s> = gas::Visitor<'s, V> where Self: 's;
    fn make_visitor<'s>(&'s mut self, state: &'s mut gas::FunctionState) -> Self::GasVisitor<'s> {
        gas::Visitor {
            offset: 0,
            model: self,
            state,
        }
    }

    fn save_outcomes(&self, state: &mut gas::FunctionState, out: &mut crate::AnalysisOutcome) {
        state.optimize();
        out.gas_offsets.push(state.offsets.drain(..).collect());
        out.gas_kinds.push(state.kinds.drain(..).collect());
        out.gas_costs.push(state.costs.drain(..).collect());
        state.frame_stack.clear();
        state.current_frame = Frame {
            stack_polymorphic: false,
            kind: BranchTargetKind::UntakenForward,
        };
        state.scheduled_instrumentation = None;
    }
}
