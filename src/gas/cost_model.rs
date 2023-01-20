use crate::{gas, visitors};

/// The configuration for the gas analysis.
///
/// Note that this trait is not intended to implement directly. Implement
/// [`finite_wasm::wasmparser::VisitOperator`](crate::wasmparser::VisitOperator) with `type Output
/// = u64`, where each of the `visit_*` methods return a gas cost for the specific instrution being
/// visited. Implementers of such trait will also implement `gas::Config` by definition.
pub trait Config<'a> {
    type GasVisitor: for<'b> visitors::VisitOperatorWithOffset<'b, Output = Result<(), gas::Error>>;
    fn to_visitor(&'a mut self, state: &'a mut gas::FunctionState) -> Self::GasVisitor;
}

/// Disable the gas analysis entirely.
pub struct NoConfig;

impl<'a> Config<'a> for NoConfig {
    type GasVisitor = visitors::NoOpVisitor<Result<(), gas::Error>>;
    fn to_visitor(&mut self, _: &mut gas::FunctionState) -> Self::GasVisitor {
        visitors::NoOpVisitor(Ok(()))
    }
}

impl<'a, V: for<'b> wasmparser::VisitOperator<'b, Output = u64> + 'a> Config<'a> for V {
    type GasVisitor = gas::Visitor<'a, V>;
    fn to_visitor(&'a mut self, state: &'a mut gas::FunctionState) -> Self::GasVisitor {
        gas::Visitor {
            offset: 0,
            model: self,
            state,
        }
    }
}
