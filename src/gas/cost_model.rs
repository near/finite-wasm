use crate::{gas, visitors};

/// The configuration for the gas analysis.
///
/// Note that this trait is not intended to implement directly. Implement
/// [`finite_wasm::wasmparser::VisitOperator`](crate::wasmparser::VisitOperator) with `type Output
/// = u64`, where each of the `visit_*` methods return a gas cost for the specific instrution being
/// visited. Implementers of such trait will also implement `gas::Config` by definition.
pub trait Config<'b> {
    type GasVisitor<'s>: visitors::VisitOperatorWithOffset<'b, Output = Result<(), gas::Error>>
    where
        Self: 's;
    fn to_visitor<'s>(&'s mut self, state: &'s mut gas::FunctionState) -> Self::GasVisitor<'s>;
}

impl<'b> Config<'b> for crate::NoConfig {
    type GasVisitor<'s> = visitors::NoOpVisitor<Result<(), gas::Error>>;
    fn to_visitor<'s>(&'s mut self, _: &'s mut gas::FunctionState) -> Self::GasVisitor<'s> {
        visitors::NoOpVisitor(Ok(()))
    }
}

impl<'b, V: wasmparser::VisitOperator<'b, Output = u64>> Config<'b> for V {
    type GasVisitor<'s> = gas::Visitor<'s, V> where Self: 's;
    fn to_visitor<'s>(&'s mut self, state: &'s mut gas::FunctionState) -> Self::GasVisitor<'s> {
        gas::Visitor {
            offset: 0,
            model: self,
            state,
        }
    }
}
