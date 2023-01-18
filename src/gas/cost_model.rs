/// The cost model for the gas analysis.
///
/// Note that this trait is not particularly useful to implement directly. Implement
/// [`finite_wasm::wasmparser::VisitOperator`](crate::wasmparser::VisitOperator) with `type Output
/// = u64`, where each of the `visit_*` methods return a gas cost for the specific instrution being
/// visited. Implementers of this trait will also implement `CostModel` by definition.
pub trait CostModel<'a> {
    type Visitor: wasmparser::VisitOperator<'a, Output = u64>;
    fn to_visitor(&mut self) -> Option<&mut Self::Visitor>;
}

macro_rules! define_unreachable_visit {
    ($(@$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
        $(
            fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output { loop {} }
        )*
    }
}

pub enum NoVisitor {}
impl<'a> wasmparser::VisitOperator<'a> for NoVisitor {
    type Output = u64;
    wasmparser::for_each_operator!(define_unreachable_visit);
}

/// No gas cost model is provided, meaning the gas analysis will not run at all.
pub struct NoCostModel;
impl<'a> CostModel<'a> for NoCostModel {
    type Visitor = NoVisitor;
    fn to_visitor(&mut self) -> Option<&mut Self::Visitor> {
        None
    }
}

impl<'a, V: wasmparser::VisitOperator<'a, Output = u64>> CostModel<'a> for V {
    type Visitor = V;
    fn to_visitor(&mut self) -> Option<&mut Self::Visitor> {
        Some(self)
    }
}
