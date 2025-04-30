mod internal {
    pub trait VisitOperatorWithOffset<'a>:
        wasmparser::VisitOperator<'a> + wasmparser::VisitSimdOperator<'a>
    {
        fn set_offset(&mut self, offset: usize);
    }

    pub struct NoOpVisitor<Output>(pub(crate) Output);
}

pub(crate) use internal::{NoOpVisitor, VisitOperatorWithOffset};

macro_rules! noop_visit {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident ($($ann:tt)*))*) => {
        $(fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output {
            self.0.clone()
        })*
    }
}

impl<'a, Output: 'static + Clone> wasmparser::VisitOperator<'a> for NoOpVisitor<Output> {
    type Output = Output;
    wasmparser::for_each_visit_operator!(noop_visit);
    fn simd_visitor(
        &mut self,
    ) -> Option<&mut dyn wasmparser::VisitSimdOperator<'a, Output = Self::Output>> {
        Some(self)
    }
}

impl<'a, Output: 'static + Clone> wasmparser::VisitSimdOperator<'a> for NoOpVisitor<Output> {
    wasmparser::for_each_visit_simd_operator!(noop_visit);
}

impl<'a, Output: 'static + Clone> VisitOperatorWithOffset<'a> for NoOpVisitor<Output> {
    fn set_offset(&mut self, _: usize) {}
}

pub(crate) struct JoinVisitor<L, R>(pub(crate) L, pub(crate) R);

macro_rules! join_visit {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident ($($ann:tt)*))*) => {
        $(fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output {
            (self.0.$visit($($($arg.clone()),*)?), self.1.$visit($($($arg),*)?))
        })*
    }
}

impl<'a, L, R> wasmparser::VisitOperator<'a> for JoinVisitor<L, R>
where
    L: wasmparser::VisitOperator<'a> + wasmparser::VisitSimdOperator<'a>,
    R: wasmparser::VisitOperator<'a> + wasmparser::VisitSimdOperator<'a>,
{
    type Output = (L::Output, R::Output);
    wasmparser::for_each_visit_operator!(join_visit);

    fn simd_visitor(
        &mut self,
    ) -> Option<&mut dyn wasmparser::VisitSimdOperator<'a, Output = Self::Output>> {
        Some(self)
    }
}

impl<'a, L, R> VisitOperatorWithOffset<'a> for JoinVisitor<L, R>
where
    L: VisitOperatorWithOffset<'a>,
    R: VisitOperatorWithOffset<'a>,
{
    fn set_offset(&mut self, offset: usize) {
        self.0.set_offset(offset);
        self.1.set_offset(offset);
    }
}

impl<'a, L, R> wasmparser::VisitSimdOperator<'a> for JoinVisitor<L, R>
where
    L: wasmparser::VisitSimdOperator<'a>,
    R: wasmparser::VisitSimdOperator<'a>,
{
    wasmparser::for_each_visit_simd_operator!(join_visit);
}
