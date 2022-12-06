pub(crate) trait VisitOperatorWithOffset<'a>: wasmparser::VisitOperator<'a> {
    fn set_offset(&mut self, offset: usize);
}

pub(crate) struct NoOpVisitor<Output>(pub(crate) Output);

macro_rules! noop_visit {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
        $(fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output {
            self.0.clone()
        })*
    }
}

impl<'a, Output: 'static + Clone> wasmparser::VisitOperator<'a> for NoOpVisitor<Output> {
    type Output = Output;
    wasmparser::for_each_operator!(noop_visit);
}

impl<'a, Output: 'static + Clone> VisitOperatorWithOffset<'a> for NoOpVisitor<Output> {
    fn set_offset(&mut self, _: usize) {}
}

pub(crate) struct JoinVisitor<'a, L: ?Sized, R: ?Sized>(pub(crate) &'a mut L, pub(crate) &'a mut R);

macro_rules! join_visit {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
        $(fn $visit(&mut self $($(,$arg: $argty)*)?) -> Self::Output {
            (self.0.$visit($($($arg.clone()),*)?), self.1.$visit($($($arg),*)?))
        })*
    }
}

impl<'a, L, R> wasmparser::VisitOperator<'a> for JoinVisitor<'a, L, R>
where
    L: wasmparser::VisitOperator<'a> + ?Sized,
    R: wasmparser::VisitOperator<'a> + ?Sized,
{
    type Output = (L::Output, R::Output);
    wasmparser::for_each_operator!(join_visit);
}

impl<'a, L, R> VisitOperatorWithOffset<'a> for JoinVisitor<'a, L, R>
where
    L: VisitOperatorWithOffset<'a> + ?Sized,
    R: VisitOperatorWithOffset<'a> + ?Sized,
{
    fn set_offset(&mut self, offset: usize) {
        self.0.set_offset(offset);
        self.1.set_offset(offset);
    }
}
