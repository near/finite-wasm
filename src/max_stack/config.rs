use super::{instruction_visit::Output, FunctionState, ModuleState, Visitor};
use crate::prefix_sum_vec::PrefixSumVec;
use crate::visitors::{self, VisitOperatorWithOffset};
use crate::wasmparser::ValType;

/// Configure size of various values that may end up on the stack.
pub trait SizeConfig {
    fn size_of_value(&self, ty: ValType) -> u8;
    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64;
}

/// The configuration for the stack analysis.
///
/// Note that this trait is not intended to implement directly. Implement [`SizeConfig`]
/// instead. Implementers of `SizeConfig` trait will also implement `max_stack::Config` by
/// definition.
pub trait Config<'b> {
    type StackVisitor<'s>: VisitOperatorWithOffset<'b, Output = Output>
    where
        Self: 's;

    fn make_visitor<'s>(
        &'s self,
        module_state: &'s ModuleState,
        function_state: &'s mut FunctionState,
    ) -> Self::StackVisitor<'s>;

    fn save_outcomes(&self, state: &mut FunctionState, out: &mut crate::AnalysisOutcome);
}

impl<'b, S: SizeConfig> Config<'b> for S {
    type StackVisitor<'s> = Visitor<'s, Self> where Self: 's;

    fn make_visitor<'s>(
        &'s self,
        module_state: &'s ModuleState,
        function_state: &'s mut FunctionState,
    ) -> Self::StackVisitor<'s> {
        Visitor {
            offset: 0,
            config: self,
            module_state,
            function_state,
        }
    }

    fn save_outcomes(&self, state: &mut FunctionState, out: &mut crate::AnalysisOutcome) {
        out.function_frame_sizes
            .push(self.size_of_function_activation(&state.locals));
        out.function_operand_stack_sizes.push(state.max_size);
        state.clear();
    }
}

impl<'b> Config<'b> for crate::NoConfig {
    type StackVisitor<'s> = visitors::NoOpVisitor<Output>;

    fn make_visitor<'s>(
        &'s self,
        _: &'s ModuleState,
        _: &'s mut FunctionState,
    ) -> Self::StackVisitor<'s> {
        visitors::NoOpVisitor(Ok(()))
    }

    fn save_outcomes(&self, _: &mut FunctionState, _: &mut crate::AnalysisOutcome) {}
}

impl<'a, C: SizeConfig + ?Sized> SizeConfig for &'a C {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(*self, locals)
    }
}

impl<'a, C: SizeConfig + ?Sized> SizeConfig for &'a mut C {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(*self, locals)
    }
}

impl<'a, C: SizeConfig + ?Sized> SizeConfig for Box<C> {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(&*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(&*self, locals)
    }
}

impl<'a, C: SizeConfig + ?Sized> SizeConfig for std::sync::Arc<C> {
    fn size_of_value(&self, ty: wasmparser::ValType) -> u8 {
        C::size_of_value(&*self, ty)
    }

    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64 {
        C::size_of_function_activation(&*self, locals)
    }
}
