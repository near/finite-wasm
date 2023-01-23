use super::Error;
use super::{instruction_visit::Output, FunctionState, ModuleState, Visitor};
use crate::prefix_sum_vec::PrefixSumVec;
use crate::visitors::{self, VisitOperatorWithOffset};
use crate::wasmparser::{Type, ValType};

/// Configure size of various values that may end up on the stack.
pub trait SizeConfig {
    fn size_of_value(&self, ty: ValType) -> u8;
    fn size_of_function_activation(&self, locals: &PrefixSumVec<ValType, u32>) -> u64;
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

    fn add_function(&self, state: &mut ModuleState, type_index: u32);
    fn add_global(&self, state: &mut ModuleState, content_type: ValType);
    fn add_table(&self, state: &mut ModuleState, content_type: ValType);
    fn add_type(&self, state: &mut ModuleState, ty: Type);

    fn populate_locals(
        &self,
        module: &ModuleState,
        fn_state: &mut FunctionState,
        fn_idx: u32,
    ) -> Result<(), Error>;
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

    fn populate_locals(
        &self,
        module: &ModuleState,
        fn_state: &mut FunctionState,
        fn_idx: u32,
    ) -> Result<(), Error> {
        let function_id_usize =
            usize::try_from(fn_idx).expect("failed converting from u32 to usize");
        let type_id = *module
            .functions
            .get(function_id_usize)
            .ok_or(Error::FunctionIndex(fn_idx))?;
        let type_id_usize =
            usize::try_from(type_id).map_err(|e| Error::TypeIndexRange(type_id, e))?;
        let fn_type = module
            .types
            .get(type_id_usize)
            .ok_or(Error::TypeIndex(type_id))?;

        match fn_type {
            wasmparser::Type::Func(fnty) => {
                for param in fnty.params() {
                    fn_state.add_locals(1, *param)?;
                }
            }
        }
        Ok(())
    }

    fn add_function(&self, state: &mut ModuleState, type_index: u32) {
        state.functions.push(type_index);
    }

    fn add_global(&self, state: &mut ModuleState, content_type: ValType) {
        state.globals.push(content_type);
    }

    fn add_table(&self, state: &mut ModuleState, content_type: ValType) {
        state.tables.push(content_type);
    }

    fn add_type(&self, state: &mut ModuleState, ty: Type) {
        state.types.push(ty);
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

    fn add_function(&self, _: &mut ModuleState, _: u32) {}
    fn add_global(&self, _: &mut ModuleState, _: ValType) {}
    fn add_table(&self, _: &mut ModuleState, _: ValType) {}
    fn add_type(&self, _: &mut ModuleState, _: Type) {}
    fn populate_locals(&self, _: &ModuleState, _: &mut FunctionState, _: u32) -> Result<(), Error> {
        Ok(())
    }
}
