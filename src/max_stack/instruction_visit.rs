use super::{Error, SizeConfig, Visitor};
use crate::instruction_categories as gen;
use wasmparser::{BlockType, BrTable, MemArg, ValType, VisitOperator};

pub(crate) type Output = Result<(), Error>;

macro_rules! instruction_category {
    ($($type:ident . const = $($insn:ident, $param: ty)|* ;)*) => {
        $($(fn $insn(&mut self, _: $param) -> Self::Output {
            self.visit_const(ValType::$type)
        })*)*
    };
    ($($type:ident . unop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_unop()
        })*)*
    };
    ($($type:ident . binop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_binop()
        })*)*
    };
    ($($type:ident . testop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_testop()
        })*)*
    };

    ($($type:ident . relop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_relop()
        })*)*
    };

    ($($type:ident . cvtop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_cvtop(ValType::$type)
        })*)*
    };

    ($($type:ident . load = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: MemArg) -> Self::Output {
            self.visit_load(ValType::$type)
        })*)*
    };

    ($($type:ident . store = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: MemArg) -> Self::Output {
            self.visit_store()
        })*)*
    };

    ($($type:ident . loadlane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: MemArg, _: u8) -> Self::Output {
            self.visit_load_lane()
        })*)*
    };

    ($($type:ident . storelane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: MemArg, _: u8) -> Self::Output {
            self.visit_store_lane()
        })*)*
    };

    ($($type:ident . vternop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_vternop()
        })*)*
    };

    ($($type:ident . vrelop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_vrelop()
        })*)*
    };

    ($($type:ident . vishiftop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_vishiftop()
        })*)*
    };

    ($($type:ident . vinarrowop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_vinarrowop()
        })*)*
    };

    ($($type:ident . vbitmask = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_vbitmask()
        })*)*
    };

    ($($type:ident . splat = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self) -> Self::Output {
            self.visit_splat()
        })*)*
    };

    ($($type:ident . replacelane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: u8) -> Self::Output {
            self.visit_replace_lane()
        })*)*
    };

    ($($type:ident . extractlane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: u8) -> Self::Output {
            self.visit_extract_lane(ValType::$type)
        })*)*
    };

    ($($type:ident . atomic.rmw = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: MemArg) -> Self::Output {
            self.visit_atomic_rmw(ValType::$type)
        })*)*
    };

    ($($type:ident . atomic.cmpxchg = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: MemArg) -> Self::Output {
            self.visit_atomic_cmpxchg(ValType::$type)
        })*)*
    };
}

impl<'a, 'cfg, Cfg: SizeConfig> VisitOperator<'a> for Visitor<'cfg, Cfg> {
    type Output = Output;

    gen::r#const!(instruction_category);

    fn visit_ref_null(&mut self, t: ValType) -> Self::Output {
        // [] -> [t]
        self.push(t);
        Ok(())
    }

    fn visit_ref_func(&mut self, _: u32) -> Self::Output {
        self.visit_ref_null(ValType::FuncRef)
    }

    gen::unop!(instruction_category);
    gen::binop!(instruction_category);
    gen::vishiftop!(instruction_category);
    gen::testop!(instruction_category);
    gen::relop!(instruction_category);
    gen::cvtop!(instruction_category);
    gen::load!(instruction_category);
    gen::loadlane!(instruction_category);
    gen::store!(instruction_category);
    gen::storelane!(instruction_category);
    gen::replacelane!(instruction_category);
    gen::extractlane!(instruction_category);
    gen::vternop!(instruction_category);
    gen::vrelop!(instruction_category);
    gen::vinarrowop!(instruction_category);
    gen::vbitmask!(instruction_category);
    gen::splat!(instruction_category);
    gen::atomic_rmw!(instruction_category);
    gen::atomic_cmpxchg!(instruction_category);

    fn visit_i8x16_shuffle(&mut self, _: [u8; 16]) -> Self::Output {
        // i8x16.shuffle laneidx^16 : [v128 v128] → [v128]
        self.pop()?;
        Ok(())
    }

    fn visit_memory_atomic_notify(&mut self, _: MemArg) -> Self::Output {
        // [i32 i32] -> [i32]
        self.pop()?;
        Ok(())
    }

    fn visit_memory_atomic_wait32(&mut self, _: MemArg) -> Self::Output {
        // [i32 i32 i64] -> [i32]
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_memory_atomic_wait64(&mut self, _: MemArg) -> Self::Output {
        // [i32 i64 i64] -> [i32]
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_atomic_fence(&mut self) -> Self::Output {
        // https://github.com/WebAssembly/threads/blob/main/proposals/threads/Overview.md#fence-operator
        // [] -> []

        // Function body intentionally left empty
        Ok(())
    }

    fn visit_local_get(&mut self, local_index: u32) -> Self::Output {
        // [] → [t]
        let local_type = self
            .function_state
            .locals
            .get(&local_index)
            .ok_or(Error::LocalIndex(local_index))?;
        self.push(*local_type);
        Ok(())
    }

    fn visit_local_set(&mut self, _: u32) -> Self::Output {
        // [t] → []
        self.pop()?;
        Ok(())
    }

    fn visit_local_tee(&mut self, _: u32) -> Self::Output {
        // [t] → [t]
        Ok(())
    }

    fn visit_global_get(&mut self, global: u32) -> Self::Output {
        // [] → [t]
        let global_usize =
            usize::try_from(global).map_err(|e| Error::GlobalIndexRange(global, e))?;
        let global_ty = self
            .module_state
            .globals
            .get(global_usize)
            .ok_or(Error::GlobalIndex(global))?;
        self.push(*global_ty);
        Ok(())
    }

    fn visit_global_set(&mut self, _: u32) -> Self::Output {
        // [t] → []
        self.pop()?;
        Ok(())
    }

    fn visit_memory_size(&mut self, _: u32, _: u8) -> Self::Output {
        // [] → [i32]
        self.push(ValType::I32);
        Ok(())
    }

    fn visit_memory_grow(&mut self, _: u32, _: u8) -> Self::Output {
        // [i32] → [i32]

        // Function body intentionally left empty.
        Ok(())
    }

    fn visit_memory_fill(&mut self, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(())
    }

    fn visit_memory_init(&mut self, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(())
    }

    fn visit_memory_copy(&mut self, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(())
    }

    fn visit_data_drop(&mut self, _: u32) -> Self::Output {
        // [] → []
        Ok(())
    }

    fn visit_table_get(&mut self, table: u32) -> Self::Output {
        // [i32] → [t]
        let table_usize = usize::try_from(table).map_err(|e| Error::TableIndexRange(table, e))?;
        let table_ty = *self
            .module_state
            .tables
            .get(table_usize)
            .ok_or(Error::TableIndex(table))?;
        self.pop()?;
        self.push(table_ty);
        Ok(())
    }

    fn visit_table_set(&mut self, _: u32) -> Self::Output {
        // [i32 t] → []
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_table_size(&mut self, _: u32) -> Self::Output {
        // [] → [i32]
        self.push(ValType::I32);
        Ok(())
    }

    fn visit_table_grow(&mut self, _: u32) -> Self::Output {
        // [t i32] → [i32]
        self.pop_many(2)?;
        self.push(ValType::I32);
        Ok(())
    }

    fn visit_table_fill(&mut self, _: u32) -> Self::Output {
        // [i32 t i32] → []
        self.pop_many(3)?;
        Ok(())
    }

    fn visit_table_copy(&mut self, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(())
    }

    fn visit_table_init(&mut self, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(())
    }

    fn visit_elem_drop(&mut self, _: u32) -> Self::Output {
        // [] → []
        Ok(())
    }

    fn visit_select(&mut self) -> Self::Output {
        // [t t i32] -> [t]
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_typed_select(&mut self, _: ValType) -> Self::Output {
        // [t t i32] -> [t]
        self.pop_many(2)?;
        Ok(())
    }

    fn visit_drop(&mut self) -> Self::Output {
        // [t] → []
        self.pop()?;
        Ok(())
    }

    fn visit_nop(&mut self) -> Self::Output {
        // [] → []
        Ok(())
    }

    fn visit_call(&mut self, function_index: u32) -> Self::Output {
        self.visit_function_call(self.function_type_index(function_index)?)
    }

    fn visit_call_indirect(&mut self, type_index: u32, _: u32, _: u8) -> Self::Output {
        self.visit_function_call(type_index)
    }

    fn visit_return_call(&mut self, _: u32) -> Self::Output {
        // `return_call` behaves as-if a regular `return` followed by the `call`. For the purposes
        // of modelling the frame size of the _current_ function, only the `return` portion of this
        // computation is relevant (as it makes the stack polymorphic)
        self.visit_return()
    }

    fn visit_return_call_indirect(&mut self, _: u32, _: u32) -> Self::Output {
        self.visit_return()
    }

    fn visit_unreachable(&mut self) -> Self::Output {
        // [*] → [*]  (stack-polymorphic)
        self.make_polymorphic();
        Ok(())
    }

    fn visit_block(&mut self, blockty: BlockType) -> Self::Output {
        // block blocktype instr* end : [t1*] → [t2*]
        self.with_block_types(blockty, |this, params, _| {
            this.new_frame(blockty, params.len())
        })?;
        Ok(())
    }

    fn visit_loop(&mut self, blockty: BlockType) -> Self::Output {
        // loop blocktype instr* end : [t1*] → [t2*]
        self.with_block_types(blockty, |this, params, _| {
            this.new_frame(blockty, params.len())
        })?;
        Ok(())
    }

    fn visit_if(&mut self, blockty: BlockType) -> Self::Output {
        // if blocktype instr* else instr* end : [t1* i32] → [t2*]
        self.pop()?;
        self.with_block_types(blockty, |this, params, _| {
            this.new_frame(blockty, params.len())
        })?;
        Ok(())
    }

    fn visit_else(&mut self) -> Self::Output {
        if let Some(frame) = self.end_frame()? {
            self.with_block_types(frame.block_type, |this, params, _| {
                this.new_frame(frame.block_type, 0)?;
                for param in params {
                    this.push(*param);
                }
                Ok(())
            })?;
            Ok(())
        } else {
            return Err(Error::TruncatedFrameStack(self.offset));
        }
    }

    fn visit_end(&mut self) -> Self::Output {
        if let Some(frame) = self.end_frame()? {
            self.with_block_types(frame.block_type, |this, _, results| {
                Ok(for result in results {
                    this.push(*result);
                })
            })?;
            Ok(())
        } else {
            // Returning from the function. Malformed WASM modules may have trailing instructions,
            // but we do ignore processing them in the operand feed loop. For that reason,
            // replacing `top_stack` with some sentinel value would work okay.
            Ok(())
        }
    }

    fn visit_br(&mut self, _: u32) -> Self::Output {
        // [t1* t*] → [t2*]  (stack-polymorphic)
        self.make_polymorphic();
        Ok(())
    }

    fn visit_br_if(&mut self, _: u32) -> Self::Output {
        // [t* i32] → [t*]

        // There are two things that could happen here.
        //
        // First is when the condition operand is true. This instruction executed as-if it was a
        // plain `br` in this place. This won’t result in the stack size of this frame increasing
        // again. The continuation of the destination label `L` will have an arity of `n`. As part
        // of executing `br`, `n` operands are popped from the operand stack, Then a number of
        // labels/frames are popped from the stack, along with the values contained therein.
        // Finally `n` operands are pushed back onto the operand stack as the “return value” of the
        // block. As thus, executing a `(br_if (i32.const 1))` will _always_ result in a smaller
        // operand stack, and so it is uninteresting to explore this branch in isolation.
        //
        // Second is if the condition was actually false and the rest of this block is executed,
        // which can potentially later increase the size of this current frame. We’re largely
        // interested in this second case, so we don’t really need to do anything much more than…
        self.pop()?;
        // …the condition.
        Ok(())
    }

    fn visit_br_table(&mut self, _: BrTable) -> Self::Output {
        // [t1* t* i32] → [t2*]  (stack-polymorphic)
        self.make_polymorphic();
        Ok(())
    }

    fn visit_return(&mut self) -> Self::Output {
        // This behaves as-if a `br` to the outer-most block.

        // NB: self.frames.len() is actually 1 less than a number of frames, due to our maintaining
        // a `self.current_frame`.
        let branch_depth =
            u32::try_from(self.function_state.frames.len()).map_err(|_| Error::TooManyFrames)?;
        self.visit_br(branch_depth)
    }

    fn visit_try(&mut self, _: BlockType) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_rethrow(&mut self, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_throw(&mut self, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_delegate(&mut self, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_catch(&mut self, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_catch_all(&mut self) -> Self::Output {
        todo!("exception handling has not been implemented");
    }
}

impl<'a, 'cfg, Cfg: SizeConfig> crate::visitors::VisitOperatorWithOffset<'a>
    for Visitor<'cfg, Cfg>
{
    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }
}
