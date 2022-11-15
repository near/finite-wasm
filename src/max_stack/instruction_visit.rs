use super::{Config, Error, Frame, StackSizeVisitor};
use wasmparser::{BlockType, BrTable, Ieee32, Ieee64, MemArg, ValType, VisitOperator, V128};

macro_rules! instruction_category {
    ($($type:ident . const = $($insn:ident, $param: ty)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: $param) -> Self::Output {
            self.visit_const(ValType::$type)
        })*)*
    };
    ($($type:ident . unop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_unop()
        })*)*
    };
    ($($type:ident . binop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_binop()
        })*)*
    };
    ($($type:ident . testop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_testop()
        })*)*
    };

    ($($type:ident . relop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_relop()
        })*)*
    };

    ($($type:ident . cvtop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_cvtop(ValType::$type)
        })*)*
    };

    ($($type:ident . load = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: MemArg) -> Self::Output {
            self.visit_load(ValType::$type)
        })*)*
    };

    ($($type:ident . store = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: MemArg) -> Self::Output {
            self.visit_store()
        })*)*
    };

    ($($type:ident . loadlane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
            self.visit_load_lane()
        })*)*
    };

    ($($type:ident . storelane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: MemArg, _: u8) -> Self::Output {
            self.visit_store_lane()
        })*)*
    };

    ($($type:ident . vternop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_vternop()
        })*)*
    };

    ($($type:ident . vrelop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_vrelop()
        })*)*
    };

    ($($type:ident . vishiftop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_vishiftop()
        })*)*
    };

    ($($type:ident . vinarrowop = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_vinarrowop()
        })*)*
    };

    ($($type:ident . vbitmask = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_vbitmask()
        })*)*
    };

    ($($type:ident . splat = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize) -> Self::Output {
            self.visit_splat()
        })*)*
    };

    ($($type:ident . replacelane = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: u8) -> Self::Output {
            self.visit_replace_lane()
        })*)*
    };

    ($($type:ident . extractlane = $($insn:ident to $unpacked:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: u8) -> Self::Output {
            self.visit_extract_lane(ValType::$unpacked)
        })*)*
    };

    ($($type:ident . atomic.rmw = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: MemArg) -> Self::Output {
            self.visit_atomic_rmw(ValType::$type)
        })*)*
    };

    ($($type:ident . atomic.cmpxchg = $($insn:ident)|* ;)*) => {
        $($(fn $insn(&mut self, _: usize, _: MemArg) -> Self::Output {
            self.visit_atomic_cmpxchg(ValType::$type)
        })*)*
    };
}

impl<'a, 'cfg, Cfg: Config> VisitOperator<'a> for StackSizeVisitor<'cfg, Cfg> {
    type Output = Result<Option<u64>, Error>;

    instruction_category! {
        I32.const = visit_i32_const, i32;
        I64.const = visit_i64_const, i64;
        F32.const = visit_f32_const, Ieee32;
        F64.const = visit_f64_const, Ieee64;
        V128.const = visit_v128_const, V128;
    }

    fn visit_ref_null(&mut self, _: usize, t: ValType) -> Self::Output {
        // [] -> [t]
        self.push(t);
        Ok(None)
    }

    fn visit_ref_func(&mut self, offset: usize, _: u32) -> Self::Output {
        self.visit_ref_null(offset, ValType::FuncRef)
    }

    instruction_category! {
        I32.unop = visit_i32_clz | visit_i32_ctz | visit_i32_popcnt;

        I64.unop = visit_i64_clz | visit_i64_ctz | visit_i64_popcnt;

        F32.unop = visit_f32_abs | visit_f32_neg | visit_f32_sqrt | visit_f32_ceil
                 | visit_f32_floor | visit_f32_trunc | visit_f32_nearest;

        F64.unop = visit_f64_abs | visit_f64_neg | visit_f64_sqrt | visit_f64_ceil
                 | visit_f64_floor | visit_f64_trunc | visit_f64_nearest;

        V128.unop = visit_v128_not;

        V128.unop = visit_i8x16_abs | visit_i8x16_neg | visit_i8x16_popcnt;
        V128.unop = visit_i16x8_abs | visit_i16x8_neg;
        V128.unop = visit_i32x4_abs | visit_i32x4_neg;
        V128.unop = visit_i64x2_abs | visit_i64x2_neg;

        V128.unop = visit_f32x4_abs | visit_f32x4_neg | visit_f32x4_sqrt | visit_f32x4_ceil
                  | visit_f32x4_floor | visit_f32x4_trunc | visit_f32x4_nearest;
        V128.unop = visit_f64x2_abs | visit_f64x2_neg | visit_f64x2_sqrt | visit_f64x2_ceil
                  | visit_f64x2_floor | visit_f64x2_trunc | visit_f64x2_nearest;

        // ishape1.extadd_pairwise_ishape2_sx : [v128] → [v128]
        V128.unop = visit_i16x8_extadd_pairwise_i8x16_s | visit_i16x8_extadd_pairwise_i8x16_u;
        V128.unop = visit_i32x4_extadd_pairwise_i16x8_s | visit_i32x4_extadd_pairwise_i16x8_u;
    }

    instruction_category! {
        I32.binop = visit_i32_add | visit_i32_sub | visit_i32_mul
                  | visit_i32_div_s | visit_i32_div_u
                  | visit_i32_rem_s | visit_i32_rem_u
                  | visit_i32_and | visit_i32_or | visit_i32_xor | visit_i32_shl
                  | visit_i32_shr_s | visit_i32_shr_u
                  | visit_i32_rotl | visit_i32_rotr;
        I64.binop = visit_i64_add | visit_i64_sub | visit_i64_mul
                  | visit_i64_div_s | visit_i64_div_u
                  | visit_i64_rem_s | visit_i64_rem_u
                  | visit_i64_and | visit_i64_or | visit_i64_xor | visit_i64_shl
                  | visit_i64_shr_s | visit_i64_shr_u
                  | visit_i64_rotl | visit_i64_rotr;
        F32.binop = visit_f32_add | visit_f32_sub | visit_f32_mul
                  | visit_f32_div | visit_f32_min | visit_f32_max | visit_f32_copysign;
        F64.binop = visit_f64_add | visit_f64_sub | visit_f64_mul
                  | visit_f64_div | visit_f64_min | visit_f64_max | visit_f64_copysign;

        V128.binop = visit_v128_and | visit_v128_andnot | visit_v128_or | visit_v128_xor;

        V128.binop = visit_i8x16_add | visit_i8x16_sub;
        V128.binop = visit_i16x8_add | visit_i16x8_sub | visit_i16x8_mul;
        V128.binop = visit_i32x4_add | visit_i32x4_sub | visit_i32x4_mul;
        V128.binop = visit_i64x2_add | visit_i64x2_sub | visit_i64x2_mul;

        V128.binop = visit_f32x4_add | visit_f32x4_sub | visit_f32x4_mul | visit_f32x4_div
                   | visit_f32x4_min | visit_f32x4_max | visit_f32x4_pmin | visit_f32x4_pmax
                   | visit_f32x4_relaxed_min | visit_f32x4_relaxed_max;
        V128.binop = visit_f64x2_add | visit_f64x2_sub | visit_f64x2_mul | visit_f64x2_div
                   | visit_f64x2_min | visit_f64x2_max | visit_f64x2_pmin | visit_f64x2_pmax
                   | visit_f64x2_relaxed_min | visit_f64x2_relaxed_max;

        V128.binop = visit_i8x16_min_s | visit_i8x16_min_u | visit_i8x16_max_s | visit_i8x16_max_u;
        V128.binop = visit_i16x8_min_s | visit_i16x8_min_u | visit_i16x8_max_s | visit_i16x8_max_u;
        V128.binop = visit_i32x4_min_s | visit_i32x4_min_u | visit_i32x4_max_s | visit_i32x4_max_u;

        V128.binop = visit_i8x16_add_sat_s | visit_i8x16_add_sat_u
                   | visit_i8x16_sub_sat_s | visit_i8x16_sub_sat_u;
        V128.binop = visit_i16x8_add_sat_s | visit_i16x8_add_sat_u
                   | visit_i16x8_sub_sat_s | visit_i16x8_sub_sat_u;

        V128.binop = visit_i8x16_avgr_u;
        V128.binop = visit_i16x8_avgr_u | visit_i16x8_q15mulr_sat_s | visit_i16x8_relaxed_q15mulr_s;

        // ishape1.dot_ishape2_s : [v128 v128] → [v128]
        V128.binop = visit_i32x4_dot_i16x8_s;
        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-integer-dot-product
        V128.binop = visit_i16x8_dot_i8x16_i7x16_s;


        // ishape1.extmul_half_ishape2_sx : [v128 v128] → [v128]
        V128.binop = visit_i16x8_extmul_low_i8x16_s | visit_i16x8_extmul_high_i8x16_s
                   | visit_i16x8_extmul_low_i8x16_u | visit_i16x8_extmul_high_i8x16_u;
        V128.binop = visit_i32x4_extmul_low_i16x8_s | visit_i32x4_extmul_high_i16x8_s
                   | visit_i32x4_extmul_low_i16x8_u | visit_i32x4_extmul_high_i16x8_u;
        V128.binop = visit_i64x2_extmul_low_i32x4_s | visit_i64x2_extmul_high_i32x4_s
                   | visit_i64x2_extmul_low_i32x4_u | visit_i64x2_extmul_high_i32x4_u;

        // i8x16.swizzle : [v128 v128] → [v128]
        // i8x16.relaxed_swizzle : [v128 v128] → [v128]
        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-swizzle
        V128.binop = visit_i8x16_swizzle | visit_i8x16_relaxed_swizzle;
    }

    instruction_category! {
        V128.vishiftop = visit_i8x16_shl | visit_i8x16_shr_s | visit_i8x16_shr_u;
        V128.vishiftop = visit_i16x8_shl | visit_i16x8_shr_s | visit_i16x8_shr_u;
        V128.vishiftop = visit_i32x4_shl | visit_i32x4_shr_s | visit_i32x4_shr_u;
        V128.vishiftop = visit_i64x2_shl | visit_i64x2_shr_s | visit_i64x2_shr_u;
    }

    instruction_category! {
        I32.testop = visit_i32_eqz;
        I64.testop = visit_i64_eqz;
        V128.testop = visit_v128_any_true
                    | visit_i8x16_all_true | visit_i16x8_all_true
                    | visit_i32x4_all_true | visit_i64x2_all_true;
        FuncRef.testop = visit_ref_is_null;
    }

    instruction_category! {
        I32.relop = visit_i32_eq | visit_i32_ne
                  | visit_i32_lt_s | visit_i32_lt_u | visit_i32_gt_s | visit_i32_gt_u
                  | visit_i32_le_s | visit_i32_le_u | visit_i32_ge_s | visit_i32_ge_u;
        I64.relop = visit_i64_eq | visit_i64_ne
                  | visit_i64_lt_s | visit_i64_lt_u | visit_i64_gt_s | visit_i64_gt_u
                  | visit_i64_le_s | visit_i64_le_u | visit_i64_ge_s | visit_i64_ge_u;
        F32.relop = visit_f32_eq | visit_f32_ne
                  | visit_f32_lt | visit_f32_gt | visit_f32_le | visit_f32_ge;
        F64.relop = visit_f64_eq | visit_f64_ne
                  | visit_f64_lt | visit_f64_gt | visit_f64_le | visit_f64_ge;
    }

    instruction_category! {
        I32.cvtop = visit_i32_wrap_i64
                  | visit_i32_extend8_s | visit_i32_extend16_s
                  | visit_i32_trunc_f32_s | visit_i32_trunc_f32_u
                  | visit_i32_trunc_f64_s | visit_i32_trunc_f64_u
                  | visit_i32_trunc_sat_f32_s | visit_i32_trunc_sat_f32_u
                  | visit_i32_trunc_sat_f64_s | visit_i32_trunc_sat_f64_u
                  | visit_i32_reinterpret_f32;

        I64.cvtop = visit_i64_extend8_s | visit_i64_extend16_s | visit_i64_extend32_s
                  | visit_i64_extend_i32_s | visit_i64_extend_i32_u
                  | visit_i64_trunc_f32_s | visit_i64_trunc_f32_u
                  | visit_i64_trunc_f64_s | visit_i64_trunc_f64_u
                  | visit_i64_trunc_sat_f32_s | visit_i64_trunc_sat_f32_u
                  | visit_i64_trunc_sat_f64_s | visit_i64_trunc_sat_f64_u
                  | visit_i64_reinterpret_f64;

        F32.cvtop = visit_f32_demote_f64
                  | visit_f32_convert_i32_s | visit_f32_convert_i32_u
                  | visit_f32_convert_i64_s | visit_f32_convert_i64_u
                  | visit_f32_reinterpret_i32;
        F64.cvtop = visit_f64_promote_f32
                  | visit_f64_convert_i32_s | visit_f64_convert_i32_u
                  | visit_f64_convert_i64_s | visit_f64_convert_i64_u
                  | visit_f64_reinterpret_i64;

        V128.cvtop = visit_i16x8_extend_low_i8x16_s | visit_i16x8_extend_high_i8x16_s
                   | visit_i16x8_extend_low_i8x16_u | visit_i16x8_extend_high_i8x16_u;

        V128.cvtop = visit_i32x4_extend_low_i16x8_s | visit_i32x4_extend_high_i16x8_s
                   | visit_i32x4_extend_low_i16x8_u | visit_i32x4_extend_high_i16x8_u
                   | visit_i32x4_trunc_sat_f32x4_s | visit_i32x4_trunc_sat_f32x4_u
                   | visit_i32x4_trunc_sat_f64x2_s_zero | visit_i32x4_trunc_sat_f64x2_u_zero
                   | visit_i32x4_relaxed_trunc_sat_f32x4_s
                   | visit_i32x4_relaxed_trunc_sat_f32x4_u
                   | visit_i32x4_relaxed_trunc_sat_f64x2_s_zero
                   | visit_i32x4_relaxed_trunc_sat_f64x2_u_zero;

        V128.cvtop = visit_i64x2_extend_low_i32x4_s | visit_i64x2_extend_high_i32x4_s
                   | visit_i64x2_extend_low_i32x4_u | visit_i64x2_extend_high_i32x4_u;

        V128.cvtop = visit_f32x4_demote_f64x2_zero
                   | visit_f32x4_convert_i32x4_s | visit_f32x4_convert_i32x4_u;

        V128.cvtop = visit_f64x2_promote_low_f32x4
                   | visit_f64x2_convert_low_i32x4_s | visit_f64x2_convert_low_i32x4_u;
    }

    instruction_category! {
        I32.load = visit_i32_load
                 | visit_i32_load8_s | visit_i32_load8_u
                 | visit_i32_load16_s | visit_i32_load16_u
                 | visit_i32_atomic_load
                 | visit_i32_atomic_load8_u
                 | visit_i32_atomic_load16_u;

        I64.load = visit_i64_load
                 | visit_i64_load8_s | visit_i64_load8_u
                 | visit_i64_load16_s | visit_i64_load16_u
                 | visit_i64_load32_s | visit_i64_load32_u
                 | visit_i64_atomic_load
                 | visit_i64_atomic_load8_u
                 | visit_i64_atomic_load16_u
                 | visit_i64_atomic_load32_u;

        F32.load = visit_f32_load;

        F64.load = visit_f64_load;

        V128.load = visit_v128_load
                  | visit_v128_load8x8_s | visit_v128_load8x8_u
                  | visit_v128_load16x4_s | visit_v128_load16x4_u
                  | visit_v128_load32x2_s | visit_v128_load32x2_u
                  | visit_v128_load32_zero | visit_v128_load64_zero
                  | visit_v128_load8_splat | visit_v128_load16_splat
                  | visit_v128_load32_splat | visit_v128_load64_splat;
    }

    instruction_category! {
        I32.store = visit_i32_store | visit_i32_store8 | visit_i32_store16
                  | visit_i32_atomic_store | visit_i32_atomic_store8 | visit_i32_atomic_store16;

        I64.store = visit_i64_store | visit_i64_store8 | visit_i64_store16 | visit_i64_store32
                  | visit_i64_atomic_store
                  | visit_i64_atomic_store8 | visit_i64_atomic_store16 | visit_i64_atomic_store32;

        F32.store = visit_f32_store;

        F64.store = visit_f64_store;

        V128.store = visit_v128_store;
    }

    instruction_category! {
        V128.loadlane = visit_v128_load8_lane | visit_v128_load16_lane
                       | visit_v128_load32_lane | visit_v128_load64_lane;
    }

    instruction_category! {
        V128.storelane = visit_v128_store8_lane | visit_v128_store16_lane
                        | visit_v128_store32_lane | visit_v128_store64_lane;
    }

    instruction_category! {
        V128.replacelane = visit_i8x16_replace_lane | visit_i16x8_replace_lane
                         | visit_i32x4_replace_lane | visit_i64x2_replace_lane
                         | visit_f32x4_replace_lane | visit_f64x2_replace_lane;
    }

    instruction_category! {
        V128.extractlane = visit_i8x16_extract_lane_s to I32 | visit_i8x16_extract_lane_u to I32
                         | visit_i16x8_extract_lane_s to I32 | visit_i16x8_extract_lane_u to I32
                         | visit_i32x4_extract_lane to I32
                         | visit_i64x2_extract_lane to I64
                         | visit_f32x4_extract_lane to F32
                         | visit_f64x2_extract_lane to F64;
    }

    instruction_category! {
        V128.vternop = visit_v128_bitselect;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-laneselect
        V128.vternop = visit_i8x16_relaxed_laneselect;
        V128.vternop = visit_i16x8_relaxed_laneselect;
        V128.vternop = visit_i32x4_relaxed_laneselect;
        V128.vternop = visit_i64x2_relaxed_laneselect;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-fused-multiply-add-and-fused-negative-multiply-add
        V128.vternop = visit_f32x4_relaxed_fma | visit_f32x4_relaxed_fnma;
        V128.vternop = visit_f64x2_relaxed_fma | visit_f64x2_relaxed_fnma;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-integer-dot-product
        V128.vternop = visit_i32x4_dot_i8x16_i7x16_add_s;

        // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-bfloat16-dot-product
        V128.vternop = visit_f32x4_relaxed_dot_bf16x8_add_f32x4;
    }

    instruction_category! {
        V128.vrelop = visit_i8x16_eq | visit_i8x16_ne
                    | visit_i8x16_lt_s | visit_i8x16_lt_u | visit_i8x16_gt_s | visit_i8x16_gt_u
                    | visit_i8x16_le_s | visit_i8x16_le_u | visit_i8x16_ge_s | visit_i8x16_ge_u;
        V128.vrelop = visit_i16x8_eq | visit_i16x8_ne
                    | visit_i16x8_lt_s | visit_i16x8_lt_u | visit_i16x8_gt_s | visit_i16x8_gt_u
                    | visit_i16x8_le_s | visit_i16x8_le_u | visit_i16x8_ge_s | visit_i16x8_ge_u;
        V128.vrelop = visit_i32x4_eq | visit_i32x4_ne
                    | visit_i32x4_lt_s | visit_i32x4_lt_u | visit_i32x4_gt_s | visit_i32x4_gt_u
                    | visit_i32x4_le_s | visit_i32x4_le_u | visit_i32x4_ge_s | visit_i32x4_ge_u;
        V128.vrelop = visit_i64x2_eq | visit_i64x2_ne
                    | visit_i64x2_lt_s | visit_i64x2_gt_s | visit_i64x2_le_s | visit_i64x2_ge_s;
        V128.vrelop = visit_f32x4_eq | visit_f32x4_ne
                    | visit_f32x4_lt | visit_f32x4_gt | visit_f32x4_le | visit_f32x4_ge;
        V128.vrelop = visit_f64x2_eq | visit_f64x2_ne
                    | visit_f64x2_lt | visit_f64x2_gt | visit_f64x2_le | visit_f64x2_ge;
    }

    instruction_category! {
        V128.vinarrowop = visit_i8x16_narrow_i16x8_s | visit_i8x16_narrow_i16x8_u;
        V128.vinarrowop = visit_i16x8_narrow_i32x4_s | visit_i16x8_narrow_i32x4_u;
    }

    instruction_category! {
        V128.vbitmask = visit_i8x16_bitmask | visit_i16x8_bitmask
                      | visit_i32x4_bitmask | visit_i64x2_bitmask;
    }

    instruction_category! {
        V128.splat = visit_i8x16_splat | visit_i16x8_splat | visit_i32x4_splat | visit_i64x2_splat
                   | visit_f32x4_splat | visit_f64x2_splat;
    }

    fn visit_i8x16_shuffle(&mut self, _: usize, _: [u8; 16]) -> Self::Output {
        // i8x16.shuffle laneidx^16 : [v128 v128] → [v128]
        self.pop()?;
        Ok(None)
    }

    instruction_category! {
        I32.atomic.rmw = visit_i32_atomic_rmw_add | visit_i32_atomic_rmw_sub
                       | visit_i32_atomic_rmw_and | visit_i32_atomic_rmw_or
                       | visit_i32_atomic_rmw_xor | visit_i32_atomic_rmw_xchg
                       | visit_i32_atomic_rmw8_add_u | visit_i32_atomic_rmw16_add_u
                       | visit_i32_atomic_rmw8_sub_u | visit_i32_atomic_rmw16_sub_u
                       | visit_i32_atomic_rmw8_and_u | visit_i32_atomic_rmw16_and_u
                       | visit_i32_atomic_rmw8_or_u | visit_i32_atomic_rmw16_or_u
                       | visit_i32_atomic_rmw8_xor_u | visit_i32_atomic_rmw16_xor_u
                       | visit_i32_atomic_rmw8_xchg_u | visit_i32_atomic_rmw16_xchg_u;
        I64.atomic.rmw = visit_i64_atomic_rmw_add | visit_i64_atomic_rmw_sub
                       | visit_i64_atomic_rmw_and | visit_i64_atomic_rmw_or
                       | visit_i64_atomic_rmw_xor | visit_i64_atomic_rmw_xchg
                       | visit_i64_atomic_rmw8_add_u | visit_i64_atomic_rmw16_add_u
                       | visit_i64_atomic_rmw32_add_u
                       | visit_i64_atomic_rmw8_sub_u | visit_i64_atomic_rmw16_sub_u
                       | visit_i64_atomic_rmw32_sub_u
                       | visit_i64_atomic_rmw8_and_u | visit_i64_atomic_rmw16_and_u
                       | visit_i64_atomic_rmw32_and_u
                       | visit_i64_atomic_rmw8_or_u | visit_i64_atomic_rmw16_or_u
                       | visit_i64_atomic_rmw32_or_u
                       | visit_i64_atomic_rmw8_xor_u | visit_i64_atomic_rmw16_xor_u
                       | visit_i64_atomic_rmw32_xor_u
                       | visit_i64_atomic_rmw8_xchg_u | visit_i64_atomic_rmw16_xchg_u
                       | visit_i64_atomic_rmw32_xchg_u;
    }

    instruction_category! {
        I32.atomic.cmpxchg = visit_i32_atomic_rmw_cmpxchg | visit_i32_atomic_rmw8_cmpxchg_u
                           | visit_i32_atomic_rmw16_cmpxchg_u;
        I64.atomic.cmpxchg = visit_i64_atomic_rmw_cmpxchg | visit_i64_atomic_rmw8_cmpxchg_u
                           | visit_i64_atomic_rmw16_cmpxchg_u | visit_i64_atomic_rmw32_cmpxchg_u;
    }

    fn visit_memory_atomic_notify(&mut self, _: usize, _: MemArg) -> Self::Output {
        // [i32 i32] -> [i32]
        self.pop()?;
        Ok(None)
    }

    fn visit_memory_atomic_wait32(&mut self, _: usize, _: MemArg) -> Self::Output {
        // [i32 i32 i64] -> [i32]
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_memory_atomic_wait64(&mut self, _: usize, _: MemArg) -> Self::Output {
        // [i32 i64 i64] -> [i32]
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_atomic_fence(&mut self, _: usize) -> Self::Output {
        // https://github.com/WebAssembly/threads/blob/main/proposals/threads/Overview.md#fence-operator
        // [] -> []

        // Function body intentionally left empty
        Ok(None)
    }

    fn visit_local_get(&mut self, _: usize, local_index: u32) -> Self::Output {
        // [] → [t]
        let local_type = self
            .locals
            .get(&local_index)
            .ok_or(Error::LocalIndex(local_index))?;
        self.push(*local_type);
        Ok(None)
    }

    fn visit_local_set(&mut self, _: usize, _: u32) -> Self::Output {
        // [t] → []
        self.pop()?;
        Ok(None)
    }

    fn visit_local_tee(&mut self, _: usize, _: u32) -> Self::Output {
        // [t] → [t]
        Ok(None)
    }

    fn visit_global_get(&mut self, _: usize, global: u32) -> Self::Output {
        // [] → [t]
        let global_usize =
            usize::try_from(global).map_err(|e| Error::GlobalIndexRange(global, e))?;
        let global_ty = self
            .globals
            .get(global_usize)
            .ok_or(Error::GlobalIndex(global))?;
        self.push(*global_ty);
        Ok(None)
    }

    fn visit_global_set(&mut self, _: usize, _: u32) -> Self::Output {
        // [t] → []
        self.pop()?;
        Ok(None)
    }

    fn visit_memory_size(&mut self, _: usize, _: u32, _: u8) -> Self::Output {
        // [] → [i32]
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_memory_grow(&mut self, _: usize, _: u32, _: u8) -> Self::Output {
        // [i32] → [i32]

        // Function body intentionally left empty.
        Ok(None)
    }

    fn visit_memory_fill(&mut self, _: usize, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(None)
    }

    fn visit_memory_init(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(None)
    }

    fn visit_memory_copy(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(None)
    }

    fn visit_data_drop(&mut self, _: usize, _: u32) -> Self::Output {
        // [] → []
        Ok(None)
    }

    fn visit_table_get(&mut self, _: usize, table: u32) -> Self::Output {
        // [i32] → [t]
        let table_usize = usize::try_from(table).map_err(|e| Error::TableIndexRange(table, e))?;
        let table_ty = *self
            .tables
            .get(table_usize)
            .ok_or(Error::TableIndex(table))?;
        self.pop()?;
        self.push(table_ty);
        Ok(None)
    }

    fn visit_table_set(&mut self, _: usize, _: u32) -> Self::Output {
        // [i32 t] → []
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_table_size(&mut self, _: usize, _: u32) -> Self::Output {
        // [] → [i32]
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_table_grow(&mut self, _: usize, _: u32) -> Self::Output {
        // [t i32] → [i32]
        self.pop_many(2)?;
        self.push(ValType::I32);
        Ok(None)
    }

    fn visit_table_fill(&mut self, _: usize, _: u32) -> Self::Output {
        // [i32 t i32] → []
        self.pop_many(3)?;
        Ok(None)
    }

    fn visit_table_copy(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(None)
    }

    fn visit_table_init(&mut self, _: usize, _: u32, _: u32) -> Self::Output {
        // [i32 i32 i32] → []
        self.pop_many(3)?;
        Ok(None)
    }

    fn visit_elem_drop(&mut self, _: usize, _: u32) -> Self::Output {
        // [] → []
        Ok(None)
    }

    fn visit_select(&mut self, _: usize) -> Self::Output {
        // [t t i32] -> [t]
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_typed_select(&mut self, _: usize, t: ValType) -> Self::Output {
        // [t t i32] -> [t]
        self.pop_many(2)?;
        Ok(None)
    }

    fn visit_drop(&mut self, _: usize) -> Self::Output {
        // [t] → []
        self.pop()?;
        Ok(None)
    }

    fn visit_nop(&mut self, _: usize) -> Self::Output {
        // [] → []
        Ok(None)
    }

    fn visit_call(&mut self, _: usize, function_index: u32) -> Self::Output {
        self.visit_function_call(self.function_type_index(function_index)?)
    }

    fn visit_call_indirect(&mut self, _: usize, type_index: u32, _: u32, _: u8) -> Self::Output {
        self.visit_function_call(type_index)
    }

    fn visit_return_call(&mut self, offset: usize, _: u32) -> Self::Output {
        // `return_call` behaves as-if a regular `return` followed by the `call`. For the purposes
        // of modelling the frame size of the _current_ function, only the `return` portion of this
        // computation is relevant (as it makes the stack polymorphic)
        self.visit_return(offset)
    }

    fn visit_return_call_indirect(&mut self, offset: usize, _: u32, _: u32) -> Self::Output {
        self.visit_return(offset)
    }

    fn visit_unreachable(&mut self, _: usize) -> Self::Output {
        // [*] → [*]  (stack-polymorphic)
        self.stack_polymorphic();
        Ok(None)
    }

    fn visit_block(&mut self, _: usize, blockty: BlockType) -> Self::Output {
        // block blocktype instr* end : [t1*] → [t2*]
        self.with_block_types(blockty, |this, params, _| {
            this.new_frame(blockty, params.len())
        })?;
        Ok(None)
    }

    fn visit_loop(&mut self, _: usize, blockty: BlockType) -> Self::Output {
        // loop blocktype instr* end : [t1*] → [t2*]
        self.with_block_types(blockty, |this, params, _| {
            this.new_frame(blockty, params.len())
        })?;
        Ok(None)
    }

    fn visit_if(&mut self, _: usize, blockty: BlockType) -> Self::Output {
        // if blocktype instr* else instr* end : [t1* i32] → [t2*]
        self.pop()?;
        self.with_block_types(blockty, |this, params, _| {
            this.new_frame(blockty, params.len())
        })?;
        Ok(None)
    }

    fn visit_else(&mut self, _: usize) -> Self::Output {
        if let Some(frame) = self.end_frame()? {
            self.with_block_types(frame.block_type, |this, params, _| {
                this.new_frame(frame.block_type, 0)?;
                for param in params {
                    this.push(*param);
                }
                Ok(())
            })?;
            Ok(None)
        } else {
            return Err(Error::TruncatedFrameStack(self.offset));
        }
    }

    fn visit_end(&mut self, _: usize) -> Self::Output {
        if let Some(frame) = self.end_frame()? {
            self.with_block_types(frame.block_type, |this, _, results| {
                Ok(for result in results {
                    this.push(*result);
                })
            })?;
            Ok(None)
        } else {
            // Returning from the function. Malformed WASM modules may have trailing instructions,
            // but we do ignore processing them in the operand feed loop. For that reason,
            // replacing `top_stack` with some sentinel value would work okay.
            self.current_frame = Frame {
                height: !0,
                block_type: BlockType::Empty,
                stack_polymorphic: true,
            };
            self.operands.clear();
            self.size = 0;
            Ok(Some(self.max_size))
        }
    }

    fn visit_br(&mut self, _: usize, _: u32) -> Self::Output {
        // [t1* t*] → [t2*]  (stack-polymorphic)
        self.stack_polymorphic();
        Ok(None)
    }

    fn visit_br_if(&mut self, _: usize, _: u32) -> Self::Output {
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
        Ok(None)
    }

    fn visit_br_table(&mut self, _: usize, _: BrTable) -> Self::Output {
        // [t1* t* i32] → [t2*]  (stack-polymorphic)
        self.stack_polymorphic();
        Ok(None)
    }

    fn visit_return(&mut self, offset: usize) -> Self::Output {
        // This behaves as-if a `br` to the outer-most block.

        // NB: self.frames.len() is actually 1 less than a number of frames, due to our maintaining
        // a `self.current_frame`.
        let branch_depth = u32::try_from(self.frames.len()).map_err(|_| Error::TooManyFrames)?;
        self.visit_br(offset, branch_depth)
    }

    fn visit_try(&mut self, _: usize, _: BlockType) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_rethrow(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_throw(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_delegate(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_catch(&mut self, _: usize, _: u32) -> Self::Output {
        todo!("exception handling has not been implemented");
    }

    fn visit_catch_all(&mut self, _: usize) -> Self::Output {
        todo!("exception handling has not been implemented");
    }
}
