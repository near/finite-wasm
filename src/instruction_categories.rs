macro_rules! r#const {
    ($generator:ident) => {
        $generator! {
            I32.const = visit_i32_const, i32;
            I64.const = visit_i64_const, i64;
            F32.const = visit_f32_const, wasmparser::Ieee32;
            F64.const = visit_f64_const, wasmparser::Ieee64;
            V128.const = visit_v128_const, wasmparser::V128;
        }
    };
}

macro_rules! unop {
    ($generator:ident) => {
        $generator! {
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
    };
}

macro_rules! binop_partial {
    ($generator:ident) => {
        $generator! {
            I32.binop = visit_i32_div_s | visit_i32_div_u
                      | visit_i32_rem_s | visit_i32_rem_u;
            I64.binop = visit_i64_div_s | visit_i64_div_u
                      | visit_i64_rem_s | visit_i64_rem_u;
        }
    };
}

macro_rules! binop_complete {
    ($generator:ident) => {
        $generator! {
            I32.binop = visit_i32_add | visit_i32_sub | visit_i32_mul
                      | visit_i32_and | visit_i32_or | visit_i32_xor | visit_i32_shl
                      | visit_i32_shr_s | visit_i32_shr_u
                      | visit_i32_rotl | visit_i32_rotr;
            I64.binop = visit_i64_add | visit_i64_sub | visit_i64_mul
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
            V128.binop = visit_i16x8_relaxed_dot_i8x16_i7x16_s;


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
    }
}

macro_rules! binop {
    ($generator:ident) => {
        $crate::instruction_categories::binop_partial!($generator);
        $crate::instruction_categories::binop_complete!($generator);
    };
}

macro_rules! vishiftop {
    ($generator:ident) => {
        $generator! {
            V128.vishiftop = visit_i8x16_shl | visit_i8x16_shr_s | visit_i8x16_shr_u;
            V128.vishiftop = visit_i16x8_shl | visit_i16x8_shr_s | visit_i16x8_shr_u;
            V128.vishiftop = visit_i32x4_shl | visit_i32x4_shr_s | visit_i32x4_shr_u;
            V128.vishiftop = visit_i64x2_shl | visit_i64x2_shr_s | visit_i64x2_shr_u;
        }
    };
}

macro_rules! testop {
    ($generator:ident) => {
        $generator! {
            I32.testop = visit_i32_eqz;
            I64.testop = visit_i64_eqz;
            V128.testop = visit_v128_any_true
                        | visit_i8x16_all_true | visit_i16x8_all_true
                        | visit_i32x4_all_true | visit_i64x2_all_true;
            FuncRef.testop = visit_ref_is_null;
        }
    };
}

macro_rules! relop {
    ($generator:ident) => {
        $generator! {
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
    };
}

macro_rules! cvtop_partial {
    ($generator:ident) => {
        $generator! {
            I32.cvtop = visit_i32_trunc_f32_s | visit_i32_trunc_f32_u
                      | visit_i32_trunc_f64_s | visit_i32_trunc_f64_u;
            I64.cvtop = visit_i64_trunc_f32_s | visit_i64_trunc_f32_u
                      | visit_i64_trunc_f64_s | visit_i64_trunc_f64_u;
        }
    };
}

macro_rules! cvtop_complete {
    ($generator:ident) => {
        $generator! {
            I32.cvtop = visit_i32_wrap_i64
                      | visit_i32_extend8_s | visit_i32_extend16_s
                      | visit_i32_trunc_sat_f32_s | visit_i32_trunc_sat_f32_u
                      | visit_i32_trunc_sat_f64_s | visit_i32_trunc_sat_f64_u
                      | visit_i32_reinterpret_f32;

            I64.cvtop = visit_i64_extend8_s | visit_i64_extend16_s | visit_i64_extend32_s
                      | visit_i64_extend_i32_s | visit_i64_extend_i32_u
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
                       | visit_i32x4_relaxed_trunc_f32x4_s
                       | visit_i32x4_relaxed_trunc_f32x4_u
                       | visit_i32x4_relaxed_trunc_f64x2_s_zero
                       | visit_i32x4_relaxed_trunc_f64x2_u_zero;

            V128.cvtop = visit_i64x2_extend_low_i32x4_s | visit_i64x2_extend_high_i32x4_s
                       | visit_i64x2_extend_low_i32x4_u | visit_i64x2_extend_high_i32x4_u;

            V128.cvtop = visit_f32x4_demote_f64x2_zero
                       | visit_f32x4_convert_i32x4_s | visit_f32x4_convert_i32x4_u;

            V128.cvtop = visit_f64x2_promote_low_f32x4
                       | visit_f64x2_convert_low_i32x4_s | visit_f64x2_convert_low_i32x4_u;
        }
    };
}

macro_rules! cvtop {
    ($generator:ident) => {
        $crate::instruction_categories::cvtop_partial!($generator);
        $crate::instruction_categories::cvtop_complete!($generator);
    };
}

macro_rules! loadlane {
    ($generator:ident) => {
        $generator! {
            V128.loadlane = visit_v128_load8_lane | visit_v128_load16_lane
                           | visit_v128_load32_lane | visit_v128_load64_lane;
        }
    };
}

macro_rules! load {
    ($generator:ident) => {
        $generator! {
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
    };
}

macro_rules! storelane {
    ($generator:ident) => {
        $generator! {
            V128.storelane = visit_v128_store8_lane | visit_v128_store16_lane
                            | visit_v128_store32_lane | visit_v128_store64_lane;
        }
    };
}

macro_rules! store {
    ($generator:ident) => {
        $generator! {
            I32.store = visit_i32_store | visit_i32_store8 | visit_i32_store16
                      | visit_i32_atomic_store
                      | visit_i32_atomic_store8 | visit_i32_atomic_store16;

            I64.store = visit_i64_store | visit_i64_store8 | visit_i64_store16 | visit_i64_store32
                      | visit_i64_atomic_store
                      | visit_i64_atomic_store8
                      | visit_i64_atomic_store16
                      | visit_i64_atomic_store32;

            F32.store = visit_f32_store;

            F64.store = visit_f64_store;

            V128.store = visit_v128_store;
        }
    };
}

macro_rules! replacelane {
    ($generator:ident) => {
        $generator! {
            V128.replacelane = visit_i8x16_replace_lane | visit_i16x8_replace_lane
                             | visit_i32x4_replace_lane | visit_i64x2_replace_lane
                             | visit_f32x4_replace_lane | visit_f64x2_replace_lane;
        }
    };
}

macro_rules! extractlane {
    ($generator:ident) => {
        $generator! {
            I32.extractlane = visit_i8x16_extract_lane_s
                            | visit_i8x16_extract_lane_u
                            | visit_i16x8_extract_lane_s
                            | visit_i16x8_extract_lane_u
                            | visit_i32x4_extract_lane;
            I64.extractlane = visit_i64x2_extract_lane;
            F32.extractlane = visit_f32x4_extract_lane;
            F64.extractlane = visit_f64x2_extract_lane;
        }
    };
}

macro_rules! vternop {
    ($generator:ident) => {
        $generator! {
            V128.vternop = visit_v128_bitselect;

            // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-laneselect
            V128.vternop = visit_i8x16_relaxed_laneselect;
            V128.vternop = visit_i16x8_relaxed_laneselect;
            V128.vternop = visit_i32x4_relaxed_laneselect;
            V128.vternop = visit_i64x2_relaxed_laneselect;

            // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-fused-multiply-add-and-fused-negative-multiply-add
            V128.vternop = visit_f32x4_relaxed_madd | visit_f32x4_relaxed_nmadd;
            V128.vternop = visit_f64x2_relaxed_madd | visit_f64x2_relaxed_nmadd;

            // https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-integer-dot-product
            V128.vternop = visit_i32x4_relaxed_dot_i8x16_i7x16_add_s;
        }
    };
}

macro_rules! vrelop {
    ($generator:ident) => {
        $generator! {
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
    };
}

macro_rules! vinarrowop {
    ($generator:ident) => {
        $generator! {
            V128.vinarrowop = visit_i8x16_narrow_i16x8_s | visit_i8x16_narrow_i16x8_u;
            V128.vinarrowop = visit_i16x8_narrow_i32x4_s | visit_i16x8_narrow_i32x4_u;
        }
    };
}

macro_rules! vbitmask {
    ($generator:ident) => {
        $generator! {
            V128.vbitmask = visit_i8x16_bitmask | visit_i16x8_bitmask
                          | visit_i32x4_bitmask | visit_i64x2_bitmask;
        }
    };
}

macro_rules! splat {
    ($generator:ident) => {
        $generator! {
            V128.splat = visit_i8x16_splat | visit_i16x8_splat | visit_i32x4_splat
                       | visit_i64x2_splat | visit_f32x4_splat | visit_f64x2_splat;
        }
    };
}

macro_rules! atomic_rmw {
    ($generator:ident) => {
        $generator! {
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
    };
}

macro_rules! atomic_cmpxchg {
    ($generator:ident) => {
        $generator! {
            I32.atomic.cmpxchg = visit_i32_atomic_rmw_cmpxchg
                               | visit_i32_atomic_rmw8_cmpxchg_u
                               | visit_i32_atomic_rmw16_cmpxchg_u;
            I64.atomic.cmpxchg = visit_i64_atomic_rmw_cmpxchg
                               | visit_i64_atomic_rmw8_cmpxchg_u
                               | visit_i64_atomic_rmw16_cmpxchg_u
                               | visit_i64_atomic_rmw32_cmpxchg_u;
        }
    };
}

pub(crate) use {
    atomic_cmpxchg, atomic_rmw, binop, binop_complete, binop_partial, cvtop, cvtop_complete,
    cvtop_partial, extractlane, load, loadlane, r#const, relop, replacelane, splat, store,
    storelane, testop, unop, vbitmask, vinarrowop, vishiftop, vrelop, vternop,
};
