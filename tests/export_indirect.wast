(module
  (import "env" "funcref" (global funcref))

  (func $f1 (param $p i64)
    (call $f2 (i32.wrap_i64 (local.get $p)))
    (return)
  )
  (func $f2 (param $p i32)
    (call $f3 (f32.convert_i32_s (local.get $p)))
    (return)
  )

  (func $f3 (param $p f32)
    (call $f4 (f64.promote_f32 (local.get $p)))
    (return)
  )

  (; not used in start/export/table so shouldn’t get indirected ;)
  (func $f4 (param $p f64)
    (call $f1 (i64.trunc_f64_s (local.get $p)))
    (return)
  )

  (; indirects start ;)
  (func $start (call $f1 (i64.const 42)))
  (start $start)

  (; indirects f2 ;)
  (export "f2" (func $f2))


  (; indirects f3 ;)
  (table 1 funcref)
  (elem (i32.const 0) $f3)
  (elem (i32.const 0) $f3)
  (elem (i32.const 0) funcref (item (ref.func $f3)))
  (; (elem (i32.const 0) funcref (item (global.get 0))) ;)
)

(assert_indirected (module

))
