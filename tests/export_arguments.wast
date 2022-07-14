(;
For exports we must introduce a trampoline function that would charge the necessary costs before
the user-controlled function is invoked
;)

(module
  (func (export "main") (param i32 i64) (local i32 i64 i32 i64 i32))
)

(assert_instrumented_gas (module
  (import "" "gas" (func $gas (param i64)))

  (func $original_main (param i32 i64) (local i32 i64 i32 i64 i32))
  (func (export "main") (param $p1 i32) (param $p2 i64)
    (; calling `$original_main` will initialize 5 locals to 0, plus the call itself ;)
    (call $gas (i64.const 6))
    (; FIXME: #1 ;)
    (call $original_main (local.get $p1) (local.get $p2))
  )
))

(assert_instrumented_stack (module
  (import "" "stack" (func $stack (param i32)))

  (func $original_main (param i32 i64) (local i32 i64 i32 i64 i32)
    (; Function execution is complete, reduce the stack height before returning ;)
    (call $stack (i32.const -8))
  )
  (func (export "main") (param $p1 i32) (param $p2 i64)
    (; function activation frame, 2 arguments and 5 locals ;)
    (call $stack (i32.const 8))
    (call $original_main (local.get $p1) (local.get $p2))
  )
))

(assert_instrumented (module
  (import "" "gas" (func $gas (param i64)))
  (import "" "stack" (func $stack (param i32)))

  (func $original_main (param i32 i64) (local i32 i64 i32 i64 i32)
    (call $stack (i32.const -8))
  )
  (func (export "main") (param $p1 i32) (param $p2 i64)
    (call $stack (i32.const 8))
    (call $gas (i64.const 6))
    (call $original_main)
  )
))
