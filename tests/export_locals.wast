(;
For exports we must introduce a trampoline function that would charge the necessary costs before
the user-controlled function is invoked
;)

(module
  (func (export "main") (local i32 i32 i32 i32 i32))
)

(assert_instrumented_gas (module
  (import "" "gas" (func $gas (param i64)))

  (func $original_main (local i32 i32 i32 i32 i32))
  (func (export "main")
    (; calling `$original_main` will initialize 5 locals to 0, +1 for the call itself ;)
    (call $gas (i64.const 6))
    (call $original_main)
  )
))

(assert_instrumented_stack (module
  (import "" "stack" (func $stack (param i32)))

  (func $original_main (local i32 i32 i32 i32 i32)
    (call $stack (i32.const -6))
  )
  (func (export "main")
    (; function activation frame and 5 locals ;)
    (call $stack (i32.const 6))
    (call $original_main)
  )
))

(assert_instrumented (module
  (import "" "gas" (func $gas (param i64)))
  (import "" "stack" (func $stack (param i32)))

  (func $original_main (local i32 i32 i32 i32 i32)
    (call $stack (i32.const -6))
  )
  (func (export "main")
    (call $stack (i32.const 6))
    (call $gas (i64.const 6))
    (call $original_main)
  )
))
