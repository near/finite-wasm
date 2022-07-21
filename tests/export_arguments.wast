(;
For exports we must introduce a trampoline function that would charge the necessary costs before
the user-controlled function is invoked
;)

(module
  (func (export "main") (param i32 i64) (local i32 i64 i32 i64 i32))
)

(skip "not implemented yet")

(assert_instrumented_gas)

(assert_instrumented_stack)

(assert_instrumented)
