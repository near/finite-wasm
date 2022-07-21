(; Meta test demonstrating the assertions made available by the test runner ;)

(; Define the input module. ;)
(module (;0;))

(; define a named module ;)
(module $named_module (data "I'm a named module"))

(; Define the input module. ;)
(module (;2;) (data "I'm a module #2"))

(skip "not implemented yet")

(; Assert a module after gas and stack instrumentation. Defaults to 0th module. ;)
(assert_instrumented)
(; Assert a module after gas instrumentation ;)
(assert_instrumented_gas)
(; Assert a module after stack instrumentation ;)
(assert_instrumented_stack)

(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented $named_module)
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_gas $named_module)
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_stack $named_module)

(; Can also refer to modules by index ;)
(assert_instrumented 2)
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_gas 2)
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_stack 2)
