(; Meta test demonstrating the assertions made available by the test runner ;)

(; Define the input module. ;)
(module (;0;))

(; define a named module ;)
(module $named_module (data "I'm a named module"))

(; Define the input module. ;)
(module (;2;) (data "I'm a module #2"))

(; Assert a module after gas and stack instrumentation. Defaults to 0th module. ;)
(assert_instrumented (module))
(; Assert a module after gas instrumentation ;)
(assert_instrumented_gas (module))
(; Assert a module after stack instrumentation ;)
(assert_instrumented_stack (module))

(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented $named_module (module (data "I'm a named module")))
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_gas $named_module (module (data "I'm a named module")))
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_stack $named_module (module (data "I'm a named module")))

(; Can also refer to modules by index ;)
(assert_instrumented 2 (module (data "I'm a module #2")))
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_gas 2 (module (data "I'm a module #2")))
(; Assert shape of a module after the named module was instrumented… ;)
(assert_instrumented_stack 2 (module (data "I'm a module #2")))
