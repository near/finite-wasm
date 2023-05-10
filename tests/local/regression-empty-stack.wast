(module
  (func
    unreachable
    f64.const 0.0
    loop (param f64)
      drop
    end)
)

(module
  (func
    unreachable
    loop (param f64)
      drop
    end)
)
