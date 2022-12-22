(module
  (func $param_is_local (param $p1 i32) (param $p2 i64))

  (func $local_is_local (param $p1 i32) (param $p2 i64) (local i32 i64))

  (func $return_is_operand (param $p1 i32) (param $p2 i32) (result i32)
    local.get $p1
  )
)
