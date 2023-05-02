(module
  (func (export "regression45") (param $count i32)
    loop $l1
      loop $l2
        loop $l3
          (i32.add (get_local $count) (i32.const 1))
          (tee_local $count)
          (br_table $l2 $l2 $l2 $l2 $l2 $l2 $l2 3)
        end
      end
    end
  )
)

(invoke "regression45" (i32.const 0))
