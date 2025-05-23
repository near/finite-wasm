open Types
open Values

type 'inst t = 'inst func
and 'inst func =
  | AstFunc of func_type * 'inst * Ast.func
  | HostFunc of func_type * (value list -> value list)
  | GasIntrinsic
  | CopyIntrinsic
  | FillIntrinsic
  | InitIntrinsic
  | StackIntrinsic
  | UnstackIntrinsic

val alloc : func_type -> 'inst -> Ast.func -> 'inst func
val alloc_host : func_type -> (value list -> value list) -> 'inst func
val type_of : 'inst func -> func_type
