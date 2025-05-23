open Values
open Types
open Instance
open Ast
open Source


(* Errors *)

module Link = Error.Make ()
module Trap = Error.Make ()
module Crash = Error.Make ()
module Exhaustion = Error.Make ()

exception Link = Link.Error
exception Trap = Trap.Error
exception Crash = Crash.Error (* failure that cannot happen in valid code *)
exception Exhaustion = Exhaustion.Error

let table_error at = function
  | Table.Bounds -> "out of bounds table access"
  | Table.SizeOverflow -> "table size overflow"
  | Table.SizeLimit -> "table size limit reached"
  | Table.Type -> Crash.error at "type mismatch at table access"
  | exn -> raise exn

let memory_error at = function
  | Memory.Bounds -> "out of bounds memory access"
  | Memory.SizeOverflow -> "memory size overflow"
  | Memory.SizeLimit -> "memory size limit reached"
  | Memory.Type -> Crash.error at "type mismatch at memory access"
  | exn -> raise exn

let numeric_error at = function
  | Ixx.Overflow -> "integer overflow"
  | Ixx.DivideByZero -> "integer divide by zero"
  | Ixx.InvalidConversion -> "invalid conversion to integer"
  | Values.TypeError (i, v, t) ->
    Crash.error at
      ("type error, expected " ^ Types.string_of_num_type t ^ " as operand " ^
       string_of_int i ^ ", got " ^ Types.string_of_num_type (type_of_num v))
  | exn -> raise exn


(* Administrative Expressions & Configurations *)

type 'a stack = 'a list

type frame =
{
  inst : module_inst;
  locals : value ref list;
}

type code = value stack * admin_instr list

and admin_instr = admin_instr' phrase
and admin_instr' =
  | Plain of instr'
  (* A plain instruction that has been produced as a result of a reduction rule *)
  | ReducedPlain of instr'
  | Refer of ref_
  | Invoke of func_inst
  | Trapping of string
  | Returning of value stack
  | Breaking of int32 * value stack
  | Label of int32 * instr list * code
  | Frame of int32 * frame * code

type config =
{
  frame : frame;
  code : code;
  budget : int;  (* to model stack overflow *)
  trace_gas : bool;
}

let frame inst locals = {inst; locals}
let config inst vs es =
  {frame = frame inst []; code = vs, es; budget = !Flags.budget; trace_gas = !Flags.trace_gas}
let config_for_init inst vs es =
  { frame = frame inst []; code = vs, es; budget = Int.max_int; trace_gas = false; }

let plain e = Plain e.it @@ e.at
let reduced_plain e = ReducedPlain e.it @@ e.at

let lookup category list x =
  try Lib.List32.nth list x.it with Failure _ ->
    Crash.error x.at ("undefined " ^ category ^ " " ^ Int32.to_string x.it)

let type_ (inst : module_inst) x = lookup "type" inst.types x
let func (inst : module_inst) x = lookup "function" inst.funcs x
let table (inst : module_inst) x = lookup "table" inst.tables x
let memory (inst : module_inst) x = lookup "memory" inst.memories x
let global (inst : module_inst) x = lookup "global" inst.globals x
let elem (inst : module_inst) x = lookup "element segment" inst.elems x
let data (inst : module_inst) x = lookup "data segment" inst.datas x
let local (frame : frame) x = lookup "local" frame.locals x

let any_ref inst x i at =
  try Table.load (table inst x) i with Table.Bounds ->
    Trap.error at ("undefined element " ^ Int32.to_string i)

let func_ref inst x i at =
  match any_ref inst x i at with
  | FuncRef f -> f
  | NullRef _ -> Trap.error at ("uninitialized element " ^ Int32.to_string i)
  | _ -> Crash.error at ("type mismatch for element " ^ Int32.to_string i)

let block_type inst bt =
  match bt with
  | VarBlockType x -> type_ inst x
  | ValBlockType None -> FuncType ([], [])
  | ValBlockType (Some t) -> FuncType ([], [t])

let take n (vs : 'a stack) at =
  try Lib.List32.take n vs with Failure _ -> Crash.error at "stack underflow"

let drop n (vs : 'a stack) at =
  try Lib.List32.drop n vs with Failure _ -> Crash.error at "stack underflow"


(* Evaluation *)

(*
 * Conventions:
 *   e  : instr
 *   v  : value
 *   es : instr list
 *   vs : value stack
 *   c : config
 *)

let mem_oob frame x i n =
  I64.gt_u (I64.add (I64_convert.extend_i32_u i) (I64_convert.extend_i32_u n))
    (Memory.bound (memory frame.inst x))

let data_oob frame x i n =
  I64.gt_u (I64.add (I64_convert.extend_i32_u i) (I64_convert.extend_i32_u n))
    (Data.size (data frame.inst x))

let table_oob frame x i n =
  I64.gt_u (I64.add (I64_convert.extend_i32_u i) (I64_convert.extend_i32_u n))
    (I64_convert.extend_i32_u (Table.size (table frame.inst x)))

let elem_oob frame x i n =
  I64.gt_u (I64.add (I64_convert.extend_i32_u i) (I64_convert.extend_i32_u n))
    (I64_convert.extend_i32_u (Elem.size (elem frame.inst x)))

let gas_fee (i: admin_instr) (vals: value list) : int64 =
  match i.it with
    | Plain _ -> 1L
    | Invoke func -> (match func with
      | Func.AstFunc (FuncType (args, _), _, fn) ->
        let arg_count = Int64.of_int (List.length args) in
        let local_count = Int64.of_int (List.length fn.it.locals) in
        Int64.add arg_count local_count
      | Func.HostFunc (_, _) -> 0L
      (* Compensate for setup and invocation of these intrinsics, as they're free *)
      (* This is validated with the `!internal-self-test-interpreter` test *)
      | Func.GasIntrinsic -> -2L
      | Func.CopyIntrinsic -> -3L
      | Func.FillIntrinsic -> -3L
      | Func.InitIntrinsic -> -3L
      | Func.StackIntrinsic -> -4L
      | Func.UnstackIntrinsic -> -3L
    )
    (* The end instruction, pretty much *)
    | Label (_, _, (_, [])) -> 0L
    (* Administrative or derived and already accounted for *)
    | ReducedPlain _ -> 0L
    | Refer _ -> 0L
    | Trapping _ -> 0L
    | Returning _ -> 0L
    | Breaking _ -> 0L
    | Label _ -> 0L
    | Frame _ -> 0L

let string_of_admin_instr (e: admin_instr) : string = Sexpr.to_string 120 (match e.it with
  | Plain i -> Arrange.instr (i @@ e.at)
  | ReducedPlain i -> let iex = Arrange.instr (i @@ e.at) in Sexpr.Node ("admin.reducedplain", [iex])
  | Refer _ -> Sexpr.Atom "admin.ref"
  | Invoke Func.GasIntrinsic -> Sexpr.Atom "admin.gasintrinsic"
  | Invoke Func.CopyIntrinsic -> Sexpr.Atom "admin.copyintrinsic"
  | Invoke Func.FillIntrinsic -> Sexpr.Atom "admin.fillintrinsic"
  | Invoke Func.InitIntrinsic -> Sexpr.Atom "admin.initintrinsic"
  | Invoke Func.StackIntrinsic -> Sexpr.Atom "admin.stackintrinsic"
  | Invoke Func.UnstackIntrinsic -> Sexpr.Atom "admin.unstackintrinsic"
  | Invoke _ -> Sexpr.Atom "admin.invoke"
  | Trapping _ -> Sexpr.Atom "admin.trapping"
  | Returning _ -> Sexpr.Atom "admin.returning"
  | Breaking _ -> Sexpr.Atom "admin.breaking"
  | Label (_, _, (_, [])) -> Sexpr.Atom "end"
  | Label _ -> Sexpr.Atom "admin.label"
  | Frame _ -> Sexpr.Atom "admin.frame"
  )

let apply_fees (c: config) : config =
  let {frame; code = vs, es; _} = c in
  let { gas; _ } = frame.inst in
  let e = List.hd es in
  let e_cost = gas_fee e vs in
  if (Int64.compare e_cost !gas) > 0 then
    Exhaustion.error e.at "gas pool is empty"
  else if (Int64.compare e_cost 0L) != 0 then
    if c.trace_gas then Printf.printf "gas: %Ld %s" e_cost (if !Flags.trace_gas_ops then string_of_admin_instr e else "\n");
    gas := Int64.sub !gas e_cost;
    c

let rec step (c : config) : config =
  let {frame; code = vs, es; _} = apply_fees c in
  let e = List.hd es in
  let vs', es' =
    match e.it, vs with
    | Plain e', vs | ReducedPlain e', vs ->
      (match e', vs with
      | Unreachable, vs ->
        vs, [Trapping "unreachable executed" @@ e.at]

      | Nop, vs ->
        vs, []

      | Block (bt, es'), vs ->
        let FuncType (ts1, ts2) = block_type frame.inst bt in
        let n1 = Lib.List32.length ts1 in
        let n2 = Lib.List32.length ts2 in
        let args, vs' = take n1 vs e.at, drop n1 vs e.at in
        vs', [Label (n2, [], (args, List.map plain es')) @@ e.at]

      | Loop (bt, es'), vs ->
        let FuncType (ts1, ts2) = block_type frame.inst bt in
        let n1 = Lib.List32.length ts1 in
        let args, vs' = take n1 vs e.at, drop n1 vs e.at in
        vs', [Label (n1, [e' @@ e.at], (args, List.map plain es')) @@ e.at]

      | If (bt, es1, es2), Num (I32 i) :: vs' ->
        if i = 0l then
          vs', [ReducedPlain (Block (bt, es2)) @@ e.at]
        else
          vs', [ReducedPlain (Block (bt, es1)) @@ e.at]

      | Br x, vs ->
        [], [Breaking (x.it, vs) @@ e.at]

      | BrIf x, Num (I32 i) :: vs' ->
        if i = 0l then
          vs', []
        else
          vs', [ReducedPlain (Br x) @@ e.at]

      | BrTable (xs, x), Num (I32 i) :: vs' ->
        if I32.ge_u i (Lib.List32.length xs) then
          vs', [ReducedPlain (Br x) @@ e.at]
        else
          vs', [ReducedPlain (Br (Lib.List32.nth xs i)) @@ e.at]

      | Return, vs ->
        [], [Returning vs @@ e.at]

      | Call x, vs ->
        vs, [Invoke (func frame.inst x) @@ e.at]

      | CallIndirect (x, y), Num (I32 i) :: vs ->
        let func = func_ref frame.inst x i e.at in
        if type_ frame.inst y <> Func.type_of func then
          vs, [Trapping "indirect call type mismatch" @@ e.at]
        else
          vs, [Invoke func @@ e.at]

      | Drop, v :: vs' ->
        vs', []

      | Select _, Num (I32 i) :: v2 :: v1 :: vs' ->
        if i = 0l then
          v2 :: vs', []
        else
          v1 :: vs', []

      | LocalGet x, vs ->
        !(local frame x) :: vs, []

      | LocalSet x, v :: vs' ->
        local frame x := v;
        vs', []

      | LocalTee x, v :: vs' ->
        local frame x := v;
        v :: vs', []

      | GlobalGet x, vs ->
        Global.load (global frame.inst x) :: vs, []

      | GlobalSet x, v :: vs' ->
        (try Global.store (global frame.inst x) v; vs', []
        with Global.NotMutable -> Crash.error e.at "write to immutable global"
           | Global.Type -> Crash.error e.at "type mismatch at global write")

      | TableGet x, Num (I32 i) :: vs' ->
        (try Ref (Table.load (table frame.inst x) i) :: vs', []
        with exn -> vs', [Trapping (table_error e.at exn) @@ e.at])

      | TableSet x, Ref r :: Num (I32 i) :: vs' ->
        (try Table.store (table frame.inst x) i r; vs', []
        with exn -> vs', [Trapping (table_error e.at exn) @@ e.at])

      | TableSize x, vs ->
        Num (I32 (Table.size (table frame.inst x))) :: vs, []

      | TableGrow x, Num (I32 delta) :: Ref r :: vs' ->
        let tab = table frame.inst x in
        let old_size = Table.size tab in
        let result =
          try Table.grow tab delta r; old_size
          with Table.SizeOverflow | Table.SizeLimit | Table.OutOfMemory -> -1l
        in Num (I32 result) :: vs', []

      | TableFill x, Num (I32 n) :: Ref r :: Num (I32 i) :: vs' ->
        if table_oob frame x i n then
          let msg = table_error e.at Table.Bounds in
          if c.trace_gas then Printf.printf "aggregate_trap: %s\n" msg;
          vs', [Trapping msg @@ e.at]
        else if n = 0l then
          vs', []
        else
          let _ = assert (I32.lt_u i 0xffff_ffffl) in
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 i @@ e.at));
            Refer r;
            ReducedPlain (TableSet x);
            ReducedPlain (Const (I32 (I32.add i 1l) @@ e.at));
            Refer r;
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (TableFill x);
          ]

      | TableCopy (x, y), Num (I32 n) :: Num (I32 s) :: Num (I32 d) :: vs' ->
        if table_oob frame x d n || table_oob frame y s n then
          let msg = table_error e.at Table.Bounds in
          if c.trace_gas then Printf.printf "aggregate_trap: %s\n" msg;
          vs', [Trapping msg @@ e.at]
        else if n = 0l then
          vs', []
        else if I32.le_u d s then
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 d @@ e.at));
            ReducedPlain (Const (I32 s @@ e.at));
            ReducedPlain (TableGet y);
            ReducedPlain (TableSet x);
            ReducedPlain (Const (I32 (I32.add d 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.add s 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (TableCopy (x, y));
          ]
        else (* d > s *)
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 (I32.add d 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.add s 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (TableCopy (x, y));
            ReducedPlain (Const (I32 d @@ e.at));
            ReducedPlain (Const (I32 s @@ e.at));
            ReducedPlain (TableGet y);
            ReducedPlain (TableSet x);
          ]

      | TableInit (x, y), Num (I32 n) :: Num (I32 s) :: Num (I32 d) :: vs' ->
        if table_oob frame x d n || elem_oob frame y s n then
          let msg = table_error e.at Table.Bounds in
          if c.trace_gas then Printf.printf "aggregate_trap: %s\n" msg;
          vs', [Trapping msg @@ e.at]
        else if n = 0l then
          vs', []
        else
          let seg = elem frame.inst y in
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 d @@ e.at));
            Refer (Elem.load seg s);
            ReducedPlain (TableSet x);
            ReducedPlain (Const (I32 (I32.add d 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.add s 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (TableInit (x, y));
          ]

      | ElemDrop x, vs ->
        let seg = elem frame.inst x in
        Elem.drop seg;
        vs, []

      | Load {offset; ty; pack; _}, Num (I32 i) :: vs' ->
        let mem = memory frame.inst (0l @@ e.at) in
        let a = I64_convert.extend_i32_u i in
        (try
          let n =
            match pack with
            | None -> Memory.load_num mem a offset ty
            | Some (sz, ext) -> Memory.load_num_packed sz ext mem a offset ty
          in Num n :: vs', []
        with exn -> vs', [Trapping (memory_error e.at exn) @@ e.at])

      | Store {offset; pack; _}, Num n :: Num (I32 i) :: vs' ->
        let mem = memory frame.inst (0l @@ e.at) in
        let a = I64_convert.extend_i32_u i in
        (try
          (match pack with
          | None -> Memory.store_num mem a offset n
          | Some sz -> Memory.store_num_packed sz mem a offset n
          );
          vs', []
        with exn -> vs', [Trapping (memory_error e.at exn) @@ e.at]);

      | VecLoad {offset; ty; pack; _}, Num (I32 i) :: vs' ->
        let mem = memory frame.inst (0l @@ e.at) in
        let addr = I64_convert.extend_i32_u i in
        (try
          let v =
            match pack with
            | None -> Memory.load_vec mem addr offset ty
            | Some (sz, ext) ->
              Memory.load_vec_packed sz ext mem addr offset ty
          in Vec v :: vs', []
        with exn -> vs', [Trapping (memory_error e.at exn) @@ e.at])

      | VecStore {offset; _}, Vec v :: Num (I32 i) :: vs' ->
        let mem = memory frame.inst (0l @@ e.at) in
        let addr = I64_convert.extend_i32_u i in
        (try
          Memory.store_vec mem addr offset v;
          vs', []
        with exn -> vs', [Trapping (memory_error e.at exn) @@ e.at]);

      | VecLoadLane ({offset; ty; pack; _}, j), Vec (V128 v) :: Num (I32 i) :: vs' ->
        let mem = memory frame.inst (0l @@ e.at) in
        let addr = I64_convert.extend_i32_u i in
        (try
          let v =
            match pack with
            | Pack8 ->
              V128.I8x16.replace_lane j v
                (I32Num.of_num 0 (Memory.load_num_packed Pack8 SX mem addr offset I32Type))
            | Pack16 ->
              V128.I16x8.replace_lane j v
                (I32Num.of_num 0 (Memory.load_num_packed Pack16 SX mem addr offset I32Type))
            | Pack32 ->
              V128.I32x4.replace_lane j v
                (I32Num.of_num 0 (Memory.load_num mem addr offset I32Type))
            | Pack64 ->
              V128.I64x2.replace_lane j v
                (I64Num.of_num 0 (Memory.load_num mem addr offset I64Type))
          in Vec (V128 v) :: vs', []
        with exn -> vs', [Trapping (memory_error e.at exn) @@ e.at])

      | VecStoreLane ({offset; ty; pack; _}, j), Vec (V128 v) :: Num (I32 i) :: vs' ->
        let mem = memory frame.inst (0l @@ e.at) in
        let addr = I64_convert.extend_i32_u i in
        (try
          (match pack with
          | Pack8 ->
            Memory.store_num_packed Pack8 mem addr offset (I32 (V128.I8x16.extract_lane_s j v))
          | Pack16 ->
            Memory.store_num_packed Pack16 mem addr offset (I32 (V128.I16x8.extract_lane_s j v))
          | Pack32 ->
            Memory.store_num mem addr offset (I32 (V128.I32x4.extract_lane_s j v))
          | Pack64 ->
            Memory.store_num mem addr offset (I64 (V128.I64x2.extract_lane_s j v))
          );
          vs', []
        with exn -> vs', [Trapping (memory_error e.at exn) @@ e.at])

      | MemorySize, vs ->
        let mem = memory frame.inst (0l @@ e.at) in
        Num (I32 (Memory.size mem)) :: vs, []

      | MemoryGrow, Num (I32 delta) :: vs' ->
        let mem = memory frame.inst (0l @@ e.at) in
        let old_size = Memory.size mem in
        let result =
          try Memory.grow mem delta; old_size
          with Memory.SizeOverflow | Memory.SizeLimit | Memory.OutOfMemory -> -1l
        in Num (I32 result) :: vs', []

      | MemoryFill, Num (I32 n) :: Num k :: Num (I32 i) :: vs' ->
        if mem_oob frame (0l @@ e.at) i n then
          let msg = memory_error e.at Memory.Bounds in
          if c.trace_gas then Printf.printf "aggregate_trap: %s\n" msg;
          vs', [Trapping msg @@ e.at]
        else if n = 0l then
          vs', []
        else
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 i @@ e.at));
            ReducedPlain (Const (k @@ e.at));
            ReducedPlain (Store
              {ty = I32Type; align = 0; offset = 0l; pack = Some Pack8});
            ReducedPlain (Const (I32 (I32.add i 1l) @@ e.at));
            ReducedPlain (Const (k @@ e.at));
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (MemoryFill);
          ]

      | MemoryCopy, Num (I32 n) :: Num (I32 s) :: Num (I32 d) :: vs' ->
        if mem_oob frame (0l @@ e.at) s n || mem_oob frame (0l @@ e.at) d n then
          let msg = memory_error e.at Memory.Bounds in
          if c.trace_gas then Printf.printf "aggregate_trap: %s\n" msg;
          vs', [Trapping msg @@ e.at]
        else if n = 0l then
          vs', []
        else if I32.le_u d s then
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 d @@ e.at));
            ReducedPlain (Const (I32 s @@ e.at));
            ReducedPlain (Load
              {ty = I32Type; align = 0; offset = 0l; pack = Some (Pack8, ZX)});
            ReducedPlain (Store
              {ty = I32Type; align = 0; offset = 0l; pack = Some Pack8});
            ReducedPlain (Const (I32 (I32.add d 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.add s 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (MemoryCopy);
          ]
        else (* d > s *)
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 (I32.add d 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.add s 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (MemoryCopy);
            ReducedPlain (Const (I32 d @@ e.at));
            ReducedPlain (Const (I32 s @@ e.at));
            ReducedPlain (Load
              {ty = I32Type; align = 0; offset = 0l; pack = Some (Pack8, ZX)});
            ReducedPlain (Store
              {ty = I32Type; align = 0; offset = 0l; pack = Some Pack8});
          ]

      | MemoryInit x, Num (I32 n) :: Num (I32 s) :: Num (I32 d) :: vs' ->
        if mem_oob frame (0l @@ e.at) d n || data_oob frame x s n then
          let msg = memory_error e.at Memory.Bounds in
          if c.trace_gas then Printf.printf "aggregate_trap: %s\n" msg;
          vs', [Trapping msg @@ e.at]
        else if n = 0l then
          vs', []
        else
          let seg = data frame.inst x in
          let a = I64_convert.extend_i32_u s in
          let b = Data.load seg a in
          vs', List.map (at e.at) [
            ReducedPlain (Const (I32 d @@ e.at));
            ReducedPlain (Const (I32 (I32.of_int_u (Char.code b)) @@ e.at));
            ReducedPlain (Store
              {ty = I32Type; align = 0; offset = 0l; pack = Some Pack8});
            ReducedPlain (Const (I32 (I32.add d 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.add s 1l) @@ e.at));
            ReducedPlain (Const (I32 (I32.sub n 1l) @@ e.at));
            Plain (MemoryInit x);
          ]

      | DataDrop x, vs ->
        let seg = data frame.inst x in
        Data.drop seg;
        vs, []

      | RefNull t, vs' ->
        Ref (NullRef t) :: vs', []

      | RefIsNull, Ref r :: vs' ->
        (match r with
        | NullRef _ ->
          Num (I32 1l) :: vs', []
        | _ ->
          Num (I32 0l) :: vs', []
        )

      | RefFunc x, vs' ->
        let f = func frame.inst x in
        Ref (FuncRef f) :: vs', []

      | Const n, vs ->
        Num n.it :: vs, []

      | Test testop, Num n :: vs' ->
        (try value_of_bool (Eval_num.eval_testop testop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | Compare relop, Num n2 :: Num n1 :: vs' ->
        (try value_of_bool (Eval_num.eval_relop relop n1 n2) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | Unary unop, Num n :: vs' ->
        (try Num (Eval_num.eval_unop unop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | Binary binop, Num n2 :: Num n1 :: vs' ->
        (try Num (Eval_num.eval_binop binop n1 n2) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | Convert cvtop, Num n :: vs' ->
        (try Num (Eval_num.eval_cvtop cvtop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecConst v, vs ->
        Vec v.it :: vs, []

      | VecTest testop, Vec n :: vs' ->
        (try value_of_bool (Eval_vec.eval_testop testop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecUnary unop, Vec n :: vs' ->
        (try Vec (Eval_vec.eval_unop unop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecBinary binop, Vec n2 :: Vec n1 :: vs' ->
        (try Vec (Eval_vec.eval_binop binop n1 n2) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecCompare relop, Vec n2 :: Vec n1 :: vs' ->
        (try Vec (Eval_vec.eval_relop relop n1 n2) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecConvert cvtop, Vec n :: vs' ->
        (try Vec (Eval_vec.eval_cvtop cvtop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecShift shiftop, Num s :: Vec v :: vs' ->
        (try Vec (Eval_vec.eval_shiftop shiftop v s) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecBitmask bitmaskop, Vec v :: vs' ->
        (try Num (Eval_vec.eval_bitmaskop bitmaskop v) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecTestBits vtestop, Vec n :: vs' ->
        (try value_of_bool (Eval_vec.eval_vtestop vtestop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecUnaryBits vunop, Vec n :: vs' ->
        (try Vec (Eval_vec.eval_vunop vunop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecBinaryBits vbinop, Vec n2 :: Vec n1 :: vs' ->
        (try Vec (Eval_vec.eval_vbinop vbinop n1 n2) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecTernaryBits vternop, Vec v3 :: Vec v2 :: Vec v1 :: vs' ->
        (try Vec (Eval_vec.eval_vternop vternop v1 v2 v3) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecSplat splatop, Num n :: vs' ->
        (try Vec (Eval_vec.eval_splatop splatop n) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecExtract extractop, Vec v :: vs' ->
        (try Num (Eval_vec.eval_extractop extractop v) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | VecReplace replaceop, Num r :: Vec v :: vs' ->
        (try Vec (Eval_vec.eval_replaceop replaceop v r) :: vs', []
        with exn -> vs', [Trapping (numeric_error e.at exn) @@ e.at])

      | _ ->
        let s1 = string_of_values (List.rev vs) in
        let s2 = string_of_value_types (List.map type_of_value (List.rev vs)) in
        Crash.error e.at
          ("missing or ill-typed operand on stack (" ^ s1 ^ " : " ^ s2 ^ ")")
      )

    | Refer r, vs ->
      Ref r :: vs, []

    | Trapping msg, vs ->
      assert false

    | Returning vs', vs ->
      Crash.error e.at "undefined frame"

    | Breaking (k, vs'), vs ->
      Crash.error e.at "undefined label"

    | Label (n, es0, (vs', [])), vs ->
      vs' @ vs, []

    | Label (n, es0, (vs', {it = Trapping msg; at} :: es')), vs ->
      vs, [Trapping msg @@ at]

    | Label (n, es0, (vs', {it = Returning vs0; at} :: es')), vs ->
      vs, [Returning vs0 @@ at]

    | Label (n, es0, (vs', {it = Breaking (0l, vs0); at} :: es')), vs ->
      take n vs0 e.at @ vs, List.map plain es0

    | Label (n, es0, (vs', {it = Breaking (k, vs0); at} :: es')), vs ->
      vs, [Breaking (Int32.sub k 1l, vs0) @@ at]

    | Label (n, es0, code'), vs ->
      let c' = step {c with code = code'} in
      vs, [Label (n, es0, c'.code) @@ e.at]

    | Frame (n, frame', (vs', [])), vs ->
      vs' @ vs, []

    | Frame (n, frame', (vs', {it = Trapping msg; at} :: es')), vs ->
      vs, [Trapping msg @@ at]

    | Frame (n, frame', (vs', {it = Returning vs0; at} :: es')), vs ->
      take n vs0 e.at @ vs, []

    | Frame (n, frame', code'), vs ->
      let c' = step { c with frame = frame'; code = code'; budget = c.budget - 1 } in
      vs, [Frame (n, c'.frame, c'.code) @@ e.at]

    | Invoke func, vs when c.budget = 0 ->
      Exhaustion.error e.at "call stack exhausted"

    | Invoke func, vs ->
      let FuncType (ins, out) = Func.type_of func in
      let n1, n2 = Lib.List32.length ins, Lib.List32.length out in
      let args, vs' = take n1 vs e.at, drop n1 vs e.at in
      (match func with
      | Func.AstFunc (t, inst', f) ->
        let locals' = List.rev args @ List.map default_value f.it.locals in
        let frame' = {inst = !inst'; locals = List.map ref locals'} in
        let instr' = [Label (n2, [], ([], List.map plain f.it.body)) @@ f.at] in
        vs', [Frame (n2, frame', ([], instr')) @@ e.at]

      | Func.GasIntrinsic ->
        (match List.hd args with
          | Num (I64 v) -> Printf.printf "charge_gas: %Lu\n" v
          | _ -> Crash.error e.at "wrong types of arguments");
        vs', []

      | Func.CopyIntrinsic | Func.FillIntrinsic | Func.InitIntrinsic ->
        (match args with
          | [Num (I64 c); Num (I64 l); Num (I32 cnt)] -> let open Int64 in
            logand (of_int32 cnt) 0xFFFF_FFFFL
            |> mul l
            |> add c
            |> Printf.printf "charge_aggregate: %Lu\n"
          | _ -> Crash.error e.at "wrong types of arguments");
        (drop 2l vs e.at), []

      | Func.StackIntrinsic ->
        (match args with
          | [Num (I64 a1); Num (I64 a2)] -> Printf.printf "reserve_stack: %Lu %Lu\n" a2 a1
          | _ -> Crash.error e.at "wrong types of arguments");
        vs', []

      | Func.UnstackIntrinsic ->
        (match args with
          | [Num (I64 a1); Num (I64 a2)] -> Printf.printf "return_stack: %Lu %Lu\n" a2 a1
          | _ -> Crash.error e.at "wrong types of arguments");
        vs', []

      | Func.HostFunc (t, f) ->
        try List.rev (f (List.rev args)) @ vs', []
        with Crash (_, msg) -> Crash.error e.at msg

      )
  in {c with code = vs', es' @ List.tl es}


let rec eval (c : config) : value stack =
  match c.code with
  | vs, [] ->
    vs

  | vs, {it = Trapping msg; at} :: _ ->
    Trap.error at msg

  | vs, es ->
    eval (step c)


(* Functions & Constants *)

let invoke (func : func_inst) (vs : value list) : value list =
  let inst, at = match func with
    | Func.AstFunc (_, inst, f) -> !inst, f.at
    | _ -> (new_module_inst !Flags.gas), no_region in
  let FuncType (ins, out) = Func.type_of func in
  if List.length vs <> List.length ins then
    Crash.error at "wrong number of arguments";
  if not (List.for_all2 (fun v -> (=) (type_of_value v)) vs ins) then
    Crash.error at "wrong types of arguments";
  let c = config inst (List.rev vs) [Invoke func @@ at] in
  try List.rev (eval c) with Stack_overflow ->
    Exhaustion.error at "call stack exhausted"

let eval_const (inst : module_inst) (const : const) : value =
  let c = config inst [] (List.map reduced_plain const.it) in
  match eval c with
  | [v] -> v
  | vs -> Crash.error const.at "wrong number of results on stack"


(* Modules *)

let create_func (inst : module_inst) (f : func) : func_inst =
  Func.alloc (type_ inst f.it.ftype) (ref inst) f

let create_table (inst : module_inst) (tab : table) : table_inst =
  let {ttype} = tab.it in
  let TableType (_lim, t) = ttype in
  Table.alloc ttype (NullRef t)

let create_memory (inst : module_inst) (mem : memory) : memory_inst =
  let {mtype} = mem.it in
  Memory.alloc mtype

let create_global (inst : module_inst) (glob : global) : global_inst =
  let {gtype; ginit} = glob.it in
  let v = eval_const inst ginit in
  Global.alloc gtype v

let create_export (inst : module_inst) (ex : export) : export_inst =
  let {name; edesc} = ex.it in
  let ext =
    match edesc.it with
    | FuncExport x -> ExternFunc (func inst x)
    | TableExport x -> ExternTable (table inst x)
    | MemoryExport x -> ExternMemory (memory inst x)
    | GlobalExport x -> ExternGlobal (global inst x)
  in (name, ext)

let create_elem (inst : module_inst) (seg : elem_segment) : elem_inst =
  let {etype; einit; _} = seg.it in
  Elem.alloc (List.map (fun c -> as_ref (eval_const inst c)) einit)

let create_data (inst : module_inst) (seg : data_segment) : data_inst =
  let {dinit; _} = seg.it in
  Data.alloc dinit


let add_import (m : module_) (ext : extern) (im : import) (inst : module_inst)
  : module_inst =
  if not (match_extern_type (extern_type_of ext) (import_type m im)) then
    Link.error im.at ("incompatible import type for " ^
      "\"" ^ Utf8.encode im.it.module_name ^ "\" " ^
      "\"" ^ Utf8.encode im.it.item_name ^ "\": " ^
      "expected " ^ Types.string_of_extern_type (import_type m im) ^
      ", got " ^ Types.string_of_extern_type (extern_type_of ext));
  match ext with
  | ExternFunc func -> {inst with funcs = func :: inst.funcs}
  | ExternTable tab -> {inst with tables = tab :: inst.tables}
  | ExternMemory mem -> {inst with memories = mem :: inst.memories}
  | ExternGlobal glob -> {inst with globals = glob :: inst.globals}

let init_func (inst : module_inst) (func : func_inst) =
  match func with
  | Func.AstFunc (_, inst_ref, _) -> inst_ref := inst
  | _ -> assert false

let run_elem i elem =
  let at = elem.it.emode.at in
  let x = i @@ at in
  match elem.it.emode.it with
  | Passive -> []
  | Active {index; offset} ->
    offset.it @ [
      Const (I32 0l @@ at) @@ at;
      Const (I32 (Lib.List32.length elem.it.einit) @@ at) @@ at;
      TableInit (index, x) @@ at;
      ElemDrop x @@ at
    ]
  | Declarative ->
    [ElemDrop x @@ at]

let run_data i data =
  let at = data.it.dmode.at in
  let x = i @@ at in
  match data.it.dmode.it with
  | Passive -> []
  | Active {index; offset} ->
    assert (index.it = 0l);
    offset.it @ [
      Const (I32 0l @@ at) @@ at;
      Const (I32 (Int32.of_int (String.length data.it.dinit)) @@ at) @@ at;
      MemoryInit x @@ at;
      DataDrop x @@ at
    ]
  | Declarative -> assert false

let run_start start =
  [Call start.it.sfunc @@ start.at]

let init (m : module_) (exts : extern list) : module_inst =
  let
    { imports; tables; memories; globals; funcs; types;
      exports; elems; datas; start
    } = m.it
  in
  if List.length exts <> List.length imports then
    Link.error m.at "wrong number of imports provided for initialisation";
  let new_mod_inst = new_module_inst !Flags.gas in
  let inst0 =
    { (List.fold_right2 (add_import m) exts imports new_mod_inst) with
      types = List.map (fun type_ -> type_.it) types }
  in
  let fs = List.map (create_func inst0) funcs in
  let inst1 = {inst0 with funcs = inst0.funcs @ fs} in
  let inst2 =
    { inst1 with
      tables = inst1.tables @ List.map (create_table inst1) tables;
      memories = inst1.memories @ List.map (create_memory inst1) memories;
      globals = inst1.globals @ List.map (create_global inst1) globals;
    }
  in
  let inst =
    { inst2 with
      exports = List.map (create_export inst2) exports;
      elems = List.map (create_elem inst2) elems;
      datas = List.map (create_data inst2) datas;
    }
  in
  List.iter (init_func inst) fs;
  let es_elem = List.concat (Lib.List32.mapi run_elem elems) in
  let es_data = List.concat (Lib.List32.mapi run_data datas) in
  let es_start = Lib.Option.get (Lib.Option.map run_start start) [] in
  let c_init = config_for_init inst [] (List.map reduced_plain (es_elem @ es_data)) in
  ignore (eval c_init);
  let c = config inst [] (List.map reduced_plain es_start) in
  ignore (eval c);
  inst
