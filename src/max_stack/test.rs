use super::{Error, Frame, FunctionState, ModuleState};
use crate::max_stack::Config;
use crate::tests::SizeConfig as TestConfig;
use wasmparser::{BlockType, ValType};

fn new_state() -> (TestConfig, ModuleState, FunctionState) {
    (
        TestConfig::default(),
        ModuleState::new(),
        FunctionState::new(),
    )
}

#[test]
fn test_function_type_index_oob() {
    let (config, mut mstate, mut fnstate) = new_state();
    mstate.functions = vec![1];
    mstate.types = vec![wasmparser::Type::Func(wasmparser::FuncType::new([], []))];
    let visitor = config.make_visitor(&mstate, &mut fnstate);

    assert_eq!(Some(1), visitor.function_type_index(0).ok());
    let Err(Error::FunctionIndex(1)) = visitor.function_type_index(1) else {
        panic!("function_type_index(1) did not fail")
    };
    let Err(Error::TypeIndex(1)) = visitor.type_params_results(1) else {
        panic!("type_params_results(1) did not fail")
    };
}

#[test]
fn test_with_block_types_empty() {
    let (config, mstate, mut fnstate) = new_state();
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);

    let mut called = false;
    visitor
        .with_block_types(BlockType::Empty, |_, params, results| {
            assert_eq!(params, []);
            assert_eq!(results, []);
            called = true;
            Ok(())
        })
        .expect("should return Ok");
    assert!(
        called,
        "BlockType::Empty should still call with_block_types callback"
    );
}

#[test]
fn test_with_block_types_type() {
    let (config, mstate, mut fnstate) = new_state();
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let mut called = false;
    visitor
        .with_block_types(BlockType::Type(ValType::V128), |_, params, results| {
            assert_eq!(params, []);
            assert_eq!(results, [ValType::V128]);
            called = true;
            Ok(())
        })
        .expect("should return Ok");
    assert!(
        called,
        "BlockType::Type should call with_block_types callback"
    );
}

#[test]
fn test_with_block_types_functype() {
    let (config, mut mstate, mut fnstate) = new_state();
    mstate.types = vec![wasmparser::Type::Func(wasmparser::FuncType::new(
        [ValType::V128],
        [ValType::FuncRef],
    ))];
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);

    let mut called = false;
    visitor
        .with_block_types(BlockType::FuncType(0), |_, params, results| {
            assert_eq!(params, [ValType::V128]);
            assert_eq!(results, [ValType::FuncRef]);
            called = true;
            Ok(())
        })
        .expect("should return Ok");
    assert!(
        called,
        "BlockType::FuncType should call with_block_types callback"
    );

    let result = visitor.with_block_types(BlockType::FuncType(1), |_, _, _| {
        panic!("should not get called");
    });
    let Err(Error::TypeIndex(1)) = result else {
        panic!("BlockType::FuncType(1) should have failed but it did not");
    };
}

#[test]
fn test_nested_polymorphic_frames() {
    let (config, mstate, mut fnstate) = new_state();

    assert_eq!(0, fnstate.size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    visitor.make_polymorphic();
    visitor
        .pop()
        .expect("pops from polymorphic frames should never fail, even with empty stack");

    visitor
        .new_frame(BlockType::Empty, 0)
        .expect("pushing a new frame should succeed");
    visitor.push(ValType::I32);
    assert_eq!(0, fnstate.size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    visitor
        .pop()
        .expect("pops from polymorphic frames should never fail");
    visitor
        .pop()
        .expect("pops from polymorphic frames should never fail");
    let Ok(Some(Frame { stack_polymorphic: true, .. })) = visitor.end_frame() else {
        panic!("pushing a frame when parent frame is already polymorphic should \
                have made this frame polymorphic too");
    };
}

#[test]
fn test_nested_polymorphic_frames_2() {
    let (config, mstate, mut fnstate) = new_state();
    assert_eq!(0, fnstate.size);

    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    visitor
        .new_frame(BlockType::Empty, 0)
        .expect("pushing a new frame should succeed");
    visitor.make_polymorphic();
    visitor
        .pop()
        .expect("pops from polymorphic frames should never fail, even with empty stack");
    let Ok(Some(Frame { stack_polymorphic: true, .. })) = visitor.end_frame() else {
        panic!("pushing a frame when parent frame is already polymorphic should \
                have made this frame polymorphic too");
    };

    assert!(!fnstate.current_frame.stack_polymorphic);

    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Err(Error::EmptyStack(_)) = visitor.pop() else {
        panic!("setting frame polymorphic should not affect the parent frames");
    };
    visitor.push(ValType::V128);
    assert!(visitor.pop().is_ok());
    assert_eq!(u64::from(visitor.config.value_size), fnstate.max_size);
}

#[test]
fn test_pop_many() {
    let (config, mstate, mut fnstate) = new_state();

    assert_eq!(0, fnstate.operands.len());
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    visitor.push(ValType::V128);
    assert_eq!(1, fnstate.operands.len());

    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Err(Error::EmptyStack(_)) = visitor.pop_many(2) else {
        panic!("pop_many cannot pop more than there are operands");
    };
    assert!(visitor.pop_many(0).is_ok());
    assert_eq!(1, fnstate.operands.len());
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    assert!(visitor.pop_many(1).is_ok());
    assert_eq!(0, fnstate.operands.len());

    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    visitor.push(ValType::V128);
    visitor.push(ValType::V128);
    assert_eq!(2, fnstate.operands.len());
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Err(Error::EmptyStack(_)) = visitor.pop_many(3) else {
        panic!("pop_many cannot pop more than there are operands");
    };
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    assert!(visitor.pop_many(2).is_ok());
    assert_eq!(0, fnstate.operands.len());
}

#[test]
fn test_operand_stack_size() {
    let (config, mstate, mut fnstate) = new_state();

    assert_eq!(0, fnstate.size);
    assert_eq!(0, fnstate.max_size);
    config.make_visitor(&mstate, &mut fnstate).push(ValType::V128);
    assert_eq!(9, fnstate.size);
    assert_eq!(9, fnstate.max_size);
    config.make_visitor(&mstate, &mut fnstate).push(ValType::V128);
    assert_eq!(18, fnstate.size);
    assert_eq!(18, fnstate.max_size);
    config.make_visitor(&mstate, &mut fnstate).push(ValType::V128);
    assert_eq!(27, fnstate.size);
    assert_eq!(27, fnstate.max_size);
    config
        .make_visitor(&mstate, &mut fnstate)
        .pop()
        .expect("non empty operand stack");
    assert_eq!(18, fnstate.size);
    assert_eq!(27, fnstate.max_size);
    config
        .make_visitor(&mstate, &mut fnstate)
        .pop()
        .expect("non empty operand stack");
    assert_eq!(9, fnstate.size);
    assert_eq!(27, fnstate.max_size);
    config
        .make_visitor(&mstate, &mut fnstate)
        .pop()
        .expect("non empty operand stack");
    assert_eq!(0, fnstate.size);
    assert_eq!(27, fnstate.max_size);
    config
        .make_visitor(&mstate, &mut fnstate)
        .pop()
        .err()
        .expect("empty operand stack");
    assert_eq!(0, fnstate.size);
    assert_eq!(27, fnstate.max_size);
}

#[test]
fn test_operand_stack_size_with_frames() {
    let (config, mut mstate, mut fnstate) = new_state();
    mstate.types = vec![wasmparser::Type::Func(wasmparser::FuncType::new(
        [ValType::V128],
        [],
    ))];

    assert_eq!(0, fnstate.size);
    assert_eq!(0, fnstate.max_size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Err(Error::EmptyStack(_)) = visitor.new_frame(BlockType::FuncType(0), 1) else {
        panic!("can't shift operands past empty stack!");
    };
    visitor.push(ValType::V128);
    assert_eq!(9, fnstate.size);
    assert_eq!(9, fnstate.max_size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Err(Error::EmptyStack(_)) = visitor.new_frame(BlockType::FuncType(0), 2) else {
        panic!("can't shift operands past empty stack!");
    };
    assert_eq!(Some(()), visitor.new_frame(BlockType::FuncType(0), 0).ok());
    assert_eq!(9, fnstate.size);
    assert_eq!(9, fnstate.max_size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Ok(Some(_)) = visitor.end_frame() else {
        panic!("should be able to end frame we pushed recently");
    };
    assert_eq!(9, fnstate.size);
    assert_eq!(9, fnstate.max_size);

    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    assert_eq!(Some(()), visitor.new_frame(BlockType::FuncType(0), 1).ok());
    assert_eq!(1, fnstate.operands.len());
    assert_eq!(9, fnstate.size);
    assert_eq!(9, fnstate.max_size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Ok(Some(_)) = visitor.end_frame() else {
        panic!("should be able to end frame we pushed recently");
    };
    assert_eq!(0, fnstate.operands.len());
    assert_eq!(0, fnstate.size);
    assert_eq!(9, fnstate.max_size);

    config.make_visitor(&mstate, &mut fnstate).push(ValType::V128);
    assert_eq!(9, fnstate.size);
    assert_eq!(9, fnstate.max_size);
    config.make_visitor(&mstate, &mut fnstate).push(ValType::V128);
    assert_eq!(18, fnstate.size);
    assert_eq!(18, fnstate.max_size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    assert_eq!(Some(()), visitor.new_frame(BlockType::FuncType(0), 1).ok());
    assert_eq!(2, fnstate.operands.len());
    assert_eq!(18, fnstate.size);
    assert_eq!(18, fnstate.max_size);
    config.make_visitor(&mstate, &mut fnstate).push(ValType::V128);
    assert_eq!(27, fnstate.size);
    assert_eq!(27, fnstate.max_size);
    let mut visitor = config.make_visitor(&mstate, &mut fnstate);
    let Ok(Some(_)) = visitor.end_frame() else {
        panic!("should be able to end frame we pushed recently");
    };
    assert_eq!(1, fnstate.operands.len());
    assert_eq!(9, fnstate.size);
    assert_eq!(27, fnstate.max_size);
}
