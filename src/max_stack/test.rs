use super::{Error, Frame, StackSizeVisitor};
use prefix_sum_vec::PrefixSumVec;
use wasmparser::{BlockType, ValType};

struct TestConfig {
    value_size: u8,
    local_size: u8,
}

impl super::Config for TestConfig {
    fn size_of_value(&self, _: wasmparser::ValType) -> u8 {
        self.value_size
    }

    fn size_of_function_activation(
        &self,
        locals: &prefix_sum_vec::PrefixSumVec<wasmparser::ValType, u32>,
    ) -> u64 {
        let locals = locals
            .max_index()
            .copied()
            .unwrap_or(0)
            .checked_sub(1)
            .unwrap_or(0);
        u64::from(locals) * u64::from(self.local_size)
    }
}

impl Default for TestConfig {
    fn default() -> Self {
        TestConfig {
            value_size: 9,
            local_size: 5,
        }
    }
}

macro_rules! visitor {
    (
        let $binding: ident = StackSizeVisitor {
            $(functions: $functions: expr,)?
            $(types: $types: expr,)?
            $(globals: $globals: expr,)?
            $(tables: $tables: expr,)?
        }
    ) => {
        let config = TestConfig::default();
        let mut frames = Vec::new();
        let mut operands = Vec::new();
        let mut locals = PrefixSumVec::new();
        #[allow(unused_variables)]
        let functions: &[u32] = &[];
        $(let functions = $functions;)?
        #[allow(unused_variables)]
        let types: &[wasmparser::Type] = &[];
        $(let types = $types;)?
        #[allow(unused_variables)]
        let globals: &[wasmparser::ValType] = &[];
        $(let globals = $globals;)?
        #[allow(unused_variables)]
        let tables: &[wasmparser::ValType] = &[];
        $(let tables = $tables;)?
        #[allow(unused_mut)]
        let mut $binding = StackSizeVisitor {
            config: &config,
            offset: 0,
            functions,
            types,
            globals,
            tables,
            locals: &mut locals,
            operands: &mut operands,
            size: 0,
            max_size: 0,
            frames: &mut frames,
            current_frame: Frame {
                height: 0,
                block_type: wasmparser::BlockType::Empty,
                stack_polymorphic: false,
            },
        };
    }
}

#[test]
fn test_function_type_index_oob() {
    visitor! { let visitor = StackSizeVisitor {
        functions: &[1],
        types: &[wasmparser::Type::Func(wasmparser::FuncType::new([], []))],
    } };
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
    visitor! { let visitor = StackSizeVisitor {} };
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
    visitor! { let visitor = StackSizeVisitor {} };
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
    visitor! { let visitor = StackSizeVisitor {
        types: &[
            wasmparser::Type::Func(wasmparser::FuncType::new([ValType::V128], [ValType::FuncRef]))
        ],
    } };
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
    visitor! { let visitor = StackSizeVisitor {} };

    assert_eq!(0, visitor.size);
    visitor.make_polymorphic();
    visitor
        .pop()
        .expect("pops from polymorphic frames should never fail, even with empty stack");

    visitor
        .new_frame(BlockType::Empty, 0)
        .expect("pushing a new frame should succeed");
    visitor.push(ValType::I32);
    assert_eq!(0, visitor.size);
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
    visitor! { let visitor = StackSizeVisitor {} };

    assert_eq!(0, visitor.size);

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

    assert!(!visitor.current_frame.stack_polymorphic);
    let Err(Error::EmptyStack(_)) = visitor.pop() else {
        panic!("setting frame polymorphic should not affect the parent frames");
    };
    visitor.push(ValType::V128);
    assert!(visitor.pop().is_ok());
    assert_eq!(u64::from(visitor.config.value_size), visitor.max_size);
}

#[test]
fn test_pop_many() {
    visitor! { let visitor = StackSizeVisitor {} };

    assert_eq!(0, visitor.operands.len());
    visitor.push(ValType::V128);
    assert_eq!(1, visitor.operands.len());

    let Err(Error::EmptyStack(_)) = visitor.pop_many(2) else {
        panic!("pop_many cannot pop more than there are operands");
    };
    assert!(visitor.pop_many(0).is_ok());
    assert_eq!(1, visitor.operands.len());
    assert!(visitor.pop_many(1).is_ok());
    assert_eq!(0, visitor.operands.len());

    visitor.push(ValType::V128);
    visitor.push(ValType::V128);
    assert_eq!(2, visitor.operands.len());
    let Err(Error::EmptyStack(_)) = visitor.pop_many(3) else {
        panic!("pop_many cannot pop more than there are operands");
    };
    assert!(visitor.pop_many(2).is_ok());
    assert_eq!(0, visitor.operands.len());
}

#[test]
fn test_operand_stack_size() {
    visitor! { let visitor = StackSizeVisitor {} };
    assert_eq!(0, visitor.size);
    assert_eq!(0, visitor.max_size);
    visitor.push(ValType::V128);
    assert_eq!(9, visitor.size);
    assert_eq!(9, visitor.max_size);
    visitor.push(ValType::V128);
    assert_eq!(18, visitor.size);
    assert_eq!(18, visitor.max_size);
    visitor.push(ValType::V128);
    assert_eq!(27, visitor.size);
    assert_eq!(27, visitor.max_size);
    visitor.pop().expect("non empty operand stack");
    assert_eq!(18, visitor.size);
    assert_eq!(27, visitor.max_size);
    visitor.pop().expect("non empty operand stack");
    assert_eq!(9, visitor.size);
    assert_eq!(27, visitor.max_size);
    visitor.pop().expect("non empty operand stack");
    assert_eq!(0, visitor.size);
    assert_eq!(27, visitor.max_size);
    visitor.pop().err().expect("empty operand stack");
    assert_eq!(0, visitor.size);
    assert_eq!(27, visitor.max_size);
}

#[test]
fn test_operand_stack_size_with_frames() {
    visitor! { let visitor = StackSizeVisitor {
        types: &[
            wasmparser::Type::Func(wasmparser::FuncType::new([ValType::V128], []))
        ],
    } };
    assert_eq!(0, visitor.size);
    assert_eq!(0, visitor.max_size);
    let Err(Error::EmptyStack(_)) = visitor.new_frame(BlockType::FuncType(0), 1) else {
        panic!("can't shift operands past empty stack!");
    };
    visitor.push(ValType::V128);
    assert_eq!(9, visitor.size);
    assert_eq!(9, visitor.max_size);
    let Err(Error::EmptyStack(_)) = visitor.new_frame(BlockType::FuncType(0), 2) else {
        panic!("can't shift operands past empty stack!");
    };
    assert_eq!(Some(()), visitor.new_frame(BlockType::FuncType(0), 0).ok());
    assert_eq!(9, visitor.size);
    assert_eq!(9, visitor.max_size);
    let Ok(Some(_)) = visitor.end_frame() else {
        panic!("should be able to end frame we pushed recently");
    };
    assert_eq!(9, visitor.size);
    assert_eq!(9, visitor.max_size);

    assert_eq!(Some(()), visitor.new_frame(BlockType::FuncType(0), 1).ok());
    assert_eq!(1, visitor.operands.len());
    assert_eq!(9, visitor.size);
    assert_eq!(9, visitor.max_size);
    let Ok(Some(_)) = visitor.end_frame() else {
        panic!("should be able to end frame we pushed recently");
    };
    assert_eq!(0, visitor.operands.len());
    assert_eq!(0, visitor.size);
    assert_eq!(9, visitor.max_size);

    visitor.push(ValType::V128);
    assert_eq!(9, visitor.size);
    assert_eq!(9, visitor.max_size);
    visitor.push(ValType::V128);
    assert_eq!(18, visitor.size);
    assert_eq!(18, visitor.max_size);
    assert_eq!(Some(()), visitor.new_frame(BlockType::FuncType(0), 1).ok());
    assert_eq!(2, visitor.operands.len());
    assert_eq!(18, visitor.size);
    assert_eq!(18, visitor.max_size);
    visitor.push(ValType::V128);
    assert_eq!(27, visitor.size);
    assert_eq!(27, visitor.max_size);
    let Ok(Some(_)) = visitor.end_frame() else {
        panic!("should be able to end frame we pushed recently");
    };
    assert_eq!(1, visitor.operands.len());
    assert_eq!(9, visitor.size);
    assert_eq!(27, visitor.max_size);
}
