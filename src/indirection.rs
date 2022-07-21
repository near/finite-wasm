use crate::transform;
use bitvec::vec::BitVec;
use wasm_encoder::{self as we, Encode};
use wasmparser as wp;
use we::Section;
use wp::{BinaryReaderError, Operator};

#[derive(Debug)]
pub enum Error {
    IncompleteInput,
    Parse(&'static str, wp::BinaryReaderError),
    InvalidConstOperator,
    FunctionIndexToUsize(std::num::TryFromIntError),
    TooManyFunctions,
    // TODO: these want some information about where the error is occuring (byte offset?)
    ParseItemExpression,
    ParseExpressionEnd,
    MalformedCodeSection,
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use Error::*;
        match self {
            IncompleteInput => None,
            Parse(_, e) => Some(e),
            InvalidConstOperator => None,
            FunctionIndexToUsize(e) => Some(e),
            TooManyFunctions => None,
            ParseItemExpression => None,
            ParseExpressionEnd => None,
            MalformedCodeSection => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;
        let output = match self {
            IncompleteInput => "input is incomplete",
            Parse(kind, _) => {
                f.write_str("could not parse the ")?;
                kind
            }
            InvalidConstOperator => "item initialization expression is not a constant op",
            FunctionIndexToUsize(_) => "could not convert function index to usize",
            TooManyFunctions => "too many functions generated during instrumentation",
            ParseItemExpression => "item expression is not one of ref.func or ref.null",
            ParseExpressionEnd => "expected `end` instruction",
            MalformedCodeSection => "malformed code section",
        };
        f.write_str(output)
    }
}

pub struct Indirector<'a> {
    /// Input Wasm module binary encoding
    input_wasm: &'a [u8],
    /// List of functions that require indirection trampolines to be generated (these functions are
    /// referenced by a start section, exported, etc.)
    to_indirect: BitVec,
    /// Mapping from function indices in the original module to indices in the module with
    /// trampolines introduced.
    index_map: Vec<u32>,
    /// Contents of the type section.
    types: Vec<wp::Type>,
    /// Signatures of the functions in the original module (values are indices to the types vector).
    function_types: Vec<u32>,
}

impl<'a> Indirector<'a> {
    pub fn analyze(wasm: &'a [u8]) -> Result<Self, Error> {
        let parser = wp::Parser::new(0);
        let mut this = Self {
            input_wasm: wasm,
            to_indirect: BitVec::new(),
            index_map: Vec::new(),
            types: Vec::new(),
            function_types: Vec::new(),
        };
        let mut code_section_entries = 0u32;
        for payload in parser.parse_all(wasm) {
            // The only payloads we’re interested in for figuring out which functions need indirection
            // are the `ExportSection` and `ElementSection`. Once we figure out the functions we want
            // to indirect, we will want to also append our new trampolines to the `CodeSection`.
            match payload.map_err(|e| Error::Parse("module", e))? {
                wp::Payload::TypeSection(type_reader) => {
                    this.types = type_reader
                        .into_iter()
                        .collect::<Result<_, _>>()
                        .map_err(|e| Error::Parse("type section", e))?;
                }
                wp::Payload::FunctionSection(func_reader) => {
                    this.function_types = func_reader
                        .into_iter()
                        .collect::<Result<_, _>>()
                        .map_err(|e| Error::Parse("type section", e))?;
                    this.to_indirect = BitVec::repeat(false, this.function_types.len());
                }
                wp::Payload::StartSection { func, range: _ } => this.to_indirect.set(
                    usize::try_from(func).map_err(Error::FunctionIndexToUsize)?,
                    true,
                ),
                wp::Payload::ExportSection(mut exports) => {
                    for _ in 0..exports.get_count() {
                        let export = exports.read().map_err(|e| Error::Parse("export", e))?;
                        if let wp::ExternalKind::Func = export.kind {
                            this.to_indirect.set(
                                usize::try_from(export.index)
                                    .map_err(Error::FunctionIndexToUsize)?,
                                true,
                            );
                        }
                    }
                }
                wp::Payload::ElementSection(mut elements) => {
                    for _ in 0..elements.get_count() {
                        let element = elements.read().map_err(|e| Error::Parse("element", e))?;
                        match element.kind {
                            wp::ElementKind::Active {
                                table_index: _,
                                init_expr,
                            } => {
                                let ops = init_expr.get_operators_reader();
                                this.analyze_operators(ops)?;
                            }
                            wp::ElementKind::Passive | wp::ElementKind::Declared => (),
                        }
                        let items = element
                            .items
                            .get_items_reader()
                            .map_err(|e| Error::Parse("items", e))?;
                        this.analyze_items(items)?;
                    }
                }
                // Sanity checks in case validator has not been run before us.
                wp::Payload::CodeSectionStart { count, .. } => this.assert_function_count(count)?,
                wp::Payload::CodeSectionEntry(_) => {
                    code_section_entries += 1;
                }
                _ => {}
            }
        }
        this.assert_function_count(code_section_entries)?;

        // Instrumentation strategy is to keep the trampoline and the function definition close
        // together. This means that we’re ultimately going to be shifting around most if not all of
        // the function indices, and as a result all of the references will need to be adjusted as
        // well. The primary reason to go through all this effort is to ensure the cheapest kind of
        // relocation (none at all) is applicable, and to give opportunity for instruction cache reuse
        // (in case the following function’s prologue happens to already have been fetched alongside
        // the trampoline itself). That said, this is only applicable for VMs that compile code in a
        // naive way and don’t take locality into accoutn – something i suspect would be applicable to
        // most if not all engines.
        //
        // We can construct the mapping from the original function indices to the indices
        // post-instrumentation right away, this is not strictly necessary due to how WebAssembly
        // requires sections to be ordered, but this does simplify the code substantially anyway.
        let mut index_after_indirection = 0u32;
        this.index_map = this
            .to_indirect
            .iter()
            .map(|will_indirect| {
                let index = index_after_indirection;
                index_after_indirection = index.checked_add(*will_indirect as u32 + 1u32)?;
                Some(index + *will_indirect as u32)
            })
            .collect::<Option<_>>()
            .ok_or(Error::TooManyFunctions)?;
        Ok(this)
    }

    /// Collect functions referenced by element items.
    fn analyze_items(&mut self, mut items: wp::ElementItemsReader) -> Result<(), Error> {
        for _ in 0..items.get_count() {
            let item = items.read().map_err(|e| Error::Parse("item", e))?;
            match item {
                wp::ElementItem::Func(func) => self.to_indirect.set(
                    usize::try_from(func).map_err(Error::FunctionIndexToUsize)?,
                    true,
                ),
                wp::ElementItem::Expr(init) => {
                    let ops = init.get_operators_reader();
                    self.analyze_operators(ops)?;
                }
            }
        }
        Ok(())
    }

    fn assert_function_count(&self, found_functions: u32) -> Result<(), Error> {
        match u32::try_from(self.function_types.len()) {
            Ok(v) if v == found_functions => Ok(()),
            _ => Err(Error::MalformedCodeSection),
        }
    }

    fn analyze_operators(&mut self, mut ops: wp::OperatorsReader) -> Result<(), Error> {
        loop {
            let op = ops
                .read()
                .map_err(|e| Error::Parse("item init expression", e))?;
            match op {
                wp::Operator::End => break,
                wp::Operator::RefFunc { function_index } => self.to_indirect.set(
                    usize::try_from(function_index).map_err(Error::FunctionIndexToUsize)?,
                    true,
                ),
                wp::Operator::RefNull { .. }
                | wp::Operator::I32Const { .. }
                | wp::Operator::I64Const { .. }
                | wp::Operator::F32Const { .. }
                | wp::Operator::F64Const { .. }
                | wp::Operator::V128Const { .. } => {}
                // TRICKY: This is made especially tricky by the fact
                // that the only way this can occur at all is with imported
                // `global`s, and in that instance there is very little we
                // can do to control how the global is initialized. Luckily in theory the
                // only functions an embedder should be able to reference are the exported
                // functions and those are already indirected! If it wasn’t the case, we’d
                // have no meaningful mechanism to handle this case.
                wp::Operator::GlobalGet { .. } => {}
                // TODO: index and description.
                _ => return Err(Error::InvalidConstOperator),
            }
        }
        Ok(())
    }

    fn function_idx(&self, original_idx: u32) -> u32 {
        self.index_map[original_idx as usize]
    }

    fn trampoline_idx(&self, original_idx: u32) -> u32 {
        debug_assert!(
            self.to_indirect
                .get(original_idx as usize)
                .as_deref()
                .unwrap_or(&false),
            "can only obtain trampoline indices for functions that require indirection"
        );
        self.function_idx(original_idx) - 1
    }

    fn should_indirect(&self, original_idx: u32) -> bool {
        let original_idx =
            usize::try_from(original_idx).expect("should’ve run out of memory by now");
        *self
            .to_indirect
            .get(original_idx)
            .expect("this cannot occur (analysis for this)")
    }

    pub fn indirect(&self) -> Result<Vec<u8>, Error> {
        // For portions of the code where we don’t make any changes we will try to transplant them
        // verbatim. This is not always straightforward, but well worth it.
        let mut parser = wp::Parser::new(0);
        let mut wasm = self.input_wasm;
        let mut output = Vec::<u8>::with_capacity(wasm.len());
        let mut remaining_functions = 0;
        let mut function_index = 0u32;
        let mut code_section = we::CodeSection::new();
        loop {
            match parser.parse(wasm, true) {
                Err(e) => return Err(Error::Parse("module", e)),
                Ok(wp::Chunk::NeedMoreData(_)) => unreachable!(),
                Ok(wp::Chunk::Parsed { consumed, payload }) => {
                    let (this_section, remaining) = wasm.split_at(consumed);
                    wasm = remaining;

                    match payload {
                        // None of these can reference functions by an index, nor they contain any
                        // operations/instructions/initexprs
                        wp::Payload::Version { .. } => output.extend(this_section),
                        wp::Payload::TypeSection(_) => output.extend(this_section),
                        wp::Payload::ImportSection(_) => output.extend(this_section),
                        wp::Payload::TableSection(_) => output.extend(this_section),
                        wp::Payload::MemorySection(_) => output.extend(this_section),
                        wp::Payload::DataCountSection { .. } => output.extend(this_section),
                        wp::Payload::TagSection(_) => output.extend(this_section),
                        wp::Payload::End(_) => {
                            output.extend(this_section);
                            break;
                        }
                        // We don’t know how to handle these sections in any way, the two options
                        // we thus have are to discard the section or keep it as-is. We take the
                        // latter approach, although the section’s contents might have gotten
                        // invalidated by our modifications.
                        wp::Payload::UnknownSection { .. } => output.extend(this_section),
                        // We must introduce type declarations for all the trampolines we’re
                        // introducing.
                        wp::Payload::FunctionSection(reader) => {
                            let new_section = self.transform_function_section(reader)?;
                            output.push(new_section.id());
                            new_section.encode(&mut output);
                        }
                        // These refer to function indices directly.
                        wp::Payload::ExportSection(reader) => {
                            let new_section = self.transform_export_section(reader)?;
                            output.push(new_section.id());
                            new_section.encode(&mut output);
                        }
                        wp::Payload::StartSection { func, range: _ } => {
                            let new_section = we::StartSection {
                                function_index: self.trampoline_idx(func),
                            };
                            output.push(new_section.id());
                            new_section.encode(&mut output);
                        }
                        // Contains `InitExpr`s.
                        wp::Payload::DataSection(reader) => {
                            let new_section = self.transform_data_section(reader)?;
                            output.push(new_section.id());
                            new_section.encode(&mut output);
                        }
                        wp::Payload::GlobalSection(reader) => {
                            let new_section = self.transform_global_section(reader)?;
                            output.push(new_section.id());
                            new_section.encode(&mut output);
                        }
                        wp::Payload::ElementSection(reader) => {
                            let new_section = self.transform_element_section(reader)?;
                            output.push(new_section.id());
                            new_section.encode(&mut output);
                        }
                        // Finally the code section where we need to add our trampolines!
                        wp::Payload::CodeSectionStart { count, .. } => {
                            remaining_functions = count;
                        }
                        wp::Payload::CodeSectionEntry(body) => {
                            remaining_functions = remaining_functions.checked_sub(1).expect("TODO");
                            let body = self.transform_function_body(body)?;
                            if self.should_indirect(function_index) {
                                code_section.function(&self.generate_trampoline(function_index)?);
                                code_section.function(&body);
                            } else {
                                code_section.function(&body);
                            }
                            function_index += 1;
                            // Figure out a nicer way to do this :/
                            if remaining_functions == 0 {
                                output.push(code_section.id());
                                code_section.encode(&mut output);
                            }
                        }
                        // Some custom sections are well-known and we can modify them to maintain
                        // e.g. reasonable debugging experience.
                        wp::Payload::CustomSection(payload) => {
                            self.transform_custom_section(&mut output, payload)?;
                        }

                        // TODO: components proposal
                        wp::Payload::ModuleSection { .. }
                        | wp::Payload::InstanceSection(_)
                        | wp::Payload::AliasSection(_)
                        | wp::Payload::CoreTypeSection(_)
                        | wp::Payload::ComponentSection { .. }
                        | wp::Payload::ComponentInstanceSection(_)
                        | wp::Payload::ComponentAliasSection(_)
                        | wp::Payload::ComponentTypeSection(_)
                        | wp::Payload::ComponentCanonicalSection(_)
                        | wp::Payload::ComponentStartSection(_)
                        | wp::Payload::ComponentImportSection(_)
                        | wp::Payload::ComponentExportSection(_) => todo!(),
                    }
                }
            }
        }
        Ok(output)
    }

    fn transform_data_section(
        &self,
        mut reader: wp::DataSectionReader,
    ) -> Result<we::DataSection, Error> {
        let mut section = we::DataSection::new();
        for _ in 0..reader.get_count() {
            let data = reader.read().map_err(|e| Error::Parse("data", e))?;
            match data.kind {
                wp::DataKind::Passive => section.passive(data.data.iter().copied()),
                wp::DataKind::Active {
                    memory_index,
                    init_expr,
                } => section.active(
                    memory_index,
                    &self.transform_init_expr(init_expr)?,
                    data.data.iter().copied(),
                ),
            };
        }
        Ok(section)
    }

    fn parse_init_expr_op(&self, init_expr: wp::InitExpr<'a>) -> Result<wp::Operator<'a>, Error> {
        let mut reader = init_expr.get_operators_reader();
        let op = reader
            .read()
            .map_err(|e| Error::Parse("init expression", e))?;
        let end = reader
            .read()
            .map_err(|e| Error::Parse("init expression", e))?;
        if matches!(end, Operator::End) {
            Ok(op)
        } else {
            Err(Error::ParseExpressionEnd)
        }
    }

    fn transform_init_expr(&self, init_expr: wp::InitExpr) -> Result<we::Instruction, Error> {
        let op = self.parse_init_expr_op(init_expr)?;
        self.transform_op(op)
    }

    fn transform_op(&self, op: wp::Operator) -> Result<we::Instruction, Error> {
        use we::Instruction as I;
        use wp::Operator as O;
        Ok(match op {
            O::Call { function_index } => I::Call(self.function_idx(function_index)),
            O::ReturnCall { function_index: _ } => unimplemented!(),
            O::RefFunc { function_index } => I::RefFunc(self.function_idx(function_index)),
            otherwise => {
                transform::transform_op(otherwise).map_err(|e| Error::Parse("operator", e))?
            }
        })
    }

    fn transform_export_section(
        &self,
        reader: wp::ExportSectionReader,
    ) -> Result<we::ExportSection, Error> {
        use we::ExportKind as EK;
        use wp::ExternalKind as ExtK;
        let mut section = we::ExportSection::new();
        for data in reader {
            let data = data.map_err(|e| Error::Parse("export section entry", e))?;
            match data.kind {
                ExtK::Func => section.export(data.name, EK::Func, self.trampoline_idx(data.index)),
                ExtK::Table => section.export(data.name, EK::Table, data.index),
                ExtK::Memory => section.export(data.name, EK::Memory, data.index),
                ExtK::Global => section.export(data.name, EK::Global, data.index),
                ExtK::Tag => section.export(data.name, EK::Tag, data.index),
            };
        }
        Ok(section)
    }

    fn transform_global_section(
        &self,
        reader: wp::GlobalSectionReader,
    ) -> Result<we::GlobalSection, Error> {
        let mut section = we::GlobalSection::new();
        for data in reader {
            let data = data.map_err(|e| Error::Parse("global section entry", e))?;
            section.global(
                we::GlobalType {
                    val_type: transform::transform_ty(data.ty.content_type),
                    mutable: data.ty.mutable,
                },
                &self.transform_init_expr(data.init_expr)?,
            );
        }
        Ok(section)
    }

    fn transform_function_section(
        &self,
        reader: wp::FunctionSectionReader,
    ) -> Result<we::FunctionSection, Error> {
        let mut section = we::FunctionSection::new();
        for (type_index, indirect) in reader.into_iter().zip(self.to_indirect.iter()) {
            let type_index = type_index.map_err(|e| Error::Parse("function section entry", e))?;
            if *indirect {
                // One for trampoline...
                section.function(type_index);
                // And another for the function definition itself.
                section.function(type_index);
            } else {
                section.function(type_index);
            }
        }
        Ok(section)
    }

    fn transform_element_section(
        &self,
        reader: wp::ElementSectionReader,
    ) -> Result<we::ElementSection, Error> {
        let mut section = we::ElementSection::new();
        for elem in reader {
            let elem = elem.map_err(|e| Error::Parse("element section entry", e))?;
            let active_offset_instr;
            let mode = match elem.kind {
                wp::ElementKind::Passive => we::ElementMode::Passive,
                wp::ElementKind::Active {
                    table_index,
                    init_expr,
                } => {
                    active_offset_instr = self.transform_init_expr(init_expr)?;
                    we::ElementMode::Active {
                        table: Some(table_index),
                        offset: &active_offset_instr,
                    }
                }
                wp::ElementKind::Declared => we::ElementMode::Declared,
            };
            let mut functions = Vec::new();
            let mut exprs = Vec::new();
            let items = elem
                .items
                .get_items_reader()
                .map_err(|e| Error::Parse("items", e))?;
            let uses_exprs = items.uses_exprs();
            for item in items {
                let item = item.map_err(|e| Error::Parse("item", e))?;
                match item {
                    wp::ElementItem::Func(f) => functions.push(self.trampoline_idx(f)),
                    wp::ElementItem::Expr(e) => {
                        let op = self.parse_init_expr_op(e)?;
                        let expr = match op {
                            wp::Operator::RefFunc { function_index } => {
                                we::Element::Func(self.trampoline_idx(function_index))
                            }
                            wp::Operator::RefNull { ty: _ } => we::Element::Null,
                            // TODO: wasm-encoder does not allow to specify this anyway?
                            wp::Operator::GlobalGet { .. } => todo!(),
                            _ => return Err(Error::ParseItemExpression),
                        };
                        exprs.push(expr);
                    }
                }
            }
            let elements = if uses_exprs {
                we::Elements::Expressions(&exprs)
            } else {
                we::Elements::Functions(&functions)
            };

            section.segment(we::ElementSegment {
                mode,
                element_type: transform::transform_ty(elem.ty),
                elements,
            });
        }
        Ok(section)
    }

    fn transform_function_body(&self, reader: wp::FunctionBody) -> Result<we::Function, Error> {
        let locals = reader
            .get_locals_reader()
            .map_err(|e| Error::Parse("function locals", e))?;
        let locals = locals
            .into_iter()
            .map(|v| v.map(|(idx, ty)| (idx, transform::transform_ty(ty))))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::Parse("function local", e))?;
        let mut function = we::Function::new(locals);
        let ops = reader
            .get_operators_reader()
            .map_err(|e| Error::Parse("function operators", e))?;
        for op in ops {
            let op = op.map_err(|e| Error::Parse("function operator", e))?;
            function.instruction(&self.transform_op(op)?);
        }
        Ok(function)
    }

    fn generate_trampoline(&self, original_idx: u32) -> Result<we::Function, Error> {
        let original_idx_usize = usize::try_from(original_idx).expect("TODO");
        let ty_index = *self.function_types.get(original_idx_usize).expect("TODO");
        let ty_index = usize::try_from(ty_index).expect("TODO");
        let wp::Type::Func(ty) = self.types.get(ty_index).expect("TODO");
        let mut function = we::Function::new([]);
        // Load all params onto the operand stack.
        for (_, idx) in ty.params[..].iter().zip(0..) {
            function.instruction(&we::Instruction::LocalGet(idx));
        }
        function.instruction(&we::Instruction::Call(self.function_idx(original_idx)));
        function.instruction(&we::Instruction::End);
        Ok(function)
    }

    fn transform_custom_section(
        &self,
        output: &mut Vec<u8>,
        reader: wp::CustomSectionReader<'a>,
    ) -> Result<(), Error> {
        let position = reader.data_offset();
        let name = reader.name();
        let data = match name {
            "name" => {
                let new_section = we::CustomSection {
                    name: name,
                    data: &self.transform_name_section(reader.data(), position)?,
                };
                output.push(new_section.id());
                new_section.encode(output);
            }
            _ => {
                let new_section = we::CustomSection {
                    name,
                    data: reader.data(),
                };
                output.push(new_section.id());
                new_section.encode(output);
            }
        };
        Ok(())
    }

    fn transform_name_section<'b>(
        &self,
        data: &'b [u8],
        position: usize,
    ) -> Result<Vec<u8>, Error> {
        let reader = wp::NameSectionReader::new(data, position)
            .map_err(|e| Error::Parse("name section", e))?;
        let mut output = we::NameSection::new();
        for name in reader {
            let name = name.map_err(|e| Error::Parse("name section entry", e))?;
            match name {
                wp::Name::Module(name) => output.module(
                    name.get_name()
                        .map_err(|e| Error::Parse("name section entry for module", e))?,
                ),
                wp::Name::Function(name_map) => self
                    .transform_function_name_map(&mut output, name_map)
                    .map_err(|e| Error::Parse("name section entry for functions", e))?,
                wp::Name::Local(name_map) => output.locals(
                    &self
                        .transform_indirect_name_map(name_map)
                        .map_err(|e| Error::Parse("name section entry for function locals", e))?,
                ),
                wp::Name::Label(name_map) => output.labels(
                    &self
                        .transform_indirect_name_map(name_map)
                        .map_err(|e| Error::Parse("name section entry for function labels", e))?,
                ),
                wp::Name::Type(name_map) => {
                    let map = name_map
                        .get_map()
                        .map_err(|e| Error::Parse("name section entry for types", e))?;
                    let types = transform::transform_name_map(map)
                        .map_err(|e| Error::Parse("name map entry for types", e))?;
                    output.types(&types);
                }
                wp::Name::Table(name_map) => {
                    let map = name_map
                        .get_map()
                        .map_err(|e| Error::Parse("name section entry for tables", e))?;
                    let tables = transform::transform_name_map(map)
                        .map_err(|e| Error::Parse("name map entry for tables", e))?;
                    output.tables(&tables);
                }
                wp::Name::Memory(name_map) => {
                    let map = name_map
                        .get_map()
                        .map_err(|e| Error::Parse("name section entry for memories", e))?;
                    let memories = transform::transform_name_map(map)
                        .map_err(|e| Error::Parse("name map entry for memories", e))?;
                    output.memories(&memories);
                }
                wp::Name::Global(name_map) => {
                    let map = name_map
                        .get_map()
                        .map_err(|e| Error::Parse("name section entry for globals", e))?;
                    let globals = transform::transform_name_map(map)
                        .map_err(|e| Error::Parse("name map entry for globals", e))?;
                    output.globals(&globals);
                }
                wp::Name::Element(name_map) => {
                    let map = name_map
                        .get_map()
                        .map_err(|e| Error::Parse("name section entry for elements", e))?;
                    let elements = transform::transform_name_map(map)
                        .map_err(|e| Error::Parse("name map entry for elements", e))?;
                    output.elements(&elements);
                }
                wp::Name::Data(name_map) => {
                    let map = name_map
                        .get_map()
                        .map_err(|e| Error::Parse("name section entry for elements", e))?;
                    let data = transform::transform_name_map(map)
                        .map_err(|e| Error::Parse("name map entry for elements", e))?;
                    output.data(&data);
                }
                wp::Name::Unknown { .. } => {}
            }
        }
        let mut out_vec = Vec::new();
        output.encode(&mut out_vec);
        Ok(out_vec)
    }

    fn transform_indirect_name_map(
        &self,
        map: wp::IndirectNameMap,
    ) -> Result<we::IndirectNameMap, BinaryReaderError> {
        let mut name_map = we::IndirectNameMap::new();
        let mut reader = map.get_indirect_map()?;
        for _ in 0..reader.get_indirect_count() {
            let name = reader.read()?;
            let function_idx = self.function_idx(name.indirect_index);
            let name_submap = transform::transform_name_map(name.get_map()?)?;
            name_map.append(function_idx, &name_submap);
        }
        Ok(name_map)
    }

    fn transform_function_name_map(
        &self,
        output: &mut we::NameSection,
        map: wp::NameMap,
    ) -> Result<(), BinaryReaderError> {
        let mut name_map = we::NameMap::new();
        let mut reader = map.get_map()?;
        for _ in 0..reader.get_count() {
            let name = reader.read()?;
            if self.should_indirect(name.index) {
                name_map.append(
                    self.trampoline_idx(name.index),
                    &format!("trampoline::{}", name.name),
                );
                name_map.append(self.function_idx(name.index), name.name);
            } else {
                name_map.append(self.function_idx(name.index), name.name);
            }
        }
        output.functions(&name_map);
        Ok(())
    }
}
