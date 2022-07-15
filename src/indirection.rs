use wasmparser::{ElementItem, ExternalKind, Operator};

#[derive(Debug)]
pub enum Error {
    IncompleteInput,
    ParseModule(wasmparser::BinaryReaderError),
    ParseExport(wasmparser::BinaryReaderError),
    ParseElement(wasmparser::BinaryReaderError),
    ParseItems(wasmparser::BinaryReaderError),
    ParseItem(wasmparser::BinaryReaderError),
    ParseItemInitExprOp(wasmparser::BinaryReaderError),
    InvalidConstOperator,
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IncompleteInput => None,
            Error::ParseModule(e) => Some(e),
            Error::ParseExport(e) => Some(e),
            Error::ParseElement(e) => Some(e),
            Error::ParseItems(e) => Some(e),
            Error::ParseItem(e) => Some(e),
            Error::ParseItemInitExprOp(e) => Some(e),
            Error::InvalidConstOperator => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Error::IncompleteInput => "input is incomplete",
            Error::ParseModule(_) => "could not parse the module",
            Error::ParseExport(_) => "could not parse the export",
            Error::ParseElement(_) => "could not parse the element",
            Error::ParseItems(_) => "could not parse the items",
            Error::ParseItem(_) => "could not parse the item",
            Error::ParseItemInitExprOp(_) => "could not parse an item initialization expression",
            Error::InvalidConstOperator => "item initialization expression is not a constant op",
        })
    }
}

pub fn indirect(wasm: &[u8]) -> Result<Vec<u8>, Error> {
    let mut parser = wasmparser::Parser::new(0);
    let mut functions_to_indirect = Vec::<u32>::new();

    for payload in parser.parse_all(wasm) {
        // The only payloads we’re interested in for figuring out which functions need indirection
        // are the `ExportSection` and `ElementSection`. Once we figure out the functions we want
        // to indirect, we will want to also append our new trampolines to the `CodeSection`.
        match payload.map_err(Error::ParseModule)? {
            wasmparser::Payload::StartSection { func, range: _ } => {
                functions_to_indirect.push(func)
            }
            wasmparser::Payload::ExportSection(mut exports) => {
                for _ in 0..exports.get_count() {
                    let export = exports.read().map_err(Error::ParseExport)?;
                    if let ExternalKind::Func = export.kind {
                        functions_to_indirect.push(export.index);
                    }
                }
            }
            wasmparser::Payload::ElementSection(mut elements) => {
                for _ in 0..elements.get_count() {
                    let element = elements.read().map_err(Error::ParseElement)?;
                    let mut items = element
                        .items
                        .get_items_reader()
                        .map_err(Error::ParseItems)?;
                    for _ in 0..items.get_count() {
                        let item = items.read().map_err(Error::ParseItem)?;
                        match dbg!(item) {
                            ElementItem::Func(func) => functions_to_indirect.push(func),
                            // global.get can only reference imported functions, so we don’t need
                            // to handle _those_. Probably?
                            ElementItem::Expr(init) => {
                                let mut ops = init.get_operators_reader();
                                loop {
                                    match ops.read().map_err(Error::ParseItemInitExprOp)? {
                                        Operator::End => break,
                                        Operator::RefFunc { function_index } => {
                                            functions_to_indirect.push(function_index)
                                        }
                                        Operator::RefNull { .. }
                                        | Operator::I32Const { .. }
                                        | Operator::I64Const { .. }
                                        | Operator::F32Const { .. }
                                        | Operator::F64Const { .. }
                                        | Operator::V128Const { .. } => {}
                                        // TODO: how should we handle this? The only kind of global
                                        // that can be referred to here is imported.
                                        Operator::GlobalGet { .. } => todo!(),
                                        // TODO: index and description.
                                        _ => return Err(Error::InvalidConstOperator),
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Q: Should we strive to keep the trampolines close to the functions themselves? Probably
    // yes for code locality?
    //
    // An alternative would be to ensure that we interposit the trampolines at the indices of the
    // original functions and to move the originals to the end of the code section. In that case
    // none of the references need to be updated (potentially nice for handling of global.get??)
    Ok(vec![])
}
