#[cfg(feature = "wast-tests")]
fn build_interpreter() {
    println!("cargo:rerun-if-changed=interpreter");
    std::process::Command::new("make")
        .args(["-Cinterpreter", "wasm"])
        .output()
        .expect("failed compiling wasm interpreter");
}

fn main() {
    #[cfg(feature = "wast-tests")]
    build_interpreter();
}
