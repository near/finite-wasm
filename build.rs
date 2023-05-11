fn build_interpreter() {
    println!("cargo:rerun-if-changed=interpreter");
    std::process::Command::new("opam")
        .args(["exec", "--", "make", "-Cinterpreter", "wasm"])
        .output()
        .expect("failed compiling wasm interpreter");
}

fn main() {
    #[cfg(feature = "wast-tests")]
    build_interpreter();
}
