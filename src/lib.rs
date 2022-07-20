pub mod indirection;
mod transform;

// Custom test harness
#[cfg(test)]
mod tests;
#[cfg(test)]
fn main() {
    tests::main();
}
