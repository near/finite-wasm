pub mod instrument;
mod partial_sum;

// Custom test harness
#[cfg(test)]
mod tests;
#[cfg(test)]
fn main() {
    tests::main();
}
