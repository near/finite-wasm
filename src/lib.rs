pub mod instrument;
mod partial_sum;
pub use partial_sum::PartialSumMap;

// Custom test harness
#[cfg(test)]
mod tests;
#[cfg(test)]
fn main() {
    tests::main();
}
