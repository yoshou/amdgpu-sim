pub fn get_bits(value: u64, to: usize, from: usize) -> u64 {
    let num = to + 1 - from;
    (value >> from) & ((1u64 << num) - 1)
}
