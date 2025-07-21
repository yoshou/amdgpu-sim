pub fn set_u16(buffer: &mut [u8], offset: usize, value: u16) {
    buffer[offset] = (value & 0xFF) as u8;
    buffer[offset + 1] = ((value >> 8) & 0xFF) as u8;
}

pub fn set_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset] = (value & 0xFF) as u8;
    buffer[offset + 1] = ((value >> 8) & 0xFF) as u8;
    buffer[offset + 2] = ((value >> 16) & 0xFF) as u8;
    buffer[offset + 3] = ((value >> 24) & 0xFF) as u8;
}

pub fn set_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset] = (value & 0xFF) as u8;
    buffer[offset + 1] = ((value >> 8) & 0xFF) as u8;
    buffer[offset + 2] = ((value >> 16) & 0xFF) as u8;
    buffer[offset + 3] = ((value >> 24) & 0xFF) as u8;
    buffer[offset + 4] = ((value >> 32) & 0xFF) as u8;
    buffer[offset + 5] = ((value >> 40) & 0xFF) as u8;
    buffer[offset + 6] = ((value >> 48) & 0xFF) as u8;
    buffer[offset + 7] = ((value >> 56) & 0xFF) as u8;
}

pub fn get_u32(buffer: &[u8], offset: usize) -> u32 {
    let b0 = buffer[offset] as u32;
    let b1 = buffer[offset + 1] as u32;
    let b2 = buffer[offset + 2] as u32;
    let b3 = buffer[offset + 3] as u32;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

pub fn get_u64(buffer: &[u8], offset: usize) -> u64 {
    let b0 = buffer[offset] as u64;
    let b1 = buffer[offset + 1] as u64;
    let b2 = buffer[offset + 2] as u64;
    let b3 = buffer[offset + 3] as u64;
    let b4 = buffer[offset + 4] as u64;
    let b5 = buffer[offset + 5] as u64;
    let b6 = buffer[offset + 6] as u64;
    let b7 = buffer[offset + 7] as u64;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
}

pub fn get_str(buffer: &[u8], offset: usize, size: usize) -> String {
    let bytes: &[u8] = &buffer[offset..(offset + size)];
    bytes.iter().map(|&s| s as char).collect::<String>()
}

pub fn get_bytes(buffer: &[u8], offset: usize, size: usize) -> Vec<u8> {
    let bytes: &[u8] = &buffer[offset..(offset + size)];
    bytes.to_vec()
}

pub fn get_bit(buffer: &[u8], offset: usize, bit: usize) -> bool {
    ((buffer[offset + (bit >> 3)] >> (bit & 0x7)) & 1) == 1
}

pub fn get_bits(buffer: &[u8], offset: usize, bit: usize, size: usize) -> u8 {
    ((get_u32(buffer, offset + (bit >> 3)) >> (bit & 0x7)) & ((1 << size) - 1)) as u8
}
