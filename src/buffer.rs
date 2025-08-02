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
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&buffer[offset..offset + 4]);
    u32::from_le_bytes(arr)
}

pub fn get_u64(buffer: &[u8], offset: usize) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&buffer[offset..offset + 8]);
    u64::from_le_bytes(arr)
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

pub fn get_bits_u64(buffer: &[u64], bit_offset: usize, bit_size: usize) -> u64 {
    assert!(bit_size > 0 && bit_size <= 64);
    assert!(bit_offset + bit_size <= buffer.len() * 64);

    let byte_offset = bit_offset / 64;
    let bit_offset = bit_offset % 64;

    if bit_offset + bit_size <= 64 {
        let val = buffer[byte_offset];
        (val >> bit_offset) & ((1u64 << bit_size) - 1)
    } else {
        let lo_val = buffer[byte_offset];
        let hi_val = buffer[byte_offset + 1];
        let lo_bits = 64 - bit_offset;
        let hi_bits = bit_size - lo_bits;
        ((lo_val >> bit_offset) & ((1u64 << lo_bits) - 1))
            | ((hi_val & ((1u64 << hi_bits) - 1)) << lo_bits)
    }
}
