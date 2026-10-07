use crate::probe::Probe;
use crate::vector::{lanes, word};

const OWN: &str = "v_mov_b32 v13, s43
                   v_add_u32 v12, vcc, s42, v41
                   v_addc_u32 v13, vcc, v13, 0, vcc";

fn bytes(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

#[test]
fn flat_loads_extend_and_flat_stores_merge() {
    let inputs = lanes(4, word);
    let body = format!(
        "{OWN}
         flat_store_dword v[12:13], v1
         s_waitcnt vmcnt(0)
         v_add_u32 v14, vcc, 1, v12
         v_addc_u32 v15, vcc, 0, v13, vcc
         flat_store_byte v[14:15], v2
         v_add_u32 v14, vcc, 2, v12
         v_addc_u32 v15, vcc, 0, v13, vcc
         flat_store_short v[14:15], v3
         s_waitcnt vmcnt(0)
         flat_load_dword v20, v[12:13]
         flat_load_ubyte v21, v[14:15]
         flat_load_sbyte v22, v[14:15]
         flat_load_ushort v23, v[14:15]
         flat_load_sshort v24, v[14:15]
         s_waitcnt vmcnt(0)"
    );
    Probe::new(4, 5, &body).check(&inputs, |_, x| {
        let mut word = bytes(x[0]);
        word[1] = bytes(x[1])[0];
        word[2..4].copy_from_slice(&bytes(x[2])[..2]);
        let merged = u32::from_le_bytes(word);
        let short = merged >> 16;
        vec![merged, short & 0xff, (short as u8 as i8) as i32 as u32, short, (short as u16 as i16) as i32 as u32]
    });
}

#[test]
fn flat_wide_accesses_move_several_words() {
    let inputs = lanes(4, word);
    let body = format!(
        "{OWN}
         flat_store_dwordx4 v[12:13], v[1:4]
         s_waitcnt vmcnt(0)
         flat_load_dwordx2 v[20:21], v[12:13]
         v_add_u32 v14, vcc, 4, v12
         v_addc_u32 v15, vcc, 0, v13, vcc
         flat_load_dwordx3 v[22:24], v[14:15]
         s_waitcnt vmcnt(0)"
    );
    Probe::new(4, 5, &body).check(&inputs, |_, x| vec![x[0], x[1], x[1], x[2], x[3]]);
}

#[test]
fn flat_atomics_return_the_old_value() {
    let inputs = lanes(3, word);
    let cases: [(&str, fn(u32, u32) -> u32); 5] = [
        ("flat_atomic_add", |m, d| m.wrapping_add(d)),
        ("flat_atomic_smin", |m, d| (m as i32).min(d as i32) as u32),
        ("flat_atomic_smax", |m, d| (m as i32).max(d as i32) as u32),
        ("flat_atomic_umin", |m, d| m.min(d)),
        ("flat_atomic_umax", |m, d| m.max(d)),
    ];
    for (op, f) in cases {
        let body = format!(
            "{OWN}
             flat_store_dword v[12:13], v1
             s_waitcnt vmcnt(0)
             {op} v20, v[12:13], v2 glc
             s_waitcnt vmcnt(0)
             flat_load_dword v21, v[12:13]
             s_waitcnt vmcnt(0)"
        );
        Probe::new(3, 2, &body).check(&inputs, |_, x| vec![x[0], f(x[0], x[1])]);
    }
    let body = format!(
        "{OWN}
         flat_store_dword v[12:13], v1
         s_waitcnt vmcnt(0)
         v_mov_b32 v4, v2
         v_mov_b32 v5, v1
         flat_atomic_cmpswap v20, v[12:13], v[4:5] glc
         s_waitcnt vmcnt(0)
         flat_load_dword v21, v[12:13]
         v_mov_b32 v5, v3
         flat_atomic_cmpswap v[12:13], v[4:5]
         s_waitcnt vmcnt(0)
         flat_load_dword v22, v[12:13]
         s_waitcnt vmcnt(0)"
    );
    Probe::new(3, 3, &body).check(&inputs, |_, x| {
        let after = x[1];
        vec![x[0], after, if after == x[2] { x[1] } else { after }]
    });
}

fn shared(body: &str) -> String {
    format!(
        "s_mov_b32 m0, -1
         v_lshlrev_b32 v10, 4, v0
         {body}"
    )
}

#[test]
fn local_data_share_reads_writes_and_pairs() {
    let inputs = lanes(4, word);
    let body = shared(
        "ds_write_b32 v10, v1
         ds_write_b8 v10, v2 offset:4
         ds_write_b16 v10, v3 offset:6
         ds_write2_b32 v10, v4, v1 offset0:2 offset1:3
         s_waitcnt lgkmcnt(0)
         ds_read_b32 v20, v10
         ds_read_u8 v21, v10 offset:4
         ds_read_i8 v22, v10 offset:4
         ds_read_u16 v23, v10 offset:6
         ds_read_i16 v24, v10 offset:6
         ds_read2_b32 v[25:26], v10 offset0:2 offset1:3
         s_waitcnt lgkmcnt(0)",
    );
    Probe::new(4, 7, &body).shared(1024).check(&inputs, |_, x| {
        vec![
            x[0],
            x[1] & 0xff,
            x[1] as u8 as i8 as i32 as u32,
            x[2] & 0xffff,
            x[2] as u16 as i16 as i32 as u32,
            x[3],
            x[0],
        ]
    });
}

#[test]
fn local_data_share_crosses_lanes_without_a_barrier_inside_a_wave() {
    let inputs = lanes(1, word);
    let body = shared(
        "ds_write_b32 v10, v1
         v_add_u32 v11, vcc, 16, v10
         v_and_b32 v11, 0x3ff, v11
         s_waitcnt lgkmcnt(0)
         ds_read_b32 v20, v11
         ds_read2st64_b32 v[21:22], v10 offset0:0 offset1:0
         s_waitcnt lgkmcnt(0)",
    );
    Probe::new(1, 3, &body).shared(1024).check(&inputs, |l, x| vec![word((l + 1) % 64, 0), x[0], x[0]]);
}

#[test]
fn local_data_share_atomics_and_wide_words() {
    let inputs = lanes(4, word);
    let body = shared(
        "ds_write_b128 v10, v[1:4]
         s_waitcnt lgkmcnt(0)
         ds_add_u32 v10, v2
         ds_add_rtn_u32 v20, v10, v3 offset:4
         s_waitcnt lgkmcnt(0)
         ds_read_b64 v[21:22], v10
         ds_read_b128 v[23:26], v10
         s_waitcnt lgkmcnt(0)",
    );
    Probe::new(4, 7, &body).shared(1024).check(&inputs, |_, x| {
        let first = x[0].wrapping_add(x[1]);
        let second = x[1].wrapping_add(x[2]);
        vec![x[1], first, second, first, second, x[2], x[3]]
    });
}

#[test]
fn private_buffers_address_each_lane_by_offset() {
    let inputs = lanes(3, |l, k| if k == 2 { (l as u32 % 3) * 4 } else { word(l, k) });
    let body = "s_add_u32 s0, s0, s7
                s_addc_u32 s1, s1, 0
                s_movk_i32 s16, 0x100
                buffer_store_dword v1, off, s[0:3], 0 offset:4
                buffer_store_dword v2, v3, s[0:3], 0 offen offset:8
                buffer_store_dword v1, off, s[0:3], s16 offset:24
                s_waitcnt vmcnt(0)
                buffer_load_dword v20, off, s[0:3], 0 offset:4
                buffer_load_dword v21, v3, s[0:3], 0 offen offset:8
                buffer_load_dword v22, off, s[0:3], 0 offset:28
                buffer_load_ubyte v23, v3, s[0:3], 0 offen offset:9
                s_waitcnt vmcnt(0)";
    Probe::new(3, 4, body).private(64, true).check(&inputs, |_, x| vec![x[0], x[1], x[0], (x[1] >> 8) & 0xff]);
}

#[test]
fn the_frontend_refuses_what_it_cannot_model_exactly() {
    let refused = |probe: Probe, reason: &str| {
        let message = probe.refusal();
        assert!(message.contains(reason), "{}", message);
    };
    refused(Probe::new(1, 1, "v_lshlrev_b32 v10, 2, v0\nds_write_b32 v10, v1").shared(256), "M0");
    refused(
        Probe::new(1, 1, "v_readfirstlane_b32 s12, v1\ns_mov_b32 m0, s12\nv_lshlrev_b32 v10, 2, v0\nds_write_b32 v10, v1").shared(256),
        "M0",
    );
    refused(Probe::new(1, 1, "buffer_store_dword v1, off, s[0:3], 4").private(16, false), "unaligned");
    refused(
        Probe::new(1, 1, "v_readfirstlane_b32 s12, v1\nbuffer_store_dword v1, off, s[0:3], s12").private(16, false),
        "unaligned",
    );
    refused(Probe::new(1, 1, "s_add_u32 s0, s0, 4\nbuffer_store_dword v1, off, s[0:3], 0").private(16, false), "private segment");
    refused(
        Probe::new(1, 1, "v_readfirstlane_b32 s12, v1\ns_setreg_b32 hwreg(HW_REG_MODE, 4, 2), s12\nv_add_f32 v20, v1, v1"),
        "denormal mode",
    );
    refused(Probe::new(1, 1, "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 1\nv_add_f32 v20, v1, v1"), "rounding");
    refused(Probe::new(1, 1, "v_mov_b32_dpp v20, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"), "DPP");
}
