use crate::probe::{Probe, LANES};
use crate::vector::{lanes, word};

fn mask(inputs: &[u32], width: usize, f: impl Fn(&[u32]) -> bool) -> u64 {
    (0..LANES).fold(0u64, |m, l| m | (f(&inputs[l * width..(l + 1) * width]) as u64) << l)
}

#[test]
fn compares_that_write_exec_also_write_their_destination() {
    let inputs = lanes(3, |l, k| if k == 2 { (l % 5 != 3) as u32 } else { word(l, k) });
    let taken = mask(&inputs, 3, |x| x[2] != 0 && x[0] > x[1]);
    for (form, destination) in [("e32", "vcc"), ("e64", "s[14:15]")] {
        let body = format!(
            "v_mov_b32 v20, 0
             s_mov_b64 s[20:21], exec
             v_cmp_ne_u32 s[22:23], 0, v3
             s_mov_b64 exec, s[22:23]
             v_cmpx_gt_u32_{form} {destination}, v1, v2
             v_mov_b32 v20, 1
             s_mov_b64 s[12:13], exec
             s_mov_b64 s[16:17], {destination}
             s_mov_b64 exec, s[20:21]
             v_mov_b32 v21, s12
             v_mov_b32 v22, s13
             v_mov_b32 v23, s16
             v_mov_b32 v24, s17"
        );
        Probe::new(3, 5, &body).check(&inputs, |l, _| {
            vec![(taken >> l & 1) as u32, taken as u32, (taken >> 32) as u32, taken as u32, (taken >> 32) as u32]
        });
    }
}

#[test]
fn readlane_and_writelane_reach_every_lane_of_a_wave_of_64() {
    let inputs = lanes(2, |l, k| if k == 0 { 0x100 + l as u32 * 3 } else { [0u32, 31, 32, 63, 7, 40][l % 6] });
    for lane in [0u32, 31, 32, 63] {
        Probe::new(2, 1, &format!("v_readlane_b32 s12, v1, {lane}\nv_mov_b32 v20, s12")).check(&inputs, |_, _| vec![0x100 + lane * 3]);
        Probe::new(2, 1, &format!("v_mov_b32 v20, v1\ns_movk_i32 s12, 0x77\nv_writelane_b32 v20, s12, {lane}")).check(&inputs, |l, x| {
            vec![if l as u32 == lane { 0x77 } else { x[0] }]
        });
    }
    Probe::new(2, 1, "v_readfirstlane_b32 s13, v2\nv_readlane_b32 s12, v1, s13\nv_mov_b32 v20, s12").check(&inputs, |_, _| vec![0x100]);
    let skewed = lanes(2, |l, k| if k == 0 { 0x100 + l as u32 * 3 } else { (l >= 37) as u32 });
    Probe::new(
        2,
        1,
        "v_mov_b32 v20, 0
         s_mov_b64 s[20:21], exec
         v_cmp_ne_u32 s[22:23], 0, v2
         s_mov_b64 exec, s[22:23]
         v_readfirstlane_b32 s12, v1
         s_mov_b64 exec, s[20:21]
         v_mov_b32 v20, s12",
    )
    .check(&skewed, |_, _| vec![0x100 + 37 * 3]);
}

#[test]
fn backward_permutes_index_the_whole_wave() {
    let inputs = lanes(2, |l, k| if k == 0 { (((l * 37 + 11) % 64) * 4) as u32 } else { 0x5000 + l as u32 });
    Probe::new(2, 1, "ds_bpermute_b32 v20, v1, v2").check(&inputs, |_, x| vec![0x5000 + x[0] / 4]);
    Probe::new(2, 1, "ds_bpermute_b32 v20, v1, v2 offset:8").check(&inputs, |_, x| vec![0x5000 + (x[0] / 4 + 2) % 64]);
}
