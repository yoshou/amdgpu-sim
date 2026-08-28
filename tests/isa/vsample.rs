//! VSAMPLE, the image format that takes a sampler: IMAGE_SAMPLE_LZ, which
//! fetches a texel of the base level without computing one.
//!
//! The image is the harness's data buffer, read as a linear two-dimensional
//! image. Texel (x, y) holds x + 64y, so a fetched value says which texel it
//! came from, and the image is 64 texels wide: narrow enough that the 128
//! bytes a row of a linear image takes at least is more than the row needs,
//! which is what makes the padding between the rows part of what is tested.
//!
//! Everything the fetch depends on is stated per test: the resource says where
//! the image is, how it is laid out and what its texels mean, and the sampler
//! says what happens to a coordinate before it names a texel.

use crate::compare::*;
use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// The image the tests sample, in texels.
const WIDTH: u32 = 64;
const HEIGHT: u32 = 4;

/// Data formats, from Table 62.
const FMT_8_UNORM: u32 = 1;
const FMT_8_SNORM: u32 = 2;
const FMT_8_UINT: u32 = 5;
const FMT_8_SINT: u32 = 6;

/// Destination selects, from Table 60: a constant or one of the channels.
const SEL_0: u32 = 0;
const SEL_1: u32 = 1;
const SEL_X: u32 = 4;
const SEL_Y: u32 = 5;
const SEL_Z: u32 = 6;
const SEL_W: u32 = 7;

/// The texels, laid out as the part reads them: `width` of them at the start
/// of every `row` bytes.
fn texture(width: u32, row: u32) -> Vec<u32> {
    let mut bytes = vec![0u8; (row * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..width {
            bytes[(y * row + x) as usize] = (x + width * y) as u8;
        }
    }
    bytes
        .chunks(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn set_bits(words: &mut [u32], position: usize, size: usize, value: u32) {
    for i in 0..size {
        if (value >> i) & 1 == 1 {
            words[(position + i) / 32] |= 1 << ((position + i) % 32);
        }
    }
}

/// The image resource, as Table 60 lays it out. The address of the image is
/// missing: the harness folds the address of its data buffer into the first two
/// words, which is the only way a test can name it.
fn image_resource(format: u32, dst_sel: [u32; 4]) -> [u32; 8] {
    image_resource_sized(format, dst_sel, WIDTH, 0)
}

/// The same with a width and a row pitch of the case's own. The pitch field
/// holds one less than the pitch and is clear where the width is the pitch.
fn image_resource_sized(format: u32, dst_sel: [u32; 4], width: u32, pitch: u32) -> [u32; 8] {
    let mut rsrc = [0u32; 8];
    set_bits(&mut rsrc, 49, 8, format);
    set_bits(&mut rsrc, 62, 16, width - 1);
    set_bits(&mut rsrc, 78, 16, HEIGHT - 1);
    if pitch != 0 {
        set_bits(&mut rsrc, 128, 16, pitch - 1);
    }
    for (i, select) in dst_sel.iter().enumerate() {
        set_bits(&mut rsrc, 96 + 3 * i, 3, *select);
    }
    set_bits(&mut rsrc, 124, 4, 9); // a two-dimensional image
    rsrc
}

/// The sampler, as Table 61 lays it out: what happens to a coordinate that
/// falls outside the image, whether the coordinates span it or count texels,
/// and which colour stands in for a texel off its edge.
fn sampler(clamp: [u32; 2], unnormalized: bool, border: u32) -> [u32; 4] {
    let mut samp = [0u32; 4];
    set_bits(&mut samp, 0, 3, clamp[0]);
    set_bits(&mut samp, 3, 3, clamp[1]);
    set_bits(&mut samp, 15, 1, unnormalized as u32);
    set_bits(&mut samp, 126, 2, border);
    samp
}

/// One coordinate sampled against one resource and sampler.
pub(crate) struct VsampleCase {
    u: f32,
    v: f32,
    /// v8..v11 after the instruction. Only the components the DMASK asks for
    /// are written; the rest stay clear.
    expected: [u32; 4],
}

/// Bit-exact comparison of one fetch against captured hardware.
fn check_vsample(dmask: u32, unrm: u32, rsrc: [u32; 8], samp: [u32; 4], cases: &[VsampleCase]) {
    check_vsample_texture(dmask, unrm, rsrc, samp, texture(WIDTH, 128), cases);
}

/// As `check_vsample`, but against texels the case lays out itself.
fn check_vsample_texture(
    dmask: u32,
    unrm: u32,
    rsrc: [u32; 8],
    samp: [u32; 4],
    data: Vec<u32>,
    cases: &[VsampleCase],
) {
    let harness = Harness::vsample();
    let words = vsample(31, 1, dmask, unrm, 8, 12, 20, [0, 1, 0, 0]);
    let uni: Vec<u32> = rsrc.iter().chain(samp.iter()).copied().collect();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        for lane in 0..LANES {
            src[lane * harness.src_stride] = case.u.to_bits();
            src[lane * harness.src_stride + 1] = case.v.to_bits();
        }
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run_with_data(engine, &words, &src, &uni, &data);
            let got: Vec<u32> = out[..4].to_vec();
            if got == case.expected {
                continue;
            }
            let show = |values: &[u32]| {
                values
                    .iter()
                    .map(|&v| show_f32(v))
                    .collect::<Vec<String>>()
                    .join(", ")
            };
            failures.push(format!(
                "  {:<11} case {} ({}, {}) hardware=({}) simulator=({})",
                engine_name(engine),
                i,
                case.u,
                case.v,
                show(&case.expected),
                show(&got),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(),
        cases.len() * 2,
        failures.join("\n"),
    );
}

/// The channels in their own order, which is what an image of one channel
/// gives to every one of them.
const RGBA: [u32; 4] = [SEL_X, SEL_Y, SEL_Z, SEL_W];

#[test]
fn image_sample_lz_texel() {
    // A normalized coordinate spans the image: the texel it names is the one
    // it reaches after being multiplied by the image's size and rounded down.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.0078125, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0234375, v: 0.125, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.375, expected: [0x0000_0040, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.625, expected: [0x0000_0080, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.875, expected: [0x0000_00C0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.5, v: 0.5, expected: [0x0000_00A0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.25, v: 0.75, expected: [0x0000_00D0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.125, v: 0.375, expected: [0x0000_0048, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.015625, v: 0.125, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_wrap() {
    // The wrapping mode, which is what a sampler of all zeros asks for: a
    // coordinate outside the image comes back round the other side.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: -0.015625, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.0, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.015625, v: 0.125, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.5, v: 0.125, expected: [0x0000_0020, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.5, v: 0.125, expected: [0x0000_0020, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.5, v: 1.25, expected: [0x0000_0060, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.5, v: -0.25, expected: [0x0000_00E0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 2.5, v: 2.5, expected: [0x0000_00A0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_mirror() {
    // Mirroring: the image is repeated, every other copy of it reversed.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([1, 1], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_000F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.25, v: 0.125, expected: [0x0000_002F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.75, v: 0.125, expected: [0x0000_000F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.75, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 2.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.25, v: -0.25, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_clamp_last_texel() {
    // Clamping: a coordinate outside the image names its last texel.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([2, 2], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.25, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -2.5, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 3.5, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.25, v: 2.5, expected: [0x0000_00D0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_mirror_once_last_texel() {
    // The image is mirrored once below zero, and clamped everywhere else.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([3, 3], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_000F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.5, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.25, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_clamp_half_border() {
    // Clamping to the half-way point of the border texel, which for a fetch
    // that takes a single texel is the same as clamping to the last one.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([4, 4], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.25, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_mirror_once_half_border() {
    // The half-border form of mirroring once, likewise.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([5, 5], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_000F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.5, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_clamp_border() {
    // A coordinate outside the image takes the border colour instead of a
    // texel, which for a sampler of all zeros is transparent black.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([6, 6], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.25, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.25, v: 1.25, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.25, v: -0.25, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_mirror_once_border() {
    // Mirrored once below zero, and the border colour beyond that.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([7, 7], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_000F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.5, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.25, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_white_border() {
    // An opaque-white border, which an integer format counts as one.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([6, 6], false, 2),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_white_border_unorm() {
    // The same border against a format that counts one as 1.0.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UNORM, RGBA),
        sampler([6, 6], false, 2),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x3D80_8081, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -0.25, v: 0.125, expected: [0x3F80_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_opaque_black_border() {
    // An opaque-black border, whose one channel is black like the
    // transparent one's.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([6, 6], false, 1),
        &[
            VsampleCase { u: -0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_8_unorm() {
    // 8_UNORM: the texel is a fraction of 255.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UNORM, RGBA),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.0078125, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0234375, v: 0.125, expected: [0x3B80_8081, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.125, expected: [0x3E7C_FCFD, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.875, expected: [0x3F40_C0C1, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.875, expected: [0x3F80_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_8_snorm() {
    // 8_SNORM: the texel is a fraction of 127, and -128 is the same as -127.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_SNORM, RGBA),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.0078125, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0234375, v: 0.125, expected: [0x3C01_0204, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.125, expected: [0x3EFD_FBF8, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.875, expected: [0xBF01_0204, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.875, expected: [0xBC01_0204, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.625, expected: [0xBF80_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_8_sint() {
    // 8_SINT: the texel is a signed integer, widened to the register.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_SINT, RGBA),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.0078125, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.875, expected: [0xFFFF_FFC0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.875, expected: [0xFFFF_FFFF, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.625, expected: [0xFFFF_FF80, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_unnormalized_sampler() {
    // Unnormalized coordinates count texels rather than spanning the image.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([0, 0], true, 0),
        &[
            VsampleCase { u: 0.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.0, v: 0.0, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.5, v: 0.5, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 63.0, v: 3.0, expected: [0x0000_00FF, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 64.0, v: 0.0, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_unnormalized_instruction() {
    // The instruction's UNRM asks for the same thing as the sampler's bit.
    check_vsample(
        0x1,
        1,
        image_resource(FMT_8_UINT, RGBA),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 1.5, v: 0.5, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 63.0, v: 3.0, expected: [0x0000_00FF, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_unnormalized_wrap() {
    // Unnormalized coordinates cannot wrap: the wrapping mode clamps.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([0, 0], true, 0),
        &[
            VsampleCase { u: -2.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 63.0, v: 0.0, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 70.0, v: 0.0, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 4.0, expected: [0x0000_00C0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_unnormalized_mirror() {
    // Neither can they mirror repeatedly: the mirroring mode mirrors once.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([1, 1], true, 0),
        &[
            VsampleCase { u: -2.0, v: 0.0, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 63.0, v: 0.0, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 70.0, v: 0.0, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 4.0, expected: [0x0000_00C0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_unnormalized_border() {
    // The border modes are the ones unnormalized coordinates keep.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([6, 6], true, 0),
        &[
            VsampleCase { u: -2.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 63.0, v: 0.0, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 70.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 4.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_unnormalized_mirror_once_border() {
    // Mirrored once below zero, and the border colour beyond that.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, RGBA),
        sampler([7, 7], true, 0),
        &[
            VsampleCase { u: -2.0, v: 0.0, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: -1.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 63.0, v: 0.0, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 70.0, v: 0.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0, v: 4.0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_select() {
    // The destination selects say what each component holds: a constant, or
    // a channel of the image -- and an image of one channel gives that
    // channel to whichever one is named.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UINT, [SEL_0, SEL_1, SEL_X, SEL_Y]),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_select_one() {
    // What the constant one means depends on the format.
    check_vsample(
        0x1,
        0,
        image_resource(FMT_8_UNORM, [SEL_1, SEL_0, SEL_X, SEL_Y]),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x3F80_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_dmask_xy() {
    // The DMASK names the components the fetch returns, which go to
    // consecutive registers.
    check_vsample(
        0x3,
        0,
        image_resource(FMT_8_UINT, [SEL_0, SEL_1, SEL_X, SEL_Y]),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0001, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_dmask_xz() {
    // The components need not be consecutive; the registers are.
    check_vsample(
        0x5,
        0,
        image_resource(FMT_8_UINT, [SEL_0, SEL_1, SEL_X, SEL_Y]),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0010, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_dmask_xyzw() {
    // All four components.
    check_vsample(
        0xF,
        0,
        image_resource(FMT_8_UINT, [SEL_0, SEL_1, SEL_X, SEL_W]),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0000, 0x0000_0001, 0x0000_0010, 0x0000_0010] },
        ],
    );
}

#[test]
fn image_sample_lz_dmask_w() {
    // One component that is not the first: it still goes to the first
    // register.
    check_vsample(
        0x8,
        0,
        image_resource(FMT_8_UINT, [SEL_0, SEL_1, SEL_X, SEL_Y]),
        sampler([0, 0], false, 0),
        &[
            VsampleCase { u: 0.25, v: 0.125, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_row_pitch() {
    // A row of a linear image takes a multiple of 128 bytes, so an image whose
    // width is not one has padding between its rows: 200 texels a row live in
    // 256 bytes.
    check_vsample_texture(
        0x1,
        0,
        image_resource_sized(FMT_8_UINT, RGBA, 200, 0),
        sampler([0, 0], false, 0),
        texture(200, 256),
        &[
            VsampleCase { u: 0.0025, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0075, v: 0.125, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9975, v: 0.125, expected: [0x0000_00C7, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0025, v: 0.375, expected: [0x0000_00C8, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.5025, v: 0.625, expected: [0x0000_00F4, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9975, v: 0.875, expected: [0x0000_001F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_row_pitch_from_the_resource() {
    // The resource can give a pitch wider than the width, and it is rounded up
    // the same way: 200 becomes 256, so the rows of a 64-texel image sit 256
    // bytes apart.
    check_vsample_texture(
        0x1,
        0,
        image_resource_sized(FMT_8_UINT, RGBA, WIDTH, 200),
        sampler([0, 0], false, 0),
        texture(WIDTH, 256),
        &[
            VsampleCase { u: 0.0078125, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0234375, v: 0.125, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.375, expected: [0x0000_0040, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.5078125, v: 0.625, expected: [0x0000_00A0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.875, expected: [0x0000_00FF, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn image_sample_lz_row_pitch_below_the_minimum() {
    // A pitch narrower than 128 bytes is rounded up to it, which for this
    // image is the same layout as no pitch at all.
    check_vsample_texture(
        0x1,
        0,
        image_resource_sized(FMT_8_UINT, RGBA, WIDTH, WIDTH),
        sampler([0, 0], false, 0),
        texture(WIDTH, 128),
        &[
            VsampleCase { u: 0.0078125, v: 0.125, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.125, expected: [0x0000_003F, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.0078125, v: 0.875, expected: [0x0000_00C0, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            VsampleCase { u: 0.9921875, v: 0.875, expected: [0x0000_00FF, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}
