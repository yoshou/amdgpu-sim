use crate::probe::Probe;
use crate::vector::lanes;

const SEL_0: u32 = 0;
const SEL_1: u32 = 1;
const SEL_R: u32 = 4;
const SEL_G: u32 = 5;

const UNORM: u32 = 0;
const SNORM: u32 = 1;
const UINT: u32 = 4;
const SINT: u32 = 5;

#[derive(Clone, Copy)]
struct Image {
    width: u32,
    height: u32,
    pitch: u32,
    data_format: u32,
    num_format: u32,
    select: [u32; 4],
    tiling: u32,
    kind: u32,
    levels: [u32; 2],
}

impl Image {
    fn new(num_format: u32, select: [u32; 4]) -> Self {
        Image {
            width: 16,
            height: 8,
            pitch: 20,
            data_format: 1,
            num_format,
            select,
            tiling: 8,
            kind: 9,
            levels: [0, 0],
        }
    }

    fn words(&self) -> [u32; 5] {
        let select = self.select.iter().enumerate().fold(0, |w, (k, &s)| w | s << (3 * k));
        [
            self.data_format << 20 | self.num_format << 26,
            (self.width - 1) | (self.height - 1) << 14,
            select | self.levels[0] << 12 | self.levels[1] << 16 | self.tiling << 20 | self.kind << 28,
            (self.pitch - 1) << 13,
            0,
        ]
    }

    fn texels(&self) -> Vec<u8> {
        let mut bytes = vec![0xee; (self.pitch * self.height) as usize];
        for y in 0..self.height {
            for x in 0..self.width {
                bytes[(y * self.pitch + x) as usize] = (x * 7 + y * 31 + 0x81) as u8;
            }
        }
        bytes
    }

    fn one(&self) -> u32 {
        match self.num_format {
            UNORM | SNORM => 1f32.to_bits(),
            _ => 1,
        }
    }

    fn convert(&self, raw: u8) -> u32 {
        match self.num_format {
            UNORM => (raw as f32 / 255.0).to_bits(),
            SNORM => ((raw as i8).max(-127) as f32 / 127.0).to_bits(),
            UINT => raw as u32,
            _ => (raw as i8) as i32 as u32,
        }
    }
}

#[derive(Clone, Copy)]
struct Sampler {
    clamp: [u32; 2],
    unnormalized: bool,
    border: u32,
    filters: u32,
}

impl Sampler {
    fn new(clamp: [u32; 2], unnormalized: bool, border: u32) -> Self {
        Sampler {
            clamp,
            unnormalized,
            border,
            filters: 0,
        }
    }

    fn words(&self) -> [u32; 4] {
        [
            self.clamp[0] | self.clamp[1] << 3 | (self.unnormalized as u32) << 15,
            0,
            self.filters,
            self.border << 30,
        ]
    }
}

fn clamp_texel(coord: i32, size: i32, mode: u32) -> Option<i32> {
    let mirror_once = |c: i32| if c < 0 { -1 - c } else { c };
    match mode {
        0 => Some(coord.rem_euclid(size)),
        1 => {
            let folded = coord.rem_euclid(2 * size);
            Some(if folded < size { folded } else { 2 * size - 1 - folded })
        }
        2 | 4 => Some(coord.clamp(0, size - 1)),
        3 | 5 => Some(mirror_once(coord).clamp(0, size - 1)),
        6 => (0..size).contains(&coord).then_some(coord),
        _ => {
            let mirrored = mirror_once(coord);
            (0..size).contains(&mirrored).then_some(mirrored)
        }
    }
}

fn sample(image: &Image, sampler: &Sampler, unorm: bool, u: f32, v: f32, select: u32) -> u32 {
    match select {
        SEL_0 => return 0,
        SEL_1 => return image.one(),
        _ => {}
    }
    let unnormalized = unorm || sampler.unnormalized;
    let texel = |c: f32, size: u32| if unnormalized { c } else { c * size as f32 }.floor() as i32;
    let mode = |axis: usize| match sampler.clamp[axis] {
        0 if unnormalized => 2,
        1 if unnormalized => 3,
        m => m,
    };
    let x = clamp_texel(texel(u, image.width), image.width as i32, mode(0));
    let y = clamp_texel(texel(v, image.height), image.height as i32, mode(1));
    match (x, y) {
        (Some(x), Some(y)) => image.convert(image.texels()[(y as u32 * image.pitch + x as u32) as usize]),
        _ if sampler.border == 2 => image.one(),
        _ => 0,
    }
}

fn body(image: &Image, sampler: &Sampler, dmask: u32, unorm: bool) -> String {
    let [w1, w2, w3, w4, w5] = image.words();
    let [s0, s1, s2, s3] = sampler.words();
    let count = dmask.count_ones();
    let data = if count == 1 { "v20".to_string() } else { format!("v[20:{}]", 19 + count) };
    format!(
        "v_mov_b32 v20, 0
         v_mov_b32 v21, 0
         v_mov_b32 v22, 0
         v_mov_b32 v23, 0
         s_lshr_b64 s[8:9], s[46:47], 8
         s_or_b32 s9, s9, {w1:#x}
         s_mov_b32 s10, {w2:#x}
         s_mov_b32 s11, {w3:#x}
         s_mov_b32 s12, {w4:#x}
         s_mov_b32 s13, {w5:#x}
         s_mov_b32 s14, 0
         s_mov_b32 s15, 0
         s_mov_b32 s16, {s0:#x}
         s_mov_b32 s17, {s1:#x}
         s_mov_b32 s18, {s2:#x}
         s_mov_b32 s19, {s3:#x}
         image_sample_lz {data}, v[1:2], s[8:15], s[16:19] dmask:{dmask:#x}{unorm}
         s_waitcnt vmcnt(0)",
        unorm = if unorm { " unorm" } else { "" }
    )
}

fn normalized(l: usize, k: usize) -> u32 {
    match k {
        0 => (-1.3 + (l % 16) as f32 * 0.24).to_bits(),
        _ => (-1.1 + (l / 16) as f32 * 0.9).to_bits(),
    }
}

fn unnormalized(l: usize, k: usize) -> u32 {
    match k {
        0 => (-5.5 + (l % 16) as f32 * 1.7).to_bits(),
        _ => (-3.25 + (l / 16) as f32 * 4.5).to_bits(),
    }
}

fn check(image: Image, sampler: Sampler, dmask: u32, unorm: bool, coordinates: fn(usize, usize) -> u32) {
    let inputs = lanes(2, coordinates);
    Probe::new(2, 4, &body(&image, &sampler, dmask, unorm)).texture(image.texels()).check(&inputs, |_, x| {
        let (u, v) = (f32::from_bits(x[0]), f32::from_bits(x[1]));
        let mut out: Vec<u32> = (0..4)
            .filter(|c| dmask >> c & 1 == 1)
            .map(|c| sample(&image, &sampler, unorm, u, v, image.select[c]))
            .collect();
        out.resize(4, 0);
        out
    });
}

#[test]
fn image_sample_lz_wraps_and_mirrors_normalized_coordinates_across_rows_a_pitch_apart() {
    let image = Image::new(UINT, [SEL_R, SEL_0, SEL_1, SEL_R]);
    check(image, Sampler::new([0, 1], false, 0), 0xf, false, normalized);
    check(image, Sampler::new([1, 0], false, 0), 0xf, false, normalized);
}

#[test]
fn image_sample_lz_clamps_unnormalized_coordinates_in_every_mode() {
    let image = Image::new(UINT, [SEL_R, SEL_1, SEL_R, SEL_0]);
    for mode in 0..8 {
        for border in [0, 2] {
            check(image, Sampler::new([mode, 7 - mode], true, border), 0xf, false, unnormalized);
        }
    }
}

#[test]
fn image_sample_lz_counts_texels_when_the_instruction_says_unorm() {
    let image = Image::new(UINT, [SEL_R, SEL_R, SEL_0, SEL_1]);
    check(image, Sampler::new([0, 1], false, 0), 0xf, true, unnormalized);
}

#[test]
fn image_sample_lz_converts_every_numeric_format_it_reads() {
    for format in [UNORM, SNORM, UINT, SINT] {
        let image = Image::new(format, [SEL_R, SEL_1, SEL_0, SEL_R]);
        check(image, Sampler::new([0, 0], false, 2), 0xf, false, normalized);
        check(image, Sampler::new([6, 6], true, 2), 0xf, false, unnormalized);
    }
}

#[test]
fn image_sample_lz_returns_only_the_components_its_mask_names() {
    let image = Image::new(UINT, [SEL_R, SEL_0, SEL_1, SEL_R]);
    for dmask in [0x1, 0x4, 0x5, 0x8, 0xa, 0xe] {
        check(image, Sampler::new([0, 0], false, 0), dmask, false, normalized);
    }
}

#[test]
fn image_sample_lz_needs_no_layout_for_constant_selectors() {
    let mut image = Image::new(UINT, [SEL_0, SEL_1, SEL_1, SEL_0]);
    image.tiling = 0;
    image.kind = 8;
    image.data_format = 2;
    check(image, Sampler::new([0, 0], false, 0), 0xf, false, normalized);
}

#[test]
fn the_frontend_refuses_image_instructions_it_cannot_model() {
    let image = Image::new(UINT, [SEL_R, SEL_0, SEL_0, SEL_0]);
    let sampler = Sampler::new([0, 0], false, 0);
    let refused = |text: String, reason: &str| {
        let message = Probe::new(2, 4, &text).texture(image.texels()).refusal();
        assert!(message.contains(reason), "{}", message);
    };
    let sampled = body(&image, &sampler, 0xf, false);
    let with = |from: &str, to: &str| sampled.replace(from, to);
    refused(with("dmask:0xf", "dmask:0xf r128"), "128-bit resource");
    refused(with("dmask:0xf", "dmask:0xf da"), "array");
    refused(with("v[20:23], v[1:2]", "v[20:24], v[1:2]").replace("dmask:0xf", "dmask:0xf tfe"), "texture faults");
    refused(with("dmask:0xf", "dmask:0xf lwe"), "texture faults");
    refused(with("dmask:0xf", "dmask:0xf d16"), "half-precision");
    refused(with("image_sample_lz", "image_sample"), "no SPMD lowering");
}

fn trapping(case: usize) -> (Image, Sampler) {
    let mut image = Image::new(UINT, [SEL_R, SEL_0, SEL_0, SEL_0]);
    let mut sampler = Sampler::new([0, 0], false, 0);
    match case {
        0 => image.tiling = 0,
        1 => image.kind = 8,
        2 => image.levels = [1, 1],
        3 => image.levels = [0, 1],
        4 => sampler.filters = 1 << 22,
        5 => sampler.filters = 1 << 20,
        6 => image.select = [SEL_G, SEL_0, SEL_0, SEL_0],
        7 => image.data_format = 2,
        8 => image.num_format = 7,
        _ => {
            sampler = Sampler::new([6, 6], true, 3);
        }
    }
    (image, sampler)
}

#[test]
#[ignore]
fn image_sample_lz_trapping_case() {
    let Ok(case) = std::env::var("GCN3_IMAGE_TRAP") else { return };
    let (image, sampler) = trapping(case.parse().unwrap());
    let coordinates = if sampler.unnormalized { unnormalized } else { normalized };
    let inputs = lanes(2, coordinates);
    Probe::new(2, 4, &body(&image, &sampler, 0xf, false)).texture(image.texels()).run(&inputs);
}

#[test]
fn image_sample_lz_traps_on_what_it_does_not_model() {
    use std::os::unix::process::ExitStatusExt;
    for case in 0..10 {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "image::image_sample_lz_trapping_case", "--ignored", "--test-threads=1"])
            .env("GCN3_IMAGE_TRAP", case.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(
            matches!(status.signal(), Some(libc::SIGILL) | Some(libc::SIGTRAP)),
            "case {} ended with {:?}",
            case,
            status
        );
    }
}
