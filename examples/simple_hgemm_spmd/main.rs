use amdgpu_sim::rdna_spmd::{Arg, Buffer, Launch, Module};
use getopts::Options;
use half::f16;
use std::env;
use std::io::{Error, ErrorKind, Result};

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [OPTIONS]", program);
    print!("{}", opts.usage(&brief));
}

fn ceil_div<T>(x: T, y: T) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + From<u32>
        + Copy,
{
    (x + y - T::from(1)) / y
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optopt("", "arch", "Architecture", "ARCH");
    opts.optopt(
        "",
        "vec_width",
        "SPMD packet width W (0,1,2,4,8,16; cross-lane ops rendezvous per wave)",
        "W",
    );
    opts.optopt("", "num_threads", "CPU dispatch thread count", "N");
    opts.optflag(
        "",
        "verify_widths",
        "Run W=0,1,2,4,8,16 on the same inputs and require bitwise-identical output",
    );
    opts.optflag("h", "help", "Print help");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => {
            panic!("{}", f.to_string())
        }
    };
    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let arch = if matches.opt_present("arch") {
        matches.opt_str("arch").unwrap()
    } else {
        "gfx1200".to_string()
    };

    if arch != "gfx1200" {
        println!("simple_hgemm_spmd supports gfx1200 only.");
        return Ok(());
    }

    let m = 256;
    let n = 256;
    let k = 256;
    let alpha = 2.1f32;
    let beta = 2.1f32;

    let lda = k;
    let ldb = k;
    let ldc = n;
    let ldd = ldc;

    const ROCWMMA_M: u32 = 16;
    const ROCWMMA_N: u32 = 16;
    const WAVE_SIZE: u32 = 32;
    const T_BLOCK_X: u32 = 4 * WAVE_SIZE;
    const T_BLOCK_Y: u32 = 4;

    let mut matrix_a = vec![f16::ZERO; m * k];
    let mut matrix_b = vec![f16::ZERO; k * n];
    let mut matrix_c = vec![f16::ZERO; m * n];
    let mut matrix_d = vec![f16::ZERO; m * n];

    for i in 0..(m * k) {
        let value = rand::random::<f32>();
        matrix_a[i] = f16::from_f32(value);
    }

    for i in 0..(k * n) {
        let value = rand::random::<f32>();
        matrix_b[i] = f16::from_f32(value);
    }

    for i in 0..(m * n) {
        let value = rand::random::<f32>();
        matrix_c[i] = f16::from_f32(value);
    }

    for i in 0..(m * n) {
        matrix_d[i] = f16::NAN;
    }

    let matrix_a_ptr = (&matrix_a[0] as *const f16) as u64;
    println!("matrix_a_ptr: 0x{:16X}", matrix_a_ptr);

    let matrix_b_ptr = (&matrix_b[0] as *const f16) as u64;
    println!("matrix_b_ptr: 0x{:16X}", matrix_b_ptr);

    let matrix_c_ptr = (&matrix_c[0] as *const f16) as u64;
    println!("matrix_c_ptr: 0x{:16X}", matrix_c_ptr);

    let matrix_d_ptr = (&matrix_d[0] as *const f16) as u64;
    println!("matrix_d_ptr: 0x{:16X}", matrix_d_ptr);

    // Reuse the same kernel object as the simple_hgemm example; the wave-wide
    // `v_wmma_f32_16x16x16_f16` matrix multiply is lifted to a coroutine yield
    let to_io = |e: amdgpu_sim::rdna_spmd::Error| Error::new(ErrorKind::Other, e);
    let module = Module::open(format!("examples/simple_hgemm/kernel_{}.o", arch)).map_err(to_io)?;
    let function = module
        .function("_Z15hgemm_rocwmma_djjjPKDF16_S0_S0_PDF16_jjjjff")
        .map_err(to_io)?;

    let vec_width = matches
        .opt_str("vec_width")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(1);
    let verify_widths = matches.opt_present("verify_widths");
    let widths: Vec<u32> = if verify_widths {
        vec![1, 2, 4, 8, 16, 32]
    } else {
        vec![vec_width]
    };
    let block_dim = [T_BLOCK_X, T_BLOCK_Y, 1];
    let grid_dim = [
        ceil_div(m as u32, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
        ceil_div(n as u32, ROCWMMA_N * T_BLOCK_Y),
        1,
    ];
    let num_threads = match matches.opt_str("num_threads") {
        Some(s) => s.parse::<usize>().unwrap().max(1),
        None => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
    };

    let bits = |matrix: &[f16]| matrix.iter().map(|value| value.to_bits()).collect::<Vec<u16>>();
    let buffer_a = Buffer::from_slice(&bits(&matrix_a));
    let buffer_b = Buffer::from_slice(&bits(&matrix_b));
    let buffer_c = Buffer::from_slice(&bits(&matrix_c));
    let mut buffer_d = Buffer::from_slice(&bits(&matrix_d));

    use std::time::Instant;
    let mut baseline_bits: Option<Vec<u16>> = None;
    for width in widths {
        buffer_d.as_mut_slice::<u16>().fill(f16::NAN.to_bits());
        println!("vec_width={}", width);
        let launch = Launch::new(grid_dim, block_dim)
            .width(width)
            .threads(num_threads);
        let args = [
            Arg::value(m as u32),
            Arg::value(n as u32),
            Arg::value(k as u32),
            Arg::read(&buffer_a),
            Arg::read(&buffer_b),
            Arg::read(&buffer_c),
            Arg::write(&mut buffer_d),
            Arg::value(lda as u32),
            Arg::value(ldb as u32),
            Arg::value(ldc as u32),
            Arg::value(ldd as u32),
            Arg::value(alpha),
            Arg::value(beta),
        ];
        function.prepare(&launch, &args).map_err(to_io)?;
        let start = Instant::now();
        function.launch(&launch, &args).map_err(to_io)?;
        println!(
            "Elapsed time: {:.3} [ms]",
            start.elapsed().as_secs_f64() * 1000.0
        );
        drop(args);
        if verify_widths {
            let actual_bits: Vec<u16> = buffer_d.to_vec();
            if width == 1 {
                baseline_bits = Some(actual_bits);
            } else {
                let baseline = baseline_bits.as_ref().unwrap();
                let mut mismatches = baseline
                    .iter()
                    .zip(&actual_bits)
                    .enumerate()
                    .filter(|(_, (expected, actual))| expected != actual);
                if let Some((index, (expected, actual))) = mismatches.next() {
                    let mismatch_count = 1 + mismatches.count();
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "vec_width={} differs from vec_width=1 at {} elements; first index {}: {:#06x} != {:#06x}",
                            width, mismatch_count, index, actual, expected
                        ),
                    ));
                }
                println!("Bitwise match with vec_width=1: passed.");
            }
        }
    }
    matrix_d = buffer_d
        .as_slice::<u16>()
        .iter()
        .map(|&b| f16::from_bits(b))
        .collect();

    let mut expect_matrix_d = vec![f16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let a = matrix_a[i * lda + p].to_f32();
                let b = matrix_b[j * ldb + p].to_f32();
                acc += a * b;
            }
            let c = matrix_c[i * ldc + j].to_f32();
            let value = alpha * acc + beta * c;
            expect_matrix_d[i * ldc + j] = f16::from_f32(value);
        }
    }

    let mut max_relative_error = 0.0f64;
    let mut mismatch_examples = Vec::new();

    for i in 0..(m * n) {
        let val1 = matrix_d[i].to_f64();
        let val2 = expect_matrix_d[i].to_f64();

        let num = (val1 - val2).abs();
        let denom = val1.abs() + val2.abs() + 1.0;

        let relative_error = num / denom;

        max_relative_error = max_relative_error.max(relative_error);
        if relative_error >= 10.0 * f16::EPSILON.to_f64() && mismatch_examples.len() < 8 {
            mismatch_examples.push((i, val1, val2, relative_error));
        }
    }
    let tolerance = 10.0;
    let eps = f16::EPSILON.to_f64();

    println!("Max relative error: {}", max_relative_error);

    if max_relative_error < tolerance * eps {
        println!("Validation passed.");
    } else {
        println!("Validation failed.");
        for (index, actual, expected, error) in mismatch_examples {
            println!(
                "  mismatch[{}] (row {}, col {}): actual={}, expected={}, relative_error={}",
                index,
                index / n,
                index % n,
                actual,
                expected,
                error
            );
        }
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("matrix validation failed: max relative error {}", max_relative_error),
        ));
    }

    Ok(())
}
