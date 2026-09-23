use amdgpu_sim::rdna_spmd::{Arg, Buffer, Launch, Module};
use getopts::Options;
use std::env;
use std::io::{Error, ErrorKind, Result};
use std::time::Instant;

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [OPTIONS]", program);
    print!("{}", opts.usage(&brief));
}

fn expected_matrix_transpose(input: &[f32], width: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; width * width];
    for y in 0..width {
        for x in 0..width {
            output[x * width + y] = input[y * width + x];
        }
    }
    output
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optopt("", "arch", "Architecture", "ARCH");
    opts.optopt("", "vec_width", "SPMD work-item packing width W", "W");
    opts.optopt("", "num_threads", "CPU dispatch thread count", "N");
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
        println!("warp_shuffle_spmd supports gfx1200 only.");
        return Ok(());
    }

    let width = 4usize;
    let size = width * width;
    let input = (1..=size)
        .map(|value| value as f32 * 10.0)
        .collect::<Vec<_>>();
    let expected = expected_matrix_transpose(&input, width);

    // Reuse the same kernel object as the warp_shuffle example; the cross-lane
    // `ds_bpermute_b32` warp shuffle is handled on the segmented de-SIMT path.
    let to_io = |e: amdgpu_sim::rdna_spmd::Error| Error::new(ErrorKind::Other, e);
    let module = Module::open(format!("examples/warp_shuffle/kernel_{}.o", arch)).map_err(to_io)?;
    let function = module.function("_Z23matrix_transpose_kernelPfPKfj").map_err(to_io)?;

    let vec_width = matches
        .opt_str("vec_width")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(1);
    let num_threads = match matches.opt_str("num_threads") {
        Some(s) => s.parse::<usize>().unwrap().max(1),
        None => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
    };
    let launch = Launch::new([1, 1, 1], [width as u32, width as u32, 1])
        .width(vec_width)
        .threads(num_threads);

    let mut output = Buffer::zeroed::<f32>(size);
    let data = Buffer::from_slice(&input);
    let args = [
        Arg::write(&mut output),
        Arg::read(&data),
        Arg::value(width as u32),
    ];
    function.prepare(&launch, &args).map_err(to_io)?;
    let start = Instant::now();
    function.launch(&launch, &args).map_err(to_io)?;
    let end = start.elapsed();
    println!("Elapsed time: {:.3} [ms]", end.as_secs_f64() * 1000.0);
    drop(args);

    let output = output.to_vec::<f32>();
    if output == expected {
        println!("Validation passed.");
    } else {
        println!("Validation failed.");
        println!("output: {:?}", output);
        println!("expected: {:?}", expected);
    }

    Ok(())
}
