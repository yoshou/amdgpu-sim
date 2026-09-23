use amdgpu_sim::rdna_spmd::{Arg, Buffer, Launch, Module};
use getopts::Options;
use std::env;
use std::io::{Error, ErrorKind, Result};

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [OPTIONS]", program);
    print!("{}", opts.usage(&brief));
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

    // Workload size, matching the SIMT `histogram` example (examples/histogram).
    let size = 1024 * 1024;
    let items_per_thread = 1024;
    let threads_per_block = 128;
    let bin_size = 256;
    let total_blocks = size / (items_per_thread * threads_per_block);

    let input: Vec<u8> = (0..size).map(|_| rand::random::<u8>()).collect();

    let arch = if matches.opt_present("arch") {
        matches.opt_str("arch").unwrap()
    } else {
        "gfx1200".to_string()
    };
    if arch != "gfx1200" {
        println!("histogram_spmd supports gfx1200 only.");
        return Ok(());
    }

    let to_io = |e: amdgpu_sim::rdna_spmd::Error| Error::new(ErrorKind::Other, e);
    let module = Module::open(format!("examples/histogram/kernel_{}.o", arch)).map_err(to_io)?;
    let function = module.function("_Z18histogram256_blockPhPji").map_err(to_io)?;

    let vec_width = matches
        .opt_str("vec_width")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(1);
    let num_threads = match matches.opt_str("num_threads") {
        Some(s) => s.parse::<usize>().unwrap().max(1),
        None => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
    };
    let launch = Launch::new([total_blocks as u32, 1, 1], [threads_per_block as u32, 1, 1])
        .width(vec_width)
        .threads(num_threads)
        .dynamic_shared((threads_per_block * bin_size) as u32);

    let data = Buffer::from_slice(&input);
    let mut block_bins = Buffer::zeroed::<u32>(bin_size * total_blocks);
    function
        .launch(
            &launch,
            &[
                Arg::read(&data),
                Arg::write(&mut block_bins),
                Arg::value(items_per_thread as u32),
            ],
        )
        .map_err(to_io)?;

    let block_bins = block_bins.as_slice::<u32>();
    let mut bins = vec![0u32; bin_size];
    for i in 0..total_blocks {
        for j in 0..bin_size {
            bins[j] += block_bins[i * bin_size + j];
        }
    }

    let mut verfy_bins = vec![0u32; bin_size];
    for i in 0..size {
        let value = input[i] as usize;
        verfy_bins[value] += 1;
    }

    let mut error = 0;
    for i in 0..bin_size {
        if bins[i] != verfy_bins[i] {
            error += 1;
        }
    }

    if error == 0 {
        println!("Validation passed.");
    } else {
        println!("Validation failed.");
    }

    Ok(())
}
