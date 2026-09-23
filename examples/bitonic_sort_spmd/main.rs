use itertools::Itertools;

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
    opts.optopt(
        "s",
        "sort",
        "Sort in decreasing (dec) or increasing (inc) order.",
        "SORT",
    );
    opts.optopt(
        "l",
        "log2length",
        "2**l will be the length of the array to be sorted.",
        "LEN",
    );
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

    let steps: u32 = matches.opt_get_default("log2length", 15).unwrap();
    let length = 1usize << steps;
    let sort_increasing = if matches.opt_get_default("sort", "inc".to_string()).unwrap() == "inc" {
        1u8
    } else {
        0u8
    };

    let input: Vec<u32> = (0..length).map(|_| rand::random::<u32>()).collect();

    let arch = if matches.opt_present("arch") {
        matches.opt_str("arch").unwrap()
    } else {
        "gfx1200".to_string()
    };
    if arch != "gfx1200" {
        println!("bitonic_sort_spmd supports gfx1200 only.");
        return Ok(());
    }

    let expect = if sort_increasing == 1 {
        input.iter().copied().sorted().collect_vec()
    } else {
        input
            .iter()
            .copied()
            .sorted_by(|a, b| b.cmp(a))
            .collect_vec()
    };

    let to_io = |e: amdgpu_sim::rdna_spmd::Error| Error::new(ErrorKind::Other, e);
    let module = Module::open(format!("examples/bitonic_sort/kernel_{}.o", arch)).map_err(to_io)?;
    let function = module.function("_Z19bitonic_sort_kernelPjjjb").map_err(to_io)?;

    let vec_width = matches
        .opt_str("vec_width")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(1);
    let num_threads = match matches.opt_str("num_threads") {
        Some(s) => s.parse::<usize>().unwrap().max(1),
        None => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
    };
    let local_threads = if length > 256 { 256 } else { length / 2 };
    let global_threads = length / 2;
    let launch = Launch::new(
        [(global_threads / local_threads) as u32, 1, 1],
        [local_threads as u32, 1, 1],
    )
    .width(vec_width)
    .threads(num_threads);

    let mut data = Buffer::from_slice(&input);
    for i in 0..steps {
        for j in 0..(i + 1) {
            let args = [
                Arg::write(&mut data),
                Arg::value(i),
                Arg::value(j),
                Arg::value(sort_increasing),
            ];
            function.launch(&launch, &args).map_err(to_io)?;
        }
    }

    if data.as_slice::<u32>() == &expect[..] {
        println!("Validation passed.");
    } else {
        println!("Validation failed.");
    }

    Ok(())
}
