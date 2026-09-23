use amdgpu_sim::buffer::set_bits_u32;
use amdgpu_sim::rdna_spmd::{Arg, Buffer, Launch, Module};
use getopts::Options;
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

const SEL_0: u32 = 0;
const SEL_X: u32 = 4;
const FMT_8_UINT: u32 = 5;
const IMAGE_CHANNEL_ORDER_A: u32 = 0;
const IMAGE_CHANNEL_TYPE_SNORM_INT8: u32 = 0;
const IMG_2D_ARRAY: u32 = 13;
const TEX_WRAP: u32 = 0;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optopt("", "arch", "Architecture", "ARCH");
    opts.optopt("", "vec_width", "SPMD work-item packing width W", "W");
    opts.optopt("", "num_threads", "CPU dispatch thread count", "N");
    opts.optflag("h", "help", "Print help");
    let matches = opts.parse(&args[1..]).unwrap_or_else(|f| panic!("{}", f));
    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let arch = matches
        .opt_str("arch")
        .unwrap_or_else(|| "gfx1200".to_string());
    if arch != "gfx1200" {
        println!("texture_spmd supports gfx1200 only.");
        return Ok(());
    }

    let vec_w = match matches.opt_str("vec_width") {
        Some(s) => s.parse::<u32>().unwrap(),
        None => 16,
    };

    let size_x = 1024;
    let size_y = 1024;
    let size = size_x * size_y;

    let texels: Vec<u8> = (0..size).map(|i| i as u8).collect();
    let data = Buffer::from_slice(&texels);
    let data_ptr = data.address();

    let hist_bin_count = 7;
    let mut histogram = Buffer::zeroed::<u32>(hist_bin_count);

    let mut tex_obj = vec![0u32; 12 + 4];
    set_bits_u32(&mut tex_obj[0..8], 0, 32, (data_ptr >> 8) as u32);
    set_bits_u32(&mut tex_obj[0..8], 32, 8, (data_ptr >> 40) as u32);
    set_bits_u32(&mut tex_obj[0..8], 49, 8, FMT_8_UINT);
    set_bits_u32(&mut tex_obj[0..8], 62, 16, size_x as u32 - 1);
    set_bits_u32(&mut tex_obj[0..8], 78, 16, size_y as u32 - 1);
    set_bits_u32(&mut tex_obj[0..8], 96, 3, SEL_X);
    set_bits_u32(&mut tex_obj[0..8], 99, 3, SEL_0);
    set_bits_u32(&mut tex_obj[0..8], 102, 3, SEL_0);
    set_bits_u32(&mut tex_obj[0..8], 105, 3, SEL_0);
    set_bits_u32(&mut tex_obj[0..8], 124, 4, IMG_2D_ARRAY);

    tex_obj[8] = IMAGE_CHANNEL_TYPE_SNORM_INT8;
    tex_obj[9] = IMAGE_CHANNEL_ORDER_A;
    tex_obj[10] = size_x as u32;

    set_bits_u32(&mut tex_obj[12..16], 0, 3, TEX_WRAP);
    set_bits_u32(&mut tex_obj[12..16], 3, 3, TEX_WRAP);
    set_bits_u32(&mut tex_obj[12..16], 6, 3, TEX_WRAP);
    let texture = Buffer::from_slice(&tex_obj);

    let to_io = |e: amdgpu_sim::rdna_spmd::Error| Error::new(ErrorKind::Other, e);
    let module = Module::open(format!("examples/texture/kernel_{}.o", arch)).map_err(to_io)?;
    let function = module.function("_Z16histogram_kernelPjjjjP13__hip_texture").map_err(to_io)?;

    let block_dim = [16u32, 16, 1];
    let grid_dim = [
        ceil_div(size_x as u32, block_dim[0]),
        ceil_div(size_y as u32, block_dim[1]),
        1,
    ];
    let num_threads = match matches.opt_str("num_threads") {
        Some(s) => s.parse::<usize>().unwrap().max(1),
        None => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
    };
    let launch = Launch::new(grid_dim, block_dim)
        .width(vec_w)
        .threads(num_threads);

    use std::time::Instant;
    println!(
        "Scalar JIT dispatching on {} threads (vec_width={})...",
        num_threads, vec_w
    );
    let start = Instant::now();
    function
        .launch(
            &launch,
            &[
                Arg::write(&mut histogram),
                Arg::value(size_x as u32),
                Arg::value(size_y as u32),
                Arg::value(hist_bin_count as u32),
                Arg::read(&texture),
            ],
        )
        .map_err(to_io)?;
    println!("Elapsed time: {:.3} [ms]", start.elapsed().as_secs_f64() * 1000.0);

    println!(
        "Equal-width histogram with {} bins of values [0, {}) mod 256:",
        hist_bin_count, size
    );

    let mut sum = 0;
    for (i, &count) in histogram.as_slice::<u32>().iter().enumerate() {
        print!("bin[{}] = {}", i, count);
        if i + 1 < hist_bin_count {
            print!(", ");
        } else {
            print!("\n");
        }
        sum += count;
    }

    println!("sum of bins: {}", sum);

    Ok(())
}
