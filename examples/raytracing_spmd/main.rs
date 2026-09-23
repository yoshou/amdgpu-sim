use amdgpu_sim::buffer::{get_u64, set_u64};
use amdgpu_sim::rdna_spmd::{Arg, Buffer, Launch, Module};
use getopts::Options;
use png::*;
use std::env;
use std::io::{BufWriter, Error, ErrorKind, Result};

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

    let res_x = 512;
    let res_y = 512;
    let ao_radius = 200f32;

    let mut pixels = Buffer::new(res_x * res_y * 4);
    for i in 0..(res_x * res_y) {
        pixels.as_mut_slice::<u8>()[i * 4 + 3] = 255;
    }

    let mut geometry = Buffer::from_slice(&std::fs::read("examples/raytracing/cornellbox.bin")?);
    println!("geometry size: {}", geometry.len());
    let geometry_ptr = geometry.address();
    let bytes = geometry.as_mut_slice::<u8>();
    let box_nodes = get_u64(bytes, 0);
    set_u64(bytes, 0, box_nodes + geometry_ptr);
    let prim_nodes = get_u64(bytes, 8);
    set_u64(bytes, 8, prim_nodes + geometry_ptr);

    let arch = if matches.opt_present("arch") {
        matches.opt_str("arch").unwrap()
    } else {
        "gfx1200".to_string()
    };
    if arch != "gfx1200" {
        println!("Unsupported architecture: {}", arch);
        return Ok(());
    }

    let to_io = |e: amdgpu_sim::rdna_spmd::Error| Error::new(ErrorKind::Other, e);
    let module = Module::open(format!("examples/raytracing/kernel_{}.o", arch)).map_err(to_io)?;
    let function = module
        .function("_Z24ambient_occlusion_kernelP14_hiprtGeometryPh15HIP_vector_typeIiLj2EEf")
        .map_err(to_io)?;

    let block_dim = [8u32, 8, 1];
    let grid_dim = [
        ceil_div(res_x as u32, block_dim[0]),
        ceil_div(res_y as u32, block_dim[1]),
        1,
    ];
    let num_threads = match matches.opt_str("num_threads") {
        Some(s) => s.parse::<usize>().unwrap().max(1),
        None => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
    };
    // --vec_width=W selects the width-W SPMD path. Default is 16.
    let vec_w: u32 = match matches.opt_str("vec_width") {
        Some(s) => s.parse::<u32>().unwrap(),
        None => 16,
    };
    let launch = Launch::new(grid_dim, block_dim)
        .width(vec_w)
        .threads(num_threads);

    let args = [
        Arg::read(&geometry),
        Arg::write(&mut pixels),
        Arg::value([res_x as i32, res_y as i32]),
        Arg::value(ao_radius),
    ];
    function.prepare(&launch, &args).map_err(to_io)?;
    use std::time::Instant;
    let start = Instant::now();
    function.launch(&launch, &args).map_err(to_io)?;
    println!("Elapsed time: {:.3} [ms]", start.elapsed().as_secs_f64() * 1000.0);
    drop(args);

    let file = std::fs::File::create("image.png")?;
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, res_x as u32, res_y as u32);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    writer.write_image_data(pixels.as_slice::<u8>())?;

    Ok(())
}
