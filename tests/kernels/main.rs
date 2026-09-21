mod harness;

use harness::{close, same, Arg, Kernels, Run, WIDTHS};

fn ramp(n: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..n).map(|k| ((k * 37 % 211) as f32) * scale + bias).collect()
}

fn ints(n: usize, m: i32) -> Vec<i32> {
    (0..n).map(|k| (k as i32 * 37) % m - m / 2).collect()
}

#[test]
fn saxpy_scales_and_adds() {
    let k = Kernels::load();
    let n = 300usize;
    let x = ramp(n, 0.25, -3.0);
    let a = 2.5f32;
    let want: Vec<f32> = x.iter().enumerate().map(|(i, &v)| a * v + i as f32).collect();
    for width in WIDTHS {
        let mut y: Vec<f32> = (0..n).map(|i| i as f32).collect();
        k.run(
            &Run {
                kernel: "saxpy",
                wg: [64, 1, 1],
                grid: [5, 1, 1],
                args: &[
                    Arg::Ptr(y.as_mut_ptr() as u64),
                    Arg::Ptr(x.as_ptr() as u64),
                    Arg::F32(a),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        same("saxpy", width, "the reference", &y, &want);
    }
}

#[test]
fn sgemm_multiplies_tiles() {
    let k = Kernels::load();
    let (m, n, kk) = (32usize, 32, 48);
    let a = ramp(m * kk, 0.05, -1.0);
    let b = ramp(kk * n, 0.03, -0.5);
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for t in 0..kk {
                acc += a[r * kk + t] * b[t * n + c];
            }
            want[r * n + c] = acc;
        }
    }
    for width in WIDTHS {
        let mut c = vec![0f32; m * n];
        k.run(
            &Run {
                kernel: "sgemm_tiled",
                wg: [16, 16, 1],
                grid: [(n / 16) as u32, (m / 16) as u32, 1],
                args: &[
                    Arg::Ptr(c.as_mut_ptr() as u64),
                    Arg::Ptr(a.as_ptr() as u64),
                    Arg::Ptr(b.as_ptr() as u64),
                    Arg::I32(m as i32),
                    Arg::I32(n as i32),
                    Arg::I32(kk as i32),
                ],
            },
            width,
        );
        close("sgemm_tiled", width, &c, &want, 1e-5);
    }
}

#[test]
fn matvec_multiplies_rows() {
    let k = Kernels::load();
    let (m, n) = (100usize, 40usize);
    let a = ramp(m * n, 0.02, -0.4);
    let x = ramp(n, 0.1, -1.0);
    let want: Vec<f32> = (0..m)
        .map(|r| (0..n).fold(0f32, |acc, c| a[r * n + c].mul_add(x[c], acc)))
        .collect();
    for width in WIDTHS {
        let mut y = vec![0f32; m];
        k.run(
            &Run {
                kernel: "matvec",
                wg: [32, 1, 1],
                grid: [4, 1, 1],
                args: &[
                    Arg::Ptr(y.as_mut_ptr() as u64),
                    Arg::Ptr(a.as_ptr() as u64),
                    Arg::Ptr(x.as_ptr() as u64),
                    Arg::I32(m as i32),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        close("matvec", width, &y, &want, 1e-5);
    }
}

#[test]
fn spmv_walks_each_row() {
    let k = Kernels::load();
    let rows = 64usize;
    let mut rp = vec![0i32];
    let mut ci = vec![];
    let mut v = vec![];
    for r in 0..rows {
        for c in r.saturating_sub(2)..(r + 3).min(rows) {
            ci.push(c as i32);
            v.push(((r * 7 + c) % 13) as f32 * 0.25);
        }
        rp.push(ci.len() as i32);
    }
    let x = ramp(rows, 0.1, -1.0);
    let want: Vec<f32> = (0..rows)
        .map(|r| {
            (rp[r]..rp[r + 1]).fold(0f32, |acc, j| acc + v[j as usize] * x[ci[j as usize] as usize])
        })
        .collect();
    for width in WIDTHS {
        let mut y = vec![0f32; rows];
        k.run(
            &Run {
                kernel: "spmv_csr",
                wg: [32, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(y.as_mut_ptr() as u64),
                    Arg::Ptr(rp.as_ptr() as u64),
                    Arg::Ptr(ci.as_ptr() as u64),
                    Arg::Ptr(v.as_ptr() as u64),
                    Arg::Ptr(x.as_ptr() as u64),
                    Arg::I32(rows as i32),
                ],
            },
            width,
        );
        close("spmv_csr", width, &y, &want, 1e-5);
    }
}

#[test]
fn reduce_sums_each_workgroup() {
    let k = Kernels::load();
    let n = 512usize;
    let input = ramp(n, 0.1, -5.0);
    for width in WIDTHS {
        let mut out = vec![0f32; 2];
        k.run(
            &Run {
                kernel: "reduce_sum",
                wg: [256, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        let want: Vec<f32> = (0..2)
            .map(|b| input[b * 256..(b + 1) * 256].iter().sum())
            .collect();
        close("reduce_sum", width, &out, &want, 1e-4);
    }
}

#[test]
fn argmax_finds_the_largest_and_its_index() {
    let k = Kernels::load();
    let n = 300usize;
    let input = ramp(n, 0.5, -20.0);
    let want_index = input
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32;
    let want_value = input[want_index as usize];
    for wg in [64u32, 128, 256] {
        for width in WIDTHS {
            let mut oi = vec![-1i32; 1];
            let mut ov = vec![0f32; 1];
            k.run(
                &Run {
                    kernel: "argmax",
                    wg: [wg, 1, 1],
                    grid: [1, 1, 1],
                    args: &[
                        Arg::Ptr(oi.as_mut_ptr() as u64),
                        Arg::Ptr(ov.as_mut_ptr() as u64),
                        Arg::Ptr(input.as_ptr() as u64),
                        Arg::I32(n as i32),
                    ],
                },
                width,
            );
            assert_eq!(
                (oi[0], ov[0]),
                (want_index, want_value),
                "argmax at wg={} W={}",
                wg,
                width
            );
        }
    }
}

#[test]
fn scan_is_an_inclusive_prefix_sum() {
    let k = Kernels::load();
    let n = 256usize;
    let input: Vec<i32> = (0..n as i32).map(|v| v % 7 - 3).collect();
    let mut want = input.clone();
    for i in 1..n {
        want[i] += want[i - 1];
    }
    for width in WIDTHS {
        let mut out = vec![0i32; n];
        k.run(
            &Run {
                kernel: "scan_block",
                wg: [256, 1, 1],
                grid: [1, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        same("scan_block", width, "the reference", &out, &want);
    }
}

#[test]
fn prefix_max_runs_along_each_row() {
    let k = Kernels::load();
    let (w, h) = (40usize, 32usize);
    let input = ramp(w * h, 0.3, -10.0);
    let mut want = vec![0f32; w * h];
    for y in 0..h {
        let mut m = f32::NEG_INFINITY;
        for x in 0..w {
            m = m.max(input[y * w + x]);
            want[y * w + x] = m;
        }
    }
    for width in WIDTHS {
        let mut out = vec![0f32; w * h];
        k.run(
            &Run {
                kernel: "prefix_max_2d",
                wg: [32, 1, 1],
                grid: [1, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(w as i32),
                    Arg::I32(h as i32),
                ],
            },
            width,
        );
        same("prefix_max_2d", width, "the reference", &out, &want);
    }
}

#[test]
fn warp_reduce_totals_the_input() {
    let k = Kernels::load();
    let n = 256usize;
    let input = ramp(n, 0.125, -4.0);
    let want: f32 = input.iter().sum();
    for width in WIDTHS {
        let mut total = vec![0f32; 1];
        k.run(
            &Run {
                kernel: "warp_reduce_sum",
                wg: [64, 1, 1],
                grid: [4, 1, 1],
                args: &[
                    Arg::Ptr(total.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        close("warp_reduce_sum", width, &total, &[want], 1e-4);
    }
}

#[test]
fn softmax_sums_to_one() {
    let k = Kernels::load();
    let n = 200usize;
    let input = ramp(n, 0.05, -2.0);
    let m = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = input.iter().map(|v| (v - m).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let want: Vec<f32> = exp.iter().map(|v| v / sum).collect();
    for width in WIDTHS {
        let mut out = vec![0f32; n];
        k.run(
            &Run {
                kernel: "softmax",
                wg: [256, 1, 1],
                grid: [1, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        close("softmax", width, &out, &want, 1e-4);
    }
}

#[test]
fn layernorm_centres_and_scales() {
    let k = Kernels::load();
    let n = 128usize;
    let rows = 2usize;
    let input = ramp(rows * n, 0.2, -6.0);
    let mut want = vec![0f32; rows * n];
    for r in 0..rows {
        let row = &input[r * n..(r + 1) * n];
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let rs = 1.0 / (var + 1e-5).sqrt();
        for (c, v) in row.iter().enumerate() {
            want[r * n + c] = (v - mean) * rs;
        }
    }
    for width in WIDTHS {
        let mut out = vec![0f32; rows * n];
        k.run(
            &Run {
                kernel: "layernorm",
                wg: [128, 1, 1],
                grid: [rows as u32, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        close("layernorm", width, &out, &want, 1e-3);
    }
}

#[test]
fn gelu_matches_its_formula() {
    let k = Kernels::load();
    let n = 128usize;
    let input = ramp(n, 0.05, -2.5);
    let want: Vec<f32> = input
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh()))
        .collect();
    for width in WIDTHS {
        let mut out = vec![0f32; n];
        k.run(
            &Run {
                kernel: "gelu",
                wg: [64, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        close("gelu", width, &out, &want, 1e-3);
    }
}

#[test]
fn relu_backward_gates_the_gradient() {
    let k = Kernels::load();
    let n = 128usize;
    let x = ramp(n, 0.1, -5.0);
    let dy = ramp(n, 0.05, 1.0);
    let want: Vec<f32> = x
        .iter()
        .zip(&dy)
        .map(|(&x, &d)| if x > 0.0 { d } else { 0.0 })
        .collect();
    for width in WIDTHS {
        let mut dx = vec![0f32; n];
        k.run(
            &Run {
                kernel: "relu_bwd",
                wg: [64, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(dx.as_mut_ptr() as u64),
                    Arg::Ptr(dy.as_ptr() as u64),
                    Arg::Ptr(x.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        same("relu_bwd", width, "the reference", &dx, &want);
    }
}

#[test]
fn transpose_exchanges_the_axes() {
    let k = Kernels::load();
    let (w, h) = (32usize, 32usize);
    let input = ramp(w * h, 0.5, 0.0);
    let want: Vec<f32> = (0..w * h).map(|k| input[(k % h) * w + k / h]).collect();
    for width in WIDTHS {
        let mut out = vec![0f32; w * h];
        k.run(
            &Run {
                kernel: "transpose_lds",
                wg: [32, 32, 1],
                grid: [1, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(w as i32),
                    Arg::I32(h as i32),
                ],
            },
            width,
        );
        same("transpose_lds", width, "the reference", &out, &want);
    }
}

#[test]
fn stencil_averages_the_neighbourhood() {
    let k = Kernels::load();
    let (w, h) = (32usize, 16usize);
    let input = ramp(w * h, 0.25, -3.0);
    let mut want = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            want[y * w + x] = 0.2
                * (input[y * w + x]
                    + input[y * w + x - 1]
                    + input[y * w + x + 1]
                    + input[(y - 1) * w + x]
                    + input[(y + 1) * w + x]);
        }
    }
    for width in WIDTHS {
        let mut out = vec![0f32; w * h];
        k.run(
            &Run {
                kernel: "stencil5",
                wg: [32, 4, 1],
                grid: [1, 4, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(w as i32),
                    Arg::I32(h as i32),
                ],
            },
            width,
        );
        close("stencil5", width, &out, &want, 1e-5);
    }
}

#[test]
fn conv1d_slides_the_kernel() {
    let k = Kernels::load();
    let n = 128usize;
    let ks = 5usize;
    let input = ramp(n, 0.2, -2.0);
    let filter = ramp(ks, 0.3, 0.1);
    let want: Vec<f32> = (0..n)
        .map(|x| {
            (0..ks).fold(0f32, |acc, j| {
                let p = x as isize + j as isize - (ks / 2) as isize;
                if p >= 0 && (p as usize) < n {
                    acc + input[p as usize] * filter[j]
                } else {
                    acc
                }
            })
        })
        .collect();
    for width in WIDTHS {
        let mut out = vec![0f32; n];
        k.run(
            &Run {
                kernel: "conv1d",
                wg: [64, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::Ptr(filter.as_ptr() as u64),
                    Arg::I32(n as i32),
                    Arg::I32(ks as i32),
                ],
            },
            width,
        );
        close("conv1d", width, &out, &want, 1e-5);
    }
}

#[test]
fn rgb_to_gray_weights_the_channels() {
    let k = Kernels::load();
    let n = 128usize;
    let rgb: Vec<u8> = (0..n * 3).map(|k| (k * 29 % 256) as u8).collect();
    let want: Vec<u8> = (0..n)
        .map(|t| {
            ((77 * rgb[t * 3] as u32 + 150 * rgb[t * 3 + 1] as u32 + 29 * rgb[t * 3 + 2] as u32)
                >> 8) as u8
        })
        .collect();
    for width in WIDTHS {
        let mut out = vec![0u8; n];
        k.run(
            &Run {
                kernel: "rgb_to_gray",
                wg: [64, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(rgb.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        same("rgb_to_gray", width, "the reference", &out, &want);
    }
}

#[test]
fn bitonic_step_sorts_its_pairs() {
    let k = Kernels::load();
    let n = 64usize;
    let start = ints(n, 100);
    for (j, kk) in [(1i32, 2i32), (2, 4), (1, 4)] {
        let mut want = start.clone();
        for i in 0..n {
            let l = i ^ j as usize;
            if l > i {
                let up = (i & kk as usize) == 0;
                if (want[i] > want[l]) == up {
                    want.swap(i, l);
                }
            }
        }
        for width in WIDTHS {
            let mut a = start.clone();
            k.run(
                &Run {
                    kernel: "bitonic_step",
                    wg: [64, 1, 1],
                    grid: [1, 1, 1],
                    args: &[Arg::Ptr(a.as_mut_ptr() as u64), Arg::I32(j), Arg::I32(kk)],
                },
                width,
            );
            same("bitonic_step", width, "the reference", &a, &want);
        }
    }
}

#[test]
fn histogram_counts_every_byte() {
    let k = Kernels::load();
    let n = 1024usize;
    let data: Vec<u8> = (0..n).map(|k| (k * 31 % 256) as u8).collect();
    let mut want = vec![0u32; 256];
    for &b in &data {
        want[b as usize] += 1;
    }
    for width in WIDTHS {
        let mut bins = vec![0u32; 256];
        k.run(
            &Run {
                kernel: "histogram",
                wg: [256, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(bins.as_mut_ptr() as u64),
                    Arg::Ptr(data.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        same("histogram", width, "the reference", &bins, &want);
    }
}

#[test]
fn compact_keeps_the_positive_values() {
    let k = Kernels::load();
    let n = 256usize;
    let input = ints(n, 40);
    let mut want: Vec<i32> = input.iter().copied().filter(|&v| v > 0).collect();
    want.sort_unstable();
    for width in WIDTHS {
        let mut out = vec![0i32; n];
        let mut count = vec![0i32; 1];
        k.run(
            &Run {
                kernel: "compact",
                wg: [64, 1, 1],
                grid: [4, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(count.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        assert_eq!(count[0] as usize, want.len(), "compact count at W={}", width);
        let mut got: Vec<i32> = out[..want.len()].to_vec();
        got.sort_unstable();
        same("compact", width, "the reference", &got, &want);
    }
}

#[test]
fn count_matches_totals_the_equal_pairs() {
    let k = Kernels::load();
    let n = 200usize;
    let a = ints(n, 11);
    let b: Vec<i32> = a.iter().enumerate().map(|(i, &v)| if i % 3 == 0 { v } else { v + 1 }).collect();
    let want = a.iter().zip(&b).filter(|(x, y)| x == y).count() as i32;
    for width in WIDTHS {
        let mut count = vec![0i32; 1];
        k.run(
            &Run {
                kernel: "count_matches",
                wg: [64, 1, 1],
                grid: [4, 1, 1],
                args: &[
                    Arg::Ptr(count.as_mut_ptr() as u64),
                    Arg::Ptr(a.as_ptr() as u64),
                    Arg::Ptr(b.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        assert_eq!(count[0], want, "count_matches at W={}", width);
    }
}

#[test]
fn dot_product_totals_the_products() {
    let k = Kernels::load();
    let n = 512usize;
    let a = ramp(n, 0.05, -1.0);
    let b = ramp(n, 0.03, 0.5);
    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    for width in WIDTHS {
        let mut out = vec![0f32; 1];
        k.run(
            &Run {
                kernel: "dot_product",
                wg: [256, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(a.as_ptr() as u64),
                    Arg::Ptr(b.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        close("dot_product", width, &out, &[want], 1e-4);
    }
}

#[test]
fn mandelbrot_counts_the_iterations() {
    let k = Kernels::load();
    let (w, h, maxit) = (32usize, 16usize, 40i32);
    let want: Vec<u32> = (0..w * h)
        .map(|p| {
            let (px, py) = (p % w, p / w);
            let cr = (px as f32 / w as f32) * 3.5 - 2.5;
            let ci = (py as f32 / h as f32) * 2.0 - 1.0;
            let (mut zr, mut zi, mut it) = (0f32, 0f32, 0i32);
            while zr * zr + zi * zi <= 4.0 && it < maxit {
                let t = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = t;
                it += 1;
            }
            it as u32
        })
        .collect();
    for width in WIDTHS {
        let mut out = vec![0u32; w * h];
        k.run(
            &Run {
                kernel: "mandelbrot",
                wg: [32, 4, 1],
                grid: [1, 4, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::I32(w as i32),
                    Arg::I32(h as i32),
                    Arg::I32(maxit),
                ],
            },
            width,
        );
        same("mandelbrot", width, "the reference", &out, &want);
    }
}

#[test]
fn nbody_advances_the_positions() {
    let k = Kernels::load();
    let n = 64usize;
    let px0 = ramp(n, 0.5, -8.0);
    let py0 = ramp(n, 0.25, -4.0);
    let mass: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32).collect();
    let dt = 0.01f32;
    let mut want_x = px0.clone();
    let mut want_y = py0.clone();
    for i in 0..n {
        let (mut ax, mut ay) = (0f32, 0f32);
        for j in 0..n {
            let dx = px0[j] - px0[i];
            let dy = py0[j] - py0[i];
            let d2 = dx * dx + dy * dy + 1e-6;
            let inv = 1.0 / d2.sqrt();
            let w = mass[j] * inv * inv * inv;
            ax += dx * w;
            ay += dy * w;
        }
        want_x[i] = px0[i] + ax * dt;
        want_y[i] = py0[i] + ay * dt;
    }
    for width in WIDTHS {
        let mut px = px0.clone();
        let mut py = py0.clone();
        k.run(
            &Run {
                kernel: "nbody_step",
                wg: [64, 1, 1],
                grid: [1, 1, 1],
                args: &[
                    Arg::Ptr(px.as_mut_ptr() as u64),
                    Arg::Ptr(py.as_mut_ptr() as u64),
                    Arg::Ptr(mass.as_ptr() as u64),
                    Arg::I32(n as i32),
                    Arg::F32(dt),
                ],
            },
            width,
        );
        close("nbody_step", width, &px, &want_x, 2e-2);
        close("nbody_step", width, &py, &want_y, 2e-2);
    }
}

#[test]
fn u64_hash_mixes_the_bits() {
    let k = Kernels::load();
    let n = 128usize;
    let input: Vec<u64> = (0..n as u64).map(|v| v.wrapping_mul(0x9e3779b97f4a7c15)).collect();
    let want: Vec<u64> = input
        .iter()
        .map(|&v| {
            let mut x = v;
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51afd7ed558ccd);
            x ^= x >> 33;
            x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
            x ^= x >> 33;
            x
        })
        .collect();
    for width in WIDTHS {
        let mut out = vec![0u64; n];
        k.run(
            &Run {
                kernel: "u64_hash",
                wg: [64, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        same("u64_hash", width, "the reference", &out, &want);
    }
}

#[test]
fn clamp_short_saturates() {
    let k = Kernels::load();
    let n = 128usize;
    let input: Vec<i32> = (0..n as i32).map(|v| (v - 64) * 2000).collect();
    let want: Vec<i16> = input.iter().map(|&v| v.clamp(-32768, 32767) as i16).collect();
    for width in WIDTHS {
        let mut out = vec![0i16; n];
        k.run(
            &Run {
                kernel: "clamp_short",
                wg: [64, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
        );
        same("clamp_short", width, "the reference", &out, &want);
    }
}

#[test]
fn mixed_math_divides_and_transcends() {
    let k = Kernels::load();
    let n = 96usize;
    let stride = 7u32;
    let input: Vec<f64> = (0..n).map(|k| (k as f64) * 0.125 - 4.0).collect();
    let want: Vec<f64> = (0..n)
        .map(|i| {
            let row = (i as u32 / stride) as f64;
            let col = (i as u32 % stride) as f64;
            let v = input[i];
            (v * v + 1.0).sqrt() / (1.0 + row) + col.sin()
        })
        .collect();
    for width in WIDTHS {
        let mut out = vec![0f64; n];
        k.run(
            &Run {
                kernel: "mixed_math",
                wg: [32, 1, 1],
                grid: [3, 1, 1],
                args: &[
                    Arg::Ptr(out.as_mut_ptr() as u64),
                    Arg::Ptr(input.as_ptr() as u64),
                    Arg::U32(n as u32),
                    Arg::U32(stride),
                ],
            },
            width,
        );
        for (at, (&g, &w)) in out.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                "mixed_math at W={} index {}: got {}, expected {}",
                width,
                at,
                g,
                w
            );
        }
    }
}

#[test]
fn several_host_threads_agree() {
    let k = Kernels::load();
    let n = 1024usize;
    let data: Vec<u8> = (0..n).map(|k| (k * 31 % 256) as u8).collect();
    let mut want = vec![0u32; 256];
    for &b in &data {
        want[b as usize] += 1;
    }
    for width in WIDTHS {
        let mut bins = vec![0u32; 256];
        k.run_threaded(
            &Run {
                kernel: "histogram",
                wg: [256, 1, 1],
                grid: [2, 1, 1],
                args: &[
                    Arg::Ptr(bins.as_mut_ptr() as u64),
                    Arg::Ptr(data.as_ptr() as u64),
                    Arg::I32(n as i32),
                ],
            },
            width,
            8,
        );
        same("histogram", width, "the reference", &bins, &want);
    }
}
