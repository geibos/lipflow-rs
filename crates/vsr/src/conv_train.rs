//! A differentiable conv2d for CPU training: im2col (custom op whose backward is col2im) and a
//! matmul. candle's own conv2d backward goes through a direct conv_transpose2d and costs 5–7×
//! its forward; here both passes are a gemm plus a memory-bound gather/scatter.

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

#[derive(Clone, Copy)]
struct Geometry {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
    ho: usize,
    wo: usize,
}

impl Geometry {
    fn new(n: usize, c: usize, h: usize, w: usize, k: usize, stride: usize, pad: usize) -> Self {
        Self { n, c, h, w, k, stride, pad, ho: (h + 2 * pad - k) / stride + 1, wo: (w + 2 * pad - k) / stride + 1 }
    }

    fn cols(&self) -> usize {
        self.c * self.k * self.k
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).min(8)
}

/// Per image: rows (oy, ox), columns (c, ky, kx) — the layout of an (Cout, C, k, k) weight.
fn im2col(x: &[f32], g: Geometry, out: &mut [f32]) {
    let per_img = g.ho * g.wo * g.cols();
    let chunk = g.n.div_ceil(threads());
    std::thread::scope(|sc| {
        for (b, block) in out.chunks_mut(per_img * chunk).enumerate() {
            sc.spawn(move || {
                for (i, img_out) in block.chunks_mut(per_img).enumerate() {
                    let img = &x[(b * chunk + i) * g.c * g.h * g.w..][..g.c * g.h * g.w];
                    for oy in 0..g.ho {
                        for ox in 0..g.wo {
                            let row = &mut img_out[(oy * g.wo + ox) * g.cols()..][..g.cols()];
                            for ch in 0..g.c {
                                for ky in 0..g.k {
                                    let iy = (oy * g.stride + ky) as isize - g.pad as isize;
                                    for kx in 0..g.k {
                                        let ix = (ox * g.stride + kx) as isize - g.pad as isize;
                                        row[(ch * g.k + ky) * g.k + kx] = if iy >= 0 && iy < g.h as isize && ix >= 0 && ix < g.w as isize {
                                            img[(ch * g.h + iy as usize) * g.w + ix as usize]
                                        } else {
                                            0.0
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

/// The adjoint of `im2col`: scatter-add columns back into (N, C, H, W).
fn col2im(col: &[f32], g: Geometry, out: &mut [f32]) {
    let per_img_in = g.c * g.h * g.w;
    let per_img_col = g.ho * g.wo * g.cols();
    let chunk = g.n.div_ceil(threads());
    std::thread::scope(|sc| {
        for (b, block) in out.chunks_mut(per_img_in * chunk).enumerate() {
            sc.spawn(move || {
                for (i, img) in block.chunks_mut(per_img_in).enumerate() {
                    let src = &col[(b * chunk + i) * per_img_col..][..per_img_col];
                    for oy in 0..g.ho {
                        for ox in 0..g.wo {
                            let row = &src[(oy * g.wo + ox) * g.cols()..][..g.cols()];
                            for ch in 0..g.c {
                                for ky in 0..g.k {
                                    let iy = (oy * g.stride + ky) as isize - g.pad as isize;
                                    if iy < 0 || iy >= g.h as isize {
                                        continue;
                                    }
                                    for kx in 0..g.k {
                                        let ix = (ox * g.stride + kx) as isize - g.pad as isize;
                                        if ix >= 0 && ix < g.w as isize {
                                            img[(ch * g.h + iy as usize) * g.w + ix as usize] += row[(ch * g.k + ky) * g.k + kx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

struct Im2col(Geometry);

impl CustomOp1 for Im2col {
    fn name(&self) -> &'static str {
        "im2col-train"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let g = self.0;
        let CpuStorage::F32(data) = s else { candle_core::bail!("im2col-train expects f32") };
        let Some((a, b)) = l.contiguous_offsets() else { candle_core::bail!("im2col-train expects a contiguous input") };
        let mut out = vec![0f32; g.n * g.ho * g.wo * g.cols()];
        im2col(&data[a..b], g, &mut out);
        Ok((CpuStorage::F32(out), Shape::from((g.n * g.ho * g.wo, g.cols()))))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let g = self.0;
        let gr = grad_res.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let mut out = vec![0f32; g.n * g.c * g.h * g.w];
        col2im(&gr, g, &mut out);
        Ok(Some(Tensor::from_vec(out, arg.shape(), arg.device())?))
    }
}

/// conv2d(x: (N, C, H, W), w: (Cout, C, k, k)), square kernel, no bias, NCHW out.
pub fn conv2d(x: &Tensor, w: &Tensor, stride: usize, pad: usize) -> Result<Tensor> {
    let (n, c, h, wd) = x.dims4()?;
    let (cout, _, k, _) = w.dims4()?;
    let g = Geometry::new(n, c, h, wd, k, stride, pad);
    let col = x.contiguous()?.apply_op1(Im2col(g))?;
    let y = col.matmul(&w.reshape((cout, g.cols()))?.t()?)?; // (N*Ho*Wo, Cout)
    y.reshape((n, g.ho, g.wo, cout))?.permute((0, 3, 1, 2))?.contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn matches_candle_conv_and_gradients() -> Result<()> {
        let dev = Device::Cpu;
        let x = Var::randn(0f32, 1., (3, 4, 9, 9), &dev)?;
        let w = Var::randn(0f32, 0.3, (5, 4, 3, 3), &dev)?;
        for (stride, pad) in [(1, 1), (2, 1), (2, 0)] {
            let a = conv2d(x.as_tensor(), w.as_tensor(), stride, pad)?;
            let b = x.as_tensor().conv2d(w.as_tensor(), pad, stride, 1, 1)?;
            let d = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(d < 1e-4, "forward differs by {d}");
            let (ga, gb) = (a.sqr()?.sum_all()?.backward()?, b.sqr()?.sum_all()?.backward()?);
            for v in [&x, &w] {
                let (p, q) = (ga.get(v.as_tensor()).expect("grad"), gb.get(v.as_tensor()).expect("grad"));
                let d = (p - q)?.abs()?.max_all()?.to_scalar::<f32>()?;
                let scale = q.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(d <= 1e-4 * scale.max(1.0), "gradient differs by {d}");
            }
        }
        Ok(())
    }
}
