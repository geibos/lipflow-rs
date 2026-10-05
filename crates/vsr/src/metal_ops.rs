//! Hand-written Metal kernels for the convolutional front end.
//!
//! candle's Metal conv2d spends ~95% of its time in generic strided copies (im2col with
//! arbitrary strides, then a transpose back to NCHW); its matmul is fast. The front end here
//! keeps activations channels-last (NHWC), so a convolution is one coalesced im2col kernel,
//! one candle matmul and one fused bias(+residual)(+swish) kernel, with no transposes.

use std::ffi::c_void;
use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Device, Layout, MetalDevice, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Frames (T+KT-1, H, W), already padded in time -> rows (t, oy, ox), cols (kt, ky, kx).
kernel void im2col_frames(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                          constant uint* p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint T = p[0], H = p[1], W = p[2], Ho = p[3], Wo = p[4], KT = p[5], KH = p[6], KW = p[7], S = p[8], P = p[9];
    uint K = KT * KH * KW;
    if (gid >= T * Ho * Wo * K) return;
    uint k = gid % K, row = gid / K;
    uint ox = row % Wo, oy = (row / Wo) % Ho, t = row / (Wo * Ho);
    uint kx = k % KW, ky = (k / KW) % KH, kt = k / (KW * KH);
    int iy = int(oy * S + ky) - int(P), ix = int(ox * S + kx) - int(P);
    float v = 0.0f;
    if (iy >= 0 && iy < int(H) && ix >= 0 && ix < int(W)) v = x[((t + kt) * H + uint(iy)) * W + uint(ix)];
    out[gid] = v;
}

// Channels of frames (C, T+KT-1, H, W), padded in time -> rows (t, oy, ox), cols (c, kt, ky, kx).
kernel void im2col_frames_c(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                            constant uint* p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint C = p[0], T = p[1], H = p[2], W = p[3], Ho = p[4], Wo = p[5], KT = p[6], KH = p[7], KW = p[8], S = p[9], P = p[10];
    uint K = KT * KH * KW, CK = C * K;
    if (gid >= T * Ho * Wo * CK) return;
    uint ck = gid % CK, row = gid / CK;
    uint c = ck / K, k = ck % K;
    uint ox = row % Wo, oy = (row / Wo) % Ho, t = row / (Wo * Ho);
    uint kx = k % KW, ky = (k / KW) % KH, kt = k / (KW * KH);
    int iy = int(oy * S + ky) - int(P), ix = int(ox * S + kx) - int(P);
    float v = 0.0f;
    if (iy >= 0 && iy < int(H) && ix >= 0 && ix < int(W)) v = x[((c * (T + KT - 1) + t + kt) * H + uint(iy)) * W + uint(ix)];
    out[gid] = v;
}

// NHWC (N, H, W, C) -> rows (n, oy, ox), cols (ky, kx, c).
kernel void im2col_nhwc(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                        constant uint* p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint N = p[0], H = p[1], W = p[2], C = p[3], Ho = p[4], Wo = p[5], KH = p[6], KW = p[7], S = p[8], P = p[9];
    uint KC = KH * KW * C;
    if (gid >= N * Ho * Wo * KC) return;
    uint c = gid % C, tap = (gid / C) % (KH * KW), row = gid / KC;
    uint kx = tap % KW, ky = tap / KW;
    uint ox = row % Wo, oy = (row / Wo) % Ho, n = row / (Wo * Ho);
    int iy = int(oy * S + ky) - int(P), ix = int(ox * S + kx) - int(P);
    float v = 0.0f;
    if (iy >= 0 && iy < int(H) && ix >= 0 && ix < int(W)) v = x[((n * H + uint(iy)) * W + uint(ix)) * C + c];
    out[gid] = v;
}

// y + bias[c] (+ residual), then an activation. p = (total, C, has_residual, act: 0 none, 1 swish, 2 relu).
kernel void bias_act(device const float* y [[buffer(0)]], device const float* b [[buffer(1)]],
                     device const float* r [[buffer(2)]], device float* out [[buffer(3)]],
                     constant uint* p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    float v = y[gid] + b[gid % p[1]];
    if (p[2] != 0) v += r[gid];
    if (p[3] == 1) v = v / (1.0f + exp(-v));
    else if (p[3] == 2) v = max(v, 0.0f);
    out[gid] = v;
}

// NHWC max pool, padding counts as -inf. p = (N, H, W, C, Ho, Wo, K, S, P).
kernel void maxpool_nhwc(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint* p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint N = p[0], H = p[1], W = p[2], C = p[3], Ho = p[4], Wo = p[5], K = p[6], S = p[7], P = p[8];
    if (gid >= N * Ho * Wo * C) return;
    uint c = gid % C, row = gid / C;
    uint ox = row % Wo, oy = (row / Wo) % Ho, n = row / (Wo * Ho);
    float m = -INFINITY;
    for (uint ky = 0; ky < K; ky++) {
        int iy = int(oy * S + ky) - int(P);
        if (iy < 0 || iy >= int(H)) continue;
        for (uint kx = 0; kx < K; kx++) {
            int ix = int(ox * S + kx) - int(P);
            if (ix < 0 || ix >= int(W)) continue;
            m = max(m, x[((n * H + uint(iy)) * W + uint(ix)) * C + c]);
        }
    }
    out[gid] = m;
}

// (N, L, C) -> (N, C) mean over L.
kernel void mean_mid(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                     constant uint* p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint N = p[0], L = p[1], C = p[2];
    if (gid >= N * C) return;
    uint c = gid % C, n = gid / C;
    float s = 0.0f;
    for (uint l = 0; l < L; l++) s += x[(n * L + l) * C + c];
    out[gid] = s / float(L);
}
"#;

pub struct Pipelines {
    im2col_frames: ComputePipeline,
    im2col_frames_c: ComputePipeline,
    im2col_nhwc: ComputePipeline,
    bias_act: ComputePipeline,
    maxpool: ComputePipeline,
    mean_mid: ComputePipeline,
}

fn merr(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("metal: {e}"))
}

impl Pipelines {
    pub fn new(dev: &Device) -> candle_core::Result<Arc<Self>> {
        let Device::Metal(m) = dev else { candle_core::bail!("Metal pipelines need a Metal device") };
        let lib = m.device().new_library_with_source(SRC, None).map_err(merr)?;
        let pl = |name: &str| -> candle_core::Result<ComputePipeline> {
            let f = lib.get_function(name, None).map_err(merr)?;
            m.device().new_compute_pipeline_state_with_function(&f).map_err(merr)
        };
        Ok(Arc::new(Self {
            im2col_frames: pl("im2col_frames")?,
            im2col_frames_c: pl("im2col_frames_c")?,
            im2col_nhwc: pl("im2col_nhwc")?,
            bias_act: pl("bias_act")?,
            maxpool: pl("maxpool_nhwc")?,
            mean_mid: pl("mean_mid")?,
        }))
    }
}

/// Buffer and byte offset of a contiguous f32 tensor storage.
fn input<'a>(s: &'a MetalStorage, l: &Layout) -> candle_core::Result<(&'a Buffer, usize)> {
    if !l.is_contiguous() || s.dtype() != DType::F32 {
        candle_core::bail!("metal front-end kernels take contiguous f32 inputs");
    }
    Ok((s.buffer(), l.start_offset() * 4))
}

fn dispatch(dev: &MetalDevice, p: &ComputePipeline, ins: &[(&Buffer, usize)], n_out: usize, params: &[u32]) -> candle_core::Result<MetalStorage> {
    let out = dev.new_buffer_builder().with_size_for(n_out, DType::F32).with_label("lipflow_frontend").build()?;
    {
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(p);
        for (i, (b, off)) in ins.iter().enumerate() {
            enc.set_input_buffer(i, Some(b), *off);
        }
        enc.set_output_buffer(ins.len(), Some(&out), 0);
        enc.set_bytes_directly(ins.len() + 1, std::mem::size_of_val(params), params.as_ptr().cast::<c_void>());
        let tg = p.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize { width: n_out, height: 1, depth: 1 }, MTLSize { width: tg, height: 1, depth: 1 });
    }
    Ok(MetalStorage::new(out, dev.clone(), n_out, DType::F32))
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("front-end tensor dimension fits in u32")
}

fn cpu_unsupported(name: &str) -> candle_core::Result<(CpuStorage, Shape)> {
    candle_core::bail!("{name} is Metal-only; the CPU path uses candle's conv2d")
}

struct Im2colFrames {
    p: Arc<Pipelines>,
    k: (usize, usize, usize),
    stride: usize,
    pad: usize,
}

impl CustomOp1 for Im2colFrames {
    fn name(&self) -> &'static str {
        "im2col_frames"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        cpu_unsupported(self.name())
    }

    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        let (tp, h, w) = l.shape().dims3()?;
        let (kt, kh, kw) = self.k;
        let t = tp + 1 - kt;
        let ho = (h + 2 * self.pad - kh) / self.stride + 1;
        let wo = (w + 2 * self.pad - kw) / self.stride + 1;
        let kk = kt * kh * kw;
        let params = [t, h, w, ho, wo, kt, kh, kw, self.stride, self.pad].map(u);
        let out = dispatch(s.device(), &self.p.im2col_frames, &[input(s, l)?], t * ho * wo * kk, &params)?;
        Ok((out, Shape::from((t * ho * wo, kk))))
    }
}

struct Im2colNhwc {
    p: Arc<Pipelines>,
    k: usize,
    stride: usize,
    pad: usize,
}

impl CustomOp1 for Im2colNhwc {
    fn name(&self) -> &'static str {
        "im2col_nhwc"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        cpu_unsupported(self.name())
    }

    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        let (n, h, w, c) = l.shape().dims4()?;
        let ho = (h + 2 * self.pad - self.k) / self.stride + 1;
        let wo = (w + 2 * self.pad - self.k) / self.stride + 1;
        let kc = self.k * self.k * c;
        let params = [n, h, w, c, ho, wo, self.k, self.k, self.stride, self.pad].map(u);
        let out = dispatch(s.device(), &self.p.im2col_nhwc, &[input(s, l)?], n * ho * wo * kc, &params)?;
        Ok((out, Shape::from((n * ho * wo, kc))))
    }
}

/// Activation after bias (+ residual): the kernel's `act` code.
#[derive(Clone, Copy)]
pub enum Act {
    None = 0,
    Swish = 1,
    Relu = 2,
}

struct BiasAct {
    p: Arc<Pipelines>,
    act: Act,
}

impl CustomOp2 for BiasAct {
    fn name(&self) -> &'static str {
        "bias_act"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        cpu_unsupported(self.name())
    }

    fn metal_fwd(&self, y: &MetalStorage, yl: &Layout, b: &MetalStorage, bl: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        let n = yl.shape().elem_count();
        let c = bl.shape().elem_count();
        let yb = input(y, yl)?;
        let params = [n, c, 0, self.act as usize].map(u);
        // The residual slot is unused (flag 0) but must be bound: pass y again.
        let out = dispatch(y.device(), &self.p.bias_act, &[yb, input(b, bl)?, yb], n, &params)?;
        Ok((out, yl.shape().clone()))
    }
}

struct BiasResAct {
    p: Arc<Pipelines>,
    act: Act,
}

impl CustomOp3 for BiasResAct {
    fn name(&self) -> &'static str {
        "bias_res_act"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        cpu_unsupported(self.name())
    }

    fn metal_fwd(
        &self,
        y: &MetalStorage,
        yl: &Layout,
        b: &MetalStorage,
        bl: &Layout,
        r: &MetalStorage,
        rl: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let n = yl.shape().elem_count();
        if rl.shape().elem_count() != n {
            candle_core::bail!("residual shape mismatch");
        }
        let params = [n, bl.shape().elem_count(), 1, self.act as usize].map(u);
        let out = dispatch(y.device(), &self.p.bias_act, &[input(y, yl)?, input(b, bl)?, input(r, rl)?], n, &params)?;
        Ok((out, yl.shape().clone()))
    }
}

struct MaxPool {
    p: Arc<Pipelines>,
}

impl CustomOp1 for MaxPool {
    fn name(&self) -> &'static str {
        "maxpool_nhwc"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        cpu_unsupported(self.name())
    }

    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        let (n, h, w, c) = l.shape().dims4()?;
        let (k, st, pad) = (3, 2, 1);
        let ho = (h + 2 * pad - k) / st + 1;
        let wo = (w + 2 * pad - k) / st + 1;
        let params = [n, h, w, c, ho, wo, k, st, pad].map(u);
        let out = dispatch(s.device(), &self.p.maxpool, &[input(s, l)?], n * ho * wo * c, &params)?;
        Ok((out, Shape::from((n, ho, wo, c))))
    }
}

struct MeanMid {
    p: Arc<Pipelines>,
}

impl CustomOp1 for MeanMid {
    fn name(&self) -> &'static str {
        "mean_mid"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        cpu_unsupported(self.name())
    }

    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        let (n, len, c) = l.shape().dims3()?;
        let out = dispatch(s.device(), &self.p.mean_mid, &[input(s, l)?], n * c, &[n, len, c].map(u))?;
        Ok((out, Shape::from((n, c))))
    }
}

/// Convolution as im2col + matmul with BN folded into `w` (K, Cout) and `b` (Cout).
pub struct ConvNhwc {
    pub w: Tensor,
    pub b: Tensor,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
}

impl ConvNhwc {
    /// (N, H, W, Cin) -> (N, Ho, Wo, Cout) before bias; returns rows (N*Ho*Wo, Cout).
    fn matmul(&self, p: &Arc<Pipelines>, x: &Tensor) -> candle_core::Result<(Tensor, [usize; 3])> {
        let (n, h, w, _) = x.dims4()?;
        let ho = (h + 2 * self.pad - self.k) / self.stride + 1;
        let wo = (w + 2 * self.pad - self.k) / self.stride + 1;
        let col = x.apply_op1_no_bwd(&Im2colNhwc { p: p.clone(), k: self.k, stride: self.stride, pad: self.pad })?;
        Ok((col.matmul(&self.w)?, [n, ho, wo]))
    }

    pub fn forward(&self, p: &Arc<Pipelines>, x: &Tensor, swish: bool) -> candle_core::Result<Tensor> {
        self.forward_act(p, x, if swish { Act::Swish } else { Act::None })
    }

    pub fn forward_act(&self, p: &Arc<Pipelines>, x: &Tensor, act: Act) -> candle_core::Result<Tensor> {
        let (y, [n, ho, wo]) = self.matmul(p, x)?;
        let y = y.apply_op2_no_bwd(&self.b, &BiasAct { p: p.clone(), act })?;
        y.reshape((n, ho, wo, self.w.dim(1)?))
    }

    /// swish(conv(x) + b + residual)
    pub fn forward_residual(&self, p: &Arc<Pipelines>, x: &Tensor, residual: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_residual_act(p, x, residual, Act::Swish)
    }

    pub fn forward_residual_act(&self, p: &Arc<Pipelines>, x: &Tensor, residual: &Tensor, act: Act) -> candle_core::Result<Tensor> {
        let (y, [n, ho, wo]) = self.matmul(p, x)?;
        let r = residual.reshape(y.shape())?;
        let y = y.apply_op3_no_bwd(&self.b, &r, &BiasResAct { p: p.clone(), act })?;
        y.reshape((n, ho, wo, self.w.dim(1)?))
    }
}

/// Temporal-spatial first convolution over padded frames (T+KT-1, H, W) -> (T, Ho, Wo, Cout).
pub fn conv_frames(p: &Arc<Pipelines>, frames: &Tensor, w: &Tensor, b: &Tensor, k: (usize, usize, usize), stride: usize, pad: usize) -> candle_core::Result<Tensor> {
    let (tp, h, wd) = frames.dims3()?;
    let t = tp + 1 - k.0;
    let ho = (h + 2 * pad - k.1) / stride + 1;
    let wo = (wd + 2 * pad - k.2) / stride + 1;
    let col = frames.apply_op1_no_bwd(&Im2colFrames { p: p.clone(), k, stride, pad })?;
    let y = col.matmul(w)?.apply_op2_no_bwd(b, &BiasAct { p: p.clone(), act: Act::Swish })?;
    y.reshape((t, ho, wo, w.dim(1)?))
}

struct Im2colFramesC {
    p: Arc<Pipelines>,
    k: (usize, usize, usize),
    stride: usize,
    pad: usize,
}

impl CustomOp1 for Im2colFramesC {
    fn name(&self) -> &'static str {
        "im2col_frames_c"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        cpu_unsupported(self.name())
    }

    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        let (c, tp, h, w) = l.shape().dims4()?;
        let (kt, kh, kw) = self.k;
        let t = tp + 1 - kt;
        let ho = (h + 2 * self.pad - kh) / self.stride + 1;
        let wo = (w + 2 * self.pad - kw) / self.stride + 1;
        let ck = c * kt * kh * kw;
        let params = [c, t, h, w, ho, wo, kt, kh, kw, self.stride, self.pad].map(u);
        let out = dispatch(s.device(), &self.p.im2col_frames_c, &[input(s, l)?], t * ho * wo * ck, &params)?;
        Ok((out, Shape::from((t * ho * wo, ck))))
    }
}

/// Multi-channel temporal-spatial convolution over frames (C, T+KT-1, H, W), padded in time,
/// with `w` (C·KT·KH·KW, Cout) → (T, Ho, Wo, Cout).
pub fn conv_frames_c(p: &Arc<Pipelines>, frames: &Tensor, w: &Tensor, b: &Tensor, k: (usize, usize, usize), stride: usize, pad: usize, act: Act) -> candle_core::Result<Tensor> {
    let (_, tp, h, wd) = frames.dims4()?;
    let t = tp + 1 - k.0;
    let ho = (h + 2 * pad - k.1) / stride + 1;
    let wo = (wd + 2 * pad - k.2) / stride + 1;
    let col = frames.apply_op1_no_bwd(&Im2colFramesC { p: p.clone(), k, stride, pad })?;
    let y = col.matmul(w)?.apply_op2_no_bwd(b, &BiasAct { p: p.clone(), act })?;
    y.reshape((t, ho, wo, w.dim(1)?))
}

pub fn maxpool3s2(p: &Arc<Pipelines>, x: &Tensor) -> candle_core::Result<Tensor> {
    x.apply_op1_no_bwd(&MaxPool { p: p.clone() })
}

/// (N, H, W, C) -> (N, C)
pub fn global_avg(p: &Arc<Pipelines>, x: &Tensor) -> candle_core::Result<Tensor> {
    let (n, h, w, c) = x.dims4()?;
    x.reshape((n, h * w, c))?.apply_op1_no_bwd(&MeanMid { p: p.clone() })
}
