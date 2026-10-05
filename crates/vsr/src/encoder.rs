//! Visual encoder: 3D-conv + ResNet-18 front end, then a 12-block Conformer
//! (ESPnet `Encoder` with `input_layer=conv3d`, `rel_mha`, macaron FFN, conv module).

use anyhow::Result;
use candle_core::{D, Tensor};

use crate::nn::{BatchNorm, FeedForward, LayerNorm, Linear, Prefixed, heads, merge_heads, pad2d, swish};

/// Frames are pushed through the 2D trunk in chunks to bound the im2col buffers.
const FRONTEND_CHUNK: usize = 64;

/// conv2d; inside a CPU training graph, the im2col version with a fast backward.
fn conv(x: &Tensor, k: &Tensor, pad: usize, stride: usize) -> Result<Tensor> {
    if x.device().is_cpu() && (x.track_op() || k.track_op()) {
        return Ok(crate::conv_train::conv2d(x, k, stride, pad)?);
    }
    Ok(x.conv2d(k, pad, stride, 1, 1)?)
}

struct BasicBlock {
    conv1: Tensor,
    bn1: BatchNorm,
    conv2: Tensor,
    bn2: BatchNorm,
    down: Option<(Tensor, BatchNorm)>,
    stride: usize,
}

impl BasicBlock {
    fn load(w: &Prefixed, stride: usize) -> Result<Self> {
        let down = match w.get("downsample.0.weight") {
            Ok(k) => Some((k, BatchNorm::load(&w.pp("downsample.1"))?)),
            Err(_) => None,
        };
        Ok(Self {
            conv1: w.get("conv1.weight")?,
            bn1: BatchNorm::load(&w.pp("bn1"))?,
            conv2: w.get("conv2.weight")?,
            bn2: BatchNorm::load(&w.pp("bn2"))?,
            down,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = conv(x, &self.conv1, 1, self.stride)?;
        let out = swish(&self.bn1.forward(&out, 1)?)?;
        let out = self.bn2.forward(&conv(&out, &self.conv2, 1, 1)?, 1)?;
        let residual = match &self.down {
            Some((k, bn)) => bn.forward(&conv(x, k, 0, self.stride)?, 1)?,
            None => x.clone(),
        };
        swish(&(out + residual)?)
    }
}

pub struct Frontend {
    conv3d: Tensor, // (64, 5, 7, 7): the 5 temporal taps as input channels
    bn0: BatchNorm,
    blocks: Vec<BasicBlock>,
    #[cfg(feature = "metal")]
    fast: Option<fast::FastFrontend>,
}

impl Frontend {
    pub fn load(w: &Prefixed) -> Result<Self> {
        let k = w.get("frontend3D.0.weight")?; // (64, 1, 5, 7, 7)
        let (o, _, kt, kh, kw) = k.dims5()?;
        let mut blocks = Vec::new();
        for (li, stride) in [(1, 1), (2, 2), (3, 2), (4, 2)] {
            blocks.push(BasicBlock::load(&w.pp(&format!("trunk.layer{li}.0")), stride)?);
            blocks.push(BasicBlock::load(&w.pp(&format!("trunk.layer{li}.1")), 1)?);
        }
        let f = Self {
            conv3d: k.reshape((o, kt, kh, kw))?,
            bn0: BatchNorm::load(&w.pp("frontend3D.1"))?,
            blocks,
            #[cfg(feature = "metal")]
            fast: None,
        };
        #[cfg(feature = "metal")]
        let f = {
            let mut f = f;
            if w.device().is_metal() && !w.trainable() {
                f.fast = Some(fast::FastFrontend::new(&f, w.device())?);
            }
            f
        };
        Ok(f)
    }

    /// (T, 88, 88) normalised frames -> (T, 512) features.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(fast) = &self.fast {
            return fast.forward(x);
        }
        self.forward_reference(x)
    }

    /// The straightforward NCHW implementation on candle ops (CPU, and the reference).
    pub fn forward_reference(&self, x: &Tensor) -> Result<Tensor> {
        let (t, h, w) = x.dims3()?;
        let kt = self.conv3d.dim(1)?;
        let half = kt / 2;
        // Conv3d with one input channel == Conv2d whose channels are the shifted frames.
        let xp = x.pad_with_zeros(0, half, half)?;
        let mut outs = Vec::new();
        let mut start = 0;
        while start < t {
            let n = FRONTEND_CHUNK.min(t - start);
            let taps: Vec<Tensor> = (0..kt).map(|k| xp.narrow(0, start + k, n)).collect::<candle_core::Result<_>>()?;
            let frames = Tensor::stack(&taps, 1)?; // (n, kt, H, W)
            debug_assert_eq!(frames.dims(), &[n, kt, h, w]);
            let y = conv(&frames, &self.conv3d, 3, 2)?;
            let y = swish(&self.bn0.forward(&y, 1)?)?;
            let padded = pad2d(&y, 1, f32::NEG_INFINITY)?;
            let mut y = if padded.track_op() { max_pool_3s2_diff(&padded)? } else { padded.max_pool2d_with_stride(3, 2)? };
            for b in &self.blocks {
                y = b.forward(&y)?;
            }
            outs.push(y.mean((2, 3))?);
            start += n;
        }
        Ok(Tensor::cat(&outs, 0)?)
    }
}

#[cfg(feature = "metal")]
mod fast {
    use std::sync::Arc;

    use anyhow::Result;
    use candle_core::{Device, Tensor};

    use super::{FRONTEND_CHUNK, Frontend};
    use crate::metal_ops::{ConvNhwc, Pipelines, conv_frames, global_avg, maxpool3s2};
    use crate::nn::BatchNorm;

    struct Block {
        conv1: ConvNhwc,
        conv2: ConvNhwc,
        down: Option<ConvNhwc>,
    }

    /// The front end on hand-written Metal kernels, channels-last, BN folded into the convs.
    pub(super) struct FastFrontend {
        p: Arc<Pipelines>,
        w0: Tensor, // (KT*KH*KW, 64)
        b0: Tensor,
        k0: (usize, usize, usize),
        blocks: Vec<Block>,
    }

    /// (Cout, Cin, k, k) conv + BN -> (k*k*Cin, Cout) im2col weight and bias.
    fn fold(k: &Tensor, bn: &BatchNorm, stride: usize) -> Result<ConvNhwc> {
        let (o, i, kh, _) = k.dims4()?;
        let (scale, shift) = bn.affine()?;
        let w = k.broadcast_mul(&scale.reshape((o, 1, 1, 1))?)?;
        let w = w.permute((2, 3, 1, 0))?.contiguous()?.reshape((kh * kh * i, o))?;
        Ok(ConvNhwc { w, b: shift, k: kh, stride, pad: kh / 2 })
    }

    impl FastFrontend {
        pub(super) fn new(f: &Frontend, dev: &Device) -> Result<Self> {
            let (o, kt, kh, kw) = f.conv3d.dims4()?;
            let (scale0, shift0) = f.bn0.affine()?;
            let w0 = f.conv3d.broadcast_mul(&scale0.reshape((o, 1, 1, 1))?)?.reshape((o, kt * kh * kw))?.t()?.contiguous()?;
            let blocks = f
                .blocks
                .iter()
                .map(|b| {
                    Ok(Block {
                        conv1: fold(&b.conv1, &b.bn1, b.stride)?,
                        conv2: fold(&b.conv2, &b.bn2, 1)?,
                        down: b.down.as_ref().map(|(k, bn)| fold(k, bn, b.stride)).transpose()?,
                    })
                })
                .collect::<Result<_>>()?;
            Ok(Self { p: Pipelines::new(dev)?, w0, b0: shift0, k0: (kt, kh, kw), blocks })
        }

        pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let t = x.dim(0)?;
            let half = self.k0.0 / 2;
            let xp = x.pad_with_zeros(0, half, half)?.contiguous()?;
            let mut outs = Vec::new();
            let mut start = 0;
            while start < t {
                let n = FRONTEND_CHUNK.min(t - start);
                let frames = xp.narrow(0, start, n + 2 * half)?;
                let y = conv_frames(&self.p, &frames, &self.w0, &self.b0, self.k0, 2, 3)?;
                let mut y = maxpool3s2(&self.p, &y)?;
                for b in &self.blocks {
                    let h = b.conv1.forward(&self.p, &y, true)?;
                    let r = match &b.down {
                        Some(d) => d.forward(&self.p, &y, false)?,
                        None => y.clone(),
                    };
                    y = b.conv2.forward_residual(&self.p, &h, &r)?;
                }
                outs.push(global_avg(&self.p, &y)?);
                start += n;
            }
            Ok(Tensor::cat(&outs, 0)?)
        }
    }
}

/// 3×3 / stride-2 max pool of an already padded (N, C, H, W) tensor built from differentiable
/// ops (candle's max_pool2d has no backward when kernel != stride).
fn max_pool_3s2_diff(x: &Tensor) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    let (ho, wo) = ((h - 3) / 2 + 1, (w - 3) / 2 + 1);
    // every other element along `dim`, starting at `off`, `len` of them
    let strided = |t: &Tensor, dim: usize, off: usize, len: usize| -> candle_core::Result<Tensor> {
        let t = t.narrow(dim, off, 2 * len - 1)?.pad_with_zeros(dim, 0, 1)?;
        let mut shape = t.dims().to_vec();
        shape[dim] = len;
        shape.insert(dim + 1, 2);
        t.reshape(shape)?.narrow(dim + 1, 0, 1)?.squeeze(dim + 1)
    };
    let mut out: Option<Tensor> = None;
    for ky in 0..3 {
        let rows = strided(x, 2, ky, ho)?;
        for kx in 0..3 {
            let v = strided(&rows, 3, kx, wo)?;
            out = Some(match out {
                Some(o) => o.maximum(&v)?,
                None => v,
            });
        }
    }
    let out = out.ok_or_else(|| anyhow::anyhow!("empty pool"))?;
    debug_assert_eq!(out.dims(), &[n, c, ho, wo]);
    Ok(out)
}

struct Block1d {
    conv1: Tensor,
    bn1: BatchNorm,
    conv2: Tensor,
    bn2: BatchNorm,
    down: Option<(Tensor, BatchNorm)>,
    stride: usize,
}

/// ESPnet Conv1dResNet: raw waveform -> 25 fps features (640 samples per frame).
pub struct AudioFrontend {
    conv1: Tensor, // (64, 1, 80), stride 4, padding 38
    bn1: BatchNorm,
    blocks: Vec<Block1d>,
}

impl AudioFrontend {
    fn load(w: &Prefixed) -> Result<Self> {
        let t = w.pp("trunk");
        let mut blocks = Vec::new();
        for (li, stride) in [(1, 1), (2, 2), (3, 2), (4, 2)] {
            for bi in 0..2 {
                let b = t.pp(&format!("layer{li}.{bi}"));
                let down = match b.get("downsample.0.weight") {
                    Ok(k) => Some((k, BatchNorm::load(&b.pp("downsample.1"))?)),
                    Err(_) => None,
                };
                blocks.push(Block1d {
                    conv1: b.get("conv1.weight")?,
                    bn1: BatchNorm::load(&b.pp("bn1"))?,
                    conv2: b.get("conv2.weight")?,
                    bn2: BatchNorm::load(&b.pp("bn2"))?,
                    down,
                    stride: if bi == 0 { stride } else { 1 },
                });
            }
        }
        Ok(Self { conv1: t.get("conv1.weight")?, bn1: BatchNorm::load(&t.pp("bn1"))?, blocks })
    }

    /// (S,) samples -> (S / 640, 512).
    pub fn forward(&self, wave: &Tensor) -> Result<Tensor> {
        let s = wave.dim(0)? / 640 * 640;
        let x = wave.narrow(0, 0, s)?.reshape((1, 1, s))?;
        let mut x = swish(&self.bn1.forward(&x.conv1d(&self.conv1, 38, 4, 1, 1)?, 1)?)?;
        for b in &self.blocks {
            let out = swish(&b.bn1.forward(&x.conv1d(&b.conv1, 1, b.stride, 1, 1)?, 1)?)?;
            let out = b.bn2.forward(&out.conv1d(&b.conv2, 1, 1, 1, 1)?, 1)?;
            let res = match &b.down {
                Some((k, bn)) => bn.forward(&x.conv1d(k, 0, b.stride, 1, 1)?, 1)?,
                None => x.clone(),
            };
            x = swish(&(out + res)?)?;
        }
        // AvgPool1d(20, 20): 20 frames of features per video frame
        let (_, c, l) = x.dims3()?;
        let t = l / 20;
        Ok(x.narrow(2, 0, t * 20)?.reshape((c, t, 20))?.mean(2)?.t()?.contiguous()?)
    }
}

/// Relative positional embeddings for lengths -(T-1)..(T-1): row j is position T-1-j.
pub fn rel_pos_emb(t: usize, d: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; (2 * t - 1) * d];
    for j in 0..2 * t - 1 {
        let p = (t as f32 - 1.0) - j as f32;
        for i in (0..d).step_by(2) {
            let div = (-(i as f32) * (10000f32.ln() / d as f32)).exp();
            v[j * d + i] = (p * div).sin();
            v[j * d + i + 1] = (p * div).cos();
        }
    }
    Ok(Tensor::from_vec(v, (1, 2 * t - 1, d), dev)?)
}

struct RelSelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    pos: Linear,
    bias_u: Tensor, // (H, 1, dk) for broadcasting over (B, H, T, dk)
    bias_v: Tensor,
    h: usize,
}

impl RelSelfAttention {
    fn load(w: &Prefixed, h: usize) -> Result<Self> {
        let bu = w.get("pos_bias_u")?;
        let dk = bu.dim(1)?;
        Ok(Self {
            q: Linear::load(&w.pp("linear_q"), true)?,
            k: Linear::load(&w.pp("linear_k"), true)?,
            v: Linear::load(&w.pp("linear_v"), true)?,
            out: Linear::load(&w.pp("linear_out"), true)?,
            pos: Linear::load(&w.pp("linear_pos"), false)?,
            bias_u: bu.reshape((h, 1, dk))?,
            bias_v: w.get("pos_bias_v")?.reshape((h, 1, dk))?,
            h,
        })
    }

    fn forward(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = heads(&self.q.forward(x)?, self.h)?;
        let k = heads(&self.k.forward(x)?, self.h)?;
        let v = heads(&self.v.forward(x)?, self.h)?;
        let p = heads(&self.pos.forward(pos)?, self.h)?; // (1, H, 2T-1, dk)
        let dk = q.dim(D::Minus1)?;
        let ac = q.broadcast_add(&self.bias_u)?.matmul(&k.t()?)?;
        let bd = q.broadcast_add(&self.bias_v)?.broadcast_matmul(&p.t()?)?; // (B, H, T, 2T-1)
        let bd = rel_shift(&bd, b, self.h, t)?;
        let scores = ((ac + bd)? / (dk as f64).sqrt())?;
        let att = crate::nn::softmax_last(&scores)?.matmul(&v)?;
        self.out.forward(&merge_heads(&att)?)
    }
}

/// Transformer-XL shift: out[.., i, j] = x[.., i, T-1-i+j].
fn rel_shift(x: &Tensor, b: usize, h: usize, t: usize) -> Result<Tensor> {
    let x = x.pad_with_zeros(3, 1, 0)?; // (B, H, T, 2T)
    let x = x.reshape((b, h, 2 * t, t))?.narrow(2, 1, 2 * t - 1)?.contiguous()?;
    Ok(x.reshape((b, h, t, 2 * t - 1))?.narrow(3, 0, t)?)
}

struct ConvModule {
    pw1: Linear,
    dw: Tensor, // (C, K)
    dw_b: Tensor,
    bn: BatchNorm,
    pw2: Linear,
}

impl ConvModule {
    fn load(w: &Prefixed) -> Result<Self> {
        Ok(Self {
            pw1: Linear::load(&w.pp("pointwise_cov1"), true)?,
            dw: w.get("depthwise_conv.weight")?.squeeze(1)?,
            dw_b: w.get("depthwise_conv.bias")?,
            bn: BatchNorm::load(&w.pp("norm"))?,
            pw2: Linear::load(&w.pp("pointwise_cov2"), true)?,
        })
    }

    /// x: (B, T, C), kept time-major throughout (conv1d k=1 == linear).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, t, c) = x.dims3()?;
        let y = self.pw1.forward(x)?;
        let a = y.narrow(2, 0, c)?;
        let g = y.narrow(2, c, c)?;
        let y = (a * candle_nn::ops::sigmoid(&g)?)?;
        let k = self.dw.dim(1)?;
        let yp = y.pad_with_zeros(1, k / 2, k / 2)?;
        let mut acc = self.dw_b.reshape((1, 1, c))?.broadcast_as(y.shape())?.contiguous()?;
        for j in 0..k {
            let wj = self.dw.narrow(1, j, 1)?.reshape((1, 1, c))?;
            acc = (acc + yp.narrow(1, j, t)?.broadcast_mul(&wj)?)?;
        }
        let y = swish(&self.bn.forward(&acc, 2)?)?;
        self.pw2.forward(&y)
    }
}

struct ConformerLayer {
    ff_mac: FeedForward,
    norm_ff_mac: LayerNorm,
    att: RelSelfAttention,
    norm_mha: LayerNorm,
    conv: ConvModule,
    norm_conv: LayerNorm,
    ff: FeedForward,
    norm_ff: LayerNorm,
    norm_final: LayerNorm,
}

impl ConformerLayer {
    fn load(w: &Prefixed, h: usize) -> Result<Self> {
        let ln = |n: &str| LayerNorm::load(&w.pp(n), 1e-12);
        Ok(Self {
            ff_mac: FeedForward::load(&w.pp("feed_forward_macaron"))?,
            norm_ff_mac: ln("norm_ff_macaron")?,
            att: RelSelfAttention::load(&w.pp("self_attn"), h)?,
            norm_mha: ln("norm_mha")?,
            conv: ConvModule::load(&w.pp("conv_module"))?,
            norm_conv: ln("norm_conv")?,
            ff: FeedForward::load(&w.pp("feed_forward"))?,
            norm_ff: ln("norm_ff")?,
            norm_final: ln("norm_final")?,
        })
    }

    fn forward(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let x = (x + (self.ff_mac.forward(&self.norm_ff_mac.forward(x)?)? * 0.5)?)?;
        let x = (&x + self.att.forward(&self.norm_mha.forward(&x)?, pos)?)?;
        let x = (&x + self.conv.forward(&self.norm_conv.forward(&x)?)?)?;
        let x = (&x + (self.ff.forward(&self.norm_ff.forward(&x)?)? * 0.5)?)?;
        self.norm_final.forward(&x)
    }
}

/// The modality-specific front end: mouth crops (3D conv + ResNet-18) or 16 kHz audio (ResNet-1D).
pub enum Front {
    Video(Frontend),
    Audio(AudioFrontend),
}

pub struct Encoder {
    pub frontend: Front,
    embed: Linear,
    layers: Vec<ConformerLayer>,
    after_norm: LayerNorm,
    d: usize,
}

impl Encoder {
    pub fn load(w: &Prefixed, n_layers: usize, heads: usize) -> Result<Self> {
        let embed = Linear::load(&w.pp("embed.0"), true)?;
        let layers = (0..n_layers)
            .map(|i| ConformerLayer::load(&w.pp(&format!("encoders.{i}")), heads))
            .collect::<Result<_>>()?;
        let d = w.get("after_norm.weight")?.dim(0)?;
        let fw = w.pp("frontend");
        let frontend = if fw.get("trunk.conv1.weight").is_ok() { Front::Audio(AudioFrontend::load(&fw)?) } else { Front::Video(Frontend::load(&fw)?) };
        Ok(Self { frontend, embed, layers, after_norm: LayerNorm::load(&w.pp("after_norm"), 1e-12)?, d })
    }

    pub fn video_frontend(&self) -> Option<&Frontend> {
        match &self.frontend {
            Front::Video(f) => Some(f),
            Front::Audio(_) => None,
        }
    }

    /// Video: (T, 88, 88) normalised frames; audio: (S,) normalised 16 kHz samples. -> (T, D).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feats = match &self.frontend {
            Front::Video(f) => f.forward(x)?,
            Front::Audio(f) => f.forward(x)?,
        };
        self.forward_features(&feats)
    }

    /// (T, 512) front-end features -> (T, D).
    pub fn forward_features(&self, feats: &Tensor) -> Result<Tensor> {
        let t = feats.dim(0)?;
        let mut h = (self.embed.forward(&feats.unsqueeze(0)?)? * (self.d as f64).sqrt())?;
        let pos = rel_pos_emb(t, self.d, h.device())?;
        for l in &self.layers {
            h = l.forward(&h, &pos)?;
        }
        Ok(self.after_norm.forward(&h)?.squeeze(0)?)
    }
}
