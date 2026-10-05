//! MultiVSR (Prajwal, Hegde & Zisserman 2025, github.com/Sindhu-Hegde/multivsr): lip reading in
//! 13 languages, Russian among them, from 96×96 RGB face crops at 25 fps.
//!
//! VTP front end (3D CNN, two linear-attention transformers, attention pooling: one 512-d vector
//! per frame), then a 12+12-layer Transformer in the "Annotated Transformer" style (pre-norm,
//! LayerNorm with the unbiased std and eps outside the root), Whisper's multilingual tokens, and
//! the Joey NMT beam search the reference uses — reproduced exactly, finished hypotheses staying
//! in the beam included.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, IndexOp, Tensor};

use crate::nn::{Linear, Prefixed, Weights, attention, heads, merge_heads, softmax_last};

pub const SOT: u32 = 50258;
pub const EOT: u32 = 50257;
/// `<|ru|>`; Whisper's language tokens follow <|startoftranscript|> in this order.
pub const RU: u32 = 50263;
pub const FIRST_SPECIAL: u32 = 50257;

// ---------------------------------------------------------------- building blocks

/// Conv3d + BatchNorm3d folded into one conv; ReLU and the optional residual are applied by
/// the caller.
struct ConvBn {
    w: Tensor, // (out, in, kt, kh, kw)
    b: Tensor,
}

impl ConvBn {
    fn load(w: &Prefixed) -> Result<Self> {
        let conv = w.pp("conv_block.0");
        let bn = w.pp("conv_block.1");
        let (weight, bias) = (conv.get("weight")?, conv.get("bias")?);
        let scale = bn.get("weight")?.div(&(bn.get("running_var")? + 1e-5)?.sqrt()?)?;
        let shift = bn.get("bias")?.sub(&bn.get("running_mean")?.mul(&scale)?)?;
        let o = weight.dim(0)?;
        Ok(Self { w: weight.broadcast_mul(&scale.reshape((o, 1, 1, 1, 1))?)?, b: (bias.mul(&scale)? + shift)? })
    }

    /// Frames (T, C, H, W) → (T, O, Ho, Wo): the temporal kernel as a sum of 2D convolutions
    /// over shifted frames (zero-padded in time by kt/2), spatial stride `s`, padding `p`.
    fn forward(&self, x: &Tensor, s: usize, p: usize) -> Result<Tensor> {
        let (o, c, kt, kh, kw) = self.w.dims5()?;
        let t = x.dim(0)?;
        let pt = kt / 2;
        let x = if pt > 0 {
            let z = Tensor::zeros((pt, c, x.dim(2)?, x.dim(3)?), x.dtype(), x.device())?;
            Tensor::cat(&[&z, x, &z], 0)?
        } else {
            x.clone()
        };
        let mut y: Option<Tensor> = None;
        for k in 0..kt {
            let wk = self.w.i((.., .., k))?.contiguous()?.reshape((o, c, kh, kw))?;
            let xk = x.narrow(0, k, t)?;
            let yk = if xk.device().is_cpu() { crate::conv_train::conv2d(&xk, &wk, s, p)? } else { xk.contiguous()?.conv2d(&wk, p, s, 1, 1)? };
            y = Some(match y {
                Some(acc) => (acc + yk)?,
                None => yk,
            });
        }
        let y = y.context("empty kernel")?;
        Ok(y.broadcast_add(&self.b.reshape((1, o, 1, 1))?)?)
    }
}

/// `nn.LayerNorm` (biased variance, eps inside the root).
struct Ln {
    w: Tensor,
    b: Tensor,
}

impl Ln {
    fn load(w: &Prefixed) -> Result<Self> {
        Ok(Self { w: w.get("weight")?, b: w.get("bias")? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::layer_norm(&x.contiguous()?, &self.w, &self.b, 1e-5)?)
    }
}

/// The Annotated Transformer's LayerNorm: a·(x−mean)/(std + eps) + b, std unbiased, eps 1e-6.
struct AtLn {
    a: Tensor,
    b: Tensor,
}

impl AtLn {
    fn load(w: &Prefixed) -> Result<Self> {
        Ok(Self { a: w.get("a_2")?, b: w.get("b_2")? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(D::Minus1)? as f64;
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let std = (xc.sqr()?.sum_keepdim(D::Minus1)? / (n - 1.0))?.sqrt()?;
        Ok(xc.broadcast_div(&(std + 1e-6)?)?.broadcast_mul(&self.a)?.broadcast_add(&self.b)?)
    }
}

/// Linear attention (lucidrains' linear_attention_transformer, global heads only): softmax of
/// q over features, of k over positions, out = q·(kᵀv)·√dh⁻¹ per head.
///
/// Heads stay interleaved in the feature dim (no (B, N, H, dh) → (B, H, N, dh) copies, which
/// candle's Metal backend does slowly): kᵀv is formed for all features at once and the
/// cross-head blocks are masked out, ~15% more multiply-adds for one copy instead of five.
struct LinAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    h: usize,
    block_diag: Tensor, // (H·dh, H·dh): 1 inside a head's block, else 0
}

impl LinAttn {
    fn new(q: Linear, k: Linear, v: Linear, out: Linear, h: usize, dim: usize, dev: &Device) -> Result<Self> {
        let dh = dim / h;
        let m: Vec<f32> = (0..dim * dim).map(|i| f32::from(u8::from(i / dim / dh == i % dim / dh))).collect();
        Ok(Self { q, k, v, out, h, block_diag: Tensor::from_vec(m, (dim, dim), dev)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, dim) = x.dims3()?;
        let dh = dim / self.h;
        let q = self.q.forward(x)?.reshape((b, n, self.h, dh))?;
        let q = (softmax_last(&q)? * (dh as f64).powf(-0.5))?.reshape((b, n, dim))?;
        let kt = softmax_last(&self.k.forward(x)?.transpose(1, 2)?.contiguous()?)?; // (B, dim, N)
        let ctx = kt.matmul(&self.v.forward(x)?)?.broadcast_mul(&self.block_diag)?; // (B, dim, dim)
        self.out.forward(&q.matmul(&ctx)?)
    }
}

struct LatLayer {
    norm1: Ln,
    attn: LinAttn,
    norm2: Ln,
    w1: Linear,
    w2: Linear,
}

impl LatLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?)?)?;
        let h = self.w1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?;
        Ok((&x + self.w2.forward(&h)?)?)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(w: &Prefixed) -> Result<Self> {
        Ok(Self { fc1: Linear::load(&w.pp("fc1"), true)?, fc2: Linear::load(&w.pp("fc2"), true)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.relu()?)
    }
}

// ---------------------------------------------------------------- VTP front end

/// The 3D CNN on hand-written Metal kernels (channels-last, im2col + matmul + fused
/// bias/residual/ReLU): candle's Metal conv2d is ~20× slower than its matmul here.
#[cfg(feature = "metal")]
struct FastCnn {
    p: std::sync::Arc<crate::metal_ops::Pipelines>,
    w0: Tensor, // (3·5·5·5, 64)
    b0: Tensor,
    c1: crate::metal_ops::ConvNhwc,
    c2: crate::metal_ops::ConvNhwc,
}

#[cfg(feature = "metal")]
impl FastCnn {
    fn new(convs: &[ConvBn; 3]) -> Result<Self> {
        let dev = convs[0].w.device();
        let nhwc = |c: &ConvBn, stride: usize| -> Result<crate::metal_ops::ConvNhwc> {
            let (o, i, _, kh, kw) = c.w.dims5()?;
            // (o, i, 1, ky, kx) → rows (ky, kx, i), cols o
            let w = c.w.reshape((o, i, kh, kw))?.permute((2, 3, 1, 0))?.reshape((kh * kw * i, o))?.contiguous()?;
            Ok(crate::metal_ops::ConvNhwc { w, b: c.b.clone(), k: kh, stride, pad: 1 })
        };
        let (o, i, kt, kh, kw) = convs[0].w.dims5()?;
        Ok(Self {
            p: crate::metal_ops::Pipelines::new(dev)?,
            w0: convs[0].w.reshape((o, i * kt * kh * kw))?.t()?.contiguous()?,
            b0: convs[0].b.clone(),
            c1: nhwc(&convs[1], 2)?,
            c2: nhwc(&convs[2], 1)?,
        })
    }

    /// Faces (T, 3, 96, 96) → (T, 24, 24, 128) channels-last.
    fn forward(&self, faces: &Tensor) -> Result<Tensor> {
        use crate::metal_ops::{Act, conv_frames_c};
        let (_, c, h, w) = faces.dims4()?;
        let z = Tensor::zeros((c, 2, h, w), faces.dtype(), faces.device())?;
        let x = Tensor::cat(&[&z, &faces.permute((1, 0, 2, 3))?, &z], 1)?.contiguous()?; // (3, T+4, H, W)
        let x = conv_frames_c(&self.p, &x, &self.w0, &self.b0, (5, 5, 5), 2, 2, Act::Relu)?;
        let x = self.c1.forward_act(&self.p, &x, Act::Relu)?;
        Ok(self.c2.forward_residual_act(&self.p, &x, &x, Act::Relu)?)
    }
}

pub struct Vtp {
    #[cfg(feature = "metal")]
    fast: Option<FastCnn>,
    convs: [ConvBn; 3],
    pos: Tensor, // (576, 128): [col_embed[x], row_embed[y]] for token y·24+x
    proj: [Mlp; 2],
    blocks: [Vec<LatLayer>; 2],
    pooler: Linear,
}

impl Vtp {
    pub fn load(w: &Prefixed) -> Result<Self> {
        let fe = w.pp("feat_extrator.encoder");
        let convs = [ConvBn::load(&fe.pp("0"))?, ConvBn::load(&fe.pp("1"))?, ConvBn::load(&fe.pp("2"))?];
        let col = w.get("hwposition.col_embed.weight")?.narrow(0, 0, 24)?; // (24, 64)
        let row = w.get("hwposition.row_embed.weight")?.narrow(0, 0, 24)?;
        let colx = col.unsqueeze(0)?.broadcast_as((24, 24, 64))?; // [y][x] = col[x]
        let rowy = row.unsqueeze(1)?.broadcast_as((24, 24, 64))?; // [y][x] = row[y]
        let pos = Tensor::cat(&[&colx, &rowy], 2)?.reshape((576, 128))?.contiguous()?;
        let enc = w.pp("encoder");
        let block = |i: usize, h: usize, dim: usize| -> Result<Vec<LatLayer>> {
            (0..3)
                .map(|l| {
                    let p = enc.pp(&format!("transformer_blocks.{i}.layers.layers.{l}"));
                    let a = p.pp("0.fn");
                    Ok(LatLayer {
                        norm1: Ln::load(&p.pp("0.norm"))?,
                        attn: LinAttn::new(
                            Linear::load(&a.pp("to_q"), false)?,
                            Linear::load(&a.pp("to_k"), false)?,
                            Linear::load(&a.pp("to_v"), false)?,
                            Linear::load(&a.pp("to_out"), true)?,
                            h,
                            dim,
                            w.device(),
                        )?,
                        norm2: Ln::load(&p.pp("1.norm"))?,
                        w1: Linear::load(&p.pp("1.fn.fn.w1"), true)?,
                        w2: Linear::load(&p.pp("1.fn.fn.w2"), true)?,
                    })
                })
                .collect()
        };
        Ok(Self {
            #[cfg(feature = "metal")]
            fast: if w.device().is_metal() { Some(FastCnn::new(&convs)?) } else { None },
            convs,
            pos,
            proj: [Mlp::load(&enc.pp("patch_projectors.0.1"))?, Mlp::load(&enc.pp("patch_projectors.1.1"))?],
            blocks: [block(0, 8, 256)?, block(1, 8, 512)?],
            pooler: Linear::load(&w.pp("pooler"), true)?,
        })
    }

    /// The 3D CNN alone: faces (T, 3, 96, 96) in [0, 1] → (T, 128, 24, 24).
    pub fn cnn(&self, faces: &Tensor) -> Result<Tensor> {
        Ok(self.cnn_nhwc(faces)?.permute((0, 3, 1, 2))?)
    }

    /// (T, 24, 24, 128) channels-last.
    fn cnn_nhwc(&self, faces: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(f) = &self.fast {
            return f.forward(faces);
        }
        Ok(self.cnn_ref(faces)?.permute((0, 2, 3, 1))?)
    }

    fn cnn_ref(&self, faces: &Tensor) -> Result<Tensor> {
        let x = self.convs[0].forward(faces, 2, 2)?.relu()?;
        let x = self.convs[1].forward(&x, 2, 1)?.relu()?;
        Ok((self.convs[2].forward(&x, 1, 1)? + x)?.relu()?)
    }

    /// Faces (T, 3, 96, 96) → features (T, 512), `chunk` frames at a time (memory).
    pub fn forward(&self, faces: &Tensor, chunk: usize) -> Result<Tensor> {
        self.forward_range(faces, 0, faces.dim(0)?, chunk)
    }

    /// Features of frames `from..to` only, each exactly as in the whole-clip pass (the time
    /// kernel sees the real neighbouring frames, zeros only beyond the clip's ends).
    pub fn forward_range(&self, faces: &Tensor, from: usize, to: usize, chunk: usize) -> Result<Tensor> {
        let t = faces.dim(0)?;
        let mut outs = Vec::new();
        let mut s = from;
        while s < to {
            let n = chunk.min(to - s);
            // the time kernel reads 2 frames either side: give each chunk its real neighbours
            let lo = s.saturating_sub(2);
            let hi = (s + n + 2).min(t);
            let c = self.cnn_nhwc(&faces.narrow(0, lo, hi - lo)?)?.narrow(0, s - lo, n)?;
            outs.push(self.tokens(&c)?);
            s += n;
        }
        Ok(Tensor::cat(&outs, 0)?)
    }

    fn tokens(&self, c: &Tensor) -> Result<Tensor> {
        let n = c.dim(0)?;
        // (n, 24, 24, 128) → (n, 576, 128) + position
        let x = c.contiguous()?.reshape((n, 576, 128))?.broadcast_add(&self.pos)?;
        let mut x = self.proj[0].forward(&x)?;
        for l in &self.blocks[0] {
            x = l.forward(&x)?;
        }
        // 24×24×256 → 2×2 patches: 144 tokens of (p1, p2, c)
        let x = x.reshape((n, 12, 2, 12, 2, 256))?.permute((0, 1, 3, 2, 4, 5))?.reshape((n, 144, 1024))?;
        let mut x = self.proj[1].forward(&x)?;
        for l in &self.blocks[1] {
            x = l.forward(&x)?;
        }
        let w = candle_nn::ops::softmax(&self.pooler.forward(&x)?, 1)?; // (n, 144, 1)
        Ok(x.broadcast_mul(&w)?.sum(1)?)
    }
}

// ---------------------------------------------------------------- Transformer

struct Mha {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Mha {
    fn load(w: &Prefixed) -> Result<Self> {
        let l = |i: usize| Linear::load(&w.pp(&format!("linears.{i}")), true);
        Ok(Self { q: l(0)?, k: l(1)?, v: l(2)?, out: l(3)? })
    }
}

struct EncLayer {
    attn: Mha,
    ff1: Linear,
    ff2: Linear,
    n1: AtLn,
    n2: AtLn,
}

struct DecLayer {
    self_attn: Mha,
    src: Mha,
    ff1: Linear,
    ff2: Linear,
    n1: AtLn,
    n2: AtLn,
    n3: AtLn,
}

/// Per-layer projected keys/values of the encoder output, (1, H, T, dk).
pub struct Memory {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

/// Per-layer self-attention keys/values of the hypotheses so far, (N, H, L, dk).
#[derive(Clone, Default)]
pub struct Cache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl Cache {
    fn select(&self, idx: &Tensor) -> Result<Self> {
        let pick = |xs: &[Tensor]| xs.iter().map(|t| t.index_select(idx, 0)).collect::<candle_core::Result<Vec<_>>>();
        Ok(Self { k: pick(&self.k)?, v: pick(&self.v)? })
    }
}

pub struct Seq2Seq {
    src_embed: Linear,
    enc: Vec<EncLayer>,
    enc_norm: AtLn,
    embed: Tensor, // (V, D)
    dec: Vec<DecLayer>,
    dec_norm: AtLn,
    proj: Linear,
    pe: Tensor, // (max_len, D)
    h: usize,
    d: usize,
}

fn positions(max_len: usize, d: usize, dev: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; max_len * d];
    for p in 0..max_len {
        for i in (0..d).step_by(2) {
            let div = (i as f32 * -(10000f32.ln() / d as f32)).exp();
            v[p * d + i] = (p as f32 * div).sin();
            v[p * d + i + 1] = (p as f32 * div).cos();
        }
    }
    Ok(Tensor::from_vec(v, (max_len, d), dev)?)
}

impl Seq2Seq {
    pub fn load(w: &Prefixed) -> Result<Self> {
        let (n_layers, h) = (12, 12);
        let ff = |p: &Prefixed| -> Result<(Linear, Linear)> { Ok((Linear::load(&p.pp("feed_forward.w_1"), true)?, Linear::load(&p.pp("feed_forward.w_2"), true)?)) };
        let enc = (0..n_layers)
            .map(|i| {
                let p = w.pp(&format!("encoder.layers.{i}"));
                let (ff1, ff2) = ff(&p)?;
                Ok(EncLayer { attn: Mha::load(&p.pp("self_attn"))?, ff1, ff2, n1: AtLn::load(&p.pp("sublayer.0.norm"))?, n2: AtLn::load(&p.pp("sublayer.1.norm"))? })
            })
            .collect::<Result<Vec<_>>>()?;
        let dec = (0..n_layers)
            .map(|i| {
                let p = w.pp(&format!("decoder.layers.{i}"));
                let (ff1, ff2) = ff(&p)?;
                Ok(DecLayer {
                    self_attn: Mha::load(&p.pp("self_attn"))?,
                    src: Mha::load(&p.pp("src_attn"))?,
                    ff1,
                    ff2,
                    n1: AtLn::load(&p.pp("sublayer.0.norm"))?,
                    n2: AtLn::load(&p.pp("sublayer.1.norm"))?,
                    n3: AtLn::load(&p.pp("sublayer.2.norm"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embed = w.get("tgt_embed.0.lut.weight")?;
        let d = embed.dim(1)?;
        Ok(Self {
            src_embed: Linear::load(&w.pp("src_embed.0"), true)?,
            enc,
            enc_norm: AtLn::load(&w.pp("encoder.norm"))?,
            embed,
            dec,
            dec_norm: AtLn::load(&w.pp("decoder.norm"))?,
            proj: Linear::load(&w.pp("generator.proj"), true)?,
            pe: positions(5000, d, w.device())?,
            h,
            d,
        })
    }

    /// Features (T, 512) → encoder output (T, 768).
    pub fn encode(&self, feats: &Tensor) -> Result<Tensor> {
        let t = feats.dim(0)?;
        let mut x = self.src_embed.forward(&feats.unsqueeze(0)?)?.broadcast_add(&self.pe.narrow(0, 0, t)?)?;
        for l in &self.enc {
            let y = l.n1.forward(&x)?;
            let (q, k, v) = (heads(&l.attn.q.forward(&y)?, self.h)?, heads(&l.attn.k.forward(&y)?, self.h)?, heads(&l.attn.v.forward(&y)?, self.h)?);
            x = (&x + l.attn.out.forward(&merge_heads(&attention(&q, &k, &v)?)?)?)?;
            x = (&x + l.ff2.forward(&l.ff1.forward(&l.n2.forward(&x)?)?.relu()?)?)?;
        }
        self.enc_norm.forward(&x.squeeze(0)?)
    }

    pub fn memory(&self, enc: &Tensor) -> Result<Memory> {
        let x = enc.unsqueeze(0)?;
        let mut m = Memory { k: Vec::new(), v: Vec::new() };
        for l in &self.dec {
            m.k.push(heads(&l.src.k.forward(&x)?, self.h)?);
            m.v.push(heads(&l.src.v.forward(&x)?, self.h)?);
        }
        Ok(m)
    }

    /// Logits (N, V) for the token after `last` (each hypothesis' newest token, at `pos`).
    pub fn step(&self, last: &[u32], pos: usize, cache: &mut Cache, mem: &Memory) -> Result<Tensor> {
        let dev = self.embed.device();
        let n = last.len();
        let ids = Tensor::from_slice(last, n, dev)?;
        let e = self.embed.index_select(&ids, 0)?.reshape((n, 1, self.d))?;
        let mut x = (e * (self.d as f64).sqrt())?.broadcast_add(&self.pe.narrow(0, pos, 1)?)?;
        for (i, l) in self.dec.iter().enumerate() {
            let y = l.n1.forward(&x)?;
            let q = heads(&l.self_attn.q.forward(&y)?, self.h)?;
            let (k, v) = (heads(&l.self_attn.k.forward(&y)?, self.h)?, heads(&l.self_attn.v.forward(&y)?, self.h)?);
            let (k, v) = if cache.k.len() == i {
                cache.k.push(k);
                cache.v.push(v);
                (cache.k[i].clone(), cache.v[i].clone())
            } else {
                cache.k[i] = Tensor::cat(&[&cache.k[i], &k], 2)?;
                cache.v[i] = Tensor::cat(&[&cache.v[i], &v], 2)?;
                (cache.k[i].clone(), cache.v[i].clone())
            };
            x = (&x + l.self_attn.out.forward(&merge_heads(&attention(&q, &k, &v)?)?)?)?;
            // every hypothesis attends to the same memory: stack them as query rows
            let q = heads(&l.src.q.forward(&l.n2.forward(&x)?.reshape((1, n, self.d))?)?, self.h)?;
            let att = merge_heads(&attention(&q, &mem.k[i], &mem.v[i])?)?.reshape((n, 1, self.d))?;
            x = (&x + l.src.out.forward(&att)?)?;
            x = (&x + l.ff2.forward(&l.ff1.forward(&l.n3.forward(&x)?)?.relu()?)?)?;
        }
        self.proj.forward(&self.dec_norm.forward(&x.squeeze(1)?)?)
    }

    /// Feed a fixed prefix (e.g. <|startoftranscript|><|ru|>) and return the logits after it.
    fn prefix(&self, prefix: &[u32], cache: &mut Cache, mem: &Memory) -> Result<Tensor> {
        let mut logits = None;
        for (pos, &tok) in prefix.iter().enumerate() {
            logits = Some(self.step(&[tok], pos, cache, mem)?);
        }
        logits.context("empty prefix")
    }

    pub fn greedy(&self, mem: &Memory, prefix: &[u32], max_len: usize) -> Result<Vec<u32>> {
        let mut cache = Cache::default();
        let mut out = prefix.to_vec();
        let mut logits = self.prefix(prefix, &mut cache, mem)?;
        for _ in 0..max_len {
            let next = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
            out.push(next);
            if next == EOT {
                break;
            }
            logits = self.step(&[next], out.len() - 1, &mut cache, mem)?;
        }
        Ok(out)
    }

    /// Joey NMT beam search (alpha 1, n_best 1) as in the reference `search.py`: returns the
    /// best hypothesis after the prefix's first token, and its length-normalised score.
    pub fn beam_search(&self, mem: &Memory, prefix: &[u32], size: usize, max_len: usize) -> Result<(Vec<u32>, f32)> {
        self.beam_search_n(mem, prefix, size, max_len)?.into_iter().next().context("beam search found nothing")
    }

    /// All finished hypotheses of the beam search, best first.
    pub fn beam_search_n(&self, mem: &Memory, prefix: &[u32], size: usize, max_len: usize) -> Result<Vec<(Vec<u32>, f32)>> {
        let dev = self.embed.device();
        let mut cache = Cache::default();
        let logits0 = self.prefix(prefix, &mut cache, mem)?;
        // all beams start identical: replicate the single-hypothesis cache
        let zeros = Tensor::zeros(size, DType::U32, dev)?;
        cache = cache.select(&zeros)?;
        let mut seqs: Vec<Vec<u32>> = vec![prefix.to_vec(); size];
        let mut topk_logp: Vec<f32> = (0..size).map(|i| if i == 0 { 0.0 } else { f32::NEG_INFINITY }).collect();
        let mut hyps: Vec<(f32, Vec<u32>)> = Vec::new();
        let mut logits = logits0.broadcast_as((size, logits0.dim(1)?))?.contiguous()?;
        for step in 0..max_len {
            let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?.to_vec2::<f32>()?;
            let v = logp[0].len();
            let lp = ((5.0 + (step + 1) as f32) / 6.0).powf(1.0);
            // top `size` over all (beam, token) of (logp + beam score) / lp
            let mut best: Vec<(f32, usize)> = Vec::with_capacity(size + 1);
            for (b, row) in logp.iter().enumerate() {
                let base = topk_logp[b];
                if base == f32::NEG_INFINITY {
                    continue;
                }
                for (tok, &l) in row.iter().enumerate() {
                    let s = (l + base) / lp;
                    if best.len() < size || s > best[best.len() - 1].0 {
                        let at = best.partition_point(|&(x, _)| x >= s);
                        best.insert(at, (s, b * v + tok));
                        best.truncate(size);
                    }
                }
            }
            if best.len() < size {
                bail!("beam search: fewer than {size} candidates");
            }
            let parents: Vec<u32> = best.iter().map(|&(_, i)| (i / v) as u32).collect();
            let tokens: Vec<u32> = best.iter().map(|&(_, i)| (i % v) as u32).collect();
            topk_logp = best.iter().map(|&(s, _)| s * lp).collect();
            seqs = parents.iter().zip(&tokens).map(|(&p, &t)| {
                let mut s = seqs[p as usize].clone();
                s.push(t);
                s
            }).collect();
            let finished: Vec<bool> = tokens.iter().map(|&t| t == EOT || step + 1 == max_len).collect();
            if finished.iter().any(|&f| f) {
                let end = finished[0];
                for (j, &fin) in finished.iter().enumerate() {
                    if fin || end {
                        // the reference keeps finished hypotheses alive; only one EOS counts
                        let body = &seqs[j][1..];
                        if body.iter().filter(|&&t| t == EOT).count() < 2 {
                            hyps.push((best[j].0, body.to_vec()));
                        }
                    }
                }
                if end {
                    break;
                }
            }
            let idx = Tensor::from_slice(&parents, size, dev)?;
            cache = cache.select(&idx)?;
            logits = self.step(&tokens, seqs[0].len() - 1, &mut cache, mem)?;
        }
        hyps.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(hyps.into_iter().map(|(score, seq)| (seq, score)).collect())
    }
}

// ---------------------------------------------------------------- tokens

/// Whisper's byte-level BPE vocabulary, for decoding only.
pub struct WhisperTokens {
    pieces: Vec<Vec<u8>>, // id → bytes; empty for special tokens
}

impl WhisperTokens {
    /// `vocab.json` of the multilingual tokenizer (ids < 50257; the rest are special).
    pub fn load(vocab_json: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(vocab_json).with_context(|| format!("reading {}", vocab_json.display()))?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&text)?;
        let unicode_to_byte = byte_decoder();
        let mut pieces = vec![Vec::new(); FIRST_SPECIAL as usize];
        for (piece, id) in vocab {
            if (id as usize) < pieces.len() {
                pieces[id as usize] = piece.chars().filter_map(|c| unicode_to_byte.get(&c).copied()).collect();
            }
        }
        Ok(Self { pieces })
    }

    /// Text of the tokens, special tokens dropped.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&i| i < FIRST_SPECIAL).flat_map(|&i| self.pieces[i as usize].iter().copied()).collect();
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

/// GPT-2's printable-unicode stand-ins for bytes, reversed.
fn byte_decoder() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).map(|(b, c)| (char::from_u32(c).unwrap_or('?'), b as u8)).collect()
}

// ---------------------------------------------------------------- weights from the release

/// The released checkpoints (`model.pth`: the Transformer, `feature_extractor.pth`: VTP, inside
/// an older English model; both with optimizer state) → the one file the app loads, keeping only
/// what the model uses, named `s2s.*` and `vtp.*`. Pure Rust: setup needs no Python.
pub fn convert_checkpoints(model_pth: &std::path::Path, vtp_pth: &std::path::Path, out: &std::path::Path) -> Result<usize> {
    let mut keep: HashMap<String, Tensor> = HashMap::new();
    for (name, t) in candle_core::pickle::read_all_with_key(model_pth, Some("state_dict")).with_context(|| format!("reading {}", model_pth.display()))? {
        let name = name.strip_prefix("module.").unwrap_or(&name);
        if t.dtype() == DType::F32 && !name.ends_with(".pe") {
            keep.insert(format!("s2s.{name}"), t);
        }
    }
    for (name, t) in candle_core::pickle::read_all_with_key(vtp_pth, Some("state_dict")).with_context(|| format!("reading {}", vtp_pth.display()))? {
        if let Some(rest) = name.strip_prefix("module.face_encoder.")
            && t.dtype() == DType::F32
        {
            keep.insert(format!("vtp.{rest}"), t);
        }
    }
    if !keep.contains_key("s2s.generator.proj.weight") || !keep.contains_key("vtp.pooler.weight") {
        bail!("these don't look like the MultiVSR checkpoints");
    }
    let n = keep.len();
    candle_core::safetensors::save(&keep, out)?;
    Ok(n)
}

// ---------------------------------------------------------------- the whole model

pub struct MultiVsr {
    pub vtp: Vtp,
    pub s2s: Seq2Seq,
    pub tokens: WhisperTokens,
    pub device: Device,
}

impl MultiVsr {
    /// `dir` holds multivsr.safetensors and vocab.json.
    pub fn load(dir: &std::path::Path, device: &Device) -> Result<Self> {
        Self::load_personal(dir, None, device)
    }

    /// With your fine-tuned tensors (a .safetensors of the same names) laid over the base.
    pub fn load_personal(dir: &std::path::Path, personal: Option<&std::path::Path>, device: &Device) -> Result<Self> {
        let mut w = Weights::from_safetensors(&dir.join("multivsr.safetensors"), device, "")?;
        if let Some(p) = personal.filter(|p| p.exists()) {
            w.overlay_file(p)?;
        }
        Ok(Self { vtp: Vtp::load(&w.pp("vtp"))?, s2s: Seq2Seq::load(&w.pp("s2s"))?, tokens: WhisperTokens::load(&dir.join("vocab.json"))?, device: device.clone() })
    }

    /// 96×96 RGB crops (T·96·96·3 bytes, HWC) → encoder output (T, 768).
    pub fn encode_frames(&self, frames: &[u8], t: usize) -> Result<Tensor> {
        let feats = self.features(frames, t, 0, t)?;
        self.s2s.encode(&feats)
    }

    /// VTP features (to − from, 512) of frames `from..to` of a clip of 96×96 RGB crops
    /// (T·96·96·3 bytes, HWC), the clip's other frames serving as context: with the features of
    /// `..from` kept from earlier calls, a clip can be read while it is still being recorded.
    pub fn features(&self, frames: &[u8], t: usize, from: usize, to: usize) -> Result<Tensor> {
        if frames.len() != t * 96 * 96 * 3 {
            bail!("{} bytes of faces, expected {t}×96×96×3", frames.len());
        }
        if from >= to || to > t {
            bail!("bad frame range {from}..{to} of {t}");
        }
        // upload only what the range needs: two frames of context either side
        let (lo, hi) = (from.saturating_sub(2), (to + 2).min(t));
        let px = 96 * 96 * 3;
        let x = Tensor::from_slice(&frames[lo * px..hi * px], (hi - lo, 96, 96, 3), &self.device)?.to_dtype(DType::F32)?;
        let x = (x.permute((0, 3, 1, 2))?.contiguous()? / 255.0)?;
        self.vtp.forward_range(&x, from - lo, to - lo, 64)
    }

    /// Features (T, 512) → encoder output (T, 768).
    pub fn encode_features(&self, feats: &Tensor) -> Result<Tensor> {
        self.s2s.encode(feats)
    }

    /// Faces (T, 3, 96, 96) in [0, 1] → encoder output (T, 768).
    pub fn encode(&self, faces: &Tensor) -> Result<Tensor> {
        let feats = self.vtp.forward(&faces.to_device(&self.device)?, 64)?;
        self.s2s.encode(&feats)
    }

    /// Russian text of the encoded clip.
    pub fn read(&self, enc: &Tensor, beam: usize) -> Result<String> {
        Ok(self.read_n(enc, beam)?.into_iter().next().unwrap_or_default())
    }

    /// Distinct Russian readings of the encoded clip, best first (one for greedy, `beam <= 1`).
    pub fn read_n(&self, enc: &Tensor, beam: usize) -> Result<Vec<String>> {
        let mem = self.s2s.memory(enc)?;
        let prefix = [SOT, RU];
        if beam <= 1 {
            return Ok(vec![self.tokens.decode(&self.s2s.greedy(&mem, &prefix, 100)?)]);
        }
        let mut out: Vec<String> = Vec::new();
        for (ids, _) in self.s2s.beam_search_n(&mem, &prefix, beam, 100)? {
            let t = self.tokens.decode(&ids);
            if !t.is_empty() && !out.contains(&t) {
                out.push(t);
            }
        }
        Ok(out)
    }

    /// Run the whole model once on a short dummy clip (compiles the Metal pipelines).
    pub fn warmup(&self) -> Result<()> {
        let enc = self.encode_frames(&vec![128u8; 8 * 96 * 96 * 3], 8)?;
        self.read_n(&enc, 1)?;
        Ok(())
    }
}
