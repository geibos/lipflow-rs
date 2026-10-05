//! Autoregressive scorers for beam search: the attention decoder and the Transformer LM.
//!
//! Both score one new token for a batch of hypotheses at a time. ESPnet caches each layer's
//! *outputs* and re-projects keys/values of the whole prefix every step; here the projected
//! keys/values themselves are cached, which is the same arithmetic done once per position.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::nn::{FeedForward, LayerNorm, Linear, Prefixed, attention, attention_masked, heads, merge_heads};

/// Per-layer self-attention keys/values, each (N, H, L, dk). Empty before the first step.
#[derive(Clone, Default)]
pub struct KvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl KvCache {
    /// Keep the hypotheses at `idx` (u32 indices into the batch), in that order.
    pub fn select(&self, idx: &Tensor) -> Result<Self> {
        let pick = |xs: &[Tensor]| xs.iter().map(|t| t.index_select(idx, 0)).collect::<candle_core::Result<Vec<_>>>();
        Ok(Self { k: pick(&self.k)?, v: pick(&self.v)? })
    }

    fn append(&mut self, layer: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        if self.k.len() == layer {
            self.k.push(k);
            self.v.push(v);
        } else {
            self.k[layer] = Tensor::cat(&[&self.k[layer], &k], 2)?;
            self.v[layer] = Tensor::cat(&[&self.v[layer], &v], 2)?;
        }
        Ok((self.k[layer].clone(), self.v[layer].clone()))
    }
}

struct Mha {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Mha {
    fn load(w: &Prefixed) -> Result<Self> {
        Ok(Self {
            q: Linear::load(&w.pp("linear_q"), true)?,
            k: Linear::load(&w.pp("linear_k"), true)?,
            v: Linear::load(&w.pp("linear_v"), true)?,
            out: Linear::load(&w.pp("linear_out"), true)?,
        })
    }

    /// Self-attention of the newest position (x: (N, 1, D)) over the cached prefix.
    fn step(&self, x: &Tensor, h: usize, cache: &mut KvCache, layer: usize) -> Result<Tensor> {
        let q = heads(&self.q.forward(x)?, h)?;
        let (k, v) = cache.append(layer, heads(&self.k.forward(x)?, h)?, heads(&self.v.forward(x)?, h)?)?;
        self.out.forward(&merge_heads(&attention(&q, &k, &v)?)?)
    }
}

fn sinusoid(pos: usize, d: usize, dev: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; d];
    for i in (0..d).step_by(2) {
        let div = (-(i as f32) * (10000f32.ln() / d as f32)).exp();
        v[i] = (pos as f32 * div).sin();
        v[i + 1] = (pos as f32 * div).cos();
    }
    Ok(Tensor::from_vec(v, (1, 1, d), dev)?)
}

fn ids(tokens: &[u32], dev: &Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(tokens, tokens.len(), dev)?)
}

struct DecoderLayer {
    self_attn: Mha,
    src_q: Linear,
    src_k: Linear,
    src_v: Linear,
    src_out: Linear,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

/// Cross-attention keys/values of the encoder output, one (1, H, T, dk) pair per layer.
pub struct Memory {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

pub struct Decoder {
    embed: Tensor, // (V, D)
    layers: Vec<DecoderLayer>,
    after_norm: LayerNorm,
    output: Linear,
    h: usize,
    d: usize,
}

impl Decoder {
    pub fn load(w: &Prefixed, n_layers: usize, h: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let l = w.pp(&format!("decoders.{i}"));
            let s = l.pp("src_attn");
            layers.push(DecoderLayer {
                self_attn: Mha::load(&l.pp("self_attn"))?,
                src_q: Linear::load(&s.pp("linear_q"), true)?,
                src_k: Linear::load(&s.pp("linear_k"), true)?,
                src_v: Linear::load(&s.pp("linear_v"), true)?,
                src_out: Linear::load(&s.pp("linear_out"), true)?,
                ff: FeedForward::load(&l.pp("feed_forward"))?,
                norm1: LayerNorm::load(&l.pp("norm1"), 1e-12)?,
                norm2: LayerNorm::load(&l.pp("norm2"), 1e-12)?,
                norm3: LayerNorm::load(&l.pp("norm3"), 1e-12)?,
            });
        }
        let embed = w.get("embed.0.weight")?;
        let d = embed.dim(1)?;
        Ok(Self {
            embed,
            layers,
            after_norm: LayerNorm::load(&w.pp("after_norm"), 1e-12)?,
            output: Linear::load(&w.pp("output_layer"), true)?,
            h,
            d,
        })
    }

    /// Project the encoder output (T, D) once per utterance.
    pub fn memory(&self, enc: &Tensor) -> Result<Memory> {
        let x = enc.unsqueeze(0)?;
        let mut m = Memory { k: Vec::new(), v: Vec::new() };
        for l in &self.layers {
            m.k.push(heads(&l.src_k.forward(&x)?, self.h)?);
            m.v.push(heads(&l.src_v.forward(&x)?, self.h)?);
        }
        Ok(m)
    }

    /// Teacher-forced pass over a whole target prefix: logits (L, V) for `ys_in` (starting with
    /// <sos>) attending to the encoder output (T, D). Differentiable w.r.t. `enc`.
    pub fn forward_train(&self, ys_in: &[u32], enc: &Tensor) -> Result<Tensor> {
        let dev = enc.device();
        let l = ys_in.len();
        let pe: Vec<Tensor> = (0..l).map(|p| sinusoid(p, self.d, dev)).collect::<Result<_>>()?;
        let pe = Tensor::cat(&pe, 1)?; // (1, L, D)
        let e = self.embed.to_device(dev)?.index_select(&ids(ys_in, dev)?, 0)?.reshape((1, l, self.d))?;
        let mut x = (e * (self.d as f64).sqrt())?.broadcast_add(&pe)?;
        let mut m = vec![0f32; l * l];
        for i in 0..l {
            for j in i + 1..l {
                m[i * l + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(m, (1, 1, l, l), dev)?;
        let mem = enc.unsqueeze(0)?;
        for layer in &self.layers {
            let h = layer.norm1.forward(&x)?;
            let sa = &layer.self_attn;
            let (q, k, v) = (heads(&sa.q.forward(&h)?, self.h)?, heads(&sa.k.forward(&h)?, self.h)?, heads(&sa.v.forward(&h)?, self.h)?);
            x = (&x + sa.out.forward(&merge_heads(&attention_masked(&q, &k, &v, Some(&mask))?)?)?)?;
            let h = layer.norm2.forward(&x)?;
            let q = heads(&layer.src_q.forward(&h)?, self.h)?;
            let (k, v) = (heads(&layer.src_k.forward(&mem)?, self.h)?, heads(&layer.src_v.forward(&mem)?, self.h)?);
            x = (&x + layer.src_out.forward(&merge_heads(&attention(&q, &k, &v)?)?)?)?;
            x = (&x + layer.ff.forward(&layer.norm3.forward(&x)?)?)?;
        }
        self.output.forward(&self.after_norm.forward(&x.squeeze(0)?)?)
    }

    /// Log-probabilities (N, V) of the token after `last` (the newest token of each
    /// hypothesis, at position `pos`), extending `cache` by one position.
    pub fn step(&self, last: &[u32], pos: usize, cache: &mut KvCache, mem: &Memory) -> Result<Tensor> {
        let dev = self.embed.device();
        let n = last.len();
        let e = self.embed.index_select(&ids(last, dev)?, 0)?.reshape((n, 1, self.d))?;
        let mut x = (e * (self.d as f64).sqrt())?.broadcast_add(&sinusoid(pos, self.d, dev)?)?;
        for (i, l) in self.layers.iter().enumerate() {
            x = (&x + l.self_attn.step(&l.norm1.forward(&x)?, self.h, cache, i)?)?;
            // Every hypothesis attends to the same memory: stack them as query rows.
            let q = heads(&l.src_q.forward(&l.norm2.forward(&x)?.reshape((1, n, self.d))?)?, self.h)?;
            let att = merge_heads(&attention(&q, &mem.k[i], &mem.v[i])?)?.reshape((n, 1, self.d))?;
            x = (&x + l.src_out.forward(&att)?)?;
            x = (&x + l.ff.forward(&l.norm3.forward(&x)?)?)?;
        }
        let y = self.output.forward(&self.after_norm.forward(&x.squeeze(1)?)?)?;
        Ok(candle_nn::ops::log_softmax(&y, 1)?)
    }
}

struct LmLayer {
    att: Mha,
    norm_mha: LayerNorm,
    ff: FeedForward,
    norm_ff: LayerNorm,
}

/// ESPnet `TransformerLM`: embedding -> Linear+LayerNorm+ReLU -> pre-norm Transformer layers
/// with causal attention and no positional encoding -> Linear.
pub struct Lm {
    embed: Tensor,
    proj: Linear,
    proj_norm: LayerNorm,
    layers: Vec<LmLayer>,
    after_norm: LayerNorm,
    out: Linear,
    h: usize,
}

impl Lm {
    pub fn load(w: &Prefixed, n_layers: usize, h: usize) -> Result<Self> {
        let layers = (0..n_layers)
            .map(|i| {
                let l = w.pp(&format!("encoder.encoders.{i}"));
                Ok(LmLayer {
                    att: Mha::load(&l.pp("self_attn"))?,
                    norm_mha: LayerNorm::load(&l.pp("norm_mha"), 1e-12)?,
                    ff: FeedForward::load(&l.pp("feed_forward"))?,
                    norm_ff: LayerNorm::load(&l.pp("norm_ff"), 1e-12)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            embed: w.get("embed.weight")?,
            proj: Linear::load(&w.pp("encoder.embed.0"), true)?,
            proj_norm: LayerNorm::load(&w.pp("encoder.embed.1"), 1e-5)?,
            layers,
            after_norm: LayerNorm::load(&w.pp("encoder.after_norm"), 1e-12)?,
            out: Linear::load(&w.pp("decoder"), true)?,
            h,
        })
    }

    pub fn step(&self, last: &[u32], cache: &mut KvCache) -> Result<Tensor> {
        let dev = self.embed.device();
        let n = last.len();
        let e = self.embed.index_select(&ids(last, dev)?, 0)?;
        let mut x = self.proj_norm.forward(&self.proj.forward(&e)?)?.relu()?.unsqueeze(1)?;
        for (i, l) in self.layers.iter().enumerate() {
            x = (&x + l.att.step(&l.norm_mha.forward(&x)?, self.h, cache, i)?)?;
            x = (&x + l.ff.forward(&l.norm_ff.forward(&x)?)?)?;
        }
        let y = self.out.forward(&self.after_norm.forward(&x.reshape((n, ()))?)?)?;
        Ok(candle_nn::ops::log_softmax(&y.to_dtype(DType::F32)?, 1)?)
    }
}
