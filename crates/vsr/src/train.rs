//! Fine-tuning the lip reader on one person's face (`lipflow/train_vsr.py`).
//!
//! Only the visual front end, the input projection and the first Conformer layer learn
//! ("frontend+encoder1"); the rest keeps what it learned from thousands of hours of speech.
//! Loss: 0.1·CTC + 0.9·label-smoothed attention, per target token. Batch-norm statistics stay
//! frozen (a few clips would wreck them).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::backprop::GradStore;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::nn::{Linear, Weights};
use crate::reader::{CROP, ROI, token_list};
use crate::spm::SentencePiece;

const MEAN: f32 = 0.421;
const STD: f32 = 0.165;
const PREFIXES: [&str; 3] = ["encoder.frontend.", "encoder.embed.", "encoder.encoders.0."];

/// Is `name` a parameter the fine-tuning updates (not a batch-norm statistic)?
pub fn is_trainable(name: &str) -> bool {
    PREFIXES.iter().any(|p| name.starts_with(p)) && !name.ends_with("running_mean") && !name.ends_with("running_var")
}

// -- CTC loss ----------------------------------------------------------------------------

#[inline]
fn lae(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// -log P(labels | logp) and its gradient w.r.t. logp (T×V), blank = 0 (Graves 2006).
/// Impossible alignments give (0, zeros), like `ctc_loss(zero_infinity=True)`.
pub fn ctc_loss_grad(logp: &[f32], t_len: usize, v: usize, labels: &[u32]) -> (f64, Vec<f32>) {
    let ext: Vec<usize> = std::iter::once(0).chain(labels.iter().flat_map(|&l| [l as usize, 0])).collect();
    let s_len = ext.len();
    let x = |t: usize, s: usize| f64::from(logp[t * v + ext[s]]);
    let skip = |s: usize| s >= 2 && ext[s] != 0 && ext[s] != ext[s - 2];
    let ninf = f64::NEG_INFINITY;
    let mut alpha = vec![ninf; t_len * s_len];
    let mut beta = vec![ninf; t_len * s_len];
    if t_len == 0 {
        return (0.0, Vec::new());
    }
    alpha[0] = x(0, 0);
    if s_len > 1 {
        alpha[1] = x(0, 1);
    }
    for t in 1..t_len {
        for s in 0..s_len {
            let mut a = alpha[(t - 1) * s_len + s];
            if s >= 1 {
                a = lae(a, alpha[(t - 1) * s_len + s - 1]);
            }
            if skip(s) {
                a = lae(a, alpha[(t - 1) * s_len + s - 2]);
            }
            alpha[t * s_len + s] = if a == ninf { ninf } else { a + x(t, s) };
        }
    }
    let last = (t_len - 1) * s_len;
    beta[last + s_len - 1] = x(t_len - 1, s_len - 1);
    if s_len > 1 {
        beta[last + s_len - 2] = x(t_len - 1, s_len - 2);
    }
    for t in (0..t_len - 1).rev() {
        for s in 0..s_len {
            let mut b = beta[(t + 1) * s_len + s];
            if s + 1 < s_len {
                b = lae(b, beta[(t + 1) * s_len + s + 1]);
            }
            if s + 2 < s_len && ext[s + 2] != 0 && ext[s + 2] != ext[s] {
                b = lae(b, beta[(t + 1) * s_len + s + 2]);
            }
            beta[t * s_len + s] = if b == ninf { ninf } else { b + x(t, s) };
        }
    }
    let log_p = lae(alpha[last + s_len - 1], if s_len > 1 { alpha[last + s_len - 2] } else { ninf });
    let mut grad = vec![0f32; t_len * v];
    if !log_p.is_finite() {
        return (0.0, grad);
    }
    for t in 0..t_len {
        let mut acc: HashMap<usize, f64> = HashMap::new();
        for s in 0..s_len {
            let ab = alpha[t * s_len + s] + beta[t * s_len + s];
            if ab.is_finite() {
                let e = acc.entry(ext[s]).or_insert(ninf);
                *e = lae(*e, ab);
            }
        }
        for (k, lab) in acc {
            // occupancy γ_t(k) = Σ α β / (y_t(k) P); d(-log P)/d logp_t(k) = -γ_t(k)
            let gamma = (lab - f64::from(logp[t * v + k]) - log_p).exp();
            grad[t * v + k] = -(gamma as f32);
        }
    }
    (-log_p, grad)
}

struct CtcLoss {
    labels: Vec<u32>,
}

impl CustomOp1 for CtcLoss {
    fn name(&self) -> &'static str {
        "ctc-loss"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let (t, v) = l.shape().dims2()?;
        let data = match s {
            CpuStorage::F32(d) => &d[l.start_offset()..l.start_offset() + t * v],
            _ => candle_core::bail!("ctc-loss expects f32"),
        };
        if !l.is_contiguous() {
            candle_core::bail!("ctc-loss expects a contiguous input");
        }
        let (loss, _) = ctc_loss_grad(data, t, v, &self.labels);
        Ok((CpuStorage::F32(vec![loss as f32]), Shape::from(())))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> candle_core::Result<Option<Tensor>> {
        let (t, v) = arg.dims2()?;
        let data = arg.to_device(&Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let (_, g) = ctc_loss_grad(&data, t, v, &self.labels);
        let g = Tensor::from_vec(g, (t, v), &Device::Cpu)?.to_device(arg.device())?;
        Ok(Some(g.broadcast_mul(&grad_res.to_device(arg.device())?)?))
    }
}

/// Sum of KL(smoothed one-hot ‖ softmax(logits)) over positions (ESPnet LabelSmoothingLoss,
/// smoothing 0.1, batch of one).
fn label_smoothing_loss(logits: &Tensor, targets: &[u32], smoothing: f64) -> Result<Tensor> {
    let (l, v) = logits.dims2()?;
    let eps = smoothing / (v - 1) as f64;
    let conf = 1.0 - smoothing;
    let mut td = vec![eps as f32; l * v];
    for (i, &t) in targets.iter().enumerate() {
        td[i * v + t as usize] = conf as f32;
    }
    let constant = l as f64 * ((v - 1) as f64 * eps * eps.ln() + conf * conf.ln());
    let td = Tensor::from_vec(td, (l, v), logits.device())?;
    let logp = candle_nn::ops::log_softmax(logits, 1)?;
    Ok(((td * logp)?.sum_all()?.neg()? + constant)?)
}

// -- data --------------------------------------------------------------------------------

/// A tiny deterministic RNG (SplitMix64) for shuffling and augmentation.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform integer in [lo, hi] (inclusive, like Python's randint).
    pub fn int(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.int(0, i);
            v.swap(i, j);
        }
    }
}

/// (T,96,96) uint8 → (T,88,88) normalised, randomly cropped/flipped/brightened/time-masked.
pub fn augment(rois: &[u8], t: usize, rng: &mut Rng) -> Vec<f32> {
    let (i0, j0) = (rng.int(0, ROI - CROP), rng.int(0, ROI - CROP));
    let flip = rng.uniform(0.0, 1.0) < 0.5;
    let (gain, bias) = (rng.uniform(0.85, 1.15), rng.uniform(-0.06, 0.06));
    let mut x = vec![0f32; t * CROP * CROP];
    for f in 0..t {
        for y in 0..CROP {
            for c in 0..CROP {
                let sc = if flip { CROP - 1 - c } else { c };
                let p = f32::from(rois[f * ROI * ROI + (y + i0) * ROI + sc + j0]) / 255.0;
                x[(f * CROP + y) * CROP + c] = p * gain + bias;
            }
        }
    }
    for _ in 0..2 {
        // time masks of up to 0.4 s, filled with the clip mean
        let w = rng.int(0, 10.min(t / 5));
        let s = rng.int(0, t.saturating_sub(w));
        let mean = x.iter().sum::<f32>() / x.len().max(1) as f32;
        for v in &mut x[s * CROP * CROP..((s + w).min(t)) * CROP * CROP] {
            *v = mean;
        }
    }
    x.iter_mut().for_each(|v| *v = (*v - MEAN) / STD);
    x
}

/// Centre crop without augmentation (evaluation, reference checks).
pub fn centre(rois: &[u8], t: usize) -> Vec<f32> {
    let o = (ROI - CROP) / 2;
    let mut x = Vec::with_capacity(t * CROP * CROP);
    for f in 0..t {
        for y in 0..CROP {
            x.extend(rois[f * ROI * ROI + (y + o) * ROI + o..][..CROP].iter().map(|&p| (f32::from(p) / 255.0 - MEAN) / STD));
        }
    }
    x
}

// -- trainer -----------------------------------------------------------------------------

pub struct LossParts {
    pub total: Tensor,
    pub ctc: f32,
    pub att: f32,
}

pub struct Trainer {
    encoder: Encoder,
    decoder: Decoder,
    ctc_lo: Linear,
    vars: Vec<(String, Var)>,
    frozen: HashMap<String, Tensor>,
    opt: AdamW,
    spm: SentencePiece,
    ids: HashMap<String, u32>,
    unk: u32,
    eos: u32,
    pub device: Device,
}

impl Trainer {
    /// Start from the base model in `models` (vsr/model.pth, lm/unigram5000.model).
    pub fn new(models: &Path, device: &Device, lr: f64) -> Result<Self> {
        // Training through candle's Metal backend rebooted the user's Mac (M1 Pro, 02.10.2026):
        // the backward pass of the conv front end is not safe there. CPU only until that is
        // understood without risking the machine.
        if !device.is_cpu() {
            bail!("training on the GPU is disabled (it crashed the machine); use the CPU");
        }
        let conf: serde_json::Value = serde_json::from_slice(&std::fs::read(models.join("vsr/model.json"))?)?;
        let conf = match conf {
            serde_json::Value::Array(mut a) if a.len() == 3 => a.remove(2),
            c => c,
        };
        let get = |k: &str| conf.get(k).and_then(serde_json::Value::as_u64).map(|x| x as usize).with_context(|| format!("model.json: {k}"));
        let (elayers, heads, dlayers) = (get("elayers")?, get("aheads")?, get("dlayers")?);
        let mut w = Weights::from_pth(&models.join("vsr/model.pth"), &Device::Cpu)?.to_device(device)?;
        w.trainable = true;
        let mut vars = Vec::new();
        let mut frozen = HashMap::new();
        let names: Vec<String> = w.names().cloned().collect();
        for name in names {
            if PREFIXES.iter().any(|p| name.starts_with(p)) {
                let t = w.get(&name)?;
                if is_trainable(&name) {
                    let v = Var::from_tensor(&t)?;
                    w.insert(&name, v.as_tensor().clone());
                    vars.push((name, v));
                } else {
                    frozen.insert(name, t);
                }
            }
        }
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        let encoder = Encoder::load(&w.pp("encoder"), elayers, heads)?;
        let decoder = Decoder::load(&w.pp("decoder"), dlayers, heads)?;
        let ctc_lo = Linear::load(&w.pp("ctc.ctc_lo"), true)?;
        let opt = AdamW::new(vars.iter().map(|v| v.1.clone()).collect(), ParamsAdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 1e-4 })?;
        let tokens = token_list();
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let unk = *ids.get("<unk>").context("no <unk> token")?;
        let eos = (tokens.len() - 1) as u32;
        let spm = SentencePiece::load(&models.join("lm/unigram5000.model")).context("SentencePiece model (lm/unigram5000.model)")?;
        Ok(Self { encoder, decoder, ctc_lo, vars, frozen, opt, spm, ids, unk, eos, device: device.clone() })
    }

    /// Letters and apostrophes only, like the model's training text: punctuation would be <unk>.
    pub fn targets(&self, text: &str) -> Vec<u32> {
        let clean: String = text.chars().map(|c| if c.is_ascii_alphabetic() || c == '\'' || c == ' ' { c.to_ascii_uppercase() } else { ' ' }).collect();
        let clean = clean.split_whitespace().collect::<Vec<_>>().join(" ");
        self.spm.encode(&clean).iter().map(|p| *self.ids.get(p).unwrap_or(&self.unk)).collect()
    }

    pub fn loss(&self, x: Vec<f32>, t: usize, ys: &[u32]) -> Result<LossParts> {
        if ys.is_empty() {
            bail!("empty target");
        }
        let x = Tensor::from_vec(x, (t, CROP, CROP), &self.device)?;
        let hs = self.encoder.forward(&x)?; // (T, D)
        let logp = candle_nn::ops::log_softmax(&self.ctc_lo.forward(&hs)?, 1)?;
        let ctc = logp.to_device(&Device::Cpu)?.contiguous()?.apply_op1(CtcLoss { labels: ys.to_vec() })?;
        let ys_in: Vec<u32> = std::iter::once(self.eos).chain(ys.iter().copied()).collect();
        let ys_out: Vec<u32> = ys.iter().copied().chain(std::iter::once(self.eos)).collect();
        let logits = self.decoder.forward_train(&ys_in, &hs)?;
        let att = label_smoothing_loss(&logits, &ys_out, 0.1)?;
        let (ctc_v, att_v) = (ctc.to_scalar::<f32>()?, att.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_scalar::<f32>()?);
        let total = (((ctc.to_device(&self.device)? * 0.1)? + (att * 0.9)?)? / ys.len() as f64)?;
        Ok(LossParts { total, ctc: ctc_v, att: att_v })
    }

    /// Gradients of the trainable parameters, and their global L2 norm.
    pub fn grads(&self, loss: &Tensor) -> Result<(GradStore, f64)> {
        let grads = loss.backward()?;
        let mut sq = 0f64;
        for (_, v) in &self.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                sq += f64::from(g.sqr()?.sum_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_scalar::<f32>()?);
            }
        }
        Ok((grads, sq.sqrt()))
    }

    pub fn grad_norm_of(&self, grads: &GradStore, name: &str) -> Option<f64> {
        let v = self.vars.iter().find(|v| v.0 == name)?;
        let g = grads.get(v.1.as_tensor())?;
        g.sqr().ok()?.sum_all().ok()?.to_device(&Device::Cpu).ok()?.to_scalar::<f32>().ok().map(|x| f64::from(x).sqrt())
    }

    /// One optimizer step with the gradients clipped to `max_norm` (clip_grad_norm_).
    pub fn step(&mut self, mut grads: GradStore, norm: f64, max_norm: f64) -> Result<()> {
        let coef = max_norm / (norm + 1e-6);
        if coef < 1.0 {
            for (_, v) in &self.vars {
                if let Some(g) = grads.remove(v.as_tensor()) {
                    grads.insert(v.as_tensor(), (g * coef)?);
                }
            }
        }
        self.opt.step(&grads)?;
        Ok(())
    }

    /// Every tensor under the trained prefixes (current values), as the Python app saved them.
    pub fn trained_state(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        for (name, v) in &self.vars {
            out.insert(name.clone(), v.as_tensor().detach().to_device(&Device::Cpu)?);
        }
        for (name, t) in &self.frozen {
            out.insert(name.clone(), t.to_device(&Device::Cpu)?);
        }
        Ok(out)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = path.with_extension("safetensors.tmp");
        candle_core::safetensors::save(&self.trained_state()?, &tmp)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Clips for fine-tuning: (T×96×96 crops, frames, text).
pub struct TrainClip<'a> {
    pub rois: &'a [u8],
    pub t: usize,
    pub text: &'a str,
}

/// Fine-tune on `clips` (6 epochs, lr 1e-4, AdamW, gradient norm 5): the Python defaults.
pub fn finetune(trainer: &mut Trainer, clips: &[TrainClip], epochs: usize, seed: u64, mut on_epoch: impl FnMut(usize, usize, f64)) -> Result<()> {
    let mut rng = Rng::new(seed);
    let mut order: Vec<usize> = (0..clips.len()).collect();
    let targets: Vec<Vec<u32>> = clips.iter().map(|c| trainer.targets(c.text)).collect();
    for ep in 0..epochs {
        rng.shuffle(&mut order);
        let mut total = 0f64;
        let mut n = 0usize;
        for &k in &order {
            let c = &clips[k];
            if targets[k].is_empty() || c.t < 5 {
                continue;
            }
            let x = augment(c.rois, c.t, &mut rng);
            let parts = trainer.loss(x, c.t, &targets[k])?;
            let (grads, norm) = trainer.grads(&parts.total)?;
            trainer.step(grads, norm, 5.0)?;
            total += f64::from(parts.total.to_device(&Device::Cpu)?.to_scalar::<f32>()?);
            n += 1;
        }
        on_epoch(ep + 1, epochs, total / n.max(1) as f64);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force over all alignments on a tiny problem.
    #[test]
    fn ctc_matches_brute_force() {
        let (t, v) = (4usize, 3usize);
        let raw: Vec<f32> = (0..t * v).map(|i| ((i * 7 % 11) as f32 - 5.0) / 3.0).collect();
        let mut logp = vec![0f32; t * v];
        for r in 0..t {
            let m = raw[r * v..][..v].iter().map(|x| x.exp()).sum::<f32>().ln();
            for k in 0..v {
                logp[r * v + k] = raw[r * v + k] - m;
            }
        }
        let labels = [1u32, 2];
        let mut p = 0f64;
        let mut paths = vec![vec![]];
        for _ in 0..t {
            paths = paths.into_iter().flat_map(|pre: Vec<usize>| (0..v).map(move |k| [pre.clone(), vec![k]].concat())).collect();
        }
        for path in &paths {
            let mut collapsed = Vec::new();
            let mut prev = usize::MAX;
            for &k in path {
                if k != prev && k != 0 {
                    collapsed.push(k as u32);
                }
                prev = k;
            }
            if collapsed == labels {
                p += path.iter().enumerate().map(|(r, &k)| f64::from(logp[r * v + k])).sum::<f64>().exp();
            }
        }
        let (loss, grad) = ctc_loss_grad(&logp, t, v, &labels);
        assert!((loss + p.ln()).abs() < 1e-5, "{loss} vs {}", -p.ln());
        // gradient rows sum to -1 (occupancies sum to 1 per frame)
        for r in 0..t {
            let s: f32 = grad[r * v..][..v].iter().sum();
            assert!((s + 1.0).abs() < 1e-4, "row {r}: {s}");
        }
    }
}
