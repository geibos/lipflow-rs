//! `LipReader`: mouth crops -> text, the Rust counterpart of `lipflow/vsr.py`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};

use crate::beam::{BeamConfig, Hyp, Scorers, beam_search};
use crate::ctc::CtcPrefixScorer;
use crate::decoder::{Decoder, Lm};
use crate::encoder::Encoder;
use crate::nn::{BatchNorm, Linear, Weights};

pub const MODEL_FPS: f64 = 25.0;
pub const ROI: usize = 96;
pub const CROP: usize = 88;
const MEAN: f32 = 0.421;
const STD: f32 = 0.165;
const UNITS: &str = include_str!("../assets/unigram5000_units.txt");

#[derive(Clone, Debug)]
pub struct ReaderOptions {
    pub models: PathBuf,
    pub beam: BeamConfig,
    pub use_lm: bool,
    /// Fine-tuned front-end tensors laid over the base model ("your face").
    pub personal_vsr: Option<PathBuf>,
    /// LM fine-tuned on the user's phrases, scored next to the general one.
    pub personal_lm: Option<PathBuf>,
    /// Run the encoder on the GPU when one is available.
    pub gpu: bool,
    /// Also run the decoder/LM steps of the beam search on the GPU.
    pub beam_gpu: bool,
}

impl ReaderOptions {
    pub fn new(models: impl Into<PathBuf>) -> Self {
        Self { models: models.into(), beam: BeamConfig::default(), use_lm: true, personal_vsr: None, personal_lm: None, gpu: true, beam_gpu: false }
    }
}

/// Whisper mode: the audio encoder and the audio-visual fusion of the AV model.
struct AvParts {
    aux: Encoder,
    fc1: Linear,
    bn1: BatchNorm,
    fc2: Linear,
}

pub const SAMPLES_PER_FRAME: usize = 640; // 16 kHz / 25 fps

pub struct LipReader {
    pub encoder: Encoder,
    av: Option<AvParts>,
    decoder: Decoder,
    ctc_lo: Linear,
    lm: Option<Lm>,
    plm: Option<Lm>,
    pub tokens: Vec<String>,
    pub cfg: BeamConfig,
    pub enc_device: Device,
    pub personal_vsr: bool,
    /// Where the decoder, CTC head and LMs live (the encoder output is moved here).
    pub beam_device: Device,
}

fn json_conf(path: &Path) -> Result<serde_json::Value> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)?;
    // ESPnet writes either the args dict or [idim, odim, args].
    Ok(match v {
        serde_json::Value::Array(mut a) if a.len() == 3 => a.remove(2),
        other => other,
    })
}

fn conf_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v.get(key).and_then(serde_json::Value::as_u64).map(|x| x as usize).with_context(|| format!("model.json: missing {key}"))
}

/// "<blank>", the 5047 subword units, "<eos>".
pub fn token_list() -> Vec<String> {
    let mut tokens = vec!["<blank>".to_string()];
    tokens.extend(UNITS.lines().filter_map(|l| l.split_whitespace().next()).map(str::to_string));
    tokens.push("<eos>".to_string());
    tokens
}

/// The personal face model: safetensors written by this app, or the Python app's .pth.
fn personal_file(p: &Path) -> Option<PathBuf> {
    let st = p.with_extension("safetensors");
    if st.exists() {
        Some(st)
    } else if p.exists() {
        Some(p.to_path_buf())
    } else {
        None
    }
}

pub fn pick_device(gpu: bool) -> Device {
    if gpu {
        #[cfg(feature = "metal")]
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

impl LipReader {
    pub fn load(opts: &ReaderOptions) -> Result<Self> {
        let conf = json_conf(&opts.models.join("vsr/model.json"))?;
        let mut w = Weights::from_pth(&opts.models.join("vsr/model.pth"), &Device::Cpu)?;
        let mut personal_vsr = false;
        if let Some(p) = opts.personal_vsr.as_deref().and_then(personal_file) {
            w.overlay_file(&p)?;
            personal_vsr = true;
        }
        Self::build(opts, w, &conf, personal_vsr, false)
    }

    /// Whisper mode: the Auto-AVSR audio-visual model (models/av), reading lips and a soft
    /// whisper together. Same vocabulary and LMs as the lip reader.
    pub fn load_av(opts: &ReaderOptions) -> Result<Self> {
        let conf = json_conf(&opts.models.join("av/config.json"))?;
        let w = Weights::from_safetensors(&opts.models.join("av/model.safetensors"), &Device::Cpu, "avsr.")?;
        Self::build(opts, w, &conf, false, true)
    }

    pub fn av_available(models: &Path) -> bool {
        models.join("av/model.safetensors").exists()
    }

    fn build(opts: &ReaderOptions, w: Weights, conf: &serde_json::Value, personal_vsr: bool, av: bool) -> Result<Self> {
        let cpu = Device::Cpu;
        let enc_device = pick_device(opts.gpu);
        if conf.get("transformer_encoder_attn_layer_type").and_then(|v| v.as_str()) != Some("rel_mha") {
            bail!("only the rel_mha Conformer (Auto-AVSR) is supported");
        }
        let beam_device = if opts.beam_gpu { enc_device.clone() } else { cpu.clone() };
        let w = if beam_device.same_device(&cpu) { w } else { w.to_device(&beam_device)? };
        let w_enc = if enc_device.same_device(w.device()) { None } else { Some(w.to_device(&enc_device)?) };
        let we = w_enc.as_ref().unwrap_or(&w);
        let encoder = Encoder::load(&we.pp("encoder"), conf_usize(conf, "elayers")?, conf_usize(conf, "aheads")?)?;
        let av = if av {
            Some(AvParts {
                aux: Encoder::load(&we.pp("aux_encoder"), conf_usize(conf, "aux_elayers")?, conf_usize(conf, "aux_aheads")?)?,
                fc1: Linear::load(&we.pp("fusion.fc1"), true)?,
                bn1: BatchNorm::load(&we.pp("fusion.bn1"))?,
                fc2: Linear::load(&we.pp("fusion.fc2"), true)?,
            })
        } else {
            None
        };
        let decoder = Decoder::load(&w.pp("decoder"), conf_usize(conf, "dlayers")?, conf_usize(conf, "aheads")?)?;
        let ctc_lo = Linear::load(&w.pp("ctc.ctc_lo"), true)?;

        let tokens = token_list();

        let (mut lm, mut plm) = (None, None);
        if opts.use_lm && opts.beam.lm_weight > 0.0 {
            let lconf = json_conf(&opts.models.join("lm/model.json"))?;
            let (layers, heads) = (conf_usize(&lconf, "layer")?, conf_usize(&lconf, "head")?);
            let lw = Weights::from_pth(&opts.models.join("lm/model.pth"), &beam_device)?;
            lm = Some(Lm::load(&lw.pp(""), layers, heads)?);
            if let Some(p) = &opts.personal_lm
                && p.exists()
                && opts.beam.plm_weight > 0.0
            {
                let pw = Weights::from_pth(p, &beam_device)?;
                plm = Some(Lm::load(&pw.pp(""), layers, heads)?);
            }
        }
        Ok(Self { encoder, av, decoder, ctc_lo, lm, plm, tokens, cfg: opts.beam.clone(), enc_device, personal_vsr, beam_device })
    }

    pub fn has_personal_lm(&self) -> bool {
        self.plm.is_some()
    }

    /// Indices that turn a variable-rate capture into a steady 25 fps sequence.
    pub fn resample(timestamps: &[f64]) -> Vec<usize> {
        let n = timestamps.len();
        if n == 0 {
            return Vec::new();
        }
        let (t0, t1) = (timestamps[0], timestamps[n - 1]);
        let step = 1.0 / MODEL_FPS;
        // np.arange(t0, t1 + 1e-9, step) has ceil((stop - start) / step) elements.
        let count = ((t1 + 1e-9 - t0) / step).ceil().max(0.0) as usize;
        (0..count)
            .map(|k| {
                let g = t0 + k as f64 * step;
                timestamps.partition_point(|&t| t < g).min(n - 1) // np.searchsorted(side="left")
            })
            .collect()
    }

    /// (T, 96, 96) uint8 crops -> (T, 88, 88) centre-cropped, normalised.
    pub fn to_input(rois: &[u8], t: usize, dev: &Device) -> Result<Tensor> {
        if rois.len() != t * ROI * ROI {
            bail!("expected {t}x{ROI}x{ROI} mouth crops, got {} bytes", rois.len());
        }
        let o = (ROI - CROP) / 2;
        let mut v = Vec::with_capacity(t * CROP * CROP);
        for f in 0..t {
            for y in 0..CROP {
                let row = &rois[f * ROI * ROI + (y + o) * ROI + o..][..CROP];
                v.extend(row.iter().map(|&p| (f32::from(p) / 255.0 - MEAN) / STD));
            }
        }
        Ok(Tensor::from_vec(v, (t, CROP, CROP), dev)?)
    }

    /// Encoder output (T, D) on the beam-search device.
    pub fn encode(&self, rois: &[u8], t: usize) -> Result<Tensor> {
        let x = Self::to_input(rois, t, &self.enc_device)?;
        Ok(self.encoder.forward(&x)?.to_device(&self.beam_device)?)
    }

    /// 16 kHz mono -> (T*640,) trimmed/padded to the video length and layer-normalised, so a
    /// whisper's low volume doesn't matter, only its shape.
    pub fn audio_input(wave: &[f32], t: usize, dev: &Device) -> Result<Tensor> {
        let want = t * SAMPLES_PER_FRAME;
        let mut w: Vec<f32> = wave.iter().take(want).copied().collect();
        w.resize(want, 0.0);
        let n = want.max(1) as f64;
        let mean = w.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = w.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        let v: Vec<f32> = w.iter().map(|&x| ((f64::from(x) - mean) * inv) as f32).collect();
        Ok(Tensor::from_vec(v, want, dev)?)
    }

    /// Whisper mode: fused audio-visual encoder output (T, D) on the beam device.
    pub fn encode_av(&self, rois: &[u8], t: usize, wave: &[f32]) -> Result<Tensor> {
        let av = self.av.as_ref().context("not the audio-visual model")?;
        let feat = self.encoder.forward(&Self::to_input(rois, t, &self.enc_device)?)?;
        let aux = av.aux.forward(&Self::audio_input(wave, t, &self.enc_device)?)?;
        let n = feat.dim(0)?.min(aux.dim(0)?);
        let x = Tensor::cat(&[&feat.narrow(0, 0, n)?, &aux.narrow(0, 0, n)?], 1)?;
        let h = av.bn1.forward(&av.fc1.forward(&x)?, 1)?.relu()?;
        Ok(av.fc2.forward(&h)?.to_device(&self.beam_device)?)
    }

    fn ctc_logp(&self, enc: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::log_softmax(&self.ctc_lo.forward(enc)?, 1)?)
    }

    /// CTC best path: about a millisecond, used for the live preview.
    pub fn greedy(&self, enc: &Tensor) -> Result<String> {
        let ids = self.ctc_lo.forward(enc)?.argmax(1)?.to_vec1::<u32>()?;
        let mut out = String::new();
        let mut prev = None;
        for i in ids {
            if Some(i) != prev && i != 0 {
                out.push_str(&self.tokens[i as usize]);
            }
            prev = Some(i);
        }
        Ok(clean(&out))
    }

    pub fn beam_hyps(&self, enc: &Tensor) -> Result<Vec<Hyp>> {
        let logp = self.ctc_logp(enc)?;
        let (t, v) = logp.dims2()?;
        let ctc = CtcPrefixScorer::new(logp.flatten_all()?.to_vec1::<f32>()?, t, v, 0, v - 1);
        let memory = self.decoder.memory(enc)?;
        let scorers = Scorers { decoder: &self.decoder, memory: &memory, ctc: &ctc, lm: self.lm.as_ref(), plm: self.plm.as_ref() };
        let eos = (v - 1) as u32;
        beam_search(&scorers, &self.cfg, eos, eos, &self.beam_device)
    }

    /// Up to `nbest` distinct transcripts, best first (UPPERCASE, as the model emits).
    pub fn beam_search(&self, enc: &Tensor, nbest: usize) -> Result<Vec<String>> {
        let mut texts: Vec<String> = Vec::new();
        for h in self.beam_hyps(enc)?.iter().take(nbest.max(1)) {
            let raw: String = h.yseq[1..].iter().map(|&i| self.tokens[i as usize].as_str()).collect();
            let t = clean(&raw.replace("<space>", " "));
            if !texts.contains(&t) {
                texts.push(t);
            }
        }
        Ok(texts)
    }

    pub fn read(&self, rois: &[u8], t: usize, fast: bool) -> Result<String> {
        let enc = self.encode(rois, t)?;
        if fast { self.greedy(&enc) } else { Ok(self.beam_search(&enc, 1)?.into_iter().next().unwrap_or_default()) }
    }

    pub fn warmup(&self) -> Result<()> {
        self.read(&vec![0u8; 25 * ROI * ROI], 25, false).map(drop)
    }
}

/// "▁HELLO▁WORLD<eos>" -> "HELLO WORLD".
pub fn clean(text: &str) -> String {
    text.replace('\u{2581}', " ").replace("<eos>", "").split_whitespace().collect::<Vec<_>>().join(" ")
}
