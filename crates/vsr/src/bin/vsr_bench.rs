//! Compare the Rust port with the PyTorch reference (`tools/baseline.py`, `tools/dump_ref.py`).
//!
//!   vsr_bench check [--cpu]   numerical check of front end + encoder on one clip
//!   vsr_bench bench [--cpu] [--beam N] [--no-lm]   all clips: timings, WER, agreement

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use lipflow_vsr::{LipReader, ReaderOptions};

/// Minimal .npy reader for C-ordered u1/f4 arrays.
fn read_npy(path: &Path) -> Result<(Vec<usize>, String, Vec<u8>)> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if &b[..6] != b"\x93NUMPY" {
        bail!("{} is not .npy", path.display());
    }
    let (hlen, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let header = std::str::from_utf8(&b[off..off + hlen])?;
    let descr = header.split("'descr': '").nth(1).and_then(|s| s.split('\'').next()).context("npy descr")?.to_string();
    if header.contains("'fortran_order': True") {
        bail!("fortran order unsupported");
    }
    let shape_s = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).context("npy shape")?;
    let shape: Vec<usize> = shape_s.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    Ok((shape, descr, b[off + hlen..].to_vec()))
}

fn npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let (shape, descr, data) = read_npy(path)?;
    if descr != "<f4" {
        bail!("expected <f4, got {descr}");
    }
    Ok((shape, data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn wer(hyp: &str, reference: &str) -> (usize, usize) {
    let (h, r) = (words(hyp), words(reference));
    let mut d: Vec<usize> = (0..=r.len()).collect();
    for (i, a) in h.iter().enumerate() {
        let mut prev = d[0];
        d[0] = i + 1;
        for (j, b) in r.iter().enumerate() {
            let cur = d[j + 1];
            d[j + 1] = (d[j + 1] + 1).min(d[j] + 1).min(prev + usize::from(a != b));
            prev = cur;
        }
    }
    (d[r.len()], r.len())
}

fn max_diff(a: &Tensor, b: &[f32]) -> Result<(f32, f32)> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let mut m = 0f32;
    let mut scale = 0f32;
    for (x, y) in a.iter().zip(b) {
        m = m.max((x - y).abs());
        scale = scale.max(y.abs());
    }
    Ok((m, scale))
}

fn base_text(refdir: &Path, i: usize) -> Result<String> {
    let base: serde_json::Value = serde_json::from_slice(&std::fs::read(refdir.join("baseline.json"))?)?;
    Ok(base["clips"][i]["text"].as_str().context("text")?.to_string())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let refdir = root.join("ref");
    let mut opts = ReaderOptions::new(root.join("lipflow/models"));
    opts.gpu = !args.iter().any(|a| a == "--cpu");
    opts.use_lm = !args.iter().any(|a| a == "--no-lm");
    opts.beam_gpu = args.iter().any(|a| a == "--beam-gpu");
    if let Some(i) = args.iter().position(|a| a == "--beam") {
        opts.beam.beam = args[i + 1].parse()?;
    }
    let t0 = Instant::now();
    let reader = LipReader::load(&opts)?;
    println!("load {:.2}s, encoder on {:?}", t0.elapsed().as_secs_f64(), reader.enc_device);

    match args.get(1).map(String::as_str) {
        Some("check") => {
            let (shape, _, rois) = read_npy(&refdir.join("bench/001.npy"))?;
            let x = LipReader::to_input(&rois, shape[0], &reader.enc_device)?;
            let feats = reader.encoder.video_frontend().context("video model")?.forward(&x)?;
            let (_, rf) = npy_f32(&refdir.join("ref_feats_001.npy"))?;
            let (m, s) = max_diff(&feats.to_device(&Device::Cpu)?, &rf)?;
            println!("frontend  max|Δ| {m:.2e} (max|ref| {s:.2})");
            let enc = reader.encoder.forward_features(&feats)?.to_device(&Device::Cpu)?;
            let (_, re) = npy_f32(&refdir.join("ref_enc_001.npy"))?;
            let (m, s) = max_diff(&enc, &re)?;
            println!("encoder   max|Δ| {m:.2e} (max|ref| {s:.2})");
            println!("greedy: {}", reader.greedy(&enc)?);
            for (i, t) in reader.beam_search(&enc, 5)?.iter().enumerate() {
                println!("beam {i}: {t}");
            }
        }
        Some("profile") => {
            let (shape, _, rois) = read_npy(&refdir.join("bench/001.npy"))?;
            let dev = reader.enc_device.clone();
            for _ in 0..3 {
                let a = Instant::now();
                let x = LipReader::to_input(&rois, shape[0], &dev)?;
                let feats = reader.encoder.video_frontend().context("video model")?.forward(&x)?;
                dev.synchronize()?;
                let tf = a.elapsed().as_secs_f64();
                let b = Instant::now();
                let enc = reader.encoder.forward_features(&feats)?;
                dev.synchronize()?;
                let tc = b.elapsed().as_secs_f64();
                let enc = enc.to_device(&reader.beam_device)?;
                let c = Instant::now();
                let _ = reader.beam_search(&enc, 5)?;
                println!("frontend {tf:.3}s  conformer {tc:.3}s  beam {:.3}s", c.elapsed().as_secs_f64());
            }
        }
        Some("av-check") => {
            drop(reader);
            let t0 = Instant::now();
            let av = LipReader::load_av(&opts)?;
            println!("AV load {:.2}s", t0.elapsed().as_secs_f64());
            let (shape, _, rois) = read_npy(&refdir.join("bench/001.npy"))?;
            let (_, wave) = npy_f32(&refdir.join("av_wave_001.npy"))?;
            let (_, want) = npy_f32(&refdir.join("av_enc_001.npy"))?;
            for _ in 0..2 {
                let t1 = Instant::now();
                let enc = av.encode_av(&rois, shape[0], &wave)?;
                let te = t1.elapsed().as_secs_f64();
                let (m, sc) = max_diff(&enc.to_device(&Device::Cpu)?, &want)?;
                let t2 = Instant::now();
                let hyp = av.beam_search(&enc, 1)?;
                println!("AV enc max|Δ| {m:.2e} (max|ref| {sc:.2}) enc {te:.2}s beam {:.2}s: {}", t2.elapsed().as_secs_f64(), hyp.first().cloned().unwrap_or_default());
            }
            return Ok(());
        }
        Some("train-check") => {
            use lipflow_vsr::train::{Trainer, centre};
            let r: serde_json::Value = serde_json::from_slice(&std::fs::read(refdir.join("train_ref.json"))?)?;
            let dev = Device::Cpu; // GPU training is disabled: it crashed the machine
            let tr = Trainer::new(&opts.models, &dev, 1e-4)?;
            for c in r.as_array().context("train_ref")? {
                let i = c["i"].as_u64().context("i")? as usize;
                let (shape, _, rois) = read_npy(&refdir.join(format!("bench/{i:03}.npy")))?;
                let text = base_text(&refdir, i)?;
                let ys = tr.targets(&text);
                let want: Vec<u32> = c["ys"].as_array().context("ys")?.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect();
                println!("clip {i}: targets {}", if ys == want { "match" } else { "DIFFER" });
                let t0 = Instant::now();
                let parts = tr.loss(centre(&rois, shape[0]), shape[0], &ys)?;
                let t_fwd = t0.elapsed().as_secs_f64();
                let (grads, norm) = tr.grads(&parts.total)?;
                let dt = t0.elapsed().as_secs_f64();
                println!("  forward {t_fwd:.2}s backward {:.2}s", dt - t_fwd);
                let total = parts.total.to_device(&Device::Cpu)?.to_scalar::<f32>()?;
                println!("  ctc {:.4} (py {:.4})  att {:.4} (py {:.4})  loss {:.5} (py {:.5})  |g| {norm:.5} (py {:.5})  fwd+bwd {dt:.2}s",
                    parts.ctc, c["ctc"].as_f64().unwrap_or(0.0), parts.att, c["att"].as_f64().unwrap_or(0.0), total, c["loss"].as_f64().unwrap_or(0.0), c["grad_total"].as_f64().unwrap_or(0.0));
                for (name, py) in c["grad"].as_object().context("grad")? {
                    println!("    {name}: {:.5} (py {:.5})", tr.grad_norm_of(&grads, name).unwrap_or(f64::NAN), py.as_f64().unwrap_or(0.0));
                }
            }
        }
        Some("bench") => {
            let base: serde_json::Value = serde_json::from_slice(&std::fs::read(refdir.join("baseline.json"))?)?;
            let clips = base["clips"].as_array().context("baseline clips")?;
            reader.warmup()?;
            let (mut errs, mut n, mut same, mut t_enc, mut t_beam) = (0, 0, 0, 0f64, 0f64);
            for c in clips {
                let i = c["i"].as_u64().context("i")? as usize;
                let (shape, _, rois) = read_npy(&refdir.join(format!("bench/{i:03}.npy")))?;
                let a = Instant::now();
                let enc = reader.encode(&rois, shape[0])?;
                let te = a.elapsed().as_secs_f64();
                let b = Instant::now();
                let hyps = reader.beam_search(&enc, 5)?;
                let tb = b.elapsed().as_secs_f64();
                let py = c["hyps"][0].as_str().unwrap_or_default();
                let top = hyps.first().cloned().unwrap_or_default();
                let (e, m) = wer(&top, c["text"].as_str().unwrap_or_default());
                errs += e;
                n += m;
                same += usize::from(top == py);
                t_enc += te;
                t_beam += tb;
                println!("{i:3} T={:3} enc {te:.3}s beam {tb:.3}s {} {top}", shape[0], if top == py { "=" } else { "≠" });
                if top != py {
                    println!("              py: {py}");
                }
            }
            println!(
                "WER {:.2}%  same-as-python {same}/{}  enc total {t_enc:.2}s  beam total {t_beam:.2}s",
                100.0 * errs as f64 / n as f64,
                clips.len()
            );
        }
        _ => bail!("usage: vsr_bench check|bench [--cpu] [--beam N] [--no-lm]"),
    }
    Ok(())
}
