//! Check the MultiVSR port against the PyTorch reference (`tools/multivsr_ref.py`).
//!
//!   mvsr_check [--cpu] [DIR] [REF]   DIR = ref/multivsr, REF = DIR/ref_silent.safetensors

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use lipflow_vsr::multivsr::{MultiVsr, RU, SOT};

fn diff(name: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let a = a.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    let (mut m, mut s) = (0f32, 0f32);
    for (x, y) in a.iter().zip(&b) {
        m = m.max((x - y).abs());
        s = s.max(y.abs());
    }
    println!("{name:8} max|Δ| {m:.2e} (max|ref| {s:.2}, n {})", a.len());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = pos.first().map_or_else(|| root.join("ref/multivsr"), PathBuf::from);
    let refp = pos.get(1).map_or_else(|| dir.join("ref_silent.safetensors"), PathBuf::from);
    let dev = if cpu { Device::Cpu } else { Device::new_metal(0).unwrap_or(Device::Cpu) };

    if let Some(out) = std::env::var_os("MVSR_CONVERT") {
        let t = Instant::now();
        let n = lipflow_vsr::multivsr::convert_checkpoints(&dir.join("model.pth"), &dir.join("feature_extractor.pth"), std::path::Path::new(&out))?;
        println!("converted {n} tensors in {:.1}s", t.elapsed().as_secs_f64());
        let a = candle_core::safetensors::load(&out, &Device::Cpu)?;
        let b = candle_core::safetensors::load(dir.join("multivsr.safetensors"), &Device::Cpu)?;
        let mut worst = 0f32;
        for (k, v) in &b {
            let w = a.get(k).with_context(|| format!("{k} missing in the Rust conversion"))?;
            worst = worst.max((w - v)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?);
        }
        println!("python {} tensors, rust {} tensors, max|Δ| {worst:e}", b.len(), a.len());
        return Ok(());
    }
    if std::env::var_os("MVSR_MM").is_some() {
        for dt in [candle_core::DType::F32, candle_core::DType::F16, candle_core::DType::BF16] {
            let a = Tensor::randn(0f32, 1., (36864, 256), &dev)?.to_dtype(dt)?;
            let w = Tensor::randn(0f32, 1., (1024, 256), &dev)?.to_dtype(dt)?;
            let wt = w.t()?.contiguous()?;
            for (name, rhs) in [("w.t()", w.t()?), ("contig", wt.clone())] {
                let _ = a.matmul(&rhs)?.sum_all()?.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
                let t = Instant::now();
                let mut acc = None;
                for _ in 0..10 {
                    acc = Some(a.matmul(&rhs)?);
                }
                let _ = acc.unwrap().sum_all()?.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
                let dt_s = t.elapsed().as_secs_f64() / 10.0;
                println!("{dt:?} {name}: {:.1} ms, {:.2} TFLOPS", dt_s * 1e3, 2.0 * 36864.0 * 256.0 * 1024.0 / dt_s / 1e12);
            }
        }
        return Ok(());
    }
    let t = Instant::now();
    let m = MultiVsr::load(&dir, &dev)?;
    println!("load {:.2}s on {dev:?}", t.elapsed().as_secs_f64());
    let r = candle_core::safetensors::load(&refp, &Device::Cpu)?;
    let get = |k: &str| r.get(k).cloned().with_context(|| format!("{k} missing"));

    let faces = get("faces")?.permute((1, 0, 2, 3))?.contiguous()?.to_device(&dev)?; // (T, 3, 96, 96)
    let n = faces.dim(0)?;
    let cnn = m.vtp.cnn(&faces.narrow(0, 0, 32)?)?.narrow(0, 0, 30)?;
    let rc = get("cnn")?.permute((1, 0, 2, 3))?.narrow(0, 0, 30)?.contiguous()?;
    diff("cnn[:30]", &cnn, &rc)?;

    if std::env::var_os("MVSR_PROFILE").is_some() {
        let t = Instant::now();
        let c = m.vtp.cnn(&faces)?;
        let _ = c.sum_all()?.to_scalar::<f32>()?;
        println!("profile: cnn {:.2}s", t.elapsed().as_secs_f64());
        for chunk in [8usize, 16, 32, 64, 128] {
            let t = Instant::now();
            let f = m.vtp.forward(&faces, chunk)?;
            let _ = f.sum_all()?.to_scalar::<f32>()?;
            println!("profile: vtp chunk {chunk}: {:.2}s", t.elapsed().as_secs_f64());
        }
    }
    let t = Instant::now();
    let feats = m.vtp.forward(&faces, 64)?;
    let _ = feats.to_device(&Device::Cpu)?;
    let t_vtp = t.elapsed().as_secs_f64();
    diff("feats", &feats, &get("feats")?)?;
    let t = Instant::now();
    let enc = m.s2s.encode(&feats)?;
    let _ = enc.to_device(&Device::Cpu)?;
    let t_enc = t.elapsed().as_secs_f64();
    diff("memory", &enc, &get("memory")?)?;
    // the rest from the reference memory, so token checks test the decoder alone
    let enc_ref = get("memory")?.to_device(&dev)?;
    let mem = m.s2s.memory(&enc_ref)?;

    let t = Instant::now();
    let g = m.s2s.greedy(&mem, &[SOT, RU], 100)?;
    let t_greedy = t.elapsed().as_secs_f64();
    let want: Vec<u32> = get("greedy")?.to_dtype(candle_core::DType::U32)?.to_vec1()?;
    println!("greedy   {} ({:.2}s): {}", if g == want { "SAME" } else { "DIFF" }, t_greedy, m.tokens.decode(&g));
    for size in [5usize, 20] {
        let t = Instant::now();
        let (seq, score) = m.s2s.beam_search(&mem, &[SOT, RU], size, 100)?;
        let dt = t.elapsed().as_secs_f64();
        let want: Vec<u32> = get(&format!("beam{size}"))?.to_dtype(candle_core::DType::U32)?.to_vec1()?;
        let ws = get(&format!("beam{size}_score"))?.to_vec1::<f32>()?[0];
        println!("beam{size:<3} {} ({dt:.2}s) score {score:.4} vs {ws:.4}: {}", if seq == want { "SAME" } else { "DIFF" }, m.tokens.decode(&seq));
    }
    // incremental features (as during recording) must equal the whole-clip pass
    let bytes: Vec<u8> = (get("faces")?.permute((1, 2, 3, 0))?.contiguous()? * 255.0)?.round()?.flatten_all()?.to_vec1::<f32>()?.iter().map(|&v| v as u8).collect();
    let whole = m.features(&bytes, n, 0, n)?;
    let cuts = [0, 37, 120, 121, 300, n];
    let parts: Vec<Tensor> = cuts.windows(2).map(|w| m.features(&bytes, n, w[0], w[1])).collect::<Result<_>>()?;
    diff("pieces", &Tensor::cat(&parts, 0)?, &whole.to_device(&Device::Cpu)?)?;
    // end to end from faces, the way the app will call it
    let t = Instant::now();
    let enc = m.encode(&faces)?;
    let text = m.read(&enc, 5)?;
    println!("end-to-end beam5 {:.2}s for {n} frames ({:.1}s of video): {text}", t.elapsed().as_secs_f64(), n as f64 / 25.0);
    println!("timings: vtp {t_vtp:.2}s, encoder {t_enc:.2}s");
    Ok(())
}
