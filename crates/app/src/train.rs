//! Training on your face and your phrasing (`lipflow/train_vsr.py`, `train_lm.py`,
//! `dictation.train_on_face`).

use std::path::Path;

use crate::data;

/// The sentences of the clips saved in `dir` (practice rounds skip sentences already recorded).
pub fn clip_texts(dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    rd.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "npz"))
        .filter_map(|p| match data::read_npz(&p) {
            Ok(arrays) => arrays.into_iter().find(|a| a.0 == "text").and_then(|a| a.1.first_string()),
            Err(e) => {
                eprintln!("[lipflow] skipping {}: {e}", p.display());
                None
            }
        })
        .collect()
}

pub struct TrainResult {
    pub before: f64,
    pub after: Option<f64>,
    pub kept: bool,
    pub clips: usize,
    pub note: String,
}

fn find_uv() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let mut cands: Vec<std::path::PathBuf> = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).map(|d| d.join("uv")).collect()).unwrap_or_default();
    cands.extend(home.map(|h| h.join(".local/bin/uv")));
    cands.push("/opt/homebrew/bin/uv".into());
    cands.into_iter().find(|p| p.is_file())
}

/// The Python checkout that holds the training code (and its uv environment).
fn python_checkout() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("LIPFLOW_PY_CHECKOUT") {
        return Some(p.into());
    }
    let dev = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lipflow");
    dev.join("lipflow/dictation.py").exists().then_some(dev)
}

fn train_script() -> Option<std::path::PathBuf> {
    bundled_or_dev("train_face.py")
}

/// A script from the app bundle's Resources, else from the source tree's scripts/.
fn bundled_or_dev(name: &str) -> Option<std::path::PathBuf> {
    let bundled = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("../Resources").join(name)));
    let dev = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts").join(name);
    bundled.into_iter().chain(std::iter::once(dev)).find(|p| p.is_file())
}

/// Personal LM, then face adaptation with a held-out check — run by the Python app's own code
/// (PyTorch): porting the training to Rust was slower on the CPU and crashed the machine on
/// Metal, with no accuracy to gain. The Rust app loads the weights it writes.
pub fn train_on_face(beam: usize, report: &dyn Fn(f64, &str)) -> anyhow::Result<TrainResult> {
    let missing = |what: &str| TrainResult { before: 0.0, after: None, kept: false, clips: 0, note: format!("Training needs the Python toolkit ({what}). See README → Training.") };
    let Some(uv) = find_uv() else { return Ok(missing("uv not found")) };
    let Some(checkout) = python_checkout() else { return Ok(missing("set LIPFLOW_PY_CHECKOUT to the Python lipflow checkout")) };
    let Some(script) = train_script() else { return Ok(missing("train_face.py not found")) };
    report(2.0, "Starting training…");
    let mut cmd = std::process::Command::new(uv);
    cmd.arg("run")
        .arg("--project")
        .arg(&checkout)
        .arg("python")
        .arg(&script)
        .env("LIPFLOW_HOME", crate::paths::home())
        .env("LIPFLOW_PY_CHECKOUT", &checkout)
        .env("LIPFLOW_BEAM", beam.to_string());
    let r = run_trainer(cmd, report)?;
    if r.kept {
        // a face model from an earlier Rust experiment would shadow the new .pth
        let _ = std::fs::remove_file(crate::paths::personal_vsr().with_extension("safetensors"));
    }
    Ok(r)
}

/// Russian: fine-tune MultiVSR on the Russian practice clips (scripts/train_face_ru.py, PyTorch on
/// the CPU, its own throwaway environment: no Python checkout needed). Writes
/// models/multivsr_face.safetensors, which the app lays over the base model.
pub fn train_on_face_ru(report: &dyn Fn(f64, &str)) -> anyhow::Result<TrainResult> {
    let missing = |what: &str| TrainResult { before: 0.0, after: None, kept: false, clips: 0, note: format!("Training needs {what}. See README → Training.") };
    let Some(uv) = find_uv() else { return Ok(missing("uv (astral.sh/uv)")) };
    let Some(script) = bundled_or_dev("train_face_ru.py") else { return Ok(missing("train_face_ru.py")) };
    report(2.0, "Preparing the training environment (first time: downloads PyTorch)…");
    let mut cmd = std::process::Command::new(uv);
    cmd.args(["run", "--no-project", "--python", "3.12", "--with", "torch", "--with", "numpy", "--with", "safetensors", "--with", "tokenizers", "python"])
        .arg(&script)
        .env("LIPFLOW_HOME", crate::paths::home())
        .env("LIPFLOW_MULTIVSR", crate::paths::multivsr());
    run_trainer(cmd, report)
}

/// Run a trainer that prints "PROGRESS <pct> <text>" lines and a final "RESULT <json>".
fn run_trainer(mut cmd: std::process::Command, report: &dyn Fn(f64, &str)) -> anyhow::Result<TrainResult> {
    use std::io::BufRead;
    let mut child = cmd.stdout(std::process::Stdio::piped()).spawn()?;
    let mut result: Option<serde_json::Value> = None;
    if let Some(out) = child.stdout.take() {
        for line in std::io::BufReader::new(out).lines() {
            let line = line?;
            if let Some(rest) = line.strip_prefix("PROGRESS ") {
                let (pct, text) = rest.split_once(' ').unwrap_or((rest, ""));
                report(pct.parse().unwrap_or(0.0), text);
            } else if let Some(json) = line.strip_prefix("RESULT ") {
                result = serde_json::from_str(json).ok();
            } else {
                println!("[train] {line}");
            }
        }
    }
    let status = child.wait()?;
    let Some(r) = result else { anyhow::bail!("the Python trainer exited ({status}) without a result") };
    let kept = r["kept"].as_bool().unwrap_or(false);
    Ok(TrainResult {
        before: r["before"].as_f64().unwrap_or(0.0),
        after: r["after"].as_f64(),
        kept,
        clips: r["clips"].as_u64().unwrap_or(0) as usize,
        note: r["note"].as_str().unwrap_or_default().to_string(),
    })
}

/// `lipflow train-lm`: the Python app's LM fine-tuning on your phrases (writes lm_phrasing.pth).
pub fn train_lm() -> anyhow::Result<()> {
    let uv = find_uv().ok_or_else(|| anyhow::anyhow!("uv not found"))?;
    let checkout = python_checkout().ok_or_else(|| anyhow::anyhow!("set LIPFLOW_PY_CHECKOUT to the Python lipflow checkout"))?;
    let status = std::process::Command::new(uv)
        .arg("run")
        .arg("--project")
        .arg(&checkout)
        .arg("lipflow")
        .arg("train-lm")
        .current_dir(&checkout)
        .env("LIPFLOW_HOME", crate::paths::home())
        .status()?;
    anyhow::ensure!(status.success(), "train-lm failed ({status})");
    Ok(())
}
