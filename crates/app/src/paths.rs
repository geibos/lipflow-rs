//! Where things live. Shared model weights: `LIPFLOW_MODELS`, else `<home>/base-models`, else the
//! Python checkout next to this workspace (development). Everything learned from *you* (clips,
//! phrases, personal models, settings) goes to the Lipflow home, the same folder the Python app
//! used, so existing data carries over. Override with `LIPFLOW_HOME`.

use std::path::PathBuf;

pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("LIPFLOW_HOME") {
        return PathBuf::from(h);
    }
    let user = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map_or(user, PathBuf::from).join("Lipflow")
    } else {
        user.join("Library/Application Support/Lipflow")
    }
}

/// The MultiVSR model (Russian and 12 other languages): multivsr.safetensors + vocab.json.
pub fn multivsr() -> PathBuf {
    if let Some(m) = std::env::var_os("LIPFLOW_MULTIVSR") {
        return PathBuf::from(m);
    }
    let base = home().join("base-models/multivsr");
    if base.join("multivsr.safetensors").exists() {
        return base;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ref/multivsr")
}

pub fn models() -> PathBuf {
    if let Some(m) = std::env::var_os("LIPFLOW_MODELS") {
        return PathBuf::from(m);
    }
    let base = home().join("base-models");
    if base.join("vsr/model.pth").exists() {
        return base;
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lipflow/models");
    if dev.join("vsr/model.pth").exists() {
        return dev;
    }
    base
}

pub fn personal_models() -> PathBuf {
    home().join("models")
}

/// Your fine-tuned MultiVSR tensors (Russian), laid over the base model.
pub fn personal_multivsr() -> PathBuf {
    personal_models().join("multivsr_face.safetensors")
}

pub fn personal_vsr() -> PathBuf {
    personal_models().join("vsr_face.pth")
}

pub fn personal_lm() -> PathBuf {
    personal_models().join("lm_phrasing.pth")
}

pub fn settings() -> PathBuf {
    home().join("settings.json")
}

pub fn history() -> PathBuf {
    home().join("history.jsonl")
}

pub fn words() -> PathBuf {
    home().join("words.txt")
}

pub fn phrases() -> PathBuf {
    home().join("phrases.txt")
}

pub fn clips(kind: &str) -> PathBuf {
    home().join("clips").join(kind)
}

/// Practice clips of a language (English keeps the Python app's folder).
pub fn onboarding_clips(lang: crate::text::Lang) -> PathBuf {
    match lang {
        crate::text::Lang::En => clips("onboarding"),
        crate::text::Lang::Ru => clips("onboarding-ru"),
    }
}
