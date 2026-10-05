//! The macOS front end: menu bar app, HUD, camera, hotkey, paste, Accessibility.

pub mod app;
pub mod camera;
pub mod context;
pub mod hotkey;
pub mod hud;
pub mod login;
pub mod main_thread;
pub mod mic;
pub mod onboarding;
pub mod paste;
pub mod settings_window;
pub mod video;
pub mod widgets;

use std::sync::{Mutex, MutexGuard};

use crate::paths;
use crate::text::cleanup::Cleaner;
use crate::text::personal::Personal;

static CLEANER: Mutex<Option<Cleaner>> = Mutex::new(None);

fn cleaner() -> MutexGuard<'static, Option<Cleaner>> {
    CLEANER.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Pick the cleanup backend ("auto": Claude key → local model → Ollama → offline rules).
pub fn init_cleanup(backend: &str) {
    let personal = Personal::load(&paths::phrases()).unwrap_or_else(|_| Personal::from_phrases(Vec::new()));
    let mut c = Cleaner::new(backend, personal, None);
    c.words_path = paths::words();
    *cleaner() = Some(c);
}

/// Dictation language for the cleanup prompts.
pub fn set_cleanup_lang(lang: crate::text::Lang) {
    if let Some(c) = cleaner().as_mut() {
        c.lang = lang;
    }
}

/// Reload your phrases after an import.
pub fn reload_personal() {
    if let Some(c) = cleaner().as_mut() {
        c.personal = Personal::load(&paths::phrases()).unwrap_or_else(|_| Personal::from_phrases(Vec::new()));
    }
}

pub fn download(url: &str, dest: &std::path::Path, status: &dyn Fn(&str)) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    if dest.metadata().is_ok_and(|m| m.len() > 0) {
        return Ok(());
    }
    if let Some(d) = dest.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut resp = ureq::get(url).call()?;
    let total: u64 = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(0);
    let part = dest.with_extension("part");
    let mut f = std::fs::File::create(&part)?;
    let mut r = resp.body_mut().as_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut done, mut shown) = (0u64, 0u64);
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        f.write_all(&buf[..n])?;
        done += n as u64;
        if total > 0 && done - shown > total / 50 {
            shown = done;
            status(&format!("Downloading the text-cleanup model… {:.0}%", 100.0 * done as f64 / total as f64));
        }
    }
    f.sync_all()?;
    std::fs::rename(&part, dest)?;
    Ok(())
}

/// The on-device cleanup model ("local"): download once (~650 MB), load, and switch the cleaner
/// to it. Used for "auto" when there is no Claude key, as in the Python app. Model worker thread.
pub fn warm_cleanup(backend: &str, status: &dyn Fn(&str)) {
    let has_key = std::env::var_os("ANTHROPIC_API_KEY").is_some() || std::env::var_os("ANTHROPIC_AUTH_TOKEN").is_some();
    if !(backend == "local" || (backend == "auto" && !has_key)) {
        return;
    }
    let result = (|| -> anyhow::Result<()> {
        let dir = paths::home().join("base-models/llm");
        download(lipflow_llm::TOKENIZER_URL, &dir.join("tokenizer.json"), status)?;
        download(lipflow_llm::MODEL_URL, &dir.join(lipflow_llm::MODEL_FILE), status)?;
        status("Loading the text-cleanup model…");
        let mut llm = lipflow_llm::LocalLlm::load(&dir, &lipflow_vsr::reader::pick_device(true))?;
        llm.chat(&[("user".into(), "HELLO".into())], 4)?;
        let llm = Mutex::new(llm);
        let local: crate::text::cleanup::LocalGenerate = Box::new(move |msgs| {
            let mut l = llm.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            l.chat(msgs, 160)
        });
        let personal = Personal::load(&paths::phrases()).unwrap_or_else(|_| Personal::from_phrases(Vec::new()));
        let mut c = Cleaner::new(backend, personal, Some(local));
        c.words_path = paths::words();
        *cleaner() = Some(c);
        Ok(())
    })();
    if let Err(e) = result {
        eprintln!("[lipflow] local cleanup model unavailable ({e:#}); keeping {}", cleanup_desc());
    }
}

pub fn cleanup_desc() -> String {
    cleaner().as_ref().map(Cleaner::describe).unwrap_or_else(|| "basic".into())
}

/// Raw guesses → the sentence to type.
pub fn clean(candidates: &[String], context: &str, names: &[String]) -> String {
    match cleaner().as_ref() {
        Some(c) => c.clean(candidates, context, None, names),
        None => candidates.first().map(|c| crate::text::rules::basic_cleanup(c)).unwrap_or_default(),
    }
}

/// `lipflow doctor`: check everything the app needs before you hit the hotkey.
pub fn doctor() -> anyhow::Result<()> {
    use objc2_av_foundation::AVAuthorizationStatus;
    use objc2_core_graphics::{CGPreflightListenEventAccess, CGPreflightPostEventAccess};
    let mut ok = true;
    let mut line = |good: bool, what: &str, fix: &str| {
        ok &= good;
        println!("  {} {what}{}", if good { "✓" } else { "✗" }, if good { String::new() } else { format!("\n      → {fix}") });
    };
    println!("Lipflow doctor\n");
    let m = paths::models();
    for (rel, size) in [("vsr/model.pth", 900e6), ("lm/model.pth", 200e6), ("lm/unigram5000.model", 3e5), ("face_landmarker.task", 3e6)] {
        let p = m.join(rel);
        let good = std::fs::metadata(&p).is_ok_and(|md| md.len() as f64 > size);
        line(good, &format!("model file {rel} ({})", p.display()), "run scripts/setup.sh");
    }
    let gpu = lipflow_vsr::reader::pick_device(true);
    line(true, &format!("encoder on {}", if gpu.is_metal() { "Apple GPU (Metal)" } else { "CPU" }), "");
    line(CGPreflightListenEventAccess(), "Input Monitoring (for the push-to-talk key)", "System Settings → Privacy & Security → Input Monitoring → enable Lipflow, then restart it");
    line(CGPreflightPostEventAccess(), "Accessibility (to paste at your cursor)", "System Settings → Privacy & Security → Accessibility → enable Lipflow");
    let st = camera::authorization();
    let name = match st {
        AVAuthorizationStatus::NotDetermined => "not asked yet (you'll be prompted on first use)",
        AVAuthorizationStatus::Restricted => "restricted",
        AVAuthorizationStatus::Denied => "denied",
        _ => "granted",
    };
    line(st == AVAuthorizationStatus::NotDetermined || st == AVAuthorizationStatus::Authorized, &format!("Camera: {name}"), "System Settings → Privacy & Security → Camera → enable Lipflow");
    init_cleanup("auto");
    let mut desc = cleanup_desc();
    let has_key = std::env::var_os("ANTHROPIC_API_KEY").is_some() || std::env::var_os("ANTHROPIC_AUTH_TOKEN").is_some();
    if !has_key && !desc.starts_with("ollama") {
        let llm = paths::home().join("base-models/llm").join(lipflow_llm::MODEL_FILE);
        desc = format!("local ({}{})", lipflow_llm::MODEL_FILE, if llm.exists() { "" } else { ", downloaded on first launch" });
    }
    let hint = if desc.starts_with("basic") { "  (set ANTHROPIC_API_KEY or run Ollama for much better accuracy)" } else { "" };
    line(true, &format!("cleanup backend: {desc}{hint}"), "");
    println!("{}", if ok { "\nAll good." } else { "\nFix the ✗ items above, then run `lipflow`." });
    if ok { Ok(()) } else { anyhow::bail!("some checks failed") }
}
