//! lipflow [run] | lipflow file VIDEO | lipflow doctor | …

mod paths;
mod audio;
mod data;
mod mouth_view;
mod pipeline;
mod ptt;
mod text;
mod train;

#[cfg(target_os = "macos")]
mod macos;

use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use lipflow_face::FaceLandmarker;
use lipflow_vsr::{LipReader, ReaderOptions};

struct Args {
    cmd: String,
    rest: Vec<String>,
}

impl Args {
    fn parse() -> Self {
        let mut v: Vec<String> = std::env::args().skip(1).collect();
        let known = ["run", "file", "ru-file", "convert-ru", "bench", "selftest", "train", "clean", "doctor", "onboard", "train-lm", "import-wispr", "-h", "--help"];
        let cmd = if v.first().is_some_and(|c| known.contains(&c.as_str())) { v.remove(0) } else { "run".into() };
        Self { cmd, rest: v }
    }

    fn flag(&self, name: &str) -> bool {
        self.rest.iter().any(|a| a == name)
    }

    fn value(&self, name: &str) -> Option<&str> {
        self.rest.iter().position(|a| a == name).and_then(|i| self.rest.get(i + 1)).map(String::as_str)
    }

    fn positional(&self) -> Option<&str> {
        let mut skip = false;
        for a in &self.rest {
            if skip {
                skip = false;
                continue;
            }
            if a.starts_with("--") {
                skip = !matches!(a.as_str(), "--copy-only" | "--no-preview" | "--cpu");
                continue;
            }
            return Some(a);
        }
        None
    }
}

const USAGE: &str = "lipflow — silent dictation by lip reading

  lipflow [run] [--key K] [--beam N] [--cleanup auto|claude|local|ollama|basic] [--camera C] [--copy-only] [--no-preview]
  lipflow file VIDEO [--start S] [--end E] [--beam N] [--cleanup …|none]
  lipflow doctor
  lipflow onboard | import-wispr [--from-text FILE]
  lipflow train | train-lm          (run the Python app's training; the app loads the weights)";

pub fn reader_options(beam: usize) -> ReaderOptions {
    let mut o = ReaderOptions::new(paths::models());
    o.beam.beam = beam;
    o.beam_gpu = true;
    o.personal_vsr = Some(paths::personal_vsr());
    o.personal_lm = Some(paths::personal_lm());
    o
}

fn face_model() -> Result<FaceLandmarker> {
    FaceLandmarker::load(&paths::models().join("face_landmarker.task"))
}

#[cfg(target_os = "macos")]
fn load_recording(video: &std::path::Path, start: f64, end: Option<f64>) -> Result<pipeline::Recording> {
    load_recording_into(video, start, end, pipeline::Recording::new())
}

/// Track a video into `rec` frame by frame (frames aren't kept: only the recording's crops).
#[cfg(target_os = "macos")]
fn load_recording_into(video: &std::path::Path, start: f64, end: Option<f64>, rec: pipeline::Recording) -> Result<pipeline::Recording> {
    let mut lm = face_model()?;
    let mut rec = Some(rec);
    macos::video::read_frames(video, start, end, |f| {
        let r = rec.take().context("recording")?;
        rec = Some(pipeline::track_clip(&mut lm, r, std::iter::once((f.t, f.bgra, f.width, f.height)))?);
        Ok(())
    })?;
    rec.context("recording")
}

#[cfg(target_os = "macos")]
fn cmd_file(a: &Args) -> Result<()> {
    let video = a.positional().context("usage: lipflow file VIDEO [--start S] [--end E]")?;
    let start: f64 = a.value("--start").map_or(Ok(0.0), str::parse)?;
    let end: Option<f64> = a.value("--end").map(str::parse).transpose()?;
    let beam: usize = a.value("--beam").map_or(Ok(10), str::parse)?;
    let t0 = Instant::now();
    let rec = load_recording(std::path::Path::new(video), start, end)?;
    let t_face = t0.elapsed().as_secs_f64();
    let (rois, t) = rec.rois().context("no face found in the video")?;
    let reader = LipReader::load(&reader_options(beam))?;
    let t1 = Instant::now();
    let enc = reader.encode(&rois, t)?;
    let cands = reader.beam_search(&enc, 5)?;
    let found = rec.anchors.iter().filter(|x| x.is_some()).count();
    println!("[{t} frames @25fps, face in {found}/{}, track {t_face:.2}s, decode {:.2}s]", rec.anchors.len(), t1.elapsed().as_secs_f64());
    println!("raw:   {}", cands.first().cloned().unwrap_or_default());
    let backend = a.value("--cleanup").unwrap_or("auto");
    if backend != "none" {
        macos::init_cleanup(backend);
        macos::warm_cleanup(backend, &|m| eprintln!("{m}"));
        let t2 = Instant::now();
        let out = macos::clean(&cands, "", &[]);
        println!("text:  {out}   [{}, {:.2}s]", macos::cleanup_desc(), t2.elapsed().as_secs_f64());
    }
    Ok(())
}

/// Russian lip reading of a video file with MultiVSR, through the app's own recording path.
///   lipflow ru-file VIDEO [--start S] [--end E] [--beam N] [--boxes OUT.csv]
#[cfg(target_os = "macos")]
fn cmd_ru_file(a: &Args) -> Result<()> {
    let video = std::path::Path::new(a.positional().context("usage: lipflow ru-file VIDEO")?);
    let start: f64 = a.value("--start").map_or(Ok(0.0), str::parse)?;
    let end: Option<f64> = a.value("--end").map(str::parse).transpose()?;
    let beam: usize = a.value("--beam").map_or(Ok(5), str::parse)?;
    let t0 = Instant::now();
    let rec = load_recording_into(video, start, end, pipeline::Recording::with_faces())?;
    if let Some(out) = a.value("--boxes") {
        // the raw landmark-derived boxes of the 25 fps frames, for tools/face_box_fit.py
        let faces = rec.faces.as_ref().context("faces")?;
        let mut csv = String::from("frame,cx,cy,s\n");
        for (k, &i) in LipReader::resample(&rec.ts).iter().enumerate() {
            if let Some((b, _)) = &faces[i] {
                csv += &format!("{k},{},{},{}\n", b.cx, b.cy, b.s);
            }
        }
        std::fs::write(out, csv)?;
    }
    let (frames, t) = rec.face_frames().context("no face found in the video")?;
    let t_face = t0.elapsed().as_secs_f64();
    if let (Some(dir), Some(text)) = (a.value("--save-npz"), a.value("--text")) {
        // a practice clip, as onboarding saves it (for testing the trainer)
        std::fs::create_dir_all(dir)?;
        let p = data::save_clip(std::path::Path::new(dir), &frames, t, text, &[], &[])?;
        println!("saved {}", p.display());
        return Ok(());
    }
    let dev = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
    let t1 = Instant::now();
    let m = lipflow_vsr::multivsr::MultiVsr::load(&paths::multivsr(), &dev)?;
    let t_load = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let enc = m.encode_frames(&frames, t)?;
    let text = m.read(&enc, beam)?;
    println!("[{t} frames, track+crop {t_face:.2}s, load {t_load:.2}s, read {:.2}s]", t2.elapsed().as_secs_f64());
    println!("{text}");
    Ok(())
}

/// Full pipeline (video → face → crops → text) on the public-domain sample clips' sentences.
#[cfg(target_os = "macos")]
fn cmd_bench(a: &Args) -> Result<()> {
    let samples = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lipflow/samples");
    let beam: usize = a.value("--beam").map_or(Ok(4), str::parse)?;
    let reader = LipReader::load(&reader_options(beam))?;
    reader.warmup()?;
    let (mut errs, mut n, mut t_track, mut t_dec, mut clips) = (0usize, 0usize, 0f64, 0f64, 0usize);
    for name in ["2016-03-12", "2017-01-07"] {
        let srt = std::fs::read_to_string(samples.join(format!("{name}.srt"))).context("run setup with samples")?;
        for (s, e, truth) in text::srt_sentences(&srt) {
            let t0 = Instant::now();
            let rec = load_recording(&samples.join(format!("{name}.mov")), s, Some(e))?;
            t_track += t0.elapsed().as_secs_f64();
            if rec.ts.is_empty() || rec.face_ratio() < 0.8 {
                continue;
            }
            let Some((rois, t)) = rec.rois() else { continue };
            let t1 = Instant::now();
            let hyp = reader.beam_search(&reader.encode(&rois, t)?, 5)?.into_iter().next().unwrap_or_default();
            t_dec += t1.elapsed().as_secs_f64();
            let (e_, m) = text::wer(&hyp, &truth);
            errs += e_;
            n += m;
            clips += 1;
            println!("{clips:3} T={t:3} {hyp}");
        }
    }
    println!("{clips} clips  WER {:.2}%  decode/clip {:.3}s  video+face/clip {:.3}s", 100.0 * errs as f64 / n.max(1) as f64, t_dec / clips.max(1) as f64, t_track / clips.max(1) as f64);
    Ok(())
}

fn real_main() -> Result<()> {
    let a = Args::parse();
    match a.cmd.as_str() {
        "-h" | "--help" => {
            println!("{USAGE}");
            Ok(())
        }
        #[cfg(target_os = "macos")]
        "file" => cmd_file(&a),
        #[cfg(target_os = "macos")]
        "ru-file" => cmd_ru_file(&a),
        "convert-ru" => {
            // the released MultiVSR checkpoints (downloaded by scripts/setup.sh) → multivsr.safetensors
            let dir = a.positional().map_or_else(paths::multivsr, std::path::PathBuf::from);
            let (model, vtp) = (dir.join("model.pth"), dir.join("feature_extractor.pth"));
            let n = lipflow_vsr::multivsr::convert_checkpoints(&model, &vtp, &dir.join("multivsr.safetensors"))?;
            println!("wrote {} ({n} tensors)", dir.join("multivsr.safetensors").display());
            if !a.flag("--keep") {
                std::fs::remove_file(&model)?;
                std::fs::remove_file(&vtp)?;
            }
            Ok(())
        }
        #[cfg(target_os = "macos")]
        "bench" => cmd_bench(&a),
        #[cfg(target_os = "macos")]
        "import-wispr" => {
            let (phrases, words) = (paths::phrases(), paths::words());
            let stats = match a.value("--from-text") {
                Some(f) => {
                    let lines: Vec<String> = std::fs::read_to_string(f)?.lines().map(str::to_string).collect();
                    text::personal::save_phrases(&lines, &phrases, &words)?
                }
                None => text::personal::import_wispr(&text::personal::wispr_dir(), &phrases, &words)?,
            };
            println!("Imported {} phrases ({} words) from {}", stats.phrases, stats.words, stats.source.as_deref().unwrap_or("text"));
            println!("  saved to {}", phrases.display());
            if !stats.new_names.is_empty() {
                println!("  added {} names/terms to {}. Review them: Lipflow menu → Edit custom words", stats.new_names.len(), words.display());
            }
            println!("Restart Lipflow to use them.");
            Ok(())
        }
        #[cfg(target_os = "macos")]
        "doctor" => macos::doctor(),
        #[cfg(target_os = "macos")]
        "clean" => {
            let raw = a.positional().context("usage: lipflow clean \"RAW GUESS\" [--cleanup B]")?.to_string();
            let backend = a.value("--cleanup").unwrap_or("auto");
            macos::init_cleanup(backend);
            macos::warm_cleanup(backend, &|m| eprintln!("{m}"));
            for _ in 0..3 {
                let t = Instant::now();
                let out = macos::clean(std::slice::from_ref(&raw), "", &[]);
                println!("{out}   [{}, {:.3}s]", macos::cleanup_desc(), t.elapsed().as_secs_f64());
            }
            Ok(())
        }
        "train-lm" => train::train_lm(),
        "train" => {
            let beam: usize = a.value("--beam").map_or(Ok(4), str::parse)?;
            let r = train::train_on_face(beam, &|pct, msg| println!("[{pct:3.0}%] {msg}"))?;
            match r.after {
                Some(after) => println!("held-out WER {:.1}% → {:.1}% ({}), {} clips", r.before * 100.0, after * 100.0, if r.kept { "kept" } else { "discarded" }, r.clips),
                None => println!("{}", r.note),
            }
            Ok(())
        }
        #[cfg(target_os = "macos")]
        "selftest" => {
            let video = a.positional().context("usage: lipflow selftest VIDEO --start S --end E")?;
            let start: f64 = a.value("--start").map_or(Ok(0.0), str::parse)?;
            let end: f64 = a.value("--end").map_or(Ok(start + 5.0), str::parse)?;
            macos::app::run(macos::app::Options {
                selftest: Some(end - start),
                camera: format!("file:{video}@{start}"),
                paste: false,
                ..Default::default()
            })
        }
        "run" => macos::app::run(macos::app::Options {
            selftest: None,
            key: a.value("--key").unwrap_or("right_option").to_string(),
            beam: a.value("--beam").map_or(Ok(4), str::parse)?,
            backend: a.value("--cleanup").unwrap_or("auto").to_string(),
            camera: a.value("--camera").unwrap_or("auto").to_string(),
            paste: !a.flag("--copy-only"),
            live_preview: !a.flag("--no-preview"),
            onboard: false,
        }),
        #[cfg(target_os = "macos")]
        "onboard" => macos::app::run(macos::app::Options { onboard: true, ..Default::default() }),
        other => bail!("`{other}` is not available yet\n\n{USAGE}"),
    }
}

/// Started from Lipflow.app there is no terminal: append stdout/stderr to ~/Library/Logs/Lipflow.log.
#[cfg(unix)]
fn log_to_file_in_bundle() {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn dup2(src: i32, dst: i32) -> i32;
    }
    let in_bundle = std::env::current_exe().is_ok_and(|p| p.to_string_lossy().contains(".app/Contents/MacOS/"));
    if !in_bundle {
        return;
    }
    let Some(home) = std::env::var_os("HOME") else { return };
    let dir = std::path::PathBuf::from(home).join("Library/Logs");
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("Lipflow.log")) {
        let fd = f.as_raw_fd();
        // SAFETY: `fd` is a valid open file descriptor for the duration of the calls; dup2 makes
        // 1 and 2 independent duplicates, so dropping `f` afterwards leaves them open.
        unsafe {
            dup2(fd, 1);
            dup2(fd, 2);
        }
    }
}

#[cfg(not(unix))]
fn log_to_file_in_bundle() {}

fn main() -> ExitCode {
    log_to_file_in_bundle();
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lipflow: {e:#}");
            ExitCode::FAILURE
        }
    }
}
