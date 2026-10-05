//! The menu-bar app (`lipflow/app.py`): hold a key, mouth the words, let go — the text appears
//! at your cursor.
//!
//! Threads: AppKit main thread (menu, HUD, hotkey tap, timers) · camera control + frame queue ·
//! one model worker that owns the lip reader and runs jobs in order · a preview ticker per
//! recording. Worker → UI goes through `with_ui`, which hops onto the main thread.

use std::cell::{Cell, OnceCell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::Result;
use lipflow_face::FaceLandmarker;
use lipflow_vsr::LipReader;
use lipflow_vsr::multivsr::MultiVsr;
use candle_core::Tensor;
use anyhow::Context as _;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject, Sel};
use objc2::{MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSControlStateValueOff, NSControlStateValueOn, NSFontWeightRegular, NSMenu,
    NSMenuItem, NSStatusBar, NSStatusItem, NSVariableStatusItemLength,
};
use objc2_av_foundation::AVAuthorizationStatus;
use objc2_core_graphics::{CGPreflightPostEventAccess, CGRequestListenEventAccess, CGRequestPostEventAccess};
use objc2_foundation::{MainThreadMarker, NSString};

use super::camera::{self, Camera, FrameEvent};
use super::hotkey::{Hotkey, KeyEvent};
use super::hud::{Hud, Mode, symbol};
use super::main_thread::{on_main, on_main_after};
use super::paste::{copy_text, paste_text};
use crate::data::{self, Settings};
use crate::mouth_view::mouth_view;
use crate::pipeline::{JOIN_WINDOW, KEEP_CLIPS, MAX_SECONDS, PREVIEW_EVERY, Recording, TAIL_SECONDS};
use crate::ptt::{Action, PushToTalk};
use crate::{paths, text};

#[derive(Clone, Debug)]
pub struct Options {
    /// Self-test: record this many seconds as soon as the model is loaded, print, quit.
    pub selftest: Option<f64>,
    pub key: String,
    pub beam: usize,
    pub backend: String,
    pub camera: String,
    pub paste: bool,
    pub live_preview: bool,
    pub onboard: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { selftest: None, key: "right_option".into(), beam: 4, backend: "auto".into(), camera: "auto".into(), paste: true, live_preview: true, onboard: false }
    }
}

enum Job {
    Load,
    Preview(u64),
    Final(Box<Recording>, Option<crate::audio::Chunks>),
    Whisper,
    Reload,
    Train,
}

/// State shared with the worker and camera threads.
pub struct Core {
    pub camera: Arc<Camera>,
    jobs: SyncSender<Job>,
    session: AtomicU64,
    preview_busy: AtomicBool,
    loading: AtomicBool,
    ui_busy: AtomicBool,
    pub onboarding: AtomicBool,
    /// The audio-visual model is loaded (whisper mode can run).
    av_ready: AtomicBool,
    /// The language being read (set when the model loads).
    pub lang: Mutex<text::Lang>,
    pub settings: Mutex<Settings>,
    opts: Mutex<Options>,
    last_output: Mutex<String>,
    last_paste_at: Mutex<Option<Instant>>,
    context: Mutex<Vec<String>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Main-thread state.
pub struct Ui {
    pub mtm: MainThreadMarker,
    pub core: Arc<Core>,
    pub hud: Hud,
    status: Retained<NSStatusItem>,
    state_item: RefCell<Option<Retained<NSMenuItem>>>,
    cleanup_item: RefCell<Option<Retained<NSMenuItem>>>,
    cam_items: RefCell<Vec<Retained<NSMenuItem>>>,
    target: Retained<MenuTarget>,
    hotkey: RefCell<Option<Hotkey>>,
    ptt: RefCell<PushToTalk>,
    pending_stop: Cell<Option<u64>>,
    hands_free: Cell<bool>,
    mic: RefCell<super::mic::Mic>,
}

thread_local! {
    static UI: OnceCell<&'static Ui> = const { OnceCell::new() };
}

pub fn ui() -> Option<&'static Ui> {
    UI.with(|u| u.get().copied())
}

/// Run `f` with the UI on the main thread.
pub fn with_ui(f: impl FnOnce(&'static Ui) + Send + 'static) {
    on_main(move || {
        if let Some(u) = ui() {
            f(u);
        }
    });
}

fn hud(mode: Mode, title: &str, body: &str, hide_after: Option<f64>) {
    let (title, body) = (title.to_string(), body.to_string());
    with_ui(move |u| u.hud.show(mode, &title, &body, hide_after));
}

fn log(msg: &str) {
    println!("[lipflow] {msg}");
}

// -- menu target ---------------------------------------------------------------------

define_class!(
    // SAFETY: NSObject subclass, main thread only, no Drop.
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    pub struct MenuTarget;

    unsafe impl NSObjectProtocol for MenuTarget {}

    impl MenuTarget {
        #[unsafe(method(copyLast:))]
        fn copy_last(&self, _sender: Option<&AnyObject>) {
            if let Some(u) = ui() {
                let t = lock(&u.core.last_output).clone();
                if !t.is_empty() {
                    copy_text(&t);
                }
            }
        }

        #[unsafe(method(openHistory:))]
        fn open_history(&self, _sender: Option<&AnyObject>) {
            let p = paths::history();
            if let Some(d) = p.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(&p);
            open_text(&p);
        }

        #[unsafe(method(openWords:))]
        fn open_words(&self, _sender: Option<&AnyObject>) {
            let _ = text::vocab::load(&paths::words());
            open_text(&paths::words());
        }

        #[unsafe(method(openData:))]
        fn open_data(&self, _sender: Option<&AnyObject>) {
            let _ = std::process::Command::new("open").arg(paths::home()).spawn();
        }

        #[unsafe(method(openPermissions:))]
        fn open_permissions(&self, _sender: Option<&AnyObject>) {
            if let Some(u) = ui() {
                super::onboarding::show_permissions(u);
            }
        }

        #[unsafe(method(openSettings:))]
        fn open_settings(&self, _sender: Option<&AnyObject>) {
            if let Some(u) = ui() {
                super::settings_window::show(u);
            }
        }

        #[unsafe(method(trainMore:))]
        fn train_more(&self, _sender: Option<&AnyObject>) {
            if let Some(u) = ui() {
                super::onboarding::show(u, true);
            }
        }

        #[unsafe(method(pickCamera:))]
        fn pick_camera(&self, sender: Option<&NSMenuItem>) {
            let (Some(u), Some(item)) = (ui(), sender) else { return };
            let Some(obj) = item.representedObject() else { return };
            let Ok(id) = obj.downcast::<NSString>() else { return };
            u.set_camera(&id.to_string());
        }

        #[unsafe(method(quit:))]
        fn quit(&self, _sender: Option<&AnyObject>) {
            if let Some(u) = ui() {
                u.core.camera.close();
                NSApplication::sharedApplication(u.mtm).terminate(None);
            }
        }
    }

    unsafe impl NSApplicationDelegate for MenuTarget {
        /// Opening Lipflow.app while it runs brings up Settings (setup the first time): the
        /// menu-bar icon can hide behind the notch.
        #[unsafe(method(applicationShouldHandleReopen:hasVisibleWindows:))]
        fn should_reopen(&self, _app: &NSApplication, _visible: bool) -> bool {
            if let Some(u) = ui()
                && !u.core.loading.load(Ordering::Acquire)
            {
                if lock(&u.core.settings).flag("onboarded", false) {
                    super::settings_window::show(u);
                } else {
                    super::onboarding::show(u, false);
                }
            }
            false
        }
    }
);

fn open_text(p: &std::path::Path) {
    let _ = std::process::Command::new("open").arg("-t").arg(p).spawn();
}

impl Ui {
    fn item(&self, menu: &NSMenu, title: &str, action: Option<Sel>, key: &str, icon: Option<&str>) -> Retained<NSMenuItem> {
        // SAFETY: NSMenuItem's designated initializer with a valid selector (or none).
        let it = unsafe { NSMenuItem::initWithTitle_action_keyEquivalent(NSMenuItem::alloc(self.mtm), &NSString::from_str(title), action, &NSString::from_str(key)) };
        if let Some(icon) = icon {
            // SAFETY: reading an immutable AppKit constant.
            it.setImage(symbol(icon, 13.0, unsafe { NSFontWeightRegular }).as_deref());
        }
        if action.is_some() {
            // SAFETY: the target outlives the menu (both live for the app's lifetime).
            unsafe { it.setTarget(Some(&self.target)) };
        } else {
            it.setEnabled(false);
        }
        menu.addItem(&it);
        it
    }

    fn set_status_icon(&self, listening: bool) {
        if let Some(button) = self.status.button(self.mtm) {
            let img = symbol(if listening { "mouth.fill" } else { "mouth" }, 15.0, super::hud::weight_semibold());
            if let Some(img) = &img {
                img.setTemplate(true); // follows the light/dark menu bar
            }
            button.setImage(img.as_deref());
        }
    }

    fn build_menu(&'static self) {
        let menu = NSMenu::new(self.mtm);
        *self.state_item.borrow_mut() = Some(self.item(&menu, "Loading model…", None, "", Some("hourglass")));
        let key = lock(&self.core.opts).key.replace('_', " ");
        let key = title_case(&key);
        self.item(&menu, &format!("Hold {key} to dictate, double-tap for hands-free"), None, "", Some("keyboard"));
        *self.cleanup_item.borrow_mut() = Some(self.item(&menu, &format!("Cleanup: {}", super::cleanup_desc()), None, "", Some("text.badge.checkmark")));
        let n = text::personal::Personal::load(&paths::phrases()).map_or(0, |p| p.phrases().len());
        self.item(
            &menu,
            &if n > 0 { format!("Personalised from {n} of your phrases") } else { "Not personalised yet: run lipflow import-wispr".into() },
            None,
            "",
            Some("person.text.rectangle"),
        );
        menu.addItem(&NSMenuItem::separatorItem(self.mtm));
        self.camera_menu(&menu);
        self.item(&menu, "Copy last dictation", Some(sel!(copyLast:)), "", Some("doc.on.clipboard"));
        self.item(&menu, "Open history", Some(sel!(openHistory:)), "", Some("clock.arrow.circlepath"));
        self.item(&menu, "Edit custom words…", Some(sel!(openWords:)), "", Some("character.book.closed"));
        self.item(&menu, "Practice & train more…", Some(sel!(trainMore:)), "", Some("person.crop.square"));
        self.item(&menu, "Check permissions…", Some(sel!(openPermissions:)), "", Some("lock.shield"));
        self.item(&menu, "Settings…", Some(sel!(openSettings:)), ",", Some("gearshape"));
        menu.addItem(&NSMenuItem::separatorItem(self.mtm));
        self.item(&menu, "Quit Lipflow", Some(sel!(quit:)), "q", Some("power"));
        self.status.setMenu(Some(&menu));
    }

    fn camera_menu(&self, menu: &NSMenu) {
        let parent = self.item(menu, "Camera", None, "", Some("camera"));
        parent.setEnabled(true);
        let sub = NSMenu::new(self.mtm);
        let current = camera::resolve_camera(&lock(&self.core.camera.source).clone()).map(|c| c.id);
        let mut items = Vec::new();
        for c in camera::list_cameras() {
            let it = self.item(&sub, &c.name, Some(sel!(pickCamera:)), "", None);
            let id = NSString::from_str(&c.id);
            // SAFETY: storing an NSString as the represented object.
            unsafe { it.setRepresentedObject(Some(&id)) };
            it.setState(if current.as_deref() == Some(c.id.as_str()) { NSControlStateValueOn } else { NSControlStateValueOff });
            items.push(it);
        }
        parent.setSubmenu(Some(&sub));
        *self.cam_items.borrow_mut() = items;
    }

    fn set_cleanup_label(&self, desc: &str) {
        if let Some(it) = self.cleanup_item.borrow().as_ref() {
            it.setTitle(&NSString::from_str(&format!("Cleanup: {desc}")));
        }
    }

    pub fn set_camera(&self, id: &str) {
        {
            let mut s = lock(&self.core.settings);
            s.set("camera", serde_json::Value::String(id.to_string()));
            let _ = s.save();
        }
        self.core.camera.set_source(id);
        for it in self.cam_items.borrow().iter() {
            let on = it.representedObject().and_then(|o| o.downcast::<NSString>().ok()).is_some_and(|s| s.to_string() == id);
            it.setState(if on { NSControlStateValueOn } else { NSControlStateValueOff });
        }
        log(&format!("camera: {id}"));
    }

    pub fn set_key(&self, key: &str) {
        if let Some(h) = self.hotkey.borrow().as_ref()
            && h.set_key(key)
        {
            lock(&self.core.opts).key = key.to_string();
            let mut s = lock(&self.core.settings);
            s.set("key", serde_json::Value::String(key.to_string()));
            let _ = s.save();
        }
    }

    pub fn key(&self) -> String {
        lock(&self.core.opts).key.clone()
    }

    pub fn reload_model(&self) {
        let _ = self.core.jobs.try_send(Job::Reload);
    }

    // -- hotkey ---------------------------------------------------------------------
    fn on_key(&'static self, ev: KeyEvent) {
        let now = camera::now();
        let action = {
            let mut p = self.ptt.borrow_mut();
            match ev {
                KeyEvent::PttDown => p.key_down(now),
                KeyEvent::PttUp => p.key_up(now),
                KeyEvent::Other { esc } => p.other_key(esc),
            }
        };
        match action {
            Some(Action::Start { hands_free }) => self.on_start(hands_free),
            Some(Action::Stop) => self.on_stop(),
            Some(Action::Cancel { silent }) => self.on_cancel(silent),
            None => {}
        }
    }

    pub fn on_start(&'static self, hands_free: bool) {
        let core = &self.core;
        if core.loading.load(Ordering::Acquire) {
            self.hud.show(Mode::Error, "Still loading", "The model is almost ready…", Some(1.5));
            return;
        }
        if let Some(tok) = self.pending_stop.get() {
            self.finish_stop(tok); // pressed again during the tail: finish the last one now
        }
        if hands_free && core.camera.is_recording() {
            self.hands_free.set(true); // the second tap of a double-tap: keep recording
            self.hud.show(Mode::Listening, "Hands-free · tap to finish", &self.hud.body_text(), None);
            return;
        }
        let session = core.session.fetch_add(1, Ordering::AcqRel) + 1;
        self.hands_free.set(hands_free);
        if lock(&core.settings).flag("use_context", true) {
            super::context::capture_into(core);
        }
        core.camera.start_recording();
        if self.whisper_on() {
            self.mic.borrow_mut().start();
        }
        self.set_status_icon(true);
        let title = if hands_free { "Hands-free · tap to finish" } else { "Listening" };
        self.hud.show(Mode::Listening, title, if core.camera.ready() { "" } else { "Starting camera…" }, None);
        let c = core.clone();
        let _ = std::thread::Builder::new().name("lipflow-preview".into()).spawn(move || preview_loop(&c, session));
    }

    pub fn on_stop(&'static self) {
        self.hands_free.set(false);
        let core = &self.core;
        if !core.camera.is_recording() {
            return;
        }
        let tok = core.session.fetch_add(1, Ordering::AcqRel) + 1;
        self.pending_stop.set(Some(tok));
        self.set_status_icon(false);
        self.hud.show(Mode::Reading, "Reading your lips", &self.hud.body_text(), None);
        on_main_after(TAIL_SECONDS, move || {
            if let Some(u) = ui() {
                u.finish_stop(tok);
            }
        });
    }

    fn finish_stop(&self, tok: u64) {
        if self.pending_stop.get() != Some(tok) {
            return;
        }
        self.pending_stop.set(None);
        let audio = self.whisper_on().then(|| self.mic.borrow_mut().stop());
        if let Some(rec) = self.core.camera.stop_recording() {
            let _ = self.core.jobs.try_send(Job::Final(Box::new(rec), audio));
        }
    }

    /// Whisper mode is on, its model is loaded, and this isn't a practice clip.
    fn whisper_on(&self) -> bool {
        let c = &self.core;
        lock(&c.settings).flag("whisper", false)
            && c.av_ready.load(Ordering::Acquire)
            && !c.onboarding.load(Ordering::Acquire)
            && *lock(&c.lang) == text::Lang::En
    }

    pub fn on_cancel(&'static self, silent: bool) {
        self.pending_stop.set(None);
        self.set_status_icon(false);
        self.core.session.fetch_add(1, Ordering::AcqRel);
        self.hands_free.set(false);
        self.core.camera.stop_recording();
        let _ = self.mic.borrow_mut().stop();
        if silent {
            self.hud.hide();
        } else {
            self.hud.show(Mode::Error, "Cancelled", "", Some(0.8));
        }
    }
}

fn title_case(s: &str) -> String {
    s.split(' ')
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// -- camera frames (camera queue) --------------------------------------------------------

fn on_frame(core: &Arc<Core>, ev: &FrameEvent) {
    if ev.recording && ev.rec_duration > MAX_SECONDS {
        with_ui(|u| u.on_stop());
        return;
    }
    let onboarding = core.onboarding.load(Ordering::Relaxed);
    if !ev.recording && !onboarding {
        return;
    }
    if core.ui_busy.swap(true, Ordering::AcqRel) {
        return; // coalesce: at most one frame waits on the main thread
    }
    let setup = onboarding.then(|| mouth_view(ev.frame, ev.face, 208, 130));
    let pill = ev.recording.then(|| mouth_view(ev.frame, ev.face, 240, 150));
    let level = ev.face.map_or(0.0, |f| f64::from(lipflow_face::mouth_open(&f.points)) * 2.6);
    let c = core.clone();
    with_ui(move |u| {
        if let Some(v) = setup {
            super::onboarding::set_frame(v);
        }
        if pill.is_some() {
            u.hud.set_frame(pill, level);
        }
        c.ui_busy.store(false, Ordering::Release);
    });
}

// -- worker --------------------------------------------------------------------------

fn preview_loop(core: &Arc<Core>, session: u64) {
    if !lock(&core.opts).live_preview {
        return;
    }
    let mut waited = 0.0;
    while core.session.load(Ordering::Acquire) == session {
        std::thread::sleep(Duration::from_secs_f64(PREVIEW_EVERY));
        waited += PREVIEW_EVERY;
        let frames = core.camera.with_recording(|r| r.ts.len()).unwrap_or(0);
        if frames == 0 && (core.camera.error().is_some() || waited > 4.0) {
            let msg = core.camera.error().unwrap_or_else(|| "The camera isn't sending frames".into());
            log(&format!("camera problem: {msg} ({})", core.camera.diagnostics()));
            hud(Mode::Error, "Camera problem", &msg.chars().take(90).collect::<String>(), Some(5.0));
            return;
        }
        if core.session.load(Ordering::Acquire) != session || core.preview_busy.load(Ordering::Acquire) || frames < 15 {
            continue;
        }
        core.preview_busy.store(true, Ordering::Release);
        if core.jobs.try_send(Job::Preview(session)).is_err() {
            core.preview_busy.store(false, Ordering::Release);
        }
    }
}

struct Worker {
    core: Arc<Core>,
    reader: Option<LipReader>,
    av_reader: Option<LipReader>,
    /// MultiVSR, when the language is Russian (then `reader` stays empty).
    ru: Option<MultiVsr>,
    /// Russian: VTP features of the frames of the current recording that can't change any
    /// more, computed during the previews so the final read only adds the last few frames.
    /// Keyed by the recording's first timestamp.
    feats: RefCell<Option<(u64, Tensor)>>,
}

/// A frame's crop is final once 6 later frames fix its smoothed box, and its features once 2
/// more frames give the time kernel its context.
const RU_SETTLED: usize = 8;

/// The loaded lip reader for the current language.
enum Model<'a> {
    En(&'a LipReader),
    Ru(&'a MultiVsr),
}

impl Worker {
    fn run(mut self, rx: Receiver<Job>) {
        while let Ok(job) = rx.recv() {
            let is_preview = matches!(job, Job::Preview(_));
            if let Err(e) = self.handle(job) {
                eprintln!("[lipflow] {e:#}");
                hud(Mode::Error, "Something went wrong", &e.to_string().chars().take(80).collect::<String>(), Some(3.0));
            }
            if is_preview {
                self.core.preview_busy.store(false, Ordering::Release);
            }
        }
    }

    fn load_reader(&self) -> Result<LipReader> {
        let beam = lock(&self.core.opts).beam;
        let r = LipReader::load(&crate::reader_options(beam))?;
        r.warmup()?;
        Ok(r)
    }

    /// Load the reader for the language in the settings (dropping the other one).
    fn load_model(&mut self) -> Result<()> {
        let lang = {
            let s = lock(&self.core.settings);
            text::Lang::from_setting(s.str("language"), paths::multivsr().join("multivsr.safetensors").exists())
        };
        if lang == text::Lang::Ru {
            self.reader = None;
            let dev = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
            let personal = paths::personal_multivsr();
            let m = MultiVsr::load_personal(&paths::multivsr(), Some(&personal), &dev)?;
            if personal.exists() {
                log("Russian model: using your face model");
            }
            m.warmup()?;
            self.ru = Some(m);
        } else {
            self.ru = None;
            self.reader = Some(self.load_reader()?);
        }
        *lock(&self.core.lang) = lang;
        self.core.camera.set_faces(lang == text::Lang::Ru);
        super::set_cleanup_lang(lang);
        Ok(())
    }

    /// Features of the whole clip (frames, t) of the recording keyed `key`: settled ones from the
    /// cache (extended and stored when `keep`), the rest computed now.
    fn ru_features(&self, key: u64, frames: &[u8], t: usize, keep: bool) -> Result<Tensor> {
        let m = self.ru.as_ref().context("Russian model not loaded")?;
        let mut cached = match self.feats.borrow_mut().take() {
            Some((k, f)) if k == key => Some(f),
            _ => None,
        };
        let n = cached.as_ref().map_or(Ok(0), |f| f.dim(0))?;
        let settled = t.saturating_sub(RU_SETTLED);
        if keep && settled > n {
            let new = m.features(frames, t, n, settled)?;
            cached = Some(match cached {
                Some(c) => Tensor::cat(&[&c, &new], 0)?,
                None => new,
            });
        }
        let n = cached.as_ref().map_or(Ok(0), |f| f.dim(0))?;
        let all = match (&cached, n < t) {
            (Some(c), true) => Tensor::cat(&[c, &m.features(frames, t, n, t)?], 0)?,
            (Some(c), false) => c.clone(),
            (None, _) => m.features(frames, t, 0, t)?,
        };
        if keep && let Some(c) = cached {
            *self.feats.borrow_mut() = Some((key, c));
        }
        Ok(all)
    }

    fn model(&self) -> Option<Model<'_>> {
        match (&self.ru, &self.reader) {
            (Some(m), _) => Some(Model::Ru(m)),
            (None, Some(r)) => Some(Model::En(r)),
            (None, None) => None,
        }
    }

    fn handle(&mut self, job: Job) -> Result<()> {
        match job {
            Job::Load => {
                let t = Instant::now();
                self.load_model()?;
                let backend = lock(&self.core.opts).backend.clone();
                super::warm_cleanup(&backend, &|msg| {
                    let msg = msg.to_string();
                    with_ui(move |u| u.hud.set_text(&msg));
                });
                let desc = super::cleanup_desc();
                with_ui(move |u| u.set_cleanup_label(&desc));
                self.core.loading.store(false, Ordering::Release);
                log(&format!("model ready in {:.1}s (language: {}, cleanup: {})", t.elapsed().as_secs_f64(), lock(&self.core.lang).code(), super::cleanup_desc()));
                let key = title_case(&lock(&self.core.opts).key.replace('_', " "));
                if let Some(secs) = lock(&self.core.opts).selftest {
                    self.core.camera.ensure_open();
                    // let the video start and the camera warm up, then hold the "key" for `secs`
                    on_main_after(1.0, move || {
                        if let Some(u) = ui() {
                            u.on_start(false);
                        }
                    });
                    on_main_after(1.0 + secs, || {
                        if let Some(u) = ui() {
                            u.on_stop();
                        }
                    });
                    return Ok(());
                }
                if lock(&self.core.settings).flag("whisper", false) {
                    let _ = self.core.jobs.try_send(Job::Whisper);
                }
                let onboard = lock(&self.core.opts).onboard || !lock(&self.core.settings).flag("onboarded", false);
                with_ui(move |u| {
                    if let Some(it) = u.state_item.borrow().as_ref() {
                        it.setTitle(&NSString::from_str("Ready"));
                        // SAFETY: reading an immutable AppKit constant.
                        it.setImage(symbol("checkmark.circle", 13.0, unsafe { NSFontWeightRegular }).as_deref());
                    }
                    if onboard {
                        super::onboarding::show(u, false);
                    } else {
                        u.hud.show(Mode::Done, "Lipflow is ready", &format!("Hold {key} and mouth your words"), Some(2.5));
                    }
                });
            }
            Job::Reload => {
                self.load_model()?;
            }
            Job::Preview(session) => {
                let Some(model) = self.model() else { return Ok(()) };
                if self.core.session.load(Ordering::Acquire) != session {
                    return Ok(());
                }
                let Some(snap) = self.core.camera.with_recording(Recording::snapshot) else { return Ok(()) };
                let ru = matches!(model, Model::Ru(_));
                let frames = if ru { snap.face_frames() } else { snap.rois() };
                match frames {
                    None => with_ui(|u| u.hud.set_text("Can't see your face…")),
                    Some((frames, t)) => {
                        let text = if ru {
                            let key = snap.ts.first().map_or(0, |x| x.to_bits());
                            let feats = self.ru_features(key, &frames, t, true)?;
                            let m = self.ru.as_ref().context("Russian model not loaded")?;
                            m.read(&m.encode_features(&feats)?, 1)?
                        } else {
                            let reader = self.reader.as_ref().context("model not loaded")?;
                            reader.greedy(&reader.encode(&frames, t)?)?.to_lowercase()
                        };
                        if self.core.session.load(Ordering::Acquire) == session && !text.is_empty() {
                            with_ui(move |u| {
                                if u.hud.mode() == Mode::Listening {
                                    u.hud.set_text(&text);
                                }
                            });
                        }
                    }
                }
            }
            Job::Final(rec, audio) => self.final_(&rec, audio.as_ref())?,
            Job::Whisper => self.load_whisper()?,
            Job::Train => {
                let ru = *lock(&self.core.lang) == text::Lang::Ru;
                let progress = |pct: f64, msg: &str| {
                    let msg = msg.to_string();
                    with_ui(move |_| super::onboarding::report(pct, &msg));
                };
                let r = if ru { crate::train::train_on_face_ru(&progress) } else { crate::train::train_on_face(lock(&self.core.opts).beam, &progress) };
                match r {
                    Ok(r) => {
                        if r.after.is_some() {
                            self.load_model()?;
                            let mut s = lock(&self.core.settings);
                            s.set(
                                if ru { "training_ru" } else { "training" },
                                serde_json::json!({"before": r.before, "after": r.after, "kept": r.kept, "clips": r.clips, "at": camera::now()}),
                            );
                            let _ = s.save();
                        }
                        self.core.loading.store(false, Ordering::Release);
                        with_ui(move |_| super::onboarding::finished(r.before, r.after, r.kept, &r.note));
                    }
                    Err(e) => {
                        self.core.loading.store(false, Ordering::Release);
                        let msg = format!("Training failed: {e}");
                        with_ui(move |_| super::onboarding::finished(0.0, None, false, &msg));
                    }
                }
            }
        }
        Ok(())
    }

    /// Download (first time, 1.8 GB) and load the audio-visual model.
    fn load_whisper(&mut self) -> Result<()> {
        if self.av_reader.is_some() {
            return Ok(());
        }
        let models = paths::models();
        if !LipReader::av_available(&models) {
            hud(Mode::Reading, "Whisper mode", "Downloading the audio-visual model (1.8 GB)…", None);
            let base = "https://huggingface.co/nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base/resolve/main/";
            let status = |m: &str| {
                let m = m.replace("text-cleanup", "audio-visual");
                with_ui(move |u| u.hud.set_text(&m));
            };
            let r = super::download(&format!("{base}config.json"), &models.join("av/config.json"), &status)
                .and_then(|()| super::download(&format!("{base}model.safetensors"), &models.join("av/model.safetensors"), &status));
            if let Err(e) = r {
                hud(Mode::Error, "Whisper mode", &format!("Download failed: {e}").chars().take(80).collect::<String>(), Some(5.0));
                return Ok(());
            }
        }
        hud(Mode::Reading, "Whisper mode", "Loading…", None);
        let beam = lock(&self.core.opts).beam;
        self.av_reader = Some(LipReader::load_av(&crate::reader_options(beam))?);
        self.core.av_ready.store(true, Ordering::Release);
        log("whisper mode ready (lips + audio)");
        hud(Mode::Done, "Whisper mode on", "Whisper or speak softly while you mouth the words", Some(3.0));
        Ok(())
    }

    /// The final read's features: the previews' settled ones plus the rest (cache dropped).
    fn ru_features_final(&self, key: u64, frames: &[u8], t: usize) -> Result<Tensor> {
        let m = self.ru.as_ref().context("Russian model not loaded")?;
        let cache = self.feats.borrow_mut().take();
        match cache {
            Some((k, c)) if k == key && c.dim(0)? <= t => {
                let n = c.dim(0)?;
                if n == t { Ok(c) } else { Ok(Tensor::cat(&[&c, &m.features(frames, t, n, t)?], 0)?) }
            }
            _ => m.features(frames, t, 0, t),
        }
    }

    fn practice_done(&self, rec: &Recording, rois: Vec<u8>, t: usize, raw: String) {
        log(&format!("practice clip ({:.1}s): {raw:?}", rec.duration()));
        with_ui(move |u| {
            super::onboarding::clip_done(u, true, "", Some((rois, t, raw)));
            u.hud.hide();
        });
    }

    /// Lips + audio when whisper mode recorded audio for this clip, else None.
    fn av_candidates(&self, rec: &Recording, rois: &[u8], t: usize, audio: Option<&crate::audio::Chunks>) -> Result<Option<Vec<String>>> {
        let (Some(av), Some(audio), Some(&t0)) = (&self.av_reader, audio, rec.ts.first()) else { return Ok(None) };
        let Some(wave) = crate::audio::segment(audio, t0, t, 25) else { return Ok(None) };
        Ok(Some(av.beam_search(&av.encode_av(rois, t, &wave)?, 5)?))
    }

    fn final_(&self, rec: &Recording, audio: Option<&crate::audio::Chunks>) -> Result<()> {
        let t0 = Instant::now();
        let Some(model) = self.model() else { return Ok(()) };
        let onboarding = self.core.onboarding.load(Ordering::Acquire);
        if let Some((title, advice)) = rec.problem() {
            log(&format!("skipped {:.1}s clip ({} frames, face in {:.0}%): {title}", rec.duration(), rec.ts.len(), rec.face_ratio() * 100.0));
            if onboarding {
                let msg = format!("{title}. {advice}.");
                with_ui(move |u| {
                    super::onboarding::clip_done(u, false, &msg, None);
                    u.hud.hide();
                });
            } else {
                hud(Mode::Error, title, advice, Some(2.2));
            }
            return Ok(());
        }
        // `rois`: the clip as saved and learned from — grayscale mouths (English) or colour faces
        let frames = match model {
            Model::En(_) => rec.rois(),
            Model::Ru(_) => rec.face_frames(),
        };
        let Some((rois, t)) = frames else { return Ok(()) };
        let beam = lock(&self.core.opts).beam.max(2);
        let (candidates, t_enc) = match model {
            Model::En(reader) => {
                let enc = reader.encode(&rois, t)?;
                let t_enc = t0.elapsed().as_secs_f64();
                if onboarding {
                    let raw = reader.greedy(&enc)?;
                    self.practice_done(rec, rois, t, raw);
                    return Ok(());
                }
                // Whisper mode reads nothing without an audible whisper (silent mouthing): fall back to lips.
                let c = match self.av_candidates(rec, &rois, t, audio)? {
                    Some(c) if c.first().is_some_and(|x| !x.is_empty()) => c,
                    Some(_) => {
                        log("lips + audio read nothing, using lips only");
                        reader.beam_search(&enc, 5)?
                    }
                    None => reader.beam_search(&enc, 5)?,
                };
                (c, t_enc)
            }
            Model::Ru(m) => {
                let key = rec.ts.first().map_or(0, |x| x.to_bits());
                let feats = self.ru_features_final(key, &rois, t)?;
                let enc = m.encode_features(&feats)?;
                let t_enc = t0.elapsed().as_secs_f64();
                if onboarding {
                    let raw = m.read(&enc, 1)?;
                    self.practice_done(rec, rois, t, raw);
                    return Ok(());
                }
                (m.read_n(&enc, beam)?, t_enc)
            }
        };
        let t_beam = t0.elapsed().as_secs_f64() - t_enc;
        let Some(first) = candidates.first().filter(|c| !c.is_empty()).cloned() else {
            log(&format!("{:.1}s clip: nothing read", rec.duration()));
            hud(Mode::Error, "Couldn't read that", "Try again, a little slower", Some(2.2));
            return Ok(());
        };
        let lower = first.to_lowercase();
        with_ui(move |u| u.hud.set_text(&lower));
        let context = {
            let c = lock(&self.core.context);
            c[c.len().saturating_sub(3)..].join(" ")
        };
        let names = super::context::names();
        let text = super::clean(&candidates, &context, &names);
        let t_all = t0.elapsed().as_secs_f64();
        log(&format!("{:.1}s clip → raw: {first:?}\n          → typed: {text:?}  (encode {t_enc:.2}s, beam {t_beam:.2}s, total {t_all:.2}s)", rec.duration()));
        if text.is_empty() {
            hud(Mode::Error, "Couldn't read that", "Try again, a little slower", Some(2.2));
            return Ok(());
        }
        let out = {
            let mut last = lock(&self.core.last_paste_at);
            let joined = last.is_some_and(|t| t.elapsed().as_secs_f64() < JOIN_WINDOW);
            *last = Some(Instant::now());
            if joined { format!(" {text}") } else { text.clone() }
        };
        *lock(&self.core.last_output) = text.clone();
        lock(&self.core.context).push(text.clone());
        if let Err(e) = data::log_history(rec.duration(), &candidates, &text, t_all, &super::cleanup_desc()) {
            eprintln!("[lipflow] history: {e}");
        }
        let (save_clips, learn) = {
            let s = lock(&self.core.settings);
            (s.flag("save_clips", true), s.flag("learn_corrections", true))
        };
        if save_clips {
            let dir = paths::clips("dictations");
            if let Err(e) = data::save_clip(&dir, &rois, t, &text, &candidates, &[]) {
                eprintln!("[lipflow] keep clip: {e}");
            }
            data::prune(&dir, KEEP_CLIPS);
        }
        let paste = lock(&self.core.opts).paste;
        let shown = text.clone();
        with_ui(move |u| {
            if paste {
                paste_text(&out, 0.6);
                if learn {
                    super::context::watch_corrections(&out, rois, t, candidates);
                }
            } else {
                copy_text(&shown);
            }
            u.hud.show(Mode::Done, if paste { "Pasted" } else { "Copied" }, &shown, Some(2.4));
        });
        if lock(&self.core.opts).selftest.is_some() {
            println!("SELFTEST text: {}", lock(&self.core.last_output));
            on_main_after(1.0, || {
                if let Some(u) = ui() {
                    NSApplication::sharedApplication(u.mtm).terminate(None);
                }
            });
        }
        Ok(())
    }
}

// -- startup -------------------------------------------------------------------------

fn request_camera(u: &'static Ui) {
    match camera::authorization() {
        AVAuthorizationStatus::NotDetermined => camera::request_access(|ok| log(&format!("camera access {}", if ok { "granted" } else { "denied" }))),
        AVAuthorizationStatus::Denied | AVAuthorizationStatus::Restricted => {
            log("camera access is denied: System Settings → Privacy & Security → Camera → enable Lipflow");
            u.hud.show(Mode::Error, "No camera access", "Enable Lipflow in Settings → Privacy → Camera", Some(6.0));
        }
        _ => {}
    }
}

pub fn run(mut opts: Options) -> Result<()> {
    let mtm = MainThreadMarker::new().ok_or_else(|| anyhow::anyhow!("the app must start on the main thread"))?;
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);

    let settings = Settings::load();
    // the hotkey only works while the app runs: start with the session unless turned off
    if super::login::in_bundle() && settings.flag("open_at_login", true) && !super::login::enabled() {
        match super::login::set(true) {
            Ok(()) => log("will open at login (Settings to turn off)"),
            Err(e) => log(&format!("couldn't add Lipflow to Login Items: {e}")),
        }
    }
    if opts.key == "right_option"
        && let Some(k) = settings.str("key")
    {
        opts.key = k.to_string();
    }
    let cam_src = if opts.camera == "auto" { settings.str("camera").unwrap_or("auto").to_string() } else { opts.camera.clone() };
    super::init_cleanup(&opts.backend);

    let tracker = FaceLandmarker::load(&paths::models().join("face_landmarker.task"))?;
    let (tx, rx) = sync_channel::<Job>(32);
    let core = Arc::new_cyclic(|weak: &std::sync::Weak<Core>| {
        let w = weak.clone();
        let on_frame_cb: camera::OnFrame = Box::new(move |ev| {
            if let Some(c) = w.upgrade() {
                on_frame(&c, ev);
            }
        });
        Core {
            camera: Camera::new(&cam_src, tracker, on_frame_cb),
            jobs: tx,
            session: AtomicU64::new(0),
            preview_busy: AtomicBool::new(false),
            loading: AtomicBool::new(true),
            ui_busy: AtomicBool::new(false),
            onboarding: AtomicBool::new(false),
            av_ready: AtomicBool::new(false),
            lang: Mutex::new(text::Lang::En),
            settings: Mutex::new(settings),
            opts: Mutex::new(opts.clone()),
            last_output: Mutex::new(String::new()),
            last_paste_at: Mutex::new(None),
            context: Mutex::new(Vec::new()),
        }
    });

    let target: Retained<MenuTarget> = {
        let this = MenuTarget::alloc(mtm).set_ivars(());
        // SAFETY: NSObject's init.
        unsafe { msg_send![super(this), init] }
    };
    app.setDelegate(Some(ProtocolObject::from_ref(&*target)));
    let status = NSStatusBar::systemStatusBar().statusItemWithLength(NSVariableStatusItemLength);
    let ui_state: &'static Ui = Box::leak(Box::new(Ui {
        mtm,
        core: core.clone(),
        hud: Hud::new(mtm),
        status,
        state_item: RefCell::new(None),
        cleanup_item: RefCell::new(None),
        cam_items: RefCell::new(Vec::new()),
        target,
        hotkey: RefCell::new(None),
        ptt: RefCell::new(PushToTalk::default()),
        pending_stop: Cell::new(None),
        hands_free: Cell::new(false),
        mic: RefCell::new(super::mic::Mic::new()),
    }));
    UI.with(|u| u.set(ui_state)).map_err(|_| anyhow::anyhow!("UI initialised twice"))?;

    ui_state.set_status_icon(false);
    ui_state.build_menu();
    match Hotkey::install(&opts.key, |ev| {
        if let Some(u) = ui() {
            u.on_key(ev);
        }
    }) {
        Ok(h) => *ui_state.hotkey.borrow_mut() = Some(h),
        Err(e) => {
            log(&format!("{e}"));
            CGRequestListenEventAccess();
            ui_state.hud.show(Mode::Error, "Needs Input Monitoring", "Allow Lipflow, then restart it", None);
        }
    }
    if opts.paste && !CGPreflightPostEventAccess() {
        CGRequestPostEventAccess();
        log("Allow Lipflow under Privacy & Security → Accessibility so Lipflow can paste.");
    }
    request_camera(ui_state);
    ui_state.hud.show(Mode::Reading, "Lipflow", "Loading the lip-reading model…", None);

    let worker = Worker { core: core.clone(), reader: None, av_reader: None, ru: None, feats: RefCell::new(None) };
    std::thread::Builder::new().name("lipflow-model".into()).spawn(move || worker.run(rx))?;
    core.jobs.try_send(Job::Load).map_err(|_| anyhow::anyhow!("model worker unavailable"))?;
    log(&format!("hold {} and mouth your words · double-tap for hands-free · Esc cancels · Ctrl-C quits", opts.key.replace('_', " ")));
    app.run();
    Ok(())
}

/// Whisper mode switched on in Settings.
pub fn enable_whisper(u: &'static Ui) {
    let _ = u.core.jobs.try_send(Job::Whisper);
}

/// Setup's last step: train on the practice clips (model worker).
pub fn start_training(u: &'static Ui) {
    u.core.loading.store(true, Ordering::Release);
    if u.core.jobs.try_send(Job::Train).is_err() {
        u.core.loading.store(false, Ordering::Release);
        super::onboarding::finished(0.0, None, false, "The model is busy; try again from the menu.");
    }
}
