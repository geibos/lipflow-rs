//! First-run setup (`lipflow/onboarding.py`): permissions → import Wispr Flow → mouth ~24
//! sentences → train on your face. The practice step uses the normal push-to-talk key, so setup
//! is also the tutorial. Clips are kept with their known text under clips/onboarding.

use std::cell::RefCell;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol};
use objc2::{MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{NSApplication, NSButton, NSFontWeightBold, NSFontWeightMedium, NSImageScaling, NSImageView, NSProgressIndicator, NSTextField, NSView};
use objc2_av_foundation::AVAuthorizationStatus;
use objc2_core_graphics::{CGPreflightListenEventAccess, CGPreflightPostEventAccess, CGRequestListenEventAccess, CGRequestPostEventAccess};
use objc2_foundation::{MainThreadMarker, NSString, NSTimer};

use super::app::{Ui, ui};
use super::hud::{ACCENT, AMBER, GREEN, Mode, VideoView, cg, make_video_view, rect, rgb, symbol, weight_semibold};
use super::main_thread::on_main;
use super::widgets::{GlassWindow, TextStyle, capsule, close_button, glass_window, present, text};
use super::{camera, settings_window};
use crate::mouth_view::MouthView;
use crate::text::practice::{N_SENTENCES, practice_sentences_in};
use crate::{data, paths, text as txt};

const WW: f64 = 620.0;
const WH: f64 = 600.0;

#[derive(Default)]
struct State {
    win: Option<Retained<GlassWindow>>,
    root: Option<Retained<NSView>>,
    target: Option<Retained<SetupTarget>>,
    page: Option<Retained<NSView>>,
    timer: Option<Retained<NSTimer>>,
    page_icon: Option<Retained<NSImageView>>,
    page_title: Option<Retained<NSTextField>>,
    page_sub: Option<Retained<NSTextField>>,
    // permissions
    perm_rows: Vec<(Retained<NSImageView>, Retained<NSButton>)>,
    perm_next: Option<Retained<NSButton>>,
    perm_restart: Option<Retained<NSButton>>,
    asked_perms: bool,
    /// Opened from "Check permissions…": the permissions page alone, with a Done button.
    perm_only: bool,
    // words
    words_status: Option<Retained<NSTextField>>,
    // practice
    sentences: Vec<String>,
    i: usize,
    count: Option<Retained<NSTextField>>,
    prompt: Option<Retained<NSTextField>>,
    video: Option<Retained<VideoView>>,
    feedback: Option<Retained<NSTextField>>,
    round_clips: Vec<PathBuf>,
    // train
    progress: Option<Retained<NSProgressIndicator>>,
    train_status: Option<Retained<NSTextField>>,
    result: Option<Retained<NSTextField>>,
    done_btn: Option<Retained<NSButton>>,
}

thread_local! {
    static ST: RefCell<State> = RefCell::new(State::default());
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    ST.with(|s| f(&mut s.borrow_mut()))
}

define_class!(
    // SAFETY: NSObject subclass on the main thread, no Drop.
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    struct SetupTarget;

    unsafe impl NSObjectProtocol for SetupTarget {}

    impl SetupTarget {
        #[unsafe(method(close:))]
        fn close_setup(&self, _s: Option<&AnyObject>) {
            close();
        }
        #[unsafe(method(goPermissions:))]
        fn go_permissions_(&self, _s: Option<&AnyObject>) {
            go_permissions();
        }
        #[unsafe(method(refreshPerms:))]
        fn refresh_perms_(&self, _s: Option<&AnyObject>) {
            refresh_perms();
        }
        #[unsafe(method(askCamera:))]
        fn ask_camera(&self, _s: Option<&AnyObject>) {
            if camera::authorization() == AVAuthorizationStatus::NotDetermined {
                camera::request_access(|_| {});
            } else {
                open_url("x-apple.systempreferences:com.apple.preference.security?Privacy_Camera");
            }
        }
        #[unsafe(method(askInput:))]
        fn ask_input(&self, _s: Option<&AnyObject>) {
            with(|s| s.asked_perms = true);
            if !CGRequestListenEventAccess() {
                open_url("x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent");
            }
        }
        #[unsafe(method(askAccess:))]
        fn ask_access(&self, _s: Option<&AnyObject>) {
            with(|s| s.asked_perms = true);
            if !CGRequestPostEventAccess() {
                open_url("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility");
            }
        }
        #[unsafe(method(restartForPerms:))]
        fn restart_for_perms(&self, _s: Option<&AnyObject>) {
            restart_app();
        }
        #[unsafe(method(goWords:))]
        fn go_words_(&self, _s: Option<&AnyObject>) {
            go_words();
        }
        #[unsafe(method(importWispr:))]
        fn import_wispr(&self, _s: Option<&AnyObject>) {
            import_wispr_async();
        }
        #[unsafe(method(goPractice:))]
        fn go_practice_(&self, _s: Option<&AnyObject>) {
            go_practice();
        }
        #[unsafe(method(redo:))]
        fn redo_(&self, _s: Option<&AnyObject>) {
            redo();
        }
        #[unsafe(method(skip:))]
        fn skip_(&self, _s: Option<&AnyObject>) {
            skip();
        }
        #[unsafe(method(finish:))]
        fn finish_(&self, _s: Option<&AnyObject>) {
            finish();
        }
    }
);

fn open_url(url: &str) {
    let _ = std::process::Command::new("open").arg(url).spawn();
}

fn mtm() -> Option<MainThreadMarker> {
    MainThreadMarker::new()
}

fn target_obj() -> Option<Retained<SetupTarget>> {
    with(|s| s.target.clone())
}

/// A fresh page with the round icon, title and subtitle; replaces the previous one.
fn new_page(icon: &str, title: &str, subtitle: &str) -> Option<Retained<NSView>> {
    let mtm = mtm()?;
    let (root, old, timer) = with(|s| (s.root.clone(), s.page.take(), s.timer.take()));
    if let Some(t) = timer {
        t.invalidate();
    }
    if let Some(p) = old {
        p.removeFromSuperview();
    }
    let root = root?;
    let page = NSView::initWithFrame(NSView::alloc(mtm), rect(0.0, 0.0, WW, WH));
    // below the close button
    root.addSubview_positioned_relativeTo(&page, objc2_app_kit::NSWindowOrderingMode::Below, None);
    let badge = NSView::initWithFrame(NSView::alloc(mtm), rect((WW - 64.0) / 2.0, WH - 130.0, 64.0, 64.0));
    badge.setWantsLayer(true);
    if let Some(l) = badge.layer() {
        l.setCornerRadius(32.0);
        l.setBackgroundColor(Some(&cg(ACCENT, 0.18)));
    }
    let iv = NSImageView::initWithFrame(NSImageView::alloc(mtm), rect(0.0, 0.0, 64.0, 64.0));
    iv.setImageScaling(NSImageScaling::ScaleNone);
    iv.setImage(symbol(icon, 26.0, weight_semibold()).as_deref());
    iv.setContentTintColor(Some(&rgb(ACCENT, 1.0)));
    badge.addSubview(&iv);
    page.addSubview(&badge);
    // SAFETY: reading an immutable AppKit constant.
    let bold = unsafe { NSFontWeightBold };
    let t = text(&page, rect(40.0, WH - 180.0, WW - 80.0, 34.0), title, TextStyle::new(24.0).weight(bold).center(), mtm);
    let sub = text(&page, rect(60.0, WH - 240.0, WW - 120.0, 56.0), subtitle, TextStyle::new(13.5).alpha(0.65).center(), mtm);
    with(|s| {
        s.page = Some(page.clone());
        s.page_icon = Some(iv);
        s.page_title = Some(t);
        s.page_sub = Some(sub);
    });
    Some(page)
}

fn button(page: &NSView, title: &str, action: objc2::runtime::Sel, x: f64, y: f64, w: f64, h: f64, primary: bool) -> Option<Retained<NSButton>> {
    let (mtm, t) = (mtm()?, target_obj()?);
    let obj: &AnyObject = &t;
    Some(capsule(page, title, obj, action, rect(x, y, w, h), primary, mtm))
}

fn steps(page: &NSView, n: usize) {
    let Some(mtm) = mtm() else { return };
    let labels = ["Permissions", "Your words", "Practice", "Train"];
    let w = 110.0;
    let x0 = (WW - w * labels.len() as f64) / 2.0;
    for (k, lab) in labels.iter().enumerate() {
        let (done, cur) = (k < n, k == n);
        let c = if cur { rgb(ACCENT, 1.0) } else if done { rgb(GREEN, 0.9) } else { rgb((1.0, 1.0, 1.0), 0.35) };
        let label = if done { format!("✓ {lab}") } else { (*lab).to_string() };
        text(page, rect(x0 + k as f64 * w, 28.0, w, 18.0), &label, TextStyle::new(11.5).weight(weight_semibold()).center().color(c), mtm);
    }
}

// -- 1. welcome --------------------------------------------------------------------------

fn welcome() {
    let Some(p) = new_page(
        "mouth",
        "Lipflow reads your lips",
        "Hold a key, mouth what you want to say without making a sound, let go, and the text appears wherever you're typing. Setup takes about 8 minutes: it learns your words and your face.",
    ) else {
        return;
    };
    if let Some(mtm) = mtm() {
        text(
            &p,
            rect(90.0, 230.0, WW - 180.0, 60.0),
            "Everything stays on this Mac: the camera video, your practice clips and the models trained on them. Nothing is uploaded.",
            TextStyle::new(13.0).alpha(0.55).center(),
            mtm,
        );
    }
    button(&p, "Get started", sel!(goPermissions:), (WW - 200.0) / 2.0, 130.0, 200.0, 40.0, true);
}

// -- 2. permissions ----------------------------------------------------------------------

fn perm_state() -> [bool; 3] {
    [camera::authorization() == AVAuthorizationStatus::Authorized, CGPreflightListenEventAccess(), CGPreflightPostEventAccess()]
}

fn go_permissions() {
    let Some(p) = new_page("lock.shield", "Three permissions", "macOS asks for these once. Grant each one, then come back here.") else { return };
    let Some(mtm) = mtm() else { return };
    let perm_only = with(|s| s.perm_only);
    if !perm_only {
        steps(&p, 0);
    }
    let rows = [
        ("camera.fill", "Camera", "to see your mouth", sel!(askCamera:)),
        ("keyboard", "Input Monitoring", "to notice the push-to-talk key", sel!(askInput:)),
        ("text.cursor", "Accessibility", "to paste into the app you're using", sel!(askAccess:)),
    ];
    let mut perm_rows = Vec::new();
    for (k, (icon, name, why, action)) in rows.into_iter().enumerate() {
        let y = WH - 320.0 - k as f64 * 64.0;
        let iv = NSImageView::initWithFrame(NSImageView::alloc(mtm), rect(80.0, y + 6.0, 28.0, 28.0));
        iv.setImage(symbol(icon, 18.0, weight_semibold()).as_deref());
        iv.setContentTintColor(Some(&rgb((1.0, 1.0, 1.0), 0.85)));
        p.addSubview(&iv);
        text(&p, rect(120.0, y + 16.0, 280.0, 20.0), name, TextStyle::new(14.0).weight(weight_semibold()), mtm);
        text(&p, rect(120.0, y - 2.0, 280.0, 18.0), why, TextStyle::new(12.0).alpha(0.55), mtm);
        let status = NSImageView::initWithFrame(NSImageView::alloc(mtm), rect(WW - 212.0, y + 8.0, 24.0, 24.0));
        p.addSubview(&status);
        if let Some(b) = button(&p, "Allow", action, WW - 180.0, y + 3.0, 96.0, 32.0, false) {
            perm_rows.push((status, b));
        }
    }
    let next = if perm_only {
        button(&p, "Done", sel!(close:), (WW - 200.0) / 2.0, 90.0, 200.0, 40.0, true)
    } else {
        button(&p, "Continue", sel!(goWords:), (WW - 200.0) / 2.0, 90.0, 200.0, 40.0, true)
    };
    // macOS keeps a denial for the life of the process: offer a restart once Allow was clicked.
    let restart = button(&p, "Turned them on? Restart Lipflow", sel!(restartForPerms:), (WW - 260.0) / 2.0, 46.0, 260.0, 30.0, false);
    if let Some(r) = &restart {
        r.setHidden(true);
    }
    with(|s| {
        s.perm_rows = perm_rows;
        s.perm_next = next;
        s.perm_restart = restart;
        s.asked_perms = false;
    });
    refresh_perms();
    let block = RcBlock::new(|_t: NonNull<NSTimer>| refresh_perms());
    // SAFETY: the block only touches main-thread state through `with`.
    let timer = unsafe { NSTimer::scheduledTimerWithTimeInterval_repeats_block(1.0, true, &block) };
    with(|s| s.timer = Some(timer));
}

fn refresh_perms() {
    let states = perm_state();
    with(|s| {
        for ((status, btn), &ok) in s.perm_rows.iter().zip(states.iter()) {
            status.setImage(symbol(if ok { "checkmark.circle.fill" } else { "circle.dashed" }, 18.0, weight_semibold()).as_deref());
            status.setContentTintColor(Some(&*if ok { rgb(GREEN, 1.0) } else { rgb((1.0, 1.0, 1.0), 0.35) }));
            btn.setHidden(ok);
        }
        let ready = states.iter().all(|&x| x);
        if let Some(n) = &s.perm_next {
            let enabled = ready || s.perm_only;
            n.setEnabled(enabled);
            if let Some(l) = n.layer() {
                l.setOpacity(if enabled { 1.0 } else { 0.35 });
            }
        }
        if let Some(r) = &s.perm_restart {
            r.setHidden(ready || !s.asked_perms);
        }
    });
}

fn restart_app() {
    let exe = std::env::current_exe().ok();
    let bundle = exe.as_ref().and_then(|e| e.ancestors().find(|a| a.extension().is_some_and(|x| x == "app")).map(std::path::Path::to_path_buf));
    let Some(app) = bundle else { return }; // run from a terminal: nothing to reopen
    let cmd = format!("sleep 0.6; open '{}'", app.display().to_string().replace('\'', "'\\''"));
    let _ = std::process::Command::new("/bin/bash").arg("-c").arg(cmd).spawn();
    if let Some(mtm) = mtm() {
        NSApplication::sharedApplication(mtm).terminate(None);
    }
}

// -- 3. your words -----------------------------------------------------------------------

fn go_words() {
    let have = paths::phrases().exists();
    let Some(p) = new_page(
        "text.book.closed",
        "Teach it your words",
        "Most of what you'll mouth is stuff you already say. If you use Wispr Flow, Lipflow can learn your phrasing and names from its history, read locally, never uploaded.",
    ) else {
        return;
    };
    let Some(mtm) = mtm() else { return };
    steps(&p, 1);
    let status = text(&p, rect(80.0, 250.0, WW - 160.0, 60.0), "", TextStyle::new(13.5).alpha(0.8).center(), mtm);
    let wispr = txt::personal::wispr_dir();
    let can = wispr.is_dir();
    if have {
        let n = std::fs::read_to_string(paths::phrases()).map_or(0, |s| s.lines().count());
        status.setStringValue(&NSString::from_str(&format!("Already imported: {n} of your phrases.")));
    } else if !can {
        status.setStringValue(&NSString::from_str("Wispr Flow isn't installed on this Mac. You can skip this."));
    }
    with(|s| s.words_status = Some(status));
    if can {
        button(&p, if have { "Re-import Wispr Flow" } else { "Import Wispr Flow" }, sel!(importWispr:), WW / 2.0 - 190.0, 110.0, 180.0, 40.0, !have);
    }
    button(&p, if have { "Continue" } else { "Skip" }, sel!(goPractice:), WW / 2.0 + if can { 10.0 } else { -90.0 }, 110.0, 180.0, 40.0, have || !can);
}

fn set_words_status(msg: &str) {
    with(|s| {
        if let Some(t) = &s.words_status {
            t.setStringValue(&NSString::from_str(msg));
        }
    });
}

fn import_wispr_async() {
    set_words_status("Reading your Wispr Flow history…");
    let _ = std::thread::Builder::new().name("lipflow-import".into()).spawn(|| {
        let msg = match txt::personal::import_wispr(&txt::personal::wispr_dir(), &paths::phrases(), &paths::words()) {
            Ok(st) => {
                let mut m = format!("Imported {} phrases.", st.phrases);
                if !st.new_names.is_empty() {
                    m.push_str(&format!(" Found {} names and terms (edit them from the menu).", st.new_names.len()));
                }
                m.push_str(" Lipflow will learn your phrasing during training.");
                m
            }
            Err(e) => format!("Couldn't import: {e}"),
        };
        on_main(move || {
            set_words_status(&msg);
            super::reload_personal();
        });
    });
}

// -- 4. practice -------------------------------------------------------------------------

fn go_practice() {
    let Some(u) = ui() else { return };
    let lang = *u.core.lang.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = paths::onboarding_clips(lang);
    let _ = std::fs::create_dir_all(&dir);
    // Every visit is a fresh round; earlier clips are kept and training uses all of them.
    let done: std::collections::HashSet<String> = saved_clip_texts(&dir);
    let prior = done.len();
    let sentences: Vec<String> = practice_sentences_in(N_SENTENCES * 2, &paths::phrases(), lang)
        .unwrap_or_default()
        .into_iter()
        .filter(|x| !done.contains(x))
        .take(N_SENTENCES)
        .collect();
    let key = u.key();
    let key_name = key.replace('_', " ");
    let mut sub = format!("Hold {key_name}, silently mouth the sentence at your normal pace, then let go. Face the camera with your mouth in good light.");
    if prior > 0 {
        sub.push_str(&format!(" You have {prior} clips from before; these add to them."));
    }
    let Some(p) = new_page("quote.bubble", "Mouth each sentence", &sub) else { return };
    let Some(mtm) = mtm() else { return };
    steps(&p, 2);
    let count = text(&p, rect(40.0, 318.0, WW - 80.0, 18.0), "", TextStyle::new(12.0).weight(weight_semibold()).center().color(rgb(ACCENT, 1.0)), mtm);
    // SAFETY: reading an immutable AppKit constant.
    let medium = unsafe { NSFontWeightMedium };
    let prompt = text(&p, rect(50.0, 250.0, WW - 100.0, 64.0), "", TextStyle::new(23.0).weight(medium).center(), mtm);
    let video = make_video_view(rect((WW - 192.0) / 2.0, 124.0, 192.0, 120.0), mtm);
    video.setWantsLayer(true);
    if let Some(l) = video.layer() {
        l.setCornerRadius(16.0);
        l.setMasksToBounds(true);
        l.setBackgroundColor(Some(&cg((0.0, 0.0, 0.0), 0.35)));
    }
    p.addSubview(&video);
    let feedback = text(&p, rect(60.0, 98.0, WW - 120.0, 20.0), "", TextStyle::new(12.5).alpha(0.65).center(), mtm);
    button(&p, "Redo last", sel!(redo:), WW / 2.0 - 170.0, 56.0, 160.0, 34.0, false);
    button(&p, "Skip sentence", sel!(skip:), WW / 2.0 + 10.0, 56.0, 160.0, 34.0, false);
    with(|s| {
        s.sentences = sentences;
        s.i = 0;
        s.count = Some(count);
        s.prompt = Some(prompt);
        s.video = Some(video);
        s.feedback = Some(feedback);
        s.round_clips.clear();
    });
    u.core.camera.set_track_always(true);
    u.core.onboarding.store(true, Ordering::Release);
    show_sentence();
}

fn saved_clip_texts(dir: &std::path::Path) -> std::collections::HashSet<String> {
    crate::train::clip_texts(dir).into_iter().collect()
}

pub fn current_sentence() -> Option<String> {
    with(|s| (!s.sentences.is_empty()).then(|| s.sentences[s.i % s.sentences.len()].clone()))
}

fn show_sentence() {
    with(|s| {
        if s.sentences.is_empty() {
            return;
        }
        let sentence = &s.sentences[s.i % s.sentences.len()];
        if let Some(c) = &s.count {
            c.setStringValue(&NSString::from_str(&format!("SENTENCE {} OF {N_SENTENCES}", s.i + 1)));
        }
        if let Some(p) = &s.prompt {
            p.setStringValue(&NSString::from_str(sentence));
        }
    });
}

/// A new mouth preview for the practice page.
pub fn set_frame(v: MouthView) {
    with(|s| {
        if let Some(video) = &s.video {
            video.set(Some(v));
        }
    });
}

fn feedback(msg: &str, warn: bool) {
    with(|s| {
        if let Some(f) = &s.feedback {
            f.setTextColor(Some(&*if warn { rgb(AMBER, 1.0) } else { rgb((1.0, 1.0, 1.0), 0.6) }));
            f.setStringValue(&NSString::from_str(msg));
        }
    });
}

/// After each practice recording (main thread).
pub fn clip_done(u: &'static Ui, ok: bool, message: &str, clip: Option<(Vec<u8>, usize, String)>) {
    if !ok {
        feedback(message, true);
        return;
    }
    let Some((rois, t, raw)) = clip else { return };
    let Some(sentence) = current_sentence() else { return };
    let lang = *u.core.lang.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match data::save_clip(&paths::onboarding_clips(lang), &rois, t, &sentence, std::slice::from_ref(&raw), &[]) {
        Ok(path) => with(|s| s.round_clips.push(path)),
        Err(e) => {
            feedback(&format!("Couldn't save the clip: {e}"), true);
            return;
        }
    }
    feedback(&format!("Saved. The model read: \"{}\"", raw.to_lowercase()), false);
    let finished = with(|s| {
        s.i += 1;
        s.i >= N_SENTENCES
    });
    if finished {
        u.core.onboarding.store(false, Ordering::Release);
        go_train();
    } else {
        show_sentence();
    }
}

fn redo() {
    let removed = with(|s| {
        if s.i > 0
            && let Some(p) = s.round_clips.pop()
        {
            let _ = std::fs::remove_file(p);
            s.i -= 1;
            return true;
        }
        false
    });
    if removed {
        feedback("Removed the last clip. Try it again.", false);
        show_sentence();
    }
}

fn skip() {
    with(|s| {
        if !s.sentences.is_empty() {
            let k = s.i % s.sentences.len();
            let x = s.sentences.remove(k);
            s.sentences.push(x);
        }
    });
    show_sentence();
}

// -- 5. train ----------------------------------------------------------------------------

fn go_train() {
    let Some(u) = ui() else { return };
    u.core.onboarding.store(false, Ordering::Release);
    let Some(p) = new_page(
        "cpu",
        "Learning your face",
        "Training runs on this Mac's GPU and takes about 5 minutes. You can keep working. Lipflow is paused until it's done.",
    ) else {
        return;
    };
    let Some(mtm) = mtm() else { return };
    steps(&p, 3);
    let progress = NSProgressIndicator::initWithFrame(NSProgressIndicator::alloc(mtm), rect(100.0, 300.0, WW - 200.0, 12.0));
    progress.setIndeterminate(false);
    progress.setMinValue(0.0);
    progress.setMaxValue(100.0);
    p.addSubview(&progress);
    let status = text(&p, rect(60.0, 220.0, WW - 120.0, 56.0), "Starting…", TextStyle::new(13.5).alpha(0.8).center(), mtm);
    let result = text(&p, rect(60.0, 150.0, WW - 120.0, 60.0), "", TextStyle::new(15.0).weight(weight_semibold()).center(), mtm);
    let done = button(&p, "Start using Lipflow", sel!(finish:), (WW - 220.0) / 2.0, 80.0, 220.0, 40.0, true);
    if let Some(d) = &done {
        d.setHidden(true);
    }
    with(|s| {
        s.progress = Some(progress);
        s.train_status = Some(status);
        s.result = Some(result);
        s.done_btn = done;
    });
    super::app::start_training(u);
}

/// Training progress (main thread).
pub fn report(pct: f64, msg: &str) {
    with(|s| {
        if let Some(p) = &s.progress {
            p.setDoubleValue(pct);
        }
        if let Some(t) = &s.train_status {
            t.setStringValue(&NSString::from_str(msg));
        }
    });
}

/// Training finished (main thread). `after` None: not enough clips.
pub fn finished(before: f64, after: Option<f64>, kept: bool, note: &str) {
    with(|s| {
        if let Some(p) = &s.progress {
            p.setDoubleValue(100.0);
        }
        if let Some(t) = &s.page_title {
            t.setStringValue(&NSString::from_str(if after.is_none() || kept { "You're all set" } else { "Setup finished" }));
        }
        if let Some(t) = &s.page_sub {
            t.setStringValue(&NSString::from_str("Hold the key anywhere and mouth your words. You can retrain any time from the menu: more practice sentences make it better."));
        }
        if let Some(i) = &s.page_icon {
            i.setImage(symbol("checkmark", 26.0, weight_semibold()).as_deref());
            i.setContentTintColor(Some(&rgb(GREEN, 1.0)));
        }
        if let Some(r) = &s.result {
            match after {
                None => r.setStringValue(&NSString::from_str(note)),
                Some(a) => {
                    r.setTextColor(Some(&*if kept { rgb(GREEN, 1.0) } else { rgb(AMBER, 1.0) }));
                    let mut m = format!("Words read correctly on sentences it didn't train on: {:.0}% → {:.0}%", (1.0 - before) * 100.0, (1.0 - a) * 100.0);
                    if !kept {
                        m.push_str("\nNo improvement, so the standard model stays.");
                    }
                    r.setStringValue(&NSString::from_str(&m));
                }
            }
        }
        if let Some(t) = &s.train_status {
            t.setStringValue(&NSString::from_str(if after.is_some() { note } else { "" }));
        }
        if let Some(d) = &s.done_btn {
            d.setHidden(false);
        }
    });
}

fn close() {
    if let Some(u) = ui() {
        u.core.onboarding.store(false, Ordering::Release);
        u.core.camera.set_track_always(false);
    }
    let win = with(|s| s.win.clone());
    if let Some(w) = win {
        w.orderOut(None);
    }
}

fn finish() {
    let Some(u) = ui() else { return };
    {
        let mut s = u.core.settings.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        s.set("onboarded", serde_json::Value::Bool(true));
        if let Err(e) = s.save() {
            eprintln!("[lipflow] saving settings: {e}");
        }
    }
    close();
    let key = u.key().replace('_', " ");
    u.hud.show(Mode::Done, "Lipflow is ready", &format!("Hold {key} anywhere and mouth your words"), Some(3.0));
}

/// The permissions page alone (menu → "Check permissions…", Settings): grant or re-grant
/// Camera, Input Monitoring and Accessibility, e.g. after an update reset them.
pub fn show_permissions(u: &'static Ui) {
    ensure_window(u);
    with(|s| s.perm_only = true);
    u.hud.hide();
    if let Some(w) = with(|s| s.win.clone()) {
        present(&w, u.mtm);
    }
    go_permissions();
}

/// Open setup (optionally straight at the practice step).
pub fn show(u: &'static Ui, practice: bool) {
    ensure_window(u);
    // a page left from "Check permissions…" isn't part of setup
    let was_perm_only = with(|s| std::mem::replace(&mut s.perm_only, false));
    if was_perm_only || with(|s| s.page.is_none()) {
        welcome();
    }
    u.hud.hide();
    u.core.camera.set_track_always(true);
    if let Some(w) = with(|s| s.win.clone()) {
        present(&w, u.mtm);
    }
    if practice {
        go_practice();
    }
    let _ = settings_window::personal_vsr_exists();
}

fn ensure_window(u: &'static Ui) {
    let mtm = u.mtm;
    let created = with(|s| s.win.is_some());
    if !created {
        let (win, root) = glass_window(WW, WH, "Set up Lipflow", mtm);
        let target: Retained<SetupTarget> = {
            let this = SetupTarget::alloc(mtm).set_ivars(());
            // SAFETY: NSObject's init.
            unsafe { msg_send![super(this), init] }
        };
        let obj: &AnyObject = &target;
        let close = close_button(&root, WW, WH, obj, sel!(close:), mtm);
        close.setToolTip(Some(&NSString::from_str("Close setup (you can reopen it from the menu)")));
        with(|s| {
            s.win = Some(win);
            s.root = Some(root);
            s.target = Some(target);
        });
    }
}
