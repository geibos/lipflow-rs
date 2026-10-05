//! Settings (`lipflow/settings_window.py`): keep training when you want to, plus the few options.

use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol};
use objc2::{MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{NSControlStateValueOff, NSControlStateValueOn, NSFontWeightBold, NSPopUpButton, NSSwitch, NSView};
use objc2_foundation::{MainThreadMarker, NSString};
use serde_json::Value;

use super::app::{Ui, ui};
use super::hotkey::KEYS;
use super::hud::{ACCENT, rect, rgb, weight_semibold};
use super::widgets::{GlassWindow, TextStyle, capsule, close_button, glass_window, present, text};
use super::{camera, cleanup_desc};
use crate::paths;

const SW: f64 = 560.0;
const SH: f64 = 780.0;

struct Win {
    win: Retained<GlassWindow>,
    root: Retained<NSView>,
    page: Option<Retained<NSView>>,
    target: Retained<SettingsTarget>,
}

thread_local! {
    static WIN: RefCell<Option<Win>> = const { RefCell::new(None) };
}

fn title_case(s: &str) -> String {
    s.split(['_', ' '])
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn count_npz(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map_or(0, |rd| rd.filter_map(Result::ok).filter(|e| e.path().extension().is_some_and(|x| x == "npz")).count())
}

pub fn personal_vsr_exists() -> bool {
    paths::personal_vsr().exists() || paths::personal_vsr().with_extension("safetensors").exists()
}

/// A fine-tuned face model exists for this language.
fn face_model_exists(lang: crate::text::Lang) -> bool {
    match lang {
        crate::text::Lang::En => personal_vsr_exists(),
        crate::text::Lang::Ru => paths::personal_multivsr().exists(),
    }
}

fn save(u: &Ui) {
    let s = u.core.settings.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Err(e) = s.save() {
        eprintln!("[lipflow] saving settings: {e}");
    }
}

fn set_flag(key: &str, on: bool) {
    if let Some(u) = ui() {
        u.core.settings.lock().unwrap_or_else(std::sync::PoisonError::into_inner).set(key, Value::Bool(on));
        save(u);
    }
}

fn switch_on(sender: Option<&AnyObject>) -> bool {
    sender.and_then(|s| s.downcast_ref::<NSSwitch>()).is_some_and(|s| s.state() == NSControlStateValueOn)
}

fn popup_title(sender: Option<&AnyObject>) -> Option<String> {
    sender.and_then(|s| s.downcast_ref::<NSPopUpButton>()).and_then(|p| p.titleOfSelectedItem()).map(|t| t.to_string())
}

define_class!(
    // SAFETY: NSObject subclass on the main thread, no Drop.
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    struct SettingsTarget;

    unsafe impl NSObjectProtocol for SettingsTarget {}

    impl SettingsTarget {
        #[unsafe(method(close:))]
        fn close_window(&self, _s: Option<&AnyObject>) {
            hide();
        }

        #[unsafe(method(trainMore:))]
        fn train_more(&self, _s: Option<&AnyObject>) {
            hide();
            if let Some(u) = ui() {
                super::onboarding::show(u, true);
            }
        }

        #[unsafe(method(resetFace:))]
        fn reset_face(&self, _s: Option<&AnyObject>) {
            let Some(u) = ui() else { return };
            let ru = *u.core.lang.lock().unwrap_or_else(std::sync::PoisonError::into_inner) == crate::text::Lang::Ru;
            let files = if ru { vec![paths::personal_multivsr()] } else { vec![paths::personal_vsr(), paths::personal_vsr().with_extension("safetensors")] };
            for p in files {
                let _ = std::fs::remove_file(p);
            }
            u.core.settings.lock().unwrap_or_else(std::sync::PoisonError::into_inner).0.remove(if ru { "training_ru" } else { "training" });
            save(u);
            u.reload_model();
            build(u);
        }

        #[unsafe(method(pickKey:))]
        fn pick_key(&self, s: Option<&AnyObject>) {
            if let (Some(u), Some(t)) = (ui(), popup_title(s)) {
                u.set_key(&t.to_lowercase().replace(' ', "_"));
            }
        }

        #[unsafe(method(pickLang:))]
        fn pick_lang(&self, s: Option<&AnyObject>) {
            let (Some(u), Some(t)) = (ui(), popup_title(s)) else { return };
            let code = if t == LANGS[0].1 { LANGS[0].0 } else { LANGS[1].0 };
            u.core.settings.lock().unwrap_or_else(std::sync::PoisonError::into_inner).set("language", Value::from(code));
            save(u);
            u.reload_model();
            build(u);
        }

        #[unsafe(method(pickCam:))]
        fn pick_cam(&self, s: Option<&AnyObject>) {
            let (Some(u), Some(name)) = (ui(), popup_title(s)) else { return };
            let id = camera::list_cameras().into_iter().find(|c| c.name == name).map_or_else(|| "auto".to_string(), |c| c.id);
            u.set_camera(&id);
        }

        #[unsafe(method(toggleContext:))]
        fn toggle_context(&self, s: Option<&AnyObject>) {
            set_flag("use_context", switch_on(s));
        }

        #[unsafe(method(toggleClips:))]
        fn toggle_clips(&self, s: Option<&AnyObject>) {
            set_flag("save_clips", switch_on(s));
        }

        #[unsafe(method(toggleLearn:))]
        fn toggle_learn(&self, s: Option<&AnyObject>) {
            set_flag("learn_corrections", switch_on(s));
        }

        #[unsafe(method(toggleWhisper:))]
        fn toggle_whisper(&self, s: Option<&AnyObject>) {
            let on = switch_on(s);
            set_flag("whisper", on);
            if on && let Some(u) = ui() {
                super::app::enable_whisper(u);
            }
        }

        #[unsafe(method(toggleLogin:))]
        fn toggle_login(&self, s: Option<&AnyObject>) {
            let on = switch_on(s);
            set_flag("open_at_login", on);
            if let Err(e) = super::login::set(on) {
                eprintln!("[lipflow] login item: {e}");
            }
        }

        #[unsafe(method(checkPerms:))]
        fn check_perms(&self, _s: Option<&AnyObject>) {
            if let Some(u) = ui() {
                hide();
                super::onboarding::show_permissions(u);
            }
        }

        #[unsafe(method(openData:))]
        fn open_data(&self, _s: Option<&AnyObject>) {
            let _ = std::process::Command::new("open").arg(paths::home()).spawn();
        }
    }
);

/// Dictation languages: settings code and menu title.
const LANGS: [(&str, &str); 2] = [("ru", "Русский"), ("en", "English")];

fn hide() {
    WIN.with(|w| {
        if let Some(w) = w.borrow().as_ref() {
            w.win.orderOut(None);
        }
    });
}

fn row_label(p: &NSView, y: f64, label: &str, mtm: MainThreadMarker) {
    text(p, rect(36.0, y + 6.0, 300.0, 20.0), label, TextStyle::new(13.5), mtm);
}

fn popup(p: &NSView, target: &SettingsTarget, y: f64, label: &str, items: &[String], current: &str, action: objc2::runtime::Sel, mtm: MainThreadMarker) {
    row_label(p, y, label, mtm);
    let pop = NSPopUpButton::initWithFrame_pullsDown(NSPopUpButton::alloc(mtm), rect(SW - 256.0, y, 220.0, 30.0), false);
    for it in items {
        pop.addItemWithTitle(&NSString::from_str(it));
    }
    pop.selectItemWithTitle(&NSString::from_str(current));
    // SAFETY: target lives as long as the window.
    unsafe {
        pop.setTarget(Some(target));
        pop.setAction(Some(action));
    }
    p.addSubview(&pop);
}

fn switch(p: &NSView, target: &SettingsTarget, y: f64, label: &str, on: bool, action: objc2::runtime::Sel, mtm: MainThreadMarker) {
    row_label(p, y, label, mtm);
    let sw = NSSwitch::initWithFrame(NSSwitch::alloc(mtm), rect(SW - 36.0 - 42.0, y + 4.0, 42.0, 24.0));
    sw.setState(if on { NSControlStateValueOn } else { NSControlStateValueOff });
    // SAFETY: as above.
    unsafe {
        sw.setTarget(Some(target));
        sw.setAction(Some(action));
    }
    p.addSubview(&sw);
}

fn build(u: &Ui) {
    let mtm = u.mtm;
    WIN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(w) = slot.as_mut() else { return };
        if let Some(old) = w.page.take() {
            old.removeFromSuperview();
        }
        let p = NSView::initWithFrame(NSView::alloc(mtm), rect(0.0, 0.0, SW, SH));
        w.root.addSubview(&p);
        let t: &AnyObject = &w.target;
        close_button(&p, SW, SH, t, sel!(close:), mtm);
        // SAFETY: reading an immutable AppKit constant.
        text(&p, rect(36.0, SH - 64.0, 300.0, 30.0), "Settings", TextStyle::new(22.0).weight(unsafe { NSFontWeightBold }), mtm);
        let s = u.core.settings.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();

        // -- training ----------------------------------------------------------------
        let y = SH - 110.0;
        text(&p, rect(36.0, y, 400.0, 18.0), "TRAINING ON YOUR FACE", TextStyle::new(11.0).weight(weight_semibold()).color(rgb(ACCENT, 1.0)), mtm);
        let lang = *u.core.lang.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = count_npz(&paths::onboarding_clips(lang));
        let mut line = match s.0.get(if lang == crate::text::Lang::Ru { "training_ru" } else { "training" }) {
            Some(tr) => {
                let f = |k: &str| tr.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                let kept = tr.get("kept").and_then(Value::as_bool).unwrap_or(false);
                format!(
                    "{} practice clips. Last training: words read correctly on held-out sentences {:.0}% → {:.0}%{}",
                    f("clips") as usize,
                    (1.0 - f("before")) * 100.0,
                    (1.0 - f("after")) * 100.0,
                    if kept { "" } else { ", not better, so the standard model is used." }
                )
            }
            None => format!("{n} practice clips so far.{}", if face_model_exists(lang) { "" } else { " Not trained yet." }),
        };
        let nc = count_npz(&paths::clips("corrections"));
        if nc > 0 {
            line.push_str(&format!(" Plus {nc} learned from your corrections."));
        }
        text(&p, rect(36.0, y - 46.0, SW - 72.0, 40.0), &line, TextStyle::new(13.0).alpha(0.8), mtm);
        text(
            &p,
            rect(36.0, y - 84.0, SW - 72.0, 36.0),
            "Each round is 24 new sentences (about 5 minutes) and it retrains on everything you've recorded. More rounds keep improving it.",
            TextStyle::new(12.0).alpha(0.55),
            mtm,
        );
        capsule(&p, "Practice & train more", t, sel!(trainMore:), rect(36.0, y - 132.0, 220.0, 36.0), true, mtm);
        if face_model_exists(lang) {
            capsule(&p, "Reset face model", t, sel!(resetFace:), rect(270.0, y - 132.0, 170.0, 36.0), false, mtm);
        }

        // -- general -----------------------------------------------------------------
        let y = SH - 296.0;
        text(&p, rect(36.0, y, 400.0, 18.0), "GENERAL", TextStyle::new(11.0).weight(weight_semibold()).color(rgb(ACCENT, 1.0)), mtm);
        let langs: Vec<String> = LANGS.iter().map(|l| l.1.to_string()).collect();
        let cur_lang = LANGS.iter().find(|l| l.0 == lang.code()).map_or(LANGS[0].1, |l| l.1);
        popup(&p, &w.target, y - 40.0, "Language I dictate in", &langs, cur_lang, sel!(pickLang:), mtm);
        let y = y - 42.0;
        let keys: Vec<String> = KEYS.iter().map(|k| title_case(k.0)).collect();
        popup(&p, &w.target, y - 40.0, "Push-to-talk key", &keys, &title_case(&u.key()), sel!(pickKey:), mtm);
        let cams = camera::list_cameras();
        let cur = s.str("camera").unwrap_or("auto").to_string();
        let cur_name = cams.iter().find(|c| c.id == cur).map_or_else(|| "Automatic (built-in)".to_string(), |c| c.name.clone());
        let mut items = vec!["Automatic (built-in)".to_string()];
        items.extend(cams.iter().map(|c| c.name.clone()));
        popup(&p, &w.target, y - 82.0, "Camera", &items, &cur_name, sel!(pickCam:), mtm);
        switch(&p, &w.target, y - 124.0, "Use names from the app I'm typing in", s.flag("use_context", true), sel!(toggleContext:), mtm);
        switch(&p, &w.target, y - 166.0, "Keep my last 100 clips to measure accuracy", s.flag("save_clips", true), sel!(toggleClips:), mtm);
        switch(&p, &w.target, y - 208.0, "Learn from words I correct after pasting", s.flag("learn_corrections", true), sel!(toggleLearn:), mtm);
        switch(&p, &w.target, y - 292.0, "Open Lipflow at login (the key works only while it runs)", super::login::enabled(), sel!(toggleLogin:), mtm);
        let whisper_label = if lang == crate::text::Lang::Ru { "Whisper mode (English only for now)" } else { "Whisper mode: lips + a soft whisper (uses the mic)" };
        switch(&p, &w.target, y - 250.0, whisper_label, s.flag("whisper", false), sel!(toggleWhisper:), mtm);
        text(&p, rect(36.0, y - 326.0, SW - 72.0, 18.0), &format!("Cleanup: {}", cleanup_desc()), TextStyle::new(12.0).alpha(0.55), mtm);
        capsule(&p, "Check permissions", t, sel!(checkPerms:), rect(36.0, 40.0, 170.0, 34.0), false, mtm);
        capsule(&p, "Open my data folder", t, sel!(openData:), rect(218.0, 40.0, 180.0, 34.0), false, mtm);
        text(&p, rect(410.0, 48.0, SW - 446.0, 18.0), "Never uploaded.", TextStyle::new(11.5).alpha(0.5), mtm);
        w.page = Some(p);
    });
}

pub fn show(u: &'static Ui) {
    let mtm = u.mtm;
    WIN.with(|cell| {
        if cell.borrow().is_none() {
            let (win, root) = glass_window(SW, SH, "Lipflow Settings", mtm);
            let target: Retained<SettingsTarget> = {
                let this = SettingsTarget::alloc(mtm).set_ivars(());
                // SAFETY: NSObject's init.
                unsafe { msg_send![super(this), init] }
            };
            *cell.borrow_mut() = Some(Win { win, root, page: None, target });
        }
    });
    build(u);
    WIN.with(|cell| {
        if let Some(w) = cell.borrow().as_ref() {
            present(&w.win, mtm);
        }
    });
}
