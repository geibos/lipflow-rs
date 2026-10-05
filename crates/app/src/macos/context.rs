//! What you're typing into (`lipflow/context.py`) and learning from your corrections
//! (`lipflow/corrections.py: Watcher`), through the Accessibility API. Read once when the key
//! goes down; nothing is saved except a corrected clip. Main thread only (AX elements stay here).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2_app_kit::NSWorkspace;
use objc2_core_foundation::{CFRetained, CFString, CFType};

use super::main_thread::on_main_after;
use crate::{data, paths, text};

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXUIElementCreateApplication(pid: i32) -> *mut c_void;
    fn AXUIElementCopyAttributeValue(element: *const c_void, attribute: *const c_void, value: *mut *const c_void) -> i32;
}

/// An AXUIElement (a CFType).
struct Element(CFRetained<CFType>);

fn ax_attr(el: &CFType, name: &str) -> Option<CFRetained<CFType>> {
    let attr = CFString::from_str(name);
    let mut out: *const c_void = std::ptr::null();
    // SAFETY: `el` is a live AXUIElement, `attr` a live CFString, `out` a valid out-pointer;
    // on success (0) AX returns a +1 retained CFType that we adopt.
    let err = unsafe { AXUIElementCopyAttributeValue(std::ptr::from_ref(el).cast(), std::ptr::from_ref::<CFString>(&attr).cast(), &mut out) };
    if err != 0 {
        return None;
    }
    // SAFETY: a non-null +1 reference from a Copy function.
    NonNull::new(out.cast_mut().cast::<CFType>()).map(|p| unsafe { CFRetained::from_raw(p) })
}

fn ax_string(el: &CFType, name: &str) -> Option<String> {
    ax_attr(el, name).and_then(|v| v.downcast::<CFString>().ok()).map(|s| s.to_string())
}

thread_local! {
    /// The focused text field and its contents before the paste (memory only).
    static FOCUSED: RefCell<Option<(Element, String)>> = const { RefCell::new(None) };
}

/// Names for this dictation (read by the model worker).
static NAMES: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn names() -> Vec<String> {
    NAMES.lock().map(|n| n.clone()).unwrap_or_default()
}

/// Snapshot of the frontmost app when the key goes down. Never fails: context is a bonus.
pub fn capture_into(_core: &super::app::Core) {
    let (mut app_name, mut title, mut near) = (String::new(), String::new(), String::new());
    let mut focused = None;
    if let Some(app) = NSWorkspace::sharedWorkspace().frontmostApplication() {
        app_name = app.localizedName().map(|s| s.to_string()).unwrap_or_default();
        // SAFETY: creating an AX handle for a pid; the result is +1 (adopted below).
        let raw = unsafe { AXUIElementCreateApplication(app.processIdentifier()) };
        // SAFETY: non-null +1 reference.
        if let Some(ax_app) = NonNull::new(raw.cast::<CFType>()).map(|p| unsafe { CFRetained::from_raw(p) }) {
            if let Some(win) = ax_attr(&ax_app, "AXFocusedWindow") {
                title = ax_string(&win, "AXTitle").unwrap_or_default();
            }
            if let Some(el) = ax_attr(&ax_app, "AXFocusedUIElement") {
                if let Some(val) = ax_string(&el, "AXValue") {
                    near = val.chars().rev().take(600).collect::<Vec<_>>().into_iter().rev().collect();
                    focused = Some((Element(el.clone()), val));
                }
                if near.is_empty() {
                    near = ax_string(&el, "AXPlaceholderValue").unwrap_or_default(); // e.g. Slack's "Message Miguel"
                }
            }
        }
    }
    let names = text::context::extract_names(&[&title, &near]);
    if !names.is_empty() {
        println!("[lipflow] context: {app_name}, {} names", names.len());
    }
    if let Ok(mut n) = NAMES.lock() {
        *n = names;
    }
    FOCUSED.with(|f| *f.borrow_mut() = focused);
}

const CHECKS: [f64; 3] = [4.0, 12.0, 30.0];
const KEEP: usize = 500;

thread_local! {
    /// Correction watchers by id; the queued closures carry only the id, so nothing
    /// main-thread-bound crosses threads.
    static WATCHES: RefCell<HashMap<u64, Watch>> = RefCell::new(HashMap::new());
}

static NEXT_WATCH: AtomicU64 = AtomicU64::new(1);

/// After a paste, re-read the same field a few times; a small fix of the pasted words becomes a
/// training clip (mouth crops + your corrected sentence).
pub fn watch_corrections(pasted: &str, rois: Vec<u8>, t: usize, raw: Vec<String>) {
    let Some((el, before)) = FOCUSED.with(|f| f.borrow_mut().take()) else { return };
    let id = NEXT_WATCH.fetch_add(1, Ordering::Relaxed);
    WATCHES.with(|w| w.borrow_mut().insert(id, Watch { el, before, pasted: pasted.trim().to_string(), rois, t, raw, best: None }));
    for (i, &secs) in CHECKS.iter().enumerate() {
        let last = i == CHECKS.len() - 1;
        on_main_after(secs, move || check(id, last));
    }
}

struct Watch {
    el: Element,
    before: String,
    pasted: String,
    rois: Vec<u8>,
    t: usize,
    raw: Vec<String>,
    best: Option<String>,
}

fn check(id: u64, last: bool) {
    let finished = WATCHES.with(|ws| {
        let mut ws = ws.borrow_mut();
        let w = ws.get_mut(&id)?;
        let val = ax_string(&w.el.0, "AXValue");
        if let Some(v) = &val
            && let Some(fix) = text::corrections::find_correction(&w.pasted, &text::corrections::inserted_span(&w.before, v))
        {
            w.best = Some(fix);
        }
        if last || val.is_none() { ws.remove(&id) } else { None }
    });
    if let Some(w) = finished
        && let Some(best) = w.best
    {
        let dir = paths::clips("corrections");
        match data::save_clip(&dir, &w.rois, w.t, &best, &w.raw, &[("pasted", &w.pasted)]) {
            Ok(_) => println!("[lipflow] learned from your correction ({} words)", text::words_of(&best).len()),
            Err(e) => eprintln!("[lipflow] saving correction: {e}"),
        }
        data::prune(&dir, KEEP);
    }
}
