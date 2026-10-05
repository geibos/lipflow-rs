//! Insert text at the cursor of the focused app: clipboard + ⌘V, then restore (`lipflow/paste.py`).

use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
use objc2_core_graphics::{CGEvent, CGEventField, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation};
use objc2_foundation::NSString;

use super::hotkey::MARK;
use super::main_thread::on_main_after;

const KEY_V: u16 = 9; // kVK_ANSI_V

fn press_cmd_v() {
    let src = CGEventSource::new(CGEventSourceStateID::HIDSystemState);
    for down in [true, false] {
        if let Some(ev) = CGEvent::new_keyboard_event(src.as_deref(), KEY_V, down) {
            CGEvent::set_flags(Some(&ev), CGEventFlags::MaskCommand); // never inherit the held modifier
            CGEvent::set_integer_value_field(Some(&ev), CGEventField::EventSourceUserData, MARK);
            CGEvent::post(CGEventTapLocation::AnnotatedSessionEventTap, Some(&ev));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn string_type() -> Option<&'static NSString> {
    // SAFETY: reading an immutable AppKit constant.
    Some(unsafe { NSPasteboardTypeString })
}

/// Paste `text`; the previous clipboard text comes back after `restore_after` seconds unless
/// something newer was copied meanwhile. Main thread.
pub fn paste_text(text: &str, restore_after: f64) {
    if text.is_empty() {
        return;
    }
    let Some(ty) = string_type() else { return };
    let pb = NSPasteboard::generalPasteboard();
    let saved = pb.stringForType(ty).map(|s| s.to_string());
    pb.clearContents();
    pb.setString_forType(&NSString::from_str(text), ty);
    let change = pb.changeCount();
    press_cmd_v();
    if let Some(saved) = saved {
        on_main_after(restore_after, move || {
            let Some(ty) = string_type() else { return };
            let pb = NSPasteboard::generalPasteboard();
            if pb.changeCount() == change {
                pb.clearContents();
                pb.setString_forType(&NSString::from_str(&saved), ty);
            }
        });
    }
}

pub fn copy_text(text: &str) {
    let Some(ty) = string_type() else { return };
    let pb = NSPasteboard::generalPasteboard();
    pb.clearContents();
    pb.setString_forType(&NSString::from_str(text), ty);
}
