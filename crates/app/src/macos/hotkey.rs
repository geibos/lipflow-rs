//! Global push-to-talk key via a listen-only Quartz event tap on the main run loop
//! (`lipflow/hotkey.py`). Needs Input Monitoring for whatever app launched Lipflow.

use std::cell::Cell;
use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::{Result, bail};
use objc2_core_foundation::{CFMachPort, CFRetained, CFRunLoop, kCFRunLoopCommonModes};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventMask, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventTapProxy, CGEventType,
};

/// name -> (virtual keycode, device-specific modifier flag mask)
pub const KEYS: [(&str, u16, u64); 5] = [
    ("right_option", 61, 0x0000_0040),
    ("left_option", 58, 0x0000_0020),
    ("right_command", 54, 0x0000_0010),
    ("right_control", 62, 0x0000_2000),
    ("fn", 63, 0x0080_0000),
];
const ESC: i64 = 53;
/// `kCGEventSourceUserData` on every event Lipflow posts, so its own tap ignores them.
pub const MARK: i64 = 0x11FF10;

#[derive(Clone, Copy, Debug)]
pub enum KeyEvent {
    PttDown,
    PttUp,
    Other { esc: bool },
}

pub fn key_spec(name: &str) -> Option<(u16, u64)> {
    KEYS.iter().find(|k| k.0 == name).map(|k| (k.1, k.2))
}

struct TapState {
    key: Cell<(u16, u64)>,
    tap: Cell<Option<NonNull<CFMachPort>>>,
    on_event: Box<dyn Fn(KeyEvent)>,
}

/// The installed tap. Lives for the whole app; dropping it is not supported.
pub struct Hotkey {
    state: &'static TapState,
    _port: CFRetained<CFMachPort>,
}

unsafe extern "C-unwind" fn callback(_proxy: CGEventTapProxy, etype: CGEventType, event: NonNull<CGEvent>, user: *mut c_void) -> *mut CGEvent {
    // SAFETY: `user` is the leaked `&'static TapState` passed to `tap_create`; the tap only runs
    // on the main run loop, the same thread that owns the `Cell`s.
    let st = unsafe { &*user.cast::<TapState>() };
    // SAFETY: CoreGraphics passes a valid event for the duration of the callback.
    let ev = unsafe { event.as_ref() };
    if etype == CGEventType::TapDisabledByTimeout || etype == CGEventType::TapDisabledByUserInput {
        if let Some(port) = st.tap.get() {
            // SAFETY: the port is kept alive by `Hotkey::_port` for the app's lifetime.
            CGEvent::tap_enable(unsafe { port.as_ref() }, true);
        }
        return event.as_ptr();
    }
    if CGEvent::integer_value_field(Some(ev), CGEventField::EventSourceUserData) == MARK {
        return event.as_ptr(); // our own typing / paste
    }
    let code = CGEvent::integer_value_field(Some(ev), CGEventField::KeyboardEventKeycode);
    if etype == CGEventType::KeyDown {
        // Option+letter is a real shortcut (e.g. special characters), not dictation.
        (st.on_event)(KeyEvent::Other { esc: code == ESC });
        return event.as_ptr();
    }
    let (keycode, mask) = st.key.get();
    if code != i64::from(keycode) {
        return event.as_ptr();
    }
    if CGEvent::flags(Some(ev)).0 & mask != 0 {
        (st.on_event)(KeyEvent::PttDown);
    } else {
        (st.on_event)(KeyEvent::PttUp);
    }
    event.as_ptr()
}

impl Hotkey {
    /// Install on the main run loop. Call on the main thread.
    pub fn install(key: &str, on_event: impl Fn(KeyEvent) + 'static) -> Result<Self> {
        let Some(spec) = key_spec(key) else { bail!("unknown key {key:?}") };
        let state: &'static TapState = Box::leak(Box::new(TapState { key: Cell::new(spec), tap: Cell::new(None), on_event: Box::new(on_event) }));
        let mask: CGEventMask = (1 << CGEventType::FlagsChanged.0) | (1 << CGEventType::KeyDown.0);
        // SAFETY: `callback` matches CGEventTapCallBack and `state` is 'static.
        let port = unsafe {
            CGEvent::tap_create(
                CGEventTapLocation::SessionEventTap,
                CGEventTapPlacement::HeadInsertEventTap,
                CGEventTapOptions::ListenOnly,
                mask,
                Some(callback),
                std::ptr::from_ref(state).cast_mut().cast(),
            )
        };
        let Some(port) = port else {
            bail!("Couldn't listen for the hotkey. Allow Lipflow in System Settings → Privacy & Security → Input Monitoring, then restart it.")
        };
        state.tap.set(Some(NonNull::from(&*port)));
        let src = CFMachPort::new_run_loop_source(None, Some(&port), 0);
        if let Some(rl) = CFRunLoop::main() {
            // SAFETY: reading an immutable CoreFoundation constant.
            let modes = unsafe { kCFRunLoopCommonModes };
            rl.add_source(src.as_deref(), modes);
        }
        CGEvent::tap_enable(&port, true);
        Ok(Self { state, _port: port })
    }

    pub fn set_key(&self, key: &str) -> bool {
        match key_spec(key) {
            Some(spec) => {
                self.state.key.set(spec);
                true
            }
            None => false,
        }
    }
}
