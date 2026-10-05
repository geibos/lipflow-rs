//! Hopping onto the AppKit main thread from worker threads.

use std::time::Duration;

use dispatch2::{DispatchQueue, DispatchTime};

pub fn on_main(f: impl FnOnce() + Send + 'static) {
    DispatchQueue::main().exec_async(f);
}

pub fn on_main_after(secs: f64, f: impl FnOnce() + Send + 'static) {
    let when = DispatchTime::try_from(Duration::from_secs_f64(secs.max(0.0))).unwrap_or(DispatchTime::NOW);
    // `after` only fails for invalid times, which `try_from` already excluded.
    let _ = DispatchQueue::main().after(when, f);
}
