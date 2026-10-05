//! Push-to-talk timing (`lipflow/ptt.py`), independent of how keys are observed.
//!
//! Hold the key to dictate, release to type. Double-tap it to start hands-free mode; tap again
//! to stop. Esc cancels the current recording. Any other key pressed while the push-to-talk key
//! is held makes it a shortcut (Option+letter, Ctrl+C…), so that cancels too.

/// Seconds between taps.
pub const DOUBLE_TAP: f64 = 0.35;
/// A press shorter than this is a tap, not a hold.
pub const TAP_MAX: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Start { hands_free: bool },
    Stop,
    Cancel { silent: bool },
}

#[derive(Default, Debug)]
pub struct PushToTalk {
    down: bool,
    down_at: f64,
    last_tap: f64,
    pub hands_free: bool,
    active: bool,
}

impl PushToTalk {
    /// A key other than the push-to-talk key went down.
    pub fn other_key(&mut self, is_esc: bool) -> Option<Action> {
        if is_esc && self.active {
            self.active = false;
            self.hands_free = false;
            return Some(Action::Cancel { silent: false });
        }
        if self.down && !self.hands_free {
            self.down = false;
            if self.active {
                self.active = false;
                return Some(Action::Cancel { silent: false });
            }
        }
        None
    }

    pub fn key_down(&mut self, now: f64) -> Option<Action> {
        if self.down {
            return None; // key repeat
        }
        self.down = true;
        self.down_at = now;
        if self.hands_free {
            return None; // stop happens on release
        }
        if !self.active {
            self.active = true;
            return Some(Action::Start { hands_free: false });
        }
        None
    }

    pub fn key_up(&mut self, now: f64) -> Option<Action> {
        if !self.down {
            return None;
        }
        self.down = false;
        let held = now - self.down_at;
        if self.hands_free {
            self.hands_free = false;
            self.active = false;
            return Some(Action::Stop);
        }
        if held < TAP_MAX {
            if now - self.last_tap < DOUBLE_TAP {
                self.hands_free = true; // second tap: keep listening until the next tap
                self.last_tap = 0.0;
                return Some(Action::Start { hands_free: true });
            }
            self.last_tap = now;
            self.active = false;
            return Some(Action::Cancel { silent: true });
        }
        if self.active {
            self.active = false;
            return Some(Action::Stop);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    // Ported from lipflow/tests/test_ptt.py
    use super::*;

    const T: f64 = 1_000_000.0; // a realistic clock: last_tap starts at 0

    fn run(events: &[(&str, f64)]) -> (PushToTalk, Vec<Action>) {
        let mut p = PushToTalk::default();
        let mut log = Vec::new();
        for &(e, t) in events {
            let a = match e {
                "down" => p.key_down(T + t),
                "up" => p.key_up(T + t),
                "esc" => p.other_key(true),
                "key" => p.other_key(false),
                _ => unreachable!("test event"),
            };
            log.extend(a);
        }
        (p, log)
    }

    #[test]
    fn hold_to_talk() {
        let (_, log) = run(&[("down", 0.0), ("up", TAP_MAX + 0.1)]);
        assert_eq!(log, [Action::Start { hands_free: false }, Action::Stop]);
    }

    #[test]
    fn key_repeat_is_one_press() {
        let (_, log) = run(&[("down", 0.0), ("down", 0.05), ("down", 0.1), ("down", 0.15), ("up", 1.0)]);
        assert_eq!(log, [Action::Start { hands_free: false }, Action::Stop]);
    }

    #[test]
    fn single_tap_is_silently_ignored() {
        let (_, log) = run(&[("down", 0.0), ("up", 0.1)]);
        assert_eq!(log, [Action::Start { hands_free: false }, Action::Cancel { silent: true }]);
    }

    #[test]
    fn double_tap_enters_hands_free_until_next_tap() {
        let (p, log) = run(&[("down", 0.0), ("up", 0.1), ("down", 0.2), ("up", 0.3)]);
        assert_eq!(log.last(), Some(&Action::Start { hands_free: true }));
        assert!(p.hands_free);
        let (p, log) = run(&[("down", 0.0), ("up", 0.1), ("down", 0.2), ("up", 0.3), ("down", 3.0), ("up", 3.1)]);
        assert_eq!(log.last(), Some(&Action::Stop));
        assert!(!p.hands_free);
    }

    #[test]
    fn slow_taps_are_not_a_double_tap() {
        let (_, log) = run(&[("down", 0.0), ("up", 0.1), ("down", 0.1 + DOUBLE_TAP + 0.1), ("up", 0.2 + DOUBLE_TAP + 0.1)]);
        assert!(!log.contains(&Action::Start { hands_free: true }));
    }

    #[test]
    fn shortcut_cancels_and_escape_cancels() {
        let (_, log) = run(&[("down", 0.0), ("key", 0.0), ("up", 1.0)]);
        assert_eq!(log, [Action::Start { hands_free: false }, Action::Cancel { silent: false }]);
        let (_, log) = run(&[("down", 0.0), ("esc", 0.0)]);
        assert_eq!(log.last(), Some(&Action::Cancel { silent: false }));
    }

    #[test]
    fn escape_cancels_hands_free() {
        let (p, log) = run(&[("down", 0.0), ("up", 0.1), ("down", 0.2), ("up", 0.3), ("esc", 0.3)]);
        assert_eq!(log.last(), Some(&Action::Cancel { silent: false }));
        assert!(!p.hands_free);
    }
}
