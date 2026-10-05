//! Names from what you're typing into (`lipflow/context.py`, pure part only): the window title and
//! the text around the cursor are captured elsewhere; this pulls the names and terms out of them.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use super::pystr::is_upper;

const STOP_WORDS: &str = "a an the and or but if of to in on at for with from by as is are was were be been am i
you he she it we they me my your our their this that these those do does did have has had not no so
just can will would could should there here what when where who how why all about up out then than
too very really also like get got go going new re fwd inbox search compose sent drafts home today
monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december untitled message messages chat thread channel reply";

/// Words never taken as names (`context._STOP`).
pub static STOP: LazyLock<HashSet<&'static str>> = LazyLock::new(|| STOP_WORDS.split_whitespace().collect());

static SENTENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?\n|•·—\-–:]+").expect("static regex"));
static TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z][A-Za-z'\-]+").expect("static regex"));

const LIMIT: usize = 30;

/// Capitalised words that aren't sentence starts or common words: names, products, places.
/// At most 30, first seen first.
pub fn extract_names(texts: &[&str]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in texts {
        for sent in SENTENCE.split(t) {
            for m in TOKEN.find_iter(sent) {
                let w = m.as_str();
                // tokens start with an ASCII letter, so the first byte is the first char
                if !w.as_bytes()[0].is_ascii_uppercase() || (is_upper(w) && w.chars().count() > 4) {
                    continue;
                }
                let lower = w.to_lowercase();
                if STOP.contains(lower.as_str()) || w.chars().count() < 3 {
                    continue;
                }
                if seen.insert(lower) {
                    out.push(w.to_string());
                }
            }
        }
    }
    out.truncate(LIMIT);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from lipflow/tests/test_cleanup.py
    #[test]
    fn context_names_from_titles() {
        assert_eq!(extract_names(&["Miguel (DM) - Vizcom - Slack"]), ["Miguel", "Vizcom", "Slack"]);
        assert!(extract_names(&["Re: design review", "Thanks Priya, I think the plan works."]).contains(&"Priya".to_string()));
    }
}
