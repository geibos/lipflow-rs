//! Snap mis-read names to the names you're likely saying, by how they *look* on the lips
//! (`lipflow/visemes.py`).
//!
//! Lip reading can't hear, so "Miguel" comes out as MCCALL / MC HALE / NICKEL. Each word maps to
//! a coarse viseme sequence (lip/tongue shape classes, from spelling), and a span of 1–2 read
//! words is replaced by a candidate name when their sequences are close. Only uncommon, non-dictionary
//! words are eligible, so ordinary words are never swapped.

use std::collections::HashSet;
use std::io::Read;
use std::sync::OnceLock;

use super::edits;

/// Digraphs first, then single letters → viseme class.
const DIGRAPHS: [(&str, &str); 14] = [
    ("ch", "J"),
    ("sh", "J"),
    ("th", "D"),
    ("ph", "F"),
    ("ck", "K"),
    ("qu", "KW"),
    ("gu", "K"),
    ("mc", "MK"),
    ("wh", "W"),
    ("ng", "K"),
    ("ee", "I"),
    ("oo", "U"),
    ("ou", "U"),
    ("ea", "I"),
];

fn single(c: u8) -> &'static str {
    match c {
        b'p' | b'b' | b'm' => "M",
        b'f' | b'v' => "F",
        b't' | b'd' | b'n' | b's' | b'z' | b'l' | b'x' => "T",
        b'k' | b'g' | b'c' | b'q' => "K",
        b'j' | b'y' => "J",
        b'r' | b'w' => "W",
        b'a' | b'e' => "A",
        b'i' => "I",
        b'o' | b'u' => "U",
        _ => "",
    }
}

/// Coarse viseme string of a word, from its spelling; doubled classes collapse to one.
pub fn visemes(word: &str) -> String {
    let mut w: Vec<u8> = word.to_lowercase().bytes().filter(u8::is_ascii_lowercase).collect();
    if w.len() > 3 && w.ends_with(b"e") && !b"aeiou".contains(&w[w.len() - 2]) {
        w.pop(); // silent e: HALE is said "hail"
    }
    let mut out = String::new();
    let mut i = 0;
    while i < w.len() {
        if let Some((dg, v)) = DIGRAPHS.iter().find(|(dg, _)| w[i..].starts_with(dg.as_bytes())) {
            out.push_str(v);
            i += dg.len();
        } else {
            out.push_str(single(w[i]));
            i += 1;
        }
    }
    let mut dedup = String::with_capacity(out.len());
    for c in out.chars() {
        if !dedup.ends_with(c) {
            dedup.push(c); // doubled letters look like one
        }
    }
    dedup
}

static WEB2: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../lipflow/lipflow/data/web2-lower.txt.gz"));

fn load_dict() -> HashSet<String> {
    if let Ok(text) = std::fs::read_to_string("/usr/share/dict/words") {
        return text
            .lines()
            .filter(|l| l.chars().next().is_some_and(char::is_lowercase))
            .map(|l| l.trim().to_string())
            .collect();
    }
    // Windows: the same list (web2, 1934, public domain) ships inside Lipflow.
    let mut text = String::new();
    match flate2::read::GzDecoder::new(WEB2).read_to_string(&mut text) {
        Ok(_) => text.split_whitespace().map(str::to_string).collect(),
        Err(_) => HashSet::new(),
    }
}

fn dict() -> &'static HashSet<String> {
    static DICT: OnceLock<HashSet<String>> = OnceLock::new();
    DICT.get_or_init(load_dict)
}

const STEMS: [(&str, &str); 10] =
    [("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("er", ""), ("est", ""), ("ly", "")];

/// A real lowercase English word (macOS's `/usr/share/dict/words`; proper nouns there are
/// capitalised, so names like Mccall don't count). Misread names are almost always non-words.
pub fn is_word(w: &str) -> bool {
    let d = dict();
    let w = w.to_lowercase().replace('\'', "");
    if d.contains(&w) {
        return true;
    }
    // the dictionary has no inflections: try stems (planks→plank, served→serve, stopped→stop)
    let chars: Vec<char> = w.chars().collect();
    for (suf, add) in STEMS {
        let n = suf.len(); // suffixes are ASCII: bytes == chars
        if w.ends_with(suf) && chars.len() >= n + 3 {
            let mut stem: Vec<char> = chars[..chars.len() - n].to_vec();
            stem.extend(add.chars());
            let s: String = stem.iter().collect();
            if d.contains(&s) {
                return true;
            }
            let k = stem.len();
            if k > 3 && stem[k - 1] == stem[k - 2] && d.contains(&stem[..k - 1].iter().collect::<String>()) {
                return true;
            }
        }
    }
    false
}

const MAX_RATIO: f64 = 0.25;

/// Replace 1–2 word spans that look like a name on the lips. `is_common(word)` protects
/// everyday words. Works on the raw uppercase guess, before cleanup.
pub fn snap_names(text: &str, names: &[String], is_common: impl Fn(&str) -> bool) -> String {
    if names.is_empty() {
        return text.to_string();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    // a "name" that is an ordinary word (Balance, Flow) would turn real words into itself
    let targets: Vec<(&str, String)> = names
        .iter()
        .map(|n| (n.as_str(), visemes(n)))
        .filter(|(n, v)| v.len() >= 3 && !is_word(n))
        .collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let mut best: Option<(f64, usize, &str)> = None;
        for span in [2, 1] {
            let Some(chunk) = words.get(i..i + span) else { continue };
            if chunk.iter().any(|w| is_common(w)) {
                continue;
            }
            if chunk.iter().all(|w| is_word(w)) {
                continue; // "four", "planks": real words are left alone
            }
            let v = visemes(&chunk.concat());
            let joined = chunk.join(" ").to_lowercase();
            for (name, nv) in &targets {
                if joined == name.to_lowercase() {
                    continue;
                }
                if v.is_empty() || v.as_bytes()[0] != nv.as_bytes()[0] {
                    continue; // the first lip shape is the most visible
                }
                let r = edits(v.as_bytes(), nv.as_bytes()) as f64 / nv.len().max(v.len()) as f64;
                if r <= MAX_RATIO && best.is_none_or(|b| r < b.0) {
                    best = Some((r, span, name));
                }
            }
            if best.is_some() {
                break;
            }
        }
        match best {
            Some((_, span, name)) => {
                out.push(name.to_uppercase());
                i += span;
            }
            None => {
                out.push(words[i].to_string());
                i += 1;
            }
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::super::practice::HARVARD;
    use super::*;

    // Ported from lipflow/tests/test_cleanup.py
    #[test]
    fn names_snap_by_lip_shape_but_everyday_words_stay() {
        let common: HashSet<&str> = [
            "i", "am", "a", "my", "hello", "sending", "you", "message", "with", "new", "school", "tool", "made", "mistake", "in", "the", "meeting",
        ]
        .into_iter()
        .collect();
        let names = vec!["Miguel".to_string()];
        let snap = |t: &str| snap_names(t, &names, |w| common.contains(w.to_lowercase().as_str()));
        assert_eq!(snap("HELLO MCCALL I AM SENDING YOU A MESSAGE"), "HELLO MIGUEL I AM SENDING YOU A MESSAGE");
        assert_eq!(snap("HELLO MC HALE I AM SENDING"), "HELLO MIGUEL I AM SENDING");
        assert_eq!(snap("HELLO MIKAEL"), "HELLO MIGUEL");
        assert_eq!(snap("I MADE A MISTAKE IN THE MEETING"), "I MADE A MISTAKE IN THE MEETING");
    }

    #[test]
    fn snapping_leaves_every_harvard_sentence_alone() {
        let names: Vec<String> = ["Priya", "Miguel", "Vizcom", "Balance", "Flow"].iter().map(|s| s.to_string()).collect();
        for h in HARVARD {
            assert_eq!(snap_names(&h.to_uppercase(), &names, |_| false), h.to_uppercase());
        }
    }

    #[test]
    fn viseme_strings_and_stems() {
        assert_eq!(visemes("Miguel"), "MIKAT");
        assert_eq!(visemes("MCCALL"), "MKAT");
        assert!(is_word("planks") && is_word("served") && is_word("stopped"));
        assert!(!is_word("mccall"));
    }

    #[test]
    fn embedded_web2_decodes() {
        let mut text = String::new();
        flate2::read::GzDecoder::new(WEB2).read_to_string(&mut text).unwrap();
        assert!(text.split_whitespace().any(|w| w == "plank"));
    }
}
