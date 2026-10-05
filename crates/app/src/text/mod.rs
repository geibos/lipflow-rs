//! Text side of Lipflow: cleanup rules and backends, your words and phrases, name snapping.

pub mod cleanup;
pub mod context;
pub mod corrections;
pub mod personal;
pub mod practice;
pub mod pystr;
pub mod rules;
pub mod visemes;
pub mod vocab;

use std::sync::LazyLock;

use regex::Regex;

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{L}\p{N}']+").expect("static regex"));

/// The dictation language: English reads with Auto-AVSR, Russian with MultiVSR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Ru,
}

impl Lang {
    /// Settings value "language"; without one, Russian when its model is installed.
    pub fn from_setting(value: Option<&str>, ru_available: bool) -> Self {
        match value {
            Some("en") => Self::En,
            Some("ru") => Self::Ru,
            _ if ru_available => Self::Ru,
            _ => Self::En,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ru => "ru",
        }
    }
}

/// Lowercase words: letters (any script), digits, apostrophes (`personal.words_of`).
pub fn words_of(text: &str) -> Vec<String> {
    WORD.find_iter(&text.to_lowercase()).map(|m| m.as_str().to_string()).collect()
}

/// Word-level Levenshtein distance between two token lists.
pub fn edits<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut d: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut prev = d[0];
        d[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cur = d[j + 1];
            d[j + 1] = (d[j + 1] + 1).min(d[j] + 1).min(prev + usize::from(x != y));
            prev = cur;
        }
    }
    d[b.len()]
}

/// (word errors, reference words).
pub fn wer(hyp: &str, reference: &str) -> (usize, usize) {
    let (h, r) = (words_of(hyp), words_of(reference));
    (edits(&h, &r), r.len())
}

fn srt_time(s: &str) -> f64 {
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.len() != 3 {
        return 0.0;
    }
    let (sec, ms) = parts[2].split_once(',').unwrap_or((parts[2], "0"));
    let ms: String = format!("{ms:0<3}").chars().take(3).collect();
    parts[0].parse::<f64>().unwrap_or(0.0) * 3600.0
        + parts[1].parse::<f64>().unwrap_or(0.0) * 60.0
        + sec.parse::<f64>().unwrap_or(0.0)
        + ms.parse::<f64>().unwrap_or(0.0) / 1000.0
}

static SPEAKER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^The President:\s*").expect("static regex"));
static SENTENCE_END: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?]\s*$").expect("static regex"));

/// Subtitle cues merged into sentences: (start, end, UPPERCASE TEXT) — `lipflow/bench.py`.
pub fn srt_sentences(srt: &str) -> Vec<(f64, f64, String)> {
    let srt = srt.replace("\r\n", "\n");
    let mut cues = Vec::new();
    for block in srt.split("\n\n") {
        let lines: Vec<&str> = block.trim().lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() >= 3 && lines[1].contains("-->") {
            let (a, b) = lines[1].split_once("-->").unwrap_or(("", ""));
            cues.push((srt_time(a), srt_time(b), lines[2..].join(" ")));
        }
    }
    let mut out = Vec::new();
    let (mut cur, mut start) = (String::new(), None);
    for (a, b, text) in cues {
        let text = SPEAKER.replace(&text, "").into_owned();
        let s = *start.get_or_insert(a);
        cur.push(' ');
        cur.push_str(&text);
        if SENTENCE_END.is_match(&text) {
            let words: Vec<String> = words_of(&cur).into_iter().filter(|w| !w.chars().all(|c| c.is_ascii_digit())).map(|w| w.to_uppercase()).collect();
            if (4..=30).contains(&words.len()) && b - s < 12.0 {
                out.push((s, b, words.join(" ")));
            }
            cur.clear();
            start = None;
        }
    }
    out
}
