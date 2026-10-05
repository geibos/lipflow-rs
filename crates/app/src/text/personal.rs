//! Learn from what you already say (`lipflow/personal.py`): import your Wispr Flow history.
//!
//! Wispr Flow's local database is read read-only and just the text of your dictations is written
//! to `phrases.txt` ([`crate::paths::phrases`]). From those phrases Lipflow builds a word-pair
//! model of how you talk, a lookup of your past sentences closest to a new guess, and suggested
//! names/terms appended to `words.txt` for you to review.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context as _, bail};
use regex::Regex;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};

use super::pystr::{is_lower, is_upper};
use super::{vocab, words_of};

/// Where Wispr Flow keeps its data (Electron app data folder).
pub fn wispr_dir() -> PathBuf {
    if cfg!(target_os = "windows") {
        let base = std::env::var_os("APPDATA").or_else(|| std::env::var_os("USERPROFILE")).map_or_else(|| PathBuf::from("."), PathBuf::from);
        return base.join("Wispr Flow");
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join("Library/Application Support/Wispr Flow")
}

/// Counter that remembers first-seen order, so `most_common` breaks ties like Python's `Counter`.
#[derive(Debug, Default, Clone)]
struct Counter {
    index: HashMap<String, usize>,
    items: Vec<(String, usize)>,
}

impl Counter {
    fn add(&mut self, key: &str, n: usize) {
        match self.index.get(key) {
            Some(&i) => self.items[i].1 += n,
            None => {
                self.index.insert(key.to_string(), self.items.len());
                self.items.push((key.to_string(), n));
            }
        }
    }

    fn get(&self, key: &str) -> usize {
        self.index.get(key).map_or(0, |&i| self.items[i].1)
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn total(&self) -> usize {
        self.items.iter().map(|x| x.1).sum()
    }

    /// The `n` most common, by count descending, ties in first-seen order.
    fn most_common(&self, n: usize) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> = self.items.iter().map(|(k, c)| (k.as_str(), *c)).collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        v.truncate(n);
        v
    }
}

// -- import ---------------------------------------------------------------------------

/// Score a column: long-ish natural-language strings, not ids/JSON/paths. `None` = not text.
fn looks_like_dictation(values: &[Option<String>]) -> f64 {
    let good = values
        .iter()
        .flatten()
        .filter(|v| {
            let n = v.split_whitespace().count();
            let len = v.chars().count();
            let letters = v.chars().filter(|c| c.is_alphabetic() || c.is_whitespace()).count() as f64 / len.max(1) as f64;
            let start = v.trim_start();
            (2..=400).contains(&n) && letters > 0.85 && !["{", "[", "/", "http"].iter().any(|p| start.starts_with(p))
        })
        .count();
    good as f64 / values.len().max(1) as f64
}

/// A text column that looks like dictated sentences.
#[derive(Debug, Clone, PartialEq)]
pub struct TextColumn {
    pub table: String,
    pub column: String,
    pub score: f64,
    pub rows: i64,
}

fn open_ro(db: &Path) -> rusqlite::Result<Connection> {
    let uri = format!("file:{}?mode=ro", db.display());
    Connection::open_with_flags(uri, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX)
}

/// Text values as `Some`, other non-null values as `None`; invalid UTF-8 is an error, as in Python.
fn text_values(con: &Connection, sql: &str) -> rusqlite::Result<Vec<Option<String>>> {
    let mut st = con.prepare(sql)?;
    let rows = st.query_map([], |r| match r.get_ref(0)? {
        ValueRef::Text(b) => std::str::from_utf8(b)
            .map(|s| Some(s.to_string()))
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))),
        _ => Ok(None),
    })?;
    rows.collect()
}

static PRIO_FINAL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)format|final|edit|clean").expect("static regex"));
static PRIO_RAW: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)asr|raw|transcri").expect("static regex"));

/// Text columns that look like dictated sentences, best first.
pub fn find_text_columns(db: &Path) -> rusqlite::Result<Vec<TextColumn>> {
    let con = open_ro(db)?;
    let mut out = Vec::new();
    let tables: Vec<String> = con.prepare("select name from sqlite_master where type='table'")?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
    for t in &tables {
        let cols: Vec<String> = con.prepare(&format!("pragma table_info(\"{t}\")"))?.query_map([], |r| r.get(1))?.collect::<Result<_, _>>()?;
        for c in &cols {
            let Ok(vals) = text_values(&con, &format!("select \"{c}\" from \"{t}\" where \"{c}\" is not null limit 300")) else {
                continue;
            };
            if vals.len() < 5 {
                continue;
            }
            let score = looks_like_dictation(&vals);
            if score > 0.6 {
                let rows: i64 = con.query_row(&format!("select count(*) from \"{t}\" where \"{c}\" is not null"), [], |r| r.get(0))?;
                out.push(TextColumn { table: t.clone(), column: c.clone(), score, rows });
            }
        }
    }
    // Prefer the final, formatted text over raw ASR when both exist
    let prio = |name: &str| {
        if PRIO_FINAL.is_match(name) {
            0
        } else if PRIO_RAW.is_match(name) {
            2
        } else {
            1
        }
    };
    out.sort_by(|x, y| prio(&x.column).cmp(&prio(&y.column)).then(y.score.total_cmp(&x.score)).then(y.rows.cmp(&x.rows)));
    Ok(out)
}

/// Every text value of `table.column`.
pub fn read_column(db: &Path, table: &str, column: &str) -> rusqlite::Result<Vec<String>> {
    let con = open_ro(db)?;
    Ok(text_values(&con, &format!("select \"{column}\" from \"{table}\" where \"{column}\" is not null"))?.into_iter().flatten().collect())
}

/// Result of an import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportStats {
    pub phrases: usize,
    pub words: usize,
    pub new_names: Vec<String>,
    /// `"<db file> → <table>.<column>"`, for imports from a database.
    pub source: Option<String>,
}

/// Files under `dir`, recursively, skipping hidden names (like Python's `glob("**/*")`).
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        if e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let p = e.path();
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            walk(&p, out);
        }
        out.push(p);
    }
}

/// Import Wispr Flow's history from `folder` into `phrases` (and suggested names into `words`).
pub fn import_wispr(folder: &Path, phrases: &Path, words: &Path) -> anyhow::Result<ImportStats> {
    let mut all = Vec::new();
    walk(folder, &mut all);
    let mut dbs: Vec<PathBuf> = all
        .into_iter()
        .filter(|p| {
            let s = p.to_string_lossy();
            [".sqlite", ".db", ".sqlite3"].iter().any(|ext| s.ends_with(ext)) && p.is_file()
        })
        .collect();
    dbs.sort();
    if dbs.is_empty() {
        bail!("No Wispr Flow database found in {}", folder.display());
    }
    let mut best: Option<(PathBuf, TextColumn)> = None;
    for db in dbs {
        let Ok(cols) = find_text_columns(&db) else { continue };
        if let Some(top) = cols.into_iter().next()
            && best.as_ref().is_none_or(|b| top.rows > b.1.rows)
        {
            best = Some((db, top));
        }
    }
    let Some((db, col)) = best else {
        bail!("Found Wispr Flow's database but no column that looks like dictated text");
    };
    let texts = read_column(&db, &col.table, &col.column).with_context(|| format!("reading {}", db.display()))?;
    let mut stats = save_phrases(&texts, phrases, words)?;
    let name = db.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    stats.source = Some(format!("{name} → {}.{}", col.table, col.column));
    Ok(stats)
}

static LINE_BREAKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\r\n]+").expect("static regex"));

/// Write unique lines of 2+ words to `phrases`, then suggest names into `words`.
pub fn save_phrases(texts: &[String], phrases: &Path, words: &Path) -> io::Result<ImportStats> {
    let mut seen = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for t in texts {
        for line in LINE_BREAKS.split(t) {
            let line = line.trim();
            if words_of(line).len() >= 2 && seen.insert(line.to_lowercase()) {
                out.push(line.to_string());
            }
        }
    }
    if let Some(dir) = phrases.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(phrases, out.join("\n") + "\n")?;
    let new_names = suggest_words(&out, words, 3)?;
    Ok(ImportStats { phrases: out.len(), words: out.iter().map(|p| words_of(p).len()).sum(), new_names, source: None })
}

static CAP_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z][A-Za-z0-9'\-]+").expect("static regex"));

/// Words you capitalise mid-sentence again and again (names, products) → appended to `words`.
pub fn suggest_words(phrases: &[String], words: &Path, min_count: usize) -> io::Result<Vec<String>> {
    let (mut caps, mut lower) = (Counter::default(), Counter::default());
    for p in phrases {
        for (i, m) in CAP_TOKEN.find_iter(p).enumerate() {
            let w = m.as_str();
            // tokens start with an ASCII letter, so the first byte is the first char
            if w.as_bytes()[0].is_ascii_uppercase() && i > 0 && !is_upper(w) && !["i", "i'm", "i'll", "i've", "i'd"].contains(&w.to_lowercase().as_str()) {
                caps.add(w, 1);
            } else if is_lower(w) {
                lower.add(w, 1);
            }
        }
    }
    let have: HashSet<String> = vocab::load(words)?.iter().map(|w| w.to_lowercase()).collect();
    let new: Vec<String> = caps
        .most_common(200)
        .into_iter()
        .filter(|&(w, n)| n >= min_count && lower.get(&w.to_lowercase()) < n && !have.contains(&w.to_lowercase()))
        .map(|(w, _)| w.to_string())
        .collect();
    if !new.is_empty() {
        let mut f = fs::OpenOptions::new().append(true).open(words)?;
        write!(f, "\n# from your Wispr Flow history (delete any that are wrong)\n{}\n", new.join("\n"))?;
    }
    Ok(new)
}

// -- use ------------------------------------------------------------------------------

const STOP_WORDS: &str = "a an the and or but if of to in on at for with from by as is are was were be been am i
        you he she it we they me my your our their this that these those do does did have has had
        not no so just can will would could should there here what when where who how why all
        about up out then than too very really also like get got go going im it's i'm don't";

static STOP: LazyLock<HashSet<&'static str>> = LazyLock::new(|| STOP_WORDS.split_whitespace().collect());

/// Bigram model + nearest-phrase lookup over your own phrases.
#[derive(Debug, Default, Clone)]
pub struct Personal {
    phrases: Vec<String>,
    /// Word counts, `<s>` included once per phrase.
    uni: Counter,
    bi: HashMap<(String, String), usize>,
    index: Vec<HashSet<String>>,
    total: usize,
    df: HashMap<String, usize>,
}

impl Personal {
    /// Phrases from `path`, one per line; a missing file gives an empty model.
    pub fn load(path: &Path) -> io::Result<Self> {
        match fs::read_to_string(path) {
            Ok(text) => Ok(Self::from_phrases(text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    pub fn from_phrases(phrases: Vec<String>) -> Self {
        let mut me = Self { phrases, ..Self::default() };
        for p in &me.phrases {
            let mut w = vec!["<s>".to_string()];
            w.extend(words_of(p));
            for x in &w {
                me.uni.add(x, 1);
            }
            for pair in w.windows(2) {
                *me.bi.entry((pair[0].clone(), pair[1].clone())).or_default() += 1;
            }
            me.index.push(w[1..].iter().cloned().collect());
        }
        me.total = me.uni.total();
        for s in &me.index {
            for x in s {
                *me.df.entry(x.clone()).or_default() += 1;
            }
        }
        me
    }

    pub fn phrases(&self) -> &[String] {
        &self.phrases
    }

    /// How often you used `word` (`personal.uni[word]`).
    pub fn count(&self, word: &str) -> usize {
        self.uni.get(word)
    }

    /// No phrases (Python's `not personal`).
    pub fn is_empty(&self) -> bool {
        self.phrases.is_empty()
    }

    /// Your most-used distinctive words (function words dropped), for the model's prompt.
    pub fn common_words(&self, n: usize) -> Vec<String> {
        self.uni
            .most_common(n + 200)
            .into_iter()
            .filter(|&(w, _)| w != "<s>" && !STOP.contains(w) && w.chars().count() > 2)
            .take(n)
            .map(|(w, _)| w.to_string())
            .collect()
    }

    /// Used at least twice.
    pub fn knows(&self, word: &str) -> bool {
        self.uni.get(word) >= 2
    }

    /// Average per-word log P under an interpolated bigram model (higher = more like you).
    pub fn logprob(&self, text: &str) -> f64 {
        let mut w = vec!["<s>".to_string()];
        w.extend(words_of(text));
        if w.len() < 2 || self.total == 0 {
            return 0.0;
        }
        let v = (self.uni.len() + 1) as f64;
        let total = self.total as f64;
        let mut lp = 0.0;
        for pair in w.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            let p_uni = (self.uni.get(b) as f64 + 0.1) / (total + 0.1 * v);
            let ua = self.uni.get(a);
            let p_bi = if ua > 0 { self.bi.get(&(a.clone(), b.clone())).copied().unwrap_or(0) as f64 / ua as f64 } else { 0.0 };
            lp += (0.6 * p_bi + 0.4 * p_uni).ln();
        }
        lp / (w.len() - 1) as f64
    }

    /// Beam order is a prior (rank r costs r*0.35); your phrasing decides close calls.
    pub fn rerank(&self, candidates: &[String]) -> Vec<String> {
        const PRIOR: f64 = 0.35;
        if self.is_empty() || candidates.len() < 2 {
            return candidates.to_vec();
        }
        let mut scored: Vec<(f64, usize, &String)> = candidates.iter().enumerate().map(|(i, c)| (self.logprob(c) - i as f64 * PRIOR, i, c)).collect();
        scored.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        scored.into_iter().map(|(_, _, c)| c.clone()).collect()
    }

    /// Up to `k` past sentences sharing the most (rarity-weighted) words with `text`.
    pub fn similar(&self, text: &str, k: usize) -> Vec<String> {
        let q: HashSet<String> = words_of(text).into_iter().collect();
        if self.is_empty() || q.is_empty() {
            return Vec::new();
        }
        let n = self.index.len() as f64;
        let mut scores: Vec<(f64, usize)> = Vec::new();
        for (i, s) in self.index.iter().enumerate() {
            let common: Vec<&String> = q.intersection(s).collect();
            if !common.is_empty() {
                scores.push((common.iter().map(|x| (n / self.df[*x] as f64).ln()).sum(), i));
            }
        }
        // Python sorts (score, i) tuples descending: ties go to the later phrase
        scores.sort_by(|x, y| y.0.total_cmp(&x.0).then(y.1.cmp(&x.1)));
        scores.into_iter().take(k).filter(|s| s.0 > 2.0).map(|(_, i)| self.phrases[i].clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("lipflow-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A Wispr-like DB: ids, JSON blobs, a raw ASR column and a formatted-text column.
    fn fake_wispr(folder: &Path) {
        fs::create_dir(folder).unwrap();
        let con = Connection::open(folder.join("flow.sqlite")).unwrap();
        con.execute("create table History (id text, app text, asrText text, formattedText text, meta text)", []).unwrap();
        let rows = [
            "Hey Miguel, can you send me the deck before the review?",
            "I'm sending you a message with my new tool.",
            "Let's ship Lipflow to Miguel tomorrow.",
            "Can you ask Miguel about the Vizcom demo?",
            "Ping Miguel when the Vizcom build is green.",
            "The Vizcom standup moved to ten.",
        ];
        for (i, r) in rows.iter().chain(rows.iter()).enumerate() {
            con.execute(
                "insert into History values (?,?,?,?,?)",
                rusqlite::params![format!("id-{i}"), "com.tinyspeck.slackmacgap", r.to_lowercase().replace(',', ""), r, "{\"x\": 1}"],
            )
            .unwrap();
        }
    }

    // Ported from lipflow/tests/test_personal.py
    #[test]
    fn import_finds_formatted_text_and_names() {
        let home = temp_home("import");
        fake_wispr(&home.join("Wispr Flow"));
        let (phrases, words) = (home.join("phrases.txt"), home.join("words.txt"));
        let stats = import_wispr(&home.join("Wispr Flow"), &phrases, &words).unwrap();
        assert!(stats.source.as_deref().unwrap().ends_with("History.formattedText"));
        assert_eq!(stats.phrases, 6);
        assert!(stats.new_names.contains(&"Vizcom".to_string()) && stats.new_names.contains(&"Miguel".to_string()));
        assert!(!stats.new_names.contains(&"Lipflow".to_string())); // already listed
        assert!(vocab::load(&words).unwrap().contains(&"Miguel".to_string()));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn personal_rerank_and_similar() {
        let home = temp_home("rerank");
        fake_wispr(&home.join("Wispr Flow"));
        let (phrases, words) = (home.join("phrases.txt"), home.join("words.txt"));
        import_wispr(&home.join("Wispr Flow"), &phrases, &words).unwrap();
        let p = Personal::load(&phrases).unwrap();
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let guesses = s(&["HELLO MIGUEL I AM SENDING YOU A BASIN WITH MY NEW SCHOOL", "HELLO MIGUEL I AM SENDING YOU A MESSAGE WITH MY NEW TOOL"]);
        assert!(p.rerank(&guesses)[0].ends_with("MESSAGE WITH MY NEW TOOL"));
        let lunch = s(&["WHAT IS FOR LUNCH", "WHAT IS FOR BRUNCH"]);
        assert_eq!(p.rerank(&lunch), lunch);
        assert!(p.similar(&guesses[0], 3).contains(&"I'm sending you a message with my new tool.".to_string()));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn missing_folder_and_empty_model() {
        let home = temp_home("missing");
        let err = import_wispr(&home.join("nope"), &home.join("p.txt"), &home.join("w.txt")).unwrap_err();
        assert!(err.to_string().starts_with("No Wispr Flow database found in "));
        let p = Personal::load(&home.join("none.txt")).unwrap();
        assert!(p.is_empty() && p.rerank(&["A B".into(), "A C".into()]) == ["A B", "A C"] && p.similar("a b", 3).is_empty());
        fs::remove_dir_all(&home).unwrap();
    }

    // Expected values computed with lipflow/personal.py on the same phrases.
    #[test]
    fn numbers_match_python() {
        let p = Personal::from_phrases(
            [
                "Hey Miguel, can you send me the deck before the review?",
                "I'm sending you a message with my new tool.",
                "Let's ship Lipflow to Miguel tomorrow.",
                "Can you ask Miguel about the Vizcom demo?",
                "Ping Miguel when the Vizcom build is green.",
                "The Vizcom standup moved to ten.",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        for (text, want) in [
            ("HELLO MIGUEL I AM SENDING YOU A BASIN WITH MY NEW SCHOOL", -4.393978937413562),
            ("HELLO MIGUEL I AM SENDING YOU A MESSAGE WITH MY NEW TOOL", -2.8998193673897696),
            ("WHAT IS FOR LUNCH", -6.674659180182662),
        ] {
            assert!((p.logprob(text) - want).abs() < 1e-12, "{text}");
        }
        assert_eq!(
            p.similar("can you send miguel the vizcom deck", 6),
            ["Hey Miguel, can you send me the deck before the review?", "Can you ask Miguel about the Vizcom demo?"]
        );
        assert_eq!(p.common_words(5), ["miguel", "vizcom", "hey", "send", "deck"]);
    }
}
