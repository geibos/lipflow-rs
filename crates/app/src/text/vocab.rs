//! Your own words (`lipflow/vocab.py`): names and terms lip reading can't guess.
//!
//! One per line in `words.txt` ([`crate::paths::words`]). They pick between the model's top
//! guesses, fix their capitalisation, and are given to the LLM.

use std::fs;
use std::io;
use std::path::Path;

use regex::{NoExpand, Regex};

const TEMPLATE: &str = "# Lipflow custom words: one name or term per line, written how you want it typed.
# When one of the model's top guesses contains a word from this list, that guess wins.
Lipflow
";

/// Words from `path` (comments and blank lines skipped); writes the template first if missing.
pub fn load(path: &Path) -> io::Result<Vec<String>> {
    if !path.exists() {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, TEMPLATE)?;
    }
    Ok(fs::read_to_string(path)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect())
}

fn hits(text: &str, words: &[String]) -> usize {
    let t = format!(" {} ", text.to_uppercase());
    words.iter().filter(|w| t.contains(&format!(" {} ", w.to_uppercase()))).count()
}

/// Stable sort: guesses containing more of your words first.
pub fn rerank(candidates: &[String], words: &[String]) -> Vec<String> {
    let mut out = candidates.to_vec();
    if !words.is_empty() {
        out.sort_by_key(|c| std::cmp::Reverse(hits(c, words)));
    }
    out
}

/// Every whole-word, case-insensitive occurrence of a word is rewritten the way you spelled it.
pub fn apply_case(text: &str, words: &[String]) -> String {
    let mut text = text.to_string();
    for w in words {
        if let Ok(re) = Regex::new(&format!(r"(?i)\b{}\b", regex::escape(w))) {
            text = re.replace_all(&text, NoExpand(w)).into_owned();
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn load_creates_template() {
        let dir = std::env::temp_dir().join(format!("lipflow-vocab-{}", std::process::id()));
        let path = dir.join("sub/words.txt");
        assert_eq!(load(&path).unwrap(), s(&["Lipflow"]));
        assert!(path.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rerank_is_stable_and_case_follows_list() {
        let c = s(&["HELLO MCCALL", "HELLO MIGUEL", "HI MIGUEL"]);
        assert_eq!(rerank(&c, &s(&["Miguel"])), s(&["HELLO MIGUEL", "HI MIGUEL", "HELLO MCCALL"]));
        assert_eq!(rerank(&c, &[]), c);
        assert_eq!(apply_case("hello miguel, MIGUELS", &s(&["Miguel"])), "hello Miguel, MIGUELS");
    }
}
