//! Offline text rules (`lipflow/cleanup.py`): numbers to digits, sentence case, end punctuation.

use std::sync::LazyLock;

use regex::Regex;

const UNITS: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [(&str, u32); 8] = [("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50), ("sixty", 60), ("seventy", 70), ("eighty", 80), ("ninety", 90)];

fn num(w: &str) -> Option<u32> {
    UNITS.iter().position(|&u| u == w).map(|i| i as u32).or_else(|| TENS.iter().find(|t| t.0 == w).map(|t| t.1))
}

fn is_tens(v: u32) -> bool {
    TENS.iter().any(|t| t.1 == v)
}

/// "fifty two" / "twelve" / "forty" from the front: (value, words used).
fn two_digit(words: &[&str]) -> Option<(u32, usize)> {
    let v = num(words.first()?)?;
    if is_tens(v)
        && let Some(n) = words.get(1).and_then(|w| num(w))
        && (1..10).contains(&n)
    {
        return Some((v + n, 2));
    }
    Some((v, 1))
}

/// "nineteen forty three" -> 1943, "eleven films" -> 11 films. Leaves "one" / "two" alone.
pub fn numbers_to_digits(t: &str) -> String {
    let words: Vec<&str> = t.split_whitespace().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let a = two_digit(&words[i..]);
        if let Some((av, 1)) = a
            && (10..=20).contains(&av)
            && let Some((bv, bn)) = two_digit(&words[i + 1..])
            && bv >= 10
        {
            out.push((av * 100 + bv).to_string());
            i += 1 + bn;
            continue;
        }
        if let Some((av, an)) = a
            && av >= 10
        {
            out.push(av.to_string());
            i += an;
            continue;
        }
        out.push(words[i].to_string());
        i += 1;
    }
    out.join(" ")
}

static LONE_I: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bi\b").expect("static regex"));
static I_CONTRACTION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bi'(m|ll|ve|d)\b").expect("static regex"));
static QUESTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(who|what|when|where|why|how|is|are|can|could|would|should|do|does|did|will)\b").expect("static regex"));

fn capital_i(t: &str) -> String {
    let t = LONE_I.replace_all(t, "I");
    I_CONTRACTION.replace_all(&t, "I'$1").into_owned()
}

fn upper_first(t: &str) -> String {
    let mut c = t.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// Sentence-case start and a capital I, whatever the model returned.
pub fn fix_case(text: &str) -> String {
    let t = text.trim();
    if t.is_empty() {
        return String::new();
    }
    upper_first(&capital_i(t))
}

/// Offline cleanup: sentence case, "I", end punctuation, numbers.
pub fn basic_cleanup(text: &str) -> String {
    let t = numbers_to_digits(&text.trim().to_lowercase());
    if t.is_empty() {
        return String::new();
    }
    let mut t = upper_first(&capital_i(&t));
    if !t.ends_with(['.', '?', '!']) {
        t.push(if QUESTION.is_match(&t) { '?' } else { '.' });
    }
    t
}

static QUESTION_RU: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(кто|что|когда|где|куда|откуда|почему|зачем|как|какой|какая|какое|какие|сколько|чей|можно|можешь|можете)\b|\bли\b").expect("static regex")
});

/// Offline cleanup of a Russian guess: sentence case and a full stop or question mark.
pub fn basic_cleanup_ru(text: &str) -> String {
    let t = text.trim().to_lowercase();
    if t.is_empty() {
        return String::new();
    }
    let mut t = upper_first(&t);
    if !t.ends_with(['.', '?', '!']) {
        t.push(if QUESTION_RU.is_match(&t) { '?' } else { '.' });
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn russian_sentence_case_and_questions() {
        assert_eq!(basic_cleanup_ru("как дела"), "Как дела?");
        assert_eq!(basic_cleanup_ru("я отправил тебе файл"), "Я отправил тебе файл.");
        assert_eq!(basic_cleanup_ru("придёшь ли ты завтра"), "Придёшь ли ты завтра?");
    }

    // Ported from lipflow/tests/test_cleanup.py
    #[test]
    fn years_and_numbers() {
        assert_eq!(numbers_to_digits("married in nineteen fifty two"), "married in 1952");
        assert_eq!(numbers_to_digits("it is twenty twenty six"), "it is 2026");
        assert_eq!(numbers_to_digits("appeared in eleven films"), "appeared in 11 films");
        assert_eq!(numbers_to_digits("one dog and two cats"), "one dog and two cats");
        assert_eq!(numbers_to_digits("forty two"), "42");
    }

    #[test]
    fn basic() {
        assert_eq!(basic_cleanup("I THINK I'LL GO"), "I think I'll go.");
        assert_eq!(basic_cleanup("WHAT TIME IS IT"), "What time is it?");
        assert_eq!(basic_cleanup(""), "");
        assert_eq!(fix_case("hello miguel i'm here"), "Hello miguel I'm here");
    }
}
