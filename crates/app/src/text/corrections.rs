//! Learn from your corrections (`lipflow/corrections.py`, text part): find what the user changed
//! in the words Lipflow typed. Clip storage lives elsewhere.
//!
//! The dictated span is found by diffing the field against its contents before the paste, then
//! aligned word-by-word with what was pasted, so text typed afterwards isn't mistaken for a
//! correction. Small edits count; rewrites don't.

use std::collections::HashMap;
use std::hash::Hash;

use super::words_of;

/// What now stands where the paste went: `after` minus `before`'s common prefix and suffix.
pub fn inserted_span(before: &str, after: &str) -> String {
    let (b, a): (Vec<char>, Vec<char>) = (before.chars().collect(), after.chars().collect());
    let n = b.len().min(a.len());
    let mut p = 0;
    while p < n && b[p] == a[p] {
        p += 1;
    }
    let mut s = 0;
    while s < n - p && b[b.len() - 1 - s] == a[a.len() - 1 - s] {
        s += 1;
    }
    a[p..a.len() - s].iter().collect()
}

/// `difflib` opcode tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Replace,
    Delete,
    Insert,
    Equal,
}

/// `(tag, i1, i2, j1, j2)`: `a[i1..i2]` becomes `b[j1..j2]`.
pub type Opcode = (Tag, usize, usize, usize, usize);

/// Port of Python's `difflib.SequenceMatcher(a=a, b=b, autojunk=False)` without an `isjunk`
/// function: no element of `b` is junk or popular, so the junk-extension passes of
/// `find_longest_match` are no-ops and are omitted.
pub struct SequenceMatcher<'s, T> {
    a: &'s [T],
    b: &'s [T],
    b2j: HashMap<T, Vec<usize>>,
}

impl<'s, T: Eq + Hash + Clone> SequenceMatcher<'s, T> {
    pub fn new(a: &'s [T], b: &'s [T]) -> Self {
        let mut b2j: HashMap<T, Vec<usize>> = HashMap::new();
        for (j, x) in b.iter().enumerate() {
            b2j.entry(x.clone()).or_default().push(j);
        }
        Self { a, b, b2j }
    }

    /// Longest matching block in `a[alo..ahi]` / `b[blo..bhi]`: `(i, j, size)`, earliest in `a`,
    /// then earliest in `b`, among the maximal ones.
    pub fn find_longest_match(&self, alo: usize, ahi: usize, blo: usize, bhi: usize) -> (usize, usize, usize) {
        let (a, b) = (self.a, self.b);
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for (i, x) in a.iter().enumerate().take(ahi).skip(alo) {
            let mut newj2len = HashMap::new();
            for &j in self.b2j.get(x).map_or(&[][..], Vec::as_slice) {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j.checked_sub(1).and_then(|p| j2len.get(&p)).copied().unwrap_or(0) + 1;
                newj2len.insert(j, k);
                if k > bestsize {
                    (besti, bestj, bestsize) = (i + 1 - k, j + 1 - k, k);
                }
            }
            j2len = newj2len;
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            (besti, bestj, bestsize) = (besti - 1, bestj - 1, bestsize + 1);
        }
        while besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize] {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    }

    /// Non-adjacent matching blocks `(i, j, n)`, ascending, ending with the `(len a, len b, 0)` sentinel.
    pub fn get_matching_blocks(&self) -> Vec<(usize, usize, usize)> {
        let (la, lb) = (self.a.len(), self.b.len());
        let mut queue = vec![(0, la, 0, lb)];
        let mut blocks = Vec::new();
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let (i, j, k) = self.find_longest_match(alo, ahi, blo, bhi);
            if k > 0 {
                blocks.push((i, j, k));
                if alo < i && blo < j {
                    queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
        blocks.sort_unstable();
        let (mut i1, mut j1, mut k1) = (0, 0, 0);
        let mut out = Vec::new();
        for (i2, j2, k2) in blocks {
            if i1 + k1 == i2 && j1 + k1 == j2 {
                k1 += k2;
            } else {
                if k1 > 0 {
                    out.push((i1, j1, k1));
                }
                (i1, j1, k1) = (i2, j2, k2);
            }
        }
        if k1 > 0 {
            out.push((i1, j1, k1));
        }
        out.push((la, lb, 0));
        out
    }

    /// How to turn `a` into `b`.
    pub fn get_opcodes(&self) -> Vec<Opcode> {
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::new();
        for (ai, bj, size) in self.get_matching_blocks() {
            let tag = if i < ai && j < bj {
                Some(Tag::Replace)
            } else if i < ai {
                Some(Tag::Delete)
            } else if j < bj {
                Some(Tag::Insert)
            } else {
                None
            };
            if let Some(tag) = tag {
                out.push((tag, i, ai, j, bj));
            }
            (i, j) = (ai + size, bj + size);
            if size > 0 {
                out.push((Tag::Equal, ai, i, bj, j));
            }
        }
        out
    }

    /// `2 * matches / (len a + len b)`, 1.0 for two empty sequences.
    pub fn ratio(&self) -> f64 {
        let matches: usize = self.get_matching_blocks().iter().map(|m| m.2).sum();
        let length = self.a.len() + self.b.len();
        if length == 0 { 1.0 } else { 2.0 * matches as f64 / length as f64 }
    }
}

/// The corrected version of `pasted` inside `span`, or `None` if it wasn't corrected (or was
/// rewritten beyond recognition). Words typed before/after the dictation are trimmed off.
pub fn find_correction(pasted: &str, span: &str) -> Option<String> {
    let pw = words_of(pasted);
    let sw_raw: Vec<&str> = span.split_whitespace().collect();
    let sw: Vec<String> = sw_raw.iter().map(|w| words_of(w).join(" ")).collect();
    if pw.is_empty() || sw.is_empty() {
        return None;
    }
    let mut ops = SequenceMatcher::new(&pw, &sw).get_opcodes();
    // trim pure insertions at either end: that's new typing, not the dictation
    while ops.first().is_some_and(|o| o.0 == Tag::Insert) {
        ops.remove(0);
    }
    while ops.last().is_some_and(|o| o.0 == Tag::Insert) {
        ops.pop();
    }
    // a replace at either end can swallow new typing next to it ("school" → "tool. Also can you…"):
    // keep only as many span words as dictated words it replaces, plus one
    let last = ops.last_mut()?;
    if let (Tag::Replace, i1, i2, j1, j2) = *last
        && j2 - j1 > i2 - i1 + 1
    {
        *last = (Tag::Replace, i1, i2, j1, j1 + (i2 - i1));
    }
    let first = ops.first_mut()?;
    if let (Tag::Replace, i1, i2, j1, j2) = *first
        && j2 - j1 > i2 - i1 + 1
    {
        *first = (Tag::Replace, i1, i2, j2 - (i2 - i1), j2);
    }
    // opcodes are ordered in b and the trims never cross, so lo <= hi
    let (lo, hi) = (ops[0].3, ops[ops.len() - 1].4);
    let fixed_raw = &sw_raw[lo..hi];
    let fixed: Vec<String> = sw[lo..hi].iter().filter(|w| !w.is_empty()).cloned().collect();
    if fixed == pw {
        return None; // untouched
    }
    let sm = SequenceMatcher::new(&pw, &fixed);
    let changed: usize = sm.get_opcodes().iter().filter(|o| o.0 != Tag::Equal).map(|&(_, i1, i2, j1, j2)| (i2 - i1).max(j2 - j1)).sum();
    if sm.ratio() < 0.5 || changed as f64 > 2f64.max(0.4 * pw.len() as f64) {
        return None; // a rewrite, not a correction
    }
    Some(fixed_raw.join(" ").trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: &str = "Hello Miguel, I am sending you a message with my new school.";

    fn fix(before: &str, after: &str, pasted: &str) -> Option<String> {
        find_correction(pasted, &inserted_span(before, after))
    }

    // Ported from lipflow/tests/test_corrections.py
    #[test]
    fn small_fix_is_learned() {
        let b = "Chat so far. ";
        assert_eq!(fix(b, &format!("{b}{P}"), P), None); // untouched
        assert_eq!(
            fix(b, &format!("{b}{}", P.replace("school", "tool")), P).as_deref(),
            Some("Hello Miguel, I am sending you a message with my new tool.")
        );
    }

    #[test]
    fn typing_more_after_is_not_part_of_the_label() {
        let after = P.replace("school", "tool") + " Also can you review it by Friday?";
        assert_eq!(fix("", &after, P).as_deref(), Some("Hello Miguel, I am sending you a message with my new tool."));
    }

    #[test]
    fn text_after_the_cursor_is_ignored() {
        let b = "Before. After text";
        let after = format!("Before. {} After text", P.replace("school", "tool"));
        assert_eq!(fix(b, &after, P).as_deref(), Some("Hello Miguel, I am sending you a message with my new tool."));
    }

    #[test]
    fn rewrite_or_deletion_is_not_a_correction() {
        assert_eq!(fix("", "Totally different words that I wrote instead of it", P), None);
        assert_eq!(fix("", "", P), None);
    }

    // Cross-checked against CPython 3 difflib.
    #[test]
    fn sequence_matcher_matches_difflib() {
        let a: Vec<char> = "qabxcd".chars().collect();
        let b: Vec<char> = "abycdf".chars().collect();
        let sm = SequenceMatcher::new(&a, &b);
        assert_eq!(
            sm.get_opcodes(),
            vec![
                (Tag::Delete, 0, 1, 0, 0),
                (Tag::Equal, 1, 3, 0, 2),
                (Tag::Replace, 3, 4, 2, 3),
                (Tag::Equal, 4, 6, 3, 5),
                (Tag::Insert, 6, 6, 5, 6)
            ]
        );
        assert!((sm.ratio() - 2.0 * 4.0 / 12.0).abs() < 1e-12);
        let (a, b): (Vec<char>, Vec<char>) = (" abcd".chars().collect(), "abcd abcd".chars().collect());
        assert_eq!(SequenceMatcher::new(&a, &b).find_longest_match(0, 5, 0, 9), (0, 4, 5));
    }
}
