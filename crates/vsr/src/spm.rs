//! SentencePiece unigram tokenizer (enough of it for the model's `unigram5000.model`): reads the
//! ModelProto and segments text with the Viterbi best path, like `sp.encode(text, out_type=str)`.
//! Input is expected in the model's own alphabet (upper-case letters, apostrophes, spaces).

use std::collections::HashMap;

use anyhow::{Result, bail};

const SPACE: char = '\u{2581}';
/// Unknown pieces score this much below the worst piece (sentencepiece kUnkPenalty).
const UNK_PENALTY: f32 = 10.0;

pub struct SentencePiece {
    /// Matchable pieces (NORMAL / USER_DEFINED) → score.
    pieces: HashMap<String, f32>,
    max_len: usize,
    unk: String,
    unk_score: f32,
    add_dummy_prefix: bool,
}

fn varint(b: &[u8], p: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let Some(&byte) = b.get(*p) else { bail!("truncated protobuf") };
        *p += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
    }
    bail!("varint too long")
}

/// (field, wire type, payload) of each top-level field of a protobuf message.
fn fields(b: &[u8]) -> Result<Vec<(u64, u8, &[u8], u64)>> {
    let mut out = Vec::new();
    let mut p = 0;
    while p < b.len() {
        let key = varint(b, &mut p)?;
        let (field, wt) = (key >> 3, (key & 7) as u8);
        match wt {
            0 => {
                let v = varint(b, &mut p)?;
                out.push((field, wt, &b[0..0], v));
            }
            1 => {
                out.push((field, wt, b.get(p..p + 8).unwrap_or(&[]), 0));
                p += 8;
            }
            2 => {
                let n = varint(b, &mut p)? as usize;
                let Some(s) = b.get(p..p + n) else { bail!("truncated protobuf field") };
                out.push((field, wt, s, 0));
                p += n;
            }
            5 => {
                out.push((field, wt, b.get(p..p + 4).unwrap_or(&[]), 0));
                p += 4;
            }
            w => bail!("unsupported protobuf wire type {w}"),
        }
    }
    Ok(out)
}

impl SentencePiece {
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let mut pieces = HashMap::new();
        let (mut unk, mut min_score, mut add_dummy_prefix) = (String::from("<unk>"), f32::MAX, true);
        for (field, wt, payload, _) in fields(b)? {
            match (field, wt) {
                (1, 2) => {
                    let (mut piece, mut score, mut ty) = (String::new(), 0f32, 1u64);
                    for (f, w, pl, v) in fields(payload)? {
                        match (f, w) {
                            (1, 2) => piece = String::from_utf8_lossy(pl).into_owned(),
                            (2, 5) if pl.len() == 4 => score = f32::from_le_bytes([pl[0], pl[1], pl[2], pl[3]]),
                            (3, 0) => ty = v,
                            _ => {}
                        }
                    }
                    match ty {
                        1 | 4 => {
                            min_score = min_score.min(score);
                            pieces.insert(piece, score);
                        }
                        2 => unk = piece,
                        _ => {}
                    }
                }
                (3, 2) => {
                    for (f, w, _, v) in fields(payload)? {
                        if (f, w) == (3, 0) {
                            add_dummy_prefix = v != 0;
                        }
                    }
                }
                _ => {}
            }
        }
        if pieces.is_empty() {
            bail!("no pieces in the SentencePiece model");
        }
        let max_len = pieces.keys().map(|p| p.chars().count()).max().unwrap_or(1);
        Ok(Self { pieces, max_len, unk, unk_score: min_score - UNK_PENALTY, add_dummy_prefix })
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Pieces of `text` (whitespace collapsed, "▁" for spaces, a leading "▁").
    pub fn encode(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }
        let mut s = String::new();
        if self.add_dummy_prefix {
            s.push(SPACE);
        }
        s.push_str(&words.join(&SPACE.to_string()));
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        // best[i]: (score, start of the last piece) of the best segmentation of chars[..i]
        let mut best: Vec<(f32, usize, bool)> = vec![(f32::NEG_INFINITY, 0, false); n + 1];
        best[0].0 = 0.0;
        for i in 0..n {
            if best[i].0 == f32::NEG_INFINITY {
                continue;
            }
            let mut any = false;
            let mut piece = String::new();
            for j in i..(i + self.max_len).min(n) {
                piece.push(chars[j]);
                if let Some(&sc) = self.pieces.get(&piece) {
                    any |= j == i;
                    let cand = best[i].0 + sc;
                    if cand > best[j + 1].0 {
                        best[j + 1] = (cand, i, false);
                    }
                }
            }
            if !any {
                // a character no piece covers becomes <unk>
                let cand = best[i].0 + self.unk_score;
                if cand > best[i + 1].0 {
                    best[i + 1] = (cand, i, true);
                }
            }
        }
        let mut out = Vec::new();
        let mut end = n;
        while end > 0 {
            let (_, start, unk) = best[end];
            out.push(if unk { self.unk.clone() } else { chars[start..end].iter().collect() });
            end = start;
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_sentencepiece() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model = root.join("../../lipflow/models/lm/unigram5000.model");
        if !model.exists() {
            return eprintln!("skipping: no model");
        }
        let sp = SentencePiece::load(&model).expect("model");
        let refs: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("tests/fixtures/spm_ref.json")).expect("fixture")).expect("json");
        let mut n = 0;
        for r in refs.as_array().expect("array") {
            let text = r["text"].as_str().expect("text");
            let want: Vec<String> = r["pieces"].as_array().expect("pieces").iter().map(|p| p.as_str().unwrap_or_default().to_string()).collect();
            assert_eq!(sp.encode(text), want, "{text}");
            n += 1;
        }
        assert!(n > 100);
    }
}
