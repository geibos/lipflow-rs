//! Joint CTC/attention batch beam search with LM shallow fusion — the algorithm of ESPnet's
//! `BatchBeamSearch` as configured by Lipflow (pre-beam on decoder scores, CTC as a partial
//! scorer, end detection, `maxlen` = number of encoder frames).

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::ctc::{CtcPrefixScorer, CtcState, LOGZERO};
use crate::decoder::{Decoder, KvCache, Lm, Memory};

#[derive(Clone, Debug)]
pub struct BeamConfig {
    pub beam: usize,
    pub ctc_weight: f32,
    pub lm_weight: f32,
    /// Weight of the personal LM, when one is loaded.
    pub plm_weight: f32,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self { beam: 4, ctc_weight: 0.1, lm_weight: 0.3, plm_weight: 0.2 }
    }
}

#[derive(Clone, Debug)]
pub struct Hyp {
    /// Token ids including the leading <sos> and the final <eos>.
    pub yseq: Vec<u32>,
    pub score: f32,
}

pub struct Scorers<'a> {
    pub decoder: &'a Decoder,
    pub memory: &'a Memory,
    pub ctc: &'a CtcPrefixScorer,
    pub lm: Option<&'a Lm>,
    pub plm: Option<&'a Lm>,
}

struct Running {
    yseq: Vec<Vec<u32>>,
    score: Vec<f32>,
    ctc: Vec<CtcState>,
    dec: KvCache,
    lm: KvCache,
    plm: KvCache,
}

/// ESPnet's `end_detect` with M=3, D_end=log(e^-10).
fn end_detect(ended: &[Hyp], i: usize) -> bool {
    let Some(best) = ended.iter().map(|h| h.score).reduce(f32::max) else { return false };
    let mut count = 0;
    for m in 0..3 {
        let Some(len) = i.checked_sub(m) else { continue };
        let same = ended.iter().filter(|h| h.yseq.len() == len).map(|h| h.score).reduce(f32::max);
        if let Some(s) = same
            && s - best < -10.0
        {
            count += 1;
        }
    }
    count == 3
}

fn top_k(xs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    let k = k.min(xs.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| xs[b].total_cmp(&xs[a]));
    idx.truncate(k);
    idx.sort_by(|&a, &b| xs[b].total_cmp(&xs[a]));
    idx
}

pub fn beam_search(s: &Scorers, cfg: &BeamConfig, sos: u32, eos: u32, dev: &Device) -> Result<Vec<Hyp>> {
    let maxlen = s.ctc.frames();
    let pre_beam = (1.5 * cfg.beam as f32) as usize;
    // Python computes 1.0 - 0.1 in double and rounds once; 1f32 - 0.1f32 is one ulp off.
    let dec_w = (1.0 - f64::from(cfg.ctc_weight)) as f32;
    let mut run = Running {
        yseq: vec![vec![sos]],
        score: vec![0.0],
        ctc: vec![s.ctc.initial()],
        dec: KvCache::default(),
        lm: KvCache::default(),
        plm: KvCache::default(),
    };
    let mut ended: Vec<Hyp> = Vec::new();

    for i in 0..maxlen {
        let n = run.yseq.len();
        let last: Vec<u32> = run.yseq.iter().map(|y| *y.last().expect("hyps start with <sos>")).collect();
        // The decoder and the LMs are independent: score them concurrently.
        let (dec, lm, plm) = std::thread::scope(|sc| -> Result<_> {
            let lm_h = s.lm.map(|m| {
                let cache = &mut run.lm;
                let last = &last;
                sc.spawn(move || m.step(last, cache).and_then(|t| Ok(t.to_vec2::<f32>()?)))
            });
            let plm_h = s.plm.map(|m| {
                let cache = &mut run.plm;
                let last = &last;
                sc.spawn(move || m.step(last, cache).and_then(|t| Ok(t.to_vec2::<f32>()?)))
            });
            let dec = s.decoder.step(&last, i, &mut run.dec, s.memory)?.to_vec2::<f32>()?;
            let join = |h: Option<std::thread::ScopedJoinHandle<'_, Result<Vec<Vec<f32>>>>>| -> Result<Option<Vec<Vec<f32>>>> {
                h.map(|h| h.join().map_err(|_| anyhow::anyhow!("LM scoring thread panicked"))?).transpose()
            };
            Ok((dec, join(lm_h)?, join(plm_h)?))
        })?;
        let v = dec[0].len();

        // Weighted scores over the whole (hyp, token) grid, accumulated in ESPnet's order.
        let mut weighted = vec![0f32; n * v];
        let mut cand_states: Vec<(Vec<usize>, Vec<f32>, Vec<Vec<[f32; 2]>>)> = Vec::with_capacity(n);
        for h in 0..n {
            let cands = top_k(&dec[h], pre_beam);
            let cs = s.ctc.score(run.yseq[h].len() - 1, last[h] as usize, &run.ctc[h], &cands);
            let s_prev = run.ctc[h].s;
            let mut ctc_full = vec![LOGZERO; v];
            for (j, &c) in cands.iter().enumerate() {
                ctc_full[c] = cs.log_psi[j];
            }
            ctc_full[eos as usize] = cs.eos;
            ctc_full[0] = LOGZERO;
            let row = &mut weighted[h * v..(h + 1) * v];
            for t in 0..v {
                let mut x = 0f32 + dec_w * dec[h][t];
                if let Some(lm) = &lm {
                    x += cfg.lm_weight * lm[h][t];
                }
                if let Some(plm) = &plm {
                    x += cfg.plm_weight * plm[h][t];
                }
                x += cfg.ctc_weight * (ctc_full[t] - s_prev);
                row[t] = x + run.score[h];
            }
            cand_states.push((cands, ctc_full, cs.r));
        }

        let best = top_k(&weighted, cfg.beam);
        let mut keep_rows: Vec<u32> = Vec::new();
        let mut next = Running {
            yseq: Vec::new(),
            score: Vec::new(),
            ctc: Vec::new(),
            dec: KvCache::default(),
            lm: KvCache::default(),
            plm: KvCache::default(),
        };
        for &flat in &best {
            let (h, tok) = (flat / v, flat % v);
            let mut yseq = run.yseq[h].clone();
            yseq.push(tok as u32);
            if i == maxlen - 1 {
                yseq.push(eos);
            }
            let score = weighted[flat];
            if *yseq.last().expect("non-empty") == eos {
                ended.push(Hyp { yseq, score });
                continue;
            }
            let (cands, ctc_full, rs) = &cand_states[h];
            // A token outside the pre-beam keeps the last candidate's frames, as ESPnet does
            // (its id map yields -1); such hypotheses score ~1e9 lower anyway.
            let j = cands.iter().position(|&c| c == tok).unwrap_or(cands.len() - 1);
            next.ctc.push(CtcState { r: rs[j].clone(), s: ctc_full[tok] });
            next.yseq.push(yseq);
            next.score.push(score);
            keep_rows.push(h as u32);
        }
        if !keep_rows.is_empty() {
            let idx = Tensor::from_slice(&keep_rows, keep_rows.len(), dev)?;
            next.dec = run.dec.select(&idx)?;
            if s.lm.is_some() {
                next.lm = run.lm.select(&idx)?;
            }
            if s.plm.is_some() {
                next.plm = run.plm.select(&idx)?;
            }
        }
        run = next;
        if end_detect(&ended, i) || run.yseq.is_empty() {
            break;
        }
    }
    ended.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(ended)
}
