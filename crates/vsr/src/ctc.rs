//! CTC prefix scoring (Watanabe et al. 2017, Alg. 2), the arithmetic of ESPnet's
//! `CTCPrefixScoreTH` for one utterance, on plain f32 slices.

/// ESPnet's stand-in for log(0); finite so that sums stay ordered the same way.
pub const LOGZERO: f32 = -1e10;

#[inline]
fn lae(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Prefix probabilities of one hypothesis: per frame, (ends in a label, ends in blank).
#[derive(Clone, Debug)]
pub struct CtcState {
    pub r: Vec<[f32; 2]>,
    /// The prefix score log ψ(g) this state was reached with.
    pub s: f32,
}

pub struct CtcScores {
    /// log ψ for each candidate token (same order as the candidates).
    pub log_psi: Vec<f32>,
    /// Next state for each candidate token.
    pub r: Vec<Vec<[f32; 2]>>,
    /// log ψ of ending the sentence here.
    pub eos: f32,
}

pub struct CtcPrefixScorer {
    logp: Vec<f32>, // (T, V) row-major
    t: usize,
    v: usize,
    blank: usize,
    eos: usize,
}

impl CtcPrefixScorer {
    pub fn new(logp: Vec<f32>, t: usize, v: usize, blank: usize, eos: usize) -> Self {
        assert_eq!(logp.len(), t * v, "CTC posteriors must be (T, V)");
        Self { logp, t, v, blank, eos }
    }

    pub fn frames(&self) -> usize {
        self.t
    }

    #[inline]
    fn x(&self, t: usize, c: usize) -> f32 {
        self.logp[t * self.v + c]
    }

    pub fn initial(&self) -> CtcState {
        let mut r = Vec::with_capacity(self.t);
        let mut acc = 0f32;
        for t in 0..self.t {
            acc += self.x(t, self.blank);
            r.push([LOGZERO, acc]);
        }
        CtcState { r, s: 0.0 }
    }

    /// Score extending a prefix of `out_len` labels ending in `last` by each of `cands`.
    pub fn score(&self, out_len: usize, last: usize, st: &CtcState, cands: &[usize]) -> CtcScores {
        let t_len = self.t;
        let r_sum: Vec<f32> = st.r.iter().map(|r| lae(r[0], r[1])).collect();
        let start = out_len.max(1);
        let mut log_psi = Vec::with_capacity(cands.len());
        let mut rs = Vec::with_capacity(cands.len());
        for &c in cands {
            // log φ: probability of the prefix ending exactly before frame t, ready for c.
            let phi = |t: usize| if c == last { st.r[t][1] } else { r_sum[t] };
            let mut r = vec![[LOGZERO, LOGZERO]; t_len];
            if out_len == 0 {
                r[0][0] = self.x(0, c);
            }
            for t in start..t_len {
                let p = r[t - 1];
                r[t][0] = lae(p[0], phi(t - 1)) + self.x(t, c);
                r[t][1] = lae(p[0], p[1]) + self.x(t, self.blank);
            }
            // log ψ = logsumexp(φ(t-1) + x_t(c) for t in start..T, r_{start-1}(c))
            let mut terms: Vec<f32> = (start..t_len)
                .map(|t| (if t == 0 { phi(0) } else { phi(t - 1) }) + self.x(t, c))
                .collect();
            terms.push(r[start - 1][0]);
            let m = terms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut psi = m + terms.iter().map(|&a| (a - m).exp()).sum::<f32>().ln();
            if c == self.eos {
                psi = r_sum[t_len - 1];
            }
            if c == self.blank {
                psi = LOGZERO;
            }
            log_psi.push(psi);
            rs.push(r);
        }
        CtcScores { log_psi, r: rs, eos: r_sum[t_len - 1] }
    }
}
