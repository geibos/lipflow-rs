//! Auto-AVSR mouth crops (`lipflow/face.py: mouth_rois`): temporally smoothed eye/nose/mouth
//! anchors -> similarity transform onto the training mean face -> 96x96 patch at the mouth.

use crate::geom::{Affine, warp_affine_gray};

/// Mean-face anchors (right eye, left eye, nose base, mouth centre) in the 256x256 frame.
pub const STABLE_REFERENCE: [[f64; 2]; 4] = [
    [102.0739430570659, 94.27230352389712],
    [156.3613054162131, 93.57815605186966],
    [129.00373787023733, 135.90343028603127],
    [129.31337322587925, 157.82299634918854],
];

const RIGHT_EYE: [usize; 16] = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246];
const LEFT_EYE: [usize; 16] = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466];
const NOSE_BASE: [usize; 5] = [97, 98, 2, 326, 327];
const LIPS: [usize; 40] = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318,
    321, 324, 375, 402, 405, 409, 415,
];
pub const OUTER_LIPS: [usize; 20] = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
pub const INNER_LIPS: [usize; 20] = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

pub type Anchors = [[f32; 2]; 4];

fn mean_of(points: &[[f32; 2]], idx: &[usize]) -> [f32; 2] {
    let (mut x, mut y) = (0f32, 0f32);
    for &i in idx {
        x += points[i][0];
        y += points[i][1];
    }
    [x / idx.len() as f32, y / idx.len() as f32]
}

/// 4x2 anchors from the 478 landmarks (pixels).
pub fn anchors(points: &[[f32; 2]]) -> Anchors {
    [mean_of(points, &RIGHT_EYE), mean_of(points, &LEFT_EYE), mean_of(points, &NOSE_BASE), mean_of(points, &LIPS)]
}

/// Inner-lip gap over mouth width: a cheap "is the mouth moving" signal.
pub fn mouth_open(points: &[[f32; 2]]) -> f32 {
    let d = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    d(points[13], points[14]) / (d(points[61], points[291]) + 1e-6)
}

/// Fill gaps linearly and hold the ends. None if there is no face at all.
pub fn interpolate(anchors: &[Option<Anchors>]) -> Option<Vec<Anchors>> {
    let valid: Vec<usize> = (0..anchors.len()).filter(|&i| anchors[i].is_some()).collect();
    let (&first, &last) = (valid.first()?, valid.last()?);
    let mut out: Vec<Anchors> = anchors.iter().map(|a| a.unwrap_or_default()).collect();
    for w in valid.windows(2) {
        let (a, b) = (w[0], w[1]);
        for k in 1..b - a {
            let t = k as f64 / (b - a) as f64;
            for p in 0..4 {
                for c in 0..2 {
                    out[a + k][p][c] = out[a][p][c] + (out[b][p][c] - out[a][p][c]) * t as f32;
                }
            }
        }
    }
    for i in 0..first {
        out[i] = out[first];
    }
    for i in last + 1..out.len() {
        out[i] = out[last];
    }
    Some(out)
}

/// OpenCV's `cv::RNG` (multiply-with-carry), seeded as LMeDS seeds it.
struct CvRng(u64);

impl CvRng {
    fn next(&mut self) -> u32 {
        self.0 = u64::from(self.0 as u32) * 4_164_903_690 + (self.0 >> 32);
        self.0 as u32
    }

    fn uniform(&mut self, a: u32, b: u32) -> u32 {
        if a == b { a } else { self.next() % (b - a) + a }
    }
}

/// AffinePartial2DEstimatorCallback::runKernel: the similarity through two point pairs.
fn similarity_2(p: [[f64; 2]; 2], q: [[f64; 2]; 2]) -> [f64; 4] {
    let ([x1, y1], [x2, y2]) = (p[0], p[1]);
    let ([bx1, by1], [bx2, by2]) = (q[0], q[1]);
    let d = 1.0 / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    let s0 = d * ((bx1 - bx2) * (x1 - x2) + (by1 - by2) * (y1 - y2));
    let s1 = d * ((by1 - by2) * (x1 - x2) - (bx1 - bx2) * (y1 - y2));
    let s2 = d * ((by1 - by2) * (x1 * y2 - x2 * y1) - (bx1 * y2 - bx2 * y1) * (y1 - y2) - (bx1 * x2 - bx2 * x1) * (x1 - x2));
    let s3 = d * (-(bx1 - bx2) * (x1 * y2 - x2 * y1) - (by1 * x2 - by2 * x1) * (x1 - x2) - (by1 * y2 - by2 * y1) * (y1 - y2));
    [s0, s1, s2, s3]
}

/// Least-squares similarity over point pairs (what OpenCV's Levenberg-Marquardt refinement
/// converges to: the problem is linear in a, b, tx, ty).
fn similarity_ls(p: &[[f64; 2]], q: &[[f64; 2]]) -> Option<[f64; 4]> {
    let n = p.len() as f64;
    let (px, py) = (p.iter().map(|v| v[0]).sum::<f64>() / n, p.iter().map(|v| v[1]).sum::<f64>() / n);
    let (qx, qy) = (q.iter().map(|v| v[0]).sum::<f64>() / n, q.iter().map(|v| v[1]).sum::<f64>() / n);
    let (mut sxx, mut num_a, mut num_b) = (0.0, 0.0, 0.0);
    for (a, b) in p.iter().zip(q) {
        let (ux, uy, vx, vy) = (a[0] - px, a[1] - py, b[0] - qx, b[1] - qy);
        sxx += ux * ux + uy * uy;
        num_a += ux * vx + uy * vy;
        num_b += ux * vy - uy * vx;
    }
    if sxx < 1e-12 {
        return None;
    }
    let (a, b) = (num_a / sxx, num_b / sxx);
    Some([a, b, qx - (a * px - b * py), qy - (b * px + a * py)])
}

/// Affine2DEstimatorCallback::computeError: squared residuals in float.
fn errors(m: &[f64; 4], p: &[[f32; 2]], q: &[[f32; 2]]) -> Vec<f32> {
    let h = [m[0] as f32, -m[1] as f32, m[2] as f32, m[1] as f32, m[0] as f32, m[3] as f32];
    p.iter()
        .zip(q)
        .map(|(f, t)| {
            let a = h[0] * f[0] + h[1] * f[1] + h[2] - t[0];
            let b = h[3] * f[0] + h[4] * f[1] + h[5] - t[1];
            a * a + b * b
        })
        .collect()
}

/// cv2.estimateAffinePartial2D(src, dst, method=LMEDS), reproduced step by step: 13 random
/// 2-point subsets from OpenCV's RNG, the model with the smallest (upper) median error, inliers
/// within the LMedS sigma, then the least-squares refinement on the inliers.
pub fn estimate_similarity(src: &Anchors, dst: &[[f64; 2]; 4]) -> Option<Affine> {
    const MODEL_POINTS: usize = 2;
    let p: Vec<[f32; 2]> = src.to_vec();
    let q: Vec<[f32; 2]> = dst.iter().map(|v| [v[0] as f32, v[1] as f32]).collect();
    let n = p.len();
    let wide = |v: [f32; 2]| [f64::from(v[0]), f64::from(v[1])];
    // RANSACUpdateNumIters(0.99, 0.45, 2, 2000)
    let niters = ((1.0f64 - 0.99).ln() / (1.0 - (1.0f64 - 0.45).powi(2)).ln()).round_ties_even() as usize;
    let mut rng = CvRng(u64::MAX);
    let mut best: Option<([f64; 4], f32)> = None;
    for _ in 0..niters.max(3) {
        let mut idx = [0usize; MODEL_POINTS];
        for i in 0..MODEL_POINTS {
            loop {
                idx[i] = rng.uniform(0, n as u32) as usize;
                if !idx[..i].contains(&idx[i]) {
                    break;
                }
            }
        }
        let m = similarity_2([wide(p[idx[0]]), wide(p[idx[1]])], [wide(q[idx[0]]), wide(q[idx[1]])]);
        let mut e = errors(&m, &p, &q);
        e.sort_by(f32::total_cmp);
        let median = e[n / 2];
        if best.is_none_or(|(_, bm)| median < bm) {
            best = Some((m, median));
        }
    }
    let (m, median) = best?;
    let sigma = (2.5 * 1.4826 * (1.0 + 5.0 / (n - MODEL_POINTS) as f64) * f64::from(median).sqrt()).max(0.001);
    let thresh = (sigma * sigma) as f32;
    let err = errors(&m, &p, &q);
    let inl: Vec<usize> = (0..n).filter(|&k| err[k] <= thresh).collect();
    if inl.len() < MODEL_POINTS {
        return None;
    }
    let ip: Vec<[f64; 2]> = inl.iter().map(|&k| wide(p[k])).collect();
    let iq: Vec<[f64; 2]> = inl.iter().map(|&k| wide(q[k])).collect();
    let m = similarity_ls(&ip, &iq).unwrap_or(m);
    Some(Affine([m[0], -m[1], m[2], m[1], m[0], m[3]]))
}

/// One grayscale frame: either the full image or a crop at `offset` (x0, y0) of the full image.
pub struct GrayFrame<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub offset: (usize, usize),
}

pub const CROP: usize = 96;
const WINDOW_MARGIN: usize = 12;

/// (T*96*96) mouth crops, or None when no frame has a face.
pub fn mouth_rois(frames: &[GrayFrame], anchors: &[Option<Anchors>]) -> Option<Vec<u8>> {
    let lms = interpolate(anchors)?;
    let n = lms.len();
    let half = CROP / 2;
    let mut out = Vec::with_capacity(frames.len() * CROP * CROP);
    for (i, f) in frames.iter().enumerate() {
        let m = (WINDOW_MARGIN / 2).min(i).min(n - 1 - i);
        let win = &lms[i - m..=i + m];
        let mut sm = [[0f32; 2]; 4];
        for a in win {
            for p in 0..4 {
                sm[p][0] += a[p][0];
                sm[p][1] += a[p][1];
            }
        }
        let k = win.len() as f32;
        sm.iter_mut().for_each(|p| {
            p[0] /= k;
            p[1] /= k;
        });
        // Keep the window's shape but the current frame's centre.
        let centre = |a: &Anchors| {
            let (mut x, mut y) = (0f32, 0f32);
            a.iter().for_each(|p| {
                x += p[0];
                y += p[1];
            });
            [x / 4.0, y / 4.0]
        };
        let (cur, smc) = (centre(&lms[i]), centre(&sm));
        sm.iter_mut().for_each(|p| {
            p[0] += cur[0] - smc[0];
            p[1] += cur[1] - smc[1];
        });
        let tf = estimate_similarity(&sm, &STABLE_REFERENCE).or_else(|| estimate_similarity(&lms[i], &STABLE_REFERENCE))?;
        let (mx, my) = tf.apply(f64::from(sm[3][0]), f64::from(sm[3][1]));
        let mut t = tf;
        if f.offset != (0, 0) {
            let (ox, oy) = (f.offset.0 as f64, f.offset.1 as f64);
            t.0[2] += t.0[0] * ox + t.0[1] * oy;
            t.0[5] += t.0[3] * ox + t.0[4] * oy;
        }
        let warped = warp_affine_gray(f.data, f.width, f.height, &t, 256, 256);
        let cx = mx.clamp(half as f64, (256 - half) as f64).round_ties_even() as usize;
        let cy = my.clamp(half as f64, (256 - half) as f64).round_ties_even() as usize;
        for y in cy - half..cy + half {
            out.extend_from_slice(&warped[y * 256 + cx - half..][..CROP]);
        }
    }
    Some(out)
}
