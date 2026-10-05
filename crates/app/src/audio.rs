//! Whisper-mode audio (`lipflow/mic.py: segment`): microphone chunks timestamped on the video
//! clock, cut to exactly the recorded frames and resampled to 16 kHz. Timing matters: on test
//! clips, audio 0.4 s off the video doubled the word error rate.

pub const RATE: usize = 16_000;
/// A frame is read ~2 frames after it was exposed.
pub const CAMERA_LATENCY: f64 = 0.06;

/// Captured chunks: (wall-clock time of the first sample, samples at `rate`).
pub struct Chunks {
    pub rate: f64,
    pub chunks: Vec<(f64, Vec<f32>)>,
}

/// Band-limited resampling (windowed sinc, 32 taps).
pub fn resample(x: &[f32], from: f64, to: f64) -> Vec<f32> {
    if (from - to).abs() < 1e-6 {
        return x.to_vec();
    }
    let ratio = from / to;
    let cutoff = (to / from).min(1.0) * 0.92; // of the input Nyquist
    let n_out = (x.len() as f64 / ratio).floor() as usize;
    const HALF: isize = 16;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let p = i as f64 * ratio;
        let c = p.floor() as isize;
        let mut acc = 0f64;
        let mut wsum = 0f64;
        for k in (c - HALF + 1)..=(c + HALF) {
            if k < 0 || k as usize >= x.len() {
                continue;
            }
            let d = p - k as f64;
            let arg = std::f64::consts::PI * d * cutoff;
            let sinc = if arg.abs() < 1e-9 { 1.0 } else { arg.sin() / arg };
            let win = 0.5 + 0.5 * (std::f64::consts::PI * d / HALF as f64).cos(); // Hann
            let w = sinc * win;
            acc += f64::from(x[k as usize]) * w;
            wsum += w;
        }
        out.push(if wsum.abs() > 1e-9 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}

/// 16 kHz audio for `n_frames` video frames starting at `t_start` (video clock), or None when
/// less than half of it was captured.
pub fn segment(c: &Chunks, t_start: f64, n_frames: usize, fps: usize) -> Option<Vec<f32>> {
    if c.chunks.is_empty() {
        return None;
    }
    let t_start = t_start - CAMERA_LATENCY;
    let want_native = (n_frames as f64 * c.rate / fps as f64).round() as usize;
    let mut native = vec![0f32; want_native];
    let mut got = 0usize;
    for (t0, a) in &c.chunks {
        let off = ((t0 - t_start) * c.rate).round() as isize;
        let lo = off.max(0) as usize;
        let hi = (off + a.len() as isize).min(want_native as isize);
        if hi > lo as isize {
            let hi = hi as usize;
            native[lo..hi].copy_from_slice(&a[(lo as isize - off) as usize..(hi as isize - off) as usize]);
            got += hi - lo;
        }
    }
    if got * 2 <= want_native {
        return None;
    }
    let mut out = resample(&native, c.rate, RATE as f64);
    out.resize(n_frames * RATE / fps, 0.0);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from tests/test_pipeline.py::test_mic_segment_aligns_to_video_clock
    #[test]
    fn segment_aligns_to_video_clock() {
        let rate = RATE as f64;
        let wave: Vec<f32> = (0..RATE * 2).map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / rate).sin() as f32).collect();
        let start = 1000.0;
        let chunks = Chunks { rate, chunks: (0..wave.len()).step_by(1600).map(|i| (start + i as f64 / rate, wave[i..(i + 1600).min(wave.len())].to_vec())).collect() };
        let seg = segment(&chunks, start + 0.5 + CAMERA_LATENCY, 25, 25).expect("enough audio");
        assert_eq!(seg.len(), 16000);
        assert!(seg.iter().zip(&wave[8000..24000]).all(|(a, b)| (a - b).abs() <= 1e-6));
    }

    #[test]
    fn resample_keeps_a_low_tone() {
        let from = 48_000.0;
        let x: Vec<f32> = (0..48_000).map(|i| (2.0 * std::f64::consts::PI * 300.0 * i as f64 / from).sin() as f32).collect();
        let y = resample(&x, from, 16_000.0);
        assert_eq!(y.len(), 16_000);
        for i in (100..15_900).step_by(997) {
            let want = (2.0 * std::f64::consts::PI * 300.0 * i as f64 / 16_000.0).sin() as f32;
            assert!((y[i] - want).abs() < 0.02, "{i}: {} vs {want}", y[i]);
        }
    }
}
