//! Face crops for MultiVSR, made the way its training data was: a syncnet-style crop around an
//! S3FD face box (square of 2.8·s, s = half the box's longer side, top edge s above the centre),
//! resized to 224, then to 160, then the centre 96×96, all with OpenCV's bilinear sampling.
//!
//! We track MediaPipe landmarks, not S3FD boxes: `FaceBox::from_points` maps the landmarks' box
//! onto the S3FD box with a fit made on the same frames (tools/face_box_fit.py).

use crate::geom::{Affine, Border, Frame, crop_rgb};

/// An S3FD-style face box: centre and half of its longer side, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBox {
    pub cx: f32,
    pub cy: f32,
    pub s: f32,
}

/// S3FD box from the landmarks' box: centre offset and size ratio, both relative to the
/// landmarks' half height. Fitted on 790 frames of two webcam clips (tools/face_box_fit.py).
pub const FIT_DX: f32 = 0.0066;
pub const FIT_DY: f32 = -0.1595;
pub const FIT_S: f32 = 1.1554;

impl FaceBox {
    pub fn from_points(points: &[[f32; 2]]) -> Option<Self> {
        Self::from_points_with(points, FIT_DX, FIT_DY, FIT_S)
    }

    pub fn from_points_with(points: &[[f32; 2]], dx: f32, dy: f32, ks: f32) -> Option<Self> {
        let (x0, y0, x1, y1) = Self::raw(points);
        if points.is_empty() || x1 <= x0 || y1 <= y0 {
            return None;
        }
        let half = (y1 - y0).max(x1 - x0) / 2.0;
        Some(Self { cx: (x0 + x1) / 2.0 + dx * half, cy: (y0 + y1) / 2.0 + dy * half, s: ks * half })
    }

    /// The landmarks' own box (x0, y0, x1, y1), for fitting.
    pub fn raw(points: &[[f32; 2]]) -> (f32, f32, f32, f32) {
        points.iter().fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |(x0, y0, x1, y1), p| (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1])))
    }
}

/// Element-wise median of each field over a window of `k` (odd) boxes, edges zero-padded like
/// scipy.signal.medfilt, which smooths the reference's tracks.
pub fn medfilt(boxes: &[FaceBox], k: usize) -> Vec<FaceBox> {
    let field = |f: &dyn Fn(&FaceBox) -> f32| -> Vec<f32> {
        let v: Vec<f32> = boxes.iter().map(f).collect();
        let h = k / 2;
        (0..v.len())
            .map(|i| {
                let mut w: Vec<f32> = (0..k).map(|j| (i + j).checked_sub(h).and_then(|t| v.get(t).copied()).unwrap_or(0.0)).collect();
                w.sort_by(f32::total_cmp);
                w[h]
            })
            .collect()
    };
    let (x, y, s) = (field(&|b| b.cx), field(&|b| b.cy), field(&|b| b.s));
    (0..boxes.len()).map(|i| FaceBox { cx: x[i], cy: y[i], s: s[i] }).collect()
}

/// Causal version for live use: each box is the median of itself and the `k - 1` before it.
pub fn median_causal(boxes: &[FaceBox], k: usize) -> Vec<FaceBox> {
    (0..boxes.len())
        .map(|i| {
            let w = &boxes[i.saturating_sub(k - 1)..=i];
            let med = |f: &dyn Fn(&FaceBox) -> f32| {
                let mut v: Vec<f32> = w.iter().map(f).collect();
                v.sort_by(f32::total_cmp);
                v[v.len() / 2]
            };
            FaceBox { cx: med(&|b| b.cx), cy: med(&|b| b.cy), s: med(&|b| b.s) }
        })
        .collect()
}

/// cv2.resize(INTER_LINEAR) as an affine map from a `w`×`h` destination onto a source region
/// of `sw`×`sh` starting at (`sx`, `sy`).
fn resize_map(sx: f64, sy: f64, sw: f64, sh: f64, w: usize, h: usize) -> Affine {
    let (fx, fy) = (sw / w as f64, sh / h as f64);
    Affine([fx, 0.0, sx + 0.5 * fx - 0.5, 0.0, fy, sy + 0.5 * fy - 0.5])
}

/// The 96×96 RGB face crop (HWC, u8) of one frame.
pub fn crop96(frame: &Frame, b: FaceBox) -> Vec<u8> {
    let (s, cs) = (f64::from(b.s), 0.4);
    let side = s * (2.0 + 2.0 * cs); // 2.8·s
    let (x0, y0) = (f64::from(b.cx) - s * (1.0 + cs), f64::from(b.cy) - s);
    let c224 = crop_rgb(frame, &resize_map(x0, y0, side, side, 224, 224), 224, 224, Border::Replicate, 0.0, 255.0);
    let c224: Vec<u8> = c224.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
    let f224 = Frame::rgb(&c224, 224, 224);
    // 224 → 160, keeping only the centre 96 (offset 32)
    let m = resize_map(0.0, 0.0, 224.0, 224.0, 160, 160);
    let m = Affine([m.0[0], 0.0, m.0[2] + 32.0 * m.0[0], 0.0, m.0[4], m.0[5] + 32.0 * m.0[4]]);
    crop_rgb(&f224, &m, 96, 96, Border::Replicate, 0.0, 255.0).iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}

/// Side of the stored face region, in pixels: the sampling density of the reference's 224 crop
/// (1.68·s·1.3 of 2.8·s at 224 px), so the last step is its 1.4× downscale too.
pub const REGION: usize = 176;
/// The region covers the final crop's square this many times over, so the crop can move with
/// the smoothed box afterwards.
pub const REGION_MARGIN: f32 = 1.3;

/// The final 96×96 crop's square in the frame for a box: the centre 60% of the 2.8·s syncnet
/// crop (224 → 160 → centre 96), i.e. side 1.68·s centred 0.4·s below the box centre.
fn final_square(b: FaceBox) -> (f32, f32, f32) {
    let side = 1.68 * b.s;
    (b.cx - side / 2.0, b.cy + 0.4 * b.s - side / 2.0, side)
}

/// A downscaled colour patch around the face, kept per frame until the boxes are smoothed.
#[derive(Clone)]
pub struct FaceRegion {
    pub rgb: Vec<u8>, // REGION × REGION × 3
    pub x0: f32,
    pub y0: f32,
    pub side: f32,
}

impl FaceRegion {
    pub fn capture(frame: &Frame, b: FaceBox) -> Self {
        let (fx, fy, fs) = final_square(b);
        let side = fs * REGION_MARGIN;
        let (x0, y0) = (fx - (side - fs) / 2.0, fy - (side - fs) / 2.0);
        let m = resize_map(f64::from(x0), f64::from(y0), f64::from(side), f64::from(side), REGION, REGION);
        let rgb = crop_rgb(frame, &m, REGION, REGION, Border::Replicate, 0.0, 255.0).iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
        Self { rgb, x0, y0, side }
    }

    /// The 96×96 RGB crop (HWC, u8) for a (smoothed) box.
    pub fn crop96(&self, b: FaceBox) -> Vec<u8> {
        let (fx, fy, fs) = final_square(b);
        let k = REGION as f32 / self.side; // frame px → region px
        let m = resize_map(f64::from((fx - self.x0) * k), f64::from((fy - self.y0) * k), f64::from(fs * k), f64::from(fs * k), 96, 96);
        let f = Frame::rgb(&self.rgb, REGION, REGION);
        crop_rgb(&f, &m, 96, 96, Border::Replicate, 0.0, 255.0).iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
    }
}
