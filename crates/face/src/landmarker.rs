//! MediaPipe FaceLandmarker in VIDEO mode, one face: BlazeFace short-range detection when not
//! tracking, the 478-point face mesh on a rotated ROI, the ROI for the next frame from the
//! landmarks, and the One-Euro landmark smoothing of the tasks graph.

use anyhow::{Context, Result};

use crate::geom::{Border, Frame, crop_rgb, rect_to_src};
use crate::tflite::{Model, Scratch, read_task_entry};

pub const NUM_LANDMARKS: usize = 478;
const DET_SIZE: usize = 128;
const MESH_SIZE: usize = 256;
const MIN_DETECTION: f32 = 0.5;
const MIN_SUPPRESSION: f32 = 0.5;
const MIN_PRESENCE: f32 = 0.5;

/// A rotated rectangle in normalised image coordinates.
#[derive(Clone, Copy, Debug)]
pub struct NormRect {
    pub xc: f32,
    pub yc: f32,
    pub w: f32,
    pub h: f32,
    pub rot: f32,
}

#[derive(Clone, Debug)]
struct Detection {
    // normalised xmin, ymin, xmax, ymax
    bbox: [f32; 4],
    keypoints: [[f32; 2]; 6],
    score: f32,
}

/// SSD anchors (x, y) for the short-range model; fixed anchor size, so w = h = 1.
fn ssd_anchors() -> Vec<[f32; 2]> {
    let strides = [8usize, 16, 16, 16];
    let mut anchors = Vec::new();
    let mut layer = 0;
    while layer < strides.len() {
        let mut last = layer;
        let mut per_cell = 0;
        while last < strides.len() && strides[last] == strides[layer] {
            per_cell += 2; // aspect ratio 1.0 + the interpolated scale
            last += 1;
        }
        let grid = DET_SIZE.div_ceil(strides[layer]);
        for y in 0..grid {
            for x in 0..grid {
                for _ in 0..per_cell {
                    anchors.push([(x as f32 + 0.5) / grid as f32, (y as f32 + 0.5) / grid as f32]);
                }
            }
        }
        layer = last;
    }
    anchors
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// MediaPipe's WEIGHTED non-max suppression.
fn weighted_nms(mut dets: Vec<Detection>) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out = Vec::new();
    while !dets.is_empty() {
        let top = dets[0].clone();
        let (cands, rest): (Vec<Detection>, Vec<Detection>) = dets.into_iter().partition(|d| iou(&top.bbox, &d.bbox) > MIN_SUPPRESSION);
        let total: f32 = cands.iter().map(|d| d.score).sum();
        let mut w = top.clone();
        if total > 0.0 {
            w.bbox = [0.0; 4];
            w.keypoints = [[0.0; 2]; 6];
            for d in &cands {
                let s = d.score / total;
                for k in 0..4 {
                    w.bbox[k] += d.bbox[k] * s;
                }
                for k in 0..6 {
                    w.keypoints[k][0] += d.keypoints[k][0] * s;
                    w.keypoints[k][1] += d.keypoints[k][1] * s;
                }
            }
        }
        out.push(w);
        dets = rest;
    }
    out
}

fn normalize_radians(a: f32) -> f32 {
    use std::f32::consts::PI;
    a - 2.0 * PI * ((a + PI) / (2.0 * PI)).floor()
}

/// One-Euro filter (mediapipe/util/filtering/one_euro_filter.cc).
#[derive(Clone)]
struct OneEuro {
    freq: f64,
    last_ns: i64,
    x: Option<(f32, f32)>, // (raw, filtered)
    dx: Option<f32>,
}

const MIN_CUTOFF: f64 = 0.05;
const BETA: f64 = 80.0;
const D_CUTOFF: f64 = 1.0;

impl OneEuro {
    fn new() -> Self {
        Self { freq: 30.0, last_ns: -1, x: None, dx: None }
    }

    fn alpha(&self, cutoff: f64) -> f64 {
        let te = 1.0 / self.freq;
        let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
        1.0 / (1.0 + tau / te)
    }

    fn apply(&mut self, ts_ns: i64, value: f32, value_scale: f64) -> f32 {
        if self.last_ns >= ts_ns {
            return value;
        }
        if self.last_ns != 0 && ts_ns != 0 {
            self.freq = 1.0 / ((ts_ns - self.last_ns) as f64 * 1e-9);
        }
        self.last_ns = ts_ns;
        let dvalue = match self.x {
            Some((raw, _)) => f64::from(value - raw) * value_scale * self.freq,
            None => 0.0,
        };
        let a_d = self.alpha(D_CUTOFF);
        let edx = match self.dx {
            Some(prev) => (a_d * dvalue + (1.0 - a_d) * f64::from(prev)) as f32,
            None => dvalue as f32,
        };
        self.dx = Some(edx);
        let cutoff = MIN_CUTOFF + BETA * f64::from(edx).abs();
        let a = self.alpha(cutoff);
        let out = match self.x {
            Some((_, prev)) => (a * f64::from(value) + (1.0 - a) * f64::from(prev)) as f32,
            None => value,
        };
        self.x = Some((value, out));
        out
    }
}

pub struct FaceLandmarker {
    detector: Model,
    mesh: Model,
    anchors: Vec<[f32; 2]>,
    scratch: Scratch,
    prev_rect: Option<NormRect>,
    filters: Vec<[OneEuro; 2]>,
}

/// Smoothed landmarks of one frame in pixels, plus the face presence score.
#[derive(Clone, Debug)]
pub struct FaceResult {
    pub points: Vec<[f32; 2]>,
    pub presence: f32,
}

impl FaceLandmarker {
    pub fn from_task(task: &[u8]) -> Result<Self> {
        let detector = Model::parse(&read_task_entry(task, "face_detector.tflite")?).context("face detector")?;
        let mesh = Model::parse(&read_task_entry(task, "face_landmarks_detector.tflite")?).context("face mesh")?;
        Ok(Self { detector, mesh, anchors: ssd_anchors(), scratch: Scratch::default(), prev_rect: None, filters: Vec::new() })
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        Self::from_task(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
    }

    /// Forget the tracked face (the next frame runs the detector).
    pub fn reset(&mut self) {
        self.prev_rect = None;
        self.filters.clear();
    }

    fn detect_faces(&mut self, f: &Frame) -> Result<Vec<Detection>> {
        // Letterbox the whole frame into 128x128 (keep aspect ratio, zero border).
        let (iw, ih) = (f.width as f32, f.height as f32);
        let (side, pad_x, pad_y) = if ih / iw < 1.0 { (iw, 0.0, (1.0 - ih / iw) / 2.0) } else { (ih, (1.0 - iw / ih) / 2.0, 0.0) };
        let m = rect_to_src(f64::from(iw) / 2.0, f64::from(ih) / 2.0, f64::from(side), f64::from(side), 0.0, DET_SIZE, DET_SIZE);
        let input = crop_rgb(f, &m, DET_SIZE, DET_SIZE, Border::Zero, -1.0, 1.0);
        let out = self.detector.run(&input, &mut self.scratch)?;
        let (boxes, scores) = (&out[0], &out[1]);
        let s = DET_SIZE as f32;
        let unpad = |x: f32, y: f32| ((x - pad_x) / (1.0 - 2.0 * pad_x), (y - pad_y) / (1.0 - 2.0 * pad_y));
        let mut dets = Vec::new();
        for (i, a) in self.anchors.iter().enumerate() {
            let score = 1.0 / (1.0 + (-scores[i].clamp(-100.0, 100.0)).exp());
            if score < MIN_DETECTION {
                continue;
            }
            let r = &boxes[i * 16..][..16];
            // reverse_output_order: x, y, w, h
            let (xc, yc) = (r[0] / s + a[0], r[1] / s + a[1]);
            let (w, h) = (r[2] / s, r[3] / s);
            let (x0, y0) = unpad(xc - w / 2.0, yc - h / 2.0);
            let (x1, y1) = unpad(xc + w / 2.0, yc + h / 2.0);
            let mut kps = [[0f32; 2]; 6];
            for (k, kp) in kps.iter_mut().enumerate() {
                let (kx, ky) = unpad(r[4 + 2 * k] / s + a[0], r[5 + 2 * k] / s + a[1]);
                *kp = [kx, ky];
            }
            dets.push(Detection { bbox: [x0, y0, x1, y1], keypoints: kps, score });
        }
        Ok(weighted_nms(dets))
    }

    fn rect_from_detection(d: &Detection, iw: f32, ih: f32) -> NormRect {
        let [x0, y0, x1, y1] = d.bbox;
        let (w, h) = (x1 - x0, y1 - y0);
        let (p0, p1) = (d.keypoints[0], d.keypoints[1]);
        let rot = normalize_radians(-((-(p1[1] - p0[1]) * ih).atan2((p1[0] - p0[0]) * iw)));
        // RectTransformation: scale 1.5, no square_long for the detector path.
        NormRect { xc: x0 + w / 2.0, yc: y0 + h / 2.0, w: w * 1.5, h: h * 1.5, rot }
    }

    fn rect_from_landmarks(lms: &[[f32; 2]], iw: f32, ih: f32) -> NormRect {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in lms {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
        let (w, h) = (x1 - x0, y1 - y0);
        let (p0, p1) = (lms[33], lms[263]);
        let rot = normalize_radians(-((-(p1[1] - p0[1]) * ih).atan2((p1[0] - p0[0]) * iw)));
        let long = (w * iw).max(h * ih);
        NormRect { xc: x0 + w / 2.0, yc: y0 + h / 2.0, w: long / iw * 1.5, h: long / ih * 1.5, rot }
    }

    /// Face mesh on `rect`: (normalised landmarks, presence score).
    fn mesh(&mut self, f: &Frame, rect: &NormRect) -> Result<(Vec<[f32; 2]>, f32)> {
        let (iw, ih) = (f.width as f32, f.height as f32);
        let (cx, cy, w, h) = (rect.xc * iw, rect.yc * ih, rect.w * iw, rect.h * ih);
        let m = rect_to_src(f64::from(cx), f64::from(cy), f64::from(w), f64::from(h), f64::from(rect.rot), MESH_SIZE, MESH_SIZE);
        let input = crop_rgb(f, &m, MESH_SIZE, MESH_SIZE, Border::Replicate, 0.0, 1.0);
        let out = self.mesh.run(&input, &mut self.scratch)?;
        let presence = 1.0 / (1.0 + (-out[1][0]).exp());
        // GetRotatedSubRectToRectTransformMatrix, then ProjectXY.
        let (c, s) = (rect.rot.cos(), rect.rot.sin());
        let (g, hh) = (1.0 / iw, 1.0 / ih);
        let m0 = w * c * g;
        let m1 = -h * s * g;
        let m3 = (-0.5 * w * c + 0.5 * h * s + cx) * g;
        let m4 = w * s * hh;
        let m5 = h * c * hh;
        let m7 = (-0.5 * h * c - 0.5 * w * s + cy) * hh;
        let n = MESH_SIZE as f32;
        let pts = out[0]
            .chunks_exact(3)
            .take(NUM_LANDMARKS)
            .map(|p| {
                let (x, y) = (p[0] / n, p[1] / n);
                [m0 * x + m1 * y + m3, m4 * x + m5 * y + m7]
            })
            .collect();
        Ok((pts, presence))
    }

    /// Track the face in the next video frame (timestamps strictly increasing, in ms).
    pub fn detect(&mut self, f: &Frame, ts_ms: i64) -> Result<Option<FaceResult>> {
        let (iw, ih) = (f.width as f32, f.height as f32);
        let rect = match self.prev_rect {
            Some(r) => r,
            None => match self.detect_faces(f)?.first() {
                Some(d) => Self::rect_from_detection(d, iw, ih),
                None => {
                    self.filters.clear();
                    return Ok(None);
                }
            },
        };
        let (norm, presence) = self.mesh(f, &rect)?;
        if presence < MIN_PRESENCE {
            self.reset();
            return Ok(None);
        }
        self.prev_rect = Some(Self::rect_from_landmarks(&norm, iw, ih));

        // Smooth in pixels, velocity scaled by the face size.
        let px: Vec<[f32; 2]> = norm.iter().map(|p| [p[0] * iw, p[1] * ih]).collect();
        let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in &px {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
        let scale = ((x1 - x0) + (y1 - y0)) / 2.0;
        if self.filters.is_empty() {
            self.filters = vec![[OneEuro::new(), OneEuro::new()]; px.len()];
        }
        let ts_ns = ts_ms * 1_000_000;
        let points = if scale < 1e-6 {
            px
        } else {
            let vs = 1.0 / f64::from(scale);
            px.iter()
                .zip(self.filters.iter_mut())
                .map(|(p, fl)| [fl[0].apply(ts_ns, p[0], vs), fl[1].apply(ts_ns, p[1], vs)])
                .collect()
        };
        Ok(Some(FaceResult { points, presence }))
    }
}
