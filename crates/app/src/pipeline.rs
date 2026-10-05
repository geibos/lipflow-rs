//! A recording (frames tracked while the key is held) and what turns it into model input:
//! the OS-independent part of `lipflow/dictation.py` and `camera.py`.

use std::sync::Arc;

use lipflow_face::face_crop::{FaceBox, FaceRegion, medfilt};
use lipflow_face::{Anchors, FaceLandmarker, FaceResult, Frame, GrayFrame, anchors, mouth_open, mouth_rois};
use lipflow_vsr::LipReader;

pub const MIN_SECONDS: f64 = 0.6;
pub const MAX_SECONDS: f64 = 60.0;
pub const PREVIEW_EVERY: f64 = 0.45;
pub const KEEP_CLIPS: usize = 100;
/// Keep filming after release: the last word needs the frames after it.
pub const TAIL_SECONDS: f64 = 0.4;
/// Dictations this close together get a separating space.
pub const JOIN_WINDOW: f64 = 45.0;

/// The face region of one frame, grayscale, and where it sits in the full frame.
pub struct FaceCrop {
    pub gray: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub offset: (usize, usize),
}

#[derive(Default)]
pub struct Recording {
    pub ts: Vec<f64>,
    pub crops: Vec<Arc<FaceCrop>>,
    pub anchors: Vec<Option<Anchors>>,
    pub mouth_open: Vec<f32>,
    /// Last face box (x0, y0, x1, y1), reused for frames without a face.
    pub box_: Option<(usize, usize, usize, usize)>,
    /// Colour face patches for MultiVSR (Russian), one per frame (None without a face); `None`
    /// for an English recording, which keeps grayscale crops for the mouth instead.
    pub faces: Option<Vec<Option<(FaceBox, Arc<FaceRegion>)>>>,
}

impl Recording {
    pub fn new() -> Self {
        Self::default()
    }

    /// A recording for MultiVSR: colour face patches instead of grayscale face crops.
    pub fn with_faces() -> Self {
        Self { faces: Some(Vec::new()), ..Self::default() }
    }

    pub fn duration(&self) -> f64 {
        match (self.ts.first(), self.ts.last()) {
            (Some(a), Some(b)) if self.ts.len() > 1 => b - a,
            _ => 0.0,
        }
    }

    pub fn face_ratio(&self) -> f64 {
        self.anchors.iter().filter(|a| a.is_some()).count() as f64 / self.anchors.len().max(1) as f64
    }

    /// Add a tracked frame: keep only the face (with margin) at full resolution, plus its offset —
    /// 720p detail without ~1 GB per minute of full frames.
    pub fn push(&mut self, t: f64, frame: &Frame, face: Option<&FaceResult>) {
        let (w, h) = (frame.width, frame.height);
        if let Some(f) = face {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for p in &f.points {
                x0 = x0.min(p[0]);
                y0 = y0.min(p[1]);
                x1 = x1.max(p[0]);
                y1 = y1.max(p[1]);
            }
            let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
            let side = (x1 - x0).max(y1 - y0) * 1.5;
            let clamp = |v: f32, hi: usize| (v.max(0.0) as usize).min(hi);
            self.box_ = Some((clamp(cx - side / 2.0, w), clamp(cy - side / 2.0, h), clamp(cx + side / 2.0, w), clamp(cy + side / 2.0, h)));
        }
        if let Some(faces) = &mut self.faces {
            let patch = face.and_then(|f| FaceBox::from_points(&f.points)).map(|b| (b, Arc::new(FaceRegion::capture(frame, b))));
            faces.push(patch);
            // no grayscale crop: keep the vectors aligned with an empty one
            self.ts.push(t);
            self.crops.push(Arc::new(FaceCrop { gray: Vec::new(), width: 0, height: 0, offset: (0, 0) }));
            self.anchors.push(face.map(|f| anchors(&f.points)));
            self.mouth_open.push(face.map_or(0.0, |f| mouth_open(&f.points)));
            return;
        }
        let (bx0, by0, bx1, by1) = self.box_.filter(|b| b.2 > b.0 && b.3 > b.1).unwrap_or((0, 0, w, h));
        let full = frame.to_gray_region(bx0, by0, bx1, by1);
        self.ts.push(t);
        self.crops.push(Arc::new(FaceCrop { gray: full, width: bx1 - bx0, height: by1 - by0, offset: (bx0, by0) }));
        self.anchors.push(face.map(|f| anchors(&f.points)));
        self.mouth_open.push(face.map_or(0.0, |f| mouth_open(&f.points)));
    }

    /// Why this recording can't be read, as (title, advice), or None if it is fine.
    pub fn problem(&self) -> Option<(&'static str, &'static str)> {
        if self.duration() < MIN_SECONDS || self.ts.len() < 12 {
            return Some(("Too short", "Hold the key while you mouth the words"));
        }
        if self.face_ratio() < 0.4 {
            return Some(("Can't see your face", "Face the camera with your mouth in view"));
        }
        let moving: Vec<f64> = self.mouth_open.iter().filter(|&&m| m > 0.0).map(|&m| f64::from(m)).collect();
        if std_dev(&moving) < 0.012 {
            return Some(("No lip movement", "Mouth the words clearly — no sound needed"));
        }
        None
    }

    /// A cheap copy for computing crops outside the camera lock (crops are shared).
    pub fn snapshot(&self) -> Snapshot {
        let n = self.ts.len().min(self.crops.len()).min(self.anchors.len());
        let faces = self.faces.as_ref().map(|f| f[..n.min(f.len())].to_vec());
        Snapshot { ts: self.ts[..n].to_vec(), crops: self.crops[..n].to_vec(), anchors: self.anchors[..n].to_vec(), faces }
    }

    pub fn rois(&self) -> Option<(Vec<u8>, usize)> {
        self.snapshot().rois()
    }

    pub fn face_frames(&self) -> Option<(Vec<u8>, usize)> {
        self.snapshot().face_frames()
    }
}

pub struct Snapshot {
    pub ts: Vec<f64>,
    pub crops: Vec<Arc<FaceCrop>>,
    pub anchors: Vec<Option<Anchors>>,
    pub faces: Option<Vec<Option<(FaceBox, Arc<FaceRegion>)>>>,
}

impl Snapshot {
    /// Mouth crops resampled to 25 fps: (T*96*96 bytes, T), or None without any face.
    pub fn rois(&self) -> Option<(Vec<u8>, usize)> {
        let idx = LipReader::resample(&self.ts);
        let frames: Vec<GrayFrame> = idx
            .iter()
            .map(|&i| {
                let c = &self.crops[i];
                GrayFrame { data: &c.gray, width: c.width, height: c.height, offset: c.offset }
            })
            .collect();
        let anchors: Vec<Option<Anchors>> = idx.iter().map(|&i| self.anchors[i]).collect();
        mouth_rois(&frames, &anchors).map(|r| (r, idx.len()))
    }
}

impl Snapshot {
    /// MultiVSR input: 96×96 RGB face crops (T·96·96·3 bytes, HWC) at 25 fps, the boxes
    /// median-smoothed over 13 frames as in the model's training data; None without any face.
    pub fn face_frames(&self) -> Option<(Vec<u8>, usize)> {
        let faces = self.faces.as_ref()?;
        let idx = LipReader::resample(&self.ts);
        // frames without a face borrow the nearest earlier patch (or the first one there is)
        let first = faces.iter().flatten().next()?.clone();
        let mut last = first;
        let picked: Vec<(FaceBox, Arc<FaceRegion>)> = idx
            .iter()
            .map(|&i| {
                if let Some(Some(p)) = faces.get(i) {
                    last = p.clone();
                }
                last.clone()
            })
            .collect();
        let boxes = medfilt(&picked.iter().map(|p| p.0).collect::<Vec<_>>(), 13);
        let mut out = Vec::with_capacity(picked.len() * 96 * 96 * 3);
        for ((_, region), b) in picked.iter().zip(&boxes) {
            out.extend(region.crop96(*b));
        }
        Some((out, picked.len()))
    }
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Track a clip of frames offline (file mode, tests) into `rec` (English or Russian kind).
pub fn track_clip(lm: &mut FaceLandmarker, mut rec: Recording, frames: impl IntoIterator<Item = (f64, Vec<u8>, usize, usize)>) -> anyhow::Result<Recording> {
    let mut last_ms = -1i64;
    for (t, bgra, w, h) in frames {
        let ms = ((t * 1000.0) as i64).max(last_ms + 1);
        last_ms = ms;
        let f = Frame::bgra(&bgra, w, h, w * 4);
        let face = lm.detect(&f, ms)?;
        rec.push(t, &f, face.as_ref());
    }
    Ok(rec)
}
