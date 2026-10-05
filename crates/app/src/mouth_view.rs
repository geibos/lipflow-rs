//! The live mouth close-up for the HUD and the setup window (`camera.mouth_view`): a mirrored crop
//! around the lips plus the lip contours in view coordinates. Drawing happens in the UI.

use lipflow_face::align::{INNER_LIPS, OUTER_LIPS};
use lipflow_face::{FaceResult, Frame};

const LIP_POINTS: [usize; 40] = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318,
    321, 324, 375, 402, 405, 409, 415,
];

/// An RGBA image (top row first) and lip geometry in its pixel coordinates (y down).
pub struct MouthView {
    pub rgba: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub face: bool,
    pub outer: Vec<[f32; 2]>,
    pub inner: Vec<[f32; 2]>,
    pub points: Vec<[f32; 2]>,
}

fn sample(frame: &Frame, x: f32, y: f32) -> [u8; 3] {
    // Bilinear, edges replicated (the overlay only needs to look right).
    let xf = x.clamp(0.0, (frame.width - 1) as f32);
    let yf = y.clamp(0.0, (frame.height - 1) as f32);
    let (x0, y0) = (xf.floor() as usize, yf.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(frame.width - 1), (y0 + 1).min(frame.height - 1));
    let (fx, fy) = (xf - x0 as f32, yf - y0 as f32);
    let px = |x: usize, y: usize, c: usize| f32::from(frame.data[y * frame.stride + x * frame.bpp + frame.rgb[c]]);
    let mut out = [0u8; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let top = px(x0, y0, c) * (1.0 - fx) + px(x1, y0, c) * fx;
        let bot = px(x0, y1, c) * (1.0 - fx) + px(x1, y1, c) * fx;
        *o = (top * (1.0 - fy) + bot * fy).round() as u8;
    }
    out
}

/// Mirrored close-up of the lips; without a face, the whole frame dimmed so you can line up.
pub fn mouth_view(frame: &Frame, face: Option<&FaceResult>, w: usize, h: usize) -> MouthView {
    let mut rgba = vec![255u8; w * h * 4];
    let Some(face) = face else {
        let (sx, sy) = (frame.width as f32 / w as f32, frame.height as f32 / h as f32);
        for y in 0..h {
            for x in 0..w {
                let p = sample(frame, (w - 1 - x) as f32 * sx, y as f32 * sy);
                let o = &mut rgba[(y * w + x) * 4..][..3];
                for c in 0..3 {
                    o[c] = (f32::from(p[c]) * 0.45) as u8;
                }
            }
        }
        return MouthView { rgba, width: w, height: h, face: false, outer: Vec::new(), inner: Vec::new(), points: Vec::new() };
    };
    let lips: Vec<[f32; 2]> = OUTER_LIPS.iter().map(|&i| face.points[i]).collect();
    let n = lips.len() as f32;
    let (cx, cy) = (lips.iter().map(|p| p[0]).sum::<f32>() / n, lips.iter().map(|p| p[1]).sum::<f32>() / n);
    let (lo, hi) = lips.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    let half_w = (hi - lo).max(10.0) * 0.9;
    let half_h = half_w * h as f32 / w as f32;
    let (x0, y0) = (cx - half_w, cy - half_h);
    let scale = w as f32 / (2.0 * half_w);
    for y in 0..h {
        for x in 0..w {
            // mirrored: view column x shows source column (w-1-x)
            let p = sample(frame, x0 + (w - 1 - x) as f32 / scale, y0 + y as f32 / scale);
            rgba[(y * w + x) * 4..][..3].copy_from_slice(&p);
        }
    }
    let to_view = |idx: &[usize]| -> Vec<[f32; 2]> {
        idx.iter()
            .map(|&i| {
                let p = face.points[i];
                [w as f32 - 1.0 - (p[0] - x0) * scale, (p[1] - y0) * scale]
            })
            .collect()
    };
    MouthView { rgba, width: w, height: h, face: true, outer: to_view(&OUTER_LIPS), inner: to_view(&INNER_LIPS), points: to_view(&LIP_POINTS) }
}
