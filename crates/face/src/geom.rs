//! Image sampling with OpenCV's fixed-point bilinear interpolation (`warpAffine` /
//! `warpPerspective` with `INTER_LINEAR` on 8-bit images), so crops match the Python pipeline.

/// A borrowed 8-bit image with interleaved channels (RGB, BGR, BGRA, gray…).
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    /// Bytes per row.
    pub stride: usize,
    /// Bytes per pixel.
    pub bpp: usize,
    /// Byte offsets of R, G, B inside a pixel.
    pub rgb: [usize; 3],
}

impl<'a> Frame<'a> {
    pub fn rgb(data: &'a [u8], width: usize, height: usize) -> Self {
        Self { data, width, height, stride: width * 3, bpp: 3, rgb: [0, 1, 2] }
    }

    pub fn bgr(data: &'a [u8], width: usize, height: usize) -> Self {
        Self { data, width, height, stride: width * 3, bpp: 3, rgb: [2, 1, 0] }
    }

    pub fn bgra(data: &'a [u8], width: usize, height: usize, stride: usize) -> Self {
        Self { data, width, height, stride, bpp: 4, rgb: [2, 1, 0] }
    }

    #[inline]
    fn px(&self, x: usize, y: usize, ch: usize) -> u8 {
        self.data[y * self.stride + x * self.bpp + ch]
    }

    /// OpenCV `COLOR_RGB2GRAY` (fixed point, BT.601 weights).
    pub fn to_gray(&self) -> Vec<u8> {
        self.to_gray_region(0, 0, self.width, self.height)
    }

    /// Grayscale of the region [x0, x1) × [y0, y1).
    pub fn to_gray_region(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            let row = &self.data[y * self.stride..];
            for x in x0..x1 {
                let p = &row[x * self.bpp..];
                let (r, g, b) = (u32::from(p[self.rgb[0]]), u32::from(p[self.rgb[1]]), u32::from(p[self.rgb[2]]));
                out.push(((r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14) as u8);
            }
        }
        out
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Border {
    Zero,
    Replicate,
}

const INTER_BITS: i32 = 5;
const INTER_TAB: i32 = 1 << INTER_BITS;

/// cvRound: round half to even.
#[inline]
fn cv_round(x: f64) -> i32 {
    x.round_ties_even().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// Bilinear sample at a fixed-point source position (X, Y in 1/32 px), all channels.
#[inline]
fn sample<const C: usize>(get: impl Fn(usize, usize, usize) -> u8, w: usize, h: usize, xq: i32, yq: i32, border: Border, out: &mut [u8; C]) {
    let (x0, y0) = (xq >> INTER_BITS, yq >> INTER_BITS);
    let (fx, fy) = (xq & (INTER_TAB - 1), yq & (INTER_TAB - 1));
    // Weights scaled to 2^15, exact for bilinear: (32-fx)(32-fy)*32 etc.
    let wts = [(INTER_TAB - fx) * (INTER_TAB - fy) * 32, fx * (INTER_TAB - fy) * 32, (INTER_TAB - fx) * fy * 32, fx * fy * 32];
    let corners = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)];
    let mut acc = [0i32; C];
    for (k, &(cx, cy)) in corners.iter().enumerate() {
        let inside = cx >= 0 && cy >= 0 && (cx as usize) < w && (cy as usize) < h;
        let (sx, sy) = match (inside, border) {
            (true, _) => (cx as usize, cy as usize),
            (false, Border::Zero) => continue,
            (false, Border::Replicate) => (cx.clamp(0, w as i32 - 1) as usize, cy.clamp(0, h as i32 - 1) as usize),
        };
        for c in 0..C {
            acc[c] += wts[k] * i32::from(get(sx, sy, c));
        }
    }
    for c in 0..C {
        out[c] = ((acc[c] + (1 << 14)) >> 15).clamp(0, 255) as u8;
    }
}

/// An affine map from destination pixels to source pixels: src = A·[x, y, 1].
#[derive(Clone, Copy, Debug)]
pub struct Affine(pub [f64; 6]);

impl Affine {
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.0;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }

    pub fn invert(&self) -> Self {
        let m = &self.0;
        let det = m[0] * m[4] - m[1] * m[3];
        let d = if det != 0.0 { 1.0 / det } else { 0.0 };
        let (a, b, c, e) = (m[4] * d, -m[1] * d, -m[3] * d, m[0] * d);
        Self([a, b, -a * m[2] - b * m[5], c, e, -c * m[2] - e * m[5]])
    }
}

/// Map from a `out_w`×`out_h` crop to the source image for a rotated rectangle in pixels
/// (cv::RotatedRect + boxPoints + getPerspectiveTransform, as MediaPipe's ImageToTensor).
pub fn rect_to_src(cx: f64, cy: f64, w: f64, h: f64, rotation: f64, out_w: usize, out_h: usize) -> Affine {
    let (s, c) = rotation.sin_cos();
    let tlx = cx - 0.5 * w * c + 0.5 * h * s;
    let tly = cy - 0.5 * w * s - 0.5 * h * c;
    let (sx, sy) = (w / out_w as f64, h / out_h as f64);
    Affine([sx * c, -sy * s, tlx, sx * s, sy * c, tly])
}

/// warpPerspective-style crop to RGB `out_w`×`out_h`, as f32 in [lo, hi].
pub fn crop_rgb(frame: &Frame, dst_to_src: &Affine, out_w: usize, out_h: usize, border: Border, lo: f32, hi: f32) -> Vec<f32> {
    let m = &dst_to_src.0;
    let scale = (hi - lo) / 255.0;
    let get = |x: usize, y: usize, c: usize| frame.px(x, y, frame.rgb[c]);
    let mut out = Vec::with_capacity(out_w * out_h * 3);
    let mut px = [0u8; 3];
    for y in 0..out_h {
        for x in 0..out_w {
            let (fx, fy) = (x as f64, y as f64);
            let xq = cv_round((m[0] * fx + m[1] * fy + m[2]) * f64::from(INTER_TAB));
            let yq = cv_round((m[3] * fx + m[4] * fy + m[5]) * f64::from(INTER_TAB));
            sample::<3>(get, frame.width, frame.height, xq, yq, border, &mut px);
            out.extend(px.iter().map(|&v| f32::from(v) * scale + lo));
        }
    }
    out
}

/// cv::warpAffine(gray, M, (out_w, out_h), INTER_LINEAR, BORDER_CONSTANT 0) where `m` maps
/// source to destination (it is inverted here, as OpenCV does).
pub fn warp_affine_gray(src: &[u8], w: usize, h: usize, m: &Affine, out_w: usize, out_h: usize) -> Vec<u8> {
    const AB_BITS: i32 = 10;
    const AB_SCALE: f64 = (1 << AB_BITS) as f64;
    let inv = m.invert().0;
    let round_delta = (AB_SCALE as i32) / INTER_TAB / 2;
    let adelta: Vec<i32> = (0..out_w).map(|x| cv_round(inv[0] * x as f64 * AB_SCALE)).collect();
    let bdelta: Vec<i32> = (0..out_w).map(|x| cv_round(inv[3] * x as f64 * AB_SCALE)).collect();
    let get = |x: usize, y: usize, _c: usize| src[y * w + x];
    let mut out = vec![0u8; out_w * out_h];
    let mut px = [0u8; 1];
    for y in 0..out_h {
        let x0 = cv_round((inv[1] * y as f64 + inv[2]) * AB_SCALE) + round_delta;
        let y0 = cv_round((inv[4] * y as f64 + inv[5]) * AB_SCALE) + round_delta;
        for x in 0..out_w {
            let xq = (x0 + adelta[x]) >> (AB_BITS - INTER_BITS);
            let yq = (y0 + bdelta[x]) >> (AB_BITS - INTER_BITS);
            sample::<1>(get, w, h, xq, yq, Border::Zero, &mut px);
            out[y * out_w + x] = px[0];
        }
    }
    out
}
