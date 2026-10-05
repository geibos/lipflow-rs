//! Against MediaPipe + OpenCV on a sample clip (`tools/face_ref.py`).

use std::path::PathBuf;

use lipflow_face::{Anchors, FaceLandmarker, Frame, GrayFrame, anchors, mouth_rois};

struct Ref {
    n: usize,
    h: usize,
    w: usize,
    ts_ms: Vec<i64>,
    idx: Vec<usize>,
    frames: Vec<u8>,
    grays: Vec<u8>,
    landmarks: Vec<f32>,
    anchors: Vec<Option<Anchors>>,
    rois: Vec<u8>,
}

fn load() -> Option<Ref> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ref/face");
    let meta: String = std::fs::read_to_string(dir.join("meta.json")).ok()?;
    let num = |key: &str| -> usize {
        let s = meta.split(&format!("\"{key}\": ")).nth(1).unwrap();
        s.split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()
    };
    let list = |key: &str| -> Vec<f64> {
        let s = meta.split(&format!("\"{key}\": [")).nth(1).unwrap();
        s.split(']').next().unwrap().split(',').map(|v| v.trim().parse().unwrap()).collect()
    };
    let f32s = |b: Vec<u8>| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let a64: Vec<f64> = std::fs::read(dir.join("anchors.f64")).ok()?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let anchors = a64
        .chunks_exact(8)
        .map(|a| if a[0].is_nan() { None } else { Some([[a[0] as f32, a[1] as f32], [a[2] as f32, a[3] as f32], [a[4] as f32, a[5] as f32], [a[6] as f32, a[7] as f32]]) })
        .collect();
    Some(Ref {
        n: num("n"),
        h: num("h"),
        w: num("w"),
        ts_ms: list("ts_ms").iter().map(|&v| v as i64).collect(),
        idx: list("idx").iter().map(|&v| v as usize).collect(),
        frames: std::fs::read(dir.join("frames.u8")).ok()?,
        grays: std::fs::read(dir.join("grays.u8")).ok()?,
        landmarks: f32s(std::fs::read(dir.join("landmarks.f32")).ok()?),
        anchors,
        rois: std::fs::read(dir.join("rois.u8")).ok()?,
    })
}

fn crops_vs(r: &Ref, anchors: &[Option<Anchors>], what: &str) -> (f64, usize) {
    let frames: Vec<GrayFrame> = r.idx.iter().map(|&i| GrayFrame { data: &r.grays[i * r.w * r.h..][..r.w * r.h], width: r.w, height: r.h, offset: (0, 0) }).collect();
    let picked: Vec<Option<Anchors>> = r.idx.iter().map(|&i| anchors[i]).collect();
    let rois = mouth_rois(&frames, &picked).unwrap();
    assert_eq!(rois.len(), r.rois.len());
    let diff: Vec<i32> = rois.iter().zip(&r.rois).map(|(&a, &b)| i32::from(a) - i32::from(b)).collect();
    for (fi, ch) in diff.chunks(96 * 96).enumerate() {
        let m = ch.iter().map(|d| f64::from(d.abs())).sum::<f64>() / ch.len() as f64;
        if m > 0.5 {
            eprintln!("  frame {fi}: MAE {m:.2}");
        }
    }
    let mae = diff.iter().map(|d| f64::from(d.abs())).sum::<f64>() / diff.len() as f64;
    let max = diff.iter().map(|d| d.unsigned_abs() as usize).max().unwrap();
    eprintln!("{what}: crop MAE {mae:.4} gray levels, max {max}, exact {:.2}%", 100.0 * diff.iter().filter(|&&d| d == 0).count() as f64 / diff.len() as f64);
    (mae, max)
}

#[test]
fn crops_from_python_anchors_match_opencv() {
    let Some(r) = load() else { return eprintln!("skipping: run tools/face_ref.py") };
    let (mae, _) = crops_vs(&r, &r.anchors, "python anchors");
    assert!(mae < 0.05, "alignment/warp differs from OpenCV");
}

#[test]
fn landmarks_and_crops_match_mediapipe() {
    let Some(r) = load() else { return eprintln!("skipping: run tools/face_ref.py") };
    let task = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lipflow/models/face_landmarker.task");
    let mut lm = FaceLandmarker::load(&task).unwrap();
    let fsz = r.w * r.h * 3;
    let mut ours: Vec<Option<Anchors>> = Vec::new();
    let (mut err_sum, mut err_max, mut cnt) = (0f64, 0f64, 0usize);
    let t = std::time::Instant::now();
    for i in 0..r.n {
        let f = Frame::rgb(&r.frames[i * fsz..][..fsz], r.w, r.h);
        let res = lm.detect(&f, r.ts_ms[i]).unwrap();
        let py = &r.landmarks[i * 478 * 3..][..478 * 3];
        match &res {
            Some(res) if !py[0].is_nan() => {
                for (k, p) in res.points.iter().enumerate() {
                    let (px, pyy) = (py[3 * k] * r.w as f32, py[3 * k + 1] * r.h as f32);
                    let e = f64::from(((p[0] - px).powi(2) + (p[1] - pyy).powi(2)).sqrt());
                    err_sum += e;
                    err_max = err_max.max(e);
                    cnt += 1;
                }
            }
            _ => assert_eq!(res.is_some(), !py[0].is_nan(), "face presence differs at frame {i}"),
        }
        ours.push(res.map(|r| anchors(&r.points)));
    }
    let per_frame = t.elapsed() / r.n as u32;
    eprintln!("landmarks: mean error {:.3} px, max {:.3} px over {cnt} points; {per_frame:?}/frame", err_sum / cnt as f64, err_max);
    let mut aerr = 0f64;
    for (a, b) in ours.iter().zip(&r.anchors) {
        if let (Some(a), Some(b)) = (a, b) {
            for p in 0..4 {
                aerr = aerr.max(f64::from((a[p][0] - b[p][0]).hypot(a[p][1] - b[p][1])));
            }
        }
    }
    eprintln!("anchors: max error {aerr:.3} px");
    crops_vs(&r, &ours, "rust landmarks");
    assert!(err_sum / (cnt as f64) < 0.5, "landmarks drift from MediaPipe");
}
