//! The interpreter against LiteRT on the same input (`tools/tflite_ref.py`).

use std::path::PathBuf;

use lipflow_face::tflite::{Model, Scratch, read_task_entry};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn f32s(path: &PathBuf) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn matches_litert() {
    let task_path = root().join("lipflow/models/face_landmarker.task");
    let refdir = root().join("ref/tflite");
    if !task_path.exists() || !refdir.join("face_detector.in.bin").exists() {
        eprintln!("skipping: run setup + tools/tflite_ref.py");
        return;
    }
    let task = std::fs::read(task_path).unwrap();
    for name in ["face_detector", "face_landmarks_detector"] {
        let model = Model::parse(&read_task_entry(&task, &format!("{name}.tflite")).unwrap()).unwrap();
        let x = f32s(&refdir.join(format!("{name}.in.bin")));
        let t = std::time::Instant::now();
        let outs = model.run(&x, &mut Scratch::default()).unwrap();
        let dt = t.elapsed();
        for (i, y) in outs.iter().enumerate() {
            let r = f32s(&refdir.join(format!("{name}.out{i}.bin")));
            assert_eq!(y.len(), r.len(), "{name} out{i} size");
            let scale = r.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let err = y.iter().zip(&r).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("{name} out{i}: max|Δ| {err:.2e} (scale {scale:.2}) in {dt:?}");
            assert!(err <= 1e-3 * scale, "{name} out{i} differs: {err}");
        }
    }
}
