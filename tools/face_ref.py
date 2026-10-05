"""MediaPipe FaceLandmarker (VIDEO mode, as lipflow uses it) on a sample clip: frames, all 478
landmarks, lipflow's anchors and mouth crops — the reference for the Rust face pipeline.

Run from the lipflow checkout: uv run python ../tools/face_ref.py [video start end]
"""
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.getcwd())
from lipflow.face import FaceTracker, mouth_rois  # noqa: E402
from lipflow.vsr import LipReader  # noqa: E402

video = sys.argv[1] if len(sys.argv) > 1 else "samples/2016-03-12.mov"
start = float(sys.argv[2]) if len(sys.argv) > 2 else 20.4
end = float(sys.argv[3]) if len(sys.argv) > 3 else 28.1
OUT = os.path.join(os.path.dirname(__file__), "..", "ref", "face")

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
tr = FaceTracker()
frames, ts, ts_ms, lms, anchors, grays = [], [], [], [], [], []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = start + i / fps
    if t > end:
        break
    ms = max(int(t * 1000), tr._last_ts + 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = tr._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
    tr._last_ts = ms
    h, w = frame.shape[:2]
    if res.face_landmarks:
        p = np.array([(q.x, q.y, q.z) for q in res.face_landmarks[0]], np.float32)
        from lipflow.face import FaceObs
        anchors.append(FaceObs(p[:, :2] * [w, h]).anchors)
    else:
        p = np.full((478, 3), np.nan, np.float32)
        anchors.append(None)
    frames.append(rgb)
    grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    ts.append(t)
    ts_ms.append(ms)
    lms.append(p)
    i += 1
idx = LipReader.resample(ts, len(ts))
rois = mouth_rois([grays[k] for k in idx], [anchors[k] for k in idx])
np.stack(frames).tofile(os.path.join(OUT, "frames.u8"))
np.stack(grays).tofile(os.path.join(OUT, "grays.u8"))
np.stack(lms).astype(np.float32).tofile(os.path.join(OUT, "landmarks.f32"))
np.stack([a if a is not None else np.full((4, 2), np.nan) for a in anchors]).astype(np.float64).tofile(os.path.join(OUT, "anchors.f64"))
rois.tofile(os.path.join(OUT, "rois.u8"))
json.dump({"n": len(frames), "h": int(frames[0].shape[0]), "w": int(frames[0].shape[1]), "ts": ts, "ts_ms": ts_ms,
           "idx": idx, "rois": int(rois.shape[0])}, open(os.path.join(OUT, "meta.json"), "w"))
print(len(frames), "frames", frames[0].shape, "faces", sum(a is not None for a in anchors), "rois", rois.shape)
print(LipReader(beam_size=4, personal=False).beam_search(LipReader(beam_size=4, personal=False).encode(rois)) if False else "")
