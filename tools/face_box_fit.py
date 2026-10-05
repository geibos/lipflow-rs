"""Fit the S3FD face box (MultiVSR's crops) from the MediaPipe landmarks' box.

  python tools/face_box_fit.py WORK NAME [NAME...]
WORK/pywork/NAME/tracks.pckl: S3FD boxes from MultiVSR's preprocess/run_pipeline.py;
WORK/ours_NAME.csv: landmark boxes from `lipflow ru-file VIDEO --boxes`, same 25 fps frames.
Prints dx, dy, ks for crates/face/src/face_crop.rs (offsets and size in landmark half-sizes).
"""
import csv
import pickle
import sys

import numpy as np

work, names = sys.argv[1], sys.argv[2:]
rows = []
for name in names:
    # our own output of run_pipeline.py on our own videos, not untrusted input
    track = pickle.load(open(f"{work}/pywork/{name}/tracks.pckl", "rb"))[0]["track"]
    s3 = {int(f): b for f, b in zip(track["frame"], track["bbox"])}
    for r in csv.DictReader(open(f"{work}/ours_{name}.csv")):
        k = int(r["frame"])
        if k not in s3:
            continue
        x0, y0, x1, y1 = (float(r[c]) for c in ("x0", "y0", "x1", "y1"))
        half = max(x1 - x0, y1 - y0) / 2
        b = s3[k]
        s3s = max(b[2] - b[0], b[3] - b[1]) / 2
        rows.append((((b[0] + b[2]) / 2 - (x0 + x1) / 2) / half, ((b[1] + b[3]) / 2 - (y0 + y1) / 2) / half, s3s / half))
a = np.array(rows)
print(f"{len(a)} frames")
for i, n in enumerate(["dx", "dy", "ks"]):
    print(f"{n}: median {np.median(a[:, i]):.4f}  mean {a[:, i].mean():.4f}  std {a[:, i].std():.4f}")
