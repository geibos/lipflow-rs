"""Python baseline: timings + WER of the original pipeline, and reference data for the Rust port.

Run from the lipflow checkout:  uv run python ../tools/baseline.py
Writes ../ref/bench/<i>.npy (mouth crops) and ../ref/baseline.json.
"""
import json
import os
import resource
import sys
import time

t_import = time.time()
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.getcwd())
from lipflow import bench  # noqa: E402
from lipflow.vsr import LipReader  # noqa: E402

t_import = time.time() - t_import

OUT = os.path.join(os.path.dirname(__file__), "..", "ref")
os.makedirs(os.path.join(OUT, "bench"), exist_ok=True)

items = bench.build()
print(f"{len(items)} clips")
for i, it in enumerate(items):
    np.save(os.path.join(OUT, "bench", f"{i:03d}.npy"), it["rois"])

t0 = time.time()
reader = LipReader(beam_size=4, personal=False)
t_load = time.time() - t0
reader.warmup()

res = {"import_s": t_import, "load_s": t_load, "clips": []}
errs = n = 0
for i, it in enumerate(items):
    t0 = time.time()
    enc = reader.encode(it["rois"])
    if reader.enc_device.type == "mps":
        torch.mps.synchronize()
    t_enc = time.time() - t0
    greedy = reader.greedy(enc)
    t1 = time.time()
    hyps = reader.beam_search(enc, nbest=5)
    t_beam = time.time() - t1
    e, m = bench.wer(hyps[0], it["text"])
    errs, n = errs + e, n + m
    res["clips"].append({"i": i, "frames": int(it["rois"].shape[0]), "text": it["text"], "greedy": greedy,
                         "hyps": hyps, "enc_s": t_enc, "beam_s": t_beam})
    print(f"{i:3d} T={it['rois'].shape[0]:3d} enc {t_enc:.3f}s beam {t_beam:.3f}s  {hyps[0]}")

res["wer_beam4"] = errs / n
res["enc_s_total"] = sum(c["enc_s"] for c in res["clips"])
res["beam_s_total"] = sum(c["beam_s"] for c in res["clips"])
res["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
json.dump(res, open(os.path.join(OUT, "baseline.json"), "w"), indent=1)
