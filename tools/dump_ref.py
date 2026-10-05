"""Reference tensors from the PyTorch model for numerical checks of the Rust port.

Run from the lipflow checkout: uv run python ../tools/dump_ref.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from lipflow.vsr import LipReader  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "ref")
r = LipReader(device="cpu", beam_size=4, personal=False)
rois = np.load(os.path.join(OUT, "bench", "001.npy"))
with torch.inference_mode():
    x = r.to_tensor(rois).unsqueeze(0)
    feats = r.model.encoder.frontend(x)
    enc, _ = r.model.encoder(x, None)
    logp = r.model.ctc.log_softmax(enc)
np.save(os.path.join(OUT, "ref_feats_001.npy"), feats[0].numpy().astype(np.float32))
np.save(os.path.join(OUT, "ref_enc_001.npy"), enc[0].numpy().astype(np.float32))
np.save(os.path.join(OUT, "ref_ctc_001.npy"), logp[0].numpy().astype(np.float32))
print("feats", tuple(feats.shape), "enc", tuple(enc.shape))
print("beam (cpu):", r.beam_search(enc[0], nbest=5))
