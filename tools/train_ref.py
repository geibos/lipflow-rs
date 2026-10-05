"""PyTorch reference for the Rust training port: loss parts and gradient norms of the trainable
parameters (scope frontend+encoder1) on one bench clip, no augmentation (centre crop).

Run from the lipflow checkout: uv run python ../tools/train_ref.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from lipflow.train_vsr import _targets, trainable  # noqa: E402
from lipflow.vsr import LipReader  # noqa: E402
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "ref")
base = json.load(open(os.path.join(OUT, "baseline.json")))
r = LipReader(device="cpu", beam_size=4, personal=False)
m = r.model
params = trainable(r, "frontend+encoder1")
m.eval()  # deterministic: no dropout (the Rust check compares exact values)
for mod in m.modules():
    if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        mod.eval()
out = []
for i in (1, 2):
    rois = np.load(os.path.join(OUT, "bench", f"{i:03d}.npy"))
    text = base["clips"][i]["text"]
    ys = _targets(r, text)
    x = r.to_tensor(rois)
    hs, _ = m.encoder(x.unsqueeze(0), None)
    logp = m.ctc.ctc_lo(hs).log_softmax(-1).transpose(0, 1)
    y = torch.tensor(ys, dtype=torch.long)
    ctc = F.ctc_loss(logp, y.unsqueeze(0), torch.tensor([logp.shape[0]]), torch.tensor([len(ys)]), blank=0, reduction="sum", zero_infinity=True)
    sos = eos = m.eos
    ys_in = torch.tensor([[sos] + ys])
    ys_out = torch.tensor([ys + [eos]])
    mask = subsequent_mask(ys_in.shape[1]).unsqueeze(0)
    pred, _ = m.decoder(ys_in, mask, hs, None)
    att = m.criterion(pred, ys_out)
    loss = (0.1 * ctc + 0.9 * att) / len(ys)
    for p in params:
        p.grad = None
    loss.backward()
    named = {n: p for n, p in m.named_parameters() if p.requires_grad}
    gn = {n: float(p.grad.norm()) for n, p in named.items() if p.grad is not None}
    total = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in params if p.grad is not None)))
    out.append({"i": i, "ys": ys, "ctc": float(ctc), "att": float(att), "loss": float(loss), "grad_total": total,
                "grad": {k: gn[k] for k in ["encoder.frontend.frontend3D.0.weight", "encoder.frontend.frontend3D.1.weight",
                                             "encoder.embed.0.weight", "encoder.encoders.0.self_attn.linear_q.weight",
                                             "encoder.encoders.0.conv_module.norm.weight"]}})
    print(i, len(ys), f"ctc {float(ctc):.4f} att {float(att):.4f} loss {float(loss):.5f} |g| {total:.5f}")
json.dump(out, open(os.path.join(OUT, "train_ref.json"), "w"), indent=1)
