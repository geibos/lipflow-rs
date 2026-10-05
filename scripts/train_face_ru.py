"""Fine-tune MultiVSR (Russian lip reading) on your practice clips, on the CPU.

The Rust app runs this and loads what it writes: <Lipflow home>/models/multivsr_face.safetensors,
the trained tensors only (laid over base-models/multivsr/multivsr.safetensors). The model is
defined here again in plain PyTorch with the same tensor names, so only torch, numpy,
safetensors and tokenizers are needed (`uv run --with torch --with numpy --with safetensors
--with tokenizers python scripts/train_face_ru.py`).

Clips: <home>/clips/onboarding-ru/*.npz with "faces" (T, 96, 96, 3) uint8 and "text". A quarter
is held out; the result is kept only if it reads the held-out clips better than the base model.
Prints "PROGRESS <pct> <text>" lines and a final "RESULT <json>" line, like train_face.py.

Env: LIPFLOW_HOME, LIPFLOW_MULTIVSR (model dir), LIPFLOW_TRAIN_PARTS (vtp | vtp+enc | all),
LIPFLOW_TRAIN_EPOCHS, LIPFLOW_TRAIN_LR, LIPFLOW_TRAIN_DRY=1 (evaluate only, write nothing).
"""
import glob
import json
import math
import os
import random
import re
import sys
import time
import zipfile
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

torch.set_grad_enabled(False)
DEV = torch.device("cpu")  # never MPS: GPU training rebooted this Mac
SOT, RU, TRANSCRIBE, NOTS, EOT = 50258, 50263, 50359, 50363, 50257
PREFIX = [SOT, RU, TRANSCRIBE, NOTS]


def report(pct, text):
    print(f"PROGRESS {pct:.0f} {text}", flush=True)


# ---------------------------------------------------------------------------- model (mirrors crates/vsr/src/multivsr.rs)

class Conv3dBn(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv3d(cin, cout, k, s, p), nn.BatchNorm3d(cout))


class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Conv3dBn(3, 64, 5, (1, 2, 2), 2), Conv3dBn(64, 128, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                                     Conv3dBn(128, 128, (1, 3, 3), 1, (0, 1, 1)))

    def forward(self, x):  # (B, 3, T, 96, 96)
        e = self.encoder
        x = F.relu(e[0].conv_block(x))
        x = F.relu(e[1].conv_block(x))
        return F.relu(e[2].conv_block(x) + x)


class Pos(nn.Module):
    def __init__(self):
        super().__init__()
        self.row_embed = nn.Embedding(64, 64)
        self.col_embed = nn.Embedding(64, 64)

    def forward(self, x):  # (N, 128, 24, 24)
        h, w = x.shape[-2:]
        pos = torch.cat([self.col_embed.weight[:w].unsqueeze(0).repeat(h, 1, 1), self.row_embed.weight[:h].unsqueeze(1).repeat(1, w, 1)], -1)
        return x + pos.permute(2, 0, 1).unsqueeze(0)


class LinAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        dh = d // self.h
        q, k, v = (f(x).reshape(b, n, self.h, dh).transpose(1, 2) for f in (self.to_q, self.to_k, self.to_v))
        q = q.softmax(-1) * dh ** -0.5
        k = k.softmax(-2)
        out = q @ (k.transpose(-2, -1) @ v)
        return self.to_out(out.transpose(1, 2).reshape(b, n, d))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn


class FF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4)
        self.w2 = nn.Linear(dim * 4, dim)


class Chunk(nn.Module):  # name holder: ...1.fn.fn.w1
    def __init__(self, fn):
        super().__init__()
        self.fn = fn


class LatBlock(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()
        self.layers = nn.Module()
        self.layers.layers = nn.ModuleList([nn.ModuleList([PreNorm(dim, LinAttn(dim, heads)), PreNorm(dim, Chunk(FF(dim)))]) for _ in range(depth)])

    def forward(self, x):
        for a, f in self.layers.layers:
            x = x + a.fn(a.norm(x))
            y = f.norm(x)
            x = x + f.fn.fn.w2(F.gelu(f.fn.fn.w1(y)))
        return x


class Mlp(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.fc2 = nn.Linear(h, o)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Proj(nn.Module):  # Sequential(Rearrange, Mlp): the Mlp is ".1"
    def __init__(self, i, o):
        super().__init__()
        self.add_module("1", Mlp(i, o, o))

    def forward(self, x):
        return getattr(self, "1")(x)


class VtpEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_projectors = nn.ModuleList([Proj(128, 256), Proj(1024, 512)])
        self.transformer_blocks = nn.ModuleList([LatBlock(256, 8, 3), LatBlock(512, 8, 3)])

    def forward(self, x):  # (N, 128, 24, 24)
        n = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (N, 576, 128)
        x = self.transformer_blocks[0](self.patch_projectors[0](x))
        x = x.reshape(n, 12, 2, 12, 2, 256).permute(0, 1, 3, 2, 4, 5).reshape(n, 144, 1024)
        return self.transformer_blocks[1](self.patch_projectors[1](x))


class Vtp(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extrator = FeatExtractor()
        self.hwposition = Pos()
        self.encoder = VtpEncoder()
        self.pooler = nn.Linear(512, 1)

    def forward(self, faces):  # (T, 3, 96, 96) → (T, 512)
        c = self.feat_extrator(faces.permute(1, 0, 2, 3).unsqueeze(0))[0].transpose(0, 1)  # (T, 128, 24, 24)
        x = self.encoder(self.hwposition(c))
        w = self.pooler(x).softmax(1)
        return (x * w).sum(1)


class AtLn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d))
        self.b_2 = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return self.a_2 * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6) + self.b_2


class Sub(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = AtLn(d)


class Mha(nn.Module):
    def __init__(self, d=768, h=12):
        super().__init__()
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])

    def forward(self, q, k, v, mask=None):
        b, d = q.shape[0], q.shape[-1]
        dk = d // self.h
        q, k, v = (l(x).view(b, -1, self.h, dk).transpose(1, 2) for l, x in zip(self.linears, (q, k, v)))
        s = q @ k.transpose(-2, -1) / math.sqrt(dk)
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)
        return self.linears[3]((s.softmax(-1) @ v).transpose(1, 2).reshape(b, -1, d))


class PFF(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.w_1 = nn.Linear(d, 4 * d)
        self.w_2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class EncLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Mha()
        self.feed_forward = PFF()
        self.sublayer = nn.ModuleList([Sub(768), Sub(768)])

    def forward(self, x):
        y = self.sublayer[0].norm(x)
        x = x + self.self_attn(y, y, y)
        return x + self.feed_forward(self.sublayer[1].norm(x))


class DecLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Mha()
        self.src_attn = Mha()
        self.feed_forward = PFF()
        self.sublayer = nn.ModuleList([Sub(768), Sub(768), Sub(768)])

    def forward(self, x, m, mask):
        y = self.sublayer[0].norm(x)
        x = x + self.self_attn(y, y, y, mask)
        y = self.sublayer[1].norm(x)
        x = x + self.src_attn(y, m, m)
        return x + self.feed_forward(self.sublayer[2].norm(x))


class Stack(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = nn.ModuleList([layer() for _ in range(n)])
        self.norm = AtLn(768)


class Lut(nn.Module):
    def __init__(self):
        super().__init__()
        self.lut = nn.Embedding(51865, 768)


class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 51865)


def positions(n, d=768):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class S2s(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_embed = nn.ModuleList([nn.Linear(512, 768)])
        self.tgt_embed = nn.ModuleList([Lut()])
        self.encoder = Stack(EncLayer, 12)
        self.decoder = Stack(DecLayer, 12)
        self.generator = Gen()
        self.register_buffer("pe", positions(5000), persistent=False)

    def encode(self, feats):  # (T, 512) → (1, T, 768)
        x = self.src_embed[0](feats.unsqueeze(0)) + self.pe[: feats.shape[0]]
        for l in self.encoder.layers:
            x = l(x)
        return self.encoder.norm(x)

    def decode(self, mem, ys):  # ys (1, L) → logits (L, V)
        n = ys.shape[1]
        x = self.tgt_embed[0].lut(ys) * math.sqrt(768) + self.pe[:n]
        mask = torch.tril(torch.ones(n, n, dtype=torch.long)).unsqueeze(0).unsqueeze(0)
        for l in self.decoder.layers:
            x = l(x, mem, mask)
        return self.generator.proj(self.decoder.norm(x))[0]


class MultiVsr(nn.Module):
    def __init__(self):
        super().__init__()
        self.vtp = Vtp()
        self.s2s = S2s()

    def greedy(self, faces, max_len=80):
        mem = self.s2s.encode(self.vtp(faces))
        ys = list(PREFIX[:2])
        for _ in range(max_len):
            nxt = int(self.s2s.decode(mem, torch.tensor([ys]))[-1].argmax())
            ys.append(nxt)
            if nxt == EOT:
                break
        return [t for t in ys if t < EOT]


# ---------------------------------------------------------------------------- data

def norm_text(t):
    t = t.lower().replace("—", " ").replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def read_npz(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            out[n[:-4]] = np.load(BytesIO(z.read(n)), allow_pickle=False)
    return out


def load_clips(home):
    clips = []
    for p in sorted(glob.glob(os.path.join(home, "clips", "onboarding-ru", "*.npz"))):
        d = read_npz(p)
        if "faces" not in d:
            continue
        text = str(d["text"].reshape(-1)[0])
        clips.append((torch.from_numpy(d["faces"]).float().div(255).permute(0, 3, 1, 2).contiguous(), norm_text(text)))
    return clips


def wer(hyp, ref):
    h, r = hyp.split(), ref.split()
    d = list(range(len(r) + 1))
    for i, x in enumerate(h, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(r, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1], len(r)


def evaluate(model, tok, clips):
    e = n = 0
    for faces, text in clips:
        hyp = tok.decode(model.greedy(faces))
        a, b = wer(norm_text(hyp), text)
        e, n = e + a, n + b
    return e / max(n, 1)


def main():
    home = os.environ.get("LIPFLOW_HOME") or os.path.expanduser("~/Library/Application Support/Lipflow")
    mdir = os.environ.get("LIPFLOW_MULTIVSR") or os.path.join(home, "base-models", "multivsr")
    parts = os.environ.get("LIPFLOW_TRAIN_PARTS", "vtp")
    epochs = int(os.environ.get("LIPFLOW_TRAIN_EPOCHS", "6"))
    lr = float(os.environ.get("LIPFLOW_TRAIN_LR", "2e-5"))
    report(2, "Loading the Russian model…")
    model = MultiVsr().eval()
    base = load_file(os.path.join(mdir, "multivsr.safetensors"))
    missing, unexpected = model.load_state_dict(base, strict=False)
    if [m for m in missing if not m.endswith("num_batches_tracked")] or unexpected:
        sys.exit(f"weights don't match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    tok = Tokenizer(models.BPE.from_file(os.path.join(mdir, "vocab.json"), os.path.join(mdir, "merges.txt")))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    clips = load_clips(home)
    if len(clips) < 8:
        print("RESULT " + json.dumps({"before": 0.0, "after": None, "kept": False, "clips": len(clips), "note": f"Only {len(clips)} Russian practice clips; record at least 8."}), flush=True)
        return
    rng = random.Random(0)
    order = list(range(len(clips)))
    rng.shuffle(order)
    n_val = max(2, len(clips) // 4)
    val = [clips[i] for i in order[:n_val]]
    train = [clips[i] for i in order[n_val:]]
    report(5, f"Checking the base model on {len(val)} held-out clips…")
    t0 = time.time()
    before = evaluate(model, tok, val)
    report(10, f"Base model: {100 * (1 - before):.0f}% of held-out words right ({time.time() - t0:.0f}s)")

    prefixes = {"vtp": ["vtp."], "vtp+enc": ["vtp.", "s2s.src_embed", "s2s.encoder"], "all": ["vtp.", "s2s."]}[parts]
    params = []
    for name, p in model.named_parameters():
        p.requires_grad_(any(name.startswith(x) for x in prefixes) and not name.startswith("s2s.tgt_embed") and not name.startswith("s2s.generator"))
        if p.requires_grad:
            params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ys_all = [torch.tensor([PREFIX + tok.encode(text).ids + [EOT]]) for _, text in train]
    steps = epochs * len(train)
    step = 0
    with torch.enable_grad():
        for ep in range(epochs):
            idx = list(range(len(train)))
            rng.shuffle(idx)
            total = 0.0
            for i in idx:
                faces, _ = train[i]
                ys = ys_all[i]
                logits = model.s2s.decode(model.s2s.encode(model.vtp(faces)), ys[:, :-1])
                # loss on the text tokens: skip the predictions of <|ru|>, <|transcribe|>, <|notimestamps|>
                loss = F.cross_entropy(logits[3:], ys[0, 4:], label_smoothing=0.1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total += loss.item()
                step += 1
                report(10 + 80 * step / steps, f"Training on your face: round {ep + 1} of {epochs}")
            print(f"epoch {ep + 1}: loss {total / len(train):.3f}", file=sys.stderr, flush=True)
    report(92, "Checking the trained model…")
    after = evaluate(model, tok, val)
    kept = after < before
    out = os.path.join(home, "models", "multivsr_face.safetensors")
    if kept and os.environ.get("LIPFLOW_TRAIN_DRY") != "1":
        os.makedirs(os.path.dirname(out), exist_ok=True)
        trained = {n: p.detach().contiguous() for n, p in model.named_parameters() if p.requires_grad}
        save_file(trained, out)
    note = f"Trained {parts} on {len(train)} clips for {epochs} rounds."
    print("RESULT " + json.dumps({"before": before, "after": after, "kept": kept, "clips": len(clips), "note": note}), flush=True)


if __name__ == "__main__":
    main()
