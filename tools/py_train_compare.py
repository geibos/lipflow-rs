"""Python fine-tuning on the same split as `lipflow train` (Rust), CPU only, for comparison.
Run from the lipflow checkout: uv run python ../tools/py_train_compare.py <clips dir>"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from lipflow.bench import wer  # noqa: E402
from lipflow.train_vsr import finetune  # noqa: E402
from lipflow.vsr import LipReader  # noqa: E402

M = (1 << 64) - 1


class SplitMix:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        return z ^ (z >> 31)

    def shuffle(self, v):
        for i in range(len(v) - 1, 0, -1):
            j = self.next() % (i + 1)
            v[i], v[j] = v[j], v[i]


clips = [dict(np.load(p, allow_pickle=True)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]
clips = [{"rois": c["rois"], "text": str(c["text"])} for c in clips]
SplitMix(1).shuffle(clips)
test, train = clips[:6], clips[6:]
base = LipReader(device="cpu", beam_size=4, personal=False)


def score(r):
    e = n = 0
    for c in test:
        a, b = wer(r.beam_search(r.encode(c["rois"])), c["text"])
        e, n = e + a, n + b
    return e / n


before = score(base)
t0 = time.time()
finetune(train, reader=base, log=print)
dt = time.time() - t0
after = score(base)
print(f"python held-out WER {before:.1%} -> {after:.1%}, finetune {dt:.0f}s on CPU")
