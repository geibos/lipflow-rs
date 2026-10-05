"""Reference for the Rust whisper mode: audio of bench clip 1, the fused AV encoder output and
the AV beam search result. Run from the lipflow checkout: uv run python ../tools/av_ref.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from lipflow.av import AVReader, load_audio  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "ref")
r = AVReader(beam_size=4, personal=False, device="cpu")
rois = np.load(os.path.join(OUT, "bench", "001.npy"))
wave = load_audio("samples/2016-03-12.mov", 20.4, 28.1)
wave = wave[: rois.shape[0] * 640]
np.save(os.path.join(OUT, "av_wave_001.npy"), wave.astype(np.float32))
enc = r.encode_av(rois, wave)
np.save(os.path.join(OUT, "av_enc_001.npy"), enc.numpy().astype(np.float32))
print(enc.shape, r.beam_search(enc, nbest=1))
