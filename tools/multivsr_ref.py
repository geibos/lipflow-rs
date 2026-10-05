"""MultiVSR (github.com/Sindhu-Hegde/multivsr) reference for the Rust port.

1. Exports the weights the model actually uses into one safetensors file (the checkpoints also
   carry optimizer state and an unused English lip reader): `vtp.*` = the visual front end,
   `s2s.*` = the 12+12-layer Transformer.
2. Dumps intermediate tensors for one face track (.avi from their preprocess/run_pipeline.py):
   faces, VTP features, encoder output, first-step logits, greedy and beam-search tokens.

Run with the eval env (ref/multivsr/env):
  ref/multivsr/env/.venv/bin/python tools/multivsr_ref.py ref/multivsr ref/multivsr/work/pycrop/silent/00000.avi
"""
import os
import sys
import types

root, avi = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "code"))
sys.argv = [sys.argv[0], "--device", "cpu", "--fp16", "False"]
sys.modules["decord"] = types.SimpleNamespace(VideoReader=None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import inference as inf  # noqa: E402
from dataloader import subsequent_mask  # noqa: E402
from models import build_model, build_visual_encoder  # noqa: E402
from search import beam_search  # noqa: E402

torch.set_grad_enabled(False)

model = build_model().eval()
s = torch.load(os.path.join(root, "model.pth"), map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict({k.removeprefix("module."): v for k, v in s.items()})
vtp = build_visual_encoder().eval()
s = torch.load(os.path.join(root, "feature_extractor.pth"), map_location="cpu", weights_only=False)["state_dict"]
vtp.load_state_dict({k.replace("module.face_encoder.", ""): v for k, v in s.items() if "face_encoder" in k})

weights_path = os.path.join(root, "multivsr.safetensors")
if not os.path.exists(weights_path):
    out = {}
    for k, v in vtp.state_dict().items():
        if not k.endswith("num_batches_tracked"):
            out["vtp." + k] = v.float().contiguous()
    for k, v in model.state_dict().items():
        if not k.endswith(".pe"):
            out["s2s." + k] = v.float().contiguous()
    save_file(out, weights_path)
    print("wrote", weights_path, len(out), "tensors")

# faces exactly as their inference reads them: RGB, 160x160, centre 96x96, [0, 1]
cap = cv2.VideoCapture(avi)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (160, 160), interpolation=cv2.INTER_LINEAR))
x = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0).unsqueeze(0).permute(0, 4, 1, 2, 3)[:, :, :, 32:128, 32:128].contiguous()

feats = vtp(x)
cnn = vtp.feat_extrator(x)
mask = torch.ones(1, 1, feats.shape[1]).long()
mem, _ = model.encode(feats, mask)
sos = [50258, inf.tokenizer.encode("<|ru|>")[0]]
ys = torch.tensor([sos])
logits0 = model.decode(mem, mask, ys, subsequent_mask(ys.size(1)).long())[-1]

greedy = list(sos)
for _ in range(100):
    ys = torch.tensor([greedy])
    nxt = int(model.decode(mem, mask, ys, subsequent_mask(ys.size(1)).long())[-1].argmax())
    greedy.append(nxt)
    if nxt == inf.tokenizer.eot:
        break

beams = {}
for size in (5, 20):
    outs, scores = beam_search(model=model, size=size, bos_index=sos, eos_index=inf.tokenizer.eot, pad_index=0,
                               encoder_output=mem, src_mask=mask, max_output_length=100, n_best=size)
    beams[size] = (outs[0][0].tolist(), scores[0][0])
    print(f"beam {size}: {inf.tokenizer.decode(beams[size][0])!r} score {scores[0][0]:.4f}")
print("greedy:", repr(inf.tokenizer.decode(greedy[len(sos):])))

name = os.path.splitext(os.path.basename(os.path.dirname(avi)))[0]
dump = {
    "faces": x[0],  # (3, T, 96, 96)
    "cnn": cnn[0],  # (128, T, 24, 24)
    "feats": feats[0],  # (T, 512)
    "memory": mem[0],  # (T, 768)
    "logits0": logits0,  # (vocab,)
    "sos": torch.tensor(sos),
    "greedy": torch.tensor(greedy),
    "beam5": torch.tensor(beams[5][0]),
    "beam5_score": torch.tensor([beams[5][1]]),
    "beam20": torch.tensor(beams[20][0]),
    "beam20_score": torch.tensor([beams[20][1]]),
}
path = os.path.join(root, f"ref_{name}.safetensors")
save_file({k: v.contiguous() for k, v in dump.items()}, path)
print("wrote", path)
