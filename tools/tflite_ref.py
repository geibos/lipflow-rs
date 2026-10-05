"""Reference outputs of the face models from LiteRT, for the Rust TFLite interpreter.

uv run --python 3.12 --with ai-edge-litert --with numpy python tools/tflite_ref.py <dir with .tflite>
"""
import os
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

src = sys.argv[1]
out = os.path.join(os.path.dirname(__file__), "..", "ref", "tflite")
for name in ("face_detector", "face_landmarks_detector"):
    it = Interpreter(model_path=os.path.join(src, name + ".tflite"))
    it.allocate_tensors()
    d = it.get_input_details()[0]
    shape = d["shape"]
    n = int(np.prod(shape))
    x = (np.sin(np.arange(n, dtype=np.float64) * 0.001) * 0.9).astype(np.float32).reshape(shape)
    it.set_tensor(d["index"], x)
    it.invoke()
    x.tofile(os.path.join(out, f"{name}.in.bin"))
    for i, od in enumerate(it.get_output_details()):
        y = it.get_tensor(od["index"]).astype(np.float32)
        y.tofile(os.path.join(out, f"{name}.out{i}.bin"))
        print(name, i, od["name"], y.shape, float(np.abs(y).max()))
