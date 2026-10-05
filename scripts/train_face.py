"""Training on your face and phrasing stays in Python (PyTorch): the Rust app runs this and then
loads the weights it writes (<Lipflow home>/models/vsr_face.pth, lm_phrasing.pth).

Run inside the Python checkout's environment:  uv run --project <lipflow checkout> python scripts/train_face.py
Prints "PROGRESS <pct> <text>" lines and a final "RESULT <json>" line.
"""
import json
import os
import sys

checkout = os.environ.get("LIPFLOW_PY_CHECKOUT")
if checkout:
    sys.path.insert(0, checkout)

from lipflow.dictation import train_on_face  # noqa: E402


def report(pct, text):
    print(f"PROGRESS {pct:.0f} {text}", flush=True)


r = train_on_face(int(os.environ.get("LIPFLOW_BEAM", "4")), report)
print("RESULT " + json.dumps({"before": r["before"], "after": r["after"], "kept": r["kept"],
                              "clips": r["clips"], "note": r["note"]}), flush=True)
