# Lipflow (Rust)

Silent dictation by lip reading — the Rust port of [Lipflow](lipflow/README.md) (the Python
original lives in `lipflow/` and is the reference for every number below). Hold a key, silently
mouth what you want to say, let go, and the text appears at your cursor. macOS only for now.

Russian (the default when its model is installed) reads with
[MultiVSR](https://github.com/Sindhu-Hegde/multivsr); English with Auto-AVSR. Switch in
Settings → "Language I dictate in".

## Setup

```sh
scripts/setup.sh            # models (~2.4 GB) into ~/Library/Application Support/Lipflow/base-models,
                            # then builds /Applications/Lipflow.app (add --no-app to skip)
open /Applications/Lipflow.app
```

Needs Apple Silicon and the Rust toolchain. No Python. On first launch the app asks for Camera,
Input Monitoring and Accessibility, then walks you through setup (your words, 24 practice
sentences, training on your face). Your data folder is the same one the Python app used, so
settings, phrases, words and clips carry over.

## Commands

| | |
|---|---|
| `lipflow` / `lipflow run [--key K] [--beam N] [--cleanup auto\|claude\|local\|ollama\|basic] [--camera C] [--copy-only] [--no-preview]` | menu-bar app |
| `lipflow file VIDEO [--start S] [--end E] [--beam N]` | lip-read a video file |
| `lipflow doctor` | check models and permissions |
| `lipflow import-wispr [--from-text FILE]` | learn your phrasing from Wispr Flow history |
| `lipflow onboard` | open setup |
| `lipflow train` | train on your practice clips (CPU) |
| `lipflow ru-file VIDEO [--start S] [--end E]` | Russian lip reading of a video file |
| `lipflow convert-ru [DIR]` | MultiVSR release checkpoints → `multivsr.safetensors` (setup runs it) |
| `lipflow selftest VIDEO --start S --end E` | the whole app flow with a video standing in for the camera |
| `lipflow bench` | full pipeline WER on the sample clips |

Logs: `~/Library/Logs/Lipflow.log` when started as the app.

## Layout

- `crates/vsr` — the Auto-AVSR model on candle: Conformer encoder, decoder + CTC, Transformer LM,
  joint CTC/attention beam search, hand-written Metal kernels for the conv front end,
  SentencePiece, and CPU fine-tuning; and MultiVSR (`multivsr.rs`: VTP front end, 12+12
  Transformer, Joey NMT beam search, Whisper tokens).
- Training on your face stays in PyTorch on the CPU: `scripts/train_face.py` (English, via the
  Python checkout) and `scripts/train_face_ru.py` (Russian, self-contained, run by `uv`).
- `crates/face` — face tracking without MediaPipe: a small TFLite interpreter for MediaPipe's face
  models, the FaceLandmarker tracking logic, and the OpenCV-exact mouth-crop alignment.
- `crates/llm` — the on-device cleanup model (Qwen3-0.6B GGUF on candle).
- `crates/app` — the app: AppKit menu bar, HUD, settings, setup, AVFoundation camera, hotkey tap,
  paste, Accessibility context, text cleanup, data files.

`PROGRESS.md` records each step, its measurements against the Python app, and what is not done.
