# Lipflow (Rust)

> **Status: stopped (October 2026), archived.** The port works and is faster than the
> original, but dictation in Russian, which is what it was for, isn't accurate enough to use.
> See [Why we stopped](#why-we-stopped).

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

## Why we stopped

The goal was performance and reliability, not a rewrite for its own sake. That part worked
(numbers are measured against the Python original, details in `PROGRESS.md`):

- The lip-reading model gives the same tokens as the PyTorch reference. It runs 4.2× faster,
  and a video goes to text in 8.8 s instead of 20.7 s. Russian (MultiVSR): 3.8 s instead of
  29 s for 16 s of video, and about 1.1 s from releasing the key to typed text.
- No Python, MediaPipe, OpenCV or PyTorch at runtime. Face tracking and crops match MediaPipe
  and OpenCV, and the on-device cleanup model is faster than the MLX one.

What stopped it was accuracy in the language the user actually dictates in:

1. **Auto-AVSR, the original's model, is English-only.** The user needed Russian, so we moved
   to MultiVSR (Oxford VGG, 2025), the best open lip-reading model with Russian we could find:
   39.5% word error rate on its own test set.
2. **On the user's own silent webcam dictation the base model is too far off.** On test videos
   of five everyday phrases it got about half of the words wrong (52% WER), against 70% for the
   original's face crops. Live, short phrases failed outright: "это тест" came out as
   "я не верю". This is the model's ceiling, not a porting error. The Rust port reproduces the
   reference token for token, the face crops were checked, and mirroring the image doesn't
   help. Silent mouthing with a laptop camera looking down at you is far from the
   read-aloud, filmed-face data these models are trained on.
3. **Fine-tuning on your own face is the remaining lever, and it is unproven.** The pipeline is
   built and tested end to end: `scripts/train_face_ru.py`, CPU only, keeps the result only if
   it beats the base model on held-out clips. For English, the same kind of fine-tuning moved
   held-out WER only from 106% to 96%. We stopped before recording enough Russian practice
   clips to measure it.
4. **Training stays in PyTorch.** Porting it to Rust was slower on the CPU than PyTorch, and
   training on the GPU (Metal) rebooted the machine, so it was not worth pursuing.

Without a usable accuracy level, the speed gains don't matter for a dictation tool, so the
project was closed rather than polished further.

### Not done

- Windows. The Python original has a Windows version; this port is macOS only.
- Whisper mode (lips plus a soft whisper) for Russian: the audio-visual model is English-only.
- A stable code-signing identity. The app is signed ad hoc, so every rebuild resets the macOS
  permissions; Settings → Check permissions walks you through them again.
- The model weights aren't in this repository: `scripts/setup.sh` downloads them from their
  publishers (MultiVSR's README states no license for its weights).
