#!/usr/bin/env bash
# Download the models (~2.4 GB; ~5.6 GB transferred the first time) into the Lipflow data folder and build Lipflow.app.
#   scripts/setup.sh [--no-app] [--samples]
set -euo pipefail
cd "$(dirname "$0")/.."
HOME_DIR="${LIPFLOW_HOME:-$HOME/Library/Application Support/Lipflow}"
M="$HOME_DIR/base-models"

get() {  # url dest
  [ -s "$2" ] && { echo "✓ $2"; return; }
  mkdir -p "$(dirname "$2")"
  echo "↓ $2"
  curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

HF=https://huggingface.co
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json "$M/vsr/model.json"
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth  "$M/vsr/model.pth"
get $HF/Amanvir/lm_en_subword/resolve/main/model.json  "$M/lm/model.json"
get $HF/Amanvir/lm_en_subword/resolve/main/model.pth   "$M/lm/model.pth"
get https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model "$M/lm/unigram5000.model"
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task "$M/face_landmarker.task"

# Russian (MultiVSR, github.com/Sindhu-Hegde/multivsr): 4.4 GB of checkpoints with optimizer
# state, converted to the 1.2 GB the app loads by `lipflow convert-ru` (no Python).
R="$M/multivsr"
if [ ! -s "$R/multivsr.safetensors" ]; then
  get https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/feature_extractor.pth "$R/feature_extractor.pth"
  get https://www.robots.ox.ac.uk/~vgg/research/multivsr/model.pth "$R/model.pth"
  cargo build --release -p lipflow
  target/release/lipflow convert-ru "$R"
fi
MV=https://raw.githubusercontent.com/Sindhu-Hegde/multivsr/master/checkpoints/multilingual
get $MV/vocab.json "$R/vocab.json"
get $MV/merges.txt "$R/merges.txt"

for arg in "$@"; do
  if [ "$arg" = "--samples" ]; then
    C=https://upload.wikimedia.org/wikipedia/commons/transcoded
    get "$C/c/ce/2016-03-12_President_Obama%27s_Weekly_Address.webm/2016-03-12_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" samples/2016-03-12.mov
  fi
done

if [[ " $* " != *" --no-app "* ]]; then
  scripts/make_app.sh /Applications
fi
echo
echo "Done. Open Lipflow from Spotlight (or: open /Applications/Lipflow.app)."
