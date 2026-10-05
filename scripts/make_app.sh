#!/usr/bin/env bash
# Build Lipflow.app around the Rust binary (no Python, no launcher).
# Why a bundle: macOS grants Camera / Input Monitoring / Accessibility per app; as an app Lipflow
# gets its own "Lipflow" entries, starts from Spotlight and can be a Login Item.
#   scripts/make_app.sh [DEST_DIR]      (default /Applications)
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-/Applications}"
APP="$DEST/Lipflow.app"

cargo build --release -p lipflow
BIN="target/release/lipflow"
[ -x "$BIN" ] || { echo "build produced no $BIN" >&2; exit 1; }

case "$APP" in */Lipflow.app) [ -d "$APP" ] && rm -rf "$APP" ;; esac
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Lipflow"
cp crates/app/assets/Lipflow.icns "$APP/Contents/Resources/Lipflow.icns"
cp scripts/train_face.py "$APP/Contents/Resources/train_face.py"
cp scripts/train_face_ru.py "$APP/Contents/Resources/train_face_ru.py"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Lipflow</string>
  <key>CFBundleDisplayName</key><string>Lipflow</string>
  <key>CFBundleIdentifier</key><string>app.lipflow.Lipflow</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Lipflow</string>
  <key>CFBundleIconFile</key><string>Lipflow</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSCameraUsageDescription</key><string>Lipflow reads your lips from the camera to type what you mouth. Video never leaves this Mac.</string>
  <key>NSMicrophoneUsageDescription</key><string>Whisper mode listens to a soft whisper while you hold the key, to read your lips more accurately. Audio never leaves this Mac.</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
# Ad-hoc signature so Accessibility / Input Monitoring bind to this app.
codesign --force --sign - --identifier app.lipflow.Lipflow "$APP"
touch "$APP"
echo "$APP"
