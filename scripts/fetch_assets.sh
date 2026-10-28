#!/usr/bin/env bash
# Fetch third-party browser assets into sensor-client/vendor/ (git-ignored: size + licensing).
#   1. @mediapipe/tasks-vision (Apache-2.0) from npm, version pinned below -> JS bundle + WASM
#   2. hand_landmarker.task (Google-hosted model, Apache-2.0) -> served locally so phones need no internet at run time
# Re-run any time; verifies the model hash recorded on first download.
set -euo pipefail
cd "$(dirname "$0")/.."
MEDIAPIPE_VERSION="1.1.1"           # keep in sync with config/camera.yaml (model.mediapipe_tasks_vision_version)
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
V=sensor-client/vendor
mkdir -p "$V/mediapipe" "$V/models"

if [ ! -f "$V/mediapipe/vision_bundle.mjs" ] || ! grep -q "\"$MEDIAPIPE_VERSION\"" "$V/mediapipe/VERSION.json" 2>/dev/null; then
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  (cd "$TMP" && npm init -y >/dev/null && npm install --no-audit --no-fund "@mediapipe/tasks-vision@$MEDIAPIPE_VERSION" >/dev/null)
  P="$TMP/node_modules/@mediapipe/tasks-vision"
  rm -rf "$V/mediapipe"; mkdir -p "$V/mediapipe"
  cp "$P/vision_bundle.mjs" "$P/LICENSE" "$V/mediapipe/" 2>/dev/null || cp "$P/vision_bundle.mjs" "$V/mediapipe/"
  cp -r "$P/wasm" "$V/mediapipe/wasm"
  echo "{\"@mediapipe/tasks-vision\": \"$MEDIAPIPE_VERSION\"}" > "$V/mediapipe/VERSION.json"
  echo "OK  mediapipe tasks-vision $MEDIAPIPE_VERSION installed"
else
  echo "OK  mediapipe tasks-vision $MEDIAPIPE_VERSION already present"
fi

MODEL="$V/models/hand_landmarker.task"
if [ ! -s "$MODEL" ]; then
  if ! curl -fL --retry 2 -o "$MODEL.part" "$MODEL_URL"; then
    rm -f "$MODEL.part"
    echo "ERROR could not download the hand-landmarker model." >&2
    echo "      Download it manually from:  $MODEL_URL" >&2
    echo "      and save it as:             $MODEL" >&2
    exit 2
  fi
  mv "$MODEL.part" "$MODEL"
fi
SUM="$(sha256sum "$MODEL" | cut -d' ' -f1)"
if [ -f "$MODEL.sha256" ]; then
  [ "$(cat "$MODEL.sha256")" = "$SUM" ] || { echo "ERROR model hash changed (expected $(cat "$MODEL.sha256"), got $SUM)" >&2; exit 3; }
else
  echo "$SUM" > "$MODEL.sha256"; echo "NOTE recorded model sha256 on first download: $SUM"
fi
echo "OK  model present ($SUM)"
