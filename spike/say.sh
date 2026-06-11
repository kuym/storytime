#!/bin/bash
# Speak text through the ONNX->MLX CPU interpreter and play it.
# Usage: spike/say.sh "text to speak" [voice] [speed]
#   voice defaults to af_heart; speed to 1.0
# Requires: espeak-ng, and the lowered artifacts in spike/art/ (run spike/lower.py once).
set -e
ROOT="/Users/kuy/Projects/storytime"
text="${1:?usage: say.sh \"text\" [voice] [speed]}"
voice="${2:-af_heart}"
speed="${3:-1.0}"

ipa=$(espeak-ng -q --ipa=3 -v en-us "$text" | tr '\n' ' ')
out=$(mktemp /tmp/mlx_say_XXXXXX.wav)
cd "$ROOT"
echo "IPA: $ipa"
echo ./spike/interp/target/release/interp --synth "$ipa" "$voice" "$out" "$speed"
./spike/interp/target/release/interp --synth "$ipa" "$voice" "$out" "$speed"
echo "playing $out ..."
afplay "$out" 2>/dev/null || echo "(afplay unavailable; file at $out)"
