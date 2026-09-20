#!/usr/bin/env bash
# ==============================================================================
# Converse: Speech-to-Sign Pipeline Interactive Demonstration Runner
# Subsystem Author: Ashwani
#
# Launches the interactive demonstration for:
#   1. Neural Voice Activity Detection (Silero VAD v5 ONNX)
#   2. Streaming Automatic Speech Recognition (Faster-Whisper INT8 / AWS Transcribe)
#   3. English-to-ASL Grammar Transformation (Topic-Comment / TSOV, Wh-movement)
#   4. Non-Manual Marker Synthesis (Eyebrows, Head, Mouth)
#   5. Out-Of-Vocabulary Fingerspelling Decomposition
#   6. 3D Spatial Locus Referent Tracking
#   7. Dynamic Co-articulation Timing Engine & SignRepresentation JSON
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure uv is in PATH
if ! command -v uv &>/dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

echo "=================================================================="
echo " Starting Converse Speech-to-Sign Demonstrator (Ashwani's Scope)"
echo "=================================================================="

# Run through the ASR virtual environment which contains faster-whisper, ctranslate2, soundfile, onnxruntime
uv run --project "$REPO_ROOT/ml/asr" python "$REPO_ROOT/scripts/demo_speech_to_sign.py" "$@"
