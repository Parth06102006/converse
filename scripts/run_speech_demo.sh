#!/usr/bin/env bash
# ==============================================================================
# Converse: Speech-to-Sign Real-Time GUI Demonstration Runner
# Subsystem Author: Ashwani
#
# Launches the broadcast-grade OpenCV GUI window with:
#   1. 3D Procedural ASL Skeletal Avatar Canvas with 60 FPS IK kinematics
#   2. Real-time Audio Waveform Oscilloscope & Silero VAD v5 status
#   3. Faster-Whisper INT8 streaming ASR speech transcription
#   4. Dual-Stream Subtitles (English text -> ASL Gloss chips timeline)
#   5. Dynamic Non-Manual Markers (furrowed/raised eyebrows, head tilts/shakes)
#   6. 3D Spatial Discourse Loci Radar
#   7. Interactive Scenarios [1-8], Microphone Push-to-Talk [SPACE], and Guide [H]
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

# Silence Qt font warnings and MediaPipe glog before execution
export QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false;QFontDatabase.warning=false;QFontDatabase.debug=false"
export GLOG_minloglevel="2"
export OPENCV_LOG_LEVEL="ERROR"

# Ensure Qt font directory exists if .venv is present
mkdir -p "$REPO_ROOT/ml/asr/.venv/lib/python3.12/site-packages/cv2/qt/fonts" 2>/dev/null || true

# Preflight: leave the user's default source alone (headset or mic) and only
# ensure the Internal Mic Boost does not saturate when the onboard mic is used.
if command -v amixer &>/dev/null; then
    # Set Internal Mic Boost to level 1 (+10dB) to prevent saturation clipping
    amixer -c 2 set 'Internal Mic Boost',0 1 2>/dev/null || true
fi

echo "=================================================================="
echo " Converse: Launching Speech-to-Sign Real-Time Demonstration GUI"
echo "=================================================================="
echo "Press 'q' or ESC inside the OpenCV window to exit cleanly."
echo "------------------------------------------------------------------"

uv run --project "$REPO_ROOT/ml/asr" python "$REPO_ROOT/scripts/demo_speech_to_sign_gui.py" "$@"
