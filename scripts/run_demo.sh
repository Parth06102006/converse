#!/usr/bin/env bash
# ==============================================================================
# Converse ASL Sign-to-Speech End-to-End Demonstration Runner
#
# Orchestrates all dependencies:
#   1. Local Kokoro Neural TTS Docker daemon (port 8880)
#   2. MediaPipe 3D Landmark & Spatiotemporal Vision Pipeline (Python / uv)
#   3. Local audio playback (ALSA / aplay)
#
# Gracefully intercepts SIGINT, SIGTERM, and EXIT to stop all subprocesses,
# terminate the Kokoro container, and remove temporary audio artifacts.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ML_DIR="$REPO_ROOT/ml/asl-vision"

KOKORO_IMAGE="ghcr.io/lucasjinreal/kokoros:main"
KOKORO_CONTAINER="kokoro-server"
KOKORO_PORT=8880
KOKORO_URL="http://127.0.0.1:${KOKORO_PORT}/v1/audio/speech"

KEEP_KOKORO=0
PASSTHROUGH_ARGS=()
DEMO_PID=""
CLEANED_UP=0

# ------------------------------------------------------------------------------
# Usage & Help
# ------------------------------------------------------------------------------
show_help() {
    cat << 'EOF'
Usage: ./scripts/run_demo.sh [OPTIONS]

Orchestrates and launches the complete Converse ASL Sign-to-Speech pipeline.
Manages the local Kokoro neural TTS Docker service, executes the live webcam
perception loop, and shuts down all services gracefully upon exit.

Options:
  --keep-kokoro          Keep the Kokoro Docker container running after exit.
  --camera <ID>          Camera device index (default: 0).
  --voice <VOICE>        Kokoro voice style (default: af_sky).
                         Options: af_sky, af_sarah, am_adam, bf_emma.
  --no-tts               Disable speech synthesis entirely.
  --checkpoint <PATH>    Path to trained PyTorch ST-GCN .pt checkpoint.
  --min-confidence <NUM> Minimum sign detection confidence (default: 0.40).
  -h, --help             Display this help message and exit.

Keyboard Controls inside OpenCV Window:
  [1] Hello! (Single word idiom)
  [2] Hello, nice to meet you. (Continuous greeting)
  [3] Thank you for your help. (Expression)
  [4] What is your name? (Wh-question inversion)
  [5] I went to the store. (Tense shift & SVO reconstruction)
  [S] Force finalize and speak active gloss buffer
  [R] Reset buffer and clear subtitle card
  [V] Cycle through Kokoro neural voice styles
  [T] Toggle TTS speech playback on/off
  [Q] or [ESC] Graceful exit
EOF
}

# Parse runner-specific arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --keep-kokoro)
            KEEP_KOKORO=1
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Graceful Cleanup Handler
# ------------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [ "$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    echo ""
    echo "========================================================"
    echo " Converse ASL: Initiating Graceful Shutdown Routine"
    echo "========================================================"

    # 1. Terminate Python demo process if still running
    if [ -n "$DEMO_PID" ] && kill -0 "$DEMO_PID" 2>/dev/null; then
        echo "[1/3] Stopping ASL Vision demo process (PID: $DEMO_PID)..."
        kill -SIGINT "$DEMO_PID" 2>/dev/null || true
        for _ in {1..10}; do
            if ! kill -0 "$DEMO_PID" 2>/dev/null; then
                break
            fi
            sleep 0.2
        done
        if kill -0 "$DEMO_PID" 2>/dev/null; then
            kill -SIGKILL "$DEMO_PID" 2>/dev/null || true
        fi
        echo "      ASL Vision process terminated."
    else
        echo "[1/3] ASL Vision process already terminated."
    fi

    # 2. Stop Kokoro Docker container unless --keep-kokoro was passed
    if [ "$KEEP_KOKORO" -eq 0 ]; then
        echo "[2/3] Stopping Kokoro TTS Docker container ($KOKORO_CONTAINER)..."
        if docker inspect -f '{{.State.Running}}' "$KOKORO_CONTAINER" 2>/dev/null | grep -q "true"; then
            docker stop -t 3 "$KOKORO_CONTAINER" >/dev/null 2>&1 || true
            echo "      Kokoro container stopped."
        else
            echo "      Kokoro container already stopped."
        fi
    else
        echo "[2/3] Preserving Kokoro container (--keep-kokoro specified)."
    fi

    # 3. Clean temporary stream audio files
    echo "[3/3] Cleaning temporary audio cache..."
    rm -f /tmp/converse_kokoro_stream*.wav 2>/dev/null || true

    echo "========================================================"
    echo " Graceful shutdown complete. All resources released."
    echo "========================================================"
    exit "$exit_code"
}

trap cleanup EXIT INT TERM HUP

# ------------------------------------------------------------------------------
# Pre-Flight Environment Checks
# ------------------------------------------------------------------------------
echo "========================================================"
echo " Converse: Real-Time ASL-to-Speech Demonstration Runner"
echo "========================================================"

# Check uv binary
if ! command -v uv &>/dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo "Error: 'uv' package manager not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        exit 1
    fi
fi

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "Warning: 'docker' not found. Speech synthesis will fall back to system TTS." >&2
    DOCKER_AVAILABLE=0
else
    if ! docker info >/dev/null 2>&1; then
        echo "Warning: Docker daemon is not running. Kokoro container cannot be launched." >&2
        DOCKER_AVAILABLE=0
    else
        DOCKER_AVAILABLE=1
    fi
fi

# Check ALSA audio playback
if ! command -v aplay &>/dev/null && ! command -v ffplay &>/dev/null; then
    echo "Warning: Neither 'aplay' nor 'ffplay' found. Real-time audio playback may not sound." >&2
fi

# Check camera device
if compgen -G "/dev/video*" > /dev/null; then
    CAM_FOUND=1
else
    CAM_FOUND=0
    echo "Notice: No physical /dev/video* devices detected."
    echo "        You can still test the pipeline using sample injection hotkeys [1-5]."
fi

# ------------------------------------------------------------------------------
# Kokoro Neural TTS Service Verification & Startup
# ------------------------------------------------------------------------------
is_kokoro_responsive() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 -X POST "$KOKORO_URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"kokoro","input":"test","voice":"af_sky","response_format":"wav"}' \
        2>/dev/null || echo "000")
    [ "$code" = "200" ]
}

if [ "$DOCKER_AVAILABLE" -eq 1 ]; then
    echo "Checking Kokoro TTS service at $KOKORO_URL..."
    if is_kokoro_responsive; then
        echo "Kokoro TTS service is already active and responsive."
    else
        CONTAINER_RUNNING=$(docker inspect -f '{{.State.Running}}' "$KOKORO_CONTAINER" 2>/dev/null || echo "not_found")

        if [ "$CONTAINER_RUNNING" = "true" ]; then
            echo "Kokoro container exists and is running. Waiting for readiness..."
        elif [ "$CONTAINER_RUNNING" = "false" ]; then
            echo "Starting existing Kokoro container ($KOKORO_CONTAINER)..."
            docker start "$KOKORO_CONTAINER" >/dev/null
        else
            echo "Launching local Kokoro neural TTS container on port $KOKORO_PORT..."
            docker run -d \
                --name "$KOKORO_CONTAINER" \
                -p "${KOKORO_PORT}:3000" \
                "$KOKORO_IMAGE" \
                openai >/dev/null
        fi

        # Wait for service readiness with timeout
        echo -n "Waiting for Kokoro neural TTS endpoint to warm up"
        READY=0
        for _ in {1..30}; do
            if is_kokoro_responsive; then
                READY=1
                break
            fi
            echo -n "."
            sleep 0.5
        done
        echo ""

        if [ "$READY" -eq 1 ]; then
            echo "Kokoro neural TTS service is ready (OpenAI-compatible HTTP API on port $KOKORO_PORT)."
        else
            echo "Warning: Kokoro server startup timed out. Demo will attempt CLI or system TTS fallback."
        fi
    fi
fi

# ------------------------------------------------------------------------------
# Ensure MediaPipe Task Asset
# ------------------------------------------------------------------------------
TASK_ASSET="$ML_DIR/models/holistic_landmarker.task"
if [ ! -f "$TASK_ASSET" ]; then
    echo "Downloading official MediaPipe Holistic Landmarker bundle..."
    mkdir -p "$ML_DIR/models"
    curl -sSL "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" \
        -o "$TASK_ASSET"
    echo "MediaPipe model asset downloaded successfully."
fi

GESTURE_ASSET="$ML_DIR/models/gesture_recognizer.task"
if [ ! -f "$GESTURE_ASSET" ]; then
    echo "Downloading official MediaPipe Gesture Recognizer bundle..."
    mkdir -p "$ML_DIR/models"
    curl -sSL "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" \
        -o "$GESTURE_ASSET"
    echo "Gesture Recognizer model asset downloaded successfully."
fi

# ------------------------------------------------------------------------------
# Launch ASL Vision & Speech Pipeline
# ------------------------------------------------------------------------------
echo "Starting Converse ASL Vision & Speech pipeline..."
echo "Press 'q' or ESC in the OpenCV window to exit cleanly."
echo "--------------------------------------------------------"

cd "$ML_DIR"

if [ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]; then
    uv run python scripts/demo_webcam.py "${PASSTHROUGH_ARGS[@]}" &
else
    uv run python scripts/demo_webcam.py &
fi

DEMO_PID=$!
wait "$DEMO_PID"
