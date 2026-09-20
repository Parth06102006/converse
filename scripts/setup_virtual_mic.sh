#!/usr/bin/env bash
# ==============================================================================
# Converse Meeting Virtual Audio Loopback Setup Script
#
# Configures a PipeWire / PulseAudio null sink ('Converse_Virtual_Mic') and
# creates a virtual microphone source ('Converse Microphone') so that any
# video conferencing or WebRTC application (Google Meet, Zoom, Teams, Slack)
# can select 'Converse Microphone' as an input audio device.
#
# Meeting injection workflow:
#   TTS Engine / Player -> Converse_Virtual_Mic (Audio/Sink)
#                       -> Loopback Remap
#                       -> Converse Microphone (Audio/Source)
#                       -> Google Meet / Zoom / WebRTC
#
# Idempotent: Subsequent executions verify existing node configurations
# and avoid duplicate virtual sinks or loopback threads.
# ==============================================================================

set -euo pipefail

SINK_NAME="Converse_Virtual_Mic"
SINK_DESC="Converse Virtual Mic"
SOURCE_NAME="Converse_Microphone"
SOURCE_DESC="Converse Microphone"

PID_DIR="${XDG_RUNTIME_DIR:-/tmp}/converse"
PID_FILE="${PID_DIR}/virtual_mic.pid"

show_help() {
    cat << 'EOF'
Usage: ./scripts/setup_virtual_mic.sh [COMMAND]

Configures the Converse virtual microphone loopback for meeting audio injection.

Commands:
  start     Create and configure virtual sink and microphone (default).
  stop      Tear down and remove virtual audio devices.
  restart   Restart the virtual audio loopback devices.
  status    Check configuration status of virtual audio devices.
  help      Display this help message.
EOF
}

check_running_pactl() {
    if ! command -v pactl >/dev/null 2>&1; then
        return 1
    fi
    local sink_exists source_exists
    sink_exists=$(pactl list short sinks 2>/dev/null | grep -F "${SINK_NAME}" || true)
    source_exists=$(pactl list short sources 2>/dev/null | grep -F "${SOURCE_NAME}" || true)
    if [[ -n "${sink_exists}" && -n "${source_exists}" ]]; then
        return 0
    fi
    return 1
}

check_running_pipewire() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid=$(cat "${PID_FILE}" 2>/dev/null || true)
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    fi

    if command -v wpctl >/dev/null 2>&1; then
        local status_out
        status_out=$(wpctl status 2>/dev/null || true)
        if echo "${status_out}" | grep -q "${SINK_NAME}" && echo "${status_out}" | grep -q "${SOURCE_NAME}"; then
            return 0
        fi
    fi

    if command -v pw-cli >/dev/null 2>&1; then
        local nodes
        nodes=$(pw-cli list-objects Node 2>/dev/null || true)
        if echo "${nodes}" | grep -q "${SINK_NAME}" && echo "${nodes}" | grep -q "${SOURCE_NAME}"; then
            return 0
        fi
    fi

    return 1
}

is_active() {
    if check_running_pactl || check_running_pipewire; then
        return 0
    fi
    return 1
}

start_virtual_mic() {
    if is_active; then
        echo "[Converse Audio] Virtual microphone is already active and configured."
        echo "  - Sink device:   ${SINK_NAME} ('${SINK_DESC}')"
        echo "  - Source device: ${SOURCE_NAME} ('${SOURCE_DESC}')"
        echo "  - Target: Select '${SOURCE_DESC}' in Google Meet, Zoom, or WebRTC."
        return 0
    fi

    echo "[Converse Audio] Initializing Converse Virtual Microphone Loopback..."
    mkdir -p "${PID_DIR}"

    # Method 1: PulseAudio / pactl if installed and operational
    if command -v pactl >/dev/null 2>&1; then
        echo "[Converse Audio] Using PulseAudio/pactl control interface..."
        local sink_id source_id
        sink_id=$(pactl load-module module-null-sink \
            sink_name="${SINK_NAME}" \
            sink_properties=device.description="${SINK_DESC}" 2>/dev/null || true)
        source_id=$(pactl load-module module-remap-source \
            master="${SINK_NAME}.monitor" \
            source_name="${SOURCE_NAME}" \
            source_properties=device.description="${SOURCE_DESC}" 2>/dev/null || true)

        if [[ -n "${sink_id}" && -n "${source_id}" ]]; then
            echo "[Converse Audio] Successfully loaded modules via pactl:"
            echo "  - Null sink module ID: ${sink_id}"
            echo "  - Remap source module ID: ${source_id}"
            echo "  - Microphone input '${SOURCE_DESC}' ready."
            return 0
        else
            echo "[Converse Audio] pactl module loading did not succeed; trying PipeWire native loopback..."
        fi
    fi

    # Method 2: Native PipeWire loopback
    if command -v pw-loopback >/dev/null 2>&1; then
        echo "[Converse Audio] Using native PipeWire loopback engine..."
        pw-loopback \
            --capture-props="media.class=Audio/Sink node.name=${SINK_NAME} node.description=\"${SINK_DESC}\"" \
            --playback-props="media.class=Audio/Source node.name=${SOURCE_NAME} node.description=\"${SOURCE_DESC}\"" \
            > /dev/null 2>&1 &
        local loop_pid=$!
        echo "${loop_pid}" > "${PID_FILE}"

        # Allow daemon node registration
        sleep 0.5

        if kill -0 "${loop_pid}" 2>/dev/null; then
            echo "[Converse Audio] Successfully launched pw-loopback daemon (PID: ${loop_pid})."
            echo "  - Null sink:   ${SINK_NAME}"
            echo "  - Virtual mic: ${SOURCE_NAME} ('${SOURCE_DESC}')"
            echo "  - Status: Active and available in browser / meeting inputs."
            return 0
        else
            echo "[Converse Audio] Error: Failed to start pw-loopback process." >&2
            rm -f "${PID_FILE}"
            return 1
        fi
    fi

    echo "[Converse Audio] Error: Neither pactl nor pw-loopback is available." >&2
    return 1
}

stop_virtual_mic() {
    echo "[Converse Audio] Stopping Converse Virtual Microphone..."
    local stopped=0

    # Stop pactl modules if present
    if command -v pactl >/dev/null 2>&1; then
        local modules
        modules=$(pactl list short modules 2>/dev/null || true)
        while IFS= read -r line; do
            if echo "${line}" | grep -q "${SINK_NAME}"; then
                local mod_id
                mod_id=$(echo "${line}" | awk '{print $1}')
                pactl unload-module "${mod_id}" 2>/dev/null || true
                stopped=1
            elif echo "${line}" | grep -q "${SOURCE_NAME}"; then
                local mod_id
                mod_id=$(echo "${line}" | awk '{print $1}')
                pactl unload-module "${mod_id}" 2>/dev/null || true
                stopped=1
            fi
        done <<< "${modules}"
    fi

    # Stop PipeWire background daemon if running
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid=$(cat "${PID_FILE}" 2>/dev/null || true)
        if [[ -n "${pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
            stopped=1
        fi
        rm -f "${PID_FILE}"
    fi

    # Kill any lingering pw-loopback instances for Converse
    pkill -f "node.name=${SINK_NAME}" 2>/dev/null || true

    if [[ ${stopped} -eq 1 ]]; then
        echo "[Converse Audio] Converse Virtual Microphone stopped."
    else
        echo "[Converse Audio] No active Converse Virtual Microphone was running."
    fi
}

show_status() {
    echo "========================================================"
    echo " Converse Virtual Audio Loopback Status"
    echo "========================================================"
    if is_active; then
        echo "Status: ACTIVE"
        echo "Sink Device:   ${SINK_NAME} ('${SINK_DESC}')"
        echo "Source Device: ${SOURCE_NAME} ('${SOURCE_DESC}')"
        if [[ -f "${PID_FILE}" ]]; then
            echo "PID File:      ${PID_FILE} (PID: $(cat "${PID_FILE}"))"
        fi
        if command -v wpctl >/dev/null 2>&1; then
            echo ""
            echo "--- Active PipeWire Endpoints ---"
            wpctl status | grep -E "Converse|Sinks|Sources" -A 3 || true
        fi
    else
        echo "Status: INACTIVE"
        echo "Run './scripts/setup_virtual_mic.sh start' to initialize."
    fi
    echo "========================================================"
}

# ------------------------------------------------------------------------------
# Main Dispatcher
# ------------------------------------------------------------------------------
COMMAND="${1:-start}"

case "${COMMAND}" in
    start)
        start_virtual_mic
        ;;
    stop|teardown)
        stop_virtual_mic
        ;;
    restart)
        stop_virtual_mic
        sleep 0.5
        start_virtual_mic
        ;;
    status)
        show_status
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "Unknown command: ${COMMAND}" >&2
        show_help
        exit 1
        ;;
esac
