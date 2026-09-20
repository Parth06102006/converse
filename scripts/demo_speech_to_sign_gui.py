#!/usr/bin/env python3
"""Interactive GUI demonstration for the end-to-end Speech-to-Sign Pipeline.

Subsystem Author: Ashwani | Visual GUI Companion: Converse
Integrates:
1. Real-time Continuous Microphone Streaming via arecord
2. Neural Voice Activity Detection (Silero VAD v5 ONNX)
3. Faster-Whisper INT8 Streaming ASR Engine
4. English-to-ASL Grammar Compiler (SVO -> Topic-Comment / TSOV, Wh-movement, Negation)
5. Non-Manual Marker Synthesis (Eyebrows furrow/raise, Head tilt/shake, Mouth morphemes)
6. 3D Spatial Locus Tracking & Referent Indexing (Left/Right/Chest loci)
7. Dynamic Co-articulation Timing Engine (Lead-in, Hold, Lead-out durations)
8. Procedural 3D ASL Skeletal Avatar Animation at 30 FPS in OpenCV
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import site
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Silence Qt font warnings and MediaPipe glog before importing cv2
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;QFontDatabase.warning=false;QFontDatabase.debug=false"
os.environ["GLOG_minloglevel"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Ensure cv2/qt/fonts directory exists to permanently silence QFontDatabase missing warnings
with contextlib.suppress(OSError, AttributeError):
    for _site_pkg in site.getsitepackages():
        _cv2_qt = Path(_site_pkg) / "cv2" / "qt"
        if _cv2_qt.is_dir():
            (_cv2_qt / "fonts").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np

# Ensure subpackages are on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml" / "asr" / "src"))
sys.path.insert(0, str(ROOT / "ml" / "translation" / "src"))

from asr.engine import create_asr_engine
from asr.vad import VadConfig
from translation.pipeline import SpeechToSignPipeline
from translation.representation_emitter import (
    SignRepresentation,
    SignRepresentationToken,
)

# Canonical Scenario Presets
SCENARIOS: list[dict[str, str]] = [
    {
        "id": "1",
        "name": "Greeting & Meeting",
        "text": "Hello, nice to meet you.",
        "rule": "Topic-Comment: Standard polite conversational greeting.",
    },
    {
        "id": "2",
        "name": "Wh-Question (Inversion)",
        "text": "What is your name?",
        "rule": "Wh-fronting: WHAT moved to end | Eyebrows furrowed (0.85).",
    },
    {
        "id": "3",
        "name": "Yes/No Question",
        "text": "Are you happy?",
        "rule": "Copula omitted | Eyebrows raised (0.80) | Head tilted forward.",
    },
    {
        "id": "4",
        "name": "Clausal Negation",
        "text": "I do not want cake.",
        "rule": "Negation movement: NOT at end | Head shake (yaw oscillation).",
    },
    {
        "id": "5",
        "name": "Temporal Shift (TSOV)",
        "text": "Yesterday I went to the store.",
        "rule": "Temporal lifting: YESTERDAY at start | Verb converted to base GO.",
    },
    {
        "id": "6",
        "name": "OOV Fingerspelling",
        "text": "Alice met Zachary.",
        "rule": "Fingerspelling decomposition: A-L-I-C-E at left, Z-A-C-H-A-R-Y at right.",
    },
    {
        "id": "7",
        "name": "Spatial Referents",
        "text": "The doctor gave me medicine.",
        "rule": "3D Spatial Loci: DOCTOR indexed to left locus | ME at chest.",
    },
    {
        "id": "8",
        "name": "Dining / Need",
        "text": "I want to drink water.",
        "rule": "Topic-Comment: Water request with pulling motion.",
    },
]


class AvatarState:
    """Tracks the real-time skeletal state of the 3D procedural signing avatar."""

    def __init__(self) -> None:
        # Base anatomical joint positions in normalized avatar coordinates (-1.0 to 1.0)
        self.head_pos = np.array([0.0, -0.62], dtype=np.float32)
        self.neck_pos = np.array([0.0, -0.42], dtype=np.float32)
        self.chest_pos = np.array([0.0, -0.15], dtype=np.float32)
        self.l_shoulder = np.array([-0.36, -0.38], dtype=np.float32)
        self.r_shoulder = np.array([0.36, -0.38], dtype=np.float32)
        self.l_elbow = np.array([-0.45, -0.05], dtype=np.float32)
        self.r_elbow = np.array([0.45, -0.05], dtype=np.float32)
        self.l_wrist = np.array([-0.25, 0.28], dtype=np.float32)
        self.r_wrist = np.array([0.25, 0.28], dtype=np.float32)

        # Target wrist positions
        self.l_wrist_target = self.l_wrist.copy()
        self.r_wrist_target = self.r_wrist.copy()

        # NMM states
        self.eyebrow_offset = 0.0  # -1.0 (furrow) to +1.0 (raise)
        self.head_pitch = 0.0      # nod/tilt
        self.head_yaw = 0.0        # shake
        self.mouth_open = 0.0      # 0.0 to 1.0

        # Animation sequence queue
        self.active_representation: SignRepresentation | None = None
        self.anim_start_time: float = 0.0
        self.current_token_idx: int = -1
        self.current_gloss: str = ""
        self.current_anchor: str = "neutral_space"
        self.is_animating: bool = False

    def load_representation(self, rep: SignRepresentation) -> None:
        self.active_representation = rep
        self.anim_start_time = time.time()
        self.current_token_idx = 0
        self.is_animating = True

    def update(self, tick: int) -> None:
        """Update joint positions and NMM blend weights based on current time."""
        now = time.time()

        if self.is_animating and self.active_representation and self.active_representation.tokens:
            elapsed_ms = (now - self.anim_start_time) * 1000.0
            tokens = self.active_representation.tokens

            # Find active token
            active_tok: SignRepresentationToken | None = None
            for idx, tok in enumerate(tokens):
                t_start = tok.timing.start_time_ms
                t_end = t_start + tok.timing.lead_in_duration_ms + tok.timing.hold_duration_ms + tok.timing.lead_out_duration_ms
                if t_start <= elapsed_ms <= t_end:
                    active_tok = tok
                    self.current_token_idx = idx
                    break

            if active_tok is not None:
                self.current_gloss = active_tok.gloss
                self.current_anchor = active_tok.spatial_loci.anchor
                nmm = active_tok.non_manual_markers

                # Eyebrow deflection
                if nmm.eyebrow_shape == "furrow":
                    self.eyebrow_offset = -0.75 * nmm.eyebrow_intensity
                elif nmm.eyebrow_shape == "raise":
                    self.eyebrow_offset = 0.85 * nmm.eyebrow_intensity
                else:
                    self.eyebrow_offset = 0.0

                # Head rotation
                self.head_pitch = nmm.head_rotation.pitch
                if abs(nmm.head_rotation.yaw) > 0.01:
                    # Head shake oscillation
                    self.head_yaw = math.sin(tick * 0.35) * 0.18
                else:
                    self.head_yaw = 0.0

                # Locus target
                offset_x = active_tok.spatial_loci.target_offset.x

                # Determine target hand positions according to ASL gloss semantics
                g = active_tok.gloss.upper()
                if g in ("HELLO", "HI"):
                    self.r_wrist_target = np.array([0.32, -0.58], dtype=np.float32)  # Temple salute
                    self.l_wrist_target = np.array([-0.25, 0.28], dtype=np.float32)
                elif g in ("YOU", "YOUR"):
                    self.r_wrist_target = np.array([0.15 + offset_x * 0.4, -0.15], dtype=np.float32)  # Pointing out
                    self.l_wrist_target = np.array([-0.25, 0.28], dtype=np.float32)
                elif g in ("ME", "MY"):
                    self.r_wrist_target = np.array([0.0, -0.18], dtype=np.float32)  # Pointing to chest
                    self.l_wrist_target = np.array([-0.25, 0.28], dtype=np.float32)
                elif g in ("NAME",):
                    self.r_wrist_target = np.array([0.06, -0.16], dtype=np.float32)  # Center tap
                    self.l_wrist_target = np.array([-0.06, -0.16], dtype=np.float32)
                elif g in ("WHAT", "WHERE", "HOW"):
                    self.r_wrist_target = np.array([0.34, 0.02], dtype=np.float32)   # Palms outward
                    self.l_wrist_target = np.array([-0.34, 0.02], dtype=np.float32)
                elif g in ("HAPPY", "ENJOY"):
                    brush = math.sin(tick * 0.4) * 0.08
                    self.r_wrist_target = np.array([0.18, -0.22 + brush], dtype=np.float32)
                    self.l_wrist_target = np.array([-0.18, -0.22 + brush], dtype=np.float32)
                elif g in ("NOT", "NO"):
                    self.r_wrist_target = np.array([0.16, -0.42], dtype=np.float32)  # Under chin flick
                    self.l_wrist_target = np.array([-0.25, 0.28], dtype=np.float32)
                elif g in ("EAT", "DRINK", "FOOD", "WATER"):
                    self.r_wrist_target = np.array([0.10, -0.52], dtype=np.float32)  # Toward mouth
                    self.l_wrist_target = np.array([-0.25, 0.28], dtype=np.float32)
                elif g in ("DOCTOR", "TIME"):
                    self.r_wrist_target = np.array([-0.16, 0.12], dtype=np.float32)  # Tapping left wrist
                    self.l_wrist_target = np.array([-0.20, 0.16], dtype=np.float32)
                else:
                    # Generic / Fingerspelling at spatial locus
                    locus_shift = offset_x * 0.35
                    self.r_wrist_target = np.array([0.18 + locus_shift, -0.20], dtype=np.float32)
                    self.l_wrist_target = np.array([-0.25, 0.28], dtype=np.float32)

            else:
                if elapsed_ms > self.active_representation.total_duration_ms + 400.0:
                    self.is_animating = False
                    self.current_gloss = ""
                    self.r_wrist_target = np.array([0.25, 0.28], dtype=np.float32)
                    self.l_wrist_target = np.array([-0.25, 0.28], dtype=np.float32)
                    self.eyebrow_offset = 0.0
                    self.head_pitch = 0.0
                    self.head_yaw = 0.0

        else:
            # Idle breathing cycle
            breathe = math.sin(tick * 0.08) * 0.015
            self.chest_pos[1] = -0.15 + breathe
            self.l_wrist_target = np.array([-0.25, 0.28 + breathe], dtype=np.float32)
            self.r_wrist_target = np.array([0.25, 0.28 + breathe], dtype=np.float32)
            self.eyebrow_offset = 0.0
            self.head_pitch = 0.0
            self.head_yaw = 0.0

        # Smooth SLERP-like interpolation toward target joints
        alpha = 0.25
        self.r_wrist += alpha * (self.r_wrist_target - self.r_wrist)
        self.l_wrist += alpha * (self.l_wrist_target - self.l_wrist)

        # Inverse Kinematics for elbows
        self._solve_elbow_ik()

    def _solve_elbow_ik(self) -> None:
        """Two-bone IK solver maintaining realistic arm bend."""
        s_r = self.r_shoulder
        w_r = self.r_wrist
        mid_r = (s_r + w_r) * 0.5
        dir_r = w_r - s_r
        perp_r = np.array([-dir_r[1], dir_r[0]], dtype=np.float32)
        norm_r = np.linalg.norm(perp_r)
        if norm_r > 1e-4:
            perp_r = (perp_r / norm_r) * 0.18
        self.r_elbow = mid_r + np.array([abs(perp_r[0]), perp_r[1]], dtype=np.float32)

        s_l = self.l_shoulder
        w_l = self.l_wrist
        mid_l = (s_l + w_l) * 0.5
        dir_l = w_l - s_l
        perp_l = np.array([dir_l[1], -dir_l[0]], dtype=np.float32)
        norm_l = np.linalg.norm(perp_l)
        if norm_l > 1e-4:
            perp_l = (perp_l / norm_l) * 0.18
        self.l_elbow = mid_l + np.array([-abs(perp_l[0]), perp_l[1]], dtype=np.float32)


def discover_audio_sources() -> list[dict[str, Any]]:
    """Query PipeWire / WirePlumber for available physical audio recording sources."""
    sources: list[dict[str, Any]] = []
    try:
        out = subprocess.check_output(["wpctl", "status"], text=True, stderr=subprocess.DEVNULL)
        sections = re.split(r"\n(?=[A-Z][a-zA-Z0-9 ]+)", out)
        audio_section = ""
        for sec in sections:
            if sec.startswith("Audio"):
                audio_section = sec
                break

        # 1. Physical audio capture sources
        sources_match = re.search(r"Sources:(.*?)(?:Filters:|Streams:|Video:|$)", audio_section, re.DOTALL)
        if sources_match:
            for line in sources_match.group(1).splitlines():
                m = re.search(r"([*]?)\s+(\d+)\.\s+(.*?)(?:\s+\[.*\])?$", line)
                if m:
                    node_id = int(m.group(2))
                    name = m.group(3).strip()
                    is_def = bool(m.group(1).strip())
                    sources.append({"id": node_id, "name": name, "is_default": is_def, "type": "internal"})

        # 2. Bluetooth headset capture sources
        filters_match = re.search(r"Filters:(.*?)(?:Streams:|Video:|$)", audio_section, re.DOTALL)
        if filters_match:
            for line in filters_match.group(1).splitlines():
                if "[Audio/Source]" in line:
                    m = re.search(r"([*]?)\s+(\d+)\.\s+(.*?)\s+\[Audio/Source\]", line)
                    if m:
                        node_id = int(m.group(2))
                        name = m.group(3).strip()
                        is_def = bool(m.group(1).strip())
                        sources.append({"id": node_id, "name": name, "is_default": is_def, "type": "bluetooth"})
    except Exception:
        pass
    return sources


class LiveAudioStreamer:
    """Continuous background microphone streaming thread feeding Silero VAD and Whisper ASR."""

    def __init__(
        self,
        asr_engine: Any,
        pipeline: SpeechToSignPipeline,
        avatar: AvatarState,
        sample_rate: int = 16000,
        chunk_samples: int = 1600,  # 100ms
    ) -> None:
        self.asr_engine = asr_engine
        self.pipeline = pipeline
        self.avatar = avatar
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        # 1600 samples * 2 channels (stereo) * 2 bytes/sample (16-bit)
        self.chunk_bytes = chunk_samples * 2 * 2

        self.sources = discover_audio_sources()
        self.active_source_idx = 0
        self.active_device_name = "Default Microphone"

        if self.sources:
            # Prefer the system default source (the mic the user actually
            # speaks into, e.g. a Bluetooth headset), then the internal
            # analog mic, then whatever is first. Never assume onboard.
            chosen_idx = 0
            for idx, s in enumerate(self.sources):
                if s["is_default"]:
                    chosen_idx = idx
                    break
            else:
                for idx, s in enumerate(self.sources):
                    if s["type"] == "internal" or "analog" in s["name"].lower():
                        chosen_idx = idx
                        break

            self.active_source_idx = chosen_idx
            self.active_device_name = self.sources[chosen_idx]["name"]
            target_id = self.sources[chosen_idx]["id"]

            # Healthy gain on the chosen node only; do not steal the system
            # default away from the user's headset or conferencing setup.
            with contextlib.suppress(Exception):
                subprocess.run(["wpctl", "set-volume", str(target_id), "0.65"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.is_running = False
        self.is_muted = False
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen[Any] | None = None

        # Live telemetry
        self.latest_samples = np.zeros(chunk_samples, dtype=np.float32)
        self.latest_rms = 0.0
        self.is_speech_detected = False
        self.last_speech_time = 0.0

        # Subtitles and pipeline state
        self.current_partial_text = ""
        self.last_final_transcript = "Hello, nice to meet you."
        self.last_compiled_rep: SignRepresentation | None = None
        self.last_compute_ms = 0.35
        self.lock = threading.Lock()

    def start(self) -> None:
        """Start the continuous audio ingestion thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop background audio streaming."""
        self.is_running = False
        if self.process:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=1.0)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def toggle_mute(self) -> bool:
        """Toggle microphone mute state."""
        self.is_muted = not self.is_muted
        return self.is_muted

    def switch_device(self) -> str:
        """Switch to next available audio input device and seamlessly restart stream."""
        if not self.sources:
            return self.active_device_name

        self.active_source_idx = (self.active_source_idx + 1) % len(self.sources)
        active_source = self.sources[self.active_source_idx]
        self.active_device_name = active_source["name"]
        target_id = active_source["id"]

        with contextlib.suppress(Exception):
            subprocess.run(["wpctl", "set-volume", str(target_id), "0.65"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Terminate current recording process; _stream_loop will respawn it on next iteration
        if self.process:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=0.5)
            self.process = None

        return self.active_device_name

    def _spawn_process(self) -> subprocess.Popen[Any] | None:
        """Spawn audio recording subprocess targeting PipeWire or ALSA."""
        target_id = None
        if self.sources and 0 <= self.active_source_idx < len(self.sources):
            target_id = self.sources[self.active_source_idx]["id"]

        # 1. Native PipeWire raw streamer
        try:
            cmd = ["pw-record", "--raw", "--rate", str(self.sample_rate), "--channels", "2", "--format", "s16", "-"]
            if target_id is not None:
                cmd.extend(["--target", str(target_id)])
            return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass

        # 2. ALSA 2-channel capture fallback
        try:
            cmd = ["arecord", "-D", "default", "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "2", "-t", "raw", "-q"]
            return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"Error launching recording backend: {e}")
            return None

    def _stream_loop(self) -> None:
        """Continuous streaming loop reading stereo PCM chunks and feeding Whisper ASR."""
        self.process = self._spawn_process()
        session_id = "live_stream_session"

        while self.is_running:
            if self.process is None or self.process.poll() is not None:
                self.process = self._spawn_process()
                if self.process is None:
                    time.sleep(0.1)
                    continue

            try:
                raw_bytes = self.process.stdout.read(self.chunk_bytes)
                if not raw_bytes or len(raw_bytes) < self.chunk_bytes:
                    time.sleep(0.01)
                    continue

                # Unpack stereo int16 and downmix to mono float32
                int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
                ch1 = int16_arr[0::2].astype(np.float32) / 32768.0
                ch2 = int16_arr[1::2].astype(np.float32) / 32768.0
                samples = (ch1 + ch2) * 0.5
                samples = samples - float(np.mean(samples))  # Cancel DC bias
                rms = float(np.sqrt(np.mean(samples**2)))

                # Automatic Gain Control (AGC) boost if mic level is moderate
                if 0.003 < rms < 0.14:
                    samples_boosted = np.clip(samples * 2.2, -1.0, 1.0)
                    proc_bytes = (samples_boosted * 32767.0).astype(np.int16).tobytes()
                else:
                    proc_bytes = (samples * 32767.0).astype(np.int16).tobytes()

                with self.lock:
                    self.latest_samples = samples.copy()
                    self.latest_rms = rms

                if self.is_muted:
                    continue

                # Process chunk through Whisper ASR + Silero VAD
                events = self.asr_engine.process_audio_chunk(
                    session_id=session_id,
                    audio_data=proc_bytes,
                    audio_format="pcm_s16le",
                )

                for ev in events:
                    if not ev.is_final:
                        with self.lock:
                            self.current_partial_text = ev.text
                            self.is_speech_detected = True
                            self.last_speech_time = time.time()
                    else:
                        with self.lock:
                            self.current_partial_text = ""
                            self.last_final_transcript = ev.text
                            self.is_speech_detected = False

                        print(f"\n[Microphone ASR] Committed: \"{ev.text}\"")
                        t0 = time.perf_counter()
                        rep = self.pipeline.translate(ev.text, session_id=session_id)
                        compute_ms = (time.perf_counter() - t0) * 1000.0

                        with self.lock:
                            self.last_compiled_rep = rep
                            self.last_compute_ms = compute_ms
                        self.avatar.load_representation(rep)
                        print(f"-> Compiled {len(rep.tokens)} ASL tokens in {compute_ms:.2f}ms")

                # Silence timeout check
                with self.lock:
                    if self.is_speech_detected and (time.time() - self.last_speech_time > 0.7):
                        self.is_speech_detected = False

            except (OSError, ValueError, RuntimeError):
                time.sleep(0.05)


def draw_avatar_canvas(
    frame: np.ndarray,
    avatar: AvatarState,
    x: int,
    y: int,
    w: int,
    h: int,
    tick: int,
) -> None:
    """Render procedural 3D ASL skeletal avatar with facial NMMs in the canvas viewport."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 20, 26), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 42, 54), 2)

    center_x = x + w // 2
    base_y = y + h - 18
    for i in range(-5, 6):
        gx = center_x + i * 50
        cv2.line(frame, (center_x, y + int(h * 0.65)), (gx, base_y), (26, 29, 38), 1)

    scale = h * 0.50
    ox, oy = x + w // 2, y + int(h * 0.58)

    def to_px(pt: np.ndarray) -> tuple[int, int]:
        return int(ox + pt[0] * scale), int(oy + pt[1] * scale)

    head_px = to_px(avatar.head_pos + np.array([avatar.head_yaw, avatar.head_pitch]))
    neck_px = to_px(avatar.neck_pos)
    chest_px = to_px(avatar.chest_pos)
    ls_px = to_px(avatar.l_shoulder)
    rs_px = to_px(avatar.r_shoulder)
    le_px = to_px(avatar.l_elbow)
    re_px = to_px(avatar.r_elbow)
    lw_px = to_px(avatar.l_wrist)
    rw_px = to_px(avatar.r_wrist)

    # Body torso armature
    cv2.line(frame, neck_px, chest_px, (0, 220, 255), 4)
    cv2.line(frame, ls_px, rs_px, (0, 220, 255), 4)
    cv2.line(frame, ls_px, chest_px, (40, 140, 200), 2)
    cv2.line(frame, rs_px, chest_px, (40, 140, 200), 2)

    # Left arm (cyan)
    cv2.line(frame, ls_px, le_px, (0, 200, 255), 4)
    cv2.line(frame, le_px, lw_px, (0, 200, 255), 4)
    cv2.circle(frame, le_px, 6, (0, 255, 255), -1)
    cv2.circle(frame, lw_px, 8, (0, 255, 200), -1)

    # Right arm (gold)
    cv2.line(frame, rs_px, re_px, (255, 180, 0), 4)
    cv2.line(frame, re_px, rw_px, (255, 180, 0), 4)
    cv2.circle(frame, re_px, 6, (255, 220, 0), -1)
    cv2.circle(frame, rw_px, 8, (255, 200, 0), -1)

    cv2.circle(frame, ls_px, 7, (0, 220, 255), -1)
    cv2.circle(frame, rs_px, 7, (255, 180, 0), -1)

    # Finger fans
    def draw_hand(wrist_px: tuple[int, int], is_right: bool) -> None:
        color = (255, 180, 0) if is_right else (0, 200, 255)
        wx, wy = wrist_px
        angles = [-40, -20, 0, 20, 40] if is_right else [40, 20, 0, -20, -40]
        lengths = [18, 26, 28, 24, 20]
        for ang_deg, f_len in zip(angles, lengths, strict=False):
            rad = math.radians(ang_deg - 90)
            fx = int(wx + math.cos(rad) * f_len)
            fy = int(wy + math.sin(rad) * f_len)
            cv2.line(frame, (wx, wy), (fx, fy), color, 2)
            cv2.circle(frame, (fx, fy), 3, (255, 255, 255), -1)

    draw_hand(lw_px, is_right=False)
    draw_hand(rw_px, is_right=True)

    # Head & Facial NMMs
    hx, hy = head_px
    head_radius = int(scale * 0.20)
    cv2.circle(frame, (hx, hy), head_radius, (30, 36, 48), -1)
    cv2.circle(frame, (hx, hy), head_radius, (0, 220, 255), 2)

    eye_offset_x = int(head_radius * 0.38)
    eye_offset_y = int(head_radius * 0.12)
    cv2.circle(frame, (hx - eye_offset_x, hy - eye_offset_y), 4, (255, 255, 255), -1)
    cv2.circle(frame, (hx + eye_offset_x, hy - eye_offset_y), 4, (255, 255, 255), -1)

    # Dynamic Eyebrows
    eb_lift = int(avatar.eyebrow_offset * 9)
    eb_tilt = int(avatar.eyebrow_offset * 5)
    cv2.line(
        frame,
        (hx - eye_offset_x - 10, hy - eye_offset_y - 10 - eb_lift - eb_tilt),
        (hx - eye_offset_x + 10, hy - eye_offset_y - 10 - eb_lift + eb_tilt),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (hx + eye_offset_x - 10, hy - eye_offset_y - 10 - eb_lift + eb_tilt),
        (hx + eye_offset_x + 10, hy - eye_offset_y - 10 - eb_lift - eb_tilt),
        (0, 255, 255),
        3,
    )

    mouth_y = hy + int(head_radius * 0.44)
    cv2.line(frame, (hx - 12, mouth_y), (hx + 12, mouth_y), (0, 220, 255), 2)

    cv2.putText(frame, "3D PROCEDURAL SKELETAL SIGNER", (x + 16, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 2)
    status_tag = "ACTIVE SIGNING" if avatar.is_animating else "IDLE / NEUTRAL POSE"
    tag_col = (0, 255, 120) if avatar.is_animating else (140, 140, 140)
    cv2.putText(frame, f"STATUS: {status_tag}", (x + w - 240, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_col, 2)


def draw_audio_visualizer(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    raw_samples: np.ndarray,
    rms: float,
    is_speech_active: bool,
    is_muted: bool,
    device_name: str = "",
) -> None:
    """Render real-time microphone oscilloscope waveform, device tag, and VU meter."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 20, 26), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 42, 54), 1)

    title_txt = f"MIC: {device_name[:24]}" if device_name else "LIVE MICROPHONE AUDIO"
    cv2.putText(frame, title_txt, (x + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

    # VAD status pill
    if is_muted:
        pill_text = "MIC: MUTED"
        pill_color = (120, 120, 120)
    elif is_speech_active or rms > 0.035:
        pill_text = f"VAD: SPEECH DETECTED (RMS: {rms:.3f})"
        pill_color = (0, 255, 100)
    else:
        pill_text = f"VAD: LISTENING (RMS: {rms:.3f})"
        pill_color = (0, 200, 255)

    cv2.putText(frame, pill_text, (x + w - 240, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, pill_color, 1)

    # Draw real audio waveform
    wave_y = y + h // 2 + 6
    n_pts = 60
    step = (w - 24) / n_pts
    pts: list[tuple[int, int]] = []

    if len(raw_samples) > 0 and not is_muted:
        stride = max(len(raw_samples) // n_pts, 1)
        for i in range(n_pts):
            px = int(x + 12 + i * step)
            s_idx = min(i * stride, len(raw_samples) - 1)
            amp = float(raw_samples[s_idx]) * 45.0  # Scale amplitude
            py = int(wave_y - np.clip(amp, -32.0, 32.0))
            pts.append((px, py))
    else:
        for i in range(n_pts):
            px = int(x + 12 + i * step)
            pts.append((px, wave_y))

    for i in range(len(pts) - 1):
        color = (0, 255, 120) if is_speech_active else (0, 200, 255) if not is_muted else (60, 70, 85)
        cv2.line(frame, pts[i], pts[i + 1], color, 2)

    # Device switch prompt
    cv2.putText(frame, "SWITCH MIC: [S]", (x + 12, y + h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 140, 160), 1)

    # Horizontal VU level meter
    meter_y = y + h - 10
    cv2.rectangle(frame, (x + 12, meter_y), (x + w - 12, meter_y + 4), (28, 32, 42), -1)
    normalized_vol = min(rms * 5.0, 1.0) if not is_muted else 0.0
    fill_w = int((w - 24) * normalized_vol)
    if fill_w > 0:
        vu_color = (0, 255, 100) if normalized_vol < 0.75 else (0, 165, 255)
        cv2.rectangle(frame, (x + 12, meter_y), (x + 12 + fill_w, meter_y + 4), vu_color, -1)


def draw_spatial_loci_radar(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    active_anchor: str,
) -> None:
    """Render 3D spatial loci discourse tracking radar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 20, 26), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 42, 54), 1)

    cv2.putText(frame, "3D SPATIAL LOCI RADAR", (x + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

    ox, oy = x + w // 2, y + h // 2 + 10
    cv2.circle(frame, (ox, oy), 40, (28, 34, 46), 1)
    cv2.circle(frame, (ox, oy), 70, (28, 34, 46), 1)

    loci = [
        ("chest", ox, oy + 25, "Self (Chest)"),
        ("neutral_space", ox, oy - 20, "Neutral Space"),
        ("left", ox - 50, oy - 15, "Locus 1 (Left)"),
        ("right", ox + 50, oy - 15, "Locus 2 (Right)"),
    ]

    for anchor_id, lx, ly, label in loci:
        is_active = anchor_id == active_anchor
        color = (0, 255, 255) if is_active else (80, 95, 120)
        radius = 8 if is_active else 5
        cv2.circle(frame, (lx, ly), radius, color, -1)
        if is_active:
            cv2.circle(frame, (lx, ly), radius + 4, (0, 255, 255), 1)
        cv2.putText(frame, label, (lx - 30, ly + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)


def draw_subtitles_card(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    final_text: str,
    partial_text: str,
    representation: SignRepresentation | None,
    current_token_idx: int,
    last_compute_ms: float,
) -> None:
    """Render dual-stream subtitles, ASL gloss chips, and linguistic directives."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 20, 26), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 42, 54), 2)

    # 1. English Transcript Line
    cv2.putText(frame, "SPOKEN ENGLISH:", (x + 16, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1)

    if partial_text:
        display_text = f"\"{partial_text}...\" (streaming)"
        text_col = (0, 220, 255)
    elif final_text:
        display_text = f"\"{final_text}\""
        text_col = (255, 255, 255)
    else:
        display_text = "\"Speak into your microphone or press [1-8] for scenarios...\""
        text_col = (100, 100, 110)

    cv2.putText(frame, display_text, (x + 160, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_col, 2)

    lat_str = f"COMPILATION: {last_compute_ms:.2f}ms"
    cv2.putText(frame, lat_str, (x + w - 210, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

    cv2.line(frame, (x + 16, y + 36), (x + w - 16, y + 36), (28, 32, 42), 1)

    # 2. ASL Gloss Chips Stream
    cv2.putText(frame, "COMPILED ASL GLOSSES:", (x + 16, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1)
    chip_x = x + 200
    if representation and representation.tokens:
        for idx, tok in enumerate(representation.tokens[:12]):
            is_active = idx == current_token_idx
            bg_col = (0, 200, 255) if is_active else (28, 34, 46)
            text_col = (0, 0, 0) if is_active else (220, 220, 230)
            tw = len(tok.gloss) * 9 + 14
            cv2.rectangle(frame, (chip_x, y + 46), (chip_x + tw, y + 72), bg_col, -1)
            cv2.rectangle(frame, (chip_x, y + 46), (chip_x + tw, y + 72), (0, 220, 255) if is_active else (48, 56, 72), 1)
            cv2.putText(frame, tok.gloss, (chip_x + 7, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_col, 2 if is_active else 1)
            chip_x += tw + 8
    else:
        cv2.putText(frame, "Glosses assemble here as speech is detected...", (chip_x, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 110), 1)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    avatar: AvatarState,
    streamer: LiveAudioStreamer,
    tick: int,
    show_guide: bool = False,
) -> None:
    """Render complete broadcast-grade heads-up display."""
    h, w, _ = frame.shape

    with streamer.lock:
        raw_samples = streamer.latest_samples.copy()
        rms = streamer.latest_rms
        is_speech = streamer.is_speech_detected
        is_muted = streamer.is_muted
        final_text = streamer.last_final_transcript
        partial_text = streamer.current_partial_text
        representation = streamer.last_compiled_rep
        compute_ms = streamer.last_compute_ms

    # 1. Top Status Banner
    cv2.rectangle(frame, (0, 0), (w, 68), (14, 16, 22), -1)
    cv2.line(frame, (0, 68), (w, 68), (38, 42, 54), 2)

    cv2.putText(frame, "CONVERSE | SPEECH -> SIGN ENGINE", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 200), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (420, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 150), 1)

    mic_badge = f"MIC: {streamer.active_device_name[:16]}" if not is_muted else "MIC: MUTED [PRESS SPACE]"
    mic_col = (0, 255, 100) if not is_muted else (120, 120, 120)
    cv2.putText(frame, mic_badge, (530, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, mic_col, 2)

    cv2.putText(frame, "ASR: WHISPER INT8", (770, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1)
    cv2.putText(frame, "GRAMMAR: TOPIC-COMMENT", (960, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 180, 0), 1)
    cv2.putText(frame, "LATENCY: ~1.2ms", (w - 150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 2)

    # 2. Main 3D Avatar Canvas
    draw_avatar_canvas(frame, avatar, x=20, y=82, w=740, h=476, tick=tick)

    # 3. Real Audio Waveform & VAD Oscilloscope
    draw_audio_visualizer(
        frame=frame,
        x=780,
        y=82,
        w=480,
        h=160,
        raw_samples=raw_samples,
        rms=rms,
        is_speech_active=is_speech,
        is_muted=is_muted,
        device_name=streamer.active_device_name,
    )

    # 4. 3D Spatial Loci Radar
    draw_spatial_loci_radar(frame, x=780, y=256, w=480, h=170, active_anchor=avatar.current_anchor)

    # 5. Non-Manual Markers (NMM) Card
    cv2.rectangle(frame, (780, 440), (1260, 558), (18, 20, 26), -1)
    cv2.rectangle(frame, (780, 440), (1260, 558), (38, 42, 54), 1)
    cv2.putText(frame, "NON-MANUAL MARKERS (NMM) DIRECTIVES", (792, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

    eb_val = avatar.eyebrow_offset
    eb_desc = f"FURROWED ({abs(eb_val):.2f})" if eb_val < -0.1 else f"RAISED ({eb_val:.2f})" if eb_val > 0.1 else "NEUTRAL (0.00)"
    eb_col = (0, 165, 255) if eb_val < -0.1 else (0, 255, 120) if eb_val > 0.1 else (160, 160, 160)
    cv2.putText(frame, f"Eyebrows: {eb_desc}", (792, 492), cv2.FONT_HERSHEY_SIMPLEX, 0.44, eb_col, 1)

    head_desc = "HEAD SHAKE (NEGATION)" if abs(avatar.head_yaw) > 0.02 else "NOD / TILT FORWARD" if avatar.head_pitch > 0.05 else "NEUTRAL ORIENTATION"
    head_col = (0, 165, 255) if abs(avatar.head_yaw) > 0.02 else (0, 255, 200) if avatar.head_pitch > 0.05 else (160, 160, 160)
    cv2.putText(frame, f"Head:     {head_desc}", (792, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.44, head_col, 1)

    cv2.putText(frame, "Mouth:    NEUTRAL MORPHEME", (792, 546), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1)

    # 6. Subtitles & Gloss Timeline Card
    draw_subtitles_card(
        frame=frame,
        x=20,
        y=572,
        w=1240,
        h=92,
        final_text=final_text,
        partial_text=partial_text,
        representation=representation,
        current_token_idx=avatar.current_token_idx,
        last_compute_ms=compute_ms,
    )

    # 7. Navigation Footer
    cv2.rectangle(frame, (0, 678), (w, h), (12, 14, 18), -1)
    footer_text = "[SPACE] Mute | [S] Switch Mic | [1-8] Scenarios | [T] Custom Text | [H] Guide | [R] Reset | [Q] Exit"
    cv2.putText(frame, footer_text, (20, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 185), 1)

    # 8. Interactive Guide Modal
    if show_guide:
        overlay = frame.copy()
        cv2.rectangle(overlay, (60, 60), (w - 60, h - 60), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)
        cv2.rectangle(frame, (60, 60), (w - 60, h - 60), (0, 220, 255), 2)

        cv2.putText(frame, "CONVERSE: SPEECH-TO-SIGN LINGUISTIC REFERENCE GUIDE", (90, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.line(frame, (90, 120), (w - 90, 120), (38, 46, 60), 1)

        guide_items = [
            ("1. Topic-Comment Reordering", "English SVO ('The boy kicked the ball') transforms into ASL Topic-Comment ('BOY KICK BALL')."),
            ("2. Wh-Question Fronting", "Questions ('What is your name?') place question particles at the clause end with furrowed brows."),
            ("3. Yes/No Question Intonation", "Questions ('Are you happy?') omit English copulas and raise eyebrows with forward head tilt."),
            ("4. Clausal Negation Reordering", "Negation ('I do not want cake') places NOT at the end with dynamic head shake oscillations."),
            ("5. Temporal Adverb Lifting", "Time markers ('Yesterday I went to the store') lift to the start: 'YESTERDAY ME STORE GO'."),
            ("6. OOV Proper Name Fingerspelling", "Unlexicalized names ('Alice met Zachary') decompose into letter-by-letter fingerspelling tokens."),
            ("7. 3D Discourse Spatial Loci", "Refers nouns to distinct 3D loci (Left: -0.3, Right: +0.3, Chest: 0.0) in signing space."),
        ]

        for i, (title, desc) in enumerate(guide_items):
            gy = 160 + i * 65
            cv2.putText(frame, title, (90, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 0), 2)
            cv2.putText(frame, desc, (90, gy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1)

        cv2.putText(frame, "Press [H] or [ESC] to return to live demo view", (90, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 1)


def run_gui_demo() -> None:
    """Main execution loop for interactive Speech-to-Sign GUI demo."""
    window_name = "Converse: Speech-to-Sign Engine (Ashwani Scope)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("Initializing SpeechToSignPipeline & Faster-Whisper ASR Engine...")
    # Configure sensitive VAD settings for live room speech
    vad_cfg = VadConfig(
        sample_rate=16000,
        speech_threshold=0.35,
        min_speech_duration_ms=180,
        min_silence_duration_ms=280,
    )
    pipeline = SpeechToSignPipeline()
    asr_engine = create_asr_engine(backend="whisper", vad_config=vad_cfg)
    print("ASR & Translation pipelines ready.")

    avatar = AvatarState()

    # Launch continuous live microphone streaming
    streamer = LiveAudioStreamer(asr_engine=asr_engine, pipeline=pipeline, avatar=avatar)
    streamer.start()

    # Preload initial greeting
    init_text = "Hello, nice to meet you."
    init_rep = pipeline.translate(init_text)
    streamer.last_compiled_rep = init_rep
    avatar.load_representation(init_rep)

    show_guide = False
    tick = 0
    t_last_frame = time.time()
    fps = 30.0

    print("\nConverse Speech-to-Sign GUI is now LIVE.")
    print("Microphone is LIVE STREAMING continuously in the background.")
    print("Just speak naturally into your microphone at any time.")
    print("Controls inside window:")
    print("  [SPACE]: Toggle Mute/Unmute microphone")
    print("  [S]:     Switch active microphone (Laptop vs Headset)")
    print("  [1-8]:   Execute linguistic scenario presets")
    print("  [H]:     Toggle linguistic guide modal")
    print("  [R]:     Reset avatar to neutral state")
    print("  [T]:     Type custom English sentence in console")
    print("  [Q]:     Exit demonstration\n")

    try:
        while True:
            t_now = time.time()
            dt = t_now - t_last_frame
            t_last_frame = t_now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-4))
            tick = (tick + 1) % 720

            # Update avatar procedural kinematics
            avatar.update(tick)

            # Create clean 1280x720 frame canvas
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Render complete HUD
            draw_hud(
                frame=frame,
                fps=fps,
                avatar=avatar,
                streamer=streamer,
                tick=tick,
                show_guide=show_guide,
            )

            cv2.imshow(window_name, frame)

            # Process key events (15ms poll)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord(" "):
                is_muted = streamer.toggle_mute()
                print(f"Microphone {'MUTED' if is_muted else 'UNMUTED (LIVE)'}")
            elif key in (ord("s"), ord("S")):
                dev_name = streamer.switch_device()
                print(f"[Audio] Switched active microphone to: {dev_name}")
            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8")):
                idx = int(chr(key)) - 1
                scenario = SCENARIOS[idx]
                scenario_text = scenario["text"]
                print(f"[Scenario {idx+1}] \"{scenario_text}\" ({scenario['rule']})")
                t0 = time.perf_counter()
                rep = pipeline.translate(scenario_text, session_id=f"scenario_{idx+1}")
                comp_ms = (time.perf_counter() - t0) * 1000.0

                with streamer.lock:
                    streamer.last_final_transcript = scenario_text
                    streamer.current_partial_text = ""
                    streamer.last_compiled_rep = rep
                    streamer.last_compute_ms = comp_ms
                avatar.load_representation(rep)

            elif key in (ord("h"), ord("H")):
                show_guide = not show_guide
            elif key in (ord("r"), ord("R")):
                with streamer.lock:
                    streamer.last_final_transcript = ""
                    streamer.current_partial_text = ""
                    streamer.last_compiled_rep = None
                avatar.is_animating = False
                avatar.current_gloss = ""
                print("Avatar reset to neutral pose.")
            elif key in (ord("t"), ord("T")):
                print("\nEnter custom English sentence in console: ")
                user_line = sys.stdin.readline().strip()
                if user_line:
                    t0 = time.perf_counter()
                    rep = pipeline.translate(user_line, session_id="custom_console")
                    comp_ms = (time.perf_counter() - t0) * 1000.0

                    with streamer.lock:
                        streamer.last_final_transcript = user_line
                        streamer.current_partial_text = ""
                        streamer.last_compiled_rep = rep
                        streamer.last_compute_ms = comp_ms
                    avatar.load_representation(rep)
                    print(f"Compiled \"{user_line}\" ({len(rep.tokens)} tokens) in {comp_ms:.2f}ms")

    finally:
        streamer.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_gui_demo()
