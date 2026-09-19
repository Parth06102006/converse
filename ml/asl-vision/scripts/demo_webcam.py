"""Interactive real-time webcam demonstration for the end-to-end ASL Sign-to-Speech pipeline.

Integrates:
1. MediaPipe 3D Landmark Extractor (Pose + Hands)
2. Mathematical Coordinate Normalization & One-Euro Jitter Filter
3. 30-frame Sliding Window ST-GCN Sign Gloss Classifier
4. Temporal Gloss Stabilizer & Debounce Buffer
5. ASL-to-English Sentence Reconstructor
6. Local Kokoro Neural TTS Audio Synthesis & Playback
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from asl_vision.engine import ASLVisionEngine, EngineConfig, SignDetection
from asl_vision.landmarks import LandmarkExtractor

# Landmark bone connections for visual skeletal overlay
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm knuckle bridge
)

POSE_CONNECTIONS = (
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
)

# Canonical ASL idioms and conversational phrases
CANONICAL_PATTERNS: dict[tuple[str, ...], str] = {
    ("HELLO",): "Hello!",
    ("HI",): "Hi there!",
    ("HELLO", "NICE", "MEET"): "Hello, nice to meet you.",
    ("HELLO", "NICE", "MEET", "YOU"): "Hello, nice to meet you.",
    ("NICE", "MEET", "YOU"): "Nice to meet you.",
    ("NICE", "MEET"): "Nice to meet you.",
    ("MEET", "YOU", "NICE"): "Nice to meet you.",
    ("HOW", "YOU"): "How are you?",
    ("HOW", "ARE", "YOU"): "How are you?",
    ("THANK-YOU",): "Thank you!",
    ("THANK", "YOU"): "Thank you!",
    ("THANK-YOU", "HELP"): "Thank you for your help.",
    ("THANK", "YOU", "HELP"): "Thank you for your help.",
    ("THANK-YOU", "VERY", "MUCH"): "Thank you very much.",
    ("WELCOME",): "You are welcome.",
    ("YOU", "WELCOME"): "You are welcome.",
    ("GOOD", "MORNING"): "Good morning.",
    ("GOOD", "AFTERNOON"): "Good afternoon.",
    ("GOOD", "NIGHT"): "Good night.",
    ("SEE", "YOU", "LATER"): "See you later.",
    ("SEE", "LATER"): "See you later.",
    ("GOODBYE",): "Goodbye.",
    ("BYE",): "Goodbye.",
    ("PLEASE",): "Please.",
    ("SORRY",): "I am sorry.",
    ("SORRY", "I", "LATE"): "Sorry, I am late.",
    ("SORRY", "LATE"): "Sorry, I am late.",
    ("PLEASE", "HELP", "ME"): "Please help me.",
    ("PLEASE", "HELP"): "Please help me.",
    ("EXCUSE", "ME"): "Excuse me.",
    ("YES",): "Yes.",
    ("NO",): "No.",
}

# Irregular verb conjugations
VERB_CONJUGATIONS: dict[str, dict[str, str]] = {
    "GO": {"base": "go", "present": "goes", "past": "went", "continuous": "going"},
    "SEE": {"base": "see", "present": "sees", "past": "saw", "continuous": "seeing"},
    "MEET": {"base": "meet", "present": "meets", "past": "met", "continuous": "meeting"},
    "EAT": {"base": "eat", "present": "eats", "past": "ate", "continuous": "eating"},
    "DRINK": {"base": "drink", "present": "drinks", "past": "drank", "continuous": "drinking"},
    "BUY": {"base": "buy", "present": "buys", "past": "bought", "continuous": "buying"},
    "WANT": {"base": "want", "present": "wants", "past": "wanted", "continuous": "wanting"},
    "NEED": {"base": "need", "present": "needs", "past": "needed", "continuous": "needing"},
    "HELP": {"base": "help", "present": "helps", "past": "helped", "continuous": "helping"},
    "LIKE": {"base": "like", "present": "likes", "past": "liked", "continuous": "liking"},
    "LOVE": {"base": "love", "present": "loves", "past": "loved", "continuous": "loving"},
    "LIVE": {"base": "live", "present": "lives", "past": "lived", "continuous": "living"},
    "HAVE": {"base": "have", "present": "has", "past": "had", "continuous": "having"},
    "COME": {"base": "come", "present": "comes", "past": "came", "continuous": "coming"},
    "LEAVE": {"base": "leave", "present": "leaves", "past": "left", "continuous": "leaving"},
    "WORK": {"base": "work", "present": "works", "past": "worked", "continuous": "working"},
    "LEARN": {"base": "learn", "present": "learns", "past": "learned", "continuous": "learning"},
    "UNDERSTAND": {"base": "understand", "present": "understands", "past": "understood", "continuous": "understanding"},
    "KNOW": {"base": "know", "present": "knows", "past": "knew", "continuous": "knowing"},
    "CALL": {"base": "call", "present": "calls", "past": "called", "continuous": "calling"},
    "DANCE": {"base": "dance", "present": "dances", "past": "danced", "continuous": "dancing"},
}

QUESTION_WORDS = {"WHAT", "WHERE", "WHEN", "WHY", "WHO", "HOW", "WHICH"}
PREDICATE_ADJECTIVES = {"HUNGRY", "THIRSTY", "TIRED", "HAPPY", "SAD", "READY", "FINE", "BUSY", "SICK", "COLD", "HOT", "LATE", "DEAF", "HEARING"}
STANDARD_LOCATIONS = {"STORE", "BATHROOM", "RESTAURANT", "HOSPITAL", "LIBRARY", "AIRPORT", "OFFICE", "BANK", "DOCTOR"}
ZERO_ARTICLE_LOCATIONS = {"SCHOOL", "WORK", "HOME", "CLASS", "BED"}


class KokoroTTSClient:
    """Manages neural TTS audio synthesis and playback via local Kokoro model."""

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8880/v1/audio/speech",
        voice: str = "af_sky",
        enabled: bool = True,
    ) -> None:
        self.api_url = api_url
        self.voice = voice
        self.enabled = enabled
        self.is_speaking = False
        self.last_latency_ms: float = 0.0
        self.last_error: str | None = None
        self._speech_queue: queue.Queue[str] = queue.Queue(maxsize=10)
        self._worker_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker_thread.start()
        self._server_ready = False
        self._check_and_ensure_server()

    def _check_and_ensure_server(self) -> None:
        """Verify if Kokoro HTTP endpoint is responsive, or attempt container launch."""
        if not self.enabled:
            return

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps({"model": "tts-1", "input": "test", "voice": self.voice, "response_format": "wav"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    self._server_ready = True
                    print(f"Kokoro TTS server connected at: {self.api_url} (Voice: {self.voice})")
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            self.last_error = str(err)

        # Attempt to launch docker container if docker is available
        if shutil.which("docker"):
            try:
                print("Starting local Kokoro OpenAI HTTP container...")
                subprocess.run(
                    [
                        "docker", "run", "-d", "--rm",
                        "--name", "kokoro-server",
                        "-p", "8880:3000",
                        "ghcr.io/lucasjinreal/kokoros:main",
                        "openai",
                    ],
                    capture_output=True,
                    check=False,
                )
                time.sleep(1.2)
                self._server_ready = True
                print(f"Kokoro container online. Listening on {self.api_url}")
            except OSError as e:
                self.last_error = str(e)
                print(f"Notice: Could not auto-launch Kokoro container: {e}")

    def speak(self, text: str) -> None:
        """Enqueue text for background synthesis and audio playback."""
        if not self.enabled or not text.strip():
            return
        try:
            self._speech_queue.put_nowait(text.strip())
        except queue.Full:
            pass

    def _playback_worker(self) -> None:
        """Dedicated background thread worker executing synthesis and playback."""
        output_wav = Path("/tmp/converse_kokoro_stream.wav")

        while True:
            text = self._speech_queue.get()
            t0 = time.perf_counter()
            self.is_speaking = True

            synthesized = False

            # Option 1: Fast HTTP API endpoint (Kokoro container / server)
            try:
                payload = json.dumps({
                    "model": "tts-1",
                    "input": text,
                    "voice": self.voice,
                    "response_format": "wav",
                }).encode("utf-8")

                req = urllib.request.Request(
                    self.api_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        output_wav.write_bytes(resp.read())
                        synthesized = True
            except (urllib.error.URLError, TimeoutError, OSError) as http_err:
                self.last_error = str(http_err)

            # Option 2: CLI docker fallback
            if not synthesized and shutil.which("docker"):
                try:
                    subprocess.run(
                        [
                            "docker", "run", "--rm",
                            "-v", "/tmp:/tmp",
                            "ghcr.io/lucasjinreal/kokoros:main",
                            "text", text,
                            "-o", str(output_wav),
                            "-s", self.voice,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    if output_wav.is_file():
                        synthesized = True
                except OSError as doc_err:
                    self.last_error = str(doc_err)

            # Option 3: System TTS fallback (spd-say)
            if not synthesized and shutil.which("spd-say"):
                try:
                    subprocess.run(["spd-say", "-r", "10", text], check=False)
                    self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
                    self.is_speaking = False
                    self._speech_queue.task_done()
                    continue
                except OSError as spd_err:
                    self.last_error = str(spd_err)

            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

            # Audio Playback
            if synthesized and output_wav.is_file():
                if shutil.which("aplay"):
                    subprocess.run(["aplay", "-q", str(output_wav)], check=False)
                elif shutil.which("ffplay"):
                    subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(output_wav)], check=False)

            self.is_speaking = False
            self._speech_queue.task_done()


class PythonGlossStabilizer:
    """Sliding-window debouncer and sentence boundary detector."""

    def __init__(
        self,
        min_confidence: float = 0.55,
        debounce_window_ms: float = 400.0,
        boundary_pause_ms: float = 850.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.debounce_window_ms = debounce_window_ms
        self.boundary_pause_ms = boundary_pause_ms

        self.buffer: list[str] = []
        self.last_detection: SignDetection | None = None
        self.last_emit_time_ms: float = 0.0

    def process_detection(self, detection: SignDetection) -> tuple[bool, bool, str | None]:
        """Ingest a candidate detection. Returns (accepted, is_duplicate, stabilized_gloss)."""
        if detection.confidence < self.min_confidence:
            return False, False, None

        normalized = detection.gloss.strip().upper()
        if not normalized:
            return False, False, None

        # Debounce identical sustained consecutive signs
        if (
            self.last_detection is not None
            and self.last_detection.gloss.upper() == normalized
            and (detection.start_time_ms - self.last_detection.end_time_ms) <= self.debounce_window_ms
        ):
            self.last_emit_time_ms = detection.end_time_ms
            return True, True, None

        # New distinct sign transition
        self.last_detection = detection
        self.last_emit_time_ms = detection.end_time_ms
        self.buffer.append(normalized)
        return True, False, normalized

    def check_boundary(self, current_time_ms: float) -> list[str] | None:
        """Trigger sentence boundary flush if signer pauses for >= boundary_pause_ms."""
        if (
            len(self.buffer) > 0
            and self.last_emit_time_ms > 0
            and (current_time_ms - self.last_emit_time_ms) >= self.boundary_pause_ms
        ):
            return self.flush()
        return None

    def flush(self) -> list[str]:
        """Flush and return the active stabilized gloss sequence."""
        flushed = list(self.buffer)
        self.buffer.clear()
        self.last_detection = None
        self.last_emit_time_ms = 0.0
        return flushed

    def clear(self) -> None:
        """Reset internal buffer state."""
        self.buffer.clear()
        self.last_detection = None
        self.last_emit_time_ms = 0.0


def reconstruct_sentence(glosses: list[str]) -> str:
    """Transforms an ASL gloss sequence into a fluent, grammatical English sentence."""
    normalized = [g.strip().upper() for g in glosses if g.strip()]
    if not normalized:
        return ""

    # 1. Exact Canonical Idioms Match
    key = tuple(normalized)
    if key in CANONICAL_PATTERNS:
        return CANONICAL_PATTERNS[key]

    # 2. Possessive Name Introductions
    if len(normalized) == 3 and normalized[0] == "MY" and normalized[1] == "NAME":
        return f"My name is {normalized[2].capitalize()}."

    # 3. Wh-Questions (Interrogative Inversion)
    for q_word in QUESTION_WORDS:
        if q_word in normalized:
            non_q = [g for g in normalized if g != q_word]
            if q_word == "WHAT":
                if "NAME" in non_q and "YOU" in non_q:
                    return "What is your name?"
                if "TIME" in non_q:
                    return "What time is it?"
                if "WANT" in non_q and "YOU" in non_q:
                    return "What do you want?"
            elif q_word == "WHERE":
                for loc in STANDARD_LOCATIONS:
                    if loc in non_q:
                        return f"Where is the {loc.lower()}?"
                if "GO" in non_q and "YOU" in non_q:
                    return "Where are you going?"
                if "LIVE" in non_q and "YOU" in non_q:
                    return "Where do you live?"
            elif q_word == "WHEN":
                if "START" in non_q and "CLASS" in non_q:
                    return "When does the class start?"
                if "ARRIVE" in non_q or "LEAVE" in non_q:
                    verb = "arrive" if "ARRIVE" in non_q else "leave"
                    return f"When will you {verb}?"
            elif q_word == "WHY" and "LATE" in non_q:
                return "Why are you late?"
            elif q_word == "HOW" and "MUCH" in non_q:
                return "How much does this cost?"

    # 4. Temporal Tense Shifting (Past / Future / Present)
    tense = "present"
    time_marker = ""
    filtered_glosses: list[str] = []

    for g in normalized:
        if g in ("YESTERDAY", "PAST", "BEFORE"):
            tense = "past"
            time_marker = "Yesterday, " if g == "YESTERDAY" else ""
        elif g in ("TOMORROW", "FUTURE", "SOON"):
            tense = "future"
            time_marker = "Tomorrow, " if g == "TOMORROW" else ""
        elif g == "TODAY":
            time_marker = "Today, "
        else:
            filtered_glosses.append(g)

    # 5. Environmental subjects like WEATHER
    if "WEATHER" in normalized:
        for adj in PREDICATE_ADJECTIVES:
            if adj in normalized:
                return f"The weather is {adj.lower()}."

    # 6. Copula & Adjective / Noun Insertion
    subj = "I"
    verb = None
    has_negation = "NOT" in filtered_glosses or "NO" in filtered_glosses

    for g in filtered_glosses:
        if g in ("ME", "I"):
            subj = "I"
        elif g == "YOU":
            subj = "You"
        elif g == "HE":
            subj = "He"
        elif g == "SHE":
            subj = "She"
        elif g == "WE":
            subj = "We"
        elif g == "THEY":
            subj = "They"

    for adj in PREDICATE_ADJECTIVES:
        if adj in filtered_glosses:
            if tense == "past":
                if has_negation:
                    copula = "was not" if subj in ("I", "He", "She") else "were not"
                else:
                    copula = "was" if subj in ("I", "He", "She") else "were"
            elif tense == "future":
                copula = "will not be" if has_negation else "will be"
            else:
                if has_negation:
                    copula = "am not" if subj == "I" else "is not" if subj in ("He", "She") else "are not"
                else:
                    copula = "am" if subj == "I" else "is" if subj in ("He", "She") else "are"
            return f"{time_marker}{subj} {copula} {adj.lower()}."

    # 6. SVO Reconstruction with Motion Prepositions
    for v in VERB_CONJUGATIONS:
        if v in filtered_glosses:
            verb = v
            break

    if verb is not None:
        conj = VERB_CONJUGATIONS[verb]
        if has_negation:
            v_str = f"did not {conj['base']}" if tense == "past" else f"will not {conj['base']}" if tense == "future" else f"do not {conj['base']}"
        elif tense == "past":
            v_str = conj["past"]
        elif tense == "future":
            v_str = f"will {conj['base']}"
        else:
            v_str = conj["present"] if subj in ("He", "She") else conj["base"]

        # Find location/object
        obj_phrase = ""
        for loc in STANDARD_LOCATIONS:
            if loc in filtered_glosses:
                obj_phrase = f" to the {loc.lower()}" if verb in ("GO", "COME", "TRAVEL") else f" the {loc.lower()}"
                break

        for loc in ZERO_ARTICLE_LOCATIONS:
            if loc in filtered_glosses:
                obj_phrase = f" {loc.lower()}" if loc == "HOME" else f" to {loc.lower()}"
                break

        if "WATER" in filtered_glosses:
            obj_phrase = " water"
        elif "COFFEE" in filtered_glosses:
            obj_phrase = " coffee"
        elif "HELP" in filtered_glosses and verb != "HELP":
            obj_phrase = " help"

        return f"{time_marker}{subj} {v_str}{obj_phrase}."

    # Fallback capitalization
    cleaned = " ".join(g.lower() for g in normalized if g not in ("NOT", "NO"))
    return f"{cleaned.capitalize()}."


def draw_landmarks(frame: np.ndarray, extracted: Any) -> None:
    """Draw anatomical bones and joint circles over the video frame."""
    h, w, _ = frame.shape

    # 1. Pose connections (cyan/yellow)
    if extracted.pose is not None and len(extracted.pose) >= 17:
        for idx1, idx2 in POSE_CONNECTIONS:
            p1, p2 = extracted.pose[idx1], extracted.pose[idx2]
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)

        for idx in (11, 12, 13, 14, 15, 16):
            p = extracted.pose[idx]
            x, y = int(p[0] * w), int(p[1] * h)
            cv2.circle(frame, (x, y), 5, (0, 220, 255), -1)

    # 2. Left hand (neon green)
    if extracted.left_hand is not None:
        for u, v in HAND_CONNECTIONS:
            p1, p2 = extracted.left_hand[u], extracted.left_hand[v]
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)

        for p in extracted.left_hand:
            x, y = int(p[0] * w), int(p[1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # 3. Right hand (magenta)
    if extracted.right_hand is not None:
        for u, v in HAND_CONNECTIONS:
            p1, p2 = extracted.right_hand[u], extracted.right_hand[v]
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 50, 255), 2)

        for p in extracted.right_hand:
            x, y = int(p[0] * w), int(p[1] * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    buffer_len: int,
    window_size: int,
    last_detection: SignDetection | None,
    time_since_detection_ms: float,
    active_glosses: list[str],
    last_reconstructed_sentence: str,
    tts_client: KokoroTTSClient,
    tick: int,
) -> None:
    """Render telemetry diagnostics heads-up display and real-time subtitle cards."""
    h, w, _ = frame.shape

    # 1. Top Bar Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (15, 17, 23), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Telemetry text
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    tts_status = f"Kokoro TTS: {'SPEAKING' if tts_client.is_speaking else 'ONLINE'} ({tts_client.voice})" if tts_client.enabled else "TTS: DISABLED"
    tts_color = (0, 255, 100) if tts_client.is_speaking else (0, 200, 255) if tts_client.enabled else (120, 120, 120)
    cv2.putText(frame, tts_status, (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tts_color, 2)

    # Sliding window progress bar
    fill_ratio = min(buffer_len / max(window_size, 1), 1.0)
    bar_width = 160
    bar_x, bar_y = 16, 46
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 16), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * fill_ratio), bar_y + 16), (0, 200, 255), -1)
    cv2.putText(frame, f"Buffer: {buffer_len}/{window_size}", (bar_x + bar_width + 10, bar_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Top-right detection indicator
    if last_detection is not None and time_since_detection_ms < 2000:
        det_text = f"DETECTED: {last_detection.gloss.upper()} ({last_detection.confidence * 100:.0f}%)"
        text_size = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        rx = w - text_size[0] - 20
        cv2.putText(frame, det_text, (rx, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
    else:
        cv2.putText(frame, "Awaiting Sign Gestures...", (w - 240, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1)

    # 2. Bottom Subtitle Card Overlay
    sub_overlay = frame.copy()
    sub_height = 110
    cv2.rectangle(sub_overlay, (0, h - sub_height), (w, h), (12, 14, 20), -1)
    cv2.addWeighted(sub_overlay, 0.88, frame, 0.12, 0, frame)

    # Active gloss tokens line
    gloss_label = "STABILIZED GLOSSES: "
    cv2.putText(frame, gloss_label, (16, h - sub_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    gx = 16 + cv2.getTextSize(gloss_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    if active_glosses:
        for g in active_glosses[-6:]:
            g_str = f"[{g}]"
            cv2.putText(frame, g_str, (gx, h - sub_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            gx += cv2.getTextSize(g_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] + 8
    else:
        cv2.putText(frame, "(Waiting for gestural sequence...)", (gx, h - sub_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)

    # Reconstructed English sentence line
    sentence_label = "SPOKEN SENTENCE: "
    cv2.putText(frame, sentence_label, (16, h - sub_height + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    sx = 16 + cv2.getTextSize(sentence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]

    if last_reconstructed_sentence:
        sentence_str = f'"{last_reconstructed_sentence}"'
        cv2.putText(frame, sentence_str, (sx, h - sub_height + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "...", (sx, h - sub_height + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)

    # Audio Equalizer Animation while speaking
    if tts_client.is_speaking:
        eq_x = w - 110
        eq_y = h - sub_height + 65
        for i in range(5):
            bar_h = int(10 + 12 * np.sin(tick * 0.4 + i * 1.2))
            cv2.line(frame, (eq_x + i * 8, eq_y), (eq_x + i * 8, eq_y - bar_h), (0, 255, 100), 3)

    # Controls instruction footer
    controls = "[Q] Quit  [R] Reset  [S] Speak  [T] Toggle TTS  [1-5] Quick Demo Injections"
    cv2.putText(frame, controls, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)


def run_webcam_demo(
    engine: ASLVisionEngine,
    extractor: LandmarkExtractor,
    camera_id: int = 0,
    tts_client: KokoroTTSClient | None = None,
) -> None:
    """Capture live camera stream and run the complete Sign-to-Speech pipeline."""
    if tts_client is None:
        tts_client = KokoroTTSClient(enabled=True)

    stabilizer = PythonGlossStabilizer(min_confidence=0.50, debounce_window_ms=350.0, boundary_pause_ms=850.0)

    print(f"Opening camera index {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with index {camera_id}. Ensure webcam is connected.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("==================================================================")
    print("Converse Sign-to-Speech Live Pipeline Activated!")
    print("Pipeline: Webcam -> MediaPipe -> ST-GCN -> Stabilizer -> Kokoro TTS")
    print("Press [1-5] for quick test injections | [S] to force speak | [Q] to quit")
    print("==================================================================")

    window_name = "Converse ASL Sign-to-Speech Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps = 30.0
    last_detection: SignDetection | None = None
    last_detection_wall_time = 0.0
    last_reconstructed = ""
    tick = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream. Exiting.")
                break

            tick += 1
            current_time = time.perf_counter()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Flip for mirror interaction
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = current_time * 1000.0

            # 1. Extract 3D landmarks
            extracted = extractor.extract(frame_rgb, timestamp_ms=timestamp_ms)

            # 2. Vision perception engine inference
            detections = engine.process_landmarks(extracted, timestamp_ms=timestamp_ms)
            if detections:
                det = detections[0]
                last_detection = det
                last_detection_wall_time = current_time

                # 3. Stabilizer & Debouncer
                accepted, is_dup, gloss = stabilizer.process_detection(det)
                if accepted and not is_dup and gloss:
                    print(f"[Vision] Detected Sign: {gloss} ({det.confidence * 100:.1f}%)")

            # 4. Check for boundary pause flush (>850ms pause)
            flushed = stabilizer.check_boundary(timestamp_ms)
            if flushed:
                sentence = reconstruct_sentence(flushed)
                last_reconstructed = sentence
                print(f"[Reconstruction] Finalized Sentence: \"{sentence}\"")
                print(f"[TTS] Synthesizing speech via Kokoro ({tts_client.voice})...")
                tts_client.speak(sentence)

            # 5. Render HUD and Subtitles
            draw_landmarks(frame, extracted)
            time_since_ms = (current_time - last_detection_wall_time) * 1000.0
            draw_hud(
                frame=frame,
                fps=fps,
                buffer_len=len(engine.buffer),
                window_size=engine.config.window_size,
                last_detection=last_detection,
                time_since_detection_ms=time_since_ms,
                active_glosses=stabilizer.buffer,
                last_reconstructed_sentence=last_reconstructed,
                tts_client=tts_client,
                tick=tick,
            )

            cv2.imshow(window_name, frame)

            # 6. Key bindings
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                engine.reset()
                stabilizer.clear()
                last_reconstructed = ""
                print("Buffer cleared.")
            elif key in (ord("s"), ord("S")):
                if stabilizer.buffer:
                    flushed = stabilizer.flush()
                    sentence = reconstruct_sentence(flushed)
                    last_reconstructed = sentence
                    print(f"[Manual Trigger] Spoken: \"{sentence}\"")
                    tts_client.speak(sentence)
            elif key == ord("t"):
                tts_client.enabled = not tts_client.enabled
                print(f"Kokoro TTS audio output: {'ENABLED' if tts_client.enabled else 'DISABLED'}")
            elif key == ord("v"):
                voices = ["af_sky", "af_sarah", "am_adam", "bf_emma"]
                cur_idx = voices.index(tts_client.voice) if tts_client.voice in voices else 0
                tts_client.voice = voices[(cur_idx + 1) % len(voices)]
                print(f"Kokoro voice set to: {tts_client.voice}")
            # Quick Scenario Injections for deterministic testing
            elif key == ord("1"):
                sentence = reconstruct_sentence(["HELLO"])
                last_reconstructed = sentence
                print(f"[Sample 1] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("2"):
                sentence = reconstruct_sentence(["HELLO", "NICE", "MEET", "YOU"])
                last_reconstructed = sentence
                print(f"[Sample 2] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("3"):
                sentence = reconstruct_sentence(["THANK-YOU", "HELP"])
                last_reconstructed = sentence
                print(f"[Sample 3] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("4"):
                sentence = reconstruct_sentence(["BATHROOM", "WHERE"])
                last_reconstructed = sentence
                print(f"[Sample 4] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("5"):
                sentence = reconstruct_sentence(["YESTERDAY", "ME", "STORE", "GO"])
                last_reconstructed = sentence
                print(f"[Sample 5] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("Camera capture terminated.")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive live webcam demonstration of Converse ASL Sign-to-Speech Pipeline."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="af_sky",
        help="Kokoro voice style (e.g., af_sky, af_sarah, am_adam, bf_emma).",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS audio speech playback.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch ST-GCN .pt checkpoint.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="Minimum confidence threshold for sign detection emission.",
    )
    return parser.parse_args(args)


def ensure_mediapipe_model(target_dir: str = "models") -> Path:
    """Ensure Google MediaPipe Holistic Landmarker bundle is available, downloading if needed."""
    models_path = Path(target_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    task_file = models_path / "holistic_landmarker.task"
    if not task_file.is_file():
        url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
        print(f"Downloading official MediaPipe model (~13.6MB) to {task_file}...")
        urllib.request.urlretrieve(url, task_file)
        print("MediaPipe model asset downloaded.")
    return task_file


def main(args: Sequence[str] | None = None) -> None:
    """Entry point for live Sign-to-Speech demo."""
    parsed = parse_args(args)

    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=parsed.min_confidence,
        min_window_landmark_confidence=0.20,
    )

    model_path = ensure_mediapipe_model()
    extractor = LandmarkExtractor(model_path=model_path)
    engine = ASLVisionEngine(config=config, extractor=extractor)

    if parsed.checkpoint is not None:
        import torch
        ckpt = torch.load(parsed.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        if engine.pytorch_model is not None:
            engine.pytorch_model.load_state_dict(state_dict)
            print(f"Loaded ST-GCN checkpoint from: {parsed.checkpoint}")

    tts_client = KokoroTTSClient(voice=parsed.voice, enabled=not parsed.no_tts)
    run_webcam_demo(engine, extractor, camera_id=parsed.camera, tts_client=tts_client)


if __name__ == "__main__":
    main()
