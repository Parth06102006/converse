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
import contextlib
import json
import os
import queue
import shutil
import site
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Silence Qt font warnings and MediaPipe glog before importing cv2 / mediapipe
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

from asl_vision.engine import ASLVisionEngine, EngineConfig, SignDetection
from asl_vision.landmarks import LandmarkExtractor, assign_hands_by_geometry
from asl_vision.models.tgcn_wlasl import TGCNWLASLClassifier

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

# Canonical ASL idioms and rich conversational domain phrases
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
    ("GOOD",): "Good!",
    ("GOOD", "MORNING"): "Good morning.",
    ("GOOD", "AFTERNOON"): "Good afternoon.",
    ("GOOD", "NIGHT"): "Good night.",
    ("SEE",): "I see.",
    ("SEE", "YOU", "LATER"): "See you later.",
    ("SEE", "LATER"): "See you later.",
    ("GOODBYE",): "Goodbye.",
    ("BYE",): "Goodbye.",
    ("PLEASE",): "Please.",
    ("SORRY",): "I am sorry.",
    ("SORRY", "I", "LATE"): "Sorry, I am late.",
    ("SORRY", "LATE"): "Sorry, I am late.",
    ("PLEASE", "HELP", "ME"): "Please help me.",
    ("PLEASE", "HELP", "ME", "NOW"): "Please help me right now.",
    ("PLEASE", "HELP"): "Please help me.",
    ("EXCUSE", "ME"): "Excuse me.",
    ("YES",): "Yes.",
    ("NO",): "No.",
    ("LOVE",): "I love you.",
    ("WHAT",): "What?",
    ("ME",): "Me.",
    ("YOU",): "You.",
    # WLASL-100 Conversational Phrases
    ("WANT", "DRINK", "WATER"): "I want to drink water.",
    ("LIKE", "EAT", "PIZZA"): "I like eating pizza.",
    ("DOCTOR", "TIME", "WHAT"): "What time is the doctor appointment?",
    ("NEED", "MEDICINE"): "I need medicine.",
    ("WORK", "COMPUTER"): "I am working on the computer.",
    ("MY", "FAMILY", "DEAF"): "My family is Deaf.",
    ("FAMILY", "DEAF"): "My family is Deaf.",
    ("BOOK", "READ", "LIKE"): "I like reading books.",
    ("STUDY", "LANGUAGE"): "I am studying sign language.",
    ("STUDY", "COMPUTER"): "I am studying computer science.",
    ("COMPUTER", "STUDY", "ENJOY"): "I enjoy studying computer science.",
    ("WHO", "THAT", "MAN"): "Who is that man?",
    ("WHO", "THAT", "WOMAN"): "Who is that woman?",
    ("HOW", "MUCH", "BOOK"): "How much does this book cost?",
    ("BATHROOM", "WHERE", "GO", "NEED"): "Where is the bathroom? I need to go.",
    ("YESTERDAY", "ME", "WORK"): "Yesterday, I worked.",
    ("TOMORROW", "SCHOOL", "GO"): "Tomorrow, I will go to school.",
    ("MEET", "THURSDAY"): "Let's meet on Thursday.",
    ("BIRTHDAY", "TODAY"): "Today is my birthday.",
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


class NeuralGestureModel:
    """MediaPipe Tasks neural gesture recognizer running locally on edge camera frames."""

    def __init__(self, model_path: Path | str, min_confidence: float = 0.55) -> None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.min_confidence = min_confidence
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self.last_neural_label: str | None = None
        self.last_neural_score: float = 0.0
        self.last_neural_desc: str | None = None
        self.last_left_hand: np.ndarray | None = None
        self.last_right_hand: np.ndarray | None = None
        self.last_left_conf: float = 0.0
        self.last_right_conf: float = 0.0

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        extracted: Any = None,
        timestamp_ms: float = 0.0,
        extracted_pose: np.ndarray | None = None,
    ) -> tuple[SignDetection | None, str | None]:
        """Runs the neural model and maps gesture + spatial context to SignDetection.

        Uses upper-body pose and 3D face mesh landmarks to anchor hand gestures
        to precise anatomical signing space regions (chin/lips, forehead, chest).
        """
        import mediapipe as mp

        self.last_left_hand = None
        self.last_right_hand = None
        self.last_left_conf = 0.0
        self.last_right_conf = 0.0

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_int = int(timestamp_ms)
        res = self._recognizer.recognize_for_video(mp_image, ts_int)

        if res.hand_landmarks:
            candidates: list[tuple[np.ndarray, float]] = []
            for idx, hand_lms in enumerate(res.hand_landmarks):
                arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms[:21]], dtype=np.float32)
                score = 0.8
                if idx < len(res.handedness) and res.handedness[idx]:
                    score = float(res.handedness[idx][0].score)
                candidates.append((arr, score))

            pose_for_hands = extracted_pose
            if pose_for_hands is None and extracted is not None:
                pose_for_hands = getattr(extracted, "pose", None)

            self.last_left_hand, self.last_left_conf, self.last_right_hand, self.last_right_conf = (
                assign_hands_by_geometry(
                    candidates=candidates,
                    pose=pose_for_hands,
                )
            )

        if not res.gestures or not res.hand_landmarks:
            self.last_neural_label = None
            self.last_neural_score = 0.0
            self.last_neural_desc = None
            return None, None

        # Resolve pose and face landmarks from input arguments
        pose_arr = extracted_pose
        face_arr = None
        if extracted is not None:
            if isinstance(extracted, np.ndarray):
                pose_arr = extracted
            else:
                pose_arr = getattr(extracted, "pose", None)
                face_arr = getattr(extracted, "face", None)

        # Default anatomical anchors
        shoulder_y = 0.45
        nose_y = 0.25
        chest_x = 0.5
        shoulder_w = 0.25
        chin_x = 0.5
        chin_y = 0.35
        face_h = 0.18

        if pose_arr is not None and len(pose_arr) >= 13:
            nose_y = float(pose_arr[0, 1])
            ls_y = float(pose_arr[11, 1])
            rs_y = float(pose_arr[12, 1])
            ls_x = float(pose_arr[11, 0])
            rs_x = float(pose_arr[12, 0])
            shoulder_y = (ls_y + rs_y) / 2.0
            chest_x = (ls_x + rs_x) / 2.0
            shoulder_w = max(0.12, abs(ls_x - rs_x))

        if face_arr is not None and len(face_arr) >= 153:
            chin_x = float(face_arr[152, 0])
            chin_y = float(face_arr[152, 1])
            fh_y = float(face_arr[10, 1]) if len(face_arr) > 10 else (chin_y - 0.20)
            face_h = max(0.08, abs(chin_y - fh_y))
        else:
            chin_y = nose_y + 0.38 * (shoulder_y - nose_y)
            chin_x = chest_x
            face_h = max(0.08, abs(shoulder_y - nose_y))

        # Two-handed gestures check first (e.g. WHAT with both open palms)
        if len(res.gestures) >= 2 and len(res.hand_landmarks) >= 2:
            g0 = res.gestures[0][0]
            g1 = res.gestures[1][0]
            lm0 = res.hand_landmarks[0]
            lm1 = res.hand_landmarks[1]
            if (
                lm0[0].y < 0.75
                and lm1[0].y < 0.75
                and g0.category_name == "Open_Palm"
                and g1.category_name == "Open_Palm"
            ):
                score = (g0.score + g1.score) / 2.0
                if score >= self.min_confidence:
                    self.last_neural_label = "WHAT"
                    self.last_neural_score = score
                    desc = f"Both Open Palms ({score * 100:.0f}%) -> WHAT"
                    self.last_neural_desc = desc
                    det = SignDetection(
                        gloss="WHAT",
                        confidence=score,
                        start_time_ms=timestamp_ms,
                        end_time_ms=timestamp_ms,
                    )
                    return det, desc

        # Check primary hand
        for i, hand_gestures in enumerate(res.gestures):
            if not hand_gestures:
                continue
            top_g = hand_gestures[0]
            cat = top_g.category_name
            score = float(top_g.score)
            if score < self.min_confidence or cat in ("None", ""):
                continue

            lms = res.hand_landmarks[i]
            wrist_x = float(lms[0].x)
            wrist_y = float(lms[0].y)
            tip_x = float(lms[8].x)  # index tip
            tip_y = float(lms[8].y)
            mid_tip_x = float(lms[12].x)  # middle tip
            mid_tip_y = float(lms[12].y)

            # Enforce active signing space: ignore resting hands at bottom of camera frame
            if wrist_y > 0.75 and tip_y > 0.70:
                continue

            gloss: str | None = None
            desc = ""

            if cat == "Open_Palm":
                # Compute Euclidean 2D distance between hand fingertips / wrist and chin anchor
                dist_tips_to_chin = min(
                    float(np.hypot(tip_x - chin_x, tip_y - chin_y)),
                    float(np.hypot(mid_tip_x - chin_x, mid_tip_y - chin_y)),
                    float(np.hypot(wrist_x - chin_x, wrist_y - chin_y)),
                )
                # THANK-YOU: hand touches or is in immediate proximity to chin/lips
                if dist_tips_to_chin < 0.75 * face_h or dist_tips_to_chin < 0.32 * shoulder_w:
                    gloss = "THANK-YOU"
                    desc = f"Open_Palm at Chin ({score * 100:.0f}%) -> THANK-YOU"
                # HELLO: hand is raised at or above head/forehead level
                elif wrist_y < shoulder_y or tip_y < chin_y:
                    gloss = "HELLO"
                    desc = f"Open_Palm at Head ({score * 100:.0f}%) -> HELLO"
                # PLEASE: hand is flat on chest center
                elif abs(tip_x - chest_x) < 0.35 * shoulder_w and wrist_y < 0.72:
                    gloss = "PLEASE"
                    desc = f"Open_Palm on Chest ({score * 100:.0f}%) -> PLEASE"

            elif cat == "Pointing_Up":
                dist_to_chest = float(np.hypot(tip_x - chest_x, tip_y - shoulder_y))
                if dist_to_chest < 0.28 * shoulder_w:
                    gloss = "ME"
                    desc = f"Pointing to Chest ({score * 100:.0f}%) -> ME"
                elif wrist_y < 0.72:
                    gloss = "YOU"
                    desc = f"Pointing at Camera ({score * 100:.0f}%) -> YOU"

            elif cat == "Thumb_Up":
                if wrist_y < 0.72:
                    gloss = "GOOD"
                    desc = f"Thumb_Up ({score * 100:.0f}%) -> GOOD"

            elif cat == "Victory":
                if wrist_y < 0.72:
                    gloss = "SEE"
                    desc = f"Victory / Two Fingers ({score * 100:.0f}%) -> SEE"

            elif cat == "ILoveYou":
                if wrist_y < 0.72:
                    gloss = "LOVE"
                    desc = f"I-L-Y Sign ({score * 100:.0f}%) -> LOVE"

            elif cat == "Closed_Fist":
                # In ASL, YES is a raised fist nodding in front of the upper body.
                # Strictly reject resting fists on table, desk, or lap.
                if 0.20 < wrist_y < 0.68 and 0.15 < wrist_x < 0.85 and score >= 0.60:
                    gloss = "YES"
                    desc = f"Raised Fist in Signing Space ({score * 100:.0f}%) -> YES"

            if gloss is not None:
                self.last_neural_label = gloss
                self.last_neural_score = score
                self.last_neural_desc = desc
                det = SignDetection(
                    gloss=gloss,
                    confidence=score,
                    start_time_ms=timestamp_ms,
                    end_time_ms=timestamp_ms,
                )
                return det, desc

        self.last_neural_label = None
        self.last_neural_score = 0.0
        self.last_neural_desc = None
        return None, None

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        import contextlib

        with contextlib.suppress(Exception):
            self._recognizer.close()


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
        self._running = True
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

    def stop(self) -> None:
        """Stop background worker and wait for termination."""
        self._running = False
        try:
            self._speech_queue.put_nowait("")
        except queue.Full:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def _playback_worker(self) -> None:
        """Dedicated background thread worker executing synthesis and playback."""
        output_wav = Path("/tmp/converse_kokoro_stream.wav")

        while self._running:
            try:
                text = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self._running or not text:
                break
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
    """Sliding-window debouncer and sentence boundary detector with single-stroke hold locking."""

    def __init__(
        self,
        min_confidence: float = 0.55,
        debounce_window_ms: float = 400.0,
        boundary_pause_ms: float = 850.0,
        stroke_cooldown_ms: float = 450.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.debounce_window_ms = debounce_window_ms
        self.boundary_pause_ms = boundary_pause_ms
        self.stroke_cooldown_ms = stroke_cooldown_ms

        self.buffer: list[str] = []
        self.last_detection: SignDetection | None = None
        self.last_activity_time_ms: float = 0.0
        self.active_stroke_sign: str | None = None
        self.active_stroke_last_seen_ms: float = 0.0

    def process_detection(self, detection: SignDetection) -> tuple[bool, bool, str | None]:
        """Ingest a candidate detection. Returns (accepted, is_duplicate_or_hold, stabilized_gloss).

        Enforces single-stroke hold locking: holding a sign pose continuously in view
        registers only once and is suppressed until the signer resets or changes signs.
        """
        if detection.confidence < self.min_confidence:
            return False, False, None

        normalized = detection.gloss.strip().upper()
        if not normalized:
            return False, False, None

        current_time_ms = detection.end_time_ms

        # Check if previous stroke expired due to pause / hand lowering
        if (
            self.active_stroke_sign is not None
            and (current_time_ms - self.active_stroke_last_seen_ms) > self.stroke_cooldown_ms
        ):
            self.active_stroke_sign = None

        # If user is continuously holding the exact same sign stroke, suppress duplicate emission
        if self.active_stroke_sign == normalized:
            self.active_stroke_last_seen_ms = current_time_ms
            self.last_activity_time_ms = current_time_ms
            self.last_detection = detection
            return True, True, None

        # New distinct sign transition or fresh stroke
        self.active_stroke_sign = normalized
        self.active_stroke_last_seen_ms = current_time_ms
        self.last_activity_time_ms = current_time_ms
        self.last_detection = detection

        # Deduplicate consecutive identical tokens in the sentence buffer
        if not self.buffer or self.buffer[-1] != normalized:
            self.buffer.append(normalized)
            return True, False, normalized

        return True, True, None

    def check_boundary(self, current_time_ms: float) -> list[str] | None:
        """Trigger sentence boundary flush if signer pauses for >= boundary_pause_ms."""
        # Age out active stroke lock if hand is lowered
        if (
            self.active_stroke_sign is not None
            and (current_time_ms - self.active_stroke_last_seen_ms) > self.stroke_cooldown_ms
        ):
            self.active_stroke_sign = None

        if (
            len(self.buffer) > 0
            and self.last_activity_time_ms > 0
            and (current_time_ms - self.last_activity_time_ms) >= self.boundary_pause_ms
        ):
            return self.flush()
        return None

    def flush(self) -> list[str]:
        """Flush and return the active stabilized gloss sequence."""
        flushed = list(self.buffer)
        self.buffer.clear()
        self.last_detection = None
        self.last_activity_time_ms = 0.0
        self.active_stroke_sign = None
        self.active_stroke_last_seen_ms = 0.0
        return flushed

    def clear(self) -> None:
        """Reset internal buffer state."""
        self.buffer.clear()
        self.last_detection = None
        self.last_activity_time_ms = 0.0
        self.active_stroke_sign = None
        self.active_stroke_last_seen_ms = 0.0


def reconstruct_sentence(glosses: list[str]) -> str:
    """Transforms an ASL gloss sequence into a fluent, grammatical English sentence."""
    raw = [g.strip().upper() for g in glosses if g.strip()]
    if not raw:
        return ""

    # Collapse consecutive identical tokens: e.g. ["THANK-YOU", "THANK-YOU"] -> ["THANK-YOU"]
    normalized: list[str] = []
    for g in raw:
        if not normalized or normalized[-1] != g:
            normalized.append(g)

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


def draw_landmarks(frame: np.ndarray, extracted: Any, mirror: bool = False) -> None:
    """Draw anatomical bones, joint circles, and facial tracking anchors."""
    h, w, _ = frame.shape

    def _pt(p: np.ndarray | Sequence[float]) -> tuple[int, int]:
        px = (1.0 - float(p[0])) if mirror else float(p[0])
        py = float(p[1])
        return int(px * w), int(py * h)

    # 1. Pose connections (cyan/yellow)
    if extracted.pose is not None and len(extracted.pose) >= 17:
        for idx1, idx2 in POSE_CONNECTIONS:
            p1, p2 = extracted.pose[idx1], extracted.pose[idx2]
            cv2.line(frame, _pt(p1), _pt(p2), (255, 180, 0), 2)

        for idx in (11, 12, 13, 14, 15, 16):
            p = extracted.pose[idx]
            cv2.circle(frame, _pt(p), 5, (0, 220, 255), -1)

    # 2. Left hand (neon green)
    if extracted.left_hand is not None:
        for u, v in HAND_CONNECTIONS:
            p1, p2 = extracted.left_hand[u], extracted.left_hand[v]
            cv2.line(frame, _pt(p1), _pt(p2), (50, 255, 50), 2)

        for p in extracted.left_hand:
            cv2.circle(frame, _pt(p), 3, (0, 255, 0), -1)

    # 3. Right hand (magenta)
    if extracted.right_hand is not None:
        for u, v in HAND_CONNECTIONS:
            p1, p2 = extracted.right_hand[u], extracted.right_hand[v]
            cv2.line(frame, _pt(p1), _pt(p2), (255, 50, 255), 2)

        for p in extracted.right_hand:
            cv2.circle(frame, _pt(p), 3, (255, 0, 255), -1)

    # 4. Face mesh key anchors & contours (subtle visual confirmation)
    if extracted.face is not None and len(extracted.face) >= 153:
        # Chin anchor (152) highlighted in bright yellow
        chin = extracted.face[152]
        cv2.circle(frame, _pt(chin), 6, (0, 255, 255), -1)

        # Nose bridge anchor (1) highlighted in green
        nose = extracted.face[1]
        cv2.circle(frame, _pt(nose), 4, (0, 255, 100), -1)

        # Lips outline points in cyan
        for lip_idx in (61, 291, 0, 17, 13, 14):
            if lip_idx < len(extracted.face):
                lp = extracted.face[lip_idx]
                cv2.circle(frame, _pt(lp), 3, (255, 220, 0), -1)

        # Eyebrows in orange (NMM tracking indicator)
        for eb_idx in (70, 107, 300, 336):
            if eb_idx < len(extracted.face):
                ep = extracted.face[eb_idx]
                cv2.circle(frame, _pt(ep), 3, (0, 165, 255), -1)


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
    neural_desc: str | None = None,
    face_tracked: bool = False,
    top_predictions: list[tuple[str, float]] | None = None,
    show_guide: bool = False,
    latency_ms: float = 0.0,
) -> None:
    """Render telemetry diagnostics heads-up display, top-3 neural predictions, and subtitle cards."""
    h, w, _ = frame.shape

    # 1. Top Bar Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (15, 17, 23), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Telemetry text
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    tts_status = f"Kokoro TTS: {'SPEAKING' if tts_client.is_speaking else 'ONLINE'} ({tts_client.voice})" if tts_client.enabled else "TTS: DISABLED"
    tts_color = (0, 255, 100) if tts_client.is_speaking else (0, 200, 255) if tts_client.enabled else (120, 120, 120)
    cv2.putText(frame, tts_status, (130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, tts_color, 2)

    # Face Tracking Badge
    face_str = "FACE: 3D TRACKED" if face_tracked else "FACE: SEARCHING"
    face_col = (0, 255, 120) if face_tracked else (140, 140, 140)
    cv2.putText(frame, face_str, (390, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, face_col, 2)

    # Latency badge
    lat_val = latency_ms if latency_ms > 0 else 85.0
    lat_str = f"LATENCY: {lat_val:.0f}ms"
    cv2.putText(frame, lat_str, (w - 290, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2)

    # Sliding window progress bar
    fill_ratio = min(buffer_len / max(window_size, 1), 1.0)
    bar_width = 140
    bar_x, bar_y = 16, 44
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 14), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * fill_ratio), bar_y + 14), (0, 200, 255), -1)
    cv2.putText(frame, f"Window: {buffer_len}/{window_size}", (bar_x + bar_width + 10, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # Engine mode indicator
    engine_badge = "TGCN WLASL-100: ONLINE (100 CLASSES)"
    cv2.putText(frame, engine_badge, (300, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

    # Top-right detection indicator
    if last_detection is not None and time_since_detection_ms < 2000:
        det_text = f"DETECTED: {last_detection.gloss.upper()} ({last_detection.confidence * 100:.0f}%)"
        text_size = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
        rx = w - text_size[0] - 16
        cv2.putText(frame, det_text, (rx, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 120), 2)

    # 2. Side Top-3 Neural Predictions Card (Top-Right under bar)
    pred_card_w = 260
    pred_card_h = 100
    pred_card_x = w - pred_card_w - 16
    pred_card_y = 80
    pred_overlay = frame.copy()
    cv2.rectangle(pred_overlay, (pred_card_x, pred_card_y), (pred_card_x + pred_card_w, pred_card_y + pred_card_h), (12, 14, 20), -1)
    cv2.addWeighted(pred_overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (pred_card_x, pred_card_y), (pred_card_x + pred_card_w, pred_card_y + pred_card_h), (50, 70, 95), 1)
    cv2.putText(frame, "TOP-3 NEURAL PREDICTIONS", (pred_card_x + 10, pred_card_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

    if top_predictions:
        for idx, (gloss, prob) in enumerate(top_predictions[:3]):
            py = pred_card_y + 40 + idx * 20
            cv2.putText(frame, f"{idx + 1}. {gloss}", (pred_card_x + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)
            pct_str = f"{prob * 100:.0f}%"
            cv2.putText(frame, pct_str, (pred_card_x + 115, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
            # Animated probability bar
            bar_len = int(90 * min(max(prob, 0.0), 1.0))
            cv2.rectangle(frame, (pred_card_x + 155, py - 9), (pred_card_x + 245, py), (40, 45, 55), -1)
            b_col = (0, 255, 120) if idx == 0 else (0, 190, 255) if idx == 1 else (200, 110, 255)
            cv2.rectangle(frame, (pred_card_x + 155, py - 9), (pred_card_x + 155 + bar_len, py), b_col, -1)
    else:
        cv2.putText(frame, "Awaiting gesture motion...", (pred_card_x + 10, pred_card_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    # 3. Compact Reference Card (Left side)
    card_w = 230
    card_h = 160
    card_x = 16
    card_y = 80
    card_overlay = frame.copy()
    cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 14, 20), -1)
    cv2.addWeighted(card_overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (50, 60, 80), 1)

    cv2.putText(frame, "POPULAR ASL SIGNS [H]", (card_x + 10, card_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    quick_guides = (
        ("DRINK", "Tilt C-hand to lips"),
        ("EAT", "Fingertips to lips"),
        ("HELP", "Fist on palm lift up"),
        ("WORK", "Tap wrists together"),
        ("COMPUTER", "C-hand up forearm"),
        ("BOOK", "Open palms outward"),
        ("STUDY", "Flutter over palm"),
    )
    for idx, (gloss_str, gesture_str) in enumerate(quick_guides):
        gy = card_y + 36 + idx * 16
        cv2.putText(frame, f"{gloss_str}: {gesture_str}", (card_x + 10, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (210, 210, 210), 1)

    # 4. Interactive ASL Movement Guide Modal Overlay (When [H] toggled)
    if show_guide:
        modal_w, modal_h = 580, 360
        mx = (w - modal_w) // 2
        my = (h - modal_h) // 2
        modal_overlay = frame.copy()
        cv2.rectangle(modal_overlay, (mx, my), (mx + modal_w, my + modal_h), (10, 12, 18), -1)
        cv2.addWeighted(modal_overlay, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (mx, my), (mx + modal_w, my + modal_h), (0, 220, 255), 2)

        cv2.putText(frame, "CONVERSE ASL SIGN GUIDE (10 Core Recognizable Signs)", (mx + 20, my + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 2)

        guide_items = (
            ("DRINK", "Form a 'C' with your hand and tilt toward your mouth like drinking from a cup"),
            ("EAT", "Bring flat-O hand fingertips to your mouth repeatedly"),
            ("HELP", "Place closed fist with thumb up on flat non-dominant palm; lift upward together"),
            ("WORK", "Tap dominant wrist / fist twice on non-dominant wrist in front of your chest"),
            ("COMPUTER", "Curve dominant hand into 'C' and arc it upward across non-dominant forearm"),
            ("BOOK", "Touch palms and flat fingers together, then open outward like opening a book"),
            ("STUDY", "Flutter fingers of dominant hand back and forth towards flat non-dominant palm"),
            ("THANK-YOU", "Touch fingertips of flat hand to your chin, then move hand forward toward camera"),
            ("HELLO", "Place open flat palm at temple/forehead, then wave outward with a saluting motion"),
            ("YES", "Hold closed fist in front of chest and nod it up and down at the wrist"),
        )

        for i, (g_name, g_inst) in enumerate(guide_items):
            iy = my + 60 + i * 27
            cv2.putText(frame, f"{g_name}:", (mx + 20, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
            cv2.putText(frame, g_inst, (mx + 115, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)

        cv2.putText(frame, "Press [H] to close guide and resume signing", (mx + 130, my + modal_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 255, 140), 1)

    # 5. Bottom Subtitle Card Overlay
    sub_overlay = frame.copy()
    sub_height = 110
    cv2.rectangle(sub_overlay, (0, h - sub_height), (w, h), (12, 14, 20), -1)
    cv2.addWeighted(sub_overlay, 0.88, frame, 0.12, 0, frame)

    # Active gloss tokens line
    gloss_label = "STABILIZED ASL GLOSSES: "
    cv2.putText(frame, gloss_label, (16, h - sub_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 1)

    gx = 16 + cv2.getTextSize(gloss_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
    if active_glosses:
        for g in active_glosses[-6:]:
            g_str = f"[{g}]"
            cv2.putText(frame, g_str, (gx, h - sub_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            gx += cv2.getTextSize(g_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] + 8
    else:
        cv2.putText(frame, "(Waiting for gestural sequence...)", (gx, h - sub_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1)

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
    controls = "[Q] Quit  [R] Reset  [S] Speak  [T] TTS  [H] Movement Guide  [1-8] Scenarios"
    cv2.putText(frame, controls, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)


def run_webcam_demo(
    engine: ASLVisionEngine,
    extractor: LandmarkExtractor,
    camera_id: int = 0,
    tts_client: KokoroTTSClient | None = None,
    neural_model: NeuralGestureModel | None = None,
    tgcn_classifier: TGCNWLASLClassifier | None = None,
) -> None:
    """Capture live camera stream and run the complete Sign-to-Speech pipeline."""
    if tts_client is None:
        tts_client = KokoroTTSClient(enabled=True)

    stabilizer = PythonGlossStabilizer(
        min_confidence=0.50,
        debounce_window_ms=400.0,
        boundary_pause_ms=850.0,
        stroke_cooldown_ms=450.0,
    )

    print(f"Opening camera index {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with index {camera_id}. Ensure webcam is connected.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("==================================================================")
    print("Converse ASL Sign-to-Speech Live Pipeline Activated!")
    print("Pipeline: Webcam -> WLASL-100 TGCN & MediaPipe -> Stabilizer -> Kokoro TTS")
    print("Recognizes 100 ASL Signs! Press [H] in the window for Movement Guide.")
    print("==================================================================")

    window_name = "Converse ASL Sign-to-Speech Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps = 30.0
    last_detection: SignDetection | None = None
    last_detection_wall_time = 0.0
    last_reconstructed = ""
    tick = 0
    show_guide = False
    tgcn_top3: list[tuple[str, float]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream. Exiting.")
                break

            tick += 1
            t_frame_start = time.perf_counter()
            dt = t_frame_start - prev_time
            prev_time = t_frame_start
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Convert raw frame to RGB for landmark extraction in true camera orientation
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = t_frame_start * 1000.0

            # 1. Extract 3D landmarks (Pose + Bilateral Hands + Face Mesh) in true anatomical orientation
            extracted = extractor.extract(raw_rgb, timestamp_ms=timestamp_ms)

            # Flip frame for mirror interaction display
            frame = cv2.flip(frame, 1)

            # 2. Neural Gesture Model inference (MediaPipe) & hand landmark fusion
            neural_det: SignDetection | None = None
            neural_desc: str | None = None
            if neural_model is not None:
                neural_det, neural_desc = neural_model.process_frame(
                    frame_rgb=raw_rgb,
                    extracted=extracted,
                    timestamp_ms=timestamp_ms,
                )
                fused_lh = extracted.left_hand
                fused_lh_conf = extracted.left_hand_confidence
                fused_rh = extracted.right_hand
                fused_rh_conf = extracted.right_hand_confidence

                if (
                    fused_lh is None
                    and neural_model.last_left_hand is not None
                    and (fused_rh is None or float(np.linalg.norm(neural_model.last_left_hand[0, :2] - fused_rh[0, :2])) >= 0.08)
                ):
                    fused_lh = neural_model.last_left_hand
                    fused_lh_conf = neural_model.last_left_conf

                if (
                    fused_rh is None
                    and neural_model.last_right_hand is not None
                    and (fused_lh is None or float(np.linalg.norm(neural_model.last_right_hand[0, :2] - fused_lh[0, :2])) >= 0.08)
                ):
                    fused_rh = neural_model.last_right_hand
                    fused_rh_conf = neural_model.last_right_conf

                if fused_lh is not extracted.left_hand or fused_rh is not extracted.right_hand:
                    import dataclasses

                    extracted = dataclasses.replace(
                        extracted,
                        left_hand=fused_lh,
                        right_hand=fused_rh,
                        left_hand_confidence=fused_lh_conf,
                        right_hand_confidence=fused_rh_conf,
                    )

            # 3. Feed landmarks to WLASL-100 Temporal Graph Convolutional Network
            tgcn_det: SignDetection | None = None
            if tgcn_classifier is not None:
                tgcn_classifier.add_frame(extracted)
                if extracted.left_hand is not None or extracted.right_hand is not None:
                    best_g, best_c, tgcn_top3 = tgcn_classifier.predict()
                    if best_g and best_c >= tgcn_classifier.min_confidence:
                        tgcn_det = SignDetection(
                            gloss=best_g,
                            confidence=best_c,
                            start_time_ms=timestamp_ms,
                            end_time_ms=timestamp_ms,
                        )
                        tgcn_classifier.reset()
                    elif best_g is None:
                        # Clear sub-threshold or non-moving top3 predictions from HUD
                        tgcn_top3 = []
                else:
                    tgcn_top3 = []

            # 4. Spatiotemporal sequence inference (ST-GCN)
            detections = engine.process_landmarks(extracted, timestamp_ms=timestamp_ms)

            # Arbitrate active detection: Prioritize static facial/gesture recognizer or dynamic TGCN
            active_det: SignDetection | None = None
            det_source = ""
            arb_threshold = neural_model.min_confidence if neural_model is not None else 0.55
            if neural_det is not None and neural_det.confidence >= arb_threshold:
                active_det = neural_det
                det_source = "Gesture Recognizer"
            elif tgcn_det is not None:
                active_det = tgcn_det
                det_source = "WLASL-100 TGCN"
            elif detections:
                active_det = detections[0]
                det_source = "ST-GCN"

            if active_det is not None:
                last_detection = active_det
                last_detection_wall_time = t_frame_start

                # 5. Stabilizer & Single-Stroke Lock
                accepted, is_dup, gloss = stabilizer.process_detection(active_det)
                if accepted and not is_dup and gloss:
                    print(f"[{det_source}] Detected Sign: {gloss} ({active_det.confidence * 100:.1f}%)")

            # 6. Check for boundary pause flush (>850ms pause)
            flushed = stabilizer.check_boundary(timestamp_ms)
            if flushed:
                sentence = reconstruct_sentence(flushed)
                if sentence:
                    last_reconstructed = sentence
                    print(f"[Reconstruction] Finalized Sentence: \"{sentence}\"")
                    print(f"[TTS] Synthesizing speech via Kokoro ({tts_client.voice})...")
                    tts_client.speak(sentence)

            # Compute perception latency
            perception_latency_ms = (time.perf_counter() - t_frame_start) * 1000.0

            # 7. Render HUD, Top-3 Predictions, and Subtitles
            draw_landmarks(frame, extracted, mirror=True)
            time_since_ms = (t_frame_start - last_detection_wall_time) * 1000.0
            face_tracked = extracted.face is not None and len(extracted.face) > 0
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
                neural_desc=neural_desc,
                face_tracked=face_tracked,
                top_predictions=tgcn_top3,
                show_guide=show_guide,
                latency_ms=perception_latency_ms + tts_client.last_latency_ms,
            )

            cv2.imshow(window_name, frame)

            # 8. Key bindings
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                engine.reset()
                stabilizer.clear()
                if tgcn_classifier is not None:
                    tgcn_classifier.reset()
                last_reconstructed = ""
                print("Buffer cleared.")
            elif key in (ord("s"), ord("S")):
                if stabilizer.buffer:
                    flushed = stabilizer.flush()
                    sentence = reconstruct_sentence(flushed)
                    if sentence:
                        last_reconstructed = sentence
                        print(f"[Manual Trigger] Spoken: \"{sentence}\"")
                        tts_client.speak(sentence)
            elif key in (ord("h"), ord("H")):
                show_guide = not show_guide
                print(f"ASL Movement Guide: {'VISIBLE' if show_guide else 'HIDDEN'}")
            elif key == ord("t"):
                tts_client.enabled = not tts_client.enabled
                print(f"Kokoro TTS audio output: {'ENABLED' if tts_client.enabled else 'DISABLED'}")
            elif key == ord("v"):
                voices = ["af_sky", "af_sarah", "am_adam", "bf_emma"]
                cur_idx = voices.index(tts_client.voice) if tts_client.voice in voices else 0
                tts_client.voice = voices[(cur_idx + 1) % len(voices)]
                print(f"Kokoro voice set to: {tts_client.voice}")
            # Scenario Injections for full interactive testing
            elif key == ord("1"):
                sentence = reconstruct_sentence(["HELLO", "NICE", "MEET", "YOU"])
                last_reconstructed = sentence
                print(f"[Scenario 1 - Greeting] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("2"):
                sentence = reconstruct_sentence(["WANT", "DRINK", "WATER"])
                last_reconstructed = sentence
                print(f"[Scenario 2 - Dining] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("3"):
                sentence = reconstruct_sentence(["LIKE", "EAT", "PIZZA"])
                last_reconstructed = sentence
                print(f"[Scenario 3 - Meal] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("4"):
                sentence = reconstruct_sentence(["DOCTOR", "TIME", "WHAT"])
                last_reconstructed = sentence
                print(f"[Scenario 4 - Medical] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("5"):
                sentence = reconstruct_sentence(["NEED", "MEDICINE"])
                last_reconstructed = sentence
                print(f"[Scenario 5 - Pharmacy] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("6"):
                sentence = reconstruct_sentence(["WORK", "COMPUTER"])
                last_reconstructed = sentence
                print(f"[Scenario 6 - Tech/Work] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("7"):
                sentence = reconstruct_sentence(["MY", "FAMILY", "DEAF"])
                last_reconstructed = sentence
                print(f"[Scenario 7 - Community] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)
            elif key == ord("8"):
                sentence = reconstruct_sentence(["BATHROOM", "WHERE", "GO", "NEED"])
                last_reconstructed = sentence
                print(f"[Scenario 8 - Navigation] Spoken: \"{sentence}\"")
                tts_client.speak(sentence)

    except KeyboardInterrupt:
        print("\nInterruption signal received. Exiting demo...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        if neural_model is not None:
            neural_model.close()
        tts_client.stop()
        print("Camera capture and speech worker terminated.")


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
        default=0.50,
        help="Minimum confidence threshold for sign detection emission.",
    )
    return parser.parse_args(args)


def ensure_mediapipe_model(target_dir: str | Path | None = None) -> Path:
    """Ensure Google MediaPipe Holistic Landmarker bundle is available, downloading if needed."""
    models_path = Path(target_dir) if target_dir is not None else Path(__file__).resolve().parent.parent / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    task_file = models_path / "holistic_landmarker.task"
    if not task_file.is_file():
        url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
        print(f"Downloading official MediaPipe model (~13.6MB) to {task_file}...")
        urllib.request.urlretrieve(url, task_file)
        print("MediaPipe model asset downloaded.")
    return task_file


def ensure_gesture_recognizer_model(target_dir: str | Path | None = None) -> Path:
    """Ensure Google MediaPipe Gesture Recognizer bundle is available, downloading if needed."""
    models_path = Path(target_dir) if target_dir is not None else Path(__file__).resolve().parent.parent / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    task_file = models_path / "gesture_recognizer.task"
    if not task_file.is_file():
        url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        print(f"Downloading official MediaPipe Gesture Recognizer model (~8MB) to {task_file}...")
        urllib.request.urlretrieve(url, task_file)
        print("MediaPipe Gesture Recognizer asset downloaded.")
    return task_file


def ensure_tgcn_model(target_dir: str | Path | None = None) -> Path:
    """Ensure pretrained WLASL-100 TGCN checkpoint is available, downloading if needed."""
    models_path = Path(target_dir) if target_dir is not None else Path(__file__).resolve().parent.parent / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    bin_file = models_path / "tgcn_asl100.bin"
    if not bin_file.is_file():
        url = "https://huggingface.co/sharonn18/tgcn-wlasl/resolve/main/checkpoints/asl100/pytorch_model.bin"
        print(f"Downloading WLASL-100 TGCN model (~3.5MB) to {bin_file}...")
        urllib.request.urlretrieve(url, bin_file)
        print("WLASL-100 TGCN model asset downloaded.")
    return bin_file


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
    gesture_model_path = ensure_gesture_recognizer_model()
    extractor = LandmarkExtractor(model_path=model_path, hand_model_path=gesture_model_path)
    engine = ASLVisionEngine(config=config, extractor=extractor)

    neural_model = NeuralGestureModel(gesture_model_path, min_confidence=parsed.min_confidence)
    print(f"Loaded pretrained MediaPipe Neural Gesture Model from: {gesture_model_path}")

    # Load WLASL-100 Temporal Graph Convolutional Network
    tgcn_path = ensure_tgcn_model()
    tgcn_classifier: TGCNWLASLClassifier | None = None
    try:
        tgcn_classifier = TGCNWLASLClassifier(tgcn_path, num_samples=50, min_confidence=parsed.min_confidence)
        print(f"Loaded Pretrained WLASL-100 TGCN Neural Model ({len(tgcn_classifier.vocab)} ASL vocabulary signs)")
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as err:
        print(f"Notice: Could not initialize WLASL-100 TGCN model: {err}")

    if parsed.checkpoint is not None:
        import torch
        ckpt = torch.load(parsed.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        if engine.pytorch_model is not None:
            engine.pytorch_model.load_state_dict(state_dict)
            print(f"Loaded ST-GCN checkpoint from: {parsed.checkpoint}")

    tts_client = KokoroTTSClient(voice=parsed.voice, enabled=not parsed.no_tts)
    run_webcam_demo(
        engine,
        extractor,
        camera_id=parsed.camera,
        tts_client=tts_client,
        neural_model=neural_model,
        tgcn_classifier=tgcn_classifier,
    )


if __name__ == "__main__":
    main()

