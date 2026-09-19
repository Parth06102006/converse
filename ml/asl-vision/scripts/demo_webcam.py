"""Interactive real-time webcam demonstration for ASL Vision perception pipeline."""

from __future__ import annotations

import argparse
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from asl_vision.engine import ASLVisionEngine, EngineConfig, SignDetection
from asl_vision.landmarks import LandmarkExtractor

# Landmark bone connections for visual overlay
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


def draw_landmarks(
    frame: np.ndarray,
    extracted: Any,
) -> None:
    """Draw anatomical bones and joint circles over the video frame."""
    h, w, _ = frame.shape

    # 1. Draw upper body pose bones
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

    # 2. Draw left hand (green)
    if extracted.left_hand is not None:
        for u, v in HAND_CONNECTIONS:
            p1, p2 = extracted.left_hand[u], extracted.left_hand[v]
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)

        for p in extracted.left_hand:
            x, y = int(p[0] * w), int(p[1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # 3. Draw right hand (magenta)
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
) -> None:
    """Render telemetry diagnostics heads-up display over the frame."""
    # Top semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # FPS counter
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 200),
        2,
    )

    # Buffer progress bar
    fill_ratio = min(buffer_len / max(window_size, 1), 1.0)
    bar_width = 180
    bar_x, bar_y = 15, 45
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 16), (60, 60, 60), -1)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_width * fill_ratio), bar_y + 16),
        (0, 200, 255),
        -1,
    )
    cv2.putText(
        frame,
        f"Buffer: {buffer_len}/{window_size}",
        (bar_x + bar_width + 12, bar_y + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )

    # Latest detection card
    if last_detection is not None and time_since_detection_ms < 2500:
        det_text = f"DETECTED: {last_detection.gloss.upper()} ({last_detection.confidence * 100:.1f}%)"
        text_size = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        right_x = frame.shape[1] - text_size[0] - 20
        cv2.putText(
            frame,
            det_text,
            (right_x, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        duration_text = f"Window: {last_detection.end_time_ms - last_detection.start_time_ms:.0f}ms"
        cv2.putText(
            frame,
            duration_text,
            (right_x, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
    else:
        cv2.putText(
            frame,
            "Awaiting Sign Input...",
            (frame.shape[1] - 220, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (140, 140, 140),
            1,
        )

    # Bottom helper text
    cv2.putText(
        frame,
        "Keys: [Q] Quit  [R] Reset Buffer",
        (15, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
    )


def run_synthetic_demo(
    engine: ASLVisionEngine,
    duration_seconds: float = 5.0,
) -> None:
    """Run simulated synthetic stream through pipeline for non-GUI or headless testing."""
    print("Running synthetic ASL Vision Perception stream...")
    print(f"Target duration: {duration_seconds:.1f}s at 30 FPS.")
    print("---------------------------------------------------------------")

    total_frames = int(duration_seconds * 30)
    start_wall_time = time.perf_counter()
    detections_count = 0

    from asl_vision.normalization import NormalizedFrame

    for frame_idx in range(total_frames):
        timestamp_ms = frame_idx * 33.333

        # Generate smooth sinusoidal joint motions
        t = frame_idx * 0.1
        pose = np.random.randn(33, 3).astype(np.float32) * 0.05
        left_hand = np.random.randn(21, 3).astype(np.float32) * 0.02 + np.sin(t) * 0.1
        right_hand = np.random.randn(21, 3).astype(np.float32) * 0.02 + np.cos(t) * 0.1

        norm_frame = NormalizedFrame(
            pose=pose,
            left_hand=left_hand,
            right_hand=right_hand,
            face=None,
            shoulder_distance=0.4,
            root_joint=np.zeros(3, dtype=np.float32),
            timestamp_ms=timestamp_ms,
            is_left_hand_visible=True,
            is_right_hand_visible=True,
        )

        step_start = time.perf_counter()
        detections = engine.process_normalized_frame(norm_frame, timestamp_ms=timestamp_ms)
        step_latency_ms = (time.perf_counter() - step_start) * 1000.0

        if detections:
            detections_count += len(detections)
            for d in detections:
                print(
                    f"[{d.end_time_ms:6.0f}ms] Candidate Detection: "
                    f"gloss='{d.gloss}', confidence={d.confidence * 100:.1f}%, "
                    f"window={d.start_time_ms:.0f}-{d.end_time_ms:.0f}ms "
                    f"(step latency: {step_latency_ms:.2f}ms)"
                )

    elapsed = time.perf_counter() - start_wall_time
    effective_fps = total_frames / max(elapsed, 1e-6)
    print("---------------------------------------------------------------")
    print(
        f"Synthetic benchmark completed: {total_frames} frames in {elapsed:.2f}s "
        f"({effective_fps:.1f} FPS, {detections_count} detections emitted)."
    )


def run_webcam_demo(
    engine: ASLVisionEngine,
    extractor: LandmarkExtractor,
    camera_id: int = 0,
) -> None:
    """Run live interactive video capture loop with skeletal overlay and HUD."""
    print(f"Opening video capture device /dev/video{camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        print("Falling back to synthetic stream demonstration...")
        run_synthetic_demo(engine, duration_seconds=5.0)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Camera initialized. Press 'Q' or 'ESC' in the window to exit.")
    window_name = "Converse ASL Vision Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps = 30.0
    last_detection: SignDetection | None = None
    last_detection_wall_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab video frame. Exiting.")
                break

            current_time = time.perf_counter()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Flip horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = current_time * 1000.0

            # 1. Extract 3D landmarks
            extracted = extractor.extract(frame_rgb, timestamp_ms=timestamp_ms)

            # 2. Normalize and pass to inference engine
            detections = engine.process_landmarks(extracted, timestamp_ms=timestamp_ms)
            if detections:
                last_detection = detections[0]
                last_detection_wall_time = current_time
                print(
                    f"Live Detection: gloss='{last_detection.gloss}', "
                    f"confidence={last_detection.confidence * 100:.1f}%"
                )

            # 3. Draw overlays
            draw_landmarks(frame, extracted)
            time_since_ms = (current_time - last_detection_wall_time) * 1000.0
            draw_hud(
                frame=frame,
                fps=fps,
                buffer_len=len(engine.buffer),
                window_size=engine.config.window_size,
                last_detection=last_detection,
                time_since_detection_ms=time_since_ms,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break
            elif key in (ord("r"), ord("R")):
                engine.reset()
                print("Engine buffer reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("Webcam capture session closed.")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive live webcam demonstration of Converse ASL Vision."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic simulated stream instead of opening physical webcam.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch ST-GCN .pt checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to exported ST-GCN .onnx model for ONNX Runtime inference.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Duration in seconds for synthetic test run.",
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
    """Entry point for webcam demo."""
    parsed = parse_args(args)

    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=parsed.min_confidence,
        min_window_landmark_confidence=0.20,
        use_onnx=parsed.onnx is not None,
        onnx_path=parsed.onnx,
    )

    if parsed.synthetic:
        extractor = LandmarkExtractor(use_mock=True)
    else:
        model_path = ensure_mediapipe_model()
        extractor = LandmarkExtractor(model_path=model_path)

    engine = ASLVisionEngine(config=config, extractor=extractor)

    if parsed.checkpoint is not None and not parsed.onnx:
        import torch

        ckpt = torch.load(parsed.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        if engine.pytorch_model is not None:
            engine.pytorch_model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from: {parsed.checkpoint}")

    if parsed.synthetic:
        run_synthetic_demo(engine, duration_seconds=parsed.duration)
    else:
        run_webcam_demo(engine, extractor, camera_id=parsed.camera)


if __name__ == "__main__":
    main()
