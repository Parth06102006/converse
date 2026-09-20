"""Offline WLASL recognition eval: real dataset clips through live preprocessing.

Feeds each video through the exact live pipeline (MediaPipe extraction +
TGCNWLASLClassifier buffer, mirroring demo_webcam.py) and reports Top-1/Top-5
accuracy against the clip gloss label.

Usage:
    uv run python scripts/eval_wlasl_offline.py /tmp/wlasl-eval/videos
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from asl_vision.landmarks import LandmarkExtractor
from asl_vision.models.tgcn_wlasl import TGCNWLASLClassifier


def evaluate_video(
    video_path: Path,
    extractor: LandmarkExtractor,
    classifier: TGCNWLASLClassifier,
    max_frames: int = 300,
    stride: int = 2,
    swap_hands: bool = False,
) -> tuple[str | None, float, list[tuple[str, float]]]:
    """Run one clip through extraction + classifier buffer, return best prediction."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    classifier.reset()
    frame_idx = 0
    best: tuple[str | None, float, list[tuple[str, float]]] = (None, 0.0, [])
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            resized = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            extracted = extractor.extract(rgb, timestamp_ms=frame_idx * 33.3)
            if swap_hands:
                import dataclasses

                extracted = dataclasses.replace(
                    extracted,
                    left_hand=extracted.right_hand,
                    right_hand=extracted.left_hand,
                )
            classifier.add_frame(extracted)
            gloss, score, top = classifier.predict()
            if gloss is not None and score > best[1]:
                best = (gloss, score, top)
        frame_idx += 1
    cap.release()
    return best


def main(argv: list[str]) -> int:
    videos_dir = Path(argv[-1] if len(argv) > 1 and not argv[-1].startswith("-") else "/tmp/wlasl-eval/videos")
    swap_hands = "--swap-hands" in argv
    clips = sorted(videos_dir.glob("*.mp4"))
    # Skip stub error pages (< 50KB are HTML, not video)
    clips = [c for c in clips if c.stat().st_size > 50_000]
    if not clips:
        print(f"No usable clips in {videos_dir}")
        return 1

    print(f"swap_hands={swap_hands}")
    classifier = TGCNWLASLClassifier("models/tgcn_asl100.bin")

    top1 = 0
    top5 = 0
    for clip in clips:
        label = clip.stem.upper()
        # Fresh extractor per clip: the MediaPipe graph holds frame-size state
        extractor = LandmarkExtractor(
            model_path="models/holistic_landmarker.task",
            hand_model_path="models/gesture_recognizer.task",
        )
        t0 = time.perf_counter()
        gloss, score, top = evaluate_video(clip, extractor, classifier, swap_hands=swap_hands)
        dt = time.perf_counter() - t0
        top_glosses = [g for g, _ in top]
        hit1 = gloss == label
        hit5 = label in top_glosses
        top1 += hit1
        top5 += hit5
        mark1 = "HIT " if hit1 else "miss"
        mark5 = "in-top5" if hit5 else "not-top5"
        print(f"[{mark1}/{mark5}] {label:8s} -> {gloss} ({score:.2f}) top5={top_glosses[:5]} [{dt:.1f}s]")

    print(f"\nTop-1: {top1}/{len(clips)} = {top1 / len(clips):.1%} (criterion: >=70%)")
    print(f"Top-5: {top5}/{len(clips)} = {top5 / len(clips):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
