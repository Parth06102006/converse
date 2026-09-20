"""Harvest WLASL training features through the live MediaPipe pipeline.

Downloads WLASL clips per gloss, extracts 55-keypoint (BODY_25-ordered) features
with the production preprocessor, stores compact .npz files plus a manifest,
and deletes videos after extraction to bound disk use.

Usage:
    uv run python scripts/harvest_wlasl_features.py /tmp/WLASL-repo/start_kit/WLASL_v0.3.json /tmp/wlasl-train [--max-per-gloss 8] [--glosses BOOK,DRINK]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from asl_vision.landmarks import LandmarkExtractor
from asl_vision.models.tgcn_wlasl import _extract_upper_body_pose


def download(url: str, dest: Path, timeout_s: int = 180) -> bool:
    """Fetch direct mp4 or YouTube URL into dest."""
    if "youtube.com" in url or "youtu.be" in url:
        r = subprocess.run(
            ["uvx", "yt-dlp", "--no-playlist", "-f", "mp4[height<=480]/best[height<=480]/best",
             "-o", str(dest), url],
            capture_output=True, timeout=timeout_s, check=False,
        )
        return r.returncode == 0 and dest.is_file()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return dest.stat().st_size > 50_000
    except Exception:  # noqa: BLE001 - per-URL failures are routine (dead links, auth walls)
        return False


def extract_features(video: Path, extractor: LandmarkExtractor, stride: int = 2) -> dict | None:
    """Extract ungated 55-keypoint features + active mask from a clip."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    feats: list[np.ndarray] = []
    active: list[bool] = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret or i > 600:
            break
        if i % stride == 0:
            small = cv2.resize(frame, (640, 480))
            ext = extractor.extract(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), timestamp_ms=i * 33.3)
            pts = np.full((55, 2), -1.0, dtype=np.float32)
            pts[:13] = _extract_upper_body_pose(ext.pose, frame_width=640, frame_height=480)
            has_l = ext.left_hand is not None and len(ext.left_hand) >= 21
            has_r = ext.right_hand is not None and len(ext.right_hand) >= 21
            l_act = has_l and (float(ext.left_hand[0, 1]) < 0.76 or float(ext.left_hand[8, 1]) < 0.72)
            r_act = has_r and (float(ext.right_hand[0, 1]) < 0.76 or float(ext.right_hand[8, 1]) < 0.72)
            if l_act:
                hand = ext.left_hand[:, :2].astype(np.float32)
                pts[13:34, 0] = 2.0 * ((hand[:, 0] * 640) / 256.0 - 0.5)
                pts[13:34, 1] = 2.0 * ((hand[:, 1] * 480) / 256.0 - 0.5)
            if r_act:
                hand = ext.right_hand[:, :2].astype(np.float32)
                pts[34:55, 0] = 2.0 * ((hand[:, 0] * 640) / 256.0 - 0.5)
                pts[34:55, 1] = 2.0 * ((hand[:, 1] * 480) / 256.0 - 0.5)
            feats.append(pts)
            active.append(bool(l_act or r_act))
        i += 1
    cap.release()
    if len(feats) < 15:
        return None
    return {"features": np.stack(feats), "active": np.array(active), "src_size": (w, h)}


def main(argv: list[str]) -> int:
    index_json = Path(argv[1])
    out_dir = Path(argv[2] if len(argv) > 2 else "/tmp/wlasl-train")
    max_per_gloss = 8
    only: set[str] | None = None
    for a in argv[3:]:
        if a.startswith("--max-per-gloss"):
            max_per_gloss = int(a.split("=")[1])
        if a.startswith("--glosses"):
            only = set(a.split("=")[1].upper().split(","))

    from asl_vision.models.tgcn_wlasl import WLASL_100_GLOSSES

    want = [g for g in WLASL_100_GLOSSES if only is None or g in only]
    with open(index_json) as f:
        entries = json.load(f)
    by_gloss = {e["gloss"].upper(): e["instances"] for e in entries}

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = set()
    if manifest.is_file():
        with open(manifest) as f:
            for line in f:
                done.add(json.loads(line)["video_id"])

    extractor = LandmarkExtractor(
        model_path="models/holistic_landmarker.task",
        hand_model_path="models/gesture_recognizer.task",
    )
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for gloss in want:
            count = 0
            for inst in by_gloss.get(gloss, []):
                if count >= max_per_gloss:
                    break
                vid = str(inst["video_id"])
                if vid in done:
                    count += 1
                    continue
                dest = Path(tmp) / f"{vid}.mp4"
                if not download(inst["url"], dest):
                    continue
                feats = extract_features(dest, extractor)
                dest.unlink(missing_ok=True)
                if feats is None:
                    continue
                np.savez_compressed(
                    out_dir / f"{vid}.npz",
                    features=feats["features"], active=feats["active"],
                )
                with open(manifest, "a") as f:
                    f.write(json.dumps({
                        "video_id": vid, "gloss": gloss,
                        "split": inst.get("split", "train"),
                        "frames": int(feats["features"].shape[0]),
                        "active_frames": int(feats["active"].sum()),
                    }) + "\n")
                count += 1
                total += 1
                print(f"harvested {gloss} {vid} ({feats['features'].shape[0]} frames)", flush=True)
    print(f"Harvest complete: {total} new clips in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
