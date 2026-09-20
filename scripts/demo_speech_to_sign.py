#!/usr/bin/env python3
"""Interactive demonstration runner for Ashwani's Speech-to-Sign Pipeline.

Demonstrates:
1. Neural Voice Activity Detection (Silero VAD v5 ONNX)
2. Streaming Automatic Speech Recognition (Faster-Whisper INT8 / AWS Transcribe)
3. English-to-ASL Grammar Transformation (SVO -> Topic-Comment / TSOV, Wh-movement, Negation)
4. Non-Manual Marker (NMM) Detection (eyebrow furrows, eyebrow raises, head shakes, mouth shapes)
5. Out-Of-Vocabulary (OOV) Fingerspelling Decomposition
6. 3D Spatial Locus Tracking (referent positioning in signing space)
7. Dynamic Co-articulation Timing Engine (lead-in, hold, lead-out durations)
8. Canonical SignRepresentation serialization matching @converse/contracts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

# Ensure subpackages are on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml" / "asr" / "src"))
sys.path.insert(0, str(ROOT / "ml" / "translation" / "src"))

from asr.engine import create_asr_engine
from translation.pipeline import SpeechToSignPipeline
from translation.representation_emitter import SignRepresentation

AUDIO_SAMPLE_PATH = ROOT / "ml" / "asr" / "tests" / "data" / "speech_sample_16k.wav"

PRESET_SCENARIOS: list[dict[str, str]] = [
    {
        "name": "Wh-Question (Inversion & Furrowed Brows)",
        "input": "What is your name?",
        "description": "Wh-word moved to clause end; eyebrows furrowed; head tilted forward.",
    },
    {
        "name": "Yes/No Question (Raised Brows & Head Tilt)",
        "input": "Are you happy?",
        "description": "Copula omitted; eyebrows raised; head tilted forward.",
    },
    {
        "name": "Clausal Negation (Head Shake)",
        "input": "I do not want cake.",
        "description": "Subject converted to ME; NOT moved to clause end; head shake marker.",
    },
    {
        "name": "Temporal Shift (TSOV Reordering)",
        "input": "Yesterday I went to the store.",
        "description": "Temporal adverb lifted to start; verb converted to base GO + PAST marker.",
    },
    {
        "name": "Out-Of-Vocabulary Fingerspelling",
        "input": "Alice met Zachary in Seattle.",
        "description": "Proper names decomposed into letter-by-letter fingerspelled tokens.",
    },
    {
        "name": "Spatial Referents & Pronouns",
        "input": "The doctor gave me medicine.",
        "description": "Indexes discourse referents to 3D spatial loci (left/right/chest).",
    },
]


def render_banner() -> None:
    """Render terminal header banner."""
    print("=" * 76)
    print(" CONVERSE: SPEECH-TO-SIGN MODEL PIPELINE INTERACTIVE DEMONSTRATOR")
    print(" Subsystem Author: Ashwani | Architecture: VAD + ASR + Grammar + NMM + Loci")
    print("=" * 76)


def render_representation(rep: SignRepresentation, compute_time_ms: float) -> None:
    """Render structured breakdown of the emitted SignRepresentation."""
    print("\n" + "-" * 76)
    print(f"PIPELINE OUTPUT | Utterance ID: {rep.utterance_id} | Total Duration: {rep.total_duration_ms:.1f}ms")
    print(f"Compilation Compute Latency: {compute_time_ms:.3f}ms")
    print("-" * 76)

    # 1. Gloss Sequence & NMM Summary
    glosses = [tok.gloss for tok in rep.tokens]
    print(f"ASL Gloss Sequence:    {' '.join(glosses)}")

    # Check NMM across tokens
    eyebrows = {tok.non_manual_markers.eyebrow_shape for tok in rep.tokens if tok.non_manual_markers.eyebrow_shape != "neutral"}
    eyebrow_str = ", ".join(eyebrows) if eyebrows else "neutral"

    # Head rotation description
    has_head_pitch = any(abs(tok.non_manual_markers.head_rotation.pitch) > 0.01 for tok in rep.tokens)
    has_head_yaw = any(abs(tok.non_manual_markers.head_rotation.yaw) > 0.01 for tok in rep.tokens)
    head_motions: list[str] = []
    if has_head_pitch:
        head_motions.append("nod/tilt")
    if has_head_yaw:
        head_motions.append("shake")
    head_str = ", ".join(head_motions) if head_motions else "neutral"

    mouths = {tok.non_manual_markers.mouth_shape for tok in rep.tokens if tok.non_manual_markers.mouth_shape != "neutral"}
    mouth_str = ", ".join(mouths) if mouths else "neutral"

    print(f"Non-Manual Markers:    Eyebrows: [{eyebrow_str}] | Head: [{head_str}] | Mouth: [{mouth_str}]")

    # 2. Token Breakdown Table
    print("\n" + "-" * 76)
    print(f"{'#':<3} {'Gloss':<14} {'Clip ID':<16} {'Start (ms)':<12} {'Duration':<10} {'3D Spatial Anchor':<16}")
    print("-" * 76)
    for idx, tok in enumerate(rep.tokens):
        dur = tok.timing.lead_in_duration_ms + tok.timing.hold_duration_ms + tok.timing.lead_out_duration_ms
        anchor = f"{tok.spatial_loci.anchor} ({tok.spatial_loci.target_offset.x:+.1f},{tok.spatial_loci.target_offset.z:+.1f})"
        print(f"{idx+1:<3} {tok.gloss:<14} {tok.clip_id:<16} {tok.timing.start_time_ms:<12.1f} {dur:<10.1f} {anchor:<16}")

    # 3. Visual Timeline Bar
    print("\n" + "-" * 76)
    print("Timeline Visualization (Lead-in [=], Hold [#], Lead-out [-]):")
    print("-" * 76)
    total_ms = max(rep.total_duration_ms, 1.0)
    for tok in rep.tokens[:8]:  # display first 8 tokens for clean layout
        start_ratio = tok.timing.start_time_ms / total_ms
        hold_ratio = tok.timing.hold_duration_ms / total_ms
        indent = int(start_ratio * 40)
        bar_len = max(int(hold_ratio * 30), 2)
        bar = "#" * bar_len
        print(f"{tok.gloss[:8]:<8} | {' ' * indent}[{bar}] ({tok.timing.start_time_ms:.0f}ms - {tok.timing.start_time_ms + tok.timing.hold_duration_ms:.0f}ms)")

    if len(rep.tokens) > 8:
        print(f"... and {len(rep.tokens) - 8} more tokens in sequence")

    # 4. JSON Payload Sample
    print("\n" + "-" * 76)
    print("Canonical SignRepresentation JSON (for WebGL 3D Avatar):")
    print("-" * 76)
    json_dict = rep.to_dict()
    # Show first 2 tokens in JSON preview for readability
    preview_dict = {
        "version": json_dict["version"],
        "sessionId": json_dict["sessionId"],
        "utteranceId": json_dict["utteranceId"],
        "totalDurationMs": json_dict["totalDurationMs"],
        "tokensSample": json_dict["tokens"][:2],
    }
    print(json.dumps(preview_dict, indent=2))
    print("-" * 76)


def process_text(pipeline: SpeechToSignPipeline, text: str, session_id: str = "demo_session") -> None:
    """Compile English text and display pipeline output."""
    print(f"\nProcessing English Input: \"{text}\"")
    t0 = time.perf_counter()
    representation = pipeline.translate(english_text=text, session_id=session_id)
    compute_ms = (time.perf_counter() - t0) * 1000.0
    render_representation(representation, compute_ms)


def process_audio_file(
    asr_engine: Any,
    pipeline: SpeechToSignPipeline,
    audio_file: Path,
    session_id: str = "audio_demo_session",
) -> None:
    """Stream audio file through VAD + ASR and compile result."""
    print(f"\nReading Audio File: {audio_file}")
    if not audio_file.is_file():
        print(f"Error: Audio file not found at {audio_file}")
        return

    with wave.open(str(audio_file), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm_data = wf.readframes(n_frames)

    duration_sec = n_frames / float(framerate)
    print(f"Audio Properties: {framerate}Hz, {n_channels} channel(s), {sampwidth*8}-bit, Duration: {duration_sec:.2f}s")
    print("Streaming through Silero VAD v5 and Faster-Whisper ASR Engine...")

    chunk_size = int(framerate * 0.2) * sampwidth * n_channels  # 200ms chunks
    offset = 0
    t_start = time.perf_counter()
    final_events = []

    while offset < len(pcm_data):
        chunk = pcm_data[offset : offset + chunk_size]
        offset += chunk_size
        events = asr_engine.process_audio_chunk(
            session_id=session_id,
            audio_data=chunk,
            audio_format="pcm_s16le",
        )
        for ev in events:
            if ev.is_final:
                final_events.append(ev)
            else:
                print(f"  [Streaming Partial] \"{ev.text}\" (confidence: {ev.confidence:.2f})")

    # Flush any remaining buffer
    flush_events = asr_engine.flush_session(session_id=session_id)
    final_events.extend(flush_events)

    total_asr_time_ms = (time.perf_counter() - t_start) * 1000.0

    if not final_events:
        print("Notice: No speech detected in audio stream.")
        return

    full_transcript = " ".join(ev.text for ev in final_events).strip()
    print(f"\nFinal ASR Transcription: \"{full_transcript}\"")
    print(f"ASR Turnaround Time:     {total_asr_time_ms:.1f}ms (RTF: {total_asr_time_ms / (duration_sec * 1000.0):.4f})")

    # Now compile into ASL SignRepresentation
    process_text(pipeline, full_transcript, session_id=session_id)


def record_microphone_and_process(
    asr_engine: Any,
    pipeline: SpeechToSignPipeline,
    duration_sec: int = 4,
) -> None:
    """Record live speech from local microphone and feed into pipeline."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = Path(tmp_file.name)

    try:
        print(f"\nPreparing to record {duration_sec} seconds from your microphone...")
        print("Please speak clearly into your microphone when recording begins.")
        for countdown in (3, 2, 1):
            print(f"Starting in {countdown}...")
            time.sleep(1)

        print(">>> RECORDING NOW - Speak your English sentence... <<<")
        cmd = [
            "arecord",
            "-d", str(duration_sec),
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            print(f"arecord failed with error: {proc.stderr.decode('utf-8')}")
            return

        print("Recording finished. Processing speech...")
        process_audio_file(asr_engine, pipeline, wav_path, session_id="live_mic_session")

    finally:
        if wav_path.is_file():
            wav_path.unlink()


def main() -> None:
    """Main interactive demonstration loop."""
    parser = argparse.ArgumentParser(description="Ashwani Speech-to-Sign Pipeline Demonstrator")
    parser.add_argument("--backend", choices=["whisper", "mock", "aws"], default="whisper", help="ASR backend model")
    parser.add_argument("--text", type=str, default=None, help="Run single text compilation and exit")
    parser.add_argument("--audio", type=str, default=None, help="Run single audio transcription and exit")
    args = parser.parse_args()

    render_banner()

    print(f"Initializing SpeechToSignPipeline and ASR Engine (backend: {args.backend})...")
    pipeline = SpeechToSignPipeline()
    asr_engine = create_asr_engine(backend=args.backend)
    print("Models and pipelines initialized successfully.\n")

    # Direct non-interactive flags
    if args.text:
        process_text(pipeline, args.text)
        return

    if args.audio:
        process_audio_file(asr_engine, pipeline, Path(args.audio))
        return

    # Interactive menu loop
    while True:
        print("\n" + "=" * 76)
        print(" SELECT A DEMONSTRATION MODE:")
        print("=" * 76)
        for i, sc in enumerate(PRESET_SCENARIOS, 1):
            print(f"  [{i}] {sc['name']}")
            print(f"      Input: \"{sc['input']}\"")
        print("  [7] Pre-recorded Speech Audio (Silero VAD + Faster-Whisper ASR)")
        print("  [8] Live Microphone Recording (Record & Translate in real time)")
        print("  [9] Custom English Text (Type your own sentence)")
        print("  [Q] Exit")
        print("=" * 76)

        try:
            choice = input("Enter option [1-9, Q]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demonstrator.")
            break

        if choice in ("q", "quit", "exit"):
            print("Exiting demonstrator. Good day.")
            break
        elif choice in ("1", "2", "3", "4", "5", "6"):
            idx = int(choice) - 1
            scenario = PRESET_SCENARIOS[idx]
            print(f"\nExecuting Scenario {choice}: {scenario['name']}")
            print(f"Linguistic Rule: {scenario['description']}")
            process_text(pipeline, scenario["input"])
        elif choice == "7":
            process_audio_file(asr_engine, pipeline, AUDIO_SAMPLE_PATH)
        elif choice == "8":
            try:
                dur_str = input("Recording duration in seconds [default: 4]: ").strip()
                dur = int(dur_str) if dur_str.isdigit() else 4
            except (KeyboardInterrupt, EOFError):
                break
            record_microphone_and_process(asr_engine, pipeline, duration_sec=dur)
        elif choice == "9":
            try:
                user_text = input("Enter English sentence: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if user_text:
                process_text(pipeline, user_text)
            else:
                print("Notice: Empty input skipped.")
        else:
            print("Invalid selection. Please choose an option from the menu.")


if __name__ == "__main__":
    main()
