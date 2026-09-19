"""Tests for the live demo Sign-to-Speech stabilization and sentence reconstruction components."""

from __future__ import annotations

from asl_vision.engine import SignDetection
from scripts.demo_webcam import (
    PythonGlossStabilizer,
    reconstruct_sentence,
)


def test_reconstruct_canonical_idioms() -> None:
    assert reconstruct_sentence(["HELLO"]) == "Hello!"
    assert reconstruct_sentence(["HELLO", "NICE", "MEET", "YOU"]) == "Hello, nice to meet you."
    assert reconstruct_sentence(["THANK-YOU", "HELP"]) == "Thank you for your help."
    assert reconstruct_sentence(["SEE", "YOU", "LATER"]) == "See you later."
    assert reconstruct_sentence(["GOOD", "MORNING"]) == "Good morning."
    assert reconstruct_sentence(["SORRY", "LATE"]) == "Sorry, I am late."
    assert reconstruct_sentence(["PLEASE", "HELP", "ME"]) == "Please help me."


def test_reconstruct_wh_questions() -> None:
    assert reconstruct_sentence(["NAME", "YOU", "WHAT"]) == "What is your name?"
    assert reconstruct_sentence(["BATHROOM", "WHERE"]) == "Where is the bathroom?"
    assert reconstruct_sentence(["TIME", "WHAT"]) == "What time is it?"
    assert reconstruct_sentence(["YOU", "GO", "WHERE"]) == "Where are you going?"
    assert reconstruct_sentence(["HOW", "MUCH"]) == "How much does this cost?"


def test_reconstruct_tense_shifting() -> None:
    assert reconstruct_sentence(["YESTERDAY", "ME", "STORE", "GO"]) == "Yesterday, I went to the store."
    assert reconstruct_sentence(["TOMORROW", "ME", "AIRPORT", "GO"]) == "Tomorrow, I will go to the airport."
    assert reconstruct_sentence(["TOMORROW", "WE", "MEET"]) == "Tomorrow, We will meet."


def test_reconstruct_copula_and_adjectives() -> None:
    assert reconstruct_sentence(["ME", "HUNGRY"]) == "I am hungry."
    assert reconstruct_sentence(["ME", "TIRED"]) == "I am tired."
    assert reconstruct_sentence(["WE", "READY"]) == "We are ready."
    assert reconstruct_sentence(["WEATHER", "HOT"]) == "The weather is hot."


def test_reconstruct_negation() -> None:
    assert reconstruct_sentence(["ME", "NOT", "KNOW"]) == "I do not know."
    assert reconstruct_sentence(["I", "NOT", "WANT", "COFFEE"]) == "I do not want coffee."
    assert reconstruct_sentence(["WE", "NOT", "READY"]) == "We are not ready."


def test_gloss_stabilizer_debouncing() -> None:
    stabilizer = PythonGlossStabilizer(min_confidence=0.5, debounce_window_ms=400.0)

    # 4 consecutive detections of HELLO within 200ms
    d1 = SignDetection("HELLO", 0.9, 100.0, 150.0)
    d2 = SignDetection("HELLO", 0.92, 160.0, 200.0)
    d3 = SignDetection("HELLO", 0.88, 210.0, 250.0)

    acc1, is_dup1, gloss1 = stabilizer.process_detection(d1)
    assert acc1 and not is_dup1 and gloss1 == "HELLO"

    acc2, is_dup2, gloss2 = stabilizer.process_detection(d2)
    assert acc2 and is_dup2 and gloss2 is None

    acc3, is_dup3, gloss3 = stabilizer.process_detection(d3)
    assert acc3 and is_dup3 and gloss3 is None

    # Only 1 unique gloss in buffer
    assert stabilizer.buffer == ["HELLO"]


def test_gloss_stabilizer_boundary_pause() -> None:
    stabilizer = PythonGlossStabilizer(min_confidence=0.5, boundary_pause_ms=800.0)

    d1 = SignDetection("THANK-YOU", 0.95, 100.0, 200.0)
    stabilizer.process_detection(d1)

    # Before pause threshold -> no flush
    assert stabilizer.check_boundary(500.0) is None
    assert len(stabilizer.buffer) == 1

    # After 850ms pause (timestamp 1100) -> triggers boundary flush
    flushed = stabilizer.check_boundary(1100.0)
    assert flushed == ["THANK-YOU"]
    assert len(stabilizer.buffer) == 0
