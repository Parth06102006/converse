"""Evaluation and benchmark suite for English-to-ASL grammar and translation pipeline.

Evaluates:
1. Syntactic reordering accuracy (SVO -> Topic-Comment, Wh-movement)
2. Non-Manual Marker (NMM) detection (Wh-furrow, Y/N-raise, Negation-shake)
3. Out-Of-Vocabulary (OOV) fingerspelling detection
4. Spatial locus assignment
5. Translation and compilation latency
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from translation.pipeline import SpeechToSignPipeline


@dataclass(frozen=True)
class GrammarTestCase:
    sentence: str
    expected_gloss_prefix: list[str]
    expected_eyebrow: str
    expected_head_motion: str
    has_fingerspelling: bool
    category: str


TEST_SUITE: list[GrammarTestCase] = [
    # Topic-Comment / SVO
    GrammarTestCase(
        sentence="The boy kicked the ball.",
        expected_gloss_prefix=["BOY", "KICK", "BALL"],
        expected_eyebrow="neutral",
        expected_head_motion="neutral",
        has_fingerspelling=False,
        category="Content Clause",
    ),
    GrammarTestCase(
        sentence="The cat ate the fish.",
        expected_gloss_prefix=["CAT", "EAT", "FISH"],
        expected_eyebrow="neutral",
        expected_head_motion="neutral",
        has_fingerspelling=False,
        category="Content Clause",
    ),
    # Wh-Questions (eyebrows furrowed, head tilted forward, wh-word moved to end)
    GrammarTestCase(
        sentence="What is your name?",
        expected_gloss_prefix=["YOUR", "NAME", "WHAT"],
        expected_eyebrow="furrowed",
        expected_head_motion="tilt_forward",
        has_fingerspelling=False,
        category="Wh-Question",
    ),
    GrammarTestCase(
        sentence="Where do you live?",
        expected_gloss_prefix=["YOU", "LIVE", "WHERE"],
        expected_eyebrow="furrowed",
        expected_head_motion="tilt_forward",
        has_fingerspelling=False,
        category="Wh-Question",
    ),
    GrammarTestCase(
        sentence="Who is your teacher?",
        expected_gloss_prefix=["YOUR", "TEACHER", "WHO"],
        expected_eyebrow="furrowed",
        expected_head_motion="tilt_forward",
        has_fingerspelling=False,
        category="Wh-Question",
    ),
    # Yes/No Questions (eyebrows raised, head tilted forward)
    GrammarTestCase(
        sentence="Are you happy?",
        expected_gloss_prefix=["YOU", "HAPPY"],
        expected_eyebrow="raised",
        expected_head_motion="tilt_forward",
        has_fingerspelling=False,
        category="Yes/No Question",
    ),
    GrammarTestCase(
        sentence="Do you like coffee?",
        expected_gloss_prefix=["YOU", "LIKE", "COFFEE"],
        expected_eyebrow="raised",
        expected_head_motion="tilt_forward",
        has_fingerspelling=False,
        category="Yes/No Question",
    ),
    # Negation (head shake, negation particle at clause end)
    GrammarTestCase(
        sentence="I do not want cake.",
        expected_gloss_prefix=["ME", "WANT", "CAKE", "NOT"],
        expected_eyebrow="neutral",
        expected_head_motion="shake",
        has_fingerspelling=False,
        category="Negation",
    ),
    GrammarTestCase(
        sentence="She is not coming.",
        expected_gloss_prefix=["SHE", "COME", "NOT"],
        expected_eyebrow="neutral",
        expected_head_motion="shake",
        has_fingerspelling=False,
        category="Negation",
    ),
    # Fingerspelling OOV
    GrammarTestCase(
        sentence="Alice met Zachary.",
        expected_gloss_prefix=["ALICE", "MEET", "ZACHARY"],
        expected_eyebrow="neutral",
        expected_head_motion="neutral",
        has_fingerspelling=True,
        category="Fingerspelling",
    ),
]


def evaluate_grammar() -> None:
    print("==================================================")
    print("English-to-ASL Grammar & Pipeline Evaluation")
    print("==================================================")

    pipeline = SpeechToSignPipeline()
    latencies_ms: list[float] = []

    passed_count = 0
    total_count = len(TEST_SUITE)

    for i, test in enumerate(TEST_SUITE, start=1):
        t0 = time.perf_counter()
        compilation = pipeline.compiler.compile(test.sentence)
        rep = pipeline.translate(
            english_text=test.sentence,
            session_id=f"bench_sess_{i}",
            utterance_id=f"utt_{i}",
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)

        comp_glosses = [t.gloss for t in compilation.tokens]

        # Verify glosses match expected structure
        glosses_match = comp_glosses == test.expected_gloss_prefix

        # Check NMM
        first_token = compilation.tokens[0] if compilation.tokens else None
        eyebrow_shape = first_token.non_manual_markers.eyebrows if first_token else "neutral"
        eyebrow_match = eyebrow_shape == test.expected_eyebrow

        # Check head motion
        head_motion = first_token.non_manual_markers.head_motion if first_token else "neutral"
        head_match = head_motion == test.expected_head_motion

        # Check fingerspelling flag
        has_fs = any(t.is_fingerspelled for t in compilation.tokens)
        fs_match = (test.has_fingerspelling == has_fs) if test.has_fingerspelling else True

        # Check emitted representation has valid tokens and monotonic timing
        has_valid_rep = len(rep.tokens) > 0 and rep.total_duration_ms > 0
        rep_timing_monotonic = all(
            rep.tokens[k].timing.start_time_ms <= rep.tokens[k + 1].timing.start_time_ms
            for k in range(len(rep.tokens) - 1)
        )

        status = "PASS" if (glosses_match and eyebrow_match and head_match and fs_match and has_valid_rep and rep_timing_monotonic) else "FAIL"
        if status == "PASS":
            passed_count += 1

        print(f"[{i:02d}] {test.category:<16} | {status} | Latency: {latency_ms:.2f}ms")
        print(f"     Input:     \"{test.sentence}\"")
        print(f"     ASL Gloss: {' '.join(comp_glosses)}")
        print(f"     NMM:       eyebrows={eyebrow_shape}, head={head_motion}")
        print(f"     Rep Clips: {len(rep.tokens)} tokens, duration={rep.total_duration_ms:.1f}ms")

        if status == "FAIL":
            print(f"     Expected:  {' '.join(test.expected_gloss_prefix)} (eyebrow: {test.expected_eyebrow}, head: {test.expected_head_motion})")

    mean_lat = sum(latencies_ms) / len(latencies_ms)
    max_lat = max(latencies_ms)

    print("\n--------------------------------------------------")
    print(f"Accuracy:     {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    print(f"Mean Latency: {mean_lat:.3f} ms (Target < 15.0 ms)")
    print(f"Max Latency:  {max_lat:.3f} ms")
    print("--------------------------------------------------")

    assert passed_count == total_count, f"Some grammar tests failed ({passed_count}/{total_count})"
    assert mean_lat < 15.0, f"Mean latency {mean_lat}ms exceeds 15ms target"

    print("\n==================================================")
    print("Grammar Evaluation PASSED: 100% test accuracy.")
    print("==================================================")


if __name__ == "__main__":
    evaluate_grammar()
