"""Unit tests for timing model, co-articulation, and end-to-end gloss compilation."""

from translation.gloss_compiler import GlossCompiler
from translation.grammar_rules import SentenceType
from translation.timing_model import TimingConfig, TimingModel


class TestTimingModel:
    """Test suite for sign duration, co-articulation, and monotonicity."""

    def setup_method(self) -> None:
        self.timing_model = TimingModel(TimingConfig(base_hold_ms=300.0, base_lead_in_ms=120.0))

    def test_monotonic_timestamps(self) -> None:
        specs = [
            {"is_fingerspelled": False, "letter_count": 1},
            {"is_fingerspelled": False, "letter_count": 1},
            {"is_fingerspelled": True, "letter_count": 4},
            {"is_fingerspelled": False, "letter_count": 1},
        ]
        timings = self.timing_model.calculate_sequence_timings(specs, start_offset_ms=0.0)

        assert len(timings) == 4
        # Assert strictly non-decreasing and non-overlapping
        for i in range(len(timings) - 1):
            curr = timings[i]
            nxt = timings[i + 1]
            assert nxt.start_time_ms > curr.start_time_ms
            assert nxt.start_time_ms >= curr.start_time_ms + curr.lead_in_duration_ms

    def test_fingerspelling_duration_proportional_to_letters(self) -> None:
        short_fs = self.timing_model.compute_token_duration(is_fingerspelled=True, letter_count=2)
        long_fs = self.timing_model.compute_token_duration(is_fingerspelled=True, letter_count=6)

        # hold duration of 6 letters should be 3x that of 2 letters
        assert long_fs[1] == short_fs[1] * 3.0

    def test_speed_factor_scaling(self) -> None:
        fast_model = TimingModel(TimingConfig(speed_factor=1.5))
        slow_model = TimingModel(TimingConfig(speed_factor=0.75))

        dur_fast = fast_model.compute_token_duration(is_fingerspelled=False)
        dur_slow = slow_model.compute_token_duration(is_fingerspelled=False)

        assert dur_fast[1] < dur_slow[1]


class TestGlossCompilerIntegration:
    """End-to-end integration tests for the full GlossCompiler pipeline."""

    def setup_method(self) -> None:
        self.compiler = GlossCompiler()

    def test_compile_statement(self) -> None:
        res = self.compiler.compile("I want water")
        assert res.sentence_type == SentenceType.STATEMENT
        assert len(res.tokens) >= 2
        assert res.gloss_sequence[0] == "ME"
        assert "WANT" in res.gloss_sequence
        assert "WATER" in res.gloss_sequence

        for tok in res.tokens:
            assert tok.timing is not None
            assert tok.timing.start_time_ms >= 0.0
            assert tok.timing.hold_duration_ms > 0

    def test_compile_wh_question(self) -> None:
        res = self.compiler.compile("Where do you live?")
        assert res.sentence_type == SentenceType.WH_QUESTION
        assert res.gloss_sequence[-1] == "WHERE"

        # Check NMM cues on tokens
        for tok in res.tokens:
            assert tok.non_manual_markers.eyebrows == "furrowed"

    def test_compile_with_oov_fingerspelling(self) -> None:
        res = self.compiler.compile("My friend Alice arrived yesterday")
        # Alice is OOV -> should be fingerspelled
        alice_tok = next((t for t in res.tokens if t.gloss == "ALICE"), None)
        assert alice_tok is not None
        assert alice_tok.is_fingerspelled
        assert alice_tok.fingerspell_sequence == ["A", "L", "I", "C", "E"]

        # Output dictionary serialization test
        dict_payload = res.to_dict()
        assert "tokens" in dict_payload
        assert "totalDurationMs" in dict_payload
        assert dict_payload["totalDurationMs"] > 0
