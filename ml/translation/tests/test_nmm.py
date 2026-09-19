"""Unit tests for Non-Manual Marker classification and fingerspelling resolution."""

from translation.fingerspelling import FingerspellResolver
from translation.grammar_rules import SentenceType
from translation.nmm_detector import NmmDetector


class TestNmmDetector:
    """Test suite for eyebrow, head motion, and mouth morpheme tagging."""

    def setup_method(self) -> None:
        self.detector = NmmDetector()

    def test_wh_question_markers(self) -> None:
        markers = self.detector.detect_sentence_markers(SentenceType.WH_QUESTION)
        assert markers.eyebrows == "furrowed"
        assert markers.head_motion == "tilt_forward"

    def test_yes_no_question_markers(self) -> None:
        markers = self.detector.detect_sentence_markers(SentenceType.YN_QUESTION)
        assert markers.eyebrows == "raised"
        assert markers.head_motion == "tilt_forward"

    def test_negation_markers(self) -> None:
        markers = self.detector.detect_sentence_markers(SentenceType.STATEMENT, has_negation=True)
        assert markers.head_motion == "shake"

    def test_mouth_morpheme_association(self) -> None:
        sent_markers = self.detector.detect_sentence_markers(SentenceType.STATEMENT)
        big_token = self.detector.detect_token_markers("BIG", sent_markers)
        assert big_token.mouth_morpheme == "cha"

        small_token = self.detector.detect_token_markers("SMALL", sent_markers)
        assert small_token.mouth_morpheme == "oo"

        normal_token = self.detector.detect_token_markers("COMFORTABLE", sent_markers)
        assert normal_token.mouth_morpheme == "mm"


class TestFingerspellResolver:
    """Test suite for in-vocabulary vs out-of-vocabulary decomposition."""

    def setup_method(self) -> None:
        self.resolver = FingerspellResolver()

    def test_in_lexicon_sign(self) -> None:
        res = self.resolver.resolve("STORE")
        assert not res.is_fingerspelled
        assert res.sequence is None
        assert res.gloss == "STORE"

    def test_oov_named_entity_decomposition(self) -> None:
        res = self.resolver.resolve("ALICE")
        assert res.is_fingerspelled
        assert res.sequence == ["A", "L", "I", "C", "E"]
        assert res.gloss == "ALICE"

    def test_custom_lexicon_extension(self) -> None:
        custom = FingerspellResolver(custom_lexicon={"PYTHON", "DJANGO"})
        res = custom.resolve("PYTHON")
        assert not res.is_fingerspelled
        assert res.sequence is None
