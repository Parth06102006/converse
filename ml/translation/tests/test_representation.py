"""Unit tests for SignRepresentation emission and pipeline integration."""

from translation.pipeline import SpeechToSignPipeline
from translation.representation_emitter import SignRepresentationEmitter


class TestSignRepresentationEmitter:
    """Test suite for SignRepresentation schema compliance and blendshapes."""

    def setup_method(self) -> None:
        self.pipeline = SpeechToSignPipeline()
        self.emitter = SignRepresentationEmitter()

    def test_representation_schema_contract(self) -> None:
        rep = self.pipeline.translate(
            english_text="Where is the store?",
            session_id="session_test_01",
            utterance_id="utt_test_01",
        )

        assert rep.version == "1.0.0"
        assert rep.session_id == "session_test_01"
        assert rep.utterance_id == "utt_test_01"
        assert rep.total_duration_ms > 0
        assert len(rep.tokens) > 0

        dict_payload = rep.to_dict()
        assert dict_payload["version"] == "1.0.0"
        assert dict_payload["sessionId"] == "session_test_01"
        assert dict_payload["utteranceId"] == "utt_test_01"
        assert "tokens" in dict_payload

        first_tok = dict_payload["tokens"][0]
        assert "tokenId" in first_tok
        assert "clipId" in first_tok
        assert "gloss" in first_tok
        assert "timing" in first_tok
        assert "spatialLoci" in first_tok
        assert "nonManualMarkers" in first_tok
        assert "interpolationCurve" in first_tok

    def test_wh_question_blendshapes(self) -> None:
        rep = self.pipeline.translate("What is your name?")
        # Wh-question must set eyebrowShape: furrow with intensity >= 0.8
        for tok in rep.tokens:
            assert tok.non_manual_markers.eyebrow_shape == "furrow"
            assert tok.non_manual_markers.eyebrow_intensity >= 0.8
            assert tok.non_manual_markers.head_rotation.pitch > 0.0  # Tilt forward

    def test_yes_no_question_blendshapes(self) -> None:
        rep = self.pipeline.translate("Are you hungry?")
        for tok in rep.tokens:
            assert tok.non_manual_markers.eyebrow_shape == "raise"
            assert tok.non_manual_markers.eyebrow_intensity >= 0.8

    def test_negation_head_shake(self) -> None:
        rep = self.pipeline.translate("I do not like tea")
        # Negation must produce head yaw rotation (head shake)
        for tok in rep.tokens:
            if tok.gloss == "NOT":
                assert tok.non_manual_markers.head_rotation.yaw >= 0.20

    def test_fingerspelling_character_expansion(self) -> None:
        rep = self.pipeline.translate("Alice arrived")
        # Alice is OOV, with expand_fingerspelling=True, it should emit tokens for A, L, I, C, E
        alice_chars = [t for t in rep.tokens if t.clip_id.startswith("asl_fs_")]
        assert len(alice_chars) == 5
        assert [t.gloss for t in alice_chars] == ["A", "L", "I", "C", "E"]
        assert alice_chars[0].clip_id == "asl_fs_a_01"
        assert alice_chars[1].clip_id == "asl_fs_l_01"

    def test_monotonic_representation_timings(self) -> None:
        rep = self.pipeline.translate("Tomorrow I will go to the market")
        for i in range(len(rep.tokens) - 1):
            curr = rep.tokens[i].timing
            nxt = rep.tokens[i + 1].timing
            assert nxt.start_time_ms > curr.start_time_ms
            assert nxt.start_time_ms >= curr.start_time_ms + curr.lead_in_duration_ms
