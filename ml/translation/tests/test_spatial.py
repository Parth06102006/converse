"""Unit tests for spatial locus referent tracking."""

from translation.spatial_loci import SpatialLociTracker


class TestSpatialLociTracker:
    """Test suite for spatial loci assignment, 3D target coordinates, and pronoun indexing."""

    def setup_method(self) -> None:
        self.tracker = SpatialLociTracker()

    def test_self_referent_indexes_chest(self) -> None:
        target = self.tracker.resolve_locus_for_gloss("ME")
        assert target.anchor == "chest"
        assert target.target_offset.z == 0.10

        target_my = self.tracker.resolve_locus_for_gloss("MY")
        assert target_my.anchor == "chest"

    def test_addressee_indexes_neutral_space(self) -> None:
        target = self.tracker.resolve_locus_for_gloss("YOU")
        assert target.anchor == "neutral_space"
        assert target.target_offset.z == 0.35

    def test_cognitive_words_index_forehead(self) -> None:
        target = self.tracker.resolve_locus_for_gloss("KNOW")
        assert target.anchor == "forehead"
        assert target.target_offset.y == 0.25

        target_think = self.tracker.resolve_locus_for_gloss("THINK")
        assert target_think.anchor == "forehead"

    def test_third_person_entity_assignment(self) -> None:
        # First entity assigned to 'left'
        locus1 = self.tracker.assign_referent("BOB")
        assert locus1.target_offset.x == -0.30

        # Second entity assigned to 'right'
        locus2 = self.tracker.assign_referent("ALICE")
        assert locus2.target_offset.x == 0.30

        # Re-querying BOB returns the same cached locus
        locus_bob = self.tracker.assign_referent("BOB")
        assert locus_bob.target_offset.x == -0.30

    def test_pronoun_resolves_to_active_referent(self) -> None:
        self.tracker.assign_referent("JOHN")
        target_he = self.tracker.resolve_locus_for_gloss("HE")
        # Should resolve to John's locus ('left')
        assert target_he.target_offset.x == -0.30

    def test_tracker_reset(self) -> None:
        self.tracker.assign_referent("BOB")
        self.tracker.reset()
        # After reset, first new entity should receive 'left' again
        locus = self.tracker.assign_referent("CAROL")
        assert locus.target_offset.x == -0.30

    def test_discourse_entity_and_pronoun_binding_across_turns(self) -> None:
        from translation.pipeline import SpeechToSignPipeline

        pipeline = SpeechToSignPipeline()
        session_id = "discourse_session_01"

        # Turn 1: Alice is introduced (OOV name -> assigned left locus)
        rep1 = pipeline.translate("Alice arrived yesterday.", session_id=session_id)
        alice_toks = [t for t in rep1.tokens if t.gloss in ("A", "L", "I", "C", "E") or "alice" in t.token_id.lower()]
        assert len(alice_toks) > 0
        assert alice_toks[0].spatial_loci.target_offset.x == -0.30  # left

        # Turn 2: "She wants coffee." -> SHE should resolve to Alice's locus (left)
        rep2 = pipeline.translate("She wants coffee.", session_id=session_id)
        she_tok = next(t for t in rep2.tokens if t.gloss == "SHE")
        assert she_tok.spatial_loci.target_offset.x == -0.30  # points to left where Alice was established

        # Reset session discourse
        pipeline.close_session(session_id)

        # Turn 3: New session -> new entity gets left locus
        rep3 = pipeline.translate("Bob is happy.", session_id=session_id)
        bob_toks = [t for t in rep3.tokens if t.gloss in ("B", "O") or "bob" in t.token_id.lower()]
        assert len(bob_toks) > 0
        assert bob_toks[0].spatial_loci.target_offset.x == -0.30

    def test_single_referent_anaphora_binding(self) -> None:
        """ALICE -> SHE: unambiguous single referent in session binds directly to ALICE."""
        from translation.pipeline import SpeechToSignPipeline

        pipeline = SpeechToSignPipeline()
        session_id = "single_referent_sess"

        # Turn 1: Alice introduced
        pipeline.translate("Alice arrived.", session_id=session_id)
        # Turn 2: She refers to Alice
        rep2 = pipeline.translate("She is happy.", session_id=session_id)
        she_tok = next(t for t in rep2.tokens if t.gloss == "SHE")
        assert she_tok.spatial_loci.target_offset.x == -0.30  # bound to Alice's 'left' locus

    def test_multiple_referents_ambiguous_anaphora_fallback_neutral(self) -> None:
        """ALICE -> BOB -> SHE: multiple referents without explicit antecedent falls back to neutral space."""
        from translation.pipeline import SpeechToSignPipeline

        pipeline = SpeechToSignPipeline()
        session_id = "multi_referent_sess"

        # Turn 1: Alice introduced -> left locus (-0.30)
        pipeline.translate("Alice arrived.", session_id=session_id)
        # Turn 2: Bob introduced -> right locus (+0.30)
        pipeline.translate("Bob arrived.", session_id=session_id)
        # Turn 3: "She is happy." -> multiple referents, ambiguous antecedent -> neutral_space
        rep3 = pipeline.translate("She is happy.", session_id=session_id)
        she_tok = next(t for t in rep3.tokens if t.gloss == "SHE")
        assert she_tok.spatial_loci.anchor == "neutral_space"
        assert she_tok.spatial_loci.target_offset.x == 0.0  # neutral space center, NOT Bob's right locus (+0.30)

    def test_oov_verb_does_not_allocate_spatial_locus(self) -> None:
        """OOV verb ('left') must NOT be allocated as a discourse referent in spatial tracker."""
        from translation.pipeline import SpeechToSignPipeline

        pipeline = SpeechToSignPipeline()
        session_id = "oov_verb_sess"

        rep = pipeline.translate("Alice left.", session_id=session_id)
        assert len(rep.tokens) > 0
        session_tracker = pipeline._session_trackers[session_id]

        # Only ALICE should be a discourse referent; LEFT must NOT be registered
        assert "ALICE" in session_tracker._session_referents
        assert "LEFT" not in session_tracker._session_referents
        assert len(session_tracker._session_referents) == 1

