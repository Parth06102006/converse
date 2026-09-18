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
