"""Unified Speech-to-Sign translation pipeline."""

from translation.gloss_compiler import GlossCompiler
from translation.representation_emitter import (
    SignRepresentation,
    SignRepresentationEmitter,
)
from translation.spatial_loci import SpatialLociTracker


class SpeechToSignPipeline:
    """End-to-end English-to-ASL model execution pipeline.

    Coordinates grammar reordering, non-manual marker synthesis, fingerspelling decomposition,
    spatial locus tracking, dynamic timing generation, and canonical SignRepresentation packaging.
    """

    def __init__(
        self,
        compiler: GlossCompiler | None = None,
        emitter: SignRepresentationEmitter | None = None,
    ) -> None:
        self.compiler = compiler or GlossCompiler()
        self.emitter = emitter or SignRepresentationEmitter()
        self._session_trackers: dict[str, SpatialLociTracker] = {}

    def get_or_create_tracker(self, session_id: str) -> SpatialLociTracker:
        """Get or initialize the spatial locus tracker for a given conversation session."""
        if session_id not in self._session_trackers:
            self._session_trackers[session_id] = SpatialLociTracker()
        return self._session_trackers[session_id]

    def close_session(self, session_id: str) -> None:
        """Clear spatial state for terminated session."""
        self._session_trackers.pop(session_id, None)

    def translate(
        self,
        english_text: str,
        session_id: str = "default_session",
        utterance_id: str = "utt_001",
    ) -> SignRepresentation:
        """Translate natural English text directly into an authoritative SignRepresentation."""
        # 1. Compile English into structured ASL glosses with NMM & Timing
        compilation = self.compiler.compile(english_text)

        # 2. Retrieve session spatial referent tracker
        tracker = self.get_or_create_tracker(session_id)

        # 3. Emit canonical SignRepresentation
        representation = self.emitter.emit(
            compilation=compilation,
            session_id=session_id,
            utterance_id=utterance_id,
            spatial_tracker=tracker,
        )

        return representation
