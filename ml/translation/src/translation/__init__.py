"""English to ASL Grammar Transformation and Sign Representation Engine."""

from translation.fingerspelling import (
    CANONICAL_ASL_LEXICON,
    FingerspellResolver,
    FingerspellResult,
)
from translation.gloss_compiler import (
    AslGlossToken,
    GlossCompilationResult,
    GlossCompiler,
)
from translation.grammar_rules import (
    GrammarRuleCompiler,
    GrammarTransformResult,
    SentenceType,
    Token,
)
from translation.nmm_detector import (
    EyebrowMarker,
    HeadMotionMarker,
    NmmDetector,
    NonManualMarkers,
)
from translation.pipeline import (
    SpeechToSignPipeline,
)
from translation.representation_emitter import (
    HeadRotation,
    InterpolationCurve,
    NonManualMarkerDirectives,
    SignRepresentation,
    SignRepresentationEmitter,
    SignRepresentationToken,
)
from translation.spatial_loci import (
    LOCI_COORDINATES,
    SpatialAnchor,
    SpatialLociTarget,
    SpatialLociTracker,
    SpatialOffset,
)
from translation.timing_model import (
    SignTiming,
    TimingConfig,
    TimingModel,
)

__all__ = [
    "CANONICAL_ASL_LEXICON",
    "LOCI_COORDINATES",
    "AslGlossToken",
    "EyebrowMarker",
    "FingerspellResolver",
    "FingerspellResult",
    "GlossCompilationResult",
    "GlossCompiler",
    "GrammarRuleCompiler",
    "GrammarTransformResult",
    "HeadMotionMarker",
    "HeadRotation",
    "InterpolationCurve",
    "NmmDetector",
    "NonManualMarkerDirectives",
    "NonManualMarkers",
    "SentenceType",
    "SignRepresentation",
    "SignRepresentationEmitter",
    "SignRepresentationToken",
    "SignTiming",
    "SpatialAnchor",
    "SpatialLociTarget",
    "SpatialLociTracker",
    "SpatialOffset",
    "SpeechToSignPipeline",
    "TimingConfig",
    "TimingModel",
    "Token",
]


def main() -> None:
    print("Converse translation engine initialized")
