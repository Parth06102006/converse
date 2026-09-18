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
from translation.timing_model import (
    SignTiming,
    TimingConfig,
    TimingModel,
)

__all__ = [
    "CANONICAL_ASL_LEXICON",
    "AslGlossToken",
    "EyebrowMarker",
    "FingerspellResolver",
    "FingerspellResult",
    "GlossCompilationResult",
    "GlossCompiler",
    "GrammarRuleCompiler",
    "GrammarTransformResult",
    "HeadMotionMarker",
    "NmmDetector",
    "NonManualMarkers",
    "SentenceType",
    "SignTiming",
    "TimingConfig",
    "TimingModel",
    "Token",
]


def main() -> None:
    print("Converse translation engine initialized")
