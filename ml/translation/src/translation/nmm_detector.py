"""Rule-based Non-Manual Marker (NMM) generator for ASL facial and head grammatical cues.

Design & Architectural Honesty:
- This module implements deterministic linguistic rules that map grammatical
  sentence mood (Wh-questions, Yes/No questions, sentential negation) and lexical
  adjectives to facial and head gesture markers.
- It is NOT a trained statistical or neural classifier; it is a deterministic
  rule-based generator.
- Numerical animation values (e.g. 0.85 for eyebrow furrow intensity, 0.80 for raise,
  and head rotation angles) are calibrated engineering constants designed to drive
  the downstream 3D avatar blendshape controller, not empirically learned values.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from translation.grammar_rules import SentenceType

EyebrowMarker = Literal["neutral", "raised", "furrowed"]
HeadMotionMarker = Literal["neutral", "nod", "shake", "tilt_forward"]


@dataclass(frozen=True)
class NonManualMarkers:
    """Non-manual grammatical cues accompanying sign strokes."""

    eyebrows: EyebrowMarker = "neutral"
    head_motion: HeadMotionMarker = "neutral"
    mouth_morpheme: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        data: dict[str, str | None] = {
            "eyebrows": self.eyebrows,
            "headMotion": self.head_motion,
        }
        if self.mouth_morpheme is not None:
            data["mouthMorpheme"] = self.mouth_morpheme
        return data


class NmmDetector:
    """Deterministic rule-based generator for non-manual markers (NMMs)."""

    # Lexical mouth morpheme mapping
    MOUTH_MORPHEMES: ClassVar[dict[str, str]] = {
        "BIG": "cha",
        "LARGE": "cha",
        "HUGE": "cha",
        "TALL": "cha",
        "HEAVY": "cha",
        "SMALL": "oo",
        "TINY": "oo",
        "THIN": "oo",
        "LITTLE": "oo",
        "LIGHT": "oo",
        "MEDIUM": "mm",
        "NORMAL": "mm",
        "REGULAR": "mm",
        "COMFORTABLE": "mm",
        "EASY": "mm",
        "RELAX": "mm",
    }

    def detect_sentence_markers(
        self,
        sentence_type: SentenceType,
        has_negation: bool = False,
    ) -> NonManualMarkers:
        """Derive sentence-level non-manual markers (eyebrows and head motion)."""
        if sentence_type == SentenceType.WH_QUESTION:
            # Wh-questions: eyebrows furrowed, head forward
            return NonManualMarkers(
                eyebrows="furrowed",
                head_motion="tilt_forward",
            )

        if sentence_type == SentenceType.YN_QUESTION:
            # Yes/No questions: eyebrows raised, head tilted forward
            return NonManualMarkers(
                eyebrows="raised",
                head_motion="tilt_forward",
            )

        if sentence_type == SentenceType.NEGATION or has_negation:
            # Negation: head shake
            return NonManualMarkers(
                eyebrows="neutral",
                head_motion="shake",
            )

        # Standard affirmative statement: neutral
        return NonManualMarkers(
            eyebrows="neutral",
            head_motion="neutral",
        )

    def detect_token_markers(
        self,
        gloss: str,
        sentence_markers: NonManualMarkers,
        is_negation_token: bool = False,
    ) -> NonManualMarkers:
        """Derive token-level markers blending sentence mood and mouth morphemes."""
        mouth = self.MOUTH_MORPHEMES.get(gloss.upper(), None)

        head = sentence_markers.head_motion
        eyebrows = sentence_markers.eyebrows

        if is_negation_token:
            head = "shake"

        return NonManualMarkers(
            eyebrows=eyebrows,
            head_motion=head,
            mouth_morpheme=mouth,
        )
