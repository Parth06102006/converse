"""Main ASL gloss compiler uniting grammar, NMM, fingerspelling, and timing."""

from dataclasses import dataclass, field

from translation.fingerspelling import FingerspellResolver
from translation.grammar_rules import GrammarRuleCompiler, SentenceType
from translation.nmm_detector import NmmDetector, NonManualMarkers
from translation.timing_model import SignTiming, TimingConfig, TimingModel


@dataclass(frozen=True)
class AslGlossToken:
    """Canonical ASL gloss token conforming to @converse/contracts."""

    gloss: str
    lemma: str
    part_of_speech: str
    non_manual_markers: NonManualMarkers
    spatial_locus: str | None = None
    is_fingerspelled: bool = False
    fingerspell_sequence: list[str] | None = None
    timing: SignTiming | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "gloss": self.gloss,
            "lemma": self.lemma,
            "partOfSpeech": self.part_of_speech,
            "nonManualMarkers": self.non_manual_markers.to_dict(),
            "isFingerspelled": self.is_fingerspelled,
        }
        if self.spatial_locus is not None:
            data["spatialLocus"] = self.spatial_locus
        if self.fingerspell_sequence is not None:
            data["fingerspellSequence"] = self.fingerspell_sequence
        if self.timing is not None:
            data["timing"] = self.timing.to_dict()
        return data


@dataclass(frozen=True)
class GlossCompilationResult:
    """Result of compiling English text into sequenced ASL gloss tokens."""

    english_text: str
    sentence_type: SentenceType
    tokens: list[AslGlossToken] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def gloss_sequence(self) -> list[str]:
        return [t.gloss for t in self.tokens]

    def to_dict(self) -> dict[str, object]:
        return {
            "englishText": self.english_text,
            "sentenceType": self.sentence_type.value,
            "totalDurationMs": self.total_duration_ms,
            "tokens": [t.to_dict() for t in self.tokens],
        }


class GlossCompiler:
    """Compiles natural English text into rich ASL gloss tokens with NMM and timing."""

    def __init__(
        self,
        grammar_compiler: GrammarRuleCompiler | None = None,
        nmm_detector: NmmDetector | None = None,
        fingerspell_resolver: FingerspellResolver | None = None,
        timing_model: TimingModel | None = None,
    ) -> None:
        self.grammar = grammar_compiler or GrammarRuleCompiler()
        self.nmm = nmm_detector or NmmDetector()
        self.fingerspell = fingerspell_resolver or FingerspellResolver()
        self.timing = timing_model or TimingModel(TimingConfig())

    def compile(self, english_text: str) -> GlossCompilationResult:
        """Execute full compilation pipeline from English sentence to sequenced ASL tokens."""
        # 1. Grammar Transformation
        transform_result = self.grammar.transform(english_text)
        if not transform_result.tokens:
            return GlossCompilationResult(
                english_text=english_text,
                sentence_type=transform_result.sentence_type,
                tokens=[],
                total_duration_ms=0.0,
            )

        # 2. Sentence-level Non-Manual Markers
        sentence_nmm = self.nmm.detect_sentence_markers(
            sentence_type=transform_result.sentence_type,
            has_negation=transform_result.has_negation,
        )

        # 3. OOV Resolution & Token-level NMM assignment
        raw_gloss_tokens: list[AslGlossToken] = []
        timing_specs: list[dict[str, object]] = []

        for parsed_tok in transform_result.tokens:
            fs_res = self.fingerspell.resolve(parsed_tok.gloss)
            tok_nmm = self.nmm.detect_token_markers(
                gloss=fs_res.gloss,
                sentence_markers=sentence_nmm,
                is_negation_token=parsed_tok.is_negation,
            )

            raw_gloss_tokens.append(
                AslGlossToken(
                    gloss=fs_res.gloss,
                    lemma=parsed_tok.lemma,
                    part_of_speech=parsed_tok.pos,
                    non_manual_markers=tok_nmm,
                    is_fingerspelled=fs_res.is_fingerspelled,
                    fingerspell_sequence=fs_res.sequence,
                )
            )

            timing_specs.append({
                "is_fingerspelled": fs_res.is_fingerspelled,
                "letter_count": len(fs_res.sequence) if fs_res.sequence else 1,
                "emphasis": parsed_tok.is_wh or parsed_tok.is_negation,
            })

        # 4. Calculate Timings
        timings = self.timing.calculate_sequence_timings(timing_specs)

        # 5. Attach timings to tokens and determine total duration
        final_tokens: list[AslGlossToken] = []
        total_duration = 0.0

        for tok, timing in zip(raw_gloss_tokens, timings, strict=False):
            timed_token = AslGlossToken(
                gloss=tok.gloss,
                lemma=tok.lemma,
                part_of_speech=tok.part_of_speech,
                non_manual_markers=tok.non_manual_markers,
                spatial_locus=tok.spatial_locus,
                is_fingerspelled=tok.is_fingerspelled,
                fingerspell_sequence=tok.fingerspell_sequence,
                timing=timing,
            )
            final_tokens.append(timed_token)
            total_duration = max(total_duration, timing.start_time_ms + timing.total_duration_ms)

        return GlossCompilationResult(
            english_text=english_text,
            sentence_type=transform_result.sentence_type,
            tokens=final_tokens,
            total_duration_ms=round(total_duration, 1),
        )
