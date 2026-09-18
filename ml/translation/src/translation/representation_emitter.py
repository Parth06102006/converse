"""Emission of authoritative SignRepresentation payloads for 3D avatar rendering."""

from dataclasses import dataclass
from typing import Literal

from translation.gloss_compiler import AslGlossToken, GlossCompilationResult
from translation.spatial_loci import SpatialLociTarget, SpatialLociTracker
from translation.timing_model import SignTiming

InterpolationCurve = Literal["linear", "ease_in_out", "bezier_slerp"]


@dataclass(frozen=True)
class HeadRotation:
    """Head orientation Euler angles in radians (pitch, yaw, roll)."""

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "pitch": round(self.pitch, 3),
            "yaw": round(self.yaw, 3),
            "roll": round(self.roll, 3),
        }


@dataclass(frozen=True)
class NonManualMarkerDirectives:
    """Facial blendshape and skeletal head directives for avatar rendering."""

    eyebrow_intensity: float
    eyebrow_shape: Literal["furrow", "raise", "neutral"]
    head_rotation: HeadRotation
    mouth_shape: str

    def to_dict(self) -> dict[str, object]:
        return {
            "eyebrowIntensity": round(self.eyebrow_intensity, 2),
            "eyebrowShape": self.eyebrow_shape,
            "headRotation": self.head_rotation.to_dict(),
            "mouthShape": self.mouth_shape,
        }


@dataclass(frozen=True)
class SignRepresentationToken:
    """Complete, rendered ASL token ready for skeletal WebGL playback."""

    token_id: str
    clip_id: str
    gloss: str
    timing: SignTiming
    spatial_loci: SpatialLociTarget
    non_manual_markers: NonManualMarkerDirectives
    interpolation_curve: InterpolationCurve = "bezier_slerp"

    def to_dict(self) -> dict[str, object]:
        return {
            "tokenId": self.token_id,
            "clipId": self.clip_id,
            "gloss": self.gloss,
            "timing": self.timing.to_dict(),
            "spatialLoci": self.spatial_loci.to_dict(),
            "nonManualMarkers": self.non_manual_markers.to_dict(),
            "interpolationCurve": self.interpolation_curve,
        }


@dataclass(frozen=True)
class SignRepresentation:
    """Canonical SignRepresentation contract conforming to @converse/contracts."""

    session_id: str
    utterance_id: str
    total_duration_ms: float
    tokens: list[SignRepresentationToken]
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sessionId": self.session_id,
            "utteranceId": self.utterance_id,
            "totalDurationMs": round(self.total_duration_ms, 1),
            "tokens": [t.to_dict() for t in self.tokens],
        }


class SignRepresentationEmitter:
    """Compiles GlossCompilationResult into resolved SignRepresentation schemas."""

    def __init__(
        self,
        default_interpolation: InterpolationCurve = "bezier_slerp",
        expand_fingerspelling: bool = True,
    ) -> None:
        self.default_interpolation = default_interpolation
        self.expand_fingerspelling = expand_fingerspelling

    def build_nmm_directives(self, token: AslGlossToken) -> NonManualMarkerDirectives:
        """Translate abstract NonManualMarkers into blendshape intensities and Euler rotations."""
        nmm = token.non_manual_markers

        # Eyebrows
        if nmm.eyebrows == "furrowed":
            eb_shape: Literal["furrow", "raise", "neutral"] = "furrow"
            eb_intensity = 0.85
        elif nmm.eyebrows == "raised":
            eb_shape = "raise"
            eb_intensity = 0.85
        else:
            eb_shape = "neutral"
            eb_intensity = 0.0

        # Head motion
        if nmm.head_motion == "shake":
            rotation = HeadRotation(pitch=0.0, yaw=0.25, roll=0.0)
        elif nmm.head_motion == "tilt_forward":
            rotation = HeadRotation(pitch=0.15, yaw=0.0, roll=0.0)
        elif nmm.head_motion == "nod":
            rotation = HeadRotation(pitch=0.20, yaw=0.0, roll=0.0)
        else:
            rotation = HeadRotation(pitch=0.0, yaw=0.0, roll=0.0)

        # Mouth shape
        mouth = nmm.mouth_morpheme or "neutral"

        return NonManualMarkerDirectives(
            eyebrow_intensity=eb_intensity,
            eyebrow_shape=eb_shape,
            head_rotation=rotation,
            mouth_shape=mouth,
        )

    def emit(
        self,
        compilation: GlossCompilationResult,
        session_id: str,
        utterance_id: str,
        spatial_tracker: SpatialLociTracker | None = None,
    ) -> SignRepresentation:
        """Transform compilation result into authoritative SignRepresentation payload."""
        tracker = spatial_tracker or SpatialLociTracker()
        tokens: list[SignRepresentationToken] = []

        for idx, tok in enumerate(compilation.tokens):
            timing = tok.timing or SignTiming(
                start_time_ms=idx * 400.0,
                lead_in_duration_ms=100.0,
                hold_duration_ms=250.0,
                lead_out_duration_ms=100.0,
            )

            # Spatial locus resolution: if a named entity or fingerspelled proper noun is introduced,
            # assign it an active 3D locus in the session discourse space
            if tok.is_fingerspelled:
                tracker.assign_referent(tok.gloss)

            spatial_target = tracker.resolve_locus_for_gloss(tok.gloss)
            nmm_directives = self.build_nmm_directives(tok)

            if self.expand_fingerspelling and tok.is_fingerspelled and tok.fingerspell_sequence:
                # Expand fingerspelling into individual letter tokens
                num_letters = len(tok.fingerspell_sequence)
                letter_hold = timing.hold_duration_ms / max(1, num_letters)
                letter_lead_in = timing.lead_in_duration_ms
                letter_lead_out = timing.lead_out_duration_ms

                for l_idx, char in enumerate(tok.fingerspell_sequence):
                    char_lower = char.lower()
                    char_token_id = f"tok_{idx}_{l_idx}_{char_lower}"
                    char_clip_id = f"asl_fs_{char_lower}_01"
                    char_start = timing.start_time_ms + (l_idx * letter_hold)

                    char_timing = SignTiming(
                        start_time_ms=round(char_start, 1),
                        lead_in_duration_ms=round(letter_lead_in if l_idx == 0 else 30.0, 1),
                        hold_duration_ms=round(letter_hold, 1),
                        lead_out_duration_ms=round(letter_lead_out if l_idx == num_letters - 1 else 30.0, 1),
                    )

                    rep_token = SignRepresentationToken(
                        token_id=char_token_id,
                        clip_id=char_clip_id,
                        gloss=char.upper(),
                        timing=char_timing,
                        spatial_loci=spatial_target,
                        non_manual_markers=nmm_directives,
                        interpolation_curve=self.default_interpolation,
                    )
                    tokens.append(rep_token)
            else:
                # Standard lexical sign token
                gloss_lower = tok.gloss.lower()
                token_id = f"tok_{idx}_{gloss_lower}"
                clip_id = f"asl_{gloss_lower}_01"

                rep_token = SignRepresentationToken(
                    token_id=token_id,
                    clip_id=clip_id,
                    gloss=tok.gloss,
                    timing=timing,
                    spatial_loci=spatial_target,
                    non_manual_markers=nmm_directives,
                    interpolation_curve=self.default_interpolation,
                )
                tokens.append(rep_token)

        total_duration = 0.0
        if tokens:
            last = tokens[-1]
            total_duration = last.timing.start_time_ms + last.timing.total_duration_ms

        return SignRepresentation(
            version="1.0.0",
            session_id=session_id,
            utterance_id=utterance_id,
            total_duration_ms=round(total_duration, 1),
            tokens=tokens,
        )
