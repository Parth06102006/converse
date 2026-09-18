"""Dynamic sign duration and co-articulation timing calculator."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SignTiming:
    """Calculated timing parameters for an individual sign token."""

    start_time_ms: float
    lead_in_duration_ms: float
    hold_duration_ms: float
    lead_out_duration_ms: float

    @property
    def total_duration_ms(self) -> float:
        return self.lead_in_duration_ms + self.hold_duration_ms + self.lead_out_duration_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "startTimeMs": self.start_time_ms,
            "leadInDurationMs": self.lead_in_duration_ms,
            "holdDurationMs": self.hold_duration_ms,
            "leadOutDurationMs": self.lead_out_duration_ms,
        }


@dataclass(frozen=True)
class TimingConfig:
    """Baseline timing constants and cadence configuration."""

    base_hold_ms: float = 300.0  # Semantic stroke hold duration
    base_lead_in_ms: float = 120.0  # Co-articulation blend into stroke
    base_lead_out_ms: float = 100.0  # Co-articulation blend out of stroke
    fingerspell_letter_ms: float = 120.0  # Per-character hold during fingerspelling
    speed_factor: float = 1.0  # Overall rate multiplier (0.7x slower to 1.5x faster)


class TimingModel:
    """Calculates non-overlapping, monotonically increasing animation timing intervals."""

    def __init__(self, config: TimingConfig | None = None) -> None:
        self.config = config or TimingConfig()

    def compute_token_duration(
        self,
        is_fingerspelled: bool = False,
        letter_count: int = 1,
        emphasis: bool = False,
    ) -> tuple[float, float, float]:
        """Compute (lead_in, hold, lead_out) durations scaled by cadence and emphasis."""
        speed = max(0.5, min(2.0, self.config.speed_factor))

        if is_fingerspelled:
            count = max(1, letter_count)
            hold = (count * self.config.fingerspell_letter_ms) / speed
            lead_in = (self.config.base_lead_in_ms * 0.8) / speed
            lead_out = (self.config.base_lead_out_ms * 0.8) / speed
        else:
            hold = self.config.base_hold_ms / speed
            lead_in = self.config.base_lead_in_ms / speed
            lead_out = self.config.base_lead_out_ms / speed

            if emphasis:
                hold *= 1.35
                lead_in *= 1.1

        return (
            round(lead_in, 1),
            round(hold, 1),
            round(lead_out, 1),
        )

    def calculate_sequence_timings(
        self,
        token_specs: list[dict[str, object]],
        start_offset_ms: float = 0.0,
    ) -> list[SignTiming]:
        """Calculate strictly monotonic, sequential timing for a list of token specifications.

        Each spec may contain:
            - is_fingerspelled: bool
            - letter_count: int
            - emphasis: bool
        """
        timings: list[SignTiming] = []
        current_time_ms = float(start_offset_ms)

        for i, spec in enumerate(token_specs):
            is_fs = bool(spec.get("is_fingerspelled", False))
            letter_count = int(spec.get("letter_count", 1))  # type: ignore[arg-type]
            emphasis = bool(spec.get("emphasis", False))

            lead_in, hold, lead_out = self.compute_token_duration(
                is_fingerspelled=is_fs,
                letter_count=letter_count,
                emphasis=emphasis,
            )

            # In natural co-articulation, the next sign's lead-in overlaps with the
            # previous sign's lead-out
            if i == 0:
                token_start = current_time_ms
            else:
                # Start immediately as the stroke completes, blending through lead-out
                token_start = current_time_ms

            sign_timing = SignTiming(
                start_time_ms=round(token_start, 1),
                lead_in_duration_ms=lead_in,
                hold_duration_ms=hold,
                lead_out_duration_ms=lead_out,
            )
            timings.append(sign_timing)

            # Advance time by the stroke duration plus transition
            # Next sign begins at previous start + lead_in + hold
            current_time_ms = token_start + lead_in + hold

        return timings
