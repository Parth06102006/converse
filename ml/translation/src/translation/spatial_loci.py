"""Session-aware spatial locus and referent indexing tracker."""

from dataclasses import dataclass
from typing import Literal

SpatialAnchor = Literal[
    "neutral_space",
    "chest",
    "forehead",
    "left_shoulder",
    "right_shoulder",
]


@dataclass(frozen=True)
class SpatialOffset:
    """3D Cartesian target coordinate offset in meters relative to anchor joint."""

    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3)}


@dataclass(frozen=True)
class SpatialLociTarget:
    """Spatial anchor and target 3D offset for pronoun indexing and directional verbs."""

    anchor: SpatialAnchor
    target_offset: SpatialOffset

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor": self.anchor,
            "targetOffset": self.target_offset.to_dict(),
        }


# Canonical 3D spatial loci offsets in normalized sign space
LOCI_COORDINATES: dict[str, SpatialLociTarget] = {
    "neutral_space": SpatialLociTarget(
        anchor="neutral_space",
        target_offset=SpatialOffset(x=0.0, y=0.0, z=0.35),
    ),
    "chest": SpatialLociTarget(
        anchor="chest",
        target_offset=SpatialOffset(x=0.0, y=0.0, z=0.10),
    ),
    "left": SpatialLociTarget(
        anchor="neutral_space",
        target_offset=SpatialOffset(x=-0.30, y=0.0, z=0.40),
    ),
    "right": SpatialLociTarget(
        anchor="neutral_space",
        target_offset=SpatialOffset(x=0.30, y=0.0, z=0.40),
    ),
    "contralateral": SpatialLociTarget(
        anchor="left_shoulder",
        target_offset=SpatialOffset(x=-0.20, y=0.10, z=0.30),
    ),
    "forehead": SpatialLociTarget(
        anchor="forehead",
        target_offset=SpatialOffset(x=0.0, y=0.25, z=0.20),
    ),
}

SELF_REFERENT_GLOSSES: set[str] = {"ME", "I", "MY", "MINE", "MYSELF"}
ADDRESSEE_REFERENT_GLOSSES: set[str] = {"YOU", "YOUR", "YOURS"}
THIRD_PERSON_GLOSSES: set[str] = {"HE", "SHE", "IT", "THEY", "HIM", "HER", "THEM", "THAT"}


class SpatialLociTracker:
    """Tracks session-level 3D spatial locus assignments for referents and pronouns."""

    def __init__(self) -> None:
        # Mapping from referent key/name to locus identifier ('left', 'right', etc.)
        self._session_referents: dict[str, str] = {}
        self._available_loci: list[str] = ["left", "right"]
        self._assigned_order: list[str] = []

    def assign_referent(self, entity_name: str) -> SpatialLociTarget:
        """Assign an entity to an available spatial locus in the signing space."""
        normalized = entity_name.upper().strip()

        if normalized in self._session_referents:
            locus_name = self._session_referents[normalized]
            return LOCI_COORDINATES[locus_name]

        # Assign next available spatial locus or alternate
        if self._available_loci:
            chosen_locus = self._available_loci.pop(0)
        else:
            # Alternate between left and right once capacity is reached
            chosen_locus = "left" if len(self._assigned_order) % 2 == 0 else "right"

        self._session_referents[normalized] = chosen_locus
        self._assigned_order.append(normalized)
        return LOCI_COORDINATES[chosen_locus]

    def resolve_locus_for_gloss(self, gloss: str, previous_entities: list[str] | None = None) -> SpatialLociTarget:
        """Resolve the 3D target coordinates and anchor for a given sign gloss."""
        upper = gloss.upper().strip()

        # 1. Self references always index toward chest
        if upper in SELF_REFERENT_GLOSSES:
            return LOCI_COORDINATES["chest"]

        # 2. Addressee references index toward neutral forward space
        if upper in ADDRESSEE_REFERENT_GLOSSES:
            return LOCI_COORDINATES["neutral_space"]

        # 3. Third person pronouns or explicit named entities resolve to active referent locus
        if upper in self._session_referents:
            return LOCI_COORDINATES[self._session_referents[upper]]

        if upper in THIRD_PERSON_GLOSSES:
            if previous_entities and previous_entities[0] in self._session_referents:
                locus = self._session_referents[previous_entities[0]]
                return LOCI_COORDINATES[locus]
            if self._assigned_order:
                # Use most recently assigned referent locus
                most_recent = self._assigned_order[-1]
                return LOCI_COORDINATES[self._session_referents[most_recent]]
            # Default third person to left locus
            return LOCI_COORDINATES["left"]

        # 4. Cognitive / mental words map toward forehead anchor
        if upper in ("KNOW", "THINK", "REMEMBER", "FORGET", "UNDERSTAND"):
            return LOCI_COORDINATES["forehead"]

        # Default sign gesture executed in neutral conversational space
        return LOCI_COORDINATES["neutral_space"]

    def reset(self) -> None:
        """Clear active referents upon discourse topic change or session end."""
        self._session_referents.clear()
        self._available_loci = ["left", "right"]
        self._assigned_order.clear()
