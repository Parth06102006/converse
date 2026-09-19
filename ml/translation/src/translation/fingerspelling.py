"""Out-of-Vocabulary (OOV) detection and fingerspelling decomposition.

Linguistic Policy & Scope:
- This module uses a curated prototype core lexicon (PROTOTYPE_ASL_LEXICON)
  derived from high-frequency core concepts in How2Sign and ASLG-PC12 corpora.
- Scope: ~120 basic vocabulary items (pronouns, question words, time markers,
  common verbs, and common objects).
- Limitations: Does not cover the comprehensive ASL lexicon, regional variations,
  or classifier predicates.
- MVP OOV Policy: In this system, any token absent from the prototype lexicon
  is decomposed into a sequence of fingerspelling tokens. In natural ASL,
  unmapped concepts may be conveyed via loan signs, classifiers, or
  circumlocution; character-by-character fingerspelling is a deliberate
  engineering fallback to ensure every English word remains renderable by
  the downstream 3D avatar engine.
"""

import re
from dataclasses import dataclass

# Curated prototype vocabulary of core ASL signs
PROTOTYPE_ASL_LEXICON: set[str] = {
    # Pronouns & Pointing
    "ME",
    "I",
    "YOU",
    "HE",
    "SHE",
    "IT",
    "WE",
    "THEY",
    "MY",
    "YOUR",
    "HIS",
    "HER",
    "OUR",
    "THEIR",
    "THIS",
    "THAT",
    # Question Words
    "WHO",
    "WHAT",
    "WHERE",
    "WHEN",
    "WHY",
    "HOW",
    "WHICH",
    # Time & Temporal
    "NOW",
    "TODAY",
    "YESTERDAY",
    "TOMORROW",
    "MORNING",
    "AFTERNOON",
    "NIGHT",
    "TONIGHT",
    "SOON",
    "LATER",
    "BEFORE",
    "AFTER",
    "PAST",
    "FUTURE",
    "ALWAYS",
    "NEVER",
    "TIME",
    "DAY",
    "WEEK",
    "MONTH",
    "YEAR",
    # Common Verbs
    "GO",
    "COME",
    "ARRIVE",
    "LEAVE",
    "SEE",
    "LOOK",
    "KNOW",
    "WANT",
    "NEED",
    "LIKE",
    "LOVE",
    "HAVE",
    "EAT",
    "DRINK",
    "SLEEP",
    "HELP",
    "TALK",
    "TELL",
    "SAY",
    "SIGN",
    "LEARN",
    "TEACH",
    "WORK",
    "PLAY",
    "BUY",
    "SELL",
    "MAKE",
    "TAKE",
    "GIVE",
    "MEET",
    "THINK",
    "FEEL",
    "UNDERSTAND",
    "FORGET",
    "REMEMBER",
    "LIVE",
    "RUN",
    "WALK",
    "DRIVE",
    "WAIT",
    "STOP",
    "START",
    "FINISH",
    # Common Nouns
    "STORE",
    "MARKET",
    "HOME",
    "HOUSE",
    "SCHOOL",
    "CAR",
    "BOOK",
    "WATER",
    "FOOD",
    "TEA",
    "COFFEE",
    "FRIEND",
    "FAMILY",
    "MOTHER",
    "FATHER",
    "BROTHER",
    "SISTER",
    "MAN",
    "WOMAN",
    "CHILD",
    "STUDENT",
    "TEACHER",
    "DOCTOR",
    "NAME",
    "MONEY",
    "PHONE",
    "COMPUTER",
    # Negation & Particles
    "NOT",
    "NO",
    "YES",
    "NONE",
    "CAN",
    "CANNOT",
    # Adjectives & Feelings
    "GOOD",
    "BAD",
    "HAPPY",
    "SAD",
    "TIRED",
    "FINE",
    "BUSY",
    "SICK",
    "BIG",
    "SMALL",
    "FAST",
    "SLOW",
    "HOT",
    "COLD",
    "NEW",
    "OLD",
    "RIGHT",
    "WRONG",
    "EASY",
    "HARD",
    "NICE",
    # Politeness & Greetings
    "HELLO",
    "BYE",
    "PLEASE",
    "THANK-YOU",
    "WELCOME",
    "SORRY",
    "AGAIN",
}

# Backward compatibility alias
CANONICAL_ASL_LEXICON = PROTOTYPE_ASL_LEXICON


@dataclass(frozen=True)
class FingerspellResult:
    """Result of vocabulary check and fingerspelling resolution."""

    gloss: str
    is_fingerspelled: bool
    sequence: list[str] | None = None


class FingerspellResolver:
    """Identifies out-of-vocabulary terms and decomposes them into fingerspelled characters."""

    def __init__(self, custom_lexicon: set[str] | None = None) -> None:
        self.lexicon: set[str] = set(CANONICAL_ASL_LEXICON)
        if custom_lexicon:
            self.lexicon.update(g.upper() for g in custom_lexicon)

    def is_in_lexicon(self, gloss: str) -> bool:
        """Check if a gloss exists in the avatar signing vocabulary."""
        return gloss.upper() in self.lexicon

    def resolve(self, gloss: str) -> FingerspellResult:
        """Resolve whether to emit a lexical sign or fingerspelled sequence."""
        cleaned = gloss.upper().strip()

        if self.is_in_lexicon(cleaned):
            return FingerspellResult(
                gloss=cleaned,
                is_fingerspelled=False,
                sequence=None,
            )

        # OOV: Decompose into individual characters [A-Z0-9]
        chars = [c for c in re.sub(r"[^A-Z0-9]", "", cleaned)]
        if not chars:
            chars = [cleaned]

        return FingerspellResult(
            gloss=cleaned,
            is_fingerspelled=True,
            sequence=chars,
        )
