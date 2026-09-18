"""Out-of-Vocabulary (OOV) detection and fingerspelling decomposition."""

import re
from dataclasses import dataclass

# Canonical baseline vocabulary of common ASL signs
CANONICAL_ASL_LEXICON: set[str] = {
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
