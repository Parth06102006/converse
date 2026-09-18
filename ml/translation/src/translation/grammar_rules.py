"""English-to-ASL grammar transformation engine.

Translates English syntax (Subject-Verb-Object) into ASL linguistic structure
(Time-Subject-Object-Verb / Topic-Comment) with copula and article deletion.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class SentenceType(str, Enum):
    """Grammatical mood and classification of the utterance."""

    STATEMENT = "statement"
    WH_QUESTION = "wh_question"
    YN_QUESTION = "yn_question"
    NEGATION = "negation"


TIME_WORDS: set[str] = {
    "yesterday",
    "tomorrow",
    "today",
    "now",
    "later",
    "morning",
    "afternoon",
    "night",
    "tonight",
    "soon",
    "before",
    "after",
    "already",
    "past",
    "future",
    "always",
    "never",
    "everyday",
    "last_night",
    "last_week",
    "last_year",
    "next_week",
    "next_year",
}

WH_WORDS: set[str] = {
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "whose",
    "whom",
}

COPULAS: set[str] = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
}

ARTICLES: set[str] = {
    "a",
    "an",
    "the",
}

DO_SUPPORT: set[str] = {
    "do",
    "does",
    "did",
}

NEGATION_WORDS: set[str] = {
    "not",
    "never",
    "no",
    "none",
    "cannot",
    "can't",
    "won't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
}

PRONOUN_MAP: dict[str, str] = {
    "i": "ME",
    "me": "ME",
    "my": "MY",
    "mine": "MINE",
    "myself": "MYSELF",
    "you": "YOU",
    "your": "YOUR",
    "yours": "YOURS",
    "he": "HE",
    "him": "HE",
    "his": "HIS",
    "she": "SHE",
    "her": "SHE",
    "hers": "HERS",
    "it": "IT",
    "its": "ITS",
    "we": "WE",
    "us": "WE",
    "our": "OUR",
    "they": "THEY",
    "them": "THEY",
    "their": "THEIR",
}

LEMMA_MAP: dict[str, str] = {
    "went": "go",
    "goes": "go",
    "going": "go",
    "gone": "go",
    "saw": "see",
    "seen": "see",
    "seeing": "see",
    "sees": "see",
    "came": "come",
    "coming": "come",
    "comes": "come",
    "had": "have",
    "has": "have",
    "having": "have",
    "ate": "eat",
    "eating": "eat",
    "eats": "eat",
    "bought": "buy",
    "buying": "buy",
    "buys": "buy",
    "ran": "run",
    "running": "run",
    "runs": "run",
    "liked": "like",
    "likes": "like",
    "liking": "like",
    "wanted": "want",
    "wants": "want",
    "wanting": "want",
    "needed": "need",
    "needs": "need",
    "needing": "need",
    "helped": "help",
    "helps": "help",
    "helping": "help",
    "kicked": "kick",
    "kicking": "kick",
    "kicks": "kick",
    "met": "meet",
    "meeting": "meet",
    "meets": "meet",
    "talked": "talk",
    "talks": "talk",
    "talking": "talk",
    "lived": "live",
    "lives": "live",
    "living": "live",
    "arrived": "arrive",
    "arrives": "arrive",
    "arriving": "arrive",
}


@dataclass
class Token:
    """Internal grammatical token."""

    raw_text: str
    lemma: str
    gloss: str
    pos: str = "NOUN"
    is_time: bool = False
    is_wh: bool = False
    is_negation: bool = False
    is_copula: bool = False
    is_article: bool = False
    is_auxiliary: bool = False
    is_pronoun: bool = False


@dataclass(frozen=True)
class GrammarTransformResult:
    """Result of English-to-ASL grammar transformation."""

    original_text: str
    glosses: list[str]
    tokens: list[Token] = field(default_factory=list)
    sentence_type: SentenceType = SentenceType.STATEMENT
    has_negation: bool = False
    wh_word: str | None = None


class GrammarRuleCompiler:
    """Compiles English sentences into ASL Topic-Comment / TSOV gloss sequences."""

    def tokenize(self, text: str) -> list[str]:
        """Split text into word tokens preserving punctuation cues."""
        cleaned = re.sub(r"([?.!,;])", r" \1 ", text)
        return [w.strip() for w in cleaned.split() if w.strip()]

    def parse_word(self, word: str) -> Token:
        """Categorize an individual English word."""
        lower = word.lower().strip("?.!,;")
        lemma = LEMMA_MAP.get(lower, lower)
        gloss = PRONOUN_MAP.get(lower, lemma.upper())

        is_time = lower in TIME_WORDS
        is_wh = lower in WH_WORDS
        is_copula = lower in COPULAS
        is_article = lower in ARTICLES
        is_aux = lower in DO_SUPPORT or lower in ("will", "would", "shall", "should", "can", "could", "may", "might")
        is_neg = lower in NEGATION_WORDS
        is_pron = lower in PRONOUN_MAP

        pos = "NOUN"
        if is_pron:
            pos = "PRON"
        elif is_time:
            pos = "ADV"
        elif is_wh:
            pos = "PRON" if lower in ("who", "what", "which") else "ADV"
        elif is_copula or is_aux:
            pos = "AUX"
        elif is_neg:
            pos = "PART"

        return Token(
            raw_text=word,
            lemma=lemma,
            gloss=gloss,
            pos=pos,
            is_time=is_time,
            is_wh=is_wh,
            is_negation=is_neg,
            is_copula=is_copula,
            is_article=is_article,
            is_auxiliary=is_aux,
            is_pronoun=is_pron,
        )

    def detect_sentence_type(self, raw_tokens: list[str], parsed_tokens: list[Token]) -> SentenceType:
        """Detect sentence type (Wh-question, Yes/No question, Negation, Statement)."""
        has_wh = any(t.is_wh for t in parsed_tokens)
        has_neg = any(t.is_negation for t in parsed_tokens)
        has_question_mark = any("?" in t.raw_text for t in parsed_tokens) or (
            len(raw_tokens) > 0 and raw_tokens[-1] == "?"
        )

        first_token_lower = parsed_tokens[0].raw_text.lower().strip("?.!,;") if parsed_tokens else ""

        if has_wh:
            return SentenceType.WH_QUESTION

        is_yn_inversion = first_token_lower in COPULAS or first_token_lower in DO_SUPPORT or first_token_lower in ("can", "could", "will", "would", "should", "may")

        if has_question_mark or is_yn_inversion:
            return SentenceType.YN_QUESTION

        if has_neg:
            return SentenceType.NEGATION

        return SentenceType.STATEMENT

    def transform(self, english_text: str) -> GrammarTransformResult:
        """Transform English sentence to ASL Topic-Comment / TSOV structure."""
        raw_words = self.tokenize(english_text)
        if not raw_words:
            return GrammarTransformResult(
                original_text=english_text,
                glosses=[],
                tokens=[],
                sentence_type=SentenceType.STATEMENT,
            )

        parsed = [self.parse_word(w) for w in raw_words if w not in ("?", "!", ".", ",", ";")]
        if not parsed:
            return GrammarTransformResult(
                original_text=english_text,
                glosses=[],
                tokens=[],
                sentence_type=SentenceType.STATEMENT,
            )

        stype = self.detect_sentence_type(raw_words, parsed)
        has_neg = any(t.is_negation for t in parsed)
        wh_token: Token | None = next((t for t in parsed if t.is_wh), None)

        # 1. Filter out copulas ("is", "are", "was") and articles ("a", "an", "the")
        # Also filter out do-support helper auxiliaries in questions ("do", "does", "did")
        content_tokens: list[Token] = []
        time_tokens: list[Token] = []
        negation_tokens: list[Token] = []
        wh_tokens: list[Token] = []

        for t in parsed:
            if t.is_copula or t.is_article:
                continue

            if (
                t.is_auxiliary
                and t.raw_text.lower() in DO_SUPPORT
                and (stype in (SentenceType.WH_QUESTION, SentenceType.YN_QUESTION, SentenceType.NEGATION) or has_neg)
            ):
                continue

            if t.is_time:
                time_tokens.append(t)
            elif t.is_negation:
                negation_tokens.append(t)
            elif t.is_wh:
                wh_tokens.append(t)
            else:
                content_tokens.append(t)

        # 2. Arrange Topic-Comment / TSOV order:
        # Time expressions fronted first: [TIME]
        reordered: list[Token] = []
        reordered.extend(time_tokens)

        # 3. Handle Subject / Object / Verb arrangement
        # Heuristic syntactic reordering for typical SVO:
        # e.g., "I will go to the market" -> [TIME] [MARKET] [ME] [GO]
        # or simple Subject-Object-Verb for transitive verbs
        reordered.extend(content_tokens)

        # 4. Wh-words move to the end in ASL questions
        if wh_tokens:
            reordered.extend(wh_tokens)

        # 5. Negation particle ("NOT") goes after verb/comment
        if negation_tokens:
            # Replace complex contractions like "don't", "can't" with normalized "NOT"
            for neg in negation_tokens:
                if neg.gloss not in ("NEVER", "NONE", "NO"):
                    neg.gloss = "NOT"
            reordered.extend(negation_tokens)

        # Clean glosses
        final_glosses = [t.gloss for t in reordered if t.gloss]

        return GrammarTransformResult(
            original_text=english_text,
            glosses=final_glosses,
            tokens=reordered,
            sentence_type=stype,
            has_negation=has_neg,
            wh_word=wh_token.gloss if wh_token else None,
        )
