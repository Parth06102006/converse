"""Unit tests for English-to-ASL grammar transformation rules."""

from translation.grammar_rules import GrammarRuleCompiler, SentenceType


class TestGrammarRuleCompiler:
    """Test suite for syntax restructuring, copula/article deletion, and topicalization."""

    def setup_method(self) -> None:
        self.compiler = GrammarRuleCompiler()

    def test_copula_and_article_deletion(self) -> None:
        # "The car is blue" -> ["CAR", "BLUE"]
        res = self.compiler.transform("The car is blue")
        assert "CAR" in res.glosses
        assert "BLUE" in res.glosses
        assert "IS" not in res.glosses
        assert "THE" not in res.glosses
        assert res.sentence_type == SentenceType.STATEMENT

    def test_article_deletion(self) -> None:
        # "I see a book" -> ["ME", "SEE", "BOOK"]
        res = self.compiler.transform("I see a book")
        assert res.glosses == ["ME", "SEE", "BOOK"]

    def test_time_adverbial_fronting(self) -> None:
        # "I will go to market tomorrow" -> ["TOMORROW", "ME", "GO", "TO", "MARKET"]
        res = self.compiler.transform("I will go to the market tomorrow")
        assert len(res.glosses) > 0
        assert res.glosses[0] == "TOMORROW"
        assert "MARKET" in res.glosses
        assert "THE" not in res.glosses

    def test_wh_question_restructuring(self) -> None:
        # "Where do you live?" -> ["YOU", "LIVE", "WHERE"]
        res = self.compiler.transform("Where do you live?")
        assert res.sentence_type == SentenceType.WH_QUESTION
        assert res.wh_word == "WHERE"
        # "DO" is dropped as do-support auxiliary
        assert "DO" not in res.glosses
        # Wh-word moved to the end
        assert res.glosses[-1] == "WHERE"

    def test_wh_question_what_your_name(self) -> None:
        # "What is your name?" -> ["YOUR", "NAME", "WHAT"]
        res = self.compiler.transform("What is your name?")
        assert res.sentence_type == SentenceType.WH_QUESTION
        assert res.glosses[-1] == "WHAT"
        assert "IS" not in res.glosses

    def test_yes_no_question_detection(self) -> None:
        # "Are you tired?" -> ["YOU", "TIRED"]
        res = self.compiler.transform("Are you tired?")
        assert res.sentence_type == SentenceType.YN_QUESTION
        assert "ARE" not in res.glosses
        assert "YOU" in res.glosses
        assert "TIRED" in res.glosses

    def test_negation_particle_placement(self) -> None:
        # "I do not want tea" -> ["ME", "WANT", "TEA", "NOT"]
        res = self.compiler.transform("I do not want tea")
        assert res.sentence_type == SentenceType.NEGATION
        assert res.has_negation
        assert res.glosses[-1] == "NOT"
        assert "DO" not in res.glosses

    def test_empty_string(self) -> None:
        res = self.compiler.transform("")
        assert res.glosses == []
        assert res.sentence_type == SentenceType.STATEMENT
