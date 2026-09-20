"""Unit tests for 1,500+ ASL meeting lexicon expansion, lemma rules, and latency."""

import time

from translation.fingerspelling import PROTOTYPE_ASL_LEXICON, FingerspellResolver
from translation.grammar_rules import (
    LEMMA_MAP,
    GrammarRuleCompiler,
)
from translation.pipeline import SpeechToSignPipeline


class TestMeetingLexiconAndGrammar:
    """Test suite for expanded lexicon, lemma normalization, and idioms."""

    def setup_method(self) -> None:
        self.compiler = GrammarRuleCompiler()
        self.resolver = FingerspellResolver()
        self.pipeline = SpeechToSignPipeline()

    def test_lexicon_size_exceeds_1500(self) -> None:
        """Verify PROTOTYPE_ASL_LEXICON contains at least 1,500 entries."""
        assert len(PROTOTYPE_ASL_LEXICON) >= 1500

    def test_core_meeting_terms_present(self) -> None:
        """Verify all core meeting and workplace terms exist in the lexicon."""
        required_terms = [
            "MEETING",
            "AGENDA",
            "PRESENT",
            "SCREEN",
            "SHARE",
            "DISCUSS",
            "QUESTION",
            "ANSWER",
            "PROJECT",
            "DEADLINE",
            "DESIGN",
            "CODE",
            "REVIEW",
            "APPROVE",
            "REJECT",
            "UPDATE",
            "RELEASE",
            "BUG",
            "FEATURE",
            "TEAM",
            "SCHEDULE",
            "CALENDAR",
            "EMAIL",
            "DOCUMENT",
            "NOTES",
            "SUMMARY",
            "GOAL",
            "PLAN",
            "STATUS",
        ]
        for term in required_terms:
            assert term in PROTOTYPE_ASL_LEXICON
            res = self.resolver.resolve(term)
            assert not res.is_fingerspelled
            assert res.sequence is None
            assert res.gloss == term

    def test_tech_and_engineering_terms(self) -> None:
        """Verify software engineering and technical vocabulary is in the lexicon."""
        tech_terms = [
            "SOFTWARE",
            "HARDWARE",
            "DATABASE",
            "SERVER",
            "CLIENT",
            "API",
            "ENDPOINT",
            "SERVICE",
            "CONTAINER",
            "REPOSITORY",
            "BRANCH",
            "COMMIT",
            "MERGE",
            "PIPELINE",
            "DEPLOY",
            "TEST",
            "DEBUG",
            "MONITOR",
            "SECURITY",
            "AUTHENTICATION",
        ]
        for term in tech_terms:
            assert term in PROTOTYPE_ASL_LEXICON
            assert not self.resolver.resolve(term).is_fingerspelled

    def test_verb_conjugation_lemmatization(self) -> None:
        """Verify verb inflections map to canonical base lemmas."""
        conjugations = {
            "discussed": "discuss",
            "discussing": "discuss",
            "discusses": "discuss",
            "presented": "present",
            "presenting": "present",
            "presents": "present",
            "reviewed": "review",
            "reviewing": "review",
            "reviews": "review",
            "approved": "approve",
            "approving": "approve",
            "approves": "approve",
            "rejected": "reject",
            "rejecting": "reject",
            "rejects": "reject",
            "scheduled": "schedule",
            "scheduling": "schedule",
            "schedules": "schedule",
            "designed": "design",
            "designing": "design",
            "designs": "design",
            "deployed": "deploy",
            "deploying": "deploy",
            "deploys": "deploy",
            "built": "build",
            "building": "build",
            "builds": "build",
            "wrote": "write",
            "written": "write",
            "writing": "write",
            "spoke": "speak",
            "spoken": "speak",
            "speaking": "speak",
        }
        for inflected, base in conjugations.items():
            assert LEMMA_MAP.get(inflected) == base

    def test_plural_noun_lemmatization(self) -> None:
        """Verify plural nouns map to singular lemmas."""
        plurals = {
            "meetings": "meeting",
            "agendas": "agenda",
            "screens": "screen",
            "questions": "question",
            "answers": "answer",
            "projects": "project",
            "deadlines": "deadline",
            "bugs": "bug",
            "features": "feature",
            "teams": "team",
            "calendars": "calendar",
            "emails": "email",
            "documents": "document",
            "notes": "note",
            "summaries": "summary",
            "goals": "goal",
            "plans": "plan",
            "statuses": "status",
            "people": "person",
            "children": "child",
            "men": "man",
            "women": "woman",
        }
        for plural, singular in plurals.items():
            assert LEMMA_MAP.get(plural) == singular

    def test_idiom_normalization(self) -> None:
        """Verify workplace and conversational idioms map to canonical concepts."""
        cases = [
            ("Can we touch base tomorrow?", "CONNECT"),
            ("Let us wrap up the meeting.", "FINISH"),
            ("Please circle back next week.", "REVIEW"),
            ("We need to kick off the sprint.", "START"),
            ("Please follow up with the team.", "CHECK"),
            ("We should look into this bug.", "INVESTIGATE"),
            ("Are we on the same page?", "AGREE"),
        ]
        for phrase, expected_gloss in cases:
            res = self.compiler.transform(phrase)
            assert expected_gloss in res.glosses

    def test_meeting_sentence_no_unwanted_fingerspelling(self) -> None:
        """Verify meeting sentences produce lexical signs without degradation to fingerspelling."""
        rep = self.pipeline.translate(
            "Good morning team, we need to discuss the project deadline and review the new feature."
        )
        # Verify core words are not fingerspelled
        fs_glosses = [t.gloss for t in rep.tokens if t.clip_id.startswith("asl_fs_")]
        assert "TEAM" not in fs_glosses
        assert "DISCUSS" not in fs_glosses
        assert "PROJECT" not in fs_glosses
        assert "DEADLINE" not in fs_glosses
        assert "REVIEW" not in fs_glosses
        assert "FEATURE" not in fs_glosses

    def test_evaluation_latency_under_15ms(self) -> None:
        """Verify end-to-end translation latency stays strictly below 15ms."""
        sentences = [
            "Good morning team, can we discuss the agenda for the meeting tomorrow?",
            "Can we touch base tomorrow to wrap up the sprint backlog?",
            "The engineering team fixed all critical bugs and deployed the release.",
            "Please circle back after you review the pull requests and approve the changes.",
            "We need to schedule a follow up call with the product manager on Friday.",
            "Did you look into the database latency bottleneck on the production server?",
            "The designer updated the wireframes and shared the new UI mockups in the chat.",
            "Our objective is to deliver the feature before the project deadline next week.",
            "Where is the conference room for the architecture review meeting?",
            "I do not think we have enough bandwidth to complete all action items today.",
        ]
        # Warm up
        for s in sentences[:3]:
            self.pipeline.translate(s)

        for s in sentences:
            t0 = time.perf_counter()
            rep = self.pipeline.translate(s)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert elapsed_ms < 15.0, f"Translation latency {elapsed_ms:.2f}ms exceeded 15ms budget"
            assert len(rep.tokens) > 0
