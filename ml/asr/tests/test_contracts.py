"""Contract compliance tests ensuring provider-independent ASR outputs and translation integration."""

import sys
from pathlib import Path

import pytest

# Ensure translation package is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "translation" / "src"))

from test_aws_transcribe import (
    MockAwsAlternative,
    MockAwsItem,
    MockAwsResult,
    MockAwsTranscript,
    MockAwsTranscriptEvent,
    MockTranscribeClient,
)
from translation.pipeline import SpeechToSignPipeline

from asr.aws_transcribe import AwsTranscribeStreamingEngine, BackgroundLoopManager
from asr.config import AsrEngineConfig
from asr.engine import (
    MockAsrBackend,
    WhisperAsrEngine,
)


def assert_conforms_to_contracts(event_dict: dict[str, object]) -> None:
    """Validate that dictionary conforms strictly to @converse/contracts AsrTranscriptEvent."""
    required_keys = {"sessionId", "sequenceId", "text", "isFinal", "confidence", "latencyMetrics", "wordTimestamps"}
    assert required_keys.issubset(event_dict.keys()), f"Missing keys in event: {required_keys - event_dict.keys()}"

    assert isinstance(event_dict["sessionId"], str)
    assert isinstance(event_dict["sequenceId"], int)
    assert isinstance(event_dict["text"], str)
    assert isinstance(event_dict["isFinal"], bool)
    assert isinstance(event_dict["confidence"], (int, float))

    metrics = event_dict["latencyMetrics"]
    assert isinstance(metrics, dict)
    assert "audioDurationMs" in metrics
    assert "processingTimeMs" in metrics
    assert isinstance(metrics["audioDurationMs"], (int, float))
    assert isinstance(metrics["processingTimeMs"], (int, float))

    timestamps = event_dict["wordTimestamps"]
    assert isinstance(timestamps, list)
    for wt in timestamps:
        assert isinstance(wt, dict)
        assert "word" in wt
        assert "startMs" in wt
        assert "endMs" in wt
        assert isinstance(wt["word"], str)
        assert isinstance(wt["startMs"], (int, float))
        assert isinstance(wt["endMs"], (int, float))


class TestAsrContractCompliance:
    """Contract tests for AWS Transcribe and Whisper ASREvents."""

    @pytest.fixture
    def loop_manager(self) -> BackgroundLoopManager:
        mgr = BackgroundLoopManager()
        yield mgr
        mgr.stop()

    def test_aws_transcribe_event_schema_contract(self, loop_manager: BackgroundLoopManager) -> None:
        mock_event = MockAwsTranscriptEvent(
            transcript=MockAwsTranscript(
                results=[
                    MockAwsResult(
                        alternatives=[
                            MockAwsAlternative(
                                transcript="The quick brown fox",
                                items=[
                                    MockAwsItem("The", 0.0, 0.2, 0.99),
                                    MockAwsItem("quick", 0.2, 0.5, 0.95),
                                    MockAwsItem("brown", 0.5, 0.8, 0.94),
                                    MockAwsItem("fox", 0.8, 1.1, 0.97),
                                ],
                            )
                        ],
                        is_partial=False,
                        start_time=0.0,
                        end_time=1.1,
                    )
                ]
            )
        )
        mock_client = MockTranscribeClient(output_events=[mock_event])
        engine = AwsTranscribeStreamingEngine(
            config=AsrEngineConfig(),
            client=mock_client,
            loop_manager=loop_manager,
        )

        ev1 = engine.process_audio_chunk("contract-sess", b"\x00\x00" * 160)
        events = engine.flush_session("contract-sess")
        all_events = ev1 + events

        assert len(all_events) == 1
        event = all_events[0]
        event_dict = event.to_dict()

        assert_conforms_to_contracts(event_dict)
        assert event_dict["text"] == "The quick brown fox"
        assert event_dict["isFinal"] is True
        assert len(event_dict["wordTimestamps"]) == 4

    def test_whisper_event_schema_contract(self) -> None:
        mock_backend = MockAsrBackend(
            default_transcript="The quick brown fox",
            default_confidence=0.96,
        )
        engine = WhisperAsrEngine(backend=mock_backend)
        engine.process_audio_chunk("contract-whisper", b"\x00\x00" * 3200)
        events = engine.flush_session("contract-whisper")

        assert len(events) == 1
        event = events[0]
        event_dict = event.to_dict()

        assert_conforms_to_contracts(event_dict)
        assert event_dict["text"] == "The quick brown fox"

    def test_downstream_translation_agnostic_to_asr_provider(self, loop_manager: BackgroundLoopManager) -> None:
        """Verify downstream translation compiles SignRepresentation identically regardless of ASR source."""
        # 1. AWS Output
        aws_event = MockAwsTranscriptEvent(
            transcript=MockAwsTranscript(
                results=[
                    MockAwsResult(
                        alternatives=[
                            MockAwsAlternative(transcript="I am happy today.")
                        ],
                        is_partial=False,
                    )
                ]
            )
        )
        mock_client = MockTranscribeClient(output_events=[aws_event])
        aws_engine = AwsTranscribeStreamingEngine(client=mock_client, loop_manager=loop_manager)
        aws_ev1 = aws_engine.process_audio_chunk("aws-pipe", b"\x00\x00" * 160)
        aws_events = aws_ev1 + aws_engine.flush_session("aws-pipe")

        # 2. Whisper Output
        whisper_backend = MockAsrBackend(default_transcript="I am happy today.")
        whisper_engine = WhisperAsrEngine(backend=whisper_backend)
        wh_ev1 = whisper_engine.process_audio_chunk("whisper-pipe", b"\x00\x00" * 3200)
        whisper_events = wh_ev1 + whisper_engine.flush_session("whisper-pipe")

        # 3. Feed both to translation pipeline
        pipeline = SpeechToSignPipeline()
        assert len(aws_events) > 0
        assert len(whisper_events) > 0
        rep_aws = pipeline.translate(aws_events[0].text, session_id="aws-pipe", utterance_id="u1")
        rep_whisper = pipeline.translate(whisper_events[0].text, session_id="whisper-pipe", utterance_id="u2")

        # Both produce valid ASL representations with matching glosses
        assert len(rep_aws.tokens) > 0
        assert len(rep_whisper.tokens) > 0
        aws_glosses = [t.gloss for t in rep_aws.tokens]
        whisper_glosses = [t.gloss for t in rep_whisper.tokens]
        assert aws_glosses == whisper_glosses
