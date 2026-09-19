"""Live integration smoke test for Amazon Transcribe Streaming against real AWS services.

Gated behind RUN_AWS_TRANSCRIBE_TESTS=1.
Uses real AWS credential provider chain and ap-south-1 region.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr.aws_transcribe import AwsTranscribeStreamingEngine, BackgroundLoopManager
from asr.config import AsrEngineConfig
from asr.exceptions import AwsAsrAuthError, AwsAsrServiceUnavailableError

AUDIO_PATH = Path(__file__).resolve().parent / "data" / "speech_sample_16k.wav"


class TestLiveAwsTranscribeStreaming:
    """Live smoke test executing real streaming transcription against AWS Transcribe."""

    @pytest.fixture
    def loop_manager(self) -> BackgroundLoopManager:
        mgr = BackgroundLoopManager()
        yield mgr
        mgr.stop()

    def test_live_aws_transcribe_streaming(self, loop_manager: BackgroundLoopManager) -> None:
        if os.environ.get("RUN_AWS_TRANSCRIBE_TESTS") != "1":
            pytest.skip("RUN_AWS_TRANSCRIBE_TESTS=1 is not set. Skipping live AWS Transcribe integration test.")

        if not AUDIO_PATH.exists():
            pytest.skip(f"Audio fixture {AUDIO_PATH} not found.")

        data, sr = sf.read(str(AUDIO_PATH))
        # Use first 3 seconds of real audio
        sample_audio = data[: sr * 3].astype(np.float32)

        config = AsrEngineConfig(
            backend="aws",
            aws_region=os.environ.get("AWS_REGION", "ap-south-1"),
            aws_transcribe_language=os.environ.get("AWS_TRANSCRIBE_LANGUAGE", "en-IN"),
            sample_rate=sr,
        )

        engine = AwsTranscribeStreamingEngine(config=config, loop_manager=loop_manager)
        session_id = "live-aws-smoke-test"

        try:
            # Send in 200ms chunks
            chunk_size = int(sr * 0.2)
            all_events = []
            for i in range(0, len(sample_audio), chunk_size):
                chunk = sample_audio[i : i + chunk_size]
                events = engine.process_audio_chunk(session_id, chunk, audio_format="pcm_s16le")
                all_events.extend(events)

            flushed = engine.flush_session(session_id)
            all_events.extend(flushed)

            # Verification of real response from AWS
            assert len(all_events) > 0, "Expected at least one transcript event from live AWS Transcribe"
            has_transcript = any(bool(e.text.strip()) for e in all_events)
            assert has_transcript, "Expected non-empty transcript text from live AWS Transcribe"

        except AwsAsrAuthError as e:
            pytest.skip(f"Live AWS test skipped due to AWS credential/subscription status: {e}")
        except AwsAsrServiceUnavailableError as e:
            pytest.skip(f"Live AWS test skipped due to AWS service unreachability: {e}")
        finally:
            engine.close_session(session_id)
            engine.close()
