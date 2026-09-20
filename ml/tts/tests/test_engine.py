"""Unit tests for EdgeTtsEngine."""

import io
import wave

import pytest

from converse_tts.engine import EdgeTtsEngine


def test_engine_init() -> None:
    """Test engine initialization defaults."""
    engine = EdgeTtsEngine()
    assert engine.default_voice == "en-US-ChristopherNeural"
    assert engine.sample_rate == 16000


def test_normalize_rate_and_pitch() -> None:
    """Test normalization of speech rate and pitch."""
    assert EdgeTtsEngine._normalize_rate("+0%") == "+0%"
    assert EdgeTtsEngine._normalize_rate("20%") == "+20%"
    assert EdgeTtsEngine._normalize_rate("-15%") == "-15%"
    assert EdgeTtsEngine._normalize_rate(1.2) == "+20%"
    assert EdgeTtsEngine._normalize_rate(0.8) == "-20%"
    assert EdgeTtsEngine._normalize_rate("") == "+0%"

    assert EdgeTtsEngine._normalize_pitch("+0Hz") == "+0Hz"
    assert EdgeTtsEngine._normalize_pitch("5Hz") == "+5Hz"
    assert EdgeTtsEngine._normalize_pitch("-5hz") == "-5Hz"
    assert EdgeTtsEngine._normalize_pitch("") == "+0Hz"


def test_pcm_to_wav() -> None:
    """Test PCM wrapping into WAV container."""
    dummy_pcm = b"\x00\x00" * 8000  # 0.5s at 16kHz
    wav_bytes = EdgeTtsEngine.pcm_to_wav(dummy_pcm, sample_rate=16000, num_channels=1)

    assert len(wav_bytes) == 44 + len(dummy_pcm)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 8000


def test_calculate_duration_ms() -> None:
    """Test calculation of duration in ms."""
    pcm_bytes = b"\x00\x00" * 16000  # 1 second of 16kHz 16-bit mono = 32000 bytes
    duration = EdgeTtsEngine.calculate_duration_ms(len(pcm_bytes), sample_rate=16000)
    assert abs(duration - 1000.0) < 0.01


@pytest.mark.asyncio
async def test_empty_text_synthesis() -> None:
    """Test synthesis with empty text returns empty or valid header."""
    engine = EdgeTtsEngine()
    wav_result = await engine.synthesize("", audio_format="wav")
    assert len(wav_result) == 44  # WAV header with 0 data

    pcm_result = await engine.synthesize("   ", audio_format="pcm")
    assert pcm_result == b""


@pytest.mark.asyncio
async def test_live_synthesis_wav() -> None:
    """Test live Edge-TTS synthesis returning WAV audio."""
    engine = EdgeTtsEngine()
    wav_data = await engine.synthesize(
        text="Hello world",
        voice="en-US-ChristopherNeural",
        rate="+0%",
        pitch="+0Hz",
        audio_format="wav",
    )
    assert len(wav_data) > 1000
    assert wav_data[:4] == b"RIFF"
    assert wav_data[8:12] == b"WAVE"

    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        frames = wf.getnframes()
        assert frames > 0


@pytest.mark.asyncio
async def test_live_synthesis_pcm_and_caching() -> None:
    """Test PCM generation and caching mechanism for low latency."""
    engine = EdgeTtsEngine()

    pcm_data1 = await engine.synthesize(
        text="Quick test",
        voice="en-US-JennyNeural",
        audio_format="pcm",
    )
    assert len(pcm_data1) > 0
    # PCM S16LE byte length must be even
    assert len(pcm_data1) % 2 == 0

    # Cache hit should be fast and identical
    pcm_data2 = await engine.synthesize(
        text="Quick test",
        voice="en-US-JennyNeural",
        audio_format="pcm",
    )
    assert pcm_data1 == pcm_data2


def test_sync_synthesis() -> None:
    """Test synchronous synthesis wrapper."""
    engine = EdgeTtsEngine()
    wav_data = engine.synthesize_sync(
        text="Testing sync call",
        voice="en-US-ChristopherNeural",
        audio_format="wav",
    )
    assert len(wav_data) > 44
    assert wav_data[:4] == b"RIFF"
