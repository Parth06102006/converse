"""Native Edge-TTS synthesis engine for Converse speech output."""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import wave
from collections import OrderedDict
from collections.abc import AsyncGenerator

import edge_tts

try:
    import miniaudio
    HAS_MINIAUDIO = True
except ImportError:
    HAS_MINIAUDIO = False

logger = logging.getLogger("converse_tts.engine")


class EdgeTtsEngineError(Exception):
    """Exception raised for Edge-TTS synthesis or conversion errors."""


class EdgeTtsEngine:
    """Native Edge-TTS Neural Voice Synthesis Engine.

    Generates 16kHz PCM S16LE or WAV audio asynchronously using Microsoft
    Neural voices such as 'en-US-ChristopherNeural' and 'en-US-JennyNeural'.
    """

    DEFAULT_VOICE = "en-US-ChristopherNeural"
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_SAMPLE_WIDTH = 2  # 16-bit signed integer (S16LE)
    CACHE_MAX_SIZE = 512

    def __init__(
        self,
        default_voice: str = DEFAULT_VOICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        cache_size: int = CACHE_MAX_SIZE,
    ) -> None:
        """Initialize the Edge-TTS synthesis engine."""
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str, str, str, str], bytes] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_rate(rate: str | float) -> str:
        """Normalize speech rate parameter to Edge-TTS format, e.g. '+0%' or '+20%'."""
        if isinstance(rate, (int, float)):
            pct = round((float(rate) - 1.0) * 100)
            return f"{pct:+d}%"
        rate_str = str(rate).strip()
        if not rate_str:
            return "+0%"
        if not rate_str.endswith("%"):
            rate_str = f"{rate_str}%"
        if not rate_str.startswith(("+", "-")):
            rate_str = f"+{rate_str}"
        return rate_str

    @staticmethod
    def _normalize_pitch(pitch: str | int) -> str:
        """Normalize speech pitch parameter to Edge-TTS format, e.g. '+0Hz' or '-5Hz'."""
        pitch_str = str(pitch).strip()
        if not pitch_str:
            return "+0Hz"
        if pitch_str.lower().endswith("hz"):
            val = pitch_str[:-2].strip()
            pitch_str = f"{val}Hz"
        else:
            pitch_str = f"{pitch_str}Hz"
        if not pitch_str.startswith(("+", "-")):
            pitch_str = f"+{pitch_str}"
        return pitch_str

    @staticmethod
    def pcm_to_wav(
        pcm_bytes: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
    ) -> bytes:
        """Wrap raw PCM bytes into a standard RIFF/WAVE header container."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

    @classmethod
    def calculate_duration_ms(
        cls,
        pcm_bytes_len: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        num_channels: int = DEFAULT_CHANNELS,
    ) -> float:
        """Calculate duration in milliseconds from PCM byte length."""
        bytes_per_sec = sample_rate * sample_width * num_channels
        if bytes_per_sec <= 0:
            return 0.0
        return (pcm_bytes_len / bytes_per_sec) * 1000.0

    def _decode_mp3_to_pcm16k(self, mp3_bytes: bytes) -> bytes:
        """Decode MP3 audio bytes to 16kHz mono signed 16-bit PCM (S16LE)."""
        if not mp3_bytes:
            return b""

        # Primary decoder: miniaudio (high performance in-memory decoding)
        if HAS_MINIAUDIO:
            try:
                decoded = miniaudio.decode(
                    mp3_bytes,
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=self.DEFAULT_CHANNELS,
                    sample_rate=self.sample_rate,
                )
                return decoded.samples.tobytes()
            except Exception as e:  # noqa: BLE001
                logger.warning("miniaudio decoding failed: %s; falling back to ffmpeg", e)

        # Secondary decoder fallback: ffmpeg subprocess
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-threads",
                    "1",
                    "-f",
                    "mp3",
                    "-i",
                    "pipe:0",
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    str(self.DEFAULT_CHANNELS),
                    "-ar",
                    str(self.sample_rate),
                    "pipe:1",
                ],
                input=mp3_bytes,
                capture_output=True,
                check=True,
            )
            return proc.stdout
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode("utf-8", errors="replace")
            raise EdgeTtsEngineError(f"FFmpeg decoding failed: {stderr_msg}") from e
        except FileNotFoundError as e:
            raise EdgeTtsEngineError(
                "Neither miniaudio nor ffmpeg is available for audio decoding"
            ) from e

    async def _fetch_edge_tts_mp3(
        self,
        text: str,
        voice: str,
        rate: str,
        pitch: str,
    ) -> bytes:
        """Fetch raw audio stream from Edge-TTS service."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )

        audio_chunks: list[bytes] = []
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
        except Exception as e:
            raise EdgeTtsEngineError(f"Edge-TTS synthesis network error: {e}") from e

        if not audio_chunks:
            raise EdgeTtsEngineError("No audio received from Edge-TTS service")

        return b"".join(audio_chunks)

    async def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        audio_format: str = "wav",
    ) -> bytes:
        """Synthesize speech audio asynchronously.

        Args:
            text: Text to synthesize.
            voice: Microsoft neural voice identifier, e.g. 'en-US-ChristopherNeural'
                or 'en-US-JennyNeural'.
            rate: Speech rate adjustment string, e.g. '+0%', '+20%', '-10%'.
            pitch: Speech pitch adjustment string, e.g. '+0Hz', '+5Hz'.
            audio_format: Desired output format: 'wav', 'pcm' / 'pcm_s16le', or 'mp3'.

        Returns:
            Audio bytes in the requested format (WAV or 16kHz S16LE PCM by default).
        """
        clean_text = text.strip()
        fmt = audio_format.lower()
        norm_rate = self._normalize_rate(rate)
        norm_pitch = self._normalize_pitch(pitch)
        norm_voice = voice or self.default_voice

        if not clean_text:
            if fmt == "wav":
                return self.pcm_to_wav(b"", sample_rate=self.sample_rate)
            return b""

        cache_key = (clean_text, norm_voice, norm_rate, norm_pitch, fmt)
        async with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

        mp3_data = await self._fetch_edge_tts_mp3(
            text=clean_text,
            voice=norm_voice,
            rate=norm_rate,
            pitch=norm_pitch,
        )

        if fmt == "mp3":
            result_bytes = mp3_data
        else:
            pcm_data = self._decode_mp3_to_pcm16k(mp3_data)
            if fmt in ("pcm", "pcm_s16le", "raw"):
                result_bytes = pcm_data
            else:
                # Default: WAV container
                result_bytes = self.pcm_to_wav(pcm_data, sample_rate=self.sample_rate)

        async with self._lock:
            self._cache[cache_key] = result_bytes
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return result_bytes

    async def synthesize_stream(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> AsyncGenerator[bytes, None]:
        """Stream raw MP3 audio chunks asynchronously from Edge-TTS."""
        clean_text = text.strip()
        if not clean_text:
            return

        norm_rate = self._normalize_rate(rate)
        norm_pitch = self._normalize_pitch(pitch)
        norm_voice = voice or self.default_voice

        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=norm_voice,
            rate=norm_rate,
            pitch=norm_pitch,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    def synthesize_sync(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        audio_format: str = "wav",
    ) -> bytes:
        """Synchronous wrapper for synthesize."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.synthesize(text, voice, rate, pitch, audio_format),
                )
                return future.result()
        else:
            return asyncio.run(
                self.synthesize(text, voice, rate, pitch, audio_format)
            )

    def clear_cache(self) -> None:
        """Clear cached audio utterances."""
        self._cache.clear()
