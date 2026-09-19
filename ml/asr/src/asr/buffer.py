"""Audio buffering, decoding, and polyphase resampling utilities for streaming ASR."""

import base64
import binascii
import io
import math
import wave
from dataclasses import dataclass

import numpy as np
import scipy.signal

# Optional PyAV container decoder for WebM/Opus and container formats
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


@dataclass(frozen=True)
class AudioChunk:
    """Decoded audio chunk metadata and samples."""

    samples: np.ndarray  # float32 normalized to [-1.0, 1.0]
    sample_rate: int
    duration_ms: float
    is_final: bool = False


def resample_to_16k(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample float32 audio array to target sample rate using polyphase filtering.

    Uses scipy.signal.resample_poly with a Kaiser window FIR filter to avoid
    high-frequency aliasing and preserve speech formant characteristics.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive: orig_sr={orig_sr}, target_sr={target_sr}")

    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = scipy.signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def decode_audio_payload(
    audio_base64: str,
    audio_format: str = "pcm_s16le",
    sample_rate: int = 16000,
    channels: int = 1,
    max_payload_bytes: int = 50 * 1024 * 1024,
) -> np.ndarray:
    """Decode base64-encoded audio payload into normalized float32 numpy array.

    Supported formats:
        - "pcm_s16le", "pcm": Raw 16-bit linear PCM little-endian
        - "wav": RIFF WAV container containing 16-bit or 8-bit PCM
        - "webm", "opus": WebM / Ogg Opus container decoded via PyAV

    Raises:
        ValueError: For malformed base64, unaligned PCM byte streams, invalid
                    sample widths, or unsupported container formats.
    """
    if not audio_base64 or not audio_base64.strip():
        return np.empty(0, dtype=np.float32)

    try:
        raw_bytes = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Malformed Base64 audio payload: {exc}") from exc

    if len(raw_bytes) > max_payload_bytes:
        raise ValueError(f"Audio payload exceeds maximum allowable size ({len(raw_bytes)} > {max_payload_bytes} bytes)")

    if len(raw_bytes) == 0:
        return np.empty(0, dtype=np.float32)

    normalized_format = audio_format.lower().strip()

    if normalized_format in ("pcm_s16le", "pcm"):
        if len(raw_bytes) % 2 != 0:
            raise ValueError(
                f"PCM-16LE payload length must be a multiple of 2 bytes, got {len(raw_bytes)} bytes."
            )
        int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
        if channels > 1:
            if len(int16_data) % channels != 0:
                raise ValueError(
                    f"PCM payload samples ({len(int16_data)}) not divisible by channel count ({channels})"
                )
            int16_data = int16_data.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return (int16_data.astype(np.float32) / 32768.0).astype(np.float32)

    if normalized_format == "wav":
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                wav_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                frames = wav_file.readframes(n_frames)

                if sampwidth == 2:
                    int16_data = np.frombuffer(frames, dtype=np.int16)
                    if n_channels > 1:
                        int16_data = int16_data.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
                    samples = (int16_data.astype(np.float32) / 32768.0).astype(np.float32)
                elif sampwidth == 1:
                    uint8_data = np.frombuffer(frames, dtype=np.uint8)
                    if n_channels > 1:
                        uint8_data = uint8_data.reshape(-1, n_channels).mean(axis=1).astype(np.uint8)
                    samples = ((uint8_data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
                elif sampwidth == 4:
                    int32_data = np.frombuffer(frames, dtype=np.int32)
                    if n_channels > 1:
                        int32_data = int32_data.reshape(-1, n_channels).mean(axis=1).astype(np.int32)
                    samples = (int32_data.astype(np.float32) / 2147483648.0).astype(np.float32)
                else:
                    raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes (supported: 1, 2, 4)")

                if wav_rate != sample_rate:
                    samples = resample_to_16k(samples, wav_rate, target_sr=sample_rate)
                return samples
        except wave.Error as exc:
            raise ValueError(f"Invalid or corrupted WAV container: {exc}") from exc

    if normalized_format in ("webm", "opus", "ogg", "mp3"):
        if not HAS_AV:
            raise ValueError(
                f"Audio format '{audio_format}' requires PyAV which is not installed in the environment."
            )
        try:
            container = av.open(io.BytesIO(raw_bytes))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                raise ValueError(f"No audio stream found in {audio_format} container.")

            pcm_chunks: list[np.ndarray] = []
            stream_rate = audio_stream.rate or sample_rate

            for frame in container.decode(audio_stream):
                arr = frame.to_ndarray()  # shape (channels, samples) or (samples,)
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)  # Downmix to mono
                # Normalize based on dtype
                if arr.dtype == np.int16:
                    pcm_chunks.append((arr.astype(np.float32) / 32768.0).astype(np.float32))
                elif arr.dtype == np.float32:
                    pcm_chunks.append(arr.astype(np.float32))
                elif arr.dtype == np.int32:
                    pcm_chunks.append((arr.astype(np.float32) / 2147483648.0).astype(np.float32))
                else:
                    pcm_chunks.append(arr.astype(np.float32))

            if not pcm_chunks:
                return np.empty(0, dtype=np.float32)

            samples = np.concatenate(pcm_chunks)
            if stream_rate != sample_rate:
                samples = resample_to_16k(samples, stream_rate, target_sr=sample_rate)
            return samples
        except Exception as exc:
            raise ValueError(f"Failed to decode {audio_format} container: {exc}") from exc

    raise ValueError(f"Unsupported audio format: '{audio_format}' (supported: 'pcm_s16le', 'pcm', 'wav', 'webm', 'opus')")


class SlidingAudioBuffer:
    """Sliding audio buffer for streaming ASR and VAD ingestion.

    Maintains sequential float32 audio frames, provides window slices,
    and supports rolling forward and flushing on utterance boundaries.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_buffer_duration_sec: float = 30.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_buffer_samples = int(sample_rate * max_buffer_duration_sec)
        self._buffer = np.empty(0, dtype=np.float32)
        self._total_samples_ingested = 0

    @property
    def sample_count(self) -> int:
        """Current number of buffered samples."""
        return len(self._buffer)

    @property
    def duration_ms(self) -> float:
        """Current duration of buffered audio in milliseconds."""
        return (len(self._buffer) / self.sample_rate) * 1000.0

    @property
    def total_duration_ms(self) -> float:
        """Total duration of all audio ingested across the session in milliseconds."""
        return (self._total_samples_ingested / self.sample_rate) * 1000.0

    @property
    def total_samples_ingested(self) -> int:
        return self._total_samples_ingested

    def append(self, samples: np.ndarray) -> None:
        """Append float32 audio samples to buffer."""
        if len(samples) == 0:
            return

        flat = samples.flatten().astype(np.float32)
        self._buffer = np.concatenate((self._buffer, flat))
        self._total_samples_ingested += len(flat)

        # Enforce maximum ring-buffer limit
        if len(self._buffer) > self.max_buffer_samples:
            overflow = len(self._buffer) - self.max_buffer_samples
            self._buffer = self._buffer[overflow:]

    def get_window(self, window_duration_ms: float) -> np.ndarray:
        """Retrieve most recent audio window of given duration in milliseconds."""
        num_samples = int((window_duration_ms / 1000.0) * self.sample_rate)
        if num_samples >= len(self._buffer):
            return self._buffer.copy()
        return self._buffer[-num_samples:].copy()

    def get_all(self) -> np.ndarray:
        """Retrieve all currently buffered audio samples."""
        return self._buffer.copy()

    def consume(self, duration_ms: float) -> np.ndarray:
        """Consume and remove specified duration in milliseconds from front of buffer."""
        num_samples = int((duration_ms / 1000.0) * self.sample_rate)
        if num_samples <= 0:
            return np.empty(0, dtype=np.float32)

        consumed_count = min(num_samples, len(self._buffer))
        consumed = self._buffer[:consumed_count].copy()
        self._buffer = self._buffer[consumed_count:]
        return consumed

    def flush(self) -> np.ndarray:
        """Flush and return all samples, clearing buffer."""
        all_samples = self._buffer.copy()
        self._buffer = np.empty(0, dtype=np.float32)
        return all_samples

    def clear(self) -> np.ndarray:
        """Clear buffer and return all flushed samples."""
        return self.flush()
