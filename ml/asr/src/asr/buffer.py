"""Audio buffering, decoding, and resampling utilities for streaming ASR."""

import base64
import io
import wave
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """Decoded audio chunk metadata and samples."""

    samples: np.ndarray  # float32 normalized to [-1.0, 1.0]
    sample_rate: int
    duration_ms: float
    is_final: bool = False


def decode_audio_payload(
    audio_base64: str,
    audio_format: str = "pcm_s16le",
    sample_rate: int = 16000,
    channels: int = 1,
) -> np.ndarray:
    """Decode base64 encoded audio payload into normalized float32 numpy array.

    Supported formats:
        - "pcm_s16le": Raw 16-bit linear PCM little-endian
        - "pcm": Alias for pcm_s16le
        - "wav": RIFF WAV container containing 16-bit PCM
    """
    raw_bytes = base64.b64decode(audio_base64)
    if not raw_bytes:
        return np.empty(0, dtype=np.float32)

    normalized_format = audio_format.lower().strip()

    if normalized_format in ("pcm_s16le", "pcm"):
        int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
        if channels > 1:
            int16_data = int16_data.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return (int16_data.astype(np.float32) / 32768.0).astype(np.float32)

    if normalized_format == "wav":
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
                # 8-bit unsigned PCM
                uint8_data = np.frombuffer(frames, dtype=np.uint8)
                if n_channels > 1:
                    uint8_data = uint8_data.reshape(-1, n_channels).mean(axis=1).astype(np.uint8)
                samples = ((uint8_data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
            else:
                raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

            if wav_rate != sample_rate:
                samples = resample_to_16k(samples, wav_rate, target_sr=sample_rate)
            return samples

    raise ValueError(f"Unsupported audio format: {audio_format}")


def resample_to_16k(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample float32 audio array to target sample rate using linear interpolation."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio

    orig_length = len(audio)
    target_length = round(orig_length * (target_sr / orig_sr))
    if target_length == 0:
        return np.empty(0, dtype=np.float32)

    x_orig = np.linspace(0.0, 1.0, orig_length, endpoint=False)
    x_target = np.linspace(0.0, 1.0, target_length, endpoint=False)
    resampled = np.interp(x_target, x_orig, audio)
    return resampled.astype(np.float32)


class SlidingAudioBuffer:
    """Sliding audio buffer for streaming ASR ingestion.

    Maintains sequential audio frames, provides overlapping window slices,
    and supports rolling forward and flushing on utterance boundaries.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_buffer_duration_sec: float = 30.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(max_buffer_duration_sec * sample_rate)
        self._buffer = np.empty(0, dtype=np.float32)
        self._total_samples_received: int = 0

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    @property
    def duration_ms(self) -> float:
        return (len(self._buffer) / self.sample_rate) * 1000.0

    @property
    def total_duration_ms(self) -> float:
        return (self._total_samples_received / self.sample_rate) * 1000.0

    def append(self, samples: np.ndarray) -> None:
        """Append incoming float32 samples to buffer."""
        if len(samples) == 0:
            return

        self._total_samples_received += len(samples)
        self._buffer = np.concatenate([self._buffer, samples])

        if len(self._buffer) > self.max_samples:
            overflow = len(self._buffer) - self.max_samples
            self._buffer = self._buffer[overflow:]

    def append_raw_pcm(self, pcm_bytes: bytes) -> None:
        """Convenience method to append raw little-endian 16-bit PCM bytes."""
        if not pcm_bytes:
            return
        int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        float32_arr = (int16_arr.astype(np.float32) / 32768.0).astype(np.float32)
        self.append(float32_arr)

    def get_window(self, window_duration_ms: float, offset_ms: float = 0.0) -> np.ndarray:
        """Extract audio window from the end of the buffer minus offset."""
        window_samples = round((window_duration_ms / 1000.0) * self.sample_rate)
        offset_samples = round((offset_ms / 1000.0) * self.sample_rate)

        if window_samples <= 0:
            return np.empty(0, dtype=np.float32)

        end_idx = len(self._buffer) - offset_samples
        if end_idx <= 0:
            return np.empty(0, dtype=np.float32)

        start_idx = max(0, end_idx - window_samples)
        return self._buffer[start_idx:end_idx].copy()

    def get_all_samples(self) -> np.ndarray:
        """Retrieve all currently buffered samples."""
        return self._buffer.copy()

    def consume(self, duration_ms: float) -> np.ndarray:
        """Consume and remove specified duration from the front of the buffer."""
        samples_to_consume = round((duration_ms / 1000.0) * self.sample_rate)
        if samples_to_consume <= 0:
            return np.empty(0, dtype=np.float32)

        consumed = self._buffer[:samples_to_consume].copy()
        self._buffer = self._buffer[samples_to_consume:]
        return consumed

    def clear(self) -> np.ndarray:
        """Flush and return all samples in the buffer."""
        samples = self._buffer.copy()
        self._buffer = np.empty(0, dtype=np.float32)
        return samples

    def reset(self) -> None:
        """Full reset of buffer state including total received counter."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._total_samples_received = 0
