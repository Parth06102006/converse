"""Rolling FIFO temporal window buffer for continuous ASL gesture recognition.

Maintains a sliding window of W normalized landmark frames with stride S, packs
coordinates into continuous (1, 3, W, V) spatiotemporal tensors using pack_frame_tensor,
and performs confidence thresholding to discard unconfident or empty signing sequences.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from asl_vision.normalization import NormalizedFrame, pack_frame_tensor


@dataclass(frozen=True)
class SlidingWindowOutput:
    """Output window containing packed spatiotemporal tensor and associated metadata."""

    tensor: torch.Tensor  # Shape (1, 3, W, V)
    confidence: float
    is_valid: bool
    window_index: int
    start_timestamp_ms: float
    end_timestamp_ms: float
    frames: tuple[NormalizedFrame, ...]

    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array of shape (1, 3, W, V)."""
        return self.tensor.detach().cpu().numpy()


class SlidingWindowBuffer:
    """Rolling FIFO buffer accumulating normalized frames for spatiotemporal sequence modeling.

    Maintains a FIFO queue of window_size frames. When the buffer reaches capacity,
    it emits a packed PyTorch tensor of shape (1, 3, window_size, V) every stride frames.
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 5,
        confidence_threshold: float = 0.5,
        include_pose: bool = True,
        include_hands: bool = True,
        include_face: bool = False,
        emit_only_valid: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the SlidingWindowBuffer.

        Args:
            window_size: Number of temporal frames per window (W, default 30 frames = 1s @ 30 FPS).
            stride: Stride step between successive emitted windows (S, default 5 frames = ~167ms).
            confidence_threshold: Minimum hand presence ratio [0.0, 1.0] for a window to be marked valid.
            include_pose: Whether to include 33 pose landmarks in the packed tensor.
            include_hands: Whether to include bilateral 21 hand landmarks (42 keypoints).
            include_face: Whether to include facial contour landmarks in the packed tensor.
            emit_only_valid: If True, add_frame returns None when window confidence is below threshold.
            device: Optional PyTorch device for output tensor (CPU or CUDA).
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be greater than 0, got {window_size}.")
        if stride <= 0:
            raise ValueError(f"stride must be greater than 0, got {stride}.")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in range [0.0, 1.0], got {confidence_threshold}."
            )
        if not include_pose and not include_hands and not include_face:
            raise ValueError(
                "At least one of include_pose, include_hands, or include_face must be True."
            )

        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.include_pose = include_pose
        self.include_hands = include_hands
        self.include_face = include_face
        self.emit_only_valid = emit_only_valid
        self.device = device

        self._buffer: deque[NormalizedFrame] = deque(maxlen=window_size)
        self._frame_count = 0
        self._frames_since_last_emission = 0
        self._window_index = 0
        self._emitted_window_count = 0

    @property
    def frame_count(self) -> int:
        """Total number of frames processed since last reset."""
        return self._frame_count

    @property
    def window_index(self) -> int:
        """Total number of candidate windows processed since last reset."""
        return self._window_index

    @property
    def emitted_window_count(self) -> int:
        """Total number of valid windows actually emitted to caller since last reset."""
        return self._emitted_window_count

    @property
    def is_full(self) -> bool:
        """Whether the buffer contains window_size frames."""
        return len(self._buffer) == self.window_size

    def __len__(self) -> int:
        """Current number of frames stored in the buffer."""
        return len(self._buffer)

    def reset(self) -> None:
        """Reset the buffer, clearing all frames and counters."""
        self._buffer.clear()
        self._frame_count = 0
        self._frames_since_last_emission = 0
        self._window_index = 0
        self._emitted_window_count = 0

    def clear(self) -> None:
        """Alias for reset()."""
        self.reset()

    def _pack_buffer_tensor(self) -> torch.Tensor:
        """Pack current deque frames into continuous (1, 3, W, V) tensor."""
        packed_frames: list[np.ndarray] = []
        for frame in self._buffer:
            packed = pack_frame_tensor(
                frame,
                include_pose=self.include_pose,
                include_hands=self.include_hands,
                include_face=self.include_face,
            )
            packed_frames.append(packed)

        # Shape (W, 3 * V)
        stacked = np.stack(packed_frames, axis=0)
        num_features = stacked.shape[1]
        num_joints = num_features // 3

        if num_features % 3 != 0:
            raise ValueError(
                f"Packed frame features ({num_features}) is not divisible by 3 coordinates."
            )

        # Reshape to (W, V, 3) where last axis is [x, y, z]
        reshaped = stacked.reshape(len(self._buffer), num_joints, 3)

        # Transpose (W, V, 3) -> (3, W, V) to place coordinate channels first
        transposed = np.transpose(reshaped, (2, 0, 1))

        # Add batch dimension: (1, 3, W, V)
        batched = np.expand_dims(transposed, axis=0)

        # Create continuous PyTorch tensor
        tensor = torch.from_numpy(batched.copy()).float()
        if self.device is not None:
            tensor = tensor.to(self.device)

        return tensor

    def _compute_window_output(
        self, window_index: int | None = None
    ) -> SlidingWindowOutput:
        """Construct SlidingWindowOutput from current buffer state."""
        tensor = self._pack_buffer_tensor()

        # Compute confidence: if hands included, check hand presence; otherwise 1.0 (pose valid)
        if self.include_hands:
            hand_visible_count = sum(
                1
                for f in self._buffer
                if f.is_left_hand_visible or f.is_right_hand_visible
            )
            confidence = float(hand_visible_count / len(self._buffer))
        else:
            confidence = 1.0

        is_valid = confidence >= self.confidence_threshold
        idx = self._window_index if window_index is None else window_index

        output = SlidingWindowOutput(
            tensor=tensor,
            confidence=confidence,
            is_valid=is_valid,
            window_index=idx,
            start_timestamp_ms=self._buffer[0].timestamp_ms,
            end_timestamp_ms=self._buffer[-1].timestamp_ms,
            frames=tuple(self._buffer),
        )

        return output

    def add_frame(self, frame: NormalizedFrame) -> SlidingWindowOutput | None:
        """Push a normalized frame into the FIFO buffer.

        If the buffer has reached window_size and the stride interval has elapsed,
        a SlidingWindowOutput is computed and returned.

        Args:
            frame: NormalizedFrame from LandmarkNormalizer.

        Returns:
            SlidingWindowOutput if window is ready and passes validation filter, else None.

        Raises:
            ValueError: If frame is None or not a NormalizedFrame.
        """
        if frame is None or not isinstance(frame, NormalizedFrame):
            raise ValueError(f"Expected NormalizedFrame instance, got {type(frame)}.")

        self._buffer.append(frame)
        self._frame_count += 1
        self._frames_since_last_emission += 1

        # Check if buffer is warmed up
        if len(self._buffer) < self.window_size:
            return None

        # On the very first time the buffer fills up, evaluate window 0
        if self._window_index == 0:
            output = self._compute_window_output(window_index=self._window_index)
            self._window_index += 1
            self._frames_since_last_emission = 0
            if self.emit_only_valid and not output.is_valid:
                return None
            self._emitted_window_count += 1
            return output

        # For subsequent windows, check stride interval
        if self._frames_since_last_emission >= self.stride:
            output = self._compute_window_output(window_index=self._window_index)
            self._window_index += 1
            self._frames_since_last_emission = 0
            if self.emit_only_valid and not output.is_valid:
                return None
            self._emitted_window_count += 1
            return output

        return None

    def get_current_window(self) -> SlidingWindowOutput | None:
        """Inspect the current buffer window without advancing stride or window index.

        Returns:
            SlidingWindowOutput if buffer is full, else None.
        """
        if len(self._buffer) < self.window_size:
            return None

        # If the window currently in buffer was just emitted at this exact step,
        # its index is _window_index - 1
        if self._window_index > 0 and self._frames_since_last_emission == 0:
            peek_idx = self._window_index - 1
        else:
            peek_idx = self._window_index

        return self._compute_window_output(window_index=peek_idx)
