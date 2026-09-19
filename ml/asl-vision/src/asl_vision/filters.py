"""One-Euro temporal filter for ASL 3D skeletal landmark smoothing.

Damps high-frequency webcam jitter during stationary hand postures while
adapting the cutoff dynamically to eliminate lag during high-velocity
signing sweeps.
"""

from __future__ import annotations

import math

import numpy as np


def _smoothing_factor(time_delta: float, cutoff: float) -> float:
    """Calculate the low-pass alpha coefficient for a given cutoff frequency."""
    r = 2.0 * math.pi * cutoff * time_delta
    return r / (r + 1.0)


def _exponential_smoothing(
    alpha: float, current_val: np.ndarray, prev_val: np.ndarray
) -> np.ndarray:
    """Apply exponential moving average filter."""
    return alpha * current_val + (1.0 - alpha) * prev_val


class OneEuroFilter:
    """Adaptive low-pass filter for real-time jitter reduction.

    Damps high-frequency jitter during stationary hand postures while
    adapting cutoff dynamically to eliminate lag during high-velocity sweeps.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialize One-Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency (Hz) for low velocities.
            beta: Velocity adaptation coefficient.
            d_cutoff: Cutoff frequency (Hz) for the derivative filter.
        """
        if min_cutoff <= 0:
            raise ValueError(f"min_cutoff must be positive, got {min_cutoff}.")
        if d_cutoff <= 0:
            raise ValueError(f"d_cutoff must be positive, got {d_cutoff}.")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}.")

        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Clear filter internal states."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def filter(self, x: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        """Filter incoming landmark coordinate vector.

        Args:
            x: Raw input array of coordinates, shape (N, D).
            timestamp: Timestamp in seconds. If None, assumes 30 FPS (1/30s delta).

        Returns:
            Filtered coordinate array matching input shape.
        """
        x_arr = np.asarray(x, dtype=np.float32)

        # Protect against NaN values poisoning the persistent filter state
        if np.isnan(x_arr).any():
            if self._x_prev is not None:
                x_arr = np.where(np.isnan(x_arr), self._x_prev, x_arr)
            else:
                x_arr = np.nan_to_num(x_arr, nan=0.0)

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x_arr.copy()
            self._dx_prev = np.zeros_like(x_arr)
            self._t_prev = timestamp if timestamp is not None else 0.0
            return x_arr

        # Handle duplicate or non-increasing timestamps without division spikes
        if timestamp is not None and timestamp <= self._t_prev:
            return self._x_prev.copy()

        t_curr = timestamp if timestamp is not None else (self._t_prev + (1.0 / 30.0))
        time_delta = max(t_curr - self._t_prev, 1e-5)
        self._t_prev = t_curr

        # Calculate derivative of raw signal
        dx = (x_arr - self._x_prev) / time_delta

        # Filter the derivative
        alpha_d = _smoothing_factor(time_delta, self.d_cutoff)
        dx_hat = _exponential_smoothing(alpha_d, dx, self._dx_prev)
        self._dx_prev = dx_hat

        # Calculate adaptive cutoff frequency based on derivative magnitude
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the signal
        alpha = _smoothing_factor(time_delta, cutoff)
        x_hat = _exponential_smoothing(alpha, x_arr, self._x_prev)
        self._x_prev = x_hat

        return x_hat

