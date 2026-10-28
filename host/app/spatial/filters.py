"""Temporal smoothing (One Euro filter) with bounded hold and explicit track-loss handling."""
from __future__ import annotations

import math

import numpy as np


class OneEuro3D:
    """One Euro filter on a 3-vector (Casiez et al., 2012). Timestamps are in SECONDS and must be increasing.

    Low speed -> strong smoothing (cutoff ~ min_cutoff); high speed -> cutoff grows with beta*|speed| so lag shrinks.
    """

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff_hz, beta, d_cutoff_hz
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx = np.zeros(3)
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @property
    def last(self) -> np.ndarray | None:
        return None if self._x is None else self._x.copy()

    def filter(self, x: np.ndarray, t_s: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._t = x.copy(), t_s
            return x.copy()
        dt = t_s - self._t
        if dt <= 0:                                   # duplicate/out-of-order timestamp: never move backwards in time
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t_s
        return self._x.copy()


class TrackSmoother:
    """Per-target track state: smoothing + bounded hold of the last valid estimate.

    * ``update`` is only called with geometrically VALIDATED estimates (smoothing never sees invalid data).
    * The filter is reset on: track key change (landmark/hand identity), calibration change, session reset, track loss.
    * ``held`` exposes the last valid estimate only for ``hold_ms`` after its measurement time; after that ``expire``
      invalidates it (the engine then reports TARGET_LOST). Nothing is ever extrapolated.
    """

    def __init__(self, enabled: bool, min_cutoff_hz: float, beta: float, d_cutoff_hz: float, hold_ms: float):
        self.enabled, self.hold_ms = enabled, hold_ms
        self.filter = OneEuro3D(min_cutoff_hz, beta, d_cutoff_hz)
        self.key = None
        self.last_xyz: np.ndarray | None = None
        self.last_t_ms: float | None = None
        self.resets: list[str] = []                   # reasons, newest last (bounded) - useful in tests/diagnostics

    def reset(self, reason: str) -> None:
        self.filter.reset()
        self.last_xyz = self.last_t_ms = None
        self.key = None
        self.resets = (self.resets + [reason])[-20:]

    def update(self, xyz: np.ndarray, t_ms: float, key=None) -> np.ndarray:
        if self.key is not None and key != self.key:
            self.reset(f"track_key_changed:{self.key}->{key}")
        self.key = key
        if self.last_t_ms is not None and t_ms <= self.last_t_ms:   # older/duplicate measurement: keep current state
            return self.last_xyz.copy()
        out = self.filter.filter(xyz, t_ms / 1000.0) if self.enabled else np.asarray(xyz, float).copy()
        self.last_xyz, self.last_t_ms = out, t_ms
        return out.copy()

    def held(self, now_ms: float) -> tuple[np.ndarray, float] | None:
        if self.last_xyz is None or self.last_t_ms is None:
            return None
        age = now_ms - self.last_t_ms
        return (self.last_xyz.copy(), age) if 0 <= age <= self.hold_ms else None

    def expire(self, now_ms: float) -> bool:
        """True exactly once when a previously valid estimate has just timed out."""
        if self.last_t_ms is not None and now_ms - self.last_t_ms > self.hold_ms:
            self.reset("hold_expired")
            return True
        return False


def measure_lag_ms(min_cutoff_hz: float, beta: float, d_cutoff_hz: float, speed_mps: float, rate_hz: float = 15.0,
                   duration_s: float = 6.0) -> float:
    """Steady-state lag of the filter on a constant-velocity ramp, in ms (lag distance / speed).
    Used to REPORT the cost of smoothing for a given parameter set (do not tune for smoothness alone)."""
    f = OneEuro3D(min_cutoff_hz, beta, d_cutoff_hz)
    n = int(duration_s * rate_hz)
    lags = []
    for i in range(n):
        t = i / rate_hz
        truth = np.array([speed_mps * t, 0.0, 1.0])
        out = f.filter(truth, t)
        if i > n * 0.6:
            lags.append((truth[0] - out[0]) / speed_mps * 1000.0)
    return float(np.mean(lags))
