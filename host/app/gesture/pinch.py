"""Pinch measurement and detection: scale-aware ratio, median smoothing, hysteresis, confirmation window.

ratio = |thumb_tip - index_tip| / |wrist - index_mcp|   (pixels, SAME view; palm length cancels distance-to-camera and resolution)
A ratio is INVALID (``None`` + reason), never 0 ("pinched"), when landmarks are missing, non-finite, low-confidence or the palm is too small.
"""
from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass
from typing import Mapping

from .config import GestureCfg


def pinch_ratio(landmarks: Mapping[str, tuple], cfg: GestureCfg) -> tuple[float | None, str | None]:
    """``landmarks``: id -> (x_px, y_px, confidence|None) from ONE observation. Returns (ratio, invalid_reason)."""
    need = ("thumb_tip", "index_tip", "wrist", "index_mcp")
    missing = [n for n in need if n not in landmarks]
    if missing:
        return None, "LANDMARKS_MISSING"
    pts = {}
    for n in need:
        x, y, c = landmarks[n][0], landmarks[n][1], (landmarks[n][2] if len(landmarks[n]) > 2 else None)
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y)):
            return None, "NON_FINITE_LANDMARK"
        if c is not None and c < cfg.min_landmark_confidence:
            return None, "LOW_LANDMARK_CONFIDENCE"
        pts[n] = (x, y)
    scale = math.dist(pts["wrist"], pts["index_mcp"])
    if scale < cfg.min_palm_scale_px:
        return None, "PALM_SCALE_TOO_SMALL"
    return math.dist(pts["thumb_tip"], pts["index_tip"]) / scale, None


def combine_view_ratios(ratios: list[float]) -> float | None:
    """Median over the views that produced a valid ratio (robust to one bad view); ``None`` if none."""
    return statistics.median(ratios) if ratios else None


@dataclass(frozen=True)
class PinchStatus:
    state: str                    # OPEN | CANDIDATE | PINCHED
    smoothed: float | None
    just_confirmed: bool = False
    just_released: bool = False
    candidate_ms: float = 0.0


class PinchDetector:
    """OPEN -> CANDIDATE (smoothed <= enter) -> PINCHED (held for confirm_ms with >= min samples) ; PINCHED -> OPEN (smoothed >= exit for release_confirm_ms).
    Between enter and exit thresholds nothing changes (hysteresis). A smoothed value above ``enter`` cancels a candidate."""

    def __init__(self, cfg: GestureCfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._win: collections.deque[float] = collections.deque(maxlen=self.cfg.smoothing_window)
        self.state = "OPEN"
        self._cand_since: float | None = None
        self._cand_n = 0
        self._open_since: float | None = None
        self._last_valid_ts: float | None = None

    def update(self, ratio: float | None, ts_ms: float) -> PinchStatus:
        c = self.cfg
        if ratio is None or not math.isfinite(ratio) or ratio < 0:
            return PinchStatus(self.state, self.smoothed, candidate_ms=self._cand_ms(ts_ms))     # invalid evidence changes nothing
        if self._last_valid_ts is not None and ts_ms - self._last_valid_ts > c.max_evidence_gap_ms and self.state == "CANDIDATE":
            self.state, self._cand_since, self._cand_n = "OPEN", None, 0                          # continuity broken: start over
            self._win.clear()
        self._last_valid_ts = ts_ms
        self._win.append(ratio)
        sm = statistics.median(self._win)
        confirmed = released = False
        if self.state in ("OPEN", "CANDIDATE"):
            if sm <= c.pinch_enter_ratio:
                if self.state == "OPEN":
                    self.state, self._cand_since, self._cand_n = "CANDIDATE", ts_ms, 0
                self._cand_n += 1
                if ts_ms - self._cand_since >= c.confirm_ms and self._cand_n >= c.min_pinch_samples:
                    self.state, confirmed, self._open_since = "PINCHED", True, None
            elif self.state == "CANDIDATE":
                self.state, self._cand_since, self._cand_n = "OPEN", None, 0
        else:                                                                                      # PINCHED
            if sm >= c.pinch_exit_ratio:
                self._open_since = ts_ms if self._open_since is None else self._open_since
                if ts_ms - self._open_since >= c.release_confirm_ms:
                    self.state, released, self._open_since = "OPEN", True, None
                    self._win.clear()
                    self._win.append(ratio)
            else:
                self._open_since = None
        return PinchStatus(self.state, sm, confirmed, released, self._cand_ms(ts_ms))

    @property
    def smoothed(self) -> float | None:
        return statistics.median(self._win) if self._win else None

    def _cand_ms(self, ts: float) -> float:
        return 0.0 if self._cand_since is None or self.state != "CANDIDATE" else ts - self._cand_since
