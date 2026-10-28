"""Gesture Agent configuration (``config/gesture.yaml``). Plain dataclasses, validated; thresholds are STARTING POINTS to tune on hardware."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class GestureConfigError(ValueError):
    pass


def _chk(name, v, lo=None, hi=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise GestureConfigError(f"{name} must be a finite number (got {v!r})")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise GestureConfigError(f"{name} must be {'>' if lo_open else '>='} {lo} (got {v})")
    if hi is not None and v > hi:
        raise GestureConfigError(f"{name} must be <= {hi} (got {v})")


@dataclass
class GestureCfg:
    # --- pinch (scale-aware: thumb-tip to index-tip distance divided by palm length |wrist - index_mcp| in the SAME view)
    pinch_enter_ratio: float = 0.25          # LOWER threshold: enter the pinched state
    pinch_exit_ratio: float = 0.45           # HIGHER threshold: leave it (hysteresis band between the two)
    confirm_ms: float = 120.0                # evidence must hold this long before a grab is accepted
    release_confirm_ms: float = 50.0         # opening must hold this long before release
    smoothing_window: int = 3                # median over the last N valid ratios
    min_pinch_samples: int = 3               # at least this many valid samples inside the confirmation window
    max_evidence_gap_ms: float = 150.0       # a gap between valid ratios longer than this restarts a pending confirmation
    min_palm_scale_px: float = 20.0          # smaller palm in the image = unreliable ratio, rejected
    min_landmark_confidence: float = 0.3     # applied only when the detector provides a confidence
    stale_landmark_ms: float = 250.0
    # --- reach / confidence (hysteresis too)
    reach_enter_m: float = 0.12
    reach_exit_m: float = 0.18
    min_confidence: float = 0.5              # fusion quality needed to ENTER hover
    exit_confidence: float = 0.35            # below this a sample is unusable (hysteresis against min_confidence)
    confidence_history: int = 3
    max_pose_age_ms: float = 250.0           # fused pose older than this is never used
    # --- tracking loss (bounded, documented default: freeze <= grab_loss_timeout_ms, then SAFE release)
    tracking_lost_report_ms: float = 100.0
    hover_loss_timeout_ms: float = 250.0
    grab_loss_timeout_ms: float = 400.0
    max_resume_jump_m: float = 0.25          # recovery further than this from the last held pose = track discontinuity -> cancel
    # --- interface contract with the (future) world adapter
    require_adapter_ack: bool = False        # True: GRABBED waits for acknowledge(GRAB_REQUEST) before GRAB_CONFIRMED
    ack_timeout_ms: float = 500.0
    release_ack_required: bool = False       # True: RELEASED waits for acknowledge(RELEASE) before IDLE
    release_ack_timeout_ms: float = 500.0
    release_settle_ms: float = 100.0         # RELEASED lasts at least this long when no ack is required
    move_max_hz: float = 30.0

    def validate(self) -> "GestureCfg":
        _chk("pinch_enter_ratio", self.pinch_enter_ratio, 0, lo_open=True)
        _chk("pinch_exit_ratio", self.pinch_exit_ratio, 0, lo_open=True)
        if not self.pinch_enter_ratio < self.pinch_exit_ratio:
            raise GestureConfigError("hysteresis: pinch_enter_ratio (lower) must be < pinch_exit_ratio (higher)")
        if not self.reach_enter_m < self.reach_exit_m:
            raise GestureConfigError("hysteresis: reach_enter_m must be < reach_exit_m")
        if not self.exit_confidence < self.min_confidence:
            raise GestureConfigError("hysteresis: exit_confidence must be < min_confidence")
        for n in ("confirm_ms", "release_confirm_ms", "max_evidence_gap_ms", "min_palm_scale_px", "stale_landmark_ms", "max_pose_age_ms",
                  "tracking_lost_report_ms", "hover_loss_timeout_ms", "grab_loss_timeout_ms", "ack_timeout_ms",
                  "release_ack_timeout_ms", "release_settle_ms", "move_max_hz", "reach_enter_m", "max_resume_jump_m"):
            _chk(n, getattr(self, n), 0, lo_open=n in ("move_max_hz", "reach_enter_m", "min_palm_scale_px"))
        _chk("grab_loss_timeout_ms", self.grab_loss_timeout_ms, 0, 5000)          # bounded: never hold stale coordinates indefinitely
        _chk("hover_loss_timeout_ms", self.hover_loss_timeout_ms, 0, 5000)
        for n in ("min_landmark_confidence", "min_confidence", "exit_confidence"):
            _chk(n, getattr(self, n), 0, 1)
        if self.smoothing_window < 1 or self.min_pinch_samples < 1 or self.confidence_history < 1:
            raise GestureConfigError("smoothing_window, min_pinch_samples and confidence_history must be >= 1")
        if self.tracking_lost_report_ms > self.grab_loss_timeout_ms:
            raise GestureConfigError("tracking_lost_report_ms must be <= grab_loss_timeout_ms")
        return self


def gesture_config_from_mapping(data: Mapping[str, Any] | None) -> GestureCfg:
    data = dict(data or {})
    known = {f.name for f in dataclasses.fields(GestureCfg)}
    if set(data) - known:
        raise GestureConfigError(f"unknown gesture config keys {sorted(set(data) - known)}")
    return GestureCfg(**data).validate()


def load_gesture_config(path: Path | str | None = None) -> GestureCfg:
    import yaml
    p = Path(path) if path else Path(__file__).resolve().parents[3] / "config" / "gesture.yaml"
    if not p.exists():
        return GestureCfg().validate()
    with p.open("r", encoding="utf-8") as fh:
        return gesture_config_from_mapping(yaml.safe_load(fh) or {})
