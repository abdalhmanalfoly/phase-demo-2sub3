"""Per-sensor offer scoring. Pure functions; every input and output is normalised to 0..1.

quality = 0.30*detector_confidence + 0.20*visibility + 0.20*freshness + 0.15*sync + 0.15*health   (weights: config)

HOW EACH COMPONENT IS CALCULATED (documented in docs/coordination.md section 3)
  detector_confidence  landmark confidence when the detector provides one. The MediaPipe hand model does NOT, so real
                       sensors send ``null``: the value is then ``imputed_detector_confidence`` (default 0.5, neutral) and
                       the offer is flagged ``detector_confidence_source="imputed"``. Never silently excellent.
  visibility           0 if no tracked hand / landmark absent / outside the frame. Else 0.5 + 0.5*clamp(edge_margin/edge_margin_frac):
                       fully inside with margin = 1.0, touching the border = 0.5. ``out_of_frame`` flag caps it at 0.3.
  freshness            1 - age_ms / freshness_horizon_ms, clamped. Unknown age -> 0.0.
  sync                 timestamp trustworthiness: 1 - delay_jitter_ms / sync_jitter_horizon_ms (jitter of the sensor's
                       received-minus-captured delay, a proxy for clock-offset error). No clock offset or unknown jitter -> 0.0.
  health               state factor (online 1.0, degraded health_degraded_factor, offline/disconnected 0) *
                       (0.5 + 0.5*clamp(update_rate_hz / rate_full_hz)). Unknown rate -> rate factor 0.0 (i.e. half credit).
"""
from __future__ import annotations

import math
from typing import Mapping

from .config import ScoreWeights, ScoringCfg


def clamp01(x: float) -> float:
    return 0.0 if x != x else max(0.0, min(1.0, x))


def check_unit(name: str, v) -> float:
    if v is None:
        raise ValueError(f"{name} is missing")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"{name}={v!r} is not a finite number in [0, 1]")
    return float(v)


def offer_quality(weights: ScoreWeights, *, detector_confidence, visibility_score, freshness_score, sync_score, health_score) -> float:
    q = (weights.detector_confidence * check_unit("detector_confidence", detector_confidence)
         + weights.visibility * check_unit("visibility_score", visibility_score)
         + weights.freshness * check_unit("freshness_score", freshness_score)
         + weights.sync * check_unit("sync_score", sync_score)
         + weights.health * check_unit("health_score", health_score))
    return clamp01(q)


def quality_of(weights: ScoreWeights, offer) -> float:
    return offer_quality(weights, detector_confidence=offer.detector_confidence, visibility_score=offer.visibility_score,
                         freshness_score=offer.freshness_score, sync_score=offer.sync_score, health_score=offer.health_score)


def detector_confidence_value(raw: float | None, cfg: ScoringCfg) -> tuple[float, str]:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        return cfg.imputed_detector_confidence, "imputed"
    return clamp01(float(raw)), "measured"


def freshness_score(age_ms: float | None, cfg: ScoringCfg) -> float:
    if age_ms is None or not math.isfinite(age_ms):
        return 0.0
    return clamp01(1.0 - max(age_ms, 0.0) / cfg.freshness_horizon_ms)


def sync_score(clock_offset_known: bool, delay_jitter_ms: float | None, cfg: ScoringCfg) -> float:
    if not clock_offset_known or delay_jitter_ms is None or not math.isfinite(delay_jitter_ms):
        return 0.0
    return clamp01(1.0 - max(delay_jitter_ms, 0.0) / cfg.sync_jitter_horizon_ms)


_STATE_FACTOR = {"online": 1.0, "degraded": None, "offline": 0.0, "disconnected": 0.0}


def health_score(state: str, update_rate_hz: float | None, cfg: ScoringCfg) -> float:
    if state not in _STATE_FACTOR:
        return 0.0                                              # unknown state is never healthy
    f = cfg.health_degraded_factor if state == "degraded" else _STATE_FACTOR[state]
    rate = 0.0 if update_rate_hz is None or not math.isfinite(update_rate_hz) else clamp01(update_rate_hz / cfg.rate_full_hz)
    return clamp01(f * (0.5 + 0.5 * rate))


def visibility_score(status: str, landmarks: Mapping[str, tuple], landmark: str, width: int, height: int,
                     flags: tuple[str, ...], cfg: ScoringCfg) -> float:
    if status != "tracked" or landmark not in landmarks or width <= 0 or height <= 0:
        return 0.0
    x, y = landmarks[landmark][0], landmarks[landmark][1]
    if not (math.isfinite(x) and math.isfinite(y)):
        return 0.0
    margin = min(x, y, width - x, height - y) / float(min(width, height))
    if margin < 0:                                              # outside the frame
        return 0.0
    v = 0.5 + 0.5 * clamp01(margin / cfg.edge_margin_frac)
    return min(v, 0.3) if "out_of_frame" in flags else v
