"""Coordination configuration (``config/coordination.yaml``).

Plain dataclasses with explicit validation (no pydantic): the coordination core therefore has no dependency beyond
the standard library + NumPy and can be unit-tested without the web stack. Unknown keys are rejected, and every
threshold is a documented STARTING POINT, not a measured optimum.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


def _finite(name: str, v: float, lo: float | None = None, hi: float | None = None, lo_open: bool = False) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ConfigError(f"{name} must be a finite number (got {v!r})")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ConfigError(f"{name} must be {'>' if lo_open else '>='} {lo} (got {v})")
    if hi is not None and v > hi:
        raise ConfigError(f"{name} must be <= {hi} (got {v})")


@dataclass
class ScoreWeights:
    """Baseline offer-quality weights. A configurable baseline, NOT a proven optimum."""
    detector_confidence: float = 0.30
    visibility: float = 0.20
    freshness: float = 0.20
    sync: float = 0.15
    health: float = 0.15

    def validate(self) -> None:
        for f in dataclasses.fields(self):
            _finite(f"weights.{f.name}", getattr(self, f.name), 0.0, 1.0)
        total = sum(getattr(self, f.name) for f in dataclasses.fields(self))
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"weights must sum to 1.0 (got {total:.6f})")


@dataclass
class ScoringCfg:
    freshness_horizon_ms: float = 250.0      # observation age at which freshness_score reaches 0
    sync_jitter_horizon_ms: float = 50.0     # transport-delay jitter at which sync_score reaches 0
    health_degraded_factor: float = 0.4      # health_score of a 'degraded' sensor (before the rate factor)
    rate_full_hz: float = 10.0               # observation rate at which the rate factor saturates
    edge_margin_frac: float = 0.05           # landmark closer than this fraction of min(w,h) to the border loses visibility
    imputed_detector_confidence: float = 0.5 # used (and FLAGGED) when the detector gives no confidence (the hand model never does)
    assumed_latency_ms: float = 100.0        # latency assumed (conservatively) when the delay is unknown

    def validate(self) -> None:
        _finite("scoring.freshness_horizon_ms", self.freshness_horizon_ms, 0, lo_open=True)
        _finite("scoring.sync_jitter_horizon_ms", self.sync_jitter_horizon_ms, 0, lo_open=True)
        _finite("scoring.health_degraded_factor", self.health_degraded_factor, 0, 1)
        _finite("scoring.rate_full_hz", self.rate_full_hz, 0, lo_open=True)
        _finite("scoring.edge_margin_frac", self.edge_margin_frac, 0, 0.5, lo_open=True)
        _finite("scoring.imputed_detector_confidence", self.imputed_detector_confidence, 0, 1)
        _finite("scoring.assumed_latency_ms", self.assumed_latency_ms, 0)


@dataclass
class EligibilityCfg:
    max_observation_age_ms: float = 250.0    # same bound the Phase 2 synchronizer uses
    min_visibility: float = 0.2
    min_detector_confidence: float = 0.2
    min_health: float = 0.3
    require_calibration: bool = True

    def validate(self) -> None:
        _finite("eligibility.max_observation_age_ms", self.max_observation_age_ms, 0, lo_open=True)
        for n in ("min_visibility", "min_detector_confidence", "min_health"):
            _finite(f"eligibility.{n}", getattr(self, n), 0, 1)


@dataclass
class TaskCfg:
    offer_collect_ms: float = 150.0          # offers are collected at most this long
    deadline_ms: float = 400.0               # hard deadline: nothing after it is accepted, no award is made after it
    window_ms: float = 250.0                 # acceptable capture-time window, ending at the announcement time
    lease_ms: float = 2000.0
    max_lease_ms: float = 5000.0             # hard cap: a lease is always bounded
    renew_lead_ms: float = 300.0             # start the successor negotiation this long before the lease expires
    recent_tasks: int = 32                   # remembered task ids (late/duplicate messages for them are recognised)

    def validate(self) -> None:
        _finite("task.offer_collect_ms", self.offer_collect_ms, 0, lo_open=True)
        _finite("task.deadline_ms", self.deadline_ms, 0, lo_open=True)
        _finite("task.window_ms", self.window_ms, 0, lo_open=True)
        _finite("task.lease_ms", self.lease_ms, 0, lo_open=True)
        _finite("task.max_lease_ms", self.max_lease_ms, 0, 60_000, lo_open=True)
        _finite("task.renew_lead_ms", self.renew_lead_ms, 0)
        if self.deadline_ms < self.offer_collect_ms:
            raise ConfigError("task.deadline_ms must be >= offer_collect_ms")
        if self.lease_ms > self.max_lease_ms:
            raise ConfigError("task.lease_ms must be <= max_lease_ms (leases are bounded)")
        if self.renew_lead_ms >= self.lease_ms:
            raise ConfigError("task.renew_lead_ms must be < lease_ms")
        if self.recent_tasks < 1:
            raise ConfigError("task.recent_tasks must be >= 1")


@dataclass
class SetSelectionCfg:
    """Set-selection POLICY (kept separate from the per-sensor offer score)."""
    min_views: int = 2
    max_set_size: int = 3                    # resource constraint: more cameras = more bandwidth/compute
    max_skew_ms: float = 50.0                # capture-time agreement inside a set (matches fusion sync.max_skew_ms)
    min_view_angle_deg: float = 10.0         # stricter than fusion's 2 deg: depth from a near-parallel pair is poor
    good_view_angle_deg: float = 30.0        # geometry term saturates here
    max_ray_gap_m: float = 0.10              # viewing rays of one landmark must (nearly) intersect
    max_latency_ms: float = 150.0            # per-sensor latency budget
    w_quality: float = 0.50
    w_geometry: float = 0.35
    w_sync: float = 0.15
    size_penalty: float = 0.02               # subtracted per sensor beyond min_views
    failed_set_ban_ms: float = 3000.0        # a set whose fusion failed is not re-selected for this long
    max_candidates: int = 6                  # only the best N eligible offers enter the subset search

    def validate(self) -> None:
        if not isinstance(self.min_views, int) or self.min_views < 2:
            raise ConfigError("set_selection.min_views must be an integer >= 2")
        if not isinstance(self.max_set_size, int) or self.max_set_size < self.min_views:
            raise ConfigError("set_selection.max_set_size must be an integer >= min_views")
        _finite("set_selection.max_skew_ms", self.max_skew_ms, 0, lo_open=True)
        _finite("set_selection.min_view_angle_deg", self.min_view_angle_deg, 0, 90)
        _finite("set_selection.good_view_angle_deg", self.good_view_angle_deg, 0, 180, lo_open=True)
        if self.good_view_angle_deg <= self.min_view_angle_deg:
            raise ConfigError("set_selection.good_view_angle_deg must exceed min_view_angle_deg")
        _finite("set_selection.max_ray_gap_m", self.max_ray_gap_m, 0, lo_open=True)
        _finite("set_selection.max_latency_ms", self.max_latency_ms, 0, lo_open=True)
        for n in ("w_quality", "w_geometry", "w_sync"):
            _finite(f"set_selection.{n}", getattr(self, n), 0, 1)
        if abs(self.w_quality + self.w_geometry + self.w_sync - 1.0) > 1e-6:
            raise ConfigError("set_selection weights (w_quality+w_geometry+w_sync) must sum to 1.0")
        _finite("set_selection.size_penalty", self.size_penalty, 0, 0.5)
        _finite("set_selection.failed_set_ban_ms", self.failed_set_ban_ms, 0)
        if self.max_candidates < self.min_views:
            raise ConfigError("set_selection.max_candidates must be >= min_views")


@dataclass
class AdaptationCfg:
    fusion_max_reprojection_px: float = 4.0  # coordinator's own, stricter-than-fusion (6 px) quality bar
    poor_results_before_reselect: int = 3    # consecutive poor/unavailable fusion results tolerated
    max_reselect_attempts: int = 3           # failure-driven re-selections before DEGRADED
    fusion_silence_ms: float = 600.0         # no fusion result at all for this long counts as a poor result
    degraded_retry_ms: float = 500.0         # first automatic retry after DEGRADED (doubles, capped below)
    degraded_retry_max_ms: float = 4000.0
    sensor_silence_timeout_ms: float = 1500.0  # no SENSOR_STATUS for this long -> sensor treated as disconnected
    disconnect_grace_ms: float = 500.0       # a reported disconnect/offline lease member is tolerated this long (reconnects)
    allow_single_view_approximate: bool = False  # labelled approximate mode; NEVER called triangulation

    def validate(self) -> None:
        _finite("adaptation.fusion_max_reprojection_px", self.fusion_max_reprojection_px, 0, lo_open=True)
        if self.poor_results_before_reselect < 1 or self.max_reselect_attempts < 0:
            raise ConfigError("adaptation: poor_results_before_reselect >= 1 and max_reselect_attempts >= 0 required")
        for n in ("fusion_silence_ms", "degraded_retry_ms", "degraded_retry_max_ms", "sensor_silence_timeout_ms"):
            _finite(f"adaptation.{n}", getattr(self, n), 0, lo_open=True)
        _finite("adaptation.disconnect_grace_ms", self.disconnect_grace_ms, 0)
        if self.degraded_retry_max_ms < self.degraded_retry_ms:
            raise ConfigError("adaptation.degraded_retry_max_ms must be >= degraded_retry_ms")


@dataclass
class CoordinationCfg:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    eligibility: EligibilityCfg = field(default_factory=EligibilityCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    set_selection: SetSelectionCfg = field(default_factory=SetSelectionCfg)
    adaptation: AdaptationCfg = field(default_factory=AdaptationCfg)
    subtract_half_pixel: bool = True         # must equal fusion.yaml spatial.subtract_half_pixel

    def validate(self) -> "CoordinationCfg":
        for part in (self.weights, self.scoring, self.eligibility, self.task, self.set_selection, self.adaptation):
            part.validate()
        if self.set_selection.max_latency_ms >= self.task.deadline_ms:
            raise ConfigError("set_selection.max_latency_ms must be < task.deadline_ms")
        return self


def _build(cls, data: Mapping[str, Any], path: str):
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a mapping")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for k, v in data.items():
        ftype = known[k].type
        sub = {"ScoreWeights": ScoreWeights, "ScoringCfg": ScoringCfg, "EligibilityCfg": EligibilityCfg, "TaskCfg": TaskCfg,
               "SetSelectionCfg": SetSelectionCfg, "AdaptationCfg": AdaptationCfg}.get(ftype if isinstance(ftype, str) else "")
        kwargs[k] = _build(sub, v, f"{path}.{k}") if sub else v
    return cls(**kwargs)


def config_from_mapping(data: Mapping[str, Any] | None) -> CoordinationCfg:
    cfg = _build(CoordinationCfg, data or {}, "coordination")
    return cfg.validate()


def load_coordination_config(path: Path | str | None = None) -> CoordinationCfg:
    import yaml
    p = Path(path) if path else Path(__file__).resolve().parents[3] / "config" / "coordination.yaml"
    if not p.exists():
        return CoordinationCfg().validate()
    with p.open("r", encoding="utf-8") as fh:
        return config_from_mapping(yaml.safe_load(fh) or {})
