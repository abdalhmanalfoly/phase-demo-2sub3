"""Interface between the coordinator and the (existing) Phase 2 fusion service.

The coordinator never triangulates. It (a) restricts which sensors fusion may use via ``fusion_gate`` and (b) reads results
through ``FusionOutcome``, a normalised view of ``FusionResult`` (message_type ``POSE_UPDATE``). ``from_fusion_result``
is duck-typed, so it works on the real pydantic object without importing pydantic here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

FusionGate = Callable[[str, "frozenset[str] | None"], None]    # (target_landmark, allowed sensor ids | None = unrestricted)


@dataclass(frozen=True)
class FusionOutcome:
    session_id: str
    target_landmark: str
    fusion_sequence: int
    status: str                            # FusionStatus value, e.g. TRACKING_MULTI_VIEW
    valid: bool
    reason: str | None
    rms_px: float | None
    ray_angle_deg: float | None
    skew_ms: float | None
    quality_score: float | None
    source_sensor_ids: tuple[str, ...]
    published_at_ms: int
    sensor_notes: dict = field(default_factory=dict)     # sensor_id -> why it did not contribute (Phase 2 ``SourceInfo.note``)
    sensor_reprojection_px: dict = field(default_factory=dict)

    @classmethod
    def from_fusion_result(cls, r) -> "FusionOutcome":
        status = getattr(r.status, "value", r.status)
        notes = {s.sensor_id: s.note for s in r.sources if getattr(s, "note", None)}
        errs = {s.sensor_id: s.reprojection_error_px for s in r.sources if getattr(s, "reprojection_error_px", None) is not None}
        q = getattr(r, "quality", None)
        return cls(session_id=r.session_id, target_landmark=r.target_landmark, fusion_sequence=int(r.fusion_sequence),
                   status=str(status), valid=bool(r.valid), reason=r.reason, rms_px=r.aggregate_reprojection_error_px,
                   ray_angle_deg=r.ray_angle_deg, skew_ms=r.skew_ms, quality_score=None if q is None else q.score,
                   source_sensor_ids=tuple(sorted(r.source_sensor_ids)), published_at_ms=int(r.published_at_ms),
                   sensor_notes=notes, sensor_reprojection_px=errs)
