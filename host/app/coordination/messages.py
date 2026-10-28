"""Coordination-channel messages, version 1.0 (separate namespace from the sensor wire protocol).

NAME COLLISIONS (resolved here, documented in docs/coordination.md):
* ``SENSOR_STATUS`` already exists as a sensor->host WIRE message (schemas/messages.py). The coordination
  ``SENSOR_STATUS`` is a HOST-INTERNAL summary (registry state + liveness) with ``channel="coordination"``; it is never
  accepted from a socket and never serialised onto one.
* ``FUSION_RESULT`` is the coordinator's EVALUATION of a Phase 2 ``POSE_UPDATE`` (it references ``fusion_sequence``; it
  never carries coordinates). ``POSE_UPDATE`` itself is unchanged.
Field names and units follow Phase 1/2: ``*_ms`` epoch/duration milliseconds on the HOST clock, ``*_m`` metres, ``*_px`` pixels.
"""
from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Mapping

PROTOCOL_VERSION = "1.0"
CHANNEL = "coordination"


class MessageError(ValueError):
    """Invalid message. ``code``: MISSING_FIELD | WRONG_TYPE | OUT_OF_RANGE | UNSUPPORTED_VERSION | UNSUPPORTED_MESSAGE_TYPE."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class Mode(str, enum.Enum):
    TRIANGULATION = "TRIANGULATION"                       # calibrated multi-view
    SINGLE_VIEW_APPROXIMATE = "SINGLE_VIEW_APPROXIMATE"   # labelled approximate; NOT triangulation
    NONE = "NONE"


class Verdict(str, enum.Enum):
    ACCEPTED = "ACCEPTED"
    POOR_QUALITY = "POOR_QUALITY"
    UNAVAILABLE = "UNAVAILABLE"


class RequestReason(str, enum.Enum):
    INITIAL = "INITIAL"
    RENEWAL = "RENEWAL"
    RESELECT_FUSION_FAILURE = "RESELECT_FUSION_FAILURE"
    RESELECT_SENSOR_LOSS = "RESELECT_SENSOR_LOSS"
    RESELECT_CALIBRATION = "RESELECT_CALIBRATION"
    RETRY_AFTER_DEGRADED = "RETRY_AFTER_DEGRADED"


@dataclass(frozen=True, kw_only=True)
class _Msg:
    message_type: ClassVar[str] = ""
    protocol_version: str = PROTOCOL_VERSION
    channel: str = CHANNEL

    def to_dict(self) -> dict:
        d = asdict(self)
        d["message_type"] = self.message_type
        return _plain(d)


def _plain(v):
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_plain(x) for x in (sorted(v) if isinstance(v, (set, frozenset)) else v)]
    return v


@dataclass(frozen=True, kw_only=True)
class TaskRequest(_Msg):
    """Stage 1 TASK ANNOUNCEMENT: observe ``target_landmark`` with capture time inside the window, answer by ``deadline_ms``."""
    message_type: ClassVar[str] = "TASK_REQUEST"
    session_id: str
    task_id: str
    parent_task_id: str | None
    sequence_number: int
    target_landmark: str
    issued_at_ms: int
    collect_until_ms: int             # offers after this are not waited for
    deadline_ms: int                  # offers arriving after this are rejected OFFER_LATE
    window_start_ms: int              # acceptable capture time (host timeline)
    window_end_ms: int
    max_lease_ms: int
    min_views: int
    trusted_3d_required: bool
    attempt: int
    reason: RequestReason

    def __post_init__(self):
        if not (self.issued_at_ms <= self.collect_until_ms <= self.deadline_ms):
            raise MessageError("OUT_OF_RANGE", "need issued_at_ms <= collect_until_ms <= deadline_ms")
        if self.window_start_ms > self.window_end_ms or self.max_lease_ms <= 0 or self.min_views < 1:
            raise MessageError("OUT_OF_RANGE", "invalid window / lease / min_views")


_OFFER_REQUIRED = (
    "session_id", "task_id", "sensor_id", "offer_sequence", "target_landmark", "detector_confidence",
    "detector_confidence_source", "visibility_score", "freshness_score", "sync_score", "health_score",
    "estimated_latency_ms", "calibration_eligible", "capabilities", "observation_effective_ms", "observation_age_ms",
    "submitted_at_ms", "view_ray", "rig_key", "hand", "notes")
_UNIT_FIELDS = ("detector_confidence", "visibility_score", "freshness_score", "sync_score", "health_score")


def _num(name: str, v: Any, *, lo: float | None = None, hi: float | None = None, nullable: bool = False):
    if v is None:
        if nullable:
            return None
        raise MessageError("MISSING_FIELD", f"{name} is null")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MessageError("WRONG_TYPE", f"{name} must be a number")
    if not math.isfinite(v):
        raise MessageError("OUT_OF_RANGE", f"{name} is not finite")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise MessageError("OUT_OF_RANGE", f"{name}={v} outside [{lo}, {hi}]")
    return v


@dataclass(frozen=True, kw_only=True)
class ObservationOffer(_Msg):
    """Stage 3 OFFER SUBMISSION. All scores are normalised 0..1; missing values are REJECTED here, never defaulted.

    ``view_ray`` = (origin_m[3], direction[3]) world-frame viewing ray through the target landmark (no 3D point);
    ``rig_key`` identifies the calibration rig/world frame so the coordinator can check compatibility.
    ``None`` for either means "no calibrated geometry" (offer is incomplete for trusted triangulation).
    """
    message_type: ClassVar[str] = "OBSERVATION_OFFER"
    session_id: str
    task_id: str
    sensor_id: str
    offer_sequence: int
    target_landmark: str
    detector_confidence: float
    detector_confidence_source: str          # "measured" | "imputed" (imputed values are flagged, never silent)
    visibility_score: float
    freshness_score: float
    sync_score: float
    health_score: float
    estimated_latency_ms: float
    calibration_eligible: bool
    capabilities: tuple[str, ...]
    observation_effective_ms: float | None   # capture time on the host timeline (Phase 2 ``effective_time_ms``)
    observation_age_ms: float | None
    submitted_at_ms: int
    view_ray: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    rig_key: str | None
    hand: str
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ObservationOffer":
        if not isinstance(d, Mapping):
            raise MessageError("WRONG_TYPE", "offer must be a mapping")
        if d.get("protocol_version") != PROTOCOL_VERSION:
            raise MessageError("UNSUPPORTED_VERSION", f"protocol_version must be {PROTOCOL_VERSION!r}")
        if d.get("message_type") != cls.message_type:
            raise MessageError("UNSUPPORTED_MESSAGE_TYPE", "not an OBSERVATION_OFFER")
        missing = [k for k in _OFFER_REQUIRED if k not in d]
        if missing:
            raise MessageError("MISSING_FIELD", f"missing {missing}")
        for k in ("session_id", "task_id", "sensor_id", "target_landmark", "hand", "detector_confidence_source"):
            if not isinstance(d[k], str) or not d[k]:
                raise MessageError("WRONG_TYPE", f"{k} must be a non-empty string")
        if d["detector_confidence_source"] not in ("measured", "imputed"):
            raise MessageError("OUT_OF_RANGE", "detector_confidence_source must be measured|imputed")
        seq = d["offer_sequence"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise MessageError("OUT_OF_RANGE", "offer_sequence must be an integer >= 0")
        vals = {k: _num(k, d[k], lo=0.0, hi=1.0) for k in _UNIT_FIELDS}
        lat = _num("estimated_latency_ms", d["estimated_latency_ms"], lo=0.0)
        if not isinstance(d["calibration_eligible"], bool):
            raise MessageError("WRONG_TYPE", "calibration_eligible must be a boolean")
        caps = d["capabilities"]
        if not isinstance(caps, (list, tuple)) or not all(isinstance(c, str) for c in caps):
            raise MessageError("WRONG_TYPE", "capabilities must be a list of strings")
        eff = _num("observation_effective_ms", d["observation_effective_ms"], nullable=True)
        age = _num("observation_age_ms", d["observation_age_ms"], nullable=True)
        sub = d["submitted_at_ms"]
        if isinstance(sub, bool) or not isinstance(sub, int) or sub <= 0:
            raise MessageError("OUT_OF_RANGE", "submitted_at_ms must be a positive integer")
        ray = d["view_ray"]
        if ray is not None:
            try:
                o, dr = ray
                o, dr = tuple(float(x) for x in o), tuple(float(x) for x in dr)
                assert len(o) == 3 and len(dr) == 3 and all(math.isfinite(x) for x in o + dr)
                n = math.sqrt(sum(x * x for x in dr))
            except Exception:
                raise MessageError("WRONG_TYPE", "view_ray must be (origin[3], direction[3]) of finite numbers") from None
            if abs(n - 1.0) > 1e-3:
                raise MessageError("OUT_OF_RANGE", "view_ray direction must be a unit vector")
            ray = (o, dr)
        if d["rig_key"] is not None and not isinstance(d["rig_key"], str):
            raise MessageError("WRONG_TYPE", "rig_key must be a string or null")
        return cls(session_id=d["session_id"], task_id=d["task_id"], sensor_id=d["sensor_id"], offer_sequence=seq,
                   target_landmark=d["target_landmark"], detector_confidence_source=d["detector_confidence_source"],
                   estimated_latency_ms=lat, calibration_eligible=d["calibration_eligible"], capabilities=tuple(caps),
                   observation_effective_ms=eff, observation_age_ms=age, submitted_at_ms=sub, view_ray=ray,
                   rig_key=d["rig_key"], hand=d["hand"], notes=tuple(d["notes"] or ()), **vals)


@dataclass(frozen=True, kw_only=True)
class TaskAward(_Msg):
    """Stage 6 AWARD / LEASE. Valid only until ``lease_expires_at_ms``; ``decision_epoch`` increases with every award."""
    message_type: ClassVar[str] = "TASK_AWARD"
    session_id: str
    task_id: str
    award_id: str
    sequence_number: int
    decision_epoch: int
    target_landmark: str
    sensor_ids: tuple[str, ...]
    mode: Mode
    trusted_3d: bool
    awarded_at_ms: int
    lease_expires_at_ms: int
    set_score: float | None
    reason: str

    def __post_init__(self):
        if self.lease_expires_at_ms <= self.awarded_at_ms:
            raise MessageError("OUT_OF_RANGE", "lease must expire after it is awarded")
        if self.trusted_3d and (self.mode != Mode.TRIANGULATION or len(self.sensor_ids) < 2):
            raise MessageError("OUT_OF_RANGE", "trusted_3d requires TRIANGULATION mode and >= 2 sensors")


@dataclass(frozen=True, kw_only=True)
class TaskReject(_Msg):
    """Stage 6 REJECTION. ``sensor_id`` set = one offer/sensor rejected; ``None`` = the whole task got no award."""
    message_type: ClassVar[str] = "TASK_REJECT"
    session_id: str
    task_id: str
    sequence_number: int
    sensor_id: str | None
    rejected_at_ms: int
    reason_code: str
    detail: str


@dataclass(frozen=True, kw_only=True)
class LeaseExpired(_Msg):
    message_type: ClassVar[str] = "LEASE_EXPIRED"
    session_id: str
    task_id: str
    award_id: str
    sequence_number: int
    sensor_ids: tuple[str, ...]
    expired_at_ms: int
    reason: str                       # LEASE_TIMEOUT | SUPERSEDED | FUSION_FAILURE | SENSOR_UNAVAILABLE | CALIBRATION_INVALID | SESSION_STOPPED | RELEASED


@dataclass(frozen=True, kw_only=True)
class FusionEvaluation(_Msg):
    """``FUSION_RESULT``: the coordinator's verdict on a Phase 2 POSE_UPDATE (no coordinates)."""
    message_type: ClassVar[str] = "FUSION_RESULT"
    session_id: str
    task_id: str
    award_id: str
    sequence_number: int
    fusion_sequence: int
    fusion_status: str
    verdict: Verdict
    reason: str | None
    aggregate_reprojection_error_px: float | None
    ray_angle_deg: float | None
    consecutive_poor: int
    evaluated_at_ms: int


@dataclass(frozen=True, kw_only=True)
class DegradedStatus(_Msg):
    message_type: ClassVar[str] = "DEGRADED_STATUS"
    session_id: str
    task_id: str | None
    sequence_number: int
    entered_at_ms: int
    reason_code: str
    detail: str
    tracking_trusted: bool
    mode: Mode
    retry_at_ms: int | None


@dataclass(frozen=True, kw_only=True)
class SensorStatusMsg(_Msg):
    """Host-internal sensor liveness/health summary (NOT the sensor wire message of the same name)."""
    message_type: ClassVar[str] = "SENSOR_STATUS"
    session_id: str
    sensor_id: str
    sequence_number: int
    observed_at_ms: int
    state: str                        # online | degraded | offline | disconnected  (Phase 1 SensorState values)
    reason: str | None = None
    calibration_ok: bool = True


MESSAGE_TYPES = {c.message_type: c for c in (TaskRequest, ObservationOffer, TaskAward, TaskReject, LeaseExpired,
                                             FusionEvaluation, DegradedStatus, SensorStatusMsg)}
