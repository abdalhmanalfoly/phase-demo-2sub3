"""The fusion RESULT contract (message_type ``POSE_UPDATE``): what Phase 3 (coordinator, gesture) and Phase 4
(world adapter / Godot) consume. All fields are always present; absent values are explicit ``null``.

Status vocabulary (see docs/spatial.md for the mapping to the Phase 1 names):
  TRACKING_MULTI_VIEW   valid calibrated multi-view triangulation  -> ``world_xyz_m`` is set (the ONLY status where it is)
  SYNC_DEGRADED         >= 2 views but their capture times are too far apart to represent one instant
  SINGLE_VIEW_FALLBACK  exactly one usable view: only a viewing RAY is given (never a point)
  INSUFFICIENT_VIEWS    no usable observation right now, last valid estimate still inside the hold window (``held_*``)
  TARGET_LOST           no usable observation and the hold window expired / never valid
  FUSION_OUTLIER        views disagree geometrically (reprojection error too high) or conflicting hand labels
  GEOMETRY_DEGENERATE   triangulation numerically unreliable/impossible (parallel rays, behind a camera, non-finite)
  CALIBRATION_REQUIRED  no valid, compatible calibration for the sensors that are observing
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class FusionStatus(str, enum.Enum):
    TRACKING_MULTI_VIEW = "TRACKING_MULTI_VIEW"
    SYNC_DEGRADED = "SYNC_DEGRADED"
    SINGLE_VIEW_FALLBACK = "SINGLE_VIEW_FALLBACK"
    INSUFFICIENT_VIEWS = "INSUFFICIENT_VIEWS"
    TARGET_LOST = "TARGET_LOST"
    FUSION_OUTLIER = "FUSION_OUTLIER"
    GEOMETRY_DEGENERATE = "GEOMETRY_DEGENERATE"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"


Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceInfo(_M):
    sensor_id: str
    camera_id: str | None
    calibration_id: str | None
    sequence_number: int | None
    frame_id: int | None
    captured_at_ms: int | None             # sensor clock, untouched
    received_at_ms: int | None             # host clock
    effective_time_ms: float | None        # captured_at_ms + estimated clock offset (common host timeline)
    age_ms: float | None                   # now - effective_time_ms at publication
    transport_delay_ms: float | None       # received_at_ms - effective_time_ms (diagnostic; includes offset error)
    reprojection_error_px: float | None
    used: bool
    note: str | None                       # why a sensor did not contribute (e.g. DIMENSION_MISMATCH, STALE)


class SingleViewRay(_M):
    sensor_id: str
    origin_m: Vec3                          # camera centre in the world frame
    direction: Vec3                         # unit viewing direction through the landmark


class QualityInfo(_M):
    score: float | None = Field(default=None, ge=0, le=1)
    reprojection: float | None
    sync: float | None
    geometry: float | None


class CalibrationRef(_M):
    rig_id: str
    rig_calibration_id: str
    world_frame_id: str
    calibration_ids: dict[str, str]         # sensor_id -> calibration_id actually used


class FusionResult(_M):
    protocol_version: Literal["1.0"] = "1.0"
    message_type: Literal["POSE_UPDATE"] = "POSE_UPDATE"
    session_id: str
    target_landmark: str
    fusion_sequence: int = Field(ge=0)                  # strictly increasing per (session, target); consumers drop stale ones
    status: FusionStatus
    valid: bool                                         # == (status is TRACKING_MULTI_VIEW)
    reason: str | None                                  # machine-readable diagnostic code
    message: str                                        # human-readable explanation
    timestamp_ms: float | None                          # effective capture time of the fused set (host timeline)
    published_at_ms: int
    unit: Literal["m"] = "m"
    world_frame_id: str | None
    world_xyz_m: Vec3 | None                            # smoothed; ONLY for TRACKING_MULTI_VIEW
    raw_world_xyz_m: Vec3 | None                        # unsmoothed triangulation of the same set
    held_world_xyz_m: Vec3 | None                       # last valid estimate while inside the hold window (NOT current)
    held_age_ms: float | None
    single_view_ray: SingleViewRay | None
    source_sensor_ids: list[str]                        # sensors that contributed to the 3D point
    sources: list[SourceInfo]
    skew_ms: float | None                               # max-min corrected capture time inside the set
    aggregate_reprojection_error_px: float | None       # RMS over contributing cameras
    ray_angle_deg: float | None
    quality: QualityInfo
    calibration: CalibrationRef | None
    smoothing_applied: bool
    warnings: list[str]
