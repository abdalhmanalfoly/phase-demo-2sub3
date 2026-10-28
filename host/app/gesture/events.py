"""Interaction-event contract, version 1.0 (Gesture Agent -> future World Adapter). High-level INTENT, no engine internals.

Discrete events (``continuous=False``) are state-transition facts: each has a unique ``event_id`` and is emitted at most once per
episode, so a consumer may de-duplicate by ``event_id`` (see ``EventDeduper``) and replays are harmless.
``MOVE_TARGET`` is the CONTINUOUS pose stream (``continuous=True``): rate-limited, only while GRABBED, only with a VALID fresh fused
position. ``hand_target_xyz_m`` is ``None`` whenever the fusion result is invalid; TRACKING_LOST never carries a position.

Consumer rules (documented in docs/gesture.md): on TRACKING_LOST freeze the object where it is; never extrapolate; on GRAB_CANCELLED or
RELEASE let the object fall/behave normally in place (no impulse is implied); ignore events whose ``sequence_number`` is lower than one seen.
"""
from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass

PROTOCOL_VERSION = "1.0"
MESSAGE_TYPE = "INTERACTION_EVENT"


class GestureState(str, enum.Enum):
    IDLE = "IDLE"
    HOVER = "HOVER"
    GRABBED = "GRABBED"
    RELEASED = "RELEASED"


class EventType(str, enum.Enum):
    HOVER_ENTER = "HOVER_ENTER"
    HOVER_EXIT = "HOVER_EXIT"
    GRAB_REQUEST = "GRAB_REQUEST"
    GRAB_CONFIRMED = "GRAB_CONFIRMED"
    MOVE_TARGET = "MOVE_TARGET"
    RELEASE = "RELEASE"
    GRAB_CANCELLED = "GRAB_CANCELLED"
    TRACKING_LOST = "TRACKING_LOST"


class CancelReason(str, enum.Enum):
    TRACKING_LOST_TIMEOUT = "TRACKING_LOST_TIMEOUT"
    PINCH_EVIDENCE_LOST = "PINCH_EVIDENCE_LOST"
    TRACK_DISCONTINUITY = "TRACK_DISCONTINUITY"
    TRACK_CHANGED = "TRACK_CHANGED"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    SESSION_STOPPED = "SESSION_STOPPED"
    ADAPTER_REJECTED = "ADAPTER_REJECTED"
    ADAPTER_ACK_TIMEOUT = "ADAPTER_ACK_TIMEOUT"
    RESET = "RESET"
    SAFETY = "SAFETY"


@dataclass(frozen=True, kw_only=True)
class InteractionEvent:
    session_id: str
    event_id: str
    sequence_number: int
    event_type: EventType
    timestamp_ms: int                         # host time at emission
    gesture_state: GestureState               # state AFTER the transition this event announces
    continuous: bool
    interaction_id: str | None
    target_object_id: str | None
    hand_target_xyz_m: tuple[float, float, float] | None
    world_frame_id: str | None
    confidence: float | None
    cancel_reason: CancelReason | None
    reason: str | None
    sample_timestamp_ms: float | None = None  # capture time of the pose behind hand_target_xyz_m
    unit: str = "m"
    protocol_version: str = PROTOCOL_VERSION
    message_type: str = MESSAGE_TYPE

    def __post_init__(self):
        et = self.event_type
        if self.hand_target_xyz_m is not None:
            if len(self.hand_target_xyz_m) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.hand_target_xyz_m):
                raise ValueError("hand_target_xyz_m must be 3 finite numbers")
        if et == EventType.MOVE_TARGET and (self.hand_target_xyz_m is None or not self.continuous):
            raise ValueError("MOVE_TARGET is continuous and requires a valid position")
        if et != EventType.MOVE_TARGET and self.continuous:
            raise ValueError("only MOVE_TARGET is a continuous event")
        if et == EventType.TRACKING_LOST and self.hand_target_xyz_m is not None:
            raise ValueError("TRACKING_LOST must not carry a position")
        if (et == EventType.GRAB_CANCELLED) != (self.cancel_reason is not None):
            raise ValueError("cancel_reason is required for (and only for) GRAB_CANCELLED")
        if self.confidence is not None and not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("event_type", "gesture_state", "cancel_reason"):
            if d[k] is not None:
                d[k] = d[k].value if hasattr(d[k], "value") else d[k]
        if d["hand_target_xyz_m"] is not None:
            d["hand_target_xyz_m"] = list(d["hand_target_xyz_m"])
        return d


class EventDeduper:
    """Consumer-side helper: ``accept(event)`` is True only the first time an ``event_id`` is seen (bounded memory)."""

    def __init__(self, max_ids: int = 4096):
        self._seen: dict[str, None] = {}
        self.max_ids = max_ids

    def accept(self, ev: InteractionEvent) -> bool:
        if ev.event_id in self._seen:
            return False
        self._seen[ev.event_id] = None
        if len(self._seen) > self.max_ids:
            self._seen.pop(next(iter(self._seen)))
        return True
