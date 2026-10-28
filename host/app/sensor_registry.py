"""Sensor registry: identity, health state, sequence/clock/rate validation, counters.

Health states (derived, see :func:`evaluate_state`):
  online        connected, heartbeats on time, no self-reported failure
  degraded      connected but heartbeat late, OR the sensor reports permission denied / camera
                unavailable / inference error
  offline       connected socket but no heartbeat for ``offline_after_ms`` (timeout)
  disconnected  socket closed (clean departure or drop); the sensor_id is kept so it can resume
"""
from __future__ import annotations

import collections
import enum
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .schemas.messages import Heartbeat, Hello, LandmarkObservation, ProtocolError, SensorStatus

MAX_RECORDS_PER_SESSION = 64      # hard cap on distinct sensor identities (memory bound)


class SensorState(str, enum.Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DISCONNECTED = "disconnected"


class RateMeter:
    """Events/second over a sliding window (elapsed floored at 1 s so start-up is not overstated)."""

    def __init__(self, window_ms: int):
        self.window_ms = window_ms
        self._ts: collections.deque[int] = collections.deque()

    def _trim(self, now: int) -> None:
        while self._ts and now - self._ts[0] > self.window_ms:
            self._ts.popleft()

    def mark(self, now: int) -> None:
        self._ts.append(now)
        self._trim(now)

    def count(self, now: int) -> int:
        self._trim(now)
        return len(self._ts)

    def rate_hz(self, now: int) -> float:
        self._trim(now)
        if not self._ts:
            return 0.0
        return len(self._ts) / (max(now - self._ts[0], 1000) / 1000.0)


class ClockOffsetEstimator:
    """Estimate ``host_clock - sensor_clock`` from heartbeats.

    Each sample is offset + one-way delay, so the minimum over a window approximates the true offset
    (plus the smallest delay seen). Phones are not clock-synchronized to the host; this is how Phase 1
    stays usable without pretending clocks agree. Phase 2 may refine it.
    """

    def __init__(self, window: int = 20):
        self._samples: collections.deque[int] = collections.deque(maxlen=window)

    def add(self, client_ms: int, host_ms: int) -> None:
        self._samples.append(host_ms - client_ms)

    @property
    def offset_ms(self) -> int | None:
        return min(self._samples) if self._samples else None


@dataclass
class SensorRecord:
    sensor_id: str
    client_id: str
    session_id: str
    registered_at_ms: int
    client_version: str
    model_version: str | None
    capabilities: dict
    rate: RateMeter
    limiter: RateMeter
    offset: ClockOffsetEstimator = field(default_factory=ClockOffsetEstimator)
    connection: Any = None
    connection_id: int | None = None
    connected: bool = True
    state: SensorState = SensorState.ONLINE
    state_reason: str = "registered"
    # last-known client-reported status
    permission: str = "unknown"
    camera: str = "idle"
    inference: str = "idle"
    health: dict | None = None
    status_detail: str | None = None
    # liveness
    last_seen_ms: int = 0
    last_heartbeat_ms: int = 0
    heartbeat_count: int = 0
    last_heartbeat_seq: int = -1
    # observations
    last_obs_seq: int = -1
    last_obs_captured_ms: int | None = None
    last_obs_received_ms: int | None = None
    last_obs_status: str | None = None
    last_obs_hand: str | None = None
    last_obs_age_estimate_ms: int | None = None
    image_size: tuple[int, int] | None = None
    observations_accepted: int = 0
    observations_rejected: int = 0
    sequence_gaps: int = 0
    rejected_by_code: collections.Counter = field(default_factory=collections.Counter)
    last_error: dict | None = None
    reconnect_count: int = 0
    disconnect_reason: str | None = None


def evaluate_state(rec: SensorRecord, now: int, hb) -> tuple[SensorState, str]:
    """Pure function: the health state a sensor has at time ``now``."""
    if not rec.connected:
        return SensorState.DISCONNECTED, rec.disconnect_reason or "socket_closed"
    hb_age = now - rec.last_heartbeat_ms
    if hb_age > hb.offline_after_ms:
        return SensorState.OFFLINE, "heartbeat_timeout"
    if hb_age > hb.degraded_after_ms:
        return SensorState.DEGRADED, "heartbeat_late"
    if rec.permission == "denied":
        return SensorState.DEGRADED, "permission_denied"
    if rec.camera == "unavailable":
        return SensorState.DEGRADED, "camera_unavailable"
    if rec.inference == "error":
        return SensorState.DEGRADED, "inference_error"
    return SensorState.ONLINE, "ok"


class SensorRegistry:
    def __init__(self, session_id: str, settings: Settings, clock):
        self.session_id = session_id
        self.settings = settings
        self.clock = clock
        self.records: dict[str, SensorRecord] = {}
        self._by_client: dict[str, str] = {}
        self._counter = 0

    # ------------------------------------------------------------------ registration
    def _next_sensor_id(self) -> str:
        while True:
            self._counter += 1
            sid = f"sensor_{self._counter:02d}"
            if sid not in self.records:
                return sid

    def connected_count(self) -> int:
        return sum(1 for r in self.records.values() if r.connected)

    def register(self, hello: Hello, connection, connection_id: int) -> tuple[SensorRecord, bool, Any]:
        """Register or resume a sensor. Returns ``(record, resumed, previous_connection_or_None)``."""
        now = self.clock.now_ms()
        existing_id = self._by_client.get(hello.client_id)
        req = hello.requested_sensor_id
        if req is not None:
            owner = self.records.get(req)
            if owner is not None and owner.client_id != hello.client_id:
                raise ProtocolError("DUPLICATE_SENSOR_ID", f"sensor_id '{req}' is already in use in this session",
                                    fatal=True, related="HELLO")
            if existing_id is not None and existing_id != req:
                raise ProtocolError("SENSOR_ID_MISMATCH",
                                    f"this client is already registered as '{existing_id}'", fatal=True, related="HELLO")
        caps = hello.capabilities.model_dump(mode="json")
        if existing_id is not None:                       # ---- resume (reconnect / takeover)
            rec = self.records[existing_id]
            previous = rec.connection if rec.connected else None
            rec.reconnect_count += 1
            rec.last_obs_seq = -1                         # a new registration starts a new sequence epoch
            rec.last_heartbeat_seq = -1
            rec.last_obs_captured_ms = None
            resumed = True
        else:                                             # ---- new identity
            if self.connected_count() >= self.settings.session.max_sensors or len(self.records) >= MAX_RECORDS_PER_SESSION:
                raise ProtocolError("SESSION_FULL", "session has reached its maximum number of sensors",
                                    fatal=True, related="HELLO")
            sid = req or self._next_sensor_id()
            rec = SensorRecord(
                sensor_id=sid, client_id=hello.client_id, session_id=self.session_id, registered_at_ms=now,
                client_version=hello.client_version, model_version=hello.model_version, capabilities=caps,
                rate=RateMeter(self.settings.thresholds.observation.rate_window_ms), limiter=RateMeter(1000),
            )
            self.records[sid] = rec
            self._by_client[hello.client_id] = sid
            previous, resumed = None, False
        rec.client_version, rec.model_version, rec.capabilities = hello.client_version, hello.model_version, caps
        rec.connection, rec.connection_id, rec.connected, rec.disconnect_reason = connection, connection_id, True, None
        rec.last_seen_ms = rec.last_heartbeat_ms = now
        rec.offset.add(hello.client_time_ms, now)
        rec.camera = rec.inference = "idle"
        rec.permission = "unknown"
        rec.health = None
        rec.state, rec.state_reason = SensorState.ONLINE, "resumed" if resumed else "registered"
        return rec, resumed, previous

    # ------------------------------------------------------------------ inbound messages
    def record_heartbeat(self, rec: SensorRecord, hb: Heartbeat, now: int) -> None:
        if hb.sequence_number <= rec.last_heartbeat_seq:
            raise ProtocolError("STALE_SEQUENCE",
                                f"heartbeat sequence_number must exceed last accepted ({rec.last_heartbeat_seq})",
                                related="HEARTBEAT")
        rec.last_heartbeat_seq = hb.sequence_number
        rec.last_heartbeat_ms = rec.last_seen_ms = now
        rec.heartbeat_count += 1
        rec.offset.add(hb.client_time_ms, now)
        rec.health = hb.health.model_dump(mode="json")
        rec.camera, rec.inference = hb.health.camera, hb.health.inference

    def record_status(self, rec: SensorRecord, st: SensorStatus, now: int) -> bool:
        changed = (rec.permission, rec.camera, rec.inference) != (st.permission, st.camera, st.inference)
        rec.permission, rec.camera, rec.inference = st.permission, st.camera, st.inference
        rec.status_detail = st.detail
        rec.last_seen_ms = now
        return changed

    def accept_observation(self, rec: SensorRecord, obs: LandmarkObservation, now: int) -> dict:
        """Apply stateful checks; on success update counters and return the entry to record."""
        th = self.settings.thresholds.observation
        if obs.sequence_number <= rec.last_obs_seq:
            raise ProtocolError("STALE_SEQUENCE",
                                f"sequence_number must exceed last accepted ({rec.last_obs_seq})",
                                related="LANDMARK_OBSERVATION")
        if rec.last_obs_captured_ms is not None and obs.captured_at_ms < rec.last_obs_captured_ms:
            raise ProtocolError("STALE_OBSERVATION", "captured_at_ms went backwards", related="LANDMARK_OBSERVATION")
        age = None
        off = rec.offset.offset_ms
        if off is not None:
            age = now - (obs.captured_at_ms + off)
            if age > th.max_ingress_age_ms:
                raise ProtocolError("STALE_OBSERVATION",
                                    f"observation is {age} ms old (limit {th.max_ingress_age_ms} ms)",
                                    related="LANDMARK_OBSERVATION")
            if age < -th.max_future_skew_ms:
                raise ProtocolError("TIMESTAMP_IN_FUTURE", "captured_at_ms is ahead of the estimated host time",
                                    related="LANDMARK_OBSERVATION")
        if rec.limiter.count(now) >= self.settings.limits.max_observation_hz_host:
            raise ProtocolError("RATE_LIMITED", "observation rate exceeds the host limit", related="LANDMARK_OBSERVATION")
        # ---- accepted
        if rec.last_obs_seq >= 0:
            rec.sequence_gaps += obs.sequence_number - rec.last_obs_seq - 1
        rec.last_obs_seq = obs.sequence_number
        rec.last_obs_captured_ms = obs.captured_at_ms
        rec.last_obs_received_ms = rec.last_seen_ms = now
        rec.last_obs_status, rec.last_obs_hand = obs.status, obs.hand
        rec.last_obs_age_estimate_ms = age
        rec.image_size = (obs.image_width, obs.image_height)
        rec.observations_accepted += 1
        rec.rate.mark(now)
        rec.limiter.mark(now)
        return {"received_at_ms": now, "estimated_age_ms": age, **obs.model_dump(mode="json")}

    # ------------------------------------------------------------------ state machine
    def refresh(self, rec: SensorRecord, now: int) -> tuple[SensorState, SensorState, str] | None:
        new, reason = evaluate_state(rec, now, self.settings.thresholds.heartbeat)
        if new != rec.state:
            old = rec.state
            rec.state, rec.state_reason = new, reason
            return old, new, reason
        if reason != rec.state_reason:
            rec.state_reason = reason
        return None

    def refresh_all(self, now: int) -> list[tuple[SensorRecord, SensorState, SensorState, str]]:
        out = []
        for rec in self.records.values():
            t = self.refresh(rec, now)
            if t:
                out.append((rec, *t))
        return out

    def mark_disconnected(self, rec: SensorRecord, connection_id: int, reason: str):
        """Only the *current* connection may mark a sensor disconnected (a takeover keeps the new one)."""
        if rec.connection_id != connection_id or not rec.connected:
            return None
        rec.connected, rec.connection, rec.disconnect_reason = False, None, reason
        return self.refresh(rec, self.clock.now_ms())

    # ------------------------------------------------------------------ reporting
    def snapshot(self, rec: SensorRecord, now: int) -> dict:
        state, reason = evaluate_state(rec, now, self.settings.thresholds.heartbeat)
        rx = rec.last_obs_received_ms
        cap_age = None
        if rec.last_obs_captured_ms is not None and rec.offset.offset_ms is not None:
            cap_age = now - (rec.last_obs_captured_ms + rec.offset.offset_ms)
        return {
            "sensor_id": rec.sensor_id, "state": state.value, "state_reason": reason, "connected": rec.connected,
            "permission": rec.permission, "camera": rec.camera, "inference": rec.inference,
            "client_version": rec.client_version, "model_version": rec.model_version,
            "last_seen_age_ms": now - rec.last_seen_ms,
            "last_heartbeat_age_ms": now - rec.last_heartbeat_ms,
            "latest_observation_received_at_ms": rx,
            "latest_observation_age_ms": None if rx is None else now - rx,
            "latest_observation_captured_at_ms": rec.last_obs_captured_ms,
            "latest_observation_capture_age_ms": cap_age,
            "latest_observation_status": rec.last_obs_status, "hand": rec.last_obs_hand,
            "update_rate_hz": round(rec.rate.rate_hz(now), 2),
            "observations_accepted": rec.observations_accepted, "observations_rejected": rec.observations_rejected,
            "rejected_by_code": dict(rec.rejected_by_code), "sequence_gaps": rec.sequence_gaps,
            "image": None if rec.image_size is None else {"width": rec.image_size[0], "height": rec.image_size[1]},
            "reconnect_count": rec.reconnect_count, "heartbeat_count": rec.heartbeat_count,
            "clock_offset_ms": rec.offset.offset_ms, "health": rec.health,
            "status_detail": rec.status_detail, "last_error": rec.last_error,
        }

    def close(self) -> None:
        for rec in self.records.values():
            rec.connection = None
            rec.connected = False
            rec.disconnect_reason = rec.disconnect_reason or "session_stopped"
