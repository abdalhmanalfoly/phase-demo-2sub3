"""Session lifecycle: create, status, orderly shutdown and cleanup."""
from __future__ import annotations

import collections
import secrets
from typing import Any

from .config import HOST_VERSION, Settings
from .schemas.messages import PROTOCOL_VERSION, ProtocolError, SessionStop
from .sensor_registry import SensorRegistry
from .services.logging_service import EventLogger
from .services.observation_recorder import ObservationRecorder
from .spatial.service import FusionService


class SessionLimitError(Exception):
    pass


class Session:
    def __init__(self, session_id: str, settings: Settings, clock, events=None):
        self.session_id = session_id
        self.settings = settings
        self.clock = clock
        self.created_at_ms = clock.now_ms()
        self.state = "active"                       # active -> stopping -> stopped
        self.stopped_at_ms: int | None = None
        self.stop_reason: str | None = None
        self.registry = SensorRegistry(session_id, settings, clock)
        self.recorder = ObservationRecorder(settings.log_dir, session_id, settings.logging.save_landmark_events)
        self.fusion: FusionService | None = (FusionService(session_id, settings, clock, events)
                                             if settings.fusion.spatial.enabled and events is not None else None)
        self.recent_errors: collections.deque[dict] = collections.deque(maxlen=50)
        self.rejected_by_code: collections.Counter = collections.Counter()
        self.messages_rejected = 0

    def note_rejection(self, sensor_id: str | None, err: ProtocolError) -> dict:
        now = self.clock.now_ms()
        entry = {"ts_ms": now, "sensor_id": sensor_id, "error_code": err.code, "message": err.message}
        self.messages_rejected += 1
        self.rejected_by_code[err.code] += 1
        self.recent_errors.append(entry)
        if sensor_id and sensor_id in self.registry.records:
            rec = self.registry.records[sensor_id]
            rec.observations_rejected += 1
            rec.rejected_by_code[err.code] += 1
            rec.last_error = entry
        return entry

    def status(self) -> dict:
        now = self.clock.now_ms()
        sensors = [self.registry.snapshot(r, now) for r in self.registry.records.values()]
        return {
            "session_id": self.session_id, "state": self.state, "created_at_ms": self.created_at_ms,
            "stopped_at_ms": self.stopped_at_ms, "stop_reason": self.stop_reason, "host_time_ms": now,
            "protocol_version": PROTOCOL_VERSION, "host_version": HOST_VERSION,
            "sensors": sensors,
            "sensors_connected": sum(1 for s in sensors if s["connected"]),
            "messages_rejected": self.messages_rejected, "rejected_by_code": dict(self.rejected_by_code),
            "recent_errors": list(self.recent_errors)[-10:],
            "landmark_log": str(self.recorder.path) if self.recorder.enabled and self.recorder.lines_written else None,
            "fusion": self.fusion.summary([s["sensor_id"] for s in sensors]) if self.fusion else None,
        }

    def cleanup(self) -> None:
        self.registry.close()
        self.recorder.close()
        if self.fusion:
            self.fusion.close()


class SessionManager:
    def __init__(self, settings: Settings, clock, events: EventLogger):
        self.settings, self.clock, self.events = settings, clock, events
        self.sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------ lifecycle
    def create_session(self) -> Session:
        active = sum(1 for s in self.sessions.values() if s.state == "active")
        if active >= self.settings.session.max_sessions:
            raise SessionLimitError("maximum number of active sessions reached")
        while True:
            sid = "sess_" + secrets.token_hex(4)
            if sid not in self.sessions:
                break
        session = Session(sid, self.settings, self.clock, self.events)
        self.sessions[sid] = session
        self.events.log("session_created", session_id=sid)
        return session

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def list_status(self) -> list[dict]:
        return [s.status() for s in self.sessions.values()]

    async def stop_session(self, session_id: str, reason: str = "operator_request") -> Session | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        if session.state != "active":
            return session                                   # idempotent
        session.state = "stopping"
        self.events.log("session_stopping", session_id=session_id, reason=reason)
        stop = SessionStop(protocol_version=PROTOCOL_VERSION, message_type="SESSION_STOP", session_id=session_id,
                           reason="host_shutdown" if reason == "host_shutdown" else "operator_request",
                           host_time_ms=self.clock.now_ms()).model_dump(mode="json")
        for rec in list(session.registry.records.values()):
            conn: Any = rec.connection
            if rec.connected and conn is not None:
                await conn.send_json(stop)
                await conn.close(1000, "session stopped")
                session.registry.mark_disconnected(rec, rec.connection_id, "session_stopped")
        session.cleanup()
        session.state, session.stop_reason, session.stopped_at_ms = "stopped", reason, self.clock.now_ms()
        self.events.log("session_stopped", session_id=session_id, reason=reason)
        self._prune_stopped()
        return session

    def _prune_stopped(self) -> None:
        stopped = sorted((s for s in self.sessions.values() if s.state == "stopped"), key=lambda s: s.stopped_at_ms or 0)
        for s in stopped[: max(0, len(stopped) - self.settings.session.retain_stopped_sessions)]:
            del self.sessions[s.session_id]

    async def shutdown(self) -> None:
        for sid in [s.session_id for s in self.sessions.values() if s.state == "active"]:
            await self.stop_session(sid, "host_shutdown")

    # ------------------------------------------------------------------ fusion tick (tracking-loss timing)
    def fusion_tick(self) -> None:
        for session in self.sessions.values():
            if session.state == "active" and session.fusion is not None:
                try:
                    session.fusion.tick()
                except Exception as exc:                       # never let fusion break the host
                    self.events.log("fusion_error", session_id=session.session_id, severity="error", reason=type(exc).__name__)

    # ------------------------------------------------------------------ periodic health evaluation
    def sweep(self) -> None:
        now = self.clock.now_ms()
        for session in self.sessions.values():
            if session.state != "active":
                continue
            for rec, old, new, reason in session.registry.refresh_all(now):
                self.events.log("sensor_state_changed", session_id=session.session_id, sensor_id=rec.sensor_id,
                                reason=reason, severity="info" if new.value == "online" else "warning",
                                from_state=old.value, to_state=new.value)
            session.recorder.flush()
