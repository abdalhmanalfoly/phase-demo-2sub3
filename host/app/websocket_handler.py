"""Sensor WebSocket endpoint: framing, validation, registration, dispatch, disconnect handling."""
from __future__ import annotations

import asyncio
import itertools
import json
import time

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .config import HOST_VERSION
from .schemas.messages import (
    PROTOCOL_VERSION, Disconnect, ErrorMessage, Heartbeat, HeartbeatAck, Hello, LandmarkObservation,
    ProtocolError, Registered, SensorStatus, parse_client_message,
)
from .session_manager import Session, SessionManager

_conn_ids = itertools.count(1)


def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name}")


class SensorConnection:
    """One accepted WebSocket plus the identity bound to it after HELLO."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.connection_id = next(_conn_ids)
        self.session: Session | None = None
        self.record = None
        self.error_sent_at: dict[str, int] = {}
        self.departure_reason: str | None = None

    async def send_json(self, payload: dict) -> bool:
        try:
            await self.ws.send_text(json.dumps(payload, separators=(",", ":")))
            return True
        except Exception:
            return False

    async def close(self, code: int = 1000, reason: str = "") -> None:
        try:
            await self.ws.close(code=code, reason=reason)
        except Exception:
            pass


async def _reply_error(conn: SensorConnection, mgr: SessionManager, err: ProtocolError, session_id: str | None) -> None:
    now = mgr.clock.now_ms()
    sensor_id = conn.record.sensor_id if conn.record else None
    session = conn.session or (mgr.get(session_id) if session_id else None)
    if session is not None:
        session.note_rejection(sensor_id, err)
    mgr.events.log("message_rejected", session_id=session.session_id if session else None, sensor_id=sensor_id,
                   reason=err.code, severity="warning", detail=err.message, related=err.related, fatal=err.fatal)
    last = conn.error_sent_at.get(err.code)
    if not err.fatal and last is not None and now - last < mgr.settings.limits.error_throttle_ms:
        return                                       # logged + counted, reply throttled
    conn.error_sent_at[err.code] = now
    msg = ErrorMessage(
        protocol_version=PROTOCOL_VERSION, message_type="ERROR",
        session_id=session.session_id if session else None, error_code=err.code, message=err.message[:300],
        related_message_type=err.related, details=err.details, fatal=err.fatal, host_time_ms=now,
    )
    await conn.send_json(msg.model_dump(mode="json"))


async def _handle_hello(conn: SensorConnection, mgr: SessionManager, hello: Hello) -> None:
    if conn.record is not None:
        raise ProtocolError("ALREADY_REGISTERED", "this socket is already registered", related="HELLO")
    session = mgr.get(hello.session_id)
    if session is None:
        raise ProtocolError("UNKNOWN_SESSION", "no such session", fatal=True, related="HELLO")
    if session.state != "active":
        raise ProtocolError("SESSION_STOPPED", "session is no longer active", fatal=True, related="HELLO")
    rec, resumed, previous = session.registry.register(hello, conn, conn.connection_id)
    conn.session, conn.record = session, rec
    if previous is not None and previous is not conn:
        await previous.close(4001, "replaced by a newer connection")
    vision = mgr.settings.thresholds.vision
    reply = Registered(
        protocol_version=PROTOCOL_VERSION, message_type="REGISTERED", session_id=session.session_id,
        sensor_id=rec.sensor_id, resumed=resumed,
        heartbeat_interval_ms=mgr.settings.thresholds.heartbeat.interval_ms,
        target_update_hz=min(vision.target_update_hz, hello.capabilities.max_update_hz),
        max_update_hz=vision.max_update_hz, required_landmarks=vision.required_landmarks,  # type: ignore[arg-type]
        session_state="active", host_time_ms=mgr.clock.now_ms(), host_version=HOST_VERSION,
    )
    await conn.send_json(reply.model_dump(mode="json"))
    mgr.events.log("sensor_resumed" if resumed else "sensor_registered", session_id=session.session_id,
                   sensor_id=rec.sensor_id, reason="reconnect" if resumed else "hello",
                   client_version=hello.client_version, model_version=hello.model_version)


async def _handle_bound(conn: SensorConnection, mgr: SessionManager, msg) -> bool:
    """Handle any non-HELLO message from a registered socket. Returns False to close the socket."""
    session, rec = conn.session, conn.record
    if session.state != "active":
        raise ProtocolError("SESSION_STOPPED", "session is no longer active", fatal=True, related=msg.message_type)
    if msg.session_id != session.session_id:
        raise ProtocolError("SESSION_MISMATCH", "message session_id does not match this connection",
                            related=msg.message_type)
    if msg.sensor_id != rec.sensor_id:
        raise ProtocolError("SENSOR_ID_MISMATCH", "message sensor_id does not match this connection",
                            related=msg.message_type)
    now = mgr.clock.now_ms()
    if isinstance(msg, LandmarkObservation):
        entry = session.registry.accept_observation(rec, msg, now)
        session.recorder.write(entry)
        if session.fusion is not None:
            try:
                session.fusion.on_observation(entry, rec)
            except Exception as exc:                           # a fusion bug must never reject/kill sensor ingestion
                mgr.events.log("fusion_error", session_id=session.session_id, sensor_id=rec.sensor_id,
                               severity="error", reason=type(exc).__name__, detail=str(exc)[:200])
    elif isinstance(msg, Heartbeat):
        session.registry.record_heartbeat(rec, msg, now)
        transition = session.registry.refresh(rec, now)
        if transition:
            mgr.events.log("sensor_state_changed", session_id=session.session_id, sensor_id=rec.sensor_id,
                           reason=transition[2], from_state=transition[0].value, to_state=transition[1].value)
        ack = HeartbeatAck(
            protocol_version=PROTOCOL_VERSION, message_type="HEARTBEAT_ACK", session_id=session.session_id,
            sensor_id=rec.sensor_id, sequence_number=msg.sequence_number, client_time_ms=msg.client_time_ms,
            host_time_ms=now, sensor_state=rec.state.value,  # type: ignore[arg-type]
        )
        await conn.send_json(ack.model_dump(mode="json"))
    elif isinstance(msg, SensorStatus):
        if session.registry.record_status(rec, msg, now):
            mgr.events.log("sensor_status_changed", session_id=session.session_id, sensor_id=rec.sensor_id,
                           reason=msg.detail, permission=msg.permission, camera=msg.camera, inference=msg.inference)
        transition = session.registry.refresh(rec, now)
        if transition:
            mgr.events.log("sensor_state_changed", session_id=session.session_id, sensor_id=rec.sensor_id,
                           reason=transition[2], from_state=transition[0].value, to_state=transition[1].value)
    elif isinstance(msg, Disconnect):
        conn.departure_reason = f"client_{msg.reason}"
        return False
    return True


async def _process(conn: SensorConnection, mgr: SessionManager, text: str) -> bool:
    session_hint = None
    try:
        if len(text.encode("utf-8")) > mgr.settings.limits.max_message_bytes:
            raise ProtocolError("MESSAGE_TOO_LARGE",
                                f"message exceeds {mgr.settings.limits.max_message_bytes} bytes")
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            raise ProtocolError("MALFORMED_JSON", "message is not valid JSON") from None
        if isinstance(data, dict) and isinstance(data.get("session_id"), str):
            session_hint = data["session_id"][:64]
        msg = parse_client_message(data)
        if isinstance(msg, Hello):
            await _handle_hello(conn, mgr, msg)
            return True
        if conn.record is None:
            raise ProtocolError("NOT_REGISTERED", "send HELLO before any other message", related=msg.message_type)
        return await _handle_bound(conn, mgr, msg)
    except ProtocolError as err:
        await _reply_error(conn, mgr, err, session_hint)
        if err.fatal:
            await conn.close(1008, err.code)
        return not err.fatal


async def handle_sensor_socket(ws: WebSocket, mgr: SessionManager) -> None:
    cfg = mgr.settings
    origin = ws.headers.get("origin")
    if cfg.server.allowed_origins and origin not in cfg.server.allowed_origins:
        mgr.events.log("ws_origin_rejected", severity="warning", reason="origin_not_allowed", origin=origin)
        await ws.close(code=1008)
        return
    await ws.accept()
    conn = SensorConnection(ws)
    deadline = time.monotonic() + cfg.session.hello_timeout_s
    try:
        while True:
            timeout = None if conn.record else max(0.0, deadline - time.monotonic())
            try:
                event = await asyncio.wait_for(ws.receive(), timeout)
            except asyncio.TimeoutError:
                await _reply_error(conn, mgr, ProtocolError("HELLO_TIMEOUT", "no HELLO received in time", fatal=True), None)
                await conn.close(1008, "HELLO_TIMEOUT")
                break
            if event["type"] == "websocket.disconnect":
                break
            text = event.get("text")
            if text is None:
                await _reply_error(conn, mgr, ProtocolError("BINARY_NOT_SUPPORTED",
                                                            "only UTF-8 JSON text frames are supported"), None)
                continue
            if not await _process(conn, mgr, text):
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rec, session = conn.record, conn.session
        if rec is not None and session is not None and rec.connection is conn:
            transition = session.registry.mark_disconnected(rec, conn.connection_id, conn.departure_reason or "socket_closed")
            mgr.events.log("sensor_disconnected", session_id=session.session_id, sensor_id=rec.sensor_id,
                           reason=conn.departure_reason or "socket_closed")
        await conn.close(1000)
