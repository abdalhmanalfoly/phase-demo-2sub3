"""FastAPI application factory. Run with: ``uvicorn host.app.main:app`` (see scripts/run_host.sh)."""
from __future__ import annotations

import asyncio
import contextlib
import mimetypes
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import HOST_VERSION, Settings, load_settings
from .schemas.messages import PROTOCOL_VERSION
from .services.clock import SystemClock
from .services.logging_service import EventLogger
from .session_manager import SessionLimitError, SessionManager
from .spatial.calibration_store import CalibrationLoadError
from .websocket_handler import handle_sensor_socket

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")


class BindRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensor_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    camera_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,32}$")     # null = unbind


class InvalidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,32}$")
    sensor_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,32}$")
    reason: str = Field(default="camera moved or replaced", min_length=1, max_length=200)


def create_app(settings: Settings | None = None, clock=None, console_log: bool = True) -> FastAPI:
    settings = settings or load_settings()
    clock = clock or SystemClock()
    events = EventLogger(settings.log_dir, clock, settings.logging.recent_events_buffer, console=console_log)
    manager = SessionManager(settings, clock, events)

    async def sweeper() -> None:
        while True:
            await asyncio.sleep(settings.session.sweep_interval_s)
            try:
                manager.sweep()
            except Exception as exc:                       # never let the sweeper die silently
                events.log("sweeper_error", severity="error", reason=type(exc).__name__)

    async def fusion_ticker() -> None:
        while True:
            await asyncio.sleep(settings.fusion.spatial.tick_ms / 1000.0)
            manager.fusion_tick()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(sweeper()) if settings.sweeper_enabled else None
        ftask = asyncio.create_task(fusion_ticker()) if settings.sweeper_enabled and settings.fusion.spatial.enabled else None
        events.log("host_started", reason=f"v{HOST_VERSION} protocol {PROTOCOL_VERSION}")
        yield
        for t in (task, ftask):
            if t:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        await manager.shutdown()
        events.log("host_stopped")
        events.close()

    app = FastAPI(title="SAGA 3D host", version=HOST_VERSION, lifespan=lifespan)
    app.state.settings, app.state.manager, app.state.events, app.state.clock = settings, manager, events, clock

    def check_origin(request: Request) -> None:
        allowed = settings.server.allowed_origins
        origin = request.headers.get("origin")
        if allowed and origin is not None and origin not in allowed:
            events.log("http_origin_rejected", severity="warning", reason="origin_not_allowed", origin=origin)
            raise HTTPException(403, "origin not allowed")

    def must_get(session_id: str):
        if not _SESSION_RE.match(session_id) or manager.get(session_id) is None:
            raise HTTPException(404, "unknown session")
        return manager.get(session_id)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "host_version": HOST_VERSION, "protocol_version": PROTOCOL_VERSION,
                "time_ms": clock.now_ms(),
                "sessions_active": sum(1 for s in manager.sessions.values() if s.state == "active")}

    @app.get("/api/client-config")
    async def client_config():
        return settings.client_config()

    @app.post("/api/sessions", status_code=201)
    async def create_session(request: Request):
        check_origin(request)
        try:
            s = manager.create_session()
        except SessionLimitError as exc:
            return JSONResponse({"error": "SESSION_LIMIT", "detail": str(exc)}, status_code=429)
        return {"session_id": s.session_id, "state": s.state, "created_at_ms": s.created_at_ms,
                "join_path": f"/client/?session={s.session_id}", "ws_path": "/ws/sensor"}

    @app.get("/api/sessions")
    async def list_sessions():
        return {"sessions": manager.list_status()}

    @app.get("/api/sessions/{session_id}")
    async def session_status(session_id: str):
        return must_get(session_id).status()

    @app.post("/api/sessions/{session_id}/stop")
    async def stop_session(session_id: str, request: Request):
        check_origin(request)
        must_get(session_id)
        session = await manager.stop_session(session_id, "operator_request")
        return session.status()

    def fusion_of(session_id: str):
        session = must_get(session_id)
        if session.fusion is None:
            raise HTTPException(404, "spatial fusion is disabled")
        return session, session.fusion

    @app.get("/api/sessions/{session_id}/fusion")
    async def session_fusion(session_id: str, recent: int = 20):
        session, fusion = fusion_of(session_id)
        return fusion.details(list(session.registry.records), recent=max(0, min(recent, 200)))

    @app.get("/api/sessions/{session_id}/calibration")
    async def session_calibration(session_id: str):
        session, fusion = fusion_of(session_id)
        return fusion.calibration_status(list(session.registry.records))

    @app.post("/api/sessions/{session_id}/calibration/bindings")
    async def bind_sensor(session_id: str, body: BindRequest, request: Request):
        check_origin(request)
        session, fusion = fusion_of(session_id)
        if body.sensor_id not in session.registry.records:
            raise HTTPException(404, "unknown sensor in this session")
        fusion.bind(body.sensor_id, body.camera_id)
        return fusion.calibration_status(list(session.registry.records))

    @app.post("/api/sessions/{session_id}/calibration/invalidate")
    async def invalidate_calibration(session_id: str, body: InvalidateRequest, request: Request):
        check_origin(request)
        session, fusion = fusion_of(session_id)
        camera_id = body.camera_id or (fusion.registry.camera_for(body.sensor_id) if body.sensor_id else None)
        if camera_id is None:
            raise HTTPException(422, "give camera_id, or a sensor_id that is bound to a camera")
        try:
            fusion.invalidate(camera_id, body.reason)
        except CalibrationLoadError as exc:
            raise HTTPException(404, f"cannot invalidate: {exc.code}") from None
        return fusion.calibration_status(list(session.registry.records))

    @app.post("/api/sessions/{session_id}/calibration/reload")
    async def reload_calibration(session_id: str, request: Request):
        check_origin(request)
        session, fusion = fusion_of(session_id)
        fusion.reload()
        return fusion.calibration_status(list(session.registry.records))

    @app.websocket("/ws/sensor")
    async def sensor_ws(ws: WebSocket):
        await handle_sensor_socket(ws, manager)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/client/")

    app.mount("/client", StaticFiles(directory=settings.client_dir, html=True), name="client")
    return app


app = create_app()
