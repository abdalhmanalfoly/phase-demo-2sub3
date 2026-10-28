import json

import pytest
from starlette.websockets import WebSocketDisconnect

from .conftest import recv, register, send


def test_health_and_client_config(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["protocol_version"] == "1.0"
    cfg = client.get("/api/client-config").json()
    assert cfg["ws_path"] == "/ws/sensor" and "wrist" in cfg["vision"]["required_landmarks"]


def test_create_status_list(client):
    r = client.post("/api/sessions")
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert r.json()["join_path"].endswith(sid)
    st = client.get(f"/api/sessions/{sid}").json()
    assert st["state"] == "active" and st["sensors"] == []
    assert sid in [s["session_id"] for s in client.get("/api/sessions").json()["sessions"]]


def test_session_ids_are_unique(client):
    ids = {client.post("/api/sessions").json()["session_id"] for _ in range(5)}
    assert len(ids) == 5


def test_unknown_or_malformed_session_ids_404(client):
    assert client.get("/api/sessions/sess_nope0000").status_code == 404
    assert client.get("/api/sessions/..%2Fetc").status_code == 404
    assert client.post("/api/sessions/sess_nope0000/stop").status_code == 404


def test_session_limit(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    app = create_app(make_settings(**{"session.max_sessions": 2}), clock, console_log=False)
    with TestClient(app) as c:
        assert c.post("/api/sessions").status_code == 201
        assert c.post("/api/sessions").status_code == 201
        r = c.post("/api/sessions")
        assert r.status_code == 429 and r.json()["error"] == "SESSION_LIMIT"


def test_stop_notifies_sensors_closes_sockets_and_cleans_up(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        r = client.post(f"/api/sessions/{session_id}/stop")
        assert r.json()["state"] == "stopped" and r.json()["stop_reason"] == "operator_request"
        stop = recv(ws)
        assert stop["message_type"] == "SESSION_STOP" and stop["reason"] == "operator_request"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    st = client.get(f"/api/sessions/{session_id}").json()
    assert st["state"] == "stopped"
    assert st["sensors"][0]["state"] == "disconnected" and st["sensors"][0]["state_reason"] == "session_stopped"
    assert st["sensors_connected"] == 0


def test_stop_is_idempotent(client, session_id):
    assert client.post(f"/api/sessions/{session_id}/stop").json()["state"] == "stopped"
    assert client.post(f"/api/sessions/{session_id}/stop").json()["state"] == "stopped"


def test_cannot_join_a_stopped_session(client, session_id):
    client.post(f"/api/sessions/{session_id}/stop")
    from .conftest import hello
    with client.websocket_connect("/ws/sensor") as ws:
        send(ws, hello(session_id))
        err = recv(ws)
        assert err["error_code"] == "SESSION_STOPPED" and err["fatal"] is True


def test_host_shutdown_stops_active_sessions(app, clock):
    import asyncio
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        sid = c.post("/api/sessions").json()["session_id"]
        with c.websocket_connect("/ws/sensor") as ws:
            register(ws, sid)
    # lifespan exit ran manager.shutdown()
    assert app.state.manager.get(sid).state == "stopped"
    assert app.state.manager.get(sid).stop_reason == "host_shutdown"


def test_stopped_sessions_are_pruned(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    app = create_app(make_settings(**{"session.retain_stopped_sessions": 1}), clock, console_log=False)
    with TestClient(app) as c:
        ids = []
        for _ in range(3):
            sid = c.post("/api/sessions").json()["session_id"]
            clock.advance(10)
            c.post(f"/api/sessions/{sid}/stop")
            ids.append(sid)
        remaining = [s["session_id"] for s in c.get("/api/sessions").json()["sessions"]]
        assert remaining == [ids[-1]]


def test_origin_policy_on_http_and_websocket(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    app = create_app(make_settings(**{"server.allowed_origins": ["https://ok.example"]}), clock, console_log=False)
    with TestClient(app) as c:
        assert c.post("/api/sessions", headers={"Origin": "https://evil.example"}).status_code == 403
        assert c.post("/api/sessions", headers={"Origin": "https://ok.example"}).status_code == 201
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/sensor", headers={"Origin": "https://evil.example"}):
                pass
