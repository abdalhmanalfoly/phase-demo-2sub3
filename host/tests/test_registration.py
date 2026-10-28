import json
import time

import pytest
from starlette.websockets import WebSocketDisconnect

from .conftest import heartbeat, hello, obs, recv, register, send


def test_two_clients_get_distinct_sensor_ids(client, session_id):
    with client.websocket_connect("/ws/sensor") as a, client.websocket_connect("/ws/sensor") as b:
        ra = register(a, session_id, client_id="client_aaaaaaaa")
        rb = register(b, session_id, client_id="client_bbbbbbbb")
        assert ra["sensor_id"] != rb["sensor_id"]
        assert ra["resumed"] is False and rb["resumed"] is False
        assert ra["heartbeat_interval_ms"] == 2000 and ra["required_landmarks"] == ["wrist", "thumb_tip", "index_tip"]
        st = client.get(f"/api/sessions/{session_id}").json()
        assert st["sensors_connected"] == 2
        assert {s["sensor_id"] for s in st["sensors"]} == {ra["sensor_id"], rb["sensor_id"]}


def test_requested_sensor_id_is_honoured_when_free(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        assert register(ws, session_id, requested="phone_left")["sensor_id"] == "phone_left"


def test_duplicate_requested_id_from_other_client_rejected(client, session_id):
    with client.websocket_connect("/ws/sensor") as a, client.websocket_connect("/ws/sensor") as b:
        register(a, session_id, client_id="client_aaaaaaaa", requested="phone_left")
        send(b, hello(session_id, client_id="client_bbbbbbbb", requested="phone_left"))
        err = recv(b)
        assert err["error_code"] == "DUPLICATE_SENSOR_ID" and err["fatal"] is True
        with pytest.raises(WebSocketDisconnect):
            b.receive_text()
        # the original sensor is unaffected
        assert client.get(f"/api/sessions/{session_id}").json()["sensors_connected"] == 1


def test_same_client_reconnect_resumes_identity_and_replaces_old_socket(client, session_id):
    with client.websocket_connect("/ws/sensor") as old:
        first = register(old, session_id)
        with client.websocket_connect("/ws/sensor") as new:
            second = register(new, session_id)               # old socket still open (half-open drop)
            assert second["sensor_id"] == first["sensor_id"] and second["resumed"] is True
            with pytest.raises(WebSocketDisconnect) as e:
                old.receive_text()
            assert e.value.code == 4001
            st = client.get(f"/api/sessions/{session_id}").json()["sensors"][0]
            assert st["connected"] is True and st["state"] == "online" and st["reconnect_count"] == 1


def test_client_cannot_register_under_a_second_id(client, session_id):
    with client.websocket_connect("/ws/sensor") as a:
        register(a, session_id, requested="first_id")
        with client.websocket_connect("/ws/sensor") as b:
            send(b, hello(session_id, requested="second_id"))
            assert recv(b)["error_code"] == "SENSOR_ID_MISMATCH"


def test_second_hello_on_same_socket_is_rejected_without_side_effects(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        send(ws, hello(session_id, client_id="client_cccccccc"))
        assert recv(ws)["error_code"] == "ALREADY_REGISTERED"
        assert len(client.get(f"/api/sessions/{session_id}").json()["sensors"]) == 1


def test_unknown_session_is_fatal(client):
    with client.websocket_connect("/ws/sensor") as ws:
        send(ws, hello("sess_doesnotexist"))
        err = recv(ws)
        assert err["error_code"] == "UNKNOWN_SESSION" and err["fatal"] is True
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_session_full(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    app = create_app(make_settings(**{"session.max_sensors": 1}), clock, console_log=False)
    with TestClient(app) as c:
        sid = c.post("/api/sessions").json()["session_id"]
        with c.websocket_connect("/ws/sensor") as a, c.websocket_connect("/ws/sensor") as b:
            register(a, sid, client_id="client_aaaaaaaa")
            send(b, hello(sid, client_id="client_bbbbbbbb"))
            assert recv(b)["error_code"] == "SESSION_FULL"


def test_slot_frees_after_disconnect(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    app = create_app(make_settings(**{"session.max_sensors": 1}), clock, console_log=False)
    with TestClient(app) as c:
        sid = c.post("/api/sessions").json()["session_id"]
        with c.websocket_connect("/ws/sensor") as a:
            register(a, sid, client_id="client_aaaaaaaa")
        with c.websocket_connect("/ws/sensor") as b:
            assert register(b, sid, client_id="client_bbbbbbbb")["sensor_id"] != "sensor_01"   # ids never reused


def test_message_before_hello_rejected(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        send(ws, obs(session_id, "sensor_01", 0))
        assert recv(ws)["error_code"] == "NOT_REGISTERED"


def test_hello_timeout_closes_silent_socket(client):
    with client.websocket_connect("/ws/sensor") as ws:
        t0 = time.monotonic()
        err = recv(ws)
        assert err["error_code"] == "HELLO_TIMEOUT" and err["fatal"] is True
        assert time.monotonic() - t0 < 3
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_malformed_and_unsupported_inputs_get_explicit_errors_and_keep_socket_open(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        send(ws, "{not json")
        assert recv(ws)["error_code"] == "MALFORMED_JSON"
        send(ws, '{"a": NaN}')
        assert recv(ws)["error_code"] == "MALFORMED_JSON"                       # NaN literal refused
        send(ws, json.dumps({"protocol_version": "9", "message_type": "HELLO"}))
        assert recv(ws)["error_code"] == "UNSUPPORTED_PROTOCOL_VERSION"
        ws.send_bytes(b"\x00\x01")
        assert recv(ws)["error_code"] == "BINARY_NOT_SUPPORTED"
        send(ws, "x" * 20000)
        assert recv(ws)["error_code"] == "MESSAGE_TOO_LARGE"
        register(ws, session_id)                                               # still usable afterwards


def test_disconnect_message_marks_sensor_disconnected_with_reason(client, session_id):
    from .conftest import T0
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        send(ws, {"protocol_version": "1.0", "message_type": "DISCONNECT", "session_id": session_id,
                  "sensor_id": sid, "client_time_ms": T0, "reason": "user_stop"})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    s = client.get(f"/api/sessions/{session_id}").json()["sensors"][0]
    assert s["state"] == "disconnected" and s["state_reason"] == "client_user_stop"
