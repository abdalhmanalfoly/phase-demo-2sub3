"""Heartbeat tracking and explicit online/degraded/offline/disconnected states (fake clock, no sleeping)."""
import pytest
from starlette.websockets import WebSocketDisconnect

from .conftest import T0, heartbeat, obs, recv, register, send, status


def sensor_status(client, session_id):
    return client.get(f"/api/sessions/{session_id}").json()["sensors"][0]


def test_heartbeat_is_acked_and_updates_last_seen(client, session_id, clock):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        clock.advance(1000)
        send(ws, heartbeat(session_id, sid, 0, t=clock.now_ms()))
        ack = recv(ws)
        assert ack["message_type"] == "HEARTBEAT_ACK" and ack["sequence_number"] == 0
        assert ack["client_time_ms"] == clock.now_ms() and ack["sensor_state"] == "online"
        s = sensor_status(client, session_id)
        assert s["last_heartbeat_age_ms"] == 0 and s["heartbeat_count"] == 1 and s["state"] == "online"


def test_stale_heartbeat_sequence_rejected(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        send(ws, heartbeat(session_id, sid, 3)); recv(ws)
        send(ws, heartbeat(session_id, sid, 3))
        assert recv(ws)["error_code"] == "STALE_SEQUENCE"


def test_online_degraded_offline_progression_and_recovery(client, session_id, manager, clock):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        assert sensor_status(client, session_id)["state"] == "online"
        clock.advance(4600)                                         # > degraded_after (4500)
        manager.sweep()
        s = sensor_status(client, session_id)
        assert s["state"] == "degraded" and s["state_reason"] == "heartbeat_late"
        clock.advance(4000)                                         # > offline_after (8000 total)
        manager.sweep()
        s = sensor_status(client, session_id)
        assert s["state"] == "offline" and s["state_reason"] == "heartbeat_timeout" and s["connected"] is True
        send(ws, heartbeat(session_id, sid, 0, t=clock.now_ms()))   # heartbeat returns -> back online
        recv(ws)
        assert sensor_status(client, session_id)["state"] == "online"


def test_state_is_correct_even_without_a_sweep(client, session_id, clock):
    with client.websocket_connect("/ws/sensor") as ws:
        register(ws, session_id)
        clock.advance(9000)
        assert sensor_status(client, session_id)["state"] == "offline"      # computed on read


def test_sweeper_logs_transitions(client, session_id, manager, clock, app):
    with client.websocket_connect("/ws/sensor") as ws:
        register(ws, session_id)
        clock.advance(9000)
        manager.sweep()
        ev = [e for e in app.state.events.recent if e["event_type"] == "sensor_state_changed"]
        assert ev and ev[-1]["to_state"] == "offline" and ev[-1]["reason"] == "heartbeat_timeout"
        assert ev[-1]["session_id"] == session_id and ev[-1]["sensor_id"] == "sensor_01" and ev[-1]["ts_ms"]


def test_self_reported_failures_degrade_the_sensor(client, session_id, manager):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        for kw, reason in [({"permission": "denied", "camera": "idle", "inference": "idle"}, "permission_denied"),
                           ({"permission": "granted", "camera": "unavailable", "inference": "idle"}, "camera_unavailable"),
                           ({"permission": "granted", "camera": "active", "inference": "error"}, "inference_error")]:
            send(ws, status(session_id, sid, **kw))
            s = sensor_status(client, session_id)
            assert s["state"] == "degraded" and s["state_reason"] == reason, s
        send(ws, status(session_id, sid))                           # healthy again
        s = sensor_status(client, session_id)
        assert s["state"] == "online" and s["camera"] == "active" and s["inference"] == "detecting"


def test_transitional_startup_states_are_not_degraded(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        send(ws, status(session_id, sid, permission="prompt", camera="starting", inference="idle"))
        send(ws, status(session_id, sid, permission="granted", camera="active", inference="loading"))
        assert sensor_status(client, session_id)["state"] == "online"


def test_socket_close_marks_disconnected_then_reconnect_resumes_with_fresh_sequences(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        send(ws, obs(session_id, sid, 10))
        send(ws, heartbeat(session_id, sid, 4)); recv(ws)
    s = sensor_status(client, session_id)
    assert s["state"] == "disconnected" and s["connected"] is False and s["state_reason"] == "socket_closed"
    with client.websocket_connect("/ws/sensor") as ws:                # same client_id by default
        again = register(ws, session_id)
        assert again["sensor_id"] == sid and again["resumed"] is True
        send(ws, obs(session_id, sid, 0))                              # new epoch: low sequence numbers are fine again
        send(ws, heartbeat(session_id, sid, 0)); assert recv(ws)["message_type"] == "HEARTBEAT_ACK"
        s = sensor_status(client, session_id)
        assert s["state"] == "online" and s["reconnect_count"] == 1 and s["observations_accepted"] == 2


def test_observations_do_not_keep_a_sensor_online_without_heartbeats(client, session_id, clock):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        for i in range(10):
            clock.advance(1000)
            send(ws, obs(session_id, sid, i, t=clock.now_ms()))
        assert sensor_status(client, session_id)["state"] == "offline"   # heartbeats are the liveness signal
