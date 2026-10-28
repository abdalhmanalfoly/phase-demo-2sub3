import json

import pytest

from .conftest import T0, heartbeat, lm, obs, recv, register, send


@pytest.fixture
def sensor(client, session_id):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        yield ws, sid


def status_of(client, session_id):
    return client.get(f"/api/sessions/{session_id}").json()


def test_valid_observation_is_accepted_recorded_and_receive_time_is_host_assigned(client, session_id, sensor, clock, settings):
    ws, sid = sensor
    clock.advance(40)
    send(ws, obs(session_id, sid, 0, t=T0 + 10))
    s = status_of(client, session_id)["sensors"][0]
    assert s["observations_accepted"] == 1 and s["observations_rejected"] == 0
    assert s["latest_observation_received_at_ms"] == clock.now_ms()            # host clock, not client's
    assert s["latest_observation_captured_at_ms"] == T0 + 10                  # sensor clock kept separately
    assert s["latest_observation_status"] == "tracked" and s["image"] == {"width": 640, "height": 480}
    client.post(f"/api/sessions/{session_id}/stop")                            # flush + close recorder
    line = json.loads((settings.log_dir / "sessions" / session_id / "observations.jsonl").read_text().splitlines()[0])
    assert line["received_at_ms"] == T0 + 40 and line["captured_at_ms"] == T0 + 10 and line["sensor_id"] == sid


def test_no_hand_observation_is_accepted(client, session_id, sensor):
    ws, sid = sensor
    send(ws, obs(session_id, sid, 0, status="no_hand", hand="unknown", landmarks=[]))
    assert status_of(client, session_id)["sensors"][0]["latest_observation_status"] == "no_hand"


def test_invalid_observation_rejected_with_error_and_counted(client, session_id, sensor):
    ws, sid = sensor
    m = obs(session_id, sid, 0); m["landmarks"][0]["x_px"] = 5000
    send(ws, m)
    err = recv(ws)
    assert err["error_code"] == "SCHEMA_VALIDATION_FAILED" and err["fatal"] is False
    assert err["details"] and "5000" not in json.dumps(err)
    st = status_of(client, session_id)
    assert st["sensors"][0]["observations_rejected"] == 1 and st["sensors"][0]["observations_accepted"] == 0
    assert st["rejected_by_code"] == {"SCHEMA_VALIDATION_FAILED": 1}
    send(ws, obs(session_id, sid, 1))                                          # socket stays usable
    assert status_of(client, session_id)["sensors"][0]["observations_accepted"] == 1


def test_stale_and_duplicate_sequence_rejected(client, session_id, sensor, clock):
    ws, sid = sensor
    send(ws, obs(session_id, sid, 5))
    for seq in (5, 4):
        send(ws, obs(session_id, sid, seq))
        assert recv(ws)["error_code"] == "STALE_SEQUENCE"
    assert status_of(client, session_id)["sensors"][0]["observations_accepted"] == 1


def test_sequence_gaps_are_counted(client, session_id, sensor):
    ws, sid = sensor
    for seq in (0, 1, 5, 6):
        send(ws, obs(session_id, sid, seq))
    s = status_of(client, session_id)["sensors"][0]
    assert s["observations_accepted"] == 4 and s["sequence_gaps"] == 3


def test_wrong_session_and_wrong_sensor_rejected(client, session_id, sensor):
    ws, sid = sensor
    send(ws, obs("sess_other0001", sid, 0))
    assert recv(ws)["error_code"] == "SESSION_MISMATCH"
    send(ws, obs(session_id, "sensor_99", 0))
    assert recv(ws)["error_code"] == "SENSOR_ID_MISMATCH"
    send(ws, obs(session_id, "bad id!", 0))
    assert recv(ws)["error_code"] == "SCHEMA_VALIDATION_FAILED"
    assert status_of(client, session_id)["sensors"][0]["observations_accepted"] == 0


def test_observation_in_another_real_session_cannot_be_injected(client, session_id, sensor):
    other = client.post("/api/sessions").json()["session_id"]
    ws, sid = sensor
    send(ws, obs(other, sid, 0))
    assert recv(ws)["error_code"] == "SESSION_MISMATCH"
    assert status_of(client, other)["sensors"] == []


def test_stale_observation_by_age_uses_clock_offset(client, session_id, sensor, clock):
    ws, sid = sensor                                  # HELLO sample: offset 0 (client_time == host time)
    clock.advance(5000)
    send(ws, obs(session_id, sid, 0, t=T0 + 100))     # captured 4.9 s ago
    assert recv(ws)["error_code"] == "STALE_OBSERVATION"


def test_fresh_observation_from_a_skewed_phone_clock_is_still_accepted(client, session_id, clock, settings):
    """Phone clock 3 h ahead of the host: offset is learned from HELLO, so nothing is 'in the future'."""
    skew = 3 * 3600 * 1000
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id, t=T0 + skew)["sensor_id"]
        clock.advance(30)
        send(ws, obs(session_id, sid, 0, t=T0 + skew + 20))
        s = status_of(client, session_id)["sensors"][0]
        assert s["observations_accepted"] == 1 and s["clock_offset_ms"] == -skew


def test_timestamp_in_future_and_backwards_rejected(client, session_id, sensor):
    ws, sid = sensor
    send(ws, obs(session_id, sid, 0, t=T0 + 60_000))
    assert recv(ws)["error_code"] == "TIMESTAMP_IN_FUTURE"
    send(ws, obs(session_id, sid, 1, t=T0))
    send(ws, obs(session_id, sid, 2, t=T0 - 5))
    assert recv(ws)["error_code"] == "STALE_OBSERVATION"


def test_host_rate_limit(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    app = create_app(make_settings(**{"limits.max_observation_hz_host": 5}), clock, console_log=False)
    with TestClient(app) as c:
        sid_ = c.post("/api/sessions").json()["session_id"]
        with c.websocket_connect("/ws/sensor") as ws:
            sid = register(ws, sid_)["sensor_id"]
            for i in range(5):
                send(ws, obs(sid_, sid, i))
            send(ws, obs(sid_, sid, 5))
            assert recv(ws)["error_code"] == "RATE_LIMITED"
            clock.advance(1100)                                                # window passes
            send(ws, obs(sid_, sid, 6, t=T0 + 1100))
            s = c.get(f"/api/sessions/{sid_}").json()["sensors"][0]
            assert s["observations_accepted"] == 6


def test_update_rate_is_measured(client, session_id, sensor, clock):
    ws, sid = sensor
    for i in range(30):                                # 15 Hz for 2 s of fake time
        send(ws, obs(session_id, sid, i, t=clock.now_ms()))
        clock.advance(66)
    rate = status_of(client, session_id)["sensors"][0]["update_rate_hz"]
    assert 13 <= rate <= 17, rate


def test_error_replies_are_throttled_but_every_rejection_is_counted(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    app = create_app(make_settings(**{"limits.error_throttle_ms": 1000}), clock, console_log=False)
    with TestClient(app) as c:
        sid_ = c.post("/api/sessions").json()["session_id"]
        with c.websocket_connect("/ws/sensor") as ws:
            sid = register(ws, sid_)["sensor_id"]
            for _ in range(10):
                send(ws, obs(sid_, sid, 0, status="no_hand"))        # schema failure x10 (no_hand + landmarks)
            assert recv(ws)["error_code"] == "SCHEMA_VALIDATION_FAILED"
            st = c.get(f"/api/sessions/{sid_}").json()
            assert st["messages_rejected"] == 10                      # all counted, only one reply sent
            clock.advance(1500)
            send(ws, obs(sid_, sid, 0, status="no_hand"))
            assert recv(ws)["error_code"] == "SCHEMA_VALIDATION_FAILED"
