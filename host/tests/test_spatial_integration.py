"""Phase 2 through the real Phase 1 host: WebSocket sensors -> registry -> fusion service -> HTTP API."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from host.app.main import create_app
from host.app.spatial import synthetic as S
from host.app.spatial.calibration_store import CalibrationStore
from host.app.spatial.calibration_tool import accuracy_report
from .conftest import T0, FakeClock, recv, register, send

CAMS = S.rig_cameras()
X = np.array([0.25, 0.1, 0.6])
A, B = "phone_left", "phone_right"


@pytest.fixture
def fsettings(make_settings, tmp_path):
    s = make_settings()
    s.fusion.spatial.calibration_dir = str(tmp_path / "cal")
    s.fusion.spatial.allow_synthetic_calibration = True
    s.fusion.smoothing.enabled = False
    st = CalibrationStore(tmp_path / "cal", "default_rig")
    for c in CAMS.values():
        st.save_camera(S.artifact_from_camera(c))
    return s


@pytest.fixture
def fapp(fsettings, clock):
    return create_app(fsettings, clock, console_log=False)


@pytest.fixture
def fclient(fapp):
    with TestClient(fapp) as c:
        yield c


@pytest.fixture
def sid(fclient):
    return fclient.post("/api/sessions").json()["session_id"]


def wire_obs(session_id, sensor_id, cam, point, seq, t, status="tracked", hand="right", size=None):
    e = S.observation_entry(sensor_id, cam, point, seq, t, t, hand=hand, status=status)
    for k in ("received_at_ms", "estimated_age_ms"):
        e.pop(k)
    e["session_id"] = session_id
    if size:
        e["image_width"], e["image_height"] = size
    return e


def fusion(client, session_id):
    return client.get(f"/api/sessions/{session_id}/fusion").json()


def pump(ws_a, ws_b, ids, session_id, clock, n=3, point=X, **kw):
    for _ in range(n):
        t = clock.now_ms()
        for ws, (sensor, cam) in ((ws_a, ids[0]), (ws_b, ids[1])):
            send(ws, wire_obs(session_id, sensor, cam, point, pump.seq.setdefault(sensor, 0), t, **kw))
            pump.seq[sensor] += 1
        clock.advance(66)


pump.seq = {}


@pytest.fixture(autouse=True)
def _reset_seq():
    pump.seq = {}


def test_two_websocket_sensors_are_fused_into_a_3d_estimate(fclient, sid, clock):
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        ia = register(a, sid, client_id="client_aaaaaaaa", requested=A)["sensor_id"]
        ib = register(b, sid, client_id="client_bbbbbbbb", requested=B)["sensor_id"]
        assert (ia, ib) == (A, B)
        pump(a, b, ((A, CAMS[A]), (B, CAMS[B])), sid, clock)
        st = fclient.get(f"/api/sessions/{sid}").json()
        assert all(s["observations_accepted"] == 3 for s in st["sensors"])                    # Phase 1 behaviour intact
        r = fusion(fclient, sid)["latest"]["wrist"]
        assert r["status"] == "TRACKING_MULTI_VIEW" and r["valid"] and r["unit"] == "m"
        assert np.linalg.norm(np.array(r["world_xyz_m"]) - X) < 1e-6
        assert r["source_sensor_ids"] == [A, B] and r["session_id"] == sid and r["aggregate_reprojection_error_px"] < 1e-6
        assert st["fusion"]["latest"]["wrist"]["status"] == "TRACKING_MULTI_VIEW"              # also in the session summary


def test_default_sensor_ids_need_an_operator_binding(fclient, sid, clock):
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        ia = register(a, sid, client_id="client_aaaaaaaa")["sensor_id"]
        ib = register(b, sid, client_id="client_bbbbbbbb")["sensor_id"]
        assert (ia, ib) == ("sensor_01", "sensor_02")
        pump(a, b, ((ia, CAMS[A]), (ib, CAMS[B])), sid, clock)
        r = fusion(fclient, sid)["latest"]["wrist"]
        assert r["status"] == "CALIBRATION_REQUIRED" and "NOT_BOUND" in r["message"] and r["world_xyz_m"] is None
        assert fclient.post(f"/api/sessions/{sid}/calibration/bindings", json={"sensor_id": "sensor_09", "camera_id": A}).status_code == 404
        assert fclient.post(f"/api/sessions/{sid}/calibration/bindings", json={"sensor_id": ia, "camera_id": "bad id!"}).status_code == 422
        assert fclient.post(f"/api/sessions/{sid}/calibration/bindings", json={"sensor_id": ia, "camera_id": A, "x": 1}).status_code == 422
        for s_, c_ in ((ia, A), (ib, B)):
            assert fclient.post(f"/api/sessions/{sid}/calibration/bindings", json={"sensor_id": s_, "camera_id": c_}).status_code == 200
        pump(a, b, ((ia, CAMS[A]), (ib, CAMS[B])), sid, clock)
        r = fusion(fclient, sid)["latest"]["wrist"]
        assert r["status"] == "TRACKING_MULTI_VIEW" and np.linalg.norm(np.array(r["world_xyz_m"]) - X) < 1e-6
        cal = fclient.get(f"/api/sessions/{sid}/calibration").json()
        assert cal["bindings"] == {ia: A, ib: B} and cal["sensors"][ia]["ok"] and cal["report"]["can_triangulate"]
        fclient.post(f"/api/sessions/{sid}/calibration/bindings", json={"sensor_id": ia, "camera_id": None})   # unbind
        assert not fclient.get(f"/api/sessions/{sid}/calibration").json()["sensors"][ia]["ok"]


def test_operator_invalidation_via_api_blocks_fusion_and_persists(fclient, sid, clock, fsettings):
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        register(a, sid, client_id="client_aaaaaaaa", requested=A); register(b, sid, client_id="client_bbbbbbbb", requested=B)
        ids = ((A, CAMS[A]), (B, CAMS[B]))
        pump(a, b, ids, sid, clock)
        assert fusion(fclient, sid)["latest"]["wrist"]["valid"]
        r = fclient.post(f"/api/sessions/{sid}/calibration/invalidate", json={"camera_id": A, "reason": "tripod bumped"})
        assert r.status_code == 200 and r.json()["sensors"][A]["problem"] == "INVALIDATED"
        pump(a, b, ids, sid, clock, n=4)
        latest = fusion(fclient, sid)["latest"]["wrist"]
        assert not latest["valid"] and latest["world_xyz_m"] is None and any("INVALIDATED" in w for w in latest["warnings"])
        on_disk = json.loads((fsettings.calibration_root / "default_rig" / f"{A}.json").read_text())
        assert on_disk["validity"]["status"] == "invalid" and "tripod" in on_disk["validity"]["reason"]
        assert fclient.post(f"/api/sessions/{sid}/calibration/invalidate", json={"camera_id": "ghost"}).status_code == 404
        assert fclient.post(f"/api/sessions/{sid}/calibration/invalidate", json={}).status_code == 422
        # recalibration (consistent rig session) + reload restores fusion
        st = CalibrationStore(fsettings.calibration_root, "default_rig")
        st.save_camera(S.artifact_from_camera(CAMS[A], calibration_id="left-recal"))
        assert fclient.post(f"/api/sessions/{sid}/calibration/reload").status_code == 200
        pump(a, b, ids, sid, clock)
        assert fusion(fclient, sid)["latest"]["wrist"]["valid"]


def test_phone_rotation_changes_image_size_and_is_refused(fclient, sid, clock):
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        register(a, sid, client_id="client_aaaaaaaa", requested=A); register(b, sid, client_id="client_bbbbbbbb", requested=B)
        send(a, wire_obs(sid, A, CAMS[A], X, 0, clock.now_ms(), size=(480, 640)))
        send(b, wire_obs(sid, B, CAMS[B], X, 0, clock.now_ms()))
        r = fusion(fclient, sid)["latest"]["wrist"]
        assert not r["valid"] and r["world_xyz_m"] is None
        assert any(s["note"] == "DIMENSION_MISMATCH" for s in r["sources"]) or "DIMENSION_MISMATCH" in " ".join(r["warnings"])


def test_a_phone_with_a_wrong_clock_still_fuses_because_phase1_estimates_the_offset(fclient, sid, clock):
    skew = 3600 * 1000
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        register(a, sid, client_id="client_aaaaaaaa", requested=A)
        register(b, sid, client_id="client_bbbbbbbb", requested=B, t=T0 + skew)           # phone B's clock is +1 h
        for i in range(3):
            t = clock.now_ms()
            send(a, wire_obs(sid, A, CAMS[A], X, i, t)); send(b, wire_obs(sid, B, CAMS[B], X, i, t + skew))
            clock.advance(66)
        r = fusion(fclient, sid)["latest"]["wrist"]
        assert r["status"] == "TRACKING_MULTI_VIEW" and r["skew_ms"] == 0.0
        assert fclient.get(f"/api/sessions/{sid}").json()["sensors"][1]["clock_offset_ms"] == -skew


def test_tracking_loss_is_driven_by_the_fusion_tick(fclient, fapp, sid, clock):
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        register(a, sid, client_id="client_aaaaaaaa", requested=A); register(b, sid, client_id="client_bbbbbbbb", requested=B)
        ids = ((A, CAMS[A]), (B, CAMS[B]))
        pump(a, b, ids, sid, clock)
        good = fusion(fclient, sid)["latest"]["wrist"]["world_xyz_m"]
        pump(a, b, ids, sid, clock, n=1, status="no_hand")
        r = fusion(fclient, sid)["latest"]["wrist"]
        assert r["status"] == "INSUFFICIENT_VIEWS" and r["world_xyz_m"] is None and r["held_world_xyz_m"] is not None
        clock.advance(700)
        fapp.state.manager.fusion_tick()
        r = fusion(fclient, sid)["latest"]["wrist"]
        assert r["status"] == "TARGET_LOST" and r["held_world_xyz_m"] is None and r["world_xyz_m"] is None


def test_a_fusion_crash_never_breaks_sensor_ingestion(fclient, fapp, sid, clock):
    with fclient.websocket_connect("/ws/sensor") as a:
        register(a, sid, client_id="client_aaaaaaaa", requested=A)
        fapp.state.manager.get(sid).fusion.engine.ingest = lambda *a, **k: 1 / 0
        send(a, wire_obs(sid, A, CAMS[A], X, 0, clock.now_ms()))
        send(a, wire_obs(sid, A, CAMS[A], X, 1, clock.now_ms()))
        st = fclient.get(f"/api/sessions/{sid}").json()
        assert st["sensors"][0]["observations_accepted"] == 2 and st["sensors"][0]["observations_rejected"] == 0
        errs = [e for e in fapp.state.events.recent if e["event_type"] == "fusion_error"]
        assert len(errs) == 2 and errs[0]["reason"] == "ZeroDivisionError"


def test_fusion_log_is_recorded_numeric_only_and_feeds_the_accuracy_report(fclient, sid, clock, fsettings):
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        register(a, sid, client_id="client_aaaaaaaa", requested=A); register(b, sid, client_id="client_bbbbbbbb", requested=B)
        pump(a, b, ((A, CAMS[A]), (B, CAMS[B])), sid, clock, n=10)
    fclient.post(f"/api/sessions/{sid}/stop")
    path = fsettings.log_dir / "sessions" / sid / "fusion.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 10 and all(l["message_type"] == "POSE_UPDATE" and l["valid"] for l in lines)
    rep = accuracy_report(path, X.tolist())
    assert rep["samples"] == 10 and rep["max_error_m"] < 1e-6 and rep["valid_fraction"] == 1.0
    assert "base64" not in path.read_text() and "image" not in json.dumps([k for l in lines for k in l])


def test_fusion_status_transitions_are_logged_as_events(fclient, fapp, sid, clock):
    with fclient.websocket_connect("/ws/sensor") as a, fclient.websocket_connect("/ws/sensor") as b:
        register(a, sid, client_id="client_aaaaaaaa", requested=A); register(b, sid, client_id="client_bbbbbbbb", requested=B)
        pump(a, b, ((A, CAMS[A]), (B, CAMS[B])), sid, clock, n=5)
    ev = [e for e in fapp.state.events.recent if e["event_type"] == "fusion_status_changed"]
    assert len(ev) == 1 and ev[0]["to_status"] == "TRACKING_MULTI_VIEW" and ev[0]["session_id"] == sid          # transitions, not 15 Hz spam


def test_fusion_can_be_disabled_without_affecting_phase1(make_settings, clock):
    s = make_settings()
    s.fusion.spatial.enabled = False
    with TestClient(create_app(s, clock, console_log=False)) as c:
        sid = c.post("/api/sessions").json()["session_id"]
        with c.websocket_connect("/ws/sensor") as ws:
            who = register(ws, sid)["sensor_id"]
            send(ws, wire_obs(sid, who, CAMS[A], X, 0, clock.now_ms()))
            assert c.get(f"/api/sessions/{sid}").json()["sensors"][0]["observations_accepted"] == 1
        assert c.get(f"/api/sessions/{sid}/fusion").status_code == 404 and c.get(f"/api/sessions/{sid}").json()["fusion"] is None


def test_session_stop_closes_fusion_resources(fclient, fapp, sid, clock):
    with fclient.websocket_connect("/ws/sensor") as a:
        register(a, sid, client_id="client_aaaaaaaa", requested=A)
        fclient.post(f"/api/sessions/{sid}/stop")
    assert fapp.state.manager.get(sid).fusion.recorder._fh is None
