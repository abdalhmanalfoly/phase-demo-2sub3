"""No raw video by default; structured logs; refusal of unsupported persistence."""
import json

import pytest

from host.app.config import load_settings
from .conftest import T0, obs, recv, register, send


def test_refuses_to_start_with_raw_video_persistence_enabled(tmp_path):
    import yaml
    from host.app.config import ROOT_DIR
    cdir = tmp_path / "cfg"; cdir.mkdir()
    for n in ("app", "thresholds", "camera"):
        (cdir / f"{n}.yaml").write_text((ROOT_DIR / "config" / f"{n}.yaml").read_text())
    app = yaml.safe_load((cdir / "app.yaml").read_text()); app["logging"]["save_raw_video"] = True
    (cdir / "app.yaml").write_text(yaml.safe_dump(app))
    with pytest.raises(Exception, match="save_raw_video"):
        load_settings(config_dir=cdir, env={})


def test_default_config_does_not_persist_video():
    assert load_settings(env={}).logging.save_raw_video is False


def test_a_session_leaves_only_numeric_jsonl_logs_and_no_video_or_images(client, session_id, settings):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        for i in range(5):
            send(ws, obs(session_id, sid, i))
        send(ws, obs(session_id, sid, 9, **{"image_width": 640}) | {"image_data": "AAAA"})   # attempted smuggling
        assert recv(ws)["error_code"] == "SCHEMA_VALIDATION_FAILED"
    client.post(f"/api/sessions/{session_id}/stop")
    files = [p for p in settings.log_dir.rglob("*") if p.is_file()]
    assert files and all(p.suffix == ".jsonl" for p in files), files
    blob = "".join(p.read_text() for p in files)
    assert "AAAA" not in blob and "base64" not in blob and "image_data" not in blob
    for p in files:
        for line in p.read_text().splitlines():
            json.loads(line)                                    # every line is valid JSON


def test_landmark_recording_can_be_disabled(make_settings, clock):
    from fastapi.testclient import TestClient
    from host.app.main import create_app
    s = make_settings(**{"logging.save_landmark_events": False})
    with TestClient(create_app(s, clock, console_log=False)) as c:
        sid_ = c.post("/api/sessions").json()["session_id"]
        with c.websocket_connect("/ws/sensor") as ws:
            sid = register(ws, sid_)["sensor_id"]
            send(ws, obs(sid_, sid, 0))
        c.post(f"/api/sessions/{sid_}/stop")
    assert not (s.log_dir / "sessions").exists()


def test_event_log_lines_have_required_fields_and_exclude_coordinates(client, session_id, settings):
    with client.websocket_connect("/ws/sensor") as ws:
        sid = register(ws, session_id)["sensor_id"]
        send(ws, obs(session_id, sid, 0)); send(ws, obs(session_id, sid, 0))      # second is a stale-sequence reject
        recv(ws)
    client.post(f"/api/sessions/{session_id}/stop")
    lines = [json.loads(l) for l in (settings.log_dir / "host-events.jsonl").read_text().splitlines()]
    types = {e["event_type"] for e in lines}
    assert {"session_created", "sensor_registered", "message_rejected", "sensor_disconnected", "session_stopped"} <= types
    for e in lines:
        assert {"ts_ms", "ts", "severity", "event_type", "session_id", "sensor_id", "reason"} <= e.keys()
    rej = next(e for e in lines if e["event_type"] == "message_rejected")
    assert rej["reason"] == "STALE_SEQUENCE" and rej["session_id"] == session_id and rej["sensor_id"] == sid
    assert "x_px" not in json.dumps(lines)
