import copy
import json

import pytest
from fastapi.testclient import TestClient

from host.app.config import load_settings
from host.app.main import create_app

T0 = 1_791_735_600_000


class FakeClock:
    def __init__(self, t=T0):
        self.t = t

    def now_ms(self):
        return self.t

    def advance(self, ms):
        self.t += ms


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_settings(tmp_path):
    def _make(**over):
        s = load_settings(env={"SAGA_LOG_DIR": str(tmp_path / "logs")})
        s.sweeper_enabled = False                  # tests call manager.sweep() explicitly
        s.limits.error_throttle_ms = 0
        s.session.hello_timeout_s = 0.4
        for path, val in over.items():
            obj = s
            *parents, leaf = path.split(".")
            for p in parents:
                obj = getattr(obj, p)
            setattr(obj, leaf, val)
        return s
    return _make


@pytest.fixture
def settings(make_settings):
    return make_settings()


@pytest.fixture
def app(settings, clock):
    return create_app(settings, clock, console_log=False)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def manager(app):
    return app.state.manager


@pytest.fixture
def session_id(client):
    return client.post("/api/sessions").json()["session_id"]


# ----------------------------------------------------------------- message builders
def hello(session_id, client_id="client_aaaaaaaa", requested=None, t=T0, **over):
    m = {
        "protocol_version": "1.0", "message_type": "HELLO", "session_id": session_id, "client_id": client_id,
        "requested_sensor_id": requested, "client_version": "test/1.0", "model_version": "m/1", "client_time_ms": t,
        "capabilities": {"landmarks": ["wrist", "thumb_tip", "index_tip"], "max_update_hz": 30, "inference": "on_device",
                         "camera": {"facing": "user", "width": 640, "height": 480}, "user_agent": None},
    }
    m.update(over)
    return m


def heartbeat(session_id, sensor_id, seq, t=T0, **over):
    m = {"protocol_version": "1.0", "message_type": "HEARTBEAT", "session_id": session_id, "sensor_id": sensor_id,
         "sequence_number": seq, "client_time_ms": t,
         "health": {"camera": "active", "inference": "detecting", "observed_hz": 15.0, "frames_skipped": 0, "send_drops": 0}}
    m.update(over)
    return m


def status(session_id, sensor_id, t=T0, **over):
    m = {"protocol_version": "1.0", "message_type": "SENSOR_STATUS", "session_id": session_id, "sensor_id": sensor_id,
         "client_time_ms": t, "permission": "granted", "camera": "active", "inference": "detecting",
         "connection": "connected", "reconnect_count": 0, "detail": None}
    m.update(over)
    return m


def lm(i, x, y, c=None):
    return {"id": i, "x_px": x, "y_px": y, "confidence": c, "x_norm": None, "y_norm": None, "z_model": None}


def obs(session_id, sensor_id, seq, t=T0, **over):
    m = {"protocol_version": "1.0", "message_type": "LANDMARK_OBSERVATION", "session_id": session_id,
         "sensor_id": sensor_id, "sequence_number": seq, "frame_id": seq, "captured_at_ms": t,
         "image_width": 640, "image_height": 480, "status": "tracked", "hand": "right",
         "landmarks": [lm("wrist", 300.0, 200.0, 0.9), lm("thumb_tip", 320.0, 190.0), lm("index_tip", 330.0, 185.0)],
         "quality": {"inference_ms": 20.0, "handedness_score": 0.9, "flags": []}}
    m.update(over)
    return m


def send(ws, msg):
    ws.send_text(msg if isinstance(msg, str) else json.dumps(msg))


def recv(ws):
    return json.loads(ws.receive_text())


def register(ws, session_id, **kw):
    """HELLO and return the REGISTERED reply."""
    send(ws, hello(session_id, **kw))
    reply = recv(ws)
    assert reply["message_type"] == "REGISTERED", reply
    return reply


def clone(m):
    return copy.deepcopy(m)
