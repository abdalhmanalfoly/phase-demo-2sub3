"""Pure schema/contract tests (no server)."""
import json
import math
from pathlib import Path

import pytest

from host.app.schemas.messages import (
    ERROR_CODES, ProtocolError, parse_client_message, parse_host_message,
)
from .conftest import clone, hello, heartbeat, lm, obs, status

EXAMPLES = Path(__file__).resolve().parents[2] / "docs" / "examples"
CLIENT_EXAMPLES = ["hello", "heartbeat", "sensor_status", "landmark_observation", "landmark_observation_no_hand", "disconnect"]
HOST_EXAMPLES = ["registered", "heartbeat_ack", "error", "session_stop"]
S, SID = "sess_test0001", "sensor_01"


def code_of(msg):
    with pytest.raises(ProtocolError) as e:
        parse_client_message(msg)
    return e.value


@pytest.mark.parametrize("name", CLIENT_EXAMPLES)
def test_client_examples_are_valid(name):
    parse_client_message(json.loads((EXAMPLES / f"{name}.json").read_text()))


@pytest.mark.parametrize("name", HOST_EXAMPLES)
def test_host_examples_are_valid(name):
    parse_host_message(json.loads((EXAMPLES / f"{name}.json").read_text()))


def test_every_error_code_is_documented_in_examples_or_protocol_doc():
    doc = (EXAMPLES.parent / "protocol.md").read_text()
    missing = [c for c in ERROR_CODES if c not in doc]
    assert not missing, f"undocumented error codes: {missing}"


def test_valid_messages_pass():
    for m in (hello(S), heartbeat(S, SID, 0), status(S, SID), obs(S, SID, 0)):
        parse_client_message(m)


def test_non_object_and_unknown_type_and_version():
    assert code_of([1, 2]).code == "MALFORMED_JSON"
    assert code_of("x").code == "MALFORMED_JSON"
    m = obs(S, SID, 0); m["message_type"] = "TELEPORT"
    assert code_of(m).code == "UNSUPPORTED_MESSAGE_TYPE"
    m = obs(S, SID, 0); m["message_type"] = ["LANDMARK_OBSERVATION"]
    assert code_of(m).code == "UNSUPPORTED_MESSAGE_TYPE"
    m = obs(S, SID, 0); m["message_type"] = "REGISTERED"            # a host->client type sent by a client
    assert code_of(m).code == "UNSUPPORTED_MESSAGE_TYPE"
    m = obs(S, SID, 0); m["protocol_version"] = "2.0"
    assert code_of(m).code == "UNSUPPORTED_PROTOCOL_VERSION"
    m = obs(S, SID, 0); del m["protocol_version"]
    assert code_of(m).code == "UNSUPPORTED_PROTOCOL_VERSION"


@pytest.mark.parametrize("field,value", [
    ("sequence_number", -1), ("sequence_number", "5"), ("sequence_number", 1.5), ("sequence_number", True),
    ("sequence_number", 2**53), ("frame_id", -3),
    ("captured_at_ms", 0), ("captured_at_ms", 10**15), ("captured_at_ms", "1791735600000"),
    ("image_width", 0), ("image_width", 99999), ("image_height", -5),
    ("session_id", "x"), ("session_id", "has space!"), ("sensor_id", ""), ("sensor_id", "../etc"), ("sensor_id", "a" * 33),
    ("status", "lost"), ("hand", "both"),
])
def test_observation_field_rejections(field, value):
    m = obs(S, SID, 0); m[field] = value
    err = code_of(m)
    assert err.code == "SCHEMA_VALIDATION_FAILED"
    assert any(field in d["field"] for d in err.details)


def test_missing_and_extra_fields_rejected():
    m = obs(S, SID, 0); del m["frame_id"]
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"
    m = obs(S, SID, 0); m["received_at_ms"] = 123                  # host-assigned only
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"
    m = obs(S, SID, 0); m["image_data"] = "data:image/jpeg;base64,AAAA"   # no video/image smuggling
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"
    m = obs(S, SID, 0); m["landmarks"][0]["extra"] = 1
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_rejected(bad):
    m = obs(S, SID, 0); m["landmarks"][0]["x_px"] = bad
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"


@pytest.mark.parametrize("x,y", [(-100.0, 10.0), (10.0, -100.0), (800.0, 10.0), (10.0, 600.0), (1e9, 1e9)])
def test_coordinates_outside_frame_rejected(x, y):
    m = obs(S, SID, 0); m["landmarks"][0] = lm("wrist", x, y, 0.5)
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"


def test_coordinates_slightly_outside_frame_accepted():
    m = obs(S, SID, 0); m["landmarks"][0] = lm("wrist", -10.0, 485.0, 0.5)    # within the 10 % margin
    parse_client_message(m)


@pytest.mark.parametrize("c", [-0.01, 1.01, "0.5", math.nan])
def test_confidence_range(c):
    m = obs(S, SID, 0); m["landmarks"][0]["confidence"] = c
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"


def test_null_confidence_is_allowed_and_stays_null():
    p = parse_client_message(obs(S, SID, 0))
    assert p.landmarks[1].confidence is None


def test_landmark_semantics():
    m = obs(S, SID, 0); m["landmarks"] = []
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"                     # tracked needs landmarks
    m = obs(S, SID, 0, status="no_hand")
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"                     # no_hand needs none
    m = obs(S, SID, 0); m["landmarks"].append(lm("wrist", 1.0, 1.0))
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"                     # duplicate id
    m = obs(S, SID, 0); m["landmarks"][0]["id"] = "left_pinky_banana"
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"                     # unknown landmark id
    m = obs(S, SID, 0); m["landmarks"] = [lm("wrist", 1.0, 1.0)] * 40
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"                     # too many


def test_normalized_coordinates_must_agree_with_pixels():
    m = obs(S, SID, 0)
    m["landmarks"][0].update(x_norm=0.5, y_norm=0.5)                          # 320,240 != 300,200
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"
    m["landmarks"][0].update(x_px=320.0, y_px=240.0, x_norm=0.5, y_norm=0.5)
    parse_client_message(m)
    m["landmarks"][0].update(y_norm=None)                                     # only one of the pair
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"


def test_errors_never_echo_client_values():
    m = obs(S, SID, 0); m["landmarks"][0]["confidence"] = "SECRET-TOKEN-VALUE"
    err = code_of(m)
    assert "SECRET-TOKEN-VALUE" not in json.dumps(err.details) + err.message


def test_hello_validation():
    for bad in (dict(client_id="short"), dict(requested_sensor_id="bad id"), dict(client_version=""), dict(client_time_ms=5)):
        assert code_of(hello(S, **bad)).code == "SCHEMA_VALIDATION_FAILED"
    m = hello(S); m["capabilities"]["landmarks"] = ["wrist", "wrist"]
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"
    m = hello(S); m["capabilities"]["max_update_hz"] = 0
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"


def test_heartbeat_and_status_enums():
    m = heartbeat(S, SID, 0); m["health"]["camera"] = "exploded"
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"
    m = status(S, SID, permission="maybe")
    assert code_of(m).code == "SCHEMA_VALIDATION_FAILED"
