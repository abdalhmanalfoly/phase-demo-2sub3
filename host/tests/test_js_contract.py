"""Cross-check: messages built by the browser JS builders must satisfy the Python contract, and the JS
validator must agree with Python on shared valid/invalid examples."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from host.app.schemas.messages import ProtocolError, parse_client_message

ROOT = Path(__file__).resolve().parents[2]
EMIT = ROOT / "sensor-client" / "tests" / "emit_messages.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_emit():
    out = subprocess.run(["node", str(EMIT)], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(out.stdout)


def test_js_built_messages_pass_python_validation():
    data = run_emit()
    assert {m["message_type"] for m in data["built"]} == {
        "HELLO", "HEARTBEAT", "SENSOR_STATUS", "LANDMARK_OBSERVATION", "DISCONNECT"}
    for m in data["built"]:
        parse_client_message(m)


def test_js_and_python_validators_agree_on_negative_cases():
    data = run_emit()
    assert len(data["cases"]) >= 15
    for case in data["cases"]:
        try:
            parse_client_message(case["message"])
            py_ok = True
        except ProtocolError:
            py_ok = False
        assert py_ok == case["js_ok"] == case["expected_ok"], case["name"]
