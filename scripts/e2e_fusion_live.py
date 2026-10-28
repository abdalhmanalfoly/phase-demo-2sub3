#!/usr/bin/env python3
"""Live SYNTHETIC end-to-end check of Phase 2 against a RUNNING host (real clock, real 50 ms fusion tick).

Two simulated sensors (named after the cameras of the example rig) stream the projection of a moving 3D hand path through
the real WebSocket protocol. Prints measured fusion error, latency, and tracking-loss timing. Says nothing about physical accuracy.

  SAGA_CALIBRATION_DIR=calibration/example_synthetic SAGA_ALLOW_SYNTHETIC_CALIBRATION=1 scripts/run_host.sh &
  .venv/bin/python scripts/e2e_fusion_live.py http://localhost:8765
"""
import contextlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from host.app.spatial import synthetic as S  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
WS = BASE.replace("http", "ws") + "/ws/sensor"
CAMS = S.rig_cameras()
results: list[bool] = []


def check(name, ok, extra=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {extra}")


def http(path, method="GET", body=None):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def truth(t_s):                                         # hand path inside both fields of view
    w = 2 * np.pi * 0.35
    return np.array([0.22 + 0.10 * np.cos(w * t_s), 0.08 + 0.05 * np.sin(w * t_s), 0.6 + 0.25 * np.sin(0.5 * w * t_s)])


def hello(sid, cid):
    return {"protocol_version": "1.0", "message_type": "HELLO", "session_id": sid, "client_id": f"client_{cid}_live",
            "requested_sensor_id": cid, "client_version": "e2e/1", "model_version": None, "client_time_ms": int(time.time() * 1000),
            "capabilities": {"landmarks": ["wrist"], "max_update_hz": 30, "inference": "on_device",
                             "camera": {"facing": "user", "width": 640, "height": 480}, "user_agent": "e2e"}}


def obs(sid, cid, seq, t_ms, point, status="tracked"):
    e = S.observation_entry(cid, CAMS[cid], point, seq, t_ms, t_ms, status=status)
    for k in ("received_at_ms", "estimated_age_ms"):
        e.pop(k)
    e["session_id"] = sid
    return e


sid = http("/api/sessions", "POST", {})["session_id"]
print("session", sid)
stack = contextlib.ExitStack()
wsA, wsB = stack.enter_context(connect(WS)), stack.enter_context(connect(WS))
for ws, cid in ((wsA, "phone_left"), (wsB, "phone_right")):
    ws.send(json.dumps(hello(sid, cid)))
    reply = json.loads(ws.recv())
    assert reply["message_type"] == "REGISTERED" and reply["sensor_id"] == cid, reply
check("both sensors registered under their camera ids (auto-bound)", True)
cal = http(f"/api/sessions/{sid}/calibration")
check("calibration status: both usable, rig ready", all(v["ok"] for v in cal["sensors"].values()) and cal["report"]["can_triangulate"], cal["report"]["summary"])

# ---- phase 1: 5 s of 15 Hz tracking, B captures 6 ms after A (realistic stagger)
t_start = time.time()
seq = 0
n = 75
for i in range(n):
    target = t_start + i / 15.0
    time.sleep(max(0, target - time.time()))
    now = time.time()
    tA = int(now * 1000)
    wsA.send(json.dumps(obs(sid, "phone_left", seq, tA, truth(now - t_start))))
    time.sleep(0.006)
    tB = int(time.time() * 1000)
    wsB.send(json.dumps(obs(sid, "phone_right", seq, tB, truth(time.time() - t_start))))
    seq += 1
time.sleep(0.3)
fus = http(f"/api/sessions/{sid}/fusion?recent=200")
recent = [r for r in fus["recent"] if r["valid"]]                 # (ring buffer: last 50 only - totals come from counters)
total_valid = fus["counters"].get("status:TRACKING_MULTI_VIEW", 0)
check("fusion produced a valid estimate for ~every frame", total_valid >= 0.95 * n, f"{total_valid}/{n} (counters)")
errs, lat, skews = [], [], []
for r in recent:
    t_eff = r["timestamp_ms"] / 1000.0 - (t_start * 0)      # effective time on the host timeline
    mean_t = t_eff - t_start
    errs.append(np.linalg.norm(np.array(r["raw_world_xyz_m"]) - truth(mean_t)) * 1000)
    lat.append(r["published_at_ms"] - r["timestamp_ms"]); skews.append(r["skew_ms"])
errs, lat, skews = map(np.array, (errs, lat, skews))
print(f"      raw 3D error vs truth path: mean {errs.mean():.2f} mm, p95 {np.percentile(errs, 95):.2f} mm (synthetic, noise-free pixels)")
print(f"      capture->published latency: median {np.median(lat):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms;  measured skew: median {np.median(skews):.1f} ms, max {skews.max():.1f} ms")
check("3D error small (calibrated synthetic cameras, noise-free)", errs.mean() < 30.0, f"mean {errs.mean():.1f} mm")
check("measured skew within the 50 ms window", skews.max() <= 50.0)
check("capture->published latency well under the 150 ms budget (host + loopback only)", np.percentile(lat, 95) < 100)
st = http(f"/api/sessions/{sid}")
check("Phase 1 counters intact", all(s["observations_accepted"] == n and s["observations_rejected"] == 0 for s in st["sensors"]),
      str([s["observations_accepted"] for s in st["sensors"]]))

# ---- phase 2: hand leaves BOTH views -> held, then TARGET_LOST on the real clock
last_valid_ts = max(r["timestamp_ms"] for r in recent)          # effective time of the last valid fix (collected in phase 1)
t_leave = time.time()
for ws, cid in ((wsA, "phone_left"), (wsB, "phone_right")):
    ws.send(json.dumps(obs(sid, cid, seq, int(t_leave * 1000), truth(0), status="no_hand")))
seq += 1
seen, t_lost, held_ok = [], None, False
while time.time() - t_leave < 2.0:
    r = http(f"/api/sessions/{sid}/fusion")["latest"]["wrist"]
    if not seen or seen[-1] != r["status"]:
        seen.append(r["status"])
        if r["status"] == "INSUFFICIENT_VIEWS":
            held_ok = r["world_xyz_m"] is None and r["held_world_xyz_m"] is not None
    if r["status"] == "TARGET_LOST":
        t_lost = time.time() - t_leave
        break
    time.sleep(0.02)
check("loss goes INSUFFICIENT_VIEWS (held, xyz null) -> TARGET_LOST", seen == ["INSUFFICIENT_VIEWS", "TARGET_LOST"] and held_ok, f"status sequence {seen}")
if t_lost is not None:
    since_valid = time.time() * 1000 - last_valid_ts
    print(f"      TARGET_LOST reported {t_lost * 1000:.0f} ms after the hand left, i.e. {since_valid:.0f} ms after the last valid estimate")
    check("TARGET_LOST ~hold_ms (500) after the last valid estimate (tick = 50 ms)", 500 <= since_valid <= 700, f"{since_valid:.0f} ms")

# ---- phase 3: hand returns -> clean recovery (not dragged towards the old position)
now = time.time(); p_back = truth(2.0)
for ws, cid in ((wsA, "phone_left"), (wsB, "phone_right")):
    ws.send(json.dumps(obs(sid, cid, seq, int(now * 1000), p_back)))
seq += 1
time.sleep(0.15)
r = http(f"/api/sessions/{sid}/fusion")["latest"]["wrist"]
check("recovers to TRACKING_MULTI_VIEW at the new position", r["status"] == "TRACKING_MULTI_VIEW" and np.linalg.norm(np.array(r["world_xyz_m"]) - p_back) < 0.02, r["status"])

# ---- phase 4: right camera disconnects -> single view, never a point
wsB.send(json.dumps({"protocol_version": "1.0", "message_type": "DISCONNECT", "session_id": sid, "sensor_id": "phone_right",
                     "client_time_ms": int(time.time() * 1000), "reason": "user_stop"}))
wsB.close()
for i in range(12):
    now = time.time()
    wsA.send(json.dumps(obs(sid, "phone_left", seq, int(now * 1000), truth(now - t_start)))); seq += 1
    time.sleep(1 / 15.0)
r = http(f"/api/sessions/{sid}/fusion")["latest"]["wrist"]
check("one camera left -> SINGLE_VIEW_FALLBACK with a ray and NO xyz", r["status"] == "SINGLE_VIEW_FALLBACK" and r["world_xyz_m"] is None and r["single_view_ray"], r["status"])

stack.close()
http(f"/api/sessions/{sid}/stop", "POST", {})
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
