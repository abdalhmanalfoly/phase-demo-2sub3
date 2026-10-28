# SAGA 3D — Spatial Agent Grid Architecture (Phase 1)

Multi-agent spatial interaction prototype: several phones/webcams observe one person, extract hand landmarks
**locally in the browser**, and send timestamped observations to a Python host. Later phases calibrate the
cameras, triangulate 3D positions, coordinate sensors (Contract Net), recognize pinch gestures and drive a Godot
scene.

**Phase 1** = foundation + browser sensor client + session registration:
HTTP/WebSocket host, session/sensor registry with health states, versioned protocol, MediaPipe hand landmarks in
the browser, operator status page, tests.

**Phase 2 (this repository state)** adds calibration and spatial fusion (`host/app/spatial/`, `docs/spatial.md`): OpenCV camera calibration
with a shared board-defined world frame, versioned calibration artifacts, capture-time synchronization, calibrated multi-view triangulation of the
wrist with reprojection checks, smoothing with bounded hold, and an explicit status for every outcome (`TRACKING_MULTI_VIEW`, `SYNC_DEGRADED`,
`SINGLE_VIEW_FALLBACK`, `INSUFFICIENT_VIEWS`, `TARGET_LOST`, `FUSION_OUTLIER`, `GEOMETRY_DEGENERATE`, `CALIBRATION_REQUIRED`).
**Not yet:** Contract Net awards, the final gesture state machine, Godot.

> Status: automated tests pass (see `docs/testing.md`). Camera, MediaPipe, phone behaviour and **all physical accuracy** are
> **unvalidated**: Phase 2 results are from synthetic scenes (known ground truth). Follow the manual smoke test and the hardware
> validation list in `docs/spatial.md` before trusting any number.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
scripts/fetch_assets.sh        # MediaPipe 1.1.1 bundle/WASM + hand_landmarker.task (internet needed once)
scripts/run_host.sh            # http://localhost:8765
```

1. Operator page: `http://localhost:8765/client/operator.html` → **Create session**.
2. Sensor page on each device: the join link (`/client/?session=<id>`) → **Start** → allow camera.
3. Watch sensors, rates, ages and errors in the operator table.
4. Phase 2: calibrate (`docs/spatial.md` §3: `scripts/calibrate.py intrinsics` → `rig` → `inspect`), bind sensors to cameras in the operator
   page, and watch the fused wrist position/status. No hardware? `scripts/calibrate.py selftest` runs the whole pipeline on synthetic boards.

**Phones need HTTPS** (browser camera APIs require a secure context): see `docs/setup.md`
(`scripts/make_dev_cert.sh <LAN-IP>`).

## Tests

```bash
scripts/run_tests.sh
```

## Layout

```
config/            app.yaml, thresholds.yaml, camera.yaml, fusion.yaml   (single source of truth; the browser-facing parts are served to it)
docs/              architecture.md, protocol.md, spatial.md (Phase 2), setup.md, testing.md, examples/ (+ examples/fusion/)
host/app/          main.py, config.py, session_manager.py, sensor_registry.py, websocket_handler.py,
                   schemas/messages.py, services/{logging_service,observation_recorder,clock}.py,
                   spatial/{geometry,calibration_schema,calibration_store,calibration_tool,synchronizer,fusion,filters,results,service,synthetic}.py
host/tests/        pytest suite
sensor-client/     index.html, operator.html, styles.css, src/*.js (ES modules, no build step), tests/
scripts/           run_host.sh, fetch_assets.sh, make_dev_cert.sh, run_tests.sh, calibrate.py, e2e_fusion_live.py
calibration/       calibration artifacts (git-ignored) + synthetic example
logs/              runtime output (git-ignored)
```

## HTTP / WebSocket API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness, versions |
| `GET /api/client-config` | Config for browsers |
| `POST /api/sessions` · `GET /api/sessions[/{id}]` · `POST /api/sessions/{id}/stop` | Session lifecycle & status |
| `GET /api/sessions/{id}/fusion` · `…/calibration` · `POST …/calibration/{bindings,invalidate,reload}` | Phase 2 fusion state and calibration management (`docs/spatial.md`) |
| `WS /ws/sensor` | Sensor protocol (`docs/protocol.md`) |
| `/client/` · `/client/operator.html` | Sensor page · operator/debug panel |

## Privacy defaults

Landmarks only; no video/images leave the device or are stored. The host refuses to start if
`logging.save_raw_video` is enabled. Camera use is visible (preview + browser indicator), *Stop* releases the
camera immediately, and leaving the page stops it. The host has **no authentication** — run on a trusted LAN only.

## Known limitations

See "Known limitations" in `docs/architecture.md` and §11 of `docs/spatial.md`; main ones: untested on real
cameras/phones, sessions are in memory (a host restart loses them), no auth, handedness-flip assumption
unverified, hand model gives no per-landmark confidence, per-sensor clock offset is a coarse estimate, and fusion accuracy is unmeasured on real hardware (two-view triangulation also cannot detect errors along the epipolar line).

Protocol spec: `docs/protocol.md` · Calibration & fusion: `docs/spatial.md` · Architecture: `docs/architecture.md` ·
Setup/troubleshooting: `docs/setup.md` · Testing: `docs/testing.md`


## Phase 3: coordination + gesture
Contract Net-style Coordination Agent and a deterministic Gesture Agent: see `docs/coordination.md`, `docs/gesture.md`, `docs/phase3_handoff.md`.
Demo (no cameras): `python scripts/demo_coordination.py`.
