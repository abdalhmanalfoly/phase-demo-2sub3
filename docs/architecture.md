# SAGA 3D — architecture (Phase 1 sensing/session layer; Phase 2 spatial layer in `docs/spatial.md`)

Phase 1 delivers the **sensing edge and session layer**: browser sensors extract hand landmarks locally and
register with a Python host, which validates, tracks health and records them. Calibration, triangulation,
Contract Net allocation, gestures and Godot come later and consume the same messages.

```
 phone / tablet / webcam browser                         host (one Python process)
┌─────────────────────────────────┐   WebSocket JSON   ┌────────────────────────────────────────────┐
│ camera.js        getUserMedia   │  (wss on phones)   │ websocket_handler.py  framing, validation, │
│ landmark_detector.js MediaPipe  │ ───────────────►   │                       dispatch, errors      │
│ websocket_client.js link/backoff│ ◄───────────────   │ schemas/messages.py   the contract          │
│ app.js   one loop, rate cap     │  REGISTERED, ACK,  │ session_manager.py    lifecycle, shutdown   │
│ sensor_state.js  UI state       │  ERROR, STOP       │ sensor_registry.py    ids, health, seq/clock│
└─────────────────────────────────┘                    │ services/  event log + landmark JSONL       │
        no video leaves the device                     │ main.py    REST API + static files          │
                                                       └────────────────────────────────────────────┘
                                                        operator.html polls /api/sessions/{id}
```

## Responsibilities

| Module | Responsibility |
|---|---|
| `schemas/messages.py` | Pydantic strict models for every message; `parse_client_message`; `ProtocolError` |
| `sensor_registry.py` | Unique session-local sensor ids, resume-by-`client_id`, heartbeat/health state machine, sequence + staleness + rate checks, clock-offset estimate, counters |
| `session_manager.py` | Create / status / stop / prune sessions; sweeper; shutdown |
| `websocket_handler.py` | Per-socket loop: HELLO deadline, size/JSON/schema checks, session & sensor binding, error replies (throttled), disconnect bookkeeping |
| `services/logging_service.py` | JSONL event log (ts, session, sensor, event, reason); ring buffer for tests/diagnostics |
| `services/observation_recorder.py` | Per-session JSONL of accepted observations (numbers only) |
| `main.py` | App factory, REST API, static serving of `sensor-client/`, Origin policy |

Logical agents stay modules in one process (per the brief). The Phase 1 "agents" are the browser
**Sensor Agent** (A1) and the host's session/registry services; the Coordinator, Fusion, Gesture and World
agents are Phase 2–4.

## Key decisions

* **Edge inference, landmarks only.** Hand Landmarker runs in the browser (GPU delegate, CPU fallback).
  The only per-frame payload is ≤ 5 landmarks plus metadata (~0.9 KB).
* **Pixels are normative**, with the true frame size in every observation; normalized copies are optional and
  checked against pixels. `z_model` is model-relative depth and explicitly not world geometry.
* **Honest absence:** no per-landmark confidence from the hand model → `confidence: null`; a lost hand sends
  `status: "no_hand"` so Phase 2 can distinguish "no data" from "no hand".
* **Sequence epochs:** observation and heartbeat sequences are per stream and restart at 0 on every
  registration; the client assigns a sequence number even to frames it drops, so the host measures gaps.
* **Clocks:** phones are not synchronized with the host. The host estimates offset from heartbeats
  (min of `host_rx − client_time`) and uses it only for staleness. `captured_at_ms` stays untouched.
* **Backpressure, not backlog:** the client processes only the latest frame (rate-capped, no queue) and drops
  observations when the socket send buffer is full; the host rate-limits and throttles error replies.
* **Health from heartbeats:** observations alone never keep a sensor "online".
* **Secure context:** one origin serves page + WebSocket; HTTPS/WSS is required on phones (docs/setup.md).
* **MediaPipe assets are served locally** (`sensor-client/vendor/`, fetched by script, git-ignored).

## Phase 2 handoff: consuming the landmark stream without protocol changes

* **Live:** add a subscriber to `SensorRegistry.accept_observation` (or pass the returned entry onward in
  `websocket_handler._handle_bound`). The entry already contains everything fusion needs: `sensor_id`,
  `captured_at_ms` (sensor clock) + `received_at_ms` (host clock) + `estimated_age_ms`, `image_width/height`,
  pixel `x_px/y_px`, `status`, `hand`. Fusion pairs observations across sensors by corrected capture time
  (`captured_at_ms + clock_offset_ms`, exposed per sensor) within `max_inter_camera_skew_ms`.
* **Replay:** `logs/sessions/<id>/observations.jsonl` is one accepted message per line, so Phase 2 can be
  developed and regression-tested offline by replaying it (feeds TEST-13).
* **Coordinator (Contract Net):** `TASK_REQUEST` / `OBSERVATION_OFFER` / `TASK_AWARD` are *new message types*
  added in a minor protocol version (`1.1`); the offer inputs already exist (`confidence`, `quality`,
  heartbeat `health`, `estimated_age_ms`, sensor state), and v1.0 clients keep working.
* **Calibration:** needs only `image_width/height` and pixel coordinates; per-camera intrinsics are stored per
  `sensor_id`/device in `calibration/` (Phase 2). Camera movement invalidation is a host decision.
* **World adapter:** reads fusion output; no change to the sensor protocol.

## Known limitations (Phase 1)

* Not exercised with a real camera, MediaPipe inference or a physical phone: only host/protocol/transport logic is
  automated (synthetic landmarks). The manual smoke test in `docs/testing.md` is required before relying on it.
* The hand-landmarker model could not be downloaded in the authoring environment, so its hash is not pinned.
* Handedness mapping (`flip_handedness: true`) follows MediaPipe's documented mirrored-image assumption but is unverified.
* Sessions and sensor registrations live in memory: a host restart ends all sessions (sensors then get `UNKNOWN_SESSION`).
* On host *process* shutdown sensors see close code 1012 rather than `SESSION_STOP` (uvicorn closes sockets first).
* No authentication or TLS termination logic in the app; HTTPS is provided by uvicorn with a dev certificate.
* The clock-offset estimate is the minimum observed `host_rx − client_time` (includes the smallest network delay);
  adequate for staleness checks, not for sub-10 ms fusion pairing.
* Safari and Firefox are untested; the GPU delegate falls back to CPU silently (reported in the page log).
* A browser tab suspended by the OS (iOS, background tabs) stops heartbeats and will be reported `degraded`/`offline`.


## Phase 2 addendum

Calibration and fusion were added **without changing Phase 1 messages, error codes or tests**: the WebSocket handler hands each accepted
observation entry to a per-session `FusionService` (exceptions isolated), a 50 ms ticker drives tracking-loss timing, and results are exposed
through `FusionResult` (`POSE_UPDATE`) subscribers, the HTTP API and `fusion.jsonl`. Interfaces, conventions, the calibration procedure, the
result contract Phase 3/4 consume, measured (synthetic) results and the hardware validation still required are in **`docs/spatial.md`**.
The Phase 2 limitation list is §11 there; it extends (and where noted supersedes) the list above — e.g. `logging.save_landmark_events` is now the master
switch for all movement logs including `fusion.jsonl`.
