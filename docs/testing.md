# Testing

## Automated

```bash
scripts/run_tests.sh          # everything below
.venv/bin/python -m pytest    # host + protocol + JS<->Python contract (per-test timeout 15 s)
cd sensor-client && node --test tests/*.test.mjs     # browser logic
```

| Area | Where |
|---|---|
| Message validation: valid/malformed, NaN/∞, ranges, confidence, timestamps, sizes, extra fields, examples | `host/tests/test_protocol_validation.py` |
| Session create/status/stop/idempotence/limits/pruning, host shutdown, Origin policy | `host/tests/test_sessions.py` |
| Unique registration, duplicate ids, resume/takeover, unknown session, session full, HELLO timeout, malformed frames, DISCONNECT | `host/tests/test_registration.py` |
| Observation acceptance, host-assigned `received_at_ms`, stale/duplicate sequence, gaps, wrong session/sensor, clock skew, future/stale timestamps, rate limit, error throttling | `host/tests/test_observations.py` |
| Heartbeats, online→degraded→offline→recovery, self-reported failures, disconnect + reconnect with fresh sequences | `host/tests/test_health.py` |
| No raw video, JSONL-only logs, log fields, raw-video setting refused, recording switch | `host/tests/test_privacy_and_logging.py` |
| JS-built messages satisfy the Python contract; both validators agree on 23 cases | `host/tests/test_js_contract.py` + `sensor-client/tests/emit_messages.mjs` |
| Shared examples in the browser validator, UI-state derivation, backoff bounds, reconnect loop, no duplicate listeners, back-pressure, re-registration, orderly stop | `sensor-client/tests/*.test.mjs` |
| **Phase 2** pure geometry: projection, DLT vs OpenCV, distorted/undistorted, depths 1–4 m, noise sensitivity, degeneracies (infinity, behind camera, NaN) | `host/tests/test_spatial_geometry.py` |
| **Phase 2** calibration: schema validation (15 corrupt/singular/inconsistent cases), store (history, atomic write, corrupt files), registry (bindings, size mismatch, synthetic refusal, rig compatibility), invalidation + recalibration, image-based pipeline on rendered boards with measured error | `host/tests/test_spatial_calibration.py` |
| **Phase 2** synchronization (windows, skew, stale, out-of-order, clock offset, bounded buffers, gaps) and filters (noise/lag, hold/expiry, resets) | `host/tests/test_spatial_sync_filters.py` |
| **Phase 2** fusion engine: every status, outlier handling, never-falsely-trusted cases, hold/loss, recovery, calibration lifecycle, result contract | `host/tests/test_spatial_fusion.py` |
| **Phase 2** through the real host: WebSocket sensors → fusion → API, bindings, invalidation, tick-driven loss, fault isolation, logs | `host/tests/test_spatial_integration.py` |
| Live two-sensor run against a running host (uses the real browser `SensorLink`, synthetic landmarks) | `sensor-client/tests/e2e_live.mjs` (manual: `node tests/e2e_live.mjs http://localhost:8765`) |

Phase 2 live synthetic run (real clock, real 50 ms tick; needs the example rig):
`SAGA_CALIBRATION_DIR=calibration/example_synthetic SAGA_ALLOW_SYNTHETIC_CALIBRATION=1 scripts/run_host.sh &` then
`.venv/bin/python scripts/e2e_fusion_live.py http://localhost:8765`; `scripts/calibrate.py selftest` prints the synthetic calibration/accuracy numbers.

**Not covered by automation:** real camera, MediaPipe inference, rendering/overlay, mobile browser behaviour,
Wi-Fi conditions, **and all physical calibration/fusion accuracy** (see `docs/spatial.md` §11). That is what the manual guide is for.

## Manual browser smoke test

Prerequisites: host running (`scripts/run_host.sh`), assets fetched, HTTPS for phones (docs/setup.md).
Use one laptop browser + one phone, or two laptop tabs/windows (two webcams or the same webcam in turn —
a webcam can usually be opened by only one tab at a time).

1. **Operator.** Open `/client/operator.html` → *Create session*. Expect a session id, join link, host line
   "ok", empty sensor table.
2. **Permission – deny path.** Open the join link, press *Start*, **Block** the camera. Expect state
   "Camera permission denied", Permission = `denied`, an actionable message; the operator shows the sensor
   `degraded` (reason `permission_denied`). Press *Stop*; re-enable the permission in site settings.
3. **Permission – allow path.** *Start* → *Allow*. Expect "Waiting for camera permission…" → live preview →
   "Loading hand-landmark model…" → "Detecting & sending landmarks". Sensor id appears (e.g. `sensor_01`).
4. **Landmark extraction.** Show one hand ~0.5–1 m from the camera. Expect green dots on wrist, thumb tip,
   index tip (and two palm points) tracking the hand; they turn red/disappear when the hand leaves. The
   operator shows `tracked`/`no_hand`, a rate near the target (15 Hz unless overloaded) and a small
   observation age (tens to a few hundred ms on a LAN).
5. **Handedness.** Raise your *right* hand: operator "Hand" should read `right`. If swapped, flip
   `camera.flip_handedness` in `config/camera.yaml`, restart the host, reload. *(Unverified assumption —
   please confirm on your camera.)*
6. **Two clients.** Open the join link on a second device/tab, *Start*. Expect a second, **different** sensor id,
   both `online`, both with rates and no rejected messages.
7. **Reconnect.** Toggle Wi-Fi off on one phone for ~15 s. Expect operator: `degraded` → `offline`/`disconnected`;
   sensor page: "Not connected to host", log lines "reconnecting in … ms" with growing delays (≤ 10 s). Turn Wi-Fi
   on: it re-registers with the **same** sensor id (log "registered as … (resumed)"), reconnect count 1.
8. **Stop behaviour.** Press *Stop* on one sensor. Expect the preview to go black, **the browser's camera
   indicator to turn off**, operator state `disconnected` (`client_user_stop`). Close a running tab instead:
   same result via `page closed`.
9. **Session stop.** Press *Stop session* in the operator. Expect the remaining sensor page to log that the host
   stopped the session, stop reconnecting, and release the camera after *Stop*; the operator shows `stopped`.
10. **Privacy.** After the run: `find logs -type f` shows only `.jsonl`; `grep -r base64 logs` finds nothing.
11. **Overload.** Throttle the CPU (DevTools → Performance → 6× slowdown) or lower `target_update_hz` to see the
    rate settle lower with `overloaded` flags and no growing delay (frames are dropped, never queued).

12. **Calibration frame.** With the camera running press *Save calibration frame*: a PNG named `<sensor>_<W>x<H>_<time>.png`
    downloads to the device, its size equals the stream resolution, and the network tab shows no upload.

Record the browser, device, OS and results for each step when you run it. Physical calibration/fusion validation: `docs/spatial.md` §3 and §11.
