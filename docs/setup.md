# Setup, deployment and troubleshooting

## Requirements

| Component | Version used / required |
|---|---|
| Python | 3.12.3 (3.11+ expected to work; only 3.12.3 was run) |
| Node.js | 22.x — **only** for the optional browser-logic tests and `fetch_assets.sh` |
| Host | FastAPI 0.143.1, Uvicorn 0.54.0, Pydantic 2.14.1, PyYAML 6.0.3 (see `requirements.txt`, exact tree in `requirements.lock.txt`) |
| Browser sensor | A current Chromium-based browser (Chrome/Edge/Android Chrome) is the intended target. Safari/Firefox are **untested**. Needs a secure context, WebAssembly and `getUserMedia` |
| MediaPipe | `@mediapipe/tasks-vision` **1.1.1** + `hand_landmarker.task` float16 v1 |
| Protocol | **1.0** (`docs/protocol.md`) |

OpenCV 4.14.0.94 (`opencv-python-headless`) and NumPy 2.5.4 are required from Phase 2 (calibration + fusion; verified on Python 3.12.3; OpenCV 5.0.0 passed an API smoke test only). Godot is not used yet.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt        # or requirements.txt for runtime only
scripts/fetch_assets.sh                              # MediaPipe bundle + WASM + hand model (needs internet once)
scripts/run_host.sh                                  # http://localhost:8765
```

1. Open `http://localhost:8765/client/operator.html`, press **Create session**, copy the join link.
2. On each sensor device open the join link (`/client/?session=sess_xxxxxxxx`), press **Start**, allow the camera.
3. Watch the sensors appear in the operator table with rate, age and state.

`fetch_assets.sh` pins the MediaPipe npm package; the model's SHA-256 is recorded on first download and
verified afterwards (it is not pinned in the repo because the model could not be downloaded while this
project was being written — record and commit the hash after your first fetch if you want it enforced).

## Configuration

`config/app.yaml` (server, session, logging, limits), `config/thresholds.yaml` (heartbeat, ingress age,
update rate, client reconnect), `config/camera.yaml` (camera constraints, model paths, `flip_handedness`), `config/fusion.yaml` (Phase 2: synchronization, triangulation, smoothing, calibration quality).
Environment overrides: `SAGA_HOST`, `SAGA_PORT`, `SAGA_LOG_DIR`, `SAGA_ALLOWED_ORIGINS` (comma list),
`SAGA_PUBLIC_WS_URL`, `SAGA_CONFIG_DIR`, `SAGA_CALIBRATION_DIR`, `SAGA_RIG_ID`, `SAGA_ALLOW_SYNTHETIC_CALIBRATION`; TLS: `SAGA_TLS_CERT`, `SAGA_TLS_KEY`. See `.env.example`.
The browser reads everything it needs from `/api/client-config`, so there is one source of truth.

## Camera access on phones: secure context

Browsers expose `getUserMedia` only in a **secure context**: `https://…` or `http://localhost`. A page at
`http://192.168.x.y:8765` will have **no camera API** on a phone — the sensor page shows a warning and a
"camera unavailable" state instead of failing silently. Options, in order of preference:

1. **Local HTTPS with mkcert (recommended).** Install [mkcert](https://github.com/FiloSottile/mkcert), then
   `scripts/make_dev_cert.sh <LAN-IP>`; set `SAGA_TLS_CERT/SAGA_TLS_KEY` in `.env`; install the mkcert root CA
   on the phone and trust it. Page and WSS share one origin, so nothing else is needed.
2. **Self-signed via openssl** (the script falls back to this without mkcert). The phone shows a warning you
   must accept; some mobile browsers still refuse camera access on an untrusted certificate.
3. **Android over USB:** `chrome://inspect` → port forwarding `8765 → localhost:8765`, then open
   `http://localhost:8765` on the phone (localhost is secure). Good for debugging; not for a demo with two
   phones at once unless each is forwarded.
4. **Chrome flag** `chrome://flags/#unsafely-treat-insecure-origin-as-secure` with the host URL — a
   development convenience; it weakens security on that browser profile.
5. Tunnels (ngrok, Cloudflare) work but route traffic off your LAN; avoid for camera-derived data unless
   you accept that.

Keep the host on a trusted LAN. Do not forward the port to the internet (no authentication in Phase 1).
Phone and host must be on the same network without client isolation (avoid "guest" Wi-Fi).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "camera blocked / insecure" warning | Not HTTPS/localhost — see above |
| State "Camera permission denied" | Allow camera in the site settings, press Start again |
| "Camera unavailable: in use" | Another app/tab holds the camera; close it |
| "Model failed to load" | `scripts/fetch_assets.sh` not run, or model blocked/missing under `sensor-client/vendor/models/` |
| `ERROR UNKNOWN_SESSION` | Session code wrong or the host was restarted (sessions are in memory) |
| `ERROR STALE_OBSERVATION` repeatedly | Phone clock jumped, or the network delays frames > 2 s; check Wi-Fi |
| Sensor `degraded`/`offline` | No heartbeat for 4.5 s / 8 s: network drop, or iOS suspended the tab (keep the screen on) |
| WSS fails on a phone, page loads | Certificate not trusted for the WebSocket; open the page first and accept/trust the cert |
| Left/right labels swapped | Flip `camera.flip_handedness` in `config/camera.yaml` |
| Fusion says `CALIBRATION_REQUIRED` | Sensor not bound to a camera, calibration missing/invalidated/synthetic, image size differs from calibration, or cameras from different rig sessions — see `docs/spatial.md` and the operator page's calibration table |
| Host starts slowly | OpenCV/NumPy import adds ~2–3 s; wait for `/api/health` instead of a fixed sleep |
| Low rate | Reduce `ideal_width/height`, lower `vision.target_update_hz`; check `overloaded` flags |

## Logs and data

`logs/host-events.jsonl` (rotating) and `logs/sessions/<id>/observations.jsonl` (landmark numbers only).
No video or images are produced anywhere in the default pipeline; `logging.save_raw_video: true` is refused at
startup. Delete a session's data with `rm -r logs/sessions/<id>`. `logs/` is git-ignored.
