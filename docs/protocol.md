# SAGA 3D wire protocol — version 1.0

One JSON object per WebSocket **text** frame, endpoint `/ws/sensor`. The Python host
(`host/app/schemas/messages.py`) and the browser (`sensor-client/src/protocol.js`) implement the same
contract; both are validated against `docs/examples/*.json` and against each other in the test suites.

## Rules (invariants)

1. Every message has `protocol_version` (`"1.0"`) and `message_type`.
2. **All fields are required.** An absent value is an explicit `null` (e.g. `confidence: null` means
   "the detector gave none" — it is never silently replaced by `1.0`).
3. Unknown fields are rejected. This also means no image/video payload can ride inside a message.
4. Max message size: 16 384 bytes (configurable). Binary frames are rejected.
5. **Pixel coordinates are normative.** `x_px`/`y_px` are pixels in a frame of
   `image_width` × `image_height` (origin top-left, x right, y down), measured on the **un-mirrored**
   frame the detector saw. Optional `x_norm`/`y_norm` are labelled normalized copies
   (`x_norm = x_px / image_width`); the host rejects them if they disagree by more than 1 px.
   `z_model` is MediaPipe's model-relative depth: **not metric and not a world coordinate**.
6. Landmarks may lie up to 10 % outside the frame (detector jitter); further out is rejected.
7. `captured_at_ms` is the **sensor** clock (epoch ms). `received_at_ms` is assigned by the **host** and
   is never accepted from a client (it appears only in host records, `logs/sessions/<id>/observations.jsonl`).
8. Sequence numbers: `sequence_number` is strictly increasing per sender **per message stream**
   (observations and heartbeats are separate streams) and restarts at 0 on every (re)registration.
   Gaps are allowed and counted (`sequence_gaps`); repeats/regressions are rejected `STALE_SEQUENCE`.
9. Sensor clocks are not synchronized with the host. The host estimates a per-sensor offset from
   heartbeats (minimum of `host_receive − client_time`, over the last 20) and uses it only for
   staleness checks. Phase 2 may refine it.

## Message types

| Type | Direction | Purpose |
|---|---|---|
| `HELLO` | sensor → host | Register (or resume) in a session; declare capabilities |
| `REGISTERED` | host → sensor | Assigned/validated `sensor_id`, heartbeat interval, update-rate targets |
| `HEARTBEAT` | sensor → host | Liveness + health; also feeds the clock-offset estimate |
| `HEARTBEAT_ACK` | host → sensor | Echo + host time + the sensor's current health state |
| `SENSOR_STATUS` | sensor → host | Permission / camera / inference / connection status |
| `LANDMARK_OBSERVATION` | sensor → host | Selected landmarks for one frame (or `no_hand`) |
| `ERROR` | host → sensor | Machine-readable `error_code` + safe explanation |
| `SESSION_STOP` | host → sensor | Operator stopped the session (host then closes the socket, code 1000). If the host *process* is stopped, uvicorn closes sockets first (code 1012), so sensors see a close instead and reconnect/retry |
| `DISCONNECT` | sensor → host | Orderly departure (`user_stop`, `page_unload`, `error`) |

Landmark IDs (Phase 1, hand model): `wrist`(0), `thumb_tip`(4), `index_tip`(8), `index_mcp`(5), `pinky_mcp`(17)
(numbers are MediaPipe hand-landmark indices). Required for the MVP: `wrist`, `thumb_tip`, `index_tip`;
the two MCP points are optional extras giving a palm-size reference for later pinch normalization.
`hand` is `left`/`right`/`unknown`. MediaPipe labels assume a mirrored image; the client swaps them by
default (`camera.yaml: flip_handedness`) because it sends the un-mirrored frame — verify in the smoke test.

Names differ from the earlier Arabic draft (`register.agent`, `sensor.observation`, …); the mapping is in
Appendix A of the project documentation. Pixel coordinates replace its normalized example, as that
document specifies.

## Session flow

```
sensor                                host
  | -- HELLO ------------------------> |   unknown session / full / duplicate id -> ERROR(fatal) + close
  | <------------------- REGISTERED -- |
  | -- SENSOR_STATUS ----------------> |   (on every change)
  | -- HEARTBEAT (every interval) ---> |
  | <------------------ HEARTBEAT_ACK  |
  | -- LANDMARK_OBSERVATION (<=max Hz) |   rejected ones get ERROR (non-fatal), never silently dropped
  | -- DISCONNECT -------------------> |   or SESSION_STOP from the host
```

Re-registration: a `HELLO` carrying a known `client_id` **resumes** the same `sensor_id`
(`resumed: true`); if the old socket is still open it is closed with code 4001. A `requested_sensor_id`
owned by a different `client_id` is rejected `DUPLICATE_SENSOR_ID`. A second `HELLO` on a registered
socket is `ALREADY_REGISTERED`.

## Sensor health states (host-side, visible in `/api/sessions/{id}`)

`online` · `degraded` (heartbeat late > 4.5 s, or permission denied / camera unavailable / inference
error) · `offline` (no heartbeat > 8 s) · `disconnected` (socket closed; id kept for resume).

## Error codes

Non-fatal unless noted. `ERROR.details` lists `{field, problem}` pairs and never echoes client values.

| `error_code` | Meaning |
|---|---|
| `MALFORMED_JSON` | Not valid JSON, or not a JSON object |
| `MESSAGE_TOO_LARGE` | Over the byte limit |
| `BINARY_NOT_SUPPORTED` | Binary frame received |
| `SCHEMA_VALIDATION_FAILED` | Missing/extra/invalid fields, out-of-range values, NaN/∞ |
| `UNSUPPORTED_PROTOCOL_VERSION` | `protocol_version` ≠ `1.0` |
| `UNSUPPORTED_MESSAGE_TYPE` | Unknown type, or a host-only type sent by a sensor |
| `UNKNOWN_SESSION` | HELLO for a session that does not exist (fatal) |
| `SESSION_STOPPED` | Session no longer active (fatal) |
| `SESSION_FULL` | Too many connected sensors (fatal) |
| `NOT_REGISTERED` | Message before HELLO |
| `ALREADY_REGISTERED` | Second HELLO on one socket |
| `SESSION_MISMATCH` | `session_id` differs from the one bound to the socket |
| `SENSOR_ID_MISMATCH` | `sensor_id` differs from the one bound to the socket (also fatal in HELLO when a client re-registers under another id) |
| `DUPLICATE_SENSOR_ID` | Requested id belongs to another client (fatal) |
| `HELLO_TIMEOUT` | No HELLO within the timeout (fatal) |
| `STALE_SEQUENCE` | Sequence number did not increase |
| `STALE_OBSERVATION` | Older than the ingress limit (clock-offset corrected) or `captured_at_ms` went backwards |
| `TIMESTAMP_IN_FUTURE` | Ahead of estimated host time by more than the allowed skew |
| `RATE_LIMITED` | Faster than the host's per-sensor limit |

At most one `ERROR` reply per code per socket per second is sent (all are logged and counted).

## Phase 2 note

The sensor protocol is unchanged in Phase 2. `POSE_UPDATE` (the fusion result) is a host-internal result type that sensors never send or
receive; its contract is in `docs/spatial.md` §7 (examples in `docs/examples/fusion/`). Sensors' pixel landmarks (`x_px`, edge-origin) are converted
to OpenCV pixel coordinates by the fusion engine (`x_px − 0.5`).

## Examples

Each file is validated by the tests: [`docs/examples/`](examples/).
`hello.json`, `registered.json`, `heartbeat.json`, `heartbeat_ack.json`, `sensor_status.json`,
`landmark_observation.json`, `landmark_observation_no_hand.json`, `error.json`, `session_stop.json`,
`disconnect.json`.

```json
{
  "protocol_version": "1.0", "message_type": "LANDMARK_OBSERVATION",
  "session_id": "sess_demo0001", "sensor_id": "sensor_01", "sequence_number": 1842, "frame_id": 5210,
  "captured_at_ms": 1791735601123, "image_width": 640, "image_height": 480,
  "status": "tracked", "hand": "right",
  "landmarks": [
    {"id": "wrist", "x_px": 412.4, "y_px": 263.8, "confidence": null, "x_norm": 0.6444, "y_norm": 0.5496, "z_model": -0.02}
  ],
  "quality": {"inference_ms": 26.4, "handedness_score": 0.97, "flags": []}
}
```

`quality.flags`: `overloaded` (inference slower than ~80 % of the frame interval), `out_of_frame`
(a landmark was clamped to the margin), `frames_dropped` (earlier observations were dropped by
back-pressure). `quality.handedness_score` is the left/right classifier score, **not** a tracking
confidence; the hand model supplies no per-landmark confidence, so `confidence` is normally `null`.
