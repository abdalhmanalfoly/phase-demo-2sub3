// Emits JS-built messages and a set of valid/invalid cases (with the JS validator's verdict) as JSON.
// Consumed by host/tests/test_js_contract.py so Python and JS are checked against each other.
import {
  buildDisconnect, buildHeartbeat, buildHello, buildObservation, buildStatus, validateMessage,
} from "../src/protocol.js";

const T = 1791735600123, S = "sess_test0001", SID = "sensor_01";
const lm = (id, x, y, c = null) => ({ id, x_px: x, y_px: y, confidence: c, x_norm: null, y_norm: null, z_model: null });
const goodObs = () => buildObservation({
  sessionId: S, sensorId: SID, sequence: 3, frameId: 9, capturedAtMs: T, imageWidth: 640, imageHeight: 480, status: "tracked",
  hand: "left", landmarks: [lm("wrist", 100, 100, 0.5), lm("index_tip", 120.5, 90)],
  quality: { inference_ms: 12.5, handedness_score: 0.8, flags: ["overloaded"] },
});
const built = [
  buildHello({ sessionId: S, clientId: "c0123456789ab", clientVersion: "x/1", modelVersion: null, clientTimeMs: T,
    capabilities: { landmarks: ["wrist"], max_update_hz: 30, inference: "on_device", camera: { facing: "user", width: 640, height: 480 }, user_agent: null } }),
  buildHeartbeat({ sessionId: S, sensorId: SID, sequence: 0, clientTimeMs: T, health: { camera: "active", inference: "ready", observed_hz: null, frames_skipped: 0, send_drops: 0 } }),
  buildStatus({ sessionId: S, sensorId: SID, clientTimeMs: T, permission: "granted", camera: "active", inference: "ready", connection: "connected", reconnectCount: 0 }),
  goodObs(),
  buildDisconnect({ sessionId: S, sensorId: SID, clientTimeMs: T, reason: "user_stop" }),
];
const mut = (fn) => { const m = JSON.parse(JSON.stringify(goodObs())); fn(m); return m; };
const cases = [
  ["valid observation", goodObs(), true],
  ["x outside frame", mut((m) => { m.landmarks[0].x_px = 5000; }), false],
  ["y outside frame", mut((m) => { m.landmarks[0].y_px = -400; }), false],
  ["confidence > 1", mut((m) => { m.landmarks[0].confidence = 1.5; }), false],
  ["string sequence", mut((m) => { m.sequence_number = "3"; }), false],
  ["negative sequence", mut((m) => { m.sequence_number = -1; }), false],
  ["fractional sequence", mut((m) => { m.sequence_number = 1.5; }), false],
  ["bad epoch", mut((m) => { m.captured_at_ms = 5; }), false],
  ["bad session id", mut((m) => { m.session_id = "x"; }), false],
  ["bad sensor id", mut((m) => { m.sensor_id = "a b"; }), false],
  ["unknown landmark", mut((m) => { m.landmarks[0].id = "elbow"; }), false],
  ["duplicate landmark", mut((m) => { m.landmarks.push(lm("wrist", 1, 1)); }), false],
  ["tracked w/o landmarks", mut((m) => { m.landmarks = []; }), false],
  ["no_hand with landmarks", mut((m) => { m.status = "no_hand"; }), false],
  ["extra field", mut((m) => { m.received_at_ms = 1; }), false],
  ["missing field", mut((m) => { delete m.frame_id; }), false],
  ["zero image width", mut((m) => { m.image_width = 0; }), false],
  ["norm disagrees with px", mut((m) => { m.landmarks[0].x_norm = 0.9; m.landmarks[0].y_norm = 0.2; }), false],
  ["norm pair incomplete", mut((m) => { m.landmarks[0].x_norm = 0.15625; }), false],
  ["norm agrees with px", mut((m) => { m.landmarks[0].x_px = 320; m.landmarks[0].y_px = 240; m.landmarks[0].x_norm = 0.5; m.landmarks[0].y_norm = 0.5; }), true],
  ["bad flag", mut((m) => { m.quality.flags = ["melting"]; }), false],
  ["wrong version", mut((m) => { m.protocol_version = "2.0"; }), false],
  ["slightly outside frame ok", mut((m) => { m.landmarks[0].x_px = -10; }), true],
].map(([name, message, expected_ok]) => ({ name, message, expected_ok, js_ok: validateMessage(message) === null }));

console.log(JSON.stringify({ built, cases }));
