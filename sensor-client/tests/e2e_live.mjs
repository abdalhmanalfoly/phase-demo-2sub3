// Live end-to-end check against a RUNNING host: two real SensorLink instances (the browser code) with
// synthetic landmarks instead of a camera. Usage: node tests/e2e_live.mjs https://localhost:8765
// (set NODE_TLS_REJECT_UNAUTHORIZED=0 for a self-signed dev certificate)
import { SensorLink } from "../src/websocket_client.js";

const base = process.argv[2] ?? "http://localhost:8765";
const wsBase = base.replace(/^http/, "ws");
const results = []; const check = (name, ok, extra = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const cfg = await (await fetch(`${base}/api/client-config`)).json();
cfg.client_version = "e2e/1";

const { session_id } = await (await fetch(`${base}/api/sessions`, { method: "POST" })).json();
console.log("session", session_id);

function makeSensor(clientId, label) {
  const events = [];
  const link = new SensorLink({
    url: `${wsBase}${cfg.ws_path}`, sessionId: session_id, clientId, config: cfg,
    capabilities: { landmarks: cfg.vision.required_landmarks, max_update_hz: 30, inference: "on_device", camera: { facing: "user", width: 640, height: 480 }, user_agent: label },
    getHealth: () => ({ camera: "active", inference: "detecting", observed_hz: 15, frames_skipped: 0, send_drops: 0 }),
    onEvent: (e) => events.push(e),
  });
  return { link, events };
}
const lm = (id, x, y) => ({ id, x_px: x, y_px: y, confidence: null, x_norm: x / 640, y_norm: y / 480, z_model: 0 });
let frame = 0;
const observe = (link, i) => link.sendObservation({
  frameId: frame++, capturedAtMs: Math.round(performance.timeOrigin + performance.now()), imageWidth: 640, imageHeight: 480,
  status: "tracked", hand: "right", landmarks: [lm("wrist", 300 + i % 50, 240), lm("thumb_tip", 320, 220), lm("index_tip", 330, 215)],
  quality: { inference_ms: 12, handedness_score: 0.9, flags: [] },
});

const A = makeSensor("client_e2e_aaaa", "A"), B = makeSensor("client_e2e_bbbb", "B");
A.link.start(); B.link.start();
await sleep(1000);
check("both sensors registered", A.link.registered && B.link.registered);
check("distinct sensor ids", A.link.sensorId && A.link.sensorId !== B.link.sensorId, `${A.link.sensorId} / ${B.link.sensorId}`);

// ~15 Hz for 4 s from both sensors
for (let i = 0; i < 60; i++) { observe(A.link, i); observe(B.link, i); await sleep(66); }
await sleep(300);
let st = await (await fetch(`${base}/api/sessions/${session_id}`)).json();
const [sa, sb] = st.sensors;
check("host received observations from both", sa.observations_accepted >= 55 && sb.observations_accepted >= 55, `${sa.observations_accepted}/${sb.observations_accepted}`);
check("update rate measured ~15 Hz", sa.update_rate_hz > 10 && sa.update_rate_hz < 20, `${sa.update_rate_hz} Hz`);
check("both online with heartbeats", st.sensors.every((s) => s.state === "online" && s.heartbeat_count >= 1), JSON.stringify(st.sensors.map((s) => s.state)));
check("latest observation fresh", sa.latest_observation_age_ms < 500, `${sa.latest_observation_age_ms} ms`);
check("clock offset estimated", typeof sa.clock_offset_ms === "number");

// malformed + out-of-session injection over a raw socket
const raw = new WebSocket(`${wsBase}${cfg.ws_path}`); const got = [];
raw.onmessage = (e) => got.push(JSON.parse(e.data));
await new Promise((r) => (raw.onopen = r));
raw.send("{oops"); raw.send(JSON.stringify({ protocol_version: "1.0", message_type: "LANDMARK_OBSERVATION" }));
await sleep(300);
check("malformed frames get explicit errors", got.some((m) => m.error_code === "MALFORMED_JSON") && got.some((m) => m.error_code === "NOT_REGISTERED" || m.error_code === "SCHEMA_VALIDATION_FAILED"), got.map((m) => m.error_code).join(","));
raw.close();

// drop B's socket (simulated network loss) -> reconnect + resume with the same id
const idB = B.link.sensorId; B.link.socket.ws.close(); B.link.socket.ws.onclose?.();   // client sees a close
await sleep(1800);
check("B reconnected and resumed same id", B.link.registered && B.link.sensorId === idB && B.link.reconnectCount === 1);
for (let i = 0; i < 5; i++) { observe(B.link, i); await sleep(66); }
await sleep(200);
st = await (await fetch(`${base}/api/sessions/${session_id}`)).json();
const sb2 = st.sensors.find((s) => s.sensor_id === idB);
check("host shows resume", sb2.reconnect_count === 1 && sb2.state === "online", `reconnects=${sb2.reconnect_count} state=${sb2.state}`);

// orderly stop of A, then session shutdown
A.link.stop(true);
await sleep(300);
st = await (await fetch(`${base}/api/sessions/${session_id}`)).json();
check("A recorded as disconnected (client_user_stop)", st.sensors.find((s) => s.sensor_id === A.link.sensorId).state_reason === "client_user_stop");
await fetch(`${base}/api/sessions/${session_id}/stop`, { method: "POST" });
await sleep(400);
check("B received SESSION_STOP and stopped reconnecting", B.events.some((e) => e.type === "session_stop") && B.link.socket.wanted === false);
st = await (await fetch(`${base}/api/sessions/${session_id}`)).json();
check("session stopped, nobody connected", st.state === "stopped" && st.sensors_connected === 0);
B.link.stop(false);
console.log(`\n${results.filter(Boolean).length}/${results.length} checks passed`);
process.exit(results.every(Boolean) ? 0 : 1);
