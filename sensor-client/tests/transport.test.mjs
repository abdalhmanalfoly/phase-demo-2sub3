import test from "node:test";
import assert from "node:assert/strict";
import { ReconnectingSocket, SensorLink } from "../src/websocket_client.js";

const T = 1791735600123;
class FakeWS {
  static instances = [];
  constructor(url) { this.url = url; this.readyState = 0; this.sent = []; this.bufferedAmount = 0; FakeWS.instances.push(this); }
  open() { this.readyState = 1; this.onopen?.(); }
  receive(obj) { this.onmessage?.({ data: JSON.stringify(obj) }); }
  drop() { this.readyState = 3; this.onclose?.(); }
  send(s) { this.sent.push(JSON.parse(s)); }
  close() { this.readyState = 3; }
}
const fakeTimers = () => {
  const q = []; let id = 0;
  return { q, setTimeout: (fn, ms) => { q.push({ id: ++id, fn, ms }); return id; },
           clearTimeout: (h) => { const i = q.findIndex((t) => t.id === h); if (i >= 0) q.splice(i, 1); },
           fire() { const t = q.shift(); t?.fn(); return t; } };
};
const backoff = { base_ms: 500, max_ms: 4000, factor: 2, jitter: 0 };
const mk = (extra = {}) => {
  FakeWS.instances.length = 0;
  const timers = fakeTimers(); const events = [];
  const sock = new ReconnectingSocket({ url: "ws://x/ws", backoff, maxBufferedBytes: 1000, WebSocketImpl: FakeWS, timers,
    random: () => 0.5, onOpen: () => events.push("open"), onMessage: (d) => events.push(["msg", d]), onState: (s, d) => events.push(["state", s, d]), ...extra });
  return { sock, timers, events };
};

test("reconnects with growing delays, resets after a successful open, never runs two loops", () => {
  const { sock, timers, events } = mk();
  sock.start(); sock.start();                                         // idempotent
  assert.equal(FakeWS.instances.length, 1);
  FakeWS.instances[0].drop();
  assert.equal(timers.q.length, 1); assert.equal(timers.q[0].ms, 500);
  timers.fire(); FakeWS.instances[1].drop();
  assert.equal(timers.q[0].ms, 1000);
  timers.fire(); FakeWS.instances[2].drop();
  assert.equal(timers.q[0].ms, 2000);
  timers.fire(); FakeWS.instances[3].open();                          // success resets the attempt counter
  FakeWS.instances[3].drop();
  assert.equal(timers.q[0].ms, 500);
  assert.equal(timers.q.length, 1);
});

test("callbacks of a replaced socket are ignored (no duplicate listeners)", () => {
  const { sock, timers, events } = mk();
  sock.start();
  const first = FakeWS.instances[0];
  const stale = { onopen: first.onopen, onmessage: first.onmessage, onclose: first.onclose };
  first.drop(); timers.fire();                                         // new socket created, old one detached
  assert.equal(first.onmessage, null);
  stale.onmessage({ data: "late" }); stale.onopen(); stale.onclose();  // would fire if generation guard were missing
  assert.equal(events.filter((e) => e === "open" || (e[0] === "msg")).length, 0);
  assert.equal(timers.q.length, 0);
});

test("stop() cancels pending reconnects and prevents further connects", () => {
  const { sock, timers } = mk();
  sock.start(); FakeWS.instances[0].drop();
  assert.equal(timers.q.length, 1);
  sock.stop();
  assert.equal(timers.q.length, 0);
  assert.equal(FakeWS.instances.length, 1);
});

test("droppable sends are refused under back-pressure; non-droppable still go out", () => {
  const { sock } = mk();
  sock.start(); const ws = FakeWS.instances[0]; ws.open();
  assert.equal(sock.send({ a: 1 }, { droppable: true }), "sent");
  ws.bufferedAmount = 5000;
  assert.equal(sock.send({ a: 2 }, { droppable: true }), "backpressure");
  assert.equal(sock.send({ a: 3 }), "sent");
  assert.equal(ws.sent.length, 2);
});

const config = { vision: { target_update_hz: 15, max_update_hz: 30 }, reconnect: backoff, max_buffered_bytes: 1000,
  client_version: "t/1", model: { version_string: "m" } };
const registered = (id, resumed = false) => ({ protocol_version: "1.0", message_type: "REGISTERED", session_id: "sess_test0001",
  sensor_id: id, resumed, heartbeat_interval_ms: 2000, target_update_hz: 15, max_update_hz: 30,
  required_landmarks: ["wrist"], session_state: "active", host_time_ms: T, host_version: "0.1.0" });
const mkLink = () => {
  FakeWS.instances.length = 0; const timers = fakeTimers(); const events = [];
  const link = new SensorLink({ url: "ws://x/ws", sessionId: "sess_test0001", clientId: "c0123456789ab", config,
    capabilities: { landmarks: ["wrist"], max_update_hz: 30, inference: "on_device", camera: null, user_agent: null },
    getHealth: () => ({ camera: "active", inference: "detecting", observed_hz: 14, frames_skipped: 0, send_drops: 0 }),
    onEvent: (e) => events.push(e), WebSocketImpl: FakeWS, timers, random: () => 0.5 });
  return { link, timers, events };
};
const lmObs = () => ({ frameId: 1, capturedAtMs: T, imageWidth: 640, imageHeight: 480, status: "tracked", hand: "right",
  landmarks: [{ id: "wrist", x_px: 10, y_px: 10, confidence: null, x_norm: null, y_norm: null, z_model: null }],
  quality: { inference_ms: 1, handedness_score: null, flags: [] } });

test("link sends HELLO on open, starts exactly one heartbeat timer, and registers", () => {
  const { link, events } = mkLink();
  link.start(); const ws = FakeWS.instances[0]; ws.open();
  assert.equal(ws.sent[0].message_type, "HELLO"); assert.equal(ws.sent[0].requested_sensor_id, null);
  assert.equal(link.sendObservation(lmObs()), "not_registered");     // nothing is sent before REGISTERED
  ws.receive(registered("sensor_01"));
  assert.equal(link.registered, true); assert.equal(link.sensorId, "sensor_01");
  assert.equal(ws.sent[1].message_type, "HEARTBEAT");
  const t1 = link.hbTimer; ws.receive(registered("sensor_01")); assert.notEqual(link.hbTimer, null);   // restarts, never stacks
  assert.equal(link.sendObservation(lmObs()), "sent");
  assert.equal(ws.sent.at(-1).sequence_number, 0);
  link.stop(false);
});

test("re-registration after a drop requests the same id and resets sequence counters", () => {
  const { link, timers, events } = mkLink();
  link.start(); FakeWS.instances[0].open(); FakeWS.instances[0].receive(registered("sensor_07"));
  link.sendObservation(lmObs()); link.sendObservation(lmObs());
  FakeWS.instances[0].drop();
  assert.equal(link.registered, false); assert.equal(link.hbTimer, null);
  timers.fire(); const ws2 = FakeWS.instances[1]; ws2.open();
  assert.equal(ws2.sent[0].requested_sensor_id, "sensor_07");
  ws2.receive(registered("sensor_07", true));
  assert.equal(link.reconnectCount, 1);
  link.sendObservation(lmObs());
  assert.equal(ws2.sent.at(-1).sequence_number, 0);
  assert.ok(events.some((e) => e.type === "registered" && e.reconnected === true));
  link.stop(false);
});

test("sequence is consumed even when an observation is dropped (host can see the gap)", () => {
  const { link } = mkLink();
  link.start(); const ws = FakeWS.instances[0]; ws.open(); ws.receive(registered("sensor_01"));
  link.sendObservation(lmObs());
  ws.bufferedAmount = 10_000; assert.equal(link.sendObservation(lmObs()), "backpressure"); assert.equal(link.frameSendDrops, 1);
  ws.bufferedAmount = 0; link.sendObservation(lmObs());
  const seqs = ws.sent.filter((m) => m.message_type === "LANDMARK_OBSERVATION").map((m) => m.sequence_number);
  assert.deepEqual(seqs, [0, 2]);
  link.stop(false);
});

test("invalid observations are never sent", () => {
  const { link, events } = mkLink();
  link.start(); const ws = FakeWS.instances[0]; ws.open(); ws.receive(registered("sensor_01"));
  const bad = lmObs(); bad.landmarks[0].x_px = NaN;
  assert.equal(link.sendObservation(bad), "invalid");
  assert.ok(events.some((e) => e.code === "CLIENT_INVALID_MESSAGE"));
  assert.equal(ws.sent.filter((m) => m.message_type === "LANDMARK_OBSERVATION").length, 0);
  link.stop(false);
});

test("fatal host error stops reconnecting; stop() sends DISCONNECT and clears timers", () => {
  const { link, timers, events } = mkLink();
  link.start(); const ws = FakeWS.instances[0]; ws.open();
  ws.receive({ protocol_version: "1.0", message_type: "ERROR", session_id: null, error_code: "UNKNOWN_SESSION", message: "no such session",
    related_message_type: "HELLO", details: null, fatal: true, host_time_ms: T });
  assert.ok(events.some((e) => e.type === "fatal" && e.code === "UNKNOWN_SESSION"));
  assert.equal(timers.q.length, 0); assert.equal(FakeWS.instances.length, 1);

  const b = mkLink(); b.link.start(); const ws2 = FakeWS.instances[0]; ws2.open(); ws2.receive(registered("sensor_01"));
  b.link.stop(true);
  assert.equal(ws2.sent.at(-1).message_type, "DISCONNECT");
  assert.equal(b.link.hbTimer, null); assert.equal(b.timers.q.length, 0);
});

test("host messages that fail validation are surfaced and ignored", () => {
  const { link, events } = mkLink();
  link.start(); const ws = FakeWS.instances[0]; ws.open();
  ws.receive({ protocol_version: "1.0", message_type: "REGISTERED", sensor_id: "x" });
  assert.equal(link.registered, false);
  assert.ok(events.some((e) => e.code === "BAD_HOST_MESSAGE"));
  link.stop(false);
});
