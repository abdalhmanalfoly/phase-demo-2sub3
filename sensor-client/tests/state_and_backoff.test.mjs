import test from "node:test";
import assert from "node:assert/strict";
import { SensorStore, deriveUiState, initialState } from "../src/sensor_state.js";
import { computeBackoffMs } from "../src/websocket_client.js";

const S = (o) => ({ ...initialState(), running: true, ...o });

test("UI state derivation covers every required state with sensible precedence", () => {
  assert.equal(deriveUiState(initialState()), "stopped");
  assert.equal(deriveUiState(S({ permission: "denied", camera: "idle" })), "permission_denied");
  assert.equal(deriveUiState(S({ permission: "granted", camera: "unavailable" })), "camera_unavailable");
  assert.equal(deriveUiState(S({ camera: "starting" })), "requesting_camera");
  assert.equal(deriveUiState(S({ camera: "active", inference: "loading", connection: "registered" })), "model_loading");
  assert.equal(deriveUiState(S({ camera: "active", inference: "detecting", connection: "reconnecting" })), "disconnected");
  assert.equal(deriveUiState(S({ camera: "active", inference: "detecting", connection: "registered" })), "detecting");
  assert.equal(deriveUiState(S({ camera: "active", inference: "ready", connection: "registered" })), "connected");
  assert.equal(deriveUiState(S({ permission: "denied", connection: "registered" })), "permission_denied");
});

test("store notifies only on real changes and unsubscribes", () => {
  const st = new SensorStore(); let n = 0;
  const off = st.subscribe(() => n++);
  st.update({ camera: "active" }); st.update({ camera: "active" });
  assert.equal(n, 1);
  off(); st.update({ camera: "idle" }); assert.equal(n, 1);
});

test("backoff grows exponentially, is bounded, and jitter stays within +/- fraction", () => {
  const cfg = { base_ms: 500, max_ms: 10000, factor: 2, jitter: 0.2 };
  const mid = () => 0.5;                                  // zero jitter
  assert.deepEqual([0, 1, 2, 3, 4, 5, 6, 20].map((a) => computeBackoffMs(a, cfg, mid)), [500, 1000, 2000, 4000, 8000, 10000, 10000, 10000]);
  for (const a of [0, 3, 10]) {
    const lo = computeBackoffMs(a, cfg, () => 0), hi = computeBackoffMs(a, cfg, () => 1);
    const raw = Math.min(10000, 500 * 2 ** a);
    assert.ok(lo >= raw * 0.8 - 1 && hi <= Math.min(10000, raw * 1.2) + 1 && lo <= hi);
  }
});
