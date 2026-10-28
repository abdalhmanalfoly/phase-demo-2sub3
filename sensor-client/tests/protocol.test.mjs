import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { validateMessage } from "../src/protocol.js";

const dir = new URL("../../docs/examples/", import.meta.url);

test("every shared example in docs/examples validates in the browser validator", () => {
  const files = readdirSync(dir).filter((f) => f.endsWith(".json"));
  assert.ok(files.length >= 10);
  for (const f of files) {
    const msg = JSON.parse(readFileSync(new URL(f, dir), "utf8"));
    assert.equal(validateMessage(msg), null, `${f}: ${validateMessage(msg)}`);
  }
});

test("validator rejects non-objects, unknown types, wrong version", () => {
  assert.ok(validateMessage(null));
  assert.ok(validateMessage("x"));
  assert.ok(validateMessage({ protocol_version: "1.0", message_type: "NOPE" }));
  assert.ok(validateMessage({ protocol_version: "9", message_type: "HELLO" }));
});

test("NaN / Infinity coordinates are rejected", () => {
  const m = JSON.parse(readFileSync(new URL("landmark_observation.json", dir), "utf8"));
  for (const bad of [NaN, Infinity, -Infinity]) {
    const c = structuredClone(m); c.landmarks[0].x_px = bad;
    assert.ok(validateMessage(c), String(bad));
  }
});
