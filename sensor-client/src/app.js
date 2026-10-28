// Sensor page controller: wires camera -> detector -> link, with exactly one loop and one socket.
import { CameraController, classifyCameraError } from "./camera.js";
import { LandmarkDetector } from "./landmark_detector.js";
import { SensorStore, UI_STATE_TEXT, deriveUiState } from "./sensor_state.js";
import { SensorLink } from "./websocket_client.js";
import { nowEpochMs } from "./protocol.js";

const $ = (id) => document.getElementById(id);
const store = new SensorStore();
const camera = new CameraController();
let cfg = null, link = null, detector = null;
let runId = 0, loopToken = 0, rvfcHandle = null, rafHandle = null;
const counters = { skipped: 0, sent: [], lastFlagDrops: 0 };

function newClientId() {
  const saved = sessionStorage.getItem("saga_client_id");           // per tab: two tabs = two sensors
  if (saved) return saved;
  const bytes = crypto.getRandomValues(new Uint8Array(12));
  const id = "c" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  sessionStorage.setItem("saga_client_id", id);
  return id;
}
function wsUrl() {
  if (cfg.public_ws_url) return cfg.public_ws_url;
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${cfg.ws_path}`;
}
function logLine(text) {
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  const ul = $("log"); ul.prepend(li);
  while (ul.children.length > 30) ul.lastChild.remove();
}
function setError(msg) { store.update({ error: msg }); if (msg) logLine(`ERROR ${msg}`); }

function render(s) {
  const ui = deriveUiState(s);
  const el = $("ui-state"); el.textContent = UI_STATE_TEXT[ui]; el.dataset.state = ui;
  $("b-permission").textContent = s.permission; $("b-camera").textContent = s.camera;
  $("b-model").textContent = s.inference; $("b-connection").textContent = s.connection;
  $("sensor-id").textContent = s.sensorId ?? "—";
  $("last-error").textContent = s.error ?? "—";
  $("start-btn").disabled = s.running; $("stop-btn").disabled = !s.running;
  $("session-input").disabled = s.running; $("facing-select").disabled = s.running;
  $("snap-btn").disabled = !(s.running && s.camera === "active");
  link?.sendStatus(s, s.error);
}
store.subscribe(render);

function onLinkEvent(ev) {
  switch (ev.type) {
    case "socket":
      if (ev.state === "connecting") store.update({ connection: "connecting" });
      else if (ev.state === "reconnecting") { store.update({ connection: "reconnecting" }); logLine(`reconnecting in ${ev.delay_ms ?? "?"} ms`); }
      else if (ev.state === "closed") store.update({ connection: "closed" });
      break;
    case "registered":
      store.update({ connection: "registered", sensorId: ev.sensor_id, reconnectCount: link.reconnectCount, error: null });
      logLine(`registered as ${ev.sensor_id}${ev.resumed ? " (resumed)" : ""}`);
      link.sendStatus(store.get(), store.get().error, true);
      break;
    case "ack": $("rtt").textContent = `${ev.rtt_ms} ms`; break;
    case "error": setError(`${ev.code}: ${ev.message}`); break;
    case "fatal": store.update({ connection: "rejected" }); logLine(`host closed the session (${ev.code})`); stop(`host: ${ev.code}`); break;
    case "session_stop": logLine(`host stopped the session (${ev.reason})`); break;
  }
}

async function start() {
  if (store.get().running) return;
  const sessionId = $("session-input").value.trim();
  if (!/^[A-Za-z0-9_-]{4,64}$/.test(sessionId)) { setError("Enter the session code from the operator page."); return; }
  const run = ++runId;
  store.reset();
  store.update({ running: true, connection: "connecting", permission: "prompt" });
  const facing = $("facing-select").value;

  link = new SensorLink({
    url: wsUrl(), sessionId, clientId: newClientId(),
    config: { ...cfg, client_version: "saga-sensor-client/0.1.0" },
    capabilities: {
      landmarks: [...cfg.vision.required_landmarks, ...cfg.vision.optional_landmarks],
      max_update_hz: cfg.vision.max_update_hz, inference: "on_device", camera: null,
      user_agent: navigator.userAgent.slice(0, 256),
    },
    getHealth: () => ({
      camera: store.get().camera, inference: store.get().inference, observed_hz: observedHz(),
      frames_skipped: counters.skipped, send_drops: link?.frameSendDrops ?? 0,
    }),
    onEvent: onLinkEvent,
  });
  link.start();

  try {
    store.update({ camera: "starting" });
    camera.onEnded = () => { if (run === runId) { store.update({ camera: "unavailable", error: "Camera stream ended." }); loopToken++; } };
    const info = await camera.start($("video"), { facing, width: cfg.camera.ideal_width, height: cfg.camera.ideal_height, frameRate: cfg.camera.ideal_frame_rate });
    if (run !== runId) return camera.stop();
    link.capabilities.camera = { facing: info.facing, width: info.width, height: info.height };
    store.update({ permission: "granted", camera: "active" });
    $("video").classList.toggle("mirror", facing === "user"); $("overlay").classList.toggle("mirror", facing === "user");
  } catch (e) {
    if (run !== runId) return;
    const c = classifyCameraError(e);
    store.update({ permission: c.permission, camera: c.camera, error: c.message });
    logLine(`camera: ${c.code}`);
    return;                                                // stay connected so the operator sees why; user can Stop
  }

  store.update({ inference: "loading" });
  try {
    detector = new LandmarkDetector(cfg);
    await detector.load();
    if (run !== runId) { detector.close(); detector = null; return; }
    logLine(`model ready (${detector.delegate})`);
    store.update({ inference: "ready" });
  } catch (e) {
    if (run !== runId) return;
    store.update({ inference: "error", error: `Model failed to load: ${e?.message ?? e}. Did you run scripts/fetch_assets.sh?` });
    return;
  }
  beginLoop(run);
}

function observedHz() {
  const now = performance.now();
  while (counters.sent.length && now - counters.sent[0] > 2000) counters.sent.shift();   // bounded
  return counters.sent.length ? Math.round((counters.sent.length / Math.max(1, Math.min(2000, now - counters.sent[0]) / 1000)) * 10) / 10 : 0;
}

function beginLoop(run) {
  const token = ++loopToken;
  const video = $("video");
  const wanted = [...cfg.vision.required_landmarks, ...cfg.vision.optional_landmarks];
  const hz = () => Math.min(link.hostTargetHz, cfg.vision.max_update_hz);
  let lastProcessed = -Infinity, frameCounter = 0, lastSeenDrops = 0;
  counters.skipped = 0; counters.sent.length = 0;
  store.update({ inference: "detecting" });

  const schedule = () => {
    if (typeof video.requestVideoFrameCallback === "function") rvfcHandle = video.requestVideoFrameCallback(tick);
    else rafHandle = requestAnimationFrame((t) => tick(t, null));
  };
  const tick = (now, meta) => {
    if (token !== loopToken || run !== runId) return;      // stale loop: do not reschedule
    schedule();                                            // schedule first so an exception cannot kill the loop
    if (video.readyState < 2 || !detector?.landmarker) return;
    const interval = 1000 / hz();
    if (now - lastProcessed < interval * 0.8) { counters.skipped++; return; }   // rate cap: skip, never queue
    lastProcessed = now;
    try {
      const d = detector.detect(video, wanted);
      const flags = [];
      if (d.inferenceMs > interval * 0.8) flags.push("overloaded");
      if (d.outOfFrame) flags.push("out_of_frame");
      if (link.frameSendDrops > lastSeenDrops) { flags.push("frames_dropped"); lastSeenDrops = link.frameSendDrops; }
      const capturedAtMs = Math.round(performance.timeOrigin + (meta?.captureTime ?? now));
      const r = link.sendObservation({
        frameId: meta?.presentedFrames ?? frameCounter, capturedAtMs, imageWidth: d.imageWidth, imageHeight: d.imageHeight,
        status: d.status, hand: d.hand, landmarks: d.landmarks,
        quality: { inference_ms: Math.round(d.inferenceMs * 10) / 10, handedness_score: d.handednessScore, flags },
      });
      frameCounter++;
      if (r === "sent") counters.sent.push(performance.now());
      draw(d);
      $("rate-sent").textContent = `${observedHz()} /s`;
      $("drops").textContent = String(link.frameSendDrops);
    } catch (e) { setError(`detection error: ${e?.message ?? e}`); }
  };
  schedule();
}

function draw(d) {
  const c = $("overlay"), ctx = c.getContext("2d");
  if (c.width !== d.imageWidth) { c.width = d.imageWidth; c.height = d.imageHeight; }
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = d.status === "tracked" ? "#22c55e" : "#ef4444";
  for (const l of d.landmarks) { ctx.beginPath(); ctx.arc(l.x_px, l.y_px, 6, 0, 7); ctx.fill(); }
}

/** Calibration aid: save the CURRENT, un-mirrored frame at the exact stream resolution as a local download.
 *  Calibrate with images taken this way (a phone's photo mode can use a different resolution/crop/orientation).
 *  Explicit user action; the file never leaves the device. */
function saveCalibrationFrame() {
  const v = $("video");
  if (!v.videoWidth) return;
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);                    // raw frame: CSS mirroring does not apply here
  c.toBlob((blob) => {
    if (!blob) { setError("Could not encode the frame."); return; }
    const a = document.createElement("a");
    const label = (store.get().sensorId ?? "camera").replace(/[^A-Za-z0-9_-]/g, "");
    a.href = URL.createObjectURL(blob);
    a.download = `${label}_${c.width}x${c.height}_${Date.now()}.png`;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    logLine(`saved calibration frame ${c.width}x${c.height} (local file only)`);
  }, "image/png");
}

/** Idempotent full teardown: loop, camera tracks, model, socket. */
function stop(reason = null) {
  runId++; loopToken++;
  const video = $("video");
  if (rvfcHandle !== null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(rvfcHandle);
  if (rafHandle !== null) cancelAnimationFrame(rafHandle);
  rvfcHandle = rafHandle = null;
  link?.stop(true, "user_stop"); link = null;
  camera.stop();
  detector?.close(); detector = null;
  const c = $("overlay"); c.getContext("2d").clearRect(0, 0, c.width, c.height);
  $("rate-sent").textContent = "—"; $("drops").textContent = "0"; $("rtt").textContent = "—";
  const keep = reason ? { error: reason } : {};
  store.update({ running: false, camera: "stopped", inference: "idle", connection: "closed", ...keep });
  logLine(`stopped${reason ? ` (${reason})` : ""}`);
}

async function init() {
  const res = await fetch("/api/client-config");
  cfg = await res.json();
  const q = new URLSearchParams(location.search).get("session");
  if (q) $("session-input").value = q;
  $("facing-select").value = cfg.camera.facing_mode;
  if (!window.isSecureContext) {
    $("secure-warning").hidden = false;                    // camera will not work; explain instead of failing silently
  }
  $("start-btn").addEventListener("click", start);
  $("stop-btn").addEventListener("click", () => stop());
  $("snap-btn").addEventListener("click", saveCalibrationFrame);
  addEventListener("pagehide", () => { if (store.get().running) stop("page closed"); });   // release camera when leaving
  render(store.get());
}
init().catch((e) => { $("last-error").textContent = `Failed to load config from host: ${e.message}`; });
