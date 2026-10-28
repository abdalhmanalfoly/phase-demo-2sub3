// Transport (ReconnectingSocket) and protocol session (SensorLink: HELLO, heartbeats, sequences).
import {
  buildDisconnect, buildHeartbeat, buildHello, buildObservation, buildStatus, nowEpochMs, validateMessage,
} from "./protocol.js";

/** Exponential backoff with symmetric jitter, bounded by maxMs. attempt starts at 0. */
export function computeBackoffMs(attempt, { base_ms, max_ms, factor, jitter }, random = Math.random) {
  const raw = Math.min(max_ms, base_ms * Math.pow(factor, attempt));
  const spread = raw * jitter * (random() * 2 - 1);
  return Math.max(0, Math.min(max_ms, Math.round(raw + spread)));
}

const FATAL_CODES = new Set([
  "UNKNOWN_SESSION", "SESSION_STOPPED", "SESSION_FULL", "DUPLICATE_SENSOR_ID", "UNSUPPORTED_PROTOCOL_VERSION",
  "SENSOR_ID_MISMATCH", "HELLO_TIMEOUT",
]);

/** One live WebSocket at a time; stale callbacks are ignored via a generation counter. */
export class ReconnectingSocket {
  constructor({ url, backoff, maxBufferedBytes, onOpen, onMessage, onState, WebSocketImpl = globalThis.WebSocket,
                timers = { setTimeout: (...a) => setTimeout(...a), clearTimeout: (...a) => clearTimeout(...a) }, random = Math.random }) {
    Object.assign(this, { url, backoff, maxBufferedBytes, onOpen, onMessage, onState, WebSocketImpl, timers, random });
    this.ws = null; this.generation = 0; this.attempt = 0; this.timer = null; this.wanted = false;
  }
  start() {
    if (this.wanted) return;                 // idempotent: never two connect loops
    this.wanted = true; this.attempt = 0; this._connect();
  }
  _detach() {
    if (!this.ws) return;
    this.ws.onopen = this.ws.onmessage = this.ws.onclose = this.ws.onerror = null;
    try { this.ws.close(1000); } catch { /* already closed */ }
    this.ws = null;
  }
  _connect() {
    this._detach();
    const gen = ++this.generation;
    this.onState(this.attempt === 0 ? "connecting" : "reconnecting");
    let ws;
    try { ws = new this.WebSocketImpl(this.url); } catch { this._scheduleReconnect(gen); return; }
    this.ws = ws;
    ws.onopen = () => { if (gen !== this.generation) return; this.attempt = 0; this.onOpen(); };
    ws.onmessage = (ev) => { if (gen !== this.generation) return; this.onMessage(ev.data); };
    ws.onerror = () => { /* onclose always follows; nothing to do */ };
    ws.onclose = () => {
      if (gen !== this.generation) return;
      this._detach();                       // drop handlers from the dead socket; the generation guard is a second line of defence
      this.onState("closed"); this._scheduleReconnect(gen);
    };
  }
  _scheduleReconnect(gen) {
    if (!this.wanted || gen !== this.generation) return;
    this.timers.clearTimeout(this.timer);
    const delay = computeBackoffMs(this.attempt++, this.backoff, this.random);
    this.timer = this.timers.setTimeout(() => { this.timer = null; if (this.wanted) this._connect(); }, delay);
    this.onState("reconnecting", delay);
  }
  /** Returns "sent" | "closed" | "backpressure". Droppable messages are refused when the buffer is full. */
  send(obj, { droppable = false } = {}) {
    const ws = this.ws;
    if (!ws || ws.readyState !== 1) return "closed";
    if (droppable && ws.bufferedAmount > this.maxBufferedBytes) return "backpressure";
    ws.send(JSON.stringify(obj));
    return "sent";
  }
  stop() {
    this.wanted = false; this.generation++;
    this.timers.clearTimeout(this.timer); this.timer = null;
    this._detach();
  }
}

/** Registers with the host, keeps heartbeats, owns sequence counters. One instance per run. */
export class SensorLink {
  constructor({ url, sessionId, clientId, config, capabilities, getHealth, onEvent, ...socketOpts }) {
    Object.assign(this, { sessionId, clientId, config, capabilities, getHealth, onEvent });
    this.sensorId = null; this.registered = false; this.reconnectCount = 0; this.everRegistered = false;
    this.obsSeq = 0; this.hbSeq = 0; this.frameSendDrops = 0; this.hbTimer = null; this.lastStatusKey = null; this.rttMs = null;
    this.fatal = false; this.hbInterval = 2000; this.hostTargetHz = config.vision.target_update_hz;
    this.socket = new ReconnectingSocket({
      url, backoff: config.reconnect, maxBufferedBytes: config.max_buffered_bytes, ...socketOpts,
      onOpen: () => this._sendHello(),
      onMessage: (data) => this._onMessage(data),
      onState: (s, delay) => this._onSocketState(s, delay),
    });
  }
  start() { this.socket.start(); }
  _onSocketState(s, delay) {
    if (s === "closed" || s === "reconnecting") { this.registered = false; this._stopHeartbeat(); }
    if (this.fatal) return;
    this.onEvent({ type: "socket", state: s, delay_ms: delay ?? null });
  }
  _sendHello() {
    const msg = buildHello({
      sessionId: this.sessionId, clientId: this.clientId, requestedSensorId: this.sensorId,
      clientVersion: this.config.client_version, modelVersion: this.config.model.version_string,
      clientTimeMs: nowEpochMs(), capabilities: this.capabilities,
    });
    const p = validateMessage(msg);
    if (p) { this.onEvent({ type: "error", code: "CLIENT_INVALID_MESSAGE", message: `HELLO invalid: ${p}` }); return; }
    this.socket.send(msg);
  }
  _onMessage(data) {
    let msg;
    try { msg = JSON.parse(data); } catch { this.onEvent({ type: "error", code: "BAD_HOST_MESSAGE", message: "host sent invalid JSON" }); return; }
    const p = validateMessage(msg);
    if (p) { this.onEvent({ type: "error", code: "BAD_HOST_MESSAGE", message: `host message invalid: ${p}` }); return; }
    switch (msg.message_type) {
      case "REGISTERED": return this._onRegistered(msg);
      case "HEARTBEAT_ACK": this.rttMs = Math.max(0, nowEpochMs() - msg.client_time_ms); this.onEvent({ type: "ack", rtt_ms: this.rttMs, sensor_state: msg.sensor_state }); return;
      case "ERROR":
        this.onEvent({ type: "error", code: msg.error_code, message: msg.message, details: msg.details });
        if (msg.fatal && FATAL_CODES.has(msg.error_code)) this._fatal(msg.error_code);
        return;
      case "SESSION_STOP": this.onEvent({ type: "session_stop", reason: msg.reason }); this._fatal("SESSION_STOP"); return;
      default: return;
    }
  }
  _fatal(code) { this.fatal = true; this.stop(false); this.onEvent({ type: "fatal", code }); }
  _onRegistered(msg) {
    const reconnected = this.everRegistered;
    if (reconnected) this.reconnectCount++;
    this.sensorId = msg.sensor_id; this.registered = true; this.everRegistered = true;
    this.obsSeq = 0; this.hbSeq = 0;                      // a new registration starts new sequence epochs
    this.hbInterval = msg.heartbeat_interval_ms;
    this.hostTargetHz = Math.min(msg.target_update_hz, msg.max_update_hz);
    this.lastStatusKey = null;
    this._startHeartbeat();
    this.onEvent({ type: "registered", sensor_id: msg.sensor_id, resumed: msg.resumed, reconnected, target_update_hz: this.hostTargetHz });
  }
  _startHeartbeat() {
    this._stopHeartbeat();                                // exactly one timer
    const beat = () => {
      const msg = buildHeartbeat({ sessionId: this.sessionId, sensorId: this.sensorId, sequence: this.hbSeq++, clientTimeMs: nowEpochMs(), health: this.getHealth() });
      const p = validateMessage(msg);
      if (p) { this.onEvent({ type: "error", code: "CLIENT_INVALID_MESSAGE", message: `HEARTBEAT invalid: ${p}` }); return; }
      this.socket.send(msg);
    };
    beat();
    this.hbTimer = setInterval(beat, this.hbInterval);
  }
  _stopHeartbeat() { if (this.hbTimer !== null) { clearInterval(this.hbTimer); this.hbTimer = null; } }

  /** Send SENSOR_STATUS when registered and changed (or when force=true). */
  sendStatus(state, detail = null, force = false) {
    if (!this.registered) return false;
    const key = `${state.permission}|${state.camera}|${state.inference}|${detail}`;
    if (!force && key === this.lastStatusKey) return false;
    const msg = buildStatus({
      sessionId: this.sessionId, sensorId: this.sensorId, clientTimeMs: nowEpochMs(), permission: state.permission,
      camera: state.camera, inference: state.inference, connection: this.reconnectCount > 0 ? "reconnected" : "connected",
      reconnectCount: this.reconnectCount, detail: detail ? String(detail).slice(0, 200) : null,
    });
    if (validateMessage(msg)) return false;
    this.lastStatusKey = key;
    return this.socket.send(msg) === "sent";
  }

  /** Fields from the detection loop; sequence is consumed even if the send is dropped (host sees gaps). */
  sendObservation(fields) {
    if (!this.registered) return "not_registered";
    const msg = buildObservation({ sessionId: this.sessionId, sensorId: this.sensorId, sequence: this.obsSeq++, ...fields });
    const p = validateMessage(msg);
    if (p) { this.onEvent({ type: "error", code: "CLIENT_INVALID_MESSAGE", message: `observation invalid: ${p}` }); return "invalid"; }
    const r = this.socket.send(msg, { droppable: true });
    if (r === "backpressure") this.frameSendDrops++;
    return r;
  }

  /** Orderly departure: DISCONNECT, then close; no reconnect afterwards. */
  stop(sendDisconnect = true, reason = "user_stop") {
    this._stopHeartbeat();
    if (sendDisconnect && this.registered) {
      this.socket.send(buildDisconnect({ sessionId: this.sessionId, sensorId: this.sensorId, clientTimeMs: nowEpochMs(), reason }));
    }
    this.registered = false;
    this.socket.stop();
  }
}
