// SAGA 3D wire protocol v1.0 for the browser. Mirrors host/app/schemas/messages.py exactly;
// tests cross-check both implementations against docs/examples/*.json and against each other.
export const PROTOCOL_VERSION = "1.0";
export const LANDMARK_IDS = ["wrist", "thumb_tip", "index_tip", "index_mcp", "pinky_mcp"];
export const OUT_OF_FRAME_MARGIN = 0.1;
const NORM_TOLERANCE_PX = 1.0;
const MIN_EPOCH_MS = 1577836800000;
const MAX_EPOCH_MS = 4102444800000;
const MAX_SEQ = Number.MAX_SAFE_INTEGER;

export const nowEpochMs = () => Math.round(performance.timeOrigin + performance.now());

// ---- tiny declarative validator: each rule returns null (ok) or a problem string ---------------
const T = {
  str: (re = null, max = 256) => (v) =>
    typeof v === "string" && v.length <= max && (!re || re.test(v)) ? null : `must be a string${re ? ` matching ${re}` : ""} (max ${max})`,
  int: (min, max) => (v) => (Number.isInteger(v) && v >= min && v <= max ? null : `must be an integer in [${min}, ${max}]`),
  num: (min, max) => (v) => (typeof v === "number" && Number.isFinite(v) && v >= min && v <= max ? null : `must be a finite number in [${min}, ${max}]`),
  bool: () => (v) => (typeof v === "boolean" ? null : "must be a boolean"),
  lit: (...vals) => (v) => (vals.includes(v) ? null : `must be one of ${vals.join(", ")}`),
  nullable: (rule) => (v) => (v === null ? null : rule(v)),
  list: (rule, min, max) => (v) => {
    if (!Array.isArray(v) || v.length < min || v.length > max) return `must be a list of ${min}..${max} items`;
    for (let i = 0; i < v.length; i++) { const p = rule(v[i]); if (p) return `[${i}] ${p}`; }
    return null;
  },
  obj: (shape) => (v) => {
    if (v === null || typeof v !== "object" || Array.isArray(v)) return "must be an object";
    for (const k of Object.keys(v)) if (!(k in shape)) return `unexpected field '${k}'`;
    for (const [k, rule] of Object.entries(shape)) {
      if (!(k in v)) return `missing field '${k}'`;
      const p = rule(v[k]);
      if (p) return `${k}: ${p}`;
    }
    return null;
  },
};
const SESSION = T.str(/^[A-Za-z0-9_-]{4,64}$/, 64);
const SENSOR = T.str(/^[A-Za-z0-9_-]{1,32}$/, 32);
const CLIENT = T.str(/^[A-Za-z0-9_-]{8,64}$/, 64);
const EPOCH = T.int(MIN_EPOCH_MS, MAX_EPOCH_MS);
const SEQ = T.int(0, MAX_SEQ);
const DIM = T.int(1, 8192);
const UNIT = T.num(0, 1);
const LM_ID = T.lit(...LANDMARK_IDS);
const PERMISSION = T.lit("unknown", "prompt", "granted", "denied");
const CAMERA = T.lit("idle", "starting", "active", "unavailable", "stopped");
const INFERENCE = T.lit("idle", "loading", "ready", "detecting", "error");
const ENV = { protocol_version: T.lit(PROTOCOL_VERSION) };

const SHAPES = {
  HELLO: T.obj({
    ...ENV, message_type: T.lit("HELLO"), session_id: SESSION, client_id: CLIENT,
    requested_sensor_id: T.nullable(SENSOR), client_version: T.str(null, 64), model_version: T.nullable(T.str(null, 128)),
    client_time_ms: EPOCH,
    capabilities: T.obj({
      landmarks: T.list(LM_ID, 1, 32), max_update_hz: T.num(Number.MIN_VALUE, 120), inference: T.lit("on_device", "none"),
      camera: T.nullable(T.obj({ facing: T.lit("user", "environment", "unknown"), width: DIM, height: DIM })),
      user_agent: T.nullable(T.str(null, 256)),
    }),
  }),
  HEARTBEAT: T.obj({
    ...ENV, message_type: T.lit("HEARTBEAT"), session_id: SESSION, sensor_id: SENSOR, sequence_number: SEQ, client_time_ms: EPOCH,
    health: T.obj({ camera: CAMERA, inference: INFERENCE, observed_hz: T.nullable(T.num(0, 1000)), frames_skipped: SEQ, send_drops: SEQ }),
  }),
  SENSOR_STATUS: T.obj({
    ...ENV, message_type: T.lit("SENSOR_STATUS"), session_id: SESSION, sensor_id: SENSOR, client_time_ms: EPOCH,
    permission: PERMISSION, camera: CAMERA, inference: INFERENCE, connection: T.lit("connected", "reconnected"),
    reconnect_count: T.int(0, 1000000), detail: T.nullable(T.str(null, 200)),
  }),
  LANDMARK_OBSERVATION: T.obj({
    ...ENV, message_type: T.lit("LANDMARK_OBSERVATION"), session_id: SESSION, sensor_id: SENSOR, sequence_number: SEQ, frame_id: SEQ,
    captured_at_ms: EPOCH, image_width: DIM, image_height: DIM, status: T.lit("tracked", "no_hand"), hand: T.lit("left", "right", "unknown"),
    landmarks: T.list(
      T.obj({
        id: LM_ID, x_px: T.num(-1e6, 1e6), y_px: T.num(-1e6, 1e6), confidence: T.nullable(UNIT),
        x_norm: T.nullable(T.num(-1e6, 1e6)), y_norm: T.nullable(T.num(-1e6, 1e6)), z_model: T.nullable(T.num(-1e6, 1e6)),
      }), 0, 32),
    quality: T.obj({
      inference_ms: T.nullable(T.num(0, 10000)), handedness_score: T.nullable(UNIT),
      flags: T.list(T.lit("frames_dropped", "overloaded", "out_of_frame"), 0, 8),
    }),
  }),
  DISCONNECT: T.obj({
    ...ENV, message_type: T.lit("DISCONNECT"), session_id: SESSION, sensor_id: SENSOR, client_time_ms: EPOCH,
    reason: T.lit("user_stop", "page_unload", "error"),
  }),
  // host -> client
  REGISTERED: T.obj({
    ...ENV, message_type: T.lit("REGISTERED"), session_id: SESSION, sensor_id: SENSOR, resumed: T.bool(),
    heartbeat_interval_ms: T.int(100, 60000), target_update_hz: T.num(Number.MIN_VALUE, 120), max_update_hz: T.num(Number.MIN_VALUE, 120),
    required_landmarks: T.list(LM_ID, 0, 32), session_state: T.lit("active"), host_time_ms: EPOCH, host_version: T.str(null, 32),
  }),
  HEARTBEAT_ACK: T.obj({
    ...ENV, message_type: T.lit("HEARTBEAT_ACK"), session_id: SESSION, sensor_id: SENSOR, sequence_number: SEQ,
    client_time_ms: EPOCH, host_time_ms: EPOCH, sensor_state: T.lit("online", "degraded", "offline", "disconnected"),
  }),
  ERROR: T.obj({
    ...ENV, message_type: T.lit("ERROR"), session_id: T.nullable(SESSION), error_code: T.str(/^[A-Z_]{3,40}$/, 40),
    message: T.str(null, 300), related_message_type: T.nullable(T.str(null, 40)),
    details: T.nullable(T.list(T.obj({ field: T.str(null, 120), problem: T.str(null, 200) }), 0, 10)),
    fatal: T.bool(), host_time_ms: EPOCH,
  }),
  SESSION_STOP: T.obj({
    ...ENV, message_type: T.lit("SESSION_STOP"), session_id: SESSION, reason: T.lit("operator_request", "host_shutdown"), host_time_ms: EPOCH,
  }),
};
export const CLIENT_TYPES = ["HELLO", "HEARTBEAT", "SENSOR_STATUS", "LANDMARK_OBSERVATION", "DISCONNECT"];
export const HOST_TYPES = ["REGISTERED", "HEARTBEAT_ACK", "ERROR", "SESSION_STOP"];

function observationSemantics(m) {
  const w = m.image_width, h = m.image_height, mg = OUT_OF_FRAME_MARGIN;
  if (m.status === "no_hand") return m.landmarks.length ? "status 'no_hand' requires an empty landmarks list" : null;
  if (!m.landmarks.length) return "status 'tracked' requires at least one landmark";
  const seen = new Set();
  for (let i = 0; i < m.landmarks.length; i++) {
    const l = m.landmarks[i];
    if (seen.has(l.id)) return `landmarks[${i}]: duplicate landmark id '${l.id}'`;
    seen.add(l.id);
    if (!(l.x_px >= -mg * w && l.x_px <= (1 + mg) * w)) return `landmarks[${i}].x_px outside frame bounds`;
    if (!(l.y_px >= -mg * h && l.y_px <= (1 + mg) * h)) return `landmarks[${i}].y_px outside frame bounds`;
    if ((l.x_norm === null) !== (l.y_norm === null)) return `landmarks[${i}]: x_norm and y_norm must both be set or both null`;
    if (l.x_norm !== null) {
      if (!(l.x_norm >= -mg && l.x_norm <= 1 + mg && l.y_norm >= -mg && l.y_norm <= 1 + mg)) return `landmarks[${i}]: normalized coordinates out of range`;
      if (Math.abs(l.x_norm * w - l.x_px) > NORM_TOLERANCE_PX || Math.abs(l.y_norm * h - l.y_px) > NORM_TOLERANCE_PX) return `landmarks[${i}]: x_norm/y_norm disagree with pixel coordinates`;
    }
  }
  return null;
}

/** Validate a message of a known type. Returns null when valid, else a human-readable problem. */
export function validateMessage(msg) {
  if (msg === null || typeof msg !== "object") return "message must be an object";
  if (msg.protocol_version !== PROTOCOL_VERSION) return `unsupported protocol_version`;
  const rule = SHAPES[msg.message_type];
  if (!rule) return "unknown message_type";
  const p = rule(msg);
  if (p) return p;
  return msg.message_type === "LANDMARK_OBSERVATION" ? observationSemantics(msg) : null;
}

// ---- builders (always produce every field; absent values are explicit null) --------------------
const env = (type, sessionId) => ({ protocol_version: PROTOCOL_VERSION, message_type: type, session_id: sessionId });

export const buildHello = ({ sessionId, clientId, requestedSensorId = null, clientVersion, modelVersion = null, clientTimeMs, capabilities }) => ({
  ...env("HELLO", sessionId), client_id: clientId, requested_sensor_id: requestedSensorId, client_version: clientVersion,
  model_version: modelVersion, client_time_ms: clientTimeMs, capabilities,
});
export const buildHeartbeat = ({ sessionId, sensorId, sequence, clientTimeMs, health }) => ({
  ...env("HEARTBEAT", sessionId), sensor_id: sensorId, sequence_number: sequence, client_time_ms: clientTimeMs, health,
});
export const buildStatus = ({ sessionId, sensorId, clientTimeMs, permission, camera, inference, connection, reconnectCount, detail = null }) => ({
  ...env("SENSOR_STATUS", sessionId), sensor_id: sensorId, client_time_ms: clientTimeMs, permission, camera, inference,
  connection, reconnect_count: reconnectCount, detail,
});
export const buildObservation = ({ sessionId, sensorId, sequence, frameId, capturedAtMs, imageWidth, imageHeight, status, hand, landmarks, quality }) => ({
  ...env("LANDMARK_OBSERVATION", sessionId), sensor_id: sensorId, sequence_number: sequence, frame_id: frameId,
  captured_at_ms: capturedAtMs, image_width: imageWidth, image_height: imageHeight, status, hand, landmarks, quality,
});
export const buildDisconnect = ({ sessionId, sensorId, clientTimeMs, reason }) => ({
  ...env("DISCONNECT", sessionId), sensor_id: sensorId, client_time_ms: clientTimeMs, reason,
});
