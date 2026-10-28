// Observable sensor state + derivation of the single user-facing state label.
export const initialState = () => ({
  running: false,
  permission: "unknown",   // unknown | prompt | granted | denied
  camera: "idle",          // idle | starting | active | unavailable | stopped
  inference: "idle",       // idle | loading | ready | detecting | error
  connection: "idle",      // idle | connecting | registered | reconnecting | closed | rejected
  sensorId: null,
  reconnectCount: 0,
  error: null,
});

/** One label for the UI. Precedence mirrors what the user most needs to act on. */
export function deriveUiState(s) {
  if (!s.running) return "stopped";
  if (s.permission === "denied") return "permission_denied";
  if (s.camera === "unavailable") return "camera_unavailable";
  if (s.camera === "starting") return "requesting_camera";
  if (s.inference === "loading") return "model_loading";
  if (s.connection !== "registered") return "disconnected";
  if (s.inference === "detecting") return "detecting";
  return "connected";
}

export const UI_STATE_TEXT = {
  stopped: "Stopped", permission_denied: "Camera permission denied", camera_unavailable: "Camera unavailable",
  requesting_camera: "Waiting for camera permission…", model_loading: "Loading hand-landmark model…",
  disconnected: "Not connected to host", detecting: "Detecting & sending landmarks", connected: "Connected (idle)",
};

export class SensorStore {
  constructor() { this.state = initialState(); this.listeners = new Set(); }
  get() { return this.state; }
  update(patch) {
    const next = { ...this.state, ...patch };
    if (Object.keys(patch).every((k) => this.state[k] === next[k])) return;   // no-op updates do not notify
    this.state = next;
    for (const fn of this.listeners) fn(this.state);
  }
  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  reset() { this.state = initialState(); for (const fn of this.listeners) fn(this.state); }
}
