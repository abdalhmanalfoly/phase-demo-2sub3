// Minimal operator/debug panel: polls the host status API once per second (single interval).
const $ = (id) => document.getElementById(id);
let sessionId = new URLSearchParams(location.search).get("session") || "";
let timer = null;

const td = (v) => { const c = document.createElement("td"); c.textContent = v ?? "—"; return c; };

async function poll() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("host-state").textContent = `Host: ok (v${h.host_version}, protocol ${h.protocol_version}), ${h.sessions_active} active session(s)`;
  } catch { $("host-state").textContent = "Host: UNREACHABLE"; return; }
  if (!sessionId) return;
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!r.ok) { $("s-state").textContent = r.status === 404 ? "unknown session" : `error ${r.status}`; return; }
  const s = await r.json();
  $("s-id").textContent = s.session_id; $("s-state").textContent = s.state;
  $("s-conn").textContent = s.sensors_connected; $("s-rej").textContent = s.messages_rejected;
  $("stop-btn").disabled = s.state !== "active";
  const body = $("sensors"); body.replaceChildren();
  for (const x of s.sensors) {
    const tr = document.createElement("tr");
    const st = document.createElement("td"); const pill = document.createElement("span");
    pill.className = `pill ${x.state}`; pill.textContent = x.state; pill.title = x.state_reason; st.append(pill);
    tr.append(td(x.sensor_id), st, td(x.permission), td(x.camera), td(x.inference), td(x.update_rate_hz),
      td(x.latest_observation_age_ms), td(x.latest_observation_capture_age_ms), td(x.hand ?? "—"),
      td(x.observations_accepted), td(x.observations_rejected), td(x.sequence_gaps), td(x.reconnect_count),
      td(x.last_error ? `${x.last_error.error_code}` : "—"));
    body.append(tr);
  }
  const ul = $("errors"); ul.replaceChildren();
  for (const e of [...s.recent_errors].reverse()) {
    const li = document.createElement("li");
    li.textContent = `${new Date(e.ts_ms).toLocaleTimeString()} ${e.sensor_id ?? "-"} ${e.error_code}: ${e.message}`;
    ul.append(li);
  }
}
const fmt = (v, d = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
async function pollSpatial() {
  if (!sessionId) return;
  const f = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/fusion`);
  if (f.ok) {
    const r = (await f.json()).latest?.wrist;
    const st = $("fusion-state");
    if (r) {
      st.textContent = `${r.status}${r.valid ? "" : " (not a trusted 3D position)"} — ${r.message}`;
      st.dataset.state = r.valid ? "detecting" : (r.status === "TARGET_LOST" || r.status === "CALIBRATION_REQUIRED" ? "permission_denied" : "disconnected");
      $("f-xyz").textContent = r.world_xyz_m ? r.world_xyz_m.map((v) => fmt(v)).join(", ") : "— (none: see status)";
      $("f-reason").textContent = r.reason ?? "—";
      $("f-metrics").textContent = `${fmt(r.skew_ms, 1)} ms / ${fmt(r.aggregate_reprojection_error_px, 2)} px / ${fmt(r.ray_angle_deg, 1)}°`;
      $("f-quality").textContent = r.quality?.score === null ? "—" : `${fmt(r.quality.score, 2)} (reproj ${fmt(r.quality.reprojection, 2)}, sync ${fmt(r.quality.sync, 2)}, geometry ${fmt(r.quality.geometry, 2)})`;
      $("f-held").textContent = r.held_world_xyz_m ? `${r.held_world_xyz_m.map((v) => fmt(v)).join(", ")} (${fmt(r.held_age_ms, 0)} ms old)` : "—";
      $("f-warn").textContent = r.warnings.length ? r.warnings.join("; ") : "—";
    } else { st.textContent = "no fusion result yet"; }
  }
  const c = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/calibration`);
  if (c.ok) {
    const cal = await c.json();
    $("cal-summary").textContent = `Rig ${cal.rig_id}: ${cal.report.summary}` + (Object.keys(cal.load_errors).length ? ` — LOAD ERRORS: ${JSON.stringify(cal.load_errors)}` : "");
    const body = $("cal-rows"); body.replaceChildren();
    for (const [sid, v] of Object.entries(cal.sensors)) {
      const tr = document.createElement("tr");
      tr.append(td(sid), td(v.camera_id), td(v.ok ? "yes" : "NO"), td(v.problem ? `${v.problem}: ${v.detail}` : "—"), td(v.calibration_id));
      body.append(tr);
    }
  }
}
async function post(path, body) {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/calibration/${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) $("cal-summary").textContent = `${path} failed: ${r.status} ${(await r.text()).slice(0, 160)}`;
  pollSpatial();
}
$("bind-btn").addEventListener("click", () => post("bindings", { sensor_id: $("bind-sensor").value.trim(), camera_id: $("bind-camera").value.trim() || null }));
$("invalidate-btn").addEventListener("click", () => { if (confirm("Mark this camera's calibration INVALID? Fusion will refuse it until you recalibrate.")) post("invalidate", { camera_id: $("bind-camera").value.trim() || null, sensor_id: $("bind-sensor").value.trim() || null, reason: "operator: camera moved or replaced" }); });
$("reload-btn").addEventListener("click", () => post("reload"));

function show(id) {
  sessionId = id; $("session-input").value = id;
  $("join").textContent = id ? `Join link: ${location.origin}/client/?session=${id}` : "Join link: —";
}
$("create-btn").addEventListener("click", async () => {
  const r = await fetch("/api/sessions", { method: "POST" });
  if (r.ok) show((await r.json()).session_id); else $("s-state").textContent = `create failed (${r.status})`;
  poll();
});
$("session-input").addEventListener("change", (e) => show(e.target.value.trim()));
$("stop-btn").addEventListener("click", async () => { await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/stop`, { method: "POST" }); poll(); });
show(sessionId); poll(); pollSpatial();
if (timer === null) timer = setInterval(() => { poll(); pollSpatial(); }, 1000);
