# SAGA 3D — Phase 2: calibration and spatial fusion

Goal: a **reliable 3D estimate of one landmark (the wrist)** from ≥ 2 calibrated cameras, and an unambiguous signal
whenever that estimate is *not* available. MediaPipe's per-camera model-relative 3D landmarks are **not** used for
geometry: they live in separate, unregistered frames. Fusion uses only the 2-D pixel landmarks plus calibrated geometry.

> Everything numeric in this document that is labelled **synthetic** comes from rendered/projected scenes with known
> ground truth. It validates the maths and the software, **not** physical accuracy. No physical camera or board was
> available while this phase was written; see "Hardware validation still required".

## 1. Module map and interfaces

```
host/app/spatial/
  geometry.py            pure maths: CameraModel, project, undistort, DLT triangulation, Gauss-Newton refine, depth/angle/residual checks
  calibration_schema.py  versioned, validated artifacts (IntrinsicsArtifact, CalibrationArtifact)
  calibration_store.py   save/load (atomic, history), invalidation, rig compatibility, CalibrationRegistry (sensor_id <-> camera_id)
  calibration_tool.py    OpenCV workflow: corner detection, intrinsics, shared-frame extrinsics (PnP), rig cross-check, accuracy report
  synchronizer.py        bounded per-sensor buffers, capture-time matching, staleness/order/clock checks
  fusion.py              FusionEngine: decision table, triangulation pipeline, outlier handling, quality, hold/loss
  filters.py             One Euro smoother + TrackSmoother (hold/expiry/reset) + lag measurement
  results.py             FusionResult (message_type POSE_UPDATE) - the contract Phase 3/4 consume
  service.py             per-session glue to the Phase 1 host (FusionService)
  synthetic.py           deterministic synthetic scenes (tests, selftest, example rig) - NOT for production use
scripts/calibrate.py     operator CLI;  scripts/e2e_fusion_live.py  live synthetic end-to-end check
```

Data flow (Phase 1 interfaces are unchanged; fusion is *additive*):

```
WebSocket handler --accepted observation entry--> FusionService.on_observation(entry, sensor_record)
   entry = Phase-1 dict: sensor_id, sequence_number, frame_id, captured_at_ms, received_at_ms, image_width/height,
           status, hand, landmarks[{id,x_px,y_px,...}]            (sensor_record.offset = per-sensor clock offset)
 -> ObservationSynchronizer.add   (order, staleness, clock sanity, bounded buffer; effective_ms = captured_at_ms + offset)
 -> CalibrationRegistry.resolve   (sensor_id -> camera_id -> validated artifact; image size must match)
 -> ObservationSynchronizer.match (>= 2 views inside max_skew_ms)
 -> FusionEngine._triangulate     (DLT -> refine -> depth/angle/residual checks -> smoothing)
 -> FusionResult --> subscribers (Phase 3/4), GET /api/sessions/{id}/fusion, logs/sessions/<id>/fusion.jsonl
FusionService.tick() every 50 ms  -> tracking-loss timing (no observations needed)
```
A fusion bug cannot break sensor ingestion: the handler catches and logs it (`fusion_error` event). Phase 1 message
fields, error codes and tests are unchanged; `POSE_UPDATE` is a new *host-internal* result type that sensors never receive.

## 2. Conventions (one set, used everywhere)

| Item | Convention |
|---|---|
| World frame | Right-handed, **metres**. Defined by the reference board: origin = board corner index 0, **X** along the corner columns, **Y** along the rows, **Z = X × Y (points into the board, away from the cameras that see its face)**. Optionally re-expressed with a `board_to_world` rigid transform (e.g. Z up). |
| Extrinsics | **world → camera**: `X_cam = R · X_world + t` (OpenCV `rvec/tvec`). Stored as `extrinsics.convention = "world_to_camera"`; camera → world (`Rᵀ`) and camera centre (`−Rᵀt`) are stored too and **cross-checked on load**. |
| Camera axes | +x right, +y down, +z forward; visible points have `X_cam[2] > 0`. |
| Pixels | u right, v down. OpenCV centre-origin: the centre of the top-left pixel is (0,0). Phase 1 / MediaPipe `x_px = x_norm · width` is **edge-origin**, so fusion applies `u_cv = x_px − 0.5` (`spatial.subtract_half_pixel`). |
| Projection | `s·[u v 1]ᵀ = K · distort(normalize(R·X + t))`, OpenCV 5-coefficient model `(k1,k2,p1,p2,k3)`. Standard (non-fisheye) model only. |
| Image size | Calibration is valid **only** for the exact `image_width × image_height` it was made at, in the same orientation. |

## 3. Operator procedure (reproducible)

**Equipment.** A flat, rigid printed checkerboard, **odd × even inner corners** (default 9 × 6 = 10 × 7 squares).
Odd × even matters: the board is then not symmetric under 180° rotation, so every camera agrees which corner is the
origin even when it sees the board upside down. *Measure the printed square size with calipers* — a 1 % error in square
size is a 1 % scale error in every result. Print at 100 % scale on a flat backing; tripods or fixed mounts for phones.

**Steps** (use the project venv: `.venv/bin/python scripts/calibrate.py …`):

1. **Capture intrinsic images**, per camera, **at the exact streaming resolution and orientation.** On the sensor page
   press **Start**, then **Save calibration frame** (a local PNG named `<sensor>_<W>x<H>_<time>.png`; nothing is uploaded).
   Do not use the phone's photo mode: it can use a different resolution, crop and field of view. Take ≥ 15 frames with the board
   at varied positions (cover corners and edges), distances and tilts (±15–40°); keep focus fixed if the device allows it.
   (USB webcams: `calibrate.py capture --device 0 --out DIR`.)
2. **Intrinsics:**
   `calibrate.py intrinsics --camera-id phone_left --images DIR --pattern 9x6 --square-size-m 0.040`
   → `calibration/intrinsics/phone_left__640x480.json`. It reports RMS, rejected images (with reasons), coverage and tilt
   spread, and **refuses** < 10 usable images or RMS > 1 px. Repeat for every camera. Distortion is estimated with `k3` fixed to 0 by default.
3. **Fix the rig** (cameras on tripods in their final positions). Place the reference board in the shared workspace,
   visible to **all** cameras, tilted 20–45° (a fronto-parallel board has an ambiguous pose). **Do not move cameras or
   board from now on.** Capture 2–5 stills per camera (Save calibration frame), then:
   `calibrate.py rig --camera phone_left=calibration/intrinsics/phone_left__640x480.json:REF_LEFT_DIR --camera phone_right=…:REF_RIGHT_DIR`
   Each camera's pose relative to the **same static board** is solved with `solvePnP` (IPPE + LM refinement), so all cameras
   share the board's frame without any camera seeing another. All artifacts of one run share a `rig_calibration_id`.
   Warnings: board moved between reference images, pose ambiguity. It prints a **rig cross-check**: board corners
   triangulated from camera pairs vs the known geometry (internal consistency only).
4. **Inspect:** `calibrate.py inspect` → per-camera status, warnings, compatibility; exit code 0 only if ≥ 2 usable
   compatible cameras.
5. **Bind sensors to cameras.** Session sensor ids (`sensor_01`…) are session-local; calibrations belong to physical cameras
   (`phone_left`). A sensor whose id equals a camera id is bound automatically (start the sensor with that requested id), otherwise
   bind in the operator page, or `POST /api/sessions/{id}/calibration/bindings {"sensor_id":"sensor_01","camera_id":"phone_left"}`.
6. **Verify with a physical point** (the only real accuracy evidence): mark a point at a measured world position
   (e.g. a target on the board plane at a known corner: `X = col·square, Y = row·square, Z = 0`), hold the wrist/marker there, record a
   session, then `calibrate.py accuracy --fusion-log logs/sessions/<id>/fusion.jsonl --truth 0.12,0.08,0` (mean/median/P95/max error,
   repeatability, bias, valid fraction). Repeat at ≥ 5 locations × ≥ 30 samples as in the project's test protocol (§18.6).

**When a camera moves, is replaced, zooms, or its resolution/orientation changes** its calibration is invalid:
* Resolution/orientation change is detected automatically (`DIMENSION_MISMATCH` → that camera is excluded).
* A physical move cannot be detected automatically in Phase 2: press **Invalidate camera** in the operator page (or
  `calibrate.py invalidate --camera-id phone_left`). The file is marked invalid **on disk** and fusion refuses it, also after a restart.
* **Recalibrate the whole rig** (steps 3–4), not just one camera. Cameras calibrated in different reference-board sessions have different
  `rig_calibration_id`s and are refused as `INCOMPATIBLE_RIG`, because a board that moved between sessions would silently shift
  one camera's world frame. Then press **Reload files**. Previous artifacts are kept in `calibration/<rig>/history/`.

## 4. Calibration artifact (`CalibrationArtifact`, `calibration_version` 1.0)

JSON file `calibration/<rig_id>/<camera_id>.json`, validated on every load (a corrupt/incompatible file is reported by name and
never silently used). Fields: `calibration_version`, `calibration_id`, `created_at`, `camera_id` (physical), `rig_id`,
`rig_calibration_id`, `world_frame_id`, `unit` (`"m"`), `image_width/height`, `target{type,inner_corners_cols/rows,square_size,unit}`,
`camera_matrix` (K), `distortion` (5), `extrinsics{convention:"world_to_camera", rotation, translation, camera_to_world_rotation, camera_center_world}`,
`projection_matrix` (3×4 = K[R|t]), `quality{intrinsic_rms_px, extrinsic_rms_px, samples_used, samples_rejected[{name,reason,error_px}], per_sample_rms_px, coverage_fraction, tilt_spread_deg, rig_check}`,
`validity{status valid|invalid, warnings[], reason, invalidated_at}`, `synthetic`, `intrinsics_id`, `notes`.
Validation on load: K form `[[fx,s,cx],[0,fy,cy],[0,0,1]]`, finite/non-singular/well-conditioned, principal point inside the image,
`fx/fy` plausible; R orthonormal with det +1; derived camera→world, centre and P consistent with K,R,t; supported version; no unknown fields.
`synthetic: true` artifacts are **refused** unless `spatial.allow_synthetic_calibration` is set. A sample (synthetic) rig is in
`calibration/example_synthetic/` — generate/refresh with `calibrate.py example`.

## 5. Observation synchronization

* **Time model.** Each observation has `captured_at_ms` (sensor clock) and `received_at_ms` (host clock). Phones are *not*
  clock-synchronized. Phase 1 estimates a per-sensor offset `host − sensor` from heartbeats (minimum of `host_rx − client_time`);
  fusion uses `effective_ms = captured_at_ms + offset` as the common timeline. `received_at_ms` is used only for transport-delay
  diagnostics and clock sanity checks, never as observation time. The residual inter-sensor error is roughly the difference between the
  sensors' minimum one-way delays (a few ms on a LAN); the **measured** skew of every fused set is reported.
* **Capture time precision.** The browser uses `requestVideoFrameCallback`'s `captureTime` when available, otherwise the callback time
  (later than capture by up to a frame). This bias is *per device* and not corrected in Phase 2.
* **Matching.** The best set of ≥ 2 sensors whose corrected capture times fit in `sync.max_skew_ms` (default **50 ms**), preferring
  more views, then the newest set. Otherwise `SYNC_DEGRADED` with the best achievable skew. At 1 m/s hand speed a 50 ms skew is up to 5 cm of
  position error: tune the window against measurements (`skew_ms` is in every result).
* **Rejections** (counted per sensor and reason in `counters`/`sync` diagnostics): `STALE` (> `max_observation_age_ms`, 250 ms),
  `OUT_OF_ORDER_SEQUENCE`, `OUT_OF_ORDER_TIME`, `CLOCK_SUSPECT` (received before captured), `NO_CLOCK_OFFSET`.
  Missing observations show as sequence gaps; per-sensor buffers hold ≤ 16 observations (oldest dropped, drops counted).
  A `no_hand` observation hides older tracked ones from that sensor.

## 6. Triangulation pipeline (`FusionEngine._triangulate`)

1 schema/identity (Phase 1 already validated the message; the engine checks sensor→camera) · 2 valid, compatible, size-matching calibration ·
3 pixels → OpenCV convention (−0.5 px) · 4 lens-distortion removal (`undistortPointsIter`, 1e-14 tolerance) · 5 same landmark and consistent
hand labels (`require_hand_agreement`; `unknown` is a wildcard; single-participant assumption) · 6 capture-time window · 7 camera matrices `[R|t]` on normalized
coordinates · 8 N-view **DLT** (SVD) + Gauss-Newton refinement of reprojection error (never worse than DLT) · 9 homogeneous validity: `|w|/‖Xh‖ ≥ 1e-9`
(else `POINT_AT_INFINITY`) · 10 reject non-finite, depth ≤ 0 (`NEGATIVE_DEPTH`) or outside [0.05, 20] m in **any** contributing camera ·
11–12 reproject into every camera *with distortion* → per-camera and RMS residual · 13 reject if RMS > 6 px or any view > 12 px
(`FUSION_OUTLIER`); with ≥ 3 views, drop the single worst view if the rest are consistent (warning `OUTLIER_VIEW_DROPPED:<sensor>`) ·
ray-angle check (`BASELINE_TOO_SMALL` below 2°) · 14 quality score · 15 smoothing **after** validation · 16 publish.

**Quality score** ∈ [0,1] = `reprojection × sync × geometry` with `reprojection = 1 − RMS/6 px`, `sync = 1 − skew/50 ms`,
`geometry = min(1, max ray angle / 20°)`. It is a heuristic ranking, not a probability; no per-landmark detector confidence is used
because the hand model provides none (it is never invented). No covariance/uncertainty is published (not implemented).

## 7. Fusion result — the contract for Phase 3 and 4

`FusionResult` (`host/app/spatial/results.py`, JSON `message_type: "POSE_UPDATE"`, `protocol_version: "1.0"`). All fields always present (`null` when absent). Example: `docs/examples/fusion/pose_update_tracking.json`.

| Field | Meaning |
|---|---|
| `session_id`, `target_landmark` | e.g. `"wrist"` |
| `fusion_sequence` | strictly increasing per (session, target); drop anything not newer than what you hold |
| `status` / `valid` / `reason` / `message` | status enum (below); `valid` ⇔ `TRACKING_MULTI_VIEW`; `reason` machine code; `message` human explanation |
| `timestamp_ms`, `published_at_ms` | effective capture time of the fused set (host timeline) and publication time (host clock) |
| `unit`, `world_frame_id` | `"m"`; frame id of the calibration |
| `world_xyz_m` | smoothed position — **non-null only when `status == TRACKING_MULTI_VIEW`** |
| `raw_world_xyz_m` | unsmoothed triangulation of the same set (use for accuracy measurement) |
| `held_world_xyz_m`, `held_age_ms` | last valid estimate while inside `tracking.hold_ms`; **not current**; null otherwise |
| `single_view_ray` | `{sensor_id, origin_m, direction}` for `SINGLE_VIEW_FALLBACK` — a ray, never a point |
| `source_sensor_ids`, `sources[]` | contributing sensors; per sensor: `camera_id, calibration_id, sequence_number, frame_id, captured_at_ms, received_at_ms, effective_time_ms, age_ms, transport_delay_ms, reprojection_error_px, used, note` |
| `skew_ms`, `aggregate_reprojection_error_px`, `ray_angle_deg` | measured set quality |
| `quality{score,reprojection,sync,geometry}` | §6 |
| `calibration{rig_id, rig_calibration_id, world_frame_id, calibration_ids{sensor→id}}` | what was used |
| `smoothing_applied`, `warnings[]` | e.g. `OUTLIER_VIEW_DROPPED:…`, `TRACK_DISCONTINUITY`, `CALIBRATION_PROBLEM:<sensor>:<code>` |

**How to consume (rules):** act on `world_xyz_m` only when `valid`; treat every other status as "no position now". Use
`held_world_xyz_m` only for bounded tolerance of brief dropouts (e.g. don't drop a grab), never as a current position. Do not extrapolate.
A gesture agent should release/cancel on `TARGET_LOST`, `CALIBRATION_REQUIRED`, and sustained non-valid statuses.

In-process: `session.fusion.engine.subscribe(callback)` (callbacks receive `FusionResult`; exceptions are isolated), `session.fusion.engine.latest[target]`.
HTTP: `GET /api/sessions/{id}/fusion?recent=N` → `{latest, recent[≤50], counters, sync, smoother_resets}`. File: `logs/sessions/<id>/fusion.jsonl`
(one result per line; enabled only while `logging.save_landmark_events` is on — the master switch for movement data).

### Status vocabulary and mapping to earlier names

| New status | When | Phase-1 / project-doc name |
|---|---|---|
| `TRACKING_MULTI_VIEW` | valid calibrated multi-view triangulation | "TRACKING: MULTI-VIEW", `MULTI_VIEW_VALID` (§14.5 example) |
| `SYNC_DEGRADED` | ≥ 2 views but no common capture-time window | "SYNC DEGRADED" |
| `SINGLE_VIEW_FALLBACK` | exactly one usable view; ray only | "SINGLE-VIEW FALLBACK" |
| `INSUFFICIENT_VIEWS` | no usable observation, inside the hold window | `INSUFFICIENT_VIEWS` (§17) |
| `TARGET_LOST` | no usable observation and hold expired / never valid | "TARGET LOST" |
| `FUSION_OUTLIER` | inconsistent views / conflicting hand labels | `FUSION_OUTLIER` (§17) |
| `GEOMETRY_DEGENERATE` | parallel rays, behind a camera, non-finite, depth out of range | *(new)* |
| `CALIBRATION_REQUIRED` | no valid compatible calibration for the observing sensors | "RECALIBRATION REQUIRED" |

Decision order: calibration → no usable observation → one view → asynchronous → outlier/geometry → valid. `SYNC_DEGRADED` and `SINGLE_VIEW_FALLBACK` are
published only after persisting `sync.degrade_grace_ms` (120 ms), so interleaved arrivals of two phones do not flicker. Results are published on a
new fused set, on any status change, and once per `TARGET_LOST` episode (not at tick rate).

## 8. Smoothing, hold and tracking loss

One Euro filter (3-D, timestamps in seconds from the *measurement* time), applied only to validated points. Reset on: session reset, calibration or binding change,
hand-identity change, implausible jump (`tracking.max_speed_mps`, 8 m/s → warning `TRACK_DISCONTINUITY`), and track loss. Older/duplicate timestamps never move it
backwards. The last valid estimate is exposed as *held* for `hold_ms` (500) from its measurement time; then `TARGET_LOST` and nothing is extrapolated. A 50 ms tick drives
this even when no observations arrive.

**Measured smoothing cost (synthetic, 15 Hz; steady-state lag on a ramp, jitter = remaining fraction of 5 mm white noise):**

| min_cutoff / beta | lag @0.1 m/s | lag @0.5 m/s | lag @1 m/s | static jitter kept |
|---|---|---|---|---|
| 1.5 / 0.3 | 101 ms | 86 ms | 75 ms | 0.49 |
| 1.5 / 1.0 | 92 ms | 64 ms | 49 ms | 0.49 |
| **3.0 / 1.0 (default)** | **50 ms** | **42 ms** | **35 ms** | **0.62** |
| 3.0 / 3.0 | 45 ms | 31 ms | 23 ms | 0.63 |
| 5.0 / 1.0 | 31 ms | 28 ms | 25 ms | 0.72 |

The first guess (1.5 Hz) would have consumed most of the 150 ms end-to-end budget, so the default favours low lag. Re-measure with real noise;
set `smoothing.enabled: false` to bypass.

## 9. HTTP API additions

`GET /api/sessions/{id}/fusion` · `GET /api/sessions/{id}/calibration` · `POST …/calibration/bindings {sensor_id, camera_id|null}` ·
`POST …/calibration/invalidate {camera_id|sensor_id, reason}` · `POST …/calibration/reload`. State-changing calls honour the Origin policy. The operator page shows fusion state, bindings and these actions.
Environment: `SAGA_CALIBRATION_DIR`, `SAGA_RIG_ID`, `SAGA_ALLOW_SYNTHETIC_CALIBRATION`. Thresholds: `config/fusion.yaml` (all commented; **all are starting points**).

## 10. Measured results (SYNTHETIC — see the caveat at the top)

Reproduce: `scripts/calibrate.py selftest`, `scripts/run_tests.sh`, `scripts/e2e_fusion_live.py`.

**Calibration from rendered board images** (OpenCV detection on 640×480 renders with 1 gray-level noise, 16 intrinsic views + 3 reference views per camera):
intrinsic RMS 0.05–0.06 px, extrinsic RMS 0.07–0.08 px, focal 601.3–601.4 px (true 600), rotation error 0.11–0.16°, camera-centre error 2.5–3.2 mm at ~1.1 m,
rig cross-check mean 0.18 mm (max 0.8 mm). Distortion recovery (point level, test tolerance): k1 within 0.03, k2 within 0.12 of truth.

**End-to-end triangulation error with the *calibrated* cameras** (≈ 60° rig, f = 600 px; 300 random points per row):

| pixel noise | depth from cameras | mean | p95 | max |
|---|---|---|---|---|
| 0.0 px | 1.12 m | 0.55 mm | 0.59 mm | 0.59 mm |
| 0.0 px | 1.80 m | 1.68 mm | 1.87 mm | 1.96 mm |
| 0.0 px | 2.57 m | 6.18 mm | 6.53 mm | 6.61 mm |
| 0.0 px | 3.85 m | 18.66 mm | 19.41 mm | 19.73 mm |
| 0.5 px | 1.12 m | 1.57 mm | 2.82 mm | 3.79 mm |
| 0.5 px | 1.80 m | 3.56 mm | 6.93 mm | 11.27 mm |
| 0.5 px | 2.57 m | 8.16 mm | 16.42 mm | 26.41 mm |
| 0.5 px | 3.85 m | 20.98 mm | 41.04 mm | 64.20 mm |
| 1.0 px | 1.12 m | 3.03 mm | 5.50 mm | 7.69 mm |
| 1.0 px | 1.80 m | 6.53 mm | 12.78 mm | 21.20 mm |
| 1.0 px | 2.57 m | 13.07 mm | 27.90 mm | 46.77 mm |
| 1.0 px | 3.85 m | 29.77 mm | 63.42 mm | 110.01 mm |

Read this as: **error grows with distance** and with pixel noise; even with perfect pixels the small calibration errors above cost millimetres to centimetres.
Real wrist landmarks are noisier than 1 px (the hand model is not a corner detector), so real error will be larger; expect to measure it, not to assume 5 cm.
Noise-free geometry (exact cameras): error ≤ 1e-9 m (float precision).
Pure maths tests: DLT = OpenCV `triangulatePoints` to 1e-9 m; distortion round trip 6e-15 m.

**Live synthetic run** against the real host (real clock, 15 Hz, B staggered 6 ms): 75/75 frames fused, raw 3-D error vs the true moving path mean 5.5 mm / p95 11 mm
(includes the 6 ms stagger on a moving hand and example-rig calibration error), capture→published latency median 3–4 ms / p95 ≤ 5 ms (host + loopback only, **excludes** camera, inference and Wi-Fi),
measured skew 6–10 ms; loss sequence `INSUFFICIENT_VIEWS → TARGET_LOST` ≈ `hold_ms` after the last valid fix; recovery clean; single view gives a ray and no XYZ.
Compute: ≈ 0.5 ms per fused frame (single core, this sandbox), 27 µs per idle tick.

## 11. Limitations and hardware validation still required

Known limitations (by design or discovered while testing):

* **No physical validation.** No camera/board hardware was used. Nothing here supports a claim like "5 cm"; measure it (§3 step 6).
* **Two views cannot detect errors along the epipolar line.** With exactly two views a landmark error along the epipolar line looks like a different depth, so the reprojection
  residual stays ~0 while XYZ is wrong (there is a test documenting this). Three or more views are needed to detect it; the residual gate still catches perpendicular errors.
* **Timing.** No hardware sync; skew accuracy is limited by the delay-difference approximation and the browser capture-time fallback. A moving hand makes skew a position error (≈ speed × skew).
* **Hand identity.** Single participant; the left/right label is noisy and its mirroring convention is unverified on hardware (`flip_handedness`), so a label flip resets smoothing and conflicting labels are refused.
* **Camera motion is not detected automatically**; invalidate manually. Phone autofocus, OIS/EIS crop and zoom can change intrinsics between calibration and use.
* **Lens model** is the standard OpenCV model; ultra-wide/fisheye lenses need a different model (not implemented).
* **Board quality** (flatness, printed square error) maps directly into scale/shape errors. Camera pose from a fronto-parallel board is ambiguous (warned, not blocked).
* The half-pixel convention (`x_px − 0.5`) follows documented MediaPipe/OpenCV conventions but was not checked on hardware (impact ≤ 0.5 px).
* Calibration cannot detect a 180° board-orientation mistake on a **symmetric** (even × even) board; use odd × even.
* Single recalibrated camera ≠ rig: the whole rig must be recalibrated after any camera move (enforced via `INCOMPATIBLE_RIG`).
* OpenCV pinned to 4.14.0.94; 5.0.0 passed an API smoke test only.
* `capture` (webcam) is untested here; no GUI preview is provided.

Hardware validation required before trusting the system: (1) print/measure the board and calibrate each real phone at its streaming resolution; (2) confirm rig cross-check and
extrinsic RMS are small on real images; (3) measure known points at ≥ 5 locations (§3.6) and report mean/P95/repeatability with and without one camera occluded; (4) measure real inter-phone skew and
the effect of `max_skew_ms` on a moving hand; (5) confirm handedness mirroring and the half-pixel offset; (6) measure end-to-end latency (capture → Godot) on the real network; (7) re-tune thresholds
(`max_reprojection_error_px`, smoothing) on real landmark noise; (8) check calibration stability over a session (focus/stabilisation) .
