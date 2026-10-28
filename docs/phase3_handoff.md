# Phase 3 handoff: coordination + gesture

## 1. What changed in the repository
```
config/coordination.yaml            NEW  Contract Net / scoring / set-selection / adaptation settings
config/gesture.yaml                 NEW  pinch / reach / loss / ack settings
host/app/coordination/              NEW  (stdlib + NumPy + the existing spatial.geometry only)
    config.py          validated dataclass config (+ yaml loader)       messages.py      8 versioned coordination messages, strict offer parsing
    scoring.py         offer score + component derivations              sensor_agent.py  sensor self-assessment -> OBSERVATION_OFFER (+ viewing ray)
    set_selection.py   set policy (geometry, skew, rig, latency, ban)   coordinator.py   Contract Net state machine, leases, adaptation
    fusion_port.py     FusionOutcome.from_fusion_result (duck-typed)    audit.py         AuditLog (EventLogger-compatible), StageTimer
    simulation.py      deterministic harness + 9 scenarios              host_adapter.py  glue to a live Session (NOT executed here, see 5)
host/app/gesture/                   NEW
    config.py  events.py (INTERACTION_EVENT 1.0)  pinch.py (ratio + PinchDetector)  agent.py (GestureAgent)
host/app/spatial/fusion.py          EDIT additive: FusionEngine.sensor_filter + set_sensor_filter()  (default = unchanged behaviour)
host/app/spatial/service.py         EDIT additive: FusionService.set_sensor_filter() passthrough + log line
host/tests/  test_coord_{scoring,selection,protocol,adaptation}.py  test_gesture_{pinch,machine}.py  coord_helpers.py   NEW
host/tests/  test_fusion_sensor_filter.py  test_coord_engine_integration.py                                NEW (need pydantic)
scripts/demo_coordination.py        NEW  operator demonstration;   logs/demo/*.jsonl  sample audit logs
docs/coordination.md  docs/gesture.md  docs/phase3_handoff.md                                              NEW
```
Phase 1/2 message schemas, field names and units are untouched. Contract conflicts and their resolutions: `docs/coordination.md` section 1.

## 2. Exact commands and ACTUAL results
**Important: this was built in a sandbox with no network, and `pydantic`, `fastapi` and `pytest` could not be installed.** Consequences:
* The Phase 3 core was deliberately written without those dependencies and **was executed here**, using a small throw-away `pytest` stand-in (`raises`, `approx`, `parametrize`, `importorskip` only; it is not part of the repo).
* The existing Phase 1/2 test suite **could not be run here**, so "Phase 1 and 2 tests continue to pass" is **not verified by me**. The only edits to Phase 1/2 code are two additive, default-off hooks (see 1); I read the surrounding code to check they are inert when unused, but you must run the suite.
* `test_fusion_sensor_filter.py`, `test_coord_engine_integration.py` and `host_adapter.py` need the real engine and were **written but not executed**.

Run on a real environment (Python 3.12, `pip install -r requirements-dev.txt`):
```
python -m pytest host/tests -q                                   # everything incl. Phase 1/2 + the two engine-dependent files
python -m pytest host/tests/test_coord_*.py host/tests/test_gesture_*.py -q      # Phase 3 only
python scripts/demo_coordination.py                              # all scenarios narrated (or: <scenario name>, --gesture)
```
Result of the Phase 3 tests that were executed here (stand-in runner, `python /tmp/shim/run_tests.py`):
```
test_coord_scoring      40 passed        test_coord_protocol    16 passed       test_gesture_pinch    19 passed
test_coord_selection    14 passed        test_coord_adaptation  16 passed       test_gesture_machine  29 passed
TOTAL 134 passed, 0 failed  (parametrized cases counted individually);  test_coord_engine_integration: skipped (needs pydantic)
```
Mapping to the 12 acceptance criteria: (1) scoring tests · (2) `test_offer_*`, protocol rejection tests · (3)(4) selection tests, scenario 1/1b · (5) scenarios 2, 4, disconnect/silence/calibration tests · (6) scenarios 5a/5b ·
(7) lease/duplicate/delayed tests · (8)(9)(10)(11) gesture tests · (12) **not verified here**.

## 3. Example negotiation logs (real output of `scripts/demo_coordination.py`)
### Scenario 1: highest individual bid cannot be paired with a near-colocated camera
```
====================================================================================================
SCENARIO 1_geometry_beats_individual_score
====================================================================================================
+    0 ms  ANNOUNCE   task_00001 target=wrist reason=INITIAL attempt=1 deadline=+400 ms asking=['cam_a', 'cam_b', 'cam_c', 'cam_d'] unavailable=[]
+   10 ms  SELECT     task_00001 offers: cam_a=0.954, cam_b=0.939, cam_c=0.879, cam_d=0.849
                         cam_a+cam_c          score=0.958 angle=60.6deg
                         cam_b+cam_c          score=0.955 angle=56.8deg
                         cam_a+cam_d          score=0.951 angle=45.2deg
                         cam_b+cam_d          score=0.947 angle=43.1deg
                         cam_a+cam_b+cam_c    score=0.942 angle=60.6deg
                         cam_a+cam_b+cam_d    score=0.937 angle=45.2deg
                         cam_c+cam_d          score=0.932 angle=45.9deg
                         cam_a+cam_c+cam_d    score=0.927 angle=60.6deg
                         cam_b+cam_c+cam_d    score=0.924 angle=56.8deg
                         cam_a+cam_b          REJECTED INSUFFICIENT_VIEW_ANGLE angle=3.8deg
                         => AWARD ['cam_a', 'cam_c']  [best of 9 feasible set(s) out of 10]
+   10 ms  AWARD      award_00001 sensors=['cam_a', 'cam_c'] mode=TRIANGULATION trusted_3d=True lease=4000 ms
+   70 ms  FUSION     ACCEPTED (TRACKING_MULTI_VIEW) rms=0.00px reason=None consecutive_poor=0
final state=ACTIVE_LEASE lease=('cam_a', 'cam_c') counters={'offers_accepted': 4, 'awards': 1, 'fusion_verdict:ACCEPTED': 5}
```
### Scenario 1b: only poorly-spread cameras: explicit "no trusted 3D" with a reason
```
====================================================================================================
SCENARIO 1b_only_poor_geometry
====================================================================================================
+    0 ms  ANNOUNCE   task_00001 target=wrist reason=INITIAL attempt=1 deadline=+400 ms asking=['cam_a', 'cam_b'] unavailable=[]
+   10 ms  SELECT     task_00001 offers: cam_a=0.954, cam_b=0.939
                         cam_a+cam_b          REJECTED INSUFFICIENT_VIEW_ANGLE angle=3.8deg
                         => NO_SET_INSUFFICIENT_GEOMETRY   [no feasible set; first-failure counts: INSUFFICIENT_VIEW_ANGLEx1]
+   10 ms  TASK_REJECT NO_SET_INSUFFICIENT_GEOMETRY no feasible set; first-failure counts: INSUFFICIENT_VIEW_ANGLEx1
+   10 ms  DEGRADED   NO_SET_INSUFFICIENT_GEOMETRY no feasible set; first-failure counts: INSUFFICIENT_VIEW_ANGLEx1
final state=DEGRADED lease=None counters={'offers_accepted': 2, 'task_rejected': 1, 'degraded_entered': 1}
```
### Scenario 5a: excessive reprojection error → new set
```
====================================================================================================
SCENARIO 5a_fusion_failure_recoverable
====================================================================================================
+    0 ms  ANNOUNCE   task_00001 target=wrist reason=INITIAL attempt=1 deadline=+400 ms asking=['cam_a', 'cam_b', 'cam_c', 'cam_d'] unavailable=[]
+   10 ms  SELECT     task_00001 offers: cam_a=0.954, cam_b=0.939, cam_c=0.879, cam_d=0.849
                         cam_a+cam_c          score=0.958 angle=60.6deg
                         cam_b+cam_c          score=0.955 angle=56.8deg
                         cam_a+cam_d          score=0.951 angle=45.2deg
                         cam_b+cam_d          score=0.947 angle=43.1deg
                         cam_a+cam_b+cam_c    score=0.942 angle=60.6deg
                         cam_a+cam_b+cam_d    score=0.937 angle=45.2deg
                         cam_c+cam_d          score=0.932 angle=45.9deg
                         cam_a+cam_c+cam_d    score=0.927 angle=60.6deg
                         cam_b+cam_c+cam_d    score=0.924 angle=56.8deg
                         cam_a+cam_b          REJECTED INSUFFICIENT_VIEW_ANGLE angle=3.8deg
                         => AWARD ['cam_a', 'cam_c']  [best of 9 feasible set(s) out of 10]
+   10 ms  AWARD      award_00001 sensors=['cam_a', 'cam_c'] mode=TRIANGULATION trusted_3d=True lease=4000 ms
+   70 ms  FUSION     ACCEPTED (TRACKING_MULTI_VIEW) rms=0.00px reason=None consecutive_poor=0
+  420 ms  FUSION     POOR_QUALITY (FUSION_OUTLIER) rms=14.65px reason=REPROJECTION_ERROR_HIGH consecutive_poor=1
+  560 ms  ADAPT      FUSION_FAILURE ['cam_a', 'cam_c'] 
+  560 ms  LEASE END  award_00001 ['cam_a', 'cam_c'] reason=FUSION_FAILURE
+  560 ms  ANNOUNCE   task_00002 target=wrist reason=RESELECT_FUSION_FAILURE attempt=2 deadline=+400 ms asking=['cam_a', 'cam_b', 'cam_c', 'cam_d'] unavailable=[]
+  570 ms  SELECT     task_00002 offers: cam_a=0.954, cam_b=0.939, cam_c=0.879, cam_d=0.849
                         cam_b+cam_c          score=0.955 angle=56.6deg
                         cam_a+cam_d          score=0.951 angle=45.2deg
                         cam_b+cam_d          score=0.947 angle=43.1deg
                         cam_a+cam_b+cam_c    score=0.942 angle=60.5deg
                         cam_a+cam_b+cam_d    score=0.937 angle=45.2deg
                         cam_c+cam_d          score=0.932 angle=43.5deg
                         cam_a+cam_c+cam_d    score=0.927 angle=60.5deg
                         cam_b+cam_c+cam_d    score=0.924 angle=56.6deg
                         cam_a+cam_b          REJECTED INSUFFICIENT_VIEW_ANGLE angle=3.8deg
                         cam_a+cam_c          REJECTED RECENTLY_FAILED_SET angle=60.5deg
                         => AWARD ['cam_b', 'cam_c']  [best of 8 feasible set(s) out of 10; highest individual sensor cam_a excluded by set rules]
+  570 ms  AWARD      award_00002 sensors=['cam_b', 'cam_c'] mode=TRIANGULATION trusted_3d=True lease=4000 ms
+  630 ms  FUSION     POOR_QUALITY (FUSION_OUTLIER) rms=14.95px reason=REPROJECTION_ERROR_HIGH consecutive_poor=1
+  770 ms  ADAPT      FUSION_FAILURE ['cam_b', 'cam_c'] 
+  770 ms  LEASE END  award_00002 ['cam_b', 'cam_c'] reason=FUSION_FAILURE
+  770 ms  ANNOUNCE   task_00003 target=wrist reason=RESELECT_FUSION_FAILURE attempt=3 deadline=+400 ms asking=['cam_a', 'cam_b', 'cam_c', 'cam_d'] unavailable=[]
+  780 ms  SELECT     task_00003 offers: cam_a=0.954, cam_b=0.939, cam_c=0.879, cam_d=0.849
                         cam_a+cam_d          score=0.951 angle=45.2deg
                         cam_b+cam_d          score=0.947 angle=43.1deg
                         cam_a+cam_b+cam_c    score=0.942 angle=60.5deg
                         cam_a+cam_b+cam_d    score=0.937 angle=45.2deg
                         cam_c+cam_d          score=0.932 angle=43.5deg
                         cam_a+cam_c+cam_d    score=0.927 angle=60.5deg
                         cam_b+cam_c+cam_d    score=0.924 angle=56.6deg
                         cam_a+cam_b          REJECTED INSUFFICIENT_VIEW_ANGLE angle=3.8deg
                         cam_a+cam_c          REJECTED RECENTLY_FAILED_SET angle=60.5deg
                         cam_b+cam_c          REJECTED RECENTLY_FAILED_SET angle=56.6deg
                         => AWARD ['cam_a', 'cam_d']  [best of 7 feasible set(s) out of 10]
+  780 ms  AWARD      award_00003 sensors=['cam_a', 'cam_d'] mode=TRIANGULATION trusted_3d=True lease=4000 ms
+  840 ms  FUSION     ACCEPTED (TRACKING_MULTI_VIEW) rms=0.00px reason=None consecutive_poor=0
final state=ACTIVE_LEASE lease=('cam_a', 'cam_d') counters={'offers_accepted': 12, 'awards': 3, 'fusion_verdict:ACCEPTED': 56, 'fusion_verdict:POOR_QUALITY': 6}
```
### Scenario 5b: nothing is consistent → bounded retries, DEGRADED, never a trusted result
```
====================================================================================================
SCENARIO 5b_fusion_failure_unrecoverable
====================================================================================================
+    0 ms  ANNOUNCE   task_00001 target=wrist reason=INITIAL attempt=1 deadline=+400 ms asking=['cam_a', 'cam_b', 'cam_c', 'cam_d'] unavailable=[]
+   10 ms  SELECT     task_00001 offers: cam_a=0.954, cam_b=0.939, cam_c=0.879, cam_d=0.849
                         cam_a+cam_c          score=0.958 angle=60.6deg
                         cam_b+cam_c          score=0.955 angle=56.8deg
                         cam_a+cam_d          score=0.951 angle=45.2deg
                         cam_b+cam_d          score=0.947 angle=43.1deg
                         cam_a+cam_b+cam_c    score=0.942 angle=60.6deg
                         cam_a+cam_b+cam_d    score=0.937 angle=45.2deg
                         cam_c+cam_d          score=0.932 angle=45.9deg
                         cam_a+cam_c+cam_d    score=0.927 angle=60.6deg
                         cam_b+cam_c+cam_d    score=0.924 angle=56.8deg
                         cam_a+cam_b          REJECTED INSUFFICIENT_VIEW_ANGLE angle=3.8deg
                         => AWARD ['cam_a', 'cam_c']  [best of 9 feasible set(s) out of 10]
+   10 ms  AWARD      award_00001 sensors=['cam_a', 'cam_c'] mode=TRIANGULATION trusted_3d=True lease=4000 ms
+   70 ms  FUSION     ACCEPTED (TRACKING_MULTI_VIEW) rms=0.00px reason=None consecutive_poor=0
+  420 ms  FUSION     POOR_QUALITY (FUSION_OUTLIER) rms=9.57px reason=REPROJECTION_ERROR_HIGH consecutive_poor=1
+  560 ms  ADAPT      FUSION_FAILURE ['cam_a', 'cam_c'] 
+  560 ms  LEASE END  award_00001 ['cam_a', 'cam_c'] reason=FUSION_FAILURE
+  560 ms  ANNOUNCE   task_00002 target=wrist reason=RESELECT_FUSION_FAILURE attempt=2 deadline=+400 ms asking=['cam_a', 'cam_b', 'cam_c', 'cam_d'] unavailable=[]
   ... (repeats with exponential back-off; the run never produces a trusted result)
```
Scenarios 2 (occlusion), 3 (stale/late), 4/4b (disconnect/silence) and 6 (renewal) are in `logs/demo/` and in `python scripts/demo_coordination.py`.

## 4. Gesture trace (safe cancellation on tracking loss)
```
GESTURE TRACE (HOVER -> GRABBED -> tracking lost -> safe cancellation)
====================================================================================================
+  120 ms open hand in reach     HOVER_ENTER     state=HOVER    reason=in_reach pos=(0.05, 0.0, 0.0)
+  420 ms pinching               GRAB_REQUEST    state=GRABBED  reason=pinch_confirmed pos=(0.05, 0.0, 0.0)
+  420 ms pinching               GRAB_CONFIRMED  state=GRABBED  reason=confirmed pos=(0.05, 0.0, 0.0)
+  540 ms (continuous stream)    MOVE_TARGET     pos=(0.05, 0.0, 0.0)
+  600 ms (continuous stream)    MOVE_TARGET     pos=(0.06, 0.0, 0.0)
+  660 ms (continuous stream)    MOVE_TARGET     pos=(0.07, 0.0, 0.0)
+  780 ms fusion INVALID         TRACKING_LOST   state=GRABBED  reason=FUSION_INVALID
+ 1080 ms fusion INVALID         GRAB_CANCELLED  state=RELEASED reason=TRACKING_LOST_TIMEOUT
final gesture state = IDLE; counters = {'transition:IDLE->HOVER': 1, 'event:GESTURE_TRANSITION': 4, 'event:HOVER_ENTER': 1, 'event:INTERACTION_EVENT': 5, 'transition:HOVER->GRABBED': 1, 'event:GRAB_REQUEST': 1, 'event:GRAB_CONFIRMED': 1, 'event:MOVE_TARGET': 4, 'event:TRACKING_LOST': 1, 'tracking_loss_incidents': 1, 'transition:GRABBED->RELEASED': 1, 'event:GRAB_CANCELLED': 1, 'transition:RELEASED->IDLE': 1}

```
Transition table: `docs/gesture.md`.

## 5. Known limitations (read before relying on this)
1. **Not run against the real engine or hardware.** Scenario numbers come from `OfflineFusion`, a stand-in that uses the same `geometry` functions and status names as `FusionEngine` but none of its buffering/smoothing. Run `test_coord_engine_integration.py` and then the scenarios against the real engine/rig.
2. `host_adapter.py` (live Session glue) has not been executed; its attribute names were checked against the source by reading only. It is **not wired into `main.py`/`Session`**; wiring (a tick loop calling `SessionCoordination.tick()` and the gesture feeder) is Phase 4 work. Coordination config is not part of `Settings` (separate YAML, loaded by `load_coordination_config`).
3. All weights and thresholds are untuned defaults; the offer formula is a baseline, not a proven optimum.
4. Two-view reprojection error cannot expose a landmark error along the epipolar direction, and a failed 2-view set cannot identify its bad camera (the whole set is banned for 3 s).
5. Offers use the newest buffered observation, not a fresh on-demand capture; `sync_score` is a jitter proxy for clock-offset error, not a measured offset error.
6. Pinch is measured from 2D landmarks of one or more views (median); there is no 3D thumb/index distance. One hand/target only.
7. Disconnection tolerance is a design choice (500 ms grace for reported disconnects, 1500 ms silence timeout); reconnect within grace keeps the lease.

## 6. Stable interfaces for Phase 4 (world adapter / Godot)
* **Consume:** `INTERACTION_EVENT` 1.0 (`host/app/gesture/events.py`): discrete events + the separate `MOVE_TARGET` stream. De-duplicate by `event_id` (`EventDeduper`); ignore lower `sequence_number`.
* **Produce (acks):** `GestureAgent.acknowledge(event_id, now_ms, accepted=True)` for `GRAB_REQUEST` (set `require_adapter_ack: true`) and `RELEASE` (`release_ack_required: true`).
* **Supply:** `objects: Sequence[InteractionObject(object_id, position_xyz_m, grabbable)]` in the same world frame as `POSE_UPDATE.world_xyz_m`, on every `GestureAgent.update`.
* **Feed:** `HandSample.from_fusion_result(result, pinch_ratio, track_epoch, calibration_valid)`; compute `pinch_ratio` with `gesture.pinch.pinch_ratio` + `combine_view_ratios`; bump `track_epoch` on hand switch/smoother reset.
* **Lifecycle:** `GestureAgent.on_session_stop`, `on_calibration_invalid/restored`, `reset`; `Coordinator.stop/release/on_calibration_invalid`; call `tick()` regularly (~20 Hz).
* **Consumer rules:** on `TRACKING_LOST` freeze the object; never extrapolate; on `GRAB_CANCELLED`/`RELEASE` let the object fall or behave normally in place; positions exist only when `hand_target_xyz_m` is non-null.
* **Coordination:** `Coordinator(session_id, cfg, clock, fusion_gate=FusionService.set_sensor_filter)`; subscribe to its outbox for `TASK_AWARD / LEASE_EXPIRED / DEGRADED_STATUS` (a UI can show "tracking not trusted" from `DEGRADED_STATUS.tracking_trusted`).
