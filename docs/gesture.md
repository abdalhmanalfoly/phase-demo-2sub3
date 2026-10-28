# Gesture Agent (Phase 3): deterministic state machine

Package `host/app/gesture/`. It consumes validated fusion outputs (`HandSample`) and emits versioned **interaction events**; it knows nothing about Godot.

## State model and transition table
Public states: `IDLE`, `HOVER`, `GRABBED`, `RELEASED`. The pinch *candidate* is internal to `PinchDetector` (`OPEN → CANDIDATE → PINCHED`) and never a public state.
At most **one** transition per `update`/`tick` call (except safe-stop, which runs to IDLE).

| From | To | Condition | Events |
|---|---|---|---|
| IDLE | HOVER | usable pose; grab-reach object within `reach_enter_m` (0.12 m); confidence ≥ `min_confidence` (0.5) over the last `confidence_history` (3) usable samples, at least 2 of them; not inhibited | `HOVER_ENTER` |
| HOVER | IDLE | beyond `reach_exit_m` (0.18 m) · tracking loss > `hover_loss_timeout_ms` (250) · object gone · track change · session stop · calibration invalid | `HOVER_EXIT` (+`TRACKING_LOST` at 100 ms of loss) |
| HOVER | GRABBED | pinch confirmed (below enter threshold for `confirm_ms` 120 with ≥3 samples, median-smoothed) **and** hover is *armed* (an open hand was seen while hovering) **and** object grabbable | `GRAB_REQUEST`, then `GRAB_CONFIRMED` (same update, or after adapter ack if `require_adapter_ack`) |
| GRABBED | RELEASED | pinch opened (≥ exit threshold for 50 ms) | `RELEASE` |
| GRABBED | RELEASED | **cancel:** tracking loss > `grab_loss_timeout_ms` (400) · no pinch evidence > 400 ms · re-acquired > 0.25 m from the held pose · track change · calibration invalid · session stop · adapter rejection/ack timeout · reset | `GRAB_CANCELLED(cancel_reason)` |
| RELEASED | IDLE | `release_settle_ms` (100) elapsed, or `acknowledge(RELEASE)` / `release_ack_timeout_ms` when `release_ack_required` | - |
| any | IDLE (+inhibited) | `on_session_stop()` / `on_calibration_invalid()` / sample with `calibration_valid=false` | cancel/exit event for the active state |

`reset()` cancels safely, starts a new event epoch (`…:g<epoch>:…` ids) and clears all history.

## Pinch detection
`ratio = |thumb_tip − index_tip| / |wrist − index_mcp|` in pixels of the **same view** (palm length cancels distance and resolution). Invalid input (missing/non-finite/low-confidence landmark, palm < 20 px) gives
`None` - never 0. Several views → median. Defaults: **enter 0.25 < exit 0.45** (validated at load), `confirm_ms` 120, `release_confirm_ms` 50, median window 3, ≥3 samples, a gap > 150 ms restarts a pending confirmation.
Between the thresholds nothing changes (hysteresis). A closed fist entering reach never grabs. These numbers are starting points; tune on hardware.

## Safety defaults (tracking loss)
Documented default while **GRABBED**: report `TRACKING_LOST` after 100 ms (consumer **freezes** the object, no `MOVE_TARGET`, no position in the event), recover if valid again within **400 ms** and within 0.25 m of the held pose,
otherwise `GRAB_CANCELLED(TRACKING_LOST_TIMEOUT)` → the object is released **in place** (no impulse implied). The timeout is validated ≤ 5 s so stale coordinates can never drive an object indefinitely.
A pose is used only if fusion `valid`, fresh (≤ 250 ms), finite, and confidence is known and ≥ `exit_confidence` (0.35). Unknown confidence is not good confidence.

## Interaction event contract (`INTERACTION_EVENT`, version 1.0)
Fields: `session_id`, `event_id` (`<session>:g<epoch>:<n>`), `sequence_number`, `event_type`, `timestamp_ms`, `gesture_state` (state **after** the transition), `continuous`, `interaction_id`, `target_object_id`,
`hand_target_xyz_m` (**null unless fusion is valid**), `world_frame_id`, `unit:"m"`, `confidence`, `cancel_reason`, `reason`, `sample_timestamp_ms`.
Types: `HOVER_ENTER`, `HOVER_EXIT`, `GRAB_REQUEST`, `GRAB_CONFIRMED`, `RELEASE`, `GRAB_CANCELLED`, `TRACKING_LOST` (discrete, at most once per episode, `EventDeduper` for consumers) and `MOVE_TARGET`
(continuous, ≤ `move_max_hz` 30, only while GRABBED+confirmed with a valid fresh pose; returned in `GestureOutput.moves`, separate from `events`). The constructor refuses contract violations (position on `TRACKING_LOST`, `MOVE_TARGET` without position, `cancel_reason` mismatch, NaN).
Cancel reasons: `TRACKING_LOST_TIMEOUT`, `PINCH_EVIDENCE_LOST`, `TRACK_DISCONTINUITY`, `TRACK_CHANGED`, `CALIBRATION_INVALID`, `SESSION_STOPPED`, `ADAPTER_REJECTED`, `ADAPTER_ACK_TIMEOUT`, `RESET`, `SAFETY`.

## Limits
Hand target is the fused `wrist`; the pinch is measured in image space from 2D landmarks, so there is no 3D pinch distance yet. One tracked hand; objects are supplied by the caller each update.
