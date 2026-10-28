# Coordination Agent (Phase 3): Contract Net-style task allocation

Package `host/app/coordination/`. Deterministic, single process, no LLM, no new infrastructure. Every decision is a documented rule over
numbers that are logged, so "why did it pick set X over set Y" is answerable from the audit log alone.

## 1. Incompatibilities found in Phases 1-2 and how they are resolved

| Finding | Resolution |
|---|---|
| `SENSOR_STATUS` already exists as a **sensor→host wire message** (`schemas/messages.py`). | The coordination `SENSOR_STATUS` is a **host-internal** summary with `channel:"coordination"`; it is never parsed from, or written to, a socket. The wire schema is untouched. |
| Phase 2 has `POSE_UPDATE` (a `FusionResult`), no `FUSION_RESULT`. | Coordination `FUSION_RESULT` is the coordinator's **evaluation** of a `POSE_UPDATE` (`fusion_sequence`, verdict, reason). It carries no coordinates. `POSE_UPDATE` is unchanged. |
| Phase 2 `FusionEngine` fuses **every** active sensor; a "selected set" would otherwise have no effect. | One **additive** hook: `FusionEngine.set_sensor_filter(target, ids)` (+ `FusionService` passthrough). Absent key = unrestricted (identical Phase 2 behaviour); empty set = nothing may feed the target. Triangulation is **not** duplicated in the coordinator. |
| The hand model reports **no per-landmark confidence** (`confidence: null`). | `detector_confidence` is **imputed** (0.5, neutral) and the offer is flagged `detector_confidence_source:"imputed"`. Never silently excellent. |
| `pydantic` models are used for Phase 1/2 contracts. | Phase 3 uses frozen dataclasses with explicit validation (`MessageError` codes) so the core runs without the web stack; `FusionOutcome.from_fusion_result` is duck-typed. Field names/units (`*_ms`, `*_m`, `*_px`, `session_id`, `sensor_id`, `sequence_number`) follow Phase 1/2. |
| Phase 2 gives no pre-triangulation geometry check. | Offers carry a **viewing ray** (origin, unit direction) from `geometry.pixel_ray_world`. A ray is not a 3D point; the set selector uses ray angle/closest approach to judge geometry. |

## 2. Protocol stages

| # | Stage | Where | Message / log |
|---|---|---|---|
| 1 | Task announcement | `Coordinator.request_observation` | `TASK_REQUEST` (task_id, session_id, target, window, `collect_until_ms`, `deadline_ms`, `max_lease_ms`, attempt, reason) |
| 2 | Sensor self-assessment | `SensorAgent.on_task_request` | - |
| 3 | Offer submission | `SensorAgent` → `Coordinator.on_offer` | `OBSERVATION_OFFER` |
| 4 | Eligibility filtering | `Coordinator._eligibility` | `OFFER_REJECTED` + `TASK_REJECT(sensor_id=…)` with a code |
| 5 | Set selection | `SetSelector.select` | `SET_SELECTION` (all candidates, scores, rejection codes) |
| 6 | Award or rejection | `Coordinator._award` / `_fail_task` | `TASK_AWARD` (lease) or `TASK_REJECT(sensor_id=None)`; `DEGRADED_STATUS` |
| 7 | Observation collection | fusion gate (`set_sensor_filter`) | `LEASE_STARTED`; only leased sensors may feed the target |
| 8 | Fusion evaluation | `Coordinator.on_fusion_outcome` | `FUSION_RESULT` (on verdict change), counters per verdict |
| 9 | Adaptation | `tick`, `_fusion_failure`, supervision | `ADAPTATION`, `LEASE_EXPIRED`, new `TASK_REQUEST` |
| 10 | Audit | `AuditLog` | JSONL, counters, per-stage timing |

Messages (all `protocol_version:"1.0"`, `channel:"coordination"`, strictly increasing `sequence_number`): `TASK_REQUEST`, `OBSERVATION_OFFER`,
`TASK_AWARD`, `TASK_REJECT`, `LEASE_EXPIRED`, `FUSION_RESULT`, `DEGRADED_STATUS`, `SENSOR_STATUS`.

### Coordinator state model

```
IDLE ──request──▶ ANNOUNCING ──announced──▶ COLLECTING_OFFERS ──all answered / collect window──▶ SELECTING_SET
                                                                                                   │
                         ┌─────────── award ──────────────────────────────────────────────────────┤
                         ▼                                              no feasible set, no valid lease
                   ACTIVE_LEASE ◀── renewal declined, old lease still valid ───────┐                ▼
                     │  ▲                                                          └──────────── DEGRADED
                     │  └── award ◀── (successor negotiation) ◀── ANNOUNCING ◀── retry (backoff) ────┘
                     └─ renewal / failure ──▶ ANNOUNCING                      any ──stop()──▶ STOPPED (terminal)
```
Allowed transitions are the `ALLOWED` table in `coordinator.py`; anything else raises. While a *renewal* negotiation runs, the old lease stays
valid and is atomically superseded by the award (`LEASE_EXPIRED reason=SUPERSEDED`). An ended lease immediately empties the fusion gate.

## 3. Offer score (baseline, configurable)

```
quality = 0.30*detector_confidence + 0.20*visibility + 0.20*freshness + 0.15*sync + 0.15*health        (weights: config/coordination.yaml)
```
| Component | Calculation (all clamped to 0..1) | Missing / unknown |
|---|---|---|
| detector_confidence | landmark confidence | imputed 0.5 **and flagged** (`DETECTOR_CONFIDENCE_IMPUTED`) |
| visibility | 0 if no tracked hand / landmark absent / outside frame; else `0.5+0.5*clamp(edge_margin/0.05)`; `out_of_frame` caps at 0.3 | 0 |
| freshness | `1 - age_ms/250` | 0 |
| sync | `1 - delay_jitter_ms/50`; needs a clock-offset estimate | 0 |
| health | state factor (online 1, degraded 0.4, else 0) × `(0.5+0.5*min(1, rate_hz/10))` | rate unknown = half credit |

Offers with a missing/NaN/out-of-range field are **rejected** by `ObservationOffer.from_dict` (`MISSING_FIELD`, `OUT_OF_RANGE`, `WRONG_TYPE`), never defaulted.
The weights are a starting point and have **not** been validated against real data.

### Eligibility rejection codes
`INVALID_OFFER:*`, `SESSION_MISMATCH`, `UNKNOWN_TASK`, `OFFER_LATE` (after `deadline_ms`), `TASK_CLOSED`, `DUPLICATE_OFFER`, `STALE_OFFER_SEQUENCE`,
`UNKNOWN_SENSOR`, `SENSOR_UNAVAILABLE`, `WRONG_TARGET`, `CAPABILITY_MISSING`, `NOT_CALIBRATED`, `SENSOR_UNHEALTHY`, `NO_OBSERVATION`, `STALE_OBSERVATION`,
`OBSERVATION_OUTSIDE_WINDOW`, `TARGET_NOT_VISIBLE`, `LOW_DETECTOR_CONFIDENCE`, `DEADLINE_UNREACHABLE`, `INCOMPLETE_OFFER`, `NO_OFFER`.
Duplicates/late/closed offers never overwrite a decision: the first processed offer per (task, sensor) wins and a decided task accepts nothing.

## 4. Set selection policy (`set_selection.py`, separate from scoring)

Every subset of 2..`max_set_size` (3) eligible sensors is evaluated. Hard rules, each with a code: `INCOMPATIBLE_RIG` (different rig/world frame),
`INCOMPLETE_GEOMETRY`, `INSUFFICIENT_VIEW_ANGLE` (largest pairwise ray angle < 10°), `RAYS_INCONSISTENT` (ray closest approach > 0.10 m or rays diverge behind a camera),
`SKEW_EXCEEDS_WINDOW` (>50 ms), `HAND_LABEL_MISMATCH`, `LATENCY_BUDGET`, `DEADLINE_VIOLATION`, `RECENTLY_FAILED_SET`.
Feasible score: `0.5*mean(quality) + 0.35*min(1, angle/30°) + 0.15*(1 - skew/50ms) - 0.02*(n-2)`; ties → fewer sensors → sensor ids.
Fewer than two eligible views never yields a trusted set (`NO_SET_INSUFFICIENT_SENSORS`); all-geometry failures yield `NO_SET_INSUFFICIENT_GEOMETRY`.
`allow_single_view_approximate` (default **false**) permits a labelled `SINGLE_VIEW_APPROXIMATE` award (`trusted_3d:false`, accompanying `DEGRADED_STATUS`); it is never reported as triangulation.

## 5. Adaptation rules

| Trigger | Reaction |
|---|---|
| `poor_results_before_reselect` (3) consecutive poor/unavailable fusion results (RMS > 4 px, `FUSION_OUTLIER`, `SYNC_DEGRADED`, `GEOMETRY_DEGENERATE`, `TARGET_LOST`, `INSUFFICIENT_VIEWS`, …) | ban that exact set for 3 s, end lease (`FUSION_FAILURE`), new negotiation. More than `max_reselect_attempts` (3) failures → `DEGRADED(FUSION_UNRELIABLE)` |
| no fusion result for 600 ms | counted as a poor result |
| lease member `disconnected/offline` for ≥ `disconnect_grace_ms` (500) or silent ≥ `sensor_silence_timeout_ms` (1500) | end lease (`SENSOR_UNAVAILABLE`), re-select without it |
| `calibration_ok=false` for a lease member / `on_calibration_invalid` | end lease immediately (`CALIBRATION_INVALID`), re-select |
| lease nearing expiry (`renew_lead_ms` 300) and healthy | successor negotiation; old lease superseded on award |
| renewal finds no set | old lease kept until it expires, then `DEGRADED` |
| DEGRADED | retry after 0.5 s, doubling to 4 s; backoff resets only after an **accepted** fusion result |
| `stop()` | lease ended (`SESSION_STOPPED`), gate emptied, state `STOPPED`, all input ignored |

## 6. Observability
`AuditLog` (JSONL; signature-compatible with `EventLogger.log`, which can be passed as `sink`): `TASK_REQUEST`, `OFFER_ACCEPTED`, `OFFER_REJECTED`, `SET_SELECTION`,
`TASK_AWARD`, `LEASE_STARTED`, `LEASE_EXPIRED`, `FUSION_EVALUATED`, `ADAPTATION`, `DEGRADED`, `TASK_REJECT`, `COORDINATOR_STATE`, `SENSOR_STATUS`, `CALIBRATION_INVALID`.
Counters (`coordinator.counters`): `awards`, `offers_accepted`, `offers_rejected`, `offer_rejected:<CODE>`, `fusion_verdict:*`, `lease_ended:<reason>`, `degraded_entered`, `msg:<TYPE>`, …
`StageTimer` records time in `select_set`. Only numbers, ids and reason codes are logged; keys like `frame`/`image`/`pixels` raise.

## 7. Known limitations
* Reprojection error cannot detect a landmark error **along the epipolar direction** with two views (it yields a wrong position with ~0 residual). The coordinator's "poor fusion" test is therefore a necessary, not sufficient, check.
* A failed set is banned as a whole; with 2-view sets the culprit camera cannot be identified, so a reselection may reuse it.
* Weights/thresholds are untuned defaults.
* Offers are built from the newest buffered observation, not a fresh capture on demand.
