"""Coordination Agent: Contract Net-style task allocation over the sensor agents. Deterministic, single-threaded, clock-injected.

Stages (docs/coordination.md): TASK_REQUEST -> sensor self-assessment -> OBSERVATION_OFFER -> eligibility filtering ->
set selection -> TASK_AWARD | TASK_REJECT -> observation collection under a bounded lease -> fusion evaluation -> adaptation.

States: IDLE, ANNOUNCING, COLLECTING_OFFERS, SELECTING_SET, ACTIVE_LEASE, DEGRADED, STOPPED (transition table ``ALLOWED``).
While a successor negotiation runs (lease renewal) the previous lease stays valid; it is superseded atomically by an award, or it
simply expires. An expired/ended lease immediately removes its sensors from the fusion gate, so it can no longer control fusion.
"""
from __future__ import annotations

import collections
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import scoring
from .audit import AuditLog, StageTimer
from .config import CoordinationCfg
from .fusion_port import FusionGate, FusionOutcome
from .messages import (DegradedStatus, FusionEvaluation, LeaseExpired, MessageError, Mode, ObservationOffer, RequestReason,
                       SensorStatusMsg, TaskAward, TaskReject, TaskRequest, Verdict)
from .set_selection import SetSelection, SetSelector


class CoordState(str, enum.Enum):
    IDLE = "IDLE"
    ANNOUNCING = "ANNOUNCING"
    COLLECTING_OFFERS = "COLLECTING_OFFERS"
    SELECTING_SET = "SELECTING_SET"
    ACTIVE_LEASE = "ACTIVE_LEASE"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"


S = CoordState
ALLOWED: dict[CoordState, frozenset] = {
    S.IDLE: frozenset({S.ANNOUNCING, S.STOPPED}),
    S.ANNOUNCING: frozenset({S.COLLECTING_OFFERS, S.IDLE, S.STOPPED}),
    S.COLLECTING_OFFERS: frozenset({S.SELECTING_SET, S.IDLE, S.STOPPED}),
    S.SELECTING_SET: frozenset({S.ACTIVE_LEASE, S.DEGRADED, S.IDLE, S.STOPPED}),
    S.ACTIVE_LEASE: frozenset({S.ANNOUNCING, S.DEGRADED, S.IDLE, S.STOPPED}),
    S.DEGRADED: frozenset({S.ANNOUNCING, S.IDLE, S.STOPPED}),
    S.STOPPED: frozenset(),
}


@dataclass
class _Task:
    request: TaskRequest
    expected: tuple[str, ...]
    offers: dict[str, ObservationOffer] = field(default_factory=dict)       # eligible offers (one per sensor, first wins)
    seen: dict[str, int] = field(default_factory=dict)                      # sensor -> offer_sequence of the offer already processed
    rejected: dict[str, str] = field(default_factory=dict)                  # sensor -> rejection code
    closed: bool = False                                                    # collection closed (no more offers accepted)
    done: bool = False                                                      # decision taken
    selection: SetSelection | None = None
    outcome: str | None = None


@dataclass
class _Lease:
    award: TaskAward
    sensor_ids: frozenset
    started_ms: int
    expires_ms: int
    target: str
    mode: Mode
    last_fusion_seq: int = -1
    last_outcome_ms: int = 0
    poor: int = 0
    accepted: int = 0
    last_verdict: tuple | None = None
    renewal_declined: str | None = None


@dataclass
class _SensorRec:
    state: str
    last_seen_ms: int
    changed_ms: int
    last_seq: int = -1
    calibration_ok: bool = True


class Coordinator:
    def __init__(self, session_id: str, cfg: CoordinationCfg, clock, *, fusion_gate: FusionGate | None = None,
                 audit: AuditLog | None = None, perf: Callable[[], float] = time.perf_counter, auto_renew: bool = True):
        cfg.validate()
        self.session_id, self.cfg, self.clock = session_id, cfg, clock
        self.audit = audit or AuditLog(clock, component="coordinator", session_id=session_id)
        self.timer = StageTimer(perf)
        self.gate = fusion_gate
        self.auto_renew = auto_renew
        self.selector = SetSelector(cfg)
        self.state = CoordState.IDLE
        self.sensors: dict[str, _SensorRec] = {}
        self.tasks: "collections.OrderedDict[str, _Task]" = collections.OrderedDict()
        self.current: _Task | None = None
        self.lease: _Lease | None = None
        self.outbox: collections.deque = collections.deque(maxlen=2000)
        self.subscribers: list[Callable[[Any], None]] = []
        self.counters = self.audit.counters
        self.banned: dict[frozenset, int] = {}
        self.reselect_attempts = 0
        self.degraded: dict | None = None
        self._degraded_n = 0
        self._msg_seq = self._task_n = self._award_n = self._epoch = 0
        self._target = "wrist"
        self.last_selection: SetSelection | None = None

    # ------------------------------------------------------------------ plumbing
    def _now(self) -> int:
        return int(self.clock.now_ms())

    def _next_seq(self) -> int:
        self._msg_seq += 1
        return self._msg_seq

    def _emit(self, msg) -> None:
        self.outbox.append(msg)
        self.counters[f"msg:{msg.message_type}"] += 1
        for fn in list(self.subscribers):
            try:
                fn(msg)
            except Exception:                                     # a consumer bug must never break coordination
                self.counters["subscriber_errors"] += 1

    def subscribe(self, fn: Callable[[Any], None]) -> None:
        self.subscribers.append(fn)

    def _transition(self, new: CoordState, reason: str) -> None:
        if new == self.state:
            return
        if new not in ALLOWED[self.state]:
            raise RuntimeError(f"illegal coordinator transition {self.state.value} -> {new.value} ({reason})")
        old, self.state = self.state, new
        self.audit.log("COORDINATOR_STATE", reason=reason, from_state=old.value, to_state=new.value)

    def _set_gate(self, target: str, ids: frozenset | None) -> None:
        if self.gate is not None:
            self.gate(target, ids)

    def _negotiating(self) -> bool:
        return self.current is not None and not self.current.done

    def _available(self, sid: str, now: int) -> bool:
        r = self.sensors.get(sid)
        return r is not None and r.state in ("online", "degraded") and now - r.last_seen_ms <= self.cfg.adaptation.sensor_silence_timeout_ms

    def _unavailable_cause(self, sid: str, now: int) -> str:
        r = self.sensors.get(sid)
        if r is None:
            return "UNKNOWN_SENSOR"
        if r.state in ("disconnected", "offline"):
            return f"SENSOR_{r.state.upper()}"
        return "SENSOR_SILENT"

    # ------------------------------------------------------------------ sensors
    def on_sensor_status(self, msg: SensorStatusMsg) -> str:
        now = self._now()
        if self.state == CoordState.STOPPED:
            return "IGNORED_STOPPED"
        if msg.session_id != self.session_id:
            self.counters["status_session_mismatch"] += 1
            return "SESSION_MISMATCH"
        rec = self.sensors.get(msg.sensor_id)
        if rec is not None and msg.sequence_number <= rec.last_seq:
            self.counters["status_stale"] += 1
            self.audit.log("SENSOR_STATUS_IGNORED", sensor_id=msg.sensor_id, reason="STALE_SEQUENCE", severity="warning")
            return "STALE_SEQUENCE"
        if rec is None:
            rec = self.sensors[msg.sensor_id] = _SensorRec(msg.state, now, now)
        elif rec.state != msg.state:
            rec.state, rec.changed_ms = msg.state, now
        rec.last_seen_ms, rec.last_seq = now, msg.sequence_number
        was_ok, rec.calibration_ok = rec.calibration_ok, msg.calibration_ok
        self.audit.log("SENSOR_STATUS", sensor_id=msg.sensor_id, reason=msg.reason, state=msg.state, calibration_ok=msg.calibration_ok)
        if was_ok and not msg.calibration_ok:
            self.on_calibration_invalid([msg.sensor_id])
        return "OK"

    def on_calibration_invalid(self, sensor_ids) -> None:
        now = self._now()
        for sid in sensor_ids:
            if sid in self.sensors:
                self.sensors[sid].calibration_ok = False
        self.audit.log("CALIBRATION_INVALID", reason="calibration invalidated", severity="warning", sensors=sorted(sensor_ids))
        if self.lease is not None and self.lease.sensor_ids & set(sensor_ids):
            self._end_lease("CALIBRATION_INVALID")
            self._after_lease_loss(RequestReason.RESELECT_CALIBRATION, "CALIBRATION_INVALID")

    # ------------------------------------------------------------------ Stage 1: announcement
    def request_observation(self, target: str = "wrist", reason: RequestReason = RequestReason.INITIAL, parent: str | None = None,
                            trusted_3d: bool = True, attempt: int = 1) -> str | None:
        if self.state == CoordState.STOPPED:
            self.counters["request_ignored_stopped"] += 1
            return None
        if self._negotiating():
            return self.current.request.task_id            # idempotent: one negotiation at a time
        now, tc = self._now(), self.cfg.task
        self._target = target
        self._task_n += 1
        req = TaskRequest(session_id=self.session_id, task_id=f"task_{self._task_n:05d}", parent_task_id=parent,
                          sequence_number=self._next_seq(), target_landmark=target, issued_at_ms=now,
                          collect_until_ms=int(now + tc.offer_collect_ms), deadline_ms=int(now + tc.deadline_ms),
                          window_start_ms=int(now - tc.window_ms), window_end_ms=now, max_lease_ms=int(tc.max_lease_ms),
                          min_views=self.cfg.set_selection.min_views if trusted_3d else 1, trusted_3d_required=trusted_3d,
                          attempt=attempt, reason=reason)
        expected = tuple(sorted(s for s in self.sensors if self._available(s, now)))
        task = _Task(request=req, expected=expected)
        self.tasks[req.task_id] = task
        while len(self.tasks) > tc.recent_tasks:
            self.tasks.popitem(last=False)
        self.current = task
        self._transition(CoordState.ANNOUNCING, reason.value)
        self.audit.log("TASK_REQUEST", reason=reason.value, task_id=req.task_id, target=target, deadline_ms=req.deadline_ms,
                       collect_until_ms=req.collect_until_ms, attempt=attempt, expected_sensors=list(expected),
                       parent_task_id=parent, unavailable=sorted(set(self.sensors) - set(expected)))
        self._transition(CoordState.COLLECTING_OFFERS, "announced")
        self._emit(req)                                       # sensor agents may answer synchronously from here
        return req.task_id

    # ------------------------------------------------------------------ Stage 3/4: offers + eligibility
    def on_offer(self, offer: "ObservationOffer | dict") -> str:
        """Process one offer. Returns ``ACCEPTED`` or the rejection code. Never raises on bad input; never overwrites a decision."""
        now = self._now()
        if self.state == CoordState.STOPPED:
            self.counters["offer_ignored_stopped"] += 1
            return "COORDINATOR_STOPPED"
        if not isinstance(offer, ObservationOffer):
            try:
                offer = ObservationOffer.from_dict(offer)
            except MessageError as exc:
                return self._reject_offer(None, None, f"INVALID_OFFER:{exc.code}", exc.detail, now, {})
        scores = self._score_fields(offer)
        if offer.session_id != self.session_id:
            return self._reject_offer(offer, None, "SESSION_MISMATCH", "offer belongs to another session", now, scores)
        task = self.tasks.get(offer.task_id)
        if task is None:
            return self._reject_offer(offer, None, "UNKNOWN_TASK", "no such task", now, scores)
        req = task.request
        if now > req.deadline_ms:
            return self._reject_offer(offer, task, "OFFER_LATE", f"arrived {now - req.deadline_ms} ms after the deadline", now, scores)
        if task.closed or task.done or task is not self.current:
            return self._reject_offer(offer, task, "TASK_CLOSED", "task no longer collecting offers (decision taken or superseded)", now, scores)
        if offer.sensor_id in task.seen:
            code = "STALE_OFFER_SEQUENCE" if offer.offer_sequence < task.seen[offer.sensor_id] else "DUPLICATE_OFFER"
            return self._reject_offer(offer, task, code, "an offer from this sensor was already processed for this task", now, scores)
        task.seen[offer.sensor_id] = offer.offer_sequence
        if offer.sensor_id not in self.sensors:
            return self._reject_offer(offer, task, "UNKNOWN_SENSOR", "sensor never reported status", now, scores)
        code, detail = self._eligibility(offer, req, now)
        if code:
            return self._reject_offer(offer, task, code, detail, now, scores)
        task.offers[offer.sensor_id] = offer
        self.counters["offers_accepted"] += 1
        self.audit.log("OFFER_ACCEPTED", sensor_id=offer.sensor_id, reason="ELIGIBLE", task_id=offer.task_id, **scores)
        self._maybe_close_collection(task)
        return "ACCEPTED"

    def _score_fields(self, o: ObservationOffer) -> dict:
        try:
            q = scoring.quality_of(self.cfg.weights, o)
        except ValueError:
            q = None
        return {"quality": q, "detector_confidence": o.detector_confidence, "detector_confidence_source": o.detector_confidence_source,
                "visibility": o.visibility_score, "freshness": o.freshness_score, "sync": o.sync_score, "health": o.health_score,
                "latency_ms": o.estimated_latency_ms, "observation_age_ms": o.observation_age_ms}

    def _eligibility(self, o: ObservationOffer, req: TaskRequest, now: int) -> tuple[str | None, str]:
        el = self.cfg.eligibility
        if not self._available(o.sensor_id, now):
            return "SENSOR_UNAVAILABLE", self._unavailable_cause(o.sensor_id, now)
        if o.target_landmark != req.target_landmark:
            return "WRONG_TARGET", f"offer is for {o.target_landmark}"
        if req.target_landmark not in o.capabilities:
            return "CAPABILITY_MISSING", "sensor does not report the target landmark"
        if el.require_calibration and (not o.calibration_eligible or not self.sensors[o.sensor_id].calibration_ok):
            return "NOT_CALIBRATED", "; ".join(n for n in o.notes if n.startswith("CALIBRATION")) or "calibration not eligible"
        if o.health_score < el.min_health:
            return "SENSOR_UNHEALTHY", f"health {o.health_score:.2f} < {el.min_health}"
        if o.observation_effective_ms is None:
            return "NO_OBSERVATION", "sensor has no observation"
        age = now - o.observation_effective_ms
        if age > el.max_observation_age_ms:
            return "STALE_OBSERVATION", f"observation is {age:.0f} ms old (limit {el.max_observation_age_ms:.0f} ms)"
        if o.observation_effective_ms < req.window_start_ms:
            return "OBSERVATION_OUTSIDE_WINDOW", "captured before the task window"
        if o.visibility_score < el.min_visibility:
            return "TARGET_NOT_VISIBLE", f"visibility {o.visibility_score:.2f} < {el.min_visibility}"
        if o.detector_confidence < el.min_detector_confidence:
            return "LOW_DETECTOR_CONFIDENCE", f"confidence {o.detector_confidence:.2f} < {el.min_detector_confidence}"
        if o.submitted_at_ms + o.estimated_latency_ms > req.deadline_ms:
            return "DEADLINE_UNREACHABLE", "cannot deliver before the deadline"
        if req.trusted_3d_required and (o.view_ray is None or o.rig_key is None):
            return "INCOMPLETE_OFFER", "no calibrated viewing geometry"
        return None, ""

    def _reject_offer(self, offer, task: _Task | None, code: str, detail: str, now: int, scores: dict) -> str:
        sid = None if offer is None else offer.sensor_id
        tid = (task.request.task_id if task else (None if offer is None else offer.task_id)) or "unknown"
        if task is not None and sid is not None and code not in ("DUPLICATE_OFFER", "STALE_OFFER_SEQUENCE", "TASK_CLOSED", "OFFER_LATE"):
            task.rejected[sid] = code
        elif task is not None and sid is not None and sid not in task.rejected and sid not in task.offers:
            task.rejected[sid] = code
        self.counters["offers_rejected"] += 1
        self.counters[f"offer_rejected:{code}"] += 1
        self.audit.log("OFFER_REJECTED", sensor_id=sid, reason=code, severity="warning", task_id=tid, detail=detail, **scores)
        self._emit(TaskReject(session_id=self.session_id, task_id=tid, sequence_number=self._next_seq(), sensor_id=sid,
                              rejected_at_ms=now, reason_code=code, detail=detail[:200]))
        if task is not None and task is self.current and not task.closed:
            self._maybe_close_collection(task)
        return code

    # ------------------------------------------------------------------ Stage 5: selection + award
    def _maybe_close_collection(self, task: _Task) -> None:
        if task.closed or task is not self.current:
            return
        if all(s in task.seen for s in task.expected):
            self._close_collection(task, "ALL_EXPECTED_RESPONDED")

    def _close_collection(self, task: _Task, why: str) -> None:
        now = self._now()
        task.closed = True
        for sid in task.expected:
            if sid not in task.seen:
                task.rejected[sid] = "NO_OFFER"
                self.counters["offer_rejected:NO_OFFER"] += 1
                self.audit.log("OFFER_REJECTED", sensor_id=sid, reason="NO_OFFER", severity="warning", task_id=task.request.task_id,
                               detail="expected sensor did not answer before collection closed")
                self._emit(TaskReject(session_id=self.session_id, task_id=task.request.task_id, sequence_number=self._next_seq(),
                                      sensor_id=sid, rejected_at_ms=now, reason_code="NO_OFFER", detail="no answer in time"))
        self._transition(CoordState.SELECTING_SET, why)
        self._select(task)

    def _select(self, task: _Task) -> None:
        now, req = self._now(), task.request
        self.banned = {k: v for k, v in self.banned.items() if v > now}
        with self.timer.stage("select_set"):
            sel = self.selector.select(sorted(task.offers.values(), key=lambda o: o.sensor_id), req, set(self.banned))
        task.selection, self.last_selection = sel, sel
        task.done = True
        self.audit.log("SET_SELECTION", reason=sel.reason, task_id=req.task_id, outcome=sel.outcome,
                       chosen=None if sel.chosen is None else list(sel.chosen.sensor_ids), best_individual=sel.best_individual,
                       individual_quality=sel.individual_quality,
                       candidates=[{"set": list(c.sensor_ids), "feasible": c.feasible, "codes": list(c.rejection_codes),
                                    "score": c.score, **{k: v for k, v in c.components.items() if k in ("angle_deg", "skew_ms", "ray_gap_m", "mean_quality")}}
                                   for c in sel.evaluated],
                       stage_ms=self.timer.last_ms.get("select_set"), collect_ms=now - req.issued_at_ms)
        if now > req.deadline_ms:
            self._fail_task(task, "NO_AWARD_BY_DEADLINE", f"decision at +{now - req.issued_at_ms} ms exceeded the deadline")
            return
        if sel.chosen is not None:
            self._award(task, sel.chosen.sensor_ids, Mode.TRIANGULATION, True, sel.chosen.score, sel.reason)
            return
        approx = self.cfg.adaptation.allow_single_view_approximate and task.offers
        if approx:
            best = max(task.offers.values(), key=lambda o: (scoring.quality_of(self.cfg.weights, o), o.sensor_id))
            self._award(task, (best.sensor_id,), Mode.SINGLE_VIEW_APPROXIMATE, False, scoring.quality_of(self.cfg.weights, best),
                        f"{sel.outcome}: single-view approximate mode (NOT triangulation)")
            return
        self._fail_task(task, sel.outcome, sel.reason)

    def _award(self, task: _Task, sensor_ids: tuple[str, ...], mode: Mode, trusted: bool, score: float | None, reason: str) -> None:
        now, req = self._now(), task.request
        if self.lease is not None:
            self._end_lease("SUPERSEDED")
        self._award_n += 1
        self._epoch += 1
        lease_ms = int(min(self.cfg.task.lease_ms, req.max_lease_ms))
        award = TaskAward(session_id=self.session_id, task_id=req.task_id, award_id=f"award_{self._award_n:05d}",
                          sequence_number=self._next_seq(), decision_epoch=self._epoch, target_landmark=req.target_landmark,
                          sensor_ids=tuple(sorted(sensor_ids)), mode=mode, trusted_3d=trusted, awarded_at_ms=now,
                          lease_expires_at_ms=now + lease_ms, set_score=score, reason=reason)
        self.lease = _Lease(award=award, sensor_ids=frozenset(award.sensor_ids), started_ms=now, expires_ms=award.lease_expires_at_ms,
                            target=req.target_landmark, mode=mode, last_outcome_ms=now)
        task.outcome = "AWARD"
        self._set_gate(req.target_landmark, self.lease.sensor_ids)
        self.degraded = None                                      # backoff counter is reset only by an ACCEPTED fusion result
        self._transition(CoordState.ACTIVE_LEASE, f"award {award.award_id}")
        self.counters["awards"] += 1
        self.audit.log("TASK_AWARD", reason=reason, task_id=req.task_id, award_id=award.award_id, sensors=list(award.sensor_ids),
                       mode=mode.value, trusted_3d=trusted, set_score=score, lease_ms=lease_ms, decision_epoch=self._epoch)
        self.audit.log("LEASE_STARTED", task_id=req.task_id, award_id=award.award_id, sensors=list(award.sensor_ids),
                       expires_at_ms=award.lease_expires_at_ms)
        self._emit(award)
        if mode == Mode.SINGLE_VIEW_APPROXIMATE:
            self._emit(DegradedStatus(session_id=self.session_id, task_id=req.task_id, sequence_number=self._next_seq(), entered_at_ms=now,
                                      reason_code="SINGLE_VIEW_APPROXIMATE", detail="only one usable view: approximate, NOT calibrated triangulation",
                                      tracking_trusted=False, mode=mode, retry_at_ms=None))

    def _fail_task(self, task: _Task, code: str, detail: str) -> None:
        now = self._now()
        task.outcome = code
        self.counters["task_rejected"] += 1
        self._emit(TaskReject(session_id=self.session_id, task_id=task.request.task_id, sequence_number=self._next_seq(), sensor_id=None,
                              rejected_at_ms=now, reason_code=code, detail=detail[:200]))
        self.audit.log("TASK_REJECT", reason=code, severity="warning", task_id=task.request.task_id, detail=detail)
        if self.lease is not None and now < self.lease.expires_ms:       # renewal declined: keep the still-valid lease
            self.lease.renewal_declined = code
            self._transition(CoordState.ACTIVE_LEASE, f"renewal declined: {code}")
            return
        self._degrade(code, detail, task.request.task_id)

    # ------------------------------------------------------------------ leases
    def _end_lease(self, reason: str) -> None:
        lease = self.lease
        if lease is None:
            return
        self.lease = None
        now = self._now()
        self._set_gate(lease.target, frozenset())                 # an ended lease controls nothing
        self.counters[f"lease_ended:{reason}"] += 1
        self.audit.log("LEASE_EXPIRED", reason=reason, task_id=lease.award.task_id, award_id=lease.award.award_id,
                       sensors=sorted(lease.sensor_ids), held_ms=now - lease.started_ms)
        self._emit(LeaseExpired(session_id=self.session_id, task_id=lease.award.task_id, award_id=lease.award.award_id,
                                sequence_number=self._next_seq(), sensor_ids=lease.award.sensor_ids, expired_at_ms=now, reason=reason))

    def _degrade(self, reason: str, detail: str, task_id: str | None) -> None:
        now, ad = self._now(), self.cfg.adaptation
        self._set_gate(self._target, frozenset())
        retry = min(ad.degraded_retry_ms * (2 ** self._degraded_n), ad.degraded_retry_max_ms)
        self._degraded_n += 1
        self.degraded = {"since_ms": now, "reason": reason, "retry_at_ms": int(now + retry)}
        self._transition(CoordState.DEGRADED, reason)
        self.counters["degraded_entered"] += 1
        self.audit.log("DEGRADED", reason=reason, severity="warning", task_id=task_id, detail=detail, retry_at_ms=int(now + retry),
                       tracking_trusted=False)
        self._emit(DegradedStatus(session_id=self.session_id, task_id=task_id, sequence_number=self._next_seq(), entered_at_ms=now,
                                  reason_code=reason, detail=detail[:200], tracking_trusted=False, mode=Mode.NONE, retry_at_ms=int(now + retry)))

    def _after_lease_loss(self, why: RequestReason, code: str) -> None:
        """A lease ended for a failure reason: start a new negotiation unless one is already running."""
        if self.state == CoordState.STOPPED or self._negotiating():
            return
        self.request_observation(self._target, why, parent=None, attempt=self.reselect_attempts + 1)

    # ------------------------------------------------------------------ Stage 8/9: fusion feedback + adaptation
    def on_fusion_outcome(self, o: FusionOutcome) -> Verdict | None:
        now, ad = self._now(), self.cfg.adaptation
        if self.state == CoordState.STOPPED:
            self.counters["fusion_ignored:stopped"] += 1
            return None
        lease = self.lease
        why = None
        if lease is None or now >= lease.expires_ms:
            why = "NO_ACTIVE_LEASE"
        elif o.session_id != self.session_id or o.target_landmark != lease.target:
            why = "OTHER_TARGET_OR_SESSION"
        elif o.fusion_sequence <= lease.last_fusion_seq:
            why = "DUPLICATE_OR_STALE_SEQUENCE"
        elif o.published_at_ms < lease.started_ms:
            why = "PUBLISHED_BEFORE_LEASE"
        elif o.source_sensor_ids and not set(o.source_sensor_ids) <= lease.sensor_ids:
            why = "SOURCES_OUTSIDE_LEASE"
        if why:
            self.counters[f"fusion_ignored:{why}"] += 1
            self.audit.log("FUSION_IGNORED", reason=why, fusion_sequence=o.fusion_sequence, severity="warning" if why == "SOURCES_OUTSIDE_LEASE" else "info")
            return None
        lease.last_fusion_seq, lease.last_outcome_ms = o.fusion_sequence, now
        if lease.mode == Mode.SINGLE_VIEW_APPROXIMATE:
            verdict, vreason = Verdict.UNAVAILABLE, "APPROXIMATE_ONLY_NOT_TRIANGULATION"
        elif o.valid and o.rms_px is not None and o.rms_px <= ad.fusion_max_reprojection_px:
            verdict, vreason = Verdict.ACCEPTED, None
        elif o.valid:
            verdict, vreason = Verdict.POOR_QUALITY, "REPROJECTION_ERROR_HIGH" if o.rms_px is not None else "REPROJECTION_UNKNOWN"
        elif o.status in ("FUSION_OUTLIER", "SYNC_DEGRADED", "GEOMETRY_DEGENERATE"):
            verdict, vreason = Verdict.POOR_QUALITY, o.reason or o.status
        else:
            verdict, vreason = Verdict.UNAVAILABLE, o.reason or o.status
        self.counters[f"fusion_verdict:{verdict.value}"] += 1
        if lease.mode == Mode.TRIANGULATION:
            if verdict == Verdict.ACCEPTED:
                lease.poor, lease.accepted, self.reselect_attempts, self._degraded_n = 0, lease.accepted + 1, 0, 0
            else:
                lease.poor += 1
        key = (verdict, vreason)
        if key != lease.last_verdict:
            lease.last_verdict = key
            self.audit.log("FUSION_EVALUATED", reason=vreason, task_id=lease.award.task_id, verdict=verdict.value, fusion_status=o.status,
                           rms_px=o.rms_px, ray_angle_deg=o.ray_angle_deg, consecutive_poor=lease.poor, sensors=sorted(lease.sensor_ids),
                           severity="info" if verdict == Verdict.ACCEPTED else "warning")
            self._emit(FusionEvaluation(session_id=self.session_id, task_id=lease.award.task_id, award_id=lease.award.award_id,
                                        sequence_number=self._next_seq(), fusion_sequence=o.fusion_sequence, fusion_status=o.status,
                                        verdict=verdict, reason=vreason, aggregate_reprojection_error_px=o.rms_px,
                                        ray_angle_deg=o.ray_angle_deg, consecutive_poor=lease.poor, evaluated_at_ms=now))
        if lease.mode == Mode.TRIANGULATION and lease.poor >= ad.poor_results_before_reselect:
            self._fusion_failure(lease, o, vreason)
        return verdict

    def _fusion_failure(self, lease: _Lease, o: FusionOutcome, vreason: str | None) -> None:
        now = self._now()
        self.banned[frozenset(lease.sensor_ids)] = int(now + self.cfg.set_selection.failed_set_ban_ms)
        self.reselect_attempts += 1
        notes = {k: v for k, v in o.sensor_notes.items()}
        self.audit.log("ADAPTATION", reason="FUSION_FAILURE", severity="warning", task_id=lease.award.task_id, failed_set=sorted(lease.sensor_ids),
                       fusion_reason=vreason, sensor_notes=notes, attempt=self.reselect_attempts, ban_ms=self.cfg.set_selection.failed_set_ban_ms)
        self._end_lease("FUSION_FAILURE")
        if self.reselect_attempts > self.cfg.adaptation.max_reselect_attempts:
            if not self._negotiating():
                self._degrade("FUSION_UNRELIABLE", f"fusion failed {self.reselect_attempts} consecutive selections (last: {vreason})", None)
                self.reselect_attempts = 0
            return
        self._after_lease_loss(RequestReason.RESELECT_FUSION_FAILURE, "FUSION_FAILURE")

    # ------------------------------------------------------------------ time
    def tick(self) -> None:
        if self.state == CoordState.STOPPED:
            return
        now, ad = self._now(), self.cfg.adaptation
        task = self.current
        if task is not None and not task.done:
            if not task.closed and now >= task.request.collect_until_ms:
                self._close_collection(task, "COLLECT_WINDOW_ELAPSED")
        # lease supervision
        lease = self.lease
        if lease is not None:
            for sid in sorted(lease.sensor_ids):
                lost = self._lost(sid, now)
                if lost:
                    self.audit.log("ADAPTATION", reason="SENSOR_UNAVAILABLE", severity="warning", sensor_id=sid, detail=lost, task_id=lease.award.task_id)
                    self._end_lease("SENSOR_UNAVAILABLE")
                    self._after_lease_loss(RequestReason.RESELECT_SENSOR_LOSS, "SENSOR_UNAVAILABLE")
                    break
        lease = self.lease
        if lease is not None:
            if now >= lease.expires_ms:
                declined = lease.renewal_declined
                self._end_lease("LEASE_TIMEOUT")
                if not self._negotiating():
                    if declined:
                        self._degrade(declined, "lease expired and renewal was declined", None)
                    elif self.auto_renew:
                        self._after_lease_loss(RequestReason.RENEWAL, "LEASE_TIMEOUT")
                    else:
                        self._transition(CoordState.IDLE, "lease expired")
            elif lease.mode == Mode.TRIANGULATION and now - lease.last_outcome_ms > ad.fusion_silence_ms:
                lease.last_outcome_ms, lease.poor = now, lease.poor + 1
                self.counters["fusion_verdict:SILENCE"] += 1
                self.audit.log("FUSION_EVALUATED", reason="NO_FUSION_RESULTS", severity="warning", task_id=lease.award.task_id,
                               verdict=Verdict.UNAVAILABLE.value, consecutive_poor=lease.poor)
                if lease.poor >= ad.poor_results_before_reselect:
                    self._fusion_failure(lease, FusionOutcome(self.session_id, lease.target, lease.last_fusion_seq, "NO_RESULTS", False,
                                                              "NO_FUSION_RESULTS", None, None, None, None, (), now), "NO_FUSION_RESULTS")
            elif (self.auto_renew and not self._negotiating() and lease.poor == 0 and lease.renewal_declined is None
                  and now >= lease.expires_ms - self.cfg.task.renew_lead_ms):
                self.request_observation(lease.target, RequestReason.RENEWAL, parent=lease.award.task_id)
        if self.state == CoordState.DEGRADED and self.degraded and now >= self.degraded["retry_at_ms"] and not self._negotiating():
            self.request_observation(self._target, RequestReason.RETRY_AFTER_DEGRADED, attempt=self._degraded_n + 1)

    def _lost(self, sid: str, now: int) -> str | None:
        r, ad = self.sensors.get(sid), self.cfg.adaptation
        if r is None:
            return "unknown sensor"
        if r.state in ("disconnected", "offline") and now - r.changed_ms >= ad.disconnect_grace_ms:
            return f"{r.state} for {now - r.changed_ms} ms (grace {ad.disconnect_grace_ms:.0f} ms)"
        if now - r.last_seen_ms > ad.sensor_silence_timeout_ms:
            return f"silent for {now - r.last_seen_ms} ms (timeout {ad.sensor_silence_timeout_ms:.0f} ms)"
        return None

    # ------------------------------------------------------------------ lifecycle
    def release(self, reason: str = "RELEASED") -> None:
        if self.state == CoordState.STOPPED:
            return
        if self.current is not None:
            self.current.done = self.current.closed = True
        self._end_lease(reason)
        if self.state != CoordState.IDLE:
            self._transition(CoordState.IDLE, reason)

    def stop(self, reason: str = "SESSION_STOPPED") -> None:
        if self.state == CoordState.STOPPED:
            return
        if self.current is not None:
            self.current.done = self.current.closed = True
        self._end_lease("SESSION_STOPPED")
        self._set_gate(self._target, frozenset())
        self._transition(CoordState.STOPPED, reason)
        self.audit.log("COORDINATOR_STOPPED", reason=reason)

    # ------------------------------------------------------------------ reporting
    def status(self) -> dict:
        l = self.lease
        return {"state": self.state.value, "lease": None if l is None else {"award_id": l.award.award_id, "sensors": sorted(l.sensor_ids),
                "mode": l.mode.value, "trusted_3d": l.award.trusted_3d, "expires_at_ms": l.expires_ms, "poor": l.poor},
                "degraded": self.degraded, "counters": dict(self.counters), "stage_timing": self.timer.summary()}
