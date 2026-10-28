"""Gesture Agent: deterministic state machine IDLE -> HOVER -> GRABBED -> RELEASED -> IDLE. No frame is classified independently.

Rules that hold for every ``update``/``tick`` call:
* AT MOST ONE state transition per call (except the safe-stop paths, which run to IDLE). Time comes only from ``now_ms`` / sample
  timestamps, never from the wall clock, so behaviour is reproducible.
* A pose is used only if the fusion result is VALID, fresh (<= max_pose_age_ms), finite and its confidence is known and >= exit_confidence.
* Loss handling is bounded: HOVER exits after hover_loss_timeout_ms; GRABBED freezes (no MOVE_TARGET, TRACKING_LOST announced) for at
  most grab_loss_timeout_ms and then cancels SAFELY (GRAB_CANCELLED). Stale coordinates are never replayed.
* A grab needs an ARMED hover (an open hand was seen while hovering) so entering reach with a closed fist never grabs.

Transition table (also docs/gesture.md):
  IDLE     -> HOVER     usable pose, nearest grab-reach object within reach_enter_m, confidence >= min_confidence over the history
  HOVER    -> IDLE      out of reach (> reach_exit_m) | confidence < exit_confidence | tracking loss > hover_loss_timeout_ms | track change |
                        session stop | calibration invalid | object gone
  HOVER    -> GRABBED   armed + pinch confirmed (confirm_ms, >= min samples, hysteresis) + object grabbable
  GRABBED  -> RELEASED  pinch opened (RELEASE) | GRAB_CANCELLED: tracking loss > grab_loss_timeout_ms, pinch evidence lost, track discontinuity,
                        track change, calibration invalid, session stop, adapter reject/ack timeout, reset
  RELEASED -> IDLE      after release_settle_ms, or after acknowledge(RELEASE) / release_ack_timeout_ms when release_ack_required
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Sequence

from ..coordination.audit import AuditLog
from .config import GestureCfg
from .events import CancelReason, EventType, GestureState, InteractionEvent
from .pinch import PinchDetector

S, E = GestureState, EventType


@dataclass(frozen=True)
class InteractionObject:
    object_id: str
    position_xyz_m: tuple[float, float, float]
    grabbable: bool = True


@dataclass(frozen=True)
class HandSample:
    """One validated fusion output (+ optional pinch evidence) for the tracked hand target.

    ``hand_xyz_m`` must be ``None`` unless ``fusion_valid`` (Phase 2: ``world_xyz_m`` is set ONLY for TRACKING_MULTI_VIEW).
    ``pinch_ratio`` comes from ``pinch.pinch_ratio`` on the contributing view(s); ``None`` = no valid evidence (never 'open', never 'closed').
    ``track_epoch`` must change whenever the hand identity/track changes (e.g. the fusion smoother reset for hand switch/discontinuity).
    """
    session_id: str
    timestamp_ms: float
    fusion_valid: bool
    hand_xyz_m: tuple[float, float, float] | None
    confidence: float | None
    pinch_ratio: float | None = None
    fusion_sequence: int | None = None
    world_frame_id: str | None = None
    track_epoch: int = 0
    calibration_valid: bool = True

    @classmethod
    def from_fusion_result(cls, r, pinch_ratio: float | None = None, track_epoch: int = 0, calibration_valid: bool = True) -> "HandSample":
        valid = bool(r.valid) and r.world_xyz_m is not None
        q = getattr(r, "quality", None)
        return cls(session_id=r.session_id, timestamp_ms=float(r.timestamp_ms) if r.timestamp_ms is not None else float("nan"),
                   fusion_valid=valid, hand_xyz_m=tuple(r.world_xyz_m) if valid else None,
                   confidence=None if q is None else q.score, pinch_ratio=pinch_ratio, fusion_sequence=int(r.fusion_sequence),
                   world_frame_id=r.world_frame_id, track_epoch=track_epoch, calibration_valid=calibration_valid)


@dataclass
class GestureOutput:
    events: list[InteractionEvent] = field(default_factory=list)     # discrete, state-transition events
    moves: list[InteractionEvent] = field(default_factory=list)      # continuous MOVE_TARGET stream
    state: GestureState = S.IDLE


class GestureAgent:
    def __init__(self, session_id: str, cfg: GestureCfg, audit: AuditLog | None = None, clock=None):
        self.cfg = cfg.validate()
        self.session_id = session_id
        self.audit = audit if audit is not None else (AuditLog(clock, component="gesture", session_id=session_id) if clock else None)
        self.counters = self.audit.counters if self.audit else collections.Counter()
        self.pinch = PinchDetector(cfg)
        self.state = S.IDLE
        self._epoch = 0
        self._n = 0
        self._full_reset()
        self.stopped = False
        self.inhibit: str | None = None

    # ------------------------------------------------------------------ internals
    def _full_reset(self) -> None:
        self.pinch.reset()
        self.state = S.IDLE
        self._hover_obj: InteractionObject | None = None
        self._interaction_id: str | None = None
        self._episode = 0
        self._loss_n = 0
        self._emitted: set[tuple[str, str]] = set()
        self._conf: collections.deque[float] = collections.deque(maxlen=self.cfg.confidence_history)
        self._last_ts: float | None = None
        self._last_seq: int | None = None
        self._last_valid_ms: int | None = None
        self._last_pose: tuple | None = None
        self._last_pose_ts: float | None = None
        self._frame: str | None = None
        self._last_conf: float | None = None
        self._track_epoch: int | None = None
        self._armed = False
        self._consumed = False
        self._loss_reported = False
        self._last_move_ms: int | None = None
        self._last_pinch_ms: int | None = None
        self._grab_req_id: str | None = None
        self._awaiting_ack_since: int | None = None
        self._confirmed = False
        self._released_at: int | None = None
        self._release_event_id: str | None = None
        self._release_acked = False

    def _count(self, name: str) -> None:
        self.counters[name] += 1

    def _ev(self, et: E, now: int, *, xyz=None, reason=None, cancel: CancelReason | None = None, continuous=False, key: str | None = None,
            state: GestureState | None = None, sample_ts=None) -> InteractionEvent | None:
        episode = key or self._interaction_id or f"hover_{self._episode}"
        if not continuous:
            token = (episode, et.value)
            if token in self._emitted:                           # idempotent: a discrete event is emitted at most once per episode
                self._count("event_suppressed_duplicate")
                return None
            self._emitted.add(token)
        self._n += 1
        ev = InteractionEvent(
            session_id=self.session_id, event_id=f"{self.session_id}:g{self._epoch}:{self._n:06d}", sequence_number=self._n, event_type=et,
            timestamp_ms=now, gesture_state=state or self.state, continuous=continuous, interaction_id=self._interaction_id,
            target_object_id=None if self._hover_obj is None else self._hover_obj.object_id, hand_target_xyz_m=None if xyz is None else tuple(float(v) for v in xyz),
            world_frame_id=self._frame if xyz is not None else None, confidence=self._last_conf if xyz is not None else None,
            cancel_reason=cancel, reason=reason, sample_timestamp_ms=sample_ts)
        self._count(f"event:{et.value}")
        if self.audit and not continuous:
            self.audit.log("INTERACTION_EVENT", reason=reason or (cancel.value if cancel else None), interaction_event=et.value, event_id=ev.event_id,
                           gesture_state=ev.gesture_state.value, interaction_id=ev.interaction_id, target=ev.target_object_id)
        return ev

    def _go(self, new: GestureState, reason: str, now: int, out: GestureOutput) -> None:
        old, self.state = self.state, new
        out.state = new
        self._count(f"transition:{old.value}->{new.value}")
        if self.audit:
            self.audit.log("GESTURE_TRANSITION", reason=reason, from_state=old.value, to_state=new.value, interaction_id=self._interaction_id,
                           target=None if self._hover_obj is None else self._hover_obj.object_id)

    def _push(self, out: GestureOutput, ev: InteractionEvent | None) -> None:
        if ev is not None:
            (out.moves if ev.continuous else out.events).append(ev)

    # ------------------------------------------------------------------ public API
    def update(self, sample: HandSample, objects: Sequence[InteractionObject], now_ms: int) -> GestureOutput:
        c = self.cfg
        out = GestureOutput(state=self.state)
        if sample.session_id != self.session_id:
            self._count("sample_ignored:SESSION_MISMATCH")
            return out
        if self.stopped:
            self._count("sample_ignored:SESSION_STOPPED")
            return out
        if not sample.calibration_valid:
            self._safe_stop(CancelReason.CALIBRATION_INVALID, "calibration_invalid", now_ms, out, inhibit="CALIBRATION_FLAG")
            return out
        if self.inhibit == "CALIBRATION_FLAG":                   # only an inhibit that came from a sample flag is lifted by a sample;
            self.inhibit = None                                  # an explicit on_calibration_invalid() persists until on_calibration_restored()
        ts = sample.timestamp_ms
        if (self._last_ts is not None and isinstance(ts, (int, float)) and math.isfinite(ts) and ts <= self._last_ts) or \
                (sample.fusion_sequence is not None and self._last_seq is not None and sample.fusion_sequence <= self._last_seq):
            self._count("sample_ignored:OUT_OF_ORDER_OR_DUPLICATE")
            return out
        if sample.track_epoch != self._track_epoch:
            first = self._track_epoch is None
            self._track_epoch = sample.track_epoch
            self.pinch.reset()
            self._conf.clear()
            if not first and self.state in (S.HOVER, S.GRABBED):
                self._cancel(CancelReason.TRACK_CHANGED, "track_changed", now_ms, out)
                self._last_ts, self._last_seq = (ts if math.isfinite(ts) else self._last_ts), sample.fusion_sequence
                return out
        if isinstance(ts, (int, float)) and math.isfinite(ts):
            self._last_ts = ts
        if sample.fusion_sequence is not None:
            self._last_seq = sample.fusion_sequence
        ok, why = self._usable(sample, now_ms)
        if sample.pinch_ratio is not None and ok:
            self._last_pinch_ms = now_ms
        if not ok:
            self._count(f"sample_unusable:{why}")
            self._timers(now_ms, out, why)
            return out
        # ---- usable pose from here on
        xyz = tuple(sample.hand_xyz_m)
        resumed_gap = self._last_valid_ms is not None and now_ms - self._last_valid_ms
        if self.state == S.GRABBED and self._last_pose is not None and resumed_gap and resumed_gap > c.tracking_lost_report_ms:
            if math.dist(xyz, self._last_pose) > c.max_resume_jump_m:
                self._last_valid_ms = now_ms
                self._cancel(CancelReason.TRACK_DISCONTINUITY, f"resumed {math.dist(xyz, self._last_pose):.2f} m from the held pose", now_ms, out)
                return out
        self._last_valid_ms, self._loss_reported = now_ms, False
        self._last_pose, self._last_pose_ts, self._last_conf, self._frame = xyz, ts, float(sample.confidence), sample.world_frame_id
        self._conf.append(float(sample.confidence))
        ps = self.pinch.update(sample.pinch_ratio, ts)
        if self.state == S.IDLE:
            self._idle(xyz, objects, now_ms, out)
        elif self.state == S.HOVER:
            self._hover(xyz, objects, ps, now_ms, out)
        elif self.state == S.GRABBED:
            self._grabbed(xyz, ps, ts, now_ms, out)
        self._timers(now_ms, out, None)
        return out

    def tick(self, now_ms: int) -> GestureOutput:
        """Advance time without a sample (tracking-loss timeouts, ack timeouts, RELEASED settling)."""
        out = GestureOutput(state=self.state)
        if not self.stopped:
            self._timers(now_ms, out, "NO_SAMPLE")
        return out

    def acknowledge(self, event_id: str, now_ms: int, accepted: bool = True) -> GestureOutput:
        """World-adapter acknowledgement of GRAB_REQUEST (accepted or not) or RELEASE. Unknown / repeated ids are ignored."""
        out = GestureOutput(state=self.state)
        if event_id == self._grab_req_id and self.state == S.GRABBED and not self._confirmed:
            self._grab_req_id, self._awaiting_ack_since = None, None
            if accepted:
                self._confirm_grab(now_ms, out)
            else:
                self._cancel(CancelReason.ADAPTER_REJECTED, "adapter rejected the grab", now_ms, out)
        elif event_id == self._release_event_id and self.state == S.RELEASED:
            self._release_acked = True
            self._timers(now_ms, out, None)
        else:
            self._count("ack_ignored")
        return out

    def on_session_stop(self, now_ms: int) -> GestureOutput:
        out = GestureOutput(state=self.state)
        self._safe_stop(CancelReason.SESSION_STOPPED, "session_stopped", now_ms, out, inhibit="SESSION_STOPPED")
        self.stopped = True
        return out

    def on_calibration_invalid(self, now_ms: int) -> GestureOutput:
        out = GestureOutput(state=self.state)
        self._safe_stop(CancelReason.CALIBRATION_INVALID, "calibration_invalid", now_ms, out, inhibit="CALIBRATION_INVALID")
        return out

    def on_calibration_restored(self) -> None:
        if self.inhibit in ("CALIBRATION_INVALID", "CALIBRATION_FLAG"):
            self.inhibit = None

    def reset(self, now_ms: int, reason: str = "reset") -> GestureOutput:
        """Session reset: cancel any grab safely, then start a new event epoch with clean history."""
        out = GestureOutput(state=self.state)
        self._safe_stop(CancelReason.RESET, reason, now_ms, out, inhibit=None)
        self._epoch += 1
        self._n = 0
        self._full_reset()
        self.stopped, self.inhibit = False, None
        return out

    # ------------------------------------------------------------------ validity
    def _usable(self, s: HandSample, now: int) -> tuple[bool, str | None]:
        c = self.cfg
        if self.inhibit:
            return False, "INHIBITED"
        if not s.fusion_valid or s.hand_xyz_m is None:
            return False, "FUSION_INVALID"
        if len(s.hand_xyz_m) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in s.hand_xyz_m):
            return False, "NON_FINITE_POSE"
        if not isinstance(s.timestamp_ms, (int, float)) or not math.isfinite(s.timestamp_ms):
            return False, "NO_TIMESTAMP"
        if now - s.timestamp_ms > c.max_pose_age_ms:
            return False, "STALE_POSE"
        if s.confidence is None or not math.isfinite(s.confidence):
            return False, "CONFIDENCE_MISSING"
        if s.confidence < c.exit_confidence:
            return False, "LOW_CONFIDENCE"
        return True, None

    # ------------------------------------------------------------------ states
    def _nearest(self, xyz, objects, radius) -> InteractionObject | None:
        best = None
        for o in sorted(objects, key=lambda o: o.object_id):
            d = math.dist(xyz, o.position_xyz_m)
            if d <= radius and (best is None or d < best[0]):
                best = (d, o)
        return None if best is None else best[1]

    def _idle(self, xyz, objects, now, out) -> None:
        c = self.cfg
        if self.inhibit:
            return
        if len(self._conf) < min(2, c.confidence_history) or min(self._conf) < c.min_confidence:
            return
        obj = self._nearest(xyz, objects, c.reach_enter_m)
        if obj is None:
            return
        self._episode += 1
        self._hover_obj, self._armed, self._consumed = obj, False, False
        self._go(S.HOVER, "in_reach", now, out)
        self._push(out, self._ev(E.HOVER_ENTER, now, xyz=xyz, reason="in_reach", sample_ts=self._last_pose_ts))

    def _hover(self, xyz, objects, ps, now, out) -> None:
        c = self.cfg
        cur = self._hover_obj
        live = next((o for o in objects if o.object_id == cur.object_id), None)
        if live is None:
            return self._hover_exit("OBJECT_GONE", now, out)
        self._hover_obj = live
        if math.dist(xyz, live.position_xyz_m) > c.reach_exit_m:
            return self._hover_exit("OUT_OF_REACH", now, out)
        if len(self._conf) and sum(self._conf) / len(self._conf) < c.exit_confidence:
            return self._hover_exit("LOW_CONFIDENCE", now, out)
        if ps.state == "OPEN":
            self._armed, self._consumed = True, False
        if ps.just_confirmed:
            if not self._armed:
                self._count("grab_rejected:NOT_ARMED")          # closed fist entered reach: never a grab
            elif not live.grabbable:
                self._count("grab_rejected:OBJECT_NOT_GRABBABLE")
            elif not self._consumed:
                return self._begin_grab(xyz, now, out)

    def _hover_exit(self, reason: str, now: int, out: GestureOutput) -> None:
        self._go(S.IDLE, reason, now, out)
        self._push(out, self._ev(E.HOVER_EXIT, now, reason=reason))
        self._hover_obj, self._armed = None, False

    def _begin_grab(self, xyz, now, out) -> None:
        self._interaction_id = f"{self.session_id}:i{self._epoch}:{self._n + 1:06d}"
        self._confirmed, self._last_move_ms, self._last_pinch_ms = False, None, now
        self._go(S.GRABBED, "pinch_confirmed", now, out)
        req = self._ev(E.GRAB_REQUEST, now, xyz=xyz, reason="pinch_confirmed", sample_ts=self._last_pose_ts)
        self._push(out, req)
        self._grab_req_id = req.event_id if req else None
        if self.cfg.require_adapter_ack:
            self._awaiting_ack_since = now
        else:
            self._confirm_grab(now, out)

    def _confirm_grab(self, now: int, out: GestureOutput) -> None:
        self._confirmed = True
        self._push(out, self._ev(E.GRAB_CONFIRMED, now, xyz=self._last_pose, reason="confirmed", sample_ts=self._last_pose_ts))

    def _grabbed(self, xyz, ps, ts, now, out) -> None:
        if ps.just_released:
            self._go(S.RELEASED, "pinch_opened", now, out)
            self._push(out, self._ev(E.RELEASE, now, xyz=xyz, reason="pinch_opened", sample_ts=ts))
            self._begin_released(now, out)
            return
        if not self._confirmed:
            return                                               # waiting for the adapter: no movement yet
        interval = 1000.0 / self.cfg.move_max_hz
        if self._last_move_ms is None or now - self._last_move_ms >= interval:
            self._last_move_ms = now
            self._push(out, self._ev(E.MOVE_TARGET, now, xyz=xyz, continuous=True, reason="pose", sample_ts=ts))

    def _begin_released(self, now: int, out: GestureOutput) -> None:
        last = [e for e in out.events if e.event_type in (E.RELEASE, E.GRAB_CANCELLED)]
        self._release_event_id = last[-1].event_id if last else None
        self._released_at, self._release_acked = now, False

    # ------------------------------------------------------------------ cancellation / safety
    def _cancel(self, reason: CancelReason, detail: str, now: int, out: GestureOutput) -> None:
        """Cancel whatever interaction is active, safely, with one transition."""
        if self.state == S.GRABBED:
            self._go(S.RELEASED, reason.value.lower(), now, out)
            self._push(out, self._ev(E.GRAB_CANCELLED, now, cancel=reason, reason=detail))
            self._begin_released(now, out)
        elif self.state == S.HOVER:
            self._hover_exit(reason.value, now, out)
        self.pinch.reset()
        self._armed = False
        self._last_move_ms = None

    def _safe_stop(self, reason: CancelReason, detail: str, now: int, out: GestureOutput, inhibit: str | None) -> None:
        self.inhibit = inhibit
        self._cancel(reason, detail, now, out)
        if self.state == S.RELEASED:                             # emitted -> processed immediately: straight to the safe no-interaction state
            self._finish_release(now, out, "safe_stop")
        self._hover_obj = None

    def _finish_release(self, now: int, out: GestureOutput, why: str) -> None:
        self._go(S.IDLE, why, now, out)
        self._interaction_id, self._hover_obj, self._grab_req_id = None, None, None
        self._released_at, self._release_event_id = None, None
        self._confirmed, self._awaiting_ack_since = False, None
        self._consumed = True                                    # the pinch that was just released must not re-grab by itself

    # ------------------------------------------------------------------ timers
    def _timers(self, now: int, out: GestureOutput, why: str | None) -> None:
        c = self.cfg
        if self.state == S.RELEASED and self._released_at is not None:
            waited = now - self._released_at
            if (c.release_ack_required and (self._release_acked or waited >= c.release_ack_timeout_ms)) or \
                    (not c.release_ack_required and waited >= c.release_settle_ms):
                if c.release_ack_required and not self._release_acked:
                    self._count("release_ack_timeout")
                self._finish_release(now, out, "release_processed")
            return
        if self.state == S.GRABBED:
            if not self._confirmed and self._awaiting_ack_since is not None and now - self._awaiting_ack_since >= c.ack_timeout_ms:
                self._cancel(CancelReason.ADAPTER_ACK_TIMEOUT, "no acknowledgement from the world adapter", now, out)
                return
            if self._last_pinch_ms is not None and now - self._last_pinch_ms > c.grab_loss_timeout_ms and self._last_valid_ms is not None \
                    and now - self._last_valid_ms <= c.grab_loss_timeout_ms:
                self._cancel(CancelReason.PINCH_EVIDENCE_LOST, "no valid pinch evidence", now, out)
                return
        if self.state in (S.HOVER, S.GRABBED) and self._last_valid_ms is not None:
            loss = now - self._last_valid_ms
            if loss > c.tracking_lost_report_ms and not self._loss_reported:
                self._loss_reported = True
                self._loss_n += 1
                self._push(out, self._ev(E.TRACKING_LOST, now, reason=why or "NO_VALID_POSE", key=f"loss_{self._loss_n}"))
                self._count("tracking_loss_incidents")
            if self.state == S.GRABBED and loss > c.grab_loss_timeout_ms:
                self._cancel(CancelReason.TRACKING_LOST_TIMEOUT, f"tracking lost for {loss} ms", now, out)
            elif self.state == S.HOVER and loss > c.hover_loss_timeout_ms:
                self._hover_exit("TRACKING_LOST", now, out)
