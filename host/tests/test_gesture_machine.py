"""Gesture state machine: every transition, safety rules, idempotent events, tracking loss, stop/calibration/reset."""
import json
import math

import pytest

from host.app.coordination.simulation import SimClock
from host.app.gesture.agent import GestureAgent, GestureOutput, HandSample, InteractionObject
from host.app.gesture.config import GestureCfg
from host.app.gesture.events import CancelReason, EventDeduper, EventType as E, GestureState as G, InteractionEvent

SID = "sess_gest0001"
OBJ = InteractionObject("cube_1", (0.0, 0.0, 0.0))
FAR = (1.0, 0.0, 0.0)
NEAR = (0.05, 0.0, 0.0)
STEP = 60


class Rig:
    """Feeds samples at a fixed rate; ``self.log`` keeps every output so tests can assert on exact event sequences."""

    def __init__(self, **over):
        self.clock = SimClock(10_000)
        self.cfg = GestureCfg(**over).validate()
        self.agent = GestureAgent(SID, self.cfg, clock=self.clock)
        self.seq = 0
        self.events, self.moves, self.states = [], [], []

    def sample(self, xyz=NEAR, ratio=0.8, conf=0.9, valid=True, **kw):
        self.seq += 1
        t = self.clock.now_ms()
        return HandSample(session_id=SID, timestamp_ms=t - 10, fusion_valid=valid, hand_xyz_m=xyz if valid else None,
                          confidence=conf if valid else None, pinch_ratio=ratio, fusion_sequence=self.seq, world_frame_id="board_frame", **kw)

    def step(self, objects=(OBJ,), **kw):
        self.clock.advance(STEP)
        out = self.agent.update(self.sample(**kw), objects, self.clock.now_ms())
        return self._rec(out)

    def tick(self, ms=STEP):
        self.clock.advance(ms)
        return self._rec(self.agent.tick(self.clock.now_ms()))

    def run(self, n, **kw):
        outs = [self.step(**kw) for _ in range(n)]
        return outs

    def _rec(self, out):
        self.events += out.events
        self.moves += out.moves
        self.states.append(self.agent.state)
        return out

    def types(self):
        return [e.event_type for e in self.events]

    def hover(self):
        self.run(3, ratio=0.9)
        assert self.agent.state == G.HOVER

    def grab(self):
        self.hover()
        self.run(6, ratio=0.1)
        assert self.agent.state == G.GRABBED

    def until_released(self, limit=12):
        for _ in range(limit):
            self.step(ratio=0.9)
            if self.agent.state == G.RELEASED:
                return
        raise AssertionError("never released")


def test_idle_to_hover_requires_reach_confidence_and_valid_fusion():
    r = Rig()
    r.run(4, xyz=FAR)
    assert r.agent.state == G.IDLE
    r.run(4, ratio=0.9, conf=0.4)                                                  # in reach, usable, but below min_confidence (0.5)
    assert r.agent.state == G.IDLE
    r.run(3, valid=False)
    assert r.agent.state == G.IDLE and not r.events
    r.step(ratio=0.9)
    assert r.agent.state == G.IDLE                                                 # one good sample is not enough: the low-confidence history still counts
    r.run(3, ratio=0.9)
    assert r.agent.state == G.HOVER and r.types() == [E.HOVER_ENTER]
    ev = r.events[0]
    assert ev.target_object_id == "cube_1" and ev.hand_target_xyz_m == NEAR and ev.gesture_state == G.HOVER and ev.confidence == 0.9


def test_hover_to_grabbed_after_confirmation_with_request_then_confirmed_and_no_repeats():
    r = Rig()
    r.hover()
    outs = r.run(10, ratio=0.1)
    assert r.agent.state == G.GRABBED
    assert r.types() == [E.HOVER_ENTER, E.GRAB_REQUEST, E.GRAB_CONFIRMED]          # exactly one of each despite many pinched frames
    req, conf = r.events[1], r.events[2]
    assert req.interaction_id == conf.interaction_id and req.event_id != conf.event_id and conf.gesture_state == G.GRABBED
    first = next(i for i, o in enumerate(outs) if o.events)
    assert first >= 2                                                              # not on the first pinched frame (confirmation window)
    assert [o.state for o in outs][:first] == [G.HOVER] * first


def test_continuous_pose_stream_is_separate_rate_limited_and_valid_only():
    r = Rig(move_max_hz=10.0)                                                      # 100 ms between moves
    r.grab()
    n_events = len(r.events)
    r.run(10, ratio=0.1, xyz=(0.06, 0.0, 0.0))
    assert len(r.events) == n_events                                               # no discrete events while merely holding
    assert r.moves and all(m.continuous and m.event_type == E.MOVE_TARGET and m.hand_target_xyz_m == (0.06, 0.0, 0.0) for m in r.moves[-3:])
    ts = [m.timestamp_ms for m in r.moves]
    assert all(b - a >= 100 for a, b in zip(ts, ts[1:])) and len(r.moves) <= 7


def test_closed_fist_entering_reach_does_not_grab_until_the_hand_opened_first():
    r = Rig()
    r.run(6, ratio=0.1, xyz=FAR)                                                   # already pinched far away
    r.run(8, ratio=0.1)                                                            # pinched hand enters reach
    assert r.agent.state == G.HOVER and E.GRAB_REQUEST not in r.types()
    r.run(4, ratio=0.9)
    r.run(6, ratio=0.1)                                                            # open, then pinch: now it grabs
    assert r.agent.state == G.GRABBED


def test_pinch_that_completes_inside_reach_without_an_open_hand_seen_there_is_rejected_as_not_armed():
    r = Rig()
    r.step(xyz=FAR, ratio=0.1)                                                     # pinch starts outside reach (candidate)
    r.run(4, ratio=0.1)                                                            # reach is entered; the confirmation completes while hovering
    assert r.agent.state == G.HOVER and E.GRAB_REQUEST not in r.types()
    assert r.agent.counters["grab_rejected:NOT_ARMED"] == 1


def test_ineligible_object_cannot_be_grabbed_and_pinch_is_not_retried_per_frame():
    r = Rig()
    obj = InteractionObject("static_wall", (0.0, 0.0, 0.0), grabbable=False)
    r.run(3, objects=(obj,), ratio=0.9)
    r.run(10, objects=(obj,), ratio=0.1)
    assert r.agent.state == G.HOVER and E.GRAB_REQUEST not in r.types()
    assert r.agent.counters["grab_rejected:OBJECT_NOT_GRABBABLE"] == 1


def test_momentary_threshold_crossing_does_not_grab():
    r = Rig()
    r.hover()
    for ratio in (0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9):
        r.step(ratio=ratio)
    assert r.agent.state == G.HOVER and E.GRAB_REQUEST not in r.types()


def test_pinch_open_releases_then_returns_to_idle_after_settling():
    r = Rig()
    r.grab()
    r.until_released()
    assert r.agent.state == G.RELEASED and r.types()[-1] == E.RELEASE
    rel = r.events[-1]
    assert rel.gesture_state == G.RELEASED and rel.cancel_reason is None and rel.interaction_id == r.events[1].interaction_id
    r.tick(150)
    assert r.agent.state == G.IDLE
    assert r.types().count(E.RELEASE) == 1


def test_released_waits_for_ack_when_required_then_times_out_safely():
    r = Rig(release_ack_required=True, release_ack_timeout_ms=500.0)
    r.grab()
    r.until_released()
    assert r.agent.state == G.RELEASED
    r.tick(300)
    assert r.agent.state == G.RELEASED                                              # still waiting for the world adapter
    rid = r.events[-1].event_id
    r.agent.acknowledge("not-an-id", r.clock.now_ms())
    assert r.agent.state == G.RELEASED and r.agent.counters["ack_ignored"] == 1
    r.agent.acknowledge(rid, r.clock.now_ms())
    assert r.agent.state == G.IDLE
    r2 = Rig(release_ack_required=True, release_ack_timeout_ms=500.0)
    r2.grab(); r2.until_released(); r2.tick(600)
    assert r2.agent.state == G.IDLE and r2.agent.counters["release_ack_timeout"] == 1


def test_tracking_loss_while_grabbed_freezes_then_cancels_safely_within_the_bound():
    r = Rig()                                                                       # defaults: report at 100 ms, cancel at 400 ms
    r.grab()
    last_move_count = len(r.moves)
    held = r.agent._last_pose
    outs = []
    for _ in range(12):
        outs.append(r.step(valid=False))
    types = r.types()
    assert E.TRACKING_LOST in types and E.GRAB_CANCELLED in types
    assert types.index(E.TRACKING_LOST) < types.index(E.GRAB_CANCELLED) and types.count(E.TRACKING_LOST) == 1
    lost, cancelled = [e for e in r.events if e.event_type == E.TRACKING_LOST][0], [e for e in r.events if e.event_type == E.GRAB_CANCELLED][0]
    assert lost.hand_target_xyz_m is None and cancelled.hand_target_xyz_m is None                # never a position without a valid fusion result
    assert cancelled.cancel_reason == CancelReason.TRACKING_LOST_TIMEOUT and cancelled.gesture_state == G.RELEASED
    assert len(r.moves) == last_move_count                                                      # frozen: no movement while lost
    freeze_ms = cancelled.timestamp_ms - lost.timestamp_ms
    assert 0 < freeze_ms <= r.cfg.grab_loss_timeout_ms
    assert r.agent.state in (G.RELEASED, G.IDLE)
    r.tick(200)
    assert r.agent.state == G.IDLE


def test_loss_is_detected_without_any_samples_via_tick():
    r = Rig()
    r.grab()
    for _ in range(10):
        r.tick(60)
    assert E.GRAB_CANCELLED in r.types() and r.agent.state in (G.RELEASED, G.IDLE)


def test_recovery_within_the_timeout_resumes_without_cancelling():
    r = Rig()
    r.grab()
    r.run(3, valid=False)                                                           # 180 ms lost: reported, not yet cancelled
    assert r.agent.state == G.GRABBED and E.TRACKING_LOST in r.types()
    n = len(r.moves)
    r.run(4, ratio=0.1, xyz=(0.08, 0.0, 0.0))
    assert r.agent.state == G.GRABBED and E.GRAB_CANCELLED not in r.types() and len(r.moves) > n


def test_recovery_far_from_the_held_pose_is_a_discontinuity_and_cancels():
    r = Rig()
    r.grab()
    r.run(3, valid=False)
    r.step(ratio=0.1, xyz=(0.9, 0.0, 0.0))                                          # re-acquired > 0.25 m away
    cancel = [e for e in r.events if e.event_type == E.GRAB_CANCELLED]
    assert cancel and cancel[0].cancel_reason == CancelReason.TRACK_DISCONTINUITY and r.agent.state == G.RELEASED


def test_stale_low_confidence_and_nonfinite_poses_are_never_used():
    r = Rig()
    r.grab()
    n = len(r.moves)
    r.clock.advance(600)                                                            # delivery gap: the next pose is newer than the last but 400 ms old
    stale = HandSample(SID, r.clock.now_ms() - 400, True, NEAR, 0.9, 0.1, r.seq + 1, "board_frame")
    r.agent.update(stale, (OBJ,), r.clock.now_ms())
    r.seq += 1
    r.step(conf=0.2)
    r.step(xyz=(float("nan"), 0.0, 0.0))
    assert len(r.moves) == n                                                         # none of the unusable poses produced movement
    assert r.agent.counters["sample_unusable:STALE_POSE"] == 1 and r.agent.counters["sample_unusable:LOW_CONFIDENCE"] == 1
    assert r.agent.counters["sample_unusable:NON_FINITE_POSE"] == 1
    r.step(conf=None)
    assert r.agent.counters["sample_unusable:CONFIDENCE_MISSING"] == 1               # unknown confidence is not good confidence


def test_hover_tracking_loss_exits_after_the_timeout():
    r = Rig()
    r.hover()
    r.run(6, valid=False)
    types = r.types()
    assert E.TRACKING_LOST in types and types[-1] == E.HOVER_EXIT and r.events[-1].reason == "TRACKING_LOST" and r.agent.state == G.IDLE


def test_reach_and_confidence_hysteresis_prevent_hover_flapping():
    r = Rig()
    r.hover()
    for x in (0.13, 0.16, 0.13, 0.17, 0.14):                                       # between enter (0.12) and exit (0.18): stay
        r.step(xyz=(x, 0, 0), ratio=0.9)
    assert r.agent.state == G.HOVER and r.types() == [E.HOVER_ENTER]
    r.step(xyz=(0.2, 0, 0), ratio=0.9)
    assert r.agent.state == G.IDLE and r.types()[-1] == E.HOVER_EXIT and r.events[-1].reason == "OUT_OF_REACH"
    r.step(xyz=(0.13, 0, 0), ratio=0.9)
    assert r.agent.state == G.IDLE                                                  # re-entering needs the ENTER radius again


def test_at_most_one_transition_per_update_and_only_documented_transitions_occur():
    r = Rig()
    r.hover(); r.run(8, ratio=0.1); r.run(6, ratio=0.9); r.tick(200); r.run(4, xyz=FAR)
    legal = {"IDLE->HOVER", "HOVER->IDLE", "HOVER->GRABBED", "GRABBED->RELEASED", "RELEASED->IDLE"}
    seen = {k[len("transition:"):] for k in r.agent.counters if k.startswith("transition:")}
    assert seen == legal                                                            # every documented transition was exercised
    assert seen <= legal and "IDLE->GRABBED" not in seen and "HOVER->RELEASED" not in seen
    for a, b in zip(r.states, r.states[1:]):
        assert (a, b) in {(x, y) for x in G for y in G if f"{x.value}->{y.value}" in legal or x == y}


def test_session_stop_cancels_safely_and_inhibits_everything():
    r = Rig()
    r.grab()
    out = r.agent.on_session_stop(r.clock.now_ms())
    assert [e.event_type for e in out.events] == [E.GRAB_CANCELLED] and out.events[0].cancel_reason == CancelReason.SESSION_STOPPED
    assert r.agent.state == G.IDLE and out.state == G.IDLE
    r.events += out.events
    r.run(6, ratio=0.1)
    assert r.agent.state == G.IDLE and len(r.events) == 4 and r.agent.counters["sample_ignored:SESSION_STOPPED"] == 6


def test_session_stop_while_hovering_and_while_idle_is_safe():
    r = Rig()
    r.hover()
    out = r.agent.on_session_stop(r.clock.now_ms())
    assert [e.event_type for e in out.events] == [E.HOVER_EXIT] and r.agent.state == G.IDLE
    assert Rig().agent.on_session_stop(1).events == []


def test_calibration_invalid_cancels_safely_and_stays_inhibited_until_restored():
    r = Rig()
    r.grab()
    out = r.agent.on_calibration_invalid(r.clock.now_ms())
    assert out.events[0].event_type == E.GRAB_CANCELLED and out.events[0].cancel_reason == CancelReason.CALIBRATION_INVALID and r.agent.state == G.IDLE
    r.run(8, ratio=0.9)
    assert r.agent.state == G.IDLE                                                  # inhibited
    r.agent.on_calibration_restored()
    r.run(4, ratio=0.9)
    assert r.agent.state == G.HOVER


def test_calibration_flag_on_a_sample_cancels_like_the_explicit_call():
    r = Rig()
    r.grab()
    r.clock.advance(STEP)
    out = r.agent.update(r.sample(calibration_valid=False), (OBJ,), r.clock.now_ms())
    assert out.events[0].cancel_reason == CancelReason.CALIBRATION_INVALID and r.agent.state == G.IDLE
    r.events += out.events
    r.run(3, ratio=0.9)                                                             # a later sample with valid calibration lifts the inhibit
    assert r.agent.state in (G.IDLE, G.HOVER) and r.agent.inhibit is None


def test_track_change_cancels_an_active_grab_and_resets_pinch_state():
    r = Rig()
    r.grab()
    r.step(ratio=0.1, track_epoch=1)
    assert r.events[-1].event_type == E.GRAB_CANCELLED and r.events[-1].cancel_reason == CancelReason.TRACK_CHANGED and r.agent.state == G.RELEASED
    assert r.agent.pinch.state == "OPEN"


def test_reset_cancels_then_starts_a_new_event_epoch_with_clean_history():
    r = Rig()
    r.grab()
    old_ids = {e.event_id for e in r.events}
    out = r.agent.reset(r.clock.now_ms())
    assert out.events[0].cancel_reason == CancelReason.RESET and r.agent.state == G.IDLE
    r.events += out.events
    r.seq = 0
    r.hover()
    new_ids = {e.event_id for e in r.events} - old_ids
    assert new_ids and all(":g1:" in i for i in new_ids if i not in old_ids and not out.events[0].event_id == i)
    assert not (old_ids & {e.event_id for e in r.events[len(old_ids) + 1:]})


def test_adapter_ack_flow_confirm_reject_and_timeout():
    r = Rig(require_adapter_ack=True, ack_timeout_ms=300.0)
    r.hover()
    r.run(6, ratio=0.1)
    assert r.agent.state == G.GRABBED and r.types() == [E.HOVER_ENTER, E.GRAB_REQUEST] and not r.moves      # waiting: no motion yet
    out = r.agent.acknowledge(r.events[1].event_id, r.clock.now_ms(), accepted=True)
    assert [e.event_type for e in out.events] == [E.GRAB_CONFIRMED]
    r.events += out.events
    r.step(ratio=0.1)
    assert r.moves
    out = r.agent.acknowledge(r.events[1].event_id, r.clock.now_ms())              # duplicate ack: ignored
    assert not out.events
    r2 = Rig(require_adapter_ack=True)
    r2.grab()
    out = r2.agent.acknowledge(r2.events[1].event_id, r2.clock.now_ms(), accepted=False)
    assert out.events[0].cancel_reason == CancelReason.ADAPTER_REJECTED and r2.agent.state == G.RELEASED
    r3 = Rig(require_adapter_ack=True, ack_timeout_ms=300.0)
    r3.grab()
    for _ in range(8):
        r3.step(ratio=0.1)
    assert [e for e in r3.events if e.event_type == E.GRAB_CANCELLED][0].cancel_reason == CancelReason.ADAPTER_ACK_TIMEOUT


def test_events_are_unique_ordered_json_and_idempotent_for_consumers():
    r = Rig()
    r.grab(); r.run(6, ratio=0.9); r.tick(200)
    all_ev = r.events + r.moves
    ids = [e.event_id for e in all_ev]
    assert len(ids) == len(set(ids))
    seqs = [e.sequence_number for e in sorted(all_ev, key=lambda e: e.sequence_number)]
    assert seqs == list(range(1, len(all_ev) + 1))
    for e in all_ev:
        d = e.to_dict()
        assert d["protocol_version"] == "1.0" and d["message_type"] == "INTERACTION_EVENT" and d["session_id"] == SID and d["unit"] == "m"
        json.dumps(d)
    dd = EventDeduper()
    assert all(dd.accept(e) for e in r.events) and not any(dd.accept(e) for e in r.events)                  # replays are dropped


def test_duplicate_or_out_of_order_samples_are_ignored():
    r = Rig()
    r.hover()
    n = len(r.events)
    r.clock.advance(STEP)
    s = r.sample(ratio=0.1)
    r.agent.update(s, (OBJ,), r.clock.now_ms())                                     # accepted
    r.agent.update(s, (OBJ,), r.clock.now_ms())                                     # same sample again: ignored
    old = HandSample(SID, s.timestamp_ms - 500, True, NEAR, 0.9, 0.1, s.fusion_sequence + 5, "board_frame")
    r.agent.update(old, (OBJ,), r.clock.now_ms())                                   # delivered late, older than what was processed: ignored
    assert r.agent.counters["sample_ignored:OUT_OF_ORDER_OR_DUPLICATE"] == 2 and len(r.events) == n


def test_samples_from_another_session_are_ignored():
    r = Rig()
    r.clock.advance(STEP)
    s = HandSample("sess_other0001", r.clock.now_ms() - 10, True, NEAR, 0.9, 0.9, 1, "board_frame")
    r.agent.update(s, (OBJ,), r.clock.now_ms())
    assert r.agent.counters["sample_ignored:SESSION_MISMATCH"] == 1 and r.agent.state == G.IDLE


def test_event_contract_validation():
    base = dict(session_id=SID, event_id="e", sequence_number=1, timestamp_ms=1, gesture_state=G.GRABBED, interaction_id="i",
                target_object_id="o", world_frame_id="f", confidence=0.5, reason=None)
    with pytest.raises(ValueError):
        InteractionEvent(event_type=E.MOVE_TARGET, continuous=True, hand_target_xyz_m=None, cancel_reason=None, **base)
    with pytest.raises(ValueError):
        InteractionEvent(event_type=E.TRACKING_LOST, continuous=False, hand_target_xyz_m=(0, 0, 0), cancel_reason=None, **base)
    with pytest.raises(ValueError):
        InteractionEvent(event_type=E.GRAB_CANCELLED, continuous=False, hand_target_xyz_m=None, cancel_reason=None, **base)
    with pytest.raises(ValueError):
        InteractionEvent(event_type=E.RELEASE, continuous=False, hand_target_xyz_m=None, cancel_reason=CancelReason.SAFETY, **base)
    with pytest.raises(ValueError):
        InteractionEvent(event_type=E.RELEASE, continuous=False, hand_target_xyz_m=(float("nan"), 0, 0), cancel_reason=None, **base)
    with pytest.raises(ValueError):
        InteractionEvent(event_type=E.RELEASE, continuous=True, hand_target_xyz_m=None, cancel_reason=None, **base)


def test_transitions_are_logged_with_reasons():
    r = Rig()
    r.grab(); r.run(6, ratio=0.9); r.tick(200)
    tr = [(e["from_state"], e["to_state"], e["reason"]) for e in r.agent.audit.of_type("GESTURE_TRANSITION")]
    assert tr[:4] == [("IDLE", "HOVER", "in_reach"), ("HOVER", "GRABBED", "pinch_confirmed"), ("GRABBED", "RELEASED", "pinch_opened"),
                      ("RELEASED", "IDLE", "release_processed")]
