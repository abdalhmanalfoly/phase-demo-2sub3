"""Set selection: geometry, timestamps, calibration compatibility, resource limits, single-view labelling."""
import math

import pytest

from host.app.coordination import simulation as S
from host.app.coordination.coordinator import CoordState
from host.app.coordination.messages import MessageError, Mode, TaskAward
from host.app.coordination.set_selection import SetSelector, ray_gap
from .coord_helpers import Bench, cfg


def _select(sensors=("cam_a", "cam_b", "cam_c", "cam_d"), mutate=None, **cfg_over):
    b = Bench(sensors=sensors)
    b.h.coord.cfg = b.h.cfg
    if mutate:
        mutate(b.h.world.sensors)
    b.c.request_observation()
    task = b.c.outbox[0]
    offers = [o for o in b.offers_for(task).values() if o is not None]
    return SetSelector(cfg(**cfg_over)).select(offers, task), offers, task


def test_two_near_colocated_cameras_are_rejected_even_when_they_have_the_best_scores():
    sel, _, _ = _select(("cam_a", "cam_b"))
    assert sel.best_individual == "cam_a" and sel.chosen is None
    assert sel.outcome == "NO_SET_INSUFFICIENT_GEOMETRY"
    assert sel.evaluated[0].rejection_codes[0] == "INSUFFICIENT_VIEW_ANGLE" and sel.evaluated[0].components["angle_deg"] < 10


def test_selection_considers_geometry_not_just_individual_score():
    sel, _, _ = _select()
    assert sel.best_individual == "cam_a"
    ranked = sorted(sel.individual_quality, key=lambda s: -sel.individual_quality[s])
    assert ranked[:2] == ["cam_a", "cam_b"]                         # the two best individual bids...
    assert sel.chosen.sensor_ids != ("cam_a", "cam_b")              # ...are not paired
    pair = [c for c in sel.evaluated if c.sensor_ids == ("cam_a", "cam_b")][0]
    assert not pair.feasible and "INSUFFICIENT_VIEW_ANGLE" in pair.rejection_codes
    assert "cam_c" in sel.chosen.sensor_ids or "cam_d" in sel.chosen.sensor_ids


def test_set_choice_depends_on_geometry_when_scores_are_equal():
    def equalise(s):
        for x in s.values():
            x.detector_confidence = 0.8
    sel, _, _ = _select(("cam_a", "cam_b", "cam_c"), mutate=equalise)
    assert set(sel.chosen.sensor_ids) & {"cam_c"}                    # the wide-baseline camera must be part of the set


def test_fewer_than_two_eligible_views_never_yields_a_trusted_set():
    sel, offers, _ = _select(("cam_a",))
    assert sel.chosen is None and sel.outcome == "NO_SET_INSUFFICIENT_SENSORS" and len(offers) == 1


def test_timestamp_skew_makes_a_set_infeasible():
    def lag(s):
        s["cam_c"].capture_lag_ms = 120.0                            # 100 ms apart from the others (> 50 ms) but still fresh enough (< 250)
        s["cam_d"].capture_lag_ms = 120.0
    sel, _, _ = _select(mutate=lag)
    for c in sel.evaluated:
        mixed = ({"cam_a", "cam_b"} & set(c.sensor_ids)) and ({"cam_c", "cam_d"} & set(c.sensor_ids))
        if mixed:
            assert not c.feasible and "SKEW_EXCEEDS_WINDOW" in c.rejection_codes
    assert sel.chosen is not None and set(sel.chosen.sensor_ids) <= {"cam_c", "cam_d"} or set(sel.chosen.sensor_ids) <= {"cam_a", "cam_b"}


def test_incompatible_calibration_rig_is_rejected():
    def other_rig(s):
        s["cam_c"].rig_key = "default_rig|rig-OTHER|board_frame|m"
    sel, _, _ = _select(("cam_a", "cam_c"), mutate=other_rig)
    assert sel.chosen is None and sel.evaluated[0].rejection_codes[0] == "INCOMPATIBLE_RIG"


def test_inconsistent_rays_for_the_same_landmark_are_rejected():
    def drift(s):
        s["cam_c"].pixel_bias_px = (0.0, 60.0)                       # cam_c "sees" the landmark far from where cam_a does
    sel, _, _ = _select(("cam_a", "cam_c"), mutate=drift)
    assert sel.chosen is None and sel.evaluated[0].rejection_codes[0] == "RAYS_INCONSISTENT"


def test_hand_label_conflict_latency_budget_and_banned_set():
    def hands(s):
        s["cam_c"].hand = "left"
    sel, _, _ = _select(("cam_a", "cam_c"), mutate=hands)
    assert "HAND_LABEL_MISMATCH" in sel.evaluated[0].rejection_codes
    sel, _, _ = _select(("cam_a", "cam_c"), mutate=lambda s: setattr(s["cam_c"], "transport_delay_ms", 300.0))
    assert sel.chosen is None                                         # 300 ms exceeds the 150 ms per-sensor budget (offer or set level)
    b = Bench(sensors=("cam_a", "cam_c"))
    b.c.request_observation()
    task = b.c.outbox[0]
    offers = [o for o in b.offers_for(task).values()]
    banned = {frozenset({"cam_a", "cam_c"})}
    sel = SetSelector(cfg()).select(offers, task, banned)
    assert sel.chosen is None and sel.evaluated[0].rejection_codes[0] == "RECENTLY_FAILED_SET"


def test_resource_limit_max_set_size():
    sel, _, _ = _select(**{"set_selection.max_set_size": 2})
    assert all(len(c.sensor_ids) == 2 for c in sel.evaluated) and len(sel.chosen.sensor_ids) == 2


def test_selection_is_deterministic_and_explainable():
    a, _, _ = _select()
    b, _, _ = _select()
    assert a.chosen.sensor_ids == b.chosen.sensor_ids and a.chosen.score == b.chosen.score
    text = a.explain()
    assert "REJECTED INSUFFICIENT_VIEW_ANGLE" in text and "decision: AWARD" in text and "individual quality" in text


def test_ray_gap_geometry_helper():
    gap, front = ray_gap((0, 0, 0), (1, 0, 0), (2, -1, 0), (0, 1, 0))       # rays cross at (2, 0, 0), in front of both origins
    assert gap == pytest.approx(0.0, abs=1e-9) and front
    gap, front = ray_gap((0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0))        # skew lines 1 m apart
    assert gap == pytest.approx(1.0)
    gap, front = ray_gap((0, 0, 0), (1, 0, 0), (-2, -1, 0), (0, 1, 0))      # crossing point is BEHIND the first origin
    assert gap == pytest.approx(0.0, abs=1e-9) and not front


def test_single_view_is_refused_by_default_and_reported_with_a_reason():
    h = S._tune(S.SimHarness(S._cfg(), ("cam_a", "cam_b", "cam_c")))
    for sid in ("cam_b", "cam_c"):
        h.world.sensors[sid].visible = False
    h.coord.request_observation()
    h.run(400)
    assert h.coord.state == CoordState.DEGRADED and h.coord.lease is None
    rej = h.events("TASK_REJECT")[0]
    assert rej["reason"] == "NO_SET_INSUFFICIENT_SENSORS"
    assert h.allowed["wrist"] == frozenset()                          # nothing may feed the target


def test_single_view_approximate_mode_is_labelled_and_never_trusted_when_enabled():
    c = S._cfg()
    c.adaptation.allow_single_view_approximate = True
    h = S._tune(S.SimHarness(c, ("cam_a", "cam_b", "cam_c")))
    for sid in ("cam_b", "cam_c"):
        h.world.sensors[sid].visible = False
    h.coord.request_observation()
    h.run(1000)
    award = [m for m in h.coord.outbox if m.message_type == "TASK_AWARD"][0]
    assert award.mode == Mode.SINGLE_VIEW_APPROXIMATE and award.trusted_3d is False and award.sensor_ids == ("cam_a",)
    deg = [m for m in h.coord.outbox if m.message_type == "DEGRADED_STATUS"][0]
    assert deg.tracking_trusted is False and deg.mode == Mode.SINGLE_VIEW_APPROXIMATE and "NOT calibrated triangulation" in deg.detail
    assert h.coord.lease.mode == Mode.SINGLE_VIEW_APPROXIMATE and h.coord.lease.poor == 0    # fusion reporting a single view is expected, not a failure
    assert h.coord.counters["fusion_verdict:ACCEPTED"] == 0                                    # never "accepted" as triangulation


def test_award_message_refuses_to_claim_trusted_triangulation_from_one_sensor():
    with pytest.raises(MessageError):
        TaskAward(session_id=S.SESSION, task_id="t", award_id="a", sequence_number=1, decision_epoch=1, target_landmark="wrist",
                  sensor_ids=("cam_a",), mode=Mode.TRIANGULATION, trusted_3d=True, awarded_at_ms=1000, lease_expires_at_ms=2000,
                  set_score=0.5, reason="x")
    with pytest.raises(MessageError):
        TaskAward(session_id=S.SESSION, task_id="t", award_id="a", sequence_number=1, decision_epoch=1, target_landmark="wrist",
                  sensor_ids=("cam_a", "cam_c"), mode=Mode.TRIANGULATION, trusted_3d=True, awarded_at_ms=1000, lease_expires_at_ms=1000,
                  set_score=0.5, reason="x")                          # zero-length lease
