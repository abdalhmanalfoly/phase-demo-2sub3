"""Adaptive behaviour, end to end through the deterministic simulation (real SensorAgent + Coordinator + SetSelector)."""
import pytest

from host.app.coordination import simulation as S
from host.app.coordination.coordinator import CoordState
from host.app.coordination.messages import Verdict

CS = CoordState


def _rejections(h, task_id=None):
    return {(e["sensor_id"], e["reason"]) for e in h.events("OFFER_REJECTED") if task_id in (None, e["task_id"])}


def test_scenario1_best_individual_camera_cannot_be_paired_with_a_near_colocated_one():
    h = S.SCENARIOS["1_geometry_beats_individual_score"]()
    sel = h.events("SET_SELECTION")[0]
    iq = sel["individual_quality"]
    assert max(iq, key=iq.get) == "cam_a"
    assert sorted(h.lease_sensors()) != ["cam_a", "cam_b"] and h.coord.state == CS.ACTIVE_LEASE
    pair = [c for c in sel["candidates"] if c["set"] == ["cam_a", "cam_b"]][0]
    assert pair["feasible"] is False and pair["codes"][0] == "INSUFFICIENT_VIEW_ANGLE" and pair["angle_deg"] < 10
    chosen = [c for c in sel["candidates"] if c["set"] == sel["chosen"]][0]
    assert chosen["feasible"] and chosen["angle_deg"] > 30 and chosen["score"] > 0.9
    assert h.coord.last_selection.explain()                                        # operator-readable explanation exists


def test_scenario1b_only_poor_geometry_reports_why_no_trusted_3d_exists():
    h = S.SCENARIOS["1b_only_poor_geometry"]()
    assert h.coord.state == CS.DEGRADED and h.coord.lease is None
    rej = h.events("TASK_REJECT")[0]
    assert rej["reason"] == "NO_SET_INSUFFICIENT_GEOMETRY"
    deg = [m for m in h.coord.outbox if m.message_type == "DEGRADED_STATUS"][0]
    assert deg.tracking_trusted is False and deg.reason_code == "NO_SET_INSUFFICIENT_GEOMETRY" and deg.retry_at_ms is not None
    assert h.allowed["wrist"] == frozenset()
    assert h.coord.counters["awards"] == 0


def test_scenario2_occlusion_changes_the_selected_set():
    h = S.SCENARIOS["2_occlusion"]()
    first, second = h.awards()[0], h.awards()[1]
    occluded = first[0]
    assert occluded not in second and h.lease_sensors() == second
    assert (occluded, "TARGET_NOT_VISIBLE") in _rejections(h)
    assert h.coord.state == CS.ACTIVE_LEASE
    ad = h.events("ADAPTATION")[0]
    assert ad["reason"] == "FUSION_FAILURE" and ad["failed_set"] == list(first)
    assert h.coord.counters["fusion_verdict:ACCEPTED"] > 10                        # tracking trusted again after the change


def test_scenario3_delayed_and_stale_offers_are_rejected_with_reasons():
    h = S.SCENARIOS["3_delay_and_stale"]()
    rej = _rejections(h)
    assert ("cam_a", "STALE_OBSERVATION") in rej and ("cam_c", "OFFER_LATE") in rej
    assert ("cam_c", "NO_OFFER") in rej                                            # it also missed the collection window
    assert h.awards() == [("cam_b", "cam_d")]                                      # neither stale nor late sensor was used
    assert "cam_a" not in h.lease_sensors() and "cam_c" not in h.lease_sensors()
    stale = [e for e in h.events("OFFER_REJECTED") if e["reason"] == "STALE_OBSERVATION"][0]
    assert stale["observation_age_ms"] > 250 and stale["freshness"] == 0.0


def test_scenario4_disconnected_camera_is_dropped_and_stays_ineligible():
    h = S.SCENARIOS["4_disconnect"]()
    victim = h.victim
    assert victim not in h.lease_sensors() and len(h.awards()) >= 2 and victim not in h.awards()[-1]
    assert h.coord.state == CS.ACTIVE_LEASE
    later = [e for e in h.events("TASK_REQUEST") if e["ts_ms"] >= h.events("LEASE_EXPIRED")[0]["ts_ms"]]
    assert later and all(victim in e["unavailable"] for e in later)                # not even asked: it is unavailable


def _lease_supervision(silent):
    h = S._tune(S.SimHarness(S._cfg(), feed_fusion=False))                         # no fusion feedback: only liveness can end the lease
    h.coord.request_observation()
    h.run(300)
    victim = h.lease_sensors()[-1]
    t0 = h.clock.now_ms()
    h.go_silent(victim) if silent else h.disconnect(victim)
    return h, victim, t0


def test_disconnect_becomes_ineligible_only_after_the_grace_timeout():
    h, victim, t0 = _lease_supervision(False)
    grace = h.cfg.adaptation.disconnect_grace_ms
    h.run(int(grace) - 100)
    assert victim in h.lease_sensors()                                             # still inside the grace period (it may reconnect)
    h.run(200)
    assert h.coord.lease is None or victim not in h.lease_sensors()
    exp = [e for e in h.events("LEASE_EXPIRED") if e["reason"] == "SENSOR_UNAVAILABLE"]
    assert exp and grace <= exp[0]["ts_ms"] - t0 <= grace + 100


def test_silent_sensor_becomes_ineligible_after_the_silence_timeout():
    h, victim, t0 = _lease_supervision(True)
    timeout = h.cfg.adaptation.sensor_silence_timeout_ms
    h.run(int(timeout) - 600)
    assert victim in h.lease_sensors()
    h.run(1000)
    exp = [e for e in h.events("LEASE_EXPIRED") if e["reason"] == "SENSOR_UNAVAILABLE"]
    assert exp and victim not in (h.lease_sensors() or ())
    assert h.events("ADAPTATION")[0]["detail"].startswith("silent for")


def test_reconnect_inside_the_grace_period_keeps_the_lease():
    h, victim, t0 = _lease_supervision(False)
    h.run(300)
    h.world.sensors[victim].state = "online"
    h.status_all()
    h.run(1500)
    assert victim in h.lease_sensors() and not [e for e in h.events("LEASE_EXPIRED") if e["reason"] == "SENSOR_UNAVAILABLE"]


def test_scenario5a_excessive_reprojection_error_triggers_a_new_set():
    h = S.SCENARIOS["5a_fusion_failure_recoverable"]()
    bad = h.bad[0]
    ev = [e for e in h.events("FUSION_EVALUATED") if e["verdict"] == "POOR_QUALITY"]
    assert ev and ev[0]["reason"] == "REPROJECTION_ERROR_HIGH" and ev[0]["rms_px"] > 4.0
    assert len(h.awards()) >= 2 and h.awards()[0] != h.awards()[1]                 # the failed set was not simply re-awarded
    banned_first = [a for a in h.events("ADAPTATION") if a["reason"] == "FUSION_FAILURE"][0]["failed_set"]
    assert tuple(banned_first) not in h.awards()[1:2]
    assert bad not in h.lease_sensors() and h.coord.state == CS.ACTIVE_LEASE and h.coord.lease.last_verdict[0] == Verdict.ACCEPTED


def test_scenario5b_unrecoverable_fusion_reports_degraded_and_never_a_trusted_result():
    h = S.SCENARIOS["5b_fusion_failure_unrecoverable"]()
    deg = h.events("DEGRADED")
    assert deg and deg[0]["reason"] == "FUSION_UNRELIABLE" and deg[0]["tracking_trusted"] is False
    assert h.coord.counters["fusion_verdict:ACCEPTED"] <= 6                         # only results from before the drift were ever accepted
    assert not [e for e in h.events("FUSION_EVALUATED") if e["verdict"] == "ACCEPTED" and e["ts_ms"] > S.T0 + 500]
    n_failures = len([a for a in h.events("ADAPTATION") if a["reason"] == "FUSION_FAILURE"])
    first_degrade = deg[0]["ts_ms"]
    before = [a for a in h.events("ADAPTATION") if a["ts_ms"] <= first_degrade]
    assert len(before) == h.cfg.adaptation.max_reselect_attempts + 1               # bounded re-selection
    assert n_failures >= len(before)


def test_degraded_retry_backs_off_exponentially_up_to_the_cap():
    h = S.SCENARIOS["1b_only_poor_geometry"]()
    h.run(12000)
    reqs = [e["ts_ms"] for e in h.events("TASK_REQUEST")]
    gaps = [b - a for a, b in zip(reqs, reqs[1:])]
    assert gaps[0] >= 500 and gaps[1] >= 2 * gaps[0] - 20 and max(gaps) <= h.cfg.adaptation.degraded_retry_max_ms + 100
    assert all(g >= 480 for g in gaps)


def test_degraded_state_recovers_when_a_compatible_set_becomes_available():
    h = S.SCENARIOS["1b_only_poor_geometry"]()
    assert h.coord.state == CS.DEGRADED
    h.add_sensor("cam_c")
    h.world.sensors["cam_c"].detector_confidence = 0.7
    h.status_all()
    h.run(2500)
    assert h.coord.state == CS.ACTIVE_LEASE and "cam_c" in h.lease_sensors() and h.coord.degraded is None


def test_scenario6_lease_renewal_hands_over_without_a_gap_and_supersedes_the_old_lease():
    h = S.SCENARIOS["6_lease_renewal"]()
    reasons = [e["reason"] for e in h.events("LEASE_EXPIRED")]
    assert reasons and set(reasons) == {"SUPERSEDED"}                              # never LEASE_TIMEOUT: successor arrived first
    assert len(h.awards()) >= 3
    awards = [m for m in h.coord.outbox if m.message_type == "TASK_AWARD"]
    assert [a.decision_epoch for a in awards] == list(range(1, len(awards) + 1))
    assert all(a.lease_expires_at_ms - a.awarded_at_ms <= 5000 for a in awards)
    assert h.allowed["wrist"]                                                      # the gate is never left empty between leases
    assert not h.events("DEGRADED")
    req = [e for e in h.events("TASK_REQUEST") if e["reason"] == "RENEWAL"][0]
    assert req["parent_task_id"] == "task_00001"


def test_renewal_declined_keeps_the_old_lease_until_it_expires_then_degrades():
    c = S._cfg()
    c.task.lease_ms, c.task.renew_lead_ms = 1500.0, 400.0
    h = S._tune(S.SimHarness(c, ("cam_a", "cam_c")))
    h.coord.request_observation()
    h.run(900)
    h.world.sensors["cam_c"].visible = False                                        # successor negotiation can no longer find a pair...
    h.feed_fusion = False                                                           # ...and fusion feedback is not what we test here
    h.run(500)
    assert [e for e in h.events("TASK_REJECT")]
    assert h.coord.state in (CS.ACTIVE_LEASE, CS.DEGRADED)
    h.run(1500)
    assert h.coord.state == CS.DEGRADED and h.coord.lease is None and h.allowed["wrist"] == frozenset()


def test_calibration_invalidation_cancels_the_lease_and_the_set_changes():
    h = S._tune(S.SimHarness(S._cfg()))
    h.coord.request_observation()
    h.run(300)
    victim = h.lease_sensors()[0]
    h.world.sensors[victim].calibration_ok = False
    h.status_all()                                                                  # SENSOR_STATUS carries calibration_ok=False
    assert [e for e in h.events("LEASE_EXPIRED") if e["reason"] == "CALIBRATION_INVALID"]       # cancelled immediately, not after a timeout
    h.run(500)
    assert victim not in h.lease_sensors() and h.coord.state == CS.ACTIVE_LEASE
    assert (victim, "NOT_CALIBRATED") in _rejections(h)                             # its fresh offer is refused, by two independent checks


def test_simulation_is_deterministic():
    a = S.SCENARIOS["2_occlusion"]()
    b = S.SCENARIOS["2_occlusion"]()
    strip = lambda h: [{k: v for k, v in e.items()} for e in h.audit.entries]
    assert strip(a) == strip(b)
    assert [m.to_dict() for m in a.coord.outbox] == [m.to_dict() for m in b.coord.outbox]
