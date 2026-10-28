"""Coordinator protocol: deadlines, duplicates/delays, leases, state machine, message contracts, audit/privacy."""
import json

import pytest

from host.app.coordination import simulation as S
from host.app.coordination.audit import AuditLog
from host.app.coordination.coordinator import ALLOWED, CoordState, Coordinator
from host.app.coordination.messages import MESSAGE_TYPES, SensorStatusMsg, TaskRequest
from .coord_helpers import Bench, outcome

CS = CoordState


def _announce(b, sensors=None):
    b.h.world.sensors  # noqa
    b.c.request_observation()
    task = b.c.outbox[0]
    return task


def _bench_no_auto():
    """Bench whose bus does NOT answer automatically, so each offer is delivered by hand."""
    b = Bench()
    b.c.subscribers.clear()
    return b


def test_task_request_has_id_deadline_target_session_and_bounded_lease():
    b = Bench()
    task = _announce(b)
    assert task.task_id == "task_00001" and task.session_id == S.SESSION and task.target_landmark == "wrist"
    assert task.deadline_ms > task.issued_at_ms and task.collect_until_ms <= task.deadline_ms and task.max_lease_ms <= 5000
    assert task.window_start_ms < task.window_end_ms == task.issued_at_ms
    d = task.to_dict()
    assert d["message_type"] == "TASK_REQUEST" and d["protocol_version"] == "1.0" and d["reason"] == "INITIAL"
    assert json.dumps(d)


def test_task_ids_are_unique_and_a_second_request_during_negotiation_is_idempotent():
    b = _bench_no_auto()
    t1 = b.c.request_observation()
    assert b.c.request_observation() == t1
    b.h.clock.advance(200)
    b.c.tick()                                                          # collection closes -> no offers -> task rejected
    t2 = b.c.request_observation()
    assert t2 != t1


def test_offer_after_the_deadline_is_rejected():
    b = _bench_no_auto()
    b.c.request_observation()
    task = b.c.outbox[0]
    late = b.offers_for(task)["cam_a"]
    b.h.clock.advance(int(task.deadline_ms - task.issued_at_ms) + 1)
    assert b.c.on_offer(late) == "OFFER_LATE"
    assert "cam_a" not in b.c.tasks[task.task_id].offers
    assert b.c.counters["offer_rejected:OFFER_LATE"] == 1


def test_duplicate_and_delayed_offers_cannot_overwrite_a_decision():
    b = _bench_no_auto()
    b.c.request_observation()
    task = b.c.outbox[0]
    offers = b.offers_for(task)
    assert b.c.on_offer(offers["cam_a"]) == "ACCEPTED"
    first = b.c.tasks[task.task_id].offers["cam_a"]
    again = b.offers_for(task)["cam_a"]                                 # same sensor, same task, higher offer_sequence
    assert b.c.on_offer(again) == "DUPLICATE_OFFER" and b.c.on_offer(offers["cam_a"]) == "DUPLICATE_OFFER"
    assert b.c.tasks[task.task_id].offers["cam_a"] is first
    for sid in ("cam_b", "cam_c", "cam_d"):
        b.c.on_offer(offers[sid])
    decided = b.c.lease
    assert decided is not None and b.c.state == CS.ACTIVE_LEASE
    assert b.c.on_offer(offers["cam_c"]) == "TASK_CLOSED"             # arrives after the decision
    assert b.c.lease is decided and b.c.counters["awards"] == 1       # decision untouched


def test_offers_for_unknown_wrong_session_or_superseded_tasks_are_rejected():
    b = _bench_no_auto()
    b.c.request_observation()
    task = b.c.outbox[0]
    offer = b.offers_for(task)["cam_a"]
    d = offer.to_dict()
    assert b.c.on_offer({**d, "task_id": "task_99999"}) == "UNKNOWN_TASK"
    assert b.c.on_offer({**d, "session_id": "sess_other0001"}) == "SESSION_MISMATCH"
    assert b.c.on_offer({**d, "sensor_id": "cam_zzz"}) == "UNKNOWN_SENSOR"
    assert b.c.on_offer({**d, "detector_confidence": None}).startswith("INVALID_OFFER:")
    assert b.c.on_offer("not an offer").startswith("INVALID_OFFER:")
    assert b.c.on_offer({}).startswith("INVALID_OFFER:")


def test_eligibility_rejections_carry_codes():
    b = _bench_no_auto()
    w = b.h.world.sensors
    w["cam_a"].capture_lag_ms = 600.0            # stale
    w["cam_b"].visible = False                   # target not visible
    w["cam_c"].calibration_ok = False            # uncalibrated
    w["cam_d"].state = "degraded"; w["cam_d"].update_rate_hz = 0.0     # unhealthy (0.4 * 0.5 = 0.2 < 0.3)
    b.h.status_all()
    b.c.request_observation()
    task = b.c.outbox[0]
    got = {sid: b.c.on_offer(o) for sid, o in b.offers_for(task).items()}
    assert got["cam_a"] == "STALE_OBSERVATION" and got["cam_b"] == "TARGET_NOT_VISIBLE"
    assert got["cam_c"] == "NOT_CALIBRATED" and got["cam_d"] == "SENSOR_UNHEALTHY"


def test_incomplete_offer_without_geometry_is_rejected_for_trusted_triangulation():
    b = _bench_no_auto()
    b.c.request_observation()
    task = b.c.outbox[0]
    d = b.offers_for(task)["cam_a"].to_dict()
    d["view_ray"], d["rig_key"] = None, None
    assert b.c.on_offer(d) == "INCOMPLETE_OFFER"


def test_lease_is_bounded_and_expiry_removes_control():
    c = S._cfg()
    c.task.lease_ms = 1000.0
    h = S._tune(S.SimHarness(c, auto_renew=False))
    h.coord.request_observation()
    h.run(300)
    lease = h.coord.lease
    assert lease is not None and lease.expires_ms - lease.started_ms <= 5000 and h.allowed["wrist"] == frozenset(lease.sensor_ids)
    h.run(1000)
    assert h.coord.lease is None and h.allowed["wrist"] == frozenset()          # expired lease controls nothing
    expired = [m for m in h.coord.outbox if m.message_type == "LEASE_EXPIRED"]
    assert expired and expired[-1].reason == "LEASE_TIMEOUT" and expired[-1].sensor_ids == lease.award.sensor_ids
    assert h.coord.state == CS.IDLE
    n = h.coord.counters["fusion_verdict:ACCEPTED"]
    h.run(300)
    assert h.coord.counters["fusion_verdict:ACCEPTED"] == n                     # results after expiry are ignored
    assert h.coord.counters["fusion_ignored:NO_ACTIVE_LEASE"] > 0


def test_lease_duration_above_the_cap_is_refused_by_configuration():
    from host.app.coordination.config import ConfigError
    c = S._cfg()
    c.task.lease_ms = 6000.0
    with pytest.raises(ConfigError):
        c.validate()


def test_duplicate_delayed_and_foreign_fusion_results_cannot_corrupt_the_decision():
    h = S._tune(S.SimHarness(S._cfg(), feed_fusion=False))
    h.coord.request_observation()
    h.run(100)
    now = h.clock.now_ms()
    c = h.coord
    assert c.on_fusion_outcome(outcome(10, now)).value == "ACCEPTED"
    assert c.on_fusion_outcome(outcome(10, now)) is None                         # duplicate sequence
    assert c.on_fusion_outcome(outcome(9, now, valid=False, status="TARGET_LOST")) is None   # delayed (older) result must not count as failure
    assert c.lease.poor == 0
    assert c.on_fusion_outcome(outcome(11, now - 5000)) is None                  # published before the lease
    assert c.on_fusion_outcome(outcome(12, now, sources=("cam_a", "cam_zzz"))) is None   # sources outside the lease
    assert c.on_fusion_outcome(outcome(13, now, target="thumb_tip")) is None
    assert c.counters["fusion_ignored:DUPLICATE_OR_STALE_SEQUENCE"] == 2


def test_coordinator_state_machine_follows_the_documented_transitions():
    h = S._tune(S.SimHarness(S._cfg()))
    seen = [h.coord.state]
    h.coord.subscribe(lambda m: None)
    h.coord.request_observation()
    h.run(200)
    states = [e["to_state"] for e in h.events("COORDINATOR_STATE")]
    assert states == ["ANNOUNCING", "COLLECTING_OFFERS", "SELECTING_SET", "ACTIVE_LEASE"]
    for e in h.events("COORDINATOR_STATE"):
        assert CS(e["to_state"]) in ALLOWED[CS(e["from_state"])]
    assert ALLOWED[CS.STOPPED] == frozenset()
    with pytest.raises(RuntimeError):
        h.coord._transition(CS.COLLECTING_OFFERS, "illegal")                     # ACTIVE_LEASE -> COLLECTING is not allowed


def test_stop_ends_the_lease_clears_the_gate_and_ignores_everything_afterwards():
    h = S._tune(S.SimHarness(S._cfg()))
    h.coord.request_observation()
    h.run(300)
    h.coord.stop("SESSION_STOPPED")
    assert h.coord.state == CS.STOPPED and h.coord.lease is None and h.allowed["wrist"] == frozenset()
    assert [m for m in h.coord.outbox if m.message_type == "LEASE_EXPIRED"][-1].reason == "SESSION_STOPPED"
    assert h.coord.request_observation() is None
    assert h.coord.on_offer({}) == "COORDINATOR_STOPPED"
    assert h.coord.on_fusion_outcome(outcome(99, h.clock.now_ms())) is None
    h.run(2000)
    assert h.coord.state == CS.STOPPED and not h.events("TASK_AWARD")[1:]


def test_stale_sensor_status_cannot_overwrite_a_newer_one():
    h = S.SimHarness(S._cfg(), ("cam_a",))
    t = h.clock.now_ms()
    assert h.coord.on_sensor_status(SensorStatusMsg(session_id=S.SESSION, sensor_id="cam_a", sequence_number=10, observed_at_ms=t, state="disconnected")) == "OK"
    assert h.coord.on_sensor_status(SensorStatusMsg(session_id=S.SESSION, sensor_id="cam_a", sequence_number=5, observed_at_ms=t, state="online")) == "STALE_SEQUENCE"
    assert h.coord.sensors["cam_a"].state == "disconnected"
    assert h.coord.on_sensor_status(SensorStatusMsg(session_id="sess_other0001", sensor_id="cam_a", sequence_number=99, observed_at_ms=t, state="online")) == "SESSION_MISMATCH"


def test_every_message_type_is_versioned_serialisable_and_registered():
    h = S._tune(S.SimHarness(S._cfg(), ("cam_a", "cam_b", "cam_c", "cam_d")))
    h.coord.request_observation()
    h.run(200)
    h.disconnect("cam_d")
    h.coord.stop()
    types = {m.message_type for m in h.coord.outbox}
    assert {"TASK_REQUEST", "TASK_AWARD", "LEASE_EXPIRED", "FUSION_RESULT"} <= types
    for m in h.coord.outbox:
        d = m.to_dict()
        assert d["protocol_version"] == "1.0" and d["channel"] == "coordination" and d["message_type"] in MESSAGE_TYPES and d["session_id"] == S.SESSION
        assert isinstance(d.get("sequence_number"), int)
        json.dumps(d)
    seqs = [m.sequence_number for m in h.coord.outbox]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)                 # strictly increasing outbound sequence
    assert set(MESSAGE_TYPES) == {"TASK_REQUEST", "OBSERVATION_OFFER", "TASK_AWARD", "TASK_REJECT", "LEASE_EXPIRED", "FUSION_RESULT",
                                  "DEGRADED_STATUS", "SENSOR_STATUS"}


def test_audit_log_records_the_negotiation_and_refuses_image_like_fields():
    h = S._tune(S.SimHarness(S._cfg()))
    h.coord.request_observation()
    h.run(200)
    for et in ("TASK_REQUEST", "OFFER_ACCEPTED", "SET_SELECTION", "TASK_AWARD", "LEASE_STARTED", "FUSION_EVALUATED", "COORDINATOR_STATE"):
        assert h.events(et), et
    sel = h.events("SET_SELECTION")[0]
    assert sel["outcome"] == "AWARD" and sel["candidates"] and sel["individual_quality"] and sel["stage_ms"] is not None
    acc = h.events("OFFER_ACCEPTED")[0]
    assert {"quality", "detector_confidence", "visibility", "freshness", "sync", "health", "latency_ms"} <= set(acc)
    assert h.coord.status()["stage_timing"]["select_set"]["calls"] == 1
    with pytest.raises(ValueError):
        h.audit.log("X", frame=[1, 2, 3])
    with pytest.raises(ValueError):
        h.audit.log("X", image_data="...")


def test_jsonl_audit_file_is_written_and_contains_only_json_lines(tmp_path=None):
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    h = S._tune(S.SimHarness(S._cfg(), jsonl_path=d / "coordination.jsonl"))
    h.coord.request_observation()
    h.run(200)
    h.audit.close()
    lines = (d / "coordination.jsonl").read_text().splitlines()
    assert len(lines) > 5 and all(json.loads(l)["component"] == "coordinator" for l in lines)
