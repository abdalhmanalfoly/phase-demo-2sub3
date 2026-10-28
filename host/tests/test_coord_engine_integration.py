"""Coordinator + the REAL Phase 2 ``FusionEngine`` (no stand-in). NEEDS pydantic (skipped without it).

NOTE: written against the engine's public API but NOT executed in the sandbox that produced Phase 3 (pydantic/fastapi unavailable offline).
Run it first on the real environment: ``pytest host/tests/test_coord_engine_integration.py -q``.
"""
import pytest

pytest.importorskip("pydantic")

import numpy as np                                                                      # noqa: E402

from host.app.coordination.config import CoordinationCfg                                # noqa: E402
from host.app.coordination.coordinator import CoordState, Coordinator                   # noqa: E402
from host.app.coordination.fusion_port import FusionOutcome                             # noqa: E402
from host.app.coordination.messages import SensorStatusMsg                              # noqa: E402
from host.app.coordination.sensor_agent import CalibrationView, ObservationView, SensorAgent, SensorSnapshot   # noqa: E402
from host.app.spatial import geometry as G                                              # noqa: E402
from .test_spatial_fusion import A, B, LEFT, OVERHEAD, RIGHT, Rig, X                    # noqa: E402

SESSION = "sess_test0001"                                                               # the Rig's engine session id
OVER = "overhead"


class RigSource:
    def __init__(self, rig, visible):
        self.rig, self.visible = rig, visible

    def snapshot(self, sid, now_ms):
        cam = self.rig.cams[sid]
        obs = None
        if self.visible[sid]:
            uv = G.project(cam, X)[0] + 0.5
            obs = ObservationView(now_ms - 20, cam.width, cam.height, "tracked", "right", {"wrist": (float(uv[0]), float(uv[1]), None)}, (), 10.0)
        return SensorSnapshot(sid, "online", ("wrist",), 15.0, True, 5.0, 15.0, obs, CalibrationView(True, None, cam, "default_rig|rig-1|board_frame|m"))


def test_award_gates_the_real_engine_and_results_are_evaluated(tmp_path):
    rig = Rig(tmp_path, cams=(LEFT, RIGHT, OVERHEAD))
    cfg = CoordinationCfg()
    cfg.task.lease_ms = 4000.0
    cfg.set_selection.min_view_angle_deg = 10.0
    cfg.validate()
    visible = {A: True, B: True, OVER: True}
    source = RigSource(rig, visible)
    coord = Coordinator(SESSION, cfg, rig.clock, fusion_gate=rig.engine.set_sensor_filter)
    rig.engine.subscribe(lambda r: coord.on_fusion_outcome(FusionOutcome.from_fusion_result(r)))
    agents = {sid: SensorAgent(sid, source, cfg, rig.clock) for sid in visible}
    coord.subscribe(lambda m: [coord.on_offer(o) for o in (ag.on_task_request(m) for ag in agents.values()) if o] if m.message_type == "TASK_REQUEST" else None)
    seq = 0

    def frame():
        nonlocal seq
        seq += 1
        for sid in visible:
            if visible[sid]:
                rig.feed(sid, X)
            coord.on_sensor_status(SensorStatusMsg(session_id=SESSION, sensor_id=sid, sequence_number=seq, observed_at_ms=rig.clock.now_ms(), state="online"))
        coord.tick()
        rig.advance(66)

    frame()
    coord.request_observation()
    for _ in range(12):
        frame()
    assert coord.state == CoordState.ACTIVE_LEASE and len(coord.lease.sensor_ids) >= 2
    assert rig.engine.sensor_filter["wrist"] == frozenset(coord.lease.sensor_ids)       # the award really gates the engine
    assert coord.counters["fusion_verdict:ACCEPTED"] > 3
    coord.stop()
    assert rig.engine.sensor_filter["wrist"] == frozenset()                              # stopped coordinator controls nothing
