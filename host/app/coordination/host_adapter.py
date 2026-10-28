"""Glue between the REAL host objects (Session / SensorRegistry / FusionEngine / CalibrationRegistry) and the coordination core.

NOT EXERCISED BY THE AUTOMATED TESTS IN THIS SANDBOX (needs pydantic/fastapi, which were not installable offline); it is
duck-typed, deliberately thin, and covered by ``host/tests/test_coord_engine_integration.py`` which is skipped when pydantic is missing.
"""
from __future__ import annotations

import statistics

from .audit import AuditLog
from .config import CoordinationCfg
from .coordinator import Coordinator
from .fusion_port import FusionOutcome
from .messages import SensorStatusMsg
from .sensor_agent import CalibrationView, ObservationView, SensorAgent, SensorSnapshot


class HostSensorSource:
    """Builds ``SensorSnapshot`` from a live ``Session`` (registry + synchronizer buffers + calibration registry)."""

    def __init__(self, session):
        self.session = session

    def snapshot(self, sensor_id: str, now_ms: int) -> SensorSnapshot | None:
        s = self.session
        rec = s.registry.records.get(sensor_id)
        if rec is None or not rec.connected or s.fusion is None:
            return None
        sync = s.fusion.engine.sync
        buf = sync.buffers.get(sensor_id)
        newest = buf[-1] if buf else None
        delays = list(sync.delays.get(sensor_id, ()))
        jitter = (max(delays) - min(delays)) if len(delays) >= 5 else None
        transport = statistics.median(delays) if delays else None
        obs = None if newest is None else ObservationView(
            effective_ms=newest.effective_ms, image_width=newest.image_width, image_height=newest.image_height, status=newest.status,
            hand=newest.hand, landmarks=dict(newest.landmarks), flags=(), inference_ms=None)
        reg = s.fusion.registry
        res = reg.resolve(sensor_id, None if newest is None else newest.image_width, None if newest is None else newest.image_height)
        art = res.artifact
        rig = None if art is None else f"{art.rig_id}|{art.rig_calibration_id}|{art.world_frame_id}|{art.unit}"
        cal = CalibrationView(eligible=res.problem is None, problem=res.problem, camera_model=res.model, rig_key=rig)
        state = s.registry.snapshot(rec, now_ms)["state"]
        return SensorSnapshot(sensor_id=sensor_id, state=state, capabilities=tuple(rec.capabilities.get("landmarks", ())),
                              update_rate_hz=rec.rate.rate_hz(now_ms), clock_offset_known=rec.offset.offset_ms is not None,
                              delay_jitter_ms=jitter, transport_delay_ms=transport, observation=obs, calibration=cal)


class SessionCoordination:
    """Attach a Coordinator to a live Session: fusion gate, result feed, sensor-status feed, agents + bus."""

    def __init__(self, session, cfg: CoordinationCfg, clock, events=None):
        self.session, self.cfg = session, cfg
        self.audit = AuditLog(clock, component="coordinator", session_id=session.session_id, sink=events)
        self.coordinator = Coordinator(session.session_id, cfg, clock, audit=self.audit,
                                       fusion_gate=lambda target, ids: session.fusion.set_sensor_filter(target, ids))
        self.source = HostSensorSource(session)
        self.agents: dict[str, SensorAgent] = {}
        self._seq = 0
        self.coordinator.subscribe(self._on_message)
        session.fusion.engine.subscribe(lambda r: self.coordinator.on_fusion_outcome(FusionOutcome.from_fusion_result(r)))

    def _on_message(self, msg) -> None:
        if msg.message_type == "TASK_REQUEST":
            for sid in sorted(self.session.registry.records):
                agent = self.agents.setdefault(sid, SensorAgent(sid, self.source, self.cfg, self.coordinator.clock))
                offer = agent.on_task_request(msg)
                if offer is not None:
                    self.coordinator.on_offer(offer)

    def publish_sensor_status(self) -> None:
        now = self.coordinator.clock.now_ms()
        for sid, rec in self.session.registry.records.items():
            snap = self.session.registry.snapshot(rec, now)
            self._seq += 1
            res = self.session.fusion.registry.resolve(sid)
            self.coordinator.on_sensor_status(SensorStatusMsg(
                session_id=self.session.session_id, sensor_id=sid, sequence_number=self._seq, observed_at_ms=now,
                state=snap["state"], reason=snap["state_reason"], calibration_ok=res.problem in (None, "DIMENSION_MISMATCH")))

    def tick(self) -> None:
        self.publish_sensor_status()
        self.coordinator.tick()
