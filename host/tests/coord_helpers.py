"""Shared helpers for the Phase 3 tests. No pydantic/fastapi: these tests run with only the standard library + NumPy + OpenCV."""
from __future__ import annotations

from host.app.coordination import simulation as S
from host.app.coordination.audit import AuditLog
from host.app.coordination.config import CoordinationCfg
from host.app.coordination.coordinator import Coordinator
from host.app.coordination.fusion_port import FusionOutcome
from host.app.coordination.messages import SensorStatusMsg

SESSION = S.SESSION


def cfg(**over) -> CoordinationCfg:
    c = CoordinationCfg()
    for k, v in over.items():
        sec, name = k.split(".")
        setattr(getattr(c, sec), name, v)
    return c.validate()


def outcome(seq, now, *, valid=True, status="TRACKING_MULTI_VIEW", rms=0.5, sources=("cam_a", "cam_c"), reason=None, target="wrist"):
    return FusionOutcome(session_id=SESSION, target_landmark=target, fusion_sequence=seq, status=status, valid=valid, reason=reason,
                         rms_px=rms if valid else None, ray_angle_deg=40.0, skew_ms=5.0, quality_score=0.9 if valid else None,
                         source_sensor_ids=tuple(sources) if valid else (), published_at_ms=now)


class Bench:
    """A harness plus helpers to feed offers manually (no automatic bus) so message-ordering cases can be forced."""

    def __init__(self, **kw):
        self.h = S._tune(S.SimHarness(S._cfg(), kw.pop("sensors", ("cam_a", "cam_b", "cam_c", "cam_d")), **kw))
        self.c = self.h.coord

    def offers_for(self, task) -> dict:
        return {sid: ag.on_task_request(task) for sid, ag in sorted(self.h.agents.items())}
