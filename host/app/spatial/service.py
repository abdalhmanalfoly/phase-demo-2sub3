"""Per-session glue between the Phase 1 host and the spatial subsystem (calibration registry + fusion engine)."""
from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..services.observation_recorder import ObservationRecorder
from .calibration_store import CalibrationRegistry, CalibrationStore
from .fusion import FusionEngine
from .results import FusionResult


class FusionService:
    def __init__(self, session_id: str, settings: Settings, clock, events):
        cfg = settings.fusion
        self.session_id, self.cfg, self.events = session_id, cfg, events
        store = CalibrationStore(settings.calibration_root, cfg.spatial.rig_id)
        self.registry = CalibrationRegistry(store, allow_synthetic=cfg.spatial.allow_synthetic_calibration)
        self.engine = FusionEngine(session_id, cfg, self.registry, clock)
        # logging.save_landmark_events is the MASTER switch for movement data: derived 3D tracks are only recorded when it is on too.
        self.recorder = ObservationRecorder(settings.log_dir, session_id,
                                            cfg.spatial.save_fusion_results and settings.logging.save_landmark_events, "fusion.jsonl")
        self._last_status: dict[str, str] = {}
        for name, err in self.registry.load_errors.items():
            events.log("calibration_load_error", session_id=session_id, severity="error", reason=err, camera=name)

    # ---- called by the WebSocket handler / sweeper -------------------------------------------------
    def on_observation(self, entry: dict, sensor_record) -> list[FusionResult]:
        return self._after(self.engine.ingest(entry, sensor_record.offset.offset_ms))

    def tick(self) -> list[FusionResult]:
        return self._after(self.engine.tick())

    def _after(self, results: list[FusionResult]) -> list[FusionResult]:
        for r in results:
            self.recorder.write(r.model_dump(mode="json"))
            prev = self._last_status.get(r.target_landmark)
            if prev != r.status.value:                         # log transitions only (results arrive at ~15 Hz)
                self._last_status[r.target_landmark] = r.status.value
                self.events.log("fusion_status_changed", session_id=self.session_id, reason=r.reason,
                                severity="info" if r.valid else "warning", target=r.target_landmark,
                                from_status=prev, to_status=r.status.value, message=r.message)
        return results

    def set_sensor_filter(self, target: str, sensor_ids) -> None:
        self.engine.set_sensor_filter(target, sensor_ids)
        self.events.log("fusion_sensor_filter", session_id=self.session_id, reason="lease", target=target,
                        sensors=None if sensor_ids is None else sorted(sensor_ids))

    # ---- operator actions ------------------------------------------------------------------------------
    def bind(self, sensor_id: str, camera_id: str | None) -> None:
        if camera_id is None:
            self.registry.unbind(sensor_id)
        else:
            self.registry.bind(sensor_id, camera_id)
        self.events.log("calibration_binding", session_id=self.session_id, sensor_id=sensor_id, reason=camera_id or "unbound")

    def invalidate(self, camera_id: str, reason: str) -> None:
        self.registry.invalidate(camera_id, reason)
        self.events.log("calibration_invalidated", session_id=self.session_id, reason=reason, severity="warning", camera=camera_id)

    def reload(self) -> None:
        self.registry.reload()
        self.events.log("calibration_reloaded", session_id=self.session_id, reason=f"{len(self.registry.artifacts)} cameras")

    # ---- reporting -------------------------------------------------------------------------------------
    def summary(self, sensor_ids: list[str]) -> dict:
        return {"latest": {t: r.model_dump(mode="json") for t, r in self.engine.latest.items()},
                "counters": dict(self.engine.counters)}

    def details(self, sensor_ids: list[str], recent: int = 20) -> dict:
        return {"latest": {t: r.model_dump(mode="json") for t, r in self.engine.latest.items()},
                "recent": [r.model_dump(mode="json") for r in list(self.engine.history)[-recent:]],
                "counters": dict(self.engine.counters), "sync": self.engine.sync.diagnostics(),
                "smoother_resets": {t: s.resets[-5:] for t, s in self.engine.smoothers.items()}}

    def calibration_status(self, sensor_ids: list[str]) -> dict:
        return self.registry.status(sorted(set(sensor_ids) | set(self.registry.bindings)))

    def close(self) -> None:
        self.recorder.close()
