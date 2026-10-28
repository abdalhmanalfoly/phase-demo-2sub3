"""Sensor-side SELF-ASSESSMENT (Contract Net stage 2/3).

A ``SensorAgent`` turns a ``SensorSnapshot`` (what the host knows about one sensor right now) into an honest
``ObservationOffer``. It reports reality even when poor (stale, occluded, uncalibrated): rejecting is the coordinator's job
and every rejection needs a recorded reason. It never fabricates a good value for missing data (see scoring.py).
``HostSensorSource`` (host_adapter.py) builds snapshots from the real registry/synchronizer/calibration registry.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from ..spatial import geometry as G
from . import scoring
from .config import CoordinationCfg
from .messages import ObservationOffer, TaskRequest


@dataclass(frozen=True)
class ObservationView:
    effective_ms: float
    image_width: int
    image_height: int
    status: str                                   # tracked | no_hand
    hand: str
    landmarks: Mapping[str, tuple]                # id -> (x_px, y_px, confidence|None)   (Phase 1 pixels, edge origin)
    flags: tuple[str, ...] = ()
    inference_ms: float | None = None


@dataclass(frozen=True)
class CalibrationView:
    eligible: bool
    problem: str | None = None
    camera_model: "G.CameraModel | None" = None
    rig_key: str | None = None                    # "<rig_id>|<rig_calibration_id>|<world_frame_id>|<unit>"


@dataclass(frozen=True)
class SensorSnapshot:
    sensor_id: str
    state: str                                    # online | degraded | offline | disconnected
    capabilities: tuple[str, ...]
    update_rate_hz: float | None
    clock_offset_known: bool
    delay_jitter_ms: float | None
    transport_delay_ms: float | None
    observation: ObservationView | None
    calibration: CalibrationView


class SensorSource(Protocol):
    def snapshot(self, sensor_id: str, now_ms: int) -> SensorSnapshot | None: ...


class SensorAgent:
    def __init__(self, sensor_id: str, source: SensorSource, cfg: CoordinationCfg, clock):
        self.sensor_id, self.source, self.cfg, self.clock = sensor_id, source, cfg, clock
        self._seq: collections.Counter = collections.Counter()

    def on_task_request(self, task: TaskRequest) -> ObservationOffer | None:
        """Return this sensor's offer, or None when the sensor is unknown to the source (disconnected: it stays silent)."""
        now = self.clock.now_ms()
        snap = self.source.snapshot(self.sensor_id, now)
        if snap is None or snap.state == "disconnected":
            return None
        sc, notes = self.cfg.scoring, []
        obs = snap.observation
        lm = task.target_landmark
        age = None if obs is None else now - obs.effective_ms
        if obs is None:
            notes.append("NO_OBSERVATION")
        elif obs.status != "tracked" or lm not in obs.landmarks:
            notes.append("TARGET_NOT_VISIBLE")
        conf_raw = None if obs is None or lm not in obs.landmarks else obs.landmarks[lm][2]
        conf, src = scoring.detector_confidence_value(conf_raw, sc)
        if src == "imputed":
            notes.append("DETECTOR_CONFIDENCE_IMPUTED")
        vis = 0.0 if obs is None else scoring.visibility_score(obs.status, obs.landmarks, lm, obs.image_width, obs.image_height, obs.flags, sc)
        # ---- calibration + geometry (a viewing RAY only: never a 3D point)
        cal, ray, rig_key, cal_ok = snap.calibration, None, snap.calibration.rig_key, snap.calibration.eligible
        if not cal_ok:
            notes.append(f"CALIBRATION:{cal.problem or 'NOT_ELIGIBLE'}")
        elif obs is not None and obs.status == "tracked" and lm in obs.landmarks and cal.camera_model is not None:
            cam = cal.camera_model
            if (obs.image_width, obs.image_height) != (cam.width, cam.height):
                cal_ok = False
                notes.append("CALIBRATION:DIMENSION_MISMATCH")
            else:
                px = G.to_cv_pixels(np.array([obs.landmarks[lm][0], obs.landmarks[lm][1]]), self.cfg.subtract_half_pixel)
                o, d = G.pixel_ray_world(cam, px)
                ray = (tuple(float(x) for x in o), tuple(float(x) for x in d))
        latency = (snap.transport_delay_ms if snap.transport_delay_ms is not None else sc.assumed_latency_ms)
        latency += 0.0 if obs is None or obs.inference_ms is None else obs.inference_ms
        if snap.transport_delay_ms is None:
            notes.append("LATENCY_ASSUMED")
        self._seq[task.task_id] += 1
        return ObservationOffer(
            session_id=task.session_id, task_id=task.task_id, sensor_id=self.sensor_id, offer_sequence=self._seq[task.task_id] - 1,
            target_landmark=lm, detector_confidence=conf, detector_confidence_source=src, visibility_score=vis,
            freshness_score=scoring.freshness_score(age, sc), sync_score=scoring.sync_score(snap.clock_offset_known, snap.delay_jitter_ms, sc),
            health_score=scoring.health_score(snap.state, snap.update_rate_hz, sc), estimated_latency_ms=max(0.0, float(latency)),
            calibration_eligible=bool(cal_ok), capabilities=tuple(snap.capabilities),
            observation_effective_ms=None if obs is None else float(obs.effective_ms),
            observation_age_ms=None if age is None else float(age), submitted_at_ms=now, view_ray=ray, rig_key=rig_key,
            hand="unknown" if obs is None else obs.hand, notes=tuple(notes))
