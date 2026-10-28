"""Deterministic simulation harness: negotiate, lease, fuse and adapt WITHOUT physical cameras.

* Real code under test: ``SensorAgent`` (self-assessment), ``Coordinator``, ``SetSelector``, scoring, messages.
* Simulated: sensors (cameras with real ``geometry.CameraModel`` projection, visibility, delay, staleness, disconnects),
  a message bus with delivery delays, and a clock.
* ``OfflineFusion`` is a STAND-IN for ``spatial.fusion.FusionEngine``: it applies the SAME ``geometry`` functions (DLT, depth,
  ray angle, reprojection) and the SAME status vocabulary / thresholds, but none of the engine's smoothing or buffering. Use
  ``EngineFusionBackend`` (needs pydantic; see tests/test_coord_engine_integration.py) to run the identical scenarios against the real engine.
No random numbers are used: every run is bit-for-bit reproducible.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..spatial import geometry as G
from .audit import AuditLog
from .config import CoordinationCfg
from .coordinator import CoordState, Coordinator
from .fusion_port import FusionOutcome
from .messages import ObservationOffer, SensorStatusMsg, TaskRequest
from .sensor_agent import CalibrationView, ObservationView, SensorAgent, SensorSnapshot

T0 = 1_791_735_600_000
SESSION = "sess_sim00001"
RIG_KEY = "default_rig|rig-sim-1|board_frame|m"
TARGET = np.array([0.25, 0.05, 0.0])


class SimClock:
    def __init__(self, t0: int = T0):
        self.t = t0

    def now_ms(self) -> int:
        return self.t

    def advance(self, ms: int) -> None:
        self.t += ms


def look_at(center, target, up=(0.0, -1.0, 0.0)):
    """World->camera (R, t); identical maths to ``spatial.synthetic.look_at`` (that module needs pydantic, this one must not)."""
    c, tg = np.asarray(center, float), np.asarray(target, float)
    z = tg - c
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float) * -1.0, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ c


def make_camera(cid: str, center, look=(0.2, 0.05, 0.0), size=(640, 480), f=600.0) -> "G.CameraModel":
    K = np.array([[f, 0, size[0] / 2 - 0.5 + 3.0], [0, f, size[1] / 2 - 0.5 - 2.0], [0, 0, 1.0]])
    R, t = look_at(center, look)
    return G.CameraModel(cid, K, np.zeros(5), R, t, size[0], size[1])


DEFAULT_RIG = {
    "cam_a": (-0.45, -0.10, -1.10),     # A and B nearly co-located: poor mutual geometry
    "cam_b": (-0.35, -0.10, -1.10),
    "cam_c": (0.85, -0.10, -1.10),      # wide baseline to A/B
    "cam_d": (0.20, -0.90, -1.00),      # overhead
}


@dataclass
class SimSensor:
    sensor_id: str
    camera: "G.CameraModel"
    state: str = "online"                 # what the registry would report
    visible: bool = True                  # occlusion switch
    detector_confidence: float | None = 0.9
    capture_lag_ms: float = 20.0          # age of the newest observation (staleness / delay)
    transport_delay_ms: float = 15.0
    delay_jitter_ms: float = 5.0
    offer_delay_ms: int = 5               # time for the OFFER to reach the coordinator
    reports_status: bool = True           # False = goes silent (no SENSOR_STATUS)
    calibration_ok: bool = True
    pixel_bias_px: tuple = (0.0, 0.0)     # injected landmark error (drift, mis-detection) -> reprojection error
    update_rate_hz: float = 15.0
    rig_key: str = RIG_KEY
    hand: str = "right"


class SimSource:
    def __init__(self, world: "SimWorld"):
        self.world = world

    def snapshot(self, sensor_id: str, now_ms: int) -> SensorSnapshot | None:
        s = self.world.sensors.get(sensor_id)
        if s is None or s.state == "disconnected":
            return None
        obs = self.world.observation(s, now_ms)
        cal = CalibrationView(eligible=s.calibration_ok, problem=None if s.calibration_ok else "INVALIDATED",
                              camera_model=s.camera if s.calibration_ok else None, rig_key=s.rig_key)
        return SensorSnapshot(sensor_id=sensor_id, state=s.state, capabilities=("wrist", "thumb_tip", "index_tip"),
                              update_rate_hz=s.update_rate_hz, clock_offset_known=True, delay_jitter_ms=s.delay_jitter_ms,
                              transport_delay_ms=s.transport_delay_ms, observation=obs, calibration=cal)


class SimWorld:
    def __init__(self, clock: SimClock, target=TARGET):
        self.clock, self.target = clock, np.asarray(target, float)
        self.sensors: dict[str, SimSensor] = {}

    def add(self, s: SimSensor) -> SimSensor:
        self.sensors[s.sensor_id] = s
        return s

    def observation(self, s: SimSensor, now_ms: int) -> ObservationView | None:
        eff = now_ms - s.capture_lag_ms
        if not s.visible:
            return ObservationView(eff, s.camera.width, s.camera.height, "no_hand", "unknown", {}, (), 10.0)
        uv = G.project(s.camera, self.target)[0] + 0.5 + np.asarray(s.pixel_bias_px)
        conf = s.detector_confidence
        return ObservationView(eff, s.camera.width, s.camera.height, "tracked", s.hand, {"wrist": (float(uv[0]), float(uv[1]), conf)}, (), 10.0)


class OfflineFusion:
    """STAND-IN for FusionEngine (see module docstring). Produces ``FusionOutcome`` with Phase 2 status names and limits."""

    def __init__(self, world: SimWorld, clock: SimClock, allowed: dict, *, max_rms_px=6.0, max_view_px=12.0, min_angle=2.0, max_skew=50.0, max_age=250.0):
        self.world, self.clock, self.allowed = world, clock, allowed
        self.lim = (max_rms_px, max_view_px, min_angle, max_skew, max_age)
        self.seq = 0

    def evaluate(self, target: str = "wrist") -> FusionOutcome:
        now = self.clock.now_ms()
        max_rms, max_view, min_angle, max_skew, max_age = self.lim
        allowed = self.allowed.get(target)
        views = []
        for sid, s in sorted(self.world.sensors.items()):
            if allowed is not None and sid not in allowed:
                continue
            if s.state in ("disconnected", "offline") or not s.calibration_ok:
                continue
            o = self.world.observation(s, now)
            if o is None or o.status != "tracked" or now - o.effective_ms > max_age:
                continue
            views.append((sid, s.camera, o))
        self.seq += 1
        base = dict(session_id=SESSION, target_landmark=target, fusion_sequence=self.seq - 1, published_at_ms=now)

        def out(status, reason=None, valid=False, **kw):
            return FusionOutcome(status=status, valid=valid, reason=reason, rms_px=kw.get("rms"), ray_angle_deg=kw.get("angle"),
                                 skew_ms=kw.get("skew"), quality_score=kw.get("q"), source_sensor_ids=tuple(kw.get("used", ())),
                                 sensor_notes=kw.get("notes", {}), sensor_reprojection_px=kw.get("errs", {}), **base)
        if len(views) == 0:
            return out("TARGET_LOST", "NO_ACTIVE_SENSORS" if allowed is not None and not allowed else "NO_USABLE_OBSERVATIONS")
        if len(views) == 1:
            return out("SINGLE_VIEW_FALLBACK", "ONE_VIEW_AVAILABLE")
        times = [v[2].effective_ms for v in views]
        if max(times) - min(times) > max_skew:
            return out("SYNC_DEGRADED", "SKEW_EXCEEDS_WINDOW", skew=max(times) - min(times))
        cams = [v[1] for v in views]
        pts = G.to_cv_pixels(np.array([[v[2].landmarks["wrist"][0], v[2].landmarks["wrist"][1]] for v in views]), True)
        r = G.triangulate_dlt(cams, pts)
        if not r.ok:
            return out("GEOMETRY_DEGENERATE", r.reason)
        X = G.refine_gauss_newton(cams, pts, r.point, 5)
        ok, _, why = G.check_depths(cams, X, 0.05, 20.0)
        if not ok:
            return out("GEOMETRY_DEGENERATE", why)
        angle = G.max_ray_angle_deg(cams, X)
        if angle < min_angle:
            return out("GEOMETRY_DEGENERATE", "BASELINE_TOO_SMALL", angle=angle)
        errs = G.reprojection_errors(cams, X, pts)
        rms = float(np.sqrt(np.mean(errs ** 2)))
        e = {v[0]: float(x) for v, x in zip(views, errs)}
        used = [v[0] for v in views]
        if rms > max_rms or errs.max() > max_view:
            return out("FUSION_OUTLIER", "REPROJECTION_ERROR_HIGH", rms=rms, angle=angle, used=(), errs=e)
        return out("TRACKING_MULTI_VIEW", None, True, rms=rms, angle=angle, skew=max(times) - min(times), q=max(0.0, 1 - rms / max_rms), used=used, errs=e)


class SimHarness:
    """Wires clock, sensors, sensor agents, bus, coordinator and a fusion backend; ``run(ms)`` advances simulated time."""

    def __init__(self, cfg: CoordinationCfg | None = None, sensor_ids=("cam_a", "cam_b", "cam_c", "cam_d"), *, auto_renew=True,
                 fusion_period_ms=70, status_period_ms=250, step_ms=10, jsonl_path=None, fusion_factory=None, feed_fusion=True):
        self.cfg = (cfg or CoordinationCfg()).validate()
        self.clock = SimClock()
        self.world = SimWorld(self.clock)
        self.allowed: dict = {}
        self.audit = AuditLog(self.clock, component="coordinator", session_id=SESSION, jsonl_path=jsonl_path)
        self.coord = Coordinator(SESSION, self.cfg, self.clock, fusion_gate=lambda t, ids: self.allowed.__setitem__(t, ids),
                                 audit=self.audit, perf=self._fake_perf, auto_renew=auto_renew)
        self._perf = 0.0
        self.fusion = (fusion_factory or OfflineFusion)(self.world, self.clock, self.allowed) if fusion_factory is None else fusion_factory(self)
        self.agents: dict[str, SensorAgent] = {}
        self.source = SimSource(self.world)
        self.pending: list[tuple[int, int, ObservationOffer]] = []
        self._n = 0
        self.fusion_period_ms, self.status_period_ms, self.step_ms = fusion_period_ms, status_period_ms, step_ms
        self.feed_fusion = feed_fusion
        self._status_seq: collections.Counter = collections.Counter()
        self.fusion_log: list[FusionOutcome] = []
        self.timeline: list[str] = []
        self.coord.subscribe(self._on_message)
        for sid in sensor_ids:
            self.add_sensor(sid)
        self.status_all()

    def _fake_perf(self) -> float:                       # deterministic "compute time"
        self._perf += 0.0005
        return self._perf

    def add_sensor(self, sid: str, **over) -> SimSensor:
        cam = make_camera(sid, DEFAULT_RIG[sid]) if sid in DEFAULT_RIG else over.pop("camera")
        s = self.world.add(SimSensor(sid, cam, **over))
        self.agents[sid] = SensorAgent(sid, self.source, self.cfg, self.clock)
        return s

    # ---- bus
    def _on_message(self, msg) -> None:
        if isinstance(msg, TaskRequest):
            for sid, agent in sorted(self.agents.items()):
                s = self.world.sensors[sid]
                if s.state == "disconnected":
                    continue
                offer = agent.on_task_request(msg)
                if offer is not None:
                    self._n += 1
                    self.pending.append((self.clock.now_ms() + s.offer_delay_ms, self._n, offer))

    def status_all(self) -> None:
        for sid, s in sorted(self.world.sensors.items()):
            if not s.reports_status:
                continue
            self._status_seq[sid] += 1
            self.coord.on_sensor_status(SensorStatusMsg(session_id=SESSION, sensor_id=sid, sequence_number=self._status_seq[sid],
                                                        observed_at_ms=self.clock.now_ms(), state=s.state, calibration_ok=s.calibration_ok))

    def disconnect(self, sid: str) -> None:
        self.world.sensors[sid].state = "disconnected"
        self.status_all()

    def go_silent(self, sid: str) -> None:
        self.world.sensors[sid].reports_status = False

    # ---- time
    def run(self, ms: int) -> None:
        end = self.clock.now_ms() + ms
        while self.clock.now_ms() < end:
            self.clock.advance(self.step_ms)
            t = self.clock.now_ms() - T0
            due = sorted(p for p in self.pending if p[0] <= self.clock.now_ms())
            self.pending = [p for p in self.pending if p[0] > self.clock.now_ms()]
            for _, _, offer in due:
                self.coord.on_offer(offer)
            if t % self.status_period_ms == 0:
                self.status_all()
            if self.feed_fusion and t % self.fusion_period_ms == 0 and self.coord.state != CoordState.STOPPED:
                o = self.fusion.evaluate("wrist")
                self.fusion_log.append(o)
                self.coord.on_fusion_outcome(o)
            self.coord.tick()

    # ---- inspection
    def events(self, event_type: str) -> list[dict]:
        return self.audit.of_type(event_type)

    def lease_sensors(self) -> tuple[str, ...] | None:
        return None if self.coord.lease is None else tuple(sorted(self.coord.lease.sensor_ids))

    def awards(self) -> list[tuple[str, ...]]:
        return [tuple(e["sensors"]) for e in self.events("TASK_AWARD")]


# ---------------------------------------------------------------------------------------------------------- scenarios
def _cfg(**over) -> CoordinationCfg:
    cfg = CoordinationCfg()
    cfg.task.lease_ms = 4000.0          # long leases keep scenarios free of unrelated renewals; renewal has its own test/scenario
    for k, v in over.items():
        sec, name = k.split(".")
        setattr(getattr(cfg, sec), name, v)
    return cfg.validate()


def _tune(h: SimHarness) -> SimHarness:
    """A = best individual sensor, B second, C and D clearly worse individually (whichever of them exist)."""
    for sid, conf in (("cam_a", 0.95), ("cam_b", 0.90), ("cam_c", 0.70), ("cam_d", 0.60)):
        if sid in h.world.sensors:
            h.world.sensors[sid].detector_confidence = conf
    return h


def scenario_geometry_beats_individual_score() -> SimHarness:
    """A has the best individual score and B the second best, but A+B are nearly co-located: a wider set must win."""
    h = _tune(SimHarness(_cfg(), ("cam_a", "cam_b", "cam_c", "cam_d")))
    h.coord.request_observation()
    h.run(400)
    return h


def scenario_only_poor_geometry() -> SimHarness:
    """Only A and B exist: no trusted triangulation is possible and the coordinator must say so."""
    h = _tune(SimHarness(_cfg(), ("cam_a", "cam_b")))
    h.coord.request_observation()
    h.run(400)
    return h


def scenario_occlusion() -> SimHarness:
    h = _tune(SimHarness(_cfg(), ("cam_a", "cam_b", "cam_c", "cam_d")))
    h.coord.request_observation()
    h.run(600)
    h.world.sensors[h.lease_sensors()[0]].visible = False       # occlude a member of the current set
    h.run(1200)
    return h


def scenario_delay_and_stale() -> SimHarness:
    h = _tune(SimHarness(_cfg(), ("cam_a", "cam_b", "cam_c", "cam_d")))
    h.world.sensors["cam_a"].capture_lag_ms = 600.0             # stale: its newest observation is 600 ms old
    h.world.sensors["cam_c"].offer_delay_ms = 900               # its offer arrives after the 400 ms deadline
    h.coord.request_observation()
    h.run(1200)
    return h


def scenario_disconnect(silent: bool = False) -> SimHarness:
    h = _tune(SimHarness(_cfg(), ("cam_a", "cam_b", "cam_c", "cam_d")))
    h.coord.request_observation()
    h.run(600)
    victim = h.lease_sensors()[-1]
    h.go_silent(victim) if silent else h.disconnect(victim)
    h.victim = victim
    h.run(3000)
    return h


def scenario_fusion_failure(all_bad: bool) -> SimHarness:
    """Landmark errors that offers cannot reveal (drift / mis-detection) inflate the reprojection error.

    one camera biased: the failed set is banned and a set that avoids the bad camera is found (recoverable).
    all cameras biased in different directions: every set fails -> bounded re-selection -> DEGRADED (never a fake trusted result)."""
    h = _tune(SimHarness(_cfg(), ("cam_a", "cam_b", "cam_c", "cam_d")))
    h.coord.request_observation()
    h.run(400)
    # VERTICAL shifts (out of the epipolar plane of this rig) with distinct values: every pair of views becomes inconsistent.
    # (A shift along the epipolar direction would give ~0 reprojection error and only a wrong position - see docs/coordination.md limitations.)
    bias = {"cam_a": (0.0, 10.0), "cam_b": (0.0, -10.0), "cam_c": (0.0, 30.0), "cam_d": (0.0, -30.0)}
    bad = list(bias) if all_bad else [h.lease_sensors()[-1]]
    for sid in bad:
        h.world.sensors[sid].pixel_bias_px = bias[sid]
    h.bad = bad
    h.run(4000)
    return h


def scenario_renewal() -> SimHarness:
    cfg = _cfg()
    cfg.task.lease_ms = 1000.0
    cfg.task.renew_lead_ms = 300.0
    h = _tune(SimHarness(cfg.validate(), ("cam_a", "cam_b", "cam_c", "cam_d")))
    h.coord.request_observation()
    h.run(2600)
    return h


SCENARIOS: dict[str, Callable[[], SimHarness]] = {
    "1_geometry_beats_individual_score": scenario_geometry_beats_individual_score,
    "1b_only_poor_geometry": scenario_only_poor_geometry,
    "2_occlusion": scenario_occlusion,
    "3_delay_and_stale": scenario_delay_and_stale,
    "4_disconnect": scenario_disconnect,
    "4b_silent_sensor": lambda: scenario_disconnect(silent=True),
    "5a_fusion_failure_recoverable": lambda: scenario_fusion_failure(False),
    "5b_fusion_failure_unrecoverable": lambda: scenario_fusion_failure(True),
    "6_lease_renewal": scenario_renewal,
}
