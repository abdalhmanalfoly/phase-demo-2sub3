"""Fusion engine: calibrated multi-view triangulation of a target landmark with explicit status for every outcome.

Decision order per target (first match wins), evaluated on observation arrival and on a periodic tick:
  1. no usable calibration for the sensors that are observing      -> CALIBRATION_REQUIRED
  2. no fresh usable observation                                   -> INSUFFICIENT_VIEWS (inside hold) | TARGET_LOST
  3. exactly one fresh usable view                                 -> SINGLE_VIEW_FALLBACK (a viewing ray, never a point)
  4. >= 2 views but no common capture-time window                  -> SYNC_DEGRADED
  5. conflicting hand labels / high reprojection error             -> FUSION_OUTLIER
  6. numerically unreliable geometry                               -> GEOMETRY_DEGENERATE
  7. otherwise                                                     -> TRACKING_MULTI_VIEW (the only status with ``world_xyz_m``)
"""
from __future__ import annotations

import collections
import itertools
from typing import Callable

import numpy as np

from ..config import SpatialCfg
from . import geometry as G
from .calibration_store import CalibrationRegistry
from .filters import TrackSmoother
from .results import (CalibrationRef, FusionResult, FusionStatus, QualityInfo, SingleViewRay, SourceInfo)
from .synchronizer import ObsRef, ObservationSynchronizer, SyncResult

ACTIVE_WINDOW_MS = 2000.0      # a sensor counts as "observing" if its newest observation is this recent
_F = FusionStatus


def _vec(a) -> list[float]:
    return [float(x) for x in np.asarray(a, dtype=float).reshape(3)]


class FusionEngine:
    def __init__(self, session_id: str, cfg: SpatialCfg, registry: CalibrationRegistry, clock):
        self.session_id, self.cfg, self.registry, self.clock = session_id, cfg, registry, clock
        self.sync = ObservationSynchronizer(cfg.sync)
        self.targets = list(cfg.spatial.targets)
        sm, tr = cfg.smoothing, cfg.tracking
        self.smoothers = {t: TrackSmoother(sm.enabled, sm.min_cutoff_hz, sm.beta, sm.d_cutoff_hz, tr.hold_ms) for t in self.targets}
        self.seq = {t: 0 for t in self.targets}
        self.latest: dict[str, FusionResult] = {}
        self.history: collections.deque[FusionResult] = collections.deque(maxlen=cfg.spatial.recent_results)
        self.counters: collections.Counter = collections.Counter()
        self.subscribers: list[Callable[[FusionResult], None]] = []
        self._gen = registry.generation
        self._pending: dict[str, float | None] = {t: None for t in self.targets}
        self._last_key: dict[str, tuple] = {}
        self._hand: dict[str, str | None] = {t: None for t in self.targets}
        self._last_raw: dict[str, tuple[np.ndarray, float] | None] = {t: None for t in self.targets}
        # Phase 3 (Contract Net): sensors the coordinator has leased for a target. ABSENT key = unrestricted (Phase 2 behaviour,
        # unchanged); an EMPTY set = no sensor may feed the target (expired/ended lease controls nothing).
        self.sensor_filter: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------ public API
    def set_sensor_filter(self, target: str, sensor_ids) -> None:
        """Restrict ``target`` to ``sensor_ids`` (``None`` removes the restriction). Re-announces the status on the next evaluation."""
        if sensor_ids is None:
            self.sensor_filter.pop(target, None)
        else:
            self.sensor_filter[target] = frozenset(sensor_ids)
        self._last_key.pop(target, None)
        self._pending[target] = None

    def subscribe(self, fn: Callable[[FusionResult], None]) -> None:
        self.subscribers.append(fn)

    def reset(self, reason: str) -> None:
        for t in self.targets:
            self.smoothers[t].reset(reason)
            self._pending[t] = None
            self._hand[t] = None
            self._last_raw[t] = None
        self.sync.clear()

    def ingest(self, entry: dict, offset_ms: int | None) -> list[FusionResult]:
        """Feed one accepted Phase-1 observation entry (the dict produced by SensorRegistry.accept_observation)."""
        now = self.clock.now_ms()
        ok, why = self.sync.add(entry, offset_ms, now)
        if not ok:
            self.counters[f"sync_rejected:{why}"] += 1
            return []
        return self._run(now, "observation")

    def tick(self) -> list[FusionResult]:
        if not self.sync.buffers:
            return []
        return self._run(self.clock.now_ms(), "tick")

    # ------------------------------------------------------------------ evaluation
    def _run(self, now: float, trigger: str) -> list[FusionResult]:
        if self.registry.generation != self._gen:          # calibration changed / sensor re-bound / recalibrated
            for t in self.targets:                          # never smooth across a calibration change; raw pixel buffers stay valid
                self.smoothers[t].reset("calibration_changed")
                self._pending[t] = None
                self._hand[t] = None
                self._last_raw[t] = None
            self._last_key.clear()                          # re-announce the (possibly new) status
            self._gen = self.registry.generation
        out = []
        for target in self.targets:
            res = self._evaluate(target, now)
            if res is not None:
                self._emit(res)
                out.append(res)
        return out

    def _emit(self, res: FusionResult) -> None:
        self.latest[res.target_landmark] = res
        self.history.append(res)
        self.counters[f"status:{res.status.value}"] += 1
        for fn in list(self.subscribers):
            try:
                fn(res)
            except Exception:                              # a consumer bug must never break fusion
                self.counters["subscriber_errors"] += 1

    def _evaluate(self, target: str, now: float) -> FusionResult | None:
        sm = self.smoothers[target]
        active = self.sync.active_sensors(now, ACTIVE_WINDOW_MS)
        allowed = self.sensor_filter.get(target)
        if allowed is not None:
            active = [sid for sid in active if sid in allowed]
        newest = {sid: self.sync.buffers[sid][-1] for sid in active}
        res = {sid: self.registry.resolve(sid, newest[sid].image_width, newest[sid].image_height) for sid in active}
        calibrated = [sid for sid in active if res[sid].problem in (None, "DIMENSION_MISMATCH")]
        compat, incompat = self.registry.compatible_ids(calibrated)
        problems: dict[str, str] = {}
        for sid in active:
            if sid in incompat:
                problems[sid] = "INCOMPATIBLE_RIG"
            elif res[sid].problem is not None:
                problems[sid] = res[sid].problem
        usable_ids = [sid for sid in active if sid in compat and res[sid].problem is None]

        if active and not usable_ids:                       # 1. nothing observing has a usable calibration
            detail = "; ".join(f"{s}: {p}" for s, p in sorted(problems.items()))
            return self._publish_if_changed(target, now, _F.CALIBRATION_REQUIRED, "NO_CALIBRATED_SENSORS",
                                            f"no valid, compatible calibration for observing sensors ({detail})",
                                            problems=problems, newest=newest, res=res)
        sync = self.sync.match(target, usable_ids, now)

        if not sync.usable:                                 # 2. no fresh usable observation
            return self._lost(target, now, sync, problems, newest, res)
        if len(sync.usable) == 1:                           # 3. one view only
            return self._single_view(target, now, sync, problems, newest, res)
        if not sync.matched:                                # 4. asynchronous
            self._pending[target] = self._pending[target] if self._pending[target] is not None else now
            if now - self._pending[target] < self.cfg.sync.degrade_grace_ms:
                return None
            return self._publish_if_changed(
                target, now, _F.SYNC_DEGRADED, "SKEW_EXCEEDS_WINDOW",
                f"views are not simultaneous: best capture-time gap {sync.skew_ms:.0f} ms > window {self.cfg.sync.max_skew_ms:.0f} ms",
                skew_ms=sync.skew_ms, problems=problems, newest=newest, res=res, sync=sync)
        self._pending[target] = None
        if not sync.is_new:                                 # same set as already published
            return None
        return self._triangulate(target, now, sync, problems, newest, res)

    # ------------------------------------------------------------------ outcomes
    def _lost(self, target, now, sync: SyncResult, problems, newest, res):
        sm = self.smoothers[target]
        self._pending[target] = None
        held = sm.held(now)
        if held is not None:
            return self._publish_if_changed(target, now, _F.INSUFFICIENT_VIEWS, "NO_USABLE_OBSERVATIONS_HOLDING",
                                            "no fresh usable observation; last valid estimate is held for a bounded time",
                                            problems=problems, newest=newest, res=res, sync=sync)
        expired = sm.expire(now)
        why = "HOLD_EXPIRED" if expired else self._dominant_unusable(sync)
        return self._publish_if_changed(target, now, _F.TARGET_LOST, why,
                                        "target lost: no fresh usable observation and no valid estimate to hold",
                                        problems=problems, newest=newest, res=res, sync=sync)

    @staticmethod
    def _dominant_unusable(sync: SyncResult) -> str:
        if not sync.unusable:
            return "NO_ACTIVE_SENSORS"
        return collections.Counter(sync.unusable.values()).most_common(1)[0][0]

    def _single_view(self, target, now, sync: SyncResult, problems, newest, res):
        (sid, obs), = sync.usable.items()
        if obs.sequence_number <= self.sync.consumed.get((target, sid), -1):
            return None                                     # already reported this observation
        if self._pending[target] is None:
            self._pending[target] = now
        if now - self._pending[target] < self.cfg.sync.degrade_grace_ms:
            return None
        cam = res[sid].model
        x, y, _ = obs.landmarks[target]
        origin, direction = G.pixel_ray_world(cam, G.to_cv_pixels(np.array([x, y]), self.cfg.spatial.subtract_half_pixel))
        self.sync.mark_consumed(target, [obs])
        return self._make(target, now, _F.SINGLE_VIEW_FALLBACK, "ONE_VIEW_AVAILABLE",
                          "only one calibrated view sees the target: a viewing ray is available, NOT a 3D position",
                          ray=SingleViewRay(sensor_id=sid, origin_m=_vec(origin), direction=_vec(direction)),
                          problems=problems, newest=newest, res=res, sync=sync, members=[obs], used=[])

    def _triangulate(self, target, now, sync: SyncResult, problems, newest, res) -> FusionResult:
        tri_cfg, sm = self.cfg.triangulation, self.smoothers[target]
        members = list(sync.members)
        self.sync.mark_consumed(target, members)
        common = dict(problems=problems, newest=newest, res=res, sync=sync, members=members, skew_ms=sync.skew_ms)
        labels = {m.hand for m in members if m.hand != "unknown"}
        if tri_cfg.require_hand_agreement and len(labels) > 1:
            return self._make(target, now, _F.FUSION_OUTLIER, "HAND_LABEL_MISMATCH",
                              f"contributing sensors disagree on which hand this is: {sorted(labels)}", used=[], **common)
        warnings: list[str] = []
        sol = self._solve(target, members, res)
        if sol["fail"] is not None:
            status, reason, msg = sol["fail"]
            return self._make(target, now, status, reason, msg, used=[], **common)
        # reprojection consistency / outlier handling
        used = members
        errs = sol["errors"]
        if sol["rms"] > tri_cfg.max_reprojection_error_px or errs.max() > tri_cfg.max_view_reprojection_px:
            alt = self._drop_one(target, members, res) if len(members) >= 3 else None
            if alt is None:
                return self._make(target, now, _F.FUSION_OUTLIER, "REPROJECTION_ERROR_HIGH",
                                  f"views are geometrically inconsistent: RMS {sol['rms']:.1f} px (limit {tri_cfg.max_reprojection_error_px} px), "
                                  f"worst view {errs.max():.1f} px", used=members, sol=sol, **common)
            sol, used, dropped = alt
            warnings.append(f"OUTLIER_VIEW_DROPPED:{dropped.sensor_id}")
            common["members"] = members
        point = sol["point"]
        t_eff = float(np.mean([m.effective_ms for m in used]))
        key_hand = self._update_hand_identity(target, used)
        raw_prev = self._last_raw[target]
        if raw_prev is not None and t_eff > raw_prev[1]:
            speed = float(np.linalg.norm(point - raw_prev[0]) / ((t_eff - raw_prev[1]) / 1000.0))
            if speed > self.cfg.tracking.max_speed_mps:
                sm.reset(f"track_discontinuity:{speed:.1f}m/s")
                warnings.append("TRACK_DISCONTINUITY")
        self._last_raw[target] = (point, t_eff)
        smoothed = sm.update(point, t_eff, key=(target, key_hand))        # smoothing only AFTER geometric validation
        quality = self._quality(sol, sync.skew_ms)
        return self._make(target, now, _F.TRACKING_MULTI_VIEW, None, "valid calibrated multi-view triangulation",
                          world=smoothed, raw=point, sol=sol, used=used, quality=quality, warnings=warnings,
                          timestamp_ms=t_eff, **{k: v for k, v in common.items()})

    # ------------------------------------------------------------------ geometry glue
    def _solve(self, target, members: list[ObsRef], res) -> dict:
        tri = self.cfg.triangulation
        cams = [res[m.sensor_id].model for m in members]
        pts = G.to_cv_pixels(np.array([[m.landmarks[target][0], m.landmarks[target][1]] for m in members]),
                             self.cfg.spatial.subtract_half_pixel)
        r = G.triangulate_dlt(cams, pts, tri.min_homogeneous_w)
        if not r.ok:
            return {"fail": (_F.GEOMETRY_DEGENERATE, r.reason, f"triangulation failed: {r.detail}")}
        X = r.point
        if tri.refine_iterations > 0:
            Xr = G.refine_gauss_newton(cams, pts, X, tri.refine_iterations)
            if np.all(np.isfinite(Xr)):
                X = Xr
        ok, depths, why = G.check_depths(cams, X, tri.min_depth_m, tri.max_depth_m)
        if not ok:
            return {"fail": (_F.GEOMETRY_DEGENERATE, why, f"point rejected: {why} (depths {', '.join(f'{k}={v:.2f}' for k, v in depths.items())})")}
        angle = G.max_ray_angle_deg(cams, X)
        if angle < tri.min_ray_angle_deg:
            return {"fail": (_F.GEOMETRY_DEGENERATE, "BASELINE_TOO_SMALL",
                             f"viewing rays differ by only {angle:.2f} deg (< {tri.min_ray_angle_deg} deg): depth is unreliable")}
        errs = G.reprojection_errors(cams, X, pts)
        return {"fail": None, "point": X, "errors": errs, "rms": float(np.sqrt(np.mean(errs ** 2))), "angle": angle,
                "ids": [m.sensor_id for m in members]}

    def _drop_one(self, target, members, res):
        tri = self.cfg.triangulation
        best = None
        for i, drop in enumerate(members):
            rest = members[:i] + members[i + 1:]
            sol = self._solve(target, rest, res)
            if sol["fail"] is not None or sol["rms"] > tri.max_reprojection_error_px or sol["errors"].max() > tri.max_view_reprojection_px:
                continue
            cam = res[drop.sensor_id].model
            x, y, _ = drop.landmarks[target]
            err_drop = float(np.linalg.norm(G.project(cam, sol["point"])[0] - G.to_cv_pixels(np.array([x, y]), self.cfg.spatial.subtract_half_pixel)[0]))
            if err_drop > tri.max_view_reprojection_px and (best is None or sol["rms"] < best[0]["rms"]):
                best = (sol, rest, drop)
        return best

    def _quality(self, sol, skew_ms) -> QualityInfo:
        """score = product of: reprojection (1 - rms/limit), sync (1 - skew/window), geometry (min(1, angle/good_angle)).
        Unknown components are omitted (the hand model gives no per-landmark confidence, so none is invented)."""
        rep = max(0.0, 1.0 - sol["rms"] / self.cfg.triangulation.max_reprojection_error_px)
        syn = max(0.0, 1.0 - (skew_ms or 0.0) / self.cfg.sync.max_skew_ms)
        geo = min(1.0, sol["angle"] / self.cfg.quality.good_ray_angle_deg)
        return QualityInfo(score=float(rep * syn * geo), reprojection=float(rep), sync=float(syn), geometry=float(geo))

    def _update_hand_identity(self, target, used: list[ObsRef]) -> str | None:
        labels = {m.hand for m in used if m.hand != "unknown"}
        if len(labels) == 1:
            lab = next(iter(labels))
            if self._hand[target] is not None and lab != self._hand[target]:
                self.smoothers[target].reset(f"hand_identity_changed:{self._hand[target]}->{lab}")
            self._hand[target] = lab
        return self._hand[target]

    # ------------------------------------------------------------------ result construction
    def _publish_if_changed(self, target, now, status, reason, message, **kw):
        key = (status, None if status == _F.TARGET_LOST else reason)      # one TARGET_LOST per loss episode
        if self._last_key.get(target) == key and status != _F.SINGLE_VIEW_FALLBACK:
            return None
        return self._make(target, now, status, reason, message, used=[], **kw)

    def _make(self, target, now, status, reason, message, *, problems=None, newest=None, res=None, sync=None, members=None,
              used=None, world=None, raw=None, sol=None, ray=None, quality=None, skew_ms=None, warnings=None,
              timestamp_ms=None) -> FusionResult:
        sm = self.smoothers[target]
        used = used or []
        used_ids = {m.sensor_id for m in used}
        sources: list[SourceInfo] = []
        errs = {}
        if sol is not None:
            errs = dict(zip(sol.get("ids", []), (float(e) for e in sol["errors"])))
        seen = set()
        for m in list(members or []) + [o for o in (sync.usable.values() if sync else []) if o not in (members or [])]:
            seen.add(m.sensor_id)
            sources.append(self._source(m, now, res, used_ids, errs, None))
        for sid, o in (newest or {}).items():
            if sid not in seen:
                note = (problems or {}).get(sid) or (sync.unusable.get(sid) if sync else None)
                sources.append(self._source(o, now, res, used_ids, errs, note))
        for s in sources:
            if not s.used and s.note is None and s.sensor_id in (problems or {}):
                s.note = problems[s.sensor_id]
        warnings = list(warnings or [])
        for sid, code in sorted((problems or {}).items()):                 # nothing is excluded silently
            if status != _F.CALIBRATION_REQUIRED:
                warnings.append(f"CALIBRATION_PROBLEM:{sid}:{code}")
        held = None if status == _F.TRACKING_MULTI_VIEW else sm.held(now)
        calib = None
        arts = {s.sensor_id: res[s.sensor_id].artifact for s in sources if res and res[s.sensor_id].artifact is not None}
        usable_arts = {sid: a for sid, a in arts.items() if sid in used_ids or status != _F.TRACKING_MULTI_VIEW}
        if usable_arts:
            a0 = next(iter(usable_arts.values()))
            calib = CalibrationRef(rig_id=a0.rig_id, rig_calibration_id=a0.rig_calibration_id, world_frame_id=a0.world_frame_id,
                                   calibration_ids={sid: a.calibration_id for sid, a in usable_arts.items()})
        self.seq[target] += 1
        valid = status == _F.TRACKING_MULTI_VIEW
        self._last_key[target] = (status, None if status == _F.TARGET_LOST else reason)
        agg = None if sol is None else float(sol["rms"])
        r = FusionResult(
            session_id=self.session_id, target_landmark=target, fusion_sequence=self.seq[target] - 1, status=status, valid=valid,
            reason=reason, message=message, timestamp_ms=timestamp_ms, published_at_ms=int(now),
            world_frame_id=calib.world_frame_id if calib else None,
            world_xyz_m=_vec(world) if valid and world is not None else None,
            raw_world_xyz_m=_vec(raw) if valid and raw is not None else None,
            held_world_xyz_m=_vec(held[0]) if held else None, held_age_ms=float(held[1]) if held else None,
            single_view_ray=ray, source_sensor_ids=sorted(used_ids), sources=sources,
            skew_ms=None if skew_ms is None else float(skew_ms),
            aggregate_reprojection_error_px=agg, ray_angle_deg=None if sol is None else float(sol["angle"]),
            quality=quality or QualityInfo(score=None, reprojection=None, sync=None, geometry=None),
            calibration=calib, smoothing_applied=bool(valid and self.cfg.smoothing.enabled), warnings=warnings)
        return r

    def _source(self, o: ObsRef, now, res, used_ids, errs, note) -> SourceInfo:
        r = res.get(o.sensor_id) if res else None
        return SourceInfo(
            sensor_id=o.sensor_id, camera_id=r.camera_id if r else None,
            calibration_id=r.artifact.calibration_id if r is not None and r.artifact is not None else None,
            sequence_number=o.sequence_number, frame_id=o.frame_id, captured_at_ms=o.captured_at_ms,
            received_at_ms=o.received_at_ms, effective_time_ms=o.effective_ms, age_ms=float(now - o.effective_ms),
            transport_delay_ms=float(o.received_at_ms - o.effective_ms), reprojection_error_px=errs.get(o.sensor_id),
            used=o.sensor_id in used_ids, note=note)
