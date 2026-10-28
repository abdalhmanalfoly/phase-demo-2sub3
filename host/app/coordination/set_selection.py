"""SET-SELECTION POLICY (Contract Net stage 5). Separate from individual offer scoring (scoring.py).

Evaluates every candidate group of 2..max_set_size eligible sensors and either picks the best FEASIBLE one or explains why
none is. The viewing rays carried in the offers are used to judge geometry; no 3D point is computed here (triangulation
stays in the Phase 2 fusion service).

Hard rules (a violation makes a set infeasible, with a machine-readable code):
  TOO_FEW_VIEWS            fewer than min_views sensors
  INCOMPATIBLE_RIG         sensors calibrated in different rigs / world frames
  SKEW_EXCEEDS_WINDOW      capture times differ by more than max_skew_ms
  INSUFFICIENT_VIEW_ANGLE  largest pairwise ray angle below min_view_angle_deg (depth ill-conditioned)
  RAYS_INCONSISTENT        rays of the "same" landmark miss each other by more than max_ray_gap_m (or diverge behind cameras)
  HAND_LABEL_MISMATCH      sensors disagree on which hand they see
  LATENCY_BUDGET           a member's estimated latency exceeds max_latency_ms
  DEADLINE_VIOLATION       a member cannot deliver before the task deadline
  RECENTLY_FAILED_SET      this exact set failed fusion within failed_set_ban_ms
Score of a feasible set: w_quality*mean(offer quality) + w_geometry*min(1, angle/good_angle) + w_sync*(1 - skew/max_skew)
                         - size_penalty*(n - min_views).  Ties: higher score, then fewer sensors, then sensor ids.
"""
from __future__ import annotations

import collections
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from . import scoring
from .config import CoordinationCfg
from .messages import ObservationOffer, TaskRequest

GEOMETRY_CODES = ("INSUFFICIENT_VIEW_ANGLE", "RAYS_INCONSISTENT")


@dataclass(frozen=True)
class CandidateEvaluation:
    sensor_ids: tuple[str, ...]
    feasible: bool
    rejection_codes: tuple[str, ...]
    score: float | None
    components: dict

    def label(self) -> str:
        return "+".join(self.sensor_ids)


@dataclass
class SetSelection:
    chosen: CandidateEvaluation | None
    evaluated: list[CandidateEvaluation]
    outcome: str                         # AWARD | NO_SET_INSUFFICIENT_SENSORS | NO_SET_INSUFFICIENT_GEOMETRY | NO_COMPATIBLE_SET
    reason: str
    individual_quality: dict[str, float]
    best_individual: str | None

    def explain(self) -> str:
        lines = [f"individual quality: " + ", ".join(f"{s}={q:.3f}" for s, q in sorted(self.individual_quality.items(), key=lambda kv: (-kv[1], kv[0])))]
        for c in sorted(self.evaluated, key=lambda c: (-(c.score or -1), c.label())):
            comp = c.components
            if c.feasible:
                lines.append(f"  {c.label():<14} feasible score={c.score:.3f} (quality={comp['mean_quality']:.3f} angle={comp['angle_deg']:.1f}deg "
                             f"geometry={comp['geometry']:.2f} skew={comp['skew_ms']:.0f}ms)")
            else:
                ang = "" if comp.get("angle_deg") is None else f" angle={comp['angle_deg']:.1f}deg"
                lines.append(f"  {c.label():<14} REJECTED {','.join(c.rejection_codes)}{ang}")
        lines.append(f"decision: {self.outcome}" + (f" -> {self.chosen.label()}" if self.chosen else "") + f" ({self.reason})")
        return "\n".join(lines)


def ray_angle_deg(d1, d2) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))))


def ray_gap(o1, d1, o2, d2) -> tuple[float, bool]:
    """(closest distance between two lines, both closest points in front of their cameras)."""
    o1, d1, o2, d2 = (np.asarray(v, float) for v in (o1, d1, o2, d2))
    n = np.cross(d1, d2)
    nn = float(np.linalg.norm(n))
    w = o2 - o1
    if nn < 1e-9:                                         # parallel
        return float(np.linalg.norm(np.cross(w, d1))), False
    gap = abs(float(np.dot(w, n))) / nn
    t1 = float(np.dot(np.cross(w, d2), n)) / nn ** 2
    t2 = float(np.dot(np.cross(w, d1), n)) / nn ** 2
    return gap, (t1 > 0 and t2 > 0)


class SetSelector:
    def __init__(self, cfg: CoordinationCfg):
        self.cfg = cfg

    def select(self, offers: list[ObservationOffer], task: TaskRequest, banned: set[frozenset] | None = None) -> SetSelection:
        sc, w = self.cfg.set_selection, self.cfg.weights
        banned = banned or set()
        indiv = {o.sensor_id: scoring.quality_of(w, o) for o in offers}
        best_ind = max(indiv, key=lambda s: (indiv[s], s)) if indiv else None
        need = max(sc.min_views, task.min_views)
        if len(offers) < need:
            return SetSelection(None, [], "NO_SET_INSUFFICIENT_SENSORS",
                                f"{len(offers)} eligible sensor(s), {need} required for triangulation", indiv, best_ind)
        pool = sorted(offers, key=lambda o: (-indiv[o.sensor_id], o.sensor_id))[: sc.max_candidates]
        pool = sorted(pool, key=lambda o: o.sensor_id)
        evaluated: list[CandidateEvaluation] = []
        for n in range(need, sc.max_set_size + 1):
            for combo in itertools.combinations(pool, n):
                evaluated.append(self._evaluate(combo, task, indiv, banned))
        feasible = [c for c in evaluated if c.feasible]
        if feasible:
            best = min(feasible, key=lambda c: (-c.score, len(c.sensor_ids), c.sensor_ids))
            why = f"best of {len(feasible)} feasible set(s) out of {len(evaluated)}"
            if best_ind is not None and best_ind not in best.sensor_ids:
                why += f"; highest individual sensor {best_ind} excluded by set rules"
            return SetSelection(best, evaluated, "AWARD", why, indiv, best_ind)
        counts = collections.Counter(code for c in evaluated for code in c.rejection_codes[:1])
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        dominant = top[0][0]
        outcome = "NO_SET_INSUFFICIENT_GEOMETRY" if dominant in GEOMETRY_CODES else "NO_COMPATIBLE_SET"
        return SetSelection(None, evaluated, outcome, "no feasible set; first-failure counts: " + ", ".join(f"{k}x{v}" for k, v in top), indiv, best_ind)

    def _evaluate(self, combo, task: TaskRequest, indiv: dict[str, float], banned: set[frozenset]) -> CandidateEvaluation:
        sc = self.cfg.set_selection
        ids = tuple(o.sensor_id for o in combo)
        codes: list[str] = []
        comp: dict = {"angle_deg": None, "skew_ms": None, "ray_gap_m": None, "baseline_m": None, "mean_quality": None, "geometry": None}
        if frozenset(ids) in banned:
            codes.append("RECENTLY_FAILED_SET")
        keys = {o.rig_key for o in combo}
        if None in keys or len(keys) > 1:
            codes.append("INCOMPATIBLE_RIG")
        times = [o.observation_effective_ms for o in combo]
        skew = (max(times) - min(times)) if all(t is not None for t in times) else math.inf
        comp["skew_ms"] = None if math.isinf(skew) else skew
        if skew > sc.max_skew_ms:
            codes.append("SKEW_EXCEEDS_WINDOW")
        hands = {o.hand for o in combo if o.hand != "unknown"}
        if len(hands) > 1:
            codes.append("HAND_LABEL_MISMATCH")
        if any(o.estimated_latency_ms > sc.max_latency_ms for o in combo):
            codes.append("LATENCY_BUDGET")
        if any(o.submitted_at_ms + o.estimated_latency_ms > task.deadline_ms for o in combo):
            codes.append("DEADLINE_VIOLATION")
        angle, gap_max, base = 0.0, 0.0, 0.0
        if any(o.view_ray is None for o in combo):
            codes.append("INCOMPLETE_GEOMETRY")
        else:
            for a, b in itertools.combinations(combo, 2):
                (oa, da), (ob, db) = a.view_ray, b.view_ray
                angle = max(angle, ray_angle_deg(da, db))
                g, front = ray_gap(oa, da, ob, db)
                gap_max = max(gap_max, g if front else math.inf)
                base = max(base, float(np.linalg.norm(np.asarray(oa) - np.asarray(ob))))
            comp.update(angle_deg=angle, ray_gap_m=None if math.isinf(gap_max) else gap_max, baseline_m=base)
            if angle < sc.min_view_angle_deg:
                codes.append("INSUFFICIENT_VIEW_ANGLE")
            elif gap_max > sc.max_ray_gap_m:
                codes.append("RAYS_INCONSISTENT")
        # order so the FIRST code is the most informative geometry/policy reason
        order = ["RECENTLY_FAILED_SET", "INCOMPATIBLE_RIG", "INCOMPLETE_GEOMETRY", "INSUFFICIENT_VIEW_ANGLE", "RAYS_INCONSISTENT",
                 "SKEW_EXCEEDS_WINDOW", "HAND_LABEL_MISMATCH", "LATENCY_BUDGET", "DEADLINE_VIOLATION"]
        codes.sort(key=order.index)
        mean_q = float(np.mean([indiv[i] for i in ids]))
        comp["mean_quality"] = mean_q
        if codes:
            return CandidateEvaluation(ids, False, tuple(codes), None, comp)
        geo = min(1.0, angle / sc.good_view_angle_deg)
        syn = max(0.0, 1.0 - skew / sc.max_skew_ms)
        comp["geometry"], comp["sync"] = geo, syn
        score = sc.w_quality * mean_q + sc.w_geometry * geo + sc.w_sync * syn - sc.size_penalty * (len(ids) - sc.min_views)
        return CandidateEvaluation(ids, True, (), float(max(0.0, score)), comp)
