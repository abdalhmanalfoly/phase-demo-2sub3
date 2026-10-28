"""Observation synchronizer: bounded per-sensor buffers + capture-time matching across sensors.

TIME MODEL
  Each observation has ``captured_at_ms`` (the SENSOR's clock) and ``received_at_ms`` (the HOST's clock).
  Phones are NOT clock-synchronized. Phase 1 estimates a per-sensor offset ``host_clock - sensor_clock`` from
  heartbeats (minimum of host_rx - client_time, so it contains the smallest one-way network delay). We use
      effective_ms = captured_at_ms + offset_ms
  as the common timeline for matching. Residual inter-sensor timing error is about the difference between the
  sensors' minimum one-way delays (typically a few ms on a LAN) - MEASURED skew is reported, never assumed zero.
  ``received_at_ms`` is used only for transport-delay diagnostics and clock sanity checks, never as observation time.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

from ..config import SyncCfg


@dataclass(frozen=True)
class ObsRef:
    sensor_id: str
    sequence_number: int
    frame_id: int
    captured_at_ms: int
    received_at_ms: int
    effective_ms: float
    image_width: int
    image_height: int
    status: str
    hand: str
    landmarks: dict                    # id -> (x_px, y_px, confidence|None)

    def has(self, landmark: str) -> bool:
        return self.status == "tracked" and landmark in self.landmarks


@dataclass
class SyncResult:
    members: list[ObsRef] = field(default_factory=list)       # one per sensor, all within the window (>=2 when matched)
    matched: bool = False
    is_new: bool = False                                      # contains data not yet consumed by fusion
    skew_ms: float | None = None                              # skew of the matched set, or best achievable pair skew
    reason: str | None = None                                 # SKEW_EXCEEDS_WINDOW | None
    usable: dict[str, ObsRef] = field(default_factory=dict)   # newest fresh usable observation per sensor
    unusable: dict[str, str] = field(default_factory=dict)    # sensor -> NO_OBSERVATION | STALE | TARGET_NOT_VISIBLE


class ObservationSynchronizer:
    def __init__(self, cfg: SyncCfg):
        self.cfg = cfg
        self.buffers: dict[str, collections.deque[ObsRef]] = {}
        self.consumed: dict[tuple[str, str], int] = {}        # (landmark, sensor) -> last consumed sequence
        self.rejected: collections.Counter = collections.Counter()
        self.sequence_gaps: collections.Counter = collections.Counter()
        self.overflow_drops: collections.Counter = collections.Counter()
        self.delays: dict[str, collections.deque[float]] = {}

    # ------------------------------------------------------------------ ingest
    def add(self, entry: dict, offset_ms: int | None, now_ms: float) -> tuple[bool, str | None]:
        """Validate ordering/clock/staleness and buffer. Returns (accepted, reject_reason)."""
        sid = entry["sensor_id"]
        if offset_ms is None:
            return self._reject(sid, "NO_CLOCK_OFFSET")
        eff = entry["captured_at_ms"] + offset_ms
        if entry["received_at_ms"] - eff < -self.cfg.clock_suspect_ms:
            return self._reject(sid, "CLOCK_SUSPECT")          # received before it was captured: offset is wrong
        if now_ms - eff > self.cfg.max_observation_age_ms:
            return self._reject(sid, "STALE")
        buf = self.buffers.setdefault(sid, collections.deque(maxlen=self.cfg.buffer_per_sensor))
        if buf:
            last = buf[-1]
            if entry["sequence_number"] <= last.sequence_number:
                return self._reject(sid, "OUT_OF_ORDER_SEQUENCE")
            if eff < last.effective_ms - self.cfg.out_of_order_tolerance_ms:
                return self._reject(sid, "OUT_OF_ORDER_TIME")
            self.sequence_gaps[sid] += entry["sequence_number"] - last.sequence_number - 1
        if len(buf) == buf.maxlen:
            self.overflow_drops[sid] += 1                      # bounded: oldest intermediate frame is dropped
        lms = {l["id"]: (l["x_px"], l["y_px"], l["confidence"]) for l in entry["landmarks"]}
        buf.append(ObsRef(sid, entry["sequence_number"], entry["frame_id"], entry["captured_at_ms"], entry["received_at_ms"],
                          float(eff), entry["image_width"], entry["image_height"], entry["status"], entry["hand"], lms))
        self.delays.setdefault(sid, collections.deque(maxlen=64)).append(entry["received_at_ms"] - eff)
        return True, None

    def _reject(self, sid: str, reason: str) -> tuple[bool, str]:
        self.rejected[(sid, reason)] += 1
        return False, reason

    def clear(self) -> None:
        self.buffers.clear()
        self.consumed.clear()

    # ------------------------------------------------------------------ matching
    def active_sensors(self, now_ms: float, window_ms: float) -> list[str]:
        return [s for s, b in self.buffers.items() if b and now_ms - b[-1].effective_ms <= window_ms]

    def _candidates(self, landmark: str, sid: str, now_ms: float) -> tuple[list[ObsRef], str | None]:
        buf = self.buffers.get(sid)
        if not buf:
            return [], "NO_OBSERVATION"
        out: list[ObsRef] = []
        for o in reversed(buf):                                # newest -> oldest, stop at the most recent "no hand"
            if o.status != "tracked" or landmark not in o.landmarks:
                if not out:
                    return [], "TARGET_NOT_VISIBLE"
                break
            if now_ms - o.effective_ms <= self.cfg.max_observation_age_ms:
                out.append(o)
        if not out:
            return [], "STALE"
        return sorted(out, key=lambda o: o.effective_ms), None

    def match(self, landmark: str, sensor_ids: list[str], now_ms: float) -> SyncResult:
        res = SyncResult()
        cands: dict[str, list[ObsRef]] = {}
        for sid in sensor_ids:
            c, why = self._candidates(landmark, sid, now_ms)
            if c:
                cands[sid] = c
                res.usable[sid] = c[-1]
            else:
                res.unusable[sid] = why or "NO_OBSERVATION"
        if len(cands) < 2:
            return res
        W = self.cfg.max_skew_ms
        best: tuple | None = None
        for psid, plist in cands.items():
            for p in plist:                                    # pivot = the newest member of the candidate set
                members = [p]
                for osid, olist in cands.items():
                    if osid == psid:
                        continue
                    inwin = [o for o in olist if p.effective_ms - W <= o.effective_ms <= p.effective_ms]
                    if inwin:
                        members.append(max(inwin, key=lambda o: o.effective_ms))
                if len(members) < 2:
                    continue
                skew = max(m.effective_ms for m in members) - min(m.effective_ms for m in members)
                rank = (len(members), p.effective_ms, -skew)   # most views, then newest, then tightest
                if best is None or rank > best[0]:
                    best = (rank, members, skew)
        if best is not None:
            _, members, skew = best
            res.members, res.matched, res.skew_ms = sorted(members, key=lambda m: m.sensor_id), True, float(skew)
            res.is_new = any(m.sequence_number > self.consumed.get((landmark, m.sensor_id), -1) for m in members)
            return res
        # no window works: report the best pair skew we could have had
        gaps = [abs(a.effective_ms - b.effective_ms)
                for i, (s1, l1) in enumerate(cands.items()) for s2, l2 in list(cands.items())[i + 1:] for a in l1 for b in l2]
        res.skew_ms, res.reason = float(min(gaps)), "SKEW_EXCEEDS_WINDOW"
        return res

    def mark_consumed(self, landmark: str, members: list[ObsRef]) -> None:
        for m in members:
            self.consumed[(landmark, m.sensor_id)] = m.sequence_number

    def diagnostics(self) -> dict:
        def stats(d):
            v = sorted(d)
            return {"n": len(v), "min_ms": v[0], "median_ms": v[len(v) // 2], "max_ms": v[-1]} if v else None
        return {"rejected": {f"{s}:{r}": n for (s, r), n in self.rejected.items()},
                "sequence_gaps": dict(self.sequence_gaps), "overflow_drops": dict(self.overflow_drops),
                "transport_delay": {s: stats(d) for s, d in self.delays.items()},
                "buffered": {s: len(b) for s, b in self.buffers.items()}}
