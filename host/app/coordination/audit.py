"""Structured audit log + counters + stage timing, shared by the Coordination and Gesture agents.

``AuditLog.log`` has the same signature as the host's ``EventLogger.log`` so a real ``EventLogger`` can be passed as
``sink`` and every coordination/gesture event also lands in ``host-events.jsonl``. Only numbers, ids and reason codes are
logged: keys that could carry imagery are refused (privacy guard), values are coerced to JSON-safe types.
"""
from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

FORBIDDEN_KEYS = frozenset({"frame", "image", "image_data", "pixels", "video", "jpeg", "png", "bitmap", "raw_frame"})


def _safe(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4) if math.isfinite(v) else None
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        items = sorted(v, key=str) if isinstance(v, (set, frozenset)) else v
        return [_safe(x) for x in items]
    if hasattr(v, "value"):                                  # enums
        return _safe(v.value)
    return str(v)


class AuditLog:
    def __init__(self, clock, *, component: str, session_id: str | None = None, sink=None,
                 jsonl_path: Path | str | None = None, max_entries: int = 5000):
        self.clock, self.component, self.session_id, self.sink = clock, component, session_id, sink
        self.entries: collections.deque[dict] = collections.deque(maxlen=max_entries)
        self.counters: collections.Counter = collections.Counter()
        self._fh = None
        if jsonl_path is not None:
            p = Path(jsonl_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("a", encoding="utf-8")

    def count(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def log(self, event_type: str, *, session_id: str | None = None, sensor_id: str | None = None,
            reason: str | None = None, severity: str = "info", **fields) -> dict:
        bad = FORBIDDEN_KEYS & set(fields)
        if bad:
            raise ValueError(f"audit log refuses image-like fields: {sorted(bad)}")
        entry = {"ts_ms": self.clock.now_ms(), "severity": severity, "component": self.component, "event_type": event_type,
                 "session_id": session_id or self.session_id, "sensor_id": sensor_id, "reason": reason,
                 **{k: _safe(v) for k, v in fields.items()}}
        self.entries.append(entry)
        self.counters[f"event:{event_type}"] += 1
        if self._fh is not None:
            self._fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        if self.sink is not None:
            try:
                self.sink.log(event_type, session_id=entry["session_id"], sensor_id=sensor_id, reason=reason,
                              severity=severity, component=self.component, **{k: v for k, v in entry.items()
                                                                              if k not in ("ts_ms", "severity", "component", "event_type", "session_id", "sensor_id", "reason")})
            except Exception:                                # logging must never break coordination
                self.counters["sink_errors"] += 1
        return entry

    def of_type(self, event_type: str) -> list[dict]:
        return [e for e in self.entries if e["event_type"] == event_type]

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class StageTimer:
    """Accumulates wall time per processing stage (``perf`` is injectable, so tests are deterministic)."""

    def __init__(self, perf: Callable[[], float] = time.perf_counter):
        self.perf = perf
        self.totals_ms: collections.Counter = collections.Counter()
        self.calls: collections.Counter = collections.Counter()
        self.last_ms: dict[str, float] = {}

    def stage(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = timer.perf()
                return self_inner

            def __exit__(self_inner, *exc):
                dt = (timer.perf() - self_inner.t0) * 1000.0
                timer.totals_ms[name] += dt
                timer.calls[name] += 1
                timer.last_ms[name] = dt
                return False
        return _Ctx()

    def summary(self) -> dict:
        return {k: {"calls": self.calls[k], "total_ms": round(v, 3), "last_ms": round(self.last_ms[k], 3)}
                for k, v in self.totals_ms.items()}
