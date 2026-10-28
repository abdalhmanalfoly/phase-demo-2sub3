"""Structured event log: one JSON object per line with timestamp, session, sensor, event type, reason.

Landmark coordinates are never written here (see ``observation_recorder``); raw video/images never
exist on the host in the default pipeline.
"""
from __future__ import annotations

import collections
import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SEVERITY = {"debug": 10, "info": 20, "warning": 30, "error": 40}
_COUNTER = 0


class EventLogger:
    def __init__(self, log_dir: Path, clock, buffer_size: int = 200, console: bool = True):
        global _COUNTER
        _COUNTER += 1
        self._clock = clock
        self.recent: collections.deque[dict] = collections.deque(maxlen=buffer_size)
        self._logger = logging.getLogger(f"saga.events.{_COUNTER}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "host-events.jsonl", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(fh)
        if console:
            ch = logging.StreamHandler(sys.stderr)
            ch.setFormatter(logging.Formatter("%(message)s"))
            ch.addFilter(lambda rec: rec.levelno >= logging.INFO)
            self._logger.addHandler(ch)
        self._console = console

    def log(self, event_type: str, *, session_id: str | None = None, sensor_id: str | None = None,
            reason: str | None = None, severity: str = "info", **fields) -> dict:
        ts = self._clock.now_ms()
        entry = {
            "ts_ms": ts,
            "ts": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "severity": severity, "event_type": event_type,
            "session_id": session_id, "sensor_id": sensor_id, "reason": reason, **fields,
        }
        self.recent.append(entry)
        line = json.dumps(entry, separators=(",", ":"), default=str)
        self._logger.log(_SEVERITY.get(severity, 20), line)
        return entry

    def close(self) -> None:
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)
