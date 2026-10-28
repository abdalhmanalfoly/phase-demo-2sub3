"""Per-session JSONL record of accepted landmark observations (numbers only, no imagery).

Layout: ``<log_dir>/sessions/<session_id>/observations.jsonl``. Each line is the validated message plus
host-assigned ``received_at_ms`` and ``estimated_age_ms``. Buffered; flushed by the sweeper and on close.
"""
from __future__ import annotations

import json
from pathlib import Path


class ObservationRecorder:
    def __init__(self, log_dir: Path, session_id: str, enabled: bool, filename: str = "observations.jsonl"):
        self.enabled = enabled
        self.path = log_dir / "sessions" / session_id / filename
        self._fh = None
        self.lines_written = 0

    def write(self, entry: dict) -> None:
        if not self.enabled:
            return
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self.lines_written += 1

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
