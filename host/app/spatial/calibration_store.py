"""Calibration persistence, validation, rig-compatibility checks, and sensor<->camera binding.

Layout:  <calibration_dir>/<rig_id>/<camera_id>.json          current artifact for a camera in a rig
         <calibration_dir>/<rig_id>/history/<camera_id>__<calibration_id>.json   replaced/older artifacts
         <calibration_dir>/intrinsics/<camera_id>__<WxH>.json   reusable intrinsics
Writes are atomic (temp file + rename). Reads validate everything and raise ``CalibrationLoadError``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .calibration_schema import CalibrationArtifact, IntrinsicsArtifact
from .geometry import CameraModel

_SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class CalibrationLoadError(Exception):
    def __init__(self, code: str, message: str, path: Path | None = None):
        super().__init__(f"{code}: {message}" + (f" ({path})" if path else ""))
        self.code, self.message, self.path = code, message, path


def _read_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CalibrationLoadError("NOT_FOUND", "calibration file does not exist", path) from None
    except OSError as exc:
        raise CalibrationLoadError("UNREADABLE", str(exc), path) from None
    try:
        data = json.loads(text)
    except ValueError:
        raise CalibrationLoadError("CORRUPT", "file is not valid JSON", path) from None
    if not isinstance(data, dict):
        raise CalibrationLoadError("CORRUPT", "top-level JSON value must be an object", path)
    return data


def _validate(model, data: dict, path: Path | None):
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "$"
        msg = str(first["msg"]).removeprefix("Value error, ")
        code = "UNSUPPORTED_VERSION" if "calibration_version" in loc else "INVALID"
        raise CalibrationLoadError(code, f"{loc}: {msg}", path) from None


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        finally:
            raise


def load_calibration_file(path: Path | str) -> CalibrationArtifact:
    path = Path(path)
    return _validate(CalibrationArtifact, _read_json(path), path)


def load_intrinsics_file(path: Path | str) -> IntrinsicsArtifact:
    path = Path(path)
    return _validate(IntrinsicsArtifact, _read_json(path), path)


def camera_model(art: CalibrationArtifact) -> CameraModel:
    return CameraModel(art.camera_id, np.array(art.camera_matrix, float), np.array(art.distortion, float),
                       np.array(art.extrinsics.rotation, float), np.array(art.extrinsics.translation, float),
                       art.image_width, art.image_height)


@dataclass
class LoadReport:
    artifacts: dict[str, CalibrationArtifact] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)        # file stem -> "CODE: message"


class CalibrationStore:
    def __init__(self, root: Path, rig_id: str):
        if not _SAFE.match(rig_id):
            raise ValueError("invalid rig_id")
        self.root, self.rig_id = Path(root), rig_id

    @property
    def rig_dir(self) -> Path:
        return self.root / self.rig_id

    # ----------------------------------------------------------------- intrinsics
    def save_intrinsics(self, art: IntrinsicsArtifact) -> Path:
        path = self.root / "intrinsics" / f"{art.camera_id}__{art.image_width}x{art.image_height}.json"
        _atomic_write(path, art.model_dump(mode="json"))
        return path

    # ----------------------------------------------------------------- rig cameras
    def camera_path(self, camera_id: str) -> Path:
        if not _SAFE.match(camera_id):
            raise ValueError("invalid camera_id")
        return self.rig_dir / f"{camera_id}.json"

    def save_camera(self, art: CalibrationArtifact) -> Path:
        """Save as the CURRENT artifact for the camera; any previous current artifact moves to history/."""
        if art.rig_id != self.rig_id:
            raise ValueError(f"artifact rig_id '{art.rig_id}' != store rig_id '{self.rig_id}'")
        path = self.camera_path(art.camera_id)
        if path.exists():
            try:
                old = load_calibration_file(path)
                hist = self.rig_dir / "history" / f"{old.camera_id}__{old.calibration_id}.json"
                hist.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, hist)
            except CalibrationLoadError:
                bad = self.rig_dir / "history" / f"{art.camera_id}__unreadable_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
                bad.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, bad)
        _atomic_write(path, art.model_dump(mode="json"))
        return path

    def load_camera(self, camera_id: str) -> CalibrationArtifact:
        return load_calibration_file(self.camera_path(camera_id))

    def load_all(self) -> LoadReport:
        rep = LoadReport()
        if not self.rig_dir.is_dir():
            return rep
        for p in sorted(self.rig_dir.glob("*.json")):
            try:
                art = load_calibration_file(p)
                if art.camera_id != p.stem:
                    raise CalibrationLoadError("INVALID", f"camera_id '{art.camera_id}' does not match file name", p)
                if art.rig_id != self.rig_id:
                    raise CalibrationLoadError("INVALID", f"rig_id '{art.rig_id}' does not match store '{self.rig_id}'", p)
                rep.artifacts[art.camera_id] = art
            except CalibrationLoadError as exc:
                rep.errors[p.stem] = f"{exc.code}: {exc.message}"
        return rep

    def invalidate(self, camera_id: str, reason: str) -> CalibrationArtifact:
        """Persistently mark a camera's calibration invalid (camera moved/replaced). Fusion refuses it until recalibrated."""
        art = self.load_camera(camera_id)
        art.validity.status = "invalid"
        art.validity.reason = reason[:200]
        art.validity.invalidated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write(self.camera_path(camera_id), art.model_dump(mode="json"))
        return art


def check_rig(artifacts: dict[str, CalibrationArtifact], allow_synthetic: bool = False) -> dict:
    """Operator-facing compatibility report for a set of cameras (used by ``calibrate.py inspect``)."""
    cams = {}
    for cid, a in artifacts.items():
        problems = []
        if a.validity.status != "valid":
            problems.append(f"INVALIDATED: {a.validity.reason or 'no reason recorded'}")
        if a.synthetic and not allow_synthetic:
            problems.append("SYNTHETIC: artifact is flagged synthetic (example/test data)")
        cams[cid] = {"calibration_id": a.calibration_id, "rig_calibration_id": a.rig_calibration_id,
                     "image": f"{a.image_width}x{a.image_height}", "created_at": a.created_at,
                     "intrinsic_rms_px": a.quality.intrinsic_rms_px, "extrinsic_rms_px": a.quality.extrinsic_rms_px,
                     "warnings": list(a.validity.warnings), "problems": problems}
    usable = [c for c, v in cams.items() if not v["problems"]]
    groups: dict[tuple, list[str]] = {}
    for c in usable:
        a = artifacts[c]
        groups.setdefault((a.rig_id, a.rig_calibration_id, a.world_frame_id, a.unit), []).append(c)
    compatible = max(groups.values(), key=len) if groups else []
    for c in usable:
        if c not in compatible:
            cams[c]["problems"].append("INCOMPATIBLE_RIG: calibrated in a different rig session / world frame than the majority")
    return {"cameras": cams, "compatible_cameras": sorted(compatible),
            "can_triangulate": len(compatible) >= 2,
            "summary": ("READY: %d compatible calibrated cameras" % len(compatible)) if len(compatible) >= 2
            else "NOT READY: need at least 2 valid, compatible calibrated cameras (have %d)" % len(compatible)}


@dataclass
class Resolution:
    """Outcome of looking up the calibration for a session-local sensor."""
    sensor_id: str
    camera_id: str | None
    model: CameraModel | None
    artifact: CalibrationArtifact | None
    problem: str | None            # None = usable; else NOT_BOUND | NOT_FOUND | INVALIDATED | DIMENSION_MISMATCH | SYNTHETIC_REFUSED | LOAD_ERROR | INCOMPATIBLE_RIG
    detail: str = ""


class CalibrationRegistry:
    """Runtime view of the store for one session: bindings (session-local sensor_id -> camera_id) + loaded artifacts.

    Auto-binding: a sensor whose id equals a known camera_id (e.g. it registered with requested_sensor_id=phone_left)
    is bound automatically. Otherwise the operator binds explicitly (API/CLI).
    """

    def __init__(self, store: CalibrationStore, allow_synthetic: bool = False):
        self.store, self.allow_synthetic = store, allow_synthetic
        self.artifacts: dict[str, CalibrationArtifact] = {}
        self.load_errors: dict[str, str] = {}
        self.bindings: dict[str, str] = {}
        self.generation = 0                  # bumps on reload/invalidate: fusion resets smoothing when it changes
        self._models: dict[str, CameraModel] = {}
        self.reload()

    def reload(self) -> None:
        rep = self.store.load_all()
        self.artifacts, self.load_errors = rep.artifacts, rep.errors
        self._models = {}
        self.generation += 1

    def bind(self, sensor_id: str, camera_id: str) -> None:
        self.bindings[sensor_id] = camera_id
        self.generation += 1

    def unbind(self, sensor_id: str) -> None:
        if self.bindings.pop(sensor_id, None) is not None:
            self.generation += 1

    def camera_for(self, sensor_id: str) -> str | None:
        if sensor_id in self.bindings:
            return self.bindings[sensor_id]
        if sensor_id in self.artifacts or sensor_id in self.load_errors:
            return sensor_id
        return None

    def invalidate(self, camera_id: str, reason: str) -> None:
        self.store.invalidate(camera_id, reason)
        self.reload()

    def resolve(self, sensor_id: str, image_width: int | None = None, image_height: int | None = None) -> Resolution:
        cid = self.camera_for(sensor_id)
        if cid is None:
            return Resolution(sensor_id, None, None, None, "NOT_BOUND", "no camera_id bound to this sensor")
        if cid in self.load_errors:
            return Resolution(sensor_id, cid, None, None, "LOAD_ERROR", self.load_errors[cid])
        art = self.artifacts.get(cid)
        if art is None:
            return Resolution(sensor_id, cid, None, None, "NOT_FOUND", f"no calibration file for camera '{cid}'")
        if art.validity.status != "valid":
            return Resolution(sensor_id, cid, None, art, "INVALIDATED", art.validity.reason or "marked invalid")
        if art.synthetic and not self.allow_synthetic:
            return Resolution(sensor_id, cid, None, art, "SYNTHETIC_REFUSED", "synthetic calibration refused")
        if image_width is not None and (image_width, image_height) != (art.image_width, art.image_height):
            return Resolution(sensor_id, cid, None, art, "DIMENSION_MISMATCH",
                              f"observation {image_width}x{image_height} != calibrated {art.image_width}x{art.image_height}")
        model = self._models.get(cid) or self._models.setdefault(cid, camera_model(art))
        return Resolution(sensor_id, cid, model, art, None)

    def compatible_ids(self, sensor_ids: list[str]) -> tuple[set[str], dict[str, str]]:
        """Among usable sensors keep the largest group sharing one rig calibration + world frame; report the rest."""
        groups: dict[tuple, list[str]] = {}
        for sid in sensor_ids:
            r = self.resolve(sid)
            if r.problem is None or r.problem == "DIMENSION_MISMATCH":
                a = r.artifact
                groups.setdefault((a.rig_id, a.rig_calibration_id, a.world_frame_id, a.unit), []).append(sid)
        if not groups:
            return set(), {}
        best = max(groups.values(), key=len)
        return set(best), {s: "INCOMPATIBLE_RIG" for g in groups.values() if g is not best for s in g}

    def status(self, sensor_ids: list[str] | None = None) -> dict:
        sids = sensor_ids if sensor_ids is not None else sorted(self.bindings)
        out = {}
        for sid in sids:
            r = self.resolve(sid)
            out[sid] = {"camera_id": r.camera_id, "ok": r.problem is None, "problem": r.problem, "detail": r.detail,
                        "calibration_id": r.artifact.calibration_id if r.artifact else None}
        return {"rig_id": self.store.rig_id, "generation": self.generation, "bindings": dict(self.bindings),
                "sensors": out, "cameras_on_disk": sorted(self.artifacts), "load_errors": dict(self.load_errors),
                "report": check_rig(self.artifacts, self.allow_synthetic)}
