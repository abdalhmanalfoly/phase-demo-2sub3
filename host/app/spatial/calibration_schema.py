"""Versioned calibration artifacts (JSON). Validated on every load; invalid/corrupt files are never silently used.

Two artifact kinds:
* ``IntrinsicsArtifact``  - per physical camera + resolution: K, distortion, quality. Reusable across rigs.
* ``CalibrationArtifact`` - one camera inside a calibrated RIG: intrinsics + extrinsics (world -> camera) in the
  shared rig/world frame. This is what fusion consumes.

Transform convention (stored explicitly in the file as ``extrinsics.convention``):
    X_cam = R @ X_world + t        ("world_to_camera", OpenCV).  Camera->world is derived and cross-checked on load.
Units: metres everywhere (``unit`` must be "m").
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .geometry import is_rotation

CALIBRATION_VERSION = "1.0"
MAX_IMAGE_DIM = 16384
_CAM_ID = r"^[A-Za-z0-9_-]{1,32}$"
_ID = r"^[A-Za-z0-9_.-]{1,64}$"

Matrix3 = list[list[float]]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _check_shape(name: str, m, shape: tuple[int, ...]):
    a = np.array(m, dtype=float) if m is not None else None
    if a is None or a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return a


class Target(_Model):
    type: Literal["checkerboard"]
    inner_corners_cols: int = Field(ge=3, le=50)
    inner_corners_rows: int = Field(ge=3, le=50)
    square_size: float = Field(gt=0, le=10)
    unit: Literal["m"]


class RejectedSample(_Model):
    name: str = Field(max_length=200)
    reason: str = Field(max_length=200)
    error_px: float | None = None


class Quality(_Model):
    intrinsic_rms_px: float | None = Field(default=None, ge=0)
    extrinsic_rms_px: float | None = Field(default=None, ge=0)
    samples_used: int = Field(ge=0)
    samples_rejected: list[RejectedSample] = []
    per_sample_rms_px: dict[str, float] = {}
    coverage_fraction: float | None = Field(default=None, ge=0, le=1)
    tilt_spread_deg: float | None = Field(default=None, ge=0)
    rig_check: dict | None = None            # triangulation of board corners across cameras (internal consistency)


class Validity(_Model):
    status: Literal["valid", "invalid"] = "valid"
    warnings: list[str] = []
    reason: str | None = None
    invalidated_at: str | None = None


class Extrinsics(_Model):
    convention: Literal["world_to_camera"]
    rotation: Matrix3
    translation: list[float]
    camera_to_world_rotation: Matrix3          # derived = R.T
    camera_center_world: list[float]           # derived = -R.T @ t


class _Base(_Model):
    calibration_version: str
    created_at: str
    synthetic: bool = False
    notes: str | None = Field(default=None, max_length=500)
    image_width: int = Field(ge=16, le=MAX_IMAGE_DIM)
    image_height: int = Field(ge=16, le=MAX_IMAGE_DIM)
    target: Target
    camera_matrix: Matrix3
    distortion: list[float]
    quality: Quality
    validity: Validity = Validity()

    @field_validator("calibration_version")
    @classmethod
    def _ver(cls, v):
        if v != CALIBRATION_VERSION:
            raise ValueError(f"unsupported calibration_version '{v}' (this build reads '{CALIBRATION_VERSION}')")
        return v

    @field_validator("created_at")
    @classmethod
    def _dt(cls, v):
        datetime.fromisoformat(v)
        return v

    @model_validator(mode="after")
    def _intrinsics_sane(self):
        K = _check_shape("camera_matrix", self.camera_matrix, (3, 3))
        _check_shape("distortion", self.distortion, (5,))
        if not (K[0, 0] > 0 and K[1, 1] > 0):
            raise ValueError("camera_matrix focal lengths must be positive")
        if not (np.isclose(K[2], [0, 0, 1]).all() and np.isclose(K[1, 0], 0) and np.isclose(K[2, 0], 0)):
            raise ValueError("camera_matrix must have the form [[fx,s,cx],[0,fy,cy],[0,0,1]]")
        if abs(np.linalg.det(K)) < 1e-9 or np.linalg.cond(K) > 1e8:
            raise ValueError("camera_matrix is singular or ill-conditioned")
        if not (0 < K[0, 2] < self.image_width and 0 < K[1, 2] < self.image_height):
            raise ValueError("principal point lies outside the image")
        ratio = K[0, 0] / K[1, 1]
        if not (0.5 < ratio < 2.0):
            raise ValueError("fx/fy ratio is implausible (> 2x)")
        return self


class IntrinsicsArtifact(_Base):
    kind: Literal["intrinsics"] = "intrinsics"
    intrinsics_id: str = Field(pattern=_ID)
    camera_id: str = Field(pattern=_CAM_ID)            # physical device label, e.g. "phone_left"


class CalibrationArtifact(_Base):
    kind: Literal["camera_calibration"] = "camera_calibration"
    calibration_id: str = Field(pattern=_ID)
    camera_id: str = Field(pattern=_CAM_ID)
    rig_id: str = Field(pattern=_ID)
    rig_calibration_id: str = Field(pattern=_ID)       # shared by every camera calibrated against the same reference-board session
    world_frame_id: str = Field(pattern=_ID)
    unit: Literal["m"]
    intrinsics_id: str | None = None
    extrinsics: Extrinsics
    projection_matrix: list[list[float]]              # K @ [R|t], 3x4 (derived, cross-checked)

    @model_validator(mode="after")
    def _extrinsics_consistent(self):
        R = _check_shape("extrinsics.rotation", self.extrinsics.rotation, (3, 3))
        t = _check_shape("extrinsics.translation", self.extrinsics.translation, (3,))
        if not is_rotation(R, tol=1e-5):
            raise ValueError("extrinsics.rotation is not a proper rotation matrix (orthonormal, det=+1)")
        Rc = _check_shape("extrinsics.camera_to_world_rotation", self.extrinsics.camera_to_world_rotation, (3, 3))
        C = _check_shape("extrinsics.camera_center_world", self.extrinsics.camera_center_world, (3,))
        if not np.allclose(Rc, R.T, atol=1e-6) or not np.allclose(C, -R.T @ t, atol=1e-6):
            raise ValueError("derived camera->world transform is inconsistent with world->camera")
        P = _check_shape("projection_matrix", self.projection_matrix, (3, 4))
        K = np.array(self.camera_matrix, float)
        if not np.allclose(P, K @ np.hstack([R, t.reshape(3, 1)]), rtol=1e-6, atol=1e-6):
            raise ValueError("projection_matrix is inconsistent with K, R, t")
        return self


def make_extrinsics(R: np.ndarray, t: np.ndarray) -> tuple[Extrinsics, list[list[float]]]:
    """Build the stored extrinsics block (and derived pieces) from world->camera R, t."""
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(3)
    ext = Extrinsics(convention="world_to_camera", rotation=R.tolist(), translation=t.tolist(),
                     camera_to_world_rotation=R.T.tolist(), camera_center_world=(-R.T @ t).tolist())
    return ext, ext_projection(R, t)


def ext_projection(R, t, K=None):  # pragma: no cover - small helper
    return None if K is None else (np.asarray(K, float) @ np.hstack([R, np.asarray(t).reshape(3, 1)])).tolist()
