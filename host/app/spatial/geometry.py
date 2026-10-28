"""Pure multi-view geometry (NumPy + OpenCV). No I/O, no host dependencies: unit-testable without cameras.

CONVENTIONS (used consistently across the whole spatial subsystem)
-------------------------------------------------------------------
* World frame: right-handed, metres. It is the frame of the reference calibration board (see docs/spatial.md).
* Extrinsics are WORLD -> CAMERA (OpenCV rvec/tvec convention):      X_cam = R @ X_world + t
  The camera centre in world coordinates is                          C = -R.T @ t
  Camera axes: +x right, +y down, +z forward (a visible point has X_cam[2] > 0).
* Projection:  s * [u, v, 1]^T = K @ (R @ X_world + t)  followed by lens distortion on the normalized point.
* Pixel coordinates (u, v): u right, v down, OpenCV convention (the CENTRE of the top-left pixel is (0, 0)).
  MediaPipe/Phase 1 ``x_px = x_norm * image_width`` uses an EDGE origin, so  u_cv = x_px - 0.5  (see ``to_cv_pixels``).
* Distortion: OpenCV model, coefficients (k1, k2, p1, p2, k3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


class GeometryError(ValueError):
    """Raised for structurally invalid inputs (shape, non-finite values). Numerical degeneracy is returned, not raised."""


@dataclass(frozen=True)
class CameraModel:
    camera_id: str
    K: np.ndarray            # 3x3
    dist: np.ndarray         # (5,)
    R: np.ndarray            # 3x3 world->camera
    t: np.ndarray            # (3,)  world->camera
    width: int
    height: int

    def __post_init__(self):
        for name, arr, shape in (("K", self.K, (3, 3)), ("R", self.R, (3, 3)), ("t", self.t, (3,)), ("dist", self.dist, (5,))):
            if np.shape(arr) != shape or not np.all(np.isfinite(arr)):
                raise GeometryError(f"{self.camera_id}: {name} must be finite with shape {shape}")

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def rvec(self) -> np.ndarray:
        return cv2.Rodrigues(self.R)[0].reshape(3)

    @property
    def focal_px(self) -> float:
        return float((self.K[0, 0] + self.K[1, 1]) / 2.0)


def to_cv_pixels(pts_px: np.ndarray, subtract_half_pixel: bool = True) -> np.ndarray:
    """Phase-1 landmark pixels (edge origin) -> OpenCV pixels (centre origin)."""
    p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    return p - 0.5 if subtract_half_pixel else p.copy()


def depth(cam: CameraModel, X_world: np.ndarray) -> np.ndarray:
    """z-coordinate of world points in the camera frame (positive = in front of the camera)."""
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    return (X @ cam.R.T + cam.t)[:, 2]


def project(cam: CameraModel, X_world: np.ndarray, distort: bool = True) -> np.ndarray:
    """Project world points to OpenCV pixel coordinates (with lens distortion unless ``distort=False``).
    Points at or behind the camera plane give meaningless pixels: check ``depth`` first."""
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    dist = cam.dist if distort else np.zeros(5)
    px, _ = cv2.projectPoints(X, cam.rvec, cam.t.reshape(3, 1), cam.K, dist)
    return px.reshape(-1, 2)


def undistort_to_normalized(cam: CameraModel, pts_cv_px: np.ndarray) -> np.ndarray:
    """Distorted OpenCV pixels -> undistorted normalized image coordinates (x/z, y/z)."""
    p = np.asarray(pts_cv_px, dtype=np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-14)   # default (5 iterations) leaves ~1e-5 error
    return cv2.undistortPointsIter(p, cam.K, cam.dist, None, None, crit).reshape(-1, 2)


def pixel_ray_world(cam: CameraModel, pt_cv_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Viewing ray (origin = camera centre, unit direction) in the world frame for one pixel. NOT a 3D point."""
    n = undistort_to_normalized(cam, pt_cv_px)[0]
    d_cam = np.array([n[0], n[1], 1.0])
    d_world = cam.R.T @ d_cam
    return cam.center, d_world / np.linalg.norm(d_world)


@dataclass
class TriangulationResult:
    ok: bool
    point: np.ndarray | None = None        # world XYZ (metres) when ok
    reason: str | None = None              # machine-readable failure code
    homogeneous_ratio: float = float("nan")  # |w| / ||X_h||: ~0 means point at infinity
    depths: dict[str, float] = field(default_factory=dict)
    detail: str = ""


def triangulate_dlt(cams: list[CameraModel], pts_cv_px: np.ndarray, min_homogeneous_w: float = 1e-9) -> TriangulationResult:
    """Linear (DLT) N-view triangulation on UNDISTORTED NORMALIZED coordinates (well conditioned).

    Returns a failure result (never a fabricated point) for: <2 views, non-finite input/output, a point at infinity
    (rays parallel / no usable baseline). Depth positivity is checked separately by ``check_depths``.
    """
    if len(cams) < 2 or len(cams) != len(np.atleast_2d(pts_cv_px)):
        return TriangulationResult(False, reason="INSUFFICIENT_VIEWS", detail="need >= 2 matching camera/point pairs")
    pts = np.asarray(pts_cv_px, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return TriangulationResult(False, reason="NON_FINITE_INPUT", detail="observation coordinates are not finite")
    norm = np.vstack([undistort_to_normalized(c, p) for c, p in zip(cams, pts)])
    if not np.all(np.isfinite(norm)):
        return TriangulationResult(False, reason="NON_FINITE_INPUT", detail="undistortion produced non-finite values")
    rows = []
    for c, (x, y) in zip(cams, norm):
        Pn = np.hstack([c.R, c.t.reshape(3, 1)])            # normalized camera matrix [R|t]
        rows.append(x * Pn[2] - Pn[0])
        rows.append(y * Pn[2] - Pn[1])
    A = np.asarray(rows)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return TriangulationResult(False, reason="SVD_FAILED", detail="SVD did not converge")
    Xh = vt[-1]
    ratio = float(abs(Xh[3]) / max(np.linalg.norm(Xh), 1e-300))
    if not np.all(np.isfinite(Xh)) or ratio < min_homogeneous_w:
        return TriangulationResult(False, reason="POINT_AT_INFINITY", homogeneous_ratio=ratio,
                                   detail=f"homogeneous w ratio {ratio:.2e}: rays are (nearly) parallel")
    X = Xh[:3] / Xh[3]
    if not np.all(np.isfinite(X)):
        return TriangulationResult(False, reason="NON_FINITE_RESULT", homogeneous_ratio=ratio, detail="non-finite XYZ")
    return TriangulationResult(True, point=X, homogeneous_ratio=ratio)


def refine_gauss_newton(cams: list[CameraModel], pts_cv_px: np.ndarray, X0: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Minimise the (pixel-scaled) reprojection error in normalized coordinates. Never returns a worse point than X0."""
    pts = np.asarray(pts_cv_px, dtype=np.float64).reshape(-1, 2)
    meas = np.vstack([undistort_to_normalized(c, p) for c, p in zip(cams, pts)])
    f = np.array([c.focal_px for c in cams])

    def residual_and_jac(X):
        r, J = [], []
        for c, m, fi in zip(cams, meas, f):
            Xc = c.R @ X + c.t
            if Xc[2] <= 1e-9:
                return None, None
            pr = Xc[:2] / Xc[2]
            r.extend(fi * (pr - m))
            dproj = np.array([[1 / Xc[2], 0, -Xc[0] / Xc[2] ** 2], [0, 1 / Xc[2], -Xc[1] / Xc[2] ** 2]])
            J.append(fi * dproj @ c.R)
        return np.asarray(r), np.vstack(J)

    X = np.asarray(X0, dtype=np.float64).copy()
    r, J = residual_and_jac(X)
    if r is None:
        return X
    cost = float(r @ r)
    for _ in range(iterations):
        try:
            step = np.linalg.solve(J.T @ J, -J.T @ r)
        except np.linalg.LinAlgError:
            break
        Xn = X + step
        rn, Jn = residual_and_jac(Xn)
        if rn is None or float(rn @ rn) >= cost:
            break
        X, r, J, cost = Xn, rn, Jn, float(rn @ rn)
        if np.linalg.norm(step) < 1e-12:
            break
    return X


def check_depths(cams: list[CameraModel], X: np.ndarray, min_depth: float, max_depth: float) -> tuple[bool, dict[str, float], str | None]:
    """Reject points behind (or too close to / too far from) any contributing camera."""
    depths = {c.camera_id: float(depth(c, X)[0]) for c in cams}
    for cid, d in depths.items():
        if not np.isfinite(d):
            return False, depths, "NON_FINITE_RESULT"
        if d <= 0:
            return False, depths, "NEGATIVE_DEPTH"
        if d < min_depth or d > max_depth:
            return False, depths, "DEPTH_OUT_OF_RANGE"
    return True, depths, None


def reprojection_errors(cams: list[CameraModel], X: np.ndarray, pts_cv_px: np.ndarray) -> np.ndarray:
    """Per-camera pixel distance between the observation and the reprojected point (distortion included)."""
    pts = np.asarray(pts_cv_px, dtype=np.float64).reshape(-1, 2)
    return np.array([np.linalg.norm(project(c, X)[0] - p) for c, p in zip(cams, pts)])


def max_ray_angle_deg(cams: list[CameraModel], X: np.ndarray) -> float:
    """Largest pairwise angle at X between the rays towards the camera centres (triangulation baseline quality)."""
    dirs = []
    for c in cams:
        v = c.center - X
        n = np.linalg.norm(v)
        if n < 1e-12:
            return 0.0
        dirs.append(v / n)
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            best = max(best, float(np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1.0, 1.0)))))
    return best


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle of the relative rotation between two rotation matrices."""
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    R = np.asarray(R, dtype=np.float64)
    return R.shape == (3, 3) and bool(np.all(np.isfinite(R))) and np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol
