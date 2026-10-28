"""Calibration workflow (OpenCV): board detection -> intrinsics -> shared-frame extrinsics -> rig consistency check.

PROCEDURE (reproducible; commands in docs/spatial.md)
1. INTRINSICS, per physical camera and per streaming resolution: >= 10 images of the printed checkerboard at varied
   positions, distances and tilts, captured at EXACTLY the resolution/orientation the sensor streams.
   Output: K and distortion (5 coefficients) + quality (per-image error, rejected images, coverage, tilt spread).
2. SHARED WORLD FRAME via a static reference board: put the board in the workspace and leave it (and the cameras) untouched.
   Every camera captures it. The board DEFINES the world frame (origin = first inner corner, X along the corner columns,
   Y along the rows, Z = X x Y pointing into the board), in metres. solvePnP of each camera against the same board gives
   each camera's world->camera pose, so all cameras share one frame without any camera needing to see another.
   An optional ``board_to_world`` rigid transform re-expresses that frame (e.g. Z up, origin at floor centre).
3. CROSS-CHECK: triangulate the detected board corners from every camera pair and compare with the known board geometry.
   This checks internal consistency (detector, intrinsics, extrinsics, triangulation); it does NOT replace measuring a known
   physical point (see ``accuracy_report``).
Moving a camera, changing its zoom/resolution/orientation, or replacing it invalidates its extrinsics (and possibly intrinsics).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from ..config import CalibrationQualityCfg
from . import geometry as G
from .calibration_schema import (CALIBRATION_VERSION, CalibrationArtifact, IntrinsicsArtifact, Quality, RejectedSample,
                                 Target, Validity, make_extrinsics)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class CalibrationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


@dataclass
class Sample:
    name: str
    corners: np.ndarray                  # (N, 2) float, OpenCV pixel convention, board order
    image_size: tuple[int, int]          # (width, height)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """Board corner coordinates in the BOARD/WORLD frame (metres, Z=0): x fastest, matching OpenCV's detection order."""
    pts = np.zeros((rows * cols, 3), np.float64)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return pts


def detect_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Detect inner corners with sub-pixel accuracy; returns (N,2) or None."""
    ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ok and c is not None:
        return c.reshape(-1, 2).astype(np.float64)
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit).reshape(-1, 2).astype(np.float64)


def load_samples(images_dir: Path | str, cols: int, rows: int) -> tuple[list[Sample], list[RejectedSample], tuple[int, int] | None]:
    """Import images from a folder. Rejects (and reports) unreadable files, boards not found, and size mismatches."""
    samples, rejected, size = [], [], None
    for p in sorted(Path(images_dir).iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rejected.append(RejectedSample(name=p.name, reason="unreadable image"))
            continue
        h, w = img.shape
        if size is None:
            size = (w, h)
        if (w, h) != size:
            rejected.append(RejectedSample(name=p.name, reason=f"image size {w}x{h} differs from {size[0]}x{size[1]}"))
            continue
        c = detect_corners(img, cols, rows)
        if c is None:
            rejected.append(RejectedSample(name=p.name, reason="checkerboard not found"))
            continue
        samples.append(Sample(p.name, c, (w, h)))
    return samples, rejected, size


# --------------------------------------------------------------------------------------------- intrinsics
def _coverage_fraction(samples: list[Sample], size: tuple[int, int]) -> float:
    w, h = size
    s = 0.25
    mask = np.zeros((int(h * s), int(w * s)), np.uint8)
    for smp in samples:
        hull = cv2.convexHull((smp.corners * s).astype(np.float32).reshape(-1, 1, 2)).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 255)
    return float(mask.mean() / 255.0)


def _per_view_rms(obj, samples, K, dist, rvecs, tvecs) -> list[float]:
    out = []
    for smp, rv, tv in zip(samples, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        out.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - smp.corners) ** 2, axis=1)))))
    return out


def calibrate_intrinsics(samples: list[Sample], size: tuple[int, int], cols: int, rows: int, square_m: float,
                         camera_id: str, qcfg: CalibrationQualityCfg, rejected: list[RejectedSample] | None = None,
                         synthetic: bool = False, notes: str | None = None) -> IntrinsicsArtifact:
    rejected = list(rejected or [])
    obj = object_points(cols, rows, square_m)
    flags = cv2.CALIB_FIX_K3 if qcfg.fix_k3 else 0
    used = list(samples)
    for _ in range(12):
        if len(used) < qcfg.min_samples:
            raise CalibrationError("TOO_FEW_SAMPLES", f"{len(used)} usable images after rejection; need >= {qcfg.min_samples}")
        try:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj.astype(np.float32)] * len(used),
                                                             [s.corners.astype(np.float32).reshape(-1, 1, 2) for s in used],
                                                             size, None, None, flags=flags)
        except cv2.error as exc:
            raise CalibrationError("OPENCV_FAILED", str(exc).splitlines()[0][:200]) from None
        per = _per_view_rms(obj, used, K, dist, rvecs, tvecs)
        worst = int(np.argmax(per))
        if per[worst] <= qcfg.max_sample_error_px:
            break
        rejected.append(RejectedSample(name=used[worst].name, reason=f"reprojection error above {qcfg.max_sample_error_px} px",
                                       error_px=per[worst]))
        del used[worst]
    else:
        raise CalibrationError("NO_CONVERGENCE", "could not reach acceptable per-image error by removing outliers")
    if rms > qcfg.max_intrinsic_rms_px:
        raise CalibrationError("RMS_TOO_HIGH", f"intrinsic RMS {rms:.3f} px exceeds {qcfg.max_intrinsic_rms_px} px")
    cov = _coverage_fraction(used, size)
    tilts = [float(np.degrees(np.arccos(np.clip(abs(cv2.Rodrigues(rv)[0][2, 2]), 0, 1)))) for rv in rvecs]
    spread = float(max(tilts) - min(tilts))
    warnings = []
    if cov < qcfg.min_coverage_fraction:
        warnings.append(f"LOW_COVERAGE: board covered {cov:.0%} of the image (< {qcfg.min_coverage_fraction:.0%}); distortion near the edges is poorly constrained")
    if spread < qcfg.min_tilt_spread_deg:
        warnings.append(f"LOW_TILT_VARIETY: board tilt spread {spread:.1f} deg (< {qcfg.min_tilt_spread_deg} deg); focal length may be inaccurate")
    d = np.zeros(5)
    d[: min(5, dist.size)] = dist.reshape(-1)[:5]
    try:
        return IntrinsicsArtifact(
            calibration_version=CALIBRATION_VERSION, created_at=_now_iso(), synthetic=synthetic, notes=notes,
            image_width=size[0], image_height=size[1], target=Target(type="checkerboard", inner_corners_cols=cols,
                                                                  inner_corners_rows=rows, square_size=square_m, unit="m"),
            camera_matrix=K.tolist(), distortion=d.tolist(),
            quality=Quality(intrinsic_rms_px=float(rms), samples_used=len(used), samples_rejected=rejected,
                            per_sample_rms_px={s.name: v for s, v in zip(used, per)}, coverage_fraction=cov, tilt_spread_deg=spread),
            validity=Validity(status="valid", warnings=warnings), intrinsics_id=f"{camera_id}-{_stamp()}", camera_id=camera_id)
    except ValidationError as exc:
        raise CalibrationError("IMPLAUSIBLE_RESULT", str(exc.errors()[0]["msg"])) from None


# --------------------------------------------------------------------------------------------- extrinsics
def _pnp_planar(obj, corners, K, dist) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    """Board pose (board->camera). Returns (rvec, tvec, rms_px, ambiguity_ratio): ratio = 2nd-best/best error (close to 1 = ambiguous)."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj.astype(np.float64), corners.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE)
    if n == 0:
        raise CalibrationError("PNP_FAILED", "solvePnP found no pose")
    order = np.argsort(np.asarray(errs).reshape(-1))
    i = int(order[0])
    rv, tv = cv2.solvePnPRefineLM(obj, corners.reshape(-1, 1, 2), K, dist, rvecs[i], tvecs[i])
    ratio = float(np.asarray(errs).reshape(-1)[order[1]] / max(np.asarray(errs).reshape(-1)[i], 1e-12)) if n > 1 else None
    proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    return rv.reshape(3), tv.reshape(3), rms, ratio


def calibrate_rig(rig_id: str, cameras: dict[str, tuple[IntrinsicsArtifact, list[Sample]]], world_frame_id: str = "board_frame",
                  board_to_world: tuple[np.ndarray, np.ndarray] | None = None, qcfg: CalibrationQualityCfg | None = None,
                  synthetic: bool = False, notes: str | None = None) -> dict[str, CalibrationArtifact]:
    """Establish the shared world frame from images of ONE static reference board seen by every camera."""
    qcfg = qcfg or CalibrationQualityCfg()
    if len(cameras) < 2:
        raise CalibrationError("TOO_FEW_CAMERAS", "a rig needs at least 2 cameras")
    tgt = {c: i.target for c, (i, _) in cameras.items()}
    first = next(iter(tgt.values()))
    if any(t != first for t in tgt.values()):
        raise CalibrationError("TARGET_MISMATCH", "all cameras must have been calibrated with the same board definition")
    obj_board = object_points(first.inner_corners_cols, first.inner_corners_rows, first.square_size)
    Rw, tw = (np.eye(3), np.zeros(3)) if board_to_world is None else (np.asarray(board_to_world[0], float), np.asarray(board_to_world[1], float))
    if not G.is_rotation(Rw, 1e-6):
        raise CalibrationError("BAD_BOARD_TO_WORLD", "board_to_world rotation is not a proper rotation")
    rig_calib_id = f"rig-{_stamp()}"
    arts: dict[str, CalibrationArtifact] = {}
    mean_corners: dict[str, np.ndarray] = {}
    models: dict[str, G.CameraModel] = {}
    for cid, (intr, samples) in cameras.items():
        if not samples:
            raise CalibrationError("NO_REFERENCE_IMAGES", f"{cid}: no reference-board images with a detected board")
        for s in samples:
            if s.image_size != (intr.image_width, intr.image_height):
                raise CalibrationError("RESOLUTION_MISMATCH", f"{cid}: reference image {s.image_size} != intrinsics {intr.image_width}x{intr.image_height}")
        stack = np.stack([s.corners for s in samples])
        mc = stack.mean(axis=0)
        spread = float(np.sqrt(((stack - mc) ** 2).sum(-1)).max())
        K, dist = np.array(intr.camera_matrix, float), np.array(intr.distortion, float)
        rv, tv, rms, ratio = _pnp_planar(obj_board, mc, K, dist)
        if rms > qcfg.max_extrinsic_rms_px:
            raise CalibrationError("EXTRINSIC_RMS_TOO_HIGH", f"{cid}: board pose RMS {rms:.2f} px exceeds {qcfg.max_extrinsic_rms_px} px")
        R_cb = cv2.Rodrigues(rv)[0]
        if (R_cb @ obj_board.mean(0) + tv)[2] <= 0:
            raise CalibrationError("BOARD_BEHIND_CAMERA", f"{cid}: board pose is behind the camera (pose flip); retake with a tilted board")
        R_cw = R_cb @ Rw.T                                   # world->camera: X_cam = R_cb R_w^T (X_world - t_w) + t_cb
        t_cw = tv - R_cw @ tw
        warns = list(intr.validity.warnings)
        if spread > 1.0:
            warns.append(f"REFERENCE_IMAGES_INCONSISTENT: corners moved up to {spread:.2f} px between reference images (board or camera moved?)")
        if ratio is not None and ratio < 1.5:
            warns.append(f"POSE_AMBIGUITY: planar pose ambiguity (alt/best error {ratio:.2f}); tilt the reference board 20-45 deg")
        ext, _ = make_extrinsics(R_cw, t_cw)
        P = (K @ np.hstack([R_cw, t_cw.reshape(3, 1)])).tolist()
        arts[cid] = CalibrationArtifact(
            calibration_version=CALIBRATION_VERSION, created_at=_now_iso(), synthetic=synthetic, notes=notes,
            image_width=intr.image_width, image_height=intr.image_height, target=intr.target,
            camera_matrix=intr.camera_matrix, distortion=intr.distortion,
            quality=Quality(intrinsic_rms_px=intr.quality.intrinsic_rms_px, extrinsic_rms_px=rms, samples_used=len(samples),
                            samples_rejected=intr.quality.samples_rejected, coverage_fraction=intr.quality.coverage_fraction,
                            tilt_spread_deg=intr.quality.tilt_spread_deg),
            validity=Validity(status="valid", warnings=warns), calibration_id=f"{cid}-{_stamp()}", camera_id=cid, rig_id=rig_id,
            rig_calibration_id=rig_calib_id, world_frame_id=world_frame_id, unit="m", intrinsics_id=intr.intrinsics_id,
            extrinsics=ext, projection_matrix=P)
        mean_corners[cid] = mc
        models[cid] = G.CameraModel(cid, K, dist, R_cw, t_cw, intr.image_width, intr.image_height)
    # ---- cross-check: triangulate board corners from every pair, compare with the known (world-frame) board points
    obj_world = obj_board @ Rw.T + tw
    pairs = {}
    for a, b in itertools.combinations(models, 2):
        errs = []
        for i in range(len(obj_world)):
            r = G.triangulate_dlt([models[a], models[b]], np.stack([mean_corners[a][i], mean_corners[b][i]]))
            errs.append(np.linalg.norm(r.point - obj_world[i]) if r.ok else np.inf)
        e = np.array(errs) * 1000.0
        pairs[f"{a}+{b}"] = {"mean_mm": float(np.mean(e)), "max_mm": float(np.max(e)), "n_corners": int(len(e)),
                             "ray_angle_deg": float(G.max_ray_angle_deg([models[a], models[b]], obj_world.mean(0)))}
    note = "internal consistency of detector+intrinsics+extrinsics+triangulation on the reference board; NOT a physical accuracy claim"
    for a in arts.values():
        a.quality.rig_check = {"pairs": pairs, "note": note}
    return arts


# --------------------------------------------------------------------------------------------- accuracy
def accuracy_report(fusion_jsonl: Path | str, truth_xyz_m: list[float], target: str = "wrist") -> dict:
    """Error statistics of valid fused estimates against a physically measured point (the ground-truth hardware test)."""
    truth = np.asarray(truth_xyz_m, float)
    xs, total = [], 0
    for line in Path(fusion_jsonl).read_text().splitlines():
        r = json.loads(line)
        if r.get("target_landmark") != target:
            continue
        total += 1
        if r.get("valid") and r.get("raw_world_xyz_m"):
            xs.append(r["raw_world_xyz_m"])
    if not xs:
        return {"samples": 0, "published": total, "valid_fraction": 0.0, "note": "no valid estimates"}
    X = np.array(xs)
    err = np.linalg.norm(X - truth, axis=1)
    return {"samples": len(X), "published": total, "valid_fraction": len(X) / total,
            "mean_error_m": float(err.mean()), "median_error_m": float(np.median(err)), "p95_error_m": float(np.percentile(err, 95)),
            "max_error_m": float(err.max()), "repeatability_std_m": float(np.sqrt(np.trace(np.cov(X.T)))),
            "bias_xyz_m": (X.mean(0) - truth).tolist(), "unit": "m"}
