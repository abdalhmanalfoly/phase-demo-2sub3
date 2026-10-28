"""Deterministic synthetic scenes: known cameras, rendered checkerboard images, projected landmark observations.
Used by the test-suite, ``scripts/calibrate.py selftest/example``. Results obtained here say NOTHING about physical accuracy."""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as G
from .calibration_tool import object_points

SIZE = (640, 480)
COLS, ROWS, SQUARE = 9, 6, 0.04
DIST = np.array([-0.18, 0.06, 0.0008, -0.0004, 0.0])


def pinhole_K(size=SIZE, f=600.0) -> np.ndarray:
    return np.array([[f, 0, size[0] / 2.0 - 0.5 + 3.0], [0, f, size[1] / 2.0 - 0.5 - 2.0], [0, 0, 1.0]])


def look_at(center, target, up=(0.0, -1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at ``center`` looking at ``target`` (camera +y = image down; default 'up' is world -Y)."""
    c, tg = np.asarray(center, float), np.asarray(target, float)
    z = tg - c
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float) * -1.0, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                 # rows = camera axes expressed in world
    return R, -R @ c


def rig_cameras(dist=DIST, size=SIZE, f=600.0, left=(-0.45, -0.10, -1.1), right=(0.85, -0.10, -1.1), look=(0.2, 0.05, 0.0)) -> dict[str, G.CameraModel]:
    K = pinhole_K(size, f)
    out = {}
    for cid, c in (("phone_left", left), ("phone_right", right)):
        R, t = look_at(c, look)
        out[cid] = G.CameraModel(cid, K.copy(), np.asarray(dist, float), R, t, size[0], size[1])
    return out


def _texture(s=40):
    m = s
    w, h = (COLS + 1) * s + 2 * m, (ROWS + 1) * s + 2 * m
    tex = np.full((h, w), 255, np.uint8)
    for a in range(COLS + 1):
        for b in range(ROWS + 1):
            if (a + b) % 2 == 1:        # parity chosen so OpenCV returns corners top-left -> bottom-right (see docs/spatial.md)
                tex[m + b * s: m + (b + 1) * s, m + a * s: m + (a + 1) * s] = 0
    return tex, s, m


def render_board(K: np.ndarray, R_cb: np.ndarray, t_cb: np.ndarray, size=SIZE, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Render the board seen by a distortion-free pinhole camera (board->camera pose R_cb, t_cb). 2x supersampled."""
    tex, s, m = _texture()
    H_tb = np.array([[SQUARE / s, 0, -(m + s) * SQUARE / s], [0, SQUARE / s, -(m + s) * SQUARE / s], [0, 0, 1.0]])
    S = 2.0
    Ks = K.copy(); Ks[:2] *= S; Ks[0, 2] += 0.5 * (S - 1); Ks[1, 2] += 0.5 * (S - 1)   # cv centre-origin -> supersampled grid
    H_bi = Ks @ np.column_stack([R_cb[:, 0], R_cb[:, 1], t_cb])
    img = cv2.warpPerspective(tex, H_bi @ H_tb, (int(size[0] * S), int(size[1] * S)), flags=cv2.INTER_LINEAR, borderValue=140)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(np.float64) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def board_poses(n: int, seed: int = 1, size=SIZE, f=600.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Varied board poses (board->camera) that keep the whole board inside the image."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        ang = np.radians(rng.uniform(15, 40))
        axis = rng.normal(size=3); axis[2] *= 0.25; axis /= np.linalg.norm(axis)
        R = cv2.Rodrigues(axis * ang)[0]
        z = rng.uniform(0.75, 1.25)
        t_center = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.1, 0.1), z])
        board_c = object_points(COLS, ROWS, SQUARE).mean(0)
        t = t_center - R @ board_c
        pts = (object_points(COLS, ROWS, SQUARE) @ R.T + t)
        uv = pts[:, :2] / pts[:, 2:3] * f + np.array(size) / 2.0
        if (uv[:, 0].min() > 25 and uv[:, 0].max() < size[0] - 25 and uv[:, 1].min() > 25 and uv[:, 1].max() < size[1] - 25 and pts[:, 2].min() > 0.3):
            out.append((R, t))
    return out


def observation_entry(sensor_id: str, cam: G.CameraModel, X_world, seq: int, captured_at_ms: int, received_at_ms: int,
                      landmark: str = "wrist", hand: str = "right", noise_px: float = 0.0, rng=None, status="tracked") -> dict:
    """A Phase-1 'accepted observation' entry for a landmark at a known 3D point (edge-origin pixels, as the sensors send)."""
    if status == "no_hand":
        lms = []
    else:
        uv = G.project(cam, np.asarray(X_world, float))[0]
        if noise_px and rng is not None:
            uv = uv + rng.normal(0, noise_px, 2)
        uv = uv + 0.5                                        # centre-origin -> MediaPipe edge-origin pixels
        lms = [{"id": landmark, "x_px": float(uv[0]), "y_px": float(uv[1]), "confidence": None, "x_norm": None, "y_norm": None, "z_model": None}]
    return {"received_at_ms": received_at_ms, "estimated_age_ms": None, "protocol_version": "1.0", "message_type": "LANDMARK_OBSERVATION",
            "session_id": "sess_synth0001", "sensor_id": sensor_id, "sequence_number": seq, "frame_id": seq,
            "captured_at_ms": captured_at_ms, "image_width": cam.width, "image_height": cam.height, "status": status,
            "hand": hand if status == "tracked" else "unknown", "landmarks": lms,
            "quality": {"inference_ms": 10.0, "handedness_score": None, "flags": []}}


def calibrate_synthetic_rig(n_intrinsic_views: int = 16, noise: float = 1.0, seed: int = 0, synthetic_flag: bool = True,
                            board_to_world=None, return_inputs: bool = False):
    """Full pipeline on rendered images (distortion-free renderer): detect -> intrinsics -> shared-frame extrinsics.
    Returns (artifacts, truth_cameras_in_world, report). ``truth`` uses the SAME world frame as the artifacts."""
    from ..config import CalibrationQualityCfg
    from .calibration_tool import Sample, calibrate_intrinsics, calibrate_rig, detect_corners
    truth_board = rig_cameras(dist=np.zeros(5))                      # board frame == world frame
    q = CalibrationQualityCfg(min_coverage_fraction=0.2)
    cams_in: dict = {}
    for k, (cid, cam) in enumerate(truth_board.items()):
        samples = []
        for i, (R, t) in enumerate(board_poses(n_intrinsic_views, seed=seed * 100 + k + 1)):
            c = detect_corners(render_board(cam.K, R, t, noise=noise, seed=seed * 1000 + k * 100 + i), COLS, ROWS)
            assert c is not None, "synthetic board not detected"
            samples.append(Sample(f"{cid}_{i}.png", c, SIZE))
        intr = calibrate_intrinsics(samples, SIZE, COLS, ROWS, SQUARE, cid, q, synthetic=synthetic_flag, notes="SYNTHETIC test data")
        ref = []
        for j in range(3):                                           # a few static reference images (independent noise)
            img = render_board(cam.K, cam.R, cam.t, noise=noise, seed=seed * 7000 + k * 10 + j)
            ref.append(Sample(f"{cid}_ref{j}.png", detect_corners(img, COLS, ROWS), SIZE))
        cams_in[cid] = (intr, ref)
    arts = calibrate_rig("default_rig", cams_in, board_to_world=board_to_world, qcfg=q, synthetic=synthetic_flag,
                         notes="SYNTHETIC example - generated from rendered images, not a real camera")
    Rw, tw = (np.eye(3), np.zeros(3)) if board_to_world is None else board_to_world
    truth = {}
    for cid, cam in truth_board.items():
        R = cam.R @ Rw.T
        truth[cid] = G.CameraModel(cid, cam.K, np.zeros(5), R, cam.t - R @ tw, cam.width, cam.height)
    return (arts, truth, cams_in) if return_inputs else (arts, truth)


def artifact_from_camera(cam: G.CameraModel, rig_calibration_id: str = "rig-synth-1", rig_id: str = "default_rig",
                         world_frame_id: str = "board_frame", synthetic: bool = True, calibration_id: str | None = None,
                         created_at: str = "2026-10-16T00:00:00+00:00"):
    """Exact (non-image) calibration artifact for a known camera: lets engine tests be fast and deterministic."""
    from .calibration_schema import CALIBRATION_VERSION, CalibrationArtifact, Quality, Target, Validity, make_extrinsics
    ext, _ = make_extrinsics(cam.R, cam.t)
    return CalibrationArtifact(
        calibration_version=CALIBRATION_VERSION, created_at=created_at, synthetic=synthetic, notes="SYNTHETIC exact camera",
        image_width=cam.width, image_height=cam.height,
        target=Target(type="checkerboard", inner_corners_cols=COLS, inner_corners_rows=ROWS, square_size=SQUARE, unit="m"),
        camera_matrix=cam.K.tolist(), distortion=cam.dist.tolist(),
        quality=Quality(intrinsic_rms_px=0.0, extrinsic_rms_px=0.0, samples_used=0), validity=Validity(),
        calibration_id=calibration_id or f"{cam.camera_id}-synth", camera_id=cam.camera_id, rig_id=rig_id,
        rig_calibration_id=rig_calibration_id, world_frame_id=world_frame_id, unit="m", extrinsics=ext,
        projection_matrix=cam.P.tolist())


def accuracy_table(arts, truth, noise_levels=(0.0, 0.5, 1.0), depths=(-0.2, 0.6, 1.5, 3.0), n=300, seed=0) -> list[dict]:
    """End-to-end synthetic accuracy: points seen through the TRUE cameras (with pixel noise), reconstructed with the
    CALIBRATED cameras. Includes calibration error + detector noise. SYNTHETIC: says nothing about physical accuracy."""
    from .calibration_store import camera_model
    cal = [camera_model(a) for a in arts.values()]
    tru = [truth[c] for c in arts]
    rows = []
    for noise in noise_levels:
        for z in depths:
            rng = np.random.default_rng(seed)
            errs = []
            for _ in range(n):
                X = np.array([0.2 + rng.uniform(-0.08, 0.08), 0.05 + rng.uniform(-0.05, 0.05), z + rng.uniform(-0.05, 0.05)])
                pts = np.array([G.project(c, X)[0] for c in tru]) + (rng.normal(0, noise, (2, 2)) if noise else 0.0)
                r = G.triangulate_dlt(cal, pts)
                if r.ok:
                    errs.append(float(np.linalg.norm(r.point - X)) * 1000.0)
            e = np.array(errs)
            rows.append({"noise_px": noise, "z_m": z, "mean_depth_from_cameras_m": float(np.mean([G.depth(c, np.array([0.2, 0.05, z]))[0] for c in tru])),
                         "mean_mm": float(e.mean()), "p95_mm": float(np.percentile(e, 95)), "max_mm": float(e.max()), "n": len(e)})
    return rows
