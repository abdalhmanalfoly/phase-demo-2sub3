"""Calibration artifacts, store, registry, and the image-based calibration pipeline (synthetic rendered boards)."""
import copy
import json

import cv2
import numpy as np
import pytest

from host.app.config import CalibrationQualityCfg
from host.app.spatial import calibration_tool as T
from host.app.spatial import geometry as G
from host.app.spatial import synthetic as S
from host.app.spatial.calibration_schema import CalibrationArtifact
from host.app.spatial.calibration_store import (CalibrationLoadError, CalibrationRegistry, CalibrationStore, camera_model,
                                                check_rig, load_calibration_file)

CAMS = S.rig_cameras()


def art(cid="phone_left", **kw):
    return S.artifact_from_camera(CAMS[cid], **kw)


def raw(a):
    return json.loads(a.model_dump_json())


# ------------------------------------------------------------------------------ schema
def test_artifact_roundtrip_and_required_contract_fields():
    a = art()
    b = CalibrationArtifact.model_validate(raw(a))
    assert b == a
    d = raw(a)
    for k in ["calibration_version", "calibration_id", "created_at", "image_width", "image_height", "target", "camera_matrix",
              "distortion", "extrinsics", "projection_matrix", "quality", "validity", "camera_id", "rig_id", "unit", "rig_calibration_id"]:
        assert k in d, k
    assert d["unit"] == "m" and d["target"]["unit"] == "m" and d["extrinsics"]["convention"] == "world_to_camera"
    assert d["extrinsics"]["camera_center_world"] == pytest.approx((-CAMS["phone_left"].R.T @ CAMS["phone_left"].t).tolist())


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["camera_matrix"][0].__setitem__(0, 0.0), "focal"),
    (lambda d: d.__setitem__("camera_matrix", [[600, 0, 320], [0, 600, 240], [0, 0, 0]]), "form"),
    (lambda d: d["camera_matrix"][0].__setitem__(2, 5000.0), "principal point"),
    (lambda d: d["camera_matrix"][1].__setitem__(1, 100.0), "fx/fy"),
    (lambda d: d.__setitem__("camera_matrix", [[1, 0, 320], [0, 1, 240], [0, 0, 1]][:2]), "shape"),
    (lambda d: d.__setitem__("distortion", [0.1, 0.2]), "shape"),
    (lambda d: d["extrinsics"]["rotation"][0].__setitem__(0, 2.0), "rotation"),
    (lambda d: d["extrinsics"]["translation"].__setitem__(0, 9.0), "inconsistent"),
    (lambda d: d["projection_matrix"][0].__setitem__(0, 1.0), "projection_matrix"),
    (lambda d: d.__setitem__("calibration_version", "9.9"), "unsupported"),
    (lambda d: d.__setitem__("unit", "mm"), "unit"),
    (lambda d: d.__setitem__("image_width", 0), "image_width"),
    (lambda d: d.__setitem__("created_at", "yesterday"), "created_at"),
    (lambda d: d.__setitem__("surprise", 1), "surprise"),
    (lambda d: d["extrinsics"].__setitem__("convention", "camera_to_world"), "convention"),
])
def test_invalid_artifacts_are_rejected(mutate, needle):
    d = raw(art())
    mutate(d)
    with pytest.raises(Exception) as e:
        CalibrationArtifact.model_validate(d)
    assert needle.lower() in str(e.value).lower(), str(e.value)[:300]


def test_singular_camera_matrix_rejected():
    d = raw(art())
    d["camera_matrix"] = [[600.0, 0, 320], [0, 1e-12, 240], [0, 0, 1]]
    with pytest.raises(Exception):
        CalibrationArtifact.model_validate(d)


def test_non_rotation_extrinsics_rejected():
    d = raw(art())
    d["extrinsics"]["rotation"] = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]          # reflection, det = -1
    with pytest.raises(Exception, match="rotation"):
        CalibrationArtifact.model_validate(d)


# ------------------------------------------------------------------------------ store
def test_save_load_history_and_atomicity(tmp_path):
    st = CalibrationStore(tmp_path, "default_rig")
    a1 = art(calibration_id="left-1")
    p = st.save_camera(a1)
    assert st.load_camera("phone_left") == a1
    a2 = art(calibration_id="left-2")
    st.save_camera(a2)
    assert st.load_camera("phone_left").calibration_id == "left-2"
    assert (tmp_path / "default_rig" / "history" / "phone_left__left-1.json").exists()      # nothing is lost on recalibration
    assert not list(tmp_path.rglob(".tmp_*")), "temp files must not be left behind"


def test_corrupt_missing_and_incompatible_files_are_never_silently_loaded(tmp_path):
    st = CalibrationStore(tmp_path, "default_rig")
    st.save_camera(art("phone_left"))
    st.save_camera(art("phone_right"))
    d = tmp_path / "default_rig"
    (d / "phone_left.json").write_text((d / "phone_left.json").read_text()[:200])          # truncated
    rep = st.load_all()
    assert "phone_right" in rep.artifacts and "phone_left" not in rep.artifacts
    assert rep.errors["phone_left"].startswith("CORRUPT")
    with pytest.raises(CalibrationLoadError) as e:
        st.load_camera("nobody")
    assert e.value.code == "NOT_FOUND"
    bad = raw(art("phone_right")); bad["calibration_version"] = "0.1"
    (d / "phone_right.json").write_text(json.dumps(bad))
    assert st.load_all().errors["phone_right"].startswith("UNSUPPORTED_VERSION")
    bad = raw(art("phone_right")); bad["camera_matrix"][0][0] = float("nan")
    (d / "phone_right.json").write_text(json.dumps(bad))                                     # NaN literal
    assert "phone_right" in st.load_all().errors
    (d / "phone_right.json").write_text("[1,2,3]")
    assert st.load_all().errors["phone_right"].startswith("CORRUPT")


def test_file_name_must_match_camera_and_rig(tmp_path):
    st = CalibrationStore(tmp_path, "default_rig")
    st.save_camera(art("phone_left"))
    (tmp_path / "default_rig" / "imposter.json").write_text(json.dumps(raw(art("phone_left"))))
    assert "does not match file name" in st.load_all().errors["imposter"]
    with pytest.raises(ValueError):
        CalibrationStore(tmp_path, "other_rig").save_camera(art("phone_left"))
    with pytest.raises(ValueError):
        st.camera_path("../escape")


def test_invalidation_persists_and_recalibration_restores(tmp_path):
    st = CalibrationStore(tmp_path, "default_rig")
    for c in CAMS:
        st.save_camera(art(c))
    reg = CalibrationRegistry(st, allow_synthetic=True)
    reg.bind("sensor_01", "phone_left")
    assert reg.resolve("sensor_01").problem is None
    gen = reg.generation
    reg.invalidate("phone_left", "tripod bumped")
    assert reg.generation > gen
    r = reg.resolve("sensor_01")
    assert r.problem == "INVALIDATED" and "tripod" in r.detail
    assert CalibrationRegistry(st, allow_synthetic=True).resolve("phone_left").problem == "INVALIDATED"    # survives restart
    st.save_camera(art("phone_left", calibration_id="left-recal"))                                          # operator recalibrated
    reg.reload()
    r = reg.resolve("sensor_01")
    assert r.problem is None and r.artifact.calibration_id == "left-recal" and r.artifact.validity.status == "valid"


# ------------------------------------------------------------------------------ registry
def test_binding_autobinding_and_problem_codes(tmp_path):
    st = CalibrationStore(tmp_path, "default_rig")
    for c in CAMS:
        st.save_camera(art(c))
    reg = CalibrationRegistry(st, allow_synthetic=True)
    assert reg.resolve("sensor_09").problem == "NOT_BOUND"
    assert reg.resolve("phone_right").problem is None                                    # id == camera_id: auto-bound
    reg.bind("sensor_01", "ghost_cam")
    assert reg.resolve("sensor_01").problem == "NOT_FOUND"
    reg.bind("sensor_01", "phone_left")
    assert reg.resolve("sensor_01", 640, 480).problem is None
    r = reg.resolve("sensor_01", 480, 640)                                               # rotated phone: orientation changed
    assert r.problem == "DIMENSION_MISMATCH" and r.model is None
    assert reg.resolve("sensor_01", 1280, 720).problem == "DIMENSION_MISMATCH"
    reg.unbind("sensor_01")
    assert reg.resolve("sensor_01").problem == "NOT_BOUND"


def test_synthetic_artifacts_are_refused_by_default(tmp_path):
    st = CalibrationStore(tmp_path, "default_rig")
    st.save_camera(art("phone_left"))
    assert CalibrationRegistry(st, allow_synthetic=False).resolve("phone_left").problem == "SYNTHETIC_REFUSED"
    assert CalibrationRegistry(st, allow_synthetic=True).resolve("phone_left").problem is None


def test_incompatible_rig_sessions_are_detected(tmp_path):
    st = CalibrationStore(tmp_path, "default_rig")
    st.save_camera(art("phone_left", rig_calibration_id="rig-A"))
    st.save_camera(art("phone_right", rig_calibration_id="rig-B"))
    reg = CalibrationRegistry(st, allow_synthetic=True)
    ok, bad = reg.compatible_ids(["phone_left", "phone_right"])
    assert len(ok) == 1 and len(bad) == 1 and set(bad.values()) == {"INCOMPATIBLE_RIG"}
    rep = check_rig(reg.artifacts, True)
    assert not rep["can_triangulate"] and "NOT READY" in rep["summary"]
    st.save_camera(art("phone_right", rig_calibration_id="rig-A")); reg.reload()
    assert check_rig(reg.artifacts, True)["can_triangulate"]
    ok, bad = reg.compatible_ids(["phone_left", "phone_right"])
    assert len(ok) == 2 and not bad


# ------------------------------------------------------------------------------ image-based pipeline
@pytest.fixture(scope="module")
def pipeline():
    return S.calibrate_synthetic_rig(noise=1.0, seed=0)


def test_pipeline_recovers_intrinsics_and_extrinsics_with_measured_error(pipeline):
    arts, truth = pipeline
    for cid, a in arts.items():
        m, tr = camera_model(a), truth[cid]
        assert a.synthetic and a.quality.intrinsic_rms_px < 0.3 and a.quality.extrinsic_rms_px < 0.5
        assert abs(m.K[0, 0] - 600) / 600 < 0.01 and abs(m.K[1, 1] - 600) / 600 < 0.01                  # focal within 1 %
        assert abs(m.K[0, 2] - tr.K[0, 2]) < 2.5 and abs(m.K[1, 2] - tr.K[1, 2]) < 2.5                 # principal point within 2.5 px
        assert G.rotation_angle_deg(m.R, tr.R) < 0.5
        assert np.linalg.norm(m.center - tr.center) < 0.015                                             # camera centre within 15 mm
        assert a.validity.status == "valid" and a.extrinsics.convention == "world_to_camera"
    chk = next(iter(arts.values())).quality.rig_check
    assert all(v["mean_mm"] < 1.5 and v["n_corners"] == 54 for v in chk["pairs"].values())
    assert len({a.rig_calibration_id for a in arts.values()}) == 1                                      # one shared rig session


def test_calibrated_cameras_triangulate_known_points_end_to_end(pipeline):
    arts, truth = pipeline
    cams = [camera_model(a) for a in arts.values()]
    tcams = [truth[c] for c in arts]
    errs = []
    for X in [[0.2, 0.05, -0.2], [0.3, 0.0, -0.3], [0.25, 0.1, 0.6], [0.3, 0.1, 1.5]]:
        X = np.array(X)
        pts = np.array([G.project(c, X)[0] for c in tcams])                # observed through the TRUE cameras
        errs.append(np.linalg.norm(G.triangulate_dlt(cams, pts).point - X))   # reconstructed with the CALIBRATED ones
    assert max(errs) < 0.02, errs                                          # calibration-induced error, synthetic


def test_board_to_world_transform_moves_the_frame_consistently():
    Rw = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]                         # flip: Z up instead of into the board
    tw = np.array([0.1, 0.2, 0.3])
    arts, truth = S.calibrate_synthetic_rig(noise=0.5, seed=2, board_to_world=(Rw, tw))
    for cid, a in arts.items():
        m, tr = camera_model(a), truth[cid]
        assert G.rotation_angle_deg(m.R, tr.R) < 0.5 and np.linalg.norm(m.center - tr.center) < 0.015


def test_distortion_is_recovered_from_corner_data():
    """Point-level path (renderer has no lens model): known K + distortion -> corners -> calibrateCamera."""
    cam = S.rig_cameras()["phone_left"]
    obj = T.object_points(S.COLS, S.ROWS, S.SQUARE)
    rng = np.random.default_rng(5)
    samples = []
    for i, (R, t) in enumerate(S.board_poses(18, seed=11)):
        uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, cam.K, cam.dist)
        samples.append(T.Sample(f"v{i}", uv.reshape(-1, 2) + rng.normal(0, 0.1, (len(obj), 2)), S.SIZE))
    a = T.calibrate_intrinsics(samples, S.SIZE, S.COLS, S.ROWS, S.SQUARE, "phone_left", CalibrationQualityCfg(min_coverage_fraction=0.1))
    k1, k2 = a.distortion[0], a.distortion[1]
    assert abs(k1 - cam.dist[0]) < 0.03 and abs(k2 - cam.dist[1]) < 0.12 and a.distortion[4] == 0.0    # k3 fixed
    assert abs(a.camera_matrix[0][0] - 600) < 6 and a.quality.intrinsic_rms_px < 0.2


def test_too_few_samples_and_outlier_rejection_are_reported():
    cam = S.rig_cameras()["phone_left"]
    obj = T.object_points(S.COLS, S.ROWS, S.SQUARE)
    rng = np.random.default_rng(8)
    good = []
    for i, (R, t) in enumerate(S.board_poses(14, seed=21)):
        uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, cam.K, cam.dist)
        good.append(T.Sample(f"g{i}", uv.reshape(-1, 2) + rng.normal(0, 0.1, (len(obj), 2)), S.SIZE))
    with pytest.raises(T.CalibrationError) as e:
        T.calibrate_intrinsics(good[:4], S.SIZE, S.COLS, S.ROWS, S.SQUARE, "x", CalibrationQualityCfg())
    assert e.value.code == "TOO_FEW_SAMPLES"
    # NOTE: a reversed corner order is just a board rotated 180 deg in-plane = a valid pose, so it is (correctly) NOT an
    # intrinsics outlier. A genuinely corrupt detection (geometry destroyed) must be rejected and reported:
    bad = T.Sample("corrupt", good[0].corners + rng.normal(0, 8.0, good[0].corners.shape), S.SIZE)
    a = T.calibrate_intrinsics(good + [bad], S.SIZE, S.COLS, S.ROWS, S.SQUARE, "phone_left", CalibrationQualityCfg(min_coverage_fraction=0.1))
    assert [r.name for r in a.quality.samples_rejected] == ["corrupt"] and a.quality.samples_rejected[0].error_px > 2
    assert a.quality.samples_used == 14


def test_image_import_rejects_unreadable_wrong_size_and_boardless_images(tmp_path):
    K = S.pinhole_K()
    R, t = S.board_poses(1, seed=4)[0]
    cv2.imwrite(str(tmp_path / "a_good.png"), S.render_board(K, R, t))
    cv2.imwrite(str(tmp_path / "b_blank.png"), np.full((480, 640), 128, np.uint8))
    cv2.imwrite(str(tmp_path / "c_small.png"), np.full((240, 320), 128, np.uint8))
    (tmp_path / "d_broken.png").write_bytes(b"not an image")
    (tmp_path / "notes.txt").write_text("ignored")
    samples, rejected, size = T.load_samples(tmp_path, S.COLS, S.ROWS)
    assert size == (640, 480) and [s.name for s in samples] == ["a_good.png"]
    reasons = {r.name: r.reason for r in rejected}
    assert "not found" in reasons["b_blank.png"] and "differs" in reasons["c_small.png"] and "unreadable" in reasons["d_broken.png"]


def test_rig_refuses_inconsistent_inputs():
    arts, _, cams_in = S.calibrate_synthetic_rig(noise=0.5, seed=3, return_inputs=True)
    with pytest.raises(T.CalibrationError) as e:
        T.calibrate_rig("default_rig", {"phone_left": cams_in["phone_left"]})
    assert e.value.code == "TOO_FEW_CAMERAS"
    with pytest.raises(T.CalibrationError) as e:
        T.calibrate_rig("default_rig", cams_in, board_to_world=(np.diag([1.0, 1, -1]), np.zeros(3)))   # reflection
    assert e.value.code == "BAD_BOARD_TO_WORLD"
    moved = {k: (v[0], [T.Sample(s.name, s.corners, (320, 240)) for s in v[1]]) for k, v in cams_in.items()}
    with pytest.raises(T.CalibrationError) as e:
        T.calibrate_rig("default_rig", moved)
    assert e.value.code == "RESOLUTION_MISMATCH"                                  # reference images at another resolution
    other = cams_in["phone_right"][0].model_copy(deep=True)
    other.target.square_size = 0.05
    with pytest.raises(T.CalibrationError) as e:
        T.calibrate_rig("default_rig", {"phone_left": cams_in["phone_left"], "phone_right": (other, cams_in["phone_right"][1])})
    assert e.value.code == "TARGET_MISMATCH"
    empty = {"phone_left": cams_in["phone_left"], "phone_right": (cams_in["phone_right"][0], [])}
    with pytest.raises(T.CalibrationError) as e:
        T.calibrate_rig("default_rig", empty)
    assert e.value.code == "NO_REFERENCE_IMAGES"


def test_inconsistent_reference_images_produce_a_warning():
    arts, _, cams_in = S.calibrate_synthetic_rig(noise=0.5, seed=4, return_inputs=True)
    intr, ref = cams_in["phone_left"]
    shifted = ref + [T.Sample("moved", ref[0].corners + 6.0, ref[0].image_size)]            # board/camera moved between images
    out = T.calibrate_rig("default_rig", {"phone_left": (intr, shifted), "phone_right": cams_in["phone_right"]},
                          qcfg=CalibrationQualityCfg(max_extrinsic_rms_px=50))
    assert any("REFERENCE_IMAGES_INCONSISTENT" in w for w in out["phone_left"].validity.warnings)


def test_accuracy_report_statistics(tmp_path):
    f = tmp_path / "fusion.jsonl"
    rng = np.random.default_rng(1)
    truth = np.array([0.2, 0.1, 0.5])
    lines = [json.dumps({"target_landmark": "wrist", "valid": True, "raw_world_xyz_m": (truth + rng.normal(0, 0.005, 3)).tolist()}) for _ in range(200)]
    lines += [json.dumps({"target_landmark": "wrist", "valid": False, "raw_world_xyz_m": None})] * 50
    f.write_text("\n".join(lines))
    rep = T.accuracy_report(f, truth.tolist())
    assert rep["samples"] == 200 and rep["published"] == 250 and abs(rep["valid_fraction"] - 0.8) < 1e-9
    assert 0.003 < rep["mean_error_m"] < 0.012 and rep["p95_error_m"] >= rep["median_error_m"]
