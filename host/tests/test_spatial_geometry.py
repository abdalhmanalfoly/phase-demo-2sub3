"""Pure geometry: projection, triangulation, degeneracies. Synthetic = proves the maths, not physical accuracy."""
import cv2
import numpy as np
import pytest

from host.app.spatial import geometry as G
from host.app.spatial import synthetic as S

CAMS = S.rig_cameras()                       # with lens distortion
CAMS_ND = S.rig_cameras(dist=np.zeros(5))
POINTS = np.array([[0.2, 0.05, -0.2], [0.3, 0.0, -0.3], [0.2, 0.1, 0.0], [0.25, 0.1, 0.6], [0.3, 0.1, 1.5], [0.2, 0.0, 3.0]])


def tri(cams, X, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.array([G.project(c, X)[0] for c in cams]) + (rng.normal(0, noise, (len(cams), 2)) if noise else 0)
    return G.triangulate_dlt(cams, pts)


def test_known_points_project_inside_both_images():
    for X in POINTS:
        for c in CAMS.values():
            uv = G.project(c, X)[0]
            assert G.depth(c, X)[0] > 0 and 0 <= uv[0] < c.width and 0 <= uv[1] < c.height, (X, c.camera_id, uv)


@pytest.mark.parametrize("cams_name", ["distorted", "undistorted"])
def test_triangulation_recovers_points_at_many_depths(cams_name):
    cams = list((CAMS if cams_name == "distorted" else CAMS_ND).values())
    worst = 0.0
    for X in POINTS:
        r = tri(cams, X)
        assert r.ok, r.reason
        worst = max(worst, np.linalg.norm(r.point - X))
    assert worst < 1e-9, worst                            # noise-free: limited only by float precision


def test_matches_opencv_triangulatePoints_on_undistorted_views():
    a, b = CAMS_ND["phone_left"], CAMS_ND["phone_right"]
    X = POINTS[1]
    pa, pb = G.project(a, X)[0], G.project(b, X)[0]
    h = cv2.triangulatePoints(a.P, b.P, pa.reshape(2, 1), pb.reshape(2, 1))
    cv_pt = (h[:3] / h[3]).ravel()
    mine = G.triangulate_dlt([a, b], np.array([pa, pb])).point
    assert np.linalg.norm(cv_pt - mine) < 1e-9


def test_three_views_and_refinement_never_worse():
    third = G.CameraModel("c", CAMS["phone_left"].K, CAMS["phone_left"].dist, *S.look_at((0.2, -0.9, -1.0), (0.2, 0.05, 0)), 640, 480)
    cams = [*CAMS.values(), third]
    X = np.array([0.2, 0.05, -0.3])
    pts = np.array([G.project(c, X)[0] for c in cams]) + np.random.default_rng(3).normal(0, 0.7, (3, 2))
    r = G.triangulate_dlt(cams, pts)
    X_ref = G.refine_gauss_newton(cams, pts, r.point, 5)
    assert np.mean(G.reprojection_errors(cams, X_ref, pts) ** 2) <= np.mean(G.reprojection_errors(cams, r.point, pts) ** 2) + 1e-12


def test_noise_sensitivity_is_bounded_and_scales_with_depth():
    """MEASURED (synthetic): error for 0.5 px pixel noise, 60 deg rig - documented in docs/spatial.md."""
    cams = list(CAMS.values())
    near, far = [], []
    for seed in range(200):
        near.append(np.linalg.norm(tri(cams, np.array([0.2, 0.05, -0.2]), 0.5, seed).point - [0.2, 0.05, -0.2]))
        far.append(np.linalg.norm(tri(cams, np.array([0.2, 0.0, 3.0]), 0.5, seed).point - [0.2, 0.0, 3.0]))
    assert np.sqrt(np.mean(np.square(near))) < 0.005          # ~1 m from cameras: millimetres
    assert np.sqrt(np.mean(np.square(far))) > np.sqrt(np.mean(np.square(near)))   # error grows with distance


def test_half_pixel_convention():
    assert np.allclose(G.to_cv_pixels(np.array([[10.5, 20.5]])), [[10.0, 20.0]])
    assert np.allclose(G.to_cv_pixels(np.array([[10.5, 20.5]]), subtract_half_pixel=False), [[10.5, 20.5]])


def test_world_to_camera_convention():
    c = CAMS["phone_left"]
    X = np.array([0.3, 0.1, 0.4])
    assert np.allclose(G.depth(c, X), (c.R @ X + c.t)[2])
    assert np.allclose(c.center, -c.R.T @ c.t) and np.allclose(c.R @ c.center + c.t, 0)
    assert G.is_rotation(c.R)


def test_pixel_ray_passes_through_the_point_and_is_not_a_point():
    c = CAMS["phone_left"]
    X = POINTS[0]
    o, d = G.pixel_ray_world(c, G.project(c, X)[0])
    assert np.isclose(np.linalg.norm(d), 1) and np.linalg.norm(np.cross(X - o, d)) < 1e-9


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_observation_rejected(bad):
    cams = list(CAMS.values())
    r = G.triangulate_dlt(cams, np.array([[100.0, bad], [200.0, 200.0]]))
    assert not r.ok and r.reason == "NON_FINITE_INPUT" and r.point is None


def test_fewer_than_two_views_fails():
    r = G.triangulate_dlt([CAMS["phone_left"]], np.array([[100.0, 100.0]]))
    assert not r.ok and r.reason == "INSUFFICIENT_VIEWS"


def test_parallel_rays_are_point_at_infinity_not_a_fabricated_point():
    a = CAMS_ND["phone_left"]
    b = G.CameraModel("b", a.K, a.dist, a.R, a.t + a.R @ np.array([0.5, 0, 0]), 640, 480)   # same orientation, shifted
    r = G.triangulate_dlt([a, b], np.array([[320.0, 240.0], [320.0, 240.0]]))                 # zero disparity
    assert not r.ok and r.reason == "POINT_AT_INFINITY" and r.homogeneous_ratio < 1e-9


def test_point_behind_the_cameras_is_never_accepted():
    """Pixels of a point behind the cameras (projectPoints mirrors it) must not yield a trusted 3D point."""
    cams = list(CAMS_ND.values())
    behind = np.array([0.2, 0.05, -3.0])                  # cameras sit at z=-1.1 looking towards +z: this is behind both
    assert all(G.depth(c, behind)[0] < 0 for c in cams)
    pts = np.array([G.project(c, behind)[0] for c in cams])
    r = G.triangulate_dlt(cams, pts)
    accepted = r.ok and G.check_depths(cams, r.point, 0.05, 20.0)[0] and \
        G.reprojection_errors(cams, r.point, pts).max() < 1.0
    assert not accepted


def test_check_depths_flags_negative_depth_explicitly():
    ok, depths, why = G.check_depths(list(CAMS.values()), np.array([0.0, 0.0, -5.0]), 0.05, 20.0)
    assert not ok and why == "NEGATIVE_DEPTH"
    ok, _, why = G.check_depths(list(CAMS.values()), np.array([0.2, 0.0, 100.0]), 0.05, 20.0)
    assert not ok and why == "DEPTH_OUT_OF_RANGE"


def test_reprojection_error_detects_inconsistent_view():
    cams = list(CAMS.values())
    X = POINTS[0]
    pts = np.array([G.project(c, X)[0] for c in cams])
    assert G.reprojection_errors(cams, X, pts).max() < 1e-9
    pts[1] += [25.0, -10.0]
    r = G.triangulate_dlt(cams, pts)
    assert G.reprojection_errors(cams, r.point, pts).max() > 5.0


def test_ray_angle_and_rotation_helpers():
    cams = list(CAMS.values())
    assert 30 < G.max_ray_angle_deg(cams, np.array([0.2, 0.05, 0.0])) < 90
    R = cv2.Rodrigues(np.array([0, 0.1, 0]))[0]
    assert abs(G.rotation_angle_deg(np.eye(3), R) - np.degrees(0.1)) < 1e-9
    assert not G.is_rotation(np.diag([1, 1, -1.0])) and not G.is_rotation(np.full((3, 3), np.nan))


def test_invalid_camera_models_are_refused():
    K = CAMS["phone_left"].K
    with pytest.raises(G.GeometryError):
        G.CameraModel("x", K, np.zeros(5), np.eye(3), np.array([np.nan, 0, 0]), 640, 480)
    with pytest.raises(G.GeometryError):
        G.CameraModel("x", K[:2], np.zeros(5), np.eye(3), np.zeros(3), 640, 480)
