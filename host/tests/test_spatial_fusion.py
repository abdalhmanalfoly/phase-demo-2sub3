"""Fusion engine: every status in the decision table, plus the 'never falsely trusted' guarantees."""
import json
from pathlib import Path

import numpy as np
import pytest

from host.app.config import load_settings
from host.app.spatial import geometry as G
from host.app.spatial import synthetic as S
from host.app.spatial.calibration_store import CalibrationRegistry, CalibrationStore
from host.app.spatial.fusion import FusionEngine
from host.app.spatial.results import FusionResult, FusionStatus as FS
from .conftest import FakeClock, T0

CAMS = S.rig_cameras()
LEFT, RIGHT = CAMS["phone_left"], CAMS["phone_right"]
OVERHEAD = G.CameraModel("overhead", LEFT.K, LEFT.dist, *S.look_at((0.2, -0.9, -1.0), (0.2, 0.05, 0)), 640, 480)
X = np.array([0.25, 0.1, 0.6])
A, B = "phone_left", "phone_right"                      # sensor ids == camera ids -> auto-bound


class Rig:
    def __init__(self, tmp_path, cams=(LEFT, RIGHT), rig_ids=None, cfg_over=None):
        self.clock = FakeClock()
        self.store = CalibrationStore(tmp_path, "default_rig")
        for c in cams:
            self.store.save_camera(S.artifact_from_camera(c, rig_calibration_id=(rig_ids or {}).get(c.camera_id, "rig-1")))
        self.registry = CalibrationRegistry(self.store, allow_synthetic=True)
        cfg = load_settings(env={}).fusion
        cfg.smoothing.enabled = False
        for k, v in (cfg_over or {}).items():
            sec, name = k.split(".")
            setattr(getattr(cfg, sec), name, v)
        self.cfg = cfg
        self.engine = FusionEngine("sess_test0001", cfg, self.registry, self.clock)
        self.seq = {}
        self.cams = {c.camera_id: c for c in cams}

    def feed(self, sid, X=X, behind_ms=0, hand="right", status="tracked", noise=0.0, offset=0, mutate=None, cam=None):
        n = self.seq[sid] = self.seq.get(sid, -1) + 1
        now = self.clock.now_ms()
        e = S.observation_entry(sid, cam or self.cams[sid], X, n, now - behind_ms - offset, now, hand=hand, status=status)
        if mutate:
            mutate(e)
        return self.engine.ingest(e, offset)

    def frame(self, X=X, sids=(A, B), **kw):
        out = []
        for sid in sids:
            out += self.feed(sid, X, **kw)
        return out

    def advance(self, ms):
        self.clock.advance(ms)


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def last(res):
    assert res, "expected a published result"
    return res[-1]


# ------------------------------------------------------------------------------ valid multi-view
def test_valid_two_view_estimate_matches_ground_truth_at_many_depths(rig):
    worst = 0.0
    for i, P in enumerate([[0.2, 0.05, -0.2], [0.3, 0.0, -0.3], [0.2, 0.1, 0.0], [0.25, 0.1, 0.6], [0.3, 0.1, 1.5], [0.2, 0.0, 3.0]]):
        r = last(rig.frame(np.array(P)))
        assert r.status == FS.TRACKING_MULTI_VIEW and r.valid and r.reason is None
        worst = max(worst, float(np.linalg.norm(np.array(r.world_xyz_m) - P)))
        rig.advance(66)
    assert worst < 1e-6, worst                          # synthetic, noise-free


def test_result_contract_fields_units_sources_quality(rig):
    r = last(rig.frame())
    d = json.loads(r.model_dump_json())
    assert d["protocol_version"] == "1.0" and d["message_type"] == "POSE_UPDATE" and d["unit"] == "m"
    assert d["session_id"] == "sess_test0001" and d["target_landmark"] == "wrist" and d["world_frame_id"] == "board_frame"
    assert d["source_sensor_ids"] == sorted([A, B]) and len(d["sources"]) == 2 and d["skew_ms"] == 0.0
    for s in d["sources"]:
        assert s["used"] and s["reprojection_error_px"] < 1e-6 and s["calibration_id"] and s["camera_id"] == s["sensor_id"]
        assert s["captured_at_ms"] == T0 and s["received_at_ms"] == T0 and s["effective_time_ms"] == T0 and s["age_ms"] == 0
    assert d["aggregate_reprojection_error_px"] < 1e-6 and 30 < d["ray_angle_deg"] < 90
    assert d["quality"]["score"] == pytest.approx(1.0) and set(d["quality"]) == {"score", "reprojection", "sync", "geometry"}
    assert d["calibration"]["rig_calibration_id"] == "rig-1" and set(d["calibration"]["calibration_ids"]) == {A, B}
    assert d["timestamp_ms"] == T0 and d["held_world_xyz_m"] is None and d["single_view_ray"] is None and d["warnings"] == []
    FusionResult.model_validate(d)                      # round-trips through the contract


def test_fusion_sequence_is_monotonic_and_each_set_is_published_once(rig):
    seqs, count = [], 0
    for _ in range(5):
        for r in rig.frame():
            seqs.append(r.fusion_sequence); count += 1
        rig.advance(66)
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and count == 5
    assert rig.engine.tick() == []                      # nothing new -> nothing re-published


def test_example_pose_update_file_is_valid_and_documents_a_real_result(rig):
    p = Path(__file__).resolve().parents[2] / "docs" / "examples" / "fusion" / "pose_update_tracking.json"
    FusionResult.model_validate(json.loads(p.read_text()))


# ------------------------------------------------------------------------------ synchronization
def test_small_skew_is_accepted_and_reported_with_reduced_quality(rig):
    rig.feed(A)
    r = last(rig.feed(B, behind_ms=30))                 # B was captured 30 ms before A
    assert r.status == FS.TRACKING_MULTI_VIEW and r.skew_ms == pytest.approx(30.0)
    assert r.quality.sync == pytest.approx(0.4) and r.quality.score < 0.5


def test_excessive_skew_never_produces_a_3d_point(rig):
    rig.feed(A); assert rig.feed(B, behind_ms=80) == []          # not published yet: inside the degrade grace
    rig.advance(130)
    r = last(rig.engine.tick())
    assert r.status == FS.SYNC_DEGRADED and r.reason == "SKEW_EXCEEDS_WINDOW" and r.world_xyz_m is None and not r.valid
    assert r.skew_ms == pytest.approx(80.0) and "80" in r.message and "50" in r.message


def test_interleaved_arrivals_do_not_flicker_into_degraded(rig):
    statuses = []
    for _ in range(20):
        statuses += [x.status for x in rig.feed(A)]
        rig.advance(7)                                  # B arrives a few ms after A, every frame
        statuses += [x.status for x in rig.feed(B, behind_ms=7)]
        rig.advance(59)
    assert set(statuses) == {FS.TRACKING_MULTI_VIEW} and len(statuses) == 20


def test_phone_clock_3_hours_ahead_is_handled_through_the_offset(rig):
    skew = 3 * 3600 * 1000
    rig.feed(A)
    r = last(rig.feed(B, offset=-skew))                 # sensor B's clock is skewed, offset estimate says so
    assert r.status == FS.TRACKING_MULTI_VIEW and r.skew_ms == 0.0
    src = {s.sensor_id: s for s in r.sources}[B]
    assert src.captured_at_ms == T0 + skew and src.effective_time_ms == T0          # sensor clock preserved, effective time on host timeline


def test_stale_and_out_of_order_observations_never_fuse(rig):
    assert rig.feed(A, behind_ms=1000) == [] and rig.engine.counters["sync_rejected:STALE"] == 1
    rig.frame()
    n = rig.engine.sync.buffers[A][-1].sequence_number
    stale = S.observation_entry(A, LEFT, X, n, T0 - 5, T0, hand="right")             # same/lower sequence
    assert rig.engine.ingest(stale, 0) == [] and rig.engine.counters["sync_rejected:OUT_OF_ORDER_SEQUENCE"] == 1
    back = S.observation_entry(A, LEFT, X, n + 1, T0 - 200, T0, hand="right")        # newer sequence, older capture time
    assert rig.engine.ingest(back, 0) == []


# ------------------------------------------------------------------------------ views / calibration
def test_single_view_gives_a_ray_never_a_point(rig):
    rig.feed(A)
    rig.advance(130)
    r = last(rig.feed(A))
    assert r.status == FS.SINGLE_VIEW_FALLBACK and not r.valid and r.world_xyz_m is None and r.raw_world_xyz_m is None
    ray = r.single_view_ray
    o, d = np.array(ray.origin_m), np.array(ray.direction)
    assert ray.sensor_id == A and np.isclose(np.linalg.norm(d), 1) and np.linalg.norm(np.cross(X - o, d)) < 1e-6
    assert np.allclose(o, LEFT.center) and "NOT a 3D position" in r.message


def test_unbound_or_uncalibrated_sensors_require_calibration(rig):
    r = last(rig.feed("sensor_01", cam=LEFT))
    assert r.status == FS.CALIBRATION_REQUIRED and r.reason == "NO_CALIBRATED_SENSORS" and r.world_xyz_m is None
    assert "NOT_BOUND" in r.message and r.sources[0].note == "NOT_BOUND"


def test_dimension_mismatch_invalidates_use_of_that_camera(rig):
    def rotate(e):
        e["image_width"], e["image_height"] = 480, 640            # phone rotated: orientation differs from calibration
    r = last(rig.feed(A, mutate=rotate))
    assert r.status == FS.CALIBRATION_REQUIRED and "DIMENSION_MISMATCH" in r.message
    r2 = rig.feed(B)                                               # the other camera alone: only one usable view
    assert not any(x.valid for x in r2)


def test_incompatible_rig_sessions_are_not_fused(tmp_path):
    rig = Rig(tmp_path, rig_ids={A: "rig-A", B: "rig-B"})
    assert not any(r.valid for r in rig.frame())
    rig.advance(130)
    r = last(rig.frame())
    assert r.status == FS.SINGLE_VIEW_FALLBACK and r.world_xyz_m is None             # only the compatible camera may speak
    assert any("INCOMPATIBLE_RIG" in w for w in r.warnings)
    assert {s.sensor_id: s.note for s in r.sources}.get(B) == "INCOMPATIBLE_RIG" or {s.sensor_id: s.note for s in r.sources}.get(A) == "INCOMPATIBLE_RIG"


def test_synthetic_calibration_is_refused_unless_explicitly_allowed(tmp_path):
    rig = Rig(tmp_path)
    rig.registry.allow_synthetic = False
    r = last(rig.feed(A))
    assert r.status == FS.CALIBRATION_REQUIRED and "SYNTHETIC_REFUSED" in r.message


# ------------------------------------------------------------------------------ geometry / outliers
def test_two_inconsistent_views_are_rejected_as_outlier(rig):
    def corrupt(e):                                     # vertical error: cameras are level, so this is NOT absorbable by a depth change
        e["landmarks"][0]["y_px"] += 40.0
    rig.feed(A)
    r = last(rig.feed(B, mutate=corrupt))
    assert r.status == FS.FUSION_OUTLIER and r.reason == "REPROJECTION_ERROR_HIGH" and r.world_xyz_m is None and not r.valid
    assert r.aggregate_reprojection_error_px > 6 and all(s.reprojection_error_px is not None for s in r.sources)


def test_known_limitation_two_views_cannot_detect_errors_along_the_epipolar_line(rig):
    """DOCUMENTED LIMITATION (docs/spatial.md): with exactly two views, a landmark error ALONG the epipolar line is
    indistinguishable from a different depth, so reprojection residual stays ~0 while XYZ is wrong. A 3rd camera fixes it."""
    def slide(e):
        e["landmarks"][0]["x_px"] += 25.0               # horizontal = along the (level-camera) epipolar line
    rig.feed(A)
    r = last(rig.feed(B, mutate=slide))
    assert r.status == FS.TRACKING_MULTI_VIEW and r.aggregate_reprojection_error_px < 1.0           # looks perfect...
    assert np.linalg.norm(np.array(r.world_xyz_m) - X) > 0.02                                       # ...but is wrong by > 2 cm


def test_third_view_lets_the_engine_drop_the_bad_one(tmp_path):
    rig = Rig(tmp_path, cams=(LEFT, RIGHT, OVERHEAD))
    def corrupt(e):
        e["landmarks"][0]["y_px"] += 30.0
    rig.feed(A); rig.feed("overhead")
    r = last(rig.feed(B, mutate=corrupt))
    assert r.status == FS.TRACKING_MULTI_VIEW and r.warnings == [f"OUTLIER_VIEW_DROPPED:{B}"]
    assert r.source_sensor_ids == sorted([A, "overhead"]) and np.linalg.norm(np.array(r.world_xyz_m) - X) < 1e-6
    bad = {s.sensor_id: s for s in r.sources}[B]
    assert not bad.used


def test_degenerate_baseline_is_not_trusted(tmp_path):
    twin = G.CameraModel("twin", LEFT.K, LEFT.dist, LEFT.R, LEFT.t - LEFT.R @ np.array([0.01, 0, 0]), 640, 480)   # centres 1 cm apart
    rig = Rig(tmp_path, cams=(LEFT, twin))
    rig.feed(A)
    r = last(rig.feed("twin"))
    assert r.status == FS.GEOMETRY_DEGENERATE and r.reason == "BASELINE_TOO_SMALL" and r.world_xyz_m is None


def test_points_behind_the_cameras_never_become_a_trusted_estimate(rig):
    behind = np.array([0.2, 0.05, -3.0])
    rig.feed(A, behind)
    r = last(rig.feed(B, behind))
    assert not r.valid and r.world_xyz_m is None and r.status in (FS.GEOMETRY_DEGENERATE, FS.FUSION_OUTLIER)


def test_non_finite_pixels_never_reach_a_result(rig):
    def nan(e):
        e["landmarks"][0]["x_px"] = float("nan")
    rig.feed(A)
    res = rig.feed(B, mutate=nan)
    assert not any(r.valid for r in res)
    assert all(r.world_xyz_m is None for r in res)


def test_conflicting_hand_labels_are_not_fused_but_unknown_is_a_wildcard(rig):
    rig.feed(A, hand="right")
    r = last(rig.feed(B, hand="left"))
    assert r.status == FS.FUSION_OUTLIER and r.reason == "HAND_LABEL_MISMATCH" and r.world_xyz_m is None
    rig.advance(66)
    rig.feed(A, hand="right")
    assert last(rig.feed(B, hand="unknown")).status == FS.TRACKING_MULTI_VIEW


# ------------------------------------------------------------------------------ tracking loss / smoothing
def test_target_visibility_loss_holds_then_reports_target_lost(rig):
    good = last(rig.frame())
    rig.advance(66)
    r = last(rig.frame(status="no_hand"))                           # hand leaves both views
    assert r.status == FS.INSUFFICIENT_VIEWS and r.world_xyz_m is None and not r.valid
    assert np.allclose(r.held_world_xyz_m, good.world_xyz_m) and 0 <= r.held_age_ms <= 500
    rig.advance(600)                                                # beyond hold_ms (500)
    lost = last(rig.engine.tick())
    assert lost.status == FS.TARGET_LOST and lost.reason == "HOLD_EXPIRED" and lost.held_world_xyz_m is None and lost.world_xyz_m is None
    assert "hold_expired" in rig.engine.smoothers["wrist"].resets
    assert rig.engine.tick() == []                                  # reported once, not every tick


def test_silent_sensors_also_end_in_target_lost_without_extrapolation(rig):
    rig.frame()
    seen = []
    for _ in range(40):                                             # no observations at all, 50 ms ticks
        rig.advance(50)
        seen += rig.engine.tick()
    st = [r.status for r in seen]
    assert st == [FS.INSUFFICIENT_VIEWS, FS.TARGET_LOST]
    assert seen[0].held_world_xyz_m is not None and all(r.world_xyz_m is None for r in seen)
    held_at = seen[0].published_at_ms - T0
    lost_at = seen[1].published_at_ms - T0
    assert 250 <= held_at <= 350 and 500 < lost_at <= 600           # stale at 250 ms, hold ends at 500 ms


def test_recovery_after_loss_restarts_cleanly(rig):
    rig.frame(); rig.advance(700); rig.engine.tick()
    rig.advance(66)
    r = last(rig.frame(np.array([0.3, 0.1, 1.0])))
    assert r.status == FS.TRACKING_MULTI_VIEW and np.allclose(r.world_xyz_m, [0.3, 0.1, 1.0], atol=1e-6)    # not dragged towards old position


def test_smoothing_is_applied_after_validation_only_and_reports_both_values(tmp_path):
    rig = Rig(tmp_path, cfg_over={"smoothing.enabled": True})
    r0 = last(rig.frame(np.array([0.25, 0.1, 0.6])))
    rig.advance(66)
    sm = rig.engine.smoothers["wrist"]
    t_before = sm.last_t_ms
    rig.feed(A)                                                      # invalid set follows: must not touch the filter
    def corrupt(e):
        e["landmarks"][0]["y_px"] += 40
    rig.feed(B, mutate=corrupt)
    assert sm.last_t_ms == t_before
    rig.advance(66)
    r1 = last(rig.frame(np.array([0.30, 0.1, 0.6])))
    raw, smooth = np.array(r1.raw_world_xyz_m), np.array(r1.world_xyz_m)
    assert np.allclose(raw, [0.30, 0.1, 0.6], atol=1e-6) and 0 < np.linalg.norm(smooth - raw) < np.linalg.norm(raw - r0.world_xyz_m)
    assert r1.smoothing_applied


def test_track_discontinuity_and_identity_change_reset_the_filter(tmp_path):
    rig = Rig(tmp_path, cfg_over={"smoothing.enabled": True})
    rig.frame(np.array([0.25, 0.1, 0.6]), hand="right"); rig.advance(66)
    r = last(rig.frame(np.array([0.2, 0.0, 3.0]), hand="right"))      # ~2.4 m in 66 ms = 36 m/s: not the same hand
    assert "TRACK_DISCONTINUITY" in r.warnings and np.allclose(r.world_xyz_m, r.raw_world_xyz_m)
    rig.advance(66)
    rig.frame(np.array([0.2, 0.0, 3.0]), hand="left")
    assert any(x.startswith("hand_identity_changed") for x in rig.engine.smoothers["wrist"].resets)


# ------------------------------------------------------------------------------ calibration lifecycle
def test_invalidation_blocks_that_camera_and_recalibration_restores_it(rig):
    assert last(rig.frame()).valid
    rig.advance(66)
    rig.registry.invalidate("phone_left", "tripod bumped")
    rig.frame(); rig.advance(130)
    r = last(rig.frame())
    assert not r.valid and r.world_xyz_m is None                                       # never a 3D point from a broken rig
    assert r.status == FS.SINGLE_VIEW_FALLBACK and r.single_view_ray.sensor_id == B     # the healthy camera can still give a ray
    assert f"CALIBRATION_PROBLEM:{A}:INVALIDATED" in r.warnings
    assert {s.sensor_id: s.note for s in r.sources}[A] == "INVALIDATED"
    rig.advance(66)
    assert not any(x.valid for x in rig.frame())                                       # still blocked
    # Recalibrating ONE camera in a different reference-board session must NOT silently join the old world frame ...
    rig.store.save_camera(S.artifact_from_camera(LEFT, calibration_id="left-solo", rig_calibration_id="rig-other-session"))
    rig.registry.reload()
    rig.advance(66)
    assert not any(x.valid for x in rig.frame())
    rig.advance(130)
    r = last(rig.frame())
    assert any("INCOMPATIBLE_RIG" in w for w in r.warnings)
    # ... only a consistent rig calibration (same reference-board session for all cameras) restores fusion
    rig.store.save_camera(S.artifact_from_camera(LEFT, calibration_id="left-recal", rig_calibration_id="rig-1"))
    rig.registry.reload()
    rig.advance(66)
    r = last(rig.frame())
    assert r.status == FS.TRACKING_MULTI_VIEW and r.calibration.calibration_ids[A] == "left-recal"
    assert "calibration_changed" in rig.engine.smoothers["wrist"].resets                # no smoothing across a calibration change


def test_all_cameras_invalid_means_calibration_required(rig):
    rig.registry.invalidate("phone_left", "moved"); rig.registry.invalidate("phone_right", "moved")
    r = last(rig.frame())
    assert r.status == FS.CALIBRATION_REQUIRED and "INVALIDATED" in r.message and r.world_xyz_m is None


def test_status_transitions_are_published_once(rig):
    log = []
    rig.engine.subscribe(lambda r: log.append(r.status))
    rig.frame(); rig.advance(66)
    rig.frame(status="no_hand"); rig.advance(600)
    rig.engine.tick(); rig.engine.tick(); rig.advance(50); rig.engine.tick()
    assert log == [FS.TRACKING_MULTI_VIEW, FS.INSUFFICIENT_VIEWS, FS.TARGET_LOST]


def test_a_failing_subscriber_does_not_break_fusion(rig):
    rig.engine.subscribe(lambda r: 1 / 0)
    got = []
    rig.engine.subscribe(got.append)
    assert last(rig.frame()).valid and len(got) == 1 and rig.engine.counters["subscriber_errors"] == 1


def test_history_is_bounded(rig):
    rig.engine.history = type(rig.engine.history)(maxlen=5)
    for _ in range(12):
        rig.frame(); rig.advance(66)
    assert len(rig.engine.history) == 5


def test_multiple_targets_are_fused_independently(tmp_path):
    rig = Rig(tmp_path, cfg_over={"spatial.targets": ["wrist", "index_tip"]})
    def both(e):
        wrist = e["landmarks"][0]
        e["landmarks"].append({**wrist, "id": "index_tip", "x_px": wrist["x_px"] + 20})
    for sid in (A, B):
        n = rig.seq[sid] = rig.seq.get(sid, -1) + 1
        e = S.observation_entry(sid, rig.cams[sid], X, n, T0, T0); both(e)
        out = rig.engine.ingest(e, 0)
    assert {r.target_landmark for r in out} == {"wrist", "index_tip"}
