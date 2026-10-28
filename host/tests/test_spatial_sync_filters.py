"""Observation synchronization (bounded buffers, matching windows, staleness/order) and the temporal smoother."""
import numpy as np
import pytest

from host.app.config import SyncCfg
from host.app.spatial import filters as F
from host.app.spatial import synthetic as S
from host.app.spatial.synchronizer import ObservationSynchronizer

CAMS = S.rig_cameras()
X = np.array([0.2, 0.05, 0.0])
T0 = 1_791_735_600_000


def entry(sid, seq, captured, received=None, status="tracked", **kw):
    received = captured + 5 if received is None else received
    return S.observation_entry(sid, CAMS["phone_left"], X, seq, captured, received, status=status, **kw)


def sync(**over):
    return ObservationSynchronizer(SyncCfg(**over))


def test_matches_within_window_and_reports_measured_skew():
    s = sync()
    assert s.add(entry("a", 0, T0), 0, T0 + 10)[0] and s.add(entry("b", 0, T0 + 30), 0, T0 + 40)[0]
    m = s.match("wrist", ["a", "b"], T0 + 40)
    assert m.matched and m.is_new and m.skew_ms == pytest.approx(30.0) and {o.sensor_id for o in m.members} == {"a", "b"}


def test_excessive_skew_is_not_matched_and_reports_reason():
    s = sync()
    s.add(entry("a", 0, T0), 0, T0 + 10)
    s.add(entry("b", 0, T0 + 120), 0, T0 + 130)
    m = s.match("wrist", ["a", "b"], T0 + 130)
    assert not m.matched and m.reason == "SKEW_EXCEEDS_WINDOW" and m.skew_ms == pytest.approx(120.0) and len(m.usable) == 2
    # a looser window accepts the very same data - the window is configuration, not magic
    s2 = sync(max_skew_ms=150)
    s2.add(entry("a", 0, T0), 0, T0 + 10); s2.add(entry("b", 0, T0 + 120), 0, T0 + 130)
    assert s2.match("wrist", ["a", "b"], T0 + 130).matched


def test_picks_the_best_aligned_pair_from_history():
    s = sync()
    for i in range(5):                                  # a at 15 Hz, b at 15 Hz offset by 20 ms
        s.add(entry("a", i, T0 + 66 * i), 0, T0 + 66 * i + 5)
        s.add(entry("b", i, T0 + 66 * i + 20), 0, T0 + 66 * i + 25)
    m = s.match("wrist", ["a", "b"], T0 + 66 * 4 + 25)
    assert m.matched and m.skew_ms == pytest.approx(20.0) and {o.sequence_number for o in m.members} == {4}


def test_clock_offset_is_applied_so_a_skewed_phone_still_matches():
    s = sync()
    skew = 3 * 3600 * 1000
    s.add(entry("a", 0, T0), 0, T0 + 10)
    s.add(entry("b", 0, T0 + 20 + skew, received=T0 + 25), -skew, T0 + 30)
    m = s.match("wrist", ["a", "b"], T0 + 30)
    assert m.matched and m.skew_ms == pytest.approx(20.0)


def test_stale_observations_are_rejected_and_never_matched():
    s = sync()
    ok, why = s.add(entry("a", 0, T0), 0, T0 + 1000)            # 1 s old when it arrives
    assert not ok and why == "STALE" and s.rejected[("a", "STALE")] == 1
    s.add(entry("a", 1, T0 + 900), 0, T0 + 910); s.add(entry("b", 1, T0 + 905), 0, T0 + 915)
    assert s.match("wrist", ["a", "b"], T0 + 915).matched
    m = s.match("wrist", ["a", "b"], T0 + 900 + 400)             # both age out (max 250 ms)
    assert not m.matched and not m.usable and set(m.unusable.values()) == {"STALE"}


def test_out_of_order_sequence_and_time_rejected():
    s = sync()
    assert s.add(entry("a", 5, T0 + 100), 0, T0 + 110)[0]
    assert s.add(entry("a", 5, T0 + 110), 0, T0 + 120) == (False, "OUT_OF_ORDER_SEQUENCE")
    assert s.add(entry("a", 4, T0 + 120), 0, T0 + 130) == (False, "OUT_OF_ORDER_SEQUENCE")
    assert s.add(entry("a", 6, T0 + 90), 0, T0 + 130) == (False, "OUT_OF_ORDER_TIME")
    assert s.add(entry("a", 7, T0 + 140), 0, T0 + 150)[0]
    assert len(s.buffers["a"]) == 2


def test_missing_observations_are_counted_as_sequence_gaps():
    s = sync()
    for seq in (0, 1, 4, 5, 9):
        s.add(entry("a", seq, T0 + seq), 0, T0 + seq + 10)
    assert s.sequence_gaps["a"] == 5                             # missing: {2,3} and {6,7,8}


def test_clock_problem_detected_when_received_before_captured():
    s = sync()
    ok, why = s.add(entry("a", 0, T0 + 5000, received=T0 + 100), 0, T0 + 100)   # claims to be captured 5 s in the future
    assert not ok and why == "CLOCK_SUSPECT"
    assert s.add(entry("a", 0, T0), None, T0 + 10) == (False, "NO_CLOCK_OFFSET")


def test_buffers_are_bounded_and_drop_the_oldest():
    s = sync(buffer_per_sensor=4)
    for i in range(20):
        s.add(entry("a", i, T0 + i), 0, T0 + i + 5)
    assert len(s.buffers["a"]) == 4 and [o.sequence_number for o in s.buffers["a"]] == [16, 17, 18, 19]
    assert s.overflow_drops["a"] == 16


def test_no_hand_hides_older_tracked_observations():
    s = sync()
    s.add(entry("a", 0, T0), 0, T0 + 5); s.add(entry("a", 1, T0 + 40, status="no_hand"), 0, T0 + 45)
    s.add(entry("b", 0, T0 + 10), 0, T0 + 15)
    m = s.match("wrist", ["a", "b"], T0 + 50)
    assert not m.matched and m.unusable["a"] == "TARGET_NOT_VISIBLE" and set(m.usable) == {"b"}


def test_consumed_sets_are_not_new_again():
    s = sync()
    s.add(entry("a", 0, T0), 0, T0 + 5); s.add(entry("b", 0, T0 + 5), 0, T0 + 10)
    m = s.match("wrist", ["a", "b"], T0 + 10)
    assert m.is_new
    s.mark_consumed("wrist", m.members)
    assert s.match("wrist", ["a", "b"], T0 + 12).is_new is False


def test_diagnostics_report_transport_delay():
    s = sync()
    for i in range(5):
        s.add(entry("a", i, T0 + 66 * i, received=T0 + 66 * i + 8 + i), 0, T0 + 66 * i + 20)
    d = s.diagnostics()["transport_delay"]["a"]
    assert d["n"] == 5 and d["min_ms"] == 8 and d["max_ms"] == 12


# ------------------------------------------------------------------------------ filters
def smoother(**kw):
    p = dict(enabled=True, min_cutoff_hz=3.0, beta=1.0, d_cutoff_hz=1.0, hold_ms=500)
    p.update(kw)
    return F.TrackSmoother(**p)


def test_first_sample_passes_through_and_constant_input_is_stable():
    sm = smoother()
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(sm.update(p, 1000.0), p)
    for i in range(1, 30):
        assert np.allclose(sm.update(p, 1000.0 + 66 * i), p)


def test_smoothing_reduces_noise_and_lag_is_measured_and_bounded():
    """MEASURED trade-off at 15 Hz (printed by the suite via -s): lag at 0.5 m/s, static-noise retention."""
    rng = np.random.default_rng(1)
    f = F.OneEuro3D(3.0, 1.0, 1.0)
    out = np.array([f.filter(np.array([0, 0, 1.0]) + rng.normal(0, 0.005, 3), i / 15) for i in range(2000)])[200:]
    kept = float(np.std(out - [0, 0, 1.0]) / 0.005)
    lag = F.measure_lag_ms(3.0, 1.0, 1.0, 0.5)
    assert kept < 0.8 and 15 < lag < 60, (kept, lag)             # real smoothing, and lag well inside a 150 ms budget
    assert F.measure_lag_ms(1.5, 0.3, 1.0, 0.5) > lag + 20       # the 'prettier' setting costs much more latency


def test_timestamps_must_advance_and_the_filter_never_runs_backwards():
    sm = smoother()
    sm.update(np.array([0.0, 0, 1]), 1000.0)
    b = sm.update(np.array([0.1, 0, 1]), 1066.0)
    again = sm.update(np.array([9.0, 9, 9]), 1066.0)             # duplicate timestamp
    older = sm.update(np.array([9.0, 9, 9]), 900.0)              # older measurement
    assert np.allclose(again, b) and np.allclose(older, b)


def test_hold_is_bounded_and_expiry_invalidates_then_resets():
    sm = smoother(hold_ms=500)
    sm.update(np.array([0.5, 0.1, 1.0]), 10_000.0)
    h = sm.held(10_300.0)
    assert h is not None and h[1] == pytest.approx(300.0) and np.allclose(h[0], [0.5, 0.1, 1.0])
    assert sm.held(10_501.0) is None and not sm.expire(10_400.0)
    assert sm.expire(10_501.0) is True and sm.last_xyz is None and "hold_expired" in sm.resets
    assert sm.expire(10_900.0) is False                          # reported once
    assert np.allclose(sm.update(np.array([2.0, 0, 1]), 12_000.0), [2.0, 0, 1])    # fresh start, nothing smoothed across the loss


def test_filter_resets_on_track_identity_change_and_explicit_reset():
    sm = smoother()
    sm.update(np.array([0.0, 0, 1]), 1000.0, key=("wrist", "right"))
    sm.update(np.array([0.1, 0, 1]), 1066.0, key=("wrist", "right"))
    out = sm.update(np.array([5.0, 0, 1]), 1132.0, key=("wrist", "left"))      # different identity: no smoothing across it
    assert np.allclose(out, [5.0, 0, 1]) and any(r.startswith("track_key_changed") for r in sm.resets)
    sm.reset("session_reset")
    assert sm.held(1200.0) is None and sm.last_xyz is None and sm.resets[-1] == "session_reset"


def test_disabled_smoothing_is_identity():
    sm = smoother(enabled=False)
    sm.update(np.array([0.0, 0, 1]), 1000.0)
    assert np.allclose(sm.update(np.array([0.3, 0, 1]), 1066.0), [0.3, 0, 1])
