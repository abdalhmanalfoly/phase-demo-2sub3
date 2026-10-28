"""Pinch measurement, hysteresis, confirmation window, smoothing and configuration validation."""
import math

import pytest

from host.app.gesture.config import GestureCfg, GestureConfigError, gesture_config_from_mapping, load_gesture_config
from host.app.gesture.pinch import PinchDetector, combine_view_ratios, pinch_ratio

CFG = GestureCfg()


def lm(thumb=(100, 100), index=(110, 100), wrist=(100, 200), mcp=(100, 100 + 0), conf=None):
    mcp = (100, 120)                                  # palm length = 80 px
    return {"thumb_tip": (*thumb, conf), "index_tip": (*index, conf), "wrist": (*wrist, conf), "index_mcp": (*mcp, conf)}


def test_ratio_is_distance_over_palm_length():
    r, why = pinch_ratio(lm(thumb=(100, 100), index=(120, 100)), CFG)
    assert why is None and r == pytest.approx(20 / 80)


def test_ratio_is_scale_invariant():
    a = lm(thumb=(100, 100), index=(120, 100))
    b = {k: (v[0] * 3.0, v[1] * 3.0, v[2]) for k, v in a.items()}          # same hand, 3x closer to the camera
    assert pinch_ratio(a, CFG)[0] == pytest.approx(pinch_ratio(b, CFG)[0])


@pytest.mark.parametrize("mutate,reason", [
    (lambda d: d.pop("thumb_tip"), "LANDMARKS_MISSING"),
    (lambda d: d.pop("index_mcp"), "LANDMARKS_MISSING"),
    (lambda d: d.update(thumb_tip=(float("nan"), 1.0, None)), "NON_FINITE_LANDMARK"),
    (lambda d: d.update(index_tip=(1.0, float("inf"), None)), "NON_FINITE_LANDMARK"),
    (lambda d: d.update(index_tip=(110, 100, 0.1)), "LOW_LANDMARK_CONFIDENCE"),
    (lambda d: d.update(wrist=(100, 121, None)), "PALM_SCALE_TOO_SMALL"),
])
def test_invalid_landmarks_give_no_ratio_never_zero(mutate, reason):
    d = lm()
    mutate(d)
    assert pinch_ratio(d, CFG) == (None, reason)


def test_missing_landmark_confidence_is_accepted_because_the_model_reports_none():
    assert pinch_ratio(lm(conf=None), CFG)[1] is None
    assert pinch_ratio(lm(conf=0.9), CFG)[1] is None


def test_combine_views_uses_the_median_and_handles_empty():
    assert combine_view_ratios([0.2, 0.9, 0.22]) == pytest.approx(0.22)
    assert combine_view_ratios([]) is None


def test_config_enforces_hysteresis_ordering_and_bounds():
    with pytest.raises(GestureConfigError):
        GestureCfg(pinch_enter_ratio=0.5, pinch_exit_ratio=0.4).validate()
    with pytest.raises(GestureConfigError):
        GestureCfg(pinch_enter_ratio=0.3, pinch_exit_ratio=0.3).validate()
    with pytest.raises(GestureConfigError):
        GestureCfg(reach_enter_m=0.2, reach_exit_m=0.1).validate()
    with pytest.raises(GestureConfigError):
        GestureCfg(min_confidence=0.3, exit_confidence=0.4).validate()
    with pytest.raises(GestureConfigError):
        GestureCfg(grab_loss_timeout_ms=60_000).validate()                        # freeze must be bounded
    with pytest.raises(GestureConfigError):
        GestureCfg(pinch_enter_ratio=float("nan")).validate()
    with pytest.raises(GestureConfigError):
        gesture_config_from_mapping({"nonsense": 1})
    assert load_gesture_config() == GestureCfg().validate()


def feed(det, values, dt=60, t0=0.0):
    out, t = [], t0
    for v in values:
        out.append(det.update(v, t))
        t += dt
    return out


def test_pinch_needs_confirmation_duration_and_samples():
    det = PinchDetector(CFG)
    st = feed(det, [0.1])
    assert st[-1].state == "CANDIDATE" and not st[-1].just_confirmed
    det = PinchDetector(CFG)
    st = feed(det, [0.1, 0.1, 0.1, 0.1])                                         # 180 ms >= 120 ms, 4 samples >= 3
    assert [s.just_confirmed for s in st] == [False, False, True, False] and st[-1].state == "PINCHED"


def test_single_noisy_dip_below_the_threshold_never_confirms():
    det = PinchDetector(CFG)
    st = feed(det, [0.6, 0.6, 0.05, 0.6, 0.6, 0.05, 0.6, 0.6])
    assert not any(s.just_confirmed for s in st) and st[-1].state == "OPEN"


def test_noisy_fingertips_crossing_the_threshold_back_and_forth_do_not_grab():
    det = PinchDetector(CFG)
    st = feed(det, [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3], dt=40)               # median stays ~0.25-0.3 > enter
    assert not any(s.just_confirmed for s in st)


def test_hysteresis_band_does_not_toggle_state():
    det = PinchDetector(CFG)
    feed(det, [0.1] * 5)
    assert det.state == "PINCHED"
    st = feed(det, [0.3, 0.4, 0.3, 0.44, 0.3, 0.4, 0.3, 0.44], t0=1000)           # between enter (0.25) and exit (0.45)
    assert all(s.state == "PINCHED" and not s.just_released for s in st)
    det2 = PinchDetector(CFG)
    st = feed(det2, [0.3, 0.4, 0.3, 0.44, 0.3, 0.4, 0.3, 0.44])
    assert all(s.state == "OPEN" for s in st)                                      # and an open hand in the band does not start a pinch


def test_release_requires_the_exit_threshold_and_a_short_hold():
    det = PinchDetector(CFG)
    feed(det, [0.1] * 5)
    st = feed(det, [0.9], t0=1000)
    assert st[-1].state == "PINCHED" and not st[-1].just_released                  # first open frame alone is not enough
    st = feed(det, [0.9, 0.9, 0.9], t0=1060)
    assert any(s.just_released for s in st) and st[-1].state == "OPEN"


def test_one_noisy_open_sample_while_pinched_does_not_release():
    det = PinchDetector(CFG)
    feed(det, [0.1] * 5)
    st = feed(det, [0.1, 0.9, 0.1, 0.1, 0.9, 0.1], t0=1000)
    assert all(s.state == "PINCHED" for s in st)


def test_invalid_evidence_changes_nothing_and_a_gap_restarts_the_candidate():
    det = PinchDetector(CFG)
    feed(det, [0.1, 0.1])
    assert det.update(None, 200).state == "CANDIDATE"
    assert det.update(float("nan"), 220).state == "CANDIDATE" and det.update(-1.0, 240).state == "CANDIDATE"
    st = det.update(0.1, 2000)                                                     # 1.8 s later: continuity broken
    assert st.state == "CANDIDATE" and not st.just_confirmed and st.candidate_ms == 0.0


def test_reset_clears_everything():
    det = PinchDetector(CFG)
    feed(det, [0.1] * 5)
    det.reset()
    assert det.state == "OPEN" and det.smoothed is None
