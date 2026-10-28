"""Offer scoring, normalisation, validation of offers (missing/invalid never silently excellent) and configuration."""
import copy
import math

import pytest

from host.app.coordination import scoring
from host.app.coordination.config import ConfigError, CoordinationCfg, ScoreWeights, ScoringCfg, config_from_mapping, load_coordination_config
from host.app.coordination.messages import MessageError, ObservationOffer
from host.app.coordination.sensor_agent import CalibrationView, ObservationView, SensorAgent, SensorSnapshot
from host.app.coordination import simulation as S

W = ScoreWeights()
SC = ScoringCfg()


def test_formula_matches_documented_weights_for_known_inputs():
    q = scoring.offer_quality(W, detector_confidence=0.8, visibility_score=0.6, freshness_score=0.5, sync_score=0.4, health_score=0.2)
    assert q == pytest.approx(0.30 * 0.8 + 0.20 * 0.6 + 0.20 * 0.5 + 0.15 * 0.4 + 0.15 * 0.2)
    assert q == pytest.approx(0.55)


def test_formula_extremes():
    assert scoring.offer_quality(W, detector_confidence=1, visibility_score=1, freshness_score=1, sync_score=1, health_score=1) == pytest.approx(1.0)
    assert scoring.offer_quality(W, detector_confidence=0, visibility_score=0, freshness_score=0, sync_score=0, health_score=0) == 0.0


def test_weights_come_from_configuration():
    w = ScoreWeights(detector_confidence=1.0, visibility=0.0, freshness=0.0, sync=0.0, health=0.0)
    assert scoring.offer_quality(w, detector_confidence=0.7, visibility_score=0, freshness_score=0, sync_score=0, health_score=0) == pytest.approx(0.7)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.01, 1.01, True, "0.5"])
def test_invalid_or_missing_score_input_is_rejected_not_defaulted(bad):
    with pytest.raises(ValueError):
        scoring.offer_quality(W, detector_confidence=bad, visibility_score=1, freshness_score=1, sync_score=1, health_score=1)


def test_weights_validation():
    with pytest.raises(ConfigError):
        ScoreWeights(detector_confidence=0.5).validate()                     # does not sum to 1
    with pytest.raises(ConfigError):
        ScoreWeights(detector_confidence=-0.1, visibility=0.5, freshness=0.3, sync=0.15, health=0.15).validate()
    with pytest.raises(ConfigError):
        config_from_mapping({"weights": {"bogus": 1}})                        # unknown key
    with pytest.raises(ConfigError):
        config_from_mapping({"task": {"lease_ms": 9000, "max_lease_ms": 5000}})   # leases are bounded


def test_yaml_file_equals_validated_defaults():
    assert load_coordination_config() == CoordinationCfg().validate()


def test_freshness_sync_health_visibility_calculations():
    assert scoring.freshness_score(0, SC) == 1.0
    assert scoring.freshness_score(125, SC) == pytest.approx(0.5)
    assert scoring.freshness_score(250, SC) == 0.0 and scoring.freshness_score(900, SC) == 0.0
    assert scoring.freshness_score(None, SC) == 0.0 and scoring.freshness_score(float("nan"), SC) == 0.0   # unknown age != fresh
    assert scoring.sync_score(True, 0.0, SC) == 1.0 and scoring.sync_score(True, 25.0, SC) == pytest.approx(0.5)
    assert scoring.sync_score(False, 0.0, SC) == 0.0 and scoring.sync_score(True, None, SC) == 0.0
    assert scoring.health_score("online", 10.0, SC) == 1.0
    assert scoring.health_score("online", None, SC) == pytest.approx(0.5)                       # unknown rate: half credit only
    assert scoring.health_score("degraded", 10.0, SC) == pytest.approx(0.4)
    assert scoring.health_score("offline", 15.0, SC) == 0.0 and scoring.health_score("???", 15.0, SC) == 0.0
    lm = {"wrist": (320.0, 240.0, None)}
    assert scoring.visibility_score("tracked", lm, "wrist", 640, 480, (), SC) == 1.0
    assert scoring.visibility_score("tracked", {"wrist": (0.0, 240.0, None)}, "wrist", 640, 480, (), SC) == pytest.approx(0.5)   # on the border
    assert scoring.visibility_score("tracked", {"wrist": (-5.0, 240.0, None)}, "wrist", 640, 480, (), SC) == 0.0              # outside
    assert scoring.visibility_score("no_hand", {}, "wrist", 640, 480, (), SC) == 0.0
    assert scoring.visibility_score("tracked", lm, "thumb_tip", 640, 480, (), SC) == 0.0
    assert scoring.visibility_score("tracked", lm, "wrist", 640, 480, ("out_of_frame",), SC) == pytest.approx(0.3)


def test_missing_detector_confidence_is_imputed_neutral_and_flagged_never_excellent():
    assert scoring.detector_confidence_value(None, SC) == (0.5, "imputed")
    assert scoring.detector_confidence_value(0.93, SC) == (0.93, "measured")
    assert scoring.detector_confidence_value(float("nan"), SC) == (0.5, "imputed")
    h = S.SimHarness(S._cfg(), ("cam_a", "cam_c"))
    h.world.sensors["cam_a"].detector_confidence = None            # what the real hand model sends
    task = _task(h)
    offer = h.agents["cam_a"].on_task_request(task)
    assert offer.detector_confidence == 0.5 and offer.detector_confidence_source == "imputed"
    assert "DETECTOR_CONFIDENCE_IMPUTED" in offer.notes


def _task(h):
    h.coord.request_observation()
    return h.coord.outbox[0]


def _valid(h=None):
    h = h or S.SimHarness(S._cfg(), ("cam_a", "cam_c"))
    return h.agents["cam_a"].on_task_request(_task(h)).to_dict()


def test_offer_roundtrip_and_required_fields():
    d = _valid()
    assert d["message_type"] == "OBSERVATION_OFFER" and d["protocol_version"] == "1.0" and d["channel"] == "coordination"
    for k in ("sensor_id", "task_id", "target_landmark", "detector_confidence", "visibility_score", "freshness_score", "sync_score", "health_score",
              "estimated_latency_ms", "calibration_eligible", "capabilities", "observation_effective_ms", "observation_age_ms", "submitted_at_ms"):
        assert k in d
    assert ObservationOffer.from_dict(d).to_dict() == d


@pytest.mark.parametrize("field", ["sensor_id", "task_id", "detector_confidence", "visibility_score", "freshness_score", "sync_score",
                                   "health_score", "estimated_latency_ms", "calibration_eligible", "capabilities", "submitted_at_ms", "view_ray"])
def test_offer_missing_field_is_rejected(field):
    d = _valid()
    del d[field]
    with pytest.raises(MessageError) as e:
        ObservationOffer.from_dict(d)
    assert e.value.code == "MISSING_FIELD"


@pytest.mark.parametrize("field,value,code", [
    ("detector_confidence", None, "MISSING_FIELD"), ("visibility_score", 1.5, "OUT_OF_RANGE"), ("freshness_score", -0.1, "OUT_OF_RANGE"),
    ("sync_score", float("nan"), "OUT_OF_RANGE"), ("health_score", "high", "WRONG_TYPE"), ("estimated_latency_ms", -1, "OUT_OF_RANGE"),
    ("calibration_eligible", "yes", "WRONG_TYPE"), ("capabilities", "wrist", "WRONG_TYPE"), ("submitted_at_ms", 0, "OUT_OF_RANGE"),
    ("offer_sequence", -1, "OUT_OF_RANGE"), ("protocol_version", "2.0", "UNSUPPORTED_VERSION"), ("detector_confidence_source", "guess", "OUT_OF_RANGE"),
    ("view_ray", [[0, 0, 0], [0, 0, 2]], "OUT_OF_RANGE")])
def test_offer_invalid_values_are_rejected(field, value, code):
    d = _valid()
    d[field] = value
    with pytest.raises(MessageError) as e:
        ObservationOffer.from_dict(d)
    assert e.value.code == code
