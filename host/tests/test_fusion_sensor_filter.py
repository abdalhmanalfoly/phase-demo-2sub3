"""Phase 3 additive hook on the Phase 2 engine: ``FusionEngine.set_sensor_filter``. NEEDS pydantic (skipped without it)."""
import pytest

pytest.importorskip("pydantic")

from host.app.spatial.results import FusionStatus as FS                      # noqa: E402
from .test_spatial_fusion import A, B, LEFT, OVERHEAD, RIGHT, Rig, X, last   # noqa: E402


def test_without_a_filter_behaviour_is_unchanged(tmp_path):
    rig = Rig(tmp_path, cams=(LEFT, RIGHT, OVERHEAD))
    r = last(rig.frame(X, sids=(A, B, "overhead")))
    assert r.status == FS.TRACKING_MULTI_VIEW and sorted(r.source_sensor_ids) == sorted([A, B, "overhead"])


def test_filter_restricts_the_contributing_sensors(tmp_path):
    rig = Rig(tmp_path, cams=(LEFT, RIGHT, OVERHEAD))
    rig.engine.set_sensor_filter("wrist", {A, B})
    r = last(rig.frame(X, sids=(A, B, "overhead")))
    assert r.status == FS.TRACKING_MULTI_VIEW and sorted(r.source_sensor_ids) == sorted([A, B])


def test_empty_filter_means_an_ended_lease_controls_nothing(tmp_path):
    rig = Rig(tmp_path, cams=(LEFT, RIGHT))
    rig.engine.set_sensor_filter("wrist", frozenset())
    res = rig.frame(X, sids=(A, B))
    assert res and all(not r.valid and r.world_xyz_m is None for r in res)


def test_clearing_the_filter_restores_fusion(tmp_path):
    rig = Rig(tmp_path, cams=(LEFT, RIGHT))
    rig.engine.set_sensor_filter("wrist", frozenset())
    rig.frame(X, sids=(A, B))
    rig.advance(66)
    rig.engine.set_sensor_filter("wrist", None)
    assert last(rig.frame(X, sids=(A, B))).status == FS.TRACKING_MULTI_VIEW
