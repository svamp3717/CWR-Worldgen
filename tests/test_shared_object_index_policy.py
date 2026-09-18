from types import SimpleNamespace
from unittest.mock import patch

from shapely.geometry import Polygon

from cwr_worldgen import osm
from cwr_worldgen import shared_object_index_policy as perf
from cwr_worldgen import vegetation_clearance_policy as vegetation
from cwr_worldgen.model import WorldObject


def _objects():
    return (
        WorldObject(1, r"data3d\str smrk.p3d", 10.0, 5.0, 10.0, 0.0),
        WorldObject(2, r"data3d\str smrk.p3d", 90.0, 5.0, 90.0, 0.0),
        WorldObject(3, r"data3d\sil25.p3d", 20.0, 5.0, 20.0, 0.0),
        WorldObject(4, r"O\Hous\zidka01.p3d", 40.0, 5.0, 40.0, 0.0),
    )


def test_object_index_is_identity_cached_and_groups_models() -> None:
    objects = _objects()
    first = perf.object_index(objects)
    second = perf.object_index(objects)
    assert first is second
    assert len(first.model_groups[r"data3d\str smrk.p3d"]) == 2
    assert first.xs.tolist() == [10.0, 90.0, 20.0, 40.0]
    assert 0 in first.nearby_indices(10.0, 10.0, 5.0)


def test_indexed_vegetation_filter_matches_previous_filter() -> None:
    objects = _objects()
    spec = SimpleNamespace(name="test_world")
    runway = Polygon(((0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)))
    with patch.object(vegetation, "_runway_clear_shapes", return_value=(runway,)), patch.object(
        vegetation, "_sports_clear_shapes", return_value=()
    ):
        expected = perf._ORIGINAL_VEGETATION_FILTER(objects, None, None, spec)
        actual = perf._indexed_filter_vegetation_objects(objects, None, None, spec)
    assert actual == expected
    assert [obj.object_id for obj in actual[0]] == [2, 3, 4]


def test_indexed_vegetation_audit_matches_previous_audit() -> None:
    objects = _objects()
    cells = 8
    elevations = (5.0,) * (cells * cells)
    spec = SimpleNamespace(
        cells=cells,
        cell_size=25.0,
        name="test_world",
        forest_single_tree_maximum_float=0.15,
        forest_cluster_tree_maximum_float=0.20,
        forest_cluster_bush_maximum_float=0.60,
        forest_single_tree_model=r"data3d\str smrk.p3d",
        forest_hillside_tree_model="",
        forest_roadside_tree_model="",
        forest_roadside_tree_models=(r"data3d\str smrk.p3d",),
    )
    expected = perf._ORIGINAL_VEGETATION_AUDIT(objects, elevations, spec)
    actual = perf._indexed_vegetation_grounding_audit(objects, elevations, spec)
    assert actual == expected


def test_shared_object_index_policy_is_live() -> None:
    assert vegetation.filter_vegetation_objects is perf._indexed_filter_vegetation_objects
    assert osm._audit_vegetation_grounding is perf._indexed_vegetation_grounding_audit
