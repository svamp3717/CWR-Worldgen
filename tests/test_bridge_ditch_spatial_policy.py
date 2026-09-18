from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_ditch_spatial_policy as policy
from cwr_worldgen import osm
from cwr_worldgen import playability
from cwr_worldgen import terrain_solver


class _Projection:
    def __init__(self) -> None:
        self.calls = 0

    def to_world(self, point):
        self.calls += 1
        return float(point[0]), float(point[1])


def _line(points, *, waterway=None, bridge=None):
    tags = {}
    if waterway is not None:
        tags["waterway"] = waterway
    if bridge is not None:
        tags["bridge"] = bridge
    return SimpleNamespace(points=tuple(points), tags=tags)


def _dataset(*watercourses):
    return SimpleNamespace(watercourses=tuple(watercourses))


def test_runtime_uses_indexed_bridge_ditch_predicate() -> None:
    assert osm.road_bridge_crosses_ditch_only is policy._indexed_road_bridge_crosses_ditch_only
    assert terrain_solver.road_bridge_crosses_ditch_only is policy._indexed_road_bridge_crosses_ditch_only
    assert playability.road_bridge_crosses_ditch_only is policy._indexed_road_bridge_crosses_ditch_only


def test_indexed_ditch_classification_matches_historical_semantics() -> None:
    road = _line(((0.0, 0.0), (1000.0, 0.0)), bridge="yes")
    ditch = _line(((250.0, -50.0), (250.0, 50.0)), waterway="ditch")
    distant_stream = _line(((5000.0, -50.0), (5000.0, 50.0)), waterway="stream")
    projection = _Projection()
    dataset = _dataset(ditch, distant_stream)

    expected = policy._ORIGINAL_DITCH_TEST(road, dataset, projection)
    policy._INDEX_CACHE.clear()
    actual = policy._indexed_road_bridge_crosses_ditch_only(road, dataset, projection)
    assert actual == expected is True

    crossing_stream = _line(((750.0, -50.0), (750.0, 50.0)), waterway="stream")
    projection = _Projection()
    dataset = _dataset(ditch, crossing_stream, distant_stream)
    expected = policy._ORIGINAL_DITCH_TEST(road, dataset, projection)
    policy._INDEX_CACHE.clear()
    actual = policy._indexed_road_bridge_crosses_ditch_only(road, dataset, projection)
    assert actual == expected is False


def test_non_explicit_road_is_never_treated_as_ditch_bridge() -> None:
    road = _line(((0.0, 0.0), (100.0, 0.0)))
    ditch = _line(((50.0, -10.0), (50.0, 10.0)), waterway="ditch")
    projection = _Projection()
    dataset = _dataset(ditch)
    policy._INDEX_CACHE.clear()

    assert policy._indexed_road_bridge_crosses_ditch_only(road, dataset, projection) is False
    assert projection.calls == 0


def test_spatial_index_prunes_distant_watercourse_segments() -> None:
    road = _line(((0.0, 0.0), (100.0, 0.0)), bridge="yes")
    near = _line(((50.0, -20.0), (50.0, 20.0)), waterway="ditch")
    distant = tuple(
        _line(
            ((10_000.0 + index * 20.0, 10_000.0), (10_000.0 + index * 20.0, 10_050.0)),
            waterway="ditch",
        )
        for index in range(500)
    )
    projection = _Projection()
    dataset = _dataset(near, *distant)
    policy._INDEX_CACHE.clear()

    original_distance = osm._segment_distance_squared
    with patch.object(osm, "_segment_distance_squared", wraps=original_distance) as distance:
        assert policy._indexed_road_bridge_crosses_ditch_only(road, dataset, projection)

    # The historical implementation would test all 501 watercourse segments.
    assert distance.call_count <= 4


def test_watercourse_projection_is_cached_across_bridge_queries() -> None:
    road = _line(((0.0, 0.0), (100.0, 0.0)), bridge="yes")
    watercourses = tuple(
        _line(((1000.0 + index * 50.0, 0.0), (1000.0 + index * 50.0, 100.0)), waterway="ditch")
        for index in range(100)
    )
    projection = _Projection()
    dataset = _dataset(*watercourses)
    policy._INDEX_CACHE.clear()

    policy._indexed_road_bridge_crosses_ditch_only(road, dataset, projection)
    first_calls = projection.calls
    policy._indexed_road_bridge_crosses_ditch_only(road, dataset, projection)

    assert first_calls == 2 + 2 * len(watercourses)
    assert projection.calls - first_calls == 2
