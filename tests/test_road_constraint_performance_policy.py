from unittest.mock import patch

from shapely.geometry import LineString, Point

from cwr_worldgen import road_constraint_performance_policy as perf
from cwr_worldgen import terrain_solver as terrain


def test_milestone9_installs_road_constraint_performance_policy() -> None:
    assert terrain._line_geometry is perf._fast_line_geometry
    assert terrain._candidate_cells is perf._fast_candidate_cells
    assert terrain.road_span_has_in_game_water is perf._fast_road_span_has_in_game_water
    assert terrain._road_corridor_intersects_mask is perf._fast_road_corridor_intersects_mask


def test_vectorized_cell_batch_matches_scalar_shapely_geometry() -> None:
    scalar_line = LineString([(0.0, 0.0), (75.0, 25.0), (125.0, 25.0)])
    fast_line = perf._FastLine(scalar_line, {"highway": "residential"})
    corridor = fast_line.buffer(28.0, cap_style=2, join_style=2)

    indices = tuple(perf._fast_candidate_cells(corridor.bounds, 8, 25.0))
    assert indices
    assert fast_line._batch is not None
    assert fast_line._batch.distances is None
    assert fast_line._batch.projections is None
    assert fast_line._batch.covered is None

    for index in indices:
        x, z = terrain._cell_center(index, 8, 25.0)
        point = Point(x, z)
        assert abs(fast_line.distance(point) - scalar_line.distance(point)) < 1.0e-9
        assert abs(fast_line.project(point) - scalar_line.project(point)) < 1.0e-9
        assert corridor.covers(point) == scalar_line.buffer(
            28.0, cap_style=2, join_style=2
        ).covers(point)

    assert fast_line._batch.distances is not None
    assert fast_line._batch.projections is not None
    assert fast_line._batch.covered is not None


def test_ordinary_at_grade_road_skips_bridge_water_probes() -> None:
    assert not perf._needs_bridge_water_test({"highway": "residential"})
    assert perf._needs_bridge_water_test({"highway": "primary", "bridge": "yes"})
    assert perf._needs_bridge_water_test({"highway": "primary", "layer": "1"})

    token = perf._ACTIVE_NEEDS_WATER_TEST.set(False)
    try:
        with patch.object(
            perf,
            "_ORIGINAL_ROAD_SPAN_WATER_TEST",
            side_effect=AssertionError("ordinary road performed a span-water probe"),
        ), patch.object(
            perf,
            "_ORIGINAL_CORRIDOR_WATER_TEST",
            side_effect=AssertionError("ordinary road performed a corridor-water probe"),
        ):
            assert not perf._fast_road_span_has_in_game_water((), ())
            assert not perf._fast_road_corridor_intersects_mask(None, (), None, 6.0)
    finally:
        perf._ACTIVE_NEEDS_WATER_TEST.reset(token)
