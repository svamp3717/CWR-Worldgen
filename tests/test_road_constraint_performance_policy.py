from types import SimpleNamespace
from unittest.mock import patch

from shapely.geometry import LineString, Point

from cwr_worldgen import building_pad_performance_policy as building_perf
from cwr_worldgen import road_constraint_pathology_policy as pathology
from cwr_worldgen import road_constraint_performance_policy as perf
from cwr_worldgen import terrain_solver as terrain


def test_milestone9_installs_road_constraint_performance_policy() -> None:
    assert terrain._line_geometry is perf._fast_line_geometry
    assert terrain._candidate_cells is building_perf._candidate_cells
    assert building_perf._ORIGINAL_CANDIDATE_CELLS is pathology._pathological_candidate_cells
    assert pathology._ORIGINAL_INNER_CANDIDATE_CELLS is perf._fast_candidate_cells
    assert terrain.Point is building_perf._point
    assert building_perf._ORIGINAL_POINT is perf._fast_point
    assert terrain._profile_height is perf._fast_profile_height
    assert terrain.road_span_has_in_game_water is perf._fast_road_span_has_in_game_water
    assert terrain._road_corridor_intersects_mask is perf._fast_road_corridor_intersects_mask


def test_vectorized_cell_batch_matches_scalar_shapely_geometry() -> None:
    scalar_line = LineString([(0.0, 0.0), (75.0, 25.0), (125.0, 25.0)])
    fast_line = perf._FastLine(scalar_line, {"highway": "residential"})
    corridor = fast_line.buffer(28.0, cap_style=2, join_style=2)

    scalar_corridor = scalar_line.buffer(28.0, cap_style=2, join_style=2)
    all_indices = tuple(perf._ORIGINAL_CANDIDATE_CELLS(scalar_corridor.bounds, 8, 25.0))
    expected_indices = tuple(
        index
        for index in all_indices
        if scalar_line.distance(Point(terrain._cell_center(index, 8, 25.0))) <= 28.0 + 1.0e-9
    )
    indices = tuple(perf._fast_candidate_cells(corridor.bounds, 8, 25.0))

    assert indices == expected_indices
    assert len(indices) <= len(all_indices)
    assert indices
    assert fast_line._batch is not None
    assert fast_line._batch.distances is not None
    assert fast_line._batch.projections is None
    assert fast_line._batch.covered is None

    for index in indices:
        x, z = terrain._cell_center(index, 8, 25.0)
        point = Point(x, z)
        assert abs(fast_line.distance(point) - scalar_line.distance(point)) < 1.0e-9
        assert abs(fast_line.project(point) - scalar_line.project(point)) < 1.0e-9
        assert corridor.covers(point) == scalar_corridor.covers(point)

    assert fast_line._batch.projections is not None
    assert fast_line._batch.covered is not None


def test_large_diagonal_corridor_uses_segment_candidates_without_eager_buffer() -> None:
    scalar_line = LineString([(0.0, 0.0), (5000.0, 5000.0)])
    fast_line = perf._FastLine(scalar_line, {"highway": "primary"})
    radius = 24.0
    cells = 256
    cell_size = 25.0
    corridor = fast_line.buffer(radius, cap_style=2, join_style=2)

    assert corridor._geometry is None
    full_count = perf._candidate_count(corridor.bounds, cells, cell_size)
    assert full_count > perf._SEGMENT_BROAD_PHASE_THRESHOLD

    original_helper = perf._candidate_indices_for_bounds
    seen_counts: list[int] = []

    def recording_helper(bounds, helper_cells, helper_cell_size):
        seen_counts.append(perf._candidate_count(bounds, helper_cells, helper_cell_size))
        return original_helper(bounds, helper_cells, helper_cell_size)

    with patch.object(perf, "_candidate_indices_for_bounds", side_effect=recording_helper):
        indices = tuple(perf._fast_candidate_cells(corridor.bounds, cells, cell_size))

    assert corridor._geometry is None
    assert seen_counts
    assert max(seen_counts) < full_count

    scalar_corridor = scalar_line.buffer(radius, cap_style=2, join_style=2)
    expected = tuple(
        index
        for index in perf._ORIGINAL_CANDIDATE_CELLS(scalar_corridor.bounds, cells, cell_size)
        if scalar_line.distance(Point(terrain._cell_center(index, cells, cell_size)))
        <= radius + 1.0e-9
    )
    assert indices == expected

    corridor.covers(Point(2500.0, 2500.0))
    assert corridor._geometry is not None


def test_long_source_segment_is_split_before_grid_rectangle_expansion() -> None:
    line = LineString([(100.0, 100.0), (12_000.0, 12_000.0)])
    radius = 30.0
    cells = 1024
    cell_size = 25.0
    bounds = (
        line.bounds[0] - radius,
        line.bounds[1] - radius,
        line.bounds[2] + radius,
        line.bounds[3] + radius,
    )
    full_count = perf._candidate_count(bounds, cells, cell_size)
    original_helper = perf._candidate_indices_for_bounds
    seen_counts: list[int] = []

    def recording_helper(piece_bounds, helper_cells, helper_cell_size):
        seen_counts.append(
            perf._candidate_count(piece_bounds, helper_cells, helper_cell_size)
        )
        return original_helper(piece_bounds, helper_cells, helper_cell_size)

    with patch.object(perf, "_candidate_indices_for_bounds", side_effect=recording_helper):
        candidates = perf._segment_candidate_indices(
            line,
            radius,
            cells,
            cell_size,
        )

    assert candidates.size
    assert seen_counts
    assert max(seen_counts) < full_count // 20


def test_grid_cell_points_and_profile_interpolation_stay_out_of_scalar_shapely_loop() -> None:
    scalar_line = LineString([(0.0, 0.0), (100.0, 50.0), (175.0, 50.0)])
    fast_line = perf._FastLine(scalar_line, {"highway": "residential"})
    corridor = fast_line.buffer(32.0, cap_style=2, join_style=2)
    indices = tuple(perf._fast_candidate_cells(corridor.bounds, 10, 25.0))
    assert indices

    index = indices[len(indices) // 2]
    centre = terrain.Point(terrain._cell_center(index, 10, 25.0))
    assert isinstance(centre, perf._CellPoint)

    along = fast_line.project(centre)
    assert isinstance(along, perf._ProjectedDistance)

    distances = [0.0, 50.0, 100.0, scalar_line.length]
    heights = [10.0, 15.0, 20.0, 30.0]
    vectorized_value = terrain._profile_height(along, distances, heights)
    scalar_value = perf._ORIGINAL_PROFILE_HEIGHT(float(along), distances, heights)
    assert abs(vectorized_value - scalar_value) < 1.0e-9

    batch = fast_line._batch
    assert batch is not None
    assert (id(distances), id(heights)) in batch.profile_values


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
        ):
            assert not perf._fast_road_span_has_in_game_water((), ())
            assert not perf._fast_road_corridor_intersects_mask(None, (), None, 6.0)
    finally:
        perf._ACTIVE_NEEDS_WATER_TEST.reset(token)


def test_bridge_water_corridor_matches_scalar_covers_semantics() -> None:
    scalar_line = LineString([(25.0, 25.0), (125.0, 75.0)])
    fast_line = perf._FastLine(scalar_line, {"highway": "primary", "bridge": "yes"})
    spec = SimpleNamespace(cells=8, cell_size=25.0)
    mask = [False] * (spec.cells * spec.cells)
    mask[2 * spec.cells + 3] = True

    radius = max(12.0 * 0.5, spec.cell_size * 0.35)
    corridor = scalar_line.buffer(radius, cap_style=2, join_style=2)
    expected = corridor.covers(Point(3 * spec.cell_size, 2 * spec.cell_size))

    token = perf._ACTIVE_NEEDS_WATER_TEST.set(True)
    try:
        assert perf._fast_road_corridor_intersects_mask(
            fast_line, mask, spec, 12.0
        ) == expected
    finally:
        perf._ACTIVE_NEEDS_WATER_TEST.reset(token)
