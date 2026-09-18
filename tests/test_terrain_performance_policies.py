from types import SimpleNamespace
from unittest.mock import patch

import pytest
from shapely.geometry import LineString, Point, box

from cwr_worldgen import building_pad_performance_policy as building_perf
from cwr_worldgen import terrain_postprocess_performance_policy as post_perf
from cwr_worldgen import terrain_solver as terrain


def test_building_cell_intersections_match_scalar_polygons() -> None:
    cells = 10
    cell_size = 25.0
    geometry = box(42.0, 37.0, 138.0, 117.0).buffer(9.0, join_style=2)

    indices = tuple(building_perf._candidate_cells(geometry.bounds, cells, cell_size))
    assert indices
    active = building_perf._ACTIVE_BUILDING_BATCH.get()
    assert active is not None

    for index in indices:
        proxy = terrain._cell_polygon(index, cells, cell_size)
        scalar = building_perf._ORIGINAL_CELL_POLYGON(index, cells, cell_size)
        assert proxy.intersects(geometry) == scalar.intersects(geometry)

    # Building transitions still need a genuine Shapely Point for
    # hard_pad.distance(centre); the policy must not feed that API a proxy.
    x, z = terrain._cell_center(indices[0], cells, cell_size)
    assert isinstance(terrain.Point((x, z)), Point)


def test_cached_road_slope_sampling_matches_scalar_formula() -> None:
    cells = 8
    cell_size = 25.0
    elevations = tuple(
        float(x * 1.5 + z * 2.25)
        for z in range(cells)
        for x in range(cells)
    )
    line = LineString([(10.0, 12.0), (82.0, 37.0), (156.0, 121.0)])
    feature = SimpleNamespace(tags={"highway": "residential"})
    dataset = SimpleNamespace(roads=(feature,))
    projection = object()
    spec = SimpleNamespace(cells=cells, cell_size=cell_size)

    spacing = max(2.0, cell_size * 0.35)
    count = max(1, int(__import__("math").ceil(line.length / spacing)))
    previous = line.interpolate(0.0)
    previous_h = terrain._sample_elevation(
        elevations, cells, cell_size, previous.x, previous.y
    )
    expected = 0.0
    for step in range(1, count + 1):
        point = line.interpolate(line.length * step / count)
        height = terrain._sample_elevation(
            elevations, cells, cell_size, point.x, point.y
        )
        travel = max(0.01, point.distance(previous))
        expected = max(expected, abs(height - previous_h) / travel * 100.0)
        previous = point
        previous_h = height

    post_perf._ROAD_PLAN_CACHE.clear()
    with patch.object(post_perf, "_raw_line", return_value=line):
        actual = post_perf._fast_road_slope_percent(
            elevations, dataset, projection, spec
        )
        # A second call exercises the cached sample plan rather than rebuilding
        # the projected road geometry.
        again = post_perf._fast_road_slope_percent(
            elevations, dataset, projection, spec
        )

    assert actual == pytest.approx(expected, abs=1.0e-12)
    assert again == pytest.approx(expected, abs=1.0e-12)
    assert len(post_perf._ROAD_PLAN_CACHE) == 1


def test_cached_downhill_repair_matches_original_algorithm() -> None:
    cells = 7
    cell_size = 20.0
    original = tuple(
        30.0 - z * 0.6 + x * 0.04
        for z in range(cells)
        for x in range(cells)
    )
    # Put an uphill bump in the middle of a nominally downhill stream so both
    # coarse and bilinear repair passes have actual work to perform.
    result_base = list(original)
    result_base[3 * cells + 3] += 4.0
    result_base[4 * cells + 3] += 2.5

    line = LineString([(60.0, 5.0), (60.0, 125.0)])
    feature = SimpleNamespace(tags={"waterway": "stream"})
    dataset = SimpleNamespace(watercourses=(feature,))
    projection = object()
    spec = SimpleNamespace(
        cells=cells,
        cell_size=cell_size,
        watercourse_minimum_gradient_percent=0.25,
        maximum_grade_adjustment=8.0,
    )

    expected = list(result_base)
    actual = list(result_base)
    expected_field = terrain._ConstraintField.create(cells * cells)
    actual_field = terrain._ConstraintField.create(cells * cells)

    with patch.object(terrain, "_line_geometry", return_value=line):
        post_perf._ORIGINAL_ENFORCE_DOWNHILL_WATERCOURSES(
            expected,
            original,
            dataset,
            projection,
            spec,
            expected_field,
        )
    with patch.object(post_perf, "_raw_line", return_value=line):
        post_perf._fast_enforce_downhill_watercourses(
            actual,
            original,
            dataset,
            projection,
            spec,
            actual_field,
        )

    assert actual == pytest.approx(expected, abs=1.0e-12)
    assert actual_field.categories == expected_field.categories


def test_postprocess_policy_is_installed_for_milestone9() -> None:
    assert terrain._road_slope_percent is post_perf._fast_road_slope_percent
    assert (
        terrain._enforce_downhill_watercourses
        is post_perf._fast_enforce_downhill_watercourses
    )
