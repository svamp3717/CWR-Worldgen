from __future__ import annotations

from cwr_worldgen import bridge_or_causeway_terrain_policy as bridge_terrain
from cwr_worldgen import bridge_postsolve_water_performance_policy as perf
from cwr_worldgen import bridge_source_water_policy as source


def _polygon(min_x, min_z, max_x, max_z, *, holes=()):
    outer = (
        (float(min_x), float(min_z)),
        (float(max_x), float(min_z)),
        (float(max_x), float(max_z)),
        (float(min_x), float(max_z)),
        (float(min_x), float(min_z)),
    )
    hole_rings = tuple(
        (
            (float(x0), float(z0)),
            (float(x1), float(z0)),
            (float(x1), float(z1)),
            (float(x0), float(z1)),
            (float(x0), float(z0)),
        )
        for x0, z0, x1, z1 in holes
    )
    return source._ProjectedWaterPolygon(
        outer=outer,
        holes=hole_rings,
        bounds=(float(min_x), float(min_z), float(max_x), float(max_z)),
    )


def _context(*polygons):
    return source._SourceContext(object(), object(), tuple(polygons))


def test_runtime_uses_vectorized_postsolve_water_runs() -> None:
    assert bridge_terrain._mapped_water_runs is perf._fast_mapped_water_runs


def test_vectorized_runs_match_historical_multiple_water_intervals() -> None:
    context = _context(
        _polygon(100.0, -20.0, 200.0, 20.0),
        _polygon(700.0, -20.0, 800.0, 20.0),
    )
    points = ((0.0, 0.0), (1000.0, 0.0))

    expected = perf._ORIGINAL_MAPPED_WATER_RUNS(points, context)
    actual = perf._fast_mapped_water_runs(points, context)

    assert len(expected) == len(actual) == 2
    for expected_run, actual_run in zip(expected, actual):
        assert abs(actual_run[0] - expected_run[0]) < 1.0e-9
        assert abs(actual_run[1] - expected_run[1]) < 1.0e-9


def test_vectorized_runs_preserve_dry_hole_as_separate_runs() -> None:
    context = _context(
        _polygon(100.0, -50.0, 900.0, 50.0, holes=((400.0, -20.0, 600.0, 20.0),))
    )
    points = ((0.0, 0.0), (1000.0, 0.0))

    expected = perf._ORIGINAL_MAPPED_WATER_RUNS(points, context)
    actual = perf._fast_mapped_water_runs(points, context)

    assert len(expected) == len(actual) == 2
    for expected_run, actual_run in zip(expected, actual):
        assert abs(actual_run[0] - expected_run[0]) < 1.0e-9
        assert abs(actual_run[1] - expected_run[1]) < 1.0e-9
