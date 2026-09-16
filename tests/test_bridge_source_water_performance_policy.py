from __future__ import annotations

import numpy as np

from cwr_worldgen import bridge_source_water_performance_policy as perf
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


def test_runtime_uses_vectorized_source_water_interval() -> None:
    assert source._source_mapped_water_interval is perf._fast_source_mapped_water_interval


def test_vectorized_interval_matches_historical_sampling_and_refinement() -> None:
    context = _context(_polygon(4500.0, -50.0, 5500.0, 50.0))
    points = ((0.0, 0.0), (10_000.0, 0.0))

    expected = perf._ORIGINAL_SOURCE_INTERVAL(points, context)
    actual = perf._fast_source_mapped_water_interval(points, context)

    assert expected is not None
    assert actual is not None
    assert abs(actual[0] - expected[0]) < 1.0e-9
    assert abs(actual[1] - expected[1]) < 1.0e-9


def test_vectorized_water_classification_preserves_outer_and_hole_boundaries() -> None:
    polygon = _polygon(0.0, 0.0, 100.0, 100.0, holes=((40.0, 40.0, 60.0, 60.0),))
    context = _context(polygon)
    indexed = perf._water_index(context).polygons
    points = (
        (0.0, 50.0),
        (20.0, 20.0),
        (40.0, 50.0),
        (50.0, 50.0),
        (60.0, 50.0),
        (100.0, 50.0),
        (120.0, 50.0),
    )
    xs = np.asarray([point[0] for point in points], dtype=np.float64)
    zs = np.asarray([point[1] for point in points], dtype=np.float64)

    expected = np.asarray(
        [source._point_in_water(point, context.water) for point in points],
        dtype=np.bool_,
    )
    actual = perf._vectorized_point_in_water(xs, zs, indexed)

    assert np.array_equal(actual, expected)


def test_water_spatial_index_prunes_distant_polygons() -> None:
    near = _polygon(490.0, -20.0, 510.0, 20.0)
    distant = tuple(
        _polygon(10_000.0 + index * 100.0, 10_000.0, 10_020.0 + index * 100.0, 10_020.0)
        for index in range(200)
    )
    context = _context(near, *distant)
    cleaned = ((0.0, 0.0), (1000.0, 0.0))

    candidates = perf._candidate_polygons(context, cleaned)

    assert len(candidates) == 1
    assert candidates[0].source is near
