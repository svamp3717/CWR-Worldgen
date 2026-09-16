from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from shapely.geometry import LineString

from cwr_worldgen import road_profile_performance_policy as policy
from cwr_worldgen import terrain_solver


def _terrain(cells: int) -> tuple[float, ...]:
    values = []
    for z in range(cells):
        for x in range(cells):
            values.append(
                100.0
                + x * 0.43
                + z * 0.71
                + math.sin(x * 0.31) * 2.4
                + math.cos(z * 0.23) * 1.7
            )
    return tuple(values)


def test_numpy_arc_length_positions_match_shapely_interpolation() -> None:
    line = LineString(
        (
            (10.0, 15.0),
            (80.0, 23.0),
            (80.0, 23.0),  # zero-length source segment must be harmless
            (135.0, 105.0),
            (260.0, 150.0),
            (420.0, 390.0),
        )
    )
    distances = np.linspace(0.0, line.length, 97)
    xs, zs = policy._line_positions(line, distances)

    expected = tuple(line.interpolate(float(distance)) for distance in distances)
    np.testing.assert_allclose(xs, [point.x for point in expected], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(zs, [point.y for point in expected], rtol=0.0, atol=1.0e-10)


def test_vector_profile_matches_original_grade_limited_profile() -> None:
    cells = 64
    cell_size = 25.0
    spec = SimpleNamespace(cells=cells, cell_size=cell_size)
    elevations = _terrain(cells)
    line = LineString(
        (
            (37.0, 42.0),
            (215.0, 83.0),
            (390.0, 330.0),
            (810.0, 415.0),
            (1210.0, 980.0),
        )
    )

    expected_distances, expected_heights = policy._ORIGINAL_PROFILE(
        line, elevations, spec, 11.5
    )
    actual_distances, actual_heights = policy._fast_profile(
        line, elevations, spec, 11.5
    )

    assert actual_distances == expected_distances
    np.testing.assert_allclose(actual_heights, expected_heights, rtol=0.0, atol=1.0e-10)


def test_vector_cross_slope_matches_original_sampling_formula() -> None:
    cells = 64
    cell_size = 25.0
    spec = SimpleNamespace(cells=cells, cell_size=cell_size)
    elevations = _terrain(cells)
    line = LineString(
        (
            (55.0, 70.0),
            (220.0, 115.0),
            (410.0, 360.0),
            (910.0, 505.0),
            (1275.0, 1150.0),
        )
    )
    distances, _heights = policy._ORIGINAL_PROFILE(line, elevations, spec, 12.0)

    expected = policy._ORIGINAL_CROSS_SLOPE_PROFILE(
        line, elevations, spec, distances, 8.0
    )
    actual = policy._fast_cross_slope_profile(
        line, elevations, spec, distances, 8.0
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-9)


def test_vector_profile_policy_is_live_after_package_startup() -> None:
    assert terrain_solver._profile is policy._fast_profile
    assert terrain_solver._road_cross_slope_profile is policy._fast_cross_slope_profile
