from unittest.mock import patch

from cwr_worldgen import object_sampling_cache_policy as perf
from cwr_worldgen import object_stage_parallel_policy as stage_parallel
from cwr_worldgen import osm
from cwr_worldgen import playability
from cwr_worldgen import road_finish_parallel_policy as road_finish


def _terrain(cells: int = 6) -> tuple[float, ...]:
    return tuple(
        3.0 + (index // cells) * 0.4 + (index % cells) * 0.15
        for index in range(cells * cells)
    )


def test_object_sampling_cache_policy_is_installed_under_multicore_wrappers() -> None:
    assert osm._sample_elevation is stage_parallel._cached_osm_sample
    assert stage_parallel._BASE_OSM_SAMPLE is perf._fast_osm_sample
    assert osm._triangle_elevation_bounds is perf._fast_triangle_elevation_bounds
    assert osm._terrain_axis_breakpoints is perf._fast_terrain_axis_breakpoints
    assert osm._terrain_patch_centres is perf._fast_terrain_patch_centres
    # Final stock-road transforms now sample terrain in one NumPy batch, so the
    # normal scalar helper can point straight at the underlying fast cache again.
    assert playability._sample_elevation is road_finish._BASE_SAMPLE
    assert road_finish._BASE_SAMPLE is perf._fast_playability_sample
    assert playability._PolylineMeasure.point is perf._fast_measure_point


def test_triangle_bounds_cache_preserves_result_and_reuses_tuple_sample() -> None:
    elevations = _terrain()
    perf._ELEVATION_CACHES.clear()
    expected = perf._ORIGINAL_OSM_TRIANGLE(elevations, 6, 25.0, 62.5, 87.5)
    original = perf._ORIGINAL_OSM_TRIANGLE
    with patch.object(perf, "_ORIGINAL_OSM_TRIANGLE", wraps=original) as wrapped:
        first = perf._fast_triangle_elevation_bounds(elevations, 6, 25.0, 62.5, 87.5)
        second = perf._fast_triangle_elevation_bounds(elevations, 6, 25.0, 62.5, 87.5)
    assert first == expected
    assert second == expected
    assert wrapped.call_count == 1


def test_mutable_elevation_arrays_bypass_cache() -> None:
    elevations = list(_terrain())
    perf._ELEVATION_CACHES.clear()
    original = perf._ORIGINAL_OSM_SAMPLE
    with patch.object(perf, "_ORIGINAL_OSM_SAMPLE", wraps=original) as wrapped:
        first = perf._fast_osm_sample(elevations, 6, 25.0, 50.0, 50.0)
        elevations[2 * 6 + 2] += 4.0
        second = perf._fast_osm_sample(elevations, 6, 25.0, 50.0, 50.0)
    assert second != first
    assert wrapped.call_count == 2


def test_regular_forest_breakpoint_tables_are_cached() -> None:
    perf._cached_terrain_axis_breakpoints.cache_clear()
    perf._cached_terrain_patch_centres.cache_clear()

    axis_original = perf._ORIGINAL_TERRAIN_AXIS_BREAKPOINTS
    patch_original = perf._ORIGINAL_TERRAIN_PATCH_CENTRES
    with patch.object(
        perf, "_ORIGINAL_TERRAIN_AXIS_BREAKPOINTS", wraps=axis_original
    ) as axis_wrapped, patch.object(
        perf, "_ORIGINAL_TERRAIN_PATCH_CENTRES", wraps=patch_original
    ) as patch_wrapped:
        axis_first = perf._fast_terrain_axis_breakpoints(25.0, 75.0, 256, 25.0)
        axis_second = perf._fast_terrain_axis_breakpoints(25.0, 75.0, 256, 25.0)
        patch_first = perf._fast_terrain_patch_centres(25.0, 75.0, 256, 25.0)
        patch_second = perf._fast_terrain_patch_centres(25.0, 75.0, 256, 25.0)

    assert axis_first == axis_second == axis_original(25.0, 75.0, 256, 25.0)
    assert patch_first == patch_second == patch_original(25.0, 75.0, 256, 25.0)
    assert axis_wrapped.call_count == 1
    assert patch_wrapped.call_count == 1


def test_stock_road_measure_point_cache_reuses_piece_endpoints() -> None:
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (30.0, 0.0), (55.0, 12.0), (90.0, 12.0))
    )
    perf._MEASURE_CACHES.clear()
    expected = perf._ORIGINAL_MEASURE_POINT(measure, 30.0)
    original = perf._ORIGINAL_MEASURE_POINT
    with patch.object(perf, "_ORIGINAL_MEASURE_POINT", wraps=original) as wrapped:
        first = perf._fast_measure_point(measure, 30.0)
        second = perf._fast_measure_point(measure, 30.0)
    assert first == expected
    assert second == expected
    assert wrapped.call_count == 1


def test_playability_endpoint_elevation_cache_preserves_sampling() -> None:
    elevations = _terrain()
    perf._ELEVATION_CACHES.clear()
    expected = perf._ORIGINAL_PLAYABILITY_SAMPLE(elevations, 6, 25.0, 75.0, 50.0)
    original = perf._ORIGINAL_PLAYABILITY_SAMPLE
    with patch.object(perf, "_ORIGINAL_PLAYABILITY_SAMPLE", wraps=original) as wrapped:
        first = perf._fast_playability_sample(elevations, 6, 25.0, 75.0, 50.0)
        second = perf._fast_playability_sample(elevations, 6, 25.0, 75.0, 50.0)
    assert first == expected
    assert second == expected
    assert wrapped.call_count == 1
