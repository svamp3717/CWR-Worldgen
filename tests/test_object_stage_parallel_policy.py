from types import SimpleNamespace

import numpy as np

from cwr_worldgen import object_stage_parallel_policy as perf
from cwr_worldgen import road_finish_parallel_policy as road_finish


def test_exact_worker_sampler_matches_live_scalar_sampler() -> None:
    cells = 5
    cell_size = 20.0
    elevations = tuple(
        2.0 + (index // cells) * 0.7 + (index % cells) * 0.15
        for index in range(cells * cells)
    )
    for x, z in ((0.0, 0.0), (7.0, 14.0), (39.5, 44.0), (79.0, 61.0), (120.0, -3.0)):
        expected = perf._BASE_PLAYABILITY_SAMPLE(
            elevations, cells, cell_size, x, z
        )
        actual = perf._sample_exact(elevations, cells, cell_size, x, z)
        assert np.isclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_object_cache_wrappers_return_precomputed_values() -> None:
    elevations = (0.0,) * 16
    context = perf._ObjectContext(elevations, 4, 10.0)
    polygon = ((1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0))
    context.polygon_extrema[polygon] = (3.0, 7.0)
    context.square_samples[(5.0, 5.0, 6.0)] = (1.0, 2.0, 3.0)
    context.samples[(5.0, 5.0)] = 4.25

    points = ((0.0, 0.0), (12.0, 0.0))
    line_key = perf._line_key(points, 6.0, 0.0)
    line_value = ((3.0, 0.0, 90.0, 6.0, 0.0, 0.0, 6.0, 0.0),)
    context.line_chunks[line_key] = line_value

    grid_key = perf._grid_key(polygon, 5.0, "seed", 0.2, 30.0)
    grid_value = ((2.0, 2.0, 15.0),)
    context.polygon_grids[grid_key] = grid_value

    variant = SimpleNamespace(name="cached_variant")
    cluster_key = perf._cluster_key(
        variant, 5.0, 5.0, 20.0, False, 0.0, 10.0, 1.0, 1.0, 0.03, True
    )
    cluster_value = ("model.p3d", 5.0, 2.0, 5.0, 20.0, "cached_variant", 0.1, 0.0, 0.0)
    context.clusters[cluster_key] = cluster_value

    token = perf._OBJECT_CONTEXT.set(context)
    try:
        assert perf._cached_polygon_extrema(elevations, 4, 10.0, polygon) == (3.0, 7.0)
        assert perf._cached_square_samples(elevations, 4, 10.0, 5.0, 5.0, 6.0) == (1.0, 2.0, 3.0)
        assert perf._cached_osm_sample(elevations, 4, 10.0, 5.0, 5.0) == 4.25
        assert perf._cached_line_chunks(points, 6.0) == line_value
        assert perf._cached_polygon_grid(
            polygon, 5.0, "seed", jitter_fraction=0.2, heading_jitter_degrees=30.0
        ) == grid_value
        assert perf._cached_place_cluster_at(
            variant=variant,
            elevations=elevations,
            raster=None,
            road_corridors=(),
            spec=None,
            x=5.0,
            z=5.0,
            heading=20.0,
            require_forest=False,
            minimum_forest_fraction=0.0,
            maximum_relief=10.0,
            maximum_burial=1.0,
            maximum_float=1.0,
            clearance=0.03,
            avoid_roads=True,
        ) == cluster_value
    finally:
        perf._OBJECT_CONTEXT.reset(token)


def test_building_submerged_cache_uses_stable_plan_identity() -> None:
    elevations = (0.0,) * 16
    plan = SimpleNamespace(
        osm_key="way/12",
        geometry_index=3,
        support_polygon=((1.0, 1.0), (4.0, 1.0), (4.0, 4.0)),
    )
    context = perf._ObjectContext(elevations, 4, 10.0)
    context.building_submerged[perf._building_key(plan)] = True
    token = perf._OBJECT_CONTEXT.set(context)
    try:
        assert perf._cached_building_submerged(plan, elevations, None, None) is True
    finally:
        perf._OBJECT_CONTEXT.reset(token)


def test_road_finish_batch_sampler_matches_scalar_sampler() -> None:
    cells = 6
    cell_size = 25.0
    elevations = tuple(float(index) * 0.125 for index in range(cells * cells))
    road_finish._init_worker(elevations, cells, cell_size)
    points = ((0.0, 0.0), (10.0, 30.0), (62.0, 87.0), (124.0, 124.0))
    values = dict(road_finish._sample_batch(points))
    for point in points:
        expected = road_finish._BASE_SAMPLE(
            elevations, cells, cell_size, point[0], point[1]
        )
        assert np.isclose(values[point], expected, rtol=0.0, atol=1.0e-12)
