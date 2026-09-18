from types import SimpleNamespace

import numpy as np

from cwr_worldgen import forest_array_cache_policy as array_cache
from cwr_worldgen import forest_vector_performance_policy as perf
from cwr_worldgen import object_stage_parallel_policy as stage_parallel
from cwr_worldgen import osm
from cwr_worldgen.procedural_forests import FOREST_UNDERGROWTH_VARIANTS


def test_primary_everon_prefilter_rejects_empty_lattice_cells() -> None:
    cells = 8
    forest = [False] * (cells * cells)
    # One 50 m primary block centred at x/z=75 has multiple forest probes.
    for row in (2, 3, 4):
        for col in (2, 3, 4):
            forest[row * cells + col] = True
    raster = SimpleNamespace(forest=tuple(forest))
    spec = SimpleNamespace(
        forest_profile="everon",
        max_forest_objects=1000,
        forest_tree_spacing=50.0,
        world_size=200.0,
        cells=cells,
        cell_size=25.0,
    )
    context = perf._primary_forest_possible(raster, spec)
    assert context is not None
    assert context.possible_primary.shape == (4, 4)
    assert np.count_nonzero(context.possible_primary) < 16
    assert bool(context.possible_primary[1, 1])


def test_batched_primary_road_hits_match_scalar_corridor_queries() -> None:
    context = perf._ForestVectorContext(
        spacing=50.0,
        world_size=200.0,
        possible_primary=np.ones((4, 4), dtype=np.bool_),
    )
    corridors = (
        ((0.0, 60.0), (200.0, 60.0), 4.0),
        ((110.0, 0.0), (110.0, 200.0), 6.0),
        ((145.0, 145.0), (195.0, 195.0), 3.0),
    )
    hits = perf._batch_primary_road_hits(corridors, context)
    for row in range(4):
        for column in range(4):
            x = (column + 0.5) * context.spacing
            z = (row + 0.5) * context.spacing
            expected = perf._ORIGINAL_FOREST_ROAD_TEST(
                corridors, x, z, block_size=context.spacing
            )
            assert bool(hits[row, column]) == expected


def test_triangle_bounds_batch_matches_scalar_terrain_sampler() -> None:
    cells = 6
    cell_size = 25.0
    elevations = tuple(
        3.0 + (index // cells) * 0.7 + (index % cells) * 0.2
        for index in range(cells * cells)
    )
    xs = np.asarray((3.0, 24.0, 31.5, 72.0, 118.0), dtype=np.float64)
    zs = np.asarray((4.0, 19.0, 54.0, 80.5, 123.0), dtype=np.float64)
    lower, upper = perf._triangle_bounds_batch(
        elevations, cells, cell_size, xs, zs
    )
    for index, (x, z) in enumerate(zip(xs, zs)):
        expected = osm._triangle_elevation_bounds(
            elevations, cells, cell_size, float(x), float(z)
        )
        assert np.isclose(lower[index], expected[0], rtol=0.0, atol=1.0e-12)
        assert np.isclose(upper[index], expected[1], rtol=0.0, atol=1.0e-12)


def test_vector_cluster_grounding_matches_previous_optimized_result() -> None:
    cells = 10
    cell_size = 25.0
    size = cells * cells
    elevations = tuple(
        5.0 + (index // cells) * 0.02 + (index % cells) * 0.015
        for index in range(size)
    )
    raster = SimpleNamespace(
        forest=(True,) * size,
        water=(False,) * size,
        roads=(False,) * size,
        buildings=(False,) * size,
    )
    spec = SimpleNamespace(
        world_size=cells * cell_size,
        cells=cells,
        cell_size=cell_size,
        name="test_world",
        forest_cluster_footprint_margin=0.75,
        forest_low_anchor=False,
        forest_cluster_tree_maximum_float=10.0,
        forest_cluster_bush_maximum_float=10.0,
    )
    variant = FOREST_UNDERGROWTH_VARIANTS[0]
    kwargs = dict(
        variant=variant,
        elevations=elevations,
        raster=raster,
        road_corridors=(),
        spec=spec,
        x=125.0,
        z=125.0,
        heading=27.0,
        require_forest=True,
        minimum_forest_fraction=0.80,
        maximum_relief=20.0,
        maximum_burial=0.8,
        maximum_float=10.0,
        clearance=0.03,
    )
    expected = perf._ORIGINAL_PLACE_CLUSTER(**kwargs)
    actual = perf._vector_place_cluster_at(**kwargs)
    assert actual is not None and expected is not None
    assert actual[0] == expected[0]
    assert actual[5] == expected[5]
    for actual_value, expected_value in zip(actual[1:5] + actual[6:], expected[1:5] + expected[6:]):
        assert np.isclose(actual_value, expected_value, rtol=0.0, atol=1.0e-9)


def test_vector_forest_policy_is_live_under_multicore_cache() -> None:
    assert osm._place_cluster_at is stage_parallel._cached_place_cluster_at
    assert stage_parallel._BASE_PLACE_CLUSTER is array_cache._cached_vector_cluster
    assert perf._vector_place_cluster_at is array_cache._cached_vector_cluster
    assert perf._triangle_bounds_batch is array_cache._cached_triangle_bounds_batch
    assert (
        osm.forest_block_intersects_road_corridors
        is perf._fast_forest_block_intersects_road_corridors
    )
