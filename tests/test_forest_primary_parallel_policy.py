from types import SimpleNamespace

import numpy as np

from cwr_worldgen import forest_primary_parallel_policy as primary
from cwr_worldgen import object_stage_parallel_policy


def _spec():
    return SimpleNamespace(
        cells=6,
        cell_size=10.0,
        world_size=60.0,
        forest_tree_spacing=20.0,
        forest_low_anchor=True,
        forest_ground_clearance=0.1,
        forest_maximum_block_relief=100.0,
        forest_block_maximum_burial=100.0,
        forest_block_maximum_float=100.0,
    )


def test_primary_worker_compacts_regular_block_supports_without_changing_fit() -> None:
    spec = _spec()
    elevations = tuple(float(index) * 0.125 for index in range(spec.cells * spec.cells))
    flats = np.asarray([4], dtype=np.int64)  # centre of the 3x3 primary lattice
    primary._init_worker(elevations, spec, 3, flats)

    offset, results = primary._evaluate_batch((0, 1))
    centre, minimum, maximum, fit_row = results[0]
    x = z = 30.0
    supports = object_stage_parallel_policy._BASE_SQUARE_SAMPLES(
        elevations, spec.cells, spec.cell_size, x, z, spec.forest_tree_spacing
    )
    expected_fit = primary._BASE_TERRAIN_FIT(
        supports,
        clearance=spec.forest_ground_clearance,
        maximum_burial=spec.forest_block_maximum_burial,
        maximum_float=spec.forest_block_maximum_float,
    )

    assert offset == 0
    assert centre == object_stage_parallel_policy._BASE_OSM_SAMPLE(
        elevations, spec.cells, spec.cell_size, x, z
    )
    assert minimum == min(supports)
    assert maximum == max(supports)
    assert primary._fit_from_row(np.asarray(fit_row)) == expected_fit


def test_primary_support_proxy_reuses_precomputed_fit() -> None:
    expected = (12.5, 0.4, 0.2)
    supports = primary._PrimarySupports(11.0, 13.0, expected)

    assert min(supports) == 11.0
    assert max(supports) == 13.0
    assert primary._cached_fit(
        supports,
        clearance=999.0,
        maximum_burial=0.0,
        maximum_float=0.0,
    ) == expected
