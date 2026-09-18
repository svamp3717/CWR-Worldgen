import math

import numpy as np

from cwr_worldgen import terrain_grid_performance_policy as perf
from cwr_worldgen import terrain_solver as terrain


def _normalise_components(components):
    return sorted(tuple(sorted(int(index) for index in component)) for component in components)


def test_milestone9_installs_grid_performance_policy() -> None:
    assert terrain._components is perf._fast_components
    assert terrain._distance_from_mask is perf._fast_distance_from_mask
    assert terrain._euclidean_distance_from_mask is perf._fast_euclidean_distance_from_mask
    assert terrain._component_touches_world_edge is perf._fast_component_touches_world_edge
    assert terrain._mask_from_components is perf._fast_mask_from_components
    assert terrain.solve_terrain_constraints is perf._fast_solve_terrain_constraints
    assert terrain.zip is perf._diagnostic_zip


def test_run_label_components_match_scalar_four_neighbour_components() -> None:
    cells = 9
    mask = [False] * (cells * cells)
    for x, z in (
        (0, 0), (1, 0), (1, 1),
        (4, 1), (4, 2), (5, 2), (6, 2),
        (8, 4), (8, 5),
        (2, 6), (3, 6), (3, 7), (2, 8),
    ):
        mask[z * cells + x] = True

    expected = perf._ORIGINAL_COMPONENTS(mask, cells)
    actual = perf._fast_components(mask, cells)
    assert _normalise_components(actual) == _normalise_components(expected)


def test_bounded_manhattan_distance_matches_queue_implementation() -> None:
    cells = 11
    mask = [False] * (cells * cells)
    for x, z in ((2, 3), (8, 2), (6, 9)):
        mask[z * cells + x] = True

    for maximum in (0, 1, 3, 7):
        assert perf._fast_distance_from_mask(mask, cells, maximum) == perf._ORIGINAL_DISTANCE_FROM_MASK(
            mask, cells, maximum
        )


def test_bounded_eight_neighbour_distance_matches_heap_implementation() -> None:
    cells = 13
    mask = [False] * (cells * cells)
    for x, z in ((1, 1), (9, 4), (4, 11)):
        mask[z * cells + x] = True

    for maximum in (0, 1, 4, 8):
        expected = np.asarray(
            perf._ORIGINAL_EUCLIDEAN_DISTANCE_FROM_MASK(mask, cells, maximum),
            dtype=np.float64,
        )
        actual = np.asarray(
            perf._fast_euclidean_distance_from_mask(mask, cells, maximum),
            dtype=np.float64,
        )
        assert np.array_equal(np.isinf(actual), np.isinf(expected))
        finite = np.isfinite(expected)
        assert np.allclose(actual[finite], expected[finite], rtol=0.0, atol=1.0e-12)


def test_vectorized_component_helpers_keep_existing_semantics() -> None:
    cells = 6
    components = [[0, 1, 7], [14, 15, 20], [34, 35]]
    for component in components:
        assert perf._fast_component_touches_world_edge(
            component, cells
        ) == perf._ORIGINAL_COMPONENT_TOUCHES_EDGE(component, cells)
    assert perf._fast_mask_from_components(components, cells * cells) == perf._ORIGINAL_MASK_FROM_COMPONENTS(
        components, cells * cells
    )


def test_vectorized_cut_fill_diagnostics_match_scalar_formula() -> None:
    before = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    after = [8.0, 10.0, 13.0, 9.5, 10.0, 14.0]
    categories = ["water", "natural", "road", "road", "natural", "building"]
    cell_size = 25.0

    actual = perf._vectorized_diagnostics(before, after, categories, cell_size)

    cell_area = cell_size * cell_size
    changed = []
    scalar_categories = {}
    for index, (left, right) in enumerate(zip(before, after)):
        delta = right - left
        if abs(delta) <= 1.0e-7:
            continue
        changed.append(delta)
        category = categories[index]
        stats = scalar_categories.setdefault(category, {
            "changed_cells": 0,
            "cut_volume_m3": 0.0,
            "fill_volume_m3": 0.0,
            "maximum_cut_m": 0.0,
            "maximum_fill_m": 0.0,
        })
        stats["changed_cells"] += 1
        if delta < 0:
            stats["cut_volume_m3"] += -delta * cell_area
            stats["maximum_cut_m"] = max(stats["maximum_cut_m"], -delta)
        else:
            stats["fill_volume_m3"] += delta * cell_area
            stats["maximum_fill_m"] = max(stats["maximum_fill_m"], delta)

    assert actual["changed_cells"] == len(changed)
    assert math.isclose(actual["maximum_cut"], max([-value for value in changed if value < 0] or [0.0]))
    assert math.isclose(actual["maximum_fill"], max([value for value in changed if value > 0] or [0.0]))
    assert math.isclose(
        actual["total_cut_volume_m3"],
        sum(-value for value in changed if value < 0) * cell_area,
    )
    assert math.isclose(
        actual["total_fill_volume_m3"],
        sum(value for value in changed if value > 0) * cell_area,
    )
    assert actual["category_adjustments"] == {
        key: scalar_categories[key] for key in sorted(scalar_categories)
    }


def test_diagnostic_zip_only_short_circuits_the_captured_full_grid_pair() -> None:
    before = (1.0, 2.0, 3.0)
    after = [4.0, 5.0, 6.0]
    capture = perf._DiagnosticCapture(expected_size=3)
    token = perf._DIAGNOSTIC_CAPTURE.set(capture)
    try:
        assert list(perf._diagnostic_zip(before, after)) == [(4.0, 4.0), (5.0, 5.0), (6.0, 6.0)]
        assert capture.before is before
        assert capture.after is after
        assert capture.used
        # Once consumed, every subsequent zip is ordinary built-in zip again.
        assert list(perf._diagnostic_zip(before, after)) == [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]
    finally:
        perf._DIAGNOSTIC_CAPTURE.reset(token)
