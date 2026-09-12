from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cwr_worldgen import bridge_abutment_terrain_policy as policy
from cwr_worldgen import bridge_render_policy as bridge


@dataclass(frozen=True)
class _Report:
    elevations: tuple[float, ...]
    changed_cells: int


def _spec():
    return SimpleNamespace(
        cells=8,
        cell_size=50.0,
        sea_level=0.0,
        water_depth=5.0,
        world_size=400.0,
    )


def _plan():
    return bridge.StockBridgeSpanPlan(
        points=((75.0, 75.0), (275.0, 75.0)),
        module_count=4,
        wet_start=(90.0, 75.0),
        wet_end=(260.0, 75.0),
        wet_length=170.0,
    )


def test_low_nominally_dry_abutments_grade_only_endpoint_cells() -> None:
    spec = _spec()
    plan = _plan()
    points = plan.points
    report = _Report((2.0,) * (spec.cells * spec.cells), changed_cells=7)

    policy._PLAN_CACHE.clear()
    with patch.object(
        policy,
        "_explicit_bridge_plans",
        return_value=((points, plan),),
    ):
        raised = policy._raise_bridge_abutments(
            report,
            None,
            None,
            spec,
        )

    expected = (
        set(policy._endpoint_cell_vertices(plan.points[0], spec))
        | set(policy._endpoint_cell_vertices(plan.points[1], spec))
    )
    target = policy._abutment_ground_target(spec)
    assert policy._ROAD_APPROACH_RAISE_METRES == pytest.approx(0.85)
    assert target == pytest.approx(6.315)
    assert len(expected) == 8
    for index, value in enumerate(raised.elevations):
        assert value == pytest.approx(target if index in expected else 2.0)

    assert raised.changed_cells == report.changed_cells + len(expected)
    assert policy._cached_bridge_plan(points, spec) == plan
    assert policy._cached_bridge_plan(tuple(reversed(points)), spec) == plan


def test_test22_height_is_raised_by_point_eight_five_metres() -> None:
    spec = _spec()
    plan = _plan()
    points = plan.points
    values = [9.0] * (spec.cells * spec.cells)
    support = policy._endpoint_cell_vertices(plan.points[0], spec)

    # atinybridgetest22 stores the endpoint terrain as 5.45 m after WRP
    # quantization. Keep the bridge untouched and lift only its road support cell.
    for index in support:
        values[index] = 5.45
    report = _Report(tuple(values), changed_cells=0)

    policy._PLAN_CACHE.clear()
    with patch.object(
        policy,
        "_explicit_bridge_plans",
        return_value=((points, plan),),
    ):
        graded = policy._raise_bridge_abutments(
            report,
            None,
            None,
            spec,
        )

    target = policy._abutment_ground_target(spec)
    assert target - 5.465 == pytest.approx(0.85)
    for index in support:
        assert graded.elevations[index] == pytest.approx(target)


def test_mixed_high_corner_is_flattened_instead_of_cutting_through_bridge() -> None:
    spec = _spec()
    plan = _plan()
    points = plan.points
    values = [8.0] * (spec.cells * spec.cells)
    support = policy._endpoint_cell_vertices(plan.points[0], spec)

    # Model the test21 failure: one old bank corner remains high while the other
    # support corners are low enough to require an abutment fill. A single plane
    # prevents the coarse terrain cell from forming a ramp through the bridge.
    for index in support:
        values[index] = 2.0
    values[support[0]] = 7.15
    report = _Report(tuple(values), changed_cells=0)

    policy._PLAN_CACHE.clear()
    with patch.object(
        policy,
        "_explicit_bridge_plans",
        return_value=((points, plan),),
    ):
        graded = policy._raise_bridge_abutments(
            report,
            None,
            None,
            spec,
        )

    target = policy._abutment_ground_target(spec)
    for index in support:
        assert graded.elevations[index] == pytest.approx(target)

    sampled = policy._osm._sample_elevation(
        graded.elevations,
        spec.cells,
        spec.cell_size,
        *plan.points[0],
    )
    assert sampled == pytest.approx(target)
    assert graded.elevations[0] == 8.0
    assert graded.changed_cells == len(support)


def test_underwater_bridge_endpoint_is_filled_to_expose_terminal_road_piece() -> None:
    spec = _spec()
    plan = _plan()
    points = plan.points
    values = [8.0] * (spec.cells * spec.cells)
    left_support = set(policy._endpoint_cell_vertices(plan.points[0], spec))
    right_support = set(policy._endpoint_cell_vertices(plan.points[1], spec))
    support = left_support | right_support

    # The fixed stock bridge ends just offshore. Its ordinary terminal road piece
    # is already present, but this support cell is underwater and therefore hides
    # the road. Raise only those endpoint cells into a short embankment.
    for index in support:
        values[index] = -1.0
    report = _Report(tuple(values), changed_cells=0)

    policy._PLAN_CACHE.clear()
    with patch.object(
        policy,
        "_explicit_bridge_plans",
        return_value=((points, plan),),
    ):
        raised = policy._raise_bridge_abutments(
            report,
            None,
            None,
            spec,
        )

    target = policy._abutment_ground_target(spec)
    for index, value in enumerate(raised.elevations):
        assert value == pytest.approx(target if index in support else 8.0)
    assert raised.changed_cells == len(support)
    assert policy._cached_bridge_plan(points, spec) == plan


def test_high_bridge_bank_is_not_lowered() -> None:
    spec = _spec()
    plan = _plan()
    points = plan.points
    report = _Report((9.0,) * (spec.cells * spec.cells), changed_cells=0)

    policy._PLAN_CACHE.clear()
    with patch.object(
        policy,
        "_explicit_bridge_plans",
        return_value=((points, plan),),
    ):
        raised = policy._raise_bridge_abutments(
            report,
            None,
            None,
            spec,
        )

    assert raised == report
    assert policy._cached_bridge_plan(points, spec) == plan


def test_endpoint_cell_is_bounded_to_four_coarse_vertices() -> None:
    spec = _spec()
    indices = policy._endpoint_cell_vertices((123.0, 176.0), spec)
    assert len(indices) == 4
    assert set(indices) == {
        3 * spec.cells + 2,
        3 * spec.cells + 3,
        4 * spec.cells + 2,
        4 * spec.cells + 3,
    }
