from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_or_causeway_terrain_policy as policy


@dataclass(frozen=True)
class _Report:
    elevations: tuple[float, ...]
    changed_cells: int


def _spec():
    return SimpleNamespace(
        cells=64,
        cell_size=50.0,
        sea_level=0.0,
        water_depth=5.0,
    )


def test_tinybridgetest7_reopens_only_crossing_row_beneath_stock_bridge() -> None:
    spec = _spec()
    centre = (1575.336, 1456.016)
    axis = (-0.038418, 0.999262)

    indices = policy._nearest_crossing_vertices(centre, axis, spec)

    # These are x=1550/1600, z=1450. The next z=1500 terrain row must remain
    # untouched so ordinary road beyond the 50 m stock bridge does not inherit a
    # second water crossing.
    assert set(indices) == {29 * 64 + 31, 29 * 64 + 32}
    assert 30 * 64 + 31 not in indices
    assert 30 * 64 + 32 not in indices


def test_long_mapped_water_run_reopens_every_crossing_row() -> None:
    spec = _spec()
    wet_start = (1575.0, 1200.0)
    wet_end = (1575.0, 1700.0)
    axis = (0.0, 1.0)

    indices = set(
        policy._wet_interval_crossing_vertices(
            wet_start, wet_end, axis, spec
        )
    )

    expected = {
        row * spec.cells + column
        for row in range(24, 35)
        for column in (31, 32)
    }
    assert indices == expected
    assert 23 * spec.cells + 31 not in indices
    assert 35 * spec.cells + 31 not in indices


def test_mapped_water_runs_keep_real_dry_gap_separate() -> None:
    polygon_type = policy._source._ProjectedWaterPolygon
    context = SimpleNamespace(
        water=(
            polygon_type(
                outer=((0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)),
                holes=(),
                bounds=(0.0, 0.0, 100.0, 20.0),
            ),
            polygon_type(
                outer=((0.0, 40.0), (100.0, 40.0), (100.0, 60.0), (0.0, 60.0)),
                holes=(),
                bounds=(0.0, 40.0, 100.0, 60.0),
            ),
        )
    )

    runs = policy._mapped_water_runs(((50.0, -10.0), (50.0, 70.0)), context)

    assert len(runs) == 2
    assert abs(runs[0][0] - 10.0) < 0.05
    assert abs(runs[0][1] - 30.0) < 0.05
    assert abs(runs[1][0] - 50.0) < 0.05
    assert abs(runs[1][1] - 70.0) < 0.05
    assert runs[1][0] - runs[0][1] > 19.9


def test_bridge_water_reopen_overrides_causeway_fill_only_under_mapped_water() -> None:
    spec = _spec()
    wet_start = (1575.0, 1451.0)
    wet_end = (1575.0, 1461.0)
    axis = (0.0, 1.0)
    original = (2.0,) * (spec.cells * spec.cells)
    filled = (5.5,) * (spec.cells * spec.cells)
    report = _Report(filled, changed_cells=0)

    with patch.object(
        policy,
        "_coarse_source_bridge_channels",
        return_value=((wet_start, wet_end, axis),),
    ):
        corrected = policy._reopen_bridge_water(
            report,
            original,
            None,
            None,
            spec,
        )

    crossing = {29 * 64 + 31, 29 * 64 + 32}
    assert policy._water_target(spec) == -3.0
    for index in crossing:
        assert corrected.elevations[index] == -3.0

    # A terrain row beyond the mapped wet run stays at the tide-safe road level.
    assert corrected.elevations[30 * 64 + 31] == 5.5
    assert corrected.elevations[30 * 64 + 32] == 5.5


def test_long_bridge_reopens_full_mapped_water_even_when_some_cells_are_already_wet() -> None:
    spec = _spec()
    wet_start = (1575.0, 1200.0)
    wet_end = (1575.0, 1700.0)
    axis = (0.0, 1.0)
    original = (2.0,) * (spec.cells * spec.cells)
    solved = [5.5] * (spec.cells * spec.cells)
    # Simulate a long crossing where the coarse solver already retained one wet
    # row. The old policy treated this as proof the whole bridge was fine.
    solved[29 * spec.cells + 31] = -1.0
    solved[29 * spec.cells + 32] = -1.0
    report = _Report(tuple(solved), changed_cells=0)

    with patch.object(
        policy,
        "_coarse_source_bridge_channels",
        return_value=((wet_start, wet_end, axis),),
    ):
        corrected = policy._reopen_bridge_water(
            report,
            original,
            None,
            None,
            spec,
        )

    for row in range(24, 35):
        for column in (31, 32):
            index = row * spec.cells + column
            expected = -1.0 if row == 29 else -3.0
            assert corrected.elevations[index] == expected
    assert corrected.elevations[23 * spec.cells + 31] == 5.5
    assert corrected.elevations[35 * spec.cells + 31] == 5.5


def test_bridge_channel_depth_scales_down_for_shallow_configured_water() -> None:
    spec = SimpleNamespace(sea_level=1.0, water_depth=1.25)
    assert policy._water_target(spec) == -0.25
