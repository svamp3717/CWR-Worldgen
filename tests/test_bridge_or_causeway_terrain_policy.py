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


def test_bridge_water_reopen_overrides_causeway_fill_only_under_bridge() -> None:
    spec = _spec()
    centre = (1575.336, 1456.016)
    axis = (-0.038418, 0.999262)
    original = (2.0,) * (spec.cells * spec.cells)
    filled = (5.5,) * (spec.cells * spec.cells)
    report = _Report(filled, changed_cells=0)

    with patch.object(
        policy,
        "_coarse_source_bridge_channels",
        return_value=((centre, axis),),
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

    # A terrain row beyond the bridge stays at the tide-safe road/embankment
    # level. Bridge and causeway are therefore selected by location, not stacked.
    assert corrected.elevations[30 * 64 + 31] == 5.5
    assert corrected.elevations[30 * 64 + 32] == 5.5


def test_bridge_channel_depth_scales_down_for_shallow_configured_water() -> None:
    spec = SimpleNamespace(sea_level=1.0, water_depth=1.25)
    assert policy._water_target(spec) == -0.25
