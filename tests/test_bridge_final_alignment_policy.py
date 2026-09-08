from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_final_alignment_policy as policy
from cwr_worldgen import bridge_render_policy as bridge
from cwr_worldgen import bridge_underlay_cleanup_policy as cleanup
from cwr_worldgen import bridge_water_deck_clamp_policy as clamp
from cwr_worldgen.model import WorldObject


def _spec():
    return SimpleNamespace(
        sea_level=0.0,
        cells=64,
        cell_size=50.0,
        world_size=3200.0,
    )


def test_tinybridgetest10_parallel_sil_inside_stock_width_is_underlay() -> None:
    span = cleanup._BridgeSpan(
        points=((591.407, 666.322), (928.224, 884.896)),
        road_width=7.0,
    )
    # PBO10 has surviving sil pieces around 5.4-5.5 m from the bridge chord.
    duplicate = WorldObject(
        1072,
        r"o\road\sil25.p3d",
        881.958,
        7.105,
        861.477,
        236.793,
        -0.248,
    )
    separate_parallel = WorldObject(
        2000,
        r"o\road\sil25.p3d",
        881.958,
        7.105,
        869.0,
        236.793,
        0.0,
    )

    assert policy._inside_emitted_stock_footprint(duplicate, span)
    assert not policy._inside_emitted_stock_footprint(separate_parallel, span)


def test_crossing_road_inside_bridge_width_is_preserved_by_heading() -> None:
    span = cleanup._BridgeSpan(
        points=((0.0, 0.0), (100.0, 0.0)),
        road_width=7.0,
    )
    crossing = WorldObject(
        1,
        r"o\road\sil25.p3d",
        50.0,
        0.0,
        5.5,
        0.0,
        0.0,
    )
    assert not policy._inside_emitted_stock_footprint(crossing, span)


def test_final_bridge_deck_matches_high_road_surface_after_legacy_tuning() -> None:
    spec = _spec()
    raw_road_surface = 7.035

    with patch.object(
        clamp,
        "_ORIGINAL_DRY_APPROACH_HEIGHT",
        return_value=raw_road_surface,
    ):
        pre_tuning = policy._final_road_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    final_deck = pre_tuning - bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - raw_road_surface) < 1e-12


def test_low_bank_still_finishes_at_tide_safe_deck() -> None:
    spec = _spec()

    with patch.object(
        clamp,
        "_ORIGINAL_DRY_APPROACH_HEIGHT",
        return_value=0.05,
    ):
        pre_tuning = policy._final_road_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    final_deck = pre_tuning - bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - 5.5) < 1e-12
