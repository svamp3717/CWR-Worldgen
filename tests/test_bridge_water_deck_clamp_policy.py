from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_render_policy as bridge_render
from cwr_worldgen import bridge_water_deck_clamp_policy as clamp
from cwr_worldgen import osm


def _spec(*, sea_level: float = 0.0):
    return SimpleNamespace(sea_level=sea_level)


def test_low_shoreline_reserves_tuning_headroom_before_final_water_clamp() -> None:
    spec = _spec(sea_level=0.0)
    minimum_final = osm.NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES

    with patch.object(clamp, "_ORIGINAL_DRY_APPROACH_HEIGHT", return_value=minimum_final):
        pre_tuning = clamp._safe_dry_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    final_deck = pre_tuning - bridge_render._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - minimum_final) < 1e-12


def test_high_bank_keeps_existing_point_seven_metre_tuning() -> None:
    spec = _spec(sea_level=0.0)
    untuned_road_surface = 7.035

    with patch.object(
        clamp,
        "_ORIGINAL_DRY_APPROACH_HEIGHT",
        return_value=untuned_road_surface,
    ):
        pre_tuning = clamp._safe_dry_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    assert pre_tuning == untuned_road_surface
    final_deck = pre_tuning - bridge_render._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - 6.335) < 1e-12


def test_none_from_underlying_sampler_is_preserved() -> None:
    with patch.object(clamp, "_ORIGINAL_DRY_APPROACH_HEIGHT", return_value=None):
        assert clamp._safe_dry_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            _spec(),
        ) is None
