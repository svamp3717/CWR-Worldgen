from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import bridge_or_causeway_terrain_policy as policy
from cwr_worldgen import bridge_render_policy as bridge
from cwr_worldgen import bridge_water_deck_clamp_policy as clamp


def _spec():
    return SimpleNamespace(sea_level=0.0)


def test_high_bridge_restores_point_seven_metre_visual_lowering() -> None:
    spec = _spec()
    offset = bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    raw_approach_surface = 7.035

    # Final alignment currently adds the render offset back before the renderer
    # subtracts it. The empirical policy removes that cancellation, so the final
    # visible deck once again sits exactly 0.7 m below the high approach target.
    current_pre_tuning = raw_approach_surface + offset
    lowered_pre_tuning = policy._empirically_lowered_pre_tuning_height(
        current_pre_tuning,
        spec,
    )
    final_deck = lowered_pre_tuning - offset

    assert abs(final_deck - (raw_approach_surface - offset)) < 1e-12


def test_low_bridge_still_respects_tide_safe_floor() -> None:
    spec = _spec()
    offset = bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    minimum_final = clamp._minimum_final_deck(spec)
    current_pre_tuning = minimum_final + offset

    lowered_pre_tuning = policy._empirically_lowered_pre_tuning_height(
        current_pre_tuning,
        spec,
    )
    final_deck = lowered_pre_tuning - offset

    assert abs(final_deck - minimum_final) < 1e-12
