# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep tuned stock bridge decks above the final in-game water surface.

The stock bridge render policy deliberately lowers bridge decks by a small
world-space tuning offset so they meet ordinary road meshes cleanly on normal
banks.  At a shoreline, however, the untuned approach height may already be at
the minimum water-deck clearance.  Subtracting the tuning offset afterwards can
therefore push the rendered deck below sea level.

Wrap the approach-height sampler so its *pre-tuning* result includes enough
headroom for the later downward offset.  The final deck then remains at least
``sea_level + NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES`` while higher banks keep
the established 0.7 m tuning unchanged.
"""
from __future__ import annotations

from . import bridge_render_policy as _bridge
from . import osm as _osm

_ORIGINAL_DRY_APPROACH_HEIGHT = None
_INSTALLED = False


def _safe_dry_approach_height(
    endpoint,
    outward,
    raster,
    elevations,
    spec,
):
    height = _ORIGINAL_DRY_APPROACH_HEIGHT(
        endpoint,
        outward,
        raster,
        elevations,
        spec,
    )
    if height is None:
        return None

    minimum_final_deck = (
        float(spec.sea_level)
        + float(_osm.NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES)
    )
    minimum_pre_tuning_height = (
        minimum_final_deck
        + float(_bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES)
    )
    return max(float(height), minimum_pre_tuning_height)


def install_bridge_water_deck_clamp_policy() -> None:
    """Install after ``bridge_render_policy`` so its final tuning cannot sink decks."""

    global _ORIGINAL_DRY_APPROACH_HEIGHT, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_DRY_APPROACH_HEIGHT = _bridge._dry_approach_height
    _bridge._dry_approach_height = _safe_dry_approach_height
    _INSTALLED = True
