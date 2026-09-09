# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the complete stock-bridge policy chain used by real builds.

Bridge length is water-authoritative.  Tide safety may raise the deck and final
alignment may fill a small ordinary-road phase gap, but neither is allowed to
turn hundreds of metres of dry approach road into stock bridge modules.
"""
from __future__ import annotations

_INSTALLED = False


def install_bridge_runtime_policy() -> None:
    """Install bridge policies and finish with the wet-only stock-span planner."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Rendering must capture the unwrapped OSM/generator functions first.
    from .bridge_render_policy import install_bridge_render_policy

    install_bridge_render_policy()

    # Keep the tide-safe vertical approach sampler.  This policy also has a
    # historical horizontal span wrapper; the final normalization below removes
    # that horizontal extension while retaining its deck-height protection.
    from .bridge_water_deck_clamp_policy import (
        install_bridge_water_deck_clamp_policy,
    )

    install_bridge_water_deck_clamp_policy()

    # Underlay cleanup must precede final alignment, which wraps its road fitter
    # and footprint test. Source-water then makes cleanup source-aware as well.
    from .bridge_underlay_cleanup_policy import install_bridge_underlay_cleanup_policy

    install_bridge_underlay_cleanup_policy()

    from .bridge_source_water_policy import install_bridge_source_water_policy

    install_bridge_source_water_policy()

    # This installer adds connected-road endpoint planning, final alignment and
    # the post-solver mapped-water reopening pass. Connected-road planning is
    # useful for diagnostics/alignment, but its 220 m dry-bank search must not be
    # the final authority on bridge length.
    from .bridge_or_causeway_terrain_policy import (
        install_bridge_or_causeway_terrain_policy,
    )

    install_bridge_or_causeway_terrain_policy()

    # Final planner normalization. The render policy's original stock planner
    # clips to first/last actual in-game water. Source-water fallback is retained
    # for coarse grids that miss a mapped crossing entirely, but both the tide
    # wrapper and connected-road wrapper are deliberately bypassed horizontally.
    from . import bridge_render_policy as _bridge
    from . import bridge_source_water_policy as _source
    from . import bridge_water_deck_clamp_policy as _clamp

    base_wet_plan = _clamp._ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    if base_wet_plan is None:
        raise RuntimeError("bridge tide policy did not capture the wet stock planner")
    _source._ORIGINAL_STOCK_PLAN = base_wet_plan
    _bridge.stock_bridge_span_plan = _source._mapped_water_stock_plan

    # Placement/terrain cache keys predate these runtime wrappers. Use a fresh
    # namespace so an existing 25-module dry-land bridge cannot be replayed from
    # the previous build cache after this policy change.
    from . import build_cache_policy as _build_cache

    _build_cache.BUILD_CACHE_REVISION = "v3-water-authoritative-bridge-length"
    _INSTALLED = True
