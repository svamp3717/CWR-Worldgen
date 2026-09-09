# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the complete stock-bridge policy chain used by real builds.

Bridge length is water-authoritative. A coarse CWA terrain grid can still leave
the ordinary road at a bridge abutment only two or three metres above nominal sea
level, which is flooded by the game's roughly five-metre tide. That is an
approach-terrain problem, not a reason to turn hundreds of metres of dry road into
bridge modules.

The terrain policy keeps the wet-only stock span and grades only the single coarse
terrain cell supporting each low, nominally dry bridge abutment. The road-side
correction raises that approach terrain by 0.85 m while leaving all stock bridge
transforms untouched. The underlay policy explicitly synthesizes ordinary road
pieces on the ground beneath the first stock bridge module at each end; it does
not merely preserve road pieces that may not have been fitted there. The planner
reuses the pre-grade wet span so this small embankment cannot shorten the bridge.
"""
from __future__ import annotations

_INSTALLED = False


def _runtime_stock_bridge_span_plan(points, elevations, spec):
    """Use the pre-abutment-grade wet span when available, otherwise plan normally."""
    from . import bridge_abutment_terrain_policy as _abutment
    from . import bridge_source_water_policy as _source

    cached = _abutment._cached_bridge_plan(points, spec)
    if cached is not None:
        return cached
    return _source._mapped_water_stock_plan(points, elevations, spec)


def install_bridge_runtime_policy() -> None:
    """Install bridge policies and finish with wet-only, graded-road approaches."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .bridge_render_policy import install_bridge_render_policy
    install_bridge_render_policy()

    from .bridge_water_deck_clamp_policy import install_bridge_water_deck_clamp_policy
    install_bridge_water_deck_clamp_policy()

    from .bridge_underlay_cleanup_policy import install_bridge_underlay_cleanup_policy
    install_bridge_underlay_cleanup_policy()

    from .bridge_source_water_policy import install_bridge_source_water_policy
    install_bridge_source_water_policy()

    from .bridge_or_causeway_terrain_policy import install_bridge_or_causeway_terrain_policy
    install_bridge_or_causeway_terrain_policy()

    from .bridge_abutment_terrain_policy import install_bridge_abutment_terrain_policy
    install_bridge_abutment_terrain_policy()

    from . import bridge_render_policy as _bridge
    from . import bridge_source_water_policy as _source
    from . import bridge_water_deck_clamp_policy as _clamp

    base_wet_plan = _clamp._ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    if base_wet_plan is None:
        raise RuntimeError("bridge tide policy did not capture the wet stock planner")
    _source._ORIGINAL_STOCK_PLAN = base_wet_plan
    _bridge.stock_bridge_span_plan = _runtime_stock_bridge_span_plan

    # Test26 proved that retaining terminal underlays was insufficient: the road
    # fitter emitted no road objects inside either terminal bridge module. Force
    # a fresh road fit with explicit terminal-mask synthesis.
    from . import build_cache_policy as _build_cache
    _build_cache.BUILD_CACHE_REVISION = "v9-synthesized-terminal-bridge-road-underlays"
    _INSTALLED = True
