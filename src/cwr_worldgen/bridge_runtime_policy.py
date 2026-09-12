# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the complete stock-bridge policy chain used by real builds.

Bridge length is water-authoritative. A coarse CWA terrain grid can still leave
the ordinary road at a bridge abutment submerged or below the stock bridge deck.
That is an approach-terrain problem, not a reason to turn hundreds of metres of
dry road into bridge modules.

The terrain policy keeps the wet-only stock span and grades the single coarse
terrain cell supporting each bridge endpoint to the road-approach height, even
when that endpoint sits just offshore. This creates the smallest possible
embankment needed to expose the terminal ordinary-road piece while leaving all
stock bridge transforms untouched. The underlay policy explicitly synthesizes
ordinary road pieces beneath the first stock bridge module at each end. The
planner reuses the pre-grade wet span so this embankment cannot shorten or move
the bridge afterward.

Final reconciliation may ask the runtime planner with a synthetic corridor built
from already-emitted bridge objects rather than with the original OSM polyline.
That corridor is matched back to the same cached pre-fill plan before any
post-fill re-planning is attempted, keeping the raised terrain and final bridge
endpoints in the same place.
"""
from __future__ import annotations

_INSTALLED = False


def _runtime_stock_bridge_span_plan(points, elevations, spec):
    """Reuse the pre-fill wet plan before considering post-fill terrain."""
    from . import bridge_abutment_terrain_policy as _abutment
    from . import bridge_source_water_policy as _source

    cached = _abutment._cached_bridge_plan(points, spec)
    if cached is None:
        cached = _abutment._cached_bridge_plan_for_corridor(points, spec)
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

    # The final planner assignment above intentionally bypasses earlier mutable
    # wrapper chains. Install the component-count guard only now, so it sees the
    # exact wet-only runtime planner that real GUI/CLI builds use.
    from .bridge_final_count_policy import install_bridge_final_count_policy
    install_bridge_final_count_policy()

    # Final placement now reuses the exact pre-fill wet plan that the terrain
    # approach was graded against. Use a fresh namespace so old placement/cache
    # state cannot retain the previous post-fill endpoint shift.
    from . import build_cache_policy as _build_cache
    _build_cache.BUILD_CACHE_REVISION = "v11-authoritative-pre-fill-bridge-plan"
    _INSTALLED = True
