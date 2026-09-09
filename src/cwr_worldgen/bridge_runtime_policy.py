# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the complete stock-bridge policy chain used by real builds.

Bridge length is water-authoritative.  A coarse CWA terrain grid can still leave
the ordinary road at a bridge abutment only two or three metres above nominal sea
level, which is flooded by the game's roughly five-metre tide.  That is an
approach-terrain problem, not a reason to turn hundreds of metres of dry road into
bridge modules.

The terrain policy therefore keeps the wet-only stock span and raises only the
single coarse terrain cell supporting each low, nominally dry bridge abutment.
The planner reuses the pre-raise wet span so that this small embankment cannot
make the next bridge-planning pass shorten the bridge again.
"""
from __future__ import annotations

_INSTALLED = False


def _runtime_stock_bridge_span_plan(points, elevations, spec):
    """Use the pre-abutment-fill wet span when available, otherwise plan normally."""
    from . import bridge_abutment_terrain_policy as _abutment
    from . import bridge_source_water_policy as _source

    cached = _abutment._cached_bridge_plan(points, spec)
    if cached is not None:
        return cached
    return _source._mapped_water_stock_plan(points, elevations, spec)


def install_bridge_runtime_policy() -> None:
    """Install bridge policies and finish with wet-only, raised-road approaches."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Rendering must capture the unwrapped OSM/generator functions first.
    from .bridge_render_policy import install_bridge_render_policy

    install_bridge_render_policy()

    # Retain the tide-safe vertical deck sampler. Its historical horizontal
    # extension is bypassed by the final planner below.
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

    # Install final alignment and the terrain pass that reopens mapped water.
    from .bridge_or_causeway_terrain_policy import (
        install_bridge_or_causeway_terrain_policy,
    )

    install_bridge_or_causeway_terrain_policy()

    # This outer terrain wrapper raises only the immediate low road abutment cell
    # after mapped water has been restored, and remembers the pre-fill wet span.
    from .bridge_abutment_terrain_policy import (
        install_bridge_abutment_terrain_policy,
    )

    install_bridge_abutment_terrain_policy()

    from . import bridge_render_policy as _bridge
    from . import bridge_source_water_policy as _source
    from . import bridge_water_deck_clamp_policy as _clamp

    # Start source-water fallback directly from the original wet-only renderer,
    # bypassing both historical dry-bank bridge extenders.
    base_wet_plan = _clamp._ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    if base_wet_plan is None:
        raise RuntimeError("bridge tide policy did not capture the wet stock planner")
    _source._ORIGINAL_STOCK_PLAN = base_wet_plan
    _bridge.stock_bridge_span_plan = _runtime_stock_bridge_span_plan

    # Test20 used the v4 maximum-tide span extender and can otherwise replay its
    # 23-module bridge from the persistent build cache.
    from . import build_cache_policy as _build_cache

    _build_cache.BUILD_CACHE_REVISION = "v5-wet-bridge-raised-road-abutments"
    _INSTALLED = True
