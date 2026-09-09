# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the complete stock-bridge policy chain used by real builds.

The individual bridge policy modules are intentionally independently testable,
but merely importing those modules does not activate their runtime wrappers.
Keep one explicit installer here so package/GUI builds use the same policy chain
the bridge tests exercise.
"""
from __future__ import annotations

_INSTALLED = False


def install_bridge_runtime_policy() -> None:
    """Install bridge rendering, water, terrain and road policies in dependency order."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Rendering must capture the unwrapped OSM/generator functions first.
    from .bridge_render_policy import install_bridge_render_policy

    install_bridge_render_policy()

    # The tide clamp must capture the render policy's stock-span planner before
    # source-water fallback and final alignment wrap that same planning chain.
    from .bridge_water_deck_clamp_policy import install_bridge_water_deck_clamp_policy

    install_bridge_water_deck_clamp_policy()

    # Underlay cleanup must precede final alignment, which wraps its road fitter
    # and footprint test. Source-water then makes cleanup source-aware as well.
    from .bridge_underlay_cleanup_policy import install_bridge_underlay_cleanup_policy

    install_bridge_underlay_cleanup_policy()

    from .bridge_source_water_policy import install_bridge_source_water_policy

    install_bridge_source_water_policy()

    # This final installer adds connected-road endpoint planning, final alignment
    # and the post-solver mapped-water reopening pass.
    from .bridge_or_causeway_terrain_policy import (
        install_bridge_or_causeway_terrain_policy,
    )

    install_bridge_or_causeway_terrain_policy()
    _INSTALLED = True
