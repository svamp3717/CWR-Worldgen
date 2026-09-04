# SPDX-License-Identifier: GPL-3.0-or-later
"""Bound expensive procedural building asset generation.

Polygon-native buildings are useful, but each unique footprint becomes a unique
P3D and can therefore bypass the ordinary reusable-variant budget.  The historic
2048 polygon cap is far too expensive once modeler-backed materials, openings,
interiors and secondary architecture are enabled.

This policy keeps polygon-native fidelity for a bounded subset, then falls back
to the existing dimension-aware reusable rectangular variants.  A power user can
restore a larger polygon budget with CWR_WORLDGEN_POLYGON_BUILDING_VARIANTS.
"""
from __future__ import annotations

import os

_DEFAULT_STANDARD_VARIANTS = 128
_DEFAULT_POLYGON_DIVISOR = 2
_DEFAULT_POLYGON_FLOOR = 16
_DEFAULT_POLYGON_CEILING = 96
_DEFAULT_BUILDING_PARALLEL_MINIMUM = 16
_ENV_POLYGON_VARIANTS = "CWR_WORLDGEN_POLYGON_BUILDING_VARIANTS"
_INSTALLED = False


def _polygon_variant_budget(maximum_variants: int) -> int:
    """Return the default exact-polygon budget for one building library."""

    override = os.environ.get(_ENV_POLYGON_VARIANTS, "").strip()
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = -1
        if requested >= 0:
            return requested

    standard = max(1, int(maximum_variants))
    return max(
        _DEFAULT_POLYGON_FLOOR,
        min(_DEFAULT_POLYGON_CEILING, standard // _DEFAULT_POLYGON_DIVISOR),
    )


def install_building_asset_budget_policy() -> None:
    """Install bounded polygon generation and earlier building parallelism."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import parallel_assets
    from . import procedural_buildings as buildings

    original_init = buildings.ProceduralBuildingLibrary.__init__
    original_write_assets = buildings.ProceduralBuildingLibrary.write_assets

    def budgeted_init(self, *args, **kwargs):
        # The constructor is keyword-only today, but accepting *args preserves
        # compatibility with any external caller that wrapped it before us.
        if "maximum_polygon_variants" not in kwargs:
            kwargs["maximum_polygon_variants"] = _polygon_variant_budget(
                int(kwargs.get("maximum_variants", _DEFAULT_STANDARD_VARIANTS))
            )
        return original_init(self, *args, **kwargs)

    def write_assets_with_budget_progress(self, source_dir, catalogue_path):
        # The generic 73% label can otherwise sit unchanged for a long time.
        # Report the actual bounded workset so a large cache miss is visible
        # rather than looking like the GUI has frozen.
        try:
            from .progress import report_progress

            total = len(getattr(self, "_usage", {}))
            polygons = sum(
                bool(getattr(key, "footprint_vertices", ()))
                for key in getattr(self, "_usage", {})
            )
            report_progress(
                73,
                f"Generating procedural building assets ({total} variants, {polygons} exact footprints)",
            )
        except Exception:
            # Progress text must never be allowed to break generation.
            pass
        return original_write_assets(self, source_dir, catalogue_path)

    buildings.ProceduralBuildingLibrary.__init__ = budgeted_init
    buildings.ProceduralBuildingLibrary.write_assets = write_assets_with_budget_progress

    # Detailed modeler-backed exterior P3Ds are now heavy enough that waiting for
    # 64 cache misses before using the process workers leaves medium maps serial.
    # Interior batches already use an even lower threshold via the interior
    # performance policy; this changes only the generic building cutoff.
    parallel_assets._BUILDING_PARALLEL_MINIMUM = min(
        int(getattr(parallel_assets, "_BUILDING_PARALLEL_MINIMUM", 64)),
        _DEFAULT_BUILDING_PARALLEL_MINIMUM,
    )

    _INSTALLED = True
