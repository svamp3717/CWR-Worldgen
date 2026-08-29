# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep paved-road repair focused on fitting real stock pieces.

The stock-road pipeline already tries measured straights, native ten-degree
curves, exact curve chains, and measured/native junction models before the late
visual passes run.  Lundby27 showed that the later seam and intersection
fallbacks could still append co-centred or heavily overlapping ``sil6`` helper
pieces after that fitting work had finished.  Those pieces can hide a visual
hole, but they also turn one geometry problem into a stack of road surfaces and
make the generated WRP harder to reason about.

Do not manufacture a paved-road fix by adding another overlapping road object.
The existing model-selection and connector-locked fitting passes get the first
and only chance to solve the geometry.  If they cannot represent a seam or
junction cleanly, preserve the fitted objects unchanged and let Road Inspector
report the unresolved case so the fitting policy can be improved instead.

The older planners remain installed and testable because they are useful
regression evidence.  This final production guard only disables their object-
appending application hooks.
"""
from __future__ import annotations

from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_intersection_edge_policy as _intersection_edge
from . import stock_road_visual_finish_policy as _finish


_ORIGINAL_VISUAL_SEAM_APPLY = None
_ORIGINAL_INTERSECTION_EDGE_APPLY = None
_ORIGINAL_EMITTED_SEAM_APPLY = None
_INSTALLED = False


def _preserve_fitted_visual_seam(report, elevations, spec):
    """Leave a residual seam visible instead of appending a cover road."""

    return report


def _preserve_fitted_intersection(
    report,
    dataset,
    projection,
    elevations,
    spec,
):
    """Keep the fitted junction result instead of adding overlapping tongues."""

    return report


def _preserve_fitted_emitted_seam(report, elevations, spec):
    """Do not add final WRP-space seam underlays after fitting has completed."""

    return report


def install_stock_road_fit_first_policy() -> None:
    """Install the final no-overlap guard after every stock fitting policy."""

    global _ORIGINAL_VISUAL_SEAM_APPLY
    global _ORIGINAL_INTERSECTION_EDGE_APPLY
    global _ORIGINAL_EMITTED_SEAM_APPLY
    global _INSTALLED

    if _INSTALLED:
        return
    if not _finish._INSTALLED:
        raise RuntimeError("stock road visual finish policy must install first")
    if not _intersection_edge._INSTALLED:
        raise RuntimeError("stock road intersection edge policy must install first")
    if not _emitted._INSTALLED:
        raise RuntimeError("stock road emitted seam policy must install first")

    _ORIGINAL_VISUAL_SEAM_APPLY = _finish._apply_curve_seam_covers
    _ORIGINAL_INTERSECTION_EDGE_APPLY = (
        _intersection_edge._seal_legacy_paved_intersections
    )
    _ORIGINAL_EMITTED_SEAM_APPLY = _emitted._apply_emitted_seam_covers

    # These calls happen only after the native straight/curve/junction fitting
    # stack has run.  Leaving a defect visible here is intentional: it forces the
    # next fix into model selection or connector fitting rather than another slab.
    _finish._apply_curve_seam_covers = _preserve_fitted_visual_seam
    _intersection_edge._seal_legacy_paved_intersections = (
        _preserve_fitted_intersection
    )
    _emitted._apply_emitted_seam_covers = _preserve_fitted_emitted_seam
    _INSTALLED = True
