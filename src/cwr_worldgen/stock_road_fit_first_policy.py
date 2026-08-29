# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep paved-road repair focused on fitting real stock pieces.

The stock-road pipeline already tries measured straights, native ten-degree
curves, exact curve chains, and measured/native junction models before the late
visual passes run.  Lundby27 showed that the later seam and intersection
fallbacks could still append co-centred or heavily overlapping ``sil6`` helper
pieces after that fitting work had finished.  Those pieces can hide a visual
hole, but they also turn one geometry problem into a stack of road surfaces and
make the generated WRP harder to reason about.

Do not manufacture an intermediate paved-road fix by adding another overlapping
road object.  The existing model-selection and connector-locked fitting passes
get the first chance to solve the geometry, and legacy intersection tongues stay
disabled.

The final emitted-seam pass is different: it measures the pitch-projected
``WorldObject`` connectors that will actually be serialized.  The Lundby34 Road
Inspector report proved that disabling this last bounded, paved-only fallback
left 80 visible grass wedges.  Keep that final physical audit active while the
older intermediate visual and junction repair hooks remain disabled.  Its
same-family six-metre pieces sit below the visible carriageway and are emitted
only for unambiguous seams not already covered by paved surface.

The older planners remain installed and testable because they are useful
regression evidence.  This final production guard disables only their
intermediate object-appending application hooks.
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
    """Disable intermediate overlap repairs but retain the final seam audit."""

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

    # These intermediate calls happen after the native straight/curve/junction
    # fitting stack has run.  Keep their broad overlap repairs disabled.
    _finish._apply_curve_seam_covers = _preserve_fitted_visual_seam
    _intersection_edge._seal_legacy_paved_intersections = (
        _preserve_fitted_intersection
    )

    # Do not replace the final emitted hook.  It is the only pass that measures
    # pitch-projected physical connectors, and its refined planner is bounded to
    # unambiguous same-family paved seams.  Disabling it was the direct cause of
    # the surviving Lundby34 grass wedges.
    _INSTALLED = True
