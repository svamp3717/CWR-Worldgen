# SPDX-License-Identifier: GPL-3.0-or-later
"""Make purpose-built stock junctions own the visible intersection centre.

A native Resistance T/X already contains the asphalt/cobble shape for the whole
intersection and exposes measured connectors about 6.25 metres from its logical
centre. Older local-fit fallbacks deliberately continued ordinary approach roads
all the way underneath a generic six-metre cap. That is correct for a plain
straight fallback, but wrong once the cap is replaced by a purpose-built native
junction: the approach borders remain visible through the native surface and the
intersection looks like several rectangular roads stacked together.

Keep the legacy under-cap behaviour only for generic straight caps. At measured
native T/X nodes, restore the quality fitter's original connector trim so each
stock approach terminates at the native footprint. Also restore connector-target
planning for all-paved T junctions. The transactional relaxation layer already
bounds and obstacle-checks those edits, then accepts them only when the resulting
node satisfies the ordinary strict native matcher.

Generated gravel keeps its existing mixed-junction rules. No new P3D is created.
"""
from __future__ import annotations

import math

from . import gravel_junction_policy as _gravel_junction
from . import road_quality_policy as _quality
from . import stock_road_connector_policy as _connector
from . import stock_road_local_fit_policy as _local
from . import stock_road_model_geometry as _geometry
from . import stock_road_paved_junction_completion_policy as _paved
from . import stock_road_surface_overlap_policy as _surface


_NATIVE_EXTENT_TOLERANCE_METRES = 1.0e-6
_ORIGINAL_QUALITY_WINDOW = None
_ORIGINAL_SURFACE_QUALITY_WINDOW = None
_INSTALLED = False


def _is_measured_native_junction(junction) -> bool:
    """Return True when quality geometry reserves a native T/X connector box."""

    if junction is None or _gravel_junction._is_gravel_junction(junction):
        return False
    extent = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    return (
        math.isclose(
            float(junction.half_length),
            extent,
            rel_tol=0.0,
            abs_tol=_NATIVE_EXTENT_TOLERANCE_METRES,
        )
        and math.isclose(
            float(junction.half_width),
            extent,
            rel_tol=0.0,
            abs_tol=_NATIVE_EXTENT_TOLERANCE_METRES,
        )
    )


def _native_ownership_quality_window(
    measure,
    pieces,
    start_distance,
    preferred_end,
    minimum_end,
    maximum_end,
    context,
):
    """Undo only the local-fit node extension at measured native junctions."""

    if _ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("native junction ownership policy is not installed")

    current = tuple(
        _ORIGINAL_QUALITY_WINDOW(
            measure,
            pieces,
            start_distance,
            preferred_end,
            minimum_end,
            maximum_end,
            context,
        )
    )
    if not pieces or _local._ORIGINAL_QUALITY_WINDOW is None:
        return current

    start_junction = context.junctions.get(
        _local._p._road_node_key(measure.points[0])
    )
    end_junction = context.junctions.get(
        _local._p._road_node_key(measure.points[-1])
    )
    restore_start = _is_measured_native_junction(start_junction)
    restore_end = _is_measured_native_junction(end_junction)
    if not restore_start and not restore_end:
        return current

    # This is the exact quality-window function captured immediately before the
    # local-fit policy installed its "continue underneath the cap" override.
    # For stock paved/ces approaches it preserves the measured native connector
    # trim. Generated gravel's own mixed-junction wrapper still returns its
    # historical node-covering window, so this does not change gravel behaviour.
    trimmed = tuple(
        _local._ORIGINAL_QUALITY_WINDOW(
            measure,
            pieces,
            start_distance,
            preferred_end,
            minimum_end,
            maximum_end,
            context,
        )
    )

    result = list(current)
    if restore_start:
        result[0] = trimmed[0]
    if restore_end:
        result[1] = trimmed[1]
        result[2] = trimmed[2]
        result[3] = trimmed[3]
    return tuple(result)


def install_stock_road_native_junction_ownership_policy() -> None:
    """Install connector-trim ownership after the paved completion layer."""

    global _ORIGINAL_QUALITY_WINDOW, _ORIGINAL_SURFACE_QUALITY_WINDOW, _INSTALLED
    if _INSTALLED:
        return
    if not _paved._INSTALLED:
        raise RuntimeError("paved junction completion policy must install first")
    if _local._ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("stock road local fit policy must install first")

    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _ORIGINAL_SURFACE_QUALITY_WINDOW = _surface._quality_window
    _quality._quality_window = _native_ownership_quality_window
    _surface._quality_window = _native_ownership_quality_window

    # The paved-completion layer used to veto connector relaxation whenever the
    # unmodified source T was visibly skewed. That forced exactly the generic
    # straight-cap overlap this policy is meant to eliminate. Restore the
    # measured connector target planner. The transaction layer still decides
    # whether a proposed skew correction is safe and strictly valid.
    if _paved._ORIGINAL_NATIVE_T_TARGETS is None:
        raise RuntimeError("paved connector target planner was not captured")
    _connector._native_t_targets = _paved._ORIGINAL_NATIVE_T_TARGETS

    _INSTALLED = True
