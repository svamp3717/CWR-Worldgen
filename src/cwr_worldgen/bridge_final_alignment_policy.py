# SPDX-License-Identifier: GPL-3.0-or-later
"""Finish stock-bridge alignment against real roads and the bridge footprint.

Road-connected bridge planning now places abutments on actual ordinary-road
approaches. Two legacy assumptions can still spoil that result:

* the historical 0.7 m bridge tuning lowers the visible deck below the fitted
  road surface even after horizontal endpoints are correctly connected; and
* underlay cleanup uses a corridor narrower than the stock bridge itself, so an
  aligned ordinary road can survive a few metres beside the bridge centreline.

Keep the historical tuning constant intact for compatibility, but compensate it
when sampling final bridge approach heights. Also treat the measured outer
Roadway width of the stock bridge as occupied bridge footprint for aligned-road
cleanup.
"""
from __future__ import annotations

from . import bridge_render_policy as _bridge
from . import bridge_underlay_cleanup_policy as _cleanup
from . import bridge_water_deck_clamp_policy as _clamp

_INSTALLED = False
_ORIGINAL_DRY_APPROACH_HEIGHT = None
_ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE = None

# Measured from O\Hous\most_stred30.p3d Roadway LOD. The outer roadway strips
# reach x = +/-6.642317 m. A small tolerance covers WRP float/heading rounding.
_STOCK_BRIDGE_OUTER_HALF_WIDTH_METRES = 6.642317295074463
_STOCK_BRIDGE_FOOTPRINT_MARGIN_METRES = 0.15


def _final_road_approach_height(
    endpoint,
    outward,
    raster,
    elevations,
    spec,
):
    """Return the pre-tuning height that makes the final deck match the road."""
    raw_sampler = getattr(_clamp, "_ORIGINAL_DRY_APPROACH_HEIGHT", None)
    if raw_sampler is None:
        raw_sampler = _ORIGINAL_DRY_APPROACH_HEIGHT
    height = raw_sampler(endpoint, outward, raster, elevations, spec)
    if height is None:
        return None

    # _anchor_stock_bridge_chains subtracts the historical tuning offset after
    # this function returns. Add it back here so a road-connected endpoint lands
    # at the actual road surface. Low banks still clamp to the 5.5 m tide-safe
    # final deck used by bridge_water_deck_clamp_policy.
    final_target = max(
        float(height),
        float(_clamp._minimum_final_deck(spec)),
    )
    return final_target + float(_bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES)


def _inside_emitted_stock_footprint(obj, span) -> bool:
    """Return whether an aligned road centre lies inside the stock bridge body."""
    if _cleanup._paved._family(obj.model_path) is None:
        return False
    if len(span.points) < 2:
        return False

    distance, heading, along, total = _cleanup._nearest_polyline_measure(
        span.points,
        (float(obj.x), float(obj.z)),
    )
    if total <= 1.0e-9:
        return False

    endpoint_keep = min(
        float(_cleanup._ENDPOINT_KEEP_METRES),
        total * 0.02,
    )
    if along <= endpoint_keep or along >= total - endpoint_keep:
        return False

    corridor = (
        _STOCK_BRIDGE_OUTER_HALF_WIDTH_METRES
        + _STOCK_BRIDGE_FOOTPRINT_MARGIN_METRES
    )
    if distance > corridor:
        return False
    if (
        _cleanup._undirected_heading_difference(
            obj.heading_degrees,
            heading,
        )
        > _cleanup._ALIGNMENT_TOLERANCE_DEGREES
    ):
        return False
    return True


def _road_object_under_bridge(obj, spans) -> bool:
    for span in spans:
        if _inside_emitted_stock_footprint(obj, span):
            return True
    return _ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE(obj, spans)


def install_bridge_final_alignment_policy() -> None:
    """Align stock decks to road height and clean the full bridge footprint."""
    global _INSTALLED
    global _ORIGINAL_DRY_APPROACH_HEIGHT
    global _ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE

    if _INSTALLED:
        return

    _ORIGINAL_DRY_APPROACH_HEIGHT = _bridge._dry_approach_height
    _ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE = _cleanup._road_object_under_bridge

    _bridge._dry_approach_height = _final_road_approach_height
    _cleanup._road_object_under_bridge = _road_object_under_bridge
    _INSTALLED = True
