# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply stock paved-road placement habits measured in ``kodiak2.wrp``.

Kodiak reinforces the earlier WrpTool reference but adds two useful production
rules.  Paved roads are built from long chains of native ten-degree curve P3Ds,
and approaches overlap purpose-built junction meshes by roughly half a metre
instead of stopping short or continuing to the logical node under the junction.

Keep this policy deliberately paved-only.  ``sil``, ``asf`` and ``kos`` receive
more eager exact curve promotion and a 0.55 m approach overlap into stock T/X
footprints.  ``ces`` and generated gravel retain their existing fitting rules.
The final cleanup also removes the characteristic stale node-to-connector short
straight if an older composed wrapper manages to reintroduce it after a native
junction has already been selected.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_model_geometry as _geometry
from . import stock_road_paved_junction_completion_policy as _paved
from . import stock_road_reference_wrp_policy as _reference
from . import stock_road_sharp_turn_policy as _sharp
from . import stock_road_visual_finish_policy as _finish


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})

# Measurements from kodiak2.wrp: ordinary sil joins have about 0.46 m median
# longitudinal overlap and native-junction approaches about 0.58 m.  Use a
# slightly conservative round value for generated junction approaches.
KODIAK_PAVED_JUNCTION_OVERLAP_METRES = 0.55

# Kodiak routinely chains native curves for long bends, including runs far past
# the old 180 m emergency-fit limit.  The exact beam/corridor/tangent gates stay
# authoritative, so broadening the candidate window does not loosen geometry.
KODIAK_MINIMUM_CURVE_PROMOTION_TURN_DEGREES = 5.0
KODIAK_MAXIMUM_CURVE_PROMOTION_TURN_DEGREES = 210.0
KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES = 420.0
KODIAK_MAXIMUM_EXTRA_CURVE_PIECES = 5

# Native T/X ownership cleanup.  The logical point recovered from an asymmetric
# P3D can be displaced slightly from the normalized source node, so use the
# nearest actual source junction within a bounded radius before testing a stale
# short road whose endpoints are both inside the native connector footprint.
KODIAK_SOURCE_NODE_RECOVERY_METRES = 2.0
KODIAK_NATIVE_NODE_ENDPOINT_TOLERANCE_METRES = 0.30
KODIAK_NATIVE_CONNECTOR_MARGIN_METRES = 0.75

_ORIGINAL_QUALITY_WINDOW = None
_ORIGINAL_FIT = None
_INSTALLED = False


def _paved_piece_family(pieces) -> str | None:
    family = None
    found = False
    for piece in pieces:
        path = str(piece.model_path)
        match = _geometry.stock_straight_match(path)
        if match is None:
            match = _geometry.stock_curve_match(path)
        if match is None:
            return None
        current = match.group("family").casefold()
        if current not in _PAVED_FAMILIES:
            return None
        if family is None:
            family = current
        elif current != family:
            return None
        found = True
    return family if found else None


def _quality_window(
    measure,
    pieces,
    start_distance,
    preferred_end,
    minimum_end,
    maximum_end,
    context,
):
    """Let paved approaches penetrate stock junction footprints by 0.55 m."""

    if _ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("Kodiak reference policy is not installed")
    current = list(
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
    if _paved_piece_family(pieces) is None or not pieces:
        return tuple(current)

    start_junction = context.junctions.get(_p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_p._road_node_key(measure.points[-1]))
    if start_junction is None and end_junction is None:
        return tuple(current)

    shortest = min(float(piece.length_metres) for piece in pieces)
    desired_start = float(current[0])
    desired_end = float(current[1])

    if start_junction is not None:
        exit_distance = _quality._exit_distance(
            start_junction,
            _quality._end_direction(measure, start=True),
        )
        desired_start = max(
            float(_quality._JUNCTION_MIN_TRIM),
            float(exit_distance) - KODIAK_PAVED_JUNCTION_OVERLAP_METRES,
        )

    if end_junction is not None:
        exit_distance = _quality._exit_distance(
            end_junction,
            _quality._end_direction(measure, start=False),
        )
        end_trim = max(
            float(_quality._JUNCTION_MIN_TRIM),
            float(exit_distance) - KODIAK_PAVED_JUNCTION_OVERLAP_METRES,
        )
        desired_end = float(measure.total) - end_trim

    # Do not consume a genuinely tiny junction-to-junction run.  The existing
    # short-run fallback remains the safer owner in that case.
    if float(measure.total) < desired_start + (float(measure.total) - desired_end) + shortest * 0.60:
        return tuple(current)

    if start_junction is not None:
        current[0] = min(float(current[0]), desired_start)
    if end_junction is not None:
        current[1] = max(float(current[1]), desired_end)
        current[3] = max(float(current[3]), float(current[1]))
    return tuple(current)


def _source_node_for_native_cap(cap, incident_map):
    logical = _paved._logical_center(cap)
    if logical is None:
        return None
    if not incident_map:
        return tuple(logical)

    nearest = min(
        incident_map.values(),
        key=lambda value: math.dist(tuple(value[0]), tuple(logical)),
        default=None,
    )
    if nearest is not None and math.dist(tuple(nearest[0]), tuple(logical)) <= KODIAK_SOURCE_NODE_RECOVERY_METRES:
        return tuple(nearest[0])
    return tuple(logical)


def _drop_native_node_stubs(report, dataset, projection, spec):
    """Remove a stock paved short that exists only from node to native connector."""

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report

    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    nodes = []
    for cap in report.objects[:cap_count]:
        if _paved._native_signature(str(cap.model_path)) is None:
            continue
        node = _source_node_for_native_cap(cap, incident_map)
        if node is not None:
            nodes.append(node)
    if not nodes:
        return report

    limit = (
        float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
        + KODIAK_NATIVE_CONNECTOR_MARGIN_METRES
    )
    remove_ids: set[int] = set()
    for obj in report.objects[cap_count:]:
        match = _geometry.stock_straight_match(str(obj.model_path))
        if match is None or match.group("family").casefold() not in _PAVED_FAMILIES:
            continue
        length = float(
            _geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(match.group("length"))]
        )
        axis = _p._model_axis(obj, length)
        for node in nodes:
            distances = tuple(math.dist(tuple(node), tuple(endpoint)) for endpoint in axis)
            if (
                min(distances) <= KODIAK_NATIVE_NODE_ENDPOINT_TOLERANCE_METRES
                and max(distances) <= limit
            ):
                remove_ids.add(int(obj.object_id))
                break

    if not remove_ids:
        return report
    return replace(
        report,
        objects=tuple(
            obj for obj in report.objects if int(obj.object_id) not in remove_ids
        ),
    )


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("Kodiak reference policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    return _drop_native_node_stubs(report, dataset, projection, spec)


def install_stock_road_kodiak_reference_policy() -> None:
    """Install the paved-only Kodiak road-placement refinements last."""

    global _ORIGINAL_QUALITY_WINDOW, _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    if not _reference._INSTALLED:
        raise RuntimeError("reference WRP paved-road policy must install first")

    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _ORIGINAL_FIT = _p.fit_road_objects

    # Make native curves normal construction primitives.  The beam still has to
    # satisfy exact stock connector geometry, source-corridor and tangent gates.
    _curve_usage._MINIMUM_BASELINE_SHORT_STRAIGHTS = 0
    _curve_usage._MINIMUM_TOTAL_TURN_DEGREES = KODIAK_MINIMUM_CURVE_PROMOTION_TURN_DEGREES
    _curve_usage._MAXIMUM_TOTAL_TURN_DEGREES = KODIAK_MAXIMUM_CURVE_PROMOTION_TURN_DEGREES
    _curve_usage._MAXIMUM_PROMOTION_RUN_METRES = KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES
    _curve_usage._MAXIMUM_EXTRA_PIECES = KODIAK_MAXIMUM_EXTRA_CURVE_PIECES
    _sharp._MAXIMUM_SPAN_METRES = KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES

    # Apply the measured half-metre paved-junction overlap after native ownership
    # restored the connector window.  This is longitudinal penetration of the
    # correct approach into the junction mesh, not another cross-axis repair slab.
    _quality._quality_window = _quality_window

    # Keep the cleanup outermost so no older fit wrapper can resurrect a short
    # node-to-connector road underneath a purpose-built T/X.
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
