# SPDX-License-Identifier: GPL-3.0-or-later
"""Own paved-road behavior learned from hand-authored reference WRPs.

The CEEB and Kodiak reference worlds point to one consistent construction style:
stock paved pieces use planar/yaw-only transforms, native ten-degree curves are
normal turn primitives, and paved approaches overlap purpose-built junction
meshes slightly instead of stopping short or leaving a stale node-to-connector
stub underneath the junction.

Keep the two historical installation stages because timing still matters. The
first reference stage establishes paved transform and curve-promotion semantics.
The later Kodiak stage widens the exact-curve window, applies the measured paved
junction overlap, and owns the one remaining reference-based fitter cleanup.
Keeping both installers in this owner removes another file boundary without
changing their position in the production pipeline.

Stock ``ces`` and generated gravel retain their terrain-following 3D connector
policy and are deliberately excluded from the paved reference refinements.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_3d_connector_policy as _three_d
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_model_geometry as _geometry
from . import stock_road_paved_junction_completion_policy as _paved
from . import stock_road_sharp_turn_policy as _sharp
from . import stock_road_visual_finish_policy as _finish


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_NATIVE_T = re.compile(
    r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?:sil|ces|asf|kos)_t\.p3d$",
    re.IGNORECASE,
)
_NATIVE_X = re.compile(
    r"^(?:.*[\\/])kr_new_silxsil\.p3d$",
    re.IGNORECASE,
)

REFERENCE_MINIMUM_BASELINE_SHORT_STRAIGHTS = 2
REFERENCE_MINIMUM_TOTAL_TURN_DEGREES = 8.0
REFERENCE_MINIMUM_PROMOTED_CURVES = 1
REFERENCE_MAXIMUM_EXTRA_CURVE_PIECES = 3
REFERENCE_INSPECTOR_CURVE_MINIMUM_TURN_DEGREES = 3.0

KODIAK_PAVED_JUNCTION_OVERLAP_METRES = 0.55
KODIAK_MINIMUM_CURVE_PROMOTION_TURN_DEGREES = 5.0
KODIAK_MAXIMUM_CURVE_PROMOTION_TURN_DEGREES = 210.0
KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES = 420.0
KODIAK_MAXIMUM_EXTRA_CURVE_PIECES = 5
KODIAK_SOURCE_NODE_RECOVERY_METRES = 2.0
KODIAK_NATIVE_NODE_ENDPOINT_TOLERANCE_METRES = 0.30
KODIAK_NATIVE_CONNECTOR_MARGIN_METRES = 0.75

_ORIGINAL_ROAD_OBJECT_ON_SLOPE = None
_ORIGINAL_USES_MEASURED_RIGID_CONNECTORS = None
_ORIGINAL_QUALITY_WINDOW = None
_ORIGINAL_FIT = None
_INSTALLED = False
_KODIAK_INSTALLED = False


def _stock_family(model_path: str) -> str | None:
    straight = _geometry.stock_straight_match(str(model_path))
    if straight is not None:
        return straight.group("family").casefold()
    curve = _geometry.stock_curve_match(str(model_path))
    if curve is not None:
        return curve.group("family").casefold()
    return None


def _is_paved_stock_surface(model_path: str) -> bool:
    """Return True for stock paved straights, curves and native paved junctions."""

    family = _stock_family(model_path)
    if family is not None:
        return family in _PAVED_FAMILIES
    normalised = str(model_path).replace("/", "\\")
    return (
        _NATIVE_T.fullmatch(normalised) is not None
        or _NATIVE_X.fullmatch(normalised) is not None
    )


def _uses_measured_rigid_connectors(pieces) -> bool:
    """Keep terrain-length fitting for dirt/gravel, but not stock paved roads."""

    if _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS is None:
        raise RuntimeError("reference WRP road policy is not installed")

    families = set()
    for piece in pieces:
        model_path = str(piece.model_path)
        if _p.is_generated_gravel_road_model(model_path):
            return _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS(pieces)
        family = _stock_family(model_path)
        if family is not None:
            families.add(family)

    if families and families <= _PAVED_FAMILIES:
        return False
    return _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS(pieces)


def _road_object_on_slope(*args, **kwargs):
    """Emit paved stock road P3Ds horizontal while preserving their fitted Y."""

    if _ORIGINAL_ROAD_OBJECT_ON_SLOPE is None:
        raise RuntimeError("reference WRP road policy is not installed")
    obj = _ORIGINAL_ROAD_OBJECT_ON_SLOPE(*args, **kwargs)
    model_path = str(args[1] if len(args) > 1 else kwargs.get("model_path", ""))
    if not _is_paved_stock_surface(model_path):
        return obj
    if abs(float(obj.pitch_degrees)) <= 1.0e-12:
        return obj
    return replace(obj, pitch_degrees=0.0)


def _paved_piece_family(pieces) -> str | None:
    family = None
    found = False
    for piece in pieces:
        current = _stock_family(str(piece.model_path))
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
        raise RuntimeError("Kodiak reference stage is not installed")
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

    if (
        float(measure.total)
        < desired_start
        + (float(measure.total) - desired_end)
        + shortest * 0.60
    ):
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
    if (
        nearest is not None
        and math.dist(tuple(nearest[0]), tuple(logical))
        <= KODIAK_SOURCE_NODE_RECOVERY_METRES
    ):
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
        if (
            match is None
            or match.group("family").casefold() not in _PAVED_FAMILIES
        ):
            continue
        length = float(
            _geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(match.group("length"))]
        )
        axis = _p._model_axis(obj, length)
        for node in nodes:
            distances = tuple(
                math.dist(tuple(node), tuple(endpoint)) for endpoint in axis
            )
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
        raise RuntimeError("Kodiak reference stage is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    return _drop_native_node_stubs(report, dataset, projection, spec)


def install_stock_road_reference_wrp_policy() -> None:
    """Install the baseline paved placement rules learned from the reference WRP."""

    global _ORIGINAL_ROAD_OBJECT_ON_SLOPE
    global _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS
    global _INSTALLED
    if _INSTALLED:
        return

    if not _three_d._INSTALLED:
        raise RuntimeError("3D stock-road connector policy must install first")
    if not _curve_usage._INSTALLED:
        raise RuntimeError("stock road curve-usage policy must install first")
    if not _candidate._INSTALLED:
        raise RuntimeError("Inspector candidate policy must install first")

    _ORIGINAL_ROAD_OBJECT_ON_SLOPE = _p._road_object_on_slope
    _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS = (
        _three_d._uses_measured_rigid_connectors
    )

    _three_d._uses_measured_rigid_connectors = _uses_measured_rigid_connectors
    _p._road_object_on_slope = _road_object_on_slope

    _curve_usage._MINIMUM_BASELINE_SHORT_STRAIGHTS = (
        REFERENCE_MINIMUM_BASELINE_SHORT_STRAIGHTS
    )
    _curve_usage._MINIMUM_TOTAL_TURN_DEGREES = REFERENCE_MINIMUM_TOTAL_TURN_DEGREES
    _curve_usage._MINIMUM_PROMOTED_CURVES = REFERENCE_MINIMUM_PROMOTED_CURVES
    _curve_usage._MAXIMUM_EXTRA_PIECES = REFERENCE_MAXIMUM_EXTRA_CURVE_PIECES
    _candidate.INSPECTOR_CURVE_MINIMUM_TURN_DEGREES = (
        REFERENCE_INSPECTOR_CURVE_MINIMUM_TURN_DEGREES
    )
    _candidate.INSPECTOR_CURVE_MAXIMUM_EXTRA_PIECES = (
        REFERENCE_MAXIMUM_EXTRA_CURVE_PIECES
    )

    _INSTALLED = True


def install_stock_road_kodiak_reference_policy() -> None:
    """Install the later Kodiak curve, junction-overlap and cleanup refinements."""

    global _ORIGINAL_QUALITY_WINDOW, _ORIGINAL_FIT, _KODIAK_INSTALLED
    if _KODIAK_INSTALLED:
        return
    if not _INSTALLED:
        raise RuntimeError("reference WRP paved-road stage must install first")

    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _ORIGINAL_FIT = _p.fit_road_objects

    _curve_usage._MINIMUM_BASELINE_SHORT_STRAIGHTS = 0
    _curve_usage._MINIMUM_TOTAL_TURN_DEGREES = (
        KODIAK_MINIMUM_CURVE_PROMOTION_TURN_DEGREES
    )
    _curve_usage._MAXIMUM_TOTAL_TURN_DEGREES = (
        KODIAK_MAXIMUM_CURVE_PROMOTION_TURN_DEGREES
    )
    _curve_usage._MAXIMUM_PROMOTION_RUN_METRES = (
        KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES
    )
    _curve_usage._MAXIMUM_EXTRA_PIECES = KODIAK_MAXIMUM_EXTRA_CURVE_PIECES
    _sharp._MAXIMUM_SPAN_METRES = KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES

    _quality._quality_window = _quality_window

    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _KODIAK_INSTALLED = True
