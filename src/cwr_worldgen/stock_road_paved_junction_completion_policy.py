# SPDX-License-Identifier: GPL-3.0-or-later
"""Own final stock-only paved junction completion and native-centre ownership.

Rigid Resistance junction meshes look good only when their measured connectors
nearly match the fitted approach directions. Keep exact native ``sil/asf/kos``
T/X models, demote visibly mismatched all-paved junctions to a low stock six-
metre central fill, and make accepted native junctions own their full visible
footprint.

The second phase restores measured connector trims at accepted native T/X nodes
and removes ordinary paved straights that still intrude through the native
centre. The two phases are installed together because no policy observes the
intermediate state; they are one final paved-junction responsibility.
"""
from __future__ import annotations

from dataclasses import replace
import itertools
import math

from . import gravel_junction_policy as _gravel_junction
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_connector_policy as _connector
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local
from . import stock_road_model_geometry as _geometry
from . import stock_road_surface_overlap_policy as _surface
from . import stock_road_visual_finish_policy as _finish


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES = 0.90
PAVED_JUNCTION_UNDERLAY_BIAS_METRES = -0.006
JUNCTION_NODE_RECOVERY_METRES = 0.40

_NATIVE_EXTENT_TOLERANCE_METRES = 1.0e-6
_NATIVE_AXIS_INTRUSION_METRES = 0.55
_NATIVE_FOOTPRINT_MARGIN_METRES = 0.20
_CONNECTOR_ALIGNMENT_COSINE = math.cos(math.radians(22.0))
_CONNECTOR_LINE_TOLERANCE_METRES = 0.75
_STOCK_SPAN_TOLERANCE_METRES = 0.30

_ORIGINAL_NATIVE_T_TARGETS = None
_ORIGINAL_REALIGN = None
_ORIGINAL_QUALITY_WINDOW = None
_INSTALLED = False
_NATIVE_OWNERSHIP_INSTALLED = False


def _all_paved_incidents(incidents) -> bool:
    return (
        len(incidents) in {3, 4}
        and all(incident.family in _PAVED_FAMILIES for incident in incidents)
    )


def _native_signature(model_path: str):
    normalized = str(model_path).replace("/", "\\").casefold()
    for (main, _branch), candidate in _junction._T_JUNCTION_MODELS.items():
        if str(candidate).casefold() == normalized:
            return main, (0.0, 180.0, 270.0)
    for family, candidate in _junction._X_JUNCTION_MODELS.items():
        if str(candidate).casefold() == normalized:
            return family, (0.0, 90.0, 180.0, 270.0)
    return None


def _connector_error_degrees(cap, incidents, local_headings) -> float:
    source = tuple(_junction._heading(incident.direction) for incident in incidents)
    connectors = tuple(
        (float(cap.heading_degrees) + float(local)) % 360.0
        for local in local_headings
    )
    if len(source) != len(connectors):
        return 180.0
    best = math.inf
    for assignment in itertools.permutations(connectors):
        error = max(
            _junction._angular_distance(actual, connector)
            for actual, connector in zip(source, assignment)
        )
        best = min(best, error)
    return float(best)


def _native_t_targets(incidents, native):
    """Do not bend paved source approaches toward a junction we will reject."""

    if _ORIGINAL_NATIVE_T_TARGETS is None:
        raise RuntimeError("paved junction completion policy is not installed")
    if (
        _all_paved_incidents(incidents)
        and float(native.maximum_heading_error_degrees)
        > MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES
    ):
        return None
    return _ORIGINAL_NATIVE_T_TARGETS(incidents, native)


def _logical_center(cap) -> tuple[float, float] | None:
    if _geometry.stock_straight_match(str(cap.model_path)) is not None:
        return float(cap.x), float(cap.z)
    local = _geometry.native_junction_intersection_offset(str(cap.model_path))
    if local is None:
        return None
    return _geometry.transform_local(
        local,
        (float(cap.x), float(cap.z)),
        float(cap.heading_degrees),
    )


def _matching_junction(incident_map, point: tuple[float, float]):
    direct = incident_map.get(_p._road_node_key(point))
    if direct is not None and math.dist(point, direct[0]) <= JUNCTION_NODE_RECOVERY_METRES:
        return direct
    nearest = min(
        incident_map.values(),
        key=lambda value: math.dist(point, value[0]),
        default=None,
    )
    if nearest is None or math.dist(point, nearest[0]) > JUNCTION_NODE_RECOVERY_METRES:
        return None
    return nearest


def _cap_family(cap) -> str | None:
    straight = _geometry.stock_straight_match(str(cap.model_path))
    if straight is not None and int(straight.group("length")) == 6:
        family = straight.group("family").casefold()
        return family if family in _PAVED_FAMILIES else None
    signature = _native_signature(str(cap.model_path))
    if signature is None:
        return None
    family, _local_name = signature
    return family if family in _PAVED_FAMILIES else None


def _dominant_heading(incidents, family: str) -> float | None:
    """Prefer the most-opposed pair in the cap family, then any through pair."""

    candidates = []
    for first in range(len(incidents)):
        if incidents[first].family != family:
            continue
        first_heading = _junction._heading(incidents[first].direction)
        for second in range(first + 1, len(incidents)):
            if incidents[second].family != family:
                continue
            second_heading = _junction._heading(incidents[second].direction)
            separation = _junction._angular_distance(first_heading, second_heading)
            candidates.append((abs(180.0 - separation), first, second))
    if candidates:
        _error, first, _second = min(candidates)
        return _junction._heading(incidents[first].direction)

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    return _junction._heading(incidents[pair[0]].direction)


def _low_stock_cap(current, node, incidents, family, elevations, spec):
    heading = _dominant_heading(incidents, family)
    if heading is None:
        return current
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    half = length * 0.5
    angle = math.radians(heading)
    direction = (math.sin(angle), math.cos(angle))
    start = (
        float(node[0]) - direction[0] * half,
        float(node[1]) - direction[1] * half,
    )
    end = (
        float(node[0]) + direction[0] * half,
        float(node[1]) + direction[1] * half,
    )
    fixed = _p._road_object_on_slope(
        int(current.object_id),
        rf"o\road\{family}6.p3d",
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            + PAVED_JUNCTION_UNDERLAY_BIAS_METRES
        ),
    )
    return replace(
        fixed,
        x=float(node[0]),
        z=float(node[1]),
        heading_degrees=heading % 360.0,
    )


def _finish_paved_junctions(report, dataset, projection, elevations, spec):
    """Keep exact native junctions; demote visible paved mismatches to low fill."""

    if _ORIGINAL_REALIGN is None:
        raise RuntimeError("paved junction completion policy is not installed")
    report = _ORIGINAL_REALIGN(report, dataset, projection, elevations, spec)

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report
    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return report

    objects = list(report.objects)
    changed = False
    for index in range(cap_count):
        current = objects[index]
        family = _cap_family(current)
        if family is None:
            continue
        logical = _logical_center(current)
        if logical is None:
            continue
        junction = _matching_junction(incident_map, logical)
        if junction is None:
            continue
        node, incidents = junction
        if not _all_paved_incidents(incidents):
            continue

        signature = _native_signature(str(current.model_path))
        if signature is not None:
            _main_family, local_headings = signature
            error = _connector_error_degrees(current, incidents, local_headings)
            if error <= MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES + 1.0e-9:
                continue

        replacement = _low_stock_cap(
            current,
            node,
            incidents,
            family,
            elevations,
            spec,
        )
        if replacement != current:
            objects[index] = replacement
            changed = True

    return replace(report, objects=tuple(objects)) if changed else report


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

    start_junction = context.junctions.get(_local._p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_local._p._road_node_key(measure.points[-1]))
    restore_start = _is_measured_native_junction(start_junction)
    restore_end = _is_measured_native_junction(end_junction)
    if not restore_start and not restore_end:
        return current

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


def _normalised(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 1.0e-9:
        return (0.0, 1.0)
    return vector[0] / length, vector[1] / length


def _physical_straight_axis(obj):
    match = _geometry.stock_straight_match(str(obj.model_path))
    if match is None:
        return None
    nominal = int(match.group("length"))
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[nominal])
    return (
        _surface._world_point(obj, (0.0, -length * 0.5)),
        _surface._world_point(obj, (0.0, length * 0.5)),
    )


def _matching_connector(connectors, family: str, center, outer):
    direction = _normalised((outer[0] - center[0], outer[1] - center[1]))
    candidates = []
    for connector in connectors:
        if connector.family != family:
            continue
        alignment = connector.outward[0] * direction[0] + connector.outward[1] * direction[1]
        if alignment < _CONNECTOR_ALIGNMENT_COSINE:
            continue
        if (
            _p._point_segment_distance(connector.point, center, outer)
            > _CONNECTOR_LINE_TOLERANCE_METRES
        ):
            continue
        candidates.append((-alignment, math.dist(connector.point, outer), connector))
    return None if not candidates else min(candidates)[2]


def _stock_span_nominals(distance: float) -> tuple[int, ...] | None:
    unit = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    units = int(round(float(distance) / unit))
    if units <= 0:
        return ()
    if abs(float(distance) - units * unit) > _STOCK_SPAN_TOLERANCE_METRES:
        return None
    result = []
    for nominal, count in ((25, 4), (12, 2), (6, 1)):
        while units >= count:
            result.append(nominal)
            units -= count
    return tuple(result) if units == 0 else None


def _build_stock_span(
    family: str,
    connector,
    outer,
    *,
    first_object_id: int,
    next_object_id: int,
    elevations,
    spec,
):
    distance = math.dist(connector.point, outer)
    nominals = _stock_span_nominals(distance)
    if nominals is None:
        return None
    if not nominals:
        return (), next_object_id

    direction = _normalised((outer[0] - connector.point[0], outer[1] - connector.point[1]))
    cursor = tuple(connector.point)
    built = []
    for index, nominal in enumerate(nominals):
        length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[nominal])
        endpoint = (
            cursor[0] + direction[0] * length,
            cursor[1] + direction[1] * length,
        )
        object_id = first_object_id if index == 0 else next_object_id
        if index > 0:
            next_object_id += 1
        built.append(
            _p._road_object_on_slope(
                object_id,
                rf"o\road\{family}{nominal}.p3d",
                cursor,
                endpoint,
                elevations,
                spec,
                vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
            )
        )
        cursor = endpoint

    if math.dist(cursor, outer) > _STOCK_SPAN_TOLERANCE_METRES:
        return None
    return tuple(built), next_object_id


def _trim_one_native_center(objects, cap, *, cap_count, elevations, spec, next_id):
    center = _logical_center(cap)
    if center is None:
        return objects, next_id, False
    connectors = _surface._native_cap_connectors(cap)
    if not connectors:
        return objects, next_id, False

    radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    output = list(objects[:cap_count])
    changed = False
    for obj in objects[cap_count:]:
        match = _geometry.stock_straight_match(str(obj.model_path))
        if match is None:
            output.append(obj)
            continue
        family = match.group("family").casefold()
        if family not in _PAVED_FAMILIES:
            output.append(obj)
            continue
        axis = _physical_straight_axis(obj)
        if axis is None or (
            _p._point_segment_distance(center, axis[0], axis[1])
            > _NATIVE_AXIS_INTRUSION_METRES
        ):
            output.append(obj)
            continue

        outside = [
            endpoint
            for endpoint in axis
            if math.dist(center, endpoint) > radius + _NATIVE_FOOTPRINT_MARGIN_METRES
        ]
        if not outside:
            changed = True
            continue

        replacements = []
        candidate_next_id = next_id
        valid = True
        first_id_available = int(obj.object_id)
        for outer in outside:
            connector = _matching_connector(connectors, family, center, outer)
            if connector is None:
                valid = False
                break
            built = _build_stock_span(
                family,
                connector,
                outer,
                first_object_id=first_id_available,
                next_object_id=candidate_next_id,
                elevations=elevations,
                spec=spec,
            )
            if built is None:
                valid = False
                break
            pieces, candidate_next_id = built
            if pieces:
                replacements.extend(pieces)
                first_id_available = candidate_next_id
                candidate_next_id += 1

        if not valid:
            output.append(obj)
            continue
        output.extend(replacements)
        next_id = candidate_next_id
        changed = True

    return output, next_id, changed


def _trim_native_center_intruders(report, elevations, spec):
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    changed = False
    for cap in tuple(objects[:cap_count]):
        if _native_signature(str(cap.model_path)) is None:
            continue
        objects, next_id, trimmed = _trim_one_native_center(
            objects,
            cap,
            cap_count=cap_count,
            elevations=elevations,
            spec=spec,
            next_id=next_id,
        )
        changed = changed or trimmed
    return replace(report, objects=tuple(objects)) if changed else report


def _native_owner_realign(report, dataset, projection, elevations, spec):
    """Keep fitted native caps; retain the low generic paved fallback."""

    if _ORIGINAL_REALIGN is None:
        raise RuntimeError("paved junction realignment baseline was not captured")

    report = _ORIGINAL_REALIGN(report, dataset, projection, elevations, spec)
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report

    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return _trim_native_center_intruders(report, elevations, spec)

    objects = list(report.objects)
    changed = False
    for index in range(cap_count):
        current = objects[index]

        if _native_signature(str(current.model_path)) is not None:
            continue

        family = _cap_family(current)
        if family is None:
            continue
        logical = _logical_center(current)
        if logical is None:
            continue
        junction = _matching_junction(incident_map, logical)
        if junction is None:
            continue
        node, incidents = junction
        if not _all_paved_incidents(incidents):
            continue

        replacement = _low_stock_cap(
            current,
            node,
            incidents,
            family,
            elevations,
            spec,
        )
        if replacement != current:
            objects[index] = replacement
            changed = True

    if changed:
        report = replace(report, objects=tuple(objects))
    return _trim_native_center_intruders(report, elevations, spec)


def install_stock_road_paved_junction_completion_policy() -> None:
    """Install all-paved selection and fallback completion."""

    global _ORIGINAL_NATIVE_T_TARGETS, _ORIGINAL_REALIGN, _INSTALLED
    if _INSTALLED:
        return
    if not _finish._INSTALLED:
        raise RuntimeError("stock road visual-finish policy must install first")

    _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = (
        MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES
    )
    _ORIGINAL_NATIVE_T_TARGETS = _connector._native_t_targets
    _ORIGINAL_REALIGN = _finish._realign_legacy_caps
    _connector._native_t_targets = _native_t_targets
    _finish._realign_legacy_caps = _finish_paved_junctions
    _INSTALLED = True


def install_stock_road_native_junction_ownership_policy() -> None:
    """Finish accepted native T/X ownership immediately after completion."""

    global _ORIGINAL_QUALITY_WINDOW, _NATIVE_OWNERSHIP_INSTALLED
    if _NATIVE_OWNERSHIP_INSTALLED:
        return
    if not _INSTALLED:
        raise RuntimeError("paved junction completion policy must install first")
    if _local._ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("stock road local fit policy must install first")
    if _ORIGINAL_REALIGN is None:
        raise RuntimeError("paved junction realignment baseline was not captured")

    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _quality._quality_window = _native_ownership_quality_window
    _surface._quality_window = _native_ownership_quality_window

    if _ORIGINAL_NATIVE_T_TARGETS is None:
        raise RuntimeError("paved connector target planner was not captured")
    _connector._native_t_targets = _ORIGINAL_NATIVE_T_TARGETS

    _finish._realign_legacy_caps = _native_owner_realign
    _NATIVE_OWNERSHIP_INSTALLED = True
