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
native T/X nodes, restore the quality fitter's original connector trim so exact
chains target the native footprint. As a final physical guard, if an ordinary
paved stock straight still crosses a fitted native centre, replace only the part
outside the native footprint with the exact 6.25/12.5/25 m stock straight
sequence. A 25 m road ending at the logical node therefore becomes a 12.5 m plus
6.25 m approach beginning at the measured connector instead of remaining under
the intersection. A co-centred short slab disappears entirely.

The old final paved-completion pass compared a successfully fitted native T back
to the *unmodified* source headings after the connector-relaxation context had
ended. That could demote a correct native T back to a generic ``sil6`` solely
because the original OSM branch was a few degrees skewed. Once the inner fitter
has selected a measured native junction, keep that selection and let its actual
WRP connectors be authoritative. Generic all-paved fallback caps are still kept
low, preserving their existing non-z-fighting behaviour.

Generated gravel and stock ``ces`` are not rewritten by the physical paved guard.
No generated P3D is created.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import gravel_junction_policy as _gravel_junction
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_connector_policy as _connector
from . import stock_road_local_fit_policy as _local
from . import stock_road_model_geometry as _geometry
from . import stock_road_paved_junction_completion_policy as _paved
from . import stock_road_surface_overlap_policy as _surface
from . import stock_road_visual_finish_policy as _finish


_NATIVE_EXTENT_TOLERANCE_METRES = 1.0e-6
_NATIVE_AXIS_INTRUSION_METRES = 0.55
_NATIVE_FOOTPRINT_MARGIN_METRES = 0.20
_CONNECTOR_ALIGNMENT_COSINE = math.cos(math.radians(22.0))
_CONNECTOR_LINE_TOLERANCE_METRES = 0.75
_STOCK_SPAN_TOLERANCE_METRES = 0.30
_ORIGINAL_QUALITY_WINDOW = None
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
        alignment = (
            connector.outward[0] * direction[0]
            + connector.outward[1] * direction[1]
        )
        if alignment < _CONNECTOR_ALIGNMENT_COSINE:
            continue
        if (
            _p._point_segment_distance(connector.point, center, outer)
            > _CONNECTOR_LINE_TOLERANCE_METRES
        ):
            continue
        candidates.append(
            (-alignment, math.dist(connector.point, outer), connector)
        )
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

    direction = _normalised(
        (outer[0] - connector.point[0], outer[1] - connector.point[1])
    )
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

    # Do not manufacture a new outer seam. The stock lengths are exact multiples
    # in normal use; this gate merely protects unusual pitched/legacy geometry.
    if math.dist(cursor, outer) > _STOCK_SPAN_TOLERANCE_METRES:
        return None
    return tuple(built), next_object_id


def _trim_one_native_center(objects, cap, *, cap_count, elevations, spec, next_id):
    center = _paved._logical_center(cap)
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
        if family not in _paved._PAVED_FAMILIES:
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
            # Entire straight lies inside the purpose-built junction footprint.
            # Keeping it can only expose a duplicate stock border.
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
        if _paved._native_signature(str(cap.model_path)) is None:
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

    if _paved._ORIGINAL_REALIGN is None:
        raise RuntimeError("paved junction realignment baseline was not captured")

    report = _paved._ORIGINAL_REALIGN(report, dataset, projection, elevations, spec)
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

        if _paved._native_signature(str(current.model_path)) is not None:
            continue

        family = _paved._cap_family(current)
        if family is None:
            continue
        logical = _paved._logical_center(current)
        if logical is None:
            continue
        junction = _paved._matching_junction(incident_map, logical)
        if junction is None:
            continue
        node, incidents = junction
        if not _paved._all_paved_incidents(incidents):
            continue

        replacement = _paved._low_stock_cap(
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


def install_stock_road_native_junction_ownership_policy() -> None:
    """Install connector-trim ownership after the paved completion layer."""

    global _ORIGINAL_QUALITY_WINDOW, _INSTALLED
    if _INSTALLED:
        return
    if not _paved._INSTALLED:
        raise RuntimeError("paved junction completion policy must install first")
    if _local._ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("stock road local fit policy must install first")
    if _paved._ORIGINAL_REALIGN is None:
        raise RuntimeError("paved junction realignment baseline was not captured")

    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _quality._quality_window = _native_ownership_quality_window
    _surface._quality_window = _native_ownership_quality_window

    if _paved._ORIGINAL_NATIVE_T_TARGETS is None:
        raise RuntimeError("paved connector target planner was not captured")
    _connector._native_t_targets = _paved._ORIGINAL_NATIVE_T_TARGETS

    _finish._realign_legacy_caps = _native_owner_realign

    _INSTALLED = True
