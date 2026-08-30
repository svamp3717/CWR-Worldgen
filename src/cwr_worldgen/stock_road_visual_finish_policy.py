# SPDX-License-Identifier: GPL-3.0-or-later
"""Final visual safeguards for stock-road junctions and curve seams.

CWA renders the whole road strip, not just its centreline. An unsupported skew T
keeps the legacy six-metre straight cap, and the core fitter may orient that
symmetric cap along the side arm. In game that produces a conspicuous rectangular
road slab across the main carriageway even though the logical road graph is valid.

Align every legacy stock cap with the most nearly continuous incident pair. For
ordinary bends, keep native curves intact and cover only real two-piece seams
whose rendered tangent axes differ enough to expose a narrow grass wedge. The
cover is an ordinary same-family six-metre straight placed slightly below both
road pieces, so it cannot replace the visible curve or create a new raised slab.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local_fit
from . import stock_road_model_geometry as _model_geometry

LEGACY_CAP_AXIS_TOLERANCE_DEGREES = 0.50
MINIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES = 0.20
MAXIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES = 3.25
CURVE_SEAM_ENDPOINT_TOLERANCE_METRES = 0.03
CURVE_SEAM_COVER_VERTICAL_BIAS_METRES = -0.010

_ORIGINAL_FIT = None
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _SeamEndpoint:
    point: tuple[float, float]
    object_id: int
    endpoint_index: int
    family: str
    tangent_axis_degrees: float
    is_curve: bool


@dataclass(frozen=True, slots=True)
class _SeamCoverPlan:
    model_path: str
    centre: tuple[float, float]
    tangent_axis_degrees: float
    turn_degrees: float = 0.0


def _axis_heading_difference(first: float, second: float) -> float:
    """Angular difference for a symmetric road-surface axis."""

    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, abs(180.0 - difference))


def _average_axis_heading(first: float, second: float) -> float:
    """Average two undirected headings without a 0/180-degree discontinuity."""

    first_radians = math.radians(float(first) * 2.0)
    second_radians = math.radians(float(second) * 2.0)
    sine = math.sin(first_radians) + math.sin(second_radians)
    cosine = math.cos(first_radians) + math.cos(second_radians)
    if abs(sine) <= 1.0e-12 and abs(cosine) <= 1.0e-12:
        return float(first) % 180.0
    return (math.degrees(math.atan2(sine, cosine)) * 0.5) % 180.0


def _junction_incident_map(dataset, projection, spec):
    raw = {}
    positions = {}
    for feature, projected in zip(
        dataset.roads, _p.projected_road_polylines(dataset, projection)
    ):
        if not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = tuple(_p._clean_road_points(projected))
        if len(points) < 2:
            continue
        model = _p.road_model_for_tags(spec, feature.tags)
        dirt = _p.road_is_dirt(feature.tags)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= 0.05:
                continue
            segment_key = f"{feature.osm_key}/{index:06d}"
            forward = _p._normalised_direction(start, end)
            reverse = (-forward[0], -forward[1])
            start_key = _p._road_node_key(start)
            end_key = _p._road_node_key(end)
            raw.setdefault(start_key, []).append(
                (forward, dirt, model, segment_key, feature.osm_key)
            )
            raw.setdefault(end_key, []).append(
                (reverse, dirt, model, segment_key, feature.osm_key)
            )
            positions.setdefault(start_key, start)
            positions.setdefault(end_key, end)

    result = {}
    for key, values in raw.items():
        unique = _p._unique_incidents(values)
        if len(unique) not in {3, 4}:
            continue
        incidents = tuple(
            _junction._Incident(value[0], _junction._family(value[2]), value[2])
            for value in unique
        )
        result[key] = (positions[key], incidents)
    return result


def _dominant_cap_heading(incidents, family: str) -> float | None:
    """Return the continuous same-family axis a legacy cap should follow."""

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    if incidents[first].family != family or incidents[second].family != family:
        return None
    return _junction._heading(incidents[first].direction)


def _realign_legacy_caps(report, dataset, projection, elevations, spec):
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report

    incident_map = _junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return report

    objects = list(report.objects)
    changed = False
    for index in range(cap_count):
        old = objects[index]
        match = _model_geometry.stock_straight_match(old.model_path)
        if match is None or int(match.group("length")) != 6:
            continue
        family = match.group("family").casefold()
        key = _p._road_node_key((float(old.x), float(old.z)))
        junction = incident_map.get(key)
        if junction is None:
            continue
        node, incidents = junction
        if math.dist((float(old.x), float(old.z)), node) > 0.25:
            continue
        heading = _dominant_cap_heading(incidents, family)
        if heading is None:
            continue
        if (
            _axis_heading_difference(float(old.heading_degrees), heading)
            <= LEGACY_CAP_AXIS_TOLERANCE_DEGREES
        ):
            continue

        half = _model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6] * 0.5
        angle = math.radians(heading)
        direction = (math.sin(angle), math.cos(angle))
        start = (
            node[0] - direction[0] * half,
            node[1] - direction[1] * half,
        )
        end = (
            node[0] + direction[0] * half,
            node[1] + direction[1] * half,
        )
        fixed = _p._road_object_on_slope(
            int(old.object_id),
            old.model_path,
            start,
            end,
            elevations,
            spec,
            vertical_offset=(
                _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                + _local_fit.LEGACY_CAP_VERTICAL_BIAS_METRES
            ),
        )
        objects[index] = replace(
            fixed,
            x=float(node[0]),
            z=float(node[1]),
            heading_degrees=heading % 360.0,
        )
        changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def _chord_heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _piece_tangents(measure, piece, start, end) -> tuple[float, float]:
    """Return rendered centreline tangent headings at a fitted piece's ends."""

    chord = _chord_heading(start, end)
    if _model_geometry.stock_curve_match(piece.model_path) is None:
        return chord, chord

    source_start = _p._nearest_polyline_heading(measure.points, start)
    source_end = _p._nearest_polyline_heading(measure.points, end)
    signed_turn = _p._signed_heading_delta(source_start, source_end)
    half_turn = _model_geometry.STOCK_CURVE_ANGLE_DEGREES * 0.5
    if signed_turn < 0.0:
        return (chord + half_turn) % 360.0, (chord - half_turn) % 360.0
    return (chord - half_turn) % 360.0, (chord + half_turn) % 360.0


def _stock_piece_geometry(model_path: str):
    straight = _model_geometry.stock_straight_match(model_path)
    if straight is not None:
        return (
            straight.group("family").casefold(),
            _model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(straight.group("length"))],
            False,
        )
    curve = _model_geometry.stock_curve_connectors(model_path)
    if curve is not None:
        return curve.family, curve.chord_length_metres, True
    return None


def _curve_endpoint_tangent_axis(obj, point: tuple[float, float]) -> float:
    """Return the rendered curve tangent axis at one physical connector."""

    geometry = _model_geometry.stock_curve_connectors(obj.model_path)
    if geometry is None:
        return float(obj.heading_degrees) % 180.0

    origin = (float(obj.x), float(obj.z))
    begin = _model_geometry.transform_local(
        geometry.begin, origin, float(obj.heading_degrees)
    )
    end = _model_geometry.transform_local(
        geometry.end, origin, float(obj.heading_degrees)
    )
    local_turn = (
        0.0
        if math.dist(point, begin) <= math.dist(point, end)
        else _model_geometry.STOCK_CURVE_ANGLE_DEGREES
    )
    return (float(obj.heading_degrees) + local_turn) % 180.0


def _seam_endpoints(report) -> tuple[_SeamEndpoint, ...]:
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    endpoints = []
    for obj in report.objects[cap_count:]:
        geometry = _stock_piece_geometry(str(obj.model_path))
        if geometry is None:
            continue
        family, length, is_curve = geometry
        axis = _p._model_axis(obj, float(length))
        for endpoint_index, point in enumerate(axis):
            point = (float(point[0]), float(point[1]))
            endpoints.append(
                _SeamEndpoint(
                    point=point,
                    object_id=int(obj.object_id),
                    endpoint_index=endpoint_index,
                    family=family,
                    tangent_axis_degrees=(
                        _curve_endpoint_tangent_axis(obj, point)
                        if is_curve
                        else float(obj.heading_degrees) % 180.0
                    ),
                    is_curve=is_curve,
                )
            )
    return tuple(endpoints)


def _endpoint_bucket(point: tuple[float, float]) -> tuple[int, int]:
    size = CURVE_SEAM_ENDPOINT_TOLERANCE_METRES
    return math.floor(point[0] / size), math.floor(point[1] / size)


def _curve_seam_cover_plans(report) -> tuple[_SeamCoverPlan, ...]:
    """Plan low same-family underlays for isolated curve surface wedges."""

    endpoints = _seam_endpoints(report)
    if not endpoints:
        return ()

    buckets: dict[tuple[int, int], list[_SeamEndpoint]] = {}
    for endpoint in endpoints:
        buckets.setdefault(_endpoint_bucket(endpoint.point), []).append(endpoint)

    plans = []
    used_pairs = set()
    tolerance = CURVE_SEAM_ENDPOINT_TOLERANCE_METRES
    for endpoint in endpoints:
        bx, bz = _endpoint_bucket(endpoint.point)
        neighbours = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for candidate in buckets.get((bx + dx, bz + dz), ()):
                    if math.dist(endpoint.point, candidate.point) <= tolerance:
                        neighbours.append(candidate)

        # Exactly two physical chain endpoints means an ordinary seam. Three or
        # more means a junction/overlap area, which is owned by the junction
        # policies and must never receive a diagonal turn-repair slab.
        unique = {
            (candidate.object_id, candidate.endpoint_index): candidate
            for candidate in neighbours
        }
        if len(unique) != 2:
            continue
        first, second = sorted(
            unique.values(), key=lambda item: (item.object_id, item.endpoint_index)
        )
        if first.object_id == second.object_id or first.family != second.family:
            continue
        if not (first.is_curve or second.is_curve):
            continue

        pair_key = (
            first.object_id,
            first.endpoint_index,
            second.object_id,
            second.endpoint_index,
        )
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        error = _axis_heading_difference(
            first.tangent_axis_degrees, second.tangent_axis_degrees
        )
        if not (
            MINIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES
            <= error
            <= MAXIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES
        ):
            continue

        plans.append(
            _SeamCoverPlan(
                model_path=rf"o\road\{first.family}6.p3d",
                centre=(
                    (first.point[0] + second.point[0]) * 0.5,
                    (first.point[1] + second.point[1]) * 0.5,
                ),
                tangent_axis_degrees=_average_axis_heading(
                    first.tangent_axis_degrees, second.tangent_axis_degrees
                ),
            )
        )
    return tuple(plans)


def _apply_curve_seam_covers(report, elevations, spec):
    plans = _curve_seam_cover_plans(report)
    if not plans:
        return report

    required = len(report.objects) + len(plans)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after curve seam coverage: "
            f"requires {required:,} objects, limit is {spec.max_road_objects:,}"
        )

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    half = _model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6] * 0.5
    for plan in plans:
        angle = math.radians(plan.tangent_axis_degrees)
        direction = (math.sin(angle), math.cos(angle))
        start = (
            plan.centre[0] - direction[0] * half,
            plan.centre[1] - direction[1] * half,
        )
        end = (
            plan.centre[0] + direction[0] * half,
            plan.centre[1] + direction[1] * half,
        )
        objects.append(
            _p._road_object_on_slope(
                next_id,
                plan.model_path,
                start,
                end,
                elevations,
                spec,
                vertical_offset=(
                    _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                    + CURVE_SEAM_COVER_VERTICAL_BIAS_METRES
                ),
            )
        )
        next_id += 1

    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0)) + len(plans)
        ),
    )


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id=1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock road visual finish policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return report
    report = _realign_legacy_caps(report, dataset, projection, elevations, spec)
    return _apply_curve_seam_covers(report, elevations, spec)


def install_stock_road_visual_finish_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
