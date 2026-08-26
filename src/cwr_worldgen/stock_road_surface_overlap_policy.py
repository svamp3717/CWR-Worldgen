# SPDX-License-Identifier: GPL-3.0-or-later
"""Close stock-road junction seams with measured, post-fit overlap pieces.

The stock road fitter connects centreline geometry, while Resistance junction
models expose fixed Memory-LOD connectors. Small heading/trim differences can
therefore leave a visible terrain strip between an otherwise-correct branch and
the junction mesh even though the road network is logically connected.

Do not move the source road. After fitting, inspect the geometry that will
actually be written to the WRP. Where a stock junction connector is within a
small bounded distance of a same-family branch endpoint, add one ordinary 6 m
road piece underneath the junction to bridge that exact connector gap. The
junction remains slightly higher, so its proper intersection surface stays
visible while the underlay prevents terrain from showing through.

Generated gravel junction models are not handled here. Mixed generated-gravel
and paved nodes already use a normal paved cap with gravel continuing beneath it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

from . import generator as _generator
from . import gravel_junction_policy as _gravel_junction
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_model_geometry as _geometry

MINIMUM_CONNECTOR_COVER_GAP_METRES = 0.05
MAXIMUM_CONNECTOR_COVER_GAP_METRES = 3.25
CONNECTOR_COVER_SPAN_METRES = 6.00
CONNECTOR_COVER_VERTICAL_BIAS_METRES = 0.003
_ENDPOINT_BUCKET_METRES = 4.0
_PAVED_FAMILIES = {"sil", "asf", "kos"}
_STOCK_FAMILIES = _PAVED_FAMILIES | {"ces"}

_T_JUNCTION = re.compile(
    r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?P<branch>sil|ces|asf|kos)_t\.p3d$",
    re.IGNORECASE,
)
_X_JUNCTION = re.compile(
    r"^(?:.*[\\/])kr_new_silxsil\.p3d$",
    re.IGNORECASE,
)

_ORIGINAL_QUALITY_WINDOW = None
_ORIGINAL_FIT = None
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _Endpoint:
    point: tuple[float, float]
    object_id: int
    endpoint_index: int
    family: str


@dataclass(frozen=True, slots=True)
class _Connector:
    point: tuple[float, float]
    outward: tuple[float, float]
    family: str


@dataclass(frozen=True, slots=True)
class _CoverPlan:
    model_path: str
    centre: tuple[float, float]
    direction: tuple[float, float]


def _stock_family(model_path: str) -> str | None:
    match = _geometry.stock_straight_match(model_path)
    if match is not None:
        return match.group("family").casefold()
    match = _geometry.stock_curve_match(model_path)
    if match is not None:
        return match.group("family").casefold()
    return None


def _piece_length(model_path: str) -> float | None:
    length = _geometry.stock_straight_length(model_path)
    if length is not None:
        return float(length)
    curve = _geometry.stock_curve_connectors(model_path)
    if curve is not None:
        return float(curve.chord_length_metres)
    return None


def _normalised(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 1.0e-9:
        return (0.0, 1.0)
    return vector[0] / length, vector[1] / length


def _world_point(obj, local: tuple[float, float]) -> tuple[float, float]:
    """Transform a local X/Z point through the WRP yaw/pitch convention."""

    x, z = local
    heading = math.radians(float(obj.heading_degrees))
    pitch = math.radians(float(obj.pitch_degrees))
    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    cosine_pitch = math.cos(pitch)
    return (
        float(obj.x) + x * cosine_heading + z * sine_heading * cosine_pitch,
        float(obj.z) - x * sine_heading + z * cosine_heading * cosine_pitch,
    )


def _object_axis(obj) -> tuple[tuple[float, float], tuple[float, float]] | None:
    length = _piece_length(obj.model_path)
    if length is None:
        return None
    return _p._model_axis(obj, length)


def _chain_endpoints(objects) -> tuple[_Endpoint, ...]:
    result = []
    for obj in objects:
        family = _stock_family(obj.model_path)
        if family not in _STOCK_FAMILIES:
            continue
        axis = _object_axis(obj)
        if axis is None:
            continue
        result.append(_Endpoint(tuple(axis[0]), int(obj.object_id), 0, family))
        result.append(_Endpoint(tuple(axis[1]), int(obj.object_id), 1, family))
    return tuple(result)


def _bucket_key(point: tuple[float, float]) -> tuple[int, int]:
    return (
        math.floor(point[0] / _ENDPOINT_BUCKET_METRES),
        math.floor(point[1] / _ENDPOINT_BUCKET_METRES),
    )


def _endpoint_index(endpoints):
    buckets: dict[tuple[str, int, int], list[_Endpoint]] = {}
    for endpoint in endpoints:
        bx, bz = _bucket_key(endpoint.point)
        buckets.setdefault((endpoint.family, bx, bz), []).append(endpoint)
    return buckets


def _nearest_endpoint(buckets, connector: _Connector) -> tuple[float, _Endpoint] | None:
    bx, bz = _bucket_key(connector.point)
    radius = max(
        1,
        int(
            math.ceil(
                MAXIMUM_CONNECTOR_COVER_GAP_METRES / _ENDPOINT_BUCKET_METRES
            )
        ),
    )
    best = None
    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            for endpoint in buckets.get(
                (connector.family, bx + dx, bz + dz), ()
            ):
                distance = math.dist(connector.point, endpoint.point)
                if distance > MAXIMUM_CONNECTOR_COVER_GAP_METRES + 1.0e-9:
                    continue
                candidate = (
                    distance,
                    endpoint.object_id,
                    endpoint.endpoint_index,
                    endpoint,
                )
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
    return None if best is None else (best[0], best[3])


def _native_cap_connectors(obj) -> tuple[_Connector, ...]:
    path = str(obj.model_path).replace("/", "\\")
    match = _T_JUNCTION.fullmatch(path)
    if match is not None:
        main = match.group("main").casefold()
        branch = match.group("branch").casefold()
        center = _geometry.native_junction_intersection_offset(path)
        if center is None:
            return ()
        cx, cz = center
        local = (
            (
                (cx, cz + _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES),
                main,
            ),
            (
                (cx, cz - _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES),
                main,
            ),
            (
                (cx - _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES, cz),
                branch,
            ),
        )
        center_world = _world_point(obj, center)
        result = []
        for point, family in local:
            world = _world_point(obj, point)
            result.append(
                _Connector(
                    world,
                    _normalised(
                        (
                            world[0] - center_world[0],
                            world[1] - center_world[1],
                        )
                    ),
                    family,
                )
            )
        return tuple(result)

    if _X_JUNCTION.fullmatch(path) is not None:
        radius = _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
        center_world = _world_point(obj, (0.0, 0.0))
        result = []
        for point in (
            (0.0, radius),
            (0.0, -radius),
            (radius, 0.0),
            (-radius, 0.0),
        ):
            world = _world_point(obj, point)
            result.append(
                _Connector(
                    world,
                    _normalised(
                        (
                            world[0] - center_world[0],
                            world[1] - center_world[1],
                        )
                    ),
                    "sil",
                )
            )
        return tuple(result)

    # Mixed generated-gravel/paved nodes intentionally use a normal stock
    # straight as their visible cap. Treat its two physical ends as connectors
    # so the paved approaches can be underlaid up to the cap as well.
    family = _stock_family(path)
    if family not in _PAVED_FAMILIES:
        return ()
    axis = _object_axis(obj)
    if axis is None:
        return ()
    center = (float(obj.x), float(obj.z))
    return tuple(
        _Connector(
            tuple(point),
            _normalised((point[0] - center[0], point[1] - center[1])),
            family,
        )
        for point in axis
    )


def _cover_model(family: str) -> str:
    return rf"o\road\{family}6.p3d"


def _connector_cover_plans(report) -> tuple[_CoverPlan, ...]:
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return ()
    caps = report.objects[:cap_count]
    chains = report.objects[cap_count:]
    endpoints = _chain_endpoints(chains)
    if not endpoints:
        return ()
    buckets = _endpoint_index(endpoints)

    used_endpoints: set[tuple[int, int]] = set()
    plans: list[_CoverPlan] = []
    for cap in caps:
        for connector in _native_cap_connectors(cap):
            if connector.family not in _STOCK_FAMILIES:
                continue
            nearest = _nearest_endpoint(buckets, connector)
            if nearest is None:
                continue
            gap, endpoint = nearest
            endpoint_key = (endpoint.object_id, endpoint.endpoint_index)
            if endpoint_key in used_endpoints:
                continue
            if gap < MINIMUM_CONNECTOR_COVER_GAP_METRES:
                continue
            if gap > MAXIMUM_CONNECTOR_COVER_GAP_METRES:
                continue

            centre = (
                (connector.point[0] + endpoint.point[0]) * 0.5,
                (connector.point[1] + endpoint.point[1]) * 0.5,
            )
            plans.append(
                _CoverPlan(
                    _cover_model(connector.family),
                    centre,
                    connector.outward,
                )
            )
            used_endpoints.add(endpoint_key)
    return tuple(plans)


def _apply_connector_covers(report, elevations, spec):
    plans = _connector_cover_plans(report)
    if not plans:
        return report

    required = len(report.objects) + len(plans)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after seam-safe junction coverage: "
            f"requires {required:,} objects, limit is {spec.max_road_objects:,}"
        )

    next_id = max((int(obj.object_id) for obj in report.objects), default=0) + 1
    objects = list(report.objects)
    half_span = CONNECTOR_COVER_SPAN_METRES * 0.5
    for plan in plans:
        direction = _normalised(plan.direction)
        start = (
            plan.centre[0] - direction[0] * half_span,
            plan.centre[1] - direction[1] * half_span,
        )
        end = (
            plan.centre[0] + direction[0] * half_span,
            plan.centre[1] + direction[1] * half_span,
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
                    + CONNECTOR_COVER_VERTICAL_BIAS_METRES
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


def _measured_stock_junction(junction) -> bool:
    if junction is None or _gravel_junction._is_gravel_junction(junction):
        return False
    extent = _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
    return (
        math.isclose(
            float(junction.half_length), extent, rel_tol=0.0, abs_tol=1.0e-6
        )
        and math.isclose(
            float(junction.half_width), extent, rel_tol=0.0, abs_tol=1.0e-6
        )
    )


def _quality_window(
    measure,
    pieces,
    start_distance,
    preferred_end,
    minimum_end,
    maximum_end,
    context,
):
    if _ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("stock road surface overlap policy is not installed")
    start_distance, preferred_end, minimum_end, maximum_end = (
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
    if not pieces:
        return start_distance, preferred_end, minimum_end, maximum_end

    start_junction = context.junctions.get(
        _p._road_node_key(measure.points[0])
    )
    end_junction = context.junctions.get(_p._road_node_key(measure.points[-1]))
    shortest = min(float(piece.length_metres) for piece in pieces)

    # Keep the fitter free to continue underneath a measured central mesh. The
    # post-fit pass above is the final authority: it measures whatever geometry
    # was actually emitted and adds a bounded underlay only where a gap remains.
    if _measured_stock_junction(start_junction):
        start_distance = 0.0
    if _measured_stock_junction(end_junction):
        preferred_end = max(start_distance, measure.total)
        minimum_end = max(start_distance, measure.total - 0.10)
        maximum_end = max(maximum_end, measure.total + shortest * 0.5)
    return start_distance, preferred_end, minimum_end, maximum_end


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
        raise RuntimeError("stock road surface overlap policy is not installed")
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
    return _apply_connector_covers(report, elevations, spec)


def install_stock_road_surface_overlap_policy() -> None:
    global _ORIGINAL_QUALITY_WINDOW, _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _ORIGINAL_FIT = _p.fit_road_objects
    _quality._quality_window = _quality_window
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
