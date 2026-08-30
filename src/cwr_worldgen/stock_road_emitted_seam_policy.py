# SPDX-License-Identifier: GPL-3.0-or-later
"""Close paved seams after every other stock-road wrapper has finished.

Several road policies legitimately replace or append objects after the older
visual-finish seam hook runs.  A seam that is perfect in the intermediate report
can also open in the actual WRP because a pitched rigid P3D has only
``length*cos(pitch)`` of horizontal connector span.  The final Road Inspector
therefore used to find asphalt gaps that no generator-side seam pass had ever
seen.

This policy is deliberately the outermost stock-road fit wrapper.  It inspects
the final ``WorldObject`` geometry through ``_p._model_axis`` (which already uses
measured 3D stock connectors), ignores gaps already covered by another
same-family paved surface, and adds low angle-matched borderless miter fills.
Straight mitres use the same 0.20 m physical connector radius as Road Inspector.
Residual curve seams may bridge up to 1.50 m when the mutually-nearest pair has
a modest tangent error; the fill follows the straight-side tangent when one is
available, avoiding a diagonal patch across the carriageway.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_model_geometry as _geometry
from . import stock_road_visual_finish_policy as _finish
from .procedural_infrastructure import (
    GENERATED_PAVED_FILL_RADIUS_METRES,
    GENERATED_PAVED_WEDGE_BASE_OVERLAP_METRES,
    paved_miter_angle_degrees,
    paved_miter_model_path,
    paved_wedge_angle_degrees,
    paved_wedge_local_points,
    paved_wedge_model_path,
)

MAXIMUM_EMITTED_STRAIGHT_GAP_METRES = 0.20
MAXIMUM_EMITTED_CURVE_GAP_METRES = 1.50
MINIMUM_EMITTED_TANGENT_ERROR_DEGREES = 0.75
MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES = 35.0
MAXIMUM_EMITTED_CURVE_TANGENT_ERROR_DEGREES = 12.0
EMITTED_SEAM_UNDERLAY_BIAS_METRES = -0.010
MINIMUM_VISIBLE_WEDGE_CLEARANCE_METRES = 0.030
MAXIMUM_OUTER_MITER_CENTER_GAP_METRES = 0.35
MAXIMUM_OUTER_MITER_EXTENSION_METRES = 10.0
TERRAIN_WEDGE_JUNCTION_EXCLUSION_METRES = 1.25
_ENDPOINT_BUCKET_METRES = MAXIMUM_EMITTED_CURVE_GAP_METRES
_PAIR_UNIQUENESS_MARGIN_METRES = 0.03
_SURFACE_MARGIN_METRES = 0.08
_VERTICAL_NEIGHBOURHOOD_METRES = 0.20
_SAMPLE_FRACTIONS = (0.20, 0.40, 0.60, 0.80)
_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})

_ORIGINAL_FIT = None
_INSTALLED = False


def _bucket(point: tuple[float, float]) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / _ENDPOINT_BUCKET_METRES),
        math.floor(float(point[1]) / _ENDPOINT_BUCKET_METRES),
    )


def _endpoint_half_width(endpoint) -> float:
    return float(_geometry.STOCK_HALF_WIDTHS_METRES[endpoint.family])


def _cross_section_edges(endpoint):
    heading = math.radians(float(endpoint.tangent_axis_degrees))
    normal = (math.cos(heading), -math.sin(heading))
    width = _endpoint_half_width(endpoint)
    return (
        (
            float(endpoint.point[0]) + normal[0] * width,
            float(endpoint.point[1]) + normal[1] * width,
        ),
        (
            float(endpoint.point[0]) - normal[0] * width,
            float(endpoint.point[1]) - normal[1] * width,
        ),
    )


def _cross(first, second) -> float:
    return float(first[0]) * float(second[1]) - float(first[1]) * float(second[0])


def _matched_edge_pairs(first, second):
    first_edges = _cross_section_edges(first)
    second_edges = _cross_section_edges(second)
    direct = (
        (first_edges[0], second_edges[0]),
        (first_edges[1], second_edges[1]),
    )
    crossed = (
        (first_edges[0], second_edges[1]),
        (first_edges[1], second_edges[0]),
    )
    return (
        direct
        if sum(math.dist(a, b) for a, b in direct)
        <= sum(math.dist(a, b) for a, b in crossed)
        else crossed
    )


def _forward_ray_intersection(first_point, first_heading, second_point, second_heading):
    first_angle = math.radians(float(first_heading))
    second_angle = math.radians(float(second_heading))
    first_direction = (math.sin(first_angle), math.cos(first_angle))
    second_direction = (math.sin(second_angle), math.cos(second_angle))
    denominator = _cross(first_direction, second_direction)
    if abs(denominator) <= 1.0e-8:
        return None
    delta = (
        float(second_point[0]) - float(first_point[0]),
        float(second_point[1]) - float(first_point[1]),
    )
    first_distance = _cross(delta, second_direction) / denominator
    second_distance = _cross(delta, first_direction) / denominator
    if first_distance <= 1.0e-5 or second_distance <= 1.0e-5:
        return None
    if (
        first_distance > MAXIMUM_OUTER_MITER_EXTENSION_METRES
        or second_distance > MAXIMUM_OUTER_MITER_EXTENSION_METRES
    ):
        return None
    return (
        float(first_point[0]) + first_direction[0] * first_distance,
        float(first_point[1]) + first_direction[1] * first_distance,
    )


def _outer_miter_geometry(first, second):
    """Return area, apex, and centroid for the larger forward outside miter."""

    if math.dist(first.point, second.point) > MAXIMUM_OUTER_MITER_CENTER_GAP_METRES:
        return None
    candidates = []
    for first_edge, second_edge in _matched_edge_pairs(first, second):
        apex = _forward_ray_intersection(
            first_edge,
            first.outward_heading_degrees,
            second_edge,
            second.outward_heading_degrees,
        )
        if apex is None:
            continue
        area = abs(_cross(
            (
                float(second_edge[0]) - float(first_edge[0]),
                float(second_edge[1]) - float(first_edge[1]),
            ),
            (
                float(apex[0]) - float(first_edge[0]),
                float(apex[1]) - float(first_edge[1]),
            ),
        )) * 0.5
        centroid = (
            (float(first_edge[0]) + float(second_edge[0]) + float(apex[0])) / 3.0,
            (float(first_edge[1]) + float(second_edge[1]) + float(apex[1])) / 3.0,
        )
        candidates.append((area, apex, centroid))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def _outer_miter_apex(first, second) -> tuple[float, float] | None:
    geometry = _outer_miter_geometry(first, second)
    return geometry[1] if geometry is not None else None


def _segment_samples(start, end):
    return tuple(
        (
            float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
            float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
        )
        for fraction in _SAMPLE_FRACTIONS
    )


def _gap_samples(first, second):
    samples = []
    for start, end in _matched_edge_pairs(first, second):
        samples.extend(_segment_samples(start, end))
    samples.extend(_segment_samples(first.point, second.point))
    return tuple(samples)


def _straight_contains(obj, point: tuple[float, float]) -> bool:
    match = _geometry.stock_straight_match(str(obj.model_path))
    if match is None:
        return False
    family = match.group("family").casefold()
    if family not in _PAVED_FAMILIES:
        return False
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(match.group("length"))])
    start, end = _p._model_axis(obj, length)
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    horizontal = math.hypot(dx, dz)
    if horizontal <= 1.0e-9:
        return False
    ux, uz = dx / horizontal, dz / horizontal
    px = float(point[0]) - float(start[0])
    pz = float(point[1]) - float(start[1])
    along = px * ux + pz * uz
    lateral = abs(px * -uz + pz * ux)
    return (
        -_SURFACE_MARGIN_METRES <= along <= horizontal + _SURFACE_MARGIN_METRES
        and lateral
        <= float(_geometry.STOCK_HALF_WIDTHS_METRES[family]) + _SURFACE_MARGIN_METRES
    )


def _covered_by_existing_surface(report, first, second) -> bool:
    samples = _gap_samples(first, second)
    if not samples:
        return False
    object_ids = {int(first.object_id), int(second.object_id)}
    road_by_id = {int(obj.object_id): obj for obj in report.objects}
    involved = [road_by_id.get(object_id) for object_id in object_ids]
    involved = [obj for obj in involved if obj is not None]
    if len(involved) != 2:
        return False
    minimum_y = min(float(obj.y) for obj in involved) - _VERTICAL_NEIGHBOURHOOD_METRES
    maximum_y = max(float(obj.y) for obj in involved) + _VERTICAL_NEIGHBOURHOOD_METRES
    candidates = tuple(
        obj
        for obj in report.objects
        if (
            int(obj.object_id) not in object_ids
            and minimum_y <= float(obj.y) <= maximum_y
            and (
                (match := _geometry.stock_straight_match(str(obj.model_path)))
                is not None
            )
            and match.group("family").casefold() == first.family
            and first.family in _PAVED_FAMILIES
        )
    )
    if not candidates:
        return False
    return all(any(_straight_contains(obj, sample) for obj in candidates) for sample in samples)


def _nearest_endpoint_pairs(endpoints):
    buckets: dict[tuple[str, int, int], list[int]] = {}
    for index, endpoint in enumerate(endpoints):
        bx, bz = _bucket(endpoint.point)
        buckets.setdefault((endpoint.family, bx, bz), []).append(index)

    nearest = []
    for index, endpoint in enumerate(endpoints):
        bx, bz = _bucket(endpoint.point)
        best = None
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for candidate_index in buckets.get(
                    (endpoint.family, bx + dx, bz + dz), ()
                ):
                    if candidate_index == index:
                        continue
                    candidate = endpoints[candidate_index]
                    if candidate.object_id == endpoint.object_id:
                        continue
                    distance = math.dist(endpoint.point, candidate.point)
                    if distance > MAXIMUM_EMITTED_CURVE_GAP_METRES + 1.0e-9:
                        continue
                    score = (
                        distance,
                        int(candidate.object_id),
                        int(candidate.endpoint_index),
                        candidate_index,
                    )
                    if best is None or score < best:
                        best = score
        nearest.append(best)

    pairs = []
    used = set()
    for index, best in enumerate(nearest):
        if best is None:
            continue
        other_index = int(best[-1])
        reverse = nearest[other_index]
        if reverse is None or int(reverse[-1]) != index:
            continue
        pair_key = tuple(sorted((index, other_index)))
        if pair_key in used:
            continue
        used.add(pair_key)
        first, second = endpoints[pair_key[0]], endpoints[pair_key[1]]
        pairs.append((float(best[0]), first, second, pair_key))
    return tuple(pairs)


def _pair_is_unambiguous(endpoints, first, second, distance: float) -> bool:
    limit = float(distance) + _PAIR_UNIQUENESS_MARGIN_METRES
    for candidate in endpoints:
        key = (int(candidate.object_id), int(candidate.endpoint_index))
        if key in {
            (int(first.object_id), int(first.endpoint_index)),
            (int(second.object_id), int(second.endpoint_index)),
        }:
            continue
        if candidate.family != first.family:
            continue
        if candidate.object_id in {first.object_id, second.object_id}:
            continue
        if (
            math.dist(candidate.point, first.point) <= limit
            or math.dist(candidate.point, second.point) <= limit
        ):
            return False
    return True


def _plan_heading(first, second) -> float:
    return _finish._average_axis_heading(
        first.tangent_axis_degrees,
        second.tangent_axis_degrees,
    )


def _emitted_seam_cover_plans(report):
    endpoints = tuple(
        endpoint
        for endpoint in _finish._seam_endpoints(report)
        if endpoint.family in _PAVED_FAMILIES
    )
    if not endpoints:
        return ()

    plans = []
    for distance, first, second, _pair_key in _nearest_endpoint_pairs(endpoints):
        if first.family != second.family:
            continue
        if not _pair_is_unambiguous(endpoints, first, second, distance):
            continue
        tangent_error = _finish._axis_heading_difference(
            first.tangent_axis_degrees,
            second.tangent_axis_degrees,
        )
        if tangent_error < MINIMUM_EMITTED_TANGENT_ERROR_DEGREES:
            continue

        curve_seam = bool(first.is_curve or second.is_curve)
        if curve_seam:
            if (
                distance > MAXIMUM_EMITTED_CURVE_GAP_METRES + 1.0e-9
                or tangent_error > MAXIMUM_EMITTED_CURVE_TANGENT_ERROR_DEGREES
            ):
                continue
        else:
            if (
                distance > MAXIMUM_EMITTED_STRAIGHT_GAP_METRES + 1.0e-9
                or tangent_error > MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES
            ):
                continue

        if _covered_by_existing_surface(report, first, second):
            continue
        plans.append(
            _finish._SeamCoverPlan(
                model_path=rf"o\road\{first.family}6.p3d",
                centre=(
                    (float(first.point[0]) + float(second.point[0])) * 0.5,
                    (float(first.point[1]) + float(second.point[1])) * 0.5,
                ),
                tangent_axis_degrees=_plan_heading(first, second),
                turn_degrees=tangent_error,
                outer_miter_apex=_outer_miter_apex(first, second),
            )
        )
    return tuple(plans)


def _generated_miter_contains(obj, point: tuple[float, float]) -> bool:
    turn = paved_miter_angle_degrees(str(obj.model_path))
    if turn is None:
        return False
    dx = float(point[0]) - float(obj.x)
    dz = float(point[1]) - float(obj.z)
    heading = math.radians(float(obj.heading_degrees))
    local_x = dx * math.cos(heading) - dz * math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(obj.pitch_degrees)))
    if abs(cosine_pitch) <= 1.0e-9:
        return False
    local_z = (dx * math.sin(heading) + dz * math.cos(heading)) / cosine_pitch
    radius = GENERATED_PAVED_FILL_RADIUS_METRES + 0.003
    margin = _SURFACE_MARGIN_METRES
    if math.hypot(local_x, local_z) <= radius + margin:
        return True
    half_angle = math.radians(float(turn) * 0.5)
    base_x = radius * math.cos(half_angle)
    apex_x = radius / math.cos(half_angle)
    absolute_x = abs(local_x)
    if absolute_x < base_x - margin or absolute_x > apex_x + margin:
        return False
    fraction = (apex_x - absolute_x) / max(1.0e-9, apex_x - base_x)
    return abs(local_z) <= radius * math.sin(half_angle) * fraction + margin


def _surface_contains(obj, point: tuple[float, float]) -> bool:
    if _straight_contains(obj, point):
        return True
    filename = str(obj.model_path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    if filename == "paved_fill.p3d":
        return math.dist((float(obj.x), float(obj.z)), point) <= (
            GENERATED_PAVED_FILL_RADIUS_METRES + 0.001
        )
    return _generated_miter_contains(obj, point)


def _surface_is_sil(obj) -> bool:
    match = _geometry.stock_straight_match(str(obj.model_path))
    if match is not None:
        return match.group("family").casefold() == "sil"
    filename = str(obj.model_path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    return (
        filename == "paved_fill.p3d"
        or paved_miter_angle_degrees(filename) is not None
    )


def _terrain_wedge_already_visible(
    report,
    first,
    second,
    samples,
    elevations,
    spec,
) -> bool:
    if elevations is None or spec is None:
        return False
    involved_ids = {int(first.object_id), int(second.object_id)}
    road_by_id = {int(obj.object_id): obj for obj in report.objects}
    involved = [road_by_id.get(object_id) for object_id in involved_ids]
    involved = [obj for obj in involved if obj is not None]
    if len(involved) != 2:
        return False
    minimum_y = min(float(obj.y) for obj in involved) - 0.15
    maximum_y = max(float(obj.y) for obj in involved) + 0.50
    candidates = tuple(
        obj
        for obj in report.objects
        if (
            int(obj.object_id) not in involved_ids
            and minimum_y <= float(obj.y) <= maximum_y
            and _surface_is_sil(obj)
            and _surface_contains(obj, samples[0])
        )
    )
    # Re-evaluate containment per sample; the first-sample clause above merely
    # keeps the candidate tuple small without accepting a centre-only overlap.
    return all(
        any(
            _surface_contains(obj, sample)
            and (
                _surface_height_at(obj, sample)
                - _p._sample_elevation(
                    elevations, spec.cells, spec.cell_size, sample[0], sample[1]
                )
                >= MINIMUM_VISIBLE_WEDGE_CLEARANCE_METRES
            )
            for obj in candidates
        )
        for sample in samples
    )


def _terrain_wedge_cover_plans(report, elevations=None, spec=None):
    """Plan the actual outside triangles even when an older helper covers X/Z.

    Earlier seam passes can add a full straight underlay. Its footprint may
    cover every inspector sample while its rigid plane is still buried by a
    cross-slope at the outside miter. The final terrain-clear triangle therefore
    audits the physical road endpoints independently of existing helper pieces.
    """

    endpoints = tuple(
        endpoint
        for endpoint in _finish._seam_endpoints(report)
        if endpoint.family == "sil"
    )
    plans = []
    for distance, first, second, _pair_key in _nearest_endpoint_pairs(endpoints):
        if not _pair_is_unambiguous(endpoints, first, second, distance):
            continue
        turn = _finish._axis_heading_difference(
            first.tangent_axis_degrees,
            second.tangent_axis_degrees,
        )
        if turn < MINIMUM_EMITTED_TANGENT_ERROR_DEGREES:
            continue
        if turn > MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES:
            continue
        seam_centre = (
            (float(first.point[0]) + float(second.point[0])) * 0.5,
            (float(first.point[1]) + float(second.point[1])) * 0.5,
        )
        involved_ids = {int(first.object_id), int(second.object_id)}
        if any(
            int(candidate.object_id) not in involved_ids
            and math.dist(candidate.point, seam_centre)
            <= TERRAIN_WEDGE_JUNCTION_EXCLUSION_METRES
            for candidate in endpoints
        ):
            continue
        geometry = _outer_miter_geometry(first, second)
        if geometry is None:
            continue
        _area, apex, centroid = geometry
        if _terrain_wedge_already_visible(
            report,
            first,
            second,
            (apex, centroid),
            elevations,
            spec,
        ):
            continue
        plans.append(_finish._SeamCoverPlan(
            model_path=r"o\road\sil6.p3d",
            centre=seam_centre,
            tangent_axis_degrees=_plan_heading(first, second),
            turn_degrees=turn,
            outer_miter_apex=apex,
        ))
    return tuple(plans)


def _surface_height_at(obj, point: tuple[float, float]) -> float:
    heading = math.radians(float(obj.heading_degrees))
    pitch = math.radians(float(obj.pitch_degrees))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) <= 1.0e-9:
        return float(obj.y)
    dx = float(point[0]) - float(obj.x)
    dz = float(point[1]) - float(obj.z)
    local_z = (dx * math.sin(heading) + dz * math.cos(heading)) / cosine_pitch
    return float(obj.y) + local_z * math.sin(pitch)


def _triangle_samples(points):
    first, second, third = points
    return (
        first,
        second,
        third,
        tuple((first[index] + second[index]) * 0.5 for index in range(3)),
        tuple((second[index] + third[index]) * 0.5 for index in range(3)),
        tuple((third[index] + first[index]) * 0.5 for index in range(3)),
        tuple((first[index] + second[index] + third[index]) / 3.0 for index in range(3)),
    )


def _terrain_clear_wedge_overlay(
    plan,
    low_miter,
    object_id,
    elevations,
    spec,
    *,
    force: bool = False,
):
    apex = getattr(plan, "outer_miter_apex", None)
    if apex is None or float(getattr(plan, "turn_degrees", 0.0)) < 0.75:
        return None
    terrain_at_apex = _p._sample_elevation(
        elevations,
        spec.cells,
        spec.cell_size,
        float(apex[0]),
        float(apex[1]),
    )
    if not force and (
        _surface_height_at(low_miter, apex) - terrain_at_apex
        >= MINIMUM_VISIBLE_WEDGE_CLEARANCE_METRES
    ):
        return None

    world_name = str(getattr(spec, "name", "cwr_worldgen"))
    model_path = paved_wedge_model_path(world_name, float(plan.turn_degrees))
    quantized_turn = paved_wedge_angle_degrees(model_path)
    if quantized_turn is None:
        return None
    local_points = paved_wedge_local_points(quantized_turn)

    dx = float(apex[0]) - float(plan.centre[0])
    dz = float(apex[1]) - float(plan.centre[1])
    radial_length = math.hypot(dx, dz)
    if radial_length <= 1.0e-8:
        return None
    ux, uz = dx / radial_length, dz / radial_length
    half_angle = math.radians(float(quantized_turn) * 0.5)
    base_distance = (
        GENERATED_PAVED_FILL_RADIUS_METRES * math.cos(half_angle)
        - GENERATED_PAVED_WEDGE_BASE_OVERLAP_METRES
    )
    origin = (
        float(plan.centre[0]) + ux * base_distance,
        float(plan.centre[1]) + uz * base_distance,
    )
    heading = math.degrees(math.atan2(ux, uz)) % 360.0
    # Keep the tiny triangle horizontal. Pitching an origin-offset wedge would
    # contract its footprint in X/Z and could pull its tip back from the exact
    # road-edge miter. Raising this borderless, wedge-only polygon to the local
    # terrain maximum cannot paint across the road like a full helper disk.
    pitch = 0.0
    heading_radians = math.radians(heading)
    pitch_radians = math.radians(pitch)
    cosine_heading = math.cos(heading_radians)
    sine_heading = math.sin(heading_radians)
    cosine_pitch = math.cos(pitch_radians)
    sine_pitch = math.sin(pitch_radians)

    required_origin_y = -math.inf
    for local_x, _local_y, local_z in _triangle_samples(local_points):
        world_x = origin[0] + local_x * cosine_heading + local_z * sine_heading * cosine_pitch
        world_z = origin[1] - local_x * sine_heading + local_z * cosine_heading * cosine_pitch
        terrain = _p._sample_elevation(
            elevations, spec.cells, spec.cell_size, world_x, world_z
        )
        required_origin_y = max(
            required_origin_y,
            terrain
            + MINIMUM_VISIBLE_WEDGE_CLEARANCE_METRES
            - local_z * sine_pitch,
        )

    return replace(
        low_miter,
        object_id=int(object_id),
        model_path=model_path,
        x=origin[0],
        y=required_origin_y,
        z=origin[1],
        heading_degrees=heading,
        pitch_degrees=pitch,
    )


def _apply_emitted_seam_covers(report, elevations, spec):
    plans = _emitted_seam_cover_plans(report)
    wedge_plans = _terrain_wedge_cover_plans(report, elevations, spec)
    if not plans and not wedge_plans:
        return report

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    half = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6]) * 0.5
    planned_wedges = set()
    for plan in plans:
        angle = math.radians(float(plan.tangent_axis_degrees))
        direction = (math.sin(angle), math.cos(angle))
        start = (
            float(plan.centre[0]) - direction[0] * half,
            float(plan.centre[1]) - direction[1] * half,
        )
        end = (
            float(plan.centre[0]) + direction[0] * half,
            float(plan.centre[1]) + direction[1] * half,
        )
        model_path = plan.model_path
        if str(model_path).replace("/", "\\").casefold() == r"o\road\sil6.p3d":
            model_path = paved_miter_model_path(
                str(getattr(spec, "name", "cwr_worldgen")),
                float(getattr(plan, "turn_degrees", 0.0)),
            )
        low_miter = _p._road_object_on_slope(
            next_id,
            model_path,
            start,
            end,
            elevations,
            spec,
            vertical_offset=(
                _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                + EMITTED_SEAM_UNDERLAY_BIAS_METRES
            ),
        )
        objects.append(low_miter)
        next_id += 1
        if "\\paved_miter_q" in str(model_path).replace("/", "\\").casefold():
            overlay = _terrain_clear_wedge_overlay(
                plan, low_miter, next_id, elevations, spec
            )
            if overlay is not None:
                objects.append(overlay)
                next_id += 1
            if plan.outer_miter_apex is not None:
                planned_wedges.add(tuple(round(float(value), 3) for value in plan.outer_miter_apex))

    for plan in wedge_plans:
        apex_key = tuple(round(float(value), 3) for value in plan.outer_miter_apex)
        if apex_key in planned_wedges:
            continue
        angle = math.radians(float(plan.tangent_axis_degrees))
        direction = (math.sin(angle), math.cos(angle))
        start = (
            float(plan.centre[0]) - direction[0] * half,
            float(plan.centre[1]) - direction[1] * half,
        )
        end = (
            float(plan.centre[0]) + direction[0] * half,
            float(plan.centre[1]) + direction[1] * half,
        )
        reference = _p._road_object_on_slope(
            next_id,
            paved_miter_model_path(
                str(getattr(spec, "name", "cwr_worldgen")),
                float(plan.turn_degrees),
            ),
            start,
            end,
            elevations,
            spec,
            vertical_offset=(
                _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                + EMITTED_SEAM_UNDERLAY_BIAS_METRES
            ),
        )
        overlay = _terrain_clear_wedge_overlay(
            plan,
            reference,
            next_id,
            elevations,
            spec,
            force=True,
        )
        if overlay is not None:
            objects.append(overlay)
            next_id += 1

    required = len(objects)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after final emitted seam coverage: "
            f"requires {required:,} objects, limit is {int(spec.max_road_objects):,}"
        )

    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0))
            + (len(objects) - len(report.objects))
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
        raise RuntimeError("final emitted seam policy is not installed")
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
    return _apply_emitted_seam_covers(report, elevations, spec)


def install_stock_road_emitted_seam_policy() -> None:
    """Install the final post-fit paved seam audit after intersection coverage."""

    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
