# SPDX-License-Identifier: GPL-3.0-or-later
"""Reusable generated-gravel road family and fitting policy.

Generated gravel straights and curves already behave like a conventional CWA
road family. This module owns the corresponding junction geometry, short-gap
repair, and reusable T/Y/crossroad models.

Three installation moments are intentionally retained. The early junction stage
teaches road-quality trimming about the measured gravel junction footprint. The
middle gap stage closes short visual breaks between separately normalized gravel
ways. The later family stage replaces fitted caps with reusable generated family
models. Keeping all three phases here removes artificial module boundaries while
preserving the historical road-pipeline order and fitter capture points.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
import re
from typing import Iterable

from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon
from shapely.ops import triangulate as shapely_triangulate, unary_union

from . import generator as _generator
from . import playability as _p
from . import procedural_infrastructure as _pi
from . import road_quality_policy as _rq

GRAVEL_JUNCTION_ARM_EXTENT_METRES = 4.0
GRAVEL_JUNCTION_CORE_RADIUS_METRES = 1.65
GRAVEL_T_JUNCTION_ANGLES = (30, 45, 60, 75, 90)
GRAVEL_X_JUNCTION_ANGLES = (30, 45, 60, 75, 90)
GRAVEL_JUNCTION_VARIANTS = (
    "t30l", "t30r", "t45l", "t45r", "t60l", "t60r",
    "t75l", "t75r", "t90", "y120",
    "x30", "x45", "x60", "x75", "x90",
)

_FAMILY_PATTERN = re.compile(
    r"^gravel_j(?P<degree>[34])_(?P<variant>"
    r"t(?:30|45|60|75)[lr]|t90|y120|x(?:30|45|60|75|90))\.p3d$",
    re.IGNORECASE,
)
_GRAVEL_JUNCTION_OVERLAP = min(0.70, _pi.GENERATED_GRAVEL_VISUAL_OVERLAP_METRES)
_GRAVEL_GAP_MAX_METRES = 8.0
_GRAVEL_GAP_ALIGNMENT_COSINE = math.cos(math.radians(20.0))
_GRAVEL_GAP_EPSILON = 0.20

_ORIGINAL_FIT = None
_GAP_ORIGINAL_FIT = None
_ORIGINAL_ROAD_LODS = _pi._road_lods
_ORIGINAL_REGISTER_MODEL_USAGE = _pi.ProceduralInfrastructureLibrary.register_model_usage
_ORIGINAL_IS_GRAVEL_JUNCTION = _pi.is_generated_gravel_junction_model
_ORIGINAL_JUNCTION_GEOMETRY = None
_ORIGINAL_EXIT_DISTANCE = None
_JUNCTION_INSTALLED = False
_GAP_INSTALLED = False
_INSTALLED = False


def _unit(direction: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(direction[0], direction[1])
    if length <= 1.0e-9:
        return (0.0, 1.0)
    return (direction[0] / length, direction[1] / length)


def _heading(direction: tuple[float, float]) -> float:
    value = _unit(direction)
    return math.degrees(math.atan2(value[0], value[1])) % 360.0


def _signed_angle(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (_heading(right) - _heading(left) + 180.0) % 360.0 - 180.0


def _bucket(value: float, candidates: tuple[int, ...]) -> int:
    return min(candidates, key=lambda candidate: (abs(float(candidate) - value), candidate))


def _angular_distance_degrees(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _direction_from_heading(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return (math.sin(angle), math.cos(angle))


def gravel_junction_variant_for_directions(
    directions: Iterable[tuple[float, float]],
) -> tuple[str, tuple[float, float]]:
    """Return the best reusable junction shape and its local +Z world axis."""

    cleaned = tuple(_unit(direction) for direction in directions)
    degree = len(cleaned)
    if degree not in {3, 4}:
        raise ValueError("gravel junction requires three or four directions")

    actual_headings = tuple(_heading(direction) for direction in cleaned)
    variants = tuple(
        variant for variant in GRAVEL_JUNCTION_VARIANTS
        if (4 if variant.startswith("x") else 3) == degree
    )
    variant_rank = {variant: index for index, variant in enumerate(variants)}

    best: tuple[float, float, int, float, str] | None = None
    for variant in variants:
        template = gravel_junction_template_headings(variant)
        rotations = {
            (actual - local) % 360.0
            for actual in actual_headings
            for local in template
        }
        for rotation in rotations:
            rotated = tuple((local + rotation) % 360.0 for local in template)
            for assignment in itertools.permutations(actual_headings):
                errors = tuple(
                    _angular_distance_degrees(model_heading, actual_heading)
                    for model_heading, actual_heading in zip(rotated, assignment)
                )
                maximum_error = max(errors)
                squared_error = sum(error * error for error in errors)
                candidate = (
                    maximum_error,
                    squared_error,
                    variant_rank[variant],
                    rotation,
                    variant,
                )
                if best is None or candidate < best:
                    best = candidate

    assert best is not None
    _maximum_error, _squared_error, _rank, rotation, variant = best
    return variant, _direction_from_heading(rotation)


def gravel_junction_template_headings(variant: str) -> tuple[float, ...]:
    value = variant.casefold()
    if value == "y120":
        return (0.0, 120.0, 240.0)
    match = re.fullmatch(r"t(30|45|60|75)([lr])", value)
    if match:
        angle = float(match.group(1))
        if match.group(2) == "l":
            angle = -angle
        return (0.0, 180.0, angle)
    if value == "t90":
        return (0.0, 180.0, 90.0)
    match = re.fullmatch(r"x(30|45|60|75|90)", value)
    if match:
        angle = float(match.group(1))
        return (0.0, 180.0, angle, angle + 180.0)
    raise ValueError(f"unknown gravel junction variant: {variant}")


def gravel_junction_model_path(world_name: str, degree: int, variant: str) -> str:
    normalized = variant.casefold()
    expected = 4 if normalized.startswith("x") else 3
    if degree != expected:
        raise ValueError("gravel junction degree does not match variant")
    gravel_junction_template_headings(normalized)
    if (degree, normalized) in {(3, "t90"), (4, "x90")}:
        return rf"{world_name}\i\gravel_j{degree}.p3d"
    return rf"{world_name}\i\gravel_j{degree}_{normalized}.p3d"


def is_generated_gravel_family_junction(model_path: str) -> bool:
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
    return _FAMILY_PATTERN.fullmatch(filename) is not None or _ORIGINAL_IS_GRAVEL_JUNCTION(model_path)


def gravel_junction_ray_exit_distance(
    variant: str,
    axis: tuple[float, float],
    direction: tuple[float, float],
) -> float:
    ray = _unit(direction)
    axis = _unit(axis)
    right = (axis[1], -axis[0])
    best = GRAVEL_JUNCTION_CORE_RADIUS_METRES
    for heading in gravel_junction_template_headings(variant):
        angle = math.radians(heading)
        local_x, local_z = math.sin(angle), math.cos(angle)
        arm = (
            right[0] * local_x + axis[0] * local_z,
            right[1] * local_x + axis[1] * local_z,
        )
        along = ray[0] * arm[0] + ray[1] * arm[1]
        if along <= 1.0e-8:
            continue
        lateral = abs(arm[0] * ray[1] - arm[1] * ray[0])
        limits = [GRAVEL_JUNCTION_ARM_EXTENT_METRES / along]
        if lateral > 1.0e-8:
            limits.append(_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES / lateral)
        best = max(best, min(limits))
    return best


def _is_gravel_junction(junction) -> bool:
    return math.isclose(
        float(junction.half_width),
        float(_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    )


def _junction_geometry(dataset, projection, spec):
    result = dict(_ORIGINAL_JUNCTION_GEOMETRY(dataset, projection, spec))
    if not result:
        return result

    models_by_key: dict[tuple[int, int], list[str]] = {}
    projected_roads = _rq._p.projected_road_polylines(dataset, projection)
    for feature, projected in zip(dataset.roads, projected_roads):
        if not _rq._p.road_is_supported(
            feature.tags,
            include_minor=spec.include_minor_roads,
        ):
            continue
        points = tuple(_rq._p._clean_road_points(projected))
        if len(points) < 2:
            continue
        model = _rq._p.road_model_for_tags(spec, feature.tags)
        for start, end in zip(points, points[1:]):
            if math.dist(start, end) <= 0.05:
                continue
            models_by_key.setdefault(_rq._p._road_node_key(start), []).append(model)
            models_by_key.setdefault(_rq._p._road_node_key(end), []).append(model)

    for key, junction in tuple(result.items()):
        models = models_by_key.get(key, ())
        if not models or not all(
            _rq._p.is_generated_gravel_road_model(model) for model in models
        ):
            continue
        if len(junction.directions) not in {3, 4}:
            continue
        _variant, axis = gravel_junction_variant_for_directions(junction.directions)
        result[key] = replace(
            junction,
            axis=axis,
            half_length=GRAVEL_JUNCTION_ARM_EXTENT_METRES,
            half_width=_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES,
        )
    return result


def _exit_distance(junction, direction: tuple[float, float]) -> float:
    if not _is_gravel_junction(junction):
        return _ORIGINAL_EXIT_DISTANCE(junction, direction)
    variant, axis = gravel_junction_variant_for_directions(junction.directions)
    return gravel_junction_ray_exit_distance(variant, axis, direction)


def _overlap_for(junction) -> float:
    if junction is not None and _is_gravel_junction(junction):
        return _GRAVEL_JUNCTION_OVERLAP
    return float(_rq._JUNCTION_OVERLAP)


def _quality_window(
    measure,
    pieces,
    start_distance,
    preferred_end,
    minimum_end,
    maximum_end,
    context,
):
    if not pieces:
        return start_distance, preferred_end, minimum_end, maximum_end
    shortest = min(piece.length_metres for piece in pieces)
    start_junction = context.junctions.get(_rq._p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_rq._p._road_node_key(measure.points[-1]))
    desired_start = start_distance
    desired_end_trim = max(0.0, measure.total - preferred_end)
    desired_end_cover = max(0.0, measure.total - minimum_end)
    adjusted_maximum = maximum_end

    if start_junction is not None:
        desired_start = max(
            _rq._JUNCTION_MIN_TRIM,
            _exit_distance(start_junction, _rq._end_direction(measure, start=True))
            - _overlap_for(start_junction),
        )
    if end_junction is not None:
        exit_distance = _exit_distance(
            end_junction,
            _rq._end_direction(measure, start=False),
        )
        overlap = _overlap_for(end_junction)
        desired_end_trim = max(_rq._JUNCTION_MIN_TRIM, exit_distance - overlap)
        desired_end_cover = exit_distance + _rq._JUNCTION_MARGIN
        adjusted_maximum = min(maximum_end, measure.total + overlap)

    if (
        start_junction is not None or end_junction is not None
    ) and measure.total >= desired_start + desired_end_trim + shortest * 0.60:
        start_distance = desired_start
        if end_junction is not None:
            preferred_end = max(start_distance, measure.total - desired_end_trim)
            minimum_end = max(start_distance, measure.total - desired_end_cover)
            maximum_end = max(preferred_end, adjusted_maximum)
    return start_distance, preferred_end, minimum_end, maximum_end


def _arm_polygon(heading: float, half_width: float) -> ShapelyPolygon:
    angle = math.radians(heading)
    direction = (math.sin(angle), math.cos(angle))
    perpendicular = (math.cos(angle), -math.sin(angle))
    inner = -0.20
    extent = GRAVEL_JUNCTION_ARM_EXTENT_METRES
    return ShapelyPolygon(
        tuple(
            (
                direction[0] * along + perpendicular[0] * across,
                direction[1] * along + perpendicular[1] * across,
            )
            for along, across in (
                (inner, half_width),
                (extent, half_width),
                (extent, -half_width),
                (inner, -half_width),
            )
        )
    )


def _junction_polygon(variant: str, half_width: float):
    arms = tuple(
        _arm_polygon(heading, half_width)
        for heading in gravel_junction_template_headings(variant)
    )
    core = ShapelyPoint(0.0, 0.0).buffer(
        GRAVEL_JUNCTION_CORE_RADIUS_METRES,
        quad_segs=8,
    )
    polygon = unary_union((core, *arms))
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    return polygon


def _triangulated_lod(polygon, *, y: float, texture: str, resolution: float):
    point_indices: dict[tuple[float, float], int] = {}
    points: list[tuple[float, float, float]] = []
    faces: list[_pi._Face] = []
    triangles = (
        triangle
        for triangle in shapely_triangulate(polygon)
        if polygon.covers(triangle.representative_point())
    )
    for triangle in triangles:
        vertices = []
        for x, z in tuple(triangle.exterior.coords)[:-1]:
            key = (round(float(x), 6), round(float(z), 6))
            index = point_indices.get(key)
            if index is None:
                index = len(points)
                point_indices[key] = index
                points.append((float(x), y, float(z)))
            vertices.append((index, 0, float(x) / 3.0, float(z) / 3.0))
        if len(vertices) == 3:
            faces.append(_pi._Face(texture, tuple(vertices)))
    properties = ()
    if resolution == _pi._VISUAL_LOD:
        properties = (("autocenter", "0"), ("class", "road"), ("map", "road"))
    return _pi._Lod(
        tuple(points),
        ((0.0, 1.0, 0.0),),
        tuple(faces),
        resolution,
        properties=properties,
    )


def _family_junction_lods(key, texture: str):
    match = re.fullmatch(r"gravel_j([34])_(.+)", key.subtype.casefold())
    if match is None:
        raise ValueError(f"invalid fixed gravel junction subtype: {key.subtype}")
    degree = int(match.group(1))
    variant = match.group(2)
    if (4 if variant.startswith("x") else 3) != degree:
        raise ValueError("gravel junction subtype has mismatched degree")
    polygon = _junction_polygon(variant, max(1.8, key.width_m * 0.5))
    visual = _triangulated_lod(
        polygon,
        y=_pi.GENERATED_GRAVEL_VISUAL_TOP_METRES,
        texture=texture,
        resolution=_pi._VISUAL_LOD,
    )
    boundary = tuple(
        (float(x), 0.0, float(z))
        for x, z in tuple(polygon.exterior.coords)[:-1]
    )
    map_geometry = _pi._Lod(
        boundary,
        (),
        (),
        _pi._GEOMETRY_LOD,
        properties=(("map", "road"),),
    )
    roadway = _triangulated_lod(
        polygon,
        y=_pi.GENERATED_GRAVEL_ROADWAY_HEIGHT_METRES,
        texture="",
        resolution=_pi._ROADWAY_LOD,
    )
    land = _pi._Lod(boundary, (), (), _pi._LAND_CONTACT_LOD)
    return visual, map_geometry, roadway, land


def _road_lods(key, texture: str):
    subtype = key.subtype.casefold()
    if subtype == "gravel_j3":
        family_key = replace(
            key,
            subtype="gravel_j3_t90",
            length_dm=int(round(GRAVEL_JUNCTION_ARM_EXTENT_METRES * 20.0)),
        )
        return _family_junction_lods(family_key, texture)
    if subtype == "gravel_j4":
        family_key = replace(
            key,
            subtype="gravel_j4_x90",
            length_dm=int(round(GRAVEL_JUNCTION_ARM_EXTENT_METRES * 20.0)),
        )
        return _family_junction_lods(family_key, texture)
    if re.fullmatch(r"gravel_j[34]_.+", key.subtype, re.IGNORECASE):
        return _family_junction_lods(key, texture)
    return _ORIGINAL_ROAD_LODS(key, texture)


def _register_model_usage(self, model_path: str, count: int = 1) -> None:
    count = max(0, int(count))
    if count == 0:
        return
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
    legacy = re.fullmatch(r"gravel_j([34])\.p3d", filename, re.IGNORECASE)
    if legacy and self.is_generated_model(model_path):
        degree = int(legacy.group(1))
        self._usage[
            _pi.InfrastructureModelKey(
                "road",
                f"gravel_j{degree}",
                int(round(_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES * 20.0)),
                int(round(GRAVEL_JUNCTION_ARM_EXTENT_METRES * 20.0)),
            )
        ] += count
        return
    match = _FAMILY_PATTERN.fullmatch(filename)
    if match and self.is_generated_model(model_path):
        degree = int(match.group("degree"))
        variant = match.group("variant").casefold()
        if (4 if variant.startswith("x") else 3) != degree:
            raise ValueError("gravel junction model degree mismatch")
        self._usage[
            _pi.InfrastructureModelKey(
                "road",
                filename[:-4].casefold(),
                int(round(_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES * 20.0)),
                int(round(GRAVEL_JUNCTION_ARM_EXTENT_METRES * 20.0)),
            )
        ] += count
        return
    _ORIGINAL_REGISTER_MODEL_USAGE(self, model_path, count)


def _replace_gravel_caps(report, dataset, projection, elevations, spec):
    if report.junction_cap_objects <= 0:
        return report
    junctions = _rq._junction_geometry(dataset, projection, spec)
    keys = sorted(junctions)
    if not keys:
        return report
    objects = list(report.objects)
    changed = False
    for index, key in enumerate(keys[: report.junction_cap_objects]):
        junction = junctions[key]
        if not math.isclose(
            float(junction.half_width),
            float(_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            continue
        variant, axis = gravel_junction_variant_for_directions(junction.directions)
        degree = len(junction.directions)
        model_path = gravel_junction_model_path(spec.name, degree, variant)
        half = GRAVEL_JUNCTION_ARM_EXTENT_METRES
        start = (
            junction.point[0] - axis[0] * half,
            junction.point[1] - axis[1] * half,
        )
        end = (
            junction.point[0] + axis[0] * half,
            junction.point[1] + axis[1] * half,
        )
        old = objects[index]
        objects[index] = _p._road_object_on_slope(
            old.object_id,
            model_path,
            start,
            end,
            elevations,
            spec,
            vertical_offset=0.060,
        )
        changed = True
    return replace(report, objects=tuple(objects)) if changed else report


@dataclass(frozen=True, slots=True)
class _GravelEndpoint:
    point: tuple[float, float]
    inward: tuple[float, float]
    feature_key: str

    @property
    def outward(self) -> tuple[float, float]:
        return (-self.inward[0], -self.inward[1])


def _segment_unit(start, end) -> tuple[float, float]:
    dx, dz = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dz)
    return (0.0, 1.0) if length <= 1.0e-9 else (dx / length, dz / length)


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _gravel_endpoints(dataset, projection, spec) -> tuple[_GravelEndpoint, ...]:
    result: list[_GravelEndpoint] = []
    projected = _p.projected_road_polylines(dataset, projection)
    for feature, raw_points in zip(dataset.roads, projected):
        if not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        model = _p.road_model_for_tags(spec, feature.tags)
        if not _p.is_generated_gravel_road_model(model):
            continue
        points = tuple(_p._clean_road_points(raw_points))
        if len(points) < 2:
            continue
        if math.dist(points[0], points[1]) > 0.05:
            result.append(
                _GravelEndpoint(points[0], _segment_unit(points[0], points[1]), feature.osm_key)
            )
        if math.dist(points[-1], points[-2]) > 0.05:
            result.append(
                _GravelEndpoint(points[-1], _segment_unit(points[-1], points[-2]), feature.osm_key)
            )
    return tuple(result)


def _gravel_piece_length(spec, nominal: int) -> float:
    return float(spec.road_segment_length) * float(nominal) / 25.0


def _gravel_bridge_nominal(spec, span: float, *, endpoint_pair: bool) -> int | None:
    tips = 4 if endpoint_pair else 3
    allowance = tips * _pi.GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
    for nominal in (3, 6, 12):
        if _gravel_piece_length(spec, nominal) + allowance >= span + _GRAVEL_GAP_EPSILON:
            return nominal
    return None


def _gravel_hub_geometry(junction) -> bool:
    return math.isclose(
        float(junction.half_width),
        float(_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )


def _bridge_short_gravel_gaps(report, dataset, projection, elevations, spec, context):
    endpoints = _gravel_endpoints(dataset, projection, spec)
    if not endpoints:
        return report, 0

    bucket_size = _GRAVEL_GAP_MAX_METRES
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, endpoint in enumerate(endpoints):
        key = (
            math.floor(endpoint.point[0] / bucket_size),
            math.floor(endpoint.point[1] / bucket_size),
        )
        buckets.setdefault(key, []).append(index)

    candidates: list[tuple[float, int, int]] = []
    minimum_gap = (
        2.0 * _pi.GENERATED_GRAVEL_VISUAL_OVERLAP_METRES + _GRAVEL_GAP_EPSILON
    )
    for left_index, left in enumerate(endpoints):
        bx = math.floor(left.point[0] / bucket_size)
        bz = math.floor(left.point[1] / bucket_size)
        for nx in range(bx - 1, bx + 2):
            for nz in range(bz - 1, bz + 2):
                for right_index in buckets.get((nx, nz), ()):
                    if right_index <= left_index:
                        continue
                    right = endpoints[right_index]
                    if left.feature_key == right.feature_key:
                        continue
                    dx = right.point[0] - left.point[0]
                    dz = right.point[1] - left.point[1]
                    distance = math.hypot(dx, dz)
                    if not (minimum_gap < distance <= _GRAVEL_GAP_MAX_METRES):
                        continue
                    direction = (dx / distance, dz / distance)
                    if _dot(left.outward, direction) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                        continue
                    if _dot(right.outward, (-direction[0], -direction[1])) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                        continue
                    candidates.append((distance, left_index, right_index))

    fillers: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    used: set[int] = set()
    for distance, left_index, right_index in sorted(candidates):
        if left_index in used or right_index in used:
            continue
        nominal = _gravel_bridge_nominal(spec, distance, endpoint_pair=True)
        if nominal is None:
            continue
        left, right = endpoints[left_index], endpoints[right_index]
        dx = right.point[0] - left.point[0]
        dz = right.point[1] - left.point[1]
        length = math.hypot(dx, dz)
        ux, uz = dx / length, dz / length
        model_length = _gravel_piece_length(spec, nominal)
        centre = (
            (left.point[0] + right.point[0]) * 0.5,
            (left.point[1] + right.point[1]) * 0.5,
        )
        start = (
            centre[0] - ux * model_length * 0.5,
            centre[1] - uz * model_length * 0.5,
        )
        end = (
            centre[0] + ux * model_length * 0.5,
            centre[1] + uz * model_length * 0.5,
        )
        fillers.append((_pi.gravel_road_model_path(spec.name, nominal), start, end))
        used.update((left_index, right_index))

    gravel_junctions = tuple(
        junction for junction in context.junctions.values() if _gravel_hub_geometry(junction)
    )
    junction_bucket_size = _GRAVEL_GAP_MAX_METRES + 4.0
    junction_buckets: dict[tuple[int, int], list[object]] = {}
    for junction in gravel_junctions:
        key = (
            math.floor(junction.point[0] / junction_bucket_size),
            math.floor(junction.point[1] / junction_bucket_size),
        )
        junction_buckets.setdefault(key, []).append(junction)

    for endpoint_index, endpoint in enumerate(endpoints):
        if endpoint_index in used:
            continue
        bx = math.floor(endpoint.point[0] / junction_bucket_size)
        bz = math.floor(endpoint.point[1] / junction_bucket_size)
        nearby_junctions = []
        for nx in range(bx - 1, bx + 2):
            for nz in range(bz - 1, bz + 2):
                nearby_junctions.extend(junction_buckets.get((nx, nz), ()))

        best = None
        for junction in nearby_junctions:
            dx = junction.point[0] - endpoint.point[0]
            dz = junction.point[1] - endpoint.point[1]
            node_distance = math.hypot(dx, dz)
            if node_distance <= 1.0e-9:
                continue
            to_junction = (dx / node_distance, dz / node_distance)
            if _dot(endpoint.outward, to_junction) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                continue
            from_junction = (-to_junction[0], -to_junction[1])
            ax, az = junction.axis
            perpendicular = (-az, ax)
            if max(
                abs(_dot(from_junction, (ax, az))),
                abs(_dot(from_junction, perpendicular)),
            ) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                continue
            hub_exit = _rq._exit_distance(junction, from_junction)
            span = node_distance - hub_exit
            visible_gap = span - _pi.GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
            if not (_GRAVEL_GAP_EPSILON < visible_gap <= _GRAVEL_GAP_MAX_METRES):
                continue
            if best is None or span < best[0]:
                best = (span, junction, from_junction, hub_exit)

        if best is None:
            continue
        span, junction, from_junction, hub_exit = best
        nominal = _gravel_bridge_nominal(spec, span, endpoint_pair=False)
        if nominal is None:
            continue
        ux, uz = from_junction
        hub_edge = (
            junction.point[0] + ux * hub_exit,
            junction.point[1] + uz * hub_exit,
        )
        centre = (
            (hub_edge[0] + endpoint.point[0]) * 0.5,
            (hub_edge[1] + endpoint.point[1]) * 0.5,
        )
        model_length = _gravel_piece_length(spec, nominal)
        start = (
            centre[0] - ux * model_length * 0.5,
            centre[1] - uz * model_length * 0.5,
        )
        end = (
            centre[0] + ux * model_length * 0.5,
            centre[1] + uz * model_length * 0.5,
        )
        fillers.append((_pi.gravel_road_model_path(spec.name, nominal), start, end))
        used.add(endpoint_index)

    if not fillers:
        return report, 0

    next_id = max((obj.object_id for obj in report.objects), default=0) + 1
    objects = list(report.objects)
    for model_path, start, end in fillers:
        centre_x = (start[0] + end[0]) * 0.5
        centre_z = (start[1] + end[1]) * 0.5
        if not (0.0 <= centre_x < spec.world_size and 0.0 <= centre_z < spec.world_size):
            continue
        objects.append(
            _p._road_object_on_slope(
                next_id,
                model_path,
                start,
                end,
                elevations,
                spec,
                vertical_offset=_p._STOCK_GRAVEL_VERTICAL_OFFSET_METRES,
            )
        )
        next_id += 1

    added = len(objects) - len(report.objects)
    if not added:
        return report, 0
    return replace(
        report,
        objects=tuple(objects),
        chain_count=report.chain_count + added,
        short_piece_objects=report.short_piece_objects + added,
    ), added


def _gap_fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _GAP_ORIGINAL_FIT is None:
        raise RuntimeError("gravel gap policy is not installed")
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return _GAP_ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )

    deferred_completion: tuple[int, str] | None = None

    def progress(value: int, message: str) -> None:
        nonlocal deferred_completion
        if value >= 100 and message.startswith("Stock road fitting complete:"):
            deferred_completion = (value, message)
            return
        if progress_callback is not None:
            progress_callback(value, message)

    report = _GAP_ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress if progress_callback is not None else None,
    )
    context = _rq._Context(
        elevations,
        spec,
        _rq._junction_geometry(dataset, projection, spec),
    )
    report, added = _bridge_short_gravel_gaps(
        report,
        dataset,
        projection,
        elevations,
        spec,
        context,
    )
    if progress_callback is not None and added:
        progress_callback(99, f"Closed {added:,} short gravel road gaps")
    if progress_callback is not None and deferred_completion is not None:
        progress_callback(
            deferred_completion[0],
            f"Stock road fitting complete: {len(report.objects):,} objects in {report.chain_count:,} chains",
        )
    return report


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
        raise RuntimeError("gravel family policy is not installed")
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
    return _replace_gravel_caps(report, dataset, projection, elevations, spec)


def install_gravel_junction_policy() -> None:
    """Install gravel-junction trimming before the gap stage."""

    global _ORIGINAL_JUNCTION_GEOMETRY, _ORIGINAL_EXIT_DISTANCE
    global _JUNCTION_INSTALLED
    if _JUNCTION_INSTALLED:
        return

    _ORIGINAL_JUNCTION_GEOMETRY = _rq._junction_geometry
    _ORIGINAL_EXIT_DISTANCE = _rq._exit_distance
    _rq._junction_geometry = _junction_geometry
    _rq._exit_distance = _exit_distance
    _rq._quality_window = _quality_window
    _JUNCTION_INSTALLED = True


def install_gravel_gap_policy() -> None:
    """Install short-gap repair between junction trimming and family caps."""

    global _GAP_ORIGINAL_FIT, _GAP_INSTALLED
    if _GAP_INSTALLED:
        return
    _GAP_ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _gap_fit
    _generator.fit_road_objects = _gap_fit
    _GAP_INSTALLED = True


def install_gravel_family_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    if not _JUNCTION_INSTALLED:
        raise RuntimeError("gravel junction trimming must install first")
    if not _GAP_INSTALLED:
        raise RuntimeError("gravel gap repair must install first")

    _ORIGINAL_FIT = _p.fit_road_objects
    _pi._road_lods = _road_lods
    _pi.ProceduralInfrastructureLibrary.register_model_usage = _register_model_usage
    _pi.is_generated_gravel_junction_model = is_generated_gravel_family_junction
    _p.is_generated_gravel_junction_model = is_generated_gravel_family_junction
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
