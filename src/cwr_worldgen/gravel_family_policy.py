# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed reusable road-piece family for generated gravel intersections.

Generated gravel straights and curves already behave like a conventional CWA
road family. This policy gives intersections the same treatment: a finite set
of reusable T/Y/crossroad models chosen by angle bucket and rotated into place.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
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
_ORIGINAL_FIT = None
_ORIGINAL_ROAD_LODS = _pi._road_lods
_ORIGINAL_REGISTER_MODEL_USAGE = _pi.ProceduralInfrastructureLibrary.register_model_usage
_ORIGINAL_IS_GRAVEL_JUNCTION = _pi.is_generated_gravel_junction_model
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
    """Return the best reusable junction shape and its local +Z world axis.

    Do not assume that a three-way junction contains a truly straight 180-degree
    main road. Real OSM service/track junctions often have three skewed arms.
    Evaluate the complete fixed family and rotate each candidate to minimize the
    worst arm-heading error, then use RMS error as the deterministic tiebreaker.
    """

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
        # An optimum rotation can always be represented by aligning at least one
        # model arm with one observed road arm. Degree is <=4, so exhaustively
        # checking the tiny assignment space is both clearer and cheaper than
        # carrying another geometry dependency into this hot path.
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
    gravel_junction_template_headings(normalized)  # validate spelling
    # Keep the historical orthogonal model paths for compatibility, but the
    # family policy rebuilds those two P3Ds with the same 4.0 m arm contract as
    # every skew/Y member. This preserves old references without preserving the
    # old short 2.45/3.0 m geometry that opened visible seams.
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
    """Distance from the node to the rendered fixed-family footprint on a ray."""

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
    arms = tuple(_arm_polygon(heading, half_width) for heading in gravel_junction_template_headings(variant))
    core = ShapelyPoint(0.0, 0.0).buffer(GRAVEL_JUNCTION_CORE_RADIUS_METRES, quad_segs=8)
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
        tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), resolution,
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
        (float(x), 0.0, float(z)) for x, z in tuple(polygon.exterior.coords)[:-1]
    )
    map_geometry = _pi._Lod(
        boundary, (), (), _pi._GEOMETRY_LOD, properties=(("map", "road"),)
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
        family_key = replace(key, subtype="gravel_j3_t90", length_dm=int(round(GRAVEL_JUNCTION_ARM_EXTENT_METRES * 20.0)))
        return _family_junction_lods(family_key, texture)
    if subtype == "gravel_j4":
        family_key = replace(key, subtype="gravel_j4_x90", length_dm=int(round(GRAVEL_JUNCTION_ARM_EXTENT_METRES * 20.0)))
        return _family_junction_lods(family_key, texture)
    if re.fullmatch(r"gravel_j[34]_.+", key.subtype, re.IGNORECASE):
        return _family_junction_lods(key, texture)
    return _ORIGINAL_ROAD_LODS(key, texture)


def _register_model_usage(self, model_path: str, count: int = 1) -> None:
    count = max(0, int(count))
    if count == 0:
        return
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
    match = _FAMILY_PATTERN.fullmatch(filename)
    if match and self.is_generated_model(model_path):
        degree = int(match.group("degree"))
        variant = match.group("variant").casefold()
        if (4 if variant.startswith("x") else 3) != degree:
            raise ValueError("gravel junction model degree mismatch")
        self._usage[_pi.InfrastructureModelKey(
            "road",
            filename[:-4].casefold(),
            int(round(_pi.GENERATED_GRAVEL_HALF_WIDTH_METRES * 20.0)),
            int(round(GRAVEL_JUNCTION_ARM_EXTENT_METRES * 20.0)),
        )] += count
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
        start = (junction.point[0] - axis[0] * half, junction.point[1] - axis[1] * half)
        end = (junction.point[0] + axis[0] * half, junction.point[1] + axis[1] * half)
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


def _fit(dataset, projection, elevations, spec, *, starting_id: int = 1, progress_callback=None):
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


def install_gravel_family_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    # Capture the current outer fitter at installation time. The module is
    # imported earlier by gravel_junction_policy, before gravel_gap_policy is
    # installed, so capturing at import time would accidentally bypass that
    # later repair layer.
    _ORIGINAL_FIT = _p.fit_road_objects
    _pi._road_lods = _road_lods
    _pi.ProceduralInfrastructureLibrary.register_model_usage = _register_model_usage
    _pi.is_generated_gravel_junction_model = is_generated_gravel_family_junction
    _p.is_generated_gravel_junction_model = is_generated_gravel_family_junction
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
