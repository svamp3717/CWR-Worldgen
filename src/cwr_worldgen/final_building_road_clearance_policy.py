# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep final procedural buildings out of the final fitted road surfaces.

Building footprints are planned before terrain solving, while stock-road fitting,
junction replacement and final road deduplication happen later.  A source-road
corridor check therefore cannot see every surface that will actually be written to
the WRP.  This policy records the final fitted road report and, immediately before
non-road placement, translates only buildings that still intersect those final
surfaces.

The pass is deliberately bounded: roads and building footprints are spatially
indexed, only conflicting buildings are searched, and relocation uses calculated
escape vectors rather than metre-by-metre stepping.  Curves and junction arms are
represented by short centre-line capsules so they participate in clearance without
requiring P3D polygon decoding at generation time.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
import re
from typing import Callable, Sequence

from . import generator as _generator
from . import osm as _osm
from . import playability as _p

PointXZ = tuple[float, float]

_ROAD_BUCKET_METRES = 25.0
_BUILDING_BUCKET_METRES = 24.0
_ROAD_CLEARANCE_METRES = 0.75
_ESCAPE_SAFETY_METRES = 0.20
_MAXIMUM_RELOCATION_METRES = 30.0
_MAXIMUM_PRIMARY_ESCAPE_SURFACES = 4
_MAXIMUM_FIRST_PASS_VECTORS = 12
_MAXIMUM_CORRECTION_VECTORS = 4
_MAXIMUM_VERTICAL_TERRAIN_GAP_METRES = 2.0
_PROGRESS_BUCKET_PERCENT = 2
_RAW_PROGRESS_PERCENT = 52
_CACHE_REVISION = "final-road-building-clearance-v1"

_WIDTHS = {
    "sil": 4.55,
    "kos": 4.55,
    "asf": 3.50,
    "ces": 1.75,
    "gravel": 2.30,
}
_STOCK_STRAIGHT = re.compile(
    r"^(?P<family>sil|kos|asf|ces)(?P<nominal>25|12|6)\.p3d$", re.I
)
_STOCK_CURVE = re.compile(
    r"^(?P<family>sil|kos|asf|ces)10 (?P<radius>25|50|75|100)\.p3d$", re.I
)
_STOCK_T = re.compile(
    r"^kr_new_(?P<main>sil|asf|kos)_(?P<branch>sil|ces|asf|kos)_t\.p3d$", re.I
)
_STOCK_X = re.compile(r"^kr_new_silxsil\.p3d$", re.I)
_GRAVEL = re.compile(
    r"^gravel(?P<nominal>25|12|6|3)(?:_(?P<side>[lr])(?P<degrees>05|10|15|20|30|45))?\.p3d$",
    re.I,
)
_GRAVEL_JUNCTION = re.compile(
    r"^gravel_j(?P<degree>[34])(?:_(?P<variant>t(?:30|45|60|75)[lr]|t90|y120|x(?:30|45|60|75|90)))?\.p3d$",
    re.I,
)

_INSTALLED = False
_ORIGINAL_FIT = None
_ORIGINAL_GENERATE_WORLD_OBJECTS = None
_ORIGINAL_LOAD_NONROAD_OBJECTS = None

# The road fitter and non-road generator run serially in one build context. Store
# the final report with object identities for the inputs that produced it. This
# prevents an unrelated direct generate_world_objects() call from consuming stale
# roads from an earlier world.
_FINAL_ROADS: ContextVar[tuple[int, int, int, int, object] | None] = ContextVar(
    "cwr_final_roads_for_building_clearance", default=None
)


@dataclass(frozen=True, slots=True)
class _RoadPrimitive:
    object_id: int
    start: PointXZ
    end: PointXZ
    half_width: float
    elevation: float

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        pad = self.half_width + _ROAD_CLEARANCE_METRES
        return (
            min(self.start[0], self.end[0]) - pad,
            min(self.start[1], self.end[1]) - pad,
            max(self.start[0], self.end[0]) + pad,
            max(self.start[1], self.end[1]) + pad,
        )


@dataclass(frozen=True, slots=True)
class FinalBuildingRoadConflictReport:
    buildings: int
    road_primitives: int
    conflicted: int
    moved: int
    rejected: int
    nearby_road_checks: int


class _RoadPrimitiveIndex:
    def __init__(self, primitives: Sequence[_RoadPrimitive]) -> None:
        self.primitives = tuple(primitives)
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, primitive in enumerate(self.primitives):
            min_x, min_z, max_x, max_z = primitive.bounds
            for bz in _bucket_range(min_z, max_z, _ROAD_BUCKET_METRES):
                for bx in _bucket_range(min_x, max_x, _ROAD_BUCKET_METRES):
                    buckets.setdefault((bx, bz), []).append(index)
        self.buckets = {key: tuple(values) for key, values in buckets.items()}

    def candidates(self, polygon: Sequence[PointXZ]) -> tuple[int, ...]:
        if not polygon:
            return ()
        min_x, min_z, max_x, max_z = _polygon_bounds(polygon)
        found: set[int] = set()
        for bz in _bucket_range(min_z, max_z, _ROAD_BUCKET_METRES):
            for bx in _bucket_range(min_x, max_x, _ROAD_BUCKET_METRES):
                found.update(self.buckets.get((bx, bz), ()))
        return tuple(sorted(found))


class _BuildingFootprintIndex:
    """Mutable spatial index of the currently accepted building footprints."""

    def __init__(self, polygons: Sequence[Sequence[PointXZ]]) -> None:
        self.polygons: list[tuple[PointXZ, ...] | None] = [
            tuple(polygon) if len(polygon) >= 3 else None for polygon in polygons
        ]
        self.memberships: list[tuple[tuple[int, int], ...]] = [
            () for _ in self.polygons
        ]
        self.buckets: dict[tuple[int, int], set[int]] = {}
        for index, polygon in enumerate(self.polygons):
            if polygon is not None:
                self._add(index, polygon)

    def _keys(self, polygon: Sequence[PointXZ]) -> tuple[tuple[int, int], ...]:
        min_x, min_z, max_x, max_z = _polygon_bounds(polygon)
        return tuple(
            (bx, bz)
            for bz in _bucket_range(min_z, max_z, _BUILDING_BUCKET_METRES)
            for bx in _bucket_range(min_x, max_x, _BUILDING_BUCKET_METRES)
        )

    def _add(self, index: int, polygon: tuple[PointXZ, ...]) -> None:
        keys = self._keys(polygon)
        self.memberships[index] = keys
        for key in keys:
            self.buckets.setdefault(key, set()).add(index)

    def update(self, index: int, polygon: Sequence[PointXZ] | None) -> None:
        for key in self.memberships[index]:
            members = self.buckets.get(key)
            if members is not None:
                members.discard(index)
                if not members:
                    self.buckets.pop(key, None)
        self.memberships[index] = ()
        stored = tuple(polygon) if polygon is not None and len(polygon) >= 3 else None
        self.polygons[index] = stored
        if stored is not None:
            self._add(index, stored)

    def overlaps_other(self, index: int, polygon: Sequence[PointXZ]) -> bool:
        keys = self._keys(polygon)
        candidates: set[int] = set()
        for key in keys:
            candidates.update(self.buckets.get(key, ()))
        for other_index in sorted(candidates):
            if other_index == index:
                continue
            other = self.polygons[other_index]
            if other is not None and _osm._polygons_intersect(polygon, other):
                return True
        return False


def _filename(path: str) -> str:
    return path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()


def _bucket_range(minimum: float, maximum: float, size: float) -> range:
    return range(math.floor(minimum / size), math.floor(maximum / size) + 1)


def _polygon_bounds(polygon: Sequence[PointXZ]) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


def _translate_polygon(
    polygon: Sequence[PointXZ], dx: float, dz: float
) -> tuple[PointXZ, ...]:
    return tuple((x + dx, z + dz) for x, z in polygon)


def _polygon_centroid(polygon: Sequence[PointXZ]) -> PointXZ:
    if not polygon:
        return 0.0, 0.0
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def _world_point(local: PointXZ, obj) -> PointXZ:
    angle = math.radians(float(obj.heading_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, z = local
    return (
        float(obj.x) + x * cosine + z * sine,
        float(obj.z) - x * sine + z * cosine,
    )


def _make_primitive(
    obj,
    start_local: PointXZ,
    end_local: PointXZ,
    half_width: float,
) -> _RoadPrimitive:
    return _RoadPrimitive(
        int(obj.object_id),
        _world_point(start_local, obj),
        _world_point(end_local, obj),
        float(half_width),
        float(obj.y),
    )


def _scaled_piece_length(nominal: int, spec) -> float:
    return float(spec.road_segment_length) * float(nominal) / 25.0


def _stock_curve_points(family: str, radius: float) -> tuple[PointXZ, PointXZ]:
    angle = math.radians(10.0)
    half = angle * 0.5
    chord = 2.0 * radius * math.sin(half)
    width = _WIDTHS[family]
    midpoint = (
        width * (1.0 - math.cos(angle)) * 0.5,
        -width * math.sin(angle) * 0.5,
    )
    unit = math.sin(half), math.cos(half)
    return (
        (
            midpoint[0] - unit[0] * chord * 0.5,
            midpoint[1] - unit[1] * chord * 0.5,
        ),
        (
            midpoint[0] + unit[0] * chord * 0.5,
            midpoint[1] + unit[1] * chord * 0.5,
        ),
    )


def _gravel_curve_points(
    length: float, side: str, degrees: float
) -> tuple[PointXZ, ...]:
    if degrees <= 1.0e-9:
        return ((0.0, -length * 0.5), (0.0, length * 0.5))
    angle = math.radians(degrees)
    radius = length / max(angle, 1.0e-6)
    sign = 1.0 if side.casefold() == "r" else -1.0
    # The model is centred at the arc midpoint and faces along +Z there.
    values = []
    steps = max(2, int(math.ceil(degrees / 10.0)))
    for index in range(steps + 1):
        alpha = -angle * 0.5 + angle * index / steps
        values.append(
            (
                sign * radius * (1.0 - math.cos(alpha)),
                radius * math.sin(alpha),
            )
        )
    return tuple(values)


def _gravel_junction_headings(
    degree: int, variant: str | None
) -> tuple[float, ...]:
    value = (variant or ("t90" if degree == 3 else "x90")).casefold()
    if value == "y120":
        return (0.0, 120.0, 240.0)
    match = re.fullmatch(r"t(30|45|60|75)([lr])", value)
    if match is not None:
        turn = float(match.group(1)) * (1.0 if match.group(2) == "r" else -1.0)
        return (0.0, 180.0, turn)
    if value == "t90":
        return (0.0, 180.0, 90.0)
    match = re.fullmatch(r"x(30|45|60|75|90)", value)
    if match is not None:
        turn = float(match.group(1))
        return (0.0, 180.0, turn, turn + 180.0)
    return ()


def _road_object_primitives(obj, spec) -> tuple[_RoadPrimitive, ...]:
    filename = _filename(obj.model_path)

    match = _STOCK_STRAIGHT.fullmatch(filename)
    if match is not None:
        family = match.group("family").casefold()
        length = _scaled_piece_length(int(match.group("nominal")), spec)
        return (
            _make_primitive(
                obj, (0.0, -length * 0.5), (0.0, length * 0.5), _WIDTHS[family]
            ),
        )

    match = _STOCK_CURVE.fullmatch(filename)
    if match is not None:
        family = match.group("family").casefold()
        start, end = _stock_curve_points(family, float(match.group("radius")))
        # The maximum sagitta of a 10-degree stock curve is below 0.4 m even at
        # radius 100. Expanding the chord capsule by that amount safely covers
        # the visible arc without requiring a curve mesh parser here.
        return (_make_primitive(obj, start, end, _WIDTHS[family] + 0.40),)

    match = _STOCK_T.fullmatch(filename)
    if match is not None:
        main = match.group("main").casefold()
        branch = match.group("branch").casefold()
        radius = 6.25
        centre_x = (radius - _WIDTHS[main]) * 0.5
        centre = (centre_x, 0.0)
        definitions = (
            ((centre_x, radius), main),
            ((centre_x, -radius), main),
            ((centre_x - radius, 0.0), branch),
        )
        return tuple(
            _make_primitive(obj, centre, endpoint, _WIDTHS[family])
            for endpoint, family in definitions
        )

    if _STOCK_X.fullmatch(filename):
        radius = 6.25
        centre = (0.0, 0.0)
        endpoints = (
            (0.0, radius),
            (0.0, -radius),
            (radius, 0.0),
            (-radius, 0.0),
        )
        return tuple(
            _make_primitive(obj, centre, endpoint, _WIDTHS["sil"])
            for endpoint in endpoints
        )

    match = _GRAVEL.fullmatch(filename)
    if match is not None:
        length = _scaled_piece_length(int(match.group("nominal")), spec)
        side = match.group("side")
        degrees = float(match.group("degrees") or 0.0)
        if side is None or degrees <= 0.0:
            points = ((0.0, -length * 0.5), (0.0, length * 0.5))
        else:
            points = _gravel_curve_points(length, side, degrees)
        return tuple(
            _make_primitive(obj, start, end, _WIDTHS["gravel"] + (0.20 if side else 0.0))
            for start, end in zip(points, points[1:])
        )

    match = _GRAVEL_JUNCTION.fullmatch(filename)
    if match is not None:
        degree = int(match.group("degree"))
        headings = _gravel_junction_headings(degree, match.group("variant"))
        centre = (0.0, 0.0)
        radius = 4.0
        return tuple(
            _make_primitive(
                obj,
                centre,
                (
                    math.sin(math.radians(direction)) * radius,
                    math.cos(math.radians(direction)) * radius,
                ),
                _WIDTHS["gravel"],
            )
            for direction in headings
        )

    return ()


def _road_primitives(report, elevations, spec) -> tuple[_RoadPrimitive, ...]:
    result: list[_RoadPrimitive] = []
    for obj in report.objects:
        for primitive in _road_object_primitives(obj, spec):
            midpoint = (
                (primitive.start[0] + primitive.end[0]) * 0.5,
                (primitive.start[1] + primitive.end[1]) * 0.5,
            )
            terrain = _osm._sample_elevation(
                elevations, spec.cells, spec.cell_size, midpoint[0], midpoint[1]
            )
            # A final road surface well above/below the local terrain is a bridge,
            # overpass or otherwise grade-separated object. Do not move a ground
            # building merely because their X/Z projections cross.
            if (
                abs(float(primitive.elevation) - float(terrain))
                <= _MAXIMUM_VERTICAL_TERRAIN_GAP_METRES
            ):
                result.append(primitive)
    return tuple(result)


def _orientation(a: PointXZ, b: PointXZ, c: PointXZ) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: PointXZ, b: PointXZ, point: PointXZ) -> bool:
    return (
        min(a[0], b[0]) - 1.0e-9 <= point[0] <= max(a[0], b[0]) + 1.0e-9
        and min(a[1], b[1]) - 1.0e-9 <= point[1] <= max(a[1], b[1]) + 1.0e-9
    )


def _segments_intersect(a: PointXZ, b: PointXZ, c: PointXZ, d: PointXZ) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    epsilon = 1.0e-9
    if (
        ((ab_c > epsilon and ab_d < -epsilon) or (ab_c < -epsilon and ab_d > epsilon))
        and ((cd_a > epsilon and cd_b < -epsilon) or (cd_a < -epsilon and cd_b > epsilon))
    ):
        return True
    return (
        (abs(ab_c) <= epsilon and _on_segment(a, b, c))
        or (abs(ab_d) <= epsilon and _on_segment(a, b, d))
        or (abs(cd_a) <= epsilon and _on_segment(c, d, a))
        or (abs(cd_b) <= epsilon and _on_segment(c, d, b))
    )


def _point_segment_distance_sq(point: PointXZ, start: PointXZ, end: PointXZ) -> float:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    denominator = dx * dx + dz * dz
    if denominator <= 1.0e-12:
        return (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    nearest = (start[0] + dx * fraction, start[1] + dz * fraction)
    return (point[0] - nearest[0]) ** 2 + (point[1] - nearest[1]) ** 2


def _segment_distance_sq(a: PointXZ, b: PointXZ, c: PointXZ, d: PointXZ) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance_sq(a, c, d),
        _point_segment_distance_sq(b, c, d),
        _point_segment_distance_sq(c, a, b),
        _point_segment_distance_sq(d, a, b),
    )


def _point_in_polygon(point: PointXZ, polygon: Sequence[PointXZ]) -> bool:
    inside = False
    x, z = point
    previous = polygon[-1]
    for current in polygon:
        if _point_segment_distance_sq(point, previous, current) <= 1.0e-12:
            return True
        x1, z1 = previous
        x2, z2 = current
        if (z1 > z) != (z2 > z):
            crossing = (x2 - x1) * (z - z1) / (z2 - z1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _primitive_intersects_polygon(
    primitive: _RoadPrimitive, polygon: Sequence[PointXZ]
) -> bool:
    if len(polygon) < 3:
        return False
    min_x, min_z, max_x, max_z = _polygon_bounds(polygon)
    p_min_x, p_min_z, p_max_x, p_max_z = primitive.bounds
    if max_x < p_min_x or p_max_x < min_x or max_z < p_min_z or p_max_z < min_z:
        return False
    limit = primitive.half_width + _ROAD_CLEARANCE_METRES
    limit_sq = limit * limit
    if _point_in_polygon(primitive.start, polygon) or _point_in_polygon(
        primitive.end, polygon
    ):
        return True
    previous = polygon[-1]
    for current in polygon:
        if _segment_distance_sq(
            primitive.start, primitive.end, previous, current
        ) <= limit_sq:
            return True
        previous = current
    return False


def _conflicts(
    polygon: Sequence[PointXZ],
    road_index: _RoadPrimitiveIndex,
) -> tuple[tuple[_RoadPrimitive, ...], int]:
    conflicts: list[_RoadPrimitive] = []
    checked = 0
    for index in road_index.candidates(polygon):
        checked += 1
        primitive = road_index.primitives[index]
        if _primitive_intersects_polygon(primitive, polygon):
            conflicts.append(primitive)
    conflicts.sort(key=lambda primitive: (primitive.object_id, primitive.start, primitive.end))
    return tuple(conflicts), checked


def _nearest_point_on_segment(
    point: PointXZ, start: PointXZ, end: PointXZ
) -> PointXZ:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    denominator = dx * dx + dz * dz
    if denominator <= 1.0e-12:
        return start
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    return start[0] + dx * fraction, start[1] + dz * fraction


def _escape_vectors(
    polygon: Sequence[PointXZ], primitive: _RoadPrimitive
) -> tuple[tuple[float, float], ...]:
    dx = primitive.end[0] - primitive.start[0]
    dz = primitive.end[1] - primitive.start[1]
    length = math.hypot(dx, dz)
    if length <= 1.0e-9:
        return ()
    tx, tz = dx / length, dz / length
    nx, nz = -tz, tx
    lateral = tuple(
        (point[0] - primitive.start[0]) * nx
        + (point[1] - primitive.start[1]) * nz
        for point in polygon
    )
    along = tuple(
        (point[0] - primitive.start[0]) * tx
        + (point[1] - primitive.start[1]) * tz
        for point in polygon
    )
    limit = primitive.half_width + _ROAD_CLEARANCE_METRES + _ESCAPE_SAFETY_METRES
    positive = max(0.0, limit - min(lateral))
    negative = max(0.0, max(lateral) + limit)
    before = max(0.0, max(along) + limit)
    after = max(0.0, length + limit - min(along))
    vectors = (
        (nx * positive, nz * positive),
        (-nx * negative, -nz * negative),
        (-tx * before, -tz * before),
        (tx * after, tz * after),
    )
    return tuple(
        vector
        for vector in sorted(
            vectors,
            key=lambda value: (
                math.hypot(value[0], value[1]),
                round(value[0], 9),
                round(value[1], 9),
            ),
        )
        if math.hypot(vector[0], vector[1]) > 1.0e-6
    )


def _maximum_shift_for_polygon(polygon: Sequence[PointXZ]) -> float:
    min_x, min_z, max_x, max_z = _polygon_bounds(polygon)
    span = max(max_x - min_x, max_z - min_z)
    return min(
        _MAXIMUM_RELOCATION_METRES,
        max(12.0, span * 0.65 + 8.0),
    )


def _terrain_relocation_allowed(
    original: Sequence[PointXZ],
    candidate: Sequence[PointXZ],
    elevations,
    spec,
) -> bool:
    current_min, current_max = _osm._polygon_elevation_extrema(
        elevations, spec.cells, spec.cell_size, original
    )
    candidate_min, candidate_max = _osm._polygon_elevation_extrema(
        elevations, spec.cells, spec.cell_size, candidate
    )
    current_relief = max(0.0, current_max - current_min)
    candidate_relief = max(0.0, candidate_max - candidate_min)
    maximum_foundation = max(
        0.0, float(getattr(spec, "building_foundation_maximum_depth", 2.5))
    )
    ground_clearance = max(
        0.0, float(getattr(spec, "building_ground_clearance", 0.05))
    )
    safety = max(0.0, float(getattr(spec, "building_foundation_safety", 0.20)))
    nominal_allowed_relief = max(
        0.0, maximum_foundation - ground_clearance - safety
    )
    # Existing terrain planning may already have accepted an unusually rough
    # footprint. A final road correction may not make that situation materially
    # worse, but it must not introduce a fresh cliff under the building.
    allowed_relief = max(nominal_allowed_relief, current_relief + 0.50)
    return candidate_relief <= allowed_relief + 1.0e-9


def _candidate_allowed(
    plan_index: int,
    original_polygon: Sequence[PointXZ],
    candidate: Sequence[PointXZ],
    building_index: _BuildingFootprintIndex,
    elevations,
    raster,
    spec,
) -> bool:
    edge = 0.25
    if not all(
        edge <= x <= float(spec.world_size) - edge
        and edge <= z <= float(spec.world_size) - edge
        for x, z in candidate
    ):
        return False
    if _osm._polygon_overlaps_mask(
        raster.water, spec.cells, spec.world_size, candidate
    ):
        return False
    if building_index.overlaps_other(plan_index, candidate):
        return False
    return _terrain_relocation_allowed(
        original_polygon, candidate, elevations, spec
    )


def _unique_escape_vectors(
    polygon: Sequence[PointXZ],
    conflicts: Sequence[_RoadPrimitive],
) -> tuple[tuple[float, float], ...]:
    values: dict[tuple[int, int], tuple[float, float]] = {}
    for primitive in conflicts[:_MAXIMUM_PRIMARY_ESCAPE_SURFACES]:
        for vector in _escape_vectors(polygon, primitive):
            key = (round(vector[0] * 1000.0), round(vector[1] * 1000.0))
            values.setdefault(key, vector)

    # At junctions the shortest vector for any one arm can still land on another
    # arm. Add one aggregate "away from the nearby road mass" direction before
    # resorting to the bounded two-vector correction below.
    centre = _polygon_centroid(polygon)
    away_x = away_z = 0.0
    scale = 0.0
    for primitive in conflicts[:_MAXIMUM_PRIMARY_ESCAPE_SURFACES]:
        nearest = _nearest_point_on_segment(centre, primitive.start, primitive.end)
        vx, vz = centre[0] - nearest[0], centre[1] - nearest[1]
        distance = math.hypot(vx, vz)
        if distance <= 1.0e-6:
            dx = primitive.end[0] - primitive.start[0]
            dz = primitive.end[1] - primitive.start[1]
            axis_length = max(1.0e-9, math.hypot(dx, dz))
            vx, vz = -dz / axis_length, dx / axis_length
            distance = 1.0
        away_x += vx / distance
        away_z += vz / distance
        primitive_vectors = _escape_vectors(polygon, primitive)
        if primitive_vectors:
            scale = max(
                scale,
                min(math.hypot(dx, dz) for dx, dz in primitive_vectors),
            )
    away_length = math.hypot(away_x, away_z)
    if away_length > 1.0e-6 and scale > 0.0:
        for multiplier in (1.0, 1.35):
            vector = (
                away_x / away_length * scale * multiplier,
                away_z / away_length * scale * multiplier,
            )
            key = (round(vector[0] * 1000.0), round(vector[1] * 1000.0))
            values.setdefault(key, vector)

    return tuple(
        sorted(
            values.values(),
            key=lambda value: (
                math.hypot(value[0], value[1]),
                round(value[0], 9),
                round(value[1], 9),
            ),
        )
    )


def resolve_final_building_road_conflicts(
    plans,
    road_report,
    elevations,
    raster,
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Translate every final building footprint that intersects a final road."""
    plans = tuple(plans or ())
    if not plans or road_report is None or not getattr(road_report, "objects", ()):
        return plans, FinalBuildingRoadConflictReport(
            len(plans), 0, 0, 0, 0, 0
        )

    primitives = _road_primitives(road_report, elevations, spec)
    if not primitives:
        return plans, FinalBuildingRoadConflictReport(
            len(plans), 0, 0, 0, 0, 0
        )
    road_index = _RoadPrimitiveIndex(primitives)
    building_index = _BuildingFootprintIndex(
        tuple(tuple(plan.support_polygon) for plan in plans)
    )
    resolved = list(plans)

    total = len(plans)
    conflicted = moved = rejected = checks = 0
    last_bucket = -1

    def report(completed: int, *, force: bool = False) -> None:
        nonlocal last_bucket
        if progress_callback is None:
            return
        percent = 100 if total <= 0 else min(100, int(completed * 100 / total))
        bucket = percent // _PROGRESS_BUCKET_PERCENT
        if (
            not force
            and completed not in {1, total}
            and bucket <= last_bucket
        ):
            return
        last_bucket = bucket
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            "Resolving final road/building conflicts "
            f"({completed:,}/{total:,}, {percent}%; {conflicted:,} conflicts; "
            f"{moved:,} moved; {rejected:,} rejected; {checks:,} nearby road surfaces)",
        )

    report(0, force=True)
    for completed, plan in enumerate(plans, start=1):
        original = tuple(plan.support_polygon)
        if len(original) < 3:
            report(completed)
            continue

        conflicts, tested = _conflicts(original, road_index)
        checks += tested
        if not conflicts:
            report(completed)
            continue

        conflicted += 1
        maximum_shift = _maximum_shift_for_polygon(original)
        accepted_plan = None
        vectors = _unique_escape_vectors(original, conflicts)[
            :_MAXIMUM_FIRST_PASS_VECTORS
        ]

        for dx, dz in vectors:
            if math.hypot(dx, dz) > maximum_shift + 1.0e-9:
                continue
            candidate = _translate_polygon(original, dx, dz)
            if not _candidate_allowed(
                completed - 1,
                original,
                candidate,
                building_index,
                elevations,
                raster,
                spec,
            ):
                continue
            remaining, tested = _conflicts(candidate, road_index)
            checks += tested
            if not remaining:
                accepted_plan = replace(
                    plan,
                    x=float(plan.x) + dx,
                    z=float(plan.z) + dz,
                    support_polygon=candidate,
                    road_nudged=True,
                )
                break

            for correction_x, correction_z in _unique_escape_vectors(
                candidate, remaining
            )[:_MAXIMUM_CORRECTION_VECTORS]:
                total_dx = dx + correction_x
                total_dz = dz + correction_z
                if math.hypot(total_dx, total_dz) > maximum_shift + 1.0e-9:
                    continue
                corrected = _translate_polygon(original, total_dx, total_dz)
                if not _candidate_allowed(
                    completed - 1,
                    original,
                    corrected,
                    building_index,
                    elevations,
                    raster,
                    spec,
                ):
                    continue
                corrected_conflicts, tested = _conflicts(corrected, road_index)
                checks += tested
                if corrected_conflicts:
                    continue
                accepted_plan = replace(
                    plan,
                    x=float(plan.x) + total_dx,
                    z=float(plan.z) + total_dz,
                    support_polygon=corrected,
                    road_nudged=True,
                )
                break
            if accepted_plan is not None:
                break

        index = completed - 1
        if accepted_plan is None:
            # No position within the bounded, terrain-safe search can clear the
            # final road surfaces without colliding with water/another building.
            # Omitting the structure is preferable to writing a road through it.
            resolved[index] = None
            building_index.update(index, None)
            rejected += 1
        else:
            resolved[index] = accepted_plan
            building_index.update(index, accepted_plan.support_polygon)
            moved += 1

        report(completed)

    final_plans = tuple(plan for plan in resolved if plan is not None)
    report(total, force=True)
    return final_plans, FinalBuildingRoadConflictReport(
        total,
        len(primitives),
        conflicted,
        moved,
        rejected,
        checks,
    )


def _road_context_matches(dataset, projection, elevations, spec):
    stored = _FINAL_ROADS.get()
    if stored is None:
        return None
    dataset_id, projection_id, elevations_id, spec_id, report = stored
    if (
        dataset_id == id(dataset)
        and projection_id == id(projection)
        and elevations_id == id(elevations)
        and spec_id == id(spec)
    ):
        return report
    return None


def install_final_building_road_clearance_policy() -> None:
    """Apply final-road building clearance after final road deduplication."""
    global _INSTALLED, _ORIGINAL_FIT
    global _ORIGINAL_GENERATE_WORLD_OBJECTS, _ORIGINAL_LOAD_NONROAD_OBJECTS
    if _INSTALLED:
        return

    _ORIGINAL_FIT = _p.fit_road_objects
    _ORIGINAL_GENERATE_WORLD_OBJECTS = _osm.generate_world_objects
    _ORIGINAL_LOAD_NONROAD_OBJECTS = _generator._load_nonroad_objects

    def recording_fit(
        dataset,
        projection,
        elevations,
        spec,
        *,
        starting_id: int = 1,
        progress_callback=None,
    ):
        report = _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        _FINAL_ROADS.set(
            (id(dataset), id(projection), id(elevations), id(spec), report)
        )
        return report

    def road_safe_generate_world_objects(
        dataset,
        projection,
        raster,
        elevations,
        spec,
        *args,
        **kwargs,
    ):
        include_roads = bool(kwargs.get("include_roads", True))
        plans = kwargs.get("building_placement_plans")
        report = _road_context_matches(dataset, projection, elevations, spec)
        if (
            not include_roads
            and plans is not None
            and report is not None
        ):
            plans, _conflict_report = resolve_final_building_road_conflicts(
                plans,
                report,
                elevations,
                raster,
                spec,
                progress_callback=kwargs.get("progress_callback"),
            )
            kwargs["building_placement_plans"] = plans
        return _ORIGINAL_GENERATE_WORLD_OBJECTS(
            dataset,
            projection,
            raster,
            elevations,
            spec,
            *args,
            **kwargs,
        )

    def cache_salted_load_nonroad_objects(*args, **kwargs):
        if "road_fingerprint" in kwargs:
            kwargs["road_fingerprint"] = (
                str(kwargs["road_fingerprint"]) + ":" + _CACHE_REVISION
            )
        return _ORIGINAL_LOAD_NONROAD_OBJECTS(*args, **kwargs)

    _p.fit_road_objects = recording_fit
    _generator.fit_road_objects = recording_fit
    _osm.generate_world_objects = road_safe_generate_world_objects
    _generator.generate_world_objects = road_safe_generate_world_objects
    _generator._load_nonroad_objects = cache_salted_load_nonroad_objects
    _INSTALLED = True
