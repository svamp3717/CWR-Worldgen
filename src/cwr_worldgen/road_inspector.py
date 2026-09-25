# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only post-build inspector for CWA road geometry."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse, csv, html, io, json, math, re, struct
from typing import Iterable, Sequence

from .pbo import read_pbo

_HEADER = struct.Struct("<4sii")
_OBJECT = struct.Struct("<12fi76s")
_TEXTURE_BYTES = 512 * 32
_LENGTHS = {25: 25.0, 12: 12.5, 6: 6.25}
_WIDTHS = {
    "sil": 4.55, "silnice": 4.55, "kos": 4.55,
    "asf": 3.50, "asfaltka": 3.50, "ces": 1.75, "gravel": 2.30,
}
_CURVE_ANGLE = 10.0
_JUNCTION_RADIUS = 6.25
_GRAVEL_JUNCTION_RADIUS = 4.0
_STRAIGHT = re.compile(
    r"^(?:.*[\\/])(?P<family>silnice|asfaltka|sil|ces|asf|kos)"
    r"(?P<length>25|12|6)\.p3d$",
    re.I,
)
_CURVE = re.compile(r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)10 (?P<radius>25|50|75|100)\.p3d$", re.I)
_T = re.compile(r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?P<branch>sil|ces|asf|kos)_t\.p3d$", re.I)
_X = re.compile(r"^(?:.*[\\/])kr_new_silxsil\.p3d$", re.I)
_GRAVEL = re.compile(r"^(?:.*[\\/])gravel(?P<length>25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d$", re.I)
_GENERATED_PAVED = re.compile(
    r"^(?:.*[\\/])paved_w(?P<width>\d{3})_l(?P<length>\d{4})"
    r"(?:_(?P<side>[lr])(?P<degrees>\d{2}))?\.p3d$",
    re.I,
)
_GENERATED_PAVED_JUNCTION = re.compile(
    r"^(?:.*[\\/])paved_j(?P<degree>[34])_w(?P<width>\d{3})_h"
    r"(?P<headings>\d{3}(?:_\d{3}){2,3})\.p3d$",
    re.I,
)
_GRAVEL_JUNCTION = re.compile(
    r"^(?:.*[\\/])gravel_j(?P<degree>[34])(?:_(?P<variant>t(?:30|45|60|75)[lr]|t90|y120|x(?:30|45|60|75|90)))?\.p3d$",
    re.I,
)

DEFAULT_ENDPOINT_TOLERANCE_METRES = 0.20
DEFAULT_NEARBY_GAP_METRES = 1.50
DEFAULT_MINIMUM_EDGE_GAP_METRES = 0.08
DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES = 0.75
_PAVED_REPLACEMENT_CURVE_BUCKETS = (2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45)
_PAVED_REPLACEMENT_CATEGORIES = frozenset({
    "straight_miter",
    "curve_transition",
    "connector_gap",
})
_PAVED_REPLACEMENT_MINIMUM_TANGENT_ERROR_DEGREES = 0.75
_STOCK_REPAIR_POSITION_TOLERANCE_METRES = 0.08
_STOCK_REPAIR_MAXIMUM_PATH_DEVIATION_METRES = 1.0
_STOCK_REPAIR_RADII_METRES = (25, 50, 75, 100)
_STOCK_REPAIR_STRAIGHT_LENGTHS = _LENGTHS
_STOCK_REPAIR_FAMILIES = frozenset({"sil", "asf", "kos"})


@dataclass(frozen=True, slots=True)
class RoadEndpoint:
    object_id: int
    model_path: str
    family: str
    kind: str
    index: int
    point: tuple[float, float]
    tangent: float
    outward: float
    half_width: float


@dataclass(frozen=True, slots=True)
class RoadObject:
    object_id: int
    model_path: str
    x: float
    y: float
    z: float
    heading: float
    pitch: float
    family: str
    kind: str
    endpoints: tuple[RoadEndpoint, ...]

    @property
    def road_type(self) -> str:
        if self.family == "gravel":
            return "gravel"
        if self.family == "ces":
            return "dirt"
        return "paved"


@dataclass(frozen=True, slots=True)
class RoadIssue:
    issue_id: str
    severity: str
    score: float
    category: str
    x: float
    z: float
    object_ids: tuple[int, ...]
    models: tuple[str, ...]
    message: str
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class PavedStockRepairPlan:
    plan_id: str
    family: str
    replace_object_ids: tuple[int, ...]
    source_models: tuple[str, ...]
    issue_ids: tuple[str, ...]
    stock_models: tuple[str, ...]
    start: tuple[float, float]
    end: tuple[float, float]
    start_heading_degrees: float
    end_heading_degrees: float
    turn_sign: int
    first_turns: int
    first_radius: int
    middle_units: int
    counter_turns: int
    counter_radius: int
    merge_nominal: int
    maximum_path_deviation_metres: float
    final_length_error_metres: float
    maximum_join_angle_error_degrees: float


@dataclass(frozen=True, slots=True)
class PavedReplacementPlan:
    plan_id: str
    action: str
    model_path: str
    replace_object_ids: tuple[int, ...]
    source_models: tuple[str, ...]
    issue_ids: tuple[str, ...]
    start: tuple[float, float]
    end: tuple[float, float]
    width_metres: float
    length_metres: float
    curve_degrees: float
    maximum_edge_gap_metres: float
    maximum_tangent_error_degrees: float


@dataclass(frozen=True, slots=True)
class InspectionResult:
    input_path: str
    wrp_entry: str
    road_objects: tuple[RoadObject, ...]
    issues: tuple[RoadIssue, ...]
    paved_stock_repairs: tuple[PavedStockRepairPlan, ...] = ()
    paved_replacements: tuple[PavedReplacementPlan, ...] = ()

    @property
    def road_object_count(self) -> int:
        return len(self.road_objects)


def _model(path: str) -> str:
    return path.replace("/", "\\").casefold()


def _angle(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _axis_angle(a: float, b: float) -> float:
    d = _angle(a, b)
    return min(d, abs(180.0 - d))


def _world_point(local: tuple[float, float], origin: tuple[float, float], yaw: float, pitch: float) -> tuple[float, float]:
    x, z = local
    h, p = math.radians(yaw), math.radians(pitch)
    ch, sh, cp = math.cos(h), math.sin(h), math.cos(p)
    return origin[0] + x * ch + z * sh * cp, origin[1] - x * sh + z * ch * cp


def _world_heading(local: float, yaw: float, pitch: float) -> float:
    a, h = math.radians(local), math.radians(yaw)
    x, z, cp = math.sin(a), math.cos(a), math.cos(math.radians(pitch))
    wx = x * math.cos(h) + z * math.sin(h) * cp
    wz = -x * math.sin(h) + z * math.cos(h) * cp
    return math.degrees(math.atan2(wx, wz)) % 360.0


def _curve_points(family: str, radius: float) -> tuple[tuple[float, float], tuple[float, float]]:
    angle = math.radians(_CURVE_ANGLE)
    half = angle * 0.5
    chord = 2.0 * radius * math.sin(half)
    width = _WIDTHS[family]
    midpoint = (width * (1.0 - math.cos(angle)) * 0.5, -width * math.sin(angle) * 0.5)
    unit = math.sin(half), math.cos(half)
    return (
        (midpoint[0] - unit[0] * chord * 0.5, midpoint[1] - unit[1] * chord * 0.5),
        (midpoint[0] + unit[0] * chord * 0.5, midpoint[1] + unit[1] * chord * 0.5),
    )


def _endpoint(road_id: int, model: str, family: str, kind: str, index: int,
              point: tuple[float, float], tangent: float, outward: float) -> RoadEndpoint:
    return RoadEndpoint(road_id, model, family, kind, index, point, tangent % 180.0, outward % 360.0, _WIDTHS[family])


def _generated_paved_local_headings(
    length: float,
    side: str | None,
    degrees: float,
) -> tuple[float, float]:
    """Return the endpoint tangents of the generated quadratic road centreline."""
    if not side or degrees <= 1.0e-9:
        return 0.0, 0.0
    signed = degrees if side.casefold() == "r" else -degrees
    theta = math.radians(abs(signed))
    radius = length / max(1.0e-9, 2.0 * math.sin(theta * 0.5))
    sagitta = math.copysign(
        radius * (1.0 - math.cos(theta * 0.5)),
        signed,
    )
    control_x = sagitta * 2.0
    start = math.degrees(math.atan2(2.0 * control_x, length))
    end = math.degrees(math.atan2(-2.0 * control_x, length))
    return start, end


def _gravel_junction_headings(degree: int, variant: str | None) -> tuple[float, ...]:
    value = (variant or ("t90" if degree == 3 else "x90")).casefold()
    if value == "y120":
        return (0.0, 120.0, 240.0)
    match = re.fullmatch(r"t(30|45|60|75)([lr])", value)
    if match:
        turn = float(match.group(1)) * (1.0 if match.group(2) == "r" else -1.0)
        return (0.0, 180.0, turn)
    if value == "t90":
        return (0.0, 180.0, 90.0)
    match = re.fullmatch(r"x(30|45|60|75|90)", value)
    if match:
        turn = float(match.group(1))
        return (0.0, 180.0, turn, turn + 180.0)
    return ()


def _road(values) -> RoadObject | None:
    raw = values[13].split(b"\0", 1)[0]
    if not raw:
        return None
    try:
        model = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    path = _model(model)
    object_id = int(values[12])
    x, y, z = map(float, values[9:12])
    yaw = math.degrees(math.atan2(-float(values[2]), float(values[0]))) % 360.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(values[7])))))
    origin = x, z

    match = _STRAIGHT.fullmatch(path)
    if match:
        family = match.group("family").casefold()
        length = _LENGTHS[int(match.group("length"))]
        begin = _world_point((0.0, -length * 0.5), origin, yaw, pitch)
        end = _world_point((0.0, length * 0.5), origin, yaw, pitch)
        tangent = _world_heading(0.0, yaw, pitch)
        endpoints = (
            _endpoint(object_id, model, family, "straight", 0, begin, tangent, _world_heading(180.0, yaw, pitch)),
            _endpoint(object_id, model, family, "straight", 1, end, tangent, tangent),
        )
        return RoadObject(object_id, model, x, y, z, yaw, pitch, family, "straight", endpoints)

    match = _CURVE.fullmatch(path)
    if match:
        family = match.group("family").casefold()
        begin_local, end_local = _curve_points(family, float(match.group("radius")))
        begin = _world_point(begin_local, origin, yaw, pitch)
        end = _world_point(end_local, origin, yaw, pitch)
        begin_heading = _world_heading(0.0, yaw, pitch)
        end_heading = _world_heading(_CURVE_ANGLE, yaw, pitch)
        endpoints = (
            _endpoint(object_id, model, family, "curve", 0, begin, begin_heading, _world_heading(180.0, yaw, pitch)),
            _endpoint(object_id, model, family, "curve", 1, end, end_heading, end_heading),
        )
        return RoadObject(object_id, model, x, y, z, yaw, pitch, family, "curve", endpoints)

    match = _T.fullmatch(path)
    if match:
        main, branch = match.group("main").casefold(), match.group("branch").casefold()
        cx = (_JUNCTION_RADIUS - _WIDTHS[main]) * 0.5
        definitions = (
            ((cx, _JUNCTION_RADIUS), main, 0.0),
            ((cx, -_JUNCTION_RADIUS), main, 180.0),
            ((cx - _JUNCTION_RADIUS, 0.0), branch, 270.0),
        )
        endpoints = tuple(
            _endpoint(object_id, model, family, "junction", i, _world_point(local, origin, yaw, pitch),
                      _world_heading(direction, yaw, pitch), _world_heading(direction, yaw, pitch))
            for i, (local, family, direction) in enumerate(definitions)
        )
        return RoadObject(object_id, model, x, y, z, yaw, pitch, main, "junction_t", endpoints)

    if _X.fullmatch(path):
        definitions = (
            ((0.0, _JUNCTION_RADIUS), 0.0), ((0.0, -_JUNCTION_RADIUS), 180.0),
            ((_JUNCTION_RADIUS, 0.0), 90.0), ((-_JUNCTION_RADIUS, 0.0), 270.0),
        )
        endpoints = tuple(
            _endpoint(object_id, model, "sil", "junction", i, _world_point(local, origin, yaw, pitch),
                      _world_heading(direction, yaw, pitch), _world_heading(direction, yaw, pitch))
            for i, (local, direction) in enumerate(definitions)
        )
        return RoadObject(object_id, model, x, y, z, yaw, pitch, "sil", "junction_x", endpoints)

    match = _GENERATED_PAVED.fullmatch(path)
    if match:
        width = int(match.group("width")) / 10.0
        length = int(match.group("length")) / 10.0
        family = "asf" if width <= 7.5 else "sil"
        side = match.group("side")
        degrees = float(match.group("degrees") or 0.0)
        begin_local = (0.0, -length * 0.5)
        end_local = (0.0, length * 0.5)
        begin = _world_point(begin_local, origin, yaw, pitch)
        end = _world_point(end_local, origin, yaw, pitch)
        begin_local_heading, end_local_heading = _generated_paved_local_headings(
            length, side, degrees
        )
        begin_heading = _world_heading(begin_local_heading, yaw, pitch)
        end_heading = _world_heading(end_local_heading, yaw, pitch)
        kind = "curve" if side else "straight"
        endpoints = (
            _endpoint(
                object_id, model, family, kind, 0, begin, begin_heading,
                _world_heading(begin_local_heading + 180.0, yaw, pitch),
            ),
            _endpoint(
                object_id, model, family, kind, 1, end, end_heading, end_heading,
            ),
        )
        return RoadObject(
            object_id, model, x, y, z, yaw, pitch, family, kind, endpoints
        )

    match = _GENERATED_PAVED_JUNCTION.fullmatch(path)
    if match:
        degree = int(match.group("degree"))
        width = int(match.group("width")) / 10.0
        family = "asf" if width <= 7.5 else "sil"
        headings = tuple(
            float(value) % 360.0
            for value in match.group("headings").split("_")
        )
        if len(headings) != degree or len(set(headings)) != degree:
            return None
        endpoints = tuple(
            _endpoint(
                object_id,
                model,
                family,
                "junction",
                index,
                _world_point(
                    (
                        math.sin(math.radians(direction)) * _JUNCTION_RADIUS,
                        math.cos(math.radians(direction)) * _JUNCTION_RADIUS,
                    ),
                    origin,
                    yaw,
                    pitch,
                ),
                _world_heading(direction, yaw, pitch),
                _world_heading(direction, yaw, pitch),
            )
            for index, direction in enumerate(headings)
        )
        return RoadObject(
            object_id,
            model,
            x,
            y,
            z,
            yaw,
            pitch,
            family,
            f"junction_generated_{degree}",
            endpoints,
        )

    match = _GRAVEL.fullmatch(path)
    if match:
        length = float(match.group("length"))
        tangent = _world_heading(0.0, yaw, pitch)
        begin = _world_point((0.0, -length * 0.5), origin, yaw, pitch)
        end = _world_point((0.0, length * 0.5), origin, yaw, pitch)
        endpoints = (
            _endpoint(object_id, model, "gravel", "gravel", 0, begin, tangent, _world_heading(180.0, yaw, pitch)),
            _endpoint(object_id, model, "gravel", "gravel", 1, end, tangent, tangent),
        )
        return RoadObject(object_id, model, x, y, z, yaw, pitch, "gravel", "gravel", endpoints)

    match = _GRAVEL_JUNCTION.fullmatch(path)
    if match:
        headings = _gravel_junction_headings(int(match.group("degree")), match.group("variant"))
        endpoints = tuple(
            _endpoint(object_id, model, "gravel", "junction", i,
                      _world_point((math.sin(math.radians(direction)) * _GRAVEL_JUNCTION_RADIUS,
                                    math.cos(math.radians(direction)) * _GRAVEL_JUNCTION_RADIUS), origin, yaw, pitch),
                      _world_heading(direction, yaw, pitch), _world_heading(direction, yaw, pitch))
            for i, direction in enumerate(headings)
        )
        return RoadObject(object_id, model, x, y, z, yaw, pitch, "gravel", "junction_gravel", endpoints)
    return None


def _roads(data: bytes) -> tuple[RoadObject, ...]:
    stream = io.BytesIO(data)
    header = stream.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise ValueError("truncated RVW4 header")
    magic, width, height = _HEADER.unpack(header)
    if magic != b"4WVR" or width <= 0 or height <= 0:
        raise ValueError("Road Inspector requires a valid RVW4 WRP")
    skip = width * height * 4 + _TEXTURE_BYTES
    if len(stream.read(skip)) != skip:
        raise ValueError("truncated RVW4 terrain/texture section")
    result = []
    while True:
        record = stream.read(_OBJECT.size)
        if len(record) != _OBJECT.size:
            raise ValueError("truncated RVW4 object list")
        values = _OBJECT.unpack(record)
        if not values[13].split(b"\0", 1)[0]:
            return tuple(result)
        road = _road(values)
        if road:
            result.append(road)


def _wrp(path: Path) -> tuple[bytes, str]:
    if path.suffix.casefold() == ".wrp":
        return path.read_bytes(), path.name
    if path.suffix.casefold() != ".pbo":
        raise ValueError("input must be a .wrp or uncompressed .pbo")
    entries = tuple(entry for entry in read_pbo(path) if entry.name.casefold().endswith(".wrp"))
    if len(entries) == 1:
        return entries[0].data, entries[0].name
    matches = tuple(entry for entry in entries if Path(entry.name.replace("\\", "/")).stem.casefold() == path.stem.casefold())
    if len(matches) == 1:
        return matches[0].data, matches[0].name
    if not entries:
        raise ValueError("PBO does not contain a WRP")
    raise ValueError("PBO contains multiple WRP entries; inspect the WRP directly")


def _bucket(point: tuple[float, float], size: float) -> tuple[int, int]:
    return math.floor(point[0] / size), math.floor(point[1] / size)


def _clusters(endpoints: Sequence[RoadEndpoint], tolerance: float) -> tuple[tuple[RoadEndpoint, ...], ...]:
    if not endpoints:
        return ()
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, endpoint in enumerate(endpoints):
        buckets.setdefault(_bucket(endpoint.point, tolerance), []).append(i)
    parent = list(range(len(endpoints)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def merge(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i, endpoint in enumerate(endpoints):
        bx, bz = _bucket(endpoint.point, tolerance)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for j in buckets.get((bx + dx, bz + dz), ()):
                    if j > i and endpoints[j].object_id != endpoint.object_id and math.dist(endpoint.point, endpoints[j].point) <= tolerance:
                        merge(i, j)
    grouped: dict[int, list[RoadEndpoint]] = {}
    for i, endpoint in enumerate(endpoints):
        grouped.setdefault(find(i), []).append(endpoint)
    return tuple(tuple(group) for group in grouped.values() if len(group) >= 2)


def _edge_gap(first: RoadEndpoint, second: RoadEndpoint) -> float:
    def edges(endpoint: RoadEndpoint):
        h = math.radians(endpoint.tangent)
        nx, nz = math.cos(h), -math.sin(h)
        x, z = endpoint.point
        w = endpoint.half_width
        return ((x + nx * w, z + nz * w), (x - nx * w, z - nz * w))

    a, b = edges(first), edges(second)
    direct = max(math.dist(a[0], b[0]), math.dist(a[1], b[1]))
    crossed = max(math.dist(a[0], b[1]), math.dist(a[1], b[0]))
    return min(direct, crossed)


def _severity(score: float) -> str:
    return "critical" if score >= 80 else "high" if score >= 55 else "medium" if score >= 30 else "low"


def _issue(first: RoadEndpoint, second: RoadEndpoint, edge_limit: float, tangent_limit: float,
           force_gap: bool = False) -> RoadIssue | None:
    center = math.dist(first.point, second.point)
    tangent = _axis_angle(first.tangent, second.tangent)
    edge = _edge_gap(first, second)
    family = first.family != second.family
    junction = (first.kind == "junction") != (second.kind == "junction")
    if not force_gap and center < 0.05 and edge < edge_limit and tangent < tangent_limit and not family:
        return None
    if junction:
        category = "junction_connector_mismatch"
    elif force_gap:
        category = "connector_gap"
    elif family:
        category = "surface_family_mismatch"
    elif first.kind == "curve" or second.kind == "curve":
        category = "curve_transition"
    elif tangent >= tangent_limit:
        category = "straight_miter"
    else:
        category = "connector_gap"
    score = min(100.0, min(45.0, center * 70.0) + min(35.0, edge * 50.0) + min(30.0, tangent * 4.0) + (20.0 if family else 0.0))
    return RoadIssue(
        "", _severity(score), score, category,
        (first.point[0] + second.point[0]) * 0.5, (first.point[1] + second.point[1]) * 0.5,
        tuple(sorted((first.object_id, second.object_id))), (first.model_path, second.model_path),
        f"{first.model_path} -> {second.model_path}: center gap {center:.3f} m, tangent mismatch {tangent:.2f}°, edge discontinuity {edge:.3f} m.",
        {"center_gap_metres": round(center, 5), "tangent_error_degrees": round(tangent, 5), "edge_gap_metres": round(edge, 5)},
    )


def _direction_count(endpoints: Sequence[RoadEndpoint], tolerance: float = 20.0) -> int:
    directions: list[float] = []
    for endpoint in endpoints:
        if all(_angle(endpoint.outward, direction) > tolerance for direction in directions):
            directions.append(endpoint.outward)
    return len(directions)


def _intersection_issue(endpoints: Sequence[RoadEndpoint]) -> RoadIssue | None:
    unique = {(endpoint.object_id, endpoint.index): endpoint for endpoint in endpoints}
    values = tuple(unique.values())
    if len({endpoint.object_id for endpoint in values}) < 3 or any(endpoint.kind == "junction" for endpoint in values):
        return None
    directions = _direction_count(values)
    if directions < 3:
        return None
    x = sum(endpoint.point[0] for endpoint in values) / len(values)
    z = sum(endpoint.point[1] for endpoint in values) / len(values)
    score = 85.0 if directions >= 4 else 70.0
    return RoadIssue(
        "", _severity(score), score, "intersection_without_junction", x, z,
        tuple(sorted({endpoint.object_id for endpoint in values})),
        tuple(sorted({endpoint.model_path for endpoint in values})),
        f"{directions}-way road intersection has {len(values)} coincident approach endpoints but no stock junction connector.",
        {"approach_endpoints": float(len(values)), "distinct_directions": float(directions)},
    )


def _segment_intersection(a: tuple[float, float], b: tuple[float, float],
                          c: tuple[float, float], d: tuple[float, float]) -> tuple[float, float, float, float] | None:
    den = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if abs(den) <= 1.0e-9:
        return None
    t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / den
    u = -((a[0] - b[0]) * (a[1] - c[1]) - (a[1] - b[1]) * (a[0] - c[0])) / den
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), t, u


def _paved_crossing_issues(roads: Sequence[RoadObject]) -> list[RoadIssue]:
    paved = tuple(road for road in roads if road.kind == "straight" and road.family in {"sil", "asf", "kos"})
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, road in enumerate(paved):
        first, last = road.endpoints[0].point, road.endpoints[-1].point
        for bx in range(math.floor(min(first[0], last[0]) / 25.0), math.floor(max(first[0], last[0]) / 25.0) + 1):
            for bz in range(math.floor(min(first[1], last[1]) / 25.0), math.floor(max(first[1], last[1]) / 25.0) + 1):
                buckets.setdefault((bx, bz), []).append(index)
    issues: list[RoadIssue] = []
    seen: set[tuple[int, int]] = set()
    junctions = tuple(
        (road.x, road.z)
        for road in roads
        if road.kind.startswith("junction_")
    )
    for indices in buckets.values():
        for i, first_index in enumerate(indices):
            for second_index in indices[i + 1:]:
                pair = tuple(sorted((first_index, second_index)))
                if pair in seen:
                    continue
                seen.add(pair)
                first, second = paved[pair[0]], paved[pair[1]]
                hit = _segment_intersection(first.endpoints[0].point, first.endpoints[-1].point,
                                            second.endpoints[0].point, second.endpoints[-1].point)
                if hit is None:
                    continue
                x, z, first_t, second_t = hit
                crossing_angle = _axis_angle(first.endpoints[0].tangent, second.endpoints[0].tangent)
                if crossing_angle < 20.0 or not (0.08 < first_t < 0.92 or 0.08 < second_t < 0.92):
                    continue
                if any(math.dist((x, z), center) <= _JUNCTION_RADIUS + 1.0 for center in junctions):
                    continue
                category = "paved_crossing_without_junction" if 0.08 < first_t < 0.92 and 0.08 < second_t < 0.92 else "paved_t_without_junction"
                score = 90.0 if category.startswith("paved_crossing") else 80.0
                issues.append(RoadIssue(
                    "", _severity(score), score, category, x, z,
                    tuple(sorted((first.object_id, second.object_id))), tuple(sorted((first.model_path, second.model_path))),
                    f"Paved road pieces intersect at {crossing_angle:.1f}° without a junction connector at the crossing.",
                    {"crossing_angle_degrees": round(crossing_angle, 5), "first_position": round(first_t, 5), "second_position": round(second_t, 5)},
                ))
    return issues


def _heading_to(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0


def _junction_issues(roads: Sequence[RoadObject], maximum_gap: float) -> list[RoadIssue]:
    approaches = tuple(endpoint for road in roads if not road.kind.startswith("junction_") for endpoint in road.endpoints)
    issues: list[RoadIssue] = []
    for junction in (road for road in roads if road.kind.startswith("junction_")):
        connector_candidates: set[tuple[int, int]] = set()
        object_ids = {junction.object_id}
        models = {junction.model_path}
        missing = duplicates = 0
        worst_gap = 0.0
        for connector in junction.endpoints:
            candidates = []
            for endpoint in approaches:
                distance = math.dist(connector.point, endpoint.point)
                facing = abs(180.0 - _angle(connector.outward, endpoint.outward))
                if distance <= maximum_gap and facing <= 25.0:
                    candidates.append((distance, endpoint))
            candidates.sort(key=lambda item: item[0])
            for _distance, endpoint in candidates:
                connector_candidates.add((endpoint.object_id, endpoint.index))
                object_ids.add(endpoint.object_id)
                models.add(endpoint.model_path)
            if not candidates:
                missing += 1
                continue
            worst_gap = max(worst_gap, candidates[0][0])
            duplicates += max(0, len(candidates) - 1)

        center = junction.x, junction.z
        extras = []
        for endpoint in approaches:
            key = endpoint.object_id, endpoint.index
            if key in connector_candidates:
                continue
            center_distance = math.dist(endpoint.point, center)
            if center_distance > _JUNCTION_RADIUS + maximum_gap:
                continue
            if center_distance <= 0.05 or _angle(endpoint.outward, _heading_to(endpoint.point, center)) <= 25.0:
                extras.append(endpoint)
                object_ids.add(endpoint.object_id)
                models.add(endpoint.model_path)
        if not missing and not duplicates and not extras:
            continue
        expected = len(junction.endpoints)
        connected = expected - missing
        score = min(100.0, 55.0 + missing * 20.0 + duplicates * 10.0 + len(extras) * 15.0)
        issues.append(RoadIssue(
            "", _severity(score), score, "bad_junction", junction.x, junction.z,
            tuple(sorted(object_ids)), tuple(sorted(models)),
            f"{junction.kind.replace('_', ' ')} has {connected}/{expected} connected stock connectors; {missing} missing, {duplicates} multiply connected, {len(extras)} extra approach(es) near the junction centre.",
            {"expected_connectors": float(expected), "connected_connectors": float(connected),
             "missing_connectors": float(missing), "duplicate_connections": float(duplicates),
             "extra_approaches": float(len(extras)), "worst_connector_gap_metres": round(worst_gap, 5)},
        ))
    return issues


def _nearby(endpoints: Sequence[RoadEndpoint], paired: set[tuple[int, int]], minimum: float, maximum: float,
            edge_limit: float, tangent_limit: float) -> list[RoadIssue]:
    buckets: dict[tuple[int, int], list[RoadEndpoint]] = {}
    for endpoint in endpoints:
        buckets.setdefault(_bucket(endpoint.point, maximum), []).append(endpoint)
    issues, used = [], set()
    for endpoint in endpoints:
        key = endpoint.object_id, endpoint.index
        if key in paired:
            continue
        bx, bz = _bucket(endpoint.point, maximum)
        candidates = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for other in buckets.get((bx + dx, bz + dz), ()):
                    other_key = other.object_id, other.index
                    pair = tuple(sorted((key, other_key)))
                    distance = math.dist(endpoint.point, other.point)
                    facing = abs(180.0 - _angle(endpoint.outward, other.outward))
                    if other.object_id != endpoint.object_id and other_key not in paired and pair not in used and minimum < distance <= maximum and facing <= 25.0:
                        candidates.append((distance, other, pair))
        if candidates:
            _, other, pair = min(candidates, key=lambda item: item[0])
            used.add(pair)
            issue = _issue(endpoint, other, edge_limit, tangent_limit, True)
            if issue:
                issues.append(issue)
    return issues


def _number(issues: Iterable[RoadIssue]) -> tuple[RoadIssue, ...]:
    result, seen = [], set()
    for issue in sorted(issues, key=lambda value: (-value.score, value.category, value.x, value.z)):
        key = issue.category, issue.object_ids, round(issue.x, 2), round(issue.z, 2)
        if key in seen:
            continue
        seen.add(key)
        result.append(RoadIssue(
            f"RI-{len(result)+1:05d}", issue.severity, round(issue.score, 2), issue.category,
            round(issue.x, 4), round(issue.z, 4), issue.object_ids, issue.models, issue.message, issue.metrics,
        ))
    return tuple(result)



def _replacement_seam_endpoints(
    first: RoadObject,
    second: RoadObject,
) -> tuple[RoadEndpoint, RoadEndpoint]:
    return min(
        (
            (a, b)
            for a in first.endpoints
            for b in second.endpoints
        ),
        key=lambda pair: (
            math.dist(pair[0].point, pair[1].point),
            pair[0].index,
            pair[1].index,
        ),
    )


def _replacement_curve_choice(
    start: RoadEndpoint,
    end: RoadEndpoint,
) -> float:
    chord_heading = _heading_to(start.point, end.point)
    length = max(0.01, math.dist(start.point, end.point))
    choices = (0.0,) + tuple(
        value
        for amount in _PAVED_REPLACEMENT_CURVE_BUCKETS
        for value in (float(amount), -float(amount))
    )
    best: tuple[tuple[float, float, float], float] | None = None
    for value in choices:
        side = "r" if value > 0.0 else "l" if value < 0.0 else None
        first_local, last_local = _generated_paved_local_headings(
            length, side, abs(value)
        )
        first_error = _axis_angle(
            chord_heading + first_local,
            start.tangent,
        )
        last_error = _axis_angle(
            chord_heading + last_local,
            end.tangent,
        )
        score = (
            max(first_error, last_error),
            first_error + last_error,
            abs(value),
        )
        if best is None or score < best[0]:
            best = score, value
    return 0.0 if best is None else best[1]


def _replacement_model_path(
    world_name: str,
    width_metres: float,
    length_metres: float,
    curve_degrees: float,
) -> str:
    width_dm = max(10, min(999, int(round(width_metres * 10.0))))
    length_dm = max(5, min(9999, int(round(length_metres * 10.0))))
    suffix = ""
    if abs(curve_degrees) >= 1.5:
        side = "r" if curve_degrees > 0.0 else "l"
        suffix = f"_{side}{int(round(abs(curve_degrees))):02d}"
    return rf"{world_name}\i\paved_w{width_dm:03d}_l{length_dm:04d}{suffix}.p3d"


def _replacement_world_name(
    roads: Sequence[RoadObject],
    wrp_entry: str,
) -> str:
    for road in roads:
        if not _GENERATED_PAVED.fullmatch(_model(road.model_path)):
            continue
        normalized = road.model_path.replace("/", "\\")
        lower = normalized.casefold()
        marker = lower.rfind("\\i\\paved_")
        if marker > 0:
            return normalized[:marker]
    return Path(wrp_entry.replace("\\", "/")).stem



def _packed_paved_model_filenames(input_path: Path) -> frozenset[str]:
    if input_path.suffix.casefold() != ".pbo":
        return frozenset()
    result = set()
    for entry in read_pbo(input_path):
        filename = entry.name.replace("/", "\\").rsplit("\\", 1)[-1]
        if re.fullmatch(
            r"paved_w\d{3}_l\d{4}(?:_[lr]\d{2})?\.p3d",
            filename,
            re.IGNORECASE,
        ):
            result.add(filename.casefold())
    return frozenset(result)



def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    denominator = dx * dx + dz * dz
    if denominator <= 1.0e-12:
        return math.dist(point, start)
    fraction = (
        (point[0] - start[0]) * dx
        + (point[1] - start[1]) * dz
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    nearest = (
        start[0] + dx * fraction,
        start[1] + dz * fraction,
    )
    return math.dist(point, nearest)


def _point_polyline_distance(
    point: tuple[float, float],
    points: Sequence[tuple[float, float]],
) -> float:
    if len(points) < 2:
        return math.dist(point, points[0]) if points else math.inf
    return min(
        _point_segment_distance(point, start, end)
        for start, end in zip(points, points[1:])
    )


def _bidirectional_path_deviation(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> float:
    return max(
        max((_point_polyline_distance(point, second) for point in first), default=0.0),
        max((_point_polyline_distance(point, first) for point in second), default=0.0),
    )


def _stock_arc_step(
    point: tuple[float, float],
    heading: float,
    turn_sign: int,
    radius: float,
    degrees: float = 10.0,
) -> tuple[tuple[float, float], float]:
    radians = math.radians(heading)
    direction = math.sin(radians), math.cos(radians)
    right = direction[1], -direction[0]
    angle = math.radians(degrees)
    side = turn_sign * radius * (1.0 - math.cos(angle))
    forward = radius * math.sin(angle)
    return (
        point[0] + right[0] * side + direction[0] * forward,
        point[1] + right[1] * side + direction[1] * forward,
    ), (heading + turn_sign * degrees) % 360.0


def _component_reference_path(
    component: Sequence[RoadObject],
    component_issues: Sequence[RoadIssue],
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[RoadEndpoint, RoadEndpoint],
] | None:
    ids = {road.object_id for road in component}
    adjacency: dict[int, set[int]] = {object_id: set() for object_id in ids}
    seam_midpoints: dict[frozenset[int], tuple[float, float]] = {}
    internal: set[tuple[int, int]] = set()
    for issue in component_issues:
        if len(issue.object_ids) != 2:
            continue
        first_id, second_id = issue.object_ids
        if first_id not in ids or second_id not in ids:
            continue
        first = next(road for road in component if road.object_id == first_id)
        second = next(road for road in component if road.object_id == second_id)
        first_endpoint, second_endpoint = _replacement_seam_endpoints(first, second)
        adjacency[first_id].add(second_id)
        adjacency[second_id].add(first_id)
        seam_midpoints[frozenset((first_id, second_id))] = (
            (first_endpoint.point[0] + second_endpoint.point[0]) * 0.5,
            (first_endpoint.point[1] + second_endpoint.point[1]) * 0.5,
        )
        internal.add((first_endpoint.object_id, first_endpoint.index))
        internal.add((second_endpoint.object_id, second_endpoint.index))

    if any(len(neighbours) > 2 for neighbours in adjacency.values()):
        return None
    terminal_ids = sorted(
        object_id for object_id, neighbours in adjacency.items()
        if len(neighbours) == 1
    )
    if len(terminal_ids) != 2:
        return None

    boundary_by_id = {
        road.object_id: tuple(
            endpoint
            for endpoint in road.endpoints
            if (endpoint.object_id, endpoint.index) not in internal
        )
        for road in component
    }
    if any(len(boundary_by_id[object_id]) != 1 for object_id in terminal_ids):
        return None

    order = [terminal_ids[0]]
    previous = None
    current = terminal_ids[0]
    while True:
        following = tuple(
            value for value in adjacency[current] if value != previous
        )
        if not following:
            break
        if len(following) != 1:
            return None
        next_id = following[0]
        order.append(next_id)
        previous, current = current, next_id
        if len(order) > len(ids):
            return None
    if set(order) != ids:
        return None

    start = boundary_by_id[order[0]][0]
    end = boundary_by_id[order[-1]][0]
    points = [start.point]
    for first_id, second_id in zip(order, order[1:]):
        midpoint = seam_midpoints.get(frozenset((first_id, second_id)))
        if midpoint is None:
            return None
        points.append(midpoint)
    points.append(end.point)
    return tuple(points), (start, end)


def _stock_repair_angle_limit(half_width: float) -> float:
    if half_width <= 1.0e-9:
        return 0.0
    ratio = min(
        1.0,
        DEFAULT_MINIMUM_EDGE_GAP_METRES / (2.0 * half_width),
    )
    return math.degrees(2.0 * math.asin(ratio))


def _stock_repair_samples(
    start: RoadEndpoint,
    end: RoadEndpoint,
    choice: tuple[int, int, int, int, int, int, int],
) -> tuple[tuple[float, float], ...]:
    (
        turn_sign,
        first_turns,
        first_radius,
        middle_units,
        counter_turns,
        counter_radius,
        _merge_nominal,
    ) = choice
    point = start.point
    heading = (start.outward + 180.0) % 360.0
    points = [point]

    def sampled_turn(
        current: tuple[float, float],
        current_heading: float,
        sign: int,
        radius: int,
    ) -> tuple[tuple[float, float], float]:
        for _index in range(4):
            current, current_heading = _stock_arc_step(
                current, current_heading, sign, radius, 2.5
            )
            points.append(current)
        return current, current_heading

    for _index in range(first_turns):
        point, heading = sampled_turn(
            point, heading, turn_sign, first_radius
        )
    for _index in range(middle_units):
        radians = math.radians(heading)
        point = (
            point[0] + math.sin(radians) * 6.25,
            point[1] + math.cos(radians) * 6.25,
        )
        points.append(point)
    for _index in range(counter_turns):
        point, heading = sampled_turn(
            point, heading, -turn_sign, counter_radius
        )
    points.append(end.point)
    return tuple(points)


def _stock_repair_models(
    family: str,
    choice: tuple[int, int, int, int, int, int, int],
) -> tuple[str, ...]:
    (
        _turn_sign,
        first_turns,
        first_radius,
        middle_units,
        counter_turns,
        counter_radius,
        merge_nominal,
    ) = choice
    models = [
        rf"o\road\{family}10 {first_radius}.p3d"
        for _index in range(first_turns)
    ]
    models.extend(
        rf"o\road\{family}6.p3d"
        for _index in range(middle_units)
    )
    models.extend(
        rf"o\road\{family}10 {counter_radius}.p3d"
        for _index in range(counter_turns)
    )
    models.append(rf"o\road\{family}{merge_nominal}.p3d")
    return tuple(models)


def _stock_repair_choice(
    family: str,
    reference: Sequence[tuple[float, float]],
    start: RoadEndpoint,
    end: RoadEndpoint,
) -> tuple[
    tuple[int, int, int, int, int, int, int],
    float,
    float,
    float,
] | None:
    if family not in _STOCK_REPAIR_FAMILIES:
        return None

    start_heading = (start.outward + 180.0) % 360.0
    target_heading = end.outward % 360.0
    preferred_sign = (
        1
        if ((target_heading - start_heading + 180.0) % 360.0 - 180.0) >= 0.0
        else -1
    )
    angle_limit = _stock_repair_angle_limit(
        max(start.half_width, end.half_width)
    )
    candidates = []

    for turn_sign in (preferred_sign, -preferred_sign):
        for first_turns in range(0, 7):
            first_radii = (
                _STOCK_REPAIR_RADII_METRES
                if first_turns else (_STOCK_REPAIR_RADII_METRES[0],)
            )
            for counter_turns in range(0, 5):
                if first_turns == 0 and counter_turns > 0:
                    continue
                counter_radii = (
                    _STOCK_REPAIR_RADII_METRES
                    if counter_turns else (_STOCK_REPAIR_RADII_METRES[0],)
                )
                for first_radius in first_radii:
                    first_point = start.point
                    first_heading = start_heading
                    for _index in range(first_turns):
                        first_point, first_heading = _stock_arc_step(
                            first_point,
                            first_heading,
                            turn_sign,
                            first_radius,
                        )
                    for counter_radius in counter_radii:
                        for middle_units in range(0, 8):
                            point = first_point
                            heading = first_heading
                            if middle_units:
                                radians = math.radians(heading)
                                point = (
                                    point[0]
                                    + math.sin(radians) * 6.25 * middle_units,
                                    point[1]
                                    + math.cos(radians) * 6.25 * middle_units,
                                )
                            for _index in range(counter_turns):
                                point, heading = _stock_arc_step(
                                    point,
                                    heading,
                                    -turn_sign,
                                    counter_radius,
                                )

                            dx = end.point[0] - point[0]
                            dz = end.point[1] - point[1]
                            distance = math.hypot(dx, dz)
                            if distance <= 0.05:
                                continue
                            merge_heading = math.degrees(
                                math.atan2(dx, dz)
                            ) % 360.0
                            in_error = _angle(heading, merge_heading)
                            out_error = _angle(merge_heading, target_heading)
                            if max(in_error, out_error) > angle_limit:
                                continue

                            for nominal, length in _STOCK_REPAIR_STRAIGHT_LENGTHS.items():
                                length_error = abs(distance - length)
                                if (
                                    length_error
                                    > _STOCK_REPAIR_POSITION_TOLERANCE_METRES
                                ):
                                    continue
                                choice = (
                                    turn_sign,
                                    first_turns,
                                    first_radius,
                                    middle_units,
                                    counter_turns,
                                    counter_radius,
                                    nominal,
                                )
                                samples = _stock_repair_samples(
                                    start, end, choice
                                )
                                deviation = _bidirectional_path_deviation(
                                    samples, reference
                                )
                                if (
                                    deviation
                                    > _STOCK_REPAIR_MAXIMUM_PATH_DEVIATION_METRES
                                ):
                                    continue
                                piece_count = (
                                    first_turns
                                    + middle_units
                                    + counter_turns
                                    + 1
                                )
                                score = (
                                    deviation,
                                    length_error,
                                    max(in_error, out_error),
                                    piece_count,
                                    first_radius,
                                    counter_radius,
                                )
                                candidates.append((
                                    score,
                                    choice,
                                    deviation,
                                    length_error,
                                    max(in_error, out_error),
                                ))

    if not candidates:
        return None
    _score, choice, deviation, length_error, angle_error = min(
        candidates, key=lambda value: value[0]
    )
    return choice, deviation, length_error, angle_error


def _is_generated_paved_road(road: RoadObject) -> bool:
    return _GENERATED_PAVED.fullmatch(_model(road.model_path)) is not None


def _compatible_paved_pair(first: RoadObject, second: RoadObject) -> bool:
    if first.road_type != "paved" or second.road_type != "paved":
        return False
    if first.kind.startswith("junction_") or second.kind.startswith("junction_"):
        return False
    first_width = max((endpoint.half_width for endpoint in first.endpoints), default=0.0)
    second_width = max((endpoint.half_width for endpoint in second.endpoints), default=0.0)
    if abs(first_width - second_width) > 0.05:
        return False
    first_stock = None if _is_generated_paved_road(first) else first.family
    second_stock = None if _is_generated_paved_road(second) else second.family
    return first_stock is None or second_stock is None or first_stock == second_stock


def _component_stock_family(component: Sequence[RoadObject]) -> str | None:
    families = {
        road.family
        for road in component
        if not _is_generated_paved_road(road)
    }
    return next(iter(families)) if len(families) == 1 else None


def _paved_replacement_plans(
    roads: Sequence[RoadObject],
    issues: Sequence[RoadIssue],
    wrp_entry: str,
    packed_paved_models: frozenset[str] = frozenset(),
) -> tuple[tuple[PavedStockRepairPlan, ...], tuple[PavedReplacementPlan, ...]]:
    road_by_id = {road.object_id: road for road in roads}
    eligible: list[RoadIssue] = []
    for issue in issues:
        if issue.category not in _PAVED_REPLACEMENT_CATEGORIES:
            continue
        # Purely axial straight-road spacing failures should be fixed by stock
        # refitting. Procedural pavement is reserved for seams whose road-edge
        # orientation cannot be made continuous with the current stock pair.
        if (
            float(issue.metrics.get("tangent_error_degrees", 0.0))
            <= _PAVED_REPLACEMENT_MINIMUM_TANGENT_ERROR_DEGREES
        ):
            continue
        if len(issue.object_ids) != 2:
            continue
        pair = tuple(road_by_id.get(value) for value in issue.object_ids)
        if any(road is None for road in pair):
            continue
        first, second = pair
        if not _compatible_paved_pair(first, second):
            continue
        eligible.append(issue)
    if not eligible:
        return (), ()

    parent: dict[int, int] = {}
    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def merge(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for issue in eligible:
        first, second = issue.object_ids
        merge(first, second)

    grouped_ids: dict[int, set[int]] = {}
    grouped_issues: dict[int, list[RoadIssue]] = {}
    for issue in eligible:
        root = find(issue.object_ids[0])
        grouped_ids.setdefault(root, set()).update(issue.object_ids)
        grouped_issues.setdefault(root, []).append(issue)

    world_name = _replacement_world_name(roads, wrp_entry)
    existing_models = Counter(
        road.model_path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
        for road in roads
        if _GENERATED_PAVED.fullmatch(_model(road.model_path))
    )
    existing_models.update(packed_paved_models)
    stock_repairs: list[PavedStockRepairPlan] = []
    plans: list[PavedReplacementPlan] = []
    for root in sorted(grouped_ids, key=lambda key: min(grouped_ids[key])):
        object_ids = tuple(sorted(grouped_ids[root]))
        component = tuple(road_by_id[value] for value in object_ids)
        internal: set[tuple[int, int]] = set()
        for issue in grouped_issues[root]:
            first = road_by_id[issue.object_ids[0]]
            second = road_by_id[issue.object_ids[1]]
            a, b = _replacement_seam_endpoints(first, second)
            internal.add((a.object_id, a.index))
            internal.add((b.object_id, b.index))
        boundaries = tuple(
            endpoint
            for road in component
            for endpoint in road.endpoints
            if (endpoint.object_id, endpoint.index) not in internal
        )
        if len(boundaries) != 2:
            continue
        reference = _component_reference_path(component, grouped_issues[root])
        if reference is None:
            continue
        reference_points, (start, end) = reference
        component_issues = tuple(grouped_issues[root])
        family = _component_stock_family(component)
        stock_choice = (
            _stock_repair_choice(
                family,
                reference_points,
                start,
                end,
            )
            if family is not None
            else None
        )
        if stock_choice is not None:
            choice, deviation, length_error, angle_error = stock_choice
            stock_repairs.append(PavedStockRepairPlan(
                plan_id=f"SR-{len(stock_repairs)+1:05d}",
                family=family,
                replace_object_ids=object_ids,
                source_models=tuple(sorted({road.model_path for road in component})),
                issue_ids=tuple(sorted(issue.issue_id for issue in component_issues)),
                stock_models=_stock_repair_models(family, choice),
                start=(round(start.point[0], 5), round(start.point[1], 5)),
                end=(round(end.point[0], 5), round(end.point[1], 5)),
                start_heading_degrees=round((start.outward + 180.0) % 360.0, 5),
                end_heading_degrees=round(end.outward % 360.0, 5),
                turn_sign=choice[0],
                first_turns=choice[1],
                first_radius=choice[2],
                middle_units=choice[3],
                counter_turns=choice[4],
                counter_radius=choice[5],
                merge_nominal=choice[6],
                maximum_path_deviation_metres=round(deviation, 5),
                final_length_error_metres=round(length_error, 5),
                maximum_join_angle_error_degrees=round(angle_error, 5),
            ))
            continue
        length = math.dist(start.point, end.point)
        if length <= 0.05:
            continue
        half_widths = tuple(endpoint.half_width for endpoint in boundaries)
        width = 2.0 * max(half_widths)
        curve = _replacement_curve_choice(start, end)
        model_path = _replacement_model_path(
            world_name, width, length, curve
        )
        model_filename = (
            model_path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
        )
        action = "reuse" if existing_models[model_filename] else "generate"
        plans.append(PavedReplacementPlan(
            plan_id=f"RP-{len(plans)+1:05d}",
            action=action,
            model_path=model_path,
            replace_object_ids=object_ids,
            source_models=tuple(sorted({road.model_path for road in component})),
            issue_ids=tuple(sorted(issue.issue_id for issue in component_issues)),
            start=(round(start.point[0], 5), round(start.point[1], 5)),
            end=(round(end.point[0], 5), round(end.point[1], 5)),
            width_metres=round(width, 3),
            length_metres=round(length, 3),
            curve_degrees=curve,
            maximum_edge_gap_metres=max(
                float(issue.metrics.get("edge_gap_metres", 0.0))
                for issue in component_issues
            ),
            maximum_tangent_error_degrees=max(
                float(issue.metrics.get("tangent_error_degrees", 0.0))
                for issue in component_issues
            ),
        ))
        # The procedural infrastructure library writes one canonical P3D per
        # filename. Treat later regions requesting the same variant as reuse
        # within this inspection plan as well as reuse of assets already packed.
        existing_models[model_filename] += 1
    return tuple(stock_repairs), tuple(plans)


def _inspect_roads(
    roads: Sequence[RoadObject],
    *,
    input_path: str,
    wrp_entry: str,
    packed_paved_models: frozenset[str] = frozenset(),
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE_METRES,
    nearby_gap: float = DEFAULT_NEARBY_GAP_METRES,
    minimum_edge_gap: float = DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    topology_checks: bool = True,
) -> InspectionResult:
    checked_roads = tuple(road for road in roads if road.family != "gravel")
    endpoints = tuple(endpoint for road in checked_roads for endpoint in road.endpoints)
    issues: list[RoadIssue] = []
    paired: set[tuple[int, int]] = set()
    for cluster in _clusters(endpoints, endpoint_tolerance):
        unique = {(endpoint.object_id, endpoint.index): endpoint for endpoint in cluster}
        if len(unique) == 2:
            first, second = tuple(unique.values())
            paired.update(unique)
            issue = _issue(first, second, minimum_edge_gap, minimum_tangent_error)
            if issue:
                issues.append(issue)
            continue
        paired.update(unique)
        issue = _intersection_issue(tuple(unique.values()))
        if issue:
            issues.append(issue)
    issues.extend(_nearby(
        endpoints, paired, endpoint_tolerance, nearby_gap,
        minimum_edge_gap, minimum_tangent_error,
    ))
    if topology_checks:
        issues.extend(_junction_issues(checked_roads, nearby_gap))
        issues.extend(_paved_crossing_issues(checked_roads))
    numbered = _number(issues)
    stock_repairs, replacements = _paved_replacement_plans(
        roads, numbered, wrp_entry, packed_paved_models,
    )
    return InspectionResult(
        input_path, wrp_entry, tuple(roads), numbered, stock_repairs, replacements,
    )


def inspect_road_objects(
    objects: Sequence[object],
    *,
    world_name: str = "world",
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE_METRES,
    nearby_gap: float = DEFAULT_NEARBY_GAP_METRES,
    minimum_edge_gap: float = DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    topology_checks: bool = True,
) -> InspectionResult:
    """Inspect final in-memory WorldObject-style transforms without a WRP round-trip.

    Values need object_id, model_path and matrix_4x3(). Unknown model families are
    ignored exactly as they are when the inspector reads a built RVW4.
    """
    roads: list[RoadObject] = []
    for obj in objects:
        model = str(getattr(obj, "model_path", ""))
        if not model:
            continue
        matrix = tuple(float(value) for value in obj.matrix_4x3())
        if len(matrix) != 12:
            raise ValueError("road inspector object matrix must contain 12 floats")
        try:
            encoded = model.encode("ascii")
        except UnicodeEncodeError:
            continue
        values = (*matrix, int(getattr(obj, "object_id")), encoded + b"\0")
        road = _road(values)
        if road is not None:
            roads.append(road)
    safe_name = str(world_name).strip() or "world"
    return _inspect_roads(
        tuple(roads),
        input_path="<memory>",
        wrp_entry=f"{safe_name}.wrp",
        endpoint_tolerance=endpoint_tolerance,
        nearby_gap=nearby_gap,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        topology_checks=topology_checks,
    )


def inspect_road_geometry(input_path: Path, *, endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE_METRES,
                          nearby_gap: float = DEFAULT_NEARBY_GAP_METRES,
                          minimum_edge_gap: float = DEFAULT_MINIMUM_EDGE_GAP_METRES,
                          minimum_tangent_error: float = DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES) -> InspectionResult:
    data, wrp_entry = _wrp(Path(input_path))
    roads = _roads(data)
    return _inspect_roads(
        roads,
        input_path=str(Path(input_path)),
        wrp_entry=wrp_entry,
        packed_paved_models=_packed_paved_model_filenames(Path(input_path)),
        endpoint_tolerance=endpoint_tolerance,
        nearby_gap=nearby_gap,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
    )

def _summary(result: InspectionResult) -> dict[str, object]:
    return {
        "input": result.input_path, "wrp_entry": result.wrp_entry, "road_objects": result.road_object_count,
        "road_type_counts": dict(Counter(road.road_type for road in result.road_objects)),
        "issue_count": len(result.issues), "severity_counts": dict(Counter(i.severity for i in result.issues)),
        "category_counts": dict(Counter(i.category for i in result.issues)),
        "paved_stock_repair_count": len(result.paved_stock_repairs),
        "paved_replacement_count": len(result.paved_replacements),
        "paved_replacement_actions": dict(Counter(plan.action for plan in result.paved_replacements)),
    }


def write_inspection_report(result: InspectionResult, output_dir: Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"issues_json": output / "issues.json", "issues_csv": output / "issues.csv",
             "summary_json": output / "summary.json", "coordinate_csv": output / "ingame-coordinates.csv",
             "stock_repairs_json": output / "paved-stock-repairs.json",
             "stock_repairs_csv": output / "paved-stock-repairs.csv",
             "replacements_json": output / "paved-replacements.json",
             "replacements_csv": output / "paved-replacements.csv",
             "html": output / "report.html"}
    paths["issues_json"].write_text(json.dumps([asdict(i) for i in result.issues], indent=2) + "\n", encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(_summary(result), indent=2) + "\n", encoding="utf-8")

    paths["stock_repairs_json"].write_text(
        json.dumps(
            [asdict(plan) for plan in result.paved_stock_repairs],
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with paths["stock_repairs_csv"].open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "plan_id", "family", "replace_object_ids", "source_models", "issue_ids",
            "stock_models", "start_x", "start_z", "end_x", "end_z",
            "start_heading_degrees", "end_heading_degrees",
            "maximum_path_deviation_metres", "final_length_error_metres",
            "maximum_join_angle_error_degrees",
        ))
        for plan in result.paved_stock_repairs:
            writer.writerow((
                plan.plan_id, plan.family,
                ";".join(map(str, plan.replace_object_ids)),
                ";".join(plan.source_models),
                ";".join(plan.issue_ids),
                ";".join(plan.stock_models),
                plan.start[0], plan.start[1], plan.end[0], plan.end[1],
                plan.start_heading_degrees, plan.end_heading_degrees,
                plan.maximum_path_deviation_metres,
                plan.final_length_error_metres,
                plan.maximum_join_angle_error_degrees,
            ))
    paths["replacements_json"].write_text(
        json.dumps([asdict(plan) for plan in result.paved_replacements], indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["replacements_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "plan_id", "action", "model_path", "replace_object_ids",
            "source_models", "issue_ids", "start_x", "start_z", "end_x", "end_z",
            "width_metres", "length_metres", "curve_degrees",
            "maximum_edge_gap_metres", "maximum_tangent_error_degrees",
        ))
        for plan in result.paved_replacements:
            writer.writerow((
                plan.plan_id, plan.action, plan.model_path,
                ";".join(map(str, plan.replace_object_ids)),
                ";".join(plan.source_models), ";".join(plan.issue_ids),
                plan.start[0], plan.start[1], plan.end[0], plan.end[1],
                plan.width_metres, plan.length_metres, plan.curve_degrees,
                plan.maximum_edge_gap_metres,
                plan.maximum_tangent_error_degrees,
            ))
    with paths["issues_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("issue_id", "severity", "score", "category", "x", "z", "object_ids", "models", "message"))
        for issue in result.issues:
            writer.writerow((issue.issue_id, issue.severity, issue.score, issue.category, issue.x, issue.z,
                             ";".join(map(str, issue.object_ids)), ";".join(issue.models), issue.message))
    related: dict[int, list[str]] = {}
    for issue in result.issues:
        for object_id in issue.object_ids:
            related.setdefault(object_id, []).append(issue.issue_id)
    with paths["coordinate_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("kind", "id", "x", "z", "model", "issues", "teleport"))
        for issue in result.issues:
            writer.writerow(("issue", issue.issue_id, issue.x, issue.z, "", issue.issue_id,
                             f"player setPos [{issue.x:.3f}, {issue.z:.3f}, 0]"))
        for road in result.road_objects:
            writer.writerow(("road", road.object_id, road.x, road.z, road.model_path,
                             ";".join(related.get(road.object_id, ())), f"player setPos [{road.x:.3f}, {road.z:.3f}, 0]"))
    rows = "".join(
        f"<tr><td>{html.escape(i.issue_id)}</td><td>{html.escape(i.severity)}</td><td>{i.score:.1f}</td>"
        f"<td>{html.escape(i.category)}</td><td>{i.x:.2f}, {i.z:.2f}</td><td>{html.escape(i.message)}</td></tr>"
        for i in result.issues
    ) or '<tr><td colspan="6">No road issues found.</td></tr>'
    stock_rows = "".join(
        f"<tr><td>{html.escape(plan.plan_id)}</td>"
        f"<td>{html.escape(', '.join(plan.stock_models))}</td>"
        f"<td>{html.escape(', '.join(map(str, plan.replace_object_ids)))}</td>"
        f"<td>{plan.maximum_path_deviation_metres:.2f} m</td></tr>"
        for plan in result.paved_stock_repairs
    ) or '<tr><td colspan="4">No stock paved repair sequence found.</td></tr>'
    replacement_rows = "".join(
        f"<tr><td>{html.escape(plan.plan_id)}</td><td>{html.escape(plan.action)}</td>"
        f"<td>{html.escape(plan.model_path)}</td>"
        f"<td>{html.escape(', '.join(map(str, plan.replace_object_ids)))}</td>"
        f"<td>{plan.width_metres:.2f} × {plan.length_metres:.2f} m</td>"
        f"<td>{plan.curve_degrees:+.0f}°</td></tr>"
        for plan in result.paved_replacements
    ) or '<tr><td colspan="6">No paved seam replacements required.</td></tr>'
    paths["html"].write_text(
        f'<!doctype html><meta charset="utf-8"><title>Road Inspector</title><style>body{{font:14px system-ui;margin:24px}}'
        f'table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #aaa;padding:6px;text-align:left}}</style>'
        f'<h1>Road Inspector</h1><p>Read-only RVW4 road audit. {result.road_object_count} road objects, '
        f'{len(result.issues)} issues, {len(result.paved_stock_repairs)} stock repair plan(s), '
        f'{len(result.paved_replacements)} paved replacement plan(s).</p>'
        f'<h2>Issues</h2><table><tr><th>ID</th><th>Severity</th><th>Score</th><th>Category</th>'
        f'<th>X/Z</th><th>Details</th></tr>{rows}</table>'
        f'<h2>Stock paved repair sequences</h2><table><tr><th>ID</th><th>Stock models</th>'
        f'<th>Replace object IDs</th><th>Max path deviation</th></tr>{stock_rows}</table>'
        f'<h2>Paved seam replacements</h2><table><tr><th>ID</th><th>Action</th><th>Model</th>'
        f'<th>Replace object IDs</th><th>Size</th><th>Curve</th></tr>{replacement_rows}</table>',
        encoding="utf-8")
    return paths


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cwr-road-inspector", description="Read-only inspection of road seams and junctions in a CWA RVW4 WRP/PBO.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("road-inspector"))
    parser.add_argument("--endpoint-tolerance", type=_positive, default=DEFAULT_ENDPOINT_TOLERANCE_METRES)
    parser.add_argument("--nearby-gap", type=_positive, default=DEFAULT_NEARBY_GAP_METRES)
    parser.add_argument("--minimum-edge-gap", type=_positive, default=DEFAULT_MINIMUM_EDGE_GAP_METRES)
    parser.add_argument("--minimum-tangent-error", type=_positive, default=DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES)
    args = parser.parse_args(argv)
    result = inspect_road_geometry(args.input, endpoint_tolerance=args.endpoint_tolerance, nearby_gap=args.nearby_gap,
                                   minimum_edge_gap=args.minimum_edge_gap, minimum_tangent_error=args.minimum_tangent_error)
    report = write_inspection_report(result, args.output)
    counts = Counter(issue.severity for issue in result.issues)
    print(
        f"Road Inspector: {result.road_object_count:,} road objects, "
        f"{len(result.issues):,} issues ({counts.get('critical', 0)} critical, "
        f"{counts.get('high', 0)} high), "
        f"{len(result.paved_stock_repairs):,} stock repair plan(s), "
        f"{len(result.paved_replacements):,} paved replacement plan(s)."
    )
    print(f"HTML report: {report['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
