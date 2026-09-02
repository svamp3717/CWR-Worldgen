# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only post-build inspector for stock CWA road geometry."""
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
_WIDTHS = {"sil": 4.55, "kos": 4.55, "asf": 3.50, "ces": 1.75}
_CURVE_ANGLE = 10.0
_JUNCTION_RADIUS = 6.25
_STRAIGHT = re.compile(r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)(?P<length>25|12|6)\.p3d$", re.I)
_CURVE = re.compile(r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)10 (?P<radius>25|50|75|100)\.p3d$", re.I)
_T = re.compile(r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?P<branch>sil|ces|asf|kos)_t\.p3d$", re.I)
_X = re.compile(r"^(?:.*[\\/])kr_new_silxsil\.p3d$", re.I)

DEFAULT_ENDPOINT_TOLERANCE_METRES = 0.20
DEFAULT_NEARBY_GAP_METRES = 1.50
DEFAULT_MINIMUM_EDGE_GAP_METRES = 0.08
DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES = 0.75


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
class InspectionResult:
    input_path: str
    wrp_entry: str
    road_objects: tuple[RoadObject, ...]
    issues: tuple[RoadIssue, ...]

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
            ((0.0, _JUNCTION_RADIUS), 0.0),
            ((0.0, -_JUNCTION_RADIUS), 180.0),
            ((_JUNCTION_RADIUS, 0.0), 90.0),
            ((-_JUNCTION_RADIUS, 0.0), 270.0),
        )
        endpoints = tuple(
            _endpoint(object_id, model, "sil", "junction", i, _world_point(local, origin, yaw, pitch),
                      _world_heading(direction, yaw, pitch), _world_heading(direction, yaw, pitch))
            for i, (local, direction) in enumerate(definitions)
        )
        return RoadObject(object_id, model, x, y, z, yaw, pitch, "sil", "junction_x", endpoints)
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
        (first.point[0] + second.point[0]) * 0.5,
        (first.point[1] + second.point[1]) * 0.5,
        tuple(sorted((first.object_id, second.object_id))),
        (first.model_path, second.model_path),
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
            f"{junction.kind.replace('_', ' ')} has {connected}/{expected} connected stock connectors"
            f"; {missing} missing, {duplicates} multiply connected, {len(extras)} extra approach(es) near the junction centre.",
            {
                "expected_connectors": float(expected),
                "connected_connectors": float(connected),
                "missing_connectors": float(missing),
                "duplicate_connections": float(duplicates),
                "extra_approaches": float(len(extras)),
                "worst_connector_gap_metres": round(worst_gap, 5),
            },
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


def inspect_road_geometry(input_path: Path, *, endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE_METRES,
                          nearby_gap: float = DEFAULT_NEARBY_GAP_METRES,
                          minimum_edge_gap: float = DEFAULT_MINIMUM_EDGE_GAP_METRES,
                          minimum_tangent_error: float = DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES) -> InspectionResult:
    data, wrp_entry = _wrp(Path(input_path))
    roads = _roads(data)
    endpoints = tuple(endpoint for road in roads for endpoint in road.endpoints)
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
    issues.extend(_nearby(endpoints, paired, endpoint_tolerance, nearby_gap, minimum_edge_gap, minimum_tangent_error))
    issues.extend(_junction_issues(roads, nearby_gap))
    return InspectionResult(str(Path(input_path)), wrp_entry, roads, _number(issues))


def _summary(result: InspectionResult) -> dict[str, object]:
    return {
        "input": result.input_path,
        "wrp_entry": result.wrp_entry,
        "road_objects": result.road_object_count,
        "issue_count": len(result.issues),
        "severity_counts": dict(Counter(i.severity for i in result.issues)),
        "category_counts": dict(Counter(i.category for i in result.issues)),
    }


def write_inspection_report(result: InspectionResult, output_dir: Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "issues_json": output / "issues.json",
        "issues_csv": output / "issues.csv",
        "summary_json": output / "summary.json",
        "coordinate_csv": output / "ingame-coordinates.csv",
        "html": output / "report.html",
    }
    paths["issues_json"].write_text(json.dumps([asdict(i) for i in result.issues], indent=2) + "\n", encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(_summary(result), indent=2) + "\n", encoding="utf-8")
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
    paths["html"].write_text(
        f'<!doctype html><meta charset="utf-8"><title>Road Inspector</title><style>body{{font:14px system-ui;margin:24px}}'
        f'table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #aaa;padding:6px;text-align:left}}</style>'
        f'<h1>Road Inspector</h1><p>Read-only RVW4 stock-road audit. {result.road_object_count} road objects, '
        f'{len(result.issues)} issues.</p><table><tr><th>ID</th><th>Severity</th><th>Score</th><th>Category</th>'
        f'<th>X/Z</th><th>Details</th></tr>{rows}</table>',
        encoding="utf-8",
    )
    return paths


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cwr-road-inspector", description="Read-only inspection of stock-road seams in a CWA RVW4 WRP/PBO.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("road-inspector"))
    parser.add_argument("--endpoint-tolerance", type=_positive, default=DEFAULT_ENDPOINT_TOLERANCE_METRES)
    parser.add_argument("--nearby-gap", type=_positive, default=DEFAULT_NEARBY_GAP_METRES)
    parser.add_argument("--minimum-edge-gap", type=_positive, default=DEFAULT_MINIMUM_EDGE_GAP_METRES)
    parser.add_argument("--minimum-tangent-error", type=_positive, default=DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES)
    args = parser.parse_args(argv)
    result = inspect_road_geometry(
        args.input,
        endpoint_tolerance=args.endpoint_tolerance,
        nearby_gap=args.nearby_gap,
        minimum_edge_gap=args.minimum_edge_gap,
        minimum_tangent_error=args.minimum_tangent_error,
    )
    report = write_inspection_report(result, args.output)
    counts = Counter(issue.severity for issue in result.issues)
    print(f"Road Inspector: {result.road_object_count:,} road objects, {len(result.issues):,} issues ({counts.get('critical', 0)} critical, {counts.get('high', 0)} high).")
    print(f"HTML report: {report['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
