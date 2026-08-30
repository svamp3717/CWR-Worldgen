# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect a real generated WRP/PBO for visible stock-road geometry defects.

This module is deliberately read-only.  It examines the geometry that CWA will
actually load rather than a synthetic RoadLab scene, reconstructs the measured
stock-road connectors, and reports likely wedges, clipping seams and intersection
mismatches.  Reports are written as JSON, CSV and a self-contained interactive
HTML file so a bad coordinate can be inspected without launching the game.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import html
import io
import json
import math
import re
import struct
from typing import Iterable, Sequence

from .pbo import read_pbo
from . import stock_road_model_geometry as _geometry


_RVW4_HEADER = struct.Struct("<4sii")
_RVW4_OBJECT = struct.Struct("<12fi76s")
_TEXTURE_TABLE_BYTES = 512 * 32

_T_JUNCTION = re.compile(
    r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?P<branch>sil|ces|asf|kos)_t\.p3d$",
    re.IGNORECASE,
)
_X_JUNCTION = re.compile(
    r"^(?:.*[\\/])kr_new_silxsil\.p3d$",
    re.IGNORECASE,
)
_PAVED_FILL = re.compile(
    r"^(?:.*[\\/])paved_fill\.p3d$",
    re.IGNORECASE,
)
_PAVED_FAMILIES = {"sil", "asf", "kos"}
_STOCK_FAMILIES = _PAVED_FAMILIES | {"ces"}

DEFAULT_ENDPOINT_TOLERANCE_METRES = 0.20
DEFAULT_MINIMUM_EDGE_GAP_METRES = 0.08
DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES = 0.75
DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES = 0.75


@dataclass(frozen=True, slots=True)
class RoadEndpoint:
    object_id: int
    model_path: str
    family: str
    object_kind: str
    endpoint_index: int
    point: tuple[float, float]
    tangent_axis_degrees: float
    outward_heading_degrees: float
    half_width_metres: float


@dataclass(frozen=True, slots=True)
class RoadObject:
    object_id: int
    model_path: str
    x: float
    y: float
    z: float
    heading_degrees: float
    pitch_degrees: float
    family: str
    kind: str
    nominal_length_metres: float
    logical_center: tuple[float, float]
    endpoints: tuple[RoadEndpoint, ...]


@dataclass(frozen=True, slots=True)
class SourceJunction:
    point: tuple[float, float]
    headings_degrees: tuple[float, ...]


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
    candidate_fix: str
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class InspectionResult:
    input_path: str
    wrp_entry: str
    road_object_count: int
    source_junction_count: int
    issues: tuple[RoadIssue, ...]
    road_objects: tuple[RoadObject, ...]


def _normalise_model(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


def _heading_unit(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


def _angular_distance(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _axis_heading_difference(first: float, second: float) -> float:
    difference = _angular_distance(first, second)
    return min(difference, abs(180.0 - difference))


def _transform_local(
    point: tuple[float, float],
    origin: tuple[float, float],
    heading_degrees: float,
) -> tuple[float, float]:
    return _geometry.transform_local(point, origin, heading_degrees)


def _endpoint(
    *,
    object_id: int,
    model_path: str,
    family: str,
    kind: str,
    endpoint_index: int,
    point: tuple[float, float],
    tangent_axis_degrees: float,
    outward_heading_degrees: float,
) -> RoadEndpoint:
    return RoadEndpoint(
        object_id=object_id,
        model_path=model_path,
        family=family,
        object_kind=kind,
        endpoint_index=endpoint_index,
        point=(float(point[0]), float(point[1])),
        tangent_axis_degrees=float(tangent_axis_degrees) % 180.0,
        outward_heading_degrees=float(outward_heading_degrees) % 360.0,
        half_width_metres=float(_geometry.STOCK_HALF_WIDTHS_METRES[family]),
    )


def _road_object_from_record(values) -> RoadObject | None:
    raw_model = values[13].split(b"\0", 1)[0]
    if not raw_model:
        return None
    try:
        model_path = raw_model.decode("ascii")
    except UnicodeDecodeError:
        return None
    normalized = _normalise_model(model_path)
    object_id = int(values[12])
    x, y, z = float(values[9]), float(values[10]), float(values[11])
    heading = math.degrees(math.atan2(-float(values[2]), float(values[0]))) % 360.0
    pitch_sine = max(-1.0, min(1.0, float(values[7])))
    pitch = math.degrees(math.asin(pitch_sine))
    origin = (x, z)

    if _PAVED_FILL.fullmatch(normalized) is not None:
        return RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            "sil",
            "paved_fill",
            float(_geometry.STOCK_HALF_WIDTHS_METRES["sil"]) * 2.0,
            origin,
            (),
        )

    straight = _geometry.stock_straight_match(normalized)
    if straight is not None:
        family = straight.group("family").casefold()
        nominal = int(straight.group("length"))
        length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[nominal])
        direction = _heading_unit(heading)
        first = (x - direction[0] * length * 0.5, z - direction[1] * length * 0.5)
        second = (x + direction[0] * length * 0.5, z + direction[1] * length * 0.5)
        endpoints = (
            _endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="straight",
                endpoint_index=0,
                point=first,
                tangent_axis_degrees=heading,
                outward_heading_degrees=heading + 180.0,
            ),
            _endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="straight",
                endpoint_index=1,
                point=second,
                tangent_axis_degrees=heading,
                outward_heading_degrees=heading,
            ),
        )
        return RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            family,
            "straight",
            length,
            origin,
            endpoints,
        )

    curve = _geometry.stock_curve_connectors(normalized)
    if curve is not None:
        family = curve.family
        begin = _transform_local(curve.begin, origin, heading)
        end = _transform_local(curve.end, origin, heading)
        endpoints = (
            _endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="curve",
                endpoint_index=0,
                point=begin,
                tangent_axis_degrees=heading,
                outward_heading_degrees=heading + 180.0,
            ),
            _endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="curve",
                endpoint_index=1,
                point=end,
                tangent_axis_degrees=heading + _geometry.STOCK_CURVE_ANGLE_DEGREES,
                outward_heading_degrees=heading + _geometry.STOCK_CURVE_ANGLE_DEGREES,
            ),
        )
        return RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            family,
            "curve",
            float(curve.chord_length_metres),
            ((begin[0] + end[0]) * 0.5, (begin[1] + end[1]) * 0.5),
            endpoints,
        )

    match = _T_JUNCTION.fullmatch(normalized)
    if match is not None:
        main = match.group("main").casefold()
        branch = match.group("branch").casefold()
        local_center = _geometry.native_junction_intersection_offset(normalized)
        if local_center is None:
            return None
        center = _transform_local(local_center, origin, heading)
        cx, cz = local_center
        radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
        local_connectors = (
            ((cx, cz + radius), main, heading),
            ((cx, cz - radius), main, heading + 180.0),
            ((cx - radius, cz), branch, heading + 270.0),
        )
        endpoints = tuple(
            _endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="junction",
                endpoint_index=index,
                point=_transform_local(local, origin, heading),
                tangent_axis_degrees=outward,
                outward_heading_degrees=outward,
            )
            for index, (local, family, outward) in enumerate(local_connectors)
        )
        return RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            main,
            "junction_t",
            radius,
            center,
            endpoints,
        )

    if _X_JUNCTION.fullmatch(normalized) is not None:
        family = "sil"
        radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
        center = origin
        local_connectors = (
            ((0.0, radius), heading),
            ((0.0, -radius), heading + 180.0),
            ((radius, 0.0), heading + 90.0),
            ((-radius, 0.0), heading + 270.0),
        )
        endpoints = tuple(
            _endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="junction",
                endpoint_index=index,
                point=_transform_local(local, origin, heading),
                tangent_axis_degrees=outward,
                outward_heading_degrees=outward,
            )
            for index, (local, outward) in enumerate(local_connectors)
        )
        return RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            family,
            "junction_x",
            radius,
            center,
            endpoints,
        )
    return None


def _read_rvw4_road_objects(data: bytes) -> tuple[RoadObject, ...]:
    stream = io.BytesIO(data)
    header = stream.read(_RVW4_HEADER.size)
    if len(header) != _RVW4_HEADER.size:
        raise ValueError("truncated RVW4 header")
    magic, width, height = _RVW4_HEADER.unpack(header)
    if magic != b"4WVR":
        raise ValueError("Road Inspector currently supports RVW4 WRP files only")
    if width <= 0 or height <= 0:
        raise ValueError("invalid RVW4 dimensions")
    cells = int(width) * int(height)
    skip = cells * 2 + cells * 2 + _TEXTURE_TABLE_BYTES
    if len(stream.read(skip)) != skip:
        raise ValueError("truncated RVW4 terrain/texture section")

    roads: list[RoadObject] = []
    while True:
        record = stream.read(_RVW4_OBJECT.size)
        if not record:
            raise ValueError("RVW4 object list is missing its terminator")
        if len(record) != _RVW4_OBJECT.size:
            raise ValueError("truncated RVW4 object record")
        values = _RVW4_OBJECT.unpack(record)
        if not values[13].split(b"\0", 1)[0]:
            break
        road = _road_object_from_record(values)
        if road is not None:
            roads.append(road)
    return tuple(roads)


def _wrp_bytes(path: Path) -> tuple[bytes, str]:
    suffix = path.suffix.casefold()
    if suffix == ".wrp":
        return path.read_bytes(), path.name
    if suffix != ".pbo":
        raise ValueError("input must be an RVW4 .wrp or an uncompressed .pbo")
    entries = [entry for entry in read_pbo(path) if entry.name.casefold().endswith(".wrp")]
    if not entries:
        raise ValueError("PBO does not contain a WRP entry")
    if len(entries) == 1:
        chosen = entries[0]
    else:
        stem = path.stem.casefold()
        matching = [entry for entry in entries if Path(entry.name.replace("\\", "/")).stem.casefold() == stem]
        if len(matching) == 1:
            chosen = matching[0]
        else:
            names = ", ".join(entry.name for entry in entries[:8])
            raise ValueError(f"PBO contains multiple WRP entries; cannot choose safely: {names}")
    return chosen.data, chosen.name


def _bucket(point: tuple[float, float], size: float) -> tuple[int, int]:
    return math.floor(point[0] / size), math.floor(point[1] / size)


def _endpoint_clusters(
    endpoints: Sequence[RoadEndpoint], tolerance: float
) -> tuple[tuple[RoadEndpoint, ...], ...]:
    if not endpoints:
        return ()
    bucket_size = max(0.01, float(tolerance))
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, endpoint in enumerate(endpoints):
        buckets.setdefault(_bucket(endpoint.point, bucket_size), []).append(index)

    parent = list(range(len(endpoints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merge(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, endpoint in enumerate(endpoints):
        bx, bz = _bucket(endpoint.point, bucket_size)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for candidate in buckets.get((bx + dx, bz + dz), ()):
                    if candidate <= index:
                        continue
                    other = endpoints[candidate]
                    if endpoint.object_id == other.object_id:
                        continue
                    if math.dist(endpoint.point, other.point) <= tolerance:
                        merge(index, candidate)

    grouped: dict[int, list[RoadEndpoint]] = {}
    for index, endpoint in enumerate(endpoints):
        grouped.setdefault(find(index), []).append(endpoint)
    return tuple(tuple(values) for values in grouped.values() if len(values) >= 2)


def _cross_section_edges(endpoint: RoadEndpoint) -> tuple[tuple[float, float], tuple[float, float]]:
    heading = math.radians(endpoint.tangent_axis_degrees)
    normal = (math.cos(heading), -math.sin(heading))
    width = endpoint.half_width_metres
    return (
        (endpoint.point[0] + normal[0] * width, endpoint.point[1] + normal[1] * width),
        (endpoint.point[0] - normal[0] * width, endpoint.point[1] - normal[1] * width),
    )


def _edge_discontinuity(first: RoadEndpoint, second: RoadEndpoint) -> tuple[float, float, float]:
    first_edges = _cross_section_edges(first)
    second_edges = _cross_section_edges(second)
    direct = (math.dist(first_edges[0], second_edges[0]), math.dist(first_edges[1], second_edges[1]))
    crossed = (math.dist(first_edges[0], second_edges[1]), math.dist(first_edges[1], second_edges[0]))
    chosen = direct if sum(direct) <= sum(crossed) else crossed
    return max(chosen), min(chosen), sum(chosen) * 0.5


def _severity(score: float) -> str:
    if score >= 80.0:
        return "critical"
    if score >= 55.0:
        return "high"
    if score >= 30.0:
        return "medium"
    return "low"


def _score_geometry(*, center_gap: float, edge_gap: float, tangent_error: float, family_mismatch: bool = False) -> float:
    score = 0.0
    score += min(35.0, max(0.0, center_gap) * 70.0)
    score += min(40.0, max(0.0, edge_gap) * 55.0)
    score += min(30.0, max(0.0, tangent_error) * 4.0)
    if family_mismatch:
        score += 20.0
    return min(100.0, score)


def _candidate_for_seam(first: RoadEndpoint, second: RoadEndpoint, tangent_error: float, center_gap: float) -> str:
    if first.family != second.family:
        return "Verify the intended surface transition; otherwise refit both sides with one road family or a measured mixed junction."
    if first.object_kind == "straight" and second.object_kind == "straight" and tangent_error >= 2.0:
        return f"Refit this local {first.family} bend with connector-locked 10-degree stock curves before falling back to rotated straight pieces."
    if first.object_kind == "curve" or second.object_kind == "curve":
        return "Refit the adjacent curve/straight pieces as one exact connector chain so their rendered tangents agree at the shared seam."
    if center_gap >= 0.05:
        return "Retarget the two physical connectors to the same world point; keep any repair piece below the visible road surface."
    return "Prefer a longer straight or native curve sequence so the visible road edges, not only the centrelines, are continuous."


def _ordinary_seam_issue(
    first: RoadEndpoint,
    second: RoadEndpoint,
    *,
    minimum_edge_gap: float,
    minimum_tangent_error: float,
) -> RoadIssue | None:
    center_gap = math.dist(first.point, second.point)
    tangent_error = _axis_heading_difference(first.tangent_axis_degrees, second.tangent_axis_degrees)
    edge_max, edge_min, edge_mean = _edge_discontinuity(first, second)
    family_mismatch = first.family != second.family
    if (
        center_gap < 0.05
        and edge_max < minimum_edge_gap
        and tangent_error < minimum_tangent_error
        and not family_mismatch
    ):
        return None

    if family_mismatch:
        category = "surface_family_mismatch"
    elif first.object_kind == "straight" and second.object_kind == "straight" and tangent_error >= minimum_tangent_error:
        category = "straight_miter"
    elif first.object_kind == "curve" or second.object_kind == "curve":
        category = "curve_transition"
    else:
        category = "connector_gap"

    score = _score_geometry(
        center_gap=center_gap,
        edge_gap=edge_max,
        tangent_error=tangent_error,
        family_mismatch=family_mismatch,
    )
    x = (first.point[0] + second.point[0]) * 0.5
    z = (first.point[1] + second.point[1]) * 0.5
    return RoadIssue(
        issue_id="",
        severity=_severity(score),
        score=score,
        category=category,
        x=x,
        z=z,
        object_ids=tuple(sorted((first.object_id, second.object_id))),
        models=(first.model_path, second.model_path),
        message=(
            f"{first.model_path} -> {second.model_path}: center gap {center_gap:.3f} m, "
            f"tangent mismatch {tangent_error:.2f}°, maximum road-edge discontinuity {edge_max:.3f} m."
        ),
        candidate_fix=_candidate_for_seam(first, second, tangent_error, center_gap),
        metrics={
            "center_gap_metres": round(center_gap, 5),
            "tangent_error_degrees": round(tangent_error, 5),
            "edge_gap_max_metres": round(edge_max, 5),
            "edge_gap_min_metres": round(edge_min, 5),
            "edge_gap_mean_metres": round(edge_mean, 5),
        },
    )


def _junction_connector_issue(
    connector: RoadEndpoint,
    approach: RoadEndpoint,
    *,
    minimum_edge_gap: float,
    minimum_tangent_error: float,
) -> RoadIssue | None:
    center_gap = math.dist(connector.point, approach.point)
    tangent_error = _axis_heading_difference(connector.tangent_axis_degrees, approach.tangent_axis_degrees)
    edge_max, edge_min, edge_mean = _edge_discontinuity(connector, approach)
    family_mismatch = connector.family != approach.family
    if (
        center_gap < 0.05
        and edge_max < minimum_edge_gap
        and tangent_error < minimum_tangent_error
        and not family_mismatch
    ):
        return None
    score = min(
        100.0,
        _score_geometry(
            center_gap=center_gap,
            edge_gap=edge_max,
            tangent_error=tangent_error,
            family_mismatch=family_mismatch,
        )
        + 10.0,
    )
    x = (connector.point[0] + approach.point[0]) * 0.5
    z = (connector.point[1] + approach.point[1]) * 0.5
    return RoadIssue(
        issue_id="",
        severity=_severity(score),
        score=score,
        category="junction_connector_mismatch",
        x=x,
        z=z,
        object_ids=tuple(sorted((connector.object_id, approach.object_id))),
        models=(connector.model_path, approach.model_path),
        message=(
            f"Junction connector does not match its approach: center gap {center_gap:.3f} m, "
            f"tangent mismatch {tangent_error:.2f}°, edge discontinuity {edge_max:.3f} m."
        ),
        candidate_fix=(
            "Rotate/position the native junction from its measured Memory-LOD connector geometry, or keep the approach visible over a lower fallback cap when the rigid junction cannot match."
        ),
        metrics={
            "center_gap_metres": round(center_gap, 5),
            "tangent_error_degrees": round(tangent_error, 5),
            "edge_gap_max_metres": round(edge_max, 5),
            "edge_gap_min_metres": round(edge_min, 5),
            "edge_gap_mean_metres": round(edge_mean, 5),
        },
    )


def _dedupe_headings(values: Iterable[float], tolerance: float = 1.0) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        value = float(value) % 360.0
        if any(_angular_distance(value, previous) <= tolerance for previous in result):
            continue
        result.append(value)
    return tuple(sorted(result))


def _line_coordinate_sequences(geometry) -> Iterable[Sequence[Sequence[float]]]:
    if not isinstance(geometry, dict):
        return
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "LineString" and isinstance(coordinates, list):
        yield coordinates
    elif kind == "MultiLineString" and isinstance(coordinates, list):
        for line in coordinates:
            if isinstance(line, list):
                yield line


def _source_junctions(path: Path | None) -> tuple[SourceJunction, ...]:
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", []) if isinstance(payload, dict) else []
    nodes: dict[tuple[int, int], tuple[tuple[float, float], list[float]]] = {}
    quantum = 0.05
    for feature in features:
        if not isinstance(feature, dict):
            continue
        for coordinates in _line_coordinate_sequences(feature.get("geometry")):
            points: list[tuple[float, float]] = []
            for coordinate in coordinates:
                if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
                    continue
                try:
                    point = (float(coordinate[0]), float(coordinate[1]))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(point[0]) and math.isfinite(point[1]):
                    points.append(point)
            for index, point in enumerate(points):
                key = (round(point[0] / quantum), round(point[1] / quantum))
                stored_point, headings = nodes.setdefault(key, (point, []))
                del stored_point
                for neighbour_index in (index - 1, index + 1):
                    if not 0 <= neighbour_index < len(points):
                        continue
                    neighbour = points[neighbour_index]
                    dx, dz = neighbour[0] - point[0], neighbour[1] - point[1]
                    if math.hypot(dx, dz) <= 0.05:
                        continue
                    headings.append(math.degrees(math.atan2(dx, dz)) % 360.0)

    result = []
    for point, headings in nodes.values():
        unique = _dedupe_headings(headings)
        if len(unique) >= 3:
            result.append(SourceJunction(point, unique))
    return tuple(result)


def _dominant_through_pair(headings: Sequence[float]) -> tuple[float, float, float] | None:
    if len(headings) < 2:
        return None
    best = None
    for first_index in range(len(headings)):
        for second_index in range(first_index + 1, len(headings)):
            first, second = headings[first_index], headings[second_index]
            separation = _angular_distance(first, second)
            candidate = (abs(180.0 - separation), first, second)
            if best is None or candidate < best:
                best = candidate
    return best


def _closest_direction_error(target: float, values: Sequence[float]) -> float:
    return min((_angular_distance(target, value) for value in values), default=180.0)


def _source_intersection_issues(
    roads: Sequence[RoadObject],
    junctions: Sequence[SourceJunction],
    *,
    match_tolerance: float,
) -> list[RoadIssue]:
    issues: list[RoadIssue] = []
    if not junctions:
        return issues

    endpoints = [endpoint for road in roads if road.kind != "junction_t" and road.kind != "junction_x" for endpoint in road.endpoints]
    for source in junctions:
        node = source.point
        caps = [
            road
            for road in roads
            if (
                road.kind in {"junction_t", "junction_x", "paved_fill"}
                or (road.kind == "straight" and road.nominal_length_metres <= 6.26)
            )
            and math.dist(road.logical_center, node) <= match_tolerance
        ]
        cap = min(caps, key=lambda road: math.dist(road.logical_center, node), default=None)
        nearby_endpoints = [
            endpoint
            for endpoint in endpoints
            if math.dist(endpoint.point, node) <= max(match_tolerance, 0.90)
        ]
        emitted_headings = _dedupe_headings(endpoint.outward_heading_degrees for endpoint in nearby_endpoints)
        heading_errors = [
            _closest_direction_error(source_heading, emitted_headings)
            for source_heading in source.headings_degrees
        ]
        maximum_approach_error = max(heading_errors, default=180.0)

        if cap is None:
            if maximum_approach_error > 3.0:
                score = min(100.0, 45.0 + maximum_approach_error * 4.0)
                issues.append(
                    RoadIssue(
                        "",
                        _severity(score),
                        score,
                        "intersection_missing_cap",
                        node[0],
                        node[1],
                        tuple(sorted({ep.object_id for ep in nearby_endpoints})),
                        tuple(sorted({ep.model_path for ep in nearby_endpoints})),
                        f"Source intersection has {len(source.headings_degrees)} arms but no stock junction/cap was found near the node; maximum approach heading error is {maximum_approach_error:.2f}°.",
                        "Refit the final approach pieces to the logical node and select a measured native T/X only when its connectors match all incident directions.",
                        {
                            "source_degree": len(source.headings_degrees),
                            "maximum_approach_heading_error_degrees": round(maximum_approach_error, 5),
                        },
                    )
                )
            continue

        expected_degree = 3 if cap.kind == "junction_t" else 4 if cap.kind == "junction_x" else None
        if expected_degree is not None and expected_degree != len(source.headings_degrees):
            score = 90.0
            issues.append(
                RoadIssue(
                    "",
                    "critical",
                    score,
                    "wrong_intersection_model",
                    node[0],
                    node[1],
                    (cap.object_id,),
                    (cap.model_path,),
                    f"{cap.model_path} has {expected_degree} measured connectors but the normalized road source has {len(source.headings_degrees)} incident directions.",
                    "Choose a T/X model matching the actual source degree; otherwise use a low central fallback while keeping each fitted approach visible.",
                    {"source_degree": len(source.headings_degrees), "model_degree": expected_degree},
                )
            )

        if cap.kind in {"junction_t", "junction_x"}:
            connector_headings = tuple(endpoint.outward_heading_degrees for endpoint in cap.endpoints)
            connector_errors = [
                _closest_direction_error(source_heading, connector_headings)
                for source_heading in source.headings_degrees
            ]
            maximum_connector_error = max(connector_errors, default=180.0)
            if maximum_connector_error >= 1.0 or maximum_approach_error >= 2.0:
                half_width = float(_geometry.STOCK_HALF_WIDTHS_METRES.get(cap.family, 4.55))
                edge_estimate = 2.0 * half_width * math.sin(math.radians(maximum_connector_error) * 0.5)
                score = min(100.0, 35.0 + maximum_connector_error * 5.0 + edge_estimate * 35.0)
                issues.append(
                    RoadIssue(
                        "",
                        _severity(score),
                        score,
                        "intersection_connector_orientation",
                        node[0],
                        node[1],
                        tuple(sorted({cap.object_id, *(ep.object_id for ep in nearby_endpoints)})),
                        tuple(sorted({cap.model_path, *(ep.model_path for ep in nearby_endpoints)})),
                        f"Native intersection connector orientation differs from the normalized incident roads by up to {maximum_connector_error:.2f}°; estimated edge offset is {edge_estimate:.3f} m.",
                        "Rotate the measured native junction only inside its connector tolerance. If the incident road turns at the node, keep the fitted approaches visible and use a lower intersection fill instead of forcing the rigid mesh.",
                        {
                            "source_degree": len(source.headings_degrees),
                            "maximum_connector_heading_error_degrees": round(maximum_connector_error, 5),
                            "maximum_approach_heading_error_degrees": round(maximum_approach_error, 5),
                            "estimated_edge_offset_metres": round(edge_estimate, 5),
                        },
                    )
                )
            continue

        if cap.kind == "paved_fill":
            # A borderless disk deliberately has no rigid connector headings.
            # It can own the centre of a turning all-paved node while the
            # fitted approaches retain their real source tangents.
            if maximum_approach_error >= 3.0:
                score = min(100.0, 35.0 + maximum_approach_error * 5.0)
                issues.append(
                    RoadIssue(
                        "",
                        _severity(score),
                        score,
                        "intersection_approach_mismatch",
                        node[0],
                        node[1],
                        tuple(sorted({cap.object_id, *(ep.object_id for ep in nearby_endpoints)})),
                        tuple(sorted({cap.model_path, *(ep.model_path for ep in nearby_endpoints)})),
                        f"One or more emitted approaches miss the normalized intersection tangent by up to {maximum_approach_error:.2f}°.",
                        "Refit the final approach pieces to the logical node, using a native curve before the intersection when the source road is already turning.",
                        {
                            "source_degree": len(source.headings_degrees),
                            "maximum_approach_heading_error_degrees": round(maximum_approach_error, 5),
                        },
                    )
                )
            continue

        # Legacy six-metre straight cap. It can fill an intersection, but it
        # must not be the visible surface when the through road changes heading.
        pair = _dominant_through_pair(source.headings_degrees)
        through_turn = float(pair[0]) if pair is not None else 0.0
        cap_axis_errors = (
            [
                _axis_heading_difference(source_heading, cap.heading_degrees)
                for source_heading in source.headings_degrees
            ]
            if source.headings_degrees
            else []
        )
        best_two = sorted(cap_axis_errors)[:2]
        maximum_main_axis_error = max(best_two, default=180.0)
        nearby_objects = {road.object_id: road for road in roads if road.object_id in {ep.object_id for ep in nearby_endpoints}}
        approach_heights = [road.y for road in nearby_objects.values()]
        visible_margin = (min(approach_heights) - cap.y) if approach_heights else 0.0
        if through_turn >= 1.0 and (visible_margin < 0.002 or maximum_approach_error >= 2.0):
            half_width = float(_geometry.STOCK_HALF_WIDTHS_METRES.get(cap.family, 4.55))
            edge_estimate = 2.0 * half_width * math.sin(math.radians(through_turn) * 0.5)
            score = min(100.0, 40.0 + through_turn * 4.0 + edge_estimate * 25.0)
            issues.append(
                RoadIssue(
                    "",
                    _severity(score),
                    score,
                    "turning_intersection_cap",
                    node[0],
                    node[1],
                    tuple(sorted({cap.object_id, *(ep.object_id for ep in nearby_endpoints)})),
                    tuple(sorted({cap.model_path, *(ep.model_path for ep in nearby_endpoints)})),
                    f"The through road turns {through_turn:.2f}° at a legacy straight intersection cap. Cap/approach vertical margin is {visible_margin:.3f} m and the estimated edge mismatch is {edge_estimate:.3f} m.",
                    "Keep the actual turning approaches as the visible top surface, lower the straight cap to central fill, and use low same-family tongues only along uncovered incident axes.",
                    {
                        "source_degree": len(source.headings_degrees),
                        "through_turn_degrees": round(through_turn, 5),
                        "maximum_main_axis_error_degrees": round(maximum_main_axis_error, 5),
                        "maximum_approach_heading_error_degrees": round(maximum_approach_error, 5),
                        "cap_below_approach_margin_metres": round(visible_margin, 5),
                        "estimated_edge_offset_metres": round(edge_estimate, 5),
                    },
                )
            )
        elif maximum_approach_error >= 3.0:
            score = min(100.0, 35.0 + maximum_approach_error * 5.0)
            issues.append(
                RoadIssue(
                    "",
                    _severity(score),
                    score,
                    "intersection_approach_mismatch",
                    node[0],
                    node[1],
                    tuple(sorted({cap.object_id, *(ep.object_id for ep in nearby_endpoints)})),
                    tuple(sorted({cap.model_path, *(ep.model_path for ep in nearby_endpoints)})),
                    f"One or more emitted approaches miss the normalized intersection tangent by up to {maximum_approach_error:.2f}°.",
                    "Refit the final stock piece on each incident road to the logical node, using a native curve before the intersection when the source road is already turning.",
                    {
                        "source_degree": len(source.headings_degrees),
                        "maximum_approach_heading_error_degrees": round(maximum_approach_error, 5),
                    },
                )
            )
    return issues


def _number_issues(issues: Iterable[RoadIssue]) -> tuple[RoadIssue, ...]:
    ordered = sorted(issues, key=lambda issue: (-issue.score, issue.category, issue.x, issue.z, issue.object_ids))
    seen = set()
    result = []
    sequence = 1
    for issue in ordered:
        key = (
            issue.category,
            tuple(issue.object_ids),
            round(issue.x, 2),
            round(issue.z, 2),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(
            RoadIssue(
                issue_id=f"RI-{sequence:05d}",
                severity=issue.severity,
                score=round(issue.score, 2),
                category=issue.category,
                x=round(issue.x, 4),
                z=round(issue.z, 4),
                object_ids=issue.object_ids,
                models=issue.models,
                message=issue.message,
                candidate_fix=issue.candidate_fix,
                metrics=issue.metrics,
            )
        )
        sequence += 1
    return tuple(result)


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
) -> InspectionResult:
    data, wrp_entry = _wrp_bytes(Path(input_path))
    roads = _read_rvw4_road_objects(data)
    endpoints = tuple(endpoint for road in roads for endpoint in road.endpoints)
    clusters = _endpoint_clusters(endpoints, endpoint_tolerance)
    issues: list[RoadIssue] = []

    for cluster in clusters:
        unique = {(endpoint.object_id, endpoint.endpoint_index): endpoint for endpoint in cluster}
        values = tuple(unique.values())
        if len(values) != 2:
            continue
        first, second = values
        first_is_junction = first.object_kind == "junction"
        second_is_junction = second.object_kind == "junction"
        if first_is_junction != second_is_junction:
            connector, approach = (first, second) if first_is_junction else (second, first)
            issue = _junction_connector_issue(
                connector,
                approach,
                minimum_edge_gap=minimum_edge_gap,
                minimum_tangent_error=minimum_tangent_error,
            )
        elif not first_is_junction and not second_is_junction:
            issue = _ordinary_seam_issue(
                first,
                second,
                minimum_edge_gap=minimum_edge_gap,
                minimum_tangent_error=minimum_tangent_error,
            )
        else:
            issue = None
        if issue is not None:
            issues.append(issue)

    source_junctions = _source_junctions(roads_geojson)
    issues.extend(
        _source_intersection_issues(
            roads,
            source_junctions,
            match_tolerance=junction_match_tolerance,
        )
    )
    numbered = _number_issues(issues)
    return InspectionResult(
        input_path=str(input_path),
        wrp_entry=wrp_entry,
        road_object_count=len(roads),
        source_junction_count=len(source_junctions),
        issues=numbered,
        road_objects=roads,
    )


def _summary_payload(result: InspectionResult) -> dict[str, object]:
    severity = Counter(issue.severity for issue in result.issues)
    categories = Counter(issue.category for issue in result.issues)
    return {
        "input": result.input_path,
        "wrp_entry": result.wrp_entry,
        "road_objects": result.road_object_count,
        "source_junctions": result.source_junction_count,
        "issue_count": len(result.issues),
        "severity_counts": dict(sorted(severity.items())),
        "category_counts": dict(sorted(categories.items())),
        "highest_score": max((issue.score for issue in result.issues), default=0.0),
    }


def _issue_payload(issue: RoadIssue) -> dict[str, object]:
    value = asdict(issue)
    value["object_ids"] = list(issue.object_ids)
    value["models"] = list(issue.models)
    return value


def _road_payload(road: RoadObject) -> dict[str, object]:
    return {
        "object_id": road.object_id,
        "model": road.model_path,
        "kind": road.kind,
        "family": road.family,
        "x": round(road.x, 4),
        "z": round(road.z, 4),
        "center": [round(road.logical_center[0], 4), round(road.logical_center[1], 4)],
        "segments": (
            []
            if not road.endpoints
            else
            [
                [round(road.logical_center[0], 4), round(road.logical_center[1], 4), round(endpoint.point[0], 4), round(endpoint.point[1], 4)]
                for endpoint in road.endpoints
            ]
            if road.kind.startswith("junction_")
            else [
                [
                    round(road.endpoints[0].point[0], 4),
                    round(road.endpoints[0].point[1], 4),
                    round(road.endpoints[-1].point[0], 4),
                    round(road.endpoints[-1].point[1], 4),
                ]
            ]
        ),
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, issues: Sequence[RoadIssue]) -> None:
    metric_names = sorted({name for issue in issues for name in issue.metrics})
    fields = [
        "issue_id", "severity", "score", "category", "x", "z", "object_ids", "models",
        "message", "candidate_fix", *metric_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            row = {
                "issue_id": issue.issue_id,
                "severity": issue.severity,
                "score": issue.score,
                "category": issue.category,
                "x": issue.x,
                "z": issue.z,
                "object_ids": ";".join(str(value) for value in issue.object_ids),
                "models": ";".join(issue.models),
                "message": issue.message,
                "candidate_fix": issue.candidate_fix,
            }
            row.update(issue.metrics)
            writer.writerow(row)


def _html_document(result: InspectionResult) -> str:
    summary = _summary_payload(result)
    issues_json = json.dumps([_issue_payload(issue) for issue in result.issues], separators=(",", ":"))
    roads_json = json.dumps([_road_payload(road) for road in result.road_objects], separators=(",", ":"))
    summary_json = json.dumps(summary, separators=(",", ":"))
    title = html.escape(f"Road Inspector - {Path(result.input_path).name}")
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>{title}</title>
<style>
:root {{ font-family: system-ui, Segoe UI, sans-serif; color-scheme: dark; background:#111; color:#eee; }}
body {{ margin:0; display:grid; grid-template-columns:minmax(360px,42vw) 1fr; height:100vh; overflow:hidden; }}
#panel {{ overflow:auto; padding:16px; border-right:1px solid #444; background:#171717; }}
#mapwrap {{ position:relative; min-width:0; background:#0c0f0c; }}
#map {{ width:100%; height:100%; display:block; }}
h1 {{ font-size:20px; margin:0 0 6px; }} .muted {{ color:#aaa; }}
.stats {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; }}
.stat {{ padding:6px 9px; border:1px solid #444; border-radius:6px; background:#202020; }}
.controls {{ position:sticky; top:-16px; z-index:3; padding:10px 0; background:#171717; display:flex; gap:8px; flex-wrap:wrap; }}
select,button,input {{ background:#222; color:#eee; border:1px solid #555; border-radius:5px; padding:6px 8px; }}
.issue {{ border:1px solid #3c3c3c; border-left-width:5px; border-radius:6px; margin:8px 0; padding:9px; cursor:pointer; background:#1d1d1d; }}
.issue:hover,.issue.active {{ background:#292929; }}
.issue.critical {{ border-left-color:#ff4242; }} .issue.high {{ border-left-color:#ff9d32; }} .issue.medium {{ border-left-color:#f1d54a; }} .issue.low {{ border-left-color:#6ca7ff; }}
.issuehead {{ display:flex; justify-content:space-between; gap:8px; font-weight:700; }}
.category {{ font-family:ui-monospace,Consolas,monospace; color:#b8d6ff; }}
.fix {{ margin-top:7px; color:#c7efc7; }}
.coords {{ font-family:ui-monospace,Consolas,monospace; }}
.road {{ stroke:#777; stroke-width:1.2; vector-effect:non-scaling-stroke; opacity:.55; }}
.road.curve {{ stroke:#a6a6a6; }} .road.junction {{ stroke:#ddd; stroke-width:2; }}
.marker {{ vector-effect:non-scaling-stroke; stroke:#111; stroke-width:1.2; cursor:pointer; }}
.marker.critical {{ fill:#ff4242; }} .marker.high {{ fill:#ff9d32; }} .marker.medium {{ fill:#f1d54a; }} .marker.low {{ fill:#6ca7ff; }}
.road.selected {{ stroke:#63d6ff; stroke-width:4; opacity:1; }}
#mapinfo {{ position:absolute; top:10px; left:10px; padding:8px 10px; border-radius:6px; background:#111d; border:1px solid #555; font-family:ui-monospace,Consolas,monospace; pointer-events:none; }}
@media (max-width:900px) {{ body {{ grid-template-columns:1fr; grid-template-rows:55vh 45vh; }} #panel {{ border-right:0; border-bottom:1px solid #444; }} }}
</style>
</head>
<body>
<section id=\"panel\">
<h1>{title}</h1><div class=\"muted\">Read-only audit of the actual emitted RVW4 road geometry.</div>
<div class=\"stats\" id=\"stats\"></div>
<div class=\"controls\"><select id=\"severity\"><option value=\"\">all severities</option><option>critical</option><option>high</option><option>medium</option><option>low</option></select><select id=\"category\"><option value=\"\">all categories</option></select><button id=\"reset\">Reset map</button></div>
<div id=\"issues\"></div>
</section>
<section id=\"mapwrap\"><svg id=\"map\"></svg><div id=\"mapinfo\">Click an issue to zoom to ±35 m</div></section>
<script>
const summary={summary_json}; const issues={issues_json}; const roads={roads_json};
const svg=document.getElementById('map'), list=document.getElementById('issues');
const ns='http://www.w3.org/2000/svg';
function allPoints(){{const p=[]; for(const r of roads)for(const s of r.segments){{p.push([s[0],s[1]],[s[2],s[3]])}} for(const i of issues)p.push([i.x,i.z]); return p;}}
const pts=allPoints(); let minx=Math.min(...pts.map(p=>p[0]),0), maxx=Math.max(...pts.map(p=>p[0]),1), minz=Math.min(...pts.map(p=>p[1]),0), maxz=Math.max(...pts.map(p=>p[1]),1);
function view(x0,z0,x1,z1){{svg.setAttribute('viewBox',`${{x0}} ${{-z1}} ${{Math.max(1,x1-x0)}} ${{Math.max(1,z1-z0)}}`);}}
function resetView(){{view(minx-10,minz-10,maxx+10,maxz+10)}} resetView();
const roadEls=new Map();
for(const r of roads){{ for(const s of r.segments){{const line=document.createElementNS(ns,'line'); line.setAttribute('x1',s[0]);line.setAttribute('y1',-s[1]);line.setAttribute('x2',s[2]);line.setAttribute('y2',-s[3]);line.classList.add('road'); if(r.kind==='curve')line.classList.add('curve'); if(r.kind.startsWith('junction_'))line.classList.add('junction'); line.dataset.object=r.object_id; svg.appendChild(line); if(!roadEls.has(r.object_id))roadEls.set(r.object_id,[]);roadEls.get(r.object_id).push(line); }}}}
for(const i of issues){{const c=document.createElementNS(ns,'circle');c.setAttribute('cx',i.x);c.setAttribute('cy',-i.z);c.setAttribute('r',2.2);c.classList.add('marker',i.severity);c.dataset.issue=i.issue_id;c.addEventListener('click',()=>selectIssue(i));svg.appendChild(c);}}
function selectIssue(i){{document.querySelectorAll('.road.selected').forEach(e=>e.classList.remove('selected')); for(const id of i.object_ids)for(const e of roadEls.get(id)||[])e.classList.add('selected'); view(i.x-35,i.z-35,i.x+35,i.z+35); document.getElementById('mapinfo').textContent=`${{i.issue_id}}  ${{i.x.toFixed(2)}}, ${{i.z.toFixed(2)}}  ${{i.category}}`; document.querySelectorAll('.issue.active').forEach(e=>e.classList.remove('active')); const row=document.querySelector(`[data-row="${{i.issue_id}}"]`); if(row){{row.classList.add('active');row.scrollIntoView({{block:'nearest'}});}}}}
function esc(s){{return String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));}}
function render(){{const sev=document.getElementById('severity').value, cat=document.getElementById('category').value; list.innerHTML=''; for(const i of issues){{if(sev&&i.severity!==sev||cat&&i.category!==cat)continue; const d=document.createElement('div');d.className=`issue ${{i.severity}}`;d.dataset.row=i.issue_id;d.innerHTML=`<div class="issuehead"><span>${{esc(i.issue_id)}} · ${{esc(i.severity.toUpperCase())}}</span><span>${{i.score.toFixed(1)}}</span></div><div class="category">${{esc(i.category)}}</div><div class="coords">${{i.x.toFixed(2)}}, ${{i.z.toFixed(2)}} · objects ${{esc(i.object_ids.join(', '))}}</div><div>${{esc(i.message)}}</div><div class="fix"><b>Candidate:</b> ${{esc(i.candidate_fix)}}</div>`;d.onclick=()=>selectIssue(i);list.appendChild(d);}}}}
const cats=[...new Set(issues.map(i=>i.category))].sort(); const catSel=document.getElementById('category');for(const c of cats){{const o=document.createElement('option');o.textContent=c;o.value=c;catSel.appendChild(o);}}
document.getElementById('severity').onchange=render;catSel.onchange=render;document.getElementById('reset').onclick=()=>{{resetView();document.querySelectorAll('.road.selected').forEach(e=>e.classList.remove('selected'));}};
document.getElementById('stats').innerHTML=`<span class="stat">${{summary.road_objects}} road objects</span><span class="stat">${{summary.issue_count}} issues</span><span class="stat">${{summary.severity_counts.critical||0}} critical</span><span class="stat">${{summary.severity_counts.high||0}} high</span><span class="stat">${{summary.source_junctions}} source junctions</span>`; render();
</script>
</body></html>"""


def write_inspection_report(result: InspectionResult, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    issues_path = output_dir / "issues.json"
    csv_path = output_dir / "issues.csv"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "report.html"
    _write_json(issues_path, [_issue_payload(issue) for issue in result.issues])
    _write_csv(csv_path, result.issues)
    _write_json(summary_path, _summary_payload(result))
    html_path.write_text(_html_document(result), encoding="utf-8")
    return {
        "issues_json": issues_path,
        "issues_csv": csv_path,
        "summary_json": summary_path,
        "html": html_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cwr-road-inspector",
        description="Inspect a generated CWA RVW4 WRP/PBO for stock-road wedges, clipping seams and intersection mismatches.",
    )
    parser.add_argument("input", type=Path, help="generated .wrp or uncompressed .pbo")
    parser.add_argument("--roads", type=Path, default=None, help="optional normalized/roads.geojson for intersection auditing")
    parser.add_argument("--output", type=Path, default=Path("road-inspector"), help="report directory")
    parser.add_argument("--endpoint-tolerance", type=float, default=DEFAULT_ENDPOINT_TOLERANCE_METRES)
    parser.add_argument("--minimum-edge-gap", type=float, default=DEFAULT_MINIMUM_EDGE_GAP_METRES)
    parser.add_argument("--minimum-tangent-error", type=float, default=DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES)
    parser.add_argument("--junction-match-tolerance", type=float, default=DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for label, value in (
        ("endpoint tolerance", args.endpoint_tolerance),
        ("minimum edge gap", args.minimum_edge_gap),
        ("minimum tangent error", args.minimum_tangent_error),
        ("junction match tolerance", args.junction_match_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"{label} must be finite and positive")
    result = inspect_road_geometry(
        args.input,
        roads_geojson=args.roads,
        endpoint_tolerance=args.endpoint_tolerance,
        minimum_edge_gap=args.minimum_edge_gap,
        minimum_tangent_error=args.minimum_tangent_error,
        junction_match_tolerance=args.junction_match_tolerance,
    )
    paths = write_inspection_report(result, args.output)
    counts = Counter(issue.severity for issue in result.issues)
    print(
        f"Road Inspector: {result.road_object_count:,} road objects, "
        f"{len(result.issues):,} issues "
        f"({counts.get('critical', 0)} critical, {counts.get('high', 0)} high)."
    )
    print(f"HTML report: {paths['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
