# SPDX-License-Identifier: GPL-3.0-or-later
"""Attach normalized source-road context to Road Inspector findings.

This is diagnostic metadata only.  It makes a reported WRP defect traceable back
to the exact normalized road feature without changing any generated object.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import math

from . import road_inspector as _core
from . import road_inspector_runtime as _runtime


_SOURCE_CONTEXT_RADIUS_METRES = 1.00
_ORIGINAL_INSPECT = None
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _SourceSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    road_id: str
    highway: str
    surface: str


def _segments(path: Path | None) -> tuple[_SourceSegment, ...]:
    if path is None:
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ()
    project = _runtime._geojson_projector(payload)
    result: list[_SourceSegment] = []
    for feature in payload.get("features", ()):
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        road_id = str(properties.get("road_id", ""))
        highway = str(properties.get("highway", ""))
        surface = str(properties.get("surface", ""))
        for coordinates in _core._line_coordinate_sequences(feature.get("geometry")):
            points = []
            for coordinate in coordinates:
                if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
                    continue
                try:
                    point = project(coordinate)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if math.isfinite(point[0]) and math.isfinite(point[1]):
                    points.append((float(point[0]), float(point[1])))
            for start, end in zip(points, points[1:]):
                if math.dist(start, end) <= 0.01:
                    continue
                result.append(
                    _SourceSegment(start, end, road_id, highway, surface)
                )
    return tuple(result)


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
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    projected = (start[0] + dx * fraction, start[1] + dz * fraction)
    return math.dist(point, projected)


def _context_metrics(issue, segments: tuple[_SourceSegment, ...]):
    if not segments:
        return {}
    point = (float(issue.x), float(issue.z))
    distances = [(_point_segment_distance(point, segment.start, segment.end), segment) for segment in segments]
    nearest_distance, nearest = min(distances, key=lambda value: value[0])
    close_limit = max(
        _SOURCE_CONTEXT_RADIUS_METRES,
        nearest_distance + 0.10,
    )
    nearby = [
        segment
        for distance, segment in distances
        if distance <= close_limit
    ]
    road_ids = sorted({segment.road_id for segment in nearby if segment.road_id})
    highways = sorted({segment.highway for segment in nearby if segment.highway})
    surfaces = sorted({segment.surface for segment in nearby if segment.surface})
    metrics: dict[str, float | str] = {
        "nearest_source_distance_metres": round(nearest_distance, 5),
    }
    if road_ids:
        metrics["source_road_ids"] = ";".join(road_ids)
    elif nearest.road_id:
        metrics["source_road_ids"] = nearest.road_id
    if highways:
        metrics["source_highways"] = ";".join(highways)
    elif nearest.highway:
        metrics["source_highways"] = nearest.highway
    if surfaces:
        metrics["source_surfaces"] = ";".join(surfaces)
    elif nearest.surface:
        metrics["source_surfaces"] = nearest.surface
    return metrics


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    if _ORIGINAL_INSPECT is None:
        raise RuntimeError("Road Inspector source context is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    source_segments = _segments(roads_geojson)
    if not source_segments or not result.issues:
        return result
    issues = tuple(
        replace(
            issue,
            metrics={
                **issue.metrics,
                **_context_metrics(issue, source_segments),
            },
        )
        for issue in result.issues
    )
    return replace(result, issues=issues)


def install() -> None:
    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
