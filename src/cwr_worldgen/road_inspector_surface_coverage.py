# SPDX-License-Identifier: GPL-3.0-or-later
"""Suppress Road Inspector seam alarms that are already covered by asphalt.

The inspector intentionally audits raw stock connectors, but the generator also
uses low same-family straight pieces as visual seam underlays. A second road can
likewise cross the same point and cover the entire open wedge. Reporting the raw
pair as a visible defect in either case sends debugging toward geometry that CWA
never exposes.

Filter only ordinary paved seam categories and only when another same-family
straight surface covers sampled points across both road-edge gaps and the
centreline gap. Dirt and mixed-family diagnostics are untouched. The test is
conservative: a nearby road centre is not enough, and vertically distant bridge
surfaces cannot hide a ground-level seam.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import re
import struct

from . import road_inspector as _core
from . import road_inspector_grass_wedge as _wedge
from .pbo import read_pbo

_COVERABLE_CATEGORIES = frozenset({"straight_miter", "curve_transition", "connector_gap"})
_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_SURFACE_MARGIN_METRES = 0.08
_VERTICAL_MARGIN_METRES = 0.15
_WEDGE_VERTICAL_MARGIN_METRES = 0.50
_MINIMUM_VISIBLE_CLEARANCE_METRES = 0.030
_RVW4_HEIGHT_SCALE_METRES = 0.05
_SAMPLE_FRACTIONS = (0.20, 0.40, 0.60, 0.80)

_ORIGINAL_INSPECT = None
_INSTALLED = False


def _terrain_context(input_path: Path):
    """Return (elevations, cells, cell_size) for a generated PBO when available."""

    input_path = Path(input_path)
    if input_path.suffix.casefold() != ".pbo":
        return None
    entries = read_pbo(input_path)
    configs = tuple(
        entry for entry in entries if entry.name.casefold().endswith("config.cpp")
    )
    if not configs:
        return None
    cell_size = None
    for entry in configs:
        text = entry.data.decode("latin1", errors="ignore")
        match = re.search(
            r"\blandGrid\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*;",
            text,
            re.IGNORECASE,
        )
        if match is not None:
            cell_size = float(match.group("value"))
            break
    if cell_size is None or not math.isfinite(cell_size) or cell_size <= 0.0:
        return None

    data, _entry_name = _core._wrp_bytes(input_path)
    if len(data) < _core._RVW4_HEADER.size:
        return None
    magic, width, height = _core._RVW4_HEADER.unpack_from(data, 0)
    if magic != b"4WVR" or width <= 0 or width != height:
        return None
    cells = int(width)
    count = cells * cells
    offset = _core._RVW4_HEADER.size
    if len(data) < offset + count * 2:
        return None
    raw = struct.unpack_from(f"<{count}h", data, offset)
    elevations = tuple(value * _RVW4_HEIGHT_SCALE_METRES for value in raw)
    return elevations, cells, cell_size


def _terrain_height(terrain, point: tuple[float, float]) -> float:
    elevations, cells, cell_size = terrain
    fx = max(0.0, min(cells - 1.0, float(point[0]) / cell_size))
    fz = max(0.0, min(cells - 1.0, float(point[1]) / cell_size))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(cells - 1, x0 + 1)
    z1 = min(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    a = elevations[z0 * cells + x0] * (1.0 - tx) + elevations[z0 * cells + x1] * tx
    b = elevations[z1 * cells + x0] * (1.0 - tx) + elevations[z1 * cells + x1] * tx
    return a * (1.0 - tz) + b * tz


def _surface_height(road, point: tuple[float, float]) -> float:
    heading = math.radians(float(road.heading_degrees))
    pitch = math.radians(float(road.pitch_degrees))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) <= 1.0e-9:
        return float(road.y)
    dx = float(point[0]) - float(road.x)
    dz = float(point[1]) - float(road.z)
    local_z = (dx * math.sin(heading) + dz * math.cos(heading)) / cosine_pitch
    return float(road.y) + local_z * math.sin(pitch)


def _visibly_above_terrain(road, point, terrain) -> bool:
    if terrain is None:
        return True
    return (
        _surface_height(road, point) - _terrain_height(terrain, point)
        >= _MINIMUM_VISIBLE_CLEARANCE_METRES
    )


def _nearest_endpoint(road, point: tuple[float, float]):
    return min(road.endpoints, key=lambda endpoint: math.dist(endpoint.point, point))


def _matched_edge_pairs(first, second):
    first_edges = _core._cross_section_edges(first)
    second_edges = _core._cross_section_edges(second)
    direct = (
        (first_edges[0], second_edges[0]),
        (first_edges[1], second_edges[1]),
    )
    crossed = (
        (first_edges[0], second_edges[1]),
        (first_edges[1], second_edges[0]),
    )
    direct_length = sum(math.dist(start, end) for start, end in direct)
    crossed_length = sum(math.dist(start, end) for start, end in crossed)
    return direct if direct_length <= crossed_length else crossed


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
    wedge = _wedge._grass_wedge_geometry(first, second)
    if wedge is not None:
        apex = wedge[3]
        centroid = wedge[4]
        samples.extend((apex, centroid))
    return tuple(samples)


def _straight_contains(road, point: tuple[float, float]) -> bool:
    if road.kind == "paved_fill":
        radius = float(_core._geometry.STOCK_HALF_WIDTHS_METRES["sil"])
        # A circular fill stops at the nominal road edge. Do not apply the
        # general surface margin here: even a shallow turn has a real miter
        # apex just beyond this radius.
        return math.dist(road.logical_center, point) <= radius + 1.0e-3
    if road.kind == "paved_miter":
        return _core._paved_miter_contains(
            road,
            point,
            margin=_SURFACE_MARGIN_METRES,
        )
    if road.kind == "paved_wedge":
        return _core._paved_wedge_contains(road, point, margin=0.005)
    if road.kind != "straight" or len(road.endpoints) != 2:
        return False
    start = road.endpoints[0].point
    end = road.endpoints[1].point
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    length = math.hypot(dx, dz)
    if length <= 1.0e-9:
        return False
    ux, uz = dx / length, dz / length
    px = float(point[0]) - float(start[0])
    pz = float(point[1]) - float(start[1])
    along = px * ux + pz * uz
    lateral = abs(px * -uz + pz * ux)
    half_width = float(road.endpoints[0].half_width_metres)
    return (
        -_SURFACE_MARGIN_METRES <= along <= length + _SURFACE_MARGIN_METRES
        and lateral <= half_width + _SURFACE_MARGIN_METRES
    )


def _covered_by_other_paved_surface(issue, roads, terrain=None) -> bool:
    if issue.category not in _COVERABLE_CATEGORIES or len(issue.object_ids) != 2:
        return False

    road_by_id = {int(road.object_id): road for road in roads}
    involved = [road_by_id.get(int(object_id)) for object_id in issue.object_ids]
    if any(road is None for road in involved):
        return False
    first_road, second_road = involved
    if first_road.family != second_road.family or first_road.family not in _PAVED_FAMILIES:
        return False

    point = (float(issue.x), float(issue.z))
    first = _nearest_endpoint(first_road, point)
    second = _nearest_endpoint(second_road, point)
    samples = _gap_samples(first, second)
    if not samples:
        return False
    wedge = _wedge._grass_wedge_geometry(first, second)
    terrain_sensitive = set()
    if wedge is not None:
        terrain_sensitive.update((wedge[3], wedge[4]))

    minimum_y = min(float(first_road.y), float(second_road.y)) - _VERTICAL_MARGIN_METRES
    maximum_y = max(float(first_road.y), float(second_road.y)) + _VERTICAL_MARGIN_METRES
    candidates = tuple(
        road
        for road in roads
        if (
            int(road.object_id) not in issue.object_ids
            and road.family == first_road.family
            and road.kind in {"straight", "paved_fill", "paved_miter", "paved_wedge"}
            and (
                minimum_y <= float(road.y) <= maximum_y
                or (
                    road.kind == "paved_wedge"
                    and minimum_y <= float(road.y)
                    <= max(float(first_road.y), float(second_road.y))
                    + _WEDGE_VERTICAL_MARGIN_METRES
                )
            )
        )
    )
    if not candidates:
        return False

    # Allow the covered area to be the union of several same-family pieces. This
    # matches what the renderer sees while still requiring every sampled point
    # across both open road edges to have actual asphalt underneath it.
    return all(
        any(
            _straight_contains(road, sample)
            and (
                sample not in terrain_sensitive
                or _visibly_above_terrain(road, sample, terrain)
            )
            for road in candidates
        )
        for sample in samples
    )


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
        raise RuntimeError("road inspector surface-coverage policy is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    terrain = _terrain_context(input_path)
    remaining = tuple(
        issue
        for issue in result.issues
        if not _covered_by_other_paved_surface(
            issue,
            result.road_objects,
            terrain,
        )
    )
    if len(remaining) == len(result.issues):
        return result
    return replace(result, issues=_core._number_issues(remaining))


def install() -> None:
    """Install the final read-only visual-coverage filter."""

    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
