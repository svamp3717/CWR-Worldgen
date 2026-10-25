# SPDX-License-Identifier: GPL-3.0-or-later
"""CWR-CE native, model-free OSM-style road way export.

The engine's feature/osm-style-road-ways reader expects local X/Z coordinates,
shared integer node IDs, and a surface *texture*, not a P3D model name.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from .osm import (
    BboxProjection,
    OsmDataset,
    PEDESTRIAN_ONLY_HIGHWAYS,
    road_is_dirt,
    road_is_gravel,
    road_is_supported,
    road_width_metres,
)

# Hard limits in CWR-CE RoadNet::ReadWayDefinitions / LoadWays.
_MAX_NODES = 100_000
_MAX_WAYS = 10_000
_MAX_BYTES = 8 * 1024 * 1024
_MAX_LINE = 8192
_MAX_NODES_PER_WAY = 512  # comfortably below the engine's 4096-node limit
_COORD_DECIMALS = 4


def is_osm_ways_profile(profile: str) -> bool:
    return profile == "cwr-ce-ways"


def _surface(tags: Mapping[str, str]) -> str:
    if tags.get("highway", "").casefold() in PEDESTRIAN_ONLY_HIGHWAYS:
        return "dirt"
    if road_is_gravel(tags):
        return "gravel"
    return "dirt" if road_is_dirt(tags) else "paved"


def _width(tags: Mapping[str, str]) -> float:
    # The normalized layer exposes a width in metres as "width". Original OSM
    # input may provide width=5.5, but free-form strings are not reliably units.
    for key in ("width", "width_m"):
        try:
            value = float(tags.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0.5 <= value <= 50.0:
            return value
    return min(50.0, max(0.5, road_width_metres(tags)))


def _clip(
    a: tuple[float, float], b: tuple[float, float], limit: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky clipping, keeping coordinates strictly inside the WRP."""
    x0, z0 = a
    dx, dz = b[0] - x0, b[1] - z0
    if not all(map(math.isfinite, (x0, z0, dx, dz))):
        return None
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, limit - x0),
                 (-dz, z0), (dz, limit - z0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                lower = max(lower, t)
            else:
                upper = min(upper, t)
            if lower > upper:
                return None
    start = (round(max(0.0, min(limit, x0 + lower * dx)), _COORD_DECIMALS),
             round(max(0.0, min(limit, z0 + lower * dz)), _COORD_DECIMALS))
    end = (round(max(0.0, min(limit, x0 + upper * dx)), _COORD_DECIMALS),
           round(max(0.0, min(limit, z0 + upper * dz)), _COORD_DECIMALS))
    if math.dist(start, end) < 0.001:
        return None
    return start, end


def _runs(
    points: Sequence[tuple[float, float]], limit: float,
) -> list[list[tuple[float, float]]]:
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for a, b in zip(points, points[1:]):
        clipped = _clip(a, b, limit)
        if clipped is None:
            if len(current) >= 2:
                runs.append(current)
            current = []
            continue
        start, end = clipped
        if current and current[-1] != start:
            if len(current) >= 2:
                runs.append(current)
            current = []
        if not current:
            current.append(start)
        if current[-1] != end:
            current.append(end)
    if len(current) >= 2:
        runs.append(current)
    return runs



# Conservative limits keep the new curve close to the graded OSM road
# corridor, rather than letting a cosmetic improvement cut across a field.
_MIN_LEG_METRES = 18.0
_MIN_BEND_DEGREES = 25.0
_MAX_BEND_DEGREES = 135.0
_MAX_TANGENT_METRES = 18.0
_JUNCTION_CLEARANCE_METRES = 24.0
_ROAD_SHOULDER_CLEARANCE_METRES = 3.0
_BUILDING_CLEARANCE_METRES = 4.0
_URBAN_CLEARANCE_METRES = 30.0
_TOWN_RADII_METRES = {
    "city": 850.0,
    "town": 550.0,
    "suburb": 350.0,
    "quarter": 250.0,
    "neighbourhood": 250.0,
    "village": 280.0,
    "hamlet": 160.0,
}


def _point_key(point: tuple[float, float]) -> tuple[int, int]:
    """Count junctions by position without conflating distinct road nodes."""
    return (round(point[0] * 10), round(point[1] * 10))


def _polygon_shapes(features: Sequence[object], projection: BboxProjection):
    """Project an OSM polygon collection to world X/Z, preserving holes."""
    for feature in features:
        for item in feature.polygons:
            outer = [projection.to_world(point) for point in item.outer]
            if len(outer) < 3:
                continue
            holes = [
                [projection.to_world(point) for point in ring]
                for ring in item.holes if len(ring) >= 3
            ]
            polygon = Polygon(outer, holes)
            if not polygon.is_empty and polygon.is_valid and polygon.area > 0:
                yield polygon


class _BendClearance:
    """Spatial index used only by the opt-in CWR-CE ways writer."""

    def __init__(
        self,
        dataset: OsmDataset,
        projection: BboxProjection,
        source_runs: Sequence[tuple[int, Mapping[str, str], float, list[tuple[float, float]]]],
    ) -> None:
        self.limit = projection.world_size - 0.01
        self.road_shapes = [LineString(run) for _, _, _, run in source_runs]
        self.road_owners = [owner for owner, _, _, _ in source_runs]
        self.road_half_widths = [width * 0.5 for _, _, width, _ in source_runs]
        self.roads = STRtree(self.road_shapes) if self.road_shapes else None

        # A corner shared by two distinct OSM ways is a graph vertex. Moving
        # it would split the RoadNet; protect its whole approach as well.
        owners_by_point: dict[tuple[int, int], set[int]] = defaultdict(set)
        point_coords: dict[tuple[int, int], tuple[float, float]] = {}
        for owner, _, _, run in source_runs:
            for p in run:
                key = _point_key(p)
                owners_by_point[key].add(owner)
                point_coords[key] = p
        self.shared = {
            key for key, owners in owners_by_point.items() if len(owners) > 1
        }
        shared_points = [Point(*point_coords[key]) for key in sorted(self.shared)]
        self.junctions = STRtree(shared_points) if shared_points else None
        self.junction_points = shared_points

        settlement_shapes = []
        for place in getattr(dataset, "places", ()):
            place_type = place.tags.get("place", "").strip().casefold()
            radius = _TOWN_RADII_METRES.get(place_type, 0.0)
            if radius:
                x, z = projection.to_world(place.point)
                settlement_shapes.append(Point(x, z).buffer(radius))
        for poly in _polygon_shapes(getattr(dataset, "urban", ()), projection):
            settlement_shapes.append(poly.buffer(_URBAN_CLEARANCE_METRES))
        for poly in _polygon_shapes(getattr(dataset, "place_areas", ()), projection):
            settlement_shapes.append(poly.buffer(_URBAN_CLEARANCE_METRES))
        self.settlement_shapes = settlement_shapes
        self.settlements = STRtree(settlement_shapes) if settlement_shapes else None

        obstacles = [
            poly.buffer(_BUILDING_CLEARANCE_METRES)
            for poly in _polygon_shapes(
                getattr(dataset, "building_polygons", ()), projection
            )
        ]
        for building in getattr(dataset, "building_points", ()):
            obstacles.append(
                Point(*projection.to_world(building.point)).buffer(
                    _BUILDING_CLEARANCE_METRES + 3.0
                )
            )
        # Ribbon geometry must not drift onto a lake or an unmodelled river.
        obstacles.extend(_polygon_shapes(getattr(dataset, "water", ()), projection))
        self.obstacle_shapes = obstacles
        self.obstacles = STRtree(obstacles) if obstacles else None

    @staticmethod
    def _intersects(tree, shapes, query) -> bool:
        return tree is not None and any(
            shapes[int(index)].intersects(query) for index in tree.query(query)
        )

    def allows(
        self,
        owner: int,
        corner: tuple[float, float],
        curve: Sequence[tuple[float, float]],
        width: float,
    ) -> bool:
        if _point_key(corner) in self.shared:
            return False
        curve_line = LineString(curve)
        footprint = curve_line.buffer(width * 0.5 + _ROAD_SHOULDER_CLEARANCE_METRES)
        minx, minz, maxx, maxz = footprint.bounds
        if minx < 0 or minz < 0 or maxx >= self.limit or maxz >= self.limit:
            return False

        if self.junctions is not None:
            vicinity = Point(*corner).buffer(_JUNCTION_CLEARANCE_METRES)
            if self._intersects(self.junctions, self.junction_points, vicinity):
                return False
        if self._intersects(self.settlements, self.settlement_shapes, footprint):
            return False
        if self._intersects(self.obstacles, self.obstacle_shapes, footprint):
            return False
        if self.roads is not None:
            # Query wide enough for the engine's maximum 50 m road width.
            nearby = curve_line.buffer(width * 0.5 + 28.0)
            for found in self.roads.query(nearby):
                index = int(found)
                if self.road_owners[index] == owner:
                    continue
                if (
                    curve_line.distance(self.road_shapes[index])
                    <= width * 0.5 + self.road_half_widths[index]
                    + _ROAD_SHOULDER_CLEARANCE_METRES
                ):
                    return False
        return True


def _round_rural_bends(
    run: Sequence[tuple[float, float]],
    *,
    owner: int,
    width: float,
    clearance: _BendClearance,
) -> tuple[list[tuple[float, float]], int]:
    """Replace eligible sharp corners with tangent-continuous quadratic arcs.

    Endpoints, shared junctions, urban bends, and constrained geometry stay
    exactly on their original OSM positions. Adjacent arcs cannot overlap
    because the trim is at most 30% of either source segment.
    """
    if len(run) < 3:
        return list(run), 0

    rounded: dict[int, list[tuple[float, float]]] = {}
    for i in range(1, len(run) - 1):
        a, b, c = run[i - 1:i + 2]
        before = math.dist(a, b)
        after = math.dist(b, c)
        if min(before, after) < _MIN_LEG_METRES:
            continue
        incoming = ((b[0] - a[0]) / before, (b[1] - a[1]) / before)
        outgoing = ((c[0] - b[0]) / after, (c[1] - b[1]) / after)
        dot = max(-1.0, min(1.0, incoming[0] * outgoing[0] + incoming[1] * outgoing[1]))
        angle = math.degrees(math.acos(dot))
        if not _MIN_BEND_DEGREES <= angle <= _MAX_BEND_DEGREES:
            continue

        tangent = min(_MAX_TANGENT_METRES, before * 0.3, after * 0.3)
        if tangent < 5.0:
            continue
        entry = (b[0] - incoming[0] * tangent, b[1] - incoming[1] * tangent)
        exit = (b[0] + outgoing[0] * tangent, b[1] + outgoing[1] * tangent)
        samples = max(5, math.ceil(tangent / 2.5))
        arc = []
        for j in range(samples + 1):
            t = j / samples
            x = (1-t)**2 * entry[0] + 2*(1-t)*t*b[0] + t*t*exit[0]
            z = (1-t)**2 * entry[1] + 2*(1-t)*t*b[1] + t*t*exit[1]
            point = (round(x, _COORD_DECIMALS), round(z, _COORD_DECIMALS))
            if not arc or point != arc[-1]:
                arc.append(point)
        if len(arc) <= 2 or not clearance.allows(owner, b, arc, width):
            continue
        rounded[i] = arc

    if not rounded:
        return list(run), 0
    result = [run[0]]
    for i in range(1, len(run) - 1):
        if i in rounded:
            for point in rounded[i]:
                if point != result[-1]:
                    result.append(point)
        elif run[i] != result[-1]:
            result.append(run[i])
    if run[-1] != result[-1]:
        result.append(run[-1])
    return result, len(rounded)


def write_osm_road_ways(
    path: Path,
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    include_minor_roads: bool,
    surface_textures: Mapping[str, str],
) -> dict[str, int | str]:
    """Export terrain-following non-bridge roads as a deterministic sidecar.

    Point identity comes from the normalized projected coordinates, so roads
    snapped to the same at-grade junction reuse a single node ID. Distinct
    coordinates (including visually crossing lines) stay disconnected.
    """
    textures: dict[str, str] = {}
    for surface in ("paved", "dirt", "gravel"):
        texture = str(surface_textures.get(surface) or "").replace("/", "\\")
        if (not texture or "#" in texture or any(char.isspace() for char in texture)
                or texture.startswith("\\") or texture.endswith("\\")
                or not texture.casefold().endswith((".paa", ".pac"))):
            raise ValueError(
                f"OSM ways require a valid single-token Roadway texture for {surface}: {texture!r}"
            )
        textures[surface] = texture

    limit = projection.world_size - 0.01
    if limit <= 0:
        raise ValueError("OSM ways require positive terrain dimensions")
    nodes: dict[tuple[float, float], int] = {}
    lines: list[str] = []
    skipped_elevated = 0
    source_runs: list[tuple[int, Mapping[str, str], float, list[tuple[float, float]]]] = []
    for owner, road in enumerate(sorted(dataset.roads, key=lambda item: item.osm_key)):
        tags = road.tags
        if not road_is_supported(tags, include_minor=include_minor_roads):
            continue
        # The native ribbon follows terrain; bridges keep their old deck models.
        if tags.get("bridge", "").strip().casefold() not in {"", "no", "false", "0"}:
            skipped_elevated += 1
            continue
        points = tuple(projection.to_world(point) for point in road.points)
        for run in _runs(points, limit):
            source_runs.append((owner, tags, _width(tags), run))

    clearance = _BendClearance(dataset, projection, source_runs)
    smoothed_bends = 0
    for owner, tags, width, original in source_runs:
        run, rounded = _round_rural_bends(
            original, owner=owner, width=width, clearance=clearance,
        )
        smoothed_bends += rounded
        for index in range(0, len(run) - 1, _MAX_NODES_PER_WAY - 1):
            chunk = run[index:index + _MAX_NODES_PER_WAY]
            if len(chunk) < 2:
                continue
            ids: list[int] = []
            for point in chunk:
                if point not in nodes:
                    nodes[point] = len(nodes) + 1
                    if len(nodes) > _MAX_NODES:
                        raise ValueError(
                            "CWR-CE .ways supports at most 100,000 nodes; "
                            "reduce the generated area or road detail"
                        )
                ids.append(nodes[point])
            if len(lines) >= _MAX_WAYS:
                raise ValueError(
                    "CWR-CE .ways supports at most 10,000 ways; "
                    "reduce the generated area or road detail"
                )
            lines.append(
                f"way {len(lines) + 1} {_width(tags):.3f} {textures[_surface(tags)]} "
                + " ".join(map(str, ids))
            )
    # Emit nodes in ID order. Both record types may occur in either order.
    node_lines = [
        f"node {node_id} {x:.4f} {z:.4f}"
        for (x, z), node_id in nodes.items()
    ]
    content = (
        "# CWR-CE native OSM-style roads; local terrain X/Z in metres\n"
        + "\n".join(node_lines + lines) + "\n"
    ).encode("utf-8")
    if len(content) > _MAX_BYTES or any(
        len(line.encode("utf-8")) > _MAX_LINE for line in node_lines + lines
    ):
        raise ValueError(
            "CWR-CE .ways size/line limit exceeded; reduce road detail or map area"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.name,
        "nodes": len(nodes),
        "ways": len(lines),
        "excluded_bridge_features": skipped_elevated,
        "smoothed_rural_bends": smoothed_bends,
        "format": "cwr-ce-osm-ways-v1",
    }
