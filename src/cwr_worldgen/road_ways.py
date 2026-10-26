# SPDX-License-Identifier: GPL-3.0-or-later
"""CWR-CE native, model-free OSM-style road way export.

The engine's feature/osm-style-road-ways reader expects local X/Z coordinates,
shared integer node IDs, and a texture or optional P3D material donor.
"""
from __future__ import annotations

import math
from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring
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

# These three real stock CWA skins trigger the isolated engine's matching
# sil25/ces25/asf25 donor UV and lighting fallback. Older data\*.pac guesses
# do not opt into the same material handling.
STOCK_WAY_TEXTURES = {
    "paved": r"o\road\sil_new.paa",
    "dirt": r"o\road\ces_hned.paa",
    "gravel": r"o\road\asf_new.paa",
}


def is_osm_ways_profile(profile: str) -> bool:
    return profile == "cwr-ce-ways"


def _non_ground(tags: Mapping[str, str]) -> bool:
    return any(
        str(tags.get(key, "")).strip().casefold() not in {"", "no", "false", "0"}
        for key in ("bridge", "tunnel")
    )


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
_MAX_TANGENT_METRES = 38.0
_MIN_CURVE_RADIUS_METRES = 12.0
_MAX_CORRIDOR_DEVIATION_METRES = 14.0
# A gently wandering centerline on long *already straight* rural stretches.
# Maximum displacements are small because the terrain grader follows OSM.
_MIN_FLOW_LENGTH_METRES = 140.0
_FLOW_AFTER_CURVE_LENGTH_METRES = 250.0
_FLOW_SAMPLE_METRES = 12.0
_MAX_FLOW_OFFSET_METRES = 4.0
# Let terrain-bound ways reach just under the deck edge to hide visual seams.
_BRIDGE_DECK_SEAM_OVERLAP_METRES = 0.7
_BRIDGE_DECK_MATCH_DISTANCE_METRES = 7.0
_BRIDGE_MIN_APPROACH_METRES = 0.05
_FLOW_AMPLITUDES_METRES = {
    "secondary": 1.4,
    "tertiary": 2.4,
    "unclassified": 3.1,
    "residential": 2.0,
    "living_street": 1.2,
    "service": 2.2,
    "road": 2.6,
    "track": 3.8,
}
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



# Consecutive roads with the same output width and skin can use a single
# CWR-CE way. Combining before smoothing also lets bends across OSM way
# boundaries become true tangent-continuous curves.
_RoadRun = tuple[int, Mapping[str, str], float, list[tuple[float, float]]]


def _merge_compatible_runs(
    source_runs: Sequence[_RoadRun],
    protected_points: set[tuple[int, int]] | None = None,
) -> list[_RoadRun]:
    """Coalesce degree-two, exactly shared endpoints without changing topology.

    Never guess a connection from proximity, and never merge across a crossing,
    an interior junction, a bridge endpoint, or a surface/width transition.
    Cyclic chains stop before closing to preserve distinct way endpoints.
    """
    ends: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
    interior: set[tuple[int, int]] = set(protected_points or ())
    for index, (_, _, _, points) in enumerate(source_runs):
        ends[_point_key(points[0])].append((index, True))
        ends[_point_key(points[-1])].append((index, False))
        interior.update(_point_key(p) for p in points[1:-1])
    protected = interior

    def signature(index: int) -> tuple[str, float]:
        _, tags, width, _ = source_runs[index]
        return _surface(tags), round(width, 3)

    visited: set[int] = set()
    merged: list[_RoadRun] = []
    for seed, (_, tags, width, points) in enumerate(source_runs):
        if seed in visited:
            continue
        visited.add(seed)
        chain = list(points)
        for left in (False, True):
            while True:
                joint = chain[0] if left else chain[-1]
                key = _point_key(joint)
                choices = ends.get(key, ())
                if key in protected or len(choices) != 2:
                    break
                neighbor_indices = [i for i, _ in choices if i not in visited]
                if len(neighbor_indices) != 1:
                    break
                candidate = neighbor_indices[0]
                if signature(seed) != signature(candidate):
                    break
                candidate_points = source_runs[candidate][3]
                if left:
                    if candidate_points[-1] == joint:
                        addition = candidate_points[:-1]
                    elif candidate_points[0] == joint:
                        addition = list(reversed(candidate_points[1:]))
                    else:
                        break
                    proposed = list(addition) + chain
                else:
                    if candidate_points[0] == joint:
                        addition = candidate_points[1:]
                    elif candidate_points[-1] == joint:
                        addition = list(reversed(candidate_points[:-1]))
                    else:
                        break
                    proposed = chain + list(addition)
                if proposed[0] == proposed[-1]:
                    break
                visited.add(candidate)
                chain = proposed
        merged.append((seed, tags, width, chain))
    return merged


class _BendClearance:
    """Spatial index used only by the opt-in CWR-CE ways writer."""

    def __init__(
        self,
        dataset: OsmDataset,
        projection: BboxProjection,
        source_runs: Sequence[tuple[int, Mapping[str, str], float, list[tuple[float, float]]]],
    ) -> None:
        self.limit = projection.world_size - 0.01
        # Omitted bridge and tunnel geometry still defines approaches that
        # must not be moved by native-road curve rounding.
        non_ground_runs = []
        for index, road in enumerate(getattr(dataset, "roads", ())):
            if not _non_ground(road.tags):
                continue
            points = tuple(projection.to_world(point) for point in road.points)
            for run in _runs(points, self.limit):
                non_ground_runs.append((-(index + 1), road.tags, _width(road.tags), run))
        clearance_runs = (*source_runs, *non_ground_runs)
        self.omitted_approaches = {
            _point_key(point)
            for _, _, _, points in non_ground_runs
            for point in points
        }
        self.road_shapes = [LineString(run) for _, _, _, run in clearance_runs]
        self.road_owners = [owner for owner, _, _, _ in clearance_runs]
        self.road_half_widths = [width * 0.5 for _, _, width, _ in clearance_runs]
        self.roads = STRtree(self.road_shapes) if self.road_shapes else None

        # A corner shared by two distinct OSM ways is a graph vertex. Moving
        # it would split the RoadNet; protect its whole approach as well.
        owners_by_point: dict[tuple[int, int], set[int]] = defaultdict(set)
        point_coords: dict[tuple[int, int], tuple[float, float]] = {}
        for owner, _, _, run in clearance_runs:
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
        for stream in getattr(dataset, "watercourses", ()):
            points = [projection.to_world(p) for p in stream.points]
            if len(points) >= 2:
                obstacles.append(LineString(points).buffer(4.0))
        self.obstacle_shapes = obstacles
        self.obstacles = STRtree(obstacles) if obstacles else None

    @staticmethod
    def _intersects(tree, shapes, query) -> bool:
        return tree is not None and any(
            shapes[int(index)].intersects(query) for index in tree.query(query)
        )

    def allows_flow(
        self,
        owner: int,
        original: Sequence[tuple[float, float]],
        candidate: Sequence[tuple[float, float]],
        width: float,
        before: Sequence[tuple[float, float]],
        after: Sequence[tuple[float, float]],
    ) -> bool:
        """Check a long rural S-curve, excluding only its fixed end approaches."""
        path = LineString(candidate)
        old_path = LineString(original)
        if (
            not path.is_simple
            or path.hausdorff_distance(old_path) > _MAX_FLOW_OFFSET_METRES + 0.25
        ):
            return False
        footprint = path.buffer(width * 0.5 + _ROAD_SHOULDER_CLEARANCE_METRES)
        minx, minz, maxx, maxz = footprint.bounds
        if minx < 0 or minz < 0 or maxx >= self.limit or maxz >= self.limit:
            return False
        if self._intersects(self.settlements, self.settlement_shapes, footprint):
            return False
        if self._intersects(self.obstacles, self.obstacle_shapes, footprint):
            return False

        # At shared junctions, the curve meets the source geometry exactly.
        # Trim the first/last 24 m only for network-proximity checks, so a
        # connected side road at an endpoint doesn't reject the whole run.
        from shapely.ops import substring
        trimmed = substring(
            path, _JUNCTION_CLEARANCE_METRES,
            max(_JUNCTION_CLEARANCE_METRES + 1.0,
                path.length - _JUNCTION_CLEARANCE_METRES),
        )
        if self._intersects(
            self.junctions, self.junction_points,
            trimmed.buffer(_JUNCTION_CLEARANCE_METRES * 0.75),
        ):
            return False
        if self.roads is not None:
            search = trimmed.buffer(width * 0.5 + 28.0)
            for found in self.roads.query(search):
                index = int(found)
                if self.road_owners[index] == owner:
                    continue
                clearance = (
                    width * 0.5 + self.road_half_widths[index]
                    + _ROAD_SHOULDER_CLEARANCE_METRES
                )
                if trimmed.distance(self.road_shapes[index]) <= clearance:
                    return False
        # A distant section of the *same* road can also pass beside the
        # candidate on hairpins. These are not separate owners in the index.
        for remainder in (before, after):
            if len(remainder) >= 2 and trimmed.distance(
                LineString(remainder)
            ) <= width + _ROAD_SHOULDER_CLEARANCE_METRES:
                return False
        return True


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
    """Replace eligible sharp corners with approximately circular cubic arcs.

    Endpoints, shared junctions, urban bends, and constrained geometry stay
    exactly on their original OSM positions. Adjacent arcs cannot overlap
    because the trim is at most 38% of either source segment.
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

        # A circular-arc approximation with cubic Bezier controls avoids
        # the triangular-looking center of the older quadratic corner. The
        # radius grows on long open roads and shrinks on slower/short roads.
        half_angle = math.radians(angle) * 0.5
        tangent_of_half = math.tan(half_angle)
        # Give long open roads a wider flowing sweep; short approaches keep
        # a tight trim or their original bend when no safe radius fits.
        desired = max(
            12.0,
            8.0 + min(before, after) * 0.12
            + _MIN_CURVE_RADIUS_METRES * tangent_of_half,
        )
        tangent = min(_MAX_TANGENT_METRES, desired, before * 0.38, after * 0.38)
        radius = tangent / tangent_of_half
        if radius < _MIN_CURVE_RADIUS_METRES:
            continue
        entry = (b[0] - incoming[0] * tangent, b[1] - incoming[1] * tangent)
        exit = (b[0] + outgoing[0] * tangent, b[1] + outgoing[1] * tangent)
        # For a circular arc of angle phi, cubic control offset is
        # (4/3)*tan(phi/4)*R. It matches each straight segment's tangent.
        control = (4.0 / 3.0) * math.tan(half_angle * 0.5) * radius
        p1 = (entry[0] + incoming[0] * control,
              entry[1] + incoming[1] * control)
        p2 = (exit[0] - outgoing[0] * control,
              exit[1] - outgoing[1] * control)
        samples = max(6, math.ceil(2.0 * tangent / 3.0))
        arc: list[tuple[float, float]] = []
        for j in range(samples + 1):
            t = j / samples
            omt = 1.0 - t
            x = (omt**3 * entry[0] + 3*omt**2*t * p1[0]
                 + 3*omt*t*t * p2[0] + t**3 * exit[0])
            z = (omt**3 * entry[1] + 3*omt**2*t * p1[1]
                 + 3*omt*t*t * p2[1] + t**3 * exit[1])
            point = (round(x, _COORD_DECIMALS), round(z, _COORD_DECIMALS))
            if not arc or point != arc[-1]:
                arc.append(point)
        # Do not stray too far from the original terrain-graded road
        # corridor: the .ways profile still uses the original grade solver.
        original_path = LineString((a, b, c))
        if (
            len(arc) <= 2
            or any(
                original_path.distance(Point(*p)) > _MAX_CORRIDOR_DEVIATION_METRES
                for p in arc
            )
            or not clearance.allows(owner, b, arc, width)
        ):
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


def _flow_rural_straights(
    run: Sequence[tuple[float, float]],
    *,
    owner: int,
    tags: Mapping[str, str],
    width: float,
    clearance: _BendClearance,
) -> tuple[list[tuple[float, float]], int]:
    """Introduce gentle deterministic S-curves on long, open straight spans.

    Consecutive almost-collinear source edges are considered together.
    Endpoints and road junction vertices remain in place. A fade envelope
    gives the synthetic curve a zero offset and zero offset derivative at
    both ends, avoiding visible kinks when it rejoins the original road.
    """
    amplitude = _FLOW_AMPLITUDES_METRES.get(tags.get("highway", "").casefold(), 0.0)
    if amplitude <= 0 or len(run) < 2:
        return list(run), 0
    result = [run[0]]
    softened = 0
    i = 0
    while i + 1 < len(run):
        j = i + 1
        start = run[i]
        # Group short straight OSM edges, but never smooth a bend or consume
        # an interior vertex used by a different road.
        while j + 1 < len(run):
            a, b, c = run[j - 1], run[j], run[j + 1]
            if _point_key(b) in clearance.shared:
                break
            previous = math.dist(a, b)
            following = math.dist(b, c)
            if min(previous, following) < 0.001:
                break
            incoming = ((b[0] - a[0]) / previous, (b[1] - a[1]) / previous)
            outgoing = ((c[0] - b[0]) / following, (c[1] - b[1]) / following)
            if incoming[0] * outgoing[0] + incoming[1] * outgoing[1] < math.cos(
                math.radians(6.0)
            ):
                break
            chord = LineString((start, c))
            if chord.distance(Point(*b)) > 1.5:
                break
            j += 1

        original = list(run[i:j + 1])
        span = math.dist(start, original[-1])
        near_rounded_curve = (
            (i > 0 and math.dist(run[i - 1], start) < 10.0)
            or (j + 1 < len(run) and math.dist(run[j], run[j + 1]) < 10.0)
        )
        min_length = (
            _FLOW_AFTER_CURVE_LENGTH_METRES if near_rounded_curve
            else _MIN_FLOW_LENGTH_METRES
        )
        accepted = False
        if (
            span >= min_length
            and _point_key(start) not in clearance.omitted_approaches
            and _point_key(original[-1]) not in clearance.omitted_approaches
        ):
            end = original[-1]
            direction = ((end[0] - start[0]) / span,
                         (end[1] - start[1]) / span)
            normal = (-direction[1], direction[0])
            # Reverse-safe sign: reversing the segment also reverses its
            # normal and sin(2*pi*t), producing exactly the same curve.
            endpoints = sorted((_point_key(start), _point_key(end)))
            signature = repr((endpoints, round(span), _surface(tags))).encode()
            sign = 1.0 if hashlib.sha256(signature).digest()[0] & 1 else -1.0
            waves = min(3, max(1, round(span / 260.0)))
            count = max(12, math.ceil(span / _FLOW_SAMPLE_METRES))
            candidate = []
            for k in range(count + 1):
                t = k / count
                # sin(pi*t)^2 smoothly eases offsets and heading into the
                # original road at both ends. No random vertex jitter.
                offset = (
                    sign * amplitude
                    * math.sin(math.pi * t) ** 2
                    * math.sin(2.0 * math.pi * waves * t)
                )
                x = start[0] + (end[0] - start[0]) * t + normal[0] * offset
                z = start[1] + (end[1] - start[1]) * t + normal[1] * offset
                point = (round(x, _COORD_DECIMALS), round(z, _COORD_DECIMALS))
                if not candidate or point != candidate[-1]:
                    candidate.append(point)
            accepted = len(candidate) > 2 and clearance.allows_flow(
                owner, original, candidate, width,
                run[:i+1], run[j:],
            )
            if accepted:
                result.extend(candidate[1:])
                softened += 1
        if not accepted:
            result.extend(original[1:])
        i = j
    return result, softened


def _bridge_dry_approach_runs(
    run: Sequence[tuple[float, float]],
    bridge_axes: Sequence[LineString],
    *,
    limit: float,
) -> list[list[tuple[float, float]]]:
    """Split a bridge-tagged OSM road at the actual stock bridge deck ends.

    The engine's terrain-following .ways must not run over open water, where
    the elevated stock bridge P3D supplies the Roadway LOD itself. OSM often
    tags dry roads far beyond the water as bridge=yes; export both dry ends.
    """
    centerline = LineString(run)
    length = centerline.length
    if length <= _BRIDGE_MIN_APPROACH_METRES:
        return []
    deck_intervals: list[tuple[float, float]] = []
    for axis in bridge_axes:
        # Distance to both deck endpoints guards against stealing a nearby
        # unrelated bridge at an intersection or a parallel water crossing.
        start = Point(axis.coords[0])
        end = Point(axis.coords[-1])
        if (
            centerline.distance(start) > _BRIDGE_DECK_MATCH_DISTANCE_METRES
            or centerline.distance(end) > _BRIDGE_DECK_MATCH_DISTANCE_METRES
        ):
            continue
        begin = centerline.project(start)
        finish = centerline.project(end)
        lo, hi = sorted((begin, finish))
        # Require appreciable common along-road length to avoid cutting
        # across perpendicular stock bridges and accidental crossings.
        if hi - lo < max(5.0, axis.length * 0.55):
            continue
        deck_intervals.append((
            max(0.0, lo + _BRIDGE_DECK_SEAM_OVERLAP_METRES),
            min(length, hi - _BRIDGE_DECK_SEAM_OVERLAP_METRES),
        ))
    if not deck_intervals:
        # No rendered bridge deck was matched. Do not put a terrain-following
        # road over possible water, because that would create a submerged road.
        return []
    deck_intervals.sort()
    merged: list[list[float]] = []
    for lo, hi in deck_intervals:
        if lo >= hi:
            continue
        if merged and lo <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    approaches: list[list[tuple[float, float]]] = []
    cursor = 0.0
    for lo, hi in (*merged, [length, length]):
        if lo - cursor > _BRIDGE_MIN_APPROACH_METRES:
            line = substring(centerline, cursor, lo)
            if isinstance(line, LineString):
                points = [
                    (round(x, _COORD_DECIMALS), round(z, _COORD_DECIMALS))
                    for x, z in line.coords
                ]
                approaches.extend(_runs(points, limit))
        cursor = max(cursor, hi)
    return approaches


def _stock_bridge_axes(bridge_objects: Sequence[object]) -> list[LineString]:
    """Measure the Roadway endpoints, not the stock P3D's object origin."""
    # Deferred import avoids bridge_render_policy -> generator -> road_ways
    # initialization cycles. The placement policy also accounts for pitch.
    from .bridge_render_policy import _axis_endpoints, _is_stock_bridge

    axes = []
    for obj in bridge_objects:
        if not _is_stock_bridge(obj):
            continue
        ends = _axis_endpoints(obj)
        if math.dist(*ends) > 5.0:
            axes.append(LineString(ends))
    return axes


def write_osm_road_ways(
    path: Path,
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    include_minor_roads: bool,
    surface_textures: Mapping[str, str],
    surface_donors: Mapping[str, str] | None = None,
    bridge_objects: Sequence[object] = (),
) -> dict[str, object]:
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

    # The isolated CWR-CE engine accepts donor=P3D and reproduces its UVs,
    # overlays, normal orientation, and separate roadway-contact texture.
    # Retain texture-only compatibility when the model is not known to exist.
    sources: dict[str, str] = dict(textures)
    for surface, donor in (surface_donors or {}).items():
        if surface not in sources or not str(donor).strip():
            continue
        name = str(donor).strip().replace("/", "\\")
        if name.casefold().startswith("donor="):
            name = name[6:]
        if (
            "#" in name or any(ch.isspace() for ch in name)
            or name.startswith("\\") or not name.casefold().endswith(".p3d")
        ):
            raise ValueError(f"Invalid CWR-CE road donor for {surface}: {donor!r}")
        sources[surface] = "donor=" + name

    limit = projection.world_size - 0.01
    if limit <= 0:
        raise ValueError("OSM ways require positive terrain dimensions")
    nodes: dict[tuple[float, float], int] = {}
    lines: list[str] = []
    skipped_elevated = 0
    bridged_approaches = 0
    unmatched_bridge_features = 0
    stock_bridge_axes = _stock_bridge_axes(bridge_objects)
    source_runs: list[tuple[int, Mapping[str, str], float, list[tuple[float, float]]]] = []
    for owner, road in enumerate(sorted(dataset.roads, key=lambda item: item.osm_key)):
        tags = road.tags
        if not road_is_supported(tags, include_minor=include_minor_roads):
            continue
        # Stock bridge P3Ds provide elevated Roadway geometry over water.
        # OSM bridge=yes often covers long dry approach roads too, so export
        # those dry ends as native ways right up to the placed deck footprint.
        # Never synthesize a terrain road across the bridge's wet interior.
        bridge_tag = _non_ground(tags)
        points = tuple(projection.to_world(point) for point in road.points)
        for run in _runs(points, limit):
            if bridge_tag:
                skipped_elevated += 1
                approaches = _bridge_dry_approach_runs(
                    run, stock_bridge_axes, limit=limit,
                )
                if not approaches:
                    unmatched_bridge_features += 1
                bridged_approaches += len(approaches)
                for approach in approaches:
                    source_runs.append((owner, tags, _width(tags), approach))
            else:
                source_runs.append((owner, tags, _width(tags), run))

    original_run_count = len(source_runs)
    # Bridge/tunnel models are not native ribbons; preserve their approaches.
    non_ground_points = {
        _point_key(projection.to_world(p))
        for road in dataset.roads
        if _non_ground(road.tags)
        for p in road.points
    }
    source_runs = _merge_compatible_runs(source_runs, non_ground_points)
    clearance = _BendClearance(dataset, projection, source_runs)
    smoothed_bends = 0
    softened_rural_straights = 0
    for owner, tags, width, original in source_runs:
        run, rounded = _round_rural_bends(
            original, owner=owner, width=width, clearance=clearance,
        )
        run, softened = _flow_rural_straights(
            run, owner=owner, tags=tags, width=width, clearance=clearance,
        )
        smoothed_bends += rounded
        softened_rural_straights += softened
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
                f"way {len(lines) + 1} {width:.3f} {sources[_surface(tags)]} "
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
        "restored_bridge_approach_ways": bridged_approaches,
        "unmatched_bridge_features": unmatched_bridge_features,
        "stock_bridge_modules": len(stock_bridge_axes),
        "smoothed_rural_bends": smoothed_bends,
        "softened_rural_straights": softened_rural_straights,
        "merged_source_runs": original_run_count - len(source_runs),
        "merged_road_chains": len(source_runs),
        "surface_sources": sources,
        "format": "cwr-ce-osm-ways-v1",
    }
