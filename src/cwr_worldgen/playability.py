# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata
from typing import Callable, Mapping, Sequence

from .model import OsmSpec, PlayabilitySpec, WorldObject
from .procedural_infrastructure import (
    GENERATED_GRAVEL_SURFACE_CLEARANCE_METRES,
    GENERATED_GRAVEL_VISUAL_TOP_METRES,
    gravel_curve_model_path,
    gravel_junction_model_path,
    gravel_road_model_path,
    is_generated_gravel_junction_model,
    is_generated_gravel_road_model,
)
from .osm import (
    BboxProjection,
    OsmDataset,
    OsmLineFeature,
    OsmRaster,
    _maximum_polygon_elevation,
    _oriented_rectangle,
    road_bridge_crosses_ditch_only,
    road_is_dirt,
    road_is_gravel,
    road_model_for_tags,
    road_is_supported,
    road_width_metres,
)


@dataclass(frozen=True, slots=True)
class RoadFitReport:
    objects: tuple[WorldObject, ...]
    chain_count: int
    connection_count: int
    failed_connections: int
    maximum_connection_gap: float
    maximum_chain_gap: float
    truncated: bool
    trimmed_junctions: int = 0
    terrain_filled_junctions: int = 0
    skipped_short_runs: int = 0
    maximum_model_overlap_metres: float = 0.0
    maximum_junction_clearance_metres: float = 0.0
    maximum_terrain_patch_radius_metres: float = 0.0
    junction_cap_objects: int = 0
    short_piece_objects: int = 0
    maximum_endpoint_gap_metres: float = 0.0
    maximum_road_pitch_degrees: float = 0.0
    suppressed_degree_two_caps: int = 0
    complex_junctions_without_caps: int = 0
    road_connection_slot_risk_nodes: int = 0
    suppressed_nearby_hubs: int = 0


@dataclass(frozen=True, slots=True)
class TerrainGradeReport:
    elevations: tuple[float, ...]
    changed_cells: int
    road_seed_cells: int
    building_pad_cells: int
    maximum_cut: float
    maximum_fill: float
    maximum_road_slope_before_percent: float
    maximum_road_slope_after_percent: float
    building_roughness_before: float
    building_roughness_after: float


@dataclass(frozen=True, slots=True)
class TransitionReport:
    indices: tuple[int, ...]
    shoreline_cells: int
    softened_landuse_cells: int


@dataclass(frozen=True, slots=True)
class TownLocation:
    class_name: str
    name: str
    x: float
    z: float
    place_type: str
    osm_key: str


# Stock road meshes sit extremely close to the terrain. At mixed crossings the
# engine can therefore z-fight when two road families share the same height.
# Keep gravel slightly below ordinary asphalt and emit paved chains
# after unpaved chains so asphalt consistently wins both geometry and draw order.
_STOCK_ROAD_VERTICAL_OFFSET_METRES = 0.035
_STOCK_GRAVEL_VERTICAL_OFFSET_METRES = 0.018


def _road_surface_priority(tags: Mapping[str, str]) -> int:
    if not road_is_dirt(tags):
        return 2  # asphalt / paved
    if road_is_gravel(tags):
        return 1  # generated gravel family
    return 0  # dirt / earth


def _road_vertical_offset(tags: Mapping[str, str]) -> float:
    return (
        _STOCK_GRAVEL_VERTICAL_OFFSET_METRES
        if road_is_gravel(tags)
        else _STOCK_ROAD_VERTICAL_OFFSET_METRES
    )


def _road_is_explicit_bridge(tags: Mapping[str, str]) -> bool:
    bridge = str(tags.get("bridge", "")).strip().casefold()
    if bridge not in {"", "no", "false", "0", "none"}:
        return True
    if str(tags.get("man_made", "")).strip().casefold() == "bridge":
        return True
    return str(tags.get("special", "")).strip().casefold() == "bridge"


def _sample_elevation(elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float) -> float:
    fx = max(0.0, min(cells - 1.0, x / cell_size))
    fz = max(0.0, min(cells - 1.0, z / cell_size))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(cells - 1, x0 + 1)
    z1 = min(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    a = elevations[z0 * cells + x0] * (1.0 - tx) + elevations[z0 * cells + x1] * tx
    b = elevations[z1 * cells + x0] * (1.0 - tx) + elevations[z1 * cells + x1] * tx
    return a * (1.0 - tz) + b * tz


def _point_at_distance(points: Sequence[tuple[float, float]], distance: float) -> tuple[float, float, float]:
    remaining = distance
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = math.hypot(dx, dz)
        if length <= 1e-9:
            continue
        if remaining <= length:
            fraction = remaining / length
            return (
                start[0] + dx * fraction,
                start[1] + dz * fraction,
                math.degrees(math.atan2(dx, dz)) % 360.0,
            )
        remaining -= length
    start, end = points[-2], points[-1]
    return end[0], end[1], math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0


def _model_endpoint(obj: WorldObject, forward: bool, length: float) -> tuple[float, float]:
    angle = math.radians(obj.heading_degrees)
    sign = 1.0 if forward else -1.0
    return (
        obj.x + math.sin(angle) * length * 0.5 * sign,
        obj.z + math.cos(angle) * length * 0.5 * sign,
    )


def _model_axis(obj: WorldObject, length: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return _model_endpoint(obj, False, length), _model_endpoint(obj, True, length)


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    denominator = dx * dx + dz * dz
    if denominator <= 1e-12:
        return math.dist(point, start)
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz) / denominator
    fraction = max(0.0, min(1.0, fraction))
    nearest = (start[0] + dx * fraction, start[1] + dz * fraction)
    return math.dist(point, nearest)


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], point: tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) - 1e-8 <= point[0] <= max(a[0], b[0]) + 1e-8
        and min(a[1], b[1]) - 1e-8 <= point[1] <= max(a[1], b[1]) + 1e-8
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    epsilon = 1e-8
    if ((ab_c > epsilon and ab_d < -epsilon) or (ab_c < -epsilon and ab_d > epsilon)) and (
        (cd_a > epsilon and cd_b < -epsilon) or (cd_a < -epsilon and cd_b > epsilon)
    ):
        return True
    return (
        (abs(ab_c) <= epsilon and _on_segment(a, b, c))
        or (abs(ab_d) <= epsilon and _on_segment(a, b, d))
        or (abs(cd_a) <= epsilon and _on_segment(c, d, a))
        or (abs(cd_b) <= epsilon and _on_segment(c, d, b))
    )


def _segment_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _turn_degrees(
    previous: tuple[float, float],
    current: tuple[float, float],
    following: tuple[float, float],
) -> float:
    incoming = (current[0] - previous[0], current[1] - previous[1])
    outgoing = (following[0] - current[0], following[1] - current[1])
    incoming_length = math.hypot(*incoming)
    outgoing_length = math.hypot(*outgoing)
    if incoming_length <= 1e-9 or outgoing_length <= 1e-9:
        return 0.0
    cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (incoming_length * outgoing_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _split_road_runs(
    points: Sequence[tuple[float, float]],
    *,
    maximum_turn_degrees: float = 35.0,
) -> list[list[tuple[float, float]]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or math.dist(point, cleaned[-1]) > 0.05:
            cleaned.append(point)
    if len(cleaned) < 2:
        return []
    runs: list[list[tuple[float, float]]] = []
    current = [cleaned[0]]
    for index in range(1, len(cleaned) - 1):
        point = cleaned[index]
        current.append(point)
        if _turn_degrees(cleaned[index - 1], point, cleaned[index + 1]) >= maximum_turn_degrees:
            if len(current) >= 2:
                runs.append(current)
            current = [point]
    current.append(cleaned[-1])
    if len(current) >= 2:
        runs.append(current)
    return runs


def _run_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(math.dist(start, end) for start, end in zip(points, points[1:]))


@dataclass(frozen=True, slots=True)
class _PolylineMeasure:
    """A polyline with one precomputed cumulative-distance table.

    Road fitting samples each run at two endpoints per stock piece. Rewalking
    the entire point list for every endpoint made long, detailed roads approach
    quadratic behaviour. This table makes each sample a binary search.
    """

    points: tuple[tuple[float, float], ...]
    cumulative: tuple[float, ...]
    total: float

    @classmethod
    def create(cls, points: Sequence[tuple[float, float]]) -> "_PolylineMeasure":
        frozen = tuple(points)
        if len(frozen) < 2:
            raise ValueError("road polyline must contain at least two points")
        cumulative = [0.0]
        for start, end in zip(frozen, frozen[1:]):
            cumulative.append(cumulative[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
        return cls(frozen, tuple(cumulative), cumulative[-1])

    def point(self, distance: float) -> tuple[float, float, float]:
        if distance < 0.0:
            start, end = self.points[0], self.points[1]
            dx, dz = end[0] - start[0], end[1] - start[1]
            length = max(1e-9, math.hypot(dx, dz))
            return (
                start[0] + dx / length * distance,
                start[1] + dz / length * distance,
                math.degrees(math.atan2(dx, dz)) % 360.0,
            )
        if distance > self.total:
            start, end = self.points[-2], self.points[-1]
            dx, dz = end[0] - start[0], end[1] - start[1]
            length = max(1e-9, math.hypot(dx, dz))
            excess = distance - self.total
            return (
                end[0] + dx / length * excess,
                end[1] + dz / length * excess,
                math.degrees(math.atan2(dx, dz)) % 360.0,
            )
        segment = min(len(self.points) - 2, max(0, bisect_right(self.cumulative, distance) - 1))
        start, end = self.points[segment], self.points[segment + 1]
        segment_start = self.cumulative[segment]
        length = max(1e-9, self.cumulative[segment + 1] - segment_start)
        fraction = (distance - segment_start) / length
        dx, dz = end[0] - start[0], end[1] - start[1]
        return (
            start[0] + dx * fraction,
            start[1] + dz * fraction,
            math.degrees(math.atan2(dx, dz)) % 360.0,
        )

    def chord_endpoint(
        self,
        start_distance: float,
        chord_length: float,
        maximum_distance: float,
    ) -> tuple[float, float, float, float] | None:
        """Find the first forward point exactly ``chord_length`` metres away.

        Stock road models have a fixed straight axis even when the OSM centreline
        curves. Advancing by arc length therefore makes the sampled chord shorter
        than the model and forces adjacent pieces to overlap. Circle/segment
        intersection keeps every fitted model axis at its real length while still
        following the source polyline.
        """

        if chord_length <= 0.0 or maximum_distance <= start_distance + 1e-9:
            return None
        origin_x, origin_z, _ = self.point(start_distance)
        breakpoints = [start_distance]
        breakpoints.extend(
            value
            for value in self.cumulative
            if start_distance + 1e-9 < value < maximum_distance - 1e-9
        )
        breakpoints.append(maximum_distance)
        radius_squared = chord_length * chord_length
        for distance0, distance1 in zip(breakpoints, breakpoints[1:]):
            ax, az, _ = self.point(distance0)
            bx, bz, _ = self.point(distance1)
            vx, vz = bx - ax, bz - az
            denominator = vx * vx + vz * vz
            if denominator <= 1e-12:
                continue
            ox, oz = ax - origin_x, az - origin_z
            linear = 2.0 * (ox * vx + oz * vz)
            constant = ox * ox + oz * oz - radius_squared
            discriminant = linear * linear - 4.0 * denominator * constant
            if discriminant < -1e-8:
                continue
            root = math.sqrt(max(0.0, discriminant))
            fractions = sorted((
                (-linear - root) / (2.0 * denominator),
                (-linear + root) / (2.0 * denominator),
            ))
            for fraction in fractions:
                if fraction < -1e-8 or fraction > 1.0 + 1e-8:
                    continue
                fraction = max(0.0, min(1.0, fraction))
                distance = distance0 + (distance1 - distance0) * fraction
                if distance <= start_distance + 1e-7:
                    continue
                x = ax + vx * fraction
                z = az + vz * fraction
                heading = math.degrees(math.atan2(x - origin_x, z - origin_z)) % 360.0
                return distance, x, z, heading
        return None

    def maximum_chord_deviation(
        self,
        start_distance: float,
        end_distance: float,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        """Return the largest centreline departure from a fitted straight chord.

        Only polyline breakpoints inside the span need to be tested because the
        distance from a straight source segment to the fitted chord reaches its
        maximum at one of that segment's endpoints.
        """

        maximum = 0.0
        for distance, point in zip(self.cumulative, self.points):
            if start_distance + 1e-7 < distance < end_distance - 1e-7:
                maximum = max(maximum, _point_segment_distance(point, start, end))
        return maximum


def _road_node_key(point: tuple[float, float], quantum: float = 0.10) -> tuple[int, int]:
    return int(round(point[0] / quantum)), int(round(point[1] / quantum))


def _clean_road_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or math.dist(point, cleaned[-1]) > 0.05:
            cleaned.append(point)
    return cleaned


def _road_node_degrees(
    projected_features: Sequence[tuple[OsmLineFeature, str, list[tuple[float, float]], float, float]],
) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for _feature, _model, points, _width, _clearance in projected_features:
        for start, end in zip(points, points[1:]):
            if math.dist(start, end) <= 0.05:
                continue
            for point in (start, end):
                key = _road_node_key(point)
                degrees[key] = degrees.get(key, 0) + 1
    return degrees


def _split_visual_road_runs(
    points: Sequence[tuple[float, float]],
    node_degrees: Mapping[tuple[int, int], int],
    clearance: float,
    *,
    maximum_turn_degrees: float = 20.0,
) -> list[tuple[list[tuple[float, float]], float, float, tuple[int, int], tuple[int, int]]]:
    """Split a road at junctions and sharp corners and reserve a texture-filled patch.

    The stock 25-metre road models are rectangular and coplanar.  Letting every
    branch extend through a T or X junction creates several overlapping meshes,
    which is exactly the asphalt lasagne visible in the user's screenshot.
    The central patch is instead represented by the already-generated road
    ground material, while model chains stop one half-width outside it.
    """

    cleaned = _clean_road_points(points)
    if len(cleaned) < 2:
        return []
    split_indices = {0, len(cleaned) - 1}
    visual_breaks: set[int] = set()
    for index in range(1, len(cleaned) - 1):
        key = _road_node_key(cleaned[index])
        if node_degrees.get(key, 0) >= 3 or _turn_degrees(cleaned[index - 1], cleaned[index], cleaned[index + 1]) >= maximum_turn_degrees:
            split_indices.add(index)
            visual_breaks.add(index)
    ordered = sorted(split_indices)
    result: list[tuple[list[tuple[float, float]], float, float, tuple[int, int], tuple[int, int]]] = []
    for start_index, end_index in zip(ordered, ordered[1:]):
        run = cleaned[start_index : end_index + 1]
        if len(run) < 2 or _run_length(run) < 0.5:
            continue
        start_key = _road_node_key(run[0])
        end_key = _road_node_key(run[-1])
        start_trim = clearance if start_index in visual_breaks or node_degrees.get(start_key, 0) >= 3 else 0.0
        end_trim = clearance if end_index in visual_breaks or node_degrees.get(end_key, 0) >= 3 else 0.0
        result.append((run, start_trim, end_trim, start_key, end_key))
    return result


def _fit_terrain_patch_road_objects(
    dataset: OsmDataset,
    projection: BboxProjection,
    elevations: Sequence[float],
    spec: PlayabilitySpec,
    *,
    starting_id: int = 1,
    progress_callback: Callable[[int, str], None] | None = None,
) -> RoadFitReport:
    objects: list[WorldObject] = []
    maximum_chain_gap = 0.0
    maximum_model_overlap = 0.0
    chain_count = 0
    truncated = False
    next_id = starting_id
    skipped_short_runs = 0
    maximum_clearance = 0.0

    projected_features: list[tuple[OsmLineFeature, str, list[tuple[float, float]], float, float]] = []
    total_roads = len(dataset.roads)
    if progress_callback is not None:
        progress_callback(0, f"Projecting {total_roads:,} normalized road lines")
    progress_step = max(1, total_roads // 20)
    for feature_index, feature in enumerate(dataset.roads, start=1):
        if progress_callback is not None and (feature_index == total_roads or feature_index % progress_step == 0):
            progress_callback(min(18, round(feature_index / max(1, total_roads) * 18)), f"Projected road lines {feature_index:,}/{total_roads:,}")
        if not road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = _clean_road_points([projection.to_world(point) for point in feature.points])
        if len(points) < 2:
            continue
        model = road_model_for_tags(spec, feature.tags)
        model_width = max(6.0, road_width_metres(feature.tags))
        clearance = model_width * 0.5 + 0.75
        projected_features.append((feature, model, points, model_width, clearance))
        maximum_clearance = max(maximum_clearance, clearance)

    if progress_callback is not None:
        progress_callback(20, f"Indexing junction degrees for {len(projected_features):,} supported roads")
    node_degrees = _road_node_degrees(projected_features)
    junction_keys = {key for key, degree in node_degrees.items() if degree >= 3}
    # Each record is the road-model axis nearest the central terrain patch and
    # the clearance already intentionally reserved around that node.
    endpoint_records: dict[
        tuple[int, int], list[tuple[tuple[tuple[float, float], tuple[float, float]], float]]
    ] = {}
    seam_records: dict[
        tuple[int, int], list[tuple[tuple[tuple[float, float], tuple[float, float]], float]]
    ] = {}

    fitting_step = max(1, len(projected_features) // 25)
    for feature_index, (feature, model, projected, model_width, clearance) in enumerate(projected_features, start=1):
        if progress_callback is not None and (feature_index == len(projected_features) or feature_index % fitting_step == 0):
            local = 25 + round(feature_index / max(1, len(projected_features)) * 70)
            progress_callback(min(95, local), f"Fitting terrain road lines {feature_index:,}/{len(projected_features):,}; {len(objects):,} objects")
        runs = _split_visual_road_runs(projected, node_degrees, clearance)
        for run, start_trim, end_trim, start_key, end_key in runs:
            measure = _PolylineMeasure.create(run)
            total_length = measure.total
            if total_length < 1.0:
                continue
            chain_count += 1
            available_start = start_trim
            available_end = total_length - end_trim
            available_length = available_end - available_start
            axis_ranges: list[tuple[float, float]] = []
            model_gap = 0.25
            pitch = spec.road_segment_length + model_gap
            if available_length >= spec.road_segment_length - 1e-6:
                count = max(1, int(math.floor((available_length + model_gap + 1e-6) / pitch)))
                used_length = count * spec.road_segment_length + max(0, count - 1) * model_gap
                leftover = max(0.0, available_length - used_length)
                if start_trim > 0.0 and end_trim <= 1e-9:
                    offset = available_start
                elif end_trim > 0.0 and start_trim <= 1e-9:
                    offset = available_end - used_length
                else:
                    offset = available_start + leftover * 0.5
                axis_ranges = [
                    (offset + segment * pitch, offset + segment * pitch + spec.road_segment_length)
                    for segment in range(count)
                ]
            elif start_trim <= 1e-9 and end_trim <= 1e-9:
                # A genuinely short dead-end road still gets one model.  It may
                # overhang the unmapped endpoint, but it never invades a junction.
                axis_ranges = [(0.0, total_length)]
            else:
                skipped_short_runs += 1
                continue

            run_chain: list[WorldObject] = []
            for axis_start, axis_end in axis_ranges:
                if len(objects) >= spec.max_road_objects:
                    truncated = True
                    break
                start_x, start_z, _ = measure.point(axis_start)
                end_x, end_z, _ = measure.point(axis_end)
                dx = end_x - start_x
                dz = end_z - start_z
                chord_length = math.hypot(dx, dz)
                if chord_length < 0.05:
                    continue
                x = (start_x + end_x) * 0.5
                z = (start_z + end_z) * 0.5
                if not (0 <= x < spec.world_size and 0 <= z < spec.world_size):
                    continue
                heading = math.degrees(math.atan2(dx, dz)) % 360.0
                # Ground against the whole fixed-size road model, not only its
                # centre.  The terrain solver makes this nearly flat, while the
                # maximum support sample prevents the model from dipping through
                # a coarse bilinear ridge at either shoulder.
                support_polygon = _oriented_rectangle(
                    x,
                    z,
                    model_width,
                    spec.road_segment_length,
                    heading,
                    margin=0.15,
                )
                support_height = _maximum_polygon_elevation(
                    elevations, spec.cells, spec.cell_size, support_polygon
                )
                obj = WorldObject(next_id, model, x, support_height + 0.04, z, heading)
                next_id += 1
                objects.append(obj)
                run_chain.append(obj)

            for previous, current in zip(run_chain, run_chain[1:]):
                previous_axis = _model_axis(previous, spec.road_segment_length)
                current_axis = _model_axis(current, spec.road_segment_length)
                centre_distance = math.dist((previous.x, previous.z), (current.x, current.z))
                maximum_model_overlap = max(
                    maximum_model_overlap,
                    max(0.0, spec.road_segment_length - centre_distance),
                )
                maximum_chain_gap = max(
                    maximum_chain_gap,
                    _segment_distance(previous_axis[0], previous_axis[1], current_axis[0], current_axis[1]),
                )

            if run_chain:
                first_axis = _model_axis(run_chain[0], spec.road_segment_length)
                last_axis = _model_axis(run_chain[-1], spec.road_segment_length)
                if start_trim > 0.0:
                    record = (first_axis, start_trim)
                    seam_records.setdefault(start_key, []).append(record)
                    if start_key in junction_keys:
                        endpoint_records.setdefault(start_key, []).append(record)
                if end_trim > 0.0:
                    record = (last_axis, end_trim)
                    seam_records.setdefault(end_key, []).append(record)
                    if end_key in junction_keys:
                        endpoint_records.setdefault(end_key, []).append(record)
            if truncated:
                break
        if truncated:
            break

    # ``maximum_chain_gap`` measures only unexpected gaps within a model
    # chain. Junction and sharp-corner patches are intentionally represented by
    # the road ground material and are reported separately below.

    connection_count = len(junction_keys)
    failed_connections = 0
    maximum_connection_gap = 0.0
    maximum_patch_radius = 0.0
    for key in sorted(junction_keys):
        records = endpoint_records.get(key, [])
        node = (key[0] * 0.10, key[1] * 0.10)
        expected_branches = node_degrees[key]
        # A long merged road passing through a junction contributes two incident
        # segments but may remain one run on each side. Requiring at least two
        # records catches an actually empty junction. The central gap itself is
        # intentional and is covered by the deterministic road ground material.
        if len(records) < 2 and expected_branches >= 3:
            failed_connections += 1
            continue
        for axis, _reserved in records:
            maximum_patch_radius = max(
                maximum_patch_radius,
                _point_segment_distance(node, axis[0], axis[1]),
            )

    if progress_callback is not None:
        progress_callback(100, f"Terrain road fitting complete: {len(objects):,} objects in {chain_count:,} chains")
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=chain_count,
        connection_count=connection_count,
        failed_connections=failed_connections,
        maximum_connection_gap=maximum_connection_gap,
        maximum_chain_gap=maximum_chain_gap,
        truncated=truncated,
        trimmed_junctions=len(junction_keys),
        terrain_filled_junctions=len(endpoint_records),
        skipped_short_runs=skipped_short_runs,
        maximum_model_overlap_metres=maximum_model_overlap,
        maximum_junction_clearance_metres=maximum_clearance,
        maximum_terrain_patch_radius_metres=maximum_patch_radius,
    )



@dataclass(frozen=True, slots=True)
class _RoadPiece:
    model_path: str
    length_metres: float
    nominal_length: int


@dataclass(frozen=True, slots=True)
class _ProjectedRoadSegment:
    osm_key: str
    model_path: str
    dirt: bool
    width_metres: float
    start: tuple[float, float]
    end: tuple[float, float]
    start_key: tuple[int, int]
    end_key: tuple[int, int]

    @property
    def length_metres(self) -> float:
        return math.dist(self.start, self.end)


def _road_model_with_length(model_path: str, nominal_length: int) -> str | None:
    """Return the sibling stock road model for a nominal straight length.

    The default CWA road families expose matching ``25``, ``12`` and ``6``
    models. Custom model paths that do not end in ``25.p3d`` safely retain only
    their configured long piece rather than having a fictional sibling guessed.
    """

    suffix = "25.p3d"
    if not model_path.casefold().endswith(suffix):
        return model_path if nominal_length == 25 else None
    return model_path[: -len(suffix)] + f"{nominal_length}.p3d"


def road_model_variants(model_path: str, configured_long_length: float) -> tuple[_RoadPiece, ...]:
    """Return deterministic road-model length variants.

    Stock OFP/CWA road families stop at 6 m. Generated gravel additionally has
    a 3 m sibling so tight bends can use shorter curved sections without
    inventing nonexistent stock assets.
    """

    if configured_long_length <= 0.0:
        raise ValueError("configured road length must be positive")
    gravel = is_generated_gravel_road_model(model_path)
    nominals = (25, 12, 6, 3) if gravel else (25, 12, 6)
    pieces: list[_RoadPiece] = []
    for nominal in nominals:
        if nominal == 3 and gravel:
            world_name = model_path.split("\\", 1)[0]
            path = gravel_road_model_path(world_name, 3)
        else:
            path = _road_model_with_length(model_path, nominal)
        if path is None:
            continue
        pieces.append(_RoadPiece(path, configured_long_length * nominal / 25.0, nominal))
    return tuple(pieces)


def road_model_variant_paths(model_path: str, configured_long_length: float) -> tuple[str, ...]:
    """Public helper used by strict-asset classification and manifests."""

    return tuple(piece.model_path for piece in road_model_variants(model_path, configured_long_length))


def _point_along_straight_segment(
    start: tuple[float, float], end: tuple[float, float], distance: float
) -> tuple[float, float]:
    length = math.dist(start, end)
    if length <= 1e-9:
        return start
    fraction = distance / length
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
    )


def _point_at_distance_extended(
    points: Sequence[tuple[float, float]], distance: float
) -> tuple[float, float, float]:
    """Sample a polyline and extrapolate beyond either endpoint."""

    return _PolylineMeasure.create(points).point(distance)


def _unique_incidents(
    values: Sequence[tuple[tuple[float, float], bool, str, str, str]],
    *,
    direction_tolerance_degrees: float = 10.0,
) -> tuple[tuple[tuple[float, float], bool, str, str, str], ...]:
    """Collapse duplicate OSM branches leaving a node in nearly one direction."""

    cosine_limit = math.cos(math.radians(direction_tolerance_degrees))
    unique: list[tuple[tuple[float, float], bool, str, str, str]] = []
    for value in sorted(values, key=lambda item: (item[4], item[3], item[2].casefold(), item[0])):
        direction = value[0]
        duplicate = False
        for existing in unique:
            dot = direction[0] * existing[0][0] + direction[1] * existing[0][1]
            if dot >= cosine_limit and value[1] == existing[1]:
                duplicate = True
                break
        if not duplicate:
            unique.append(value)
    return tuple(unique)


def _rounded_road_run(
    points: Sequence[tuple[float, float]],
    *,
    minimum_turn_degrees: float = 7.5,
    maximum_turn_degrees: float = 135.0,
    maximum_tangent_metres: float = 9.0,
    tangent_fraction: float = 0.30,
    samples_per_corner: int = 4,
) -> tuple[tuple[float, float], ...]:
    """Round ordinary degree-two road corners without moving run endpoints.

    Stock CWA road meshes are straight. Feeding a hard OSM vertex directly to
    the fitter therefore produces a visible mitre: one piece ends at the node
    and the next leaves at the new heading.  A small quadratic fillet gives the
    fixed 6/12/25 m pieces several intermediate headings to follow instead.

    Runs are already split at real intersections before this helper is called,
    so the exact junction position is preserved.  The fillet is deliberately
    compact enough to remain inside the normal road/shoulder ground underlay.
    """

    cleaned = _clean_road_points(points)
    if len(cleaned) < 3:
        return tuple(cleaned)
    samples_per_corner = max(2, int(samples_per_corner))
    rounded: list[tuple[float, float]] = [cleaned[0]]
    for index in range(1, len(cleaned) - 1):
        previous, corner, following = cleaned[index - 1], cleaned[index], cleaned[index + 1]
        incoming = (corner[0] - previous[0], corner[1] - previous[1])
        outgoing = (following[0] - corner[0], following[1] - corner[1])
        incoming_length = math.hypot(*incoming)
        outgoing_length = math.hypot(*outgoing)
        turn = _turn_degrees(previous, corner, following)
        if (
            incoming_length <= 0.10
            or outgoing_length <= 0.10
            or turn < minimum_turn_degrees
            or turn > maximum_turn_degrees
        ):
            if math.dist(rounded[-1], corner) > 0.05:
                rounded.append(corner)
            continue

        tangent = min(
            maximum_tangent_metres,
            incoming_length * tangent_fraction,
            outgoing_length * tangent_fraction,
        )
        if tangent <= 0.20:
            if math.dist(rounded[-1], corner) > 0.05:
                rounded.append(corner)
            continue

        in_unit = (incoming[0] / incoming_length, incoming[1] / incoming_length)
        out_unit = (outgoing[0] / outgoing_length, outgoing[1] / outgoing_length)
        entry = (corner[0] - in_unit[0] * tangent, corner[1] - in_unit[1] * tangent)
        exit = (corner[0] + out_unit[0] * tangent, corner[1] + out_unit[1] * tangent)
        if math.dist(rounded[-1], entry) > 0.05:
            rounded.append(entry)
        for sample in range(1, samples_per_corner):
            t = sample / samples_per_corner
            one_minus = 1.0 - t
            point = (
                one_minus * one_minus * entry[0]
                + 2.0 * one_minus * t * corner[0]
                + t * t * exit[0],
                one_minus * one_minus * entry[1]
                + 2.0 * one_minus * t * corner[1]
                + t * t * exit[1],
            )
            if math.dist(rounded[-1], point) > 0.05:
                rounded.append(point)
        if math.dist(rounded[-1], exit) > 0.05:
            rounded.append(exit)
    if math.dist(rounded[-1], cleaned[-1]) > 0.05:
        rounded.append(cleaned[-1])
    return tuple(rounded)


def _split_polyline_at_keys(
    points: Sequence[tuple[float, float]],
    split_keys: set[tuple[int, int]],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    cleaned = _clean_road_points(points)
    if len(cleaned) < 2:
        return ()
    runs: list[tuple[tuple[float, float], ...]] = []
    current: list[tuple[float, float]] = [cleaned[0]]
    for point in cleaned[1:-1]:
        current.append(point)
        if _road_node_key(point) in split_keys:
            if len(current) >= 2:
                runs.append(tuple(current))
            current = [point]
    current.append(cleaned[-1])
    if len(current) >= 2:
        runs.append(tuple(current))
    return tuple(runs)


def _road_piece_sequence(target_length: float, pieces: Sequence[_RoadPiece]) -> tuple[_RoadPiece, ...]:
    """Return the preferred next road piece for the remaining run length.

    The chain fitter only consumes the first item from this helper, so avoid an
    expensive whole-run knapsack search. Prefer the longest sibling that does
    not overshoot by more than half the shortest piece; geometry fidelity can
    still force a shorter 6 m or generated 3 m piece on a bend.
    """

    if target_length <= 0.05 or not pieces:
        return ()
    ordered = tuple(sorted(
        {piece.nominal_length: piece for piece in pieces}.values(),
        key=lambda piece: (-piece.length_metres, piece.model_path.casefold()),
    ))
    shortest = min(piece.length_metres for piece in ordered)
    for piece in ordered:
        if piece.length_metres <= target_length + shortest * 0.5:
            return (piece,)
    return (ordered[-1],)

def _normalised_direction(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length = math.hypot(dx, dz)
    return (0.0, 1.0) if length <= 1e-9 else (dx / length, dz / length)


def _heading_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _dominant_node_axis(
    incidents: Sequence[tuple[tuple[float, float], bool, str, str]],
) -> tuple[float, float]:
    """Choose the most nearly continuous incident pair at a road node."""

    ordered = sorted(incidents, key=lambda item: (item[3], item[2].casefold(), item[0]))
    if len(ordered) == 1:
        return ordered[0][0]
    best_pair: tuple[float, tuple[float, float], tuple[float, float], tuple[str, str]] | None = None
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            dot = first[0][0] * second[0][0] + first[0][1] * second[0][1]
            tie = tuple(sorted((first[3], second[3])))
            candidate = (dot, first[0], second[0], tie)
            if best_pair is None or (candidate[0], candidate[3]) < (best_pair[0], best_pair[3]):
                best_pair = candidate
    assert best_pair is not None
    first, second = best_pair[1], best_pair[2]
    axis = (first[0] - second[0], first[1] - second[1])
    length = math.hypot(*axis)
    if length <= 1e-9:
        axis = first
        length = max(1e-9, math.hypot(*axis))
    return axis[0] / length, axis[1] / length


def _generated_gravel_terrain_raise(
    start: tuple[float, float],
    end: tuple[float, float],
    start_height: float,
    end_height: float,
    elevations: Sequence[float],
    spec: PlayabilitySpec,
    *,
    vertical_offset: float,
    pitch_degrees: float,
) -> float:
    """Return a small centreline correction for a short gravel ribbon.

    Version 0.9.84 sampled the full six-metre width and raised the complete
    rigid model above the highest shoulder. On cross-slopes that turned a road
    into a floating ramp. Gravel now uses only six-metre pieces and samples
    the centreline, allowing the ribbon edges to meet the terrain rather than
    lifting the entire object because one roadside point is high.
    """

    dx = end[0] - start[0]
    dz = end[1] - start[1]
    top_vertical = GENERATED_GRAVEL_VISUAL_TOP_METRES * math.cos(
        math.radians(pitch_degrees)
    )
    required_raise = 0.0
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = start[0] + dx * fraction
        z = start[1] + dz * fraction
        fitted_height = (
            start_height * (1.0 - fraction)
            + end_height * fraction
            + vertical_offset
            + top_vertical
        )
        terrain_height = _sample_elevation(
            elevations, spec.cells, spec.cell_size, x, z
        )
        required_raise = max(
            required_raise,
            terrain_height
            + GENERATED_GRAVEL_SURFACE_CLEARANCE_METRES
            - fitted_height,
        )
    return max(0.0, required_raise)



def _point_segment_distance_2d(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length2 = dx * dx + dz * dz
    if length2 <= 1e-12:
        return math.dist(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz) / length2
    t = max(0.0, min(1.0, t))
    nearest = (start[0] + dx * t, start[1] + dz * t)
    return math.dist(point, nearest)


def _nearest_polyline_heading(
    points: Sequence[tuple[float, float]], point: tuple[float, float]
) -> float:
    best: tuple[float, int, float] | None = None
    for index, (start, end) in enumerate(zip(points, points[1:])):
        if math.dist(start, end) <= 1e-9:
            continue
        distance = _point_segment_distance_2d(point, start, end)
        heading = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0
        candidate = (distance, index, heading)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2] if best is not None else 0.0


def _signed_heading_delta(first: float, second: float) -> float:
    return (second - first + 180.0) % 360.0 - 180.0


def _curved_gravel_model_for_run(
    model_path: str,
    run: Sequence[tuple[float, float]],
    start: tuple[float, float],
    end: tuple[float, float],
) -> str:
    if not is_generated_gravel_road_model(model_path) or len(run) < 3:
        return model_path
    start_heading = _nearest_polyline_heading(run, start)
    end_heading = _nearest_polyline_heading(run, end)
    delta = _signed_heading_delta(start_heading, end_heading)
    magnitude = abs(delta)
    if magnitude < 2.5:
        return gravel_curve_model_path(model_path, 0)
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    nominal_match = re.fullmatch(
        r"gravel(25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d", filename
    )
    nominal = int(nominal_match.group(1)) if nominal_match else 6
    maximum_curve = {25: 20, 12: 20, 6: 30, 3: 45}.get(nominal, 20)
    candidates = tuple(value for value in (5, 10, 15, 20, 30, 45) if value <= maximum_curve)
    bucket = min(candidates, key=lambda value: (abs(value - magnitude), value))
    return gravel_curve_model_path(model_path, bucket if delta > 0.0 else -bucket)

def _road_object_on_slope(
    object_id: int,
    model_path: str,
    start: tuple[float, float],
    end: tuple[float, float],
    elevations: Sequence[float],
    spec: PlayabilitySpec,
    *,
    vertical_offset: float,
) -> WorldObject:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    horizontal_length = max(0.01, math.hypot(dx, dz))
    start_height = _sample_elevation(elevations, spec.cells, spec.cell_size, start[0], start[1])
    end_height = _sample_elevation(elevations, spec.cells, spec.cell_size, end[0], end[1])
    heading = math.degrees(math.atan2(dx, dz)) % 360.0
    pitch = math.degrees(math.atan2(end_height - start_height, horizontal_length))
    pitch = max(-35.0, min(35.0, pitch))
    terrain_raise = 0.0
    placement_offset = vertical_offset
    if is_generated_gravel_road_model(model_path) or is_generated_gravel_junction_model(model_path):
        # Gravel is a normal terrain-following road, not a raised slab. Place
        # its rendered surface and Roadway LOD exactly on the fitted terrain
        # plane and never lift the whole piece to clear a local terrain bump.
        # Terrain grading already owns the road surface underneath.
        placement_offset = -GENERATED_GRAVEL_VISUAL_TOP_METRES * math.cos(
            math.radians(pitch)
        )
    return WorldObject(
        object_id,
        model_path,
        (start[0] + end[0]) * 0.5,
        (start_height + end_height) * 0.5 + placement_offset + terrain_raise,
        (start[1] + end[1]) * 0.5,
        heading,
        pitch,
    )


def _stock_piece_chain(
    measure: _PolylineMeasure,
    pieces: Sequence[_RoadPiece],
    *,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
) -> tuple[tuple[_RoadPiece, tuple[float, float], tuple[float, float]], ...]:
    """Fit fixed-axis stock pieces along a polyline without axial overlap."""

    if not pieces or preferred_end_distance <= start_distance + 0.05:
        return ()
    ordered = tuple(sorted(pieces, key=lambda piece: (-piece.length_metres, piece.model_path.casefold())))
    shortest = min(piece.length_metres for piece in ordered)
    current = start_distance
    fitted: list[tuple[_RoadPiece, tuple[float, float], tuple[float, float]]] = []
    maximum_objects = max(1, int(math.ceil((maximum_end_distance - start_distance) / shortest)) + 2)
    for _ in range(maximum_objects):
        if current >= preferred_end_distance - 0.05:
            break
        remaining = preferred_end_distance - current
        if current >= minimum_end_distance - 0.05 and remaining < shortest * 0.45:
            break
        preferred_sequence = _road_piece_sequence(remaining, ordered)
        preferred_piece = preferred_sequence[0] if preferred_sequence else ordered[-1]
        start_x, start_z, start_heading = measure.point(current)
        candidates: list[
            tuple[tuple[float, ...], _RoadPiece, tuple[float, float, float, float]]
        ] = []
        for piece in ordered:
            endpoint = measure.chord_endpoint(current, piece.length_metres, maximum_end_distance)
            if endpoint is None:
                continue
            end_distance, end_x, end_z, chord_heading = endpoint
            end_heading = measure.point(end_distance)[2]
            turn = max(
                _heading_difference(chord_heading, start_heading),
                _heading_difference(chord_heading, end_heading),
            )
            deviation = measure.maximum_chord_deviation(
                current, end_distance, (start_x, start_z), (end_x, end_z)
            )
            # Long slabs are efficient on straight roads but make city bends
            # look like coarse polygon corners. Prefer progressively shorter
            # stock siblings when a chord cuts noticeably away from the rounded
            # centreline or its heading differs from the local tangents.
            if is_generated_gravel_road_model(piece.model_path):
                # Generated gravel has actual curved P3D siblings. Prefer longer
                # curved slabs on gentle bends so the road has fewer object seams
                # than stock roads, while still dropping to shorter pieces for
                # genuinely tight geometry.
                if piece.nominal_length >= 25:
                    turn_limit, deviation_limit = 15.0, 0.85
                elif piece.nominal_length >= 12:
                    turn_limit, deviation_limit = 22.0, 0.55
                elif piece.nominal_length >= 6:
                    turn_limit, deviation_limit = 30.0, 0.35
                else:
                    # Three-metre gravel pieces are reserved for tight bends
                    # where a 6 m chord still cuts visibly across the curve.
                    turn_limit, deviation_limit = 42.0, 0.20
            elif piece.nominal_length >= 25:
                turn_limit, deviation_limit = 7.0, 0.45
            elif piece.nominal_length >= 12:
                turn_limit, deviation_limit = 11.0, 0.30
            else:
                turn_limit, deviation_limit = 18.0, 0.22
            fidelity_penalty = (
                0
                if turn <= turn_limit and deviation <= deviation_limit
                else 1
            )
            if is_generated_gravel_road_model(piece.model_path):
                score = (
                    fidelity_penalty,
                    0 if piece == preferred_piece else 1,
                    -piece.length_metres,
                    max(turn / turn_limit, deviation / deviation_limit),
                    turn,
                    deviation,
                    abs(preferred_end_distance - end_distance),
                )
            else:
                score = (
                    fidelity_penalty,
                    max(turn / turn_limit, deviation / deviation_limit),
                    0 if piece == preferred_piece else 1,
                    turn,
                    deviation,
                    abs(preferred_end_distance - end_distance),
                    -piece.length_metres,
                )
            candidates.append((score, piece, endpoint))
        if not candidates:
            if current >= minimum_end_distance - 0.05:
                break
            # A very short final remainder cannot contain even the shortest
            # stock model. Extend that model along the final local direction so
            # its first endpoint remains exactly connected to the chain.
            piece = ordered[-1]
            target_distance = min(preferred_end_distance, measure.total)
            target_x, target_z, target_heading = measure.point(target_distance)
            dx, dz = target_x - start_x, target_z - start_z
            length = math.hypot(dx, dz)
            if length <= 1e-9:
                angle = math.radians(target_heading)
                dx, dz, length = math.sin(angle), math.cos(angle), 1.0
            end = (
                start_x + dx / length * piece.length_metres,
                start_z + dz / length * piece.length_metres,
            )
            fitted.append((piece, (start_x, start_z), end))
            break
        _score, piece, endpoint = min(candidates, key=lambda item: item[0])
        end_distance, end_x, end_z, _heading = endpoint
        fitted.append((piece, (start_x, start_z), (end_x, end_z)))
        if end_distance <= current + 1e-7:
            break
        current = end_distance
    return tuple(fitted)


def _short_run_fallback_piece(
    measure: _PolylineMeasure,
    pieces: Sequence[_RoadPiece],
    *,
    start_trim: float,
    end_trim: float,
) -> tuple[tuple[_RoadPiece, tuple[float, float], tuple[float, float]], ...]:
    """Return one aligned short stock piece for a run hidden by hub trimming.

    Mixed asphalt/dirt junctions use one asphalt cap. A very short dirt or
    service-road branch can therefore be geometrically covered by that cap and
    disappear completely. One shortest sibling piece preserves the branch's
    visible surface, allowing a small deterministic overlap beneath the hub.
    """

    if not pieces or measure.total <= 0.05:
        return ()
    piece = min(pieces, key=lambda item: (item.length_metres, item.model_path.casefold()))
    if start_trim > 0.0 and end_trim <= 1e-9:
        start_distance = min(start_trim, max(0.0, measure.total - piece.length_metres))
        end_distance = start_distance + piece.length_metres
    elif end_trim > 0.0 and start_trim <= 1e-9:
        end_distance = max(measure.total - end_trim, min(measure.total, piece.length_metres))
        start_distance = end_distance - piece.length_metres
    else:
        centre = measure.total * 0.5
        start_distance = centre - piece.length_metres * 0.5
        end_distance = centre + piece.length_metres * 0.5
    start_x, start_z, _ = measure.point(start_distance)
    end_x, end_z, _ = measure.point(end_distance)
    if math.hypot(end_x - start_x, end_z - start_z) <= 0.05:
        return ()
    return ((piece, (start_x, start_z), (end_x, end_z)),)


def _fit_stock_piece_road_objects(
    dataset: OsmDataset,
    projection: BboxProjection,
    elevations: Sequence[float],
    spec: PlayabilitySpec,
    *,
    starting_id: int = 1,
    progress_callback: Callable[[int, str], None] | None = None,
) -> RoadFitReport:
    """Fit a complete stock-road network or reject an undersized object budget.

    Ordinary degree-two OSM vertices stay inside one polyline run and do not
    receive separate road-cap objects. Their corners are rounded locally and
    followed with curvature-aware short stock pieces. Only real three/four-way
    junctions receive one short stock cap. CWA road links have four connection
    slots, so nodes with more than four
    effective branches deliberately receive no hub object instead of creating
    the engine's ``No more slot to add connection`` failure.

    Road objects are planned before any are emitted. Older releases placed all
    junction caps first and then stopped as soon as ``max_road_objects`` was
    reached, leaving isolated map stubs and silently omitting later asphalt
    roads. A positive budget must now fit the entire planned network; zero still
    means that road-object placement is deliberately disabled.
    """

    if spec.max_road_objects == 0:
        if progress_callback is not None:
            progress_callback(100, "Stock road placement disabled by zero object budget")
        return RoadFitReport(
            objects=(),
            chain_count=0,
            connection_count=0,
            failed_connections=0,
            maximum_connection_gap=0.0,
            maximum_chain_gap=0.0,
            truncated=False,
        )

    projected_features: list[
        tuple[OsmLineFeature, str, bool, float, tuple[tuple[float, float], ...]]
    ] = []
    incidents: dict[
        tuple[int, int], list[tuple[tuple[float, float], bool, str, str, str]]
    ] = {}
    node_positions: dict[tuple[int, int], tuple[float, float]] = {}
    bend_keys: set[tuple[int, int]] = set()

    total_roads = len(dataset.roads)
    if progress_callback is not None:
        progress_callback(0, f"Projecting {total_roads:,} normalized road lines")
    projection_step = max(1, total_roads // 20)
    for feature_index, feature in enumerate(dataset.roads, start=1):
        if progress_callback is not None and (
            feature_index == total_roads or feature_index % projection_step == 0
        ):
            progress_callback(
                min(20, round(feature_index / max(1, total_roads) * 20)),
                f"Projected road lines {feature_index:,}/{total_roads:,}",
            )
        if not road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        if (
            bool(getattr(spec, "bridges_enabled", True))
            and not bool(getattr(spec, "procedural_bridges", False))
            and _road_is_explicit_bridge(feature.tags)
            and not road_bridge_crosses_ditch_only(feature, dataset, projection)
        ):
            # most_stred30 is the actual stock bridge/road model. Do not also
            # lay sil/kos road pieces through the same OSM bridge span; the
            # overlapping road simulations can fight for rendering/road links.
            continue
        points = tuple(_clean_road_points([projection.to_world(point) for point in feature.points]))
        if len(points) < 2:
            continue
        dirt = road_is_dirt(feature.tags)
        model = road_model_for_tags(spec, feature.tags)
        width = max(6.0, road_width_metres(feature.tags))
        projected_features.append((feature, model, dirt, width, points))
        for index, (start_point, end_point) in enumerate(zip(points, points[1:])):
            if math.dist(start_point, end_point) <= 0.05:
                continue
            start_key = _road_node_key(start_point)
            end_key = _road_node_key(end_point)
            segment_key = f"{feature.osm_key}/{index:06d}"
            forward = _normalised_direction(start_point, end_point)
            reverse = (-forward[0], -forward[1])
            incidents.setdefault(start_key, []).append(
                (forward, dirt, model, segment_key, feature.osm_key)
            )
            incidents.setdefault(end_key, []).append(
                (reverse, dirt, model, segment_key, feature.osm_key)
            )
            node_positions.setdefault(start_key, start_point)
            node_positions.setdefault(end_key, end_point)
        for index in range(1, len(points) - 1):
            if _turn_degrees(points[index - 1], points[index], points[index + 1]) >= 15.0:
                bend_keys.add(_road_node_key(points[index]))

    effective_incidents = {key: _unique_incidents(values) for key, values in incidents.items()}
    degree_two_turn_keys: set[tuple[int, int]] = set()
    for key, values in effective_incidents.items():
        if len(values) != 2:
            continue
        first, second = values[0][0], values[1][0]
        dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
        outgoing_angle = math.degrees(math.acos(dot))
        bend = abs(180.0 - outgoing_angle)
        if bend >= 15.0:
            degree_two_turn_keys.add(key)

    true_junction_keys = {
        key for key, values in effective_incidents.items() if 3 <= len(values) <= 4
    }
    complex_keys = {key for key, values in effective_incidents.items() if len(values) > 4}
    candidate_cap_keys = true_junction_keys - complex_keys

    if progress_callback is not None:
        progress_callback(
            24,
            f"Classified {len(candidate_cap_keys):,} real road junctions; "
            f"{len(degree_two_turn_keys | bend_keys):,} ordinary bends use rounded piece chains",
        )

    # Only genuine three/four-way intersections need a hub. Degree-two bends
    # are now rounded in the centreline and followed with short stock pieces.
    # This avoids the tiny straight cap laid across every corner, which was the
    # main source of pinched/diamond-shaped turns in dense street networks.
    cap_keys = set(candidate_cap_keys)
    suppressed_nearby_hubs = 0
    degree_two_keys = {key for key, values in effective_incidents.items() if len(values) == 2}
    suppressed_degree_two_caps = len(degree_two_keys)

    variant_cache: dict[str, tuple[_RoadPiece, ...]] = {}

    def variants_for(model_path: str) -> tuple[_RoadPiece, ...]:
        variants = variant_cache.get(model_path)
        if variants is None:
            variants = road_model_variants(model_path, spec.road_segment_length)
            if is_generated_gravel_road_model(model_path):
                # Keep the 25 m gravel slab out of terrain-following chains, but
                # allow 12 m curved ribbons as well as 6 m ones. The previous
                # six-metre-only rule created a visible seam every few metres.
                # Twelve-metre pieces still follow graded terrain closely while
                # roughly halving the number of visible joins on gentle roads.
                variants = tuple(
                    piece for piece in variants if piece.nominal_length in {12, 6, 3}
                ) or variants[-1:]
            variant_cache[model_path] = variants
        return variants

    # Plan cap geometry and trim distances before constructing WorldObjects.
    cap_plans: dict[
        tuple[int, int], tuple[_RoadPiece, tuple[float, float], tuple[float, float]]
    ] = {}
    cap_trim_lengths: dict[tuple[int, int], float] = {}
    cap_cover_lengths: dict[tuple[int, int], float] = {}
    for key in sorted(cap_keys):
        values = effective_incidents[key]
        use_dirt = all(value[1] for value in values)
        all_gravel = all(is_generated_gravel_road_model(value[2]) for value in values)
        incident_models = {value[2].casefold(): value[2] for value in values}
        if all_gravel:
            degree = len(values)
            base_model = gravel_junction_model_path(spec.name, degree)
            hub_length = 5.4 if degree == 3 else 6.0
            cap_piece = _RoadPiece(base_model, hub_length, 6)
        else:
            if len(incident_models) == 1:
                base_model = next(iter(incident_models.values()))
            else:
                base_model = spec.dirt_road_model if use_dirt else spec.paved_road_model
            variants = variants_for(base_model)
            cap_piece = next((piece for piece in variants if piece.nominal_length == 6), variants[-1])
        dominant_values = tuple((value[0], value[1], value[2], value[3]) for value in values)
        axis = _dominant_node_axis(dominant_values)
        node = node_positions[key]
        half = cap_piece.length_metres * 0.5
        start_point = (node[0] - axis[0] * half, node[1] - axis[1] * half)
        end_point = (node[0] + axis[0] * half, node[1] + axis[1] * half)
        cap_plans[key] = (cap_piece, start_point, end_point)
        # Branches extend 0.70 m beneath the cap. This hides interpolation and
        # pitch rounding seams without creating an extra road-link object.
        cap_trim_lengths[key] = max(0.40, half - 0.70)
        cap_cover_lengths[key] = half + 0.15

    # Nodes with more than four branches cannot receive a stock hub object, but
    # their road lines still split at the shared centre.
    virtual_cover_lengths = {
        key: spec.road_segment_length * 6.0 / 25.0 * 0.5 + 0.15 for key in complex_keys
    }
    virtual_trim_lengths = {
        key: max(0.40, cover - 0.85) for key, cover in virtual_cover_lengths.items()
    }
    split_keys = cap_keys | complex_keys

    # Each feature plan contains the original run, endpoint metadata, fitted
    # pieces that actually intersect the world, and whether adjacent hub caps
    # fully cover a run too short to need its own model.
    planned_features: list[
        tuple[
            OsmLineFeature,
            tuple[
                tuple[
                    tuple[tuple[float, float], ...],
                    tuple[int, int],
                    tuple[int, int],
                    float,
                    float,
                    tuple[tuple[_RoadPiece, tuple[float, float], tuple[float, float]], ...],
                    bool,
                ],
                ...,
            ],
        ]
    ] = []
    skipped_short_runs = 0
    required_chain_objects = 0
    planning_step = max(1, len(projected_features) // 30)
    for feature_index, (feature, model, _dirt, _width, points) in enumerate(
        projected_features, start=1
    ):
        if progress_callback is not None and (
            feature_index == len(projected_features) or feature_index % planning_step == 0
        ):
            local = 35 + round(feature_index / max(1, len(projected_features)) * 23)
            progress_callback(
                min(58, local),
                f"Planning stock road lines {feature_index:,}/{len(projected_features):,}; "
                f"{required_chain_objects + len(cap_plans):,} objects",
            )
        feature_plans: list[
            tuple[
                tuple[tuple[float, float], ...],
                tuple[int, int],
                tuple[int, int],
                float,
                float,
                tuple[tuple[_RoadPiece, tuple[float, float], tuple[float, float]], ...],
                bool,
            ]
        ] = []
        for raw_run in _split_polyline_at_keys(points, split_keys):
            run = _rounded_road_run(raw_run)
            measure = _PolylineMeasure.create(run)
            total_length = measure.total
            if total_length <= 0.05:
                skipped_short_runs += 1
                continue
            start_key = _road_node_key(run[0])
            end_key = _road_node_key(run[-1])
            start_trim = cap_trim_lengths.get(
                start_key, virtual_trim_lengths.get(start_key, 0.0)
            )
            end_trim = cap_trim_lengths.get(end_key, virtual_trim_lengths.get(end_key, 0.0))
            start_cover = cap_cover_lengths.get(
                start_key, virtual_cover_lengths.get(start_key, 0.0)
            )
            end_cover = cap_cover_lengths.get(end_key, virtual_cover_lengths.get(end_key, 0.0))
            start_distance = min(total_length, start_trim)
            preferred_end = max(start_distance, total_length - end_trim)
            minimum_end = max(start_distance, total_length - end_cover)
            variants = variants_for(model)
            shortest = min(piece.length_metres for piece in variants)
            maximum_end = total_length + (0.70 if end_cover > 0.0 else shortest * 0.5)
            fitted_pieces = _stock_piece_chain(
                measure,
                variants,
                start_distance=start_distance,
                preferred_end_distance=preferred_end,
                minimum_end_distance=minimum_end,
                maximum_end_distance=maximum_end,
            )
            covered_by_hubs = False
            if not fitted_pieces:
                covered_by_hubs = total_length <= start_cover + end_cover + 1e-6
                variant_paths = {piece.model_path.casefold() for piece in variants}
                cap_surface_mismatch = any(
                    key in cap_plans
                    and cap_plans[key][0].model_path.casefold() not in variant_paths
                    for key in (start_key, end_key)
                )
                if not covered_by_hubs or cap_surface_mismatch:
                    fitted_pieces = _short_run_fallback_piece(
                        measure,
                        variants,
                        start_trim=start_trim,
                        end_trim=end_trim,
                    )
                    if fitted_pieces:
                        covered_by_hubs = False
                if not fitted_pieces:
                    skipped_short_runs += 1

            placeable: list[
                tuple[_RoadPiece, tuple[float, float], tuple[float, float]]
            ] = []
            for piece, start_point, end_point in fitted_pieces:
                start_x, start_z = start_point
                end_x, end_z = end_point
                if math.hypot(end_x - start_x, end_z - start_z) <= 0.05:
                    continue
                centre = ((start_x + end_x) * 0.5, (start_z + end_z) * 0.5)
                # Never serialize a road object whose origin is outside the
                # finite WRP. The 0.9.205 debug PBO contained six such pieces
                # (stock and gravel) just beyond 0/6400 m. Their geometry was
                # valid, but feeding out-of-world objects to the legacy clipper
                # is a concrete candidate for the remaining ClipDraw/orIn
                # assertions. Border-crossing roads simply stop at the map edge.
                if not (
                    0.0 <= centre[0] < spec.world_size
                    and 0.0 <= centre[1] < spec.world_size
                ):
                    continue
                placeable.append((piece, start_point, end_point))
            required_chain_objects += len(placeable)
            feature_plans.append(
                (
                    run,
                    start_key,
                    end_key,
                    start_cover,
                    end_cover,
                    tuple(placeable),
                    covered_by_hubs,
                )
            )
        planned_features.append((feature, tuple(feature_plans)))

    # Put dirt first, gravel next, and paved roads last. Object
    # ordering alone is not relied upon for depth, but it gives CWA a stable
    # surface precedence that matches the explicit vertical offsets below.
    planned_features.sort(
        key=lambda item: (_road_surface_priority(item[0].tags), item[0].osm_key)
    )

    required_objects = len(cap_plans) + required_chain_objects
    if progress_callback is not None:
        progress_callback(
            59,
            f"Road object plan requires {required_objects:,}; budget {spec.max_road_objects:,}",
        )
    if required_objects > spec.max_road_objects:
        raise ValueError(
            "road object budget is too small for a complete network: "
            f"requires {required_objects:,} objects, limit is {spec.max_road_objects:,}; "
            f"increase --max-road-objects to at least {required_objects:,}. "
            "Partial road networks are not emitted."
        )

    objects: list[WorldObject] = []
    next_id = starting_id
    cap_objects: dict[tuple[int, int], WorldObject] = {}
    maximum_pitch = 0.0
    if progress_callback is not None:
        progress_callback(61, f"Placing {len(cap_plans):,} junction caps")
    for key in sorted(cap_plans):
        cap_piece, start_point, end_point = cap_plans[key]
        obj = _road_object_on_slope(
            next_id,
            cap_piece.model_path,
            start_point,
            end_point,
            elevations,
            spec,
            vertical_offset=0.060,
        )
        next_id += 1
        objects.append(obj)
        cap_objects[key] = obj
        maximum_pitch = max(maximum_pitch, abs(obj.pitch_degrees))

    maximum_chain_gap = 0.0
    maximum_model_overlap = 0.0
    maximum_endpoint_gap = 0.0
    short_piece_objects = 0
    chain_count = 0
    endpoint_axes: dict[
        tuple[int, int], list[tuple[tuple[float, float], tuple[float, float]]]
    ] = {}

    fitting_step = max(1, len(planned_features) // 30)
    for feature_index, (feature, feature_plans) in enumerate(planned_features, start=1):
        if progress_callback is not None and (
            feature_index == len(planned_features) or feature_index % fitting_step == 0
        ):
            local = 63 + round(feature_index / max(1, len(planned_features)) * 33)
            progress_callback(
                min(96, local),
                f"Fitting stock road lines {feature_index:,}/{len(planned_features):,}; "
                f"{len(objects):,}/{required_objects:,} objects",
            )
        for (
            run,
            start_key,
            end_key,
            start_cover,
            end_cover,
            fitted_pieces,
            covered_by_hubs,
        ) in feature_plans:
            if not fitted_pieces:
                if covered_by_hubs and start_key in cap_keys:
                    node = node_positions[start_key]
                    endpoint_axes.setdefault(start_key, []).append((node, node))
                if covered_by_hubs and end_key in cap_keys:
                    node = node_positions[end_key]
                    endpoint_axes.setdefault(end_key, []).append((node, node))
                continue
            chain_count += 1
            chain: list[tuple[WorldObject, float]] = []
            for piece, start_point, end_point in fitted_pieces:
                placed_model = _curved_gravel_model_for_run(
                    piece.model_path, run, start_point, end_point
                )
                obj = _road_object_on_slope(
                    next_id,
                    placed_model,
                    start_point,
                    end_point,
                    elevations,
                    spec,
                    vertical_offset=_road_vertical_offset(feature.tags),
                )
                next_id += 1
                objects.append(obj)
                chain.append((obj, piece.length_metres))
                if piece.nominal_length != 25:
                    short_piece_objects += 1
                maximum_pitch = max(maximum_pitch, abs(obj.pitch_degrees))

            for (previous, previous_length), (current, current_length) in zip(
                chain, chain[1:]
            ):
                previous_axis = _model_axis(previous, previous_length)
                current_axis = _model_axis(current, current_length)
                previous_end = previous_axis[1]
                current_start = current_axis[0]
                gap = math.dist(previous_end, current_start)
                maximum_chain_gap = max(maximum_chain_gap, gap)
                angle = math.radians(previous.heading_degrees)
                direction = (math.sin(angle), math.cos(angle))
                offset = (
                    current_start[0] - previous_end[0],
                    current_start[1] - previous_end[1],
                )
                lateral = abs(direction[0] * offset[1] - direction[1] * offset[0])
                longitudinal = direction[0] * offset[0] + direction[1] * offset[1]
                if lateral <= 0.10 and longitudinal < 0.0:
                    maximum_model_overlap = max(maximum_model_overlap, -longitudinal)

            first_axis = _model_axis(chain[0][0], chain[0][1])
            last_axis = _model_axis(chain[-1][0], chain[-1][1])
            endpoint_axes.setdefault(start_key, []).append(first_axis)
            endpoint_axes.setdefault(end_key, []).append(last_axis)
            maximum_endpoint_gap = max(
                maximum_endpoint_gap,
                _point_segment_distance(run[0], first_axis[0], first_axis[1]),
                _point_segment_distance(run[-1], last_axis[0], last_axis[1]),
            )

    failed_connections = 0
    maximum_connection_gap = 0.0
    maximum_clearance = 0.0
    for key in sorted(cap_keys):
        cover = cap_cover_lengths.get(key, 0.0)
        maximum_clearance = max(maximum_clearance, cover)
        if key not in cap_objects:
            failed_connections += 1
            continue
        node = node_positions[key]
        axes = endpoint_axes.get(key, [])
        expected = len(effective_incidents[key])
        if len(axes) < expected:
            failed_connections += expected - len(axes)
        for axis in axes:
            uncovered = max(0.0, _point_segment_distance(node, axis[0], axis[1]) - cover)
            maximum_connection_gap = max(maximum_connection_gap, uncovered)
            if uncovered > spec.road_connection_tolerance:
                failed_connections += 1

    if progress_callback is not None:
        progress_callback(
            100,
            f"Stock road fitting complete: {len(objects):,} objects in {chain_count:,} chains",
        )
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=chain_count,
        connection_count=len(cap_keys),
        failed_connections=failed_connections,
        maximum_connection_gap=maximum_connection_gap,
        maximum_chain_gap=maximum_chain_gap,
        truncated=False,
        trimmed_junctions=0,
        skipped_short_runs=skipped_short_runs,
        maximum_model_overlap_metres=maximum_model_overlap,
        maximum_junction_clearance_metres=maximum_clearance,
        maximum_terrain_patch_radius_metres=0.0,
        junction_cap_objects=len(cap_objects),
        short_piece_objects=short_piece_objects,
        maximum_endpoint_gap_metres=maximum_endpoint_gap,
        maximum_road_pitch_degrees=maximum_pitch,
        suppressed_degree_two_caps=suppressed_degree_two_caps,
        terrain_filled_junctions=len(complex_keys),
        complex_junctions_without_caps=len(complex_keys),
        road_connection_slot_risk_nodes=0,
        suppressed_nearby_hubs=suppressed_nearby_hubs,
    )

def fit_road_objects(
    dataset: OsmDataset,
    projection: BboxProjection,
    elevations: Sequence[float],
    spec: PlayabilitySpec,
    *,
    starting_id: int = 1,
    progress_callback: Callable[[int, str], None] | None = None,
) -> RoadFitReport:
    if bool(getattr(spec, "stock_road_piece_fitting", False)):
        return _fit_stock_piece_road_objects(
            dataset, projection, elevations, spec, starting_id=starting_id, progress_callback=progress_callback
        )
    return _fit_terrain_patch_road_objects(
        dataset, projection, elevations, spec, starting_id=starting_id, progress_callback=progress_callback
    )

def _road_seed_targets(
    elevations: Sequence[float], dataset: OsmDataset, projection: BboxProjection, spec: PlayabilitySpec
) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    max_rise_ratio = spec.maximum_road_grade_percent / 100.0
    sample_spacing = max(2.0, spec.cell_size * 0.45)
    for feature in dataset.roads:
        if not road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = [projection.to_world(point) for point in feature.points]
        samples: list[tuple[float, float, float, float]] = []
        cumulative = 0.0
        for start, end in zip(points, points[1:]):
            length = math.dist(start, end)
            if length < 1.0:
                continue
            count = max(1, int(math.ceil(length / sample_spacing)))
            for index in range(count):
                fraction = index / count
                x = start[0] + (end[0] - start[0]) * fraction
                z = start[1] + (end[1] - start[1]) * fraction
                samples.append((x, z, cumulative + length * fraction, _sample_elevation(elevations, spec.cells, spec.cell_size, x, z)))
            cumulative += length
        if points:
            x, z = points[-1]
            samples.append((x, z, cumulative, _sample_elevation(elevations, spec.cells, spec.cell_size, x, z)))
        if len(samples) < 2:
            continue
        heights = [sample[3] for sample in samples]
        # Forward/backward grade clamps create a profile that remains close to
        # terrain but cannot demand mountain-goat road pieces.
        for index in range(1, len(samples)):
            distance = max(0.01, samples[index][2] - samples[index - 1][2])
            limit = distance * max_rise_ratio
            heights[index] = min(max(heights[index], heights[index - 1] - limit), heights[index - 1] + limit)
        for index in range(len(samples) - 2, -1, -1):
            distance = max(0.01, samples[index + 1][2] - samples[index][2])
            limit = distance * max_rise_ratio
            heights[index] = min(max(heights[index], heights[index + 1] - limit), heights[index + 1] + limit)
        for sample, height in zip(samples, heights):
            x, z = sample[0], sample[1]
            if not (0 <= x < spec.world_size and 0 <= z < spec.world_size):
                continue
            cx = min(spec.cells - 1, max(0, int(x / spec.cell_size)))
            cz = min(spec.cells - 1, max(0, int(z / spec.cell_size)))
            values.setdefault(cz * spec.cells + cx, []).append(height)
    return {index: sum(items) / len(items) for index, items in values.items()}


def _building_components(mask: Sequence[bool], cells: int) -> list[list[int]]:
    remaining = {index for index, value in enumerate(mask) if value}
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = deque([seed])
        component: list[int] = []
        while queue:
            index = queue.popleft()
            component.append(index)
            x, z = index % cells, index // cells
            for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                neighbour = nz * cells + nx
                if 0 <= nx < cells and 0 <= nz < cells and neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def _spread_targets(
    seeds: Mapping[int, float], cells: int, radius_cells: int, blocked: Sequence[bool]
) -> tuple[list[float | None], list[int]]:
    targets: list[float | None] = [None] * (cells * cells)
    distances = [10**9] * (cells * cells)
    queue: deque[int] = deque()
    for index in sorted(seeds):
        if blocked[index]:
            continue
        targets[index] = seeds[index]
        distances[index] = 0
        queue.append(index)
    while queue:
        index = queue.popleft()
        distance = distances[index]
        if distance >= radius_cells:
            continue
        x, z = index % cells, index // cells
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if not (0 <= nx < cells and 0 <= nz < cells):
                continue
            neighbour = nz * cells + nx
            if blocked[neighbour]:
                continue
            next_distance = distance + 1
            if next_distance < distances[neighbour]:
                distances[neighbour] = next_distance
                targets[neighbour] = targets[index]
                queue.append(neighbour)
            elif next_distance == distances[neighbour] and targets[neighbour] is not None:
                targets[neighbour] = (targets[neighbour] + targets[index]) * 0.5  # type: ignore[operator]
    return targets, distances


def _road_slope_percent(
    elevations: Sequence[float], dataset: OsmDataset, projection: BboxProjection, spec: PlayabilitySpec
) -> float:
    maximum = 0.0
    spacing = max(2.0, spec.cell_size * 0.35)
    for feature in dataset.roads:
        if not road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = [projection.to_world(point) for point in feature.points]
        for start, end in zip(points, points[1:]):
            length = math.dist(start, end)
            if length < 1.0:
                continue
            count = max(1, int(math.ceil(length / spacing)))
            previous_x, previous_z = start
            previous_h = _sample_elevation(elevations, spec.cells, spec.cell_size, previous_x, previous_z)
            for index in range(1, count + 1):
                fraction = index / count
                x = start[0] + (end[0] - start[0]) * fraction
                z = start[1] + (end[1] - start[1]) * fraction
                height = _sample_elevation(elevations, spec.cells, spec.cell_size, x, z)
                distance = max(0.01, math.hypot(x - previous_x, z - previous_z))
                maximum = max(maximum, abs(height - previous_h) / distance * 100.0)
                previous_x, previous_z, previous_h = x, z, height
    return maximum


def _component_roughness(elevations: Sequence[float], components: Sequence[Sequence[int]]) -> float:
    maximum = 0.0
    for component in components:
        if not component:
            continue
        heights = [elevations[index] for index in component]
        maximum = max(maximum, max(heights) - min(heights))
    return maximum


def grade_terrain(
    elevations: Sequence[float], dataset: OsmDataset, projection: BboxProjection, raster: OsmRaster, spec: PlayabilitySpec
) -> TerrainGradeReport:
    original = tuple(elevations)
    result = list(original)
    road_seeds = _road_seed_targets(original, dataset, projection, spec)
    building_components = _building_components(raster.buildings, spec.cells)
    building_seeds: dict[int, float] = {}
    for component in building_components:
        target = sum(original[index] for index in component) / len(component)
        for index in component:
            building_seeds[index] = target

    blocked = raster.water
    road_radius = max(0, int(math.ceil(spec.road_grade_radius / spec.cell_size)))
    building_radius = max(0, int(math.ceil(spec.building_grade_radius / spec.cell_size)))
    road_targets, road_distances = _spread_targets(road_seeds, spec.cells, road_radius, blocked)
    building_targets, building_distances = _spread_targets(building_seeds, spec.cells, building_radius, blocked)

    maximum_cut = 0.0
    maximum_fill = 0.0
    changed = 0
    for index, value in enumerate(original):
        if blocked[index]:
            continue
        target = value
        weight = 0.0
        if road_targets[index] is not None:
            distance = road_distances[index]
            road_weight = 1.0 if distance <= 1 else (1.0 if road_radius == 0 else max(0.0, 1.0 - (distance - 1) / max(1.0, road_radius)) ** 2)
            target = road_targets[index]  # type: ignore[assignment]
            weight = road_weight
        if building_targets[index] is not None:
            distance = building_distances[index]
            building_weight = 1.0 if building_radius == 0 else max(0.0, 1.0 - distance / (building_radius + 1.0)) ** 2
            if building_weight >= weight:
                target = building_targets[index]  # type: ignore[assignment]
                weight = building_weight
        adjustment = max(-spec.maximum_grade_adjustment, min(spec.maximum_grade_adjustment, (target - value) * weight))
        if abs(adjustment) > 1e-9:
            changed += 1
            result[index] = value + adjustment
            maximum_cut = max(maximum_cut, -adjustment)
            maximum_fill = max(maximum_fill, adjustment)

    road_cells = tuple(sorted(road_seeds))
    return TerrainGradeReport(
        elevations=tuple(result),
        changed_cells=changed,
        road_seed_cells=len(road_seeds),
        building_pad_cells=sum(len(component) for component in building_components),
        maximum_cut=maximum_cut,
        maximum_fill=maximum_fill,
        maximum_road_slope_before_percent=_road_slope_percent(original, dataset, projection, spec),
        maximum_road_slope_after_percent=_road_slope_percent(result, dataset, projection, spec),
        building_roughness_before=_component_roughness(original, building_components),
        building_roughness_after=_component_roughness(result, building_components),
    )


def _stable_fraction(seed: str, index: int, label: str) -> float:
    digest = hashlib.blake2s(f"{seed}:{label}:{index}".encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") / 0xFFFFFFFF


def improve_transitions(indices: Sequence[int], raster: OsmRaster, spec: PlayabilitySpec) -> TransitionReport:
    result = list(indices)
    cells = spec.cells
    width = spec.transition_cells
    if width <= 0:
        return TransitionReport(tuple(result), 0, 0)

    water_distance = [10**9] * len(result)
    queue: deque[int] = deque()
    for index, water in enumerate(raster.water):
        if water:
            water_distance[index] = 0
            queue.append(index)
    while queue:
        index = queue.popleft()
        if water_distance[index] >= width:
            continue
        x, z = index % cells, index // cells
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if not (0 <= nx < cells and 0 <= nz < cells):
                continue
            neighbour = nz * cells + nx
            if water_distance[neighbour] > water_distance[index] + 1:
                water_distance[neighbour] = water_distance[index] + 1
                queue.append(neighbour)

    shoreline = 0
    softened = 0
    original = tuple(indices)
    for index, material in enumerate(original):
        if raster.water[index] or raster.roads[index] or raster.buildings[index]:
            continue
        distance = water_distance[index]
        if 0 < distance <= width:
            probability = (width - distance + 1) / (width + 1)
            if _stable_fraction(spec.deterministic_seed, index, "shore") <= probability:
                if result[index] != 1:
                    result[index] = 1  # sand transition
                    shoreline += 1
                continue
        if material not in {4, 5, 6}:
            continue
        x, z = index % cells, index // cells
        boundary = False
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if 0 <= nx < cells and 0 <= nz < cells:
                neighbour = original[nz * cells + nx]
                if neighbour != material and neighbour not in {0, 7}:
                    boundary = True
                    break
        if boundary and _stable_fraction(spec.deterministic_seed, index, "landuse") < 0.42:
            result[index] = 2  # grass breaks up hard polygon edges
            softened += 1
    return TransitionReport(tuple(result), shoreline, softened)


def _ascii_name(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = " ".join(normalized.replace('"', "").split()).strip()
    return normalized[:63] or fallback


def town_locations(dataset: OsmDataset, projection: BboxProjection, limit: int) -> tuple[TownLocation, ...]:
    ranked = {"city": 0, "town": 1, "village": 2, "suburb": 3, "quarter": 4, "hamlet": 5, "isolated_dwelling": 6}
    def has_visible_name(place: OsmPointFeature) -> bool:
        name = str(place.tags.get("name", "")).strip()
        return bool(name) and name.casefold() not in {"unnamed", "unnamed isolated dwelling"}

    candidates = sorted(
        (place for place in dataset.places if has_visible_name(place)),
        key=lambda place: (ranked.get(place.tags.get("place", ""), 99), place.osm_key),
    )[:limit]
    locations: list[TownLocation] = []
    for index, place in enumerate(candidates, start=1):
        x, z = projection.to_world(place.point)
        if not (0 <= x <= projection.world_size and 0 <= z <= projection.world_size):
            continue
        locations.append(
            TownLocation(
                class_name=f"Town{index:03d}",
                name=_ascii_name(place.tags.get("name", ""), f"Place {index}"),
                x=x,
                z=z,
                place_type=place.tags.get("place", "place"),
                osm_key=place.osm_key,
            )
        )
    return tuple(locations)
