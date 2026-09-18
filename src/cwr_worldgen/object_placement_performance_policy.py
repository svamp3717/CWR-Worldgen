# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance refinements for paved junctions, stock roads, and forests.

These helpers sit in object-generation hot loops where the same small geometric
questions are asked tens of thousands of times.  The replacements preserve the
existing placement and scoring rules while avoiding broad scans, repeated sorts,
and duplicate terrain samples.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import heapq
import math
from typing import Any, Iterable, Mapping, Sequence

from . import osm as _osm
from . import paved_junction_performance_policy as _paved_perf
from . import paved_junction_policy as _paved
from . import playability as _playability

_INSTALLED = False
_ORIGINAL_APPROACH_CHOICE: Any = None
_ORIGINAL_PLAN_APPLICATION: Any = None
_ORIGINAL_CHORD_ENDPOINT: Any = None
_ORIGINAL_MAXIMUM_CHORD_DEVIATION: Any = None
_ORIGINAL_ROAD_PIECE_SEQUENCE: Any = None
_ORIGINAL_NEAREST_POLYLINE_HEADING: Any = None
_ORIGINAL_CORRIDOR_INTERSECTS_RECTANGLE: Any = None
_ORIGINAL_ROADSIDE_VEGETATION_CANDIDATES: Any = None
_ORIGINAL_PLACE_CLUSTER_AT: Any = None


# ---------------------------------------------------------------------------
# Paved-junction planning


def _fast_approach_choice_to_target(plan: Any, arm: Any, target: Any, tolerance: float):
    """Reject impossible path templates in connector-local space first.

    Rotation/translation do not change the distance from the path endpoint to
    the merge target.  Most templates miss every stock 6/12/25 m straight, so
    reject those before transforming points back to world space and doing the
    heading comparisons.  Survivors use the historical calculations verbatim.
    """

    connector = arm.connector
    delta = _paved._signed_angle(connector.direction, target.continuation)
    preferred_sign = 1 if delta >= 0.0 else -1
    initial_heading = _paved._heading(connector.direction)
    local_target = _paved_perf._target_local_point(arm, target)
    straight_lengths = tuple(
        (nominal, float(_paved._STRAIGHTS[nominal])) for nominal in (6, 12, 25)
    )
    best = None

    for turn_sign in (preferred_sign, -preferred_sign):
        for path in _paved_perf._candidate_templates(arm, target, tolerance, turn_sign):
            local_dx = local_target[0] - path.point[0]
            local_dz = local_target[1] - path.point[1]
            local_distance = math.hypot(local_dx, local_dz)
            if local_distance <= 0.05:
                continue

            nominal_errors = []
            for nominal, straight_length in straight_lengths:
                length_error = abs(local_distance - straight_length)
                if length_error <= tolerance:
                    nominal_errors.append((nominal, length_error))
            if not nominal_errors:
                continue

            # Preserve the original world-space arithmetic for viable templates
            # so scoring/tie behaviour remains identical to the existing policy.
            world_point = _paved._world(
                path.point,
                connector.point,
                connector.direction,
            )
            merge_vector = (
                target.point[0] - world_point[0],
                target.point[1] - world_point[1],
            )
            merge_distance = math.hypot(*merge_vector)
            if merge_distance <= 0.05:
                continue
            merge_direction = (
                merge_vector[0] / merge_distance,
                merge_vector[1] / merge_distance,
            )
            path_direction = _paved._direction(
                (initial_heading + path.heading) % 360.0
            )
            in_error = _paved._angle(path_direction, merge_direction)
            out_error = _paved._angle(merge_direction, target.continuation)
            if max(in_error, out_error) > 12.0:
                continue

            # Recompute the tiny length error from the historical world-space
            # distance for exact score compatibility after the local prefilter.
            for nominal, _local_error in nominal_errors:
                length_error = abs(
                    merge_distance - float(_paved._STRAIGHTS[nominal])
                )
                if length_error > tolerance:
                    continue
                piece_count = (
                    path.first_turns
                    + path.counter_turns
                    + path.middle_units
                    + 1
                )
                score = (
                    length_error * 20.0
                    + in_error
                    + out_error
                    + piece_count * 0.25
                    + path.first_radius * 0.001
                    + path.counter_radius * 0.001
                )
                choice = _paved._ApproachChoice(
                    path.turn_sign,
                    path.first_turns,
                    path.first_radius,
                    path.middle_units,
                    path.counter_turns,
                    path.counter_radius,
                    nominal,
                    target.point,
                )
                if best is None or score < best[0]:
                    best = score, choice
    return best


def _fast_plan_application(state: Any, plan: Any, spec: Any):
    """Find the same lowest-score distinct-arm assignment with pruning."""

    options = tuple(
        _paved_perf._arm_options(state, plan, arm, spec)
        for arm in plan.arms
    )
    if any(not values for values in options):
        return None

    # A lower bound for every remaining arm lets us skip Cartesian subtrees that
    # cannot beat the incumbent.  Traversal order is unchanged, and equal scores
    # are still ignored just like the original ``product`` implementation.
    minimum_tail = [0.0] * (len(options) + 1)
    for index in range(len(options) - 1, -1, -1):
        minimum_tail[index] = minimum_tail[index + 1] + min(
            float(value[0]) for value in options[index]
        )

    best_score = math.inf
    best_combination: tuple[Any, ...] | None = None
    chosen: list[Any] = []
    used_object_ids: set[int] = set()

    def visit(arm_index: int, score: float) -> None:
        nonlocal best_score, best_combination
        if arm_index == len(options):
            if score < best_score:
                best_score = score
                best_combination = tuple(chosen)
            return
        if score + minimum_tail[arm_index] >= best_score:
            return
        for value in options[arm_index]:
            object_id = int(value[1].object_id)
            if object_id in used_object_ids:
                continue
            next_score = score + float(value[0])
            if next_score + minimum_tail[arm_index + 1] >= best_score:
                continue
            used_object_ids.add(object_id)
            chosen.append(value)
            visit(arm_index + 1, next_score)
            chosen.pop()
            used_object_ids.remove(object_id)

    visit(0, 0.0)
    return best_combination


# ---------------------------------------------------------------------------
# Stock-road fitting


def _fast_chord_endpoint(
    self: Any,
    start_distance: float,
    chord_length: float,
    maximum_distance: float,
):
    """Use cumulative-distance bisection instead of rescanning every vertex."""

    if chord_length <= 0.0 or maximum_distance <= start_distance + 1.0e-9:
        return None
    origin_x, origin_z, _ = self.point(start_distance)

    lower = bisect_right(self.cumulative, start_distance + 1.0e-9)
    upper = bisect_left(self.cumulative, maximum_distance - 1.0e-9)
    internal = self.cumulative[lower:upper]
    breakpoints = (start_distance, *internal, maximum_distance)
    radius_squared = chord_length * chord_length

    for distance0, distance1 in zip(breakpoints, breakpoints[1:]):
        ax, az, _ = self.point(distance0)
        bx, bz, _ = self.point(distance1)
        vx, vz = bx - ax, bz - az
        denominator = vx * vx + vz * vz
        if denominator <= 1.0e-12:
            continue
        ox, oz = ax - origin_x, az - origin_z
        linear = 2.0 * (ox * vx + oz * vz)
        constant = ox * ox + oz * oz - radius_squared
        discriminant = linear * linear - 4.0 * denominator * constant
        if discriminant < -1.0e-8:
            continue
        root = math.sqrt(max(0.0, discriminant))
        fractions = sorted((
            (-linear - root) / (2.0 * denominator),
            (-linear + root) / (2.0 * denominator),
        ))
        for fraction in fractions:
            if fraction < -1.0e-8 or fraction > 1.0 + 1.0e-8:
                continue
            fraction = max(0.0, min(1.0, fraction))
            distance = distance0 + (distance1 - distance0) * fraction
            if distance <= start_distance + 1.0e-7:
                continue
            x = ax + vx * fraction
            z = az + vz * fraction
            heading = math.degrees(
                math.atan2(x - origin_x, z - origin_z)
            ) % 360.0
            return distance, x, z, heading
    return None


def _fast_maximum_chord_deviation(
    self: Any,
    start_distance: float,
    end_distance: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    lower = bisect_right(self.cumulative, start_distance + 1.0e-7)
    upper = bisect_left(self.cumulative, end_distance - 1.0e-7)
    maximum = 0.0
    for point in self.points[lower:upper]:
        maximum = max(
            maximum,
            _playability._point_segment_distance(point, start, end),
        )
    return maximum


@lru_cache(maxsize=128)
def _ordered_road_pieces(pieces: tuple[Any, ...]) -> tuple[Any, ...]:
    unique = {piece.nominal_length: piece for piece in pieces}
    return tuple(sorted(
        unique.values(),
        key=lambda piece: (-piece.length_metres, piece.model_path.casefold()),
    ))


def _fast_road_piece_sequence(target_length: float, pieces: Sequence[Any]):
    if target_length <= 0.05 or not pieces:
        return ()
    ordered = _ordered_road_pieces(tuple(pieces))
    shortest = min(piece.length_metres for piece in ordered)
    for piece in ordered:
        if piece.length_metres <= target_length + shortest * 0.5:
            return (piece,)
    return (ordered[-1],)


@dataclass(frozen=True, slots=True)
class _HeadingIndex:
    bucket_size: float
    segments: tuple[tuple[tuple[float, float], tuple[float, float], float], ...]
    buckets: Mapping[tuple[int, int], tuple[int, ...]]


@lru_cache(maxsize=2048)
def _heading_index(points: tuple[tuple[float, float], ...]) -> _HeadingIndex:
    bucket_size = 16.0
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    mutable: dict[tuple[int, int], list[int]] = {}
    for start, end in zip(points, points[1:]):
        if math.dist(start, end) <= 1.0e-9:
            continue
        heading = math.degrees(
            math.atan2(end[0] - start[0], end[1] - start[1])
        ) % 360.0
        segment_index = len(segments)
        segments.append((start, end, heading))
        x0 = math.floor(min(start[0], end[0]) / bucket_size)
        x1 = math.floor(max(start[0], end[0]) / bucket_size)
        z0 = math.floor(min(start[1], end[1]) / bucket_size)
        z1 = math.floor(max(start[1], end[1]) / bucket_size)
        for bz in range(z0, z1 + 1):
            for bx in range(x0, x1 + 1):
                mutable.setdefault((bx, bz), []).append(segment_index)
    return _HeadingIndex(
        bucket_size,
        tuple(segments),
        {key: tuple(values) for key, values in mutable.items()},
    )


def _fast_nearest_polyline_heading(
    points: Sequence[tuple[float, float]],
    point: tuple[float, float],
) -> float:
    frozen = tuple(points)
    if len(frozen) < 2:
        return _ORIGINAL_NEAREST_POLYLINE_HEADING(points, point)
    index = _heading_index(frozen)
    key = (
        math.floor(point[0] / index.bucket_size),
        math.floor(point[1] / index.bucket_size),
    )
    best: tuple[float, int, float] | None = None
    for segment_index in index.buckets.get(key, ()):
        start, end, heading = index.segments[segment_index]
        distance = _playability._point_segment_distance_2d(point, start, end)
        candidate = (distance, segment_index, heading)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    # Fitted stock-piece endpoints lie on the source run.  A local zero-distance
    # hit is therefore also the global minimum.  Unexpected/off-run callers keep
    # the historical full scan.
    if best is not None and best[0] <= 1.0e-9:
        return best[2]
    return _ORIGINAL_NEAREST_POLYLINE_HEADING(points, point)


# ---------------------------------------------------------------------------
# Forest placement


@dataclass(slots=True)
class _CorridorSeenState:
    owner: Any
    marks: list[int]
    epoch: int = 0


_CORRIDOR_SEEN: dict[int, _CorridorSeenState] = {}


def _corridor_seen_state(owner: Any) -> _CorridorSeenState:
    identity = id(owner)
    state = _CORRIDOR_SEEN.get(identity)
    if state is None or state.owner is not owner or len(state.marks) != len(owner.corridors):
        state = _CorridorSeenState(owner, [0] * len(owner.corridors))
        _CORRIDOR_SEEN[identity] = state
        while len(_CORRIDOR_SEEN) > 8:
            oldest = next(iter(_CORRIDOR_SEEN))
            if oldest == identity and len(_CORRIDOR_SEEN) > 1:
                oldest = next(key for key in _CORRIDOR_SEEN if key != identity)
            _CORRIDOR_SEEN.pop(oldest, None)
    return state


def _fast_corridor_intersects_rectangle(
    self: Any,
    minimum_x: float,
    minimum_z: float,
    maximum_x: float,
    maximum_z: float,
) -> bool:
    bucket = self.bucket_size
    x0 = math.floor(minimum_x / bucket)
    z0 = math.floor(minimum_z / bucket)
    x1 = math.floor(maximum_x / bucket)
    z1 = math.floor(maximum_z / bucket)

    def intersects(index: int) -> bool:
        start, end, radius = self.corridors[index]
        return _osm._segment_intersects_rectangle(
            start,
            end,
            minimum_x - radius,
            minimum_z - radius,
            maximum_x + radius,
            maximum_z + radius,
        )

    if x0 == x1 and z0 == z1:
        return any(intersects(index) for index in self.buckets.get((x0, z0), ()))

    state = _corridor_seen_state(self)
    state.epoch += 1
    if state.epoch >= 2_000_000_000:
        state.marks[:] = [0] * len(state.marks)
        state.epoch = 1
    epoch = state.epoch
    marks = state.marks
    for bz in range(z0, z1 + 1):
        for bx in range(x0, x1 + 1):
            for index in self.buckets.get((bx, bz), ()):
                if marks[index] == epoch:
                    continue
                marks[index] = epoch
                if intersects(index):
                    return True
    return False


def _fast_roadside_vegetation_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    block_size: float,
    *,
    label: str,
    minimum_spacing: float,
    candidate_count: int,
) -> Iterable[tuple[float, float, float, int]]:
    """Preserve priority order while replacing a full sort with a lazy heap."""

    root = hashlib.blake2s(
        f"{seed}:forest-roadside:{label}:{column}:{row}".encode("utf-8"),
        digest_size=16,
    ).digest()
    half_span = block_size * 0.46
    # Include generation index as the secondary heap key. The original stable
    # sort used only priority, so equal priorities retained generation order.
    raw: list[tuple[int, int, float, float, float, int]] = []
    for candidate_index in range(max(1, int(candidate_count))):
        digest = hashlib.blake2s(
            root + candidate_index.to_bytes(4, "little"), digest_size=16
        ).digest()
        priority = int.from_bytes(digest[:4], "little")
        unit_x = int.from_bytes(digest[4:8], "little") / 0xFFFFFFFF
        unit_z = int.from_bytes(digest[8:12], "little") / 0xFFFFFFFF
        candidate_x = x + (unit_x * 2.0 - 1.0) * half_span
        candidate_z = z + (unit_z * 2.0 - 1.0) * half_span
        heading = float(int.from_bytes(digest[12:14], "little") % 360)
        variant = int.from_bytes(digest[14:], "little")
        raw.append((
            priority,
            candidate_index,
            candidate_x,
            candidate_z,
            heading,
            variant,
        ))
    heapq.heapify(raw)

    minimum_distance = max(0.0, float(minimum_spacing))
    if minimum_distance <= 0.0:
        while raw:
            _priority, _index, candidate_x, candidate_z, heading, variant = heapq.heappop(raw)
            yield candidate_x, candidate_z, heading, variant
        return

    minimum_distance2 = minimum_distance * minimum_distance
    inverse_spacing = 1.0 / minimum_distance
    span = half_span * 2.0
    bucket_columns = max(1, int(math.floor(span * inverse_spacing)) + 1)
    accepted: list[tuple[float, float, float, int]] = []
    accepted_buckets: list[list[int]] = [
        [] for _ in range(bucket_columns * bucket_columns)
    ]
    origin_x = x - half_span
    origin_z = z - half_span

    while raw:
        _priority, _index, candidate_x, candidate_z, heading, variant = heapq.heappop(raw)
        bucket_x = min(
            bucket_columns - 1,
            max(0, int((candidate_x - origin_x) * inverse_spacing)),
        )
        bucket_z = min(
            bucket_columns - 1,
            max(0, int((candidate_z - origin_z) * inverse_spacing)),
        )
        too_close = False
        for neighbour_z in range(max(0, bucket_z - 1), min(bucket_columns - 1, bucket_z + 1) + 1):
            bucket_offset = neighbour_z * bucket_columns
            for neighbour_x in range(max(0, bucket_x - 1), min(bucket_columns - 1, bucket_x + 1) + 1):
                for accepted_index in accepted_buckets[bucket_offset + neighbour_x]:
                    other_x, other_z, _heading, _variant = accepted[accepted_index]
                    if (
                        (candidate_x - other_x) ** 2
                        + (candidate_z - other_z) ** 2
                        < minimum_distance2
                    ):
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            continue
        accepted_index = len(accepted)
        result = (candidate_x, candidate_z, heading, variant)
        accepted.append(result)
        accepted_buckets[bucket_z * bucket_columns + bucket_x].append(accepted_index)
        yield result


def _fast_place_cluster_at(
    *,
    variant: Any,
    elevations: Sequence[float],
    raster: Any,
    road_corridors: Sequence[object],
    spec: object,
    x: float,
    z: float,
    heading: float,
    require_forest: bool,
    minimum_forest_fraction: float,
    maximum_relief: float,
    maximum_burial: float,
    maximum_float: float,
    clearance: float,
    avoid_roads: bool = True,
):
    """Ground one cluster while sampling every proxy's terrain only once."""

    world_size = float(getattr(spec, "world_size"))
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))
    world_name = str(getattr(spec, "name"))
    margin = max(
        0.0, float(getattr(spec, "forest_cluster_footprint_margin", 0.75))
    )
    if not (0.0 <= x < world_size and 0.0 <= z < world_size):
        return None

    gradient_x, gradient_z = _osm._local_terrain_gradient(
        elevations, cells, cell_size, x, z
    )
    heading, grade = _osm._cluster_heading_and_grade(
        gradient_x, gradient_z, heading % 360.0, variant.slope_axis
    )
    polygon = _osm._oriented_rectangle(
        x, z, variant.width_m, variant.length_m, heading, margin=margin
    )
    if not all(0.0 <= px < world_size and 0.0 <= pz < world_size for px, pz in polygon):
        return None
    if (
        bool(getattr(spec, "forest_low_anchor", False))
        and world_size >= 200.0
        and variant.category in {"border", "interior", "undergrowth"}
    ):
        proxy_edge_guard = max(8.0, min(18.0, cell_size * 0.55))
        if not all(
            proxy_edge_guard <= px <= world_size - proxy_edge_guard
            and proxy_edge_guard <= pz <= world_size - proxy_edge_guard
            for px, pz in polygon
        ):
            return None

    # Road rejection is independent of terrain support. Do this cheap indexed
    # test before polygon/proxy sampling on the many candidates beside roads.
    if avoid_roads and _osm.forest_block_intersects_road_corridors(
        road_corridors,
        x,
        z,
        block_size=max(variant.width_m, variant.length_m) + 2.0 * margin,
    ):
        return None

    minimum, maximum = _osm._polygon_elevation_extrema(
        elevations, cells, cell_size, polygon
    )
    relief = maximum - minimum
    if relief > min(max(0.0, maximum_relief), variant.maximum_relief_m):
        return None

    angle = math.radians(heading)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    supports: list[float] = []
    proxy_records: list[tuple[bool, float, float]] = []
    forest_points = 0
    proxy_points = 0

    for model, local_x, local_z, _scale, _proxy_heading in variant.proxy_layout:
        world_x = x + local_x * width_axis[0] + local_z * length_axis[0]
        world_z = z + local_x * width_axis[1] + local_z * length_axis[1]
        if not (0.0 <= world_x < world_size and 0.0 <= world_z < world_size):
            return None
        in_forest = _osm._mask_at(
            raster.forest, cells, world_size, world_x, world_z
        )
        proxy_points += 1
        forest_points += int(in_forest)
        if require_forest and not in_forest:
            return None
        if (
            _osm._mask_at(raster.water, cells, world_size, world_x, world_z)
            or _osm._mask_at(raster.roads, cells, world_size, world_x, world_z)
            or _osm._mask_at(raster.buildings, cells, world_size, world_x, world_z)
        ):
            return None
        model_y = grade * (
            local_x if variant.slope_axis == "width" else local_z
        )
        minimum_ground, maximum_ground = _osm._triangle_elevation_bounds(
            elevations, cells, cell_size, world_x, world_z
        )
        supports.extend((minimum_ground - model_y, maximum_ground - model_y))
        proxy_records.append((
            _osm._forest_proxy_is_tree(model),
            model_y,
            minimum_ground,
        ))

    if not supports:
        return None
    if forest_points / proxy_points < minimum_forest_fraction:
        return None

    if variant.category in {"border", "undergrowth", "ditch", "rural"}:
        fitted = _osm._non_buried_vegetation_fit(
            supports,
            clearance=clearance,
            maximum_float=maximum_float,
        )
        if fitted is None:
            return None
        anchor, floating = fitted
        burial = 0.0
    else:
        fitted = _osm._terrain_fit_anchor(
            supports,
            clearance=clearance,
            maximum_burial=maximum_burial,
            maximum_float=maximum_float,
        )
        if fitted is None:
            return None
        anchor, burial, floating = fitted

    maximum_tree_float = 0.0
    maximum_bush_float = 0.0
    tree_count = 0
    bush_count = 0
    for is_tree, model_y, minimum_ground in proxy_records:
        proxy_float = max(0.0, (anchor + model_y) - minimum_ground)
        if is_tree:
            tree_count += 1
            maximum_tree_float = max(maximum_tree_float, proxy_float)
        else:
            bush_count += 1
            maximum_bush_float = max(maximum_bush_float, proxy_float)

    tree_limit = min(
        max(0.0, maximum_float),
        max(
            0.0,
            float(getattr(spec, "forest_cluster_tree_maximum_float", 0.20)),
        ),
    )
    bush_limit = min(
        max(0.0, maximum_float),
        max(
            0.0,
            float(getattr(spec, "forest_cluster_bush_maximum_float", 0.60)),
        ),
    )
    if (
        tree_count and maximum_tree_float > tree_limit + 1.0e-9
    ) or (
        bush_count and maximum_bush_float > bush_limit + 1.0e-9
    ):
        return None

    floating = max(floating, maximum_tree_float, maximum_bush_float)
    return (
        _osm.cluster_model_path(world_name, variant.name, grade),
        x,
        anchor,
        z,
        heading,
        variant.name,
        relief,
        burial,
        floating,
    )


def install_object_placement_performance_policy() -> None:
    """Install object-placement hot-loop optimizations after base policies."""

    global _INSTALLED
    global _ORIGINAL_APPROACH_CHOICE, _ORIGINAL_PLAN_APPLICATION
    global _ORIGINAL_CHORD_ENDPOINT, _ORIGINAL_MAXIMUM_CHORD_DEVIATION
    global _ORIGINAL_ROAD_PIECE_SEQUENCE, _ORIGINAL_NEAREST_POLYLINE_HEADING
    global _ORIGINAL_CORRIDOR_INTERSECTS_RECTANGLE
    global _ORIGINAL_ROADSIDE_VEGETATION_CANDIDATES, _ORIGINAL_PLACE_CLUSTER_AT
    if _INSTALLED:
        return

    _ORIGINAL_APPROACH_CHOICE = _paved_perf._approach_choice_to_target
    _ORIGINAL_PLAN_APPLICATION = _paved_perf._plan_application
    _ORIGINAL_CHORD_ENDPOINT = _playability._PolylineMeasure.chord_endpoint
    _ORIGINAL_MAXIMUM_CHORD_DEVIATION = (
        _playability._PolylineMeasure.maximum_chord_deviation
    )
    _ORIGINAL_ROAD_PIECE_SEQUENCE = _playability._road_piece_sequence
    _ORIGINAL_NEAREST_POLYLINE_HEADING = _playability._nearest_polyline_heading
    _ORIGINAL_CORRIDOR_INTERSECTS_RECTANGLE = (
        _osm.IndexedRoadCorridors.intersects_rectangle
    )
    _ORIGINAL_ROADSIDE_VEGETATION_CANDIDATES = (
        _osm._roadside_vegetation_candidates
    )
    _ORIGINAL_PLACE_CLUSTER_AT = _osm._place_cluster_at

    _paved_perf._approach_choice_to_target = _fast_approach_choice_to_target
    _paved_perf._plan_application = _fast_plan_application
    _playability._PolylineMeasure.chord_endpoint = _fast_chord_endpoint
    _playability._PolylineMeasure.maximum_chord_deviation = (
        _fast_maximum_chord_deviation
    )
    _playability._road_piece_sequence = _fast_road_piece_sequence
    _playability._nearest_polyline_heading = _fast_nearest_polyline_heading
    _osm.IndexedRoadCorridors.intersects_rectangle = (
        _fast_corridor_intersects_rectangle
    )
    _osm._roadside_vegetation_candidates = _fast_roadside_vegetation_candidates
    _osm._place_cluster_at = _fast_place_cluster_at
    _INSTALLED = True
