# SPDX-License-Identifier: GPL-3.0-or-later
"""Place stock CWA bridges only over water and join modules through shared 3-D seams.

Generated bridge P3Ds proved unreliable in OFP/CWA, so Worldgen forces bridge
output through the stock Resistance/Nogova ``O\\Hous\\most_stred30.p3d`` model.

The stock model's Roadway LOD is about 50.190 m long.  Bridge-tagged OSM ways can
extend hundreds of metres beyond the actual water crossing, while the stock core
planner historically spread a whole-number module count across that complete
length.  That compressed neighbouring modules and exposed their end geometry as
walls in the carriageway.

This policy therefore:

* clips stock bridge planning to the first/last actual in-game water on the way;
* uses an exact whole number of measured stock Roadway lengths;
* leaves the dry prefix/suffix to the ordinary road fitter;
* reconstructs every module from shared visible-deck 3-D joint points; and
* validates every internal seam before returning world objects.

A final 0.7 m world-space downward tuning offset is retained from in-game testing
so the rendered stock bridge deck matches the adjoining stock road surface.
"""
from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import math
from typing import Sequence

from . import generator as _generator
from . import osm as _osm

_ORIGINAL_GENERATE_WORLD_OBJECTS = None
_ORIGINAL_LOAD_NONROAD_OBJECTS = None
_ORIGINAL_EXTEND_BRIDGE_SPAN = None
_INSTALLED = False

_STOCK_MODEL = _osm.NOGOVA_BRIDGE_MODEL.casefold()

# Measured from the original CWA/Resistance O\Hous\most_stred30.p3d ODOL7.
_STOCK_ROADWAY_HALF_LENGTH_METRES = 25.095142364501953
_STOCK_ROADWAY_LENGTH_METRES = _STOCK_ROADWAY_HALF_LENGTH_METRES * 2.0
# The placement/cache identity uses the actual Roadway join length.  Neighbouring
# modules therefore meet at one joint instead of being compressed across a span.
_STOCK_MODULE_SPACING_METRES = _STOCK_ROADWAY_LENGTH_METRES
_STOCK_ROADWAY_LOCAL_Y_METRES = 12.982887268066406
_STOCK_VISIBLE_DECK_LOCAL_Y_METRES = 13.049331665039062
_BRIDGE_WORLD_DOWNWARD_OFFSET_METRES = 0.7

_WATER_SAMPLE_STEP_METRES = 2.0
_WATER_BOUNDARY_REFINEMENT_STEPS = 12
_CHAIN_ENDPOINT_TOLERANCE_METRES = 6.0
_CHAIN_MINIMUM_CENTER_SPACING_METRES = _STOCK_MODULE_SPACING_METRES - 6.0
_CHAIN_MAXIMUM_CENTER_SPACING_METRES = _STOCK_MODULE_SPACING_METRES + 1.0
_CHAIN_HEADING_TOLERANCE_DEGREES = 15.0
_APPROACH_SEARCH_METRES = 60.0
_APPROACH_SEARCH_STEP_METRES = 2.0
_MAXIMUM_ANCHORED_BRIDGE_PITCH_DEGREES = 12.0
_MAXIMUM_SEAM_HORIZONTAL_ERROR_METRES = 0.10
_MAXIMUM_SEAM_VERTICAL_ERROR_METRES = 0.02


@dataclass(frozen=True, slots=True)
class StockBridgeSpanPlan:
    """Straight stock-model corridor covering only the actual wet crossing."""

    points: tuple[tuple[float, float], tuple[float, float]]
    module_count: int
    wet_start: tuple[float, float]
    wet_end: tuple[float, float]
    wet_length: float

    @property
    def length(self) -> float:
        return self.module_count * _STOCK_MODULE_SPACING_METRES


class _StockBridgeSpecProxy:
    """Read-through spec view for non-dataclass compatibility callers/tests."""

    __slots__ = ("_base", "procedural_bridges", "bridge_module_length")

    def __init__(self, base) -> None:
        self._base = base
        self.procedural_bridges = False
        self.bridge_module_length = _STOCK_MODULE_SPACING_METRES

    def __getattr__(self, name):
        return getattr(self._base, name)


def _stock_bridge_spec(spec):
    """Return a stock-bridge spec whose cache identity includes real join length."""

    procedural = bool(getattr(spec, "procedural_bridges", True))
    try:
        module_length = float(getattr(spec, "bridge_module_length", 0.0))
    except (TypeError, ValueError):
        module_length = 0.0
    if (
        not procedural
        and abs(module_length - _STOCK_MODULE_SPACING_METRES) <= 1.0e-9
    ):
        return spec
    if is_dataclass(spec):
        return replace(
            spec,
            procedural_bridges=False,
            bridge_module_length=_STOCK_MODULE_SPACING_METRES,
        )
    return _StockBridgeSpecProxy(spec)


def _is_stock_bridge(obj) -> bool:
    return (
        str(getattr(obj, "model_path", "")).replace("/", "\\").casefold()
        == _STOCK_MODEL
    )


def _clean_points(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        value = (float(point[0]), float(point[1]))
        if not cleaned or math.dist(value, cleaned[-1]) > 0.05:
            cleaned.append(value)
    return tuple(cleaned)


def _polyline_measure(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    cleaned = _clean_points(points)
    if len(cleaned) < 2:
        return cleaned, (0.0,)
    cumulative = [0.0]
    for start, end in zip(cleaned, cleaned[1:]):
        cumulative.append(cumulative[-1] + math.dist(start, end))
    return cleaned, tuple(cumulative)


def _point_at_measure(
    points: Sequence[tuple[float, float]],
    cumulative: Sequence[float],
    distance: float,
) -> tuple[float, float]:
    if len(points) < 2:
        return tuple(points[0]) if points else (0.0, 0.0)
    target = max(0.0, min(float(cumulative[-1]), float(distance)))
    for index in range(len(points) - 1):
        start_distance = float(cumulative[index])
        end_distance = float(cumulative[index + 1])
        if target > end_distance and index + 2 < len(points):
            continue
        length = end_distance - start_distance
        if length <= 1.0e-9:
            return float(points[index + 1][0]), float(points[index + 1][1])
        fraction = (target - start_distance) / length
        start, end = points[index], points[index + 1]
        return (
            float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
            float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
        )
    return float(points[-1][0]), float(points[-1][1])


def _ground_at_measure(points, cumulative, distance, elevations, spec) -> float:
    x, z = _point_at_measure(points, cumulative, distance)
    return float(
        _osm._sample_elevation(
            elevations, spec.cells, spec.cell_size, x, z
        )
    )


def _wet_at_measure(points, cumulative, distance, elevations, spec) -> bool:
    epsilon = float(getattr(_osm, "BRIDGE_WATER_EPSILON_METRES", 0.05))
    return _ground_at_measure(
        points, cumulative, distance, elevations, spec
    ) < float(spec.sea_level) - epsilon


def _refine_wet_start(
    points, cumulative, dry_distance, wet_distance, elevations, spec
) -> float:
    low = float(dry_distance)
    high = float(wet_distance)
    for _ in range(_WATER_BOUNDARY_REFINEMENT_STEPS):
        middle = (low + high) * 0.5
        if _wet_at_measure(points, cumulative, middle, elevations, spec):
            high = middle
        else:
            low = middle
    return high


def _refine_wet_end(
    points, cumulative, wet_distance, dry_distance, elevations, spec
) -> float:
    low = float(wet_distance)
    high = float(dry_distance)
    for _ in range(_WATER_BOUNDARY_REFINEMENT_STEPS):
        middle = (low + high) * 0.5
        if _wet_at_measure(points, cumulative, middle, elevations, spec):
            low = middle
        else:
            high = middle
    return low


def stock_bridge_span_plan(
    points: Sequence[tuple[float, float]],
    elevations,
    spec,
) -> StockBridgeSpanPlan | None:
    """Return the stock bridge footprint covering only actual below-sea terrain.

    The OSM bridge way remains the candidate corridor, but its dry prefix/suffix
    are not converted into bridge modules.  A straight stock chain is centred on
    the wet interval and expanded only to the next whole measured Roadway length.
    """

    if elevations is None or spec is None:
        return None
    cleaned, cumulative = _polyline_measure(points)
    if len(cleaned) < 2 or cumulative[-1] <= 0.1:
        return None

    total = float(cumulative[-1])
    step = max(
        0.5,
        min(
            _WATER_SAMPLE_STEP_METRES,
            max(0.5, float(getattr(spec, "cell_size", 10.0)) * 0.20),
        ),
    )
    sample_count = max(1, int(math.ceil(total / step)))
    distances = [min(total, index * total / sample_count) for index in range(sample_count + 1)]
    wet = [
        _wet_at_measure(cleaned, cumulative, distance, elevations, spec)
        for distance in distances
    ]
    wet_indices = [index for index, state in enumerate(wet) if state]
    if not wet_indices:
        return None

    first = wet_indices[0]
    last = wet_indices[-1]
    wet_start_distance = distances[first]
    wet_end_distance = distances[last]
    if first > 0 and not wet[first - 1]:
        wet_start_distance = _refine_wet_start(
            cleaned,
            cumulative,
            distances[first - 1],
            distances[first],
            elevations,
            spec,
        )
    if last + 1 < len(distances) and not wet[last + 1]:
        wet_end_distance = _refine_wet_end(
            cleaned,
            cumulative,
            distances[last],
            distances[last + 1],
            elevations,
            spec,
        )

    wet_start = _point_at_measure(cleaned, cumulative, wet_start_distance)
    wet_end = _point_at_measure(cleaned, cumulative, wet_end_distance)
    dx = wet_end[0] - wet_start[0]
    dz = wet_end[1] - wet_start[1]
    wet_length = math.hypot(dx, dz)
    if wet_length <= 0.10:
        return None
    unit_x, unit_z = dx / wet_length, dz / wet_length

    # Use the smallest whole number of stock modules that covers the wet chord.
    # Any unavoidable excess becomes a small land overhang instead of hundreds
    # of metres of bridge-tagged dry OSM approach.
    tolerance = max(1.0e-6, _STOCK_MODULE_SPACING_METRES * 1.0e-9)
    module_count = max(
        1,
        int(
            math.ceil(
                (wet_length - tolerance) / _STOCK_MODULE_SPACING_METRES
            )
        ),
    )
    target_length = module_count * _STOCK_MODULE_SPACING_METRES
    half_target = target_length * 0.5
    centre = (
        (wet_start[0] + wet_end[0]) * 0.5,
        (wet_start[1] + wet_end[1]) * 0.5,
    )

    # If the mapped bridge way is longer than the necessary stock chain, keep the
    # complete chain inside that source extent. This maximises ordinary road on
    # both dry approaches without changing the wet coverage.
    projections = [
        (point[0] - centre[0]) * unit_x + (point[1] - centre[1]) * unit_z
        for point in cleaned
    ]
    source_min = min(projections)
    source_max = max(projections)
    if source_max - source_min >= target_length:
        minimum_shift = source_min + half_target
        maximum_shift = source_max - half_target
        shift = max(minimum_shift, min(0.0, maximum_shift))
        centre = (
            centre[0] + unit_x * shift,
            centre[1] + unit_z * shift,
        )

    start = (
        centre[0] - unit_x * half_target,
        centre[1] - unit_z * half_target,
    )
    end = (
        centre[0] + unit_x * half_target,
        centre[1] + unit_z * half_target,
    )
    world_size = float(getattr(spec, "world_size", 0.0) or 0.0)
    if world_size > 0.0 and not all(
        0.0 <= value < world_size
        for point in (start, end)
        for value in point
    ):
        # Edge-of-world bridges are rare; retain the established core behaviour
        # rather than silently distort a fixed-length stock module chain.
        return None

    return StockBridgeSpanPlan(
        points=(start, end),
        module_count=module_count,
        wet_start=wet_start,
        wet_end=wet_end,
        wet_length=wet_length,
    )


def _extend_stock_bridge_to_wet_span(
    points,
    elevations,
    spec,
    module_length,
    *,
    feature=None,
    dataset=None,
    projection=None,
    raster=None,
):
    """Replace plateau extension with a minimal wet-only stock bridge corridor."""

    if bool(getattr(spec, "procedural_bridges", True)):
        return _ORIGINAL_EXTEND_BRIDGE_SPAN(
            points,
            elevations,
            spec,
            module_length,
            feature=feature,
            dataset=dataset,
            projection=projection,
            raster=raster,
        )
    plan = stock_bridge_span_plan(points, elevations, spec)
    if plan is not None:
        return plan.points
    return _ORIGINAL_EXTEND_BRIDGE_SPAN(
        points,
        elevations,
        spec,
        module_length,
        feature=feature,
        dataset=dataset,
        projection=projection,
        raster=raster,
    )


def _visible_deck_point(obj, local_z: float) -> tuple[float, float, float]:
    """Transform a central visible-deck point from model to world coordinates."""

    heading = math.radians(float(obj.heading_degrees))
    pitch = math.radians(float(obj.pitch_degrees))
    sh, ch = math.sin(heading), math.cos(heading)
    sp, cp = math.sin(pitch), math.cos(pitch)
    local_y = _STOCK_VISIBLE_DECK_LOCAL_Y_METRES
    z = float(local_z)
    return (
        float(obj.x) - sh * sp * local_y + sh * cp * z,
        float(obj.y) + cp * local_y + sp * z,
        float(obj.z) - ch * sp * local_y + ch * cp * z,
    )


def _axis_endpoints(obj) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the actual visible-deck plan endpoints including object pitch."""

    first = _visible_deck_point(obj, -_STOCK_ROADWAY_HALF_LENGTH_METRES)
    second = _visible_deck_point(obj, _STOCK_ROADWAY_HALF_LENGTH_METRES)
    return ((first[0], first[2]), (second[0], second[2]))


def _axial_heading_difference(a: float, b: float) -> float:
    delta = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return min(delta, abs(180.0 - delta))


def _bridge_components(objects) -> tuple[tuple[int, ...], ...]:
    """Group physically connected stock bridge modules into independent chains."""

    indices = [index for index, obj in enumerate(objects) if _is_stock_bridge(obj)]
    if not indices:
        return ()

    endpoints = {index: _axis_endpoints(objects[index]) for index in indices}
    neighbours: dict[int, list[int]] = {index: [] for index in indices}
    tolerance_sq = _CHAIN_ENDPOINT_TOLERANCE_METRES ** 2

    for position, left_index in enumerate(indices):
        left = objects[left_index]
        left_endpoints = endpoints[left_index]
        for right_index in indices[position + 1 :]:
            right = objects[right_index]
            left_centre = _visible_deck_point(left, 0.0)
            right_centre = _visible_deck_point(right, 0.0)
            centre_distance = math.hypot(
                left_centre[0] - right_centre[0],
                left_centre[2] - right_centre[2],
            )
            if not (
                _CHAIN_MINIMUM_CENTER_SPACING_METRES
                <= centre_distance
                <= _CHAIN_MAXIMUM_CENTER_SPACING_METRES
            ):
                continue
            if (
                _axial_heading_difference(
                    left.heading_degrees, right.heading_degrees
                )
                > _CHAIN_HEADING_TOLERANCE_DEGREES
            ):
                continue
            right_endpoints = endpoints[right_index]
            if min(
                (ax - bx) ** 2 + (az - bz) ** 2
                for ax, az in left_endpoints
                for bx, bz in right_endpoints
            ) > tolerance_sq:
                continue
            neighbours[left_index].append(right_index)
            neighbours[right_index].append(left_index)

    components: list[tuple[int, ...]] = []
    remaining = set(indices)
    while remaining:
        root = min(remaining)
        stack = [root]
        component: list[int] = []
        remaining.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in neighbours[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _ordered_component(
    objects,
    component: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[float, float], tuple[float, float]] | None:
    """Return chain indices in travel order, common axis and plan midpoint."""

    if not component:
        return None
    if len(component) == 1:
        obj = objects[component[0]]
        heading = math.radians(float(obj.heading_degrees))
        axis = (math.sin(heading), math.cos(heading))
        deck_centre = _visible_deck_point(obj, 0.0)
        return component, axis, (deck_centre[0], deck_centre[2])

    centres = {
        index: _visible_deck_point(objects[index], 0.0)
        for index in component
    }
    best = None
    for offset, first_index in enumerate(component):
        first = centres[first_index]
        for second_index in component[offset + 1 :]:
            second = centres[second_index]
            distance_sq = (
                (second[0] - first[0]) ** 2
                + (second[2] - first[2]) ** 2
            )
            if best is None or distance_sq > best[0]:
                best = (distance_sq, first_index, second_index)
    if best is None or best[0] <= 1.0e-9:
        return None

    first = centres[best[1]]
    second = centres[best[2]]
    dx = second[0] - first[0]
    dz = second[2] - first[2]
    length = math.hypot(dx, dz)
    axis = (dx / length, dz / length)
    projections = {
        index: centres[index][0] * axis[0] + centres[index][2] * axis[1]
        for index in component
    }
    ordered = tuple(sorted(component, key=lambda index: projections[index]))

    # Orient the common axis to agree with the first module's forward heading.
    first_heading = math.radians(float(objects[ordered[0]].heading_degrees))
    forward = (math.sin(first_heading), math.cos(first_heading))
    if forward[0] * axis[0] + forward[1] * axis[1] < 0.0:
        axis = (-axis[0], -axis[1])
        ordered = tuple(reversed(ordered))

    first_centre = centres[ordered[0]]
    last_centre = centres[ordered[-1]]
    midpoint = (
        (first_centre[0] + last_centre[0]) * 0.5,
        (first_centre[2] + last_centre[2]) * 0.5,
    )
    return ordered, axis, midpoint


def _dry_approach_height(
    endpoint: tuple[float, float],
    outward: tuple[float, float],
    raster,
    elevations,
    spec,
) -> float | None:
    """Find nearest dry stock-road surface height outside one bridge end."""

    world_size = float(spec.world_size)
    distance = 0.0
    while distance <= _APPROACH_SEARCH_METRES + 1.0e-9:
        x = endpoint[0] + outward[0] * distance
        z = endpoint[1] + outward[1] * distance
        if 0.0 <= x < world_size and 0.0 <= z < world_size:
            # Final in-game water is determined by terrain below sea level.  The
            # OSM water mask alone is not sufficient after terrain solving.
            ground = _osm._sample_elevation(
                elevations, spec.cells, spec.cell_size, x, z
            )
            epsilon = float(
                getattr(_osm, "BRIDGE_WATER_EPSILON_METRES", 0.05)
            )
            if ground >= float(spec.sea_level) - epsilon:
                deck = (
                    ground
                    + float(_osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES)
                )
                return max(
                    deck,
                    float(spec.sea_level)
                    + float(_osm.NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES),
                )
        distance += _APPROACH_SEARCH_STEP_METRES
    return None


def _model_origin_for_visible_deck_center(
    deck_x: float,
    deck_y: float,
    deck_z: float,
    heading_degrees: float,
    pitch_degrees: float,
) -> tuple[float, float, float]:
    """Convert a desired visible-deck centre to the RVW4 model origin."""

    heading = math.radians(float(heading_degrees))
    pitch = math.radians(float(pitch_degrees))
    sh, ch = math.sin(heading), math.cos(heading)
    sp, cp = math.sin(pitch), math.cos(pitch)
    local_y = _STOCK_VISIBLE_DECK_LOCAL_Y_METRES
    return (
        float(deck_x) + sh * sp * local_y,
        float(deck_y) - cp * local_y,
        float(deck_z) + ch * sp * local_y,
    )


def _model_origin_y_for_visible_deck(
    visible_deck_y: float, pitch_degrees: float
) -> float:
    """Compatibility helper retained for callers/tests that only need Y."""

    pitch = math.radians(float(pitch_degrees))
    return float(visible_deck_y) - (
        _STOCK_VISIBLE_DECK_LOCAL_Y_METRES * math.cos(pitch)
    )


def _component_seam_errors(
    objects,
    ordered: Sequence[int],
) -> tuple[float, float]:
    maximum_horizontal = 0.0
    maximum_vertical = 0.0
    for left_index, right_index in zip(ordered, ordered[1:]):
        left_end = _visible_deck_point(
            objects[left_index], _STOCK_ROADWAY_HALF_LENGTH_METRES
        )
        right_start = _visible_deck_point(
            objects[right_index], -_STOCK_ROADWAY_HALF_LENGTH_METRES
        )
        maximum_horizontal = max(
            maximum_horizontal,
            math.hypot(
                right_start[0] - left_end[0],
                right_start[2] - left_end[2],
            ),
        )
        maximum_vertical = max(
            maximum_vertical,
            abs(right_start[1] - left_end[1]),
        )
    return maximum_horizontal, maximum_vertical


def _anchor_stock_bridge_chains(result, raster, elevations, spec):
    """Rebuild each stock chain from shared visible-deck 3-D joints."""

    if result is None or raster is None or elevations is None or spec is None:
        return result
    objects = list(tuple(getattr(result, "objects", ()) or ()))
    if not objects:
        return result

    changed = False
    for component in _bridge_components(objects):
        ordered_state = _ordered_component(objects, component)
        if ordered_state is None:
            continue
        ordered, axis, midpoint = ordered_state
        count = len(ordered)
        if count <= 0:
            continue

        # First estimate assumes a level chain.  Re-sample once after converting
        # the bank height difference into the stock model's fixed 3-D length.
        horizontal_step = _STOCK_MODULE_SPACING_METRES
        start_height = end_height = None
        for _ in range(2):
            half_span = horizontal_step * count * 0.5
            start = (
                midpoint[0] - axis[0] * half_span,
                midpoint[1] - axis[1] * half_span,
            )
            end = (
                midpoint[0] + axis[0] * half_span,
                midpoint[1] + axis[1] * half_span,
            )
            start_height = _dry_approach_height(
                start, (-axis[0], -axis[1]), raster, elevations, spec
            )
            end_height = _dry_approach_height(
                end, axis, raster, elevations, spec
            )
            if start_height is None or end_height is None:
                break
            per_module_rise = (end_height - start_height) / count
            if abs(per_module_rise) >= _STOCK_MODULE_SPACING_METRES:
                start_height = end_height = None
                break
            horizontal_step = math.sqrt(
                max(
                    0.0,
                    _STOCK_MODULE_SPACING_METRES ** 2
                    - per_module_rise ** 2,
                )
            )

        if start_height is None or end_height is None:
            continue

        start_deck_y = (
            float(start_height) - _BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
        )
        end_deck_y = (
            float(end_height) - _BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
        )
        per_module_rise = (end_deck_y - start_deck_y) / count
        ratio = per_module_rise / _STOCK_MODULE_SPACING_METRES
        if not -1.0 < ratio < 1.0:
            continue
        pitch = math.degrees(math.asin(ratio))
        if abs(pitch) > _MAXIMUM_ANCHORED_BRIDGE_PITCH_DEGREES:
            continue
        horizontal_step = math.sqrt(
            max(
                0.0,
                _STOCK_MODULE_SPACING_METRES ** 2
                - per_module_rise ** 2,
            )
        )
        heading = math.degrees(math.atan2(axis[0], axis[1])) % 360.0
        half_span = horizontal_step * count * 0.5
        first_joint_x = midpoint[0] - axis[0] * half_span
        first_joint_z = midpoint[1] - axis[1] * half_span

        for position, index in enumerate(ordered):
            joint0 = (
                first_joint_x + axis[0] * horizontal_step * position,
                start_deck_y + per_module_rise * position,
                first_joint_z + axis[1] * horizontal_step * position,
            )
            joint1 = (
                first_joint_x + axis[0] * horizontal_step * (position + 1),
                start_deck_y + per_module_rise * (position + 1),
                first_joint_z + axis[1] * horizontal_step * (position + 1),
            )
            deck_centre = (
                (joint0[0] + joint1[0]) * 0.5,
                (joint0[1] + joint1[1]) * 0.5,
                (joint0[2] + joint1[2]) * 0.5,
            )
            origin_x, origin_y, origin_z = _model_origin_for_visible_deck_center(
                deck_centre[0],
                deck_centre[1],
                deck_centre[2],
                heading,
                pitch,
            )
            objects[index] = replace(
                objects[index],
                x=origin_x,
                y=origin_y,
                z=origin_z,
                heading_degrees=heading,
                pitch_degrees=pitch,
            )
            changed = True

        horizontal_error, vertical_error = _component_seam_errors(objects, ordered)
        if (
            horizontal_error > _MAXIMUM_SEAM_HORIZONTAL_ERROR_METRES
            or vertical_error > _MAXIMUM_SEAM_VERTICAL_ERROR_METRES
        ):
            ids = tuple(int(objects[index].object_id) for index in ordered)
            raise RuntimeError(
                "stock bridge seam validation failed for objects "
                f"{ids}: horizontal={horizontal_error:.4f} m, "
                f"vertical={vertical_error:.4f} m"
            )

    if not changed:
        return result
    return replace(result, objects=tuple(objects))


def _generate_world_objects(
    dataset,
    projection,
    raster,
    elevations,
    spec,
    *args,
    **kwargs,
):
    """Use stock wet-only bridge planning, then rebuild shared 3-D seams."""

    stock_spec = _stock_bridge_spec(spec)
    result = _ORIGINAL_GENERATE_WORLD_OBJECTS(
        dataset,
        projection,
        raster,
        elevations,
        stock_spec,
        *args,
        **kwargs,
    )
    return _anchor_stock_bridge_chains(
        result, raster, elevations, stock_spec
    )


def _load_nonroad_objects(*args, **kwargs):
    """Use stock cache identity and repair cached bridge transforms."""

    positional = list(args)
    named = dict(kwargs)

    raster = named.get("raster")
    elevations = named.get("elevations")
    spec = named.get("spec")
    if raster is None and len(positional) >= 3:
        raster = positional[2]
    if elevations is None and len(positional) >= 4:
        elevations = positional[3]
    if spec is None and len(positional) >= 5:
        spec = positional[4]
    if spec is None:
        return _ORIGINAL_LOAD_NONROAD_OBJECTS(*positional, **named)

    stock_spec = _stock_bridge_spec(spec)
    if "spec" in named:
        named["spec"] = stock_spec
    elif len(positional) >= 5:
        positional[4] = stock_spec

    value = _ORIGINAL_LOAD_NONROAD_OBJECTS(*positional, **named)
    if not isinstance(value, tuple) or not value:
        return value
    anchored = _anchor_stock_bridge_chains(
        value[0], raster, elevations, stock_spec
    )
    if anchored is value[0]:
        return value
    return (anchored, *value[1:])


def install_bridge_render_policy() -> None:
    """Install stock wet-span planning before final non-road wrappers capture it."""

    global _ORIGINAL_GENERATE_WORLD_OBJECTS
    global _ORIGINAL_LOAD_NONROAD_OBJECTS
    global _ORIGINAL_EXTEND_BRIDGE_SPAN
    global _INSTALLED

    if _INSTALLED:
        return

    _osm.NOGOVA_BRIDGE_MODULE_LENGTH_METRES = _STOCK_MODULE_SPACING_METRES

    _ORIGINAL_EXTEND_BRIDGE_SPAN = (
        _osm._extend_procedural_bridge_to_approach_plateaus
    )
    _osm._extend_procedural_bridge_to_approach_plateaus = (
        _extend_stock_bridge_to_wet_span
    )

    _ORIGINAL_GENERATE_WORLD_OBJECTS = _osm.generate_world_objects
    _ORIGINAL_LOAD_NONROAD_OBJECTS = _generator._load_nonroad_objects

    _osm.generate_world_objects = _generate_world_objects
    _generator.generate_world_objects = _generate_world_objects
    _generator._load_nonroad_objects = _load_nonroad_objects
    _INSTALLED = True
