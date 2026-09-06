# SPDX-License-Identifier: GPL-3.0-or-later
"""Route CWA bridges through stock models and anchor their ends to road banks.

The OSM bridge planner already has two bridge implementations:

* ``procedural_bridges=True`` authors one world-local generated P3D spanning the
  complete extended bridge; and
* ``procedural_bridges=False`` places the stock Resistance/Nogova 30 m bridge
  modules, with stock-specific footprints and object budgeting.

The generated-P3D route has proven unreliable in OFP/CWA.  Use the stock bridge
planner instead, then correct one weakness in its legacy height recipe: it keeps
an entire bridge chain at the highest sampled approach elevation.  On unequal
banks that makes one end hover above its road.  After stock placement, identify
connected stock bridge chains and fit one continuous deck plane between the two
actual dry-land approach elevations.  The end modules therefore bury naturally
into the banks while every intermediate module remains exactly connected.

The same correction is applied to cached non-road placements, so users do not
need to clear their placement cache after upgrading this policy.
"""
from __future__ import annotations

from dataclasses import is_dataclass, replace
import math

from . import generator as _generator
from . import osm as _osm

_ORIGINAL_GENERATE_WORLD_OBJECTS = None
_ORIGINAL_LOAD_NONROAD_OBJECTS = None
_INSTALLED = False

_STOCK_MODEL = _osm.NOGOVA_BRIDGE_MODEL.casefold()
_STOCK_LENGTH = float(_osm.NOGOVA_BRIDGE_MODULE_LENGTH_METRES)
_CHAIN_ENDPOINT_TOLERANCE_METRES = 6.0
_CHAIN_HEADING_TOLERANCE_DEGREES = 40.0
_APPROACH_SEARCH_METRES = 60.0
_APPROACH_SEARCH_STEP_METRES = 2.0
_MAXIMUM_ANCHORED_BRIDGE_PITCH_DEGREES = 12.0


class _StockBridgeSpecProxy:
    """Read-through spec view for non-dataclass compatibility callers/tests."""

    __slots__ = ("_base", "procedural_bridges")

    def __init__(self, base) -> None:
        self._base = base
        self.procedural_bridges = False

    def __getattr__(self, name):
        return getattr(self._base, name)


def _stock_bridge_spec(spec):
    """Return an equivalent spec whose bridge implementation is stock CWA."""

    if not bool(getattr(spec, "procedural_bridges", True)):
        return spec
    if is_dataclass(spec):
        return replace(spec, procedural_bridges=False)
    return _StockBridgeSpecProxy(spec)


def _is_stock_bridge(obj) -> bool:
    return str(getattr(obj, "model_path", "")).replace("/", "\\").casefold() == _STOCK_MODEL


def _axis_endpoints(obj) -> tuple[tuple[float, float], tuple[float, float]]:
    angle = math.radians(float(getattr(obj, "heading_degrees", 0.0)))
    dx = math.sin(angle) * _STOCK_LENGTH * 0.5
    dz = math.cos(angle) * _STOCK_LENGTH * 0.5
    x = float(obj.x)
    z = float(obj.z)
    return ((x - dx, z - dz), (x + dx, z + dz))


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
            centre_distance = math.hypot(float(left.x) - float(right.x), float(left.z) - float(right.z))
            if centre_distance > _STOCK_LENGTH + _CHAIN_ENDPOINT_TOLERANCE_METRES:
                continue
            if _axial_heading_difference(left.heading_degrees, right.heading_degrees) > _CHAIN_HEADING_TOLERANCE_DEGREES:
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


def _outer_chain_endpoints(objects, component: tuple[int, ...]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the two most distant model endpoints for one connected chain."""

    candidates = [point for index in component for point in _axis_endpoints(objects[index])]
    if len(candidates) < 2:
        return None
    best: tuple[float, tuple[float, float], tuple[float, float]] | None = None
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            distance_sq = (second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2
            if best is None or distance_sq > best[0]:
                best = (distance_sq, first, second)
    if best is None or best[0] <= 1.0e-6:
        return None
    return best[1], best[2]


def _dry_approach_height(
    endpoint: tuple[float, float],
    outward: tuple[float, float],
    raster,
    elevations,
    spec,
) -> float | None:
    """Find the nearest dry road-bank height at or just outside a bridge end."""

    world_size = float(spec.world_size)
    distance = 0.0
    while distance <= _APPROACH_SEARCH_METRES + 1.0e-9:
        x = endpoint[0] + outward[0] * distance
        z = endpoint[1] + outward[1] * distance
        if 0.0 <= x < world_size and 0.0 <= z < world_size:
            if not _osm._mask_at(raster.water, spec.cells, world_size, x, z):
                ground = _osm._sample_elevation(
                    elevations, spec.cells, spec.cell_size, x, z
                )
                deck = ground + float(_osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES)
                return max(
                    deck,
                    float(spec.sea_level) + float(_osm.NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES),
                )
        distance += _APPROACH_SEARCH_STEP_METRES
    return None


def _anchor_stock_bridge_chains(result, raster, elevations, spec):
    """Fit each stock bridge chain continuously between its two dry approaches."""

    if result is None or raster is None or elevations is None or spec is None:
        return result
    objects = list(tuple(getattr(result, "objects", ()) or ()))
    if not objects:
        return result

    changed = False
    for component in _bridge_components(objects):
        outer = _outer_chain_endpoints(objects, component)
        if outer is None:
            continue
        start, end = outer
        span_dx = end[0] - start[0]
        span_dz = end[1] - start[1]
        span_length = math.hypot(span_dx, span_dz)
        if span_length <= 1.0e-6:
            continue
        unit_x = span_dx / span_length
        unit_z = span_dz / span_length

        start_y = _dry_approach_height(
            start, (-unit_x, -unit_z), raster, elevations, spec
        )
        end_y = _dry_approach_height(
            end, (unit_x, unit_z), raster, elevations, spec
        )
        if start_y is None or end_y is None:
            continue

        grade = (end_y - start_y) / span_length
        bridge_pitch = math.degrees(math.atan(grade))
        if abs(bridge_pitch) > _MAXIMUM_ANCHORED_BRIDGE_PITCH_DEGREES:
            # A very steep mismatch usually means the connected-component or
            # approach probe is not describing one sensible bridge. Keep the
            # core stock placement rather than inventing a roller-coaster span.
            continue

        for index in component:
            obj = objects[index]
            centre_dx = float(obj.x) - start[0]
            centre_dz = float(obj.z) - start[1]
            along = centre_dx * unit_x + centre_dz * unit_z
            y = start_y + grade * along

            # Pitch is expressed along the model's local forward axis. Preserve
            # continuity even when a module heading points 180 degrees opposite
            # the chain direction or follows a modest plan-view bend.
            heading = math.radians(float(obj.heading_degrees))
            local_x = math.sin(heading)
            local_z = math.cos(heading)
            local_grade = grade * (local_x * unit_x + local_z * unit_z)
            pitch = math.degrees(math.atan(local_grade))
            objects[index] = replace(obj, y=y, pitch_degrees=pitch)
            changed = True

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
    """Use stock bridge planning, then anchor each emitted bridge to both banks."""

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
    return _anchor_stock_bridge_chains(result, raster, elevations, stock_spec)


def _load_nonroad_objects(*args, **kwargs):
    """Use stock cache identity and repair cached stock bridge deck transforms."""

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
    """Install before later non-road/building wrappers capture generation hooks."""

    global _ORIGINAL_GENERATE_WORLD_OBJECTS, _ORIGINAL_LOAD_NONROAD_OBJECTS, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_GENERATE_WORLD_OBJECTS = _osm.generate_world_objects
    _ORIGINAL_LOAD_NONROAD_OBJECTS = _generator._load_nonroad_objects

    _osm.generate_world_objects = _generate_world_objects
    _generator.generate_world_objects = _generate_world_objects
    _generator._load_nonroad_objects = _load_nonroad_objects
    _INSTALLED = True
