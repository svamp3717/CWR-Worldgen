# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply measured Memory-LOD geometry to native stock-road junctions.

The native T meshes are not centered on their logical intersection and their
branch connector is on local -X. Their three connector centers are 6.25 metres
from the logical intersection. This policy replaces the earlier filename-based
placement assumptions and reserves the real connector footprint while fitting
approach roads.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import road_quality_policy as _quality
from . import stock_road_connector_policy as _connector
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _geometry
from . import playability as _p

MAXIMUM_RELAXED_APPROACH_METRES = 2.0
_ORIGINAL_QUALITY_JUNCTION_GEOMETRY = None
_INSTALLED = False


def _native_t_junction(incidents):
    """Fit a native T using its measured local 0/180/-90 connector headings."""

    if len(incidents) != 3:
        return None
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    main_family = incidents[first].family
    if main_family is None or incidents[second].family != main_family:
        return None
    branch_family = incidents[branch].family
    if branch_family is None:
        return None
    model = _junction._T_JUNCTION_MODELS.get((main_family, branch_family))
    if model is None:
        return None

    main_a = _junction._heading(incidents[first].direction)
    main_b = _junction._heading(incidents[second].direction)
    branch_heading = _junction._heading(incidents[branch].direction)
    fits = []
    for actual_zero, actual_180 in ((main_a, main_b), (main_b, main_a)):
        pairs = (
            (0.0, actual_zero),
            (180.0, actual_180),
            (270.0, branch_heading),
        )
        rotation, maximum_error = _junction._best_rotation(pairs)
        fits.append((maximum_error, rotation))
    maximum_error, rotation = min(fits)
    if maximum_error > _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES:
        return None
    return _junction._NativeJunction(model, rotation, maximum_error, main_family)


def _native_junction_for_incidents(incidents):
    # Resolve the T selector dynamically. Late policies deliberately tighten the
    # allowed connector error after this measured-geometry layer is installed;
    # calling the module-local implementation here would silently bypass them.
    if len(incidents) == 3:
        return _junction._native_t_junction(incidents)
    if len(incidents) == 4:
        return _junction._native_x_junction(incidents)
    return None


def _connector_half_extent(_spec) -> float:
    """Distance from a native junction's logical center to each connector."""

    return _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES


def _native_junction_object(old, native, elevations, spec):
    """Place a junction so its measured logical center remains on the road node."""

    half = _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
    angle = math.radians(native.heading_degrees)
    direction = (math.sin(angle), math.cos(angle))
    node = (float(old.x), float(old.z))
    start = (node[0] - direction[0] * half, node[1] - direction[1] * half)
    end = (node[0] + direction[0] * half, node[1] + direction[1] * half)
    placed = _p._road_object_on_slope(
        old.object_id,
        native.model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            + _junction.NATIVE_JUNCTION_VERTICAL_BIAS_METRES
        ),
    )

    intersection_local = _geometry.native_junction_intersection_offset(native.model_path)
    if intersection_local is None:
        return placed
    intersection_world_offset = _geometry.rotate_local(
        intersection_local, native.heading_degrees
    )
    return replace(
        placed,
        x=node[0] - intersection_world_offset[0],
        z=node[1] - intersection_world_offset[1],
        heading_degrees=native.heading_degrees,
    )


def _lower_legacy_stock_cap(old, elevations, spec):
    """Ground a legacy straight cap using the real 6.25 m stock-piece length."""

    if _junction._STOCK_CAP_MODEL.fullmatch(old.model_path.replace("/", "\\")) is None:
        return old
    half = _geometry.STOCK_STRAIGHT_LENGTHS_METRES[6] * 0.5
    angle = math.radians(old.heading_degrees)
    direction = (math.sin(angle), math.cos(angle))
    start = (old.x - direction[0] * half, old.z - direction[1] * half)
    end = (old.x + direction[0] * half, old.z + direction[1] * half)
    return _p._road_object_on_slope(
        old.object_id,
        old.model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )


def _quality_junction_geometry(dataset, projection, spec):
    """Reserve native-junction connectors before branch road pieces are fitted."""

    if _ORIGINAL_QUALITY_JUNCTION_GEOMETRY is None:
        raise RuntimeError("measured stock junction policy is not installed")
    result = dict(_ORIGINAL_QUALITY_JUNCTION_GEOMETRY(dataset, projection, spec))
    native = _junction._junction_incidents(dataset, projection, spec)
    extent = _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
    for key in native:
        current = result.get(key)
        if current is None:
            continue
        result[key] = replace(
            current,
            half_length=extent,
            half_width=extent,
        )
    return result


def _native_t_targets(incidents, native):
    """Map incident arms to the measured T connector headings in world space."""

    if len(incidents) != 3:
        return None
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    rotation = float(native.heading_degrees) % 360.0
    target_zero = rotation
    target_180 = (rotation + 180.0) % 360.0
    target_branch = (rotation + 270.0) % 360.0

    actual_first = _connector._heading(incidents[first].direction)
    actual_second = _connector._heading(incidents[second].direction)
    direct = (
        _connector._angular_distance(actual_first, target_zero)
        + _connector._angular_distance(actual_second, target_180)
    )
    swapped = (
        _connector._angular_distance(actual_first, target_180)
        + _connector._angular_distance(actual_second, target_zero)
    )
    targets = [0.0, 0.0, 0.0]
    if direct <= swapped:
        targets[first], targets[second] = target_zero, target_180
    else:
        targets[first], targets[second] = target_180, target_zero
    targets[branch] = target_branch
    return tuple(targets)


def install_stock_road_measured_junction_policy() -> None:
    global _ORIGINAL_QUALITY_JUNCTION_GEOMETRY, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_QUALITY_JUNCTION_GEOMETRY = _quality._junction_geometry
    _junction._native_t_junction = _native_t_junction
    _junction._native_junction_for_incidents = _native_junction_for_incidents
    _junction._connector_half_extent = _connector_half_extent
    _junction._native_junction_object = _native_junction_object
    _junction._lower_legacy_stock_cap = _lower_legacy_stock_cap
    _quality._junction_geometry = _quality_junction_geometry

    # The mixed-surface relaxation layer must target the same measured branch
    # side and has enough room to absorb the maximum accepted 18-degree skew at
    # a 6.25 m connector radius while remaining inside the narrow gravel road.
    _connector._native_t_targets = _native_t_targets
    _connector.MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES = (
        MAXIMUM_RELAXED_APPROACH_METRES
    )
    _INSTALLED = True
