# SPDX-License-Identifier: GPL-3.0-or-later
"""Own native stock-road junction selection, placement and endpoint contracts.

The stock piece fitter historically represents every non-gravel T/X junction with
one six-metre straight road model centred on the node. Resistance ships
purpose-built junction models for the common paved/asphalt/cobble families; use
them when the incident geometry is close enough to their fixed connector
template.

This module owns the staged junction refinements that used to be spread across
several policy files: base native selection, measured Memory-LOD placement,
bounded mixed-surface skew matching, and the late endpoint-window contract. Each
installer still runs at its historical pipeline position, so ownership is
simplified without changing mutation timing.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
import re
from typing import Sequence

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_wrp_catalogue as _catalogue


_STOCK_ROAD_FAMILY = re.compile(
    r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)(?:25|12|6)\.p3d$",
    re.IGNORECASE,
)
_STOCK_CAP_MODEL = re.compile(
    r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)6\.p3d$",
    re.IGNORECASE,
)
_GENERATED_GRAVEL_FILENAME = re.compile(
    r"^gravel(?:25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d$",
    re.IGNORECASE,
)

MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = 7.5
NATIVE_JUNCTION_VERTICAL_BIAS_METRES = 0.006
MAXIMUM_RELAXED_JUNCTION_HEADING_ERROR_DEGREES = 18.0
MAXIMUM_RELAXED_APPROACH_METRES = 2.0
MINIMUM_ENDPOINT_RECOVERY_IMPROVEMENT_METRES = 0.05
_WINDOW_EPSILON_METRES = 1.0e-6

_T_JUNCTION_MODELS = dict(_catalogue.WRPTOOL_T_JUNCTION_MODELS)
_X_JUNCTION_MODELS = dict(_catalogue.WRPTOOL_X_JUNCTION_MODELS)
_ALL_NATIVE_JUNCTION_MODELS = tuple(_catalogue.WRPTOOL_NATIVE_JUNCTION_MODELS)

_ORIGINAL_FIT = None
_ORIGINAL_VARIANT_PATHS = None
_INSTALLED = False

_ORIGINAL_MEASURED_QUALITY_JUNCTION_GEOMETRY = None
_MEASURED_CONNECTOR = None
_MEASURED_INSTALLED = False

_ORIGINAL_SKEW_FAMILY = None
_ORIGINAL_SKEW_NATIVE_JUNCTION_FOR_INCIDENTS = None
_SKEW_INSTALLED = False

_ORIGINAL_ENDPOINT_CHAIN = None
_ENDPOINT_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _Incident:
    direction: tuple[float, float]
    family: str | None
    model_path: str


@dataclass(frozen=True, slots=True)
class _NativeJunction:
    model_path: str
    heading_degrees: float
    maximum_heading_error_degrees: float
    cap_family: str


def _family(model_path: str) -> str | None:
    match = _STOCK_ROAD_FAMILY.fullmatch(str(model_path).replace("/", "\\"))
    return match.group("family").casefold() if match is not None else None


def _heading(direction: tuple[float, float]) -> float:
    dx, dz = direction
    length = math.hypot(dx, dz)
    if length <= 1.0e-9:
        return 0.0
    return math.degrees(math.atan2(dx / length, dz / length)) % 360.0


def _angular_distance(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _rotation_score(
    pairs: Sequence[tuple[float, float]],
    rotation: float,
) -> tuple[float, float, float]:
    errors = tuple(
        _angular_distance((local + rotation) % 360.0, actual)
        for local, actual in pairs
    )
    return max(errors), sum(error * error for error in errors), rotation % 360.0


def _best_rotation(pairs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    offsets = tuple((actual - local) % 360.0 for local, actual in pairs)
    candidates = set(offsets)
    for first, second in itertools.combinations(offsets, 2):
        delta = (second - first + 180.0) % 360.0 - 180.0
        candidates.add((first + delta * 0.5) % 360.0)
    score = min(
        (_rotation_score(pairs, value) for value in candidates),
        key=lambda value: value,
    )
    return score[2], score[0]


def _dominant_pair(incidents: Sequence[_Incident]) -> tuple[int, int] | None:
    if len(incidents) < 2:
        return None
    best = None
    for first in range(len(incidents)):
        a = incidents[first].direction
        alen = max(1.0e-9, math.hypot(*a))
        au = (a[0] / alen, a[1] / alen)
        for second in range(first + 1, len(incidents)):
            b = incidents[second].direction
            blen = max(1.0e-9, math.hypot(*b))
            bu = (b[0] / blen, b[1] / blen)
            dot = au[0] * bu[0] + au[1] * bu[1]
            candidate = (dot, first, second)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else (best[1], best[2])


def _native_t_junction(incidents: Sequence[_Incident]) -> _NativeJunction | None:
    if len(incidents) != 3:
        return None
    pair = _dominant_pair(incidents)
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
    model = _T_JUNCTION_MODELS.get((main_family, branch_family))
    if model is None:
        return None

    main_a = _heading(incidents[first].direction)
    main_b = _heading(incidents[second].direction)
    branch_heading = _heading(incidents[branch].direction)
    fits = []
    for actual_zero, actual_180 in ((main_a, main_b), (main_b, main_a)):
        rotation, maximum_error = _best_rotation(
            ((0.0, actual_zero), (180.0, actual_180), (90.0, branch_heading))
        )
        fits.append((maximum_error, rotation))
    maximum_error, rotation = min(fits)
    if maximum_error > MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES:
        return None
    return _NativeJunction(model, rotation, maximum_error, main_family)


def _native_x_junction(incidents: Sequence[_Incident]) -> _NativeJunction | None:
    if len(incidents) != 4:
        return None
    families = {incident.family for incident in incidents}
    if len(families) != 1 or None in families:
        return None
    family = next(iter(families))
    assert family is not None
    model = _X_JUNCTION_MODELS.get(family)
    if model is None:
        return None

    actual = tuple(_heading(incident.direction) for incident in incidents)
    local = (0.0, 90.0, 180.0, 270.0)
    best = None
    for assignment in itertools.permutations(actual):
        rotation, maximum_error = _best_rotation(tuple(zip(local, assignment)))
        candidate = (maximum_error, rotation)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    maximum_error, rotation = best
    if maximum_error > MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES:
        return None
    return _NativeJunction(model, rotation, maximum_error, family)


def _native_junction_for_incidents(
    incidents: Sequence[_Incident],
) -> _NativeJunction | None:
    if len(incidents) == 3:
        return _native_t_junction(incidents)
    if len(incidents) == 4:
        return _native_x_junction(incidents)
    return None


def _junction_incidents(dataset, projection, spec):
    raw = {}
    positions = {}
    for feature, projected in zip(
        dataset.roads, _p.projected_road_polylines(dataset, projection)
    ):
        if not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = tuple(_p._clean_road_points(projected))
        if len(points) < 2:
            continue
        model = _p.road_model_for_tags(spec, feature.tags)
        dirt = _p.road_is_dirt(feature.tags)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= 0.05:
                continue
            segment_key = f"{feature.osm_key}/{index:06d}"
            forward = _p._normalised_direction(start, end)
            reverse = (-forward[0], -forward[1])
            start_key, end_key = _p._road_node_key(start), _p._road_node_key(end)
            raw.setdefault(start_key, []).append(
                (forward, dirt, model, segment_key, feature.osm_key)
            )
            raw.setdefault(end_key, []).append(
                (reverse, dirt, model, segment_key, feature.osm_key)
            )
            positions.setdefault(start_key, start)
            positions.setdefault(end_key, end)

    result = {}
    for key, values in raw.items():
        unique = _p._unique_incidents(values)
        if len(unique) not in {3, 4}:
            continue
        incidents = tuple(
            _Incident(value[0], _family(value[2]), value[2]) for value in unique
        )
        native = _native_junction_for_incidents(incidents)
        if native is not None:
            result[key] = (positions[key], native)
    return result


def _connector_half_extent(spec) -> float:
    return float(spec.road_segment_length) * 6.0 / 25.0 * 0.5


def _native_junction_object(old, native: _NativeJunction, elevations, spec):
    half = _connector_half_extent(spec)
    angle = math.radians(native.heading_degrees)
    direction = (math.sin(angle), math.cos(angle))
    start = (old.x - direction[0] * half, old.z - direction[1] * half)
    end = (old.x + direction[0] * half, old.z + direction[1] * half)
    return _p._road_object_on_slope(
        old.object_id,
        native.model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            + NATIVE_JUNCTION_VERTICAL_BIAS_METRES
        ),
    )


def _lower_legacy_stock_cap(old, elevations, spec):
    if _STOCK_CAP_MODEL.fullmatch(old.model_path.replace("/", "\\")) is None:
        return old
    half = _connector_half_extent(spec)
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


def _replace_stock_junction_caps(report, dataset, projection, elevations, spec):
    count = int(getattr(report, "junction_cap_objects", 0))
    if count <= 0 or not report.objects:
        return report
    junctions = _junction_incidents(dataset, projection, spec)

    objects = list(report.objects)
    changed = False
    for index in range(min(count, len(objects))):
        old = objects[index]
        if _STOCK_CAP_MODEL.fullmatch(old.model_path.replace("/", "\\")) is None:
            continue
        key = _p._road_node_key((old.x, old.z))
        match = junctions.get(key)
        if match is not None:
            point, native = match
            asset = _catalogue.native_junction_asset(str(native.model_path))
            if asset is not None and math.dist((old.x, old.z), point) <= 0.20:
                objects[index] = _native_junction_object(old, native, elevations, spec)
                changed = True
                continue

        lowered = _lower_legacy_stock_cap(old, elevations, spec)
        if lowered != old:
            objects[index] = lowered
            changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def _native_asset_paths(model_path: str) -> tuple[str, ...]:
    return _ALL_NATIVE_JUNCTION_MODELS if _family(model_path) is not None else ()


def _road_model_variant_paths(
    model_path: str, configured_long_length: float
) -> tuple[str, ...]:
    if _ORIGINAL_VARIANT_PATHS is None:
        raise RuntimeError("stock road junction policy is not installed")
    paths = list(_ORIGINAL_VARIANT_PATHS(model_path, configured_long_length))
    paths.extend(_native_asset_paths(model_path))
    return tuple(dict.fromkeys(paths))


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock road junction policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return report
    return _replace_stock_junction_caps(report, dataset, projection, elevations, spec)


# Measured Memory-LOD stage -------------------------------------------------

def _measured_native_t_junction(incidents):
    if len(incidents) != 3:
        return None
    pair = _dominant_pair(incidents)
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
    model = _T_JUNCTION_MODELS.get((main_family, branch_family))
    if model is None:
        return None

    main_a = _heading(incidents[first].direction)
    main_b = _heading(incidents[second].direction)
    branch_heading = _heading(incidents[branch].direction)
    fits = []
    for actual_zero, actual_180 in ((main_a, main_b), (main_b, main_a)):
        rotation, maximum_error = _best_rotation(
            ((0.0, actual_zero), (180.0, actual_180), (270.0, branch_heading))
        )
        fits.append((maximum_error, rotation))
    maximum_error, rotation = min(fits)
    if maximum_error > MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES:
        return None
    return _NativeJunction(model, rotation, maximum_error, main_family)


def _measured_native_junction_for_incidents(incidents):
    if len(incidents) == 3:
        return _native_t_junction(incidents)
    if len(incidents) == 4:
        return _native_x_junction(incidents)
    return None


def _measured_connector_half_extent(_spec) -> float:
    return _model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES


def _measured_native_junction_object(old, native, elevations, spec):
    half = _model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
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
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES + NATIVE_JUNCTION_VERTICAL_BIAS_METRES
        ),
    )

    intersection_local = _model_geometry.native_junction_intersection_offset(native.model_path)
    if intersection_local is None:
        return placed
    intersection_world_offset = _model_geometry.rotate_local(
        intersection_local, native.heading_degrees
    )
    return replace(
        placed,
        x=node[0] - intersection_world_offset[0],
        z=node[1] - intersection_world_offset[1],
        heading_degrees=native.heading_degrees,
    )


def _measured_lower_legacy_stock_cap(old, elevations, spec):
    if _STOCK_CAP_MODEL.fullmatch(old.model_path.replace("/", "\\")) is None:
        return old
    half = _model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6] * 0.5
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


def _measured_quality_junction_geometry(dataset, projection, spec):
    if _ORIGINAL_MEASURED_QUALITY_JUNCTION_GEOMETRY is None:
        raise RuntimeError("measured stock junction stage is not installed")
    result = dict(_ORIGINAL_MEASURED_QUALITY_JUNCTION_GEOMETRY(dataset, projection, spec))
    native = _junction_incidents(dataset, projection, spec)
    extent = _model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
    for key in native:
        current = result.get(key)
        if current is None:
            continue
        result[key] = replace(current, half_length=extent, half_width=extent)
    return result


def _measured_native_t_targets(incidents, native):
    connector = _MEASURED_CONNECTOR
    if connector is None:
        raise RuntimeError("measured stock junction stage is not installed")
    if len(incidents) != 3:
        return None
    pair = _dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    rotation = float(native.heading_degrees) % 360.0
    target_zero = rotation
    target_180 = (rotation + 180.0) % 360.0
    target_branch = (rotation + 270.0) % 360.0

    actual_first = connector._heading(incidents[first].direction)
    actual_second = connector._heading(incidents[second].direction)
    direct = (
        connector._angular_distance(actual_first, target_zero)
        + connector._angular_distance(actual_second, target_180)
    )
    swapped = (
        connector._angular_distance(actual_first, target_180)
        + connector._angular_distance(actual_second, target_zero)
    )
    targets = [0.0, 0.0, 0.0]
    if direct <= swapped:
        targets[first], targets[second] = target_zero, target_180
    else:
        targets[first], targets[second] = target_180, target_zero
    targets[branch] = target_branch
    return tuple(targets)


# Mixed skew stage ---------------------------------------------------------

def _is_generated_gravel_model(model_path: str) -> bool:
    filename = str(model_path).replace("/", "\\").rsplit("\\", 1)[-1]
    return _GENERATED_GRAVEL_FILENAME.fullmatch(filename) is not None


def _family_with_generated_gravel(model_path: str) -> str | None:
    if _ORIGINAL_SKEW_FAMILY is None:
        raise RuntimeError("stock road skew stage is not installed")
    family = _ORIGINAL_SKEW_FAMILY(model_path)
    if family is not None:
        return family
    if _is_generated_gravel_model(model_path):
        return "ces"
    return None


def _eligible_relaxed_mixed_t(incidents) -> bool:
    if len(incidents) != 3:
        return False
    gravel = [incident for incident in incidents if _is_generated_gravel_model(incident.model_path)]
    if len(gravel) != 1:
        return False
    stock_families = [
        incident.family
        for incident in incidents
        if not _is_generated_gravel_model(incident.model_path)
    ]
    return (
        len(stock_families) == 2
        and stock_families[0] == stock_families[1]
        and stock_families[0] in {"sil", "asf", "kos"}
    )


def _native_junction_with_bounded_mixed_skew(incidents):
    global MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES
    if _ORIGINAL_SKEW_NATIVE_JUNCTION_FOR_INCIDENTS is None:
        raise RuntimeError("stock road skew stage is not installed")

    native = _ORIGINAL_SKEW_NATIVE_JUNCTION_FOR_INCIDENTS(incidents)
    if native is not None or not _eligible_relaxed_mixed_t(incidents):
        return native

    original_limit = MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES
    try:
        MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = MAXIMUM_RELAXED_JUNCTION_HEADING_ERROR_DEGREES
        return _ORIGINAL_SKEW_NATIVE_JUNCTION_FOR_INCIDENTS(incidents)
    finally:
        MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = original_limit


# Late endpoint stage ------------------------------------------------------

def _effective_window(
    measure,
    pieces,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
):
    context = _quality._CONTEXT.get()
    if context is None:
        return (
            float(start_distance),
            float(preferred_end_distance),
            float(minimum_end_distance),
            float(maximum_end_distance),
        )
    return tuple(
        float(value)
        for value in _quality._quality_window(
            measure,
            pieces,
            start_distance,
            preferred_end_distance,
            minimum_end_distance,
            maximum_end_distance,
            context,
        )
    )


def _covered_endpoint_errors(
    measure,
    fitted,
    *,
    recover_start: bool,
    recover_end: bool,
    effective_start: float,
    effective_end: float,
) -> tuple[float, ...]:
    if not fitted:
        return (math.inf,) if recover_start or recover_end else ()
    errors = []
    if recover_start:
        errors.append(math.dist(tuple(fitted[0][1]), measure.point(effective_start)[:2]))
    if recover_end:
        errors.append(math.dist(tuple(fitted[-1][2]), measure.point(effective_end)[:2]))
    return tuple(errors)


def _junction_endpoint_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_ENDPOINT_CHAIN is None:
        raise RuntimeError("stock road junction-endpoint stage is not installed")

    raw_start = float(start_distance)
    raw_preferred = float(preferred_end_distance)
    baseline = _ORIGINAL_ENDPOINT_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    effective_start, effective_preferred, effective_minimum, effective_maximum = _effective_window(
        measure,
        pieces,
        start_distance,
        preferred_end_distance,
        minimum_end_distance,
        maximum_end_distance,
    )

    trim_start = effective_start > raw_start + _WINDOW_EPSILON_METRES
    trim_end = effective_preferred < raw_preferred - _WINDOW_EPSILON_METRES
    if trim_start or trim_end:
        return _ORIGINAL_ENDPOINT_CHAIN(
            measure,
            pieces,
            start_distance=effective_start,
            preferred_end_distance=effective_preferred,
            minimum_end_distance=effective_minimum,
            maximum_end_distance=effective_maximum,
        )

    recover_start = effective_start < raw_start - _WINDOW_EPSILON_METRES
    recover_end = effective_preferred > raw_preferred + _WINDOW_EPSILON_METRES
    if not recover_start and not recover_end:
        return baseline

    baseline_errors = _covered_endpoint_errors(
        measure,
        baseline,
        recover_start=recover_start,
        recover_end=recover_end,
        effective_start=effective_start,
        effective_end=effective_preferred,
    )
    if baseline_errors and max(baseline_errors) <= MINIMUM_ENDPOINT_RECOVERY_IMPROVEMENT_METRES:
        return baseline

    recovered = _ORIGINAL_ENDPOINT_CHAIN(
        measure,
        pieces,
        start_distance=effective_start,
        preferred_end_distance=effective_preferred,
        minimum_end_distance=effective_minimum,
        maximum_end_distance=effective_maximum,
    )
    if not recovered:
        return baseline
    recovered_errors = _covered_endpoint_errors(
        measure,
        recovered,
        recover_start=recover_start,
        recover_end=recover_end,
        effective_start=effective_start,
        effective_end=effective_preferred,
    )
    if not recovered_errors:
        return baseline

    baseline_max = max(baseline_errors) if baseline_errors else math.inf
    recovered_max = max(recovered_errors)
    if recovered_max + MINIMUM_ENDPOINT_RECOVERY_IMPROVEMENT_METRES < baseline_max:
        return recovered
    return baseline


# Staged installers --------------------------------------------------------

def install_stock_road_junction_policy() -> None:
    global _ORIGINAL_FIT, _ORIGINAL_VARIANT_PATHS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _ORIGINAL_VARIANT_PATHS = _p.road_model_variant_paths
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _p.road_model_variant_paths = _road_model_variant_paths
    _generator.road_model_variant_paths = _road_model_variant_paths
    _INSTALLED = True


def install_stock_road_measured_junction_policy() -> None:
    """Install measured Memory-LOD junction geometry at its historical stage."""

    global _ORIGINAL_MEASURED_QUALITY_JUNCTION_GEOMETRY, _MEASURED_CONNECTOR
    global _MEASURED_INSTALLED
    global _native_t_junction, _native_junction_for_incidents
    global _connector_half_extent, _native_junction_object, _lower_legacy_stock_cap
    if _MEASURED_INSTALLED:
        return

    # The old measured-junction module imported the connector module at this
    # stage. Delay the import until now so folding files does not move that side
    # effect earlier in startup.
    from . import stock_road_connector_policy as connector

    _MEASURED_CONNECTOR = connector
    _ORIGINAL_MEASURED_QUALITY_JUNCTION_GEOMETRY = _quality._junction_geometry
    _native_t_junction = _measured_native_t_junction
    _native_junction_for_incidents = _measured_native_junction_for_incidents
    _connector_half_extent = _measured_connector_half_extent
    _native_junction_object = _measured_native_junction_object
    _lower_legacy_stock_cap = _measured_lower_legacy_stock_cap
    _quality._junction_geometry = _measured_quality_junction_geometry
    connector._native_t_targets = _measured_native_t_targets
    connector.MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES = MAXIMUM_RELAXED_APPROACH_METRES
    _MEASURED_INSTALLED = True


def install_stock_road_skew_policy() -> None:
    global _ORIGINAL_SKEW_FAMILY, _ORIGINAL_SKEW_NATIVE_JUNCTION_FOR_INCIDENTS
    global _SKEW_INSTALLED, _family, _native_junction_for_incidents
    if _SKEW_INSTALLED:
        return
    _ORIGINAL_SKEW_FAMILY = _family
    _ORIGINAL_SKEW_NATIVE_JUNCTION_FOR_INCIDENTS = _native_junction_for_incidents
    _family = _family_with_generated_gravel
    _native_junction_for_incidents = _native_junction_with_bounded_mixed_skew
    _SKEW_INSTALLED = True


def install_stock_road_junction_endpoint_policy() -> None:
    global _ORIGINAL_ENDPOINT_CHAIN, _ENDPOINT_INSTALLED
    if _ENDPOINT_INSTALLED:
        return
    _ORIGINAL_ENDPOINT_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _junction_endpoint_chain
    _ENDPOINT_INSTALLED = True
