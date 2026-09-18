# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance refinements for late terrain drainage and road-grade diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from . import road_constraint_performance_policy as _road_perf
from . import terrain_solver as _terrain


@dataclass(frozen=True, slots=True)
class _RoadSlopePlan:
    x: np.ndarray
    z: np.ndarray
    previous: np.ndarray
    travel: np.ndarray


@dataclass(frozen=True, slots=True)
class _CoarseDrainagePlan:
    distances: tuple[float, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _FineDrainagePlan:
    distances: tuple[float, ...]
    x: tuple[float, ...]
    z: tuple[float, ...]
    indices: tuple[tuple[int, ...], ...]


_ORIGINAL_ROAD_SLOPE_PERCENT = _terrain._road_slope_percent
_ORIGINAL_ENFORCE_DOWNHILL_WATERCOURSES = _terrain._enforce_downhill_watercourses
_ROAD_PLAN_CACHE: dict[tuple[int, int, int, float, int], _RoadSlopePlan] = {}
_INSTALLED = False


def _raw_line(feature: Any, projection: Any) -> Any | None:
    return _road_perf._ORIGINAL_LINE_GEOMETRY(feature, projection)


def _sample_many(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    grid = np.asarray(elevations, dtype=np.float64).reshape((cells, cells))
    fx = np.clip(x / float(cell_size), 0.0, cells - 1.0)
    fz = np.clip(z / float(cell_size), 0.0, cells - 1.0)
    x0 = np.floor(fx).astype(np.int64)
    z0 = np.floor(fz).astype(np.int64)
    x1 = np.minimum(cells - 1, x0 + 1)
    z1 = np.minimum(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    a = grid[z0, x0] * (1.0 - tx) + grid[z0, x1] * tx
    b = grid[z1, x0] * (1.0 - tx) + grid[z1, x1] * tx
    return a * (1.0 - tz) + b * tz


def _build_road_slope_plan(dataset: Any, projection: Any, spec: Any) -> _RoadSlopePlan:
    spacing = max(2.0, float(spec.cell_size) * 0.35)
    xs: list[float] = []
    zs: list[float] = []
    previous: list[int] = []
    travel: list[float] = []

    for feature in dataset.roads:
        if feature.tags.get("tunnel") not in {None, "", "no"}:
            continue
        line = _raw_line(feature, projection)
        if line is None:
            continue
        count = max(1, int(math.ceil(line.length / spacing)))
        last_position = -1
        previous_point = line.interpolate(0.0)
        for step in range(count + 1):
            point = previous_point if step == 0 else line.interpolate(line.length * step / count)
            current_position = len(xs)
            xs.append(float(point.x))
            zs.append(float(point.y))
            if step == 0:
                previous.append(-1)
                travel.append(0.0)
            else:
                previous.append(last_position)
                travel.append(max(0.01, float(point.distance(previous_point))))
            last_position = current_position
            previous_point = point

    return _RoadSlopePlan(
        x=np.asarray(xs, dtype=np.float64),
        z=np.asarray(zs, dtype=np.float64),
        previous=np.asarray(previous, dtype=np.int64),
        travel=np.asarray(travel, dtype=np.float64),
    )


def _road_slope_plan(dataset: Any, projection: Any, spec: Any) -> _RoadSlopePlan:
    key = (
        id(dataset),
        id(projection),
        int(spec.cells),
        float(spec.cell_size),
        len(dataset.roads),
    )
    plan = _ROAD_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    plan = _build_road_slope_plan(dataset, projection, spec)
    if len(_ROAD_PLAN_CACHE) >= 4:
        _ROAD_PLAN_CACHE.clear()
    _ROAD_PLAN_CACHE[key] = plan
    return plan


def _fast_road_slope_percent(
    elevations: Sequence[float],
    dataset: Any,
    projection: Any,
    spec: Any,
) -> float:
    plan = _road_slope_plan(dataset, projection, spec)
    if plan.x.size == 0:
        return 0.0
    heights = _sample_many(
        elevations,
        int(spec.cells),
        float(spec.cell_size),
        plan.x,
        plan.z,
    )
    valid = plan.previous >= 0
    if not np.any(valid):
        return 0.0
    current = heights[valid]
    earlier = heights[plan.previous[valid]]
    slopes = np.abs(current - earlier) / plan.travel[valid] * 100.0
    return float(np.max(slopes)) if slopes.size else 0.0


def _build_drainage_plans(
    original: Sequence[float],
    dataset: Any,
    projection: Any,
    spec: Any,
) -> tuple[tuple[_CoarseDrainagePlan, ...], tuple[_FineDrainagePlan, ...]]:
    coarse: list[_CoarseDrainagePlan] = []
    fine: list[_FineDrainagePlan] = []
    coarse_spacing = max(3.0, float(spec.cell_size) * 0.45)
    fine_spacing = max(2.0, float(spec.cell_size) / 3.0)

    for feature in dataset.watercourses:
        if feature.tags.get("tunnel") not in {None, "", "no"}:
            continue
        line = _raw_line(feature, projection)
        if line is None:
            continue
        start_x, start_z = map(float, line.coords[0])
        end_x, end_z = map(float, line.coords[-1])
        reverse = (
            _terrain._sample_elevation(
                original, spec.cells, spec.cell_size, start_x, start_z
            )
            < _terrain._sample_elevation(
                original, spec.cells, spec.cell_size, end_x, end_z
            )
        )

        coarse_count = max(1, int(math.ceil(line.length / coarse_spacing)))
        coarse_distances: list[float] = []
        coarse_indices: list[int] = []
        for step in range(coarse_count + 1):
            distance = line.length * (coarse_count - step if reverse else step) / coarse_count
            point = line.interpolate(distance)
            coarse_distances.append(float(distance))
            coarse_indices.append(_terrain._point_cell_index(point, spec))
        coarse.append(
            _CoarseDrainagePlan(tuple(coarse_distances), tuple(coarse_indices))
        )

        fine_count = max(1, int(math.ceil(line.length / fine_spacing)))
        fine_distances: list[float] = []
        fine_x: list[float] = []
        fine_z: list[float] = []
        fine_indices: list[tuple[int, ...]] = []
        for step in range(fine_count + 1):
            distance = line.length * (fine_count - step if reverse else step) / fine_count
            point = line.interpolate(distance)
            x = float(point.x)
            z = float(point.y)
            fine_distances.append(float(distance))
            fine_x.append(x)
            fine_z.append(z)
            fine_indices.append(_terrain._bilinear_indices(x, z, spec))
        fine.append(
            _FineDrainagePlan(
                tuple(fine_distances),
                tuple(fine_x),
                tuple(fine_z),
                tuple(fine_indices),
            )
        )

    return tuple(coarse), tuple(fine)


def _fast_enforce_downhill_watercourses(
    result: list[float],
    original: Sequence[float],
    dataset: Any,
    projection: Any,
    spec: Any,
    field: Any,
) -> None:
    protected = _terrain._dilate_mask(
        [priority > _terrain.PRIORITY_WATERCOURSE for priority in field.priorities],
        spec.cells,
        2,
    )
    minimum_ratio = float(spec.watercourse_minimum_gradient_percent) / 100.0
    coarse_plans, fine_plans = _build_drainage_plans(
        original, dataset, projection, spec
    )

    # Geometry interpolation and bilinear-index discovery are invariant across
    # repair passes, so perform them once above rather than up to 24 times.
    for _ in range(12):
        changed = 0
        for plan in coarse_plans:
            previous_index: int | None = None
            previous_height: float | None = None
            previous_distance: float | None = None
            for distance, index in zip(plan.distances, plan.indices):
                if index == previous_index:
                    continue
                if protected[index]:
                    previous_index = None
                    previous_height = None
                    previous_distance = None
                    continue
                current = result[index]
                if previous_height is not None and previous_distance is not None:
                    travel = abs(distance - previous_distance)
                    desired_maximum = previous_height - travel * minimum_ratio
                    if current > desired_maximum:
                        result[index] = max(
                            original[index] - spec.maximum_grade_adjustment,
                            desired_maximum,
                        )
                        current = result[index]
                        field.categories[index] = "watercourse"
                        changed += 1
                previous_index = index
                previous_height = current
                previous_distance = distance
        if changed == 0:
            break

    for _ in range(12):
        changed = 0
        for plan in fine_plans:
            previous_height: float | None = None
            previous_distance: float | None = None
            for distance, x, z, indices in zip(
                plan.distances, plan.x, plan.z, plan.indices
            ):
                if any(protected[index] for index in indices):
                    previous_height = None
                    previous_distance = None
                    continue
                current = _terrain._sample_elevation(
                    result, spec.cells, spec.cell_size, x, z
                )
                if previous_height is not None and previous_distance is not None:
                    travel = abs(distance - previous_distance)
                    desired_maximum = previous_height - travel * minimum_ratio
                    if current > desired_maximum:
                        excess = current - desired_maximum
                        for index in indices:
                            result[index] = max(
                                original[index] - spec.maximum_grade_adjustment,
                                result[index] - excess,
                            )
                            field.categories[index] = "watercourse"
                        changed += 1
                        current = _terrain._sample_elevation(
                            result, spec.cells, spec.cell_size, x, z
                        )
                previous_height = current
                previous_distance = distance
        if changed == 0:
            break


def install_terrain_postprocess_performance_policy() -> None:
    """Cache invariant sampling work in late terrain post-processing."""
    global _INSTALLED
    if _INSTALLED:
        return
    _terrain._road_slope_percent = _fast_road_slope_percent
    _terrain._enforce_downhill_watercourses = _fast_enforce_downhill_watercourses
    _INSTALLED = True
