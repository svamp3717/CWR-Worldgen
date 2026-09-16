# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance policy for the road-constraint phase of the terrain solver.

Dense road networks used to spend most of this stage crossing the Python/GEOS
boundary one terrain cell at a time. Keep the solver's existing grading and
priority semantics, but batch geometric queries with Shapely 2, discard bounding
box cells that cannot touch the road before entering the Python loop, avoid
constructing Shapely Point objects for grid-cell centres, interpolate road
profiles in NumPy batches, and prevent long winding roads from materializing
world-scale bounding-box candidate arrays.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely import (
    covers as vectorized_covers,
    distance as vectorized_distance,
    line_locate_point as vectorized_line_locate_point,
    points as vectorized_points,
)

from . import terrain_solver as _terrain

_SEGMENT_BROAD_PHASE_THRESHOLD = 16_384
_SEGMENT_MINIMUM_SPAN_METRES = 256.0
_SEGMENT_SPAN_CELLS = 32.0
_SEGMENT_SPAN_RADII = 8.0
_PATHOLOGICAL_BBOX_CANDIDATES = 250_000
_PATHOLOGICAL_POINT_COUNT = 4_096


class _CellPoint:
    """Lightweight terrain-cell centre used inside an active vectorized batch."""

    __slots__ = ("x", "y", "batch", "position")

    def __init__(self, x: float, y: float, batch: "_CellBatch", position: int) -> None:
        self.x = float(x)
        self.y = float(y)
        self.batch = batch
        self.position = int(position)


class _ProjectedDistance(float):
    """Projected road distance carrying the batch slot used to obtain it."""

    def __new__(
        cls,
        value: float,
        batch: "_CellBatch",
        position: int,
    ) -> "_ProjectedDistance":
        result = float.__new__(cls, value)
        result.batch = batch
        result.position = int(position)
        return result


@dataclass(slots=True)
class _CellBatch:
    indices: np.ndarray
    cells: int
    cell_size: float
    points: Any
    positions: dict[int, int]
    distances: np.ndarray | None = None
    projections: np.ndarray | None = None
    covered: np.ndarray | None = None
    profile_values: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        indices: Sequence[int] | np.ndarray,
        cells: int,
        cell_size: float,
    ) -> "_CellBatch":
        values = np.asarray(indices, dtype=np.int64)
        xs = (values % cells).astype(np.float64) * float(cell_size)
        zs = (values // cells).astype(np.float64) * float(cell_size)
        return cls(
            indices=values,
            cells=int(cells),
            cell_size=float(cell_size),
            points=vectorized_points(xs, zs),
            positions={int(index): position for position, index in enumerate(values)},
        )

    def subset(self, keep: np.ndarray) -> "_CellBatch":
        values = self.indices[keep]
        points = self.points[keep]
        distances = self.distances[keep] if self.distances is not None else None
        return _CellBatch(
            indices=values,
            cells=self.cells,
            cell_size=self.cell_size,
            points=points,
            positions={int(index): position for position, index in enumerate(values)},
            distances=distances,
        )

    def position_for_coordinates(self, x: float, z: float) -> int | None:
        if self.cell_size <= 0.0:
            return None
        grid_x = int(round(float(x) / self.cell_size))
        grid_z = int(round(float(z) / self.cell_size))
        if not (0 <= grid_x < self.cells and 0 <= grid_z < self.cells):
            return None
        expected_x = grid_x * self.cell_size
        expected_z = grid_z * self.cell_size
        tolerance = max(1.0e-7, self.cell_size * 1.0e-9)
        if abs(float(x) - expected_x) > tolerance or abs(float(z) - expected_z) > tolerance:
            return None
        return self.positions.get(grid_z * self.cells + grid_x)

    def position_for_point(self, point: Any) -> int | None:
        if isinstance(point, _CellPoint) and point.batch is self:
            return point.position
        try:
            return self.position_for_coordinates(float(point.x), float(point.y))
        except (AttributeError, TypeError, ValueError):
            return None

    def ensure_distances(self, geometry: Any) -> np.ndarray:
        if self.distances is None:
            self.distances = np.asarray(
                vectorized_distance(geometry, self.points),
                dtype=np.float64,
            )
        return self.distances

    def ensure_projections(self, geometry: Any) -> np.ndarray:
        if self.projections is None:
            self.projections = np.asarray(
                vectorized_line_locate_point(geometry, self.points),
                dtype=np.float64,
            )
        return self.projections

    def interpolated_profile(
        self,
        distances: Sequence[float],
        heights: Sequence[float],
    ) -> np.ndarray:
        key = (id(distances), id(heights))
        values = self.profile_values.get(key)
        if values is None:
            values = np.interp(
                self.ensure_projection_values(),
                np.asarray(distances, dtype=np.float64),
                np.asarray(heights, dtype=np.float64),
            )
            self.profile_values[key] = values
        return values

    def ensure_projection_values(self) -> np.ndarray:
        if self.projections is None:
            raise RuntimeError("road projection batch was requested before projections were computed")
        return self.projections


class _FastLine:
    __slots__ = ("_geometry", "_batch", "_tags")

    def __init__(self, geometry: Any, tags: Mapping[str, str]) -> None:
        self._geometry = geometry
        self._batch: _CellBatch | None = None
        self._tags = tags

    def __getattr__(self, name: str) -> Any:
        return getattr(self._geometry, name)

    def buffer(self, *args: Any, **kwargs: Any) -> "_FastCorridor":
        radius = args[0] if args else kwargs.get("distance")
        corridor = _FastCorridor(
            self,
            args,
            kwargs,
            None if radius is None else float(radius),
        )
        _PENDING_CORRIDOR.set(corridor)
        return corridor

    def distance(self, other: Any) -> float:
        batch = self._batch
        if batch is not None:
            position = batch.position_for_point(other)
            if position is not None:
                return float(batch.ensure_distances(self._geometry)[position])
        return float(self._geometry.distance(other))

    def project(self, other: Any, normalized: bool = False) -> float:
        batch = self._batch
        if batch is not None:
            position = batch.position_for_point(other)
            if position is not None:
                value = float(batch.ensure_projections(self._geometry)[position])
                if normalized:
                    length = float(self._geometry.length)
                    return 0.0 if length <= 0.0 else value / length
                return _ProjectedDistance(value, batch, position)
        return float(self._geometry.project(other, normalized=normalized))


class _FastCorridor:
    """Lazy road buffer used by terrain line-distance loops."""

    __slots__ = (
        "line", "_geometry", "_batch", "radius", "_buffer_args",
        "_buffer_kwargs", "_bounds",
    )

    def __init__(
        self,
        line: _FastLine,
        buffer_args: Sequence[Any],
        buffer_kwargs: Mapping[str, Any],
        radius: float | None,
    ) -> None:
        self.line = line
        self._geometry: Any | None = None
        self._batch: _CellBatch | None = None
        self.radius = radius
        self._buffer_args = tuple(buffer_args)
        self._buffer_kwargs = dict(buffer_kwargs)
        min_x, min_z, max_x, max_z = (
            float(value) for value in self.line._geometry.bounds
        )
        padding = max(0.0, float(radius)) if radius is not None else 0.0
        self._bounds = (
            min_x - padding,
            min_z - padding,
            max_x + padding,
            max_z + padding,
        )

    def _materialize_geometry(self) -> Any:
        geometry = self._geometry
        if geometry is None:
            geometry = self.line._geometry.buffer(
                *self._buffer_args,
                **self._buffer_kwargs,
            )
            self._geometry = geometry
        return geometry

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if self.radius is None:
            return tuple(float(value) for value in self._materialize_geometry().bounds)
        return self._bounds

    def __getattr__(self, name: str) -> Any:
        return getattr(self._materialize_geometry(), name)

    def attach_batch(self, batch: _CellBatch) -> None:
        self._batch = batch
        self.line._batch = batch

    def covers(self, other: Any) -> bool:
        batch = self._batch
        if batch is not None:
            position = batch.position_for_point(other)
            if position is not None:
                if batch.covered is None:
                    batch.covered = np.asarray(
                        vectorized_covers(self._materialize_geometry(), batch.points),
                        dtype=np.bool_,
                    )
                return bool(batch.covered[position])
        return bool(self._materialize_geometry().covers(other))


_PENDING_CORRIDOR: ContextVar[_FastCorridor | None] = ContextVar(
    "cwr_road_constraint_pending_corridor",
    default=None,
)
_ACTIVE_CELL_BATCH: ContextVar[_CellBatch | None] = ContextVar(
    "cwr_road_constraint_active_cell_batch",
    default=None,
)
_ACTIVE_NEEDS_WATER_TEST: ContextVar[bool | None] = ContextVar(
    "cwr_road_constraint_needs_water_test",
    default=None,
)
_MASK_ARRAY_CACHE: ContextVar[tuple[int, np.ndarray] | None] = ContextVar(
    "cwr_road_constraint_mask_array_cache",
    default=None,
)

_ORIGINAL_LINE_GEOMETRY = _terrain._line_geometry
_ORIGINAL_CANDIDATE_CELLS = _terrain._candidate_cells
_ORIGINAL_POINT = _terrain.Point
_ORIGINAL_PROFILE_HEIGHT = _terrain._profile_height
_ORIGINAL_ROAD_SPAN_WATER_TEST = _terrain.road_span_has_in_game_water
_INSTALLED = False


def _bounds_match(left: tuple[float, float, float, float], right: Any) -> bool:
    try:
        values = tuple(float(value) for value in right)
    except (TypeError, ValueError):
        return False
    if len(values) != 4:
        return False
    return all(
        math.isclose(float(a), b, rel_tol=0.0, abs_tol=1.0e-7)
        for a, b in zip(left, values)
    )


def _candidate_window(
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> tuple[int, int, int, int] | None:
    if cells <= 0 or cell_size <= 0.0:
        return None
    min_x, min_z, max_x, max_z = (float(value) for value in bounds)
    x0 = max(0, min(cells - 1, int(math.ceil(min_x / cell_size - 0.5))))
    z0 = max(0, min(cells - 1, int(math.ceil(min_z / cell_size - 0.5))))
    x1 = max(0, min(cells - 1, int(math.floor(max_x / cell_size + 0.5))))
    z1 = max(0, min(cells - 1, int(math.floor(max_z / cell_size + 0.5))))
    if x1 < x0 or z1 < z0:
        return None
    return x0, z0, x1, z1


def _candidate_count(
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> int:
    window = _candidate_window(bounds, cells, cell_size)
    if window is None:
        return 0
    x0, z0, x1, z1 = window
    return (x1 - x0 + 1) * (z1 - z0 + 1)


def _candidate_indices_for_bounds(
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> np.ndarray:
    """Vector form of terrain_solver._candidate_cells for one rectangle."""

    window = _candidate_window(bounds, cells, cell_size)
    if window is None:
        return np.empty(0, dtype=np.int64)
    x0, z0, x1, z1 = window
    xs = np.arange(x0, x1 + 1, dtype=np.int64)
    zs = np.arange(z0, z1 + 1, dtype=np.int64)
    return (zs[:, None] * int(cells) + xs[None, :]).reshape(-1)


def _segment_candidate_indices(
    geometry: Any,
    radius: float,
    cells: int,
    cell_size: float,
) -> np.ndarray:
    """Return conservative cells around short pieces of the source polyline."""

    coords = np.asarray(geometry.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 2:
        min_x, min_z, max_x, max_z = (
            float(value) for value in geometry.bounds
        )
        return _candidate_indices_for_bounds(
            (min_x - radius, min_z - radius, max_x + radius, max_z + radius),
            cells,
            cell_size,
        )

    max_span = max(
        _SEGMENT_MINIMUM_SPAN_METRES,
        float(cell_size) * _SEGMENT_SPAN_CELLS,
        max(0.0, float(radius)) * _SEGMENT_SPAN_RADII,
    )
    arrays: list[np.ndarray] = []
    for start, end in zip(coords[:-1, :2], coords[1:, :2]):
        delta = end - start
        length = float(np.hypot(delta[0], delta[1]))
        if length <= 1.0e-12:
            continue
        piece_count = max(1, int(math.ceil(length / max_span)))
        for piece_index in range(piece_count):
            t0 = piece_index / piece_count
            t1 = (piece_index + 1) / piece_count
            a = start + delta * t0
            b = start + delta * t1
            piece_bounds = (
                min(float(a[0]), float(b[0])) - radius,
                min(float(a[1]), float(b[1])) - radius,
                max(float(a[0]), float(b[0])) + radius,
                max(float(a[1]), float(b[1])) + radius,
            )
            values = _candidate_indices_for_bounds(piece_bounds, cells, cell_size)
            if values.size:
                arrays.append(values)

    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return arrays[0]
    return np.unique(np.concatenate(arrays))


def _corridor_candidate_indices(
    geometry: Any,
    radius: float,
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> tuple[np.ndarray, int]:
    full_count = _candidate_count(bounds, cells, cell_size)
    if full_count <= _SEGMENT_BROAD_PHASE_THRESHOLD:
        return _candidate_indices_for_bounds(bounds, cells, cell_size), full_count
    return _segment_candidate_indices(geometry, radius, cells, cell_size), full_count


def _needs_bridge_water_test(tags: Mapping[str, str]) -> bool:
    bridge_value = str(tags.get("bridge", "")).casefold()
    explicit_bridge = (
        bridge_value not in {"", "no", "false", "0"}
        or str(tags.get("man_made", "")).casefold() == "bridge"
    )
    if explicit_bridge:
        return True
    try:
        return float(str(tags.get("layer", "0")).replace(",", ".")) > 0.0
    except ValueError:
        return False


def _fast_line_geometry(feature: Any, projection: Any) -> Any:
    _ACTIVE_CELL_BATCH.set(None)
    _PENDING_CORRIDOR.set(None)
    geometry = _ORIGINAL_LINE_GEOMETRY(feature, projection)
    if geometry is None:
        _ACTIVE_NEEDS_WATER_TEST.set(None)
        return None
    tags = getattr(feature, "tags", {})
    _ACTIVE_NEEDS_WATER_TEST.set(_needs_bridge_water_test(tags))
    return _FastLine(geometry, tags)


def _fast_candidate_cells(
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> Iterable[int]:
    pending = _PENDING_CORRIDOR.get()
    _PENDING_CORRIDOR.set(None)

    # Do not build a bbox-sized NumPy array before checking for a road corridor.
    if pending is None or not _bounds_match(bounds, pending.bounds):
        _ACTIVE_CELL_BATCH.set(None)
        return _ORIGINAL_CANDIDATE_CELLS(bounds, cells, cell_size)

    if pending.radius is None:
        indices = _candidate_indices_for_bounds(bounds, cells, cell_size)
        if indices.size == 0:
            _ACTIVE_CELL_BATCH.set(None)
            return iter(())
        batch = _CellBatch.create(indices, cells, cell_size)
        pending.attach_batch(batch)
        _ACTIVE_CELL_BATCH.set(batch)
        return (int(index) for index in batch.indices)

    geometry = pending.line._geometry
    full_count = _candidate_count(bounds, cells, cell_size)
    point_count = len(geometry.coords)
    diagnostic = (
        full_count >= _PATHOLOGICAL_BBOX_CANDIDATES
        or point_count >= _PATHOLOGICAL_POINT_COUNT
    )
    if diagnostic:
        print(
            "[road-constraint] large corridor broad phase: "
            f"points={point_count:,}, length={float(geometry.length):,.1f}m, "
            f"bbox_candidates={full_count:,}, radius={pending.radius:.1f}m; "
            "using segment-wise candidates",
            flush=True,
        )

    indices, _ = _corridor_candidate_indices(
        geometry,
        pending.radius,
        bounds,
        cells,
        cell_size,
    )
    if indices.size == 0:
        _ACTIVE_CELL_BATCH.set(None)
        return iter(())

    batch = _CellBatch.create(indices, cells, cell_size)
    distances = batch.ensure_distances(geometry)
    keep = distances <= pending.radius + 1.0e-9
    if not np.all(keep):
        batch = batch.subset(keep)

    if diagnostic:
        print(
            "[road-constraint] large corridor broad phase complete: "
            f"segment_candidates={indices.size:,}, exact_candidates={batch.indices.size:,}",
            flush=True,
        )

    pending.attach_batch(batch)
    _ACTIVE_CELL_BATCH.set(batch)
    return (int(index) for index in batch.indices)


def _fast_point(*args: Any, **kwargs: Any) -> Any:
    batch = _ACTIVE_CELL_BATCH.get()
    if batch is not None and not kwargs:
        try:
            if len(args) == 1:
                value = args[0]
                x = float(value[0])
                z = float(value[1])
            elif len(args) == 2:
                x = float(args[0])
                z = float(args[1])
            else:
                raise ValueError
            position = batch.position_for_coordinates(x, z)
            if position is not None:
                return _CellPoint(x, z, batch, position)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    return _ORIGINAL_POINT(*args, **kwargs)


def _fast_profile_height(
    distance: float,
    distances: Sequence[float],
    heights: Sequence[float],
) -> float:
    if isinstance(distance, _ProjectedDistance):
        batch = distance.batch
        values = batch.interpolated_profile(distances, heights)
        return float(values[distance.position])
    return float(_ORIGINAL_PROFILE_HEIGHT(distance, distances, heights))


def _fast_road_span_has_in_game_water(*args: Any, **kwargs: Any) -> bool:
    if _ACTIVE_NEEDS_WATER_TEST.get() is False:
        return False
    return bool(_ORIGINAL_ROAD_SPAN_WATER_TEST(*args, **kwargs))


def _mask_array(mask: Sequence[bool]) -> np.ndarray:
    cached = _MASK_ARRAY_CACHE.get()
    identity = id(mask)
    if cached is not None and cached[0] == identity:
        return cached[1]
    values = np.asarray(mask, dtype=np.bool_)
    _MASK_ARRAY_CACHE.set((identity, values))
    return values


def _fast_road_corridor_intersects_mask(
    line: Any,
    mask: Sequence[bool],
    spec: Any,
    width: float,
) -> bool:
    if _ACTIVE_NEEDS_WATER_TEST.get() is False:
        return False

    geometry = line._geometry if isinstance(line, _FastLine) else line
    radius = max(float(width) * 0.5, float(spec.cell_size) * 0.35)

    # Square caps and mitred joins can extend beyond the simple radius around
    # the line. Shapely's default mitre_limit is 5, so use that as a conservative
    # broad phase and only build the exact GEOS buffer if a water cell is nearby.
    broad_radius = radius * 5.0 + 1.0e-9
    min_x, min_z, max_x, max_z = (float(value) for value in geometry.bounds)
    broad_bounds = (
        min_x - broad_radius,
        min_z - broad_radius,
        max_x + broad_radius,
        max_z + broad_radius,
    )
    indices, _ = _corridor_candidate_indices(
        geometry,
        broad_radius,
        broad_bounds,
        int(spec.cells),
        float(spec.cell_size),
    )
    if indices.size == 0:
        return False
    selected = _mask_array(mask)[indices]
    if not np.any(selected):
        return False

    indices = indices[selected]
    xs = (indices % spec.cells).astype(np.float64) * float(spec.cell_size)
    zs = (indices // spec.cells).astype(np.float64) * float(spec.cell_size)
    points = vectorized_points(xs, zs)
    corridor = geometry.buffer(radius, cap_style=2, join_style=2)
    return bool(np.any(vectorized_covers(corridor, points)))


def install_road_constraint_performance_policy() -> None:
    """Batch hot road/cell geometry while preserving solver results."""

    global _INSTALLED, _ORIGINAL_ROAD_SPAN_WATER_TEST
    if _INSTALLED:
        return

    _ORIGINAL_ROAD_SPAN_WATER_TEST = _terrain.road_span_has_in_game_water

    _terrain._line_geometry = _fast_line_geometry
    _terrain._candidate_cells = _fast_candidate_cells
    _terrain.Point = _fast_point
    _terrain._profile_height = _fast_profile_height
    _terrain.road_span_has_in_game_water = _fast_road_span_has_in_game_water
    _terrain._road_corridor_intersects_mask = _fast_road_corridor_intersects_mask
    _INSTALLED = True
