# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance policy for the road-constraint phase of the terrain solver.

Dense road networks used to spend most of this stage crossing the Python/GEOS
boundary one terrain cell at a time.  Keep the solver's existing grading and
priority semantics, but batch geometric queries with Shapely 2, discard bounding
box cells that cannot touch the road before entering the Python loop, avoid
constructing Shapely Point objects for grid-cell centres, and interpolate road
profiles in NumPy batches.
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
            self._geometry.buffer(*args, **kwargs),
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
    __slots__ = ("line", "_geometry", "_batch", "radius")

    def __init__(self, line: _FastLine, geometry: Any, radius: float | None) -> None:
        self.line = line
        self._geometry = geometry
        self._batch: _CellBatch | None = None
        self.radius = radius

    def __getattr__(self, name: str) -> Any:
        return getattr(self._geometry, name)

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
                        vectorized_covers(self._geometry, batch.points),
                        dtype=np.bool_,
                    )
                return bool(batch.covered[position])
        return bool(self._geometry.covers(other))


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
    indices = np.fromiter(
        _ORIGINAL_CANDIDATE_CELLS(bounds, cells, cell_size),
        dtype=np.int64,
    )
    pending = _PENDING_CORRIDOR.get()
    _PENDING_CORRIDOR.set(None)
    if pending is None or not _bounds_match(bounds, pending.bounds) or indices.size == 0:
        _ACTIVE_CELL_BATCH.set(None)
        return (int(index) for index in indices)

    # The terrain solver's line-buffer loops immediately reject cells whose
    # centre lies farther from the source line than the requested corridor
    # radius. Do that rejection once in vectorized GEOS before entering Python.
    # The bridge/water corridor predicate has its own exact vectorized `covers`
    # implementation below because mitred buffer corners have different semantics.
    batch = _CellBatch.create(indices, cells, cell_size)
    if pending.radius is not None:
        distances = batch.ensure_distances(pending.line._geometry)
        keep = distances <= pending.radius + 1.0e-9
        if not np.all(keep):
            batch = batch.subset(keep)

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
    corridor = geometry.buffer(radius, cap_style=2, join_style=2)
    indices = np.fromiter(
        _ORIGINAL_CANDIDATE_CELLS(corridor.bounds, spec.cells, spec.cell_size),
        dtype=np.int64,
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
    return bool(np.any(vectorized_covers(corridor, points)))


def install_road_constraint_performance_policy() -> None:
    """Batch hot road/cell geometry while preserving solver results."""

    global _INSTALLED, _ORIGINAL_ROAD_SPAN_WATER_TEST
    if _INSTALLED:
        return

    # Bridge runtime installation can replace the water-span predicate. Capture
    # whichever implementation is final at install time, then add only the
    # cheap ordinary-road guard around it.
    _ORIGINAL_ROAD_SPAN_WATER_TEST = _terrain.road_span_has_in_game_water

    _terrain._line_geometry = _fast_line_geometry
    _terrain._candidate_cells = _fast_candidate_cells
    _terrain.Point = _fast_point
    _terrain._profile_height = _fast_profile_height
    _terrain.road_span_has_in_game_water = _fast_road_span_has_in_game_water
    _terrain._road_corridor_intersects_mask = _fast_road_corridor_intersects_mask
    _INSTALLED = True
