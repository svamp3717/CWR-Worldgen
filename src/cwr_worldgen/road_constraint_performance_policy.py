# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance policy for the road-constraint phase of the terrain solver.

The terrain solver intentionally keeps its existing per-cell grading semantics,
but dense road networks used to cross the Python/GEOS boundary two or three
times for every candidate terrain cell.  This policy batches those geometry
queries with Shapely 2 ufuncs and skips bridge-water probing for ordinary
at-grade roads that can never become bridges.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
from shapely import (
    covers as vectorized_covers,
    distance as vectorized_distance,
    line_locate_point as vectorized_line_locate_point,
    points as vectorized_points,
)

from . import terrain_solver as _terrain


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

    @classmethod
    def create(
        cls,
        indices: tuple[int, ...],
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

    def position_for_point(self, point: Any) -> int | None:
        try:
            x = float(point.x)
            z = float(point.y)
        except (AttributeError, TypeError, ValueError):
            return None
        if self.cell_size <= 0.0:
            return None
        grid_x = int(round(x / self.cell_size))
        grid_z = int(round(z / self.cell_size))
        if not (0 <= grid_x < self.cells and 0 <= grid_z < self.cells):
            return None
        expected_x = grid_x * self.cell_size
        expected_z = grid_z * self.cell_size
        tolerance = max(1.0e-7, self.cell_size * 1.0e-9)
        if abs(x - expected_x) > tolerance or abs(z - expected_z) > tolerance:
            return None
        return self.positions.get(grid_z * self.cells + grid_x)


class _FastLine:
    __slots__ = ("_geometry", "_batch", "_tags")

    def __init__(self, geometry: Any, tags: Mapping[str, str]) -> None:
        self._geometry = geometry
        self._batch: _CellBatch | None = None
        self._tags = tags

    def __getattr__(self, name: str) -> Any:
        return getattr(self._geometry, name)

    def buffer(self, *args: Any, **kwargs: Any) -> "_FastCorridor":
        corridor = _FastCorridor(self, self._geometry.buffer(*args, **kwargs))
        _PENDING_CORRIDOR.set(corridor)
        return corridor

    def distance(self, other: Any) -> float:
        batch = self._batch
        if batch is not None:
            position = batch.position_for_point(other)
            if position is not None:
                if batch.distances is None:
                    batch.distances = np.asarray(
                        vectorized_distance(self._geometry, batch.points),
                        dtype=np.float64,
                    )
                return float(batch.distances[position])
        return float(self._geometry.distance(other))

    def project(self, other: Any, normalized: bool = False) -> float:
        batch = self._batch
        if batch is not None:
            position = batch.position_for_point(other)
            if position is not None:
                if batch.projections is None:
                    batch.projections = np.asarray(
                        vectorized_line_locate_point(self._geometry, batch.points),
                        dtype=np.float64,
                    )
                value = float(batch.projections[position])
                if normalized:
                    length = float(self._geometry.length)
                    return 0.0 if length <= 0.0 else value / length
                return value
        return float(self._geometry.project(other, normalized=normalized))


class _FastCorridor:
    __slots__ = ("line", "_geometry", "_batch")

    def __init__(self, line: _FastLine, geometry: Any) -> None:
        self.line = line
        self._geometry = geometry
        self._batch: _CellBatch | None = None

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
_ACTIVE_NEEDS_WATER_TEST: ContextVar[bool | None] = ContextVar(
    "cwr_road_constraint_needs_water_test",
    default=None,
)

_ORIGINAL_LINE_GEOMETRY = _terrain._line_geometry
_ORIGINAL_CANDIDATE_CELLS = _terrain._candidate_cells
_ORIGINAL_ROAD_SPAN_WATER_TEST = _terrain.road_span_has_in_game_water
_ORIGINAL_CORRIDOR_WATER_TEST = _terrain._road_corridor_intersects_mask
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
    geometry = _ORIGINAL_LINE_GEOMETRY(feature, projection)
    if geometry is None:
        _ACTIVE_NEEDS_WATER_TEST.set(None)
        return None
    tags = getattr(feature, "tags", {})
    needs_water = _needs_bridge_water_test(tags)
    _ACTIVE_NEEDS_WATER_TEST.set(needs_water)
    return _FastLine(geometry, tags)


def _fast_candidate_cells(
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> Iterable[int]:
    indices = tuple(_ORIGINAL_CANDIDATE_CELLS(bounds, cells, cell_size))
    pending = _PENDING_CORRIDOR.get()
    _PENDING_CORRIDOR.set(None)
    if pending is not None and _bounds_match(bounds, pending.bounds) and indices:
        pending.attach_batch(_CellBatch.create(indices, cells, cell_size))
    return iter(indices)


def _fast_road_span_has_in_game_water(*args: Any, **kwargs: Any) -> bool:
    needs_water = _ACTIVE_NEEDS_WATER_TEST.get()
    if needs_water is False:
        return False
    return bool(_ORIGINAL_ROAD_SPAN_WATER_TEST(*args, **kwargs))


def _fast_road_corridor_intersects_mask(*args: Any, **kwargs: Any) -> bool:
    needs_water = _ACTIVE_NEEDS_WATER_TEST.get()
    if needs_water is False:
        return False
    return bool(_ORIGINAL_CORRIDOR_WATER_TEST(*args, **kwargs))


def install_road_constraint_performance_policy() -> None:
    """Batch hot road/cell geometry while preserving solver results."""

    global _INSTALLED, _ORIGINAL_ROAD_SPAN_WATER_TEST, _ORIGINAL_CORRIDOR_WATER_TEST
    if _INSTALLED:
        return

    # Bridge runtime installation can replace the water-span predicate. Capture
    # whichever implementation is final at install time, then add only the
    # cheap ordinary-road guard around it.
    _ORIGINAL_ROAD_SPAN_WATER_TEST = _terrain.road_span_has_in_game_water
    _ORIGINAL_CORRIDOR_WATER_TEST = _terrain._road_corridor_intersects_mask

    _terrain._line_geometry = _fast_line_geometry
    _terrain._candidate_cells = _fast_candidate_cells
    _terrain.road_span_has_in_game_water = _fast_road_span_has_in_game_water
    _terrain._road_corridor_intersects_mask = _fast_road_corridor_intersects_mask
    _INSTALLED = True
