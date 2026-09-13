# SPDX-License-Identifier: GPL-3.0-or-later
"""Vectorized cell/footprint tests for final building terrain pads."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from shapely import box as vectorized_box, intersects as vectorized_intersects

from . import road_constraint_performance_policy as _road_perf
from . import terrain_solver as _terrain


@dataclass(slots=True)
class _BuildingBatch:
    batch: Any
    boxes: Any
    intersections: dict[int, np.ndarray]

    @classmethod
    def create(cls, batch: Any) -> "_BuildingBatch":
        indices = batch.indices
        cells = int(batch.cells)
        cell_size = float(batch.cell_size)
        half = cell_size * 0.5
        xs = (indices % cells).astype(np.float64) * cell_size
        zs = (indices // cells).astype(np.float64) * cell_size
        boxes = vectorized_box(xs - half, zs - half, xs + half, zs + half)
        return cls(batch=batch, boxes=boxes, intersections={})

    def intersects(self, geometry: Any) -> np.ndarray:
        key = id(geometry)
        values = self.intersections.get(key)
        if values is None:
            values = np.asarray(
                vectorized_intersects(geometry, self.boxes),
                dtype=np.bool_,
            )
            self.intersections[key] = values
        return values


class _CellPolygonProxy:
    __slots__ = ("_building_batch", "_position", "_index", "_cells", "_cell_size")

    def __init__(
        self,
        building_batch: _BuildingBatch,
        position: int,
        index: int,
        cells: int,
        cell_size: float,
    ) -> None:
        self._building_batch = building_batch
        self._position = int(position)
        self._index = int(index)
        self._cells = int(cells)
        self._cell_size = float(cell_size)

    def intersects(self, geometry: Any) -> bool:
        return bool(self._building_batch.intersects(geometry)[self._position])

    def _scalar(self) -> Any:
        return _ORIGINAL_CELL_POLYGON(
            self._index,
            self._cells,
            self._cell_size,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scalar(), name)


_ACTIVE_BUILDING_BATCH: ContextVar[_BuildingBatch | None] = ContextVar(
    "cwr_building_pad_active_batch", default=None
)
_ORIGINAL_CANDIDATE_CELLS: Any = None
_ORIGINAL_CELL_POLYGON: Any = None
_ORIGINAL_POINT: Any = None
_INSTALLED = False


def _candidate_cells(
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> Iterable[int]:
    values = tuple(_ORIGINAL_CANDIDATE_CELLS(bounds, cells, cell_size))

    # The road/watercourse performance layer creates its own batch whenever a
    # line-buffer loop is active. Leave that batch alone. Building-pad loops are
    # the remaining candidate-cell users that arrive here without one.
    existing = _road_perf._ACTIVE_CELL_BATCH.get()
    if existing is not None:
        _ACTIVE_BUILDING_BATCH.set(None)
        return iter(values)

    if not values:
        _ACTIVE_BUILDING_BATCH.set(None)
        return iter(values)

    batch = _road_perf._CellBatch.create(values, cells, cell_size)
    _road_perf._ACTIVE_CELL_BATCH.set(batch)
    _ACTIVE_BUILDING_BATCH.set(_BuildingBatch.create(batch))
    return iter(values)


def _cell_polygon(index: int, cells: int, cell_size: float) -> Any:
    active = _ACTIVE_BUILDING_BATCH.get()
    if active is not None:
        batch = active.batch
        if batch.cells == cells and abs(batch.cell_size - float(cell_size)) <= 1.0e-12:
            position = batch.positions.get(int(index))
            if position is not None:
                return _CellPolygonProxy(active, position, index, cells, cell_size)
    return _ORIGINAL_CELL_POLYGON(index, cells, cell_size)


def _point(*args: Any, **kwargs: Any) -> Any:
    # Building transition code passes the centre to Polygon.distance(). That API
    # requires a real Shapely geometry. Only bypass the road layer while the
    # currently active terrain-cell batch is the building batch itself. A later
    # road/watercourse batch must still receive its lightweight point proxy.
    active = _ACTIVE_BUILDING_BATCH.get()
    current = _road_perf._ACTIVE_CELL_BATCH.get()
    if active is not None and active.batch is current:
        return _road_perf._ORIGINAL_POINT(*args, **kwargs)
    return _ORIGINAL_POINT(*args, **kwargs)


def install_building_pad_performance_policy() -> None:
    """Batch terrain-cell polygon intersection tests for building pads."""
    global _INSTALLED, _ORIGINAL_CANDIDATE_CELLS, _ORIGINAL_CELL_POLYGON, _ORIGINAL_POINT
    if _INSTALLED:
        return

    # Install after road_constraint_performance_policy so line-buffer candidate
    # pruning remains the inner implementation.
    _ORIGINAL_CANDIDATE_CELLS = _terrain._candidate_cells
    _ORIGINAL_CELL_POLYGON = _terrain._cell_polygon
    _ORIGINAL_POINT = _terrain.Point
    _terrain._candidate_cells = _candidate_cells
    _terrain._cell_polygon = _cell_polygon
    _terrain.Point = _point
    _INSTALLED = True
