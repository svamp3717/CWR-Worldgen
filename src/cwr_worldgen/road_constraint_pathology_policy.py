# SPDX-License-Identifier: GPL-3.0-or-later
"""Bound pathological road/cell geometry work in the terrain solver.

The first road-constraint performance layer narrows large line corridors to
segment-local candidate cells. For unusually complex roads, however, asking
GEOS for distance and projection from every surviving cell to the *entire*
polyline can still be quadratic in practice. This layer computes the exact
point-to-segment minimum distance and corresponding along-line projection while
building the local broad phase, so the later terrain loop reuses those arrays
without another full-line GEOS pass.

It also arms a one-shot Python traceback dump when a reported road-constraint
checkpoint makes no progress for 60 seconds. That diagnostic is deliberately
silent during normal runs and resets at each progress checkpoint.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import faulthandler
import math
import re
import sys
from typing import Any, Callable, Iterable

import numpy as np

from . import building_pad_performance_policy as _building_perf
from . import road_constraint_performance_policy as _road_perf
from . import terrain_solver as _terrain


_ANALYTIC_BBOX_THRESHOLD = 16_384
_ANALYTIC_POINT_THRESHOLD = 512
_WATCHDOG_AFTER_ROAD = 45_708
_WATCHDOG_SECONDS = 60.0

_ORIGINAL_INNER_CANDIDATE_CELLS: Any = None
_ORIGINAL_SOLVE_TERRAIN: Any = None
_INSTALLED = False

_ROAD_PROGRESS_RE = re.compile(
    r"^Applying road terrain constraints\s+([\d,]+)/([\d,]+)$"
)


@dataclass(slots=True)
class _RoadWatchState:
    active: bool = False
    last_checkpoint: int = 0
    total: int = 0


_WATCH_STATE: ContextVar[_RoadWatchState | None] = ContextVar(
    "cwr_road_constraint_pathology_watch_state",
    default=None,
)


def _candidate_indices_for_piece(
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    cells: int,
    cell_size: float,
) -> np.ndarray:
    bounds = (
        min(float(a[0]), float(b[0])) - radius,
        min(float(a[1]), float(b[1])) - radius,
        max(float(a[0]), float(b[0])) + radius,
        max(float(a[1]), float(b[1])) + radius,
    )
    return _road_perf._candidate_indices_for_bounds(bounds, cells, cell_size)


def _analytic_corridor_values(
    geometry: Any,
    radius: float,
    cells: int,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return exact nearby indices, distances and along-line projections.

    Candidate rectangles are generated for short pieces of each source segment.
    Distances/projections are then calculated against those pieces analytically.
    Sorting by cell index, squared distance and projection chooses the same
    nearest-segment result a full polyline query would use, with the earliest
    along-line position breaking exact-distance ties.
    """

    coords = np.asarray(geometry.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0,
        )

    radius = max(0.0, float(radius))
    radius_limit2 = (radius + 1.0e-9) ** 2
    max_span = max(
        _road_perf._SEGMENT_MINIMUM_SPAN_METRES,
        float(cell_size) * _road_perf._SEGMENT_SPAN_CELLS,
        radius * _road_perf._SEGMENT_SPAN_RADII,
    )

    all_indices: list[np.ndarray] = []
    all_distance2: list[np.ndarray] = []
    all_projections: list[np.ndarray] = []
    local_candidate_instances = 0
    cumulative = 0.0

    for start, end in zip(coords[:-1, :2], coords[1:, :2]):
        delta = end - start
        segment_length = float(np.hypot(delta[0], delta[1]))
        if segment_length <= 1.0e-12:
            continue

        piece_count = max(1, int(math.ceil(segment_length / max_span)))
        for piece_index in range(piece_count):
            t0 = piece_index / piece_count
            t1 = (piece_index + 1) / piece_count
            a = start + delta * t0
            b = start + delta * t1
            piece = b - a
            piece_length2 = float(piece[0] * piece[0] + piece[1] * piece[1])
            if piece_length2 <= 1.0e-24:
                continue

            indices = _candidate_indices_for_piece(
                a, b, radius, cells, cell_size
            )
            if indices.size == 0:
                continue
            local_candidate_instances += int(indices.size)

            xs = (indices % cells).astype(np.float64) * float(cell_size)
            zs = (indices // cells).astype(np.float64) * float(cell_size)
            offsets_x = xs - float(a[0])
            offsets_z = zs - float(a[1])
            piece_t = (
                offsets_x * float(piece[0]) + offsets_z * float(piece[1])
            ) / piece_length2
            piece_t = np.clip(piece_t, 0.0, 1.0)
            nearest_x = float(a[0]) + piece_t * float(piece[0])
            nearest_z = float(a[1]) + piece_t * float(piece[1])
            dx = xs - nearest_x
            dz = zs - nearest_z
            distance2 = dx * dx + dz * dz
            keep = distance2 <= radius_limit2
            if not np.any(keep):
                continue

            source_t = t0 + piece_t[keep] * (t1 - t0)
            projections = cumulative + source_t * segment_length
            all_indices.append(indices[keep])
            all_distance2.append(distance2[keep])
            all_projections.append(projections)

        cumulative += segment_length

    if not all_indices:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            local_candidate_instances,
        )

    indices = np.concatenate(all_indices)
    distance2 = np.concatenate(all_distance2)
    projections = np.concatenate(all_projections)

    order = np.lexsort((projections, distance2, indices))
    indices = indices[order]
    distance2 = distance2[order]
    projections = projections[order]

    first = np.empty(indices.size, dtype=np.bool_)
    first[0] = True
    first[1:] = indices[1:] != indices[:-1]
    indices = indices[first]
    distance2 = distance2[first]
    projections = projections[first]

    return indices, np.sqrt(distance2), projections, local_candidate_instances


def _pathological_candidate_cells(
    bounds: tuple[float, float, float, float],
    cells: int,
    cell_size: float,
) -> Iterable[int]:
    pending = _road_perf._PENDING_CORRIDOR.get()
    if (
        pending is None
        or pending.radius is None
        or not _road_perf._bounds_match(bounds, pending.bounds)
    ):
        return _ORIGINAL_INNER_CANDIDATE_CELLS(bounds, cells, cell_size)

    geometry = pending.line._geometry
    full_count = _road_perf._candidate_count(bounds, cells, cell_size)
    point_count = len(geometry.coords)
    if (
        full_count <= _ANALYTIC_BBOX_THRESHOLD
        and point_count <= _ANALYTIC_POINT_THRESHOLD
    ):
        return _ORIGINAL_INNER_CANDIDATE_CELLS(bounds, cells, cell_size)

    # We own this pending corridor now; prevent the inner layer from consuming it.
    _road_perf._PENDING_CORRIDOR.set(None)
    print(
        "[road-constraint] analytic complex-road corridor: "
        f"points={point_count:,}, length={float(geometry.length):,.1f}m, "
        f"bbox_candidates={full_count:,}, radius={float(pending.radius):.1f}m",
        flush=True,
    )

    indices, distances, projections, local_instances = _analytic_corridor_values(
        geometry,
        float(pending.radius),
        int(cells),
        float(cell_size),
    )
    if indices.size == 0:
        _road_perf._ACTIVE_CELL_BATCH.set(None)
        print(
            "[road-constraint] analytic complex-road corridor complete: "
            f"local_candidate_instances={local_instances:,}, exact_candidates=0",
            flush=True,
        )
        return iter(())

    batch = _road_perf._CellBatch.create(indices, cells, cell_size)
    batch.distances = distances
    batch.projections = projections
    pending.attach_batch(batch)
    _road_perf._ACTIVE_CELL_BATCH.set(batch)

    print(
        "[road-constraint] analytic complex-road corridor complete: "
        f"local_candidate_instances={local_instances:,}, "
        f"exact_candidates={indices.size:,}",
        flush=True,
    )
    return (int(index) for index in indices)


def _cancel_watchdog() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except (AttributeError, RuntimeError):
        pass


def _arm_watchdog() -> None:
    _cancel_watchdog()
    try:
        faulthandler.dump_traceback_later(
            _WATCHDOG_SECONDS,
            repeat=False,
            file=sys.stderr,
        )
    except (AttributeError, RuntimeError, ValueError):
        pass


def _solve_with_road_watchdog(*args: Any, **kwargs: Any) -> Any:
    original_callback: Callable[[int, str], None] | None = kwargs.get(
        "progress_callback"
    )
    state = _RoadWatchState()
    token = _WATCH_STATE.set(state)

    def progress(percent: int, stage: str) -> None:
        match = _ROAD_PROGRESS_RE.match(str(stage))
        if match is not None:
            state.active = True
            state.last_checkpoint = int(match.group(1).replace(",", ""))
            state.total = int(match.group(2).replace(",", ""))
            if state.last_checkpoint >= _WATCHDOG_AFTER_ROAD:
                _arm_watchdog()
                print(
                    "[road-constraint] no-progress watchdog armed at "
                    f"{state.last_checkpoint:,}/{state.total:,}; "
                    f"a Python traceback will be printed after "
                    f"{int(_WATCHDOG_SECONDS)}s without the next checkpoint",
                    flush=True,
                )
        elif state.active and not str(stage).startswith(
            "Preparing terrain profiles"
        ):
            state.active = False
            _cancel_watchdog()

        if original_callback is not None:
            original_callback(percent, stage)

    kwargs["progress_callback"] = progress
    try:
        return _ORIGINAL_SOLVE_TERRAIN(*args, **kwargs)
    finally:
        _cancel_watchdog()
        _WATCH_STATE.reset(token)


def install_road_constraint_pathology_policy() -> None:
    """Install complex-road exact geometry and a no-progress watchdog."""

    global _INSTALLED, _ORIGINAL_INNER_CANDIDATE_CELLS, _ORIGINAL_SOLVE_TERRAIN
    if _INSTALLED:
        return

    inner = _building_perf._ORIGINAL_CANDIDATE_CELLS
    if inner is None:
        raise RuntimeError(
            "road constraint pathology policy must install after building pad policy"
        )

    _ORIGINAL_INNER_CANDIDATE_CELLS = inner
    _ORIGINAL_SOLVE_TERRAIN = _terrain.solve_terrain_constraints
    _building_perf._ORIGINAL_CANDIDATE_CELLS = _pathological_candidate_cells
    _terrain.solve_terrain_constraints = _solve_with_road_watchdog
    _INSTALLED = True
