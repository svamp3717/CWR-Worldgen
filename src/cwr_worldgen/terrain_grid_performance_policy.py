# SPDX-License-Identifier: GPL-3.0-or-later
"""Grid-wide performance refinements for the constraint terrain solver.

The terrain solver has several operations whose cost scales with every terrain
cell rather than with the number of OSM features.  Keep their observable
semantics, but move the bulk work out of Python loops:

* label four-neighbour water components from horizontal runs rather than a set
  and deque entry for every wet cell;
* build bounded Manhattan and eight-neighbour distance fields with NumPy array
  propagation rather than Python queues/heaps;
* suppress the solver's final per-cell cut/fill diagnostics loop, capture the
  exact internal before/after arrays, and compute the same report values in
  NumPy before returning the report.
"""
from __future__ import annotations

import builtins
from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Sequence

import numpy as np

from . import terrain_solver as _terrain


@dataclass(slots=True)
class _DiagnosticCapture:
    expected_size: int
    before: Sequence[float] | None = None
    after: Sequence[float] | None = None
    used: bool = False


_DIAGNOSTIC_CAPTURE: ContextVar[_DiagnosticCapture | None] = ContextVar(
    "cwr_terrain_grid_diagnostic_capture", default=None
)
_ACTIVE_FIELD: ContextVar[Any | None] = ContextVar(
    "cwr_terrain_grid_active_constraint_field", default=None
)

_ORIGINAL_COMPONENTS = _terrain._components
_ORIGINAL_DISTANCE_FROM_MASK = _terrain._distance_from_mask
_ORIGINAL_EUCLIDEAN_DISTANCE_FROM_MASK = _terrain._euclidean_distance_from_mask
_ORIGINAL_COMPONENT_TOUCHES_EDGE = _terrain._component_touches_world_edge
_ORIGINAL_MASK_FROM_COMPONENTS = _terrain._mask_from_components
_ORIGINAL_FIELD_CREATE = _terrain._ConstraintField.create.__func__
_ORIGINAL_SOLVE = _terrain.solve_terrain_constraints
_ORIGINAL_ZIP = builtins.zip
_INSTALLED = False


def _find(parent: list[int], value: int) -> int:
    root = value
    while parent[root] != root:
        root = parent[root]
    while parent[value] != value:
        next_value = parent[value]
        parent[value] = root
        value = next_value
    return root


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if left_root < right_root:
        parent[right_root] = left_root
    else:
        parent[left_root] = right_root


def _row_runs(row: np.ndarray) -> tuple[tuple[int, int], ...]:
    if row.size == 0 or not np.any(row):
        return ()
    padded = np.empty(row.size + 2, dtype=np.bool_)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = row
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(changes[0::2], changes[1::2])
    )


def _fast_components(mask: Sequence[bool], cells: int) -> list[list[int]]:
    """Return four-neighbour components without a Python set entry per cell."""
    cells = int(cells)
    values = np.asarray(mask, dtype=np.bool_)
    if cells <= 0 or values.size % cells != 0:
        return _ORIGINAL_COMPONENTS(mask, cells)
    rows = values.reshape((-1, cells))
    if not np.any(rows):
        return []

    # Each horizontal run is already a connected set.  Union only overlapping
    # runs in adjacent rows, reducing Python work from O(wet cells) to O(runs).
    parent: list[int] = []
    runs: list[tuple[int, int, int]] = []  # row, start, stop(exclusive)
    previous: list[tuple[int, int, int]] = []  # start, stop, run id

    for z, row in enumerate(rows):
        current: list[tuple[int, int, int]] = []
        for start, stop in _row_runs(row):
            run_id = len(parent)
            parent.append(run_id)
            runs.append((z, start, stop))
            current.append((start, stop, run_id))

        left = 0
        right = 0
        while left < len(current) and right < len(previous):
            current_start, current_stop, current_id = current[left]
            previous_start, previous_stop, previous_id = previous[right]
            if current_stop <= previous_start:
                left += 1
                continue
            if previous_stop <= current_start:
                right += 1
                continue
            _union(parent, current_id, previous_id)
            if current_stop <= previous_stop:
                left += 1
            if previous_stop <= current_stop:
                right += 1
        previous = current

    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for run_id, run in enumerate(runs):
        root = _find(parent, run_id)
        grouped.setdefault(root, []).append(run)

    components: list[list[int]] = []
    for component_runs in grouped.values():
        chunks = [
            np.arange(z * cells + start, z * cells + stop, dtype=np.int64)
            for z, start, stop in component_runs
        ]
        if chunks:
            components.append(np.concatenate(chunks).tolist())
    components.sort(key=lambda component: component[0] if component else math.inf)
    return components


def _fast_distance_from_mask(
    mask: Sequence[bool], cells: int, maximum: int
) -> list[int]:
    """Return the exact bounded four-neighbour distance field using NumPy."""
    cells = int(cells)
    maximum = int(maximum)
    values = np.asarray(mask, dtype=np.bool_)
    if cells <= 0 or values.size % cells != 0:
        return _ORIGINAL_DISTANCE_FROM_MASK(mask, cells, maximum)
    grid = values.reshape((-1, cells))
    distances = np.full(grid.shape, 10**9, dtype=np.int32)
    distances[grid] = 0
    if maximum <= 0 or not np.any(grid):
        return distances.reshape(-1).tolist()

    reached = grid.copy()
    frontier = grid.copy()
    neighbours = np.empty_like(grid)
    for distance in range(1, maximum + 1):
        neighbours.fill(False)
        neighbours[1:, :] |= frontier[:-1, :]
        neighbours[:-1, :] |= frontier[1:, :]
        neighbours[:, 1:] |= frontier[:, :-1]
        neighbours[:, :-1] |= frontier[:, 1:]
        np.logical_and(neighbours, np.logical_not(reached), out=frontier)
        if not np.any(frontier):
            break
        distances[frontier] = distance
        np.logical_or(reached, frontier, out=reached)
    return distances.reshape(-1).tolist()


def _relax_shift(
    target: np.ndarray,
    source: np.ndarray,
    target_rows: slice,
    target_cols: slice,
    source_rows: slice,
    source_cols: slice,
    cost: float,
) -> None:
    np.minimum(
        target[target_rows, target_cols],
        source[source_rows, source_cols] + cost,
        out=target[target_rows, target_cols],
    )


def _fast_euclidean_distance_from_mask(
    mask: Sequence[bool], cells: int, maximum: int
) -> list[float]:
    """Return the solver's bounded 8-neighbour chamfer distance field.

    The historical heap implementation is Dijkstra on an 8-neighbour grid with
    costs 1 and sqrt(2).  A path whose total cost is <= maximum can contain at
    most ``maximum`` grid edges, so ``maximum`` synchronous relaxation rounds
    produce the same bounded shortest paths while each round runs in NumPy.
    """
    cells = int(cells)
    maximum = int(maximum)
    values = np.asarray(mask, dtype=np.bool_)
    if cells <= 0 or values.size % cells != 0:
        return _ORIGINAL_EUCLIDEAN_DISTANCE_FROM_MASK(mask, cells, maximum)
    grid = values.reshape((-1, cells))
    distances = np.full(grid.shape, math.inf, dtype=np.float64)
    distances[grid] = 0.0
    if maximum <= 0 or not np.any(grid):
        return distances.reshape(-1).tolist()

    diagonal = math.sqrt(2.0)
    for _ in range(maximum):
        updated = distances.copy()
        _relax_shift(updated, distances, slice(1, None), slice(None), slice(None, -1), slice(None), 1.0)
        _relax_shift(updated, distances, slice(None, -1), slice(None), slice(1, None), slice(None), 1.0)
        _relax_shift(updated, distances, slice(None), slice(1, None), slice(None), slice(None, -1), 1.0)
        _relax_shift(updated, distances, slice(None), slice(None, -1), slice(None), slice(1, None), 1.0)
        _relax_shift(updated, distances, slice(1, None), slice(1, None), slice(None, -1), slice(None, -1), diagonal)
        _relax_shift(updated, distances, slice(1, None), slice(None, -1), slice(None, -1), slice(1, None), diagonal)
        _relax_shift(updated, distances, slice(None, -1), slice(1, None), slice(1, None), slice(None, -1), diagonal)
        _relax_shift(updated, distances, slice(None, -1), slice(None, -1), slice(1, None), slice(1, None), diagonal)
        updated[updated > float(maximum)] = math.inf
        if np.array_equal(updated, distances):
            break
        distances = updated
    return distances.reshape(-1).tolist()


def _fast_component_touches_world_edge(component: Sequence[int], cells: int) -> bool:
    values = np.asarray(component, dtype=np.int64)
    if values.size == 0:
        return False
    x = values % int(cells)
    z = values // int(cells)
    return bool(
        np.any((x == 0) | (x == int(cells) - 1) | (z == 0) | (z == int(cells) - 1))
    )


def _fast_mask_from_components(
    components: Sequence[Sequence[int]], size: int
) -> list[bool]:
    result = np.zeros(int(size), dtype=np.bool_)
    for component in components:
        if len(component):
            result[np.asarray(component, dtype=np.int64)] = True
    return result.tolist()


def _capture_field_create(cls: type, size: int) -> Any:
    field = _ORIGINAL_FIELD_CREATE(cls, size)
    _ACTIVE_FIELD.set(field)
    return field


def _diagnostic_zip(*iterables: Iterable[Any], strict: bool = False) -> Any:
    """Skip only the solver's unique ``zip(original, result)`` diagnostics scan."""
    capture = _DIAGNOSTIC_CAPTURE.get()
    if capture is not None and not capture.used and len(iterables) == 2:
        before, after = iterables
        try:
            size_matches = len(before) == capture.expected_size and len(after) == capture.expected_size
        except TypeError:
            size_matches = False
        if (
            size_matches
            and isinstance(before, tuple)
            and isinstance(after, list)
            and (not before or isinstance(before[0], float))
            and (not after or isinstance(after[0], float))
        ):
            capture.before = before
            capture.after = after
            capture.used = True
            # Feed equal values to the historical loop so it takes its immediate
            # zero-delta continue for every cell.  The vectorized replacement is
            # installed into the returned immutable report below.
            return _ORIGINAL_ZIP(after, after, strict=strict)
    return _ORIGINAL_ZIP(*iterables, strict=strict)


def _vectorized_diagnostics(
    before: Sequence[float],
    after: Sequence[float],
    categories: Sequence[str],
    cell_size: float,
) -> dict[str, Any]:
    before_values = np.asarray(before, dtype=np.float64)
    after_values = np.asarray(after, dtype=np.float64)
    delta = after_values - before_values
    changed_mask = np.abs(delta) > 1.0e-7
    changed_delta = delta[changed_mask]
    cell_area = float(cell_size) * float(cell_size)

    if changed_delta.size:
        cuts = np.maximum(-changed_delta, 0.0)
        fills = np.maximum(changed_delta, 0.0)
        maximum_cut = float(np.max(cuts))
        maximum_fill = float(np.max(fills))
        total_cut = float(np.sum(cuts) * cell_area)
        total_fill = float(np.sum(fills) * cell_area)
    else:
        maximum_cut = maximum_fill = total_cut = total_fill = 0.0

    category_stats: dict[str, dict[str, float | int]] = {}
    if changed_delta.size:
        changed_categories = np.asarray(categories, dtype=object)[changed_mask]
        for raw_category in np.unique(changed_categories):
            category = str(raw_category)
            selected = changed_categories == raw_category
            values = changed_delta[selected]
            cuts = np.maximum(-values, 0.0)
            fills = np.maximum(values, 0.0)
            category_stats[category] = {
                "changed_cells": int(values.size),
                "cut_volume_m3": float(np.sum(cuts) * cell_area),
                "fill_volume_m3": float(np.sum(fills) * cell_area),
                "maximum_cut_m": float(np.max(cuts)) if cuts.size else 0.0,
                "maximum_fill_m": float(np.max(fills)) if fills.size else 0.0,
            }

    return {
        "changed_cells": int(np.count_nonzero(changed_mask)),
        "maximum_cut": maximum_cut,
        "maximum_fill": maximum_fill,
        "total_cut_volume_m3": total_cut,
        "total_fill_volume_m3": total_fill,
        "category_adjustments": {key: category_stats[key] for key in sorted(category_stats)},
    }


def _fast_solve_terrain_constraints(*args: Any, **kwargs: Any) -> Any:
    elevations = args[0] if args else kwargs.get("elevations")
    spec = args[4] if len(args) >= 5 else kwargs.get("spec")
    if elevations is None or spec is None:
        return _ORIGINAL_SOLVE(*args, **kwargs)

    capture = _DiagnosticCapture(expected_size=len(elevations))
    capture_token = _DIAGNOSTIC_CAPTURE.set(capture)
    field_token = _ACTIVE_FIELD.set(None)
    try:
        report = _ORIGINAL_SOLVE(*args, **kwargs)
        field = _ACTIVE_FIELD.get()
    finally:
        _DIAGNOSTIC_CAPTURE.reset(capture_token)
        _ACTIVE_FIELD.reset(field_token)

    if (
        not capture.used
        or capture.before is None
        or capture.after is None
        or field is None
    ):
        return report

    diagnostics = _vectorized_diagnostics(
        capture.before,
        capture.after,
        field.categories,
        float(spec.cell_size),
    )
    return replace(report, **diagnostics)


def install_terrain_grid_performance_policy() -> None:
    """Install grid-wide NumPy implementations after terrain bridge policies."""
    global _INSTALLED
    if _INSTALLED:
        return

    _terrain._components = _fast_components
    _terrain._distance_from_mask = _fast_distance_from_mask
    _terrain._euclidean_distance_from_mask = _fast_euclidean_distance_from_mask
    _terrain._component_touches_world_edge = _fast_component_touches_world_edge
    _terrain._mask_from_components = _fast_mask_from_components
    _terrain._ConstraintField.create = classmethod(_capture_field_create)
    # ``zip`` is looked up in the terrain_solver module globals before builtins.
    # The wrapper delegates every call except the one distinctive diagnostics
    # pair described above.
    _terrain.zip = _diagnostic_zip
    _terrain.solve_terrain_constraints = _fast_solve_terrain_constraints
    _INSTALLED = True
