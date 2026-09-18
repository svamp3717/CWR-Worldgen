# SPDX-License-Identifier: GPL-3.0-or-later
"""Vectorize the scalar road-profile hot path in the terrain solver.

The road constraint loop already batches corridor/cell geometry, but profile
construction still called ``LineString.interpolate`` several times per profile
station and sampled terrain point-by-point. Long or node-heavy roads therefore
became disproportionately expensive on large worlds. This policy computes the
same arc-length positions from the line's coordinate segments in NumPy and
samples the terrain grid in batches, while keeping the solver's existing grade
limiting and sidehill formulas unchanged.
"""
from __future__ import annotations

from contextvars import ContextVar
import math
from typing import Any, Sequence

import numpy as np

from . import terrain_solver as _terrain

_INSTALLED = False
_ORIGINAL_PROFILE = _terrain._profile
_ORIGINAL_CROSS_SLOPE_PROFILE = _terrain._road_cross_slope_profile
_ELEVATION_ARRAY_CACHE: ContextVar[tuple[int, np.ndarray] | None] = ContextVar(
    "cwr_road_profile_elevation_array_cache",
    default=None,
)


def _elevation_array(values: Sequence[float]) -> np.ndarray:
    cached = _ELEVATION_ARRAY_CACHE.get()
    identity = id(values)
    if cached is not None and cached[0] == identity:
        return cached[1]
    array = np.asarray(values, dtype=np.float64)
    _ELEVATION_ARRAY_CACHE.set((identity, array))
    return array


def _line_positions(line: Any, distances: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/z coordinates at arc-length distances along a LineString.

    This reproduces Shapely's non-normalized ``line.interpolate(distance)`` for
    the non-negative, in-range distances used by the terrain solver. Degenerate
    duplicate vertices are ignored because they contribute zero arc length.
    """

    coords = np.asarray(line.coords, dtype=np.float64)
    requested = np.asarray(distances, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.zeros_like(requested), np.zeros_like(requested)
    if coords.shape[0] == 1:
        return (
            np.full(requested.shape, float(coords[0, 0]), dtype=np.float64),
            np.full(requested.shape, float(coords[0, 1]), dtype=np.float64),
        )

    starts = coords[:-1, :2]
    deltas = coords[1:, :2] - starts
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    keep = lengths > 1.0e-12
    if not np.any(keep):
        return (
            np.full(requested.shape, float(coords[0, 0]), dtype=np.float64),
            np.full(requested.shape, float(coords[0, 1]), dtype=np.float64),
        )

    starts = starts[keep]
    deltas = deltas[keep]
    lengths = lengths[keep]
    ends = np.cumsum(lengths)
    beginnings = ends - lengths
    total = float(ends[-1])
    values = np.clip(requested, 0.0, total)
    indices = np.searchsorted(ends, values, side="left")
    indices = np.clip(indices, 0, len(lengths) - 1)
    fractions = (values - beginnings[indices]) / lengths[indices]
    positions = starts[indices] + deltas[indices] * fractions[:, None]
    return positions[:, 0], positions[:, 1]


def _sample_elevations(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    xs: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """Vector form of terrain_solver._sample_elevation."""

    cells = int(cells)
    cell_size = float(cell_size)
    values = _elevation_array(elevations)
    fx = np.clip(np.asarray(xs, dtype=np.float64) / cell_size, 0.0, cells - 1.0)
    fz = np.clip(np.asarray(zs, dtype=np.float64) / cell_size, 0.0, cells - 1.0)
    x0 = np.floor(fx).astype(np.int64)
    z0 = np.floor(fz).astype(np.int64)
    x1 = np.minimum(cells - 1, x0 + 1)
    z1 = np.minimum(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    a = values[z0 * cells + x0] * (1.0 - tx) + values[z0 * cells + x1] * tx
    b = values[z1 * cells + x0] * (1.0 - tx) + values[z1 * cells + x1] * tx
    return a * (1.0 - tz) + b * tz


def _fast_profile(
    line: Any,
    original: Sequence[float],
    spec: Any,
    maximum_grade: float,
) -> tuple[list[float], list[float]]:
    spacing = max(2.0, min(10.0, float(spec.cell_size) / 3.0))
    length = float(line.length)
    count = max(1, int(math.ceil(length / spacing)))
    # Keep the historical arithmetic/order for deterministic profile distances.
    distances = [length * index / count for index in range(count + 1)]
    xs, zs = _line_positions(line, distances)
    heights = _sample_elevations(
        original,
        int(spec.cells),
        float(spec.cell_size),
        xs,
        zs,
    ).tolist()

    ratio = float(maximum_grade) / 100.0
    for index in range(1, len(heights)):
        limit = max(0.01, distances[index] - distances[index - 1]) * ratio
        heights[index] = min(
            max(heights[index], heights[index - 1] - limit),
            heights[index - 1] + limit,
        )
    for index in range(len(heights) - 2, -1, -1):
        limit = max(0.01, distances[index + 1] - distances[index]) * ratio
        heights[index] = min(
            max(heights[index], heights[index + 1] - limit),
            heights[index + 1] + limit,
        )
    return distances, heights


def _fast_cross_slope_profile(
    line: Any,
    original: Sequence[float],
    spec: Any,
    distances: Sequence[float],
    width: float,
) -> list[float]:
    if not distances:
        return []

    near_offset = max(
        float(width) * 0.75,
        float(spec.cell_size) * 0.65,
        2.0,
    )
    sample_offsets = (
        near_offset,
        max(near_offset * 1.5, float(width) * 1.25, float(spec.cell_size) * 1.15),
        max(near_offset * 2.5, float(width) * 2.0, float(spec.cell_size) * 2.0),
    )
    tangent_probe = max(2.0, min(12.0, float(spec.cell_size) * 0.5))
    distance_values = np.asarray(distances, dtype=np.float64)
    length = float(line.length)
    before_x, before_z = _line_positions(
        line, np.maximum(0.0, distance_values - tangent_probe)
    )
    after_x, after_z = _line_positions(
        line, np.minimum(length, distance_values + tangent_probe)
    )
    centre_x, centre_z = _line_positions(line, distance_values)

    dx = after_x - before_x
    dz = after_z - before_z
    magnitude = np.hypot(dx, dz)
    valid = magnitude > 1.0e-6
    nx = np.zeros_like(magnitude)
    nz = np.zeros_like(magnitude)
    nx[valid] = -dz[valid] / magnitude[valid]
    nz[valid] = dx[valid] / magnitude[valid]

    maximum = np.zeros_like(distance_values)
    for sample_offset in sample_offsets:
        left = _sample_elevations(
            original,
            int(spec.cells),
            float(spec.cell_size),
            centre_x + nx * sample_offset,
            centre_z + nz * sample_offset,
        )
        right = _sample_elevations(
            original,
            int(spec.cells),
            float(spec.cell_size),
            centre_x - nx * sample_offset,
            centre_z - nz * sample_offset,
        )
        slope = np.abs(left - right) / max(0.01, sample_offset * 2.0) * 100.0
        maximum = np.maximum(maximum, slope)
    maximum[~valid] = 0.0
    return maximum.tolist()


def install_road_profile_performance_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _terrain._profile = _fast_profile
    _terrain._road_cross_slope_profile = _fast_cross_slope_profile
    _INSTALLED = True
