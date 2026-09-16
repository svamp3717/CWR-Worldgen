# SPDX-License-Identifier: GPL-3.0-or-later
"""Accelerate mapped-water run detection after terrain solving.

The bridge-or-causeway terrain wrapper reopens mapped water only after the core
terrain solver returns. Its historical `_mapped_water_runs` duplicated the old
0.5 m scalar source-water sampler, so a long explicit bridge road could spend
minutes in Python point-in-ring tests after progress had already reported
"Terrain constraint solution ready".

Reuse the indexed/vectorized source-water classifier while preserving separate
contiguous wet runs and the existing binary shoreline refinement semantics.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from . import bridge_source_water_performance_policy as _water_perf
from . import bridge_source_water_policy as _source

_INSTALLED = False
_ORIGINAL_MAPPED_WATER_RUNS: Any = None
_DIAGNOSTIC_SAMPLE_THRESHOLD = 10_000

PointXZ = tuple[float, float]


def _fast_mapped_water_runs(
    points: Sequence[PointXZ],
    context: Any,
) -> tuple[tuple[float, float], ...]:
    if context is None or not context.water:
        return ()

    cleaned, cumulative = _source._polyline_measure(points)
    if len(cleaned) < 2 or cumulative[-1] <= 0.01:
        return ()

    candidates = _water_perf._candidate_polygons(context, cleaned)
    if not candidates:
        return ()

    total = float(cumulative[-1])
    step = max(
        0.1,
        float(getattr(_source, "_SOURCE_WATER_SAMPLE_STEP_METRES", 0.5)),
    )
    count = max(1, int(math.ceil(total / step)))
    distances = np.asarray(
        [total * index / count for index in range(count + 1)],
        dtype=np.float64,
    )

    diagnostic = distances.size >= _DIAGNOSTIC_SAMPLE_THRESHOLD
    if diagnostic:
        print(
            "[bridge-postsolve-water] batched mapped-water runs: "
            f"length={total:,.1f}m, samples={distances.size:,}, "
            f"candidate_polygons={len(candidates):,}",
            flush=True,
        )

    xs, zs = _water_perf._sample_positions(cleaned, cumulative, distances)
    wet = _water_perf._vectorized_point_in_water(xs, zs, candidates)
    if not np.any(wet):
        if diagnostic:
            print(
                "[bridge-postsolve-water] mapped-water runs complete: no wet samples",
                flush=True,
            )
        return ()

    refinement_steps = max(
        0,
        int(getattr(_source, "_SOURCE_WATER_REFINEMENT_STEPS", 12)),
    )
    source_candidates = tuple(candidate.source for candidate in candidates)
    runs: list[tuple[float, float]] = []
    index = 0
    while index < wet.size:
        if not bool(wet[index]):
            index += 1
            continue

        first = index
        while index + 1 < wet.size and bool(wet[index + 1]):
            index += 1
        last = index

        start = float(distances[first])
        end = float(distances[last])

        if first > 0 and not bool(wet[first - 1]):
            low, high = float(distances[first - 1]), float(distances[first])
            for _ in range(refinement_steps):
                middle = (low + high) * 0.5
                point = _source._point_at(cleaned, cumulative, middle)
                if _source._point_in_water(point, source_candidates):
                    high = middle
                else:
                    low = middle
            start = high

        if last + 1 < wet.size and not bool(wet[last + 1]):
            low, high = float(distances[last]), float(distances[last + 1])
            for _ in range(refinement_steps):
                middle = (low + high) * 0.5
                point = _source._point_at(cleaned, cumulative, middle)
                if _source._point_in_water(point, source_candidates):
                    low = middle
                else:
                    high = middle
            end = low

        if end > start + 1.0e-4:
            runs.append((start, end))
        index += 1

    result = tuple(runs)
    if diagnostic:
        print(
            "[bridge-postsolve-water] mapped-water runs complete: "
            f"runs={len(result):,}",
            flush=True,
        )
    return result


def install_bridge_postsolve_water_performance_policy() -> None:
    global _INSTALLED, _ORIGINAL_MAPPED_WATER_RUNS
    if _INSTALLED:
        return

    from . import bridge_or_causeway_terrain_policy as _bridge_terrain

    _ORIGINAL_MAPPED_WATER_RUNS = _bridge_terrain._mapped_water_runs
    _bridge_terrain._mapped_water_runs = _fast_mapped_water_runs
    _INSTALLED = True
