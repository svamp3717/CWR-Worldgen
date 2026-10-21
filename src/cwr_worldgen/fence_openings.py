# SPDX-License-Identifier: GPL-3.0-or-later
"""Seeded, collision-free footpath openings in fitted fence-piece sequences."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

# Leave a proper walking-sized opening, not just a crack between two posts.
FENCE_OPENING_MINIMUM_WIDTH_METRES = 5.0
FENCE_OPENING_MINIMUM_BOUNDARY_METRES = 18.0
FENCE_OPENING_AVERAGE_SPACING_METRES = 120.0
FENCE_OPENING_MINIMUM_SEPARATION_METRES = 55.0

# _line_chunks returns (centre_x, centre_z, heading, length, x0, z0, x1, z1).
FenceChunk = tuple[float, float, float, float, float, float, float, float]


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def fence_opening_chunk_indices(
    chunks: Sequence[FenceChunk],
    identity: str,
    *,
    minimum_width: float = FENCE_OPENING_MINIMUM_WIDTH_METRES,
) -> frozenset[int]:
    """Choose whole fence pieces to omit, leaving deterministic walkable gaps.

    Selections depend only on world seed / feature identity and segment
    geometry, never on Python's process hash or unrelated generation order.
    Group short donor pieces until the opening spans at least minimum_width.
    Never split across a sharp OSM corner or leave an edge with no endpoint
    segment. If a boundary is too short for a usable entrance, leave it alone.
    """
    lengths = [max(0.0, float(chunk[3])) for chunk in chunks]
    total = sum(lengths)
    if len(chunks) < 3 or total < max(
        FENCE_OPENING_MINIMUM_BOUNDARY_METRES, minimum_width * 2.5
    ):
        return frozenset()
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    candidates: list[tuple[bytes, int, int, float]] = []
    for start in range(1, len(chunks) - 1):
        # Avoid openings immediately after a corner. Both sides should
        # contain fence pieces so the gap reads as a passage, not an ending.
        if _angle_difference(chunks[start - 1][2], chunks[start][2]) > 7.0:
            continue
        end = start
        while end < len(chunks) - 1 and (
            end == start or _angle_difference(chunks[end - 1][2], chunks[end][2]) <= 7.0
        ) and cumulative[end + 1] - cumulative[start] < minimum_width - 1e-6:
            end += 1
        if end >= len(chunks) - 1:
            continue
        width = cumulative[end + 1] - cumulative[start]
        if width < minimum_width - 1e-6:
            continue
        if _angle_difference(chunks[end][2], chunks[end + 1][2]) > 7.0:
            continue
        middle = (cumulative[start] + cumulative[end + 1]) / 2.0
        rank = hashlib.blake2s(
            f"{identity}:opening:{start}:{end}".encode("utf-8"), digest_size=8
        ).digest()
        candidates.append((rank, start, end, middle))
    if not candidates:
        return frozenset()

    target = max(1, int(total / FENCE_OPENING_AVERAGE_SPACING_METRES + 0.5))
    chosen: list[tuple[int, int, float]] = []
    closed = (
        math.hypot(
            chunks[0][4] - chunks[-1][6], chunks[0][5] - chunks[-1][7],
        ) < 0.05
    )
    for _rank, start, end, middle in sorted(candidates):
        if any(
            min(abs(middle - previous), total - abs(middle - previous))
            if closed else abs(middle - previous)
            < FENCE_OPENING_MINIMUM_SEPARATION_METRES
            for _first, _last, previous in chosen
        ):
            continue
        if any(start <= last + 1 and end >= first - 1 for first, last, _ in chosen):
            continue
        chosen.append((start, end, middle))
        if len(chosen) >= target:
            break
    return frozenset(i for first, last, _ in chosen for i in range(first, last + 1))
