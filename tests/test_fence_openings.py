"""Deterministic, collision-free openings in generated fenced boundaries."""
from __future__ import annotations

import math

import pytest

from cwr_worldgen.fence_openings import (
    FENCE_OPENING_MINIMUM_WIDTH_METRES,
    fence_opening_chunk_indices,
)
from cwr_worldgen.osm import _line_chunks


def _runs(indices: frozenset[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in sorted(indices):
        if not runs or index != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(index)
    return runs


def test_long_fence_gets_traversable_seeded_openings() -> None:
    chunks = _line_chunks(((0.0, 0.0), (240.0, 0.0)), 5.0)
    openings = fence_opening_chunk_indices(chunks, "world-seed:way/17")
    assert len(_runs(openings)) == 2
    assert 0 not in openings and len(chunks) - 1 not in openings
    for run in _runs(openings):
        assert sum(chunks[i][3] for i in run) >= FENCE_OPENING_MINIMUM_WIDTH_METRES
        # Adjacent fence panels must remain either side of the opening.
        assert run[0] > 0 and run[-1] < len(chunks) - 1
    assert fence_opening_chunk_indices(chunks, "world-seed:way/17") == openings
    assert fence_opening_chunk_indices(chunks, "other-seed:way/17") != openings


def test_short_donors_create_multi_piece_gap_but_short_fences_stay_intact() -> None:
    pieces = _line_chunks(((0, 0), (82, 0)), 1.5)
    opening = fence_opening_chunk_indices(pieces, "small-pieces")
    assert opening
    assert sum(pieces[i][3] for i in opening) >= 5.0
    assert len(opening) >= 4
    short = _line_chunks(((0, 0), (12, 0)), 4.0)
    assert fence_opening_chunk_indices(short, "short") == frozenset()


def test_closed_field_ring_gets_opening_without_destroying_corners() -> None:
    vertices = ((0, 0), (90, 0), (90, 90), (0, 90), (0, 0))
    pieces = _line_chunks(vertices, 5.0)
    chosen = fence_opening_chunk_indices(pieces, "seed:field:9")
    assert chosen
    assert len(_runs(chosen)) >= 2  # 360 m perimeter receives several entrances.
    for index in chosen:
        assert 0 < index < len(pieces) - 1
        # Do not create a hole at a 90-degree turn.
        if index - 1 not in chosen:
            assert abs((pieces[index - 1][2] - pieces[index][2] + 180) % 360 - 180) <= 7
        if index + 1 not in chosen:
            assert abs((pieces[index + 1][2] - pieces[index][2] + 180) % 360 - 180) <= 7


def test_openings_do_not_touch_walls_or_hedges() -> None:
    # Fence openings are selected by the fence-only caller. The helper
    # never changes the original fitted chunks, their elevations, or models.
    chunks = _line_chunks(((0, 0), (160, 0)), 5.0)
    original = tuple(chunks)
    fence_opening_chunk_indices(chunks, "fence")
    assert chunks == original


@pytest.mark.parametrize("length", (4.0, 6.0, 9.5))
def test_one_accessible_gap_even_for_medium_sized_boundaries(length: float) -> None:
    chunks = _line_chunks(((0, 0), (55, 0)), length)
    omitted = fence_opening_chunk_indices(chunks, f"model:{length}")
    assert len(_runs(omitted)) == 1
    assert sum(chunks[i][3] for i in omitted) >= 5.0
