# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified Everon ps sandy ground to tn grass transitions.

The original Eden WRP texture-path table contains thirteen ps/tn direction
tiles. sz.paa is stone artwork, not sand; this preset therefore uses ps as
the beach/seabed base and tn as land, without introducing sz or guessed PAAs.
A 180-degree filename rotation matches original geographic quadrants.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Both wet and dry shore (as well as underwater) use sandy ps, not stone sz.
_EVERON_SHORE_BASE = (r"eden\ps.paa", r"eden\tn.paa")
_CODES = ("ps", "tn")

# Exactly the thirteen ps/tn quartets present in the user's eden.wrp export.
# Geographic NE+SW land diagonal (mask 9) is absent; never invent a PAA.
_EVERON_SHORE_QUARTETS = frozenset("""
pstntnps pspstntn pstntntn tntnpsps pspspstn pstnpstn pstnpsps
tntnpstn tnpstntn tntntnps tnpstnps tnpspsps pspstnps
""".split())


def everon_shore_blend_path(
    north_east: int, north_west: int, south_east: int, south_west: int,
) -> str | None:
    """Return a verified Eden tile for four GEOGRAPHIC corner materials.

    Every filename's source corners are the geographic opposite, in the
    original NE,NW,SE,SW order. An unavailable three-way or diagonal tile
    returns None instead of silently referencing a non-existent asset.
    """
    corners = (north_east, north_west, south_east, south_west)
    if any(code not in (0, 1) for code in corners):
        return None
    if len(set(corners)) == 1:
        return _EVERON_SHORE_BASE[corners[0]]
    stem = "".join(_CODES[code] for code in reversed(corners))
    return rf"eden\{stem}.paa" if stem in _EVERON_SHORE_QUARTETS else None


def is_everon_water_blend_path(path: str) -> bool:
    """True only for a literal directional quartet from original eden.wrp."""
    normalized = path.replace("/", "\\").casefold()
    prefix = "eden\\"
    suffix = ".paa"
    return (
        normalized.startswith(prefix) and normalized.endswith(suffix)
        and normalized[len(prefix):-len(suffix)] in _EVERON_SHORE_QUARTETS
    )


def _resolve_missing_ps_tn_diagonal(vertices: np.ndarray) -> None:
    """Resolve missing NE+SW grass diagonal without breaking shared corners.

    Eden has 13/14 binary quartet paths: only geographic mask 9 is missing.
    Remove the SW grass vertex from such a cell, updating the same vertex for
    its neighbours. Removal is monotone and terminates even for patterns of
    multiple adjacent missing diagonals.
    """
    while True:
        ne, nw = vertices[1:, 1:], vertices[1:, :-1]
        se, sw = vertices[:-1, 1:], vertices[:-1, :-1]
        missing = (ne == 1) & (nw == 0) & (se == 0) & (sw == 1)
        if not np.any(missing):
            return
        rows, cols = np.nonzero(missing)
        vertices[rows, cols] = 0


def apply_everon_water_borders(
    texture_paths: Sequence[str],
    texture_slots: Sequence[int],
    *,
    width: int,
    maximum_texture_slots: int = 512,
) -> tuple[tuple[str, ...], list[int]]:
    """Blend ps sand and tn grass with shared north-up shoreline corners.

    This runs after optional forest blends, so forest/road/farmland artwork
    remains untouched. A vertex next to protected ground is left alone; any
    cell depending on it retains its original full texture. Missing quartet
    artwork and texture-table exhaustion also retain the original cell slot.

    Returned WRP row order is south to north, with no height/semantic edits.
    """
    paths = list(texture_paths)
    flat = np.asarray(texture_slots, dtype=np.int32).reshape(-1)
    if width <= 0 or not len(flat) or len(flat) % width:
        raise ValueError("shoreline texture grid has invalid dimensions")
    if np.any((flat < 0) | (flat >= len(paths))):
        raise ValueError("shoreline texture slots refer to missing paths")
    height = len(flat) // width
    original = flat.reshape(height, width)
    kind_by_slot = np.full(len(paths), -1, dtype=np.int8)
    for slot, path in enumerate(paths):
        normalized = path.replace("/", "\\").casefold()
        for code, family in enumerate(_EVERON_SHORE_BASE):
            if normalized == family:
                kind_by_slot[slot] = code
                break
    centres = kind_by_slot[original]
    if not np.any(centres == 0):
        return tuple(paths), flat.tolist()  # No sea or underwater ground.

    # Each vertex belongs to all adjoining 50-metre tiles. In a two-versus-two
    # tie the geographic grass corner wins, putting the gradual border
    # on the ps-facing cell instead of flooding inland grass.
    padded = np.pad(centres, 1, constant_values=-2)
    neighbours = (
        padded[:-1, :-1], padded[:-1, 1:],
        padded[1:, :-1], padded[1:, 1:],
    )
    protected = np.logical_or.reduce(tuple(a == -1 for a in neighbours))
    priorities = (0, 1)
    scores = np.stack([
        sum((piece == code).astype(np.int8) for piece in neighbours) * 4
        + priorities[code]
        for code in range(2)
    ])
    vertices = np.argmax(scores, axis=0).astype(np.int8)
    vertices[protected] = -1
    _resolve_missing_ps_tn_diagonal(vertices)
    ne, nw = vertices[1:, 1:], vertices[1:, :-1]
    se, sw = vertices[:-1, 1:], vertices[:-1, :-1]

    valid = (
        (centres >= 0) & (ne >= 0) & (nw >= 0)
        & (se >= 0) & (sw >= 0)
    )
    # Keep the original centre class when a feature is smaller than a tile.
    # A one-cell watercourse must never turn into solid grass.
    valid &= (
        (ne == centres) | (nw == centres)
        | (se == centres) | (sw == centres)
    )
    # A solid cell needs no new texture-table entry or remapping.
    valid &= ~((ne == centres) & (nw == centres)
               & (se == centres) & (sw == centres))
    if not np.any(valid):
        return tuple(paths), flat.tolist()

    code_grid = (
        np.maximum(ne, 0).astype(np.int32)
        + 2 * np.maximum(nw, 0).astype(np.int32)
        + 4 * np.maximum(se, 0).astype(np.int32)
        + 8 * np.maximum(sw, 0).astype(np.int32)
    )
    result = original.copy()
    path_slots = {
        path.replace("/", "\\").casefold(): i
        for i, path in enumerate(paths)
    }
    for key in sorted(int(k) for k in np.unique(code_grid[valid])):
        code = key
        quadrants = []
        for _ in range(4):
            quadrants.append(code % 2)
            code //= 2
        path = everon_shore_blend_path(*quadrants)
        if path is None:
            continue
        existing = path_slots.get(path)
        if existing is None:
            if len(paths) >= maximum_texture_slots:
                continue
            existing = len(paths)
            path_slots[path] = existing
            paths.append(path)
        result[valid & (code_grid == key)] = existing

    return tuple(paths), result.ravel().tolist()
