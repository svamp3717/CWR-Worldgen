# SPDX-License-Identifier: GPL-3.0-or-later
"""Everon's original water/sand/grass ground transitions.

The original eden.wrp texture atlas (256x256 at 50 m) confirms the three
base tiles ps (underwater), sz (shore) and tn (land), plus the directional
four-quadrant tiles below. A 180-degree filename-to-world transformation
gives zero discontinuous shared corners over 227,376 comparisons against
the stock export; unrotated names give 16,825 mismatches.

Only literal stock WRP paths are emitted. No original art is repackaged.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Base paths in sea-to-land order. Textures are loaded from the game's Eden PBO.
_EVERON_SHORE_BASE = (
    r"eden\ps.paa",
    r"eden\sz.paa",
    r"eden\tn.paa",
)
_CODES = ("ps", "sz", "tn")

# The EXACT two- and three-material quartet filenames in the supplied
# original Eden WRP texture-location export. Do not manufacture missing
# combinations: the original table is not a complete 3**4 permutation set.
_EVERON_SHORE_QUARTETS = frozenset("""
pspspssz pspspstn pspsszps pspsszsz pspssztn pspstnps pspstnsz
pspstntn psszpsps psszpssz psszpstn psszszps psszszsz psszsztn
pssztnsz pssztntn pstnpsps pstnpssz pstnpstn pstnszps pstnszsz
pstnsztn pstntnps pstntnsz pstntntn szpspsps szpspssz szpsszps
szpsszsz szpssztn szpstnps szpstnsz szpstntn szszpsps szszpssz
szszpstn szszszps szszsztn szsztnps szsztnsz szsztntn sztnpsps
sztnpssz sztnpstn sztnszps sztnszsz sztnsztn sztntnps sztntnsz
sztntntn tnpspsps tnpspssz tnpsszps tnpsszsz tnpstnps tnpstnsz
tnpstntn tnszpsps tnszpssz tnszpstn tnszszps tnszszsz tnszsztn
tnsztnps tnsztnsz tnsztntn tntnpsps tntnpssz tntnpstn tntnszps
tntnszsz tntnsztn tntntnps tntntnsz
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
    if any(code not in (0, 1, 2) for code in corners):
        return None
    if len(set(corners)) == 1:
        return _EVERON_SHORE_BASE[corners[0]]
    stem = "".join(_CODES[code] for code in reversed(corners))
    return rf"eden\{stem}.paa" if stem in _EVERON_SHORE_QUARTETS else None


def apply_everon_water_borders(
    texture_paths: Sequence[str],
    texture_slots: Sequence[int],
    *,
    width: int,
    maximum_texture_slots: int = 512,
) -> tuple[tuple[str, ...], list[int]]:
    """Blend existing ps/sz/tn cells with shared, north-up shoreline corners.

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

    # At every shared vertex, inspect the four touching tile centres once.
    # The vote gives each vertex a single material, so adjoining quarter
    # tiles are always continuous. For ties, favour shore then dry ground
    # over the seabed, preventing water bleed past the real land boundary.
    # Outside-map neighbours do not vote; protected interiors do not blend.
    padded = np.pad(centres, 1, constant_values=-2)
    neighbours = (
        padded[:-1, :-1], padded[:-1, 1:],
        padded[1:, :-1], padded[1:, 1:],
    )
    protected = np.logical_or.reduce(tuple(a == -1 for a in neighbours))
    priorities = (0, 2, 1)
    scores = np.stack([
        sum((piece == code).astype(np.int8) for piece in neighbours) * 4
        + priorities[code]
        for code in range(3)
    ])
    vertices = np.argmax(scores, axis=0).astype(np.int8)
    vertices[protected] = -1
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
        + 3 * np.maximum(nw, 0).astype(np.int32)
        + 9 * np.maximum(se, 0).astype(np.int32)
        + 27 * np.maximum(sw, 0).astype(np.int32)
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
            quadrants.append(code % 3)
            code //= 3
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
