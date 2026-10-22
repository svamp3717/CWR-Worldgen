# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified Everon ps submerged seabed to tn dry-land ground transitions.

ps is only applied where terrain elevations are at/below sea level; never
on above-water beach cells. Directional ps/tn blends from original Eden WRP
are restricted to wet cells and end in grass along the dry side of the border.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ps is submerged seabed only; tn is the ordinary dry-ground texture.
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
    underwater_mask: Sequence[bool],
    maximum_texture_slots: int = 512,
) -> tuple[tuple[str, ...], list[int]]:
    """Blend submerged ps into grass without painting any ps on dry land.

    The mask is computed from FINAL WRP terrain heights and the configured
    water level, not from beach/sand semantic classes or source OSM masks.
    Dry ps slots are converted to tn, and only underwater cells may receive
    mixed ps/tn PAA paths. The shared dry-side vertices are always grass.
    Protected forest, roads, farmland and rock keep their original slots.
    """
    paths = list(texture_paths)
    flat = np.asarray(texture_slots, dtype=np.int32).reshape(-1)
    if width <= 0 or not len(flat) or len(flat) % width:
        raise ValueError("shoreline texture grid has invalid dimensions")
    if np.any((flat < 0) | (flat >= len(paths))):
        raise ValueError("shoreline texture slots refer to missing paths")
    wet = np.asarray(underwater_mask, dtype=np.bool_).reshape(-1)
    if len(wet) != len(flat):
        raise ValueError("underwater mask and texture grid dimensions must agree")
    height = len(flat) // width
    original = flat.reshape(height, width).copy()
    submerged = wet.reshape(height, width)
    kind_by_slot = np.full(len(paths), -1, dtype=np.int8)
    for slot, path in enumerate(paths):
        normalized = path.replace("/", "\\\\").casefold()
        for code, family in enumerate(_EVERON_SHORE_BASE):
            if normalized == family:
                kind_by_slot[slot] = code
                break
    centres = kind_by_slot[original]

    # Guard against a source 'water' class on dry ground (e.g. an OSM
    # polygon that stays above the global water plane). The texture's
    # physical location, not its semantic label, controls ps placement.
    dry_ps = (centres == 0) & (~submerged)
    if np.any(dry_ps):
        grass_slot = next(
            (i for i, code in enumerate(kind_by_slot) if code == 1), None
        )
        if grass_slot is None:
            raise ValueError("Everon tn grass slot required for dry ground")
        original[dry_ps] = grass_slot
        centres[dry_ps] = 1
    if not np.any((centres == 0) & submerged):
        return tuple(paths), original.ravel().tolist()

    # A vertex on the dry side MUST be tn, even though its underwater
    # neighbor may be ps. Therefore the wet tile's border blends all the
    # way into tn at the coastline; a dry cell stays pure tn. Shared
    # vertices also ensure adjacent wet tiles agree on their borders.
    padded = np.pad(centres, 1, constant_values=-2)
    neighbours = (
        padded[:-1, :-1], padded[:-1, 1:],
        padded[1:, :-1], padded[1:, 1:],
    )
    protected = np.logical_or.reduce(tuple(a == -1 for a in neighbours))
    grass_touches = np.logical_or.reduce(tuple(a == 1 for a in neighbours))
    vertices = np.where(grass_touches, 1, 0).astype(np.int8)
    vertices[protected] = -1
    _resolve_missing_ps_tn_diagonal(vertices)
    ne, nw = vertices[1:, 1:], vertices[1:, :-1]
    se, sw = vertices[:-1, 1:], vertices[:-1, :-1]

    # No ps-containing directional texture ever gets assigned above water.
    valid = (
        (centres == 0) & submerged
        & (ne >= 0) & (nw >= 0) & (se >= 0) & (sw >= 0)
    )
    # Avoid swallowing a small water body whose four vertices face grass.
    valid &= ((ne == 0) | (nw == 0) | (se == 0) | (sw == 0))
    # There is no need to remap an all-ps cell.
    valid &= ~((ne == 0) & (nw == 0) & (se == 0) & (sw == 0))
    if not np.any(valid):
        return tuple(paths), original.ravel().tolist()

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
