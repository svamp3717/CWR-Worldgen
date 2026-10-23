# SPDX-License-Identifier: GPL-3.0-or-later
"""Nogova's original three-material shoreline: ps sea, pt coast, t1 grass.

Unlike Everon/Abel/Cain, Nogova's WRP has two complete directional stock
quartet families: ps/pt and pt/t1. They require a two-cell intermediate
underwater pt band. Original PAA/PAC references are used, not redistributed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NOGOVA_WATER_PATH = r"o\ps.paa"
NOGOVA_COAST_PATH = r"o\pt.paa"
NOGOVA_LAND_PATH = r"o\t1.paa"

# Exact PAC filename stems found in noe.wrp texture-paths.csv.
# Source filename quarters are rotated by 180° against geographic corners.
_NOGOVA_WATER_COAST = frozenset("""
pspspspt psptpspt psptpsps pspsptpt ptptpsps psptptpt ptptpspt
ptptptps ptpsptpt pspsptps ptpspsps ptpsptps ptpspspt psptptps
""".split())
_NOGOVA_COAST_LAND = frozenset("""
ptt1ptpt t1t1ptpt t1t1t1pt t1ptptpt ptptt1pt ptptptt1 ptt1ptt1
ptt1t1t1 t1t1ptt1 ptptt1t1 t1ptt1t1 t1ptt1pt t1ptptt1 ptt1t1pt
""".split())


def nogova_water_blend_path(ne: int, nw: int, se: int, sw: int) -> str | None:
    """Choose source Nogova art for NE/NW/SE/SW (0=ps, 1=pt, 2=t1)."""
    geo = (ne, nw, se, sw)
    if any(v not in (0, 1, 2) for v in geo):
        return None
    if len(set(geo)) == 1:
        return (NOGOVA_WATER_PATH, NOGOVA_COAST_PATH, NOGOVA_LAND_PATH)[geo[0]]
    if 0 in geo and 2 in geo:
        return None  # no complete ps/t1 quartet family
    codes = ("ps", "pt", "t1")
    name = "".join(codes[code] for code in reversed(geo))
    if 0 in geo:
        allowed = _NOGOVA_WATER_COAST
    else:
        allowed = _NOGOVA_COAST_LAND
    return rf"o\{name}.pac" if name in allowed else None


def is_nogova_water_blend_path(path: str) -> bool:
    normalized = path.replace("/", "\\").casefold()
    if not (normalized.startswith("o\\") and normalized.endswith(".pac")):
        return False
    name = normalized[2:-4]
    return name in _NOGOVA_WATER_COAST or name in _NOGOVA_COAST_LAND


def apply_nogova_underwater_borders(
    texture_paths: Sequence[str],
    texture_slots: Sequence[int],
    *,
    width: int,
    underwater_mask: Sequence[bool],
    maximum_texture_slots: int = 512,
) -> tuple[tuple[str, ...], list[int]]:
    """Put original ps->pt->t1 shore ramps on submerged water tiles only.

    Two wet-cell rings of intermediary pt keep ps/pt and pt/t1 quartets
    disjoint. A shared corner always has exactly one material. Dry or
    protected cells keep their original slots, even at the shoreline.
    One-cell submerged water surrounded by grass falls back to t1 ground
    since the original texture set cannot represent a wet centre under
    four all-grass corners, without changing the water height or mask.
    """
    paths = list(texture_paths)
    slots = np.asarray(texture_slots, dtype=np.int32).reshape(-1)
    wet = np.asarray(underwater_mask, dtype=np.bool_).reshape(-1)
    if width <= 0 or not len(slots) or len(slots) % width:
        raise ValueError("Nogova underwater texture grid has invalid dimensions")
    if len(slots) != len(wet):
        raise ValueError("Nogova underwater mask and texture grid dimensions must agree")
    if np.any((slots < 0) | (slots >= len(paths))):
        raise ValueError("Nogova underwater texture slots refer to missing paths")
    height = len(slots) // width
    painted = slots.reshape(height, width).copy()
    submerged = wet.reshape(height, width)
    normalized = [path.replace("/", "\\").casefold() for path in paths]
    water_slots = np.asarray([value == NOGOVA_WATER_PATH for value in normalized])
    land_slots = np.asarray([value == NOGOVA_LAND_PATH for value in normalized])
    water = water_slots[painted] & submerged
    if not np.any(water) and not np.any(water_slots[painted] & ~submerged):
        return tuple(paths), painted.ravel().tolist()
    land_slot = next((i for i, flag in enumerate(land_slots) if flag), None)
    if land_slot is None:
        raise ValueError("Nogova t1 dry ground slot is required")
    painted[water_slots[painted] & ~submerged] = land_slot
    if not np.any(water):
        return tuple(paths), painted.ravel().tolist()

    # A radius-2 coast band of wet cells ensures that a tile never touches
    # BOTH sea and dry-ground corners. Nogova has no full ps/t1 tile set.
    dry_or_protected = ~water
    padded = np.pad(dry_or_protected, 2, constant_values=False)
    near_dry = np.zeros((height, width), dtype=np.bool_)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            near_dry |= padded[2 + dy:2 + dy + height, 2 + dx:2 + dx + width]
    center = np.where(dry_or_protected, 2, np.where(near_dry, 1, 0)).astype(np.int8)
    # Outside the finite island map is ocean, not arbitrary land.
    padded_center = np.pad(center, 1, constant_values=0)
    neighbors = (
        padded_center[:-1, :-1], padded_center[:-1, 1:],
        padded_center[1:, :-1], padded_center[1:, 1:],
    )
    corner_dry = np.logical_or.reduce(tuple(v == 2 for v in neighbors))
    corner_coast = np.logical_or.reduce(tuple(v == 1 for v in neighbors))
    vertices = np.where(corner_dry, 2, np.where(corner_coast, 1, 0)).astype(np.int8)
    ne, nw = vertices[1:, 1:], vertices[1:, :-1]
    se, sw = vertices[:-1, 1:], vertices[:-1, :-1]
    all_t1 = (ne == 2) & (nw == 2) & (se == 2) & (sw == 2)
    painted[water & all_t1] = land_slot
    mixed = water & ~((ne == nw) & (nw == se) & (se == sw))
    homogenous_pt = water & (ne == 1) & (nw == 1) & (se == 1) & (sw == 1)
    targets = mixed | homogenous_pt
    if not np.any(targets):
        return tuple(paths), painted.ravel().tolist()

    # Every wet tile draws on just one of the 0/1 or 1/2 stock families.
    mask = (ne.astype(np.int32) + 3 * nw.astype(np.int32) +
            9 * se.astype(np.int32) + 27 * sw.astype(np.int32))
    path_slots = {
        path.replace("/", "\\").casefold(): i for i, path in enumerate(paths)
    }
    for combo in sorted(int(x) for x in np.unique(mask[targets])):
        code = combo
        values = []
        for _ in range(4):
            values.append(code % 3)
            code //= 3
        name = nogova_water_blend_path(*values)
        if name is None:
            continue
        existing = path_slots.get(name)
        if existing is None:
            if len(paths) >= maximum_texture_slots:
                continue
            existing = len(paths)
            paths.append(name)
            path_slots[name] = existing
        painted[targets & (mask == combo)] = existing
    return tuple(paths), painted.ravel().tolist()
