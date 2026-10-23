# SPDX-License-Identifier: GPL-3.0-or-later
"""Original Abel and Cain underwater-only texture families.

Verified against the Island Studio 256x256, 50m texture exports from original
abel.wrp and cain.wrp: the water base fills the entire 1,020-cell map perimeter
on both maps. The listed mixed ground PAAs/PACs exist in the original WRP.
Their filename quadrant order is rotated 180 degrees on the world map.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StockCoastFamily:
    namespace: str
    water: str
    land: str
    extension: str
    quartets: frozenset[str]

    @property
    def water_path(self) -> str:
        return f"{self.namespace}\\{self.water}.paa"

    @property
    def land_path(self) -> str:
        return f"{self.namespace}\\{self.land}.paa"


# Only exact quartet filenames present in the original map are listed.
# Abel has all 14 mixed pi/tt corner arrangements; Cain is missing both
# geographic diagonal combinations (6, 9) of the l4/j9 family.
STOCK_COAST_FAMILIES: dict[str, StockCoastFamily] = {
    "malden": StockCoastFamily(
        "abel", "pi", "tt", "pac",
        frozenset("""
            pipipitt pipittpi pipitttt pittpipi pittpitt pittttpi
            pitttttt ttpipipi ttpipitt ttpittpi ttpitttt ttttpipi
            ttttpitt ttttttpi
        """.split()),
    ),
    "kolgujev": StockCoastFamily(
        "cain", "l4", "j9", "paa",
        frozenset("""
            j9j9j9l4 j9j9l4j9 j9j9l4l4 j9l4j9j9 j9l4j9l4
            j9l4l4l4 l4j9j9j9 l4j9l4j9 l4j9l4l4 l4l4j9j9
            l4l4j9l4 l4l4l4j9
        """.split()),
    ),
}


def stock_water_blend_path(
    profile: str, ne: int, nw: int, se: int, sw: int,
) -> str | None:
    """Find existing game art using GEOGRAPHIC NE,NW,SE,SW (0=sea,1=land)."""
    family = STOCK_COAST_FAMILIES.get(profile)
    if family is None:
        return None
    corners = (ne, nw, se, sw)
    if any(value not in (0, 1) for value in corners):
        return None
    if all(value == 0 for value in corners):
        return family.water_path
    if all(value == 1 for value in corners):
        return family.land_path
    tokens = (family.water, family.land)
    # Every graphic quadrant is 180 degrees opposite the filename token.
    stem = "".join(tokens[code] for code in reversed(corners))
    return (
        f"{family.namespace}\\{stem}.{family.extension}"
        if stem in family.quartets else None
    )


def _resolve_missing_diagonals(vertices: np.ndarray, family: StockCoastFamily) -> None:
    """Only expand grass at shared vertices, never introduce sea onto dry land.

    Cain lacks both diagonal quartet textures. Expanding grass resolves the
    unusable corner arrangements globally so adjoining PAAs share the exact
    same edge materials. No unverified filenames or quadrant rotations.
    """
    while True:
        ne, nw = vertices[1:, 1:], vertices[1:, :-1]
        se, sw = vertices[:-1, 1:], vertices[:-1, :-1]
        missing_9 = ((ne == 1) & (nw == 0) & (se == 0) & (sw == 1))
        missing_6 = ((ne == 0) & (nw == 1) & (se == 1) & (sw == 0))
        changed = False
        if "l4" in (family.water,):
            if np.any(missing_9):
                rows, cols = np.nonzero(missing_9)
                vertices[rows + 1, cols] = 1  # NW
                changed = True
            if np.any(missing_6):
                rows, cols = np.nonzero(missing_6)
                vertices[rows + 1, cols + 1] = 1  # NE
                changed = True
        if not changed:
            return


def apply_stock_underwater_borders(
    texture_paths: Sequence[str],
    texture_slots: Sequence[int],
    *,
    profile: str,
    width: int,
    underwater_mask: Sequence[bool],
    maximum_texture_slots: int = 512,
) -> tuple[tuple[str, ...], list[int]]:
    """Blend stock sea ground into grass strictly on underwater texture cells.

    All above-water water-labelled cells are forced to the normal land
    background. The chosen 50m water-cell border is a source-WRP directional
    tile only when its four corners connect seamlessly with neighbouring
    grass. Tiny isolated water cells with four all-grass corners use land
    ground instead of conspicuous rectangular full-water tiles, without
    changing their submerged elevation or water semantics. Protected ground
    tiles (forest, farmland, roads etc.) keep their existing texture slots.
    """
    family = STOCK_COAST_FAMILIES[profile]
    paths = list(texture_paths)
    slots = np.asarray(texture_slots, dtype=np.int32).reshape(-1)
    wet = np.asarray(underwater_mask, dtype=np.bool_).reshape(-1)
    if width <= 0 or len(slots) == 0 or len(slots) % width:
        raise ValueError("stock underwater texture grid has invalid dimensions")
    if len(wet) != len(slots):
        raise ValueError("underwater mask and texture grid dimensions must agree")
    if np.any((slots < 0) | (slots >= len(paths))):
        raise ValueError("stock underwater texture slots refer to missing paths")
    height = len(slots) // width
    originals = slots.reshape(height, width).copy()
    submerged = wet.reshape(height, width)
    kinds = np.full(len(paths), -1, dtype=np.int8)
    for slot, path in enumerate(paths):
        normalized = path.replace("/", "\\").casefold()
        if normalized == family.water_path:
            kinds[slot] = 0
        elif normalized == family.land_path:
            kinds[slot] = 1
    centres = kinds[originals]
    dry_sea = (centres == 0) & ~submerged
    if np.any(dry_sea):
        land_slot = next((i for i, kind in enumerate(kinds) if kind == 1), None)
        if land_slot is None:
            raise ValueError("stock coastal profile needs its land texture slot")
        originals[dry_sea] = land_slot
        centres[dry_sea] = 1
    underwater = (centres == 0) & submerged
    if not np.any(underwater):
        return tuple(paths), originals.ravel().tolist()

    padded = np.pad(centres, 1, constant_values=-2)
    neighbours = (
        padded[:-1, :-1], padded[:-1, 1:],
        padded[1:, :-1], padded[1:, 1:],
    )
    touches_dry_or_protected = np.logical_or.reduce(
        tuple((item == 1) | (item == -1) for item in neighbours)
    )
    vertices = np.where(touches_dry_or_protected, 1, 0).astype(np.int8)
    _resolve_missing_diagonals(vertices, family)
    ne, nw = vertices[1:, 1:], vertices[1:, :-1]
    se, sw = vertices[:-1, 1:], vertices[:-1, :-1]
    full_land = (ne == 1) & (nw == 1) & (se == 1) & (sw == 1)
    full_sea = (ne == 0) & (nw == 0) & (se == 0) & (sw == 0)
    # A one-cell ocean strip surrounded by 4 grass corners has no
    # original mixed artwork; replace only the visible ground tile.
    if np.any(underwater & full_land):
        land_slot = next(i for i, kind in enumerate(kinds) if kind == 1)
        originals[underwater & full_land] = land_slot
    border = underwater & (~full_land) & (~full_sea)
    if not np.any(border):
        return tuple(paths), originals.ravel().tolist()

    code_grid = (
        ne.astype(np.int32) + 2 * nw.astype(np.int32)
        + 4 * se.astype(np.int32) + 8 * sw.astype(np.int32)
    )
    path_to_slot = {
        path.replace("/", "\\").casefold(): i for i, path in enumerate(paths)
    }
    for corner_mask in sorted(int(code) for code in np.unique(code_grid[border])):
        geo_codes = tuple((corner_mask >> index) & 1 for index in range(4))
        path = stock_water_blend_path(profile, *geo_codes)
        if path is None:
            # Never insert nonexistent game art.
            continue
        slot = path_to_slot.get(path)
        if slot is None:
            if len(paths) >= maximum_texture_slots:
                continue
            slot = len(paths)
            paths.append(path)
            path_to_slot[path] = slot
        originals[border & (code_grid == corner_mask)] = slot
    return tuple(paths), originals.ravel().tolist()


def is_stock_water_blend_path(profile: str, path: str) -> bool:
    family = STOCK_COAST_FAMILIES.get(profile)
    if family is None:
        return False
    normalized = path.replace("/", "\\").casefold()
    prefix, suffix = f"{family.namespace}\\", f".{family.extension}"
    return (
        normalized.startswith(prefix) and normalized.endswith(suffix)
        and normalized[len(prefix):-len(suffix)] in family.quartets
    )
