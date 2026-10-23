# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-measured terrain clutter for Nogova, Eden, Abel and Cain.

All four island palettes reuse exact original WRP P3D paths with their
measured relative counts. Enabled only by the opt-in checkbox and the
corresponding tree preset; no geographic stock placements are copied.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from .stock_terrain_clutter_profiles import (
    STOCK_CLUTTER_WEIGHTS, stock_clutter_tree_preset,
)

from .osm import WorldObject, _sample_elevation

# Leaf and pine are the two supported Nogova/Resistance tree presets.
_NOGOVA_FOREST_BLOCKS = frozenset({
    r"o\tree\les_nw_ctver_pruhozi_t1.p3d",
    r"o\tree\les_nw_jehl_ctver_pruhozi.p3d",
})
_NOGOVA_SINGLE_TREES = frozenset({
    r"o\tree\javor01.p3d", r"o\tree\smrk_maly.p3d",
})

# Exact model counts from noe.wrp object-models.csv, restricted to small
# grounded bushes and saplings. Larger trees, forest polygons, structures
# and rocks are handled by their existing dedicated placement stages.
NOGOVA_CLUTTER_WEIGHTS: tuple[tuple[str, int], ...] = (
    (r"o\tree\dd_bush02d.p3d", 27602),
    (r"o\tree\dd_bush01c.p3d", 27045),
    (r"o\tree\dd_bush01b.p3d", 10815),
    (r"o\tree\dd_bush02.p3d", 9650),
    (r"o\tree\dd_bush03.p3d", 7560),
    (r"o\tree\dd_bush02b.p3d", 4709),
    (r"o\tree\dd_bush01.p3d", 4604),
    (r"o\tree\dd_bush02c.p3d", 4129),
    (r"o\tree\dd_bush02_s.p3d", 2729),
    (r"o\tree\dd_bush01d_s.p3d", 1968),
    (r"o\tree\smrk_maly.p3d", 21859),
)
_TOTAL_WEIGHT = sum(weight for _, weight in NOGOVA_CLUTTER_WEIGHTS)


def nogova_tree_preset_selected(spec: object) -> bool:
    """Require Nogova *tree* art, regardless of selected ground textures.

    Check the actual authoritative model paths, not forest_profile="everon"
    (used by Nogova leaf/pine GUI presets) or ground_textures="nogova"
    (which may be combined with Everon trees).
    """
    block = str(getattr(spec, "forest_tree_model", "")).replace("/", "\\").casefold()
    single = str(getattr(spec, "forest_single_tree_model", "")).replace("/", "\\").casefold()
    return block in _NOGOVA_FOREST_BLOCKS and single in _NOGOVA_SINGLE_TREES


def terrain_clutter_preset(spec: object) -> str | None:
    """Return matching vegetation family only, never the ground texture."""
    if nogova_tree_preset_selected(spec):
        return "nogova"
    return stock_clutter_tree_preset(spec)


def nogova_clutter_active(spec: object) -> bool:
    """Legacy Nogova-only predicate, preserved for existing consumers."""
    return bool(getattr(spec, "nogova_terrain_clutter_enabled", False)) and nogova_tree_preset_selected(spec)


def terrain_clutter_active(spec: object) -> bool:
    """True for any exact supported stock vegetation preset when opted in."""
    return bool(getattr(spec, "nogova_terrain_clutter_enabled", False)) and terrain_clutter_preset(spec) is not None


def _weighted_model(
    selection: int,
    weights: tuple[tuple[str, int], ...] = NOGOVA_CLUTTER_WEIGHTS,
) -> str:
    total = _TOTAL_WEIGHT if weights is NOGOVA_CLUTTER_WEIGHTS else sum(w for _, w in weights)
    index = selection % total
    for name, weight in weights:
        if index < weight:
            return name
        index -= weight
    return weights[-1][0]


def place_nogova_terrain_clutter(
    spec: object,
    raster: object,
    elevations: Sequence[float],
    existing_objects: Sequence[WorldObject],
) -> tuple[WorldObject, ...]:
    """Place matching stock island bushes/saplings in empty dry natural ground.

    Checks coarse road/building/water/urban/farmland/forest masks, final terrain
    heights, local slope, and a spatial exclusion against already placed 3D
    objects. Extra clutter never replaces generated roads, forests or houses,
    and this stage does not alter any OSM raster, terrain material or height.
    """
    if not terrain_clutter_active(spec):
        return ()
    palette = terrain_clutter_preset(spec)
    # Palettes were read from the original island object-models.csv exports.
    # An explicit tree-model pair, not a matching texture, selects each pool.
    weights = (
        NOGOVA_CLUTTER_WEIGHTS if palette == "nogova"
        else STOCK_CLUTTER_WEIGHTS[palette]
    )
    cells = int(spec.cells)
    size = float(spec.cell_size)
    side = cells * size
    if cells < 2 or size <= 0 or len(elevations) != cells * cells:
        raise ValueError("Invalid terrain dimensions for Nogova clutter")
    forbidden_masks = tuple(
        getattr(raster, name) for name in
        ("water", "roads", "buildings", "urban", "farmland", "forest")
    )
    if any(len(mask) != cells * cells for mask in forbidden_masks):
        raise ValueError("Nogova clutter mask dimensions do not match the terrain")

    clearance = 7.0
    bucket_size = 12.0
    occupied: dict[tuple[int, int], list[tuple[float, float]]] = {}
    def bucket(x: float, z: float) -> tuple[int, int]:
        return (int(x // bucket_size), int(z // bucket_size))
    def record(x: float, z: float) -> None:
        occupied.setdefault(bucket(x, z), []).append((x, z))
    for obj in existing_objects:
        record(float(obj.x), float(obj.z))

    def near_existing(x: float, z: float) -> bool:
        bx, bz = bucket(x, z)
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for px, pz in occupied.get((bx + dx, bz + dz), ()):
                    if (px - x) ** 2 + (pz - z) ** 2 < clearance ** 2:
                        return True
        return False

    # 45-m candidate spacing and ~48% initial acceptance bound *density*,
    # not the total object count. Larger worlds naturally receive more
    # eligible clutter without a per-world ceiling.
    spacing = 45.0
    grid = max(0, int(side // spacing))
    if grid == 0:
        return ()
    seed = str(getattr(spec, "deterministic_seed", 0)).encode("utf-8")
    # Keep prior Nogova seeds identical to avoid changing already-approved
    # Resistance builds. Other palettes have their own deterministic streams.
    seed = hashlib.blake2s(
        seed + (b":nogova-terrain-clutter-v1" if palette == "nogova"
                else f":{palette}-terrain-clutter-v1".encode("ascii")),
        digest_size=12,
    ).digest()
    start_id = max((int(obj.object_id) for obj in existing_objects), default=0) + 1
    out: list[WorldObject] = []
    water_plane = float(getattr(spec, "sea_level", 0.0))
    # Interior margin also avoids placing bushes on a wraparound sea edge.
    for gz in range(grid):
        for gx in range(grid):
            digest = hashlib.blake2s(
                seed + gx.to_bytes(4, "little") + gz.to_bytes(4, "little"),
                digest_size=16,
            ).digest()
            if digest[0] >= 123:  # 48% acceptance before all safety gates
                continue
            x = (gx + 0.15 + digest[1] / 255 * 0.70) * spacing
            z = (gz + 0.15 + digest[2] / 255 * 0.70) * spacing
            if not (size <= x < side - size and size <= z < side - size):
                continue
            # Test at the point and 5 m around it so the little shrub crown
            # doesn't enter roads, fields, or buildings at cell boundaries.
            safe = True
            for ox, oz in ((0, 0), (-5, 0), (5, 0), (0, -5), (0, 5)):
                col = min(cells - 1, int((x + ox) / size))
                row = min(cells - 1, int((z + oz) / size))
                idx = row * cells + col
                if any(mask[idx] for mask in forbidden_masks):
                    safe = False
                    break
            if not safe or near_existing(x, z):
                continue
            y = _sample_elevation(elevations, cells, size, x, z)
            if y <= water_plane + 0.35:
                continue
            heights = [
                _sample_elevation(elevations, cells, size, x + dx, z + dz)
                for dx, dz in ((-1.5, 0), (1.5, 0), (0, -1.5), (0, 1.5))
            ]
            if max(heights) - min(heights) > 0.65:
                continue
            model = _weighted_model(int.from_bytes(digest[3:7], "little"), weights)
            heading = int.from_bytes(digest[7:9], "little") % 360
            # Slight root sink avoids the visibly floating bottom of DD bushes.
            obj = WorldObject(start_id + len(out), model, x, y - 0.025, z, float(heading))
            out.append(obj)
            record(x, z)
    return tuple(out)
