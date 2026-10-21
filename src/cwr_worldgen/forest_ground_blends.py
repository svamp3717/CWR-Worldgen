# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified OFP/CWA forest-floor quadrant blends for the stock ground presets.

Four two-letter components are in NE, NW, SE, SW quadrant order.  The
original island WRP texture-location atlases establish both the names and
their orientation. Do not synthesize absent diagonal texture assets.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# (virtual namespace, forest code, ordinary grass code, blend extension,
#  verified quadrant masks in the source WRP texture table).
# Bit order: 1=NE, 2=NW, 4=SE, 8=SW.
STOCK_FOREST_BLEND_FAMILIES: dict[str, tuple[str, str, str, str, frozenset[int]]] = {
    "nogova": ("o", "j1", "t1", "pac", frozenset(range(1, 15))),
    "everon": ("eden", "pl", "tn", "paa", frozenset(set(range(1, 15)) - {9})),
    "malden": ("abel", "pb", "tt", "pac", frozenset(set(range(1, 15)) - {6, 9})),
    "kolgujev": ("cain", "u2", "j9", "paa", frozenset(range(1, 15))),
}


def stock_forest_blend_path(profile: str, quadrant_mask: int) -> str | None:
    """Return only a path verified in the matching original island's WRP."""
    family = STOCK_FOREST_BLEND_FAMILIES.get(profile.casefold())
    if family is None:
        return None
    namespace, forest, grass, extension, valid_masks = family
    if quadrant_mask not in valid_masks:
        return None
    components = "".join(
        forest if quadrant_mask & (1 << quadrant) else grass
        for quadrant in range(4)
    )
    return rf"{namespace}\{components}.{extension}"


def _quadrants(forest: np.ndarray) -> np.ndarray:
    """Select forest cover at a ground cell's NE,NW,SE,SW quadrants.

    Raster rows increase northward, unlike north-up image pixels. Each corner
    looks at four neighbouring land-use samples; >=3 forest samples means
    forest in that quadrant. The original centre remains part of all four
    footprints. This generates valid straight and convex/concave corner masks
    from the same OSM polygon used to place forest vegetation.
    """
    p = np.pad(forest.astype(np.uint8), 1)
    centre = p[1:-1, 1:-1]
    south = p[:-2, 1:-1]
    north = p[2:, 1:-1]
    west = p[1:-1, :-2]
    east = p[1:-1, 2:]
    sw = p[:-2, :-2]
    se = p[:-2, 2:]
    nw = p[2:, :-2]
    ne = p[2:, 2:]
    return (
        (centre + north + east + ne >= 3).astype(np.uint8)
        | ((centre + north + west + nw >= 3).astype(np.uint8) << 1)
        | ((centre + south + east + se >= 3).astype(np.uint8) << 2)
        | ((centre + south + west + sw >= 3).astype(np.uint8) << 3)
    )


def apply_stock_forest_border_blends(
    material_indices: Sequence[int],
    forest_raster: Sequence[bool],
    base_texture_paths: Sequence[str],
    *,
    width: int,
    profile: str,
    forest_material: int,
    forest_edge_material: int | None,
    enabled: bool = False,
    maximum_texture_slots: int = 512,
    blendable_exterior_materials: Sequence[int] = (),
) -> tuple[tuple[str, ...], list[int]]:
    """Build an expanded WRP texture table and exact per-cell texture slots.

    base_texture_paths includes slot zero (the dummy), followed by one texture
    per semantic material. With the option off it is byte-for-byte compatible
    with the pre-existing material+1 index mapping. With the option on, only
    cells in the original forest/edge semantic classes can change.

    Forest polygon boundaries are taken from the OSM forest raster, not guessed
    from object locations. Verified quadrant paths are appended only if used.
    Missing stock diagonal assets fall back to a verified one-quadrant-nearest
    blend; never emit a guessed path. An exhausted WRP slot table falls back
    to the original semantic texture slot. For Nogova, optionally complete
    the directional transition across immediately adjoining ordinary grass
    cells where the original forest tiles would otherwise expose a forest
    corner. Farmland, roads, water and other protected materials stay intact.
    """
    paths = list(base_texture_paths)
    indices = np.asarray(material_indices, dtype=np.int32).reshape(-1)
    count = len(indices)
    if width <= 0 or count % width or len(forest_raster) != count:
        raise ValueError("texture and forest raster dimensions must agree")
    slots = (indices + 1).astype(np.int32)
    if not enabled or profile.casefold() not in STOCK_FOREST_BLEND_FAMILIES:
        return tuple(paths), slots.tolist()
    height = count // width
    forest = np.asarray(forest_raster, dtype=np.bool_).reshape(height, width)
    material = indices.reshape(height, width)
    candidate = material == forest_material
    if forest_edge_material is not None:
        candidate |= material == forest_edge_material
    # The semantic overlay can have punched roads, rivers or buildings through
    # the forest raster; preserve those instead of painting them as forest.
    forest &= candidate
    masks = _quadrants(forest)
    current = slots.reshape(height, width)
    # Entire forest cells use the standalone forest floor; entirely external
    # edge cells use the existing background edge slot.  The edge floor slot
    # must therefore be configured as ordinary ground, not an invented blend.
    if forest_edge_material is not None:
        fully_forest_edge = candidate & (material == forest_edge_material) & (masks == 15)
        current[fully_forest_edge] = forest_material + 1
    # A forest-interior cell on a boundary still gets a directional transition.
    requested = candidate & (masks > 0) & (masks < 15)
    if not np.any(requested):
        return tuple(paths), slots.tolist()

    possible = STOCK_FOREST_BLEND_FAMILIES[profile.casefold()][4]
    assigned_blends = np.zeros((height, width), dtype=np.bool_)
    for mask in sorted(int(v) for v in np.unique(masks[requested])):
        output_mask = mask
        if output_mask not in possible:
            # Some diagonal combinations are missing in original Everon/Malden
            # texture tables; prefer a verified neighbour with one quadrant
            # more or fewer rather than fabricating a PAA/PAC path.
            output_mask = min(
                possible,
                key=lambda v: (
                    (v ^ mask).bit_count(),
                    abs(v.bit_count() - mask.bit_count()),
                    v,
                ),
            )
        path = stock_forest_blend_path(profile, output_mask)
        assert path is not None
        if path in paths:
            slot = paths.index(path)
        elif len(paths) < maximum_texture_slots:
            slot = len(paths)
            paths.append(path)
        else:
            continue
        assigned = requested & (masks == mask)
        current[assigned] = slot
        assigned_blends |= assigned

    # Every vertex shared by adjacent terrain cells must agree about whether
    # it contains forest. A transition written only to f/e semantic cells can
    # leave an exposed j1 corner against otherwise eligible ordinary grass.
    # Fill these immediate exterior neighbours with the matching stock
    # quarter-blend, instead of painting a flat t1 apron. This is based on the
    # already assigned WRP artwork, not a rotation of the texture images.
    if (profile.casefold() == "nogova" and blendable_exterior_materials
            and np.any(assigned_blends)):
        source_masks = np.zeros((height, width), dtype=np.uint8)
        source_masks[assigned_blends] = masks[assigned_blends]
        # Solid forest floor reaches all four corners of its WRP cell.
        source_masks[candidate & (current == forest_material + 1)] = 15

        vertices = np.zeros((height + 1, width + 1), dtype=np.bool_)
        # Stock quartet bit order: NE=1, NW=2, SE=4, SW=8.
        vertices[1:, 1:] |= (source_masks & 1) != 0
        vertices[1:, :-1] |= (source_masks & 2) != 0
        vertices[:-1, 1:] |= (source_masks & 4) != 0
        vertices[:-1, :-1] |= (source_masks & 8) != 0

        exterior_masks = (
            vertices[1:, 1:].astype(np.uint8)
            | (vertices[1:, :-1].astype(np.uint8) << 1)
            | (vertices[:-1, 1:].astype(np.uint8) << 2)
            | (vertices[:-1, :-1].astype(np.uint8) << 3)
        )
        allowed = (~candidate) & np.isin(
            material, tuple(blendable_exterior_materials)
        )
        requested_exterior = allowed & (exterior_masks != 0)
        # A rare one-cell hole can be forest at all four corners.
        current[requested_exterior & (exterior_masks == 15)] = forest_material + 1
        for mask in sorted(int(v) for v in np.unique(exterior_masks[
            requested_exterior & (exterior_masks != 15)
        ])):
            path = stock_forest_blend_path(profile, mask)
            assert path is not None  # Nogova has all fourteen stock masks.
            if path in paths:
                slot = paths.index(path)
            elif len(paths) < maximum_texture_slots:
                slot = len(paths)
                paths.append(path)
            else:
                continue
            current[requested_exterior & (exterior_masks == mask)] = slot
    return tuple(paths), slots.tolist()
