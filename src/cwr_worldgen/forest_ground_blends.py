# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified OFP/CWA forest-floor quadrant blends for the stock ground presets.

Worldgen uses NE, NW, SE, SW geographic mask bits. The corrected Nogova
and Everon stock artwork needs a 180-degree filename-mask transform.
Apply the same stock-quartet mapping to Abel/Malden and Cain/Kolgujev,
with independent visual verification still needed for those two islands.
Never synthesize absent assets.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# (virtual namespace, forest code, ordinary grass code, blend extension,
#  verified FILENAME quadrant masks in the original WRP texture table).
# Geographic and filename mask bit order: 1=NE, 2=NW, 4=SE, 8=SW.
# Everon's missing filename diagonal mask 9 remains geographic mask 9,
# because a 180-degree rotation leaves the two diagonal masks unchanged.
STOCK_FOREST_BLEND_FAMILIES: dict[str, tuple[str, str, str, str, frozenset[int]]] = {
    "nogova": ("o", "j1", "t1", "pac", frozenset(range(1, 15))),
    "everon": ("eden", "pl", "tn", "paa", frozenset(set(range(1, 15)) - {9})),
    "malden": ("abel", "pb", "tt", "pac", frozenset(set(range(1, 15)) - {6, 9})),
    "kolgujev": ("cain", "u2", "j9", "paa", frozenset(range(1, 15))),
}


def _opposite_filename_mask(world_mask: int) -> int:
    """Map geographic quadrants to opposite named stock-artwork quadrants.

    Nogova PAC and Everon PAA artwork establish this 180-degree layout;
    Abel PAC and Cain PAA share the stock quartet convention. Change
    filename selection only, never WRP coordinates or texture pixels.
    """
    return (
        ((world_mask & 1) << 3)
        | ((world_mask & 2) << 1)
        | ((world_mask & 4) >> 1)
        | ((world_mask & 8) >> 3)
    )


def stock_forest_blend_path(profile: str, quadrant_mask: int) -> str | None:
    """Return only a verified path whose visible corners match the WRP."""
    family = STOCK_FOREST_BLEND_FAMILIES.get(profile.casefold())
    if family is None:
        return None
    namespace, forest, grass, extension, valid_masks = family
    if profile.casefold() in STOCK_FOREST_BLEND_FAMILIES:
        quadrant_mask = _opposite_filename_mask(quadrant_mask)
    # Validate the FILENAME mask after the orientation mapping. A
    # half-turn preserves both opposed diagonal masks 6 and 9.
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


def _consistent_stock_masks(
    forest: np.ndarray, editable: np.ndarray, *,
    missing_masks: frozenset[int] = frozenset(),
) -> np.ndarray:
    """Derive all forest corners once, shared by four adjacent WRP cells.

    A protected water, farmland, road or urban cell must retain its source
    ground texture. Its corners must therefore remain nonforest, even when
    neighbouring forest cells would otherwise draw forest into it. Everon
    lacks diagonal 9, and Abel lacks both diagonals 6 and 9. Each missing
    case must be resolved at a SHARED vertex, not independently per cell,
    so that all four neighbouring tiles remain corner-consistent.
    """
    original = _quadrants(forest)
    height, width = forest.shape
    vertices = np.zeros((height + 1, width + 1), dtype=np.bool_)
    vertices[1:, 1:] |= (original & 1) != 0   # NE
    vertices[1:, :-1] |= (original & 2) != 0  # NW
    vertices[:-1, 1:] |= (original & 4) != 0  # SE
    vertices[:-1, :-1] |= (original & 8) != 0 # SW

    protected = ~editable
    blocked = np.zeros_like(vertices)
    blocked[1:, 1:] |= protected
    blocked[1:, :-1] |= protected
    blocked[:-1, 1:] |= protected
    blocked[:-1, :-1] |= protected
    vertices &= ~blocked

    while True:
        masks = (
            vertices[1:, 1:].astype(np.uint8)
            | (vertices[1:, :-1].astype(np.uint8) << 1)
            | (vertices[:-1, 1:].astype(np.uint8) << 2)
            | (vertices[:-1, :-1].astype(np.uint8) << 3)
        )
        # Do not alter individual tile masks: remove a shared vertex,
        # then regenerate every adjacent cell's corners together.
        missing = np.isin(masks, tuple(missing_masks))
        if not np.any(missing):
            return masks
        if not missing_masks.issubset({6, 9}):
            raise ValueError("unsupported missing stock forest corner masks")
        if 6 in missing_masks:
            # Mask 6 = NW + SE. Remove its SE shared vertex.
            row, col = np.nonzero(masks == 6)
            vertices[row, col + 1] = False
        if 9 in missing_masks:
            # Mask 9 = NE + SW. Remove its SW shared vertex.
            row, col = np.nonzero(masks == 9)
            vertices[row, col] = False


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
    with the pre-existing material+1 index mapping. With the option on, forest
    and edge cells may change; every supported stock-island preset can also
    use explicitly approved adjacent ordinary-grass materials for continuity.

    Forest polygon boundaries are taken from the OSM forest raster, not guessed
    from object locations. Verified quadrant paths are appended only if used.
    Missing stock diagonal assets fall back to a verified one-quadrant-nearest
    blend; never emit a guessed path. An exhausted WRP slot table falls back
    to the original semantic texture slot. All four stock presets can derive
    a shared forest-corner lattice across grass and forest tiles, suppressing
    forest corners shared with protected farmland, roads and water.
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
    current = slots.reshape(height, width)
    profile_key = profile.casefold()
    # All original stock islands require consistent corners across forest
    # and ordinary grass. The filename transform is independent of these
    # geographic masks. Do not paint protected semantic materials.
    use_shared_corners = bool(blendable_exterior_materials)
    if use_shared_corners:
        editable_exterior = (~candidate) & np.isin(
            material, tuple(blendable_exterior_materials)
        )
        editable = candidate | editable_exterior
        missing_masks = frozenset(
            mask for mask in range(1, 15)
            if stock_forest_blend_path(profile_key, mask) is None
        )
        masks = _consistent_stock_masks(
            forest, editable, missing_masks=missing_masks,
        )
        namespace, _forest_code, background, _ext, _possible = STOCK_FOREST_BLEND_FAMILIES[profile_key]
        backslash = chr(92)
        background_name = f"{namespace}{backslash}{background}.paa".casefold()

        def canonical(path: str) -> str:
            return path.replace("/", backslash).replace(backslash * 2, backslash).casefold()

        background_slot = next(
            (i for i, path in enumerate(paths) if canonical(path) == background_name),
            None,
        )
        if background_slot is None:
            # An incomplete palette cannot safely draw this transition.
            use_shared_corners = False
        else:
            # A forest-classified cell may have lost all forest corners beside
            # protected ground. Do not leave its original solid forest tile.
            current[candidate & (masks == 0)] = background_slot
            # The corresponding fully forested tiles use the stock forest floor.
            current[editable & (masks == 15)] = forest_material + 1
            requested = editable & (masks > 0) & (masks < 15)

    if not use_shared_corners:
        masks = _quadrants(forest)
        if forest_edge_material is not None:
            full_edge = candidate & (material == forest_edge_material) & (masks == 15)
            current[full_edge] = forest_material + 1
        requested = candidate & (masks > 0) & (masks < 15)
    if not np.any(requested):
        return tuple(paths), slots.tolist()

    # Valid filename masks may differ from geographic masks when a stock
    # artwork family uses a 180-degree filename transform.
    possible = frozenset(
        mask for mask in range(1, 15)
        if stock_forest_blend_path(profile, mask) is not None
    )
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
        current[requested & (masks == mask)] = slot

    return tuple(paths), slots.tolist()
