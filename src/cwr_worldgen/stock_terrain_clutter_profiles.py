# SPDX-License-Identifier: GPL-3.0-or-later
"""Eden, Abel and Cain small-terrain-clutter palettes from original WRP exports.

All model paths and integer weights come from object-models.csv in the
user-supplied everon_objects.zip, malden_objects.zip and
kolgujev_objects.zip. They represent *relative* low-vegetation model
frequencies, never permission to copy original world coordinates/density.
"""
from __future__ import annotations

from typing import Final

# Original source WRP: eden.wrp. 14 low shrub, grass, small-conifer models,
# 17,805 placements; omit trees, big shrubs, reeds, forest blocks, structures.
EDEN_CLUTTER_WEIGHTS: Final[tuple[tuple[str, int], ...]] = (
    (r"data3d\krovi3.p3d", 4106),
    (r"data3d\krovi2.p3d", 3016),
    (r"data3d\krovi4.p3d", 2315),
    (r"data3d\str_smrcicicek.p3d", 2013),
    (r"data3d\ker trs travy7.p3d", 1898),
    (r"data3d\ker trs travy kvetina.p3d", 1129),
    (r"data3d\ker trs travy.p3d", 688),
    (r"data3d\ker listnac.p3d", 475),
    (r"data3d\ker pichlavej.p3d", 462),
    (r"data3d\krovi_long.p3d", 457),
    (r"data3d\krovi.p3d", 440),
    (r"data3d\ker trs travy2.p3d", 378),
    (r"data3d\ker trs travy3.p3d", 296),
    (r"data3d\ker trs travy4.p3d", 132),
)

# Original abel.wrp: 10 small shrub, low foliage and grass varieties,
# 44,560 source placements. Avoid long-canopy individual-tree models.
ABEL_CLUTTER_WEIGHTS: Final[tuple[tuple[str, int], ...]] = (
    (r"data3d\ker trs travy.p3d", 9722),
    (r"data3d\str_fikovnik_ker.p3d", 6713),
    (r"data3d\ker listnac.p3d", 5621),
    (r"data3d\str_liskac.p3d", 4521),
    (r"data3d\krovi3.p3d", 4461),
    (r"data3d\krovi2.p3d", 3606),
    (r"data3d\krovi4.p3d", 3222),
    (r"data3d\str jalovec.p3d", 3135),
    (r"data3d\krovi.p3d", 2324),
    (r"data3d\krovi_long.p3d", 1235),
)

# Original cain.wrp: 8 conifer sapling, small bush and grass models,
# 27,423 source placements. Full spruce and forest polygons are separate.
CAIN_CLUTTER_WEIGHTS: Final[tuple[tuple[str, int], ...]] = (
    (r"data3d\str_smrcicicek.p3d", 8032),
    (r"data3d\ker trs travy.p3d", 7677),
    (r"data3d\krovi2.p3d", 5209),
    (r"data3d\krovi3.p3d", 4818),
    (r"data3d\krovi4.p3d", 1178),
    (r"data3d\ker deravej.p3d", 188),
    (r"data3d\ker pichlavej.p3d", 180),
    (r"data3d\ker listnac.p3d", 141),
)

STOCK_CLUTTER_WEIGHTS: Final[dict[str, tuple[tuple[str, int], ...]]] = {
    "everon": EDEN_CLUTTER_WEIGHTS,
    "malden": ABEL_CLUTTER_WEIGHTS,
    "kolgujev": CAIN_CLUTTER_WEIGHTS,
}

# Require both authoritative forest-block AND solo tree model from the
# selected tree preset, not matching ground texture or forest_profile alone.
_TREE_PRESET_MODELS: Final[dict[str, tuple[str, str]]] = {
    "everon": (
        r"data3d\les ctverec pruchozi_t1.p3d",
        r"data3d\str smrk_medium.p3d",
    ),
    "malden": (
        r"data3d\les_su_ctver_pruchozi.p3d",
        r"data3d\str_fikovnik.p3d",
    ),
    "kolgujev": (
        r"data3d\les ctverec pruchozi_t1.p3d",
        r"data3d\str smrk.p3d",
    ),
}


def stock_clutter_tree_preset(spec: object) -> str | None:
    """Identify the original tree preset, independent of terrain textures."""
    block = str(getattr(spec, "forest_tree_model", "")).replace("/", "\\").casefold()
    single = str(getattr(spec, "forest_single_tree_model", "")).replace("/", "\\").casefold()
    for name, paths in _TREE_PRESET_MODELS.items():
        if (block, single) == paths:
            return name
    return None
