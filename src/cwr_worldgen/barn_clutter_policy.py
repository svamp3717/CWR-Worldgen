# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from . import generator as _generator
from . import osm as _osm

STOH_MODEL = r"data3d\Stoh.p3d"

_INSTALLED = False


def install_barn_clutter_policy() -> None:
    """Add the stock Stoh earth pile to barn-only settlement clutter."""
    global _INSTALLED
    if _INSTALLED:
        return

    barn_models = tuple(
        dict.fromkeys((*_osm.STOCK_SETTLEMENT_BARN_CLUTTER_MODELS, STOH_MODEL))
    )
    detail_models = tuple(
        dict.fromkeys(
            (
                *_osm.STOCK_SETTLEMENT_GENERIC_DETAIL_MODELS,
                *barn_models,
                *_osm.STOCK_SETTLEMENT_FRUIT_TREE_MODELS,
            )
        )
    )

    # OSM placement reads the barn-specific tuple directly. Generator keeps a
    # separate imported binding for strict asset validation/cache identity, so
    # keep both views synchronized.
    _osm.STOCK_SETTLEMENT_BARN_CLUTTER_MODELS = barn_models
    _osm.STOCK_SETTLEMENT_DETAIL_MODELS = detail_models
    _generator.STOCK_SETTLEMENT_DETAIL_MODELS = detail_models

    _INSTALLED = True
