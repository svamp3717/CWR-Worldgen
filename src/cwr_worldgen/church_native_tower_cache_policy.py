# SPDX-License-Identifier: GPL-3.0-or-later
"""Invalidate only church P3Ds after adding polygon-native tower geometry."""
from __future__ import annotations

from typing import Mapping

_BUILDING_MODEL_CACHE_V57 = "procedural-building-model-v57-native-church-towers"
_INSTALLED = False


def install_church_native_tower_cache_policy() -> None:
    """Promote incoming church model cache requests to the tower-aware revision."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import procedural_buildings as buildings

    original_cache_key = buildings.cache_key

    def revised_cache_key(namespace: str, payload):
        variant = payload.get("variant") if isinstance(payload, Mapping) else None
        family = variant.get("family") if isinstance(variant, Mapping) else None
        if (
            str(namespace).startswith("procedural-building-model-v")
            and str(family or "").casefold() == "church"
        ):
            namespace = _BUILDING_MODEL_CACHE_V57
        return original_cache_key(namespace, payload)

    buildings.cache_key = revised_cache_key
    _INSTALLED = True
