# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep generated bridge modules on their authored WRP deck plane in OFP/CWA.

Procedural bridge modules are static terrain objects whose absolute Y/pitch is
already solved by the bridge planner. A LandContact LOD asks the engine to fit a
structure to the local terrain surface; once a long bridge is split into several
P3Ds, that can make each module settle independently onto banks or the seabed.
Bridges therefore omit LandContact entirely and identify themselves explicitly
as bridge/road objects in their first Resolution LOD.
"""
from __future__ import annotations

from dataclasses import replace

from . import procedural_infrastructure as _infra

_OLD_BRIDGE_MODEL_CACHE_NAMESPACE = (
    "procedural-infrastructure-model-v17-single-span-segmented-collision"
)
_NEW_BRIDGE_MODEL_CACHE_NAMESPACE = (
    "procedural-infrastructure-model-v18-fixed-deck-no-landcontact"
)

_ORIGINAL_BRIDGE_LODS = None
_ORIGINAL_CACHE_KEY = None
_INSTALLED = False


def _resolution_properties(properties):
    """Return deterministic bridge semantics without duplicate named properties."""

    kept = [
        (str(name), str(value))
        for name, value in tuple(properties or ())
        if str(name).casefold() not in {"autocenter", "class", "map"}
    ]
    kept.extend((
        ("autocenter", "0"),
        ("class", "bridge"),
        ("map", "road"),
    ))
    return tuple(kept)


def _bridge_lods_without_landcontact(key, texture):
    """Preserve bridge visuals/collision/Roadway, but never terrain-fit the P3D."""

    lods = tuple(_ORIGINAL_BRIDGE_LODS(key, texture))
    if not lods:
        return lods

    rewritten = []
    for lod in lods:
        # LandContact is useful for terrain-following structures, but a bridge
        # module must obey its explicit WRP transform so every piece shares the
        # same solved deck plane instead of settling independently to terrain.
        if float(lod.resolution) == float(_infra._LAND_CONTACT_LOD):
            continue
        if float(lod.resolution) == float(_infra._VISUAL_LOD):
            lod = replace(lod, properties=_resolution_properties(lod.properties))
        rewritten.append(lod)
    return tuple(rewritten)


def _bridge_cache_key(namespace, payload):
    """Invalidate only generated bridge P3Ds made with the old LandContact recipe."""

    if namespace == _OLD_BRIDGE_MODEL_CACHE_NAMESPACE:
        namespace = _NEW_BRIDGE_MODEL_CACHE_NAMESPACE
    return _ORIGINAL_CACHE_KEY(namespace, payload)


def install_bridge_grounding_policy() -> None:
    """Install bridge-specific MLOD semantics before infrastructure assets are written."""

    global _ORIGINAL_BRIDGE_LODS, _ORIGINAL_CACHE_KEY, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_BRIDGE_LODS = _infra._bridge_lods
    _ORIGINAL_CACHE_KEY = _infra.cache_key
    _infra._bridge_lods = _bridge_lods_without_landcontact
    _infra.cache_key = _bridge_cache_key
    _INSTALLED = True
