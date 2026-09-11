# SPDX-License-Identifier: GPL-3.0-or-later
"""Render dirt-like OSM parking surfaces with the gravel parking treatment."""
from __future__ import annotations

from dataclasses import replace


_INSTALLED = False

# CWA has no useful procedural "dirt parking lot" family here. These OSM
# surfaces all describe unsealed lots, so normalize them to the existing gravel
# renderer instead of allowing a nearby paved road to turn the lot into asphalt.
DIRT_PARKING_SURFACES = frozenset({
    "dirt",
    "earth",
    "ground",
    "mud",
    "sand",
    "unpaved",
    "compacted",
    "gravel",
    "fine_gravel",
    "pebblestone",
})


def is_dirt_parking_surface(tags) -> bool:
    """Return whether an OSM parking surface should use the gravel renderer."""
    raw = str((tags or {}).get("surface", "") or "").strip().casefold()
    if not raw:
        return False
    # OSM occasionally contains semicolon-separated values. Treat the lot as
    # unsealed when any declared surface is dirt/gravel-like.
    values = {
        value.strip()
        for value in raw.replace(",", ";").split(";")
        if value.strip()
    }
    return bool(values & DIRT_PARKING_SURFACES)


def _dirt_parking_keys(dataset) -> set[str]:
    keys: set[str] = set()
    for feature in getattr(dataset, "sites", ()):
        tags = getattr(feature, "tags", {}) or {}
        if str(tags.get("site", "")).strip().casefold() != "parking":
            continue
        if is_dirt_parking_surface(tags):
            keys.add(str(getattr(feature, "osm_key", "")))
    return keys


def force_dirt_parking_to_gravel(geometries, dataset):
    """Replace nearest-road asphalt classification for explicitly dirt lots."""
    dirt_keys = _dirt_parking_keys(dataset)
    if not dirt_keys:
        return tuple(geometries)
    return tuple(
        replace(geometry, surface="gravel")
        if str(getattr(geometry, "osm_key", "")) in dirt_keys
        and str(getattr(geometry, "surface", "")) != "gravel"
        else geometry
        for geometry in geometries
    )


def install_parking_dirt_surface_policy() -> None:
    """Make explicit dirt-like parking tags override nearest-road paving."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import parking_surface_policy as parking

    original_parking_geometries = parking._parking_geometries

    def parking_geometries_with_dirt_override(dataset, projection):
        geometries = original_parking_geometries(dataset, projection)
        return force_dirt_parking_to_gravel(geometries, dataset)

    parking._parking_geometries = parking_geometries_with_dirt_override
    _INSTALLED = True
