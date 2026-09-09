# SPDX-License-Identifier: GPL-3.0-or-later
"""Use verified in-game terrain artwork for the Desert ground profile.

Desert used to synthesize every Milestone 9 semantic ground tile as a world-local
PAA. Apart from making each new desert world spend minutes in Python DXT1
compression, that duplicated artwork the game already ships. This policy maps
every Desert material to existing CWA/Resistance terrain textures and makes the
core generator treat Desert as an external/stock texture profile.

The generator's historical local-texture predicate is hard-coded around the two
original stock profiles (Everon and Nogova). ``_StockDesertProfile`` is a narrow
compatibility bridge for that predicate: it serializes and displays as ``desert``
but compares as the already-stock Everon profile when the old membership check is
performed. Ground-path helpers are replaced explicitly, so the WRP still receives
the Desert-specific stock palette below rather than Everon's palette.
"""
from __future__ import annotations

from typing import Mapping

from . import generator as _generator
from . import surface_pass as _surface
from . import terrain as _terrain


# Desert semantics still distinguish forests, fields, settlements and roads for
# object placement, terrain grading, reporting and overview generation. They do
# not need different WRP ground artwork. Road P3Ds provide the visible road deck
# and buildings provide the settlement geometry, so painting a second Eden earth
# tile beneath those masks merely creates coloured halos around roads and houses.
# Use one verified stock sand tile for every non-rock material and reserve only
# the stock rock/scree tiles for exposed stone.
_DESERT_SAND_TEXTURE = r"o\ps.paa"

DESERT_STOCK_SURFACE_TEXTURES: Mapping[str, str] = {
    "w": _DESERT_SAND_TEXTURE,      # seabed / water-adjacent ground
    "q": _DESERT_SAND_TEXTURE,      # wet shoreline semantics
    "s": _DESERT_SAND_TEXTURE,      # dry shoreline sand
    "g": _DESERT_SAND_TEXTURE,      # grass semantics, desert ground
    "h": _DESERT_SAND_TEXTURE,      # dry grass / bare desert ground
    "r": r"o\l1.paa",             # rock
    "k": r"o\lom2.paa",           # steep rock / scree
    "f": _DESERT_SAND_TEXTURE,      # forest interior beneath vegetation
    "e": _DESERT_SAND_TEXTURE,      # forest edge beneath vegetation
    "a": _DESERT_SAND_TEXTURE,      # farmland light
    "b": _DESERT_SAND_TEXTURE,      # farmland dark
    "c": _DESERT_SAND_TEXTURE,      # field boundary / dry strip
    "u": _DESERT_SAND_TEXTURE,      # urban surface beneath buildings
    "i": _DESERT_SAND_TEXTURE,      # industrial surface beneath buildings
    "p": _DESERT_SAND_TEXTURE,      # paved-road underlay beneath road P3Ds
    "o": _DESERT_SAND_TEXTURE,      # road shoulder
    "d": _DESERT_SAND_TEXTURE,      # dirt road underlay
    "t": _DESERT_SAND_TEXTURE,      # dirt-road blend
    "v": _DESERT_SAND_TEXTURE,      # gravel underlay
    "j": _DESERT_SAND_TEXTURE,      # park semantics, desert ground
    "y": _DESERT_SAND_TEXTURE,      # sports field semantics, desert ground
    "x": _DESERT_SAND_TEXTURE,      # mapped beach
}

_INSTALLED = False
_ORIGINAL_GROUND_TEXTURE_PROFILE = None
_ORIGINAL_GROUND_TEXTURE_PATHS = None
_ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS = None
_ORIGINAL_TERRAIN_GROUND_TEXTURE_PATH = None


class _StockDesertProfile(str):
    """String-compatible Desert marker that satisfies the legacy stock test."""

    def __new__(cls):
        return super().__new__(cls, "desert")

    def __hash__(self) -> int:
        # The old generator checks membership in {"everon", "nogova"}. Sharing
        # Everon's hash lets that existing set probe reach our equality method.
        return hash("everon")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str(other).casefold() in {"desert", "everon"}
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


_STOCK_DESERT_PROFILE = _StockDesertProfile()


def _is_desert(value: object) -> bool:
    return str(value or "").strip().casefold() == "desert"


def _desert_material_paths(materials) -> tuple[str, ...]:
    paths: list[str] = []
    for material in materials:
        code = str(getattr(material, "code", ""))
        try:
            paths.append(DESERT_STOCK_SURFACE_TEXTURES[code])
        except KeyError as exc:
            raise ValueError(f"Desert stock terrain has no mapping for material {code!r}") from exc
    return tuple(paths)


def _stock_desert_ground_texture_profile(spec) -> str:
    profile = _ORIGINAL_GROUND_TEXTURE_PROFILE(spec)
    return _STOCK_DESERT_PROFILE if _is_desert(profile) else profile


def _stock_desert_ground_texture_paths(spec) -> tuple[str, ...]:
    if _is_desert(getattr(spec, "ground_texture_profile", "")):
        materials = (
            _surface.MILESTONE9_MATERIALS
            if _generator._surface_ground_enabled(spec)
            else _generator.OSM_MATERIALS
        )
        return _desert_material_paths(materials)
    return _ORIGINAL_GROUND_TEXTURE_PATHS(spec)


def _stock_desert_external_ground_texture_paths(spec) -> tuple[str, ...]:
    if _is_desert(getattr(spec, "ground_texture_profile", "")):
        # Asset scanning only needs each physical dependency once even though the
        # WRP semantic table intentionally reuses the sand texture many times.
        return tuple(dict.fromkeys(_stock_desert_ground_texture_paths(spec)))
    return _ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS(spec)


def _stock_desert_terrain_texture_path(
    world_name: str,
    material_code: str,
    profile: str = "generated",
) -> str:
    if _is_desert(profile):
        try:
            return DESERT_STOCK_SURFACE_TEXTURES[str(material_code)]
        except KeyError as exc:
            raise ValueError(f"unknown Desert terrain material code: {material_code}") from exc
    return _ORIGINAL_TERRAIN_GROUND_TEXTURE_PATH(world_name, material_code, profile)


def install_stock_desert_surface_policy() -> None:
    """Make Desert use only stock game ground textures in every build path."""
    global _INSTALLED
    global _ORIGINAL_GROUND_TEXTURE_PROFILE, _ORIGINAL_GROUND_TEXTURE_PATHS
    global _ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS, _ORIGINAL_TERRAIN_GROUND_TEXTURE_PATH
    if _INSTALLED:
        return

    _ORIGINAL_GROUND_TEXTURE_PROFILE = _generator._ground_texture_profile
    _ORIGINAL_GROUND_TEXTURE_PATHS = _generator._ground_texture_paths
    _ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS = _generator._external_ground_texture_paths
    _ORIGINAL_TERRAIN_GROUND_TEXTURE_PATH = _terrain.ground_texture_path

    # Direct surface-pass callers should see Desert as a stock profile too. The
    # original object is a dict despite its Mapping annotation; replace it rather
    # than mutating in place so tests/importers never observe a half-installed map.
    stock_profiles = dict(_surface.STOCK_SURFACE_TEXTURES)
    stock_profiles["desert"] = dict(DESERT_STOCK_SURFACE_TEXTURES)
    _surface.STOCK_SURFACE_TEXTURES = stock_profiles

    # Core Milestone 9 helpers need both the actual Desert paths and the old
    # generated/local-texture predicate suppressed. Keep the latter compatibility
    # entirely inside this policy instead of teaching every caller about it.
    _generator._ground_texture_profile = _stock_desert_ground_texture_profile
    _generator._ground_texture_paths = _stock_desert_ground_texture_paths
    _generator._external_ground_texture_paths = _stock_desert_external_ground_texture_paths

    # Milestone 8/direct terrain API callers also receive stock Desert paths.
    _terrain.ground_texture_path = _stock_desert_terrain_texture_path
    _generator.ground_texture_path = _stock_desert_terrain_texture_path

    _INSTALLED = True

    # Runways are a dedicated stock terrain material. Install this only after the
    # Desert wrapper owns the final ground-path helpers so the runway policy can
    # extend both ordinary and Desert stock tables without wrapper-order races.
    from .runway_surface_policy import install_runway_surface_policy

    install_runway_surface_policy()
