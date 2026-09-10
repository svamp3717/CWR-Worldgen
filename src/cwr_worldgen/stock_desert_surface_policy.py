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
    "w": _DESERT_SAND_TEXTURE,
    "q": _DESERT_SAND_TEXTURE,
    "s": _DESERT_SAND_TEXTURE,
    "g": _DESERT_SAND_TEXTURE,
    "h": _DESERT_SAND_TEXTURE,
    "r": r"o\l1.paa",
    "k": r"o\lom2.paa",
    "f": _DESERT_SAND_TEXTURE,
    "e": _DESERT_SAND_TEXTURE,
    "a": _DESERT_SAND_TEXTURE,
    "b": _DESERT_SAND_TEXTURE,
    "c": _DESERT_SAND_TEXTURE,
    "u": _DESERT_SAND_TEXTURE,
    "i": _DESERT_SAND_TEXTURE,
    "p": _DESERT_SAND_TEXTURE,
    "o": _DESERT_SAND_TEXTURE,
    "d": _DESERT_SAND_TEXTURE,
    "t": _DESERT_SAND_TEXTURE,
    "v": _DESERT_SAND_TEXTURE,
    "j": _DESERT_SAND_TEXTURE,
    "y": _DESERT_SAND_TEXTURE,
    "x": _DESERT_SAND_TEXTURE,
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
            raise ValueError(f"unknown Desert terrain material code: {material_code!r}") from exc
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

    stock_profiles = dict(_surface.STOCK_SURFACE_TEXTURES)
    stock_profiles["desert"] = dict(DESERT_STOCK_SURFACE_TEXTURES)
    _surface.STOCK_SURFACE_TEXTURES = stock_profiles

    _generator._ground_texture_profile = _stock_desert_ground_texture_profile
    _generator._ground_texture_paths = _stock_desert_ground_texture_paths
    _generator._external_ground_texture_paths = _stock_desert_external_ground_texture_paths

    _terrain.ground_texture_path = _stock_desert_terrain_texture_path
    _generator.ground_texture_path = _stock_desert_terrain_texture_path

    _INSTALLED = True

    # Runways prefer generated per-cell WRP textures so arbitrary OSM bearings
    # remain correctly oriented and visible in the editor terrain view. Install
    # after Desert owns the final base palette; the runway policy then generates
    # Desert-coloured cell backgrounds and uses P3D tiles only on table overflow.
    from .runway_surface_policy import install_runway_surface_policy

    install_runway_surface_policy()

    # The first path-aware Nogova approximation was still visibly too bright in
    # CWA. Keep that calibration as the fallback for builds where stock PAA bytes
    # cannot be read from local files or the game installation.
    from .runway_nogova_calibration_policy import (
        install_runway_nogova_calibration_policy,
    )

    install_runway_nogova_calibration_policy()

    # Prefer the exact terrain texture for every preset, not only Nogova. Local
    # generated/Malden PAAs are read from the world source tree; stock Nogova,
    # Everon and Desert PAAs are resolved from asset roots, CWR_GAME_ROOT, or the
    # parent of the deployment @mod folder. The calibrated/profile colours above
    # remain a bounded fallback when the source texture is unavailable.
    from .runway_exact_background_policy import install_runway_exact_background_policy

    install_runway_exact_background_policy()

    # Rendering/compression is expensive and does not need to be repeated on
    # unchanged builds. Persist three role textures per runway and restore
    # them on later builds from the dedicated runway-ground-textures cache.
    from .runway_texture_cache_policy import install_runway_texture_cache_policy

    install_runway_texture_cache_policy()

    # Remember one game installation in the desktop GUI and automatically feed
    # it into --asset-root on every build. This lets all runway presets resolve
    # their original stock background textures after application restarts.
    from .game_folder_gui_policy import install_game_folder_gui_policy

    install_game_folder_gui_policy()

    # Building validation only needs the selected game assets and the texture
    # dependencies those models actually reference. Avoid recursively cataloguing
    # the entire CWA installation at 82/83% and cache PBO header indexes instead.
    from .fast_asset_scan_policy import install_fast_asset_scan_policy

    install_fast_asset_scan_policy()
