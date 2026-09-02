# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime wiring for the OSM House Modeler building upgrade.

This small layer keeps the adapter tolerant of both the positional calls used by
CWR's legacy rectangular P3D writer and the keyword-only polygon-native writer.
"""
from __future__ import annotations

from . import osm_house_modeler_upgrade as _upgrade
from . import procedural_buildings as _pb


def _argument(args, kwargs, name: str, position: int, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def _visual_lod(*args, **kwargs):
    original = _upgrade._ORIGINAL_VISUAL_LOD
    if original is None:
        raise RuntimeError("OSM House Modeler building adapter is not installed")
    depth = int(getattr(_upgrade._CALL_STATE, "depth", 0))
    if depth:
        return original(*args, **kwargs)
    _upgrade._CALL_STATE.depth = depth + 1
    try:
        lod = original(*args, **kwargs)
    finally:
        _upgrade._CALL_STATE.depth = depth

    key = args[0]
    wall_texture = args[1]
    roof_texture = args[2]
    roof_pitch = float(_argument(args, kwargs, "roof_pitch_degrees", 3, 35.0))
    foundation_texture = (
        _argument(args, kwargs, "foundation_texture", 5, None) or wall_texture
    )
    foundation_depth = float(
        _argument(args, kwargs, "foundation_depth", 6, 0.0) or 0.0
    )
    return _upgrade._append_details(
        lod,
        key,
        frame=_upgrade._front_frame_rectangular(key),
        wall_texture=wall_texture,
        roof_texture=roof_texture,
        foundation_texture=foundation_texture,
        roof_pitch_degrees=roof_pitch,
        foundation_depth=foundation_depth,
    )


def _polygon_visual_lod(*args, **kwargs):
    original = _upgrade._ORIGINAL_POLYGON_VISUAL_LOD
    if original is None:
        raise RuntimeError("OSM House Modeler polygon adapter is not installed")
    polygon_depth = int(getattr(_upgrade._CALL_STATE, "polygon_depth", 0))
    if polygon_depth:
        return original(*args, **kwargs)
    normal_depth = int(getattr(_upgrade._CALL_STATE, "depth", 0))
    _upgrade._CALL_STATE.polygon_depth = polygon_depth + 1
    _upgrade._CALL_STATE.depth = normal_depth + 1
    try:
        lod = original(*args, **kwargs)
    finally:
        _upgrade._CALL_STATE.polygon_depth = polygon_depth
        _upgrade._CALL_STATE.depth = normal_depth

    key = args[0]
    wall_texture = args[1]
    roof_texture = args[2]
    roof_pitch = float(kwargs.get("roof_pitch_degrees", 35.0) or 35.0)
    foundation_texture = kwargs.get("foundation_texture") or wall_texture
    foundation_depth = float(kwargs.get("foundation_depth", 0.0) or 0.0)
    return _upgrade._append_details(
        lod,
        key,
        frame=_upgrade._front_frame_polygon(key),
        wall_texture=wall_texture,
        roof_texture=roof_texture,
        foundation_texture=foundation_texture,
        roof_pitch_degrees=roof_pitch,
        foundation_depth=foundation_depth,
    )


def install_osm_house_modeler_upgrade() -> None:
    """Install the modeler adapter with CWR call-signature compatibility."""

    _upgrade.install_osm_house_modeler_upgrade()
    if getattr(_pb._visual_lod, "_cwr_osm_house_modeler_runtime", False):
        return
    _visual_lod._cwr_osm_house_modeler_runtime = True  # type: ignore[attr-defined]
    _polygon_visual_lod._cwr_osm_house_modeler_runtime = True  # type: ignore[attr-defined]
    _pb._visual_lod = _visual_lod
    if _upgrade._ORIGINAL_POLYGON_VISUAL_LOD is not None:
        _pb._polygon_native_visual_lod = _polygon_visual_lod
