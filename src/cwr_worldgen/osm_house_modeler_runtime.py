# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime fixes for OSM House Modeler facade reuse and light window trim.

The full adapter implementation stays in ``osm_house_modeler_runtime_base``.
This thin layer keeps the existing integration intact while applying regression
fixes that depend on the detailed modeler style fields added by the upgrade.
"""
from __future__ import annotations

import base64
import json
import threading

from . import osm_house_modeler_runtime_base as _base
from . import osm_house_modeler_texture_bridge as _texture_bridge
from . import osm_house_modeler_upgrade as _upgrade
from . import procedural_buildings as _pb


_ORIGINAL_REUSE_CANDIDATES = None
_ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE = None
_ORIGINAL_CACHE_KEY = None
_ORIGINAL_APPEND_ROOF_STOREY_WINDOWS = None
_ORIGINAL_ADD_GABLE_WINDOW = None
_GABLE_TRIM_STATE = threading.local()


def __getattr__(name: str):
    """Keep private compatibility imports working through the runtime shim."""
    return getattr(_base, name)


def _normalised_text(value: object) -> str:
    return str(value or "").strip().casefold()


def _primary_colour(key: object) -> str:
    palette = getattr(key, "colour_palette", ()) or ()
    return next(
        (_normalised_text(value) for value in palette if str(value).strip()),
        "",
    )


def _facade_appearance_signature(key: object) -> tuple[str, str, str, str]:
    """Return the modeler fields that materially control the wall appearance."""
    return (
        _normalised_text(getattr(key, "country_style_identifier", "")),
        _normalised_text(getattr(key, "regional_style", "")),
        _normalised_text(getattr(key, "wall_material", "")),
        _primary_colour(key),
    )


def _reuse_candidates(self, requested, candidates):
    """Keep facade colour/material when several physically valid models exist.

    CWR's base selector deliberately prioritises footprint fit over cosmetics.
    Preserve that rule: only narrow the already-selected pool when at least one
    candidate remains inside the normal reuse envelope.
    """
    pool = list(_ORIGINAL_REUSE_CANDIDATES(self, requested, candidates))
    if len(pool) < 2:
        return pool

    requested_signature = _facade_appearance_signature(requested)
    # Legacy/manual keys do not carry the detailed modeler style signature.
    if not any(
        (
            getattr(requested, "country_style_identifier", ""),
            getattr(requested, "wall_material", ""),
            getattr(requested, "colour_palette", ()),
        )
    ):
        return pool

    strict = [
        candidate
        for candidate in pool
        if self._variant_within_fit_envelope(requested, candidate)
    ]
    if not strict:
        return pool

    matching = [
        candidate
        for candidate in strict
        if _facade_appearance_signature(candidate) == requested_signature
    ]
    return matching or pool


def _light_window_style_token() -> str:
    """Build metadata for the shared neutral/light window casing texture."""
    metadata = {
        "window": {
            "frame_material": "white",
            "trim": "white",
        }
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"default|~{encoded}|white"


_LIGHT_WINDOW_STYLE_TOKEN = _light_window_style_token()


def _window_frame_texture_image(
    size: int = 128,
    regional_style: str = "default",
    texture_variant: int = 0,
):
    """Make the shared trim atlas actually light instead of reddish timber."""
    style = str(regional_style or "default")
    if style.strip().casefold() in {"", "default"}:
        style = _LIGHT_WINDOW_STYLE_TOKEN
    return _ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE(
        size,
        style,
        texture_variant,
    )


def _cache_key(namespace, *args, **kwargs):
    """Invalidate cached reddish trim atlases from the previous renderer."""
    if namespace == "procedural-building-window-frame-modeler-v1-cwa84":
        namespace = "procedural-building-window-frame-modeler-v2-light-trim-cwa84"
    return _ORIGINAL_CACHE_KEY(namespace, *args, **kwargs)


def _shared_light_trim_path(reference_texture: str) -> str | None:
    reference = str(reference_texture or "")
    if "\\" not in reference:
        return None
    world_prefix = reference.split("\\", 1)[0].strip()
    if not world_prefix:
        return None
    return rf"{world_prefix}\d\t.paa"


def _add_gable_window(*args, **kwargs):
    trim_texture = getattr(_GABLE_TRIM_STATE, "texture", None)
    if trim_texture:
        kwargs = dict(kwargs)
        kwargs["trim_texture"] = trim_texture
    return _ORIGINAL_ADD_GABLE_WINDOW(*args, **kwargs)


def _append_roof_storey_windows(
    points,
    normals,
    faces,
    key,
    *,
    roof_pitch_degrees: float,
    reference_texture: str,
):
    """Use the same light casing on roof-storey windows as normal windows."""
    previous = getattr(_GABLE_TRIM_STATE, "texture", None)
    trim_texture = None
    if _pb._uses_light_window_trim(key):
        trim_texture = _shared_light_trim_path(reference_texture)
    _GABLE_TRIM_STATE.texture = trim_texture
    try:
        return _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS(
            points,
            normals,
            faces,
            key,
            roof_pitch_degrees=roof_pitch_degrees,
            reference_texture=reference_texture,
        )
    finally:
        _GABLE_TRIM_STATE.texture = previous


def _install_facade_reuse_fix() -> None:
    global _ORIGINAL_REUSE_CANDIDATES
    current = _pb.ProceduralBuildingLibrary._reuse_candidates
    if getattr(current, "_cwr_modeler_facade_reuse_fix", False):
        return
    _ORIGINAL_REUSE_CANDIDATES = current
    _reuse_candidates._cwr_modeler_facade_reuse_fix = True
    _pb.ProceduralBuildingLibrary._reuse_candidates = _reuse_candidates


def _install_light_window_texture_fix() -> None:
    global _ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE, _ORIGINAL_CACHE_KEY

    current_texture = _texture_bridge.modeler_window_frame_texture_image
    if not getattr(current_texture, "_cwr_light_window_trim_fix", False):
        _ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE = current_texture
        _window_frame_texture_image._cwr_light_window_trim_fix = True
        _texture_bridge.modeler_window_frame_texture_image = _window_frame_texture_image
        if hasattr(_pb, "modeler_window_frame_texture_image"):
            _pb.modeler_window_frame_texture_image = _window_frame_texture_image

    current_cache_key = _pb.cache_key
    if not getattr(current_cache_key, "_cwr_light_window_trim_cache_v2", False):
        _ORIGINAL_CACHE_KEY = current_cache_key
        _cache_key._cwr_light_window_trim_cache_v2 = True
        _pb.cache_key = _cache_key


def _install_roof_storey_trim_fix() -> None:
    global _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS, _ORIGINAL_ADD_GABLE_WINDOW

    current_append = _upgrade._append_roof_storey_windows
    if getattr(current_append, "_cwr_light_roof_storey_trim_fix", False):
        return

    _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS = current_append
    _ORIGINAL_ADD_GABLE_WINDOW = _upgrade._add_gable_window
    _append_roof_storey_windows._cwr_light_roof_storey_trim_fix = True
    _add_gable_window._cwr_light_roof_storey_trim_fix = True
    _upgrade._add_gable_window = _add_gable_window
    _upgrade._append_roof_storey_windows = _append_roof_storey_windows


def install_osm_house_modeler_upgrade() -> None:
    """Install the modeler adapter plus Sweden/interior appearance regressions."""
    _base.install_osm_house_modeler_upgrade()
    _install_facade_reuse_fix()
    _install_light_window_texture_fix()
    _install_roof_storey_trim_fix()


__all__ = ["install_osm_house_modeler_upgrade"]
