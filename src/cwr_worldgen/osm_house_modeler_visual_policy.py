# SPDX-License-Identifier: GPL-3.0-or-later
"""Final visual policy for modeler-backed facade colours and enterable windows.

Installed after the country material policy so weighted country colours remain
source-of-truth while late visual/cache fixes see the final geometry stack.
"""
from __future__ import annotations

import base64
from dataclasses import replace
import json
import threading

from PIL import ImageOps

_INSTALLED = False
_ORIGINAL_RESOLVE_STYLE = None
_ORIGINAL_REUSE_CANDIDATES = None
_ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE = None
_ORIGINAL_CACHE_KEY = None
_ORIGINAL_APPEND_ROOF_STOREY_WINDOWS = None
_ORIGINAL_ADD_GABLE_WINDOW = None
_ORIGINAL_ADD_WINDOW_CROSSES = None
_ORIGINAL_POLYGON_VISUAL_LOD = None
_GABLE_TRIM_STATE = threading.local()

_BUILDING_MODEL_CACHE_NAMES = frozenset({
    "procedural-building-model-v49-robust-polygon-roof-triangulation",
    "procedural-building-model-v50-foundation-skin-offset",
    "procedural-building-model-v51-modeler-opening-dimensions",
    "procedural-building-model-v52-no-porch-geometry",
    "procedural-building-model-v53-single-chimney",
})
_BUILDING_MODEL_CACHE_V54 = "procedural-building-model-v54-window-glass-light-trim"
_WINDOW_TRIM_CACHE_V1 = "procedural-building-window-frame-modeler-v1-cwa84"
_WINDOW_TRIM_CACHE_V2 = "procedural-building-window-frame-modeler-v2-neutral-light-cwa84"


def _normalised_text(value: object) -> str:
    return str(value or "").strip().casefold()


def _resolved_style(*args, **kwargs):
    choice = _ORIGINAL_RESOLVE_STYLE(*args, **kwargs)
    if _normalised_text(getattr(choice, "country_profile_identifier", "")) != "se_sweden":
        return choice
    window = dict(getattr(choice, "window_spec", {}) or {})
    frame = _normalised_text(window.get("frame_material", ""))
    changed = False
    if frame in {"painted timber", "painted wood"}:
        window["frame_material"] = "white-painted timber"
        changed = True
    if _normalised_text(window.get("trim", "")) != "white":
        window["trim"] = "white"
        changed = True
    return replace(choice, window_spec=window) if changed else choice


def _primary_colour(key: object) -> str:
    palette = getattr(key, "colour_palette", ()) or ()
    return next((_normalised_text(v) for v in palette if str(v).strip()), "")


def _facade_appearance_signature(key: object) -> tuple[str, str, str, str]:
    return (
        _normalised_text(getattr(key, "country_style_identifier", "")),
        _normalised_text(getattr(key, "regional_style", "")),
        _normalised_text(getattr(key, "wall_material", "")),
        _primary_colour(key),
    )


def _reuse_candidates(self, requested, candidates):
    pool = list(_ORIGINAL_REUSE_CANDIDATES(self, requested, candidates))
    if len(pool) < 2:
        return pool
    if not any((
        getattr(requested, "country_style_identifier", ""),
        getattr(requested, "wall_material", ""),
        getattr(requested, "colour_palette", ()),
    )):
        return pool
    strict = [c for c in pool if self._variant_within_fit_envelope(requested, c)]
    if not strict:
        return pool
    signature = _facade_appearance_signature(requested)
    matching = [c for c in strict if _facade_appearance_signature(c) == signature]
    return matching or pool


def _light_window_style_token() -> str:
    metadata = {"window": {"frame_material": "white", "trim": "white"}}
    encoded = base64.urlsafe_b64encode(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"default|~{encoded}|white"


_LIGHT_WINDOW_STYLE_TOKEN = _light_window_style_token()


def _window_frame_texture_image(size=128, regional_style="default", texture_variant=0):
    style = str(regional_style or "default")
    if style.strip().casefold() in {"", "default"}:
        style = _LIGHT_WINDOW_STYLE_TOKEN
    image = _ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE(size, style, texture_variant).convert("RGB")
    if style == _LIGHT_WINDOW_STYLE_TOKEN:
        image = ImageOps.grayscale(image).convert("RGB")
    return image


def _cache_key(namespace, *args, **kwargs):
    if namespace in _BUILDING_MODEL_CACHE_NAMES:
        namespace = _BUILDING_MODEL_CACHE_V54
    elif namespace == _WINDOW_TRIM_CACHE_V1:
        namespace = _WINDOW_TRIM_CACHE_V2
    return _ORIGINAL_CACHE_KEY(namespace, *args, **kwargs)


def _shared_light_trim_path(reference_texture: str) -> str | None:
    reference = str(reference_texture or "")
    if "\\" not in reference:
        return None
    world = reference.split("\\", 1)[0].strip()
    return rf"{world}\d\t.paa" if world else None


def _add_gable_window(*args, **kwargs):
    trim = getattr(_GABLE_TRIM_STATE, "texture", None)
    if trim:
        kwargs = dict(kwargs)
        kwargs["trim_texture"] = trim
    return _ORIGINAL_ADD_GABLE_WINDOW(*args, **kwargs)


def _append_roof_storey_windows(points, normals, faces, key, *, roof_pitch_degrees, reference_texture):
    from . import procedural_buildings as buildings
    previous = getattr(_GABLE_TRIM_STATE, "texture", None)
    trim = None
    if (
        bool(getattr(key, "interiors", False))
        and getattr(key, "family", "") not in buildings.UTILITY_INTERIOR_FAMILIES
        and buildings._uses_light_window_trim(key)
    ):
        trim = _shared_light_trim_path(reference_texture)
    _GABLE_TRIM_STATE.texture = trim
    try:
        return _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS(
            points, normals, faces, key,
            roof_pitch_degrees=roof_pitch_degrees,
            reference_texture=reference_texture,
        )
    finally:
        _GABLE_TRIM_STATE.texture = previous


def _glass_texture(reference_texture: str) -> str | None:
    reference = str(reference_texture or "")
    if "\\" not in reference:
        return None
    from .osm_house_modeler_fidelity import detail_texture_path
    return detail_texture_path(reference, "glass")


def _add_rectangular_window_glass(key, points, faces, *, wall_top, reference_texture):
    from . import procedural_buildings as buildings
    if not bool(getattr(key, "interiors", False)):
        return points, faces
    glass = _glass_texture(reference_texture)
    if not glass:
        return points, faces
    hw = float(key.width_m) * 0.5
    hl = float(key.length_m) * 0.5
    door_half, _door_height, _pivot = buildings._door_dimensions(key)
    front = buildings._interior_window_openings(
        key, -hw, hw, wall_top,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    )
    back = buildings._interior_window_openings(key, -hw, hw, wall_top)
    sides = buildings._interior_window_openings(key, -hl, hl, wall_top)
    inset = 0.015

    def pane(h0, h1, y0, y1, plane, axis, normal):
        nonlocal points, faces
        h0 += inset; h1 -= inset; y0 += inset; y1 -= inset
        if h1 <= h0 or y1 <= y0:
            return
        start = len(points)
        if axis == "x":
            points = points + ((h0, y0, plane), (h0, y1, plane), (h1, y1, plane), (h1, y0, plane))
        else:
            points = points + ((plane, y0, h0), (plane, y1, h0), (plane, y1, h1), (plane, y0, h1))
        face = buildings._Face(glass, (
            (start + 0, normal, 0.0, 1.0), (start + 1, normal, 0.0, 0.0),
            (start + 2, normal, 1.0, 0.0), (start + 3, normal, 1.0, 1.0),
        ))
        faces = faces + (face, buildings._reverse_face(face))

    for axis, plane, openings, normal in (
        ("x", -hl - 0.008, front, 0),
        ("z", hw + 0.008, sides, 1),
        ("x", hl + 0.008, back, 2),
        ("z", -hw - 0.008, sides, 3),
    ):
        for opening in openings:
            pane(*opening, plane, axis, normal)
    return points, faces


def _add_window_crosses(key, points, faces, *, wall_top, texture):
    points, faces = _ORIGINAL_ADD_WINDOW_CROSSES(
        key, points, faces, wall_top=wall_top, texture=texture
    )
    return _add_rectangular_window_glass(
        key, points, faces, wall_top=wall_top, reference_texture=texture
    )


def _polygon_visual_lod(*args, **kwargs):
    from . import procedural_buildings as buildings
    lod = _ORIGINAL_POLYGON_VISUAL_LOD(*args, **kwargs)
    key = args[0] if args else kwargs.get("key")
    if key is None or not bool(getattr(key, "interiors", False)):
        return lod
    reference = str(
        kwargs.get("window_trim_texture")
        or kwargs.get("wall_texture")
        or (args[1] if len(args) > 1 else "")
        or ""
    )
    glass = _glass_texture(reference)
    if not glass:
        return lod
    pitch = float(kwargs.get("roof_pitch_degrees", 35.0) or 35.0)
    eave, _triangles, _roof_height = buildings._polygon_native_roof_mesh(key, pitch)
    outer = tuple(getattr(key, "footprint_vertices", ()) or ())
    holes = tuple(tuple(r) for r in (getattr(key, "footprint_holes", ()) or ()))
    if len(outer) < 3:
        return lod
    points = list(lod.points); normals = list(lod.normals); faces = list(lod.faces)
    added = False
    for ring_index, ring in enumerate((outer, *holes)):
        for edge_index in range(len(ring)):
            start = ring[edge_index]; end = ring[(edge_index + 1) % len(ring)]
            span, tx, tz, ix, iz = buildings._polygon_native_edge_frame(start, end)
            if span <= 1.0e-6:
                continue
            openings = buildings._polygon_native_edge_openings(
                key, edge_index, span, eave, courtyard=ring_index > 0
            )
            for h0, h1, y0, y1 in openings:
                if y0 <= 0.05 or y1 - y0 < 0.30 or h1 - h0 < 0.30:
                    continue
                h0 += 0.015; h1 -= 0.015; y0 += 0.015; y1 -= 0.015
                if h1 <= h0 or y1 <= y0:
                    continue
                def point(h, y):
                    outward = -0.010
                    return (float(start[0]) + tx * h + ix * outward, y,
                            float(start[1]) + tz * h + iz * outward)
                base = len(points)
                points.extend((point(h0, y0), point(h0, y1), point(h1, y1), point(h1, y0)))
                ni = len(normals); normals.append((-ix, 0.0, -iz))
                face = buildings._Face(glass, (
                    (base + 0, ni, 0.0, 1.0), (base + 1, ni, 0.0, 0.0),
                    (base + 2, ni, 1.0, 0.0), (base + 3, ni, 1.0, 1.0),
                ))
                faces.extend((face, buildings._reverse_face(face)))
                added = True
    if not added:
        return lod
    selections = []
    for selection in lod.selections:
        selections.append(replace(
            selection,
            point_weights=selection.point_weights + bytes(max(0, len(points) - len(selection.point_weights))),
            face_flags=selection.face_flags + bytes(max(0, len(faces) - len(selection.face_flags))),
        ))
    return replace(lod, points=tuple(points), normals=tuple(normals),
                   faces=tuple(faces), selections=tuple(selections))


def install_osm_house_modeler_visual_policy() -> None:
    global _INSTALLED
    global _ORIGINAL_RESOLVE_STYLE, _ORIGINAL_REUSE_CANDIDATES
    global _ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE, _ORIGINAL_CACHE_KEY
    global _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS, _ORIGINAL_ADD_GABLE_WINDOW
    global _ORIGINAL_ADD_WINDOW_CROSSES, _ORIGINAL_POLYGON_VISUAL_LOD
    if _INSTALLED:
        return
    from . import osm_house_modeler_runtime as runtime
    from . import osm_house_modeler_texture_bridge as bridge
    from . import osm_house_modeler_upgrade as upgrade
    from . import procedural_buildings as buildings

    _ORIGINAL_RESOLVE_STYLE = runtime.resolve_style
    runtime.resolve_style = _resolved_style
    _ORIGINAL_REUSE_CANDIDATES = buildings.ProceduralBuildingLibrary._reuse_candidates
    buildings.ProceduralBuildingLibrary._reuse_candidates = _reuse_candidates
    _ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE = bridge.modeler_window_frame_texture_image
    bridge.modeler_window_frame_texture_image = _window_frame_texture_image
    _ORIGINAL_CACHE_KEY = buildings.cache_key
    buildings.cache_key = _cache_key
    _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS = upgrade._append_roof_storey_windows
    _ORIGINAL_ADD_GABLE_WINDOW = upgrade._add_gable_window
    upgrade._append_roof_storey_windows = _append_roof_storey_windows
    upgrade._add_gable_window = _add_gable_window
    _ORIGINAL_ADD_WINDOW_CROSSES = buildings._add_window_crosses
    buildings._add_window_crosses = _add_window_crosses
    _ORIGINAL_POLYGON_VISUAL_LOD = buildings._polygon_native_visual_lod
    buildings._polygon_native_visual_lod = _polygon_visual_lod
    _INSTALLED = True
