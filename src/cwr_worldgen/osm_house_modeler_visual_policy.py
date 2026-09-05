# SPDX-License-Identifier: GPL-3.0-or-later
"""Final visual policy for modeler-backed facade colours and enterable windows.

Installed after the country material policy so weighted country colours remain
source-of-truth while late visual/cache fixes see the final geometry stack.
"""
from __future__ import annotations

import base64
from dataclasses import replace
from hashlib import sha256
import json
import math
import threading

from PIL import ImageOps

_INSTALLED = False
_ORIGINAL_RESOLVE_STYLE = None
_ORIGINAL_REUSE_CANDIDATES = None
_ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE = None
_ORIGINAL_CACHE_KEY = None
_ORIGINAL_APPEND_ROOF_STOREY_WINDOWS = None
_ORIGINAL_ADD_GABLE_WINDOW = None
_GABLE_WINDOW_STATE = threading.local()

_BUILDING_MODEL_CACHE_NAMES = frozenset({
    "procedural-building-model-v49-robust-polygon-roof-triangulation",
    "procedural-building-model-v50-foundation-skin-offset",
    "procedural-building-model-v51-modeler-opening-dimensions",
    "procedural-building-model-v52-no-porch-geometry",
    "procedural-building-model-v53-single-chimney",
    "procedural-building-model-v54-window-glass-light-trim",
    "procedural-building-model-v55-enterable-open-windows",
})
_BUILDING_MODEL_CACHE_V56 = "procedural-building-model-v56-enterable-gable-openings"
_WINDOW_TRIM_CACHE_V1 = "procedural-building-window-frame-modeler-v1-cwa84"
_WINDOW_TRIM_CACHE_V2 = "procedural-building-window-frame-modeler-v2-neutral-light-cwa84"


def _normalised_text(value: object) -> str:
    return str(value or "").strip().casefold()


def _stable_unit(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts)
    digest = sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _resolved_style(*args, **kwargs):
    """Apply the small Sweden-only presentation tuning after country selection.

    Country/material selection itself remains data-driven and generic. Sweden's
    white-painted timber frame convention is normalized here because generic
    ``painted timber`` otherwise resolves to the modeler's brown/red wood colour.

    Rural Swedish painted-timber houses also move half of the procedurally chosen
    *white* facades to Falun red. That raises red cottages only a few percentage
    points overall while leaving mapped building:colour values authoritative.
    """
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

    palette = tuple(
        str(value) for value in (getattr(choice, "colour_palette", ()) or ())
        if str(value).strip()
    )
    tags = kwargs.get("tags") or {}
    explicit_colour = ""
    if hasattr(tags, "get"):
        explicit_colour = str(
            tags.get("building:colour")
            or tags.get("building:color")
            or ""
        ).strip()
    timber = _normalised_text(getattr(choice, "wall_material", ""))
    primary = _normalised_text(palette[0] if palette else "")
    rural_residential = (
        _normalised_text(getattr(choice, "context", "")) == "rural"
        and _normalised_text(getattr(choice, "family", "")) == "residential"
    )
    if (
        not explicit_colour
        and rural_residential
        and "timber" in timber
        and primary == "white"
        and _stable_unit(
            kwargs.get("seed", "cwr-worldgen"),
            kwargs.get("width_m", 0.0),
            kwargs.get("length_m", 0.0),
            getattr(choice, "building_class", ""),
            getattr(choice, "facade_style", ""),
            "sweden-rural-red-balance-v1",
        ) < 0.50
    ):
        palette = ("falun red",) + tuple(
            value for value in palette
            if _normalised_text(value) != "falun red"
        )
        changed = True

    if not changed:
        return choice
    return replace(choice, window_spec=window, colour_palette=palette)


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
    """Prefer matching facade appearance only among already-valid physical fits."""
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
        namespace = _BUILDING_MODEL_CACHE_V56
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
    """Keep decorative fake glass on closed models, never on enterable ones."""
    trim = getattr(_GABLE_WINDOW_STATE, "trim_texture", None)
    omit_glass = bool(getattr(_GABLE_WINDOW_STATE, "omit_glass", False))
    changed = dict(kwargs)
    if trim:
        changed["trim_texture"] = trim

    if not omit_glass:
        return _ORIGINAL_ADD_GABLE_WINDOW(*args, **changed)

    openings = getattr(_GABLE_WINDOW_STATE, "openings", None)
    if openings is not None:
        openings.append({
            "centre_x": float(changed.get("centre_x", 0.0) or 0.0),
            "z": float(changed.get("z", 0.0) or 0.0),
            "y0": float(changed.get("y0", 0.0) or 0.0),
            "width": float(changed.get("width", 0.0) or 0.0),
            "height": float(changed.get("height", 0.0) or 0.0),
            "front": bool(changed.get("front", False)),
        })

    # The upstream helper adds its glass quad first and then the physical trim
    # boxes. Run it normally so frame geometry stays identical, then remove only
    # the newly-added faces that reference the fake closed-facade glass material.
    faces = args[2] if len(args) > 2 else None
    if faces is None:
        return _ORIGINAL_ADD_GABLE_WINDOW(*args, **changed)
    glass_texture = str(changed.get("glass_texture", "") or "")
    face_start = len(faces)
    result = _ORIGINAL_ADD_GABLE_WINDOW(*args, **changed)
    if glass_texture:
        kept = [
            face for face in faces[face_start:]
            if str(getattr(face, "texture", "")) != glass_texture
        ]
        del faces[face_start:]
        faces.extend(kept)
    return result


def _cut_enterable_gable_openings(
    points,
    normals,
    faces,
    key,
    *,
    roof_pitch_degrees: float,
    openings,
    base_face_count: int,
) -> None:
    """Replace solid gable triangles with wall triangles containing real holes."""
    if (
        not openings
        or not bool(getattr(key, "interiors", False))
        or bool(getattr(key, "footprint_vertices", ()))
        or str(getattr(key, "roof_style", "")) != "gabled"
    ):
        return

    from . import osm_house_modeler_upgrade as upgrade
    from . import procedural_buildings as buildings

    eave_y, roof_rise, _slope = buildings._gabled_profile(key, roof_pitch_degrees)
    if roof_rise <= 1.0e-6:
        return
    half_width = float(key.width_m) * 0.5
    half_length = float(key.length_m) * 0.5
    apex_y = eave_y + roof_rise

    for front, target_z in ((True, -half_length), (False, half_length)):
        side_openings = [opening for opening in openings if bool(opening["front"]) == front]
        if not side_openings:
            continue

        candidate_indices: list[int] = []
        for index, face in enumerate(faces[:base_face_count]):
            vertices = getattr(face, "vertices", ()) or ()
            if len(vertices) < 3:
                continue
            try:
                coordinates = [points[int(vertex[0])] for vertex in vertices]
            except (IndexError, TypeError, ValueError):
                continue
            if not all(abs(float(point[2]) - target_z) <= 1.0e-4 for point in coordinates):
                continue
            ys = [float(point[1]) for point in coordinates]
            if min(ys) < eave_y - 1.0e-4 or max(ys) <= eave_y + 1.0e-4:
                continue
            candidate_indices.append(index)

        if not candidate_indices:
            continue

        texture = str(getattr(faces[candidate_indices[0]], "texture", "") or "")
        holes = []
        for opening in side_openings:
            half = max(0.01, float(opening["width"]) * 0.5)
            y0 = float(opening["y0"])
            y1 = y0 + max(0.01, float(opening["height"]))
            centre_x = float(opening["centre_x"])
            holes.append((
                (centre_x - half, y0),
                (centre_x + half, y0),
                (centre_x + half, y1),
                (centre_x - half, y1),
            ))

        triangles = buildings._triangulate_polygon_coordinates(
            ((-half_width, eave_y), (half_width, eave_y), (0.0, apex_y)),
            tuple(holes),
        )
        if not triangles:
            continue

        # Remove both authored sides of the original solid gable before adding
        # the hole-aware replacement. Trim geometry was appended after
        # base_face_count, so it is deliberately excluded from this deletion.
        for index in reversed(candidate_indices):
            del faces[index]

        for triangle in triangles:
            coordinates = [
                (float(point[0]), float(point[1]), target_z)
                for point in triangle
            ]
            if len(coordinates) != 3:
                continue
            normal = upgrade._normal(coordinates[0], coordinates[1], coordinates[2])
            desired_sign = -1.0 if front else 1.0
            if normal[2] * desired_sign < 0.0:
                coordinates[1], coordinates[2] = coordinates[2], coordinates[1]
                normal = upgrade._normal(coordinates[0], coordinates[1], coordinates[2])

            start = len(points)
            points.extend(coordinates)
            normal_index = len(normals)
            normals.append(normal)
            vertices = []
            for offset, coordinate in enumerate(coordinates):
                u = (coordinate[0] + half_width) / max(0.01, half_width * 2.0)
                v = (coordinate[1] - eave_y) / max(0.01, roof_rise)
                vertices.append((start + offset, normal_index, u, v))
            face = buildings._Face(texture, tuple(vertices))
            faces.extend(buildings._double_sided_faces((face,)))


def _append_roof_storey_windows(
    points, normals, faces, key, *, roof_pitch_degrees, reference_texture
):
    """Apply light casing and true open-window semantics to roof-storey windows."""
    from . import procedural_buildings as buildings

    previous_trim = getattr(_GABLE_WINDOW_STATE, "trim_texture", None)
    previous_omit = bool(getattr(_GABLE_WINDOW_STATE, "omit_glass", False))
    previous_openings = getattr(_GABLE_WINDOW_STATE, "openings", None)
    trim = None
    if (
        bool(getattr(key, "interiors", False))
        and getattr(key, "family", "") not in buildings.UTILITY_INTERIOR_FAMILIES
        and buildings._uses_light_window_trim(key)
    ):
        trim = _shared_light_trim_path(reference_texture)
    opening_records = [] if bool(getattr(key, "interiors", False)) else None
    base_face_count = len(faces)
    _GABLE_WINDOW_STATE.trim_texture = trim
    _GABLE_WINDOW_STATE.omit_glass = bool(getattr(key, "interiors", False))
    _GABLE_WINDOW_STATE.openings = opening_records
    try:
        result = _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS(
            points,
            normals,
            faces,
            key,
            roof_pitch_degrees=roof_pitch_degrees,
            reference_texture=reference_texture,
        )
        if opening_records:
            _cut_enterable_gable_openings(
                points,
                normals,
                faces,
                key,
                roof_pitch_degrees=roof_pitch_degrees,
                openings=opening_records,
                base_face_count=base_face_count,
            )
        return result
    finally:
        _GABLE_WINDOW_STATE.trim_texture = previous_trim
        _GABLE_WINDOW_STATE.omit_glass = previous_omit
        _GABLE_WINDOW_STATE.openings = previous_openings


def install_osm_house_modeler_visual_policy() -> None:
    """Install final facade, reuse, trim, and enterable-window corrections."""
    global _INSTALLED
    global _ORIGINAL_RESOLVE_STYLE, _ORIGINAL_REUSE_CANDIDATES
    global _ORIGINAL_WINDOW_FRAME_TEXTURE_IMAGE, _ORIGINAL_CACHE_KEY
    global _ORIGINAL_APPEND_ROOF_STOREY_WINDOWS, _ORIGINAL_ADD_GABLE_WINDOW

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

    _INSTALLED = True
