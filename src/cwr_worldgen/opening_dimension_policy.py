# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep modeler door/window dimensions physically correct in generated P3Ds."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

_BUILDING_MODEL_CACHE_V49 = "procedural-building-model-v49-robust-polygon-roof-triangulation"
_BUILDING_MODEL_CACHE_V50 = "procedural-building-model-v50-foundation-skin-offset"
_BUILDING_MODEL_CACHE_V51 = "procedural-building-model-v51-modeler-opening-dimensions"
_INSTALLED = False
_ORIGINAL_DOOR_DIMENSIONS = None
_ORIGINAL_INTERIOR_WINDOW_OPENINGS = None
_ORIGINAL_POLYGON_EDGE_OPENINGS = None
_ORIGINAL_VISUAL_LOD = None


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _door_metadata(key: object) -> dict[str, Any]:
    from .osm_house_modeler_full_style import texture_metadata_from_token

    metadata = texture_metadata_from_token(str(getattr(key, "texture_style_token", "") or ""))
    door = metadata.get("door") or {}
    return dict(door) if isinstance(door, Mapping) else {}


def _styled_door_dimensions(key):
    half, height, pivot = _ORIGINAL_DOOR_DIMENSIONS(key)
    door = _door_metadata(key)
    utility = key.family in {"industrial", "agricultural"} or (
        key.family == "outbuilding" and _is_garage(key)
    )
    if utility:
        width = _number(door.get("utility_width_m"), 0.0)
        styled_height = _number(door.get("utility_height_m"), 0.0)
    else:
        width = _number(getattr(key, "door_width_m", 0.0), 0.0)
        styled_height = _number(getattr(key, "door_height_m", 0.0), 0.0)
    if width <= 0.0:
        width = _number(door.get("width_m"), 0.0)
    if styled_height <= 0.0:
        styled_height = _number(door.get("height_m"), 0.0)

    if width > 0.0:
        corner = max(
            0.12,
            _number(getattr(key, "door_corner_clearance_m", 0.0), 0.0)
            or _number(door.get("corner_clearance_m"), 0.0)
            or 0.20,
        )
        maximum_width = max(0.40, float(key.width_m) - corner * 2.0)
        half = max(0.20, min(width, maximum_width) * 0.5)
    if styled_height > 0.0:
        maximum_height = max(1.0, float(_main_building_height(key)) - 0.10)
        height = max(1.0, min(styled_height, maximum_height))
    return half, height, pivot


def _is_garage(key) -> bool:
    from . import procedural_buildings as buildings
    return bool(buildings._outbuilding_is_garage(key))


def _main_building_height(key) -> float:
    from . import procedural_buildings as buildings
    return float(buildings._main_building_height(key))


def _style_window_openings(
    key,
    horizontal_min: float,
    horizontal_max: float,
    wall_top: float,
    *,
    ground_exclusions: Sequence[tuple[float, float]] = (),
):
    from . import procedural_buildings as buildings

    target_width = _number(getattr(key, "window_width_m", 0.0), 0.0)
    target_height = _number(getattr(key, "window_height_m", 0.0), 0.0)
    target_sill = _number(getattr(key, "window_sill_height_m", 0.0), 0.0)
    edge_margin = max(0.0, _number(getattr(key, "window_edge_margin_m", 0.0), 0.0))
    bay_spacing = max(0.8, _number(getattr(key, "window_bay_spacing_m", 0.0), 3.8) or 3.8)
    density = max(0.0, _number(getattr(key, "window_density_multiplier", 1.0), 1.0))
    if density <= 0.0:
        return ()
    if target_width <= 0.0 or target_height <= 0.0:
        return _ORIGINAL_INTERIOR_WINDOW_OPENINGS(
            key,
            horizontal_min,
            horizontal_max,
            wall_top,
            ground_exclusions=ground_exclusions,
        )

    left = float(horizontal_min)
    right = float(horizontal_max)
    span = right - left
    if span <= 0.5:
        return ()
    edge_margin = min(edge_margin, max(0.0, span * 0.5 - 0.20))
    usable_left = left + edge_margin
    usable_right = right - edge_margin
    usable_span = usable_right - usable_left
    if usable_span < min(0.40, target_width):
        return ()

    nominal_count = max(1, int(math.floor((usable_span / bay_spacing) * density + 0.5)))
    minimum_gap = 0.20
    fit_count = max(1, int(math.floor((usable_span + minimum_gap) / (target_width + minimum_gap))))
    count = max(1, min(12, nominal_count, fit_count))
    cell_width = usable_span / count
    actual_width = min(target_width, max(0.35, cell_width - minimum_gap))

    visible_storeys = max(1, int(buildings._visible_window_storey_count(key, wall_top=wall_top)))
    walkable_storeys = max(1, int(buildings._interior_storey_count(key, wall_top=wall_top)))
    storey_step = (
        float(buildings.INTERIOR_SECOND_STOREY_FLOOR_Y_M)
        if walkable_storeys >= 2
        else max(2.4, _number(getattr(key, "storey_height_m", 3.0), 3.0))
    )
    sill = target_sill if target_sill > 0.0 else float(buildings.INTERIOR_WINDOW_SILL_M)
    clearance_extra = max(
        0.0,
        _number(getattr(key, "door_window_clearance_m", 0.0), 0.0) - 0.35,
    )
    expanded_exclusions = tuple(
        (float(x0) - clearance_extra, float(x1) + clearance_extra)
        for x0, x1 in ground_exclusions
    )

    openings: list[tuple[float, float, float, float]] = []
    for storey in range(visible_storeys):
        floor_y = storey * storey_step
        y0 = floor_y + sill
        y1 = min(y0 + target_height, wall_top - 0.20)
        if walkable_storeys >= 2 and storey < walkable_storeys - 1:
            y1 = min(y1, floor_y + storey_step - 0.15)
        if y1 - y0 < min(0.35, target_height * 0.60):
            continue
        for index in range(count):
            centre = usable_left + (index + 0.5) * cell_width
            x0 = centre - actual_width * 0.5
            x1 = centre + actual_width * 0.5
            if storey == 0 and any(
                x0 < excluded_max and x1 > excluded_min
                for excluded_min, excluded_max in expanded_exclusions
            ):
                continue
            openings.append((x0, x1, y0, y1))
    return tuple(openings)


def _styled_interior_window_openings(*args, **kwargs):
    if not args:
        return _ORIGINAL_INTERIOR_WINDOW_OPENINGS(*args, **kwargs)
    key = args[0]
    horizontal_min = float(args[1]) if len(args) > 1 else float(kwargs["horizontal_min"])
    horizontal_max = float(args[2]) if len(args) > 2 else float(kwargs["horizontal_max"])
    wall_top = float(args[3]) if len(args) > 3 else float(kwargs["wall_top"])
    exclusions = kwargs.get("ground_exclusions", ())
    return _style_window_openings(
        key,
        horizontal_min,
        horizontal_max,
        wall_top,
        ground_exclusions=exclusions,
    )


def _styled_polygon_edge_openings(
    key,
    edge_index: int,
    span: float,
    wall_top: float,
    *,
    courtyard: bool = False,
):
    from . import procedural_buildings as buildings

    if not key.interiors:
        return ()
    door = None if courtyard else buildings._polygon_native_door_opening(
        key, edge_index, span, wall_top
    )
    exclusions: tuple[tuple[float, float], ...] = ()
    if door is not None:
        clearance = max(0.0, _number(getattr(key, "door_window_clearance_m", 0.0), 0.35))
        exclusions = ((max(0.0, door[0] - clearance), min(span, door[1] + clearance)),)
    windows = ()
    if key.family not in buildings.UTILITY_INTERIOR_FAMILIES and span >= 1.0:
        windows = _style_window_openings(
            key, 0.0, float(span), float(wall_top), ground_exclusions=exclusions
        )
    return windows + ((door,) if door is not None else ())


def _front_door_uv(key) -> tuple[float, float, float, float]:
    door = _door_metadata(key)
    width = _number(door.get("width_m"), 0.95) or 0.95
    height = _number(door.get("height_m"), 2.05) or 2.05
    u_span = max(0.14, min(0.34, width / 4.0))
    v_span = max(0.48, min(0.84, height / 3.0))
    return ((1.0 - u_span) * 0.5, (1.0 + u_span) * 0.5, 1.0 - v_span, 1.0)


def _correct_front_wall_uv(lod, key, wall_texture: str):
    from . import procedural_buildings as buildings

    half_width = float(key.width_m) * 0.5
    front_z = -float(key.length_m) * 0.5
    faces = []
    changed = False
    for face in lod.faces:
        if face.texture != wall_texture or not face.vertices:
            faces.append(face)
            continue
        indices = [int(vertex[0]) for vertex in face.vertices]
        if not all(abs(float(lod.points[index][2]) - front_z) <= 1.0e-4 for index in indices):
            faces.append(face)
            continue
        vertices = []
        for point_index, normal_index, _u, v in face.vertices:
            x = float(lod.points[int(point_index)][0])
            u = (x + half_width) / 4.0
            vertices.append((point_index, normal_index, u, v))
        faces.append(buildings._Face(face.texture, tuple(vertices), face.flags))
        changed = True
    return replace(lod, faces=tuple(faces)) if changed else lod


def _append_closed_door_overlay(
    lod,
    key,
    *,
    door_texture: str,
    plain_wall_texture: str,
):
    from . import procedural_buildings as buildings

    half, height, _pivot = buildings._door_dimensions(key)
    if half <= 0.0 or height <= 0.0:
        return lod
    wall_half_length = float(key.length_m) * 0.5
    clearance = max(0.08, _number(getattr(key, "door_window_clearance_m", 0.0), 0.20))
    patch_half = min(float(key.width_m) * 0.48, half + clearance)
    patch_height = min(_main_building_height(key) - 0.05, height + 0.18)
    if patch_height <= 0.2:
        return lod

    point_start = len(lod.points)
    patch_z = -wall_half_length - 0.012
    door_z = -wall_half_length - 0.026
    points = lod.points + (
        (-patch_half, 0.0, patch_z), (-patch_half, patch_height, patch_z),
        (patch_half, patch_height, patch_z), (patch_half, 0.0, patch_z),
        (-half, 0.02, door_z), (-half, height - 0.02, door_z),
        (half, height - 0.02, door_z), (half, 0.02, door_z),
    )
    normal_start = len(lod.normals)
    normals = lod.normals + ((0.0, 0.0, -1.0),)
    patch_u = max(0.1, patch_half * 2.0 / 4.0)
    patch_v = max(0.1, patch_height / 3.0)
    patch_face = buildings._Face(plain_wall_texture, (
        (point_start + 0, normal_start, 0.0, patch_v),
        (point_start + 1, normal_start, 0.0, 0.0),
        (point_start + 2, normal_start, patch_u, 0.0),
        (point_start + 3, normal_start, patch_u, patch_v),
    ))
    u0, u1, v0, v1 = _front_door_uv(key)
    door_face = buildings._Face(door_texture, (
        (point_start + 4, normal_start, u0, v1),
        (point_start + 5, normal_start, u0, v0),
        (point_start + 6, normal_start, u1, v0),
        (point_start + 7, normal_start, u1, v1),
    ))
    faces = lod.faces + (
        patch_face, buildings._reverse_face(patch_face),
        door_face, buildings._reverse_face(door_face),
    )
    selections = []
    for selection in lod.selections:
        point_weights = selection.point_weights + bytes(max(0, len(points) - len(selection.point_weights)))
        face_flags = selection.face_flags + bytes(max(0, len(faces) - len(selection.face_flags)))
        selections.append(replace(selection, point_weights=point_weights, face_flags=face_flags))
    return replace(
        lod,
        points=points,
        normals=normals,
        faces=faces,
        selections=tuple(selections),
    )


def _styled_visual_lod(*args, **kwargs):
    if not args:
        return _ORIGINAL_VISUAL_LOD(*args, **kwargs)
    key = args[0]
    if key.interiors or key.family == "church":
        return _ORIGINAL_VISUAL_LOD(*args, **kwargs)

    wall_texture = str(args[1])
    original_front = (
        kwargs.get("front_texture")
        if "front_texture" in kwargs
        else (args[4] if len(args) > 4 else wall_texture)
    ) or wall_texture
    plain_wall = (
        kwargs.get("plain_wall_texture")
        if "plain_wall_texture" in kwargs
        else (args[10] if len(args) > 10 else wall_texture)
    ) or wall_texture

    positional = list(args)
    changed_kwargs = dict(kwargs)
    if len(positional) > 4:
        positional[4] = wall_texture
        changed_kwargs.pop("front_texture", None)
    else:
        changed_kwargs["front_texture"] = wall_texture
    lod = _ORIGINAL_VISUAL_LOD(*positional, **changed_kwargs)
    lod = _correct_front_wall_uv(lod, key, wall_texture)
    return _append_closed_door_overlay(
        lod,
        key,
        door_texture=str(original_front),
        plain_wall_texture=str(plain_wall),
    )


def _enrich_texture_metadata(original):
    def enriched(choice):
        metadata = dict(original(choice))
        metadata["texture_renderer_revision"] = 5
        source_window = dict(choice.window_spec or {})
        source_door = dict(choice.door_spec or {})
        window = dict(metadata.get("window") or {})
        window["edge_margin_m"] = _number(source_window.get("edge_margin_m"), 0.0)
        metadata["window"] = window
        door = dict(metadata.get("door") or {})
        door["utility_width_m"] = _number(source_door.get("utility_width_m"), 0.0)
        door["utility_height_m"] = _number(source_door.get("utility_height_m"), 0.0)
        door["utility_role"] = str(source_door.get("utility_role", "") or "")
        door["corner_clearance_m"] = _number(source_door.get("corner_clearance_m"), 0.0)
        door["keep_clear_of_windows_m"] = _number(source_door.get("keep_clear_of_windows_m"), 0.0)
        metadata["door"] = door
        return metadata
    return enriched


def install_opening_dimension_policy() -> None:
    """Install exact meter-scale modeler openings and invalidate only building P3Ds."""
    global _INSTALLED, _ORIGINAL_DOOR_DIMENSIONS, _ORIGINAL_INTERIOR_WINDOW_OPENINGS
    global _ORIGINAL_POLYGON_EDGE_OPENINGS, _ORIGINAL_VISUAL_LOD
    if _INSTALLED:
        return

    from . import osm_house_modeler_full_style as full_style
    from . import procedural_buildings as buildings

    original_metadata = full_style._texture_metadata
    if not getattr(original_metadata, "_cwr_opening_dimensions", False):
        enriched = _enrich_texture_metadata(original_metadata)
        enriched._cwr_opening_dimensions = True  # type: ignore[attr-defined]
        full_style._texture_metadata = enriched

    _ORIGINAL_DOOR_DIMENSIONS = buildings._door_dimensions
    _ORIGINAL_INTERIOR_WINDOW_OPENINGS = buildings._interior_window_openings
    _ORIGINAL_POLYGON_EDGE_OPENINGS = buildings._polygon_native_edge_openings
    _ORIGINAL_VISUAL_LOD = buildings._visual_lod
    original_cache_key = buildings.cache_key

    def revised_building_cache_key(namespace: str, payload):
        if namespace in {_BUILDING_MODEL_CACHE_V49, _BUILDING_MODEL_CACHE_V50}:
            namespace = _BUILDING_MODEL_CACHE_V51
        return original_cache_key(namespace, payload)

    buildings._door_dimensions = _styled_door_dimensions
    buildings._interior_window_openings = _styled_interior_window_openings
    buildings._polygon_native_edge_openings = _styled_polygon_edge_openings
    buildings._visual_lod = _styled_visual_lod
    buildings.cache_key = revised_building_cache_key
    _INSTALLED = True
