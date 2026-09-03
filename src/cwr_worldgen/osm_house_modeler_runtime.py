# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime wiring for the OSM House Modeler building upgrade.

The adapter keeps CWR's P3D/MLOD, collision and enterable-building machinery,
while resolving the upstream modeler's complete country/region StyleChoice per
building and translating the useful values into the immutable CWR variant key.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
import threading
from typing import Mapping, Sequence

from . import osm_house_modeler_upgrade as _upgrade
from . import procedural_buildings as _pb
from .house_style_catalogue import HOUSE_STYLE_PRESET_AUTO, house_style_preset_profile
from .osm_house_modeler_full_style import (
    detail_spec_from_key,
    key_fields,
    requested_height,
    requested_levels,
    resolve_style,
    split_texture_token,
    tint_texture,
    visual_style_alias,
)

_STYLE_STATE = threading.local()
_ORIGINAL_KEY_FOR = None
_ORIGINAL_PREPARE_GEO = None
_ORIGINAL_ITER_DATASET_KEYS = None
_ORIGINAL_PLAN_POLYGON = None
_ORIGINAL_PLAN_POINT = None
_ORIGINAL_REGISTER_PLACEMENT = None
_ORIGINAL_DOOR_DIMENSIONS = None
_ORIGINAL_INTERIOR_WALL_THICKNESS = None
_ORIGINAL_INTERIOR_WINDOW_OPENINGS = None
_ORIGINAL_TEXTURE_FUNCTIONS: dict[str, object] = {}


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
    roof_pitch = float(
        getattr(key, "roof_pitch_degrees", 0.0)
        or _argument(args, kwargs, "roof_pitch_degrees", 3, 35.0)
        or 35.0
    )
    foundation_texture = (
        _argument(args, kwargs, "foundation_texture", 5, None) or roof_texture
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
    roof_pitch = float(
        getattr(key, "roof_pitch_degrees", 0.0)
        or kwargs.get("roof_pitch_degrees", 35.0)
        or 35.0
    )
    foundation_texture = kwargs.get("foundation_texture") or roof_texture
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


def _projection_for(library):
    return getattr(library, "_modeler_projection", None)


@contextmanager
def _style_location(library, x: float | None, z: float | None):
    previous = getattr(_STYLE_STATE, "location", None)
    projection = _projection_for(library)
    if projection is not None and x is not None and z is not None:
        try:
            latitude, longitude = projection.to_latlon((float(x), float(z)))
            _STYLE_STATE.location = (float(latitude), float(longitude))
        except (AttributeError, TypeError, ValueError):
            _STYLE_STATE.location = previous
    try:
        yield
    finally:
        _STYLE_STATE.location = previous


def _prepare_geographic_context(self, dataset, projection):
    self._modeler_projection = projection
    return _ORIGINAL_PREPARE_GEO(self, dataset, projection)


def _regional_preset(library) -> str:
    requested = str(getattr(library, "house_style_preset", HOUSE_STYLE_PRESET_AUTO) or "auto")
    if requested.casefold() == HOUSE_STYLE_PRESET_AUTO:
        return "auto"
    profile = house_style_preset_profile(requested)
    return profile.house_style_identifier if profile is not None else requested


def _style_key_for(
    self,
    tags: Mapping[str, str],
    width_m: float,
    length_m: float,
    *,
    foundation_depth_m: float | None = None,
    settlement_context: str = "rural",
):
    base = _ORIGINAL_KEY_FOR(
        self,
        tags,
        width_m,
        length_m,
        foundation_depth_m=foundation_depth_m,
        settlement_context=settlement_context,
    )
    location = getattr(_STYLE_STATE, "location", None)
    if location is None:
        projection = _projection_for(self)
        if projection is None:
            return base
        location = (
            (float(projection.south) + float(projection.north)) * 0.5,
            (float(projection.west) + float(projection.east)) * 0.5,
        )
    latitude, longitude = location
    try:
        choice = resolve_style(
            tags=tags,
            latitude=latitude,
            longitude=longitude,
            width_m=width_m,
            length_m=length_m,
            settlement_context=settlement_context,
            regional_preset=_regional_preset(self),
            seed=str(getattr(self, "world_name", "cwr-worldgen")),
        )
    except (LookupError, TypeError, ValueError):
        # A malformed custom style profile should not make the world generator
        # unusable. The mature CWR selection remains a safe fallback.
        return base

    fields = key_fields(choice)
    family = base.family if base.family == "church" else str(choice.family or base.family)
    roof_style = base.roof_style if family == "church" else str(choice.roof_style or base.roof_style)
    fallback_height = _pb._height(tags, family, max(2.4, float(choice.storey_height_m or self.default_level_height)))
    target_height = base.height_m if family == "church" else requested_height(tags, choice, fallback_height)
    quantized_height = _pb._quantize(
        target_height,
        self.height_quantum,
        self.minimum_height,
        self.maximum_height,
    )
    levels = base.facade_storeys if family == "church" else requested_levels(tags, choice)
    roof_storey = bool(fields["roof_storey"] and roof_style == "gabled" and levels >= 2 and family in {"residential", "townhouse", "urban"})
    facade_storeys = max(1, levels - 1) if roof_storey else max(1, levels)

    interior_max_width, interior_max_length, interior_max_height = (
        _pb.INTERIOR_FAMILY_MAXIMUM_DIMENSIONS_M.get(
            family,
            (
                _pb.INTERIOR_MAXIMUM_WIDTH_M,
                _pb.INTERIOR_MAXIMUM_LENGTH_M,
                _pb.INTERIOR_MAXIMUM_HEIGHT_M,
            ),
        )
    )
    width, length = sorted((max(0.1, float(width_m)), max(0.1, float(length_m))))
    interiors = (
        bool(self.generate_interiors)
        and family in _pb.INTERIOR_ELIGIBLE_FAMILIES
        and width <= interior_max_width
        and length <= interior_max_length
        and quantized_height <= interior_max_height
    )
    physically_supports_upper = (
        width >= _pb.INTERIOR_SECOND_STOREY_MINIMUM_WIDTH_M
        and length >= _pb.INTERIOR_SECOND_STOREY_MINIMUM_LENGTH_M
        and quantized_height >= _pb.INTERIOR_SECOND_STOREY_MINIMUM_HEIGHT_M
    )
    explicit_levels = _pb._parse_number(tags.get("building:levels"))
    second_storey = (
        interiors
        and family in _pb.SECOND_STOREY_INTERIOR_FAMILIES
        and levels >= 2
        and physically_supports_upper
        and not (explicit_levels is not None and explicit_levels < 2.0)
    )

    style_foundation = float(fields["style_foundation_depth_m"])
    return replace(
        base,
        family=family,
        roof_style=roof_style,
        height_m=quantized_height,
        foundation_depth_m=max(float(base.foundation_depth_m), style_foundation),
        regional_style=str(choice.facade_style or base.regional_style),
        interiors=interiors,
        second_storey=second_storey,
        outbuilding_kind=(str(choice.outbuilding_kind or base.outbuilding_kind) if family == "outbuilding" else ""),
        facade_storeys=facade_storeys,
        **fields,
    )


def _iter_dataset_keys(self, dataset, projection, point_footprint):
    for feature in dataset.building_polygons:
        for polygon in feature.polygons:
            projected = [projection.to_world(point) for point in polygon.outer[:-1]]
            if len(projected) < 3:
                continue
            projected_holes = tuple(
                tuple(projection.to_world(point) for point in ring[:-1])
                for ring in polygon.holes
                if len(ring) >= 4
            )
            rectangle = _pb._simple_rectangle_footprint(projected, projected_holes)
            if rectangle is not None:
                footprint, centre_x, centre_z = rectangle
            else:
                polygon_geometry, footprint = _pb._polygon_with_footprint(projected, projected_holes)
                centre = polygon_geometry.centroid
                centre_x, centre_z = float(centre.x), float(centre.y)
            with _style_location(self, centre_x, centre_z):
                key = self.key_for(
                    feature.tags,
                    footprint.width_m,
                    footprint.length_m,
                    settlement_context=self._settlement_context(centre_x, centre_z),
                )
            yield key
    for feature in dataset.building_points:
        x, z = projection.to_world(feature.point)
        with _style_location(self, x, z):
            key = self.key_for(
                feature.tags,
                point_footprint,
                point_footprint,
                settlement_context=self._settlement_context(x, z),
            )
        yield key


def _plan_polygon(self, tags, points: Sequence[tuple[float, float]], **kwargs):
    if points:
        centre_x = sum(float(point[0]) for point in points) / len(points)
        centre_z = sum(float(point[1]) for point in points) / len(points)
    else:
        centre_x = centre_z = None
    with _style_location(self, centre_x, centre_z):
        return _ORIGINAL_PLAN_POLYGON(self, tags, points, **kwargs)


def _plan_point(self, tags, footprint, heading_degrees, *, x=None, z=None, **kwargs):
    with _style_location(self, x, z):
        return _ORIGINAL_PLAN_POINT(
            self,
            tags,
            footprint,
            heading_degrees,
            x=x,
            z=z,
            **kwargs,
        )


def _register_placement(self, placement, *, foundation_depth_m=None):
    style_minimum = float(getattr(placement.selected, "style_foundation_depth_m", 0.0) or 0.0)
    requested = max(style_minimum, float(foundation_depth_m or 0.0)) if (style_minimum or foundation_depth_m is not None) else None
    return _ORIGINAL_REGISTER_PLACEMENT(self, placement, foundation_depth_m=requested)


def _door_dimensions(key):
    half, height, pivot = _ORIGINAL_DOOR_DIMENSIONS(key)
    if key.family in {"industrial", "agricultural"} or (
        key.family == "outbuilding" and _pb._outbuilding_is_garage(key)
    ):
        return half, height, pivot
    width = float(getattr(key, "door_width_m", 0.0) or 0.0)
    styled_height = float(getattr(key, "door_height_m", 0.0) or 0.0)
    if width > 0.0:
        half = max(0.32, min(max(0.35, key.width_m * 0.32), width * 0.5))
    if styled_height > 0.0:
        height = max(1.75, min(_pb._main_building_height(key) - 0.10, styled_height))
    return half, height, pivot


def _interior_wall_thickness(key):
    styled = float(getattr(key, "wall_thickness_m", 0.0) or 0.0)
    if styled > 0.0:
        return max(0.10, min(0.60, styled))
    return _ORIGINAL_INTERIOR_WALL_THICKNESS(key)


def _interior_window_openings(*args, **kwargs):
    openings = _ORIGINAL_INTERIOR_WINDOW_OPENINGS(*args, **kwargs)
    if not args:
        return openings
    key = args[0]
    target_width = float(getattr(key, "window_width_m", 0.0) or 0.0)
    target_height = float(getattr(key, "window_height_m", 0.0) or 0.0)
    target_sill = float(getattr(key, "window_sill_height_m", 0.0) or 0.0)
    density = max(0.0, float(getattr(key, "window_density_multiplier", 1.0) or 1.0))
    values = list(openings)
    if density < 0.999 and values:
        keep = max(0, min(len(values), int(round(len(values) * density))))
        if keep == 0:
            return type(openings)()
        if keep < len(values):
            step = len(values) / keep
            values = [values[min(len(values) - 1, int(index * step))] for index in range(keep)]
    result = []
    storey_height = max(2.4, float(getattr(key, "storey_height_m", 3.0) or 3.0))
    for opening in values:
        if not isinstance(opening, (tuple, list)) or len(opening) != 4:
            result.append(opening)
            continue
        x0, x1, y0, y1 = (float(v) for v in opening)
        if target_width > 0.0:
            centre = (x0 + x1) * 0.5
            half_width = min((x1 - x0) * 0.5, target_width * 0.5)
            x0, x1 = centre - half_width, centre + half_width
        if target_height > 0.0 or target_sill > 0.0:
            storey = max(0, int(math.floor((y0 + 0.05) / storey_height)))
            floor_y = storey * storey_height
            sill = target_sill if target_sill > 0.0 else max(0.45, y0 - floor_y)
            height = target_height if target_height > 0.0 else max(0.45, y1 - y0)
            y0 = floor_y + sill
            y1 = min(floor_y + storey_height - 0.20, y0 + height)
        result.append((x0, x1, y0, y1))
    return tuple(result) if isinstance(openings, tuple) else result


def _detail_plan_for_key(key, *, foundation_depth=0.0):
    spec = detail_spec_from_key(key)
    if not spec:
        return _ORIGINAL_DETAIL_PLAN(key, foundation_depth=foundation_depth)
    def enabled(name: str) -> bool:
        block = spec.get(name) or {}
        return bool(block.get("enabled", False)) if isinstance(block, Mapping) else False
    def count(name: str, default: int = 1) -> int:
        block = spec.get(name) or {}
        if not isinstance(block, Mapping):
            return 0
        try:
            return max(0, int(block.get("count", default)))
        except (TypeError, ValueError):
            return default
    pedestrian = not (
        key.family in {"industrial", "agricultural"}
        or (key.family == "outbuilding" and _pb._outbuilding_is_garage(key))
    )
    stairs = enabled("stairs") and pedestrian and not key.interiors and foundation_depth >= 0.18
    porch = enabled("porches") and pedestrian and key.family in {"residential", "townhouse"}
    chimneys = count("chimneys") if enabled("chimneys") and key.family in {"residential", "townhouse"} else 0
    balconies = count("balconies") if enabled("balconies") and key.family in {"residential", "townhouse", "urban"} else 0
    gutters = enabled("rainwater") and key.roof_style not in {"flat", "dome", "onion"}
    return _upgrade.ArchitecturalDetailPlan(stairs, porch, chimneys, balconies, gutters)


def _styled_image_wrapper(name: str, position: int, *, strength: float):
    original = _ORIGINAL_TEXTURE_FUNCTIONS[name]
    def wrapped(*args, **kwargs):
        positional = list(args)
        token = kwargs.get("regional_style")
        if token is None and len(positional) > position:
            token = positional[position]
        token = str(token or "default")
        facade, material, palette = split_texture_token(token)
        alias = visual_style_alias(facade, material)
        if "regional_style" in kwargs:
            kwargs = dict(kwargs)
            kwargs["regional_style"] = alias
        elif len(positional) > position:
            positional[position] = alias
        image = original(*positional, **kwargs)
        return tint_texture(image, palette, strength=strength)
    return wrapped


def _roof_image_wrapper(*args, **kwargs):
    original = _ORIGINAL_TEXTURE_FUNCTIONS["_roof_texture_image"]
    positional = list(args)
    token = kwargs.get("roof_style", kwargs.get("roof"))
    if token is None and positional:
        token = positional[0]
    roof_style, _material, palette = split_texture_token(str(token or "gabled"))
    if "roof_style" in kwargs:
        kwargs = dict(kwargs); kwargs["roof_style"] = roof_style
    elif "roof" in kwargs:
        kwargs = dict(kwargs); kwargs["roof"] = roof_style
    elif positional:
        positional[0] = roof_style
    image = original(*positional, **kwargs)
    return tint_texture(image, palette, strength=0.22)


def _install_detailed_style_adapter() -> None:
    global _ORIGINAL_KEY_FOR, _ORIGINAL_PREPARE_GEO, _ORIGINAL_ITER_DATASET_KEYS
    global _ORIGINAL_PLAN_POLYGON, _ORIGINAL_PLAN_POINT, _ORIGINAL_REGISTER_PLACEMENT
    global _ORIGINAL_DOOR_DIMENSIONS, _ORIGINAL_INTERIOR_WALL_THICKNESS
    global _ORIGINAL_INTERIOR_WINDOW_OPENINGS, _ORIGINAL_DETAIL_PLAN

    if getattr(_pb.ProceduralBuildingLibrary.key_for, "_cwr_full_modeler_style", False):
        return
    _ORIGINAL_KEY_FOR = _pb.ProceduralBuildingLibrary.key_for
    _ORIGINAL_PREPARE_GEO = _pb.ProceduralBuildingLibrary._prepare_geographic_context
    _ORIGINAL_ITER_DATASET_KEYS = _pb.ProceduralBuildingLibrary._iter_dataset_keys
    _ORIGINAL_PLAN_POLYGON = _pb.ProceduralBuildingLibrary.plan_polygon
    _ORIGINAL_PLAN_POINT = _pb.ProceduralBuildingLibrary.plan_point
    _ORIGINAL_REGISTER_PLACEMENT = _pb.ProceduralBuildingLibrary.register_placement
    _ORIGINAL_DOOR_DIMENSIONS = _pb._door_dimensions
    _ORIGINAL_INTERIOR_WALL_THICKNESS = _pb._interior_wall_thickness
    _ORIGINAL_INTERIOR_WINDOW_OPENINGS = _pb._interior_window_openings
    _ORIGINAL_DETAIL_PLAN = _upgrade.detail_plan_for_key

    _style_key_for._cwr_full_modeler_style = True  # type: ignore[attr-defined]
    _pb.ProceduralBuildingLibrary.key_for = _style_key_for
    _pb.ProceduralBuildingLibrary._prepare_geographic_context = _prepare_geographic_context
    _pb.ProceduralBuildingLibrary._iter_dataset_keys = _iter_dataset_keys
    _pb.ProceduralBuildingLibrary.plan_polygon = _plan_polygon
    _pb.ProceduralBuildingLibrary.plan_point = _plan_point
    _pb.ProceduralBuildingLibrary.register_placement = _register_placement
    _pb._door_dimensions = _door_dimensions
    _pb._interior_wall_thickness = _interior_wall_thickness
    _pb._interior_window_openings = _interior_window_openings
    _upgrade.detail_plan_for_key = _detail_plan_for_key

    for name, position, strength in (
        ("_wall_texture_image", 2, 0.30),
        ("_open_wall_texture_image", 2, 0.22),
        ("_interior_wall_texture_image", 2, 0.15),
        ("_front_texture_image", 2, 0.28),
        ("_door_texture_image", 2, 0.18),
    ):
        original = getattr(_pb, name, None)
        if original is None:
            continue
        _ORIGINAL_TEXTURE_FUNCTIONS[name] = original
        setattr(_pb, name, _styled_image_wrapper(name, position, strength=strength))
    roof_image = getattr(_pb, "_roof_texture_image", None)
    if roof_image is not None:
        _ORIGINAL_TEXTURE_FUNCTIONS["_roof_texture_image"] = roof_image
        _pb._roof_texture_image = _roof_image_wrapper


def install_osm_house_modeler_upgrade() -> None:
    """Install modeler geometry details plus complete per-building style wiring."""

    _upgrade.install_osm_house_modeler_upgrade()
    if not getattr(_pb._visual_lod, "_cwr_osm_house_modeler_runtime", False):
        _visual_lod._cwr_osm_house_modeler_runtime = True  # type: ignore[attr-defined]
        _polygon_visual_lod._cwr_osm_house_modeler_runtime = True  # type: ignore[attr-defined]
        _pb._visual_lod = _visual_lod
        if _upgrade._ORIGINAL_POLYGON_VISUAL_LOD is not None:
            _pb._polygon_native_visual_lod = _polygon_visual_lod
    _install_detailed_style_adapter()
