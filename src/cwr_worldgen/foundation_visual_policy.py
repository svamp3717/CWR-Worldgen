# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep visible procedural foundations from z-fighting with wall shells in CWA."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

FOUNDATION_SKIN_PROJECTION_M = 0.018
_BUILDING_MODEL_CACHE_V49 = "procedural-building-model-v49-robust-polygon-roof-triangulation"
_BUILDING_MODEL_CACHE_V50 = "procedural-building-model-v50-foundation-skin-offset"
_INSTALLED = False


def _offset_polygon_foundation_skin(
    lod: Any,
    *,
    foundation_texture: str | None,
    foundation_depth: float,
    foundation_top: float,
    projection: float = FOUNDATION_SKIN_PROJECTION_M,
):
    """Project only vertical foundation faces outward along their authored normals."""
    if not foundation_texture or foundation_depth <= 0.0 or projection <= 0.0:
        return lod

    points = list(lod.points)
    moved: set[int] = set()
    top_limit = max(0.0, float(foundation_top)) + 1.0e-5
    for face in lod.faces:
        if face.texture != foundation_texture or not face.vertices:
            continue
        indices = tuple(int(vertex[0]) for vertex in face.vertices)
        ys = tuple(float(points[index][1]) for index in indices)
        # Foundation side faces cross below local Y=0 and terminate at the
        # visible reveal. Interior floors can use the same texture but are
        # horizontal and never satisfy this vertical range test.
        if min(ys) >= -1.0e-6 or max(ys) > top_limit:
            continue
        normal_index = int(face.vertices[0][1])
        if normal_index < 0 or normal_index >= len(lod.normals):
            continue
        nx, _ny, nz = lod.normals[normal_index]
        horizontal = math.hypot(float(nx), float(nz))
        if horizontal <= 1.0e-6:
            continue
        dx = float(nx) / horizontal * projection
        dz = float(nz) / horizontal * projection
        for index in indices:
            if index in moved:
                continue
            x, y, z = points[index]
            points[index] = (float(x) + dx, float(y), float(z) + dz)
            moved.add(index)
    if not moved:
        return lod
    return replace(lod, points=tuple(points))


def install_foundation_visual_policy() -> None:
    """Install the foundation skin projection and one-time P3D cache revision."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import procedural_buildings as buildings

    original_skirt = buildings._add_foundation_skirt
    original_polygon_visual = buildings._polygon_native_visual_lod
    original_cache_key = buildings.cache_key

    def projected_foundation_skirt(
        points,
        faces,
        *,
        half_width: float,
        half_length: float,
        texture: str,
        depth: float,
        top_height: float = 0.0,
    ):
        # The wall shell starts at local Y=0. A visible plinth also extends above
        # Y=0, so leaving both on the exact same footprint creates coplanar faces
        # and severe CWA z-fighting. Give the visual skirt a tiny architectural
        # projection like a real foundation/plinth cap. Buried geometry may share
        # the same projection because it is visual-only and collision is separate.
        reveal = max(0.0, float(top_height))
        extra = FOUNDATION_SKIN_PROJECTION_M if reveal > 0.0 else 0.0
        return original_skirt(
            points,
            faces,
            half_width=float(half_width) + extra,
            half_length=float(half_length) + extra,
            texture=texture,
            depth=depth,
            top_height=top_height,
        )

    def projected_polygon_visual(*args, **kwargs):
        lod = original_polygon_visual(*args, **kwargs)
        key = args[0] if args else kwargs.get("key")
        depth = float(kwargs.get("foundation_depth", 0.0) or 0.0)
        texture = kwargs.get("foundation_texture")
        visible = max(
            buildings.FOUNDATION_VISIBLE_REVEAL_M if depth > 0.0 else 0.0,
            max(0.0, float(getattr(key, "visible_plinth_m", 0.0) or 0.0)),
        )
        return _offset_polygon_foundation_skin(
            lod,
            foundation_texture=texture,
            foundation_depth=depth,
            foundation_top=visible,
        )

    def revised_building_cache_key(namespace: str, payload):
        if namespace == _BUILDING_MODEL_CACHE_V49:
            namespace = _BUILDING_MODEL_CACHE_V50
        return original_cache_key(namespace, payload)

    buildings._add_foundation_skirt = projected_foundation_skirt
    buildings._polygon_native_visual_lod = projected_polygon_visual
    buildings.cache_key = revised_building_cache_key
    _INSTALLED = True

    # Install after the runtime style adapter and foundation wrapper so opening
    # sizes see the final modeler key fields and can advance the P3D cache from
    # v50 without discarding the already-correct texture cache.
    from .opening_dimension_policy import install_opening_dimension_policy
    from .opening_texture_policy import install_opening_texture_policy

    install_opening_dimension_policy()
    install_opening_texture_policy()

    # Performance wrappers must be last among geometry policies: they memoize the
    # final opening functions and compact only the generated distance LOD, never
    # the detail LOD.
    from .interior_performance_policy import install_interior_performance_policy

    install_interior_performance_policy()

    # Bound exact polygon P3Ds after the final geometry/performance wrappers are
    # installed. This keeps complex footprint fidelity without letting a hidden
    # 2048-model side budget dominate the asset-generation stage.
    from .building_asset_budget_policy import install_building_asset_budget_policy

    install_building_asset_budget_policy()

    # The asset budget knows which modeler textures/P3Ds are expensive. Install
    # streaming completion reporting after it so worker processes can publish
    # completed/total counters without changing any other procedural asset stage.
    from .building_progress_policy import install_building_progress_policy

    install_building_progress_policy()

    # Preparing variants is a separate serial phase before texture/P3D workers.
    # Expose its style-resolution and capped-reuse counters as well so the GUI no
    # longer appears frozen at the same percentage on building-heavy worlds.
    from .building_prepare_progress_policy import install_building_prepare_progress_policy

    install_building_prepare_progress_policy()

    # Utility material selection is deliberately data-only: all barn/shed/garage/
    # warehouse/hangar/industrial pools live in the selected country JSON. The
    # runtime merely consumes those explicit distributions and supplies matching
    # utility material renderers.
    from .country_utility_material_policy import install_country_utility_material_policy

    install_country_utility_material_policy()

    # Keep this final among style/geometry wrappers: it consumes the country-
    # adjusted resolver and final geometry stack without bypassing weighted
    # material/colour selection.
    from .osm_house_modeler_visual_policy import install_osm_house_modeler_visual_policy

    install_osm_house_modeler_visual_policy()

    # The GUI no longer exposes broad regional architecture presets. Explicit
    # building style selection is country-only, while Automatic still detects the
    # country from the selected map area. Install after the visual resolver so a
    # forced country flows through every existing material/colour/window fix.
    from .building_country_policy import install_building_country_policy

    install_building_country_policy()

    # Worship semantics are global, not country-specific. Install this last so a
    # church/mosque/synagogue can consume the selected country context first, then
    # replace only residential-looking materials/colours with its class defaults.
    # Explicit OSM building/roof material and colour tags still remain final.
    from .worship_building_policy import install_worship_building_policy

    install_worship_building_policy()

    # Irregular churches used to be the one semantic family barred from the
    # polygon-native model path. Install after worship classification so churches
    # retain their exact mapped outline without bypassing the final worship style.
    from .church_native_polygon_policy import install_church_native_polygon_policy

    install_church_native_polygon_policy()

    # The model-cache call site still enters the wrapper chain with the historical
    # v49 namespace. Install one final church-only cache gate outside that chain so
    # tower-aware native church P3Ds cannot reuse the old towerless asset bytes.
    from .church_native_tower_cache_policy import install_church_native_tower_cache_policy

    install_church_native_tower_cache_policy()

    # Desert ground is stock game artwork, not generated world-local PAA data.
    # Install after the core surface modules are loaded so both Milestone 8 and 9
    # paths share the same external texture table and skip DXT1 ground generation.
    from .stock_desert_surface_policy import install_stock_desert_surface_policy

    install_stock_desert_surface_policy()
