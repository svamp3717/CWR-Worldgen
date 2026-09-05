# SPDX-License-Identifier: GPL-3.0-or-later
"""Let irregular churches keep their mapped polygon footprint.

The polygon-native building renderer supports the ordinary roof shapes used by
churches, but ``ProceduralBuildingLibrary.plan_polygon`` historically excluded the
entire ``church`` family from that path. Large irregular churches therefore fell
back to a minimum-rotated rectangle. In dense historic centres that rectangle can
cover streets and courtyards that are outside the real building, causing the final
road/building safety pass to reject a perfectly valid church.

Keep the mature rectangular path for genuinely rectangular churches and retain the
same polygon complexity/budget gates as every other native building. This policy
only supplies the native branch that the core method currently skips for churches;
all non-church placement remains delegated to the original implementation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Mapping, Sequence

from . import final_building_road_clearance_policy as _clearance
from . import procedural_buildings as _buildings

PointXZ = tuple[float, float]

_CACHE_REVISION = "final-road-building-clearance-v4-church-native-polygons"
_INSTALLED = False
_ORIGINAL_PLAN_POLYGON = None


def _native_church_plan_polygon(
    self,
    tags: Mapping[str, str],
    points: Sequence[PointXZ],
    *,
    holes: Sequence[Sequence[PointXZ]] = (),
    road_point: PointXZ | None = None,
    entrance_point: PointXZ | None = None,
    allow_native_polygon: bool = True,
):
    """Use the existing native-polygon recipe for eligible church footprints."""
    if not allow_native_polygon or not _buildings.is_actual_church(tags):
        return _ORIGINAL_PLAN_POLYGON(
            self,
            tags,
            points,
            holes=holes,
            road_point=road_point,
            entrance_point=entrance_point,
            allow_native_polygon=allow_native_polygon,
        )

    # Rectangular churches already use the mature rectangular model path and do
    # not need a bespoke polygon P3D merely because of tiny survey differences.
    rectangle = _buildings._simple_rectangle_footprint(points, holes)
    if rectangle is not None:
        return _ORIGINAL_PLAN_POLYGON(
            self,
            tags,
            points,
            holes=holes,
            road_point=road_point,
            entrance_point=entrance_point,
            allow_native_polygon=allow_native_polygon,
        )

    polygon, footprint = _buildings._polygon_with_footprint(points, holes)
    centre = polygon.centroid
    centre_x, centre_z = float(centre.x), float(centre.y)
    requested = self.key_for(
        tags,
        footprint.width_m,
        footprint.length_m,
        settlement_context=self._settlement_context(centre_x, centre_z),
    )
    if requested.family != "church":
        return _ORIGINAL_PLAN_POLYGON(
            self,
            tags,
            points,
            holes=holes,
            road_point=road_point,
            entrance_point=entrance_point,
            allow_native_polygon=allow_native_polygon,
        )

    native_profile = _buildings._native_polygon_profile(
        points,
        holes,
        footprint=footprint,
        polygon=polygon,
    )
    if native_profile is None:
        return _ORIGINAL_PLAN_POLYGON(
            self,
            tags,
            points,
            holes=holes,
            road_point=road_point,
            entrance_point=entrance_point,
            allow_native_polygon=allow_native_polygon,
        )

    (
        native_vertices,
        native_holes,
        native_heading,
        native_width,
        native_length,
    ) = native_profile
    native_requested = self.key_for(
        tags,
        native_width,
        native_length,
        settlement_context=self._settlement_context(centre_x, centre_z),
    )

    # Match the core native-polygon roof contract exactly. Churches with a
    # supported polygon-following roof retain it; exotic roofs keep their exact
    # walls but use the existing conservative flat-top fallback.
    native_roof_style = (
        native_requested.roof_style
        if native_requested.roof_style in {"flat", "gabled", "hipped", "pyramidal"}
        else "flat"
    )

    entrance_edge = -1
    entrance_fraction = 0.5
    frontage_point = entrance_point if entrance_point is not None else road_point
    if frontage_point is not None and native_vertices:
        angle = math.radians(native_heading)
        dx = float(frontage_point[0]) - centre_x
        dz = float(frontage_point[1]) - centre_z
        local_frontage = (
            dx * math.cos(angle) - dz * math.sin(angle),
            dx * math.sin(angle) + dz * math.cos(angle),
        )
        ranked_edges: list[tuple[float, int, float]] = []
        for edge_index, start in enumerate(native_vertices):
            end = native_vertices[(edge_index + 1) % len(native_vertices)]
            edge_x, edge_z = end[0] - start[0], end[1] - start[1]
            length_sq = edge_x * edge_x + edge_z * edge_z
            if length_sq <= 1.0e-8:
                continue
            fraction = max(
                0.0,
                min(
                    1.0,
                    (
                        (local_frontage[0] - start[0]) * edge_x
                        + (local_frontage[1] - start[1]) * edge_z
                    )
                    / length_sq,
                ),
            )
            nearest_x = start[0] + edge_x * fraction
            nearest_z = start[1] + edge_z * fraction
            distance_sq = (
                (local_frontage[0] - nearest_x) ** 2
                + (local_frontage[1] - nearest_z) ** 2
            )
            ranked_edges.append((distance_sq, edge_index, fraction))
        if ranked_edges:
            _distance, entrance_edge, entrance_fraction = min(ranked_edges)
            quantum = _buildings.POLYGON_NATIVE_ENTRANCE_FRACTION_QUANTUM
            entrance_fraction = max(
                0.0,
                min(1.0, round(entrance_fraction / quantum) * quantum),
            )

    native_requested = replace(
        native_requested,
        roof_style=native_roof_style,
        interiors=native_requested.interiors,
        second_storey=False,
        footprint_vertices=native_vertices,
        footprint_holes=native_holes,
        entrance_edge=entrance_edge,
        entrance_fraction=entrance_fraction,
    )
    hash_coordinates = tuple(points) + tuple(
        point for ring in holes for point in ring
    )
    placement_hash = _buildings._placement_hash_u32(tags, hash_coordinates)
    selected = replace(
        native_requested,
        texture_variant=_buildings._placement_texture_variant(
            tags,
            hash_coordinates,
            variant_count=(
                min(
                    self.texture_variants,
                    _buildings.INTERIOR_MODEL_TEXTURE_VARIANTS,
                )
                if native_requested.interiors
                else self.texture_variants
            ),
            placement_hash=placement_hash,
        ),
    )
    if (
        selected in self._polygon_native_keys
        or len(self._polygon_native_keys) < self.maximum_polygon_variants
    ):
        self._polygon_native_keys.add(selected)
        return _buildings.BuildingPlacement(
            self.model_path(selected),
            native_heading,
            native_requested,
            selected,
        )

    # Preserve the established bounded fallback if the polygon-native side
    # budget is ever exhausted.
    return _ORIGINAL_PLAN_POLYGON(
        self,
        tags,
        points,
        holes=holes,
        road_point=road_point,
        entrance_point=entrance_point,
        allow_native_polygon=allow_native_polygon,
    )


def install_church_native_polygon_policy() -> None:
    """Install church polygon fidelity after final worship/style resolution."""
    global _INSTALLED, _ORIGINAL_PLAN_POLYGON
    if _INSTALLED:
        return
    _ORIGINAL_PLAN_POLYGON = _buildings.ProceduralBuildingLibrary.plan_polygon
    _buildings.ProceduralBuildingLibrary.plan_polygon = _native_church_plan_polygon

    # The model path changes because the exact footprint enters the immutable
    # variant key, but an old non-road placement cache could still omit a church
    # rejected under its rectangular proxy. Force one fresh placement generation.
    _clearance._CACHE_REVISION = _CACHE_REVISION
    _INSTALLED = True
