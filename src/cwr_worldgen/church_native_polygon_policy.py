# SPDX-License-Identifier: GPL-3.0-or-later
"""Let irregular churches keep their mapped polygon footprint and church silhouette.

The polygon-native building renderer supports the ordinary roof shapes used by
churches, but ``ProceduralBuildingLibrary.plan_polygon`` historically excluded the
entire ``church`` family from that path. Large irregular churches therefore fell
back to a minimum-rotated rectangle. In dense historic centres that rectangle can
cover streets and courtyards that are outside the real building, causing the final
road/building safety pass to reject a perfectly valid church.

Once churches were allowed onto the native-polygon path another old assumption
became visible: the rectangular visual renderer owns the Christian tower/spire
mesh, while the native renderer only authored the generic footprint-following
shell. A native church therefore kept its semantic key and worship materials but
looked like an ordinary civic building. This policy also appends a compact tower
and spire to polygon-native Christian churches, fitted wholly inside the mapped
footprint near the selected entrance/front edge.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Mapping, Sequence

from shapely.geometry import Polygon as ShapelyPolygon

from . import final_building_road_clearance_policy as _clearance
from . import procedural_buildings as _buildings

PointXZ = tuple[float, float]

_CACHE_REVISION = "final-road-building-clearance-v5-church-native-towers"
_BUILDING_MODEL_CACHE_V56 = "procedural-building-model-v56-enterable-gable-openings"
_BUILDING_MODEL_CACHE_V57 = "procedural-building-model-v57-native-church-towers"
_INSTALLED = False
_ORIGINAL_PLAN_POLYGON = None
_ORIGINAL_POLYGON_VISUAL = None
_ORIGINAL_CACHE_KEY = None


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


def _church_tower_base(key):
    """Return four local X/Z tower corners wholly inside a native church."""
    outer = tuple(key.footprint_vertices)
    if len(outer) < 3:
        return None
    try:
        shape = _buildings._polygon_native_shape(key)
    except (TypeError, ValueError):
        return None
    if shape.is_empty:
        return None

    preferred = _buildings._polygon_native_front_edge(key)
    spans: list[tuple[float, int]] = []
    for index, start in enumerate(outer):
        end = outer[(index + 1) % len(outer)]
        span = math.hypot(end[0] - start[0], end[1] - start[1])
        if span >= 3.0:
            spans.append((span, index))
    ordered = []
    if any(index == preferred for _span, index in spans):
        ordered.append(preferred)
    ordered.extend(
        index for _span, index in sorted(spans, reverse=True)
        if index != preferred
    )

    for edge_index in ordered:
        start = outer[edge_index]
        end = outer[(edge_index + 1) % len(outer)]
        span, tx, tz, inward_x, inward_z = _buildings._polygon_native_edge_frame(
            start, end
        )
        if span <= 1.0e-6:
            continue
        desired_half = min(
            4.0,
            max(1.5, min(float(key.width_m) * 0.22, span * 0.28)),
        )
        desired_depth = min(
            6.0,
            max(3.0, desired_half * 1.45, float(key.length_m) * 0.08),
        )
        requested_fraction = (
            float(getattr(key, "entrance_fraction", 0.5) or 0.5)
            if edge_index == preferred else 0.5
        )
        for scale in (1.0, 0.88, 0.76, 0.64):
            half = desired_half * scale
            depth = desired_depth * scale
            if half * 2.0 >= span - 0.10:
                continue
            minimum_fraction = (half + 0.05) / span
            fraction = max(
                minimum_fraction,
                min(1.0 - minimum_fraction, requested_fraction),
            )
            edge_x = float(start[0]) + (float(end[0]) - float(start[0])) * fraction
            edge_z = float(start[1]) + (float(end[1]) - float(start[1])) * fraction
            # Stay a few centimetres inside the authored wall so the tower and
            # facade meet cleanly without extending the collision footprint.
            front_x = edge_x + inward_x * 0.04
            front_z = edge_z + inward_z * 0.04
            back_x = front_x + inward_x * depth
            back_z = front_z + inward_z * depth
            corners = (
                (front_x - tx * half, front_z - tz * half),
                (front_x + tx * half, front_z + tz * half),
                (back_x + tx * half, back_z + tz * half),
                (back_x - tx * half, back_z - tz * half),
            )
            try:
                tower_shape = ShapelyPolygon(corners)
            except (TypeError, ValueError):
                continue
            if not tower_shape.is_empty and shape.buffer(0.02).covers(tower_shape):
                return corners, half, depth
    return None


def _append_native_church_tower(
    lod,
    key,
    *,
    wall_texture: str,
    roof_texture: str,
    front_texture: str | None,
    plain_wall_texture: str | None,
):
    """Append the familiar CWR Christian tower/spire to a native church shell."""
    if key.family != "church" or not key.footprint_vertices:
        return lod
    base = _church_tower_base(key)
    if base is None:
        return lod
    corners, tower_half, _tower_depth = base

    tower_height = max(
        18.0,
        float(key.height_m) + 8.0,
        _buildings._main_building_height(key) + 10.0,
    )
    apex_height = tower_height + min(8.0, max(5.0, tower_half * 2.0))
    ground_top = min(3.0, tower_height)
    window_bottom = max(ground_top + 2.0, tower_height - 5.0)
    window_top = min(tower_height - 0.8, max(window_bottom + 1.8, tower_height - 2.2))
    if window_top <= window_bottom + 0.25:
        window_bottom = max(ground_top + 1.0, tower_height - 4.0)
        window_top = tower_height - 1.0
    heights = (0.0, ground_top, window_bottom, window_top, tower_height)

    points = list(lod.points)
    normals = list(lod.normals)
    faces = list(lod.faces)
    rings: list[tuple[int, int, int, int]] = []
    for height in heights:
        indices = []
        for x, z in corners:
            indices.append(len(points))
            points.append((float(x), float(height), float(z)))
        rings.append(tuple(indices))
    centre_x = sum(point[0] for point in corners) * 0.25
    centre_z = sum(point[1] for point in corners) * 0.25
    apex = len(points)
    points.append((centre_x, apex_height, centre_z))

    def add_face(indices: Sequence[int], texture: str) -> None:
        ordered = list(indices)

        def normal_for(values: Sequence[int]) -> tuple[float, float, float]:
            p0, p1, p2 = (points[values[index]] for index in range(3))
            ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length <= 1.0e-9:
                return (0.0, 1.0, 0.0)
            return (nx / length, ny / length, nz / length)

        normal = normal_for(ordered)
        mx = sum(points[index][0] for index in ordered) / len(ordered)
        mz = sum(points[index][2] for index in ordered) / len(ordered)
        if normal[0] * (mx - centre_x) + normal[2] * (mz - centre_z) < 0.0:
            ordered.reverse()
            normal = normal_for(ordered)
        normal_index = len(normals)
        normals.append(normal)
        if len(ordered) == 3:
            uv = ((0.0, 1.0), (0.5, 0.0), (1.0, 1.0))
        else:
            uv = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        faces.append(_buildings._Face(
            texture,
            tuple(
                (point_index, normal_index, uv_index[0], uv_index[1])
                for point_index, uv_index in zip(ordered, uv)
            ),
        ))

    wall = wall_texture
    plain = plain_wall_texture or wall_texture
    front = front_texture or wall_texture
    # Four wall bands: entrance level, plain shaft, belfry windows, plain crown.
    for band_index in range(4):
        lower = rings[band_index]
        upper = rings[band_index + 1]
        for side in range(4):
            next_side = (side + 1) % 4
            if band_index == 0:
                texture = front if side == 0 else wall
            elif band_index == 2:
                texture = wall
            else:
                texture = plain
            add_face(
                (lower[side], lower[next_side], upper[next_side], upper[side]),
                texture,
            )

    top = rings[-1]
    for side in range(4):
        add_face((top[side], top[(side + 1) % 4], apex), roof_texture)

    return replace(
        lod,
        points=tuple(points),
        normals=tuple(normals),
        faces=tuple(faces),
    )


def _polygon_visual_with_church_tower(*args, **kwargs):
    lod = _ORIGINAL_POLYGON_VISUAL(*args, **kwargs)
    key = args[0] if args else kwargs.get("key")
    if key is None or getattr(key, "family", "") != "church" or not getattr(key, "footprint_vertices", ()):
        return lod
    wall_texture = args[1] if len(args) > 1 else kwargs.get("wall_texture", "")
    roof_texture = args[2] if len(args) > 2 else kwargs.get("roof_texture", "")
    return _append_native_church_tower(
        lod,
        key,
        wall_texture=str(wall_texture or ""),
        roof_texture=str(roof_texture or ""),
        front_texture=kwargs.get("front_texture"),
        plain_wall_texture=kwargs.get("plain_wall_texture"),
    )


def _revised_cache_key(namespace: str, payload):
    if namespace == _BUILDING_MODEL_CACHE_V56:
        namespace = _BUILDING_MODEL_CACHE_V57
    return _ORIGINAL_CACHE_KEY(namespace, payload)


def install_church_native_polygon_policy() -> None:
    """Install church polygon fidelity and native church silhouette support."""
    global _INSTALLED, _ORIGINAL_PLAN_POLYGON, _ORIGINAL_POLYGON_VISUAL, _ORIGINAL_CACHE_KEY
    if _INSTALLED:
        return
    _ORIGINAL_PLAN_POLYGON = _buildings.ProceduralBuildingLibrary.plan_polygon
    _ORIGINAL_POLYGON_VISUAL = _buildings._polygon_native_visual_lod
    _ORIGINAL_CACHE_KEY = _buildings.cache_key
    _buildings.ProceduralBuildingLibrary.plan_polygon = _native_church_plan_polygon
    _buildings._polygon_native_visual_lod = _polygon_visual_with_church_tower
    _buildings.cache_key = _revised_cache_key

    # The model path changes because the exact footprint enters the immutable
    # variant key, but an old non-road placement cache could still omit a church
    # rejected under its rectangular proxy. Force one fresh placement generation.
    _clearance._CACHE_REVISION = _CACHE_REVISION
    _INSTALLED = True
