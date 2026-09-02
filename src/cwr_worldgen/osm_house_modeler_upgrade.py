# SPDX-License-Identifier: GPL-3.0-or-later
"""OSM House Modeler architectural-detail adapter for CWR's P3D generator.

The standalone osm-house-modeler v0.13.6 has a useful secondary-architecture
pass (stairs, porches, balconies, chimneys, gutters/downspouts) and a simple
interior shell. CWR already has a stronger OFP/CWA-specific shell: arbitrary
footprints, courtyard-aware roofs, Geometry/Roadway/Memory/Paths LODs, real
window/door openings, multi-storey floors, and animated entrance doors.

This module ports the modeler's secondary-architecture policy into CWR while
leaving CWR's engine-specific shell/collision implementation authoritative.
The modeler project declares MIT licensing; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import threading
from typing import Callable

from . import procedural_buildings as _pb

OSM_HOUSE_MODELER_SOURCE_VERSION = "0.13.6"
OSM_HOUSE_MODELER_SOURCE_URL = "https://github.com/svamp3717/osm-house-modeler"

_INSTALL_LOCK = threading.Lock()
_CALL_STATE = threading.local()
_ORIGINAL_VISUAL_LOD: Callable[..., object] | None = None
_ORIGINAL_POLYGON_VISUAL_LOD: Callable[..., object] | None = None


@dataclass(frozen=True, slots=True)
class ArchitecturalDetailPlan:
    """Resolved modeler-style secondary architecture for one CWR variant."""

    stairs: bool
    porch: bool
    chimney_count: int
    balcony_count: int
    gutters: bool

    @property
    def enabled(self) -> bool:
        return (
            self.stairs
            or self.porch
            or self.chimney_count > 0
            or self.balcony_count > 0
            or self.gutters
        )


def _unit(key: _pb.BuildingVariantKey, label: str) -> float:
    digest = sha256(
        f"osm-house-modeler:{key.canonical()}:{label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _chance(key: _pb.BuildingVariantKey, label: str, probability: float) -> bool:
    probability = max(0.0, min(1.0, float(probability)))
    return _unit(key, label) < probability


def _style_probabilities(
    key: _pb.BuildingVariantKey,
) -> tuple[float, float, float, float]:
    """Return porch/chimney/balcony/rainwater probabilities.

    The standalone modeler resolves these from its larger country/region schema.
    CWR currently carries a compact 24-region facade catalogue, so use the
    selected regional facade style as a deterministic compatibility layer.
    """

    style = str(key.regional_style or "").casefold()
    timber = any(
        token in style
        for token in (
            "sweden",
            "nord",
            "alpine",
            "timber",
            "wood",
            "cottage",
            "eastern",
        )
    )
    dry = any(
        token in style
        for token in (
            "middle_east",
            "africa",
            "desert",
            "adobe",
            "mediterranean",
        )
    )
    urban = key.family in {"urban", "shop", "school"}

    porch = 0.62 if timber else (0.34 if not urban else 0.08)
    chimney = 0.78 if timber else (0.48 if not dry else 0.18)
    balcony = 0.48 if urban else (
        0.28 if key.family in {"residential", "townhouse"} else 0.04
    )
    rainwater = 0.90 if timber else (0.72 if not dry else 0.24)
    return porch, chimney, balcony, rainwater


def detail_plan_for_key(
    key: _pb.BuildingVariantKey,
    *,
    foundation_depth: float = 0.0,
) -> ArchitecturalDetailPlan:
    """Resolve a repeatable OSM-House-Modeler-style detail plan.

    ``regional_style == "default"`` intentionally stays byte-compatible with
    old direct/manual CWR variants and tests. Real world generation normally
    resolves a concrete regional facade style for variants that use this pass.
    """

    style = str(key.regional_style or "").strip().casefold()
    if style in {"", "default"} or key.family == "church":
        return ArchitecturalDetailPlan(False, False, 0, 0, False)

    porch_p, chimney_p, balcony_p, rain_p = _style_probabilities(key)
    pedestrian = not (
        key.family in {"industrial", "agricultural"}
        or (
            key.family == "outbuilding"
            and _pb._outbuilding_is_garage(key)
        )
    )

    # CWR's enterable variants already have collision-aware terrain stairs. Do
    # not place a second visual staircase over them.
    stairs = (
        not key.interiors
        and pedestrian
        and foundation_depth >= 0.18
        and key.width_m >= 2.8
    )
    porch = (
        pedestrian
        and key.family in {"residential", "townhouse"}
        and key.width_m >= 4.8
        and _chance(key, "porch", porch_p)
    )
    chimney_count = int(
        key.family in {"residential", "townhouse"}
        and key.roof_style not in {"flat", "dome", "onion"}
        and _chance(key, "chimney", chimney_p)
    )
    if (
        chimney_count
        and key.width_m >= 12.0
        and _chance(key, "chimney-second", 0.18)
    ):
        chimney_count = 2

    # A balcony without a corresponding collision/opening treatment is a lie in
    # an enterable P3D. Keep balconies on exterior-only variants until a balcony
    # door is also represented in Geometry/Memory/animation LODs.
    balcony_count = 0
    if (
        not key.interiors
        and key.family in {"residential", "townhouse", "urban"}
        and key.height_m >= 5.5
        and key.width_m >= 5.5
        and _chance(key, "balcony", balcony_p)
    ):
        balcony_count = 1 + int(
            key.height_m >= 9.0 and _chance(key, "balcony-second", 0.22)
        )

    gutters = (
        key.roof_style not in {"flat", "dome", "onion"}
        and key.family != "industrial"
        and _chance(key, "rainwater", rain_p)
    )
    return ArchitecturalDetailPlan(
        stairs,
        porch,
        chimney_count,
        balcony_count,
        gutters,
    )


def _normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    value = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(part * part for part in value))
    if length <= 1.0e-10:
        return (0.0, 1.0, 0.0)
    return tuple(part / length for part in value)


def _add_quad(
    points: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
    faces: list[_pb._Face],
    coordinates: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    texture: str,
    *,
    u_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    start = len(points)
    points.extend(coordinates)
    normal_index = len(normals)
    normals.append(_normal(coordinates[0], coordinates[1], coordinates[2]))
    face = _pb._Face(
        texture,
        (
            (start + 0, normal_index, 0.0, v_scale),
            (start + 1, normal_index, 0.0, 0.0),
            (start + 2, normal_index, u_scale, 0.0),
            (start + 3, normal_index, u_scale, v_scale),
        ),
    )
    faces.extend(_pb._double_sided_faces((face,)))


def _add_box(
    points: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
    faces: list[_pb._Face],
    *,
    center: tuple[float, float],
    axis_width: tuple[float, float],
    axis_depth: tuple[float, float],
    width: float,
    depth: float,
    y0: float,
    y1: float,
    texture: str,
) -> None:
    if y1 <= y0 + 1.0e-6:
        return
    wx, wz = axis_width
    dx, dz = axis_depth
    wl = math.hypot(wx, wz) or 1.0
    dl = math.hypot(dx, dz) or 1.0
    wx, wz = wx / wl, wz / wl
    dx, dz = dx / dl, dz / dl
    half_w = max(0.01, width) * 0.5
    half_d = max(0.01, depth) * 0.5
    corners = (
        (
            center[0] - wx * half_w - dx * half_d,
            center[1] - wz * half_w - dz * half_d,
        ),
        (
            center[0] + wx * half_w - dx * half_d,
            center[1] + wz * half_w - dz * half_d,
        ),
        (
            center[0] + wx * half_w + dx * half_d,
            center[1] + wz * half_w + dz * half_d,
        ),
        (
            center[0] - wx * half_w + dx * half_d,
            center[1] - wz * half_w + dz * half_d,
        ),
    )

    bottom = tuple((x, y0, z) for x, z in corners)
    top = tuple((x, y1, z) for x, z in corners)
    _add_quad(
        points,
        normals,
        faces,
        (bottom[0], bottom[3], bottom[2], bottom[1]),
        texture,
        u_scale=width,
        v_scale=depth,
    )
    _add_quad(
        points,
        normals,
        faces,
        (top[0], top[1], top[2], top[3]),
        texture,
        u_scale=width,
        v_scale=depth,
    )
    _add_quad(
        points,
        normals,
        faces,
        (bottom[0], bottom[1], top[1], top[0]),
        texture,
        u_scale=width,
        v_scale=y1 - y0,
    )
    _add_quad(
        points,
        normals,
        faces,
        (bottom[1], bottom[2], top[2], top[1]),
        texture,
        u_scale=depth,
        v_scale=y1 - y0,
    )
    _add_quad(
        points,
        normals,
        faces,
        (bottom[2], bottom[3], top[3], top[2]),
        texture,
        u_scale=width,
        v_scale=y1 - y0,
    )
    _add_quad(
        points,
        normals,
        faces,
        (bottom[3], bottom[0], top[0], top[3]),
        texture,
        u_scale=depth,
        v_scale=y1 - y0,
    )


def _front_frame_rectangular(
    key: _pb.BuildingVariantKey,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
]:
    # Existing CWR rectangular variants define the entrance on -Z.
    return (
        (0.0, -key.length_m * 0.5),
        (1.0, 0.0),
        (0.0, -1.0),
        key.width_m,
    )


def _front_frame_polygon(
    key: _pb.BuildingVariantKey,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
] | None:
    ring = tuple(key.footprint_vertices)
    if len(ring) < 3:
        return None
    edge = _pb._polygon_native_front_edge(key)
    if not 0 <= edge < len(ring):
        return None
    a = ring[edge]
    b = ring[(edge + 1) % len(ring)]
    vx, vz = b[0] - a[0], b[1] - a[1]
    length = math.hypot(vx, vz)
    if length <= 1.0e-6:
        return None
    tangent = (vx / length, vz / length)
    # Native outer rings are canonicalized CCW, so the exterior is on the right.
    outward = (tangent[1], -tangent[0])
    fraction = max(0.08, min(0.92, float(key.entrance_fraction)))
    anchor = (a[0] + vx * fraction, a[1] + vz * fraction)
    return anchor, tangent, outward, length


def _roof_base_y(
    key: _pb.BuildingVariantKey,
    roof_pitch_degrees: float,
) -> float:
    if key.roof_style == "flat":
        return _pb._main_building_height(key)
    try:
        eave, _rise, _slope = _pb._gabled_profile(key, roof_pitch_degrees)
        return float(eave)
    except (TypeError, ValueError, ZeroDivisionError):
        return max(2.4, _pb._main_building_height(key) - 1.2)


def _append_details(
    lod: _pb._Lod,
    key: _pb.BuildingVariantKey,
    *,
    frame: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        float,
    ] | None,
    wall_texture: str,
    roof_texture: str,
    foundation_texture: str,
    roof_pitch_degrees: float,
    foundation_depth: float,
) -> _pb._Lod:
    plan = detail_plan_for_key(key, foundation_depth=foundation_depth)
    if (
        not plan.enabled
        or frame is None
        or abs(lod.resolution - 1.0) > 1.0e-6
    ):
        return lod

    anchor, tangent, outward, frontage = frame
    points = list(lod.points)
    normals = list(lod.normals)
    faces = list(lod.faces)
    detail_texture = foundation_texture or wall_texture
    eave_y = _roof_base_y(key, roof_pitch_degrees)

    if plan.stairs:
        door_half, _door_height, _pivot = _pb._door_dimensions(key)
        rise = 0.16
        count = max(
            1,
            min(5, int(math.ceil(max(0.18, foundation_depth) / rise))),
        )
        tread = 0.30
        width = min(
            max(door_half * 2.0 + 0.45, 1.35),
            max(1.35, frontage - 0.35),
        )
        for index in range(count):
            d0 = index * tread
            d1 = (index + 1) * tread
            height = (index + 1) * max(0.10, foundation_depth / count)
            center = (
                anchor[0] + outward[0] * ((d0 + d1) * 0.5),
                anchor[1] + outward[1] * ((d0 + d1) * 0.5),
            )
            _add_box(
                points,
                normals,
                faces,
                center=center,
                axis_width=tangent,
                axis_depth=outward,
                width=width,
                depth=tread * 0.98,
                y0=-max(0.08, foundation_depth),
                y1=-max(0.0, foundation_depth - height),
                texture=detail_texture,
            )

    if plan.porch:
        width = min(
            max(2.2, frontage * 0.26),
            max(2.2, frontage - 0.45),
        )
        depth = 1.10
        centre = (
            anchor[0] + outward[0] * depth * 0.5,
            anchor[1] + outward[1] * depth * 0.5,
        )
        # Keep the deck wafer-thin on enterable models so CWR's collision-aware
        # Roadway entrance stairs remain authoritative.
        _add_box(
            points,
            normals,
            faces,
            center=centre,
            axis_width=tangent,
            axis_depth=outward,
            width=width,
            depth=depth,
            y0=0.015,
            y1=0.085,
            texture=wall_texture,
        )
        canopy_y = min(max(2.25, eave_y - 0.55), 2.65)
        _add_box(
            points,
            normals,
            faces,
            center=centre,
            axis_width=tangent,
            axis_depth=outward,
            width=width + 0.12,
            depth=depth + 0.12,
            y0=canopy_y,
            y1=canopy_y + 0.11,
            texture=roof_texture,
        )
        post_offset = max(0.25, width * 0.5 - 0.12)
        front = (
            anchor[0] + outward[0] * (depth - 0.08),
            anchor[1] + outward[1] * (depth - 0.08),
        )
        for sign in (-1.0, 1.0):
            post = (
                front[0] + tangent[0] * sign * post_offset,
                front[1] + tangent[1] * sign * post_offset,
            )
            _add_box(
                points,
                normals,
                faces,
                center=post,
                axis_width=tangent,
                axis_depth=outward,
                width=0.10,
                depth=0.10,
                y0=0.08,
                y1=canopy_y,
                texture=wall_texture,
            )

    if plan.balcony_count:
        floor_height = min(
            3.1,
            max(2.55, eave_y / max(2, int(round(eave_y / 3.0)))),
        )
        for level in range(plan.balcony_count):
            y = floor_height * (level + 1)
            if y > eave_y - 0.55:
                break
            width = min(
                max(2.4, frontage * 0.28),
                max(2.4, frontage - 0.55),
            )
            depth = 1.0
            centre = (
                anchor[0] + outward[0] * depth * 0.5,
                anchor[1] + outward[1] * depth * 0.5,
            )
            _add_box(
                points,
                normals,
                faces,
                center=centre,
                axis_width=tangent,
                axis_depth=outward,
                width=width,
                depth=depth,
                y0=y - 0.11,
                y1=y,
                texture=foundation_texture or wall_texture,
            )
            rail_y0, rail_y1 = y, y + 0.95
            front = (
                anchor[0] + outward[0] * (depth - 0.04),
                anchor[1] + outward[1] * (depth - 0.04),
            )
            _add_box(
                points,
                normals,
                faces,
                center=front,
                axis_width=tangent,
                axis_depth=outward,
                width=width,
                depth=0.055,
                y0=rail_y1 - 0.08,
                y1=rail_y1,
                texture=roof_texture,
            )
            posts = max(3, int(math.ceil(width / 1.2)) + 1)
            for index in range(posts):
                offset = -width * 0.5 + width * index / (posts - 1)
                post = (
                    front[0] + tangent[0] * offset,
                    front[1] + tangent[1] * offset,
                )
                _add_box(
                    points,
                    normals,
                    faces,
                    center=post,
                    axis_width=tangent,
                    axis_depth=outward,
                    width=0.05,
                    depth=0.05,
                    y0=rail_y0,
                    y1=rail_y1,
                    texture=roof_texture,
                )

    if plan.chimney_count:
        for index in range(plan.chimney_count):
            offset = (
                index - (plan.chimney_count - 1) * 0.5
            ) * min(2.4, key.length_m * 0.22)
            centre = (offset, 0.0)
            if key.footprint_vertices:
                shape = _pb._polygon_native_shape(key)
                representative = shape.representative_point()
                centre = (
                    float(representative.x),
                    float(representative.y),
                )
            base_y = max(
                eave_y,
                _pb._main_building_height(key) - 0.35,
            )
            _add_box(
                points,
                normals,
                faces,
                center=centre,
                axis_width=(1.0, 0.0),
                axis_depth=(0.0, 1.0),
                width=0.48,
                depth=0.40,
                y0=base_y,
                y1=base_y + 1.15,
                texture=detail_texture,
            )
            _add_box(
                points,
                normals,
                faces,
                center=centre,
                axis_width=(1.0, 0.0),
                axis_depth=(0.0, 1.0),
                width=0.55,
                depth=0.47,
                y0=base_y + 1.15,
                y1=base_y + 1.23,
                texture=detail_texture,
            )

    if plan.gutters:
        gutter = 0.085
        # Rectangular variants get exact eave gutters. Polygon-native variants
        # get a frontage gutter plus corner downspouts, preserving arbitrary
        # footprint safety without guessing the complete eave topology.
        if not key.footprint_vertices:
            half_width = key.width_m * 0.5
            half_length = key.length_m * 0.5
            for z, depth_axis in (
                (-half_length - gutter * 0.45, (0.0, -1.0)),
                (half_length + gutter * 0.45, (0.0, 1.0)),
            ):
                _add_box(
                    points,
                    normals,
                    faces,
                    center=(0.0, z),
                    axis_width=(1.0, 0.0),
                    axis_depth=depth_axis,
                    width=key.width_m + 0.08,
                    depth=gutter,
                    y0=eave_y - gutter * 0.45,
                    y1=eave_y + gutter * 0.45,
                    texture=roof_texture,
                )
            down_positions = (
                (-half_width - 0.04, -half_length - 0.04),
                (half_width + 0.04, half_length + 0.04),
            )
        else:
            gutter_center = (
                anchor[0] + outward[0] * gutter * 0.45,
                anchor[1] + outward[1] * gutter * 0.45,
            )
            _add_box(
                points,
                normals,
                faces,
                center=gutter_center,
                axis_width=tangent,
                axis_depth=outward,
                width=max(0.5, frontage),
                depth=gutter,
                y0=eave_y - gutter * 0.45,
                y1=eave_y + gutter * 0.45,
                texture=roof_texture,
            )
            half = frontage * 0.5
            down_positions = tuple(
                (
                    anchor[0]
                    + tangent[0] * sign * max(0.0, half - 0.06)
                    + outward[0] * 0.04,
                    anchor[1]
                    + tangent[1] * sign * max(0.0, half - 0.06)
                    + outward[1] * 0.04,
                )
                for sign in (-1.0, 1.0)
            )
        for position in down_positions:
            _add_box(
                points,
                normals,
                faces,
                center=position,
                axis_width=tangent,
                axis_depth=outward,
                width=0.075,
                depth=0.075,
                y0=-0.04,
                y1=eave_y,
                texture=roof_texture,
            )

    return _pb._Lod(
        tuple(points),
        tuple(normals),
        tuple(faces),
        lod.resolution,
        lod.mass_per_point,
        lod.selections,
        lod.properties,
    )


def _visual_lod_wrapper(*args, **kwargs):
    original = _ORIGINAL_VISUAL_LOD
    if original is None:
        raise RuntimeError("OSM House Modeler building adapter is not installed")
    depth = int(getattr(_CALL_STATE, "depth", 0))
    if depth:
        return original(*args, **kwargs)
    _CALL_STATE.depth = depth + 1
    try:
        lod = original(*args, **kwargs)
    finally:
        _CALL_STATE.depth = depth

    key = args[0]
    wall_texture = args[1]
    roof_texture = args[2]
    roof_pitch = float(args[3])
    foundation_texture = kwargs.get("foundation_texture") or wall_texture
    foundation_depth = float(kwargs.get("foundation_depth", 0.0) or 0.0)
    return _append_details(
        lod,
        key,
        frame=_front_frame_rectangular(key),
        wall_texture=wall_texture,
        roof_texture=roof_texture,
        foundation_texture=foundation_texture,
        roof_pitch_degrees=roof_pitch,
        foundation_depth=foundation_depth,
    )


def _polygon_visual_lod_wrapper(*args, **kwargs):
    original = _ORIGINAL_POLYGON_VISUAL_LOD
    if original is None:
        raise RuntimeError("OSM House Modeler polygon adapter is not installed")
    polygon_depth = int(getattr(_CALL_STATE, "polygon_depth", 0))
    if polygon_depth:
        return original(*args, **kwargs)
    normal_depth = int(getattr(_CALL_STATE, "depth", 0))
    _CALL_STATE.polygon_depth = polygon_depth + 1
    _CALL_STATE.depth = normal_depth + 1
    try:
        lod = original(*args, **kwargs)
    finally:
        _CALL_STATE.polygon_depth = polygon_depth
        _CALL_STATE.depth = normal_depth

    key = args[0]
    wall_texture = args[1]
    roof_texture = args[2]
    roof_pitch = float(kwargs.get("roof_pitch_degrees", 35.0) or 35.0)
    foundation_texture = kwargs.get("foundation_texture") or wall_texture
    foundation_depth = float(kwargs.get("foundation_depth", 0.0) or 0.0)
    return _append_details(
        lod,
        key,
        frame=_front_frame_polygon(key),
        wall_texture=wall_texture,
        roof_texture=roof_texture,
        foundation_texture=foundation_texture,
        roof_pitch_degrees=roof_pitch,
        foundation_depth=foundation_depth,
    )


def install_osm_house_modeler_upgrade() -> None:
    """Install the modeler-style architecture pass once per interpreter."""

    global _ORIGINAL_VISUAL_LOD, _ORIGINAL_POLYGON_VISUAL_LOD
    with _INSTALL_LOCK:
        if getattr(_pb, "_osm_house_modeler_upgrade_installed", False):
            return
        _ORIGINAL_VISUAL_LOD = _pb._visual_lod
        _pb._visual_lod = _visual_lod_wrapper
        polygon = getattr(_pb, "_polygon_native_visual_lod", None)
        if polygon is not None:
            _ORIGINAL_POLYGON_VISUAL_LOD = polygon
            _pb._polygon_native_visual_lod = _polygon_visual_lod_wrapper
        _pb._osm_house_modeler_upgrade_installed = True
