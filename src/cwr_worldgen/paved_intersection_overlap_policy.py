# SPDX-License-Identifier: GPL-3.0-or-later
"""Cover fallback paved intersections with one edge-free generated hub.

The stock T/X models cannot represent every skewed road node.  When fitting one
of those junctions fails, the ordinary road fitter restores a short straight cap
at the node.  A fixed stock approach can also overshoot its planned stop point by
several metres.  Layering those rectangular models cannot hide every shoulder:
whichever slab is on top leaves a bright edge across the other road.

This final road pass replaces only that fallback cap with a generated T/Y/X hub.
The hub is a single union polygon with an opaque, edge-free paved texture, placed
slightly above the source-aligned approaches.  It masks their central overhangs
without adding another crossing rectangle.  Native stock T/X junctions, dirt and
gravel are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from . import final_building_road_clearance_policy as _clearance
from . import generator as _generator
from . import gravel_family_policy as _gravel
from . import paved_junction_policy as _paved
from . import playability as _p
from . import procedural_infrastructure as _pi
from . import road_quality_policy as _quality


_NODE_BUCKET_METRES = 1.0
_NODE_MATCH_METRES = 0.75
_CAP_UNDERLAY_DROP_METRES = 0.006
_PAVED_JUNCTION_ARM_EXTENT_METRES = 6.25
_PAVED_JUNCTION_SURFACE_OFFSET_METRES = (
    _p._STOCK_ROAD_VERTICAL_OFFSET_METRES + 0.002
)
_PAVED_JUNCTION_PLACEMENT_OFFSET_METRES = (
    _PAVED_JUNCTION_SURFACE_OFFSET_METRES
    - _pi.GENERATED_GRAVEL_VISUAL_TOP_METRES
)
_SHORT_STRAIGHT = re.compile(r"(?<!\d)6\.p3d$", re.IGNORECASE)
_VARIANT = r"t(?:30|45|60|75)[lr]|t90|y120|x(?:30|45|60|75|90)"
_PAVED_JUNCTION_MODEL = re.compile(
    rf"^paved_j(?P<degree>[34])_(?P<family>sil|asf|kos)_"
    rf"(?P<variant>{_VARIANT})\.p3d$",
    re.IGNORECASE,
)
_PAVED_JUNCTION_SUBTYPE = re.compile(
    rf"^paved_j(?P<degree>[34])_(?P<family>sil|asf|kos)_"
    rf"(?P<variant>{_VARIANT})$",
    re.IGNORECASE,
)

_ORIGINAL_FIT = None
_ORIGINAL_ROAD_LODS = None
_ORIGINAL_REGISTER_MODEL_USAGE = None
_ORIGINAL_TEXTURE_KIND = None
_INSTALLED = False


def _normalise_model_path(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


def _bucket(point: tuple[float, float]) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / _NODE_BUCKET_METRES),
        math.floor(float(point[1]) / _NODE_BUCKET_METRES),
    )


def _nearby_objects(buckets, point: tuple[float, float]):
    bx, bz = _bucket(point)
    for nx in range(bx - 1, bx + 2):
        for nz in range(bz - 1, bz + 2):
            yield from buckets.get((nx, nz), ())


def _ordinary_short_paved_family(obj) -> str | None:
    path = _normalise_model_path(obj.model_path)
    filename = path.rsplit("\\", 1)[-1]
    if _SHORT_STRAIGHT.search(filename) is None:
        return None
    family = _paved._family(path)
    if family is None or _paved._kind(family) != "paved":
        return None
    return family


def paved_junction_model_path(
    world_name: str,
    degree: int,
    family: str,
    variant: str,
) -> str:
    family = str(family).casefold()
    variant = str(variant).casefold()
    expected_degree = 4 if variant.startswith("x") else 3
    if degree != expected_degree or family not in _paved._WIDTH:
        raise ValueError("invalid generated paved junction family or degree")
    _gravel.gravel_junction_template_headings(variant)
    return rf"{world_name}\i\paved_j{degree}_{family}_{variant}.p3d"


def is_generated_paved_junction_model(model_path: str) -> bool:
    filename = str(model_path).replace("/", "\\").rsplit("\\", 1)[-1]
    return _PAVED_JUNCTION_MODEL.fullmatch(filename) is not None


def _paved_junction_lods(key, texture: str):
    match = _PAVED_JUNCTION_SUBTYPE.fullmatch(str(key.subtype))
    if match is None:
        raise ValueError(f"invalid generated paved junction subtype: {key.subtype}")
    degree = int(match.group("degree"))
    variant = match.group("variant").casefold()
    if (4 if variant.startswith("x") else 3) != degree:
        raise ValueError("generated paved junction degree does not match variant")

    half_width = max(1.8, float(key.width_m) * 0.5)
    polygon = _gravel._junction_polygon(
        variant,
        half_width,
        arm_extent=_PAVED_JUNCTION_ARM_EXTENT_METRES,
        core_radius=max(1.65, half_width * 0.72),
    )
    visual = _gravel._triangulated_lod(
        polygon,
        y=_pi.GENERATED_GRAVEL_VISUAL_TOP_METRES,
        texture=texture,
        resolution=_pi._VISUAL_LOD,
    )
    boundary = tuple(
        (float(x), 0.0, float(z))
        for x, z in tuple(polygon.exterior.coords)[:-1]
    )
    map_geometry = _pi._Lod(
        boundary,
        (),
        (),
        _pi._GEOMETRY_LOD,
        properties=(("map", "road"),),
    )
    roadway = _gravel._triangulated_lod(
        polygon,
        y=_pi.GENERATED_GRAVEL_ROADWAY_HEIGHT_METRES,
        texture="",
        resolution=_pi._ROADWAY_LOD,
    )
    land = _pi._Lod(boundary, (), (), _pi._LAND_CONTACT_LOD)
    return visual, map_geometry, roadway, land


def _road_lods(key, texture: str):
    if _PAVED_JUNCTION_SUBTYPE.fullmatch(str(key.subtype)) is not None:
        return _paved_junction_lods(key, texture)
    if _ORIGINAL_ROAD_LODS is None:
        raise RuntimeError("paved intersection asset policy is not installed")
    return _ORIGINAL_ROAD_LODS(key, texture)


def _register_model_usage(self, model_path: str, count: int = 1) -> None:
    count = max(0, int(count))
    if count == 0:
        return
    filename = str(model_path).replace("/", "\\").rsplit("\\", 1)[-1]
    match = _PAVED_JUNCTION_MODEL.fullmatch(filename)
    if match is not None and self.is_generated_model(model_path):
        family = match.group("family").casefold()
        self._usage[_pi.InfrastructureModelKey(
            "road",
            filename[:-4].casefold(),
            int(round(_paved._WIDTH[family] * 20.0)),
            int(round(_PAVED_JUNCTION_ARM_EXTENT_METRES * 20.0)),
        )] += count
        return
    if _ORIGINAL_REGISTER_MODEL_USAGE is None:
        raise RuntimeError("paved intersection asset policy is not installed")
    _ORIGINAL_REGISTER_MODEL_USAGE(self, model_path, count)


def _infrastructure_texture_kind(key) -> str:
    if key.kind == "road":
        match = _PAVED_JUNCTION_SUBTYPE.fullmatch(str(key.subtype))
        if match is not None:
            return f"paved_junction_{match.group('family').casefold()}"
    if _ORIGINAL_TEXTURE_KIND is None:
        raise RuntimeError("paved intersection asset policy is not installed")
    return _ORIGINAL_TEXTURE_KIND(key)


def _lower_cap(cap, point, elevations, spec):
    length = _quality._piece_length(
        str(cap.model_path),
        float(spec.road_segment_length),
    )
    half = length * 0.5
    direction = _paved._direction(float(cap.heading_degrees))
    start = (
        point[0] - direction[0] * half,
        point[1] - direction[1] * half,
    )
    end = (
        point[0] + direction[0] * half,
        point[1] + direction[1] * half,
    )
    lowered = _p._road_object_on_slope(
        int(cap.object_id),
        str(cap.model_path),
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            - _CAP_UNDERLAY_DROP_METRES
        ),
    )
    return replace(
        lowered,
        x=float(point[0]),
        z=float(point[1]),
        heading_degrees=float(cap.heading_degrees) % 360.0,
    )


def _generated_hub(cap, plan, family: str, elevations, spec):
    generated_family = _paved._junction_family(family)
    world_name = str(getattr(spec, "name", "")).strip()
    if generated_family not in _paved._WIDTH or not world_name:
        return _lower_cap(cap, plan.point, elevations, spec)

    variant, axis = _gravel.gravel_junction_variant_for_directions(
        arm.source_direction for arm in plan.arms
    )
    degree = len(plan.arms)
    model_path = paved_junction_model_path(
        world_name,
        degree,
        generated_family,
        variant,
    )
    start = (
        plan.point[0] - axis[0] * _PAVED_JUNCTION_ARM_EXTENT_METRES,
        plan.point[1] - axis[1] * _PAVED_JUNCTION_ARM_EXTENT_METRES,
    )
    end = (
        plan.point[0] + axis[0] * _PAVED_JUNCTION_ARM_EXTENT_METRES,
        plan.point[1] + axis[1] * _PAVED_JUNCTION_ARM_EXTENT_METRES,
    )
    return _p._road_object_on_slope(
        int(cap.object_id),
        model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=_PAVED_JUNCTION_PLACEMENT_OFFSET_METRES,
    )


def finish_paved_intersection_overlaps(
    report,
    plans,
    elevations,
    spec,
):
    """Replace ordinary paved caps left by failed stock-junction plans."""

    objects = list(getattr(report, "objects", ()))
    if not objects or not plans or int(getattr(report, "junction_cap_objects", 0)) <= 0:
        return report

    buckets: dict[tuple[int, int], list[object]] = {}
    for obj in objects:
        buckets.setdefault(_bucket((float(obj.x), float(obj.z))), []).append(obj)

    index_by_id = {
        int(obj.object_id): index
        for index, obj in enumerate(objects)
    }
    cap_count = min(int(report.junction_cap_objects), len(objects))
    cap_ids = {
        int(obj.object_id)
        for obj in objects[:cap_count]
    }
    replaced_ids: set[int] = set()
    for key in sorted(plans):
        plan = plans[key]
        nearby = tuple(
            obj
            for obj in _nearby_objects(buckets, plan.point)
            if math.dist((float(obj.x), float(obj.z)), plan.point)
            <= _NODE_MATCH_METRES
        )
        native_model = _normalise_model_path(plan.model_path)
        if any(
            _normalise_model_path(obj.model_path) == native_model
            for obj in nearby
        ):
            continue

        candidates = []
        for obj in nearby:
            family = _ordinary_short_paved_family(obj)
            if (
                family is None
                or int(obj.object_id) not in cap_ids
                or int(obj.object_id) in replaced_ids
            ):
                continue
            candidates.append((
                math.dist((float(obj.x), float(obj.z)), plan.point),
                int(obj.object_id),
                family,
                obj,
            ))
        if not candidates:
            continue

        _distance, object_id, family, cap = min(candidates)
        objects[index_by_id[object_id]] = _generated_hub(
            cap,
            plan,
            family,
            elevations,
            spec,
        )
        replaced_ids.add(object_id)

    if not replaced_ids:
        return report

    return replace(report, objects=tuple(objects))


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("paved intersection overlap policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    if (
        not bool(getattr(spec, "stock_road_piece_fitting", False))
        or int(getattr(report, "junction_cap_objects", 0)) <= 0
    ):
        return report
    plans = _paved._plans(dataset, projection, spec)
    finished = finish_paved_intersection_overlaps(
        report,
        plans,
        elevations,
        spec,
    )
    stored = _clearance._FINAL_ROADS.get()
    if stored is not None and stored[-1] is report and finished is not report:
        _clearance._FINAL_ROADS.set((*stored[:-1], finished))
    return finished


def install_paved_intersection_overlap_policy() -> None:
    """Install the final source-aware paved-intersection surface pass."""

    global _ORIGINAL_FIT, _ORIGINAL_ROAD_LODS
    global _ORIGINAL_REGISTER_MODEL_USAGE, _ORIGINAL_TEXTURE_KIND, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _ORIGINAL_ROAD_LODS = _pi._road_lods
    _ORIGINAL_REGISTER_MODEL_USAGE = (
        _pi.ProceduralInfrastructureLibrary.register_model_usage
    )
    _ORIGINAL_TEXTURE_KIND = _pi._infrastructure_texture_kind
    _pi._road_lods = _road_lods
    _pi.ProceduralInfrastructureLibrary.register_model_usage = _register_model_usage
    _pi._infrastructure_texture_kind = _infrastructure_texture_kind
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
