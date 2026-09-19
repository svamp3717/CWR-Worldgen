# SPDX-License-Identifier: GPL-3.0-or-later
"""Stock CWA/OFP building placement backed by measured model dimensions.

The stock mode deliberately reuses the existing procedural-building planning
interface so terrain grading, road clearance, collision avoidance, and WRP
placement continue to see the final rigid model footprint. It never writes a
procedural P3D: every selected model path points at an original game asset.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s, sha256
from pathlib import Path
import json
import math
from typing import Callable, Mapping, Sequence

from shapely.geometry import Point, Polygon

from .procedural_buildings import BuildingGenerationResult, BuildingPlacement, footprint_from_polygon

STOCK_BUILDING_PRESET = "stock"
STOCK_BUILDING_PRESET_LABEL = "Stock CWA/OFP buildings only"
_STOCK_CATALOGUE_PATH = Path(__file__).with_name("data") / "stock_building_models.json"
_STOCK_MODEL_FIT_TOLERANCE_METRES = 0.25

_FAMILY_FALLBACKS: Mapping[str, tuple[str, ...]] = {
    "residential": ("residential", "townhouse"),
    "townhouse": ("townhouse", "urban", "residential"),
    "urban": ("urban", "townhouse", "residential"),
    "shop": ("shop", "urban", "residential"),
    "school": ("school", "urban"),
    "church": ("church",),
    "industrial": ("industrial", "agricultural"),
    "agricultural": ("agricultural", "industrial", "outbuilding"),
    "outbuilding": ("outbuilding", "agricultural", "residential"),
}

_REVIEWED_CATEGORY_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "residential": ("residential",),
    "commercial": ("shop", "urban"),
    "industrial": ("industrial",),
    "agricultural": ("agricultural", "outbuilding"),
    "civic / public": ("school", "urban"),
    "religious": ("church",),
}


@dataclass(frozen=True, slots=True)
class StockBuildingModel:
    model_path: str
    families: tuple[str, ...]
    categories: tuple[str, ...]
    placement: str
    width_m: float
    length_m: float
    height_m: float
    origin_to_bottom_m: float = 0.0


@dataclass(frozen=True, slots=True)
class StockBuildingKey:
    """Small compatibility key consumed by the existing placement pipeline."""

    family: str
    building_class: str
    outbuilding_kind: str
    width_m: float
    length_m: float
    height_m: float
    foundation_depth_m: float
    stock_model_path: str
    texture_variant: int = 0
    interiors: bool = False
    second_storey: bool = False
    footprint_vertices: tuple[tuple[float, float], ...] = ()
    footprint_holes: tuple[tuple[tuple[float, float], ...], ...] = ()


def _reviewed_families(categories: Sequence[str], placement: str) -> tuple[str, ...]:
    """Map hand-reviewed catalogue categories onto the legacy selector families."""
    result: list[str] = []
    for category in categories:
        for family in _REVIEWED_CATEGORY_FAMILIES.get(str(category).strip().casefold(), ()):
            if family not in result:
                result.append(family)

    # Urban-only residential stock should be preferred as townhouse/city fabric,
    # while Rural-only stock must never leak into town/city selection.
    if "residential" in result and placement == "Urban":
        result = ["townhouse", "urban", *result]
    elif "residential" in result and placement == "Both":
        result.extend(family for family in ("townhouse", "urban") if family not in result)
    return tuple(result)


def _load_catalogue(path: Path = _STOCK_CATALOGUE_PATH) -> tuple[StockBuildingModel, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load stock building catalogue: {path}") from exc

    schema = int(document.get("schema", 0))
    if schema not in {1, 5}:
        raise RuntimeError(f"Unsupported stock building catalogue schema in {path}")

    models: list[StockBuildingModel] = []
    for row in document.get("models", ()):
        if not isinstance(row, Mapping):
            continue
        try:
            model_path = str(row["model_path"]).replace("/", "\\").lstrip("\\")
            width = float(row["width_m"])
            length = float(row["length_m"])
            height = float(row["height_m"])
            origin = float(row.get("origin_to_bottom_m", 0.0))
        except (KeyError, TypeError, ValueError):
            continue

        if schema == 5:
            categories = tuple(
                str(value).strip()
                for value in row.get("categories", ())
                if str(value).strip()
            )
            placement = str(row.get("placement", "")).strip()
            if placement not in {"Urban", "Rural", "Both"}:
                continue
            families = _reviewed_families(categories, placement)
        else:
            categories = ()
            placement = ""
            families = tuple(
                str(value).strip().casefold()
                for value in row.get("families", ())
                if str(value).strip()
            )

        if not model_path or min(width, length, height) <= 0.0:
            continue
        models.append(
            StockBuildingModel(
                model_path,
                families,
                categories,
                placement,
                width,
                length,
                height,
                origin,
            )
        )

    if not models:
        raise RuntimeError(f"Stock building catalogue contains no usable models: {path}")
    return tuple(models)


def _target_height(
    tags: Mapping[str, str],
    family: str,
    level_height: float,
) -> float:
    """Use the procedural generator's height semantics for stock model fitting."""
    from . import procedural_buildings as buildings

    return float(buildings._height(tags, family, level_height))


def _classification(tags: Mapping[str, str], width: float, length: float, settlement: str):
    # Use the same live style classifier and settlement adapter as procedural
    # generation. School/worship policies intentionally wrap this classifier.
    from . import osm_house_modeler_styles as styles
    from .osm_house_modeler_full_style import modeler_context

    return styles.classify_building(
        tags,
        width,
        length,
        settlement=modeler_context(settlement),
    )


def _engine_family(
    tags: Mapping[str, str],
    width_m: float,
    length_m: float,
    settlement: str,
) -> str:
    """Use the live procedural family classifier for stock-model selection.

    Stock and procedural buildings should interpret the same OSM footprint the
    same way. Delegating here also means semantic policies that wrap the live
    procedural family classifier (for example school campuses and worship
    buildings) automatically apply to stock presets instead of drifting into a
    second, less capable classification system.
    """
    from . import procedural_buildings as buildings

    return str(
        buildings._family(
            tags,
            width_m,
            length_m,
            settlement_context=settlement,
        )
    )


def _engine_outbuilding_kind(
    tags: Mapping[str, str],
    width_m: float,
    length_m: float,
) -> str:
    """Use the procedural shed/garage inference for stock metadata."""
    from . import procedural_buildings as buildings

    return str(buildings._outbuilding_kind(tags, width_m, length_m))


def _model_orientation_dimensions(
    model: StockBuildingModel,
    *,
    swapped: bool,
) -> tuple[float, float]:
    return (
        (model.length_m, model.width_m)
        if swapped
        else (model.width_m, model.length_m)
    )


def _orientation_fits_target(
    model: StockBuildingModel,
    target_width: float,
    target_length: float,
    *,
    swapped: bool,
    tolerance: float = _STOCK_MODEL_FIT_TOLERANCE_METRES,
) -> bool:
    width, length = _model_orientation_dimensions(model, swapped=swapped)
    return (
        width <= max(0.1, float(target_width)) + max(0.0, float(tolerance))
        and length <= max(0.1, float(target_length)) + max(0.0, float(tolerance))
    )


def _model_support_polygon(
    centre_x: float,
    centre_z: float,
    model: StockBuildingModel,
    heading_degrees: float,
) -> tuple[tuple[float, float], ...]:
    half_width = max(0.05, float(model.width_m) * 0.5)
    half_length = max(0.05, float(model.length_m) * 0.5)
    angle = math.radians(float(heading_degrees))
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    return tuple(
        (
            centre_x
            + width_sign * half_width * width_axis[0]
            + length_sign * half_length * length_axis[0],
            centre_z
            + width_sign * half_width * width_axis[1]
            + length_sign * half_length * length_axis[1],
        )
        for width_sign, length_sign in (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        )
    )


def _dimension_score(
    model: StockBuildingModel,
    target_width: float,
    target_length: float,
    target_height: float,
    *,
    swapped: bool,
) -> float:
    width = model.length_m if swapped else model.width_m
    length = model.width_m if swapped else model.length_m
    target_width = max(0.5, target_width)
    target_length = max(0.5, target_length)
    target_height = max(1.0, target_height)
    score = abs(math.log(width / target_width)) + abs(math.log(length / target_length))
    score += 0.20 * abs(math.log(max(1.0, model.height_m) / target_height))
    model_area = width * length
    target_area = target_width * target_length
    score += 0.20 * abs(math.log(model_area / max(1.0, target_area)))
    # A slightly smaller stock model is usually harmless. Oversized rigid models
    # are much more likely to eat roads or neighboring buildings, so punish them.
    for actual, requested in ((width, target_width), (length, target_length)):
        ratio = actual / max(0.5, requested)
        if ratio > 1.25:
            score += (ratio - 1.25) * 1.8
    return score


class StockBuildingLibrary:
    """Drop-in placement library selecting measured original-game buildings."""

    def __init__(
        self,
        *,
        world_name: str,
        width_quantum: float = 2.0,
        length_quantum: float = 2.0,
        height_quantum: float = 3.0,
        minimum_width: float = 4.0,
        maximum_width: float = 80.0,
        minimum_length: float = 4.0,
        maximum_length: float = 160.0,
        minimum_height: float = 3.0,
        maximum_height: float = 48.0,
        default_level_height: float = 3.0,
        maximum_variants: int = 128,
        roof_pitch_degrees: float = 35.0,
        foundation_depth: float = 0.5,
        maximum_foundation_depth: float = 8.0,
        foundation_depth_quantum: float = 0.25,
        church_plinth_height: float = 0.0,
        generate_interiors: bool = False,
        high_quality_textures: bool = False,
        texture_variants: int = 10,
        house_style_preset: str = STOCK_BUILDING_PRESET,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        cache_refresh: bool = False,
        **_ignored,
    ) -> None:
        self.world_name = world_name
        self.width_quantum = width_quantum
        self.length_quantum = length_quantum
        self.height_quantum = height_quantum
        self.minimum_width = minimum_width
        self.maximum_width = maximum_width
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        self.minimum_height = minimum_height
        self.maximum_height = maximum_height
        self.default_level_height = default_level_height
        self.maximum_variants = maximum_variants
        self.roof_pitch_degrees = roof_pitch_degrees
        self.foundation_depth = foundation_depth
        self.maximum_foundation_depth = maximum_foundation_depth
        self.foundation_depth_quantum = foundation_depth_quantum
        self.church_plinth_height = church_plinth_height
        self.generate_interiors = False
        self.high_quality_textures = False
        self.texture_variants = texture_variants
        self.house_style_preset = STOCK_BUILDING_PRESET
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.cache_refresh = cache_refresh
        self.models = _load_catalogue()
        self._usage: dict[str, int] = {}
        # Keep the same settlement evidence used by procedural generation.
        # _settlements remains as a tiny compatibility/debug view of place nodes.
        self._settlements: tuple[tuple[float, float, str], ...] = ()
        self._settlement_points: tuple[tuple[float, float, float, str], ...] = ()
        self._settlement_scale_x = 1.0
        self._settlement_scale_z = 1.0
        self._settlement_bucket_size = 1000.0
        self._settlement_buckets: dict[tuple[int, int], tuple[int, ...]] = {}
        self._isolated_dwelling_cabins: tuple[Polygon, ...] = ()

    def prepare(self, dataset, projection, point_building_footprint: float) -> None:
        del point_building_footprint
        # Procedural generation measures settlement radii in source-ground
        # metres, not projected world-space metres. Reuse the same scale-aware
        # 1 km rule for city/town/village/hamlet place nodes.
        self._settlement_scale_x = max(1.0e-9, float(projection.scale_x))
        self._settlement_scale_z = max(1.0e-9, float(projection.scale_z))
        settlement_points: list[tuple[float, float, float, str]] = []
        for feature in getattr(dataset, "places", ()):
            kind = str(getattr(feature, "tags", {}).get("place", "")).casefold()
            if kind not in {"city", "town", "village", "hamlet"}:
                continue
            try:
                x, z = projection.to_world(feature.point)
            except Exception:
                continue
            settlement_points.append((float(x), float(z), 1000.0, kind))
        self._settlement_points = tuple(settlement_points)
        self._settlements = tuple((x, z, kind) for x, z, _radius, kind in settlement_points)

        self._settlement_bucket_size = max(
            1.0,
            1000.0 * self._settlement_scale_x,
            1000.0 * self._settlement_scale_z,
        )
        mutable_buckets: dict[tuple[int, int], list[int]] = {}
        for index, (centre_x, centre_z, _radius, _kind) in enumerate(self._settlement_points):
            key = (
                math.floor(centre_x / self._settlement_bucket_size),
                math.floor(centre_z / self._settlement_bucket_size),
            )
            mutable_buckets.setdefault(key, []).append(index)
        self._settlement_buckets = {
            key: tuple(values) for key, values in mutable_buckets.items()
        }

        # Match procedural generation's exact isolated-dwelling rule: only the
        # lone generic footprint inside a mapped place=isolated_dwelling polygon
        # receives isolated_dwelling_single context.
        building_geometries: list[tuple[Polygon, Mapping[str, str]]] = []
        for building_feature in getattr(dataset, "building_polygons", ()):
            for geo_polygon in getattr(building_feature, "polygons", ()):
                outer = [projection.to_world(point) for point in geo_polygon.outer]
                holes = [
                    [projection.to_world(point) for point in hole]
                    for hole in geo_polygon.holes
                ]
                if len(outer) < 4:
                    continue
                geometry = Polygon(outer, holes)
                if not geometry.is_empty:
                    building_geometries.append((geometry, building_feature.tags))

        isolated_dwelling_cabins: list[Polygon] = []
        for place_feature in getattr(dataset, "place_areas", ()):
            if str(getattr(place_feature, "tags", {}).get("place", "")).casefold() != "isolated_dwelling":
                continue
            for geo_polygon in getattr(place_feature, "polygons", ()):
                outer = [projection.to_world(point) for point in geo_polygon.outer]
                holes = [
                    [projection.to_world(point) for point in hole]
                    for hole in geo_polygon.holes
                ]
                if len(outer) < 4:
                    continue
                area = Polygon(outer, holes)
                if area.is_empty:
                    continue
                inside: list[tuple[Polygon, str]] = []
                for building_geometry, building_tags in building_geometries:
                    if not area.covers(building_geometry.representative_point()):
                        continue
                    inside.append((
                        building_geometry,
                        str(building_tags.get("building", "")).casefold(),
                    ))
                plausible = [
                    geometry
                    for geometry, building_kind in inside
                    if building_kind in {"", "yes"}
                ]
                if len(plausible) == 1:
                    isolated_dwelling_cabins.append(plausible[0])
        self._isolated_dwelling_cabins = tuple(isolated_dwelling_cabins)

    def _settlement_context(self, x: float, z: float) -> str:
        priority = {"city": 0, "town": 1, "village": 2, "hamlet": 3}
        if self._isolated_dwelling_cabins:
            point = Point(float(x), float(z))
            if any(footprint.covers(point) for footprint in self._isolated_dwelling_cabins):
                return "isolated_dwelling_single"

        matches: list[tuple[int, float, str]] = []
        if self._settlement_points:
            bucket_size = self._settlement_bucket_size
            bucket_x = math.floor(float(x) / bucket_size)
            bucket_z = math.floor(float(z) / bucket_size)
            candidate_indices: list[int] = []
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    candidate_indices.extend(
                        self._settlement_buckets.get((bucket_x + dx, bucket_z + dz), ())
                    )
            candidates = (self._settlement_points[index] for index in candidate_indices)
        else:
            # Backward-compatible support for tests/tools that populated the old
            # internal place-node tuple directly.
            candidates = (
                (centre_x, centre_z, 1000.0, kind)
                for centre_x, centre_z, kind in self._settlements
            )

        for centre_x, centre_z, radius, kind in candidates:
            distance = math.hypot(
                (float(x) - centre_x) / self._settlement_scale_x,
                (float(z) - centre_z) / self._settlement_scale_z,
            )
            if distance <= radius:
                matches.append((priority.get(kind, 99), distance, kind))
        if not matches:
            return "rural"
        return min(matches)[2]

    def _candidate_model_groups(
        self,
        family: str,
        settlement: str = "rural",
    ) -> tuple[tuple[str, tuple[StockBuildingModel, ...]], ...]:
        """Return semantic fallback groups inside the correct settlement pool."""
        context = str(settlement or "rural").strip().casefold()
        # Procedural semantics treat village, hamlet, isolated dwellings and
        # ordinary rural space as non-urban. Only explicit town/city context may
        # draw from stock models reviewed as Urban.
        wanted_placement = (
            "Urban"
            if context in {"urban", "town", "city", "town_city"}
            else "Rural"
        )
        eligible = tuple(
            model
            for model in self.models
            if not model.placement or model.placement in {"Both", wanted_placement}
        )
        if not eligible:
            eligible = self.models

        if family == "residential":
            wanted = (
                ("residential",)
                if wanted_placement == "Rural"
                else ("townhouse", "urban", "residential")
            )
        else:
            wanted = _FAMILY_FALLBACKS.get(family, (family, "residential"))

        groups = tuple(
            (wanted_family, candidates)
            for wanted_family in wanted
            if (
                candidates := tuple(
                    model for model in eligible if wanted_family in model.families
                )
            )
        )
        return groups or (("fallback", eligible),)

    def _candidate_models(
        self,
        family: str,
        settlement: str = "rural",
    ) -> tuple[StockBuildingModel, ...]:
        # Retain the historical helper contract for tests/tools that inspect it.
        return self._candidate_model_groups(family, settlement)[0][1]

    def _select(
        self,
        *,
        family: str,
        building_class: str,
        outbuilding_kind: str,
        target_width: float,
        target_length: float,
        target_height: float,
        seed: str,
        settlement: str = "rural",
        fit_predicate: Callable[[StockBuildingModel, bool], bool] | None = None,
    ) -> tuple[StockBuildingKey, bool]:
        groups = self._candidate_model_groups(family, settlement)

        # Prefer the first semantic family that has a model physically small
        # enough for the mapped footprint. This is the important difference from
        # merely penalising oversize models: a 32 m hangar must not beat a 24 m
        # barn on style score when the source footprint is only 20 m wide.
        scored: list[tuple[float, str, StockBuildingModel, bool]] = []
        for _group_name, candidates in groups:
            group_scored: list[tuple[float, str, StockBuildingModel, bool]] = []
            for model in candidates:
                for use_swapped in (False, True):
                    if not _orientation_fits_target(
                        model,
                        target_width,
                        target_length,
                        swapped=use_swapped,
                    ):
                        continue
                    if fit_predicate is not None and not fit_predicate(model, use_swapped):
                        continue
                    score = _dimension_score(
                        model,
                        target_width,
                        target_length,
                        target_height,
                        swapped=use_swapped,
                    )
                    group_scored.append(
                        (score, model.model_path.casefold(), model, use_swapped)
                    )
            if group_scored:
                scored = group_scored
                break

        if not scored:
            # No stock asset can fit entirely inside the requested envelope.
            # Choose the least-overflowing orientation across the semantic
            # fallback chain. Later road/building collision gates may still
            # reject it, but we never knowingly choose a larger alternative.
            fallback: list[
                tuple[float, int, float, str, StockBuildingModel, bool]
            ] = []
            target_w = max(0.1, float(target_width))
            target_l = max(0.1, float(target_length))
            for group_index, (_group_name, candidates) in enumerate(groups):
                for model in candidates:
                    for use_swapped in (False, True):
                        width, length = _model_orientation_dimensions(
                            model, swapped=use_swapped
                        )
                        overflow = max(
                            width / target_w,
                            length / target_l,
                            1.0,
                        )
                        score = _dimension_score(
                            model,
                            target_width,
                            target_length,
                            target_height,
                            swapped=use_swapped,
                        )
                        fallback.append(
                            (
                                overflow,
                                group_index,
                                score,
                                model.model_path.casefold(),
                                model,
                                use_swapped,
                            )
                        )
            fallback.sort(key=lambda item: item[:4])
            choice = fallback[0]
            model, swapped = choice[4], choice[5]
        else:
            scored.sort(key=lambda item: (item[0], item[1], item[3]))
            best_score = scored[0][0]
            shortlist = [
                item for item in scored if item[0] <= best_score + 0.12
            ][:4]
            digest = blake2s(seed.encode("utf-8", "ignore"), digest_size=4).digest()
            choice = shortlist[int.from_bytes(digest, "little") % len(shortlist)]
            model, swapped = choice[2], choice[3]

        key = StockBuildingKey(
            family=family,
            building_class=building_class,
            outbuilding_kind=outbuilding_kind,
            width_m=model.width_m,
            length_m=model.length_m,
            height_m=model.height_m,
            foundation_depth_m=0.0,
            stock_model_path=model.model_path,
        )
        return key, swapped

    def plan_polygon(
        self,
        tags: Mapping[str, str],
        points: Sequence[tuple[float, float]],
        *,
        holes: Sequence[Sequence[tuple[float, float]]] = (),
        road_point=None,
        entrance_point=None,
        allow_native_polygon: bool = True,
    ) -> BuildingPlacement:
        del road_point, entrance_point, allow_native_polygon
        source_shape = Polygon(
            tuple((float(x), float(z)) for x, z in points),
            [tuple((float(x), float(z)) for x, z in ring) for ring in holes if len(ring) >= 3],
        )
        if source_shape.is_empty or source_shape.area <= 0.0:
            source_shape = Polygon(tuple((float(x), float(z)) for x, z in points))
        footprint = footprint_from_polygon(points)
        centre = source_shape.centroid
        centre_x, centre_z = float(centre.x), float(centre.y)
        settlement = self._settlement_context(centre_x, centre_z)
        classification = _classification(tags, footprint.width_m, footprint.length_m, settlement)
        family = _engine_family(
            tags, footprint.width_m, footprint.length_m, settlement
        )
        building_class = str(getattr(classification, "building_class", family))
        outbuilding_kind = str(getattr(classification, "outbuilding_kind", ""))
        if family == "outbuilding":
            outbuilding_kind = _engine_outbuilding_kind(
                tags, footprint.width_m, footprint.length_m
            )
            building_class = outbuilding_kind
        key, swapped = self._select(
            family=family,
            building_class=building_class,
            outbuilding_kind=outbuilding_kind,
            target_width=footprint.width_m,
            target_length=footprint.length_m,
            target_height=_target_height(tags, family, self.default_level_height),
            seed=f"polygon:{centre_x:.2f}:{centre_z:.2f}:{footprint.width_m:.2f}:{footprint.length_m:.2f}:{family}",
            settlement=settlement,
            fit_predicate=lambda model, swapped: source_shape.buffer(
                _STOCK_MODEL_FIT_TOLERANCE_METRES
            ).covers(
                Polygon(
                    _model_support_polygon(
                        centre_x,
                        centre_z,
                        model,
                        (
                            footprint.heading_degrees
                            + (90.0 if swapped else 0.0)
                        )
                        % 360.0,
                    )
                )
            ),
        )
        heading = (footprint.heading_degrees + (90.0 if swapped else 0.0)) % 360.0
        return BuildingPlacement(key.stock_model_path, heading, key, key)

    def place_polygon(self, tags, points, *, road_point=None) -> BuildingPlacement:
        return self.plan_polygon(tags, points, road_point=road_point)

    def plan_point(
        self,
        tags: Mapping[str, str],
        footprint_m: float,
        heading_degrees: float,
        *,
        x: float = 0.0,
        z: float = 0.0,
        road_point=None,
    ) -> BuildingPlacement:
        del road_point
        settlement = self._settlement_context(float(x), float(z))
        classification = _classification(tags, footprint_m, footprint_m, settlement)
        family = _engine_family(tags, footprint_m, footprint_m, settlement)
        building_class = str(getattr(classification, "building_class", family))
        outbuilding_kind = str(getattr(classification, "outbuilding_kind", ""))
        if family == "outbuilding":
            outbuilding_kind = _engine_outbuilding_kind(tags, footprint_m, footprint_m)
            building_class = outbuilding_kind
        key, swapped = self._select(
            family=family,
            building_class=building_class,
            outbuilding_kind=outbuilding_kind,
            target_width=footprint_m,
            target_length=footprint_m,
            target_height=_target_height(tags, family, self.default_level_height),
            seed=f"point:{x:.2f}:{z:.2f}:{footprint_m:.2f}:{family}",
            settlement=settlement,
        )
        heading = (float(heading_degrees) + (90.0 if swapped else 0.0)) % 360.0
        return BuildingPlacement(key.stock_model_path, heading, key, key)

    def place_point(self, tags, footprint_m: float, heading_degrees: float) -> BuildingPlacement:
        return self.plan_point(tags, footprint_m, heading_degrees)

    def model_path(self, key: StockBuildingKey) -> str:
        return key.stock_model_path

    def register_placement(
        self,
        placement: BuildingPlacement,
        *,
        foundation_depth_m: float | None = None,
    ) -> BuildingPlacement:
        # Stock P3Ds cannot grow a generated foundation skirt. Terrain grading is
        # already based on their measured support polygon, so keep the original
        # model and record usage only.
        del foundation_depth_m
        self._usage[placement.model_path] = self._usage.get(placement.model_path, 0) + 1
        return placement

    @staticmethod
    def is_generated_model(model_path: str) -> bool:
        del model_path
        return False

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:
        del source_dir
        records = [
            {
                "model_path": model_path,
                "placements": count,
                "stock": True,
                # The legacy Milestone 8 report predates stock-only mode and only
                # checks for at least three model LODs. Original CWA/OFP building
                # ODOLs satisfy that requirement; no new LOD is authored here.
                "lod_count": 3,
            }
            for model_path, count in sorted(self._usage.items(), key=lambda item: item[0].casefold())
        ]
        placements = sum(self._usage.values())
        reused = max(0, placements - len(records))
        payload = {
            "schema": 1,
            "mode": STOCK_BUILDING_PRESET,
            "generated_models": 0,
            "generated_variants": 0,
            "stock_models": len(records),
            "placements": placements,
            "models": records,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_bytes(encoded)
        return BuildingGenerationResult(
            enabled=True,
            placements=placements,
            unique_requested_variants=len(records),
            generated_variants=0,
            reused_placements=reused,
            reuse_ratio=(reused / placements) if placements else 0.0,
            capped_variants=0,
            model_assets=(),
            texture_files=(),
            catalogue_sha256=sha256(encoded).hexdigest(),
            cache_hits=0,
            cache_misses=0,
        )


def _install_cli_and_validation() -> None:
    from . import cli, milestone8

    original_normalise = milestone8.normalise_house_style_preset

    def normalise_with_stock(value):
        if str(value or "").strip().casefold() == STOCK_BUILDING_PRESET:
            return STOCK_BUILDING_PRESET
        return original_normalise(value)

    milestone8.normalise_house_style_preset = normalise_with_stock
    if STOCK_BUILDING_PRESET not in cli.HOUSE_STYLE_PRESET_IDENTIFIERS:
        cli.HOUSE_STYLE_PRESET_IDENTIFIERS = (
            *tuple(cli.HOUSE_STYLE_PRESET_IDENTIFIERS),
            STOCK_BUILDING_PRESET,
        )


def _install_gui_option() -> None:
    from . import gui_entry

    original_configure = gui_entry._configure_gui

    def configure_gui(gui, base_dir: Path) -> None:
        original_configure(gui, base_dir)
        options = tuple(getattr(gui, "HOUSE_STYLE_PRESET_OPTIONS", ()))
        labels = tuple(getattr(gui, "HOUSE_STYLE_PRESET_LABELS", ()))
        if not any(identifier == STOCK_BUILDING_PRESET for identifier, _label in options):
            gui.HOUSE_STYLE_PRESET_OPTIONS = (*options, (STOCK_BUILDING_PRESET, STOCK_BUILDING_PRESET_LABEL))
        if STOCK_BUILDING_PRESET_LABEL not in labels:
            gui.HOUSE_STYLE_PRESET_LABELS = (*labels, STOCK_BUILDING_PRESET_LABEL)

        label_to_identifier = dict(getattr(gui, "_HOUSE_STYLE_LABEL_TO_IDENTIFIER", {}))
        identifier_to_label = dict(getattr(gui, "_HOUSE_STYLE_IDENTIFIER_TO_LABEL", {}))
        label_to_identifier[STOCK_BUILDING_PRESET_LABEL] = STOCK_BUILDING_PRESET
        identifier_to_label[STOCK_BUILDING_PRESET] = STOCK_BUILDING_PRESET_LABEL
        gui._HOUSE_STYLE_LABEL_TO_IDENTIFIER = label_to_identifier
        gui._HOUSE_STYLE_IDENTIFIER_TO_LABEL = identifier_to_label

        original_identifier = gui.gui_house_style_preset_identifier
        original_label = gui.gui_house_style_preset_label

        def stock_identifier(value: object) -> str:
            text = str(value or "").strip()
            if text == STOCK_BUILDING_PRESET_LABEL or text.casefold() == STOCK_BUILDING_PRESET:
                return STOCK_BUILDING_PRESET
            return original_identifier(value)

        def stock_label(value: object) -> str:
            if str(value or "").strip().casefold() == STOCK_BUILDING_PRESET:
                return STOCK_BUILDING_PRESET_LABEL
            return original_label(value)

        gui.gui_house_style_preset_identifier = stock_identifier
        gui.gui_house_style_preset_label = stock_label

    gui_entry._configure_gui = configure_gui


def install_stock_building_policy() -> None:
    """Expose stock-only buildings as a building-preset choice.

    The generator still asks for a ``ProceduralBuildingLibrary`` because that is
    its established planning interface. A tiny factory intercepts only the
    ``stock`` preset and returns :class:`StockBuildingLibrary`; every other preset
    continues to instantiate the original procedural implementation unchanged.
    """
    from . import generator

    original_library = generator.ProceduralBuildingLibrary

    class BuildingLibraryFactory:
        def __new__(cls, *args, **kwargs):
            preset = str(kwargs.get("house_style_preset", "") or "").strip().casefold()
            if preset == STOCK_BUILDING_PRESET:
                return StockBuildingLibrary(*args, **kwargs)
            return original_library(*args, **kwargs)

    generator.ProceduralBuildingLibrary = BuildingLibraryFactory
    _install_cli_and_validation()
    _install_gui_option()
