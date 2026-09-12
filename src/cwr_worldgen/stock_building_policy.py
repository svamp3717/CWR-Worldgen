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
from typing import Mapping, Sequence

from .procedural_buildings import BuildingGenerationResult, BuildingPlacement, footprint_from_polygon

STOCK_BUILDING_PRESET = "stock"
STOCK_BUILDING_PRESET_LABEL = "Stock CWA/OFP buildings only"
_STOCK_CATALOGUE_PATH = Path(__file__).with_name("data") / "stock_building_models.json"

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

_DEFAULT_HEIGHTS: Mapping[str, float] = {
    "residential": 6.0,
    "townhouse": 8.0,
    "urban": 12.0,
    "shop": 4.0,
    "school": 8.0,
    "church": 14.0,
    "industrial": 8.0,
    "agricultural": 7.0,
    "outbuilding": 3.5,
}

_SETTLEMENT_RADIUS_M: Mapping[str, float] = {
    "city": 1600.0,
    "town": 900.0,
    "village": 450.0,
    "hamlet": 260.0,
}


@dataclass(frozen=True, slots=True)
class StockBuildingModel:
    model_path: str
    families: tuple[str, ...]
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


def _load_catalogue(path: Path = _STOCK_CATALOGUE_PATH) -> tuple[StockBuildingModel, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load stock building catalogue: {path}") from exc
    if int(document.get("schema", 0)) != 1:
        raise RuntimeError(f"Unsupported stock building catalogue schema in {path}")
    models: list[StockBuildingModel] = []
    for row in document.get("models", ()):
        if not isinstance(row, Mapping):
            continue
        try:
            model_path = str(row["model_path"]).replace("/", "\\").lstrip("\\")
            families = tuple(str(value).strip().casefold() for value in row["families"] if str(value).strip())
            width = float(row["width_m"])
            length = float(row["length_m"])
            height = float(row["height_m"])
            origin = float(row.get("origin_to_bottom_m", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not model_path or not families or min(width, length, height) <= 0.0:
            continue
        models.append(StockBuildingModel(model_path, families, width, length, height, origin))
    if not models:
        raise RuntimeError(f"Stock building catalogue contains no usable models: {path}")
    return tuple(models)


def _parse_number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    token = []
    seen_digit = False
    for char in text:
        if char.isdigit() or char in ".-+":
            token.append(char)
            seen_digit = seen_digit or char.isdigit()
        elif seen_digit:
            break
    try:
        result = float("".join(token))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _target_height(tags: Mapping[str, str], family: str) -> float:
    explicit = _parse_number(tags.get("height"))
    if explicit is not None and explicit > 0.5:
        return explicit
    levels = _parse_number(tags.get("building:levels"))
    if levels is not None and levels > 0.0:
        return max(2.5, levels * 3.0)
    return _DEFAULT_HEIGHTS.get(family, 6.0)


def _classification(tags: Mapping[str, str], width: float, length: float, settlement: str):
    # Use the live classifier because school/worship policies intentionally wrap
    # it during package initialization.
    from . import osm_house_modeler_styles as styles

    return styles.classify_building(
        tags,
        width,
        length,
        settlement=settlement,
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
        self._settlements: tuple[tuple[float, float, str], ...] = ()

    def prepare(self, dataset, projection, point_building_footprint: float) -> None:
        del point_building_footprint
        settlements: list[tuple[float, float, str]] = []
        for feature in getattr(dataset, "places", ()):
            kind = str(getattr(feature, "tags", {}).get("place", "")).casefold()
            if kind not in _SETTLEMENT_RADIUS_M:
                continue
            try:
                x, z = projection.to_world(feature.point)
            except Exception:
                continue
            settlements.append((float(x), float(z), kind))
        self._settlements = tuple(settlements)

    def _settlement_context(self, x: float, z: float) -> str:
        best: tuple[float, str] | None = None
        for sx, sz, kind in self._settlements:
            radius = _SETTLEMENT_RADIUS_M[kind]
            distance = math.hypot(x - sx, z - sz)
            if distance > radius:
                continue
            candidate = (distance / radius, kind)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return "rural"
        return "city" if best[1] == "city" else "town" if best[1] == "town" else "village"

    def _candidate_models(self, family: str) -> tuple[StockBuildingModel, ...]:
        wanted = _FAMILY_FALLBACKS.get(family, (family, "residential"))
        for wanted_family in wanted:
            candidates = tuple(model for model in self.models if wanted_family in model.families)
            if candidates:
                return candidates
        return self.models

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
    ) -> tuple[StockBuildingKey, bool]:
        scored: list[tuple[float, str, StockBuildingModel, bool]] = []
        for model in self._candidate_models(family):
            direct = _dimension_score(model, target_width, target_length, target_height, swapped=False)
            swapped = _dimension_score(model, target_width, target_length, target_height, swapped=True)
            use_swapped = swapped + 1.0e-9 < direct
            score = swapped if use_swapped else direct
            scored.append((score, model.model_path.casefold(), model, use_swapped))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_score = scored[0][0]
        shortlist = [item for item in scored if item[0] <= best_score + 0.12][:4]
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
        del holes, road_point, entrance_point, allow_native_polygon
        footprint = footprint_from_polygon(points)
        centre_x = sum(float(point[0]) for point in points) / max(1, len(points))
        centre_z = sum(float(point[1]) for point in points) / max(1, len(points))
        settlement = self._settlement_context(centre_x, centre_z)
        classification = _classification(tags, footprint.width_m, footprint.length_m, settlement)
        family = str(classification.family)
        key, swapped = self._select(
            family=family,
            building_class=str(getattr(classification, "building_class", family)),
            outbuilding_kind=str(getattr(classification, "outbuilding_kind", "")),
            target_width=footprint.width_m,
            target_length=footprint.length_m,
            target_height=_target_height(tags, family),
            seed=f"polygon:{centre_x:.2f}:{centre_z:.2f}:{footprint.width_m:.2f}:{footprint.length_m:.2f}:{family}",
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
        family = str(classification.family)
        key, swapped = self._select(
            family=family,
            building_class=str(getattr(classification, "building_class", family)),
            outbuilding_kind=str(getattr(classification, "outbuilding_kind", "")),
            target_width=footprint_m,
            target_length=footprint_m,
            target_height=_target_height(tags, family),
            seed=f"point:{x:.2f}:{z:.2f}:{footprint_m:.2f}:{family}",
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
            {"model_path": model_path, "placements": count}
            for model_path, count in sorted(self._usage.items(), key=lambda item: item[0].casefold())
        ]
        payload = {
            "schema": 1,
            "mode": STOCK_BUILDING_PRESET,
            "generated_models": 0,
            "placements": sum(self._usage.values()),
            "models": records,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_bytes(encoded)
        return BuildingGenerationResult(
            enabled=True,
            placements=sum(self._usage.values()),
            unique_requested_variants=len(records),
            generated_variants=0,
            reused_placements=0,
            reuse_ratio=0.0,
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
