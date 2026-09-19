# SPDX-License-Identifier: GPL-3.0-or-later
"""Mixed building-preset transport and runtime delegation.

The historical house_style_preset field remains the public transport so saved
profiles and CLI callers do not need a schema migration. Composite selections
use a compact building-multi: encoding and may include procedural presets plus
one or more stock/mod catalogue presets.
"""
from __future__ import annotations

from collections import Counter
from hashlib import blake2s, sha256
from pathlib import Path
import json
from typing import Callable, Mapping, Sequence

from .procedural_buildings import BuildingGenerationResult, BuildingPlacement
from . import stock_building_policy as stock

BUILDING_MULTI_PREFIX = "building-multi:"
PROCEDURAL_TOKEN_PREFIX = "proc:"
PROCEDURAL_AUTOMATIC_PRESET = "auto"


def _stock_extensions():
    from . import stock_building_extensions as ext
    return ext


def building_preset_selection(value: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return canonical (procedural presets, stock/mod presets)."""
    ext = _stock_extensions()
    text = str(value or "").strip()
    folded = text.casefold()

    stock_only = ext.stock_building_preset_ids(folded)
    if stock_only:
        return (), stock_only

    if not folded.startswith(BUILDING_MULTI_PREFIX):
        return ((folded or PROCEDURAL_AUTOMATIC_PRESET),), ()

    raw = text[len(BUILDING_MULTI_PREFIX):]
    procedural: list[str] = []
    stock_presets: list[str] = []
    unknown: list[str] = []
    for token in (item.strip() for item in raw.split(",")):
        if not token:
            continue
        lowered = token.casefold()
        if lowered.startswith(PROCEDURAL_TOKEN_PREFIX):
            preset = token[len(PROCEDURAL_TOKEN_PREFIX):].strip()
            if not preset:
                raise ValueError("procedural building preset token is empty")
            folded_preset = preset.casefold()
            if folded_preset not in procedural:
                procedural.append(folded_preset)
            continue
        if lowered in ext.STOCK_BUILDING_PRESETS:
            if lowered not in stock_presets:
                stock_presets.append(lowered)
            continue
        unknown.append(token)
    if unknown:
        raise ValueError("unknown building preset(s): " + ", ".join(unknown))
    if not procedural and not stock_presets:
        raise ValueError("building-multi requires at least one building preset")

    ordered_stock = tuple(
        identifier for identifier in ext.STOCK_BUILDING_PRESETS
        if identifier in stock_presets
    )
    return tuple(procedural), ordered_stock


def encode_building_presets(
    procedural_presets: Sequence[str] = (),
    stock_presets: Sequence[str] = (),
) -> str:
    """Encode a heterogeneous preset selection without breaking single values."""
    ext = _stock_extensions()
    procedural: list[str] = []
    for value in procedural_presets:
        preset = str(value or "").strip().casefold()
        if preset and preset not in procedural:
            procedural.append(preset)

    stock_requested = {
        str(value or "").strip().casefold()
        for value in stock_presets
        if str(value or "").strip()
    }
    unknown_stock = sorted(stock_requested.difference(ext.STOCK_BUILDING_PRESETS))
    if unknown_stock:
        raise ValueError("unknown stock building preset(s): " + ", ".join(unknown_stock))
    ordered_stock = tuple(
        identifier for identifier in ext.STOCK_BUILDING_PRESETS
        if identifier in stock_requested
    )

    if procedural and not ordered_stock and len(procedural) == 1:
        return procedural[0]
    if ordered_stock and not procedural:
        return ext.encode_stock_building_presets(ordered_stock)
    if not procedural and not ordered_stock:
        return PROCEDURAL_AUTOMATIC_PRESET

    tokens = [PROCEDURAL_TOKEN_PREFIX + preset for preset in procedural]
    tokens.extend(ordered_stock)
    return BUILDING_MULTI_PREFIX + ",".join(tokens)


def contains_stock_building_presets(value: object) -> bool:
    try:
        _procedural, stock_presets = building_preset_selection(value)
    except ValueError:
        return False
    return bool(stock_presets)


class MixedBuildingLibrary:
    """Delegate each building deterministically across selected preset families."""

    def __init__(
        self,
        *,
        procedural_factory: Callable[..., object],
        procedural_presets: Sequence[str],
        stock_presets: Sequence[str],
        encoded_preset: str,
        **kwargs,
    ) -> None:
        ext = _stock_extensions()
        self.house_style_preset = encoded_preset
        self._procedural_presets = tuple(procedural_presets)
        self._stock_presets = tuple(stock_presets)
        self._procedural_contexts: dict[str, dict[str, object]] = {}
        self._procedural = None
        self._stock = None

        if self._procedural_presets:
            proc_kwargs = dict(kwargs)
            proc_kwargs["house_style_preset"] = self._procedural_presets[0]
            self._procedural = procedural_factory(**proc_kwargs)
        if self._stock_presets:
            stock_kwargs = dict(kwargs)
            stock_kwargs["house_style_preset"] = ext.encode_stock_building_presets(
                self._stock_presets
            )
            self._stock = stock.StockBuildingLibrary(**stock_kwargs)

        if self._procedural is None and self._stock is None:
            raise ValueError("mixed building library requires at least one preset")

    def __getattr__(self, name: str):
        procedural = self.__dict__.get("_procedural")
        if procedural is not None and hasattr(procedural, name):
            return getattr(procedural, name)
        stock_library = self.__dict__.get("_stock")
        if stock_library is not None and hasattr(stock_library, name):
            return getattr(stock_library, name)
        raise AttributeError(name)

    @property
    def cache_dir(self):
        for child in (self._procedural, self._stock):
            if child is not None:
                return getattr(child, "cache_dir", None)
        return None

    @cache_dir.setter
    def cache_dir(self, value) -> None:
        for child in (self.__dict__.get("_procedural"), self.__dict__.get("_stock")):
            if child is not None:
                child.cache_dir = value

    @property
    def cache_enabled(self) -> bool:
        for child in (self._procedural, self._stock):
            if child is not None:
                return bool(getattr(child, "cache_enabled", True))
        return True

    @cache_enabled.setter
    def cache_enabled(self, value: bool) -> None:
        for child in (self.__dict__.get("_procedural"), self.__dict__.get("_stock")):
            if child is not None:
                child.cache_enabled = bool(value)

    @property
    def cache_refresh(self) -> bool:
        for child in (self._procedural, self._stock):
            if child is not None:
                return bool(getattr(child, "cache_refresh", False))
        return False

    @cache_refresh.setter
    def cache_refresh(self, value: bool) -> None:
        for child in (self.__dict__.get("_procedural"), self.__dict__.get("_stock")):
            if child is not None:
                child.cache_refresh = bool(value)

    @property
    def cache_hits(self) -> int:
        return sum(
            int(getattr(child, "cache_hits", 0))
            for child in (self._procedural, self._stock)
            if child is not None
        )

    @cache_hits.setter
    def cache_hits(self, value: int) -> None:
        for child in (self.__dict__.get("_procedural"), self.__dict__.get("_stock")):
            if child is not None:
                child.cache_hits = int(value)

    @property
    def cache_misses(self) -> int:
        return sum(
            int(getattr(child, "cache_misses", 0))
            for child in (self._procedural, self._stock)
            if child is not None
        )

    @cache_misses.setter
    def cache_misses(self, value: int) -> None:
        for child in (self.__dict__.get("_procedural"), self.__dict__.get("_stock")):
            if child is not None:
                child.cache_misses = int(value)

    def _capture_procedural_context(self) -> dict[str, object]:
        assert self._procedural is not None
        return {
            name: getattr(self._procedural, name, None)
            for name in (
                "region_identifier",
                "country_style_identifier",
                "detected_house_style_identifier",
                "house_style_identifier",
            )
        }

    def _activate_procedural_preset(self, preset: str) -> None:
        assert self._procedural is not None
        self._procedural.house_style_preset = preset
        for name, value in self._procedural_contexts.get(preset, {}).items():
            setattr(self._procedural, name, value)

    def prepare(self, dataset, projection, point_building_footprint: float) -> None:
        if self._stock is not None:
            self._stock.prepare(dataset, projection, point_building_footprint)

        if self._procedural is None:
            return

        if len(self._procedural_presets) == 1:
            preset = self._procedural_presets[0]
            self._procedural.house_style_preset = preset
            self._procedural.prepare(dataset, projection, point_building_footprint)
            self._procedural_contexts[preset] = self._capture_procedural_context()
            return

        merged_mapping = {}
        merged_request_counts = Counter()
        for preset in self._procedural_presets:
            self._procedural.house_style_preset = preset
            self._procedural.prepare(dataset, projection, point_building_footprint)
            self._procedural_contexts[preset] = self._capture_procedural_context()
            merged_mapping.update(getattr(self._procedural, "_mapping", {}))
            merged_request_counts.update(
                getattr(self._procedural, "_request_counts", Counter())
            )

        self._procedural._mapping = merged_mapping
        self._procedural._request_counts = merged_request_counts
        self._procedural._selection_cache.clear()
        self._procedural._usage.clear()
        self._procedural._polygon_native_keys.clear()
        self._procedural._prepared = True
        self._activate_procedural_preset(self._procedural_presets[0])

    def _sources(self) -> tuple[tuple[str, str], ...]:
        result = [("procedural", preset) for preset in self._procedural_presets]
        if self._stock is not None:
            result.append(("stock", "stock"))
        return tuple(result)

    def _choose_source(self, payload: Mapping[str, object]) -> tuple[str, str]:
        sources = self._sources()
        if len(sources) == 1:
            return sources[0]
        encoded = json.dumps(
            {
                "preset": self.house_style_preset,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = blake2s(encoded.encode("utf-8"), digest_size=4).digest()
        return sources[int.from_bytes(digest, "little") % len(sources)]

    def plan_polygon(
        self,
        tags,
        points,
        *,
        holes=(),
        road_point=None,
        entrance_point=None,
        allow_native_polygon: bool = True,
    ) -> BuildingPlacement:
        kind, preset = self._choose_source({
            "kind": "polygon",
            "tags": dict(tags),
            "points": tuple(tuple(point) for point in points),
            "holes": tuple(tuple(tuple(point) for point in ring) for ring in holes),
        })
        if kind == "stock":
            assert self._stock is not None
            return self._stock.plan_polygon(
                tags,
                points,
                holes=holes,
                road_point=road_point,
                entrance_point=entrance_point,
                allow_native_polygon=allow_native_polygon,
            )
        assert self._procedural is not None
        self._activate_procedural_preset(preset)
        return self._procedural.plan_polygon(
            tags,
            points,
            holes=holes,
            road_point=road_point,
            entrance_point=entrance_point,
            allow_native_polygon=allow_native_polygon,
        )

    def plan_point(
        self,
        tags,
        footprint_m: float,
        heading_degrees: float,
        *,
        x: float | None = None,
        z: float | None = None,
        road_point=None,
    ) -> BuildingPlacement:
        kind, preset = self._choose_source({
            "kind": "point",
            "tags": dict(tags),
            "footprint_m": float(footprint_m),
            "heading_degrees": float(heading_degrees),
            "x": x,
            "z": z,
        })
        if kind == "stock":
            assert self._stock is not None
            return self._stock.plan_point(
                tags,
                footprint_m,
                heading_degrees,
                x=x or 0.0,
                z=z or 0.0,
                road_point=road_point,
            )
        assert self._procedural is not None
        self._activate_procedural_preset(preset)
        return self._procedural.plan_point(
            tags,
            footprint_m,
            heading_degrees,
            x=x,
            z=z,
            road_point=road_point,
        )

    def register_placement(
        self,
        placement: BuildingPlacement,
        *,
        foundation_depth_m: float | None = None,
    ) -> BuildingPlacement:
        if self._procedural is not None and self._procedural.is_generated_model(
            placement.model_path
        ):
            return self._procedural.register_placement(
                placement, foundation_depth_m=foundation_depth_m
            )
        if self._stock is None:
            raise ValueError(
                f"building placement {placement.model_path!r} has no owning preset"
            )
        return self._stock.register_placement(
            placement, foundation_depth_m=foundation_depth_m
        )

    def place_polygon(self, tags, points, *, road_point=None) -> BuildingPlacement:
        return self.register_placement(
            self.plan_polygon(tags, points, road_point=road_point)
        )

    def place_point(
        self,
        tags,
        footprint_m: float,
        heading_degrees: float,
        *,
        x: float | None = None,
        z: float | None = None,
        road_point=None,
    ) -> BuildingPlacement:
        return self.register_placement(
            self.plan_point(
                tags,
                footprint_m,
                heading_degrees,
                x=x,
                z=z,
                road_point=road_point,
            )
        )

    def model_path(self, key) -> str:
        if hasattr(key, "stock_model_path"):
            assert self._stock is not None
            return self._stock.model_path(key)
        assert self._procedural is not None
        return self._procedural.model_path(key)

    def is_generated_model(self, model_path: str) -> bool:
        return bool(
            self._procedural is not None
            and self._procedural.is_generated_model(model_path)
        )

    def origin_lift_for_model(self, model_path: str) -> float:
        if self._stock is None:
            return 0.0
        return float(self._stock.origin_lift_for_model(model_path))

    def write_assets(
        self,
        source_dir: Path,
        catalogue_path: Path,
    ) -> BuildingGenerationResult:
        child_results: list[BuildingGenerationResult] = []
        child_documents: list[dict[str, object]] = []
        temporary_paths: list[Path] = []

        if self._procedural is not None:
            path = catalogue_path.with_name(catalogue_path.stem + "-procedural.json")
            temporary_paths.append(path)
            child_results.append(self._procedural.write_assets(source_dir, path))
            child_documents.append(json.loads(path.read_text(encoding="utf-8")))

        if self._stock is not None:
            path = catalogue_path.with_name(catalogue_path.stem + "-stock.json")
            temporary_paths.append(path)
            child_results.append(self._stock.write_assets(source_dir, path))
            child_documents.append(json.loads(path.read_text(encoding="utf-8")))

        placements = sum(result.placements for result in child_results)
        generated_variants = sum(result.generated_variants for result in child_results)
        unique_requested = sum(
            result.unique_requested_variants for result in child_results
        )
        reused = sum(result.reused_placements for result in child_results)
        capped = sum(result.capped_variants for result in child_results)
        model_assets = tuple(
            asset
            for result in child_results
            for asset in result.model_assets
        )
        texture_files = tuple(sorted({
            path
            for result in child_results
            for path in result.texture_files
        }))

        models = []
        for document in child_documents:
            models.extend(document.get("models", ()))

        document = {
            "schema": 1,
            "generator": "cwr-worldgen mixed building preset library",
            "mode": self.house_style_preset,
            "procedural_presets": list(self._procedural_presets),
            "stock_presets": list(self._stock_presets),
            "placements": placements,
            "unique_requested_variants": unique_requested,
            "generated_variants": generated_variants,
            "stock_models": sum(
                1 for row in models
                if isinstance(row, Mapping) and bool(row.get("stock"))
            ),
            "reused_placements": reused,
            "reuse_ratio": round(reused / placements, 6) if placements else 0.0,
            "capped_variants": capped,
            "textures": list(texture_files),
            "models": models,
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        document["catalogue_sha256"] = digest
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        for path in temporary_paths:
            try:
                path.unlink()
            except OSError:
                pass

        return BuildingGenerationResult(
            enabled=True,
            placements=placements,
            unique_requested_variants=unique_requested,
            generated_variants=generated_variants,
            reused_placements=reused,
            reuse_ratio=round(reused / placements, 6) if placements else 0.0,
            capped_variants=capped,
            model_assets=model_assets,
            texture_files=texture_files,
            catalogue_sha256=digest,
            cache_hits=sum(result.cache_hits for result in child_results),
            cache_misses=sum(result.cache_misses for result in child_results),
        )


_INSTALLED = False


def install_building_preset_mix() -> None:
    """Allow one build to use procedural and stock/mod building presets together."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import generator, milestone8

    previous_factory = generator.ProceduralBuildingLibrary
    previous_normalise = milestone8.normalise_house_style_preset

    class MixedPresetBuildingLibraryFactory:
        def __new__(cls, *args, **kwargs):
            if args:
                return previous_factory(*args, **kwargs)
            raw = kwargs.get("house_style_preset", PROCEDURAL_AUTOMATIC_PRESET)
            procedural, stock_presets = building_preset_selection(raw)
            encoded = encode_building_presets(procedural, stock_presets)
            mixed = len(procedural) > 1 or bool(procedural and stock_presets)
            if not mixed:
                canonical_kwargs = dict(kwargs)
                canonical_kwargs["house_style_preset"] = encoded
                return previous_factory(**canonical_kwargs)
            child_kwargs = dict(kwargs)
            child_kwargs.pop("house_style_preset", None)
            return MixedBuildingLibrary(
                procedural_factory=previous_factory,
                procedural_presets=procedural,
                stock_presets=stock_presets,
                encoded_preset=encoded,
                **child_kwargs,
            )

    def normalise_mixed(value):
        text = str(value or "").strip()
        if not text.casefold().startswith(BUILDING_MULTI_PREFIX):
            return previous_normalise(value)
        procedural, stock_presets = building_preset_selection(text)
        normalized_procedural = tuple(previous_normalise(preset) for preset in procedural)
        return encode_building_presets(normalized_procedural, stock_presets)

    generator.ProceduralBuildingLibrary = MixedPresetBuildingLibraryFactory
    milestone8.normalise_house_style_preset = normalise_mixed
    _INSTALLED = True
