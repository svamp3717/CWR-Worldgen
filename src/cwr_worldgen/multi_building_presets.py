# SPDX-License-Identifier: GPL-3.0-or-later
"""General multi-select building presets across stock, modded, and procedural sources.

The historical house_style_preset field stays the transport so saved specs and
CLI callers remain compatible. Single presets retain their old representation;
heterogeneous or multi-procedural selections use building-multi:...
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from hashlib import blake2s, sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from .procedural_buildings import BuildingGenerationResult
from . import stock_building_policy as stock
from . import stock_building_extensions as stock_ext

BUILDING_MULTI_PREFIX = "building-multi:"
PROCEDURAL_AUTO_PRESET = "procedural-auto"
PROCEDURAL_AUTO_LABEL = "Procedural automatic (area / country)"
_PROCEDURAL_MULTI_MARKER = "cwr-procedural-multi:"
_INSTALLED = False


@lru_cache(maxsize=1)
def _procedural_options() -> tuple[tuple[str, str], ...]:
    from .building_country_policy import building_country_options

    return (
        (PROCEDURAL_AUTO_PRESET, PROCEDURAL_AUTO_LABEL),
        *building_country_options(),
    )


@lru_cache(maxsize=1)
def _procedural_identifiers() -> tuple[str, ...]:
    return tuple(identifier for identifier, _label in _procedural_options())


@lru_cache(maxsize=1)
def _procedural_json_by_identifier() -> dict[str, str]:
    """Return country preset identifiers mapped to their source JSON filenames."""
    from .osm_house_modeler_styles import discover_country_style_dir

    directory = discover_country_style_dir()
    if directory is None:
        return {}
    result: dict[str, str] = {}
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name.casefold()):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping) or not document.get("iso_alpha2"):
            continue
        identifier = str(document.get("identifier", path.stem)).strip().casefold()
        if identifier:
            result[identifier] = f"country_styles/{path.name}"
    return result


def selected_building_jsons(value: object) -> tuple[str, ...]:
    """Return the concrete catalogue/style JSONs behind an explicit selection."""
    selected = building_preset_ids(value)
    result: list[str] = []
    for identifier in selected:
        if identifier in stock_ext.STOCK_BUILDING_PRESETS:
            path = stock_ext._STOCK_CATALOGUE_BY_PRESET[identifier]
            name = f"data/{path.name}"
        elif identifier == PROCEDURAL_AUTO_PRESET:
            # Automatic is deliberately not one concrete JSON selection.
            continue
        else:
            name = _procedural_json_by_identifier().get(identifier, "")
        if name and name not in result:
            result.append(name)
    return tuple(result)


def building_selection_state(values: Sequence[str]) -> dict[str, object]:
    """Return stable persisted metadata for the visible building checkboxes."""
    encoded = encode_building_presets(values)
    selected = building_preset_ids(encoded)
    return {
        "house_style_preset": encoded,
        "selected_building_presets": list(selected),
        "selected_building_jsons": list(selected_building_jsons(encoded)),
    }


def _annotate_building_catalogue(
    catalogue_path: Path,
    source_dir: Path,
    *,
    preset: object,
) -> str | None:
    """Persist source JSON names in both external and embedded catalogues."""
    try:
        document = json.loads(catalogue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    document["selected_building_jsons"] = list(selected_building_jsons(preset))
    document.pop("catalogue_sha256", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    document["catalogue_sha256"] = digest
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    catalogue_path.write_text(rendered, encoding="utf-8", newline="\n")
    embedded = source_dir / "g" / "buildings.json"
    if embedded.is_file():
        embedded.write_text(rendered, encoding="utf-8", newline="\n")
    return digest


@lru_cache(maxsize=1)
def _canonical_building_order() -> tuple[str, ...]:
    return (*stock_ext.STOCK_BUILDING_PRESETS, *_procedural_identifiers())


@lru_cache(maxsize=512)
def _building_preset_ids_text(text: str) -> tuple[str, ...]:
    if not text or text == "auto":
        return ()

    stock_ids = stock_ext.stock_building_preset_ids(text)
    if stock_ids:
        return stock_ids

    procedural = frozenset(_procedural_identifiers())
    if text in procedural:
        return (text,)
    if not text.startswith(BUILDING_MULTI_PREFIX):
        return ()

    raw = text[len(BUILDING_MULTI_PREFIX):]
    requested = {item.strip().casefold() for item in raw.split(",") if item.strip()}
    if not requested:
        raise ValueError("building-multi requires at least one building preset")
    known = frozenset(_canonical_building_order())
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError("unknown building preset(s): " + ", ".join(unknown))
    return tuple(identifier for identifier in _canonical_building_order() if identifier in requested)


def building_preset_ids(value: object) -> tuple[str, ...]:
    """Return canonical selected preset IDs, or an empty tuple for automatic mode."""
    text = str(value or "").strip().casefold()
    return _building_preset_ids_text(text)


def encode_building_presets(values: Sequence[str]) -> str:
    """Encode selected preset IDs while preserving legacy single/stock encodings."""
    requested = {str(value).strip().casefold() for value in values if str(value).strip()}
    known = frozenset(_canonical_building_order())
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError("unknown building preset(s): " + ", ".join(unknown))
    ordered = tuple(identifier for identifier in _canonical_building_order() if identifier in requested)
    if not ordered:
        return "auto"
    if ordered == (PROCEDURAL_AUTO_PRESET,):
        # Keep an explicitly checked automatic-procedural source distinct from
        # the legacy "no checkboxes selected" automatic fallback. Generation
        # normalizes both to the same runtime behavior, but GUI profiles can now
        # round-trip the checkbox state faithfully.
        return PROCEDURAL_AUTO_PRESET
    if all(identifier in stock_ext.STOCK_BUILDING_PRESETS for identifier in ordered):
        return stock_ext.encode_stock_building_presets(ordered)
    if len(ordered) == 1:
        return ordered[0]
    return BUILDING_MULTI_PREFIX + ",".join(ordered)


def _split_presets(value: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected = building_preset_ids(value)
    stock_ids = tuple(identifier for identifier in selected if identifier in stock_ext.STOCK_BUILDING_PRESETS)
    procedural_ids = tuple(identifier for identifier in selected if identifier not in stock_ext.STOCK_BUILDING_PRESETS)
    return stock_ids, procedural_ids


def has_stock_building_presets(value: object) -> bool:
    return bool(_split_presets(value)[0])


def _procedural_transport(values: Sequence[str]) -> str:
    values = tuple(values)
    if not values or values == (PROCEDURAL_AUTO_PRESET,):
        return "auto"
    return encode_building_presets(values)


def _stable_bucket(parts: Sequence[object], modulo: int) -> int:
    payload = json.dumps(
        parts,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8", "ignore")
    return int.from_bytes(blake2s(payload, digest_size=8).digest(), "little") % max(1, modulo)


class MultiBuildingLibrary:
    """Delegate each mapped building to either the selected P3D or procedural pool."""

    def __init__(
        self,
        *,
        stock_library: stock.StockBuildingLibrary,
        procedural_library,
        house_style_preset: str,
        stock_presets: Sequence[str],
        procedural_presets: Sequence[str],
    ) -> None:
        self.stock_library = stock_library
        self.procedural_library = procedural_library
        self.house_style_preset = house_style_preset
        self.stock_presets = tuple(stock_presets)
        self.procedural_presets = tuple(procedural_presets)

    def __getattr__(self, name: str):
        # The generator reads the procedural library's quantization/budget fields
        # directly. Keep that mature interface rather than mirroring dozens of
        # attributes and eventually forgetting the one added next Tuesday.
        #
        # Pickle probes special attributes before instance state has been restored.
        # Avoid recursively looking up procedural_library during that window.
        procedural_library = self.__dict__.get("procedural_library")
        if procedural_library is None:
            raise AttributeError(name)
        return getattr(procedural_library, name)

    @property
    def cache_dir(self):
        return getattr(self.procedural_library, "cache_dir", None)

    @cache_dir.setter
    def cache_dir(self, value) -> None:
        self.procedural_library.cache_dir = value
        self.stock_library.cache_dir = value

    @property
    def cache_enabled(self) -> bool:
        return bool(getattr(self.procedural_library, "cache_enabled", True))

    @cache_enabled.setter
    def cache_enabled(self, value: bool) -> None:
        self.procedural_library.cache_enabled = bool(value)
        self.stock_library.cache_enabled = bool(value)

    @property
    def cache_refresh(self) -> bool:
        return bool(getattr(self.procedural_library, "cache_refresh", False))

    @cache_refresh.setter
    def cache_refresh(self, value: bool) -> None:
        self.procedural_library.cache_refresh = bool(value)
        self.stock_library.cache_refresh = bool(value)

    @property
    def cache_hits(self) -> int:
        return int(getattr(self.procedural_library, "cache_hits", 0)) + int(
            getattr(self.stock_library, "cache_hits", 0)
        )

    @cache_hits.setter
    def cache_hits(self, value: int) -> None:
        self.procedural_library.cache_hits = int(value)
        self.stock_library.cache_hits = int(value)

    @property
    def cache_misses(self) -> int:
        return int(getattr(self.procedural_library, "cache_misses", 0)) + int(
            getattr(self.stock_library, "cache_misses", 0)
        )

    @cache_misses.setter
    def cache_misses(self, value: int) -> None:
        self.procedural_library.cache_misses = int(value)
        self.stock_library.cache_misses = int(value)

    @staticmethod
    def _tag_signature(tags: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(key), str(value)) for key, value in tags.items()))

    def _choose_stock(self, signature: Sequence[object]) -> bool:
        # Source groups are weighted equally. Catalogue-combination checkboxes often
        # overlap heavily, so weighting by the raw number of checked stock sets would
        # make selecting a convenience union accidentally dominate procedural output.
        return _stable_bucket(signature, 2) == 0

    def prepare(self, dataset, projection, point_building_footprint: float) -> None:
        self.stock_library.prepare(dataset, projection, point_building_footprint)
        self.procedural_library.prepare(dataset, projection, point_building_footprint)

    def plan_polygon(self, tags, points, **kwargs):
        holes = kwargs.get("holes", ()) or ()
        signature = (
            "polygon",
            self._tag_signature(tags),
            tuple((round(float(x), 4), round(float(z), 4)) for x, z in points),
            tuple(
                tuple((round(float(x), 4), round(float(z), 4)) for x, z in ring)
                for ring in holes
            ),
        )
        library = self.stock_library if self._choose_stock(signature) else self.procedural_library
        return library.plan_polygon(tags, points, **kwargs)

    def place_polygon(self, tags, points, **kwargs):
        return self.register_placement(self.plan_polygon(tags, points, **kwargs))

    def plan_point(self, tags, footprint_m, heading_degrees, **kwargs):
        signature = (
            "point",
            self._tag_signature(tags),
            round(float(footprint_m), 4),
            round(float(kwargs.get("x", 0.0) or 0.0), 4),
            round(float(kwargs.get("z", 0.0) or 0.0), 4),
        )
        library = self.stock_library if self._choose_stock(signature) else self.procedural_library
        return library.plan_point(tags, footprint_m, heading_degrees, **kwargs)

    def place_point(self, tags, footprint_m, heading_degrees, **kwargs):
        return self.register_placement(
            self.plan_point(tags, footprint_m, heading_degrees, **kwargs)
        )

    def model_path(self, key) -> str:
        if hasattr(key, "stock_model_path"):
            return self.stock_library.model_path(key)
        return self.procedural_library.model_path(key)

    def register_placement(self, placement, *, foundation_depth_m=None):
        if self.procedural_library.is_generated_model(placement.model_path):
            return self.procedural_library.register_placement(
                placement,
                foundation_depth_m=foundation_depth_m,
            )
        return self.stock_library.register_placement(
            placement,
            foundation_depth_m=foundation_depth_m,
        )

    def is_generated_model(self, model_path: str) -> bool:
        return bool(self.procedural_library.is_generated_model(model_path))

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:
        procedural_result = self.procedural_library.write_assets(source_dir, catalogue_path)
        stock_catalogue = catalogue_path.with_name(catalogue_path.stem + "-stock.json")
        stock_result = self.stock_library.write_assets(source_dir, stock_catalogue)

        try:
            procedural_document = json.loads(catalogue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            procedural_document = {}
        try:
            stock_document = json.loads(stock_catalogue.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stock_document = {}
        try:
            stock_catalogue.unlink()
        except OSError:
            pass

        document = dict(procedural_document)
        placements = procedural_result.placements + stock_result.placements
        reused = procedural_result.reused_placements + stock_result.reused_placements
        unique_requested = (
            procedural_result.unique_requested_variants
            + stock_result.unique_requested_variants
        )
        capped = procedural_result.capped_variants + stock_result.capped_variants
        document["mode"] = self.house_style_preset
        document["house_style_preset"] = self.house_style_preset
        document["selected_stock_presets"] = list(self.stock_presets)
        document["selected_procedural_presets"] = list(self.procedural_presets)
        document["selected_building_jsons"] = list(
            selected_building_jsons(self.house_style_preset)
        )
        document["stock_models"] = len(stock_document.get("models", ()))
        document["placements"] = placements
        document["unique_requested_variants"] = unique_requested
        document["generated_variants"] = procedural_result.generated_variants
        document["reused_placements"] = reused
        document["reuse_ratio"] = round(reused / placements, 6) if placements else 0.0
        document["capped_variants"] = capped
        document["models"] = [
            *list(procedural_document.get("models", ())),
            *list(stock_document.get("models", ())),
        ]
        document.pop("catalogue_sha256", None)
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        document["catalogue_sha256"] = digest
        rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_text(rendered, encoding="utf-8", newline="\n")
        embedded_catalogue = source_dir / "g" / "buildings.json"
        embedded_catalogue.parent.mkdir(parents=True, exist_ok=True)
        embedded_catalogue.write_text(rendered, encoding="utf-8", newline="\n")

        return BuildingGenerationResult(
            enabled=procedural_result.enabled or stock_result.enabled,
            placements=placements,
            unique_requested_variants=unique_requested,
            generated_variants=procedural_result.generated_variants,
            reused_placements=reused,
            reuse_ratio=(reused / placements) if placements else 0.0,
            capped_variants=capped,
            model_assets=procedural_result.model_assets,
            texture_files=procedural_result.texture_files,
            catalogue_sha256=digest,
            cache_hits=procedural_result.cache_hits + stock_result.cache_hits,
            cache_misses=procedural_result.cache_misses + stock_result.cache_misses,
        )


def _install_transport_and_factory() -> None:
    from . import generator, milestone8, procedural_buildings as buildings
    from . import osm_house_modeler_runtime as runtime

    previous_spec_normalise = milestone8.normalise_house_style_preset
    previous_building_normalise = buildings.normalise_house_style_preset
    previous_building_profile = buildings.house_style_preset_profile
    previous_runtime_profile = runtime.house_style_preset_profile
    previous_factory = generator.ProceduralBuildingLibrary

    def normalise_spec(value):
        selected = building_preset_ids(value)
        raw_multi = str(value or "").strip().casefold().startswith(BUILDING_MULTI_PREFIX)
        if len(selected) > 1 or (raw_multi and selected):
            return encode_building_presets(selected)
        if selected == (PROCEDURAL_AUTO_PRESET,):
            return "auto"
        return previous_spec_normalise(value)

    def normalise_procedural(value):
        stock_ids, procedural_ids = _split_presets(value)
        raw_multi = str(value or "").strip().casefold().startswith(BUILDING_MULTI_PREFIX)
        if procedural_ids and not stock_ids and (
            len(procedural_ids) > 1 or raw_multi
        ):
            return _procedural_transport(procedural_ids)
        if procedural_ids == (PROCEDURAL_AUTO_PRESET,) and not stock_ids:
            return "auto"
        return previous_building_normalise(value)

    def profile_for_multi(value):
        stock_ids, procedural_ids = _split_presets(value)
        if not stock_ids and len(procedural_ids) > 1:
            return None
        if procedural_ids == (PROCEDURAL_AUTO_PRESET,) and not stock_ids:
            return None
        return previous_building_profile(value)

    def runtime_profile_for_multi(value):
        stock_ids, procedural_ids = _split_presets(value)
        if not stock_ids and len(procedural_ids) > 1:
            return None
        if procedural_ids == (PROCEDURAL_AUTO_PRESET,) and not stock_ids:
            return None
        return previous_runtime_profile(value)

    milestone8.normalise_house_style_preset = normalise_spec
    buildings.normalise_house_style_preset = normalise_procedural
    buildings.house_style_preset_profile = profile_for_multi
    runtime.house_style_preset_profile = runtime_profile_for_multi

    class MultiPresetBuildingLibraryFactory:
        def __new__(cls, *args, **kwargs):
            preset = str(kwargs.get("house_style_preset", "") or "").strip().casefold()
            stock_ids, procedural_ids = _split_presets(preset)
            if preset.startswith(BUILDING_MULTI_PREFIX) and not (
                stock_ids and procedural_ids
            ):
                # Validation returns a canonical value but intentionally does not
                # mutate the caller's frozen spec. Canonicalize again at the actual
                # routing boundary so generalized stock-only composites become
                # stock-multi, while multi-procedural composites keep their
                # building-multi transport.
                canonical_kwargs = dict(kwargs)
                canonical_kwargs["house_style_preset"] = encode_building_presets(
                    (*stock_ids, *procedural_ids)
                )
                return previous_factory(*args, **canonical_kwargs)
            if not (stock_ids and procedural_ids):
                return previous_factory(*args, **kwargs)

            stock_kwargs = dict(kwargs)
            stock_kwargs["house_style_preset"] = stock_ext.encode_stock_building_presets(stock_ids)
            procedural_kwargs = dict(kwargs)
            procedural_kwargs["house_style_preset"] = _procedural_transport(procedural_ids)
            stock_library = stock.StockBuildingLibrary(*args, **stock_kwargs)
            procedural_library = previous_factory(*args, **procedural_kwargs)
            return MultiBuildingLibrary(
                stock_library=stock_library,
                procedural_library=procedural_library,
                house_style_preset=encode_building_presets((*stock_ids, *procedural_ids)),
                stock_presets=stock_ids,
                procedural_presets=procedural_ids,
            )

    generator.ProceduralBuildingLibrary = MultiPresetBuildingLibraryFactory


def _install_multi_procedural_runtime() -> None:
    from . import osm_house_modeler_runtime as runtime

    previous_regional_preset = runtime._regional_preset
    previous_resolve_style = runtime.resolve_style

    def regional_preset(library) -> str:
        stock_ids, procedural_ids = _split_presets(
            getattr(library, "house_style_preset", "auto")
        )
        if not stock_ids and len(procedural_ids) > 1:
            return _PROCEDURAL_MULTI_MARKER + ",".join(procedural_ids)
        if procedural_ids == (PROCEDURAL_AUTO_PRESET,) and not stock_ids:
            return "auto"
        return previous_regional_preset(library)

    def resolve_style_multi(*args, **kwargs):
        marker = str(kwargs.get("regional_preset", "") or "")
        if not marker.startswith(_PROCEDURAL_MULTI_MARKER):
            return previous_resolve_style(*args, **kwargs)

        values = tuple(
            item.strip().casefold()
            for item in marker[len(_PROCEDURAL_MULTI_MARKER):].split(",")
            if item.strip()
        )
        if not values:
            revised = dict(kwargs)
            revised["regional_preset"] = "auto"
            return previous_resolve_style(*args, **revised)

        tags = kwargs.get("tags") or {}
        signature = (
            tuple(sorted((str(key), str(value)) for key, value in tags.items())),
            round(float(kwargs.get("latitude", 0.0) or 0.0), 6),
            round(float(kwargs.get("longitude", 0.0) or 0.0), 6),
            round(float(kwargs.get("width_m", 0.0) or 0.0), 3),
            round(float(kwargs.get("length_m", 0.0) or 0.0), 3),
            str(kwargs.get("settlement_context", "")),
            str(kwargs.get("seed", "")),
        )
        selected = values[_stable_bucket(signature, len(values))]
        revised = dict(kwargs)
        revised["regional_preset"] = (
            "auto"
            if selected == PROCEDURAL_AUTO_PRESET
            else "country:" + selected
        )
        return previous_resolve_style(*args, **revised)

    runtime._regional_preset = regional_preset
    runtime.resolve_style = resolve_style_multi

    from . import procedural_buildings as buildings

    previous_write_assets = buildings.ProceduralBuildingLibrary.write_assets

    def write_assets_with_selected_jsons(self, source_dir: Path, catalogue_path: Path):
        result = previous_write_assets(self, source_dir, catalogue_path)
        digest = _annotate_building_catalogue(
            catalogue_path,
            source_dir,
            preset=getattr(self, "house_style_preset", "auto"),
        )
        return replace(result, catalogue_sha256=digest) if digest else result

    buildings.ProceduralBuildingLibrary.write_assets = write_assets_with_selected_jsons

    previous_stock_write_assets = stock.StockBuildingLibrary.write_assets

    def stock_write_assets_with_selected_jsons(self, source_dir: Path, catalogue_path: Path):
        result = previous_stock_write_assets(self, source_dir, catalogue_path)
        digest = _annotate_building_catalogue(
            catalogue_path,
            source_dir,
            preset=getattr(self, "house_style_preset", stock.STOCK_BUILDING_PRESET),
        )
        return replace(result, catalogue_sha256=digest) if digest else result

    stock.StockBuildingLibrary.write_assets = stock_write_assets_with_selected_jsons


def _checkbox_key(identifier: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in identifier)
    return "building_preset__" + safe


def _find_widgets_by_text(root, text: str):
    return stock_ext._find_widgets_by_text(root, text)


def _set_procedural_options_enabled(instance, enabled: bool) -> None:
    for _key, label in stock_ext._STOCK_DISABLED_GUI_OPTIONS:
        for widget in _find_widgets_by_text(instance, label):
            stock_ext._set_widget_enabled(widget, enabled)


def _install_gui() -> None:
    from . import gui_entry

    previous_configure = gui_entry._configure_gui

    def configure_gui(gui, base_dir: Path) -> None:
        previous_configure(gui, base_dir)
        original_class = gui.WorldgenGui
        procedural_options = _procedural_options()
        previous_identifier = gui.gui_house_style_preset_identifier
        previous_label = gui.gui_house_style_preset_label

        def multi_identifier(value: object) -> str:
            text = str(value or "").strip()
            folded = text.casefold()
            if folded == PROCEDURAL_AUTO_PRESET:
                return PROCEDURAL_AUTO_PRESET
            if folded.startswith(BUILDING_MULTI_PREFIX):
                selected = building_preset_ids(folded)
                return encode_building_presets(selected)
            return previous_identifier(value)

        def multi_label(value: object) -> str:
            text = str(value or "").strip()
            folded = text.casefold()
            if folded.startswith(BUILDING_MULTI_PREFIX):
                selected = building_preset_ids(folded)
                return encode_building_presets(selected)
            if folded == PROCEDURAL_AUTO_PRESET:
                # The old combobox is hidden; preserve the transport token rather
                # than converting it to the indistinguishable Automatic label.
                return PROCEDURAL_AUTO_PRESET
            return previous_label(value)

        gui.gui_house_style_preset_identifier = multi_identifier
        gui.gui_house_style_preset_label = multi_label

        class MultiPresetWorldgenGui(original_class):
            def _normalise_building_preset_heading(self) -> None:
                for text in (
                    "Building country",
                    "Building countrys",
                    "Building preset",
                    "Building presets",
                ):
                    for widget in _find_widgets_by_text(self, text):
                        try:
                            widget.configure(text="Building presets")
                        except Exception:
                            pass

            def _install_stock_building_checkboxes(self) -> None:
                """Install stock/mod checkboxes after country-label rewriting."""
                preset_var = self.vars.get("house_style_preset")
                if preset_var is None:
                    return
                for identifier, _label in stock_ext.STOCK_BUILDING_OPTIONS:
                    self._var(
                        stock_ext._stock_checkbox_key(identifier),
                        False,
                        boolean=True,
                    )

                labels = []
                for text in (
                    "Building country",
                    "Building countrys",
                    "Building preset",
                    "Building presets",
                ):
                    labels.extend(_find_widgets_by_text(self, text))
                if not labels:
                    return
                label = labels[0]
                parent = label.master
                try:
                    label.configure(text="Building presets")
                except Exception:
                    pass

                combo = stock_ext._find_combobox_for_variable(self, preset_var)
                if combo is not None:
                    stock_ext._hide_widget(combo)

                # Country policy rewrites the original hint before this subclass
                # gets control. Hide either wording, leaving the selection summary
                # below the checkbox groups as the one source of truth.
                for widget in tuple(parent.winfo_children()):
                    try:
                        text = str(widget.cget("text"))
                    except Exception:
                        text = ""
                    if (
                        text == stock_ext._BUILDING_PRESET_HINT_TEXT
                        or "Choose a country here to use that country's procedural building architecture"
                        in text
                        or "Automatic uses the selected map area/country" in text
                    ):
                        stock_ext._hide_widget(widget)

                box = gui.ttk.Frame(parent)
                box.grid(row=2, column=1, sticky="w", pady=3)
                for index, (identifier, text) in enumerate(
                    stock_ext.STOCK_BUILDING_OPTIONS
                ):
                    gui.ttk.Checkbutton(
                        box,
                        text=text,
                        variable=self.vars[stock_ext._stock_checkbox_key(identifier)],
                    ).grid(
                        row=index // 2,
                        column=index % 2,
                        sticky="w",
                        padx=(0, 18),
                        pady=2,
                    )
                self.stock_building_selection_var = gui.tk.StringVar(
                    master=self, value=""
                )
                gui.ttk.Label(
                    parent,
                    textvariable=self.stock_building_selection_var,
                    style="Hint.TLabel",
                    wraplength=700,
                ).grid(
                    row=3,
                    column=0,
                    columnspan=2,
                    sticky="w",
                    pady=(6, 0),
                )

            def _selected_procedural_presets(self) -> tuple[str, ...]:
                selected = []
                for identifier, _label in procedural_options:
                    variable = self.vars.get(_checkbox_key(identifier))
                    if variable is not None and bool(variable.get()):
                        selected.append(identifier)
                return tuple(selected)

            def _all_selected_building_presets(self) -> tuple[str, ...]:
                stock_selected = tuple(self._selected_stock_building_presets())
                return (*stock_selected, *self._selected_procedural_presets())

            def _migrate_stock_dropdown_selection(self) -> None:
                # Override the stock-only migration. Preserve a country or composite
                # value until the procedural checkbox set has also been created.
                preset_var = self.vars.get("house_style_preset")
                if preset_var is None:
                    return
                try:
                    selected = building_preset_ids(
                        gui.gui_house_style_preset_identifier(preset_var.get())
                    )
                except (TypeError, ValueError):
                    selected = ()
                selected_set = frozenset(selected)
                for identifier, _label in stock_ext.STOCK_BUILDING_OPTIONS:
                    variable = self.vars.get(stock_ext._stock_checkbox_key(identifier))
                    if variable is not None:
                        variable.set(identifier in selected_set)
                # During initial construction these variables do not exist yet.
                # During profile loading they do, so restore the whole mixed
                # selection before the stock superclass performs its sync pass.
                for identifier, _label in procedural_options:
                    variable = self.vars.get(_checkbox_key(identifier))
                    if variable is not None:
                        variable.set(identifier in selected_set)

            def _install_procedural_building_checkboxes(self) -> None:
                for identifier, _label in procedural_options:
                    self._var(_checkbox_key(identifier), False, boolean=True)

                labels = _find_widgets_by_text(self, "Building presets")
                if not labels:
                    return
                parent = labels[0].master
                for widget in _find_widgets_by_text(self, "Stock building sets"):
                    try:
                        widget.configure(text="Stock / modded building sets")
                    except Exception:
                        pass
                gui.ttk.Label(parent, text="Procedural building styles").grid(
                    row=4, column=0, sticky="nw", padx=(0, 10), pady=(8, 3)
                )

                holder = gui.ttk.Frame(parent)
                holder.grid(row=4, column=1, sticky="nsew", pady=(8, 3))
                canvas = gui.tk.Canvas(
                    holder,
                    height=190,
                    width=700,
                    highlightthickness=0,
                    borderwidth=0,
                )
                scrollbar = gui.ttk.Scrollbar(
                    holder, orient="vertical", command=canvas.yview
                )
                inner = gui.ttk.Frame(canvas)
                window = canvas.create_window((0, 0), window=inner, anchor="nw")
                canvas.configure(yscrollcommand=scrollbar.set)
                canvas.grid(row=0, column=0, sticky="nsew")
                scrollbar.grid(row=0, column=1, sticky="ns")
                holder.columnconfigure(0, weight=1)

                def resize_scrollregion(_event=None):
                    try:
                        canvas.configure(scrollregion=canvas.bbox("all"))
                    except Exception:
                        pass

                def resize_inner(event):
                    try:
                        canvas.itemconfigure(window, width=event.width)
                    except Exception:
                        pass

                inner.bind("<Configure>", resize_scrollregion)
                canvas.bind("<Configure>", resize_inner)

                for index, (identifier, text) in enumerate(procedural_options):
                    gui.ttk.Checkbutton(
                        inner,
                        text=text,
                        variable=self.vars[_checkbox_key(identifier)],
                    ).grid(
                        row=index // 2,
                        column=index % 2,
                        sticky="w",
                        padx=(0, 18),
                        pady=1,
                    )

                self.multi_building_selection_var = gui.tk.StringVar(master=self, value="")
                gui.ttk.Label(
                    parent,
                    textvariable=self.multi_building_selection_var,
                    style="Hint.TLabel",
                    wraplength=700,
                ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

            def _apply_encoded_selection(self, raw_value: object) -> None:
                try:
                    selected = building_preset_ids(raw_value)
                except (TypeError, ValueError):
                    selected = ()
                selected_set = frozenset(selected)
                for identifier, _label in stock_ext.STOCK_BUILDING_OPTIONS:
                    variable = self.vars.get(stock_ext._stock_checkbox_key(identifier))
                    if variable is not None:
                        variable.set(identifier in selected_set)
                for identifier, _label in procedural_options:
                    variable = self.vars.get(_checkbox_key(identifier))
                    if variable is not None:
                        variable.set(identifier in selected_set)

            def _sync_multi_building_controls(self) -> None:
                # Base WorldgenGui.__init__ refreshes views before this subclass
                # has created the dynamic stock/procedural checkbox variables.
                # Treating that temporary absence as "nothing selected" used to
                # overwrite a remembered/profile selection with auto during startup.
                if not bool(
                    getattr(self, "_building_preset_controls_ready", False)
                ):
                    return
                selected = self._all_selected_building_presets()
                persisted = building_selection_state(selected)
                encoded = str(persisted["house_style_preset"])
                preset_var = self.vars.get("house_style_preset")
                if preset_var is not None:
                    current = str(preset_var.get() or "")
                    if current != encoded:
                        preset_var.set(encoded)
                state_path = getattr(self, "state_path", None)
                if state_path is not None:
                    try:
                        gui.update_gui_state(state_path, persisted)
                    except OSError:
                        pass

                stock_ids = tuple(
                    value for value in selected
                    if value in stock_ext.STOCK_BUILDING_PRESETS
                )
                procedural_ids = tuple(
                    value for value in selected
                    if value not in stock_ext.STOCK_BUILDING_PRESETS
                )
                _set_procedural_options_enabled(
                    self,
                    enabled=bool(procedural_ids) or not stock_ids,
                )

                if hasattr(self, "multi_building_selection_var"):
                    if not selected:
                        message = (
                            "No presets checked: automatic area/country procedural "
                            "buildings are used."
                        )
                    elif stock_ids and procedural_ids:
                        message = (
                            f"Mixed building pool: {len(stock_ids)} stock/mod set(s) "
                            f"and {len(procedural_ids)} procedural style(s)."
                        )
                    elif stock_ids:
                        message = f"Stock/mod building pool: {len(stock_ids)} selected set(s)."
                    else:
                        message = (
                            f"Procedural building pool: {len(procedural_ids)} selected style(s)."
                        )
                    self.multi_building_selection_var.set(message)

            def _stock_building_controls_are_exclusive(self) -> bool:
                # Stock P3Ds only disable procedural-specific options when they
                # are the whole pool. In mixed mode those settings still apply
                # to every building delegated to the procedural child library.
                return (
                    bool(self._selected_stock_building_presets())
                    and not bool(self._selected_procedural_presets())
                )

            def _sync_stock_building_controls(self) -> None:
                super()._sync_stock_building_controls()
                if any(
                    self.vars.get(_checkbox_key(identifier)) is not None
                    for identifier, _label in procedural_options
                ):
                    self._sync_multi_building_controls()

            def _collect_build_values(self) -> dict[str, object]:
                values = super()._collect_build_values()
                values["house_style_preset"] = encode_building_presets(
                    self._all_selected_building_presets()
                )
                return values

            def _profile_document(self) -> dict[str, object]:
                document = super()._profile_document()
                persisted = building_selection_state(
                    self._all_selected_building_presets()
                )
                values = document.get("values")
                if isinstance(values, dict):
                    values["house_style_preset"] = persisted["house_style_preset"]
                document.update(persisted)
                return document

            def __init__(self, *args, **kwargs):
                self._building_preset_controls_ready = False
                super().__init__(*args, **kwargs)
                raw = (
                    self.vars.get("house_style_preset").get()
                    if self.vars.get("house_style_preset") is not None
                    else "auto"
                )
                self._normalise_building_preset_heading()
                self._install_procedural_building_checkboxes()
                self._apply_encoded_selection(raw)
                self._building_preset_controls_ready = True

                self._multi_preset_traces = []
                for identifier, _label in (
                    *stock_ext.STOCK_BUILDING_OPTIONS,
                    *procedural_options,
                ):
                    key = (
                        stock_ext._stock_checkbox_key(identifier)
                        if identifier in stock_ext.STOCK_BUILDING_PRESETS
                        else _checkbox_key(identifier)
                    )
                    variable = self.vars.get(key)
                    if variable is None:
                        continue
                    try:
                        token = variable.trace_add(
                            "write",
                            lambda *_args: self._sync_multi_building_controls(),
                        )
                        self._multi_preset_traces.append((variable, token))
                    except Exception:
                        pass
                self._sync_multi_building_controls()

            def _load_profile(self) -> None:
                super()._load_profile()
                preset_var = self.vars.get("house_style_preset")
                encoded = preset_var.get() if preset_var is not None else "auto"
                profile_path = getattr(self, "profile_path", None)
                if profile_path is not None:
                    try:
                        document = json.loads(profile_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        document = {}
                    saved = document.get("selected_building_presets")
                    if isinstance(saved, list):
                        try:
                            encoded = encode_building_presets(
                                tuple(str(item) for item in saved)
                            )
                        except ValueError:
                            pass
                if preset_var is not None:
                    preset_var.set(encoded)
                self._apply_encoded_selection(encoded)
                self._sync_multi_building_controls()

            def _refresh_views(self) -> None:
                super()._refresh_views()
                self._normalise_building_preset_heading()
                self._sync_multi_building_controls()

        gui.WorldgenGui = MultiPresetWorldgenGui

    gui_entry._configure_gui = configure_gui


def _install_mixed_grounding_and_cache() -> None:
    from . import generator, osm
    from dataclasses import replace

    previous_generate = osm.generate_world_objects

    def generate_with_mixed_stock_lift(*args, **kwargs):
        result = previous_generate(*args, **kwargs)
        library = kwargs.get("building_asset_library")
        if not isinstance(library, MultiBuildingLibrary):
            return result
        plans = tuple(kwargs.get("building_placement_plans") or ())
        lifted = stock_ext._lift_stock_objects(
            result.objects,
            library.stock_library,
            plans,
        )
        if lifted == tuple(result.objects):
            return result
        return replace(result, objects=lifted)

    osm.generate_world_objects = generate_with_mixed_stock_lift
    generator.generate_world_objects = generate_with_mixed_stock_lift

    previous_cache_key = generator.cache_key

    def cache_key_with_mixed_stock(namespace: str, payload):
        if namespace == stock_ext._STOCK_PLACEMENT_CACHE_V96:
            spec = payload.get("spec") if isinstance(payload, Mapping) else None
            preset = spec.get("house_style_preset") if isinstance(spec, Mapping) else None
            if has_stock_building_presets(preset):
                namespace = stock_ext.STOCK_PLACEMENT_CACHE_NAMESPACE
        return previous_cache_key(namespace, payload)

    generator.cache_key = cache_key_with_mixed_stock


def install_multi_building_presets() -> None:
    """Install heterogeneous checkbox selection and mixed building generation."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_transport_and_factory()
    _install_multi_procedural_runtime()
    _install_gui()
    _install_mixed_grounding_and_cache()
    _INSTALLED = True
