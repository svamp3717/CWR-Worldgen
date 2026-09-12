# SPDX-License-Identifier: GPL-3.0-or-later
"""Extended stock-building presets, GUI behavior, and stock-model grounding.

This module layers on top of :mod:`stock_building_policy`. The original mixed
stock preset stays backward compatible, while source-specific presets filter the
same measured catalogue. Stock model origins are also lifted by the measured
distance from model origin to visible base before RVW4 serialization.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
from typing import Iterable, Mapping, Sequence

from .model import WorldObject
from .procedural_buildings import BuildingGenerationResult
from . import stock_building_policy as stock

STOCK_BUILDING_VANILLA_PRESET = "stock-vanilla"
STOCK_BUILDING_RESISTANCE_PRESET = "stock-resistance"

STOCK_BUILDING_VANILLA_LABEL = "Stock vanilla buildings only"
STOCK_BUILDING_RESISTANCE_LABEL = "Stock Resistance buildings only"

STOCK_BUILDING_PRESETS = (
    stock.STOCK_BUILDING_PRESET,
    STOCK_BUILDING_VANILLA_PRESET,
    STOCK_BUILDING_RESISTANCE_PRESET,
)
STOCK_BUILDING_OPTIONS = (
    (stock.STOCK_BUILDING_PRESET, stock.STOCK_BUILDING_PRESET_LABEL),
    (STOCK_BUILDING_VANILLA_PRESET, STOCK_BUILDING_VANILLA_LABEL),
    (STOCK_BUILDING_RESISTANCE_PRESET, STOCK_BUILDING_RESISTANCE_LABEL),
)

_STOCK_PLACEMENT_CACHE_V96 = "nonroad-object-placement-v96-road-safe-settlement-clutter"
_STOCK_PLACEMENT_CACHE_V97 = "nonroad-object-placement-v97-stock-model-origin-grounding"
_INTERIOR_CHECKBOX_TEXT = "Enterable procedural-building interiors"
_HIGH_QUALITY_TEXTURE_CHECKBOX_TEXT = "Higher-quality building textures (256 px)"
_MATCH_TEXTURE_CHECKBOX_TEXT = "Match nearby same-shape town/city building textures"
_PROCEDURAL_BRIDGES_CHECKBOX_TEXT = "Procedural bridges (instead of Nogova)"
_STOCK_DISABLED_GUI_OPTIONS: tuple[tuple[str, str], ...] = (
    ("procedural_building_interiors", _INTERIOR_CHECKBOX_TEXT),
    ("high_quality_building_textures", _HIGH_QUALITY_TEXTURE_CHECKBOX_TEXT),
    ("match_nearby_building_textures", _MATCH_TEXTURE_CHECKBOX_TEXT),
)
_INSTALLED = False


def _canonical_model_path(value: object) -> str:
    return str(value or "").replace("/", "\\").lstrip("\\").casefold()


def stock_model_source(model_path: object) -> str:
    """Return ``resistance`` for O.pbo models and ``vanilla`` otherwise."""
    return "resistance" if _canonical_model_path(model_path).startswith("o\\") else "vanilla"


def is_stock_building_preset(value: object) -> bool:
    return str(value or "").strip().casefold() in STOCK_BUILDING_PRESETS


def stock_disabled_gui_option_keys(preset: object) -> tuple[str, ...]:
    """Return GUI settings that are meaningless for original stock P3Ds."""
    if not is_stock_building_preset(preset):
        return ()
    return tuple(key for key, _label in _STOCK_DISABLED_GUI_OPTIONS)


def _filter_models(models: Iterable[stock.StockBuildingModel], preset: str):
    values = tuple(models)
    if preset == STOCK_BUILDING_VANILLA_PRESET:
        return tuple(model for model in values if stock_model_source(model.model_path) == "vanilla")
    if preset == STOCK_BUILDING_RESISTANCE_PRESET:
        return tuple(model for model in values if stock_model_source(model.model_path) == "resistance")
    return values


def _origin_lift_for_model(self, model_path: str) -> float:
    """Return the measured model-origin-to-visible-base distance in metres."""
    wanted = _canonical_model_path(model_path)
    for model in getattr(self, "models", ()):
        if _canonical_model_path(model.model_path) == wanted:
            return max(0.0, float(getattr(model, "origin_to_bottom_m", 0.0) or 0.0))
    # Old cached/pickled libraries may predate the source-filtered ``models``
    # tuple. Fall back to the bundled catalogue rather than grounding them at 0.
    for model in stock._load_catalogue():
        if _canonical_model_path(model.model_path) == wanted:
            return max(0.0, float(getattr(model, "origin_to_bottom_m", 0.0) or 0.0))
    return 0.0


def _lift_stock_objects(
    objects: Sequence[WorldObject],
    library: stock.StockBuildingLibrary,
    plans: Sequence[object],
) -> tuple[WorldObject, ...]:
    """Lift only stock building objects by each P3D's measured vertical origin."""
    lifts: dict[tuple[str, float, float], float] = {}
    for plan in plans or ():
        model_path = str(getattr(plan, "model_path", "") or "")
        if not model_path:
            continue
        lift = float(library.origin_lift_for_model(model_path))
        if lift <= 0.0:
            continue
        key = (
            _canonical_model_path(model_path),
            float(getattr(plan, "x", 0.0)),
            float(getattr(plan, "z", 0.0)),
        )
        lifts[key] = lift

    if not lifts:
        return tuple(objects)

    changed = False
    result: list[WorldObject] = []
    for obj in objects:
        key = (_canonical_model_path(obj.model_path), float(obj.x), float(obj.z))
        lift = lifts.get(key, 0.0)
        if lift > 0.0:
            obj = replace(obj, y=float(obj.y) + lift)
            changed = True
        result.append(obj)
    return tuple(result) if changed else tuple(objects)


def _stock_options_first(
    options: Sequence[tuple[str, str]],
    labels: Sequence[str],
    *,
    auto_label: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Put the three stock choices directly after Automatic."""
    stock_ids = frozenset(STOCK_BUILDING_PRESETS)
    stock_labels = frozenset(label for _identifier, label in STOCK_BUILDING_OPTIONS)
    remaining_options = tuple(
        (identifier, label)
        for identifier, label in options
        if str(identifier).casefold() not in stock_ids
    )
    remaining_labels = tuple(
        label
        for label in labels
        if label != auto_label and label not in stock_labels
    )
    return (
        (*STOCK_BUILDING_OPTIONS, *remaining_options),
        (auto_label, *(label for _identifier, label in STOCK_BUILDING_OPTIONS), *remaining_labels),
    )


def _find_widgets_by_text(root, text: str):
    result = []
    try:
        children = root.winfo_children()
    except Exception:
        children = ()
    for child in children:
        try:
            if str(child.cget("text")) == text:
                result.append(child)
        except Exception:
            pass
        result.extend(_find_widgets_by_text(child, text))
    return result


def _set_widget_enabled(widget, enabled: bool) -> None:
    try:
        widget.state(["!disabled"] if enabled else ["disabled"])
    except Exception:
        try:
            widget.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass


def _hide_widget(widget) -> None:
    """Remove a built GUI control without changing the underlying CLI setting."""
    for method_name in ("grid_remove", "pack_forget", "place_forget"):
        method = getattr(widget, method_name, None)
        if method is None:
            continue
        try:
            method()
            return
        except Exception:
            continue


def _install_library_presets() -> None:
    original_init = stock.StockBuildingLibrary.__init__

    def stock_init(self, *args, **kwargs):
        requested = str(kwargs.get("house_style_preset", stock.STOCK_BUILDING_PRESET) or "").strip().casefold()
        if requested not in STOCK_BUILDING_PRESETS:
            requested = stock.STOCK_BUILDING_PRESET
        original_init(self, *args, **kwargs)
        self.house_style_preset = requested
        filtered = _filter_models(self.models, requested)
        if not filtered:
            raise RuntimeError(f"Stock building preset {requested!r} has no measured models")
        self.models = filtered

    stock.StockBuildingLibrary.__init__ = stock_init
    stock.StockBuildingLibrary.origin_lift_for_model = _origin_lift_for_model

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:
        del source_dir
        records = []
        for model_path, count in sorted(self._usage.items(), key=lambda item: item[0].casefold()):
            records.append({
                "model_path": model_path,
                "placements": count,
                "stock": True,
                "source_set": stock_model_source(model_path),
                "origin_lift_m": round(float(self.origin_lift_for_model(model_path)), 4),
                # Original CWA/OFP ODOLs contain their own LOD stack. No P3D is
                # authored by Worldgen in any stock preset.
                "lod_count": 3,
            })
        placements = sum(self._usage.values())
        reused = max(0, placements - len(records))
        mode = str(getattr(self, "house_style_preset", stock.STOCK_BUILDING_PRESET) or stock.STOCK_BUILDING_PRESET)
        payload = {
            "schema": 2,
            "mode": mode,
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

    stock.StockBuildingLibrary.write_assets = write_assets


def _install_factory_and_cli() -> None:
    from . import cli, generator, milestone8

    previous_factory = generator.ProceduralBuildingLibrary

    class StockPresetBuildingLibraryFactory:
        def __new__(cls, *args, **kwargs):
            preset = str(kwargs.get("house_style_preset", "") or "").strip().casefold()
            if preset in STOCK_BUILDING_PRESETS:
                return stock.StockBuildingLibrary(*args, **kwargs)
            return previous_factory(*args, **kwargs)

    generator.ProceduralBuildingLibrary = StockPresetBuildingLibraryFactory

    previous_normalise = milestone8.normalise_house_style_preset

    def normalise_with_stock_sources(value):
        preset = str(value or "").strip().casefold()
        if preset in STOCK_BUILDING_PRESETS:
            return preset
        return previous_normalise(value)

    milestone8.normalise_house_style_preset = normalise_with_stock_sources

    non_stock = tuple(
        identifier for identifier in tuple(cli.HOUSE_STYLE_PRESET_IDENTIFIERS)
        if str(identifier).casefold() not in STOCK_BUILDING_PRESETS
    )
    cli.HOUSE_STYLE_PRESET_IDENTIFIERS = (*STOCK_BUILDING_PRESETS, *non_stock)


def _install_gui() -> None:
    from . import gui_entry

    previous_configure = gui_entry._configure_gui

    def configure_gui(gui, base_dir: Path) -> None:
        previous_configure(gui, base_dir)

        auto_label = str(getattr(gui, "HOUSE_STYLE_AUTO_LABEL", "Automatic (area / country)"))
        options, labels = _stock_options_first(
            tuple(getattr(gui, "HOUSE_STYLE_PRESET_OPTIONS", ())),
            tuple(getattr(gui, "HOUSE_STYLE_PRESET_LABELS", ())),
            auto_label=auto_label,
        )
        gui.HOUSE_STYLE_PRESET_OPTIONS = options
        gui.HOUSE_STYLE_PRESET_LABELS = labels

        existing_ids = tuple(getattr(gui, "HOUSE_STYLE_PRESET_IDENTIFIERS", ()))
        gui.HOUSE_STYLE_PRESET_IDENTIFIERS = (
            *STOCK_BUILDING_PRESETS,
            *(value for value in existing_ids if str(value).casefold() not in STOCK_BUILDING_PRESETS),
        )

        label_to_identifier = dict(getattr(gui, "_HOUSE_STYLE_LABEL_TO_IDENTIFIER", {}))
        identifier_to_label = dict(getattr(gui, "_HOUSE_STYLE_IDENTIFIER_TO_LABEL", {}))
        for identifier, label in STOCK_BUILDING_OPTIONS:
            label_to_identifier[label] = identifier
            identifier_to_label[identifier] = label
        gui._HOUSE_STYLE_LABEL_TO_IDENTIFIER = label_to_identifier
        gui._HOUSE_STYLE_IDENTIFIER_TO_LABEL = identifier_to_label

        previous_identifier = gui.gui_house_style_preset_identifier
        previous_label = gui.gui_house_style_preset_label

        def stock_identifier(value: object) -> str:
            text = str(value or "").strip()
            mapped = label_to_identifier.get(text)
            if mapped in STOCK_BUILDING_PRESETS:
                return mapped
            folded = text.casefold()
            if folded in STOCK_BUILDING_PRESETS:
                return folded
            return previous_identifier(value)

        def stock_label(value: object) -> str:
            folded = str(value or "").strip().casefold()
            if folded in STOCK_BUILDING_PRESETS:
                return identifier_to_label[folded]
            return previous_label(value)

        gui.gui_house_style_preset_identifier = stock_identifier
        gui.gui_house_style_preset_label = stock_label

        original_class = gui.WorldgenGui

        class StockBuildingControlsWorldgenGui(original_class):
            def _sync_stock_building_controls(self) -> None:
                # Procedural bridges remain the product default, but the GUI no
                # longer exposes a second implementation switch. Old profiles
                # that saved stock bridges are pushed back to the current default.
                bridge_var = self.vars.get("procedural_bridges")
                if bridge_var is not None and not bool(bridge_var.get()):
                    bridge_var.set(True)
                for widget in _find_widgets_by_text(self, _PROCEDURAL_BRIDGES_CHECKBOX_TEXT):
                    _hide_widget(widget)

                preset_var = self.vars.get("house_style_preset")
                if preset_var is None:
                    return
                try:
                    preset = gui.gui_house_style_preset_identifier(preset_var.get())
                except Exception:
                    preset = str(preset_var.get() or "").strip().casefold()
                stock_mode = preset in STOCK_BUILDING_PRESETS

                for key, label in _STOCK_DISABLED_GUI_OPTIONS:
                    variable = self.vars.get(key)
                    if stock_mode and variable is not None and bool(variable.get()):
                        variable.set(False)
                    for widget in _find_widgets_by_text(self, label):
                        _set_widget_enabled(widget, not stock_mode)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                preset_var = self.vars.get("house_style_preset")
                if preset_var is not None:
                    try:
                        self._stock_preset_trace = preset_var.trace_add(
                            "write", lambda *_args: self._sync_stock_building_controls()
                        )
                    except Exception:
                        self._stock_preset_trace = None
                self._sync_stock_building_controls()

            def _refresh_views(self) -> None:
                super()._refresh_views()
                self._sync_stock_building_controls()

        gui.WorldgenGui = StockBuildingControlsWorldgenGui

    gui_entry._configure_gui = configure_gui


def _install_grounding() -> None:
    from . import generator, osm

    original_generate = osm.generate_world_objects

    def generate_with_stock_origin_lift(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        library = kwargs.get("building_asset_library")
        if not isinstance(library, stock.StockBuildingLibrary):
            return result
        plans = tuple(kwargs.get("building_placement_plans") or ())
        lifted = _lift_stock_objects(result.objects, library, plans)
        if lifted == tuple(result.objects):
            return result
        return replace(result, objects=lifted)

    osm.generate_world_objects = generate_with_stock_origin_lift
    # generator imported this function by name, so patch its bound reference too.
    generator.generate_world_objects = generate_with_stock_origin_lift

    original_cache_key = generator.cache_key

    def cache_key_with_stock_grounding(namespace: str, payload: Mapping[str, object]):
        if namespace == _STOCK_PLACEMENT_CACHE_V96:
            spec = payload.get("spec") if isinstance(payload, Mapping) else None
            preset = spec.get("house_style_preset") if isinstance(spec, Mapping) else None
            if is_stock_building_preset(preset):
                namespace = _STOCK_PLACEMENT_CACHE_V97
        return original_cache_key(namespace, payload)

    generator.cache_key = cache_key_with_stock_grounding


def install_stock_building_extensions() -> None:
    """Install source-filtered stock presets, GUI controls, and origin grounding."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_library_presets()
    _install_factory_and_cli()
    _install_gui()
    _install_grounding()
    _INSTALLED = True
