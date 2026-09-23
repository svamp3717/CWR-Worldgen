# SPDX-License-Identifier: GPL-3.0-or-later
"""Extended stock-building presets, GUI behavior, and stock-model grounding.

This module layers on top of :mod:`stock_building_policy`. The original mixed
stock preset stays backward compatible, while source-specific presets load
their own measured catalogue files. Stock model origins are also lifted by the measured
distance from model origin to visible base before RVW4 serialization.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
from typing import Iterable, Mapping, Sequence

from .model import WorldObject
from .procedural_buildings import BuildingGenerationResult
from . import stock_building_policy as stock

STOCK_BUILDING_VANILLA_PRESET = "stock-vanilla"
STOCK_BUILDING_RESISTANCE_PRESET = "stock-resistance"
STOCK_BUILDING_HAUS_ONLY_PRESET = "stock-haus-only"
STOCK_BUILDING_AGS_ONLY_PRESET = "stock-ags-only"
STOCK_BUILDING_BAS_O_GENERAL_PRESET = "stock-bas-o-general"
STOCK_BUILDING_BAS_O_MIDDLEAST_PRESET = "stock-bas-o-middleast"
STOCK_BUILDING_BAS_O_SHANTY_PRESET = "stock-bas-o-shanty"
STOCK_BUILDING_BAS_O_AFRICAHUT_PRESET = "stock-bas-o-africahut"
STOCK_BUILDING_ART_BD_PRESET = "stock-art-bd"
STOCK_BUILDING_CAF_KKK_BUILDINGS2_PRESET = "stock-caf-kkk-buildings2"
STOCK_BUILDING_DMA_LIBYA_O_PRESET = "stock-dma-libya-o"
STOCK_BUILDING_CATINTRO_PRESET = "stock-catintro"
STOCK_BUILDING_FDF_PRESET = "stock-fdf"

# Legacy combined identifiers remain accepted when loading old profiles, but no
# combined catalogue JSONs are shipped anymore. They expand into source sets.
STOCK_BUILDING_HAUS_COMBINED_PRESET = "stock-haus-combined"
STOCK_BUILDING_AGS_COMBINED_PRESET = "stock-ags-combined"

_DATA_DIR = Path(__file__).with_name("data")
_STOCK_NON_RESISTANCE_CATALOGUE_PATH = _DATA_DIR / "stock_building_models_non_resistance.json"
_STOCK_RESISTANCE_CATALOGUE_PATH = _DATA_DIR / "stock_building_models_resistance.json"
_STOCK_HAUS_ONLY_CATALOGUE_PATH = _DATA_DIR / "haus.pbo buildings only.json"
_STOCK_AGS_ONLY_CATALOGUE_PATH = _DATA_DIR / "ags inds+port.json"
_STOCK_BAS_O_GENERAL_CATALOGUE_PATH = _DATA_DIR / "BAS_O.pbo general.json"
_STOCK_BAS_O_MIDDLEAST_CATALOGUE_PATH = _DATA_DIR / "BAS_O.pbo middleast.json"
_STOCK_BAS_O_SHANTY_CATALOGUE_PATH = _DATA_DIR / "BAS_O.pbo shanty.json"
_STOCK_BAS_O_AFRICAHUT_CATALOGUE_PATH = _DATA_DIR / "BAS_O.pbo africahut.json"
_STOCK_ART_BD_CATALOGUE_PATH = _DATA_DIR / "art_bd.json"
_STOCK_CAF_KKK_BUILDINGS2_CATALOGUE_PATH = (
    _DATA_DIR / "caf_kkk_buildings2.json"
)
_STOCK_DMA_LIBYA_O_CATALOGUE_PATH = _DATA_DIR / "DMA_libya_o.json"
_STOCK_CATINTRO_CATALOGUE_PATH = _DATA_DIR / "catintro.json"
_STOCK_FDF_CATALOGUE_PATH = _DATA_DIR / "fdf.json"


def _catalogue_display_name(path: Path, fallback: str) -> str:
    """Read the human-facing preset name from one source catalogue."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return str(document.get("display_name", "")).strip() or fallback


STOCK_BUILDING_VANILLA_LABEL = _catalogue_display_name(
    _STOCK_NON_RESISTANCE_CATALOGUE_PATH,
    "Stock non-Resistance buildings",
)
STOCK_BUILDING_RESISTANCE_LABEL = _catalogue_display_name(
    _STOCK_RESISTANCE_CATALOGUE_PATH,
    "Stock Resistance buildings",
)
STOCK_BUILDING_HAUS_ONLY_LABEL = _catalogue_display_name(
    _STOCK_HAUS_ONLY_CATALOGUE_PATH,
    "Haus.pbo buildings",
)
STOCK_BUILDING_AGS_ONLY_LABEL = _catalogue_display_name(
    _STOCK_AGS_ONLY_CATALOGUE_PATH,
    "ags_inds.pbo and ags_port.pbo buildings",
)
STOCK_BUILDING_BAS_O_GENERAL_LABEL = _catalogue_display_name(
    _STOCK_BAS_O_GENERAL_CATALOGUE_PATH,
    "BAS_O.pbo General buildings",
)
STOCK_BUILDING_BAS_O_MIDDLEAST_LABEL = _catalogue_display_name(
    _STOCK_BAS_O_MIDDLEAST_CATALOGUE_PATH,
    "BAS_O.pbo Middle East buildings",
)
STOCK_BUILDING_BAS_O_SHANTY_LABEL = _catalogue_display_name(
    _STOCK_BAS_O_SHANTY_CATALOGUE_PATH,
    "BAS_O.pbo Shanty buildings",
)
STOCK_BUILDING_BAS_O_AFRICAHUT_LABEL = _catalogue_display_name(
    _STOCK_BAS_O_AFRICAHUT_CATALOGUE_PATH,
    "BAS_O.pbo African hut buildings",
)
STOCK_BUILDING_ART_BD_LABEL = _catalogue_display_name(
    _STOCK_ART_BD_CATALOGUE_PATH,
    "ART_BD.pbo buildings",
)
STOCK_BUILDING_CAF_KKK_BUILDINGS2_LABEL = _catalogue_display_name(
    _STOCK_CAF_KKK_BUILDINGS2_CATALOGUE_PATH,
    "CAF_KKK_Buildings2.pbo buildings",
)
STOCK_BUILDING_DMA_LIBYA_O_LABEL = _catalogue_display_name(
    _STOCK_DMA_LIBYA_O_CATALOGUE_PATH,
    "DMA_libya_o.pbo buildings",
)
STOCK_BUILDING_CATINTRO_LABEL = _catalogue_display_name(
    _STOCK_CATINTRO_CATALOGUE_PATH,
    "catintro.pbo buildings",
)
STOCK_BUILDING_FDF_LABEL = _catalogue_display_name(
    _STOCK_FDF_CATALOGUE_PATH,
    "finmod buildings",
)

# Compatibility labels for code/imports that still know the old combined IDs.
STOCK_BUILDING_COMBINED_LABEL = "Stock combined (non-Resistance + Resistance) buildings"
STOCK_BUILDING_HAUS_COMBINED_LABEL = "Haus.pbo + Resistance + vanilla buildings"
STOCK_BUILDING_AGS_COMBINED_LABEL = "ags_inds.pbo and ags_port.pbo buildings + combined stock"
stock.STOCK_BUILDING_PRESET_LABEL = STOCK_BUILDING_COMBINED_LABEL

# These are the only real source presets exposed in the GUI. Multi-selection
# composes them dynamically, so pre-combined JSON catalogues would just duplicate
# the same model rows and eventually drift.
STOCK_BUILDING_PRESETS = (
    STOCK_BUILDING_VANILLA_PRESET,
    STOCK_BUILDING_RESISTANCE_PRESET,
    STOCK_BUILDING_HAUS_ONLY_PRESET,
    STOCK_BUILDING_AGS_ONLY_PRESET,
    STOCK_BUILDING_BAS_O_GENERAL_PRESET,
    STOCK_BUILDING_BAS_O_MIDDLEAST_PRESET,
    STOCK_BUILDING_BAS_O_SHANTY_PRESET,
    STOCK_BUILDING_BAS_O_AFRICAHUT_PRESET,
    STOCK_BUILDING_ART_BD_PRESET,
    STOCK_BUILDING_CAF_KKK_BUILDINGS2_PRESET,
    STOCK_BUILDING_DMA_LIBYA_O_PRESET,
    STOCK_BUILDING_CATINTRO_PRESET,
    STOCK_BUILDING_FDF_PRESET,
)
STOCK_BUILDING_OPTIONS = (
    (STOCK_BUILDING_VANILLA_PRESET, STOCK_BUILDING_VANILLA_LABEL),
    (STOCK_BUILDING_RESISTANCE_PRESET, STOCK_BUILDING_RESISTANCE_LABEL),
    (STOCK_BUILDING_HAUS_ONLY_PRESET, STOCK_BUILDING_HAUS_ONLY_LABEL),
    (STOCK_BUILDING_AGS_ONLY_PRESET, STOCK_BUILDING_AGS_ONLY_LABEL),
    (STOCK_BUILDING_BAS_O_GENERAL_PRESET, STOCK_BUILDING_BAS_O_GENERAL_LABEL),
    (STOCK_BUILDING_BAS_O_MIDDLEAST_PRESET, STOCK_BUILDING_BAS_O_MIDDLEAST_LABEL),
    (STOCK_BUILDING_BAS_O_SHANTY_PRESET, STOCK_BUILDING_BAS_O_SHANTY_LABEL),
    (STOCK_BUILDING_BAS_O_AFRICAHUT_PRESET, STOCK_BUILDING_BAS_O_AFRICAHUT_LABEL),
    (STOCK_BUILDING_ART_BD_PRESET, STOCK_BUILDING_ART_BD_LABEL),
    (STOCK_BUILDING_CAF_KKK_BUILDINGS2_PRESET, STOCK_BUILDING_CAF_KKK_BUILDINGS2_LABEL),
    (STOCK_BUILDING_DMA_LIBYA_O_PRESET, STOCK_BUILDING_DMA_LIBYA_O_LABEL),
    (STOCK_BUILDING_CATINTRO_PRESET, STOCK_BUILDING_CATINTRO_LABEL),
    (STOCK_BUILDING_FDF_PRESET, STOCK_BUILDING_FDF_LABEL),
)

STOCK_BUILDING_MULTI_PREFIX = "stock-multi:"
_LEGACY_COMBINED_PRESET_EXPANSIONS = {
    stock.STOCK_BUILDING_PRESET: (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_RESISTANCE_PRESET,
    ),
    STOCK_BUILDING_HAUS_COMBINED_PRESET: (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_RESISTANCE_PRESET,
        STOCK_BUILDING_HAUS_ONLY_PRESET,
    ),
    STOCK_BUILDING_AGS_COMBINED_PRESET: (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_RESISTANCE_PRESET,
        STOCK_BUILDING_AGS_ONLY_PRESET,
    ),
}
_ACCEPTED_STOCK_BUILDING_PRESETS = (
    *STOCK_BUILDING_PRESETS,
    *_LEGACY_COMBINED_PRESET_EXPANSIONS,
)
_STOCK_CATALOGUE_BY_PRESET = {
    STOCK_BUILDING_VANILLA_PRESET: _STOCK_NON_RESISTANCE_CATALOGUE_PATH,
    STOCK_BUILDING_RESISTANCE_PRESET: _STOCK_RESISTANCE_CATALOGUE_PATH,
    STOCK_BUILDING_HAUS_ONLY_PRESET: _STOCK_HAUS_ONLY_CATALOGUE_PATH,
    STOCK_BUILDING_AGS_ONLY_PRESET: _STOCK_AGS_ONLY_CATALOGUE_PATH,
    STOCK_BUILDING_BAS_O_GENERAL_PRESET: _STOCK_BAS_O_GENERAL_CATALOGUE_PATH,
    STOCK_BUILDING_BAS_O_MIDDLEAST_PRESET: _STOCK_BAS_O_MIDDLEAST_CATALOGUE_PATH,
    STOCK_BUILDING_BAS_O_SHANTY_PRESET: _STOCK_BAS_O_SHANTY_CATALOGUE_PATH,
    STOCK_BUILDING_BAS_O_AFRICAHUT_PRESET: _STOCK_BAS_O_AFRICAHUT_CATALOGUE_PATH,
    STOCK_BUILDING_ART_BD_PRESET: _STOCK_ART_BD_CATALOGUE_PATH,
    STOCK_BUILDING_CAF_KKK_BUILDINGS2_PRESET: _STOCK_CAF_KKK_BUILDINGS2_CATALOGUE_PATH,
    STOCK_BUILDING_DMA_LIBYA_O_PRESET: _STOCK_DMA_LIBYA_O_CATALOGUE_PATH,
    STOCK_BUILDING_CATINTRO_PRESET: _STOCK_CATINTRO_CATALOGUE_PATH,
    STOCK_BUILDING_FDF_PRESET: _STOCK_FDF_CATALOGUE_PATH,
}


def _expand_stock_identifier(identifier: str) -> tuple[str, ...]:
    if identifier in _LEGACY_COMBINED_PRESET_EXPANSIONS:
        return _LEGACY_COMBINED_PRESET_EXPANSIONS[identifier]
    if identifier in STOCK_BUILDING_PRESETS:
        return (identifier,)
    return ()


def stock_building_preset_ids(value: object) -> tuple[str, ...]:
    """Return canonical source-catalogue IDs from a stock preset selection."""
    text = str(value or "").strip().casefold()
    direct = _expand_stock_identifier(text)
    if direct:
        return direct
    if not text.startswith(STOCK_BUILDING_MULTI_PREFIX):
        return ()

    raw = text[len(STOCK_BUILDING_MULTI_PREFIX):]
    requested: set[str] = set()
    unknown: list[str] = []
    for item in (part.strip().casefold() for part in raw.split(",")):
        if not item:
            continue
        expanded = _expand_stock_identifier(item)
        if not expanded:
            unknown.append(item)
            continue
        requested.update(expanded)
    if not requested and not unknown:
        raise ValueError("stock-multi requires at least one stock building preset")
    if unknown:
        raise ValueError("unknown stock building preset(s): " + ", ".join(sorted(set(unknown))))
    return tuple(identifier for identifier in STOCK_BUILDING_PRESETS if identifier in requested)


def encode_stock_building_presets(values: Sequence[str]) -> str:
    """Encode one or more source catalogue IDs into the existing preset field."""
    requested: set[str] = set()
    unknown: list[str] = []
    for value in values:
        identifier = str(value).strip().casefold()
        if not identifier:
            continue
        expanded = _expand_stock_identifier(identifier)
        if not expanded:
            unknown.append(identifier)
            continue
        requested.update(expanded)
    if unknown:
        raise ValueError("unknown stock building preset(s): " + ", ".join(sorted(set(unknown))))
    ordered = tuple(identifier for identifier in STOCK_BUILDING_PRESETS if identifier in requested)
    if not ordered:
        raise ValueError("at least one stock building preset is required")
    if len(ordered) == 1:
        return ordered[0]
    return STOCK_BUILDING_MULTI_PREFIX + ",".join(ordered)


def _load_stock_preset_models(presets: Sequence[str]):
    """Load and de-duplicate models from selected source catalogues."""
    by_path = {}
    for preset in presets:
        path = _STOCK_CATALOGUE_BY_PRESET[preset]
        for model in stock._load_catalogue(path):
            by_path.setdefault(_canonical_model_path(model.model_path), model)
    return tuple(by_path.values())


def _stock_checkbox_key(identifier: str) -> str:
    return "stock_building_set__" + identifier.replace("-", "_")


_STOCK_PLACEMENT_CACHE_V96 = "nonroad-object-placement-v96-road-safe-settlement-clutter"
_STOCK_PLACEMENT_CACHE_V99 = "nonroad-object-placement-v99-malden-modern-forest"
_STOCK_PLACEMENT_CACHE_V100 = "nonroad-object-placement-v100-final-stock-road-audit"
_STOCK_PLACEMENT_CACHE_V101 = "nonroad-object-placement-v101-stock-road-model-rescue"
# Stable cross-module symbol. Mixed preset routing must not depend on a private
# version-suffixed constant, otherwise every cache bump becomes an import-time
# AttributeError waiting for one wrapper to miss the rename.
STOCK_PLACEMENT_CACHE_NAMESPACE = _STOCK_PLACEMENT_CACHE_V101
_BUILDING_PLACEMENT_CACHE_REVISION = "final-road-building-clearance-v10-stock-road-model-rescue"
_INTERIOR_CHECKBOX_TEXT = "Enterable procedural-building interiors"
_HIGH_QUALITY_TEXTURE_CHECKBOX_TEXT = "Higher-quality building textures (256 px)"
_MATCH_TEXTURE_CHECKBOX_TEXT = "Match nearby same-shape town/city building textures"
_PROCEDURAL_BRIDGES_CHECKBOX_TEXT = "Procedural bridges (instead of Nogova)"
_BUILDING_PRESET_HINT_TEXT = (
    "Automatic uses the selected map area/country. Choose one of the 23 regional presets here "
    "to override procedural building façades and roof defaults for the entire world."
)
_STOCK_DISABLED_GUI_OPTIONS: tuple[tuple[str, str], ...] = (
    ("procedural_building_interiors", _INTERIOR_CHECKBOX_TEXT),
    ("high_quality_building_textures", _HIGH_QUALITY_TEXTURE_CHECKBOX_TEXT),
    ("match_nearby_building_textures", _MATCH_TEXTURE_CHECKBOX_TEXT),
)
_INSTALLED = False


def _canonical_model_path(value: object) -> str:
    return str(value or "").replace("/", "\\").lstrip("\\").casefold()


def stock_model_source(model_path: object) -> str:
    """Return the stock source family for a model path."""
    path = _canonical_model_path(model_path)
    if path.startswith("o\\"):
        return "resistance"
    if path.startswith("haus\\"):
        return "haus"
    if path.startswith("ags_inds\\") or path.startswith("ags_port\\"):
        return "ags"
    if path.startswith("bas_o\\"):
        return "bas_o"
    if path.startswith("art_bd\\"):
        return "art_bd"
    if path.startswith("caf_kkk_buildings2\\"):
        return "caf_kkk_buildings2"
    if path.startswith("dma_libya_o\\"):
        return "dma_libya_o"
    if path.startswith("catintro\\"):
        return "catintro"
    if path.startswith("fdf_s\\"):
        return "fdf"
    return "vanilla"


def is_stock_building_preset(value: object) -> bool:
    return bool(stock_building_preset_ids(value))


def stock_disabled_gui_option_keys(preset: object) -> tuple[str, ...]:
    """Return GUI settings that are meaningless for original stock P3Ds."""
    if not is_stock_building_preset(preset):
        return ()
    return tuple(key for key, _label in _STOCK_DISABLED_GUI_OPTIONS)


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
    """Put the stock catalogue choices directly after Automatic."""
    stock_ids = frozenset(_ACCEPTED_STOCK_BUILDING_PRESETS)
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


def _building_preset_grid_row(label, default: int = 3) -> int:
    """Return the current Building presets row instead of assuming an old layout."""

    try:
        info = label.grid_info()
        if info and "row" in info:
            return int(info["row"])
    except (AttributeError, TypeError, ValueError):
        pass
    return int(default)


def _find_combobox_for_variable(root, variable):
    try:
        children = root.winfo_children()
    except Exception:
        return None
    target = str(variable)
    for child in children:
        try:
            if child.winfo_class() == "TCombobox" and str(child.cget("textvariable")) == target:
                return child
        except Exception:
            pass
        nested = _find_combobox_for_variable(child, variable)
        if nested is not None:
            return nested
    return None


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
        requested = str(
            kwargs.get("house_style_preset", stock.STOCK_BUILDING_PRESET) or ""
        ).strip().casefold()
        selected = stock_building_preset_ids(requested)
        if not selected:
            selected = stock_building_preset_ids(stock.STOCK_BUILDING_PRESET)
        requested = encode_stock_building_presets(selected)
        original_init(self, *args, **kwargs)
        self.house_style_preset = requested

        models = _load_stock_preset_models(selected)
        if not models:
            raise RuntimeError(f"Stock building preset {requested!r} has no measured models")
        self.models = tuple(models)

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
        selected_sources = stock_building_preset_ids(mode)
        payload = {
            "schema": 2,
            "mode": mode,
            "ground_texture_profile": str(
                getattr(self, "ground_texture_profile", "generated") or "generated"
            ),
            "selected_building_jsons": [
                f"data/{_STOCK_CATALOGUE_BY_PRESET[preset].name}"
                for preset in selected_sources
            ],
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
            if is_stock_building_preset(preset):
                return stock.StockBuildingLibrary(*args, **kwargs)
            return previous_factory(*args, **kwargs)

    generator.ProceduralBuildingLibrary = StockPresetBuildingLibraryFactory

    previous_normalise = milestone8.normalise_house_style_preset

    def normalise_with_stock_sources(value):
        selected = stock_building_preset_ids(value)
        if selected:
            return encode_stock_building_presets(selected)
        return previous_normalise(value)

    milestone8.normalise_house_style_preset = normalise_with_stock_sources

    non_stock = tuple(
        identifier for identifier in tuple(cli.HOUSE_STYLE_PRESET_IDENTIFIERS)
        if str(identifier).casefold() not in _ACCEPTED_STOCK_BUILDING_PRESETS
    )
    cli.HOUSE_STYLE_PRESET_IDENTIFIERS = (*_ACCEPTED_STOCK_BUILDING_PRESETS, *non_stock)


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
            *_ACCEPTED_STOCK_BUILDING_PRESETS,
            *(value for value in existing_ids if str(value).casefold() not in _ACCEPTED_STOCK_BUILDING_PRESETS),
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
            selected = stock_building_preset_ids(text)
            if selected:
                return encode_stock_building_presets(selected)
            return previous_identifier(value)

        def stock_label(value: object) -> str:
            selected = stock_building_preset_ids(value)
            if len(selected) == 1:
                return identifier_to_label[selected[0]]
            if selected:
                # Preserve the encoded selection during legacy profile loading;
                # the GUI subclass migrates it into the checkbox variables.
                return encode_stock_building_presets(selected)
            return previous_label(value)

        gui.gui_house_style_preset_identifier = stock_identifier
        gui.gui_house_style_preset_label = stock_label

        original_class = gui.WorldgenGui

        class StockBuildingControlsWorldgenGui(original_class):
            def _selected_stock_building_presets(self) -> tuple[str, ...]:
                selected = []
                for identifier, _label in STOCK_BUILDING_OPTIONS:
                    variable = self.vars.get(_stock_checkbox_key(identifier))
                    if variable is not None and bool(variable.get()):
                        selected.append(identifier)
                return tuple(selected)

            def _migrate_stock_dropdown_selection(self) -> None:
                """Migrate an old single stock dropdown choice, then retire the dropdown value."""
                preset_var = self.vars.get("house_style_preset")
                if preset_var is None:
                    return
                try:
                    selected = stock_building_preset_ids(
                        gui.gui_house_style_preset_identifier(preset_var.get())
                    )
                except ValueError:
                    selected = ()
                for identifier in selected:
                    variable = self.vars.get(_stock_checkbox_key(identifier))
                    if variable is not None:
                        variable.set(True)
                # The country/procedural dropdown is no longer a user-facing choice.
                # With no stock boxes selected, Automatic remains the fallback.
                preset_var.set(gui.HOUSE_STYLE_AUTO_LABEL)

            def _install_stock_building_checkboxes(self) -> None:
                """Replace the old building-country combobox with multi-select stock sets."""
                preset_var = self.vars.get("house_style_preset")
                if preset_var is None:
                    return
                for identifier, _label in STOCK_BUILDING_OPTIONS:
                    self._var(_stock_checkbox_key(identifier), False, boolean=True)

                labels = _find_widgets_by_text(self, "Building preset")
                if not labels:
                    return
                label = labels[0]
                parent = label.master
                try:
                    label.configure(text="Building presets")
                except Exception:
                    pass

                combo = _find_combobox_for_variable(self, preset_var)
                if combo is not None:
                    _hide_widget(combo)
                for hint in _find_widgets_by_text(self, _BUILDING_PRESET_HINT_TEXT):
                    _hide_widget(hint)

                building_row = _building_preset_grid_row(label)
                box = gui.ttk.Frame(parent)
                box.grid(row=building_row, column=1, sticky="w", pady=3)
                for index, (identifier, text) in enumerate(STOCK_BUILDING_OPTIONS):
                    gui.ttk.Checkbutton(
                        box,
                        text=text,
                        variable=self.vars[_stock_checkbox_key(identifier)],
                    ).grid(
                        row=index // 2,
                        column=index % 2,
                        sticky="w",
                        padx=(0, 18),
                        pady=2,
                    )
                self.stock_building_selection_var = gui.tk.StringVar(master=self, value="")
                gui.ttk.Label(
                    parent,
                    textvariable=self.stock_building_selection_var,
                    style="Hint.TLabel",
                    wraplength=700,
                ).grid(
                    row=building_row + 1,
                    column=0,
                    columnspan=2,
                    sticky="w",
                    pady=(6, 0),
                )

            def _stock_building_controls_are_exclusive(self) -> bool:
                """Whether stock selection should disable procedural-only settings."""
                return bool(self._selected_stock_building_presets())

            def _sync_stock_building_controls(self) -> None:
                # Procedural bridges remain the product default, but the GUI no
                # longer exposes a second implementation switch. Old profiles
                # that saved stock bridges are pushed back to the current default.
                bridge_var = self.vars.get("procedural_bridges")
                if bridge_var is not None and not bool(bridge_var.get()):
                    bridge_var.set(True)
                for widget in _find_widgets_by_text(self, _PROCEDURAL_BRIDGES_CHECKBOX_TEXT):
                    _hide_widget(widget)

                selected = self._selected_stock_building_presets()
                stock_mode = self._stock_building_controls_are_exclusive()
                if hasattr(self, "stock_building_selection_var"):
                    if selected:
                        labels = dict(STOCK_BUILDING_OPTIONS)
                        self.stock_building_selection_var.set(
                            "Selected stock sets: "
                            + ", ".join(labels[identifier] for identifier in selected)
                            + ". Model paths are merged and duplicates are removed."
                        )
                    else:
                        self.stock_building_selection_var.set(
                            "No building presets selected. Automatic country/procedural buildings are used."
                        )

                for key, label in _STOCK_DISABLED_GUI_OPTIONS:
                    variable = self.vars.get(key)
                    if stock_mode and variable is not None and bool(variable.get()):
                        variable.set(False)
                    for widget in _find_widgets_by_text(self, label):
                        _set_widget_enabled(widget, not stock_mode)

            def _collect_build_values(self) -> dict[str, object]:
                values = super()._collect_build_values()
                selected = self._selected_stock_building_presets()
                values["house_style_preset"] = (
                    encode_stock_building_presets(selected)
                    if selected
                    else gui.HOUSE_STYLE_PRESET_AUTO
                )
                return values

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._install_stock_building_checkboxes()
                self._migrate_stock_dropdown_selection()
                preset_var = self.vars.get("house_style_preset")
                if preset_var is not None:
                    try:
                        self._stock_preset_trace = preset_var.trace_add(
                            "write", lambda *_args: self._sync_stock_building_controls()
                        )
                    except Exception:
                        self._stock_preset_trace = None
                self._sync_stock_building_controls()

            def _load_profile(self) -> None:
                super()._load_profile()
                self._migrate_stock_dropdown_selection()
                self._sync_stock_building_controls()

            def _refresh_views(self) -> None:
                super()._refresh_views()
                self._sync_stock_building_controls()

        gui.WorldgenGui = StockBuildingControlsWorldgenGui

    gui_entry._configure_gui = configure_gui


def _stock_model_allowed_in_settlement(model, settlement: str) -> bool:
    """Keep rescue models inside the same reviewed urban/rural placement pool."""
    context = str(settlement or "rural").strip().casefold()
    wanted = "Urban" if context in {"urban", "town", "city", "town_city"} else "Rural"
    placement = str(getattr(model, "placement", "") or "")
    return not placement or placement in {"Both", wanted}


def _smaller_stock_road_replacement(
    obj,
    original_model,
    stock_library: stock.StockBuildingLibrary,
    road_index,
    physical,
    elevations,
    spec,
    *,
    preferred_family: str = "",
):
    """Return the largest smaller stock model that clears the road at this origin.

    Preserve the mapped semantic family first. If every smaller model in that
    family still intersects the road, allow a residential fallback before giving
    up. Candidates must fit inside the original model's oriented rectangle, so
    this late safety gate cannot create a new building/building overlap elsewhere.
    """
    original_width = max(0.05, float(original_model.width_m))
    original_length = max(0.05, float(original_model.length_m))
    original_area = original_width * original_length
    original_path = _canonical_model_path(original_model.model_path)
    settlement = stock_library._settlement_context(float(obj.x), float(obj.z))

    family_order: list[str] = []
    requested = str(preferred_family or "").strip().casefold()
    if requested:
        family_order.append(requested)
    else:
        for family in tuple(getattr(original_model, "families", ()) or ()):
            folded = str(family).strip().casefold()
            if folded and folded not in family_order:
                family_order.append(folded)
    if "residential" not in family_order:
        family_order.append("residential")

    for family in family_order:
        safe: list[tuple[float, float, str, float, object]] = []
        for candidate in tuple(getattr(stock_library, "models", ()) or ()):
            if _canonical_model_path(candidate.model_path) == original_path:
                continue
            candidate_families = {
                str(value).strip().casefold()
                for value in tuple(getattr(candidate, "families", ()) or ())
            }
            if family not in candidate_families:
                continue
            if not _stock_model_allowed_in_settlement(candidate, settlement):
                continue

            candidate_area = float(candidate.width_m) * float(candidate.length_m)
            if candidate_area >= original_area - 1.0e-6:
                continue

            for turn in (0.0, 90.0):
                effective_width = (
                    float(candidate.width_m)
                    if turn == 0.0
                    else float(candidate.length_m)
                )
                effective_length = (
                    float(candidate.length_m)
                    if turn == 0.0
                    else float(candidate.width_m)
                )
                if (
                    effective_width > original_width + 1.0e-6
                    or effective_length > original_length + 1.0e-6
                ):
                    continue

                heading = (float(obj.heading_degrees) + turn) % 360.0
                polygon = stock._model_support_polygon(
                    float(obj.x),
                    float(obj.z),
                    candidate,
                    heading,
                )
                conflicts, _checked = physical.conflicts_at_clearance(
                    polygon,
                    road_index,
                    0.0,
                )
                if conflicts:
                    continue

                # Largest safe footprint first means the rescue shrinks only as
                # much as necessary. The remaining fields make ties deterministic.
                dimension_loss = (
                    (original_width - effective_width)
                    + (original_length - effective_length)
                )
                safe.append(
                    (
                        -candidate_area,
                        dimension_loss,
                        _canonical_model_path(candidate.model_path),
                        turn,
                        candidate,
                    )
                )

        if safe:
            safe.sort(key=lambda item: item[:4])
            _area, _loss, _path, turn, candidate = safe[0]
            heading = (float(obj.heading_degrees) + turn) % 360.0
            polygon = stock._model_support_polygon(
                float(obj.x),
                float(obj.z),
                candidate,
                heading,
            )
            from . import osm as osm_module

            _minimum_height, maximum_height = osm_module._polygon_elevation_extrema(
                elevations,
                spec.cells,
                spec.cell_size,
                polygon,
            )
            new_lift = float(stock_library.origin_lift_for_model(candidate.model_path))
            ground_clearance = max(
                0.0,
                float(getattr(spec, "building_ground_clearance", 0.10)),
            )
            return replace(
                obj,
                model_path=candidate.model_path,
                y=float(maximum_height) + ground_clearance + new_lift,
                heading_degrees=heading,
            )
    return None


def _remove_stock_buildings_overlapping_final_roads(
    result,
    stock_library: stock.StockBuildingLibrary,
    road_report,
    elevations,
    spec,
    *,
    building_plans: Sequence[object] = (),
):
    """Rescue serialized stock buildings that still physically overlap a road.

    This is a last safety gate over the actual object transforms. It deliberately
    runs after placement-cache restore and after minor-road suppression. A
    conflicting building first tries progressively smaller models from its mapped
    semantic family, then a residential model. It is removed only if no selected
    stock model can clear the final road at the same origin.
    """
    from . import final_building_road_clearance_policy as clearance
    from . import physical_road_overlap_policy as physical
    from . import road_building_priority_policy as priority

    models = {
        _canonical_model_path(model.model_path): model
        for model in tuple(getattr(stock_library, "models", ()) or ())
    }
    if not models:
        return result, ()

    plan_families: dict[tuple[str, float, float], str] = {}
    position_families: dict[tuple[float, float], str] = {}
    for plan in tuple(building_plans or ()):
        x = float(getattr(plan, "x", 0.0))
        z = float(getattr(plan, "z", 0.0))
        family = str(getattr(plan, "building_family", "") or "").strip().casefold()
        if not family:
            continue
        position_families[(x, z)] = family
        plan_families[
            (_canonical_model_path(getattr(plan, "model_path", "")), x, z)
        ] = family

    road_objects = priority._filter_suppressed_roads(
        getattr(road_report, "objects", ()) or ()
    )
    primitives = clearance._road_primitives(
        SimpleNamespace(objects=road_objects),
        elevations,
        spec,
    )
    if not primitives:
        return result, ()
    road_index = clearance._RoadPrimitiveIndex(primitives)

    kept = []
    removed = []
    changed = False
    for obj in tuple(getattr(result, "objects", ()) or ()):
        model_path = _canonical_model_path(getattr(obj, "model_path", ""))
        model = models.get(model_path)
        if model is None:
            kept.append(obj)
            continue
        polygon = stock._model_support_polygon(
            float(obj.x),
            float(obj.z),
            model,
            float(obj.heading_degrees),
        )
        conflicts, _checked = physical.conflicts_at_clearance(
            polygon,
            road_index,
            0.0,
        )
        if not conflicts:
            kept.append(obj)
            continue

        x = float(obj.x)
        z = float(obj.z)
        family = plan_families.get(
            (model_path, x, z),
            position_families.get((x, z), ""),
        )
        replacement = _smaller_stock_road_replacement(
            obj,
            model,
            stock_library,
            road_index,
            physical,
            elevations,
            spec,
            preferred_family=family,
        )
        if replacement is not None:
            kept.append(replacement)
            changed = True
            continue

        removed.append(obj)
        changed = True

    if not changed:
        return result, ()

    usage: dict[str, int] = {}
    stock_usage: dict[str, int] = {}
    for obj in kept:
        key = str(obj.model_path)
        usage[key] = usage.get(key, 0) + 1
        if _canonical_model_path(obj.model_path) in models:
            stock_usage[key] = stock_usage.get(key, 0) + 1
    stock_library._usage = stock_usage

    revised = replace(
        result,
        objects=tuple(kept),
        building_objects=max(
            0,
            int(getattr(result, "building_objects", 0)) - len(removed),
        ),
        model_usage=tuple(
            sorted(usage.items(), key=lambda item: item[0].casefold())
        ),
    )
    return revised, tuple(removed)


def _install_grounding() -> None:
    from . import final_building_road_clearance_policy as clearance
    from . import generator, osm

    # This module installs after the road/church policy chain, several members
    # of which intentionally replace the shared placement-cache salt. Make the
    # stock-fit/overlap revision the final active salt so existing cached plans
    # cannot replay oversized or rural-urban-mismatched model selections.
    clearance._CACHE_REVISION = _BUILDING_PLACEMENT_CACHE_REVISION

    original_generate = osm.generate_world_objects
    original_load_nonroad_objects = generator._load_nonroad_objects

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

    def load_nonroad_objects_with_final_stock_road_audit(
        dataset,
        projection,
        raster,
        elevations,
        spec,
        **kwargs,
    ):
        loaded = original_load_nonroad_objects(
            dataset,
            projection,
            raster,
            elevations,
            spec,
            **kwargs,
        )
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            return loaded

        result = loaded[0]
        library = loaded[1]
        stock_library = (
            library
            if isinstance(library, stock.StockBuildingLibrary)
            else getattr(library, "stock_library", None)
        )
        if not isinstance(stock_library, stock.StockBuildingLibrary):
            return loaded

        road_report = kwargs.get("road_report")
        if road_report is None:
            road_report = clearance._road_context_matches(
                dataset,
                projection,
                elevations,
                spec,
            )
        if road_report is None:
            return loaded

        revised, removed = _remove_stock_buildings_overlapping_final_roads(
            result,
            stock_library,
            road_report,
            elevations,
            spec,
            building_plans=kwargs.get("building_placement_plans", ()),
        )
        if revised is result:
            return loaded

        original_by_id = {
            int(obj.object_id): obj
            for obj in tuple(getattr(result, "objects", ()) or ())
        }
        replaced_count = sum(
            1
            for obj in tuple(getattr(revised, "objects", ()) or ())
            if (
                int(obj.object_id) in original_by_id
                and _canonical_model_path(original_by_id[int(obj.object_id)].model_path)
                != _canonical_model_path(obj.model_path)
            )
        )
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback(
                52,
                "Resolving final stock buildings that physically overlap roads "
                f"({replaced_count:,} replaced with smaller models; "
                f"{len(removed):,} rejected after cache/final-road audit)",
            )
        return (revised, *loaded[1:])

    generator._load_nonroad_objects = load_nonroad_objects_with_final_stock_road_audit

    original_cache_key = generator.cache_key

    def cache_key_with_stock_grounding(namespace: str, payload: Mapping[str, object]):
        if namespace == _STOCK_PLACEMENT_CACHE_V96:
            spec = payload.get("spec") if isinstance(payload, Mapping) else None
            preset = spec.get("house_style_preset") if isinstance(spec, Mapping) else None
            if is_stock_building_preset(preset):
                namespace = STOCK_PLACEMENT_CACHE_NAMESPACE
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
