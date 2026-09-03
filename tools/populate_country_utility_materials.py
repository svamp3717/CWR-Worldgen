#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Populate explicit utility-building material pools in every country profile.

This is intentionally a data migration, not a runtime inference system.  Each
country/context receives concrete class-specific material distributions derived
from the country's already-curated facade/material vocabulary (falling back to
its parent regional profile only when a country context omits material data).
Explicit OSM building/roof material tags remain authoritative at runtime.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
COUNTRY_DIR = ROOT / "src" / "cwr_worldgen" / "country_styles"
REGION_DIR = ROOT / "src" / "cwr_worldgen" / "house_styles"
REVISION = "2026-09-country-utility-materials-v1"

CLASS_RULES = OrderedDict((
    ("barn", {
        "wall": ("wood", "earth", "brick", "stone", "metal", "concrete", "render"),
        "roof": ("metal", "thatch", "tile", "shingle", "slate", "membrane"),
    }),
    ("shed", {
        "wall": ("wood", "metal", "concrete", "earth", "brick", "render", "stone"),
        "roof": ("metal", "shingle", "membrane", "tile", "slate", "thatch"),
    }),
    ("garage", {
        "wall": ("concrete", "metal", "brick", "render", "wood", "stone", "earth"),
        "roof": ("metal", "membrane", "shingle", "tile", "slate", "thatch"),
    }),
    ("warehouse", {
        "wall": ("metal", "concrete", "brick", "render", "stone", "wood", "earth"),
        "roof": ("metal", "membrane", "shingle", "slate", "tile", "thatch"),
    }),
    ("hangar", {
        "wall": ("metal", "concrete", "wood", "brick", "render", "stone", "earth"),
        "roof": ("metal", "membrane", "shingle", "slate", "tile", "thatch"),
    }),
    ("industrial", {
        "wall": ("concrete", "metal", "brick", "render", "stone", "wood", "earth"),
        "roof": ("metal", "membrane", "shingle", "slate", "tile", "thatch"),
    }),
))

WALL_WEIGHTS = (60, 42, 28, 18, 12, 8, 5)
ROOF_WEIGHTS = (62, 40, 24, 15, 10, 6)


def _tokens(value: str) -> set[str]:
    text = str(value or "").casefold().replace("-", " ").replace("_", " ")
    groups = {
        "metal": ("metal", "steel", "aluminium", "aluminum", "zinc", "galvan", "corrugated", "sheet", "tin"),
        "corrugated": ("corrugated", "profiled", "sheet metal", "sheet steel"),
        "wood": ("wood", "timber", "board", "plank", "clapboard", "bamboo"),
        "concrete": ("concrete", "precast", "cement", "panel", "block"),
        "brick": ("brick",),
        "stone": ("stone", "granite", "limestone", "masonry"),
        "earth": ("adobe", "earth", "mud", "rammed", "laterite"),
        "render": ("stucco", "plaster", "render"),
        "tile": ("tile", "clay", "terracotta"),
        "thatch": ("thatch", "reed", "palm"),
        "shingle": ("shingle", "asphalt"),
        "slate": ("slate",),
        "membrane": ("membrane", "bitumen", "bituminous", "felt", "tar"),
    }
    return {name for name, needles in groups.items() if any(needle in text for needle in needles)}


def _style_hint(style: str) -> tuple[str, str] | None:
    text = str(style or "").casefold()
    if any(token in text for token in ("wood", "timber", "swedish", "nordic")):
        return "wood", "painted timber"
    if "concrete" in text or "panel" in text:
        return "concrete", "concrete"
    if "brick" in text:
        return "brick", "brick"
    if "stone" in text:
        return "stone", "stone"
    if any(token in text for token in ("earth", "adobe", "mud")):
        return "earth", "earth/adobe"
    if any(token in text for token in ("stucco", "plaster", "render")):
        return "render", "stucco/render"
    return None


def _wall_label(category: str, source: str) -> str:
    source_text = str(source or "").casefold()
    if category == "metal":
        return "utility corrugated metal cladding" if "corrugated" in source_text else "utility sheet metal cladding"
    if category == "wood":
        return "utility painted timber board cladding" if "paint" in source_text else "utility rough timber board cladding"
    if category == "concrete":
        return "utility precast concrete panels"
    if category == "brick":
        return "utility structural brick"
    if category == "stone":
        return "utility rough stone masonry"
    if category == "earth":
        return "utility earth/adobe wall"
    if category == "render":
        return "utility rendered masonry wall"
    return f"utility {source}".strip()


def _roof_label(category: str, source: str) -> str:
    source_text = str(source or "").casefold()
    if category == "metal":
        return "utility corrugated metal roof" if "corrugated" in source_text else "utility sheet metal roof"
    if category == "membrane":
        return "utility membrane/bituminous roof"
    if category == "shingle":
        return "utility shingle roof"
    if category == "slate":
        return "utility slate roof"
    if category == "tile":
        return "utility clay/concrete tile roof"
    if category == "thatch":
        return "utility thatch/reed roof"
    return f"utility {source}".strip()


def _unique_distribution(entries: Sequence[tuple[str, int]]) -> list[dict[str, Any]]:
    merged: OrderedDict[str, int] = OrderedDict()
    for material, weight in entries:
        material = str(material).strip()
        if not material:
            continue
        merged[material] = max(merged.get(material, 0), int(weight))
    return [{"material": material, "weight": weight} for material, weight in merged.items()]


def _source_materials(context: Mapping[str, Any], parent_context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    def extract(source: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        details = source.get("architectural_details") or {}
        if not isinstance(details, Mapping):
            return [], []
        materials = details.get("materials") or {}
        geometry = details.get("geometry_defaults") or {}
        roof = geometry.get("roof") or {} if isinstance(geometry, Mapping) else {}
        walls = [str(v) for v in (materials.get("common_wall_materials") or [])] if isinstance(materials, Mapping) else []
        roofs = [str(v) for v in (roof.get("materials") or materials.get("common_roof_materials") or [])] if isinstance(roof, Mapping) and isinstance(materials, Mapping) else []
        return walls, roofs

    walls, roofs = extract(context)
    parent_walls, parent_roofs = extract(parent_context)
    return walls or parent_walls, roofs or parent_roofs


def _facade_hints(context: Mapping[str, Any], role: str) -> list[tuple[str, str]]:
    selection = context.get("selection") or {}
    if not isinstance(selection, Mapping):
        return []
    distributions = selection.get("family_distributions") or {}
    if not isinstance(distributions, Mapping):
        return []
    family = "agricultural" if role == "barn" else "outbuilding" if role in {"shed", "garage"} else "industrial"
    rows = distributions.get(family) or distributions.get("*") or []
    hints: list[tuple[str, str]] = []
    if isinstance(rows, Sequence):
        for row in rows:
            if isinstance(row, Mapping):
                hint = _style_hint(str(row.get("style", "")))
                if hint is not None and hint not in hints:
                    hints.append(hint)
    return hints


def _derive_wall_distribution(context: Mapping[str, Any], role: str, walls: Sequence[str], roofs: Sequence[str]) -> list[dict[str, Any]]:
    sources: dict[str, list[str]] = {name: [] for name in CLASS_RULES[role]["wall"]}
    for value in walls:
        for category in _tokens(value):
            if category in sources and value not in sources[category]:
                sources[category].append(value)
    # Corrugated/profiled metal already present in a country's roof palette is
    # credible evidence for utility sheet cladding. Other roof materials are not
    # promoted to wall materials.
    for value in roofs:
        categories = _tokens(value)
        if "metal" in categories and "corrugated" in categories and value not in sources.get("metal", []):
            sources.setdefault("metal", []).append(value)
    for category, value in _facade_hints(context, role):
        if category in sources and value not in sources[category]:
            sources[category].append(value)

    entries: list[tuple[str, int]] = []
    for index, category in enumerate(CLASS_RULES[role]["wall"]):
        weight = WALL_WEIGHTS[min(index, len(WALL_WEIGHTS) - 1)]
        for source in sources.get(category, ())[:2]:
            entries.append((_wall_label(category, source), weight))
    if not entries:
        entries.append(("utility rendered masonry wall", 100))
    return _unique_distribution(entries)


def _derive_roof_distribution(role: str, roofs: Sequence[str]) -> list[dict[str, Any]]:
    sources: dict[str, list[str]] = {name: [] for name in CLASS_RULES[role]["roof"]}
    for value in roofs:
        for category in _tokens(value):
            if category in sources and value not in sources[category]:
                sources[category].append(value)
    entries: list[tuple[str, int]] = []
    for index, category in enumerate(CLASS_RULES[role]["roof"]):
        weight = ROOF_WEIGHTS[min(index, len(ROOF_WEIGHTS) - 1)]
        for source in sources.get(category, ())[:2]:
            entries.append((_roof_label(category, source), weight))
    if not entries:
        entries.append(("utility sheet metal roof", 100))
    return _unique_distribution(entries)


def _region_profiles() -> dict[str, Mapping[str, Any]]:
    profiles: dict[str, Mapping[str, Any]] = {}
    for path in sorted(REGION_DIR.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, Mapping):
            profiles[str(doc.get("identifier", path.stem)).casefold()] = doc
    return profiles


def update_country(path: Path, regions: Mapping[str, Mapping[str, Any]]) -> tuple[bool, int]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return False, 0
    parent = regions.get(str(doc.get("parent_region_identifier", "")).casefold(), {})
    parent_contexts = parent.get("contexts") or {} if isinstance(parent, Mapping) else {}
    contexts = doc.get("contexts") or {}
    if not isinstance(contexts, dict):
        return False, 0

    changed_contexts = 0
    for context_name, context in contexts.items():
        if not isinstance(context, dict):
            continue
        details = context.setdefault("architectural_details", {})
        if not isinstance(details, dict):
            continue
        materials = details.setdefault("materials", {})
        if not isinstance(materials, dict):
            continue
        parent_context = parent_contexts.get(context_name) or parent_contexts.get("rural") or {}
        if not isinstance(parent_context, Mapping):
            parent_context = {}
        walls, roofs = _source_materials(context, parent_context)
        overrides = materials.setdefault("building_class_overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}
            materials["building_class_overrides"] = overrides
        for role in CLASS_RULES:
            overrides[role] = {
                "wall_materials": _derive_wall_distribution(context, role, walls, roofs),
                "roof_materials": _derive_roof_distribution(role, roofs),
                "selection_note": (
                    "Class-specific utility material pool. Explicit OSM building:material/roof:material tags override this default."
                ),
            }
        materials["utility_materials_revision"] = REVISION
        materials["utility_materials_provenance"] = (
            "Explicit country/context class pools derived from this profile's curated material/facade vocabulary; parent-region materials are used only when the country context omits them."
        )
        changed_contexts += 1

    doc["detail_revision"] = REVISION
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return True, changed_contexts


def main() -> int:
    regions = _region_profiles()
    files = sorted(path for path in COUNTRY_DIR.glob("*.json") if path.name != "index.json")
    if len(files) != 249:
        raise RuntimeError(f"expected 249 country profiles, found {len(files)}")
    updated = contexts = 0
    for path in files:
        changed, count = update_country(path, regions)
        updated += int(changed)
        contexts += count
    if updated != 249:
        raise RuntimeError(f"updated {updated} country profiles instead of 249")
    print(f"updated {updated} country profiles across {contexts} contexts ({REVISION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
