# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from .assets import canonical_asset_path
from .procedural_infrastructure import gravel_road_model_path, _texture_file_stem

_ALLOWED_GEOMETRIES = {"any", "point", "line", "polygon"}
_KNOWN_LAYERS = {
    "coastlines", "water", "forests", "farmland", "urban", "roads", "gravel_roads",
    "watercourses", "building_polygons", "building_points", "places",
    "landmarks", "sites", "barriers", "cutlines", "tree_rows",
    "individual_trees", "aeroway_lines", "aeroway_areas", "utility_points",
    "surface_areas", "rural_vegetation",
}


@dataclass(frozen=True, slots=True)
class OsmAssetRule:
    rule_id: str
    layers: tuple[str, ...]
    geometry: str
    match: tuple[tuple[str, tuple[str, ...]], ...]
    exclude: tuple[tuple[str, tuple[str, ...]], ...]
    models: tuple[str, ...]
    textures: tuple[str, ...]
    description: str = ""
    enabled: bool = True

    def to_manifest(self) -> dict[str, object]:
        return {
            "id": self.rule_id,
            "layers": self.layers,
            "geometry": self.geometry,
            "match": {key: values for key, values in self.match},
            "exclude": {key: values for key, values in self.exclude},
            "models": self.models,
            "textures": self.textures,
            "description": self.description,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class OsmAssetMapping:
    rules: tuple[OsmAssetRule, ...]
    global_models: tuple[str, ...] = ()
    global_textures: tuple[str, ...] = ()
    source: str = "built-in defaults"
    inherit_defaults: bool = True

    @property
    def sha256(self) -> str:
        document = {
            "rules": [rule.to_manifest() for rule in self.rules],
            "global_models": self.global_models,
            "global_textures": self.global_textures,
            "source": self.source,
            "inherit_defaults": self.inherit_defaults,
        }
        raw = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OsmAssetRuleMatch:
    rule_id: str
    feature_count: int
    models: tuple[str, ...]
    textures: tuple[str, ...]
    sample_osm_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OsmAssetMappingReport:
    source: str
    inherit_defaults: bool
    mapping_sha256: str
    feature_count: int
    matched_feature_count: int
    selected_models: tuple[str, ...]
    selected_textures: tuple[str, ...]
    rule_matches: tuple[OsmAssetRuleMatch, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "source": self.source,
            "inherit_defaults": self.inherit_defaults,
            "mapping_sha256": self.mapping_sha256,
            "feature_count": self.feature_count,
            "matched_feature_count": self.matched_feature_count,
            "selected_models": self.selected_models,
            "selected_textures": self.selected_textures,
            "rule_matches": [asdict(item) for item in self.rule_matches],
        }


def _normalise_values(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{label} must be a string or list of strings")
    result = tuple(item.strip().casefold() for item in values if item.strip())
    if not result:
        raise ValueError(f"{label} must contain at least one value")
    return result


def _normalise_conditions(value: Any, *, label: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if value in (None, {}):
        return ()
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result: list[tuple[str, tuple[str, ...]]] = []
    for raw_key, raw_values in value.items():
        key = str(raw_key).strip().casefold()
        if not key:
            raise ValueError(f"{label} contains an empty tag name")
        result.append((key, _normalise_values(raw_values, label=f"{label}.{key}")))
    return tuple(sorted(result))


def _normalise_assets(value: Any, *, suffixes: tuple[str, ...], label: str) -> tuple[str, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a list")
    result: list[str] = []
    for item in value:
        path = str(item).replace("/", "\\").strip().lstrip("\\")
        if not path:
            raise ValueError(f"{label} contains an empty path")
        try:
            path.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{label} paths must be ASCII") from exc
        if not path.casefold().endswith(suffixes):
            raise ValueError(f"{label} path has the wrong suffix: {path}")
        result.append(path)
    return tuple(dict.fromkeys(result))


def _parse_rule(document: Mapping[str, Any]) -> OsmAssetRule:
    rule_id = str(document.get("id", "")).strip()
    if not rule_id:
        raise ValueError("OSM asset mapping rule is missing id")
    raw_layers = document.get("layers", ["*"])
    if isinstance(raw_layers, str):
        layers = (raw_layers.strip(),)
    elif isinstance(raw_layers, Sequence):
        layers = tuple(str(item).strip() for item in raw_layers if str(item).strip())
    else:
        raise ValueError(f"rule {rule_id}: layers must be a string or list")
    if not layers:
        raise ValueError(f"rule {rule_id}: layers must not be empty")
    unknown = sorted(layer for layer in layers if layer != "*" and layer not in _KNOWN_LAYERS)
    if unknown:
        raise ValueError(f"rule {rule_id}: unknown layers {unknown}")
    geometry = str(document.get("geometry", "any")).strip().casefold()
    if geometry not in _ALLOWED_GEOMETRIES:
        raise ValueError(f"rule {rule_id}: geometry must be one of {sorted(_ALLOWED_GEOMETRIES)}")
    return OsmAssetRule(
        rule_id=rule_id,
        layers=layers,
        geometry=geometry,
        match=_normalise_conditions(document.get("match"), label=f"rule {rule_id}.match"),
        exclude=_normalise_conditions(document.get("exclude"), label=f"rule {rule_id}.exclude"),
        models=_normalise_assets(document.get("models"), suffixes=(".p3d",), label=f"rule {rule_id}.models"),
        textures=_normalise_assets(document.get("textures"), suffixes=(".paa", ".pac"), label=f"rule {rule_id}.textures"),
        description=str(document.get("description", "")).strip(),
        enabled=bool(document.get("enabled", True)),
    )


def _merge_rules(defaults: Sequence[OsmAssetRule], custom: Sequence[OsmAssetRule]) -> tuple[OsmAssetRule, ...]:
    by_id = {rule.rule_id: rule for rule in defaults}
    order = [rule.rule_id for rule in defaults]
    for rule in custom:
        if rule.rule_id not in by_id:
            order.append(rule.rule_id)
        by_id[rule.rule_id] = rule
    return tuple(by_id[rule_id] for rule_id in order if by_id[rule_id].enabled)


def load_osm_asset_mapping(path: Path | None, defaults: OsmAssetMapping) -> OsmAssetMapping:
    if path is None:
        return defaults
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read OSM asset mapping {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("OSM asset mapping root must be an object")
    if int(document.get("schema", 1)) != 1:
        raise ValueError("OSM asset mapping schema must be 1")
    inherit_defaults = bool(document.get("inherit_defaults", True))
    raw_rules = document.get("rules", [])
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes, bytearray)):
        raise ValueError("OSM asset mapping rules must be a list")
    custom_rules = tuple(_parse_rule(item) for item in raw_rules if isinstance(item, Mapping))
    if len(custom_rules) != len(raw_rules):
        raise ValueError("every OSM asset mapping rule must be an object")
    global_section = document.get("global", {})
    if not isinstance(global_section, Mapping):
        raise ValueError("OSM asset mapping global section must be an object")
    custom_global_models = _normalise_assets(global_section.get("models"), suffixes=(".p3d",), label="global.models")
    custom_global_textures = _normalise_assets(global_section.get("textures"), suffixes=(".paa", ".pac"), label="global.textures")
    rules = _merge_rules(defaults.rules if inherit_defaults else (), custom_rules)
    return OsmAssetMapping(
        rules=rules,
        global_models=tuple(dict.fromkeys((*defaults.global_models, *custom_global_models))) if inherit_defaults else custom_global_models,
        global_textures=tuple(dict.fromkeys((*defaults.global_textures, *custom_global_textures))) if inherit_defaults else custom_global_textures,
        source=str(Path(path).resolve()),
        inherit_defaults=inherit_defaults,
    )


def _condition_matches(tags: Mapping[str, str], conditions: Sequence[tuple[str, tuple[str, ...]]]) -> bool:
    folded = {str(key).casefold(): str(value).casefold() for key, value in tags.items()}
    for key, values in conditions:
        actual = folded.get(key)
        if "*" in values:
            if actual is None or actual == "":
                return False
            continue
        if actual is None or actual not in values:
            return False
    return True


def _rule_matches(rule: OsmAssetRule, layer: str, geometry: str, tags: Mapping[str, str]) -> bool:
    if "*" not in rule.layers and layer not in rule.layers:
        return False
    if rule.geometry != "any" and rule.geometry != geometry:
        return False
    if not _condition_matches(tags, rule.match):
        return False
    if rule.exclude and _condition_matches(tags, rule.exclude):
        return False
    return True


def _iter_features(dataset: Any) -> Iterable[tuple[str, str, str, Mapping[str, str]]]:
    point_layers = {"building_points", "places", "landmarks", "individual_trees", "utility_points"}
    line_layers = {"coastlines", "roads", "gravel_roads", "watercourses", "barriers", "cutlines", "tree_rows", "aeroway_lines"}
    polygon_layers = {"water", "forests", "farmland", "urban", "building_polygons", "sites", "aeroway_areas", "surface_areas", "rural_vegetation"}
    gravel_features = tuple(getattr(dataset, "gravel_roads", ()) or ())
    if not gravel_features:
        gravel_surfaces = {"gravel", "fine_gravel", "compacted", "pebblestone", "unpaved"}
        gravel_features = tuple(
            feature for feature in (getattr(dataset, "roads", ()) or ())
            if str((getattr(feature, "tags", {}) or {}).get("surface", "")).strip().casefold() in gravel_surfaces
        )
    gravel_keys = {str(getattr(feature, "osm_key", "")) for feature in gravel_features}
    for layer in sorted(_KNOWN_LAYERS):
        features = gravel_features if layer == "gravel_roads" else (getattr(dataset, layer, ()) or ())
        geometry = "point" if layer in point_layers else "line" if layer in line_layers else "polygon" if layer in polygon_layers else "any"
        for feature in features:
            osm_key = str(getattr(feature, "osm_key", ""))
            if layer == "roads" and osm_key in gravel_keys:
                continue
            yield layer, geometry, osm_key, getattr(feature, "tags", {}) or {}


def collect_osm_asset_requirements(dataset: Any, mapping: OsmAssetMapping) -> OsmAssetMappingReport:
    matches: dict[str, list[str]] = {rule.rule_id: [] for rule in mapping.rules}
    feature_count = 0
    matched_keys: set[str] = set()
    selected_models = {canonical_asset_path(path) for path in mapping.global_models}
    selected_textures = {canonical_asset_path(path) for path in mapping.global_textures}
    for layer, geometry, osm_key, tags in _iter_features(dataset):
        feature_count += 1
        for rule in mapping.rules:
            if not _rule_matches(rule, layer, geometry, tags):
                continue
            matches[rule.rule_id].append(osm_key)
            matched_keys.add(f"{layer}:{osm_key}")
            selected_models.update(canonical_asset_path(path) for path in rule.models)
            selected_textures.update(canonical_asset_path(path) for path in rule.textures)
    rule_matches = tuple(
        OsmAssetRuleMatch(
            rule_id=rule.rule_id,
            feature_count=len(matches[rule.rule_id]),
            models=tuple(canonical_asset_path(path) for path in rule.models),
            textures=tuple(canonical_asset_path(path) for path in rule.textures),
            sample_osm_keys=tuple(matches[rule.rule_id][:8]),
        )
        for rule in mapping.rules
        if matches[rule.rule_id]
    )
    return OsmAssetMappingReport(
        source=mapping.source,
        inherit_defaults=mapping.inherit_defaults,
        mapping_sha256=mapping.sha256,
        feature_count=feature_count,
        matched_feature_count=len(matched_keys),
        selected_models=tuple(sorted(selected_models)),
        selected_textures=tuple(sorted(selected_textures)),
        rule_matches=rule_matches,
    )


def _rule(
    rule_id: str,
    layers: Sequence[str],
    match: Mapping[str, Any],
    *,
    models: Sequence[str] = (),
    textures: Sequence[str] = (),
    exclude: Mapping[str, Any] | None = None,
    geometry: str = "any",
    description: str = "",
) -> OsmAssetRule:
    return _parse_rule({
        "id": rule_id,
        "layers": list(layers),
        "geometry": geometry,
        "match": dict(match),
        "exclude": dict(exclude or {}),
        "models": list(models),
        "textures": list(textures),
        "description": description,
    })


def default_osm_asset_mapping(spec: Any, milestone_number: int, *, global_textures: Sequence[str] = ()) -> OsmAssetMapping:
    rules: list[OsmAssetRule] = []
    paved_highways = ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "living_street")
    dirt_highways = ("service", "track", "unclassified", "road")
    gravel_surfaces = ("gravel", "fine_gravel", "compacted", "pebblestone", "unpaved")
    dirt_surfaces = ("dirt", "earth", "ground", "mud", "sand")
    if bool(getattr(spec, "procedural_gravel_roads", False)):
        rules.append(_rule(
            "road-gravel", ("gravel_roads",), {},
            models=tuple(gravel_road_model_path(spec.name, nominal) for nominal in (25, 12, 6, 3)),
            textures=(rf"{spec.name}\i\{_texture_file_stem('gravel')}.paa",),
            geometry="line",
            description="Generated world-local gravel road ribbon family",
        ))
    else:
        rules.append(_rule(
            "road-gravel", ("gravel_roads",), {},
            models=(getattr(spec, "dirt_road_model", r"o\road\ces25.p3d"),),
            geometry="line", description="Gravel roads using the current dirt-road fallback",
        ))
    rules.append(_rule(
        "road-paved", ("roads",), {"highway": paved_highways},
        exclude={"surface": (*gravel_surfaces, *dirt_surfaces)}, models=(getattr(spec, "paved_road_model", r"o\road\sil25.p3d"),),
        geometry="line", description="Current paved-road model family",
    ))
    rules.append(_rule(
        "road-dirt-by-class", ("roads",), {"highway": dirt_highways},
        exclude={"surface": ("asphalt", "paved", "concrete", "concrete:plates", "sett")}, models=(getattr(spec, "dirt_road_model", r"o\road\ces25.p3d"),),
        geometry="line", description="Current dirt-road model family",
    ))
    rules.append(_rule(
        "road-dirt-by-surface", ("roads",), {"surface": dirt_surfaces},
        models=(getattr(spec, "dirt_road_model", r"o\road\ces25.p3d"),), geometry="line",
    ))
    rules.append(_rule(
        "primary-forest", ("forests",), {}, models=(getattr(spec, "forest_tree_model", r"data3d\les_su_ctver_pruhozi.p3d"),), geometry="polygon",
    ))
    if bool(getattr(spec, "barriers_enabled", milestone_number >= 9)):
        rules.append(_rule("hedges", ("barriers",), {"barrier": "hedge"}, models=tuple(getattr(spec, "stock_hedge_models", ())), geometry="line"))
        rules.append(_rule("stone-walls", ("barriers",), {"barrier": ("wall", "retaining_wall")}, models=tuple(getattr(spec, "stock_wall_models", ())), geometry="line"))
        metal_values = ("chain_link", "chainlink", "metal", "metal_bars", "wire", "wire_mesh", "welded_wire_mesh", "mesh")
        rules.append(_rule("metal-fences-by-type", ("barriers",), {"barrier": "fence", "fence_type": metal_values}, models=tuple(getattr(spec, "stock_metal_fence_models", ())), geometry="line"))
        rules.append(_rule("metal-fences-by-material", ("barriers",), {"barrier": "fence", "material": ("metal", "steel", "iron", "wire", "wire_mesh")}, models=tuple(getattr(spec, "stock_metal_fence_models", ())), geometry="line"))
    if bool(getattr(spec, "bus_stops_enabled", False)):
        rules.append(_rule("bus-stop-signs", ("landmarks",), {"landmark": "bus_stop"}, models=(getattr(spec, "bus_stop_model", r"o\misc\aut_z_st.p3d"),), geometry="point"))
    if bool(getattr(spec, "cemeteries_enabled", False)):
        rules.append(_rule("cemetery-graves", ("sites",), {"site": "cemetery"}, models=tuple(getattr(spec, "grave_models", ())), geometry="polygon"))
    if bool(getattr(spec, "wetland_reeds_enabled", False)):
        rules.append(_rule("wetland-reeds", ("rural_vegetation",), {"natural": "wetland"}, models=tuple(getattr(spec, "wetland_reed_models", ())), geometry="polygon"))
    if milestone_number >= 9:
        rules.append(_rule(
            "mapped-individual-trees", ("individual_trees",), {"natural": "tree"},
            models=(
                r"data3d\str briza.p3d", r"data3d\str dub.p3d", r"data3d\str javor.p3d",
                r"data3d\str lipa.p3d", r"data3d\str vrba.p3d", r"data3d\str smrk.p3d",
                r"data3d\str borovice.p3d", r"data3d\str jedle.p3d",
            ), geometry="point", description="Stock CWA models used for individually mapped OSM trees",
        ))
        world_name = getattr(spec, "name", "cwr_world")
        rules.append(_rule(
            "osm-power-utilities", ("utility_points",), {"utility": ("power_pole", "power_tower")},
            models=(rf"{world_name}\i\util_power_pole.p3d", rf"{world_name}\i\util_power_tower.p3d"),
            geometry="point", description="Generated power pole/tower models",
        ))
        rules.append(_rule(
            "osm-water-towers", ("utility_points",), {"utility": "water_tower"},
            models=(rf"{world_name}\i\util_water_tower.p3d",), geometry="point",
            description="Generated water-tower model",
        ))
    if milestone_number < 8:
        rules.extend((
            _rule("industrial-buildings", ("building_polygons", "building_points"), {"building": ("industrial", "warehouse", "hangar", "factory")}, models=(getattr(spec, "industrial_building_model", r"o\hous\hangar_2.p3d"),)),
            _rule("urban-buildings", ("building_polygons", "building_points"), {"building": ("apartments", "commercial", "office", "retail", "hotel", "hospital")}, models=(getattr(spec, "urban_building_model", r"data3d\dum_mesto2.p3d"),)),
            _rule("generic-buildings", ("building_polygons", "building_points"), {"building": "*"}, models=(getattr(spec, "generic_building_model", r"o\hous\domek_sedy.p3d"),)),
        ))
    return OsmAssetMapping(
        rules=tuple(rule for rule in rules if rule.models or rule.textures),
        global_textures=tuple(dict.fromkeys(global_textures)),
        source="built-in defaults",
        inherit_defaults=True,
    )


def write_default_osm_asset_mapping(path: Path, mapping: OsmAssetMapping) -> None:
    document = {
        "schema": 1,
        "inherit_defaults": False,
        "global": {
            "models": list(mapping.global_models),
            "textures": list(mapping.global_textures),
        },
        "rules": [rule.to_manifest() for rule in mapping.rules],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
