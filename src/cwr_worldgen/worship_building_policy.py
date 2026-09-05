# SPDX-License-Identifier: GPL-3.0-or-later
"""Global semantic and appearance rules for places of worship.

Country profiles remain authoritative for ordinary architecture. Worship buildings
are a semantic exception: a church, mosque or synagogue must not accidentally
inherit a residential facade palette merely because it stands among houses. This
policy classifies worship types globally and applies conservative class-specific
materials/colours after country selection. Explicit OSM material, colour and roof
tags remain authoritative.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .building_semantics import is_actual_church, worship_building_class

_RULES_PATH = Path(__file__).with_name("data") / "worship_building_styles.json"
_NON_CHRISTIAN_WORSHIP = frozenset({
    "mosque",
    "synagogue",
    "temple",
    "shrine",
    "place_of_worship",
})
_COLOUR_METADATA_TAGS = (
    "building:colour",
    "building:color",
    "roof:colour",
    "roof:color",
)
_INSTALLED = False
_ORIGINAL_STYLE_CLASSIFIER = None
_ORIGINAL_CWR_FAMILY = None
_ORIGINAL_CWR_KEY_FOR = None
_ORIGINAL_RUNTIME_RESOLVE = None


@lru_cache(maxsize=1)
def load_worship_style_rules() -> Mapping[str, Mapping[str, Any]]:
    try:
        document = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load worship building style rules: {_RULES_PATH}") from exc
    if not isinstance(document, Mapping) or int(document.get("schema_version", 0)) != 1:
        raise RuntimeError("Unsupported worship building style rule schema")
    classes = document.get("classes") or {}
    if not isinstance(classes, Mapping):
        raise RuntimeError("Worship building style rules must contain a classes mapping")
    return {
        str(name): dict(value)
        for name, value in classes.items()
        if isinstance(value, Mapping)
    }


def _weighted_pick(
    values: object,
    seed: str,
    *,
    value_key: str,
    fallback: str,
) -> str:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return str(fallback or "")
    choices: list[tuple[str, float]] = []
    for entry in values:
        if not isinstance(entry, Mapping):
            continue
        value = str(entry.get(value_key, "") or "").strip()
        try:
            weight = max(0.0, float(entry.get("weight", 1.0)))
        except (TypeError, ValueError):
            weight = 0.0
        if value and weight > 0.0:
            choices.append((value, weight))
    if not choices:
        return str(fallback or "")
    total = sum(weight for _value, weight in choices)
    unit = int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big") / 2**64
    target = unit * total
    running = 0.0
    for value, weight in choices:
        running += weight
        if target < running:
            return value
    return choices[-1][0]


def _wall_thickness_for(material: str, fallback: float) -> float:
    value = str(material or "").casefold()
    if any(token in value for token in ("timber", "wood")):
        return 0.18
    if any(token in value for token in ("stone", "brick")):
        return 0.34
    if any(token in value for token in ("concrete", "cement", "precast")):
        return 0.26
    if any(token in value for token in ("render", "stucco", "plaster", "masonry")):
        return 0.28
    return max(0.12, min(0.55, float(fallback or 0.22)))


def _explicit_colour(tags: Mapping[str, str]) -> str:
    return str(tags.get("building:colour") or tags.get("building:color") or "").strip()


def _class_rules(building_class: str) -> Mapping[str, Any]:
    rules = load_worship_style_rules()
    return rules.get(building_class) or rules.get("place_of_worship") or {}


def apply_global_worship_style(
    choice,
    tags: Mapping[str, str],
    *,
    width_m: float,
    length_m: float,
    seed: str,
):
    """Apply global worship defaults after the selected country style."""
    building_class = worship_building_class(tags)
    if not building_class:
        return choice
    rules = _class_rules(building_class)
    if not rules:
        return replace(choice, building_class=building_class)

    signature = ":".join((
        str(seed or "cwr-worldgen"),
        building_class,
        str(getattr(choice, "country_profile_identifier", "") or "global"),
        str(getattr(choice, "context", "") or ""),
        f"{float(width_m):.2f}",
        f"{float(length_m):.2f}",
        str(tags.get("denomination", "") or ""),
    ))

    explicit_wall = bool(str(tags.get("building:material", "") or "").strip())
    explicit_roof = bool(str(tags.get("roof:material", "") or "").strip())
    explicit_shape = bool(str(tags.get("roof:shape", "") or "").strip())
    osm_colour = _explicit_colour(tags)

    wall_material = str(getattr(choice, "wall_material", "") or "")
    if not explicit_wall:
        wall_material = _weighted_pick(
            rules.get("wall_materials"),
            signature + ":wall",
            value_key="material",
            fallback=wall_material,
        )

    roof_material = str(getattr(choice, "roof_material", "") or "")
    if not explicit_roof:
        roof_material = _weighted_pick(
            rules.get("roof_materials"),
            signature + ":roof",
            value_key="material",
            fallback=roof_material,
        )

    roof_style = str(getattr(choice, "roof_style", "") or "gabled")
    if not explicit_shape:
        roof_style = _weighted_pick(
            rules.get("roof_styles"),
            signature + ":roof-style",
            value_key="style",
            fallback=roof_style,
        )

    if osm_colour:
        palette = (osm_colour,)
    else:
        primary = _weighted_pick(
            rules.get("facade_colours"),
            signature + ":facade-colour",
            value_key="colour",
            fallback="white",
        )
        # Do not carry a residential country palette behind the worship primary.
        # Texture renderers currently tint from the first colour, but keeping the
        # entire palette class-safe prevents a future renderer from reviving a
        # country house colour as a secondary church facade by accident.
        allowed: list[str] = []
        seen = {primary.casefold()}
        for entry in rules.get("facade_colours") or ():
            if isinstance(entry, Mapping):
                value = str(entry.get("colour", "") or "").strip()
                folded = value.casefold()
                if value and folded not in seen:
                    seen.add(folded)
                    allowed.append(value)
        palette = (primary, *allowed[:5])

    window_spec = dict(getattr(choice, "window_spec", {}) or {})
    try:
        density = max(0.0, float(rules.get("window_density_multiplier", 1.0)))
    except (TypeError, ValueError):
        density = 1.0
    window_spec["density_multiplier"] = density
    if building_class in {"church", "orthodox_church", "mosque", "synagogue"}:
        window_spec.setdefault("placement_style", "regular_aligned")

    return replace(
        choice,
        family="school",
        building_class=building_class,
        outbuilding_kind="",
        facade_style=str(rules.get("facade_style") or "worship"),
        wall_material=wall_material,
        roof_material=roof_material,
        roof_style=roof_style,
        wall_thickness_m=(
            float(getattr(choice, "wall_thickness_m", 0.22) or 0.22)
            if explicit_wall
            else _wall_thickness_for(
                wall_material,
                float(getattr(choice, "wall_thickness_m", 0.22) or 0.22),
            )
        ),
        colour_palette=tuple(palette),
        window_spec=window_spec,
    )


def _install_normalization_colour_metadata() -> None:
    """Keep explicit OSM facade/roof colours through normalized bundles."""
    from . import normalization

    existing = tuple(getattr(normalization, "_BUILDING_METADATA_TAGS", ()))
    normalization._BUILDING_METADATA_TAGS = tuple(dict.fromkeys((*existing, *_COLOUR_METADATA_TAGS)))


def _install_style_classification() -> None:
    global _ORIGINAL_STYLE_CLASSIFIER
    from . import osm_house_modeler_styles as styles

    _ORIGINAL_STYLE_CLASSIFIER = styles.classify_building

    def classify_building(
        tags: Mapping[str, str],
        width_m: float | None = None,
        length_m: float | None = None,
        *,
        settlement: str = "rural",
    ):
        building_class = worship_building_class(tags)
        if building_class:
            # The style system uses the school/public family as a neutral civic
            # envelope. CWR's engine family is selected separately below, so
            # Christian churches still retain their dedicated church geometry.
            return styles.BuildingClassification("school", building_class)
        return _ORIGINAL_STYLE_CLASSIFIER(
            tags,
            width_m,
            length_m,
            settlement=settlement,
        )

    styles.classify_building = classify_building


def _install_cwr_family_classification() -> None:
    global _ORIGINAL_CWR_FAMILY
    from . import procedural_buildings as buildings

    _ORIGINAL_CWR_FAMILY = buildings._family
    # ``procedural_buildings`` imported this helper by name. Point it at the new
    # semantic implementation so Orthodox churches remain church geometry.
    buildings.is_actual_church = is_actual_church

    def family(
        tags: Mapping[str, str],
        width_m: float | None = None,
        length_m: float | None = None,
        *,
        settlement_context: str = "rural",
    ) -> str:
        building_class = worship_building_class(tags)
        if building_class in _NON_CHRISTIAN_WORSHIP:
            # Use the existing civic/school shell and interior rather than the
            # Christian tower/spire family. Class-specific visual details can be
            # layered later without lying about the underlying religion today.
            return "school"
        return _ORIGINAL_CWR_FAMILY(
            tags,
            width_m,
            length_m,
            settlement_context=settlement_context,
        )

    buildings._family = family


def _install_cwr_key_adjustment() -> None:
    """Let Orthodox churches use the roof shape selected by their global rule."""
    global _ORIGINAL_CWR_KEY_FOR
    from . import procedural_buildings as buildings

    _ORIGINAL_CWR_KEY_FOR = buildings.ProceduralBuildingLibrary.key_for

    def key_for(
        self,
        tags: Mapping[str, str],
        width_m: float,
        length_m: float,
        *,
        foundation_depth_m: float | None = None,
        settlement_context: str = "rural",
    ):
        key = _ORIGINAL_CWR_KEY_FOR(
            self,
            tags,
            width_m,
            length_m,
            foundation_depth_m=foundation_depth_m,
            settlement_context=settlement_context,
        )
        building_class = worship_building_class(tags)
        if building_class != "orthodox_church" or str(tags.get("roof:shape", "") or "").strip():
            return key
        rules = _class_rules(building_class)
        roof_style = _weighted_pick(
            rules.get("roof_styles"),
            ":".join((
                str(getattr(self, "world_name", "cwr-worldgen") or "cwr-worldgen"),
                building_class,
                str(getattr(key, "country_style_identifier", "") or "global"),
                f"{float(width_m):.2f}",
                f"{float(length_m):.2f}",
                str(tags.get("denomination", "") or ""),
                "cwr-roof",
            )),
            value_key="style",
            fallback=str(getattr(key, "roof_style", "gabled") or "gabled"),
        )
        return replace(key, roof_style=roof_style, building_class=building_class)

    buildings.ProceduralBuildingLibrary.key_for = key_for


def _install_runtime_style_override() -> None:
    global _ORIGINAL_RUNTIME_RESOLVE
    from . import osm_house_modeler_runtime as runtime

    _ORIGINAL_RUNTIME_RESOLVE = runtime.resolve_style

    def resolve_style_with_worship(*args, **kwargs):
        choice = _ORIGINAL_RUNTIME_RESOLVE(*args, **kwargs)
        tags = kwargs.get("tags") or {}
        if not isinstance(tags, Mapping):
            return choice
        return apply_global_worship_style(
            choice,
            tags,
            width_m=float(kwargs.get("width_m", 0.0) or 0.0),
            length_m=float(kwargs.get("length_m", 0.0) or 0.0),
            seed=str(kwargs.get("seed", "cwr-worldgen") or "cwr-worldgen"),
        )

    runtime.resolve_style = resolve_style_with_worship


def install_worship_building_policy() -> None:
    """Install global worship semantics after country/visual style policies."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_normalization_colour_metadata()
    _install_style_classification()
    _install_cwr_family_classification()
    _install_runtime_style_override()
    _install_cwr_key_adjustment()
    _INSTALLED = True
