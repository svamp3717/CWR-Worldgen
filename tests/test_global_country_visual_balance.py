from __future__ import annotations

import json
from pathlib import Path

from cwr_worldgen.country_utility_material_policy import apply_country_utility_materials
from cwr_worldgen.osm_house_modeler_styles import (
    StyleChoice,
    choose_style,
    load_country_profiles,
    load_profiles,
)

ROOT = Path(__file__).resolve().parents[1]
COUNTRY_DIR = ROOT / "src" / "cwr_worldgen" / "country_styles"


def _values(entries, key: str) -> set[str]:
    return {str(entry[key]).casefold() for entry in entries}


def test_all_249_country_contexts_have_explicit_ordinary_visual_distributions() -> None:
    profiles = []
    for path in sorted(COUNTRY_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("iso_alpha2"):
            profiles.append((path, document))
    assert len(profiles) == 249
    context_count = 0
    for path, document in profiles:
        for context in document["contexts"].values():
            materials = context["architectural_details"]["materials"]
            walls = [str(value) for value in materials.get("common_wall_materials", []) if str(value)]
            wall_distribution = materials.get("common_wall_material_distribution", [])
            if walls:
                assert wall_distribution, path.name
                assert _values(wall_distribution, "material") <= {value.casefold() for value in walls}
                assert all(float(entry["weight"]) > 0 for entry in wall_distribution)

            colours = [str(value) for value in materials.get("typical_colour_palette", []) if str(value)]
            colour_distribution = materials.get("facade_colour_distribution", [])
            if colours:
                assert colour_distribution, path.name
                assert _values(colour_distribution, "colour") <= {value.casefold() for value in colours}
                assert all(float(entry["weight"]) > 0 for entry in colour_distribution)
                if len(colour_distribution) >= 3:
                    total = sum(float(entry["weight"]) for entry in colour_distribution)
                    assert max(float(entry["weight"]) for entry in colour_distribution) / total <= 0.55

            assert materials["global_visual_balance_revision"] == "2026-09-global-country-visual-balance-v1"
            context_count += 1
    assert context_count == 498


def _choice_for(country_identifier: str, context: str = "rural") -> StyleChoice:
    profile = next(p for p in load_country_profiles() if p.identifier == country_identifier)
    raw = profile.contexts[context]
    materials = raw["architectural_details"]["materials"]
    walls = materials.get("common_wall_materials") or ["stucco/render"]
    roofs = materials.get("common_roof_materials") or ["tile"]
    palette = tuple(str(value) for value in materials.get("typical_colour_palette") or ["cream"])
    return StyleChoice(
        region_identifier=profile.parent_region_identifier,
        region_name=profile.parent_region_identifier,
        facade_style=str(raw["selection"].get("default_style", "default")),
        roof_style=str(raw["roof_defaults"].get("residential", "gabled")),
        context=context,
        family="residential",
        building_class="residential",
        country_code=profile.iso_alpha2,
        country_name=profile.display_name,
        country_profile_identifier=profile.identifier,
        wall_material=str(walls[0]),
        roof_material=str(roofs[0]),
        colour_palette=palette,
    )


def test_representative_countries_are_not_pinned_to_first_colour_or_material() -> None:
    for country_identifier in ("se_sweden", "ke_kenya", "jp_japan"):
        base = _choice_for(country_identifier)
        colours: set[str] = set()
        walls: set[str] = set()
        for index in range(180):
            tuned = apply_country_utility_materials(
                base,
                {},
                seed=f"global-balance-{country_identifier}",
                width_m=6.0 + index * 0.11,
                length_m=8.0 + (index % 29) * 0.13,
            )
            colours.add(tuned.colour_palette[0])
            walls.add(tuned.wall_material)
        assert len(colours) >= 2, country_identifier
        assert len(walls) >= 2, country_identifier


def test_sweden_is_country_style_only_and_uses_northern_europe_parent() -> None:
    assert not (ROOT / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json").exists()
    regions = load_profiles()
    assert len(regions) == 23
    assert all(profile.identifier != "sweden" for profile in regions)

    sweden = next(profile for profile in load_country_profiles() if profile.identifier == "se_sweden")
    assert sweden.parent_region_identifier == "northern_europe"

    choice = choose_style(
        regions,
        15.0,
        62.0,
        {"building": "house", "addr:country": "SE"},
        12345,
        width_m=10.0,
        length_m=14.0,
        seed="sweden-country-only",
    )
    assert choice.country_profile_identifier == "se_sweden"
    assert choice.region_identifier == "northern_europe"


def test_sweden_barns_are_red_dominant_and_stucco_avoids_timber_red_bias() -> None:
    profile = next(p for p in load_country_profiles() if p.identifier == "se_sweden")
    rural = profile.contexts["rural"]
    materials = rural["architectural_details"]["materials"]
    barn_colours = materials["building_class_overrides"]["barn"]["facade_colour_distribution"]
    weights = {str(entry["colour"]): float(entry["weight"]) for entry in barn_colours}
    assert weights["falun red"] >= 65
    assert weights["falun red"] == max(weights.values())

    barn_base = StyleChoice(
        region_identifier="northern_europe", region_name="Northern Europe",
        facade_style="swedish_wood", roof_style="gabled", context="rural",
        family="agricultural", building_class="barn",
        country_code="SE", country_name="Sweden",
        country_profile_identifier="se_sweden",
        wall_material="utility painted timber board cladding",
        roof_material="utility sheet metal roof",
        colour_palette=("falun red", "ochre yellow", "white", "cream", "grey", "dark green", "natural timber"),
    )
    sampled = []
    for index in range(300):
        tuned = apply_country_utility_materials(
            barn_base, {}, seed="sweden-red-barns",
            width_m=6.0 + index * 0.031, length_m=18.0 + (index % 41) * 0.17,
        )
        sampled.append(tuned.colour_palette[0])
    assert sampled.count("falun red") / len(sampled) >= 0.58

    stucco_base = StyleChoice(
        region_identifier="northern_europe", region_name="Northern Europe",
        facade_style="western_stucco", roof_style="gabled", context="rural",
        family="residential", building_class="residential",
        country_code="SE", country_name="Sweden",
        country_profile_identifier="se_sweden",
        wall_material="stucco/render", roof_material="clay/concrete tile",
        colour_palette=("falun red", "ochre yellow", "white", "cream", "grey", "dark green", "natural timber"),
    )
    stucco_colours = []
    for index in range(240):
        tuned = apply_country_utility_materials(
            stucco_base, {"building:material": "stucco"},
            seed="sweden-stucco-colours",
            width_m=8.0 + index * 0.021, length_m=10.0 + (index % 37) * 0.11,
        )
        stucco_colours.append(tuned.colour_palette[0])
    neutral = sum(colour in {"cream", "white", "grey"} for colour in stucco_colours)
    assert neutral / len(stucco_colours) >= 0.78
    assert stucco_colours.count("falun red") / len(stucco_colours) <= 0.06
