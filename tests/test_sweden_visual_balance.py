from __future__ import annotations

from collections import Counter

from cwr_worldgen.country_utility_material_policy import apply_country_utility_materials
from cwr_worldgen.osm_house_modeler_styles import StyleChoice, load_country_profiles
from cwr_worldgen.osm_house_modeler_texture_bridge import modeler_roof_texture_image


def _choice(*, building_class: str = "residential", family: str = "residential") -> StyleChoice:
    return StyleChoice(
        region_identifier="sweden",
        region_name="Sweden",
        facade_style="swedish_wood",
        roof_style="gabled",
        context="rural",
        family=family,
        building_class=building_class,
        country_code="SE",
        country_name="Sweden",
        country_profile_identifier="se_sweden",
        wall_material="brick",
        roof_material="standing-seam metal",
        colour_palette=(
            "falun red", "ochre yellow", "white", "cream",
            "dark green", "grey", "black", "natural timber",
        ),
    )


def test_sweden_profile_has_explicit_visual_balance_data_in_both_contexts() -> None:
    sweden = next(profile for profile in load_country_profiles() if profile.identifier == "se_sweden")
    for context in ("rural", "town_city"):
        materials = sweden.contexts[context]["architectural_details"]["materials"]
        assert materials["common_wall_material_distribution"]
        colours = materials["facade_colour_distribution"]
        red = next(item["weight"] for item in colours if item["colour"] == "falun red")
        assert red <= 24
        barn = materials["building_class_overrides"]["barn"]["wall_materials"]
        brick = next(item["weight"] for item in barn if item["material"] == "utility structural brick")
        assert brick <= 6


def test_sweden_weighted_facade_colours_are_not_stuck_on_falun_red() -> None:
    colours = Counter()
    materials = Counter()
    base = _choice()
    for index in range(240):
        tuned = apply_country_utility_materials(
            base,
            {},
            seed="SwedenBalance",
            width_m=6.0 + index * 0.07,
            length_m=8.0 + (index % 17) * 0.11,
        )
        colours[tuned.colour_palette[0]] += 1
        materials[tuned.wall_material] += 1
    assert len(colours) >= 5
    assert colours["falun red"] < 90
    assert materials["brick"] < 70
    assert materials["painted vertical timber cladding"] > materials["brick"]


def test_sweden_barn_brick_is_rare_and_osm_overrides_still_win() -> None:
    base = _choice(building_class="barn", family="agricultural")
    walls = Counter()
    for index in range(300):
        tuned = apply_country_utility_materials(
            base,
            {},
            seed="SwedenBarnBalance",
            width_m=8.0 + index * 0.09,
            length_m=15.0 + (index % 23) * 0.13,
        )
        walls[tuned.wall_material] += 1
    assert walls["utility structural brick"] < 35
    assert walls["utility painted timber board cladding"] > 150

    resolved = __import__("dataclasses").replace(
        base, wall_material="brick", roof_material="tile"
    )
    explicit = apply_country_utility_materials(
        resolved,
        {"building:material": "brick", "building:colour": "white", "roof:material": "tile"},
        seed="SwedenExplicit",
        width_m=12.0,
        length_m=24.0,
    )
    assert explicit.wall_material == "brick"
    assert explicit.roof_material == "tile"
    assert explicit.colour_palette[0] == "white"


def test_roof_material_is_not_repainted_by_facade_palette() -> None:
    red = modeler_roof_texture_image("gabled|standing-seam metal|falun red,white", size=128)
    yellow = modeler_roof_texture_image("gabled|standing-seam metal|ochre yellow,white", size=128)
    assert red.tobytes() == yellow.tobytes()
