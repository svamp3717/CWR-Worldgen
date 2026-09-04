from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cwr_worldgen.country_utility_material_policy import apply_country_utility_materials
from cwr_worldgen.osm_house_modeler_styles import choose_style, load_country_profiles, load_profiles

EXPECTED_CLASSES = {"barn", "shed", "garage", "warehouse", "hangar", "industrial"}
REVISION = "2026-09-country-utility-materials-v1"


def _distribution(block, name: str):
    values = block.get(name)
    assert isinstance(values, list) and values
    for row in values:
        assert isinstance(row, dict)
        assert str(row.get("material", "")).startswith("utility ")
        assert float(row.get("weight", 0.0)) > 0.0
    return values


def test_all_249_country_profiles_have_explicit_utility_material_pools() -> None:
    profiles = load_country_profiles()
    assert len(profiles) == 249
    for profile in profiles:
        assert profile.contexts
        for context_name, context in profile.contexts.items():
            details = context.get("architectural_details") or {}
            materials = details.get("materials") or {}
            assert materials.get("utility_materials_revision") == REVISION, (
                profile.identifier,
                context_name,
            )
            overrides = materials.get("building_class_overrides") or {}
            assert EXPECTED_CLASSES <= set(overrides), (profile.identifier, context_name)
            for building_class in EXPECTED_CLASSES:
                block = overrides[building_class]
                _distribution(block, "wall_materials")
                _distribution(block, "roof_materials")


def _sweden_choice(building: str):
    profiles = load_profiles()
    tags = {"building": building, "addr:country": "SE"}
    return choose_style(
        profiles,
        15.0,
        62.0,
        tags,
        12345,
        "rural",
        "auto",
        country_preset="auto",
        width_m=12.0,
        length_m=20.0,
        seed="utility-material-test",
    ), tags


def test_barn_consumes_explicit_sweden_material_pool() -> None:
    choice, tags = _sweden_choice("barn")
    selected = apply_country_utility_materials(
        choice,
        tags,
        seed="utility-material-test",
        width_m=12.0,
        length_m=20.0,
    )
    assert selected.building_class == "barn"
    assert selected.wall_material.startswith("utility ")
    assert selected.roof_material.startswith("utility ")
    assert selected.wall_material != choice.wall_material or selected.roof_material != choice.roof_material


def test_residential_choice_is_not_given_utility_materials() -> None:
    choice, tags = _sweden_choice("house")
    selected = apply_country_utility_materials(
        choice,
        tags,
        seed="utility-material-test",
        width_m=10.0,
        length_m=8.0,
    )
    assert selected.building_class == choice.building_class
    assert not selected.wall_material.startswith("utility ")
    assert not selected.roof_material.startswith("utility ")
    assert set(selected.colour_palette) == set(choice.colour_palette)


def test_explicit_osm_material_tags_override_country_utility_defaults() -> None:
    choice, _tags = _sweden_choice("warehouse")
    explicit = replace(choice, wall_material="mapped brick", roof_material="mapped slate")
    tags = {
        "building": "warehouse",
        "addr:country": "SE",
        "building:material": "brick",
        "roof:material": "slate",
    }
    selected = apply_country_utility_materials(
        explicit,
        tags,
        seed="utility-material-test",
        width_m=28.0,
        length_m=60.0,
    )
    assert selected.wall_material == "mapped brick"
    assert selected.roof_material == "mapped slate"


def test_country_files_are_the_source_of_truth_not_a_runtime_fallback() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "cwr_worldgen" / "country_styles" / "SE_Sweden.json"
    text = path.read_text(encoding="utf-8")
    assert '"building_class_overrides"' in text
    assert '"utility_materials_revision": "2026-09-country-utility-materials-v1"' in text
