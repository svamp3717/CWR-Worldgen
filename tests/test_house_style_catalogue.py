from __future__ import annotations

from pathlib import Path

from cwr_worldgen.house_style_catalogue import (
    HOUSE_STYLE_PRESET_AUTO,
    HOUSE_STYLE_PRESET_IDENTIFIERS,
    REGION_PROFILES,
    get_house_style_context,
    house_style_preset_profile,
    normalise_house_style_preset,
    select_regional_style,
)
from cwr_worldgen.osm_house_modeler_styles import (
    choose_country,
    load_country_profiles,
    load_profiles,
)


def test_modeler_region_catalogue_excludes_sweden_country_duplicate() -> None:
    assert [profile.map_region_number for profile in REGION_PROFILES] == list(range(1, 24))
    assert len(load_profiles()) == 23
    sweden_file = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json"
    assert not sweden_file.exists()
    assert all(profile.identifier != "sweden" for profile in load_profiles())


def test_country_catalogue_contains_all_modeler_profiles() -> None:
    countries = load_country_profiles()
    assert len(countries) == 249
    sweden = choose_country(countries, 15.0, 62.0, {})
    assert sweden is not None
    assert sweden.iso_alpha2 == "SE"
    assert sweden.parent_region_identifier == "northern_europe"
    assert sweden.detail_level == "country-expanded-curated"
    context = get_house_style_context(sweden.identifier, "rural")
    assert context is not None
    assert context.selection["default_style"]
    assert context.roof_defaults["residential"]


def test_explicit_country_code_overrides_coordinate_guess() -> None:
    countries = load_country_profiles()
    sweden = choose_country(countries, -100.0, 40.0, {"addr:country": "SE"})
    assert sweden is not None and sweden.iso_alpha2 == "SE"


def test_country_context_drives_existing_cwr_style_selector() -> None:
    styles = {
        select_regional_style(
            "se_sweden", "residential", {"building": "house", "name": f"House {index}"},
            10.0, 16.0, settlement_context="rural",
        )
        for index in range(80)
    }
    assert "swedish_wood" in styles


def test_house_style_preset_catalogue_exposes_23_regions_without_sweden_duplicate() -> None:
    assert HOUSE_STYLE_PRESET_AUTO == "auto"
    assert HOUSE_STYLE_PRESET_IDENTIFIERS == tuple(
        profile.house_style_identifier for profile in REGION_PROFILES
    )
    assert len(HOUSE_STYLE_PRESET_IDENTIFIERS) == 23
    assert "sweden" not in HOUSE_STYLE_PRESET_IDENTIFIERS
    assert normalise_house_style_preset("AUTO") == "auto"
    assert normalise_house_style_preset("east_asia") == "east_asia"
    assert house_style_preset_profile("auto") is None
    assert house_style_preset_profile("east_asia").display_name == "East Asia"
