from __future__ import annotations

from cwr_worldgen.house_style_catalogue import (
    HOUSE_STYLE_PRESET_AUTO,
    HOUSE_STYLE_PRESET_IDENTIFIERS,
    REGION_PROFILES,
    get_house_style_context,
    house_style_preset_profile,
    normalise_house_style_preset,
    select_regional_style,
    settlement_style_description,
)


def test_all_24_regions_have_rural_and_town_city_contexts() -> None:
    assert [profile.map_region_number for profile in REGION_PROFILES] == list(range(1, 25))
    assert all(profile.contexts is not None for profile in REGION_PROFILES)
    assert all(set(profile.contexts or ()) == {"rural", "town_city"} for profile in REGION_PROFILES)
    assert all((profile.contexts or {})["rural"].description for profile in REGION_PROFILES)
    assert all((profile.contexts or {})["town_city"].description for profile in REGION_PROFILES)


def test_town_and_city_use_the_town_city_reference_profile() -> None:
    expected = "Shophouses and narrow mixed-use urban row buildings."
    assert settlement_style_description("southeast_asia", "town") == expected
    assert settlement_style_description("southeast_asia", "city") == expected
    assert settlement_style_description("southeast_asia", "urban") == expected
    assert settlement_style_description("southeast_asia", "village") != expected


def test_northern_europe_city_profile_prefers_brick_and_compact_urban_materials() -> None:
    context = get_house_style_context("northern_europe", "city")
    assert context is not None
    urban_styles = {style for _threshold, style in context.selection["family_distributions"]["urban"]}
    assert "western_brick" in urban_styles
    assert "eastern_panel" in urban_styles
    assert "sweden_red" not in urban_styles


def test_settlement_context_is_used_by_regional_style_selection() -> None:
    # Northern Europe is deliberately the strongest contrast in the supplied
    # references: rural detached timber homes versus brick rowhouses/apartments.
    rural_styles = {
        select_regional_style(
            "northern_europe", "residential", {"building": "house", "name": f"House {index}"},
            10.0, 16.0, settlement_context="rural",
        )
        for index in range(80)
    }
    city_styles = {
        select_regional_style(
            "northern_europe", "residential", {"building": "house", "name": f"House {index}"},
            10.0, 16.0, settlement_context="city",
        )
        for index in range(80)
    }
    assert "sweden_red" not in rural_styles
    assert "sweden_red" not in city_styles
    assert "western_brick" in rural_styles
    assert "western_brick" in city_styles
    rural_sequence = [
        select_regional_style(
            "northern_europe", "residential", {"building": "house", "name": f"House {index}"},
            10.0, 16.0, settlement_context="rural",
        )
        for index in range(80)
    ]
    city_sequence = [
        select_regional_style(
            "northern_europe", "residential", {"building": "house", "name": f"House {index}"},
            10.0, 16.0, settlement_context="city",
        )
        for index in range(80)
    ]
    assert rural_sequence != city_sequence


def test_house_style_preset_catalogue_exposes_auto_plus_24_precise_regions() -> None:
    assert HOUSE_STYLE_PRESET_AUTO == "auto"
    assert HOUSE_STYLE_PRESET_IDENTIFIERS == tuple(
        profile.house_style_identifier for profile in REGION_PROFILES
    )
    assert len(HOUSE_STYLE_PRESET_IDENTIFIERS) == 24
    assert normalise_house_style_preset("AUTO") == "auto"
    assert normalise_house_style_preset("east_asia") == "east_asia"
    assert house_style_preset_profile("auto") is None
    assert house_style_preset_profile("east_asia").display_name == "East Asia"


def test_sweden_is_a_dedicated_region_24_preset() -> None:
    sweden = house_style_preset_profile("sweden")
    northern = house_style_preset_profile("northern_europe")
    assert sweden is not None and northern is not None
    assert sweden.map_region_number == 24
    assert sweden.display_name == "Sweden"
    assert "sweden" in sweden.country_aliases
    assert "sweden" not in northern.country_aliases
    rural_styles = {
        style
        for entries in sweden.contexts["rural"].selection["family_distributions"].values()
        for _threshold, style in entries
    }
    northern_styles = {
        style
        for entries in northern.contexts["rural"].selection["family_distributions"].values()
        for _threshold, style in entries
    }
    assert "sweden_red" in rural_styles
    assert "sweden_red" not in northern_styles


def test_unknown_house_style_preset_is_rejected() -> None:
    try:
        normalise_house_style_preset("moon_base")
    except ValueError as exc:
        assert "unknown house-style preset" in str(exc)
    else:
        raise AssertionError("invalid preset unexpectedly accepted")
