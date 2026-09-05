from __future__ import annotations

import cwr_worldgen.cli as cli
from cwr_worldgen import osm_house_modeler_runtime as runtime
from cwr_worldgen.building_country_policy import (
    _replace_building_preset_labels,
    building_country_options,
    normalise_building_country,
)


def test_country_catalogue_drives_public_building_override_choices() -> None:
    options = dict(building_country_options())
    assert len(options) == 249
    assert options["se_sweden"] == "SE — Sweden"
    assert normalise_building_country("SE") == "se_sweden"
    assert normalise_building_country("SE — Sweden") == "se_sweden"
    assert "se_sweden" in cli.HOUSE_STYLE_PRESET_IDENTIFIERS
    assert "east_asia" not in cli.HOUSE_STYLE_PRESET_IDENTIFIERS


def test_forced_country_overrides_map_location() -> None:
    class Library:
        house_style_preset = "se_sweden"

    marker = runtime._regional_preset(Library())
    assert marker == "country:se_sweden"
    choice = runtime.resolve_style(
        tags={"building": "house"},
        latitude=40.0,
        longitude=-100.0,
        width_m=10.0,
        length_m=16.0,
        settlement_context="rural",
        regional_preset=marker,
        seed="country-selector-regression",
    )
    assert choice.country_code == "SE"
    assert choice.country_profile_identifier == "se_sweden"
    assert choice.region_identifier == "northern_europe"


def test_country_selector_relabels_region_copy_without_tk_state() -> None:
    class Widget:
        def __init__(self, text: str = "", children=()):
            self.text = text
            self.children = list(children)

        def cget(self, name: str):
            assert name == "text"
            return self.text

        def configure(self, **kwargs):
            self.text = str(kwargs["text"])

        def winfo_children(self):
            return list(self.children)

    title = Widget("Building preset")
    hint = Widget(
        "Automatic uses the selected map area/country. Choose one of the 23 regional presets here to override procedural building façades and roof defaults for the entire world."
    )
    root = Widget(children=(title, hint))

    _replace_building_preset_labels(root)

    assert title.text == "Building country"
    assert "regional presets" not in hint.text
    assert "Choose a country" in hint.text
