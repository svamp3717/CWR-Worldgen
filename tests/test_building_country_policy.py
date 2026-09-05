from __future__ import annotations

from pathlib import Path

import cwr_worldgen.cli as cli
import cwr_worldgen.gui as gui
import cwr_worldgen.gui_entry as gui_entry
from cwr_worldgen import osm_house_modeler_runtime as runtime
from cwr_worldgen.building_country_policy import (
    BUILDING_COUNTRY_AUTO_LABEL,
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


def test_gui_replaces_region_dropdown_with_country_dropdown(tmp_path: Path) -> None:
    gui_entry._configure_gui(gui, tmp_path)

    assert gui.HOUSE_STYLE_PRESET_LABELS[0] == BUILDING_COUNTRY_AUTO_LABEL
    assert "SE — Sweden" in gui.HOUSE_STYLE_PRESET_LABELS
    assert not any("East Asia" in label for label in gui.HOUSE_STYLE_PRESET_LABELS)
    assert gui.gui_house_style_preset_identifier("SE — Sweden") == "se_sweden"
    assert gui.gui_house_style_preset_identifier("east_asia") == "auto"

    values = gui.default_gui_values()
    values["house_style_preset"] = "SE — Sweden"
    command = gui.build_milestone9_command(values, python="python")
    index = command.index("--house-style-preset")
    assert command[index + 1] == "se_sweden"
