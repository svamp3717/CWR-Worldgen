# SPDX-License-Identifier: GPL-3.0-or-later
"""Expose country, rather than broad region, as the explicit building-style override."""
from __future__ import annotations

from pathlib import Path
from typing import Any

BUILDING_COUNTRY_AUTO = "auto"
BUILDING_COUNTRY_AUTO_LABEL = "Automatic (area / country)"
_COUNTRY_MARKER_PREFIX = "country:"
_INSTALLED = False


def _country_profiles():
    from .osm_house_modeler_styles import load_country_profiles

    return tuple(sorted(load_country_profiles(), key=lambda item: (item.display_name.casefold(), item.iso_alpha2)))


def _country_profile(value: object):
    from .osm_house_modeler_styles import find_country_profile

    text = str(value or "").strip()
    if not text or text.casefold() == BUILDING_COUNTRY_AUTO:
        return None
    return find_country_profile(_country_profiles(), text)


def normalise_building_country(value: object) -> str:
    """Return ``auto`` or the canonical country-profile identifier."""
    profile = _country_profile(value)
    return BUILDING_COUNTRY_AUTO if profile is None else profile.identifier


def building_country_options() -> tuple[tuple[str, str], ...]:
    """Return stable country identifiers and human-readable ISO-labelled names."""
    from .osm_house_modeler_styles import country_selector_label

    return tuple((profile.identifier, country_selector_label(profile)) for profile in _country_profiles())


def _install_catalogue_transport() -> tuple[str, ...]:
    """Let the existing transport field carry countries without changing region APIs."""
    from . import house_style_catalogue as catalogue
    from . import procedural_buildings as buildings
    from . import osm_house_modeler_runtime as runtime
    from . import cli

    old_region_identifiers = tuple(catalogue.HOUSE_STYLE_PRESET_IDENTIFIERS)
    country_identifiers = tuple(identifier for identifier, _label in building_country_options())
    original_normalise = buildings.normalise_house_style_preset
    original_profile = runtime.house_style_preset_profile

    def normalise_transport(value: str | None) -> str:
        try:
            return normalise_building_country(value)
        except ValueError:
            # Preserve programmatic compatibility for callers that still carry an
            # old region id, but the GUI/CLI no longer offer those values.
            return original_normalise(value)

    def profile_for_transport(value: str | None):
        try:
            country = _country_profile(value)
        except ValueError:
            return original_profile(value)
        if country is not None:
            return None
        return original_profile(value)

    # cli.py has already been imported during package initialization, but parser
    # construction happens later in cli.main(). Replace only its choice tuple so
    # the public command line now accepts countries rather than broad regions.
    cli.HOUSE_STYLE_PRESET_IDENTIFIERS = country_identifiers

    # These modules imported the helpers by name before this policy is installed.
    buildings.normalise_house_style_preset = normalise_transport
    buildings.house_style_preset_profile = profile_for_transport
    runtime.house_style_preset_profile = profile_for_transport
    return old_region_identifiers


def _install_runtime_country_override() -> None:
    from . import osm_house_modeler_runtime as runtime
    from . import procedural_buildings as buildings

    original_resolve_style = runtime.resolve_style
    original_prepare = buildings.ProceduralBuildingLibrary._prepare_geographic_context
    original_regional_preset = runtime._regional_preset

    def regional_preset(library) -> str:
        raw = getattr(library, "house_style_preset", BUILDING_COUNTRY_AUTO)
        try:
            requested = normalise_building_country(raw)
        except ValueError:
            return original_regional_preset(library)
        if requested == BUILDING_COUNTRY_AUTO:
            return BUILDING_COUNTRY_AUTO
        return _COUNTRY_MARKER_PREFIX + requested

    def resolve_style_with_country(*args, **kwargs):
        marker = str(kwargs.get("regional_preset", BUILDING_COUNTRY_AUTO) or BUILDING_COUNTRY_AUTO)
        if marker.startswith(_COUNTRY_MARKER_PREFIX):
            identifier = marker[len(_COUNTRY_MARKER_PREFIX):]
            profile = _country_profile(identifier)
            if profile is None:
                raise ValueError(f"Unknown building country {identifier!r}")
            kwargs = dict(kwargs)
            tags = dict(kwargs.get("tags") or {})
            # choose_country() gives explicit country tags priority over geometry.
            # This therefore selects the requested country while preserving the
            # real building coordinate for every other geographic calculation.
            tags["addr:country"] = profile.iso_alpha2
            kwargs["tags"] = tags
            kwargs["regional_preset"] = BUILDING_COUNTRY_AUTO
        return original_resolve_style(*args, **kwargs)

    def prepare_geographic_context(self, dataset, projection):
        result = original_prepare(self, dataset, projection)
        try:
            profile = _country_profile(getattr(self, "house_style_preset", BUILDING_COUNTRY_AUTO))
        except ValueError:
            profile = None
        if profile is not None:
            # Keep the detected region separately for compatibility, but report
            # the explicit country as the style actually driving generated assets.
            self.country_style_identifier = profile.identifier
            self.house_style_identifier = profile.identifier
        return result

    runtime._regional_preset = regional_preset
    runtime.resolve_style = resolve_style_with_country
    buildings.ProceduralBuildingLibrary._prepare_geographic_context = prepare_geographic_context


def _widget_text(widget: Any) -> str:
    try:
        return str(widget.cget("text"))
    except Exception:
        return ""


def _replace_building_preset_labels(widget: Any) -> None:
    text = _widget_text(widget)
    if text:
        replaced = text.replace("Building preset", "Building country")
        replaced = replaced.replace(
            "Choose one of the 23 regional presets here to override procedural building façades and roof defaults for the entire world.",
            "Choose a country here to use that country's procedural building architecture for the entire world.",
        )
        if replaced != text:
            try:
                widget.configure(text=replaced)
            except Exception:
                pass
    try:
        children = widget.winfo_children()
    except Exception:
        children = ()
    for child in children:
        _replace_building_preset_labels(child)


def _install_gui_country_dropdown(old_region_identifiers: tuple[str, ...]) -> None:
    from . import gui_entry

    original_configure = gui_entry._configure_gui
    country_options = building_country_options()
    country_labels = (BUILDING_COUNTRY_AUTO_LABEL, *(label for _identifier, label in country_options))
    label_to_identifier = {label: identifier for identifier, label in country_options}
    identifier_to_label = {identifier: label for identifier, label in country_options}
    old_regions = frozenset(value.casefold() for value in old_region_identifiers)

    def configure_gui(gui: Any, base_dir: Path) -> None:
        original_configure(gui, base_dir)

        # gui.py still uses the historical variable/function names so profiles and
        # command construction remain compatible, but every visible option is now
        # a country. Saved regional selections migrate to Automatic.
        gui.HOUSE_STYLE_PRESET_IDENTIFIERS = tuple(identifier_to_label)
        gui.HOUSE_STYLE_PRESET_OPTIONS = country_options
        gui.HOUSE_STYLE_PRESET_LABELS = country_labels
        gui._HOUSE_STYLE_LABEL_TO_IDENTIFIER = label_to_identifier
        gui._HOUSE_STYLE_IDENTIFIER_TO_LABEL = identifier_to_label

        def country_identifier(value: object) -> str:
            text = str(value or "").strip()
            if not text or text == BUILDING_COUNTRY_AUTO_LABEL or text.casefold() == BUILDING_COUNTRY_AUTO:
                return BUILDING_COUNTRY_AUTO
            identifier = label_to_identifier.get(text)
            if identifier is not None:
                return identifier
            folded = text.casefold()
            if folded in identifier_to_label:
                return folded
            try:
                return normalise_building_country(text)
            except ValueError:
                if folded in old_regions:
                    return BUILDING_COUNTRY_AUTO
                raise ValueError(f"Unknown building country: {text}")

        def country_label(value: object) -> str:
            identifier = country_identifier(value)
            if identifier == BUILDING_COUNTRY_AUTO:
                return BUILDING_COUNTRY_AUTO_LABEL
            return identifier_to_label[identifier]

        gui.gui_house_style_preset_identifier = country_identifier
        gui.gui_house_style_preset_label = country_label

        original_class = gui.WorldgenGui

        class CountryWorldgenGui(original_class):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                _replace_building_preset_labels(self)

            def _refresh_views(self) -> None:
                super()._refresh_views()
                _replace_building_preset_labels(self)

        gui.WorldgenGui = CountryWorldgenGui

    gui_entry._configure_gui = configure_gui


def install_building_country_policy() -> None:
    """Replace the explicit broad-region building preset with a country selector."""
    global _INSTALLED
    if _INSTALLED:
        return
    old_regions = _install_catalogue_transport()
    _install_runtime_country_override()
    _install_gui_country_dropdown(old_regions)
    _INSTALLED = True
