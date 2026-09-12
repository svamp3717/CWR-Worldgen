# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional GUI/CLI gates for generated parking, sports, and runway surfaces.

These controls deliberately sit above the existing surface renderers. All three
features remain enabled by default, preserving existing worlds. Disabling one
prevents its world-local terrain textures from being generated while leaving the
underlying OSM feature available to the rest of the terrain pipeline.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


DYNAMIC_PARKING_KEY = "dynamic_parking_lots"
DYNAMIC_SPORTS_KEY = "dynamic_sports_fields"
DYNAMIC_RUNWAYS_KEY = "dynamic_runways"

DYNAMIC_SURFACE_OPTIONS: tuple[tuple[str, str, str, str], ...] = (
    (DYNAMIC_PARKING_KEY, "Dynamic parking lots", "--no-dynamic-parking-lots", "CWR_DYNAMIC_PARKING_LOTS"),
    (DYNAMIC_SPORTS_KEY, "Dynamic sports fields", "--no-dynamic-sports-fields", "CWR_DYNAMIC_SPORTS_FIELDS"),
    (DYNAMIC_RUNWAYS_KEY, "Dynamic runways", "--no-dynamic-runways", "CWR_DYNAMIC_RUNWAYS"),
)

_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_INSTALLED = False


def _dynamic_enabled(environment_name: str) -> bool:
    """Return whether one generated-surface family is enabled in this process."""
    return str(os.environ.get(environment_name, "1")).strip().casefold() not in _FALSE_VALUES


def _defaults_with_dynamic_surfaces(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    for key, _label, _flag, _environment in DYNAMIC_SURFACE_OPTIONS:
        result.setdefault(key, True)
    return result


def _append_disable_flags(command: Sequence[str], values: Mapping[str, object]) -> list[str]:
    result = list(command)
    for key, _label, flag, _environment in DYNAMIC_SURFACE_OPTIONS:
        if not bool(values.get(key, True)) and flag not in result:
            result.append(flag)
    return result


def _find_widgets_by_text(root, text: str):
    result = []
    try:
        children = root.winfo_children()
    except Exception:
        children = ()
    for child in children:
        try:
            if str(child.cget("text")) == text:
                result.append(child)
        except Exception:
            pass
        result.extend(_find_widgets_by_text(child, text))
    return result


def _next_grid_row(parent) -> int:
    """Return the first unused grid row in a Tk container."""
    rows: list[int] = []
    try:
        children = parent.winfo_children()
    except Exception:
        children = ()
    for child in children:
        try:
            info = child.grid_info()
            if info and "row" in info:
                rows.append(int(info["row"]))
        except Exception:
            continue
    return max(rows, default=-1) + 1


def _install_gui_controls() -> None:
    from . import gui_entry

    previous_configure = gui_entry._configure_gui

    def configure_gui(gui, base_dir: Path) -> None:
        previous_configure(gui, base_dir)
        if bool(getattr(gui, "_DYNAMIC_SURFACE_CONTROLS_INSTALLED", False)):
            return

        original_defaults = gui.default_gui_values
        original_command = gui.build_milestone9_command

        def default_gui_values() -> dict[str, object]:
            return _defaults_with_dynamic_surfaces(original_defaults())

        def build_milestone9_command(values: dict[str, object], python: str | None = None) -> list[str]:
            return _append_disable_flags(original_command(values, python=python), values)

        gui.default_gui_values = default_gui_values
        gui.build_milestone9_command = build_milestone9_command

        original_class = gui.WorldgenGui

        class DynamicSurfaceControlsWorldgenGui(original_class):
            def _build_appearance_page(self) -> None:
                super()._build_appearance_page()
                anchors = _find_widgets_by_text(self, "Include minor roads")
                if not anchors:
                    return
                parent = anchors[0].master
                first_row = _next_grid_row(parent)
                for index, (key, label, _flag, _environment) in enumerate(
                    DYNAMIC_SURFACE_OPTIONS
                ):
                    variable = self._var(key, True, boolean=True)
                    widget = gui.ttk.Checkbutton(parent, text=label, variable=variable)
                    self._register_advanced_setting(
                        key,
                        widget,
                        normal_style="TCheckbutton",
                        changed_style="AdvancedChanged.TCheckbutton",
                    )
                    # The Common choices container is grid-managed. Mixing pack
                    # into the same parent raises TclError during GUI startup.
                    widget.grid(
                        row=first_row + index // 2,
                        column=index % 2,
                        sticky="w",
                        padx=(0, 32),
                        pady=3,
                    )

        gui.WorldgenGui = DynamicSurfaceControlsWorldgenGui
        gui._DYNAMIC_SURFACE_CONTROLS_INSTALLED = True

    gui_entry._configure_gui = configure_gui


def _install_cli_controls() -> None:
    from . import cli

    original_parser = cli._parser
    original_main = cli.main

    def parser() -> argparse.ArgumentParser:
        result = original_parser()
        subparsers = next(
            (
                action
                for action in result._actions
                if isinstance(action, argparse._SubParsersAction)
            ),
            None,
        )
        milestone9 = None if subparsers is None else subparsers.choices.get("milestone9")
        if milestone9 is not None:
            existing = {
                option
                for action in milestone9._actions
                for option in getattr(action, "option_strings", ())
            }
            help_text = {
                DYNAMIC_PARKING_KEY: "disable generated OSM parking-lot terrain textures",
                DYNAMIC_SPORTS_KEY: "disable generated OSM sports-field terrain textures",
                DYNAMIC_RUNWAYS_KEY: "disable generated OSM runway terrain textures and P3D fallback",
            }
            for key, _label, flag, _environment in DYNAMIC_SURFACE_OPTIONS:
                if flag in existing:
                    continue
                milestone9.add_argument(
                    flag,
                    action="store_false",
                    dest=key,
                    default=True,
                    help=help_text[key],
                )
        return result

    def main(argv: list[str] | None = None) -> int:
        raw = list(sys.argv[1:] if argv is None else argv)
        if "milestone9" not in raw:
            return original_main(argv)

        previous: dict[str, str | None] = {}
        try:
            for _key, _label, flag, environment in DYNAMIC_SURFACE_OPTIONS:
                previous[environment] = os.environ.get(environment)
                os.environ[environment] = "0" if flag in raw else "1"
            return original_main(raw)
        finally:
            for environment, value in previous.items():
                if value is None:
                    os.environ.pop(environment, None)
                else:
                    os.environ[environment] = value

    cli._parser = parser
    cli.main = main


def _install_surface_gates() -> None:
    from . import parking_surface_policy as parking
    from . import runway_surface_policy as runway
    from . import sports_pitch_surface_policy as sports
    from . import surface_pass as surface

    parking_environment = next(value[3] for value in DYNAMIC_SURFACE_OPTIONS if value[0] == DYNAMIC_PARKING_KEY)
    sports_environment = next(value[3] for value in DYNAMIC_SURFACE_OPTIONS if value[0] == DYNAMIC_SPORTS_KEY)
    runway_environment = next(value[3] for value in DYNAMIC_SURFACE_OPTIONS if value[0] == DYNAMIC_RUNWAYS_KEY)

    original_parking_geometries = parking._parking_geometries
    original_pitch_geometries = sports._pitch_geometries
    original_runway_cells = runway.runway_texture_cell_indices
    dynamic_aeroway_mask = surface._aeroway_mask
    original_aeroway_mask = runway._ORIGINAL_AEROWAY_MASK

    def parking_geometries(*args, **kwargs):
        if not _dynamic_enabled(parking_environment):
            return ()
        return original_parking_geometries(*args, **kwargs)

    def pitch_geometries(*args, **kwargs):
        if not _dynamic_enabled(sports_environment):
            return ()
        return original_pitch_geometries(*args, **kwargs)

    def runway_texture_cell_indices(*args, **kwargs):
        if not _dynamic_enabled(runway_environment):
            return ()
        return original_runway_cells(*args, **kwargs)

    def aeroway_mask(dataset, projection, cells):
        if not _dynamic_enabled(runway_environment) and callable(original_aeroway_mask):
            # With dynamic runways off, restore the ordinary surface-pass runway
            # material rather than silently erasing the OSM aeroway altogether.
            return original_aeroway_mask(dataset, projection, cells)
        return dynamic_aeroway_mask(dataset, projection, cells)

    parking._parking_geometries = parking_geometries
    sports._pitch_geometries = pitch_geometries
    runway.runway_texture_cell_indices = runway_texture_cell_indices
    surface._aeroway_mask = aeroway_mask


def install_dynamic_surface_controls() -> None:
    """Install default-on GUI/CLI switches for generated terrain overlays."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_gui_controls()
    _install_cli_controls()
    _install_surface_gates()
    _INSTALLED = True
