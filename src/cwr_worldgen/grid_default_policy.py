# SPDX-License-Identifier: GPL-3.0-or-later
"""User-facing terrain-grid defaults for the GUI and CLI entry points."""
from __future__ import annotations

from functools import wraps
import sys
from typing import Sequence

DEFAULT_TERRAIN_CELL_SIZE_METRES = 50.0


def _with_default_cli_cell_size(argv: Sequence[str]) -> list[str]:
    """Inject the 50 m application default without overriding explicit choices."""
    args = list(argv)
    if not args or "--cell-size" in args:
        return args
    command = args[0]
    if command in {"milestone1", "milestone2", "milestone3", "milestone4", "fetch-sources"}:
        args.extend(("--cell-size", str(DEFAULT_TERRAIN_CELL_SIZE_METRES)))
    elif command == "regrid-sources" and "--cells" not in args:
        # Preserve the useful --cells-only inference mode. When neither grid
        # dimension is supplied, however, the application default is now 50 m.
        args.extend(("--cell-size", str(DEFAULT_TERRAIN_CELL_SIZE_METRES)))
    return args


def _install_cli_default() -> None:
    from . import cli as cli_module

    original = cli_module.main
    if getattr(original, "_cwr_50m_default", False):
        return

    @wraps(original)
    def main_with_50m_default(argv: list[str] | None = None) -> int:
        raw = sys.argv[1:] if argv is None else argv
        return original(_with_default_cli_cell_size(raw))

    main_with_50m_default._cwr_50m_default = True
    cli_module.main = main_with_50m_default


def _install_gui_default() -> None:
    from . import gui_entry

    original_entry = gui_entry.main
    if getattr(original_entry, "_cwr_50m_default", False):
        return

    @wraps(original_entry)
    def main_with_50m_default(argv: list[str] | None = None) -> int:
        from . import gui

        gui.DEFAULT_GUI_CELL_SIZE_METRES = DEFAULT_TERRAIN_CELL_SIZE_METRES
        original_defaults = gui.default_gui_values
        if not getattr(original_defaults, "_cwr_50m_default", False):
            @wraps(original_defaults)
            def defaults_with_50m_grid() -> dict[str, object]:
                values = original_defaults()
                cells = int(values.get("fetch_cells", gui.DEFAULT_GUI_TERRAIN_CELLS))
                lat = float(values.get("center_lat", 59.45))
                lon = float(values.get("center_lon", 17.0))
                south, west, north, east = gui.square_bbox(
                    lat,
                    lon,
                    cells * DEFAULT_TERRAIN_CELL_SIZE_METRES,
                )
                values.update({
                    "fetch_cell_size": f"{DEFAULT_TERRAIN_CELL_SIZE_METRES:g}",
                    "south": f"{south:.7f}",
                    "west": f"{west:.7f}",
                    "north": f"{north:.7f}",
                    "east": f"{east:.7f}",
                })
                return values

            defaults_with_50m_grid._cwr_50m_default = True
            gui.default_gui_values = defaults_with_50m_grid
        return original_entry(argv)

    main_with_50m_default._cwr_50m_default = True
    gui_entry.main = main_with_50m_default


def install_default_grid_policy() -> None:
    """Make 50 m the GUI/CLI default while preserving explicit and API values."""
    _install_cli_default()
    _install_gui_default()
