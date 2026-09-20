# SPDX-License-Identifier: GPL-3.0-or-later
"""Terrain ReadMe generation for final CWR-Worldgen runtime folders."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
from functools import wraps
import json
from pathlib import Path
import sys
from typing import Any

from ._version import __version__

_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')
_ACTIVE_TERRAIN_SPEC: ContextVar[Any | None] = ContextVar(
    "cwr_worldgen_active_terrain_readme_spec",
    default=None,
)


def terrain_readme_filename(display_name: str) -> str:
    """Return a Windows-safe human-readable ReadMe filename."""
    cleaned = "".join(
        "_" if character in _INVALID_FILENAME_CHARS or ord(character) < 32 else character
        for character in str(display_name).strip()
    ).strip(" .")
    return f"{cleaned or 'Terrain'} ReadMe.txt"


def _format_coordinate(value: object) -> str:
    try:
        return f"{float(value):.7f}"
    except (TypeError, ValueError):
        return "unknown"


def _selection_details(source_manifest_path: Path) -> dict[str, Any]:
    try:
        document = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    selection = document.get("selection")
    return dict(selection) if isinstance(selection, dict) else {}


def terrain_readme_text(
    *,
    display_name: str,
    pbo_name: str,
    source_manifest_path: Path,
    cells: int,
    cell_size_metres: float,
    created_at: datetime | None = None,
) -> str:
    """Build the user-facing terrain reproduction ReadMe."""
    selection = _selection_details(source_manifest_path)
    bbox = selection.get("bbox_south_west_north_east")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        bbox = ()
    center = selection.get("center_latitude_longitude")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        if bbox:
            center = (
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                (float(bbox[1]) + float(bbox[3])) / 2.0,
            )
        else:
            center = ()

    selection_kind = str(selection.get("kind", "unknown")).strip() or "unknown"
    selection_labels = {
        "bbox": "Bounding box",
        "center": "Center coordinates",
        "opentopomap-url": "OpenTopoMap URL",
    }
    selection_method = selection_labels.get(selection_kind, selection_kind)

    selected_cells = selection.get("cells", cells)
    selected_cell_size = selection.get("cell_size_metres", cell_size_metres)
    try:
        selected_cells = int(selected_cells)
    except (TypeError, ValueError):
        selected_cells = int(cells)
    try:
        selected_cell_size = float(selected_cell_size)
    except (TypeError, ValueError):
        selected_cell_size = float(cell_size_metres)

    timestamp = (created_at or datetime.now().astimezone()).strftime("%Y%m%d%H%M")
    lines = [
        str(display_name),
        f"PBO: {pbo_name}",
        f"Version: {timestamp}",
        "",
        "Terrain Informations",
        f"Selection method: {selection_method}",
    ]
    if center:
        lines.append(
            "Center coordinates (Latitude, Longitude): "
            f"{_format_coordinate(center[0])}, {_format_coordinate(center[1])}"
        )
    if bbox:
        lines.append(
            "Coordinates (South, West, North, East): "
            + ", ".join(_format_coordinate(value) for value in bbox)
        )
    source_value = selection.get("value")
    if isinstance(source_value, str) and source_value.strip():
        lines.append(f"Selection source: {source_value.strip()}")
    lines.extend(
        [
            f"Terrain cells: {selected_cells} x {selected_cells}",
            f"Cell size: {selected_cell_size:g} m",
            f"World size: {selected_cells * selected_cell_size:g} m x {selected_cells * selected_cell_size:g} m",
            "",
            f"This Terrain is created by CWR-Worldgen {__version__}",
            "",
        ]
    )
    return "\n".join(lines)


def terrain_readme_path(result: Any, spec: Any) -> Path:
    """Return the diagnostic ReadMe path outside the deployable runtime."""
    return result.output_dir / terrain_readme_filename(spec.display_name)


def write_terrain_readme(result: Any, spec: Any) -> Path:
    """Write the terrain ReadMe as build metadata, not as a runtime addon file."""
    readme_path = terrain_readme_path(result, spec)
    readme_path.write_text(
        terrain_readme_text(
            display_name=spec.display_name,
            pbo_name=result.pbo_path.name,
            source_manifest_path=Path(spec.source_dir) / "source.json",
            cells=int(getattr(spec, "cells", 256)),
            cell_size_metres=float(getattr(spec, "cell_size", 25.0)),
        ),
        encoding="utf-8",
        newline="\n",
    )
    return readme_path


def _sync_cli_build_binding(build_callable: Any) -> None:
    """Point an already-imported CLI at the final Milestone 9 build callable."""
    cli_module = sys.modules.get(f"{__package__}.cli")
    if cli_module is not None:
        cli_module.build_milestone9 = build_callable


def install_milestone9_terrain_readme() -> None:
    """Keep the terrain ReadMe as local build metadata for Milestone 9."""
    from . import milestone9 as milestone9_module

    original_build = milestone9_module.build_milestone9
    if bool(getattr(original_build, "_cwr_terrain_readme", False)):
        _sync_cli_build_binding(original_build)
        return

    original_deploy = milestone9_module._deploy_runtime_to_existing_mod

    @wraps(original_deploy)
    def deploy_with_terrain_readme(result: Any, target_root: Path):
        # Deployment is intentionally PBO-only. Keep the reproduction note in
        # the build output instead of placing another file in the mod's Addons.
        spec = _ACTIVE_TERRAIN_SPEC.get()
        if spec is not None:
            write_terrain_readme(result, spec)
        return original_deploy(result, target_root)

    deploy_with_terrain_readme._cwr_terrain_readme = True  # type: ignore[attr-defined]
    milestone9_module._deploy_runtime_to_existing_mod = deploy_with_terrain_readme

    @wraps(original_build)
    def build_with_terrain_readme(output_dir: Path, spec: Any, *, clean: bool = True):
        token = _ACTIVE_TERRAIN_SPEC.set(spec)
        try:
            result = original_build(output_dir, spec, clean=clean)
            # The ReadMe remains local build metadata regardless of deployment.
            readme_path = terrain_readme_path(result, spec)
            if not readme_path.is_file():
                write_terrain_readme(result, spec)
            return result
        finally:
            _ACTIVE_TERRAIN_SPEC.reset(token)

    build_with_terrain_readme._cwr_terrain_readme = True  # type: ignore[attr-defined]
    milestone9_module.build_milestone9 = build_with_terrain_readme

    # cli.py imports build_milestone9 by value. Always synchronize that binding
    # with the final wrapped callable, regardless of which earlier policy wrapper
    # happened to be present when cli.py was first imported.
    _sync_cli_build_binding(build_with_terrain_readme)
