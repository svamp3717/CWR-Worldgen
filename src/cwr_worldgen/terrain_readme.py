# SPDX-License-Identifier: GPL-3.0-or-later
"""Terrain ReadMe generation for final CWR-Worldgen runtime folders."""
from __future__ import annotations

from datetime import datetime
from functools import wraps
import json
from pathlib import Path
from typing import Any

from ._version import __version__

_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')


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


def write_terrain_readme(result: Any, spec: Any) -> Path:
    """Write the terrain ReadMe beside the generated PBO."""
    readme_path = result.pbo_path.parent / terrain_readme_filename(spec.display_name)
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


def _deploy_readme(result: Any, spec: Any, readme_path: Path) -> Path | None:
    target = getattr(spec, "deploy_mod_dir", None)
    if target is None:
        return None

    # Reuse the same case-insensitive Addons-folder and atomic-copy rules as the
    # normal Milestone 9 deployment.
    from . import milestone9 as milestone9_module

    target_root = milestone9_module._normalise_mod_root(Path(target))
    addons_dir = milestone9_module._existing_mod_child(target_root, "Addons")
    addons_dir.mkdir(parents=True, exist_ok=True)
    destination = addons_dir / readme_path.name
    milestone9_module._atomic_copy_file(readme_path, destination)

    report_path = result.output_dir / "deployment-report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    if isinstance(report, dict):
        files = report.get("files")
        if not isinstance(files, list):
            files = []
            report["files"] = files
        destination_text = str(destination)
        files[:] = [
            item
            for item in files
            if not isinstance(item, dict) or str(item.get("destination", "")) != destination_text
        ]
        files.append(
            {
                "kind": "addon",
                "source": str(readme_path),
                "destination": destination_text,
                "sha256": milestone9_module._sha256(destination),
            }
        )
        report["readme"] = destination_text
        report["file_count"] = len(files)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return destination


def install_milestone9_terrain_readme() -> None:
    """Wrap the final Milestone 9 build so every terrain gets its Addons ReadMe."""
    from . import milestone9 as milestone9_module

    original = milestone9_module.build_milestone9
    if bool(getattr(original, "_cwr_terrain_readme", False)):
        return

    @wraps(original)
    def build_with_terrain_readme(output_dir: Path, spec: Any, *, clean: bool = True):
        result = original(output_dir, spec, clean=clean)
        readme_path = write_terrain_readme(result, spec)
        _deploy_readme(result, spec, readme_path)
        return result

    build_with_terrain_readme._cwr_terrain_readme = True  # type: ignore[attr-defined]
    milestone9_module.build_milestone9 = build_with_terrain_readme
