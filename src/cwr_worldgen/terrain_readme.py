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


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(document) if isinstance(document, dict) else {}


def _humanize_identifier(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    return text.replace("_", " ").replace("-", " ").title()


def _building_preset_label(value: object) -> str:
    identifier = str(value or "auto").strip() or "auto"
    if identifier.casefold() == "auto":
        return "Automatic (area / country)"

    try:
        from .stock_building_extensions import STOCK_BUILDING_OPTIONS, stock_building_preset_ids
        stock_labels = dict(STOCK_BUILDING_OPTIONS)
        selected_stock = stock_building_preset_ids(identifier)
        if selected_stock:
            return " + ".join(stock_labels.get(item, _humanize_identifier(item)) for item in selected_stock)
    except (ImportError, RuntimeError, ValueError):
        pass

    try:
        from .building_country_policy import building_country_options
        country_labels = dict(building_country_options())
        if identifier in country_labels:
            return country_labels[identifier]
    except (ImportError, RuntimeError):
        pass

    try:
        from .house_style_catalogue import house_style_preset_profile
        profile = house_style_preset_profile(identifier)
        if profile is not None:
            return str(profile.display_name)
    except (ImportError, RuntimeError, ValueError):
        pass

    return _humanize_identifier(identifier)


def _appearance_label(ground_profile: object, forest_profile: object) -> str:
    ground = str(ground_profile or "generated").strip().casefold() or "generated"
    forest = str(forest_profile or "everon").strip().casefold() or "everon"
    exact = {
        ("nogova", "everon"): "Nogova textures + Everon trees",
        ("nogova", "malden"): "Nogova textures + Malden vegetation",
        ("everon", "everon"): "Everon classic",
        ("malden", "malden"): "Malden classic",
    }
    if (ground, forest) in exact:
        return exact[(ground, forest)]
    return f"{_humanize_identifier(ground)} terrain + {_humanize_identifier(forest)} vegetation"


def _building_readme_details(result: Any, spec: Any) -> dict[str, object]:
    catalogue_path = getattr(result, "building_catalogue_path", None)
    if catalogue_path is None:
        candidate = Path(getattr(result, "output_dir", ".")) / "building-asset-catalogue.json"
        catalogue_path = candidate if candidate.is_file() else None
    catalogue = _read_json_object(Path(catalogue_path) if catalogue_path is not None else None)

    preset_identifier = str(
        catalogue.get("house_style_preset")
        or catalogue.get("mode")
        or getattr(spec, "house_style_preset", "auto")
        or "auto"
    )
    resolved_identifier = str(
        catalogue.get("house_style_region")
        or catalogue.get("detected_house_style_region")
        or catalogue.get("region")
        or ""
    ).strip()

    style_identifiers: set[str] = set()
    class_names: set[str] = set()
    for row in catalogue.get("request_mapping", ()):
        if not isinstance(row, dict):
            continue
        selected = row.get("selected")
        if not isinstance(selected, dict):
            continue
        value = str(selected.get("regional_style", "")).strip()
        if value:
            style_identifiers.add(value)
        value = str(selected.get("building_class", "")).strip()
        if value:
            class_names.add(value)

    if not style_identifiers:
        for row in catalogue.get("models", ()):
            if not isinstance(row, dict):
                continue
            value = str(row.get("source_set", "")).strip()
            if value:
                style_identifiers.add(value)

    return {
        "preset_identifier": preset_identifier,
        "preset_label": _building_preset_label(preset_identifier),
        "resolved_identifier": resolved_identifier,
        "resolved_label": _building_preset_label(resolved_identifier) if resolved_identifier else "",
        "styles": tuple(sorted(style_identifiers)),
        "classes": tuple(sorted(class_names)),
    }


def terrain_readme_text(
    *,
    display_name: str,
    pbo_name: str,
    source_manifest_path: Path,
    cells: int,
    cell_size_metres: float,
    building_preset: str = "Unknown",
    building_preset_identifier: str = "",
    resolved_building_style: str = "",
    resolved_building_style_identifier: str = "",
    building_styles: tuple[str, ...] = (),
    building_classes: tuple[str, ...] = (),
    terrain_style: str = "Unknown",
    forest_style: str = "Unknown",
    appearance_preset: str = "Unknown",
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
        "Build Presets",
        (
            f"Building preset: {building_preset} [{building_preset_identifier}]"
            if building_preset_identifier
            else f"Building preset: {building_preset}"
        ),
    ]
    if resolved_building_style:
        lines.append(
            (
                f"Resolved building style: {resolved_building_style} [{resolved_building_style_identifier}]"
                if resolved_building_style_identifier
                else f"Resolved building style: {resolved_building_style}"
            )
        )
    if building_styles:
        lines.append(
            "Building styles used: "
            + ", ".join(_humanize_identifier(value) for value in building_styles)
        )
    if building_classes:
        lines.append(
            "Building classes used: "
            + ", ".join(_humanize_identifier(value) for value in building_classes)
        )
    lines.extend(
        [
            f"Appearance preset: {appearance_preset}",
            f"Terrain style: {_humanize_identifier(terrain_style)}",
            f"Vegetation / forest preset: {_humanize_identifier(forest_style)}",
            "",
            "Terrain Informations",
            f"Selection method: {selection_method}",
        ]
    )
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
    """Return the ReadMe path beside the generated terrain PBO."""
    return result.pbo_path.parent / terrain_readme_filename(spec.display_name)


def write_terrain_readme(result: Any, spec: Any) -> Path:
    """Write the terrain ReadMe beside the generated terrain PBO."""
    readme_path = terrain_readme_path(result, spec)
    building = _building_readme_details(result, spec)
    ground_profile = str(getattr(spec, "ground_texture_profile", "generated"))
    forest_profile = str(getattr(spec, "forest_profile", "everon"))
    readme_path.write_text(
        terrain_readme_text(
            display_name=spec.display_name,
            pbo_name=result.pbo_path.name,
            source_manifest_path=Path(spec.source_dir) / "source.json",
            cells=int(getattr(spec, "cells", 256)),
            cell_size_metres=float(getattr(spec, "cell_size", 25.0)),
            building_preset=str(building["preset_label"]),
            building_preset_identifier=str(building["preset_identifier"]),
            resolved_building_style=str(building["resolved_label"]),
            resolved_building_style_identifier=str(building["resolved_identifier"]),
            building_styles=tuple(building["styles"]),
            building_classes=tuple(building["classes"]),
            terrain_style=ground_profile,
            forest_style=forest_profile,
            appearance_preset=_appearance_label(ground_profile, forest_profile),
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
    """Keep the terrain ReadMe beside the PBO and deploy it with the PBO."""
    from . import milestone9 as milestone9_module

    original_build = milestone9_module.build_milestone9
    if bool(getattr(original_build, "_cwr_terrain_readme", False)):
        _sync_cli_build_binding(original_build)
        return

    original_deploy = milestone9_module._deploy_runtime_to_existing_mod

    @wraps(original_deploy)
    def deploy_with_terrain_readme(result: Any, target_root: Path):
        # Create the human-readable terrain note before deployment so the normal
        # deploy pass can copy it beside the PBO together with the required menu intro.
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
            # Builds without deployment still keep the ReadMe beside the PBO.
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
