# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import time
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from typing import Iterable, Mapping

from ._version import __version__
from .location_example import parse_opentopomap_link, square_bbox
from .map_picker import AreaSelectionPlan, OsmAreaPicker, zoom_for_bbox
from .model import DEFAULT_MAX_BUILDINGS, DEFAULT_MAX_FOREST_OBJECTS, DEFAULT_MAX_ROAD_OBJECTS, validate_world_identity

APP_TITLE = f"CWR Worldgen {__version__}"
PROFILE_VERSION = 2
GUI_STATE_VERSION = 1
DEFAULT_GUI_TERRAIN_CELLS = 256
DEFAULT_GUI_CELL_SIZE_METRES = 25.0
TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")
FROZEN_CLI_MARKER = "--cwr-cli"

RECOMMENDED_APPEARANCE_PRESET = "Nogova textures + Everon trees (recommended)"
RESISTANCE_APPEARANCE_PRESET = "Nogova Resistance leaf forests"
LEGACY_RESISTANCE_APPEARANCE_PRESET = "Nogova Resistance forests"
PINE_NOGOVA_APPEARANCE_PRESET = "Nogova Resistance pine forests"
LEGACY_NOGOVA_APPEARANCE_PRESET = "Nogova (recommended)"
APPEARANCE_PRESETS = (
    RECOMMENDED_APPEARANCE_PRESET,
    RESISTANCE_APPEARANCE_PRESET,
    PINE_NOGOVA_APPEARANCE_PRESET,
    "Malden classic",
    "Everon classic",
    "Desert ground textures",
    "Generated ground textures",
    "Custom",
)
NOGOVA_FOREST_BLOCK_MODEL = r"o\tree\les_nw_ctver_pruhozi_T1.p3d"
NOGOVA_FOREST_STEEP_MODEL = r"o\tree\les_nw_trojuhelnik.p3d"
NOGOVA_PINE_FOREST_BLOCK_MODEL = r"o\tree\les_nw_jehl_ctver_pruhozi.p3d"
NOGOVA_PINE_FOREST_STEEP_MODEL = r"o\tree\les_nw_jehl_trojuhelnik.p3d"
NOGOVA_LEAF_SINGLE_TREE_MODEL = r"o\tree\Javor01.p3d"
NOGOVA_PINE_SINGLE_TREE_MODEL = r"o\tree\smrk_maly.p3d"
NOGOVA_SINGLE_TREE_MODEL = NOGOVA_PINE_SINGLE_TREE_MODEL  # compatibility alias
EVERON_SINGLE_TREE_MODEL = r"data3d\str smrk_medium.p3d"


def cli_command_prefix(python: str | None = None) -> list[str]:
    """Return a CLI launcher that works both from source and PyInstaller.

    ``sys.executable`` is the Python interpreter in a normal install, but in a
    frozen GUI it is the GUI executable itself. Launching it with ``-m`` simply
    starts another copy of the GUI. Frozen builds therefore re-enter this
    executable through a private dispatch marker handled by :func:`main`.
    """
    if python is not None:
        return [python, "-m", "cwr_worldgen"]
    if bool(getattr(sys, "frozen", False)):
        return [sys.executable, FROZEN_CLI_MARKER]
    return [sys.executable, "-m", "cwr_worldgen"]


def application_base_dir() -> Path:
    """Return the stable directory used for GUI-relative files and child jobs.

    Source installs intentionally retain the caller's working directory. A
    PyInstaller executable is different: Windows shortcuts, shells and launchers
    are free to give it an unrelated current directory, while users reasonably
    expect ``build`` and ``source-data`` to appear beside the EXE.
    """
    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def resolve_gui_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve a user-entered relative filesystem path against the app base."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir or application_base_dir()) / path
    return path.resolve()


def default_gui_path(relative: str | Path) -> str:
    """Return a default path, EXE-relative when running as a frozen GUI."""
    path = Path(relative)
    if bool(getattr(sys, "frozen", False)):
        return str(resolve_gui_path(path))
    return str(path)


SOURCE_PREVIEW_MAX_WIDTH = 720
SOURCE_PREVIEW_MAX_HEIGHT = 380
_SOURCE_PREVIEW_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def existing_source_preview_path(source_dir: Path | str) -> Path | None:
    """Return the best stored reference preview for an existing source bundle.

    Prefer the exact image recorded by ``source.json``. Older or hand-maintained
    bundles may not have that manifest field, so fall back to common image names
    directly inside ``reference/``. Tile-cache images are deliberately ignored.
    """

    root = Path(source_dir).expanduser()
    manifest_path = root / "source.json"
    if manifest_path.is_file():
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            document = {}
        reference = document.get("reference_map") if isinstance(document, Mapping) else None
        relative = reference.get("path") if isinstance(reference, Mapping) else None
        if isinstance(relative, str) and relative.strip():
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                candidate = Path()
            if (
                candidate
                and candidate.is_file()
                and candidate.suffix.casefold() in _SOURCE_PREVIEW_SUFFIXES
            ):
                return candidate

    reference_dir = root / "reference"
    if not reference_dir.is_dir():
        return None
    by_name = ("opentopomap.png", "preview.png", "reference.png", "map.png")
    for name in by_name:
        candidate = reference_dir / name
        if candidate.is_file():
            return candidate
    images = sorted(
        candidate
        for candidate in reference_dir.iterdir()
        if candidate.is_file() and candidate.suffix.casefold() in _SOURCE_PREVIEW_SUFFIXES
    )
    return images[0] if images else None


@dataclass(frozen=True, slots=True)
class WizardStep:
    title: str
    subtitle: str


WIZARD_STEPS = (
    WizardStep("Choose map area", "Select a new OSM area or use an existing frozen source bundle."),
    WizardStep("Name the world", "Set the game-visible name, internal identifier and output folder."),
    WizardStep("Choose appearance", "Pick a preset or reveal advanced world-generation tuning."),
    WizardStep("Build", "Fetch source data if needed, then build the world and watch progress."),
)

APPEARANCE_STEP_INDEX = 2
PROGRESS_STEP_INDEX = 3


def quote_command(parts: Iterable[str]) -> str:
    """Return a copyable command line, using Windows quoting on Windows."""
    values = list(parts)
    if os.name == "nt":
        return subprocess.list2cmdline(values)
    return shlex.join(values)


def command_option_value(command: Iterable[str], option: str) -> str | None:
    """Return the value following *option* in an argv sequence."""
    values = list(command)
    try:
        index = values.index(option)
    except ValueError:
        return None
    if index + 1 >= len(values):
        return None
    return values[index + 1]


def terminate_process_tree(process: subprocess.Popen[str], *, grace_seconds: float = 1.5) -> None:
    """Stop a GUI-launched command and any child tools it started.

    World generation can launch helper executables and the old Stop button only
    signalled the immediate Python process. On Windows, ``taskkill /T`` is the
    reliable way to tear down that complete process tree. On POSIX we launch the
    command in a new session and signal its process group.
    """

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        return

    try:
        group_id = os.getpgid(process.pid)
    except (OSError, ProcessLookupError):
        group_id = None
    try:
        if group_id is not None:
            os.killpg(group_id, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        return
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            if group_id is not None:
                os.killpg(group_id, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass


def gui_state_path() -> Path:
    """Return the per-user GUI state path without depending on Tk."""
    override = os.environ.get("CWR_WORLDGEN_GUI_STATE")
    if override:
        return Path(override)
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "cwr-worldgen" / "gui-state.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "cwr-worldgen" / "gui-state.json"
    return Path.home() / ".config" / "cwr-worldgen" / "gui-state.json"


def load_gui_state(path: Path) -> dict[str, object]:
    """Load non-profile GUI state; malformed or stale files are ignored."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(document, dict) or document.get("state_version") != GUI_STATE_VERSION:
        return {}
    return document


def save_gui_state(path: Path, values: Mapping[str, object]) -> None:
    """Atomically save the small per-user GUI state document."""
    document = {"state_version": GUI_STATE_VERSION, **values}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def update_gui_state(path: Path, values: Mapping[str, object]) -> None:
    """Merge selected values into the per-user GUI state document."""
    existing = load_gui_state(path)
    existing.pop("state_version", None)
    save_gui_state(path, {**existing, **values})


def increment_trailing_number(value: str) -> str:
    """Increase a final decimal suffix, or append one when the name has none."""
    text = value.strip()
    if not text:
        return text
    match = TRAILING_NUMBER.fullmatch(text)
    if match is None:
        return f"{text}2"
    prefix, number = match.groups()
    return f"{prefix}{int(number) + 1:0{len(number)}d}"


def defaults_with_recent_source(
    defaults: dict[str, object], state: Mapping[str, object]
) -> dict[str, object]:
    """Prefer recent source/deployment paths and remembered GUI mapping settings."""
    result = dict(defaults)
    source_text = str(state.get("last_downloaded_source", "")).strip()
    source_path = Path(source_text).expanduser() if source_text else None
    if source_path is not None and (source_path / "source.json").is_file():
        source_text = str(source_path.resolve())
        result["source_mode"] = "existing"
        result["source_dir"] = source_text
        result["fetch_source_dir"] = source_text
    deploy_text = str(state.get("last_deploy_mod_dir", "")).strip()
    deploy_path = Path(deploy_text).expanduser() if deploy_text else None
    if deploy_path is not None and deploy_path.is_dir():
        result["deploy_mod_dir"] = str(deploy_path.resolve())
        result["deploy_to_mod_folder"] = bool(state.get("deploy_to_mod_folder", True))
    last_name = str(state.get("last_world_name", "")).strip()
    if last_name:
        next_name = increment_trailing_number(last_name)
        result["name"] = next_name
        output_text = str(state.get("last_world_output", "")).strip()
        if output_text:
            result["output"] = increment_trailing_number(output_text)
        else:
            output_slug = next_name[4:] if next_name.casefold().startswith("cwr_") else next_name
            default_output = Path(str(result.get("output", default_gui_path(Path("build") / "my_world"))))
            result["output"] = str(default_output.parent / output_slug)
    last_display_name = str(state.get("last_world_display_name", "")).strip()
    if last_display_name:
        result["display_name"] = increment_trailing_number(last_display_name)
    for key in (
        "osm_asset_mapping_enabled",
        "osm_asset_mapping_inherit_defaults",
        "osm_asset_mapping_rules",
        "osm_asset_mapping_global_models",
        "osm_asset_mapping_global_textures",
    ):
        if key in state:
            result[key] = state[key]
    return result


def _mapping_asset_lines(value: object) -> list[str]:
    """Return unique non-empty asset paths from a newline/comma separated GUI field."""
    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[\r\n,]+", str(value or ""))
    result: list[str] = []
    for item in raw:
        path = item.strip().replace("/", "\\").lstrip("\\")
        if path and path not in result:
            result.append(path)
    return result


def parse_gui_osm_asset_rules(value: object) -> list[dict[str, object]]:
    """Decode the GUI's stored custom rule list with useful validation errors."""
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        document = value
    else:
        try:
            document = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"The GUI OSM mapping rules are invalid JSON: {exc}") from exc
    if not isinstance(document, list) or not all(isinstance(item, dict) for item in document):
        raise ValueError("The GUI OSM mapping rules must be a list of rule objects.")
    return [dict(item) for item in document]


def build_gui_osm_asset_mapping_document(values: Mapping[str, object]) -> dict[str, object]:
    """Build a schema-1 mapping document from visual-editor values."""
    return {
        "schema": 1,
        "inherit_defaults": bool(values.get("osm_asset_mapping_inherit_defaults", True)),
        "global": {
            "models": _mapping_asset_lines(values.get("osm_asset_mapping_global_models", "")),
            "textures": _mapping_asset_lines(values.get("osm_asset_mapping_global_textures", "")),
        },
        "rules": parse_gui_osm_asset_rules(values.get("osm_asset_mapping_rules", "[]")),
    }


def write_gui_osm_asset_mapping(path: Path, values: Mapping[str, object]) -> Path:
    """Write and validate the mapping generated by the visual editor."""
    document = build_gui_osm_asset_mapping_document(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Use the production parser, not a friendlier but divergent GUI imitation.
    from .asset_mapping import default_osm_asset_mapping, load_osm_asset_mapping
    from .milestone9 import Milestone9Spec
    defaults = default_osm_asset_mapping(Milestone9Spec(source_dir=Path(".")), 9)
    try:
        load_osm_asset_mapping(temporary, defaults)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def build_milestone9_command(values: dict[str, object], python: str | None = None) -> list[str]:
    """Build the CLI argv used by the GUI. Kept GUI-free for tests."""
    required = ("source_dir", "output", "name", "display_name")
    missing = [key for key in required if not str(values.get(key, "")).strip()]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    validate_world_identity(
        name=str(values.get("name", "")),
        display_name=str(values.get("display_name", "")),
        profile=str(values.get("profile", "cwr-ce")),
    )

    command = cli_command_prefix(python) + ["milestone9"]
    command.extend(("--surface-ground-mode", "milestone9"))
    pairs = (
        ("--source-dir", "source_dir"),
        ("--output", "output"),
        ("--name", "name"),
        ("--display-name", "display_name"),
        ("--profile", "profile"),
        ("--ground-textures", "ground_textures"),
        ("--forest-profile", "forest_profile"),
        ("--forest-single-tree-model", "forest_single_tree_model"),
        ("--max-forest-single-tree-objects", "max_forest_single_tree_objects"),
        ("--forest-single-tree-spacing", "forest_single_tree_spacing"),
        ("--forest-single-tree-max-float", "forest_single_tree_max_float"),
        ("--forest-undergrowth-max-objects", "forest_undergrowth_max_objects"),
        ("--forest-undergrowth-spacing", "forest_undergrowth_spacing"),
        ("--max-steep-hill-bush-objects", "max_steep_hill_bush_objects"),
        ("--steep-hill-bush-spacing", "steep_hill_bush_spacing"),
        ("--steep-hill-bush-min-slope", "steep_hill_bush_min_slope"),
        ("--max-wetland-reed-objects", "max_wetland_reed_objects"),
        ("--wetland-reed-spacing", "wetland_reed_spacing"),
        ("--max-meadow-grass-objects", "max_meadow_grass_objects"),
        ("--meadow-grass-spacing", "meadow_grass_spacing"),
        ("--cache-dir", "cache_dir"),
        ("--max-road-objects", "max_road_objects"),
        ("--max-buildings", "max_buildings"),
        ("--building-ground-clearance", "building_ground_clearance"),
        ("--church-ground-clearance", "church_ground_clearance"),
        ("--building-foundation-depth", "building_foundation_depth"),
        ("--bus-stop-ground-clearance", "bus_stop_ground_clearance"),
        ("--max-grave-objects", "max_grave_objects"),
        ("--grave-spacing", "grave_spacing"),
        ("--grave-ground-clearance", "grave_ground_clearance"),
        ("--grave-road-clearance", "grave_road_clearance"),
        ("--grave-building-clearance", "grave_building_clearance"),
        ("--bridge-module-length", "bridge_module_length"),
        ("--bridge-deck-clearance", "bridge_deck_clearance"),
        ("--bridge-water-clearance", "bridge_water_clearance"),
        ("--max-residential-infill-buildings", "max_residential_infill_buildings"),
        ("--residential-infill-spacing", "residential_infill_spacing"),
        ("--residential-infill-min-area", "residential_infill_min_area"),
        ("--max-forest-objects", "max_forest_objects"),
        ("--forest-max-block-relief", "forest_max_block_relief"),
        ("--forest-steep-max-relief", "forest_steep_max_relief"),
        ("--forest-polygon-sink-fraction", "forest_polygon_sink_fraction"),
        ("--forest-severe-hill-relief", "forest_severe_hill_relief"),
        ("--forest-severe-hill-trees-per-block", "forest_severe_hill_trees_per_block"),
        ("--forest-cluster-max-relief", "forest_cluster_max_relief"),
        ("--rocky-forest-rocks-per-patch", "rocky_forest_rocks_per_patch"),
        ("--rocky-forest-spread", "rocky_forest_spread"),
        ("--lake-shore-smoothing-cells", "lake_shore_smoothing_cells"),
        ("--lake-shore-max-slope", "lake_shore_max_slope"),
        ("--solver-iterations", "solver_iterations"),
        ("--overview-size", "overview_size"),
        ("--pbo-backend", "pbo_backend"),
        ("--poseidon-tools", "poseidon_tools"),
        ("--osm-asset-map", "osm_asset_map"),
    )
    for option, key in pairs:
        value = values.get(key)
        if value not in (None, ""):
            command.extend((option, str(value)))

    for root in values.get("asset_roots", []) or []:
        root_text = str(root).strip()
        if root_text:
            command.extend(("--asset-root", root_text))

    flags = {
        "strict_assets": "--strict-assets",
        "keep_output": "--keep-output",
        "include_minor_roads": "--include-minor-roads",
        "cache_refresh": "--cache-refresh",
        "no_cache": "--no-cache",
        "normalization_refresh": "--normalization-refresh",
        "verify_regeneration": "--verify-regeneration",
        "procedural_building_interiors": "--procedural-building-interiors",
        "high_quality_building_textures": "--high-quality-building-textures",
    }
    for key, option in flags.items():
        if bool(values.get(key, False)):
            command.append(option)
    bus_stop_signs = values.get("bus_stop_signs", values.get("bus_stops"))
    if bus_stop_signs is True:
        command.append("--bus-stop-signs")
    elif bus_stop_signs is False:
        command.append("--no-bus-stop-signs")

    if bool(values.get("procedural_bridges", False)):
        command.append("--procedural-bridges")

    negative_flags = {
        "include_minor_roads": "--no-minor-roads",
        "forest_clusters": "--no-forest-clusters",
        "forest_severe_hill_fallback": "--no-severe-hill-forest-fallback",
        "forest_undergrowth": "--no-forest-undergrowth",
        "forest_borders": "--no-forest-borders",
        "forest_single_trees": "--no-forest-single-trees",
        "ditch_grass": "--no-ditch-grass",
        "barriers": "--no-barriers",
        "bridges": "--no-bridges",
        "residential_infill": "--no-residential-infill",
        "overture_buildings": "--no-overture-buildings",
        "rural_vegetation": "--no-rural-vegetation",
        "meadow_grass": "--no-meadow-grass",
        "rocky_forest_fallback": "--no-rocky-forest-fallback",
        "steep_hill_bushes": "--no-steep-hill-bushes",
        "wetland_reeds": "--no-wetland-reeds",
        "cemeteries": "--no-cemeteries",
    }
    for key, option in negative_flags.items():
        if values.get(key, True) is False:
            command.append(option)

    if bool(values.get("deploy_to_mod_folder", False)):
        deploy_mod_dir = str(values.get("deploy_mod_dir", "")).strip()
        if not deploy_mod_dir:
            raise ValueError("Choose an existing game mod folder for deployment.")
        command.extend(("--deploy-mod-dir", deploy_mod_dir))

    advanced = str(values.get("advanced_args", "")).strip()
    if advanced:
        command.extend(shlex.split(advanced, posix=os.name != "nt"))

    preset = str(values.get("appearance_preset", "")).strip()
    if preset == PINE_NOGOVA_APPEARANCE_PRESET:
        # Clone of the Resistance/Nogova preset using its separate conifer
        # (jehl = needle-tree) polygon forest pieces. The GUI fields already
        # carry the preset's single-tree and sink settings, so append only the
        # polygon models here to avoid duplicate command-line options.
        command.extend((
            "--forest-block-model", NOGOVA_PINE_FOREST_BLOCK_MODEL,
            "--forest-steep-model", NOGOVA_PINE_FOREST_STEEP_MODEL,
        ))
    elif preset in {RESISTANCE_APPEARANCE_PRESET, LEGACY_RESISTANCE_APPEARANCE_PRESET, LEGACY_NOGOVA_APPEARANCE_PRESET}:
        # Append only the defining polygon models after Advanced arguments.
        # Single-tree and sink values are emitted once from the normal GUI fields.
        command.extend((
            "--forest-block-model", NOGOVA_FOREST_BLOCK_MODEL,
            "--forest-steep-model", NOGOVA_FOREST_STEEP_MODEL,
        ))
    return command


def build_fetch_command(values: dict[str, object], python: str | None = None) -> list[str]:
    source_dir = str(values.get("source_dir", "")).strip()
    if not source_dir:
        raise ValueError("Source directory is required")
    command = cli_command_prefix(python) + ["fetch-sources", "--source-dir", source_dir]
    mode = str(values.get("selection_mode", "map_url"))
    if mode == "map_url":
        map_url = str(values.get("map_url", "")).strip()
        if not map_url:
            raise ValueError("Map URL is required")
        command.extend(("--map-url", map_url))
    elif mode == "center":
        center = [str(values.get("center_lat", "")).strip(), str(values.get("center_lon", "")).strip()]
        if not all(center):
            raise ValueError("Centre latitude and longitude are required")
        command.extend(("--center", *center))
    else:
        bbox = [str(values.get(k, "")).strip() for k in ("south", "west", "north", "east")]
        if not all(bbox):
            raise ValueError("South, west, north and east are required")
        command.extend(("--bbox", *bbox))
    command.extend((
        "--cells", str(values.get("cells", DEFAULT_GUI_TERRAIN_CELLS)),
        "--cell-size", str(values.get("cell_size", DEFAULT_GUI_CELL_SIZE_METRES)),
    ))
    if values.get("refresh"):
        command.append("--refresh")
    if values.get("reference_map"):
        command.append("--reference-map")
    return command


def build_wizard_pipeline_commands(
    *,
    source_mode: str,
    fetch_values: dict[str, object],
    build_values: dict[str, object],
    python: str | None = None,
) -> list[list[str]]:
    """Return the final wizard pipeline for the selected source mode.

    Source selection never performs network or build work. A new area is fetched
    only after Review, immediately before the Milestone 9 build; an existing
    frozen bundle skips the fetch and starts the build directly.
    """
    build_command = build_milestone9_command(build_values, python=python)
    if source_mode == "new":
        return [
            build_fetch_command(fetch_values, python=python),
            build_command,
        ]
    if source_mode == "existing":
        return [build_command]
    raise ValueError(f"Unknown source mode: {source_mode}")


def slugify_world_name(value: str) -> str:
    """Return a conservative OFP/CWA-friendly world identifier."""
    text = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not text:
        text = "my_world"
    if text[0].isdigit():
        text = "world_" + text
    return text[:48]


def suggested_world_values(
    display_name: str,
    *,
    source_mode: str,
    source_dir: str,
) -> dict[str, str]:
    """Return name suggestions without replacing an existing source bundle."""
    slug = slugify_world_name(display_name)
    values = {
        "name": "cwr_" + slug if not slug.startswith("cwr_") else slug,
        "output": default_gui_path(Path("build") / slug),
        "source_dir": source_dir,
    }
    if source_mode == "new":
        values["source_dir"] = default_gui_path(Path("source-data") / slug)
    return values


def resolve_wizard_world_grid(values: dict[str, object]) -> tuple[int, float]:
    """Return the grid represented by the selected wizard source mode."""
    if str(values.get("source_mode", "new")) == "existing":
        source_text = str(values.get("source_dir", "")).strip()
        if not source_text:
            raise ValueError("Choose a source bundle directory.")
        manifest_path = resolve_gui_path(source_text) / "source.json"
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read source manifest {manifest_path}: {exc}") from exc
        selection = document.get("selection") if isinstance(document, Mapping) else None
        if not isinstance(selection, Mapping):
            raise ValueError("source manifest is missing its selection section")
        try:
            cells = int(selection.get("cells", 0))
            cell_size = float(selection.get("cell_size_metres", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("source manifest contains an invalid terrain grid") from exc
    else:
        cells = int(str(values.get("fetch_cells", values.get("cells", "0"))))
        cell_size = float(str(values.get("fetch_cell_size", values.get("cell_size", "0"))))
    if cells < 16 or cells > 2048 or cells & (cells - 1):
        raise ValueError("terrain cells must be a power of two between 16 and 2048")
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("terrain cell size must be positive and finite")
    return cells, cell_size


def format_wizard_world_size(values: dict[str, object]) -> str:
    """Format the selected source's real grid for the review page."""
    cells, cell_size = resolve_wizard_world_grid(values)
    size_km = cells * cell_size / 1000.0
    return f"{size_km:g} km ({cells}×{cells} at {cell_size:g} m)"


def default_gui_values() -> dict[str, object]:
    """Recommended wizard defaults, kept separate so tests can guard them."""
    south, west, north, east = square_bbox(59.45, 17.0, DEFAULT_GUI_TERRAIN_CELLS * DEFAULT_GUI_CELL_SIZE_METRES)
    return {
        "source_mode": "new",
        "source_dir": default_gui_path(Path("source-data") / "my_world"),
        "fetch_source_dir": default_gui_path(Path("source-data") / "my_world"),
        "selection_mode": "bbox",
        "map_url": "",
        "center_lat": "59.4500000",
        "center_lon": "17.0000000",
        "south": f"{south:.7f}",
        "west": f"{west:.7f}",
        "north": f"{north:.7f}",
        "east": f"{east:.7f}",
        "fetch_cells": str(DEFAULT_GUI_TERRAIN_CELLS),
        "fetch_cell_size": "25",
        "fetch_refresh": False,
        "reference_map": True,
        "output": default_gui_path(Path("build") / "my_world"),
        "deploy_to_mod_folder": False,
        "deploy_mod_dir": "",
        "name": "cwr_my_world",
        "display_name": "My CWA World",
        "profile": "cwr-ce",
        "appearance_preset": RECOMMENDED_APPEARANCE_PRESET,
        "ground_textures": "nogova",
        "forest_profile": "everon",
        "strict_assets": False,
        "osm_asset_map": "",
        "osm_asset_mapping_enabled": False,
        "osm_asset_mapping_inherit_defaults": True,
        "osm_asset_mapping_rules": "[]",
        "osm_asset_mapping_global_models": "",
        "osm_asset_mapping_global_textures": "",
        "keep_output": False,
        "include_minor_roads": True,
        "cache_refresh": False,
        "no_cache": False,
        "normalization_refresh": False,
        "verify_regeneration": False,
        "procedural_building_interiors": False,
        "high_quality_building_textures": False,
        "forest_clusters": True,
        "forest_severe_hill_fallback": True,
        "forest_severe_hill_relief": "5",
        "forest_severe_hill_trees_per_block": "10",
        "forest_undergrowth": True,
        "forest_undergrowth_max_objects": "120000",
        "forest_undergrowth_spacing": "30",
        "steep_hill_bushes": True,
        "max_steep_hill_bush_objects": "80000",
        "steep_hill_bush_spacing": "24",
        "steep_hill_bush_min_slope": "16",
        "forest_borders": True,
        "forest_single_trees": True,
        "forest_single_tree_model": r"data3d\str smrk_medium.p3d",
        "max_forest_single_tree_objects": "1000",
        "forest_single_tree_spacing": "45",
        "forest_single_tree_max_float": "0.5",
        "ditch_grass": True,
        "barriers": True,
        "bridges": True,
        "procedural_bridges": False,
        "bridge_module_length": "30",
        "bridge_deck_clearance": "1.25",
        "bridge_water_clearance": "18.0",
        "residential_infill": True,
        "overture_buildings": True,
        "max_residential_infill_buildings": "1500",
        "residential_infill_spacing": "68",
        "residential_infill_min_area": "1800",
        "rural_vegetation": True,
        "meadow_grass": True,
        "max_meadow_grass_objects": "20000",
        "meadow_grass_spacing": "24",
        "wetland_reeds": True,
        "max_wetland_reed_objects": "100000",
        "wetland_reed_spacing": "18",
        "rocky_forest_fallback": True,
        "bus_stop_signs": True,
        "bus_stop_ground_clearance": "0.12",
        "cemeteries": True,
        "max_grave_objects": "12000",
        "grave_spacing": "3.5",
        "grave_ground_clearance": "0.12",
        "grave_road_clearance": "1.0",
        "grave_building_clearance": "1.5",
        "rocky_forest_rocks_per_patch": "3",
        "rocky_forest_spread": "18",
        "lake_shore_smoothing_cells": "8",
        "lake_shore_max_slope": "8",
        "forest_max_block_relief": "8",
        "forest_block_max_ground_sink": "0",
        "forest_steep_max_relief": "18",
        "forest_polygon_sink_fraction": "0.5",
        "forest_steep_max_ground_sink": "0",
        "forest_cluster_max_relief": "48",
        "max_road_objects": str(DEFAULT_MAX_ROAD_OBJECTS),
        "max_buildings": str(DEFAULT_MAX_BUILDINGS),
        "building_ground_clearance": "0.10",
        "church_ground_clearance": "3.00",
        "building_foundation_depth": "0.50",
        "max_forest_objects": str(DEFAULT_MAX_FOREST_OBJECTS),
        "solver_iterations": "20",
        "overview_size": "1024",
        "cache_dir": "",
        "pbo_backend": "auto",
        "poseidon_tools": "",
        "advanced_args": "",
        "show_manual_area": False,
        "show_advanced": False,
        "show_command": False,
    }


def validate_wizard_step(step_index: int, values: dict[str, object], asset_roots: Iterable[str] = ()) -> None:
    """Validate only the information visible up to one wizard step."""
    if step_index == 0:
        source_dir = str(values.get("source_dir", "")).strip()
        if not source_dir:
            raise ValueError("Choose a source bundle directory.")
        if str(values.get("source_mode", "new")) == "existing":
            if not (resolve_gui_path(source_dir) / "source.json").is_file():
                raise ValueError("The selected folder is not a frozen source bundle. It needs a source.json file.")
            resolve_wizard_world_grid(values)
        else:
            build_fetch_command(values, python="python")
            cells = int(str(values.get("cells", "0")))
            cell_size = float(str(values.get("cell_size", "0")))
            if cells < 16 or cell_size <= 0:
                raise ValueError("Terrain cells must be at least 16 and cell size must be positive.")
    elif step_index == 1:
        build_milestone9_command(values, python="python")
        if bool(values.get("deploy_to_mod_folder", False)):
            deploy_path = resolve_gui_path(str(values.get("deploy_mod_dir", "")).strip())
            if not deploy_path.is_dir():
                raise ValueError("The deployment target must be an existing game mod folder.")
    elif step_index == 2:
        # Asset checking/mapping is optional and lives outside the main wizard,
        # but validate it here when somebody has explicitly enabled it.
        mapping_path = str(values.get("osm_asset_map", "")).strip()
        if mapping_path and not resolve_gui_path(mapping_path).is_file():
            raise ValueError("The OSM asset mapping JSON file does not exist.")
        integer_fields = (
            "max_road_objects",
            "max_buildings",
            "max_forest_objects",
            "forest_undergrowth_max_objects",
            "forest_severe_hill_trees_per_block",
            "max_steep_hill_bush_objects",
            "max_wetland_reed_objects",
            "max_grave_objects",
            "max_residential_infill_buildings",
            "rocky_forest_rocks_per_patch",
            "lake_shore_smoothing_cells",
            "solver_iterations",
            "overview_size",
        )
        float_fields = (
            "forest_max_block_relief",
            "forest_block_max_ground_sink",
            "forest_steep_max_relief",
            "forest_polygon_sink_fraction",
            "forest_severe_hill_relief",
            "forest_steep_max_ground_sink",
            "forest_cluster_max_relief",
            "rocky_forest_spread",
            "lake_shore_max_slope",
            "building_ground_clearance",
            "church_ground_clearance",
            "building_foundation_depth",
            "bus_stop_ground_clearance",
            "grave_spacing",
            "grave_ground_clearance",
            "grave_road_clearance",
            "grave_building_clearance",
            "bridge_module_length",
            "bridge_deck_clearance",
            "bridge_water_clearance",
            "residential_infill_spacing",
            "residential_infill_min_area",
        )
        for key in integer_fields:
            if int(str(values.get(key, "0"))) <= 0:
                raise ValueError(f"{key.replace('_', ' ').title()} must be a positive integer.")
        for key in float_fields:
            if float(str(values.get(key, "0"))) < 0:
                raise ValueError(f"{key.replace('_', ' ').title()} cannot be negative.")
        if float(str(values.get("bridge_module_length", "30"))) <= 0:
            raise ValueError("Bridge Module Length must be positive.")
        sink_fraction = float(str(values.get("forest_polygon_sink_fraction", "0.5")))
        if sink_fraction > 1.0:
            raise ValueError("Forest Polygon Sink Fraction must be within 0..1.")


class ScrollFrame(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.body = ttk.Frame(canvas, padding=18)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        self.body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class Disclosure(ttk.Frame):
    """Tiny progressive-disclosure panel for information humans should not face all at once."""

    def __init__(self, master: tk.Misc, title: str, variable: tk.Variable):
        super().__init__(master)
        self.title = title
        self.variable = variable
        self.button = ttk.Button(self, command=self._toggle)
        self.button.pack(anchor="w")
        self.body = ttk.Frame(self, padding=(18, 6, 0, 0))
        self.variable.trace_add("write", lambda *_args: self._render())
        self._render()

    def _toggle(self) -> None:
        self.variable.set(not bool(self.variable.get()))

    def _render(self) -> None:
        expanded = bool(self.variable.get())
        self.button.configure(text=("▼ " if expanded else "▶ ") + self.title)
        if expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()


class WorldgenGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1050x760")
        self.minsize(860, 620)
        self.process: subprocess.Popen[str] | None = None
        self._stop_requested = threading.Event()
        self.output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self.entry_widgets: dict[str, ttk.Entry] = {}
        self.browse_buttons: dict[str, ttk.Button] = {}
        self.advanced_setting_widgets: dict[str, list[tuple[ttk.Widget, str, str]]] = {}
        self._advanced_default_values = default_gui_values()
        self.asset_roots: list[str] = []
        self.profile_path: Path | None = None
        self.step_index = 0
        self.page_frames: list[ttk.Frame] = []
        self.step_labels: list[ttk.Label] = []
        self._refresh_scheduled = False
        self._preset_guard = False
        self._source_preview_key: tuple[str, int, int] | None = None
        self._source_preview_photo: ImageTk.PhotoImage | None = None
        self._pipeline_jobs: list[tuple[list[str], str]] = []
        self._pipeline_position = 0
        self._pipeline_kind = ""
        self._pipeline_success_message = ""
        self._operation_success = False
        self._pipeline_active = False
        self._pipeline_show_progress_page = True
        self._operation_return_step = APPEARANCE_STEP_INDEX
        self.state_path = gui_state_path()
        self._configure_style()
        self._build_ui()
        self._set_defaults()
        self._show_step(0)
        self.after(100, self._drain_output)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("AppTitle.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("PageTitle.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("PageSubtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Step.TLabel", font=("Segoe UI", 10), padding=(8, 8))
        style.configure("StepCurrent.TLabel", font=("Segoe UI", 10, "bold"), padding=(8, 8))
        style.configure("Hint.TLabel", font=("Segoe UI", 9))
        style.configure("Summary.TLabel", font=("Segoe UI", 11))
        style.configure("AdvancedChanged.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("AdvancedChanged.TCheckbutton", font=("Segoe UI", 9, "bold"))

    def _var(self, name: str, value: object = "", boolean: bool = False) -> tk.Variable:
        existing = self.vars.get(name)
        if existing is not None:
            return existing
        var: tk.Variable = tk.BooleanVar(value=bool(value)) if boolean else tk.StringVar(value=str(value))
        self.vars[name] = var
        var.trace_add("write", lambda *_args: self._schedule_refresh())
        return var

    def _schedule_refresh(self) -> None:
        if self._refresh_scheduled:
            return
        self._refresh_scheduled = True
        self.after_idle(self._refresh_views)

    def _refresh_views(self) -> None:
        self._refresh_scheduled = False
        self._apply_appearance_preset()
        self._update_source_mode_visibility()
        self._update_source_summary()
        self._update_source_preview()
        self._update_deploy_controls()
        self._update_review()
        self._update_command_preview()
        self._refresh_osm_mapping_tree()
        self._update_advanced_setting_styles()
        self._update_navigation()

    @staticmethod
    def _advanced_value_matches_default(current: object, default: object) -> bool:
        """Compare a GUI value with its default without bolding cosmetic number edits."""
        if isinstance(default, bool):
            return bool(current) is default
        current_text = str(current).strip()
        default_text = str(default).strip()
        if current_text == default_text:
            return True
        try:
            current_number = float(current_text)
            default_number = float(default_text)
        except ValueError:
            return False
        return (
            math.isfinite(current_number)
            and math.isfinite(default_number)
            and math.isclose(current_number, default_number, rel_tol=0.0, abs_tol=1.0e-12)
        )

    def _register_advanced_setting(
        self,
        key: str,
        widget: ttk.Widget,
        *,
        normal_style: str,
        changed_style: str,
    ) -> ttk.Widget:
        self.advanced_setting_widgets.setdefault(key, []).append(
            (widget, normal_style, changed_style)
        )
        return widget

    def _update_advanced_setting_styles(self) -> None:
        """Bold Advanced-setting text whenever its effective value is non-default."""
        for key, widgets in self.advanced_setting_widgets.items():
            variable = self.vars.get(key)
            if variable is None or key not in self._advanced_default_values:
                continue
            matches_default = self._advanced_value_matches_default(
                variable.get(), self._advanced_default_values[key]
            )
            for widget, normal_style, changed_style in widgets:
                widget.configure(style=normal_style if matches_default else changed_style)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(16, 12, 16, 8))
        header.pack(fill="x")
        ttk.Label(header, text="CWR Worldgen", style="AppTitle.TLabel").pack(side="left")
        ttk.Label(header, text=f"Wizard · version {__version__}").pack(side="left", padx=12)
        ttk.Button(header, text="Utilities…", command=self._open_utilities).pack(side="right")
        ttk.Button(header, text="Save profile", command=self._save_profile).pack(side="right", padx=6)
        ttk.Button(header, text="Load profile", command=self._load_profile).pack(side="right")

        separator = ttk.Separator(self)
        separator.pack(fill="x")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, padding=(12, 18, 8, 12), width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Build steps", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(0, 8))
        for index, step in enumerate(WIZARD_STEPS):
            label = ttk.Label(sidebar, text=f"{index + 1}.  {step.title}", style="Step.TLabel", anchor="w")
            label.pack(fill="x", pady=1)
            self.step_labels.append(label)

        ttk.Separator(body, orient="vertical").pack(side="left", fill="y")

        main = ttk.Frame(body, padding=(20, 14, 18, 8))
        main.pack(side="left", fill="both", expand=True)
        self.page_title_var = tk.StringVar()
        self.page_subtitle_var = tk.StringVar()
        ttk.Label(main, textvariable=self.page_title_var, style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(main, textvariable=self.page_subtitle_var, style="PageSubtitle.TLabel", wraplength=760).pack(anchor="w", pady=(2, 12))

        self.page_host = ttk.Frame(main)
        self.page_host.pack(fill="both", expand=True)
        self._build_source_page()
        self._build_world_page()
        self._build_appearance_page()
        self._build_progress_page()

        ttk.Separator(self).pack(fill="x")
        footer = ttk.Frame(self, padding=(14, 10))
        footer.pack(fill="x")
        self.footer_status_var = tk.StringVar(value="Ready")
        # Stop is deliberately ordered before the status label. The status text
        # can grow from a short "Ready" to a long stage/timestamp string while a
        # build runs; putting Stop after that label made the button slide sideways
        # as the label width changed. Keep a reference so Stop can be inserted
        # immediately before it whenever cancellation becomes available.
        self.footer_status_label = ttk.Label(footer, textvariable=self.footer_status_var)
        self.footer_status_label.pack(side="left")
        # Keep cancellation away from the primary action. Previously the same
        # right-hand button changed from "Build world" to "Stop" as soon as a
        # build started, so an ordinary double-click could immediately cancel
        # the job. The dedicated Stop control lives on the opposite side of the
        # footer and only appears while a process/pipeline is active.
        self.stop_button = ttk.Button(footer, text="Stop", command=self._stop_process)
        self.primary_button = ttk.Button(footer, text="Next", command=self._primary_action)
        self.primary_button.pack(side="right")
        self.back_button = ttk.Button(footer, text="Back", command=self._go_back)
        self.back_button.pack(side="right", padx=(0, 8))

    def _new_scroll_page(self) -> tuple[ttk.Frame, ttk.Frame]:
        page = ttk.Frame(self.page_host)
        scroll = ScrollFrame(page)
        scroll.pack(fill="both", expand=True)
        self.page_frames.append(page)
        return page, scroll.body

    def _entry_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        *,
        browse: str | None = None,
        width: int = 54,
        advanced: bool = False,
    ) -> ttk.Entry:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        if advanced:
            self._register_advanced_setting(
                key,
                label_widget,
                normal_style="TLabel",
                changed_style="AdvancedChanged.TLabel",
            )
        entry = ttk.Entry(parent, textvariable=self._var(key), width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        self.entry_widgets[key] = entry
        if browse:
            button = ttk.Button(parent, text="Browse…", command=lambda: self._browse(key, browse))
            button.grid(row=row, column=2, padx=(6, 0), pady=4)
            self.browse_buttons[key] = button
        parent.columnconfigure(1, weight=1)
        return entry

    def _build_source_page(self) -> None:
        _page, body = self._new_scroll_page()
        mode_box = ttk.LabelFrame(body, text="Source data", style="Section.TLabelframe", padding=12)
        mode_box.pack(fill="x")
        source_mode = self._var("source_mode", "new")
        ttk.Radiobutton(mode_box, text="Select a new area and fetch its heightmap + OSM data", variable=source_mode, value="new").pack(anchor="w", pady=2)
        ttk.Radiobutton(mode_box, text="Use an existing frozen source bundle", variable=source_mode, value="existing").pack(anchor="w", pady=2)

        path_box = ttk.Frame(body)
        path_box.pack(fill="x", pady=(12, 0))
        self._entry_row(path_box, 0, "Source bundle folder", "source_dir", browse="directory")
        ttk.Label(
            path_box,
            text="The wizard keeps downloaded map data here so later builds can be repeated without downloading it again.",
            style="Hint.TLabel",
            wraplength=560,
        ).grid(row=1, column=1, sticky="w", pady=(0, 8))

        self.source_summary_var = tk.StringVar(value="")
        area_box = ttk.LabelFrame(body, text="Selected area", style="Section.TLabelframe", padding=12)
        area_box.pack(fill="x", pady=(10, 0))
        self.new_area_box = area_box
        ttk.Label(area_box, textvariable=self.source_summary_var, style="Summary.TLabel", wraplength=720).pack(anchor="w")
        buttons = ttk.Frame(area_box)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Select area on OSM map…", command=self._open_osm_picker).pack(side="left")
        ttk.Button(buttons, text="Reset to recommended 6.4 km default", command=self._reset_default_area).pack(side="left", padx=8)
        ttk.Checkbutton(area_box, text="Download a reference map image", variable=self._var("reference_map", True, boolean=True)).pack(anchor="w", pady=(10, 0))

        existing_box = ttk.LabelFrame(body, text="Source preview", style="Section.TLabelframe", padding=12)
        existing_box.pack(fill="x", pady=(10, 0))
        self.existing_source_box = existing_box
        ttk.Label(existing_box, textvariable=self.source_summary_var, style="Summary.TLabel", wraplength=720).pack(anchor="w", pady=(0, 8))
        self.source_preview_label = ttk.Label(existing_box, anchor="center")
        self.source_preview_label.pack(fill="x")
        self.source_preview_caption_var = tk.StringVar(value="")
        ttk.Label(
            existing_box,
            textvariable=self.source_preview_caption_var,
            style="Hint.TLabel",
            wraplength=720,
            anchor="center",
            justify="center",
        ).pack(fill="x", pady=(7, 0))

        manual = Disclosure(body, "Manual coordinates and fetch settings", self._var("show_manual_area", False, boolean=True))
        manual.pack(fill="x", pady=(14, 0))
        self.manual_area_panel = manual
        selection = self._var("selection_mode", "bbox")
        row = ttk.Frame(manual.body)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="Selection method:").pack(side="left")
        for text, value in (("Bounding box", "bbox"), ("Centre", "center"), ("OpenTopoMap URL", "map_url")):
            ttk.Radiobutton(row, text=text, variable=selection, value=value).pack(side="left", padx=7)
        grid = ttk.Frame(manual.body)
        grid.pack(fill="x")
        self._entry_row(grid, 0, "OpenTopoMap URL", "map_url")
        self._entry_row(grid, 1, "Centre latitude", "center_lat", width=20)
        self._entry_row(grid, 2, "Centre longitude", "center_lon", width=20)
        for index, (label, key) in enumerate((("South", "south"), ("West", "west"), ("North", "north"), ("East", "east")), start=3):
            self._entry_row(grid, index, label, key, width=20)
        self._entry_row(grid, 7, "Terrain cells", "fetch_cells", width=20)
        self._entry_row(grid, 8, "Cell size (m)", "fetch_cell_size", width=20)
        ttk.Checkbutton(grid, text="Replace an existing source snapshot", variable=self._var("fetch_refresh", False, boolean=True)).grid(row=9, column=1, sticky="w", pady=4)

    def _build_world_page(self) -> None:
        _page, body = self._new_scroll_page()
        intro = ttk.Label(
            body,
            text="These defaults produce a valid, tidy mod folder. Change the display name first, then use the suggestion button if you want matching file names.",
            wraplength=760,
        )
        intro.pack(anchor="w", pady=(0, 12))
        world = ttk.LabelFrame(body, text="World identity", style="Section.TLabelframe", padding=12)
        world.pack(fill="x")
        self._entry_row(world, 0, "Display name", "display_name")
        self._entry_row(world, 1, "Internal world name", "name")
        self._entry_row(world, 2, "Output folder", "output", browse="directory")
        ttk.Checkbutton(
            world,
            text="Copy generated Addons and Anims into an existing game mod folder",
            variable=self._var("deploy_to_mod_folder", False, boolean=True),
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 2))
        self._entry_row(world, 4, "Existing game mod folder", "deploy_mod_dir", browse="directory")
        ttk.Label(
            world,
            text="Choosing a folder enables deployment automatically. The builder updates its existing Addons and Anims children and does not create another @mod wrapper.",
            style="Hint.TLabel",
            wraplength=700,
        ).grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(world, text="Game profile").grid(row=6, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(world, textvariable=self._var("profile"), values=("cwr-ce", "cwa"), state="readonly", width=18).grid(row=6, column=1, sticky="w", pady=4)
        ttk.Button(world, text="Suggest names from display name", command=self._suggest_names).grid(row=7, column=1, sticky="w", pady=(10, 0))
        ttk.Label(
            world,
            text="The internal name becomes part of the WRP and mission folder names, so it deliberately permits fewer characters than the display name.",
            style="Hint.TLabel",
            wraplength=700,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _update_deploy_controls(self) -> None:
        """Disable the deployment folder picker until copying is enabled."""
        if "deploy_to_mod_folder" not in self.vars:
            return
        state = "normal" if bool(self.vars["deploy_to_mod_folder"].get()) else "disabled"
        entry = self.entry_widgets.get("deploy_mod_dir")
        if entry is not None:
            entry.configure(state=state)
        button = self.browse_buttons.get("deploy_mod_dir")
        if button is not None:
            button.configure(state=state)

    def _build_assets_page(self) -> None:
        _page, body = self._new_scroll_page()
        ttk.Label(
            body,
            text="Add the CWA/CWR-CE folders or PBO archives that contain Eden textures, O objects and core Data/Data3D assets. The wizard can search common install locations, but Windows game installations have spent decades resisting standardization.",
            wraplength=760,
        ).pack(anchor="w", pady=(0, 12))
        assets = ttk.LabelFrame(body, text="Asset roots", style="Section.TLabelframe", padding=12)
        assets.pack(fill="x")
        self.asset_list = tk.Listbox(assets, height=8)
        self.asset_list.pack(side="left", fill="both", expand=True)
        asset_buttons = ttk.Frame(assets)
        asset_buttons.pack(side="right", fill="y", padx=(10, 0))
        ttk.Button(asset_buttons, text="Auto-detect", command=lambda: self._auto_detect_assets(silent=False)).pack(fill="x")
        ttk.Button(asset_buttons, text="Add game folder…", command=self._add_asset_folder).pack(fill="x", pady=(8, 0))
        ttk.Button(asset_buttons, text="Add PBO…", command=self._add_asset_file).pack(fill="x", pady=4)
        ttk.Button(asset_buttons, text="Remove selected", command=self._remove_asset).pack(fill="x")
        ttk.Checkbutton(
            body,
            text="Check every referenced game asset before building",
            variable=self._var("strict_assets", False, boolean=True),
        ).pack(anchor="w", pady=(12, 0))
        mapping = ttk.LabelFrame(body, text="OSM asset mapping", style="Section.TLabelframe", padding=12)
        mapping.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Checkbutton(
            mapping,
            text="Customize OSM-to-model and texture validation rules",
            variable=self._var("osm_asset_mapping_enabled", False, boolean=True),
            command=self._mapping_settings_changed,
        ).pack(anchor="w")
        ttk.Label(
            mapping,
            text="Optional. Disabled uses the current built-in mappings unchanged. Enable it to edit rules here; JSON import and export remain available for sharing.",
            style="Hint.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(3, 8))

        self.osm_mapping_tree = ttk.Treeview(
            mapping,
            columns=("source", "enabled", "layers", "geometry", "match", "assets"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        headings = {
            "source": "Source",
            "enabled": "On",
            "layers": "OSM layer",
            "geometry": "Geometry",
            "match": "Tag match",
            "assets": "Models / textures",
        }
        widths = {"source": 70, "enabled": 40, "layers": 110, "geometry": 75, "match": 220, "assets": 210}
        for column in self.osm_mapping_tree["columns"]:
            self.osm_mapping_tree.heading(column, text=headings[column])
            self.osm_mapping_tree.column(column, width=widths[column], minwidth=40, stretch=column in {"match", "assets"})
        self.osm_mapping_tree.pack(fill="x")
        self.osm_mapping_tree.bind("<Double-1>", lambda _event: self._edit_selected_osm_mapping_rule())

        mapping_buttons = ttk.Frame(mapping)
        mapping_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(mapping_buttons, text="Add rule…", command=self._add_osm_mapping_rule).pack(side="left")
        ttk.Button(mapping_buttons, text="Edit selected…", command=self._edit_selected_osm_mapping_rule).pack(side="left", padx=5)
        ttk.Button(mapping_buttons, text="Disable / remove", command=self._remove_osm_mapping_rule).pack(side="left")
        ttk.Button(mapping_buttons, text="Reset overrides", command=self._reset_osm_mapping_rules).pack(side="left", padx=5)
        ttk.Button(mapping_buttons, text="Global assets…", command=self._edit_osm_mapping_globals).pack(side="left")
        ttk.Button(mapping_buttons, text="Import JSON…", command=self._import_osm_mapping).pack(side="right")
        ttk.Button(mapping_buttons, text="Export JSON…", command=self._export_osm_mapping).pack(side="right", padx=5)

        options = ttk.Frame(mapping)
        options.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            options,
            text="Inherit built-in defaults",
            variable=self._var("osm_asset_mapping_inherit_defaults", True, boolean=True),
            command=self._mapping_settings_changed,
        ).pack(side="left")
        self.osm_mapping_summary_var = tk.StringVar(value="")
        ttk.Label(options, textvariable=self.osm_mapping_summary_var, style="Hint.TLabel").pack(side="right")
        self.asset_hint_var = tk.StringVar(value="")
        ttk.Label(body, textvariable=self.asset_hint_var, style="Hint.TLabel", wraplength=760).pack(anchor="w", pady=(6, 0))

    def _open_asset_tools(self) -> None:
        existing = getattr(self, "_asset_tools_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    return
            except tk.TclError:
                pass

        dialog = tk.Toplevel(self)
        self._asset_tools_dialog = dialog
        dialog.title(f"Optional asset tools - {APP_TITLE}")
        dialog.transient(self)
        dialog.geometry("960x720")
        dialog.minsize(820, 600)
        scroll = ScrollFrame(dialog)
        scroll.pack(fill="both", expand=True)
        body = scroll.body

        ttk.Label(
            body,
            text="Optional: use this only when you want strict game-asset checking or custom OSM-to-model/texture mappings. Normal wizard builds do not require this screen.",
            wraplength=820,
        ).pack(anchor="w", pady=(0, 12))

        assets = ttk.LabelFrame(body, text="Asset roots (optional)", style="Section.TLabelframe", padding=12)
        assets.pack(fill="x")
        self.asset_list = tk.Listbox(assets, height=7)
        self.asset_list.pack(side="left", fill="both", expand=True)
        for root in self.asset_roots:
            self.asset_list.insert("end", root)
        asset_buttons = ttk.Frame(assets)
        asset_buttons.pack(side="right", fill="y", padx=(10, 0))
        ttk.Button(asset_buttons, text="Auto-detect", command=lambda: self._auto_detect_assets(silent=False)).pack(fill="x")
        ttk.Button(asset_buttons, text="Add game folder…", command=self._add_asset_folder).pack(fill="x", pady=(8, 0))
        ttk.Button(asset_buttons, text="Add PBO…", command=self._add_asset_file).pack(fill="x", pady=4)
        ttk.Button(asset_buttons, text="Remove selected", command=self._remove_asset).pack(fill="x")
        ttk.Checkbutton(
            body,
            text="Check every referenced game asset before building",
            variable=self._var("strict_assets", False, boolean=True),
        ).pack(anchor="w", pady=(12, 0))

        mapping = ttk.LabelFrame(body, text="OSM asset mapping (optional)", style="Section.TLabelframe", padding=12)
        mapping.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Checkbutton(
            mapping,
            text="Customize OSM-to-model and texture validation rules",
            variable=self._var("osm_asset_mapping_enabled", False, boolean=True),
            command=self._mapping_settings_changed,
        ).pack(anchor="w")
        ttk.Label(
            mapping,
            text="Leave this disabled to use the built-in mappings unchanged.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 8))
        self.osm_mapping_tree = ttk.Treeview(
            mapping,
            columns=("source", "enabled", "layers", "geometry", "match", "assets"),
            show="headings", height=7, selectmode="browse",
        )
        headings = {
            "source": "Source", "enabled": "On", "layers": "OSM layer",
            "geometry": "Geometry", "match": "Tag match", "assets": "Models / textures",
        }
        widths = {"source": 70, "enabled": 40, "layers": 110, "geometry": 75, "match": 220, "assets": 210}
        for column in self.osm_mapping_tree["columns"]:
            self.osm_mapping_tree.heading(column, text=headings[column])
            self.osm_mapping_tree.column(column, width=widths[column], minwidth=40, stretch=column in {"match", "assets"})
        self.osm_mapping_tree.pack(fill="x")
        self.osm_mapping_tree.bind("<Double-1>", lambda _event: self._edit_selected_osm_mapping_rule())
        mapping_buttons = ttk.Frame(mapping)
        mapping_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(mapping_buttons, text="Add rule…", command=self._add_osm_mapping_rule).pack(side="left")
        ttk.Button(mapping_buttons, text="Edit selected…", command=self._edit_selected_osm_mapping_rule).pack(side="left", padx=5)
        ttk.Button(mapping_buttons, text="Disable / remove", command=self._remove_osm_mapping_rule).pack(side="left")
        ttk.Button(mapping_buttons, text="Reset overrides", command=self._reset_osm_mapping_rules).pack(side="left", padx=5)
        ttk.Button(mapping_buttons, text="Global assets…", command=self._edit_osm_mapping_globals).pack(side="left")
        ttk.Button(mapping_buttons, text="Import JSON…", command=self._import_osm_mapping).pack(side="right")
        ttk.Button(mapping_buttons, text="Export JSON…", command=self._export_osm_mapping).pack(side="right", padx=5)
        options = ttk.Frame(mapping)
        options.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            options, text="Inherit built-in defaults",
            variable=self._var("osm_asset_mapping_inherit_defaults", True, boolean=True),
            command=self._mapping_settings_changed,
        ).pack(side="left")
        self.osm_mapping_summary_var = tk.StringVar(value="")
        ttk.Label(options, textvariable=self.osm_mapping_summary_var, style="Hint.TLabel").pack(side="right")
        self.asset_hint_var = tk.StringVar(value="")
        ttk.Label(body, textvariable=self.asset_hint_var, style="Hint.TLabel", wraplength=820).pack(anchor="w", pady=(6, 0))
        ttk.Button(body, text="Close", command=dialog.destroy).pack(anchor="e", pady=(14, 0))

        def cleanup() -> None:
            self._asset_tools_dialog = None
            self.asset_list = None
            self.osm_mapping_tree = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cleanup)
        # The close button should perform the same cleanup.
        for child in body.winfo_children():
            if isinstance(child, ttk.Button) and child.cget("text") == "Close":
                child.configure(command=cleanup)
        self._refresh_osm_mapping_tree()
        self._refresh_views()

    def _gui_osm_mapping_path(self) -> Path:
        return self.state_path.parent / "osm-asset-map-gui.json"

    def _builtin_osm_mapping_rules(self) -> list[dict[str, object]]:
        cached = getattr(self, "_builtin_mapping_rule_cache", None)
        if cached is not None:
            return [dict(rule) for rule in cached]
        from .asset_mapping import default_osm_asset_mapping
        from .milestone9 import Milestone9Spec
        mapping = default_osm_asset_mapping(Milestone9Spec(source_dir=Path(".")), 9)
        cached = [rule.to_manifest() for rule in mapping.rules]
        self._builtin_mapping_rule_cache = cached
        return [dict(rule) for rule in cached]

    def _custom_osm_mapping_rules(self) -> list[dict[str, object]]:
        try:
            return parse_gui_osm_asset_rules(self.vars["osm_asset_mapping_rules"].get())
        except ValueError:
            return []

    @staticmethod
    def _mapping_conditions_text(value: object) -> str:
        if not isinstance(value, Mapping):
            return ""
        lines: list[str] = []
        for key, raw_values in value.items():
            values = raw_values if isinstance(raw_values, (list, tuple)) else [raw_values]
            lines.append(f"{key}=" + ", ".join(str(item) for item in values))
        return "\n".join(lines)

    @staticmethod
    def _parse_mapping_conditions(text: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Condition line {line_number} needs tag=value1,value2")
            key, raw_values = line.split("=", 1)
            key = key.strip()
            values = [item.strip() for item in raw_values.split(",") if item.strip()]
            if not key or not values:
                raise ValueError(f"Condition line {line_number} is incomplete")
            result[key] = values
        return result

    @staticmethod
    def _mapping_rule_assets(rule: Mapping[str, object]) -> str:
        models = rule.get("models", [])
        textures = rule.get("textures", [])
        model_count = len(models) if isinstance(models, list) else 0
        texture_count = len(textures) if isinstance(textures, list) else 0
        return f"{model_count} model(s), {texture_count} texture(s)"

    def _refresh_osm_mapping_tree(self) -> None:
        tree = getattr(self, "osm_mapping_tree", None)
        if tree is None:
            return
        try:
            if not tree.winfo_exists():
                return
        except tk.TclError:
            return
        for item in tree.get_children():
            tree.delete(item)
        builtins = self._builtin_osm_mapping_rules()
        customs = self._custom_osm_mapping_rules()
        custom_by_id = {str(rule.get("id", "")): rule for rule in customs}
        inherit = bool(self.vars["osm_asset_mapping_inherit_defaults"].get())
        ordered_ids = [str(rule.get("id", "")) for rule in builtins]
        ordered_ids.extend(
            str(rule.get("id", "")) for rule in customs
            if str(rule.get("id", "")) not in ordered_ids
        )
        builtin_by_id = {str(rule.get("id", "")): rule for rule in builtins}
        self._mapping_tree_rule_data: dict[str, tuple[dict[str, object], bool]] = {}
        for index, rule_id in enumerate(ordered_ids):
            custom = custom_by_id.get(rule_id)
            builtin = builtin_by_id.get(rule_id)
            if custom is not None:
                rule = custom
                source = "Override" if builtin is not None else "Custom"
                is_custom = True
            elif builtin is not None:
                rule = builtin
                source = "Built-in" if inherit else "Inactive"
                is_custom = False
            else:
                continue
            enabled = bool(rule.get("enabled", True)) and (inherit or is_custom)
            layers = rule.get("layers", ["*"])
            layer_text = ", ".join(str(item) for item in layers) if isinstance(layers, list) else str(layers)
            match = self._mapping_conditions_text(rule.get("match", {})).replace("\n", "; ") or "all tags"
            iid = f"mapping-{index}"
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(source, "Yes" if enabled else "No", layer_text, rule.get("geometry", "any"), match, self._mapping_rule_assets(rule)),
                tags=("disabled",) if not enabled else (),
            )
            self._mapping_tree_rule_data[iid] = (dict(rule), is_custom)
        tree.tag_configure("disabled", foreground="#777777")
        enabled_custom = sum(1 for rule in customs if bool(rule.get("enabled", True)))
        state = "enabled" if bool(self.vars["osm_asset_mapping_enabled"].get()) else "disabled"
        self.osm_mapping_summary_var.set(
            f"Customization {state}; {len(customs)} override(s), {enabled_custom} enabled"
        )

    def _mapping_settings_changed(self) -> None:
        enabled = bool(self.vars["osm_asset_mapping_enabled"].get())
        if enabled:
            try:
                self._ensure_gui_osm_mapping_file()
            except (OSError, ValueError) as exc:
                messagebox.showerror(APP_TITLE, f"Could not save OSM asset mapping:\n{exc}")
                self.vars["osm_asset_mapping_enabled"].set(False)
        else:
            if self.vars["osm_asset_map"].get():
                self.vars["osm_asset_map"].set("")
        self._persist_osm_mapping_state()
        self._refresh_osm_mapping_tree()

    def _ensure_gui_osm_mapping_file(self) -> Path | None:
        if "osm_asset_mapping_enabled" not in self.vars or not bool(self.vars["osm_asset_mapping_enabled"].get()):
            return None
        values = {name: var.get() for name, var in self.vars.items()}
        path = write_gui_osm_asset_mapping(self._gui_osm_mapping_path(), values)
        if str(self.vars["osm_asset_map"].get()) != str(path):
            self.vars["osm_asset_map"].set(str(path))
        return path

    def _persist_osm_mapping_state(self) -> None:
        values = {
            key: self.vars[key].get()
            for key in (
                "osm_asset_mapping_enabled",
                "osm_asset_mapping_inherit_defaults",
                "osm_asset_mapping_rules",
                "osm_asset_mapping_global_models",
                "osm_asset_mapping_global_textures",
            )
            if key in self.vars
        }
        try:
            update_gui_state(self.state_path, values)
        except OSError:
            pass

    def _selected_osm_mapping_rule(self) -> tuple[dict[str, object], bool] | None:
        if not hasattr(self, "osm_mapping_tree"):
            return None
        selection = self.osm_mapping_tree.selection()
        if not selection:
            return None
        return self._mapping_tree_rule_data.get(selection[0])

    def _add_osm_mapping_rule(self) -> None:
        self._open_osm_mapping_rule_editor({
            "id": "custom-rule",
            "layers": ["roads"],
            "geometry": "line",
            "match": {},
            "exclude": {},
            "models": [],
            "textures": [],
            "description": "",
            "enabled": True,
        })

    def _edit_selected_osm_mapping_rule(self) -> None:
        selected = self._selected_osm_mapping_rule()
        if selected is None:
            messagebox.showinfo(APP_TITLE, "Select a mapping rule first.")
            return
        rule, _is_custom = selected
        self._open_osm_mapping_rule_editor(rule)

    def _open_osm_mapping_rule_editor(self, initial: Mapping[str, object]) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Edit OSM asset mapping rule")
        dialog.transient(self)
        dialog.geometry("780x720")
        body = ttk.Frame(dialog, padding=14)
        body.pack(fill="both", expand=True)
        rule_id = tk.StringVar(value=str(initial.get("id", "")))
        enabled = tk.BooleanVar(value=bool(initial.get("enabled", True)))
        layers = tk.StringVar(value=", ".join(str(item) for item in initial.get("layers", ["*"])))
        geometry = tk.StringVar(value=str(initial.get("geometry", "any")))
        description = tk.StringVar(value=str(initial.get("description", "")))

        form = ttk.Frame(body)
        form.pack(fill="x")
        ttk.Label(form, text="Rule ID").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=rule_id).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Checkbutton(form, text="Enabled", variable=enabled).grid(row=0, column=2, padx=(10, 0))
        ttk.Label(form, text="Layers").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=layers).grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(form, text="Geometry").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(form, textvariable=geometry, values=("any", "point", "line", "polygon"), state="readonly").grid(row=2, column=1, sticky="w", pady=3)
        ttk.Label(form, text="Description").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=description).grid(row=3, column=1, columnspan=2, sticky="ew", pady=3)
        form.columnconfigure(1, weight=1)

        def text_box(title: str, value: str, height: int, hint: str) -> tk.Text:
            frame = ttk.LabelFrame(body, text=title, padding=7)
            frame.pack(fill="both", expand=True, pady=(8, 0))
            widget = tk.Text(frame, height=height, wrap="none", font=("Consolas", 9))
            widget.insert("1.0", value)
            widget.pack(fill="both", expand=True)
            ttk.Label(frame, text=hint, style="Hint.TLabel").pack(anchor="w", pady=(4, 0))
            return widget

        match_text = text_box("Required OSM tags", self._mapping_conditions_text(initial.get("match", {})), 4, "One per line: highway=service, track")
        exclude_text = text_box("Excluded OSM tags", self._mapping_conditions_text(initial.get("exclude", {})), 3, "One per line: surface=asphalt, paved")
        models_text = text_box("Required P3D models", "\n".join(str(item) for item in initial.get("models", [])), 5, "One model path per line, ending in .p3d")
        textures_text = text_box("Required PAA/PAC textures", "\n".join(str(item) for item in initial.get("textures", [])), 4, "One texture path per line, ending in .paa or .pac")

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(12, 0))

        def save() -> None:
            try:
                document: dict[str, object] = {
                    "id": rule_id.get().strip(),
                    "layers": [item.strip() for item in layers.get().split(",") if item.strip()],
                    "geometry": geometry.get(),
                    "match": self._parse_mapping_conditions(match_text.get("1.0", "end")),
                    "exclude": self._parse_mapping_conditions(exclude_text.get("1.0", "end")),
                    "models": [item.strip() for item in models_text.get("1.0", "end").splitlines() if item.strip()],
                    "textures": [item.strip() for item in textures_text.get("1.0", "end").splitlines() if item.strip()],
                    "description": description.get().strip(),
                    "enabled": enabled.get(),
                }
                from .asset_mapping import _parse_rule
                normalised = _parse_rule(document).to_manifest()
                customs = self._custom_osm_mapping_rules()
                customs = [rule for rule in customs if str(rule.get("id", "")) != str(normalised["id"])]
                customs.append(normalised)
                self.vars["osm_asset_mapping_rules"].set(json.dumps(customs, sort_keys=True))
                self.vars["osm_asset_mapping_enabled"].set(True)
                self._mapping_settings_changed()
            except (ValueError, OSError) as exc:
                messagebox.showerror(APP_TITLE, str(exc), parent=dialog)
                return
            dialog.destroy()

        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="Save rule", command=save).pack(side="right", padx=6)
        dialog.grab_set()

    def _remove_osm_mapping_rule(self) -> None:
        selected = self._selected_osm_mapping_rule()
        if selected is None:
            messagebox.showinfo(APP_TITLE, "Select a mapping rule first.")
            return
        rule, is_custom = selected
        rule_id = str(rule.get("id", ""))
        customs = self._custom_osm_mapping_rules()
        if is_custom:
            customs = [item for item in customs if str(item.get("id", "")) != rule_id]
        else:
            disabled = dict(rule)
            disabled["enabled"] = False
            customs.append(disabled)
        self.vars["osm_asset_mapping_rules"].set(json.dumps(customs, sort_keys=True))
        self.vars["osm_asset_mapping_enabled"].set(True)
        self._mapping_settings_changed()

    def _reset_osm_mapping_rules(self) -> None:
        if not messagebox.askyesno(APP_TITLE, "Remove all custom mapping rules and overrides?"):
            return
        self.vars["osm_asset_mapping_rules"].set("[]")
        self.vars["osm_asset_mapping_global_models"].set("")
        self.vars["osm_asset_mapping_global_textures"].set("")
        self.vars["osm_asset_mapping_inherit_defaults"].set(True)
        self._mapping_settings_changed()

    def _edit_osm_mapping_globals(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Global OSM mapping assets")
        dialog.transient(self)
        body = ttk.Frame(dialog, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="These assets are always required whenever this custom mapping is active.", wraplength=600).pack(anchor="w")
        models = tk.Text(body, height=8, width=76, font=("Consolas", 9))
        models.insert("1.0", str(self.vars["osm_asset_mapping_global_models"].get()))
        ttk.Label(body, text="P3D models").pack(anchor="w", pady=(10, 2))
        models.pack(fill="both", expand=True)
        textures = tk.Text(body, height=8, width=76, font=("Consolas", 9))
        textures.insert("1.0", str(self.vars["osm_asset_mapping_global_textures"].get()))
        ttk.Label(body, text="PAA/PAC textures").pack(anchor="w", pady=(10, 2))
        textures.pack(fill="both", expand=True)
        buttons = ttk.Frame(body); buttons.pack(fill="x", pady=(10, 0))
        def save() -> None:
            self.vars["osm_asset_mapping_global_models"].set(models.get("1.0", "end").strip())
            self.vars["osm_asset_mapping_global_textures"].set(textures.get("1.0", "end").strip())
            self.vars["osm_asset_mapping_enabled"].set(True)
            try:
                self._mapping_settings_changed()
            except Exception:
                return
            dialog.destroy()
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=save).pack(side="right", padx=6)
        dialog.grab_set()

    def _import_osm_mapping(self) -> None:
        path = filedialog.askopenfilename(filetypes=(("OSM asset mapping", "*.json"), ("All files", "*.*")))
        if not path:
            return
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(document, dict) or int(document.get("schema", 1)) != 1:
                raise ValueError("The selected file is not a schema-1 OSM asset mapping.")
            rules = document.get("rules", [])
            if not isinstance(rules, list):
                raise ValueError("Mapping rules must be a list.")
            global_section = document.get("global", {})
            if not isinstance(global_section, dict):
                raise ValueError("Mapping global section must be an object.")
            self.vars["osm_asset_mapping_inherit_defaults"].set(bool(document.get("inherit_defaults", True)))
            self.vars["osm_asset_mapping_rules"].set(json.dumps(rules, sort_keys=True))
            self.vars["osm_asset_mapping_global_models"].set("\n".join(str(item) for item in global_section.get("models", [])))
            self.vars["osm_asset_mapping_global_textures"].set("\n".join(str(item) for item in global_section.get("textures", [])))
            self.vars["osm_asset_mapping_enabled"].set(True)
            self._mapping_settings_changed()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror(APP_TITLE, f"Could not import mapping:\n{exc}")

    def _export_osm_mapping(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("OSM asset mapping", "*.json"),))
        if not path:
            return
        try:
            values = {name: var.get() for name, var in self.vars.items()}
            Path(path).write_text(json.dumps(build_gui_osm_asset_mapping_document(values), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.footer_status_var.set(f"Exported OSM asset mapping: {path}")
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_TITLE, f"Could not export mapping:\n{exc}")

    def _build_appearance_page(self) -> None:
        _page, body = self._new_scroll_page()
        preset = ttk.LabelFrame(body, text="Recommended preset", style="Section.TLabelframe", padding=12)
        preset.pack(fill="x")
        ttk.Label(preset, text="Appearance").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Combobox(
            preset,
            textvariable=self._var("appearance_preset", RECOMMENDED_APPEARANCE_PRESET),
            values=APPEARANCE_PRESETS,
            state="readonly",
            width=48,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(
            preset,
            text="The recommended preset mixes Nogova/Resistance ground textures with stock Everon forest and tree models. Nogova Resistance leaf forests uses the ordinary Resistance broadleaf forest family and Resistance leaf trees. Nogova Resistance pine forests uses the separate jehl conifer polygons and Resistance pine/spruce trees.",
            style="Hint.TLabel",
            wraplength=700,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        common = ttk.LabelFrame(body, text="Common choices", style="Section.TLabelframe", padding=12)
        common.pack(fill="x", pady=(12, 0))
        choices = (
            ("Include minor roads", "include_minor_roads", True),
            ("Enterable procedural-building interiors", "procedural_building_interiors", False),
            ("Higher-quality building textures (256 px)", "high_quality_building_textures", False),
            ("Forest undergrowth", "forest_undergrowth", True),
            ("Fences, walls and hedges", "barriers", True),
            ("Bridge decks", "bridges", True),
            ("Procedural bridges (instead of Nogova)", "procedural_bridges", False),
            ("Fill empty residential areas", "residential_infill", True),
            ("Use Overture buildings before randomly generated", "overture_buildings", True),
            ("Rural vegetation", "rural_vegetation", True),
            ("Tall grass in OSM meadows", "meadow_grass", True),
            ("Wetland reeds", "wetland_reeds", True),
            ("Steep-hill bushes", "steep_hill_bushes", True),
            ("Rocky forest fallback", "rocky_forest_fallback", True),
            ("Bus-stop signs", "bus_stop_signs", True),
            ("Cemetery gravestones", "cemeteries", True),
        )
        for index, (label, key, default) in enumerate(choices):
            ttk.Checkbutton(common, text=label, variable=self._var(key, default, boolean=True)).grid(
                row=index // 2,
                column=index % 2,
                sticky="w",
                padx=(0, 32),
                pady=3,
            )

        advanced = Disclosure(body, "Advanced generation, cache and PBO settings", self._var("show_advanced", False, boolean=True))
        advanced.pack(fill="x", pady=(14, 0))
        custom = ttk.LabelFrame(advanced.body, text="Custom appearance", padding=10)
        custom.pack(fill="x", pady=(0, 10))
        ground_label = ttk.Label(custom, text="Ground textures")
        ground_label.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        self._register_advanced_setting(
            "ground_textures", ground_label, normal_style="TLabel", changed_style="AdvancedChanged.TLabel"
        )
        ttk.Combobox(custom, textvariable=self._var("ground_textures"), values=("nogova", "malden", "everon", "desert", "generated"), state="readonly", width=18).grid(row=0, column=1, sticky="w")
        forest_profile_label = ttk.Label(custom, text="Forest profile")
        forest_profile_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        self._register_advanced_setting(
            "forest_profile", forest_profile_label, normal_style="TLabel", changed_style="AdvancedChanged.TLabel"
        )
        ttk.Combobox(custom, textvariable=self._var("forest_profile"), values=("everon", "malden"), state="readonly", width=18).grid(row=1, column=1, sticky="w")

        features = ttk.LabelFrame(advanced.body, text="Additional generated features", padding=10)
        features.pack(fill="x", pady=(0, 10))
        feature_values = (
            ("Steep forest clusters", "forest_clusters"),
            ("Steep/severe underbrush/tree fallback", "forest_severe_hill_fallback"),
            ("Extra single forest trees", "forest_single_trees"),
            ("Forest borders", "forest_borders"),
            ("Ditch grass", "ditch_grass"),
        )
        for index, (label, key) in enumerate(feature_values):
            check = ttk.Checkbutton(features, text=label, variable=self._var(key, True, boolean=True))
            check.grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 24))
            self._register_advanced_setting(
                key, check, normal_style="TCheckbutton", changed_style="AdvancedChanged.TCheckbutton"
            )

        tuning = ttk.LabelFrame(advanced.body, text="Limits and solver", padding=10)
        tuning.pack(fill="x", pady=(0, 10))
        fields = (
            ("Square forest max relief", "forest_max_block_relief", "8"),
            ("Triangle forest max relief", "forest_steep_max_relief", "18"),
            ("All-slope triangle sink fraction", "forest_polygon_sink_fraction", "0.5"),
            ("Triangle-unavailable fallback relief", "forest_severe_hill_relief", "5"),
            ("Steep fallback trees/block", "forest_severe_hill_trees_per_block", "10"),
            ("Cluster max relief", "forest_cluster_max_relief", "48"),
            ("Interior undergrowth max", "forest_undergrowth_max_objects", "120000"),
            ("Interior undergrowth spacing (m)", "forest_undergrowth_spacing", "30"),
            ("Steep-hill bushes max", "max_steep_hill_bush_objects", "80000"),
            ("Steep-hill bush spacing (m)", "steep_hill_bush_spacing", "24"),
            ("Steep-hill minimum slope (°)", "steep_hill_bush_min_slope", "16"),
            ("Wetland reeds max", "max_wetland_reed_objects", "100000"),
            ("Wetland reed spacing (m)", "wetland_reed_spacing", "18"),
            ("Meadow grass max", "max_meadow_grass_objects", "20000"),
            ("Meadow grass spacing (m)", "meadow_grass_spacing", "24"),
            ("Maximum extra single trees at 6.4 km", "max_forest_single_tree_objects", "1000"),
            ("Geographic single-tree spacing (m)", "forest_single_tree_spacing", "45"),
            ("Individual-tree maximum float (m)", "forest_single_tree_max_float", "0.5"),
            ("Rocks per rejected forest patch", "rocky_forest_rocks_per_patch", "3"),
            ("Rock scatter radius (m)", "rocky_forest_spread", "18"),
            ("Base lake smoothing at 6.4 km (cells)", "lake_shore_smoothing_cells", "8"),
            ("Maximum lake-bank slope (%)", "lake_shore_max_slope", "8"),
            ("Maximum road objects", "max_road_objects", str(DEFAULT_MAX_ROAD_OBJECTS)),
            ("Maximum buildings", "max_buildings", str(DEFAULT_MAX_BUILDINGS)),
            ("Visible building foundation (m)", "building_ground_clearance", "0.10"),
            ("Church minimum foundation depth (m)", "church_ground_clearance", "3.00"),
            ("Minimum foundation skirt (m)", "building_foundation_depth", "0.50"),
            ("Bus-stop ground clearance (m)", "bus_stop_ground_clearance", "0.12"),
            ("Maximum cemetery graves", "max_grave_objects", "12000"),
            ("Grave spacing (m)", "grave_spacing", "3.5"),
            ("Grave ground clearance (m)", "grave_ground_clearance", "0.12"),
            ("Procedural bridge module length (m)", "bridge_module_length", "30"),
            ("Bridge water clearance (minimum 18 m)", "bridge_water_clearance", "18.0"),
            ("Maximum infill houses", "max_residential_infill_buildings", "1500"),
            ("Residential infill spacing (m)", "residential_infill_spacing", "68"),
            ("Residential infill min area (m²)", "residential_infill_min_area", "1800"),
            ("Maximum forest objects", "max_forest_objects", str(DEFAULT_MAX_FOREST_OBJECTS)),
            ("Base terrain solver iterations", "solver_iterations", "20"),
            ("Overview size", "overview_size", "1024"),
        )
        for index, (label, key, default) in enumerate(fields):
            row, column = divmod(index, 2)
            cell = ttk.Frame(tuning)
            cell.grid(row=row, column=column, sticky="ew", padx=(0, 18), pady=3)
            field_label = ttk.Label(cell, text=label, width=27)
            field_label.pack(side="left")
            self._register_advanced_setting(
                key, field_label, normal_style="TLabel", changed_style="AdvancedChanged.TLabel"
            )
            ttk.Entry(cell, textvariable=self._var(key, default), width=12).pack(side="left")
            tuning.columnconfigure(column, weight=1)

        build_opts = ttk.LabelFrame(advanced.body, text="Build behavior", padding=10)
        build_opts.pack(fill="x", pady=(0, 10))
        option_values = (
            ("Keep existing output", "keep_output"),
            ("Refresh matching cache entries", "cache_refresh"),
            ("Disable all caches", "no_cache"),
            ("Refresh normalized geometry", "normalization_refresh"),
            ("Verify deterministic regeneration (slow)", "verify_regeneration"),
        )
        for index, (label, key) in enumerate(option_values):
            check = ttk.Checkbutton(build_opts, text=label, variable=self._var(key, False, boolean=True))
            check.grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 28), pady=3)
            self._register_advanced_setting(
                key, check, normal_style="TCheckbutton", changed_style="AdvancedChanged.TCheckbutton"
            )

        pbo = ttk.LabelFrame(advanced.body, text="PBO and CLI", padding=10)
        pbo.pack(fill="x")
        self._entry_row(pbo, 0, "Custom cache folder", "cache_dir", browse="directory", advanced=True)
        pbo_backend_label = ttk.Label(pbo, text="PBO backend")
        pbo_backend_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        self._register_advanced_setting(
            "pbo_backend", pbo_backend_label, normal_style="TLabel", changed_style="AdvancedChanged.TLabel"
        )
        ttk.Combobox(pbo, textvariable=self._var("pbo_backend", "auto"), values=("auto", "python", "poseidon"), state="readonly", width=18).grid(row=1, column=1, sticky="w")
        self._entry_row(pbo, 2, "PoseidonTools executable", "poseidon_tools", browse="file", advanced=True)
        self._entry_row(pbo, 3, "Additional CLI arguments", "advanced_args", advanced=True)

    def _build_progress_page(self) -> None:
        page = ttk.Frame(self.page_host)
        self.page_frames.append(page)
        status_box = ttk.Frame(page, padding=(4, 6))
        status_box.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_box, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(status_box, variable=self.progress_var, maximum=100).pack(fill="x", pady=(8, 0))
        log_frame = ttk.LabelFrame(page, text="Output", style="Section.TLabelframe", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.log = tk.Text(log_frame, wrap="word", font=("Consolas", 9), state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _set_defaults(self) -> None:
        defaults = defaults_with_recent_source(default_gui_values(), load_gui_state(self.state_path))
        for key, value in defaults.items():
            if key not in self.vars:
                self._var(key, value, boolean=isinstance(value, bool))
            self.vars[key].set(value)
        self._sync_source_paths()
        self._refresh_views()

    def _sync_source_paths(self) -> None:
        source = str(self.vars["source_dir"].get()).strip()
        if "fetch_source_dir" in self.vars:
            self.vars["fetch_source_dir"].set(source)

    def _collect_build_values(self) -> dict[str, object]:
        if bool(self.vars.get("osm_asset_mapping_enabled").get() if "osm_asset_mapping_enabled" in self.vars else False):
            try:
                self._ensure_gui_osm_mapping_file()
            except (OSError, ValueError):
                # Validation presents the useful error; previews should remain usable.
                pass
        elif "osm_asset_map" in self.vars and self.vars["osm_asset_map"].get():
            self.vars["osm_asset_map"].set("")
        values = {name: var.get() for name, var in self.vars.items()}
        for key in ("source_dir", "output", "cache_dir", "osm_asset_map", "deploy_mod_dir"):
            text = str(values.get(key, "")).strip()
            if text:
                values[key] = str(resolve_gui_path(text))
        values["asset_roots"] = [str(resolve_gui_path(root)) for root in self.asset_roots]
        return values

    def _collect_fetch_values(self) -> dict[str, object]:
        source_text = str(self.vars["source_dir"].get()).strip()
        return {
            "source_dir": str(resolve_gui_path(source_text)) if source_text else "",
            "selection_mode": self.vars["selection_mode"].get(),
            "map_url": self.vars["map_url"].get(),
            "center_lat": self.vars["center_lat"].get(),
            "center_lon": self.vars["center_lon"].get(),
            "south": self.vars["south"].get(),
            "west": self.vars["west"].get(),
            "north": self.vars["north"].get(),
            "east": self.vars["east"].get(),
            "cells": self.vars["fetch_cells"].get(),
            "cell_size": self.vars["fetch_cell_size"].get(),
            "refresh": self.vars["fetch_refresh"].get(),
            "reference_map": self.vars["reference_map"].get(),
        }

    def _validation_values(self) -> dict[str, object]:
        values = self._collect_build_values()
        values.update({
            "source_mode": self.vars["source_mode"].get(),
            "selection_mode": self.vars["selection_mode"].get(),
            "map_url": self.vars["map_url"].get(),
            "center_lat": self.vars["center_lat"].get(),
            "center_lon": self.vars["center_lon"].get(),
            "south": self.vars["south"].get(),
            "west": self.vars["west"].get(),
            "north": self.vars["north"].get(),
            "east": self.vars["east"].get(),
            "cells": self.vars["fetch_cells"].get(),
            "cell_size": self.vars["fetch_cell_size"].get(),
            "refresh": self.vars["fetch_refresh"].get(),
            "reference_map": self.vars["reference_map"].get(),
        })
        return values

    def _show_step(self, index: int) -> None:
        index = max(0, min(index, len(self.page_frames) - 1))
        if (self.process is not None or self._pipeline_active) and index != PROGRESS_STEP_INDEX:
            return
        for page in self.page_frames:
            page.pack_forget()
        self.step_index = index
        step = WIZARD_STEPS[index]
        self.page_title_var.set(step.title)
        self.page_subtitle_var.set(step.subtitle)
        self.page_frames[index].pack(fill="both", expand=True)
        for label_index, label in enumerate(self.step_labels):
            label.configure(style="StepCurrent.TLabel" if label_index == index else "Step.TLabel")
        self._refresh_views()

    def _update_navigation(self) -> None:
        if not hasattr(self, "primary_button"):
            return
        running = self.process is not None or self._pipeline_active
        self.back_button.configure(state="disabled" if running or self.step_index == 0 else "normal")
        if running:
            if not self.stop_button.winfo_manager():
                self.stop_button.pack(
                    side="left", padx=(0, 12), before=self.footer_status_label
                )
            self.stop_button.configure(text="Stop", state="normal")
            # Leave the primary control in its usual right-hand position but
            # disable it. A queued second click from "Build world" therefore
            # lands on a harmless disabled button rather than on Stop.
            self.primary_button.configure(text="Working…", state="disabled")
            return
        if self.stop_button.winfo_manager():
            self.stop_button.pack_forget()
        if self.step_index == 0:
            self.primary_button.configure(text="Next", state="normal")
        elif self.step_index < APPEARANCE_STEP_INDEX:
            self.primary_button.configure(text="Next", state="normal")
        elif self.step_index == APPEARANCE_STEP_INDEX:
            self.primary_button.configure(text="Build world", state="normal")
        elif self._operation_success and self._pipeline_kind == "build":
            self.primary_button.configure(text="Open output folder", state="normal")
        else:
            self.primary_button.configure(text="Return", state="normal")

    def _primary_action(self) -> None:
        # Cancellation is deliberately *not* a primary-button action. This also
        # protects against a double-click queued just as the build starts.
        if self.process is not None or self._pipeline_active:
            return
        if self.step_index < APPEARANCE_STEP_INDEX:
            self._go_next()
        elif self.step_index == APPEARANCE_STEP_INDEX:
            self._run_build_pipeline()
        elif self._operation_success and self._pipeline_kind == "build":
            self._open_path(str(self.vars["output"].get()))
        else:
            self._show_step(self._operation_return_step)

    def _go_next(self) -> None:
        try:
            validate_wizard_step(self.step_index, self._validation_values(), self.asset_roots)
        except (ValueError, TypeError) as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._show_step(self.step_index + 1)

    def _go_back(self) -> None:
        if self.process is None and not self._pipeline_active:
            if self.step_index == PROGRESS_STEP_INDEX:
                self._show_step(self._operation_return_step)
            else:
                self._show_step(self.step_index - 1)

    def _apply_appearance_preset(self) -> None:
        if self._preset_guard or "appearance_preset" not in self.vars:
            return
        preset = str(self.vars["appearance_preset"].get())
        desired: tuple[str, str, str] | None
        if preset == RECOMMENDED_APPEARANCE_PRESET:
            desired = ("nogova", "everon", EVERON_SINGLE_TREE_MODEL)
        elif preset == PINE_NOGOVA_APPEARANCE_PRESET:
            desired = ("nogova", "everon", NOGOVA_PINE_SINGLE_TREE_MODEL)
        elif preset in {RESISTANCE_APPEARANCE_PRESET, LEGACY_RESISTANCE_APPEARANCE_PRESET, LEGACY_NOGOVA_APPEARANCE_PRESET}:
            desired = ("nogova", "everon", NOGOVA_LEAF_SINGLE_TREE_MODEL)
        elif preset == "Malden classic":
            desired = ("malden", "malden", r"data3d\str_fikovnik.p3d")
        elif preset in {"Everon classic", "Everon classic (recommended)"}:
            desired = ("everon", "everon", EVERON_SINGLE_TREE_MODEL)
        elif preset == "Desert ground textures":
            desired = ("desert", "everon", EVERON_SINGLE_TREE_MODEL)
        elif preset == "Generated ground textures":
            desired = ("generated", "everon", EVERON_SINGLE_TREE_MODEL)
        else:
            desired = None
        if desired is None:
            return
        self._preset_guard = True
        try:
            if self.vars["ground_textures"].get() != desired[0]:
                self.vars["ground_textures"].set(desired[0])
            if self.vars["forest_profile"].get() != desired[1]:
                self.vars["forest_profile"].set(desired[1])
            if "forest_single_tree_model" in self.vars and self.vars["forest_single_tree_model"].get() != desired[2]:
                self.vars["forest_single_tree_model"].set(desired[2])
        finally:
            self._preset_guard = False

    def _update_source_mode_visibility(self) -> None:
        if not hasattr(self, "new_area_box"):
            return
        show_new = str(self.vars["source_mode"].get()) == "new"
        if show_new:
            if self.existing_source_box.winfo_manager():
                self.existing_source_box.pack_forget()
            if not self.new_area_box.winfo_manager():
                self.new_area_box.pack(fill="x", pady=(10, 0))
            if not self.manual_area_panel.winfo_manager():
                self.manual_area_panel.pack(fill="x", pady=(14, 0))
        else:
            self.new_area_box.pack_forget()
            self.manual_area_panel.pack_forget()
            if not self.existing_source_box.winfo_manager():
                self.existing_source_box.pack(fill="x", pady=(10, 0))

    def _update_source_summary(self) -> None:
        if not hasattr(self, "source_summary_var"):
            return
        source_text = str(self.vars["source_dir"].get()).strip()
        source_dir = resolve_gui_path(source_text) if source_text else application_base_dir()
        mode = str(self.vars["source_mode"].get())
        if mode == "existing":
            found = (source_dir / "source.json").is_file()
            text = f"Existing bundle: {source_dir or '(not selected)'}\n" + ("source.json found and ready." if found else "No source.json found yet.")
        else:
            try:
                cells = int(str(self.vars["fetch_cells"].get()))
                cell_size = float(str(self.vars["fetch_cell_size"].get()))
                size_km = cells * cell_size / 1000.0
                if cell_size == 25 and cells <= 128:
                    size_note = "compact high-detail default"
                elif cell_size == 25 and cells <= 256:
                    size_note = "standard high-detail grid"
                elif cell_size == 25:
                    size_note = "larger than the standard 256×256 grid"
                else:
                    size_note = "custom cell scale"
                text = f"{size_km:g} km × {size_km:g} km · {cells}×{cells} cells at {cell_size:g} m · {size_note}."
            except ValueError:
                text = "Area size is incomplete."
        self.source_summary_var.set(text)

    def _update_source_preview(self) -> None:
        if not hasattr(self, "source_preview_label"):
            return
        if str(self.vars["source_mode"].get()) != "existing":
            return
        source_text = str(self.vars["source_dir"].get()).strip()
        source_dir = resolve_gui_path(source_text) if source_text else application_base_dir()
        preview_path = existing_source_preview_path(source_dir)
        if preview_path is None:
            self._source_preview_key = None
            self._source_preview_photo = None
            self.source_preview_label.configure(image="")
            self.source_preview_caption_var.set(
                "No stored reference image was found. Sources fetched with “Download a reference map image” enabled will show it here."
            )
            return
        try:
            stat = preview_path.stat()
            key = (str(preview_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            key = (str(preview_path), 0, 0)
        if key == self._source_preview_key and self._source_preview_photo is not None:
            return
        try:
            with Image.open(preview_path) as source_image:
                original_width, original_height = source_image.size
                image = source_image.convert("RGB")
                image.thumbnail(
                    (SOURCE_PREVIEW_MAX_WIDTH, SOURCE_PREVIEW_MAX_HEIGHT),
                    Image.Resampling.LANCZOS,
                )
            photo = ImageTk.PhotoImage(image, master=self)
        except (OSError, ValueError, tk.TclError) as exc:
            self._source_preview_key = None
            self._source_preview_photo = None
            self.source_preview_label.configure(image="")
            self.source_preview_caption_var.set(f"Could not open reference preview: {exc}")
            return
        self._source_preview_key = key
        self._source_preview_photo = photo
        self.source_preview_label.configure(image=photo)
        try:
            relative = preview_path.relative_to(source_dir)
        except ValueError:
            relative = preview_path.name
        self.source_preview_caption_var.set(
            f"{relative} · {original_width}×{original_height} · stored source reference"
        )

    def _review_warnings(self) -> list[str]:
        warnings: list[str] = []
        if bool(self.vars["strict_assets"].get()) and not self.asset_roots:
            warnings.append("⚠ Strict asset validation is enabled, but no asset roots are listed.")
        try:
            cells, _cell_size = resolve_wizard_world_grid(self._validation_values())
            if cells > 256:
                warnings.append(f"⚠ {cells}×{cells} cells is larger than the recommended OFP/CWA 256×256 grid.")
        except (OSError, TypeError, ValueError) as exc:
            warnings.append(f"⚠ Source grid is invalid: {exc}")
        if str(self.vars["ground_textures"].get()) == "desert":
            warnings.append("• Desert generated ground textures are selected instead of stock Everon/Eden textures.")
        elif str(self.vars["ground_textures"].get()) not in {"nogova", "everon"}:
            warnings.append("• Generated ground textures are selected instead of stock Everon/Eden textures.")
        if bool(self.vars["verify_regeneration"].get()):
            warnings.append("• Slow deterministic regeneration verification is enabled.")
        if bool(self.vars["no_cache"].get()):
            warnings.append("• All caches are disabled, so the build will do more work.")
        if not warnings:
            warnings.append("✓ Recommended defaults are selected. No obvious configuration warnings.")
        return warnings

    def _update_review(self) -> None:
        if not hasattr(self, "review_text_var"):
            return
        source_mode = str(self.vars["source_mode"].get())
        action = "Build from the frozen source bundle" if source_mode == "existing" else "Fetch selected source, then build world"
        try:
            world_size = format_wizard_world_size(self._validation_values())
        except (OSError, TypeError, ValueError) as exc:
            world_size = f"invalid size ({exc})"
        summary = (
            f"Action: {action}\n"
            f"World: {self.vars['display_name'].get()}  [{self.vars['name'].get()}]\n"
            f"Area: {world_size}\n"
            f"Source: {self.vars['source_dir'].get()}\n"
            f"Output: {self.vars['output'].get()}\n"
            f"Deploy: {self.vars['deploy_mod_dir'].get() if self.vars['deploy_to_mod_folder'].get() else 'disabled'}\n"
            f"Profile: {self.vars['profile'].get()}\n"
            f"Appearance: {self.vars['appearance_preset'].get()}\n"
            f"Undergrowth: half amount\n"
            f"Lake banks: {self.vars['lake_shore_smoothing_cells'].get()} cells, max {self.vars['lake_shore_max_slope'].get()}% rise\n"
            f"Buildings: {self.vars['building_ground_clearance'].get()} m visible foundation, {self.vars['church_ground_clearance'].get()} m church minimum foundation, {self.vars['building_foundation_depth'].get()} m minimum skirt depth\n"
            f"Asset checker: {'enabled' if self.vars['strict_assets'].get() else 'optional / off'} ({len(self.asset_roots)} root(s))\n"
            f"OSM asset mapping: {'custom optional rules' if self.vars['osm_asset_mapping_enabled'].get() else 'built-in defaults'}"
        )
        self.review_text_var.set(summary)
        self.review_warning_var.set("\n".join(self._review_warnings()))
        if hasattr(self, "asset_hint_var"):
            if self.asset_roots:
                self.asset_hint_var.set(f"{len(self.asset_roots)} asset root(s) will be searched during validation.")
            else:
                self.asset_hint_var.set("No asset roots detected. Add the game folder or the required PBO archives before a strict build.")

    def _update_command_preview(self) -> None:
        if not hasattr(self, "command_preview"):
            return
        lines: list[str] = []
        try:
            commands = build_wizard_pipeline_commands(
                source_mode=str(self.vars["source_mode"].get()),
                fetch_values=self._collect_fetch_values(),
                build_values=self._collect_build_values(),
            )
            lines.extend(quote_command(command) for command in commands)
        except ValueError as exc:
            lines.append(str(exc))
        self.command_preview.configure(state="normal")
        self.command_preview.delete("1.0", "end")
        self.command_preview.insert("1.0", "\n\n".join(lines))
        self.command_preview.configure(state="disabled")

    def _suggest_names(self) -> None:
        values = suggested_world_values(
            str(self.vars["display_name"].get()),
            source_mode=str(self.vars["source_mode"].get()),
            source_dir=str(self.vars["source_dir"].get()),
        )
        for key, value in values.items():
            self.vars[key].set(value)
        self._sync_source_paths()

    def _reset_default_area(self) -> None:
        defaults = default_gui_values()
        for key in ("selection_mode", "center_lat", "center_lon", "south", "west", "north", "east", "fetch_cells", "fetch_cell_size"):
            self.vars[key].set(defaults[key])
        self.footer_status_var.set("Restored the recommended 6.4 km default area.")

    def _initial_map_selection(self) -> tuple[tuple[float, float], int, tuple[float, float, float, float] | None]:
        bbox: tuple[float, float, float, float] | None = None
        try:
            values = tuple(float(str(self.vars[key].get()).strip()) for key in ("south", "west", "north", "east"))
            if values[0] < values[2] and values[1] < values[3]:
                bbox = values  # type: ignore[assignment]
        except (ValueError, KeyError):
            bbox = None
        if bbox is not None:
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            return center, zoom_for_bbox(bbox, 900, 600), bbox
        try:
            url = str(self.vars["map_url"].get()).strip()
            if url:
                link = parse_opentopomap_link(url)
                return (link.latitude, link.longitude), int(round(link.zoom)), None
        except ValueError:
            pass
        try:
            latitude = float(str(self.vars["center_lat"].get()).strip())
            longitude = float(str(self.vars["center_lon"].get()).strip())
            return (latitude, longitude), 10, None
        except ValueError:
            return (59.45, 17.0), 9, None

    def _open_osm_picker(self) -> None:
        center, zoom, bbox = self._initial_map_selection()
        try:
            cell_size = float(str(self.vars["fetch_cell_size"].get()).strip())
        except (ValueError, KeyError):
            cell_size = 25.0
        try:
            cells = int(str(self.vars["fetch_cells"].get()).strip())
        except (ValueError, KeyError):
            cells = None
        OsmAreaPicker(
            self,
            initial_center=center,
            initial_zoom=zoom,
            initial_bbox=bbox,
            initial_cells=cells,
            cell_size_metres=cell_size,
            on_apply=self._apply_osm_selection,
        )

    def _apply_osm_selection(self, plan: AreaSelectionPlan, fetch: bool) -> None:
        south, west, north, east = plan.bbox
        for key, value in (("south", south), ("west", west), ("north", north), ("east", east)):
            self.vars[key].set(f"{value:.7f}")
        self.vars["selection_mode"].set("bbox")
        self.vars["source_mode"].set("new")
        self.vars["fetch_cells"].set(str(plan.cells))
        self.vars["fetch_cell_size"].set(f"{plan.cell_size_metres:g}")
        self.footer_status_var.set(f"Selected {plan.world_size_metres / 1000:.1f} km area ({plan.cells}×{plan.cells} cells).")
        if fetch:
            self._run_fetch_only()

    def _browse(self, key: str, kind: str) -> None:
        current = str(self.vars[key].get()).strip()
        initial = str(resolve_gui_path(current)) if current else str(application_base_dir())
        if kind == "directory":
            selected = filedialog.askdirectory(initialdir=initial)
        else:
            selected = filedialog.askopenfilename(initialdir=initial)
        if selected:
            self.vars[key].set(selected)
            if key == "source_dir":
                self._sync_source_paths()

    def _add_asset_folder(self) -> None:
        value = filedialog.askdirectory()
        if value:
            self._add_asset(value)

    def _add_asset_file(self) -> None:
        value = filedialog.askopenfilename(filetypes=(("PBO archives", "*.pbo"), ("All files", "*.*")))
        if value:
            self._add_asset(value)

    def _add_asset(self, value: str) -> None:
        normalized = str(Path(value))
        if normalized not in self.asset_roots:
            self.asset_roots.append(normalized)
            asset_list = getattr(self, "asset_list", None)
            if asset_list is not None:
                try:
                    if asset_list.winfo_exists():
                        asset_list.insert("end", normalized)
                except tk.TclError:
                    pass
            self._refresh_views()

    def _remove_asset(self) -> None:
        asset_list = getattr(self, "asset_list", None)
        if asset_list is None:
            return
        selected = list(asset_list.curselection())
        for index in reversed(selected):
            asset_list.delete(index)
            del self.asset_roots[index]
        self._refresh_views()

    def _candidate_game_roots(self) -> list[Path]:
        candidates: list[Path] = []
        for variable in ("CWA_HOME", "CWR_CE_HOME", "OFP_HOME"):
            value = os.environ.get(variable)
            if value:
                candidates.append(Path(value))
        if os.name == "nt":
            for base_var in ("ProgramFiles", "ProgramFiles(x86)"):
                base = os.environ.get(base_var)
                if not base:
                    continue
                base_path = Path(base)
                candidates.extend((
                    base_path / "Bohemia Interactive" / "Cold War Assault",
                    base_path / "Steam" / "steamapps" / "common" / "Arma Cold War Assault",
                    base_path / "GOG Galaxy" / "Games" / "Arma Cold War Assault",
                ))
        app_dir = application_base_dir()
        candidates.extend((app_dir, app_dir.parent))
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate).casefold()
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    def _auto_detect_assets(self, *, silent: bool) -> None:
        found: list[Path] = []
        for root in self._candidate_game_roots():
            if not root.exists():
                continue
            for relative in (
                Path("Dta") / "Eden.pbo",
                Path("Dta") / "eden.pbo",
                Path("Res") / "AddOns" / "O.pbo",
                Path("Res") / "Addons" / "O.pbo",
                Path("Dta"),
            ):
                candidate = root / relative
                if candidate.exists():
                    found.append(candidate)
        for candidate in found:
            self._add_asset(str(candidate))
        if not silent:
            if found:
                messagebox.showinfo(APP_TITLE, f"Added {len(found)} detected asset path(s).")
            else:
                messagebox.showinfo(APP_TITLE, "No common CWA/CWR-CE installation was detected. Add the game folder or PBO files manually.")

    def _run_fetch_only(self) -> None:
        try:
            validate_wizard_step(0, self._validation_values(), self.asset_roots)
            command = build_fetch_command(self._collect_fetch_values())
        except (ValueError, TypeError) as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._start_pipeline(
            [(command, "Fetching heightmap and OSM source data")],
            kind="fetch",
            success_message="Source data fetched successfully.",
            return_step=1,
            show_progress_page=False,
        )

    def _run_build_pipeline(self) -> None:
        values = self._validation_values()
        source_mode = str(self.vars["source_mode"].get())
        try:
            for step in range(3):
                validate_wizard_step(step, values, self.asset_roots)
            commands = build_wizard_pipeline_commands(
                source_mode=source_mode,
                fetch_values=self._collect_fetch_values(),
                build_values=self._collect_build_values(),
            )
        except (ValueError, TypeError) as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        if bool(self.vars["strict_assets"].get()) and not self.asset_roots:
            if not messagebox.askyesno(APP_TITLE, "Strict asset validation is enabled but no asset roots are listed. Start anyway?"):
                self._open_asset_tools()
                return
        jobs: list[tuple[list[str], str]] = []
        for command in commands:
            if "fetch-sources" in command:
                jobs.append((command, "Fetching heightmap and OSM source data"))
            else:
                jobs.append((command, "Building Milestone 9 world"))
        self._start_pipeline(
            jobs,
            kind="build",
            success_message="World build completed successfully.",
            return_step=APPEARANCE_STEP_INDEX,
        )

    def _start_pipeline(
        self,
        jobs: list[tuple[list[str], str]],
        *,
        kind: str,
        success_message: str,
        return_step: int | None = None,
        show_progress_page: bool = True,
    ) -> None:
        if self.process is not None or self._pipeline_active:
            messagebox.showwarning(APP_TITLE, "A process is already running")
            return
        self._stop_requested.clear()
        self._pipeline_jobs = jobs
        self._pipeline_position = 0
        self._pipeline_kind = kind
        self._pipeline_success_message = success_message
        self._pipeline_show_progress_page = bool(show_progress_page)
        self._operation_return_step = self.step_index if return_step is None else return_step
        self._operation_success = False
        self._pipeline_active = True
        self.progress_var.set(0.0)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        if self._pipeline_show_progress_page:
            self._show_step(PROGRESS_STEP_INDEX)
        else:
            self._update_navigation()
        self._start_current_job()

    def _start_current_job(self) -> None:
        command, description = self._pipeline_jobs[self._pipeline_position]
        self.status_var.set(description + "…")
        self.footer_status_var.set(description + "…")
        self._append_log("\n> " + quote_command(command) + "\n")
        self._update_navigation()

        def worker() -> None:
            try:
                child_env = os.environ.copy()
                child_env["CWR_PROGRESS"] = "1"
                child_env["CWR_PROGRESS_FORMAT"] = "machine"
                if self._stop_requested.is_set():
                    self.output_queue.put(("cancelled", None))
                    return
                process = subprocess.Popen(
                    command,
                    cwd=application_base_dir(),
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                    start_new_session=os.name != "nt",
                )
                self.process = process
                # The user can click Stop during the tiny window between the
                # pipeline becoming active and Popen returning. Honour that
                # request immediately instead of silently losing the click.
                if self._stop_requested.is_set():
                    terminate_process_tree(process)
                assert process.stdout is not None
                for line in process.stdout:
                    self.output_queue.put(("line", line))
                code = process.wait()
                self.output_queue.put(("done", code))
            except OSError as exc:
                self.output_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_output(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "line":
                    item = str(payload)
                    if item.startswith("CWR_PROGRESS\t"):
                        parts = item.rstrip("\r\n").split("\t")
                        if len(parts) >= 3:
                            try:
                                child_progress = float(parts[1])
                                overall = (self._pipeline_position + child_progress / 100.0) / max(1, len(self._pipeline_jobs)) * 100.0
                                self.progress_var.set(overall)
                            except ValueError:
                                child_progress = 0.0
                            stage = parts[2]
                            if len(parts) >= 5:
                                timestamp, elapsed = parts[3], parts[4]
                                status = f"{stage}  •  {elapsed} elapsed  •  {timestamp}"
                                self._append_log(
                                    f"[{timestamp}] [+{elapsed}] {child_progress:3.0f}%  {stage}\n"
                                )
                            else:
                                status = stage
                            self.status_var.set(status)
                            self.footer_status_var.set(status)
                    else:
                        self._append_log(item)
                elif kind == "done":
                    code = int(payload)
                    self.process = None
                    self._append_log(f"\nProcess finished with exit code {code}.\n")
                    if self._stop_requested.is_set():
                        self._operation_success = False
                        self._pipeline_active = False
                        self._pipeline_jobs = []
                        self.status_var.set("Stopped by user.")
                        self.footer_status_var.set("Stopped by user.")
                        self._append_log("Stop requested. Remaining pipeline jobs were cancelled.\n")
                        self._update_navigation()
                        continue
                    if code == 0:
                        command, _description = self._pipeline_jobs[self._pipeline_position]
                        if "fetch-sources" in command:
                            self._remember_downloaded_source(command)
                        if "milestone9" in command:
                            self._remember_world_name(command)
                            self._remember_deploy_mod_folder(command)
                    if code == 0 and self._pipeline_position + 1 < len(self._pipeline_jobs):
                        self._pipeline_position += 1
                        self._start_current_job()
                    elif code == 0:
                        self._operation_success = True
                        self._pipeline_active = False
                        self.progress_var.set(100.0)
                        self.status_var.set(self._pipeline_success_message)
                        self.footer_status_var.set(self._pipeline_success_message)
                        if self._pipeline_kind == "fetch" and not self._pipeline_show_progress_page:
                            # Fetching belongs to step 1. Once the frozen source
                            # exists, continue to the actual next wizard page
                            # instead of flashing the step-5 build log page.
                            self._show_step(self._operation_return_step)
                        else:
                            self._update_navigation()
                    else:
                        self._operation_success = False
                        self._pipeline_active = False
                        self.status_var.set(f"Operation failed with exit code {code}.")
                        if self._pipeline_kind == "fetch" and not self._pipeline_show_progress_page:
                            self.footer_status_var.set("Source fetch failed. Check the error dialog and try again.")
                            self._update_navigation()
                            try:
                                output_tail = self.log.get("1.0", "end").strip()[-2400:]
                            except tk.TclError:
                                output_tail = ""
                            detail = f"\n\n{output_tail}" if output_tail else ""
                            messagebox.showerror(APP_TITLE, f"Source fetch failed with exit code {code}.{detail}")
                        else:
                            self.footer_status_var.set("Build failed. Review the output above.")
                            self._update_navigation()
                elif kind == "cancelled":
                    self.process = None
                    self._operation_success = False
                    self._pipeline_active = False
                    self._pipeline_jobs = []
                    self.status_var.set("Stopped by user.")
                    self.footer_status_var.set("Stopped by user.")
                    self._append_log("Stop requested before the next process started.\n")
                    self._update_navigation()
                elif kind == "error":
                    self.process = None
                    self._operation_success = False
                    self._pipeline_active = False
                    if self._stop_requested.is_set():
                        self._pipeline_jobs = []
                        self.status_var.set("Stopped by user.")
                        self.footer_status_var.set("Stopped by user.")
                        self._append_log("Stop requested before the process could start.\n")
                        self._update_navigation()
                        continue
                    self._append_log(f"\nFailed to start process: {payload}\n")
                    self.status_var.set("Could not start the process.")
                    self.footer_status_var.set("Could not start the process.")
                    self._update_navigation()
                    if self._pipeline_kind == "fetch" and not self._pipeline_show_progress_page:
                        messagebox.showerror(APP_TITLE, f"Could not start source fetch:\n\n{payload}")
        except queue.Empty:
            pass
        self.after(100, self._drain_output)

    def _remember_downloaded_source(self, command: list[str]) -> None:
        source_dir = command_option_value(command, "--source-dir") or str(self.vars["source_dir"].get()).strip()
        source_path = Path(source_dir).expanduser() if source_dir else None
        if source_path is None or not (source_path / "source.json").is_file():
            return
        source_dir = str(source_path.resolve())
        try:
            update_gui_state(self.state_path, {"last_downloaded_source": source_dir})
        except OSError as exc:
            self._append_log(f"\nCould not remember downloaded source: {exc}\n")
            return
        self.vars["source_dir"].set(source_dir)
        self.vars["source_mode"].set("existing")
        self._sync_source_paths()
        self.footer_status_var.set(f"Remembered downloaded source: {source_dir}")

    def _remember_deploy_mod_folder(self, command: list[str]) -> None:
        deploy_dir = command_option_value(command, "--deploy-mod-dir")
        if not deploy_dir:
            try:
                update_gui_state(self.state_path, {"deploy_to_mod_folder": False})
            except OSError:
                pass
            return
        deploy_path = Path(deploy_dir).expanduser()
        if not deploy_path.is_dir():
            return
        deploy_dir = str(deploy_path.resolve())
        try:
            update_gui_state(
                self.state_path,
                {"last_deploy_mod_dir": deploy_dir, "deploy_to_mod_folder": True},
            )
        except OSError as exc:
            self._append_log(f"\nCould not remember deployment mod folder: {exc}\n")
            return
        self.vars["deploy_mod_dir"].set(deploy_dir)
        self.vars["deploy_to_mod_folder"].set(True)

    def _remember_world_name(self, command: list[str]) -> None:
        name = command_option_value(command, "--name") or str(self.vars["name"].get()).strip()
        display_name = command_option_value(command, "--display-name") or str(self.vars["display_name"].get()).strip()
        output = command_option_value(command, "--output") or str(self.vars["output"].get()).strip()
        values: dict[str, object] = {}
        if name:
            values["last_world_name"] = name
        if display_name:
            values["last_world_display_name"] = display_name
        if output:
            values["last_world_output"] = output
        if not values:
            return
        try:
            update_gui_state(self.state_path, values)
        except OSError as exc:
            self._append_log(f"\nCould not remember world name: {exc}\n")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _stop_process(self) -> None:
        if self.process is None and not self._pipeline_active:
            return
        # Set this before looking at self.process: Popen happens on a worker
        # thread, so Stop may be clicked while the child is still being created.
        self._stop_requested.set()
        self.status_var.set("Stopping…")
        self.footer_status_var.set("Stopping…")
        self.stop_button.configure(text="Stopping…", state="disabled")
        self.primary_button.configure(text="Working…", state="disabled")
        process = self.process
        if process is not None:
            threading.Thread(
                target=terminate_process_tree, args=(process,), daemon=True
            ).start()

    def _open_utilities(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title(f"Utilities - {APP_TITLE}")
        dialog.transient(self)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Source tools", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Button(body, text="Inspect source bundle", command=lambda: self._run_tool("inspect-sources", "source_dir", "--source-dir")).pack(fill="x", pady=(8, 3))
        ttk.Button(body, text="Normalize source bundle", command=lambda: self._run_tool("normalize-sources", "source_dir", "--source-dir")).pack(fill="x", pady=3)
        ttk.Button(body, text="Inspect normalized bundle", command=self._inspect_normalized).pack(fill="x", pady=3)
        ttk.Separator(body).pack(fill="x", pady=12)
        ttk.Label(body, text="Optional build tools", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Button(body, text="Asset checker & OSM mapping…", command=self._open_asset_tools).pack(fill="x", pady=(8, 3))
        ttk.Separator(body).pack(fill="x", pady=12)
        ttk.Label(body, text="Cleanup", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Button(body, text="Delete build output", command=self._clear_output).pack(fill="x", pady=(8, 3))
        ttk.Button(body, text="Delete persistent cache", command=self._clear_cache).pack(fill="x", pady=3)
        ttk.Button(body, text="Delete output and cache", command=self._clear_all).pack(fill="x", pady=3)
        ttk.Button(body, text="Open output folder", command=lambda: self._open_path(str(self.vars["output"].get()))).pack(fill="x", pady=(12, 3))
        ttk.Button(body, text="Open source folder", command=lambda: self._open_path(str(self.vars["source_dir"].get()))).pack(fill="x", pady=3)
        ttk.Button(body, text="Close", command=dialog.destroy).pack(anchor="e", pady=(14, 0))

    def _run_tool(self, command_name: str, key: str, option: str) -> None:
        value = str(self.vars[key].get()).strip()
        if not value:
            messagebox.showerror(APP_TITLE, f"{key.replace('_', ' ').title()} is required")
            return
        self._start_pipeline(
            [(cli_command_prefix() + [command_name, option, value], command_name.replace("-", " ").title())],
            kind="tool",
            success_message="Utility completed successfully.",
        )

    def _inspect_normalized(self) -> None:
        source = resolve_gui_path(str(self.vars["source_dir"].get()))
        normalized = source / "normalized"
        self._start_pipeline(
            [(cli_command_prefix() + ["inspect-normalized", "--normalized-dir", str(normalized)], "Inspecting normalized data")],
            kind="tool",
            success_message="Normalized bundle inspection completed.",
        )

    def _clear_output(self) -> None:
        self._delete_path(resolve_gui_path(str(self.vars["output"].get())), "build output")

    def _cache_path(self) -> Path:
        custom = str(self.vars["cache_dir"].get()).strip()
        return resolve_gui_path(custom) if custom else resolve_gui_path(str(self.vars["source_dir"].get())) / ".cwr-cache"

    def _clear_cache(self) -> None:
        self._delete_path(self._cache_path(), "persistent cache")

    def _clear_all(self) -> None:
        if not messagebox.askyesno(APP_TITLE, "Delete both the build output and persistent cache?"):
            return
        for path in (resolve_gui_path(str(self.vars["output"].get())), self._cache_path()):
            if path.exists():
                shutil.rmtree(path)
        self.footer_status_var.set("Deleted output and cache.")

    def _delete_path(self, path: Path, label: str) -> None:
        if not path.exists():
            messagebox.showinfo(APP_TITLE, f"The {label} does not exist:\n{path}")
            return
        if messagebox.askyesno(APP_TITLE, f"Delete {label}?\n\n{path}"):
            shutil.rmtree(path)
            self.footer_status_var.set(f"Deleted {label}: {path}")

    def _open_path(self, value: str) -> None:
        path = resolve_gui_path(value)
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _profile_document(self) -> dict[str, object]:
        return {
            "profile_version": PROFILE_VERSION,
            "values": {name: var.get() for name, var in self.vars.items()},
            "asset_roots": list(self.asset_roots),
        }

    def _save_profile(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("CWR GUI profile", "*.json"),))
        if not path:
            return
        Path(path).write_text(json.dumps(self._profile_document(), indent=2, sort_keys=True), encoding="utf-8")
        self.profile_path = Path(path)
        self.footer_status_var.set(f"Saved profile: {path}")

    def _load_profile(self) -> None:
        path = filedialog.askopenfilename(filetypes=(("CWR GUI profile", "*.json"), ("All files", "*.*")))
        if not path:
            return
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            values = document.get("values", {})
            if not isinstance(values, dict):
                raise ValueError("Profile values are invalid")
            for key, value in values.items():
                if key in self.vars:
                    self.vars[key].set(value)
            loaded_preset = str(values.get("appearance_preset", "")).strip()
            if loaded_preset in {LEGACY_NOGOVA_APPEARANCE_PRESET, LEGACY_RESISTANCE_APPEARANCE_PRESET}:
                # Historical generic Resistance/Nogova presets now map to the
                # explicitly named leaf family.
                self.vars["appearance_preset"].set(RESISTANCE_APPEARANCE_PRESET)
            if "bus_stop_signs" not in values and "bus_stops" in values:
                self.vars["bus_stop_signs"].set(values["bus_stops"])
            if "appearance_preset" not in values:
                ground = str(values.get("ground_textures", "nogova"))
                forest = str(values.get("forest_profile", "everon"))
                if ground == "nogova" and forest == "everon":
                    self.vars["appearance_preset"].set(RECOMMENDED_APPEARANCE_PRESET)
                elif ground == "malden" and forest == "malden":
                    self.vars["appearance_preset"].set("Malden classic")
                elif ground == "everon" and forest == "everon":
                    self.vars["appearance_preset"].set("Everon classic")
                elif ground == "desert" and forest == "everon":
                    self.vars["appearance_preset"].set("Desert ground textures")
                elif ground == "generated" and forest == "everon":
                    self.vars["appearance_preset"].set("Generated ground textures")
                else:
                    self.vars["appearance_preset"].set("Custom")
            # Older profiles used fetch_source_dir as the visible source field.
            if not str(self.vars["source_dir"].get()).strip() and "fetch_source_dir" in values:
                self.vars["source_dir"].set(values["fetch_source_dir"])
            self._sync_source_paths()
            if bool(self.vars.get("osm_asset_mapping_enabled").get() if "osm_asset_mapping_enabled" in self.vars else False):
                self._mapping_settings_changed()
            self.asset_roots = [str(item) for item in document.get("asset_roots", [])]
            self.asset_list.delete(0, "end")
            for root in self.asset_roots:
                self.asset_list.insert("end", root)
            self.profile_path = Path(path)
            self.footer_status_var.set(f"Loaded profile: {path}")
            self._show_step(0)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror(APP_TITLE, f"Could not load profile:\n{exc}")

    def _on_close(self) -> None:
        if (self.process is not None or self._pipeline_active) and not messagebox.askyesno(APP_TITLE, "A process is still running. Stop it and exit?"):
            return
        self._persist_osm_mapping_state()
        self._stop_process()
        self.destroy()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == FROZEN_CLI_MARKER:
        # PyInstaller sets sys.executable to this GUI executable. Internal GUI
        # jobs come back through this marker so they run the CLI rather than
        # constructing another WorldgenGui instance.
        from .cli import main as cli_main
        return cli_main(args[1:])
    if bool(getattr(sys, "frozen", False)):
        # One-file executables may be started by Explorer, shortcuts or launchers
        # with an arbitrary working directory. Make every remaining relative GUI
        # operation consistently live beside the actual EXE, not beside whatever
        # directory Windows happened to assign to the process.
        os.chdir(application_base_dir())
    try:
        app = WorldgenGui()
    except tk.TclError as exc:
        print(f"error: could not start GUI: {exc}", file=sys.stderr)
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
