# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePath
from itertools import chain
import hashlib
import json
import math
import shutil
from typing import Callable, Sequence

from PIL import Image

from .cache import (
    cache_key, file_snapshot, float_sequence_sha256, int_sequence_sha256,
    load_or_create_pickle, restore_bundle, store_bundle, streaming_hash,
)
from .images import HeightmapLoadResult, load_heightmap, load_material_mask
from .model import ConstraintPlayabilitySpec, HeightmapSpec, OsmSpec, PlayabilitySpec, WorldObject, WorldSpec
from .paa import inspect_paa, write_rgb_dxt1_paa, write_solid_dxt1_paa
from .pbo import PboPackResult, pack_directory, pack_directory_cached, read_pbo
from ._version import GENERATOR_VERSION
from .output_ownership import prepare_output_directory, record_build_ownership
from .assets import canonical_asset_path, scan_assets, write_asset_catalogue
from .asset_mapping import (
    collect_osm_asset_requirements,
    default_osm_asset_mapping,
    load_osm_asset_mapping,
)
from .progress import report_progress
from .procedural_buildings import BuildingGenerationResult, ProceduralBuildingLibrary
from .procedural_infrastructure import InfrastructureAssetResult, ProceduralInfrastructureLibrary, _texture_file_stem
from .procedural_forests import (
    ForestClusterAssetResult,
    ProceduralForestClusterLibrary,
    is_generated_cluster_model,
)
from .semantic_features import (
    GRAVE_MODEL_GROUNDING_PROFILES, ProceduralSiteLibrary, SemanticGenerationResult,
    SiteAssetResult, generate_semantic_objects,
)
from .png import write_rgb_png
from .templates import (
    WORLD_INTRO_NAME,
    render_config,
    render_mission,
    render_world_intro_mission,
    render_world_intro_script,
    validate_cwa_config,
)
from .terrain import (
    DEFAULT_MATERIALS,
    OSM_MATERIALS,
    SpawnPoint,
    calculate_slopes,
    choose_spawn,
    classify_materials,
    ground_texture_path,
    material_counts,
    material_colour_for_profile,
)
from .wrp import inspect_rvw4, quantize_elevations, quantize_height, write_rvw4
from .surface_pass import (
    MILESTONE9_MATERIALS,
    SurfacePassReport,
    build_surface_pass,
    external_surface_texture_paths,
    render_building_source_reference,
    render_overview_map,
    surface_texture_wire_paths,
    write_surface_textures,
    write_world_icon,
)
from .osm import (
    BuildingPlacementPlan,
    BboxProjection,
    IterativeGroundingReport,
    ObjectGenerationResult,
    OsmDataset,
    OsmRaster,
    OSM_INDIVIDUAL_TREE_MODELS,
    NOGOVA_LEAF_INDIVIDUAL_TREE_MODELS,
    NOGOVA_PINE_INDIVIDUAL_TREE_MODELS,
    STOCK_STONE_MODELS,
    STOCK_FARMLAND_FENCE_MODELS,
    STOCK_SETTLEMENT_DETAIL_MODELS,
    apply_water_elevations,
    attribution_text,
    forest_block_intersects_road_corridors,
    generate_world_objects,
    plan_building_placements,
    project_road_corridors,
    refine_iterative_grounding_terrain,
    load_osm_json,
    overlay_materials,
    parse_overpass_json,
    plan_iterative_grounding_objects,
    raster_counts,
    rasterize_osm,
    write_geography_preview,
    write_meadow_grass_placement_preview,
)
from .playability import (
    RoadFitReport,
    TerrainGradeReport,
    TransitionReport,
    fit_road_objects,
    grade_terrain,
    road_model_variant_paths,
    improve_transitions,
    town_locations,
)


@dataclass(frozen=True, slots=True)
class BuildResult:
    output_dir: Path
    source_dir: Path
    wrp_path: Path
    texture_paths: tuple[Path, ...]
    pbo_path: Path
    mission_path: Path
    intro_mission_path: Path
    intro_script_path: Path
    preview_path: Path
    height_preview_path: Path | None
    material_preview_path: Path | None
    manifest_path: Path
    report_path: Path
    osm_preview_path: Path | None = None
    meadow_grass_preview_path: Path | None = None
    osm_source_path: Path | None = None
    osm_query_path: Path | None = None
    attribution_path: Path | None = None
    asset_catalogue_path: Path | None = None
    road_report_path: Path | None = None
    grading_report_path: Path | None = None
    reproducibility_path: Path | None = None
    source_manifest_path: Path | None = None
    source_validation_path: Path | None = None
    normalized_dir: Path | None = None
    solver_heightmap_path: Path | None = None
    building_catalogue_path: Path | None = None
    forest_cluster_catalogue_path: Path | None = None
    infrastructure_catalogue_path: Path | None = None
    surface_report_path: Path | None = None
    building_source_reference_path: Path | None = None
    overview_map_path: Path | None = None
    overview_paa_path: Path | None = None
    world_icon_path: Path | None = None
    cache_report_path: Path | None = None

    @property
    def texture_path(self) -> Path:
        """Milestone 1 compatibility alias for its single texture."""
        return self.texture_paths[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _forest_proxy_profile(spec: object) -> str:
    model = str(getattr(spec, "forest_tree_model", "")).casefold()
    if model.startswith(r"o\tree\les_nw_jehl_"):
        return "nogova_pine"
    if model.startswith(r"o\tree\les_nw_"):
        return "nogova_leaf"
    return "everon"


def _generate_milestone1_elevations(spec: WorldSpec) -> list[float]:
    elevations: list[float] = []
    coast_end = spec.sea_border_cells + spec.shore_cells
    for z in range(spec.cells):
        for x in range(spec.cells):
            edge_distance = min(x, z, spec.cells - 1 - x, spec.cells - 1 - z)
            if edge_distance < spec.sea_border_cells:
                value = spec.sea_height
            elif edge_distance < coast_end:
                fraction = (edge_distance - spec.sea_border_cells + 1) / spec.shore_cells
                fraction = max(0.0, min(1.0, fraction))
                fraction = fraction * fraction * (3.0 - 2.0 * fraction)
                value = spec.sea_height + (spec.land_height - spec.sea_height) * fraction
            else:
                value = spec.land_height
            elevations.append(value)
    return elevations


def _write_milestone1_preview(path: Path, spec: WorldSpec, elevations: Sequence[float]) -> None:
    pixels = bytearray(spec.cells * spec.cells * 3)
    for index, elevation in enumerate(elevations):
        if elevation < 0:
            colour = (45, 95, 150)
        elif elevation < spec.land_height * 0.7:
            colour = (183, 169, 112)
        else:
            colour = spec.texture_colour
        offset = index * 3
        pixels[offset : offset + 3] = bytes(colour)
    write_rgb_png(path, spec.cells, spec.cells, bytes(pixels))


def _normalise_byte(value: float, minimum: float, maximum: float) -> int:
    if maximum <= minimum:
        return 127
    return max(0, min(255, int(round((value - minimum) * 255.0 / (maximum - minimum)))))


def _write_height_preview(path: Path, width: int, height: int, elevations: Sequence[float]) -> None:
    minimum = min(elevations)
    maximum = max(elevations)
    pixels = bytearray(width * height * 3)
    for index, elevation in enumerate(elevations):
        value = _normalise_byte(elevation, minimum, maximum)
        pixels[index * 3 : index * 3 + 3] = bytes((value, value, value))
    write_rgb_png(path, width, height, bytes(pixels))


def _write_material_preview(
    path: Path,
    width: int,
    height: int,
    indices: Sequence[int],
    materials=DEFAULT_MATERIALS,
) -> None:
    pixels = bytearray(width * height * 3)
    for index, material_index in enumerate(indices):
        colour = materials[material_index].colour
        pixels[index * 3 : index * 3 + 3] = bytes(colour)
    write_rgb_png(path, width, height, bytes(pixels))


def _write_composite_preview(
    path: Path,
    width: int,
    height: int,
    elevations: Sequence[float],
    slopes: Sequence[float],
    material_indices: Sequence[int],
    materials=DEFAULT_MATERIALS,
) -> None:
    minimum = min(elevations)
    maximum = max(elevations)
    pixels = bytearray(width * height * 3)
    for index, (elevation, slope, material_index) in enumerate(zip(elevations, slopes, material_indices)):
        base = materials[material_index].colour
        height_factor = _normalise_byte(elevation, minimum, maximum) / 255.0
        shade = max(0.55, min(1.15, 0.72 + height_factor * 0.33 - min(slope, 50.0) / 250.0))
        colour = tuple(max(0, min(255, int(round(channel * shade)))) for channel in base)
        pixels[index * 3 : index * 3 + 3] = bytes(colour)
    write_rgb_png(path, width, height, bytes(pixels))




def _flip_png_vertical(path: Path) -> None:
    with Image.open(path) as image:
        flipped = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        flipped.save(path, format="PNG", optimize=False)


def _validate_config(result: BuildResult, name: str, *, icon_filename: str = "g.paa") -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
    try:
        validate_cwa_config(config)
        config_syntax_ok = True
        config_syntax_detail = "no bare class declarations; braces balanced"
    except ValueError as exc:
        config_syntax_ok = False
        config_syntax_detail = str(exc)
    checks.append(("CWA-compatible config structure", config_syntax_ok, config_syntax_detail))
    checks.append(("World inherits Intro", f"class {name}: Intro" in config, name))
    checks.append((
        "CfgPatches declares world ownership",
        f'worlds[] = {{"{name}"}};' in config,
        name,
    ))
    checks.append(("CfgPatches has no synthetic units", "units[] = {};" in config, name))
    checks.append(("World registered in CfgWorldList", f"class {name} {{}};" in config, name))
    checks.append(("Config references generated WRP", f"\\{name}\\{name}.wrp" in config, name + ".wrp"))
    checks.append(("Legacy world start time/date present", 'startTime = "12:00";' in config and 'startDate = "1/6/85";' in config, name))
    checks.append((
        "World has a menu-exit cutscene",
        f'cutscenes[] = {{"{WORLD_INTRO_NAME}"}};' in config and "cutscenes[] = {};" not in config,
        f"{WORLD_INTRO_NAME}.{name}",
    ))
    checks.append((
        "World icon references an embedded texture",
        f'icon = "\\{name}\\data\\{icon_filename}";' in config,
        rf"\{name}\data\{icon_filename}",
    ))
    return checks


def _validate_milestone1(result: BuildResult, spec: WorldSpec) -> list[str]:
    checks: list[tuple[str, bool, str]] = []
    summary = inspect_rvw4(result.wrp_path, height_scale=spec.height_scale)
    checks.append(("RVW4 signature and dimensions", summary.width == spec.cells and summary.height == spec.cells, f"{summary.width}x{summary.height}"))
    checks.append(("Playable positive centre terrain", summary.maximum_height > 0, f"max={summary.maximum_height:.3f}m"))
    checks.append(("Sea border below zero", summary.minimum_height < 0, f"min={summary.minimum_height:.3f}m"))
    checks.append(("Reserved texture slot zero is unique dummy", summary.texture_slots[0] == spec.dummy_texture_path and summary.texture_slots[1] == spec.terrain_texture_path and summary.texture_slots[0] != summary.texture_slots[1], f"0={summary.texture_slots[0]}, 1={summary.texture_slots[1]}"))
    checks.append(("Reserved texture slot zero unused", bool(summary.texture_index_counts) and summary.texture_index_counts[0] == 0, str(summary.texture_index_counts)))
    checks.append(("Self-contained texture table", set(summary.texture_paths) == {spec.dummy_texture_path, spec.terrain_texture_path}, ", ".join(summary.texture_paths)))
    checks.append(("No unverified external static objects", summary.object_count == 0, str(summary.object_count)))
    checks.append(("RVW4 object terminator present", summary.has_object_terminator, "128-byte empty SingleObject4"))

    for path in result.texture_paths:
        texture = inspect_paa(path)
        checks.append((f"Legacy-compatible DXT1 PAA {path.name}", texture.magic == 0xFF01 and texture.width == 128 and texture.height == 128 and (texture.minimum_mip_width, texture.minimum_mip_height) == (4, 4) and {"AVGC", "OFFS"}.issubset(set(texture.tags)), f"{texture.width}x{texture.height}, {texture.mipmap_count} mips, tags={texture.tags}"))
    checks.extend(_validate_config(result, spec.name))
    checks.extend(_validate_world_intro(result, spec.name))

    pbo_entries = {entry.name.casefold(): entry for entry in read_pbo(result.pbo_path)}
    checks.append(("PBO contains config.cpp", "config.cpp" in pbo_entries, str(len(pbo_entries))))
    checks.append(("PBO contains generated WRP", f"{spec.name}.wrp" in pbo_entries, str(len(pbo_entries))))
    checks.append(("PBO contains generated terrain texture", r"data\g.paa" in pbo_entries, str(len(pbo_entries))))
    checks.append(("PBO contains reserved dummy texture", r"data\d.paa" in pbo_entries, str(len(pbo_entries))))
    checks.append(("Smoke-test mission exists", result.mission_path.is_file(), result.mission_path.name))
    mission = result.mission_path.read_text(encoding="ascii")
    checks.append(("Smoke-test mission uses stock SoldierWB", 'vehicle="SoldierWB";' in mission, "SoldierWB"))
    checks.append(("Smoke-test mission explicitly requires world addon", f'addOns[]={{"{spec.name}"}};' in mission, spec.name))
    checks.append(("Smoke-test mission has explicit player leader", 'leader=1;' in mission, "leader=1"))
    checks.append(("PNG preview exists", result.preview_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), result.preview_path.name))

    lines = ["CWR World Generator - Milestone 1 validation", ""]
    failures = 0
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        failures += int(not passed)
        lines.append(f"[{status}] {label}: {detail}")
    lines.extend(
        [
            "",
            f"WRP SHA-256: {_sha256(result.wrp_path)}",
            *[f"{path.name.upper()} SHA-256: {_sha256(path)}" for path in result.texture_paths],
            f"PBO SHA-256: {_sha256(result.pbo_path)}",
            f"Failures: {failures}",
        ]
    )
    if failures:
        raise RuntimeError("generated world failed validation:\n" + "\n".join(lines))
    return lines


def _validate_world_intro(result: BuildResult, name: str) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "Menu intro mission exists",
        result.intro_mission_path.is_file(),
        str(result.intro_mission_path),
    ))
    checks.append((
        "Menu intro camera script exists",
        result.intro_script_path.is_file(),
        str(result.intro_script_path),
    ))
    if result.intro_mission_path.is_file():
        intro = result.intro_mission_path.read_text(encoding="ascii")
        checks.append(("Menu intro defines Intro section", "class Intro" in intro, result.intro_mission_path.name))
        checks.append(("Menu intro uses stock SoldierWB", 'vehicle="SoldierWB";' in intro, "SoldierWB"))
        checks.append(("Menu intro retains world addon", f'addOns[]={{"{name}"}};' in intro, name))
        checks.append(("Menu intro has a player leader", 'player="PLAYER COMMANDER";' in intro and "leader=1;" in intro, "player leader"))
    if result.intro_script_path.is_file():
        script = result.intro_script_path.read_text(encoding="ascii")
        checks.append(("Menu intro creates a camera", 'camCreate' in script and 'cameraEffect ["internal","back"]' in script, result.intro_script_path.name))
    return checks


def _maximum_quantization_error(elevations: Sequence[float], height_scale: float) -> float:
    maximum = 0.0
    for elevation in elevations:
        raw = quantize_height(elevation, height_scale)
        maximum = max(maximum, abs(elevation - raw * height_scale))
    return maximum


def _validate_milestone2(
    result: BuildResult,
    spec: HeightmapSpec,
    loaded: HeightmapLoadResult,
    elevations: Sequence[float],
    material_indices: Sequence[int],
    spawn: SpawnPoint,
) -> list[str]:
    checks: list[tuple[str, bool, str]] = []
    summary = inspect_rvw4(result.wrp_path, height_scale=spec.height_scale)
    expected_texture_paths = {spec.dummy_texture_path} | {spec.terrain_texture_path(material.code) for material in DEFAULT_MATERIALS}
    checks.append(("RVW4 signature and dimensions", summary.width == spec.cells and summary.height == spec.cells, f"{summary.width}x{summary.height}"))
    checks.append(("Heightmap elevations are finite", math.isfinite(summary.minimum_height) and math.isfinite(summary.maximum_height), f"{summary.minimum_height:.3f}..{summary.maximum_height:.3f}m"))
    checks.append(("All elevations fit RVW4 int16 storage", all(-32768 <= quantize_height(value, spec.height_scale) <= 32767 for value in elevations), f"scale={spec.height_scale}"))
    quantization_error = _maximum_quantization_error(elevations, spec.height_scale)
    checks.append(("Height quantization error bounded", quantization_error <= spec.height_scale / 2.0 + 1e-9, f"max={quantization_error:.6f}m"))
    checks.append(("Five self-contained texture-table entries", set(summary.texture_paths) == expected_texture_paths, ", ".join(sorted(set(summary.texture_paths)))))
    checks.append(("Reserved texture slot zero is unique dummy", summary.texture_slots[0] == spec.dummy_texture_path and summary.texture_slots[0] not in summary.texture_slots[1:5], f"0={summary.texture_slots[0]}"))
    checks.append(("Reserved texture slot zero unused", bool(summary.texture_index_counts) and summary.texture_index_counts[0] == 0, str(summary.texture_index_counts)))
    counts = material_counts(material_indices, len(DEFAULT_MATERIALS))
    checks.append(("Multiple terrain materials assigned", sum(count > 0 for count in counts) >= 2, str(counts)))
    checks.append(("No unverified external static objects", summary.object_count == 0, str(summary.object_count)))
    checks.append(("RVW4 object terminator present", summary.has_object_terminator, "128-byte empty SingleObject4"))
    checks.append(("Playable spawn selected", spawn.y > spec.sea_level and spawn.slope_degrees <= spec.maximum_spawn_slope_degrees, f"cell={spawn.cell_x},{spawn.cell_z} y={spawn.y:.3f} slope={spawn.slope_degrees:.3f}"))
    checks.append(("Input clipping reported", loaded.clipped_low >= 0 and loaded.clipped_high >= 0, f"low={loaded.clipped_low}, high={loaded.clipped_high}"))

    for path in result.texture_paths:
        texture = inspect_paa(path)
        checks.append((f"Legacy-compatible DXT1 PAA {path.name}", texture.magic == 0xFF01 and texture.width == 128 and texture.height == 128 and (texture.minimum_mip_width, texture.minimum_mip_height) == (4, 4) and {"AVGC", "OFFS"}.issubset(set(texture.tags)), f"{texture.width}x{texture.height}, {texture.mipmap_count} mips, tags={texture.tags}"))
    checks.extend(_validate_config(result, spec.name))
    checks.extend(_validate_world_intro(result, spec.name))

    pbo_entries = {entry.name.casefold(): entry for entry in read_pbo(result.pbo_path)}
    expected_entries = {"config.cpp", f"{spec.name}.wrp", r"data\d.paa"} | {rf"data\{material.code}.paa" for material in DEFAULT_MATERIALS}
    checks.append(("PBO contains complete world package", set(pbo_entries) == expected_entries, ", ".join(sorted(pbo_entries))))
    checks.append(("Smoke-test mission exists", result.mission_path.is_file(), result.mission_path.name))
    mission = result.mission_path.read_text(encoding="ascii")
    checks.append(("Smoke-test mission uses stock SoldierWB", 'vehicle="SoldierWB";' in mission, "SoldierWB"))
    checks.append(("Smoke-test mission explicitly requires world addon", f'addOns[]={{"{spec.name}"}};' in mission, spec.name))
    checks.append(("Smoke-test mission has explicit player leader", 'leader=1;' in mission, "leader=1"))
    for preview in (result.preview_path, result.height_preview_path, result.material_preview_path):
        assert preview is not None
        checks.append((f"PNG preview exists: {preview.name}", preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), preview.name))

    lines = ["CWR World Generator - Milestone 2 validation", ""]
    lines.extend(
        [
            f"Source heightmap: {spec.heightmap_path}",
            f"Source dimensions/mode: {loaded.source_width}x{loaded.source_height} {loaded.source_mode}",
            f"Resampled elevation range: {min(elevations):.6f}..{max(elevations):.6f} m",
            f"Material counts (water, sand, grass, rock): {counts}",
            "",
        ]
    )
    failures = 0
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        failures += int(not passed)
        lines.append(f"[{status}] {label}: {detail}")
    lines.extend(
        [
            "",
            f"WRP SHA-256: {_sha256(result.wrp_path)}",
            *[f"{path.name.upper()} SHA-256: {_sha256(path)}" for path in result.texture_paths],
            f"PBO SHA-256: {_sha256(result.pbo_path)}",
            f"Failures: {failures}",
        ]
    )
    if failures:
        raise RuntimeError("generated world failed validation:\n" + "\n".join(lines))
    return lines


def build_milestone1(output_dir: Path, spec: WorldSpec | None = None, *, clean: bool = True) -> BuildResult:
    spec = spec or WorldSpec()
    spec.validate()
    output_dir = output_dir.resolve()
    prepare_output_directory(output_dir, spec.name, clean=clean)

    source_dir = output_dir / "source" / spec.name
    wrp_path = source_dir / f"{spec.name}.wrp"
    texture_path = source_dir / "data" / "g.paa"
    dummy_texture_path = source_dir / "data" / "d.paa"
    mod_root = output_dir / "@CWR-Milestone1"
    pbo_path = mod_root / "Addons" / f"{spec.name}.pbo"
    mission_path = output_dir / "Missions" / f"test_mission.{spec.name}" / "mission.sqm"
    intro_dir = mod_root / "Anims" / f"{WORLD_INTRO_NAME}.{spec.name}"
    intro_mission_path = intro_dir / "mission.sqm"
    intro_script_path = intro_dir / "intro.sqs"
    preview_path = output_dir / "preview.png"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "validation-report.txt"
    cache_report_path = output_dir / "cache-report.json"

    source_dir.mkdir(parents=True, exist_ok=True)
    mission_path.parent.mkdir(parents=True, exist_ok=True)
    intro_dir.mkdir(parents=True, exist_ok=True)

    elevations = _generate_milestone1_elevations(spec)
    texture_indices = [1] * (spec.cells * spec.cells)
    texture_table = [spec.dummy_texture_path, spec.terrain_texture_path]

    write_solid_dxt1_paa(texture_path, colour=spec.texture_colour)
    write_solid_dxt1_paa(dummy_texture_path, colour=(255, 0, 255))
    report_progress(82, "Writing world file")
    write_rvw4(
        wrp_path,
        spec.cells,
        spec.cells,
        elevations,
        texture_indices,
        texture_table,
        (),
        height_scale=spec.height_scale,
    )
    config_text = render_config(spec, milestone=1)
    validate_cwa_config(config_text)
    (source_dir / "config.cpp").write_text(config_text, encoding="ascii", newline="\n")
    mission_path.write_text(
        render_mission(
            spec,
            spawn_x=spec.centre,
            spawn_z=spec.centre,
            milestone=1,
        ),
        encoding="ascii",
        newline="\n",
    )
    intro_mission_path.write_text(
        render_world_intro_mission(spec, spawn_x=spec.centre, spawn_z=spec.centre),
        encoding="ascii",
        newline="\n",
    )
    intro_script_path.write_text(
        render_world_intro_script(spawn_x=spec.centre, spawn_z=spec.centre),
        encoding="ascii",
        newline="\n",
    )
    _write_milestone1_preview(preview_path, spec, elevations)
    pack_directory(source_dir, pbo_path)

    result = BuildResult(
        output_dir=output_dir,
        source_dir=source_dir,
        wrp_path=wrp_path,
        texture_paths=(texture_path, dummy_texture_path),
        pbo_path=pbo_path,
        mission_path=mission_path,
        intro_mission_path=intro_mission_path,
        intro_script_path=intro_script_path,
        preview_path=preview_path,
        height_preview_path=None,
        material_preview_path=None,
        manifest_path=manifest_path,
        report_path=report_path,
    )

    manifest = {
        "schema": 1,
        "milestone": 1,
        "generator": GENERATOR_VERSION,
        "world": asdict(spec),
        "height_scale": spec.height_scale,
        "objects": [],
        "outputs": {
            "source/config.cpp": _sha256(source_dir / "config.cpp"),
            f"source/{spec.name}.wrp": _sha256(wrp_path),
            "source/data/g.paa": _sha256(texture_path),
            "source/data/d.paa": _sha256(dummy_texture_path),
            f"@CWR-Milestone1/Addons/{spec.name}.pbo": _sha256(pbo_path),
            f"Missions/test_mission.{spec.name}/mission.sqm": _sha256(mission_path),
            f"@CWR-Milestone1/Anims/{WORLD_INTRO_NAME}.{spec.name}/mission.sqm": _sha256(intro_mission_path),
            f"@CWR-Milestone1/Anims/{WORLD_INTRO_NAME}.{spec.name}/intro.sqs": _sha256(intro_script_path),
            "preview.png": _sha256(preview_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_json_dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    lines = _validate_milestone1(result, spec)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    record_build_ownership(output_dir, spec.name, manifest_path, merge=False)
    report_progress(100, "Build complete")
    return result


def build_milestone2(output_dir: Path, spec: HeightmapSpec, *, clean: bool = True) -> BuildResult:
    spec.validate()
    output_dir = output_dir.resolve()
    prepare_output_directory(output_dir, spec.name, clean=clean)

    source_dir = output_dir / "source" / spec.name
    wrp_path = source_dir / f"{spec.name}.wrp"
    material_texture_paths = tuple(source_dir / "data" / f"{material.code}.paa" for material in DEFAULT_MATERIALS)
    dummy_texture_path = source_dir / "data" / "d.paa"
    texture_paths = material_texture_paths + (dummy_texture_path,)
    mod_root = output_dir / "@CWR-Milestone2"
    pbo_path = mod_root / "Addons" / f"{spec.name}.pbo"
    mission_path = output_dir / "Missions" / f"test_mission.{spec.name}" / "mission.sqm"
    intro_dir = mod_root / "Anims" / f"{WORLD_INTRO_NAME}.{spec.name}"
    intro_mission_path = intro_dir / "mission.sqm"
    intro_script_path = intro_dir / "intro.sqs"
    preview_path = output_dir / "preview.png"
    height_preview_path = output_dir / "height-preview.png"
    material_preview_path = output_dir / "material-preview.png"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "validation-report.txt"

    source_dir.mkdir(parents=True, exist_ok=True)
    mission_path.parent.mkdir(parents=True, exist_ok=True)
    intro_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_heightmap(
        spec.heightmap_path,
        spec.cells,
        spec.cells,
        input_mode=spec.input_mode,
        elevation_minimum=spec.elevation_minimum,
        elevation_maximum=spec.elevation_maximum,
        input_minimum=spec.input_minimum,
        input_maximum=spec.input_maximum,
        flip_y=spec.flip_y,
        source_grid=spec.heightmap_grid,
    )
    elevations = loaded.elevations
    # Validate representability before any files are emitted. A half-built addon
    # is not a useful souvenir of an oversized mountain range.
    for elevation in elevations:
        quantize_height(elevation, spec.height_scale)

    slopes = calculate_slopes(elevations, spec.cells, spec.cells, spec.cell_size)
    if spec.material_mask_path is None:
        material_indices = classify_materials(
            elevations,
            slopes,
            sea_level=spec.sea_level,
            beach_height=spec.beach_height,
            rock_height=spec.rock_height,
            rock_slope_degrees=spec.rock_slope_degrees,
        )
        mask_metadata: dict[str, object] | None = None
    else:
        mask = load_material_mask(
            spec.material_mask_path,
            spec.cells,
            spec.cells,
            palette=tuple(material.colour for material in DEFAULT_MATERIALS),
            flip_y=spec.flip_y,
        )
        material_indices = mask.indices
        mask_metadata = {
            "path": str(spec.material_mask_path),
            "source_width": mask.source_width,
            "source_height": mask.source_height,
            "source_mode": mask.source_mode,
        }

    spawn = choose_spawn(
        elevations,
        slopes,
        spec.cells,
        spec.cells,
        spec.cell_size,
        sea_level=spec.sea_level,
        minimum_clearance=spec.spawn_clearance,
        maximum_slope_degrees=spec.maximum_spawn_slope_degrees,
    )

    texture_table_paths = [spec.dummy_texture_path] + [
        spec.terrain_texture_path(material.code) for material in DEFAULT_MATERIALS
    ]
    wrp_texture_indices = [index + 1 for index in material_indices]
    for path, material in zip(material_texture_paths, DEFAULT_MATERIALS):
        write_solid_dxt1_paa(path, colour=material.colour)
    write_solid_dxt1_paa(dummy_texture_path, colour=(255, 0, 255))

    write_rvw4(
        wrp_path,
        spec.cells,
        spec.cells,
        elevations,
        wrp_texture_indices,
        texture_table_paths,
        (),
        height_scale=spec.height_scale,
    )
    config_text = render_config(spec, milestone=2)
    validate_cwa_config(config_text)
    (source_dir / "config.cpp").write_text(config_text, encoding="ascii", newline="\n")
    mission_path.write_text(
        render_mission(
            spec,
            spawn_x=spawn.x,
            spawn_z=spawn.z,
            milestone=2,
        ),
        encoding="ascii",
        newline="\n",
    )
    intro_mission_path.write_text(
        render_world_intro_mission(spec, spawn_x=spawn.x, spawn_z=spawn.z),
        encoding="ascii",
        newline="\n",
    )
    intro_script_path.write_text(
        render_world_intro_script(spawn_x=spawn.x, spawn_z=spawn.z),
        encoding="ascii",
        newline="\n",
    )
    _write_composite_preview(preview_path, spec.cells, spec.cells, elevations, slopes, material_indices)
    _write_height_preview(height_preview_path, spec.cells, spec.cells, elevations)
    _write_material_preview(material_preview_path, spec.cells, spec.cells, material_indices)
    pack_directory(source_dir, pbo_path)

    result = BuildResult(
        output_dir=output_dir,
        source_dir=source_dir,
        wrp_path=wrp_path,
        texture_paths=texture_paths,
        pbo_path=pbo_path,
        mission_path=mission_path,
        intro_mission_path=intro_mission_path,
        intro_script_path=intro_script_path,
        preview_path=preview_path,
        height_preview_path=height_preview_path,
        material_preview_path=material_preview_path,
        manifest_path=manifest_path,
        report_path=report_path,
    )

    counts = material_counts(material_indices, len(DEFAULT_MATERIALS))
    spec_manifest = asdict(spec)
    spec_manifest["heightmap_path"] = str(spec.heightmap_path)
    spec_manifest["material_mask_path"] = str(spec.material_mask_path) if spec.material_mask_path else None
    manifest = {
        "schema": 2,
        "milestone": 2,
        "generator": GENERATOR_VERSION,
        "world": spec_manifest,
        "height_scale": spec.height_scale,
        "heightmap": {
            "source_width": loaded.source_width,
            "source_height": loaded.source_height,
            "source_mode": loaded.source_mode,
            "source_minimum": loaded.source_minimum,
            "source_maximum": loaded.source_maximum,
            "mapping_minimum": loaded.mapping_minimum,
            "mapping_maximum": loaded.mapping_maximum,
            "clipped_low": loaded.clipped_low,
            "clipped_high": loaded.clipped_high,
            "source_grid": loaded.source_grid,
            "runtime_grid": loaded.runtime_grid,
            "legacy_centre_to_vertex_conversion": loaded.legacy_centre_to_vertex_conversion,
            "resampled_minimum_metres": min(elevations),
            "resampled_maximum_metres": max(elevations),
        },
        "material_mask": mask_metadata,
        "materials": [
            {
                "index": index,
                "wrp_texture_index": index + 1,
                "code": material.code,
                "name": material.name,
                "colour": material.colour,
                "cells": counts[index],
                "texture_path": spec.terrain_texture_path(material.code),
            }
            for index, material in enumerate(DEFAULT_MATERIALS)
        ],
        "spawn": asdict(spawn),
        "objects": [],
        "outputs": {
            "source/config.cpp": _sha256(source_dir / "config.cpp"),
            f"source/{spec.name}.wrp": _sha256(wrp_path),
            **{f"source/data/{path.name}": _sha256(path) for path in texture_paths},
            f"@CWR-Milestone2/Addons/{spec.name}.pbo": _sha256(pbo_path),
            f"Missions/test_mission.{spec.name}/mission.sqm": _sha256(mission_path),
            f"@CWR-Milestone2/Anims/{WORLD_INTRO_NAME}.{spec.name}/mission.sqm": _sha256(intro_mission_path),
            f"@CWR-Milestone2/Anims/{WORLD_INTRO_NAME}.{spec.name}/intro.sqs": _sha256(intro_script_path),
            "preview.png": _sha256(preview_path),
            "height-preview.png": _sha256(height_preview_path),
            "material-preview.png": _sha256(material_preview_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_json_dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    lines = _validate_milestone2(result, spec, loaded, elevations, material_indices, spawn)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    record_build_ownership(output_dir, spec.name, manifest_path, merge=False)
    report_progress(100, "Build complete")
    return result


def _validate_milestone3(
    result: BuildResult,
    spec: OsmSpec,
    loaded: HeightmapLoadResult,
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    material_indices: Sequence[int],
    spawn: SpawnPoint,
    generated: ObjectGenerationResult,
) -> list[str]:
    checks: list[tuple[str, bool, str]] = []
    summary = inspect_rvw4(result.wrp_path, height_scale=spec.height_scale)
    expected_texture_paths = {spec.dummy_texture_path} | {
        spec.terrain_texture_path(material.code) for material in OSM_MATERIALS
    }
    expected_models = {
        spec.paved_road_model,
        spec.dirt_road_model,
        spec.generic_building_model,
        spec.urban_building_model,
        spec.industrial_building_model,
        spec.forest_tree_model,
        str(getattr(spec, "forest_hillside_tree_model", spec.forest_tree_model)),
        str(getattr(spec, "forest_everon_steep_model", spec.forest_tree_model)),
    }
    counts = material_counts(material_indices, len(OSM_MATERIALS))
    geography_counts = raster_counts(raster)

    checks.append((
        "RVW4 signature and dimensions",
        summary.width == spec.cells and summary.height == spec.cells,
        f"{summary.width}x{summary.height}",
    ))
    checks.append((
        "Terrain and object grounding use the WRP vertex contract",
        loaded.runtime_grid == "game-terrain-vertices" and spec.height_scale == 0.05,
        (
            f"source={loaded.source_grid}, runtime={loaded.runtime_grid}, "
            f"legacy conversion={loaded.legacy_centre_to_vertex_conversion}, "
            f"height scale={spec.height_scale:.3f}m"
        ),
    ))
    checks.append((
        "All OSM-adjusted elevations fit RVW4 int16 storage",
        all(-32768 <= quantize_height(value, spec.height_scale) <= 32767 for value in elevations),
        f"{min(elevations):.3f}..{max(elevations):.3f}m",
    ))
    checks.append((
        "Nine self-contained texture-table entries",
        set(summary.texture_paths) == expected_texture_paths,
        ", ".join(sorted(set(summary.texture_paths))),
    ))
    checks.append((
        "Reserved texture slot zero is unique dummy",
        summary.texture_slots[0] == spec.dummy_texture_path
        and summary.texture_slots[0] not in summary.texture_slots[1 : len(OSM_MATERIALS) + 1],
        f"0={summary.texture_slots[0]}",
    ))
    checks.append((
        "OSM material overlays assigned",
        sum(count > 0 for count in counts[4:]) >= 1,
        str(counts),
    ))
    checks.append((
        "OSM source parsed",
        dataset.element_count > 0,
        f"{dataset.element_count} elements from {dataset.source_generator}",
    ))
    checks.append((
        "Bounding box projected onto world",
        projection.source_width_metres > 0 and projection.source_height_metres > 0,
        f"{projection.source_width_metres:.1f}x{projection.source_height_metres:.1f}m -> {spec.world_size:.1f}m square",
    ))
    checks.append((
        "Coastline/water raster available",
        len(raster.water) == spec.cells * spec.cells,
        f"{geography_counts['water']} cells, {raster.coastline_seed_count} coastline seeds",
    ))
    checks.append((
        "Forest and land-use rasters available",
        len(raster.forest) == len(raster.farmland) == len(raster.urban) == spec.cells * spec.cells,
        f"forest={geography_counts['forest']}, farmland={geography_counts['farmland']}, urban={geography_counts['urban']}",
    ))
    checks.append((
        "Basic road raster available",
        len(raster.roads) == spec.cells * spec.cells,
        f"{geography_counts['roads']} cells",
    ))
    checks.append((
        "WRP object count matches generated geography",
        summary.object_count == len(generated.objects),
        f"wrp={summary.object_count}, generated={len(generated.objects)}",
    ))
    checks.append((
        "Static object IDs are unique and deterministic",
        summary.object_ids == tuple(range(1, summary.object_count + 1)),
        f"{summary.object_count} IDs",
    ))
    checks.append((
        "WRP object stream preserves generated priority ordering",
        summary.object_models == tuple(obj.model_path for obj in generated.objects),
        "roads -> forests/trees -> rural vegetation -> buildings -> infrastructure -> semantic sites",
    ))
    checks.append((
        "Static models are from the confirmed compatibility set",
        set(summary.object_models).issubset(expected_models),
        ", ".join(sorted(set(summary.object_models))),
    ))
    checks.append((
        "Basic road objects generated",
        generated.road_objects >= 0,
        f"{generated.road_objects}, truncated={generated.road_objects_truncated}",
    ))
    checks.append((
        "Generic building placement completed",
        generated.building_objects >= 0,
        f"{generated.building_objects}, truncated={generated.building_objects_truncated}, max support raise={generated.maximum_building_grounding_raise:.3f}m",
    ))
    checks.append((
        "Forest placement completed",
        generated.forest_objects >= 0,
        (
            f"total={generated.forest_objects}, blocks={generated.forest_block_objects}, "
            f"hillside trees={generated.forest_hillside_tree_objects}, "
            f"fallback blocks={generated.forest_hillside_fallback_blocks}, "
            f"unfilled slope blocks={generated.forest_hillside_unfilled_blocks}, "
            f"truncated={generated.forest_objects_truncated}, "
            f"max support raise={generated.maximum_forest_grounding_raise:.3f}m"
        ),
    ))
    road_corridors = project_road_corridors(dataset, projection, spec)
    hillside_model = str(getattr(spec, "forest_hillside_tree_model", "")).casefold()
    hillside_footprint = float(getattr(spec, "forest_hillside_tree_footprint", 4.0))
    everon_steep_model = str(getattr(spec, "forest_everon_steep_model", "")).casefold()
    everon_steep_footprint = float(getattr(spec, "forest_everon_steep_footprint", 35.0))
    forest_on_roads = [
        obj for obj in generated.objects
        if (
            obj.model_path.casefold() == spec.forest_tree_model.casefold()
            and forest_block_intersects_road_corridors(
                road_corridors, obj.x, obj.z, block_size=spec.forest_tree_spacing
            )
        ) or (
            hillside_model
            and obj.model_path.casefold() == hillside_model
            and forest_block_intersects_road_corridors(
                road_corridors, obj.x, obj.z, block_size=hillside_footprint
            )
        ) or (
            everon_steep_model
            and obj.model_path.casefold() == everon_steep_model
            and forest_block_intersects_road_corridors(
                road_corridors, obj.x, obj.z, block_size=everon_steep_footprint
            )
        )
    ]
    checks.append((
        "Forest objects avoid complete road corridors",
        not forest_on_roads,
        f"overlaps={len(forest_on_roads)}, rejected={generated.forest_road_rejections}, clearance={spec.forest_road_clearance:.1f}m",
    ))
    if bool(getattr(spec, "forest_hillside_fallback", False)):
        checks.append((
            "Steep forest blocks use individually grounded hillside trees",
            (
                generated.forest_hillside_fallback_blocks
                + generated.forest_hillside_unfilled_blocks
                == generated.forest_slope_rejections
            ),
            (
                f"slope blocks={generated.forest_slope_rejections}, "
                f"filled={generated.forest_hillside_fallback_blocks}, "
                f"unfilled={generated.forest_hillside_unfilled_blocks}, "
                f"trees={generated.forest_hillside_tree_objects}, "
                f"max local relief={generated.maximum_hillside_tree_relief:.3f}m"
            ),
        ))
    if str(getattr(spec, "forest_profile", "malden")).casefold() == "everon":
        checks.append((
            "Steep forest blocks use the normal/sunk triangle or reusable fallback ladder",
            (
                generated.forest_hillside_fallback_blocks
                + generated.forest_hillside_unfilled_blocks
                == generated.forest_slope_rejections
            ),
            (
                f"slope blocks={generated.forest_slope_rejections}, "
                f"triangles={generated.forest_everon_steep_objects}, "
                f"sunk triangles={generated.forest_sunk_polygon_objects}, "
                f"small clusters={generated.forest_cluster_objects}, "
                f"unfilled={generated.forest_hillside_unfilled_blocks}, "
                f"single-tree objects={generated.forest_hillside_tree_objects}"
            ),
        ))
        maximum_burial = max(
            float(getattr(spec, "forest_block_maximum_burial", 0.0)),
            (
                float(getattr(spec, "forest_everon_steep_maximum_burial", 0.0))
                + float(getattr(spec, "forest_everon_steep_maximum_relief", 0.0))
                * float(getattr(spec, "forest_polygon_sink_fraction", 0.5))
            ),
            float(getattr(spec, "forest_cluster_maximum_burial", 0.0)),
            float(getattr(spec, "forest_undergrowth_maximum_burial", 0.0)),
            float(getattr(spec, "forest_border_maximum_burial", 0.0)),
        )
        maximum_float = max(
            float(getattr(spec, "forest_block_maximum_float", 0.0)),
            float(getattr(spec, "forest_everon_steep_maximum_float", 0.0)),
            float(getattr(spec, "forest_cluster_maximum_float", 0.0)),
            float(getattr(spec, "forest_undergrowth_maximum_float", 0.0)),
            float(getattr(spec, "forest_border_maximum_float", 0.0)),
        )
        checks.append((
            "Forest terrain-fit anchoring stays within burial and floating limits",
            (
                generated.maximum_forest_burial <= maximum_burial + 1e-6
                and generated.maximum_forest_float <= maximum_float + 1e-6
            ),
            (
                f"burial={generated.maximum_forest_burial:.3f}/{maximum_burial:.3f}m, "
                f"floating={generated.maximum_forest_float:.3f}/{maximum_float:.3f}m"
            ),
        ))
        checks.append((
            "Final vegetation grounding audit has no tree/proxy violations",
            generated.vegetation_audit_violations == 0,
            (
                f"violations={generated.vegetation_audit_violations}, "
                f"tree-float={generated.vegetation_audit_maximum_tree_float:.3f}m, "
                f"bush-float={generated.vegetation_audit_maximum_bush_float:.3f}m"
            ),
        ))
    checks.append((
        "RVW4 object terminator present",
        summary.has_object_terminator,
        "128-byte empty SingleObject4",
    ))
    checks.append((
        "Playable spawn avoids water and buildings",
        spawn.y > spec.sea_level
        and not raster.water[spawn.cell_z * spec.cells + spawn.cell_x]
        and not raster.buildings[spawn.cell_z * spec.cells + spawn.cell_x],
        f"cell={spawn.cell_x},{spawn.cell_z} y={spawn.y:.3f} slope={spawn.slope_degrees:.3f}",
    ))

    for path in result.texture_paths:
        texture = inspect_paa(path)
        checks.append((
            f"Legacy-compatible DXT1 PAA {path.name}",
            texture.magic == 0xFF01
            and texture.width == 128
            and texture.height == 128
            and (texture.minimum_mip_width, texture.minimum_mip_height) == (4, 4)
            and {"AVGC", "OFFS"}.issubset(set(texture.tags)),
            f"{texture.width}x{texture.height}, {texture.mipmap_count} mips, tags={texture.tags}",
        ))
    checks.extend(_validate_config(result, spec.name))
    checks.extend(_validate_world_intro(result, spec.name))

    pbo_entries = {entry.name.casefold(): entry for entry in read_pbo(result.pbo_path)}
    expected_entries = {"config.cpp", f"{spec.name}.wrp", r"data\d.paa"} | {
        rf"data\{material.code}.paa" for material in OSM_MATERIALS
    }
    checks.append((
        "PBO contains complete Milestone 3 world package",
        set(pbo_entries) == expected_entries,
        ", ".join(sorted(pbo_entries)),
    ))
    for path, label in (
        (result.osm_source_path, "Saved OSM source JSON"),
        (result.osm_query_path, "Saved Overpass query"),
        (result.attribution_path, "OpenStreetMap attribution"),
        (result.osm_preview_path, "OSM geography preview"),
    ):
        checks.append((label, path is not None and path.is_file(), str(path)))
    mod_attribution = result.pbo_path.parent.parent / "OSM-ATTRIBUTION.txt"
    checks.append((
        "Mod folder carries OSM attribution",
        mod_attribution.is_file(),
        str(mod_attribution),
    ))
    if result.attribution_path is not None and result.attribution_path.is_file():
        attribution = result.attribution_path.read_text(encoding="utf-8")
        checks.append((
            "OSM attribution names contributors and ODbL",
            "OpenStreetMap contributors" in attribution and "ODbL" in attribution,
            result.attribution_path.name,
        ))
    for preview in (
        result.preview_path,
        result.height_preview_path,
        result.material_preview_path,
        result.osm_preview_path,
    ):
        assert preview is not None
        checks.append((
            f"PNG preview exists: {preview.name}",
            preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"),
            preview.name,
        ))

    lines = ["CWR World Generator - Milestone 3 validation", ""]
    lines.extend(
        [
            f"Source heightmap: {spec.heightmap_path}",
            f"Source dimensions/mode: {loaded.source_width}x{loaded.source_height} {loaded.source_mode}",
            f"OSM bbox: {spec.bbox}",
            f"OSM elements: {dataset.element_count}",
            f"Geography cells: {geography_counts}",
            f"Material counts: {counts}",
            f"Objects (road, building, forest): {(generated.road_objects, generated.building_objects, generated.forest_objects)}",
            "",
        ]
    )
    failures = 0
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        failures += int(not passed)
        lines.append(f"[{status}] {label}: {detail}")
    lines.extend(
        [
            "",
            f"WRP SHA-256: {_sha256(result.wrp_path)}",
            *[f"{path.name.upper()} SHA-256: {_sha256(path)}" for path in result.texture_paths],
            f"PBO SHA-256: {_sha256(result.pbo_path)}",
            f"OSM JSON SHA-256: {_sha256(result.osm_source_path) if result.osm_source_path else 'missing'}",
            f"Failures: {failures}",
        ]
    )
    if failures:
        raise RuntimeError("generated world failed validation:\n" + "\n".join(lines))
    return lines


def build_milestone3(output_dir: Path, spec: OsmSpec, *, clean: bool = True) -> BuildResult:
    spec.validate()
    output_dir = output_dir.resolve()
    prepare_output_directory(output_dir, spec.name, clean=clean)

    source_dir = output_dir / "source" / spec.name
    wrp_path = source_dir / f"{spec.name}.wrp"
    material_texture_paths = tuple(
        source_dir / "data" / f"{material.code}.paa" for material in OSM_MATERIALS
    )
    dummy_texture_path = source_dir / "data" / "d.paa"
    texture_paths = material_texture_paths + (dummy_texture_path,)
    mod_root = output_dir / "@CWR-Milestone3"
    pbo_path = mod_root / "Addons" / f"{spec.name}.pbo"
    mission_path = output_dir / "Missions" / f"test_mission.{spec.name}" / "mission.sqm"
    intro_dir = mod_root / "Anims" / f"{WORLD_INTRO_NAME}.{spec.name}"
    intro_mission_path = intro_dir / "mission.sqm"
    intro_script_path = intro_dir / "intro.sqs"
    preview_path = output_dir / "preview.png"
    height_preview_path = output_dir / "height-preview.png"
    material_preview_path = output_dir / "material-preview.png"
    osm_preview_path = output_dir / "osm-geography-preview.png"
    osm_source_path = output_dir / "osm-source.json"
    osm_query_path = output_dir / "overpass-query.txt"
    attribution_path = output_dir / "OSM-ATTRIBUTION.txt"
    mod_attribution_path = mod_root / "OSM-ATTRIBUTION.txt"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "validation-report.txt"

    source_dir.mkdir(parents=True, exist_ok=True)
    mission_path.parent.mkdir(parents=True, exist_ok=True)
    intro_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_heightmap(
        spec.heightmap_path,
        spec.cells,
        spec.cells,
        input_mode=spec.input_mode,
        elevation_minimum=spec.elevation_minimum,
        elevation_maximum=spec.elevation_maximum,
        input_minimum=spec.input_minimum,
        input_maximum=spec.input_maximum,
        flip_y=spec.flip_y,
        source_grid=spec.heightmap_grid,
    )
    report_progress(10, "Loading OpenStreetMap data")
    osm_bytes, query_text = load_osm_json(spec)
    dataset = parse_overpass_json(osm_bytes)
    projection = BboxProjection.create(spec.bbox, spec.world_size)
    raster = rasterize_osm(
        dataset,
        projection,
        cells=spec.cells,
        include_minor_roads=spec.include_minor_roads,
    )
    elevations = apply_water_elevations(
        loaded.elevations,
        raster,
        sea_level=spec.sea_level,
        water_depth=spec.water_depth,
        beach_height=spec.beach_height,
        blend_cells=spec.coastline_blend_cells,
        cell_size=spec.cell_size,
        maximum_shore_slope_percent=float(getattr(spec, "lake_shore_maximum_slope_percent", 8.0)),
    )
    for elevation in elevations:
        quantize_height(elevation, spec.height_scale)

    slopes = calculate_slopes(elevations, spec.cells, spec.cells, spec.cell_size)
    if spec.material_mask_path is None:
        base_material_indices = classify_materials(
            elevations,
            slopes,
            sea_level=spec.sea_level,
            beach_height=spec.beach_height,
            rock_height=spec.rock_height,
            rock_slope_degrees=spec.rock_slope_degrees,
        )
        mask_metadata: dict[str, object] | None = None
    else:
        mask = load_material_mask(
            spec.material_mask_path,
            spec.cells,
            spec.cells,
            palette=tuple(material.colour for material in DEFAULT_MATERIALS),
            flip_y=spec.flip_y,
        )
        base_material_indices = mask.indices
        mask_metadata = {
            "path": str(spec.material_mask_path),
            "source_width": mask.source_width,
            "source_height": mask.source_height,
            "source_mode": mask.source_mode,
        }
    material_indices = overlay_materials(base_material_indices, raster)
    excluded = tuple(water or building for water, building in zip(raster.water, raster.buildings))
    spawn = choose_spawn(
        elevations,
        slopes,
        spec.cells,
        spec.cells,
        spec.cell_size,
        sea_level=spec.sea_level,
        minimum_clearance=spec.spawn_clearance,
        maximum_slope_degrees=spec.maximum_spawn_slope_degrees,
        excluded=excluded,
    )
    generated = generate_world_objects(dataset, projection, raster, elevations, spec)

    texture_table_paths = [spec.dummy_texture_path] + [
        spec.terrain_texture_path(material.code) for material in OSM_MATERIALS
    ]
    wrp_texture_indices = [index + 1 for index in material_indices]
    for path, material in zip(material_texture_paths, OSM_MATERIALS):
        write_solid_dxt1_paa(path, colour=material.colour)
    write_solid_dxt1_paa(dummy_texture_path, colour=(255, 0, 255))
    write_rvw4(
        wrp_path,
        spec.cells,
        spec.cells,
        elevations,
        wrp_texture_indices,
        texture_table_paths,
        generated.objects,
        height_scale=spec.height_scale,
    )

    config_text = render_config(spec, milestone=3)
    validate_cwa_config(config_text)
    (source_dir / "config.cpp").write_text(config_text, encoding="ascii", newline="\n")
    mission_path.write_text(
        render_mission(spec, spawn_x=spawn.x, spawn_z=spawn.z, milestone=3),
        encoding="ascii",
        newline="\n",
    )
    intro_mission_path.write_text(
        render_world_intro_mission(spec, spawn_x=spawn.x, spawn_z=spawn.z),
        encoding="ascii",
        newline="\n",
    )
    intro_script_path.write_text(
        render_world_intro_script(spawn_x=spawn.x, spawn_z=spawn.z),
        encoding="ascii",
        newline="\n",
    )

    _write_composite_preview(
        preview_path,
        spec.cells,
        spec.cells,
        elevations,
        slopes,
        material_indices,
        OSM_MATERIALS,
    )
    _write_height_preview(height_preview_path, spec.cells, spec.cells, elevations)
    _write_material_preview(
        material_preview_path,
        spec.cells,
        spec.cells,
        material_indices,
        OSM_MATERIALS,
    )
    # The terrain arrays use WRP south-to-north row order. Present all Milestone
    # 3 previews north-up to match ordinary OSM maps.
    _flip_png_vertical(preview_path)
    _flip_png_vertical(height_preview_path)
    _flip_png_vertical(material_preview_path)
    write_geography_preview(osm_preview_path, raster)

    osm_source_path.write_bytes(osm_bytes)
    osm_query_path.write_text(query_text, encoding="utf-8", newline="\n")
    attribution = attribution_text(spec)
    attribution_path.write_text(attribution, encoding="utf-8", newline="\n")
    mod_attribution_path.write_text(attribution, encoding="utf-8", newline="\n")
    pack_directory(source_dir, pbo_path)

    result = BuildResult(
        output_dir=output_dir,
        source_dir=source_dir,
        wrp_path=wrp_path,
        texture_paths=texture_paths,
        pbo_path=pbo_path,
        mission_path=mission_path,
        intro_mission_path=intro_mission_path,
        intro_script_path=intro_script_path,
        preview_path=preview_path,
        height_preview_path=height_preview_path,
        material_preview_path=material_preview_path,
        manifest_path=manifest_path,
        report_path=report_path,
        osm_preview_path=osm_preview_path,
        osm_source_path=osm_source_path,
        osm_query_path=osm_query_path,
        attribution_path=attribution_path,
    )

    counts = material_counts(material_indices, len(OSM_MATERIALS))
    spec_manifest = asdict(spec)
    spec_manifest["heightmap_path"] = str(spec.heightmap_path)
    spec_manifest["material_mask_path"] = str(spec.material_mask_path) if spec.material_mask_path else None
    spec_manifest["osm_json_path"] = str(spec.osm_json_path) if spec.osm_json_path else None
    manifest = {
        "schema": 3,
        "milestone": 3,
        "generator": GENERATOR_VERSION,
        "world": spec_manifest,
        "height_scale": spec.height_scale,
        "heightmap": {
            "source_width": loaded.source_width,
            "source_height": loaded.source_height,
            "source_mode": loaded.source_mode,
            "source_minimum": loaded.source_minimum,
            "source_maximum": loaded.source_maximum,
            "mapping_minimum": loaded.mapping_minimum,
            "mapping_maximum": loaded.mapping_maximum,
            "clipped_low": loaded.clipped_low,
            "clipped_high": loaded.clipped_high,
            "source_grid": loaded.source_grid,
            "runtime_grid": loaded.runtime_grid,
            "legacy_centre_to_vertex_conversion": loaded.legacy_centre_to_vertex_conversion,
            "resampled_minimum_metres": min(elevations),
            "resampled_maximum_metres": max(elevations),
        },
        "material_mask": mask_metadata,
        "osm": {
            "bbox": spec.bbox,
            "source_generator": dataset.source_generator,
            "element_count": dataset.element_count,
            "source_width_metres": projection.source_width_metres,
            "source_height_metres": projection.source_height_metres,
            "scale_x": projection.scale_x,
            "scale_z": projection.scale_z,
            "feature_counts": {
                "coastlines": len(dataset.coastlines),
                "water": len(dataset.water),
                "forests": len(dataset.forests),
                "farmland": len(dataset.farmland),
                "urban": len(dataset.urban),
                "roads": len(dataset.roads),
                "building_polygons": len(dataset.building_polygons),
                "building_points": len(dataset.building_points),
            },
            "raster_cell_counts": raster_counts(raster),
            "coastline_seed_count": raster.coastline_seed_count,
        },
        "materials": [
            {
                "index": index,
                "wrp_texture_index": index + 1,
                "code": material.code,
                "name": material.name,
                "colour": material.colour,
                "cells": counts[index],
                "texture_path": spec.terrain_texture_path(material.code),
            }
            for index, material in enumerate(OSM_MATERIALS)
        ],
        "spawn": asdict(spawn),
        "objects": {
            "total": len(generated.objects),
            "roads": generated.road_objects,
            "buildings": generated.building_objects,
            "forest": generated.forest_objects,
            "forest_road_rejections": generated.forest_road_rejections,
            "forest_slope_rejections": generated.forest_slope_rejections,
            "maximum_forest_relief_metres": generated.maximum_forest_relief,
            "forest_block_objects": generated.forest_block_objects,
            "forest_hillside_tree_objects": generated.forest_hillside_tree_objects,
            "forest_single_tree_objects": generated.forest_single_tree_objects,
            "forest_hillside_fallback_blocks": generated.forest_hillside_fallback_blocks,
            "forest_hillside_unfilled_blocks": generated.forest_hillside_unfilled_blocks,
            "forest_hillside_candidate_rejections": generated.forest_hillside_candidate_rejections,
            "forest_everon_steep_objects": generated.forest_everon_steep_objects,
            "forest_sunk_polygon_objects": generated.forest_sunk_polygon_objects,
            "forest_everon_steep_rejections": generated.forest_everon_steep_rejections,
            "maximum_hillside_tree_relief_metres": generated.maximum_hillside_tree_relief,
            "grounding": {
                "building_clearance_metres": spec.building_ground_clearance,
                "forest_clearance_metres": spec.forest_ground_clearance,
                "maximum_building_raise_metres": generated.maximum_building_grounding_raise,
                "maximum_forest_raise_metres": generated.maximum_forest_grounding_raise,
            },
            "truncated": {
                "roads": generated.road_objects_truncated,
                "buildings": generated.building_objects_truncated,
                "forest": generated.forest_objects_truncated,
            },
            "models": sorted({obj.model_path for obj in generated.objects}),
        },
        "outputs": {
            "source/config.cpp": _sha256(source_dir / "config.cpp"),
            f"source/{spec.name}.wrp": _sha256(wrp_path),
            **{f"source/data/{path.name}": _sha256(path) for path in texture_paths},
            f"@CWR-Milestone3/Addons/{spec.name}.pbo": _sha256(pbo_path),
            f"Missions/test_mission.{spec.name}/mission.sqm": _sha256(mission_path),
            f"@CWR-Milestone3/Anims/{WORLD_INTRO_NAME}.{spec.name}/mission.sqm": _sha256(intro_mission_path),
            f"@CWR-Milestone3/Anims/{WORLD_INTRO_NAME}.{spec.name}/intro.sqs": _sha256(intro_script_path),
            "preview.png": _sha256(preview_path),
            "height-preview.png": _sha256(height_preview_path),
            "material-preview.png": _sha256(material_preview_path),
            "osm-geography-preview.png": _sha256(osm_preview_path),
            "osm-source.json": _sha256(osm_source_path),
            "overpass-query.txt": _sha256(osm_query_path),
            "OSM-ATTRIBUTION.txt": _sha256(attribution_path),
            "@CWR-Milestone3/OSM-ATTRIBUTION.txt": _sha256(mod_attribution_path),
        },
    }
    manifest_path.write_text(
        _json_dumps(manifest) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = _validate_milestone3(
        result,
        spec,
        loaded,
        dataset,
        projection,
        raster,
        elevations,
        material_indices,
        spawn,
        generated,
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    record_build_ownership(output_dir, spec.name, manifest_path, merge=False)
    report_progress(100, "Build complete")
    return result


def _json_default(value: object) -> object:
    """Convert supported filesystem values for deterministic JSON output."""

    if isinstance(value, PurePath):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _json_dumps(document: object, *, compact: bool = False) -> str:
    options: dict[str, object] = {
        "sort_keys": True,
        "default": _json_default,
    }
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    return json.dumps(document, **options)


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(document) + "\n", encoding="utf-8", newline="\n")


def _generation_fingerprint(
    spec: PlayabilitySpec,
    elevations: Sequence[float],
    material_indices: Sequence[int],
    objects: Sequence[object],
    towns: Sequence[object],
    *,
    heightmap_sha256: str,
    osm_sha256: str,
) -> str:
    # Do not turn a million compact objects into a million dictionaries and one
    # enormous JSON/UTF-8 buffer merely to hash them. The versioned streaming
    # encoding walks dataclasses and generators directly.
    return streaming_hash(
        "generation-fingerprint-v2",
        spec.deterministic_seed,
        heightmap_sha256,
        osm_sha256,
        spec.height_scale,
        (quantize_height(value, spec.height_scale) for value in elevations),
        material_indices,
        objects,
        towns,
    )


def _ground_texture_profile(spec: PlayabilitySpec) -> str:
    return str(getattr(spec, "ground_texture_profile", "generated"))


def _surface_pass_enabled(spec: PlayabilitySpec) -> bool:
    return bool(getattr(spec, "surface_pass_enabled", False))


def _surface_ground_mode(spec: PlayabilitySpec) -> str:
    mode = str(getattr(spec, "surface_ground_mode", "milestone8"))
    if mode not in {"milestone8", "milestone9"}:
        raise ValueError("surface ground mode must be milestone8 or milestone9")
    return mode


def _surface_ground_enabled(spec: PlayabilitySpec) -> bool:
    return _surface_pass_enabled(spec) and _surface_ground_mode(spec) == "milestone9"


def _material_definitions(spec: PlayabilitySpec):
    return MILESTONE9_MATERIALS if _surface_ground_enabled(spec) else OSM_MATERIALS


def _ground_texture_paths(spec: PlayabilitySpec) -> tuple[str, ...]:
    profile = _ground_texture_profile(spec)
    if _surface_ground_enabled(spec):
        return surface_texture_wire_paths(spec.name, profile)
    return tuple(ground_texture_path(spec.name, material.code, profile) for material in OSM_MATERIALS)


def _external_ground_texture_paths(spec: PlayabilitySpec) -> tuple[str, ...]:
    profile = _ground_texture_profile(spec)
    if _surface_ground_enabled(spec):
        return external_surface_texture_paths(profile)
    return _ground_texture_paths(spec) if profile in {"everon", "nogova"} else ()


def _world_icon_filename(spec: PlayabilitySpec) -> str:
    return "icon.paa" if _surface_pass_enabled(spec) else "g.paa"


def _trusted_legacy_asset_paths(spec: PlayabilitySpec, milestone_number: int) -> tuple[str, ...]:
    """Return assets inherited from an earlier milestone and trusted at runtime.

    Milestone 8 changes building generation and terrain textures, but deliberately
    keeps the Milestone 7 road and forest references. Those assets may be supplied
    by a game installation or mod search path outside the roots used to verify the
    new Milestone 8 assets, so they remain catalogued without becoming blockers.
    """
    if milestone_number < 8:
        return ()
    configured_roads = [spec.paved_road_model, spec.dirt_road_model]
    road_models = {
        canonical_asset_path(path)
        for configured in configured_roads
        for path in (
            road_model_variant_paths(configured, spec.road_segment_length)
            if bool(getattr(spec, "stock_road_piece_fitting", False))
            else (configured,)
        )
    }
    trusted = {
        *road_models,
        canonical_asset_path(spec.forest_tree_model),
    }
    if milestone_number >= 9:
        if str(getattr(spec, "forest_profile", "malden")).casefold() == "everon":
            trusted.add(canonical_asset_path(str(getattr(spec, "forest_everon_steep_model", ""))))
        # Road-cut forest blocks use individually checked stock trees and bushes
        # in both the Everon and Malden profiles. Keep those original game assets
        # in the same trusted-legacy bucket as the primary forest model.
        trusted.add(canonical_asset_path(str(getattr(spec, "forest_single_tree_model", r"data3d\str smrk_medium.p3d"))))
        trusted.add(canonical_asset_path(str(getattr(spec, "forest_roadside_tree_model", r"data3d\str smrk vysoky.p3d"))))
        trusted.update(
            canonical_asset_path(str(path))
            for path in (
                *getattr(spec, "forest_roadside_tree_models", ()),
                *getattr(spec, "forest_roadside_bush_models", ()),
                *getattr(spec, "steep_hill_bush_models", ()),
            )
        )
    if milestone_number >= 9 and bool(getattr(spec, "forest_hillside_fallback", False)):
        trusted.add(canonical_asset_path(str(getattr(spec, "forest_hillside_tree_model", ""))))
    if milestone_number >= 9:
        proxy_profile = _forest_proxy_profile(spec)
        mapped_tree_models = (
            NOGOVA_PINE_INDIVIDUAL_TREE_MODELS
            if proxy_profile == "nogova_pine"
            else NOGOVA_LEAF_INDIVIDUAL_TREE_MODELS
            if proxy_profile == "nogova_leaf"
            else OSM_INDIVIDUAL_TREE_MODELS
        )
        trusted.update(canonical_asset_path(path) for path in mapped_tree_models)
        trusted.update(canonical_asset_path(path) for path in STOCK_STONE_MODELS)
    if (milestone_number >= 9
            and bool(getattr(spec, "semantic_landmarks", False))
            and bool(getattr(spec, "bus_stops_enabled", False))
            and int(getattr(spec, "maximum_landmark_objects", 0)) > 0):
        trusted.add(canonical_asset_path(str(getattr(spec, "bus_stop_model", ""))))
    if (milestone_number >= 9
            and bool(getattr(spec, "semantic_landmarks", False))
            and bool(getattr(spec, "cemeteries_enabled", True))
            and int(getattr(spec, "maximum_grave_objects", 0)) > 0):
        trusted.update(
            canonical_asset_path(str(path))
            for path in getattr(spec, "grave_models", ())
        )
    if milestone_number >= 9 and bool(getattr(spec, "barriers_enabled", False)):
        trusted.update(
            canonical_asset_path(str(path))
            for path in (
                *getattr(spec, "stock_hedge_models", ()),
                *getattr(spec, "stock_wall_models", ()),
                *getattr(spec, "stock_metal_fence_models", ()),
                *STOCK_FARMLAND_FENCE_MODELS,
            )
        )
    if milestone_number >= 9 and bool(getattr(spec, "street_furniture_enabled", False)):
        trusted.update(canonical_asset_path(path) for path in STOCK_SETTLEMENT_DETAIL_MODELS)
    return tuple(sorted(path for path in trusted if path))


def _validate_milestone4(
    result: BuildResult,
    spec: PlayabilitySpec,
    loaded: HeightmapLoadResult,
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    material_indices: Sequence[int],
    spawn: SpawnPoint,
    generated: ObjectGenerationResult,
    road_fit: RoadFitReport,
    grading: TerrainGradeReport,
    transitions: TransitionReport,
    towns: Sequence[object],
    asset_scan,
    strict_asset_scan,
    osm_asset_mapping_report,
    trusted_legacy_assets: Sequence[str],
    reproducibility: dict[str, object],
    building_generation: BuildingGenerationResult | None,
    pbo_layout: dict[str, object],
    *,
    mod_directory_name: str = "@CWR-Milestone4",
    milestone_number: int = 4,
) -> list[str]:
    checks: list[tuple[str, bool, str]] = []
    summary = inspect_rvw4(result.wrp_path, height_scale=spec.height_scale)
    materials = _material_definitions(spec)
    counts = material_counts(material_indices, len(materials))
    expected_texture_paths = {spec.dummy_texture_path, *_ground_texture_paths(spec)}
    checks.append((
        "WRP terrain texture profile",
        set(summary.texture_paths) == expected_texture_paths,
        f"profile={_ground_texture_profile(spec)}, textures={summary.texture_paths}",
    ))
    checks.append((
        "RVW4 signature and dimensions",
        summary.width == spec.cells and summary.height == spec.cells,
        f"{summary.width}x{summary.height}",
    ))
    checks.append((
        "Terrain and object grounding use the WRP vertex contract",
        loaded.runtime_grid == "game-terrain-vertices" and spec.height_scale == 0.05,
        (
            f"source={loaded.source_grid}, runtime={loaded.runtime_grid}, "
            f"legacy conversion={loaded.legacy_centre_to_vertex_conversion}, "
            f"height scale={spec.height_scale:.3f}m"
        ),
    ))
    checks.append((
        "Road object budget covers the complete imported network",
        (not road_fit.truncated) or spec.max_road_objects == 0,
        (
            f"objects={len(road_fit.objects)}, limit={spec.max_road_objects}, "
            f"truncated={road_fit.truncated}"
        ),
    ))
    checks.append((
        "Road junctions use bounded deterministic stock-road hubs",
        road_fit.failed_connections == 0 and road_fit.maximum_connection_gap <= spec.road_connection_tolerance,
        (
            f"connections={road_fit.connection_count}, failed={road_fit.failed_connections}, "
            f"caps={road_fit.junction_cap_objects}, "
            f"maximum uncovered gap={road_fit.maximum_connection_gap:.3f}m"
        ),
    ))
    checks.append((
        "Road graph respects CWA's four-connection object limit",
        road_fit.road_connection_slot_risk_nodes == 0,
        (
            f"slot-risk nodes={road_fit.road_connection_slot_risk_nodes}, "
            f"suppressed degree-two caps={road_fit.suppressed_degree_two_caps}, "
            f"complex unhubbed junctions={road_fit.complex_junctions_without_caps}, "
            f"nearby hubs suppressed={road_fit.suppressed_nearby_hubs}"
        ),
    ))
    checks.append((
        "Road object chains remain connected",
        road_fit.maximum_chain_gap <= spec.road_connection_tolerance,
        f"max={road_fit.maximum_chain_gap:.3f}m, tolerance={spec.road_connection_tolerance:.3f}m",
    ))
    checks.append((
        "Road model axes do not overlap longitudinally",
        road_fit.maximum_model_overlap_metres <= 1e-4,
        f"maximum overlap={road_fit.maximum_model_overlap_metres:.3f}m",
    ))
    checks.append((
        "Terrain grading changed playable corridors",
        grading.changed_cells > 0 if (dataset.roads or dataset.building_polygons) else True,
        f"cells={grading.changed_cells}, cut={grading.maximum_cut:.3f}m, fill={grading.maximum_fill:.3f}m",
    ))
    grade_tolerance_percent = 3.0
    checks.append((
        "Road grading reaches the target or does not materially worsen maximum sampled grade",
        (
            grading.maximum_road_slope_after_percent
            <= spec.maximum_road_grade_percent + grade_tolerance_percent
            or grading.maximum_road_slope_after_percent
            <= grading.maximum_road_slope_before_percent + grade_tolerance_percent
        ),
        (
            f"{grading.maximum_road_slope_before_percent:.2f}% -> "
            f"{grading.maximum_road_slope_after_percent:.2f}% "
            f"(target={spec.maximum_road_grade_percent:.2f}%, tolerance={grade_tolerance_percent:.2f}%)"
        ),
    ))
    checks.append((
        "Building pads are no rougher after grading",
        grading.building_roughness_after <= grading.building_roughness_before + 1e-6,
        f"{grading.building_roughness_before:.3f}m -> {grading.building_roughness_after:.3f}m",
    ))
    checks.append((
        "Final building footprints fit their calculated foundations",
        generated.building_foundation_rejections == 0,
        (
            f"maximum pad relief={generated.maximum_building_pad_relief:.3f}m, "
            f"maximum foundation={generated.maximum_building_foundation_depth:.3f}m, "
            f"rejected={generated.building_foundation_rejections}, "
            f"interior fallbacks={generated.building_interior_fallbacks}, "
            f"fully submerged omitted={generated.building_fully_submerged_rejections}, "
            f"road nudges={generated.building_road_nudges}"
        ),
    ))
    checks.append((
        "Material transition accounting is valid",
        transitions.shoreline_cells >= 0 and transitions.softened_landuse_cells >= 0,
        f"shore={transitions.shoreline_cells}, landuse={transitions.softened_landuse_cells}",
    ))
    checks.append((
        "Town-name catalogue emitted",
        len(towns) == min(
            spec.town_name_limit,
            sum(
                1 for place in dataset.places
                if str(place.tags.get("name", "")).strip()
                and str(place.tags.get("name", "")).strip().casefold()
                not in {"unnamed", "unnamed isolated dwelling"}
            ),
        ),
        f"{len(towns)} named places",
    ))
    checks.append((
        "Asset catalogue written",
        result.asset_catalogue_path is not None and result.asset_catalogue_path.is_file(),
        f"assets={len(asset_scan.records)}, verified={asset_scan.verified}",
    ))
    if milestone_number >= 9:
        meadow_preview = result.meadow_grass_preview_path
        checks.append((
            "Meadow grass placement diagnostic written",
            meadow_preview is not None
            and meadow_preview.is_file()
            and meadow_preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"),
            str(meadow_preview),
        ))
    checks.append((
        "Strict asset validation satisfied",
        not spec.strict_assets or strict_asset_scan.verified,
        (
            f"missing required models={len(strict_asset_scan.missing_models)}, "
            f"dependencies={len(strict_asset_scan.missing_dependencies)}, "
            f"trusted legacy assets={len(trusted_legacy_assets)}"
        ),
    ))
    checks.append((
        "OSM objects resolve through the asset mapping",
        bool(osm_asset_mapping_report.mapping_sha256),
        (
            f"source={osm_asset_mapping_report.source}, "
            f"matched features={osm_asset_mapping_report.matched_feature_count}, "
            f"models={len(osm_asset_mapping_report.selected_models)}, "
            f"textures={len(osm_asset_mapping_report.selected_textures)}"
        ),
    ))
    checks.append((
        "Generated roads and infrastructure share the world PBO",
        bool(pbo_layout.get("verified")) and not bool(pbo_layout.get("separate_road_pbo")),
        (
            f"mode={pbo_layout.get('mode')}, "
            f"road models={len(pbo_layout.get('generated_road_models', []))}, "
            f"road textures={len(pbo_layout.get('generated_road_textures', []))}"
        ),
    ))
    if building_generation is not None:
        checks.append((
            "Procedural building catalogue written",
            result.building_catalogue_path is not None and result.building_catalogue_path.is_file(),
            f"variants={building_generation.generated_variants}, placements={building_generation.placements}",
        ))
        checks.append((
            "Procedural building models reuse assets",
            building_generation.generated_variants <= building_generation.placements,
            f"reused={building_generation.reused_placements}, ratio={building_generation.reuse_ratio:.3f}",
        ))
        checks.append((
            "Procedural building models contain required LODs",
            all(asset.lod_count >= 3 and asset.face_count > 0 for asset in building_generation.model_assets),
            f"models={len(building_generation.model_assets)}",
        ))
    if bool(reproducibility.get("verification_enabled", False)):
        checks.append((
            "Deterministic regeneration matched",
            all(bool(value) for key, value in reproducibility.items() if key.endswith("_match")),
            str({key: value for key, value in reproducibility.items() if key.endswith("_match")}),
        ))
    else:
        checks.append((
            "Deterministic regeneration is opt-in",
            reproducibility.get("verification_status") == "skipped",
            "skipped; use --verify-regeneration for release verification",
        ))
    road_corridors = project_road_corridors(dataset, projection, spec)
    hillside_model = str(getattr(spec, "forest_hillside_tree_model", "")).casefold()
    hillside_footprint = float(getattr(spec, "forest_hillside_tree_footprint", 4.0))
    everon_steep_model = str(getattr(spec, "forest_everon_steep_model", "")).casefold()
    everon_steep_footprint = float(getattr(spec, "forest_everon_steep_footprint", 35.0))
    forest_on_roads = [
        obj for obj in generated.objects
        if (
            obj.model_path.casefold() == spec.forest_tree_model.casefold()
            and forest_block_intersects_road_corridors(
                road_corridors, obj.x, obj.z, block_size=spec.forest_tree_spacing
            )
        ) or (
            hillside_model
            and obj.model_path.casefold() == hillside_model
            and forest_block_intersects_road_corridors(
                road_corridors, obj.x, obj.z, block_size=hillside_footprint
            )
        ) or (
            everon_steep_model
            and obj.model_path.casefold() == everon_steep_model
            and forest_block_intersects_road_corridors(
                road_corridors, obj.x, obj.z, block_size=everon_steep_footprint
            )
        )
    ]
    checks.append((
        "Forest objects avoid complete road corridors",
        not forest_on_roads,
        f"overlaps={len(forest_on_roads)}, rejected={generated.forest_road_rejections}, clearance={spec.forest_road_clearance:.1f}m",
    ))
    if bool(getattr(spec, "forest_hillside_fallback", False)):
        checks.append((
            "Steep forest blocks use individually grounded hillside trees",
            (
                generated.forest_hillside_fallback_blocks
                + generated.forest_hillside_unfilled_blocks
                == generated.forest_slope_rejections
            ),
            (
                f"slope blocks={generated.forest_slope_rejections}, "
                f"filled={generated.forest_hillside_fallback_blocks}, "
                f"unfilled={generated.forest_hillside_unfilled_blocks}, "
                f"trees={generated.forest_hillside_tree_objects}, "
                f"max local relief={generated.maximum_hillside_tree_relief:.3f}m"
            ),
        ))
    if str(getattr(spec, "forest_profile", "malden")).casefold() == "everon":
        checks.append((
            "Steep forest blocks use the normal/sunk triangle or reusable fallback ladder",
            (
                generated.forest_hillside_fallback_blocks
                + generated.forest_hillside_unfilled_blocks
                == generated.forest_slope_rejections
            ),
            (
                f"slope blocks={generated.forest_slope_rejections}, "
                f"triangles={generated.forest_everon_steep_objects}, "
                f"sunk triangles={generated.forest_sunk_polygon_objects}, "
                f"small clusters={generated.forest_cluster_objects}, "
                f"unfilled={generated.forest_hillside_unfilled_blocks}, "
                f"single-tree objects={generated.forest_hillside_tree_objects}"
            ),
        ))
        maximum_burial = max(
            float(getattr(spec, "forest_block_maximum_burial", 0.0)),
            (
                float(getattr(spec, "forest_everon_steep_maximum_burial", 0.0))
                + float(getattr(spec, "forest_everon_steep_maximum_relief", 0.0))
                * float(getattr(spec, "forest_polygon_sink_fraction", 0.5))
            ),
            float(getattr(spec, "forest_cluster_maximum_burial", 0.0)),
            float(getattr(spec, "forest_undergrowth_maximum_burial", 0.0)),
            float(getattr(spec, "forest_border_maximum_burial", 0.0)),
        )
        maximum_float = max(
            float(getattr(spec, "forest_block_maximum_float", 0.0)),
            float(getattr(spec, "forest_everon_steep_maximum_float", 0.0)),
            float(getattr(spec, "forest_cluster_maximum_float", 0.0)),
            float(getattr(spec, "forest_undergrowth_maximum_float", 0.0)),
            float(getattr(spec, "forest_border_maximum_float", 0.0)),
        )
        checks.append((
            "Forest terrain-fit anchoring stays within burial and floating limits",
            (
                generated.maximum_forest_burial <= maximum_burial + 1e-6
                and generated.maximum_forest_float <= maximum_float + 1e-6
            ),
            (
                f"burial={generated.maximum_forest_burial:.3f}/{maximum_burial:.3f}m, "
                f"floating={generated.maximum_forest_float:.3f}/{maximum_float:.3f}m"
            ),
        ))
        checks.append((
            "Final vegetation grounding audit has no tree/proxy violations",
            generated.vegetation_audit_violations == 0,
            (
                f"violations={generated.vegetation_audit_violations}, "
                f"tree-float={generated.vegetation_audit_maximum_tree_float:.3f}m, "
                f"bush-float={generated.vegetation_audit_maximum_bush_float:.3f}m"
            ),
        ))
    checks.append((
        "All final elevations fit RVW4 int16 storage",
        all(-32768 <= quantize_height(value, spec.height_scale) <= 32767 for value in elevations),
        f"{min(elevations):.3f}..{max(elevations):.3f}m",
    ))
    checks.append((
        "All generated objects have stable unique IDs",
        summary.object_count == len(generated.objects)
        and summary.object_ids == tuple(range(1, summary.object_count + 1)),
        f"{summary.object_count} objects",
    ))
    checks.append((
        "WRP object stream preserves tree-before-building priority",
        summary.object_models == tuple(obj.model_path for obj in generated.objects),
        "roads -> forests/trees -> rural vegetation -> buildings -> infrastructure -> semantic sites",
    ))
    checks.append((
        "OSM playability materials assigned",
        sum(count > 0 for count in counts[4:]) >= 1,
        str(counts),
    ))
    checks.extend(_validate_config(result, spec.name, icon_filename=_world_icon_filename(spec)))
    config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
    checks.append((
        "CfgWorlds Names contains imported places",
        ("class Names" in config) == bool(towns),
        f"{len(towns)} names",
    ))
    checks.extend(_validate_world_intro(result, spec.name))
    checks.append(("Smoke-test mission exists", result.mission_path.is_file(), result.mission_path.name))
    checks.append(("OSM attribution accompanies mod", (result.output_dir / mod_directory_name / "OSM-ATTRIBUTION.txt").is_file(), "ODbL attribution"))

    lines = [f"CWR World Generator - Milestone {milestone_number} validation", ""]
    failures = 0
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        failures += int(not passed)
        lines.append(f"[{status}] {label}: {detail}")
    lines.extend([
        "",
        f"Generation fingerprint: {reproducibility['generation_fingerprint']}",
        f"WRP SHA-256: {_sha256(result.wrp_path)}",
        f"PBO SHA-256: {_sha256(result.pbo_path)}",
        f"Failures: {failures}",
    ])
    if failures:
        raise RuntimeError(f"generated Milestone {milestone_number} world failed validation:\n" + "\n".join(lines))
    return lines


def _write_float_heightmap(path: Path, elevations: Sequence[float], cells: int) -> None:
    if len(elevations) != cells * cells:
        raise ValueError("solved heightmap has the wrong size")
    # Internal/WRP rows are south-to-north; exported TIFF is conventional north-up.
    north_up = [
        float(elevations[z * cells + x])
        for z in range(cells - 1, -1, -1)
        for x in range(cells)
    ]
    image = Image.new("F", (cells, cells))
    image.putdata(north_up)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="TIFF")




_STAGE_CACHE_SCHEMA = 3


def _cache_settings(spec: object) -> tuple[Path | None, bool, bool]:
    return (
        getattr(spec, "cache_dir", None),
        bool(getattr(spec, "cache_enabled", True)),
        bool(getattr(spec, "cache_refresh", False)),
    )


def _dataset_identity(dataset: OsmDataset, osm_bytes: bytes) -> str:
    return dataset.normalized_fingerprint or hashlib.sha256(osm_bytes).hexdigest()


def _spec_stage_payload(spec: object, *, omit: tuple[str, ...] = ()) -> dict[str, object]:
    document = asdict(spec)  # type: ignore[arg-type]
    for field in ("cache_dir", "cache_enabled", "cache_refresh", *omit):
        document.pop(field, None)
    return document


def _spec_fields(spec: object, names: tuple[str, ...]) -> dict[str, object]:
    return {name: getattr(spec, name, None) for name in names}


_TERRAIN_CACHE_FIELDS = (
    "cells", "cell_size", "bbox", "sea_level", "water_depth", "beach_height",
    "coastline_blend_cells", "include_minor_roads", "maximum_road_grade_percent",
    "major_road_grade_percent", "road_grade_radius", "building_grade_radius",
    "maximum_grade_adjustment", "shoreline_transition_cells", "lake_shore_smoothing_cells",
    "lake_shore_maximum_slope_percent", "building_pad_margin",
    "stream_channel_depth", "river_channel_depth", "watercourse_minimum_gradient_percent",
    "natural_smoothing_strength", "solver_iterations", "world_edge_blend_cells",
)

_SURFACE_CACHE_FIELDS = (
    "cells", "cell_size", "sea_level", "beach_height", "rock_height",
    "rock_slope_degrees", "flip_y", "include_minor_roads", "transition_cells",
    "deterministic_seed", "surface_pass_enabled", "surface_ground_mode",
    "surface_shoreline_wet_cells", "surface_shoreline_sand_cells",
    "surface_transition_cells", "surface_forest_edge_cells",
    "surface_farmland_strip_cells", "surface_road_shoulder_metres",
    "surface_dirt_blend_metres", "surface_steep_slope_degrees",
    "surface_colour_reference_strength",
)

_PLACEMENT_CACHE_FIELDS = (
    "name", "cells", "cell_size", "bbox", "include_minor_roads",
    "max_buildings", "building_minimum_area", "building_ground_clearance",
    "point_building_footprint", "generic_building_model", "urban_building_model",
    "industrial_building_model", "procedural_buildings", "procedural_building_interiors",
    "high_quality_building_textures", "house_style_preset",
    "building_width_quantum",
    "building_length_quantum", "building_height_quantum", "building_minimum_width",
    "building_maximum_width", "building_minimum_length", "building_maximum_length",
    "building_minimum_height", "building_maximum_height", "building_level_height",
    "building_maximum_variants", "building_roof_pitch_degrees", "building_foundation_depth",
    "building_foundation_maximum_depth", "building_foundation_depth_quantum",
    "building_foundation_safety", "building_maximum_pad_relief",
    "church_ground_clearance",
    "max_forest_objects", "forest_tree_spacing", "forest_road_clearance",
    "forest_ground_clearance", "forest_profile", "forest_individual_objects_only", "forest_tree_model",
    "forest_low_anchor", "forest_maximum_block_relief",
    "forest_block_maximum_burial", "forest_block_maximum_float",
    "forest_block_maximum_ground_sink",
    "forest_everon_steep_model", "forest_everon_steep_footprint",
    "forest_everon_steep_maximum_relief", "forest_everon_steep_maximum_burial",
    "forest_everon_steep_maximum_float", "forest_everon_steep_maximum_ground_sink",
    "forest_polygon_sink_fraction",
    "forest_severe_hill_fallback", "forest_severe_hill_relief",
    "forest_severe_hill_trees_per_block",
    "forest_cluster_fallback",
    "forest_cluster_search_radius", "forest_cluster_maximum_relief",
    "forest_cluster_maximum_burial", "forest_cluster_maximum_float",
    "forest_cluster_tree_maximum_float", "forest_cluster_bush_maximum_float",
    "forest_cluster_footprint_margin", "forest_undergrowth_enabled",
    "forest_undergrowth_maximum_objects", "forest_undergrowth_spacing",
    "forest_undergrowth_maximum_relief", "forest_undergrowth_maximum_burial",
    "forest_undergrowth_maximum_float", "forest_undergrowth_ground_clearance",
    "steep_hill_bushes_enabled", "maximum_steep_hill_bush_objects",
    "steep_hill_bush_spacing", "steep_hill_bush_minimum_slope_degrees",
    "steep_hill_bush_maximum_relief", "steep_hill_bush_maximum_burial",
    "steep_hill_bush_maximum_float", "steep_hill_bush_ground_clearance",
    "steep_hill_bush_models", "forest_border_enabled",
    "forest_border_maximum_objects", "forest_border_spacing",
    "forest_border_inset", "forest_border_maximum_relief",
    "forest_border_maximum_burial", "forest_border_maximum_float",
    "forest_single_tree_enabled", "forest_single_tree_model", "forest_roadside_tree_model",
    "forest_roadside_tree_models", "forest_roadside_trees_per_cut_block",
    "forest_roadside_bush_models", "forest_roadside_bushes_per_cut_block",
    "forest_roadside_bush_footprint",
    "maximum_forest_single_tree_objects", "forest_single_tree_spacing",
    "forest_single_tree_footprint", "forest_single_tree_maximum_relief",
    "forest_single_tree_root_sink", "forest_single_tree_maximum_burial",
    "forest_single_tree_maximum_float", "forest_gap_infill_enabled",
    "forest_gap_infill_spacing",
    "ditch_grass_enabled", "maximum_ditch_grass_objects",
    "ditch_grass_spacing", "ditch_grass_endpoint_trim",
    "ditch_grass_maximum_relief", "ditch_grass_maximum_burial",
    "ditch_grass_maximum_float", "ditch_grass_ground_clearance",
    "forest_hillside_fallback", "forest_hillside_tree_model",
    "forest_hillside_trees_per_block", "forest_hillside_tree_footprint",
    "forest_hillside_tree_maximum_relief",
    "barriers_enabled", "maximum_barrier_objects", "barrier_segment_length",
    "stock_hedge_models", "stock_wall_models", "stock_metal_fence_models",
    "sidewalks_enabled", "maximum_sidewalk_objects", "sidewalk_width",
    "sidewalk_segment_length", "street_furniture_enabled",
    "maximum_street_furniture_objects", "street_light_spacing",
    "street_bench_every", "street_bin_every",
    "match_nearby_building_textures", "nearby_building_texture_match_distance",
    "bridges_enabled", "procedural_bridges", "maximum_bridge_objects", "bridge_module_length",
    "bridge_deck_clearance", "bridge_water_clearance",
    "residential_infill_enabled", "maximum_residential_infill_buildings",
    "residential_infill_spacing", "residential_infill_minimum_area",
    "residential_infill_road_clearance", "residential_infill_building_clearance",
    "rural_vegetation_enabled", "maximum_rural_vegetation_objects", "rural_vegetation_spacing",
    "meadow_grass_enabled", "maximum_meadow_grass_objects", "meadow_grass_spacing",
    "haybales_enabled", "maximum_haybale_objects", "haybale_spacing", "haybale_field_percent",
    "wetland_reeds_enabled", "maximum_wetland_reed_objects", "wetland_reed_spacing",
    "wetland_reed_maximum_relief", "wetland_reed_maximum_burial",
    "wetland_reed_maximum_float", "wetland_reed_ground_clearance", "wetland_reed_models",
    "rocky_forest_fallback_enabled", "maximum_rocky_forest_objects",
    "rocky_forest_rocks_per_patch", "rocky_forest_spread",
    "rocky_forest_maximum_relief", "rocky_forest_maximum_burial",
    "rocky_forest_maximum_float", "deterministic_seed",
    "cemeteries_enabled", "maximum_grave_objects", "grave_spacing",
    "grave_inset", "grave_footprint", "grave_ground_clearance", "grave_models",
)


def _scaled_progress_callback(start: int, end: int) -> Callable[[int, str], None]:
    """Map a local 0..100 stage meter into the build's overall percentage."""

    def callback(percent: int, stage: str) -> None:
        local = max(0, min(100, int(percent)))
        report_progress(start + round((end - start) * local / 100.0), stage)

    return callback


def _load_processed_dem(
    spec: PlayabilitySpec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[HeightmapLoadResult, bool, str, str | None]:
    cache_dir, enabled, refresh = _cache_settings(spec)
    payload = {
        "source": file_snapshot(spec.heightmap_path),
        "width": spec.cells,
        "height": spec.cells,
        "input_mode": spec.input_mode,
        "elevation_minimum": spec.elevation_minimum,
        "elevation_maximum": spec.elevation_maximum,
        "input_minimum": spec.input_minimum,
        "input_maximum": spec.input_maximum,
        "flip_y": spec.flip_y,
        "source_grid": spec.heightmap_grid,
    }
    key = cache_key("processed-dem-v2-runtime-vertices", payload)
    path = cache_dir / "dem" / f"{key}.pickle" if cache_dir is not None else None
    def produce() -> HeightmapLoadResult:
        if progress_callback is not None:
            progress_callback(10, f"Reading heightmap file {spec.heightmap_path.name}")
        result = load_heightmap(
            spec.heightmap_path,
            spec.cells,
            spec.cells,
            input_mode=spec.input_mode,
            elevation_minimum=spec.elevation_minimum,
            elevation_maximum=spec.elevation_maximum,
            input_minimum=spec.input_minimum,
            input_maximum=spec.input_maximum,
            flip_y=spec.flip_y,
            source_grid=spec.heightmap_grid,
        )
        if progress_callback is not None:
            conversion = (
                "; legacy cell centres shifted onto WRP vertices"
                if result.legacy_centre_to_vertex_conversion
                else "; WRP vertex grid preserved"
            )
            progress_callback(
                90,
                f"Heightmap decoded and resampled to {spec.cells:,}×{spec.cells:,}{conversion}",
            )
        return result

    if progress_callback is not None:
        progress_callback(0, "Checking processed elevation cache")
    value, hit = load_or_create_pickle(
        cache_path=path,
        producer=produce,
        enabled=enabled,
        refresh=refresh,
        stage_schema=_STAGE_CACHE_SCHEMA,
        validator=lambda item: isinstance(item, HeightmapLoadResult) and len(item.elevations) == spec.cells * spec.cells,
    )
    if progress_callback is not None:
        progress_callback(100, "Processed elevation grid ready" + (" from cache" if hit else ""))
    return value, hit, key, str(path) if path is not None else None


def _load_osm_raster(
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: PlayabilitySpec,
    dataset_identity: str,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[OsmRaster, bool, str, str | None]:
    cache_dir, enabled, refresh = _cache_settings(spec)
    payload = {
        "dataset": dataset_identity,
        "bbox": spec.bbox,
        "world_size": spec.world_size,
        "cells": spec.cells,
        "include_minor_roads": spec.include_minor_roads,
    }
    key = cache_key("osm-raster-v2-conservative-water-coverage", payload)
    path = cache_dir / "spatial" / f"raster-{key}.pickle" if cache_dir is not None else None
    if progress_callback is not None:
        progress_callback(0, "Checking OpenStreetMap raster cache")
    value, hit = load_or_create_pickle(
        cache_path=path,
        producer=lambda: rasterize_osm(
            dataset, projection, cells=spec.cells, include_minor_roads=spec.include_minor_roads,
            progress_callback=progress_callback,
        ),
        enabled=enabled,
        refresh=refresh,
        stage_schema=_STAGE_CACHE_SCHEMA,
        validator=lambda item: isinstance(item, OsmRaster) and item.cells == spec.cells,
    )
    if progress_callback is not None:
        progress_callback(100, "OpenStreetMap raster ready" + (" from cache" if hit else ""))
    return value, hit, key, str(path) if path is not None else None


def _load_terrain_solution(
    loaded: HeightmapLoadResult,
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: PlayabilitySpec,
    dem_key: str,
    raster_key: str,
    dataset_identity: str,
    *,
    building_placement_plans: Sequence[BuildingPlacementPlan] = (),
    progress_callback: Callable[[int, str], None] | None = None,
):
    cache_dir, enabled, refresh = _cache_settings(spec)
    payload = {
        "dem": dem_key,
        "raster": raster_key,
        "dataset": dataset_identity,
        "spec": _spec_fields(spec, _TERRAIN_CACHE_FIELDS),
        "out_of_bounds_dem": file_snapshot(getattr(spec, "out_of_bounds_dem_path", None)),
        "building_plans": streaming_hash(
            "terrain-building-plans-v2", building_placement_plans
        ),
    }
    key = cache_key("terrain-solution-v26-road-platform-dry-bridge-filter", payload)
    path = cache_dir / "terrain" / f"{key}.pickle" if cache_dir is not None else None

    def produce():
        if isinstance(spec, ConstraintPlayabilitySpec):
            from .terrain_solver import solve_terrain_constraints
            grading = solve_terrain_constraints(
                loaded.elevations, dataset, projection, raster, spec,
                building_placement_plans=building_placement_plans,
                progress_callback=progress_callback,
            )
        else:
            if progress_callback is not None:
                progress_callback(10, "Applying water elevations and shoreline blend")
            solver_input = apply_water_elevations(
                loaded.elevations,
                raster,
                sea_level=spec.sea_level,
                water_depth=spec.water_depth,
                beach_height=spec.beach_height,
                blend_cells=spec.coastline_blend_cells,
                cell_size=spec.cell_size,
                maximum_shore_slope_percent=float(getattr(spec, "lake_shore_maximum_slope_percent", 8.0)),
            )
            if progress_callback is not None:
                progress_callback(45, "Grading roads and building pads")
            grading = grade_terrain(solver_input, dataset, projection, raster, spec)
        if progress_callback is not None:
            progress_callback(95, "Calculating terrain slope grid")
        slopes = tuple(calculate_slopes(grading.elevations, spec.cells, spec.cells, spec.cell_size))
        return grading, slopes

    if progress_callback is not None:
        progress_callback(0, "Checking terrain-solution cache")
    value, hit = load_or_create_pickle(
        cache_path=path,
        producer=produce,
        enabled=enabled,
        refresh=refresh,
        stage_schema=_STAGE_CACHE_SCHEMA,
        validator=lambda item: (
            isinstance(item, tuple) and len(item) == 2
            and hasattr(item[0], "elevations")
            and len(item[0].elevations) == spec.cells * spec.cells
            and len(item[1]) == spec.cells * spec.cells
        ),
    )
    grading, slopes = value
    if progress_callback is not None:
        progress_callback(100, "Terrain solution ready" + (" from cache" if hit else ""))
    return grading, tuple(slopes), hit, key, str(path) if path is not None else None


def _surface_pipeline_value(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    slopes: Sequence[float],
    spec: PlayabilitySpec,
    progress_callback: Callable[[int, str], None] | None = None,
):
    if progress_callback is not None:
        progress_callback(0, "Classifying base terrain materials")
    if spec.material_mask_path is None:
        base_material_indices = classify_materials(
            elevations,
            slopes,
            sea_level=spec.sea_level,
            beach_height=spec.beach_height,
            rock_height=spec.rock_height,
            rock_slope_degrees=spec.rock_slope_degrees,
        )
        mask_metadata = None
    else:
        mask = load_material_mask(
            spec.material_mask_path,
            spec.cells,
            spec.cells,
            palette=tuple(material.colour for material in DEFAULT_MATERIALS),
            flip_y=spec.flip_y,
        )
        base_material_indices = mask.indices
        mask_metadata = {
            "path": str(spec.material_mask_path),
            "source_width": mask.source_width,
            "source_height": mask.source_height,
            "source_mode": mask.source_mode,
        }
    if progress_callback is not None:
        progress_callback(25, "Building Milestone 9 shoreline and land-use materials")
    surface_report = (
        build_surface_pass(dataset, projection, raster, elevations, slopes, spec)
        if _surface_pass_enabled(spec)
        else None
    )
    if progress_callback is not None:
        progress_callback(78, "Overlaying OSM water, roads and buildings")
    overlaid = overlay_materials(base_material_indices, raster)
    if progress_callback is not None:
        progress_callback(88, "Softening shoreline and land-use transitions")
    ground_transitions = improve_transitions(overlaid, raster, spec)
    if _surface_ground_enabled(spec):
        assert surface_report is not None
        material_indices = surface_report.indices
        transitions = surface_report
    else:
        material_indices = ground_transitions.indices
        transitions = ground_transitions
    if progress_callback is not None:
        progress_callback(100, "Surface and transition masks ready")
    return {
        "base_material_indices": tuple(base_material_indices),
        "mask_metadata": mask_metadata,
        "surface_report": surface_report,
        "ground_transitions": ground_transitions,
        "material_indices": tuple(material_indices),
        "transitions": transitions,
    }


def _load_surface_pipeline(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    slopes: Sequence[float],
    spec: PlayabilitySpec,
    terrain_key: str,
    dataset_identity: str,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    cache_dir, enabled, refresh = _cache_settings(spec)
    payload = {
        "terrain": terrain_key,
        "dataset": dataset_identity,
        "material_mask": file_snapshot(spec.material_mask_path),
        "colour_reference": file_snapshot(getattr(spec, "surface_colour_reference_path", None)),
        "spec": _spec_fields(spec, _SURFACE_CACHE_FIELDS),
    }
    key = cache_key("surface-pipeline-v11-vectorized-material-pass", payload)
    path = cache_dir / "surfaces" / f"{key}.pickle" if cache_dir is not None else None
    if progress_callback is not None:
        progress_callback(0, "Checking surface-mask cache")
    value, hit = load_or_create_pickle(
        cache_path=path,
        producer=lambda: _surface_pipeline_value(
            dataset, projection, raster, elevations, slopes, spec, progress_callback
        ),
        enabled=enabled,
        refresh=refresh,
        stage_schema=_STAGE_CACHE_SCHEMA,
        validator=lambda item: isinstance(item, dict) and len(item.get("material_indices", ())) == spec.cells * spec.cells,
    )
    if progress_callback is not None:
        progress_callback(100, "Surface masks ready" + (" from cache" if hit else ""))
    return value, hit, key, str(path) if path is not None else None




def _placement_driven_surface_overlay(
    indices: Sequence[int],
    surface_report: SurfacePassReport | None,
    nonroads: ObjectGenerationResult,
    raster: OsmRaster,
    spec: PlayabilitySpec,
    slopes: Sequence[float] | None = None,
) -> tuple[tuple[int, ...], SurfacePassReport | None, dict[str, int]]:
    """Apply forest/rock materials from actual successful placement results.

    Stock and compact forest models paint forest ground. Rock fallback models
    paint steep rock. This deliberately runs after placement so surfaces cannot
    claim a forest where every placement tier failed.
    """
    if not _surface_pass_enabled(spec) or len(indices) != spec.cells * spec.cells:
        return tuple(indices), surface_report, {"accepted_forest_cells": 0, "rocky_forest_cells": 0}
    if slopes is None:
        slopes = (0.0,) * len(indices)
    if len(slopes) != len(indices):
        raise ValueError("placement surface overlay slope grid size mismatch")
    from .surface_pass import MATERIAL_INDEX

    active_material_index = {
        material.code: index for index, material in enumerate(_material_definitions(spec))
    }
    forest_index = active_material_index["f"]
    rock_index = active_material_index.get("k", active_material_index["r"])
    result = list(indices)
    forest_mask = [False] * len(result)
    rocky_mask = [False] * len(result)
    forest_radius = max(spec.cell_size * 0.55, float(getattr(spec, "forest_tree_spacing", 50.0)) * 0.48)
    # One failed 50 m forest block is a rocky *patch*, not a ten-metre coin
    # beneath one object. Paint enough of the rejected patch to make the stock
    # Resistance stone tile visible; explicit rock outcrops may override nearby canopy.
    rock_radius = max(
        spec.cell_size * 0.75,
        float(getattr(spec, "forest_tree_spacing", 50.0)) * 0.48,
        float(getattr(spec, "rocky_forest_spread", 18.0)) + spec.cell_size * 0.15,
    )

    def paint(mask: list[bool], x: float, z: float, radius: float, *, require_forest: bool) -> None:
        col0 = max(0, int((x - radius) // spec.cell_size))
        col1 = min(spec.cells - 1, int((x + radius) // spec.cell_size))
        row0 = max(0, int((z - radius) // spec.cell_size))
        row1 = min(spec.cells - 1, int((z + radius) // spec.cell_size))
        rr = radius * radius
        for row in range(row0, row1 + 1):
            cz = (row + 0.5) * spec.cell_size
            for col in range(col0, col1 + 1):
                cx = (col + 0.5) * spec.cell_size
                idx = row * spec.cells + col
                if (cx - x) ** 2 + (cz - z) ** 2 > rr:
                    continue
                if require_forest and not raster.forest[idx]:
                    continue
                water_mask = getattr(raster, "water", None)
                road_mask = getattr(raster, "roads", None)
                building_mask = getattr(raster, "buildings", None)
                if ((water_mask is not None and water_mask[idx])
                        or (road_mask is not None and road_mask[idx])
                        or (building_mask is not None and building_mask[idx])):
                    continue
                mask[idx] = True

    # Placement now records only the coordinates relevant to this overlay while
    # objects are emitted. Avoid another million-object pass and repeated path
    # casefold/prefix classification here. Direct API/tests may still construct
    # ObjectGenerationResult manually, so retain a compatibility fallback when
    # no emission-time model aggregate exists.
    if getattr(nonroads, "model_usage", ()) or not getattr(nonroads, "objects", ()):
        forest_positions = tuple(getattr(nonroads, "surface_forest_positions", ()))
        rock_positions = tuple(getattr(nonroads, "surface_rock_positions", ()))
    else:
        forest_models = {
            str(spec.forest_tree_model).casefold(),
            str(getattr(spec, "forest_everon_steep_model", "")).casefold(),
            str(getattr(spec, "forest_single_tree_model", "")).casefold(),
            str(getattr(spec, "forest_hillside_tree_model", "")).casefold(),
            str(getattr(spec, "forest_roadside_tree_model", "")).casefold(),
            *(str(model).casefold() for model in getattr(spec, "forest_roadside_tree_models", ())),
        }
        forest_models.discard("")
        compact_prefix = (spec.name + r"\f\c_").casefold()
        rock_prefix = (spec.name + r"\i\rock_").casefold()
        stock_rock_models = {path.casefold() for path in STOCK_STONE_MODELS}
        fallback_forest: list[tuple[float, float]] = []
        fallback_rock: list[tuple[float, float]] = []
        for obj in nonroads.objects:
            model = obj.model_path.casefold()
            if model in forest_models or model.startswith(compact_prefix):
                fallback_forest.append((obj.x, obj.z))
            elif model.startswith(rock_prefix) or model in stock_rock_models:
                fallback_rock.append((obj.x, obj.z))
        forest_positions = tuple(fallback_forest)
        rock_positions = tuple(fallback_rock)

    for x, z in forest_positions:
        paint(forest_mask, x, z, forest_radius, require_forest=True)
    minimum_rock_slope = float(getattr(spec, "rock_slope_degrees", 44.0))
    for x, z in rock_positions:
        col = min(spec.cells - 1, max(0, int(x // spec.cell_size)))
        row = min(spec.cells - 1, max(0, int(z // spec.cell_size)))
        index = row * spec.cells + col
        if slopes[index] >= minimum_rock_slope:
            paint(rocky_mask, x, z, rock_radius, require_forest=False)

    # Forest blocks often fail on sharply broken terrain. Do not leave those
    # holes as inexplicable bright grass. Where an OSM forest cell has no
    # successful tree/cluster coverage and is already a steep hillside, expose
    # the Resistance stone material. Procedural rock objects below/nearby then
    # break up the otherwise flat terrain tile.
    gap_slope = max(
        float(getattr(spec, "rock_slope_degrees", 44.0)),
        float(getattr(spec, "surface_steep_slope_degrees", 52.0)) - 4.0,
    )
    water_mask = getattr(raster, "water", ())
    road_mask = getattr(raster, "roads", ())
    building_mask = getattr(raster, "buildings", ())
    for idx, is_forest in enumerate(raster.forest):
        if not is_forest or forest_mask[idx] or rocky_mask[idx] or slopes[idx] < gap_slope:
            continue
        if ((water_mask and water_mask[idx]) or (road_mask and road_mask[idx]) or (building_mask and building_mask[idx])):
            continue
        rocky_mask[idx] = True

    for idx, accepted in enumerate(forest_mask):
        if accepted:
            result[idx] = forest_index
    for idx, rocky in enumerate(rocky_mask):
        if rocky:
            # A visible rock outcrop is stronger evidence than a nearby tree.
            # Let the stone patch show through beneath scattered boulders instead
            # of repainting it as forest merely because canopy coverage overlaps.
            result[idx] = rock_index

    updated = tuple(result)
    if surface_report is not None:
        # The report and overview always use the expanded Milestone 9 index
        # space, even when the caller explicitly requests legacy Milestone 8
        # ground application. Never write Milestone 9 indexes into the old WRP
        # texture table: index 7 is forest in M9 but paved road in M8.
        report_indices = list(surface_report.indices)
        report_forest_index = MATERIAL_INDEX["f"]
        report_rock_index = MATERIAL_INDEX["k"]
        for idx, accepted in enumerate(forest_mask):
            if accepted:
                report_indices[idx] = report_forest_index
        for idx, rocky in enumerate(rocky_mask):
            if rocky:
                report_indices[idx] = report_rock_index
        updated_report = tuple(report_indices)
        surface_report = replace(
            surface_report,
            indices=updated_report,
            rock_cells=updated_report.count(MATERIAL_INDEX["r"]),
            steep_rock_cells=updated_report.count(report_rock_index),
        )
    return updated, surface_report, {
        "accepted_forest_cells": sum(forest_mask),
        "rocky_forest_cells": sum(rocky_mask),
    }

def _iterative_grounding_pass(
    elevations: Sequence[float],
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: PlayabilitySpec,
    building_library: ProceduralBuildingLibrary | None,
    building_placement_plans: Sequence[BuildingPlacementPlan],
    building_plans_truncated: bool,
    terrain_key: str,
    dataset_identity: str,
) -> tuple[tuple[float, ...], IterativeGroundingReport, str]:
    """Plan rigid supports, correct them, then return final quantized terrain."""

    if not bool(getattr(spec, "iterative_grounding_enabled", True)):
        return tuple(elevations), IterativeGroundingReport(), terrain_key
    provisional = plan_iterative_grounding_objects(
        dataset,
        projection,
        raster,
        elevations,
        spec,
        building_placement_plans,
        progress_callback=_scaled_progress_callback(32, 34),
    )
    refined, report = refine_iterative_grounding_terrain(
        elevations,
        provisional,
        building_placement_plans,
        raster,
        spec,
    )
    quantized = quantize_elevations(refined, spec.height_scale)
    refined_key = cache_key(
        "iterative-grounding-terrain-v2-lightweight-rigid-supports",
        {
            "base_terrain": terrain_key,
            "elevations": float_sequence_sha256(quantized),
            "report": asdict(report),
        },
    )
    return quantized, report, refined_key


def _load_nonroad_objects(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    spec: PlayabilitySpec,
    *,
    starting_object_id: int,
    building_library: ProceduralBuildingLibrary | None,
    terrain_key: str,
    dataset_identity: str,
    road_fingerprint: str,
    building_placement_plans: Sequence[BuildingPlacementPlan] = (),
    building_plans_truncated: bool = False,
    progress_callback: Callable[[int, str], None] = report_progress,
):
    cache_dir, enabled, refresh = _cache_settings(spec)
    payload = {
        "terrain": terrain_key,
        "dataset": dataset_identity,
        "road_fingerprint": road_fingerprint,
        "building_plan_count": len(building_placement_plans),
        "building_plans_truncated": building_plans_truncated,
        "starting_object_id": starting_object_id,
        "spec": _spec_fields(spec, _PLACEMENT_CACHE_FIELDS),
    }
    key = cache_key("nonroad-object-placement-v96-road-safe-settlement-clutter", payload)
    path = cache_dir / "placements" / f"{key}.pickle" if cache_dir is not None else None

    def produce():
        result = generate_world_objects(
            dataset,
            projection,
            raster,
            elevations,
            spec,
            include_roads=False,
            starting_object_id=starting_object_id,
            building_asset_library=building_library,
            building_placement_plans=building_placement_plans,
            building_plans_truncated=building_plans_truncated,
            progress_callback=progress_callback,
        )
        return result, building_library

    value, hit = load_or_create_pickle(
        cache_path=path,
        producer=produce,
        enabled=enabled,
        refresh=refresh,
        stage_schema=_STAGE_CACHE_SCHEMA,
        validator=lambda item: isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ObjectGenerationResult),
    )
    result, cached_library = value
    if hit and cached_library is not None:
        cached_library.cache_dir = cache_dir
        cached_library.cache_enabled = enabled
        cached_library.cache_refresh = refresh
        cached_library.cache_hits = 0
        cached_library.cache_misses = 0
    return result, cached_library, hit, key, str(path) if path is not None else None


def _assemble_world_objects(
    road_objects: Sequence[WorldObject],
    nonroads: ObjectGenerationResult,
    semantic_objects: Sequence[WorldObject],
    *,
    renumber: bool = True,
) -> tuple[WorldObject, ...]:
    """Order static WRP records by gameplay importance.

    Build-scale callers pass ``renumber=False`` and let :func:`write_rvw4`
    assign the final sequential wire IDs.  That avoids cloning up to a million
    frozen ``WorldObject`` instances solely to change one integer.  The default
    remains the historical renumbered result for direct/API callers and tests.
    """

    if nonroads.road_objects:
        raise ValueError("non-road object assembly received embedded road objects")

    urban_detail_count = nonroads.sidewalk_objects + nonroads.street_furniture_objects
    building_count = nonroads.building_objects
    forest_count = (
        nonroads.forest_objects
        + nonroads.rocky_forest_objects
        + nonroads.forest_undergrowth_objects
        + nonroads.steep_hill_bush_objects
        + nonroads.forest_border_objects
    )
    ditch_count = nonroads.ditch_grass_objects
    barrier_count = nonroads.barrier_objects
    bridge_count = nonroads.bridge_objects
    rural_count = (
        nonroads.tree_row_objects
        + nonroads.orchard_objects
        + nonroads.vineyard_objects
        + nonroads.scrub_objects
        + nonroads.rural_rock_objects
        + nonroads.wetland_reed_objects
        + nonroads.meadow_grass_objects
        + nonroads.haybale_objects
    )
    mapped_tree_count = nonroads.mapped_tree_objects
    utility_count = nonroads.utility_objects

    objects = nonroads.objects
    expected_count = (
        urban_detail_count + building_count + forest_count + ditch_count + barrier_count
        + bridge_count + rural_count + mapped_tree_count + utility_count
    )
    if expected_count != len(objects):
        raise ValueError(
            "non-road object category counts do not match the generated object list: "
            f"categories={expected_count}, objects={len(objects)}"
        )

    cursor = 0
    urban_details = objects[cursor : cursor + urban_detail_count]
    cursor += urban_detail_count
    buildings = objects[cursor : cursor + building_count]
    cursor += building_count
    forests_and_trees = objects[cursor : cursor + forest_count]
    cursor += forest_count
    ditch = objects[cursor : cursor + ditch_count]
    cursor += ditch_count
    barriers = objects[cursor : cursor + barrier_count]
    cursor += barrier_count
    bridges = objects[cursor : cursor + bridge_count]
    cursor += bridge_count
    rural_vegetation = objects[cursor : cursor + rural_count]
    cursor += rural_count
    mapped_trees = objects[cursor : cursor + mapped_tree_count]
    cursor += mapped_tree_count
    utilities = objects[cursor : cursor + utility_count]

    # One tuple allocation instead of a chain of tuple concatenations.  Repeated
    # ``+`` copies an ever-growing prefix and gets surprisingly expensive at the
    # million-object cap.
    ordered = tuple(chain(
        road_objects,
        urban_details,
        forests_and_trees,
        mapped_trees,
        rural_vegetation,
        buildings,
        ditch,
        barriers,
        bridges,
        utilities,
        semantic_objects,
    ))
    if not renumber:
        return ordered
    return tuple(
        replace(obj, object_id=index)
        for index, obj in enumerate(ordered, 1)
    )


def _road_object_fingerprint(objects: Sequence[object]) -> str:
    return streaming_hash(
        "road-objects-v2",
        (
            (
                getattr(obj, "model_path", ""),
                round(float(getattr(obj, "x", 0.0)), 6),
                round(float(getattr(obj, "y", 0.0)), 6),
                round(float(getattr(obj, "z", 0.0)), 6),
                round(float(getattr(obj, "heading_degrees", 0.0)), 6),
                round(float(getattr(obj, "pitch_degrees", 0.0)), 6),
            )
            for obj in objects
        ),
    )


def _verify_single_world_pbo_layout(
    pbo_path: Path,
    world_name: str,
    infrastructure_generation: InfrastructureAssetResult | None,
) -> dict[str, object]:
    """Verify generated infrastructure is packed beside the WRP in one PBO.

    Generated gravel-road model references use ``<world>\\i\\...``. They are only
    reachable at runtime when the corresponding ``i\\`` entries are inside the
    world's own PBO, so verify the archive rather than trusting the source tree.
    """
    archive_entries = {entry.name.casefold() for entry in read_pbo(pbo_path)}
    expected_entries = ["config.cpp", f"{world_name}.wrp"]
    generated_entries: list[str] = []
    if infrastructure_generation is not None:
        generated_entries.extend(infrastructure_generation.model_files)
        generated_entries.extend(infrastructure_generation.texture_files)
        generated_entries.append("i/infrastructure.json")
    expected_entries.extend(generated_entries)

    canonical_expected = tuple(
        dict.fromkeys(entry.replace("/", "\\") for entry in expected_entries)
    )
    missing = tuple(
        entry for entry in canonical_expected if entry.casefold() not in archive_entries
    )
    if missing:
        raise RuntimeError(
            "world PBO is missing generated runtime entries: " + ", ".join(missing)
        )

    generated_road_models = tuple(
        entry for entry in canonical_expected
        if entry.casefold().startswith("i\\gravel") and entry.casefold().endswith(".p3d")
    )
    gravel_texture_entries = {
        rf"i\{_texture_file_stem('gravel')}.paa".casefold(),
        rf"i\{_texture_file_stem('gravel_edge')}.paa".casefold(),
    }
    generated_road_textures = tuple(
        entry for entry in canonical_expected
        if entry.casefold() in gravel_texture_entries
    )
    return {
        "mode": "single_world_pbo",
        "world_pbo": pbo_path.name,
        "separate_road_pbo": False,
        "verified": True,
        "generated_infrastructure_entries": list(generated_entries),
        "generated_road_models": list(generated_road_models),
        "generated_road_textures": list(generated_road_textures),
        "missing_entries": [],
    }

def build_milestone4(
    output_dir: Path,
    spec: PlayabilitySpec,
    *,
    clean: bool = True,
    mod_directory_name: str = "@CWR-Milestone4",
    milestone_number: int = 4,
    dataset_override: OsmDataset | None = None,
) -> BuildResult:
    spec.validate()
    output_dir = output_dir.resolve()
    prepare_output_directory(output_dir, spec.name, clean=clean)

    source_dir = output_dir / "source" / spec.name
    wrp_path = source_dir / f"{spec.name}.wrp"
    materials = _material_definitions(spec)
    material_texture_paths = tuple(source_dir / "data" / f"{material.code}.paa" for material in materials)
    dummy_texture_path = source_dir / "data" / "d.paa"
    if _surface_ground_enabled(spec):
        generated_material_texture_paths = tuple(
            path for path, material in zip(material_texture_paths, materials)
            if _ground_texture_profile(spec) not in {"everon", "nogova"} or getattr(material, "everon_path", None) is None
        )
    else:
        generated_material_texture_paths = material_texture_paths if _ground_texture_profile(spec) in {"generated", "desert", "malden"} else ()
    texture_paths = generated_material_texture_paths + (dummy_texture_path,)
    mod_root = output_dir / mod_directory_name
    pbo_path = mod_root / "Addons" / f"{spec.name}.pbo"
    mission_path = output_dir / "Missions" / f"test_mission.{spec.name}" / "mission.sqm"
    intro_dir = mod_root / "Anims" / f"{WORLD_INTRO_NAME}.{spec.name}"
    intro_mission_path = intro_dir / "mission.sqm"
    intro_script_path = intro_dir / "intro.sqs"
    preview_path = output_dir / "preview.png"
    height_preview_path = output_dir / "height-preview.png"
    material_preview_path = output_dir / "material-preview.png"
    osm_preview_path = output_dir / "osm-geography-preview.png"
    osm_source_path = output_dir / "osm-source.json"
    osm_query_path = output_dir / "overpass-query.txt"
    attribution_path = output_dir / "OSM-ATTRIBUTION.txt"
    mod_attribution_path = mod_root / "OSM-ATTRIBUTION.txt"
    asset_catalogue_path = output_dir / "asset-catalogue.json"
    meadow_grass_preview_path = (
        output_dir / "meadow-grass-placement.png" if milestone_number >= 9 else None
    )
    road_report_path = output_dir / "road-fit-report.json"
    grading_report_path = output_dir / "terrain-grading-report.json"
    reproducibility_path = output_dir / "reproducibility-report.json"
    building_catalogue_path = output_dir / "building-asset-catalogue.json"
    semantic_site_catalogue_path = output_dir / "semantic-site-catalogue.json"
    forest_cluster_catalogue_path = output_dir / "forest-cluster-catalogue.json"
    infrastructure_catalogue_path = output_dir / "infrastructure-asset-catalogue.json"
    surface_report_path = output_dir / "surface-pass-report.json"
    building_source_reference_path = output_dir / "building-source-reference.png"
    overview_map_path = output_dir / "overview-map.png"
    overview_paa_path = source_dir / "data" / "overview.paa"
    world_icon_path = source_dir / "data" / "icon.paa"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "validation-report.txt"
    cache_report_path = output_dir / "cache-report.json"

    source_dir.mkdir(parents=True, exist_ok=True)
    mission_path.parent.mkdir(parents=True, exist_ok=True)
    intro_dir.mkdir(parents=True, exist_ok=True)
    cache_dir, cache_enabled, cache_refresh = _cache_settings(spec)

    report_progress(0, "Starting core world generation")
    loaded, dem_cache_hit, dem_cache_key, dem_cache_path = _load_processed_dem(
        spec, progress_callback=_scaled_progress_callback(0, 6)
    )
    report_progress(6, "Reading OpenStreetMap source bundle")
    osm_bytes, query_text = load_osm_json(spec)
    report_progress(7, f"OpenStreetMap source loaded ({len(osm_bytes) / (1024 * 1024):.1f} MiB)")
    if dataset_override is not None:
        report_progress(8, f"Using normalized OpenStreetMap dataset with {dataset_override.element_count:,} source elements")
        dataset = dataset_override
        report_progress(13, "Normalized OpenStreetMap features ready")
    else:
        dataset = parse_overpass_json(
            osm_bytes, progress_callback=_scaled_progress_callback(7, 13)
        )
    dataset_identity = _dataset_identity(dataset, osm_bytes)
    projection = BboxProjection.create(spec.bbox, spec.world_size)
    raster, raster_cache_hit, raster_cache_key, raster_cache_path = _load_osm_raster(
        dataset, projection, spec, dataset_identity,
        progress_callback=_scaled_progress_callback(13, 23),
    )
    report_progress(23, "Planning final building footprints")
    building_library: ProceduralBuildingLibrary | None = None
    if bool(getattr(spec, "procedural_buildings", False)):
        report_progress(23, "Preparing procedural building variants")
        building_library = ProceduralBuildingLibrary(
            world_name=spec.name,
            width_quantum=float(getattr(spec, "building_width_quantum", 2.0)),
            length_quantum=float(getattr(spec, "building_length_quantum", 2.0)),
            height_quantum=float(getattr(spec, "building_height_quantum", 3.0)),
            minimum_width=float(getattr(spec, "building_minimum_width", 4.0)),
            maximum_width=float(getattr(spec, "building_maximum_width", 80.0)),
            minimum_length=float(getattr(spec, "building_minimum_length", 4.0)),
            maximum_length=float(getattr(spec, "building_maximum_length", 160.0)),
            minimum_height=float(getattr(spec, "building_minimum_height", 3.0)),
            maximum_height=float(getattr(spec, "building_maximum_height", 48.0)),
            default_level_height=float(getattr(spec, "building_level_height", 3.0)),
            maximum_variants=int(getattr(spec, "building_maximum_variants", 128)),
            roof_pitch_degrees=float(getattr(spec, "building_roof_pitch_degrees", 35.0)),
            foundation_depth=float(getattr(spec, "building_foundation_depth", 0.5)),
            maximum_foundation_depth=float(getattr(spec, "building_foundation_maximum_depth", 2.5)),
            foundation_depth_quantum=float(getattr(spec, "building_foundation_depth_quantum", 0.25)),
            church_plinth_height=0.0,
            generate_interiors=bool(
                getattr(spec, "procedural_building_interiors", False)
            ),
            high_quality_textures=bool(
                getattr(spec, "high_quality_building_textures", False)
            ),
            house_style_preset=str(getattr(spec, "house_style_preset", "auto")),
            cache_dir=getattr(spec, "cache_dir", None),
            cache_enabled=bool(getattr(spec, "cache_enabled", True)),
            cache_refresh=bool(getattr(spec, "cache_refresh", False)),
        )
        building_library.prepare(dataset, projection, spec.point_building_footprint)
    report_progress(23, "Resolving final building footprints and entrances")
    building_placement_plans, building_plans_truncated = plan_building_placements(
        dataset, projection, raster, spec, building_library,
        progress_callback=_scaled_progress_callback(23, 24),
    )
    report_progress(24, f"Planned {len(building_placement_plans):,} final building footprints")
    grading, slopes, terrain_cache_hit, terrain_cache_key, terrain_cache_path = _load_terrain_solution(
        loaded, dataset, projection, raster, spec, dem_cache_key, raster_cache_key, dataset_identity,
        building_placement_plans=building_placement_plans,
        progress_callback=_scaled_progress_callback(24, 32),
    )
    initial_terrain_cache_key = terrain_cache_key
    report_progress(31, "Finalizing terrain at RVW4 height precision")
    solver_elevations = tuple(grading.elevations)
    elevations = quantize_elevations(solver_elevations, spec.height_scale)
    report_progress(32, "Running provisional building grounding")
    elevations, iterative_grounding, terrain_cache_key = _iterative_grounding_pass(
        elevations,
        dataset,
        projection,
        raster,
        spec,
        building_library,
        building_placement_plans,
        building_plans_truncated,
        initial_terrain_cache_key,
        dataset_identity,
    )
    grading = replace(grading, elevations=elevations)
    # Every object is fitted after this point against the exact values written
    # into the WRP. Recalculate slopes as well so spawn and surface decisions do
    # not quietly refer to a terrain grid the game will never actually receive.
    slopes = tuple(calculate_slopes(elevations, spec.cells, spec.cells, spec.cell_size))

    if isinstance(spec, ConstraintPlayabilitySpec):
        solver_input_elevations = loaded.elevations
    else:
        solver_input_elevations = apply_water_elevations(
            loaded.elevations,
            raster,
            sea_level=spec.sea_level,
            water_depth=spec.water_depth,
            beach_height=spec.beach_height,
            blend_cells=spec.coastline_blend_cells,
            cell_size=spec.cell_size,
            maximum_shore_slope_percent=float(getattr(spec, "lake_shore_maximum_slope_percent", 8.0)),
        )

    surface_cached, surface_cache_hit, surface_cache_key, surface_cache_path = _load_surface_pipeline(
        dataset, projection, raster, elevations, slopes, spec, terrain_cache_key, dataset_identity,
        progress_callback=_scaled_progress_callback(34, 38),
    )
    base_material_indices = surface_cached["base_material_indices"]
    mask_metadata = surface_cached["mask_metadata"]
    surface_report = surface_cached["surface_report"]
    ground_transitions = surface_cached["ground_transitions"]
    material_indices = surface_cached["material_indices"]
    transitions = surface_cached["transitions"]
    excluded = tuple(water or building for water, building in zip(raster.water, raster.buildings))
    report_progress(38, "Selecting a safe map spawn")
    spawn = choose_spawn(
        elevations,
        slopes,
        spec.cells,
        spec.cells,
        spec.cell_size,
        sea_level=spec.sea_level,
        minimum_clearance=spec.spawn_clearance,
        maximum_slope_degrees=spec.maximum_spawn_slope_degrees,
        excluded=excluded,
    )

    report_progress(40, "Preparing remaining procedural model libraries")
    site_library: ProceduralSiteLibrary | None = None
    if milestone_number >= 9 and bool(getattr(spec, "semantic_landmarks", False)):
        site_library = ProceduralSiteLibrary(
            spec.name,
            int(getattr(spec, "semantic_site_maximum_variants", 64)),
            cache_dir=getattr(spec, "cache_dir", None),
            cache_enabled=bool(getattr(spec, "cache_enabled", True)),
            cache_refresh=bool(getattr(spec, "cache_refresh", False)),
        )
        site_library.prepare(dataset, projection)

    report_progress(42, "Fitting road geometry to terrain")
    road_fit = fit_road_objects(
        dataset, projection, elevations, spec, starting_id=1,
        progress_callback=_scaled_progress_callback(42, 49),
    )
    road_fingerprint = _road_object_fingerprint(road_fit.objects)
    report_progress(49, "Road fitting complete")
    report_progress(52, "Placing buildings and vegetation")
    nonroads, cached_building_library, placement_cache_hit, placement_cache_key, placement_cache_path = _load_nonroad_objects(
        dataset,
        projection,
        raster,
        elevations,
        spec,
        starting_object_id=len(road_fit.objects) + 1,
        building_library=building_library,
        terrain_key=terrain_cache_key,
        dataset_identity=dataset_identity,
        road_fingerprint=road_fingerprint,
        building_placement_plans=building_placement_plans,
        building_plans_truncated=building_plans_truncated,
    )
    report_progress(67, "Applying forest and rocky terrain materials")
    material_indices, surface_report, placement_surface_counts = _placement_driven_surface_overlay(
        material_indices, surface_report, nonroads, raster, spec, slopes=slopes
    )
    transitions = replace(transitions, indices=material_indices) if hasattr(transitions, "indices") else transitions
    if cached_building_library is not None:
        building_library = cached_building_library
    report_progress(69, "Placing semantic landmarks")
    semantic = (
        generate_semantic_objects(
            dataset, projection, elevations, spec, site_library,
            starting_object_id=len(road_fit.objects) + len(nonroads.objects) + 1,
            raster=raster,
            building_placement_plans=building_placement_plans,
        )
        if site_library is not None
        else SemanticGenerationResult((), 0, 0, 0, 0, 0, 0.0)
    )
    report_progress(71, "Ordering roads, trees, buildings and infrastructure")
    all_objects = _assemble_world_objects(
        road_fit.objects, nonroads, semantic.objects, renumber=False
    )
    combined_model_usage: Counter[str] = Counter(dict(nonroads.model_usage))
    # Roads and semantic objects are much smaller populations than dense forest/
    # placement output. Scan each once, then use the aggregate everywhere else.
    combined_model_usage.update(obj.model_path for obj in road_fit.objects)
    combined_model_usage.update(obj.model_path for obj in semantic.objects)
    generated = ObjectGenerationResult(
        objects=all_objects,
        road_objects=len(road_fit.objects),
        building_objects=nonroads.building_objects,
        forest_objects=nonroads.forest_objects,
        road_objects_truncated=road_fit.truncated,
        building_objects_truncated=nonroads.building_objects_truncated,
        forest_objects_truncated=nonroads.forest_objects_truncated,
        forest_road_rejections=nonroads.forest_road_rejections,
        maximum_building_grounding_raise=nonroads.maximum_building_grounding_raise,
        maximum_building_pad_relief=nonroads.maximum_building_pad_relief,
        maximum_building_foundation_depth=nonroads.maximum_building_foundation_depth,
        building_foundation_rejections=nonroads.building_foundation_rejections,
        building_interior_fallbacks=nonroads.building_interior_fallbacks,
        building_fully_submerged_rejections=nonroads.building_fully_submerged_rejections,
        building_road_nudges=nonroads.building_road_nudges,
        maximum_forest_grounding_raise=nonroads.maximum_forest_grounding_raise,
        forest_slope_rejections=nonroads.forest_slope_rejections,
        maximum_forest_relief=nonroads.maximum_forest_relief,
        forest_block_objects=nonroads.forest_block_objects,
        forest_hillside_tree_objects=nonroads.forest_hillside_tree_objects,
        forest_hillside_fallback_blocks=nonroads.forest_hillside_fallback_blocks,
        forest_hillside_unfilled_blocks=nonroads.forest_hillside_unfilled_blocks,
        forest_hillside_candidate_rejections=nonroads.forest_hillside_candidate_rejections,
        maximum_hillside_tree_relief=nonroads.maximum_hillside_tree_relief,
        forest_everon_steep_objects=nonroads.forest_everon_steep_objects,
        forest_sunk_polygon_objects=nonroads.forest_sunk_polygon_objects,
        forest_everon_steep_rejections=nonroads.forest_everon_steep_rejections,
        forest_cluster_objects=nonroads.forest_cluster_objects,
        forest_cluster_rejections=nonroads.forest_cluster_rejections,
        forest_cluster_maximum_burial=nonroads.forest_cluster_maximum_burial,
        forest_cluster_maximum_float=nonroads.forest_cluster_maximum_float,
        forest_cluster_variant_counts=nonroads.forest_cluster_variant_counts,
        forest_undergrowth_objects=nonroads.forest_undergrowth_objects,
        forest_undergrowth_rejections=nonroads.forest_undergrowth_rejections,
        forest_undergrowth_maximum_burial=nonroads.forest_undergrowth_maximum_burial,
        forest_undergrowth_maximum_float=nonroads.forest_undergrowth_maximum_float,
        steep_hill_bush_objects=nonroads.steep_hill_bush_objects,
        steep_hill_bush_rejections=nonroads.steep_hill_bush_rejections,
        wetland_reed_objects=nonroads.wetland_reed_objects,
        wetland_reed_rejections=nonroads.wetland_reed_rejections,
        forest_border_objects=nonroads.forest_border_objects,
        forest_border_rejections=nonroads.forest_border_rejections,
        forest_border_maximum_burial=nonroads.forest_border_maximum_burial,
        forest_border_maximum_float=nonroads.forest_border_maximum_float,
        forest_single_tree_objects=nonroads.forest_single_tree_objects,
        forest_gap_infill_tree_objects=nonroads.forest_gap_infill_tree_objects,
        ditch_grass_objects=nonroads.ditch_grass_objects,
        ditch_grass_rejections=nonroads.ditch_grass_rejections,
        ditch_grass_maximum_burial=nonroads.ditch_grass_maximum_burial,
        ditch_grass_maximum_float=nonroads.ditch_grass_maximum_float,
        maximum_forest_burial=nonroads.maximum_forest_burial,
        maximum_forest_float=nonroads.maximum_forest_float,
        barrier_objects=nonroads.barrier_objects,
        fence_objects=nonroads.fence_objects,
        wall_objects=nonroads.wall_objects,
        hedge_objects=nonroads.hedge_objects,
        barrier_rejections=nonroads.barrier_rejections,
        bridge_objects=nonroads.bridge_objects,
        bridge_segments=nonroads.bridge_segments,
        bridge_rejections=nonroads.bridge_rejections,
        residential_infill_objects=nonroads.residential_infill_objects,
        residential_infill_areas=nonroads.residential_infill_areas,
        tree_row_objects=nonroads.tree_row_objects,
        orchard_objects=nonroads.orchard_objects,
        vineyard_objects=nonroads.vineyard_objects,
        scrub_objects=nonroads.scrub_objects,
        rural_rock_objects=nonroads.rural_rock_objects,
        rural_vegetation_rejections=nonroads.rural_vegetation_rejections,
        meadow_grass_objects=nonroads.meadow_grass_objects,
        meadow_grass_rejections=nonroads.meadow_grass_rejections,
        haybale_objects=nonroads.haybale_objects,
        haybale_rejections=nonroads.haybale_rejections,
        haybale_fields_total=nonroads.haybale_fields_total,
        haybale_fields_selected=nonroads.haybale_fields_selected,
        meadow_grass_positions=nonroads.meadow_grass_positions,
        meadow_grass_rejection_positions=nonroads.meadow_grass_rejection_positions,
        rocky_forest_objects=nonroads.rocky_forest_objects,
        rocky_forest_rejections=nonroads.rocky_forest_rejections,
        mapped_tree_objects=nonroads.mapped_tree_objects,
        mapped_tree_rejections=nonroads.mapped_tree_rejections,
        utility_objects=nonroads.utility_objects,
        utility_rejections=nonroads.utility_rejections,
        vegetation_audit_tree_objects=nonroads.vegetation_audit_tree_objects,
        vegetation_audit_cluster_tree_proxies=nonroads.vegetation_audit_cluster_tree_proxies,
        vegetation_audit_cluster_bush_proxies=nonroads.vegetation_audit_cluster_bush_proxies,
        vegetation_audit_violations=nonroads.vegetation_audit_violations,
        vegetation_audit_maximum_tree_float=nonroads.vegetation_audit_maximum_tree_float,
        vegetation_audit_maximum_bush_float=nonroads.vegetation_audit_maximum_bush_float,
        model_usage=tuple(sorted(combined_model_usage.items(), key=lambda item: item[0].casefold())),
        surface_forest_positions=nonroads.surface_forest_positions,
        surface_rock_positions=nonroads.surface_rock_positions,
    )
    building_generation: BuildingGenerationResult | None = None
    report_progress(73, "Generating procedural building assets")
    if building_library is not None:
        building_generation = building_library.write_assets(source_dir, building_catalogue_path)
    report_progress(75, "Generating semantic landmark assets")
    site_generation: SiteAssetResult | None = None
    if site_library is not None:
        site_generation = site_library.write_assets(source_dir, semantic_site_catalogue_path)

    report_progress(77, "Generating forest cluster assets")
    forest_cluster_library: ProceduralForestClusterLibrary | None = None
    forest_cluster_generation: ForestClusterAssetResult | None = None
    generated_cluster_usage = tuple(
        (model_path, count)
        for model_path, count in nonroads.model_usage
        if milestone_number >= 9 and is_generated_cluster_model(spec.name, model_path)
    )
    if generated_cluster_usage:
        forest_cluster_library = ProceduralForestClusterLibrary(
            spec.name,
            proxy_profile=_forest_proxy_profile(spec),
            cache_dir=getattr(spec, "cache_dir", None),
            cache_enabled=bool(getattr(spec, "cache_enabled", True)),
            cache_refresh=bool(getattr(spec, "cache_refresh", False)),
        )
        for model_path, count in generated_cluster_usage:
            forest_cluster_library.register_model_usage(model_path, count)
        forest_cluster_generation = forest_cluster_library.write_assets(
            source_dir, forest_cluster_catalogue_path
        )

    report_progress(79, "Generating infrastructure and rock assets")
    infrastructure_library: ProceduralInfrastructureLibrary | None = None
    infrastructure_generation: InfrastructureAssetResult | None = None
    infrastructure_prefix = (spec.name + "\\i\\").casefold()
    generated_infrastructure_usage = tuple(
        (model_path, count)
        for model_path, count in combined_model_usage.items()
        if milestone_number >= 9 and model_path.casefold().startswith(infrastructure_prefix)
    )
    if generated_infrastructure_usage:
        infrastructure_library = ProceduralInfrastructureLibrary(
            spec.name,
            road_segment_length=spec.road_segment_length,
            cache_dir=getattr(spec, "cache_dir", None),
            cache_enabled=bool(getattr(spec, "cache_enabled", True)),
            cache_refresh=bool(getattr(spec, "cache_refresh", False)),
        )
        for model_path, count in generated_infrastructure_usage:
            infrastructure_library.register_model_usage(model_path, count)
        infrastructure_generation = infrastructure_library.write_assets(
            source_dir, infrastructure_catalogue_path
        )

    report_progress(81, "Collecting towns and external asset dependencies")
    towns = town_locations(dataset, projection, spec.town_name_limit)
    selected_models = sorted(combined_model_usage)
    externally_selected_models = [
        model for model in selected_models
        if (building_library is None or not building_library.is_generated_model(model))
        and (site_library is None or not site_library.is_generated_model(model))
        and (forest_cluster_library is None or not forest_cluster_library.is_generated_model(model))
        and (infrastructure_library is None or not infrastructure_library.is_generated_model(model))
    ]
    forest_proxy_models = (
        forest_cluster_library.required_proxy_models()
        if forest_cluster_library is not None
        else ()
    )
    ground_texture_paths = _ground_texture_paths(spec)
    external_ground_textures = _external_ground_texture_paths(spec)
    default_asset_mapping = default_osm_asset_mapping(
        spec, milestone_number, global_textures=external_ground_textures
    )
    osm_asset_mapping = load_osm_asset_mapping(
        getattr(spec, "osm_asset_mapping_path", None), default_asset_mapping
    )
    osm_asset_mapping_report = collect_osm_asset_requirements(dataset, osm_asset_mapping)
    selected_external_assets = tuple(dict.fromkeys(
        tuple(externally_selected_models)
        + tuple(forest_proxy_models)
        + tuple(external_ground_textures)
        + tuple(osm_asset_mapping_report.selected_models)
        + tuple(osm_asset_mapping_report.selected_textures)
    ))
    report_progress(82, "Scanning configured CWA asset roots and OSM asset mapping")
    asset_scan = scan_assets(
        spec.asset_roots,
        selected_external_assets,
        cache_dir=getattr(spec, "cache_dir", None),
        use_cache=bool(getattr(spec, "cache_enabled", True)),
        refresh=bool(getattr(spec, "cache_refresh", False)),
    )

    trusted_legacy_assets = _trusted_legacy_asset_paths(spec, milestone_number)
    trusted_legacy_set = set(trusted_legacy_assets)
    strict_selected_assets = tuple(
        asset for asset in selected_external_assets
        if canonical_asset_path(asset) not in trusted_legacy_set
    )
    report_progress(83, "Validating required model and texture dependencies")
    strict_asset_scan = scan_assets(
        spec.asset_roots,
        strict_selected_assets,
        cache_dir=getattr(spec, "cache_dir", None),
        use_cache=bool(getattr(spec, "cache_enabled", True)),
        refresh=bool(getattr(spec, "cache_refresh", False)),
    )
    if spec.strict_assets and not strict_asset_scan.verified:
        raise ValueError(
            "strict asset validation failed: "
            f"missing required assets={strict_asset_scan.missing_models}, "
            f"dependencies={strict_asset_scan.missing_dependencies}; "
            f"trusted legacy assets={trusted_legacy_assets}"
        )

    report_progress(84, "Preparing terrain texture table")
    texture_table_paths = [spec.dummy_texture_path, *ground_texture_paths]
    wrp_texture_indices = [index + 1 for index in material_indices]
    surface_texture_cache_hit = False
    surface_texture_cache_path: str | None = None
    if _surface_ground_enabled(spec):
        surface_texture_key = cache_key(
            "surface-texture-assets-v2-hq",
            {
                "world_name": spec.name,
                "profile": _ground_texture_profile(spec),
                "seed": spec.deterministic_seed,
                "size": int(getattr(spec, "surface_texture_size", 512)),
            },
        )
        surface_texture_bundle = cache_dir / "surface-assets" / surface_texture_key if cache_dir is not None else None
        surface_texture_names = tuple(f"data/{path.name}" for path in generated_material_texture_paths)
        surface_texture_cache_hit = restore_bundle(
            surface_texture_bundle, source_dir, surface_texture_names,
            enabled=cache_enabled, refresh=cache_refresh,
        )
        surface_texture_cache_path = str(surface_texture_bundle) if surface_texture_bundle is not None else None
        if not surface_texture_cache_hit:
            write_surface_textures(
                source_dir,
                spec.name,
                _ground_texture_profile(spec),
                spec.deterministic_seed,
                int(getattr(spec, "surface_texture_size", 512)),
            )
            store_bundle(surface_texture_bundle, source_dir, surface_texture_names, enabled=cache_enabled)
    elif _ground_texture_profile(spec) in {"generated", "desert", "malden"}:
        profile = _ground_texture_profile(spec)
        surface_texture_key = cache_key(
            "milestone8-ground-texture-assets-v1",
            {"world_name": spec.name, "profile": profile, "materials": [(m.code, material_colour_for_profile(m, profile)) for m in OSM_MATERIALS]},
        )
        surface_texture_bundle = cache_dir / "surface-assets" / surface_texture_key if cache_dir is not None else None
        surface_texture_names = tuple(f"data/{path.name}" for path in material_texture_paths)
        surface_texture_cache_hit = restore_bundle(
            surface_texture_bundle, source_dir, surface_texture_names,
            enabled=cache_enabled, refresh=cache_refresh,
        )
        surface_texture_cache_path = str(surface_texture_bundle) if surface_texture_bundle is not None else None
        if not surface_texture_cache_hit:
            for path, material in zip(material_texture_paths, OSM_MATERIALS):
                write_solid_dxt1_paa(path, colour=material_colour_for_profile(material, profile))
            store_bundle(surface_texture_bundle, source_dir, surface_texture_names, enabled=cache_enabled)
    report_progress(86, "Writing RVW4 world file")
    write_solid_dxt1_paa(dummy_texture_path, colour=(255, 0, 255))
    write_rvw4(
        wrp_path,
        spec.cells,
        spec.cells,
        elevations,
        wrp_texture_indices,
        texture_table_paths,
        all_objects,
        height_scale=spec.height_scale,
        renumber_object_ids=True,
    )

    surface_manifest: dict[str, object] | None = None
    overview_cache_hit = False
    overview_cache_path: str | None = None
    if _surface_pass_enabled(spec):
        report_progress(87, "Preparing overview map and world icon")
        assert surface_report is not None
        surface_manifest = surface_report.to_manifest()
        surface_manifest["ground_application_mode"] = _surface_ground_mode(spec)
        surface_manifest["wrp_material_cells"] = {
            material.code: material_indices.count(index)
            for index, material in enumerate(materials)
        }
        surface_manifest["wrp_texture_paths"] = list(ground_texture_paths)
        active_material_index = {material.code: index for index, material in enumerate(materials)}
        rocky_index = active_material_index.get("k", active_material_index.get("r", 0))
        surface_manifest["rocky_forest_texture_path"] = ground_texture_paths[rocky_index]
        surface_manifest["rocky_forest_material_cells"] = placement_surface_counts["rocky_forest_cells"]
        _write_json(surface_report_path, surface_manifest)
        overview_key = cache_key(
            "overview-assets-v2",
            {
                "surface": surface_cache_key,
                "terrain": terrain_cache_key,
                "size": int(getattr(spec, "surface_overview_size", 1024)),
                "towns": [asdict(town) for town in towns],
                "reference": file_snapshot(getattr(spec, "surface_colour_reference_path", None)),
            },
        )
        overview_bundle = cache_dir / "overview" / overview_key if cache_dir is not None else None
        overview_cache_path = str(overview_bundle) if overview_bundle is not None else None
        if (
            cache_enabled and not cache_refresh and overview_bundle is not None
            and (overview_bundle / "overview-map.png").is_file()
            and (overview_bundle / "overview.paa").is_file()
            and (overview_bundle / "icon.paa").is_file()
        ):
            overview_map_path.parent.mkdir(parents=True, exist_ok=True)
            overview_paa_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(overview_bundle / "overview-map.png", overview_map_path)
            shutil.copyfile(overview_bundle / "overview.paa", overview_paa_path)
            shutil.copyfile(overview_bundle / "icon.paa", world_icon_path)
            overview_cache_hit = True
        else:
            report_progress(87, "Rendering overview terrain and vector layers")
            overview = render_overview_map(
                overview_map_path,
                surface_report.indices,
                elevations,
                slopes,
                dataset,
                projection,
                spec.cells,
                int(getattr(spec, "surface_overview_size", 1024)),
                towns=towns,
                reference_path=getattr(spec, "surface_colour_reference_path", None),
                building_mask=raster.buildings,
                progress_callback=_scaled_progress_callback(87, 88),
            )
            report_progress(88, "Encoding overview texture and world icon")
            write_rgb_dxt1_paa(overview_paa_path, overview)
            write_world_icon(world_icon_path, overview)
            if cache_enabled and overview_bundle is not None:
                overview_bundle.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(overview_map_path, overview_bundle / "overview-map.png")
                shutil.copyfile(overview_paa_path, overview_bundle / "overview.paa")
                shutil.copyfile(world_icon_path, overview_bundle / "icon.paa")

    report_progress(89, "Writing configuration and intro mission files")
    animated_building_models = (
        tuple(
            asset.model_path for asset in building_generation.model_assets
            if asset.key.interiors
        )
        if building_generation is not None
        else ()
    )
    config_text = render_config(
        spec,
        milestone=milestone_number,
        town_names=towns,
        animated_building_models=animated_building_models,
    )
    validate_cwa_config(config_text)
    (source_dir / "config.cpp").write_text(config_text, encoding="ascii", newline="\n")
    mission_path.write_text(render_mission(spec, spawn_x=spawn.x, spawn_z=spawn.z, milestone=milestone_number), encoding="ascii", newline="\n")
    intro_mission_path.write_text(render_world_intro_mission(spec, spawn_x=spawn.x, spawn_z=spawn.z), encoding="ascii", newline="\n")
    intro_script_path.write_text(render_world_intro_script(spawn_x=spawn.x, spawn_z=spawn.z), encoding="ascii", newline="\n")

    report_progress(90, "Rendering build previews and diagnostics")
    _write_composite_preview(preview_path, spec.cells, spec.cells, elevations, slopes, material_indices, materials)
    _write_height_preview(height_preview_path, spec.cells, spec.cells, elevations)
    _write_material_preview(material_preview_path, spec.cells, spec.cells, material_indices, materials)
    _flip_png_vertical(preview_path)
    _flip_png_vertical(height_preview_path)
    _flip_png_vertical(material_preview_path)
    write_geography_preview(osm_preview_path, raster)
    if meadow_grass_preview_path is not None:
        write_meadow_grass_placement_preview(
            meadow_grass_preview_path,
            dataset,
            projection,
            raster,
            nonroads,
            spec,
            size=int(getattr(spec, "surface_overview_size", 1024)),
        )
    render_building_source_reference(
        building_source_reference_path,
        dataset,
        projection,
        building_placement_plans,
        int(getattr(spec, "surface_overview_size", 1024)),
    )
    osm_source_path.write_bytes(osm_bytes)
    osm_query_path.write_text(query_text, encoding="utf-8", newline="\n")
    attribution = attribution_text(spec)
    attribution_path.write_text(attribution, encoding="utf-8", newline="\n")
    mod_attribution_path.write_text(attribution, encoding="utf-8", newline="\n")
    write_asset_catalogue(
        asset_catalogue_path, asset_scan,
        osm_asset_mapping=osm_asset_mapping_report.to_manifest(),
    )
    _write_json(road_report_path, asdict(road_fit) | {"objects": len(road_fit.objects)})
    # Avoid duplicating every graded elevation in the human report.
    grading_doc = asdict(grading)
    grading_doc.pop("elevations", None)
    grading_doc["transitions"] = asdict(transitions) | {"indices": len(transitions.indices)}
    grading_doc["transitions"].pop("indices", None)
    _write_json(grading_report_path, grading_doc)
    solver_heightmap_path = None
    if isinstance(spec, ConstraintPlayabilitySpec):
        solver_heightmap_path = output_dir / "terrain-solved-meters.tif"
        _write_float_heightmap(solver_heightmap_path, elevations, spec.cells)
    report_progress(92, "Packing primary PBO")
    pbo_pack = pack_directory_cached(
        source_dir,
        pbo_path,
        cache_dir=cache_dir,
        enabled=cache_enabled,
        refresh=cache_refresh,
        backend=str(getattr(spec, "pbo_backend", "python")),
        poseidon_tools_path=getattr(spec, "poseidon_tools_path", None),
    )
    report_progress(94, "Verifying generated assets inside the world PBO")
    pbo_layout = _verify_single_world_pbo_layout(
        pbo_path, spec.name, infrastructure_generation
    )

    report_progress(95, "Calculating build fingerprints")
    heightmap_hash = _sha256(spec.heightmap_path)
    osm_hash = hashlib.sha256(osm_bytes).hexdigest()
    fingerprint = _generation_fingerprint(
        spec,
        elevations,
        material_indices,
        all_objects,
        towns,
        heightmap_sha256=heightmap_hash,
        osm_sha256=osm_hash,
    )
    verification_enabled = bool(getattr(spec, "verify_regeneration", False))
    reproducibility = {
        "schema": 2,
        "verification_enabled": verification_enabled,
        "deterministic_seed": spec.deterministic_seed,
        "heightmap_sha256": heightmap_hash,
        "osm_sha256": osm_hash,
        "generation_fingerprint": fingerprint,
    }
    if verification_enabled:
        report_progress(96, "Regenerating world for deterministic verification")
        if isinstance(spec, ConstraintPlayabilitySpec):
            from .terrain_solver import solve_terrain_constraints

            repeat_grade = solve_terrain_constraints(
                solver_input_elevations, dataset, projection, raster, spec,
                building_placement_plans=building_placement_plans,
            )
        else:
            repeat_grade = grade_terrain(solver_input_elevations, dataset, projection, raster, spec)
        repeat_elevations = quantize_elevations(repeat_grade.elevations, spec.height_scale)
        repeat_elevations, repeat_iterative_grounding, _repeat_terrain_key = _iterative_grounding_pass(
            repeat_elevations,
            dataset,
            projection,
            raster,
            spec,
            building_library,
            building_placement_plans,
            building_plans_truncated,
            initial_terrain_cache_key,
            dataset_identity,
        )
        repeat_grade = replace(repeat_grade, elevations=repeat_elevations)
        repeat_slopes = calculate_slopes(repeat_elevations, spec.cells, spec.cells, spec.cell_size)
        repeat_surface_report = (
            build_surface_pass(
                dataset,
                projection,
                raster,
                repeat_elevations,
                repeat_slopes,
                spec,
            )
            if _surface_pass_enabled(spec)
            else None
        )
        if spec.material_mask_path is None:
            repeat_base_material_indices = classify_materials(
                repeat_elevations,
                repeat_slopes,
                sea_level=spec.sea_level,
                beach_height=spec.beach_height,
                rock_height=spec.rock_height,
                rock_slope_degrees=spec.rock_slope_degrees,
            )
        else:
            repeat_base_material_indices = base_material_indices
        repeat_overlaid = overlay_materials(repeat_base_material_indices, raster)
        repeat_ground_transitions = improve_transitions(repeat_overlaid, raster, spec)
        if _surface_ground_enabled(spec):
            assert repeat_surface_report is not None
            repeat_material_indices = repeat_surface_report.indices
        else:
            repeat_material_indices = repeat_ground_transitions.indices
        repeat_building_library: ProceduralBuildingLibrary | None = None
        if building_library is not None:
            repeat_building_library = ProceduralBuildingLibrary(
                world_name=spec.name,
                width_quantum=building_library.width_quantum,
                length_quantum=building_library.length_quantum,
                height_quantum=building_library.height_quantum,
                minimum_width=building_library.minimum_width,
                maximum_width=building_library.maximum_width,
                minimum_length=building_library.minimum_length,
                maximum_length=building_library.maximum_length,
                minimum_height=building_library.minimum_height,
                maximum_height=building_library.maximum_height,
                default_level_height=building_library.default_level_height,
                maximum_variants=building_library.maximum_variants,
                roof_pitch_degrees=building_library.roof_pitch_degrees,
                foundation_depth=building_library.foundation_depth,
                maximum_foundation_depth=building_library.maximum_foundation_depth,
                foundation_depth_quantum=building_library.foundation_depth_quantum,
                church_plinth_height=building_library.church_plinth_height,
                generate_interiors=building_library.generate_interiors,
                high_quality_textures=building_library.high_quality_textures,
                texture_variants=building_library.texture_variants,
                house_style_preset=building_library.house_style_preset,
                cache_dir=building_library.cache_dir,
                cache_enabled=building_library.cache_enabled,
                cache_refresh=building_library.cache_refresh,
            )
            repeat_building_library.prepare(dataset, projection, spec.point_building_footprint)
        repeat_site_library: ProceduralSiteLibrary | None = None
        if site_library is not None:
            repeat_site_library = ProceduralSiteLibrary(
                site_library.world_name,
                site_library.maximum_variants,
                cache_dir=site_library.cache_dir,
                cache_enabled=site_library.cache_enabled,
                cache_refresh=site_library.cache_refresh,
            )
            repeat_site_library.prepare(dataset, projection)
        repeat_roads = fit_road_objects(dataset, projection, repeat_elevations, spec, starting_id=1)
        repeat_nonroads = generate_world_objects(
            dataset,
            projection,
            raster,
            repeat_elevations,
            spec,
            include_roads=False,
            starting_object_id=len(repeat_roads.objects) + 1,
            building_asset_library=repeat_building_library,
            building_placement_plans=building_placement_plans,
            building_plans_truncated=building_plans_truncated,
        )
        repeat_material_indices, repeat_surface_report, _repeat_surface_counts = _placement_driven_surface_overlay(
            repeat_material_indices, repeat_surface_report, repeat_nonroads, raster, spec, slopes=repeat_slopes
        )
        repeat_ground_transitions = (
            replace(repeat_ground_transitions, indices=repeat_material_indices)
            if hasattr(repeat_ground_transitions, "indices") else repeat_ground_transitions
        )
        repeat_forest_cluster_library: ProceduralForestClusterLibrary | None = None
        repeat_cluster_paths = tuple(
            obj.model_path for obj in repeat_nonroads.objects
            if milestone_number >= 9 and is_generated_cluster_model(spec.name, obj.model_path)
        )
        if repeat_cluster_paths:
            repeat_forest_cluster_library = ProceduralForestClusterLibrary(
                spec.name,
                proxy_profile=_forest_proxy_profile(spec),
                cache_dir=getattr(spec, "cache_dir", None),
                cache_enabled=bool(getattr(spec, "cache_enabled", True)),
                cache_refresh=bool(getattr(spec, "cache_refresh", False)),
            )
            repeat_forest_cluster_library.register_models(repeat_cluster_paths)

        repeat_infrastructure_library: ProceduralInfrastructureLibrary | None = None
        repeat_infrastructure_paths = tuple(
            obj.model_path for obj in (*repeat_roads.objects, *repeat_nonroads.objects)
            if milestone_number >= 9 and obj.model_path.casefold().startswith((spec.name + "\\i\\").casefold())
        )
        if repeat_infrastructure_paths:
            repeat_infrastructure_library = ProceduralInfrastructureLibrary(
                spec.name,
                road_segment_length=spec.road_segment_length,
                cache_dir=getattr(spec, "cache_dir", None),
                cache_enabled=bool(getattr(spec, "cache_enabled", True)),
                cache_refresh=bool(getattr(spec, "cache_refresh", False)),
            )
            repeat_infrastructure_library.register_models(repeat_infrastructure_paths)

        repeat_semantic = (
            generate_semantic_objects(
                dataset, projection, repeat_elevations, spec, repeat_site_library,
                starting_object_id=len(repeat_roads.objects) + len(repeat_nonroads.objects) + 1,
                raster=raster,
                building_placement_plans=building_placement_plans,
            )
            if repeat_site_library is not None
            else SemanticGenerationResult((), 0, 0, 0, 0, 0, 0.0)
        )
        repeat_objects = _assemble_world_objects(
            repeat_roads.objects, repeat_nonroads, repeat_semantic.objects, renumber=False
        )
        repeat_fingerprint = _generation_fingerprint(
            spec,
            repeat_elevations,
            repeat_material_indices,
            repeat_objects,
            town_locations(dataset, projection, spec.town_name_limit),
            heightmap_sha256=heightmap_hash,
            osm_sha256=osm_hash,
        )
        temp_dir = output_dir / ".regeneration-check"
        temp_source = temp_dir / spec.name
        temp_wrp = temp_source / f"{spec.name}.wrp"
        temp_source.mkdir(parents=True, exist_ok=True)
        write_rvw4(
            temp_wrp,
            spec.cells,
            spec.cells,
            repeat_elevations,
            [index + 1 for index in repeat_material_indices],
            texture_table_paths,
            repeat_objects,
            height_scale=spec.height_scale,
            renumber_object_ids=True,
        )
        shutil.copy2(source_dir / "config.cpp", temp_source / "config.cpp")
        generated_directories = {"g", "d"} if repeat_building_library is not None else set()
        if repeat_site_library is not None:
            generated_directories.add("s")
        if repeat_forest_cluster_library is not None:
            generated_directories.add("f")
        if repeat_infrastructure_library is not None:
            generated_directories.add("i")
        if _surface_pass_enabled(spec):
            generated_directories.add("data")
        for source_item in sorted(source_dir.iterdir(), key=lambda item: item.name.casefold()):
            if source_item.name in {"config.cpp", wrp_path.name} | generated_directories:
                continue
            target_item = temp_source / source_item.name
            if source_item.is_dir():
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)
        repeat_building_catalogue: Path | None = None
        if repeat_building_library is not None:
            repeat_building_catalogue = temp_dir / "building-asset-catalogue.json"
            repeat_building_library.write_assets(temp_source, repeat_building_catalogue)
        repeat_site_catalogue: Path | None = None
        if repeat_site_library is not None:
            repeat_site_catalogue = temp_dir / "semantic-site-catalogue.json"
            repeat_site_library.write_assets(temp_source, repeat_site_catalogue)
        repeat_forest_cluster_catalogue: Path | None = None
        if repeat_forest_cluster_library is not None:
            repeat_forest_cluster_catalogue = temp_dir / "forest-cluster-catalogue.json"
            repeat_forest_cluster_library.write_assets(
                temp_source, repeat_forest_cluster_catalogue
            )
        repeat_infrastructure_catalogue: Path | None = None
        if repeat_infrastructure_library is not None:
            repeat_infrastructure_catalogue = temp_dir / "infrastructure-asset-catalogue.json"
            repeat_infrastructure_library.write_assets(
                temp_source, repeat_infrastructure_catalogue
            )
        if _surface_ground_enabled(spec):
            write_surface_textures(
                temp_source,
                spec.name,
                _ground_texture_profile(spec),
                spec.deterministic_seed,
                int(getattr(spec, "surface_texture_size", 512)),
            )
        elif _ground_texture_profile(spec) in {"generated", "desert", "malden"}:
            profile = _ground_texture_profile(spec)
            for path, material in zip(
                (temp_source / "data" / f"{material.code}.paa" for material in OSM_MATERIALS),
                OSM_MATERIALS,
            ):
                write_solid_dxt1_paa(path, colour=material_colour_for_profile(material, profile))
        if _surface_pass_enabled(spec):
            assert repeat_surface_report is not None
            write_solid_dxt1_paa(temp_source / "data" / "d.paa", colour=(255, 0, 255))
            repeat_overview = render_overview_map(
                temp_dir / "overview-map.png",
                repeat_surface_report.indices,
                repeat_elevations,
                repeat_slopes,
                dataset,
                projection,
                spec.cells,
                int(getattr(spec, "surface_overview_size", 1024)),
                towns=town_locations(dataset, projection, spec.town_name_limit),
                reference_path=getattr(spec, "surface_colour_reference_path", None),
                building_mask=raster.buildings,
            )
            write_rgb_dxt1_paa(temp_source / "data" / "overview.paa", repeat_overview)
            write_world_icon(temp_source / "data" / "icon.paa", repeat_overview)
        report_progress(98, "Packing reproducibility PBO")
        temp_pbo = temp_dir / f"{spec.name}.pbo"
        pack_directory(
            temp_source,
            temp_pbo,
            backend=pbo_pack.backend,
            poseidon_tools_path=pbo_pack.poseidon_tools_path,
        )
        reproducibility.update({
            "verification_status": "completed",
            "deterministic_seed": spec.deterministic_seed,
            "heightmap_sha256": heightmap_hash,
            "osm_sha256": osm_hash,
            "generation_fingerprint": fingerprint,
            "repeat_generation_fingerprint": repeat_fingerprint,
            "pipeline_repeat_match": fingerprint == repeat_fingerprint,
            "iterative_grounding_repeat_match": (
                iterative_grounding == repeat_iterative_grounding
            ),
            "wrp_byte_match": _sha256(wrp_path) == _sha256(temp_wrp),
            "pbo_byte_match": _sha256(pbo_path) == _sha256(temp_pbo),
            "building_assets_byte_match": (
                building_generation is None
                or (
                    repeat_building_catalogue is not None
                    and _sha256(building_catalogue_path) == _sha256(repeat_building_catalogue)
                )
            ),
            "semantic_site_assets_byte_match": (
                site_generation is None
                or (
                    repeat_site_catalogue is not None
                    and _sha256(semantic_site_catalogue_path) == _sha256(repeat_site_catalogue)
                )
            ),
            "forest_cluster_assets_byte_match": (
                forest_cluster_generation is None
                or (
                    repeat_forest_cluster_catalogue is not None
                    and _sha256(forest_cluster_catalogue_path)
                    == _sha256(repeat_forest_cluster_catalogue)
                )
            ),
            "infrastructure_assets_byte_match": (
                infrastructure_generation is None
                or (
                    repeat_infrastructure_catalogue is not None
                    and _sha256(infrastructure_catalogue_path)
                    == _sha256(repeat_infrastructure_catalogue)
                )
            ),
            "surface_assets_byte_match": (
                not _surface_pass_enabled(spec)
                or all(
                    _sha256(source_dir / "data" / filename) == _sha256(temp_source / "data" / filename)
                    for filename in [
                        *(path.name for path in generated_material_texture_paths),
                        "d.paa",
                        "overview.paa",
                        "icon.paa",
                    ]
                )
            ),
        })
        _write_json(reproducibility_path, reproducibility)
        shutil.rmtree(temp_dir)
    else:
        reproducibility["verification_status"] = "skipped"
        _write_json(reproducibility_path, reproducibility)

    report_progress(99, "Finalizing build reports")
    result = BuildResult(
        output_dir=output_dir,
        source_dir=source_dir,
        wrp_path=wrp_path,
        texture_paths=texture_paths,
        pbo_path=pbo_path,
        mission_path=mission_path,
        intro_mission_path=intro_mission_path,
        intro_script_path=intro_script_path,
        preview_path=preview_path,
        height_preview_path=height_preview_path,
        material_preview_path=material_preview_path,
        manifest_path=manifest_path,
        report_path=report_path,
        osm_preview_path=osm_preview_path,
        osm_source_path=osm_source_path,
        osm_query_path=osm_query_path,
        attribution_path=attribution_path,
        asset_catalogue_path=asset_catalogue_path,
        road_report_path=road_report_path,
        grading_report_path=grading_report_path,
        reproducibility_path=reproducibility_path,
        meadow_grass_preview_path=meadow_grass_preview_path,
        solver_heightmap_path=solver_heightmap_path,
        building_catalogue_path=building_catalogue_path if building_generation is not None else None,
        forest_cluster_catalogue_path=(
            forest_cluster_catalogue_path if forest_cluster_generation is not None else None
        ),
        infrastructure_catalogue_path=(
            infrastructure_catalogue_path if infrastructure_generation is not None else None
        ),
        surface_report_path=surface_report_path if _surface_pass_enabled(spec) else None,
        building_source_reference_path=building_source_reference_path,
        overview_map_path=overview_map_path if _surface_pass_enabled(spec) else None,
        overview_paa_path=overview_paa_path if _surface_pass_enabled(spec) else None,
        world_icon_path=world_icon_path if _surface_pass_enabled(spec) else None,
        cache_report_path=cache_report_path,
    )

    counts = material_counts(material_indices, len(materials))
    spec_manifest = asdict(spec)
    spec_manifest["heightmap_path"] = str(spec.heightmap_path)
    spec_manifest["material_mask_path"] = str(spec.material_mask_path) if spec.material_mask_path else None
    spec_manifest["osm_json_path"] = str(spec.osm_json_path) if spec.osm_json_path else None
    spec_manifest["asset_roots"] = [str(path) for path in spec.asset_roots]
    spec_manifest["osm_asset_mapping_path"] = (
        str(spec.osm_asset_mapping_path) if getattr(spec, "osm_asset_mapping_path", None) else None
    )
    for cache_field in (
        "cache_dir", "cache_enabled", "cache_refresh", "pbo_backend", "poseidon_tools_path"
    ):
        spec_manifest.pop(cache_field, None)
    if isinstance(spec, ConstraintPlayabilitySpec):
        spec_manifest["out_of_bounds_dem_path"] = str(spec.out_of_bounds_dem_path) if spec.out_of_bounds_dem_path else None
    manifest = {
        "schema": 4,
        "milestone": 4,
        "generator": GENERATOR_VERSION,
        "world": spec_manifest,
        "height_scale": spec.height_scale,
        "heightmap": {
            "source_width": loaded.source_width,
            "source_height": loaded.source_height,
            "source_mode": loaded.source_mode,
            "source_minimum": loaded.source_minimum,
            "source_maximum": loaded.source_maximum,
            "mapping_minimum": loaded.mapping_minimum,
            "mapping_maximum": loaded.mapping_maximum,
            "clipped_low": loaded.clipped_low,
            "clipped_high": loaded.clipped_high,
            "source_grid": loaded.source_grid,
            "runtime_grid": loaded.runtime_grid,
            "legacy_centre_to_vertex_conversion": loaded.legacy_centre_to_vertex_conversion,
            "final_minimum_metres": min(elevations),
            "final_maximum_metres": max(elevations),
        },
        "iterative_grounding": {
            "mode": "six-stage-buildings-only",
            **asdict(iterative_grounding),
        },
        "material_mask": mask_metadata,
        "ground_texture_profile": _ground_texture_profile(spec),
        "surface_visual_pass": surface_manifest,
        "osm": {
            "bbox": spec.bbox,
            "source_generator": dataset.source_generator,
            "element_count": dataset.element_count,
            "source_width_metres": projection.source_width_metres,
            "source_height_metres": projection.source_height_metres,
            "scale_x": projection.scale_x,
            "scale_z": projection.scale_z,
            "feature_counts": {
                "coastlines": len(dataset.coastlines),
                "water": len(dataset.water),
                "forests": len(dataset.forests),
                "farmland": len(dataset.farmland),
                "urban": len(dataset.urban),
                "roads": len(dataset.roads),
                "building_polygons": len(dataset.building_polygons),
                "building_points": len(dataset.building_points),
                "places": len(dataset.places),
                "landmarks": len(dataset.landmarks),
                "sites": len(dataset.sites),
            },
            "raster_cell_counts": raster_counts(raster),
            "coastline_seed_count": raster.coastline_seed_count,
        },
        "pbo_layout": pbo_layout,
        "playability": {
            "road_fitting": {key: value for key, value in asdict(road_fit).items() if key != "objects"},
            "terrain_grading": grading_doc,
            "town_names": [asdict(town) for town in towns],
            "asset_catalogue": {
                **asset_scan.to_manifest(),
                "osm_asset_mapping": osm_asset_mapping_report.to_manifest(),
            },
            "strict_asset_validation": {
                **strict_asset_scan.to_manifest(),
                "trusted_legacy_assets": list(trusted_legacy_assets),
            },
            "procedural_buildings": building_generation.to_manifest() if building_generation else {"enabled": False},
            "procedural_forest_clusters": (
                forest_cluster_generation.to_manifest()
                if forest_cluster_generation
                else {"enabled": False}
            ),
            "semantic_features": {
                "bus_stop_objects": semantic.bus_stop_objects,
                "sports_pitch_objects": semantic.sports_pitch_objects,
                "parking_objects": semantic.parking_objects,
                "grave_objects": semantic.grave_objects,
                "cemetery_sites": semantic.cemetery_sites,
                "rejected_graves": semantic.rejected_graves,
                "rejected_site_relief": semantic.rejected_site_relief,
                "rejected_landmarks": semantic.rejected_landmarks,
                "maximum_site_relief": semantic.maximum_site_relief,
                "site_assets": (
                    {key: value for key, value in asdict(site_generation).items() if key not in {"cache_hits", "cache_misses"}}
                    if site_generation else None
                ),
            },
            "reproducibility": reproducibility,
        },
        "materials": [
            {
                "index": index,
                "wrp_texture_index": index + 1,
                "code": material.code,
                "name": material.name,
                "colour": material.colour,
                "cells": counts[index],
                "texture_path": ground_texture_paths[index],
            }
            for index, material in enumerate(materials)
        ],
        "spawn": asdict(spawn),
        "objects": {
            "total": len(all_objects),
            "wrp_serialization_order": [
                "roads",
                "forest_and_trees",
                "rural_vegetation_and_rocks",
                "buildings",
                "ditch_vegetation",
                "barriers",
                "bridges",
                "semantic_sites",
            ],
            "roads": len(road_fit.objects),
            "buildings": nonroads.building_objects,
            "building_final_footprint_plans": len(building_placement_plans),
            "building_road_nudges": nonroads.building_road_nudges,
            "residential_infill_objects": nonroads.residential_infill_objects,
            "residential_infill_areas": nonroads.residential_infill_areas,
            "building_foundation_rejections": nonroads.building_foundation_rejections,
            "building_interior_fallbacks": nonroads.building_interior_fallbacks,
            "building_fully_submerged_rejections": nonroads.building_fully_submerged_rejections,
            "maximum_building_pad_relief_metres": nonroads.maximum_building_pad_relief,
            "maximum_building_foundation_depth_metres": nonroads.maximum_building_foundation_depth,
            "forest": nonroads.forest_objects,
            "bus_stops": semantic.bus_stop_objects,
            "sports_pitches": semantic.sports_pitch_objects,
            "parking_lots": semantic.parking_objects,
            "cemeteries": semantic.cemetery_sites,
            "gravestones": semantic.grave_objects,
            "forest_road_rejections": nonroads.forest_road_rejections,
            "forest_slope_rejections": nonroads.forest_slope_rejections,
            "maximum_forest_relief_metres": nonroads.maximum_forest_relief,
            "forest_block_objects": nonroads.forest_block_objects,
            "forest_hillside_tree_objects": nonroads.forest_hillside_tree_objects,
            "forest_single_tree_objects": nonroads.forest_single_tree_objects,
            "forest_gap_infill_tree_objects": nonroads.forest_gap_infill_tree_objects,
            "forest_hillside_fallback_blocks": nonroads.forest_hillside_fallback_blocks,
            "forest_hillside_unfilled_blocks": nonroads.forest_hillside_unfilled_blocks,
            "forest_hillside_candidate_rejections": nonroads.forest_hillside_candidate_rejections,
            "forest_everon_steep_objects": nonroads.forest_everon_steep_objects,
            "forest_sunk_polygon_objects": nonroads.forest_sunk_polygon_objects,
            "forest_everon_steep_rejections": nonroads.forest_everon_steep_rejections,
            "forest_cluster_objects": nonroads.forest_cluster_objects,
            "forest_cluster_rejections": nonroads.forest_cluster_rejections,
            "forest_cluster_variant_counts": dict(nonroads.forest_cluster_variant_counts),
            "forest_cluster_maximum_burial_metres": nonroads.forest_cluster_maximum_burial,
            "forest_cluster_maximum_float_metres": nonroads.forest_cluster_maximum_float,
            "forest_undergrowth_objects": nonroads.forest_undergrowth_objects,
            "forest_undergrowth_rejections": nonroads.forest_undergrowth_rejections,
            "forest_undergrowth_maximum_burial_metres": nonroads.forest_undergrowth_maximum_burial,
            "forest_undergrowth_maximum_float_metres": nonroads.forest_undergrowth_maximum_float,
            "steep_hill_bush_objects": nonroads.steep_hill_bush_objects,
            "steep_hill_bush_rejections": nonroads.steep_hill_bush_rejections,
            "wetland_reed_objects": nonroads.wetland_reed_objects,
            "wetland_reed_rejections": nonroads.wetland_reed_rejections,
            "forest_border_objects": nonroads.forest_border_objects,
            "forest_border_rejections": nonroads.forest_border_rejections,
            "forest_border_maximum_burial_metres": nonroads.forest_border_maximum_burial,
            "forest_border_maximum_float_metres": nonroads.forest_border_maximum_float,
            "ditch_grass_objects": nonroads.ditch_grass_objects,
            "ditch_grass_rejections": nonroads.ditch_grass_rejections,
            "ditch_grass_maximum_burial_metres": nonroads.ditch_grass_maximum_burial,
            "ditch_grass_maximum_float_metres": nonroads.ditch_grass_maximum_float,
            "barrier_objects": nonroads.barrier_objects,
            "fence_objects": nonroads.fence_objects,
            "wall_objects": nonroads.wall_objects,
            "hedge_objects": nonroads.hedge_objects,
            "barrier_rejections": nonroads.barrier_rejections,
            "bridge_objects": nonroads.bridge_objects,
            "bridge_segments": nonroads.bridge_segments,
            "bridge_rejections": nonroads.bridge_rejections,
            "tree_row_objects": nonroads.tree_row_objects,
            "orchard_objects": nonroads.orchard_objects,
            "vineyard_objects": nonroads.vineyard_objects,
            "scrub_objects": nonroads.scrub_objects,
            "rural_rock_objects": nonroads.rural_rock_objects,
            "rural_vegetation_rejections": nonroads.rural_vegetation_rejections,
            "meadow_grass_objects": nonroads.meadow_grass_objects,
            "meadow_grass_rejections": nonroads.meadow_grass_rejections,
            "haybale_objects": nonroads.haybale_objects,
            "haybale_rejections": nonroads.haybale_rejections,
            "haybale_fields_total": nonroads.haybale_fields_total,
            "haybale_fields_selected": nonroads.haybale_fields_selected,
            "rocky_forest_objects": nonroads.rocky_forest_objects,
            "rocky_forest_rejections": nonroads.rocky_forest_rejections,
            "mapped_tree_objects": nonroads.mapped_tree_objects,
            "mapped_tree_rejections": nonroads.mapped_tree_rejections,
            "utility_objects": nonroads.utility_objects,
            "utility_rejections": nonroads.utility_rejections,
            "accepted_forest_material_cells": placement_surface_counts["accepted_forest_cells"],
            "rocky_forest_material_cells": placement_surface_counts["rocky_forest_cells"],
            "maximum_forest_burial_metres": nonroads.maximum_forest_burial,
            "maximum_forest_float_metres": nonroads.maximum_forest_float,
            "maximum_hillside_tree_relief_metres": nonroads.maximum_hillside_tree_relief,
            "grounding": {
                "terrain_quantized_before_object_placement": True,
                "terrain_sample_layout": loaded.runtime_grid,
                "source_heightmap_layout": loaded.source_grid,
                "legacy_centre_to_vertex_conversion": loaded.legacy_centre_to_vertex_conversion,
                "wrp_height_scale_metres": spec.height_scale,
                "maximum_solver_to_wrp_height_change_metres": max(
                    (abs(before - after) for before, after in zip(solver_elevations, elevations)),
                    default=0.0,
                ),
                "church_final_footprint_validation": True,
                "grave_model_specific_profiles": {
                    model: asdict(profile)
                    for model, profile in sorted(GRAVE_MODEL_GROUNDING_PROFILES.items())
                },
                "building_clearance_metres": spec.building_ground_clearance,
                "forest_clearance_metres": spec.forest_ground_clearance,
                "maximum_building_raise_metres": nonroads.maximum_building_grounding_raise,
                "maximum_forest_raise_metres": nonroads.maximum_forest_grounding_raise,
                "vegetation_audit": {
                    "tree_objects": nonroads.vegetation_audit_tree_objects,
                    "cluster_tree_proxies": nonroads.vegetation_audit_cluster_tree_proxies,
                    "cluster_bush_proxies": nonroads.vegetation_audit_cluster_bush_proxies,
                    "violations": nonroads.vegetation_audit_violations,
                    "maximum_tree_float_metres": nonroads.vegetation_audit_maximum_tree_float,
                    "maximum_bush_float_metres": nonroads.vegetation_audit_maximum_bush_float,
                    "tree_limit_metres": float(getattr(spec, "forest_cluster_tree_maximum_float", getattr(spec, "forest_single_tree_maximum_float", 0.15))),
                    "bush_limit_metres": float(getattr(spec, "forest_cluster_bush_maximum_float", 0.60)),
                },
            },
            "models": selected_models,
            "truncated": {
                "roads": road_fit.truncated,
                "buildings": nonroads.building_objects_truncated,
                "forest": nonroads.forest_objects_truncated,
            },
        },
        "outputs": {
            "source/config.cpp": _sha256(source_dir / "config.cpp"),
            f"source/{spec.name}.wrp": _sha256(wrp_path),
            **{f"source/data/{path.name}": _sha256(path) for path in texture_paths},
            **({
                f"source/{asset.relative_path}": asset.sha256
                for asset in building_generation.model_assets
            } if building_generation else {}),
            **({
                f"source/{relative}": _sha256(source_dir / relative)
                for relative in building_generation.texture_files
            } if building_generation else {}),
            **({"building-asset-catalogue.json": _sha256(building_catalogue_path)} if building_generation else {}),
            **({"semantic-site-catalogue.json": _sha256(semantic_site_catalogue_path)} if site_generation else {}),
            **({
                **{
                    f"source/{relative}": _sha256(source_dir / relative)
                    for relative in (
                        forest_cluster_generation.model_files
                        + forest_cluster_generation.texture_files
                    )
                },
                "forest-cluster-catalogue.json": _sha256(forest_cluster_catalogue_path),
            } if forest_cluster_generation else {}),
            **({
                **{
                    f"source/{relative}": _sha256(source_dir / relative)
                    for relative in infrastructure_generation.model_files + infrastructure_generation.texture_files
                },
                "infrastructure-asset-catalogue.json": _sha256(infrastructure_catalogue_path),
            } if infrastructure_generation else {}),
            f"{mod_directory_name}/Addons/{spec.name}.pbo": _sha256(pbo_path),
            f"Missions/test_mission.{spec.name}/mission.sqm": _sha256(mission_path),
            f"{mod_directory_name}/Anims/{WORLD_INTRO_NAME}.{spec.name}/mission.sqm": _sha256(intro_mission_path),
            f"{mod_directory_name}/Anims/{WORLD_INTRO_NAME}.{spec.name}/intro.sqs": _sha256(intro_script_path),
            "preview.png": _sha256(preview_path),
            "height-preview.png": _sha256(height_preview_path),
            "material-preview.png": _sha256(material_preview_path),
            "osm-geography-preview.png": _sha256(osm_preview_path),
            **({
                "meadow-grass-placement.png": _sha256(meadow_grass_preview_path)
            } if meadow_grass_preview_path is not None else {}),
            "building-source-reference.png": _sha256(building_source_reference_path),
            "osm-source.json": _sha256(osm_source_path),
            "overpass-query.txt": _sha256(osm_query_path),
            "OSM-ATTRIBUTION.txt": _sha256(attribution_path),
            f"{mod_directory_name}/OSM-ATTRIBUTION.txt": _sha256(mod_attribution_path),
            "asset-catalogue.json": _sha256(asset_catalogue_path),
            "road-fit-report.json": _sha256(road_report_path),
            "terrain-grading-report.json": _sha256(grading_report_path),
            **({"terrain-solved-meters.tif": _sha256(solver_heightmap_path)} if solver_heightmap_path else {}),
            "reproducibility-report.json": _sha256(reproducibility_path),
            **({
                "surface-pass-report.json": _sha256(surface_report_path),
                "overview-map.png": _sha256(overview_map_path),
                "source/data/overview.paa": _sha256(overview_paa_path),
                "source/data/icon.paa": _sha256(world_icon_path),
            } if _surface_pass_enabled(spec) else {}),
        },
    }
    cache_report = {
        "schema": 2,
        "generator": GENERATOR_VERSION,
        "enabled": cache_enabled,
        "refresh": cache_refresh,
        "directory": str(cache_dir) if cache_dir else None,
        "processed_dem": {"hit": dem_cache_hit, "key": dem_cache_key, "path": dem_cache_path},
        "osm_raster": {"hit": raster_cache_hit, "key": raster_cache_key, "path": raster_cache_path},
        "terrain_solution": {"hit": terrain_cache_hit, "key": terrain_cache_key, "path": terrain_cache_path},
        "surface_pipeline": {"hit": surface_cache_hit, "key": surface_cache_key, "path": surface_cache_path},
        "forest_and_building_placement": {
            "hit": placement_cache_hit, "key": placement_cache_key, "path": placement_cache_path,
            "forest_objects": nonroads.forest_objects,
            "building_objects": nonroads.building_objects,
        },
        "surface_textures": {"hit": surface_texture_cache_hit, "path": surface_texture_cache_path},
        "overview_assets": {"hit": overview_cache_hit, "path": overview_cache_path},
        "incremental_pbo": {
            "archive_hit": pbo_pack.archive_hit,
            "archive_key": pbo_pack.archive_key,
            "total_entries": pbo_pack.total_entries,
            "reused_blob_entries": pbo_pack.reused_blob_entries,
            "new_blob_entries": pbo_pack.new_blob_entries,
            "requested_backend": pbo_pack.requested_backend,
            "backend": pbo_pack.backend,
            "poseidon_tools_path": pbo_pack.poseidon_tools_path,
            "fallback_reason": pbo_pack.fallback_reason,
        },
        "asset_catalogue": asset_scan.cache_info(),
        "strict_asset_catalogue": strict_asset_scan.cache_info(),
        "procedural_assets": {
            "building_hits": building_generation.cache_hits if building_generation else 0,
            "building_misses": building_generation.cache_misses if building_generation else 0,
            "site_hits": site_generation.cache_hits if site_generation else 0,
            "site_misses": site_generation.cache_misses if site_generation else 0,
            "forest_cluster_hits": (
                forest_cluster_generation.cache_hits if forest_cluster_generation else 0
            ),
            "forest_cluster_misses": (
                forest_cluster_generation.cache_misses if forest_cluster_generation else 0
            ),
            "infrastructure_hits": infrastructure_generation.cache_hits if infrastructure_generation else 0,
            "infrastructure_misses": infrastructure_generation.cache_misses if infrastructure_generation else 0,
        },
    }
    _write_json(cache_report_path, cache_report)
    _write_json(manifest_path, manifest)
    try:
        lines = _validate_milestone4(
            result,
            spec,
            loaded,
            dataset,
            projection,
            raster,
            elevations,
            material_indices,
            spawn,
            generated,
            road_fit,
            grading,
            transitions,
            towns,
            asset_scan,
            strict_asset_scan,
            osm_asset_mapping_report,
            trusted_legacy_assets,
            reproducibility,
            building_generation,
            pbo_layout,
            mod_directory_name=mod_directory_name,
            milestone_number=milestone_number,
        )
        manifest["final_validation"] = {"status": "succeeded"}
    except Exception as exc:
        report_progress(99, "Final validation failed; preserving generated runtime")
        manifest["final_validation"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "deployment_allowed": True,
        }
        _write_json(manifest_path, manifest)
        lines = [
            "[FAIL] Final validation checks raised an exception",
            f"Reason: {type(exc).__name__}: {exc}",
            "Generated runtime preserved; deployment may still copy the PBO and intro files.",
            f"PBO SHA-256: {_sha256(pbo_path) if pbo_path.is_file() else 'missing'}",
        ]
    else:
        _write_json(manifest_path, manifest)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    record_build_ownership(output_dir, spec.name, manifest_path, merge=False)
    report_progress(100, "Build complete")
    return result
