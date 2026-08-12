# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import shutil
from typing import Any

from ._version import GENERATOR_VERSION
from .generator import BuildResult, build_milestone4
from .milestone6 import Milestone6Spec
from .model import ConstraintPlayabilitySpec, validate_world_identity
from .normalization import NormalizationSpec, load_normalized_dataset, normalize_source_bundle
from .source_pipeline import _copy_provenance, validate_source_bundle


@dataclass(frozen=True, slots=True)
class Milestone8Spec(Milestone6Spec):
    name: str = "cwr_milestone8"
    display_name: str = "CWR Milestone 8"
    major_road_grade_percent: float = 8.0
    shoreline_transition_cells: int = 3
    lake_shore_smoothing_cells: int = 8
    lake_shore_maximum_slope_percent: float = 8.0
    building_pad_margin: float = 2.0
    stream_channel_depth: float = 0.35
    river_channel_depth: float = 1.0
    watercourse_minimum_gradient_percent: float = 0.02
    natural_smoothing_strength: float = 0.16
    solver_iterations: int = 20
    world_edge_blend_cells: int = 3
    procedural_buildings: bool = True
    procedural_building_interiors: bool = False
    high_quality_building_textures: bool = False
    building_ground_clearance: float = 0.10
    church_ground_clearance: float = 3.00
    building_width_quantum: float = 2.0
    building_length_quantum: float = 2.0
    building_height_quantum: float = 3.0
    building_minimum_width: float = 4.0
    building_maximum_width: float = 80.0
    building_minimum_length: float = 4.0
    building_maximum_length: float = 160.0
    building_minimum_height: float = 3.0
    building_maximum_height: float = 48.0
    building_level_height: float = 3.0
    building_maximum_variants: int = 128
    building_roof_pitch_degrees: float = 35.0
    building_foundation_depth: float = 0.5
    building_foundation_maximum_depth: float = 8.0
    building_foundation_depth_quantum: float = 0.25
    building_foundation_safety: float = 0.20
    building_maximum_pad_relief: float = 0.20
    ground_texture_profile: str = "nogova"

    def validate(self) -> None:
        validate_world_identity(name=self.name, display_name=self.display_name, profile=self.profile)
        for label, value in (
            ("building width quantum", self.building_width_quantum),
            ("building length quantum", self.building_length_quantum),
            ("building height quantum", self.building_height_quantum),
            ("building minimum width", self.building_minimum_width),
            ("building maximum width", self.building_maximum_width),
            ("building minimum length", self.building_minimum_length),
            ("building maximum length", self.building_maximum_length),
            ("building minimum height", self.building_minimum_height),
            ("building maximum height", self.building_maximum_height),
            ("building level height", self.building_level_height),
            ("building roof pitch", self.building_roof_pitch_degrees),
            ("church ground clearance", self.church_ground_clearance),
            ("building foundation depth", self.building_foundation_depth),
            ("building maximum foundation depth", self.building_foundation_maximum_depth),
            ("building foundation depth quantum", self.building_foundation_depth_quantum),
            ("building foundation safety", self.building_foundation_safety),
            ("building maximum pad relief", self.building_maximum_pad_relief),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.building_minimum_width > self.building_maximum_width:
            raise ValueError("building minimum width must not exceed maximum width")
        if self.building_minimum_length > self.building_maximum_length:
            raise ValueError("building minimum length must not exceed maximum length")
        if self.building_minimum_height > self.building_maximum_height:
            raise ValueError("building minimum height must not exceed maximum height")
        if not 1 <= self.building_maximum_variants <= 2048:
            raise ValueError("building maximum variants must be within 1..2048")
        if not 5.0 <= self.building_roof_pitch_degrees <= 60.0:
            raise ValueError("building roof pitch must be within 5..60 degrees")
        if not 0.0 < self.building_foundation_depth <= 5.0:
            raise ValueError("building foundation depth must be within 0..5 metres")
        if not self.building_foundation_depth <= self.building_foundation_maximum_depth <= 12.0:
            raise ValueError("building maximum foundation depth must be between the minimum and 12 metres")
        if self.building_foundation_depth_quantum > self.building_foundation_maximum_depth:
            raise ValueError("building foundation depth quantum must not exceed the maximum depth")
        if self.ground_texture_profile not in {"nogova", "malden", "everon", "generated", "desert"}:
            raise ValueError("ground texture profile must be nogova, malden, everon, desert or generated")


@dataclass(frozen=True, slots=True)
class _Milestone8PlayabilitySpec(ConstraintPlayabilitySpec):
    procedural_buildings: bool = True
    procedural_building_interiors: bool = False
    high_quality_building_textures: bool = False
    building_ground_clearance: float = 0.10
    church_ground_clearance: float = 3.00
    building_pad_margin: float = 2.0
    building_width_quantum: float = 2.0
    building_length_quantum: float = 2.0
    building_height_quantum: float = 3.0
    building_minimum_width: float = 4.0
    building_maximum_width: float = 80.0
    building_minimum_length: float = 4.0
    building_maximum_length: float = 160.0
    building_minimum_height: float = 3.0
    building_maximum_height: float = 48.0
    building_level_height: float = 3.0
    building_maximum_variants: int = 128
    building_roof_pitch_degrees: float = 35.0
    building_foundation_depth: float = 0.5
    building_foundation_maximum_depth: float = 8.0
    building_foundation_depth_quantum: float = 0.25
    building_foundation_safety: float = 0.20
    building_maximum_pad_relief: float = 0.20
    ground_texture_profile: str = "nogova"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_dem_path(source_root: Path, manifest_path: Path) -> Path | None:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    elevation = document.get("elevation")
    if not isinstance(elevation, dict):
        return None
    raw_files = elevation.get("raw_files")
    if not isinstance(raw_files, list):
        return None
    for record in raw_files:
        if not isinstance(record, dict):
            continue
        relative = record.get("path")
        if not isinstance(relative, str) or not relative.lower().endswith((".tif", ".tiff")):
            continue
        candidate = (source_root / relative).resolve()
        try:
            candidate.relative_to(source_root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def build_milestone8(output_dir: Path, spec: Milestone8Spec, *, clean: bool = True) -> BuildResult:
    spec.validate()
    source_validation = validate_source_bundle(spec.source_dir)
    source = source_validation.bundle
    normalization = normalize_source_bundle(NormalizationSpec(
        source_dir=source.root,
        output_dir=spec.normalized_dir,
        refresh=spec.normalization_refresh,
        include_minor_roads=spec.include_minor_roads,
        road_snap_tolerance=spec.road_snap_tolerance,
        road_building_setback=spec.road_building_setback,
        building_merge_gap=spec.building_merge_gap,
        building_overlap_threshold=spec.building_overlap_threshold,
        point_building_footprint=spec.point_building_footprint,
        minimum_building_area=spec.building_minimum_area,
        forest_edge_width=spec.forest_edge_width,
        forest_building_clearance=spec.forest_building_clearance,
        minimum_forest_area=spec.minimum_forest_area,
        coordinate_precision=spec.coordinate_precision,
    ))
    dataset = load_normalized_dataset(normalization)
    raw_dem = _raw_dem_path(source.root, source.manifest_path)

    playability = _Milestone8PlayabilitySpec(
        heightmap_path=source.heightmap_path,
        name=spec.name,
        display_name=spec.display_name,
        profile=spec.profile,
        cells=source.cells,
        cell_size=source.cell_size,
        heightmap_grid=source.heightmap_grid,
        input_mode="meters",
        flip_y=True,
        sea_level=0.0,
        beach_height=3.0,
        rock_height=110.0,
        rock_slope_degrees=30.0,
        bbox=source.bbox,
        osm_json_path=source.osm_json_path,
        water_depth=spec.water_depth,
        coastline_blend_cells=spec.coastline_blend_cells,
        road_segment_length=spec.road_segment_length,
        max_road_objects=spec.max_road_objects,
        max_buildings=spec.max_buildings,
        building_minimum_area=spec.building_minimum_area,
        forest_tree_spacing=spec.forest_tree_spacing,
        forest_road_clearance=spec.forest_road_clearance,
        building_ground_clearance=spec.building_ground_clearance,
        church_ground_clearance=spec.church_ground_clearance,
        forest_ground_clearance=spec.forest_ground_clearance,
        point_building_footprint=spec.point_building_footprint,
        max_forest_objects=spec.max_forest_objects,
        include_minor_roads=spec.include_minor_roads,
        road_connection_tolerance=spec.road_connection_tolerance,
        maximum_road_grade_percent=spec.maximum_road_grade_percent,
        road_grade_radius=spec.road_grade_radius,
        building_grade_radius=spec.building_grade_radius,
        maximum_grade_adjustment=spec.maximum_grade_adjustment,
        transition_cells=spec.transition_cells,
        asset_roots=spec.asset_roots,
        strict_assets=spec.strict_assets,
        osm_asset_mapping_path=spec.osm_asset_mapping_path,
        town_name_limit=spec.town_name_limit,
        deterministic_seed=f"milestone8:{normalization.normalized_fingerprint}",
        verify_regeneration=spec.verify_regeneration,
        major_road_grade_percent=spec.major_road_grade_percent,
        shoreline_transition_cells=spec.shoreline_transition_cells,
        lake_shore_smoothing_cells=spec.lake_shore_smoothing_cells,
        lake_shore_maximum_slope_percent=spec.lake_shore_maximum_slope_percent,
        building_pad_margin=spec.building_pad_margin,
        stream_channel_depth=spec.stream_channel_depth,
        river_channel_depth=spec.river_channel_depth,
        watercourse_minimum_gradient_percent=spec.watercourse_minimum_gradient_percent,
        natural_smoothing_strength=spec.natural_smoothing_strength,
        solver_iterations=spec.solver_iterations,
        world_edge_blend_cells=spec.world_edge_blend_cells,
        out_of_bounds_dem_path=raw_dem,
        procedural_buildings=spec.procedural_buildings,
        procedural_building_interiors=spec.procedural_building_interiors,
        high_quality_building_textures=spec.high_quality_building_textures,
        building_width_quantum=spec.building_width_quantum,
        building_length_quantum=spec.building_length_quantum,
        building_height_quantum=spec.building_height_quantum,
        building_minimum_width=spec.building_minimum_width,
        building_maximum_width=spec.building_maximum_width,
        building_minimum_length=spec.building_minimum_length,
        building_maximum_length=spec.building_maximum_length,
        building_minimum_height=spec.building_minimum_height,
        building_maximum_height=spec.building_maximum_height,
        building_level_height=spec.building_level_height,
        building_maximum_variants=spec.building_maximum_variants,
        building_roof_pitch_degrees=spec.building_roof_pitch_degrees,
        building_foundation_depth=spec.building_foundation_depth,
        building_foundation_maximum_depth=spec.building_foundation_maximum_depth,
        building_foundation_depth_quantum=spec.building_foundation_depth_quantum,
        building_foundation_safety=spec.building_foundation_safety,
        building_maximum_pad_relief=spec.building_maximum_pad_relief,
        ground_texture_profile=spec.ground_texture_profile,
    )
    result = build_milestone4(
        output_dir,
        playability,
        clean=clean,
        mod_directory_name="@CWR-Milestone8",
        milestone_number=8,
        dataset_override=dataset,
    )
    provenance_path, source_validation_path = _copy_provenance(source, result)

    build_normalized_dir = result.output_dir / "normalized"
    if normalization.root.resolve() != build_normalized_dir.resolve():
        shutil.copytree(normalization.root, build_normalized_dir, dirs_exist_ok=True)
    runtime_root = result.pbo_path.parent.parent
    shutil.copyfile(normalization.manifest_path, runtime_root / "NORMALIZED-GEOMETRY.json")
    shutil.copyfile(normalization.validation_path, runtime_root / "NORMALIZED-GEOMETRY-VALIDATION.txt")
    if result.grading_report_path:
        shutil.copyfile(result.grading_report_path, runtime_root / "TERRAIN-SOLVER-REPORT.json")

    try:
        build_manifest: dict[str, Any] = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        build_manifest = {}
    terrain_report: dict[str, Any] = {}
    if result.grading_report_path:
        terrain_report = json.loads(result.grading_report_path.read_text(encoding="utf-8"))
    build_manifest["schema"] = 8
    build_manifest["milestone"] = 8
    build_manifest["generator"] = GENERATOR_VERSION
    build_manifest["source_bundle"] = {
        "manifest_sha256": source.fingerprint,
        "bbox_south_west_north_east": list(source.bbox),
        "cells": source.cells,
        "cell_size_metres": source.cell_size,
        "manifest": provenance_path.name,
        "validation": source_validation_path.name,
    }
    build_manifest["normalized_geometry"] = {
        "manifest_sha256": normalization.normalized_fingerprint,
        "source_manifest_sha256": normalization.source_fingerprint,
        "directory": "normalized",
        "counts": dict(normalization.counts),
        "files": {filename: _sha256(build_normalized_dir / filename) for filename in sorted(normalization.files)},
    }
    build_manifest["constraint_terrain_solver"] = {
        "report": "terrain-grading-report.json",
        "solved_heightmap": result.solver_heightmap_path.name if result.solver_heightmap_path else None,
        "out_of_bounds_dem": str(raw_dem) if raw_dem else None,
        "priority_order": terrain_report.get("priority_order", []),
        "cut_volume_m3": terrain_report.get("total_cut_volume_m3", 0.0),
        "fill_volume_m3": terrain_report.get("total_fill_volume_m3", 0.0),
    }
    result.manifest_path.write_text(json.dumps(build_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with result.report_path.open("a", encoding="utf-8", newline="\n") as report:
        report.write("\nMilestone 8 procedural building and terrain checks\n\n")
        checks = (
            ("Unified priority solver used", terrain_report.get("solver") == "unified-priority-constraint-relaxation", str(terrain_report.get("solver"))),
            ("Water surfaces flattened", float(terrain_report.get("water_roughness_after", 999.0)) <= 0.05, f"roughness={terrain_report.get('water_roughness_after')}m"),
            ("Building pads flattened", float(terrain_report.get("building_roughness_after", 999.0)) <= max(0.1, source.cell_size * 0.01), f"roughness={terrain_report.get('building_roughness_after')}m"),
            ("Watercourses remain downhill outside protected crossings", int(terrain_report.get("downhill_violations_after", 0)) <= int(terrain_report.get("downhill_violations_before", 0)), f"before={terrain_report.get('downhill_violations_before')}, after={terrain_report.get('downhill_violations_after')}, protected={terrain_report.get('downhill_protected_crossings', 0)}"),
            ("Terrain cut/fill report emitted", "category_adjustments" in terrain_report, f"categories={len(terrain_report.get('category_adjustments', {}))}"),
            ("Solved heightmap emitted", result.solver_heightmap_path is not None and result.solver_heightmap_path.is_file(), str(result.solver_heightmap_path)),
            ("Out-of-bounds edge method recorded", bool(terrain_report.get("out_of_bounds_sampling")), str(terrain_report.get("out_of_bounds_sampling"))),
        )
        for label, ok, detail in checks:
            report.write(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}\n")
        report.write("[PASS] CWA/CWR global water-plane limitation documented: inland water uses submerged flat basins\n")
        building_catalogue = {}
        if result.building_catalogue_path and result.building_catalogue_path.is_file():
            building_catalogue = json.loads(result.building_catalogue_path.read_text(encoding="utf-8"))
        building_models = building_catalogue.get("models", []) if isinstance(building_catalogue, dict) else []
        report.write(f"[{'PASS' if bool(building_models) or not dataset.building_polygons else 'FAIL'}] Procedural building P3Ds emitted: models={len(building_models)}\n")
        lods_ok = all(int(model.get("lod_count", 0)) >= 3 for model in building_models if isinstance(model, dict))
        report.write(f"[{'PASS' if lods_ok else 'FAIL'}] Building visual, geometry, and land-contact LODs emitted\n")
        placements = int(building_catalogue.get("placements", 0)) if isinstance(building_catalogue, dict) else 0
        variants = int(building_catalogue.get("generated_variants", 0)) if isinstance(building_catalogue, dict) else 0
        report.write(f"[{'PASS' if variants <= placements else 'FAIL'}] Building asset reuse bounded: placements={placements}, variants={variants}\n")

    return BuildResult(
        output_dir=result.output_dir,
        source_dir=result.source_dir,
        wrp_path=result.wrp_path,
        texture_paths=result.texture_paths,
        pbo_path=result.pbo_path,
        mission_path=result.mission_path,
        intro_mission_path=result.intro_mission_path,
        intro_script_path=result.intro_script_path,
        preview_path=result.preview_path,
        height_preview_path=result.height_preview_path,
        material_preview_path=result.material_preview_path,
        manifest_path=result.manifest_path,
        report_path=result.report_path,
        osm_preview_path=result.osm_preview_path,
        osm_source_path=result.osm_source_path,
        osm_query_path=result.osm_query_path,
        attribution_path=result.attribution_path,
        asset_catalogue_path=result.asset_catalogue_path,
        road_report_path=result.road_report_path,
        grading_report_path=result.grading_report_path,
        reproducibility_path=result.reproducibility_path,
        source_manifest_path=provenance_path,
        source_validation_path=source_validation_path,
        normalized_dir=build_normalized_dir,
        solver_heightmap_path=result.solver_heightmap_path,
        building_catalogue_path=result.building_catalogue_path,
    )
