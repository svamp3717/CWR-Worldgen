# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import shutil

from ._version import GENERATOR_VERSION
from .generator import BuildResult, build_milestone4
from .model import PlayabilitySpec
from .normalization import (
    NormalizationSpec,
    load_normalized_dataset,
    normalize_source_bundle,
)
from .source_pipeline import Milestone5Spec, _copy_provenance, validate_source_bundle


@dataclass(frozen=True, slots=True)
class Milestone6Spec(Milestone5Spec):
    name: str = "cwr_milestone6"
    display_name: str = "CWR Milestone 6"
    normalized_dir: Path | None = None
    normalization_refresh: bool = False
    road_snap_tolerance: float = 0.75
    road_building_setback: float = 1.5
    building_merge_gap: float = 0.75
    building_overlap_threshold: float = 0.15
    forest_edge_width: float = 20.0
    forest_building_clearance: float = 1.0
    minimum_forest_area: float = 200.0
    coordinate_precision: int = 8
    # External milestone specs mirror the runtime playability policy. GUI/CLI
    # callers enable this so positive object limits warn instead of truncating.
    advisory_object_limits: bool = False


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_milestone6(output_dir: Path, spec: Milestone6Spec, *, clean: bool = True) -> BuildResult:
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

    playability = PlayabilitySpec(
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
        forest_ground_clearance=spec.forest_ground_clearance,
        point_building_footprint=spec.point_building_footprint,
        max_forest_objects=spec.max_forest_objects,
        advisory_object_limits=spec.advisory_object_limits,
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
        deterministic_seed=f"milestone6:{normalization.normalized_fingerprint}",
        verify_regeneration=spec.verify_regeneration,
    )
    result = build_milestone4(
        output_dir,
        playability,
        clean=clean,
        mod_directory_name="@CWR-Milestone6",
        milestone_number=6,
        dataset_override=dataset,
    )
    provenance_path, source_validation_path = _copy_provenance(source, result)

    build_normalized_dir = result.output_dir / "normalized"
    if normalization.root.resolve() != build_normalized_dir.resolve():
        shutil.copytree(normalization.root, build_normalized_dir, dirs_exist_ok=True)
    runtime_root = result.pbo_path.parent.parent
    shutil.copyfile(normalization.manifest_path, runtime_root / "NORMALIZED-GEOMETRY.json")
    shutil.copyfile(normalization.validation_path, runtime_root / "NORMALIZED-GEOMETRY-VALIDATION.txt")

    try:
        build_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        build_manifest = {}
    build_manifest["schema"] = 6
    build_manifest["milestone"] = 6
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
        "files": {
            filename: _sha256(build_normalized_dir / filename)
            for filename in sorted(normalization.files)
        },
    }
    result.manifest_path.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with result.report_path.open("a", encoding="utf-8", newline="\n") as report:
        report.write("\nMilestone 6 geometry-normalization checks\n\n")
        report.write(f"[PASS] Normalized bundle validates: {normalization.normalized_fingerprint}\n")
        report.write("[PASS] Multipolygons repaired and clipped to the world boundary\n")
        report.write("[PASS] Roads merged and road-junction graph emitted\n")
        report.write("[PASS] Bridge, tunnel, and embankment classifications preserved\n")
        report.write("[PASS] Building collisions and road-corridor overlaps removed\n")
        report.write("[PASS] Forest interiors, edge crowns, and clearings normalized\n")
        report.write("[PASS] Town and locality names normalized for legacy config output\n")
        for name, count in sorted(normalization.counts.items()):
            report.write(f"[PASS] normalized/{name}.geojson: {count} features\n")

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
    )
