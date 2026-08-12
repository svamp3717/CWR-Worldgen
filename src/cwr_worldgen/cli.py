# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import sys

from ._version import GENERATOR_VERSION
from .generator import build_milestone1, build_milestone2, build_milestone3, build_milestone4
from .model import (
    DEFAULT_MAX_BUILDINGS,
    DEFAULT_MAX_FOREST_OBJECTS,
    DEFAULT_MAX_ROAD_OBJECTS,
    HeightmapSpec,
    OsmSpec,
    PlayabilitySpec,
    WorldSpec,
)
from .milestone6 import Milestone6Spec, build_milestone6
from .milestone7 import Milestone7Spec, build_milestone7
from .milestone8 import Milestone8Spec, build_milestone8
from .milestone9 import Milestone9Spec, build_milestone9
from .normalization import NormalizationSpec, normalize_source_bundle, validate_normalized_bundle
from .progress import format_duration, progress_elapsed_seconds, start_progress_session
from .terrain import GROUND_TEXTURE_PROFILES
from .source_pipeline import (
    Milestone5Spec,
    SourceFetchSpec,
    SourceRegridSpec,
    build_milestone5,
    fetch_sources,
    regrid_sources,
    validate_source_bundle,
)


def _add_common_world_arguments(
    parser: argparse.ArgumentParser, *, default_name: str, default_display_name: str, include_grid: bool = True, default_profile: str = "cwa"
) -> None:
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--name", default=default_name, help="3-20 character lowercase world and PBO root name")
    parser.add_argument("--display-name", default=default_display_name, help="name shown in the island list")
    parser.add_argument("--profile", choices=("cwa", "cwr-ce"), default=default_profile)
    if include_grid:
        parser.add_argument("--cells", type=int, default=256, help="power-of-two terrain grid size")
        parser.add_argument("--cell-size", type=float, default=25.0, help="terrain cell size in metres")
    parser.add_argument("--keep-output", action="store_true", help="do not delete the output directory first")


def _add_heightmap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--heightmap", type=Path, required=True, help="single-channel PNG or TIFF")
    parser.add_argument(
        "--input-mode",
        choices=("normalized", "meters"),
        default="normalized",
        help="map source samples to an elevation range, or treat them as metres",
    )
    parser.add_argument("--elevation-min", type=float, default=-10.0, help="metres for normalized sample zero")
    parser.add_argument("--elevation-max", type=float, default=250.0, help="metres for normalized sample maximum")
    parser.add_argument("--input-min", type=float, help="override the source sample minimum")
    parser.add_argument("--input-max", type=float, help="override the source sample maximum")
    parser.add_argument("--material-mask", type=Path, help="optional PNG/TIFF base material mask")
    parser.add_argument(
        "--heightmap-grid",
        choices=("game-cell-centres", "game-terrain-vertices"),
        default="game-cell-centres",
        help="source sample layout; legacy heightmaps use cell centres",
    )
    parser.add_argument("--flip-y", action="store_true", help="flip source images vertically before resampling")
    parser.add_argument("--sea-level", type=float, default=0.0)
    parser.add_argument("--beach-height", type=float, default=4.0)
    parser.add_argument("--rock-height", type=float, default=140.0)
    parser.add_argument("--rock-slope", type=float, default=28.0, help="degrees")
    parser.add_argument("--spawn-clearance", type=float, default=1.0, help="metres above sea level")
    parser.add_argument("--spawn-max-slope", type=float, default=18.0, help="degrees")


def _heightmap_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "heightmap_path": args.heightmap,
        "name": args.name,
        "display_name": args.display_name,
        "profile": args.profile,
        "cells": args.cells,
        "cell_size": args.cell_size,
        "heightmap_grid": args.heightmap_grid,
        "input_mode": args.input_mode,
        "elevation_minimum": args.elevation_min,
        "elevation_maximum": args.elevation_max,
        "input_minimum": args.input_min,
        "input_maximum": args.input_max,
        "material_mask_path": args.material_mask,
        "flip_y": args.flip_y,
        "sea_level": args.sea_level,
        "beach_height": args.beach_height,
        "rock_height": args.rock_height,
        "rock_slope_degrees": args.rock_slope,
        "spawn_clearance": args.spawn_clearance,
        "maximum_spawn_slope_degrees": args.spawn_max_slope,
    }


def _add_osm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("SOUTH", "WEST", "NORTH", "EAST"), required=True, help="OpenStreetMap bounding box mapped onto the square WRP world")
    parser.add_argument("--osm-json", type=Path, help="saved Overpass JSON; omit to fetch the bbox from Overpass")
    parser.add_argument("--overpass-url", default="https://overpass-api.de/api/interpreter", help="Overpass API interpreter endpoint")
    parser.add_argument("--overpass-timeout", type=int, default=90, help="Overpass query timeout in seconds")
    parser.add_argument("--water-depth", type=float, default=5.0, help="metres below sea level for OSM water")
    parser.add_argument("--coast-blend-cells", type=int, default=2, help="shore smoothing distance")
    parser.add_argument("--road-segment-length", type=float, default=24.5, help="road model spacing in metres")
    parser.add_argument("--max-road-objects", type=int, default=DEFAULT_MAX_ROAD_OBJECTS, help=f"complete stock-road object ceiling (default: {DEFAULT_MAX_ROAD_OBJECTS:,})")
    parser.add_argument("--max-buildings", type=int, default=DEFAULT_MAX_BUILDINGS, help=f"maximum placed building footprints (default: {DEFAULT_MAX_BUILDINGS:,})")
    parser.add_argument("--building-min-area", type=float, default=20.0, help="minimum OSM footprint area in world m2")
    parser.add_argument("--forest-tree-spacing", type=float, default=50.0, help="classic stock forest-block spacing in metres")
    parser.add_argument("--forest-road-clearance", type=float, default=0.0, help="extra clearance beyond the mapped road edge; zero still rejects tree footprints that touch the road")
    parser.add_argument("--building-ground-clearance", type=float, default=0.10, help="small visible foundation reveal above the highest final model-footprint terrain")
    parser.add_argument("--forest-ground-clearance", type=float, default=0.15, help="vertical clearance above the highest sampled forest-block terrain")
    parser.add_argument("--point-building-footprint", type=float, default=12.0, help="assumed square footprint for OSM building nodes in metres")
    parser.add_argument("--max-forest-objects", type=int, default=DEFAULT_MAX_FOREST_OBJECTS, help=f"maximum placed primary forest and tree objects (default: {DEFAULT_MAX_FOREST_OBJECTS:,})")
    parser.add_argument("--include-minor-roads", action="store_true", help="also import paths, footways, cycleways, bridleways, and pedestrian ways")


def _osm_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        **_heightmap_kwargs(args),
        "bbox": tuple(args.bbox),
        "osm_json_path": args.osm_json,
        "overpass_url": args.overpass_url,
        "overpass_timeout_seconds": args.overpass_timeout,
        "water_depth": args.water_depth,
        "coastline_blend_cells": args.coast_blend_cells,
        "road_segment_length": args.road_segment_length,
        "max_road_objects": args.max_road_objects,
        "max_buildings": args.max_buildings,
        "building_minimum_area": args.building_min_area,
        "forest_tree_spacing": args.forest_tree_spacing,
        "forest_road_clearance": args.forest_road_clearance,
        "building_ground_clearance": args.building_ground_clearance,
        "forest_ground_clearance": args.forest_ground_clearance,
        "point_building_footprint": args.point_building_footprint,
        "max_forest_objects": args.max_forest_objects,
        "include_minor_roads": args.include_minor_roads,
    }


def _add_playability_arguments(parser: argparse.ArgumentParser, *, default_seed: str | None = None) -> None:
    parser.add_argument("--road-connection-tolerance", type=float, default=5.0, help="maximum uncovered road-chain or junction gap in metres")
    parser.add_argument("--maximum-road-grade", type=float, default=12.0, help="maximum fitted road grade in percent")
    parser.add_argument("--road-grade-radius", type=float, default=100.0, help="terrain grading radius around roads in metres")
    parser.add_argument("--building-grade-radius", type=float, default=25.0, help="terrain grading radius around buildings in metres")
    parser.add_argument("--maximum-grade-adjustment", type=float, default=12.0, help="maximum terrain cut or fill in metres")
    parser.add_argument("--transition-cells", type=int, default=2, help="material transition width in cells")
    parser.add_argument("--asset-root", type=Path, action="append", default=[], help="game/addon directory or PBO to scan; repeatable")
    parser.add_argument("--strict-assets", action="store_true", help="fail when required selected models, external ground textures, or readable P3D dependencies are missing; Milestone 8 trusts its inherited Milestone 7 road references")
    parser.add_argument("--osm-asset-map", type=Path, help="JSON rules mapping OSM layers and tags to required P3D models and PAA/PAC textures; built-in current mappings remain the default")
    parser.add_argument("--cache-dir", type=Path, help="persistent cache directory; defaults to SOURCE_DIR/.cwr-cache for frozen-source builds")
    parser.add_argument("--no-cache", action="store_true", help="disable all persistent pipeline caches, including DEM, terrain, placement, surfaces, overview, and PBO reuse")
    parser.add_argument("--cache-refresh", action="store_true", help="ignore matching cache entries and replace them")
    parser.add_argument("--verify-regeneration", action="store_true", help="run a second full generation and compare deterministic WRP/PBO output; slow and disabled by default")
    parser.add_argument("--town-name-limit", type=int, default=64)
    if default_seed is not None:
        parser.add_argument("--deterministic-seed", default=default_seed)


def _add_source_feature_arguments(
    parser: argparse.ArgumentParser, *, include_minor_roads_default: bool = False
) -> None:
    parser.add_argument("--water-depth", type=float, default=5.0, help="metres below sea level for OSM water")
    parser.add_argument("--coast-blend-cells", type=int, default=2, help="shore smoothing distance")
    parser.add_argument("--road-segment-length", type=float, default=24.5, help="road model spacing in metres")
    parser.add_argument("--max-road-objects", type=int, default=DEFAULT_MAX_ROAD_OBJECTS, help=f"complete stock-road object ceiling (default: {DEFAULT_MAX_ROAD_OBJECTS:,})")
    parser.add_argument("--max-buildings", type=int, default=DEFAULT_MAX_BUILDINGS, help=f"maximum placed building footprints (default: {DEFAULT_MAX_BUILDINGS:,})")
    parser.add_argument("--building-min-area", type=float, default=20.0, help="minimum OSM footprint area in world m2")
    parser.add_argument("--forest-tree-spacing", type=float, default=50.0, help="classic stock forest-block spacing in metres")
    parser.add_argument("--forest-road-clearance", type=float, default=0.0, help="extra clearance beyond mapped road edges; zero still rejects tree footprints that touch roads")
    parser.add_argument("--building-ground-clearance", type=float, default=0.10, help="small visible foundation reveal above the highest final model-footprint terrain")
    parser.add_argument("--forest-ground-clearance", type=float, default=0.15, help="vertical clearance above the highest sampled forest-block terrain")
    parser.add_argument("--point-building-footprint", type=float, default=12.0, help="assumed square footprint for OSM building nodes in metres")
    parser.add_argument("--max-forest-objects", type=int, default=DEFAULT_MAX_FOREST_OBJECTS, help=f"maximum placed primary forest and tree objects (default: {DEFAULT_MAX_FOREST_OBJECTS:,})")
    parser.add_argument("--include-minor-roads", action="store_true", dest="include_minor_roads", help="include service, track and other minor OSM roads")
    parser.add_argument("--no-minor-roads", action="store_false", dest="include_minor_roads", help="exclude service, track and other minor OSM roads")
    parser.set_defaults(include_minor_roads=include_minor_roads_default)




def _add_normalization_arguments(parser: argparse.ArgumentParser, *, include_minor_flag: bool) -> None:
    parser.add_argument("--normalized-dir", type=Path, help="derived normalized GeoJSON directory; defaults to SOURCE_DIR/normalized")
    parser.add_argument("--normalization-refresh", action="store_true", help="rebuild normalized geometry even when its source fingerprint matches")
    parser.add_argument("--road-snap-tolerance", type=float, default=0.75, help="endpoint snapping grid in metres before connected-road merging")
    parser.add_argument("--road-building-setback", type=float, default=1.5, help="extra building clearance beyond each mapped road edge")
    parser.add_argument("--building-merge-gap", type=float, default=0.75, help="maximum gap for merging adjacent small footprints")
    parser.add_argument("--building-overlap-threshold", type=float, default=0.15, help="fractional overlap above which the smaller footprint is removed")
    parser.add_argument("--forest-edge-width", type=float, default=20.0, help="forest edge crown width in metres")
    parser.add_argument("--forest-building-clearance", type=float, default=1.0, help="forest clearing around cleaned buildings in metres")
    parser.add_argument("--minimum-forest-area", type=float, default=200.0, help="discard normalized forest fragments below this area")
    parser.add_argument("--coordinate-precision", type=int, default=8, help="WGS84 GeoJSON decimal places")
    if include_minor_flag:
        parser.add_argument("--include-minor-roads", action="store_true")


def _normalization_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "output_dir": args.normalized_dir,
        "refresh": args.normalization_refresh,
        "include_minor_roads": args.include_minor_roads,
        "road_snap_tolerance": args.road_snap_tolerance,
        "road_building_setback": args.road_building_setback,
        "building_merge_gap": args.building_merge_gap,
        "building_overlap_threshold": args.building_overlap_threshold,
        "point_building_footprint": args.point_building_footprint,
        "minimum_building_area": args.building_min_area,
        "forest_edge_width": args.forest_edge_width,
        "forest_building_clearance": args.forest_building_clearance,
        "minimum_forest_area": args.minimum_forest_area,
        "coordinate_precision": args.coordinate_precision,
    }



def _add_procedural_building_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--procedural-building-interiors",
        action="store_true",
        help="generate enterable ground-floor interiors for eligible procedural buildings",
    )
    parser.add_argument(
        "--high-quality-building-textures",
        action="store_true",
        help="use optional 256px procedural building textures instead of the default legacy-style 128px textures",
    )
    parser.add_argument("--building-width-quantum", type=float, default=2.0, help="width reuse bucket in metres")
    parser.add_argument("--building-length-quantum", type=float, default=2.0, help="length reuse bucket in metres")
    parser.add_argument("--building-height-quantum", type=float, default=3.0, help="height reuse bucket in metres")
    parser.add_argument("--building-min-width", type=float, default=4.0)
    parser.add_argument("--building-max-width", type=float, default=80.0)
    parser.add_argument("--building-min-length", type=float, default=4.0)
    parser.add_argument("--building-max-length", type=float, default=160.0)
    parser.add_argument("--building-min-height", type=float, default=3.0)
    parser.add_argument("--building-max-height", type=float, default=48.0)
    parser.add_argument("--building-level-height", type=float, default=3.0)
    parser.add_argument("--building-max-variants", type=int, default=128, help="maximum generated P3D variants")
    parser.add_argument("--building-roof-pitch", type=float, default=35.0, help="gabled roof pitch in degrees")
    parser.add_argument("--church-ground-clearance", type=float, default=3.00, help="minimum hidden foundation-skirt depth for large churches on uneven terrain")
    parser.add_argument("--building-foundation-depth", type=float, default=0.5, help="minimum per-building stone foundation skirt depth in metres")
    parser.add_argument("--building-foundation-max-depth", type=float, default=8.0, help="normal foundation limit; enterable buildings exceeding it use a non-enterable variant instead of being rejected")
    parser.add_argument("--building-foundation-depth-quantum", type=float, default=0.25, help="foundation-depth reuse bucket in metres")
    parser.add_argument("--building-foundation-safety", type=float, default=0.20, help="extra buried foundation below the lowest sampled footprint terrain")
    parser.add_argument("--building-max-pad-relief", type=float, default=0.20, help="target maximum relief across a final building footprint after grading")




def _add_surface_pass_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--surface-wet-shore-cells", type=int, default=1, help="wet shoreline band width in terrain cells")
    parser.add_argument("--surface-sand-shore-cells", type=int, default=2, help="dry sand shoreline band width in terrain cells")
    parser.add_argument("--surface-transition-cells", type=int, default=2, help="outer deterministic material-dither width in terrain cells")
    parser.add_argument("--surface-forest-edge-cells", type=int, default=1, help="forest-edge ground band width in terrain cells")
    parser.add_argument("--surface-farmland-strip-cells", type=int, default=4, help="deterministic field subdivision stripe width")
    parser.add_argument("--surface-road-shoulder", type=float, default=5.0, help="paved-road shoulder width in metres")
    parser.add_argument("--surface-dirt-blend", type=float, default=6.0, help="dirt-road blend width in metres")
    parser.add_argument("--surface-steep-slope", type=float, default=38.0, help="steep rock/scree threshold in degrees")
    parser.add_argument("--colour-reference", type=Path, help="optional satellite or topographic PNG/TIFF colour reference; defaults to the frozen reference map when available")
    parser.add_argument("--colour-reference-strength", type=float, default=0.25, help="natural-material colour guidance within 0..1")
    parser.add_argument("--overview-size", type=int, default=1024, help="power-of-two overview map size")
    parser.add_argument("--surface-texture-size", type=int, default=512, help="power-of-two generated ground texture size when Milestone 9 ground application is enabled")
    parser.add_argument(
        "--surface-ground-mode",
        choices=("milestone8", "milestone9"),
        default="milestone9",
        help="write the expanded Milestone 9 WRP palette by default; milestone8 retains the legacy eight-class palette",
    )

def _add_constraint_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--major-road-grade", type=float, default=8.0, help="maximum longitudinal grade for major roads in percent")
    parser.add_argument("--shoreline-transition-cells", type=int, default=3, help="base protected terrain ramp for a 6.4 km world; larger worlds scale it proportionally")
    parser.add_argument("--lake-shore-smoothing-cells", type=int, default=8, help="base inland-lake bank smoothing width for a 6.4 km world")
    parser.add_argument("--lake-shore-max-slope", type=float, default=8.0, help="maximum intended inland-lake bank rise in percent")
    parser.add_argument("--building-pad-margin", type=float, default=2.0, help="extra flat pad around final selected building model footprints in metres")
    parser.add_argument("--stream-channel-depth", type=float, default=0.35, help="stream/ditch terrain carving depth in metres")
    parser.add_argument("--river-channel-depth", type=float, default=1.0, help="river/canal terrain carving depth in metres")
    parser.add_argument("--watercourse-minimum-gradient", type=float, default=0.02, help="minimum downstream fall in percent")
    parser.add_argument("--natural-smoothing-strength", type=float, default=0.16, help="base 6.4 km per-iteration relaxation strength within 0..1; larger worlds scale diffusion automatically")
    parser.add_argument("--solver-iterations", type=int, default=20, help="base unified constraint-relaxation iterations")
    parser.add_argument("--world-edge-blend-cells", type=int, default=3, help="base 6.4 km out-of-bounds terrain blending width")


def _add_fetch_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path, required=True, help="frozen source bundle directory")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--map-url", help="OpenTopoMap URL ending in #map=ZOOM/LAT/LON")
    selection.add_argument("--center", type=float, nargs=2, metavar=("LAT", "LON"))
    selection.add_argument("--bbox", type=float, nargs=4, metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    parser.add_argument("--cells", type=int, default=256)
    parser.add_argument("--cell-size", type=float, default=25.0, help="metres")
    parser.add_argument("--refresh", action="store_true", help="replace the frozen snapshot explicitly")
    parser.add_argument("--reference-map", action="store_true", help="also freeze an OpenTopoMap comparison image")
    parser.add_argument("--dem-provider", choices=("dem-stitcher", "hgt"), default="dem-stitcher")
    parser.add_argument("--dem-name", default="glo_30", help="dem-stitcher dataset shortname")
    parser.add_argument("--overpass-url", action="append", default=[], help="Overpass interpreter endpoint; repeatable")
    parser.add_argument("--overpass-timeout", type=int, default=240)
    parser.add_argument("--hgt-url-template", action="append", default=[], help="HGT mirror template containing {latitude_band} and {tile}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cwr-worldgen", description="Generate OFP/CWA-compatible worlds")
    parser.add_argument("--version", action="version", version=GENERATOR_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    milestone1 = subparsers.add_parser("milestone1", help="build the deterministic self-contained flat world")
    _add_common_world_arguments(
        milestone1, default_name="cwr_milestone1", default_display_name="CWR Milestone 1"
    )

    milestone2 = subparsers.add_parser(
        "milestone2", help="build a self-contained world from a PNG/TIFF heightmap"
    )
    _add_common_world_arguments(
        milestone2, default_name="cwr_milestone2", default_display_name="CWR Milestone 2"
    )
    _add_heightmap_arguments(milestone2)

    milestone3 = subparsers.add_parser(
        "milestone3",
        help="add OpenStreetMap coastline, water, land use, roads, forests, and buildings",
    )
    _add_common_world_arguments(
        milestone3, default_name="cwr_milestone3", default_display_name="CWR Milestone 3"
    )
    _add_heightmap_arguments(milestone3)
    _add_osm_arguments(milestone3)

    milestone4 = subparsers.add_parser(
        "milestone4",
        help="add road fitting, terrain grading, asset scans, town names, transitions, and regeneration checks",
    )
    _add_common_world_arguments(
        milestone4, default_name="cwr_milestone4", default_display_name="CWR Milestone 4"
    )
    _add_heightmap_arguments(milestone4)
    _add_osm_arguments(milestone4)
    _add_playability_arguments(milestone4, default_seed="cwr-worldgen-milestone4")

    fetch = subparsers.add_parser("fetch-sources", help="download and freeze OSM and elevation inputs for offline builds")
    _add_fetch_source_arguments(fetch)

    regrid = subparsers.add_parser(
        "regrid-sources",
        help="create a new frozen source bundle at another terrain grid without network access",
    )
    regrid.add_argument("--source-dir", type=Path, required=True, help="existing validated frozen source bundle")
    regrid.add_argument("--output-source-dir", type=Path, required=True, help="new frozen source bundle directory")
    regrid.add_argument("--cells", type=int, default=None, help="target power-of-two terrain grid; inferred from world size when omitted")
    regrid.add_argument("--cell-size", type=float, default=None, help="target terrain cell size in metres (default: 25; inferred from --cells when supplied alone)")
    regrid.add_argument("--replace-output", action="store_true", help="atomically replace an existing output source bundle")

    inspect_sources = subparsers.add_parser("inspect-sources", help="validate a frozen source bundle and print DEM quality diagnostics")
    inspect_sources.add_argument("--source-dir", type=Path, required=True)

    milestone5 = subparsers.add_parser("milestone5", help="build the playability pipeline exclusively from a validated frozen source bundle")
    _add_common_world_arguments(
        milestone5, default_name="cwr_milestone5", default_display_name="CWR Milestone 5", include_grid=False, default_profile="cwr-ce"
    )
    milestone5.add_argument("--source-dir", type=Path, required=True)
    _add_source_feature_arguments(milestone5)
    _add_playability_arguments(milestone5)

    normalize = subparsers.add_parser("normalize-sources", help="repair and normalize a frozen source bundle into deterministic GeoJSON")
    normalize.add_argument("--source-dir", type=Path, required=True)
    normalize.add_argument("--point-building-footprint", type=float, default=12.0)
    normalize.add_argument("--building-min-area", type=float, default=20.0)
    _add_normalization_arguments(normalize, include_minor_flag=True)

    inspect_normalized = subparsers.add_parser("inspect-normalized", help="validate and summarize a normalized geometry bundle")
    inspect_normalized.add_argument("--normalized-dir", type=Path, required=True)

    milestone6 = subparsers.add_parser("milestone6", help="normalize frozen geometry, emit GeoJSON, and build the world from that intermediate representation")
    _add_common_world_arguments(
        milestone6, default_name="cwr_milestone6", default_display_name="CWR Milestone 6", include_grid=False, default_profile="cwr-ce"
    )
    milestone6.add_argument("--source-dir", type=Path, required=True)
    _add_source_feature_arguments(milestone6)
    _add_playability_arguments(milestone6)
    _add_normalization_arguments(milestone6, include_minor_flag=False)

    milestone7 = subparsers.add_parser("milestone7", help="solve water, roads, buildings, streams, smoothing, and world edges in one priority-based terrain field")
    _add_common_world_arguments(
        milestone7, default_name="cwr_milestone7", default_display_name="CWR Milestone 7", include_grid=False, default_profile="cwr-ce"
    )
    milestone7.add_argument("--source-dir", type=Path, required=True)
    _add_source_feature_arguments(milestone7)
    _add_playability_arguments(milestone7)
    _add_normalization_arguments(milestone7, include_minor_flag=False)
    _add_constraint_solver_arguments(milestone7)

    milestone8 = subparsers.add_parser("milestone8", help="generate reusable procedural MLOD buildings from normalized OSM footprints")
    _add_common_world_arguments(
        milestone8, default_name="cwr_milestone8", default_display_name="CWR Milestone 8", include_grid=False, default_profile="cwr-ce"
    )
    milestone8.add_argument("--source-dir", type=Path, required=True)
    _add_source_feature_arguments(milestone8)
    _add_playability_arguments(milestone8)
    _add_normalization_arguments(milestone8, include_minor_flag=False)
    _add_constraint_solver_arguments(milestone8)
    _add_procedural_building_arguments(milestone8)
    milestone8.add_argument(
        "--ground-textures",
        choices=GROUND_TEXTURE_PROFILES,
        default="nogova",
        help="terrain texture profile: Nogova transition palette (default), Malden-style generated CWC palette, original Everon/Eden assets, desert palette, or packaged generated colours",
    )

    milestone9 = subparsers.add_parser("milestone9", help="apply deterministic surface transitions, shoreline/forest/farm/road materials, overview map, and improved icon")
    _add_common_world_arguments(
        milestone9, default_name="cwr_milestone9", default_display_name="CWR Milestone 9", include_grid=False, default_profile="cwr-ce"
    )
    milestone9.add_argument("--source-dir", type=Path, required=True)
    milestone9.add_argument(
        "--deploy-mod-dir",
        type=Path,
        help="copy the generated PBO and intro mission into this existing mod folder without creating another @mod directory",
    )
    _add_source_feature_arguments(milestone9, include_minor_roads_default=True)
    _add_playability_arguments(milestone9)
    _add_normalization_arguments(milestone9, include_minor_flag=False)
    _add_constraint_solver_arguments(milestone9)
    _add_procedural_building_arguments(milestone9)
    _add_surface_pass_arguments(milestone9)
    milestone9.add_argument("--forest-profile", choices=("everon", "malden"), default="everon", help="Everon square/triangle/cluster ladder by default; malden restores the older block plus individual-tree fallback")
    milestone9.add_argument("--forest-block-model", default=r"data3d\les ctverec pruchozi_T1.p3d", help="primary stock forest block model")
    milestone9.add_argument("--forest-max-block-relief", type=float, default=8.0, help="maximum relief for the primary stock square before the triangle fallback")
    milestone9.add_argument("--forest-block-max-burial", type=float, default=8.0, help="maximum terrain burial under a stock square forest block")
    milestone9.add_argument("--forest-block-max-float", type=float, default=0.5, help="maximum low-side floating under a stock square forest block")
    milestone9.add_argument("--forest-block-max-ground-sink", type=float, default=0.0, help="deprecated compatibility option; rigid stock polygons are no longer hill-sunk")
    milestone9.add_argument("--forest-steep-model", default=r"data3d\les trojuhelnik pruchozi.p3d", help="smaller Everon forest model used on moderate slopes")
    milestone9.add_argument("--forest-steep-footprint", type=float, default=35.0, help="support/clearance footprint for the Everon triangle forest model")
    milestone9.add_argument("--forest-steep-max-relief", type=float, default=18.0, help="maximum local relief allowed beneath the stock Everon triangle")
    milestone9.add_argument("--forest-steep-max-burial", type=float, default=18.0, help="maximum burial under the stock Everon triangle")
    milestone9.add_argument("--forest-steep-max-float", type=float, default=0.5, help="maximum low-side floating under the stock Everon triangle")
    milestone9.add_argument("--forest-steep-max-ground-sink", type=float, default=0.0, help="deprecated compatibility option; rigid stock polygons are no longer hill-sunk")
    milestone9.add_argument("--forest-polygon-sink-fraction", type=float, default=0.5, help="fraction of local relief used to lower every non-flat Everon triangle polygon")
    milestone9.add_argument("--no-severe-hill-forest-fallback", action="store_false", dest="forest_severe_hill_fallback", help="disable the sunk-polygon and individual-tree tiers for severe terrain")
    milestone9.set_defaults(forest_severe_hill_fallback=True)
    milestone9.add_argument("--forest-severe-hill-relief", type=float, default=5.0, help="compatibility threshold used only when a triangle polygon cannot be placed; all placed non-flat triangles use the configured sink")
    milestone9.add_argument("--forest-severe-hill-trees-per-block", type=int, default=10, help="individually grounded trees used for each severe or too-steep rigid-forest rejection")
    milestone9.add_argument("--no-forest-clusters", action="store_false", dest="forest_cluster_fallback", help="disable generated reusable steep-slope forest clusters")
    milestone9.set_defaults(forest_cluster_fallback=True)
    milestone9.add_argument("--forest-cluster-search-radius", type=float, default=10.0, help="candidate search radius for a steep-slope cluster")
    milestone9.add_argument("--forest-cluster-max-relief", type=float, default=48.0, help="maximum footprint relief considered for generated clusters")
    milestone9.add_argument("--forest-cluster-max-burial", type=float, default=1.25, help="maximum buried trunk-base depth in generated clusters")
    milestone9.add_argument("--forest-cluster-max-float", type=float, default=1.25, help="maximum floating trunk-base height in generated clusters")
    milestone9.add_argument("--no-forest-undergrowth", action="store_false", dest="forest_undergrowth_enabled", help="disable reusable interior bush and small-spruce clusters")
    milestone9.set_defaults(forest_undergrowth_enabled=True)
    milestone9.add_argument("--forest-undergrowth-max-objects", type=int, default=120000, help="maximum interior undergrowth cluster objects")
    milestone9.add_argument("--forest-undergrowth-spacing", type=float, default=30.0, help="interior undergrowth grid spacing in metres")
    milestone9.add_argument("--forest-undergrowth-max-relief", type=float, default=20.0)
    milestone9.add_argument("--forest-undergrowth-max-burial", type=float, default=0.8)
    milestone9.add_argument("--forest-undergrowth-max-float", type=float, default=0.8)
    milestone9.add_argument("--forest-undergrowth-ground-clearance", type=float, default=0.03)
    milestone9.add_argument("--no-steep-hill-bushes", action="store_false", dest="steep_hill_bushes_enabled", help="disable extra stock bushes on steep forested hills")
    milestone9.set_defaults(steep_hill_bushes_enabled=True)
    milestone9.add_argument("--max-steep-hill-bush-objects", type=int, default=80000)
    milestone9.add_argument("--steep-hill-bush-spacing", type=float, default=24.0)
    milestone9.add_argument("--steep-hill-bush-min-slope", type=float, default=16.0)
    milestone9.add_argument("--steep-hill-bush-max-relief", type=float, default=8.0)
    milestone9.add_argument("--steep-hill-bush-max-burial", type=float, default=0.6)
    milestone9.add_argument("--steep-hill-bush-max-float", type=float, default=0.8)
    milestone9.add_argument("--steep-hill-bush-ground-clearance", type=float, default=0.03)
    milestone9.add_argument("--no-forest-borders", action="store_false", dest="forest_border_enabled", help="disable Nogova-style forest brush borders")
    milestone9.set_defaults(forest_border_enabled=True)
    milestone9.add_argument("--forest-border-max-objects", type=int, default=2000)
    milestone9.add_argument("--forest-border-spacing", type=float, default=34.0)
    milestone9.add_argument("--forest-border-inset", type=float, default=5.0)
    milestone9.add_argument("--forest-border-max-relief", type=float, default=24.0)
    milestone9.add_argument("--forest-border-max-burial", type=float, default=1.0)
    milestone9.add_argument("--forest-border-max-float", type=float, default=1.0)
    milestone9.add_argument("--no-forest-single-trees", action="store_false", dest="forest_single_tree_enabled", help="disable sparse individual spruce trees inside Everon forests")
    milestone9.set_defaults(forest_single_tree_enabled=True)
    milestone9.add_argument("--forest-single-tree-model", default=r"data3d\str smrk_medium.p3d", help="stock individual tree model used by the Everon forest scatter pass")
    milestone9.add_argument("--max-forest-single-tree-objects", type=int, default=1000, help="extra single-tree safety limit for a 6.4 km world; scales by physical world area (4000 at 12.8 km)")
    milestone9.add_argument("--forest-single-tree-spacing", type=float, default=45.0, help="geographically anchored individual-tree spacing in metres at every world size")
    milestone9.add_argument("--forest-single-tree-footprint", type=float, default=2.0)
    milestone9.add_argument("--forest-single-tree-max-relief", type=float, default=8.0)
    milestone9.add_argument("--forest-single-tree-max-float", type=float, default=0.5, help="maximum triangle-ambiguity lift for individual trees; unsafe candidates are skipped")
    milestone9.add_argument("--no-ditch-grass", action="store_false", dest="ditch_grass_enabled", help="disable reusable tall-grass strips along OSM ditches")
    milestone9.set_defaults(ditch_grass_enabled=True)
    milestone9.add_argument("--max-ditch-grass-objects", type=int, default=2000)
    milestone9.add_argument("--ditch-grass-spacing", type=float, default=18.0)
    milestone9.add_argument("--ditch-grass-endpoint-trim", type=float, default=6.0)
    milestone9.add_argument("--ditch-grass-max-relief", type=float, default=18.0)
    milestone9.add_argument("--ditch-grass-max-burial", type=float, default=0.6)
    milestone9.add_argument("--ditch-grass-max-float", type=float, default=0.8)
    milestone9.add_argument("--ditch-grass-ground-clearance", type=float, default=0.05)
    milestone9.add_argument("--no-barriers", action="store_false", dest="barriers_enabled", help="disable OSM fences, walls and hedges")
    milestone9.set_defaults(barriers_enabled=True)
    milestone9.add_argument("--max-barrier-objects", type=int, default=4000)
    milestone9.add_argument("--barrier-segment-length", type=float, default=6.0)
    milestone9.add_argument("--no-bridges", action="store_false", dest="bridges_enabled", help="disable modular bridge decks on OSM bridge roads")
    milestone9.set_defaults(bridges_enabled=True)
    milestone9.add_argument("--procedural-bridges", action="store_true", dest="procedural_bridges", help="generate world-local bridge deck/rail models instead of using the stock Nogova bridge module (default)")
    milestone9.add_argument("--stock-bridges", action="store_false", dest="procedural_bridges", help="use the stock Nogova bridge module instead of procedural bridges")
    milestone9.set_defaults(procedural_bridges=True)
    milestone9.add_argument("--max-bridge-objects", type=int, default=1000)
    milestone9.add_argument("--bridge-module-length", type=float, default=30.0, help="target module length for procedural bridges; stock Nogova bridges remain fixed at 30 m")
    milestone9.add_argument("--bridge-deck-clearance", type=float, default=1.25, help="minimum procedural bridge roadway clearance above the highest terrain under the full span; default 1.25 m")
    milestone9.add_argument("--bridge-water-clearance", type=float, default=18.0, help="empty clearance above the global water plane beneath the lowest bridge geometry; values below the 18 m safety floor are raised to 18 m")
    milestone9.add_argument("--no-residential-infill", action="store_false", dest="residential_infill_enabled", help="disable deterministic fallback houses in completely empty residential OSM polygons")
    milestone9.set_defaults(residential_infill_enabled=True)
    milestone9.add_argument("--max-residential-infill-buildings", type=int, default=1500)
    milestone9.add_argument("--residential-infill-spacing", type=float, default=68.0)
    milestone9.add_argument("--residential-infill-min-area", type=float, default=1800.0)
    milestone9.add_argument("--residential-infill-road-clearance", type=float, default=0.5)
    milestone9.add_argument("--residential-infill-building-clearance", type=float, default=6.0)
    milestone9.add_argument("--no-overture-buildings", action="store_false", dest="overture_buildings_enabled", help="skip optional Overture Maps building footprints before synthetic residential infill")
    milestone9.set_defaults(overture_buildings_enabled=True)
    milestone9.add_argument("--overture-buildings-geojson", type=Path, default=None, help="pre-downloaded Overture building GeoJSON to use before synthetic residential infill")
    milestone9.add_argument("--no-rural-vegetation", action="store_false", dest="rural_vegetation_enabled", help="disable tree rows, orchards, vineyards, scrub and mapped rock areas")
    milestone9.set_defaults(rural_vegetation_enabled=True)
    milestone9.add_argument("--max-rural-vegetation-objects", type=int, default=3000)
    milestone9.add_argument("--rural-vegetation-spacing", type=float, default=28.0)
    milestone9.add_argument("--no-meadow-grass", action="store_false", dest="meadow_grass_enabled", help="disable randomized tall-grass clusters in OSM landuse=meadow polygons")
    milestone9.set_defaults(meadow_grass_enabled=True)
    milestone9.add_argument("--max-meadow-grass-objects", type=int, default=20000)
    milestone9.add_argument("--meadow-grass-spacing", type=float, default=24.0)
    milestone9.add_argument("--no-wetland-reeds", action="store_false", dest="wetland_reeds_enabled", help="disable stock reed placement in mapped OSM wetlands")
    milestone9.set_defaults(wetland_reeds_enabled=True)
    milestone9.add_argument("--max-wetland-reed-objects", type=int, default=100000)
    milestone9.add_argument("--wetland-reed-spacing", type=float, default=18.0)
    milestone9.add_argument("--wetland-reed-max-relief", type=float, default=4.0)
    milestone9.add_argument("--wetland-reed-max-burial", type=float, default=0.5)
    milestone9.add_argument("--wetland-reed-max-float", type=float, default=1.0)
    milestone9.add_argument("--wetland-reed-ground-clearance", type=float, default=0.03)
    milestone9.add_argument("--no-rocky-forest-fallback", action="store_false", dest="rocky_forest_fallback_enabled", help="disable sparse rocks on forest cells too steep for all forest tiers")
    milestone9.set_defaults(rocky_forest_fallback_enabled=True)
    milestone9.add_argument("--max-rocky-forest-objects", type=int, default=1200)
    milestone9.add_argument("--rocky-forest-rocks-per-patch", type=int, default=3, help="deterministic rock groups placed across each forest patch rejected by every tree tier")
    milestone9.add_argument("--rocky-forest-spread", type=float, default=18.0, help="maximum scatter radius for rocks inside one rejected forest patch")
    milestone9.add_argument("--rocky-forest-max-relief", type=float, default=42.0)
    milestone9.add_argument("--rocky-forest-max-burial", type=float, default=1.0)
    milestone9.add_argument("--rocky-forest-max-float", type=float, default=1.0)
    milestone9.add_argument("--forest-hillside-trees-per-block", type=int, default=5, help="Malden-profile fallback trees per rejected block; ignored by the default Everon profile")
    milestone9.add_argument("--forest-hillside-tree-model", default=r"data3d\str_fikovnik.p3d", help="Malden-profile individual-tree fallback model")
    milestone9.add_argument("--forest-hillside-tree-footprint", type=float, default=4.0, help="support and clearance footprint for each hillside tree in metres")
    milestone9.add_argument("--forest-hillside-tree-max-relief", type=float, default=2.5, help="maximum local relief beneath one hillside tree")
    milestone9.add_argument("--bus-stop-signs", "--bus-stops", action="store_true", dest="bus_stops_enabled", help="place stock bus-stop signs at mapped OSM bus stops (default)")
    milestone9.add_argument("--no-bus-stop-signs", action="store_false", dest="bus_stops_enabled", help="disable stock bus-stop signs")
    milestone9.set_defaults(bus_stops_enabled=True)
    milestone9.add_argument("--bus-stop-model", default=r"o\misc\aut_z_st.p3d", help="stock bus-stop sign model placed at OSM bus stops")
    milestone9.add_argument("--bus-stop-footprint", type=float, default=1.6, help="terrain support footprint sampled beneath bus-stop signs in metres")
    milestone9.add_argument("--bus-stop-ground-clearance", type=float, default=0.12, help="vertical clearance above the highest terrain sample beneath bus-stop signs")
    milestone9.add_argument("--max-landmark-objects", type=int, default=1000, help="maximum bus-stop and semantic landmark objects")
    milestone9.add_argument("--no-cemeteries", action="store_false", dest="cemeteries_enabled", help="disable grave placement in OSM cemeteries and graveyards")
    milestone9.set_defaults(cemeteries_enabled=True)
    milestone9.add_argument("--max-grave-objects", type=int, default=12000, help="maximum stock gravestones placed inside mapped cemeteries")
    milestone9.add_argument("--grave-spacing", type=float, default=3.5, help="average row spacing for cemetery gravestones in metres")
    milestone9.add_argument("--grave-inset", type=float, default=2.0, help="clear margin inside cemetery boundaries in metres")
    milestone9.add_argument("--grave-ground-clearance", type=float, default=0.12, help="vertical clearance above the highest terrain sample beneath each gravestone")
    milestone9.add_argument("--grave-road-clearance", type=float, default=1.0, help="extra clearance around complete road corridors for gravestones")
    milestone9.add_argument("--grave-building-clearance", type=float, default=1.5, help="clearance around final generated building footprints for gravestones")
    milestone9.add_argument("--semantic-site-max-relief", type=float, default=1.5, help="maximum relief for procedural sports-pitch and parking surfaces")
    milestone9.add_argument("--semantic-site-max-variants", type=int, default=64, help="maximum procedural pitch/parking P3D variants")
    milestone9.add_argument(
        "--ground-textures",
        choices=GROUND_TEXTURE_PROFILES,
        default="nogova",
        help="Milestone 9 ground palette: Nogova transition preset (default), Malden-style generated CWC palette, stock Everon/Eden, desert generated textures, or fully generated colours",
    )
    milestone9.add_argument(
        "--pbo-backend",
        choices=("auto", "python", "poseidon"),
        default="auto",
        help="PBO writer: prefer PoseidonTools when available, force the built-in Python writer, or require PoseidonTools",
    )
    milestone9.add_argument(
        "--poseidon-tools",
        type=Path,
        help="path to PoseidonTools executable; otherwise CWR_POSEIDON_TOOLS and PATH are searched",
    )

    return parser


def _print_result(result, display_name: str, name: str) -> None:
    print(f"World:      {display_name} ({name})")
    print(f"WRP:        {result.wrp_path}")
    for texture in result.texture_paths:
        print(f"Texture:    {texture}")
    print(f"Mod root:   {result.pbo_path.parent.parent}")
    print(f"PBO:        {result.pbo_path}")
    print("Mission unit: SoldierWB")
    print(f"Mission:    {result.mission_path}")
    print(f"Menu intro: {result.intro_mission_path}")
    print(f"Preview:    {result.preview_path}")
    if result.height_preview_path:
        print(f"Height:     {result.height_preview_path}")
    if result.material_preview_path:
        print(f"Materials:  {result.material_preview_path}")
    if result.osm_preview_path:
        print(f"OSM:        {result.osm_preview_path}")
    if result.meadow_grass_preview_path:
        print(f"Meadows:    {result.meadow_grass_preview_path}")
    if result.osm_source_path:
        print(f"OSM data:   {result.osm_source_path}")
    if result.attribution_path:
        print(f"Attribution:{result.attribution_path}")
    if result.asset_catalogue_path:
        print(f"Assets:     {result.asset_catalogue_path}")
    if result.road_report_path:
        print(f"Road fit:   {result.road_report_path}")
    if result.grading_report_path:
        print(f"Grading:    {result.grading_report_path}")
    if result.reproducibility_path:
        print(f"Repro:      {result.reproducibility_path}")
    if result.source_manifest_path:
        print(f"Sources:    {result.source_manifest_path}")
    if result.source_validation_path:
        print(f"Source QA:  {result.source_validation_path}")
    if result.normalized_dir:
        print(f"Normalized: {result.normalized_dir}")
    if result.solver_heightmap_path:
        print(f"Solved DEM: {result.solver_heightmap_path}")
    if result.building_catalogue_path:
        print(f"Buildings:  {result.building_catalogue_path}")
    if result.surface_report_path:
        print(f"Surfaces:   {result.surface_report_path}")
    if result.overview_map_path:
        print(f"Overview:   {result.overview_map_path}")
    if result.world_icon_path:
        print(f"World icon: {result.world_icon_path}")
    print(f"Report:     {result.report_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    milestone_command = str(getattr(args, "command", "")).startswith("milestone")
    progress_command = milestone_command or args.command in {"fetch-sources", "regrid-sources"}
    if progress_command:
        os.environ.setdefault("CWR_PROGRESS", "1")
        os.environ.setdefault("CWR_PROGRESS_FORMAT", "human")
        start_progress_session()
    try:
        if args.command == "milestone1":
            spec = WorldSpec(
                name=args.name,
                display_name=args.display_name,
                profile=args.profile,
                cells=args.cells,
                cell_size=args.cell_size,
            )
            result = build_milestone1(args.output, spec, clean=not args.keep_output)
        elif args.command == "milestone2":
            spec = HeightmapSpec(**_heightmap_kwargs(args))
            result = build_milestone2(args.output, spec, clean=not args.keep_output)
        elif args.command == "milestone3":
            spec = OsmSpec(**_osm_kwargs(args))
            result = build_milestone3(args.output, spec, clean=not args.keep_output)
        elif args.command == "milestone4":
            spec = PlayabilitySpec(
                **_osm_kwargs(args),
                road_connection_tolerance=args.road_connection_tolerance,
                maximum_road_grade_percent=args.maximum_road_grade,
                road_grade_radius=args.road_grade_radius,
                building_grade_radius=args.building_grade_radius,
                maximum_grade_adjustment=args.maximum_grade_adjustment,
                transition_cells=args.transition_cells,
                asset_roots=tuple(args.asset_root),
                strict_assets=args.strict_assets,
                osm_asset_mapping_path=args.osm_asset_map,
                cache_dir=args.cache_dir,
                cache_enabled=not args.no_cache,
                cache_refresh=args.cache_refresh,
                verify_regeneration=args.verify_regeneration,
                town_name_limit=args.town_name_limit,
                deterministic_seed=args.deterministic_seed,
            )
            result = build_milestone4(args.output, spec, clean=not args.keep_output)
        elif args.command == "fetch-sources":
            overpass_urls = tuple(args.overpass_url) if args.overpass_url else SourceFetchSpec.__dataclass_fields__["overpass_urls"].default
            hgt_templates = tuple(args.hgt_url_template) if args.hgt_url_template else SourceFetchSpec.__dataclass_fields__["hgt_url_templates"].default
            fetch_spec = SourceFetchSpec(
                source_dir=args.source_dir,
                map_url=args.map_url,
                center=None if args.center is None else tuple(args.center),
                bbox=None if args.bbox is None else tuple(args.bbox),
                cells=args.cells,
                cell_size=args.cell_size,
                refresh=args.refresh,
                reference_map=args.reference_map,
                dem_provider=args.dem_provider,
                dem_name=args.dem_name,
                overpass_urls=overpass_urls,
                overpass_timeout_seconds=args.overpass_timeout,
                hgt_url_templates=hgt_templates,
            )
            bundle = fetch_sources(fetch_spec)
            print(f"Source bundle: {bundle.root}")
            print(f"Manifest:      {bundle.manifest_path}")
            print(f"Heightmap:     {bundle.heightmap_path}")
            print(f"OSM JSON:      {bundle.osm_json_path}")
            print(f"Fingerprint:   {bundle.fingerprint}")
            print(f"Total runtime: {format_duration(progress_elapsed_seconds())}")
            return 0
        elif args.command == "regrid-sources":
            bundle = regrid_sources(SourceRegridSpec(
                source_dir=args.source_dir,
                output_source_dir=args.output_source_dir,
                cells=args.cells,
                cell_size=args.cell_size,
                replace_output=args.replace_output,
            ))
            print(f"Regridded bundle: {bundle.root}")
            print(f"Grid:             {bundle.cells}x{bundle.cells} @ {bundle.cell_size:g}m")
            print(f"Manifest:         {bundle.manifest_path}")
            print(f"Heightmap:        {bundle.heightmap_path}")
            print(f"OSM JSON:         {bundle.osm_json_path}")
            print(f"Fingerprint:      {bundle.fingerprint}")
            print(f"Total runtime:    {format_duration(progress_elapsed_seconds())}")
            return 0
        elif args.command == "inspect-sources":
            report = validate_source_bundle(args.source_dir)
            document = json.loads(report.bundle.manifest_path.read_text(encoding="utf-8"))
            elevation = document.get("elevation", {})
            resampling = elevation.get("resampling", {}) if isinstance(elevation, dict) else {}
            print(f"Source bundle: {report.bundle.root}")
            print(f"Grid:          {report.bundle.cells}x{report.bundle.cells} @ {report.bundle.cell_size:g}m")
            print(f"BBox:          {report.bundle.bbox}")
            print(f"DEM product:   {elevation.get('product')}")
            print(f"Elevation:     {elevation.get('minimum_metres')}..{elevation.get('maximum_metres')}m")
            print(f"Resampling:    {resampling.get('method')} onto {resampling.get('target_grid')}")
            if resampling.get("raw_width") is not None:
                print(f"Raw raster:    {resampling.get('raw_width')}x{resampling.get('raw_height')} {resampling.get('raw_crs')}")
                print(f"Raw bounds:    {resampling.get('raw_bounds_west_south_east_north')}")
                print(f"Raw finite:    {float(resampling.get('raw_finite_fraction', 0.0)) * 100.0:.3f}%")
            if resampling.get("output_standard_deviation_metres") is not None:
                print(f"Std deviation: {float(resampling['output_standard_deviation_metres']):.3f}m")
                print(f"Neighbour Δ:   {float(resampling.get('output_mean_neighbour_delta_metres', 0.0)):.3f}m")
                print(f"Percentiles:   {resampling.get('output_percentiles_metres')}")
            print(f"Validation:    {report.report_path}")
            return 0
        elif args.command == "normalize-sources":
            normalized = normalize_source_bundle(NormalizationSpec(
                source_dir=args.source_dir,
                **_normalization_kwargs(args),
            ))
            print(f"Normalized:  {normalized.root}")
            print(f"Manifest:    {normalized.manifest_path}")
            print(f"Validation:  {normalized.validation_path}")
            print(f"Fingerprint: {normalized.normalized_fingerprint}")
            for name, count in sorted(normalized.counts.items()):
                print(f"{name:16} {count}")
            return 0
        elif args.command == "inspect-normalized":
            normalized = validate_normalized_bundle(args.normalized_dir)
            print(f"Normalized:  {normalized.root}")
            print(f"Source:      {normalized.source_fingerprint}")
            print(f"Fingerprint: {normalized.normalized_fingerprint}")
            for name, count in sorted(normalized.counts.items()):
                print(f"{name:16} {count}")
            print(f"Validation:  {normalized.validation_path}")
            return 0
        elif args.command == "milestone5":
            spec = Milestone5Spec(
                source_dir=args.source_dir,
                name=args.name,
                display_name=args.display_name,
                profile=args.profile,
                include_minor_roads=args.include_minor_roads,
                forest_road_clearance=args.forest_road_clearance,
                building_ground_clearance=args.building_ground_clearance,
                forest_ground_clearance=args.forest_ground_clearance,
                point_building_footprint=args.point_building_footprint,
                water_depth=args.water_depth,
                coastline_blend_cells=args.coast_blend_cells,
                road_segment_length=args.road_segment_length,
                max_road_objects=args.max_road_objects,
                max_buildings=args.max_buildings,
                building_minimum_area=args.building_min_area,
                forest_tree_spacing=args.forest_tree_spacing,
                max_forest_objects=args.max_forest_objects,
                forest_profile=args.forest_profile,
                forest_tree_model=args.forest_block_model,
                forest_maximum_block_relief=args.forest_max_block_relief,
                forest_everon_steep_model=args.forest_steep_model,
                forest_everon_steep_footprint=args.forest_steep_footprint,
                forest_everon_steep_maximum_relief=args.forest_steep_max_relief,
                forest_hillside_fallback=args.forest_profile == "malden",
                forest_hillside_tree_model=args.forest_hillside_tree_model,
                forest_hillside_trees_per_block=args.forest_hillside_trees_per_block,
                forest_hillside_tree_footprint=args.forest_hillside_tree_footprint,
                forest_hillside_tree_maximum_relief=args.forest_hillside_tree_max_relief,
                road_connection_tolerance=args.road_connection_tolerance,
                maximum_road_grade_percent=args.maximum_road_grade,
                road_grade_radius=args.road_grade_radius,
                building_grade_radius=args.building_grade_radius,
                maximum_grade_adjustment=args.maximum_grade_adjustment,
                transition_cells=args.transition_cells,
                asset_roots=tuple(args.asset_root),
                strict_assets=args.strict_assets,
                osm_asset_mapping_path=args.osm_asset_map,
                cache_dir=args.cache_dir,
                cache_enabled=not args.no_cache,
                cache_refresh=args.cache_refresh,
                verify_regeneration=args.verify_regeneration,
                town_name_limit=args.town_name_limit,
            )
            result = build_milestone5(args.output, spec, clean=not args.keep_output)
        elif args.command == "milestone6":
            spec = Milestone6Spec(
                source_dir=args.source_dir,
                name=args.name,
                display_name=args.display_name,
                profile=args.profile,
                include_minor_roads=args.include_minor_roads,
                forest_road_clearance=args.forest_road_clearance,
                building_ground_clearance=args.building_ground_clearance,
                forest_ground_clearance=args.forest_ground_clearance,
                point_building_footprint=args.point_building_footprint,
                water_depth=args.water_depth,
                coastline_blend_cells=args.coast_blend_cells,
                road_segment_length=args.road_segment_length,
                max_road_objects=args.max_road_objects,
                max_buildings=args.max_buildings,
                building_minimum_area=args.building_min_area,
                forest_tree_spacing=args.forest_tree_spacing,
                max_forest_objects=args.max_forest_objects,
                road_connection_tolerance=args.road_connection_tolerance,
                maximum_road_grade_percent=args.maximum_road_grade,
                road_grade_radius=args.road_grade_radius,
                building_grade_radius=args.building_grade_radius,
                maximum_grade_adjustment=args.maximum_grade_adjustment,
                transition_cells=args.transition_cells,
                asset_roots=tuple(args.asset_root),
                strict_assets=args.strict_assets,
                osm_asset_mapping_path=args.osm_asset_map,
                cache_dir=args.cache_dir,
                cache_enabled=not args.no_cache,
                cache_refresh=args.cache_refresh,
                verify_regeneration=args.verify_regeneration,
                town_name_limit=args.town_name_limit,
                normalized_dir=args.normalized_dir,
                normalization_refresh=args.normalization_refresh,
                road_snap_tolerance=args.road_snap_tolerance,
                road_building_setback=args.road_building_setback,
                building_merge_gap=args.building_merge_gap,
                building_overlap_threshold=args.building_overlap_threshold,
                forest_edge_width=args.forest_edge_width,
                forest_building_clearance=args.forest_building_clearance,
                minimum_forest_area=args.minimum_forest_area,
                coordinate_precision=args.coordinate_precision,
            )
            result = build_milestone6(args.output, spec, clean=not args.keep_output)
        elif args.command == "milestone7":
            spec = Milestone7Spec(
                source_dir=args.source_dir,
                name=args.name,
                display_name=args.display_name,
                profile=args.profile,
                include_minor_roads=args.include_minor_roads,
                forest_road_clearance=args.forest_road_clearance,
                building_ground_clearance=args.building_ground_clearance,
                forest_ground_clearance=args.forest_ground_clearance,
                point_building_footprint=args.point_building_footprint,
                water_depth=args.water_depth,
                coastline_blend_cells=args.coast_blend_cells,
                road_segment_length=args.road_segment_length,
                max_road_objects=args.max_road_objects,
                max_buildings=args.max_buildings,
                building_minimum_area=args.building_min_area,
                forest_tree_spacing=args.forest_tree_spacing,
                max_forest_objects=args.max_forest_objects,
                road_connection_tolerance=args.road_connection_tolerance,
                maximum_road_grade_percent=args.maximum_road_grade,
                road_grade_radius=args.road_grade_radius,
                building_grade_radius=args.building_grade_radius,
                maximum_grade_adjustment=args.maximum_grade_adjustment,
                transition_cells=args.transition_cells,
                asset_roots=tuple(args.asset_root),
                strict_assets=args.strict_assets,
                osm_asset_mapping_path=args.osm_asset_map,
                cache_dir=args.cache_dir,
                cache_enabled=not args.no_cache,
                cache_refresh=args.cache_refresh,
                verify_regeneration=args.verify_regeneration,
                town_name_limit=args.town_name_limit,
                normalized_dir=args.normalized_dir,
                normalization_refresh=args.normalization_refresh,
                road_snap_tolerance=args.road_snap_tolerance,
                road_building_setback=args.road_building_setback,
                building_merge_gap=args.building_merge_gap,
                building_overlap_threshold=args.building_overlap_threshold,
                forest_edge_width=args.forest_edge_width,
                forest_building_clearance=args.forest_building_clearance,
                minimum_forest_area=args.minimum_forest_area,
                coordinate_precision=args.coordinate_precision,
                major_road_grade_percent=args.major_road_grade,
                shoreline_transition_cells=args.shoreline_transition_cells,
                lake_shore_smoothing_cells=args.lake_shore_smoothing_cells,
                lake_shore_maximum_slope_percent=args.lake_shore_max_slope,
                building_pad_margin=args.building_pad_margin,
                stream_channel_depth=args.stream_channel_depth,
                river_channel_depth=args.river_channel_depth,
                watercourse_minimum_gradient_percent=args.watercourse_minimum_gradient,
                natural_smoothing_strength=args.natural_smoothing_strength,
                solver_iterations=args.solver_iterations,
                world_edge_blend_cells=args.world_edge_blend_cells,
            )
            result = build_milestone7(args.output, spec, clean=not args.keep_output)
        elif args.command == "milestone8":
            spec = Milestone8Spec(
                source_dir=args.source_dir,
                name=args.name,
                display_name=args.display_name,
                profile=args.profile,
                include_minor_roads=args.include_minor_roads,
                forest_road_clearance=args.forest_road_clearance,
                building_ground_clearance=args.building_ground_clearance,
                forest_ground_clearance=args.forest_ground_clearance,
                point_building_footprint=args.point_building_footprint,
                water_depth=args.water_depth,
                coastline_blend_cells=args.coast_blend_cells,
                road_segment_length=args.road_segment_length,
                max_road_objects=args.max_road_objects,
                max_buildings=args.max_buildings,
                building_minimum_area=args.building_min_area,
                forest_tree_spacing=args.forest_tree_spacing,
                max_forest_objects=args.max_forest_objects,
                road_connection_tolerance=args.road_connection_tolerance,
                maximum_road_grade_percent=args.maximum_road_grade,
                road_grade_radius=args.road_grade_radius,
                building_grade_radius=args.building_grade_radius,
                maximum_grade_adjustment=args.maximum_grade_adjustment,
                transition_cells=args.transition_cells,
                asset_roots=tuple(args.asset_root),
                strict_assets=args.strict_assets,
                osm_asset_mapping_path=args.osm_asset_map,
                cache_dir=args.cache_dir,
                cache_enabled=not args.no_cache,
                cache_refresh=args.cache_refresh,
                verify_regeneration=args.verify_regeneration,
                town_name_limit=args.town_name_limit,
                normalized_dir=args.normalized_dir,
                normalization_refresh=args.normalization_refresh,
                road_snap_tolerance=args.road_snap_tolerance,
                road_building_setback=args.road_building_setback,
                building_merge_gap=args.building_merge_gap,
                building_overlap_threshold=args.building_overlap_threshold,
                forest_edge_width=args.forest_edge_width,
                forest_building_clearance=args.forest_building_clearance,
                minimum_forest_area=args.minimum_forest_area,
                coordinate_precision=args.coordinate_precision,
                major_road_grade_percent=args.major_road_grade,
                shoreline_transition_cells=args.shoreline_transition_cells,
                lake_shore_smoothing_cells=args.lake_shore_smoothing_cells,
                lake_shore_maximum_slope_percent=args.lake_shore_max_slope,
                building_pad_margin=args.building_pad_margin,
                stream_channel_depth=args.stream_channel_depth,
                river_channel_depth=args.river_channel_depth,
                watercourse_minimum_gradient_percent=args.watercourse_minimum_gradient,
                natural_smoothing_strength=args.natural_smoothing_strength,
                solver_iterations=args.solver_iterations,
                world_edge_blend_cells=args.world_edge_blend_cells,
                building_width_quantum=args.building_width_quantum,
                procedural_building_interiors=args.procedural_building_interiors,
                high_quality_building_textures=args.high_quality_building_textures,
                building_length_quantum=args.building_length_quantum,
                building_height_quantum=args.building_height_quantum,
                building_minimum_width=args.building_min_width,
                building_maximum_width=args.building_max_width,
                building_minimum_length=args.building_min_length,
                building_maximum_length=args.building_max_length,
                building_minimum_height=args.building_min_height,
                building_maximum_height=args.building_max_height,
                building_level_height=args.building_level_height,
                building_maximum_variants=args.building_max_variants,
                building_roof_pitch_degrees=args.building_roof_pitch,
                church_ground_clearance=args.church_ground_clearance,
                building_foundation_depth=args.building_foundation_depth,
                building_foundation_maximum_depth=args.building_foundation_max_depth,
                building_foundation_depth_quantum=args.building_foundation_depth_quantum,
                building_foundation_safety=args.building_foundation_safety,
                building_maximum_pad_relief=args.building_max_pad_relief,
                ground_texture_profile=args.ground_textures,
            )
            result = build_milestone8(args.output, spec, clean=not args.keep_output)
        elif args.command == "milestone9":
            spec = Milestone9Spec(
                source_dir=args.source_dir,
                name=args.name,
                display_name=args.display_name,
                profile=args.profile,
                include_minor_roads=args.include_minor_roads,
                forest_road_clearance=args.forest_road_clearance,
                building_ground_clearance=args.building_ground_clearance,
                forest_ground_clearance=args.forest_ground_clearance,
                point_building_footprint=args.point_building_footprint,
                water_depth=args.water_depth,
                coastline_blend_cells=args.coast_blend_cells,
                road_segment_length=args.road_segment_length,
                max_road_objects=args.max_road_objects,
                max_buildings=args.max_buildings,
                building_minimum_area=args.building_min_area,
                forest_tree_spacing=args.forest_tree_spacing,
                max_forest_objects=args.max_forest_objects,
                road_connection_tolerance=args.road_connection_tolerance,
                maximum_road_grade_percent=args.maximum_road_grade,
                road_grade_radius=args.road_grade_radius,
                building_grade_radius=args.building_grade_radius,
                maximum_grade_adjustment=args.maximum_grade_adjustment,
                transition_cells=args.transition_cells,
                asset_roots=tuple(args.asset_root),
                strict_assets=args.strict_assets,
                osm_asset_mapping_path=args.osm_asset_map,
                cache_dir=args.cache_dir,
                cache_enabled=not args.no_cache,
                cache_refresh=args.cache_refresh,
                verify_regeneration=args.verify_regeneration,
                town_name_limit=args.town_name_limit,
                normalized_dir=args.normalized_dir,
                normalization_refresh=args.normalization_refresh,
                road_snap_tolerance=args.road_snap_tolerance,
                road_building_setback=args.road_building_setback,
                building_merge_gap=args.building_merge_gap,
                building_overlap_threshold=args.building_overlap_threshold,
                forest_edge_width=args.forest_edge_width,
                forest_building_clearance=args.forest_building_clearance,
                minimum_forest_area=args.minimum_forest_area,
                coordinate_precision=args.coordinate_precision,
                major_road_grade_percent=args.major_road_grade,
                shoreline_transition_cells=args.shoreline_transition_cells,
                lake_shore_smoothing_cells=args.lake_shore_smoothing_cells,
                lake_shore_maximum_slope_percent=args.lake_shore_max_slope,
                building_pad_margin=args.building_pad_margin,
                stream_channel_depth=args.stream_channel_depth,
                river_channel_depth=args.river_channel_depth,
                watercourse_minimum_gradient_percent=args.watercourse_minimum_gradient,
                natural_smoothing_strength=args.natural_smoothing_strength,
                solver_iterations=args.solver_iterations,
                world_edge_blend_cells=args.world_edge_blend_cells,
                building_width_quantum=args.building_width_quantum,
                procedural_building_interiors=args.procedural_building_interiors,
                high_quality_building_textures=args.high_quality_building_textures,
                building_length_quantum=args.building_length_quantum,
                building_height_quantum=args.building_height_quantum,
                building_minimum_width=args.building_min_width,
                building_maximum_width=args.building_max_width,
                building_minimum_length=args.building_min_length,
                building_maximum_length=args.building_max_length,
                building_minimum_height=args.building_min_height,
                building_maximum_height=args.building_max_height,
                building_level_height=args.building_level_height,
                building_maximum_variants=args.building_max_variants,
                building_roof_pitch_degrees=args.building_roof_pitch,
                church_ground_clearance=args.church_ground_clearance,
                building_foundation_depth=args.building_foundation_depth,
                building_foundation_maximum_depth=args.building_foundation_max_depth,
                building_foundation_depth_quantum=args.building_foundation_depth_quantum,
                building_foundation_safety=args.building_foundation_safety,
                building_maximum_pad_relief=args.building_max_pad_relief,
                forest_profile=args.forest_profile,
                forest_tree_model=args.forest_block_model,
                forest_maximum_block_relief=args.forest_max_block_relief,
                forest_block_maximum_burial=args.forest_block_max_burial,
                forest_block_maximum_float=args.forest_block_max_float,
                forest_block_maximum_ground_sink=args.forest_block_max_ground_sink,
                forest_everon_steep_model=args.forest_steep_model,
                forest_everon_steep_footprint=args.forest_steep_footprint,
                forest_everon_steep_maximum_relief=args.forest_steep_max_relief,
                forest_everon_steep_maximum_burial=args.forest_steep_max_burial,
                forest_everon_steep_maximum_float=args.forest_steep_max_float,
                forest_everon_steep_maximum_ground_sink=args.forest_steep_max_ground_sink,
                forest_polygon_sink_fraction=args.forest_polygon_sink_fraction,
                forest_severe_hill_fallback=args.forest_severe_hill_fallback,
                forest_severe_hill_relief=args.forest_severe_hill_relief,
                forest_severe_hill_trees_per_block=args.forest_severe_hill_trees_per_block,
                forest_cluster_fallback=args.forest_cluster_fallback,
                forest_cluster_search_radius=args.forest_cluster_search_radius,
                forest_cluster_maximum_relief=args.forest_cluster_max_relief,
                forest_cluster_maximum_burial=args.forest_cluster_max_burial,
                forest_cluster_maximum_float=args.forest_cluster_max_float,
                forest_undergrowth_enabled=args.forest_undergrowth_enabled,
                forest_undergrowth_maximum_objects=args.forest_undergrowth_max_objects,
                forest_undergrowth_spacing=args.forest_undergrowth_spacing,
                forest_undergrowth_maximum_relief=args.forest_undergrowth_max_relief,
                forest_undergrowth_maximum_burial=args.forest_undergrowth_max_burial,
                forest_undergrowth_maximum_float=args.forest_undergrowth_max_float,
                forest_undergrowth_ground_clearance=args.forest_undergrowth_ground_clearance,
                steep_hill_bushes_enabled=args.steep_hill_bushes_enabled,
                maximum_steep_hill_bush_objects=args.max_steep_hill_bush_objects,
                steep_hill_bush_spacing=args.steep_hill_bush_spacing,
                steep_hill_bush_minimum_slope_degrees=args.steep_hill_bush_min_slope,
                steep_hill_bush_maximum_relief=args.steep_hill_bush_max_relief,
                steep_hill_bush_maximum_burial=args.steep_hill_bush_max_burial,
                steep_hill_bush_maximum_float=args.steep_hill_bush_max_float,
                steep_hill_bush_ground_clearance=args.steep_hill_bush_ground_clearance,
                forest_border_enabled=args.forest_border_enabled,
                forest_border_maximum_objects=args.forest_border_max_objects,
                forest_border_spacing=args.forest_border_spacing,
                forest_border_inset=args.forest_border_inset,
                forest_border_maximum_relief=args.forest_border_max_relief,
                forest_border_maximum_burial=args.forest_border_max_burial,
                forest_border_maximum_float=args.forest_border_max_float,
                forest_single_tree_enabled=args.forest_single_tree_enabled,
                forest_single_tree_model=args.forest_single_tree_model,
                maximum_forest_single_tree_objects=args.max_forest_single_tree_objects,
                forest_single_tree_spacing=args.forest_single_tree_spacing,
                forest_single_tree_footprint=args.forest_single_tree_footprint,
                forest_single_tree_maximum_relief=args.forest_single_tree_max_relief,
                forest_single_tree_maximum_float=args.forest_single_tree_max_float,
                ditch_grass_enabled=args.ditch_grass_enabled,
                maximum_ditch_grass_objects=args.max_ditch_grass_objects,
                ditch_grass_spacing=args.ditch_grass_spacing,
                ditch_grass_endpoint_trim=args.ditch_grass_endpoint_trim,
                ditch_grass_maximum_relief=args.ditch_grass_max_relief,
                ditch_grass_maximum_burial=args.ditch_grass_max_burial,
                ditch_grass_maximum_float=args.ditch_grass_max_float,
                ditch_grass_ground_clearance=args.ditch_grass_ground_clearance,
                forest_hillside_tree_model=args.forest_hillside_tree_model,
                forest_hillside_trees_per_block=args.forest_hillside_trees_per_block,
                forest_hillside_tree_footprint=args.forest_hillside_tree_footprint,
                forest_hillside_tree_maximum_relief=args.forest_hillside_tree_max_relief,
                barriers_enabled=args.barriers_enabled,
                maximum_barrier_objects=args.max_barrier_objects,
                barrier_segment_length=args.barrier_segment_length,
                bridges_enabled=args.bridges_enabled,
                procedural_bridges=args.procedural_bridges,
                maximum_bridge_objects=args.max_bridge_objects,
                bridge_module_length=args.bridge_module_length,
                bridge_deck_clearance=args.bridge_deck_clearance,
                bridge_water_clearance=args.bridge_water_clearance,
                residential_infill_enabled=args.residential_infill_enabled,
                maximum_residential_infill_buildings=args.max_residential_infill_buildings,
                residential_infill_spacing=args.residential_infill_spacing,
                residential_infill_minimum_area=args.residential_infill_min_area,
                residential_infill_road_clearance=args.residential_infill_road_clearance,
                residential_infill_building_clearance=args.residential_infill_building_clearance,
                overture_buildings_enabled=args.overture_buildings_enabled,
                overture_buildings_geojson=args.overture_buildings_geojson,
                rural_vegetation_enabled=args.rural_vegetation_enabled,
                maximum_rural_vegetation_objects=args.max_rural_vegetation_objects,
                rural_vegetation_spacing=args.rural_vegetation_spacing,
                meadow_grass_enabled=args.meadow_grass_enabled,
                maximum_meadow_grass_objects=args.max_meadow_grass_objects,
                meadow_grass_spacing=args.meadow_grass_spacing,
                wetland_reeds_enabled=args.wetland_reeds_enabled,
                maximum_wetland_reed_objects=args.max_wetland_reed_objects,
                wetland_reed_spacing=args.wetland_reed_spacing,
                wetland_reed_maximum_relief=args.wetland_reed_max_relief,
                wetland_reed_maximum_burial=args.wetland_reed_max_burial,
                wetland_reed_maximum_float=args.wetland_reed_max_float,
                wetland_reed_ground_clearance=args.wetland_reed_ground_clearance,
                rocky_forest_fallback_enabled=args.rocky_forest_fallback_enabled,
                maximum_rocky_forest_objects=args.max_rocky_forest_objects,
                rocky_forest_rocks_per_patch=args.rocky_forest_rocks_per_patch,
                rocky_forest_spread=args.rocky_forest_spread,
                rocky_forest_maximum_relief=args.rocky_forest_max_relief,
                rocky_forest_maximum_burial=args.rocky_forest_max_burial,
                rocky_forest_maximum_float=args.rocky_forest_max_float,
                bus_stops_enabled=args.bus_stops_enabled,
                bus_stop_model=args.bus_stop_model,
                bus_stop_footprint=args.bus_stop_footprint,
                bus_stop_ground_clearance=args.bus_stop_ground_clearance,
                maximum_landmark_objects=args.max_landmark_objects,
                cemeteries_enabled=args.cemeteries_enabled,
                maximum_grave_objects=args.max_grave_objects,
                grave_spacing=args.grave_spacing,
                grave_inset=args.grave_inset,
                grave_ground_clearance=args.grave_ground_clearance,
                grave_road_clearance=args.grave_road_clearance,
                grave_building_clearance=args.grave_building_clearance,
                semantic_site_maximum_relief=args.semantic_site_max_relief,
                semantic_site_maximum_variants=args.semantic_site_max_variants,
                ground_texture_profile=args.ground_textures,
                surface_shoreline_wet_cells=args.surface_wet_shore_cells,
                surface_shoreline_sand_cells=args.surface_sand_shore_cells,
                surface_transition_cells=args.surface_transition_cells,
                surface_forest_edge_cells=args.surface_forest_edge_cells,
                surface_farmland_strip_cells=args.surface_farmland_strip_cells,
                surface_road_shoulder_metres=args.surface_road_shoulder,
                surface_dirt_blend_metres=args.surface_dirt_blend,
                surface_steep_slope_degrees=args.surface_steep_slope,
                surface_colour_reference_path=args.colour_reference,
                surface_colour_reference_strength=args.colour_reference_strength,
                surface_overview_size=args.overview_size,
                surface_texture_size=args.surface_texture_size,
                surface_ground_mode=args.surface_ground_mode,
                pbo_backend=args.pbo_backend,
                poseidon_tools_path=args.poseidon_tools,
                deploy_mod_dir=args.deploy_mod_dir,
            )
            result = build_milestone9(args.output, spec, clean=not args.keep_output)
        else:
            raise AssertionError("argparse accepted an unknown command")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if progress_command:
            print(
                f"Total runtime before failure: {format_duration(progress_elapsed_seconds())}",
                file=sys.stderr,
            )
        return 1

    _print_result(result, spec.display_name, spec.name)
    if milestone_command:
        print(f"Total runtime: {format_duration(progress_elapsed_seconds())}")
    return 0
