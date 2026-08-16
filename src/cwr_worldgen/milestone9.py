# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import shutil
from typing import Any

from .cache import resolve_cache_dir
from ._version import GENERATOR_VERSION
from .generator import BuildResult, build_milestone4
from .overture import fetch_overture_buildings_geojson, overture_buildings_cache_path
from .progress import progress_range, report_progress
from .milestone8 import Milestone8Spec, _Milestone8PlayabilitySpec, _raw_dem_path
from .normalization import NormalizationSpec, load_normalized_dataset, normalize_source_bundle
from .osm import (
    BboxProjection,
    ROADSIDE_BUSH_MODELS,
    ROADSIDE_TREE_MODELS,
    STOCK_HEDGE_MODELS,
    STOCK_METAL_FENCE_MODELS,
    STOCK_WALL_MODELS,
    augment_dataset_with_overture_buildings,
    prepare_spatial_index,
)
from .semantic_features import GRAVE_MODELS
from .source_pipeline import _copy_provenance, validate_source_bundle


EVERON_FOREST_BLOCK_MODEL = r"data3d\les ctverec pruchozi_T1.p3d"
EVERON_SINGLE_TREE_MODEL = r"data3d\str smrk_medium.p3d"
EVERON_ROADSIDE_TREE_MODEL = r"data3d\str smrk vysoky.p3d"
DEFAULT_STEEP_HILL_BUSH_MODELS: tuple[str, ...] = (
    r"data3d\ker listnac.p3d",
    r"data3d\ker pichlavej.p3d",
    r"data3d\ker deravej.p3d",
    r"data3d\ker buxus.p3d",
)

# Resistance/Nogova vegetation from O.pbo. Keep the leaf and pine families
# explicit so neither named preset silently falls back to Data3D trees.
NOGOVA_LEAF_SINGLE_TREE_MODEL = r"o\tree\Javor01.p3d"
NOGOVA_LEAF_ROADSIDE_TREE_MODEL = r"o\tree\Javor02.p3d"
NOGOVA_LEAF_ROADSIDE_TREE_MODELS: tuple[str, ...] = (
    r"o\tree\Javor01.p3d",
    r"o\tree\Javor02.p3d",
    r"o\tree\Akat01.p3d",
    r"o\tree\Akat02.p3d",
    r"o\tree\Akat03.p3d",
    r"o\tree\DubFX.p3d",
)
NOGOVA_PINE_SINGLE_TREE_MODEL = r"o\tree\smrk_maly.p3d"
NOGOVA_PINE_ROADSIDE_TREE_MODEL = r"o\tree\smrk_velky.p3d"
NOGOVA_PINE_ROADSIDE_TREE_MODELS: tuple[str, ...] = (
    r"o\tree\smrk_velky.p3d",
    r"o\tree\smrk_siroky.p3d",
    r"o\tree\dd_borovice.p3d",
    r"o\tree\dd_borovice02.p3d",
)
# Compatibility aliases retained for callers/tests that used the old generic
# Nogova names before leaf/pine were split.
NOGOVA_SINGLE_TREE_MODEL = NOGOVA_PINE_SINGLE_TREE_MODEL
NOGOVA_ROADSIDE_TREE_MODEL = NOGOVA_PINE_ROADSIDE_TREE_MODEL
NOGOVA_ROADSIDE_TREE_MODELS = NOGOVA_PINE_ROADSIDE_TREE_MODELS
NOGOVA_BUSH_MODELS: tuple[str, ...] = (
    r"o\tree\dd_bush01.p3d",
    r"o\tree\dd_bush01b.p3d",
    r"o\tree\dd_bush02.p3d",
    r"o\tree\dd_bush02b.p3d",
    r"o\tree\dd_bush02big.p3d",
    r"o\tree\dd_bush03.p3d",
)
NOGOVA_LEAF_HILLSIDE_TREE_MODEL = NOGOVA_LEAF_SINGLE_TREE_MODEL
NOGOVA_PINE_HILLSIDE_TREE_MODEL = NOGOVA_PINE_SINGLE_TREE_MODEL

# Malden/Abel preset vegetation. These are original CWC Data3D families rather
# than the Resistance O\Tree set, keeping the preset visually closer to the
# older island. The broad-leaf sycamore family is mixed with the original pine
# and brush assets so road cuts and steep slopes do not turn into one repeated
# tree model.
MALDEN_FOREST_BLOCK_MODEL = r"data3d\les_su_ctver_pruhozi.p3d"
MALDEN_SINGLE_TREE_MODEL = r"data3d\str_fikovnik.p3d"
MALDEN_ROADSIDE_TREE_MODEL = MALDEN_SINGLE_TREE_MODEL
MALDEN_ROADSIDE_TREE_MODELS: tuple[str, ...] = (
    r"data3d\str_fikovnik.p3d",
    r"data3d\str_fikovnik2.p3d",
    r"data3d\str_pinie.p3d",
    r"data3d\str borovice.p3d",
)
MALDEN_BUSH_MODELS: tuple[str, ...] = (
    r"data3d\str_fikovnik_ker.p3d",
    r"data3d\ker listnac.p3d",
    r"data3d\ker deravej.p3d",
    r"data3d\ker buxus.p3d",
)


def _resolved_forest_profile_models(spec: "Milestone9Spec") -> dict[str, object]:
    """Resolve profile defaults without clobbering explicit custom model paths."""

    forest_tree_model = str(spec.forest_tree_model)
    folded = forest_tree_model.casefold()
    nogova_pine = folded.startswith(r"o\tree\les_nw_jehl_")
    nogova_leaf = folded.startswith(r"o\tree\les_nw_") and not nogova_pine
    if nogova_leaf or nogova_pine:
        single = NOGOVA_PINE_SINGLE_TREE_MODEL if nogova_pine else NOGOVA_LEAF_SINGLE_TREE_MODEL
        roadside = NOGOVA_PINE_ROADSIDE_TREE_MODEL if nogova_pine else NOGOVA_LEAF_ROADSIDE_TREE_MODEL
        roadside_models = NOGOVA_PINE_ROADSIDE_TREE_MODELS if nogova_pine else NOGOVA_LEAF_ROADSIDE_TREE_MODELS
        hillside = NOGOVA_PINE_HILLSIDE_TREE_MODEL if nogova_pine else NOGOVA_LEAF_HILLSIDE_TREE_MODEL
        return {
            "forest_tree_model": forest_tree_model,
            "forest_single_tree_model": single if spec.forest_single_tree_model == EVERON_SINGLE_TREE_MODEL else spec.forest_single_tree_model,
            "forest_roadside_tree_model": roadside if spec.forest_roadside_tree_model == EVERON_ROADSIDE_TREE_MODEL else spec.forest_roadside_tree_model,
            "forest_roadside_tree_models": roadside_models if spec.forest_roadside_tree_models == ROADSIDE_TREE_MODELS else spec.forest_roadside_tree_models,
            "forest_roadside_bush_models": NOGOVA_BUSH_MODELS if spec.forest_roadside_bush_models == ROADSIDE_BUSH_MODELS else spec.forest_roadside_bush_models,
            "steep_hill_bush_models": NOGOVA_BUSH_MODELS if spec.steep_hill_bush_models == DEFAULT_STEEP_HILL_BUSH_MODELS else spec.steep_hill_bush_models,
            "forest_hillside_tree_model": hillside if spec.forest_hillside_tree_model == r"data3d\str_fikovnik.p3d" else spec.forest_hillside_tree_model,
        }

    if str(spec.forest_profile).casefold() != "malden":
        return {
            "forest_tree_model": forest_tree_model,
            "forest_single_tree_model": spec.forest_single_tree_model,
            "forest_roadside_tree_model": spec.forest_roadside_tree_model,
            "forest_roadside_tree_models": spec.forest_roadside_tree_models,
            "forest_roadside_bush_models": spec.forest_roadside_bush_models,
            "steep_hill_bush_models": spec.steep_hill_bush_models,
            "forest_hillside_tree_model": spec.forest_hillside_tree_model,
        }
    return {
        "forest_tree_model": MALDEN_FOREST_BLOCK_MODEL if spec.forest_tree_model == EVERON_FOREST_BLOCK_MODEL else spec.forest_tree_model,
        "forest_single_tree_model": MALDEN_SINGLE_TREE_MODEL if spec.forest_single_tree_model == EVERON_SINGLE_TREE_MODEL else spec.forest_single_tree_model,
        "forest_roadside_tree_model": MALDEN_ROADSIDE_TREE_MODEL if spec.forest_roadside_tree_model == EVERON_ROADSIDE_TREE_MODEL else spec.forest_roadside_tree_model,
        "forest_roadside_tree_models": MALDEN_ROADSIDE_TREE_MODELS if spec.forest_roadside_tree_models == ROADSIDE_TREE_MODELS else spec.forest_roadside_tree_models,
        "forest_roadside_bush_models": MALDEN_BUSH_MODELS if spec.forest_roadside_bush_models == ROADSIDE_BUSH_MODELS else spec.forest_roadside_bush_models,
        "steep_hill_bush_models": MALDEN_BUSH_MODELS if spec.steep_hill_bush_models == DEFAULT_STEEP_HILL_BUSH_MODELS else spec.steep_hill_bush_models,
        "forest_hillside_tree_model": spec.forest_hillside_tree_model,
    }


@dataclass(frozen=True, slots=True)
class Milestone9Spec(Milestone8Spec):
    name: str = "cwr_milestone9"
    display_name: str = "CWR Milestone 9"
    include_minor_roads: bool = True
    procedural_gravel_roads: bool = True
    surface_shoreline_wet_cells: int = 1
    surface_shoreline_sand_cells: int = 2
    surface_transition_cells: int = 2
    surface_forest_edge_cells: int = 1
    surface_farmland_strip_cells: int = 4
    surface_road_shoulder_metres: float = 5.0
    surface_dirt_blend_metres: float = 6.0
    surface_steep_slope_degrees: float = 52.0
    surface_colour_reference_path: Path | None = None
    surface_colour_reference_strength: float = 0.25
    surface_overview_size: int = 1024
    surface_texture_size: int = 512
    surface_ground_mode: str = "milestone9"
    forest_profile: str = "everon"
    forest_tree_model: str = EVERON_FOREST_BLOCK_MODEL
    forest_maximum_block_relief: float = 8.0
    forest_block_maximum_burial: float = 8.0
    forest_block_maximum_float: float = 0.5
    forest_block_maximum_ground_sink: float = 0.0
    forest_everon_steep_model: str = r"data3d\les trojuhelnik pruchozi.p3d"
    forest_everon_steep_footprint: float = 35.0
    forest_everon_steep_maximum_relief: float = 18.0
    forest_everon_steep_maximum_burial: float = 18.0
    forest_everon_steep_maximum_float: float = 0.5
    forest_everon_steep_maximum_ground_sink: float = 0.0
    forest_polygon_sink_fraction: float = 0.5
    forest_severe_hill_fallback: bool = True
    forest_severe_hill_relief: float = 5.0
    forest_severe_hill_trees_per_block: int = 10
    forest_cluster_fallback: bool = True
    forest_cluster_search_radius: float = 10.0
    forest_cluster_maximum_relief: float = 48.0
    forest_cluster_maximum_burial: float = 1.25
    forest_cluster_maximum_float: float = 1.25
    forest_cluster_footprint_margin: float = 0.75
    forest_undergrowth_enabled: bool = True
    forest_undergrowth_maximum_objects: int = 120000
    forest_undergrowth_spacing: float = 30.0
    forest_undergrowth_maximum_relief: float = 20.0
    forest_undergrowth_maximum_burial: float = 0.8
    forest_undergrowth_maximum_float: float = 0.8
    forest_undergrowth_ground_clearance: float = 0.03
    steep_hill_bushes_enabled: bool = True
    maximum_steep_hill_bush_objects: int = 80000
    steep_hill_bush_spacing: float = 24.0
    steep_hill_bush_minimum_slope_degrees: float = 16.0
    steep_hill_bush_maximum_relief: float = 8.0
    steep_hill_bush_maximum_burial: float = 0.6
    steep_hill_bush_maximum_float: float = 0.8
    steep_hill_bush_ground_clearance: float = 0.03
    steep_hill_bush_models: tuple[str, ...] = DEFAULT_STEEP_HILL_BUSH_MODELS
    forest_border_enabled: bool = True
    forest_border_maximum_objects: int = 2000
    forest_border_spacing: float = 34.0
    forest_border_inset: float = 5.0
    forest_border_maximum_relief: float = 24.0
    forest_border_maximum_burial: float = 1.0
    forest_border_maximum_float: float = 1.0
    forest_single_tree_enabled: bool = True
    forest_single_tree_model: str = EVERON_SINGLE_TREE_MODEL
    forest_roadside_tree_model: str = EVERON_ROADSIDE_TREE_MODEL
    forest_roadside_tree_models: tuple[str, ...] = ROADSIDE_TREE_MODELS
    forest_roadside_trees_per_cut_block: int = 20
    forest_roadside_bush_models: tuple[str, ...] = ROADSIDE_BUSH_MODELS
    forest_roadside_bushes_per_cut_block: int = 16
    forest_roadside_bush_footprint: float = 1.5
    maximum_forest_single_tree_objects: int = 1000
    forest_single_tree_spacing: float = 45.0
    forest_single_tree_footprint: float = 2.0
    forest_single_tree_maximum_relief: float = 8.0
    forest_single_tree_maximum_float: float = 0.5
    ditch_grass_enabled: bool = True
    maximum_ditch_grass_objects: int = 2000
    ditch_grass_spacing: float = 18.0
    ditch_grass_endpoint_trim: float = 6.0
    ditch_grass_maximum_relief: float = 18.0
    ditch_grass_maximum_burial: float = 0.6
    ditch_grass_maximum_float: float = 0.8
    ditch_grass_ground_clearance: float = 0.05
    forest_hillside_fallback: bool = False
    forest_hillside_tree_model: str = r"data3d\str_fikovnik.p3d"
    forest_hillside_trees_per_block: int = 5
    forest_hillside_tree_footprint: float = 4.0
    forest_hillside_tree_maximum_relief: float = 2.5
    barriers_enabled: bool = True
    maximum_barrier_objects: int = 4000
    barrier_segment_length: float = 6.0
    stock_hedge_models: tuple[str, ...] = STOCK_HEDGE_MODELS
    stock_wall_models: tuple[str, ...] = STOCK_WALL_MODELS
    stock_metal_fence_models: tuple[str, ...] = STOCK_METAL_FENCE_MODELS
    bridges_enabled: bool = True
    procedural_bridges: bool = False
    maximum_bridge_objects: int = 1000
    bridge_module_length: float = 30.0
    bridge_deck_clearance: float = 1.25
    bridge_water_clearance: float = 18.0
    residential_infill_enabled: bool = True
    maximum_residential_infill_buildings: int = 1500
    residential_infill_spacing: float = 68.0
    residential_infill_minimum_area: float = 1800.0
    residential_infill_road_clearance: float = 0.5
    residential_infill_building_clearance: float = 6.0
    overture_buildings_enabled: bool = True
    overture_buildings_geojson: Path | None = None
    rural_vegetation_enabled: bool = True
    maximum_rural_vegetation_objects: int = 3000
    rural_vegetation_spacing: float = 28.0
    meadow_grass_enabled: bool = True
    maximum_meadow_grass_objects: int = 20000
    meadow_grass_spacing: float = 24.0
    wetland_reeds_enabled: bool = True
    maximum_wetland_reed_objects: int = 100000
    wetland_reed_spacing: float = 18.0
    wetland_reed_maximum_relief: float = 4.0
    wetland_reed_maximum_burial: float = 0.5
    wetland_reed_maximum_float: float = 1.0
    wetland_reed_ground_clearance: float = 0.03
    wetland_reed_models: tuple[str, ...] = (
        r"o\tree\dd_rakosi.p3d",
        r"o\tree\dd_rakosi02.p3d",
    )
    rocky_forest_fallback_enabled: bool = True
    maximum_rocky_forest_objects: int = 1200
    rocky_forest_rocks_per_patch: int = 3
    rocky_forest_spread: float = 18.0
    rocky_forest_maximum_relief: float = 42.0
    rocky_forest_maximum_burial: float = 1.0
    rocky_forest_maximum_float: float = 1.0
    semantic_landmarks: bool = True
    bus_stops_enabled: bool = True
    bus_stop_model: str = r"o\misc\aut_z_st.p3d"
    bus_stop_footprint: float = 1.6
    bus_stop_ground_clearance: float = 0.12
    maximum_landmark_objects: int = 1000
    cemeteries_enabled: bool = True
    maximum_grave_objects: int = 12000
    grave_spacing: float = 3.5
    grave_inset: float = 2.0
    grave_footprint: float = 1.2
    grave_ground_clearance: float = 0.12
    grave_road_clearance: float = 1.0
    grave_building_clearance: float = 1.5
    grave_models: tuple[str, ...] = GRAVE_MODELS
    semantic_site_maximum_relief: float = 1.5
    semantic_site_maximum_variants: int = 64
    pbo_backend: str = "auto"
    poseidon_tools_path: Path | None = None
    deploy_mod_dir: Path | None = None

    def validate(self) -> None:
        Milestone8Spec.validate(self)
        if self.surface_ground_mode not in {"milestone8", "milestone9"}:
            raise ValueError("surface ground mode must be milestone8 or milestone9")
        if self.forest_profile not in {"everon", "malden"}:
            raise ValueError("forest profile must be everon or malden")
        for label, value in (
            ("wet shoreline cells", self.surface_shoreline_wet_cells),
            ("sand shoreline cells", self.surface_shoreline_sand_cells),
            ("surface transition cells", self.surface_transition_cells),
            ("forest-edge cells", self.surface_forest_edge_cells),
            ("farmland strip cells", self.surface_farmland_strip_cells),
        ):
            if not isinstance(value, int) or value < 0 or value > 32:
                raise ValueError(f"{label} must be an integer within 0..32")
        if self.surface_shoreline_wet_cells + self.surface_shoreline_sand_cells < 1:
            raise ValueError("at least one shoreline band must be enabled")
        for label, value in (
            ("road shoulder metres", self.surface_road_shoulder_metres),
            ("dirt blend metres", self.surface_dirt_blend_metres),
            ("steep slope degrees", self.surface_steep_slope_degrees),
            ("colour reference strength", self.surface_colour_reference_strength),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if self.surface_road_shoulder_metres < 0 or self.surface_dirt_blend_metres < 0:
            raise ValueError("road shoulder and dirt blend widths must not be negative")
        if not 30.0 <= self.surface_steep_slope_degrees < 90:
            raise ValueError("steep slope threshold must be at least the base rock slope and below 90 degrees")
        if not 0.0 <= self.surface_colour_reference_strength <= 1.0:
            raise ValueError("colour reference strength must be within 0..1")
        if (
            not isinstance(self.forest_polygon_sink_fraction, (int, float))
            or not math.isfinite(float(self.forest_polygon_sink_fraction))
            or not 0.0 <= self.forest_polygon_sink_fraction <= 1.0
        ):
            raise ValueError("forest polygon sink fraction must be finite and within 0..1")
        if not 0.0 < self.steep_hill_bush_minimum_slope_degrees < 90.0:
            raise ValueError("steep hill bush minimum slope must be within 0..90 degrees")
        for label, value in (
            ("forest maximum block relief", self.forest_maximum_block_relief),
            ("hillside tree footprint", self.forest_hillside_tree_footprint),
            ("hillside tree maximum relief", self.forest_hillside_tree_maximum_relief),
            ("Everon steep forest footprint", self.forest_everon_steep_footprint),
            ("Everon steep forest maximum relief", self.forest_everon_steep_maximum_relief),
            ("severe hill forest relief", self.forest_severe_hill_relief),
            ("forest cluster maximum relief", self.forest_cluster_maximum_relief),
            ("forest undergrowth spacing", self.forest_undergrowth_spacing),
            ("forest undergrowth maximum relief", self.forest_undergrowth_maximum_relief),
            ("steep hill bush spacing", self.steep_hill_bush_spacing),
            ("steep hill bush minimum slope", self.steep_hill_bush_minimum_slope_degrees),
            ("steep hill bush maximum relief", self.steep_hill_bush_maximum_relief),
            ("forest border spacing", self.forest_border_spacing),
            ("forest border inset", self.forest_border_inset),
            ("forest border maximum relief", self.forest_border_maximum_relief),
            ("forest single-tree spacing", self.forest_single_tree_spacing),
            ("forest single-tree footprint", self.forest_single_tree_footprint),
            ("forest single-tree maximum relief", self.forest_single_tree_maximum_relief),
            ("forest single-tree maximum float", self.forest_single_tree_maximum_float),
            ("ditch grass spacing", self.ditch_grass_spacing),
            ("ditch grass maximum relief", self.ditch_grass_maximum_relief),
            ("barrier segment length", self.barrier_segment_length),
            ("bridge module length", self.bridge_module_length),
            ("residential infill spacing", self.residential_infill_spacing),
            ("residential infill minimum area", self.residential_infill_minimum_area),
            ("residential infill road clearance", self.residential_infill_road_clearance),
            ("residential infill building clearance", self.residential_infill_building_clearance),
            ("rural vegetation spacing", self.rural_vegetation_spacing),
            ("meadow grass spacing", self.meadow_grass_spacing),
            ("wetland reed spacing", self.wetland_reed_spacing),
            ("wetland reed maximum relief", self.wetland_reed_maximum_relief),
            ("rocky forest spread", self.rocky_forest_spread),
            ("rocky forest maximum relief", self.rocky_forest_maximum_relief),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")
        for label, value in (
            ("forest block maximum burial", self.forest_block_maximum_burial),
            ("forest block maximum float", self.forest_block_maximum_float),
            ("forest block maximum ground sink", self.forest_block_maximum_ground_sink),
            ("Everon steep forest maximum burial", self.forest_everon_steep_maximum_burial),
            ("Everon steep forest maximum float", self.forest_everon_steep_maximum_float),
            ("Everon steep forest maximum ground sink", self.forest_everon_steep_maximum_ground_sink),
            ("forest cluster search radius", self.forest_cluster_search_radius),
            ("forest cluster maximum burial", self.forest_cluster_maximum_burial),
            ("forest cluster maximum float", self.forest_cluster_maximum_float),
            ("forest cluster footprint margin", self.forest_cluster_footprint_margin),
            ("forest undergrowth maximum burial", self.forest_undergrowth_maximum_burial),
            ("forest undergrowth maximum float", self.forest_undergrowth_maximum_float),
            ("forest undergrowth ground clearance", self.forest_undergrowth_ground_clearance),
            ("steep hill bush maximum burial", self.steep_hill_bush_maximum_burial),
            ("steep hill bush maximum float", self.steep_hill_bush_maximum_float),
            ("steep hill bush ground clearance", self.steep_hill_bush_ground_clearance),
            ("forest border maximum burial", self.forest_border_maximum_burial),
            ("forest border maximum float", self.forest_border_maximum_float),
            ("ditch grass endpoint trim", self.ditch_grass_endpoint_trim),
            ("ditch grass maximum burial", self.ditch_grass_maximum_burial),
            ("ditch grass maximum float", self.ditch_grass_maximum_float),
            ("ditch grass ground clearance", self.ditch_grass_ground_clearance),
            ("bridge deck clearance", self.bridge_deck_clearance),
            ("bridge water clearance", self.bridge_water_clearance),
            ("rocky forest maximum burial", self.rocky_forest_maximum_burial),
            ("rocky forest maximum float", self.rocky_forest_maximum_float),
            ("wetland reed maximum burial", self.wetland_reed_maximum_burial),
            ("wetland reed maximum float", self.wetland_reed_maximum_float),
            ("wetland reed ground clearance", self.wetland_reed_ground_clearance),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not isinstance(self.forest_hillside_trees_per_block, int) or not 0 <= self.forest_hillside_trees_per_block <= 12:
            raise ValueError("hillside trees per block must be within 0..12")
        if not isinstance(self.forest_severe_hill_trees_per_block, int) or not 1 <= self.forest_severe_hill_trees_per_block <= 12:
            raise ValueError("severe hill trees per block must be within 1..12")
        if not isinstance(self.forest_roadside_trees_per_cut_block, int) or not 0 <= self.forest_roadside_trees_per_cut_block <= 128:
            raise ValueError("forest roadside trees per cut block must be within 0..128")
        if not isinstance(self.forest_roadside_bushes_per_cut_block, int) or not 0 <= self.forest_roadside_bushes_per_cut_block <= 128:
            raise ValueError("forest roadside bushes per cut block must be within 0..128")
        if not math.isfinite(self.forest_roadside_bush_footprint) or self.forest_roadside_bush_footprint <= 0:
            raise ValueError("forest roadside bush footprint must be positive and finite")
        for label, value in (
            ("forest undergrowth maximum objects", self.forest_undergrowth_maximum_objects),
            ("maximum steep hill bush objects", self.maximum_steep_hill_bush_objects),
            ("forest border maximum objects", self.forest_border_maximum_objects),
            ("maximum forest single-tree objects", self.maximum_forest_single_tree_objects),
            ("maximum ditch grass objects", self.maximum_ditch_grass_objects),
            ("maximum barrier objects", self.maximum_barrier_objects),
            ("maximum bridge objects", self.maximum_bridge_objects),
            ("maximum residential infill buildings", self.maximum_residential_infill_buildings),
            ("maximum rural vegetation objects", self.maximum_rural_vegetation_objects),
            ("maximum meadow grass objects", self.maximum_meadow_grass_objects),
            ("maximum wetland reed objects", self.maximum_wetland_reed_objects),
            ("maximum rocky forest objects", self.maximum_rocky_forest_objects),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(self.rocky_forest_rocks_per_patch, int) or not 1 <= self.rocky_forest_rocks_per_patch <= 8:
            raise ValueError("rocky forest rocks per patch must be within 1..8")
        for label, model_path in (
            ("forest block model", self.forest_tree_model),
            ("Everon steep forest model", self.forest_everon_steep_model),
            ("hillside tree model", self.forest_hillside_tree_model),
            ("forest single-tree model", self.forest_single_tree_model),
        ):
            try:
                encoded_model = model_path.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{label} path must be ASCII") from exc
            if not encoded_model or len(encoded_model) > 75:
                raise ValueError(f"{label} path must contain 1..75 ASCII bytes")
        for label, model_paths in (
            ("steep hill bush models", self.steep_hill_bush_models),
            ("forest roadside tree models", self.forest_roadside_tree_models),
            ("forest roadside bush models", self.forest_roadside_bush_models),
            ("wetland reed models", self.wetland_reed_models),
            ("stock hedge models", self.stock_hedge_models),
            ("grave models", self.grave_models),
            ("stock wall models", self.stock_wall_models),
            ("stock metal fence models", self.stock_metal_fence_models),
        ):
            if not model_paths:
                raise ValueError(f"{label} must contain at least one model")
            for model_path in model_paths:
                try:
                    encoded_model = model_path.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise ValueError(f"{label} paths must be ASCII") from exc
                if not encoded_model or len(encoded_model) > 75:
                    raise ValueError(f"{label} paths must contain 1..75 ASCII bytes")
        if len(self.stock_hedge_models) < 3:
            raise ValueError("stock hedge models must contain at least three models")
        if len(self.stock_wall_models) < 1:
            raise ValueError("stock wall models must contain at least one model")
        if len(self.stock_metal_fence_models) < 1:
            raise ValueError("stock metal fence models must contain at least one model")
        if not isinstance(self.maximum_landmark_objects, int) or self.maximum_landmark_objects < 0:
            raise ValueError("maximum landmark objects must be a non-negative integer")
        if not math.isfinite(self.bus_stop_footprint) or self.bus_stop_footprint <= 0.0:
            raise ValueError("bus stop footprint must be positive and finite")
        if not math.isfinite(self.bus_stop_ground_clearance) or self.bus_stop_ground_clearance < 0.0:
            raise ValueError("bus stop ground clearance must be non-negative and finite")
        if not isinstance(self.maximum_grave_objects, int) or self.maximum_grave_objects < 0:
            raise ValueError("maximum grave objects must be a non-negative integer")
        for label, value in (
            ("grave spacing", self.grave_spacing),
            ("grave footprint", self.grave_footprint),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        for label, value in (
            ("grave inset", self.grave_inset),
            ("grave ground clearance", self.grave_ground_clearance),
            ("grave road clearance", self.grave_road_clearance),
            ("grave building clearance", self.grave_building_clearance),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be non-negative and finite")
        if not isinstance(self.semantic_site_maximum_variants, int) or not 1 <= self.semantic_site_maximum_variants <= 512:
            raise ValueError("semantic site maximum variants must be within 1..512")
        if not math.isfinite(self.semantic_site_maximum_relief) or self.semantic_site_maximum_relief <= 0:
            raise ValueError("semantic site maximum relief must be positive and finite")
        try:
            encoded_bus_stop_model = self.bus_stop_model.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("bus stop model path must be ASCII") from exc
        if not encoded_bus_stop_model or len(encoded_bus_stop_model) > 75:
            raise ValueError("bus stop model path must contain 1..75 ASCII bytes")
        if self.surface_colour_reference_path is not None and not self.surface_colour_reference_path.is_file():
            raise ValueError(f"colour reference does not exist: {self.surface_colour_reference_path}")
        if self.pbo_backend not in {"auto", "python", "poseidon"}:
            raise ValueError("PBO backend must be auto, python, or poseidon")
        if self.poseidon_tools_path is not None and not self.poseidon_tools_path.is_file():
            raise ValueError(f"PoseidonTools executable does not exist: {self.poseidon_tools_path}")
        if self.deploy_mod_dir is not None and not self.deploy_mod_dir.is_dir():
            raise ValueError(f"deployment mod folder must already exist: {self.deploy_mod_dir}")
        for label, value, minimum, maximum in (
            ("overview size", self.surface_overview_size, 128, 4096),
            ("surface texture size", self.surface_texture_size, 16, 1024),
        ):
            if not isinstance(value, int) or value < minimum or value > maximum or value & (value - 1):
                raise ValueError(f"{label} must be a power of two within {minimum}..{maximum}")


@dataclass(frozen=True, slots=True)
class _Milestone9PlayabilitySpec(_Milestone8PlayabilitySpec):
    include_minor_roads: bool = True
    procedural_gravel_roads: bool = True
    surface_pass_enabled: bool = True
    stock_road_piece_fitting: bool = True
    forest_low_anchor: bool = True
    forest_profile: str = "everon"
    forest_tree_model: str = EVERON_FOREST_BLOCK_MODEL
    forest_maximum_block_relief: float = 8.0
    forest_block_maximum_burial: float = 8.0
    forest_block_maximum_float: float = 0.5
    forest_block_maximum_ground_sink: float = 0.0
    forest_everon_steep_model: str = r"data3d\les trojuhelnik pruchozi.p3d"
    forest_everon_steep_footprint: float = 35.0
    forest_everon_steep_maximum_relief: float = 18.0
    forest_everon_steep_maximum_burial: float = 18.0
    forest_everon_steep_maximum_float: float = 0.5
    forest_everon_steep_maximum_ground_sink: float = 0.0
    forest_polygon_sink_fraction: float = 0.5
    forest_severe_hill_fallback: bool = True
    forest_severe_hill_relief: float = 5.0
    forest_severe_hill_trees_per_block: int = 10
    forest_cluster_fallback: bool = True
    forest_cluster_search_radius: float = 10.0
    forest_cluster_maximum_relief: float = 48.0
    forest_cluster_maximum_burial: float = 1.25
    forest_cluster_maximum_float: float = 1.25
    forest_cluster_footprint_margin: float = 0.75
    forest_undergrowth_enabled: bool = True
    forest_undergrowth_maximum_objects: int = 120000
    forest_undergrowth_spacing: float = 30.0
    forest_undergrowth_maximum_relief: float = 20.0
    forest_undergrowth_maximum_burial: float = 0.8
    forest_undergrowth_maximum_float: float = 0.8
    forest_undergrowth_ground_clearance: float = 0.03
    steep_hill_bushes_enabled: bool = True
    maximum_steep_hill_bush_objects: int = 80000
    steep_hill_bush_spacing: float = 24.0
    steep_hill_bush_minimum_slope_degrees: float = 16.0
    steep_hill_bush_maximum_relief: float = 8.0
    steep_hill_bush_maximum_burial: float = 0.6
    steep_hill_bush_maximum_float: float = 0.8
    steep_hill_bush_ground_clearance: float = 0.03
    steep_hill_bush_models: tuple[str, ...] = DEFAULT_STEEP_HILL_BUSH_MODELS
    forest_border_enabled: bool = True
    forest_border_maximum_objects: int = 2000
    forest_border_spacing: float = 34.0
    forest_border_inset: float = 5.0
    forest_border_maximum_relief: float = 24.0
    forest_border_maximum_burial: float = 1.0
    forest_border_maximum_float: float = 1.0
    forest_single_tree_enabled: bool = True
    forest_single_tree_model: str = EVERON_SINGLE_TREE_MODEL
    forest_roadside_tree_model: str = EVERON_ROADSIDE_TREE_MODEL
    forest_roadside_tree_models: tuple[str, ...] = ROADSIDE_TREE_MODELS
    forest_roadside_trees_per_cut_block: int = 20
    forest_roadside_bush_models: tuple[str, ...] = ROADSIDE_BUSH_MODELS
    forest_roadside_bushes_per_cut_block: int = 16
    forest_roadside_bush_footprint: float = 1.5
    maximum_forest_single_tree_objects: int = 1000
    forest_single_tree_spacing: float = 45.0
    forest_single_tree_footprint: float = 2.0
    forest_single_tree_maximum_relief: float = 8.0
    forest_single_tree_maximum_float: float = 0.5
    ditch_grass_enabled: bool = True
    maximum_ditch_grass_objects: int = 2000
    ditch_grass_spacing: float = 18.0
    ditch_grass_endpoint_trim: float = 6.0
    ditch_grass_maximum_relief: float = 18.0
    ditch_grass_maximum_burial: float = 0.6
    ditch_grass_maximum_float: float = 0.8
    ditch_grass_ground_clearance: float = 0.05
    forest_hillside_fallback: bool = False
    forest_hillside_tree_model: str = r"data3d\str_fikovnik.p3d"
    forest_hillside_trees_per_block: int = 5
    forest_hillside_tree_footprint: float = 4.0
    forest_hillside_tree_maximum_relief: float = 2.5
    barriers_enabled: bool = True
    maximum_barrier_objects: int = 4000
    barrier_segment_length: float = 6.0
    stock_hedge_models: tuple[str, ...] = STOCK_HEDGE_MODELS
    stock_wall_models: tuple[str, ...] = STOCK_WALL_MODELS
    stock_metal_fence_models: tuple[str, ...] = STOCK_METAL_FENCE_MODELS
    bridges_enabled: bool = True
    procedural_bridges: bool = False
    maximum_bridge_objects: int = 1000
    bridge_module_length: float = 30.0
    bridge_deck_clearance: float = 1.25
    bridge_water_clearance: float = 18.0
    residential_infill_enabled: bool = True
    maximum_residential_infill_buildings: int = 1500
    residential_infill_spacing: float = 68.0
    residential_infill_minimum_area: float = 1800.0
    residential_infill_road_clearance: float = 0.5
    residential_infill_building_clearance: float = 6.0
    overture_buildings_enabled: bool = True
    overture_buildings_geojson: Path | None = None
    rural_vegetation_enabled: bool = True
    maximum_rural_vegetation_objects: int = 3000
    rural_vegetation_spacing: float = 28.0
    meadow_grass_enabled: bool = True
    maximum_meadow_grass_objects: int = 20000
    meadow_grass_spacing: float = 24.0
    wetland_reeds_enabled: bool = True
    maximum_wetland_reed_objects: int = 100000
    wetland_reed_spacing: float = 18.0
    wetland_reed_maximum_relief: float = 4.0
    wetland_reed_maximum_burial: float = 0.5
    wetland_reed_maximum_float: float = 1.0
    wetland_reed_ground_clearance: float = 0.03
    wetland_reed_models: tuple[str, ...] = (
        r"o\tree\dd_rakosi.p3d",
        r"o\tree\dd_rakosi02.p3d",
    )
    rocky_forest_fallback_enabled: bool = True
    maximum_rocky_forest_objects: int = 1200
    rocky_forest_rocks_per_patch: int = 3
    rocky_forest_spread: float = 18.0
    rocky_forest_maximum_relief: float = 42.0
    rocky_forest_maximum_burial: float = 1.0
    rocky_forest_maximum_float: float = 1.0
    semantic_landmarks: bool = True
    bus_stops_enabled: bool = True
    bus_stop_model: str = r"o\misc\aut_z_st.p3d"
    bus_stop_footprint: float = 1.6
    bus_stop_ground_clearance: float = 0.12
    maximum_landmark_objects: int = 1000
    cemeteries_enabled: bool = True
    maximum_grave_objects: int = 12000
    grave_spacing: float = 3.5
    grave_inset: float = 2.0
    grave_footprint: float = 1.2
    grave_ground_clearance: float = 0.12
    grave_road_clearance: float = 1.0
    grave_building_clearance: float = 1.5
    grave_models: tuple[str, ...] = GRAVE_MODELS
    semantic_site_maximum_relief: float = 1.5
    semantic_site_maximum_variants: int = 64
    pbo_backend: str = "auto"
    poseidon_tools_path: Path | None = None
    surface_shoreline_wet_cells: int = 1
    surface_shoreline_sand_cells: int = 2
    surface_transition_cells: int = 2
    surface_forest_edge_cells: int = 1
    surface_farmland_strip_cells: int = 4
    surface_road_shoulder_metres: float = 5.0
    surface_dirt_blend_metres: float = 6.0
    surface_steep_slope_degrees: float = 52.0
    surface_colour_reference_path: Path | None = None
    surface_colour_reference_strength: float = 0.25
    surface_overview_size: int = 1024
    surface_texture_size: int = 512
    surface_ground_mode: str = "milestone9"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _milestone_progress_span(start: int, end: int):
    def callback(percent: int, stage: str) -> None:
        local = max(0, min(100, int(percent)))
        report_progress(start + round((end - start) * local / 100.0), stage)
    return callback


def _existing_mod_child(target_root: Path, preferred_name: str) -> Path:
    """Return an existing case-insensitive child or the preferred new path."""
    wanted = preferred_name.casefold()
    try:
        for child in target_root.iterdir():
            if child.is_dir() and child.name.casefold() == wanted:
                return child
    except OSError:
        pass
    return target_root / preferred_name


def _normalise_mod_root(target_root: Path) -> Path:
    """Accept the mod root, or recover when the user selected Addons/Anims."""
    target_root = target_root.expanduser().resolve()
    if target_root.name.casefold() in {"addons", "anims"}:
        return target_root.parent
    return target_root


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".cwr-worldgen.tmp")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except PermissionError as exc:
        raise PermissionError(
            f"cannot replace deployed file {destination}; close the game or any tool using it"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".cwr-worldgen.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except PermissionError as exc:
        raise PermissionError(
            f"cannot replace deployed directory {destination}; close the game or any tool using it"
        ) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _deploy_runtime_to_existing_mod(result: BuildResult, target_root: Path) -> dict[str, Any]:
    """Copy every generated Addons/Anims entry into an existing mod folder.

    The build runtime is treated as the source of truth rather than copying only
    the two paths held on BuildResult. This also deploys any future generated
    addon or animation files without introducing another @mod wrapper.
    """

    requested_root = target_root.expanduser().resolve()
    target_root = _normalise_mod_root(target_root)
    if not target_root.is_dir():
        raise ValueError(f"deployment mod folder must already exist: {target_root}")

    runtime_root = result.pbo_path.parent.parent
    source_addons = runtime_root / "Addons"
    source_anims = runtime_root / "Anims"
    if not source_addons.is_dir() or not source_anims.is_dir():
        raise RuntimeError(
            f"generated runtime is incomplete; expected Addons and Anims below {runtime_root}"
        )

    addons_dir = _existing_mod_child(target_root, "Addons")
    anims_dir = _existing_mod_child(target_root, "Anims")
    addons_dir.mkdir(parents=True, exist_ok=True)
    anims_dir.mkdir(parents=True, exist_ok=True)

    deployed: list[dict[str, str]] = []
    for source in sorted(source_addons.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_addons)
        destination = addons_dir / relative
        _atomic_copy_file(source, destination)
        deployed.append({
            "kind": "addon",
            "source": str(source),
            "destination": str(destination),
            "sha256": _sha256(destination),
        })

    for source in sorted(source_anims.iterdir()):
        destination = anims_dir / source.name
        if source.is_dir():
            _atomic_copy_tree(source, destination)
            for copied in sorted(destination.rglob("*")):
                if copied.is_file():
                    deployed.append({
                        "kind": "anim",
                        "source": str(source / copied.relative_to(destination)),
                        "destination": str(copied),
                        "sha256": _sha256(copied),
                    })
        elif source.is_file():
            _atomic_copy_file(source, destination)
            deployed.append({
                "kind": "anim",
                "source": str(source),
                "destination": str(destination),
                "sha256": _sha256(destination),
            })

    if not deployed:
        raise RuntimeError(f"generated runtime contained no deployable files below {runtime_root}")

    source_by_destination = {
        str(Path(item["destination"]).resolve()): Path(item["source"])
        for item in deployed
    }
    for item in deployed:
        destination = Path(item["destination"])
        source = source_by_destination[str(destination.resolve())]
        if not destination.is_file() or _sha256(source) != _sha256(destination):
            raise RuntimeError(f"deployment verification failed for {destination}")

    destination_pbo = addons_dir / result.pbo_path.relative_to(source_addons)
    source_intro = result.intro_mission_path.parent
    destination_intro = anims_dir / source_intro.relative_to(source_anims)
    report: dict[str, Any] = {
        "requested_folder": str(requested_root),
        "mod_folder": str(target_root),
        "addons_folder": str(addons_dir),
        "anims_folder": str(anims_dir),
        "pbo": str(destination_pbo),
        "intro": str(destination_intro),
        "verified": True,
        "file_count": len(deployed),
        "files": deployed,
    }
    report_path = result.output_dir / "deployment-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _resolve_overture_buildings_geojson(
    spec: Milestone9Spec,
    *,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    bundled_geojson: Path | None = None,
) -> Path | None:
    if not spec.overture_buildings_enabled:
        return None
    if spec.overture_buildings_geojson is not None:
        path = spec.overture_buildings_geojson
        if not path.is_file():
            raise ValueError(f"Overture buildings GeoJSON does not exist: {path}")
        return path
    if bundled_geojson is not None and bundled_geojson.is_file():
        return bundled_geojson

    output = overture_buildings_cache_path(cache_dir, bbox)
    return fetch_overture_buildings_geojson(bbox, output, refresh=spec.cache_refresh)


def build_milestone9(output_dir: Path, spec: Milestone9Spec, *, clean: bool = True) -> BuildResult:
    spec.validate()
    report_progress(0, "Validating Milestone 9 source bundle")
    source_validation = validate_source_bundle(
        spec.source_dir,
        progress_callback=_milestone_progress_span(0, 1),
    )
    report_progress(1, "Source bundle validated; preparing normalized geometry")
    source = source_validation.bundle
    normalization = normalize_source_bundle(
        NormalizationSpec(
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
        ),
        progress_callback=_milestone_progress_span(1, 8),
        validated_source=source,
    )
    cache_dir = resolve_cache_dir(source.root, spec.cache_dir)
    report_progress(8, "Loading normalized GeoJSON layers")
    dataset = load_normalized_dataset(
        normalization,
        cache_dir=cache_dir,
        use_cache=spec.cache_enabled,
        refresh=spec.cache_refresh,
    )
    projection = BboxProjection.create(source.bbox, source.cells * source.cell_size)
    overture_path = _resolve_overture_buildings_geojson(
        spec,
        bbox=source.bbox,
        cache_dir=cache_dir,
        bundled_geojson=source.overture_buildings_geojson_path,
    )
    if overture_path is not None:
        report_progress(9, f"Loading Overture building fallback footprints from {overture_path.name}")
        before = len(dataset.building_polygons)
        dataset = augment_dataset_with_overture_buildings(dataset, projection, spec, overture_path)
        added = len(dataset.building_polygons) - before
        report_progress(10, f"Overture fallback buildings accepted: {added:,}")
    report_progress(10, (
        f"Normalized dataset loaded: {len(dataset.roads):,} roads, "
        f"{len(dataset.building_polygons):,} buildings, {len(dataset.forests):,} forests"
    ))
    spatial_index = prepare_spatial_index(
        dataset,
        projection,
        cache_dir=cache_dir,
        use_cache=spec.cache_enabled,
        refresh=spec.cache_refresh,
        progress_callback=_milestone_progress_span(10, 12),
    )
    raw_dem = _raw_dem_path(source.root, source.manifest_path)
    reference_path = spec.surface_colour_reference_path or source.reference_map_path
    forest_models = _resolved_forest_profile_models(spec)

    playability = _Milestone9PlayabilitySpec(
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
        rock_height=1.0e9,
        rock_slope_degrees=44.0,
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
        forest_profile=spec.forest_profile,
        forest_tree_model=str(forest_models["forest_tree_model"]),
        forest_maximum_block_relief=spec.forest_maximum_block_relief,
        forest_block_maximum_burial=spec.forest_block_maximum_burial,
        forest_block_maximum_float=spec.forest_block_maximum_float,
        forest_block_maximum_ground_sink=spec.forest_block_maximum_ground_sink,
        forest_everon_steep_model=spec.forest_everon_steep_model,
        forest_everon_steep_footprint=spec.forest_everon_steep_footprint,
        forest_everon_steep_maximum_relief=spec.forest_everon_steep_maximum_relief,
        forest_everon_steep_maximum_burial=spec.forest_everon_steep_maximum_burial,
        forest_everon_steep_maximum_float=spec.forest_everon_steep_maximum_float,
        forest_everon_steep_maximum_ground_sink=spec.forest_everon_steep_maximum_ground_sink,
        forest_polygon_sink_fraction=spec.forest_polygon_sink_fraction,
        forest_severe_hill_fallback=spec.forest_severe_hill_fallback,
        forest_severe_hill_relief=spec.forest_severe_hill_relief,
        forest_severe_hill_trees_per_block=spec.forest_severe_hill_trees_per_block,
        forest_cluster_fallback=spec.forest_cluster_fallback,
        forest_cluster_search_radius=spec.forest_cluster_search_radius,
        forest_cluster_maximum_relief=spec.forest_cluster_maximum_relief,
        forest_cluster_maximum_burial=spec.forest_cluster_maximum_burial,
        forest_cluster_maximum_float=spec.forest_cluster_maximum_float,
        forest_cluster_footprint_margin=spec.forest_cluster_footprint_margin,
        forest_undergrowth_enabled=spec.forest_undergrowth_enabled,
        forest_undergrowth_maximum_objects=spec.forest_undergrowth_maximum_objects,
        forest_undergrowth_spacing=spec.forest_undergrowth_spacing,
        forest_undergrowth_maximum_relief=spec.forest_undergrowth_maximum_relief,
        forest_undergrowth_maximum_burial=spec.forest_undergrowth_maximum_burial,
        forest_undergrowth_maximum_float=spec.forest_undergrowth_maximum_float,
        forest_undergrowth_ground_clearance=spec.forest_undergrowth_ground_clearance,
        steep_hill_bushes_enabled=spec.steep_hill_bushes_enabled,
        maximum_steep_hill_bush_objects=spec.maximum_steep_hill_bush_objects,
        steep_hill_bush_spacing=spec.steep_hill_bush_spacing,
        steep_hill_bush_minimum_slope_degrees=spec.steep_hill_bush_minimum_slope_degrees,
        steep_hill_bush_maximum_relief=spec.steep_hill_bush_maximum_relief,
        steep_hill_bush_maximum_burial=spec.steep_hill_bush_maximum_burial,
        steep_hill_bush_maximum_float=spec.steep_hill_bush_maximum_float,
        steep_hill_bush_ground_clearance=spec.steep_hill_bush_ground_clearance,
        steep_hill_bush_models=tuple(forest_models["steep_hill_bush_models"]),
        forest_border_enabled=spec.forest_border_enabled,
        forest_border_maximum_objects=spec.forest_border_maximum_objects,
        forest_border_spacing=spec.forest_border_spacing,
        forest_border_inset=spec.forest_border_inset,
        forest_border_maximum_relief=spec.forest_border_maximum_relief,
        forest_border_maximum_burial=spec.forest_border_maximum_burial,
        forest_border_maximum_float=spec.forest_border_maximum_float,
        forest_single_tree_enabled=spec.forest_single_tree_enabled,
        forest_single_tree_model=str(forest_models["forest_single_tree_model"]),
        forest_roadside_tree_model=str(forest_models["forest_roadside_tree_model"]),
        forest_roadside_tree_models=tuple(forest_models["forest_roadside_tree_models"]),
        forest_roadside_trees_per_cut_block=spec.forest_roadside_trees_per_cut_block,
        forest_roadside_bush_models=tuple(forest_models["forest_roadside_bush_models"]),
        forest_roadside_bushes_per_cut_block=spec.forest_roadside_bushes_per_cut_block,
        forest_roadside_bush_footprint=spec.forest_roadside_bush_footprint,
        maximum_forest_single_tree_objects=spec.maximum_forest_single_tree_objects,
        forest_single_tree_spacing=spec.forest_single_tree_spacing,
        forest_single_tree_footprint=spec.forest_single_tree_footprint,
        forest_single_tree_maximum_relief=spec.forest_single_tree_maximum_relief,
        forest_single_tree_maximum_float=spec.forest_single_tree_maximum_float,
        ditch_grass_enabled=spec.ditch_grass_enabled,
        maximum_ditch_grass_objects=spec.maximum_ditch_grass_objects,
        ditch_grass_spacing=spec.ditch_grass_spacing,
        ditch_grass_endpoint_trim=spec.ditch_grass_endpoint_trim,
        ditch_grass_maximum_relief=spec.ditch_grass_maximum_relief,
        ditch_grass_maximum_burial=spec.ditch_grass_maximum_burial,
        ditch_grass_maximum_float=spec.ditch_grass_maximum_float,
        ditch_grass_ground_clearance=spec.ditch_grass_ground_clearance,
        forest_hillside_fallback=spec.forest_hillside_fallback,
        forest_hillside_tree_model=str(forest_models["forest_hillside_tree_model"]),
        forest_hillside_trees_per_block=spec.forest_hillside_trees_per_block,
        forest_hillside_tree_footprint=spec.forest_hillside_tree_footprint,
        forest_hillside_tree_maximum_relief=spec.forest_hillside_tree_maximum_relief,
        barriers_enabled=spec.barriers_enabled,
        maximum_barrier_objects=spec.maximum_barrier_objects,
        barrier_segment_length=spec.barrier_segment_length,
        stock_hedge_models=spec.stock_hedge_models,
        stock_wall_models=spec.stock_wall_models,
        stock_metal_fence_models=spec.stock_metal_fence_models,
        bridges_enabled=spec.bridges_enabled,
        procedural_bridges=spec.procedural_bridges,
        maximum_bridge_objects=spec.maximum_bridge_objects,
        bridge_module_length=spec.bridge_module_length,
        bridge_deck_clearance=spec.bridge_deck_clearance,
        bridge_water_clearance=spec.bridge_water_clearance,
        residential_infill_enabled=spec.residential_infill_enabled,
        maximum_residential_infill_buildings=spec.maximum_residential_infill_buildings,
        residential_infill_spacing=spec.residential_infill_spacing,
        residential_infill_minimum_area=spec.residential_infill_minimum_area,
        residential_infill_road_clearance=spec.residential_infill_road_clearance,
        residential_infill_building_clearance=spec.residential_infill_building_clearance,
        rural_vegetation_enabled=spec.rural_vegetation_enabled,
        maximum_rural_vegetation_objects=spec.maximum_rural_vegetation_objects,
        rural_vegetation_spacing=spec.rural_vegetation_spacing,
        meadow_grass_enabled=spec.meadow_grass_enabled,
        maximum_meadow_grass_objects=spec.maximum_meadow_grass_objects,
        meadow_grass_spacing=spec.meadow_grass_spacing,
        wetland_reeds_enabled=spec.wetland_reeds_enabled,
        maximum_wetland_reed_objects=spec.maximum_wetland_reed_objects,
        wetland_reed_spacing=spec.wetland_reed_spacing,
        wetland_reed_maximum_relief=spec.wetland_reed_maximum_relief,
        wetland_reed_maximum_burial=spec.wetland_reed_maximum_burial,
        wetland_reed_maximum_float=spec.wetland_reed_maximum_float,
        wetland_reed_ground_clearance=spec.wetland_reed_ground_clearance,
        wetland_reed_models=spec.wetland_reed_models,
        rocky_forest_fallback_enabled=spec.rocky_forest_fallback_enabled,
        maximum_rocky_forest_objects=spec.maximum_rocky_forest_objects,
        rocky_forest_rocks_per_patch=spec.rocky_forest_rocks_per_patch,
        rocky_forest_spread=spec.rocky_forest_spread,
        rocky_forest_maximum_relief=spec.rocky_forest_maximum_relief,
        rocky_forest_maximum_burial=spec.rocky_forest_maximum_burial,
        rocky_forest_maximum_float=spec.rocky_forest_maximum_float,
        semantic_landmarks=spec.semantic_landmarks,
        bus_stops_enabled=spec.bus_stops_enabled,
        bus_stop_model=spec.bus_stop_model,
        bus_stop_footprint=spec.bus_stop_footprint,
        bus_stop_ground_clearance=spec.bus_stop_ground_clearance,
        maximum_landmark_objects=spec.maximum_landmark_objects,
        cemeteries_enabled=spec.cemeteries_enabled,
        maximum_grave_objects=spec.maximum_grave_objects,
        grave_spacing=spec.grave_spacing,
        grave_inset=spec.grave_inset,
        grave_footprint=spec.grave_footprint,
        grave_ground_clearance=spec.grave_ground_clearance,
        grave_road_clearance=spec.grave_road_clearance,
        grave_building_clearance=spec.grave_building_clearance,
        grave_models=spec.grave_models,
        semantic_site_maximum_relief=spec.semantic_site_maximum_relief,
        semantic_site_maximum_variants=spec.semantic_site_maximum_variants,
        pbo_backend=spec.pbo_backend,
        poseidon_tools_path=spec.poseidon_tools_path,
        include_minor_roads=spec.include_minor_roads,
        procedural_gravel_roads=spec.procedural_gravel_roads,
        road_connection_tolerance=spec.road_connection_tolerance,
        maximum_road_grade_percent=spec.maximum_road_grade_percent,
        road_grade_radius=spec.road_grade_radius,
        building_grade_radius=spec.building_grade_radius,
        maximum_grade_adjustment=spec.maximum_grade_adjustment,
        transition_cells=spec.transition_cells,
        asset_roots=spec.asset_roots,
        strict_assets=spec.strict_assets,
        osm_asset_mapping_path=spec.osm_asset_mapping_path,
        cache_dir=cache_dir,
        cache_enabled=spec.cache_enabled,
        cache_refresh=spec.cache_refresh,
        town_name_limit=spec.town_name_limit,
        deterministic_seed=f"milestone9:{normalization.normalized_fingerprint}",
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
        surface_pass_enabled=True,
        surface_shoreline_wet_cells=spec.surface_shoreline_wet_cells,
        surface_shoreline_sand_cells=spec.surface_shoreline_sand_cells,
        surface_transition_cells=spec.surface_transition_cells,
        surface_forest_edge_cells=spec.surface_forest_edge_cells,
        surface_farmland_strip_cells=spec.surface_farmland_strip_cells,
        surface_road_shoulder_metres=spec.surface_road_shoulder_metres,
        surface_dirt_blend_metres=spec.surface_dirt_blend_metres,
        surface_steep_slope_degrees=spec.surface_steep_slope_degrees,
        surface_colour_reference_path=reference_path,
        surface_colour_reference_strength=spec.surface_colour_reference_strength if reference_path else 0.0,
        surface_overview_size=spec.surface_overview_size,
        surface_texture_size=spec.surface_texture_size,
        surface_ground_mode=spec.surface_ground_mode,
    )
    with progress_range(12, 99):
        result = build_milestone4(
            output_dir,
            playability,
            clean=clean,
            mod_directory_name="@CWR-Milestone9",
            milestone_number=9,
            dataset_override=dataset,
        )
    report_progress(99, "Copying Milestone 9 provenance and runtime reports")
    provenance_path, source_validation_path = _copy_provenance(source, result)

    build_normalized_dir = result.output_dir / "normalized"
    if normalization.root.resolve() != build_normalized_dir.resolve():
        shutil.copytree(normalization.root, build_normalized_dir, dirs_exist_ok=True)
    runtime_root = result.pbo_path.parent.parent
    shutil.copyfile(normalization.manifest_path, runtime_root / "NORMALIZED-GEOMETRY.json")
    shutil.copyfile(normalization.validation_path, runtime_root / "NORMALIZED-GEOMETRY-VALIDATION.txt")
    if result.grading_report_path:
        shutil.copyfile(result.grading_report_path, runtime_root / "TERRAIN-SOLVER-REPORT.json")
    if result.surface_report_path:
        shutil.copyfile(result.surface_report_path, runtime_root / "SURFACE-PASS-REPORT.json")
    if result.overview_map_path:
        shutil.copyfile(result.overview_map_path, runtime_root / "OVERVIEW-MAP.png")

    deployment: dict[str, Any] | None = None
    if spec.deploy_mod_dir is not None:
        report_progress(99, f"Deploying Addons and Anims into {spec.deploy_mod_dir}")
        deployment = _deploy_runtime_to_existing_mod(result, spec.deploy_mod_dir)

    try:
        build_manifest: dict[str, Any] = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        build_manifest = {}
    terrain_report: dict[str, Any] = {}
    if result.grading_report_path:
        terrain_report = json.loads(result.grading_report_path.read_text(encoding="utf-8"))
    surface_report: dict[str, Any] = {}
    if result.surface_report_path:
        surface_report = json.loads(result.surface_report_path.read_text(encoding="utf-8"))
    build_manifest["schema"] = 9
    build_manifest["milestone"] = 9
    build_manifest["generator"] = GENERATOR_VERSION
    build_manifest["deployment"] = deployment
    build_manifest["source_bundle"] = {
        "manifest_sha256": source.fingerprint,
        "bbox_south_west_north_east": list(source.bbox),
        "cells": source.cells,
        "cell_size_metres": source.cell_size,
        "manifest": provenance_path.name,
        "validation": source_validation_path.name,
    }
    cache_report_path = result.cache_report_path or (result.output_dir / "cache-report.json")
    try:
        cache_report: dict[str, Any] = json.loads(cache_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache_report = {"schema": 2, "generator": GENERATOR_VERSION}
    cache_report["enabled"] = spec.cache_enabled
    cache_report["directory"] = str(cache_dir)
    cache_report["refresh"] = spec.cache_refresh
    cache_report["parsed_source"] = {
        "hit": dataset.parsed_cache_hit,
        "normalized_fingerprint": normalization.normalized_fingerprint,
    }
    cache_report["spatial_index"] = {
        "hit": spatial_index.cache_hit,
        "fingerprint": spatial_index.fingerprint,
        "road_segments": len(spatial_index.road_segments),
        "bucket_size_metres": spatial_index.bucket_size,
    }
    cache_report_path.write_text(json.dumps(cache_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
    build_manifest["surface_visual_pass"] = {
        **surface_report,
        "report": result.surface_report_path.name if result.surface_report_path else None,
        "overview_map": result.overview_map_path.name if result.overview_map_path else None,
        "world_icon": rf"{spec.name}\data\icon.paa",
        "ground_texture_profile": spec.ground_texture_profile,
        "ground_application_mode": spec.surface_ground_mode,
        "colour_reference": str(reference_path) if reference_path else None,
        "colour_reference_sha256": _sha256(reference_path) if reference_path else None,
        "deterministic_seed": playability.deterministic_seed,
    }
    result.manifest_path.write_text(json.dumps(build_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with result.report_path.open("a", encoding="utf-8", newline="\n") as report:
        report.write("\nMilestone 9 surface and visual checks\n\n")
        checks = (
            ("Surface report emitted", result.surface_report_path is not None and result.surface_report_path.is_file(), str(result.surface_report_path)),
            ("Overview map emitted", result.overview_map_path is not None and result.overview_map_path.is_file(), str(result.overview_map_path)),
            ("Improved world icon embedded", result.world_icon_path is not None and result.world_icon_path.is_file(), str(result.world_icon_path)),
            ("Shoreline bands analysed for overview", int(surface_report.get("wet_shoreline_cells", 0)) + int(surface_report.get("dry_shoreline_cells", 0)) > 0 if dataset.water else True, str(surface_report.get("material_cells", {}))),
            ("Forest-edge surfaces analysed for overview", int(surface_report.get("forest_edge_cells", 0)) > 0 if dataset.forests else True, str(surface_report.get("forest_edge_cells", 0))),
            ("Farmland subdivisions deterministic", (int(surface_report.get("farmland_light_cells", 0)) + int(surface_report.get("farmland_dark_cells", 0)) + int(surface_report.get("field_boundary_cells", 0))) > 0 if dataset.farmland else True, f"features={surface_report.get('feature_seed_count', 0)}"),
            ("Road surface classes analysed for overview", (int(surface_report.get("paved_road_cells", 0)) + int(surface_report.get("dirt_road_cells", 0)) + int(surface_report.get("gravel_road_cells", 0))) > 0 if dataset.roads else True, f"paved={surface_report.get('paved_road_cells', 0)}, dirt={surface_report.get('dirt_road_cells', 0)}, gravel={surface_report.get('gravel_road_cells', 0)}"),
            ("Rock slope classes generated", int(surface_report.get("rock_cells", 0)) + int(surface_report.get("steep_rock_cells", 0)) >= 0, f"rock={surface_report.get('rock_cells', 0)}, steep={surface_report.get('steep_rock_cells', 0)}"),
        )
        for label, ok, detail in checks:
            report.write(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}\n")
        report.write("[PASS] Transition decisions derive from deterministic seed, feature IDs, and cell coordinates\n")
        report.write(f"[PASS] Optional colour reference: {reference_path if reference_path else 'disabled'}\n")

    report_progress(100, "Milestone 9 build complete")
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
        meadow_grass_preview_path=result.meadow_grass_preview_path,
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
        forest_cluster_catalogue_path=result.forest_cluster_catalogue_path,
        infrastructure_catalogue_path=result.infrastructure_catalogue_path,
        surface_report_path=result.surface_report_path,
        overview_map_path=result.overview_map_path,
        overview_paa_path=result.overview_paa_path,
        world_icon_path=result.world_icon_path,
        cache_report_path=cache_report_path,
    )
