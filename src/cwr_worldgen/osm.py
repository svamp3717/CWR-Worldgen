# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import heapq
import json
import pickle
import math
import time
from typing import Any, TYPE_CHECKING, Callable, Iterable, Mapping, Sequence
from urllib import error, parse, request

from PIL import Image, ImageDraw

from ._version import __version__
from .cache import CACHE_SCHEMA_VERSION, atomic_write_bytes, cache_key, streaming_hash
from .building_semantics import is_actual_church
from .model import OsmSpec, WorldObject
from .network import (
    OVERPASS_RATE_LIMIT_BACKOFF_SECONDS,
    OVERPASS_REFERER,
    OVERPASS_USER_AGENT,
    UNVERIFIED_SSL_CONTEXT,
)
from .procedural_infrastructure import (
    GENERATED_BRIDGE_MAXIMUM_DEPTH_METRES,
    GENERATED_BRIDGE_RAIL_OVERHANG_METRES,
    GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES,
    ProceduralInfrastructureLibrary,
    gravel_road_model_path,
)
from .procedural_forests import (
    DITCH_GRASS_VARIANTS,
    FOREST_BORDER_VARIANTS,
    FOREST_CLUSTER_GRADES,
    FOREST_CLUSTER_VARIANTS,
    FOREST_UNDERGROWTH_VARIANTS,
    RURAL_VEGETATION_VARIANTS,
    ForestClusterVariant,
    cluster_model_path,
    cluster_variant,
    quantize_cluster_grade,
)

if TYPE_CHECKING:
    from .procedural_buildings import BuildingPlacement, ProceduralBuildingLibrary

EARTH_RADIUS_METRES = 6_371_008.8
PointLL = tuple[float, float]  # latitude, longitude
PointXZ = tuple[float, float]  # world east/x, north/z in metres

STOCK_HEDGE_MODELS: tuple[str, ...] = (
    r"data3d\Krovi_long.p3d",
    r"data3d\Krovi_bigest.p3d",
    r"data3d\Krovi2.p3d",
    r"data3d\Krovi3.p3d",
    r"data3d\Krovi4.p3d",
    r"data3d\Krovia.p3d",
    r"data3d\Krovib.p3d",
)

STOCK_WALL_MODELS: tuple[str, ...] = (
    r"O\Hous\zidka01.p3d",
    r"O\Hous\zidka02.p3d",
)

STOCK_METAL_FENCE_MODELS: tuple[str, ...] = (
    r"O\Hous\DD_pletivo.p3d",
)

# Stock OFP/CWA fence objects used for occasional farmland boundaries. These are
# original game models, not generated infrastructure. ``ohrada_sama`` is the
# rural/pasture-looking fence; ``plot_provizorni`` is the stock wire fence.
STOCK_FARMLAND_FENCE_MODELS: tuple[str, ...] = (
    r"data3d\ohrada_sama.p3d",
    r"data3d\plot_provizorni.p3d",
)
FARMLAND_FENCE_FIELD_PERCENT = 25.0
FARMLAND_FENCE_SEGMENT_LENGTH_METRES = 5.0
FARMLAND_FENCE_MINIMUM_FIELD_AREA_M2 = 1600.0
FARMLAND_FENCE_HEADING_OFFSET_DEGREES = 90.0
FARMLAND_FENCE_DUPLICATE_DISTANCE_METRES = 2.4
FARMLAND_FENCE_DUPLICATE_MINIMUM_OVERLAP_METRES = 1.0
FARMLAND_FENCE_DUPLICATE_HEADING_TOLERANCE_DEGREES = 12.0
RURAL_FENCE_LANDUSES = frozenset({"farmland", "meadow"})
RURAL_FENCE_NATURALS = frozenset({"grassland"})

# Urban-detail assets. Keep this layer vanilla-only: every model below ships
# with OFP/Cold War Assault. The base game does not provide a modern modular
# narrow sidewalk kit, so ``Nam_dlazba`` (the stock cobbled-square pavement
# object) is tiled along suitable road edges instead of generating new P3Ds.
STOCK_SIDEWALK_MODELS: tuple[str, ...] = (r"O\Hous\Nam_dlazba.p3d",)
STOCK_STREET_LIGHT_MODELS: tuple[str, ...] = (
    r"data3d\lampazel.p3d",
    r"o\misc\vo_seda.p3d",
    r"data3d\lampadrevo.p3d",
    r"o\misc\vo_stara1.p3d",
)
STOCK_STREET_BENCH_MODELS: tuple[str, ...] = (r"o\misc\lavicka_1.p3d",)
STOCK_STREET_BIN_MODELS: tuple[str, ...] = (
    r"o\misc\popelnice.p3d",
    r"o\misc\kos_hist.p3d",
)
STOCK_STREET_NOTICEBOARD_MODELS: tuple[str, ...] = (
    r"o\misc\nastenka1.p3d",
    r"o\misc\nastenka2.p3d",
    r"o\misc\nastenka3.p3d",
    r"o\misc\cedule_info.p3d",
)
STOCK_STREET_BICYCLE_MODELS: tuple[str, ...] = (r"kolo\koloslozeny.p3d",)
STOCK_STREET_BUS_SHELTER_MODELS: tuple[str, ...] = (r"O\Misc\aut_zast.p3d",)
STOCK_STREET_TREE_SURROUND_MODELS: tuple[str, ...] = (r"O\Hous\Nam_okruzi.p3d",)
STOCK_STREET_TREE_MODELS: tuple[str, ...] = (
    r"O\Tree\Javor01.p3d",
    r"O\Tree\Javor02.p3d",
)
STOCK_SETTLEMENT_UTILITY_POLE_MODELS: tuple[str, ...] = (
    r"data3d\sloupyelA.p3d",
    r"data3d\sloupyell.p3d",
    r"O\Hous\stozarvn_1.p3d",
)
STOCK_SETTLEMENT_FRUIT_TREE_MODELS: tuple[str, ...] = (
    r"data3d\jablon.p3d",
    r"data3d\hrusen.p3d",
    r"data3d\str_jablon.p3d",
)
# Barn-only stock clutter. These are deliberately never scattered in fields,
# house yards, sheds, stables, or generic agricultural buildings.
STOCK_SETTLEMENT_BARN_CLUTTER_MODELS: tuple[str, ...] = (
    r"O\Misc\seno_balik.p3d",
    r"data3d\drevo_hromada.p3d",
    r"O\Misc\sekyraspalek.p3d",
    r"data3d\paletyC.p3d",
)
STOCK_SETTLEMENT_DRIVEWAY_MODEL = r"O\Road\ces6.p3d"
STOCK_SETTLEMENT_DETAIL_MODELS: tuple[str, ...] = tuple(dict.fromkeys((
    *STOCK_SETTLEMENT_UTILITY_POLE_MODELS,
    *STOCK_STREET_NOTICEBOARD_MODELS,
    *STOCK_STREET_BENCH_MODELS,
    *STOCK_STREET_BIN_MODELS,
    *STOCK_STREET_BICYCLE_MODELS,
    *STOCK_HEDGE_MODELS,
    *STOCK_FARMLAND_FENCE_MODELS,
    *STOCK_SETTLEMENT_FRUIT_TREE_MODELS,
    *STOCK_SETTLEMENT_BARN_CLUTTER_MODELS,
    STOCK_SETTLEMENT_DRIVEWAY_MODEL,
)))

SETTLEMENT_DETAIL_RADIUS_METRES: Mapping[str, float] = {
    "city": 1600.0,
    "town": 800.0,
    "village": 420.0,
    "hamlet": 240.0,
}
SETTLEMENT_UTILITY_POLE_SPACING_METRES: Mapping[str, float] = {
    # Sparse cues only, roughly one third as many poles as the first pass.
    "city": 280.0,
    "town": 240.0,
    "residential": 240.0,
    "village": 190.0,
    "hamlet": 210.0,
}
SETTLEMENT_DETAIL_MINIMUM_SEPARATION_METRES = 3.5
SIDEWALK_DEFAULT_WIDTH_METRES = 1.8
SIDEWALK_DEFAULT_SEGMENT_LENGTH_METRES = 5.0
SIDEWALK_CURB_GAP_METRES = 0.12
SIDEWALKS_TEMPORARILY_DISABLED = True
STREET_FURNITURE_MINIMUM_SEPARATION_METRES = 7.0
STREET_FURNITURE_GROUND_CLEARANCE_METRES = 0.04
# Keep stock settlement/town props fully outside the drivable carriageway, with
# a small shoulder margin.  Individual prop footprints are added on top of this
# value by the placement checks below.
STREET_FURNITURE_ROAD_CLEARANCE_METRES = 0.35
TOWN_STREET_FURNITURE_RADIUS_METRES = 800.0
CITY_STREET_FURNITURE_RADIUS_METRES = 1600.0
PEDESTRIAN_ONLY_HIGHWAYS = frozenset({"path", "footway", "cycleway", "bridleway", "pedestrian", "steps"})
DEFAULT_NEARBY_BUILDING_TEXTURE_MATCH_DISTANCE_METRES = 90.0
MINIMUM_NEARBY_BUILDING_TEXTURE_MATCH_CLUSTER = 3

# User-configurable object-count "maximums" are advisory thresholds.  A positive
# value warns when generation exceeds it but never truncates the category; zero
# retains the existing explicit meaning of disabling that category.
_ADVISORY_OBJECT_LIMIT_SENTINEL = 1 << 60


def _advisory_object_limit(value: object, *, enabled: bool = False) -> int:
    configured = max(0, int(value))
    if configured == 0:
        return 0
    return _ADVISORY_OBJECT_LIMIT_SENTINEL if enabled else configured


def _object_threshold_warning(label: str, generated: int, configured: object) -> str | None:
    threshold = max(0, int(configured))
    if threshold <= 0 or generated <= threshold:
        return None
    return (
        f"WARNING: {label} warning threshold exceeded: generated {generated:,}, "
        f"configured threshold {threshold:,}; continuing with the complete category."
    )

HEDGE_MODEL_HEADING_OFFSET_DEGREES = 90.0
WALL_MODEL_HEADING_OFFSET_DEGREES = 90.0
METAL_FENCE_MODEL_HEADING_OFFSET_DEGREES = 90.0
ROADSIDE_NUDGE_DISTANCE_METRES = 1.25
ROADSIDE_BARRIER_CLEARANCE_METRES = 0.35
ROADSIDE_BARRIER_MAXIMUM_NUDGE_METRES = 15.0
HEDGE_FOOTPRINT_HALF_WIDTH_METRES = 1.25
HEDGE_FOOTPRINT_END_MARGIN_METRES = 0.35
HEDGE_GROUND_CLEARANCE_METRES = 0.10
HEDGE_MODEL_VERTICAL_ORIGIN_OFFSETS: Mapping[str, float] = {
    r"data3d\Krovi_long.p3d".casefold(): 1.00,
    r"data3d\Krovi_bigest.p3d".casefold(): 0.90,
    r"data3d\Krovi2.p3d".casefold(): 0.75,
    r"data3d\Krovi3.p3d".casefold(): 0.75,
    r"data3d\Krovi4.p3d".casefold(): 0.75,
    r"data3d\Krovia.p3d".casefold(): 0.70,
    r"data3d\Krovib.p3d".casefold(): 0.70,
}
HEDGE_DEFAULT_VERTICAL_ORIGIN_OFFSET_METRES = 0.85
LARGE_BUILDING_ROAD_NUDGE_DISTANCE_METRES = 2.0
LARGE_BUILDING_ROAD_NUDGE_MINIMUM_SPAN_METRES = 18.0
GARAGE_CLUSTER_RADIUS_METRES = 45.0
# Only CWR-inferred outbuilding subtypes are subject to this cap. Explicit OSM
# building=garage/garages remains authoritative regardless of local density.
GARAGE_CLUSTER_MAXIMUM_GARAGES = 3
ENTERABLE_BUILDING_MINIMUM_GROUND_CLEARANCE_METRES = 0.25
ENTERABLE_BUILDING_ENTRANCE_APRON_DEPTH_METRES = 1.25
ENTERABLE_BUILDING_ENTRANCE_APRON_INSET_METRES = 0.25
ENTERABLE_BUILDING_ENTRANCE_APRON_SIDE_MARGIN_METRES = 0.45
STOCK_WALL_EFFECTIVE_LENGTH_METRES = 2.45
ROADSIDE_LARGE_TREE_MODEL = r"data3d\str smrk vysoky.p3d"
SMALL_SETTLEMENT_INFILL_RADIUS_METRES: Mapping[str, float] = {
    "village": 150.0,
    "hamlet": 95.0,
    "farm": 70.0,
    "isolated_dwelling": 36.0,
}
# OSM place nodes are labels for settlements/properties, not guaranteed building
# centroids.  Before inventing a small-settlement residential patch, look much
# farther than the patch itself for any source-backed building.  This prevents
# an isolated_dwelling label beside its real farmstead (for example Amerika)
# from receiving a synthetic bonus house merely because the label is 40-80 m
# away from the mapped roof.
SMALL_SETTLEMENT_EXISTING_BUILDING_GUARD_METRES: Mapping[str, float] = {
    # Keep village/hamlet guards close to their existing synthetic footprint so
    # legitimately missing buildings at the far side of a settlement can still
    # use Overture fallback. Farm and isolated-dwelling labels are much more
    # likely to sit at a property/driveway label rather than on a roof, so they
    # get the larger offset tolerance.
    "village": 150.0,
    "hamlet": 120.0,
    "farm": 110.0,
    "isolated_dwelling": 100.0,
}
ROADSIDE_TREE_MODELS: tuple[str, ...] = (
    r"data3d\str smrk vysoky.p3d",
    r"data3d\str smrk.p3d",
    r"data3d\str smrk ridky.p3d",
    r"data3d\str jedle.p3d",
    r"data3d\str borovice.p3d",
)
OSM_BROADLEAF_TREE_MODELS: tuple[str, ...] = (
    r"data3d\str briza.p3d",
    r"data3d\str dub.p3d",
    r"data3d\str javor.p3d",
    r"data3d\str lipa.p3d",
    r"data3d\str vrba.p3d",
)
OSM_CONIFER_TREE_MODELS: tuple[str, ...] = (
    r"data3d\str smrk.p3d",
    r"data3d\str borovice.p3d",
    r"data3d\str jedle.p3d",
)
OSM_INDIVIDUAL_TREE_MODELS: tuple[str, ...] = OSM_BROADLEAF_TREE_MODELS + OSM_CONIFER_TREE_MODELS
# Resistance/Nogova-only individual-tree families. These are selected when the
# active polygon forest model is from O.pbo, preventing mapped OSM trees from
# quietly reintroducing Data3D vegetation into the leaf/pine Nogova presets.
NOGOVA_LEAF_INDIVIDUAL_TREE_MODELS: tuple[str, ...] = (
    r"o\tree\Javor01.p3d",
    r"o\tree\Javor02.p3d",
    r"o\tree\Akat01.p3d",
    r"o\tree\Akat02.p3d",
    r"o\tree\Akat03.p3d",
    r"o\tree\DubFX.p3d",
)
NOGOVA_PINE_INDIVIDUAL_TREE_MODELS: tuple[str, ...] = (
    r"o\tree\Smrk_maly.p3d",
    r"o\tree\Smrk_siroky.p3d",
    r"o\tree\Smrk_velky.p3d",
    r"o\tree\DD_borovice.p3d",
    r"o\tree\DD_borovice02.p3d",
)
# The 256x256, 25 m world is the visual-density baseline requested for
# synthetic individual forest trees. Larger worlds retain the same density per
# square kilometre, so their safety limits scale with physical area.
INDIVIDUAL_TREE_REFERENCE_WORLD_SIZE_METRES = 6_400.0
# Resistance's Nogova bridge is one fixed 30 m road module. Its truss and
# supports extend below the roadway origin, so clearance is measured from a
# conservative four-metre lower bound rather than from the deck surface.
NOGOVA_BRIDGE_MODEL = r"o\hous\most_stred30.p3d"
NOGOVA_BRIDGE_MODULE_LENGTH_METRES = 30.0
NOGOVA_BRIDGE_HALF_WIDTH_METRES = 4.5
NOGOVA_BRIDGE_LOWEST_POINT_METRES = -4.0
# The stock Nogova bridge road deck is aligned to the ordinary stock-road
# surface instead of being lifted to keep its supports clear of terrain. Its
# long fixed module is deliberately allowed to bury its ends/supports into the
# banks so bridge approaches meet the fitted road network cleanly.
NOGOVA_BRIDGE_APPROACH_OFFSET_METRES = 0.035
NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES = 0.05
CHURCH_EXTRA_GROUND_CLEARANCE_METRES = 0.0
# RVW4 stores ``cells`` terrain vertices, so the final directly sampled vertex
# sits at (cells - 1) * cell_size. Large rigid building footprints that run into
# the outer unsampled strip can render as if terrain cuts through their lower
# walls. Keep the complete model footprint just inside the directly sampled grid.
BUILDING_TERRAIN_EDGE_MARGIN_METRES = 0.50
# Synthetic residential infill must not appear as a second, invented house
# immediately beside source-backed OSM/Overture buildings. Keep this larger
# than synthetic-to-synthetic spacing because real source geometry wins.
RESIDENTIAL_INFILL_SOURCE_BUILDING_CLEARANCE_METRES = 25.0
# Unmatched Overture footprints can be slightly displaced from their OSM
# counterpart. A conservative size-aware near-duplicate gate catches those
# without broadly deleting legitimate neighbouring buildings.
OVERTURE_FALLBACK_NEAR_DUPLICATE_MAXIMUM_DISTANCE_METRES = 10.0
PROCEDURAL_BRIDGE_LAND_INSET_METRES = 12.0

# Original OFP/Resistance terrain-object models. These are deliberately used
# directly instead of generating world-local rock P3Ds. The BIS model catalogue
# lists the Kamen/Kopa models as unclassed island assets, which is exactly how
# WRP terrain objects can reference them.
STOCK_STONE_MODELS = (
    r"data3d\kamen1_zula.p3d",
    r"data3d\kamen2_zula.p3d",
    r"data3d\kamen3_zula.p3d",
    r"data3d\kamen4_zula.p3d",
    r"data3d\kamen5_zula.p3d",
    r"data3d\kopa_kameni.p3d",
    r"data3d\kopa_kameni2.p3d",
)
MINIMUM_BRIDGE_TERRAIN_CLEARANCE_METRES = 0.50
MINIMUM_BRIDGE_WATER_CLEARANCE_METRES = 18.0
ROADSIDE_BUSH_MODELS: tuple[str, ...] = (
    r"data3d\ker listnac.p3d",
    r"data3d\ker pichlavej.p3d",
    r"data3d\ker deravej.p3d",
    r"data3d\ker buxus.p3d",
)


def _scaled_synthetic_tree_limit(limit: int, world_size: float) -> int:
    """Scale a 6.4 km single-tree safety limit by physical world area."""

    limit = max(0, int(limit))
    if limit == 0:
        return 0
    linear_scale = max(0.0, float(world_size)) / INDIVIDUAL_TREE_REFERENCE_WORLD_SIZE_METRES
    return max(1, int(round(limit * linear_scale * linear_scale)))


def stock_hedge_model(
    length_m: float,
    identity: str,
    models: Sequence[str] = STOCK_HEDGE_MODELS,
) -> str:
    """Choose an original stock hedge model for one mapped hedge segment."""
    if len(models) < 3:
        raise ValueError("stock hedge model list must contain at least three models")
    if length_m >= 4.5:
        return models[0]
    if length_m >= 3.0:
        return models[1]
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=2).digest()
    return models[2 + int.from_bytes(digest, "little") % (len(models) - 2)]


def stock_wall_model(
    identity: str,
    models: Sequence[str] = STOCK_WALL_MODELS,
) -> str:
    """Choose one original Resistance stone-wall variant deterministically."""
    if not models:
        raise ValueError("stock wall model list must contain at least one model")
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=2).digest()
    return models[int.from_bytes(digest, "little") % len(models)]


def osm_fence_is_metal(tags: Mapping[str, str]) -> bool:
    """Return whether an OSM fence should use the stock chain-link model."""
    if tags.get("barrier", "").casefold() != "fence":
        return False
    fence_type = tags.get("fence_type", "").casefold().replace("-", "_").replace(" ", "_")
    material = tags.get("material", "").casefold().replace("-", "_").replace(" ", "_")
    return (
        fence_type in {
            "chain_link", "chainlink", "metal", "metal_bars", "wire",
            "wire_mesh", "welded_wire_mesh", "mesh",
        }
        or "chain" in fence_type
        or material in {"metal", "steel", "iron", "wire", "wire_mesh"}
    )


def stock_metal_fence_model(
    identity: str,
    models: Sequence[str] = STOCK_METAL_FENCE_MODELS,
) -> str:
    """Choose a stock chain-link fence model deterministically."""
    if not models:
        raise ValueError("stock metal fence model list must contain at least one model")
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=2).digest()
    return models[int.from_bytes(digest, "little") % len(models)]


def stock_farmland_fence_model(
    identity: str,
    models: Sequence[str] = STOCK_FARMLAND_FENCE_MODELS,
) -> str:
    """Choose one original OFP/CWA fence family deterministically.

    This chooser is intended to run once per farmland field, not once per
    segment. A selected field therefore uses either the rural/pasture fence or
    the stock wire fence around its whole perimeter, never a patchwork.
    """
    if not models:
        raise ValueError("stock farmland fence model list must contain at least one model")
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=2).digest()
    roll = int.from_bytes(digest, "little") / 65535.0
    if len(models) == 1 or roll < 0.78:
        return models[0]
    return models[1 + (int.from_bytes(digest, "little") % (len(models) - 1))]


def _rural_fence_segments_duplicate(
    candidate: tuple[float, float, float, float],
    existing: tuple[float, float, float, float],
) -> bool:
    """Return whether two nearby field-fence chunks are redundant parallels."""

    x0, z0, x1, z1 = candidate
    ex0, ez0, ex1, ez1 = existing
    dx, dz = x1 - x0, z1 - z0
    edx, edz = ex1 - ex0, ez1 - ez0
    length = math.hypot(dx, dz)
    existing_length = math.hypot(edx, edz)
    if length <= 1.0e-6 or existing_length <= 1.0e-6:
        return False
    ux, uz = dx / length, dz / length
    eux, euz = edx / existing_length, edz / existing_length
    parallel = abs(ux * eux + uz * euz)
    if parallel < math.cos(math.radians(FARMLAND_FENCE_DUPLICATE_HEADING_TOLERANCE_DEGREES)):
        return False

    nx, nz = -uz, ux
    t0 = (ex0 - x0) * ux + (ez0 - z0) * uz
    t1 = (ex1 - x0) * ux + (ez1 - z0) * uz
    c0 = (ex0 - x0) * nx + (ez0 - z0) * nz
    c1 = (ex1 - x0) * nx + (ez1 - z0) * nz
    if max(abs(c0), abs(c1)) > FARMLAND_FENCE_DUPLICATE_DISTANCE_METRES:
        return False
    overlap = min(length, max(t0, t1)) - max(0.0, min(t0, t1))
    return overlap >= FARMLAND_FENCE_DUPLICATE_MINIMUM_OVERLAP_METRES


@dataclass(frozen=True, slots=True)
class GeoPolygon:
    outer: tuple[PointLL, ...]
    holes: tuple[tuple[PointLL, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class OsmLineFeature:
    osm_key: str
    tags: Mapping[str, str]
    points: tuple[PointLL, ...]


@dataclass(frozen=True, slots=True)
class OsmPolygonFeature:
    osm_key: str
    tags: Mapping[str, str]
    polygons: tuple[GeoPolygon, ...]


@dataclass(frozen=True, slots=True)
class OsmPointFeature:
    osm_key: str
    tags: Mapping[str, str]
    point: PointLL


@dataclass(frozen=True, slots=True)
class OsmDataset:
    source_generator: str
    element_count: int
    coastlines: tuple[OsmLineFeature, ...]
    water: tuple[OsmPolygonFeature, ...]
    forests: tuple[OsmPolygonFeature, ...]
    farmland: tuple[OsmPolygonFeature, ...]
    urban: tuple[OsmPolygonFeature, ...]
    roads: tuple[OsmLineFeature, ...]
    gravel_roads: tuple[OsmLineFeature, ...] = ()
    watercourses: tuple[OsmLineFeature, ...] = ()
    building_polygons: tuple[OsmPolygonFeature, ...] = ()
    building_points: tuple[OsmPointFeature, ...] = ()
    places: tuple[OsmPointFeature, ...] = ()
    place_areas: tuple[OsmPolygonFeature, ...] = ()
    landmarks: tuple[OsmPointFeature, ...] = ()
    sites: tuple[OsmPolygonFeature, ...] = ()
    barriers: tuple[OsmLineFeature, ...] = ()
    cutlines: tuple[OsmLineFeature, ...] = ()
    tree_rows: tuple[OsmLineFeature, ...] = ()
    individual_trees: tuple[OsmPointFeature, ...] = ()
    aeroway_lines: tuple[OsmLineFeature, ...] = ()
    aeroway_areas: tuple[OsmPolygonFeature, ...] = ()
    utility_points: tuple[OsmPointFeature, ...] = ()
    surface_areas: tuple[OsmPolygonFeature, ...] = ()
    rural_vegetation: tuple[OsmPolygonFeature, ...] = ()
    building_entrances: tuple[OsmPointFeature, ...] = ()
    normalized_fingerprint: str = ""
    parsed_cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class BboxProjection:
    south: float
    west: float
    north: float
    east: float
    world_size: float
    source_width_metres: float
    source_height_metres: float
    scale_x: float
    scale_z: float

    @classmethod
    def create(cls, bbox: tuple[float, float, float, float], world_size: float) -> "BboxProjection":
        south, west, north, east = bbox
        middle_latitude = math.radians((south + north) / 2.0)
        width = EARTH_RADIUS_METRES * math.radians(east - west) * math.cos(middle_latitude)
        height = EARTH_RADIUS_METRES * math.radians(north - south)
        if width <= 0 or height <= 0:
            raise ValueError("OSM bounding box has zero projected area")
        return cls(
            south=south,
            west=west,
            north=north,
            east=east,
            world_size=world_size,
            source_width_metres=width,
            source_height_metres=height,
            scale_x=world_size / width,
            scale_z=world_size / height,
        )

    def to_world(self, point: PointLL) -> PointXZ:
        latitude, longitude = point
        # ``create`` uses the same constant-latitude equirectangular projection
        # for width and scale, so the Earth-radius/trig terms cancel exactly.
        # Keep the hot path as two affine transforms.
        return (
            (longitude - self.west) * self.world_size / (self.east - self.west),
            (latitude - self.south) * self.world_size / (self.north - self.south),
        )

    def to_latlon(self, point: PointXZ) -> PointLL:
        x, z = point
        return (
            self.south + z * (self.north - self.south) / self.world_size,
            self.west + x * (self.east - self.west) / self.world_size,
        )

    def to_pixel(self, point: PointLL, resolution: int) -> tuple[float, float]:
        x, z = self.to_world(point)
        denominator = max(1, resolution - 1)
        return (
            x / self.world_size * denominator,
            (1.0 - z / self.world_size) * denominator,
        )


@dataclass(frozen=True, slots=True)
class OsmRaster:
    cells: int
    water: tuple[bool, ...]
    forest: tuple[bool, ...]
    farmland: tuple[bool, ...]
    urban: tuple[bool, ...]
    roads: tuple[bool, ...]
    buildings: tuple[bool, ...]
    high_resolution: int
    coastline_seed_count: int
    # Terrain vertices near a coarse shoreline can be marked as water even when
    # most of the supersampled cell is dry land. Keep a second conservative mask
    # so only strongly water-covered vertices are eligible for full-depth carving.
    water_interior: tuple[bool, ...] = ()


@dataclass(frozen=True, slots=True)
class CompactOrientedRectangle(Sequence[PointXZ]):
    """Four-corner support footprint stored as five scalars instead of 13 objects.

    Million-building worlds are dominated by Python container overhead. Most
    procedural supports are rectangles, so calculate their four points lazily
    when a polygon consumer actually asks for them.
    """

    x: float
    z: float
    width: float
    length: float
    heading_degrees: float

    def __len__(self) -> int:
        return 4

    def _point(self, index: int) -> PointXZ:
        if index < 0:
            index += 4
        if not 0 <= index < 4:
            raise IndexError(index)
        half_width = max(0.05, self.width * 0.5)
        half_length = max(0.05, self.length * 0.5)
        angle = math.radians(self.heading_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        width_sign, length_sign = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))[index]
        return (
            self.x + width_sign * half_width * cosine + length_sign * half_length * sine,
            self.z - width_sign * half_width * sine + length_sign * half_length * cosine,
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        return self._point(int(index))

    def __iter__(self):
        # Iteration is the common polygon-consumer path. Compute trig once for
        # all four corners rather than paying for four separate ``_point`` calls.
        half_width = max(0.05, self.width * 0.5)
        half_length = max(0.05, self.length * 0.5)
        angle = math.radians(self.heading_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for width_sign, length_sign in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            yield (
                self.x + width_sign * half_width * cosine + length_sign * half_length * sine,
                self.z - width_sign * half_width * sine + length_sign * half_length * cosine,
            )


def _compact_support_polygon(
    points: Sequence[PointXZ], x: float, z: float, heading_degrees: float
) -> Sequence[PointXZ]:
    """Compress a support polygon when it is the expected oriented rectangle."""

    if len(points) != 4:
        return tuple(points)
    angle = math.radians(heading_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    local: list[PointXZ] = []
    for px, pz in points:
        dx, dz = float(px) - x, float(pz) - z
        local.append((dx * cosine - dz * sine, dx * sine + dz * cosine))
    half_width = max(abs(px) for px, _pz in local)
    half_length = max(abs(pz) for _px, pz in local)
    if half_width <= 1.0e-9 or half_length <= 1.0e-9:
        return tuple(points)
    tolerance = max(1.0e-5, max(half_width, half_length) * 1.0e-7)
    corners: set[tuple[int, int]] = set()
    for local_x, local_z in local:
        if abs(abs(local_x) - half_width) > tolerance or abs(abs(local_z) - half_length) > tolerance:
            return tuple(points)
        corners.add((1 if local_x >= 0.0 else -1, 1 if local_z >= 0.0 else -1))
    if len(corners) != 4:
        return tuple(points)
    return CompactOrientedRectangle(
        float(x), float(z), half_width * 2.0, half_length * 2.0, float(heading_degrees)
    )


@dataclass(frozen=True, slots=True)
class BuildingPlacementPlan:
    osm_key: str
    geometry_index: int
    geometry_kind: str
    x: float
    z: float
    heading_degrees: float
    model_path: str
    support_polygon: Sequence[PointXZ]
    procedural_placement: "BuildingPlacement | None" = None
    road_nudged: bool = False
    building_family: str = ""
    synthetic_infill: bool = False


@dataclass(frozen=True, slots=True)
class ObjectGenerationResult:
    objects: tuple[WorldObject, ...]
    road_objects: int
    building_objects: int
    forest_objects: int
    road_objects_truncated: bool
    building_objects_truncated: bool
    forest_objects_truncated: bool
    forest_road_rejections: int = 0
    maximum_building_grounding_raise: float = 0.0
    maximum_building_pad_relief: float = 0.0
    maximum_building_foundation_depth: float = 0.0
    building_foundation_rejections: int = 0
    building_interior_fallbacks: int = 0
    building_fully_submerged_rejections: int = 0
    building_road_nudges: int = 0
    maximum_forest_grounding_raise: float = 0.0
    forest_slope_rejections: int = 0
    maximum_forest_relief: float = 0.0
    forest_block_objects: int = 0
    forest_hillside_tree_objects: int = 0
    forest_hillside_fallback_blocks: int = 0
    forest_hillside_unfilled_blocks: int = 0
    forest_hillside_candidate_rejections: int = 0
    maximum_hillside_tree_relief: float = 0.0
    forest_everon_steep_objects: int = 0
    forest_sunk_polygon_objects: int = 0
    forest_everon_steep_rejections: int = 0
    forest_cluster_objects: int = 0
    forest_cluster_rejections: int = 0
    forest_cluster_maximum_burial: float = 0.0
    forest_cluster_maximum_float: float = 0.0
    forest_cluster_variant_counts: tuple[tuple[str, int], ...] = ()
    forest_undergrowth_objects: int = 0
    forest_undergrowth_rejections: int = 0
    forest_undergrowth_maximum_burial: float = 0.0
    forest_undergrowth_maximum_float: float = 0.0
    steep_hill_bush_objects: int = 0
    steep_hill_bush_rejections: int = 0
    wetland_reed_objects: int = 0
    wetland_reed_rejections: int = 0
    forest_border_objects: int = 0
    forest_border_rejections: int = 0
    forest_border_maximum_burial: float = 0.0
    forest_border_maximum_float: float = 0.0
    forest_single_tree_objects: int = 0
    forest_gap_infill_tree_objects: int = 0
    ditch_grass_objects: int = 0
    ditch_grass_rejections: int = 0
    ditch_grass_maximum_burial: float = 0.0
    ditch_grass_maximum_float: float = 0.0
    maximum_forest_burial: float = 0.0
    maximum_forest_float: float = 0.0
    barrier_objects: int = 0
    fence_objects: int = 0
    wall_objects: int = 0
    hedge_objects: int = 0
    barrier_rejections: int = 0
    bridge_objects: int = 0
    bridge_segments: int = 0
    bridge_rejections: int = 0
    residential_infill_objects: int = 0
    residential_infill_areas: int = 0
    tree_row_objects: int = 0
    orchard_objects: int = 0
    vineyard_objects: int = 0
    scrub_objects: int = 0
    rural_rock_objects: int = 0
    rural_vegetation_rejections: int = 0
    meadow_grass_objects: int = 0
    meadow_grass_rejections: int = 0
    haybale_objects: int = 0
    haybale_rejections: int = 0
    haybale_fields_total: int = 0
    haybale_fields_selected: int = 0
    meadow_grass_positions: tuple[PointXZ, ...] = ()
    meadow_grass_rejection_positions: tuple[PointXZ, ...] = ()
    rocky_forest_objects: int = 0
    rocky_forest_rejections: int = 0
    mapped_tree_objects: int = 0
    mapped_tree_rejections: int = 0
    utility_objects: int = 0
    utility_rejections: int = 0
    sidewalk_objects: int = 0
    street_furniture_objects: int = 0
    street_light_objects: int = 0
    street_bench_objects: int = 0
    street_bin_objects: int = 0
    street_noticeboard_objects: int = 0
    street_bicycle_objects: int = 0
    street_bus_shelter_objects: int = 0
    street_tree_objects: int = 0
    urban_detail_rejections: int = 0
    vegetation_audit_tree_objects: int = 0
    vegetation_audit_cluster_tree_proxies: int = 0
    vegetation_audit_cluster_bush_proxies: int = 0
    vegetation_audit_violations: int = 0
    vegetation_audit_maximum_tree_float: float = 0.0
    vegetation_audit_maximum_bush_float: float = 0.0
    # Aggregates accumulated at emission time. They let later build stages use
    # unique-model counts and surface-relevant coordinates without rescanning
    # and reclassifying every WorldObject.
    model_usage: tuple[tuple[str, int], ...] = ()
    surface_forest_positions: tuple[PointXZ, ...] = ()
    surface_rock_positions: tuple[PointXZ, ...] = ()


@dataclass(frozen=True, slots=True)
class IterativeGroundingReport:
    building_supports: int = 0
    tree_supports: int = 0
    adjusted_cells: int = 0
    maximum_adjustment: float = 0.0
    mean_adjustment: float = 0.0


HAYBALE_MODEL = r"o\misc\seno_balik.p3d"
HAYBALE_ANCHOR_ACCEPTANCE = 0.45
HAYBALE_FIELD_PERCENT = 25.0
HAYBALE_CLUSTER_SINGLE_FRACTION = 0.70
HAYBALE_CLUSTER_PAIR_FRACTION = 0.20
HAYBALE_CLUSTER_RADIUS_MIN = 7.0
HAYBALE_CLUSTER_RADIUS_MAX = 18.0

_FOREST_LANDUSES = {"forest"}
_FARMLAND_LANDUSES = {
    "farmland",
    "meadow",
    "orchard",
    "vineyard",
    "grass",
    "allotments",
    "plant_nursery",
    "greenhouse_horticulture",
    "recreation_ground",
    "village_green",
}
_URBAN_LANDUSES = {
    "residential",
    "commercial",
    "industrial",
    "retail",
    "construction",
    "farmyard",
    "garages",
    "railway",
    "education",
    "institutional",
    "civic",
}
_WATER_LANDUSES = {"reservoir", "basin"}
_MAJOR_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
    "track",
}
_MINOR_HIGHWAYS = {"path", "footway", "cycleway", "bridleway", "pedestrian"}
_GRAVEL_SURFACES = {"gravel", "fine_gravel", "compacted", "pebblestone", "unpaved"}
_UNPAVED_SURFACES = {"unpaved", *_GRAVEL_SURFACES, "dirt", "earth", "ground", "sand", "mud"}

# Legacy stock-road aliases for gravel are kept only for historical mapping
# compatibility. Milestone 9 now defaults to generated world-local gravel road
# ribbons instead of the stock Resistance cobblestone family.
STOCK_GRAVEL_ROAD_MODELS: tuple[str, ...] = (
    r"O\Road\kos25.p3d",
    r"O\Road\kos12.p3d",
    r"O\Road\kos6.p3d",
)
STOCK_GRAVEL_ROAD_MODEL = STOCK_GRAVEL_ROAD_MODELS[0]
_PAVED_SURFACES = {"paved", "asphalt", "concrete", "concrete:lanes", "concrete:plates", "paving_stones", "sett", "cobblestone"}
_DEFAULT_DIRT_HIGHWAYS = {"track", "service", "unclassified"}


def build_overpass_query(bbox: tuple[float, float, float, float], *, timeout_seconds: int = 90) -> str:
    south, west, north, east = bbox
    bbox_text = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    return f'''[out:json][timeout:{timeout_seconds}][bbox:{bbox_text}];
(
  way["natural"="coastline"];
  nwr["natural"="water"];
  nwr["waterway"="riverbank"];
  way["waterway"~"^(river|stream|canal|drain|ditch)$"];
  nwr["landuse"~"^(reservoir|basin)$"];
  nwr["natural"="wood"];
  nwr["natural"="wetland"];
  nwr["natural"="grassland"];
  nwr["landcover"="trees"];
  nwr["landuse"~"^(forest|farmland|meadow|orchard|vineyard|grass|allotments|plant_nursery|greenhouse_horticulture|recreation_ground|village_green|residential|commercial|industrial|retail|construction|farmyard|garages|railway|education|institutional|civic)$"];
  way["highway"];
  nwr["building"];
  node["entrance"];
  nwr["amenity"~"^(place_of_worship|school|social_facility|parking)$"];
  nwr["social_facility"];
  nwr["amenity"="grave_yard"];
  nwr["landuse"="cemetery"];
  nwr["shop"];
  nwr["leisure"="pitch"];
  nwr["sport"="soccer"];
  way["barrier"~"^(fence|wall|hedge|retaining_wall)$"];
  way["man_made"="cutline"];
  way["power"~"^(line|minor_line)$"];
  way["natural"="tree_row"];
  node["natural"="tree"];
  nwr["aeroway"~"^(aerodrome|runway|taxiway|apron|helipad)$"];
  node["power"~"^(pole|tower)$"];
  nwr["man_made"="water_tower"];
  nwr["natural"~"^(scrub|bare_rock|rock|scree)$"];
  nwr["natural"~"^(grassland|sand|beach|desert|dune)$"];
  nwr["landcover"="sand"];
  nwr["surface"="sand"];
  nwr["leisure"="park"];
  node["highway"="bus_stop"];
  node["public_transport"="platform"]["bus"="yes"];
  nwr["place"~"^(city|town|village|suburb|quarter|hamlet|locality)$"]["name"];
  nwr["place"="isolated_dwelling"];
);
out body geom;
'''


def fetch_overpass_json(url: str, query_text: str, *, timeout_seconds: int) -> bytes:
    payload = parse.urlencode({"data": query_text}).encode("ascii")
    req = request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded; charset=ascii",
            "User-Agent": OVERPASS_USER_AGENT,
            "Referer": OVERPASS_REFERER,
        },
        method="POST",
    )
    # The direct OSM importer has one configured endpoint rather than the
    # multi-mirror source-fetch path. On HTTP 429, respect Overpass guidance by
    # waiting 30 seconds before making exactly one retry.
    for attempt in range(2):
        try:
            with request.urlopen(req, timeout=timeout_seconds + 15, context=UNVERIFIED_SSL_CONTEXT) as response:
                data = response.read()
            break
        except error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                time.sleep(OVERPASS_RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise RuntimeError(f"Overpass request failed: HTTP {exc.code}") from exc
        except OSError as exc:
            raise RuntimeError(f"Overpass request failed: {exc}") from exc
    else:  # pragma: no cover - loop always returns, breaks, or raises.
        raise RuntimeError("Overpass request failed")
    if not data:
        raise RuntimeError("Overpass returned an empty response")
    return data


def load_osm_json(spec: OsmSpec) -> tuple[bytes, str]:
    query_text = build_overpass_query(spec.bbox, timeout_seconds=spec.overpass_timeout_seconds)
    if spec.osm_json_path is not None:
        return spec.osm_json_path.read_bytes(), query_text
    return (
        fetch_overpass_json(
            spec.overpass_url,
            query_text,
            timeout_seconds=spec.overpass_timeout_seconds,
        ),
        query_text,
    )


def _point(value: Mapping[str, object]) -> PointLL:
    try:
        latitude = float(value["lat"])
        longitude = float(value["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OSM geometry point is missing numeric lat/lon") from exc
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("OSM geometry contains non-finite coordinates")
    return latitude, longitude


def _geometry(element: Mapping[str, object]) -> tuple[PointLL, ...]:
    raw = element.get("geometry")
    if isinstance(raw, list):
        return tuple(_point(item) for item in raw if isinstance(item, Mapping))
    if "lat" in element and "lon" in element:
        return (_point(element),)
    return ()


def _same_point(a: PointLL, b: PointLL) -> bool:
    return abs(a[0] - b[0]) <= 1e-8 and abs(a[1] - b[1]) <= 1e-8


def _closed(points: Sequence[PointLL]) -> tuple[PointLL, ...] | None:
    if len(points) < 3:
        return None
    result = tuple(points)
    if not _same_point(result[0], result[-1]):
        result = result + (result[0],)
    if len(result) < 4:
        return None
    return result


def _stitch_lines(lines: Iterable[Sequence[PointLL]]) -> tuple[tuple[PointLL, ...], ...]:
    pending = [list(line) for line in lines if len(line) >= 2]
    rings: list[tuple[PointLL, ...]] = []
    while pending:
        chain = pending.pop(0)
        changed = True
        while changed and not _same_point(chain[0], chain[-1]):
            changed = False
            for index, candidate in enumerate(pending):
                if _same_point(chain[-1], candidate[0]):
                    chain.extend(candidate[1:])
                elif _same_point(chain[-1], candidate[-1]):
                    chain.extend(reversed(candidate[:-1]))
                elif _same_point(chain[0], candidate[-1]):
                    chain = candidate[:-1] + chain
                elif _same_point(chain[0], candidate[0]):
                    chain = list(reversed(candidate[1:])) + chain
                else:
                    continue
                pending.pop(index)
                changed = True
                break
        ring = _closed(chain)
        if ring is not None:
            rings.append(ring)
    return tuple(rings)


def _relation_polygons(element: Mapping[str, object]) -> tuple[GeoPolygon, ...]:
    members = element.get("members")
    if not isinstance(members, list):
        return ()
    outer_lines: list[tuple[PointLL, ...]] = []
    inner_lines: list[tuple[PointLL, ...]] = []
    for member in members:
        if not isinstance(member, Mapping):
            continue
        points = _geometry(member)
        if len(points) < 2:
            continue
        role = str(member.get("role", ""))
        if role == "inner":
            inner_lines.append(points)
        else:
            outer_lines.append(points)
    outers = _stitch_lines(outer_lines)
    inners = _stitch_lines(inner_lines)
    # Hole-to-outer assignment is unnecessary for raster subtraction at this
    # milestone. Keep all relation holes attached to the first outer so each is
    # drawn exactly once and data remains deterministic.
    return tuple(
        GeoPolygon(outer=outer, holes=inners if index == 0 else ())
        for index, outer in enumerate(outers)
    )


def _element_polygons(element: Mapping[str, object]) -> tuple[GeoPolygon, ...]:
    if element.get("type") == "relation":
        return _relation_polygons(element)
    ring = _closed(_geometry(element))
    return (GeoPolygon(ring),) if ring is not None else ()


def _tag_map(element: Mapping[str, object]) -> dict[str, str]:
    raw = element.get("tags")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _osm_key(element: Mapping[str, object]) -> str:
    return f"{element.get('type', 'unknown')}/{element.get('id', 'unknown')}"


def parse_overpass_json(
    data: bytes,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> OsmDataset:
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    progress(0, f"Decoding OpenStreetMap JSON ({len(data) / (1024 * 1024):.1f} MiB)")
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Overpass JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Overpass JSON root must be an object")
    elements = document.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Overpass JSON does not contain an elements array")

    coastlines: list[OsmLineFeature] = []
    water: list[OsmPolygonFeature] = []
    forests: list[OsmPolygonFeature] = []
    farmland: list[OsmPolygonFeature] = []
    urban: list[OsmPolygonFeature] = []
    roads: list[OsmLineFeature] = []
    watercourses: list[OsmLineFeature] = []
    building_polygons: list[OsmPolygonFeature] = []
    building_points: list[OsmPointFeature] = []
    building_entrances: list[OsmPointFeature] = []
    places: list[OsmPointFeature] = []
    place_areas: list[OsmPolygonFeature] = []
    landmarks: list[OsmPointFeature] = []
    sites: list[OsmPolygonFeature] = []
    barriers: list[OsmLineFeature] = []
    cutlines: list[OsmLineFeature] = []
    tree_rows: list[OsmLineFeature] = []
    individual_trees: list[OsmPointFeature] = []
    aeroway_lines: list[OsmLineFeature] = []
    aeroway_areas: list[OsmPolygonFeature] = []
    utility_points: list[OsmPointFeature] = []
    surface_areas: list[OsmPolygonFeature] = []
    rural_vegetation: list[OsmPolygonFeature] = []

    total = len(elements)
    progress(5, f"Classifying {total:,} OpenStreetMap elements")
    update_interval = max(1, total // 24)
    for element_index, element in enumerate(elements, start=1):
        if element_index == total or element_index % update_interval == 0:
            local = 5 + round(84 * element_index / max(1, total))
            progress(local, f"Classifying OpenStreetMap elements {element_index:,}/{total:,}")
        if not isinstance(element, Mapping):
            continue
        tags = _tag_map(element)
        key = _osm_key(element)
        geometry = _geometry(element)
        polygons = _element_polygons(element)
        natural = tags.get("natural", "")
        landuse = tags.get("landuse", "")
        landcover = tags.get("landcover", "")

        if natural == "coastline" and len(geometry) >= 2:
            coastlines.append(OsmLineFeature(key, tags, geometry))

        is_water = (
            natural == "water"
            or tags.get("waterway") == "riverbank"
            or landuse in _WATER_LANDUSES
            or landcover == "water"
        )
        if is_water and polygons:
            water.append(OsmPolygonFeature(key, tags, polygons))

        is_forest = natural == "wood" or landuse in _FOREST_LANDUSES or landcover == "trees"
        if is_forest and polygons:
            forests.append(OsmPolygonFeature(key, tags, polygons))
        elif (landuse in _FARMLAND_LANDUSES or natural == "grassland") and polygons:
            farmland.append(OsmPolygonFeature(key, tags, polygons))
        elif landuse in _URBAN_LANDUSES and polygons:
            urban.append(OsmPolygonFeature(key, tags, polygons))

        if "highway" in tags and len(geometry) >= 2:
            roads.append(OsmLineFeature(key, tags, geometry))

        if tags.get("waterway") in {"river", "stream", "canal", "drain", "ditch"} and len(geometry) >= 2:
            watercourses.append(OsmLineFeature(key, tags, geometry))

        if "building" in tags:
            if polygons:
                building_polygons.append(OsmPolygonFeature(key, tags, polygons))
            elif len(geometry) == 1:
                building_points.append(OsmPointFeature(key, tags, geometry[0]))

        entrance_kind = tags.get("entrance", "").strip().casefold()
        if entrance_kind and entrance_kind not in {"no", "none"} and len(geometry) == 1:
            building_entrances.append(OsmPointFeature(key, tags, geometry[0]))

        place_kind = str(tags.get("place", "")).casefold()
        keep_place = bool(place_kind) and (bool(tags.get("name")) or place_kind == "isolated_dwelling")
        if keep_place:
            if len(geometry) == 1:
                places.append(OsmPointFeature(key, tags, geometry[0]))
            elif polygons:
                # Preserve mapped place areas exactly. Unnamed isolated-dwelling
                # polygons are meaningful even without a label: membership in
                # the polygon is what decides whether the lone small building is
                # a cabin rather than a generic shed.
                place_areas.append(OsmPolygonFeature(key, tags, polygons))
                ring = polygons[0].outer[:-1]
                if ring:
                    places.append(OsmPointFeature(
                        key, tags,
                        (sum(point[0] for point in ring) / len(ring), sum(point[1] for point in ring) / len(ring)),
                    ))

        is_bus_stop = tags.get("highway") == "bus_stop" or (
            tags.get("public_transport") == "platform" and tags.get("bus") in {"yes", "designated"}
        )
        if is_bus_stop:
            if len(geometry) == 1:
                landmarks.append(OsmPointFeature(key, tags | {"landmark": "bus_stop"}, geometry[0]))
            elif polygons:
                ring = polygons[0].outer[:-1]
                if ring:
                    landmarks.append(OsmPointFeature(
                        key, tags | {"landmark": "bus_stop"},
                        (sum(point[0] for point in ring) / len(ring), sum(point[1] for point in ring) / len(ring)),
                    ))

        site_kind = "parking" if tags.get("amenity") == "parking" else (
            "sports_pitch" if (tags.get("leisure") == "pitch" or tags.get("sport") == "soccer") else (
                "cemetery" if (tags.get("landuse") == "cemetery" or tags.get("amenity") == "grave_yard") else ""
            )
        )
        if site_kind and polygons:
            sites.append(OsmPolygonFeature(key, tags | {"site": site_kind}, polygons))

        barrier = tags.get("barrier", "")
        if barrier in {"fence", "wall", "hedge", "retaining_wall"} and len(geometry) >= 2:
            barriers.append(OsmLineFeature(key, tags, geometry))

        if (tags.get("man_made") == "cutline" or tags.get("power") in {"line", "minor_line"}) and len(geometry) >= 2:
            cutlines.append(OsmLineFeature(key, tags, geometry))

        if natural == "tree_row" and len(geometry) >= 2:
            tree_rows.append(OsmLineFeature(key, tags, geometry))

        if natural == "tree":
            if len(geometry) == 1:
                individual_trees.append(OsmPointFeature(key, tags, geometry[0]))
            elif polygons:
                ring = polygons[0].outer[:-1]
                if ring:
                    individual_trees.append(OsmPointFeature(
                        key, tags,
                        (sum(point[0] for point in ring) / len(ring), sum(point[1] for point in ring) / len(ring)),
                    ))

        aeroway = tags.get("aeroway", "")
        if aeroway in {"runway", "taxiway"} and len(geometry) >= 2 and not polygons:
            aeroway_lines.append(OsmLineFeature(key, tags, geometry))
        if aeroway in {"aerodrome", "runway", "taxiway", "apron", "helipad"} and polygons:
            aeroway_areas.append(OsmPolygonFeature(key, tags, polygons))

        utility_kind = ""
        if tags.get("power") in {"pole", "tower"}:
            utility_kind = f"power_{tags.get('power')}"
        elif tags.get("man_made") == "water_tower":
            utility_kind = "water_tower"
        if utility_kind:
            if len(geometry) == 1:
                utility_points.append(OsmPointFeature(key, tags | {"utility": utility_kind}, geometry[0]))
            elif polygons:
                ring = polygons[0].outer[:-1]
                if ring:
                    utility_points.append(OsmPointFeature(
                        key, tags | {"utility": utility_kind},
                        (sum(point[0] for point in ring) / len(ring), sum(point[1] for point in ring) / len(ring)),
                    ))

        surface_kind = (
            natural if natural == "grassland"
            else "beach" if natural == "beach"
            else "sand" if natural in {"sand", "desert", "dune"} or landcover == "sand" or tags.get("surface") == "sand"
            else "park" if tags.get("leisure") == "park"
            else "sports_pitch" if (tags.get("leisure") == "pitch" or tags.get("sport") == "soccer")
            else ""
        )
        if surface_kind and polygons:
            surface_areas.append(OsmPolygonFeature(key, tags | {"surface_kind": surface_kind}, polygons))

        rural_kind = natural if natural in {"scrub", "bare_rock", "rock", "scree", "wetland"} else (
            landuse if landuse in {"orchard", "vineyard"} else ""
        )
        if rural_kind and polygons:
            rural_vegetation.append(OsmPolygonFeature(key, tags | {"rural_kind": rural_kind}, polygons))

    progress(92, "Sorting and freezing OpenStreetMap feature groups")
    sort_key = lambda feature: feature.osm_key
    result = OsmDataset(
        source_generator=str(document.get("generator", "unknown")),
        element_count=len(elements),
        coastlines=tuple(sorted(coastlines, key=sort_key)),
        water=tuple(sorted(water, key=sort_key)),
        forests=tuple(sorted(forests, key=sort_key)),
        farmland=tuple(sorted(farmland, key=sort_key)),
        urban=tuple(sorted(urban, key=sort_key)),
        roads=tuple(sorted(roads, key=sort_key)),
        gravel_roads=tuple(sorted((road for road in roads if road_is_gravel(road.tags)), key=sort_key)),
        watercourses=tuple(sorted(watercourses, key=sort_key)),
        building_polygons=tuple(sorted(building_polygons, key=sort_key)),
        building_points=tuple(sorted(building_points, key=sort_key)),
        building_entrances=tuple(sorted(building_entrances, key=sort_key)),
        places=tuple(sorted(places, key=sort_key)),
        place_areas=tuple(sorted(place_areas, key=sort_key)),
        landmarks=tuple(sorted(landmarks, key=sort_key)),
        sites=tuple(sorted(sites, key=sort_key)),
        barriers=tuple(sorted(barriers, key=sort_key)),
        cutlines=tuple(sorted(cutlines, key=sort_key)),
        tree_rows=tuple(sorted(tree_rows, key=sort_key)),
        individual_trees=tuple(sorted(individual_trees, key=sort_key)),
        aeroway_lines=tuple(sorted(aeroway_lines, key=sort_key)),
        aeroway_areas=tuple(sorted(aeroway_areas, key=sort_key)),
        utility_points=tuple(sorted(utility_points, key=sort_key)),
        surface_areas=tuple(sorted(surface_areas, key=sort_key)),
        rural_vegetation=tuple(sorted(rural_vegetation, key=sort_key)),
    )
    progress(100, (
        f"OpenStreetMap features ready: {len(result.roads):,} roads, "
        f"{len(result.building_polygons) + len(result.building_points):,} buildings, "
        f"{len(result.forests):,} forests"
    ))
    return result


def road_is_supported(tags: Mapping[str, str], *, include_minor: bool) -> bool:
    highway = tags.get("highway", "")
    if tags.get("tunnel") not in {None, "", "no"}:
        return False
    if highway in _MAJOR_HIGHWAYS:
        return True
    return include_minor and highway in _MINOR_HIGHWAYS


def road_width_metres(tags: Mapping[str, str]) -> float:
    highway = tags.get("highway", "")
    if highway in {"motorway", "trunk"}:
        return 12.0
    if highway in {"primary", "secondary"}:
        return 9.0
    if highway in {"tertiary", "unclassified"}:
        return 7.0
    if highway in {"residential", "living_street"}:
        return 6.0
    if highway in {"service", "road"}:
        return 4.5
    if highway == "track":
        return 3.5
    return 2.5


def _urban_detail_road_eligible(tags: Mapping[str, str]) -> bool:
    highway = tags.get("highway", "").casefold()
    if tags.get("tunnel") not in {None, "", "no"}:
        return False
    if tags.get("bridge") not in {None, "", "no"}:
        return False
    if highway in {"motorway", "motorway_link", "trunk", "trunk_link", "track"}:
        return False
    return highway in {
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "unclassified", "residential",
        "living_street", "service", "road",
    }


def _sidewalk_sides(tags: Mapping[str, str], *, inferred: bool) -> tuple[int, ...]:
    """Return -1 for left and +1 for right relative to OSM way direction."""

    value = tags.get("sidewalk", "").strip().casefold().replace("-", "_")
    left = tags.get("sidewalk:left", "").strip().casefold()
    right = tags.get("sidewalk:right", "").strip().casefold()
    negative = {"no", "none", "separate"}
    positive = {"yes", "designated"}
    if value in negative and left not in positive and right not in positive:
        return ()
    sides: list[int] = []
    if value in {"left"} or left in positive:
        sides.append(-1)
    if value in {"right"} or right in positive:
        sides.append(1)
    if value in {"both", "yes"}:
        return (-1, 1)
    if sides:
        return tuple(sides)
    return (-1, 1) if inferred else ()


def _stock_street_light_model(tags: Mapping[str, str], identity: str) -> str:
    highway = tags.get("highway", "").casefold()
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=2).digest()
    roll = int.from_bytes(digest, "little") % 100
    if highway in {"primary", "secondary", "tertiary"}:
        if roll < 58:
            return STOCK_STREET_LIGHT_MODELS[1]
        if roll < 88:
            return STOCK_STREET_LIGHT_MODELS[0]
        return STOCK_STREET_LIGHT_MODELS[3]
    if roll < 68:
        return STOCK_STREET_LIGHT_MODELS[0]
    if roll < 91:
        return STOCK_STREET_LIGHT_MODELS[2]
    return STOCK_STREET_LIGHT_MODELS[3]


def road_is_gravel(tags: Mapping[str, str]) -> bool:
    """Return whether OSM surface tagging selects the generated gravel family.

    Generic ``surface=unpaved`` is intentionally treated as gravel. More
    specific earth, dirt, ground, sand, and mud values remain dirt roads.
    """

    return tags.get("surface", "").strip().casefold() in _GRAVEL_SURFACES


def road_model_for_tags(spec: OsmSpec, tags: Mapping[str, str]) -> str:
    """Select the paved, dirt, or dedicated gravel model family for one road.

    Pedestrian-only OSM ways must never become the stock asphalt vehicle road.
    Even an explicitly paved footway is represented with the narrow dirt-path
    fallback until a dedicated pedestrian-path asset family exists.
    """

    if tags.get("highway", "").strip().casefold() in PEDESTRIAN_ONLY_HIGHWAYS:
        return spec.dirt_road_model
    if road_is_gravel(tags):
        if bool(getattr(spec, "procedural_gravel_roads", False)):
            return gravel_road_model_path(spec.name, 25)
        return spec.dirt_road_model
    return spec.dirt_road_model if road_is_dirt(tags) else spec.paved_road_model


def road_is_dirt(tags: Mapping[str, str]) -> bool:
    """Return whether a road should use the stock CWA dirt-road family.

    Explicit OSM surface tagging wins.  In the absence of a surface tag,
    ``track``, ``service`` and ``unclassified`` roads default to the ``ces``
    dirt-road models.  This keeps small access and local roads visually
    distinct while still respecting explicitly paved OSM data.
    """

    highway = tags.get("highway", "").strip().casefold()
    surface = tags.get("surface", "").strip().casefold()
    if surface in _PAVED_SURFACES:
        return False
    if surface in _UNPAVED_SURFACES:
        return True
    return highway in _DEFAULT_DIRT_HIGHWAYS


def _segment_intersects_rectangle(
    start: PointXZ,
    end: PointXZ,
    minimum_x: float,
    minimum_z: float,
    maximum_x: float,
    maximum_z: float,
) -> bool:
    """Return whether a line segment intersects an axis-aligned rectangle."""
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    lower = 0.0
    upper = 1.0
    for origin, delta, minimum, maximum in (
        (start[0], dx, minimum_x, maximum_x),
        (start[1], dz, minimum_z, maximum_z),
    ):
        if abs(delta) <= 1e-12:
            if origin < minimum or origin > maximum:
                return False
            continue
        enter = (minimum - origin) / delta
        leave = (maximum - origin) / delta
        if enter > leave:
            enter, leave = leave, enter
        lower = max(lower, enter)
        upper = min(upper, leave)
        if lower > upper:
            return False
    return True



RoadCorridor = tuple[PointXZ, PointXZ, float]


@dataclass(frozen=True, slots=True)
class ProjectedRoadSegment:
    start: PointXZ
    end: PointXZ
    tags: tuple[tuple[str, str], ...]
    osm_key: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            min(self.start[0], self.end[0]),
            min(self.start[1], self.end[1]),
            max(self.start[0], self.end[0]),
            max(self.start[1], self.end[1]),
        )


@dataclass(frozen=True, slots=True)
class SpatialLookupIndex:
    fingerprint: str
    bucket_size: float
    road_segments: tuple[ProjectedRoadSegment, ...]
    road_buckets: Mapping[tuple[int, int], tuple[int, ...]]
    road_polylines: tuple[tuple[PointXZ, ...], ...]
    cache_hit: bool = False

    def candidate_segment_indices(
        self, minimum_x: float, minimum_z: float, maximum_x: float, maximum_z: float
    ) -> tuple[int, ...]:
        bucket = self.bucket_size
        x0 = math.floor(minimum_x / bucket)
        z0 = math.floor(minimum_z / bucket)
        x1 = math.floor(maximum_x / bucket)
        z1 = math.floor(maximum_z / bucket)
        found: set[int] = set()
        for bz in range(z0, z1 + 1):
            for bx in range(x0, x1 + 1):
                found.update(self.road_buckets.get((bx, bz), ()))
        return tuple(sorted(found))


@dataclass(frozen=True, slots=True)
class IndexedRoadCorridors(Sequence[RoadCorridor]):
    corridors: tuple[RoadCorridor, ...]
    bucket_size: float
    buckets: Mapping[tuple[int, int], tuple[int, ...]]

    def __len__(self) -> int:
        return len(self.corridors)

    def __getitem__(self, index):
        return self.corridors[index]

    def intersects_rectangle(
        self,
        minimum_x: float,
        minimum_z: float,
        maximum_x: float,
        maximum_z: float,
    ) -> bool:
        bucket = self.bucket_size
        x0 = math.floor(minimum_x / bucket)
        z0 = math.floor(minimum_z / bucket)
        x1 = math.floor(maximum_x / bucket)
        z1 = math.floor(maximum_z / bucket)

        def intersects(index: int) -> bool:
            start, end, radius = self.corridors[index]
            return _segment_intersects_rectangle(
                start,
                end,
                minimum_x - radius,
                minimum_z - radius,
                maximum_x + radius,
                maximum_z + radius,
            )

        # Forest placement performs this query hundreds of thousands of times.
        # Most 50 m footprints fit inside one 100 m spatial bucket, so avoid
        # allocating and sorting a temporary set for the overwhelmingly common
        # case. For multi-bucket queries, test each newly-seen segment
        # immediately so road hits can return early. Query order cannot affect
        # the boolean result.
        if x0 == x1 and z0 == z1:
            return any(intersects(index) for index in self.buckets.get((x0, z0), ()))

        seen: set[int] = set()
        for bz in range(z0, z1 + 1):
            for bx in range(x0, x1 + 1):
                for index in self.buckets.get((bx, bz), ()):
                    if index in seen:
                        continue
                    seen.add(index)
                    if intersects(index):
                        return True
        return False


_SPATIAL_CACHE_SCHEMA = 2
_SPATIAL_INDEX_REGISTRY: dict[str, SpatialLookupIndex] = {}
# Repeated nearest-road queries must not rebuild the content fingerprint for the
# same in-memory dataset. The content-key registry remains the cross-object/cache
# authority; this identity registry is only a fast path within one build.
_SPATIAL_FAST_REGISTRY: dict[
    tuple[int, float, float, float, float, float],
    tuple[OsmDataset, SpatialLookupIndex],
] = {}
# Road corridors are derived entirely from the canonical spatial index plus two
# spec knobs. Several placement passes request the same structure, so retain a
# small bounded in-memory cache instead of rebuilding its tuples and buckets.
_ROAD_CORRIDOR_REGISTRY: dict[tuple[str, bool, float], IndexedRoadCorridors] = {}


def _spatial_fast_key(
    dataset: OsmDataset, projection: BboxProjection
) -> tuple[int, float, float, float, float, float]:
    return (
        id(dataset),
        float(projection.south), float(projection.west),
        float(projection.north), float(projection.east),
        float(projection.world_size),
    )


def _remember_spatial_index(
    dataset: OsmDataset, projection: BboxProjection, index: SpatialLookupIndex
) -> SpatialLookupIndex:
    key = _spatial_fast_key(dataset, projection)
    _SPATIAL_FAST_REGISTRY[key] = (dataset, index)
    # Keep identity caching bounded so a long-lived GUI session does not retain
    # every large OSM dataset it has ever generated. Eight active/recent worlds
    # is ample for normal use and still prevents object-id reuse while cached.
    while len(_SPATIAL_FAST_REGISTRY) > 8:
        oldest = next(iter(_SPATIAL_FAST_REGISTRY))
        if oldest == key and len(_SPATIAL_FAST_REGISTRY) > 1:
            oldest = next(item for item in _SPATIAL_FAST_REGISTRY if item != key)
        _SPATIAL_FAST_REGISTRY.pop(oldest, None)
    return index


def _spatial_registry_key(dataset: OsmDataset, projection: BboxProjection) -> str:
    if dataset.normalized_fingerprint:
        dataset_identity = dataset.normalized_fingerprint
    else:
        dataset_identity = streaming_hash(
            "osm-dataset-road-content-v2",
            (
                (
                    feature.osm_key,
                    tuple(sorted((str(key), str(value)) for key, value in feature.tags.items())),
                    feature.points,
                )
                for feature in dataset.roads
            ),
        )
    return cache_key(
        "spatial-index-registry-v1",
        {
            "dataset": dataset_identity,
            "bbox": [projection.south, projection.west, projection.north, projection.east],
            "world_size": projection.world_size,
        },
    )


def prepare_spatial_index(
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh: bool = False,
    bucket_size: float = 100.0,
    progress_callback: Callable[[int, str], None] | None = None,
) -> SpatialLookupIndex:
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    progress(0, f"Checking spatial road index for {len(dataset.roads):,} roads")
    registry_key = _spatial_registry_key(dataset, projection)
    if use_cache and not refresh and registry_key in _SPATIAL_INDEX_REGISTRY:
        previous = _SPATIAL_INDEX_REGISTRY[registry_key]
        progress(100, "Using in-memory spatial road index")
        loaded = SpatialLookupIndex(
            previous.fingerprint,
            previous.bucket_size,
            previous.road_segments,
            previous.road_buckets,
            previous.road_polylines,
            True,
        )
        return _remember_spatial_index(dataset, projection, loaded)
    disk_key = cache_key(
        "projected-spatial-index-v1",
        {
            "schema": _SPATIAL_CACHE_SCHEMA,
            "registry": registry_key,
            "bucket_size": bucket_size,
        },
    )
    cache_path = cache_dir / "spatial" / f"{disk_key}.pickle" if cache_dir is not None else None
    if use_cache and not refresh and cache_path is not None and cache_path.is_file():
        try:
            with cache_path.open("rb") as stream:
                payload = pickle.load(stream)
            index = payload.get("index") if isinstance(payload, dict) else None
            if (
                isinstance(payload, dict)
                and payload.get("schema") == CACHE_SCHEMA_VERSION
                and payload.get("spatial_schema") == _SPATIAL_CACHE_SCHEMA
                and payload.get("registry_key") == registry_key
                and isinstance(index, SpatialLookupIndex)
            ):
                loaded = SpatialLookupIndex(
                    index.fingerprint,
                    index.bucket_size,
                    index.road_segments,
                    index.road_buckets,
                    index.road_polylines,
                    True,
                )
                _SPATIAL_INDEX_REGISTRY[registry_key] = loaded
                _remember_spatial_index(dataset, projection, loaded)
                progress(100, "Loaded spatial road index from cache")
                return loaded
        except (OSError, ValueError, TypeError, AttributeError, pickle.PickleError, EOFError):
            pass

    segments: list[ProjectedRoadSegment] = []
    road_polylines: list[tuple[PointXZ, ...]] = []
    mutable_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    road_total = len(dataset.roads)
    road_interval = max(1, road_total // 16)
    for road_index, feature in enumerate(dataset.roads, start=1):
        if road_index == road_total or road_index % road_interval == 0:
            value = 5 + round(82 * road_index / max(1, road_total))
            progress(value, f"Indexing projected roads {road_index:,}/{road_total:,}")
        points = tuple(projection.to_world(point) for point in feature.points)
        road_polylines.append(points)
        tags = tuple(sorted((str(key), str(value)) for key, value in feature.tags.items()))
        for start, end in zip(points, points[1:]):
            if math.hypot(end[0] - start[0], end[1] - start[1]) <= 0.01:
                continue
            index_number = len(segments)
            segment = ProjectedRoadSegment(start, end, tags, feature.osm_key)
            segments.append(segment)
            minimum_x, minimum_z, maximum_x, maximum_z = segment.bounds
            for bz in range(math.floor(minimum_z / bucket_size), math.floor(maximum_z / bucket_size) + 1):
                for bx in range(math.floor(minimum_x / bucket_size), math.floor(maximum_x / bucket_size) + 1):
                    mutable_buckets[(bx, bz)].append(index_number)
    progress(90, f"Freezing {len(mutable_buckets):,} spatial lookup buckets")
    frozen_buckets = {key: tuple(sorted(set(values))) for key, values in mutable_buckets.items()}
    index = SpatialLookupIndex(
        fingerprint=disk_key,
        bucket_size=bucket_size,
        road_segments=tuple(segments),
        road_buckets=frozen_buckets,
        road_polylines=tuple(road_polylines),
        cache_hit=False,
    )
    _SPATIAL_INDEX_REGISTRY[registry_key] = index
    _remember_spatial_index(dataset, projection, index)
    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        try:
            with temporary.open("wb") as stream:
                pickle.dump(
                    {
                        "schema": CACHE_SCHEMA_VERSION,
                        "spatial_schema": _SPATIAL_CACHE_SCHEMA,
                        "registry_key": registry_key,
                        "index": index,
                    },
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    progress(100, f"Spatial road index ready: {len(segments):,} segments")
    return index


def get_spatial_index(dataset: OsmDataset, projection: BboxProjection) -> SpatialLookupIndex | None:
    fast_key = _spatial_fast_key(dataset, projection)
    fast = _SPATIAL_FAST_REGISTRY.get(fast_key)
    if fast is not None and fast[0] is dataset:
        return fast[1]
    index = _SPATIAL_INDEX_REGISTRY.get(_spatial_registry_key(dataset, projection))
    if index is not None:
        _remember_spatial_index(dataset, projection, index)
    return index


def projected_road_polylines(
    dataset: OsmDataset, projection: BboxProjection
) -> tuple[tuple[PointXZ, ...], ...]:
    """Return the one canonical world-space projection of every OSM road."""

    spatial = get_spatial_index(dataset, projection)
    if spatial is None:
        spatial = prepare_spatial_index(dataset, projection, use_cache=False)
    if len(spatial.road_polylines) != len(dataset.roads):
        # Defensive fallback for manually constructed/legacy indexes.
        return tuple(
            tuple(projection.to_world(point) for point in feature.points)
            for feature in dataset.roads
        )
    return spatial.road_polylines


def nearest_road_point(
    dataset: OsmDataset, projection: BboxProjection, x: float, z: float
) -> tuple[float, float] | None:
    spatial = get_spatial_index(dataset, projection)
    if spatial is None:
        spatial = prepare_spatial_index(dataset, projection, use_cache=False)
    search_radius = max(spatial.bucket_size, 100.0)
    candidate_indices: tuple[int, ...] = ()
    while not candidate_indices and search_radius <= max(projection.world_size, 100.0) * 2.0:
        candidate_indices = spatial.candidate_segment_indices(x-search_radius, z-search_radius, x+search_radius, z+search_radius)
        search_radius *= 2.0
    best: tuple[float, float, float] | None = None
    for index in candidate_indices:
        segment = spatial.road_segments[index]
        dx, dz = segment.end[0]-segment.start[0], segment.end[1]-segment.start[1]
        length2 = dx*dx + dz*dz
        if length2 <= 1e-9:
            continue
        t = max(0.0, min(1.0, ((x-segment.start[0])*dx + (z-segment.start[1])*dz)/length2))
        px, pz = segment.start[0]+t*dx, segment.start[1]+t*dz
        candidate = ((x-px)**2 + (z-pz)**2, px, pz)
        if best is None or candidate < best:
            best = candidate
    return (best[1], best[2]) if best is not None else None


def nearest_road_heading(
    dataset: OsmDataset,
    projection: BboxProjection,
    x: float,
    z: float,
) -> float:
    spatial = get_spatial_index(dataset, projection)
    if spatial is None:
        spatial = prepare_spatial_index(dataset, projection, use_cache=False)
    search_radius = max(spatial.bucket_size, 100.0)
    candidate_indices: tuple[int, ...] = ()
    while not candidate_indices and search_radius <= max(projection.world_size, 100.0) * 2.0:
        candidate_indices = spatial.candidate_segment_indices(
            x - search_radius, z - search_radius, x + search_radius, z + search_radius
        )
        search_radius *= 2.0
    best: tuple[float, float] | None = None
    for index in candidate_indices:
        segment = spatial.road_segments[index]
        dx, dz = segment.end[0] - segment.start[0], segment.end[1] - segment.start[1]
        length2 = dx * dx + dz * dz
        if length2 <= 1e-9:
            continue
        t = max(0.0, min(1.0, ((x - segment.start[0]) * dx + (z - segment.start[1]) * dz) / length2))
        px, pz = segment.start[0] + t * dx, segment.start[1] + t * dz
        distance2 = (x - px) ** 2 + (z - pz) ** 2
        heading = math.degrees(math.atan2(dx, dz)) % 360.0
        candidate = (distance2, heading)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best is not None else 0.0


def _heading_along_vector(heading_degrees: float) -> tuple[float, float]:
    radians = math.radians(heading_degrees)
    return math.sin(radians), math.cos(radians)


def _heading_right_vector(heading_degrees: float) -> tuple[float, float]:
    radians = math.radians(heading_degrees)
    return math.cos(radians), -math.sin(radians)


def nudge_point_away_from_road(
    dataset: OsmDataset,
    projection: BboxProjection,
    x: float,
    z: float,
    *,
    distance: float,
    fallback_heading: float | None = None,
    world_size: float | None = None,
) -> tuple[float, float]:
    """Move a roadside point a small deterministic distance away from the road.

    When the source point already lies on the road centreline, the fallback uses a
    perpendicular to the nearest road heading so objects do not remain marooned
    on the asphalt.
    """

    distance = max(0.0, float(distance))
    if distance <= 0.0:
        return x, z
    road_point = nearest_road_point(dataset, projection, x, z)
    if road_point is not None:
        rx, rz = road_point
        offset_x = x - rx
        offset_z = z - rz
        offset_length = math.hypot(offset_x, offset_z)
        if offset_length > 1e-6:
            unit_x = offset_x / offset_length
            unit_z = offset_z / offset_length
        else:
            heading = fallback_heading if fallback_heading is not None else nearest_road_heading(dataset, projection, x, z)
            unit_x, unit_z = _heading_right_vector(heading)
    elif fallback_heading is not None:
        unit_x, unit_z = _heading_right_vector(fallback_heading)
    else:
        return x, z

    candidates = ((x + unit_x * distance, z + unit_z * distance), (x - unit_x * distance, z - unit_z * distance))
    if world_size is None:
        return candidates[0]
    for candidate_x, candidate_z in candidates:
        if 0.0 <= candidate_x < world_size and 0.0 <= candidate_z < world_size:
            return candidate_x, candidate_z
    return x, z


def _point_to_segment_distance_squared(
    point: PointXZ, start: PointXZ, end: PointXZ
) -> float:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length_squared = dx * dx + dz * dz
    if length_squared <= 1e-12:
        return (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2
    fraction = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz)
            / length_squared,
        ),
    )
    closest_x = start[0] + fraction * dx
    closest_z = start[1] + fraction * dz
    return (point[0] - closest_x) ** 2 + (point[1] - closest_z) ** 2


def _orientation(a: PointXZ, b: PointXZ, c: PointXZ) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a0: PointXZ, a1: PointXZ, b0: PointXZ, b1: PointXZ) -> bool:
    epsilon = 1e-9
    orientations = (
        _orientation(a0, a1, b0),
        _orientation(a0, a1, b1),
        _orientation(b0, b1, a0),
        _orientation(b0, b1, a1),
    )
    if orientations[0] * orientations[1] < -epsilon and orientations[2] * orientations[3] < -epsilon:
        return True

    def on_segment(point: PointXZ, start: PointXZ, end: PointXZ) -> bool:
        return (
            min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
            and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
            and abs(_orientation(start, end, point)) <= epsilon
        )

    return (
        (abs(orientations[0]) <= epsilon and on_segment(b0, a0, a1))
        or (abs(orientations[1]) <= epsilon and on_segment(b1, a0, a1))
        or (abs(orientations[2]) <= epsilon and on_segment(a0, b0, b1))
        or (abs(orientations[3]) <= epsilon and on_segment(a1, b0, b1))
    )


def _segment_distance_squared(a0: PointXZ, a1: PointXZ, b0: PointXZ, b1: PointXZ) -> float:
    if _segments_intersect(a0, a1, b0, b1):
        return 0.0
    return min(
        _point_to_segment_distance_squared(a0, b0, b1),
        _point_to_segment_distance_squared(a1, b0, b1),
        _point_to_segment_distance_squared(b0, a0, a1),
        _point_to_segment_distance_squared(b1, a0, a1),
    )


def line_intersects_road_corridors(
    corridors: Sequence[RoadCorridor],
    start: PointXZ,
    end: PointXZ,
    *,
    clearance: float = 0.0,
) -> bool:
    """Return whether a complete barrier segment touches any buffered road.

    The full segment is checked, rather than its centre, so T-junctions and
    dirt-road branches cannot hide between samples.
    """

    extra = max(0.0, float(clearance))
    minimum_x = min(start[0], end[0]) - extra
    minimum_z = min(start[1], end[1]) - extra
    maximum_x = max(start[0], end[0]) + extra
    maximum_z = max(start[1], end[1]) + extra
    if isinstance(corridors, IndexedRoadCorridors):
        bucket = corridors.bucket_size
        candidate_indices: set[int] = set()
        for bz in range(math.floor(minimum_z / bucket), math.floor(maximum_z / bucket) + 1):
            for bx in range(math.floor(minimum_x / bucket), math.floor(maximum_x / bucket) + 1):
                candidate_indices.update(corridors.buckets.get((bx, bz), ()))
        candidates = (corridors.corridors[index] for index in sorted(candidate_indices))
    else:
        candidates = iter(corridors)
    for road_start, road_end, radius in candidates:
        allowed = max(0.0, radius) + extra
        if _segment_distance_squared(start, end, road_start, road_end) <= allowed * allowed:
            return True
    return False


def polygon_intersects_road_corridors(
    corridors: Sequence[RoadCorridor],
    polygon: Sequence[PointXZ],
    *,
    clearance: float = 0.0,
) -> bool:
    """Return whether a complete building footprint touches a road corridor.

    This uses the projected road geometry rather than the coarse terrain raster,
    which otherwise treats an entire 25 m cell as road and can either miss a
    narrow diagonal crossing or push a large church much farther than necessary.
    """

    if len(polygon) < 3:
        return False
    extra = max(0.0, float(clearance))
    minimum_x = min(point[0] for point in polygon) - extra
    minimum_z = min(point[1] for point in polygon) - extra
    maximum_x = max(point[0] for point in polygon) + extra
    maximum_z = max(point[1] for point in polygon) + extra
    if isinstance(corridors, IndexedRoadCorridors):
        bucket = corridors.bucket_size
        candidate_indices: set[int] = set()
        for bz in range(math.floor(minimum_z / bucket), math.floor(maximum_z / bucket) + 1):
            for bx in range(math.floor(minimum_x / bucket), math.floor(maximum_x / bucket) + 1):
                candidate_indices.update(corridors.buckets.get((bx, bz), ()))
        candidates = (corridors.corridors[index] for index in sorted(candidate_indices))
    else:
        candidates = iter(corridors)
    edges = tuple(zip(polygon, polygon[1:] + polygon[:1]))
    for road_start, road_end, radius in candidates:
        allowed = max(0.0, radius) + extra
        if _point_in_polygon(road_start, polygon) or _point_in_polygon(road_end, polygon):
            return True
        if any(
            _segment_distance_squared(edge_start, edge_end, road_start, road_end)
            <= allowed * allowed
            for edge_start, edge_end in edges
        ):
            return True
    return False


def _nearest_road_corridor_point(
    corridors: Sequence[RoadCorridor],
    x: float,
    z: float,
    *,
    search_radius: float,
) -> tuple[float, float, float, float] | None:
    if not corridors:
        return None
    radius = max(1.0, float(search_radius))
    if isinstance(corridors, IndexedRoadCorridors):
        bucket = corridors.bucket_size
        candidate_indices: set[int] = set()
        for bz in range(math.floor((z - radius) / bucket), math.floor((z + radius) / bucket) + 1):
            for bx in range(math.floor((x - radius) / bucket), math.floor((x + radius) / bucket) + 1):
                candidate_indices.update(corridors.buckets.get((bx, bz), ()))
        candidates = (corridors.corridors[index] for index in sorted(candidate_indices))
    else:
        candidates = iter(corridors)
    best: tuple[float, float, float, float] | None = None
    for road_start, road_end, road_radius in candidates:
        dx = road_end[0] - road_start[0]
        dz = road_end[1] - road_start[1]
        length_squared = dx * dx + dz * dz
        if length_squared <= 1e-12:
            closest_x, closest_z = road_start
        else:
            fraction = max(
                0.0,
                min(
                    1.0,
                    ((x - road_start[0]) * dx + (z - road_start[1]) * dz)
                    / length_squared,
                ),
            )
            closest_x = road_start[0] + fraction * dx
            closest_z = road_start[1] + fraction * dz
        centre_distance = math.hypot(x - closest_x, z - closest_z)
        edge_distance = max(0.0, centre_distance - max(0.0, road_radius))
        candidate = (edge_distance, closest_x, closest_z, max(0.0, road_radius))
        if best is None or candidate < best:
            best = candidate
    return best


def _pull_point_toward_road_frontage(
    corridors: Sequence[RoadCorridor],
    x: float,
    z: float,
    *,
    target_edge_distance: float,
    search_radius: float,
) -> tuple[float, float, float]:
    nearest = _nearest_road_corridor_point(
        corridors, x, z, search_radius=search_radius
    )
    if nearest is None:
        return x, z, math.inf
    edge_distance, road_x, road_z, road_radius = nearest
    target = max(0.0, float(target_edge_distance))
    if edge_distance <= target:
        return x, z, edge_distance
    away_x = x - road_x
    away_z = z - road_z
    length = math.hypot(away_x, away_z)
    if length <= 1e-9:
        return x, z, edge_distance
    scale = (road_radius + target) / length
    shifted_x = road_x + away_x * scale
    shifted_z = road_z + away_z * scale
    return shifted_x, shifted_z, target


def _point_to_road_corridor_distance(
    corridors: Sequence[RoadCorridor],
    x: float,
    z: float,
    *,
    search_radius: float,
) -> float:
    nearest = _nearest_road_corridor_point(corridors, x, z, search_radius=search_radius)
    return nearest[0] if nearest is not None else math.inf


def offset_line_clear_of_roads(
    dataset: OsmDataset,
    projection: BboxProjection,
    corridors: Sequence[RoadCorridor],
    x: float,
    z: float,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    *,
    minimum_distance: float,
    maximum_distance: float = ROADSIDE_BARRIER_MAXIMUM_NUDGE_METRES,
    clearance: float = ROADSIDE_BARRIER_CLEARANCE_METRES,
    world_size: float | None = None,
) -> tuple[float, float, float, float, float, float] | None:
    """Shift a complete line sideways until it clears every road corridor.

    Both sides are tried at increasing distances. If a perpendicular branch or
    intersection still crosses the segment, the caller can omit that piece and
    leave a sensible gate-sized gap rather than placing fence across the road.
    """

    dx = x1 - x0
    dz = z1 - z0
    length = math.hypot(dx, dz)
    if length <= 1e-9:
        return None
    right_x, right_z = dz / length, -dx / length
    left_x, left_z = -right_x, -right_z
    road_point = nearest_road_point(dataset, projection, x, z)
    if road_point is not None:
        away_x = x - road_point[0]
        away_z = z - road_point[1]
        right_score = right_x * away_x + right_z * away_z
        left_score = left_x * away_x + left_z * away_z
        directions = ((right_x, right_z), (left_x, left_z)) if right_score >= left_score else ((left_x, left_z), (right_x, right_z))
    else:
        directions = ((right_x, right_z), (left_x, left_z))

    step = max(0.25, float(minimum_distance))
    limit = max(step, float(maximum_distance))
    distance = step
    while distance <= limit + 1e-9:
        for unit_x, unit_z in directions:
            shifted_x = x + unit_x * distance
            shifted_z = z + unit_z * distance
            shifted_x0 = x0 + unit_x * distance
            shifted_z0 = z0 + unit_z * distance
            shifted_x1 = x1 + unit_x * distance
            shifted_z1 = z1 + unit_z * distance
            if world_size is not None and not (
                0.0 <= shifted_x0 < world_size
                and 0.0 <= shifted_z0 < world_size
                and 0.0 <= shifted_x1 < world_size
                and 0.0 <= shifted_z1 < world_size
            ):
                continue
            if not line_intersects_road_corridors(
                corridors,
                (shifted_x0, shifted_z0),
                (shifted_x1, shifted_z1),
                clearance=clearance,
            ):
                return (
                    shifted_x,
                    shifted_z,
                    shifted_x0,
                    shifted_z0,
                    shifted_x1,
                    shifted_z1,
                )
        distance += step
    return None


def nudge_line_centre_away_from_road(
    dataset: OsmDataset,
    projection: BboxProjection,
    x: float,
    z: float,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    *,
    distance: float,
    world_size: float | None = None,
) -> tuple[float, float]:
    """Move a fitted line segment sideways, preferring the side away from roads."""

    distance = max(0.0, float(distance))
    if distance <= 0.0:
        return x, z
    dx = x1 - x0
    dz = z1 - z0
    length = math.hypot(dx, dz)
    if length <= 1e-9:
        return nudge_point_away_from_road(dataset, projection, x, z, distance=distance, world_size=world_size)
    right_x, right_z = dz / length, -dx / length
    left_x, left_z = -right_x, -right_z
    road_point = nearest_road_point(dataset, projection, x, z)
    if road_point is not None:
        to_object_x = x - road_point[0]
        to_object_z = z - road_point[1]
        right_score = right_x * to_object_x + right_z * to_object_z
        left_score = left_x * to_object_x + left_z * to_object_z
        unit_x, unit_z = (right_x, right_z) if right_score >= left_score else (left_x, left_z)
    else:
        unit_x, unit_z = right_x, right_z
    candidates = ((x + unit_x * distance, z + unit_z * distance), (x - unit_x * distance, z - unit_z * distance))
    if world_size is None:
        return candidates[0]
    for candidate_x, candidate_z in candidates:
        if 0.0 <= candidate_x < world_size and 0.0 <= candidate_z < world_size:
            return candidate_x, candidate_z
    return x, z


def _project_vehicle_road_corridors(
    dataset: OsmDataset, projection: BboxProjection
) -> IndexedRoadCorridors:
    """Project indexed carriageway corridors for road-safe settlement clutter.

    This deliberately excludes pedestrian-only ways and uses the physical road
    half-width without the forest-clearance padding used by
    :func:`project_road_corridors`.  Street furniture should stay off asphalt,
    not disappear several metres into nearby gardens.
    """

    spatial = get_spatial_index(dataset, projection)
    if spatial is None:
        spatial = prepare_spatial_index(dataset, projection, use_cache=False)
    corridors: list[RoadCorridor] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for segment in spatial.road_segments:
        tags = dict(segment.tags)
        highway = tags.get("highway", "").casefold()
        if tags.get("tunnel") not in {None, "", "no"}:
            continue
        if highway in PEDESTRIAN_ONLY_HIGHWAYS or highway not in _MAJOR_HIGHWAYS:
            continue
        radius = max(0.0, road_width_metres(tags) * 0.5)
        corridor_index = len(corridors)
        corridors.append((segment.start, segment.end, radius))
        minimum_x, minimum_z, maximum_x, maximum_z = segment.bounds
        minimum_x -= radius
        minimum_z -= radius
        maximum_x += radius
        maximum_z += radius
        for bz in range(math.floor(minimum_z / spatial.bucket_size), math.floor(maximum_z / spatial.bucket_size) + 1):
            for bx in range(math.floor(minimum_x / spatial.bucket_size), math.floor(maximum_x / spatial.bucket_size) + 1):
                buckets[(bx, bz)].append(corridor_index)
    return IndexedRoadCorridors(
        tuple(corridors),
        spatial.bucket_size,
        {key: tuple(sorted(set(values))) for key, values in buckets.items()},
    )


def project_road_corridors(
    dataset: OsmDataset, projection: BboxProjection, spec: OsmSpec
) -> IndexedRoadCorridors:
    spatial = get_spatial_index(dataset, projection)
    if spatial is None:
        spatial = prepare_spatial_index(dataset, projection, use_cache=False)
    registry_key = (
        spatial.fingerprint,
        bool(spec.include_minor_roads),
        float(spec.forest_road_clearance),
    )
    cached = _ROAD_CORRIDOR_REGISTRY.get(registry_key)
    if cached is not None:
        return cached
    corridors: list[RoadCorridor] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for segment in spatial.road_segments:
        tags = dict(segment.tags)
        if not road_is_supported(tags, include_minor=spec.include_minor_roads):
            continue
        radius = road_width_metres(tags) * 0.5 + spec.forest_road_clearance
        corridor_index = len(corridors)
        corridors.append((segment.start, segment.end, radius))
        minimum_x, minimum_z, maximum_x, maximum_z = segment.bounds
        minimum_x -= radius
        minimum_z -= radius
        maximum_x += radius
        maximum_z += radius
        for bz in range(math.floor(minimum_z / spatial.bucket_size), math.floor(maximum_z / spatial.bucket_size) + 1):
            for bx in range(math.floor(minimum_x / spatial.bucket_size), math.floor(maximum_x / spatial.bucket_size) + 1):
                buckets[(bx, bz)].append(corridor_index)
    result = IndexedRoadCorridors(
        tuple(corridors),
        spatial.bucket_size,
        {key: tuple(sorted(set(values))) for key, values in buckets.items()},
    )
    _ROAD_CORRIDOR_REGISTRY[registry_key] = result
    while len(_ROAD_CORRIDOR_REGISTRY) > 16:
        _ROAD_CORRIDOR_REGISTRY.pop(next(iter(_ROAD_CORRIDOR_REGISTRY)))
    return result


def forest_block_intersects_road_corridors(
    corridors: Sequence[RoadCorridor],
    x: float,
    z: float,
    *,
    block_size: float,
) -> bool:
    half_extent = block_size * 0.5
    minimum_x = x - half_extent
    minimum_z = z - half_extent
    maximum_x = x + half_extent
    maximum_z = z + half_extent
    if isinstance(corridors, IndexedRoadCorridors):
        return corridors.intersects_rectangle(minimum_x, minimum_z, maximum_x, maximum_z)
    for start, end, radius in corridors:
        if _segment_intersects_rectangle(
            start,
            end,
            minimum_x - radius,
            minimum_z - radius,
            maximum_x + radius,
            maximum_z + radius,
        ):
            return True
    return False


def forest_block_intersects_roads(
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: OsmSpec,
    x: float,
    z: float,
    *,
    block_size: float | None = None,
) -> bool:
    """Check the complete square forest footprint against buffered OSM road lines.

    The previous raster-only test sampled the centre and four inset corners. A
    road could therefore clip the outer seven metres of a 50 m forest block and
    still receive a tree model. Expanding the block by half the mapped road width
    plus a configurable margin makes any segment intersection a rejection.
    """
    size = spec.forest_tree_spacing if block_size is None else block_size
    return forest_block_intersects_road_corridors(
        project_road_corridors(dataset, projection, spec), x, z, block_size=size
    )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: GeoPolygon,
    projection: BboxProjection,
    resolution: int,
    *,
    fill: int,
) -> None:
    outer = [projection.to_pixel(point, resolution) for point in polygon.outer]
    if len(outer) >= 3:
        draw.polygon(outer, fill=fill)
    for hole in polygon.holes:
        points = [projection.to_pixel(point, resolution) for point in hole]
        if len(points) >= 3:
            draw.polygon(points, fill=0)


def _mask_coverage_to_cells(image: Image.Image, cells: int) -> tuple[float, ...]:
    """Return supersampled coverage fractions in WRP row order."""

    reduced = image.resize((cells, cells), resample=Image.Resampling.BOX)
    getter = getattr(reduced, "get_flattened_data", None)
    values = getter() if getter is not None else reduced.getdata()
    pixels = tuple(max(0.0, min(1.0, int(value) / 255.0)) for value in values)
    # WRP rows increase from south to north while image rows increase from
    # north to south. Store masks in WRP order and flip only for previews.
    return tuple(
        pixels[(cells - 1 - z) * cells + x]
        for z in range(cells)
        for x in range(cells)
    )


def _mask_to_cells(image: Image.Image, cells: int, *, threshold: int = 64) -> tuple[bool, ...]:
    coverage = _mask_coverage_to_cells(image, cells)
    fraction = max(0.0, min(1.0, int(threshold) / 255.0))
    return tuple(value >= fraction for value in coverage)


def _coastline_water_mask(
    dataset: OsmDataset,
    projection: BboxProjection,
    resolution: int,
) -> tuple[Image.Image, int]:
    barrier = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(barrier)
    seeds: list[tuple[int, int]] = []
    line_width = 2
    offset = 4.0
    for feature in dataset.coastlines:
        pixels = [projection.to_pixel(point, resolution) for point in feature.points]
        if len(pixels) < 2:
            continue
        draw.line(pixels, fill=255, width=line_width)
        for start, end in zip(pixels, pixels[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            middle_x = (start[0] + end[0]) / 2.0
            middle_y = (start[1] + end[1]) / 2.0
            # Image coordinates increase downward. (-dy, dx) is the right
            # normal, matching OSM's coastline convention that water is right.
            seed_x = int(round(middle_x + (-dy / length) * offset))
            seed_y = int(round(middle_y + (dx / length) * offset))
            if 0 <= seed_x < resolution and 0 <= seed_y < resolution:
                seeds.append((seed_x, seed_y))

    result = Image.new("L", (resolution, resolution), 0)
    if not seeds:
        return result, 0
    barrier_pixels = barrier.load()
    result_pixels = result.load()
    queue: deque[tuple[int, int]] = deque()
    for seed in seeds:
        if barrier_pixels[seed] == 0 and result_pixels[seed] == 0:
            result_pixels[seed] = 255
            queue.append(seed)
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if not (0 <= nx < resolution and 0 <= ny < resolution):
                continue
            if barrier_pixels[nx, ny] != 0 or result_pixels[nx, ny] != 0:
                continue
            result_pixels[nx, ny] = 255
            queue.append((nx, ny))
    return result, len(seeds)


def rasterize_osm(
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    cells: int,
    include_minor_roads: bool,
    supersample: int = 4,
    progress_callback: Callable[[int, str], None] | None = None,
) -> OsmRaster:
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    def draw_polygons(
        features: Sequence[OsmPolygonFeature],
        draw: ImageDraw.ImageDraw,
        start_percent: int,
        end_percent: int,
        label: str,
    ) -> None:
        total = len(features)
        progress(start_percent, f"Rasterizing {label}: 0/{total:,}")
        interval = max(1, total // 12)
        for feature_index, feature in enumerate(features, start=1):
            for polygon in feature.polygons:
                _draw_polygon(draw, polygon, projection, resolution, fill=255)
            if feature_index == total or feature_index % interval == 0:
                value = start_percent + round((end_percent - start_percent) * feature_index / max(1, total))
                progress(value, f"Rasterizing {label}: {feature_index:,}/{total:,}")
        if total == 0:
            progress(end_percent, f"Rasterizing {label}: none")

    resolution = cells * supersample
    progress(0, f"Allocating {resolution}×{resolution} OSM raster layers")
    progress(3, f"Rasterizing {len(dataset.coastlines):,} coastline features")
    water_image, seed_count = _coastline_water_mask(dataset, projection, resolution)
    progress(10, f"Coastline water mask ready ({seed_count:,} flood seeds)")
    water_draw = ImageDraw.Draw(water_image)
    forest_image = Image.new("L", (resolution, resolution), 0)
    farmland_image = Image.new("L", (resolution, resolution), 0)
    urban_image = Image.new("L", (resolution, resolution), 0)
    road_image = Image.new("L", (resolution, resolution), 0)
    building_image = Image.new("L", (resolution, resolution), 0)
    forest_draw = ImageDraw.Draw(forest_image)
    farmland_draw = ImageDraw.Draw(farmland_image)
    urban_draw = ImageDraw.Draw(urban_image)
    road_draw = ImageDraw.Draw(road_image)
    building_draw = ImageDraw.Draw(building_image)

    draw_polygons(dataset.water, water_draw, 10, 24, "water polygons")
    draw_polygons(dataset.forests, forest_draw, 24, 40, "forest polygons")
    draw_polygons(dataset.farmland, farmland_draw, 40, 52, "farmland polygons")
    draw_polygons(dataset.urban, urban_draw, 52, 62, "urban polygons")
    draw_polygons(dataset.building_polygons, building_draw, 62, 73, "building footprints")

    total_points = len(dataset.building_points)
    progress(73, f"Rasterizing building points: 0/{total_points:,}")
    point_interval = max(1, total_points // 8)
    for point_index, feature in enumerate(dataset.building_points, start=1):
        x, y = projection.to_pixel(feature.point, resolution)
        radius = max(1, int(round(4.0 / projection.world_size * resolution)))
        building_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        if point_index == total_points or point_index % point_interval == 0:
            value = 73 + round(5 * point_index / max(1, total_points))
            progress(value, f"Rasterizing building points: {point_index:,}/{total_points:,}")
    if total_points == 0:
        progress(78, "Rasterizing building points: none")

    supported_roads = [
        feature for feature in dataset.roads
        if road_is_supported(feature.tags, include_minor=include_minor_roads)
    ]
    total_roads = len(supported_roads)
    progress(78, f"Rasterizing roads: 0/{total_roads:,}")
    road_interval = max(1, total_roads // 12)
    for road_index, feature in enumerate(supported_roads, start=1):
        pixels = [projection.to_pixel(point, resolution) for point in feature.points]
        width = max(1, int(round(road_width_metres(feature.tags) / projection.world_size * resolution)))
        road_draw.line(pixels, fill=255, width=width)
        if road_index == total_roads or road_index % road_interval == 0:
            value = 78 + round(12 * road_index / max(1, total_roads))
            progress(value, f"Rasterizing roads: {road_index:,}/{total_roads:,}")
    if total_roads == 0:
        progress(90, "Rasterizing roads: none")

    progress(91, f"Downsampling raster layers to {cells}×{cells} terrain cells")
    water_coverage = _mask_coverage_to_cells(water_image, cells)
    # The ordinary water mask remains permissive enough to preserve small bays
    # and narrow mapped water. Full-depth terrain carving is restricted to the
    # conservative interior mask. This prevents a mostly-dry 25 m shoreline
    # vertex from being excavated several tens of metres merely because a sliver
    # of the supersampled cell intersects water.
    water_mask = tuple(value >= (64.0 / 255.0) for value in water_coverage)
    water_interior_mask = tuple(value >= (224.0 / 255.0) for value in water_coverage)
    progress(93, "Downsampling forest and farmland masks")
    forest_mask = _mask_to_cells(forest_image, cells)
    farmland_mask = _mask_to_cells(farmland_image, cells)
    progress(96, "Downsampling urban, road and building masks")
    urban_mask = _mask_to_cells(urban_image, cells)
    road_mask = _mask_to_cells(road_image, cells, threshold=32)
    building_mask = _mask_to_cells(building_image, cells, threshold=16)
    progress(100, "OpenStreetMap raster layers ready")
    return OsmRaster(
        cells=cells,
        water=water_mask,
        forest=forest_mask,
        farmland=farmland_mask,
        urban=urban_mask,
        roads=road_mask,
        buildings=building_mask,
        high_resolution=resolution,
        coastline_seed_count=seed_count,
        water_interior=water_interior_mask,
    )



def conservative_water_interior_mask(raster: OsmRaster) -> tuple[bool, ...]:
    """Return cells safe to excavate to the configured deep-water floor.

    New raster snapshots carry supersampled coverage. Old cached rasters do not,
    so fall back to a one-cell cardinal erosion instead of treating every coarse
    shoreline hit as deep water.
    """

    expected = raster.cells * raster.cells
    if len(raster.water_interior) == expected:
        return tuple(bool(value) for value in raster.water_interior)
    result = [False] * expected
    cells = raster.cells
    for index, is_water in enumerate(raster.water):
        if not is_water:
            continue
        x, z = index % cells, index // cells
        if x <= 0 or z <= 0 or x >= cells - 1 or z >= cells - 1:
            continue
        neighbours = (index - 1, index + 1, index - cells, index + cells)
        result[index] = all(raster.water[n] for n in neighbours)
    return tuple(result)


def renderable_water_mask(
    elevations: Sequence[float],
    raster: OsmRaster,
    *,
    sea_level: float,
    water_depth: float,
) -> tuple[bool, ...]:
    """Return water cells that can plausibly meet CWA's global water plane.

    Strongly covered interior cells are always retained. Uncertain shoreline
    cells are retained only when their DEM sample is already near sea level.
    This prevents a 25 m mixed land/water vertex from being excavated to -5 m.
    """

    expected = raster.cells * raster.cells
    if len(elevations) != expected:
        raise ValueError("elevation grid does not match OSM raster")
    interior = conservative_water_interior_mask(raster)
    depth = max(0.0, float(water_depth))
    boundary_ceiling = float(sea_level) + max(0.75, min(2.0, depth * 0.40))
    interior_ceiling = float(sea_level) + max(1.5, min(3.0, depth * 0.60))
    return tuple(
        bool(is_water)
        and float(elevations[index]) <= (interior_ceiling if interior[index] else boundary_ceiling)
        for index, is_water in enumerate(raster.water)
    )


def _cardinal_distance_from_mask(mask: Sequence[bool], cells: int, maximum: int) -> tuple[int, ...]:
    distances = [-1] * (cells * cells)
    queue: deque[int] = deque()
    for index, value in enumerate(mask):
        if value:
            distances[index] = 0
            queue.append(index)
    while queue:
        index = queue.popleft()
        distance = distances[index]
        if distance >= maximum:
            continue
        x, z = index % cells, index // cells
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if 0 <= nx < cells and 0 <= nz < cells:
                neighbour = nz * cells + nx
                if distances[neighbour] < 0:
                    distances[neighbour] = distance + 1
                    queue.append(neighbour)
    return tuple(distances)

def apply_water_elevations(
    elevations: Sequence[float],
    raster: OsmRaster,
    *,
    sea_level: float,
    water_depth: float,
    beach_height: float,
    blend_cells: int,
    cell_size: float = 25.0,
    maximum_shore_slope_percent: float = 8.0,
) -> tuple[float, ...]:
    """Apply conservative water beds plus an adaptive shoreline ramp.

    Only strongly water-covered cells receive the full water-depth excavation.
    Low, uncertain shoreline cells are kept just under the global water plane.
    Dry terrain is lowered only as much as needed to satisfy the requested
    shoreline grade, up to the engine-oriented 32-cell safety cap.
    """

    cells = raster.cells
    expected = cells * cells
    if len(elevations) != expected:
        raise ValueError("elevation grid does not match OSM raster")
    result = [float(value) for value in elevations]
    active = renderable_water_mask(
        elevations, raster, sea_level=sea_level, water_depth=water_depth
    )
    deep = conservative_water_interior_mask(raster)
    if not any(active):
        return tuple(result)

    deep_target = float(sea_level) - max(0.0, float(water_depth))
    shallow_depth = min(0.35, max(0.05, max(0.0, float(water_depth)) * 0.10))
    shallow_target = float(sea_level) - shallow_depth
    for index, is_active in enumerate(active):
        if not is_active:
            continue
        target = deep_target if deep[index] else shallow_target
        result[index] = min(result[index], target)

    slope_percent = max(0.1, float(maximum_shore_slope_percent))
    rise_per_cell = max(0.05, float(cell_size) * slope_percent / 100.0)
    configured = max(0, int(blend_cells))
    # Inspect the first dry ring. A mixed shoreline vertex can be tens of metres
    # above sea level; determine the transition width from that actual rise.
    first = _cardinal_distance_from_mask(active, cells, 1)
    first_ring_high = max(
        (float(elevations[i]) for i, d in enumerate(first) if d == 1),
        default=float(sea_level),
    )
    required = int(math.ceil(max(0.0, first_ring_high - float(sea_level)) / rise_per_cell)) + 1
    effective = min(32, max(configured, required))
    if effective <= 0:
        return tuple(result)
    distances = _cardinal_distance_from_mask(active, cells, effective)
    for index, distance in enumerate(distances):
        if distance <= 0 or distance > effective:
            continue
        slope_target = float(sea_level) + rise_per_cell * distance
        # Never bulldoze a real coastal bluff merely to make the numerical slope
        # target true. Adaptive width handles ordinary low banks; cells more than
        # one beach-height above the ideal ramp are preserved as natural cliffs.
        cut_budget = max(0.5, max(0.0, float(beach_height)))
        if float(elevations[index]) > slope_target + cut_budget:
            continue
        configured_target = float(sea_level) + max(0.0, float(beach_height)) * distance / max(1, effective)
        target = max(slope_target, configured_target)
        result[index] = min(result[index], target)
    return tuple(result)

def overlay_materials(base_indices: Sequence[int], raster: OsmRaster) -> tuple[int, ...]:
    if len(base_indices) != raster.cells * raster.cells:
        raise ValueError("material grid does not match OSM raster")
    result = list(base_indices)
    for index in range(len(result)):
        if raster.water[index]:
            result[index] = 0
        elif raster.roads[index]:
            result[index] = 7
        elif raster.buildings[index] or raster.urban[index]:
            result[index] = 6
        elif raster.forest[index]:
            # Preserve the base rock material on forest cells that the terrain
            # classifier considers too steep or high for a full forest block.
            # Ordinary forest cells still receive the Everon forest tile.
            if result[index] != 3:
                result[index] = 4
        elif raster.farmland[index]:
            result[index] = 5
    return tuple(result)


def _sample_elevation(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> float:
    fx = max(0.0, min(cells - 1.0, x / cell_size))
    fz = max(0.0, min(cells - 1.0, z / cell_size))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(cells - 1, x0 + 1)
    z1 = min(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    a = elevations[z0 * cells + x0] * (1.0 - tx) + elevations[z0 * cells + x1] * tx
    b = elevations[z1 * cells + x0] * (1.0 - tx) + elevations[z1 * cells + x1] * tx
    return a * (1.0 - tz) + b * tz


def _triangle_elevation_bounds(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> tuple[float, float]:
    """Return the low/high terrain surface for either RVW4 quad diagonal.

    RVW4 stores one height at each terrain sample, but the game displays two
    triangles between neighbouring samples.  A bilinear sample can sit between
    the two possible triangulations on a saddle-shaped quad, leaving a rigid
    building or bridge visibly below the rendered ground.  The file format does
    not record a per-quad diagonal, and game/rendering variants do not all agree
    on that implicit choice, so final object grounding deliberately brackets
    both.  Terrain solving itself remains bilinear and is therefore unchanged.
    """

    fx = max(0.0, min(cells - 1.0, x / cell_size))
    fz = max(0.0, min(cells - 1.0, z / cell_size))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(cells - 1, x0 + 1)
    z1 = min(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    h00 = elevations[z0 * cells + x0]
    h10 = elevations[z0 * cells + x1]
    h01 = elevations[z1 * cells + x0]
    h11 = elevations[z1 * cells + x1]

    # Diagonal h00--h11.
    if tz <= tx:
        main_diagonal = h00 + tx * (h10 - h00) + tz * (h11 - h10)
    else:
        main_diagonal = h00 + tx * (h11 - h01) + tz * (h01 - h00)

    # Diagonal h10--h01.
    if tx + tz <= 1.0:
        cross_diagonal = h00 + tx * (h10 - h00) + tz * (h01 - h00)
    else:
        cross_diagonal = (
            h11
            + (1.0 - tz) * (h10 - h11)
            + (1.0 - tx) * (h01 - h11)
        )
    return min(main_diagonal, cross_diagonal), max(main_diagonal, cross_diagonal)


def _minimum_ground_elevation(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> float:
    return _triangle_elevation_bounds(elevations, cells, cell_size, x, z)[0]


def _maximum_ground_elevation(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> float:
    return _triangle_elevation_bounds(elevations, cells, cell_size, x, z)[1]


def _terrain_axis_breakpoints(
    minimum: float, maximum: float, cells: int, cell_size: float
) -> tuple[float, ...]:
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    values = {minimum, maximum}
    first = max(0, int(math.ceil(minimum / cell_size)))
    last = min(cells - 1, int(math.floor(maximum / cell_size)))
    for index in range(first, last + 1):
        coordinate = index * cell_size
        if minimum < coordinate < maximum:
            values.add(coordinate)
    return tuple(sorted(values))


def _terrain_patch_centres(
    minimum: float, maximum: float, cells: int, cell_size: float
) -> tuple[float, ...]:
    """Return crossings of both implicit diagonals between terrain samples."""

    if maximum < minimum:
        minimum, maximum = maximum, minimum
    first = max(0, int(math.ceil(minimum / cell_size - 0.5)))
    last = min(cells - 2, int(math.floor(maximum / cell_size - 0.5)))
    return tuple((index + 0.5) * cell_size for index in range(first, last + 1))


def _point_in_polygon(point: PointXZ, polygon: Sequence[PointXZ]) -> bool:
    x, z = point
    inside = False
    previous_x, previous_z = polygon[-1]
    for current_x, current_z in polygon:
        if ((current_z > z) != (previous_z > z)) and (
            x < (previous_x - current_x) * (z - current_z) / (previous_z - current_z) + current_x
        ):
            inside = not inside
        previous_x, previous_z = current_x, current_z
    return inside


def _edge_ground_elevation_samples(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    start: PointXZ,
    end: PointXZ,
    x_breakpoints: Sequence[float],
    z_breakpoints: Sequence[float],
) -> tuple[float, ...]:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    parameters = {0.0, 1.0}
    if abs(dx) > 1e-12:
        for coordinate in x_breakpoints:
            parameter = (coordinate - start[0]) / dx
            if 0.0 < parameter < 1.0:
                parameters.add(parameter)
    if abs(dz) > 1e-12:
        for coordinate in z_breakpoints:
            parameter = (coordinate - start[1]) / dz
            if 0.0 < parameter < 1.0:
                parameters.add(parameter)
    # Each grid patch has two possible triangle creases. Add intersections with
    # both diagonals so extrema of either piecewise-planar surface cannot fall
    # between the ordinary X/Z grid crossings.
    ordered = sorted(parameters)
    for lower, upper in zip(ordered, ordered[1:]):
        if upper - lower <= 1e-12:
            continue
        middle = (lower + upper) * 0.5
        middle_x = start[0] + dx * middle
        middle_z = start[1] + dz * middle
        patch_x = max(0, min(cells - 2, int(math.floor(middle_x / cell_size))))
        patch_z = max(0, min(cells - 2, int(math.floor(middle_z / cell_size))))
        origin_x = patch_x * cell_size
        origin_z = patch_z * cell_size
        for diagonal_sum in (False, True):
            if diagonal_sum:
                denominator = dx + dz
                numerator = cell_size - (start[0] - origin_x) - (start[1] - origin_z)
            else:
                denominator = dx - dz
                numerator = (start[1] - origin_z) - (start[0] - origin_x)
            if abs(denominator) <= 1e-12:
                continue
            parameter = numerator / denominator
            if lower + 1e-12 < parameter < upper - 1e-12:
                parameters.add(parameter)

    return tuple(
        bound
        for parameter in parameters
        for bound in _triangle_elevation_bounds(
            elevations,
            cells,
            cell_size,
            start[0] + dx * parameter,
            start[1] + dz * parameter,
        )
    )


def _polygon_ground_elevation_samples(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    polygon: Sequence[PointXZ],
) -> tuple[float, ...]:
    """Sample every arrangement vertex needed for exact triangle extrema."""

    if len(polygon) < 3:
        raise ValueError("terrain support polygon requires at least three points")
    minimum_x = min(point[0] for point in polygon)
    maximum_x = max(point[0] for point in polygon)
    minimum_z = min(point[1] for point in polygon)
    maximum_z = max(point[1] for point in polygon)
    x_breakpoints = _terrain_axis_breakpoints(
        minimum_x, maximum_x, cells, cell_size
    )
    z_breakpoints = _terrain_axis_breakpoints(
        minimum_z, maximum_z, cells, cell_size
    )
    points = list(polygon)
    points.extend(
        (x, z)
        for x in x_breakpoints
        for z in z_breakpoints
        if _point_in_polygon((x, z), polygon)
    )
    points.extend(
        (centre_x, centre_z)
        for centre_x in _terrain_patch_centres(
            minimum_x, maximum_x, cells, cell_size
        )
        for centre_z in _terrain_patch_centres(
            minimum_z, maximum_z, cells, cell_size
        )
        if _point_in_polygon((centre_x, centre_z), polygon)
    )
    values = [
        bound
        for x, z in points
        for bound in _triangle_elevation_bounds(
            elevations, cells, cell_size, x, z
        )
    ]
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        values.extend(
            _edge_ground_elevation_samples(
                elevations,
                cells,
                cell_size,
                start,
                end,
                x_breakpoints,
                z_breakpoints,
            )
        )
    return tuple(values)


def _maximum_polygon_elevation(
    elevations: Sequence[float], cells: int, cell_size: float, polygon: Sequence[PointXZ]
) -> float:
    return max(
        _polygon_ground_elevation_samples(
            elevations, cells, cell_size, polygon
        )
    )




def _oriented_rectangle(
    x: float,
    z: float,
    width: float,
    length: float,
    heading_degrees: float,
    *,
    margin: float = 0.0,
) -> tuple[PointXZ, ...]:
    """Return the actual world-space footprint of a centred rectangular model.

    ``WorldObject.matrix_4x3`` maps the model's local X axis across its width
    and local Z axis along its length.  Grounding must use this rectangle, not
    merely the source OSM polygon, because procedural dimensions are quantized
    and capped variants may be larger than the requested footprint.
    """

    half_width = max(0.05, width * 0.5 + margin)
    half_length = max(0.05, length * 0.5 + margin)
    angle = math.radians(heading_degrees)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    return tuple(
        (
            x + width_sign * half_width * width_axis[0] + length_sign * half_length * length_axis[0],
            z + width_sign * half_width * width_axis[1] + length_sign * half_length * length_axis[1],
        )
        for width_sign, length_sign in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    )




def _procedural_support_polygon(
    x: float,
    z: float,
    heading_degrees: float,
    selected: Any,
) -> tuple[PointXZ, ...]:
    """Return the exact authored footprint of one procedural building model."""

    local_vertices = tuple(getattr(selected, "footprint_vertices", ()) or ())
    if not local_vertices:
        return _oriented_rectangle(
            x, z, float(selected.width_m), float(selected.length_m), heading_degrees
        )
    angle = math.radians(heading_degrees)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    return tuple(
        (
            x + float(local_x) * width_axis[0] + float(local_z) * length_axis[0],
            z + float(local_x) * width_axis[1] + float(local_z) * length_axis[1],
        )
        for local_x, local_z in local_vertices
    )

def _nudge_building_inside_sampled_terrain(
    x: float,
    z: float,
    support_polygon: Sequence[PointXZ],
    spec: OsmSpec,
) -> tuple[float, float, tuple[PointXZ, ...], bool]:
    """Translate a rigid building footprint inside the directly sampled WRP grid.

    The debug world supplied for 0.9.206 exposed Vansö kyrka with its procedural
    footprint ending only about 6 cm from the nominal 6400 m world boundary.
    With a 256 x 25 m RVW4 grid, however, the last stored terrain vertex is at
    6375 m. Houses are usually small enough not to reach that outer strip; a
    church can span almost the whole final cell and appear buried.
    """

    if len(support_polygon) < 3 or int(spec.cells) < 16:
        # Tiny synthetic grids used by low-level tests do not model a practical
        # OFP island boundary. Preserve their historical grounding semantics.
        return x, z, tuple(support_polygon), False
    minimum = BUILDING_TERRAIN_EDGE_MARGIN_METRES
    maximum = (int(spec.cells) - 1) * float(spec.cell_size) - BUILDING_TERRAIN_EDGE_MARGIN_METRES
    if maximum <= minimum:
        return x, z, tuple(support_polygon), False
    min_x = min(point[0] for point in support_polygon)
    max_x = max(point[0] for point in support_polygon)
    min_z = min(point[1] for point in support_polygon)
    max_z = max(point[1] for point in support_polygon)
    dx = 0.0
    dz = 0.0
    if min_x < minimum:
        dx += minimum - min_x
    if max_x + dx > maximum:
        dx += maximum - (max_x + dx)
    if min_z < minimum:
        dz += minimum - min_z
    if max_z + dz > maximum:
        dz += maximum - (max_z + dz)
    if abs(dx) <= 1e-9 and abs(dz) <= 1e-9:
        return x, z, tuple(support_polygon), False
    shifted = tuple((px + dx, pz + dz) for px, pz in support_polygon)
    return x + dx, z + dz, shifted, True

def _enterable_building_entrance_apron(
    plan: BuildingPlacementPlan,
) -> tuple[PointXZ, ...]:
    """Return a small terrain-check polygon across and outside the front door."""

    selected = getattr(
        getattr(plan, "procedural_placement", None), "selected", None
    )
    if selected is None or not bool(getattr(selected, "interiors", False)):
        return ()
    width = float(getattr(selected, "width_m", 0.0))
    length = float(getattr(selected, "length_m", 0.0))
    if width <= 0.0 or length <= 0.0:
        return ()
    angle = math.radians(plan.heading_degrees)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    front_x = plan.x - length * 0.5 * length_axis[0]
    front_z = plan.z - length * 0.5 * length_axis[1]
    door_half = min(0.8, max(0.6, width * 0.5 * 0.18))
    apron_half_width = door_half + ENTERABLE_BUILDING_ENTRANCE_APRON_SIDE_MARGIN_METRES

    def point(width_offset: float, length_offset: float) -> PointXZ:
        return (
            front_x
            + width_offset * width_axis[0]
            + length_offset * length_axis[0],
            front_z
            + width_offset * width_axis[1]
            + length_offset * length_axis[1],
        )

    inside = ENTERABLE_BUILDING_ENTRANCE_APRON_INSET_METRES
    outside = -ENTERABLE_BUILDING_ENTRANCE_APRON_DEPTH_METRES
    return (
        point(-apron_half_width, inside),
        point(apron_half_width, inside),
        point(apron_half_width, outside),
        point(-apron_half_width, outside),
    )


def _minimum_polygon_elevation(
    elevations: Sequence[float], cells: int, cell_size: float, polygon: Sequence[PointXZ]
) -> float:
    return min(
        _polygon_ground_elevation_samples(
            elevations, cells, cell_size, polygon
        )
    )


def _polygon_elevation_extrema(
    elevations: Sequence[float], cells: int, cell_size: float, polygon: Sequence[PointXZ]
) -> tuple[float, float]:
    samples = _polygon_ground_elevation_samples(
        elevations, cells, cell_size, polygon
    )
    return min(samples), max(samples)


def _building_plan_fully_submerged(
    plan: BuildingPlacementPlan,
    elevations: Sequence[float],
    raster: OsmRaster,
    spec: OsmSpec,
) -> bool:
    """Reject only a footprint wholly in water and wholly below sea level."""

    footprint = tuple(plan.support_polygon)
    world_size = float(
        getattr(spec, "world_size", float(spec.cells) * float(spec.cell_size))
    )
    if not _polygon_fully_covered_by_mask(
        raster.water, spec.cells, world_size, footprint
    ):
        return False
    maximum = _maximum_polygon_elevation(
        elevations, spec.cells, spec.cell_size, footprint
    )
    return maximum < float(getattr(spec, "sea_level", 0.0)) - 1e-6


def refine_iterative_grounding_terrain(
    elevations: Sequence[float],
    provisional: ObjectGenerationResult,
    building_placement_plans: Sequence[BuildingPlacementPlan],
    raster: OsmRaster,
    spec: OsmSpec,
) -> tuple[tuple[float, ...], IterativeGroundingReport]:
    """Run the object-aware terrain correction stage of the six-step pass.

    Buildings receive a flat support target at the highest provisional terrain
    beneath their complete footprint. Vegetation deliberately contributes no
    terrain constraints: after terrain quantization, trees and forest proxies must
    fit the final terrain or be rejected/fallen back rather than reshaping it.

    Road and water cells are immutable here. Corrections are blended and capped
    so the second pass improves structural contact without creating tree shelves.
    """

    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    if len(elevations) != cells * cells:
        raise ValueError("iterative grounding terrain has the wrong dimensions")
    if not provisional.objects:
        return tuple(float(value) for value in elevations), IterativeGroundingReport()

    maximum_adjustment = max(
        0.0, float(getattr(spec, "iterative_grounding_maximum_adjustment", 2.0))
    )
    blend = max(
        0.0, min(1.0, float(getattr(spec, "iterative_grounding_strength", 0.70)))
    )
    if maximum_adjustment <= 0.0 or blend <= 0.0:
        return tuple(float(value) for value in elevations), IterativeGroundingReport()

    target_sums = [0.0] * (cells * cells)
    target_counts = [0] * (cells * cells)
    priorities = [0] * (cells * cells)
    building_supports = 0
    tree_supports = 0

    def add_support(
        polygon: Sequence[PointXZ],
        target_at: Callable[[float, float], float],
        *,
        priority: int,
        exclude_buildings: bool,
    ) -> bool:
        if len(polygon) < 3:
            return False
        minimum_x = min(point[0] for point in polygon)
        maximum_x = max(point[0] for point in polygon)
        minimum_z = min(point[1] for point in polygon)
        maximum_z = max(point[1] for point in polygon)
        col0 = max(0, int(math.ceil(minimum_x / cell_size)))
        col1 = min(cells - 1, int(math.floor(maximum_x / cell_size)))
        row0 = max(0, int(math.ceil(minimum_z / cell_size)))
        row1 = min(cells - 1, int(math.floor(maximum_z / cell_size)))
        candidates: list[tuple[int, float, float]] = []
        for row in range(row0, row1 + 1):
            z = row * cell_size
            for col in range(col0, col1 + 1):
                x = col * cell_size
                if _point_in_polygon((x, z), polygon):
                    candidates.append((row * cells + col, x, z))
        if not candidates:
            centre_x = sum(point[0] for point in polygon) / len(polygon)
            centre_z = sum(point[1] for point in polygon) / len(polygon)
            col = max(0, min(cells - 1, int(centre_x // cell_size)))
            row = max(0, min(cells - 1, int(centre_z // cell_size)))
            candidates.append((row * cells + col, col * cell_size, row * cell_size))

        accepted = False
        for index, x, z in candidates:
            if raster.water[index] or raster.roads[index]:
                continue
            if exclude_buildings and raster.buildings[index]:
                continue
            target = float(target_at(x, z))
            if priority > priorities[index]:
                priorities[index] = priority
                target_sums[index] = target
                target_counts[index] = 1
            elif priority == priorities[index]:
                target_sums[index] += target
                target_counts[index] += 1
            accepted = True
        return accepted

    building_objects = provisional.objects[:provisional.building_objects]
    active_building_plans = tuple(
        plan
        for plan in building_placement_plans
        if not _building_plan_fully_submerged(plan, elevations, raster, spec)
    )
    for plan, obj in zip(active_building_plans, building_objects):
        footprint = tuple(plan.support_polygon)
        # Every source footprint is one building object. Terrain support is
        # therefore computed from that one final authored footprint, whether it
        # is rectangular or polygon-native.
        target = _maximum_polygon_elevation(
            elevations, cells, cell_size, footprint
        )
        accepted = add_support(
            footprint,
            lambda _x, _z, value=target: value,
            priority=2,
            exclude_buildings=False,
        )
        building_supports += int(accepted)

    # Vegetation is grounded only after this terrain is final.
    tree_supports = 0

    refined = [float(value) for value in elevations]
    adjustments: list[float] = []
    for index, count in enumerate(target_counts):
        if count <= 0:
            continue
        target = target_sums[index] / count
        raw_adjustment = target - refined[index]
        adjustment = max(
            -maximum_adjustment,
            min(maximum_adjustment, raw_adjustment),
        ) * blend
        if abs(adjustment) <= 1e-9:
            continue
        refined[index] += adjustment
        adjustments.append(abs(adjustment))

    report = IterativeGroundingReport(
        building_supports=building_supports,
        tree_supports=tree_supports,
        adjusted_cells=len(adjustments),
        maximum_adjustment=max(adjustments, default=0.0),
        mean_adjustment=(sum(adjustments) / len(adjustments) if adjustments else 0.0),
    )
    return tuple(refined), report


def plan_iterative_grounding_objects(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    spec: OsmSpec,
    building_placement_plans: Sequence[BuildingPlacementPlan],
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> ObjectGenerationResult:
    """Plan structural supports needed by iterative terrain grounding.

    Vegetation is intentionally excluded. Buildings may engineer terrain; trees
    must accept the final terrain and are rejected or decomposed when they cannot.
    """

    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    objects: list[WorldObject] = []
    next_id = 1
    progress(0, "Planning building grounding supports")
    active_building_plans = tuple(
        plan
        for plan in building_placement_plans
        if not _building_plan_fully_submerged(plan, elevations, raster, spec)
    )
    for plan in active_building_plans:
        objects.append(
            WorldObject(
                next_id,
                plan.model_path,
                plan.x,
                _sample_elevation(
                    elevations, spec.cells, spec.cell_size, plan.x, plan.z
                ),
                plan.z,
                plan.heading_degrees,
            )
        )
        next_id += 1

    # Terrain refinement deliberately stops at structures. Forest placement is
    # performed only after the quantized terrain is final, so no tree or proxy
    # can create a terrain shelf merely to make its model fit.
    forest_count = 0

    progress(100, "Grounding support plan ready")
    return ObjectGenerationResult(
        objects=tuple(objects),
        road_objects=0,
        building_objects=len(active_building_plans),
        forest_objects=forest_count,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
    )


def _local_terrain_gradient(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> tuple[float, float]:
    step = max(2.0, cell_size * 0.5)
    world_limit = cells * cell_size - 0.001
    west = _sample_elevation(elevations, cells, cell_size, max(0.0, x - step), z)
    east = _sample_elevation(elevations, cells, cell_size, min(world_limit, x + step), z)
    south = _sample_elevation(elevations, cells, cell_size, x, max(0.0, z - step))
    north = _sample_elevation(elevations, cells, cell_size, x, min(world_limit, z + step))
    x_span = max(1e-6, min(world_limit, x + step) - max(0.0, x - step))
    z_span = max(1e-6, min(world_limit, z + step) - max(0.0, z - step))
    return (east - west) / x_span, (north - south) / z_span


def _terrain_fit_anchor(
    supports: Sequence[float],
    *,
    clearance: float,
    maximum_burial: float,
    maximum_float: float,
) -> tuple[float, float, float] | None:
    """Choose an origin height near terrain level with bounded visual error.

    Each support value is the origin height that would put one trunk/model base
    exactly on terrain. The selected anchor is biased toward the median, then
    clamped so no base is buried or floating beyond the configured limits.
    """

    if not supports:
        return None
    ordered = sorted(float(value) for value in supports)
    lower = ordered[-1] - max(0.0, maximum_burial)
    upper = ordered[0] + max(0.0, maximum_float)
    if lower > upper + 1e-9:
        return None
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) * 0.5
    )
    desired = median + clearance
    anchor = max(lower, min(upper, desired))
    burial = max(0.0, ordered[-1] - anchor)
    floating = max(0.0, anchor - ordered[0])
    return anchor, burial, floating


def _square_elevation_samples(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float, size: float
) -> tuple[float, ...]:
    half = size * 0.5
    minimum_x = max(0.0, x - half)
    maximum_x = min(cells * cell_size - 0.001, x + half)
    minimum_z = max(0.0, z - half)
    maximum_z = min(cells * cell_size - 0.001, z + half)
    polygon = (
        (minimum_x, minimum_z),
        (maximum_x, minimum_z),
        (maximum_x, maximum_z),
        (minimum_x, maximum_z),
    )
    world_limit = cells * cell_size - 0.001
    if (
        x - half < 0.0
        or z - half < 0.0
        or x + half > world_limit
        or z + half > world_limit
    ):
        # Keep the generic clipping/boundary semantics for footprints touching
        # the finite terrain edge. Primary forest blocks are guarded inland, so
        # their hot path uses the specialized rectangle sampler below.
        return _polygon_ground_elevation_samples(
            elevations, cells, cell_size, polygon
        )

    # This is the same arrangement sampled by
    # _polygon_ground_elevation_samples(), specialized for an axis-aligned
    # rectangle. Forest placement calls it tens of thousands of times, while
    # the generic path repeatedly performs point-in-polygon tests for grid and
    # patch-centre points that are trivially inside this rectangle. Preserve
    # the exact support-value multiset (and therefore min/max/median fitting)
    # without that geometry bookkeeping.
    x_breakpoints = _terrain_axis_breakpoints(
        minimum_x, maximum_x, cells, cell_size
    )
    z_breakpoints = _terrain_axis_breakpoints(
        minimum_z, maximum_z, cells, cell_size
    )
    values: list[float] = []

    for sample_x, sample_z in polygon:
        values.extend(
            _triangle_elevation_bounds(
                elevations, cells, cell_size, sample_x, sample_z
            )
        )

    # For this vertex ordering, the generic ray-cast includes the minimum X/Z
    # boundaries and excludes the maximum boundaries. Polygon corners above
    # already retain all four explicit corner samples, matching the old
    # duplicate semantics used by the median-based terrain fit.
    for sample_x in x_breakpoints:
        if not (minimum_x <= sample_x < maximum_x):
            continue
        for sample_z in z_breakpoints:
            if minimum_z <= sample_z < maximum_z:
                values.extend(
                    _triangle_elevation_bounds(
                        elevations, cells, cell_size, sample_x, sample_z
                    )
                )

    patch_xs = _terrain_patch_centres(
        minimum_x, maximum_x, cells, cell_size
    )
    patch_zs = _terrain_patch_centres(
        minimum_z, maximum_z, cells, cell_size
    )
    for sample_x in patch_xs:
        if not (minimum_x <= sample_x < maximum_x):
            continue
        for sample_z in patch_zs:
            if minimum_z <= sample_z < maximum_z:
                values.extend(
                    _triangle_elevation_bounds(
                        elevations, cells, cell_size, sample_x, sample_z
                    )
                )

    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        values.extend(
            _edge_ground_elevation_samples(
                elevations,
                cells,
                cell_size,
                start,
                end,
                x_breakpoints,
                z_breakpoints,
            )
        )
    return tuple(values)


def _non_buried_vegetation_anchor(
    supports: Sequence[float], *, clearance: float
) -> float:
    """Keep a vegetation base above every supplied terrain support."""

    if not supports:
        raise ValueError("vegetation grounding requires at least one support")
    return max(float(value) for value in supports) + max(0.0, float(clearance))


def _non_buried_vegetation_fit(
    supports: Sequence[float], *, clearance: float, maximum_float: float
) -> tuple[float, float] | None:
    """Return a non-buried anchor only when its low-side gap stays bounded."""

    anchor = _non_buried_vegetation_anchor(supports, clearance=clearance)
    floating = max(0.0, anchor - min(float(value) for value in supports))
    if floating > max(0.0, float(maximum_float)) + 1.0e-9:
        return None
    return anchor, floating


def _rooted_tree_fit(
    supports: Sequence[float], *, root_sink: float, maximum_burial: float
) -> tuple[float, float] | None:
    """Ground a tree root conservatively without creating a visible air gap.

    4WVR stores only the four corner elevations of a terrain quad, not an
    explicit diagonal.  The two possible rendered triangle surfaces can differ
    substantially on a saddle/steep cell.  Treating a tree like a rigid building
    and lifting it above *both* surfaces rejects most useful hillside positions.

    Trees are point-like and tolerate a modest amount of trunk burial much
    better than any floating root.  Anchor at the lower possible surface, sink
    the root a few centimetres, and reject only when the alternate surface would
    bury more than ``maximum_burial``.  This guarantees zero positive root float
    whichever terrain diagonal the engine chooses.
    """

    if not supports:
        raise ValueError("tree grounding requires at least one support")
    lower = min(float(value) for value in supports)
    upper = max(float(value) for value in supports)
    sink = max(0.0, float(root_sink))
    anchor = lower - sink
    burial = max(0.0, upper - anchor)
    if burial > max(0.0, float(maximum_burial)) + 1.0e-9:
        return None
    return anchor, burial


def _cluster_heading_and_grade(
    gradient_x: float,
    gradient_z: float,
    heading: float,
    slope_axis: str,
) -> tuple[float, float]:
    """Orient the model so its positive incline axis points uphill."""

    angle = math.radians(heading)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    axis = width_axis if slope_axis == "width" else length_axis
    signed_grade = gradient_x * axis[0] + gradient_z * axis[1]
    if signed_grade < 0.0:
        heading = (heading + 180.0) % 360.0
        signed_grade = -signed_grade
    return heading, quantize_cluster_grade(signed_grade)


def _cluster_supports(
    variant: ForestClusterVariant,
    *,
    grade: float,
    heading: float,
    x: float,
    z: float,
    elevations: Sequence[float],
    raster: OsmRaster,
    cells: int,
    cell_size: float,
    world_size: float,
    require_forest: bool,
    minimum_forest_fraction: float,
) -> tuple[float, ...] | None:
    angle = math.radians(heading)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    supports: list[float] = []
    forest_points = 0
    proxy_points = 0
    for _model, local_x, local_z, _scale, _proxy_heading in variant.proxy_layout:
        world_x = x + local_x * width_axis[0] + local_z * length_axis[0]
        world_z = z + local_x * width_axis[1] + local_z * length_axis[1]
        if not (0.0 <= world_x < world_size and 0.0 <= world_z < world_size):
            return None
        in_forest = _mask_at(raster.forest, cells, world_size, world_x, world_z)
        proxy_points += 1
        forest_points += int(in_forest)
        if require_forest and not in_forest:
            return None
        if (
            _mask_at(raster.water, cells, world_size, world_x, world_z)
            or _mask_at(raster.roads, cells, world_size, world_x, world_z)
            or _mask_at(raster.buildings, cells, world_size, world_x, world_z)
        ):
            return None
        model_y = grade * (local_x if variant.slope_axis == "width" else local_z)
        minimum_ground, maximum_ground = _triangle_elevation_bounds(
            elevations, cells, cell_size, world_x, world_z
        )
        supports.extend((minimum_ground - model_y, maximum_ground - model_y))
    if not supports:
        return None
    if forest_points / proxy_points < minimum_forest_fraction:
        return None
    return tuple(supports)



def _forest_proxy_is_tree(model_path: str) -> bool:
    folded = str(model_path).replace("/", "\\").casefold()
    # Known bushes/ground vegetation receive the looser bush tolerance. Unknown
    # vegetation is treated as a tree, which is intentionally conservative.
    bush_tokens = ("\\ker", "bush", "rakosi", "travy", "grass", "reed")
    return not any(token in folded for token in bush_tokens)


def _cluster_proxy_floats(
    variant: ForestClusterVariant,
    *,
    grade: float,
    heading: float,
    x: float,
    y: float,
    z: float,
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
) -> tuple[float, float, int, int]:
    angle = math.radians(heading)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    maximum_tree_float = 0.0
    maximum_bush_float = 0.0
    tree_count = 0
    bush_count = 0
    for model, local_x, local_z, _scale, _proxy_heading in variant.proxy_layout:
        world_x = x + local_x * width_axis[0] + local_z * length_axis[0]
        world_z = z + local_x * width_axis[1] + local_z * length_axis[1]
        local_y = grade * (local_x if variant.slope_axis == "width" else local_z)
        minimum_ground, _maximum_ground = _triangle_elevation_bounds(
            elevations, cells, cell_size, world_x, world_z
        )
        floating = max(0.0, (y + local_y) - minimum_ground)
        if _forest_proxy_is_tree(model):
            tree_count += 1
            maximum_tree_float = max(maximum_tree_float, floating)
        else:
            bush_count += 1
            maximum_bush_float = max(maximum_bush_float, floating)
    return maximum_tree_float, maximum_bush_float, tree_count, bush_count


def _parse_generated_cluster_model(model_path: str, world_name: str) -> tuple[ForestClusterVariant, float] | None:
    folded = str(model_path).replace("/", "\\")
    prefix = world_name + "\\f\\"
    if not folded.casefold().startswith(prefix.casefold()) or not folded.casefold().endswith(".p3d"):
        return None
    stem = folded.rsplit("\\", 1)[-1][:-4]
    if len(stem) < 5 or stem[1] != "_":
        return None
    body = stem[2:]
    try:
        variant_name, grade_label = body.rsplit("_", 1)
        variant = cluster_variant(variant_name)
        grade = quantize_cluster_grade(int(grade_label) / 100.0)
    except (ValueError, KeyError):
        return None
    return variant, grade


def _audit_vegetation_grounding(
    objects: Sequence[WorldObject],
    elevations: Sequence[float],
    spec: object,
) -> tuple[int, int, int, int, float, float]:
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))
    world_name = str(getattr(spec, "name", "cwr_world"))
    tree_limit = max(0.0, float(getattr(spec, "forest_single_tree_maximum_float", 0.15)))
    cluster_tree_limit = max(0.0, float(getattr(spec, "forest_cluster_tree_maximum_float", 0.20)))
    cluster_bush_limit = max(0.0, float(getattr(spec, "forest_cluster_bush_maximum_float", 0.60)))
    individual_models = {value.casefold() for value in OSM_INDIVIDUAL_TREE_MODELS}
    individual_models.update(
        str(getattr(spec, field, "")).casefold()
        for field in ("forest_single_tree_model", "forest_hillside_tree_model", "forest_roadside_tree_model")
        if getattr(spec, field, "")
    )
    individual_models.update(
        str(value).casefold() for value in getattr(spec, "forest_roadside_tree_models", ROADSIDE_TREE_MODELS)
    )
    tree_objects = cluster_trees = cluster_bushes = violations = 0
    maximum_tree_float = maximum_bush_float = 0.0
    parsed_model_cache: dict[str, tuple[ForestClusterVariant, float] | None] = {}
    folded_model_cache: dict[str, str] = {}
    for obj in objects:
        if obj.model_path in parsed_model_cache:
            parsed = parsed_model_cache[obj.model_path]
        else:
            parsed = _parse_generated_cluster_model(obj.model_path, world_name)
            parsed_model_cache[obj.model_path] = parsed
        if parsed is not None:
            variant, grade = parsed
            tree_float, bush_float, trees, bushes = _cluster_proxy_floats(
                variant, grade=grade, heading=obj.heading_degrees, x=obj.x, y=obj.y, z=obj.z,
                elevations=elevations, cells=cells, cell_size=cell_size,
            )
            cluster_trees += trees
            cluster_bushes += bushes
            maximum_tree_float = max(maximum_tree_float, tree_float)
            maximum_bush_float = max(maximum_bush_float, bush_float)
            violations += int(trees > 0 and tree_float > cluster_tree_limit + 1.0e-6)
            violations += int(bushes > 0 and bush_float > cluster_bush_limit + 1.0e-6)
            continue
        model = folded_model_cache.get(obj.model_path)
        if model is None:
            model = obj.model_path.casefold()
            folded_model_cache[obj.model_path] = model
        if model.startswith("data3d\\str") or model in individual_models:
            tree_objects += 1
            minimum_ground, _maximum_ground = _triangle_elevation_bounds(
                elevations, cells, cell_size, obj.x, obj.z
            )
            floating = max(0.0, obj.y - minimum_ground)
            maximum_tree_float = max(maximum_tree_float, floating)
            violations += int(floating > tree_limit + 1.0e-6)
    return tree_objects, cluster_trees, cluster_bushes, violations, maximum_tree_float, maximum_bush_float

def _place_cluster_at(
    *,
    variant: ForestClusterVariant,
    elevations: Sequence[float],
    raster: OsmRaster,
    road_corridors: Sequence[object],
    spec: object,
    x: float,
    z: float,
    heading: float,
    require_forest: bool,
    minimum_forest_fraction: float,
    maximum_relief: float,
    maximum_burial: float,
    maximum_float: float,
    clearance: float,
    avoid_roads: bool = True,
) -> tuple[str, float, float, float, float, str, float, float, float] | None:
    world_size = float(getattr(spec, "world_size"))
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))
    world_name = str(getattr(spec, "name"))
    margin = max(0.0, float(getattr(spec, "forest_cluster_footprint_margin", 0.75)))
    if not (0.0 <= x < world_size and 0.0 <= z < world_size):
        return None

    gradient_x, gradient_z = _local_terrain_gradient(elevations, cells, cell_size, x, z)
    heading, grade = _cluster_heading_and_grade(
        gradient_x, gradient_z, heading % 360.0, variant.slope_axis
    )
    polygon = _oriented_rectangle(
        x, z, variant.width_m, variant.length_m, heading, margin=margin
    )
    if not all(0.0 <= px < world_size and 0.0 <= pz < world_size for px, pz in polygon):
        return None
    if (
        bool(getattr(spec, "forest_low_anchor", False))
        and world_size >= 200.0
        and variant.category in {"border", "interior", "undergrowth"}
    ):
        proxy_edge_guard = max(8.0, min(18.0, cell_size * 0.55))
        if not all(
            proxy_edge_guard <= px <= world_size - proxy_edge_guard
            and proxy_edge_guard <= pz <= world_size - proxy_edge_guard
            for px, pz in polygon
        ):
            return None
    minimum, maximum = _polygon_elevation_extrema(elevations, cells, cell_size, polygon)
    relief = maximum - minimum
    if relief > min(max(0.0, maximum_relief), variant.maximum_relief_m):
        return None
    if avoid_roads and forest_block_intersects_road_corridors(
        road_corridors, x, z, block_size=max(variant.width_m, variant.length_m) + 2.0 * margin
    ):
        return None
    supports = _cluster_supports(
        variant,
        grade=grade,
        heading=heading,
        x=x,
        z=z,
        elevations=elevations,
        raster=raster,
        cells=cells,
        cell_size=cell_size,
        world_size=world_size,
        require_forest=require_forest,
        minimum_forest_fraction=minimum_forest_fraction,
    )
    if supports is None:
        return None
    if variant.category in {"border", "undergrowth", "ditch", "rural"}:
        # These generated models contain separate tree, bush, grass, or reed
        # proxies.  Use the highest triangle-aware proxy support so none of the
        # proxy origins is buried.  If that would exceed the existing floating
        # tolerance, reject the complete cluster instead of sinking it.
        fitted = _non_buried_vegetation_fit(
            supports,
            clearance=clearance,
            maximum_float=maximum_float,
        )
        if fitted is None:
            return None
        anchor, floating = fitted
        burial = 0.0
    else:
        fitted = _terrain_fit_anchor(
            supports,
            clearance=clearance,
            maximum_burial=maximum_burial,
            maximum_float=maximum_float,
        )
        if fitted is None:
            return None
        anchor, burial, floating = fitted

    # A single fitted plane is only acceptable when every tree proxy remains
    # close to the rendered terrain. Bushes/grass can tolerate a larger gap.
    tree_float, bush_float, tree_count, bush_count = _cluster_proxy_floats(
        variant, grade=grade, heading=heading, x=x, y=anchor, z=z,
        elevations=elevations, cells=cells, cell_size=cell_size,
    )
    tree_limit = min(
        max(0.0, maximum_float),
        max(0.0, float(getattr(spec, "forest_cluster_tree_maximum_float", 0.20))),
    )
    bush_limit = min(
        max(0.0, maximum_float),
        max(0.0, float(getattr(spec, "forest_cluster_bush_maximum_float", 0.60))),
    )
    if (tree_count and tree_float > tree_limit + 1.0e-9) or (
        bush_count and bush_float > bush_limit + 1.0e-9
    ):
        return None
    floating = max(floating, tree_float, bush_float)
    return (
        cluster_model_path(world_name, variant.name, grade),
        x,
        anchor,
        z,
        heading,
        variant.name,
        relief,
        burial,
        floating,
    )


def _forest_cluster_candidate_centres(
    seed: str, column: int, row: int, x: float, z: float, radius: float
) -> tuple[PointXZ, ...]:
    if radius <= 0.0:
        return ((x, z),)
    digest = hashlib.blake2s(
        f"{seed}:forest-cluster-centres:{column}:{row}".encode("utf-8"), digest_size=8
    ).digest()
    angle = int.from_bytes(digest[:2], "little") / 65535.0 * math.tau
    ring = (
        (0.0, 0.0),
        (radius, 0.0),
        (-radius, 0.0),
        (0.0, radius),
        (0.0, -radius),
        (radius * 0.70710678, radius * 0.70710678),
        (-radius * 0.70710678, radius * 0.70710678),
        (radius * 0.70710678, -radius * 0.70710678),
        (-radius * 0.70710678, -radius * 0.70710678),
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = [
        (x + ox * cosine + oz * sine, z - ox * sine + oz * cosine)
        for ox, oz in ring
    ]
    # Keep the centre first. The remaining candidates are deterministically
    # permuted so adjacent cells do not all lean toward the same compass point.
    tail = rotated[1:]
    shift = int.from_bytes(digest[2:4], "little") % len(tail)
    return (rotated[0], *tail[shift:], *tail[:shift])


def _forest_cluster_placement(
    *,
    elevations: Sequence[float],
    raster: OsmRaster,
    road_corridors: Sequence[object],
    spec: object,
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    search_radius_override: float | None = None,
) -> tuple[str, float, float, float, float, str, float, float, float] | None:
    """Find one reusable small forest cluster for a rejected forest patch."""

    search_radius = (
        max(0.0, float(search_radius_override))
        if search_radius_override is not None
        else max(0.0, float(getattr(spec, "forest_cluster_search_radius", 10.0)))
    )
    maximum_relief = max(0.0, float(getattr(spec, "forest_cluster_maximum_relief", 36.0)))
    maximum_burial = max(0.0, float(getattr(spec, "forest_cluster_maximum_burial", 1.25)))
    maximum_float = max(0.0, float(getattr(spec, "forest_cluster_maximum_float", 1.25)))
    clearance = float(getattr(spec, "forest_ground_clearance", 0.15))
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))

    digest = hashlib.blake2s(
        f"{seed}:forest-cluster-order:{column}:{row}".encode("utf-8"), digest_size=4
    ).digest()
    variants = sorted(FOREST_CLUSTER_VARIANTS, key=lambda variant: (-variant.area_m2, variant.name))
    rotation = int.from_bytes(digest[:2], "little") % len(variants)
    variants = variants[rotation:] + variants[:rotation]

    for candidate_x, candidate_z in _forest_cluster_candidate_centres(
        seed, column, row, x, z, search_radius
    ):
        gradient_x, gradient_z = _local_terrain_gradient(
            elevations, cells, cell_size, candidate_x, candidate_z
        )
        gradient = math.hypot(gradient_x, gradient_z)
        uphill_x, uphill_z = (0.0, 1.0) if gradient <= 1e-8 else (gradient_x / gradient, gradient_z / gradient)
        for variant in variants:
            if variant.slope_axis == "width":
                heading = math.degrees(math.atan2(-uphill_z, uphill_x))
            else:
                heading = math.degrees(math.atan2(uphill_x, uphill_z))
            jitter_seed = hashlib.blake2s(
                f"{seed}:forest-cluster-heading:{column}:{row}:{variant.name}".encode("utf-8"),
                digest_size=2,
            ).digest()
            heading = (
                heading + (int.from_bytes(jitter_seed, "little") / 65535.0 - 0.5) * 8.0
            ) % 360.0
            placed = _place_cluster_at(
                variant=variant,
                elevations=elevations,
                raster=raster,
                road_corridors=road_corridors,
                spec=spec,
                x=candidate_x,
                z=candidate_z,
                heading=heading,
                require_forest=True,
                minimum_forest_fraction=1.0,
                maximum_relief=maximum_relief,
                maximum_burial=maximum_burial,
                maximum_float=maximum_float,
                clearance=clearance,
            )
            if placed is not None:
                return placed
    return None


def _forest_polygon_replacement_clusters(
    *,
    elevations: Sequence[float],
    raster: OsmRaster,
    road_corridors: Sequence[object],
    spec: object,
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    spacing: float,
    maximum_clusters: int = 4,
) -> tuple[tuple[str, float, float, float, float, str, float, float, float], ...]:
    """Tile generated clusters across one skipped stock forest footprint.

    A stock square/triangle P3D visually covers a much larger patch than one
    generated proxy cluster.  Replacing the stock model with a single cluster
    therefore leaves most of the former footprint visually empty while the
    coverage bookkeeping can still suppress gap infill.  Use a deterministic
    2x2 pattern of independently terrain-fitted clusters instead.
    """

    maximum_clusters = max(0, int(maximum_clusters))
    if maximum_clusters <= 0:
        return ()
    offset = max(5.0, min(float(spacing) * 0.20, 12.0))
    digest = hashlib.blake2s(
        f"{seed}:forest-polygon-replacement:{column}:{row}".encode("utf-8"),
        digest_size=4,
    ).digest()
    # Rotate the quarter-patch pattern so the generated clusters do not form a
    # conspicuous world-aligned checkerboard on broad forests.
    angle = (int.from_bytes(digest[:2], "little") / 65535.0) * (math.pi * 0.5)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    local_offsets = ((-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset))
    placements: list[tuple[str, float, float, float, float, str, float, float, float]] = []
    centres: list[PointXZ] = []
    for slot, (local_x, local_z) in enumerate(local_offsets):
        if len(placements) >= maximum_clusters:
            break
        candidate_x = x + local_x * cosine + local_z * sine
        candidate_z = z - local_x * sine + local_z * cosine
        placed = _forest_cluster_placement(
            elevations=elevations,
            raster=raster,
            road_corridors=road_corridors,
            spec=spec,
            seed=f"{seed}:polygon-replacement:{slot}",
            column=column,
            row=row,
            x=candidate_x,
            z=candidate_z,
            # Keep each replacement near its quarter of the old rigid footprint
            # instead of letting all four searches drift back toward the centre.
            search_radius_override=min(4.0, max(0.0, float(spacing) * 0.08)),
        )
        if placed is None:
            continue
        placed_x, placed_z = placed[1], placed[3]
        if any((placed_x - px) ** 2 + (placed_z - pz) ** 2 < 7.0 ** 2 for px, pz in centres):
            continue
        placements.append(placed)
        centres.append((placed_x, placed_z))

    # Small maps and forest edges can make the quarter-patch centres invalid.
    # Still try one central cluster before handing the patch to tree fallback.
    if not placements:
        central = _forest_cluster_placement(
            elevations=elevations, raster=raster, road_corridors=road_corridors, spec=spec,
            seed=f"{seed}:polygon-replacement:center", column=column, row=row, x=x, z=z,
        )
        if central is not None:
            placements.append(central)
    return tuple(placements)


def _replacement_cluster_coverage_radius(variant_name: str) -> float:
    """Conservative visual coverage for replacement clusters.

    Do not mark a 20+ metre disk as covered merely because a small generated
    cluster contains two or three proxies.  Under-marking is intentional: the
    later gap-infill pass can add grounded trees between clusters.
    """

    variant = cluster_variant(variant_name)
    return max(5.0, min(9.0, min(variant.width_m, variant.length_m) * 0.38))


def _severe_hill_underbrush_placement(
    *,
    elevations: Sequence[float],
    raster: OsmRaster,
    road_corridors: Sequence[object],
    spec: object,
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
) -> tuple[str, float, float, float, float, str, float, float, float] | None:
    """Find one low, slope-fitted underbrush patch for a rejected rigid forest."""

    search_radius = min(
        8.0,
        max(0.0, float(getattr(spec, "forest_cluster_search_radius", 10.0))),
    )
    maximum_relief = max(
        0.0, float(getattr(spec, "forest_undergrowth_maximum_relief", 20.0))
    )
    maximum_burial = max(
        0.0, float(getattr(spec, "forest_undergrowth_maximum_burial", 0.8))
    )
    maximum_float = max(
        0.0, float(getattr(spec, "forest_undergrowth_maximum_float", 0.8))
    )
    clearance = float(
        getattr(spec, "forest_undergrowth_ground_clearance", 0.03)
    )
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))
    digest = hashlib.blake2s(
        f"{seed}:severe-underbrush-order:{column}:{row}".encode("utf-8"),
        digest_size=4,
    ).digest()
    variants = list(FOREST_UNDERGROWTH_VARIANTS)
    rotation = int.from_bytes(digest[:2], "little") % len(variants)
    variants = variants[rotation:] + variants[:rotation]

    for candidate_x, candidate_z in _forest_cluster_candidate_centres(
        f"{seed}:severe-underbrush", column, row, x, z, search_radius
    ):
        gradient_x, gradient_z = _local_terrain_gradient(
            elevations, cells, cell_size, candidate_x, candidate_z
        )
        gradient = math.hypot(gradient_x, gradient_z)
        uphill_x, uphill_z = (
            (0.0, 1.0)
            if gradient <= 1e-8
            else (gradient_x / gradient, gradient_z / gradient)
        )
        for variant in variants:
            heading = (
                math.degrees(math.atan2(-uphill_z, uphill_x))
                if variant.slope_axis == "width"
                else math.degrees(math.atan2(uphill_x, uphill_z))
            )
            placed = _place_cluster_at(
                variant=variant,
                elevations=elevations,
                raster=raster,
                road_corridors=road_corridors,
                spec=spec,
                x=candidate_x,
                z=candidate_z,
                heading=heading,
                require_forest=True,
                minimum_forest_fraction=0.75,
                maximum_relief=maximum_relief,
                maximum_burial=maximum_burial,
                maximum_float=maximum_float,
                clearance=clearance,
            )
            if placed is not None:
                return placed
    return None


def _polygon_contains_with_holes(
    point: PointXZ, outer: Sequence[PointXZ], holes: Sequence[Sequence[PointXZ]]
) -> bool:
    return _point_in_polygon(point, outer) and not any(
        _point_in_polygon(point, hole) for hole in holes if len(hole) >= 3
    )



def _selected_farmland_fence_field_keys(
    features: Sequence[OsmPolygonFeature], seed: str, field_percent: float
) -> frozenset[str]:
    """Choose a stable random-looking subset of eligible farmland/meadow features."""
    eligible = [
        feature
        for feature in features
        if feature.polygons
        and (
            feature.tags.get("landuse", "").casefold() in RURAL_FENCE_LANDUSES
            or feature.tags.get("natural", "").casefold() in RURAL_FENCE_NATURALS
        )
    ]
    if not eligible:
        return frozenset()
    percent = min(100.0, max(0.0, float(field_percent)))
    if percent <= 0.0:
        return frozenset()
    target = int(math.floor(len(eligible) * percent / 100.0 + 0.5))
    if target == 0:
        target = 1
    ranked = sorted(
        eligible,
        key=lambda feature: (
            hashlib.blake2s(
                f"{seed}:farmland-fence-field:{feature.osm_key}".encode("utf-8"),
                digest_size=8,
            ).digest(),
            feature.osm_key,
        ),
    )
    return frozenset(feature.osm_key for feature in ranked[: min(target, len(ranked))])


def _selected_haybale_field_keys(
    features: Sequence[OsmPolygonFeature], seed: str, field_percent: float
) -> frozenset[str]:
    """Choose a deterministic field-level subset for hay-bale placement.

    Selection is by OSM farmland feature rather than by individual candidate,
    so a field either participates in the hay pass or does not.  The requested
    percentage is converted to a deterministic quota and the quota is filled
    by hashing each feature key with the world seed.
    """
    eligible = [
        feature
        for feature in features
        if feature.tags.get("landuse", "").casefold() == "farmland" and feature.polygons
    ]
    if not eligible:
        return frozenset()
    percent = min(100.0, max(0.0, float(field_percent)))
    if percent <= 0.0:
        return frozenset()
    target = int(math.floor(len(eligible) * percent / 100.0 + 0.5))
    if target == 0:
        target = 1
    target = min(len(eligible), target)
    ranked = sorted(
        eligible,
        key=lambda feature: (
            hashlib.blake2s(
                f"{seed}:haybale-field:{feature.osm_key}".encode("utf-8"), digest_size=8
            ).digest(),
            feature.osm_key,
        ),
    )
    return frozenset(feature.osm_key for feature in ranked[:target])

def _haybale_cluster_members(
    seed: str, key: str, x: float, z: float, heading: float
) -> tuple[tuple[float, float, float], ...]:
    """Return a small deterministic single/pair/group around a farmland anchor.

    Most anchors remain single bales, while a minority form pairs or loose
    3-5 bale groups.  Member offsets are deliberately wider than the model
    footprint so the result reads as farm storage rather than object overlap.
    """
    size_digest = hashlib.blake2s(
        f"{seed}:haybale-group-size:{key}".encode("utf-8"), digest_size=4
    ).digest()
    roll = int.from_bytes(size_digest[:2], "little") / 65536.0
    if roll < HAYBALE_CLUSTER_SINGLE_FRACTION:
        count = 1
    elif roll < HAYBALE_CLUSTER_SINGLE_FRACTION + HAYBALE_CLUSTER_PAIR_FRACTION:
        count = 2
    else:
        count = 3 + (int.from_bytes(size_digest[2:], "little") % 3)

    members: list[tuple[float, float, float]] = [(x, z, heading % 360.0)]
    for member_index in range(1, count):
        digest = hashlib.blake2s(
            f"{seed}:haybale-group-member:{key}:{member_index}".encode("utf-8"),
            digest_size=8,
        ).digest()
        angle = (int.from_bytes(digest[0:2], "little") / 65536.0) * math.tau
        radius_fraction = int.from_bytes(digest[2:4], "little") / 65535.0
        radius = HAYBALE_CLUSTER_RADIUS_MIN + radius_fraction * (
            HAYBALE_CLUSTER_RADIUS_MAX - HAYBALE_CLUSTER_RADIUS_MIN
        )
        member_heading = (int.from_bytes(digest[4:6], "little") / 65536.0) * 360.0
        members.append((x + math.cos(angle) * radius, z + math.sin(angle) * radius, member_heading))
    return tuple(members)


def _forest_border_candidates(
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    spacing: float,
    inset: float,
    seed: str,
) -> tuple[tuple[str, float, float, float], ...]:
    candidates: list[tuple[str, float, float, float]] = []
    spacing = max(4.0, spacing)
    inset = max(0.5, inset)
    for feature in sorted(dataset.forests, key=lambda item: item.osm_key):
        for polygon_index, polygon in enumerate(feature.polygons):
            outer = tuple(projection.to_world(point) for point in polygon.outer)
            if len(outer) >= 2 and outer[0] == outer[-1]:
                outer = outer[:-1]
            holes = []
            for raw_hole in polygon.holes:
                hole = tuple(projection.to_world(point) for point in raw_hole)
                if len(hole) >= 2 and hole[0] == hole[-1]:
                    hole = hole[:-1]
                holes.append(hole)
            if len(outer) < 3:
                continue
            rings = (outer, *holes)
            for ring_index, ring in enumerate(rings):
                if len(ring) < 3:
                    continue
                for edge_index, (start, end) in enumerate(zip(ring, ring[1:] + ring[:1])):
                    dx = end[0] - start[0]
                    dz = end[1] - start[1]
                    length = math.hypot(dx, dz)
                    if length < max(5.0, spacing * 0.35):
                        continue
                    tangent_x, tangent_z = dx / length, dz / length
                    count = max(1, int(round(length / spacing)))
                    normal_options = ((-tangent_z, tangent_x), (tangent_z, -tangent_x))
                    for sample_index in range(count):
                        distance = (sample_index + 0.5) * length / count
                        boundary_x = start[0] + tangent_x * distance
                        boundary_z = start[1] + tangent_z * distance
                        inward = None
                        for normal_x, normal_z in normal_options:
                            probe = (boundary_x + normal_x * inset, boundary_z + normal_z * inset)
                            if _polygon_contains_with_holes(probe, outer, holes):
                                inward = (normal_x, normal_z)
                                break
                        if inward is None:
                            continue
                        centre_x = boundary_x + inward[0] * inset
                        centre_z = boundary_z + inward[1] * inset
                        heading = math.degrees(math.atan2(dx, dz)) % 360.0
                        key = (
                            f"{feature.osm_key}:{polygon_index}:{ring_index}:"
                            f"{edge_index}:{sample_index}"
                        )
                        candidates.append((key, centre_x, centre_z, heading))
    # Stable feature ordering plus a seeded tie key avoids dependence on the
    # source parser's incidental polygon order.
    return tuple(sorted(
        candidates,
        key=lambda item: (
            hashlib.blake2s(f"{seed}:forest-border:{item[0]}".encode("utf-8"), digest_size=8).digest(),
            item[0],
        ),
    ))


def _polyline_samples(
    points: Sequence[PointXZ], *, spacing: float, endpoint_trim: float
) -> tuple[tuple[float, float, float], ...]:
    if len(points) < 2:
        return ()
    segments: list[tuple[PointXZ, PointXZ, float]] = []
    total = 0.0
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length > 1e-6:
            segments.append((start, end, length))
            total += length
    trim = max(0.0, endpoint_trim)
    usable = total - 2.0 * trim
    if usable <= 1.0 or not segments:
        return ()
    count = max(1, int(round(usable / max(4.0, spacing))))
    distances = tuple(trim + (index + 0.5) * usable / count for index in range(count))
    result: list[tuple[float, float, float]] = []
    segment_index = 0
    segment_start_distance = 0.0
    for distance in distances:
        while (
            segment_index < len(segments) - 1
            and distance > segment_start_distance + segments[segment_index][2]
        ):
            segment_start_distance += segments[segment_index][2]
            segment_index += 1
        start, end, length = segments[segment_index]
        local = max(0.0, min(1.0, (distance - segment_start_distance) / length))
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        result.append((
            start[0] + dx * local,
            start[1] + dz * local,
            math.degrees(math.atan2(dx, dz)) % 360.0,
        ))
    return tuple(result)


def _ditch_grass_candidates(
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    spacing: float,
    endpoint_trim: float,
    seed: str,
) -> tuple[tuple[str, float, float, float], ...]:
    candidates: list[tuple[str, float, float, float]] = []
    for feature in sorted(dataset.watercourses, key=lambda item: item.osm_key):
        if feature.tags.get("waterway") != "ditch" or feature.tags.get("tunnel") == "yes":
            continue
        points = tuple(projection.to_world(point) for point in feature.points)
        for index, (x, z, heading) in enumerate(
            _polyline_samples(points, spacing=spacing, endpoint_trim=endpoint_trim)
        ):
            candidates.append((f"{feature.osm_key}:{index}", x, z, heading))
    return tuple(sorted(
        candidates,
        key=lambda item: (
            hashlib.blake2s(f"{seed}:ditch-grass:{item[0]}".encode("utf-8"), digest_size=8).digest(),
            item[0],
        ),
    ))


def _square_elevation_extrema(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float, size: float
) -> tuple[float, float]:
    half = size * 0.5
    minimum_x = max(0.0, x - half)
    maximum_x = min(cells * cell_size - 0.001, x + half)
    minimum_z = max(0.0, z - half)
    maximum_z = min(cells * cell_size - 0.001, z + half)
    polygon = (
        (minimum_x, minimum_z),
        (maximum_x, minimum_z),
        (maximum_x, maximum_z),
        (minimum_x, maximum_z),
    )
    return _polygon_elevation_extrema(elevations, cells, cell_size, polygon)


def _oriented_footprint_elevation_samples(
    elevations: Sequence[float], cells: int, cell_size: float,
    x: float, z: float, width: float, length: float, heading: float,
) -> tuple[float, ...]:
    polygon = _oriented_rectangle(x, z, width, length, heading)
    return _polygon_ground_elevation_samples(
        elevations, cells, cell_size, polygon
    )


def _maximum_square_elevation(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float, size: float
) -> float:
    return _square_elevation_extrema(elevations, cells, cell_size, x, z, size)[1]


def _mask_at(mask: Sequence[bool], cells: int, world_size: float, x: float, z: float) -> bool:
    if not (0 <= x < world_size and 0 <= z < world_size):
        return False
    cell_x = min(cells - 1, int(x / world_size * cells))
    cell_z = min(cells - 1, int(z / world_size * cells))
    return mask[cell_z * cells + cell_x]


def _polygon_overlaps_mask(
    mask: Sequence[bool], cells: int, world_size: float, polygon: Sequence[PointXZ]
) -> bool:
    if len(polygon) < 3:
        return any(_mask_at(mask, cells, world_size, x, z) for x, z in polygon)
    area, centre_x, centre_z = _polygon_area_centroid(polygon)
    samples = [*polygon, (centre_x, centre_z)]
    samples.extend(
        (
            (start[0] + end[0]) * 0.5,
            (start[1] + end[1]) * 0.5,
        )
        for start, end in zip(polygon, polygon[1:] + polygon[:1])
    )
    if area > 0.0:
        min_x = min(point[0] for point in polygon)
        max_x = max(point[0] for point in polygon)
        min_z = min(point[1] for point in polygon)
        max_z = max(point[1] for point in polygon)
        step = max(4.0, world_size / max(1, cells) * 0.75)
        x = min_x + step * 0.5
        while x < max_x:
            z = min_z + step * 0.5
            while z < max_z:
                if _point_in_polygon((x, z), polygon):
                    samples.append((x, z))
                z += step
            x += step
    return any(_mask_at(mask, cells, world_size, sample_x, sample_z) for sample_x, sample_z in samples)


def _polygon_fully_covered_by_mask(
    mask: Sequence[bool], cells: int, world_size: float, polygon: Sequence[PointXZ]
) -> bool:
    """Return true only when every raster cell touched by a polygon is masked."""

    if len(polygon) < 3:
        return bool(polygon) and all(
            _mask_at(mask, cells, world_size, x, z) for x, z in polygon
        )
    cell_size = world_size / max(1, cells)
    minimum_x = max(0.0, min(point[0] for point in polygon))
    maximum_x = min(world_size, max(point[0] for point in polygon))
    minimum_z = max(0.0, min(point[1] for point in polygon))
    maximum_z = min(world_size, max(point[1] for point in polygon))
    col0 = max(0, min(cells - 1, int(minimum_x // cell_size)))
    col1 = max(0, min(cells - 1, int(max(0.0, maximum_x - 1e-9) // cell_size)))
    row0 = max(0, min(cells - 1, int(minimum_z // cell_size)))
    row1 = max(0, min(cells - 1, int(max(0.0, maximum_z - 1e-9) // cell_size)))
    touched = False
    for row in range(row0, row1 + 1):
        z0 = row * cell_size
        z1 = min(world_size, (row + 1) * cell_size)
        for col in range(col0, col1 + 1):
            x0 = col * cell_size
            x1 = min(world_size, (col + 1) * cell_size)
            cell_polygon = ((x0, z0), (x1, z0), (x1, z1), (x0, z1))
            if not _polygons_intersect(polygon, cell_polygon):
                continue
            touched = True
            if not mask[row * cells + col]:
                return False
    return touched


def _translate_polygon(polygon: Sequence[PointXZ], dx: float, dz: float) -> tuple[PointXZ, ...]:
    return tuple((x + dx, z + dz) for x, z in polygon)


def _polygon_maximum_span(polygon: Sequence[PointXZ]) -> float:
    if not polygon:
        return 0.0
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_z = min(point[1] for point in polygon)
    max_z = max(point[1] for point in polygon)
    return max(max_x - min_x, max_z - min_z)


def _expand_polygon_from_centroid(
    polygon: Sequence[PointXZ], margin: float
) -> tuple[PointXZ, ...]:
    """Expand a convex support polygon conservatively around its centroid."""

    if len(polygon) < 3 or margin <= 0.0:
        return tuple(polygon)
    centre_x = sum(point[0] for point in polygon) / len(polygon)
    centre_z = sum(point[1] for point in polygon) / len(polygon)
    expanded: list[PointXZ] = []
    for x, z in polygon:
        dx, dz = x - centre_x, z - centre_z
        distance = math.hypot(dx, dz)
        if distance <= 1e-9:
            expanded.append((x, z))
            continue
        scale = (distance + margin) / distance
        expanded.append((centre_x + dx * scale, centre_z + dz * scale))
    return tuple(expanded)


def _polygon_area_centroid(points: Sequence[PointXZ]) -> tuple[float, float, float]:
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    area_twice = 0.0
    cx = 0.0
    cz = 0.0
    for (x0, z0), (x1, z1) in zip(points, points[1:] + points[:1]):
        cross = x0 * z1 - x1 * z0
        area_twice += cross
        cx += (x0 + x1) * cross
        cz += (z0 + z1) * cross
    if abs(area_twice) < 1e-9:
        return 0.0, sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)
    area = abs(area_twice) / 2.0
    return area, cx / (3.0 * area_twice), cz / (3.0 * area_twice)


def _longest_edge_heading(points: Sequence[PointXZ]) -> float:
    best_length = -1.0
    best_heading = 0.0
    for start, end in zip(points, points[1:] + points[:1]):
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = dx * dx + dz * dz
        if length > best_length:
            best_length = length
            best_heading = math.degrees(math.atan2(dx, dz)) % 360.0
    return best_heading


def _building_family(tags: Mapping[str, str]) -> str:
    amenity = tags.get("amenity", "").casefold()
    building = tags.get("building", "").casefold()
    if is_actual_church(tags):
        return "church"
    if amenity == "place_of_worship":
        return "urban"
    if amenity == "school" or building in {"school", "kindergarten", "college", "university"}:
        return "school"
    if amenity == "social_facility" or bool(tags.get("social_facility")):
        # Social-care buildings can be mapped as building=yes, warehouse, or
        # other generic/legacy values. Their amenity semantics outrank the rural
        # footprint heuristics so they never become barns or warehouses.
        return "urban"
    if tags.get("shop") or building in {"retail", "supermarket", "kiosk"}:
        return "shop"
    if building in {"industrial", "warehouse", "hangar", "factory", "manufacture"}:
        return "industrial"
    if building in {
        "barn", "farm_auxiliary", "agricultural", "cowshed", "stable",
        "sty", "shed", "greenhouse", "storage",
    }:
        return "agricultural"
    if building in {
        "apartments", "commercial", "office", "hotel", "hospital",
        "civic", "public", "government",
    }:
        return "urban"
    return "residential"


def _building_model(spec: OsmSpec, tags: Mapping[str, str]) -> str:
    building = tags.get("building", "")
    if tags.get("amenity", "").casefold() == "social_facility" or bool(tags.get("social_facility")):
        return spec.urban_building_model
    if building in {"industrial", "warehouse", "hangar", "factory"}:
        return spec.industrial_building_model
    if building in {"apartments", "commercial", "office", "retail", "hotel", "hospital"}:
        return spec.urban_building_model
    return spec.generic_building_model


def _building_placement_priority(tags: Mapping[str, str]) -> int:
    """Keep rare semantic buildings ahead of ordinary footprint budgets."""

    amenity = tags.get("amenity", "").casefold()
    building = tags.get("building", "").casefold()
    if is_actual_church(tags):
        return 0
    if amenity == "school" or building in {"school", "kindergarten", "college", "university"}:
        return 1
    if amenity == "social_facility" or bool(tags.get("social_facility")):
        return 2
    if tags.get("shop") or building in {"retail", "supermarket", "kiosk"}:
        return 3
    return 4


_HILLSIDE_TREE_OFFSETS: tuple[tuple[float, float], ...] = (
    (-0.31, -0.29), (0.02, -0.34), (0.31, -0.21),
    (-0.15, -0.11), (0.17, -0.03), (-0.35, 0.12),
    (-0.04, 0.18), (0.30, 0.24), (-0.23, 0.34),
    (0.10, 0.36), (0.36, 0.05), (0.00, 0.00),
)


def _roadside_vegetation_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    block_size: float,
    *,
    label: str,
    minimum_spacing: float,
    candidate_count: int,
) -> Iterable[tuple[float, float, float, int]]:
    """Yield deterministic blue-noise-like candidates inside one forest block.

    Candidate points come from hashes rather than a visible lattice. A greedy
    minimum-distance pass removes close pairs, leaving irregular gaps and small
    clusters without allowing tree trunks to pile on top of one another.
    """

    root = hashlib.blake2s(
        f"{seed}:forest-roadside:{label}:{column}:{row}".encode("utf-8"),
        digest_size=16,
    ).digest()
    half_span = block_size * 0.46
    raw: list[tuple[int, float, float, float, int]] = []
    for index in range(max(1, int(candidate_count))):
        digest = hashlib.blake2s(root + index.to_bytes(4, "little"), digest_size=16).digest()
        priority = int.from_bytes(digest[:4], "little")
        unit_x = int.from_bytes(digest[4:8], "little") / 0xFFFFFFFF
        unit_z = int.from_bytes(digest[8:12], "little") / 0xFFFFFFFF
        candidate_x = x + (unit_x * 2.0 - 1.0) * half_span
        candidate_z = z + (unit_z * 2.0 - 1.0) * half_span
        heading = float(int.from_bytes(digest[12:14], "little") % 360)
        variant = int.from_bytes(digest[14:], "little")
        raw.append((priority, candidate_x, candidate_z, heading, variant))
    raw.sort(key=lambda item: item[0])

    accepted: list[tuple[float, float, float, int]] = []
    minimum_distance = max(0.0, float(minimum_spacing))
    minimum_distance2 = minimum_distance * minimum_distance
    if minimum_distance <= 0.0:
        for _priority, candidate_x, candidate_z, heading, variant in raw:
            yield candidate_x, candidate_z, heading, variant
        return

    # The old greedy blue-noise pass compared every new candidate with every
    # previously accepted point. Road-cut forest blocks can run this hundreds
    # of times per block, making the spacing filter quadratic. Bucket accepted
    # points by the minimum spacing: any conflicting point must live in the
    # candidate's own bucket or one of its eight neighbours. The greedy order
    # and therefore the generated vegetation are unchanged.
    inverse_spacing = 1.0 / minimum_distance
    # Candidates are bounded to this one forest block, so a compact local grid
    # is faster than a dictionary of global bucket coordinates. It also avoids
    # nine hash lookups per raw candidate while preserving the exact greedy
    # acceptance order.
    span = half_span * 2.0
    bucket_columns = max(1, int(math.floor(span * inverse_spacing)) + 1)
    accepted_buckets: list[list[int]] = [
        [] for _ in range(bucket_columns * bucket_columns)
    ]
    origin_x = x - half_span
    origin_z = z - half_span
    for _priority, candidate_x, candidate_z, heading, variant in raw:
        bucket_x = min(
            bucket_columns - 1,
            max(0, int((candidate_x - origin_x) * inverse_spacing)),
        )
        bucket_z = min(
            bucket_columns - 1,
            max(0, int((candidate_z - origin_z) * inverse_spacing)),
        )
        too_close = False
        neighbour_x0 = max(0, bucket_x - 1)
        neighbour_x1 = min(bucket_columns - 1, bucket_x + 1)
        neighbour_z0 = max(0, bucket_z - 1)
        neighbour_z1 = min(bucket_columns - 1, bucket_z + 1)
        for neighbour_z in range(neighbour_z0, neighbour_z1 + 1):
            bucket_offset = neighbour_z * bucket_columns
            for neighbour_x in range(neighbour_x0, neighbour_x1 + 1):
                for accepted_index in accepted_buckets[bucket_offset + neighbour_x]:
                    other_x, other_z, _other_heading, _other_variant = accepted[accepted_index]
                    if (
                        (candidate_x - other_x) ** 2 + (candidate_z - other_z) ** 2
                        < minimum_distance2
                    ):
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            continue
        accepted_index = len(accepted)
        accepted_candidate = (candidate_x, candidate_z, heading, variant)
        accepted.append(accepted_candidate)
        accepted_buckets[bucket_z * bucket_columns + bucket_x].append(accepted_index)
        # Yield immediately. Callers stop once a road-cut block has enough
        # successfully grounded vegetation, so there is no reason to spacing-
        # filter the tail of all 384/192 raw candidates first. Future candidates
        # cannot affect whether an earlier greedy candidate was accepted.
        yield accepted_candidate


def _roadside_tree_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    block_size: float,
) -> Iterable[tuple[float, float, float, int]]:
    return _roadside_vegetation_candidates(
        seed, column, row, x, z, block_size,
        label="tree", minimum_spacing=3.75, candidate_count=384,
    )


def _roadside_bush_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    block_size: float,
) -> Iterable[tuple[float, float, float, int]]:
    return _roadside_vegetation_candidates(
        seed, column, row, x, z, block_size,
        label="bush", minimum_spacing=2.25, candidate_count=192,
    )


def _dense_hillside_tree_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    block_size: float,
) -> Iterable[tuple[float, float, float]]:
    """Yield a dense deterministic pool for rejected hill-forest patches.

    Rigid square/triangle forests and grouped proxy clusters are intentionally
    rejected when they cannot hug the final RVW4 terrain safely. Refill those
    patches with individually grounded trees instead of leaving a conspicuous
    bald square. Candidates use the same fast deterministic blue-noise sampler
    as road-cut vegetation, but stay inset from the 50 m block edge so adjacent
    rejected blocks do not form double-density seams. The caller still performs
    the final per-tree terrain/road/water/building checks before placement.
    """

    candidate_span = max(8.0, float(block_size) * 0.90)
    minimum_spacing = max(4.5, min(7.0, float(block_size) * 0.12))
    for candidate_x, candidate_z, heading, _variant in _roadside_vegetation_candidates(
        seed,
        column,
        row,
        x,
        z,
        candidate_span,
        label="hillside-dense",
        minimum_spacing=minimum_spacing,
        candidate_count=256,
    ):
        yield candidate_x, candidate_z, heading


def _hillside_tree_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    block_size: float,
) -> tuple[tuple[float, float, float], ...]:
    """Return stable, well-spaced tree candidates inside one rejected forest block.

    Candidate locations come from a fixed blue-noise-like pattern. The world seed
    and grid address only select its order and cardinal orientation, avoiding the
    process-dependent behavior of Python's random module.
    """

    root = hashlib.blake2s(
        f"{seed}:forest-hillside:{column}:{row}".encode("utf-8"), digest_size=16
    ).digest()
    quarter_turns = root[0] % 4
    ranked: list[tuple[int, float, float, float]] = []
    for index, (offset_x, offset_z) in enumerate(_HILLSIDE_TREE_OFFSETS):
        for _ in range(quarter_turns):
            offset_x, offset_z = -offset_z, offset_x
        digest = hashlib.blake2s(
            root + index.to_bytes(2, "little"), digest_size=8
        ).digest()
        priority = int.from_bytes(digest[:4], "little")
        heading = float(int.from_bytes(digest[4:], "little") % 360)
        ranked.append((priority, x + offset_x * block_size, z + offset_z * block_size, heading))
    ranked.sort(key=lambda item: item[0])
    return tuple((candidate_x, candidate_z, heading) for _, candidate_x, candidate_z, heading in ranked)



_ROCKY_FOREST_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.00, 0.00), (-0.31, -0.18), (0.27, 0.20), (-0.12, 0.34),
    (0.34, -0.09), (-0.37, 0.10), (0.12, -0.35), (0.29, 0.35),
)


_FOREST_SINGLE_TREE_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.00, 0.00), (-0.33, -0.17), (0.29, 0.20), (-0.15, 0.31),
    (0.36, -0.08), (-0.28, 0.27), (0.14, -0.34), (0.11, 0.37),
)


def _geographic_lattice_identity(
    projection: BboxProjection,
    x: float,
    z: float,
    spacing: float,
) -> tuple[int, int]:
    """Return a stable Earth-anchored identity for a local metric grid point."""

    latitude, longitude = projection.to_latlon((x, z))
    row = int(round(EARTH_RADIUS_METRES * math.radians(latitude) / spacing - 0.5))
    band_latitude = math.degrees((row + 0.5) * spacing / EARTH_RADIUS_METRES)
    cosine = max(1.0e-9, math.cos(math.radians(band_latitude)))
    column = int(round(
        EARTH_RADIUS_METRES * math.radians(longitude) * cosine / spacing - 0.5
    ))
    return column, row


def _geographic_forest_single_tree_cells(
    projection: BboxProjection,
    spacing: float,
) -> Iterable[tuple[int, int, float, float, float, float]]:
    """Yield globally anchored 45 m-class cells intersecting this world.

    Each row is a fixed north/south metric band. Longitude spacing is evaluated
    at that band's centre latitude so neighbouring candidates remain about
    ``spacing`` metres apart while retaining identical identities and latitude/
    longitude positions in overlapping source bounds.
    """

    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("geographic tree spacing must be positive and finite")
    south_units = EARTH_RADIUS_METRES * math.radians(projection.south) / spacing
    north_units = EARTH_RADIUS_METRES * math.radians(projection.north) / spacing
    first_row = math.ceil(south_units - 0.5 - 1.0e-9)
    last_row = math.floor(north_units - 0.5 + 1.0e-9)
    for row in range(first_row, last_row + 1):
        latitude = math.degrees((row + 0.5) * spacing / EARTH_RADIUS_METRES)
        cosine = max(1.0e-9, math.cos(math.radians(latitude)))
        west_units = EARTH_RADIUS_METRES * math.radians(projection.west) * cosine / spacing
        east_units = EARTH_RADIUS_METRES * math.radians(projection.east) * cosine / spacing
        first_column = math.ceil(west_units - 0.5 - 1.0e-9)
        last_column = math.floor(east_units - 0.5 + 1.0e-9)
        for column in range(first_column, last_column + 1):
            longitude = math.degrees(
                (column + 0.5) * spacing / (EARTH_RADIUS_METRES * cosine)
            )
            x, z = projection.to_world((latitude, longitude))
            yield column, row, latitude, longitude, x, z


def _forest_single_tree_candidates(
    seed: str,
    projection: BboxProjection,
    column: int,
    row: int,
    latitude: float,
    longitude: float,
    spacing: float,
) -> tuple[tuple[float, float, float], ...]:
    """Return deterministic Earth-anchored candidates inside one forest cell."""

    root = hashlib.blake2s(
        f"{seed}:forest-single:{column}:{row}".encode("utf-8"), digest_size=16
    ).digest()
    angle = int.from_bytes(root[:2], "little") / 65535.0 * math.tau
    cosine = math.cos(angle)
    sine = math.sin(angle)
    ranked: list[tuple[int, float, float, float]] = []
    spread = spacing * 0.38
    for index, (offset_x, offset_z) in enumerate(_FOREST_SINGLE_TREE_OFFSETS):
        digest = hashlib.blake2s(root + index.to_bytes(2, "little"), digest_size=8).digest()
        priority = int.from_bytes(digest[:4], "little")
        rotated_x = offset_x * cosine + offset_z * sine
        rotated_z = -offset_x * sine + offset_z * cosine
        heading = float(int.from_bytes(digest[4:], "little") % 360)
        candidate_latitude = latitude + math.degrees(
            rotated_z * spread / EARTH_RADIUS_METRES
        )
        candidate_cosine = max(1.0e-9, math.cos(math.radians(candidate_latitude)))
        candidate_longitude = longitude + math.degrees(
            rotated_x * spread / (EARTH_RADIUS_METRES * candidate_cosine)
        )
        candidate_x, candidate_z = projection.to_world(
            (candidate_latitude, candidate_longitude)
        )
        ranked.append((priority, candidate_x, candidate_z, heading))
    ranked.sort(key=lambda item: item[0])
    return tuple((px, pz, heading) for _, px, pz, heading in ranked)


def _forest_single_tree_rank(seed: str, column: int, row: int) -> int:
    """Rank one global candidate consistently across overlapping worlds."""

    return int.from_bytes(
        hashlib.blake2s(
            f"{seed}:forest-single-rank:{column}:{row}".encode("utf-8"),
            digest_size=8,
        ).digest(),
        "little",
    )


def _rocky_forest_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    count: int,
    spread: float,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return deterministic x/z/heading/size candidates for one rocky patch."""

    if count <= 0:
        return ()
    root = hashlib.blake2s(
        f"{seed}:rocky-forest:{column}:{row}".encode("utf-8"), digest_size=16
    ).digest()
    angle = int.from_bytes(root[:2], "little") / 65535.0 * math.tau
    cosine = math.cos(angle)
    sine = math.sin(angle)

    def candidate(index: int, offset_x: float, offset_z: float) -> tuple[int, float, float, float, float]:
        digest = hashlib.blake2s(root + index.to_bytes(2, "little"), digest_size=8).digest()
        priority = int.from_bytes(digest[:4], "little")
        rotated_x = offset_x * cosine + offset_z * sine
        rotated_z = -offset_x * sine + offset_z * cosine
        heading = float(int.from_bytes(digest[4:6], "little") % 360)
        size = 4.5 + (int.from_bytes(digest[6:], "little") / 65535.0) * 4.0
        return priority, x + rotated_x * spread, z + rotated_z * spread, heading, size

    centre = candidate(0, *_ROCKY_FOREST_OFFSETS[0])
    tail = [candidate(index, *offset) for index, offset in enumerate(_ROCKY_FOREST_OFFSETS[1:], start=1)]
    tail.sort(key=lambda item: item[0])
    selected = [centre, *tail[: max(0, count - 1)]]
    return tuple((px, pz, heading, size) for _, px, pz, heading, size in selected)

def _line_chunks(points: Sequence[PointXZ], target_length: float, *, endpoint_trim: float = 0.0) -> tuple[tuple[float, float, float, float, float, float, float, float], ...]:
    """Return deterministic fitted line chunks as x, z, heading, length, x0, z0, x1/z1 flattened."""
    chunks: list[tuple[float, float, float, float, float, float, float, float]] = []
    target = max(1.0, float(target_length))
    trim = max(0.0, float(endpoint_trim))
    for a, b in zip(points, points[1:]):
        dx, dz = b[0] - a[0], b[1] - a[1]
        total = math.hypot(dx, dz)
        if total <= max(1.0, 2.0 * trim):
            continue
        ux, uz = dx / total, dz / total
        usable = total - 2.0 * trim
        count = max(1, int(math.ceil(usable / target)))
        length = usable / count
        heading = math.degrees(math.atan2(dx, dz)) % 360.0
        for index in range(count):
            d0 = trim + index * length
            d1 = trim + (index + 1) * length
            x0, z0 = a[0] + ux * d0, a[1] + uz * d0
            x1, z1 = a[0] + ux * d1, a[1] + uz * d1
            chunks.append(((x0 + x1) * 0.5, (z0 + z1) * 0.5, heading, length, x0, z0, x1, z1))
    return tuple(chunks)


def _bridge_module_chunks(
    points: Sequence[PointXZ],
    module_length: float = NOGOVA_BRIDGE_MODULE_LENGTH_METRES,
) -> tuple[tuple[float, float, float, float, float, float, float, float], ...]:
    """Fit fixed bridge modules across a complete polyline by arc length.

    A short span receives one centred module whose ends extend into both banks.
    Longer spans receive ``ceil(span / module_length)`` modules; their centres
    are evenly distributed, so any unavoidable excess length becomes uniform
    overlap instead of a gap or a stretched stock model. Intermediate OSM nodes
    do not create extra modules.
    """

    target = max(1.0, float(module_length))
    segments: list[tuple[PointXZ, PointXZ, float]] = []
    total_length = 0.0
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= 1.0e-6:
            continue
        segments.append((start, end, length))
        total_length += length
    if not segments:
        return ()

    def point_at(distance: float) -> PointXZ:
        remaining = min(total_length, max(0.0, distance))
        for start, end, length in segments:
            if remaining <= length:
                fraction = remaining / length
                return (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            remaining -= length
        return segments[-1][1]

    # Geographic projection can leave an exact 30 m multiple a few billionths
    # over the boundary.  Tolerate that noise so a nominal 60 m span stays two
    # modules rather than acquiring a third, heavily overlapping one.
    rounding_tolerance = max(1.0e-6, target * 1.0e-9)
    count = max(1, int(math.ceil((total_length - rounding_tolerance) / target)))
    coverage_length = total_length / count
    chunks: list[tuple[float, float, float, float, float, float, float, float]] = []
    for index in range(count):
        start = point_at(index * coverage_length)
        end = point_at((index + 1) * coverage_length)
        dx, dz = end[0] - start[0], end[1] - start[1]
        chord_length = math.hypot(dx, dz)
        if chord_length <= 1.0e-6:
            continue
        heading = math.degrees(math.atan2(dx, dz)) % 360.0
        chunks.append((
            (start[0] + end[0]) * 0.5,
            (start[1] + end[1]) * 0.5,
            heading,
            coverage_length,
            start[0],
            start[1],
            end[0],
            end[1],
        ))
    return tuple(chunks)




def _connected_bridge_approach_path(
    feature: OsmLineFeature,
    dataset: OsmDataset,
    projection: BboxProjection,
    endpoint: PointXZ,
    inward: PointXZ,
    spec,
    maximum_probe: float,
) -> tuple[PointXZ, ...]:
    """Follow connected non-bridge road geometry away from one bridge end.

    The literal bridge way frequently begins at the beach, while the ordinary
    road bends uphill before reaching a stable crest. Extrapolating the bridge
    tangent misses that bend. Follow connected OSM road ways instead, preferring
    the same ref/name and the straightest continuation.
    """

    dx = endpoint[0] - inward[0]
    dz = endpoint[1] - inward[1]
    length = max(1.0e-9, math.hypot(dx, dz))
    desired = (dx / length, dz / length)
    current = endpoint
    path: list[PointXZ] = [endpoint]
    used = {feature.osm_key}
    travelled = 0.0
    tolerance = max(2.0, min(8.0, float(spec.cell_size) * 0.20))
    source_ref = str(feature.tags.get("ref", "")).strip().casefold()
    source_name = str(feature.tags.get("name", "")).strip().casefold()
    source_highway = str(feature.tags.get("highway", "")).strip().casefold()

    for _ in range(10):
        candidates: list[tuple[float, OsmLineFeature, tuple[PointXZ, ...]]] = []
        for road in dataset.roads:
            if road.osm_key in used or len(road.points) < 2:
                continue
            bridge_value = str(road.tags.get("bridge", "")).strip().casefold()
            if bridge_value not in {"", "no", "false", "0", "none"}:
                continue
            if str(road.tags.get("special", "")).strip().casefold() == "bridge":
                continue
            projected = tuple(projection.to_world(point) for point in road.points)
            orientations: list[tuple[PointXZ, ...]] = []
            if math.dist(projected[0], current) <= tolerance:
                orientations.append(projected)
            if math.dist(projected[-1], current) <= tolerance:
                orientations.append(tuple(reversed(projected)))
            for oriented in orientations:
                first_index = next(
                    (i for i in range(1, len(oriented)) if math.dist(oriented[0], oriented[i]) > 0.25),
                    None,
                )
                if first_index is None:
                    continue
                vx = oriented[first_index][0] - current[0]
                vz = oriented[first_index][1] - current[1]
                span = max(1.0e-9, math.hypot(vx, vz))
                alignment = (vx / span) * desired[0] + (vz / span) * desired[1]
                if alignment < -0.30:
                    continue
                score = alignment * 3.0 - math.dist(oriented[0], current) * 0.10
                road_ref = str(road.tags.get("ref", "")).strip().casefold()
                road_name = str(road.tags.get("name", "")).strip().casefold()
                road_highway = str(road.tags.get("highway", "")).strip().casefold()
                if source_ref and road_ref == source_ref:
                    score += 1.25
                if source_name and road_name == source_name:
                    score += 0.75
                if source_highway and road_highway == source_highway:
                    score += 0.30
                candidates.append((score, road, oriented))
        if not candidates:
            break
        _score, road, oriented = max(candidates, key=lambda item: (item[0], item[1].osm_key))
        used.add(road.osm_key)
        oriented = (current,) + oriented[1:]
        appended = False
        for target in oriented[1:]:
            segment = math.dist(current, target)
            if segment <= 1.0e-6:
                continue
            if travelled + segment >= maximum_probe:
                fraction = (maximum_probe - travelled) / segment
                clipped = (
                    current[0] + (target[0] - current[0]) * fraction,
                    current[1] + (target[1] - current[1]) * fraction,
                )
                path.append(clipped)
                return tuple(path)
            path.append(target)
            travelled += segment
            previous = current
            current = target
            vx = current[0] - previous[0]
            vz = current[1] - previous[1]
            span = max(1.0e-9, math.hypot(vx, vz))
            desired = (vx / span, vz / span)
            appended = True
        if not appended or travelled >= maximum_probe - 1.0e-6:
            break
    return tuple(path)


def _approach_plateau_samples(
    path: Sequence[PointXZ],
    elevations: Sequence[float],
    spec,
    step: float,
    raster: OsmRaster | None = None,
) -> tuple[PointXZ, ...]:
    """Return the road path from the bridge end to the first stable upper level."""

    if len(path) < 2:
        return tuple(path)
    cumulative = [0.0]
    for first, second in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    total = cumulative[-1]
    if total <= 1.0:
        return tuple(path)

    def point_at(distance: float) -> PointXZ:
        distance = max(0.0, min(total, distance))
        for index in range(len(path) - 1):
            if cumulative[index + 1] + 1.0e-9 < distance:
                continue
            span = max(1.0e-9, cumulative[index + 1] - cumulative[index])
            fraction = (distance - cumulative[index]) / span
            return (
                path[index][0] + (path[index + 1][0] - path[index][0]) * fraction,
                path[index][1] + (path[index + 1][1] - path[index][1]) * fraction,
            )
        return path[-1]

    distances = [0.0]
    cursor = step
    while cursor < total - 1.0e-6:
        distances.append(cursor)
        cursor += step
    distances.append(total)
    samples = []
    for distance in distances:
        x, z = point_at(distance)
        height = _sample_elevation(elevations, spec.cells, spec.cell_size, x, z)
        samples.append((distance, x, z, height))
    start_height = samples[0][3]
    running_high = start_height
    chosen = None
    for index in range(1, max(1, len(samples) - 2)):
        height = samples[index][3]
        running_high = max(running_high, height)
        if height < start_height + 0.60:
            continue
        forward = samples[index:min(len(samples), index + 3)]
        if len(forward) < 2:
            continue
        local_relief = max(item[3] for item in forward) - min(item[3] for item in forward)
        if height >= running_high - 0.30 and local_relief <= 0.55:
            chosen = index
            break
    if chosen is None:
        heights = sorted((item[3] for item in samples), reverse=True)
        target = heights[min(1, len(heights) - 1)] - 0.35
        chosen = next((i for i, item in enumerate(samples) if item[3] >= target), len(samples) - 1)

    # Do not let the first visible bridge module begin exactly on the shoreline
    # or at the first crest sample. Keep following the ordinary road a short
    # distance onto stable dry land. This is intentionally measured along the
    # road path, so a curved approach stays curved instead of projecting a fake
    # straight extension across the hillside.
    chosen_distance = samples[chosen][0]
    required_distance = chosen_distance + PROCEDURAL_BRIDGE_LAND_INSET_METRES
    if raster is not None:
        last_wet_distance = 0.0
        for distance, x, z, _height in samples:
            if _mask_at(raster.water, spec.cells, spec.world_size, x, z):
                last_wet_distance = max(last_wet_distance, distance)
        required_distance = max(
            required_distance,
            last_wet_distance + PROCEDURAL_BRIDGE_LAND_INSET_METRES,
        )
    chosen = next(
        (i for i, item in enumerate(samples) if item[0] >= required_distance - 1e-6),
        len(samples) - 1,
    )
    return tuple((item[1], item[2]) for item in samples[:chosen + 1])

def _extend_procedural_bridge_to_approach_plateaus(
    points: Sequence[PointXZ],
    elevations: Sequence[float],
    spec,
    module_length: float,
    *,
    feature: OsmLineFeature | None = None,
    dataset: OsmDataset | None = None,
    projection: BboxProjection | None = None,
    raster: OsmRaster | None = None,
) -> tuple[PointXZ, ...]:
    """Extend a generated bridge uphill to the road level before a beach descent.

    OSM bridge ways often begin only at the shoreline/wet span. A road can start
    descending toward that shoreline tens of metres earlier, which leaves the
    procedural rail/deck beginning at the bottom of the hill even when its deck
    elevation was sampled from the upper approach. Probe outward along the end
    tangents and prepend/append the first stable high approach point.
    """

    if len(points) < 2:
        return tuple(points)
    # The visible shoreline descent can begin well before the OSM bridge tag.
    # Probe far enough inland to reach the actual road crest/plateau instead of
    # stopping halfway up the beach slope.
    maximum_probe = max(
        90.0,
        min(150.0, float(spec.cell_size) * 5.0, max(90.0, float(module_length) * 6.0)),
    )
    step = max(3.0, min(6.0, float(spec.cell_size) * 0.22))

    def plateau_point(
        origin: PointXZ, inward: PointXZ, *, direction: float
    ) -> PointXZ:
        dx = inward[0] - origin[0]
        dz = inward[1] - origin[1]
        length = max(1.0e-9, math.hypot(dx, dz))
        # ``inward`` points into the OSM bridge. Reverse that tangent at the
        # start and keep it at the end to walk out onto the ordinary road.
        ux = dx / length * direction
        uz = dz / length * direction
        distances = [0.0]
        cursor = step
        while cursor < maximum_probe - 1.0e-6:
            distances.append(cursor)
            cursor += step
        distances.append(maximum_probe)
        samples: list[tuple[float, float, float, float]] = []
        for distance in distances:
            x = min(float(spec.world_size), max(0.0, origin[0] + ux * distance))
            z = min(float(spec.world_size), max(0.0, origin[1] + uz * distance))
            samples.append((distance, x, z, _sample_elevation(
                elevations, spec.cells, spec.cell_size, x, z
            )))
        heights = sorted((sample[3] for sample in samples), reverse=True)
        # Reject one isolated DEM spike but otherwise target the upper road.
        plateau_height = heights[1] if len(heights) > 1 else heights[0]
        tolerance = 0.40
        selected_index = None
        for index, sample in enumerate(samples):
            if sample[3] < plateau_height - tolerance:
                continue
            # Prefer a point where the next sample no longer climbs materially.
            next_height = samples[min(len(samples) - 1, index + 1)][3]
            if next_height <= sample[3] + 0.55:
                selected_index = index
                break
        if selected_index is None:
            selected_index = max(
                range(len(samples)),
                key=lambda index: (samples[index][3], -samples[index][0]),
            )
        required_distance = samples[selected_index][0] + PROCEDURAL_BRIDGE_LAND_INSET_METRES
        selected_index = next(
            (index for index, sample in enumerate(samples) if sample[0] >= required_distance - 1e-6),
            len(samples) - 1,
        )
        return (samples[selected_index][1], samples[selected_index][2])

    connected_start: tuple[PointXZ, ...] = ()
    connected_end: tuple[PointXZ, ...] = ()
    if feature is not None and dataset is not None and projection is not None:
        raw_start = _connected_bridge_approach_path(
            feature, dataset, projection, points[0], points[1], spec, maximum_probe
        )
        raw_end = _connected_bridge_approach_path(
            feature, dataset, projection, points[-1], points[-2], spec, maximum_probe
        )
        connected_start = _approach_plateau_samples(raw_start, elevations, spec, step, raster)
        connected_end = _approach_plateau_samples(raw_end, elevations, spec, step, raster)

    if len(connected_start) < 2 or math.dist(connected_start[-1], points[0]) <= 1.0:
        start = plateau_point(points[0], points[1], direction=-1.0)
        connected_start = (points[0], start) if math.dist(start, points[0]) > 1.0 else (points[0],)
    if len(connected_end) < 2 or math.dist(connected_end[-1], points[-1]) <= 1.0:
        end = plateau_point(points[-1], points[-2], direction=-1.0)
        connected_end = (points[-1], end) if math.dist(end, points[-1]) > 1.0 else (points[-1],)

    # Start-side approach samples run from bridge -> crest, so reverse them to
    # make the generated bridge begin at the actual road crest. The end side is
    # already ordered bridge -> crest. Preserve the bridge's own internal nodes.
    extended = list(reversed(connected_start))
    extended.extend(points[1:-1])
    extended.extend(connected_end)
    deduplicated: list[PointXZ] = []
    for point in extended:
        if not deduplicated or math.dist(point, deduplicated[-1]) > 0.05:
            deduplicated.append(point)
    return tuple(deduplicated)

def _numeric_tag(tags: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(str(tags.get(key, "")).replace(",", "."))
    except ValueError:
        return default
    return value if math.isfinite(value) else default


BRIDGE_DITCH_CROSSING_TOLERANCE_METRES = 2.0
BRIDGE_WATER_EPSILON_METRES = 0.05


def road_span_has_in_game_water(
    points: Sequence[PointXZ],
    elevations: Sequence[float],
    *,
    cells: int,
    cell_size: float,
    sea_level: float,
    width: float = 6.0,
) -> bool:
    """Return whether CWA's global water plane is visible below a road span.

    OSM ``bridge=yes`` is not enough on its own. Some datasets retain bridge
    tagging after a stream is culverted, rerouted, seasonal, or simply absent
    from the generated world's water. Since CWA renders water wherever terrain
    falls below the global sea plane, sample the *final terrain* under the road
    centre and both carriageway edges. A bridge is only useful when at least one
    of those samples is genuinely underwater in game.
    """

    if len(points) < 2 or not elevations:
        return False
    half_width = max(1.5, float(width) * 0.5)
    spacing = max(2.0, min(8.0, float(cell_size) * 0.30, half_width))
    threshold = float(sea_level) - BRIDGE_WATER_EPSILON_METRES
    for start, end in zip(points, points[1:]):
        dx = float(end[0]) - float(start[0])
        dz = float(end[1]) - float(start[1])
        length = math.hypot(dx, dz)
        if length <= 0.05:
            continue
        nx, nz = -dz / length, dx / length
        count = max(1, int(math.ceil(length / spacing)))
        for index in range(count + 1):
            fraction = index / count
            x = float(start[0]) + dx * fraction
            z = float(start[1]) + dz * fraction
            for offset in (0.0, half_width, -half_width):
                height = _sample_elevation(
                    elevations, cells, cell_size,
                    x + nx * offset, z + nz * offset,
                )
                if height < threshold:
                    return True
    return False


def road_bridge_crosses_ditch_only(
    feature: OsmLineFeature,
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    tolerance_metres: float = BRIDGE_DITCH_CROSSING_TOLERANCE_METRES,
) -> bool:
    """Return whether an explicit OSM road bridge only crosses mapped ditches.

    Tiny ditch crossings are better represented as an ordinary road on OFP/CWA
    terrain.  Creating a 30 m Nogova bridge for them is wildly out of scale and
    can replace a perfectly usable road segment with a bridge that spans half a
    farm.  The test is deliberately spatial rather than tag-only because the
    road way normally carries ``bridge=yes`` while the crossed line carries
    ``waterway=ditch``.
    """

    bridge = str(feature.tags.get("bridge", "")).strip().casefold()
    explicit = (
        bridge not in {"", "no", "false", "0", "none"}
        or str(feature.tags.get("man_made", "")).strip().casefold() == "bridge"
        or str(feature.tags.get("special", "")).strip().casefold() == "bridge"
    )
    if not explicit or len(feature.points) < 2:
        return False

    road_points = tuple(projection.to_world(point) for point in feature.points)
    road_segments = tuple(zip(road_points, road_points[1:]))
    tolerance_squared = max(0.0, float(tolerance_metres)) ** 2
    crossing_kinds: list[str] = []
    for watercourse in dataset.watercourses:
        if len(watercourse.points) < 2:
            continue
        water_points = tuple(projection.to_world(point) for point in watercourse.points)
        crossed = any(
            _segment_distance_squared(a0, a1, b0, b1) <= tolerance_squared
            for a0, a1 in road_segments
            for b0, b1 in zip(water_points, water_points[1:])
        )
        if crossed:
            crossing_kinds.append(
                str(watercourse.tags.get("waterway", "stream")).strip().casefold()
            )
    return bool(crossing_kinds) and all(kind == "ditch" for kind in crossing_kinds)


def _road_needs_bridge_deck(
    feature: OsmLineFeature,
    chunks: Sequence[tuple[float, float, float, float, float, float, float, float]],
    raster: OsmRaster,
    spec,
    elevations: Sequence[float],
) -> bool:
    bridge = str(feature.tags.get("bridge", "")).strip().casefold()
    explicit = (
        bridge not in {"", "no", "false", "0", "none"}
        or str(feature.tags.get("man_made", "")).strip().casefold() == "bridge"
        or str(feature.tags.get("special", "")).strip().casefold() == "bridge"
    )
    elevated = _numeric_tag(feature.tags, "layer", 0.0) > 0.0
    if not explicit and not elevated:
        return False
    if not chunks:
        return False

    points: list[PointXZ] = [(chunks[0][4], chunks[0][5])]
    points.extend((chunk[6], chunk[7]) for chunk in chunks)
    width = max(6.0, road_width_metres(feature.tags))
    return road_span_has_in_game_water(
        points, elevations,
        cells=spec.cells, cell_size=spec.cell_size,
        sea_level=spec.sea_level, width=width,
    )


def _line_footprint_polygon(
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    *,
    half_width: float,
    end_margin: float = 0.0,
) -> tuple[PointXZ, ...]:
    """Return a rectangular footprint around a fitted linear object.

    Stock hedge geometry is substantially wider than its mapped centreline.
    Grounding only from the two line endpoints can therefore bury leaves and
    trunks where terrain rises beside or between those endpoints.
    """

    dx = x1 - x0
    dz = z1 - z0
    length = math.hypot(dx, dz)
    if length <= 1e-9:
        radius = max(0.05, float(half_width))
        return (
            (x0 - radius, z0 - radius),
            (x0 + radius, z0 - radius),
            (x0 + radius, z0 + radius),
            (x0 - radius, z0 + radius),
        )
    unit_x = dx / length
    unit_z = dz / length
    margin = max(0.0, float(end_margin))
    start_x = x0 - unit_x * margin
    start_z = z0 - unit_z * margin
    end_x = x1 + unit_x * margin
    end_z = z1 + unit_z * margin
    width = max(0.05, float(half_width))
    normal_x = unit_z * width
    normal_z = -unit_x * width
    return (
        (start_x + normal_x, start_z + normal_z),
        (end_x + normal_x, end_z + normal_z),
        (end_x - normal_x, end_z - normal_z),
        (start_x - normal_x, start_z - normal_z),
    )


def _hedge_anchor_height(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    *,
    model_path: str | None = None,
    half_width: float = HEDGE_FOOTPRINT_HALF_WIDTH_METRES,
    end_margin: float = HEDGE_FOOTPRINT_END_MARGIN_METRES,
    clearance: float = HEDGE_GROUND_CLEARANCE_METRES,
) -> float:
    footprint = _line_footprint_polygon(
        x0,
        z0,
        x1,
        z1,
        half_width=half_width,
        end_margin=end_margin,
    )
    origin_offset = (
        HEDGE_MODEL_VERTICAL_ORIGIN_OFFSETS.get(
            model_path.casefold(), HEDGE_DEFAULT_VERTICAL_ORIGIN_OFFSET_METRES
        )
        if model_path
        else 0.0
    )
    return (
        _maximum_polygon_elevation(elevations, cells, cell_size, footprint)
        + float(clearance)
        + origin_offset
    )


def _infrastructure_anchor(
    elevations: Sequence[float], cells: int, cell_size: float,
    x0: float, z0: float, x1: float, z1: float, *, clearance: float = 0.04,
) -> tuple[float, float]:
    h0 = _sample_elevation(elevations, cells, cell_size, x0, z0)
    h1 = _sample_elevation(elevations, cells, cell_size, x1, z1)
    horizontal = max(0.01, math.hypot(x1 - x0, z1 - z0))
    pitch = math.degrees(math.atan2(h1 - h0, horizontal))
    return (h0 + h1) * 0.5 + clearance, pitch


def _distributed_grid_indices(columns: int, seed: str, label: str) -> Iterable[int]:
    """Yield every square-grid index once in a deterministic, well-spread order.

    A row-major cap empties one corner of a large world before it reaches the
    rest. A coprime modular walk preserves determinism while distributing any
    capped subset over the complete map.
    """
    columns = max(1, int(columns))
    total = columns * columns
    if total == 1:
        yield 0
        return
    digest = hashlib.blake2s(f"{seed}:{label}:{columns}".encode("utf-8"), digest_size=8).digest()
    start = int.from_bytes(digest[:4], "little") % total
    step = int.from_bytes(digest[4:], "little") % total
    step = step or 1
    while math.gcd(step, total) != 1:
        step = (step + 1) % total or 1
    for offset in range(total):
        yield (start + offset * step) % total


def _polygon_grid_candidates(
    points: Sequence[PointXZ],
    spacing: float,
    seed: str,
    *,
    jitter_fraction: float = 0.0,
    heading_jitter_degrees: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    if len(points) < 3:
        return ()
    minimum_x = min(p[0] for p in points)
    maximum_x = max(p[0] for p in points)
    minimum_z = min(p[1] for p in points)
    maximum_z = max(p[1] for p in points)
    step = max(8.0, spacing)
    digest = hashlib.blake2s(seed.encode('utf-8'), digest_size=8).digest()
    offset_x = (int.from_bytes(digest[:4], 'little') / 2**32) * step
    offset_z = (int.from_bytes(digest[4:], 'little') / 2**32) * step
    base_heading = _longest_edge_heading(points)
    jitter_fraction = max(0.0, min(0.95, jitter_fraction))
    heading_jitter_degrees = max(0.0, heading_jitter_degrees)
    candidates: list[tuple[float, float, float]] = []
    x = math.floor((minimum_x - offset_x) / step) * step + offset_x
    column = 0
    while x <= maximum_x:
        z = math.floor((minimum_z - offset_z) / step) * step + offset_z
        row = 0
        while z <= maximum_z:
            candidate_x = x
            candidate_z = z
            heading = base_heading
            if jitter_fraction > 0.0 or heading_jitter_degrees > 0.0:
                cell_digest = hashlib.blake2s(
                    f"{seed}:{column}:{row}".encode("utf-8"), digest_size=8
                ).digest()
                candidate_x += (
                    int.from_bytes(cell_digest[0:2], "little") / 65535.0 - 0.5
                ) * step * jitter_fraction
                candidate_z += (
                    int.from_bytes(cell_digest[2:4], "little") / 65535.0 - 0.5
                ) * step * jitter_fraction
                heading = (
                    base_heading
                    + (int.from_bytes(cell_digest[4:6], "little") / 65535.0 - 0.5)
                    * heading_jitter_degrees
                ) % 360.0
            if _point_in_polygon((candidate_x, candidate_z), points):
                candidates.append((candidate_x, candidate_z, heading))
            z += step
            row += 1
        x += step
        column += 1
    return tuple(candidates)


def _nudge_building_footprint_off_roads(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: OsmSpec,
    x: float,
    z: float,
    heading: float,
    support_polygon: Sequence[PointXZ],
    road_corridors: Sequence[RoadCorridor] | None = None,
) -> tuple[float, float, tuple[PointXZ, ...], bool]:
    if road_corridors is None:
        road_corridors = project_road_corridors(dataset, projection, spec)
    road_clearance = 0.75
    if not polygon_intersects_road_corridors(
        road_corridors, support_polygon, clearance=road_clearance
    ):
        return x, z, tuple(support_polygon), False
    road_point = nearest_road_point(dataset, projection, x, z)
    if road_point is not None:
        away_x, away_z = x - road_point[0], z - road_point[1]
        length = math.hypot(away_x, away_z)
    else:
        away_x = away_z = length = 0.0
    if length <= 1e-6:
        road_heading = nearest_road_heading(dataset, projection, x, z)
        away_x, away_z = _heading_right_vector(
            road_heading if road_point is not None else heading
        )
    else:
        away_x, away_z = away_x / length, away_z / length

    span = _polygon_maximum_span(support_polygon)
    step = max(1.0, LARGE_BUILDING_ROAD_NUDGE_DISTANCE_METRES)
    maximum_distance = max(12.0, span * 0.65 + 8.0)
    distance = step
    while distance <= maximum_distance + 1e-9:
        for sign in (1.0, -1.0):
            shifted_x = x + away_x * distance * sign
            shifted_z = z + away_z * distance * sign
            shifted_polygon = _translate_polygon(
                support_polygon, shifted_x - x, shifted_z - z
            )
            if not all(
                0.0 <= px < spec.world_size and 0.0 <= pz < spec.world_size
                for px, pz in shifted_polygon
            ):
                continue
            if _polygon_overlaps_mask(
                raster.water, spec.cells, spec.world_size, shifted_polygon
            ):
                continue
            if polygon_intersects_road_corridors(
                road_corridors, shifted_polygon, clearance=road_clearance
            ):
                continue
            return shifted_x, shifted_z, shifted_polygon, True
        distance += step
    return x, z, tuple(support_polygon), False



def _polygons_intersect(a: Sequence[PointXZ], b: Sequence[PointXZ]) -> bool:
    """Return whether two simple polygons overlap or touch."""
    if len(a) < 3 or len(b) < 3:
        return False
    if any(_point_in_polygon(point, b) for point in a):
        return True
    if any(_point_in_polygon(point, a) for point in b):
        return True
    return any(
        _segments_intersect(a0, a1, b0, b1)
        for a0, a1 in zip(a, a[1:] + a[:1])
        for b0, b1 in zip(b, b[1:] + b[:1])
    )


def _world_bbox(points: Sequence[PointXZ]) -> tuple[float, float, float, float]:
    return (
        min((x for x, _z in points), default=0.0),
        min((z for _x, z in points), default=0.0),
        max((x for x, _z in points), default=0.0),
        max((z for _x, z in points), default=0.0),
    )


def _bboxes_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _bbox_contains_point(bbox: tuple[float, float, float, float], x: float, z: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= z <= bbox[3]


def _polygon_minimum_distance_squared(
    a: Sequence[PointXZ],
    b: Sequence[PointXZ],
) -> float:
    """Return the minimum horizontal separation between two polygons.

    Zero includes touching/intersecting polygons. This intentionally uses the
    existing lightweight segment geometry instead of requiring Shapely in the
    normal building placement path.
    """
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    if _polygons_intersect(a, b):
        return 0.0
    best = float("inf")
    for point in a:
        for start, end in zip(b, b[1:] + b[:1]):
            best = min(best, _point_to_segment_distance_squared(point, start, end))
    for point in b:
        for start, end in zip(a, a[1:] + a[:1]):
            best = min(best, _point_to_segment_distance_squared(point, start, end))
    return best


def _residential_area_has_mapped_building(
    dataset: OsmDataset,
    projection: BboxProjection,
    outer: Sequence[PointXZ],
    holes: Sequence[Sequence[PointXZ]],
) -> bool:
    """Only infill residential polygons that contain no source-backed building."""
    for feature in dataset.building_points:
        point = projection.to_world(feature.point)
        if _polygon_contains_with_holes(point, outer, holes):
            return True
    for feature in dataset.building_polygons:
        for polygon in feature.polygons:
            projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            if len(projected) < 3:
                continue
            _area, cx, cz = _polygon_area_centroid(projected)
            if _polygon_contains_with_holes((cx, cz), outer, holes):
                return True
            if any(_polygon_contains_with_holes(point, outer, holes) for point in projected):
                return True
    return False


def _mapped_building_near_world_point(
    dataset: OsmDataset,
    projection: BboxProjection,
    x: float,
    z: float,
    radius: float,
    *,
    include_overture: bool = False,
) -> bool:
    radius_squared = radius * radius
    for feature in dataset.building_points:
        if not include_overture and feature.tags.get("source") == "overturemaps":
            continue
        px, pz = projection.to_world(feature.point)
        if (px - x) * (px - x) + (pz - z) * (pz - z) <= radius_squared:
            return True
    for feature in dataset.building_polygons:
        if not include_overture and feature.tags.get("source") == "overturemaps":
            continue
        for polygon in feature.polygons:
            projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            if len(projected) < 3:
                continue
            _area, cx, cz = _polygon_area_centroid(projected)
            if (cx - x) * (cx - x) + (cz - z) * (cz - z) <= radius_squared:
                return True
            if any((px - x) * (px - x) + (pz - z) * (pz - z) <= radius_squared for px, pz in projected):
                return True
    return False


def _overture_road_ending_fallback_areas(
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: OsmSpec,
) -> tuple[tuple[tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]], ...]:
    world_size = float(getattr(spec, "world_size", projection.world_size))
    radius = 80.0
    endpoint_counts: dict[tuple[int, int], int] = {}
    endpoints: list[PointXZ] = []
    for road in dataset.roads:
        if not road_is_supported(road.tags, include_minor=True) or len(road.points) < 2:
            continue
        for point in (road.points[0], road.points[-1]):
            x, z = projection.to_world(point)
            if not (0.0 <= x < world_size and 0.0 <= z < world_size):
                continue
            key = (round(x), round(z))
            endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
            endpoints.append((x, z))

    areas: list[tuple[tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]]] = []
    seen: set[tuple[int, int]] = set()
    for x, z in endpoints:
        key = (round(x), round(z))
        if key in seen or endpoint_counts.get(key, 0) != 1:
            continue
        seen.add(key)
        if _mapped_building_near_world_point(dataset, projection, x, z, radius):
            continue
        min_x = max(0.0, x - radius)
        max_x = min(world_size, x + radius)
        min_z = max(0.0, z - radius)
        max_z = min(world_size, z + radius)
        if max_x - min_x < 24.0 or max_z - min_z < 24.0:
            continue
        areas.append((((min_x, min_z), (max_x, min_z), (max_x, max_z), (min_x, max_z)), ()))
    return tuple(areas)


def _place_inside_residential_area(
    place: OsmPointFeature,
    dataset: OsmDataset,
    projection: BboxProjection,
) -> bool:
    point = projection.to_world(place.point)
    for feature in dataset.urban:
        if feature.tags.get("landuse", "").casefold() != "residential":
            continue
        for polygon in feature.polygons:
            outer = tuple(projection.to_world(p) for p in polygon.outer[:-1])
            holes = tuple(tuple(projection.to_world(p) for p in hole[:-1]) for hole in polygon.holes)
            if len(outer) >= 3 and _polygon_contains_with_holes(point, outer, holes):
                return True
    return False


def _small_settlement_infill_feature(
    place: OsmPointFeature,
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: OsmSpec,
) -> OsmPolygonFeature | None:
    kind = str(place.tags.get("place", "")).strip().casefold()
    radius = SMALL_SETTLEMENT_INFILL_RADIUS_METRES.get(kind)
    if radius is None or _place_inside_residential_area(place, dataset, projection):
        return None
    x, z = projection.to_world(place.point)
    world_size = float(getattr(spec, "world_size", projection.world_size))
    if not (0.0 <= x < world_size and 0.0 <= z < world_size):
        return None
    # The place label can be tens of metres from the actual dwelling roofs.
    # Treat either OSM or accepted Overture buildings as evidence that this
    # settlement already exists before creating a synthetic residential patch.
    guard_radius = SMALL_SETTLEMENT_EXISTING_BUILDING_GUARD_METRES.get(kind, radius)
    if _mapped_building_near_world_point(
        dataset, projection, x, z, guard_radius, include_overture=True
    ):
        return None
    min_x = max(0.0, x - radius)
    max_x = min(world_size, x + radius)
    min_z = max(0.0, z - radius)
    max_z = min(world_size, z + radius)
    if max_x - min_x < 24.0 or max_z - min_z < 24.0:
        return None
    ring = tuple(
        projection.to_latlon(point)
        for point in (
            (min_x, min_z),
            (max_x, min_z),
            (max_x, max_z),
            (min_x, max_z),
            (min_x, min_z),
        )
    )
    tags = {
        "landuse": "residential",
        "place": kind,
        "cwr:synthetic": "small_settlement_infill_area",
    }
    name = place.tags.get("name")
    if name:
        tags["name"] = name
    return OsmPolygonFeature(place.osm_key, tags, (GeoPolygon(ring),))


_OVERTURE_BUILDING_CLASS_TO_OSM: Mapping[str, str] = {
    # Overture classes that already have direct OSM/procedural equivalents.
    # Keep these explicit so semantic barn/warehouse/shed selection wins over
    # footprint-only fallback heuristics whenever Overture actually knows the
    # building purpose.
    "agricultural": "agricultural",
    "barn": "barn",
    "cowshed": "cowshed",
    "farm_auxiliary": "farm_auxiliary",
    "greenhouse": "greenhouse",
    "stable": "stable",
    "sty": "sty",
    "factory": "factory",
    "hangar": "hangar",
    "industrial": "industrial",
    "manufacture": "manufacture",
    "warehouse": "warehouse",
    "carport": "carport",
    "garage": "garage",
    "garages": "garages",
    "outbuilding": "outbuilding",
    "shed": "shed",
    # Preserve common non-rural semantics too instead of degrading a known
    # Overture class into a generic footprint.
    "apartments": "apartments",
    "civic": "civic",
    "commercial": "commercial",
    "detached": "detached",
    "government": "government",
    "hospital": "hospital",
    "hotel": "hotel",
    "house": "house",
    "kindergarten": "kindergarten",
    "kiosk": "kiosk",
    "office": "office",
    "public": "public",
    "residential": "residential",
    "retail": "retail",
    "school": "school",
    "supermarket": "supermarket",
    "university": "university",
    "college": "college",
}

_OVERTURE_BUILDING_SUBTYPE_TO_OSM: Mapping[str, str] = {
    "agricultural": "agricultural",
    "industrial": "industrial",
    "outbuilding": "outbuilding",
    "residential": "residential",
}


def _overture_building_tag(properties: Mapping[str, Any] | None) -> str:
    """Return the best procedural/OSM building value carried by Overture."""

    if properties is None:
        return "yes"
    building_class = str(properties.get("class") or "").strip().casefold()
    if building_class in _OVERTURE_BUILDING_CLASS_TO_OSM:
        return _OVERTURE_BUILDING_CLASS_TO_OSM[building_class]
    subtype = str(properties.get("subtype") or "").strip().casefold()
    if subtype in _OVERTURE_BUILDING_SUBTYPE_TO_OSM:
        return _OVERTURE_BUILDING_SUBTYPE_TO_OSM[subtype]
    # Unknown Overture footprints are intentionally generic, not forced houses.
    # That lets the same tiny-shed and oversized rural barn heuristics used for
    # generic OSM buildings classify fallback footprints by their dimensions.
    return "yes"


def _overture_source_osm_keys(properties: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return CWR OSM keys referenced by Overture source provenance."""

    if properties is None:
        return ()
    raw_sources = properties.get("sources")
    if not isinstance(raw_sources, list):
        return ()
    keys: set[str] = set()
    prefix_names = {"w": "way", "n": "node", "r": "relation"}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            continue
        dataset_name = str(source.get("dataset") or "").strip().casefold().replace(" ", "")
        if dataset_name not in {"openstreetmap", "osm"}:
            continue
        record_id = str(source.get("record_id") or "").strip().casefold()
        if len(record_id) < 2 or record_id[0] not in prefix_names:
            continue
        numeric = record_id[1:].split("@", 1)[0]
        if numeric.isdigit():
            keys.add(f"{prefix_names[record_id[0]]}/{numeric}")
    return tuple(sorted(keys))


def _overture_text_number(
    value: Any, *, integer: bool = False, allow_zero: bool = False
) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if integer:
        rounded = int(round(number))
        return str(rounded) if rounded > 0 else ""
    if number < 0.0 or (number == 0.0 and not allow_zero):
        return ""
    return format(number, ".6g")


def _overture_enrichment_tags(properties: Mapping[str, Any] | None) -> dict[str, str]:
    """Map useful Overture building attributes into OSM-style CWR tags."""

    if properties is None:
        return {}
    tags: dict[str, str] = {}
    source_id = str(properties.get("id") or "").strip()
    if source_id:
        tags["cwr:overture_id"] = source_id
    building_class = str(properties.get("class") or "").strip().casefold()
    subtype = str(properties.get("subtype") or "").strip().casefold()
    if building_class:
        tags["cwr:overture_class"] = building_class
    if subtype:
        tags["cwr:overture_subtype"] = subtype
    osm_keys = _overture_source_osm_keys(properties)
    if osm_keys:
        tags["cwr:overture_osm_keys"] = "|".join(osm_keys)

    height = _overture_text_number(properties.get("height"))
    floors = _overture_text_number(properties.get("num_floors"), integer=True)
    roof_height = _overture_text_number(properties.get("roof_height"))
    roof_direction = _overture_text_number(
        properties.get("roof_direction"), allow_zero=True
    )
    if height:
        tags["height"] = height
    if floors:
        tags["building:levels"] = floors
    if roof_height:
        tags["roof:height"] = roof_height
    if roof_direction:
        tags["roof:direction"] = roof_direction

    roof_shape = str(properties.get("roof_shape") or "").strip().casefold().replace("_", "-")
    roof_orientation = str(properties.get("roof_orientation") or "").strip().casefold()
    facade_material = str(properties.get("facade_material") or "").strip().casefold()
    facade_color = str(properties.get("facade_color") or "").strip().casefold()
    roof_material = str(properties.get("roof_material") or "").strip().casefold()
    roof_color = str(properties.get("roof_color") or "").strip().casefold()
    if roof_shape:
        tags["roof:shape"] = roof_shape
    if roof_orientation:
        tags["roof:orientation"] = roof_orientation
    if facade_material:
        tags["building:material"] = facade_material
    if facade_color:
        tags["building:colour"] = facade_color
    if roof_material:
        tags["roof:material"] = roof_material
    if roof_color:
        tags["roof:colour"] = roof_color

    min_height = _overture_text_number(properties.get("min_height"))
    min_floor = _overture_text_number(properties.get("min_floor"), integer=True)
    if min_height:
        # Keep floating-part data for diagnostics/future part support without
        # pretending the current ground-based building generator can honour it.
        tags["cwr:overture_min_height"] = min_height
    if min_floor:
        tags["cwr:overture_min_floor"] = min_floor
    if properties.get("has_parts") is True:
        tags["cwr:overture_has_parts"] = "yes"
    return tags


def _merge_overture_tags(
    osm_tags: Mapping[str, str],
    overture_tags: Mapping[str, str],
    *,
    match_method: str,
) -> dict[str, str]:
    """Enrich OSM tags without overwriting explicit OSM building knowledge."""

    merged = {str(key): str(value) for key, value in osm_tags.items()}
    overture_building = str(overture_tags.get("building", "")).strip().casefold()
    if merged.get("building", "").strip().casefold() in {"", "yes"} and overture_building not in {"", "yes"}:
        merged["building"] = overture_building

    for key, value in overture_tags.items():
        if key in {"building", "source", "cwr:synthetic"} or not value:
            continue
        if key.startswith("cwr:overture_"):
            merged[key] = value
        elif not str(merged.get(key, "")).strip():
            merged[key] = value
    merged["cwr:overture_match"] = match_method
    return merged


def _load_overture_geojson_features(path: Path) -> list[Mapping[str, Any]]:
    """Load a GeoJSON FeatureCollection without first materializing one giant text string.

    ``json.load`` still parses the document in memory, but unlike ``read_text`` +
    ``json.loads`` it avoids holding an additional copy of a 100-200+ MB source
    string while the decoded feature tree is alive. Geometry conversion is kept
    lazy by the conflation pass below, which is the much larger win on big worlds.
    """

    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Overture buildings GeoJSON {path}: {exc}") from exc
    features = document.get("features") if isinstance(document, Mapping) else None
    if not isinstance(features, list):
        raise ValueError(f"Overture buildings GeoJSON {path} does not contain a FeatureCollection")
    return [feature for feature in features if isinstance(feature, Mapping)]


def _overture_geojson_feature_identity(
    feature: Mapping[str, Any], feature_index: int
) -> tuple[str, str, Mapping[str, Any] | None]:
    raw_properties = feature.get("properties")
    properties = raw_properties if isinstance(raw_properties, Mapping) else None
    source_id = (
        str(properties.get("id"))
        if properties is not None and properties.get("id")
        else str(feature.get("id", feature_index))
    )
    return source_id, f"overture/{source_id}", properties


def _overture_geojson_feature_tags(
    feature: Mapping[str, Any], feature_index: int
) -> tuple[str, dict[str, str], Mapping[str, Any] | None]:
    source_id, osm_key, properties = _overture_geojson_feature_identity(feature, feature_index)
    tags = {
        "building": _overture_building_tag(properties),
        "source": "overturemaps",
        "cwr:synthetic": "overture_building",
        **_overture_enrichment_tags(properties),
    }
    tags.setdefault("cwr:overture_id", source_id)
    return osm_key, tags, properties


def _overture_geojson_feature_polygons(feature: Mapping[str, Any]) -> tuple[GeoPolygon, ...]:
    """Convert one Overture GeoJSON feature geometry only when geometry is needed."""

    def polygon_from_coordinates(coordinates: Any) -> GeoPolygon | None:
        if not isinstance(coordinates, list) or not coordinates:
            return None
        rings: list[tuple[PointLL, ...]] = []
        for raw_ring in coordinates:
            if not isinstance(raw_ring, list):
                continue
            ring: list[PointLL] = []
            for point in raw_ring:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    lon = float(point[0])
                    lat = float(point[1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(lat) and math.isfinite(lon):
                    ring.append((lat, lon))
            if len(ring) >= 3:
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                rings.append(tuple(ring))
        if not rings:
            return None
        return GeoPolygon(rings[0], tuple(rings[1:]))

    geometry = feature.get("geometry")
    if not isinstance(geometry, Mapping):
        return ()
    geometry_type = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates")
    polygons: list[GeoPolygon] = []
    if geometry_type == "Polygon":
        polygon = polygon_from_coordinates(coordinates)
        if polygon is not None:
            polygons.append(polygon)
    elif geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        for raw_polygon in coordinates:
            polygon = polygon_from_coordinates(raw_polygon)
            if polygon is not None:
                polygons.append(polygon)
    return tuple(polygons)


def _overture_geojson_building_feature(
    feature: Mapping[str, Any], feature_index: int
) -> OsmPolygonFeature | None:
    polygons = _overture_geojson_feature_polygons(feature)
    if not polygons:
        return None
    osm_key, tags, _properties = _overture_geojson_feature_tags(feature, feature_index)
    return OsmPolygonFeature(osm_key, tags, polygons)


def _geojson_overture_building_polygons(path: Path) -> tuple[OsmPolygonFeature, ...]:
    """Compatibility helper used by tests/tools that still want a materialized tuple."""

    buildings: list[OsmPolygonFeature] = []
    for feature_index, feature in enumerate(_load_overture_geojson_features(path)):
        building = _overture_geojson_building_feature(feature, feature_index)
        if building is not None:
            buildings.append(building)
    return tuple(sorted(buildings, key=lambda item: item.osm_key))

def augment_dataset_with_overture_buildings(
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: OsmSpec,
    geojson_path: Path,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> OsmDataset:
    """Merge Overture metadata into OSM buildings, then add true gaps.

    Large-world optimization notes:
    - GeoJSON features stay in their raw decoded form until geometry is needed.
    - provenance/source-ID matches never convert coordinate rings at all.
    - one Shapely STRtree replaces repeated map-wide building scans for geometry
      matching, fallback occupancy, near-duplicate suppression and settlement
      guards.
    - Overture records retain the historical sorted-key processing order so the
      optimized path remains deterministic and compatible with earlier output.
    """

    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    try:
        size_mb = geojson_path.stat().st_size / (1024.0 * 1024.0)
    except OSError:
        size_mb = 0.0
    progress(0, f"Parsing Overture GeoJSON ({size_mb:.1f} MiB)")
    overture_features = _load_overture_geojson_features(geojson_path)
    if not overture_features:
        progress(100, "Overture GeoJSON contains no buildings")
        return dataset

    ordered_features = sorted(
        enumerate(overture_features),
        key=lambda item: _overture_geojson_feature_identity(item[1], item[0])[1],
    )
    progress(8, f"Loaded {len(ordered_features):,} Overture building records")

    polygon_buildings = list(dataset.building_polygons)
    point_buildings = list(dataset.building_points)
    polygon_key_index = {feature.osm_key: index for index, feature in enumerate(polygon_buildings)}
    point_key_index = {feature.osm_key: index for index, feature in enumerate(point_buildings)}
    matched_overture: set[int] = set()
    matched_polygon_indices: set[int] = set()
    matched_point_indices: set[int] = set()
    exact_matches = 0
    spatial_matches = 0

    # Provenance is the cheapest and most reliable match. Crucially, inspect it
    # before converting geometry: an Overture record that already says "OSM
    # way/123" does not need thousands of coordinate floats copied into another
    # Python polygon just to enrich way/123's tags.
    progress(10, "Matching Overture provenance to OSM building IDs")
    for overture_index, raw_feature in ordered_features:
        _source_id, _overture_key, properties = _overture_geojson_feature_identity(
            raw_feature, overture_index
        )
        source_keys = _overture_source_osm_keys(properties)
        target_kind = ""
        target_index = -1
        for source_key in source_keys:
            if source_key in polygon_key_index:
                target_kind, target_index = "polygon", polygon_key_index[source_key]
                break
            if source_key in point_key_index:
                target_kind, target_index = "point", point_key_index[source_key]
                break
        if target_index < 0:
            continue
        _osm_key, overture_tags, _properties = _overture_geojson_feature_tags(
            raw_feature, overture_index
        )
        if target_kind == "polygon":
            original = polygon_buildings[target_index]
            polygon_buildings[target_index] = OsmPolygonFeature(
                original.osm_key,
                _merge_overture_tags(original.tags, overture_tags, match_method="source-id"),
                original.polygons,
            )
            matched_polygon_indices.add(target_index)
        else:
            original = point_buildings[target_index]
            point_buildings[target_index] = OsmPointFeature(
                original.osm_key,
                _merge_overture_tags(original.tags, overture_tags, match_method="source-id"),
                original.point,
            )
            matched_point_indices.add(target_index)
        matched_overture.add(overture_index)
        exact_matches += 1
    progress(
        22,
        f"Overture provenance matched {exact_matches:,}/{len(ordered_features):,} records; building spatial index",
    )

    try:
        from shapely.geometry import Point as ShapelyPoint
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
        from shapely.strtree import STRtree
    except ImportError:
        ShapelyPoint = None  # type: ignore[assignment]
        ShapelyPolygon = None  # type: ignore[assignment]
        unary_union = None  # type: ignore[assignment]
        STRtree = None  # type: ignore[assignment]

    def shape_from_polygons(polygons: Sequence[GeoPolygon]):
        if ShapelyPolygon is None or unary_union is None:
            return None
        shapes = []
        for polygon in polygons:
            outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            holes = tuple(
                tuple(projection.to_world(point) for point in hole[:-1])
                for hole in polygon.holes
            )
            if len(outer) < 3:
                continue
            try:
                shape = ShapelyPolygon(outer, [hole for hole in holes if len(hole) >= 3])
                if not shape.is_valid:
                    shape = shape.buffer(0)
                if not shape.is_empty and shape.area > 0.01:
                    shapes.append(shape)
            except Exception:
                continue
        if not shapes:
            return None
        try:
            return shapes[0] if len(shapes) == 1 else unary_union(shapes)
        except Exception:
            return None

    def shape_from_world_polygon(
        outer: Sequence[PointXZ], holes: Sequence[Sequence[PointXZ]] = ()
    ):
        if ShapelyPolygon is None or len(outer) < 3:
            return None
        try:
            shape = ShapelyPolygon(outer, [hole for hole in holes if len(hole) >= 3])
            if not shape.is_valid:
                shape = shape.buffer(0)
            return shape if not shape.is_empty else None
        except Exception:
            return None

    polygon_shapes: list[Any] = [None] * len(polygon_buildings)
    tree_shapes: list[Any] = []
    tree_osm_indices: list[int] = []
    polygon_tree = None
    if STRtree is not None and ShapelyPolygon is not None:
        for osm_index, feature in enumerate(polygon_buildings):
            shape = shape_from_polygons(feature.polygons)
            polygon_shapes[osm_index] = shape
            if shape is not None and not shape.is_empty:
                tree_shapes.append(shape)
                tree_osm_indices.append(osm_index)
        if tree_shapes:
            polygon_tree = STRtree(tree_shapes)

    point_shapes: list[Any] = []
    point_world: list[PointXZ] = []
    point_tree = None
    if STRtree is not None and ShapelyPoint is not None:
        for feature in point_buildings:
            x, z = projection.to_world(feature.point)
            point_world.append((x, z))
            point_shapes.append(ShapelyPoint(x, z))
        if point_shapes:
            point_tree = STRtree(point_shapes)

    def tree_positions(tree: Any, geometry: Any) -> tuple[int, ...]:
        if tree is None or geometry is None:
            return ()
        try:
            result = tree.query(geometry)
        except Exception:
            return ()
        try:
            return tuple(int(value) for value in result)
        except TypeError:
            return ()

    # Remaining records need geometry. STRtree lookup turns the old broad bucket
    # candidate collection into a direct native spatial query, while preserving
    # the exact historical IoU/area/centroid thresholds and sorted processing.
    progress(30, "Geometry-matching Overture records that lack OSM provenance")
    geometry_examined = 0
    if polygon_tree is not None:
        for overture_index, raw_feature in ordered_features:
            if overture_index in matched_overture:
                continue
            overture = _overture_geojson_building_feature(raw_feature, overture_index)
            if overture is None:
                continue
            shape = shape_from_polygons(overture.polygons)
            if shape is None or shape.is_empty or shape.area <= 0.01:
                continue
            geometry_examined += 1
            shape_area = float(shape.area)
            shape_centroid = shape.centroid
            best: tuple[float, float, int] | None = None
            for tree_position in tree_positions(polygon_tree, shape):
                osm_index = tree_osm_indices[tree_position]
                if osm_index in matched_polygon_indices:
                    continue
                osm_shape = polygon_shapes[osm_index]
                if osm_shape is None or osm_shape.is_empty:
                    continue
                try:
                    intersection = float(shape.intersection(osm_shape).area)
                    if intersection <= 0.0:
                        continue
                    osm_area = float(osm_shape.area)
                    union_area = shape_area + osm_area - intersection
                    if union_area <= 0.0:
                        continue
                    iou = intersection / union_area
                    area_ratio = min(shape_area, osm_area) / max(shape_area, osm_area)
                    centroid_distance = float(shape_centroid.distance(osm_shape.centroid))
                except Exception:
                    continue
                accepted_match = (
                    iou >= 0.65
                    or (iou >= 0.50 and area_ratio >= 0.65 and centroid_distance <= 5.0)
                )
                if not accepted_match:
                    continue
                candidate = (iou, area_ratio, osm_index)
                if best is None or candidate > best:
                    best = candidate
            if best is None:
                continue
            osm_index = best[2]
            original = polygon_buildings[osm_index]
            polygon_buildings[osm_index] = OsmPolygonFeature(
                original.osm_key,
                _merge_overture_tags(original.tags, overture.tags, match_method="geometry"),
                original.polygons,
            )
            matched_overture.add(overture_index)
            matched_polygon_indices.add(osm_index)
            spatial_matches += 1
    progress(
        58,
        f"Geometry matching examined {geometry_examined:,} records and matched {spatial_matches:,}",
    )

    merged_dataset = replace(
        dataset,
        building_polygons=tuple(polygon_buildings),
        building_points=tuple(point_buildings),
    )

    # Reuse the same spatial index for fallback-area occupancy. These helpers are
    # intentionally local to the Overture stage so ordinary OSM behavior stays
    # untouched while the six-figure-building case avoids O(N*M) Python scans.
    def area_has_mapped_building(
        outer: Sequence[PointXZ], holes: Sequence[Sequence[PointXZ]]
    ) -> bool:
        area_shape = shape_from_world_polygon(outer, holes)
        if area_shape is None or polygon_tree is None:
            return _residential_area_has_mapped_building(
                merged_dataset, projection, outer, holes
            )
        if point_tree is not None:
            for point_index in tree_positions(point_tree, area_shape):
                try:
                    if area_shape.covers(point_shapes[point_index]):
                        return True
                except Exception:
                    continue
        for tree_position in tree_positions(polygon_tree, area_shape):
            mapped = tree_shapes[tree_position]
            try:
                if area_shape.intersects(mapped):
                    return True
            except Exception:
                continue
        return False

    def mapped_building_near(x: float, z: float, radius: float) -> bool:
        if ShapelyPoint is None or polygon_tree is None:
            return _mapped_building_near_world_point(
                merged_dataset,
                projection,
                x,
                z,
                radius,
                include_overture=True,
            )
        point = ShapelyPoint(x, z)
        try:
            search_shape = point.buffer(radius)
        except Exception:
            return False
        if point_tree is not None:
            for point_index in tree_positions(point_tree, search_shape):
                px, pz = point_world[point_index]
                if (px - x) * (px - x) + (pz - z) * (pz - z) <= radius * radius:
                    return True
        for tree_position in tree_positions(polygon_tree, search_shape):
            mapped = tree_shapes[tree_position]
            try:
                if mapped.distance(point) <= radius:
                    return True
            except Exception:
                continue
        return False

    progress(62, "Finding genuinely empty Overture fallback areas")
    fallback_areas: list[
        tuple[tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...], tuple[float, float, float, float]]
    ] = []
    for feature in sorted(merged_dataset.urban, key=lambda item: item.osm_key):
        if feature.tags.get("landuse", "").casefold() != "residential":
            continue
        for polygon in feature.polygons:
            outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            holes = tuple(
                tuple(projection.to_world(point) for point in hole[:-1])
                for hole in polygon.holes
            )
            if len(outer) >= 3 and not area_has_mapped_building(outer, holes):
                fallback_areas.append((outer, holes, _world_bbox(outer)))

    for place in sorted(merged_dataset.places, key=lambda item: item.osm_key):
        kind = str(place.tags.get("place", "")).strip().casefold()
        radius = SMALL_SETTLEMENT_INFILL_RADIUS_METRES.get(kind)
        if radius is None or _place_inside_residential_area(place, merged_dataset, projection):
            continue
        x, z = projection.to_world(place.point)
        world_size = float(getattr(spec, "world_size", projection.world_size))
        if not (0.0 <= x < world_size and 0.0 <= z < world_size):
            continue
        guard_radius = SMALL_SETTLEMENT_EXISTING_BUILDING_GUARD_METRES.get(kind, radius)
        if mapped_building_near(x, z, guard_radius):
            continue
        min_x = max(0.0, x - radius)
        max_x = min(world_size, x + radius)
        min_z = max(0.0, z - radius)
        max_z = min(world_size, z + radius)
        if max_x - min_x < 24.0 or max_z - min_z < 24.0:
            continue
        outer = (
            (min_x, min_z),
            (max_x, min_z),
            (max_x, max_z),
            (min_x, max_z),
        )
        if not area_has_mapped_building(outer, ()):
            fallback_areas.append((outer, (), _world_bbox(outer)))

    # Road-ending fallback used to call a full building scan for every dangling
    # endpoint. With 100k+ buildings that alone can dominate this stage.
    world_size = float(getattr(spec, "world_size", projection.world_size))
    endpoint_radius = 80.0
    endpoint_counts: dict[tuple[int, int], int] = {}
    endpoints: list[PointXZ] = []
    for road in merged_dataset.roads:
        if not road_is_supported(road.tags, include_minor=True) or len(road.points) < 2:
            continue
        for point_ll in (road.points[0], road.points[-1]):
            x, z = projection.to_world(point_ll)
            if not (0.0 <= x < world_size and 0.0 <= z < world_size):
                continue
            key = (round(x), round(z))
            endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
            endpoints.append((x, z))
    seen_endpoints: set[tuple[int, int]] = set()
    for x, z in endpoints:
        key = (round(x), round(z))
        if key in seen_endpoints or endpoint_counts.get(key, 0) != 1:
            continue
        seen_endpoints.add(key)
        if mapped_building_near(x, z, endpoint_radius):
            continue
        min_x = max(0.0, x - endpoint_radius)
        max_x = min(world_size, x + endpoint_radius)
        min_z = max(0.0, z - endpoint_radius)
        max_z = min(world_size, z + endpoint_radius)
        if max_x - min_x < 24.0 or max_z - min_z < 24.0:
            continue
        outer = (
            (min_x, min_z),
            (max_x, min_z),
            (max_x, max_z),
            (min_x, max_z),
        )
        fallback_areas.append((outer, (), _world_bbox(outer)))

    fallback_shapes: list[Any] = []
    fallback_tree = None
    if STRtree is not None and ShapelyPolygon is not None:
        for outer, holes, _bbox in fallback_areas:
            shape = shape_from_world_polygon(outer, holes)
            if shape is not None:
                fallback_shapes.append(shape)
        if fallback_shapes:
            fallback_tree = STRtree(fallback_shapes)

    def candidate_in_fallback_area(x: float, z: float, bbox: tuple[float, float, float, float]) -> bool:
        if ShapelyPoint is not None and fallback_tree is not None:
            point = ShapelyPoint(x, z)
            for fallback_index in tree_positions(fallback_tree, point):
                try:
                    if fallback_shapes[fallback_index].covers(point):
                        return True
                except Exception:
                    continue
            return False
        return any(
            _bboxes_intersect(bbox, fallback_bbox)
            and _polygon_contains_with_holes((x, z), outer, holes)
            for outer, holes, fallback_bbox in fallback_areas
        )

    def conflicts_with_mapped_building(
        projected: Sequence[PointXZ],
        bbox: tuple[float, float, float, float],
        cx: float,
        cz: float,
        candidate_shape: Any,
    ) -> bool:
        candidate_area, _candidate_cx, _candidate_cz = _polygon_area_centroid(projected)
        candidate_scale = math.sqrt(max(0.01, candidate_area))
        point_duplicate_radius = min(12.0, max(6.0, candidate_scale * 0.70))

        if ShapelyPoint is None or polygon_tree is None or candidate_shape is None:
            # Rare Shapely-unavailable fallback preserves the pre-optimization
            # behavior even though it is intentionally slower.
            mapped_points = tuple(projection.to_world(feature.point) for feature in point_buildings)
            for point in mapped_points:
                if _bbox_contains_point(bbox, point[0], point[1]) and _point_in_polygon(point, projected):
                    return True
                if math.hypot(point[0] - cx, point[1] - cz) <= point_duplicate_radius:
                    return True
            for osm_shape in polygon_shapes:
                if osm_shape is None or osm_shape.is_empty:
                    continue
                try:
                    if candidate_shape is not None and candidate_shape.intersects(osm_shape):
                        return True
                except Exception:
                    continue
            return False

        centroid_point = ShapelyPoint(cx, cz)
        if point_tree is not None:
            try:
                point_search = centroid_point.buffer(point_duplicate_radius)
            except Exception:
                point_search = candidate_shape
            for point_index in tree_positions(point_tree, point_search):
                px, pz = point_world[point_index]
                try:
                    if candidate_shape.covers(point_shapes[point_index]):
                        return True
                except Exception:
                    pass
                if math.hypot(px - cx, pz - cz) <= point_duplicate_radius:
                    return True

        try:
            polygon_search = candidate_shape.buffer(
                OVERTURE_FALLBACK_NEAR_DUPLICATE_MAXIMUM_DISTANCE_METRES
            )
        except Exception:
            polygon_search = candidate_shape
        for tree_position in tree_positions(polygon_tree, polygon_search):
            mapped_shape = tree_shapes[tree_position]
            try:
                if candidate_shape.intersects(mapped_shape):
                    return True
                if mapped_shape.covers(centroid_point):
                    return True
                mapped_area = float(mapped_shape.area)
                if candidate_area <= 0.01 or mapped_area <= 0.01:
                    continue
                area_ratio = min(candidate_area, mapped_area) / max(candidate_area, mapped_area)
                mapped_bbox = tuple(float(value) for value in mapped_shape.bounds)
                candidate_min_span = max(1.0, min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
                mapped_min_span = max(
                    1.0,
                    min(mapped_bbox[2] - mapped_bbox[0], mapped_bbox[3] - mapped_bbox[1]),
                )
                duplicate_radius = min(
                    OVERTURE_FALLBACK_NEAR_DUPLICATE_MAXIMUM_DISTANCE_METRES,
                    max(4.0, min(candidate_min_span, mapped_min_span) * 1.15),
                )
                mapped_centroid = mapped_shape.centroid
                if (
                    area_ratio >= 0.55
                    and math.hypot(float(mapped_centroid.x) - cx, float(mapped_centroid.y) - cz)
                    <= duplicate_radius
                ):
                    return True
            except Exception:
                continue
        return False

    accepted: list[OsmPolygonFeature] = []
    seen_keys = {feature.osm_key for feature in polygon_buildings}
    progress(76, f"Checking unmatched Overture buildings against {len(fallback_areas):,} fallback areas")
    fallback_examined = 0
    if fallback_areas:
        for overture_index, raw_feature in ordered_features:
            if overture_index in matched_overture:
                continue
            feature = _overture_geojson_building_feature(raw_feature, overture_index)
            if feature is None:
                continue
            kept_polygons: list[GeoPolygon] = []
            for polygon in feature.polygons:
                projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
                if len(projected) < 3:
                    continue
                bbox = _world_bbox(projected)
                area, x, z = _polygon_area_centroid(projected)
                if area < getattr(spec, "building_minimum_area", 10.0):
                    continue
                if not candidate_in_fallback_area(x, z, bbox):
                    continue
                fallback_examined += 1
                candidate_shape = shape_from_polygons((polygon,))
                if conflicts_with_mapped_building(projected, bbox, x, z, candidate_shape):
                    continue
                kept_polygons.append(polygon)
            if kept_polygons and feature.osm_key not in seen_keys:
                accepted.append(OsmPolygonFeature(feature.osm_key, feature.tags, tuple(kept_polygons)))
                seen_keys.add(feature.osm_key)

    progress(
        94,
        f"Overture fallback examined {fallback_examined:,} candidates and accepted {len(accepted):,}",
    )
    if exact_matches == 0 and spatial_matches == 0 and not accepted:
        progress(100, "Overture enrichment made no dataset changes")
        return dataset

    final_polygons = tuple(sorted((*polygon_buildings, *accepted), key=lambda item: item.osm_key))
    final_points = tuple(sorted(point_buildings, key=lambda item: item.osm_key))
    overture_fingerprint = cache_key(
        "overture-buildings-conflated-v5",
        {
            "base": dataset.normalized_fingerprint,
            "exact_matches": exact_matches,
            "spatial_matches": spatial_matches,
            "polygon_tags": tuple(
                (feature.osm_key, tuple(sorted(feature.tags.items())))
                for feature in final_polygons
                if feature.tags.get("cwr:overture_match") or feature.tags.get("source") == "overturemaps"
            ),
            "point_tags": tuple(
                (feature.osm_key, tuple(sorted(feature.tags.items())))
                for feature in final_points
                if feature.tags.get("cwr:overture_match")
            ),
        },
    )
    progress(
        100,
        f"Overture enrichment complete: {exact_matches:,} provenance, {spatial_matches:,} geometry, {len(accepted):,} fallback",
    )
    return replace(
        dataset,
        building_polygons=final_polygons,
        building_points=final_points,
        element_count=dataset.element_count + len(accepted),
        normalized_fingerprint=overture_fingerprint,
    )

@dataclass(slots=True)
class _PolygonBucketIndex:
    """Dynamic bbox index for the small polygon collision tests in infill."""

    bucket_size: float
    polygons: list[tuple[PointXZ, ...]]
    buckets: dict[tuple[int, int], set[int]]

    @classmethod
    def from_polygons(
        cls, polygons: Sequence[Sequence[PointXZ]], *, bucket_size: float = 64.0
    ) -> "_PolygonBucketIndex":
        index = cls(max(8.0, float(bucket_size)), [], defaultdict(set))
        for polygon in polygons:
            index.add(polygon)
        return index

    def add(self, polygon: Sequence[PointXZ]) -> int:
        stored = tuple(polygon)
        index = len(self.polygons)
        self.polygons.append(stored)
        if not stored:
            return index
        minimum_x = min(point[0] for point in stored)
        maximum_x = max(point[0] for point in stored)
        minimum_z = min(point[1] for point in stored)
        maximum_z = max(point[1] for point in stored)
        for bz in range(
            math.floor(minimum_z / self.bucket_size),
            math.floor(maximum_z / self.bucket_size) + 1,
        ):
            for bx in range(
                math.floor(minimum_x / self.bucket_size),
                math.floor(maximum_x / self.bucket_size) + 1,
            ):
                self.buckets[(bx, bz)].add(index)
        return index

    def candidates(
        self, polygon: Sequence[PointXZ], *, padding: float = 0.0
    ) -> tuple[tuple[PointXZ, ...], ...]:
        if not polygon or not self.polygons:
            return ()
        padding = max(0.0, float(padding))
        minimum_x = min(point[0] for point in polygon) - padding
        maximum_x = max(point[0] for point in polygon) + padding
        minimum_z = min(point[1] for point in polygon) - padding
        maximum_z = max(point[1] for point in polygon) + padding
        indices: set[int] = set()
        for bz in range(
            math.floor(minimum_z / self.bucket_size),
            math.floor(maximum_z / self.bucket_size) + 1,
        ):
            for bx in range(
                math.floor(minimum_x / self.bucket_size),
                math.floor(maximum_x / self.bucket_size) + 1,
            ):
                indices.update(self.buckets.get((bx, bz), ()))
        return tuple(self.polygons[index] for index in sorted(indices))


def _infill_candidate_rectangles(
    feature: OsmPolygonFeature,
    polygon_index: int,
    polygon: GeoPolygon,
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: OsmSpec,
    road_corridors: Sequence[RoadCorridor],
    occupied: list[tuple[PointXZ, ...]],
    *,
    source_occupied: Sequence[Sequence[PointXZ]] = (),
    source_occupied_index: _PolygonBucketIndex | None = None,
    occupied_index: _PolygonBucketIndex | None = None,
    budget: int,
) -> tuple[tuple[int, float, float, float, float, float, tuple[PointXZ, ...]], ...]:
    """Generate deterministic, sparse house footprints inside one empty residential area."""
    if budget <= 0:
        return ()
    outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
    holes = tuple(tuple(projection.to_world(point) for point in hole[:-1]) for hole in polygon.holes)
    if len(outer) < 3 or _residential_area_has_mapped_building(dataset, projection, outer, holes):
        return ()
    area, _cx, _cz = _polygon_area_centroid(outer)
    minimum_area = max(400.0, float(getattr(spec, "residential_infill_minimum_area", 1800.0)))
    if area < minimum_area:
        return ()
    spacing = max(24.0, float(getattr(spec, "residential_infill_spacing", 68.0)))
    road_clearance = max(0.0, float(getattr(spec, "residential_infill_road_clearance", 0.5)))
    road_frontage_target_distance = 12.0
    road_frontage_search_radius = max(128.0, spacing * 4.0)
    building_clearance = max(1.0, float(getattr(spec, "residential_infill_building_clearance", 6.0)))
    min_x, max_x = min(x for x, _ in outer), max(x for x, _ in outer)
    min_z, max_z = min(z for _, z in outer), max(z for _, z in outer)
    root = f"{getattr(spec, 'deterministic_seed', 'cwr-worldgen')}:residential-infill:{feature.osm_key}:{polygon_index}".encode("utf-8")
    candidates: list[tuple[float, bytes, int, float, float, float, float, float, tuple[PointXZ, ...]]] = []
    cols = max(1, int(math.ceil((max_x - min_x) / spacing)))
    rows = max(1, int(math.ceil((max_z - min_z) / spacing)))
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            digest = hashlib.blake2s(root + index.to_bytes(4, "little"), digest_size=16).digest()
            jitter_x = (int.from_bytes(digest[0:2], "little") / 65535.0 - 0.5) * spacing * 0.55
            jitter_z = (int.from_bytes(digest[2:4], "little") / 65535.0 - 0.5) * spacing * 0.55
            x = min_x + (col + 0.5) * spacing + jitter_x
            z = min_z + (row + 0.5) * spacing + jitter_z
            x, z, road_distance = _pull_point_toward_road_frontage(
                road_corridors,
                x,
                z,
                target_edge_distance=road_frontage_target_distance,
                search_radius=road_frontage_search_radius,
            )
            if not _polygon_contains_with_holes((x, z), outer, holes):
                continue
            # Three restrained house footprints keep generated-model counts bounded.
            template = int.from_bytes(digest[4:6], "little") % 3
            width, length = ((8.0, 11.0), (9.5, 14.0), (11.0, 17.0))[template]
            heading = float((int.from_bytes(digest[6:8], "little") % 12) * 15)
            footprint = _oriented_rectangle(x, z, width, length, heading)
            if not all(_polygon_contains_with_holes(point, outer, holes) for point in footprint):
                continue
            if _polygon_fully_covered_by_mask(
                raster.water, spec.cells, spec.world_size, footprint
            ):
                continue
            if polygon_intersects_road_corridors(road_corridors, footprint, clearance=road_clearance):
                continue
            source_clearance = RESIDENTIAL_INFILL_SOURCE_BUILDING_CLEARANCE_METRES
            source_clearance_sq = source_clearance ** 2
            nearby_source = (
                source_occupied_index.candidates(footprint, padding=source_clearance)
                if source_occupied_index is not None
                else source_occupied
            )
            if any(
                _polygon_minimum_distance_squared(footprint, prior) < source_clearance_sq
                for prior in nearby_source
            ):
                continue
            expanded = _expand_polygon_from_centroid(footprint, building_clearance)
            nearby_occupied = (
                occupied_index.candidates(expanded)
                if occupied_index is not None
                else occupied
            )
            if any(_polygons_intersect(expanded, prior) for prior in nearby_occupied):
                continue
            candidates.append((road_distance, digest[8:], index, x, z, heading, width, length, footprint))
    candidates.sort(key=lambda item: (item[0], item[1]))
    # Sparse by design: at most roughly one building per spacing-square and no
    # more than the caller's global budget.
    chosen = candidates[:budget]
    return tuple((idx, x, z, heading, width, length, footprint) for _distance, _prio, idx, x, z, heading, width, length, footprint in chosen)


def _demote_dense_garage_clusters_to_sheds(
    plans: Sequence[BuildingPlacementPlan],
    dataset: OsmDataset,
    projection: BboxProjection,
    building_asset_library: "ProceduralBuildingLibrary | None",
) -> tuple[BuildingPlacementPlan, ...]:
    """Treat a minority of dense garage-like outbuildings as sheds.

    Mapping datasets sometimes label every small accessory building in a yard or
    garage court as a garage. In a dense local cluster, keep the buildings nearest
    the road as the most likely real garages and deterministically turn roughly a
    third of the farther car-sized outbuildings into sheds. Isolated outbuildings
    still use the footprint-size rule unchanged.
    """

    if building_asset_library is None:
        return tuple(plans)

    # Explicit OSM semantics are authoritative. Cluster heuristics may only
    # reinterpret an outbuilding whose subtype CWR inferred from an untyped or
    # generic footprint. In particular, never demote building=garage/garages.
    source_building_tags = {
        feature.osm_key: str(feature.tags.get("building", "")).casefold()
        for feature in (*dataset.building_polygons, *dataset.building_points)
    }
    candidates: list[int] = []
    for index, plan in enumerate(plans):
        placement = plan.procedural_placement
        selected = getattr(placement, "selected", None) if placement is not None else None
        source_building = source_building_tags.get(plan.osm_key, "")
        subtype_is_inferred = (
            plan.synthetic_infill
            or source_building in {"", "yes", "outbuilding"}
        )
        if (
            selected is not None
            and getattr(selected, "family", "") == "outbuilding"
            and getattr(selected, "outbuilding_kind", "") == "garage"
            and subtype_is_inferred
        ):
            candidates.append(index)
    if len(candidates) <= GARAGE_CLUSTER_MAXIMUM_GARAGES:
        return tuple(plans)

    radius = GARAGE_CLUSTER_RADIUS_METRES
    radius_sq = radius * radius
    # Spatial buckets preserve the exact connected-component rule while avoiding
    # a full scan of every remaining garage for every BFS node. Typical work is
    # now proportional to nearby garages instead of degrading toward O(G^2).
    bucket_size = max(1.0, radius)
    candidate_buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for index in candidates:
        plan = plans[index]
        candidate_buckets[(
            math.floor(plan.x / bucket_size),
            math.floor(plan.z / bucket_size),
        )].add(index)

    remaining = set(candidates)
    clusters: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        seed_plan = plans[seed]
        candidate_buckets[(
            math.floor(seed_plan.x / bucket_size),
            math.floor(seed_plan.z / bucket_size),
        )].discard(seed)
        cluster = [seed]
        queue = deque((seed,))
        while queue:
            current = queue.popleft()
            cx, cz = plans[current].x, plans[current].z
            bucket_x = math.floor(cx / bucket_size)
            bucket_z = math.floor(cz / bucket_size)
            nearby: set[int] = set()
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nearby.update(candidate_buckets.get((bucket_x + dx, bucket_z + dz), ()))
            neighbours = sorted(
                other for other in nearby
                if other in remaining
                and (plans[other].x - cx) ** 2 + (plans[other].z - cz) ** 2 <= radius_sq
            )
            for other in neighbours:
                remaining.remove(other)
                other_plan = plans[other]
                candidate_buckets[(
                    math.floor(other_plan.x / bucket_size),
                    math.floor(other_plan.z / bucket_size),
                )].discard(other)
                cluster.append(other)
                queue.append(other)
        clusters.append(cluster)

    updated = list(plans)
    for cluster in clusters:
        if len(cluster) <= GARAGE_CLUSTER_MAXIMUM_GARAGES:
            continue
        ranked: list[tuple[float, bytes, int]] = []
        for index in cluster:
            plan = plans[index]
            road_point = nearest_road_point(dataset, projection, plan.x, plan.z)
            road_distance = (
                math.hypot(road_point[0] - plan.x, road_point[1] - plan.z)
                if road_point is not None
                else float("inf")
            )
            digest = hashlib.blake2s(
                f"garage-cluster:{plan.osm_key}:{plan.geometry_index}".encode("utf-8"),
                digest_size=8,
            ).digest()
            ranked.append((road_distance, digest, index))
        ranked.sort(key=lambda item: (item[0], item[1]))

        # Keep at most the three strongest vehicle-garage candidates. Road
        # proximity is the primary signal; the stable digest only breaks ties.
        # Everything beyond that cap is an inferred shed. This is an actual
        # maximum, unlike the old 30% demotion heuristic which could leave many
        # more than four garages in a large cluster.
        demotion_pool = ranked[GARAGE_CLUSTER_MAXIMUM_GARAGES:]
        for _distance, _digest, index in demotion_pool:
            plan = updated[index]
            placement = plan.procedural_placement
            assert placement is not None
            selected = replace(placement.selected, outbuilding_kind="shed")
            requested = replace(placement.requested, outbuilding_kind="shed")
            changed_placement = replace(
                placement,
                model_path=building_asset_library.model_path(selected),
                requested=requested,
                selected=selected,
            )
            updated[index] = replace(
                plan,
                model_path=changed_placement.model_path,
                procedural_placement=changed_placement,
            )
    return tuple(updated)


@dataclass(frozen=True, slots=True)
class _ProjectedBuildingEntranceIndex:
    """Cheap world-space lookup for OSM building entrances.

    Building placement used to project and scan every entrance for every
    footprint. On dense extracts that becomes O(buildings * entrances), even
    though only entrances within a few metres of a footprint can match.
    """

    points: tuple[PointXZ, ...]
    features: tuple[OsmPointFeature, ...]
    bucket_size: float
    buckets: Mapping[tuple[int, int], tuple[int, ...]]

    def candidates_for_footprint(
        self, footprint: Sequence[PointXZ], padding: float
    ) -> tuple[int, ...]:
        if not footprint or not self.points:
            return ()
        minimum_x = min(point[0] for point in footprint) - padding
        maximum_x = max(point[0] for point in footprint) + padding
        minimum_z = min(point[1] for point in footprint) - padding
        maximum_z = max(point[1] for point in footprint) + padding
        indices: set[int] = set()
        for bz in range(
            math.floor(minimum_z / self.bucket_size),
            math.floor(maximum_z / self.bucket_size) + 1,
        ):
            for bx in range(
                math.floor(minimum_x / self.bucket_size),
                math.floor(maximum_x / self.bucket_size) + 1,
            ):
                indices.update(self.buckets.get((bx, bz), ()))
        return tuple(sorted(indices))


def _project_building_entrances(
    dataset: OsmDataset, projection: BboxProjection, *, bucket_size: float = 64.0
) -> _ProjectedBuildingEntranceIndex:
    bucket_size = max(8.0, float(bucket_size))
    features = tuple(dataset.building_entrances)
    points = tuple(projection.to_world(feature.point) for feature in features)
    mutable: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x, z) in enumerate(points):
        mutable[(math.floor(x / bucket_size), math.floor(z / bucket_size))].append(index)
    return _ProjectedBuildingEntranceIndex(
        points=points,
        features=features,
        bucket_size=bucket_size,
        buckets={key: tuple(values) for key, values in mutable.items()},
    )


def _mapped_entrance_for_building(
    dataset: OsmDataset,
    projection: BboxProjection,
    footprint: Sequence[PointXZ],
    *,
    maximum_boundary_distance: float = 3.0,
    entrance_index: _ProjectedBuildingEntranceIndex | None = None,
) -> PointXZ | None:
    """Return the best mapped OSM entrance associated with one building polygon.

    Entrance nodes are not reliably delivered with parent-way membership in the
    compact Overpass geometry response, so association is geometric. A node may
    lie just inside, exactly on, or a few centimetres outside the source outline.
    Explicit ``entrance=main`` wins, then the closest usable entrance to the
    footprint boundary.
    """

    if len(footprint) < 3 or not dataset.building_entrances:
        return None
    polygon = tuple(footprint)
    limit_sq = max(0.0, float(maximum_boundary_distance)) ** 2
    priorities = {
        "main": 0,
        "yes": 1,
        "home": 1,
        "staircase": 2,
        "service": 3,
        "emergency": 4,
    }
    candidates: list[tuple[int, float, str, PointXZ]] = []
    if entrance_index is None:
        indexed_features = tuple(dataset.building_entrances)
        indexed_points = tuple(projection.to_world(feature.point) for feature in indexed_features)
        candidate_indices = range(len(indexed_features))
    else:
        indexed_features = entrance_index.features
        indexed_points = entrance_index.points
        candidate_indices = entrance_index.candidates_for_footprint(
            polygon, max(0.0, float(maximum_boundary_distance))
        )
    for index in candidate_indices:
        feature = indexed_features[index]
        point = indexed_points[index]
        distance_sq = min(
            _point_to_segment_distance_squared(point, start, end)
            for start, end in zip(polygon, polygon[1:] + polygon[:1])
        )
        if not _point_in_polygon(point, polygon) and distance_sq > limit_sq:
            continue
        kind = feature.tags.get("entrance", "").strip().casefold()
        candidates.append((priorities.get(kind, 2), distance_sq, feature.osm_key, point))
    if not candidates:
        return None
    return min(candidates)[3]


def _match_nearby_same_shape_building_textures(
    plans: Sequence[BuildingPlacementPlan],
    building_asset_library: "ProceduralBuildingLibrary | None",
    *,
    enabled: bool,
    distance_metres: float,
) -> tuple[BuildingPlacementPlan, ...]:
    """Give local clusters of identical town/city buildings one facade variant.

    Shape identity is deliberately strict: every selected procedural key field
    except ``texture_variant`` must match. Connected groups need at least three
    buildings and every member must resolve to town/city settlement context.
    This makes repeated terraces/estate houses read as one development without
    synchronising unrelated matching buildings across an entire world.
    """

    if not enabled or building_asset_library is None or len(plans) < MINIMUM_NEARBY_BUILDING_TEXTURE_MATCH_CLUSTER:
        return tuple(plans)
    radius = max(10.0, float(distance_metres))
    radius2 = radius * radius
    candidates: list[int] = []
    signatures: dict[int, object] = {}
    for index, plan in enumerate(plans):
        placement = plan.procedural_placement
        if placement is None:
            continue
        context = building_asset_library._settlement_context(plan.x, plan.z)
        if context not in {"town", "city"}:
            continue
        selected = placement.selected
        signatures[index] = replace(selected, texture_variant=0)
        candidates.append(index)
    if len(candidates) < MINIMUM_NEARBY_BUILDING_TEXTURE_MATCH_CLUSTER:
        return tuple(plans)

    bucket_size = radius
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in candidates:
        plan = plans[index]
        buckets[(math.floor(plan.x / bucket_size), math.floor(plan.z / bucket_size))].append(index)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for index in candidates:
        plan = plans[index]
        bx = math.floor(plan.x / bucket_size)
        bz = math.floor(plan.z / bucket_size)
        signature = signatures[index]
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for other in buckets.get((bx + dx, bz + dz), ()):
                    if other <= index or signatures.get(other) != signature:
                        continue
                    candidate = plans[other]
                    if (plan.x - candidate.x) ** 2 + (plan.z - candidate.z) ** 2 <= radius2:
                        adjacency[index].append(other)
                        adjacency[other].append(index)

    updated = list(plans)
    visited: set[int] = set()
    from .procedural_buildings import BuildingPlacement
    for start in candidates:
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for other in adjacency.get(current, ()):
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        if len(component) < MINIMUM_NEARBY_BUILDING_TEXTURE_MATCH_CLUSTER:
            continue
        leader = min(component, key=lambda idx: (plans[idx].osm_key, plans[idx].geometry_index))
        leader_placement = plans[leader].procedural_placement
        assert leader_placement is not None
        texture_variant = leader_placement.selected.texture_variant
        for index in component:
            plan = updated[index]
            placement = plan.procedural_placement
            assert placement is not None
            selected = replace(placement.selected, texture_variant=texture_variant)
            replacement = BuildingPlacement(
                building_asset_library.model_path(selected),
                placement.heading_degrees,
                placement.requested,
                selected,
            )
            updated[index] = replace(
                plan,
                model_path=replacement.model_path,
                procedural_placement=replacement,
            )
    return tuple(updated)


def plan_building_placements(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: OsmSpec,
    building_asset_library: "ProceduralBuildingLibrary | None" = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[tuple[BuildingPlacementPlan, ...], bool]:
    """Resolve final building models, headings, positions, and exact footprints.

    This runs before terrain solving for Milestone 8/9 builds. Terrain pads and
    object placement therefore consume the same final footprint instead of one
    stage flattening the OSM polygon while another stage places a larger rotated
    model somewhere else, a surprisingly effective recipe for hovering houses.
    """

    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    progress(0, "Counting mapped building candidates")
    candidate_total = sum(len(feature.polygons) for feature in dataset.building_polygons) + len(dataset.building_points)

    # Classify each feature once. The old lazy implementation rescanned both
    # building collections for every one of the five priority bands, repeating
    # semantic tag work five times before actual placement even started.
    polygon_features_by_priority: list[list[OsmPolygonFeature]] = [[] for _ in range(5)]
    point_features_by_priority: list[list[OsmPointFeature]] = [[] for _ in range(5)]
    for feature in dataset.building_polygons:
        priority = _building_placement_priority(feature.tags)
        if 0 <= priority < 5:
            polygon_features_by_priority[priority].append(feature)
    for feature in dataset.building_points:
        priority = _building_placement_priority(feature.tags)
        if 0 <= priority < 5:
            point_features_by_priority[priority].append(feature)

    def candidates_for_priority(priority: int):
        # Input feature groups are already sorted by OSM key, so bucketing above
        # preserves the previous deterministic ordering inside each band.
        polygons = (
            (feature.osm_key, polygon_index, "polygon", feature, polygon)
            for feature in polygon_features_by_priority[priority]
            for polygon_index, polygon in enumerate(feature.polygons)
        )
        points = (
            (feature.osm_key, 0, "point", feature, feature.point)
            for feature in point_features_by_priority[priority]
        )
        yield from heapq.merge(
            polygons, points, key=lambda item: (item[0], item[1], item[2])
        )

    candidates = (
        (priority, *candidate)
        for priority in range(5)
        for candidate in candidates_for_priority(priority)
    )
    progress(4, f"Streaming {candidate_total:,} mapped building candidates by priority")

    progress(7, "Preparing building road and entrance lookups")
    building_road_corridors = project_road_corridors(dataset, projection, spec)
    building_entrance_index = (
        _project_building_entrances(dataset, projection)
        if building_asset_library is not None and dataset.building_entrances
        else None
    )
    plans: list[BuildingPlacementPlan] = []
    truncated = False
    advisory_limits = bool(getattr(spec, "advisory_object_limits", False))
    building_warning_threshold = max(0, int(spec.max_buildings))
    building_limit = _advisory_object_limit(building_warning_threshold, enabled=advisory_limits)
    progress_interval = max(1, candidate_total // 40)
    polygon_planner = (
        getattr(building_asset_library, "plan_polygon", None)
        if building_asset_library is not None else None
    )
    point_planner = (
        getattr(building_asset_library, "plan_point", None)
        if building_asset_library is not None else None
    )
    for candidate_number, (_priority, osm_key, geometry_index, geometry_kind, feature, geometry) in enumerate(candidates, start=1):
        if candidate_number == 1 or candidate_number == candidate_total or candidate_number % progress_interval == 0:
            progress(
                10 + round(72 * candidate_number / max(1, candidate_total)),
                f"Resolving mapped buildings {candidate_number:,}/{candidate_total:,} ({len(plans):,} accepted)",
            )
        if len(plans) >= building_limit:
            truncated = True
            break

        procedural_placement = None
        building_family = _building_family(feature.tags)
        if geometry_kind == "polygon":
            projected = [projection.to_world(point) for point in geometry.outer[:-1]]
            projected_holes = tuple(
                tuple(projection.to_world(point) for point in ring[:-1])
                for ring in geometry.holes
                if len(ring) >= 4
            )
            parent_projected = tuple(projected)
            if projected_holes:
                from shapely.geometry import Polygon as ShapelyPolygon

                source_shape = ShapelyPolygon(projected, projected_holes)
                if source_shape.is_empty or not source_shape.is_valid:
                    continue
                area = float(source_shape.area)
                source_centre = source_shape.centroid
                x, z = float(source_centre.x), float(source_centre.y)
            else:
                area, x, z = _polygon_area_centroid(projected)
            if area < spec.building_minimum_area or len(projected) < 3:
                continue
            if not (0 <= x < spec.world_size and 0 <= z < spec.world_size):
                continue
            if building_asset_library is None:
                heading = _longest_edge_heading(projected)
                model_path = _building_model(spec, feature.tags)
                support_polygon = tuple(projected)
            else:
                road_point = nearest_road_point(dataset, projection, x, z)
                entrance_point = _mapped_entrance_for_building(
                    dataset, projection, parent_projected,
                    entrance_index=building_entrance_index,
                )
                procedural_placement = (
                    polygon_planner(
                        feature.tags, projected, holes=projected_holes,
                        road_point=road_point,
                        entrance_point=entrance_point,
                        allow_native_polygon=True,
                    )
                    if polygon_planner is not None
                    else building_asset_library.place_polygon(
                        feature.tags, projected, road_point=(entrance_point or road_point)
                    )
                )
                heading = procedural_placement.heading_degrees
                model_path = procedural_placement.model_path
                building_family = str(getattr(procedural_placement.selected, "family", building_family))
                support_polygon = _procedural_support_polygon(
                    x, z, heading, procedural_placement.selected
                )
        else:
            x, z = projection.to_world(geometry)
            if not (0 <= x < spec.world_size and 0 <= z < spec.world_size):
                continue
            digest = hashlib.blake2s(feature.osm_key.encode("ascii", "ignore"), digest_size=2).digest()
            heading = (int.from_bytes(digest, "little") % 4) * 90.0
            if building_asset_library is None:
                model_path = _building_model(spec, feature.tags)
                support_polygon = _oriented_rectangle(
                    x, z, spec.point_building_footprint, spec.point_building_footprint, heading
                )
            else:
                road_point = nearest_road_point(dataset, projection, x, z)
                procedural_placement = (
                    point_planner(
                        feature.tags, spec.point_building_footprint, heading,
                        x=x, z=z, road_point=road_point,
                    )
                    if point_planner is not None
                    else building_asset_library.place_point(
                        feature.tags, spec.point_building_footprint, heading
                    )
                )
                heading = procedural_placement.heading_degrees
                model_path = procedural_placement.model_path
                building_family = str(getattr(procedural_placement.selected, "family", building_family))
                support_polygon = _oriented_rectangle(
                    x, z,
                    procedural_placement.selected.width_m,
                    procedural_placement.selected.length_m,
                    heading,
                )

        road_nudged = False
        if _polygon_maximum_span(support_polygon) >= LARGE_BUILDING_ROAD_NUDGE_MINIMUM_SPAN_METRES:
            x, z, support_polygon, road_nudged = _nudge_building_footprint_off_roads(
                dataset, projection, raster, spec, x, z, heading, support_polygon,
                road_corridors=building_road_corridors,
            )

        # Keep the complete rigid model inside the terrain vertices that RVW4
        # actually stores. This is a generic building rule, not a church-only
        # vertical offset; it fixes the supplied edge-clipped Vansö church while
        # preserving the same final grounding logic houses already use.
        x, z, support_polygon, _edge_nudged = _nudge_building_inside_sampled_terrain(
            x, z, support_polygon, spec
        )

        plans.append(BuildingPlacementPlan(
            osm_key=osm_key,
            geometry_index=geometry_index,
            geometry_kind=geometry_kind,
            x=x,
            z=z,
            heading_degrees=heading,
            model_path=model_path,
            support_polygon=_compact_support_polygon(support_polygon, x, z, heading),
            procedural_placement=procedural_placement,
            road_nudged=road_nudged,
            building_family=building_family,
        ))

    # Dense accessory-building groups are often over-tagged as garages. Keep
    # the road-facing/road-nearest members as garages and make a minority of the
    # farther buildings sheds before any terrain grounding or model registration
    # occurs, so both the visuals and collision/door sizes use the final type.
    progress(84, f"Post-processing {len(plans):,} mapped building placements")
    plans = list(_demote_dense_garage_clusters_to_sheds(
        plans, dataset, projection, building_asset_library
    ))

    # Missing-building fallback: empty landuse=residential polygons are eligible,
    # and named village/hamlet-style places get a modest synthetic residential
    # patch when OSM has no mapped buildings there. Infill comes after real OSM
    # buildings, so it can never consume the budget ahead of source data.
    if bool(getattr(spec, "residential_infill_enabled", False)) and len(plans) < building_limit:
        progress(90, "Planning residential infill")
        infill_warning_threshold = max(0, int(getattr(spec, "maximum_residential_infill_buildings", 1500)))
        infill_limit = _advisory_object_limit(infill_warning_threshold, enabled=advisory_limits)
        infill_sources: list[tuple[OsmPolygonFeature, int, GeoPolygon]] = []
        for feature in sorted(dataset.urban, key=lambda item: item.osm_key):
            if feature.tags.get("landuse", "").casefold() != "residential":
                continue
            for polygon_index, polygon in enumerate(feature.polygons):
                infill_sources.append((feature, polygon_index, polygon))
        for place in sorted(dataset.places, key=lambda item: item.osm_key):
            settlement = _small_settlement_infill_feature(place, dataset, projection, spec)
            if settlement is None:
                continue
            infill_sources.append((settlement, 0, settlement.polygons[0]))
        source_occupied = [plan.support_polygon for plan in plans]
        occupied = list(source_occupied)
        source_occupied_index = _PolygonBucketIndex.from_polygons(source_occupied)
        occupied_index = _PolygonBucketIndex.from_polygons(occupied)
        generated = 0
        for feature, polygon_index, polygon in infill_sources:
            remaining = min(infill_limit - generated, building_limit - len(plans))
            if remaining <= 0:
                truncated = truncated or generated >= infill_limit
                break
            rectangles = _infill_candidate_rectangles(
                feature, polygon_index, polygon, dataset, projection, raster, spec,
                building_road_corridors, occupied, source_occupied=source_occupied,
                source_occupied_index=source_occupied_index, occupied_index=occupied_index,
                budget=remaining,
            )
            for candidate_index, x, z, heading, width, length, footprint in rectangles:
                # Generated fallback footprints use the same generic building
                # semantics as untyped source footprints. This keeps current
                # house-sized templates residential while allowing the shared
                # tiny-shed and oversized-rural barn selection rules to apply
                # automatically if generated footprint sizes vary.
                tags = {"building": "yes", "cwr:synthetic": "residential_infill"}
                procedural_placement = None
                model_path = spec.generic_building_model
                family = "residential"
                final_heading = heading
                final_footprint = footprint
                if building_asset_library is not None:
                    road_point = nearest_road_point(dataset, projection, x, z)
                    placement = building_asset_library.plan_polygon(
                        tags, footprint, road_point=road_point
                    )
                    # Keep settlement-aware requested family even when the
                    # precomputed variant cap contains no matching family.
                    requested = replace(
                        placement.requested,
                        texture_variant=placement.selected.texture_variant,
                    )
                    if placement.selected.family != requested.family:
                        from .procedural_buildings import BuildingPlacement
                        placement = BuildingPlacement(
                            building_asset_library.model_path(requested), heading,
                            requested, requested,
                        )
                    procedural_placement = placement
                    model_path = placement.model_path
                    family = placement.selected.family
                    final_heading = placement.heading_degrees
                    final_footprint = _oriented_rectangle(
                        x, z, placement.selected.width_m, placement.selected.length_m, final_heading
                    )
                    if not all(
                        _polygon_contains_with_holes(
                            point,
                            tuple(projection.to_world(p) for p in polygon.outer[:-1]),
                            tuple(tuple(projection.to_world(p) for p in h[:-1]) for h in polygon.holes),
                        )
                        for point in final_footprint
                    ):
                        continue
                    if polygon_intersects_road_corridors(
                        building_road_corridors, final_footprint,
                        clearance=max(0.0, float(getattr(spec, "residential_infill_road_clearance", 0.5))),
                    ):
                        continue
                    source_clearance = RESIDENTIAL_INFILL_SOURCE_BUILDING_CLEARANCE_METRES
                    source_clearance_sq = source_clearance ** 2
                    if any(
                        _polygon_minimum_distance_squared(final_footprint, prior) < source_clearance_sq
                        for prior in source_occupied_index.candidates(
                            final_footprint, padding=source_clearance
                        )
                    ):
                        continue
                    expanded_final = _expand_polygon_from_centroid(
                        final_footprint,
                        max(1.0, float(getattr(spec, "residential_infill_building_clearance", 6.0))),
                    )
                    if any(
                        _polygons_intersect(expanded_final, prior)
                        for prior in occupied_index.candidates(expanded_final)
                    ):
                        continue
                if _polygon_fully_covered_by_mask(
                    raster.water, spec.cells, spec.world_size, final_footprint
                ):
                    continue
                key = f"infill/{feature.osm_key}/{polygon_index}-{candidate_index}"
                plans.append(BuildingPlacementPlan(
                    osm_key=key, geometry_index=candidate_index, geometry_kind="synthetic",
                    x=x, z=z, heading_degrees=final_heading, model_path=model_path,
                    support_polygon=_compact_support_polygon(
                        final_footprint, x, z, final_heading
                    ),
                    procedural_placement=procedural_placement, road_nudged=False,
                    building_family=family, synthetic_infill=True,
                ))
                occupied.append(tuple(final_footprint))
                occupied_index.add(final_footprint)
                generated += 1
            if generated >= infill_limit or len(plans) >= building_limit:
                break
    plans = list(_match_nearby_same_shape_building_textures(
        plans,
        building_asset_library,
        enabled=bool(getattr(spec, "match_nearby_building_textures", False)),
        distance_metres=float(getattr(
            spec,
            "nearby_building_texture_match_distance",
            DEFAULT_NEARBY_BUILDING_TEXTURE_MATCH_DISTANCE_METRES,
        )),
    ))
    building_warning = _object_threshold_warning(
        "building footprint", len(plans), building_warning_threshold
    )
    if advisory_limits and building_warning is not None:
        progress(99, building_warning)
    if bool(getattr(spec, "residential_infill_enabled", False)):
        infill_count = sum(1 for plan in plans if plan.synthetic_infill)
        infill_warning = _object_threshold_warning(
            "residential infill building",
            infill_count,
            getattr(spec, "maximum_residential_infill_buildings", 1500),
        )
        if advisory_limits and infill_warning is not None:
            progress(99, infill_warning)
    cap_note = "; building placement disabled" if building_warning_threshold == 0 and candidate_total else ""
    progress(100, f"Resolved {len(plans):,} final building footprints{cap_note}")
    return tuple(plans), truncated


def generate_world_objects(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    spec: OsmSpec,
    *,
    include_roads: bool = True,
    starting_object_id: int = 1,
    building_asset_library: "ProceduralBuildingLibrary | None" = None,
    building_placement_plans: Sequence[BuildingPlacementPlan] | None = None,
    building_plans_truncated: bool = False,
    progress_callback: Callable[[int, str], None] | None = None,
) -> ObjectGenerationResult:
    objects: list[WorldObject] = []
    model_usage: Counter[str] = Counter()
    surface_forest_positions: list[PointXZ] = []
    surface_rock_positions: list[PointXZ] = []
    surface_tracking = bool(getattr(spec, "surface_pass_enabled", False))
    surface_forest_models = {
        str(getattr(spec, "forest_tree_model", "")).casefold(),
        str(getattr(spec, "forest_everon_steep_model", "")).casefold(),
        str(getattr(spec, "forest_single_tree_model", "")).casefold(),
        str(getattr(spec, "forest_hillside_tree_model", "")).casefold(),
        str(getattr(spec, "forest_roadside_tree_model", "")).casefold(),
        *(str(model).casefold() for model in getattr(spec, "forest_roadside_tree_models", ())),
    }
    surface_forest_models.discard("")
    surface_compact_prefix = (str(getattr(spec, "name", "cwr_world")) + r"\f\c_").casefold()
    surface_rock_prefix = (str(getattr(spec, "name", "cwr_world")) + r"\i\rock_").casefold()
    surface_stock_rock_models = {path.casefold() for path in STOCK_STONE_MODELS}
    surface_class_cache: dict[str, int] = {}

    def emit(obj: WorldObject) -> None:
        objects.append(obj)
        model_usage[obj.model_path] += 1
        if not surface_tracking:
            return
        kind = surface_class_cache.get(obj.model_path)
        if kind is None:
            folded = obj.model_path.casefold()
            kind = (
                1 if folded in surface_forest_models or folded.startswith(surface_compact_prefix)
                else 2 if folded.startswith(surface_rock_prefix) or folded in surface_stock_rock_models
                else 0
            )
            # Store 3 for the ordinary/non-surface case so ``dict.get`` can use
            # None as its missing sentinel.
            surface_class_cache[obj.model_path] = kind or 3
        elif kind == 3:
            kind = 0
        if kind == 1:
            surface_forest_positions.append((obj.x, obj.z))
        elif kind == 2:
            surface_rock_positions.append((obj.x, obj.z))

    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(percent, stage)

    progress(53, "Preparing building and vegetation placement")
    infrastructure_library = ProceduralInfrastructureLibrary(str(getattr(spec, "name", "cwr_world")), cache_enabled=False)
    next_id = starting_object_id
    road_count = 0
    road_truncated = False
    advisory_limits = bool(getattr(spec, "advisory_object_limits", False))
    road_warning_threshold = max(0, int(spec.max_road_objects))
    road_limit = _advisory_object_limit(road_warning_threshold, enabled=advisory_limits)

    road_features = dataset.roads if include_roads else ()
    road_polylines = projected_road_polylines(dataset, projection) if include_roads else ()
    for feature, points in zip(road_features, road_polylines):
        if not road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        model = road_model_for_tags(spec, feature.tags)
        for start, end in zip(points, points[1:]):
            dx = end[0] - start[0]
            dz = end[1] - start[1]
            length = math.hypot(dx, dz)
            if length < 1.0:
                continue
            count = max(1, int(round(length / spec.road_segment_length)))
            for segment in range(count):
                if road_count >= road_limit:
                    road_truncated = True
                    break
                fraction = (segment + 0.5) / count
                x = start[0] + dx * fraction
                z = start[1] + dz * fraction
                if not (0 <= x < spec.world_size and 0 <= z < spec.world_size):
                    continue
                if _mask_at(raster.water, spec.cells, spec.world_size, x, z):
                    continue
                heading = math.degrees(math.atan2(dx, dz)) % 360.0
                y = _sample_elevation(elevations, spec.cells, spec.cell_size, x, z) + 0.05
                emit(WorldObject(next_id, model, x, y, z, heading))
                next_id += 1
                road_count += 1
            if road_truncated:
                break
        if road_truncated:
            break

    # Urban-detail pass: stock OFP/CWA pavement and street furniture only.
    # Sidewalks remain disabled; furniture is allowed in mapped towns/cities and
    # explicit landuse=residential areas, but not generic village/rural roads.
    sidewalk_objects = 0
    street_furniture_objects = 0
    street_light_objects = 0
    street_bench_objects = 0
    street_bin_objects = 0
    street_noticeboard_objects = 0
    street_bicycle_objects = 0
    street_bus_shelter_objects = 0
    street_tree_objects = 0
    urban_detail_rejections = 0
    sidewalks_enabled = (
        bool(getattr(spec, "sidewalks_enabled", False))
        and not SIDEWALKS_TEMPORARILY_DISABLED
    )
    street_furniture_enabled = bool(getattr(spec, "street_furniture_enabled", False))
    sidewalk_warning_threshold = max(0, int(getattr(spec, "maximum_sidewalk_objects", 30000)))
    street_furniture_warning_threshold = max(0, int(getattr(spec, "maximum_street_furniture_objects", 12000)))
    maximum_sidewalk_objects = _advisory_object_limit(sidewalk_warning_threshold, enabled=advisory_limits)
    maximum_street_furniture_objects = _advisory_object_limit(street_furniture_warning_threshold, enabled=advisory_limits)
    sidewalk_width = max(0.8, float(getattr(spec, "sidewalk_width", SIDEWALK_DEFAULT_WIDTH_METRES)))
    sidewalk_segment_length = max(2.0, float(getattr(spec, "sidewalk_segment_length", SIDEWALK_DEFAULT_SEGMENT_LENGTH_METRES)))
    street_light_spacing = max(14.0, float(getattr(spec, "street_light_spacing", 32.0)))
    street_bench_every = max(1, int(getattr(spec, "street_bench_every", 4)))
    street_bin_every = max(1, int(getattr(spec, "street_bin_every", 6)))
    street_furniture_road_corridors = (
        _project_vehicle_road_corridors(dataset, projection)
        if street_furniture_enabled else None
    )

    def street_detail_clear_of_roads(
        x: float,
        z: float,
        footprint: float,
        *,
        line_segment: tuple[PointXZ, PointXZ] | None = None,
    ) -> bool:
        if street_furniture_road_corridors is None or not street_furniture_road_corridors:
            return True
        footprint = max(0.0, float(footprint))
        if line_segment is not None:
            return not line_intersects_road_corridors(
                street_furniture_road_corridors,
                line_segment[0],
                line_segment[1],
                clearance=footprint + STREET_FURNITURE_ROAD_CLEARANCE_METRES,
            )
        square = (
            (x - footprint, z - footprint),
            (x + footprint, z - footprint),
            (x + footprint, z + footprint),
            (x - footprint, z + footprint),
        )
        return not polygon_intersects_road_corridors(
            street_furniture_road_corridors,
            square,
            clearance=STREET_FURNITURE_ROAD_CLEARANCE_METRES,
        )

    settlement_place_centres: list[tuple[str, float, float, float]] = []
    for place in dataset.places:
        kind = place.tags.get("place", "").casefold()
        radius = float(SETTLEMENT_DETAIL_RADIUS_METRES.get(kind, 0.0))
        if radius <= 0.0:
            continue
        px, pz = projection.to_world(place.point)
        settlement_place_centres.append((kind, px, pz, radius))

    settlement_polygons: list[tuple[str, tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]]] = []
    residential_polygons: list[tuple[tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]]] = []
    for feature in getattr(dataset, "urban", ()):
        if feature.tags.get("landuse", "").casefold() != "residential":
            continue
        for polygon in feature.polygons:
            outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            holes = tuple(
                tuple(projection.to_world(point) for point in ring[:-1])
                for ring in polygon.holes
                if len(ring) >= 4
            )
            if len(outer) >= 3:
                residential_polygons.append((outer, holes))
    for place_area in getattr(dataset, "place_areas", ()):
        kind = place_area.tags.get("place", "").casefold()
        if kind not in SETTLEMENT_DETAIL_RADIUS_METRES:
            continue
        for polygon in place_area.polygons:
            outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            holes = tuple(
                tuple(projection.to_world(point) for point in ring[:-1])
                for ring in polygon.holes
                if len(ring) >= 4
            )
            if len(outer) >= 3:
                settlement_polygons.append((kind, outer, holes))

    def settlement_kind_at(x: float, z: float) -> str:
        # Prefer explicit place polygons over the conventional point-radius
        # fallback. Denser settlement classes win when overlapping labels exist.
        priority = {"city": 4, "town": 3, "village": 2, "hamlet": 1}
        matches = [
            kind for kind, outer, holes in settlement_polygons
            if _polygon_contains_with_holes((x, z), outer, holes)
        ]
        if matches:
            return max(matches, key=lambda item: priority[item])
        point_matches = [
            kind for kind, px, pz, radius in settlement_place_centres
            if (x - px) ** 2 + (z - pz) ** 2 <= radius * radius
        ]
        if point_matches:
            return max(point_matches, key=lambda item: priority[item])
        return "residential" if residential_position(x, z) else ""

    def town_city_position(x: float, z: float) -> bool:
        return settlement_kind_at(x, z) in {"town", "city"}

    def village_hamlet_position(x: float, z: float) -> bool:
        return settlement_kind_at(x, z) in {"village", "hamlet"}

    def residential_position(x: float, z: float) -> bool:
        return any(
            _polygon_contains_with_holes((x, z), outer, holes)
            for outer, holes in residential_polygons
        )

    def furniture_position(x: float, z: float) -> bool:
        return town_city_position(x, z) or residential_position(x, z)

    def urban_detail_position(x: float, z: float) -> bool:
        return furniture_position(x, z) or _mask_at(
            raster.urban, spec.cells, spec.world_size, x, z
        )

    lamp_grid_size = STREET_FURNITURE_MINIMUM_SEPARATION_METRES
    lamp_grid: dict[tuple[int, int], list[PointXZ]] = defaultdict(list)

    def lamp_position_available(x: float, z: float) -> bool:
        gx, gz = int(math.floor(x / lamp_grid_size)), int(math.floor(z / lamp_grid_size))
        limit2 = STREET_FURNITURE_MINIMUM_SEPARATION_METRES ** 2
        for ix in range(gx - 1, gx + 2):
            for iz in range(gz - 1, gz + 2):
                for ox, oz in lamp_grid.get((ix, iz), ()):
                    if (x - ox) ** 2 + (z - oz) ** 2 < limit2:
                        return False
        lamp_grid[(gx, gz)].append((x, z))
        return True

    if sidewalks_enabled or street_furniture_enabled:
        detail_road_features = dataset.roads
        detail_road_polylines = projected_road_polylines(dataset, projection)
        for feature, points in zip(detail_road_features, detail_road_polylines):
            if not road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
                continue
            if not _urban_detail_road_eligible(feature.tags):
                continue
            highway = feature.tags.get("highway", "").casefold()
            explicit_sidewalk = any(
                key in feature.tags for key in ("sidewalk", "sidewalk:left", "sidewalk:right")
            )
            # Dirt/gravel streets only receive sidewalks when OSM explicitly
            # says they exist. This avoids paving rural tracks merely because a
            # coarse residential landuse polygon happens to cover them.
            surface_allows_inference = not road_is_dirt(feature.tags) and not road_is_gravel(feature.tags)
            road_half_width = max(2.0, road_width_metres(feature.tags) * 0.5)

            if sidewalks_enabled and sidewalk_objects < maximum_sidewalk_objects:
                for chunk_index, (x, z, heading, length, x0, z0, x1, z1) in enumerate(
                    _line_chunks(points, sidewalk_segment_length)
                ):
                    inferred = urban_detail_position(x, z) and surface_allows_inference and highway != "service"
                    sides = _sidewalk_sides(feature.tags, inferred=inferred)
                    if not sides:
                        continue
                    dx, dz = x1 - x0, z1 - z0
                    line_length = max(1.0e-6, math.hypot(dx, dz))
                    ux, uz = dx / line_length, dz / line_length
                    right_x, right_z = uz, -ux
                    offset = road_half_width + SIDEWALK_CURB_GAP_METRES + sidewalk_width * 0.5
                    for side in sides:
                        if sidewalk_objects >= maximum_sidewalk_objects:
                            break
                        sx = x + right_x * offset * side
                        sz = z + right_z * offset * side
                        sx0 = x0 + right_x * offset * side
                        sz0 = z0 + right_z * offset * side
                        sx1 = x1 + right_x * offset * side
                        sz1 = z1 + right_z * offset * side
                        if not (0.0 <= sx < spec.world_size and 0.0 <= sz < spec.world_size):
                            urban_detail_rejections += 1
                            continue
                        if (
                            _mask_at(raster.water, spec.cells, spec.world_size, sx, sz)
                            or _mask_at(raster.buildings, spec.cells, spec.world_size, sx, sz)
                            or _mask_at(raster.buildings, spec.cells, spec.world_size, sx0, sz0)
                            or _mask_at(raster.buildings, spec.cells, spec.world_size, sx1, sz1)
                        ):
                            urban_detail_rejections += 1
                            continue
                        y, pitch = _infrastructure_anchor(
                            elevations, spec.cells, spec.cell_size, sx0, sz0, sx1, sz1, clearance=0.0
                        )
                        if abs(pitch) > 28.0:
                            urban_detail_rejections += 1
                            continue
                        # Vanilla-only sidewalk: tile the stock cobbled-square
                        # pavement object along the road edge. It cannot be scaled,
                        # so width/segment settings control spacing/offset rather
                        # than generating a bespoke mesh.
                        model = STOCK_SIDEWALK_MODELS[0]
                        emit(WorldObject(next_id, model, sx, y, sz, heading, pitch_degrees=pitch))
                        next_id += 1
                        sidewalk_objects += 1

            if (
                not street_furniture_enabled
                or street_furniture_objects >= maximum_street_furniture_objects
                or highway == "service"
            ):
                continue

            for furniture_index, (x, z, heading, _length, x0, z0, x1, z1) in enumerate(
                _line_chunks(points, street_light_spacing, endpoint_trim=2.5)
            ):
                if street_furniture_objects >= maximum_street_furniture_objects:
                    break
                if not furniture_position(x, z):
                    continue
                dx, dz = x1 - x0, z1 - z0
                line_length = max(1.0e-6, math.hypot(dx, dz))
                ux, uz = dx / line_length, dz / line_length
                right_x, right_z = uz, -ux
                identity = f"{getattr(spec, 'deterministic_seed', 'cwr-worldgen')}:street:{feature.osm_key}:{furniture_index}"
                digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=4).digest()
                side = -1 if digest[0] & 1 else 1
                furniture_offset = road_half_width + (sidewalk_width + 0.55 if sidewalks_enabled else 1.25)
                fx = x + right_x * furniture_offset * side
                fz = z + right_z * furniture_offset * side
                if not (0.0 <= fx < spec.world_size and 0.0 <= fz < spec.world_size):
                    urban_detail_rejections += 1
                    continue
                if (
                    _mask_at(raster.water, spec.cells, spec.world_size, fx, fz)
                    or _mask_at(raster.buildings, spec.cells, spec.world_size, fx, fz)
                    or not street_detail_clear_of_roads(fx, fz, 0.55)
                    or not lamp_position_available(fx, fz)
                ):
                    urban_detail_rejections += 1
                    continue
                _minimum, maximum = _square_elevation_extrema(
                    elevations, spec.cells, spec.cell_size, fx, fz, 0.9
                )
                lamp_model = _stock_street_light_model(feature.tags, identity)
                emit(WorldObject(next_id, lamp_model, fx, maximum + STREET_FURNITURE_GROUND_CLEARANCE_METRES, fz, heading))
                next_id += 1
                street_furniture_objects += 1
                street_light_objects += 1

                # Benches and bins are much sparser than lights. Shift them
                # along the sidewalk so they do not occupy exactly the same
                # origin as a lamp post.
                if street_furniture_objects < maximum_street_furniture_objects and furniture_index % street_bench_every == 0:
                    bx = fx + ux * 2.0
                    bz = fz + uz * 2.0
                    if (
                        town_city_position(bx, bz)
                        and 0.0 <= bx < spec.world_size and 0.0 <= bz < spec.world_size
                        and not _mask_at(raster.water, spec.cells, spec.world_size, bx, bz)
                        and not _mask_at(raster.buildings, spec.cells, spec.world_size, bx, bz)
                        and street_detail_clear_of_roads(bx, bz, 0.9)
                    ):
                        _mn, bmax = _square_elevation_extrema(elevations, spec.cells, spec.cell_size, bx, bz, 1.8)
                        emit(WorldObject(next_id, STOCK_STREET_BENCH_MODELS[0], bx, bmax + 0.03, bz, (heading + 90.0) % 360.0))
                        next_id += 1
                        street_furniture_objects += 1
                        street_bench_objects += 1
                if street_furniture_objects < maximum_street_furniture_objects and furniture_index % street_bin_every == 2:
                    qx = fx - ux * 1.35
                    qz = fz - uz * 1.35
                    if (
                        0.0 <= qx < spec.world_size and 0.0 <= qz < spec.world_size
                        and not _mask_at(raster.water, spec.cells, spec.world_size, qx, qz)
                        and not _mask_at(raster.buildings, spec.cells, spec.world_size, qx, qz)
                        and street_detail_clear_of_roads(qx, qz, 0.5)
                    ):
                        qy = _sample_elevation(elevations, spec.cells, spec.cell_size, qx, qz)
                        bin_model = STOCK_STREET_BIN_MODELS[digest[1] % len(STOCK_STREET_BIN_MODELS)]
                        emit(WorldObject(next_id, bin_model, qx, qy + 0.03, qz, float(digest[1] % 360)))
                        next_id += 1
                        street_furniture_objects += 1
                        street_bin_objects += 1

                # Additional vanilla town/city clutter. These are intentionally
                # sparse and deterministic so the world feels occupied without
                # converting every pavement into an object-count stress test.
                if street_furniture_objects < maximum_street_furniture_objects and furniture_index % 11 == 3:
                    nx = fx + ux * 3.2
                    nz = fz + uz * 3.2
                    if (
                        0.0 <= nx < spec.world_size and 0.0 <= nz < spec.world_size
                        and not _mask_at(raster.water, spec.cells, spec.world_size, nx, nz)
                        and not _mask_at(raster.buildings, spec.cells, spec.world_size, nx, nz)
                        and street_detail_clear_of_roads(nx, nz, 0.7)
                    ):
                        ny = _sample_elevation(elevations, spec.cells, spec.cell_size, nx, nz)
                        notice_model = STOCK_STREET_NOTICEBOARD_MODELS[digest[2] % len(STOCK_STREET_NOTICEBOARD_MODELS)]
                        emit(WorldObject(next_id, notice_model, nx, ny + 0.04, nz, (heading + 90.0) % 360.0))
                        next_id += 1
                        street_furniture_objects += 1
                        street_noticeboard_objects += 1

                if street_furniture_objects < maximum_street_furniture_objects and furniture_index % 7 == 4:
                    cx = fx - ux * 2.7
                    cz = fz - uz * 2.7
                    if (
                        0.0 <= cx < spec.world_size and 0.0 <= cz < spec.world_size
                        and not _mask_at(raster.water, spec.cells, spec.world_size, cx, cz)
                        and not _mask_at(raster.buildings, spec.cells, spec.world_size, cx, cz)
                        and street_detail_clear_of_roads(cx, cz, 0.7)
                    ):
                        cy = _sample_elevation(elevations, spec.cells, spec.cell_size, cx, cz)
                        emit(WorldObject(next_id, STOCK_STREET_BICYCLE_MODELS[0], cx, cy + 0.03, cz, heading))
                        next_id += 1
                        street_furniture_objects += 1
                        street_bicycle_objects += 1

                # Every few lamp intervals add a stock pavement tree surround
                # plus a vanilla maple. This is furniture/streetscape, so it is
                # limited to the mapped settlement/residential envelope above.
                if street_furniture_objects + 1 < maximum_street_furniture_objects and furniture_index % 5 == 1:
                    tree_side = -side
                    tx = x + right_x * (road_half_width + sidewalk_width + 1.05) * tree_side
                    tz = z + right_z * (road_half_width + sidewalk_width + 1.05) * tree_side
                    if (
                        0.0 <= tx < spec.world_size and 0.0 <= tz < spec.world_size
                        and not _mask_at(raster.water, spec.cells, spec.world_size, tx, tz)
                        and not _mask_at(raster.buildings, spec.cells, spec.world_size, tx, tz)
                        and street_detail_clear_of_roads(tx, tz, 1.3)
                    ):
                        ty = _sample_elevation(elevations, spec.cells, spec.cell_size, tx, tz)
                        emit(WorldObject(next_id, STOCK_STREET_TREE_SURROUND_MODELS[0], tx, ty + 0.02, tz, heading))
                        next_id += 1
                        tree_model = STOCK_STREET_TREE_MODELS[digest[3] % len(STOCK_STREET_TREE_MODELS)]
                        emit(WorldObject(next_id, tree_model, tx, ty + 0.05, tz, float(digest[2] % 360)))
                        next_id += 1
                        street_furniture_objects += 2
                        street_tree_objects += 1

    # Mapped bus stops already receive the stock sign through the semantic
    # pass. In towns/cities or mapped residential landuse, street-furniture mode
    # may also add the vanilla bus shelter. Rural bus stops remain sign-only.
    if street_furniture_enabled and street_furniture_objects < maximum_street_furniture_objects:
        for landmark in sorted(dataset.landmarks, key=lambda item: item.osm_key):
            if street_furniture_objects >= maximum_street_furniture_objects:
                break
            if landmark.tags.get("landmark") != "bus_stop":
                continue
            bx, bz = projection.to_world(landmark.point)
            if not furniture_position(bx, bz):
                continue
            road_heading = nearest_road_heading(dataset, projection, bx, bz)
            bx, bz = nudge_point_away_from_road(
                dataset,
                projection,
                bx,
                bz,
                # A bus-stop node is commonly mapped on the road centreline.
                # Seven metres clears even a 9 m primary/secondary carriageway
                # plus the stock shelter footprint and safety margin.
                distance=7.0,
                fallback_heading=road_heading,
                world_size=spec.world_size,
            )
            if (
                not (0.0 <= bx < spec.world_size and 0.0 <= bz < spec.world_size)
                or _mask_at(raster.water, spec.cells, spec.world_size, bx, bz)
                or _mask_at(raster.buildings, spec.cells, spec.world_size, bx, bz)
                or not street_detail_clear_of_roads(bx, bz, 1.8)
            ):
                urban_detail_rejections += 1
                continue
            _minimum, by = _square_elevation_extrema(
                elevations, spec.cells, spec.cell_size, bx, bz, 1.8
            )
            emit(WorldObject(
                next_id,
                STOCK_STREET_BUS_SHELTER_MODELS[0],
                bx,
                by + STREET_FURNITURE_GROUND_CLEARANCE_METRES,
                bz,
                (road_heading + 90.0) % 360.0,
            ))
            next_id += 1
            street_furniture_objects += 1
            street_bus_shelter_objects += 1

    progress(54, "Placing buildings")
    building_count = 0
    building_truncated = building_plans_truncated
    maximum_building_grounding_raise = 0.0
    maximum_building_pad_relief = 0.0
    maximum_building_foundation_depth = 0.0
    building_foundation_rejections = 0
    building_interior_fallbacks = 0
    building_fully_submerged_rejections = 0
    building_road_nudges = 0

    if building_placement_plans is None:
        building_placement_plans, planned_truncated = plan_building_placements(
            dataset, projection, raster, spec, building_asset_library
        )
        building_truncated = building_truncated or planned_truncated

    # Settlement-detail pass.  Unlike the generic town/city streetscape above,
    # this layer also serves villages and hamlets with domestic/farm clutter.
    # Every model is stock OFP/CWA; fences remain vanilla-only by construction.
    if street_furniture_enabled and street_furniture_objects < maximum_street_furniture_objects:
        settlement_seed = str(getattr(spec, "deterministic_seed", "cwr-worldgen"))
        building_supports = tuple(
            tuple(plan.support_polygon) for plan in building_placement_plans
            if len(plan.support_polygon) >= 3
        )
        support_index = _PolygonBucketIndex.from_polygons(building_supports)
        detail_grid_size = SETTLEMENT_DETAIL_MINIMUM_SEPARATION_METRES
        detail_grid: dict[tuple[int, int], list[PointXZ]] = defaultdict(list)

        def stable_roll(identity: str, modulus: int = 100) -> int:
            return int.from_bytes(
                hashlib.blake2s(identity.encode("utf-8"), digest_size=4).digest(), "little"
            ) % max(1, modulus)

        def plan_radius(plan: BuildingPlacementPlan) -> float:
            return max(
                (math.hypot(px - plan.x, pz - plan.z) for px, pz in plan.support_polygon),
                default=3.0,
            )

        def prop_position_available(
            x: float,
            z: float,
            footprint: float = 0.8,
            *,
            road_segment: tuple[PointXZ, PointXZ] | None = None,
            allow_road_overlap: bool = False,
        ) -> bool:
            if not (0.5 <= x < spec.world_size - 0.5 and 0.5 <= z < spec.world_size - 0.5):
                return False
            if _mask_at(raster.water, spec.cells, spec.world_size, x, z):
                return False
            footprint = max(0.25, float(footprint))
            square = (
                (x - footprint, z - footprint), (x + footprint, z - footprint),
                (x + footprint, z + footprint), (x - footprint, z + footprint),
            )
            if not allow_road_overlap and not street_detail_clear_of_roads(
                x, z, footprint, line_segment=road_segment
            ):
                return False
            for polygon in support_index.candidates(square, padding=1.0):
                if _polygons_intersect(square, polygon):
                    return False
            gx = int(math.floor(x / detail_grid_size))
            gz = int(math.floor(z / detail_grid_size))
            limit2 = detail_grid_size * detail_grid_size
            for ix in range(gx - 1, gx + 2):
                for iz in range(gz - 1, gz + 2):
                    for ox, oz in detail_grid.get((ix, iz), ()):
                        if (x - ox) ** 2 + (z - oz) ** 2 < limit2:
                            return False
            return True

        def emit_prop(
            model: str,
            x: float,
            z: float,
            heading: float,
            *,
            footprint: float = 0.8,
            clearance: float = 0.03,
            hedge_segment: tuple[PointXZ, PointXZ] | None = None,
            line_segment: tuple[PointXZ, PointXZ] | None = None,
            allow_road_overlap: bool = False,
        ) -> bool:
            nonlocal next_id, street_furniture_objects, urban_detail_rejections
            if street_furniture_objects >= maximum_street_furniture_objects:
                return False
            road_segment = hedge_segment if hedge_segment is not None else line_segment
            if not prop_position_available(
                x, z, footprint,
                road_segment=road_segment,
                allow_road_overlap=allow_road_overlap,
            ):
                urban_detail_rejections += 1
                return False
            if hedge_segment is not None:
                y = _hedge_anchor_height(
                    elevations, spec.cells, spec.cell_size,
                    hedge_segment[0][0], hedge_segment[0][1],
                    hedge_segment[1][0], hedge_segment[1][1],
                    model_path=model,
                )
                pitch = 0.0
            elif line_segment is not None:
                y, pitch = _infrastructure_anchor(
                    elevations, spec.cells, spec.cell_size,
                    line_segment[0][0], line_segment[0][1],
                    line_segment[1][0], line_segment[1][1],
                )
                if abs(pitch) > 38.0:
                    urban_detail_rejections += 1
                    return False
            else:
                minimum, maximum = _square_elevation_extrema(
                    elevations, spec.cells, spec.cell_size, x, z, footprint
                )
                if maximum - minimum > 2.5:
                    urban_detail_rejections += 1
                    return False
                y = maximum + clearance
                pitch = 0.0
            emit(WorldObject(next_id, model, x, y, z, heading % 360.0, pitch_degrees=pitch))
            next_id += 1
            street_furniture_objects += 1
            gx = int(math.floor(x / detail_grid_size))
            gz = int(math.floor(z / detail_grid_size))
            detail_grid[(gx, gz)].append((x, z))
            return True

        # Stock telegraph/utility poles follow inhabited minor roads very
        # sparsely. They are visual cues, not a continuous picket line.
        settlement_roads = projected_road_polylines(dataset, projection)
        for feature, points in zip(dataset.roads, settlement_roads):
            if street_furniture_objects >= maximum_street_furniture_objects:
                break
            highway = feature.tags.get("highway", "").casefold()
            if not _urban_detail_road_eligible(feature.tags):
                continue
            if highway not in {"tertiary", "unclassified", "residential", "living_street", "service", "road"}:
                continue
            for pole_index, (x, z, heading, _length, x0, z0, x1, z1) in enumerate(_line_chunks(points, 56.0)):
                kind = settlement_kind_at(x, z)
                if not kind:
                    continue
                if kind in {"city", "town", "residential"} and highway not in {"unclassified", "residential", "living_street", "service", "road"}:
                    continue
                requested_spacing = SETTLEMENT_UTILITY_POLE_SPACING_METRES.get(kind, 240.0)
                skip_factor = max(1, int(round(requested_spacing / 56.0)))
                # OSM often fragments one street into many short ways. Starting
                # every way at pole #0 would still create a forest of poles even
                # with large spacing. Give each way a deterministic phase so
                # short fragments only contribute at the intended sparse rate.
                phase = stable_roll(
                    f"{settlement_seed}:settlement-pole-phase:{feature.osm_key}",
                    skip_factor,
                )
                if pole_index % skip_factor != phase:
                    continue
                dx, dz = x1 - x0, z1 - z0
                line_length = max(1.0e-6, math.hypot(dx, dz))
                ux, uz = dx / line_length, dz / line_length
                right_x, right_z = uz, -ux
                identity = f"{settlement_seed}:settlement-pole:{feature.osm_key}:{pole_index}"
                side = -1 if stable_roll(identity, 2) == 0 else 1
                offset = road_width_metres(feature.tags) * 0.5 + 2.0
                px = x + right_x * offset * side
                pz = z + right_z * offset * side
                model = STOCK_SETTLEMENT_UTILITY_POLE_MODELS[
                    stable_roll(identity + ":model", len(STOCK_SETTLEMENT_UTILITY_POLE_MODELS))
                ]
                emit_prop(model, px, pz, heading, footprint=0.55, clearance=0.04)

        building_tags: dict[str, Mapping[str, str]] = {}
        for feature in (*dataset.building_polygons, *dataset.building_points):
            building_tags.setdefault(feature.osm_key, feature.tags)

        def front_detail_position(plan: BuildingPlacementPlan, extra: float = 2.2) -> tuple[float, float, float, float]:
            road_point = nearest_road_point(dataset, projection, plan.x, plan.z)
            if road_point is not None:
                dx, dz = road_point[0] - plan.x, road_point[1] - plan.z
                length = math.hypot(dx, dz)
            else:
                length = 0.0
            if length > 1.0e-6:
                ux, uz = dx / length, dz / length
            else:
                angle = math.radians(plan.heading_degrees)
                ux, uz = math.sin(angle), math.cos(angle)
            radius = plan_radius(plan)
            x = plan.x + ux * (radius + extra)
            z = plan.z + uz * (radius + extra)
            return x, z, ux, uz

        # Community anchors get the sort of sparse civic clutter visible in
        # small OFP settlements: noticeboard/bench, with shops also receiving a
        # few bins and bicycles.  The same rule applies in towns/cities.
        community_amenities = {
            "community_centre", "townhall", "library", "post_office",
            "clinic", "doctors", "pharmacy", "social_facility",
        }
        for plan in building_placement_plans:
            if street_furniture_objects >= maximum_street_furniture_objects:
                break
            kind = settlement_kind_at(plan.x, plan.z)
            if not kind:
                continue
            tags = building_tags.get(plan.osm_key, {})
            amenity = str(tags.get("amenity", "")).casefold()
            is_community = plan.building_family in {"church", "shop", "school"} or amenity in community_amenities
            if not is_community:
                continue
            identity = f"{settlement_seed}:community:{plan.osm_key}:{plan.geometry_index}"
            x, z, ux, uz = front_detail_position(plan)
            right_x, right_z = uz, -ux
            notice_model = STOCK_STREET_NOTICEBOARD_MODELS[
                stable_roll(identity + ":notice", len(STOCK_STREET_NOTICEBOARD_MODELS))
            ]
            if emit_prop(notice_model, x + right_x * 1.4, z + right_z * 1.4, plan.heading_degrees + 90.0, footprint=0.7):
                street_noticeboard_objects += 1
            bench_x = x - right_x * 2.4
            bench_z = z - right_z * 2.4
            if town_city_position(bench_x, bench_z) and emit_prop(
                STOCK_STREET_BENCH_MODELS[0], bench_x, bench_z,
                plan.heading_degrees + 90.0, footprint=0.9,
            ):
                street_bench_objects += 1
            if plan.building_family == "shop":
                sparse_factor = {"hamlet": 18, "village": 32, "town": 62, "city": 72, "residential": 48}.get(kind, 30)
                if stable_roll(identity + ":bin") < sparse_factor:
                    model = STOCK_STREET_BIN_MODELS[stable_roll(identity + ":bin-model", len(STOCK_STREET_BIN_MODELS))]
                    if emit_prop(model, x + ux * 1.6, z + uz * 1.6, stable_roll(identity + ":bin-heading", 360), footprint=0.5):
                        street_bin_objects += 1
                if stable_roll(identity + ":bike") < max(8, sparse_factor // 2):
                    if emit_prop(STOCK_STREET_BICYCLE_MODELS[0], x - ux * 1.8, z - uz * 1.8, plan.heading_degrees, footprint=0.7):
                        street_bicycle_objects += 1

        # One civic-looking cluster near the central road junction of each named
        # settlement.  This avoids carpeting every three-way junction in signs.
        junction_degree: Counter[tuple[int, int]] = Counter()
        junction_points: dict[tuple[int, int], PointXZ] = {}
        for feature, points in zip(dataset.roads, settlement_roads):
            if not _urban_detail_road_eligible(feature.tags):
                continue
            for start, end in zip(points, points[1:]):
                for point in (start, end):
                    key = (round(point[0]), round(point[1]))
                    junction_degree[key] += 1
                    junction_points.setdefault(key, point)
        junction_candidates = [junction_points[key] for key, degree in junction_degree.items() if degree >= 3]
        for kind, cx, cz, radius in settlement_place_centres:
            if not junction_candidates or street_furniture_objects >= maximum_street_furniture_objects:
                break
            nearest = min(junction_candidates, key=lambda point: (point[0] - cx) ** 2 + (point[1] - cz) ** 2)
            if (nearest[0] - cx) ** 2 + (nearest[1] - cz) ** 2 > (radius * 0.55) ** 2:
                continue
            identity = f"{settlement_seed}:central-junction:{kind}:{round(cx,1)}:{round(cz,1)}"
            angle = math.radians(stable_roll(identity, 360))
            ux, uz = math.sin(angle), math.cos(angle)
            nx, nz = nearest[0] + ux * 5.0, nearest[1] + uz * 5.0
            model = STOCK_STREET_NOTICEBOARD_MODELS[
                stable_roll(identity + ":model", len(STOCK_STREET_NOTICEBOARD_MODELS))
            ]
            if emit_prop(model, nx, nz, math.degrees(angle) + 90.0, footprint=0.7):
                street_noticeboard_objects += 1
            bench_x = nx + uz * 2.3
            bench_z = nz - ux * 2.3
            if kind in {"town", "city"} and town_city_position(bench_x, bench_z) and emit_prop(
                STOCK_STREET_BENCH_MODELS[0], bench_x, bench_z, math.degrees(angle), footprint=0.9,
            ):
                street_bench_objects += 1

        # Residential yards: sparse stock hedges/fences, fruit trees, and a
        # short stock dirt-road entrance. Never place stock sheds/barns here.
        yard_probability = {"hamlet": 64, "village": 58, "town": 36, "city": 16, "residential": 34}
        residential_plans: list[BuildingPlacementPlan] = []
        for plan in building_placement_plans:
            if street_furniture_objects >= maximum_street_furniture_objects:
                break
            if plan.building_family not in {"residential", "townhouse"}:
                continue
            kind = settlement_kind_at(plan.x, plan.z)
            if not kind:
                continue
            residential_plans.append(plan)
            identity = f"{settlement_seed}:yard:{plan.osm_key}:{plan.geometry_index}"
            if stable_roll(identity + ":select") >= yard_probability.get(kind, 25):
                continue
            angle = math.radians(plan.heading_degrees)
            forward_x, forward_z = math.sin(angle), math.cos(angle)
            right_x, right_z = math.cos(angle), -math.sin(angle)
            radius = plan_radius(plan)
            rear_x = plan.x - forward_x * (radius + 4.0)
            rear_z = plan.z - forward_z * (radius + 4.0)

            # One boundary family per yard: hedge OR stock fence, never a mixed
            # little catalogue around the same property.
            if stable_roll(identity + ":boundary") < 62:
                hedge = stable_roll(identity + ":hedge") < 62
                model = (
                    STOCK_HEDGE_MODELS[stable_roll(identity + ":hedge-model", len(STOCK_HEDGE_MODELS))]
                    if hedge else stock_farmland_fence_model(identity + ":fence-model")
                )
                for segment in (-1, 1):
                    sx = rear_x + right_x * segment * 2.4
                    sz = rear_z + right_z * segment * 2.4
                    start = (sx - right_x * 2.3, sz - right_z * 2.3)
                    end = (sx + right_x * 2.3, sz + right_z * 2.3)
                    if hedge:
                        emit_prop(model, sx, sz, plan.heading_degrees + 90.0 + HEDGE_MODEL_HEADING_OFFSET_DEGREES, footprint=0.7, hedge_segment=(start, end))
                    else:
                        emit_prop(model, sx, sz, plan.heading_degrees + 90.0 + FARMLAND_FENCE_HEADING_OFFSET_DEGREES, footprint=0.6, line_segment=(start, end))

            tree_count = 1 + stable_roll(identity + ":tree-count", 3)
            for tree_index in range(tree_count):
                tree_angle = math.radians(stable_roll(identity + f":tree:{tree_index}:angle", 360))
                distance = radius + 5.0 + stable_roll(identity + f":tree:{tree_index}:distance", 50) / 10.0
                tx = plan.x + math.sin(tree_angle) * distance
                tz = plan.z + math.cos(tree_angle) * distance
                tree_model = STOCK_SETTLEMENT_FRUIT_TREE_MODELS[
                    stable_roll(identity + f":tree:{tree_index}:model", len(STOCK_SETTLEMENT_FRUIT_TREE_MODELS))
                ]
                if emit_prop(tree_model, tx, tz, stable_roll(identity + f":tree:{tree_index}:heading", 360), footprint=1.3, clearance=0.05):
                    street_tree_objects += 1

            road_point = nearest_road_point(dataset, projection, plan.x, plan.z)
            if road_point is not None and stable_roll(identity + ":driveway") < 44:
                dx, dz = road_point[0] - plan.x, road_point[1] - plan.z
                distance = math.hypot(dx, dz)
                if radius + 2.5 < distance <= radius + 18.0:
                    ux, uz = dx / distance, dz / distance
                    driveway_x = plan.x + ux * (radius + 3.0)
                    driveway_z = plan.z + uz * (radius + 3.0)
                    driveway_heading = math.degrees(math.atan2(dx, dz)) % 360.0
                    # This is itself a short road-surface object whose purpose is
                    # to meet the mapped carriageway, so it is the sole settlement
                    # detail allowed to overlap the road corridor deliberately.
                    emit_prop(
                        STOCK_SETTLEMENT_DRIVEWAY_MODEL, driveway_x, driveway_z, driveway_heading,
                        footprint=1.3, clearance=0.025, allow_road_overlap=True,
                    )

        # Small fruit-tree groups in the gaps between close residential houses.
        # Pair each house once, keeping this much sparser in cities.
        paired: set[int] = set()
        for index, plan in enumerate(residential_plans):
            if index in paired or street_furniture_objects >= maximum_street_furniture_objects:
                continue
            kind = settlement_kind_at(plan.x, plan.z)
            pair_probability = {"hamlet": 45, "village": 42, "town": 25, "city": 8, "residential": 22}.get(kind, 15)
            identity = f"{settlement_seed}:between-yard:{plan.osm_key}:{plan.geometry_index}"
            if stable_roll(identity) >= pair_probability:
                continue
            best_index = -1
            best_distance2 = float("inf")
            for other_index in range(index + 1, len(residential_plans)):
                if other_index in paired:
                    continue
                other = residential_plans[other_index]
                distance2 = (other.x - plan.x) ** 2 + (other.z - plan.z) ** 2
                if 18.0 ** 2 <= distance2 <= 48.0 ** 2 and distance2 < best_distance2:
                    best_index, best_distance2 = other_index, distance2
            if best_index < 0:
                continue
            paired.update({index, best_index})
            other = residential_plans[best_index]
            mx, mz = (plan.x + other.x) * 0.5, (plan.z + other.z) * 0.5
            dx, dz = other.x - plan.x, other.z - plan.z
            length = max(1.0, math.hypot(dx, dz))
            right_x, right_z = dz / length, -dx / length
            for tree_index, side in enumerate((-1.0, 1.0)):
                tx, tz = mx + right_x * side * 2.8, mz + right_z * side * 2.8
                model = STOCK_SETTLEMENT_FRUIT_TREE_MODELS[
                    stable_roll(identity + f":pair-tree:{tree_index}", len(STOCK_SETTLEMENT_FRUIT_TREE_MODELS))
                ]
                if emit_prop(model, tx, tz, stable_roll(identity + f":pair-heading:{tree_index}", 360), footprint=1.3, clearance=0.05):
                    street_tree_objects += 1

        # Barn-only clutter. Hay bales, wood piles, axes/stumps, and pallets
        # belong beside explicit OSM barns and synthetic infill that actually
        # resolved to the agricultural/barn family. Generic mapped agricultural
        # buildings, sheds, stables, outbuildings, and fields still receive none.
        farm_probability = {"hamlet": 76, "village": 70, "town": 46, "city": 18, "residential": 36}
        for plan in building_placement_plans:
            if street_furniture_objects >= maximum_street_furniture_objects:
                break
            tags = building_tags.get(plan.osm_key, {})
            explicit_barn = str(tags.get("building", "")).casefold() == "barn"
            generated_barn = plan.synthetic_infill and plan.building_family.casefold() == "agricultural"
            if not (explicit_barn or generated_barn):
                continue
            kind = settlement_kind_at(plan.x, plan.z)
            if not kind:
                continue
            identity = f"{settlement_seed}:barn-clutter:{plan.osm_key}:{plan.geometry_index}"
            if stable_roll(identity) >= farm_probability.get(kind, 30):
                continue
            angle = math.radians(plan.heading_degrees)
            forward_x, forward_z = math.sin(angle), math.cos(angle)
            right_x, right_z = math.cos(angle), -math.sin(angle)
            radius = plan_radius(plan)
            count = 2 + stable_roll(identity + ":count", 3)
            for clutter_index in range(count):
                side = -1.0 if clutter_index % 2 else 1.0
                x = plan.x - forward_x * (radius + 3.0 + clutter_index * 1.7) + right_x * side * (3.0 + clutter_index)
                z = plan.z - forward_z * (radius + 3.0 + clutter_index * 1.7) + right_z * side * (3.0 + clutter_index)
                model = STOCK_SETTLEMENT_BARN_CLUTTER_MODELS[
                    stable_roll(identity + f":model:{clutter_index}", len(STOCK_SETTLEMENT_BARN_CLUTTER_MODELS))
                ]
                emit_prop(model, x, z, stable_roll(identity + f":heading:{clutter_index}", 360), footprint=1.1, clearance=0.03)

        # Village/hamlet bus stops stay modest: no mandatory city shelter, just
        # occasional bin/bicycle clutter. Town/city stops can receive the same
        # details in addition to the shelter already placed above.
        for landmark in sorted(dataset.landmarks, key=lambda item: item.osm_key):
            if street_furniture_objects >= maximum_street_furniture_objects:
                break
            if landmark.tags.get("landmark") != "bus_stop":
                continue
            bx, bz = projection.to_world(landmark.point)
            kind = settlement_kind_at(bx, bz)
            if not kind:
                continue
            identity = f"{settlement_seed}:stop-clutter:{landmark.osm_key}"
            heading = nearest_road_heading(dataset, projection, bx, bz)
            bx, bz = nudge_point_away_from_road(
                dataset, projection, bx, bz, distance=6.0,
                fallback_heading=heading, world_size=spec.world_size,
            )
            bin_chance = {"hamlet": 18, "village": 30, "town": 58, "city": 68, "residential": 42}.get(kind, 25)
            bike_chance = {"hamlet": 8, "village": 18, "town": 38, "city": 48, "residential": 28}.get(kind, 15)
            if stable_roll(identity + ":bin") < bin_chance:
                model = STOCK_STREET_BIN_MODELS[stable_roll(identity + ":bin-model", len(STOCK_STREET_BIN_MODELS))]
                if emit_prop(model, bx + 1.6, bz, heading, footprint=0.5):
                    street_bin_objects += 1
            if stable_roll(identity + ":bike") < bike_chance:
                if emit_prop(STOCK_STREET_BICYCLE_MODELS[0], bx - 1.8, bz, heading, footprint=0.7):
                    street_bicycle_objects += 1

    minimum_foundation_depth = max(0.0, float(getattr(spec, "building_foundation_depth", 0.5)))
    foundation_safety = max(0.0, float(getattr(spec, "building_foundation_safety", 0.20)))

    for plan in building_placement_plans:
        if _building_plan_fully_submerged(plan, elevations, raster, spec):
            building_fully_submerged_rejections += 1
            continue
        is_church = plan.building_family == "church"
        # Churches use the exact same grounding footprint as every other
        # procedural building. Earlier church-only margins sampled terrain well
        # outside the actual model and could make the final visual contact appear
        # unchanged even after moving the object origin.
        grounding_polygon = tuple(plan.support_polygon)
        minimum_height, footprint_maximum_height = _polygon_elevation_extrema(
            elevations, spec.cells, spec.cell_size, grounding_polygon
        )
        maximum_height = footprint_maximum_height
        relief = maximum_height - minimum_height
        # Interior and non-interior variants use exactly the same world placement
        # and foundation calculation. Enterable variants now carry their own
        # exterior stairs down the selected foundation depth instead of raising
        # against a special doorway apron or falling back to a closed model.
        ground_clearance = float(spec.building_ground_clearance)
        serialization_safety = 0.0
        required_foundation_depth = max(
            minimum_foundation_depth,
            relief + ground_clearance + foundation_safety,
        )
        placement_for_registration = plan.procedural_placement

        maximum_building_pad_relief = max(maximum_building_pad_relief, relief)
        centre_height = _sample_elevation(
            elevations, spec.cells, spec.cell_size, plan.x, plan.z
        )
        maximum_building_grounding_raise = max(
            maximum_building_grounding_raise, maximum_height - centre_height
        )
        model_path = plan.model_path
        if building_asset_library is not None and placement_for_registration is not None:
            register = getattr(building_asset_library, "register_placement", None)
            if register is not None:
                placement = register(
                    placement_for_registration,
                    foundation_depth_m=required_foundation_depth,
                )
                model_path = placement.model_path
                required_foundation_depth = placement.selected.foundation_depth_m
        maximum_building_foundation_depth = max(
            maximum_building_foundation_depth, required_foundation_depth
        )
        y = maximum_height + ground_clearance
        emit(WorldObject(
            next_id, model_path, plan.x, y, plan.z, plan.heading_degrees
        ))
        next_id += 1
        building_count += 1
        building_road_nudges += int(plan.road_nudged)

    progress(57, "Placing primary forest blocks")
    forest_count = 0
    forest_truncated = False
    maximum_forest_grounding_raise = 0.0
    forest_road_rejections = 0
    forest_slope_rejections = 0
    maximum_forest_relief = 0.0
    maximum_forest_burial = 0.0
    maximum_forest_float = 0.0
    forest_block_objects = 0
    forest_hillside_tree_objects = 0
    forest_hillside_fallback_blocks = 0
    forest_hillside_unfilled_blocks = 0
    forest_hillside_candidate_rejections = 0
    maximum_hillside_tree_relief = 0.0
    forest_everon_steep_objects = 0
    forest_sunk_polygon_objects = 0
    forest_everon_steep_rejections = 0
    forest_cluster_objects = 0
    forest_cluster_rejections = 0
    forest_cluster_maximum_burial = 0.0
    forest_cluster_maximum_float = 0.0
    forest_cluster_variant_counts: dict[str, int] = defaultdict(int)
    forest_undergrowth_objects = 0
    forest_undergrowth_rejections = 0
    forest_undergrowth_maximum_burial = 0.0
    forest_undergrowth_maximum_float = 0.0
    steep_hill_bush_objects = 0
    steep_hill_bush_rejections = 0
    wetland_reed_objects = 0
    wetland_reed_rejections = 0
    forest_border_objects = 0
    forest_border_rejections = 0
    forest_border_maximum_burial = 0.0
    forest_border_maximum_float = 0.0
    forest_single_tree_objects = 0
    forest_gap_infill_tree_objects = 0
    ditch_grass_objects = 0
    ditch_grass_rejections = 0
    ditch_grass_maximum_burial = 0.0
    ditch_grass_maximum_float = 0.0
    barrier_objects = fence_objects = wall_objects = hedge_objects = barrier_rejections = 0
    bridge_objects = bridge_segments = bridge_rejections = 0
    tree_row_objects = orchard_objects = vineyard_objects = scrub_objects = rural_rock_objects = 0
    rural_vegetation_rejections = 0
    meadow_grass_objects = meadow_grass_rejections = 0
    haybale_objects = haybale_rejections = 0
    haybale_fields_total = haybale_fields_selected = 0
    meadow_grass_positions: list[PointXZ] = []
    meadow_grass_rejection_positions: list[PointXZ] = []
    rocky_forest_objects = rocky_forest_rejections = 0
    mapped_tree_objects = mapped_tree_rejections = 0
    utility_objects = utility_rejections = 0

    road_corridors = project_road_corridors(dataset, projection, spec)
    seed = str(getattr(spec, "deterministic_seed", "cwr-worldgen"))
    low_anchor = bool(getattr(spec, "forest_low_anchor", False))
    forest_profile = str(getattr(spec, "forest_profile", "malden")).casefold()
    # Legacy field name from 0.9.252. In 0.9.254+ this means "replace the
    # rigid stock square/triangle forest polygon models with tiled generated
    # clusters". Individually grounded trees remain the last-resort fallback.
    forest_polygon_models_disabled = bool(
        getattr(spec, "forest_individual_objects_only", False)
    )
    individual_tree_root_sink = max(
        0.0,
        float(getattr(spec, "forest_single_tree_root_sink", 0.05)),
    )
    individual_tree_maximum_burial = max(
        0.0,
        float(getattr(spec, "forest_single_tree_maximum_burial", 1.50)),
    )
    individual_tree_maximum_float = max(
        0.0,
        float(getattr(spec, "forest_single_tree_maximum_float", 0.15)),
    )
    accepted_forest_cells: set[int] = set()
    rocky_forest_cells: set[int] = set()
    # CWA can render vegetation proxy children at nonsense heights when their
    # parent forest model sits too close to the finite WRP edge. Keep the large
    # proxy-bearing forest blocks safely inland rather than letting child trees
    # sample beyond the terrain grid.
    edge_guard_enabled = bool(getattr(spec, "forest_low_anchor", False)) and float(spec.world_size) >= 200.0
    forest_world_edge_margin = (
        max(24.0, min(40.0, float(spec.cell_size) * 1.25))
        if edge_guard_enabled else 0.0
    )

    def forest_point_inside_edge_guard(x: float, z: float, margin: float | None = None) -> bool:
        guard = forest_world_edge_margin if margin is None else max(0.0, float(margin))
        if not edge_guard_enabled:
            guard = 0.0
        return guard <= x <= float(spec.world_size) - guard and guard <= z <= float(spec.world_size) - guard

    def mark_accepted_forest(x: float, z: float, radius: float) -> None:
        col0 = max(0, int((x - radius) // spec.cell_size))
        col1 = min(spec.cells - 1, int((x + radius) // spec.cell_size))
        row0 = max(0, int((z - radius) // spec.cell_size))
        row1 = min(spec.cells - 1, int((z + radius) // spec.cell_size))
        rr = radius * radius
        for accepted_row in range(row0, row1 + 1):
            centre_z = (accepted_row + 0.5) * spec.cell_size
            for accepted_col in range(col0, col1 + 1):
                centre_x = (accepted_col + 0.5) * spec.cell_size
                accepted_index = accepted_row * spec.cells + accepted_col
                if raster.forest[accepted_index] and (centre_x - x) ** 2 + (centre_z - z) ** 2 <= rr:
                    accepted_forest_cells.add(accepted_index)

    forest_warning_threshold = max(0, int(spec.max_forest_objects))
    forest_limit = _advisory_object_limit(forest_warning_threshold, enabled=advisory_limits)
    if forest_limit > 0:
        # Performance-oriented ladder: broad stock square, then the smaller
        # stock triangle, then one reusable grouped fallback cluster.
        spacing = spec.forest_tree_spacing
        columns = max(1, int(math.ceil(spec.world_size / spacing)))
        rows = columns
        half = spacing * 0.5
        clearance = min(half * 0.7, max(1.0, spec.cell_size * 0.45))
        maximum_allowed_relief = max(
            0.0, float(getattr(spec, "forest_maximum_block_relief", 8.0))
        )
        block_maximum_burial = max(
            0.0, float(getattr(spec, "forest_block_maximum_burial", 8.0))
        )
        block_maximum_float = max(
            0.0, float(getattr(spec, "forest_block_maximum_float", 0.5))
        )
        everon_steep_model = str(
            getattr(spec, "forest_everon_steep_model", r"data3d\les trojuhelnik pruchozi.p3d")
        )
        everon_steep_footprint = max(
            8.0, float(getattr(spec, "forest_everon_steep_footprint", 35.0))
        )
        everon_steep_maximum_relief = max(
            0.0, float(getattr(spec, "forest_everon_steep_maximum_relief", 18.0))
        )
        everon_steep_maximum_burial = max(
            0.0, float(getattr(spec, "forest_everon_steep_maximum_burial", 18.0))
        )
        everon_steep_maximum_float = max(
            0.0, float(getattr(spec, "forest_everon_steep_maximum_float", 0.5))
        )
        polygon_sink_fraction = min(
            1.0,
            max(0.0, float(getattr(spec, "forest_polygon_sink_fraction", 0.5))),
        )
        severe_fallback_enabled = bool(
            getattr(spec, "forest_severe_hill_fallback", True)
        )
        severe_fallback_relief = max(
            0.0, float(getattr(spec, "forest_severe_hill_relief", 5.0))
        )
        severe_fallback_tree_target = max(
            1, int(getattr(spec, "forest_severe_hill_trees_per_block", 10))
        )
        forest_clusters_enabled = bool(
            getattr(spec, "forest_cluster_fallback", False)
        )
        legacy_roadside_tree_model = str(
            getattr(spec, "forest_roadside_tree_model", ROADSIDE_LARGE_TREE_MODEL)
        )
        configured_roadside_tree_models = tuple(
            str(model) for model in getattr(spec, "forest_roadside_tree_models", ROADSIDE_TREE_MODELS)
        )
        roadside_tree_models = (
            (legacy_roadside_tree_model,)
            if legacy_roadside_tree_model.casefold() != ROADSIDE_LARGE_TREE_MODEL.casefold()
            else configured_roadside_tree_models
        )
        if not roadside_tree_models:
            roadside_tree_models = (legacy_roadside_tree_model,)
        roadside_bush_models = tuple(
            str(model) for model in getattr(spec, "forest_roadside_bush_models", ROADSIDE_BUSH_MODELS)
        )
        roadside_tree_footprint = max(
            1.5, float(getattr(spec, "forest_single_tree_footprint", 2.0))
        )
        roadside_tree_maximum_relief = max(
            1.5, float(getattr(spec, "forest_single_tree_maximum_relief", 8.0))
        )
        roadside_trees_per_cut_block = max(
            0, int(getattr(spec, "forest_roadside_trees_per_cut_block", 40))
        )
        roadside_bushes_per_cut_block = max(
            0, int(getattr(spec, "forest_roadside_bushes_per_cut_block", 16))
        )
        roadside_bush_footprint = max(
            0.5, float(getattr(spec, "forest_roadside_bush_footprint", 1.5))
        )
        roadside_bush_maximum_relief = max(
            1.0, float(getattr(spec, "steep_hill_bush_maximum_relief", 8.0))
        )
        roadside_bush_maximum_float = max(
            0.0, float(getattr(spec, "steep_hill_bush_maximum_float", 0.8))
        )
        roadside_bush_clearance = float(
            getattr(spec, "steep_hill_bush_ground_clearance", 0.03)
        )

        # The legacy Malden profile remains available for comparisons. Its
        # individually grounded trees are never used by the default Everon path.
        hillside_enabled = bool(getattr(spec, "forest_hillside_fallback", False))
        hillside_model = str(
            getattr(spec, "forest_hillside_tree_model", r"data3d\str_fikovnik.p3d")
        )
        hillside_target = max(
            0, int(getattr(spec, "forest_hillside_trees_per_block", 5))
        )
        steep_infill_tree_target = severe_fallback_tree_target
        hillside_footprint = max(
            0.5, float(getattr(spec, "forest_hillside_tree_footprint", 4.0))
        )
        hillside_maximum_relief = max(
            0.0, float(getattr(spec, "forest_hillside_tree_maximum_relief", 2.5))
        )
        pending_hillside_trees: list[tuple[tuple[float, float, float, float], ...]] = []
        total_primary_blocks = rows * columns
        primary_progress_stride = max(1, rows // 20)
        mask_scale = spec.cells / spec.world_size

        for row in range(rows):
            if row % primary_progress_stride == 0:
                processed = row * columns
                progress(
                    57,
                    f"Placing primary forest blocks {processed:,}/{total_primary_blocks:,} "
                    f"({forest_count:,} objects)",
                )
            for column in range(columns):
                if forest_count >= forest_limit:
                    forest_truncated = True
                    break
                x = min(spec.world_size - 0.001, (column + 0.5) * spacing)
                z = min(spec.world_size - 0.001, (row + 0.5) * spacing)
                block_edge_guard = max(forest_world_edge_margin, spacing * 0.58)
                if not forest_point_inside_edge_guard(x, z, block_edge_guard):
                    continue
                samples = (
                    (x, z),
                    (x - clearance, z - clearance),
                    (x + clearance, z - clearance),
                    (x - clearance, z + clearance),
                    (x + clearance, z + clearance),
                )
                # Convert the five lattice probes to raster indices once. The
                # old path called _mask_at repeatedly for forest, water, roads
                # and buildings, repeating bounds/division work up to 20 times
                # for the same block.
                sample_indices = tuple(
                    min(spec.cells - 1, int(sz * mask_scale)) * spec.cells
                    + min(spec.cells - 1, int(sx * mask_scale))
                    for sx, sz in samples
                    if 0 <= sx < spec.world_size and 0 <= sz < spec.world_size
                )
                forest_sample_count = sum(raster.forest[index] for index in sample_indices)

                # Everon's road-cut fallback only needs blocks with at least two
                # forest probes. Empty/non-forest lattice cells therefore do not
                # need a road-index query at all. On a 50 km world this removes
                # road lookups from most of the ~1,000,000 primary candidates.
                if forest_profile == "everon" and forest_sample_count < 2:
                    continue
                if forest_profile != "everon" and forest_sample_count != len(samples):
                    # Preserve legacy-profile accounting: it historically tests
                    # road overlap before rejecting partial forest blocks.
                    block_intersects_road = forest_block_intersects_road_corridors(
                        road_corridors, x, z, block_size=spacing
                    )
                    if block_intersects_road:
                        forest_road_rejections += 1
                    continue

                block_intersects_road = forest_block_intersects_road_corridors(
                    road_corridors, x, z, block_size=spacing
                )

                if block_intersects_road and forest_profile == "everon":
                    geographic_column, geographic_row = _geographic_lattice_identity(
                        projection, x, z, spacing
                    )
                    # A 50 m stock forest block contains several baked-in tree
                    # proxies. Placing it across a road puts hidden trunks on the
                    # carriageway, but rejecting the whole block creates a square
                    # bald patch. Replace only road-cut blocks with individually
                    # checked stock trees, preserving wooded verges on both sides.
                    if forest_sample_count < 2:
                        continue
                    forest_road_rejections += 1
                    roadside_placed = 0
                    roadside_tree_positions: list[tuple[float, float]] = []
                    for tree_x, tree_z, tree_heading, tree_variant in _roadside_tree_candidates(
                        seed, geographic_column, geographic_row, x, z, spacing
                    ):
                        if (
                            roadside_placed >= roadside_trees_per_cut_block
                            or forest_count >= forest_limit
                        ):
                            break
                        if not (
                            0 <= tree_x < spec.world_size
                            and 0 <= tree_z < spec.world_size
                            and _mask_at(
                                raster.forest, spec.cells, spec.world_size, tree_x, tree_z
                            )
                            and not _mask_at(
                                raster.water, spec.cells, spec.world_size, tree_x, tree_z
                            )
                            and not _mask_at(
                                raster.roads, spec.cells, spec.world_size, tree_x, tree_z
                            )
                            and not _mask_at(
                                raster.buildings, spec.cells, spec.world_size, tree_x, tree_z
                            )
                        ):
                            continue
                        if forest_block_intersects_road_corridors(
                            road_corridors,
                            tree_x,
                            tree_z,
                            block_size=roadside_tree_footprint,
                        ):
                            continue
                        tree_supports = _square_elevation_samples(
                            elevations,
                            spec.cells,
                            spec.cell_size,
                            tree_x,
                            tree_z,
                            roadside_tree_footprint,
                        )
                        tree_relief = max(tree_supports) - min(tree_supports)
                        maximum_hillside_tree_relief = max(
                            maximum_hillside_tree_relief, tree_relief
                        )
                        if tree_relief > roadside_tree_maximum_relief:
                            continue
                        tree_model = roadside_tree_models[tree_variant % len(roadside_tree_models)]
                        tree_fit = _rooted_tree_fit(
                            _triangle_elevation_bounds(
                                elevations,
                                spec.cells,
                                spec.cell_size,
                                tree_x,
                                tree_z,
                            ),
                            root_sink=individual_tree_root_sink,
                            maximum_burial=individual_tree_maximum_burial,
                        )
                        if tree_fit is None:
                            continue
                        tree_y, _tree_burial = tree_fit
                        emit(
                            WorldObject(
                                next_id,
                                tree_model,
                                tree_x,
                                tree_y,
                                tree_z,
                                tree_heading,
                            )
                        )
                        next_id += 1
                        forest_count += 1
                        forest_single_tree_objects += 1
                        roadside_placed += 1
                        roadside_tree_positions.append((tree_x, tree_z))
                        mark_accepted_forest(tree_x, tree_z, spacing * 0.24)

                    roadside_bushes_placed = 0
                    if roadside_bush_models:
                        for bush_x, bush_z, bush_heading, bush_variant in _roadside_bush_candidates(
                            seed, geographic_column, geographic_row, x, z, spacing
                        ):
                            if (
                                roadside_bushes_placed >= roadside_bushes_per_cut_block
                                or forest_count >= forest_limit
                            ):
                                break
                            if any(
                                (bush_x - tree_x) ** 2 + (bush_z - tree_z) ** 2 < 1.5 ** 2
                                for tree_x, tree_z in roadside_tree_positions
                            ):
                                continue
                            if not (
                                0 <= bush_x < spec.world_size
                                and 0 <= bush_z < spec.world_size
                                and _mask_at(
                                    raster.forest, spec.cells, spec.world_size, bush_x, bush_z
                                )
                                and not _mask_at(
                                    raster.water, spec.cells, spec.world_size, bush_x, bush_z
                                )
                                and not _mask_at(
                                    raster.roads, spec.cells, spec.world_size, bush_x, bush_z
                                )
                                and not _mask_at(
                                    raster.buildings, spec.cells, spec.world_size, bush_x, bush_z
                                )
                            ):
                                continue
                            if forest_block_intersects_road_corridors(
                                road_corridors,
                                bush_x,
                                bush_z,
                                block_size=roadside_bush_footprint,
                            ):
                                continue
                            bush_supports = _square_elevation_samples(
                                elevations,
                                spec.cells,
                                spec.cell_size,
                                bush_x,
                                bush_z,
                                roadside_bush_footprint,
                            )
                            if max(bush_supports) - min(bush_supports) > roadside_bush_maximum_relief:
                                continue
                            bush_model = roadside_bush_models[bush_variant % len(roadside_bush_models)]
                            bush_fit = _non_buried_vegetation_fit(
                                _triangle_elevation_bounds(
                                    elevations,
                                    spec.cells,
                                    spec.cell_size,
                                    bush_x,
                                    bush_z,
                                ),
                                clearance=roadside_bush_clearance,
                                maximum_float=roadside_bush_maximum_float,
                            )
                            if bush_fit is None:
                                continue
                            bush_y, _bush_float = bush_fit
                            emit(
                                WorldObject(
                                    next_id,
                                    bush_model,
                                    bush_x,
                                    bush_y,
                                    bush_z,
                                    bush_heading,
                                )
                            )
                            next_id += 1
                            forest_count += 1
                            roadside_bushes_placed += 1
                    if forest_count >= forest_limit:
                        forest_truncated = True
                    continue

                if block_intersects_road:
                    forest_road_rejections += 1
                    continue

                if forest_sample_count != len(samples):
                    continue
                if any(
                    raster.water[index] or raster.roads[index] or raster.buildings[index]
                    for index in sample_indices
                ):
                    continue

                geographic_column, geographic_row = _geographic_lattice_identity(
                    projection, x, z, spacing
                )
                digest = hashlib.blake2s(
                    f"{seed}:forest:{geographic_column}:{geographic_row}".encode("utf-8"),
                    digest_size=2,
                ).digest()
                heading = float((int.from_bytes(digest, "little") % 4) * 90)

                # Optional stock-polygon replacement mode.  A single generated
                # cluster is much smaller than the stock square/triangle model it
                # replaces, so tile several independently fitted clusters across
                # the former footprint.  Remaining holes are deliberately left
                # visible to the later individual-tree gap-infill pass.
                if forest_polygon_models_disabled:
                    replacements = _forest_polygon_replacement_clusters(
                        elevations=elevations,
                        raster=raster,
                        road_corridors=road_corridors,
                        spec=spec,
                        seed=seed,
                        column=geographic_column,
                        row=geographic_row,
                        x=x,
                        z=z,
                        spacing=spacing,
                        maximum_clusters=min(4, max(0, forest_limit - forest_count)),
                    )
                    for cluster in replacements:
                        (
                            cluster_model,
                            cluster_x,
                            cluster_y,
                            cluster_z,
                            cluster_heading,
                            cluster_variant,
                            cluster_relief,
                            cluster_burial,
                            cluster_float,
                        ) = cluster
                        emit(
                            WorldObject(
                                next_id, cluster_model, cluster_x, cluster_y,
                                cluster_z, cluster_heading,
                            )
                        )
                        next_id += 1
                        forest_count += 1
                        forest_cluster_objects += 1
                        forest_cluster_variant_counts[cluster_variant] += 1
                        mark_accepted_forest(
                            cluster_x, cluster_z,
                            _replacement_cluster_coverage_radius(cluster_variant),
                        )
                        forest_cluster_maximum_burial = max(
                            forest_cluster_maximum_burial, cluster_burial
                        )
                        forest_cluster_maximum_float = max(
                            forest_cluster_maximum_float, cluster_float
                        )
                        maximum_forest_burial = max(
                            maximum_forest_burial, cluster_burial
                        )
                        maximum_forest_float = max(
                            maximum_forest_float, cluster_float
                        )
                        maximum_hillside_tree_relief = max(
                            maximum_hillside_tree_relief, cluster_relief
                        )

                    if replacements:
                        forest_hillside_fallback_blocks += 1
                        if forest_count >= forest_limit:
                            forest_truncated = True
                        continue

                    # A particular patch can still be too rough for a safe planar
                    # cluster.  Keep the existing individually grounded fallback
                    # rather than restoring a rigid floating forest object.
                    forest_cluster_rejections += 1
                    individual_placed = 0
                    for tree_x, tree_z, tree_heading in _dense_hillside_tree_candidates(
                        f"{seed}:no-stock-polygons",
                        geographic_column,
                        geographic_row,
                        x,
                        z,
                        spacing,
                    ):
                        if (
                            individual_placed >= steep_infill_tree_target
                            or forest_count >= forest_limit
                        ):
                            break
                        if not (
                            0 <= tree_x < spec.world_size
                            and 0 <= tree_z < spec.world_size
                            and _mask_at(raster.forest, spec.cells, spec.world_size, tree_x, tree_z)
                            and not _mask_at(raster.water, spec.cells, spec.world_size, tree_x, tree_z)
                            and not _mask_at(raster.roads, spec.cells, spec.world_size, tree_x, tree_z)
                            and not _mask_at(raster.buildings, spec.cells, spec.world_size, tree_x, tree_z)
                        ):
                            continue
                        if forest_block_intersects_road_corridors(
                            road_corridors,
                            tree_x,
                            tree_z,
                            block_size=roadside_tree_footprint,
                        ):
                            continue
                        tree_supports = _square_elevation_samples(
                            elevations,
                            spec.cells,
                            spec.cell_size,
                            tree_x,
                            tree_z,
                            roadside_tree_footprint,
                        )
                        tree_relief = max(tree_supports) - min(tree_supports)
                        maximum_hillside_tree_relief = max(
                            maximum_hillside_tree_relief, tree_relief
                        )
                        if tree_relief > roadside_tree_maximum_relief:
                            continue
                        tree_fit = _rooted_tree_fit(
                            _triangle_elevation_bounds(
                                elevations,
                                spec.cells,
                                spec.cell_size,
                                tree_x,
                                tree_z,
                            ),
                            root_sink=individual_tree_root_sink,
                            maximum_burial=individual_tree_maximum_burial,
                        )
                        if tree_fit is None:
                            continue
                        tree_y, _tree_burial = tree_fit
                        tree_model = roadside_tree_models[
                            (
                                geographic_column * 17
                                + geographic_row * 31
                                + individual_placed
                            )
                            % len(roadside_tree_models)
                        ]
                        emit(
                            WorldObject(
                                next_id,
                                tree_model,
                                tree_x,
                                tree_y,
                                tree_z,
                                tree_heading,
                            )
                        )
                        next_id += 1
                        forest_count += 1
                        forest_hillside_tree_objects += 1
                        individual_placed += 1
                        mark_accepted_forest(tree_x, tree_z, spacing * 0.22)
                    if individual_placed:
                        forest_hillside_fallback_blocks += 1
                    else:
                        forest_hillside_unfilled_blocks += 1
                    continue

                centre_height = _sample_elevation(
                    elevations, spec.cells, spec.cell_size, x, z
                )
                block_supports = _square_elevation_samples(
                    elevations, spec.cells, spec.cell_size, x, z, spacing
                )
                minimum_height = min(block_supports)
                maximum_height = max(block_supports)
                relief = maximum_height - minimum_height
                maximum_forest_relief = max(maximum_forest_relief, relief)

                if not low_anchor:
                    support_height = maximum_height
                    maximum_forest_grounding_raise = max(
                        maximum_forest_grounding_raise, support_height - centre_height
                    )
                    emit(
                        WorldObject(
                            next_id,
                            spec.forest_tree_model,
                            x,
                            support_height + spec.forest_ground_clearance,
                            z,
                            heading,
                        )
                    )
                    next_id += 1
                    forest_count += 1
                    forest_block_objects += 1
                    mark_accepted_forest(x, z, spacing * 0.58)
                    continue

                block_fit = (
                    _terrain_fit_anchor(
                        block_supports,
                        clearance=spec.forest_ground_clearance,
                        maximum_burial=block_maximum_burial,
                        maximum_float=block_maximum_float,
                    )
                    if relief <= maximum_allowed_relief
                    else None
                )
                if block_fit is not None:
                    anchor, burial, floating = block_fit
                    emit(
                        WorldObject(next_id, spec.forest_tree_model, x, anchor, z, heading)
                    )
                    next_id += 1
                    forest_count += 1
                    forest_block_objects += 1
                    mark_accepted_forest(x, z, spacing * 0.58)
                    maximum_forest_burial = max(maximum_forest_burial, burial)
                    maximum_forest_float = max(maximum_forest_float, floating)
                    maximum_forest_grounding_raise = max(
                        maximum_forest_grounding_raise, anchor - centre_height
                    )
                    continue

                forest_slope_rejections += 1
                if forest_profile == "everon":
                    triangle_blocked_by_road = forest_block_intersects_road_corridors(
                        road_corridors, x, z, block_size=everon_steep_footprint
                    )
                    gradient_x, gradient_z = _local_terrain_gradient(
                        elevations, spec.cells, spec.cell_size, x, z
                    )
                    if abs(gradient_x) + abs(gradient_z) > 1.0e-9:
                        heading = math.degrees(math.atan2(-gradient_z, gradient_x)) % 360.0
                    steep_supports = _oriented_footprint_elevation_samples(
                        elevations,
                        spec.cells,
                        spec.cell_size,
                        x,
                        z,
                        everon_steep_footprint * 0.58,
                        everon_steep_footprint,
                        heading,
                    )
                    steep_relief = max(steep_supports) - min(steep_supports)
                    maximum_hillside_tree_relief = max(
                        maximum_hillside_tree_relief, steep_relief
                    )
                    polygon_too_steep = (
                        steep_relief > everon_steep_maximum_relief
                    )
                    severe_rigid_fallback = severe_fallback_enabled and (
                        polygon_too_steep
                        or (
                            severe_fallback_relief > 0.0
                            and steep_relief > severe_fallback_relief
                        )
                    )
                    steep_fit = (
                        _terrain_fit_anchor(
                            steep_supports,
                            clearance=spec.forest_ground_clearance,
                            maximum_burial=everon_steep_maximum_burial,
                            maximum_float=everon_steep_maximum_float,
                        )
                        if (
                            not triangle_blocked_by_road
                            and steep_relief <= everon_steep_maximum_relief
                        )
                        else None
                    )
                    if steep_fit is not None:
                        base_anchor, burial, floating = steep_fit
                        sink_depth = (
                            steep_relief * polygon_sink_fraction
                            if (
                                severe_fallback_enabled
                                and polygon_sink_fraction > 0.0
                                and steep_relief > 1.0e-9
                            )
                            else 0.0
                        )
                        anchor = base_anchor - sink_depth
                        burial += sink_depth
                        floating = max(0.0, floating - sink_depth)
                        emit(
                            WorldObject(
                                next_id, everon_steep_model, x, anchor, z, heading
                            )
                        )
                        next_id += 1
                        forest_count += 1
                        forest_everon_steep_objects += 1
                        forest_sunk_polygon_objects += int(sink_depth > 0.0)
                        forest_hillside_fallback_blocks += 1
                        mark_accepted_forest(x, z, everon_steep_footprint * 0.68)
                        maximum_forest_burial = max(maximum_forest_burial, burial)
                        maximum_forest_float = max(maximum_forest_float, floating)
                        continue

                    forest_everon_steep_rejections += 1
                    if triangle_blocked_by_road:
                        forest_road_rejections += 1
                    cluster = (
                        _forest_cluster_placement(
                            elevations=elevations,
                            raster=raster,
                            road_corridors=road_corridors,
                            spec=spec,
                            seed=seed,
                            column=geographic_column,
                            row=geographic_row,
                            x=x,
                            z=z,
                        )
                        if forest_clusters_enabled and not severe_rigid_fallback
                        else None
                    )
                    if cluster is not None:
                        (
                            cluster_model,
                            cluster_x,
                            cluster_y,
                            cluster_z,
                            cluster_heading,
                            cluster_variant,
                            cluster_relief,
                            cluster_burial,
                            cluster_float,
                        ) = cluster
                        emit(
                            WorldObject(
                                next_id,
                                cluster_model,
                                cluster_x,
                                cluster_y,
                                cluster_z,
                                cluster_heading,
                            )
                        )
                        next_id += 1
                        forest_count += 1
                        forest_cluster_objects += 1
                        forest_cluster_variant_counts[cluster_variant] += 1
                        mark_accepted_forest(cluster_x, cluster_z, spacing * 0.42)
                        forest_cluster_maximum_burial = max(
                            forest_cluster_maximum_burial, cluster_burial
                        )
                        forest_cluster_maximum_float = max(
                            forest_cluster_maximum_float, cluster_float
                        )
                        maximum_forest_burial = max(
                            maximum_forest_burial, cluster_burial
                        )
                        maximum_forest_float = max(
                            maximum_forest_float, cluster_float
                        )
                        maximum_hillside_tree_relief = max(
                            maximum_hillside_tree_relief, cluster_relief
                        )
                        forest_hillside_fallback_blocks += 1
                    else:
                        if not severe_rigid_fallback:
                            forest_cluster_rejections += 1
                        severe_underbrush_placed = False
                        if severe_rigid_fallback:
                            underbrush_limit = (
                                _advisory_object_limit(
                                    getattr(spec, "forest_undergrowth_maximum_objects", 120000),
                                    enabled=advisory_limits,
                                )
                                + 1
                            ) // 2
                            underbrush = (
                                _severe_hill_underbrush_placement(
                                    elevations=elevations,
                                    raster=raster,
                                    road_corridors=road_corridors,
                                    spec=spec,
                                    seed=seed,
                                    column=geographic_column,
                                    row=geographic_row,
                                    x=x,
                                    z=z,
                                )
                                if forest_undergrowth_objects < underbrush_limit
                                else None
                            )
                            if underbrush is not None:
                                (
                                    underbrush_model,
                                    underbrush_x,
                                    underbrush_y,
                                    underbrush_z,
                                    underbrush_heading,
                                    underbrush_variant,
                                    _underbrush_relief,
                                    underbrush_burial,
                                    underbrush_float,
                                ) = underbrush
                                emit(
                                    WorldObject(
                                        next_id,
                                        underbrush_model,
                                        underbrush_x,
                                        underbrush_y,
                                        underbrush_z,
                                        underbrush_heading,
                                    )
                                )
                                next_id += 1
                                forest_undergrowth_objects += 1
                                forest_cluster_variant_counts[
                                    underbrush_variant
                                ] += 1
                                forest_undergrowth_maximum_burial = max(
                                    forest_undergrowth_maximum_burial,
                                    underbrush_burial,
                                )
                                forest_undergrowth_maximum_float = max(
                                    forest_undergrowth_maximum_float,
                                    underbrush_float,
                                )
                                maximum_forest_burial = max(
                                    maximum_forest_burial, underbrush_burial
                                )
                                maximum_forest_float = max(
                                    maximum_forest_float, underbrush_float
                                )
                                mark_accepted_forest(
                                    underbrush_x, underbrush_z, spacing * 0.32
                                )
                                severe_underbrush_placed = True
                        rocky_warning_threshold = max(0, int(getattr(spec, "maximum_rocky_forest_objects", 1200)))
                        rocky_limit = _advisory_object_limit(rocky_warning_threshold, enabled=advisory_limits)
                        rocky_enabled = bool(getattr(spec, "rocky_forest_fallback_enabled", False))
                        rocks_per_patch = max(1, int(getattr(spec, "rocky_forest_rocks_per_patch", 3)))
                        rock_spread = min(
                            spacing * 0.42,
                            max(1.0, float(getattr(spec, "rocky_forest_spread", 18.0))),
                        )
                        rocky_placed = 0
                        remaining = max(0, rocky_limit - rocky_forest_objects)
                        rock_candidates = (
                            _rocky_forest_candidates(
                                seed, geographic_column, geographic_row, x, z,
                                min(rocks_per_patch, remaining), rock_spread,
                            )
                            if (
                                rocky_enabled
                                and not severe_rigid_fallback
                                and remaining > 0
                            )
                            else ()
                        )
                        for rock_x, rock_z, rock_heading, rock_size in rock_candidates:
                            if not (
                                0 <= rock_x < spec.world_size
                                and 0 <= rock_z < spec.world_size
                                and _mask_at(raster.forest, spec.cells, spec.world_size, rock_x, rock_z)
                                and not _mask_at(raster.water, spec.cells, spec.world_size, rock_x, rock_z)
                                and not _mask_at(raster.buildings, spec.cells, spec.world_size, rock_x, rock_z)
                            ):
                                continue
                            rock_supports = _square_elevation_samples(
                                elevations, spec.cells, spec.cell_size, rock_x, rock_z, rock_size
                            )
                            rock_relief = max(rock_supports) - min(rock_supports)
                            rock_fit = _terrain_fit_anchor(
                                rock_supports,
                                clearance=0.04,
                                maximum_burial=max(0.0, float(getattr(spec, "rocky_forest_maximum_burial", 1.0))),
                                maximum_float=max(0.0, float(getattr(spec, "rocky_forest_maximum_float", 1.0))),
                            ) if rock_relief <= max(0.0, float(getattr(spec, "rocky_forest_maximum_relief", 42.0))) else None
                            if rock_fit is None or forest_block_intersects_road_corridors(
                                road_corridors, rock_x, rock_z, block_size=rock_size
                            ):
                                continue
                            rock_anchor, _burial, _floating = rock_fit
                            rock_model = STOCK_STONE_MODELS[
                                (geographic_column + geographic_row + rocky_placed) % len(STOCK_STONE_MODELS)
                            ]
                            emit(WorldObject(next_id, rock_model, rock_x, rock_anchor, rock_z, rock_heading))
                            next_id += 1
                            rocky_forest_objects += 1
                            rock_col = min(spec.cells - 1, max(0, int(rock_x // spec.cell_size)))
                            rock_row = min(spec.cells - 1, max(0, int(rock_z // spec.cell_size)))
                            rocky_forest_cells.add(rock_row * spec.cells + rock_col)
                            rocky_placed += 1
                        if not rocky_placed:
                            rocky_forest_rejections += int(
                                rocky_enabled and not severe_rigid_fallback
                            )
                        steep_tree_placed = 0
                        # Refill every rejected rigid/cluster hill patch with a
                        # larger pool of individually grounded trees. The old
                        # fixed 12-point pattern often ran out of valid candidates
                        # after water/road/relief checks, making wooded hills look
                        # noticeably thinner after the grounding-safety changes.
                        for tree_x, tree_z, tree_heading in _dense_hillside_tree_candidates(
                            f"{seed}:steep-infill",
                            geographic_column,
                            geographic_row,
                            x,
                            z,
                            spacing,
                        ):
                            if (
                                steep_tree_placed >= steep_infill_tree_target
                                or forest_count >= forest_limit
                            ):
                                break
                            if not (
                                0 <= tree_x < spec.world_size
                                and 0 <= tree_z < spec.world_size
                                and _mask_at(raster.forest, spec.cells, spec.world_size, tree_x, tree_z)
                                and not _mask_at(raster.water, spec.cells, spec.world_size, tree_x, tree_z)
                                and not _mask_at(raster.roads, spec.cells, spec.world_size, tree_x, tree_z)
                                and not _mask_at(raster.buildings, spec.cells, spec.world_size, tree_x, tree_z)
                            ):
                                continue
                            if forest_block_intersects_road_corridors(
                                road_corridors, tree_x, tree_z, block_size=roadside_tree_footprint
                            ):
                                continue
                            tree_supports = _square_elevation_samples(
                                elevations, spec.cells, spec.cell_size, tree_x, tree_z, roadside_tree_footprint
                            )
                            if max(tree_supports) - min(tree_supports) > roadside_tree_maximum_relief:
                                continue
                            tree_fit = _rooted_tree_fit(
                                _triangle_elevation_bounds(
                                    elevations,
                                    spec.cells,
                                    spec.cell_size,
                                    tree_x,
                                    tree_z,
                                ),
                                root_sink=individual_tree_root_sink,
                                maximum_burial=individual_tree_maximum_burial,
                            )
                            if tree_fit is None:
                                continue
                            tree_y, _tree_burial = tree_fit
                            tree_model = roadside_tree_models[
                                (
                                    geographic_column * 17
                                    + geographic_row * 31
                                    + steep_tree_placed
                                )
                                % len(roadside_tree_models)
                            ]
                            emit(WorldObject(
                                next_id, tree_model, tree_x, tree_y, tree_z, tree_heading
                            ))
                            next_id += 1
                            forest_count += 1
                            forest_hillside_tree_objects += 1
                            steep_tree_placed += 1
                            mark_accepted_forest(tree_x, tree_z, spacing * 0.22)
                        if steep_tree_placed or severe_underbrush_placed:
                            forest_hillside_fallback_blocks += 1
                        else:
                            forest_hillside_unfilled_blocks += 1
                    continue

                # Legacy Malden fallback, retained only behind an explicit profile.
                accepted_candidates: list[tuple[float, float, float, float]] = []
                if hillside_enabled and hillside_target > 0:
                    for tree_x, tree_z, tree_heading in _hillside_tree_candidates(
                        seed, geographic_column, geographic_row, x, z, spacing
                    ):
                        if len(accepted_candidates) >= hillside_target:
                            break
                        tree_samples = _square_elevation_samples(
                            elevations,
                            spec.cells,
                            spec.cell_size,
                            tree_x,
                            tree_z,
                            hillside_footprint,
                        )
                        if not (
                            0 <= tree_x < spec.world_size
                            and 0 <= tree_z < spec.world_size
                            and _mask_at(
                                raster.forest,
                                spec.cells,
                                spec.world_size,
                                tree_x,
                                tree_z,
                            )
                            and not _mask_at(
                                raster.water,
                                spec.cells,
                                spec.world_size,
                                tree_x,
                                tree_z,
                            )
                            and not _mask_at(
                                raster.roads,
                                spec.cells,
                                spec.world_size,
                                tree_x,
                                tree_z,
                            )
                            and not _mask_at(
                                raster.buildings,
                                spec.cells,
                                spec.world_size,
                                tree_x,
                                tree_z,
                            )
                        ):
                            forest_hillside_candidate_rejections += 1
                            continue
                        if forest_block_intersects_road_corridors(
                            road_corridors,
                            tree_x,
                            tree_z,
                            block_size=hillside_footprint,
                        ):
                            forest_hillside_candidate_rejections += 1
                            continue
                        tree_relief = max(tree_samples) - min(tree_samples)
                        maximum_hillside_tree_relief = max(
                            maximum_hillside_tree_relief, tree_relief
                        )
                        if tree_relief > hillside_maximum_relief:
                            forest_hillside_candidate_rejections += 1
                            continue
                        tree_fit = _rooted_tree_fit(
                            _triangle_elevation_bounds(
                                elevations,
                                spec.cells,
                                spec.cell_size,
                                tree_x,
                                tree_z,
                            ),
                            root_sink=individual_tree_root_sink,
                            maximum_burial=individual_tree_maximum_burial,
                        )
                        if tree_fit is None:
                            forest_hillside_candidate_rejections += 1
                            continue
                        tree_y, _tree_burial = tree_fit
                        accepted_candidates.append(
                            (tree_x, tree_z, tree_heading, tree_y)
                        )
                if accepted_candidates:
                    tree_x, tree_z, tree_heading, tree_y = accepted_candidates[0]
                    emit(
                        WorldObject(
                            next_id,
                            hillside_model,
                            tree_x,
                            tree_y,
                            tree_z,
                            tree_heading,
                        )
                    )
                    next_id += 1
                    forest_count += 1
                    forest_hillside_tree_objects += 1
                    forest_hillside_fallback_blocks += 1
                    pending_hillside_trees.append(tuple(accepted_candidates[1:]))
                else:
                    forest_hillside_unfilled_blocks += 1

            if forest_truncated:
                break

        if not forest_truncated and pending_hillside_trees:
            maximum_extra_count = max(
                (len(candidates) for candidates in pending_hillside_trees), default=0
            )
            for extra_index in range(maximum_extra_count):
                for candidates in pending_hillside_trees:
                    if extra_index >= len(candidates):
                        continue
                    if forest_count >= forest_limit:
                        forest_truncated = True
                        break
                    tree_x, tree_z, tree_heading, tree_y = candidates[extra_index]
                    emit(
                        WorldObject(
                            next_id,
                            hillside_model,
                            tree_x,
                            tree_y,
                            tree_z,
                            tree_heading,
                        )
                    )
                    next_id += 1
                    forest_count += 1
                    forest_hillside_tree_objects += 1
                if forest_truncated:
                    break

        # A sparse extra pass of individual trees softens the regular stock forest
        # ladder so woods do not end up looking like a regimented block pattern.
        # This deliberately uses a Resistance/Nogova spruce from O.pbo rather
        # than the Malden ``str_fikovnik`` model, whose texture commonly lives in
        # a separate Data package and therefore broke strict validation.
        progress(57, f"Placed primary forest blocks ({forest_count:,} forest objects so far)")
        extra_single_enabled = bool(getattr(spec, "forest_single_tree_enabled", True))
        extra_single_model = str(getattr(spec, "forest_single_tree_model", r"data3d\str smrk_medium.p3d"))
        extra_single_warning_threshold = _scaled_synthetic_tree_limit(
            int(getattr(spec, "maximum_forest_single_tree_objects", 1000)),
            spec.world_size,
        )
        extra_single_limit = _advisory_object_limit(extra_single_warning_threshold, enabled=advisory_limits)
        extra_single_spacing = max(
            20.0,
            float(getattr(spec, "forest_single_tree_spacing", 45.0)),
        )
        extra_single_footprint = max(1.5, float(getattr(spec, "forest_single_tree_footprint", 2.0)))
        extra_single_relief = max(1.5, float(getattr(spec, "forest_single_tree_maximum_relief", 8.0)))
        extra_single_candidates_per_cell = 1

        # Fill mapped-forest cells that no accepted square, triangle, cluster,
        # road-cut tree, or hillside fallback actually covers. The historical
        # extra-single pass below deliberately required an *already accepted*
        # forest cell, so it could soften existing stands but could never repair
        # the conspicuous bald holes left by rejected rigid models.
        gap_infill_enabled = bool(getattr(spec, "forest_gap_infill_enabled", True))
        gap_infill_spacing = max(
            spec.cell_size,
            float(getattr(spec, "forest_gap_infill_spacing", spec.cell_size)),
        )
        if (
            (forest_profile == "everon" or forest_polygon_models_disabled)
            and gap_infill_enabled
            and not forest_truncated
            and forest_count < forest_limit
        ):
            progress(59, "Filling uncovered mapped forest with rooted trees")
            gap_columns = max(1, int(math.ceil(spec.world_size / gap_infill_spacing)))
            for grid_index in _distributed_grid_indices(
                gap_columns, seed, "forest-gap-tree-infill"
            ):
                if forest_count >= forest_limit:
                    forest_truncated = True
                    break
                gap_row, gap_column = divmod(grid_index, gap_columns)
                digest = hashlib.blake2s(
                    f"{seed}:forest-gap-tree:{gap_column}:{gap_row}".encode("utf-8"),
                    digest_size=8,
                ).digest()
                jitter_x = (
                    int.from_bytes(digest[:2], "little") / 65535.0 - 0.5
                ) * gap_infill_spacing * 0.50
                jitter_z = (
                    int.from_bytes(digest[2:4], "little") / 65535.0 - 0.5
                ) * gap_infill_spacing * 0.50
                tree_x = min(
                    spec.world_size - 0.001,
                    max(0.0, (gap_column + 0.5) * gap_infill_spacing + jitter_x),
                )
                tree_z = min(
                    spec.world_size - 0.001,
                    max(0.0, (gap_row + 0.5) * gap_infill_spacing + jitter_z),
                )
                tree_col = min(spec.cells - 1, max(0, int(tree_x // spec.cell_size)))
                tree_row = min(spec.cells - 1, max(0, int(tree_z // spec.cell_size)))
                tree_index = tree_row * spec.cells + tree_col
                if tree_index in accepted_forest_cells:
                    continue
                if (
                    not raster.forest[tree_index]
                    or raster.water[tree_index]
                    or raster.roads[tree_index]
                    or raster.buildings[tree_index]
                ):
                    continue
                if forest_block_intersects_road_corridors(
                    road_corridors, tree_x, tree_z, block_size=extra_single_footprint
                ):
                    continue
                tree_samples = _square_elevation_samples(
                    elevations, spec.cells, spec.cell_size,
                    tree_x, tree_z, extra_single_footprint,
                )
                if max(tree_samples) - min(tree_samples) > extra_single_relief:
                    continue
                tree_fit = _rooted_tree_fit(
                    _triangle_elevation_bounds(
                        elevations, spec.cells, spec.cell_size, tree_x, tree_z
                    ),
                    root_sink=individual_tree_root_sink,
                    maximum_burial=individual_tree_maximum_burial,
                )
                if tree_fit is None:
                    continue
                tree_y, _tree_burial = tree_fit
                tree_model = roadside_tree_models[
                    int.from_bytes(digest[4:6], "little") % len(roadside_tree_models)
                ]
                tree_heading = float(int.from_bytes(digest[6:], "little") % 360)
                emit(WorldObject(
                    next_id, tree_model, tree_x, tree_y, tree_z, tree_heading
                ))
                next_id += 1
                forest_count += 1
                forest_single_tree_objects += 1
                forest_gap_infill_tree_objects += 1
                # Mark the source raster cell itself without swallowing adjacent
                # 25 m cells. Adjacent gaps may therefore receive their own tree,
                # producing a natural ~20-30 m hillside spacing instead of a
                # second 50 m bald lattice.
                mark_accepted_forest(
                    tree_x, tree_z, max(8.0, gap_infill_spacing * 0.42)
                )

        progress(60, "Scattering individual forest trees")
        if (
            forest_profile == "everon"
            and extra_single_enabled
            and extra_single_limit > 0
            and not forest_truncated
            and forest_count < forest_limit
        ):
            eligible_extra_single_trees: list[
                tuple[int, int, int, float, float, float, float]
            ] = []
            for (
                single_column,
                single_row,
                single_latitude,
                single_longitude,
                single_x,
                single_z,
            ) in _geographic_forest_single_tree_cells(projection, extra_single_spacing):
                if not _mask_at(raster.forest, spec.cells, spec.world_size, single_x, single_z):
                    continue
                if _mask_at(raster.water, spec.cells, spec.world_size, single_x, single_z) or _mask_at(raster.buildings, spec.cells, spec.world_size, single_x, single_z):
                    continue
                accepted = 0
                for tree_x, tree_z, tree_heading in _forest_single_tree_candidates(
                    seed,
                    projection,
                    single_column,
                    single_row,
                    single_latitude,
                    single_longitude,
                    extra_single_spacing,
                ):
                    if accepted >= extra_single_candidates_per_cell:
                        break
                    if not (0 <= tree_x < spec.world_size and 0 <= tree_z < spec.world_size):
                        continue
                    if not _mask_at(raster.forest, spec.cells, spec.world_size, tree_x, tree_z):
                        continue
                    tree_col = min(spec.cells - 1, max(0, int(tree_x // spec.cell_size)))
                    tree_row = min(spec.cells - 1, max(0, int(tree_z // spec.cell_size)))
                    if tree_row * spec.cells + tree_col not in accepted_forest_cells:
                        continue
                    if _mask_at(raster.water, spec.cells, spec.world_size, tree_x, tree_z) or _mask_at(raster.roads, spec.cells, spec.world_size, tree_x, tree_z) or _mask_at(raster.buildings, spec.cells, spec.world_size, tree_x, tree_z):
                        continue
                    if forest_block_intersects_road_corridors(road_corridors, tree_x, tree_z, block_size=extra_single_footprint):
                        continue
                    tree_samples = _square_elevation_samples(elevations, spec.cells, spec.cell_size, tree_x, tree_z, extra_single_footprint)
                    tree_relief = max(tree_samples) - min(tree_samples)
                    if tree_relief > extra_single_relief:
                        continue
                    tree_fit = _rooted_tree_fit(
                        _triangle_elevation_bounds(
                            elevations,
                            spec.cells,
                            spec.cell_size,
                            tree_x,
                            tree_z,
                        ),
                        root_sink=individual_tree_root_sink,
                        maximum_burial=individual_tree_maximum_burial,
                    )
                    if tree_fit is None:
                        continue
                    tree_y, _tree_burial = tree_fit
                    rank = _forest_single_tree_rank(
                        seed, single_column, single_row
                    )
                    eligible_extra_single_trees.append((
                        rank,
                        single_row,
                        single_column,
                        tree_x,
                        tree_y,
                        tree_z,
                        tree_heading,
                    ))
                    accepted += 1
            eligible_extra_single_trees.sort(key=lambda item: item[:3])
            extra_single_available = min(
                extra_single_limit,
                max(0, forest_limit - forest_count),
            )
            for (
                _rank,
                _single_row,
                _single_column,
                tree_x,
                tree_y,
                tree_z,
                tree_heading,
            ) in eligible_extra_single_trees[:extra_single_available]:
                emit(WorldObject(
                    next_id,
                    extra_single_model,
                    tree_x,
                    tree_y,
                    tree_z,
                    tree_heading,
                ))
                next_id += 1
                forest_count += 1
                forest_single_tree_objects += 1
                mark_accepted_forest(tree_x, tree_z, max(spec.cell_size * 0.35, extra_single_spacing * 0.20))
            if (
                forest_count >= forest_limit
                and len(eligible_extra_single_trees) > extra_single_available
            ):
                forest_truncated = True

        progress(61, "Adding rocky outcrops to uncovered forest hills")
        if bool(getattr(spec, "rocky_forest_fallback_enabled", False)):
            rocky_warning_threshold = max(0, int(getattr(spec, "maximum_rocky_forest_objects", 1200)))
            rocky_limit = _advisory_object_limit(rocky_warning_threshold, enabled=advisory_limits)
            remaining_rocks = max(0, rocky_limit - rocky_forest_objects)
            if remaining_rocks > 0:
                gap_spacing = max(spec.cell_size, min(64.0, float(getattr(spec, "forest_tree_spacing", 50.0)) * 0.90))
                gap_columns = max(1, int(math.ceil(spec.world_size / gap_spacing)))
                gap_minimum_slope = max(12.0, min(26.0, float(getattr(spec, "steep_hill_bush_minimum_slope_degrees", 16.0)) + 2.0))
                rock_spread = min(
                    gap_spacing * 0.34,
                    max(1.0, float(getattr(spec, "rocky_forest_spread", 18.0))),
                )
                for grid_index in _distributed_grid_indices(gap_columns, seed, "rocky-forest-gap-infill"):
                    if remaining_rocks <= 0:
                        break
                    gap_row, gap_column = divmod(grid_index, gap_columns)
                    digest = hashlib.blake2s(
                        f"{seed}:rocky-gap:{gap_column}:{gap_row}".encode("utf-8"), digest_size=8
                    ).digest()
                    jitter_x = (int.from_bytes(digest[:2], "little") / 65535.0 - 0.5) * gap_spacing * 0.42
                    jitter_z = (int.from_bytes(digest[2:4], "little") / 65535.0 - 0.5) * gap_spacing * 0.42
                    gap_x = min(spec.world_size - 0.001, max(0.0, (gap_column + 0.5) * gap_spacing + jitter_x))
                    gap_z = min(spec.world_size - 0.001, max(0.0, (gap_row + 0.5) * gap_spacing + jitter_z))
                    gap_col = min(spec.cells - 1, max(0, int(gap_x // spec.cell_size)))
                    gap_cell_row = min(spec.cells - 1, max(0, int(gap_z // spec.cell_size)))
                    gap_index = gap_cell_row * spec.cells + gap_col
                    if (
                        gap_index in accepted_forest_cells
                        or gap_index in rocky_forest_cells
                        or not raster.forest[gap_index]
                        or raster.water[gap_index]
                        or raster.roads[gap_index]
                        or raster.buildings[gap_index]
                    ):
                        continue
                    gradient_x, gradient_z = _local_terrain_gradient(
                        elevations, spec.cells, spec.cell_size, gap_x, gap_z
                    )
                    slope_degrees = math.degrees(math.atan(math.hypot(gradient_x, gradient_z)))
                    # Missing-tree gaps on moderate slopes get an occasional stone;
                    # genuinely steep gaps almost always get one or two.
                    chance = int.from_bytes(digest[4:6], "little") / 65535.0
                    if slope_degrees < gap_minimum_slope and chance > 0.28:
                        continue
                    desired = 2 if slope_degrees >= gap_minimum_slope + 6.0 else 1
                    candidates = _rocky_forest_candidates(
                        f"{seed}:gap", gap_column, gap_row, gap_x, gap_z,
                        min(desired, remaining_rocks), rock_spread,
                    )
                    placed_here = 0
                    for rock_x, rock_z, rock_heading, rock_size in candidates:
                        if remaining_rocks <= 0:
                            break
                        if not (
                            0 <= rock_x < spec.world_size
                            and 0 <= rock_z < spec.world_size
                            and _mask_at(raster.forest, spec.cells, spec.world_size, rock_x, rock_z)
                            and not _mask_at(raster.water, spec.cells, spec.world_size, rock_x, rock_z)
                            and not _mask_at(raster.roads, spec.cells, spec.world_size, rock_x, rock_z)
                            and not _mask_at(raster.buildings, spec.cells, spec.world_size, rock_x, rock_z)
                        ):
                            continue
                        rock_supports = _square_elevation_samples(
                            elevations, spec.cells, spec.cell_size, rock_x, rock_z, rock_size
                        )
                        rock_relief = max(rock_supports) - min(rock_supports)
                        if rock_relief > max(0.0, float(getattr(spec, "rocky_forest_maximum_relief", 42.0))):
                            continue
                        rock_fit = _terrain_fit_anchor(
                            rock_supports, clearance=0.04,
                            maximum_burial=max(0.0, float(getattr(spec, "rocky_forest_maximum_burial", 1.0))),
                            maximum_float=max(0.0, float(getattr(spec, "rocky_forest_maximum_float", 1.0))),
                        )
                        if rock_fit is None or forest_block_intersects_road_corridors(
                            road_corridors, rock_x, rock_z, block_size=rock_size
                        ):
                            continue
                        rock_anchor, _burial, _floating = rock_fit
                        rock_model = STOCK_STONE_MODELS[
                            (gap_column + gap_row + placed_here) % len(STOCK_STONE_MODELS)
                        ]
                        emit(WorldObject(next_id, rock_model, rock_x, rock_anchor, rock_z, rock_heading))
                        next_id += 1
                        rocky_forest_objects += 1
                        remaining_rocks -= 1
                        placed_here += 1
                    if placed_here:
                        rocky_forest_cells.add(gap_index)
                    else:
                        rocky_forest_rejections += 1

        progress(62, "Placing forest undergrowth across forest interiors")
        # Dense reusable undergrowth islands fill the complete forest interior.
        # A deterministic modular walk distributes capped placement over the
        # whole world instead of exhausting a row-major allowance in one corner.
        if (
            bool(getattr(spec, "forest_undergrowth_enabled", False))
        ):
            undergrowth_warning_threshold = max(0, int(getattr(spec, "forest_undergrowth_maximum_objects", 120000)))
            undergrowth_base_limit = _advisory_object_limit(undergrowth_warning_threshold, enabled=advisory_limits)
            undergrowth_limit = (undergrowth_base_limit + 1) // 2
            undergrowth_spacing = max(10.0, float(getattr(spec, "forest_undergrowth_spacing", 30.0)))
            undergrowth_maximum_relief = max(0.0, float(getattr(spec, "forest_undergrowth_maximum_relief", 20.0)))
            undergrowth_maximum_burial = max(0.0, float(getattr(spec, "forest_undergrowth_maximum_burial", 0.8)))
            undergrowth_maximum_float = max(0.0, float(getattr(spec, "forest_undergrowth_maximum_float", 0.8)))
            undergrowth_clearance = float(getattr(spec, "forest_undergrowth_ground_clearance", 0.03))
            undergrowth_columns = max(1, int(math.ceil(spec.world_size / undergrowth_spacing)))
            undergrowth_placeable_index = 0
            for grid_index in _distributed_grid_indices(undergrowth_columns, seed, "forest-undergrowth"):
                if forest_undergrowth_objects >= undergrowth_limit:
                    break
                undergrowth_row, undergrowth_column = divmod(grid_index, undergrowth_columns)
                jitter = hashlib.blake2s(
                    f"{seed}:forest-undergrowth:{undergrowth_column}:{undergrowth_row}".encode("utf-8"),
                    digest_size=8,
                ).digest()
                jitter_x = (int.from_bytes(jitter[:2], "little") / 65535.0 - 0.5) * undergrowth_spacing * 0.42
                jitter_z = (int.from_bytes(jitter[2:4], "little") / 65535.0 - 0.5) * undergrowth_spacing * 0.42
                undergrowth_x = min(spec.world_size - 0.001, max(0.0, (undergrowth_column + 0.5) * undergrowth_spacing + jitter_x))
                undergrowth_z = min(spec.world_size - 0.001, max(0.0, (undergrowth_row + 0.5) * undergrowth_spacing + jitter_z))
                if not _mask_at(raster.forest, spec.cells, spec.world_size, undergrowth_x, undergrowth_z):
                    continue
                variant = FOREST_UNDERGROWTH_VARIANTS[
                    int.from_bytes(jitter[4:6], "little") % len(FOREST_UNDERGROWTH_VARIANTS)
                ]
                heading = float(int.from_bytes(jitter[6:], "little") % 360)
                placed = _place_cluster_at(
                    variant=variant,
                    elevations=elevations,
                    raster=raster,
                    road_corridors=road_corridors,
                    spec=spec,
                    x=undergrowth_x,
                    z=undergrowth_z,
                    heading=heading,
                    require_forest=True,
                    minimum_forest_fraction=0.80,
                    maximum_relief=undergrowth_maximum_relief,
                    maximum_burial=undergrowth_maximum_burial,
                    maximum_float=undergrowth_maximum_float,
                    clearance=undergrowth_clearance,
                )
                if placed is None:
                    forest_undergrowth_rejections += 1
                    continue
                undergrowth_placeable_index += 1
                if undergrowth_placeable_index % 2 == 0:
                    continue
                (
                    model_path, placed_x, placed_y, placed_z, placed_heading,
                    variant_name, _relief, burial, floating,
                ) = placed
                emit(WorldObject(next_id, model_path, placed_x, placed_y, placed_z, placed_heading))
                next_id += 1
                forest_undergrowth_objects += 1
                forest_cluster_variant_counts[variant_name] += 1
                forest_undergrowth_maximum_burial = max(forest_undergrowth_maximum_burial, burial)
                forest_undergrowth_maximum_float = max(forest_undergrowth_maximum_float, floating)
                maximum_forest_burial = max(maximum_forest_burial, burial)
                maximum_forest_float = max(maximum_forest_float, floating)

        progress(63, "Adding stock bushes to steep forested hills")
        if bool(getattr(spec, "steep_hill_bushes_enabled", False)):
            bush_warning_threshold = max(0, int(getattr(spec, "maximum_steep_hill_bush_objects", 80000)))
            bush_limit = _advisory_object_limit(bush_warning_threshold, enabled=advisory_limits)
            bush_spacing = max(8.0, float(getattr(spec, "steep_hill_bush_spacing", 24.0)))
            bush_minimum_slope = max(0.1, float(getattr(spec, "steep_hill_bush_minimum_slope_degrees", 16.0)))
            bush_maximum_relief = max(0.0, float(getattr(spec, "steep_hill_bush_maximum_relief", 8.0)))
            bush_maximum_float = max(0.0, float(getattr(spec, "steep_hill_bush_maximum_float", 0.8)))
            bush_clearance = float(getattr(spec, "steep_hill_bush_ground_clearance", 0.03))
            bush_models = tuple(getattr(spec, "steep_hill_bush_models", (r"data3d\ker listnac.p3d", r"data3d\ker pichlavej.p3d", r"data3d\ker deravej.p3d")))
            bush_columns = max(1, int(math.ceil(spec.world_size / bush_spacing)))
            for grid_index in _distributed_grid_indices(bush_columns, seed, "steep-hill-bushes"):
                if steep_hill_bush_objects >= bush_limit:
                    break
                bush_row, bush_column = divmod(grid_index, bush_columns)
                digest = hashlib.blake2s(
                    f"{seed}:steep-hill-bush:{bush_column}:{bush_row}".encode("utf-8"), digest_size=8
                ).digest()
                jitter_x = (int.from_bytes(digest[:2], "little") / 65535.0 - 0.5) * bush_spacing * 0.55
                jitter_z = (int.from_bytes(digest[2:4], "little") / 65535.0 - 0.5) * bush_spacing * 0.55
                bush_x = min(spec.world_size - 0.001, max(0.0, (bush_column + 0.5) * bush_spacing + jitter_x))
                bush_z = min(spec.world_size - 0.001, max(0.0, (bush_row + 0.5) * bush_spacing + jitter_z))
                if not _mask_at(raster.forest, spec.cells, spec.world_size, bush_x, bush_z):
                    continue
                if (
                    _mask_at(raster.water, spec.cells, spec.world_size, bush_x, bush_z)
                    or _mask_at(raster.roads, spec.cells, spec.world_size, bush_x, bush_z)
                    or _mask_at(raster.buildings, spec.cells, spec.world_size, bush_x, bush_z)
                ):
                    steep_hill_bush_rejections += 1
                    continue
                gradient_x, gradient_z = _local_terrain_gradient(
                    elevations, spec.cells, spec.cell_size, bush_x, bush_z
                )
                slope_degrees = math.degrees(math.atan(math.hypot(gradient_x, gradient_z)))
                if slope_degrees < bush_minimum_slope:
                    continue
                bush_footprint = 4.0
                bush_supports = _square_elevation_samples(
                    elevations, spec.cells, spec.cell_size, bush_x, bush_z, bush_footprint
                )
                bush_relief = max(bush_supports) - min(bush_supports)
                if bush_relief > bush_maximum_relief or forest_block_intersects_road_corridors(
                    road_corridors, bush_x, bush_z, block_size=bush_footprint
                ):
                    steep_hill_bush_rejections += 1
                    continue
                bush_point_supports = _triangle_elevation_bounds(
                    elevations,
                    spec.cells,
                    spec.cell_size,
                    bush_x,
                    bush_z,
                )
                bush_fit = _non_buried_vegetation_fit(
                    bush_point_supports,
                    clearance=bush_clearance,
                    maximum_float=bush_maximum_float,
                )
                if bush_fit is None:
                    steep_hill_bush_rejections += 1
                    continue
                bush_anchor, _bush_float = bush_fit
                bush_model = bush_models[int.from_bytes(digest[4:6], "little") % len(bush_models)]
                bush_heading = float(int.from_bytes(digest[6:], "little") % 360)
                emit(WorldObject(next_id, bush_model, bush_x, bush_anchor, bush_z, bush_heading))
                next_id += 1
                steep_hill_bush_objects += 1

        progress(64, "Softening forest borders")
        # Nogova-style soft borders are a separate sparse pass along actual OSM
        # forest boundaries. They use reusable proxy clusters, not rows of WRP trees.
        if (
            bool(getattr(spec, "forest_border_enabled", False))
        ):
            forest_border_warning_threshold = max(0, int(getattr(spec, "forest_border_maximum_objects", 2000)))
            border_limit = _advisory_object_limit(forest_border_warning_threshold, enabled=advisory_limits)
            border_spacing = max(
                8.0, float(getattr(spec, "forest_border_spacing", 34.0))
            )
            border_inset = max(0.5, float(getattr(spec, "forest_border_inset", 5.0)))
            border_maximum_relief = max(
                0.0, float(getattr(spec, "forest_border_maximum_relief", 24.0))
            )
            border_maximum_burial = max(
                0.0, float(getattr(spec, "forest_border_maximum_burial", 1.0))
            )
            border_maximum_float = max(
                0.0, float(getattr(spec, "forest_border_maximum_float", 1.0))
            )
            for key, x, z, heading in _forest_border_candidates(
                dataset,
                projection,
                spacing=border_spacing,
                inset=border_inset,
                seed=seed,
            ):
                if not forest_point_inside_edge_guard(x, z, max(16.0, forest_world_edge_margin * 0.75)):
                    forest_border_rejections += 1
                    continue
                if forest_border_objects >= border_limit:
                    break
                digest = hashlib.blake2s(
                    f"{seed}:forest-border-variant:{key}".encode("utf-8"),
                    digest_size=4,
                ).digest()
                variant = FOREST_BORDER_VARIANTS[
                    int.from_bytes(digest, "little") % len(FOREST_BORDER_VARIANTS)
                ]
                placed = _place_cluster_at(
                    variant=variant,
                    elevations=elevations,
                    raster=raster,
                    road_corridors=road_corridors,
                    spec=spec,
                    x=x,
                    z=z,
                    heading=heading,
                    require_forest=False,
                    minimum_forest_fraction=0.55,
                    maximum_relief=border_maximum_relief,
                    maximum_burial=border_maximum_burial,
                    maximum_float=border_maximum_float,
                    clearance=spec.forest_ground_clearance,
                )
                if placed is None:
                    forest_border_rejections += 1
                    continue
                (
                    model_path,
                    placed_x,
                    placed_y,
                    placed_z,
                    placed_heading,
                    variant_name,
                    _relief,
                    burial,
                    floating,
                ) = placed
                emit(
                    WorldObject(
                        next_id,
                        model_path,
                        placed_x,
                        placed_y,
                        placed_z,
                        placed_heading,
                    )
                )
                next_id += 1
                forest_border_objects += 1
                forest_cluster_variant_counts[variant_name] += 1
                forest_border_maximum_burial = max(
                    forest_border_maximum_burial, burial
                )
                forest_border_maximum_float = max(
                    forest_border_maximum_float, floating
                )
                maximum_forest_burial = max(maximum_forest_burial, burial)
                maximum_forest_float = max(maximum_forest_float, floating)

    progress(64, "Placing ditch vegetation")
    # Tall grass follows explicitly mapped OSM ditches. It has its own budget so
    # a dense drainage network cannot consume the forest object allowance.
    if bool(getattr(spec, "ditch_grass_enabled", False)):
        ditch_warning_threshold = max(0, int(getattr(spec, "maximum_ditch_grass_objects", 2000)))
        ditch_limit = _advisory_object_limit(ditch_warning_threshold, enabled=advisory_limits)
        ditch_spacing = max(6.0, float(getattr(spec, "ditch_grass_spacing", 18.0)))
        ditch_trim = max(0.0, float(getattr(spec, "ditch_grass_endpoint_trim", 6.0)))
        ditch_maximum_relief = max(
            0.0, float(getattr(spec, "ditch_grass_maximum_relief", 18.0))
        )
        ditch_maximum_burial = max(
            0.0, float(getattr(spec, "ditch_grass_maximum_burial", 0.6))
        )
        ditch_maximum_float = max(
            0.0, float(getattr(spec, "ditch_grass_maximum_float", 0.8))
        )
        ditch_clearance = float(getattr(spec, "ditch_grass_ground_clearance", 0.05))
        for key, x, z, heading in _ditch_grass_candidates(
            dataset,
            projection,
            spacing=ditch_spacing,
            endpoint_trim=ditch_trim,
            seed=seed,
        ):
            if ditch_grass_objects >= ditch_limit:
                break
            digest = hashlib.blake2s(
                f"{seed}:ditch-grass-variant:{key}".encode("utf-8"), digest_size=4
            ).digest()
            variant = DITCH_GRASS_VARIANTS[
                int.from_bytes(digest, "little") % len(DITCH_GRASS_VARIANTS)
            ]
            placed = _place_cluster_at(
                variant=variant,
                elevations=elevations,
                raster=raster,
                road_corridors=road_corridors,
                spec=spec,
                x=x,
                z=z,
                heading=heading,
                require_forest=False,
                minimum_forest_fraction=0.0,
                maximum_relief=ditch_maximum_relief,
                maximum_burial=ditch_maximum_burial,
                maximum_float=ditch_maximum_float,
                clearance=ditch_clearance,
            )
            if placed is None:
                ditch_grass_rejections += 1
                continue
            (
                model_path,
                placed_x,
                placed_y,
                placed_z,
                placed_heading,
                _variant_name,
                _relief,
                burial,
                floating,
            ) = placed
            emit(
                WorldObject(
                    next_id,
                    model_path,
                    placed_x,
                    placed_y,
                    placed_z,
                    placed_heading,
                )
            )
            next_id += 1
            ditch_grass_objects += 1
            ditch_grass_maximum_burial = max(ditch_grass_maximum_burial, burial)
            ditch_grass_maximum_float = max(ditch_grass_maximum_float, floating)

    # Linear OSM barriers become fitted reusable CWA models. Garden-cartography
    # enthusiasts can map every hedge in Europe, but the dedicated cap remains sovereign.
    progress(65, "Placing fences, walls and hedges")
    if bool(getattr(spec, "barriers_enabled", False)):
        barrier_warning_threshold = max(0, int(getattr(spec, "maximum_barrier_objects", 4000)))
        barrier_limit = _advisory_object_limit(barrier_warning_threshold, enabled=advisory_limits)
        barrier_length = max(2.0, float(getattr(spec, "barrier_segment_length", 6.0)))
        stock_hedge_models = tuple(getattr(spec, "stock_hedge_models", STOCK_HEDGE_MODELS))
        stock_wall_models = tuple(getattr(spec, "stock_wall_models", STOCK_WALL_MODELS))
        stock_metal_fence_models = tuple(getattr(spec, "stock_metal_fence_models", STOCK_METAL_FENCE_MODELS))
        for feature in sorted(dataset.barriers, key=lambda item: item.osm_key):
            subtype = feature.tags.get("barrier", "fence").casefold()
            subtype = "wall" if subtype in {"wall", "retaining_wall"} else "hedge" if subtype == "hedge" else "fence"
            metal_fence = subtype == "fence" and osm_fence_is_metal(feature.tags)
            # Every fence now uses a stock OFP/CWA model. Pick the non-metal
            # fence family once per mapped feature so a single OSM fence never
            # alternates between pasture and wire pieces from segment to segment.
            stock_fence_model = (
                stock_metal_fence_model(
                    f"{seed}:mapped-fence-style:{feature.osm_key}",
                    stock_metal_fence_models,
                )
                if metal_fence
                else stock_farmland_fence_model(
                    f"{seed}:mapped-fence-style:{feature.osm_key}"
                )
                if subtype == "fence"
                else ""
            )
            points = tuple(projection.to_world(point) for point in feature.points)
            fitted_length = STOCK_WALL_EFFECTIVE_LENGTH_METRES if subtype == "wall" else barrier_length
            for chunk_index, (x, z, heading, length, x0, z0, x1, z1) in enumerate(_line_chunks(points, fitted_length)):
                if barrier_objects >= barrier_limit:
                    break
                if not (0 <= x < spec.world_size and 0 <= z < spec.world_size):
                    barrier_rejections += 1
                    continue
                if _mask_at(raster.water, spec.cells, spec.world_size, x, z) or _mask_at(raster.buildings, spec.cells, spec.world_size, x, z):
                    barrier_rejections += 1
                    continue
                identity = f"{seed}:{feature.osm_key}:{chunk_index}"
                if subtype == "hedge":
                    model = stock_hedge_model(length, identity, stock_hedge_models)
                elif subtype == "wall":
                    model = stock_wall_model(identity, stock_wall_models)
                else:
                    model = stock_fence_model
                placed_x, placed_z = x, z
                placed_x0, placed_z0, placed_x1, placed_z1 = x0, z0, x1, z1
                placed_heading = heading
                if subtype == "wall":
                    # Stone walls must remain on their mapped boundary. Earlier
                    # versions offset every short wall chunk independently, which
                    # turned intersections into scattered freestanding slabs. A
                    # road crossing now creates a clean entrance-sized omission
                    # while all unaffected chunks stay exactly on the OSM line.
                    placed_heading = (heading + WALL_MODEL_HEADING_OFFSET_DEGREES) % 360.0
                    if line_intersects_road_corridors(
                        road_corridors,
                        (x0, z0),
                        (x1, z1),
                        clearance=0.35,
                    ):
                        barrier_rejections += 1
                        continue
                elif subtype == "hedge" or subtype == "fence":
                    heading_offset = (
                        HEDGE_MODEL_HEADING_OFFSET_DEGREES
                        if subtype == "hedge"
                        else METAL_FENCE_MODEL_HEADING_OFFSET_DEGREES
                        if metal_fence
                        else FARMLAND_FENCE_HEADING_OFFSET_DEGREES
                    )
                    placed_heading = (heading + heading_offset) % 360.0
                    shifted = offset_line_clear_of_roads(
                        dataset,
                        projection,
                        road_corridors,
                        x,
                        z,
                        x0,
                        z0,
                        x1,
                        z1,
                        minimum_distance=ROADSIDE_NUDGE_DISTANCE_METRES,
                        world_size=spec.world_size,
                    )
                    if shifted is None:
                        barrier_rejections += 1
                        continue
                    (
                        placed_x,
                        placed_z,
                        placed_x0,
                        placed_z0,
                        placed_x1,
                        placed_z1,
                    ) = shifted
                if subtype == "hedge":
                    y = _hedge_anchor_height(
                        elevations,
                        spec.cells,
                        spec.cell_size,
                        placed_x0,
                        placed_z0,
                        placed_x1,
                        placed_z1,
                        model_path=model,
                    )
                    pitch = 0.0
                else:
                    y, pitch = _infrastructure_anchor(
                        elevations,
                        spec.cells,
                        spec.cell_size,
                        placed_x0,
                        placed_z0,
                        placed_x1,
                        placed_z1,
                    )
                    if abs(pitch) > 38.0:
                        barrier_rejections += 1
                        continue
                placed_pitch = pitch
                emit(WorldObject(next_id, model, placed_x, y, placed_z, placed_heading, pitch_degrees=placed_pitch))
                next_id += 1
                barrier_objects += 1
                fence_objects += int(subtype == "fence")
                wall_objects += int(subtype == "wall")
                hedge_objects += int(subtype == "hedge")
            if barrier_objects >= barrier_limit:
                break

        # Add a sparse, deterministic set of stock rural fences around whole
        # farmland/meadow polygons. This deliberately uses only original OFP/CWA P3Ds;
        # generated barrier meshes are reserved for explicitly mapped barriers.
        # Road intersections are omitted, naturally leaving gate-sized openings.
        if barrier_objects < barrier_limit:
            farmland_features = tuple(
                feature
                for feature in sorted(dataset.farmland, key=lambda item: item.osm_key)
                if feature.polygons
                and (
                    feature.tags.get("landuse", "").casefold() in RURAL_FENCE_LANDUSES
                    or feature.tags.get("natural", "").casefold() in RURAL_FENCE_NATURALS
                )
            )
            selected_farmland_fences = _selected_farmland_fence_field_keys(
                farmland_features, seed, FARMLAND_FENCE_FIELD_PERCENT
            )
            rural_fence_grid_size = max(
                FARMLAND_FENCE_SEGMENT_LENGTH_METRES
                + 2.0 * FARMLAND_FENCE_DUPLICATE_DISTANCE_METRES,
                8.0,
            )
            rural_fence_grid: dict[
                tuple[int, int],
                list[tuple[str, str, tuple[float, float, float, float]]],
            ] = defaultdict(list)
            for feature in farmland_features:
                if feature.osm_key not in selected_farmland_fences:
                    continue
                # Pick the stock fence family once for the whole field. Do not
                # vary by segment: a perimeter is either rural/pasture fence or
                # wire fence, never an accidental alternating catalogue demo.
                field_fence_model = stock_farmland_fence_model(
                    f"{seed}:farmland-fence-style:{feature.osm_key}"
                )
                for polygon_index, polygon in enumerate(feature.polygons):
                    boundary = tuple(
                        projection.to_world(point) for point in polygon.outer[:-1]
                    )
                    if len(boundary) < 3:
                        continue
                    area, _cx, _cz = _polygon_area_centroid(boundary)
                    if area < FARMLAND_FENCE_MINIMUM_FIELD_AREA_M2:
                        continue
                    closed_boundary = boundary + (boundary[0],)
                    for chunk_index, (x, z, heading, _length, x0, z0, x1, z1) in enumerate(
                        _line_chunks(closed_boundary, FARMLAND_FENCE_SEGMENT_LENGTH_METRES)
                    ):
                        if barrier_objects >= barrier_limit:
                            break
                        if not (0.0 <= x < spec.world_size and 0.0 <= z < spec.world_size):
                            barrier_rejections += 1
                            continue
                        if (
                            _mask_at(raster.water, spec.cells, spec.world_size, x, z)
                            or _mask_at(raster.buildings, spec.cells, spec.world_size, x, z)
                        ):
                            barrier_rejections += 1
                            continue
                        if line_intersects_road_corridors(
                            road_corridors,
                            (x0, z0),
                            (x1, z1),
                            clearance=0.75,
                        ):
                            # Keep a practical entrance wherever a road/track
                            # meets the field instead of fencing straight across.
                            continue
                        model = field_fence_model
                        candidate_segment = (x0, z0, x1, z1)
                        grid_x = int(math.floor(x / rural_fence_grid_size))
                        grid_z = int(math.floor(z / rural_fence_grid_size))
                        duplicate = False
                        for ix in range(grid_x - 1, grid_x + 2):
                            if duplicate:
                                break
                            for iz in range(grid_z - 1, grid_z + 2):
                                for existing_field, existing_model, existing_segment in rural_fence_grid.get((ix, iz), ()):
                                    if existing_field == feature.osm_key:
                                        continue
                                    if existing_model.casefold() != model.casefold():
                                        continue
                                    if _rural_fence_segments_duplicate(candidate_segment, existing_segment):
                                        duplicate = True
                                        break
                                if duplicate:
                                    break
                        if duplicate:
                            # Adjacent selected fields can encode essentially the
                            # same shared boundary twice. If both chose the same
                            # stock fence family, keep only the first line.
                            continue
                        y, pitch = _infrastructure_anchor(
                            elevations,
                            spec.cells,
                            spec.cell_size,
                            x0,
                            z0,
                            x1,
                            z1,
                        )
                        if abs(pitch) > 32.0:
                            barrier_rejections += 1
                            continue
                        emit(
                            WorldObject(
                                next_id,
                                model,
                                x,
                                y,
                                z,
                                (heading + FARMLAND_FENCE_HEADING_OFFSET_DEGREES) % 360.0,
                                pitch_degrees=pitch,
                            )
                        )
                        next_id += 1
                        barrier_objects += 1
                        fence_objects += 1
                        rural_fence_grid[(grid_x, grid_z)].append(
                            (feature.osm_key, model, candidate_segment)
                        )
                    if barrier_objects >= barrier_limit:
                        break
                if barrier_objects >= barrier_limit:
                    break

    # Bridge-tagged roads use the original Nogova 30 m bridge module by default.
    # Procedural mode remains available as an opt-in fallback. Stock bridge ways
    # are omitted from the ordinary road-piece pass so the game does not receive
    # two overlapping road simulations across the same span.
    progress(66, "Placing bridge modules")
    if bool(getattr(spec, "bridges_enabled", False)):
        bridge_warning_threshold = max(0, int(getattr(spec, "maximum_bridge_objects", 1000)))
        bridge_limit = _advisory_object_limit(bridge_warning_threshold, enabled=advisory_limits)
        procedural_bridges = bool(getattr(spec, "procedural_bridges", True))
        module_length = (
            max(3.0, float(getattr(spec, "bridge_module_length", 30.0)))
            if procedural_bridges
            else NOGOVA_BRIDGE_MODULE_LENGTH_METRES
        )
        for feature, points in zip(dataset.roads, projected_road_polylines(dataset, projection)):
            # A mapped bridge over waterway=ditch stays an ordinary road. The
            # stock 30 m bridge is absurdly large for a farm/roadside ditch, and
            # the ordinary road fitter already follows the graded terrain cleanly.
            if road_bridge_crosses_ditch_only(feature, dataset, projection):
                continue
            source_chunks = _bridge_module_chunks(points, module_length)
            if not source_chunks:
                continue
            if not _road_needs_bridge_deck(
                feature, source_chunks, raster, spec, elevations
            ):
                continue
            # OSM bridge ways often begin only after the ordinary road has
            # already started descending toward the shoreline. Extend both stock
            # Nogova and procedural bridge objects back onto stable road/land.
            points = _extend_procedural_bridge_to_approach_plateaus(
                points, elevations, spec, module_length,
                feature=feature, dataset=dataset, projection=projection, raster=raster,
            )
            chunks = _bridge_module_chunks(points, module_length)
            if not chunks:
                continue
            bridge_segments += 1
            # Procedural bridges are authored as one world-local P3D covering
            # the complete extended span. Stock Nogova bridges still need their
            # fixed 30 m module chain. Never emit a partial bridge when the
            # object budget is tight.
            required_bridge_objects = 1 if procedural_bridges else len(chunks)
            if required_bridge_objects > bridge_limit - bridge_objects:
                bridge_rejections += 1
                continue
            width = max(4.0, road_width_metres(feature.tags) + 1.4)
            deck_clearance = max(
                MINIMUM_BRIDGE_TERRAIN_CLEARANCE_METRES,
                float(getattr(spec, "bridge_deck_clearance", 1.25)),
            )
            water_clearance = max(
                MINIMUM_BRIDGE_WATER_CLEARANCE_METRES,
                float(getattr(spec, "bridge_water_clearance", 18.0)),
            )
            total_length = sum(chunk[3] for chunk in chunks)
            feature_start = (chunks[0][4], chunks[0][5])
            feature_end = (chunks[-1][6], chunks[-1][7])
            start_ground = _sample_elevation(
                elevations, spec.cells, spec.cell_size, *feature_start
            )
            end_ground = _sample_elevation(
                elevations, spec.cells, spec.cell_size, *feature_end
            )
            if procedural_bridges:
                # Resolve the exact deck height after footprint/water checks.
                deck_start = start_ground
                deck_end = end_ground
                generated_width = max(3.5, round(width * 10.0) / 10.0)
                # One generated structure spans from the extended stable-road
                # start to end. Most mapped bridges are straight; using the full
                # chord avoids seams, duplicate collision boundaries and the
                # loading cost of a chain of almost-identical little P3Ds.
                span_dx = feature_end[0] - feature_start[0]
                span_dz = feature_end[1] - feature_start[1]
                span_length = math.hypot(span_dx, span_dz)
                if span_length < 3.0:
                    bridge_rejections += 1
                    continue
                span_heading = math.degrees(math.atan2(span_dx, span_dz)) % 360.0
                span_x = (feature_start[0] + feature_end[0]) * 0.5
                span_z = (feature_start[1] + feature_end[1]) * 0.5
                model_length = max(3.0, round(span_length * 10.0) / 10.0)
                placement_chunks = ((
                    span_x, span_z, span_heading, span_length,
                    feature_start[0], feature_start[1], feature_end[0], feature_end[1],
                ),)
                model_lengths = (model_length,)
                total_length = span_length
                half_width = generated_width * 0.5 + GENERATED_BRIDGE_RAIL_OVERHANG_METRES
                vertical_depth = GENERATED_BRIDGE_MAXIMUM_DEPTH_METRES
            else:
                generated_width = width
                placement_chunks = chunks
                model_lengths = (NOGOVA_BRIDGE_MODULE_LENGTH_METRES,) * len(chunks)
                half_width = max(width * 0.5, NOGOVA_BRIDGE_HALF_WIDTH_METRES)
                vertical_depth = -NOGOVA_BRIDGE_LOWEST_POINT_METRES

            # Evaluate every complete rotated module footprint against both
            # possible RVW4 triangle diagonals. Sparse centre/edge or bilinear
            # samples can miss a narrow wet cell or saddle-shaped terrain peak.
            actual_module_segments = tuple(
                (
                    x - math.sin(math.radians(heading)) * model_length * 0.5,
                    z - math.cos(math.radians(heading)) * model_length * 0.5,
                    x + math.sin(math.radians(heading)) * model_length * 0.5,
                    z + math.cos(math.radians(heading)) * model_length * 0.5,
                )
                for (x, z, heading, _length, _x0, _z0, _x1, _z1), model_length
                in zip(placement_chunks, model_lengths)
            )
            bridge_footprints = tuple(
                _line_footprint_polygon(
                    x0,
                    z0,
                    x1,
                    z1,
                    half_width=half_width,
                    end_margin=0.05,
                )
                for x0, z0, x1, z1 in actual_module_segments
            )
            maximum_support = max(
                _maximum_polygon_elevation(
                    elevations, spec.cells, spec.cell_size, footprint
                )
                for footprint in bridge_footprints
            )
            minimum_support = min(
                _minimum_polygon_elevation(
                    elevations, spec.cells, spec.cell_size, footprint
                )
                for footprint in bridge_footprints
            )
            spans_water = any(
                _polygon_overlaps_mask(
                    raster.water, spec.cells, spec.world_size, footprint
                )
                for footprint in bridge_footprints
            )
            if procedural_bridges:
                # The span endpoints were already extended to the stable upper
                # road before the shoreline descent. Use those endpoint terrain
                # levels directly and hold every generated module perfectly flat.
                # If opposite banks differ, the higher road level wins; terrain
                # under the lower approach can be supported by the bridge rather
                # than forcing the deck to slope down toward the water.
                centreline_support = max(
                    _sample_elevation(
                        elevations, spec.cells, spec.cell_size, sx, sz
                    )
                    for chunk in placement_chunks
                    for sx, sz in (
                        (chunk[4], chunk[5]),
                        (chunk[0], chunk[1]),
                        (chunk[6], chunk[7]),
                    )
                )
                road_approach_level = max(start_ground, end_ground, centreline_support)
                if spans_water or minimum_support < float(spec.sea_level):
                    road_approach_level = max(
                        road_approach_level, float(spec.sea_level) + 0.35
                    )
                # Keep the earlier land inset, but never let later modules vanish
                # into a ridge or coarse terrain triangle. The whole procedural
                # bridge remains perfectly flat and its roadway floats above the
                # highest terrain under the complete extended footprint.
                fly_clearance = deck_clearance
                roadway_level = max(
                    road_approach_level,
                    maximum_support + fly_clearance,
                )
                deck_start = deck_end = (
                    roadway_level - GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES
                )
            else:
                # The stock Nogova bridge is itself a proper game road/bridge
                # model. Do not slope individual 30 m modules down toward the
                # beach. Use the highest road-centre terrain sampled across the
                # complete land-extended span and keep every module at exactly
                # one elevation. This mirrors the successful procedural-bridge
                # approach logic while letting the original BIS model provide
                # its own roadway/collision/rendering behaviour.
                centreline_support = max(
                    _sample_elevation(
                        elevations, spec.cells, spec.cell_size, sx, sz
                    )
                    for chunk in placement_chunks
                    for sx, sz in (
                        (chunk[4], chunk[5]),
                        (chunk[0], chunk[1]),
                        (chunk[6], chunk[7]),
                    )
                )
                roadway_level = max(start_ground, end_ground, centreline_support)
                deck_level = roadway_level + NOGOVA_BRIDGE_APPROACH_OFFSET_METRES
                if spans_water or minimum_support < float(spec.sea_level):
                    deck_level = max(
                        deck_level,
                        float(spec.sea_level) + NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES,
                    )
                deck_start = deck_end = deck_level
            travelled = 0.0
            bridge_plan: list[tuple[str, float, float, float, float, float]] = []
            chunk_count = len(placement_chunks)
            for index, (chunk, model_length) in enumerate(zip(placement_chunks, model_lengths)):
                x, z, heading, length, _x0, _z0, _x1, _z1 = chunk
                fraction0 = travelled / max(0.01, total_length)
                fraction1 = (travelled + length) / max(0.01, total_length)
                y0 = deck_start + (deck_end - deck_start) * fraction0
                y1 = deck_start + (deck_end - deck_start) * fraction1
                pitch = math.degrees(math.atan2(y1 - y0, max(0.01, length)))
                y = (y0 + y1) * 0.5
                travelled += length
                if abs(pitch) > 28.0:
                    bridge_rejections += 1
                    bridge_plan.clear()
                    break
                if procedural_bridges:
                    model = infrastructure_library.bridge_model(
                        "single", generated_width, model_length
                    )
                else:
                    model = NOGOVA_BRIDGE_MODEL
                bridge_plan.append((model, x, y, z, heading, pitch))
            for model, x, y, z, heading, pitch in bridge_plan:
                emit(WorldObject(
                    next_id,
                    model,
                    x,
                    y,
                    z,
                    heading,
                    pitch_degrees=pitch,
                ))
                next_id += 1
                bridge_objects += 1
            if bridge_objects >= bridge_limit:
                break

    progress(67, "Placing meadow grass, rural vegetation, wetland reeds and rocks")
    # Structured rural vegetation reuses a small cluster library. Rows stay rows;
    # orchards no longer become a vaguely green field and call the matter settled.
    rural_enabled = bool(getattr(spec, "rural_vegetation_enabled", False))
    wetland_enabled = bool(getattr(spec, "wetland_reeds_enabled", False))
    meadow_enabled = bool(getattr(spec, "meadow_grass_enabled", False))
    # Legacy field-hay scattering is disabled. Hay bales are barn-only
    # settlement clutter, even if an old profile still passes --haybales.
    haybales_enabled = False
    if rural_enabled or wetland_enabled or meadow_enabled or haybales_enabled:
        rural_warning_threshold = max(0, int(getattr(spec, "maximum_rural_vegetation_objects", 3000)))
        rural_limit = (
            _advisory_object_limit(getattr(spec, "maximum_rural_vegetation_objects", 3000), enabled=advisory_limits)
            if rural_enabled else 0
        )
        rural_spacing = max(10.0, float(getattr(spec, "rural_vegetation_spacing", 28.0)))
        variants = {variant.name: variant for variant in RURAL_VEGETATION_VARIANTS}
        meadow_warning_threshold = max(0, int(getattr(spec, "maximum_meadow_grass_objects", 20000)))
        meadow_limit = _advisory_object_limit(meadow_warning_threshold, enabled=advisory_limits)
        meadow_spacing = max(10.0, float(getattr(spec, "meadow_grass_spacing", 24.0)))
        if meadow_enabled and meadow_limit:
            meadow_variant = DITCH_GRASS_VARIANTS[0]
            for feature in sorted(dataset.farmland, key=lambda item: item.osm_key):
                if feature.tags.get("landuse", "").casefold() != "meadow":
                    continue
                for polygon_index, polygon in enumerate(feature.polygons):
                    projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
                    for x, z, heading in _polygon_grid_candidates(
                        projected,
                        meadow_spacing,
                        f"{seed}:meadow-grass:{feature.osm_key}:{polygon_index}",
                        jitter_fraction=0.80,
                        heading_jitter_degrees=360.0,
                    ):
                        if meadow_grass_objects >= meadow_limit:
                            break
                        placed = _place_cluster_at(
                            variant=meadow_variant, elevations=elevations, raster=raster,
                            road_corridors=road_corridors, spec=spec, x=x, z=z, heading=heading,
                            require_forest=False, minimum_forest_fraction=0.0,
                            maximum_relief=8.0, maximum_burial=0.6, maximum_float=0.8,
                            clearance=0.03,
                        )
                        if placed is None:
                            meadow_grass_rejections += 1
                            if len(meadow_grass_rejection_positions) < 20000:
                                meadow_grass_rejection_positions.append((x, z))
                            continue
                        model, px, py, pz, ph, _vn, _relief, _burial, _floating = placed
                        emit(WorldObject(next_id, model, px, py, pz, ph)); next_id += 1
                        meadow_grass_objects += 1
                        meadow_grass_positions.append((px, pz))
                    if meadow_grass_objects >= meadow_limit:
                        break
                if meadow_grass_objects >= meadow_limit:
                    break
        haybale_warning_threshold = max(0, int(getattr(spec, "maximum_haybale_objects", 800)))
        haybale_limit = _advisory_object_limit(haybale_warning_threshold, enabled=advisory_limits)
        haybale_spacing = max(40.0, float(getattr(spec, "haybale_spacing", 110.0)))
        haybale_field_percent = min(
            100.0, max(0.0, float(getattr(spec, "haybale_field_percent", HAYBALE_FIELD_PERCENT)))
        )
        if haybales_enabled and haybale_limit:
            farmland_features = tuple(
                feature
                for feature in sorted(dataset.farmland, key=lambda item: item.osm_key)
                if feature.tags.get("landuse", "").casefold() == "farmland" and feature.polygons
            )
            haybale_fields_total = len(farmland_features)
            selected_field_keys = _selected_haybale_field_keys(
                farmland_features, seed, haybale_field_percent
            )
            haybale_fields_selected = len(selected_field_keys)
            haybale_candidates: list[tuple[str, float, float, float, tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]]] = []
            for feature in farmland_features:
                if feature.osm_key not in selected_field_keys:
                    continue
                field_candidates: list[tuple[str, float, float, float, tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]]] = []
                accepted_field_candidates: list[tuple[str, float, float, float, tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]]] = []
                for polygon_index, polygon in enumerate(feature.polygons):
                    outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
                    if len(outer) < 3:
                        continue
                    holes = tuple(
                        tuple(projection.to_world(point) for point in hole[:-1])
                        for hole in polygon.holes
                        if len(hole) >= 4
                    )
                    for candidate_index, (x, z, heading) in enumerate(_polygon_grid_candidates(
                        outer,
                        haybale_spacing,
                        f"{seed}:haybale:{feature.osm_key}:{polygon_index}",
                        jitter_fraction=0.90,
                        heading_jitter_degrees=360.0,
                    )):
                        if holes and not _polygon_contains_with_holes((x, z), outer, holes):
                            continue
                        key = f"{feature.osm_key}:{polygon_index}:{candidate_index}"
                        item = (key, x, z, heading, outer, holes)
                        field_candidates.append(item)
                        density_digest = hashlib.blake2s(
                            f"{seed}:haybale-density:{key}".encode("utf-8"), digest_size=2
                        ).digest()
                        if int.from_bytes(density_digest, "little") < int(65536 * HAYBALE_ANCHOR_ACCEPTANCE):
                            accepted_field_candidates.append(item)
                # A field selected for hay should visibly have hay whenever it has
                # at least one geometric candidate.  If the density roll rejected
                # all anchors, keep the deterministic best candidate for that field.
                if not accepted_field_candidates and field_candidates:
                    accepted_field_candidates.append(
                        min(
                            field_candidates,
                            key=lambda item: (
                                hashlib.blake2s(
                                    f"{seed}:haybale-density:{item[0]}".encode("utf-8"),
                                    digest_size=8,
                                ).digest(),
                                item[0],
                            ),
                        )
                    )
                haybale_candidates.extend(accepted_field_candidates)

            haybale_candidates.sort(
                key=lambda item: (
                    hashlib.blake2s(
                        f"{seed}:haybale-order:{item[0]}".encode("utf-8"), digest_size=8
                    ).digest(),
                    item[0],
                )
            )
            for key, x, z, heading, outer, holes in haybale_candidates:
                if haybale_objects >= haybale_limit:
                    break
                for member_x, member_z, member_heading in _haybale_cluster_members(
                    seed, key, x, z, heading
                ):
                    if haybale_objects >= haybale_limit:
                        break
                    if not _polygon_contains_with_holes((member_x, member_z), outer, holes):
                        haybale_rejections += 1
                        continue
                    if (
                        _mask_at(raster.water, spec.cells, spec.world_size, member_x, member_z)
                        or _mask_at(raster.roads, spec.cells, spec.world_size, member_x, member_z)
                        or _mask_at(raster.buildings, spec.cells, spec.world_size, member_x, member_z)
                        or forest_block_intersects_road_corridors(
                            road_corridors, member_x, member_z, block_size=5.0
                        )
                    ):
                        haybale_rejections += 1
                        continue
                    supports = _square_elevation_samples(
                        elevations, spec.cells, spec.cell_size, member_x, member_z, 2.5
                    )
                    if max(supports) - min(supports) > 0.45:
                        haybale_rejections += 1
                        continue
                    fitted = _terrain_fit_anchor(
                        supports, clearance=0.03, maximum_burial=0.18, maximum_float=0.18
                    )
                    if fitted is None:
                        haybale_rejections += 1
                        continue
                    anchor, _burial, _floating = fitted
                    emit(
                        WorldObject(
                            next_id, HAYBALE_MODEL, member_x, anchor, member_z, member_heading
                        )
                    )
                    next_id += 1
                    haybale_objects += 1

        for feature in sorted(dataset.tree_rows, key=lambda item: item.osm_key):
            points = tuple(projection.to_world(point) for point in feature.points)
            for index, (x, z, heading, _length, _x0, _z0, _x1, _z1) in enumerate(_line_chunks(points, rural_spacing, endpoint_trim=2.0)):
                if tree_row_objects + orchard_objects + vineyard_objects + scrub_objects + rural_rock_objects >= rural_limit:
                    break
                placed = _place_cluster_at(
                    variant=variants["tree_row"], elevations=elevations, raster=raster,
                    road_corridors=road_corridors, spec=spec, x=x, z=z, heading=heading,
                    require_forest=False, minimum_forest_fraction=0.0,
                    maximum_relief=24.0, maximum_burial=1.0, maximum_float=1.0,
                    clearance=0.03, avoid_roads=False,
                )
                if placed is None:
                    rural_vegetation_rejections += 1
                    continue
                model, px, py, pz, ph, _vn, _relief, _burial, _floating = placed
                emit(WorldObject(next_id, model, px, py, pz, ph)); next_id += 1
                tree_row_objects += 1

        wetland_warning_threshold = max(0, int(getattr(spec, "maximum_wetland_reed_objects", 100000)))
        wetland_limit = _advisory_object_limit(wetland_warning_threshold, enabled=advisory_limits)
        wetland_spacing = max(6.0, float(getattr(spec, "wetland_reed_spacing", 18.0)))
        wetland_maximum_relief = max(0.0, float(getattr(spec, "wetland_reed_maximum_relief", 4.0)))
        wetland_maximum_float = max(0.0, float(getattr(spec, "wetland_reed_maximum_float", 1.0)))
        wetland_clearance = float(getattr(spec, "wetland_reed_ground_clearance", 0.03))
        wetland_models = tuple(getattr(spec, "wetland_reed_models", (r"o\tree\dd_rakosi.p3d", r"o\tree\dd_rakosi02.p3d")))

        for feature in sorted(dataset.rural_vegetation, key=lambda item: item.osm_key):
            natural = feature.tags.get("natural", "").casefold()
            landuse = feature.tags.get("landuse", "").casefold()
            rural_kind = feature.tags.get("rural_kind", "").casefold()
            if not natural and rural_kind in {"scrub", "bare_rock", "rock", "scree", "wetland"}:
                natural = rural_kind
            if not landuse and rural_kind in {"orchard", "vineyard"}:
                landuse = rural_kind
            if natural == "wetland":
                category = "wetland"
            elif natural in {"bare_rock", "rock", "scree"}:
                category = "rock"
            elif natural == "scrub":
                category = "scrub"
            elif landuse == "orchard":
                category = "orchard"
            elif landuse == "vineyard":
                category = "vineyard"
            else:
                continue
            for polygon_index, polygon in enumerate(feature.polygons):
                projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
                candidate_spacing = (
                    wetland_spacing
                    if category == "wetland"
                    else max(16.0, rural_spacing)
                    if category == "scrub"
                    else rural_spacing
                )
                for candidate_index, (x, z, heading) in enumerate(
                    _polygon_grid_candidates(
                        projected,
                        candidate_spacing,
                        f"{seed}:{feature.osm_key}:{polygon_index}",
                        jitter_fraction=0.70 if category == "scrub" else 0.0,
                        heading_jitter_degrees=180.0 if category == "scrub" else 0.0,
                    )
                ):
                    if category == "wetland":
                        if not wetland_enabled or wetland_reed_objects >= wetland_limit:
                            break
                        if (
                            _mask_at(raster.roads, spec.cells, spec.world_size, x, z)
                            or _mask_at(raster.buildings, spec.cells, spec.world_size, x, z)
                            or forest_block_intersects_road_corridors(road_corridors, x, z, block_size=4.0)
                        ):
                            wetland_reed_rejections += 1
                            continue
                        supports = _square_elevation_samples(
                            elevations, spec.cells, spec.cell_size, x, z, 4.0
                        )
                        relief = max(supports) - min(supports)
                        if relief > wetland_maximum_relief:
                            wetland_reed_rejections += 1
                            continue
                        point_supports = _triangle_elevation_bounds(
                            elevations, spec.cells, spec.cell_size, x, z
                        )
                        reed_fit = _non_buried_vegetation_fit(
                            point_supports,
                            clearance=wetland_clearance,
                            maximum_float=wetland_maximum_float,
                        )
                        if reed_fit is None:
                            wetland_reed_rejections += 1
                            continue
                        anchor, _reed_float = reed_fit
                        digest = hashlib.blake2s(
                            f"{seed}:wetland-reed:{feature.osm_key}:{polygon_index}:{candidate_index}".encode("utf-8"),
                            digest_size=4,
                        ).digest()
                        model = wetland_models[int.from_bytes(digest[:2], "little") % len(wetland_models)]
                        reed_heading = float(int.from_bytes(digest[2:], "little") % 360)
                        emit(WorldObject(next_id, model, x, anchor, z, reed_heading))
                        next_id += 1
                        wetland_reed_objects += 1
                        continue

                    if tree_row_objects + orchard_objects + vineyard_objects + scrub_objects + rural_rock_objects >= rural_limit:
                        break
                    if category == "rock":
                        supports = _square_elevation_samples(elevations, spec.cells, spec.cell_size, x, z, 9.0)
                        fitted = _terrain_fit_anchor(supports, clearance=0.03, maximum_burial=1.0, maximum_float=1.0)
                        if fitted is None or _mask_at(raster.water, spec.cells, spec.world_size, x, z):
                            rural_vegetation_rejections += 1
                            continue
                        anchor, _burial, _floating = fitted
                        model = STOCK_STONE_MODELS[candidate_index % len(STOCK_STONE_MODELS)]
                        emit(WorldObject(next_id, model, x, anchor, z, heading)); next_id += 1
                        rural_rock_objects += 1
                        continue
                    if category == "scrub":
                        variant = (
                            DITCH_GRASS_VARIANTS[0]
                            if candidate_index % 4 == 0
                            else variants["scrub_patch"]
                        )
                    else:
                        variant = variants[{"orchard": "orchard_row", "vineyard": "vineyard_row"}[category]]
                    placed = _place_cluster_at(
                        variant=variant, elevations=elevations, raster=raster,
                        road_corridors=road_corridors, spec=spec, x=x, z=z, heading=heading,
                        require_forest=False, minimum_forest_fraction=0.0,
                        maximum_relief=28.0, maximum_burial=1.0, maximum_float=1.0,
                        clearance=0.03,
                    )
                    if placed is None:
                        rural_vegetation_rejections += 1
                        continue
                    model, px, py, pz, ph, _vn, _relief, _burial, _floating = placed
                    emit(WorldObject(next_id, model, px, py, pz, ph)); next_id += 1
                    orchard_objects += int(category == "orchard")
                    vineyard_objects += int(category == "vineyard")
                    scrub_objects += int(category == "scrub")

    progress(69, "Placing mapped trees and utility infrastructure")
    mapped_tree_warning_threshold = max(0, int(getattr(spec, "maximum_mapped_tree_objects", 5000)))
    mapped_tree_limit = _advisory_object_limit(mapped_tree_warning_threshold, enabled=advisory_limits)
    mapped_tree_clearance = max(0.0, float(getattr(spec, "mapped_tree_ground_clearance", 0.04)))
    for mapped_tree_index, feature in enumerate(sorted(dataset.individual_trees, key=lambda item: item.osm_key)):
        if mapped_tree_objects >= mapped_tree_limit:
            break
        if mapped_tree_index % 2 == 1:
            mapped_tree_rejections += 1
            continue
        x, z = projection.to_world(feature.point)
        if not (0.0 <= x < spec.world_size and 0.0 <= z < spec.world_size):
            mapped_tree_rejections += 1
            continue
        if (
            _mask_at(raster.water, spec.cells, spec.world_size, x, z)
            or _mask_at(raster.buildings, spec.cells, spec.world_size, x, z)
            or forest_block_intersects_road_corridors(road_corridors, x, z, block_size=2.0)
        ):
            mapped_tree_rejections += 1
            continue
        leaf_type = feature.tags.get("leaf_type", "").casefold()
        species_text = " ".join((feature.tags.get("species", ""), feature.tags.get("genus", ""))).casefold()
        active_forest_model = str(getattr(spec, "forest_tree_model", "")).casefold()
        if active_forest_model.startswith(r"o\tree\les_nw_jehl_"):
            models = NOGOVA_PINE_INDIVIDUAL_TREE_MODELS
        elif active_forest_model.startswith(r"o\tree\les_nw_"):
            models = NOGOVA_LEAF_INDIVIDUAL_TREE_MODELS
        elif leaf_type == "needleleaved" or any(word in species_text for word in ("picea", "pinus", "abies", "spruce", "pine", "fir")):
            models = OSM_CONIFER_TREE_MODELS
        elif leaf_type == "broadleaved" or species_text:
            models = OSM_BROADLEAF_TREE_MODELS
        else:
            models = OSM_INDIVIDUAL_TREE_MODELS
        digest = hashlib.blake2s(f"{seed}:mapped-tree:{feature.osm_key}".encode("utf-8"), digest_size=4).digest()
        model = models[int.from_bytes(digest[:2], "little") % len(models)]
        heading = float(int.from_bytes(digest[2:], "little") % 360)
        tree_fit = _rooted_tree_fit(
            _triangle_elevation_bounds(
                elevations, spec.cells, spec.cell_size, x, z
            ),
            root_sink=max(individual_tree_root_sink, mapped_tree_clearance),
            maximum_burial=individual_tree_maximum_burial,
        )
        if tree_fit is None:
            mapped_tree_rejections += 1
            continue
        anchor, _tree_burial = tree_fit
        emit(WorldObject(next_id, model, x, anchor, z, heading))
        next_id += 1
        mapped_tree_objects += 1

    utility_warning_threshold = max(0, int(getattr(spec, "maximum_utility_objects", 3000)))
    utility_limit = _advisory_object_limit(utility_warning_threshold, enabled=advisory_limits)
    utility_clearance = max(0.0, float(getattr(spec, "utility_ground_clearance", 0.05)))
    utility_sizes = {"power_pole": 1.0, "power_tower": 9.0, "water_tower": 6.5}
    for feature in sorted(dataset.utility_points, key=lambda item: item.osm_key):
        if utility_objects >= utility_limit:
            break
        kind = feature.tags.get("utility", "").casefold()
        if kind not in utility_sizes:
            continue
        x, z = projection.to_world(feature.point)
        if not (0.0 <= x < spec.world_size and 0.0 <= z < spec.world_size):
            utility_rejections += 1
            continue
        footprint = utility_sizes[kind]
        if kind == "power_pole" and forest_block_intersects_road_corridors(road_corridors, x, z, block_size=footprint):
            x, z = nudge_point_away_from_road(
                dataset, projection, x, z, distance=2.0, world_size=spec.world_size
            )
        if (
            _mask_at(raster.water, spec.cells, spec.world_size, x, z)
            or _mask_at(raster.buildings, spec.cells, spec.world_size, x, z)
            or forest_block_intersects_road_corridors(road_corridors, x, z, block_size=footprint)
        ):
            utility_rejections += 1
            continue
        _minimum, maximum = _square_elevation_extrema(
            elevations, spec.cells, spec.cell_size, x, z, footprint
        )
        model = infrastructure_library.utility_model(kind)
        heading_digest = hashlib.blake2s(f"{seed}:utility:{feature.osm_key}".encode("utf-8"), digest_size=2).digest()
        heading = float(int.from_bytes(heading_digest, "little") % 360)
        emit(WorldObject(next_id, model, x, maximum + utility_clearance, z, heading))
        next_id += 1
        utility_objects += 1

    advisory_counts = (
        ("legacy road object", road_count, road_warning_threshold),
        ("sidewalk object", sidewalk_objects, sidewalk_warning_threshold),
        ("street furniture object", street_furniture_objects, street_furniture_warning_threshold),
        ("primary forest object", forest_count, forest_warning_threshold),
        ("forest undergrowth object", forest_undergrowth_objects, getattr(spec, "forest_undergrowth_maximum_objects", 120000)),
        ("steep-hill bush object", steep_hill_bush_objects, getattr(spec, "maximum_steep_hill_bush_objects", 80000)),
        ("forest border object", forest_border_objects, getattr(spec, "forest_border_maximum_objects", 2000)),
        ("extra forest single-tree object", forest_single_tree_objects, extra_single_warning_threshold if 'extra_single_warning_threshold' in locals() else getattr(spec, "maximum_forest_single_tree_objects", 1000)),
        ("ditch grass object", ditch_grass_objects, getattr(spec, "maximum_ditch_grass_objects", 2000)),
        ("barrier object", barrier_objects, getattr(spec, "maximum_barrier_objects", 4000)),
        ("bridge object", bridge_objects, getattr(spec, "maximum_bridge_objects", 1000)),
        ("residential infill building", sum(1 for plan in building_placement_plans if plan.synthetic_infill), getattr(spec, "maximum_residential_infill_buildings", 1500)),
        ("rural vegetation object", tree_row_objects + orchard_objects + vineyard_objects + scrub_objects + rural_rock_objects, getattr(spec, "maximum_rural_vegetation_objects", 3000)),
        ("meadow grass object", meadow_grass_objects, getattr(spec, "maximum_meadow_grass_objects", 20000)),
        ("hay bale object", haybale_objects, getattr(spec, "maximum_haybale_objects", 800)),
        ("wetland reed object", wetland_reed_objects, getattr(spec, "maximum_wetland_reed_objects", 100000)),
        ("rocky forest object", rocky_forest_objects, getattr(spec, "maximum_rocky_forest_objects", 1200)),
        ("mapped tree object", mapped_tree_objects, getattr(spec, "maximum_mapped_tree_objects", 5000)),
        ("utility object", utility_objects, getattr(spec, "maximum_utility_objects", 3000)),
    )
    if advisory_limits:
        for label, generated_count, configured_threshold in advisory_counts:
            warning = _object_threshold_warning(label, generated_count, configured_threshold)
            if warning is not None:
                progress(70, warning)

    (
        vegetation_audit_tree_objects,
        vegetation_audit_cluster_tree_proxies,
        vegetation_audit_cluster_bush_proxies,
        vegetation_audit_violations,
        vegetation_audit_maximum_tree_float,
        vegetation_audit_maximum_bush_float,
    ) = _audit_vegetation_grounding(objects, elevations, spec)

    return ObjectGenerationResult(
        objects=tuple(objects),
        road_objects=road_count,
        building_objects=building_count,
        forest_objects=forest_count,
        road_objects_truncated=road_truncated,
        building_objects_truncated=building_truncated,
        forest_objects_truncated=forest_truncated,
        forest_road_rejections=forest_road_rejections,
        maximum_building_grounding_raise=maximum_building_grounding_raise,
        maximum_building_pad_relief=maximum_building_pad_relief,
        maximum_building_foundation_depth=maximum_building_foundation_depth,
        building_foundation_rejections=building_foundation_rejections,
        building_interior_fallbacks=building_interior_fallbacks,
        building_fully_submerged_rejections=building_fully_submerged_rejections,
        building_road_nudges=building_road_nudges,
        maximum_forest_grounding_raise=maximum_forest_grounding_raise,
        forest_slope_rejections=forest_slope_rejections,
        maximum_forest_relief=maximum_forest_relief,
        forest_block_objects=forest_block_objects,
        forest_hillside_tree_objects=forest_hillside_tree_objects,
        forest_hillside_fallback_blocks=forest_hillside_fallback_blocks,
        forest_hillside_unfilled_blocks=forest_hillside_unfilled_blocks,
        forest_hillside_candidate_rejections=forest_hillside_candidate_rejections,
        maximum_hillside_tree_relief=maximum_hillside_tree_relief,
        forest_everon_steep_objects=forest_everon_steep_objects,
        forest_sunk_polygon_objects=forest_sunk_polygon_objects,
        forest_everon_steep_rejections=forest_everon_steep_rejections,
        forest_cluster_objects=forest_cluster_objects,
        forest_cluster_rejections=forest_cluster_rejections,
        forest_cluster_maximum_burial=forest_cluster_maximum_burial,
        forest_cluster_maximum_float=forest_cluster_maximum_float,
        forest_cluster_variant_counts=tuple(sorted(forest_cluster_variant_counts.items())),
        forest_undergrowth_objects=forest_undergrowth_objects,
        forest_undergrowth_rejections=forest_undergrowth_rejections,
        forest_undergrowth_maximum_burial=forest_undergrowth_maximum_burial,
        forest_undergrowth_maximum_float=forest_undergrowth_maximum_float,
        steep_hill_bush_objects=steep_hill_bush_objects,
        steep_hill_bush_rejections=steep_hill_bush_rejections,
        wetland_reed_objects=wetland_reed_objects,
        wetland_reed_rejections=wetland_reed_rejections,
        forest_border_objects=forest_border_objects,
        forest_border_rejections=forest_border_rejections,
        forest_border_maximum_burial=forest_border_maximum_burial,
        forest_border_maximum_float=forest_border_maximum_float,
        forest_single_tree_objects=forest_single_tree_objects,
        forest_gap_infill_tree_objects=forest_gap_infill_tree_objects,
        ditch_grass_objects=ditch_grass_objects,
        ditch_grass_rejections=ditch_grass_rejections,
        ditch_grass_maximum_burial=ditch_grass_maximum_burial,
        ditch_grass_maximum_float=ditch_grass_maximum_float,
        maximum_forest_burial=maximum_forest_burial,
        maximum_forest_float=maximum_forest_float,
        barrier_objects=barrier_objects,
        fence_objects=fence_objects,
        wall_objects=wall_objects,
        hedge_objects=hedge_objects,
        barrier_rejections=barrier_rejections,
        bridge_objects=bridge_objects,
        bridge_segments=bridge_segments,
        bridge_rejections=bridge_rejections,
        residential_infill_objects=sum(1 for plan in building_placement_plans if plan.synthetic_infill),
        residential_infill_areas=len({plan.osm_key.rsplit("/", 1)[0] for plan in building_placement_plans if plan.synthetic_infill}),
        tree_row_objects=tree_row_objects,
        orchard_objects=orchard_objects,
        vineyard_objects=vineyard_objects,
        scrub_objects=scrub_objects,
        rural_rock_objects=rural_rock_objects,
        rural_vegetation_rejections=rural_vegetation_rejections,
        meadow_grass_objects=meadow_grass_objects,
        meadow_grass_rejections=meadow_grass_rejections,
        haybale_objects=haybale_objects,
        haybale_rejections=haybale_rejections,
        haybale_fields_total=haybale_fields_total,
        haybale_fields_selected=haybale_fields_selected,
        meadow_grass_positions=tuple(meadow_grass_positions),
        meadow_grass_rejection_positions=tuple(meadow_grass_rejection_positions),
        rocky_forest_objects=rocky_forest_objects,
        rocky_forest_rejections=rocky_forest_rejections,
        mapped_tree_objects=mapped_tree_objects,
        mapped_tree_rejections=mapped_tree_rejections,
        utility_objects=utility_objects,
        utility_rejections=utility_rejections,
        sidewalk_objects=sidewalk_objects,
        street_furniture_objects=street_furniture_objects,
        street_light_objects=street_light_objects,
        street_bench_objects=street_bench_objects,
        street_bin_objects=street_bin_objects,
        street_noticeboard_objects=street_noticeboard_objects,
        street_bicycle_objects=street_bicycle_objects,
        street_bus_shelter_objects=street_bus_shelter_objects,
        street_tree_objects=street_tree_objects,
        urban_detail_rejections=urban_detail_rejections,
        vegetation_audit_tree_objects=vegetation_audit_tree_objects,
        vegetation_audit_cluster_tree_proxies=vegetation_audit_cluster_tree_proxies,
        vegetation_audit_cluster_bush_proxies=vegetation_audit_cluster_bush_proxies,
        vegetation_audit_violations=vegetation_audit_violations,
        vegetation_audit_maximum_tree_float=vegetation_audit_maximum_tree_float,
        vegetation_audit_maximum_bush_float=vegetation_audit_maximum_bush_float,
        model_usage=tuple(sorted(model_usage.items(), key=lambda item: item[0].casefold())),
        surface_forest_positions=tuple(surface_forest_positions),
        surface_rock_positions=tuple(surface_rock_positions),
    )


def raster_counts(raster: OsmRaster) -> dict[str, int]:
    return {
        "water": sum(raster.water),
        "forest": sum(raster.forest),
        "farmland": sum(raster.farmland),
        "urban": sum(raster.urban),
        "roads": sum(raster.roads),
        "buildings": sum(raster.buildings),
    }


def _geography_preview_image(raster: OsmRaster) -> Image.Image:
    colours = {
        "base": (92, 112, 78),
        "water": (35, 92, 156),
        "forest": (38, 82, 43),
        "farmland": (154, 145, 79),
        "urban": (150, 145, 137),
        "roads": (49, 48, 45),
        "buildings": (173, 74, 62),
    }
    image = Image.new("RGB", (raster.cells, raster.cells), colours["base"])
    pixels = image.load()
    for y in range(raster.cells):
        for x in range(raster.cells):
            index = (raster.cells - 1 - y) * raster.cells + x
            colour = colours["base"]
            if raster.farmland[index]:
                colour = colours["farmland"]
            if raster.forest[index]:
                colour = colours["forest"]
            if raster.urban[index]:
                colour = colours["urban"]
            if raster.water[index]:
                colour = colours["water"]
            if raster.roads[index] and not raster.water[index]:
                colour = colours["roads"]
            if raster.buildings[index] and not raster.water[index]:
                colour = colours["buildings"]
            pixels[x, y] = colour
    return image


def write_geography_preview(path: Path, raster: OsmRaster) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _geography_preview_image(raster).save(path, format="PNG", optimize=False)


def write_meadow_grass_placement_preview(
    path: Path,
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    generated: ObjectGenerationResult,
    spec: object,
    *,
    size: int = 1024,
) -> None:
    """Write a north-up diagnostic of normalized meadows and grass placement."""
    size = max(256, int(size))
    legend_height = 96
    base = _geography_preview_image(raster).resize(
        (size, size), resample=Image.Resampling.NEAREST
    ).convert("RGBA")
    world_size = float(getattr(spec, "world_size"))

    def pixel(point: PointXZ) -> tuple[int, int]:
        x, z = point
        return (
            int(round(max(0.0, min(1.0, x / world_size)) * (size - 1))),
            int(round((1.0 - max(0.0, min(1.0, z / world_size))) * (size - 1))),
        )

    meadow_features = tuple(
        feature
        for feature in dataset.farmland
        if feature.tags.get("landuse", "").casefold() == "meadow"
    )
    meadow_polygons = tuple(
        polygon for feature in meadow_features for polygon in feature.polygons
    )
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    outlines: list[list[tuple[int, int]]] = []
    for polygon in meadow_polygons:
        outer = [pixel(projection.to_world(point)) for point in polygon.outer]
        if len(outer) < 3:
            continue
        mask_draw.polygon(outer, fill=88)
        outlines.append(outer)
        for hole in polygon.holes:
            projected_hole = [pixel(projection.to_world(point)) for point in hole]
            if len(projected_hole) >= 3:
                mask_draw.polygon(projected_hole, fill=0)

    meadow_colour = Image.new("RGBA", (size, size), (255, 224, 48, 0))
    meadow_colour.putalpha(mask)
    base = Image.alpha_composite(base, meadow_colour)
    draw = ImageDraw.Draw(base)
    for outline in outlines:
        draw.line(outline, fill=(255, 238, 78, 255), width=3, joint="curve")

    for point in generated.meadow_grass_rejection_positions:
        x, y = pixel(point)
        draw.line((x - 2, y - 2, x + 2, y + 2), fill=(232, 72, 59, 220), width=1)
        draw.line((x - 2, y + 2, x + 2, y - 2), fill=(232, 72, 59, 220), width=1)
    for point in generated.meadow_grass_positions:
        x, y = pixel(point)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(37, 235, 92, 255), outline=(7, 55, 22, 255), width=1)

    image = Image.new("RGB", (size, size + legend_height), (22, 25, 27))
    image.paste(base.convert("RGB"), (0, 0))
    legend = ImageDraw.Draw(image)
    legend.rectangle((0, size, size - 1, size + legend_height - 1), fill=(22, 25, 27), outline=(90, 96, 99), width=1)
    enabled = bool(getattr(spec, "meadow_grass_enabled", False))
    spacing = float(getattr(spec, "meadow_grass_spacing", 24.0))
    limit = int(getattr(spec, "maximum_meadow_grass_objects", 20000))
    legend.text((14, size + 10), "Meadow grass placement diagnostic (north up)", fill=(245, 245, 245))
    legend.text(
        (14, size + 31),
        (
            f"meadow features={len(meadow_features)}  polygons={len(meadow_polygons)}  "
            f"placed={generated.meadow_grass_objects}  rejected={generated.meadow_grass_rejections}"
        ),
        fill=(225, 225, 225),
    )
    legend.text(
        (14, size + 52),
        f"enabled={enabled}  spacing={spacing:g}m  warning threshold={limit}  rejected points shown={len(generated.meadow_grass_rejection_positions)}",
        fill=(205, 205, 205),
    )
    legend.rectangle((14, size + 75, 26, size + 87), fill=(255, 224, 48), outline=(255, 238, 78))
    legend.text((32, size + 74), "normalized meadow", fill=(225, 225, 225))
    legend.ellipse((178, size + 77, 186, size + 85), fill=(37, 235, 92), outline=(7, 55, 22))
    legend.text((192, size + 74), "placed grass", fill=(225, 225, 225))
    legend.line((292, size + 77, 300, size + 85), fill=(232, 72, 59), width=2)
    legend.line((292, size + 85, 300, size + 77), fill=(232, 72, 59), width=2)
    legend.text((306, size + 74), "rejected candidate", fill=(225, 225, 225))
    if not meadow_polygons:
        legend.text((size - 310, size + 31), "NO NORMALIZED MEADOW POLYGONS", fill=(255, 108, 92))
    elif not generated.meadow_grass_positions:
        legend.text((size - 260, size + 31), "NO GRASS CLUSTERS PLACED", fill=(255, 108, 92))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def attribution_text(spec: OsmSpec) -> str:
    south, west, north, east = spec.bbox
    return f"""OpenStreetMap attribution

Contains information from OpenStreetMap, made available under the Open Database License (ODbL).
Attribution: © OpenStreetMap contributors
License and attribution details: https://www.openstreetmap.org/copyright

Imported bounding box (south, west, north, east):
{south:.7f}, {west:.7f}, {north:.7f}, {east:.7f}
"""
