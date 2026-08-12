# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import pickle
import math
from typing import Any, TYPE_CHECKING, Callable, Iterable, Mapping, Sequence
from urllib import parse, request

from PIL import Image, ImageDraw

from ._version import __version__
from .cache import CACHE_SCHEMA_VERSION, atomic_write_bytes, cache_key
from .building_semantics import is_actual_church
from .model import OsmSpec, WorldObject
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
        middle_latitude = math.radians((self.south + self.north) / 2.0)
        source_x = EARTH_RADIUS_METRES * math.radians(longitude - self.west) * math.cos(middle_latitude)
        source_z = EARTH_RADIUS_METRES * math.radians(latitude - self.south)
        return source_x * self.scale_x, source_z * self.scale_z

    def to_latlon(self, point: PointXZ) -> PointLL:
        x, z = point
        middle_latitude = math.radians((self.south + self.north) / 2.0)
        source_x = x / self.scale_x
        source_z = z / self.scale_z
        latitude = self.south + math.degrees(source_z / EARTH_RADIUS_METRES)
        longitude = self.west + math.degrees(
            source_x / (EARTH_RADIUS_METRES * math.cos(middle_latitude))
        )
        return latitude, longitude

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


@dataclass(frozen=True, slots=True)
class BuildingPlacementPlan:
    osm_key: str
    geometry_index: int
    geometry_kind: str
    x: float
    z: float
    heading_degrees: float
    model_path: str
    support_polygon: tuple[PointXZ, ...]
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
    meadow_grass_positions: tuple[PointXZ, ...] = ()
    meadow_grass_rejection_positions: tuple[PointXZ, ...] = ()
    rocky_forest_objects: int = 0
    rocky_forest_rejections: int = 0
    mapped_tree_objects: int = 0
    mapped_tree_rejections: int = 0
    utility_objects: int = 0
    utility_rejections: int = 0


@dataclass(frozen=True, slots=True)
class IterativeGroundingReport:
    building_supports: int = 0
    tree_supports: int = 0
    adjusted_cells: int = 0
    maximum_adjustment: float = 0.0
    mean_adjustment: float = 0.0


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
  nwr["landcover"="trees"];
  nwr["landuse"~"^(forest|farmland|meadow|orchard|vineyard|grass|allotments|plant_nursery|greenhouse_horticulture|recreation_ground|village_green|residential|commercial|industrial|retail|construction|farmyard|garages|railway|education|institutional|civic)$"];
  way["highway"];
  nwr["building"];
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
            "Content-Type": "application/x-www-form-urlencoded; charset=ascii",
            "User-Agent": f"cwr-worldgen/{__version__} (OSM geography importer)",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds + 15) as response:
            data = response.read()
    except OSError as exc:
        raise RuntimeError(f"Overpass request failed: {exc}") from exc
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
        elif landuse in _FARMLAND_LANDUSES and polygons:
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


def road_is_gravel(tags: Mapping[str, str]) -> bool:
    """Return whether OSM surface tagging selects the generated gravel family.

    Generic ``surface=unpaved`` is intentionally treated as gravel. More
    specific earth, dirt, ground, sand, and mud values remain dirt roads.
    """

    return tags.get("surface", "").strip().casefold() in _GRAVEL_SURFACES


def road_model_for_tags(spec: OsmSpec, tags: Mapping[str, str]) -> str:
    """Select the paved, dirt, or dedicated gravel model family for one road."""

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
        candidates: set[int] = set()
        for bz in range(z0, z1 + 1):
            for bx in range(x0, x1 + 1):
                candidates.update(self.buckets.get((bx, bz), ()))
        for index in sorted(candidates):
            start, end, radius = self.corridors[index]
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


_SPATIAL_CACHE_SCHEMA = 1
_SPATIAL_INDEX_REGISTRY: dict[str, SpatialLookupIndex] = {}


def _spatial_registry_key(dataset: OsmDataset, projection: BboxProjection) -> str:
    if dataset.normalized_fingerprint:
        dataset_identity = dataset.normalized_fingerprint
    else:
        dataset_identity = cache_key(
            "osm-dataset-road-content-v1",
            [
                {
                    "key": feature.osm_key,
                    "tags": sorted((str(key), str(value)) for key, value in feature.tags.items()),
                    "points": feature.points,
                }
                for feature in dataset.roads
            ],
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
        return SpatialLookupIndex(
            previous.fingerprint,
            previous.bucket_size,
            previous.road_segments,
            previous.road_buckets,
            True,
        )
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
            payload = pickle.loads(cache_path.read_bytes())
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
                    True,
                )
                _SPATIAL_INDEX_REGISTRY[registry_key] = loaded
                progress(100, "Loaded spatial road index from cache")
                return loaded
        except (OSError, ValueError, TypeError, pickle.PickleError, EOFError):
            pass

    segments: list[ProjectedRoadSegment] = []
    mutable_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    road_total = len(dataset.roads)
    road_interval = max(1, road_total // 16)
    for road_index, feature in enumerate(dataset.roads, start=1):
        if road_index == road_total or road_index % road_interval == 0:
            value = 5 + round(82 * road_index / max(1, road_total))
            progress(value, f"Indexing projected roads {road_index:,}/{road_total:,}")
        points = [projection.to_world(point) for point in feature.points]
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
        cache_hit=False,
    )
    _SPATIAL_INDEX_REGISTRY[registry_key] = index
    if use_cache and cache_path is not None:
        atomic_write_bytes(
            cache_path,
            pickle.dumps(
                {
                    "schema": CACHE_SCHEMA_VERSION,
                    "spatial_schema": _SPATIAL_CACHE_SCHEMA,
                    "registry_key": registry_key,
                    "index": index,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )
    progress(100, f"Spatial road index ready: {len(segments):,} segments")
    return index


def get_spatial_index(dataset: OsmDataset, projection: BboxProjection) -> SpatialLookupIndex | None:
    return _SPATIAL_INDEX_REGISTRY.get(_spatial_registry_key(dataset, projection))


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


def project_road_corridors(
    dataset: OsmDataset, projection: BboxProjection, spec: OsmSpec
) -> IndexedRoadCorridors:
    spatial = get_spatial_index(dataset, projection)
    if spatial is None:
        spatial = prepare_spatial_index(dataset, projection, use_cache=False)
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
    return IndexedRoadCorridors(
        tuple(corridors),
        spatial.bucket_size,
        {key: tuple(sorted(set(values))) for key, values in buckets.items()},
    )


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


def _mask_to_cells(image: Image.Image, cells: int, *, threshold: int = 64) -> tuple[bool, ...]:
    reduced = image.resize((cells, cells), resample=Image.Resampling.BOX)
    getter = getattr(reduced, "get_flattened_data", None)
    values = getter() if getter is not None else reduced.getdata()
    pixels = tuple(int(value) >= threshold for value in values)
    # WRP rows increase from south to north while image rows increase from
    # north to south. Store masks in WRP order and flip only for previews.
    return tuple(
        pixels[(cells - 1 - z) * cells + x]
        for z in range(cells)
        for x in range(cells)
    )


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
    water_mask = _mask_to_cells(water_image, cells)
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
    )


def apply_water_elevations(
    elevations: Sequence[float],
    raster: OsmRaster,
    *,
    sea_level: float,
    water_depth: float,
    beach_height: float,
    blend_cells: int,
) -> tuple[float, ...]:
    cells = raster.cells
    if len(elevations) != cells * cells:
        raise ValueError("elevation grid does not match OSM raster")
    result = list(elevations)
    water_indices = {index for index, value in enumerate(raster.water) if value}
    for index in water_indices:
        result[index] = min(result[index], sea_level - water_depth)
    if blend_cells <= 0 or not water_indices:
        return tuple(result)

    frontier = set(water_indices)
    visited = set(water_indices)
    for distance in range(1, blend_cells + 1):
        expanded: set[int] = set()
        for index in frontier:
            x = index % cells
            z = index // cells
            for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                if 0 <= nx < cells and 0 <= nz < cells:
                    neighbour = nz * cells + nx
                    if neighbour not in visited:
                        visited.add(neighbour)
                        expanded.add(neighbour)
        target = sea_level + beach_height * distance / max(1, blend_cells)
        for index in expanded:
            result[index] = min(result[index], target)
        frontier = expanded
        if not frontier:
            break
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
    beneath their complete footprint. Planned enterable models also contribute
    their front-door apron, even when the provisional pass selected a closed
    fallback. Rigid square/triangle forests are eased toward their provisional
    support plane; generated sloped clusters retain their encoded grade, and
    individual trees contribute only their small local support patch.

    Road and water cells are immutable here. Corrections are blended and capped
    so the second pass improves contact without turning hills into large shelves.
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
        support_polygons = [footprint]
        entrance_apron = _enterable_building_entrance_apron(plan)
        if entrance_apron:
            support_polygons.append(entrance_apron)
        target = max(
            _maximum_polygon_elevation(
                elevations, cells, cell_size, support_polygon
            )
            for support_polygon in support_polygons
        )
        accepted = False
        for support_polygon in support_polygons:
            accepted = add_support(
                support_polygon,
                lambda _x, _z, value=target: value,
                priority=2,
                exclude_buildings=False,
            ) or accepted
        building_supports += int(accepted)

    square_model = str(getattr(spec, "forest_tree_model", "")).casefold()
    triangle_model = str(
        getattr(spec, "forest_everon_steep_model", r"data3d\les trojuhelnik pruchozi.p3d")
    ).casefold()
    forest_clearance = max(0.0, float(getattr(spec, "forest_ground_clearance", 0.15)))
    spacing = max(1.0, float(getattr(spec, "forest_tree_spacing", 50.0)))
    triangle_footprint = max(
        8.0, float(getattr(spec, "forest_everon_steep_footprint", 35.0))
    )
    individual_footprint = max(
        0.5, float(getattr(spec, "forest_single_tree_footprint", 2.0))
    )
    cluster_variants = (
        *FOREST_CLUSTER_VARIANTS,
        *FOREST_BORDER_VARIANTS,
        *FOREST_UNDERGROWTH_VARIANTS,
        *DITCH_GRASS_VARIANTS,
        *RURAL_VEGETATION_VARIANTS,
    )
    cluster_models: dict[str, tuple[ForestClusterVariant, float]] = {}
    individual_models = {value.casefold() for value in OSM_INDIVIDUAL_TREE_MODELS}
    individual_models.update(
        str(getattr(spec, field, "")).casefold()
        for field in (
            "forest_single_tree_model",
            "forest_hillside_tree_model",
            "forest_roadside_tree_model",
        )
        if getattr(spec, field, "")
    )
    individual_models.update(
        str(value).casefold()
        for value in getattr(spec, "forest_roadside_tree_models", ROADSIDE_TREE_MODELS)
    )
    for variant in cluster_variants:
        if variant.category not in {"interior", "rural"}:
            continue
        for grade in FOREST_CLUSTER_GRADES:
            cluster_models[
                cluster_model_path(str(getattr(spec, "name", "cwr_world")), variant.name, grade).casefold()
            ] = (variant, grade)

    for obj in provisional.objects[provisional.building_objects:]:
        model = obj.model_path.casefold()
        polygon: tuple[PointXZ, ...] | None = None
        target_at: Callable[[float, float], float] | None = None
        if model == square_model:
            polygon = _oriented_rectangle(
                obj.x, obj.z, spacing, spacing, obj.heading_degrees
            )
            target = obj.y - forest_clearance
            target_at = lambda _x, _z, value=target: value
        elif model == triangle_model:
            polygon = _oriented_rectangle(
                obj.x,
                obj.z,
                triangle_footprint * 0.58,
                triangle_footprint,
                obj.heading_degrees,
            )
            target = obj.y - forest_clearance
            target_at = lambda _x, _z, value=target: value
        elif model in cluster_models:
            variant, grade = cluster_models[model]
            margin = max(
                0.0, float(getattr(spec, "forest_cluster_footprint_margin", 0.75))
            )
            polygon = _oriented_rectangle(
                obj.x,
                obj.z,
                variant.width_m,
                variant.length_m,
                obj.heading_degrees,
                margin=margin,
            )
            angle = math.radians(obj.heading_degrees)
            width_axis = (math.cos(angle), -math.sin(angle))
            length_axis = (math.sin(angle), math.cos(angle))

            def cluster_target(
                x: float,
                z: float,
                *,
                origin_x: float = obj.x,
                origin_z: float = obj.z,
                origin_y: float = obj.y - (
                    0.03 if variant.category == "rural" else forest_clearance
                ),
                slope: float = grade,
                slope_axis: str = variant.slope_axis,
                wx: PointXZ = width_axis,
                lx: PointXZ = length_axis,
            ) -> float:
                dx = x - origin_x
                dz = z - origin_z
                local = dx * wx[0] + dz * wx[1] if slope_axis == "width" else dx * lx[0] + dz * lx[1]
                return origin_y + slope * local

            target_at = cluster_target
        elif model.startswith("data3d\\str") or model in individual_models:
            polygon = _oriented_rectangle(
                obj.x,
                obj.z,
                individual_footprint,
                individual_footprint,
                obj.heading_degrees,
            )
            target = obj.y - min(forest_clearance, 0.05)
            target_at = lambda _x, _z, value=target: value

        if polygon is not None and target_at is not None and add_support(
            polygon, target_at, priority=1, exclude_buildings=True
        ):
            tree_supports += 1

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
    """Plan only the rigid supports needed by iterative terrain grounding.

    The former provisional pass called :func:`generate_world_objects`, which
    constructed buildings, every forest fallback, grass, barriers, bridges and
    rural objects before discarding them.  Ground refinement only needs final
    building footprints and the large rigid square/triangle forest anchors.
    This compact planner deliberately creates placeholder ``WorldObject``
    records for those supports and leaves the complete placement pass to run
    once, after the terrain has been corrected and quantized.
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

    progress(20, "Planning primary forest grounding supports")
    forest_count = 0
    if spec.max_forest_objects > 0:
        road_corridors = project_road_corridors(dataset, projection, spec)
        spacing = max(1.0, float(spec.forest_tree_spacing))
        columns = max(1, int(math.ceil(spec.world_size / spacing)))
        rows = columns
        half = spacing * 0.5
        clearance = min(half * 0.7, max(1.0, spec.cell_size * 0.45))
        seed = str(getattr(spec, "deterministic_seed", "cwr-worldgen"))
        low_anchor = bool(getattr(spec, "forest_low_anchor", False))
        maximum_allowed_relief = max(
            0.0, float(getattr(spec, "forest_maximum_block_relief", 8.0))
        )
        block_maximum_burial = max(
            0.0, float(getattr(spec, "forest_block_maximum_burial", 8.0))
        )
        block_maximum_float = max(
            0.0, float(getattr(spec, "forest_block_maximum_float", 0.5))
        )
        triangle_model = str(
            getattr(
                spec,
                "forest_everon_steep_model",
                r"data3d\les trojuhelnik pruchozi.p3d",
            )
        )
        triangle_footprint = max(
            8.0, float(getattr(spec, "forest_everon_steep_footprint", 35.0))
        )
        triangle_maximum_relief = max(
            0.0,
            float(getattr(spec, "forest_everon_steep_maximum_relief", 18.0)),
        )
        triangle_maximum_burial = max(
            0.0,
            float(getattr(spec, "forest_everon_steep_maximum_burial", 18.0)),
        )
        triangle_maximum_float = max(
            0.0,
            float(getattr(spec, "forest_everon_steep_maximum_float", 0.5)),
        )
        severe_fallback_enabled = bool(
            getattr(spec, "forest_severe_hill_fallback", True)
        )
        polygon_sink_fraction = min(
            1.0,
            max(0.0, float(getattr(spec, "forest_polygon_sink_fraction", 0.5))),
        )
        total_rows = max(1, rows)

        for row in range(rows):
            if row == rows - 1 or row % max(1, rows // 12) == 0:
                progress(
                    20 + round(80 * (row + 1) / total_rows),
                    f"Planning primary forest supports {row + 1:,}/{rows:,} rows",
                )
            for column in range(columns):
                if forest_count >= spec.max_forest_objects:
                    break
                x = min(spec.world_size - 0.001, (column + 0.5) * spacing)
                z = min(spec.world_size - 0.001, (row + 0.5) * spacing)
                samples = (
                    (x, z),
                    (x - clearance, z - clearance),
                    (x + clearance, z - clearance),
                    (x - clearance, z + clearance),
                    (x + clearance, z + clearance),
                )
                if any(
                    not (0.0 <= sx < spec.world_size and 0.0 <= sz < spec.world_size)
                    or not _mask_at(raster.forest, spec.cells, spec.world_size, sx, sz)
                    or _mask_at(raster.water, spec.cells, spec.world_size, sx, sz)
                    or _mask_at(raster.roads, spec.cells, spec.world_size, sx, sz)
                    or _mask_at(raster.buildings, spec.cells, spec.world_size, sx, sz)
                    for sx, sz in samples
                ):
                    continue
                if forest_block_intersects_road_corridors(
                    road_corridors, x, z, block_size=spacing
                ):
                    continue

                geographic_column, geographic_row = _geographic_lattice_identity(
                    projection, x, z, spacing
                )
                digest = hashlib.blake2s(
                    f"{seed}:forest:{geographic_column}:{geographic_row}".encode(
                        "utf-8"
                    ),
                    digest_size=2,
                ).digest()
                heading = float((int.from_bytes(digest, "little") % 4) * 90)
                block_supports = _square_elevation_samples(
                    elevations, spec.cells, spec.cell_size, x, z, spacing
                )
                relief = max(block_supports) - min(block_supports)
                anchor: float | None
                if not low_anchor:
                    anchor = max(block_supports) + spec.forest_ground_clearance
                else:
                    fit = (
                        _terrain_fit_anchor(
                            block_supports,
                            clearance=spec.forest_ground_clearance,
                            maximum_burial=block_maximum_burial,
                            maximum_float=block_maximum_float,
                        )
                        if relief <= maximum_allowed_relief
                        else None
                    )
                    anchor = fit[0] if fit is not None else None
                model = str(spec.forest_tree_model)
                if anchor is None and str(
                    getattr(spec, "forest_profile", "malden")
                ).casefold() == "everon":
                    gradient_x, gradient_z = _local_terrain_gradient(
                        elevations, spec.cells, spec.cell_size, x, z
                    )
                    if abs(gradient_x) + abs(gradient_z) > 1.0e-9:
                        heading = math.degrees(
                            math.atan2(-gradient_z, gradient_x)
                        ) % 360.0
                    triangle_supports = _oriented_footprint_elevation_samples(
                        elevations,
                        spec.cells,
                        spec.cell_size,
                        x,
                        z,
                        triangle_footprint * 0.58,
                        triangle_footprint,
                        heading,
                    )
                    triangle_relief = max(triangle_supports) - min(
                        triangle_supports
                    )
                    triangle_fit = (
                        _terrain_fit_anchor(
                            triangle_supports,
                            clearance=spec.forest_ground_clearance,
                            maximum_burial=triangle_maximum_burial,
                            maximum_float=triangle_maximum_float,
                        )
                        if triangle_relief <= triangle_maximum_relief
                        else None
                    )
                    if triangle_fit is not None:
                        model = triangle_model
                        anchor = triangle_fit[0]
                        if (
                            severe_fallback_enabled
                            and polygon_sink_fraction > 0.0
                            and triangle_relief > 1.0e-9
                        ):
                            anchor -= triangle_relief * polygon_sink_fraction
                if anchor is None:
                    continue
                objects.append(WorldObject(next_id, model, x, anchor, z, heading))
                next_id += 1
                forest_count += 1
            if forest_count >= spec.max_forest_objects:
                break

    progress(100, "Grounding support plan ready")
    return ObjectGenerationResult(
        objects=tuple(objects),
        road_objects=0,
        building_objects=len(active_building_plans),
        forest_objects=forest_count,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=forest_count >= spec.max_forest_objects,
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
    # Vegetation has no foundation skirt to hide a wrong interpolation choice.
    # Keep both possible RVW4 triangle heights at every complete-footprint
    # arrangement vertex so fitting can bound burial and floating exactly.
    return _polygon_ground_elevation_samples(
        elevations, cells, cell_size, polygon
    )


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
) -> tuple[str, float, float, float, float, str, float, float, float] | None:
    """Find one reusable small forest cluster for a steep rejected block."""

    search_radius = max(0.0, float(getattr(spec, "forest_cluster_search_radius", 10.0)))
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
) -> tuple[tuple[float, float, float, int], ...]:
    """Return deterministic blue-noise-like candidates inside one forest block.

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
    minimum_distance2 = max(0.0, float(minimum_spacing)) ** 2
    for _priority, candidate_x, candidate_z, heading, variant in raw:
        if minimum_distance2 and any(
            (candidate_x - other_x) ** 2 + (candidate_z - other_z) ** 2 < minimum_distance2
            for other_x, other_z, _other_heading, _other_variant in accepted
        ):
            continue
        accepted.append((candidate_x, candidate_z, heading, variant))
    return tuple(accepted)


def _roadside_tree_candidates(
    seed: str,
    column: int,
    row: int,
    x: float,
    z: float,
    block_size: float,
) -> tuple[tuple[float, float, float, int], ...]:
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
) -> tuple[tuple[float, float, float, int], ...]:
    return _roadside_vegetation_candidates(
        seed, column, row, x, z, block_size,
        label="bush", minimum_spacing=2.25, candidate_count=192,
    )


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


def _road_needs_bridge_deck(
    feature: OsmLineFeature,
    chunks: Sequence[tuple[float, float, float, float, float, float, float, float]],
    raster: OsmRaster,
    spec,
) -> bool:
    bridge = str(feature.tags.get("bridge", "")).strip().casefold()
    if bridge not in {"", "no", "false", "0", "none"}:
        return True
    if str(feature.tags.get("man_made", "")).strip().casefold() == "bridge":
        return True
    if str(feature.tags.get("special", "")).strip().casefold() == "bridge":
        return True
    if _numeric_tag(feature.tags, "layer", 0.0) <= 0.0:
        return False
    return any(
        _mask_at(raster.water, spec.cells, spec.world_size, sx, sz)
        for chunk in chunks
        for sx, sz in ((chunk[4], chunk[5]), (chunk[0], chunk[1]), (chunk[6], chunk[7]))
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
) -> bool:
    radius_squared = radius * radius
    for feature in dataset.building_points:
        if feature.tags.get("source") == "overturemaps":
            continue
        px, pz = projection.to_world(feature.point)
        if (px - x) * (px - x) + (pz - z) * (pz - z) <= radius_squared:
            return True
    for feature in dataset.building_polygons:
        if feature.tags.get("source") == "overturemaps":
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


def _geojson_overture_building_polygons(path: Path) -> tuple[OsmPolygonFeature, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Overture buildings GeoJSON {path}: {exc}") from exc
    features = document.get("features") if isinstance(document, Mapping) else None
    if not isinstance(features, list):
        raise ValueError(f"Overture buildings GeoJSON {path} does not contain a FeatureCollection")

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

    buildings: list[OsmPolygonFeature] = []
    for feature_index, feature in enumerate(features):
        if not isinstance(feature, Mapping):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping):
            continue
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
        if not polygons:
            continue
        properties = feature.get("properties")
        source_id = (
            str(properties.get("id"))
            if isinstance(properties, Mapping) and properties.get("id")
            else str(feature.get("id", feature_index))
        )
        tags = {
            "building": "house",
            "source": "overturemaps",
            "cwr:synthetic": "overture_building",
        }
        buildings.append(OsmPolygonFeature(
            f"overture/{source_id}",
            tags,
            tuple(polygons),
        ))
    return tuple(sorted(buildings, key=lambda item: item.osm_key))


def augment_dataset_with_overture_buildings(
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: OsmSpec,
    geojson_path: Path,
) -> OsmDataset:
    """Use Overture building footprints before synthetic residential infill."""
    overture_buildings = _geojson_overture_building_polygons(geojson_path)
    if not overture_buildings:
        return dataset

    fallback_areas: list[tuple[tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...], tuple[float, float, float, float]]] = []
    for feature in sorted(dataset.urban, key=lambda item: item.osm_key):
        if feature.tags.get("landuse", "").casefold() != "residential":
            continue
        for polygon in feature.polygons:
            outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            holes = tuple(tuple(projection.to_world(point) for point in hole[:-1]) for hole in polygon.holes)
            if len(outer) >= 3 and not _residential_area_has_mapped_building(dataset, projection, outer, holes):
                fallback_areas.append((outer, holes, _world_bbox(outer)))
    for place in sorted(dataset.places, key=lambda item: item.osm_key):
        settlement = _small_settlement_infill_feature(place, dataset, projection, spec)
        if settlement is None:
            continue
        polygon = settlement.polygons[0]
        outer = tuple(projection.to_world(point) for point in polygon.outer[:-1])
        holes = tuple(tuple(projection.to_world(point) for point in hole[:-1]) for hole in polygon.holes)
        if len(outer) >= 3 and not _residential_area_has_mapped_building(dataset, projection, outer, holes):
            fallback_areas.append((outer, holes, _world_bbox(outer)))
    fallback_areas.extend(
        (outer, holes, _world_bbox(outer))
        for outer, holes in _overture_road_ending_fallback_areas(dataset, projection, spec)
    )
    if not fallback_areas:
        return dataset

    mapped_points = tuple(
        projection.to_world(feature.point)
        for feature in dataset.building_points
        if feature.tags.get("source") != "overturemaps"
    )
    mapped_polygons: list[tuple[tuple[PointXZ, ...], tuple[float, float, float, float]]] = []
    for feature in dataset.building_polygons:
        if feature.tags.get("source") == "overturemaps":
            continue
        for polygon in feature.polygons:
            projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            if len(projected) >= 3:
                mapped_polygons.append((projected, _world_bbox(projected)))

    def conflicts_with_mapped_building(
        projected: Sequence[PointXZ],
        bbox: tuple[float, float, float, float],
        cx: float,
        cz: float,
    ) -> bool:
        for point in mapped_points:
            if _bbox_contains_point(bbox, point[0], point[1]) and _point_in_polygon(point, projected):
                return True
            if math.hypot(point[0] - cx, point[1] - cz) <= 6.0:
                return True
        for mapped, mapped_bbox in mapped_polygons:
            if not _bboxes_intersect(bbox, mapped_bbox):
                continue
            if _polygons_intersect(projected, mapped):
                return True
            if _point_in_polygon((cx, cz), mapped):
                return True
        return False

    accepted: list[OsmPolygonFeature] = []
    seen_keys = {feature.osm_key for feature in dataset.building_polygons}
    for feature in overture_buildings:
        kept_polygons: list[GeoPolygon] = []
        for polygon in feature.polygons:
            projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            if len(projected) < 3:
                continue
            bbox = _world_bbox(projected)
            area, x, z = _polygon_area_centroid(projected)
            if area < getattr(spec, "building_minimum_area", 10.0):
                continue
            if conflicts_with_mapped_building(projected, bbox, x, z):
                continue
            if any(
                _bboxes_intersect(bbox, fallback_bbox)
                and _polygon_contains_with_holes((x, z), outer, holes)
                for outer, holes, fallback_bbox in fallback_areas
            ):
                kept_polygons.append(polygon)
        if kept_polygons and feature.osm_key not in seen_keys:
            accepted.append(OsmPolygonFeature(feature.osm_key, feature.tags, tuple(kept_polygons)))
            seen_keys.add(feature.osm_key)
    if not accepted:
        return dataset
    overture_fingerprint = cache_key(
        "overture-buildings-accepted-v2",
        {
            "base": dataset.normalized_fingerprint,
            "keys": tuple(feature.osm_key for feature in accepted),
            "polygons": tuple(
                tuple(tuple(round(value, 7) for point in polygon.outer for value in point) for polygon in feature.polygons)
                for feature in accepted
            ),
        },
    )
    return replace(
        dataset,
        building_polygons=tuple(sorted((*dataset.building_polygons, *accepted), key=lambda item: item.osm_key)),
        element_count=dataset.element_count + len(accepted),
        normalized_fingerprint=overture_fingerprint,
    )


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
            expanded = _expand_polygon_from_centroid(footprint, building_clearance)
            if any(_polygons_intersect(expanded, prior) for prior in occupied):
                continue
            candidates.append((road_distance, digest[8:], index, x, z, heading, width, length, footprint))
    candidates.sort(key=lambda item: (item[0], item[1]))
    # Sparse by design: at most roughly one building per spacing-square and no
    # more than the caller's global budget.
    chosen = candidates[:budget]
    return tuple((idx, x, z, heading, width, length, footprint) for _distance, _prio, idx, x, z, heading, width, length, footprint in chosen)

def plan_building_placements(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: OsmSpec,
    building_asset_library: "ProceduralBuildingLibrary | None" = None,
) -> tuple[tuple[BuildingPlacementPlan, ...], bool]:
    """Resolve final building models, headings, positions, and exact footprints.

    This runs before terrain solving for Milestone 8/9 builds. Terrain pads and
    object placement therefore consume the same final footprint instead of one
    stage flattening the OSM polygon while another stage places a larger rotated
    model somewhere else, a surprisingly effective recipe for hovering houses.
    """

    candidates: list[tuple[int, str, int, str, Any, Any]] = []
    for feature in dataset.building_polygons:
        for polygon_index, polygon in enumerate(feature.polygons):
            candidates.append((
                _building_placement_priority(feature.tags),
                feature.osm_key,
                polygon_index,
                "polygon",
                feature,
                polygon,
            ))
    for feature in dataset.building_points:
        candidates.append((
            _building_placement_priority(feature.tags),
            feature.osm_key,
            0,
            "point",
            feature,
            feature.point,
        ))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    building_road_corridors = project_road_corridors(dataset, projection, spec)
    plans: list[BuildingPlacementPlan] = []
    truncated = False
    for _priority, osm_key, geometry_index, geometry_kind, feature, geometry in candidates:
        if len(plans) >= spec.max_buildings:
            truncated = True
            break

        procedural_placement = None
        building_family = _building_family(feature.tags)
        if geometry_kind == "polygon":
            projected = [projection.to_world(point) for point in geometry.outer[:-1]]
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
                planner = getattr(building_asset_library, "plan_polygon", None)
                procedural_placement = (
                    planner(feature.tags, projected, road_point=road_point)
                    if planner is not None
                    else building_asset_library.place_polygon(feature.tags, projected, road_point=road_point)
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
                planner = getattr(building_asset_library, "plan_point", None)
                procedural_placement = (
                    planner(
                        feature.tags, spec.point_building_footprint, heading, x=x, z=z
                    )
                    if planner is not None
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
            support_polygon=tuple(support_polygon),
            procedural_placement=procedural_placement,
            road_nudged=road_nudged,
            building_family=building_family,
        ))

    # Missing-building fallback: empty landuse=residential polygons are eligible,
    # and named village/hamlet-style places get a modest synthetic residential
    # patch when OSM has no mapped buildings there. Infill comes after real OSM
    # buildings, so it can never consume the budget ahead of source data.
    if bool(getattr(spec, "residential_infill_enabled", False)) and len(plans) < spec.max_buildings:
        infill_limit = max(0, int(getattr(spec, "maximum_residential_infill_buildings", 1500)))
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
        occupied = [tuple(plan.support_polygon) for plan in plans]
        generated = 0
        for feature, polygon_index, polygon in infill_sources:
            remaining = min(infill_limit - generated, spec.max_buildings - len(plans))
            if remaining <= 0:
                truncated = truncated or generated >= infill_limit
                break
            rectangles = _infill_candidate_rectangles(
                feature, polygon_index, polygon, dataset, projection, raster, spec,
                building_road_corridors, occupied, budget=remaining,
            )
            for candidate_index, x, z, heading, width, length, footprint in rectangles:
                tags = {"building": "house", "cwr:synthetic": "residential_infill"}
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
                    if any(_polygons_intersect(
                        _expand_polygon_from_centroid(final_footprint, max(1.0, float(getattr(spec, "residential_infill_building_clearance", 6.0)))), prior
                    ) for prior in occupied):
                        continue
                if _polygon_fully_covered_by_mask(
                    raster.water, spec.cells, spec.world_size, final_footprint
                ):
                    continue
                key = f"infill/{feature.osm_key}/{polygon_index}-{candidate_index}"
                plans.append(BuildingPlacementPlan(
                    osm_key=key, geometry_index=candidate_index, geometry_kind="synthetic",
                    x=x, z=z, heading_degrees=final_heading, model_path=model_path,
                    support_polygon=tuple(final_footprint),
                    procedural_placement=procedural_placement, road_nudged=False,
                    building_family=family, synthetic_infill=True,
                ))
                occupied.append(tuple(final_footprint))
                generated += 1
            if generated >= infill_limit or len(plans) >= spec.max_buildings:
                break
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

    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(percent, stage)

    progress(53, "Preparing building and vegetation placement")
    infrastructure_library = ProceduralInfrastructureLibrary(str(getattr(spec, "name", "cwr_world")), cache_enabled=False)
    next_id = starting_object_id
    road_count = 0
    road_truncated = False

    road_features = dataset.roads if include_roads else ()
    for feature in road_features:
        if not road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = [projection.to_world(point) for point in feature.points]
        model = road_model_for_tags(spec, feature.tags)
        for start, end in zip(points, points[1:]):
            dx = end[0] - start[0]
            dz = end[1] - start[1]
            length = math.hypot(dx, dz)
            if length < 1.0:
                continue
            count = max(1, int(round(length / spec.road_segment_length)))
            for segment in range(count):
                if road_count >= spec.max_road_objects:
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
                objects.append(WorldObject(next_id, model, x, y, z, heading))
                next_id += 1
                road_count += 1
            if road_truncated:
                break
        if road_truncated:
            break

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

    minimum_foundation_depth = max(0.0, float(getattr(spec, "building_foundation_depth", 0.5)))
    maximum_foundation_limit = max(
        minimum_foundation_depth,
        float(getattr(spec, "building_foundation_maximum_depth", 2.5)),
    )
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
        entrance_apron = _enterable_building_entrance_apron(plan)
        if entrance_apron:
            _entrance_minimum, entrance_maximum = _polygon_elevation_extrema(
                elevations, spec.cells, spec.cell_size, entrance_apron
            )
            maximum_height = max(maximum_height, entrance_maximum)
        relief = maximum_height - minimum_height
        # Identical to houses: one shared visible ground clearance, no church
        # offset, no special plinth and no church-only foundation minimum.
        ground_clearance = float(spec.building_ground_clearance)
        if entrance_apron:
            ground_clearance = max(
                ground_clearance,
                ENTERABLE_BUILDING_MINIMUM_GROUND_CLEARANCE_METRES,
            )
        serialization_safety = 0.0
        required_foundation_depth = max(
            minimum_foundation_depth,
            relief + ground_clearance + foundation_safety,
        )
        placement_for_registration = plan.procedural_placement
        if (
            entrance_apron
            and required_foundation_depth > maximum_foundation_limit + 1e-9
            and building_asset_library is not None
            and placement_for_registration is not None
        ):
            # A doorway cannot be safely kept above its approach terrain within
            # the configured normal foundation limit. Preserve the building and
            # its visual style, but use the matching closed/non-enterable model.
            fallback_selected = replace(
                placement_for_registration.selected,
                interiors=False,
            )
            placement_for_registration = replace(
                placement_for_registration,
                selected=fallback_selected,
                model_path=building_asset_library.model_path(fallback_selected),
            )
            building_interior_fallbacks += 1
            entrance_apron = ()
            maximum_height = footprint_maximum_height
            relief = maximum_height - minimum_height
            ground_clearance = float(spec.building_ground_clearance)
            required_foundation_depth = max(
                minimum_foundation_depth,
                relief + ground_clearance + foundation_safety,
            )

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
        objects.append(WorldObject(
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
    ditch_grass_objects = 0
    ditch_grass_rejections = 0
    ditch_grass_maximum_burial = 0.0
    ditch_grass_maximum_float = 0.0
    barrier_objects = fence_objects = wall_objects = hedge_objects = barrier_rejections = 0
    bridge_objects = bridge_segments = bridge_rejections = 0
    tree_row_objects = orchard_objects = vineyard_objects = scrub_objects = rural_rock_objects = 0
    rural_vegetation_rejections = 0
    meadow_grass_objects = meadow_grass_rejections = 0
    meadow_grass_positions: list[PointXZ] = []
    meadow_grass_rejection_positions: list[PointXZ] = []
    rocky_forest_objects = rocky_forest_rejections = 0
    mapped_tree_objects = mapped_tree_rejections = 0
    utility_objects = utility_rejections = 0

    road_corridors = project_road_corridors(dataset, projection, spec)
    seed = str(getattr(spec, "deterministic_seed", "cwr-worldgen"))
    low_anchor = bool(getattr(spec, "forest_low_anchor", False))
    forest_profile = str(getattr(spec, "forest_profile", "malden")).casefold()
    individual_tree_maximum_float = max(
        0.0,
        float(getattr(spec, "forest_single_tree_maximum_float", 0.5)),
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

    if spec.max_forest_objects > 0:
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
        forest_clusters_enabled = bool(getattr(spec, "forest_cluster_fallback", False))
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

        for row in range(rows):
            for column in range(columns):
                if forest_count >= spec.max_forest_objects:
                    forest_truncated = True
                    break
                x = min(spec.world_size - 0.001, (column + 0.5) * spacing)
                z = min(spec.world_size - 0.001, (row + 0.5) * spacing)
                block_edge_guard = max(forest_world_edge_margin, spacing * 0.58)
                if not forest_point_inside_edge_guard(x, z, block_edge_guard):
                    continue
                geographic_column, geographic_row = _geographic_lattice_identity(
                    projection, x, z, spacing
                )
                samples = (
                    (x, z),
                    (x - clearance, z - clearance),
                    (x + clearance, z - clearance),
                    (x - clearance, z + clearance),
                    (x + clearance, z + clearance),
                )
                in_bounds_samples = tuple(
                    (sx, sz)
                    for sx, sz in samples
                    if 0 <= sx < spec.world_size and 0 <= sz < spec.world_size
                )
                forest_sample_count = sum(
                    _mask_at(raster.forest, spec.cells, spec.world_size, sx, sz)
                    for sx, sz in in_bounds_samples
                )
                block_intersects_road = forest_block_intersects_road_corridors(
                    road_corridors, x, z, block_size=spacing
                )

                if block_intersects_road and forest_profile == "everon":
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
                            or forest_count >= spec.max_forest_objects
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
                        tree_fit = _non_buried_vegetation_fit(
                            _triangle_elevation_bounds(
                                elevations,
                                spec.cells,
                                spec.cell_size,
                                tree_x,
                                tree_z,
                            ),
                            clearance=spec.forest_ground_clearance,
                            maximum_float=individual_tree_maximum_float,
                        )
                        if tree_fit is None:
                            continue
                        tree_y, _tree_float = tree_fit
                        objects.append(
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
                                or forest_count >= spec.max_forest_objects
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
                            objects.append(
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
                    if forest_count >= spec.max_forest_objects:
                        forest_truncated = True
                    continue

                if block_intersects_road:
                    forest_road_rejections += 1
                    continue

                if forest_sample_count != len(samples):
                    continue
                if any(
                    _mask_at(raster.water, spec.cells, spec.world_size, sx, sz)
                    or _mask_at(raster.roads, spec.cells, spec.world_size, sx, sz)
                    or _mask_at(raster.buildings, spec.cells, spec.world_size, sx, sz)
                    for sx, sz in samples
                ):
                    continue

                digest = hashlib.blake2s(
                    f"{seed}:forest:{geographic_column}:{geographic_row}".encode("utf-8"),
                    digest_size=2,
                ).digest()
                heading = float((int.from_bytes(digest, "little") % 4) * 90)
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
                    objects.append(
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
                    objects.append(
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
                        objects.append(
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
                        objects.append(
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
                                max(
                                    0,
                                    int(
                                        getattr(
                                            spec,
                                            "forest_undergrowth_maximum_objects",
                                            120000,
                                        )
                                    ),
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
                                objects.append(
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
                        rocky_limit = max(0, int(getattr(spec, "maximum_rocky_forest_objects", 1200)))
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
                            objects.append(WorldObject(next_id, rock_model, rock_x, rock_anchor, rock_z, rock_heading))
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
                        for tree_x, tree_z, tree_heading in _hillside_tree_candidates(
                            f"{seed}:steep-infill",
                            geographic_column,
                            geographic_row,
                            x,
                            z,
                            spacing,
                        ):
                            if (
                                steep_tree_placed >= steep_infill_tree_target
                                or forest_count >= spec.max_forest_objects
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
                            tree_fit = _non_buried_vegetation_fit(
                                _triangle_elevation_bounds(
                                    elevations,
                                    spec.cells,
                                    spec.cell_size,
                                    tree_x,
                                    tree_z,
                                ),
                                clearance=spec.forest_ground_clearance,
                                maximum_float=individual_tree_maximum_float,
                            )
                            if tree_fit is None:
                                continue
                            tree_y, _tree_float = tree_fit
                            tree_model = roadside_tree_models[
                                (
                                    geographic_column * 17
                                    + geographic_row * 31
                                    + steep_tree_placed
                                )
                                % len(roadside_tree_models)
                            ]
                            objects.append(WorldObject(
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
                        tree_fit = _non_buried_vegetation_fit(
                            _triangle_elevation_bounds(
                                elevations,
                                spec.cells,
                                spec.cell_size,
                                tree_x,
                                tree_z,
                            ),
                            clearance=spec.forest_ground_clearance,
                            maximum_float=individual_tree_maximum_float,
                        )
                        if tree_fit is None:
                            forest_hillside_candidate_rejections += 1
                            continue
                        tree_y, _tree_float = tree_fit
                        accepted_candidates.append(
                            (tree_x, tree_z, tree_heading, tree_y)
                        )
                if accepted_candidates:
                    tree_x, tree_z, tree_heading, tree_y = accepted_candidates[0]
                    objects.append(
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
                    if forest_count >= spec.max_forest_objects:
                        forest_truncated = True
                        break
                    tree_x, tree_z, tree_heading, tree_y = candidates[extra_index]
                    objects.append(
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
        progress(60, "Scattering individual forest trees")
        extra_single_enabled = bool(getattr(spec, "forest_single_tree_enabled", True))
        extra_single_model = str(getattr(spec, "forest_single_tree_model", r"data3d\str smrk_medium.p3d"))
        extra_single_limit = _scaled_synthetic_tree_limit(
            int(getattr(spec, "maximum_forest_single_tree_objects", 1000)),
            spec.world_size,
        )
        extra_single_spacing = max(
            20.0,
            float(getattr(spec, "forest_single_tree_spacing", 45.0)),
        )
        extra_single_footprint = max(1.5, float(getattr(spec, "forest_single_tree_footprint", 2.0)))
        extra_single_relief = max(1.5, float(getattr(spec, "forest_single_tree_maximum_relief", 8.0)))
        extra_single_candidates_per_cell = 1
        if (
            forest_profile == "everon"
            and extra_single_enabled
            and extra_single_limit > 0
            and not forest_truncated
            and forest_count < spec.max_forest_objects
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
                    tree_fit = _non_buried_vegetation_fit(
                        _triangle_elevation_bounds(
                            elevations,
                            spec.cells,
                            spec.cell_size,
                            tree_x,
                            tree_z,
                        ),
                        clearance=spec.forest_ground_clearance,
                        maximum_float=individual_tree_maximum_float,
                    )
                    if tree_fit is None:
                        continue
                    tree_y, _tree_float = tree_fit
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
                max(0, spec.max_forest_objects - forest_count),
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
                objects.append(WorldObject(
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
                forest_count >= spec.max_forest_objects
                and len(eligible_extra_single_trees) > extra_single_available
            ):
                forest_truncated = True

        progress(61, "Adding rocky outcrops to uncovered forest hills")
        if bool(getattr(spec, "rocky_forest_fallback_enabled", False)):
            rocky_limit = max(0, int(getattr(spec, "maximum_rocky_forest_objects", 1200)))
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
                        objects.append(WorldObject(next_id, rock_model, rock_x, rock_anchor, rock_z, rock_heading))
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
        if bool(getattr(spec, "forest_undergrowth_enabled", False)):
            undergrowth_base_limit = max(0, int(getattr(spec, "forest_undergrowth_maximum_objects", 120000)))
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
                objects.append(WorldObject(next_id, model_path, placed_x, placed_y, placed_z, placed_heading))
                next_id += 1
                forest_undergrowth_objects += 1
                forest_cluster_variant_counts[variant_name] += 1
                forest_undergrowth_maximum_burial = max(forest_undergrowth_maximum_burial, burial)
                forest_undergrowth_maximum_float = max(forest_undergrowth_maximum_float, floating)
                maximum_forest_burial = max(maximum_forest_burial, burial)
                maximum_forest_float = max(maximum_forest_float, floating)

        progress(63, "Adding stock bushes to steep forested hills")
        if bool(getattr(spec, "steep_hill_bushes_enabled", False)):
            bush_limit = max(0, int(getattr(spec, "maximum_steep_hill_bush_objects", 80000)))
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
                objects.append(WorldObject(next_id, bush_model, bush_x, bush_anchor, bush_z, bush_heading))
                next_id += 1
                steep_hill_bush_objects += 1

        progress(64, "Softening forest borders")
        # Nogova-style soft borders are a separate sparse pass along actual OSM
        # forest boundaries. They use reusable proxy clusters, not rows of WRP trees.
        if bool(getattr(spec, "forest_border_enabled", False)):
            border_limit = max(0, int(getattr(spec, "forest_border_maximum_objects", 2000)))
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
                objects.append(
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
        ditch_limit = max(0, int(getattr(spec, "maximum_ditch_grass_objects", 2000)))
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
            objects.append(
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
        barrier_limit = max(0, int(getattr(spec, "maximum_barrier_objects", 4000)))
        barrier_length = max(2.0, float(getattr(spec, "barrier_segment_length", 6.0)))
        stock_hedge_models = tuple(getattr(spec, "stock_hedge_models", STOCK_HEDGE_MODELS))
        stock_wall_models = tuple(getattr(spec, "stock_wall_models", STOCK_WALL_MODELS))
        stock_metal_fence_models = tuple(getattr(spec, "stock_metal_fence_models", STOCK_METAL_FENCE_MODELS))
        for feature in sorted(dataset.barriers, key=lambda item: item.osm_key):
            subtype = feature.tags.get("barrier", "fence").casefold()
            subtype = "wall" if subtype in {"wall", "retaining_wall"} else "hedge" if subtype == "hedge" else "fence"
            metal_fence = subtype == "fence" and osm_fence_is_metal(feature.tags)
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
                elif metal_fence:
                    model = stock_metal_fence_model(identity, stock_metal_fence_models)
                else:
                    model = infrastructure_library.barrier_model(subtype, length)
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
                elif subtype == "hedge" or metal_fence:
                    heading_offset = (
                        HEDGE_MODEL_HEADING_OFFSET_DEGREES
                        if subtype == "hedge"
                        else METAL_FENCE_MODEL_HEADING_OFFSET_DEGREES
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
                objects.append(WorldObject(next_id, model, placed_x, y, placed_z, placed_heading, pitch_degrees=placed_pitch))
                next_id += 1
                barrier_objects += 1
                fence_objects += int(subtype == "fence")
                wall_objects += int(subtype == "wall")
                hedge_objects += int(subtype == "hedge")
            if barrier_objects >= barrier_limit:
                break

    # Bridge-tagged roads retain their ordinary fitted road pieces underneath for
    # network continuity. By default the deck uses the stock Nogova 30 m module;
    # procedural mode instead emits world-local width/length-fitted deck modules.
    progress(66, "Placing bridge modules")
    if bool(getattr(spec, "bridges_enabled", False)):
        bridge_limit = max(0, int(getattr(spec, "maximum_bridge_objects", 1000)))
        procedural_bridges = bool(getattr(spec, "procedural_bridges", False))
        module_length = (
            max(3.0, float(getattr(spec, "bridge_module_length", 30.0)))
            if procedural_bridges
            else NOGOVA_BRIDGE_MODULE_LENGTH_METRES
        )
        for feature in sorted(dataset.roads, key=lambda item: item.osm_key):
            points = tuple(projection.to_world(point) for point in feature.points)
            source_chunks = _bridge_module_chunks(points, module_length)
            if not source_chunks:
                continue
            if not _road_needs_bridge_deck(feature, source_chunks, raster, spec):
                continue
            if procedural_bridges:
                # The tagged bridge span may start only after the road has begun
                # descending toward a beach/river bank. Extend the actual bridge
                # objects back to the stable upper road on both approaches.
                points = _extend_procedural_bridge_to_approach_plateaus(
                    points, elevations, spec, module_length,
                    feature=feature, dataset=dataset, projection=projection, raster=raster,
                )
            chunks = _bridge_module_chunks(points, module_length)
            if not chunks:
                continue
            bridge_segments += 1
            # Never emit a bridge that stops mid-span. A complete module set is
            # either accepted or rejected when the bridge-object budget is tight.
            if len(chunks) > bridge_limit - bridge_objects:
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
                model_lengths = tuple(
                    max(3.0, round(chunk[3] * 10.0) / 10.0)
                    for chunk in chunks
                )
                half_width = generated_width * 0.5 + GENERATED_BRIDGE_RAIL_OVERHANG_METRES
                vertical_depth = GENERATED_BRIDGE_MAXIMUM_DEPTH_METRES
            else:
                generated_width = width
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
                in zip(chunks, model_lengths)
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
                    for chunk in chunks
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
                # The stock Nogova bridge is a fixed 30 m module whose supports
                # and unused ends are expected to disappear into the banks. Fit
                # its roadway origin directly to the approach-road elevations
                # and never raise it just to clear terrain under the span.
                deck_start = start_ground + NOGOVA_BRIDGE_APPROACH_OFFSET_METRES
                deck_end = end_ground + NOGOVA_BRIDGE_APPROACH_OFFSET_METRES
                if spans_water or minimum_support < float(spec.sea_level):
                    minimum_water_deck = (
                        float(spec.sea_level) + NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES
                    )
                    deck_start = max(deck_start, minimum_water_deck)
                    deck_end = max(deck_end, minimum_water_deck)
            travelled = 0.0
            bridge_plan: list[tuple[str, float, float, float, float, float]] = []
            chunk_count = len(chunks)
            for index, (chunk, model_length) in enumerate(zip(chunks, model_lengths)):
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
                    if chunk_count == 1:
                        subtype = "single"
                    elif index == 0:
                        subtype = "start"
                    elif index == chunk_count - 1:
                        subtype = "end"
                    else:
                        subtype = "middle"
                    model = infrastructure_library.bridge_model(
                        subtype, generated_width, model_length
                    )
                else:
                    model = NOGOVA_BRIDGE_MODEL
                bridge_plan.append((model, x, y, z, heading, pitch))
            for model, x, y, z, heading, pitch in bridge_plan:
                objects.append(WorldObject(
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
    if rural_enabled or wetland_enabled or meadow_enabled:
        rural_limit = (
            max(0, int(getattr(spec, "maximum_rural_vegetation_objects", 3000)))
            if rural_enabled else 0
        )
        rural_spacing = max(10.0, float(getattr(spec, "rural_vegetation_spacing", 28.0)))
        variants = {variant.name: variant for variant in RURAL_VEGETATION_VARIANTS}
        meadow_limit = max(0, int(getattr(spec, "maximum_meadow_grass_objects", 20000)))
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
                        objects.append(WorldObject(next_id, model, px, py, pz, ph)); next_id += 1
                        meadow_grass_objects += 1
                        meadow_grass_positions.append((px, pz))
                    if meadow_grass_objects >= meadow_limit:
                        break
                if meadow_grass_objects >= meadow_limit:
                    break
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
                objects.append(WorldObject(next_id, model, px, py, pz, ph)); next_id += 1
                tree_row_objects += 1

        wetland_limit = max(0, int(getattr(spec, "maximum_wetland_reed_objects", 100000)))
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
                        objects.append(WorldObject(next_id, model, x, anchor, z, reed_heading))
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
                        objects.append(WorldObject(next_id, model, x, anchor, z, heading)); next_id += 1
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
                    objects.append(WorldObject(next_id, model, px, py, pz, ph)); next_id += 1
                    orchard_objects += int(category == "orchard")
                    vineyard_objects += int(category == "vineyard")
                    scrub_objects += int(category == "scrub")

    progress(69, "Placing mapped trees and utility infrastructure")
    mapped_tree_limit = max(0, int(getattr(spec, "maximum_mapped_tree_objects", 5000)))
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
        if leaf_type == "needleleaved" or any(word in species_text for word in ("picea", "pinus", "abies", "spruce", "pine", "fir")):
            models = OSM_CONIFER_TREE_MODELS
        elif leaf_type == "broadleaved" or species_text:
            models = OSM_BROADLEAF_TREE_MODELS
        else:
            models = OSM_INDIVIDUAL_TREE_MODELS
        digest = hashlib.blake2s(f"{seed}:mapped-tree:{feature.osm_key}".encode("utf-8"), digest_size=4).digest()
        model = models[int.from_bytes(digest[:2], "little") % len(models)]
        heading = float(int.from_bytes(digest[2:], "little") % 360)
        tree_fit = _non_buried_vegetation_fit(
            _triangle_elevation_bounds(
                elevations, spec.cells, spec.cell_size, x, z
            ),
            clearance=mapped_tree_clearance,
            maximum_float=individual_tree_maximum_float,
        )
        if tree_fit is None:
            mapped_tree_rejections += 1
            continue
        anchor, _tree_float = tree_fit
        objects.append(WorldObject(next_id, model, x, anchor, z, heading))
        next_id += 1
        mapped_tree_objects += 1

    utility_limit = max(0, int(getattr(spec, "maximum_utility_objects", 3000)))
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
        objects.append(WorldObject(next_id, model, x, maximum + utility_clearance, z, heading))
        next_id += 1
        utility_objects += 1

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
        meadow_grass_positions=tuple(meadow_grass_positions),
        meadow_grass_rejection_positions=tuple(meadow_grass_rejection_positions),
        rocky_forest_objects=rocky_forest_objects,
        rocky_forest_rejections=rocky_forest_rejections,
        mapped_tree_objects=mapped_tree_objects,
        mapped_tree_rejections=mapped_tree_rejections,
        utility_objects=utility_objects,
        utility_rejections=utility_rejections,
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
        f"enabled={enabled}  spacing={spacing:g}m  object cap={limit}  rejected points shown={len(generated.meadow_grass_rejection_positions)}",
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
