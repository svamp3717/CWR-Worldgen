# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import pickle
import re
import shutil
import tempfile
import unicodedata
from typing import Any, Callable, Iterable, Mapping, Sequence

from shapely import (
    area as vector_area,
    buffer as vector_buffer,
    difference as vector_difference,
    intersection as vector_intersection,
    make_valid,
    set_precision,
    union_all,
)
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.ops import linemerge, polygonize, transform, unary_union
from shapely.strtree import STRtree

from ._version import GENERATOR_VERSION
from .cache import CACHE_SCHEMA_VERSION, atomic_write_bytes, cache_key
from .building_semantics import is_actual_church
from .osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPointFeature,
    OsmPolygonFeature,
    road_is_dirt,
    road_is_gravel,
    road_width_metres,
)
from .source_pipeline import FrozenSourceBundle, validate_source_bundle

NORMALIZED_SCHEMA = "cwr-worldgen-normalized-bundle"
NORMALIZED_SCHEMA_VERSION = 19
_ROAD_CLIP_GUARD_METRES = 0.10
_REQUIRED_FILES = (
    "roads.geojson",
    "gravel-roads.geojson",
    "road-junctions.geojson",
    "buildings.geojson",
    "forests.geojson",
    "forest-edges.geojson",
    "water.geojson",
    "watercourses.geojson",
    "landuse.geojson",
    "places.geojson",
    "landmarks.geojson",
    "sites.geojson",
    "barriers.geojson",
    "cutlines.geojson",
    "trees.geojson",
    "aeroway-lines.geojson",
    "aeroway-areas.geojson",
    "utility-points.geojson",
    "surface-areas.geojson",
    "rural-vegetation.geojson",
)

_MAJOR_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road", "track",
}
_MINOR_HIGHWAYS = {"path", "footway", "cycleway", "bridleway", "pedestrian", "steps"}
_WATERCOURSES = {"river", "stream", "canal", "drain", "ditch"}
_FOREST_TAGS = {"forest", "wood"}
_FARMLAND = {
    "farmland", "meadow", "orchard", "vineyard", "grass", "allotments",
    "plant_nursery", "greenhouse_horticulture", "recreation_ground", "village_green",
}
_URBAN = {
    "residential", "commercial", "industrial", "retail", "construction", "farmyard",
    "garages", "railway", "education", "institutional", "civic",
}
_PLACE_RANK = {
    "city": 0,
    "town": 1,
    "village": 2,
    "suburb": 3,
    "quarter": 4,
    "hamlet": 5,
    "isolated_dwelling": 6,
    "locality": 7,
}


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    source_dir: Path
    output_dir: Path | None = None
    refresh: bool = False
    include_minor_roads: bool = False
    road_snap_tolerance: float = 0.75
    road_building_setback: float = 1.5
    building_merge_gap: float = 0.75
    building_overlap_threshold: float = 0.15
    point_building_footprint: float = 12.0
    minimum_building_area: float = 20.0
    forest_edge_width: float = 20.0
    forest_building_clearance: float = 1.0
    minimum_forest_area: float = 200.0
    coordinate_precision: int = 8

    def validate(self) -> None:
        for label, value in (
            ("road snap tolerance", self.road_snap_tolerance),
            ("road-building setback", self.road_building_setback),
            ("building merge gap", self.building_merge_gap),
            ("point-building footprint", self.point_building_footprint),
            ("minimum building area", self.minimum_building_area),
            ("forest edge width", self.forest_edge_width),
            ("forest-building clearance", self.forest_building_clearance),
            ("minimum forest area", self.minimum_forest_area),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not 0.0 <= self.building_overlap_threshold <= 1.0:
            raise ValueError("building overlap threshold must be within 0..1")
        if not 5 <= self.coordinate_precision <= 12:
            raise ValueError("coordinate precision must be within 5..12 decimal places")


@dataclass(frozen=True, slots=True)
class NormalizedBundle:
    root: Path
    manifest_path: Path
    validation_path: Path
    source_fingerprint: str
    normalized_fingerprint: str
    bbox: tuple[float, float, float, float]
    world_size: float
    files: Mapping[str, Path]
    counts: Mapping[str, int]


@dataclass(slots=True)
class _PolygonCandidate:
    geometry: Polygon
    source_ids: set[str]
    properties: dict[str, Any]


@dataclass(slots=True)
class _RoadCandidate:
    geometry: LineString
    source_ids: set[str]
    properties: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _tags(element: Mapping[str, Any]) -> dict[str, str]:
    raw = element.get("tags")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _osm_id(element: Mapping[str, Any]) -> str:
    return f"{element.get('type', 'unknown')}/{element.get('id', 'unknown')}"


def _geometry_points(element: Mapping[str, Any], projection: BboxProjection) -> list[tuple[float, float]]:
    raw = element.get("geometry")
    if not isinstance(raw, list):
        if "lat" in element and "lon" in element:
            raw = [element]
        else:
            return []
    result: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        try:
            latitude = float(item["lat"])
            longitude = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(latitude) and math.isfinite(longitude):
            result.append(projection.to_world((latitude, longitude)))
    return result


def _iter_polygons(geometry: Any) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from _iter_polygons(child)


def _iter_lines(geometry: Any) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from _iter_lines(child)


def _union_geometries(geometries: Iterable[Any]) -> Any:
    items = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not items:
        return GeometryCollection()
    if len(items) == 1:
        return items[0]
    return union_all(items)


def _batched_tree_groups(
    tree: STRtree,
    queries: Sequence[Any],
    *,
    predicate: str = "intersects",
    distance: float | None = None,
    batch_size: int = 2048,
) -> Iterable[tuple[int, list[int]]]:
    """Yield candidate tree indexes for each query without a map-wide pair scan.

    Shapely can query an array of geometries in one GEOS call. Processing fixed
    batches keeps peak memory bounded for very large normalized bundles while
    avoiding one Python-to-GEOS call per feature.
    """

    size = max(1, int(batch_size))
    for start in range(0, len(queries), size):
        batch = queries[start : start + size]
        kwargs: dict[str, Any] = {"predicate": predicate}
        if distance is not None:
            kwargs["distance"] = distance
        pairs = tree.query(batch, **kwargs)
        grouped: dict[int, list[int]] = {}
        if getattr(pairs, "size", 0):
            for left, right in zip(pairs[0].tolist(), pairs[1].tolist()):
                grouped.setdefault(int(left), []).append(int(right))
        for offset in range(len(batch)):
            yield start + offset, grouped.get(offset, [])


def _pairwise_polygon_overlap_area(
    geometries: Sequence[Any],
    *,
    progress_callback: Callable[[int, str], None] | None = None,
    label: str = "geometry",
    chunk_size: int = 16384,
) -> float:
    """Return pairwise positive-area overlap using one STRtree candidate query.

    Normalized buildings should already be collision-free. A global union is a
    disproportionately expensive way to confirm that fact, especially for tens
    of thousands of disjoint footprints. Candidate pairs are spatially indexed
    and exact intersection areas are evaluated in vectorized chunks.
    """

    if len(geometries) < 2:
        return 0.0
    tree = STRtree(geometries)
    pairs = tree.query(geometries, predicate="intersects")
    unique_pairs = [
        (int(left), int(right))
        for left, right in zip(pairs[0].tolist(), pairs[1].tolist())
        if int(right) > int(left)
    ]
    if not unique_pairs:
        return 0.0
    total = 0.0
    size = max(1, int(chunk_size))
    for start in range(0, len(unique_pairs), size):
        chunk = unique_pairs[start : start + size]
        left_items = [geometries[left] for left, _ in chunk]
        right_items = [geometries[right] for _, right in chunk]
        areas = vector_area(vector_intersection(left_items, right_items))
        total += sum(float(value) for value in areas if float(value) > 0.0)
        if progress_callback is not None:
            progress_callback(
                min(100, int((start + len(chunk)) * 100 / len(unique_pairs))),
                f"Checked {start + len(chunk):,}/{len(unique_pairs):,} nearby {label} pairs",
            )
    return total


def _local_overlap_details(
    subjects: Sequence[Any],
    exclusions: Sequence[Any],
    *,
    progress_callback: Callable[[int, str], None] | None = None,
    label: str = "features",
) -> tuple[float, float, list[tuple[int, float]]]:
    """Intersect each subject only with nearby exclusions and avoid double counts."""

    if not subjects or not exclusions:
        return 0.0, 0.0, []
    tree = STRtree(exclusions)
    total = 0.0
    maximum = 0.0
    offenders: list[tuple[int, float]] = []
    interval = max(1, len(subjects) // 50)
    for subject_index, candidate_indexes in _batched_tree_groups(tree, subjects):
        if candidate_indexes:
            nearby = [exclusions[index] for index in sorted(set(candidate_indexes))]
            local_exclusion = nearby[0] if len(nearby) == 1 else union_all(nearby)
            area = float(subjects[subject_index].intersection(local_exclusion).area)
            if area > 1.0e-9:
                total += area
                maximum = max(maximum, area)
                offenders.append((subject_index, area))
        completed = subject_index + 1
        if progress_callback is not None and (completed % interval == 0 or completed == len(subjects)):
            progress_callback(
                min(100, int(completed * 100 / len(subjects))),
                f"Checked {completed:,}/{len(subjects):,} {label}",
            )
    return total, maximum, offenders


def _forest_edge_outside_area(
    edges: Sequence[Any],
    forests: Sequence[Any],
    *,
    tolerance: float = 0.05,
    progress_callback: Callable[[int, str], None] | None = None,
) -> float:
    """Measure crown area outside nearby forests without dissolving all forests."""

    if not edges:
        return 0.0
    if not forests:
        return sum(float(edge.area) for edge in edges)
    tree = STRtree(forests)
    total = 0.0
    interval = max(1, len(edges) // 50)
    for edge_index, candidate_indexes in _batched_tree_groups(
        tree, edges, predicate="dwithin", distance=tolerance
    ):
        edge = edges[edge_index]
        if candidate_indexes:
            nearby = [forests[index] for index in sorted(set(candidate_indexes))]
            local_forest = nearby[0] if len(nearby) == 1 else union_all(nearby)
            total += float(edge.difference(local_forest.buffer(tolerance)).area)
        else:
            total += float(edge.area)
        completed = edge_index + 1
        if progress_callback is not None and (completed % interval == 0 or completed == len(edges)):
            progress_callback(
                min(100, int(completed * 100 / len(edges))),
                f"Checked {completed:,}/{len(edges):,} forest edge crowns",
            )
    return total


def _spatial_union_polygonal(geometries: Iterable[Any], *, dense_pair_factor: int = 64) -> Any:
    """Union polygonal geometry by independent spatial components.

    Most OSM land-use and forest polygons are geographically separate. Sending
    all of them through one global union forces GEOS to build a map-wide noding
    graph. An STRtree first groups only intersecting polygons; disjoint groups
    can be assembled directly into a MultiPolygon. Extremely dense inputs fall
    back to GEOS union_all before the candidate scan becomes pathological.
    """

    items = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not items:
        return GeometryCollection()
    if len(items) == 1:
        return items[0]
    tree = STRtree(items)
    parent = list(range(len(items)))
    rank = [0] * len(items)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merge(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    pair_budget = max(len(items) * max(8, int(dense_pair_factor)), 4096)
    seen_pairs = 0
    for index, geometry in enumerate(items):
        candidates = tree.query(geometry, predicate="intersects")
        seen_pairs += len(candidates)
        if seen_pairs > pair_budget:
            return union_all(items)
        for candidate in candidates:
            candidate_index = int(candidate)
            if candidate_index > index:
                merge(index, candidate_index)

    groups: dict[int, list[Any]] = {}
    for index, geometry in enumerate(items):
        groups.setdefault(find(index), []).append(geometry)
    polygons: list[Polygon] = []
    for group in groups.values():
        merged = group[0] if len(group) == 1 else union_all(group)
        polygons.extend(_iter_polygons(merged))
    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _repair_polygonal(geometry: Any, boundary: Polygon, *, minimum_area: float = 0.01) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    try:
        repaired = make_valid(geometry)
    except Exception:  # noqa: BLE001 - malformed external geometry should be recoverable
        repaired = geometry.buffer(0)
    try:
        clipped = repaired.intersection(boundary)
    except Exception:  # noqa: BLE001
        clipped = make_valid(repaired).intersection(boundary)
    result: list[Polygon] = []
    for polygon in _iter_polygons(clipped):
        if polygon.area < minimum_area:
            continue
        try:
            polygon = set_precision(polygon, 0.001, mode="valid_output")
        except Exception:  # noqa: BLE001
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area >= minimum_area:
            result.append(polygon)
    return result


def _relation_polygon(element: Mapping[str, Any], projection: BboxProjection, boundary: Polygon) -> list[Polygon]:
    members = element.get("members")
    if not isinstance(members, list):
        return []
    outer_lines: list[LineString] = []
    inner_lines: list[LineString] = []
    for member in members:
        if not isinstance(member, Mapping):
            continue
        points = _geometry_points(member, projection)
        if len(points) < 2:
            continue
        line = LineString(points)
        if str(member.get("role", "")) == "inner":
            inner_lines.append(line)
        else:
            outer_lines.append(line)
    if not outer_lines:
        return []
    outer_faces = list(polygonize(unary_union(outer_lines)))
    if not outer_faces:
        outer_faces = [Polygon(line.coords) for line in outer_lines if line.is_ring]
    if not outer_faces:
        return []
    geometry: Any = unary_union(outer_faces)
    if inner_lines:
        inner_faces = list(polygonize(unary_union(inner_lines)))
        if not inner_faces:
            inner_faces = [Polygon(line.coords) for line in inner_lines if line.is_ring]
        if inner_faces:
            geometry = geometry.difference(unary_union(inner_faces))
    return _repair_polygonal(geometry, boundary)


def _element_polygons(element: Mapping[str, Any], projection: BboxProjection, boundary: Polygon) -> list[Polygon]:
    if element.get("type") == "relation":
        return _relation_polygon(element, projection, boundary)
    points = _geometry_points(element, projection)
    if len(points) < 3:
        return []
    if points[0] != points[-1]:
        points.append(points[0])
    return _repair_polygonal(Polygon(points), boundary)


def _element_lines(element: Mapping[str, Any], projection: BboxProjection, boundary: Polygon) -> list[LineString]:
    points = _geometry_points(element, projection)
    if len(points) < 2:
        return []
    line = LineString(points)
    clipped = line.intersection(boundary)
    return [candidate for candidate in _iter_lines(clipped) if candidate.length > 0.05]


def _truthy_tag(tags: Mapping[str, str], key: str) -> bool:
    return tags.get(key, "").casefold() not in {"", "no", "false", "0"}


def _parse_number(value: str | None, default: float = 0.0) -> float:
    if not value:
        return default
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value.replace(" ", ""))
    if not match:
        return default
    try:
        number = float(match.group(0).replace(",", "."))
    except ValueError:
        return default
    return number if math.isfinite(number) else default


def _road_properties(tags: Mapping[str, str]) -> dict[str, Any]:
    highway = tags.get("highway", "road")
    width = _parse_number(tags.get("width"), road_width_metres(tags))
    width = max(1.0, min(40.0, width))
    bridge = _truthy_tag(tags, "bridge") or tags.get("man_made") == "bridge"
    tunnel = _truthy_tag(tags, "tunnel") or _truthy_tag(tags, "covered")
    embankment = _truthy_tag(tags, "embankment")
    if bridge:
        special = "bridge"
    elif tunnel:
        special = "tunnel"
    elif embankment:
        special = "embankment"
    else:
        special = "normal"
    return {
        "highway": highway,
        "surface": tags.get("surface", ""),
        "width_m": round(width, 3),
        "special": special,
        "bridge": bridge,
        "tunnel": tunnel,
        "embankment": embankment,
        "layer": int(round(_parse_number(tags.get("layer"), 0.0))),
        "oneway": tags.get("oneway", "").casefold() in {"yes", "true", "1", "-1"},
        "dirt": road_is_dirt(tags),
    }


def _snap_line(line: LineString, tolerance: float) -> LineString | None:
    if tolerance <= 0:
        return line if line.length > 0.05 else None
    coordinates: list[tuple[float, float]] = []
    for x, y in line.coords:
        snapped = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if not coordinates or snapped != coordinates[-1]:
            coordinates.append(snapped)
    if len(coordinates) < 2:
        return None
    result = LineString(coordinates)
    return result if result.length > 0.05 else None


def _road_group_key(properties: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        properties["highway"], properties["surface"], properties["width_m"],
        properties["special"], properties["layer"], properties["oneway"], properties["dirt"],
    )


def _normalize_roads(
    elements: Sequence[Mapping[str, Any]],
    projection: BboxProjection,
    boundary: Polygon,
    spec: NormalizationSpec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[_RoadCandidate]:
    """Normalize roads with indexed source attribution.

    Older releases compared every merged line against every original candidate
    in its style group. That quadratic attribution step dominated large source
    bundles. An STRtree now limits distance checks to nearby candidates.
    """
    grouped: dict[tuple[Any, ...], list[_RoadCandidate]] = {}
    allowed = set(_MAJOR_HIGHWAYS)
    if spec.include_minor_roads:
        allowed.update(_MINOR_HIGHWAYS)

    total_elements = len(elements)
    road_element_count = 0
    snapped_segment_count = 0
    report_every = max(1000, total_elements // 100 if total_elements else 1000)
    for element_index, element in enumerate(elements, start=1):
        tags = _tags(element)
        if tags.get("highway") not in allowed:
            if progress_callback is not None and element_index % report_every == 0:
                progress_callback(
                    min(35, int(element_index * 35 / max(1, total_elements))),
                    f"Scanning roads {element_index:,}/{total_elements:,}; accepted {road_element_count:,} ways",
                )
            continue
        road_element_count += 1
        properties = _road_properties(tags)
        source_id = _osm_id(element)
        for line in _element_lines(element, projection, boundary):
            snapped = _snap_line(line, spec.road_snap_tolerance)
            if snapped is None:
                continue
            candidate = _RoadCandidate(snapped, {source_id}, dict(properties))
            grouped.setdefault(_road_group_key(properties), []).append(candidate)
            snapped_segment_count += 1
        if progress_callback is not None and element_index % report_every == 0:
            progress_callback(
                min(35, int(element_index * 35 / max(1, total_elements))),
                f"Scanning roads {element_index:,}/{total_elements:,}; snapped {snapped_segment_count:,} segments",
            )

    merged_roads: list[_RoadCandidate] = []
    ordered_groups = sorted(grouped, key=lambda item: tuple(str(value) for value in item))
    group_count = len(ordered_groups)
    attribution_distance = max(0.01, spec.road_snap_tolerance)
    for group_index, key in enumerate(ordered_groups, start=1):
        candidates = grouped[key]
        geometries = [candidate.geometry for candidate in candidates]
        unioned = unary_union(geometries)
        merged = linemerge(unioned) if isinstance(unioned, MultiLineString) else unioned
        merged_lines = tuple(_iter_lines(merged))

        # STRtree.query returns integer indexes under Shapely 2.x. The dwithin
        # predicate performs the expensive distance filtering inside GEOS.
        tree = STRtree(geometries)
        for line in merged_lines:
            nearby_indexes = tree.query(line, predicate="dwithin", distance=attribution_distance)
            source_ids: set[str] = set()
            for candidate_index in nearby_indexes:
                source_ids.update(candidates[int(candidate_index)].source_ids)
            # Precision edge cases should not erase provenance. Querying by the
            # expanded envelope remains indexed and is still far cheaper than a
            # full group scan.
            if not source_ids:
                nearby_indexes = tree.query(line.buffer(attribution_distance).envelope)
                for candidate_index in nearby_indexes:
                    candidate = candidates[int(candidate_index)]
                    if candidate.geometry.distance(line) <= attribution_distance:
                        source_ids.update(candidate.source_ids)
            merged_roads.append(_RoadCandidate(line, source_ids, dict(candidates[0].properties)))

        if progress_callback is not None:
            progress_callback(
                35 + int(group_index * 60 / max(1, group_count)),
                f"Merging road groups {group_index:,}/{group_count:,}; produced {len(merged_roads):,} lines",
            )

    merged_roads.sort(key=lambda road: (road.properties["layer"], road.properties["special"], road.geometry.wkb_hex))
    for index, road in enumerate(merged_roads, start=1):
        road.properties["road_id"] = f"road-{index:06d}"
        road.properties["source_ids"] = sorted(road.source_ids)
        road.properties["length_m"] = round(road.geometry.length, 3)
    if progress_callback is not None:
        progress_callback(100, f"Normalized {road_element_count:,} road ways into {len(merged_roads):,} merged lines")
    return merged_roads


def _road_corridor(roads: Sequence[_RoadCandidate], setback: float) -> Any:
    active = [road for road in roads if not road.properties.get("tunnel")]
    if not active:
        return GeometryCollection()
    geometries = [road.geometry for road in active]
    distances = [float(road.properties["width_m"]) / 2.0 + setback for road in active]
    # Shapely's vectorized buffer avoids thousands of Python-to-GEOS calls.
    # union_all is the Shapely 2 equivalent of unary_union for geometry arrays.
    buffers = vector_buffer(geometries, distances, cap_style="flat", join_style="mitre")
    return _union_geometries(buffers)


def _junction_features(
    roads: Sequence[_RoadCandidate],
    tolerance: float,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[tuple[Point, dict[str, Any]]]:
    """Build noded road junctions with batched indexed road attribution.

    Older releases performed a full candidate-road scan for every noded segment
    endpoint. Large road networks therefore spent most of normalization in an
    O(nodes * roads) loop immediately after the road-merge progress message.
    STRtree now attributes all unique node points to nearby roads in one batched
    GEOS query per layer.
    """

    quant = max(0.05, tolerance)
    groups: dict[int, list[_RoadCandidate]] = {}
    for road in roads:
        if road.properties.get("bridge") or road.properties.get("tunnel"):
            continue
        groups.setdefault(int(road.properties.get("layer", 0)), []).append(road)

    nodes: dict[tuple[int, int, int], dict[str, Any]] = {}
    ordered_groups = sorted(groups.items())
    group_count = len(ordered_groups)
    for group_index, (layer, candidates) in enumerate(ordered_groups, start=1):
        geometries = [candidate.geometry for candidate in candidates]
        if progress_callback is not None:
            progress_callback(
                int((group_index - 1) * 45 / max(1, group_count)),
                f"Noding road layer {group_index:,}/{group_count:,} ({len(candidates):,} lines)",
            )
        noded = unary_union(geometries)
        segments = [line for line in _iter_lines(noded) if line.length > 0.05]
        layer_records: dict[tuple[int, int, int], dict[str, Any]] = {}
        for segment in segments:
            for x, y in (segment.coords[0], segment.coords[-1]):
                key = (layer, round(x / quant), round(y / quant))
                record = layer_records.setdefault(
                    key,
                    {"x": x, "y": y, "degree": 0, "road_ids": set(), "layer": layer},
                )
                record["degree"] += 1

        # Attribute all unique nodes to source roads in one vectorized STRtree
        # query. For Shapely 2, querying a geometry array returns two index rows:
        # input node indexes and matching tree geometry indexes.
        records = list(layer_records.values())
        if records and geometries:
            points = [Point(record["x"], record["y"]) for record in records]
            tree = STRtree(geometries)
            pairs = tree.query(points, predicate="dwithin", distance=quant)
            if getattr(pairs, "ndim", 1) == 2 and len(pairs) == 2:
                for point_index, candidate_index in zip(pairs[0], pairs[1]):
                    records[int(point_index)]["road_ids"].add(
                        candidates[int(candidate_index)].properties["road_id"]
                    )
            else:
                # Defensive fallback for unusual Shapely builds. This remains
                # spatially indexed and never returns to the old full scan.
                for point_index, point in enumerate(points):
                    nearby_indexes = tree.query(point, predicate="dwithin", distance=quant)
                    for candidate_index in nearby_indexes:
                        records[point_index]["road_ids"].add(
                            candidates[int(candidate_index)].properties["road_id"]
                        )

        nodes.update(layer_records)
        if progress_callback is not None:
            progress_callback(
                45 + int(group_index * 55 / max(1, group_count)),
                f"Indexed road junctions for layer {group_index:,}/{group_count:,}; {len(nodes):,} nodes",
            )

    result: list[tuple[Point, dict[str, Any]]] = []
    ordered = sorted(nodes.values(), key=lambda value: (value["layer"], value["x"], value["y"]))
    index = 1
    for record in ordered:
        degree = int(record["degree"])
        if degree == 2:
            continue
        properties = {
            "junction_id": f"junction-{index:06d}",
            "degree": degree,
            "kind": "endpoint" if degree == 1 else "junction",
            "layer": record["layer"],
            "road_ids": sorted(record["road_ids"]),
        }
        result.append((Point(record["x"], record["y"]), properties))
        index += 1
    if progress_callback is not None:
        progress_callback(100, f"Generated {len(result):,} road junction and endpoint records")
    return result


def _clean_name(value: str) -> tuple[str, str]:
    display = " ".join(unicodedata.normalize("NFKC", value).split())
    display = "".join(character for character in display if unicodedata.category(character)[0] != "C").strip()
    ascii_name = unicodedata.normalize("NFKD", display).encode("ascii", "ignore").decode("ascii")
    ascii_name = " ".join(ascii_name.split()).strip() or "Unnamed"
    return display or "Unnamed", ascii_name


def _polygon_kind(tags: Mapping[str, str]) -> str:
    return tags.get("building") or tags.get("man_made") or "yes"

_BUILDING_METADATA_TAGS = (
    "height",
    "building:levels",
    "roof:shape",
    "roof:levels",
    "min_height",
    "building:material",
    "roof:material",
    "amenity",
    "social_facility",
    "social_facility:for",
    "shop",
    "religion",
    "denomination",
    "office",
    "building:use",
)


def _building_properties(tags: Mapping[str, str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "building_kind": _polygon_kind(tags),
        "name": tags.get("name", ""),
    }
    for key in _BUILDING_METADATA_TAGS:
        value = tags.get(key)
        if value:
            properties[key] = value
    return properties


def _semantic_building_kind(tags: Mapping[str, str]) -> str:
    amenity = str(tags.get("amenity", "")).casefold()
    building = str(tags.get("building", "")).casefold()
    if is_actual_church(tags):
        return "church"
    if amenity == "school" or building in {"school", "kindergarten", "college", "university"}:
        return "school"
    if amenity == "social_facility" or bool(tags.get("social_facility")):
        return "social_facility"
    if tags.get("shop") or building in {"retail", "supermarket", "kiosk"}:
        return "shop"
    return ""


def _apply_semantic_building_tags(
    properties: dict[str, Any], tags: Mapping[str, str], semantic_kind: str
) -> None:
    """Transfer POI semantics onto the real containing building footprint.

    OSM commonly stores a shop, school, or place-of-worship tag on a node or
    campus polygon while the physical footprint is merely ``building=yes``.
    Keeping the geometry but discarding the POI turns every civic building into
    an ordinary house during procedural model selection.
    """

    semantic = _building_properties(tags)
    if semantic.get("name"):
        properties["name"] = semantic["name"]
    for key in _BUILDING_METADATA_TAGS:
        value = semantic.get(key)
        if value not in {None, ""}:
            properties[key] = value
    properties["building_kind"] = {
        "church": "church",
        "school": "school",
        "social_facility": "public",
        "shop": "retail",
    }[semantic_kind]


def _merge_polygon_candidates(
    candidates: Sequence[_PolygonCandidate],
    gap: float,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[_PolygonCandidate]:
    """Merge overlapping and nearby small building polygons using an STRtree.

    The previous implementation rescanned every remaining building after each
    accepted candidate, giving large extracts quadratic behaviour even when most
    footprints were nowhere near one another. A static spatial index narrows each
    pass to geometries whose envelopes can actually overlap or fall within the
    configured small-building merge gap.
    """

    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (-item.geometry.area, item.geometry.wkb_hex))
    tree = STRtree([item.geometry for item in ordered])
    active = [True] * len(ordered)
    result: list[_PolygonCandidate] = []
    processed = 0
    report_interval = max(1, len(ordered) // 100)

    for root_index, seed in enumerate(ordered):
        if not active[root_index]:
            continue
        current = seed
        active[root_index] = False
        while True:
            min_x, min_y, max_x, max_y = current.geometry.bounds
            expansion = gap if current.geometry.area < 60.0 else 0.0
            search_geometry = box(
                min_x - expansion, min_y - expansion,
                max_x + expansion, max_y + expansion,
            )
            nearby = sorted(
                int(index) for index in tree.query(search_geometry)
                if active[int(index)]
            )
            changed = False
            for other_index in nearby:
                other = ordered[other_index]
                if current.properties.get("building_kind") != other.properties.get("building_kind"):
                    continue
                overlap = current.geometry.intersects(other.geometry)
                both_small = current.geometry.area < 60.0 and other.geometry.area < 60.0
                close_small = both_small and current.geometry.distance(other.geometry) <= gap
                if not (overlap or close_small):
                    continue
                merged = union_all([current.geometry, other.geometry])
                polygons = list(_iter_polygons(make_valid(merged)))
                if not polygons:
                    continue
                current.geometry = max(polygons, key=lambda polygon: polygon.area)
                current.source_ids.update(other.source_ids)
                active[other_index] = False
                changed = True
            if not changed:
                break
        result.append(current)
        processed += 1
        if progress_callback is not None and (processed % report_interval == 0 or processed == len(ordered)):
            progress_callback(
                min(100, int((root_index + 1) * 100 / len(ordered))),
                f"Spatially merged {root_index + 1:,}/{len(ordered):,} building candidates into {len(result):,} footprints",
            )
    if progress_callback is not None:
        progress_callback(100, f"Merged {len(ordered):,} building candidates into {len(result):,} footprints")
    return result


def _oriented_box_properties(polygon: Polygon) -> dict[str, float]:
    rectangle = polygon.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    edges: list[tuple[float, float, float]] = []
    for start, end in zip(coordinates, coordinates[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        edges.append((length, dx, dy))
    edges.sort(reverse=True, key=lambda item: item[0])
    length, dx, dy = edges[0]
    width = edges[-1][0]
    heading = math.degrees(math.atan2(dy, dx)) % 180.0
    return {"length_m": round(length, 3), "width_m": round(width, 3), "heading_deg": round(heading, 3)}


def _normalize_buildings(
    elements: Sequence[Mapping[str, Any]],
    projection: BboxProjection,
    boundary: Polygon,
    roads: Sequence[_RoadCandidate],
    spec: NormalizationSpec,
    *,
    road_corridor: Any | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[list[_PolygonCandidate], dict[str, int]]:
    candidates: list[_PolygonCandidate] = []
    semantic_anchors: list[tuple[Any, dict[str, str], str, str]] = []
    semantic_areas: list[tuple[Polygon, dict[str, str], str, str]] = []
    statistics = {
        "input": 0,
        "road_removed": 0,
        "collision_removed": 0,
        "sliver_removed": 0,
        "semantic_attached": 0,
        "semantic_synthetic": 0,
    }
    element_total = len(elements)
    scan_interval = max(1, element_total // 100)

    # Avoid allocating a normalized tag dictionary for every untagged OSM node.
    # Large Overpass extracts contain many such elements and only a small fraction
    # can possibly become buildings. Semantic POIs are retained separately so a
    # node inside ``building=yes`` can annotate that real footprint instead of
    # becoming a synthetic rectangle that collision cleanup later deletes.
    for element_index, element in enumerate(elements, start=1):
        raw_tags = element.get("tags")
        if not isinstance(raw_tags, Mapping):
            if progress_callback is not None and element_index % scan_interval == 0:
                progress_callback(
                    int(element_index * 20 / max(1, element_total)),
                    f"Scanning building elements {element_index:,}/{element_total:,}; found {len(candidates):,} candidates",
                )
            continue
        tags = {str(key): str(value) for key, value in raw_tags.items()}
        semantic_kind = _semantic_building_kind(tags)
        building_value = str(raw_tags.get("building", "")).strip().casefold()
        has_building = bool(building_value) and building_value not in {"no", "false", "0", "none"}
        has_man_made = bool(raw_tags.get("man_made"))
        if semantic_kind in {"school", "social_facility"}:
            for semantic_polygon in _element_polygons(element, projection, boundary):
                if not semantic_polygon.is_empty:
                    semantic_areas.append((semantic_polygon, tags, _osm_id(element), semantic_kind))
        if not has_building and not has_man_made and not semantic_kind:
            if progress_callback is not None and element_index % scan_interval == 0:
                progress_callback(
                    int(element_index * 20 / max(1, element_total)),
                    f"Scanning building elements {element_index:,}/{element_total:,}; found {len(candidates):,} candidates",
                )
            continue

        if semantic_kind and not has_building and not has_man_made:
            semantic_geometry: Any | None = None
            semantic_polygons = _element_polygons(element, projection, boundary)
            if semantic_polygons:
                semantic_geometry = max(semantic_polygons, key=lambda polygon: polygon.area)
            else:
                points = _geometry_points(element, projection)
                if len(points) == 1:
                    semantic_geometry = Point(points[0])
            if semantic_geometry is not None and not semantic_geometry.is_empty:
                semantic_anchors.append((semantic_geometry, tags, _osm_id(element), semantic_kind))
            if progress_callback is not None and (element_index % scan_interval == 0 or element_index == element_total):
                progress_callback(
                    int(element_index * 20 / max(1, element_total)),
                    f"Scanning building elements {element_index:,}/{element_total:,}; found {len(candidates):,} footprints and {len(semantic_anchors):,} semantic POIs",
                )
            continue

        polygons = _element_polygons(element, projection, boundary)
        if not polygons:
            points = _geometry_points(element, projection)
            if len(points) == 1:
                x, y = points[0]
                width, length = {
                    "church": (14.0, 24.0),
                    "school": (18.0, 32.0),
                    "social_facility": (16.0, 24.0),
                    "shop": (10.0, 14.0),
                }.get(semantic_kind, (spec.point_building_footprint, spec.point_building_footprint))
                polygons = _repair_polygonal(
                    box(x - width / 2.0, y - length / 2.0, x + width / 2.0, y + length / 2.0),
                    boundary,
                )
        for polygon in polygons:
            statistics["input"] += 1
            if polygon.area < spec.minimum_building_area:
                statistics["sliver_removed"] += 1
                continue
            candidates.append(_PolygonCandidate(
                polygon,
                {_osm_id(element)},
                _building_properties(tags),
            ))
        if progress_callback is not None and (element_index % scan_interval == 0 or element_index == element_total):
            progress_callback(
                int(element_index * 20 / max(1, element_total)),
                f"Scanning building elements {element_index:,}/{element_total:,}; found {len(candidates):,} footprints and {len(semantic_anchors):,} semantic POIs",
            )

    if semantic_anchors:
        if progress_callback is not None:
            progress_callback(21, f"Matching {len(semantic_anchors):,} civic, school, shop and worship POIs to building footprints")
        geometries = [candidate.geometry for candidate in candidates]
        tree = STRtree(geometries) if geometries else None
        semantic_dimensions = {
            "church": (14.0, 24.0),
            "school": (18.0, 32.0),
            "social_facility": (16.0, 24.0),
            "shop": (10.0, 14.0),
        }
        semantic_priority = {"church": 0, "school": 1, "social_facility": 2, "shop": 3}
        for anchor, tags, source_id, semantic_kind in sorted(
            semantic_anchors,
            key=lambda item: (semantic_priority[item[3]], item[2]),
        ):
            matched_index: int | None = None
            if tree is not None and isinstance(anchor, Point):
                exact = [
                    int(index) for index in tree.query(anchor)
                    if candidates[int(index)].geometry.covers(anchor)
                ]
                if exact:
                    matched_index = min(exact, key=lambda index: (candidates[index].geometry.area, index))
                else:
                    nearby = [int(index) for index in tree.query(anchor, predicate="dwithin", distance=20.0)]
                    if nearby:
                        matched_index = min(
                            nearby,
                            key=lambda index: (
                                candidates[index].geometry.distance(anchor),
                                candidates[index].geometry.area,
                                index,
                            ),
                        )
            elif tree is not None:
                nearby = [int(index) for index in tree.query(anchor, predicate="intersects")]
                inside = [
                    index for index in nearby
                    if anchor.covers(candidates[index].geometry.representative_point())
                ]
                if inside:
                    if semantic_kind in {"school", "social_facility"}:
                        # School and social-facility campuses are semantic
                        # areas, not merely hints for one largest footprint. Every
                        # physical building inside the mapped area inherits the
                        # amenity so generic/large footprints cannot fall through
                        # to barn or warehouse heuristics.
                        for index in sorted(inside):
                            candidate = candidates[index]
                            _apply_semantic_building_tags(candidate.properties, tags, semantic_kind)
                            candidate.source_ids.add(source_id)
                        statistics["semantic_attached"] += len(inside)
                        continue
                    # Church/shop areas normally correspond to one principal
                    # footprint, so retain the largest-footprint behaviour.
                    matched_index = max(inside, key=lambda index: (candidates[index].geometry.area, -index))

            if matched_index is not None:
                candidate = candidates[matched_index]
                _apply_semantic_building_tags(candidate.properties, tags, semantic_kind)
                candidate.source_ids.add(source_id)
                statistics["semantic_attached"] += 1
                continue

            width, length = semantic_dimensions[semantic_kind]
            point = anchor if isinstance(anchor, Point) else anchor.representative_point()
            synthetic = box(
                point.x - width / 2.0,
                point.y - length / 2.0,
                point.x + width / 2.0,
                point.y + length / 2.0,
            ).intersection(boundary)
            polygons = [
                polygon for polygon in _iter_polygons(make_valid(synthetic))
                if polygon.area >= spec.minimum_building_area
            ]
            if polygons:
                properties = _building_properties(tags)
                _apply_semantic_building_tags(properties, tags, semantic_kind)
                candidates.append(_PolygonCandidate(
                    max(polygons, key=lambda polygon: polygon.area),
                    {source_id},
                    properties,
                ))
                statistics["input"] += 1
                statistics["semantic_synthetic"] += 1

    if progress_callback is not None:
        progress_callback(22, f"Spatially indexing {len(candidates):,} building candidates")
    candidates = _merge_polygon_candidates(
        candidates,
        spec.building_merge_gap,
        progress_callback=(
            (lambda value, message: progress_callback(22 + int(value * 28 / 100), message))
            if progress_callback is not None else None
        ),
    )

    # Build an index of individually buffered road pieces. Clipping each building
    # against the full map-wide dissolved road corridor forces GEOS to inspect a
    # needlessly complex polygon. Querying only local road buffers keeps overlay
    # work proportional to nearby roads instead of the entire road network.
    active_roads = [road for road in roads if not road.properties.get("tunnel")]
    clipping_buffers: list[Any] = []
    clipping_tree: STRtree | None = None
    if active_roads:
        if progress_callback is not None:
            progress_callback(52, f"Indexing {len(active_roads):,} road buffers for local building clipping")
        base_buffers = vector_buffer(
            [road.geometry for road in active_roads],
            [float(road.properties["width_m"]) / 2.0 + spec.road_building_setback for road in active_roads],
            cap_style="flat",
            join_style="mitre",
        )
        guarded_buffers = vector_buffer(base_buffers, _ROAD_CLIP_GUARD_METRES, join_style="mitre")
        clipping_buffers = [geometry for geometry in guarded_buffers if geometry is not None and not geometry.is_empty]
        if clipping_buffers:
            clipping_tree = STRtree(clipping_buffers)

    fallback_corridor = GeometryCollection()
    if clipping_tree is None:
        corridor = road_corridor if road_corridor is not None else _road_corridor(roads, spec.road_building_setback)
        fallback_corridor = (
            corridor.buffer(_ROAD_CLIP_GUARD_METRES, join_style=2)
            if not corridor.is_empty else corridor
        )

    cropped: list[_PolygonCandidate] = []
    clip_total = len(candidates)
    clip_interval = max(1, clip_total // 100)
    for candidate_index, candidate in enumerate(candidates, start=1):
        original_area = candidate.geometry.area
        geometry = candidate.geometry
        if clipping_tree is not None:
            nearby = clipping_tree.query(candidate.geometry, predicate="intersects")
            if len(nearby):
                local_corridor = union_all([clipping_buffers[int(index)] for index in nearby])
                geometry = candidate.geometry.difference(local_corridor)
        elif not fallback_corridor.is_empty:
            geometry = candidate.geometry.difference(fallback_corridor)
        pieces = [polygon for polygon in _iter_polygons(make_valid(geometry)) if polygon.area >= spec.minimum_building_area]
        if not pieces:
            statistics["road_removed"] += 1
        else:
            largest = max(pieces, key=lambda polygon: polygon.area)
            if largest.area < original_area * 0.35:
                statistics["road_removed"] += 1
            else:
                candidate.geometry = largest
                cropped.append(candidate)
        if progress_callback is not None and (candidate_index % clip_interval == 0 or candidate_index == clip_total):
            progress_callback(
                55 + int(candidate_index * 20 / max(1, clip_total)),
                f"Clipping buildings near roads {candidate_index:,}/{clip_total:,}; retained {len(cropped):,}",
            )

    # Deterministic largest-first collision cleanup. Spatial bucketing keeps this
    # linear for normal map densities. The overlap union is constructed once per
    # candidate and reused for both the ratio test and subtraction.
    accepted: list[_PolygonCandidate] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    bucket_size = 50.0

    def bucket_keys(polygon: Polygon) -> Iterable[tuple[int, int]]:
        min_x, min_y, max_x, max_y = polygon.bounds
        for x in range(math.floor(min_x / bucket_size), math.floor(max_x / bucket_size) + 1):
            for y in range(math.floor(min_y / bucket_size), math.floor(max_y / bucket_size) + 1):
                yield x, y

    ordered_cropped = sorted(cropped, key=lambda item: (-item.geometry.area, item.geometry.wkb_hex))
    cleanup_total = len(ordered_cropped)
    cleanup_interval = max(1, cleanup_total // 100)
    for cleanup_index, candidate in enumerate(ordered_cropped, start=1):
        nearby_indices = sorted({index for key in bucket_keys(candidate.geometry) for index in buckets.get(key, [])})
        overlapping = [accepted[index].geometry for index in nearby_indices if accepted[index].geometry.intersects(candidate.geometry)]
        if overlapping:
            overlap_union = union_all(overlapping)
            overlap = candidate.geometry.intersection(overlap_union).area
            if overlap / max(candidate.geometry.area, 1e-9) > spec.building_overlap_threshold:
                statistics["collision_removed"] += 1
                if progress_callback is not None and (cleanup_index % cleanup_interval == 0 or cleanup_index == cleanup_total):
                    progress_callback(
                        76 + int(cleanup_index * 20 / max(1, cleanup_total)),
                        f"Resolving building overlaps {cleanup_index:,}/{cleanup_total:,}; accepted {len(accepted):,}",
                    )
                continue
            reduced = candidate.geometry.difference(overlap_union)
            pieces = [polygon for polygon in _iter_polygons(make_valid(reduced)) if polygon.area >= spec.minimum_building_area]
            if not pieces:
                statistics["collision_removed"] += 1
                continue
            candidate.geometry = max(pieces, key=lambda polygon: polygon.area)
        index = len(accepted)
        accepted.append(candidate)
        for key in bucket_keys(candidate.geometry):
            buckets.setdefault(key, []).append(index)
        if progress_callback is not None and (cleanup_index % cleanup_interval == 0 or cleanup_index == cleanup_total):
            progress_callback(
                76 + int(cleanup_index * 20 / max(1, cleanup_total)),
                f"Resolving building overlaps {cleanup_index:,}/{cleanup_total:,}; accepted {len(accepted):,}",
            )

    # Make school/social-facility campus membership authoritative after all
    # geometry cleanup. This catches generic wings and prevents a semantic campus
    # envelope tagged building=no from becoming or leaving unrelated farm/industrial
    # buildings inside the area.
    if semantic_areas and accepted:
        accepted_tree = STRtree([candidate.geometry for candidate in accepted])
        for semantic_area, semantic_tags, source_id, semantic_kind in semantic_areas:
            for candidate_index in accepted_tree.query(semantic_area, predicate="intersects"):
                candidate = accepted[int(candidate_index)]
                if semantic_area.covers(candidate.geometry.representative_point()):
                    _apply_semantic_building_tags(candidate.properties, semantic_tags, semantic_kind)
                    candidate.source_ids.add(source_id)

    accepted.sort(key=lambda item: (item.geometry.centroid.x, item.geometry.centroid.y, item.geometry.wkb_hex))
    for index, candidate in enumerate(accepted, start=1):
        candidate.properties.update({
            "building_id": f"building-{index:06d}",
            "source_ids": sorted(candidate.source_ids),
            "area_m2": round(candidate.geometry.area, 3),
            **_oriented_box_properties(candidate.geometry),
        })
    if progress_callback is not None:
        progress_callback(100, f"Cleaned {statistics['input']:,} source footprints into {len(accepted):,} buildings")
    return accepted, statistics


def _coastline_ocean(lines: Sequence[LineString], boundary: Polygon) -> list[Polygon]:
    if not lines:
        return []
    clipped = [line.intersection(boundary) for line in lines]
    clipped_lines = [line for geometry in clipped for line in _iter_lines(geometry) if line.length > 0.05]
    if not clipped_lines:
        return []
    faces = list(polygonize(unary_union([boundary.boundary, *clipped_lines])))
    if not faces:
        return []

    segments: list[tuple[LineString, tuple[float, float], tuple[float, float]]] = []
    for line in clipped_lines:
        coords = list(line.coords)
        for start, end in zip(coords, coords[1:]):
            segment = LineString([start, end])
            if segment.length > 0.01:
                segments.append((segment, start, end))

    ocean: list[Polygon] = []
    for face in faces:
        point = face.representative_point()
        segment, start, end = min(segments, key=lambda item: item[0].distance(point))
        cross = (end[0] - start[0]) * (point.y - start[1]) - (end[1] - start[1]) * (point.x - start[0])
        # OSM coastline direction has land on the left and water on the right.
        if cross < 0:
            ocean.append(face)
    return _repair_polygonal(unary_union(ocean), boundary)


def _extract_category_polygons(
    elements: Sequence[Mapping[str, Any]],
    projection: BboxProjection,
    boundary: Polygon,
    predicate,
) -> list[tuple[Polygon, str, Mapping[str, str]]]:
    result: list[tuple[Polygon, str, Mapping[str, str]]] = []
    for element in elements:
        tags = _tags(element)
        category = predicate(tags)
        if category is None:
            continue
        for polygon in _element_polygons(element, projection, boundary):
            result.append((polygon, _osm_id(element), tags | {"_category": category}))
    return result


def _normalize_forest(
    forest_inputs: Sequence[tuple[Polygon, str, Mapping[str, str]]],
    water: Any,
    landuse_priority: Any,
    road_corridor: Any,
    cutline_corridor: Any,
    buildings: Sequence[_PolygonCandidate],
    boundary: Polygon,
    spec: NormalizationSpec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[list[_PolygonCandidate], list[_PolygonCandidate], dict[str, float]]:
    def report(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), message)

    report(5, f"Unioning {len(forest_inputs):,} source forest polygons")
    original = _spatial_union_polygonal(polygon for polygon, _, _ in forest_inputs)
    repaired_original = _repair_polygonal(
        original, boundary, minimum_area=spec.minimum_forest_area
    )
    original = _spatial_union_polygonal(repaired_original)

    report(25, f"Building forest exclusions from {len(buildings):,} buildings")
    if buildings:
        building_buffers = vector_buffer(
            [building.geometry for building in buildings],
            spec.forest_building_clearance,
            join_style="mitre",
        )
        building_priority = _spatial_union_polygonal(building_buffers)
    else:
        building_priority = GeometryCollection()
    priorities = [
        geometry
        for geometry in (water, landuse_priority, road_corridor, cutline_corridor, building_priority)
        if not geometry.is_empty
    ]
    exclusions = _union_geometries(priorities)

    report(45, "Subtracting water, roads, land use, cutlines and buildings from forests")
    cleaned = original.difference(exclusions) if not original.is_empty else original
    cleaned_polygons = _repair_polygonal(
        cleaned, boundary, minimum_area=spec.minimum_forest_area
    )

    report(65, f"Generating edge crowns for {len(cleaned_polygons):,} cleaned forest polygons")
    if spec.forest_edge_width > 0 and cleaned_polygons:
        interior_geometries = vector_buffer(
            cleaned_polygons, -spec.forest_edge_width, join_style="mitre"
        )
        edge_geometries = vector_difference(
            cleaned_polygons, interior_geometries, grid_size=0.001
        )
    else:
        interior_geometries = cleaned_polygons
        edge_geometries = [GeometryCollection() for _ in cleaned_polygons]

    forest_features: list[_PolygonCandidate] = []
    edge_features: list[_PolygonCandidate] = []
    polygon_total = max(1, len(cleaned_polygons))
    for polygon_index, (polygon, interior_geometry, edge_geometry) in enumerate(
        zip(cleaned_polygons, interior_geometries, edge_geometries), start=1
    ):
        if interior_geometry is None or interior_geometry.is_empty:
            interior_area = 0.0
        else:
            if not interior_geometry.is_valid:
                interior_geometry = make_valid(interior_geometry)
            interior_area = sum(part.area for part in _iter_polygons(interior_geometry))
        forest_features.append(_PolygonCandidate(polygon, set(), {
            "area_m2": round(polygon.area, 3),
            "interior_area_m2": round(interior_area, 3),
            "edge_area_m2": round(max(0.0, polygon.area - interior_area), 3),
        }))
        if edge_geometry is None or edge_geometry.is_empty:
            continue
        if not edge_geometry.is_valid:
            edge_geometry = make_valid(edge_geometry)
        for edge in _iter_polygons(edge_geometry):
            if edge.area >= 1.0:
                edge_features.append(_PolygonCandidate(edge, set(), {"area_m2": round(edge.area, 3)}))
        if polygon_index % 1000 == 0:
            report(65 + int(polygon_index * 30 / polygon_total), f"Generated forest crowns {polygon_index:,}/{len(cleaned_polygons):,}")

    forest_features.sort(key=lambda item: (item.geometry.bounds, round(item.geometry.area, 6)))
    edge_features.sort(key=lambda item: (item.geometry.bounds, round(item.geometry.area, 6)))
    for index, candidate in enumerate(forest_features, start=1):
        candidate.properties["forest_id"] = f"forest-{index:06d}"
    for index, candidate in enumerate(edge_features, start=1):
        candidate.properties["edge_id"] = f"forest-edge-{index:06d}"
    original_area = original.area if not original.is_empty else 0.0
    cleaned_area = sum(item.geometry.area for item in forest_features)
    statistics = {
        "original_area_m2": round(original_area, 3),
        "cleaned_area_m2": round(cleaned_area, 3),
        "clearings_area_m2": round(max(0.0, original_area - cleaned_area), 3),
    }
    report(100, f"Cleaned {len(forest_features):,} forests and generated {len(edge_features):,} edge crowns")
    return forest_features, edge_features, statistics


def _local_to_lonlat(projection: BboxProjection, geometry: Any) -> Any:
    def converter(x, y, z=None):
        try:
            points = [projection.to_latlon((float(px), float(py))) for px, py in zip(x, y)]
            lon = [point[1] for point in points]
            lat = [point[0] for point in points]
            return (lon, lat) if z is None else (lon, lat, z)
        except TypeError:
            latitude, longitude = projection.to_latlon((float(x), float(y)))
            return (longitude, latitude) if z is None else (longitude, latitude, z)
    return transform(converter, geometry)


def _lonlat_to_local(projection: BboxProjection, geometry: Any) -> Any:
    def converter(x, y, z=None):
        try:
            points = [projection.to_world((float(lat), float(lon))) for lon, lat in zip(x, y)]
            east = [point[0] for point in points]
            north = [point[1] for point in points]
            return (east, north) if z is None else (east, north, z)
        except TypeError:
            east, north = projection.to_world((float(y), float(x)))
            return (east, north) if z is None else (east, north, z)
    return transform(converter, geometry)


def _round_coordinates(value: Any, digits: int) -> Any:
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (int, float)):
            return [round(float(number), digits) for number in value]
        return [_round_coordinates(item, digits) for item in value]
    return value


def _feature(geometry: Any, properties: Mapping[str, Any], projection: BboxProjection, digits: int) -> dict[str, Any]:
    wgs = _local_to_lonlat(projection, geometry)
    geometry_document = mapping(wgs)
    geometry_document["coordinates"] = _round_coordinates(geometry_document["coordinates"], digits)
    return {"type": "Feature", "properties": dict(properties), "geometry": geometry_document}


def _write_geojson(
    path: Path,
    name: str,
    features: Sequence[dict[str, Any]],
    bundle: FrozenSourceBundle,
) -> None:
    document = {
        "type": "FeatureCollection",
        "name": name,
        "bbox": [bundle.bbox[1], bundle.bbox[0], bundle.bbox[3], bundle.bbox[2]],
        "cwr_world": {
            "coordinate_reference": "WGS84 longitude/latitude",
            "local_origin": "south-west",
            "world_size_metres": bundle.cells * bundle.cell_size,
            "grid_cells": bundle.cells,
            "cell_size_metres": bundle.cell_size,
        },
        "features": list(features),
    }
    _write_atomic(path, (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def _load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _normalization_settings(spec: NormalizationSpec) -> dict[str, Any]:
    document = asdict(spec)
    document.pop("source_dir", None)
    document.pop("output_dir", None)
    document.pop("refresh", None)
    return document


def _normalized_bundle_from_validated_manifest(
    root: Path,
    manifest: Mapping[str, Any],
) -> NormalizedBundle:
    """Return a previously validated normalized bundle without reparsing GeoJSON.

    Full geometry and checksum validation runs when the bundle is created and
    remains available through ``validate_normalized_bundle``. Normal build reuse
    only needs to confirm that the manifest matches the current schema and that
    every required layer and the validation report still exist.
    """

    if manifest.get("schema") != NORMALIZED_SCHEMA:
        raise ValueError("normalized geometry schema does not match")
    if manifest.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
        raise ValueError("normalized geometry schema version does not match")

    source_fingerprint = str(manifest.get("source_manifest_sha256", ""))
    if len(source_fingerprint) != 64:
        raise ValueError("normalized source fingerprint is missing")

    bbox_values = manifest.get("bbox_south_west_north_east")
    try:
        bbox = tuple(float(value) for value in bbox_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalized bbox is invalid") from exc
    if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError("normalized bbox is invalid")

    try:
        world_size = float(manifest.get("world_size_metres", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("normalized world size is invalid") from exc
    if not math.isfinite(world_size) or world_size <= 0:
        raise ValueError("normalized world size is invalid")

    raw_counts = manifest.get("counts")
    if not isinstance(raw_counts, Mapping):
        raise ValueError("normalized feature counts are missing")
    counts = {str(key): int(value) for key, value in raw_counts.items()}

    files = {filename: root / filename for filename in _REQUIRED_FILES}
    missing = [
        filename
        for filename, file_path in files.items()
        if not file_path.is_file() or file_path.stat().st_size <= 0
    ]
    if missing:
        raise ValueError("normalized layer files are missing or empty: " + ", ".join(missing))

    validation_path = root / "validation-report.txt"
    if not validation_path.is_file() or validation_path.stat().st_size <= 0:
        raise ValueError("normalized validation report is missing")

    manifest_path = root / "manifest.json"
    return NormalizedBundle(
        root=root,
        manifest_path=manifest_path,
        validation_path=validation_path,
        source_fingerprint=source_fingerprint,
        normalized_fingerprint=_sha256(manifest_path),
        bbox=bbox,  # type: ignore[arg-type]
        world_size=world_size,
        files=files,
        counts=counts,
    )


def normalize_source_bundle(
    spec: NormalizationSpec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
    validated_source: FrozenSourceBundle | None = None,
) -> NormalizedBundle:
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    spec.validate()
    if validated_source is None:
        progress(0, "Validating source bundle before normalization")
        source_validation = validate_source_bundle(
            spec.source_dir,
            progress_callback=lambda value, message: progress(int(value * 3 / 100), message),
        )
        source = source_validation.bundle
    else:
        source = validated_source
        if source.root.resolve() != spec.source_dir.resolve():
            raise ValueError("prevalidated source bundle does not match normalization source directory")
        progress(0, "Using already validated source bundle")
    root = (spec.output_dir or source.root / "normalized").resolve()
    manifest_path = root / "manifest.json"
    settings = _normalization_settings(spec)

    if manifest_path.is_file() and not spec.refresh:
        progress(1, "Checking existing normalized geometry manifest")
        try:
            manifest = _load_document(manifest_path)
            source_matches = manifest.get("source_manifest_sha256") == source.fingerprint
            settings_match = manifest.get("settings") == settings
            if source_matches and settings_match:
                progress(2, "Matching normalized manifest found; checking required layer files")
                existing = _normalized_bundle_from_validated_manifest(root, manifest)
                progress(100, f"Reusing previously validated normalized geometry ({len(existing.files)} layers)")
                return existing
            mismatch = []
            if not source_matches:
                mismatch.append("source fingerprint")
            if not settings_match:
                mismatch.append("normalization settings")
            progress(3, "Existing normalized geometry differs in " + " and ".join(mismatch) + "; rebuilding")
        except (OSError, ValueError) as exc:
            progress(3, f"Existing normalized geometry cannot be reused ({exc}); rebuilding")

    progress(4, f"Reading frozen OpenStreetMap source {source.osm_json_path.name}")
    try:
        raw_document = json.loads(source.osm_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen OSM JSON: {exc}") from exc
    elements = raw_document.get("elements") if isinstance(raw_document, Mapping) else None
    if not isinstance(elements, list):
        raise ValueError("frozen OSM JSON has no elements array")
    typed_elements = [element for element in elements if isinstance(element, Mapping)]
    progress(9, f"Loaded {len(typed_elements):,} source elements for normalization")

    projection = BboxProjection.create(source.bbox, source.cells * source.cell_size)
    boundary = box(0.0, 0.0, projection.world_size, projection.world_size)

    progress(13, "Normalizing and snapping road geometry")
    roads = _normalize_roads(
        typed_elements,
        projection,
        boundary,
        spec,
        progress_callback=lambda value, message: progress(13 + int(value * 8 / 100), message),
    )
    progress(21, "Noding roads and indexing junctions")
    junctions = _junction_features(
        roads,
        spec.road_snap_tolerance,
        progress_callback=lambda value, message: progress(21 + int(value / 100), message),
    )
    progress(22, f"Building one shared road-clearance corridor from {len(roads):,} lines")
    road_corridor = _road_corridor(roads, spec.road_building_setback)

    progress(23, f"Classifying {len(typed_elements):,} source elements for later geometry passes")
    barrier_features: list[tuple[LineString, dict[str, Any]]] = []
    cutline_features: list[tuple[LineString, dict[str, Any]]] = []
    tree_row_features: list[tuple[LineString, dict[str, Any]]] = []
    individual_tree_features: list[tuple[Point, dict[str, Any]]] = []
    aeroway_line_features: list[tuple[LineString, dict[str, Any]]] = []
    aeroway_area_features: list[tuple[Polygon, dict[str, Any]]] = []
    utility_point_features: list[tuple[Point, dict[str, Any]]] = []
    surface_area_features: list[tuple[Polygon, dict[str, Any]]] = []
    rural_features: list[tuple[Polygon, dict[str, Any]]] = []
    cutline_buffers: list[Any] = []
    coastline_elements: list[Mapping[str, Any]] = []
    water_polygon_elements: list[Mapping[str, Any]] = []
    landuse_elements: list[Mapping[str, Any]] = []
    forest_elements: list[Mapping[str, Any]] = []
    watercourse_elements: list[Mapping[str, Any]] = []
    place_elements: list[Mapping[str, Any]] = []
    landmark_site_elements: list[Mapping[str, Any]] = []
    element_total = max(1, len(typed_elements))
    for element_index, element in enumerate(typed_elements, start=1):
        tags = _tags(element)
        if tags.get("natural") == "coastline":
            coastline_elements.append(element)
        if (
            tags.get("natural") == "water"
            or tags.get("waterway") == "riverbank"
            or tags.get("landuse") in {"reservoir", "basin"}
            or tags.get("landcover") == "water"
        ):
            water_polygon_elements.append(element)
        if tags.get("landuse") in _FARMLAND or tags.get("landuse") in _URBAN:
            landuse_elements.append(element)
        if tags.get("landuse") == "forest" or tags.get("natural") == "wood" or tags.get("landcover") == "trees":
            forest_elements.append(element)
        if tags.get("waterway") in _WATERCOURSES:
            watercourse_elements.append(element)
        place_kind = str(tags.get("place", "")).casefold()
        if place_kind and (tags.get("name") or place_kind == "isolated_dwelling"):
            place_elements.append(element)
        if (
            tags.get("highway") == "bus_stop"
            or (tags.get("public_transport") == "platform" and tags.get("bus") in {"yes", "designated"})
            or tags.get("amenity") in {"parking", "grave_yard"}
            or tags.get("landuse") == "cemetery"
            or tags.get("leisure") == "pitch"
            or tags.get("sport") == "soccer"
        ):
            landmark_site_elements.append(element)
        if element_index % 50000 == 0:
            progress(23 + int(element_index * 8 / element_total), f"Classified {element_index:,}/{len(typed_elements):,} source elements")
        barrier = tags.get("barrier", "")
        if barrier in {"fence", "wall", "hedge", "retaining_wall"}:
            for line in _element_lines(element, projection, boundary):
                barrier_features.append((line, {
                    "barrier_id": "", "source_id": _osm_id(element),
                    "kind": "wall" if barrier == "retaining_wall" else barrier,
                    "fence_type": tags.get("fence_type", ""),
                    "material": tags.get("material", ""),
                    "height_m": round(max(0.0, _parse_number(tags.get("height"), 0.0)), 3),
                    "length_m": round(line.length, 3),
                }))
        is_cutline = tags.get("man_made") == "cutline" or tags.get("power") in {"line", "minor_line"}
        if is_cutline:
            default_width = 10.0 if tags.get("power") else 8.0
            width = max(2.0, _parse_number(tags.get("width"), default_width))
            for line in _element_lines(element, projection, boundary):
                cutline_features.append((line, {
                    "cutline_id": "", "source_id": _osm_id(element),
                    "kind": "power_line" if tags.get("power") else tags.get("cutline", "cutline"),
                    "width_m": round(width, 3), "length_m": round(line.length, 3),
                }))
                cutline_buffers.append(line.buffer(width * 0.5, cap_style=2, join_style=2))
        if tags.get("natural") == "tree_row":
            for line in _element_lines(element, projection, boundary):
                tree_row_features.append((line, {
                    "rural_id": "", "source_id": _osm_id(element), "kind": "tree_row",
                    "length_m": round(line.length, 3),
                }))

        projected_points = _geometry_points(element, projection)
        if tags.get("natural") == "tree" and projected_points:
            point = Point(projected_points[0]) if len(projected_points) == 1 else None
            if point is None:
                polygons_for_point = _element_polygons(element, projection, boundary)
                if polygons_for_point:
                    point = polygons_for_point[0].centroid
            if point is not None and boundary.covers(point):
                individual_tree_features.append((point, {
                    "tree_id": "", "source_id": _osm_id(element),
                    "species": tags.get("species", ""), "genus": tags.get("genus", ""),
                    "leaf_type": tags.get("leaf_type", ""), "leaf_cycle": tags.get("leaf_cycle", ""),
                }))

        aeroway_kind = tags.get("aeroway", "")
        geometry_is_closed = (
            element.get("type") == "relation"
            or _truthy_tag(tags, "area")
            or (len(projected_points) >= 4 and projected_points[0] == projected_points[-1])
        )
        if aeroway_kind in {"runway", "taxiway"} and not geometry_is_closed:
            for line in _element_lines(element, projection, boundary):
                aeroway_line_features.append((line, {
                    "aeroway_id": "", "source_id": _osm_id(element), "kind": aeroway_kind,
                    "surface": tags.get("surface", ""), "width_m": round(max(0.0, _parse_number(tags.get("width"), 0.0)), 3),
                    "length_m": round(line.length, 3),
                }))
        if aeroway_kind in {"aerodrome", "runway", "taxiway", "apron", "helipad"} and geometry_is_closed:
            for polygon in _element_polygons(element, projection, boundary):
                if polygon.area < 5.0:
                    continue
                aeroway_area_features.append((polygon, {
                    "aeroway_id": "", "source_id": _osm_id(element), "kind": aeroway_kind,
                    "surface": tags.get("surface", ""), "area_m2": round(polygon.area, 3),
                }))

        utility_kind = ""
        if tags.get("power") in {"pole", "tower"}:
            utility_kind = f"power_{tags.get('power')}"
        elif tags.get("man_made") == "water_tower":
            utility_kind = "water_tower"
        if utility_kind:
            point = Point(projected_points[0]) if len(projected_points) == 1 else None
            if point is None:
                polygons_for_utility = _element_polygons(element, projection, boundary)
                if polygons_for_utility:
                    point = polygons_for_utility[0].centroid
            if point is not None and boundary.covers(point):
                utility_point_features.append((point, {
                    "utility_id": "", "source_id": _osm_id(element), "kind": utility_kind,
                    "operator": tags.get("operator", ""), "height_m": round(max(0.0, _parse_number(tags.get("height"), 0.0)), 3),
                }))

        surface_kind = tags.get("natural", "") if tags.get("natural") == "grassland" else (
            "beach" if tags.get("natural") == "beach" else (
                "sand" if tags.get("natural") in {"sand", "desert", "dune"} or tags.get("landcover") == "sand" or tags.get("surface") == "sand" else (
                    "park" if tags.get("leisure") == "park" else (
                        "sports_pitch" if (tags.get("leisure") == "pitch" or tags.get("sport") == "soccer") else ""
                    )
                )
            )
        )
        if surface_kind:
            for polygon in _element_polygons(element, projection, boundary):
                if polygon.area >= 10.0:
                    surface_area_features.append((polygon, {
                        "surface_id": "", "source_id": _osm_id(element), "kind": surface_kind,
                        "natural": tags.get("natural", ""), "leisure": tags.get("leisure", ""),
                        "sport": tags.get("sport", ""), "surface": tags.get("surface", ""),
                        "area_m2": round(polygon.area, 3),
                    }))

        rural_kind = tags.get("natural", "") if tags.get("natural") in {"scrub", "bare_rock", "rock", "scree", "wetland"} else (
            tags.get("landuse", "") if tags.get("landuse") in {"orchard", "vineyard"} else ""
        )
        if rural_kind:
            for polygon in _element_polygons(element, projection, boundary):
                if polygon.area >= 20.0:
                    rural_features.append((polygon, {
                        "rural_id": "", "source_id": _osm_id(element), "kind": rural_kind,
                        "area_m2": round(polygon.area, 3),
                    }))

    barrier_features.sort(key=lambda item: (item[0].bounds, item[1]["source_id"]))
    for index, (_, properties) in enumerate(barrier_features, start=1):
        properties["barrier_id"] = f"barrier-{index:06d}"
    cutline_features.sort(key=lambda item: (item[0].bounds, item[1]["source_id"]))
    for index, (_, properties) in enumerate(cutline_features, start=1):
        properties["cutline_id"] = f"cutline-{index:06d}"
    individual_tree_features.sort(key=lambda item: (item[0].x, item[0].y, item[1]["source_id"]))
    for index, (_, properties) in enumerate(individual_tree_features, start=1):
        properties["tree_id"] = f"tree-{index:06d}"
    aeroway_line_features.sort(key=lambda item: (item[1]["kind"], item[0].bounds, item[1]["source_id"]))
    aeroway_area_features.sort(key=lambda item: (item[1]["kind"], item[0].bounds, item[1]["source_id"]))
    for index, (_, properties) in enumerate(aeroway_line_features, start=1):
        properties["aeroway_id"] = f"aeroway-line-{index:06d}"
    for index, (_, properties) in enumerate(aeroway_area_features, start=1):
        properties["aeroway_id"] = f"aeroway-area-{index:06d}"
    utility_point_features.sort(key=lambda item: (item[1]["kind"], item[0].x, item[0].y, item[1]["source_id"]))
    for index, (_, properties) in enumerate(utility_point_features, start=1):
        properties["utility_id"] = f"utility-{index:06d}"
    surface_area_features.sort(key=lambda item: (item[1]["kind"], item[0].bounds, item[1]["source_id"]))
    for index, (_, properties) in enumerate(surface_area_features, start=1):
        properties["surface_id"] = f"surface-{index:06d}"
    combined_rural = [(geometry, properties) for geometry, properties in tree_row_features] + rural_features
    combined_rural.sort(key=lambda item: (item[1]["kind"], item[0].bounds, item[1]["source_id"]))
    for index, (_, properties) in enumerate(combined_rural, start=1):
        properties["rural_id"] = f"rural-{index:06d}"
    cutline_corridor = _union_geometries(cutline_buffers) if cutline_buffers else GeometryCollection()

    progress(34, "Cleaning, merging and clipping building footprints")
    buildings, building_statistics = _normalize_buildings(
        typed_elements, projection, boundary, roads, spec, road_corridor=road_corridor,
        progress_callback=lambda value, message: progress(34 + int(value * 10 / 100), message),
    )

    progress(45, "Reconstructing coastline and inland-water polygons")
    coastline_lines: list[LineString] = []
    for element in coastline_elements:
        coastline_lines.extend(_element_lines(element, projection, boundary))
    ocean_polygons = _coastline_ocean(coastline_lines, boundary)

    water_inputs = _extract_category_polygons(
        water_polygon_elements, projection, boundary,
        lambda tags: "inland" if (
            tags.get("natural") == "water" or tags.get("waterway") == "riverbank"
            or tags.get("landuse") in {"reservoir", "basin"} or tags.get("landcover") == "water"
        ) else None,
    )
    inland_water = _repair_polygonal(_spatial_union_polygonal(item[0] for item in water_inputs), boundary) if water_inputs else []
    water_geometry = _union_geometries([*ocean_polygons, *inland_water]) if ocean_polygons or inland_water else GeometryCollection()

    progress(55, f"Extracting farmland and urban polygons from {len(landuse_elements):,} tagged elements")
    landuse_inputs = _extract_category_polygons(
        landuse_elements, projection, boundary,
        lambda tags: "farmland" if tags.get("landuse") in _FARMLAND else (
            "urban" if tags.get("landuse") in _URBAN else None
        ),
    )
    farmland_polygons: list[Polygon] = []
    meadow_polygons: list[Polygon] = []
    industrial_polygons: list[Polygon] = []
    general_urban_polygons: list[Polygon] = []
    for polygon, _source_id, tags in landuse_inputs:
        if tags["_category"] == "farmland":
            farmland_polygons.append(polygon)
            if tags.get("landuse") == "meadow":
                meadow_polygons.append(polygon)
        elif tags.get("landuse") in {"industrial", "commercial", "railway", "construction"}:
            industrial_polygons.append(polygon)
        else:
            general_urban_polygons.append(polygon)
    progress(57, f"Unioning {len(farmland_polygons):,} farmland polygons by spatial component")
    farmland_geometry = _spatial_union_polygonal(farmland_polygons)
    # Keep a semantic meadow overlay in addition to the unified farmland mask.
    # The mask continues to drive pole1/pole2 terrain, while the overlay preserves
    # landuse=meadow for object placement after normalized bundles are reloaded.
    meadow_geometry = _spatial_union_polygonal(meadow_polygons)
    progress(59, f"Unioning {len(general_urban_polygons):,} urban and {len(industrial_polygons):,} industrial polygons")
    industrial_geometry = _spatial_union_polygonal(industrial_polygons)
    general_urban_geometry = _spatial_union_polygonal(general_urban_polygons)
    progress(62, "Combining farmland and urban priority masks")
    urban_geometry = _union_geometries(geometry for geometry in (general_urban_geometry, industrial_geometry) if not geometry.is_empty)
    landuse_priority = _union_geometries(geometry for geometry in (farmland_geometry, urban_geometry) if not geometry.is_empty)

    progress(64, f"Extracting forest polygons from {len(forest_elements):,} tagged elements")
    forest_inputs = _extract_category_polygons(
        forest_elements, projection, boundary,
        lambda tags: "forest" if (
            tags.get("landuse") == "forest" or tags.get("natural") == "wood" or tags.get("landcover") == "trees"
        ) else None,
    )
    forests, forest_edges, forest_statistics = _normalize_forest(
        forest_inputs, water_geometry, landuse_priority, road_corridor, cutline_corridor,
        buildings, boundary, spec,
        progress_callback=lambda value, message: progress(64 + int(value * 9 / 100), message),
    )

    progress(73, "Normalizing rivers, streams, canals and ditches")
    watercourse_features: list[tuple[LineString, dict[str, Any]]] = []
    for element in watercourse_elements:
        tags = _tags(element)
        kind = tags.get("waterway")
        if kind not in _WATERCOURSES:
            continue
        for line in _element_lines(element, projection, boundary):
            watercourse_features.append((line, {
                "watercourse_id": "",
                "source_id": _osm_id(element),
                "kind": kind,
                "name": tags.get("name", ""),
                "tunnel": _truthy_tag(tags, "tunnel"),
                "length_m": round(line.length, 3),
            }))
    watercourse_features.sort(key=lambda item: (item[0].bounds, item[0].wkb_hex))
    for index, (_, properties) in enumerate(watercourse_features, start=1):
        properties["watercourse_id"] = f"watercourse-{index:06d}"

    progress(79, "Collecting places, landmarks and semantic sites")
    place_candidates: list[tuple[Any, dict[str, Any]]] = []
    seen_places: dict[tuple[str, str, str], tuple[Any, dict[str, Any]]] = {}
    for element in place_elements:
        tags = _tags(element)
        place_kind = tags.get("place")
        raw_name = tags.get("name", "")
        if not place_kind:
            continue
        if not raw_name and place_kind.casefold() != "isolated_dwelling":
            continue
        points = _geometry_points(element, projection)
        if len(points) == 1:
            place_geometry: Any = Point(points[0])
        else:
            polygons = _element_polygons(element, projection, boundary)
            if not polygons:
                continue
            # Keep mapped place polygons intact. Older normalization collapsed
            # them to one representative point, making exact containment rules
            # impossible later in the pipeline.
            place_geometry = unary_union(polygons)
        representative = place_geometry if isinstance(place_geometry, Point) else place_geometry.representative_point()
        if not boundary.covers(representative):
            continue
        if raw_name:
            name, ascii_name = _clean_name(raw_name)
        else:
            # Unnamed isolated-dwelling polygons are semantic boundaries, not
            # labels. Keep the name blank so they can drive cabin containment
            # without becoming fake town names on the in-game map.
            name, ascii_name = "", ""
        source_id = _osm_id(element)
        key = (
            ascii_name.casefold(),
            place_kind.casefold(),
            source_id if not raw_name else "",
        )
        properties = {
            "place_id": "",
            "source_id": _osm_id(element),
            "name": name,
            "name_ascii": ascii_name,
            "place": place_kind,
            "rank": _PLACE_RANK.get(place_kind, 99),
            "population": int(round(max(0.0, _parse_number(tags.get("population"), 0.0)))),
        }
        previous = seen_places.get(key)
        prefer_geometry = (
            previous is not None
            and properties["population"] == previous[1]["population"]
            and isinstance(previous[0], Point)
            and not isinstance(place_geometry, Point)
        )
        if previous is None or properties["population"] > previous[1]["population"] or prefer_geometry:
            seen_places[key] = (place_geometry, properties)
    place_candidates = sorted(
        seen_places.values(),
        key=lambda item: (
            item[1]["rank"], item[1]["name_ascii"].casefold(),
            (item[0] if isinstance(item[0], Point) else item[0].representative_point()).x,
            (item[0] if isinstance(item[0], Point) else item[0].representative_point()).y,
        ),
    )
    for index, (_, properties) in enumerate(place_candidates, start=1):
        properties["place_id"] = f"place-{index:06d}"

    landmark_candidates: list[tuple[Point, dict[str, Any]]] = []
    site_candidates: list[tuple[Polygon, dict[str, Any]]] = []
    for element in landmark_site_elements:
        tags = _tags(element)
        is_bus_stop = tags.get("highway") == "bus_stop" or (
            tags.get("public_transport") == "platform" and tags.get("bus") in {"yes", "designated"}
        )
        if is_bus_stop:
            points = _geometry_points(element, projection)
            if len(points) == 1:
                point = Point(points[0])
            else:
                polygons = _element_polygons(element, projection, boundary)
                point = unary_union(polygons).representative_point() if polygons else None
            if point is not None and boundary.covers(point):
                landmark_candidates.append((point, {
                    "landmark_id": "",
                    "source_id": _osm_id(element),
                    "kind": "bus_stop",
                    "name": tags.get("name", ""),
                    "shelter": tags.get("shelter", ""),
                    "direction": tags.get("direction", ""),
                }))

        site_kind = "parking" if tags.get("amenity") == "parking" else (
            "sports_pitch" if (tags.get("leisure") == "pitch" or tags.get("sport") == "soccer") else (
                "cemetery" if (tags.get("landuse") == "cemetery" or tags.get("amenity") == "grave_yard") else ""
            )
        )
        if site_kind:
            for polygon in _element_polygons(element, projection, boundary):
                if polygon.area < 20.0:
                    continue
                properties = {
                    "site_id": "",
                    "source_id": _osm_id(element),
                    "kind": site_kind,
                    "sport": tags.get("sport", ""),
                    "surface": tags.get("surface", ""),
                    "name": tags.get("name", ""),
                    "area_m2": round(polygon.area, 3),
                    **_oriented_box_properties(polygon),
                }
                site_candidates.append((polygon, properties))

    landmark_candidates.sort(key=lambda item: (item[0].x, item[0].y, item[1]["source_id"]))
    for index, (_, properties) in enumerate(landmark_candidates, start=1):
        properties["landmark_id"] = f"landmark-{index:06d}"
    site_candidates.sort(key=lambda item: (item[0].centroid.x, item[0].centroid.y, item[1]["source_id"]))
    for index, (_, properties) in enumerate(site_candidates, start=1):
        properties["site_id"] = f"site-{index:06d}"

    progress(86, "Writing normalized GeoJSON layers")
    temporary_root = Path(tempfile.mkdtemp(prefix="cwr-normalized-", dir=str(root.parent)))
    try:
        gravel_roads = [road for road in roads if road_is_gravel(road.properties)]
        standard_roads = [road for road in roads if not road_is_gravel(road.properties)]
        roads_features = [_feature(road.geometry, road.properties, projection, spec.coordinate_precision) for road in standard_roads]
        gravel_road_features = [_feature(road.geometry, road.properties, projection, spec.coordinate_precision) for road in gravel_roads]
        junction_features = [_feature(point, properties, projection, spec.coordinate_precision) for point, properties in junctions]
        building_features = [_feature(item.geometry, item.properties, projection, spec.coordinate_precision) for item in buildings]
        forest_features = [_feature(item.geometry, item.properties, projection, spec.coordinate_precision) for item in forests]
        edge_features = [_feature(item.geometry, item.properties, projection, spec.coordinate_precision) for item in forest_edges]

        water_features: list[dict[str, Any]] = []
        water_local_features: list[tuple[Any, Mapping[str, Any]]] = []
        water_index = 1
        for kind, polygons in (("ocean", ocean_polygons), ("inland", inland_water)):
            for polygon in sorted(polygons, key=lambda geometry: (geometry.centroid.x, geometry.centroid.y, geometry.wkb_hex)):
                properties = {
                    "water_id": f"water-{water_index:06d}",
                    "kind": kind,
                    "area_m2": round(polygon.area, 3),
                }
                water_local_features.append((polygon, properties))
                water_features.append(_feature(polygon, properties, projection, spec.coordinate_precision))
                water_index += 1

        landuse_features: list[dict[str, Any]] = []
        landuse_local_features: list[tuple[Any, Mapping[str, Any]]] = []
        landuse_index = 1
        for category, geometry in (
            ("farmland", farmland_geometry),
            ("meadow", meadow_geometry),
            ("urban", general_urban_geometry),
            ("industrial", industrial_geometry),
        ):
            for polygon in _repair_polygonal(geometry, boundary):
                properties = {
                    "landuse_id": f"landuse-{landuse_index:06d}",
                    "category": category,
                    "area_m2": round(polygon.area, 3),
                }
                landuse_local_features.append((polygon, properties))
                landuse_features.append(_feature(polygon, properties, projection, spec.coordinate_precision))
                landuse_index += 1

        watercourse_documents = [_feature(line, properties, projection, spec.coordinate_precision) for line, properties in watercourse_features]
        place_documents = [_feature(point, properties, projection, spec.coordinate_precision) for point, properties in place_candidates]
        landmark_documents = [_feature(point, properties, projection, spec.coordinate_precision) for point, properties in landmark_candidates]
        site_documents = [_feature(polygon, properties, projection, spec.coordinate_precision) for polygon, properties in site_candidates]
        barrier_documents = [_feature(line, properties, projection, spec.coordinate_precision) for line, properties in barrier_features]
        cutline_documents = [_feature(line, properties, projection, spec.coordinate_precision) for line, properties in cutline_features]
        tree_documents = [_feature(point, properties, projection, spec.coordinate_precision) for point, properties in individual_tree_features]
        aeroway_line_documents = [_feature(line, properties, projection, spec.coordinate_precision) for line, properties in aeroway_line_features]
        aeroway_area_documents = [_feature(polygon, properties, projection, spec.coordinate_precision) for polygon, properties in aeroway_area_features]
        utility_point_documents = [_feature(point, properties, projection, spec.coordinate_precision) for point, properties in utility_point_features]
        surface_area_documents = [_feature(polygon, properties, projection, spec.coordinate_precision) for polygon, properties in surface_area_features]
        rural_documents = [_feature(geometry, properties, projection, spec.coordinate_precision) for geometry, properties in combined_rural]

        local_collections: dict[str, list[tuple[Any, Mapping[str, Any]]]] = {
            "roads.geojson": [(road.geometry, road.properties) for road in standard_roads],
            "gravel-roads.geojson": [(road.geometry, road.properties) for road in gravel_roads],
            "road-junctions.geojson": list(junctions),
            "buildings.geojson": [(item.geometry, item.properties) for item in buildings],
            "forests.geojson": [(item.geometry, item.properties) for item in forests],
            "forest-edges.geojson": [(item.geometry, item.properties) for item in forest_edges],
            "water.geojson": water_local_features,
            "watercourses.geojson": list(watercourse_features),
            "landuse.geojson": landuse_local_features,
            "places.geojson": list(place_candidates),
            "landmarks.geojson": list(landmark_candidates),
            "sites.geojson": list(site_candidates),
            "barriers.geojson": list(barrier_features),
            "cutlines.geojson": list(cutline_features),
            "trees.geojson": list(individual_tree_features),
            "aeroway-lines.geojson": list(aeroway_line_features),
            "aeroway-areas.geojson": list(aeroway_area_features),
            "utility-points.geojson": list(utility_point_features),
            "surface-areas.geojson": list(surface_area_features),
            "rural-vegetation.geojson": list(combined_rural),
        }

        collections = {
            "roads.geojson": ("roads", roads_features),
            "gravel-roads.geojson": ("gravel-roads", gravel_road_features),
            "road-junctions.geojson": ("road-junctions", junction_features),
            "buildings.geojson": ("buildings", building_features),
            "forests.geojson": ("forests", forest_features),
            "forest-edges.geojson": ("forest-edges", edge_features),
            "water.geojson": ("water", water_features),
            "watercourses.geojson": ("watercourses", watercourse_documents),
            "landuse.geojson": ("landuse", landuse_features),
            "places.geojson": ("places", place_documents),
            "landmarks.geojson": ("landmarks", landmark_documents),
            "sites.geojson": ("sites", site_documents),
            "barriers.geojson": ("barriers", barrier_documents),
            "cutlines.geojson": ("cutlines", cutline_documents),
            "trees.geojson": ("trees", tree_documents),
            "aeroway-lines.geojson": ("aeroway-lines", aeroway_line_documents),
            "aeroway-areas.geojson": ("aeroway-areas", aeroway_area_documents),
            "utility-points.geojson": ("utility-points", utility_point_documents),
            "surface-areas.geojson": ("surface-areas", surface_area_documents),
            "rural-vegetation.geojson": ("rural-vegetation", rural_documents),
        }
        for filename, (name, features) in collections.items():
            _write_geojson(temporary_root / filename, name, features, source)

        counts = {filename.removesuffix(".geojson"): len(features) for filename, (_, features) in collections.items()}
        hashes = {filename: _sha256(temporary_root / filename) for filename in _REQUIRED_FILES}
        manifest = {
            "schema": NORMALIZED_SCHEMA,
            "schema_version": NORMALIZED_SCHEMA_VERSION,
            "generator": GENERATOR_VERSION,
            "source_manifest_sha256": source.fingerprint,
            "bbox_south_west_north_east": list(source.bbox),
            "world_size_metres": projection.world_size,
            "settings": settings,
            "counts": counts,
            "statistics": {
                "buildings": building_statistics,
                "forest": forest_statistics,
                "coastline_segments": len(coastline_lines),
            },
            "files": hashes,
        }
        _write_atomic(temporary_root / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        if root.exists():
            shutil.rmtree(root)
        os.replace(temporary_root, root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    progress(95, "Validating normalized geometry and checksums from in-memory results")
    result = validate_normalized_bundle(
        root,
        _preloaded_manifest=manifest,
        _precomputed_hashes=hashes,
        _precomputed_local_features=local_collections,
        progress_callback=lambda value, message: progress(95 + int(value * 5 / 100), message),
    )
    progress(100, "Normalized geometry bundle ready")
    return result


def validate_normalized_bundle(
    root: Path,
    *,
    write_report: bool = True,
    progress_callback: Callable[[int, str], None] | None = None,
    _preloaded_manifest: Mapping[str, Any] | None = None,
    _precomputed_hashes: Mapping[str, str] | None = None,
    _precomputed_local_features: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]] | None = None,
) -> NormalizedBundle:
    def report(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), message)

    root = root.resolve()
    manifest_path = root / "manifest.json"
    report(2, "Validating normalized bundle metadata")
    document = dict(_preloaded_manifest) if _preloaded_manifest is not None else _load_document(manifest_path)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Normalized schema", document.get("schema") == NORMALIZED_SCHEMA, str(document.get("schema"))))
    checks.append(("Normalized schema version", document.get("schema_version") == NORMALIZED_SCHEMA_VERSION, str(document.get("schema_version"))))
    source_fingerprint = str(document.get("source_manifest_sha256", ""))
    checks.append(("Source fingerprint present", len(source_fingerprint) == 64, source_fingerprint))
    bbox_values = document.get("bbox_south_west_north_east")
    try:
        bbox = tuple(float(value) for value in bbox_values)
        bbox_ok = len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]
    except (TypeError, ValueError):
        bbox = (0.0, 0.0, 1.0, 1.0)
        bbox_ok = False
    checks.append(("Normalized bbox", bbox_ok, str(bbox)))
    world_size = float(document.get("world_size_metres", 0.0))
    checks.append(("World size", world_size > 0 and math.isfinite(world_size), f"{world_size:g}m"))

    file_hashes = document.get("files")
    counts = document.get("counts")
    files: dict[str, Path] = {}
    feature_cache: dict[str, list[tuple[Any, Mapping[str, Any]]]] = {}
    using_precomputed = _precomputed_local_features is not None
    report(8, "Verifying normalized layer files and feature counts")
    for file_index, filename in enumerate(_REQUIRED_FILES, start=1):
        path = root / filename
        files[filename] = path
        expected = file_hashes.get(filename) if isinstance(file_hashes, Mapping) else None
        if _precomputed_hashes is not None and filename in _precomputed_hashes:
            actual = str(_precomputed_hashes[filename]) if path.is_file() else "missing"
        else:
            actual = _sha256(path) if path.is_file() else "missing"
        checks.append((f"Checksum {filename}", expected == actual, actual))
        geometry_count = 0
        valid = False
        detail = "missing"
        parsed: list[tuple[Any, Mapping[str, Any]]] = []
        if using_precomputed:
            parsed = list(_precomputed_local_features.get(filename, ()))  # type: ignore[union-attr]
            geometry_count = len(parsed)
            valid = path.is_file() and all(
                geometry is not None and not geometry.is_empty and geometry.is_valid
                and isinstance(properties, Mapping)
                for geometry, properties in parsed
            )
            detail = f"{geometry_count} in-memory valid features"
        elif path.is_file():
            try:
                geojson = _load_document(path)
                features = geojson.get("features")
                if isinstance(features, list):
                    valid = True
                    geometry_count = len(features)
                    for feature in features:
                        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
                            valid = False
                            break
                        geometry_document = feature.get("geometry")
                        properties = feature.get("properties")
                        if geometry_document is None or not isinstance(properties, Mapping):
                            valid = False
                            break
                        geometry = shape(geometry_document)
                        if geometry.is_empty or not geometry.is_valid:
                            valid = False
                            break
                        parsed.append((geometry, properties))
                    detail = f"{geometry_count} valid features"
            except (ValueError, TypeError) as exc:
                detail = str(exc)
        feature_cache[filename] = parsed
        expected_count = counts.get(filename.removesuffix(".geojson")) if isinstance(counts, Mapping) else None
        checks.append((f"GeoJSON {filename}", valid and expected_count == geometry_count, detail))
        report(8 + int(file_index * 32 / len(_REQUIRED_FILES)), f"Verified normalized layer {file_index}/{len(_REQUIRED_FILES)}: {filename}")

    if bbox_ok and world_size > 0 and math.isfinite(world_size):
        projection = BboxProjection.create(bbox, world_size)  # type: ignore[arg-type]
        world_boundary = box(0.0, 0.0, world_size, world_size)
        report(42, "Preparing local geometry for topology validation")
        if using_precomputed:
            local_cache = feature_cache
        else:
            local_cache: dict[str, list[tuple[Any, Mapping[str, Any]]]] = {
                filename: [(_lonlat_to_local(projection, geometry), properties) for geometry, properties in features]
                for filename, features in feature_cache.items()
            }
        validation_boundary = world_boundary.buffer(0.25)
        outside_count = sum(
            1
            for features in local_cache.values()
            for geometry, _ in features
            if not validation_boundary.covers(geometry)
        )
        checks.append(("All normalized geometry clipped to world boundary", outside_count == 0, f"outside={outside_count}"))

        report(52, "Validating roads and building topology")
        road_features = [
            *local_cache.get("roads.geojson", []),
            *local_cache.get("gravel-roads.geojson", []),
        ]
        settings = document.get("settings") if isinstance(document.get("settings"), Mapping) else {}
        setback = float(settings.get("road_building_setback", 1.5))
        active_road_geometries: list[Any] = []
        active_road_distances: list[float] = []
        bad_special = 0
        for geometry, properties in road_features:
            special = str(properties.get("special", "normal"))
            if special not in {"normal", "bridge", "tunnel", "embankment"}:
                bad_special += 1
            if bool(properties.get("tunnel", False)):
                continue
            active_road_geometries.append(geometry)
            active_road_distances.append(float(properties.get("width_m", 2.5)) / 2.0 + setback)
        checks.append(("Road special classifications", bad_special == 0, f"invalid={bad_special}"))
        report(54, f"Buffering {len(active_road_geometries):,} active roads for topology checks")
        road_buffers = list(vector_buffer(
            active_road_geometries,
            active_road_distances,
            cap_style="flat",
            join_style="mitre",
        )) if active_road_geometries else []

        building_features = local_cache.get("buildings.geojson", [])
        building_geometries = [geometry for geometry, _ in building_features]
        report(56, f"Spatially checking {len(building_geometries):,} building footprints for collisions")
        building_overlap = _pairwise_polygon_overlap_area(
            building_geometries,
            progress_callback=lambda percent, message: report(56 + int(percent * 4 / 100), message),
            label="building",
        )
        checks.append(("Building collisions removed", building_overlap <= 0.05, f"overlap={building_overlap:.4f}m2"))

        report(61, f"Checking {len(building_geometries):,} buildings against nearby road buffers")
        building_road_overlap, maximum_building_road_overlap, building_road_pairs = _local_overlap_details(
            building_geometries,
            road_buffers,
            progress_callback=lambda percent, message: report(61 + int(percent * 11 / 100), message),
            label="buildings against local road corridors",
        )
        building_road_overlaps = [
            (str(building_features[index][1].get("building_id", "unknown")), area)
            for index, area in building_road_pairs
            if area > 1e-6
        ]
        offenders = ", ".join(
            f"{building_id}:{area:.4f}"
            for building_id, area in sorted(building_road_overlaps, key=lambda item: (-item[1], item[0]))[:5]
        ) or "none"
        checks.append((
            "Buildings clipped from road corridors",
            building_road_overlap <= 0.05 and maximum_building_road_overlap <= 0.01,
            (
                f"total={building_road_overlap:.4f}m2, "
                f"max={maximum_building_road_overlap:.4f}m2, "
                f"offenders={len(building_road_overlaps)} [{offenders}]"
            ),
        ))

        report(76, "Validating forest exclusions and edge crowns")
        semantic_area_tolerance = max(0.1, world_size * world_size * 1e-6)
        forest_geometries = [geometry for geometry, _ in local_cache.get("forests.geojson", [])]
        edge_geometries = [geometry for geometry, _ in local_cache.get("forest-edges.geojson", [])]
        water_geometries = [geometry for geometry, _ in local_cache.get("water.geojson", [])]

        report(77, f"Checking {len(forest_geometries):,} forests against nearby water polygons")
        forest_water_overlap, _maximum_water_overlap, _water_offenders = _local_overlap_details(
            forest_geometries,
            water_geometries,
            progress_callback=lambda percent, message: report(77 + int(percent * 4 / 100), message),
            label="forest polygons against water",
        )
        report(81, f"Checking {len(forest_geometries):,} forests against nearby road buffers")
        forest_road_overlap, _maximum_road_overlap, _road_offenders = _local_overlap_details(
            forest_geometries,
            road_buffers,
            progress_callback=lambda percent, message: report(81 + int(percent * 4 / 100), message),
            label="forest polygons against roads",
        )
        report(85, f"Checking {len(forest_geometries):,} forests against nearby buildings")
        forest_building_overlap, _maximum_building_overlap, _building_offenders = _local_overlap_details(
            forest_geometries,
            building_geometries,
            progress_callback=lambda percent, message: report(85 + int(percent * 4 / 100), message),
            label="forest polygons against buildings",
        )
        report(89, f"Checking {len(edge_geometries):,} edge crowns against nearby forests")
        edge_outside = _forest_edge_outside_area(
            edge_geometries,
            forest_geometries,
            tolerance=0.05,
            progress_callback=lambda percent, message: report(89 + int(percent * 4 / 100), message),
        )
        checks.append(("Forest clearings exclude water", forest_water_overlap <= semantic_area_tolerance, f"overlap={forest_water_overlap:.4f}m2, tolerance={semantic_area_tolerance:.4f}m2"))
        checks.append(("Forest clearings exclude roads", forest_road_overlap <= semantic_area_tolerance, f"overlap={forest_road_overlap:.4f}m2, tolerance={semantic_area_tolerance:.4f}m2"))
        checks.append(("Forest clearings exclude buildings", forest_building_overlap <= semantic_area_tolerance, f"overlap={forest_building_overlap:.4f}m2, tolerance={semantic_area_tolerance:.4f}m2"))
        checks.append(("Forest edge crowns stay inside forests", edge_outside <= semantic_area_tolerance, f"outside={edge_outside:.4f}m2, tolerance={semantic_area_tolerance:.4f}m2"))

        place_failures = 0
        for _, properties in feature_cache.get("places.geojson", []):
            ascii_name = str(properties.get("name_ascii", ""))
            # Unnamed isolated-dwelling polygons are semantic containment areas,
            # not map labels. They intentionally have no printable name.
            if (
                not ascii_name.strip()
                and str(properties.get("place", "")).casefold() == "isolated_dwelling"
            ):
                continue
            try:
                encoded = ascii_name.encode("ascii")
            except UnicodeEncodeError:
                place_failures += 1
                continue
            if not encoded.strip():
                place_failures += 1
        checks.append(("Place names have non-empty ASCII fallbacks", place_failures == 0, f"invalid={place_failures}"))

    report(94, "Writing normalized geometry validation report")
    failures = [label for label, passed, _ in checks if not passed]
    validation_path = root / "validation-report.txt"
    lines = ["CWR World Generator - Milestone 6 geometry validation", ""]
    lines.extend(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}" for label, passed, detail in checks)
    lines.extend(["", f"Failures: {len(failures)}"])
    if write_report:
        _write_atomic(validation_path, ("\n".join(lines) + "\n").encode("utf-8"))
    if failures:
        raise ValueError("normalized geometry validation failed: " + "; ".join(failures))
    normalized_fingerprint = _sha256(manifest_path)
    report(100, "Normalized geometry validation complete")
    return NormalizedBundle(
        root=root,
        manifest_path=manifest_path,
        validation_path=validation_path,
        source_fingerprint=source_fingerprint,
        normalized_fingerprint=normalized_fingerprint,
        bbox=bbox,  # type: ignore[arg-type]
        world_size=world_size,
        files=files,
        counts={str(key): int(value) for key, value in counts.items()} if isinstance(counts, Mapping) else {},
    )


def _geojson_features(path: Path) -> list[Mapping[str, Any]]:
    document = _load_document(path)
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError(f"{path} does not contain a features array")
    return [feature for feature in features if isinstance(feature, Mapping)]


def _point_ll(coordinate: Sequence[float]) -> tuple[float, float]:
    return float(coordinate[1]), float(coordinate[0])


def _line_ll(geometry: Any) -> tuple[tuple[float, float], ...]:
    return tuple(_point_ll(coordinate) for coordinate in geometry.coords)


def _polygon_ll(geometry: Polygon) -> GeoPolygon:
    outer = tuple(_point_ll(coordinate) for coordinate in geometry.exterior.coords)
    holes = tuple(tuple(_point_ll(coordinate) for coordinate in ring.coords) for ring in geometry.interiors)
    return GeoPolygon(outer=outer, holes=holes)


def _properties(feature: Mapping[str, Any]) -> dict[str, str]:
    raw = feature.get("properties")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            result[str(key)] = "yes" if value else "no"
        elif isinstance(value, (str, int, float)):
            result[str(key)] = str(value)
    return result


def _parse_normalized_dataset(bundle: NormalizedBundle) -> OsmDataset:
    roads: list[OsmLineFeature] = []
    gravel_roads: list[OsmLineFeature] = []

    def read_road_layer(filename: str, *, gravel: bool) -> None:
        for feature in _geojson_features(bundle.files[filename]):
            geometry = shape(feature["geometry"])
            properties = _properties(feature)
            tags = {
                "highway": properties.get("highway", "road"),
                "surface": properties.get("surface", ""),
                "width": properties.get("width_m", ""),
                "special": properties.get("special", "normal"),
            }
            def normalized_truthy(value: Any) -> bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return value != 0
                return str(value or "").strip().casefold() not in {"", "no", "false", "0", "none"}

            # Normalization writes these fields as JSON booleans. Older loaders
            # compared them to the literal string "yes", silently dropping bridge
            # tags from every frozen source bundle. Preserve both boolean fields
            # and the historical `special` discriminator for backwards bundles.
            special = str(properties.get("special", "")).strip().casefold()
            if normalized_truthy(properties.get("bridge")) or special == "bridge":
                tags["bridge"] = "yes"
            if normalized_truthy(properties.get("tunnel")) or special == "tunnel":
                tags["tunnel"] = "yes"
            if normalized_truthy(properties.get("embankment")) or special == "embankment":
                tags["embankment"] = "yes"
            if properties.get("layer"):
                tags["layer"] = properties["layer"]
            for index, line in enumerate(_iter_lines(geometry)):
                item = OsmLineFeature(
                    f"normalized/{properties.get('road_id', len(roads) + 1)}-{index}", tags, _line_ll(line)
                )
                roads.append(item)
                if gravel:
                    gravel_roads.append(item)

    read_road_layer("roads.geojson", gravel=False)
    read_road_layer("gravel-roads.geojson", gravel=True)

    def polygon_features(filename: str, tags_for) -> list[OsmPolygonFeature]:
        result: list[OsmPolygonFeature] = []
        for feature in _geojson_features(bundle.files[filename]):
            geometry = shape(feature["geometry"])
            properties = _properties(feature)
            polygons = tuple(_polygon_ll(polygon) for polygon in _iter_polygons(geometry))
            if polygons:
                result.append(OsmPolygonFeature(
                    f"normalized/{properties.get(next((key for key in properties if key.endswith('_id')), 'id'), len(result) + 1)}",
                    tags_for(properties), polygons,
                ))
        return result

    def building_tags(properties: Mapping[str, Any]) -> dict[str, str]:
        tags = {
            "building": str(properties.get("building_kind", "yes")),
            "name": str(properties.get("name", "")),
        }
        for key in _BUILDING_METADATA_TAGS:
            value = properties.get(key)
            if value not in {None, ""}:
                tags[key] = str(value)
        return tags

    buildings = polygon_features("buildings.geojson", building_tags)
    forests = polygon_features("forests.geojson", lambda properties: {"landuse": "forest"})
    water = polygon_features("water.geojson", lambda properties: {"natural": "water", "water": properties.get("kind", "")})
    landuse_all = _geojson_features(bundle.files["landuse.geojson"])
    farmland: list[OsmPolygonFeature] = []
    urban: list[OsmPolygonFeature] = []
    for feature in landuse_all:
        geometry = shape(feature["geometry"])
        properties = _properties(feature)
        polygons = tuple(_polygon_ll(polygon) for polygon in _iter_polygons(geometry))
        category = properties.get("category", "urban")
        target = farmland if category in _FARMLAND else urban
        if polygons:
            target.append(OsmPolygonFeature(
                f"normalized/{properties.get('landuse_id', len(target) + 1)}",
                {"landuse": category, "category": category}, polygons,
            ))

    watercourses: list[OsmLineFeature] = []
    for feature in _geojson_features(bundle.files["watercourses.geojson"]):
        geometry = shape(feature["geometry"])
        properties = _properties(feature)
        tags = {
            "waterway": properties.get("kind", "stream"),
            "name": properties.get("name", ""),
        }
        if properties.get("tunnel") == "yes":
            tags["tunnel"] = "yes"
        for index, line in enumerate(_iter_lines(geometry)):
            watercourses.append(OsmLineFeature(
                f"normalized/{properties.get('watercourse_id', len(watercourses) + 1)}-{index}",
                tags,
                _line_ll(line),
            ))

    places: list[OsmPointFeature] = []
    place_areas: list[OsmPolygonFeature] = []
    for feature in _geojson_features(bundle.files["places.geojson"]):
        geometry = shape(feature["geometry"])
        properties = _properties(feature)
        tags = {
            "place": properties.get("place", "locality"),
            "name": properties.get("name_ascii") or properties.get("name") or "",
            "population": properties.get("population", "0"),
        }
        osm_key = f"normalized/{properties.get('place_id', len(places) + len(place_areas) + 1)}"
        if isinstance(geometry, Point):
            places.append(OsmPointFeature(osm_key, tags, (geometry.y, geometry.x)))
        else:
            polygons = tuple(_polygon_ll(polygon) for polygon in _iter_polygons(geometry))
            if polygons:
                place_areas.append(OsmPolygonFeature(osm_key, tags, polygons))
                point = geometry.representative_point()
                places.append(OsmPointFeature(osm_key, tags, (point.y, point.x)))

    landmarks: list[OsmPointFeature] = []
    for feature in _geojson_features(bundle.files["landmarks.geojson"]):
        geometry = shape(feature["geometry"])
        properties = _properties(feature)
        if isinstance(geometry, Point):
            landmarks.append(OsmPointFeature(
                f"normalized/{properties.get('landmark_id', len(landmarks) + 1)}",
                {
                    "landmark": properties.get("kind", ""),
                    "name": properties.get("name", ""),
                    "shelter": properties.get("shelter", ""),
                    "direction": properties.get("direction", ""),
                },
                (geometry.y, geometry.x),
            ))

    sites: list[OsmPolygonFeature] = []
    for feature in _geojson_features(bundle.files["sites.geojson"]):
        geometry = shape(feature["geometry"])
        properties = _properties(feature)
        polygons = tuple(_polygon_ll(polygon) for polygon in _iter_polygons(geometry))
        if polygons:
            sites.append(OsmPolygonFeature(
                f"normalized/{properties.get('site_id', len(sites) + 1)}",
                {
                    "site": properties.get("kind", ""),
                    "sport": properties.get("sport", ""),
                    "surface": properties.get("surface", ""),
                    "name": properties.get("name", ""),
                },
                polygons,
            ))

    barriers: list[OsmLineFeature] = []
    for feature in _geojson_features(bundle.files["barriers.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature)
        tags = {
            "barrier": properties.get("kind", "fence"),
            "height": properties.get("height_m", ""),
            "fence_type": properties.get("fence_type", ""),
            "material": properties.get("material", ""),
        }
        for index, line in enumerate(_iter_lines(geometry)):
            barriers.append(OsmLineFeature(f"normalized/{properties.get('barrier_id', len(barriers)+1)}-{index}", tags, _line_ll(line)))

    cutlines: list[OsmLineFeature] = []
    for feature in _geojson_features(bundle.files["cutlines.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature)
        tags = {"man_made": "cutline", "cutline": properties.get("kind", "cutline"), "width": properties.get("width_m", "8")}
        for index, line in enumerate(_iter_lines(geometry)):
            cutlines.append(OsmLineFeature(f"normalized/{properties.get('cutline_id', len(cutlines)+1)}-{index}", tags, _line_ll(line)))

    individual_trees: list[OsmPointFeature] = []
    for feature in _geojson_features(bundle.files["trees.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature)
        if isinstance(geometry, Point):
            individual_trees.append(OsmPointFeature(
                f"normalized/{properties.get('tree_id', len(individual_trees)+1)}",
                {"natural": "tree", "species": properties.get("species", ""), "genus": properties.get("genus", ""),
                 "leaf_type": properties.get("leaf_type", ""), "leaf_cycle": properties.get("leaf_cycle", "")},
                (geometry.y, geometry.x),
            ))

    aeroway_lines: list[OsmLineFeature] = []
    for feature in _geojson_features(bundle.files["aeroway-lines.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature)
        tags = {"aeroway": properties.get("kind", "runway"), "surface": properties.get("surface", ""), "width": properties.get("width_m", "")}
        for index, line in enumerate(_iter_lines(geometry)):
            aeroway_lines.append(OsmLineFeature(f"normalized/{properties.get('aeroway_id', len(aeroway_lines)+1)}-{index}", tags, _line_ll(line)))

    aeroway_areas: list[OsmPolygonFeature] = []
    for feature in _geojson_features(bundle.files["aeroway-areas.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature)
        polygons = tuple(_polygon_ll(polygon) for polygon in _iter_polygons(geometry))
        if polygons:
            aeroway_areas.append(OsmPolygonFeature(
                f"normalized/{properties.get('aeroway_id', len(aeroway_areas)+1)}",
                {"aeroway": properties.get("kind", ""), "surface": properties.get("surface", "")}, polygons,
            ))

    utility_points: list[OsmPointFeature] = []
    for feature in _geojson_features(bundle.files["utility-points.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature)
        if isinstance(geometry, Point):
            kind = str(properties.get("kind", ""))
            tags = {"utility": kind, "height": properties.get("height_m", "")}
            if kind.startswith("power_"):
                tags["power"] = kind.removeprefix("power_")
            elif kind == "water_tower":
                tags["man_made"] = "water_tower"
            utility_points.append(OsmPointFeature(f"normalized/{properties.get('utility_id', len(utility_points)+1)}", tags, (geometry.y, geometry.x)))

    surface_areas: list[OsmPolygonFeature] = []
    for feature in _geojson_features(bundle.files["surface-areas.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature); kind = str(properties.get("kind", ""))
        polygons = tuple(_polygon_ll(polygon) for polygon in _iter_polygons(geometry))
        if polygons:
            tags = {"surface_kind": kind}
            natural = str(properties.get("natural", ""))
            leisure = str(properties.get("leisure", ""))
            sport = str(properties.get("sport", ""))
            surface = str(properties.get("surface", ""))
            if natural: tags["natural"] = natural
            if leisure: tags["leisure"] = leisure
            if sport: tags["sport"] = sport
            if surface: tags["surface"] = surface
            if kind == "park" and "leisure" not in tags: tags["leisure"] = "park"
            elif kind == "sports_pitch" and "leisure" not in tags: tags["leisure"] = "pitch"
            elif kind == "beach" and "natural" not in tags: tags["natural"] = "beach"
            elif kind not in {"park", "sports_pitch"} and "natural" not in tags: tags["natural"] = kind
            surface_areas.append(OsmPolygonFeature(f"normalized/{properties.get('surface_id', len(surface_areas)+1)}", tags, polygons))

    tree_rows: list[OsmLineFeature] = []
    rural_vegetation: list[OsmPolygonFeature] = []
    for feature in _geojson_features(bundle.files["rural-vegetation.geojson"]):
        geometry = shape(feature["geometry"]); properties = _properties(feature); kind = properties.get("kind", "scrub")
        if kind == "tree_row":
            for index, line in enumerate(_iter_lines(geometry)):
                tree_rows.append(OsmLineFeature(f"normalized/{properties.get('rural_id', len(tree_rows)+1)}-{index}", {"natural": "tree_row"}, _line_ll(line)))
        else:
            polygons = tuple(_polygon_ll(polygon) for polygon in _iter_polygons(geometry))
            if polygons:
                tags = {"rural_kind": kind}
                tags["landuse" if kind in {"orchard", "vineyard"} else "natural"] = kind
                rural_vegetation.append(OsmPolygonFeature(f"normalized/{properties.get('rural_id', len(rural_vegetation)+1)}", tags, polygons))

    return OsmDataset(
        source_generator="cwr-worldgen-normalized/1",
        element_count=sum(bundle.counts.values()),
        coastlines=(),
        water=tuple(water),
        forests=tuple(forests),
        farmland=tuple(farmland),
        urban=tuple(urban),
        roads=tuple(roads),
        gravel_roads=tuple(gravel_roads),
        watercourses=tuple(watercourses),
        building_polygons=tuple(buildings),
        building_points=(),
        places=tuple(places),
        place_areas=tuple(place_areas),
        landmarks=tuple(landmarks),
        sites=tuple(sites),
        barriers=tuple(barriers),
        cutlines=tuple(cutlines),
        tree_rows=tuple(tree_rows),
        individual_trees=tuple(individual_trees),
        aeroway_lines=tuple(aeroway_lines),
        aeroway_areas=tuple(aeroway_areas),
        utility_points=tuple(utility_points),
        surface_areas=tuple(surface_areas),
        rural_vegetation=tuple(rural_vegetation),
        normalized_fingerprint=bundle.normalized_fingerprint,
        parsed_cache_hit=False,
    )


_PARSED_DATASET_CACHE_SCHEMA = 10


def load_normalized_dataset(
    bundle_or_path: NormalizedBundle | Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> OsmDataset:
    bundle = bundle_or_path if isinstance(bundle_or_path, NormalizedBundle) else validate_normalized_bundle(bundle_or_path)
    key = cache_key(
        "normalized-dataset-v12-park-pitch-beach-textures",
        {
            "schema": _PARSED_DATASET_CACHE_SCHEMA,
            "normalized_fingerprint": bundle.normalized_fingerprint,
        },
    )
    cache_path = cache_dir / "sources" / f"{key}.pickle" if cache_dir is not None else None
    if use_cache and not refresh and cache_path is not None and cache_path.is_file():
        try:
            payload = pickle.loads(cache_path.read_bytes())
            if (
                isinstance(payload, dict)
                and payload.get("schema") == CACHE_SCHEMA_VERSION
                and payload.get("dataset_schema") == _PARSED_DATASET_CACHE_SCHEMA
                and payload.get("normalized_fingerprint") == bundle.normalized_fingerprint
                and isinstance(payload.get("dataset"), OsmDataset)
            ):
                return replace(payload["dataset"], parsed_cache_hit=True)
        except (OSError, ValueError, TypeError, pickle.PickleError, EOFError):
            pass

    dataset = _parse_normalized_dataset(bundle)
    if use_cache and cache_path is not None:
        atomic_write_bytes(
            cache_path,
            pickle.dumps(
                {
                    "schema": CACHE_SCHEMA_VERSION,
                    "dataset_schema": _PARSED_DATASET_CACHE_SCHEMA,
                    "normalized_fingerprint": bundle.normalized_fingerprint,
                    "dataset": dataset,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )
    return dataset
