# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from shapely.geometry import LineString, box, shape

from cwr_worldgen.milestone6 import Milestone6Spec, build_milestone6
from cwr_worldgen.osm import build_overpass_query, parse_overpass_json
from cwr_worldgen.normalization import (
    NormalizationSpec,
    _PolygonCandidate,
    _RoadCandidate,
    _feature,
    _junction_features,
    _lonlat_to_local,
    _forest_edge_outside_area,
    _local_overlap_details,
    _merge_polygon_candidates,
    _normalize_buildings,
    _pairwise_polygon_overlap_area,
    _road_corridor,
    _spatial_union_polygonal,
    normalize_source_bundle,
    validate_normalized_bundle,
    load_normalized_dataset,
)
from cwr_worldgen.osm import BboxProjection
from cwr_worldgen.source_pipeline import SourceFetchSpec, fetch_sources, validate_source_bundle


FIXTURES = Path(__file__).parent / "fixtures"


def _pt(lat: float, lon: float) -> dict[str, float]:
    return {"lat": lat, "lon": lon}


def _complex_osm() -> dict[str, object]:
    return {
        "version": 0.6,
        "generator": "milestone6-test",
        "elements": [
            # Coastline runs south to north. OSM therefore defines land to its
            # west and ocean to its east.
            {"type": "way", "id": 1, "tags": {"natural": "coastline"}, "geometry": [_pt(0.0, 0.005), _pt(0.01, 0.005)]},
            # Split multipolygon outer plus a closed inner hole.
            {"type": "relation", "id": 10, "tags": {"natural": "water", "type": "multipolygon"}, "members": [
                {"type": "way", "ref": 11, "role": "outer", "geometry": [_pt(0.001, 0.001), _pt(0.001, 0.003), _pt(0.003, 0.003)]},
                {"type": "way", "ref": 12, "role": "outer", "geometry": [_pt(0.003, 0.003), _pt(0.003, 0.001), _pt(0.001, 0.001)]},
                {"type": "way", "ref": 13, "role": "inner", "geometry": [_pt(0.0017, 0.0017), _pt(0.0017, 0.0023), _pt(0.0023, 0.0023), _pt(0.0023, 0.0017), _pt(0.0017, 0.0017)]},
            ]},
            # Two compatible segments must merge. A perpendicular road creates
            # a noded degree-four junction. Bridge and tunnel are classified but
            # do not create at-grade crossing nodes.
            {"type": "way", "id": 20, "tags": {"highway": "residential"}, "geometry": [_pt(0.004, 0.001), _pt(0.004, 0.004)]},
            {"type": "way", "id": 21, "tags": {"highway": "residential"}, "geometry": [_pt(0.004, 0.004), _pt(0.004, 0.008)]},
            {"type": "way", "id": 22, "tags": {"highway": "secondary"}, "geometry": [_pt(0.001, 0.004), _pt(0.008, 0.004)]},
            {"type": "way", "id": 23, "tags": {"highway": "service", "bridge": "yes", "layer": "1"}, "geometry": [_pt(0.006, 0.001), _pt(0.006, 0.008)]},
            {"type": "way", "id": 24, "tags": {"highway": "service", "tunnel": "yes", "layer": "-1"}, "geometry": [_pt(0.007, 0.001), _pt(0.007, 0.008)]},
            {"type": "way", "id": 25, "tags": {"highway": "track", "embankment": "yes", "surface": "gravel"}, "geometry": [_pt(0.008, 0.001), _pt(0.008, 0.008)]},
            {"type": "way", "id": 26, "tags": {"highway": "service", "surface": "unpaved"}, "geometry": [_pt(0.0092, 0.001), _pt(0.0092, 0.008)]},
            # Buildings: one overlaps the road corridor, two overlap each other,
            # and one is a self-intersecting bow tie that must be repaired.
            {"type": "way", "id": 30, "tags": {"building": "house"}, "geometry": [_pt(0.0037, 0.0037), _pt(0.0037, 0.0045), _pt(0.0043, 0.0045), _pt(0.0043, 0.0037), _pt(0.0037, 0.0037)]},
            {"type": "way", "id": 31, "tags": {"building": "house"}, "geometry": [_pt(0.0050, 0.0010), _pt(0.0050, 0.0020), _pt(0.0060, 0.0020), _pt(0.0060, 0.0010), _pt(0.0050, 0.0010)]},
            {"type": "way", "id": 32, "tags": {"building": "house"}, "geometry": [_pt(0.0052, 0.0012), _pt(0.0052, 0.0018), _pt(0.0058, 0.0018), _pt(0.0058, 0.0012), _pt(0.0052, 0.0012)]},
            {"type": "way", "id": 33, "tags": {"building": "shed"}, "geometry": [_pt(0.0065, 0.0020), _pt(0.0071, 0.0028), _pt(0.0065, 0.0028), _pt(0.0071, 0.0020), _pt(0.0065, 0.0020)]},
            # Forest with road, building, water, and farmland exclusions.
            {"type": "way", "id": 40, "tags": {"landuse": "forest"}, "geometry": [_pt(0.0005, 0.0005), _pt(0.0005, 0.0090), _pt(0.0090, 0.0090), _pt(0.0090, 0.0005), _pt(0.0005, 0.0005)]},
            {"type": "way", "id": 41, "tags": {"landuse": "farmland"}, "geometry": [_pt(0.0020, 0.0060), _pt(0.0020, 0.0080), _pt(0.0030, 0.0080), _pt(0.0030, 0.0060), _pt(0.0020, 0.0060)]},
            {"type": "way", "id": 1295713286, "tags": {"landuse": "meadow"}, "geometry": [_pt(0.0031, 0.0060), _pt(0.0031, 0.0068), _pt(0.0037, 0.0068), _pt(0.0037, 0.0060), _pt(0.0031, 0.0060)]},
            {"type": "way", "id": 791409714, "tags": {"landuse": "meadow"}, "geometry": [_pt(0.0031, 0.0070), _pt(0.0031, 0.0078), _pt(0.0037, 0.0078), _pt(0.0037, 0.0070), _pt(0.0031, 0.0070)]},
            {"type": "way", "id": 42, "tags": {"landuse": "residential"}, "geometry": [_pt(0.0050, 0.0060), _pt(0.0050, 0.0080), _pt(0.0060, 0.0080), _pt(0.0060, 0.0060), _pt(0.0050, 0.0060)]},
            {"type": "way", "id": 50, "tags": {"waterway": "stream", "name": "Test Stream"}, "geometry": [_pt(0.001, 0.009), _pt(0.009, 0.009)]},
            {"type": "way", "id": 55, "tags": {"barrier": "fence", "fence_type": "chain_link", "material": "metal"}, "geometry": [_pt(0.002, 0.0095), _pt(0.008, 0.0095)]},
            {"type": "node", "id": 60, "lat": 0.005, "lon": 0.005, "tags": {"place": "village", "name": "Åby   tätort", "population": "1250"}},
            {"type": "node", "id": 61, "lat": 0.0051, "lon": 0.0051, "tags": {"place": "village", "name": "Åby tätort", "population": "120"}},
            {"type": "way", "id": 62, "tags": {"place": "isolated_dwelling", "name": "Polygon Home"}, "geometry": [_pt(0.0075, 0.0002), _pt(0.0075, 0.0008), _pt(0.0083, 0.0008), _pt(0.0083, 0.0002), _pt(0.0075, 0.0002)]},
            # Expanded OSM feature classes introduced in 0.9.89.
            {"type": "node", "id": 70, "lat": 0.0010, "lon": 0.0095, "tags": {"natural": "tree", "leaf_type": "broadleaved", "species": "Quercus robur"}},
            {"type": "node", "id": 71, "lat": 0.0015, "lon": 0.0095, "tags": {"power": "pole"}},
            {"type": "node", "id": 72, "lat": 0.0020, "lon": 0.0095, "tags": {"power": "tower"}},
            {"type": "way", "id": 73, "tags": {"man_made": "water_tower"}, "geometry": [_pt(0.0025, 0.0088), _pt(0.0025, 0.0092), _pt(0.0029, 0.0092), _pt(0.0029, 0.0088), _pt(0.0025, 0.0088)]},
            {"type": "way", "id": 74, "tags": {"aeroway": "runway", "surface": "asphalt", "width": "30"}, "geometry": [_pt(0.0096, 0.0008), _pt(0.0096, 0.0060)]},
            {"type": "way", "id": 75, "tags": {"aeroway": "apron", "surface": "asphalt"}, "geometry": [_pt(0.0087, 0.0065), _pt(0.0087, 0.0085), _pt(0.0091, 0.0085), _pt(0.0091, 0.0065), _pt(0.0087, 0.0065)]},
            {"type": "way", "id": 76, "tags": {"natural": "grassland"}, "geometry": [_pt(0.0007, 0.0060), _pt(0.0007, 0.0068), _pt(0.0014, 0.0068), _pt(0.0014, 0.0060), _pt(0.0007, 0.0060)]},
            {"type": "way", "id": 77, "tags": {"leisure": "park"}, "geometry": [_pt(0.0015, 0.0060), _pt(0.0015, 0.0068), _pt(0.0021, 0.0068), _pt(0.0021, 0.0060), _pt(0.0015, 0.0060)]},
            {"type": "way", "id": 78, "tags": {"natural": "beach"}, "geometry": [_pt(0.0005, 0.0042), _pt(0.0005, 0.0048), _pt(0.0012, 0.0048), _pt(0.0012, 0.0042), _pt(0.0005, 0.0042)]},
            {"type": "way", "id": 79, "tags": {"natural": "scrub"}, "geometry": [_pt(0.0070, 0.0060), _pt(0.0070, 0.0067), _pt(0.0077, 0.0067), _pt(0.0077, 0.0060), _pt(0.0070, 0.0060)]},
            {"type": "way", "id": 80, "tags": {"natural": "wetland"}, "geometry": [_pt(0.0079, 0.0060), _pt(0.0079, 0.0067), _pt(0.0086, 0.0067), _pt(0.0086, 0.0060), _pt(0.0079, 0.0060)]},
        ],
    }


class Milestone6Tests(unittest.TestCase):
    def test_fetch_query_requests_watercourses_and_localities(self) -> None:
        query = build_overpass_query((0.0, 0.0, 0.01, 0.01))
        self.assertIn('way["waterway"~"^(river|stream|canal|drain|ditch)$"]', query)
        self.assertIn('nwr["place"="isolated_dwelling"]', query)

    def test_raw_osm_preserves_isolated_dwelling_polygon(self) -> None:
        document = {
            "version": 0.6, "generator": "place-area-test",
            "elements": [{
                "type": "way", "id": 9001,
                "tags": {"place": "isolated_dwelling", "name": "Remote Farm"},
                "geometry": [
                    _pt(0.002, 0.002), _pt(0.002, 0.004), _pt(0.004, 0.004),
                    _pt(0.004, 0.002), _pt(0.002, 0.002),
                ],
            }],
        }
        dataset = parse_overpass_json(json.dumps(document).encode("utf-8"))
        self.assertEqual(len(dataset.place_areas), 1)
        self.assertEqual(dataset.place_areas[0].tags["place"], "isolated_dwelling")
        self.assertEqual(len(dataset.places), 1)

    def _source(self, root: Path) -> Path:
        elevation = root / "elevation" / "raw"
        elevation.mkdir(parents=True)
        values = (72, 73, 74, 71, 72, 73, 70, 71, 72)
        payload = b"".join(value.to_bytes(2, "big", signed=True) for value in values)
        with ZipFile(elevation / "N00E000.hgt.zip", "w") as archive:
            archive.writestr("N00E000.hgt", payload)
        osm = root / "osm"
        osm.mkdir(parents=True)
        (osm / "raw-overpass.json").write_text(json.dumps(_complex_osm()), encoding="utf-8")
        overture = root / "overture"
        overture.mkdir(parents=True)
        (overture / "buildings.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}\n',
            encoding="utf-8",
        )
        fetch_sources(SourceFetchSpec(
            source_dir=root,
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=64,
            cell_size=50.0,
            dem_provider="hgt",
            reference_map=False,
        ))
        return root

    def test_junction_indexing_reports_progress_and_preserves_road_ids(self) -> None:
        roads = [
            _RoadCandidate(
                LineString([(0.0, 50.0), (100.0, 50.0)]),
                {"way/1"},
                {"road_id": "road-000001", "layer": 0, "bridge": False, "tunnel": False},
            ),
            _RoadCandidate(
                LineString([(50.0, 0.0), (50.0, 100.0)]),
                {"way/2"},
                {"road_id": "road-000002", "layer": 0, "bridge": False, "tunnel": False},
            ),
        ]
        updates: list[tuple[int, str]] = []
        junctions = _junction_features(roads, 0.1, progress_callback=lambda value, message: updates.append((value, message)))
        centre = next(properties for point, properties in junctions if point.distance(shape({"type": "Point", "coordinates": [50.0, 50.0]})) < 0.01)
        self.assertEqual(centre["degree"], 4)
        self.assertEqual(centre["road_ids"], ["road-000001", "road-000002"])
        self.assertTrue(any("Noding road layer" in message for _, message in updates))
        self.assertTrue(any("Generated" in message for _, message in updates))

    def test_geojson_round_trip_does_not_reintroduce_road_building_overlap(self) -> None:
        bbox = (59.398881748360814, 16.837989602089053, 59.45643825163919, 16.95115039791095)
        projection = BboxProjection.create(bbox, 6400.0)
        road = _RoadCandidate(
            LineString([(200.0, 300.0), (6200.0, 6100.0)]),
            {"way/1"},
            {
                "road_id": "road-000001",
                "highway": "residential",
                "surface": "asphalt",
                "width_m": 6.0,
                "special": "normal",
                "layer": 0,
                "oneway": False,
                "dirt": False,
                "tunnel": False,
            },
        )
        corridor = _road_corridor([road], 1.5)
        guarded = corridor.buffer(0.10, join_style=2)
        total_overlap = 0.0
        for index in range(80):
            center = 350.0 + index * 70.0
            footprint = box(center - 35.0, center - 35.0, center + 35.0, center + 35.0)
            clipped = footprint.difference(guarded)
            for polygon in getattr(clipped, "geoms", (clipped,)):
                if polygon.is_empty or polygon.geom_type != "Polygon":
                    continue
                document = _feature(polygon, {"building_id": f"building-{index:06d}"}, projection, 8)
                restored = _lonlat_to_local(projection, shape(document["geometry"]))
                total_overlap += restored.intersection(corridor).area
        self.assertLessEqual(total_overlap, 0.05)

    def test_semantic_poi_tags_attach_to_real_building_footprints(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def ll(point: tuple[float, float]) -> dict[str, float]:
            latitude, longitude = projection.to_latlon(point)
            return {"lat": latitude, "lon": longitude}

        elements: list[dict[str, object]] = []
        cases = (
            ("church", 100.0, {"amenity": "place_of_worship", "religion": "christian", "name": "Test Church"}),
            ("school", 400.0, {"amenity": "school", "name": "Test School"}),
            ("shop", 700.0, {"shop": "convenience", "name": "Test Shop"}),
        )
        for index, (_kind, x, semantic_tags) in enumerate(cases, start=1):
            ring = ((x, 100.0), (x + 40.0, 100.0), (x + 40.0, 130.0), (x, 130.0), (x, 100.0))
            elements.append({
                "type": "way",
                "id": index,
                "tags": {"building": "yes"},
                "geometry": [ll(point) for point in ring],
            })
            latitude, longitude = projection.to_latlon((x + 20.0, 115.0))
            elements.append({
                "type": "node",
                "id": 100 + index,
                "lat": latitude,
                "lon": longitude,
                "tags": semantic_tags,
            })

        buildings, statistics = _normalize_buildings(
            elements,
            projection,
            box(0.0, 0.0, 1000.0, 1000.0),
            [],
            NormalizationSpec(source_dir=Path("unused")),
        )
        self.assertEqual(len(buildings), 3)
        self.assertEqual(statistics["semantic_attached"], 3)
        self.assertEqual(statistics["semantic_synthetic"], 0)
        by_name = {building.properties["name"]: building for building in buildings}
        self.assertEqual(by_name["Test Church"].properties["amenity"], "place_of_worship")
        self.assertEqual(by_name["Test Church"].properties["building_kind"], "church")
        self.assertEqual(by_name["Test School"].properties["amenity"], "school")
        self.assertEqual(by_name["Test School"].properties["building_kind"], "school")
        self.assertEqual(by_name["Test Shop"].properties["shop"], "convenience")
        self.assertEqual(by_name["Test Shop"].properties["building_kind"], "retail")
        self.assertTrue(all(len(building.source_ids) == 2 for building in buildings))

    def test_school_campus_polygon_marks_every_contained_building_as_school(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def ll(point: tuple[float, float]) -> dict[str, float]:
            latitude, longitude = projection.to_latlon(point)
            return {"lat": latitude, "lon": longitude}

        elements: list[dict[str, object]] = []
        for index, (x0, y0, x1, y1) in enumerate((
            (210.0, 210.0, 240.0, 225.0),
            (260.0, 210.0, 290.0, 225.0),
            (235.0, 250.0, 270.0, 270.0),
        ), start=1):
            ring = ((x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0))
            elements.append({
                "type": "way", "id": index,
                "tags": {"building": "yes"},
                "geometry": [ll(point) for point in ring],
            })
        campus = ((190.0,190.0),(310.0,190.0),(310.0,290.0),(190.0,290.0),(190.0,190.0))
        elements.append({
            "type": "way", "id": 100,
            "tags": {"amenity": "school", "name": "Three Building School"},
            "geometry": [ll(point) for point in campus],
        })

        buildings, statistics = _normalize_buildings(
            elements, projection, box(0.0, 0.0, 1000.0, 1000.0), [],
            NormalizationSpec(source_dir=Path("unused")),
        )
        self.assertEqual(len(buildings), 3)
        self.assertEqual(statistics["semantic_attached"], 3)
        self.assertTrue(all(building.properties.get("amenity") == "school" for building in buildings))
        self.assertTrue(all(building.properties.get("building_kind") == "school" for building in buildings))
        self.assertTrue(all(building.properties.get("name") == "Three Building School" for building in buildings))

    def test_social_facility_area_marks_contained_barns_and_warehouses_as_public(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def ll(point: tuple[float, float]) -> dict[str, float]:
            latitude, longitude = projection.to_latlon(point)
            return {"lat": latitude, "lon": longitude}

        campus = ((180.0, 180.0), (340.0, 180.0), (340.0, 310.0), (180.0, 310.0), (180.0, 180.0))
        elements: list[dict[str, object]] = [{
            "type": "way", "id": 700,
            "tags": {
                "amenity": "social_facility",
                "social_facility": "group_home",
                "social_facility:for": "senior",
                "building": "no",
                "name": "Care Campus",
            },
            "geometry": [ll(point) for point in campus],
        }]
        for index, (building_kind, x0) in enumerate((("barn", 205.0), ("warehouse", 270.0)), start=1):
            ring = ((x0, 215.0), (x0 + 42.0, 215.0), (x0 + 42.0, 245.0), (x0, 245.0), (x0, 215.0))
            elements.append({
                "type": "way", "id": 700 + index,
                "tags": {"building": building_kind},
                "geometry": [ll(point) for point in ring],
            })

        buildings, _statistics = _normalize_buildings(
            elements, projection, box(0.0, 0.0, 1000.0, 1000.0), [],
            NormalizationSpec(source_dir=Path("unused")),
        )
        self.assertEqual(len(buildings), 2)
        for item in buildings:
            self.assertEqual(item.properties.get("amenity"), "social_facility")
            self.assertEqual(item.properties.get("social_facility"), "group_home")
            self.assertEqual(item.properties.get("social_facility:for"), "senior")
            self.assertEqual(item.properties.get("building_kind"), "public")

    def test_school_campus_building_no_still_marks_all_physical_buildings(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def ll(point: tuple[float, float]) -> dict[str, float]:
            latitude, longitude = projection.to_latlon(point)
            return {"lat": latitude, "lon": longitude}

        campus = ((180.0, 180.0), (320.0, 180.0), (320.0, 300.0), (180.0, 300.0), (180.0, 180.0))
        elements: list[dict[str, object]] = [{
            "type": "way", "id": 900,
            "tags": {"amenity": "school", "building": "no", "name": "No Building Campus"},
            "geometry": [ll(point) for point in campus],
        }]
        for index, x0 in enumerate((205.0, 255.0), start=1):
            ring = ((x0, 210.0), (x0 + 32.0, 210.0), (x0 + 32.0, 228.0), (x0, 228.0), (x0, 210.0))
            elements.append({
                "type": "way", "id": 900 + index,
                "tags": {"building": "yes"},
                "geometry": [ll(point) for point in ring],
            })

        buildings, _statistics = _normalize_buildings(
            elements, projection, box(0.0, 0.0, 1000.0, 1000.0), [],
            NormalizationSpec(source_dir=Path("unused")),
        )
        self.assertEqual(len(buildings), 2)
        self.assertTrue(all(item.properties.get("amenity") == "school" for item in buildings))
        self.assertTrue(all(item.properties.get("building_kind") == "school" for item in buildings))

    def test_unnamed_isolated_dwelling_polygon_is_preserved_from_raw_osm(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def ll(point: tuple[float, float]) -> dict[str, float]:
            latitude, longitude = projection.to_latlon(point)
            return {"lat": latitude, "lon": longitude}

        ring = ((400.0, 400.0), (470.0, 400.0), (470.0, 470.0), (400.0, 470.0), (400.0, 400.0))
        dataset = parse_overpass_json(json.dumps({
            "elements": [{
                "type": "way", "id": 1295713271,
                "tags": {"place": "isolated_dwelling"},
                "geometry": [ll(point) for point in ring],
            }]
        }).encode("utf-8"))
        self.assertEqual(len(dataset.place_areas), 1)
        self.assertEqual(dataset.place_areas[0].osm_key, "way/1295713271")
        self.assertEqual(dataset.place_areas[0].tags.get("place"), "isolated_dwelling")

    def test_sparse_building_merge_reports_spatial_progress(self) -> None:
        candidates = [
            _PolygonCandidate(
                box((index % 40) * 30.0, (index // 40) * 30.0, (index % 40) * 30.0 + 10.0, (index // 40) * 30.0 + 10.0),
                {f"way/{index}"},
                {"building_kind": "house"},
            )
            for index in range(400)
        ]
        events: list[tuple[int, str]] = []
        merged = _merge_polygon_candidates(
            candidates, 0.75,
            progress_callback=lambda percent, stage: events.append((percent, stage)),
        )
        self.assertEqual(len(merged), 400)
        self.assertTrue(any("Spatially merged" in stage for _percent, stage in events))
        self.assertEqual(events[-1][0], 100)

    def test_spatial_polygon_union_preserves_disjoint_landuse_components(self) -> None:
        polygons = [
            box((index % 20) * 30.0, (index // 20) * 30.0, (index % 20) * 30.0 + 10.0, (index // 20) * 30.0 + 10.0)
            for index in range(400)
        ]
        merged = _spatial_union_polygonal(polygons)
        self.assertAlmostEqual(merged.area, sum(polygon.area for polygon in polygons), places=6)
        self.assertEqual(len(getattr(merged, "geoms", (merged,))), 400)

    def test_normalization_reuses_prevalidated_source_without_second_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            validation = validate_source_bundle(source)
            events: list[tuple[int, str]] = []
            with patch("cwr_worldgen.normalization.validate_source_bundle") as duplicate_validation:
                normalize_source_bundle(
                    NormalizationSpec(source_dir=source, output_dir=root / "normalized"),
                    validated_source=validation.bundle,
                    progress_callback=lambda percent, stage: events.append((percent, stage)),
                )
            duplicate_validation.assert_not_called()
            self.assertTrue(any("already validated" in stage for _percent, stage in events))

    def test_normalized_bundle_round_trips_isolated_dwelling_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            bundle = normalize_source_bundle(
                NormalizationSpec(source_dir=source, output_dir=root / "normalized", refresh=True)
            )
            parsed = load_normalized_dataset(bundle, use_cache=False)
            areas = [area for area in parsed.place_areas if area.tags.get("name") == "Polygon Home"]
            self.assertEqual(len(areas), 1)
            self.assertEqual(areas[0].tags.get("place"), "isolated_dwelling")

    def test_matching_normalized_bundle_reuses_manifest_without_full_disk_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            spec = NormalizationSpec(source_dir=source, output_dir=root / "normalized")
            first = normalize_source_bundle(spec)
            events: list[tuple[int, str]] = []
            with patch("cwr_worldgen.normalization.validate_normalized_bundle") as full_validation:
                second = normalize_source_bundle(
                    spec,
                    progress_callback=lambda percent, stage: events.append((percent, stage)),
                )
            full_validation.assert_not_called()
            self.assertEqual(first.normalized_fingerprint, second.normalized_fingerprint)
            self.assertEqual(first.counts, second.counts)
            self.assertTrue(any("previously validated" in stage for _percent, stage in events))

    def test_normalization_reports_building_cleanup_substages(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            events: list[tuple[int, str]] = []
            normalize_source_bundle(
                NormalizationSpec(source_dir=source, output_dir=root / "normalized"),
                progress_callback=lambda percent, stage: events.append((percent, stage)),
            )
            stages = [stage for _percent, stage in events]
            self.assertTrue(any("Spatially indexing" in stage for stage in stages))
            self.assertTrue(any("Clipping buildings near roads" in stage for stage in stages))
            self.assertTrue(any("Resolving building overlaps" in stage for stage in stages))
            self.assertTrue(any("Unioning" in stage and "forest" in stage for stage in stages))
            self.assertTrue(any("in-memory" in stage or "Verified normalized layer" in stage for stage in stages))

    def test_normalized_geojson_is_complete_valid_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            one = normalize_source_bundle(NormalizationSpec(source_dir=source, output_dir=root / "one"))
            two = normalize_source_bundle(NormalizationSpec(source_dir=source, output_dir=root / "two"))
            self.assertEqual(one.counts, two.counts)
            self.assertEqual(one.normalized_fingerprint, two.normalized_fingerprint)
            self.assertTrue(validate_normalized_bundle(one.root).validation_path.is_file())
            for filename in one.files:
                self.assertEqual((one.root / filename).read_bytes(), (two.root / filename).read_bytes())
                document = json.loads((one.root / filename).read_text(encoding="utf-8"))
                self.assertTrue(all(shape(feature["geometry"]).is_valid for feature in document["features"]))

    def test_normalization_repairs_topology_and_emits_required_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            bundle = normalize_source_bundle(NormalizationSpec(source_dir=source))

            roads = json.loads(bundle.files["roads.geojson"].read_text(encoding="utf-8"))["features"]
            gravel_roads = json.loads(bundle.files["gravel-roads.geojson"].read_text(encoding="utf-8"))["features"]
            specials = {feature["properties"]["special"] for feature in roads}
            self.assertTrue({"bridge", "tunnel"}.issubset(specials))
            self.assertEqual({feature["properties"]["surface"] for feature in gravel_roads}, {"gravel", "unpaved"})
            self.assertIn("embankment", {feature["properties"]["special"] for feature in gravel_roads})
            self.assertTrue(any(len(feature["properties"]["source_ids"]) >= 2 for feature in roads))
            parsed = load_normalized_dataset(bundle, use_cache=False)
            self.assertEqual(len(parsed.gravel_roads), len(gravel_roads))
            self.assertEqual({feature.tags.get("surface") for feature in parsed.gravel_roads}, {"gravel", "unpaved"})
            # Frozen GeoJSON stores bridge/tunnel/embankment as booleans. Those
            # flags must survive reloading or bridge decks silently disappear in
            # later builds even though the raw OSM way was correctly tagged.
            self.assertTrue(any(feature.tags.get("bridge") == "yes" for feature in parsed.roads))
            self.assertTrue(any(feature.tags.get("tunnel") == "yes" for feature in parsed.roads))
            self.assertTrue(any(feature.tags.get("embankment") == "yes" for feature in parsed.roads))

            junctions = json.loads(bundle.files["road-junctions.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertTrue(any(feature["properties"]["degree"] >= 4 for feature in junctions))

            water = json.loads(bundle.files["water.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertEqual({feature["properties"]["kind"] for feature in water}, {"ocean", "inland"})

            landuse = json.loads(bundle.files["landuse.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertIn("meadow", {feature["properties"]["category"] for feature in landuse})
            self.assertTrue(any(feature.tags.get("landuse") == "meadow" for feature in parsed.farmland))

            buildings = json.loads(bundle.files["buildings.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertGreater(len(buildings), 0)
            self.assertTrue(all(feature["properties"]["area_m2"] >= 20 for feature in buildings))

            barriers = json.loads(bundle.files["barriers.geojson"].read_text(encoding="utf-8"))["features"]
            chain_link = next(feature for feature in barriers if feature["properties"]["source_id"] == "way/55")
            self.assertEqual(chain_link["properties"]["fence_type"], "chain_link")
            self.assertEqual(chain_link["properties"]["material"], "metal")

            forests = json.loads(bundle.files["forests.geojson"].read_text(encoding="utf-8"))["features"]
            edges = json.loads(bundle.files["forest-edges.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertGreater(len(forests), 0)
            self.assertGreater(len(edges), 0)
            self.assertTrue(any(feature["properties"]["edge_area_m2"] > 0 for feature in forests))

            watercourses = json.loads(bundle.files["watercourses.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertEqual(len(watercourses), 1)
            self.assertEqual(watercourses[0]["properties"]["kind"], "stream")

            places = json.loads(bundle.files["places.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertEqual(len(places), 2)
            self.assertTrue(any(feature["geometry"]["type"] in {"Polygon", "MultiPolygon"} for feature in places))
            self.assertEqual(places[0]["properties"]["name_ascii"], "Aby tatort")
            self.assertEqual(places[0]["properties"]["population"], 1250)

            trees = json.loads(bundle.files["trees.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertEqual(len(trees), 1)
            self.assertEqual(trees[0]["properties"]["leaf_type"], "broadleaved")

            aeroway_lines = json.loads(bundle.files["aeroway-lines.geojson"].read_text(encoding="utf-8"))["features"]
            aeroway_areas = json.loads(bundle.files["aeroway-areas.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertEqual({feature["properties"]["kind"] for feature in aeroway_lines}, {"runway"})
            self.assertEqual({feature["properties"]["kind"] for feature in aeroway_areas}, {"apron"})

            utilities = json.loads(bundle.files["utility-points.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertEqual(
                {feature["properties"]["kind"] for feature in utilities},
                {"power_pole", "power_tower", "water_tower"},
            )

            surfaces = json.loads(bundle.files["surface-areas.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertEqual(
                {feature["properties"]["kind"] for feature in surfaces},
                {"grassland", "park", "beach"},
            )
            rural = json.loads(bundle.files["rural-vegetation.geojson"].read_text(encoding="utf-8"))["features"]
            self.assertTrue({"scrub", "wetland"}.issubset(
                {feature["properties"]["kind"] for feature in rural}
            ))
            self.assertEqual(len(parsed.individual_trees), 1)
            self.assertEqual(len(parsed.aeroway_lines), 1)
            self.assertEqual(len(parsed.aeroway_areas), 1)
            self.assertEqual(len(parsed.utility_points), 3)
            self.assertEqual(len(parsed.surface_areas), 3)
            self.assertTrue({"scrub", "wetland"}.issubset(
                {feature.tags.get("natural") for feature in parsed.rural_vegetation}
            ))

    def test_indexed_topology_helpers_match_exact_local_overlaps(self) -> None:
        buildings = [box(0, 0, 10, 10), box(9, 0, 19, 10), box(30, 0, 40, 10)]
        self.assertAlmostEqual(_pairwise_polygon_overlap_area(buildings), 10.0, places=6)

        roads = [box(4, -2, 6, 12), box(34, -2, 36, 12)]
        total, maximum, offenders = _local_overlap_details(buildings, roads)
        exact = [building.intersection(roads[0].union(roads[1])).area for building in buildings]
        self.assertAlmostEqual(total, sum(exact), places=6)
        self.assertAlmostEqual(maximum, max(exact), places=6)
        self.assertEqual([index for index, _ in offenders], [0, 2])

    def test_indexed_forest_edge_validation_uses_nearby_forests(self) -> None:
        forests = [box(0, 0, 20, 20), box(1000, 1000, 1020, 1020)]
        edges = [box(1, 1, 5, 5), box(18, 18, 22, 22)]
        outside = _forest_edge_outside_area(edges, forests, tolerance=0.05)
        expected = sum(edge.difference(forests[0].union(forests[1]).buffer(0.05)).area for edge in edges)
        self.assertAlmostEqual(outside, expected, places=6)

    def test_milestone6_build_uses_normalized_geometry_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            spec = Milestone6Spec(
                source_dir=source,
                name="cwr_m6_test",
                display_name="CWR M6 Test",
                asset_roots=(FIXTURES / "assets",),
                strict_assets=True,
            )
            with patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
                first = build_milestone6(root / "one", spec)
                second = build_milestone6(root / "two", spec)
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.pbo_path.read_bytes(), second.pbo_path.read_bytes())
            self.assertEqual(first.pbo_path.parent.parent.name, "@CWR-Milestone6")
            self.assertTrue(first.normalized_dir.is_dir())
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["milestone"], 6)
            self.assertEqual(manifest["normalized_geometry"]["manifest_sha256"], (source / "normalized" / "manifest.json").is_file() and validate_normalized_bundle(source / "normalized").normalized_fingerprint)
            self.assertTrue((first.pbo_path.parent.parent / "NORMALIZED-GEOMETRY.json").is_file())


if __name__ == "__main__":
    unittest.main()
