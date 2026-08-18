# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from cwr_worldgen.model import OsmSpec
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmPolygonFeature,
    augment_dataset_with_overture_buildings,
)


def _dataset(feature: OsmPolygonFeature) -> OsmDataset:
    return OsmDataset(
        source_generator="test",
        element_count=1,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
        building_polygons=(feature,),
        normalized_fingerprint="base-test",
    )


def _polygon(points: list[tuple[float, float]]) -> GeoPolygon:
    ring = tuple(points + [points[0]])
    return GeoPolygon(ring)


def _geojson_polygon(points: list[tuple[float, float]]) -> list[list[list[float]]]:
    # Internal PointLL is lat/lon; GeoJSON is lon/lat.
    ring = [[lon, lat] for lat, lon in points + [points[0]]]
    return [ring]


def _write_overture(path: Path, *, feature_id: str, points, properties) -> None:
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": feature_id,
                "properties": {"id": feature_id, **properties},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": _geojson_polygon(points),
                },
            }
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_overture_source_id_merge_keeps_osm_geometry_and_explicit_tags() -> None:
    bbox = (59.0, 18.0, 59.01, 18.01)
    spec = OsmSpec(heightmap_path=Path("dummy.png"), bbox=bbox)
    projection = BboxProjection.create(bbox, spec.world_size)
    points = [
        (59.0040, 18.0040),
        (59.0040, 18.0042),
        (59.0042, 18.0042),
        (59.0042, 18.0040),
    ]
    osm = OsmPolygonFeature(
        "way/123",
        {"building": "house", "roof:shape": "pyramidal"},
        (_polygon(points),),
    )
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "overture.geojson"
        _write_overture(
            path,
            feature_id="gers-123",
            points=points,
            properties={
                "sources": [
                    {
                        "dataset": "OpenStreetMap",
                        "record_id": "w123@7",
                    }
                ],
                "class": "residential",
                "subtype": "residential",
                "height": 8.6,
                "num_floors": 2,
                "roof_shape": "hipped",
                "roof_height": 2.1,
                "facade_material": "brick",
                "roof_material": "tile",
            },
        )
        result = augment_dataset_with_overture_buildings(
            _dataset(osm), projection, spec, path
        )

    assert len(result.building_polygons) == 1
    merged = result.building_polygons[0]
    assert merged.osm_key == "way/123"
    assert merged.polygons == osm.polygons
    assert merged.tags["building"] == "house"
    assert merged.tags["roof:shape"] == "pyramidal"  # explicit OSM wins
    assert merged.tags["height"] == "8.6"
    assert merged.tags["building:levels"] == "2"
    assert merged.tags["roof:height"] == "2.1"
    assert merged.tags["building:material"] == "brick"
    assert merged.tags["roof:material"] == "tile"
    assert merged.tags["cwr:overture_id"] == "gers-123"
    assert merged.tags["cwr:overture_match"] == "source-id"
    assert "source" not in merged.tags
    assert "cwr:synthetic" not in merged.tags


def test_overture_geometry_merge_fills_missing_roof_without_duplicate() -> None:
    bbox = (59.0, 18.0, 59.01, 18.01)
    spec = OsmSpec(heightmap_path=Path("dummy.png"), bbox=bbox)
    projection = BboxProjection.create(bbox, spec.world_size)
    osm_points = [
        (59.0060, 18.0060),
        (59.0060, 18.0062),
        (59.0062, 18.0062),
        (59.0062, 18.0060),
    ]
    # Small offset represents a second source tracing the same roof/footprint.
    overture_points = [(lat + 0.000005, lon + 0.000005) for lat, lon in osm_points]
    osm = OsmPolygonFeature("way/456", {"building": "yes"}, (_polygon(osm_points),))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "overture.geojson"
        _write_overture(
            path,
            feature_id="gers-456",
            points=overture_points,
            properties={
                "sources": [{"dataset": "Microsoft", "record_id": "abc"}],
                "class": "house",
                "height": 7.4,
                "roof_shape": "onion",
                "facade_color": "#f4e8d0",
            },
        )
        result = augment_dataset_with_overture_buildings(
            _dataset(osm), projection, spec, path
        )

    assert len(result.building_polygons) == 1
    merged = result.building_polygons[0]
    assert merged.osm_key == "way/456"
    assert merged.tags["building"] == "house"
    assert merged.tags["height"] == "7.4"
    assert merged.tags["roof:shape"] == "onion"
    assert merged.tags["building:colour"] == "#f4e8d0"
    assert merged.tags["cwr:overture_match"] == "geometry"


def test_overture_half_hipped_name_is_normalized_for_cwr_roof_parser() -> None:
    bbox = (59.0, 18.0, 59.01, 18.01)
    spec = OsmSpec(heightmap_path=Path("dummy.png"), bbox=bbox)
    projection = BboxProjection.create(bbox, spec.world_size)
    points = [
        (59.0020, 18.0020),
        (59.0020, 18.0022),
        (59.0022, 18.0022),
        (59.0022, 18.0020),
    ]
    osm = OsmPolygonFeature("way/789", {"building": "house"}, (_polygon(points),))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "overture.geojson"
        _write_overture(
            path,
            feature_id="gers-789",
            points=points,
            properties={
                "sources": [{"dataset": "OpenStreetMap", "record_id": "w789@1"}],
                "roof_shape": "half_hipped",
            },
        )
        result = augment_dataset_with_overture_buildings(
            _dataset(osm), projection, spec, path
        )

    assert result.building_polygons[0].tags["roof:shape"] == "half-hipped"
