from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cwr_worldgen import overture


def _world_50km_bbox() -> tuple[float, float, float, float]:
    # About 50 x 50 km near latitude 55 N.
    south = 55.0
    north = south + (50.0 / 111.32)
    mid = math.radians((south + north) * 0.5)
    west = 12.0
    east = west + (50.0 / (111.32 * math.cos(mid)))
    return south, west, north, east


def test_50km_world_splits_into_25_roughly_10km_tiles() -> None:
    bbox = _world_50km_bbox()
    tiles = overture._overture_bbox_tiles(bbox, maximum_edge_km=10.0)
    assert len(tiles) == 25
    for tile in tiles:
        latitude_km, longitude_km = overture._bbox_size_km(tile)
        assert latitude_km <= 10.1
        assert longitude_km <= 10.1


def test_adaptive_tile_timeout_is_longer_than_old_global_timeout() -> None:
    south, west, _, _ = _world_50km_bbox()
    small = (south, west, south + 3.0 / 111.32, west + 3.0 / (111.32 * math.cos(math.radians(south))))
    medium = (south, west, south + 9.0 / 111.32, west + 9.0 / (111.32 * math.cos(math.radians(south))))
    assert overture._overture_tile_timeout_seconds(small) == 180
    assert overture._overture_tile_timeout_seconds(medium) == 300


def test_tile_merge_deduplicates_buildings_by_overture_id() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first.geojson"
        second = root / "second.geojson"
        output = root / "merged.geojson"
        shared = {
            "type": "Feature",
            "id": "same-building",
            "properties": {"id": "same-building", "height": 8.5},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        first.write_text(json.dumps({"type": "FeatureCollection", "features": [shared]}), encoding="utf-8")
        second.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        shared,
                        {
                            "type": "Feature",
                            "id": "other-building",
                            "properties": {"id": "other-building"},
                            "geometry": {"type": "Polygon", "coordinates": [[[2, 0], [3, 0], [3, 1], [2, 0]]]},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        written, duplicates = overture._merge_overture_tile_geojson([first, second], output)
        document = json.loads(output.read_text(encoding="utf-8"))
        assert written == 2
        assert duplicates == 1
        assert [feature["id"] for feature in document["features"]] == ["same-building", "other-building"]


def test_interrupted_fetch_resumes_completed_tiles() -> None:
    bbox = _world_50km_bbox()
    with TemporaryDirectory() as directory:
        output = Path(directory) / "overture-buildings.geojson"
        first_calls: list[int] = []

        def fail_on_third(tile_bbox, tile_output, **kwargs):
            number = int(kwargs["tile_number"])
            first_calls.append(number)
            if number == 3:
                return False
            tile_output.parent.mkdir(parents=True, exist_ok=True)
            tile_output.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "id": f"building-{number}",
                                "properties": {"id": f"building-{number}"},
                                "geometry": {"type": "Point", "coordinates": [0, 0]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return True

        with patch.object(overture, "selected_overture_release", return_value="2026-07-22.0"), patch.object(
            overture, "_run_overture_tile_worker", side_effect=fail_on_third
        ):
            result = overture.fetch_overture_buildings_geojson(
                bbox,
                output,
                max_attempts=1,
            )
        assert result is None
        assert first_calls == [1, 2, 3]
        manifest_path = output.parent / "overture-tiles" / "2026-07-22.0" / f"world-{overture._bbox_digest(bbox)}.tiles.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["complete"] is False
        assert len(manifest["completed"]) == 2

        second_calls: list[int] = []

        def succeed_remaining(tile_bbox, tile_output, **kwargs):
            number = int(kwargs["tile_number"])
            second_calls.append(number)
            tile_output.parent.mkdir(parents=True, exist_ok=True)
            tile_output.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "id": f"building-{number}",
                                "properties": {"id": f"building-{number}"},
                                "geometry": {"type": "Point", "coordinates": [0, 0]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return True

        with patch.object(overture, "selected_overture_release", return_value="2026-07-22.0"), patch.object(
            overture, "_run_overture_tile_worker", side_effect=succeed_remaining
        ):
            result = overture.fetch_overture_buildings_geojson(
                bbox,
                output,
                max_attempts=1,
            )
        assert result == output
        assert second_calls == list(range(3, 26))
        document = json.loads(output.read_text(encoding="utf-8"))
        assert len(document["features"]) == 25
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["complete"] is True
        assert manifest["features"] == 25


def test_incomplete_refresh_session_reuses_tiles_completed_in_that_session() -> None:
    bbox = _world_50km_bbox()
    with TemporaryDirectory() as directory:
        output = Path(directory) / "overture-buildings.geojson"
        release = "2026-07-22.0"
        tiles = overture._overture_bbox_tiles(bbox)
        cache_root = output.parent / "overture-tiles" / release
        first_tile = overture._tile_cache_path(cache_root, tiles[0])
        first_tile.parent.mkdir(parents=True, exist_ok=True)
        first_tile.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        overture._write_tile_manifest(
            cache_root / f"world-{overture._bbox_digest(bbox)}.tiles.json",
            {
                "schema": 1,
                "release": release,
                "world_bbox_digest": overture._bbox_digest(bbox),
                "bbox": list(bbox),
                "tile_edge_km": 10.0,
                "tile_count": len(tiles),
                "completed": [overture._bbox_digest(tiles[0])],
                "complete": False,
            },
        )
        calls: list[int] = []

        def succeed(tile_bbox, tile_output, **kwargs):
            number = int(kwargs["tile_number"])
            calls.append(number)
            tile_output.parent.mkdir(parents=True, exist_ok=True)
            tile_output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            return True

        with patch.object(overture, "selected_overture_release", return_value=release), patch.object(
            overture, "_run_overture_tile_worker", side_effect=succeed
        ):
            result = overture.fetch_overture_buildings_geojson(
                bbox,
                output,
                refresh=True,
                max_attempts=1,
            )
        assert result == output
        assert calls == list(range(2, 26))
