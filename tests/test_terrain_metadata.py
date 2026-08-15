from datetime import datetime
import json
from pathlib import Path

from cwr_worldgen.map_picker_coords import (
    parse_bbox_coordinates,
    parse_center_coordinates,
)
from cwr_worldgen.terrain_readme import (
    terrain_readme_filename,
    terrain_readme_text,
)


def test_coordinate_parsers_accept_valid_values() -> None:
    assert parse_center_coordinates("60.1699", "24.9384") == (60.1699, 24.9384)
    assert parse_bbox_coordinates("60", "24", "61", "25") == (60.0, 24.0, 61.0, 25.0)


def test_coordinate_parsers_reject_invalid_order() -> None:
    try:
        parse_bbox_coordinates("61", "24", "60", "25")
    except ValueError as exc:
        assert "south/north" in str(exc)
    else:
        raise AssertionError("invalid bbox order was accepted")


def test_terrain_readme_contains_reproduction_metadata(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "selection": {
                    "kind": "bbox",
                    "bbox_south_west_north_east": [60.0, 24.0, 61.0, 25.0],
                    "center_latitude_longitude": [60.5, 24.5],
                    "cells": 512,
                    "cell_size_metres": 25.0,
                }
            }
        ),
        encoding="utf-8",
    )

    text = terrain_readme_text(
        display_name="Finland Test",
        pbo_name="cwr_finland.pbo",
        source_manifest_path=source_manifest,
        cells=256,
        cell_size_metres=25.0,
        created_at=datetime(2026, 8, 15, 16, 39),
    )

    assert text.startswith("Finland Test\nPBO: cwr_finland.pbo\nVersion: 202608151639\n")
    assert "Selection method: Bounding box" in text
    assert "Center coordinates (Latitude, Longitude): 60.5000000, 24.5000000" in text
    assert "Coordinates (South, West, North, East): 60.0000000, 24.0000000, 61.0000000, 25.0000000" in text
    assert "Terrain cells: 512 x 512" in text
    assert "Cell size: 25 m" in text
    assert "World size: 12800 m x 12800 m" in text
    assert "This Terrain is created by CWR-Worldgen " in text


def test_terrain_readme_filename_is_windows_safe() -> None:
    assert terrain_readme_filename('North:Lake/Test*') == "North_Lake_Test_ ReadMe.txt"
