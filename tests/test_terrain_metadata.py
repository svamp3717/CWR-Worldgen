from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import cli, milestone9
from cwr_worldgen import terrain_readme as terrain_readme_module
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


def test_cli_milestone9_build_uses_terrain_readme_wrapper() -> None:
    assert cli.build_milestone9 is milestone9.build_milestone9
    assert getattr(cli.build_milestone9, "_cwr_terrain_readme", False)
    assert getattr(milestone9._deploy_runtime_to_existing_mod, "_cwr_terrain_readme", False)


def test_terrain_readme_is_copied_by_same_deployment_pass_as_pbo(tmp_path: Path) -> None:
    output_dir = tmp_path / "build"
    runtime_root = output_dir / "runtime"
    source_addons = runtime_root / "Addons"
    source_anims = runtime_root / "Anims"
    source_addons.mkdir(parents=True)
    intro_dir = source_anims / "intro.wg_test"
    intro_dir.mkdir(parents=True)

    pbo_path = source_addons / "wg_test.pbo"
    pbo_path.write_bytes(b"pbo")
    intro_path = intro_dir / "mission.sqm"
    intro_path.write_text("mission", encoding="utf-8")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "source.json").write_text(
        json.dumps(
            {
                "selection": {
                    "kind": "center",
                    "center_latitude_longitude": [18.5, -69.9],
                    "cells": 256,
                    "cell_size_metres": 25.0,
                }
            }
        ),
        encoding="utf-8",
    )

    result = SimpleNamespace(
        output_dir=output_dir,
        pbo_path=pbo_path,
        intro_mission_path=intro_path,
    )
    spec = SimpleNamespace(
        display_name="Test Terrain",
        source_dir=source_dir,
        cells=256,
        cell_size=25.0,
    )
    deploy_root = tmp_path / "@test"
    deploy_root.mkdir()

    token = terrain_readme_module._ACTIVE_TERRAIN_SPEC.set(spec)
    try:
        report = milestone9._deploy_runtime_to_existing_mod(result, deploy_root)
    finally:
        terrain_readme_module._ACTIVE_TERRAIN_SPEC.reset(token)

    deployed_addons = deploy_root / "Addons"
    deployed_readme = deployed_addons / "Test Terrain ReadMe.txt"
    assert (deployed_addons / "wg_test.pbo").read_bytes() == b"pbo"
    assert deployed_readme.is_file()
    assert "PBO: wg_test.pbo" in deployed_readme.read_text(encoding="utf-8")
    assert {Path(item["destination"]).name for item in report["files"]} >= {
        "wg_test.pbo",
        "Test Terrain ReadMe.txt",
    }
