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
        building_preset="Automatic (area / country)",
        building_preset_identifier="auto",
        resolved_building_style="Finland",
        resolved_building_style_identifier="fi_finland",
        building_styles=("nordic_wood", "nordic_stucco"),
        building_classes=("house", "apartments"),
        terrain_style="everon",
        forest_style="malden",
        appearance_preset="Everon terrain + Malden vegetation",
        created_at=datetime(2026, 8, 15, 16, 39),
    )

    assert text.startswith("Finland Test\nPBO: cwr_finland.pbo\nVersion: 202608151639\n")
    assert "Build Presets" in text
    assert "Building preset: Automatic (area / country) [auto]" in text
    assert "Resolved building style: Finland [fi_finland]" in text
    assert "Building styles used: Nordic Stucco, Nordic Wood" in text
    assert "Building classes used: Apartments, House" in text
    assert "Appearance preset: Everon terrain + Malden vegetation" in text
    assert "Terrain style: Everon" in text
    assert "Vegetation / forest preset: Malden" in text
    assert "Selection method: Bounding box" in text
    assert "Center coordinates (Latitude, Longitude): 60.5000000, 24.5000000" in text
    assert "Coordinates (South, West, North, East): 60.0000000, 24.0000000, 61.0000000, 25.0000000" in text
    assert "Terrain cells: 512 x 512" in text
    assert "Cell size: 25 m" in text
    assert "World size: 12800 m x 12800 m" in text
    assert "This Terrain is created by CWR-Worldgen " in text


def test_terrain_readme_uses_actual_building_catalogue_styles(tmp_path: Path) -> None:
    output_dir = tmp_path / "build"
    addons = output_dir / "runtime" / "Addons"
    addons.mkdir(parents=True)
    pbo_path = addons / "wg_styles.pbo"
    pbo_path.write_bytes(b"pbo")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "source.json").write_text(
        json.dumps({"selection": {"kind": "center", "cells": 256, "cell_size_metres": 25.0}}),
        encoding="utf-8",
    )
    catalogue = output_dir / "building-asset-catalogue.json"
    catalogue.write_text(
        json.dumps(
            {
                "house_style_preset": "auto",
                "house_style_region": "se_sweden",
                "detected_house_style_region": "northern_europe",
                "request_mapping": [
                    {
                        "selected": {
                            "regional_style": "sweden_red",
                            "country_style_identifier": "se_sweden",
                            "building_class": "house",
                        }
                    },
                    {
                        "selected": {
                            "regional_style": "sweden_yellow",
                            "country_style_identifier": "se_sweden",
                            "building_class": "apartments",
                        }
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result = SimpleNamespace(
        output_dir=output_dir,
        pbo_path=pbo_path,
        building_catalogue_path=catalogue,
    )
    spec = SimpleNamespace(
        display_name="Style Test",
        source_dir=source_dir,
        cells=256,
        cell_size=25.0,
        house_style_preset="auto",
        ground_texture_profile="malden",
        forest_profile="malden",
    )

    readme = terrain_readme_module.write_terrain_readme(result, spec).read_text(encoding="utf-8")

    assert "Building preset: Automatic (area / country) [auto]" in readme
    assert "[se_sweden]" in readme
    assert "Building styles used: Se Sweden, Sweden Red, Sweden Yellow" in readme
    assert "Building classes used: Apartments, House" in readme
    assert "Appearance preset: Malden classic" in readme
    assert "Terrain style: Malden" in readme
    assert "Vegetation / forest preset: Malden" in readme


def test_terrain_readme_filename_is_windows_safe() -> None:
    assert terrain_readme_filename('North:Lake/Test*') == "North_Lake_Test_ ReadMe.txt"


def test_cli_milestone9_build_uses_terrain_readme_wrapper() -> None:
    assert cli.build_milestone9 is milestone9.build_milestone9
    assert getattr(cli.build_milestone9, "_cwr_terrain_readme", False)
    assert getattr(milestone9._deploy_runtime_to_existing_mod, "_cwr_terrain_readme", False)


def test_cli_readme_binding_replaces_any_stale_build_reference() -> None:
    current = cli.build_milestone9
    try:
        cli.build_milestone9 = lambda *_args, **_kwargs: None
        assert cli.build_milestone9 is not milestone9.build_milestone9
        terrain_readme_module._sync_cli_build_binding(milestone9.build_milestone9)
        assert cli.build_milestone9 is milestone9.build_milestone9
    finally:
        cli.build_milestone9 = current


def test_terrain_readme_is_deployed_beside_pbo_without_missions(tmp_path: Path) -> None:
    output_dir = tmp_path / "build"
    runtime_root = output_dir / "runtime"
    source_addons = runtime_root / "Addons"
    source_addons.mkdir(parents=True)

    pbo_path = source_addons / "wg_test.pbo"
    pbo_path.write_bytes(b"pbo")

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
    local_readme = source_addons / "Test Terrain ReadMe.txt"
    deployed_readme = deployed_addons / "Test Terrain ReadMe.txt"
    assert (deployed_addons / "wg_test.pbo").read_bytes() == b"pbo"
    assert local_readme.is_file()
    assert deployed_readme.is_file()
    assert "PBO: wg_test.pbo" in deployed_readme.read_text(encoding="utf-8")
    assert {Path(item["destination"]).name for item in report["files"]} == {
        "wg_test.pbo",
        "Test Terrain ReadMe.txt",
    }
