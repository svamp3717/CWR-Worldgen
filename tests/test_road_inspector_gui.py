from pathlib import Path

import pytest

from cwr_worldgen.road_inspector_gui import (
    default_output_dir,
    discover_roads_geojson,
    positive_float,
)


def test_default_output_dir_matches_drag_and_drop_launcher(tmp_path: Path) -> None:
    world = tmp_path / "wg_lundby.pbo"
    assert default_output_dir(world) == tmp_path / "wg_lundby-road-inspector"


def test_discover_roads_geojson_beside_world(tmp_path: Path) -> None:
    world = tmp_path / "wg_lundby.wrp"
    roads = tmp_path / "normalized" / "roads.geojson"
    roads.parent.mkdir()
    roads.write_text("{}", encoding="utf-8")

    assert discover_roads_geojson(world) == roads.resolve()


def test_discover_roads_geojson_one_level_above_world(tmp_path: Path) -> None:
    world = tmp_path / "build" / "wg_lundby.pbo"
    world.parent.mkdir()
    roads = tmp_path / "normalized" / "roads.geojson"
    roads.parent.mkdir()
    roads.write_text("{}", encoding="utf-8")

    assert discover_roads_geojson(world) == roads.resolve()


def test_positive_float_accepts_valid_threshold() -> None:
    assert positive_float("0.75", "Tolerance") == 0.75


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_positive_float_rejects_invalid_threshold(value: str) -> None:
    with pytest.raises(ValueError):
        positive_float(value, "Tolerance")
