# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused coverage for generated urban sidewalks and their bundled assets."""
from __future__ import annotations

from pathlib import Path

from cwr_worldgen.cli import _parser
from cwr_worldgen.milestone9 import Milestone9Spec, _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    OsmDataset,
    OsmLineFeature,
    OsmPointFeature,
    OsmRaster,
    _sidewalk_sides,
    generate_world_objects,
)
from cwr_worldgen.procedural_infrastructure import ProceduralInfrastructureLibrary
from cwr_worldgen.procedural_buildings import inspect_mlod


def _generate(tags: dict[str, str], *, place: str = "city", enabled: bool = True):
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    road = OsmLineFeature(
        "way/urban-road", tags,
        tuple(projection.to_latlon(point) for point in ((20.0, 100.0), (180.0, 100.0))),
    )
    places = (
        (OsmPointFeature("node/place", {"place": place}, projection.to_latlon((100.0, 100.0))),)
        if place else ()
    )
    dataset = OsmDataset(
        source_generator="city-sidewalk-test", element_count=1 + len(places),
        coastlines=(), water=(), forests=(), farmland=(), urban=(),
        roads=(road,), places=places,
    )
    raster = OsmRaster(
        cells=8, water=(False,) * 64, forest=(False,) * 64,
        farmland=(False,) * 64, urban=(False,) * 64,
        roads=(False,) * 64, buildings=(False,) * 64,
        high_resolution=8, coastline_seed_count=0,
    )
    spec = _Milestone9PlayabilitySpec(
        name="citywalks", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
        cells=8, cell_size=25.0, include_minor_roads=True,
        max_road_objects=0, max_buildings=0, max_forest_objects=0,
        sidewalks_enabled=enabled, sidewalk_width=1.8,
        sidewalk_segment_length=5.0, maximum_sidewalk_objects=1000,
        street_furniture_enabled=False,
        rural_vegetation_enabled=False, meadow_grass_enabled=False,
        wetland_reeds_enabled=False, barriers_enabled=False,
        bridges_enabled=False, forest_undergrowth_enabled=False,
        forest_border_enabled=False, rocky_forest_fallback_enabled=False,
        steep_hill_bushes_enabled=False, strict_assets=False,
    )
    return generate_world_objects(
        dataset, projection, raster, (5.0,) * 64, spec,
        include_roads=False, building_placement_plans=(),
    )


def _slabs(result):
    return [obj for obj in result.objects if r"\i\sw_w" in obj.model_path]


def test_sidewalks_on_both_sides_of_city_road_and_deterministic() -> None:
    first = _generate({"highway": "residential"})
    second = _generate({"highway": "residential"})
    slabs = _slabs(first)
    assert first.sidewalk_objects == len(slabs) > 0
    assert any(obj.z > 100.0 for obj in slabs)
    assert any(obj.z < 100.0 for obj in slabs)
    assert slabs == _slabs(second)
    assert all("sw_w018_" in obj.model_path for obj in slabs)


def test_osm_sidewalk_tags_control_sides() -> None:
    left = _slabs(_generate({"highway": "residential", "sidewalk": "left"}))
    both = _slabs(_generate({"highway": "residential", "sidewalk": "both"}))
    assert len(left) > 0
    assert len(both) == 2 * len(left)
    assert all(obj.z > 100.0 for obj in left)
    assert not _slabs(_generate({"highway": "residential", "sidewalk": "no"}))
    assert _sidewalk_sides(
        {"sidewalk": "both", "sidewalk:right": "no"}, inferred=True
    ) == (-1,)
    assert _sidewalk_sides(
        {"sidewalk": "separate"}, inferred=True
    ) == ()


def test_rural_and_village_roads_do_not_gain_inferred_sidewalks() -> None:
    for place in ("village", "hamlet", ""):
        assert not _slabs(_generate({"highway": "residential"}, place=place))
    assert not _slabs(_generate({"highway": "residential"}, enabled=False))


def test_dirt_road_requires_explicit_sidewalk_tag() -> None:
    assert not _slabs(_generate({"highway": "residential", "surface": "dirt"}))
    assert _slabs(_generate({
        "highway": "residential", "surface": "dirt", "sidewalk": "yes"
    }))


def test_sidewalk_dimensions_are_packaged_with_collision_lod(tmp_path: Path) -> None:
    library = ProceduralInfrastructureLibrary("citywalks", cache_enabled=False)
    model = library.sidewalk_model(1.8, 4.6)
    assert model == r"citywalks\i\sw_w018_l0046.p3d"
    result = library.write_assets(tmp_path, tmp_path / "catalogue.json")
    path = tmp_path / "i" / "sw_w018_l0046.p3d"
    assert path.is_file()
    assert (tmp_path / "i" / "sw.paa").is_file()
    assert "i/sw_w018_l0046.p3d" in result.model_files
    lods = inspect_mlod(path).resolutions
    assert 1.0 in lods
    assert any(abs(value - 1.0e13) < 1.0e12 for value in lods)


def test_milestone9_sidewalks_are_on_by_default_and_can_be_disabled() -> None:
    assert Milestone9Spec(source_dir=Path("source")).sidewalks_enabled
    args = ["milestone9", "--source-dir", "source", "--output", "build/test"]
    assert _parser().parse_args(args).sidewalks_enabled
    assert not _parser().parse_args(args + ["--no-sidewalks"]).sidewalks_enabled
