"""Regression checks for optional direct-placement fence P3D donors."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cwr_worldgen.cli import _parser
from cwr_worldgen.gui import (
    build_milestone9_command,
    default_gui_values,
    defaults_with_recent_source,
    load_gui_state,
    save_gui_state,
)
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPolygonFeature,
    OsmRaster,
    configured_fence_donor,
    generate_world_objects,
)


def _scene(*, barriers=(), farmland=()):
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    dataset = OsmDataset(
        source_generator="donor-fences", element_count=len(barriers) + len(farmland),
        coastlines=(), water=(), forests=(), farmland=farmland,
        urban=(), roads=(), barriers=barriers,
    )
    raster = OsmRaster(
        cells=8, water=(False,) * 64, forest=(False,) * 64,
        farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
        buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
    )
    spec = _Milestone9PlayabilitySpec(
        name="donor_fences", heightmap_path=Path("unused.png"),
        bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
        max_road_objects=0, max_buildings=0, max_forest_objects=0,
        barriers_enabled=True, maximum_barrier_objects=1000,
        forest_undergrowth_enabled=False, forest_border_enabled=False,
        ditch_grass_enabled=False, rural_vegetation_enabled=False,
        bridges_enabled=False, strict_assets=False,
        fence_donor_model=r"sfp\wood\rail.p3d",
        fence_donor_length=4.0,
        fence_donor_heading_offset=0.0,
        metal_fence_donor_model=r"sfp\steel\mesh.p3d",
        metal_fence_donor_length=8.0,
        metal_fence_donor_heading_offset=180.0,
    )
    return projection, dataset, raster, spec


def test_configured_fence_donors_keep_stock_fallback_and_normalize_paths() -> None:
    assert configured_fence_donor(SimpleNamespace()) is None
    assert configured_fence_donor(SimpleNamespace(), metal=True) is None
    spec = SimpleNamespace(
        fence_donor_model=r"SFP/wood/rail.p3d",
        fence_donor_length=4,
        fence_donor_heading_offset=0,
    )
    assert configured_fence_donor(spec) == (r"SFP\wood\rail.p3d", 4.0, 0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        configured_fence_donor(
            SimpleNamespace(fence_donor_model=r"addon\fence.p3d", fence_donor_length=0)
        )


def test_mapped_regular_and_metal_fences_use_distinct_donors_and_lengths() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    nonmetal = OsmLineFeature(
        "way/fence-wood", {"barrier": "fence", "material": "wood"},
        tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
    )
    metal = OsmLineFeature(
        "way/fence-metal", {"barrier": "fence", "fence_type": "chain_link"},
        tuple(projection.to_latlon(p) for p in ((20.0, 150.0), (180.0, 150.0))),
    )
    _, dataset, raster, spec = _scene(barriers=(nonmetal, metal))
    result = generate_world_objects(
        dataset, projection, raster, (2.0,) * 64, spec, include_roads=False,
    )
    ordinary = [o for o in result.objects if o.model_path == spec.fence_donor_model]
    chainlink = [o for o in result.objects if o.model_path == spec.metal_fence_donor_model]
    assert ordinary and chainlink
    assert len(ordinary) > len(chainlink), "4 m fence segments should be denser than 8 m ones"
    assert result.fence_objects == len(ordinary) + len(chainlink)
    # Each family remains consistent along its mapped line and is repeatable.
    repeat = generate_world_objects(
        dataset, projection, raster, (2.0,) * 64, spec, include_roads=False,
    )
    assert result.objects == repeat.objects


def test_farmland_boundaries_use_ordinary_donor_instead_of_stock() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    boundary = tuple(projection.to_latlon(point) for point in (
        (20.0, 20.0), (180.0, 20.0), (180.0, 180.0),
        (20.0, 180.0), (20.0, 20.0),
    ))
    farmland = OsmPolygonFeature(
        "way/field", {"landuse": "farmland"}, (GeoPolygon(boundary),),
    )
    projection, dataset, raster, spec = _scene(farmland=(farmland,))
    with patch(
        "cwr_worldgen.osm._selected_farmland_fence_field_keys",
        return_value={"way/field"},
    ):
        result = generate_world_objects(
            dataset, projection, raster, (2.0,) * 64, spec, include_roads=False,
        )
    fences = [o for o in result.objects if o.model_path == spec.fence_donor_model]
    assert fences
    assert result.fence_objects == len(fences)


def test_fence_donor_gui_and_cli_settings_survive_restart(tmp_path: Path) -> None:
    base = default_gui_values()
    assert base["fence_donor_model"] == ""
    assert base["metal_fence_donor_model"] == ""
    command = build_milestone9_command(base, python="python")
    assert "--fence-donor-model" not in command
    assert "--metal-fence-donor-model" not in command

    state_path = tmp_path / "gui-state.json"
    selected = {
        "fence_donor_model": r"sfp\wood\rail.p3d",
        "fence_donor_length": "4.0",
        "fence_donor_heading_offset": "0",
        "metal_fence_donor_model": r"sfp\steel\mesh.p3d",
        "metal_fence_donor_length": "8.0",
        "metal_fence_donor_heading_offset": "180",
    }
    save_gui_state(state_path, selected)
    restored = defaults_with_recent_source(default_gui_values(), load_gui_state(state_path))
    for key, value in selected.items():
        assert restored[key] == value
    command = build_milestone9_command(restored, python="python")
    for key, option in (
        ("fence_donor_model", "--fence-donor-model"),
        ("fence_donor_length", "--fence-donor-length"),
        ("fence_donor_heading_offset", "--fence-donor-heading-offset"),
        ("metal_fence_donor_model", "--metal-fence-donor-model"),
        ("metal_fence_donor_length", "--metal-fence-donor-length"),
        ("metal_fence_donor_heading_offset", "--metal-fence-donor-heading-offset"),
    ):
        assert command[command.index(option) + 1] == selected[key]

    args = _parser().parse_args([
        "milestone9", "--output", str(tmp_path / "world"),
        "--source-dir", str(tmp_path / "sources"),
        *[item for key, value in (
            ("--fence-donor-model", selected["fence_donor_model"]),
            ("--fence-donor-length", selected["fence_donor_length"]),
            ("--metal-fence-donor-model", selected["metal_fence_donor_model"]),
            ("--metal-fence-donor-heading-offset", selected["metal_fence_donor_heading_offset"]),
        ) for item in (key, value)],
    ])
    assert args.fence_donor_model == selected["fence_donor_model"]
    assert args.fence_donor_length == 4.0
    assert args.metal_fence_donor_model == selected["metal_fence_donor_model"]
    assert args.metal_fence_donor_heading_offset == 180.0
