"""Regression coverage for automatically sized wall, hedge and ditch P3D donors."""
from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from cwr_worldgen.cli import _parser
from cwr_worldgen.gui import (
    build_milestone9_command, default_gui_values, defaults_with_recent_source,
    load_gui_state, save_gui_state,
)
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection, OsmDataset, OsmLineFeature, OsmRaster,
    configured_linear_donor, generate_world_objects,
)
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.procedural_buildings import _Face, _Lod, _MLOD_HEADER, _write_lod


def _model(*, x_length: float, z_length: float, base_y: float = -0.5) -> bytes:
    xs = (-x_length * 0.5, x_length * 0.5)
    zs = (-z_length * 0.5, z_length * 0.5)
    vertices = (
        (xs[0], base_y, zs[0]),
        (xs[0], base_y + 2.0, zs[1]),
        (xs[1], base_y + 2.0, zs[1]),
        (xs[1], base_y, zs[0]),
    )
    lod = _Lod(
        vertices, ((0.0, 1.0, 0.0),),
        (_Face(r"test\asset.paa", (
            (0, 0, 0.0, 0.0), (1, 0, 0.0, 1.0),
            (2, 0, 1.0, 1.0), (3, 0, 1.0, 0.0),
        ), 0),),
        1.0, properties=(("autocenter", "0"),), point_flags=(0,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _environment():
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    def line(name, tags, z):
        return OsmLineFeature(
            name, tags,
            tuple(projection.to_latlon(p) for p in ((20.0, z), (180.0, z))),
        )
    dataset = OsmDataset(
        source_generator="linear-donor-tests", element_count=3,
        coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        barriers=(
            line("way/wall", {"barrier": "wall"}, 35.0),
            line("way/hedge", {"barrier": "hedge"}, 90.0),
        ),
        watercourses=(
            line("way/ditch", {"waterway": "ditch"}, 155.0),
        ),
    )
    raster = OsmRaster(
        cells=8, water=(False,) * 64, forest=(False,) * 64,
        farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
        buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
    )
    spec = _Milestone9PlayabilitySpec(
        name="linear_donor_test", heightmap_path=Path("unused.png"),
        bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
        max_road_objects=0, max_buildings=0, max_forest_objects=0,
        barriers_enabled=True, maximum_barrier_objects=1000,
        ditch_grass_enabled=True, maximum_ditch_grass_objects=1000,
        forest_undergrowth_enabled=False, forest_border_enabled=False,
        rural_vegetation_enabled=False, bridges_enabled=False, strict_assets=False,
        wall_donor_model=r"addon\stone_wall.p3d",
        wall_donor_length=5.0, wall_donor_heading_offset=90.0,
        hedge_donor_model=r"addon\green_hedge.p3d",
        hedge_donor_length=4.0, hedge_donor_heading_offset=0.0,
        ditch_vegetation_donor_model=r"addon\ditch_reed.p3d",
        ditch_vegetation_donor_length=7.0,
        ditch_vegetation_donor_heading_offset=90.0,
    )
    return projection, dataset, raster, spec


def test_mapped_walls_hedges_and_ditches_use_separate_selected_donors():
    projection, dataset, raster, spec = _environment()
    result = generate_world_objects(
        dataset, projection, raster, (2.0,) * 64, spec, include_roads=False,
    )
    assert result.wall_objects > 0
    assert result.hedge_objects > 0
    assert result.ditch_grass_objects > 0
    paths = [obj.model_path for obj in result.objects]
    for kind in ("wall_donor", "hedge_donor", "ditch_vegetation_donor"):
        assert getattr(spec, f"{kind}_model") in paths
    assert all(
        o.model_path in (
            spec.wall_donor_model, spec.hedge_donor_model,
            spec.ditch_vegetation_donor_model,
        ) for o in result.objects
    )
    again = generate_world_objects(
        dataset, projection, raster, (2.0,) * 64, spec, include_roads=False,
    )
    assert again.objects == result.objects


def test_blank_donors_preserve_existing_stock_and_procedural_behavior():
    projection, dataset, raster, spec = _environment()
    blank = replace(
        spec,
        wall_donor_model="", wall_donor_length=None, wall_donor_heading_offset=None,
        hedge_donor_model="", hedge_donor_length=None, hedge_donor_heading_offset=None,
        ditch_vegetation_donor_model="", ditch_vegetation_donor_length=None,
        ditch_vegetation_donor_heading_offset=None,
    )
    result = generate_world_objects(
        dataset, projection, raster, (2.0,) * 64, blank, include_roads=False,
    )
    paths = {obj.model_path for obj in result.objects}
    assert spec.wall_donor_model not in paths
    assert spec.hedge_donor_model not in paths
    assert spec.ditch_vegetation_donor_model not in paths
    assert result.wall_objects > 0
    assert result.hedge_objects > 0


def test_auto_inspects_all_three_pbo_donors_and_origin_offsets(tmp_path: Path):
    pbo = tmp_path / "addon.pbo"
    write_pbo(pbo, (
        PboEntry("stone_wall.p3d", _model(x_length=5.5, z_length=0.3)),
        PboEntry("green_hedge.p3d", _model(x_length=0.8, z_length=4.75)),
        PboEntry("ditch_reed.p3d", _model(x_length=1.5, z_length=1.5)),
    ))
    kwargs = {
        "asset_roots": (pbo,), "cache_dir": None,
        "cache_enabled": False, "cache_refresh": False,
    }
    for kind, file in (
        ("wall_donor", "stone_wall"),
        ("hedge_donor", "green_hedge"),
        ("ditch_vegetation_donor", "ditch_reed"),
    ):
        kwargs[f"{kind}_model"] = f"addon\\{file}.p3d"
    spec = SimpleNamespace(**kwargs)

    wall = configured_linear_donor(spec, "wall_donor")
    hedge = configured_linear_donor(spec, "hedge_donor")
    ditch = configured_linear_donor(spec, "ditch_vegetation_donor")
    assert wall[1] == pytest.approx(5.5)
    assert wall[2] == 90.0
    assert wall[4] == pytest.approx(-0.5)
    assert hedge[1] == pytest.approx(4.75)
    assert hedge[2] == 0.0
    # A near-square grass clump still gets a useful automatic extent.
    assert ditch[1] == pytest.approx(1.5)
    assert ditch[2] == 0.0

    spec.wall_donor_heading_offset = 180.0
    assert configured_linear_donor(spec, "wall_donor")[2] == 180.0


def test_invalid_and_missing_model_measurements_fail_clearly():
    with pytest.raises(ValueError, match="Asset roots"):
        configured_linear_donor(
            SimpleNamespace(wall_donor_model=r"addon\wall.p3d"), "wall_donor",
        )
    with pytest.raises(ValueError, match="positive and finite"):
        configured_linear_donor(
            SimpleNamespace(hedge_donor_model=r"addon\hedge.p3d",
                            hedge_donor_length=0, hedge_donor_heading_offset=90),
            "hedge_donor",
        )
    with pytest.raises(ValueError, match="unknown linear donor"):
        configured_linear_donor(SimpleNamespace(), "incorrect_kind")


def test_gui_and_cli_accept_and_restore_auto_donor_values(tmp_path: Path):
    defaults = default_gui_values()
    for kind in ("wall_donor", "hedge_donor", "ditch_vegetation_donor"):
        assert defaults[f"{kind}_model"] == ""
        assert defaults[f"{kind}_length"] == ""
        assert defaults[f"{kind}_heading_offset"] == ""
    selected = {
        "wall_donor_model": r"addon\wall.p3d",
        "hedge_donor_model": r"addon\hedge.p3d",
        "ditch_vegetation_donor_model": r"addon\reed.p3d",
        "hedge_donor_length": "5.5",
        "wall_donor_heading_offset": "0",
    }
    state_path = tmp_path / "gui-state.json"
    save_gui_state(state_path, selected)
    restored = defaults_with_recent_source(default_gui_values(), load_gui_state(state_path))
    command = build_milestone9_command(restored, python="python")
    for key, value in selected.items():
        assert restored[key] == value
        option = "--" + key.replace("_", "-")
        assert command[command.index(option) + 1] == value
    assert "--wall-donor-length" not in command
    assert "--ditch-vegetation-donor-length" not in command

    args = _parser().parse_args([
        "milestone9", "--source-dir", "data", "--output", "build",
        "--wall-donor-model", selected["wall_donor_model"],
        "--hedge-donor-model", selected["hedge_donor_model"],
        "--ditch-vegetation-donor-model", selected["ditch_vegetation_donor_model"],
    ])
    assert args.wall_donor_length is None
    assert args.hedge_donor_heading_offset is None
    assert args.ditch_vegetation_donor_length is None
