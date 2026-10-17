"""Regression tests for opt-in drivable bridge module P3D donors."""
from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cwr_worldgen import bridge_render_policy, osm
from cwr_worldgen.bridge_donors import configured_bridge_donor
from cwr_worldgen.cli import _parser
from cwr_worldgen.gui import (
    build_milestone9_command, default_gui_values, defaults_with_recent_source,
    load_gui_state, save_gui_state,
)
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.procedural_buildings import _Face, _Lod, _MLOD_HEADER, _write_lod


def _lod(*, width: float, length: float, roadway_y: float, resolution: float) -> _Lod:
    points = (
        (-width * 0.5, roadway_y, -length * 0.5),
        (-width * 0.5, roadway_y, length * 0.5),
        (width * 0.5, roadway_y, length * 0.5),
        (width * 0.5, roadway_y, -length * 0.5),
    )
    return _Lod(
        points, ((0.0, -1.0, 0.0),),
        (_Face(r"bridge\deck.paa", (
            (0, 0, 0.0, 0.0), (1, 0, 0.0, 1.0),
            (2, 0, 1.0, 1.0), (3, 0, 1.0, 0.0),
        ), 0),),
        resolution, properties=(("autocenter", "0"),), point_flags=(0,) * 4,
    )


def _bridge_mlod(*, length: float = 24.0, width: float = 8.0,
                 deck_y: float = 1.5, has_roadway: bool = True,
                 warped: bool = False) -> bytes:
    visual = _lod(width=width, length=length, roadway_y=deck_y + 2.0, resolution=1.0)
    roadway = _lod(width=width - 0.5, length=length,
                   roadway_y=deck_y, resolution=3.0e15)
    if warped:
        points = list(roadway.points)
        x, y, z = points[-1]
        points[-1] = (x, y + 2.0, z)
        roadway = replace(roadway, points=tuple(points))
    lods = (visual, roadway) if has_roadway else (visual,)
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, len(lods)))
    for lod in lods:
        _write_lod(stream, lod)
    return stream.getvalue()


def _donor_spec(pbo: Path, **changes) -> _Milestone9PlayabilitySpec:
    values = {
        "name": "donor_bridge_world",
        "heightmap_path": Path("unused.png"),
        "bbox": (0, 0, 1, 1),
        "cells": 8,
        "cell_size": 25.0,
        "max_road_objects": 0,
        "max_buildings": 0,
        "max_forest_objects": 0,
        "asset_roots": (pbo,),
        "strict_assets": False,
        "bridge_donor_model": r"addonbridge\segment.p3d",
        "bridges_enabled": True,
        "procedural_bridges": True,  # Donor wins without changing old toggle.
        "forest_undergrowth_enabled": False,
        "forest_border_enabled": False,
        "ditch_grass_enabled": False,
        "barriers_enabled": False,
        "rural_vegetation_enabled": False,
        "bridges_enabled": True,
    }
    values.update(changes)
    return _Milestone9PlayabilitySpec(**values)


def test_bridge_donor_measures_length_width_heading_and_roadway_deck(tmp_path: Path):
    archive = tmp_path / "addonbridge.pbo"
    write_pbo(archive, (PboEntry("segment.p3d", _bridge_mlod()),))
    donor = configured_bridge_donor(_donor_spec(archive))
    assert donor is not None
    assert donor.model == r"addonbridge\segment.p3d"
    assert donor.source_format == "MLOD"
    assert donor.length_metres == pytest.approx(24.0)
    assert donor.width_metres == pytest.approx(8.0)
    assert donor.heading_offset_degrees == 0.0
    assert donor.roadway_local_y_metres == pytest.approx(1.5)
    assert len(donor.content_sha256) == 64


def test_bridge_donor_can_override_length_rotation_and_deck_height(tmp_path: Path):
    archive = tmp_path / "addonbridge.pbo"
    write_pbo(archive, (PboEntry("segment.p3d", _bridge_mlod()),))
    donor = configured_bridge_donor(_donor_spec(
        archive, bridge_donor_length=27.0,
        bridge_donor_heading_offset=180.0,
        bridge_donor_deck_height=2.2,
    ))
    assert donor.length_metres == 27.0
    assert donor.heading_offset_degrees == 180.0
    assert donor.roadway_local_y_metres == 2.2


@pytest.mark.parametrize("payload,error", [
    (_bridge_mlod(has_roadway=False), "no Roadway LOD"),
    (_bridge_mlod(warped=True), "not level"),
    (b"not-a-P3D", "unsupported fence P3D format"),
])
def test_bridge_donor_rejects_undrivable_or_unreadable_models(
    tmp_path: Path, payload: bytes, error: str,
):
    archive = tmp_path / "addonbridge.pbo"
    write_pbo(archive, (PboEntry("segment.p3d", payload),))
    with pytest.raises(ValueError, match=error):
        configured_bridge_donor(_donor_spec(archive))


def test_bridge_donor_without_asset_roots_fails_even_with_manual_overrides():
    spec = SimpleNamespace(
        bridge_donor_model=r"addonbridge\segment.p3d",
        bridge_donor_length=25,
        bridge_donor_heading_offset=0,
        bridge_donor_deck_height=2,
        asset_roots=(),
    )
    with pytest.raises(ValueError, match="Asset roots"):
        configured_bridge_donor(spec)


def test_nogova_seam_policy_does_not_replace_selected_custom_bridge():
    spec = SimpleNamespace(
        bridge_donor_model=r"addonbridge\segment.p3d",
        bridge_module_length=12.0,
        procedural_bridges=True,
    )
    adapted = bridge_render_policy._stock_bridge_spec(spec)
    assert adapted.procedural_bridges is False
    assert adapted.bridge_module_length == 12.0
    with patch.object(
        bridge_render_policy, "_ORIGINAL_EXTEND_BRIDGE_SPAN",
        return_value=((20.0, 20.0), (80.0, 20.0)),
    ) as original:
        span = bridge_render_policy._extend_stock_bridge_to_wet_span(
            ((20.0, 20.0), (80.0, 20.0)), [0.0], adapted, 24.0,
        )
    assert span == ((20.0, 20.0), (80.0, 20.0))
    original.assert_called_once()


def test_bridge_donor_generates_real_model_modules_and_aligns_deck(tmp_path: Path):
    archive = tmp_path / "addonbridge.pbo"
    write_pbo(archive, (PboEntry("segment.p3d", _bridge_mlod(length=20.0)),))
    spec = _donor_spec(archive)
    projection = osm.BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    bridge = osm.OsmLineFeature(
        "way/donor-bridge", {"highway": "secondary", "bridge": "yes"},
        tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
    )
    dataset = osm.OsmDataset(
        source_generator="donor-bridge", element_count=1,
        coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,),
    )
    raster = osm.OsmRaster(
        cells=8, water=(False,) * 64, forest=(False,) * 64,
        farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
        buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
    )
    elevations = (5.0,) * 64
    with patch.object(osm, "_road_needs_bridge_deck", return_value=True), patch.object(
        osm, "_extend_procedural_bridge_to_approach_plateaus",
        side_effect=lambda points, *_args, **_kwargs: points,
    ):
        result = osm.generate_world_objects(
            dataset, projection, raster, elevations, spec, include_roads=False,
        )
    modules = [o for o in result.objects if o.model_path == spec.bridge_donor_model]
    assert len(modules) == 8
    assert result.bridge_objects == 8
    assert result.bridge_segments == 1
    for piece in modules:
        assert piece.y + 1.5 == pytest.approx(5.035)
        assert piece.heading_degrees == pytest.approx(90.0)
        assert piece.pitch_degrees == pytest.approx(0.0)


def test_bridge_donor_gui_cli_defaults_and_saved_overrides(tmp_path: Path):
    defaults = default_gui_values()
    assert defaults["bridge_donor_model"] == ""
    for key in ("bridge_donor_length", "bridge_donor_heading_offset", "bridge_donor_deck_height"):
        assert defaults[key] == ""
    selected = {
        "bridge_donor_model": r"addonbridge\segment.p3d",
        "bridge_donor_length": "25",
        "bridge_donor_heading_offset": "90",
        "bridge_donor_deck_height": "1.5",
    }
    state = tmp_path / "gui.json"
    save_gui_state(state, selected)
    restored = defaults_with_recent_source(default_gui_values(), load_gui_state(state))
    cmd = build_milestone9_command(restored, python="python")
    for key, value in selected.items():
        assert restored[key] == value
        option = "--" + key.replace("_", "-")
        assert cmd[cmd.index(option) + 1] == value
    args = _parser().parse_args([
        "milestone9", "--source-dir", "source", "--output", "out",
        "--bridge-donor-model", selected["bridge_donor_model"],
    ])
    assert args.bridge_donor_model == selected["bridge_donor_model"]
    assert args.bridge_donor_length is None
    assert args.bridge_donor_deck_height is None
