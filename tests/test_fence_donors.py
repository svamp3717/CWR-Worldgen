"""Regression checks for optional direct-placement fence P3D donors."""
from __future__ import annotations

from pathlib import Path
import io
import struct
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
from cwr_worldgen.fence_donor_geometry import inspect_fence_donor_geometry
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.procedural_buildings import _Face, _Lod, _MLOD_HEADER, _write_lod
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPolygonFeature,
    OsmRaster,
    configured_fence_donor,
    generate_world_objects,
    _line_chunks,
)
from cwr_worldgen.fence_openings import fence_opening_chunk_indices



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


def test_mapped_fence_openings_remove_entire_custom_donor_panels() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    way = OsmLineFeature(
        "way/traversable", {"barrier": "fence", "material": "wood"},
        tuple(projection.to_latlon(point) for point in ((10.0, 100.0), (190.0, 100.0))),
    )
    projection, dataset, raster, spec = _scene(barriers=(way,))
    result = generate_world_objects(
        dataset, projection, raster, (2.0,) * 64, spec, include_roads=False,
    )
    all_chunks = _line_chunks(((10.0, 100.0), (190.0, 100.0)), 4.0)
    indices = fence_opening_chunk_indices(
        all_chunks, f"{spec.deterministic_seed}:mapped-fence-opening:way/traversable",
    )
    expected_x = [chunk[0] for index, chunk in enumerate(all_chunks) if index not in indices]
    placed_x = [o.x for o in result.objects if o.model_path == spec.fence_donor_model]
    assert indices, "the 180 m perimeter needs a walking entrance"
    assert placed_x == pytest.approx(expected_x)
    assert result.fence_objects == len(placed_x)
    assert len(placed_x) == len(all_chunks) - len(indices)


def test_farmland_polygon_has_seeded_gaps_without_overlapping_panels() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    corners = (
        (20.0, 20.0), (180.0, 20.0), (180.0, 180.0),
        (20.0, 180.0), (20.0, 20.0),
    )
    farmland = OsmPolygonFeature(
        "way/open-field", {"landuse": "farmland"},
        (GeoPolygon(tuple(projection.to_latlon(pt) for pt in corners)),),
    )
    projection, dataset, raster, spec = _scene(farmland=(farmland,))
    with patch(
        "cwr_worldgen.osm._selected_farmland_fence_field_keys",
        return_value={"way/open-field"},
    ):
        result = generate_world_objects(
            dataset, projection, raster, (2.0,) * 64, spec, include_roads=False,
        )
    chunks = _line_chunks(corners, spec.fence_donor_length)
    indices = fence_opening_chunk_indices(
        chunks, f"{spec.deterministic_seed}:farmland-fence-opening:way/open-field:0",
    )
    fences = [o for o in result.objects if o.model_path == spec.fence_donor_model]
    assert indices and len(indices) >= 2
    assert len(fences) == len(chunks) - len(indices)
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


def _fence_mlod(*, long_axis: str, length: float, width: float = 0.22) -> bytes:
    half_length, half_width = length / 2, width / 2
    if long_axis == "X":
        points = (
            (-half_length, 0.0, -half_width),
            (-half_length, 2.0, -half_width),
            (half_length, 2.0, half_width),
            (half_length, 0.0, half_width),
        )
    else:
        points = (
            (-half_width, 0.0, -half_length),
            (-half_width, 2.0, half_length),
            (half_width, 2.0, half_length),
            (half_width, 0.0, -half_length),
        )
    lod = _Lod(
        points, ((0.0, 1.0, 0.0),),
        (_Face(
            r"addon\fence.paa",
            ((0, 0, 0.0, 0.0), (1, 0, 0.0, 1.0),
             (2, 0, 1.0, 1.0), (3, 0, 1.0, 0.0)),
            0,
        ),),
        1.0, properties=(("autocenter", "0"),), point_flags=(0,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _fence_odol_v7() -> bytes:
    points = ((-2.5, 0.0, -0.2), (2.5, 0.0, -0.2),
              (2.5, 2.0, 0.2), (-2.5, 2.0, 0.2))
    stream = io.BytesIO()
    stream.write(b"ODOL")
    stream.write(struct.pack("<II", 7, 1))
    stream.write(struct.pack("<I", len(points)))
    stream.write(b"\0" * (len(points) * 4))  # point flags
    stream.write(struct.pack("<I", len(points)))
    stream.write(b"\0" * (len(points) * 8))  # UVs
    stream.write(struct.pack("<I", len(points)))
    for xyz in points:
        stream.write(struct.pack("<fff", *xyz))
    stream.write(struct.pack("<I", len(points)))
    stream.write(b"\0" * (len(points) * 12))  # normals
    stream.write(b"\0" * 48)
    stream.write(struct.pack("<I", 0))  # textures
    stream.write(struct.pack("<I", 0))  # MLOD edge indices
    stream.write(struct.pack("<I", 0))  # vertex edge indices
    stream.write(struct.pack("<II", 0, 0))  # faces, section offset
    return stream.getvalue()


@pytest.mark.parametrize("axis,length,angle", [
    ("X", 4.25, 90.0),
    ("Z", 7.5, 0.0),
])
def test_auto_detects_fence_length_and_axis_from_mlod(
    axis: str, length: float, angle: float,
) -> None:
    measurement = inspect_fence_donor_geometry(_fence_mlod(long_axis=axis, length=length))
    assert measurement.source_format == "MLOD"
    assert measurement.long_axis == axis
    assert measurement.length_metres == pytest.approx(length)
    assert measurement.heading_offset_degrees == angle


def test_auto_detects_odol_v7_fence_axis() -> None:
    measurement = inspect_fence_donor_geometry(_fence_odol_v7())
    assert measurement.source_format == "ODOL"
    assert measurement.length_metres == pytest.approx(5.0)
    assert measurement.heading_offset_degrees == 90.0


def test_pbo_fence_donor_automatically_supplies_length_and_rotation(tmp_path: Path) -> None:
    pbo = tmp_path / "myfences.pbo"
    write_pbo(pbo, (
        PboEntry("post_rail.p3d", _fence_mlod(long_axis="Z", length=5.5)),
        PboEntry("chain_mesh.p3d", _fence_mlod(long_axis="X", length=3.2)),
    ))
    spec = SimpleNamespace(
        fence_donor_model=r"myfences\post_rail.p3d",
        fence_donor_length=None, fence_donor_heading_offset=None,
        metal_fence_donor_model=r"myfences\chain_mesh.p3d",
        metal_fence_donor_length=None, metal_fence_donor_heading_offset=None,
        asset_roots=(pbo,), cache_dir=None, cache_enabled=False,
        cache_refresh=False,
    )
    assert configured_fence_donor(spec) == (spec.fence_donor_model, 5.5, 0.0)
    measured_metal = configured_fence_donor(spec, metal=True)
    assert measured_metal[0] == spec.metal_fence_donor_model
    assert measured_metal[1] == pytest.approx(3.2)
    assert measured_metal[2] == 90.0
    spec.fence_donor_length = 6.0
    assert configured_fence_donor(spec) == (spec.fence_donor_model, 6.0, 0.0)


def test_auto_geometry_requires_readable_asset_and_unambiguous_axis() -> None:
    with pytest.raises(ValueError, match="Asset roots"):
        configured_fence_donor(SimpleNamespace(fence_donor_model=r"addon\fence.p3d"))
    with pytest.raises(ValueError, match="ambiguous fence axis"):
        inspect_fence_donor_geometry(_fence_mlod(long_axis="X", length=3, width=3))
    with pytest.raises(ValueError, match="unsupported fence P3D format"):
        inspect_fence_donor_geometry(b"aaaa")


def test_empty_gui_fence_dimensions_use_auto_detection_flags(tmp_path: Path) -> None:
    defaults = default_gui_values()
    assert defaults["fence_donor_length"] == ""
    assert defaults["fence_donor_heading_offset"] == ""
    assert defaults["metal_fence_donor_length"] == ""
    assert defaults["metal_fence_donor_heading_offset"] == ""
    defaults["fence_donor_model"] = r"addon\segment.p3d"
    command = build_milestone9_command(defaults, python="python")
    assert "--fence-donor-model" in command
    assert "--fence-donor-length" not in command
    assert "--fence-donor-heading-offset" not in command
    args = _parser().parse_args([
        "milestone9", "--output", str(tmp_path / "world"),
        "--source-dir", str(tmp_path / "sources"),
        "--fence-donor-model", r"addon\segment.p3d",
    ])
    assert args.fence_donor_length is None
    assert args.fence_donor_heading_offset is None
