from __future__ import annotations

import math
from pathlib import Path

from cwr_worldgen.generator import _verify_single_world_pbo_layout
from cwr_worldgen.paa import inspect_paa
from cwr_worldgen.pbo import pack_directory
from cwr_worldgen.procedural_buildings import inspect_mlod
from cwr_worldgen.procedural_infrastructure import (
    GENERATED_PAVED_FILL_RADIUS_METRES,
    GENERATED_PAVED_MITER_SAFETY_METRES,
    ProceduralInfrastructureLibrary,
    _paved_miter_lods,
    create_paved_fill_texture_image,
    is_generated_paved_fill_model,
    paved_fill_model_path,
    paved_miter_angle_degrees,
    paved_miter_model_path,
    paved_wedge_angle_degrees,
    paved_wedge_local_points,
    paved_wedge_model_path,
)


def test_generated_paved_fill_is_opaque_borderless_roadway(tmp_path: Path) -> None:
    image = create_paved_fill_texture_image(64)
    assert image.mode == "RGB"
    assert image.getextrema() != ((58, 58), (58, 58), (56, 56))

    model_path = paved_fill_model_path("cwr_paved_fill")
    assert is_generated_paved_fill_model(model_path)
    library = ProceduralInfrastructureLibrary("cwr_paved_fill")
    library.register_model(model_path)
    assets = library.write_assets(tmp_path, tmp_path / "infrastructure.json")

    assert assets.model_files == ("i/paved_fill.p3d",)
    assert "i/pf.paa" in assets.texture_files
    summary = inspect_mlod(tmp_path / "i" / "paved_fill.p3d")
    assert r"cwr_paved_fill\i\pf.paa" in summary.texture_paths
    assert any(
        math.isclose(value, 3.0e15, rel_tol=1.0e-6)
        for value in summary.resolutions
    )
    paa = inspect_paa(tmp_path / "i" / "pf.paa")
    assert (paa.width, paa.height) == (256, 256)


def test_paved_miter_reaches_quantized_outer_edge_apex(tmp_path: Path) -> None:
    model_path = paved_miter_model_path("cwr_paved_miter", 6.01)
    assert model_path.endswith(r"\paved_miter_q025.p3d")
    turn = paved_miter_angle_degrees(model_path)
    assert turn == 6.25
    lods = _paved_miter_lods(r"cwr_paved_miter\i\pf.paa", turn)
    expected_apex = (
        GENERATED_PAVED_FILL_RADIUS_METRES + GENERATED_PAVED_MITER_SAFETY_METRES
    ) / math.cos(math.radians(turn * 0.5))
    assert math.isclose(
        max(abs(point[0]) for point in lods[0].points),
        expected_apex,
        abs_tol=1.0e-9,
    )

    library = ProceduralInfrastructureLibrary("cwr_paved_miter")
    library.register_model(model_path)
    assets = library.write_assets(tmp_path, tmp_path / "infrastructure.json")
    assert assets.model_files == ("i/paved_miter_q025.p3d",)
    assert assets.texture_files == ("i/pf.paa",)


def test_paved_wedge_is_a_narrow_borderless_outer_triangle(tmp_path: Path) -> None:
    model_path = paved_wedge_model_path("cwr_paved_wedge", 12.01)
    assert model_path.endswith(r"\paved_wedge_q049.p3d")
    turn = paved_wedge_angle_degrees(model_path)
    assert turn == 12.25
    points = paved_wedge_local_points(turn)
    assert len(points) == 3
    assert points[0][2] > 0.0
    assert points[1][2] == points[2][2] == 0.0
    assert math.isclose(points[1][0], -points[2][0], abs_tol=1.0e-12)

    library = ProceduralInfrastructureLibrary("cwr_paved_wedge")
    library.register_model(model_path)
    assets = library.write_assets(tmp_path, tmp_path / "infrastructure.json")
    assert assets.model_files == ("i/paved_wedge_q049.p3d",)
    assert assets.texture_files == ("i/pf.paa",)
    summary = inspect_mlod(tmp_path / "i" / "paved_wedge_q049.p3d")
    assert r"cwr_paved_wedge\i\pf.paa" in summary.texture_paths


def test_paved_fill_runtime_assets_are_verified_inside_world_pbo(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.cpp").write_text("class CfgPatches {};\n", encoding="ascii")
    (source / "cwr_paved_bundle.wrp").write_bytes(b"4WVR")
    library = ProceduralInfrastructureLibrary("cwr_paved_bundle")
    library.register_model(paved_fill_model_path("cwr_paved_bundle"))
    library.register_model(paved_miter_model_path("cwr_paved_bundle", 6.0))
    library.register_model(paved_wedge_model_path("cwr_paved_bundle", 6.0))
    generation = library.write_assets(
        source,
        tmp_path / "infrastructure-asset-catalogue.json",
    )
    pbo = tmp_path / "cwr_paved_bundle.pbo"
    pack_directory(source, pbo)

    layout = _verify_single_world_pbo_layout(
        pbo,
        "cwr_paved_bundle",
        generation,
    )

    assert layout["generated_road_models"] == [
        r"i\paved_fill.p3d",
        r"i\paved_miter_q024.p3d",
        r"i\paved_wedge_q024.p3d",
    ]
    assert layout["generated_road_textures"] == [r"i\pf.paa"]
