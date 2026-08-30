from __future__ import annotations

import math
from pathlib import Path

from cwr_worldgen.generator import _verify_single_world_pbo_layout
from cwr_worldgen.paa import inspect_paa
from cwr_worldgen.pbo import pack_directory
from cwr_worldgen.procedural_buildings import inspect_mlod
from cwr_worldgen.procedural_infrastructure import (
    ProceduralInfrastructureLibrary,
    create_paved_fill_texture_image,
    is_generated_paved_fill_model,
    paved_fill_model_path,
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


def test_paved_fill_runtime_assets_are_verified_inside_world_pbo(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.cpp").write_text("class CfgPatches {};\n", encoding="ascii")
    (source / "cwr_paved_bundle.wrp").write_bytes(b"4WVR")
    library = ProceduralInfrastructureLibrary("cwr_paved_bundle")
    library.register_model(paved_fill_model_path("cwr_paved_bundle"))
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

    assert layout["generated_road_models"] == [r"i\paved_fill.p3d"]
    assert layout["generated_road_textures"] == [r"i\pf.paa"]
