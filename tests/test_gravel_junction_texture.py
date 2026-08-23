from __future__ import annotations

import json
from pathlib import Path

from cwr_worldgen.procedural_buildings import inspect_mlod
from cwr_worldgen.procedural_infrastructure import (
    ProceduralInfrastructureLibrary,
    create_gravel_junction_texture_image,
)


def test_generated_gravel_junction_uses_opaque_texture_without_internal_grass_seams(
    tmp_path: Path,
) -> None:
    image = create_gravel_junction_texture_image(128)
    assert image.mode == "RGB"

    library = ProceduralInfrastructureLibrary("cwr_junction_opaque")
    library.register_models((
        r"cwr_junction_opaque\i\gravel6.p3d",
        r"cwr_junction_opaque\i\gravel_j3.p3d",
    ))
    assets = library.write_assets(tmp_path, tmp_path / "infrastructure.json")

    assert "i/g.paa" in assets.texture_files
    assert "i/gj.paa" in assets.texture_files

    straight = inspect_mlod(tmp_path / "i" / "gravel6.p3d")
    junction = inspect_mlod(tmp_path / "i" / "gravel_j3.p3d")
    assert r"cwr_junction_opaque\i\g.paa" in straight.texture_paths
    assert r"cwr_junction_opaque\i\gj.paa" in junction.texture_paths

    catalogue = json.loads((tmp_path / "infrastructure.json").read_text(encoding="utf-8"))
    source = catalogue["gravel_texture_source"]
    assert source["junction_texture"] == "i/gj.paa"
    assert source["junction_texture_alpha"] == "opaque"
