from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat

from cwr_worldgen.procedural_buildings import inspect_mlod
from cwr_worldgen.procedural_infrastructure import (
    ProceduralInfrastructureLibrary,
    _GRAVEL_REFERENCE_TEXTURE,
    create_gravel_junction_texture_image,
    create_gravel_road_texture_image,
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

def test_generated_gravel_object_finish_is_darker_but_terrain_reference_stays_neutral() -> None:
    size = 128
    road = create_gravel_road_texture_image(size)
    terrain = create_gravel_road_texture_image(size, object_finish=False)
    junction = create_gravel_junction_texture_image(size)

    with Image.open(_GRAVEL_REFERENCE_TEXTURE) as source:
        reference = source.convert("RGB").resize(
            (size, size), Image.Resampling.LANCZOS
        )

    margin = size // 10
    box = (margin, margin, size - margin, size - margin)

    def mean_luma(image: Image.Image) -> float:
        return float(
            ImageStat.Stat(image.convert("RGB").crop(box).convert("L")).mean[0]
        )

    reference_luma = mean_luma(reference)
    road_ratio = mean_luma(road) / reference_luma
    junction_ratio = mean_luma(junction) / reference_luma

    # Generated road objects should stay dark enough to sit beside stock paved
    # slabs while preserving enough range that the aggregate remains readable.
    assert 0.62 <= road_ratio <= 0.82
    assert 0.66 <= junction_ratio <= 0.86

    def mean_saturation(image: Image.Image) -> float:
        return float(
            ImageStat.Stat(
                image.convert("RGB").crop(box).convert("HSV").getchannel("S")
            ).mean[0]
        )

    reference_saturation = mean_saturation(reference)
    road_saturation = mean_saturation(road)
    junction_saturation = mean_saturation(junction)
    # The object texture deliberately moves into the neutral grey family used
    # by paved roads instead of retaining the reference photograph's beige cast.
    assert road_saturation <= reference_saturation * 0.55 + 2.0
    assert junction_saturation <= reference_saturation * 0.55 + 2.0

    # The terrain surface pass opts out, so changing object gravel cannot also
    # darken every terrain cell that happens to reuse the reference photograph.
    assert terrain.convert("RGB").tobytes() == reference.tobytes()

    alpha = road.getchannel("A")
    assert alpha.getpixel((0, size // 2)) == 0
    assert alpha.getpixel((size // 2, size // 2)) == 255

