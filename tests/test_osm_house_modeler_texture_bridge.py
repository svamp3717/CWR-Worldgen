from __future__ import annotations

import base64
import json
from pathlib import Path
import random
import tempfile

from PIL import Image, ImageStat

from cwr_worldgen import osm_house_modeler_textures as upstream_textures
from cwr_worldgen.osm_house_modeler_full_style import split_texture_token
from cwr_worldgen.osm_house_modeler_texture_bridge import (
    CWA_EXTERIOR_EXPOSURE,
    UPSTREAM_TEXTURE_CANONICAL_SIZE,
    _seed,
    _wall_material_image,
    cwa_exposure_compensate,
    modeler_front_texture_image,
    modeler_interior_wall_texture_image,
    modeler_open_wall_texture_image,
    modeler_roof_texture_image,
    modeler_wall_texture_image,
)
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary


def _token(
    facade: str,
    material: str,
    palette: str,
    *,
    window_material: str = "painted timber",
    door_material: str = "timber",
) -> str:
    metadata = {
        "texture_renderer_revision": 2,
        "window": {
            "width_m": 1.2,
            "height_m": 1.35,
            "sill_height_m": 0.85,
            "target_bay_spacing_m": 3.6,
            "density_multiplier": 1.0,
            "type": "paired casement",
            "placement_style": "regular",
            "frame_material": window_material,
        },
        "door": {
            "width_m": 0.95,
            "height_m": 2.05,
            "type": "glazed panel",
            "material": door_material,
        },
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"{facade}|{material}~{encoded}|{palette}"


def _mean_luma(image) -> float:
    r, g, b = ImageStat.Stat(image.convert("RGB")).mean
    return r * 0.2126 + g * 0.7152 + b * 0.0722


def test_cwa_exposure_compensation_reduces_diffuse_brightness() -> None:
    source = Image.new("RGB", (8, 8), (220, 200, 180))
    adjusted = cwa_exposure_compensate(source)
    assert CWA_EXTERIOR_EXPOSURE < 1.0
    assert _mean_luma(adjusted) < _mean_luma(source) * 0.82


def test_modeler_wall_uses_material_palette_and_stays_below_washed_out_range() -> None:
    token = _token("western_stucco", "plaster", "cream")
    image = modeler_wall_texture_image(
        "residential", 128, token, texture_variant=2
    )
    assert 80.0 < _mean_luma(image) < 165.0
    extrema = ImageStat.Stat(image).extrema
    assert any(high - low > 30 for low, high in extrema)


def test_modeler_material_is_rendered_at_native_256_before_downsampling() -> None:
    token = _token("western_brick", "brick", "red brick")
    facade, material, palette = split_texture_token(token)
    kind, base = upstream_textures._choose_wall_base(facade, facade, material, palette)

    native_rng = random.Random(_seed(f"wall:{token}:3"))
    native_pixels = upstream_textures._render_wall(
        kind, base, native_rng, UPSTREAM_TEXTURE_CANONICAL_SIZE
    )
    native = Image.new(
        "RGB", (UPSTREAM_TEXTURE_CANONICAL_SIZE, UPSTREAM_TEXTURE_CANONICAL_SIZE)
    )
    native.putdata(native_pixels)
    expected = native.resize((128, 128), Image.Resampling.LANCZOS)

    _wall_material_image.cache_clear()
    actual = _wall_material_image(token, 3, 128)
    assert actual.tobytes() == expected.tobytes()

    # This is the bug visible in the CWA screenshot: asking the upstream private
    # renderer to draw directly at 128 changes all of its fixed pixel-space brick
    # dimensions. The bridge must never regress to that output again.
    wrong_rng = random.Random(_seed(f"wall:{token}:3"))
    wrong = Image.new("RGB", (128, 128))
    wrong.putdata(upstream_textures._render_wall(kind, base, wrong_rng, 128))
    assert actual.tobytes() != wrong.tobytes()


def test_wall_material_native_render_is_reused_across_facade_outputs(monkeypatch) -> None:
    token = _token("western_brick", "brick", "red brick")
    calls = 0
    original = upstream_textures._render_wall

    def counted(kind, base, rng, size):
        nonlocal calls
        calls += 1
        return original(kind, base, rng, size)

    monkeypatch.setattr(upstream_textures, "_render_wall", counted)
    _wall_material_image.cache_clear()

    modeler_wall_texture_image("residential", 128, token, 4)
    modeler_open_wall_texture_image("residential", 128, token, 4)
    modeler_interior_wall_texture_image("residential", 128, token, 4)
    modeler_front_texture_image("residential", 128, token, 4, "")

    assert calls == 1
    info = _wall_material_image.cache_info()
    assert info.misses == 1
    assert info.hits >= 3


def test_cached_material_is_not_mutated_by_front_composition() -> None:
    token = _token("western_stucco", "plaster", "cream")
    _wall_material_image.cache_clear()
    before = modeler_open_wall_texture_image("residential", 128, token, 7).tobytes()
    modeler_front_texture_image("residential", 128, token, 7, "")
    after = modeler_open_wall_texture_image("residential", 128, token, 7).tobytes()
    assert before == after


def test_modeler_front_composites_real_window_and_door_materials() -> None:
    token = _token(
        "swedish_wood", "painted timber", "falun red",
        window_material="uPVC", door_material="natural timber",
    )
    wall = modeler_wall_texture_image("residential", 128, token, 0)
    front = modeler_front_texture_image("residential", 128, token, 0, "")
    assert wall.tobytes() != front.tobytes()
    assert sum(high - low for low, high in ImageStat.Stat(front).extrema) > 180


def test_modeler_roof_uses_roof_material_generator_and_exposure() -> None:
    metal = modeler_roof_texture_image(
        "gabled|standing-seam metal|dark green", 128, 1
    )
    tile = modeler_roof_texture_image(
        "gabled|clay/concrete tile|red brick", 128, 1
    )
    assert metal.tobytes() != tile.tobytes()
    assert _mean_luma(metal) < 150.0
    assert _mean_luma(tile) < 150.0


def test_empty_building_catalogue_does_not_depend_on_last_variant_key() -> None:
    library = ProceduralBuildingLibrary(world_name="empty_modeler")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result = library.write_assets(root, root / "catalogue.json")
        assert result.placements == 0
        assert result.generated_variants == 0
        assert (root / "catalogue.json").is_file()
