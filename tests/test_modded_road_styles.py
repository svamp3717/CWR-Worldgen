from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cwr_worldgen import generator
from cwr_worldgen import paved_junction_policy
from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen.assets import model_texture_dependencies, scan_assets
from cwr_worldgen.osm import road_model_for_tags
from cwr_worldgen.gui import build_milestone9_command, default_gui_values


def _write_fake_mod_asset(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative.replace("\\", "/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_modded_road_texture_is_discovered_from_donor_p3d(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_fake_mod_asset(
        root,
        r"myroads\asphalt25.p3d",
        b"MLOD synthetic myroads\\textures\\asphalt_main.paa\x00",
    )
    _write_fake_mod_asset(
        root,
        r"myroads\textures\asphalt_main.paa",
        b"synthetic-paa",
    )

    scan = scan_assets(
        (root,),
        (r"myroads\asphalt25.p3d",),
        use_cache=False,
    )
    dependencies = model_texture_dependencies(
        scan.records,
        r"myroads\asphalt25.p3d",
    )

    assert dependencies == (r"myroads\textures\asphalt_main.paa",)
    assert generator._preferred_road_texture(
        r"myroads\asphalt25.p3d",
        dependencies,
    ) == r"myroads\textures\asphalt_main.paa"


def test_modded_family_reuses_only_existing_sibling_models(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_fake_mod_asset(root, r"myroads\asphalt25.p3d", b"donor")
    _write_fake_mod_asset(root, r"myroads\asphalt6.p3d", b"short")
    # asphalt12.p3d is deliberately absent.

    spec = SimpleNamespace(
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )
    availability = generator._modded_road_variant_availability(spec)
    token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    try:
        variants = playability.road_model_variants(
            spec.paved_road_model,
            spec.road_segment_length,
        )
    finally:
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(token)

    assert [piece.model_path for piece in variants] == [
        r"myroads\asphalt25.p3d",
        r"myroads\asphalt6.p3d",
    ]


def test_modded_family_without_asset_roots_never_invents_siblings() -> None:
    spec = SimpleNamespace(
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )
    availability = generator._modded_road_variant_availability(spec)
    token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    try:
        variants = playability.road_model_variants(
            spec.paved_road_model,
            spec.road_segment_length,
        )
    finally:
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(token)

    assert [piece.model_path for piece in variants] == [
        r"myroads\asphalt25.p3d",
    ]


def test_configured_modded_gravel_model_is_selected() -> None:
    spec = SimpleNamespace(
        name="world",
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model=r"myroads\gravel25.p3d",
        dirt_road_model=r"myroads\dirt25.p3d",
        procedural_gravel_roads=True,
    )

    assert road_model_for_tags(
        spec,
        {"highway": "track", "surface": "gravel"},
    ) == r"myroads\gravel25.p3d"


@pytest.mark.parametrize(
    ("surface", "texture"),
    (
        ("paved", r"modroads\textures\paved.paa"),
        ("gravel", r"modroads\textures\gravel.paa"),
        ("dirt", r"modroads\textures\dirt.paa"),
    ),
)
def test_generated_missing_shapes_reuse_mod_texture(
    tmp_path: Path,
    surface: str,
    texture: str,
) -> None:
    model = infrastructure.custom_road_model_path(
        "donorworld",
        surface,
        width_metres=5.8,
        length_metres=11.7,
        curve_degrees=27.0,
    )
    library = infrastructure.ProceduralInfrastructureLibrary(
        "donorworld",
        paved_texture_path=r"modroads\textures\paved.paa",
        gravel_texture_path=r"modroads\textures\gravel.paa",
        dirt_texture_path=r"modroads\textures\dirt.paa",
        cache_enabled=False,
    )
    library.register_model(model)
    catalogue = tmp_path / "infrastructure.json"
    result = library.write_assets(tmp_path, catalogue)

    assert result.generated_variants == 1
    assert result.texture_files == ()
    relative = model.split("\\", 1)[1].replace("\\", "/")
    summary = infrastructure.inspect_mlod(tmp_path / relative)
    assert texture.casefold() in {
        value.casefold() for value in summary.texture_paths
    }

    document = json.loads(catalogue.read_text(encoding="utf-8"))
    source = document[f"{surface}_texture_source"]
    assert source["type"] == "external-road-texture"
    assert source["texture"].casefold() == texture.casefold()


def test_generated_paved_junction_accepts_modded_width() -> None:
    incidents = (
        ((0.0, 1.0), "sil"),
        ((0.0, -1.0), "sil"),
        ((1.0, 0.0), "sil"),
    )
    plan = paved_junction_policy._generated_plan(
        (100.0, 100.0),
        incidents,
        world_name="donorworld",
        width_override=6.0,
    )

    assert plan is not None
    assert plan.model_path.endswith(
        r"\i\paved_j3_w060_h000_090_180.p3d"
    )


def test_generated_custom_paved_is_recognized_by_junction_policy() -> None:
    model = infrastructure.custom_road_model_path(
        "donorworld", "paved", 6.0, 10.0, 27.0
    )
    assert paved_junction_policy._family(model) == "sil"


def test_gui_command_exposes_all_three_modded_road_donors() -> None:
    values = default_gui_values()
    values.update({
        "source_dir": "source",
        "output": "build",
        "name": "donorworld",
        "display_name": "Donor World",
        "paved_road_model": r"myroads\asphalt25.p3d",
        "gravel_road_model": r"myroads\gravel25.p3d",
        "dirt_road_model": r"myroads\track25.p3d",
    })
    command = build_milestone9_command(values, python="python")

    for option, model in (
        ("--paved-road-model", r"myroads\asphalt25.p3d"),
        ("--gravel-road-model", r"myroads\gravel25.p3d"),
        ("--dirt-road-model", r"myroads\track25.p3d"),
    ):
        index = command.index(option)
        assert command[index + 1] == model
