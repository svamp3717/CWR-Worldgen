from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from cwr_worldgen import assets as asset_module
from cwr_worldgen import cli
from cwr_worldgen import generator
from cwr_worldgen import paved_junction_policy
from cwr_worldgen import paved_road_generated_fallback_policy as fallback
from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen.assets import model_texture_dependencies, scan_assets
from cwr_worldgen.osm import road_model_for_tags
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.procedural_buildings import _Face, _Lod, _MLOD_HEADER, _write_lod
from cwr_worldgen.gui import build_milestone9_command, default_gui_values
from cwr_worldgen.source_pipeline import Milestone5Spec
from cwr_worldgen.milestone6 import Milestone6Spec
from cwr_worldgen.milestone7 import Milestone7Spec
from cwr_worldgen.milestone8 import Milestone8Spec
from cwr_worldgen.milestone9 import Milestone9Spec
from cwr_worldgen.legacy_proxy_models import inspect_visual_model_dimensions


def _write_fake_mod_asset(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative.replace("\\", "/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _mlod_road(
    width: float,
    length: float,
    texture: str,
) -> bytes:
    half_width = width * 0.5
    half_length = length * 0.5
    lod = _Lod(
        (
            (-half_width, 0.025, -half_length),
            (-half_width, 0.025, half_length),
            (half_width, 0.025, half_length),
            (half_width, 0.025, -half_length),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 0.0, 1.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 1.0, 0.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _mlod_curve_sample(
    width: float,
    length: float,
    lateral_shift: float,
    texture: str,
) -> bytes:
    half_width = width * 0.5
    half_length = length * 0.5
    lower_center = -lateral_shift * 0.5
    upper_center = lateral_shift * 0.5
    lod = _Lod(
        (
            (lower_center - half_width, 0.025, -half_length),
            (upper_center - half_width, 0.025, half_length),
            (upper_center + half_width, 0.025, half_length),
            (lower_center + half_width, 0.025, -half_length),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 0.0, 1.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 1.0, 0.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _mlod_skewed_road(
    width: float,
    chord_length: float,
    lateral_shift: float,
    texture: str,
) -> bytes:
    half_width = width * 0.5
    half_length = chord_length * 0.5
    half_shift = lateral_shift * 0.5
    lod = _Lod(
        (
            (-half_width - half_shift, 0.025, -half_length),
            (half_width - half_shift, 0.025, -half_length),
            (half_width + half_shift, 0.025, half_length),
            (-half_width + half_shift, 0.025, half_length),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 1.0, 0.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 0.0, 1.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


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
        "paved_road_curve_model": r"myroads\asphalt_curve.p3d",
        "gravel_road_model": r"myroads\gravel25.p3d",
        "gravel_road_curve_model": r"myroads\gravel_curve.p3d",
        "dirt_road_model": r"myroads\track25.p3d",
        "dirt_road_curve_model": r"myroads\track_curve.p3d",
    })
    command = build_milestone9_command(values, python="python")

    for option, model in (
        ("--paved-road-model", r"myroads\asphalt25.p3d"),
        ("--paved-road-curve-model", r"myroads\asphalt_curve.p3d"),
        ("--gravel-road-model", r"myroads\gravel25.p3d"),
        ("--gravel-road-curve-model", r"myroads\gravel_curve.p3d"),
        ("--dirt-road-model", r"myroads\track25.p3d"),
        ("--dirt-road-curve-model", r"myroads\track_curve.p3d"),
    ):
        index = command.index(option)
        assert command[index + 1] == model


def test_modded_road_family_and_texture_are_discovered_inside_pbo(
    tmp_path: Path,
) -> None:
    pbo = tmp_path / "myroads.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "asphalt25.p3d",
                b"MLOD donor myroads\\textures\\asphalt_main.paa\x00",
            ),
            PboEntry(
                "asphalt12.p3d",
                b"MLOD short myroads\\textures\\asphalt_main.paa\x00",
            ),
            PboEntry("textures/asphalt_main.paa", b"synthetic-paa"),
        ),
    )

    spec = SimpleNamespace(
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(pbo,),
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
        r"myroads\asphalt12.p3d",
    ]

    scan = scan_assets(
        (pbo,),
        (spec.paved_road_model,),
        use_cache=False,
    )
    dependencies = model_texture_dependencies(
        scan.records,
        spec.paved_road_model,
    )
    assert dependencies == (r"myroads\textures\asphalt_main.paa",)


def test_modded_donor_geometry_controls_generated_width_and_length(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    donor = r"myroads\asphalt_long.p3d"
    texture = r"myroads\tex\road.paa"
    _write_fake_mod_asset(root, donor, _mlod_road(7.2, 21.4, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    spec = SimpleNamespace(
        name="donorworld",
        paved_road_model=donor,
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
    )
    dimensions = generator._modded_road_model_dimensions(spec)
    key = playability._road_model_key(donor)
    assert dimensions[key] == pytest.approx((7.2, 21.4))

    dimension_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    tag_token = playability._ACTIVE_ROAD_TAGS.set(
        {"highway": "residential", "surface": "asphalt"}
    )
    try:
        variants = playability.road_model_variants(donor, spec.road_segment_length)
        assert len(variants) == 1
        assert variants[0].model_path == donor
        assert variants[0].length_metres == pytest.approx(21.4)

        # Residential OSM width is 6 m, but a generated bend joining this donor
        # must stay 7.2 m wide or it would visibly neck down at the seam.
        assert fallback._generated_width(
            variants,
            spec,
            "paved",
        ) == pytest.approx(7.2)
    finally:
        playability._ACTIVE_ROAD_TAGS.reset(tag_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimension_token)


def test_modded_donor_geometry_is_measured_inside_pbo(tmp_path: Path) -> None:
    texture = r"myroads\textures\track.paa"
    pbo = tmp_path / "myroads.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "track_long.p3d",
                _mlod_road(3.8, 18.6, texture),
            ),
            PboEntry("textures/track.paa", b"synthetic-paa"),
        ),
    )
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model="",
        dirt_road_model=r"myroads\track_long.p3d",
        road_segment_length=25.0,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    dimensions = generator._modded_road_model_dimensions(spec)
    assert dimensions[
        playability._road_model_key(spec.dirt_road_model)
    ] == pytest.approx((3.8, 18.6))


def test_milestone9_cli_accepts_modded_road_donor_flags() -> None:
    args = cli._parser().parse_args(
        [
            "milestone9",
            "--source-dir",
            "source",
            "--output",
            "build",
            "--paved-road-model",
            r"myroads\paved25.p3d",
            "--paved-road-curve-model",
            r"myroads\paved_curve.p3d",
            "--gravel-road-model",
            r"myroads\gravel25.p3d",
            "--gravel-road-curve-model",
            r"myroads\gravel_curve.p3d",
            "--dirt-road-model",
            r"myroads\track25.p3d",
            "--dirt-road-curve-model",
            r"myroads\track_curve.p3d",
        ]
    )

    assert args.paved_road_model == r"myroads\paved25.p3d"
    assert args.paved_road_curve_model == r"myroads\paved_curve.p3d"
    assert args.gravel_road_model == r"myroads\gravel25.p3d"
    assert args.gravel_road_curve_model == r"myroads\gravel_curve.p3d"
    assert args.dirt_road_model == r"myroads\track25.p3d"
    assert args.dirt_road_curve_model == r"myroads\track_curve.p3d"


@pytest.mark.parametrize("command", ("milestone5", "milestone6", "milestone7", "milestone8", "milestone9"))
def test_source_bundle_cli_exposes_road_donor_flags(command: str) -> None:
    args = cli._parser().parse_args(
        [
            command,
            "--source-dir",
            "source",
            "--output",
            "build",
            "--paved-road-model",
            r"myroads\paved25.p3d",
        ]
    )
    assert args.paved_road_model == r"myroads\paved25.p3d"


@pytest.mark.parametrize(
    "spec_type",
    (Milestone5Spec, Milestone6Spec, Milestone7Spec, Milestone8Spec, Milestone9Spec),
)
def test_source_bundle_specs_accept_road_donor_models(spec_type) -> None:
    spec = spec_type(
        source_dir=Path("source"),
        paved_road_model=r"myroads\paved25.p3d",
        paved_road_curve_model=r"myroads\paved_curve.p3d",
        gravel_road_model=r"myroads\gravel25.p3d",
        gravel_road_curve_model=r"myroads\gravel_curve.p3d",
        dirt_road_model=r"myroads\track25.p3d",
        dirt_road_curve_model=r"myroads\track_curve.p3d",
    )

    assert spec.paved_road_model == r"myroads\paved25.p3d"
    assert spec.paved_road_curve_model == r"myroads\paved_curve.p3d"
    assert spec.gravel_road_model == r"myroads\gravel25.p3d"
    assert spec.gravel_road_curve_model == r"myroads\gravel_curve.p3d"
    assert spec.dirt_road_model == r"myroads\track25.p3d"
    assert spec.dirt_road_curve_model == r"myroads\track_curve.p3d"


def test_milestone9_spec_accepts_gui_road_arguments_exactly() -> None:
    args = cli._parser().parse_args(
        [
            "milestone9",
            "--source-dir",
            r"G:\cwa_worldgen\source-data\tiny_bjorsund",
            "--output",
            r"G:\cwa_worldgen\build\terrtest82",
            "--name",
            "wg_terrtest82",
            "--display-name",
            "terrtest82",
            "--paved-road-model",
            r"o\road\sil25.p3d",
            "--gravel-road-model",
            r"o\road\sil25.p3d",
            "--dirt-road-model",
            r"o\road\ces25.p3d",
        ]
    )

    spec = Milestone9Spec(
        source_dir=args.source_dir,
        name=args.name,
        display_name=args.display_name,
        paved_road_model=args.paved_road_model,
        gravel_road_model=args.gravel_road_model,
        dirt_road_model=args.dirt_road_model,
    )

    assert spec.paved_road_model == r"o\road\sil25.p3d"
    assert spec.gravel_road_model == r"o\road\sil25.p3d"
    assert spec.dirt_road_model == r"o\road\ces25.p3d"


def test_pre_zstd_asset_cache_is_invalidated_for_nested_mod_donor(
    tmp_path: Path,
) -> None:
    inner = tmp_path / "sebnam_obj.pbo"
    texture = r"sebnam_obj\trail.paa"
    donor = r"sebnam_obj\sebtrailpath10 25.p3d"
    write_pbo(
        inner,
        (
            PboEntry("sebtrailpath10 25.p3d", _mlod_road(4.8, 24.5, texture)),
            PboEntry("trail.paa", b"synthetic-paa"),
        ),
    )
    outer = tmp_path / "mod-package.pbo"
    write_pbo(
        outer,
        (PboEntry(r"addons\sebnam_obj.pbo", inner.read_bytes()),),
    )
    wrapped = tmp_path / "mod-package.pbo.zst"
    wrapped.write_bytes(
        zstandard.ZstdCompressor(level=1).compress(outer.read_bytes())
    )
    cache_dir = tmp_path / "cache"

    first = scan_assets(
        (wrapped,),
        (donor,),
        cache_dir=cache_dir,
        use_cache=True,
        refresh=False,
    )
    assert donor in {record.path for record in first.records}
    assert first.cache_path is not None

    cache_path = Path(first.cache_path)
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document["asset_schema"] = 1
    document["records"] = []
    cache_path.write_text(json.dumps(document), encoding="utf-8")
    asset_module._CATALOGUE_MEMORY.clear()

    second = scan_assets(
        (wrapped,),
        (donor,),
        cache_dir=cache_dir,
        use_cache=True,
        refresh=False,
    )

    assert donor in {record.path for record in second.records}
    assert second.cache_hit is False
    deps = model_texture_dependencies(second.records, donor)
    assert deps == (texture,)


def test_unresolved_mod_gravel_donor_never_silently_uses_generic_texture() -> None:
    donor = r"sebnam_obj\sebtrailpath10 25.p3d"
    with pytest.raises(ValueError, match="could not resolve the gravel road donor"):
        generator._resolved_road_donor_texture(
            (),
            surface="gravel",
            donor_model=donor,
        )


def test_stock_road_donors_keep_known_texture_fallbacks() -> None:
    assert generator._resolved_road_donor_texture(
        (),
        surface="paved",
        donor_model=r"o\road\sil25.p3d",
    ) == r"landtext\silnice.pac"
    assert generator._resolved_road_donor_texture(
        (),
        surface="dirt",
        donor_model=r"o\road\ces25.p3d",
    ) == r"o\road\ces_hned.paa"


def test_sebnam_curve_donor_resolves_to_straight_family(tmp_path: Path) -> None:
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    curved = r"sebnam_obj\sebtrailpath10 25.p3d"
    pbo = tmp_path / "sebnam_obj.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "sebtrailpath10 25.p3d",
                _mlod_skewed_road(3.5, 4.36, 0.40, texture),
            ),
            PboEntry("sebtrailpath25.p3d", _mlod_road(3.5, 25.0, texture)),
            PboEntry("sebtrailpath12.p3d", _mlod_road(3.5, 12.5, texture)),
            PboEntry("sebtrailpath6.p3d", _mlod_road(3.5, 6.25, texture)),
            PboEntry("p/sebtrailpath.paa", b"synthetic-paa"),
        ),
    )
    spec = SimpleNamespace(
        name="sebworld",
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model=curved,
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=24.5,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
        procedural_gravel_roads=True,
    )

    scan = scan_assets((pbo,), (curved,), use_cache=False)
    by_path = {record.path: record for record in scan.records}
    curved_shape = inspect_visual_model_dimensions(
        asset_module.read_asset_record_bytes(by_path[curved])
    )
    straight_shape = inspect_visual_model_dimensions(
        asset_module.read_asset_record_bytes(
            by_path[r"sebnam_obj\sebtrailpath25.p3d"]
        )
    )
    assert curved_shape.is_straight_road_candidate is False
    assert straight_shape.is_straight_road_candidate is True
    assert model_texture_dependencies(scan.records, curved) == (texture,)
    assert model_texture_dependencies(
        scan.records, r"sebnam_obj\sebtrailpath25.p3d"
    ) == (texture,)
    match = generator._CURVED_ROAD_DONOR_NAME.fullmatch(
        r"sebtrailpath10 25.p3d"
    )
    assert match is not None
    assert match.group("prefix") == "sebtrailpath"
    preferred = rf"sebnam_obj\{match.group('prefix')}25.p3d"
    assert preferred == r"sebnam_obj\sebtrailpath25.p3d"
    assert preferred in by_path

    effective = generator._modded_road_effective_donors(spec)
    assert effective[playability._road_model_key(curved)] == (
        r"sebnam_obj\sebtrailpath25.p3d"
    )

    availability = generator._modded_road_variant_availability(spec, effective)
    dimensions = generator._modded_road_model_dimensions(spec, effective)
    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(effective)
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    try:
        assert playability.road_model_for_tags(
            spec,
            {"highway": "track", "surface": "gravel"},
        ) == r"sebnam_obj\sebtrailpath25.p3d"

        variants = playability.road_model_variants(
            curved,
            spec.road_segment_length,
        )
        assert [piece.model_path for piece in variants] == [
            r"sebnam_obj\sebtrailpath25.p3d",
            r"sebnam_obj\sebtrailpath12.p3d",
            r"sebnam_obj\sebtrailpath6.p3d",
        ]
        assert [piece.length_metres for piece in variants] == pytest.approx(
            [25.0, 12.5, 6.25]
        )
        assert fallback._generated_width(
            variants,
            spec,
            "gravel",
        ) == pytest.approx(3.5)
        assert playability.gravel_filler_piece(spec, 6).model_path == (
            r"sebnam_obj\sebtrailpath6.p3d"
        )
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)


def test_curved_mod_donor_without_straight_sibling_is_rejected(tmp_path: Path) -> None:
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    pbo = tmp_path / "sebnam_obj.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "sebtrailpath10 25.p3d",
                _mlod_skewed_road(3.5, 4.36, 0.40, texture),
            ),
            PboEntry("p/sebtrailpath.paa", b"synthetic-paa"),
        ),
    )
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model=r"sebnam_obj\sebtrailpath10 25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=24.5,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    with pytest.raises(ValueError, match="curved road piece"):
        generator._modded_road_effective_donors(spec)


def test_sebnam_style_uses_straight_family_and_explicit_curve_reference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    straight25 = r"sebnam_obj\sebtrailpath25.p3d"
    straight12 = r"sebnam_obj\sebtrailpath12.p3d"
    straight6 = r"sebnam_obj\sebtrailpath6.p3d"
    curve = r"sebnam_obj\sebtrailpath10 25.p3d"

    _write_fake_mod_asset(root, straight25, _mlod_road(3.5, 25.0, texture))
    _write_fake_mod_asset(root, straight12, _mlod_road(3.5, 12.5, texture))
    _write_fake_mod_asset(root, straight6, _mlod_road(3.5, 6.25, texture))
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(3.5, 4.65, 0.38, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    spec = SimpleNamespace(
        name="sebworld",
        paved_road_model=r"o\road\sil25.p3d",
        paved_road_curve_model="",
        gravel_road_model=straight25,
        gravel_road_curve_model=curve,
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
    )

    effective = generator._modded_road_effective_donors(spec)
    assert effective[playability._road_model_key(straight25)] == straight25
    assert generator._road_style_donor(spec, "gravel") == curve

    availability = generator._modded_road_variant_availability(spec, effective)
    dimensions = generator._modded_road_model_dimensions(spec, effective)
    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(effective)
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    try:
        pieces = playability.road_model_variants(straight25, 25.0)
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)

    assert [piece.model_path for piece in pieces] == [
        straight25,
        straight12,
        straight6,
    ]
    assert [piece.length_metres for piece in pieces] == pytest.approx(
        [25.0, 12.5, 6.25]
    )


def test_legacy_curved_sebnam_selection_auto_resolves_to_straight_sibling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    curve = r"sebnam_obj\sebtrailpath10 25.p3d"
    straight = r"sebnam_obj\sebtrailpath25.p3d"
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(3.5, 4.65, 0.38, texture))
    _write_fake_mod_asset(root, straight, _mlod_road(3.5, 25.0, texture))
    _write_fake_mod_asset(root, r"sebnam_obj\sebtrailpath12.p3d", _mlod_road(3.5, 12.5, texture))
    _write_fake_mod_asset(root, r"sebnam_obj\sebtrailpath6.p3d", _mlod_road(3.5, 6.25, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model=curve,
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    effective = generator._modded_road_effective_donors(spec)
    assert effective[playability._road_model_key(curve)] == straight


def test_legacy_curve_resolution_never_walks_unrelated_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "mod"
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    curve = r"sebnam_obj\sebtrailpath10 25.p3d"
    straight = r"sebnam_obj\sebtrailpath25.p3d"
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(3.5, 4.65, 0.38, texture))
    _write_fake_mod_asset(root, straight, _mlod_road(3.5, 25.0, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    # Hundreds of unrelated invalid P3Ds reproduce the old pathological case.
    # The resolver must not open any of them merely to identify one road family.
    for index in range(300):
        _write_fake_mod_asset(
            root,
            rf"unrelated\junk{index:03d}.p3d",
            b"not-a-p3d",
        )

    inspected: list[bytes] = []
    real_inspect = generator.inspect_visual_model_dimensions

    def counted_inspect(data: bytes):
        inspected.append(data[:8])
        return real_inspect(data)

    monkeypatch.setattr(
        generator,
        "inspect_visual_model_dimensions",
        counted_inspect,
    )

    spec = SimpleNamespace(
        paved_road_model="",
        paved_road_curve_model="",
        gravel_road_model=curve,
        gravel_road_curve_model="",
        dirt_road_model="",
        dirt_road_curve_model="",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    effective = generator._modded_road_effective_donors(spec)

    assert effective[playability._road_model_key(curve)] == straight
    # The filename + exact sibling lookup is enough; no model mesh walk occurs.
    assert inspected == []


def test_exact_road_family_lookup_ignores_missing_texture_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    donor = r"myroads\road25.p3d"
    _write_fake_mod_asset(
        root,
        donor,
        _mlod_road(4.0, 25.0, r"myroads\missing.paa"),
    )

    from cwr_worldgen.fast_asset_scan_policy import locate_assets_fast

    result = locate_assets_fast(
        (root,),
        (donor,),
        use_cache=False,
    )

    assert result.missing_models == ()
    assert result.missing_dependencies == ()
    assert [record.path for record in result.records] == [donor]
