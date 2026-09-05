from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile

from PIL import ImageChops

from cwr_worldgen import osm_house_modeler_upgrade as upgrade
from cwr_worldgen import procedural_buildings as pb
from cwr_worldgen.osm import BboxProjection
from cwr_worldgen.osm_house_modeler_fidelity import DETAIL_MATERIAL_CODES
from cwr_worldgen.osm_house_modeler_full_style import texture_metadata_from_token
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary


def _empty_dataset():
    return SimpleNamespace(
        places=(), building_polygons=(), building_points=(), place_areas=(),
    )


def _sweden_key(*, extra_tags: dict[str, str] | None = None, interiors: bool = False):
    projection = BboxProjection.create((58.0, 17.0, 60.0, 19.0), 8_000.0)
    library = ProceduralBuildingLibrary(
        world_name="WGFidelity", maximum_variants=16, generate_interiors=interiors,
    )
    library.prepare(_empty_dataset(), projection, 8.0)
    x, z = projection.to_world((59.33, 18.07))
    tags = {
        "building": "cottage",
        "addr:country": "SE",
        "building:levels": "2",
    }
    tags.update(extra_tags or {})
    key = library.plan_point(tags, 10.0, 0.0, x=x, z=z).requested
    return library, key


def _visual(key: pb.BuildingVariantKey):
    return pb._visual_lod(
        key,
        r"WGFidelity\d\wall.paa",
        r"WGFidelity\d\roof.paa",
        key.roof_pitch_degrees or 35.0,
        front_texture=r"WGFidelity\d\front.paa",
        foundation_texture=r"WGFidelity\d\foundation.paa",
        foundation_depth=key.foundation_depth_m,
        plain_wall_texture=r"WGFidelity\d\plain.paa",
        interior_texture=r"WGFidelity\d\inside.paa",
        window_trim_texture=r"WGFidelity\d\trim.paa",
    )


def test_texture_token_carries_resolved_window_and_door_metadata() -> None:
    _library, key = _sweden_key()
    metadata = texture_metadata_from_token(key.texture_style_token)
    assert metadata
    assert metadata["window"]["width_m"] == key.window_width_m
    assert metadata["window"]["height_m"] == key.window_height_m
    assert metadata["window"]["sill_height_m"] == key.window_sill_height_m
    assert metadata["window"]["type"] == key.window_type
    assert metadata["door"]["width_m"] == key.door_width_m
    assert metadata["door"]["height_m"] == key.door_height_m


def test_closed_facade_is_repainted_from_modeler_opening_dimensions() -> None:
    _library, key = _sweden_key()
    painted = pb._wall_texture_image(
        key.family, 128, key.texture_style_token, key.texture_variant
    )
    plain = pb._open_wall_texture_image(
        key.family, 128, key.texture_style_token, key.texture_variant
    )
    assert painted.size == plain.size == (128, 128)
    assert ImageChops.difference(painted, plain).getbbox() is not None

    front = pb._front_texture_image(
        key.family, 128, key.texture_style_token, key.texture_variant,
        key.outbuilding_kind,
    )
    assert ImageChops.difference(front, painted).getbbox() is not None


def test_modeler_visible_plinth_controls_foundation_reveal() -> None:
    _library, key = _sweden_key()
    assert key.visible_plinth_m > 0.0
    core_only = replace(
        key,
        regional_style="default",
        exterior_detail_spec_json="",
        eave_overhang_m=0.0,
        roof_storey=False,
    )
    lod = _visual(core_only)
    foundation_faces = [
        face for face in lod.faces
        if face.texture.endswith(r"foundation.paa")
    ]
    assert foundation_faces
    y_values = [
        lod.points[vertex[0]][1]
        for face in foundation_faces
        for vertex in face.vertices
        if vertex[0] >= 0
    ]
    assert max(y_values) >= key.visible_plinth_m - 1.0e-6


def test_eave_overhang_adds_real_geometry_outside_wall_footprint() -> None:
    _library, key = _sweden_key()
    assert key.eave_overhang_m > 0.03
    core_only = replace(
        key,
        regional_style="default",
        exterior_detail_spec_json="",
        roof_storey=False,
    )
    lod = _visual(core_only)
    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    assert any(
        abs(x) > half_width + 0.04 or abs(z) > half_length + 0.04
        for x, _y, z in lod.points
    )
    detail_suffixes = tuple(rf"\d\{code}.paa" for code in DETAIL_MATERIAL_CODES.values())
    assert any(face.texture.endswith(detail_suffixes) for face in lod.faces)


def test_roof_storey_creates_real_gable_glass_and_frame_geometry() -> None:
    _library, key = _sweden_key(extra_tags={"roof:levels": "1"})
    assert key.roof_storey
    assert key.roof_style == "gabled"
    core_only = replace(
        key,
        regional_style="default",
        exterior_detail_spec_json="",
        eave_overhang_m=0.0,
    )
    lod = _visual(core_only)
    glass_suffix = rf"\d\{DETAIL_MATERIAL_CODES['glass']}.paa"
    assert any(face.texture.endswith(glass_suffix) for face in lod.faces)
    glass_points = {
        vertex[0]
        for face in lod.faces
        if face.texture.endswith(glass_suffix)
        for vertex in face.vertices
        if vertex[0] >= 0
    }
    assert glass_points
    assert min(lod.points[index][1] for index in glass_points) > 0.0


def test_country_selected_feature_materials_reach_generated_mlod() -> None:
    projection = BboxProjection.create((58.0, 17.0, 60.0, 19.0), 8_000.0)
    library = ProceduralBuildingLibrary(world_name="WGDetailMats", maximum_variants=16)
    library.prepare(_empty_dataset(), projection, 8.0)
    x, z = projection.to_world((59.33, 18.07))
    placement = library.plan_point(
        {
            "building": "cottage",
            "addr:country": "SE",
            "building:levels": "2",
            "roof:levels": "1",
            "entrance:steps": "yes",
            "porch": "yes",
            "chimney": "yes",
            "balcony": "yes",
        },
        10.0,
        0.0,
        x=x,
        z=z,
    )
    spec = upgrade.detail_spec_from_key(placement.requested)
    # Source country metadata may still describe/force a porch, but CWR no
    # longer emits porch geometry because it does not read well in-game.
    assert spec["porches"]["enabled"]
    assert not upgrade.detail_plan_for_key(
        placement.requested, foundation_depth=placement.requested.foundation_depth_m
    ).porch
    assert spec["chimneys"]["enabled"]
    assert spec["balconies"]["enabled"]
    library.register_placement(placement)

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        result = library.write_assets(root, root / "buildings.json")
        expected_files = {f"d/{code}.paa" for code in DETAIL_MATERIAL_CODES.values()}
        assert expected_files <= set(result.texture_files)
        assert result.model_assets
        summary = pb.inspect_mlod(root / result.model_assets[0].relative_path)
        paths = set(summary.texture_paths)
        assert any(path.endswith(rf"\d\{DETAIL_MATERIAL_CODES['balcony']}.paa") for path in paths)
        assert any(path.endswith(rf"\d\{DETAIL_MATERIAL_CODES['glass']}.paa") for path in paths)
        # At least one material-selected porch/stair/chimney/rainwater surface
        # must use the dedicated masonry/wood/metal set rather than facade art.
        material_suffixes = tuple(
            rf"\d\{DETAIL_MATERIAL_CODES[kind]}.paa"
            for kind in ("masonry", "wood", "metal")
        )
        assert any(path.endswith(material_suffixes) for path in paths)
