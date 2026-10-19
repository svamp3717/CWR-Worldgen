"""MLOD/ODOL metadata classification, PBO lookup, and WRP overlays."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct

from cwr_island_studio.model_classification import (
    ModelCategory, read_p3d_category, load_p3d_category,
)
from cwr_island_studio.pbo_textures import PboTextures
from cwr_island_studio.oprw import open_island_document
from cwr_worldgen.model import WorldObject
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_binarize import binarize_rvw4
from cwr_worldgen.odol_v7 import mlod_to_odol
from cwr_worldgen.procedural_buildings import _Lod, _write_lod


def model(*, geometry: tuple[tuple[str, str], ...] = (),
          visual: tuple[tuple[str, str], ...] = ()) -> bytes:
    points = ((0., 0., 0.), (2., 0., 0.), (2., 0., 3.))
    first = _Lod(points, (), (), 1., properties=visual)
    second = _Lod(points, (), (), 1e13, properties=geometry)
    stream = BytesIO()
    stream.write(struct.pack("<4sBBHI", b"MLOD", 1, 1, 0, 2))
    _write_lod(stream, first)
    _write_lod(stream, second)
    return stream.getvalue()


def pbo(path: Path, name: str, data: bytes) -> None:
    path.write_bytes(
        name.encode("ascii") + b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, len(data))
        + b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0) + data
    )


def world(path: Path) -> None:
    write_rvw4(
        path, 4, 4, [0.] * 16, [0] * 16, [r"data\grass.paa"],
        (
            WorldObject(1, r"custom\mystery_road.p3d", 75, 0, 75),
            WorldObject(2, r"custom\mystery_green.p3d", 125, 0, 75),
            WorldObject(3, r"o\road\sil25.p3d", 175, 0, 75),
        ),
        height_scale=0.05,
    )


def test_mlod_map_and_class_properties_override_ambiguous_names() -> None:
    assert read_p3d_category(model(geometry=(("map", "road"),))) == ModelCategory(road=True)
    assert read_p3d_category(model(geometry=(("class", "bushsoft"),))) == ModelCategory(forest=True)
    assert read_p3d_category(model(geometry=(("map", "forest square"),))) == ModelCategory(forest=True)
    assert read_p3d_category(model(visual=(("map", "tree"),))) == ModelCategory(forest=True)
    assert read_p3d_category(model(
        visual=(("map", "tree"),), geometry=(("map", "house"),),
    )) == ModelCategory()
    assert read_p3d_category(model(geometry=(("map", "house"),))) == ModelCategory()
    assert read_p3d_category(model()) is None
    assert read_p3d_category(b"ODOL" + b"\0" * 32) is None


def test_odol_v7_real_model_metadata_round_trip() -> None:
    tree = mlod_to_odol(model(geometry=(("map", "tree"), ("autocenter", "0"))))
    house = mlod_to_odol(model(geometry=(("map", "house"), ("autocenter", "0"))))
    road = mlod_to_odol(model(
        geometry=(("map", "hide"), ("autocenter", "0")),
        visual=(("class", "road"),),
    ))
    assert read_p3d_category(tree) == ModelCategory(forest=True)
    assert read_p3d_category(house) == ModelCategory()
    assert read_p3d_category(road) == ModelCategory(road=True)


def test_model_lookup_uses_loose_folder_or_pbo_without_eager_mesh(tmp_path: Path) -> None:
    name = r"custom\mystery_road.p3d"
    archive = tmp_path / "custom.pbo"
    payload = model(geometry=(("map", "road"),))
    pbo(archive, r"mystery_road.p3d", payload)
    with_resources = PboTextures((archive,))
    try:
        assert load_p3d_category(name, None, with_resources) == ModelCategory(road=True)
        folder = tmp_path / "models"
        target = folder / "custom" / "mystery_road.p3d"
        target.parent.mkdir(parents=True)
        target.write_bytes(model(geometry=(("map", "tree"),)))
        assert load_p3d_category(name, folder, with_resources) == ModelCategory(forest=True)
        assert load_p3d_category(r"..\secrets.p3d", folder, with_resources) is None
        assert load_p3d_category(r"custom\missing.p3d", folder, with_resources) is None
    finally:
        with_resources.close()


def test_rvw4_metadata_populates_road_and_forest_overlays(tmp_path: Path) -> None:
    source = tmp_path / "sample.wrp"
    world(source)
    with open_island_document(source) as doc:
        doc.build_object_index()
        assert [x.object_id for x in doc.visible_objects(
            0, 0, 200, 200, roads_only=True,
        )] == [3]
        doc.set_model_categories({
            r"custom\mystery_road.p3d": ModelCategory(road=True),
            r"custom\mystery_green.p3d": ModelCategory(forest=True),
            r"o\road\sil25.p3d": ModelCategory(),
        })
        assert [x.object_id for x in doc.visible_objects(
            0, 0, 200, 200, roads_only=True,
        )] == [1]
        assert [x.object_id for x in doc.visible_objects(
            0, 0, 200, 200, forest_only=True,
        )] == [2]


def test_oprw_metadata_uses_model_table_and_not_placement_names(tmp_path: Path) -> None:
    source = tmp_path / "sample.wrp"
    binary = tmp_path / "sample.oprw.wrp"
    world(source)
    binarize_rvw4(source, binary, height_scale=0.05)
    with open_island_document(binary) as doc:
        doc.build_object_index()
        doc.set_model_categories({
            r"custom\mystery_road.p3d": ModelCategory(road=True),
            r"custom\mystery_green.p3d": ModelCategory(forest=True),
            r"o\road\sil25.p3d": ModelCategory(),
        })
        assert [x.object_id for x in doc.visible_objects(
            0, 0, 200, 200, roads_only=True,
        )] == [1]
        assert [x.object_id for x in doc.visible_objects(
            0, 0, 200, 200, forest_only=True,
        )] == [2]
