"""Independent forest and real-size layers, including million-object-safe filters."""
from __future__ import annotations

from pathlib import Path
import struct

import pytest

from cwr_island_studio.core import IslandDocument
from cwr_island_studio.object_layers import is_forest_model, object_marker_mode

_HEADER = struct.Struct("<4sii")
_OBJECT = struct.Struct("<12fi76s")


def _item(object_id: int, name: str, x: float, z: float) -> bytes:
    return _OBJECT.pack(
        1, 0, 0, 0, 1, 0, 0, 0, 1, x, 0, z, object_id,
        name.encode("ascii").ljust(76, b"\0"),
    )


def _world(path: Path, objects: tuple[bytes, ...]) -> None:
    path.write_bytes(
        _HEADER.pack(b"4WVR", 16, 16)
        + bytes(16 * 16 * 4 + 512 * 32)
        + b"".join(objects) + bytes(128)
    )


@pytest.mark.parametrize("name", (
    r"data3d\les ctverec pruchozi_T1.p3d",
    r"data3d\les trojuhelnik pruchozi.p3d",
    r"data3d\str smrk_medium.p3d",
    r"data3d\ker listnac.p3d",
    r"data3d\str_fikovnik.p3d",
    r"o\tree\les_nw_jehl_ctverec.p3d",
    r"o\tree\smrk_velky.p3d",
    r"sebnam_obj\sebles_su_ctver_pruhozi.p3d",
    r"sebnam_obj\sebkrovi_long.p3d",
    r"generated_island\f\c_pine_stand_00.p3d",
    r"generated_island\f\b_border_00.p3d",
    r"generated_island\f\u_understory_00.p3d",
    r"generated_island\f\r_rural_00.p3d",
))
def test_recognizes_generated_and_stock_forest_assets(name: str) -> None:
    assert is_forest_model(name)
    assert is_forest_model(name.upper().replace("\\", "/").encode("ascii") + b"\0padding")


@pytest.mark.parametrize("name", (
    r"o\road\sil25.p3d",
    r"data3d\dum.p3d",
    r"world\f\g_ditch_grass_00.p3d",
    r"data3d\trava.p3d",
    r"data3d\les_square.paa",
    r"forest",
    "data3d/treehouse.p3d",
))
def test_forest_overlay_does_not_mix_road_house_ditch_grass(name: str) -> None:
    assert not is_forest_model(name)


@pytest.mark.parametrize(
    ("dots", "scaled", "known", "pixels", "expected"),
    (
        (False, False, True, 25.0, "hidden"),
        (True, False, True, 25.0, "dot"),
        (False, True, True, 25.0, "footprint"),
        (True, True, True, 25.0, "footprint"),
        (False, True, False, 0.0, "hidden"),
        (True, True, False, 0.0, "dot"),
        (False, True, True, 1.0, "hidden"),
        (True, True, True, 1.0, "dot"),
    ),
)
def test_model_size_layer_is_independent_of_generic_object_dots(
    dots: bool, scaled: bool, known: bool, pixels: float, expected: str,
) -> None:
    assert object_marker_mode(dots, scaled, known, pixels) == expected


def test_forest_only_query_before_and_after_index_and_sparse_edits(tmp_path: Path) -> None:
    path = tmp_path / "forest.wrp"
    _world(path, (
        _item(1, r"data3d\str smrk_medium.p3d", 120, 140),
        _item(2, r"data3d\dum.p3d", 120, 150),
        _item(3, r"o\road\sil25.p3d", 120, 160),
        _item(4, r"world\f\c_stand_00.p3d", 150, 140),
        _item(5, r"o\tree\smrk_velky.p3d", 155, 140),
    ))
    with IslandDocument(path, cell_size=25) as doc:
        assert [o.object_id for o in doc.visible_objects(0, 0, 400, 400, forest_only=True)] == [1, 4, 5]
        doc.build_object_index()
        assert [o.object_id for o in doc.visible_objects(0, 0, 400, 400, forest_only=True)] == [1, 4, 5]
        assert doc.visible_objects(0, 0, 400, 400, roads_only=True)[0].object_id == 3
        with pytest.raises(ValueError, match="either"):
            doc.visible_objects(0, 0, 400, 400, forest_only=True, roads_only=True)
        doc.set_model(1, r"data3d\str smrk_medium.p3d")
        doc.set_model(0, r"data3d\dum.p3d")
        doc.delete_object(3)
        doc.add_object(r"world\f\u_undergrowth_00.p3d", x=160, y=0, z=145)
        assert sorted(o.object_id for o in doc.visible_objects(
            0, 0, 400, 400, forest_only=True,
        )) == [2, 5, 6]
        # The original source file never changes during an edit.
        with IslandDocument(path, cell_size=25) as unchanged:
            assert unchanged.object_count == 5


def test_forest_filter_happens_before_viewport_marker_budget(tmp_path: Path) -> None:
    path = tmp_path / "dense.wrp"
    # Most objects are not vegetation. The forest-only layer must not
    # disappear when a general viewport hit list is capped.
    ordinary = tuple(
        _item(i + 1, r"data3d\dum.p3d", 100, 100)
        for i in range(500)
    )
    forest = tuple(
        _item(i + 501, r"o\tree\smrk_velky.p3d", 100, 100)
        for i in range(50)
    )
    _world(path, ordinary + forest)
    with IslandDocument(path, cell_size=25) as doc:
        doc.build_object_index()
        assert doc._forest_mask is not None
        visible = doc.visible_objects(0, 0, 400, 400, limit=10, forest_only=True)
        assert len(visible) == 10
        assert all(is_forest_model(obj.model_path) for obj in visible)
