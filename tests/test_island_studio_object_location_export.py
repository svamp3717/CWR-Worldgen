"""Read-only Island Studio object exports for both RVW4 and OPRW maps."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import numpy as np
from PIL import Image
import pytest

from cwr_island_studio.app import IslandStudio
from cwr_island_studio.core import IslandDocument
from cwr_island_studio.export_naming import island_export_stem
from cwr_island_studio.object_location_export import export_object_location_atlas
from cwr_island_studio.oprw import open_island_document
from test_island_studio import make_world, record
from test_island_studio_oprw import make_oprw


def _read_csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


def test_object_export_includes_sparse_edits_and_north_up_positions(tmp_path: Path) -> None:
    source = tmp_path / "wg_test.wrp"
    original = make_world(
        source, objects=(
            record(1, r"o\road\sil25.p3d", 70, 350, shear=0.25),
            record(2, r"data3d\tree.p3d", 180, 200),
        ),
    )
    with IslandDocument(source, read_only=False) as doc:
        doc.move_object(0, x=20.0, y=3.0, z=375.0)
        doc.set_model(0, r"o\road\ces6.p3d")
        doc.delete_object(1)
        added = doc.add_object(r"data3d\other.p3d", x=370.0, y=5.0, z=25.0, heading=45)
        assert added == 2
        output = export_object_location_atlas(doc, tmp_path / "nogova_objects.zip")
        assert doc.changed
    assert source.read_bytes() == original
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "object-placements.csv", "object-models.csv", "object-density.png",
            "metadata.json", "README.txt",
        }
        placements = _read_csv(archive, "object-placements.csv")
        assert len(placements) == 2
        assert [int(v["record_index"]) for v in placements] == [0, 2]
        assert [int(v["object_id"]) for v in placements] == [1, 3]
        assert placements[0]["model_path"] == r"o\road\ces6.p3d"
        assert float(placements[0]["x_metres"]) == pytest.approx(20)
        assert float(placements[0]["z_metres"]) == pytest.approx(375)
        assert float(placements[1]["heading_degrees"]) == pytest.approx(45)
        assert "basis_xz" in placements[0] and "basis_zz" in placements[0]
        models = _read_csv(archive, "object-models.csv")
        assert {v["model_path"]: int(v["object_count"]) for v in models} == {
            r"o\road\ces6.p3d": 1, r"data3d\other.p3d": 1,
        }
        meta = json.loads(archive.read("metadata.json"))
        assert meta["schema"] == "island-studio-object-atlas-v1"
        assert meta["object_count"] == 2
        assert meta["original_record_count"] == 2
        assert meta["deleted_object_records_omitted"] == 1
        assert meta["pending_object_edits_included"] is True
        assert "north-up" in meta["orientation"]
        with Image.open(io.BytesIO(archive.read("object-density.png"))) as pic:
            pixels = np.asarray(pic.convert("RGB"))
            assert pixels.shape == (8, 8, 3)
            assert not np.array_equal(pixels[0, 0], pixels[4, 4])
            assert not np.array_equal(pixels[7, 7], pixels[4, 4])


def test_binarized_oprw_object_export_preserves_model_table_and_readonly_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "original.wrp"
    original = make_oprw(map_size=(4, 4), layer_size=(2, 3))
    source.write_bytes(original)
    with open_island_document(source, cell_size=50.0) as doc:
        assert doc.read_only
        output = export_object_location_atlas(doc, tmp_path / "everon_objects.zip")
    assert source.read_bytes() == original
    with zipfile.ZipFile(output) as archive:
        placements = _read_csv(archive, "object-placements.csv")
        assert [p["object_id"] for p in placements] == ["100", "101"]
        assert placements[0]["model_path"] == r"o\road\sil25.p3d"
        assert placements[1]["model_path"] == r"data3d\tree.p3d"
        assert [float(p["y_metres"]) for p in placements] == [12.5, 12.5]
        meta = json.loads(archive.read("metadata.json"))
        assert meta["object_count"] == 2
        assert meta["pending_object_edits_included"] is False
        assert (meta["density_image_width"], meta["density_image_height"]) == (4, 4)


def test_empty_object_export_is_valid(tmp_path: Path) -> None:
    source = tmp_path / "empty.wrp"
    original = make_world(source, objects=())
    with IslandDocument(source, read_only=True) as doc:
        output = export_object_location_atlas(doc, tmp_path / "empty_objects.zip")
    assert original == source.read_bytes()
    with zipfile.ZipFile(output) as archive:
        assert _read_csv(archive, "object-placements.csv") == []
        assert _read_csv(archive, "object-models.csv") == []
        assert json.loads(archive.read("metadata.json"))["object_count"] == 0


@pytest.mark.parametrize("source_label, expected", [
    (r"noe.pbo / noe\noe.wrp", "nogova"),
    (r"Everon.pbo / eden\maps\everon.wrp", "everon"),
    (r"Data\Islands\Malden.wrp", "malden"),
    ("cain.wrp", "kolgujev"),
    ("wg_generated.wrp", "nogova"),
])
def test_island_type_export_naming_from_source_or_stock_assets(
    tmp_path: Path, source_label: str, expected: str,
) -> None:
    source = tmp_path / "selected-world.wrp"
    make_world(source)
    with IslandDocument(source, read_only=True) as doc:
        doc.source_label = source_label
        assert island_export_stem(doc) == expected


def test_object_export_gui_name_and_background_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected-world.wrp"
    make_world(source)
    output = tmp_path / "result.zip"
    with IslandDocument(source, read_only=True) as doc:
        doc.source_label = r"Noe.pbo / noe\noe.wrp"
        doc._archive_owner = SimpleNamespace(
            path=tmp_path / "Noe.pbo", close=lambda: None,
        )
        dialogs = []
        monkeypatch.setattr(
            "cwr_island_studio.app.filedialog.asksaveasfilename",
            lambda **kwargs: dialogs.append(kwargs) or str(output),
        )
        messages = queue.Queue()
        editor = SimpleNamespace(
            document=doc, _busy=False, status=Mock(), _messages=messages,
            _executor=SimpleNamespace(submit=lambda func: func()),
        )
        IslandStudio._export_object_location_map(editor)
        assert dialogs[0]["initialfile"] == "nogova_objects.zip"
        assert dialogs[0]["initialdir"] == str(tmp_path)
        assert output.exists()
        received = []
        while not messages.empty():
            received.append(messages.get_nowait()[0])
        assert received[0] == "object_map_progress"
        assert received[-1] == "object_map_exported"
        assert editor._busy
    assert source.exists()


def test_object_export_rejects_non_zip_and_removes_partial_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "original.wrp"
    make_world(source)
    with IslandDocument(source) as doc:
        with pytest.raises(ValueError, match="zip"):
            export_object_location_atlas(doc, tmp_path / "bad.csv")

        def fail(_index: int):
            raise RuntimeError("object decode failed")

        monkeypatch.setattr(doc, "object_at", fail)
        output = tmp_path / "failed.zip"
        with pytest.raises(RuntimeError, match="decode failed"):
            export_object_location_atlas(doc, output)
        assert not output.exists()
        assert not output.with_name("failed.zip.partial").exists()


def test_unrecognized_island_name_is_sanitized(tmp_path: Path) -> None:
    source = tmp_path / "selected-world.wrp"
    make_world(source)
    with IslandDocument(source) as doc:
        doc.texture_paths = ()
        doc.source_label = "addon.pbo / maps/forest<edge>.wrp"
        assert island_export_stem(doc) == "forest_edge_"
