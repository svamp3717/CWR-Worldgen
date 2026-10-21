"""Exact north-up texture atlas exports for loose and packed Island Studio worlds."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import queue
import struct
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import numpy as np
from PIL import Image
import pytest

from cwr_island_studio.app import IslandStudio
from cwr_island_studio.core import IslandDocument
from cwr_island_studio.oprw import open_island_document
from cwr_island_studio.texture_location_export import export_texture_location_atlas
from test_island_studio_oprw import make_oprw


def _world(path: Path) -> bytes:
    width, height = 4, 3
    rows_south_to_north = (
        0, 0, 1, 1,
        0, 2, 2, 1,
        2, 2, 1, 1,
    )
    table = b"".join(
        name.encode("ascii").ljust(32, b"\0") for name in (
            r"o\j1.paa", r"o\j2.paa", r"eden\pl.paa",
        )
    ) + bytes(509 * 32)
    data = (
        struct.pack("<4sii", b"4WVR", width, height)
        + bytes(2 * width * height)
        + struct.pack("<12h", *rows_south_to_north)
        + table
    )
    path.write_bytes(data)
    return data


def _rows(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8"))))


def _rgb(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(member))) as im:
        return np.asarray(im.convert("RGB"))


def _id(pixels: np.ndarray, x: int, y: int) -> int:
    r, g, b = [int(c) for c in pixels[y, x]]
    value = (r << 16) | (g << 8) | b
    return value - (1 << 24) if value >= (1 << 23) else value


def test_exact_north_up_wrp_texture_map_and_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "island.wrp"
    original = _world(source)
    output = tmp_path / "atlas.zip"
    with IslandDocument(source, cell_size=10.0, read_only=True) as doc:
        assert export_texture_location_atlas(doc, output) == output
        assert doc.read_only
    assert source.read_bytes() == original
    with zipfile.ZipFile(output) as archive:
        assert "texture-map.html" not in archive.namelist()
        assert set(archive.namelist()) == {
            "texture-map.png", "texture-ids.png", "texture-paths.csv",
            "texture-runs.csv", "texture-adjacencies.csv",
            "metadata.json", "README.txt",
        }
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["texture_grid_width"] == 4
        assert metadata["texture_grid_height"] == 3
        assert metadata["texture_cell_width_metres"] == 10
        assert metadata["texture_cell_height_metres"] == 10
        assert metadata["covered_cells"] == 12
        assert "north-up" in metadata["orientation"]
        assert metadata["texture_paths"]["0"] == r"o\j1.paa"
        assert metadata["texture_paths"]["1"] == r"o\j2.paa"
        texture_ids = _rgb(archive, "texture-ids.png")
        assert texture_ids.shape == (3, 4, 3)
        assert [[_id(texture_ids, x, y) for x in range(4)]
                for y in range(3)] == [
                    [2, 2, 1, 1],
                    [0, 2, 2, 1],
                    [0, 0, 1, 1],
                ]
        visual = _rgb(archive, "texture-map.png")
        assert visual.shape == (3, 4, 3)
        assert not np.array_equal(visual[0, 0], visual[0, 2])
        rows = _rows(archive, "texture-runs.csv")
        assert len(rows) == 7
        first = rows[0]
        assert first["north_row"] == "0"
        assert first["game_z_row"] == "2"
        assert first["texture_path"] == r"eden\pl.paa"
        assert first["x_first_cell"] == "0"
        assert first["x_last_cell_inclusive"] == "1"
        assert float(first["z_min_metres"]) == 20
        assert float(first["z_max_metres"]) == 30
        assert sum(int(r["x_last_cell_inclusive"]) -
                   int(r["x_first_cell"]) + 1 for r in rows) == 12
        legend = _rows(archive, "texture-paths.csv")
        assert len(legend) == 512  # export every WRP table slot, including unused
        assert {int(row["texture_slot"]): int(row["cell_count"])
                for row in legend[:3]} == {0: 3, 1: 5, 2: 4}
        adjacency = {
            (int(row["texture_slot_a"]), int(row["texture_slot_b"])):
            int(row["shared_cell_edges"])
            for row in _rows(archive, "texture-adjacencies.csv")
        }
        assert adjacency == {(0, 1): 1, (0, 2): 3, (1, 2): 4}


def test_pending_texture_edits_are_included_without_saving_wrp(tmp_path: Path) -> None:
    source = tmp_path / "editable.wrp"
    initial = _world(source)
    with IslandDocument(source, cell_size=50.0) as doc:
        doc.texture_edits[1 * 4 + 1] = 0  # game southern-row index 1
        output = export_texture_location_atlas(doc, tmp_path / "edited.zip")
        assert doc.changed
    assert source.read_bytes() == initial
    with zipfile.ZipFile(output) as archive:
        ids = _rgb(archive, "texture-ids.png")
        assert _id(ids, 1, 1) == 0
        assert json.loads(archive.read("metadata.json"))[
            "sparse_pending_texture_edits_included"
        ] is True


def test_oprw_differing_texture_and_terrain_grids_export_exact_layer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "binarized.wrp"
    before = make_oprw(map_size=(8, 8), layer_size=(2, 3), objects=False)
    source.write_bytes(before)
    with open_island_document(source, cell_size=25.0) as doc:
        assert doc.read_only
        assert doc.width == 8 and doc.height == 8
        assert doc.textures.shape == (3, 2)
        output = export_texture_location_atlas(doc, tmp_path / "oprw.zip")
    assert source.read_bytes() == before
    with zipfile.ZipFile(output) as archive:
        meta = json.loads(archive.read("metadata.json"))
        assert (meta["texture_grid_width"], meta["texture_grid_height"]) == (2, 3)
        assert meta["texture_cell_width_metres"] == 100
        assert meta["texture_cell_height_metres"] == pytest.approx(200 / 3)
        ids = _rgb(archive, "texture-ids.png")
        assert [[_id(ids, x, y) for x in range(2)]
                for y in range(3)] == [[0, 1], [0, 1], [0, 1]]
        assert meta["covered_cells"] == 6


def test_export_survives_invalid_texture_slot_without_invented_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid_slot.wrp"
    _world(source)
    with IslandDocument(source, cell_size=10.0) as doc:
        doc.texture_edits[0] = -2
        output = export_texture_location_atlas(doc, tmp_path / "slots.zip")
    with zipfile.ZipFile(output) as archive:
        ids = _rgb(archive, "texture-ids.png")
        assert _id(ids, 0, 2) == -2
        legend = _rows(archive, "texture-paths.csv")
        invalid = next(row for row in legend if row["texture_slot"] == "-2")
        assert invalid["texture_path"] == ""
        assert invalid["cell_count"] == "1"


def test_gui_utility_exports_without_unchecking_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "readonly.wrp"
    _world(source)
    with IslandDocument(source, cell_size=10.0, read_only=True) as doc:
        dest = tmp_path / "gui-map.zip"
        dialogs = []
        monkeypatch.setattr(
            "cwr_island_studio.app.filedialog.asksaveasfilename",
            lambda **kwargs: dialogs.append(kwargs) or str(dest),
        )
        messages = queue.Queue()
        status = Mock()
        editor = SimpleNamespace(
            document=doc, _busy=False,
            _messages=messages, status=status,
            _executor=SimpleNamespace(submit=lambda work: work()),
        )
        IslandStudio._export_texture_location_map(editor)
        assert dest.exists()
        assert editor._busy
        received = []
        while not messages.empty():
            received.append(messages.get_nowait()[0])
        assert received[0] == "texture_map_progress"
        assert received[-1] == "texture_map_exported"
        assert dialogs[0]["filetypes"][0][1] == "*.zip"
        assert dialogs[0]["initialfile"] == "readonly.zip"
    assert source.exists()


@pytest.mark.parametrize(
    ("source_label", "expected"),
    [
        ("Nogova.wrp", "Nogova.zip"),
        (r"Everon.pbo / eden\\maps\\everon.wrp", "everon.zip"),
        (r"Data\\Islands\\Malden.wrp", "Malden.zip"),
        ("cain.wrp", "cain.zip"),
        ("addon.pbo / maps/forest<edge>.wrp", "forest_edge_.zip"),
    ],
)
def test_export_dialog_names_zip_after_original_island_not_temp_wrp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    source_label: str, expected: str,
) -> None:
    extracted = tmp_path / "selected-world.wrp"
    _world(extracted)
    with IslandDocument(extracted, read_only=True) as doc:
        doc.source_label = source_label
        doc._archive_owner = SimpleNamespace(
            path=tmp_path / "islands.pbo", close=lambda: None,
        )
        calls: list[dict] = []
        monkeypatch.setattr(
            "cwr_island_studio.app.filedialog.asksaveasfilename",
            lambda **kwargs: calls.append(kwargs) or "",
        )
        editor = SimpleNamespace(document=doc, _busy=False)
        IslandStudio._export_texture_location_map(editor)
        assert len(calls) == 1
        assert calls[0]["initialfile"] == expected
        assert calls[0]["initialdir"] == str(tmp_path)


def test_gui_cancel_and_busy_conditions_do_not_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "readonly.wrp"
    _world(source)
    with IslandDocument(source, read_only=True) as doc:
        calls = []
        monkeypatch.setattr(
            "cwr_island_studio.app.filedialog.asksaveasfilename",
            lambda **kwargs: calls.append(kwargs) or "",
        )
        editor = SimpleNamespace(document=doc, _busy=True)
        IslandStudio._export_texture_location_map(editor)
        assert not calls
        editor._busy = False
        IslandStudio._export_texture_location_map(editor)
        assert len(calls) == 1


def test_non_zip_destination_is_rejected_without_side_effects(tmp_path: Path) -> None:
    source = tmp_path / "source.wrp"
    _world(source)
    with IslandDocument(source) as doc:
        with pytest.raises(ValueError, match="zip"):
            export_texture_location_atlas(doc, tmp_path / "wrong.csv")
    assert not list(tmp_path.glob("*.partial"))
