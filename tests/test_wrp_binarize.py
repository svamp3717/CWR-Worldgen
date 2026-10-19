"""Experimental OFP OPRW v3 binarization and compatibility regression tests."""
from __future__ import annotations

from pathlib import Path
import struct

import pytest

from cwr_worldgen.model import WorldObject
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_binarize import _packed, binarize_rvw4
from cwr_island_studio.oprw import OprwIslandDocument, open_island_document


def _source(path: Path, cells: int = 16) -> None:
    elevations = [((i % cells) + i // cells) * 0.5 - 15 for i in range(cells*cells)]
    textures = [1 if i % 3 else 2 for i in range(cells*cells)]
    write_rvw4(
        path, cells, cells, elevations, textures,
        [r"landtext\mo.pac", r"o\data\s.paa", r"o\data\w.paa"],
        (
            WorldObject(10, r"o\road\sil25.p3d", 120, 4, 200, 90),
            WorldObject(11, r"data3d\strom.p3d", 200, 8, 120, 35),
            WorldObject(12, r"o\road\sil25.p3d", 100, 9, 50, 0),
        ),
        height_scale=0.5,
    )


def test_binarize_generated_rvw4_oprw_v3_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "world.wrp"
    target = tmp_path / "world.binarized.wrp"
    _source(source)
    original = source.read_bytes()
    object_count = binarize_rvw4(source, target, height_scale=0.5)
    assert object_count == 3
    assert source.read_bytes() == original
    assert target.read_bytes()[:8] == b"OPRW\x03\x00\x00\x00"
    with open_island_document(target) as doc:
        assert isinstance(doc, OprwIslandDocument)
        assert (doc.width, doc.height) == (16, 16)
        assert doc.oprw_version == 3
        assert doc.terrain_at(0, 0) == pytest.approx(-15.0)
        assert doc.terrain_at(3, 2) == pytest.approx(-12.5)
        assert doc.texture_at(0, 0) == 2
        assert doc.texture_at(1, 0) == 1
        assert doc.texture_paths[2] == r"o\data\w.paa"
        assert doc.model_names == (
            r"o\road\sil25.p3d", r"data3d\strom.p3d",
        )
        assert [doc.object_at(i).object_id for i in range(3)] == [10, 11, 12]
        assert [doc.object_at(i).x for i in range(3)] == pytest.approx([120, 200, 100])
        doc.build_object_index()
        assert doc.object_count == 3
        with pytest.raises(PermissionError):
            doc.export(tmp_path / "no.wrp")


def test_binarization_supports_large_compressed_grids(tmp_path: Path) -> None:
    source = tmp_path / "large.wrp"
    target = tmp_path / "large.oprw.wrp"
    _source(source, cells=64)
    assert binarize_rvw4(source, target, height_scale=0.5) == 3
    with open_island_document(target) as doc:
        assert doc.terrain_at(63, 63) == pytest.approx(48.0)
        assert doc.texture_at(63, 63) == 2
        assert doc.object_count == 3


def test_binarize_rejects_overwriting_source_or_truncated_objects(tmp_path: Path) -> None:
    source = tmp_path / "test.wrp"
    _source(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        binarize_rvw4(source, source)
    assert source.read_bytes() == before
    source.write_bytes(before[:-1])
    target = tmp_path / "out.wrp"
    with pytest.raises(ValueError, match="misaligned"):
        binarize_rvw4(source, target)
    assert not target.exists()


def test_literal_and_repeated_sscompress_blocks_are_readable(tmp_path: Path) -> None:
    # A large all-zero grid collapses into backward references.
    raw = bytes(4096)
    encoded = _packed(raw)
    assert len(encoded) < 1500
    from cwr_island_studio.oprw import _read_packed
    import io
    assert _read_packed(io.BytesIO(encoded), len(raw), "grid") == raw
    assert _packed(b"test") == b"test"
    # Repeating four-byte patterns uses a different reference distance.
    raw = (b"ABCD" * 2048)
    assert _read_packed(io.BytesIO(_packed(raw)), len(raw), "grid") == raw
