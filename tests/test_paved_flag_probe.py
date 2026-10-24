"""Guard against changing texture, geometry or Roadway when testing SFP flags."""
from __future__ import annotations

import io
import json
import struct
from zipfile import ZipFile

import pytest

from cwr_worldgen.pbo import read_pbo
from tools.build_paved_flag_probe import (
    build_probe_pair,
    patch_stock_visual_flags,
    visual_flag_spans,
)


def _native_lod(*, point_flag: int, face_flag: int) -> bytes:
    out = io.BytesIO()
    put = out.write
    p = lambda value: put(struct.pack("<I", value))
    p(4)
    for _ in range(4):
        p(point_flag)
    p(4)
    put(struct.pack("<8f", 0, 0, 0, 1, 1, 1, 1, 0))
    p(4)
    for xyz in (
        (-4.55, 0, -12.5), (4.55, 0, -12.5),
        (4.55, 0, 12.5), (-4.55, 0, 12.5),
    ):
        put(struct.pack("<3f", *xyz))
    p(4)
    for _ in range(4):
        put(struct.pack("<3f", 0, -1, 0))
    put(b"\0" * 48)
    p(1)
    put(b"o\\road\\sil_new.paa\0")
    p(0)  # point-to-vertex
    p(0)  # vertex-to-point
    p(2)
    p(0)  # face-allocation size
    for indices in ((0, 1, 2), (0, 2, 3)):
        p(face_flag)
        put(struct.pack("<hB", 0, 3))
        put(struct.pack("<3H", *indices))
    for _ in range(4):
        p(0)  # sections, selections, properties, animations
    put(b"\0" * 12)
    p(0)  # no proxies
    return out.getvalue()


def _native_model() -> bytes:
    return (
        b"ODOL" + struct.pack("<II", 7, 2)
        + _native_lod(point_flag=0x13F, face_flag=0x24102)
        + _native_lod(point_flag=0x13F, face_flag=0x2C102)
        + struct.pack("<2f", 1.0, 3.0e15)
        + b"\x7f" * 32  # trailing ODOL metadata preserved byte-for-byte
    )


def test_paved_flag_probe_patches_visual_lod_not_roadway() -> None:
    original = _native_model()
    modified, changed = patch_stock_visual_flags(original)
    assert changed == 1
    assert len(modified) == len(original)
    assert modified[-32:] == original[-32:]
    before, after = visual_flag_spans(original), visual_flag_spans(modified)
    assert before[0][0][1] == [0x13F] * 4
    assert after[0][0][1] == [1] * 4
    assert [flag for _, flag in after[0][0][2]] == [0, 0]
    assert after[1] == before[1]
    assert sum(a != b for a, b in zip(original, modified)) == 14


def test_probe_rejects_unsupported_stock_model_flags() -> None:
    original = _native_model().replace(
        struct.pack("<I", 0x24102), struct.pack("<I", 0x7c),
    )
    with pytest.raises(ValueError, match="expected stock values"):
        patch_stock_visual_flags(original)


def test_probe_writes_control_and_experimental_archives(tmp_path) -> None:
    archive = tmp_path / "Road(1).zip"
    names = (
        "sil25.p3d", "sil12.p3d", "sil6.p3d",
        "sil10 25.p3d", "sil10 50.p3d",
        "sil10 75.p3d", "sil10 100.p3d",
    )
    with ZipFile(archive, "w") as contents:
        for name in names:
            contents.writestr("Road/" + name, _native_model())

    output_dir = tmp_path / "out"
    report = build_probe_pair(archive, output_dir)
    assert report["game_rendering_verified"] is False
    assert report["models"]["sil25.p3d"]["visual_lods_patched"] == 1
    assert report["models"]["sil25.p3d"]["length_unchanged"] is True
    assert json.loads(
        (output_dir / "paved-flag-probe-manifest.json").read_text()
    )["control_pbo"] == "cwr_paved_stock_control.pbo"
    control = read_pbo(output_dir / "cwr_paved_stock_control.pbo")
    experiment = read_pbo(output_dir / "cwr_paved_flag_probe.pbo")
    assert len(control) == len(experiment) == len(names)
    assert [entry.name for entry in control] == [entry.name for entry in experiment]
    assert control[0].data == _native_model()
    assert experiment[0].data == patch_stock_visual_flags(_native_model())[0]
