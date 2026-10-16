# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for experimental CWA ODOL v7 building export."""
from __future__ import annotations

from io import BytesIO
import struct

import pytest

from cwr_worldgen import odol_v7
from cwr_worldgen.legacy_proxy_models import inspect_visual_model_dimensions
from cwr_worldgen.procedural_buildings import (
    _Face, _Lod, _NamedSelection, _write_lod,
)


def _quad(texture: str, *, flags: int = 0, normal_index: int = 0) -> _Face:
    return _Face(
        texture,
        ((0, normal_index, 0., 0.), (1, normal_index, 1., 0.),
         (2, normal_index, 1., 1.), (3, normal_index, 0., 1.)),
        flags,
    )


def _mlod(*, many_faces: bool = False) -> bytes:
    points = (
        (-2., 0., -3.),
        (2., 0., -3.),
        (2., 0., 3.),
        (-2., 0., 3.),
    )
    visual = _Lod(
        points,
        ((0., 1., 0.),),
        (_quad(r"o\road\sil_new.paa"),) * (300 if many_faces else 1),
        1.0,
        point_flags=(0x1,)*4,
        properties=(("class", "house"),),
    )
    # The model must survive a face-bearing Geometry LOD without normals,
    # plus a Memory LOD containing points without any visual faces.
    collision = _Lod(
        points, (), (_quad("", normal_index=-1),), 1e13,
        selections=(
            _NamedSelection("component01", bytes([1, 1, 1, 1]), b"\x01"),
        ),
        mass_per_point=(100., 100., 100., 100.),
        properties=(("map", "house"),),
    )
    memory = _Lod(
        ((0., 0., 0.), (0., 1., 0.)), (), (), 1e15,
        selections=(
            _NamedSelection("door1_axis", bytes([1, 1]), b""),
        ),
    )
    buf = BytesIO()
    buf.write(struct.pack("<4sBBHI", b"MLOD", 1, 1, 0, 3))
    for lod in (visual, collision, memory):
        _write_lod(buf, lod)
    return buf.getvalue()


def test_python_odol_writer_has_native_v7_header_and_preserves_lods() -> None:
    odol = odol_v7.mlod_to_odol(_mlod())
    assert odol[:4] == b"ODOL"
    assert struct.unpack_from("<II", odol, 4) == (7, 3)
    assert b"component01\0" in odol
    assert b"door1_axis\0" in odol
    shape = inspect_visual_model_dimensions(odol)
    assert shape.source_format == "ODOL"
    assert shape.width_metres == pytest.approx(4.0)
    assert shape.length_metres == pytest.approx(6.0)


def test_large_building_arrays_use_decodable_sscompress_stream() -> None:
    odol = odol_v7.mlod_to_odol(_mlod(many_faces=True))
    assert struct.unpack_from("<II", odol, 4) == (7, 3)
    report = inspect_visual_model_dimensions(odol)
    assert report.width_metres == pytest.approx(4.0)


def test_building_export_atomic_and_idempotent_failure(tmp_path) -> None:
    target = tmp_path / "test_building.p3d"
    target.write_bytes(_mlod())
    odol_v7.binarize_building(target)
    assert target.read_bytes()[:8] == b"ODOL\x07\x00\x00\x00"
    original = target.read_bytes()
    with pytest.raises(ValueError, match="MLOD"):
        odol_v7.binarize_building(target)
    assert target.read_bytes() == original


def test_bad_mlod_is_not_written_as_odol() -> None:
    with pytest.raises(ValueError, match="MLOD"):
        odol_v7.mlod_to_odol(b"not a p3d")


def test_python_flags_match_cwr_ce_clip_flags() -> None:
    assert odol_v7._point_flag(0) == 0x3F
    assert odol_v7._point_flag(1) == 0x13F
    assert odol_v7._point_flag(0x10) & 0xFF00000 == 200 * 0x100000


def test_gui_advanced_checkbox_defaults_off_and_sets_cli_flag() -> None:
    from cwr_worldgen.gui import default_gui_values, build_milestone9_command
    values = default_gui_values()
    assert values["odol_buildings"] is False
    assert "--odol-buildings" not in build_milestone9_command(values, python="python")
    values["odol_buildings"] = True
    assert "--odol-buildings" in build_milestone9_command(values, python="python")


def test_milestone9_cli_and_specs_accept_odol_buildings() -> None:
    from cwr_worldgen.cli import _parser
    from cwr_worldgen.milestone9 import Milestone9Spec, _Milestone9PlayabilitySpec
    opts = _parser().parse_args([
        "milestone9", "--source-dir", "dummy", "--output", "build/dummy",
        "--odol-buildings",
    ])
    assert opts.odol_buildings is True
    assert Milestone9Spec.__dataclass_fields__["odol_buildings"].default is False
    assert _Milestone9PlayabilitySpec.__dataclass_fields__["odol_buildings"].default is False


def test_mlod_collision_missing_normal_sentinel_is_valid() -> None:
    lods = odol_v7.read_mlod(_mlod())
    assert len(lods) == 3
    geometry = lods[1]
    assert not geometry.normals
    assert geometry.faces[0][2][0][1] == -1
    # This is the same intentional -1 normal generated for polygon-native and
    # regular procedural building Geometry LODs; conversion must not reject it.
    result = odol_v7.mlod_to_odol(_mlod())
    assert result.startswith(b"ODOL\\x07\\x00\\x00\\x00")


def test_missing_normal_sentinel_does_not_select_last_normal() -> None:
    lod = odol_v7._Lod(
        resolution=1.,
        points=[(-1., 0., -1., 0), (1., 0., -1., 0),
                (1., 0., 1., 0), (-1., 0., 1., 0)],
        normals=[(0., 1., 0.)],
        faces=[("", 0, [(0, -1, 0., 0.), (1, -1, 1., 0.),
                        (2, -1, 1., 1.), (3, -1, 0., 1.)])],
        selections=[], properties=[], mass=[],
    )
    # Array counts are u32; the zero-normal flags are uint32, UV pairs
    # are 2 floats and positions/normals are 3 floats each.
    data = odol_v7._write_lod(lod)
    normal_offset = (4 + 8 * 4) + (4 + 8 * 8) + (4 + 8 * 12) + 4
    # Four original points precede four per-face vertices.
    assert struct.unpack_from("<fff", data, normal_offset + 4 * 12) == (0., 0., 0.)


@pytest.mark.parametrize("point,normal", [(4, -1), (-1, -1), (0, -2), (0, 1)])
def test_corrupt_mlod_face_index_still_fails(point: int, normal: int) -> None:
    raw = bytearray(_mlod())
    # The first visual face begins after the 12-byte MLOD header,
    # 28-byte SP3X header, four 16-byte points, one 12-byte normal,
    # 32-byte texture name and four-byte corner count.
    corner_offset = 12 + 28 + 4 * 16 + 12 + 32 + 4
    struct.pack_into("<ii", raw, corner_offset, point, normal)
    with pytest.raises(ValueError, match="face 0 corner 0 references"):
        odol_v7.read_mlod(bytes(raw))
