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


def _quad(texture: str, *, flags: int = 0) -> _Face:
    return _Face(
        texture,
        ((0, 0, 0., 0.), (1, 0, 1., 0.),
         (2, 0, 1., 1.), (3, 0, 0., 1.)),
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
        points, (), (_quad(""),), 1e13,
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
