# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from pathlib import Path

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_embedded_paved_geometry as _embedded
from cwr_worldgen import road_inspector_paved_wedge_audit as _audit
from cwr_worldgen.pbo import PboEntry, write_pbo


def _triangle_mlod(points: tuple[tuple[float, float], ...]) -> bytes:
    payload = bytearray()
    payload.extend(_embedded._MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    payload.extend(_embedded._SP3X_HEADER.pack(b"SP3X", _embedded._SP3X_HEADER.size, 1, 3, 0, 0, 0))
    for x, z in points:
        payload.extend(_embedded._POINT.pack(float(x), 0.0, float(z), 0))
    return bytes(payload)


def test_reads_actual_embedded_wedge_visual_triangle(tmp_path: Path) -> None:
    narrow = ((0.0, 0.04868436), (0.21846822, 0.0), (-0.21846822, 0.0))
    pbo = tmp_path / "wg_test.pbo"
    write_pbo(pbo, (PboEntry(r"i\paved_wedge_q020.p3d", _triangle_mlod(narrow)),))
    footprints = _embedded._embedded_wedge_footprints(pbo)
    actual = footprints["paved_wedge_q020.p3d"]
    assert actual is not None
    for measured, expected in zip(actual, narrow):
        assert math.dist(measured, expected) < 1.0e-7


def test_embedded_narrow_wedge_beats_current_filename_recipe() -> None:
    current = _audit._paved_wedge_local_points(5.0)
    assert abs(float(current[1][0])) > 0.35
    narrow = ((0.0, 0.04868436), (0.21846822, 0.0), (-0.21846822, 0.0))
    road = _core.RoadObject(
        object_id=10,
        model_path=r"wg_test\i\paved_wedge_q020.p3d",
        x=0.0,
        y=1.0,
        z=0.0,
        heading_degrees=0.0,
        pitch_degrees=0.0,
        family="sil",
        kind="paved_wedge",
        nominal_length_metres=0.05,
        logical_center=(0.0, 0.0),
        endpoints=(),
    )
    assert not _embedded._embedded_wedge_contains(road, (0.35, 0.0), narrow, margin=0.003)


def test_malformed_embedded_helper_is_not_treated_as_current_geometry(tmp_path: Path) -> None:
    pbo = tmp_path / "wg_test.pbo"
    write_pbo(pbo, (PboEntry(r"i\paved_wedge_q020.p3d", b"invalid"),))
    assert _embedded._embedded_wedge_footprints(pbo) == {"paved_wedge_q020.p3d": None}
