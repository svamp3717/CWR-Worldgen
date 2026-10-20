# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import struct

import pytest

from cwr_worldgen.assets import AssetRecord
from cwr_worldgen.model import WorldObject
from cwr_worldgen.vegetation_origin_policy import (
    _odol_visual_base_offset,
    apply_vegetation_origin_lifts,
    vegetation_origin_lifts,
)


def _centered_odol(min_y: float, max_y: float) -> bytes:
    # The first LOD is enough for the same ODOL v7 geometry table read in
    # normal asset scans. Low-count arrays are stored uncompressed.
    points = ((-1.0, min_y, -1.0), (1.0, max_y, 1.0), (0.0, 0.0, 0.0))
    return (
        b"ODOL" + struct.pack("<II", 7, 1)
        + struct.pack("<I3I", 3, 0, 0, 0)  # point flags
        + struct.pack("<I6f", 3, *(0.0,) * 6)  # UVs
        + struct.pack("<I", 3)
        + b"".join(struct.pack("<fff", *pt) for pt in points)
    )


def test_vegetation_pivot_comes_from_actual_odol_visual_bottom() -> None:
    assert _odol_visual_base_offset(_centered_odol(-4.25, 4.25)) == pytest.approx(4.25)
    assert _odol_visual_base_offset(_centered_odol(0.0, 8.0)) == 0.0


def test_single_trees_yard_fruit_and_field_bushes_share_pivot_lift(
    tmp_path: Path,
) -> None:
    model_paths = (
        r"data3d\str smrk_medium.p3d",
        r"data3d\ker listnac.p3d",
        r"data3d\jablon.p3d",
        r"o\tree\Javor01.p3d",
    )
    objects = tuple(
        WorldObject(index + 1, model, 100.0, 10.05, 100.0)
        for index, model in enumerate(model_paths)
    ) + (
        WorldObject(5, r"data3d\les ctverec pruchozi_T1.p3d", 150, 10.0, 150),
        WorldObject(6, r"world\f\c_101.p3d", 160, 10.0, 160),
    )
    records = []
    for index, model in enumerate(model_paths):
        source = tmp_path / f"vegetation_{index}.p3d"
        source.write_bytes(_centered_odol(-2.0 - index, 2.0 + index))
        records.append(AssetRecord(model, str(source), source.stat().st_size, None))

    spec = SimpleNamespace(name="world")
    lifts, report = vegetation_origin_lifts(objects, records, spec)
    transformed = apply_vegetation_origin_lifts(objects, lifts)
    assert report.lifted_objects == 4
    assert report.lifted_models == 4
    assert not report.unavailable_models
    assert not report.unreadable_models
    for index, obj in enumerate(transformed[:4]):
        assert obj.y == pytest.approx(10.05 + 2.0 + index)
        # The stock first-LOD trunk base is restored to intended terrain Y.
        assert obj.y - (2.0 + index) == pytest.approx(10.05)
        assert obj.object_id == index + 1
    assert transformed[4:] == objects[4:]


def test_missing_asset_is_reported_instead_of_guessing_tree_height() -> None:
    obj = WorldObject(1, r"data3d\hrusen.p3d", 30, 4.0, 30)
    lifts, report = vegetation_origin_lifts((obj,), (), SimpleNamespace(name="world"))
    assert not lifts
    assert report.unavailable_models == (obj.model_path,)
    assert apply_vegetation_origin_lifts((obj,), lifts) == (obj,)


def test_broken_odol_is_not_used_for_placement(tmp_path: Path) -> None:
    source = tmp_path / "bad.p3d"
    source.write_bytes(b"ODOL" + b"\0" * 8)
    model = r"data3d\ker pichlavej.p3d"
    obj = WorldObject(1, model, 0, 0, 0)
    record = AssetRecord(model, str(source), source.stat().st_size, None)
    lifts, report = vegetation_origin_lifts((obj,), (record,), SimpleNamespace(name="world"))
    assert not lifts
    assert report.unreadable_models
