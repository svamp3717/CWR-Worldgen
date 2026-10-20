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
    RUNTIME_SNAP_BIAS_METRES,
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
    assert report.measured_models == 4
    assert not report.runtime_snap_models
    assert not report.unavailable_models
    assert not report.unreadable_models
    for index, obj in enumerate(transformed[:4]):
        assert obj.y == pytest.approx(10.05 + 2.0 + index)
        # The stock first-LOD trunk base is restored to intended terrain Y.
        assert obj.y - (2.0 + index) == pytest.approx(10.05)
        assert obj.object_id == index + 1
    assert transformed[4:] == objects[4:]


def test_missing_asset_uses_builtin_cwr_ce_ground_snap_without_p3ds() -> None:
    obj = WorldObject(1, r"data3d\hrusen.p3d", 30, 4.0, 30)
    lifts, report = vegetation_origin_lifts((obj,), (), SimpleNamespace(name="world"))
    assert lifts[obj.model_path] == RUNTIME_SNAP_BIAS_METRES
    assert report.runtime_snap_models == (obj.model_path,)
    assert report.unavailable_models == (obj.model_path,)
    adjusted = apply_vegetation_origin_lifts((obj,), lifts)
    assert adjusted[0].y == 4.0 + RUNTIME_SNAP_BIAS_METRES
    assert adjusted[0].object_id == obj.object_id

    # Landscape::ObjectCreate moves the object's centre down by the amount
    # its original model zero (centre minus bounding centre) exceeds terrain.
    # This is the engine's actual placement calculation, with different
    # stock models requiring different centring. No P3D read is needed.
    for bounding_center_y in (0.0, 1.5, 7.0, 25.0):
        above = (adjusted[0].y - bounding_center_y) - 4.0
        assert above > 0
        engine_centre_y = adjusted[0].y - above
        engine_original_zero_y = engine_centre_y - bounding_center_y
        assert engine_original_zero_y == pytest.approx(4.0)


def test_broken_odol_is_not_used_for_placement(tmp_path: Path) -> None:
    source = tmp_path / "bad.p3d"
    source.write_bytes(b"ODOL" + b"\0" * 8)
    model = r"data3d\ker pichlavej.p3d"
    obj = WorldObject(1, model, 0, 0, 0)
    record = AssetRecord(model, str(source), source.stat().st_size, None)
    lifts, report = vegetation_origin_lifts((obj,), (record,), SimpleNamespace(name="world"))
    assert lifts[model] == RUNTIME_SNAP_BIAS_METRES
    assert report.runtime_snap_models == (model,)
    assert report.unreadable_models


def test_no_p3d_assets_cover_house_street_forest_and_bush_models() -> None:
    model_paths = (
        r"data3d\jablon.p3d",
        r"data3d\hrusen.p3d",
        r"o\tree\Javor01.p3d",
        r"data3d\str smrk_medium.p3d",
        r"data3d\ker listnac.p3d",
        r"data3d\ker pichlavej.p3d",
    )
    objects = tuple(
        WorldObject(i + 1, path, i * 10.0, 9.0, 20.0)
        for i, path in enumerate(model_paths)
    )
    lifts, report = vegetation_origin_lifts(
        objects, (), SimpleNamespace(name="world")
    )
    assert report.lifted_models == len(model_paths)
    assert report.lifted_objects == len(model_paths)
    assert report.measured_models == 0
    assert len(report.runtime_snap_models) == len(model_paths)
    assert all(value == RUNTIME_SNAP_BIAS_METRES for value in lifts.values())
    assert all(
        new.y == pytest.approx(old.y + RUNTIME_SNAP_BIAS_METRES)
        for old, new in zip(objects, apply_vegetation_origin_lifts(objects, lifts))
    )


def test_runtime_ground_snap_survives_rvw4_to_oprw_binarization(tmp_path: Path) -> None:
    from cwr_worldgen.wrp import write_rvw4
    from cwr_worldgen.wrp_binarize import binarize_rvw4
    from cwr_island_studio.oprw import open_island_document

    original = WorldObject(5, r"data3d\ker listnac.p3d", 100.0, 9.0, 100.0)
    lifts, _ = vegetation_origin_lifts((original,), (), SimpleNamespace(name="world"))
    translated = apply_vegetation_origin_lifts((original,), lifts)[0]
    input_path = tmp_path / "vegetation.wrp"
    output_path = tmp_path / "vegetation.oprw"
    write_rvw4(
        input_path, 16, 16, [9.0] * 256, [0] * 256,
        [r"landtext\main.paa"], (translated,), height_scale=0.045,
    )
    assert binarize_rvw4(input_path, output_path) == 1
    with open_island_document(input_path, height_scale=0.045) as loose:
        assert loose.object_at(0).y == pytest.approx(
            9.0 + RUNTIME_SNAP_BIAS_METRES
        )
    with open_island_document(output_path, cwa_runtime_scale=True) as packed:
        assert packed.object_at(0).y == pytest.approx(
            9.0 + RUNTIME_SNAP_BIAS_METRES
        )
