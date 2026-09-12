from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from cwr_worldgen import parking_surface_policy as parking
from cwr_worldgen.parking_visual_detail_policy import (
    DETAIL_CACHE_SCHEMA,
    _world_grid,
    apply_parking_visual_detail,
)


def _geometry(surface: str):
    return parking._ParkingGeometry(
        osm_key=f"way/{surface}",
        outer=((3.0, 3.0), (47.0, 3.0), (47.0, 47.0), (3.0, 47.0)),
        holes=(),
        centre_x=25.0,
        centre_z=25.0,
        half_length=22.0,
        half_width=22.0,
        ux=1.0,
        uz=0.0,
        px=0.0,
        pz=1.0,
        surface=surface,
    )


def _compact_gravel_geometry():
    return parking._ParkingGeometry(
        osm_key="way/compact-gravel",
        outer=((19.0, 21.0), (31.0, 21.0), (31.0, 29.0), (19.0, 29.0)),
        holes=(),
        centre_x=25.0,
        centre_z=25.0,
        half_length=6.0,
        half_width=4.0,
        ux=1.0,
        uz=0.0,
        px=0.0,
        pz=1.0,
        surface="gravel",
    )


def _spec():
    return SimpleNamespace(cells=1, cell_size=50.0)


def test_paved_detail_adds_clear_stop_lines_and_keeps_outside_untouched() -> None:
    spec = _spec()
    geometry = _geometry("paved")
    base = Image.new("RGB", (128, 128), (66, 67, 66))

    rendered = apply_parking_visual_detail(
        parking,
        base,
        cell_index=0,
        geometries=(geometry,),
        spec=spec,
    )
    actual = np.asarray(rendered)
    assert np.array_equal(actual[0, 0], np.asarray(base)[0, 0])

    world_x, world_z, pixel_scale = _world_grid(cell_index=0, spec=spec, size=128)
    lateral = world_z - geometry.centre_z
    edge_depth = min(5.2, geometry.half_width * 0.42)
    inner_edge = geometry.half_width - edge_depth
    line_half = max(0.18, pixel_scale * 0.72)
    stop_lines = (
        (world_x >= 3.0)
        & (world_x <= 47.0)
        & (world_z >= 3.0)
        & (world_z <= 47.0)
        & (np.abs(np.abs(lateral) - inner_edge) <= line_half)
    )

    assert np.count_nonzero(stop_lines) > 0
    assert float(actual[stop_lines].mean()) > 150.0


def test_gravel_detail_has_compacted_aisle_ruts_and_parking_row_wear() -> None:
    spec = _spec()
    geometry = _geometry("gravel")
    base = Image.new("RGB", (128, 128), (104, 96, 80))

    rendered = apply_parking_visual_detail(
        parking,
        base,
        cell_index=0,
        geometries=(geometry,),
        spec=spec,
    )
    actual = np.asarray(rendered, dtype=np.float32)
    source = np.asarray(base, dtype=np.float32)
    assert np.array_equal(actual[0, 0], source[0, 0])
    assert np.count_nonzero(np.any(actual != source, axis=2)) > 1000

    world_x, world_z, pixel_scale = _world_grid(cell_index=0, spec=spec, size=128)
    lateral = world_z - geometry.centre_z
    drive_half = min(4.2, max(0.82, geometry.half_width * 0.28), max(0.72, geometry.half_width * 0.62))
    rut_offset = max(0.36, drive_half * 0.43)
    rut_half = max(0.20, pixel_scale * 0.72)
    lot = (
        (world_x >= 3.0)
        & (world_x <= 47.0)
        & (world_z >= 3.0)
        & (world_z <= 47.0)
    )
    ruts = lot & (np.abs(np.abs(lateral) - rut_offset) <= rut_half)
    aisle_centre = lot & (np.abs(lateral) <= 0.35)

    assert np.count_nonzero(ruts) > 0
    assert np.count_nonzero(aisle_centre) > 0
    assert float(actual[ruts].mean()) < float(actual[aisle_centre].mean()) - 2.0
    assert float(actual.std()) > float(source.std())


def test_compact_gravel_lot_gets_structured_aisle_and_ruts() -> None:
    """Lots below the old 16 x 10 m cutoff must not fall back to plain gravel."""
    spec = _spec()
    geometry = _compact_gravel_geometry()
    base = Image.new("RGB", (128, 128), (104, 96, 80))

    rendered = apply_parking_visual_detail(
        parking,
        base,
        cell_index=0,
        geometries=(geometry,),
        spec=spec,
    )
    actual = np.asarray(rendered, dtype=np.float32)
    source = np.asarray(base, dtype=np.float32)
    world_x, world_z, pixel_scale = _world_grid(cell_index=0, spec=spec, size=128)
    lot = (
        (world_x >= 19.0)
        & (world_x <= 31.0)
        & (world_z >= 21.0)
        & (world_z <= 29.0)
    )
    lateral = world_z - geometry.centre_z
    drive_half = min(4.2, max(0.82, geometry.half_width * 0.28), max(0.72, geometry.half_width * 0.62))
    rut_offset = max(0.36, drive_half * 0.43)
    rut_half = max(0.20, pixel_scale * 0.72)
    ruts = lot & (np.abs(np.abs(lateral) - rut_offset) <= rut_half)
    aisle_centre = lot & (np.abs(lateral) <= 0.30)

    assert np.count_nonzero(np.any(actual[lot] != source[lot], axis=1)) > 300
    assert np.count_nonzero(ruts) > 0
    assert np.count_nonzero(aisle_centre) > 0
    assert float(actual[ruts].mean()) < float(actual[aisle_centre].mean()) - 3.0


def test_visual_detail_bumps_parking_cache_schema() -> None:
    assert DETAIL_CACHE_SCHEMA >= 3
