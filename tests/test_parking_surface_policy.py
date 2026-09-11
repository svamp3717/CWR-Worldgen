from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from cwr_worldgen.parking_surface_policy import (
    _parking_cells,
    _parking_geometries,
    _render_parking_cell,
)


class _IdentityProjection:
    @staticmethod
    def to_world(point):
        return float(point[0]), float(point[1])


def _polygon(x0: float, z0: float, x1: float, z1: float):
    return SimpleNamespace(
        outer=((x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0)),
        holes=(),
    )


def _feature(key: str, polygon):
    return SimpleNamespace(
        osm_key=key,
        tags={"site": "parking"},
        polygons=(polygon,),
    )


def _road(key: str, z: float, *, surface: str):
    return SimpleNamespace(
        osm_key=key,
        tags={"highway": "primary", "surface": surface},
        points=((0.0, z), (200.0, z)),
    )


def _dataset():
    return SimpleNamespace(
        sites=(
            _feature("way/paved", _polygon(20.0, 20.0, 80.0, 55.0)),
            _feature("way/gravel", _polygon(20.0, 135.0, 80.0, 175.0)),
        ),
        roads=(
            _road("way/asphalt-road", 10.0, surface="asphalt"),
            _road("way/gravel-road", 190.0, surface="gravel"),
        ),
    )


def test_parking_surface_follows_nearest_road() -> None:
    geometries = _parking_geometries(_dataset(), _IdentityProjection())
    surfaces = {geometry.osm_key: geometry.surface for geometry in geometries}
    assert surfaces == {
        "way/paved": "paved",
        "way/gravel": "gravel",
    }


def test_parking_cells_follow_true_osm_polygons() -> None:
    geometries = _parking_geometries(_dataset(), _IdentityProjection())
    cells = _parking_cells(geometries, cells=4, cell_size=50.0)
    assert {0, 1}.issubset(cells)
    assert {8, 9, 12, 13}.intersection(cells)


def test_paved_and_gravel_render_over_background_without_replacing_outside() -> None:
    geometries = _parking_geometries(_dataset(), _IdentityProjection())
    spec = SimpleNamespace(cells=4, cell_size=50.0)
    base = Image.new("RGB", (128, 128), (68, 86, 52))

    paved = next(item for item in geometries if item.surface == "paved")
    rendered_paved = _render_parking_cell(
        cell_index=0,
        geometries=(paved,),
        spec=spec,
        base=base,
    )
    source = np.asarray(base)
    actual_paved = np.asarray(rendered_paved)
    assert np.array_equal(actual_paved[0, 0], source[0, 0])
    assert np.count_nonzero(np.any(actual_paved != source, axis=2)) > 0
    # Painted parking bays should create some substantially brighter pixels.
    assert int(actual_paved.max()) > 150

    gravel = next(item for item in geometries if item.surface == "gravel")
    rendered_gravel = _render_parking_cell(
        cell_index=8,
        geometries=(gravel,),
        spec=spec,
        base=base,
    )
    actual_gravel = np.asarray(rendered_gravel)
    assert np.count_nonzero(np.any(actual_gravel != source, axis=2)) > 0
    assert float(actual_gravel[:, :, 0].mean()) > float(actual_paved[:, :, 0].mean())
