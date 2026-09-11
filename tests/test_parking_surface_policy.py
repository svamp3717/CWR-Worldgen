from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
from PIL import Image

from cwr_worldgen import generator
from cwr_worldgen import runway_exact_background_policy as exact
from cwr_worldgen import runway_surface_policy as runway
from cwr_worldgen.parking_surface_policy import (
    _parking_cells,
    _parking_geometries,
    _render_parking_cell,
    apply_parking_textures,
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


def _single_dataset():
    return SimpleNamespace(
        sites=(_feature("way/paved", _polygon(20.0, 20.0, 80.0, 55.0)),),
        roads=(_road("way/asphalt-road", 10.0, surface="asphalt"),),
    )


def _spec(tmp_path=None):
    return SimpleNamespace(
        name="parkingworld",
        cells=4,
        cell_size=50.0,
        ground_texture_profile="generated",
        deterministic_seed="parking-test",
        cache_enabled=True,
        cache_refresh=False,
        cache_dir=(tmp_path / "cache") if tmp_path is not None else None,
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
    spec = _spec()
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


def test_parking_texture_generation_reuses_persistent_cache(tmp_path, monkeypatch) -> None:
    spec = _spec(tmp_path)
    dataset = _single_dataset()
    projection = _IdentityProjection()
    base = Image.new("RGB", (128, 128), (68, 86, 52))

    monkeypatch.setattr(
        generator,
        "_material_definitions",
        lambda _spec: (SimpleNamespace(code="g", name="grass", colour=(68, 86, 52)),),
    )
    monkeypatch.setattr(
        generator,
        "_ground_texture_paths",
        lambda _spec: (r"parkingworld\data\g.paa",),
    )
    monkeypatch.setattr(exact, "_load_exact_texture", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runway, "_background_texture", lambda *_args, **_kwargs: base.copy())

    source_dir = tmp_path / "parkingworld"
    source_dir.mkdir()
    texture_indices = (1,) * (spec.cells * spec.cells)
    texture_paths = (r"parkingworld\data\d.paa", r"parkingworld\data\g.paa")

    first_indices, first_paths, first_generated = apply_parking_textures(
        source_dir, dataset, projection, spec, texture_indices, texture_paths
    )
    first_report = json.loads((source_dir / "parking-lot-textures.json").read_text())
    assert first_generated
    assert first_report["cache_misses"] == 4
    assert first_report["paved_parking_count"] == 1
    assert any(first_indices[index] >= len(texture_paths) for index in (0, 1, 4, 5))

    second_indices, second_paths, second_generated = apply_parking_textures(
        source_dir, dataset, projection, spec, texture_indices, texture_paths
    )
    second_report = json.loads((source_dir / "parking-lot-textures.json").read_text())
    assert second_report["cache_hits"] == 4
    assert second_report["cache_misses"] == 0
    assert second_indices == first_indices
    assert second_paths == first_paths
    assert second_generated == first_generated
