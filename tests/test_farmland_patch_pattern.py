"""Seeded, irregular field patches instead of axis-aligned farmland bands."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cwr_worldgen.osm import (
    BboxProjection, GeoPolygon, OsmDataset, OsmPolygonFeature, rasterize_osm,
)
from cwr_worldgen.surface_pass import (
    MATERIAL_INDEX, _farmland_patch_light, build_surface_pass,
)


def _patches(seed: str, feature: str, width: int = 4) -> np.ndarray:
    n = 96
    coords_x = np.tile(np.arange(n, dtype=np.int32), n)
    coords_z = np.repeat(np.arange(n, dtype=np.int32), n)
    return _farmland_patch_light(
        coords_x, coords_z, width=width, seed=seed, feature_id=feature,
    ).reshape(n, n)


def test_field_materials_make_coherent_irregular_clusters_not_rows() -> None:
    patches = _patches("world-one", "way/55")
    assert np.array_equal(patches, _patches("world-one", "way/55"))
    assert not np.array_equal(patches, _patches("world-two", "way/55"))
    assert not np.array_equal(patches, _patches("world-one", "way/56"))
    assert 0.10 < patches.mean() < 0.90

    # Old implementation used one orientation and phase for all of a field,
    # making complete parallel bands. New neighboring rows must be distinct.
    assert len({tuple(row) for row in patches[10:80:3]}) >= 14

    # Smooth 2-D noise should form clusters: most horizontal and vertical
    # adjacent cells share their terrain material.
    assert np.mean(patches[:, :-1] == patches[:, 1:]) > 0.72
    assert np.mean(patches[:-1, :] == patches[1:, :]) > 0.72


def test_field_patch_width_controls_typical_cluster_size() -> None:
    small = _patches("same-world", "way/55", width=2)
    large = _patches("same-world", "way/55", width=10)
    small_changes = np.count_nonzero(small[:, 1:] != small[:, :-1])
    small_changes += np.count_nonzero(small[1:, :] != small[:-1, :])
    large_changes = np.count_nonzero(large[:, 1:] != large[:, :-1])
    large_changes += np.count_nonzero(large[1:, :] != large[:-1, :])
    assert large_changes < small_changes * 0.7


def test_surface_pass_respects_field_boundaries_with_irregular_patches() -> None:
    cells = 128
    projection = BboxProjection.create((0.0, 0.0, 0.06, 0.06), 6400.0)
    corners = [(500, 500), (5800, 500), (5800, 5800), (500, 5800)]
    ring = tuple(projection.to_latlon(point) for point in (*corners, corners[0]))
    farmland = OsmPolygonFeature(
        "way/55", {"landuse": "farmland"}, (GeoPolygon(ring),),
    )
    dataset = OsmDataset(
        source_generator="field-patches-test", element_count=1,
        coastlines=(), water=(), forests=(),
        farmland=(farmland,), urban=(), roads=(),
    )
    raster = rasterize_osm(dataset, projection, cells=cells, include_minor_roads=True)
    spec = SimpleNamespace(
        cells=cells, cell_size=50.0, include_minor_roads=True,
        sea_level=0.0, rock_slope_degrees=30.0, deterministic_seed="field-test",
        surface_shoreline_wet_cells=1, surface_shoreline_sand_cells=2,
        surface_transition_cells=2, surface_forest_edge_cells=1,
        surface_farmland_strip_cells=4, surface_road_shoulder_metres=5.0,
        surface_dirt_blend_metres=6.0, surface_steep_slope_degrees=38.0,
        surface_colour_reference_path=None, surface_colour_reference_strength=0.0,
    )
    first = build_surface_pass(
        dataset, projection, raster, [60.0] * (cells * cells),
        [3.0] * (cells * cells), spec,
    )
    assert first.farmland_light_cells > 0
    assert first.farmland_dark_cells > 0
    assert first.field_boundary_cells > 0
    surface = np.asarray(first.indices, dtype=np.int16).reshape(cells, cells)
    farm = np.isin(surface, [
        MATERIAL_INDEX["a"], MATERIAL_INDEX["b"], MATERIAL_INDEX["c"],
    ])
    assert farm[0, 0] == 0  # no material leaks outside the mapped polygon

    # Entire horizontal/vertical bands previously had identical divisions
    # across all rows of the field. The patch pattern must vary in both axes.
    interior = surface[25:100, 25:100]
    assert len({tuple(row) for row in interior[::4]}) >= 8
    assert len({tuple(column) for column in interior.T[::4]}) >= 8

    second = build_surface_pass(
        dataset, projection, raster, [60.0] * (cells * cells),
        [3.0] * (cells * cells), spec,
    )
    assert first.indices == second.indices
