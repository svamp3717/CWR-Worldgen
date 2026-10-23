"""Keep all safe OSM trees and all grounded forest scatter candidates.

The old non-advisory shortcuts removed every second mapped tree and
stopped after one accepted candidate per geographic individual-tree cell.
These tests generate actual WRP-ready world objects, not merely inspect
implementation strings.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    OsmDataset,
    OsmPointFeature,
    OsmRaster,
    _geographic_lattice_identity,
    generate_world_objects,
)


def _dataset(**overrides) -> OsmDataset:
    defaults = {
        "source_generator": "tree-sampling-regression",
        "element_count": 0,
        "coastlines": (),
        "water": (),
        "forests": (),
        "farmland": (),
        "urban": (),
        "roads": (),
    }
    defaults.update(overrides)
    return OsmDataset(**defaults)


def _raster(cells: int, *, forest: bool = False, water: bool = False) -> OsmRaster:
    def fill(v: bool):
        return (v,) * (cells * cells)
    return OsmRaster(
        cells=cells,
        water=fill(water),
        forest=fill(forest),
        farmland=fill(False),
        urban=fill(False),
        roads=fill(False),
        buildings=fill(False),
        high_resolution=cells,
        coastline_seed_count=0,
    )


def _spec(cells: int, **overrides) -> _Milestone9PlayabilitySpec:
    attrs = {
        "name": "tree_sampling_regression",
        "heightmap_path": Path("unused.png"),
        "bbox": (0.0, 0.0, 0.01, 0.01),
        "cells": cells,
        "cell_size": 25.0,
        "max_road_objects": 0,
        "max_buildings": 0,
        "max_forest_objects": 500_000,
        "advisory_object_limits": True,
        "forest_tree_spacing": 50.0,
        "forest_single_tree_spacing": 45.0,
        "forest_single_tree_enabled": True,
        "forest_gap_infill_enabled": False,
        "forest_undergrowth_enabled": False,
        "forest_border_enabled": False,
        "steep_hill_bushes_enabled": False,
        "ditch_grass_enabled": False,
        "rocky_forest_fallback_enabled": False,
        "rural_vegetation_enabled": False,
        "meadow_grass_enabled": False,
        "wetland_reeds_enabled": False,
        "strict_assets": False,
    }
    attrs.update(overrides)
    return _Milestone9PlayabilitySpec(**attrs)


def test_every_mapped_osm_tree_is_evaluated_instead_of_skipping_half() -> None:
    cells = 16
    spec = _spec(cells, max_forest_objects=0, maximum_mapped_tree_objects=2)
    projection = BboxProjection.create(spec.bbox, spec.world_size)
    locations = ((45.0, 45.0), (110.0, 55.0), (170.0, 120.0),
                 (240.0, 250.0), (340.0, 310.0))
    mapped = tuple(
        OsmPointFeature(
            f"node/{i:02d}",
            {"natural": "tree", "leaf_type": "broadleaved" if i % 2 else "needleleaved"},
            projection.to_latlon((x, z)),
        )
        for i, (x, z) in enumerate(locations)
    )
    dataset = _dataset(individual_trees=mapped, element_count=len(mapped))
    raster = _raster(cells)
    heights = (10.0,) * (cells * cells)

    result = generate_world_objects(
        dataset, projection, raster, heights, spec, include_roads=False,
    )
    assert result.mapped_tree_objects == len(mapped)
    assert {
        (round(obj.x, 3), round(obj.z, 3))
        for obj in result.objects
    } == {(round(x, 3), round(z, 3)) for x, z in locations}
    assert result.mapped_tree_rejections == 0
    assert len({obj.object_id for obj in result.objects}) == len(mapped)

    # Disabling the category still works. Direct non-advisory callers also
    # retain their explicitly requested count budget; only sampling is gone.
    disabled = _spec(cells, max_forest_objects=0, maximum_mapped_tree_objects=0)
    assert generate_world_objects(
        dataset, projection, raster, heights, disabled, include_roads=False,
    ).mapped_tree_objects == 0
    bounded = _spec(
        cells, max_forest_objects=0, maximum_mapped_tree_objects=3,
        advisory_object_limits=False,
    )
    assert generate_world_objects(
        dataset, projection, raster, heights, bounded, include_roads=False,
    ).mapped_tree_objects == 3


def test_forest_scatter_uses_more_than_one_candidate_per_geographic_cell() -> None:
    cells = 16
    spec = _spec(cells, maximum_forest_single_tree_objects=1000)
    projection = BboxProjection.create(spec.bbox, spec.world_size)
    dataset = _dataset()
    raster = _raster(cells, forest=True)
    heights = (15.0,) * (cells * cells)

    first = generate_world_objects(
        dataset, projection, raster, heights, spec, include_roads=False,
    )
    again = generate_world_objects(
        dataset, projection, raster, heights, spec, include_roads=False,
    )
    assert first.objects == again.objects
    assert first.forest_single_tree_objects > 0
    singles = [
        obj for obj in first.objects
        if obj.model_path.casefold() in {
            r"data3d\str smrk_medium.p3d",
            r"data3d\str smrk.p3d",
        }
    ]
    assert singles, "no rooted forest scatter trees were generated"
    counts = Counter(
        _geographic_lattice_identity(
            projection, obj.x, obj.z, spec.forest_single_tree_spacing,
        )
        for obj in singles
    )
    # The removed shortcut made this impossible for an accepted forest cell.
    assert max(counts.values()) >= 2
    assert first.vegetation_audit_violations == 0
