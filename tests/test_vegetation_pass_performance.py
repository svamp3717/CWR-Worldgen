"""Determinism and progress checks for large-world vegetation passes."""

from types import SimpleNamespace

from cwr_worldgen import osm


def _border_fixture():
    polygon = SimpleNamespace(
        outer=((20.0, 20.0), (180.0, 20.0), (180.0, 180.0),
               (20.0, 180.0), (20.0, 20.0)),
        holes=(),
    )
    feature = SimpleNamespace(osm_key="way/forest-1", polygons=(polygon,))
    dataset = SimpleNamespace(forests=(feature,))
    projection = SimpleNamespace(to_world=lambda point: point)
    return dataset, projection


def test_forest_border_progress_preserves_sorted_candidate_order() -> None:
    dataset, projection = _border_fixture()
    args = dict(spacing=18.0, inset=5.0, seed="same-world-seed")
    reference = osm._forest_border_candidates(dataset, projection, **args)
    updates = []
    monitored = osm._forest_border_candidates(
        dataset, projection, **args, progress_callback=updates.append,
    )
    assert monitored == reference
    assert monitored
    assert any("Sampling mapped forest boundaries" in item for item in updates)
    assert any("Sorting forest-border candidates" in item for item in updates)


def test_steep_bush_single_mask_index_is_equivalent_to_four_scalar_checks() -> None:
    cells, size = 7, 175.0
    total = cells * cells
    raster = SimpleNamespace(
        forest=tuple(i % 5 != 0 for i in range(total)),
        water=tuple(i % 7 == 0 for i in range(total)),
        roads=tuple(i % 11 == 0 for i in range(total)),
        buildings=tuple(i % 13 == 0 for i in range(total)),
    )
    for x in (0.0, 14.5, 48.9, 97.2, 174.99):
        for z in (0.0, 20.5, 70.1, 148.0, 174.99):
            index = min(cells - 1, int(z / size * cells)) * cells
            index += min(cells - 1, int(x / size * cells))
            forest = osm._mask_at(raster.forest, cells, size, x, z)
            blocked = (
                osm._mask_at(raster.water, cells, size, x, z)
                or osm._mask_at(raster.roads, cells, size, x, z)
                or osm._mask_at(raster.buildings, cells, size, x, z)
            )
            assert bool(raster.forest[index]) == bool(forest)
            assert bool(
                raster.water[index]
                or raster.roads[index]
                or raster.buildings[index]
            ) == bool(blocked)
