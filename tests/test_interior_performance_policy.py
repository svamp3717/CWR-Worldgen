from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import parallel_assets
from cwr_worldgen import procedural_buildings as pb


def _key(*, interiors: bool) -> pb.BuildingVariantKey:
    return pb.BuildingVariantKey(
        family="residential",
        roof_style="flat",
        width_m=10.0,
        length_m=8.0,
        height_m=6.0,
        interiors=interiors,
        second_storey=interiors,
        facade_storeys=2,
        window_width_m=1.2,
        window_height_m=1.35,
        window_sill_height_m=0.85,
        window_edge_margin_m=0.7,
        window_bay_spacing_m=3.6,
        door_width_m=0.95,
        door_height_m=2.1,
    )


def _task(*, interiors: bool):
    return SimpleNamespace(
        key=_key(interiors=interiors),
        cache_enabled=False,
        cache_refresh=False,
        cache_path=None,
    )


def test_interior_batches_parallelize_at_eight_without_lowering_exterior_cutoff(monkeypatch) -> None:
    calls: list[int] = []

    def fake_worker(task):
        return ("serial", task.key.interiors)

    # The scheduler deliberately recognizes the real building worker by stable
    # module/name identity so tests do not need to generate eight actual P3Ds.
    fake_worker.__module__ = "cwr_worldgen.procedural_buildings"
    fake_worker.__name__ = "_write_building_asset_task"

    monkeypatch.setattr(parallel_assets, "asset_worker_count", lambda _count: 2)

    def fake_process_map(worker, tasks, workers):
        calls.append(len(tasks))
        return [("parallel", task.key.interiors) for task in tasks]

    monkeypatch.setattr(parallel_assets, "_process_map", fake_process_map)

    interior = pb.process_asset_tasks(fake_worker, [_task(interiors=True) for _ in range(8)])
    assert calls == [8]
    assert all(result[0] == "parallel" for result in interior)

    calls.clear()
    exterior = pb.process_asset_tasks(fake_worker, [_task(interiors=False) for _ in range(8)])
    assert calls == []
    assert all(result[0] == "serial" for result in exterior)


def test_repeated_interior_door_and_window_layouts_hit_worker_caches() -> None:
    key = _key(interiors=True)

    door_cache = getattr(pb._door_dimensions, "cache_info", None)
    assert callable(door_cache)
    before_door = door_cache()
    first_door = pb._door_dimensions(key)
    second_door = pb._door_dimensions(key)
    after_door = door_cache()
    assert first_door == second_door
    assert after_door.hits >= before_door.hits + 1

    window_cache = getattr(pb._interior_window_openings, "__name__", "")
    assert window_cache == "memoized_window_openings"
    first_windows = pb._interior_window_openings(
        key, -5.0, 5.0, 5.8,
        ground_exclusions=((-0.8, 0.8),),
    )
    second_windows = pb._interior_window_openings(
        key, -5.0, 5.0, 5.8,
        ground_exclusions=((-0.8, 0.8),),
    )
    assert first_windows is second_windows


def test_polygon_shape_and_roof_mesh_are_memoized_per_worker() -> None:
    key = _key(interiors=True)
    key = pb.replace(
        key,
        roof_style="gabled",
        footprint_vertices=((-5.0, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0)),
        entrance_edge=0,
        entrance_fraction=0.5,
    )

    shape_cache = getattr(pb._polygon_native_shape, "cache_info", None)
    roof_cache = getattr(pb._polygon_native_roof_mesh, "cache_info", None)
    assert callable(shape_cache)
    assert callable(roof_cache)

    shape_a = pb._polygon_native_shape(key)
    shape_b = pb._polygon_native_shape(key)
    assert shape_a is shape_b

    roof_a = pb._polygon_native_roof_mesh(key, 35.0)
    roof_b = pb._polygon_native_roof_mesh(key, 35.0)
    assert roof_a is roof_b
