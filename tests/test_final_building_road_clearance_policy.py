from types import SimpleNamespace

from cwr_worldgen import final_building_road_clearance_policy as policy
from cwr_worldgen.model import WorldObject
from cwr_worldgen.osm import BuildingPlacementPlan


def _spec(**overrides):
    values = dict(
        road_segment_length=25.0,
        cells=64,
        cell_size=2.0,
        world_size=128.0,
        building_foundation_maximum_depth=2.5,
        building_ground_clearance=0.05,
        building_foundation_safety=0.20,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _raster(spec):
    size = spec.cells * spec.cells
    return SimpleNamespace(water=(False,) * size)


def _plan(key, x, z, width=6.0, length=8.0):
    half_w = width * 0.5
    half_l = length * 0.5
    polygon = (
        (x - half_w, z - half_l),
        (x + half_w, z - half_l),
        (x + half_w, z + half_l),
        (x - half_w, z + half_l),
    )
    return BuildingPlacementPlan(
        osm_key=key,
        geometry_index=0,
        geometry_kind="polygon",
        x=x,
        z=z,
        heading_degrees=0.0,
        model_path=r"test\building.p3d",
        support_polygon=polygon,
        building_family="residential",
    )


def _report(*objects):
    return SimpleNamespace(objects=tuple(objects))


def _flat(spec):
    return (0.0,) * (spec.cells * spec.cells)


def test_small_building_on_final_straight_is_moved():
    spec = _spec()
    plan = _plan("way/1", 50.0, 50.0, width=6.0, length=8.0)
    road = WorldObject(1, r"O\Road\sil25.p3d", 50.0, 0.035, 50.0, 0.0)

    resolved, report = policy.resolve_final_building_road_conflicts(
        (plan,), _report(road), _flat(spec), _raster(spec), spec
    )

    assert len(resolved) == 1
    assert report.conflicted == 1
    assert report.moved == 1
    assert report.rejected == 0
    assert resolved[0].road_nudged
    assert abs(resolved[0].x - plan.x) > 1.0

    primitives = policy._road_primitives(_report(road), _flat(spec), spec)
    conflicts, _checks = policy._conflicts(
        resolved[0].support_polygon, policy._RoadPrimitiveIndex(primitives)
    )
    assert conflicts == ()


def test_clear_building_is_unchanged():
    spec = _spec()
    plan = _plan("way/clear", 90.0, 90.0)
    road = WorldObject(1, r"O\Road\sil25.p3d", 20.0, 0.035, 20.0, 0.0)

    resolved, report = policy.resolve_final_building_road_conflicts(
        (plan,), _report(road), _flat(spec), _raster(spec), spec
    )

    assert resolved == (plan,)
    assert report.conflicted == 0
    assert report.moved == 0
    assert report.rejected == 0


def test_relocation_uses_other_side_when_neighbor_blocks_first_choice():
    spec = _spec()
    road = WorldObject(1, r"O\Road\sil25.p3d", 50.0, 0.035, 50.0, 0.0)
    conflicted = _plan("way/move", 50.0, 50.0)
    # For a north/south road the deterministic first equal-length lateral escape
    # is toward -X. Occupy that side so the resolver must use +X.
    neighbor = _plan("way/neighbor", 39.0, 50.0, width=10.0, length=12.0)

    resolved, report = policy.resolve_final_building_road_conflicts(
        (conflicted, neighbor), _report(road), _flat(spec), _raster(spec), spec
    )

    moved = next(plan for plan in resolved if plan.osm_key == "way/move")
    assert moved.x > conflicted.x
    assert report.moved >= 1


def test_building_is_rejected_when_bounded_safe_search_has_no_space():
    spec = _spec(world_size=40.0, cells=40, cell_size=1.0)
    road = WorldObject(1, r"O\Road\sil25.p3d", 8.0, 0.035, 20.0, 0.0)
    conflicted = _plan("way/trapped", 8.0, 20.0, width=6.0, length=8.0)
    blocker = _plan("way/blocker", 24.0, 20.0, width=16.0, length=30.0)

    resolved, report = policy.resolve_final_building_road_conflicts(
        (conflicted, blocker), _report(road), _flat(spec), _raster(spec), spec
    )

    assert all(plan.osm_key != "way/trapped" for plan in resolved)
    assert report.rejected == 1


def test_grade_separated_road_does_not_move_ground_building():
    spec = _spec()
    plan = _plan("way/ground", 50.0, 50.0)
    elevated = WorldObject(1, r"O\Road\sil25.p3d", 50.0, 8.0, 50.0, 0.0)

    resolved, report = policy.resolve_final_building_road_conflicts(
        (plan,), _report(elevated), _flat(spec), _raster(spec), spec
    )

    assert resolved == (plan,)
    assert report.road_primitives == 0
    assert report.conflicted == 0


def test_stock_curve_and_junction_generate_clearance_primitives():
    spec = _spec()
    curve = WorldObject(1, r"O\Road\sil10 25.p3d", 50.0, 0.035, 50.0, 0.0)
    junction = WorldObject(
        2, r"O\Road\kr_new_sil_asf_t.p3d", 70.0, 0.035, 70.0, 0.0
    )

    assert len(policy._road_object_primitives(curve, spec)) == 1
    assert len(policy._road_object_primitives(junction, spec)) == 3


def test_spatial_index_keeps_far_roads_out_of_building_checks():
    spec = _spec(world_size=5000.0, cells=64, cell_size=80.0)
    plan = _plan("way/local", 50.0, 50.0)
    objects = [
        WorldObject(1, r"O\Road\sil25.p3d", 50.0, 0.035, 50.0, 0.0)
    ]
    objects.extend(
        WorldObject(
            index + 2,
            r"O\Road\sil25.p3d",
            1000.0 + (index % 50) * 40.0,
            0.035,
            1000.0 + (index // 50) * 40.0,
            0.0,
        )
        for index in range(2000)
    )

    resolved, report = policy.resolve_final_building_road_conflicts(
        (plan,), _report(*objects), _flat(spec), _raster(spec), spec
    )

    assert len(resolved) == 1
    assert report.moved == 1
    assert report.nearby_road_checks < 20


def test_progress_reports_move_and_rejection_counters():
    spec = _spec()
    plan = _plan("way/progress", 50.0, 50.0)
    road = WorldObject(1, r"O\Road\sil25.p3d", 50.0, 0.035, 50.0, 0.0)
    events = []

    policy.resolve_final_building_road_conflicts(
        (plan,),
        _report(road),
        _flat(spec),
        _raster(spec),
        spec,
        progress_callback=lambda value, text: events.append((value, text)),
    )

    assert events
    assert all(value == 52 for value, _text in events)
    assert any("moved" in text and "rejected" in text for _value, text in events)
