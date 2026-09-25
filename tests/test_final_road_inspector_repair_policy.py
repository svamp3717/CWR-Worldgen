from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import final_road_inspector_repair_policy as repair
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen import road_inspector as inspector
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


def _spec():
    return SimpleNamespace(
        name="repair_test",
        procedural_paved_road_fallback=True,
        cells=64,
        cell_size=10.0,
        road_segment_length=25.0,
        max_road_objects=10000,
        advisory_object_limits=True,
        include_minor_roads=False,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        procedural_gravel_roads=True,
    )


def _report(*objects: WorldObject, caps: int = 0) -> RoadFitReport:
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
        junction_cap_objects=caps,
    )


def test_final_repair_replaces_irreducible_paved_seam_with_generated_model() -> None:
    spec = _spec()
    report = _report(
        WorldObject(1, r"o\road\sil25.p3d", 100.0, 0.035, 100.0, 0.0, 0.0),
        WorldObject(2, r"o\road\sil25.p3d", 100.0, 0.035, 125.0, 5.0, 0.0),
    )

    result = repair.repair_final_road_geometry(
        report,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )

    assert len(result.objects) == 1
    obj = result.objects[0]
    assert infrastructure.is_generated_paved_road_model(obj.model_path)
    assert math.isclose(
        obj.y + infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES,
        0.035,
        abs_tol=1.0e-6,
    )
    post = inspector.inspect_road_objects(
        result.objects,
        world_name=spec.name,
        topology_checks=False,
    )
    assert post.paved_stock_repairs == ()
    assert post.paved_replacements == ()
    assert repair._LAST_REPAIR_REPORT is not None
    assert repair._LAST_REPAIR_REPORT.generated_regions == 1


def test_axial_stock_overlap_remains_a_stock_refit_diagnostic() -> None:
    spec = _spec()
    report = _report(
        WorldObject(1, r"o\road\sil25.p3d", 100.0, 0.035, 100.0, 0.0, 0.0),
        WorldObject(2, r"o\road\sil25.p3d", 100.0, 0.035, 124.5, 0.0, 0.0),
    )

    result = repair.repair_final_road_geometry(
        report,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )

    assert result is report
    assert all(
        not infrastructure.is_generated_paved_road_model(obj.model_path)
        for obj in result.objects
    )
    assert repair._LAST_REPAIR_REPORT is not None
    assert repair._LAST_REPAIR_REPORT.generated_regions == 0


def test_explicitly_protected_bridge_terminal_region_is_never_replaced() -> None:
    spec = _spec()
    report = _report(
        WorldObject(1, r"o\road\sil25.p3d", 100.0, 0.035, 100.0, 0.0, 0.0),
        WorldObject(2, r"o\road\sil25.p3d", 100.0, 0.035, 125.0, 5.0, 0.0),
    )

    result = repair.repair_final_road_geometry(
        report,
        [0.0] * (spec.cells * spec.cells),
        spec,
        protected_object_ids=(1,),
    )

    assert result is report
    assert repair._LAST_REPAIR_REPORT is not None
    assert repair._LAST_REPAIR_REPORT.generated_regions == 0
    assert repair._LAST_REPAIR_REPORT.unresolved_generated_regions == 1


def test_protected_junction_prefix_is_never_replaced() -> None:
    spec = _spec()
    report = _report(
        WorldObject(1, r"o\road\sil25.p3d", 100.0, 0.060, 100.0, 0.0, 0.0),
        WorldObject(2, r"o\road\sil25.p3d", 100.0, 0.060, 125.0, 5.0, 0.0),
        caps=1,
    )

    result = repair.repair_final_road_geometry(
        report,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )

    assert result is report
    assert tuple(obj.object_id for obj in result.objects) == (1, 2)
    assert repair._LAST_REPAIR_REPORT is not None
    assert repair._LAST_REPAIR_REPORT.unresolved_generated_regions == 1


def test_stock_repair_plan_reconstructs_clean_vanilla_curve_sequence() -> None:
    spec = _spec()
    start = (100.0, 100.0)
    curve_end, end_heading = inspector._stock_arc_step(
        start, 0.0, 1, 25
    )
    direction = (
        math.sin(math.radians(end_heading)),
        math.cos(math.radians(end_heading)),
    )
    end = (
        curve_end[0] + direction[0] * 25.0,
        curve_end[1] + direction[1] * 25.0,
    )
    plan = inspector.PavedStockRepairPlan(
        plan_id="SR-test",
        family="sil",
        replace_object_ids=(10, 11),
        source_models=(r"o\road\sil25.p3d",),
        issue_ids=("RI-test",),
        stock_models=(r"o\road\sil10 25.p3d", r"o\road\sil25.p3d"),
        start=start,
        end=end,
        start_heading_degrees=0.0,
        end_heading_degrees=end_heading,
        turn_sign=1,
        first_turns=1,
        first_radius=25,
        middle_units=0,
        counter_turns=0,
        counter_radius=25,
        merge_nominal=25,
        maximum_path_deviation_metres=0.0,
        final_length_error_metres=0.0,
        maximum_join_angle_error_degrees=0.0,
    )

    objects, next_id = repair._stock_repair_objects(
        plan,
        20,
        [0.0] * (spec.cells * spec.cells),
        spec,
        vertical_offset=0.035,
    )

    assert next_id == 22
    assert tuple(obj.model_path for obj in objects) == plan.stock_models
    result = inspector.inspect_road_objects(
        objects,
        world_name=spec.name,
        topology_checks=False,
    )
    assert result.issues == ()




def test_final_guard_restores_missing_generated_paved_t_hub() -> None:
    spec = _spec()
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, spec.cells * spec.cell_size)
    centre = (320.0, 320.0)

    def ll(point):
        return projection.to_latlon(point)

    dataset = OsmDataset(
        source_generator="final-paved-hub-guard",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            OsmLineFeature(
                "way/main",
                {"highway": "residential"},
                (ll((320.0, 180.0)), ll(centre), ll((320.0, 460.0))),
            ),
            OsmLineFeature(
                "way/branch",
                {"highway": "residential"},
                (ll(centre), ll((470.0, 320.0))),
            ),
        ),
    )
    # Simulate the exact failure visible in-game: the approach chains were
    # trimmed to the 6.25 m hub connector radius, but the cap itself vanished.
    report = _report(
        WorldObject(1, r"o\road\sil6.p3d", 320.0, 0.035, 329.375, 0.0, 0.0),
        WorldObject(2, r"o\road\sil6.p3d", 320.0, 0.035, 310.625, 0.0, 0.0),
        WorldObject(3, r"o\road\sil6.p3d", 329.375, 0.035, 320.0, 90.0, 0.0),
    )

    result = repair.ensure_final_paved_junction_hubs(
        report,
        dataset,
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )

    assert len(result.objects) == len(report.objects) + 1
    hub = result.objects[-1]
    assert hub.model_path.startswith(r"repair_test\i\paved_j3_w091_h")
    parsed = inspector.inspect_road_objects(
        result.objects,
        world_name=spec.name,
    )
    generated = [
        road for road in parsed.road_objects
        if road.kind == "junction_generated_3"
    ]
    assert len(generated) == 1
    assert math.dist((generated[0].x, generated[0].z), centre) <= 0.05


def test_inspector_repair_is_captured_by_final_building_clearance() -> None:
    assert clearance._ORIGINAL_FIT is repair._fit
