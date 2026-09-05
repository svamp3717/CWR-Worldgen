from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import physical_road_overlap_policy as physical


@dataclass(frozen=True)
class _Plan:
    osm_key: str
    geometry_index: int
    geometry_kind: str
    support_polygon: tuple[tuple[float, float], ...]
    x: float = 0.0
    z: float = 0.0


def _plan() -> _Plan:
    # Nearest wall is z=1.97. A ces road surface with half-width 1.75 therefore
    # stops 0.22 m short of the mapped building, matching the Strangnas cathedral
    # failure mode closely enough to lock the semantics down.
    return _Plan(
        "way/64524654",
        0,
        "polygon",
        ((-4.0, 1.97), (4.0, 1.97), (4.0, 7.0), (-4.0, 7.0)),
    )


def _primitive() -> clearance._RoadPrimitive:
    return clearance._RoadPrimitive(
        77,
        (-10.0, 0.0),
        (10.0, 0.0),
        1.75,
        0.0,
    )


def _report(rejected: int = 1) -> clearance.FinalBuildingRoadConflictReport:
    return clearance.FinalBuildingRoadConflictReport(
        buildings=1,
        road_primitives=1,
        conflicted=1,
        moved=0,
        rejected=rejected,
        nearby_road_checks=3,
    )


def test_preferred_margin_is_not_physical_overlap() -> None:
    plan = _plan()
    index = clearance._RoadPrimitiveIndex((_primitive(),))

    preferred, _ = physical.conflicts_at_clearance(
        plan.support_polygon,
        index,
        clearance._ROAD_CLEARANCE_METRES,
    )
    actual, _ = physical.conflicts_at_clearance(
        plan.support_polygon,
        index,
        0.0,
    )

    assert preferred
    assert actual == ()


def test_failed_preferred_relocation_restores_physically_clear_source_building(monkeypatch) -> None:
    plan = _plan()
    footprint_was_reserved = []

    monkeypatch.setattr(
        clearance,
        "_road_primitives",
        lambda report, elevations, spec: (_primitive(),),
    )

    def fake_step2(plans, road_report, elevations, raster, spec, *, progress_callback=None):
        # Simulate the old Step-2 failure path. The physical policy's guarded
        # update must keep the original footprint in this index while later plans
        # would be processed.
        index = clearance._BuildingFootprintIndex(
            tuple(item.support_polygon for item in plans)
        )
        index.update(0, None)
        footprint_was_reserved.append(index.polygons[0] is not None)
        return (), _report()

    monkeypatch.setattr(physical, "_STEP2_RESOLVE", fake_step2)

    plans, report = physical._preferred_step2_resolve(
        (plan,),
        SimpleNamespace(objects=(SimpleNamespace(object_id=77),)),
        (),
        None,
        SimpleNamespace(),
    )

    assert footprint_was_reserved == [True]
    assert plans == (plan,)
    assert report.rejected == 0
    assert report.nearby_road_checks > 3


def test_true_surface_overlap_is_still_rejected(monkeypatch) -> None:
    plan = _Plan(
        "way/2",
        0,
        "polygon",
        ((-4.0, 1.0), (4.0, 1.0), (4.0, 7.0), (-4.0, 7.0)),
    )
    monkeypatch.setattr(
        clearance,
        "_road_primitives",
        lambda report, elevations, spec: (_primitive(),),
    )
    monkeypatch.setattr(
        physical,
        "_STEP2_RESOLVE",
        lambda *args, **kwargs: ((), _report()),
    )

    plans, report = physical._preferred_step2_resolve(
        (plan,),
        SimpleNamespace(objects=(SimpleNamespace(object_id=77),)),
        (),
        None,
        SimpleNamespace(),
    )

    assert plans == ()
    assert report.rejected == 1


def test_contextual_conflicts_switch_between_preferred_and_physical_modes() -> None:
    plan = _plan()
    index = clearance._RoadPrimitiveIndex((_primitive(),))

    token = physical._CONFLICT_CLEARANCE.set(clearance._ROAD_CLEARANCE_METRES)
    try:
        preferred, _ = clearance._conflicts(plan.support_polygon, index)
    finally:
        physical._CONFLICT_CLEARANCE.reset(token)

    token = physical._CONFLICT_CLEARANCE.set(0.0)
    try:
        actual, _ = clearance._conflicts(plan.support_polygon, index)
    finally:
        physical._CONFLICT_CLEARANCE.reset(token)

    assert preferred
    assert actual == ()


def test_new_cache_revision_invalidates_clearance_only_deletions() -> None:
    assert clearance._CACHE_REVISION == "final-road-building-clearance-v3-physical-overlap"
