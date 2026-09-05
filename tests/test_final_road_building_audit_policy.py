from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import final_road_building_audit_policy as audit
from cwr_worldgen import road_building_priority_policy as priority


@dataclass(frozen=True)
class _Plan:
    osm_key: str
    geometry_index: int
    geometry_kind: str
    support_polygon: tuple[tuple[float, float], ...]


def _plan(key: str = "way/1", x: float = 0.0, z: float = 0.0) -> _Plan:
    return _Plan(
        key,
        0,
        "polygon",
        (
            (x - 2.0, z - 2.0),
            (x + 2.0, z - 2.0),
            (x + 2.0, z + 2.0),
            (x - 2.0, z + 2.0),
        ),
    )


def _road(object_id: int):
    return SimpleNamespace(object_id=object_id, model_path=rf"world\i\gravel6.p3d")


def _primitive(object_id: int, z: float):
    return clearance._RoadPrimitive(
        object_id,
        (-6.0, z),
        (6.0, z),
        2.30,
        0.0,
    )


def _report(rejected: int = 0, checks: int = 0):
    return clearance.FinalBuildingRoadConflictReport(
        buildings=1,
        road_primitives=1,
        conflicted=1,
        moved=0,
        rejected=rejected,
        nearby_road_checks=checks,
    )


def _patch_primitives(monkeypatch, positions):
    def primitives(report, elevations, spec):
        return tuple(
            _primitive(int(obj.object_id), positions[int(obj.object_id)])
            for obj in report.objects
        )

    monkeypatch.setattr(clearance, "_road_primitives", primitives)


def test_building_kept_when_its_only_conflict_is_an_approved_suppression(monkeypatch) -> None:
    plan = _plan()
    roads = (_road(10),)
    road_report = SimpleNamespace(objects=roads)
    _patch_primitives(monkeypatch, {10: 0.0})
    monkeypatch.setattr(audit, "_ORIGINAL_RESOLVE", lambda *args, **kwargs: ((plan,), _report()))
    priority._PRIORITY_STATE.set(
        priority.RoadBuildingPriorityState(id(roads), frozenset({10}), 1, 0, 0)
    )

    plans, report = audit.audit_final_road_building_conflicts(
        (plan,), road_report, (), None, SimpleNamespace(),
    )

    assert plans == (plan,)
    assert report.rejected == 0
    state = priority._PRIORITY_STATE.get()
    assert state is not None
    assert state.suppressed_object_ids == frozenset({10})
    final = audit._FINAL_AUDIT.get()
    assert final is not None
    assert final.violations == 0


def test_surviving_conflict_with_unsuppressed_road_is_rejected(monkeypatch) -> None:
    plan = _plan()
    roads = (_road(20),)
    road_report = SimpleNamespace(objects=roads)
    _patch_primitives(monkeypatch, {20: 0.0})
    monkeypatch.setattr(audit, "_ORIGINAL_RESOLVE", lambda *args, **kwargs: ((plan,), _report()))
    priority._PRIORITY_STATE.set(
        priority.RoadBuildingPriorityState(id(roads), frozenset(), 0, 0, 0)
    )

    plans, report = audit.audit_final_road_building_conflicts(
        (plan,), road_report, (), None, SimpleNamespace(),
    )

    assert plans == ()
    assert report.rejected == 1
    final = audit._FINAL_AUDIT.get()
    assert final is not None
    assert final.violations == 1
    assert final.rejected == 1


def test_unused_minor_road_suppression_is_released(monkeypatch) -> None:
    plan = _plan()
    roads = (_road(30), _road(31))
    road_report = SimpleNamespace(objects=roads)
    _patch_primitives(monkeypatch, {30: 0.0, 31: 40.0})
    monkeypatch.setattr(audit, "_ORIGINAL_RESOLVE", lambda *args, **kwargs: ((plan,), _report()))
    priority._PRIORITY_STATE.set(
        priority.RoadBuildingPriorityState(
            id(roads), frozenset({30, 31}), 1, 0, 0
        )
    )

    plans, _report_value = audit.audit_final_road_building_conflicts(
        (plan,), road_report, (), None, SimpleNamespace(),
    )

    assert plans == (plan,)
    state = priority._PRIORITY_STATE.get()
    assert state is not None
    assert state.suppressed_object_ids == frozenset({30})
    final = audit._FINAL_AUDIT.get()
    assert final is not None
    assert final.released_suppressions == 1


def test_rejected_building_does_not_leave_an_unneeded_minor_road_gap(monkeypatch) -> None:
    plan = _plan()
    roads = (_road(40), _road(41))
    road_report = SimpleNamespace(objects=roads)
    # Road 40 was suppressed for this building, but road 41 remains authoritative
    # and still intersects it. The audit rejects the building, then releases 40.
    _patch_primitives(monkeypatch, {40: 0.0, 41: 0.5})
    monkeypatch.setattr(audit, "_ORIGINAL_RESOLVE", lambda *args, **kwargs: ((plan,), _report()))
    priority._PRIORITY_STATE.set(
        priority.RoadBuildingPriorityState(id(roads), frozenset({40}), 1, 0, 0)
    )

    plans, report = audit.audit_final_road_building_conflicts(
        (plan,), road_report, (), None, SimpleNamespace(),
    )

    assert plans == ()
    assert report.rejected == 1
    state = priority._PRIORITY_STATE.get()
    assert state is not None
    assert state.suppressed_object_ids == frozenset()
    assert state.preserved_buildings == 0
    assert state.protected_rejections == 1


def test_audit_reports_bounded_progress(monkeypatch) -> None:
    plans = tuple(_plan(f"way/{index}", x=float(index) * 20.0) for index in range(8))
    roads = (_road(50),)
    road_report = SimpleNamespace(objects=roads)
    _patch_primitives(monkeypatch, {50: 200.0})
    monkeypatch.setattr(audit, "_ORIGINAL_RESOLVE", lambda *args, **kwargs: (plans, _report()))
    priority._PRIORITY_STATE.set(
        priority.RoadBuildingPriorityState(id(roads), frozenset(), 0, 0, 0)
    )
    events = []

    kept, _report_value = audit.audit_final_road_building_conflicts(
        plans,
        road_report,
        (),
        None,
        SimpleNamespace(),
        progress_callback=lambda percent, stage: events.append((percent, stage)),
    )

    assert kept == plans
    assert events
    assert all(percent == 52 for percent, _stage in events)
    assert any("Final road/building conflict audit complete" in stage for _percent, stage in events)
