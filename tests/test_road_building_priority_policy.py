from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import road_building_priority_policy as priority


@dataclass(frozen=True)
class _Plan:
    osm_key: str
    geometry_index: int
    geometry_kind: str
    support_polygon: tuple[tuple[float, float], ...]


def _plan(key: str = "way/1") -> _Plan:
    return _Plan(
        key,
        0,
        "polygon",
        ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)),
    )


def _road(object_id: int, model_path: str):
    return SimpleNamespace(object_id=object_id, model_path=model_path)


def _primitive(object_id: int, z: float = 0.0):
    return clearance._RoadPrimitive(
        object_id,
        (-4.0, z),
        (4.0, z),
        2.30,
        0.0,
    )


def _base_report(rejected: int = 1):
    return clearance.FinalBuildingRoadConflictReport(
        buildings=1,
        road_primitives=1,
        conflicted=1,
        moved=0,
        rejected=rejected,
        nearby_road_checks=3,
    )


def _run(monkeypatch, *, roads, primitives, junction_caps=0):
    original = _plan()
    monkeypatch.setattr(
        priority,
        "_ORIGINAL_RESOLVE",
        lambda *args, **kwargs: ((), _base_report()),
    )
    monkeypatch.setattr(
        clearance,
        "_road_primitives",
        lambda report, elevations, spec: tuple(primitives),
    )
    report = SimpleNamespace(objects=tuple(roads), junction_cap_objects=junction_caps)
    plans, conflict_report = priority.resolve_road_building_priorities(
        (original,),
        report,
        (),
        None,
        SimpleNamespace(road_segment_length=25.0),
    )
    return original, report, plans, conflict_report, priority._PRIORITY_STATE.get()


def test_unmovable_building_is_preserved_over_one_minor_gravel_piece(monkeypatch) -> None:
    original, report, plans, conflict_report, state = _run(
        monkeypatch,
        roads=(_road(10, r"testworld\i\gravel6.p3d"),),
        primitives=(_primitive(10),),
    )

    assert plans == (original,)
    assert conflict_report.rejected == 0
    assert state is not None
    assert state.road_objects_identity == id(report.objects)
    assert state.suppressed_object_ids == frozenset({10})
    assert state.preserved_buildings == 1


def test_major_paved_road_keeps_right_of_way(monkeypatch) -> None:
    _original, _report, plans, conflict_report, state = _run(
        monkeypatch,
        roads=(_road(11, r"data3d\asf6.p3d"),),
        primitives=(_primitive(11),),
    )

    assert plans == ()
    assert conflict_report.rejected == 1
    assert state is not None
    assert not state.suppressed_object_ids
    assert state.protected_rejections == 1


def test_junction_cap_slot_is_never_suppressed_even_when_model_is_minor(monkeypatch) -> None:
    _original, _report, plans, conflict_report, state = _run(
        monkeypatch,
        roads=(_road(12, r"testworld\i\gravel6.p3d"),),
        primitives=(_primitive(12),),
        junction_caps=1,
    )

    assert plans == ()
    assert conflict_report.rejected == 1
    assert state is not None
    assert not state.suppressed_object_ids


def test_priority_budget_refuses_a_fifty_metre_minor_road_gap(monkeypatch) -> None:
    _original, _report, plans, conflict_report, state = _run(
        monkeypatch,
        roads=(
            _road(20, r"testworld\i\gravel25.p3d"),
            _road(21, r"testworld\i\gravel25.p3d"),
        ),
        primitives=(_primitive(20, -0.5), _primitive(21, 0.5)),
    )

    assert plans == ()
    assert conflict_report.rejected == 1
    assert state is not None
    assert not state.suppressed_object_ids


def test_minor_generated_gravel_curves_are_suppressible_but_junctions_are_not() -> None:
    spec = SimpleNamespace(road_segment_length=25.0)
    assert priority._minor_object_length(
        _road(30, r"world\i\gravel12_l15.p3d"), spec, frozenset()
    ) == 12.0
    assert priority._minor_object_length(
        _road(31, r"world\i\gravel_j3_t90.p3d"), spec, frozenset()
    ) is None


def test_final_assembly_filter_is_scoped_to_the_recorded_road_tuple() -> None:
    roads = (
        _road(40, r"world\i\gravel6.p3d"),
        _road(41, r"data3d\asf6.p3d"),
    )
    priority._PRIORITY_STATE.set(
        priority.RoadBuildingPriorityState(
            id(roads), frozenset({40}), 1, 0, 0
        )
    )

    assert priority._filter_suppressed_roads(roads) == (roads[1],)
    copied = tuple(list(roads))
    assert id(copied) != id(roads)
    assert priority._filter_suppressed_roads(copied) == copied
