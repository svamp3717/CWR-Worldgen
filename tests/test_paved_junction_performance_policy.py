from types import SimpleNamespace

from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen import paved_junction_policy as paved
from cwr_worldgen import paved_junction_performance_policy as performance
from cwr_worldgen import road_audit_performance_policy as audit_progress


def _spec():
    return SimpleNamespace(
        road_segment_length=25.0,
        road_connection_tolerance=0.35,
    )


def _north_plan():
    connector = paved._Connector(
        "sil",
        (0.0, paved._JUNCTION_RADIUS),
        (0.0, 1.0),
    )
    arm = paved._Arm("sil", (0.0, 1.0), connector)
    plan = paved._Plan(
        paved._t_models()[0][0],
        (0.0, 0.0),
        (0.0, 1.0),
        (arm,),
    )
    return plan, arm


def test_spatial_target_lookup_matches_reference_full_scan() -> None:
    plan, arm = _north_plan()
    # A 25 m straight centred at z=43.75 has its near endpoint at z=31.25,
    # exactly 25 m beyond the stock junction connector at z=6.25.
    local = WorldObject(
        1,
        r"o\road\sil25.p3d",
        0.0,
        0.0,
        43.75,
        heading_degrees=0.0,
    )
    far = WorldObject(
        2,
        r"o\road\sil25.p3d",
        1000.0,
        0.0,
        1000.0,
        heading_degrees=0.0,
    )
    report = RoadFitReport(
        objects=(local, far),
        chain_count=1,
        connection_count=1,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )
    spec = _spec()
    state = performance._build_spatial_state(
        report,
        spec,
        performance._Reporter(None),
    )

    expected = paved._target_candidates(report, plan, arm, spec)
    actual = performance._target_candidates(state, plan, arm)

    assert actual == expected
    assert actual
    assert actual[0].object_id == local.object_id


def test_spatial_target_lookup_does_not_rescan_road_geometry(monkeypatch) -> None:
    plan, arm = _north_plan()
    objects = tuple(
        WorldObject(
            index + 1,
            r"o\road\sil25.p3d",
            0.0 if index == 0 else 1000.0 + index * 30.0,
            0.0,
            43.75 if index == 0 else 1000.0,
            heading_degrees=0.0,
        )
        for index in range(200)
    )
    report = RoadFitReport(
        objects=objects,
        chain_count=1,
        connection_count=1,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )
    calls = 0
    original = paved._object_axis

    def counted(obj, spec):
        nonlocal calls
        calls += 1
        return original(obj, spec)

    monkeypatch.setattr(paved, "_object_axis", counted)
    state = performance._build_spatial_state(
        report,
        _spec(),
        performance._Reporter(None),
    )
    build_calls = calls
    assert build_calls == len(objects)

    for _index in range(5):
        performance._target_candidates(state, plan, arm)
    assert calls == build_calls


def test_precomputed_approach_templates_match_reference_search() -> None:
    plan, arm = _north_plan()
    target = paved._Target(
        9,
        (0.0, 31.25),
        (0.0, 1.0),
    )

    expected = paved._approach_choice_to_target(
        plan,
        arm,
        target,
        0.35,
    )
    actual = performance._approach_choice_to_target(
        plan,
        arm,
        target,
        0.35,
    )

    assert expected is not None
    assert actual == expected


def test_audit_completion_is_not_reported_twice() -> None:
    performance.install_paved_junction_performance_policy()
    events = []
    reporter = audit_progress._AuditReporter(
        lambda percent, message: events.append((percent, message)),
        1,
    )

    reporter.junction(1, failed=0, indexed_axes=3)
    reporter.junction(
        1,
        failed=0,
        indexed_axes=3,
        force=True,
    )

    assert len(events) == 1
    assert "(1/1, 100%;" in events[0][1]


def test_performance_policy_is_the_live_paved_apply_target() -> None:
    performance.install_paved_junction_performance_policy()
    assert paved._apply_plans is performance.apply_paved_junctions_fast
    assert performance._ORIGINAL_APPLY is not None
