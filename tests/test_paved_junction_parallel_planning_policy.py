from itertools import product
from types import SimpleNamespace

from cwr_worldgen import paved_junction_fallback_policy as fallback
from cwr_worldgen import paved_junction_parallel_planning_policy as parallel
from cwr_worldgen import paved_junction_performance_policy as performance
from cwr_worldgen import paved_junction_policy as paved


def _value(score: float, object_id: int):
    return (
        score,
        SimpleNamespace(object_id=object_id),
        SimpleNamespace(name=f"choice-{object_id}"),
    )


def _reference_combination(options):
    best = None
    for combination in product(*options):
        object_ids = tuple(value[1].object_id for value in combination)
        if len(set(object_ids)) != len(object_ids):
            continue
        score = sum(value[0] for value in combination)
        if best is None or score < best[0]:
            best = score, combination
    return None if best is None else best[1]


def test_branch_and_bound_matches_cartesian_reference_and_tie_order() -> None:
    options = (
        (_value(1.0, 10), _value(1.0, 11), _value(3.0, 12)),
        (_value(1.0, 10), _value(1.0, 20), _value(4.0, 21)),
        (_value(0.5, 30), _value(2.0, 20), _value(2.5, 31)),
    )
    expected = _reference_combination(options)
    actual = parallel._best_combination(options)
    assert actual == expected


def test_cached_choice_rebinds_new_object_id_when_geometry_is_identical() -> None:
    old_target = paved._Target(7, (10.0, 20.0), (0.0, 1.0))
    new_target = paved._Target(7007, (10.0, 20.0), (0.0, 1.0))
    choice = paved._ApproachChoice(
        1,
        1,
        25,
        0,
        0,
        25,
        25,
        old_target.point,
    )
    cached = ((2.5, old_target, choice),)
    rebound = parallel._rebind_choices(((new_target,),), cached)
    assert rebound is not None
    assert rebound[0][0] == 2.5
    assert rebound[0][1].object_id == 7007
    assert rebound[0][2] == choice


def test_ambiguous_duplicate_target_geometry_forces_replan() -> None:
    old_target = paved._Target(7, (10.0, 20.0), (0.0, 1.0))
    choice = paved._ApproachChoice(
        1,
        1,
        25,
        0,
        0,
        25,
        25,
        old_target.point,
    )
    cached = ((2.5, old_target, choice),)
    duplicate_targets = (
        paved._Target(8, old_target.point, old_target.continuation),
        paved._Target(9, old_target.point, old_target.continuation),
    )
    assert parallel._rebind_choices((duplicate_targets,), cached) is None


def test_cached_plan_application_uses_live_plan_identity() -> None:
    plan = SimpleNamespace(point=(123.4, 567.8))
    sentinel = object()
    token = parallel._ACTIVE_CHOICES.set({id(plan): sentinel})
    try:
        assert parallel._cached_plan_application(None, plan, None) is sentinel
    finally:
        parallel._ACTIVE_CHOICES.reset(token)


def test_fallback_dependency_radius_marks_only_near_active_plans() -> None:
    changed = (0, 0)
    near = (1, 0)
    far = (2, 0)
    plans = {
        changed: SimpleNamespace(point=(0.0, 0.0)),
        near: SimpleNamespace(point=(100.0, 0.0)),
        far: SimpleNamespace(point=(500.0, 0.0)),
    }
    affected = fallback._affected_plan_keys(
        plans,
        (near, far),
        (changed,),
    )
    assert affected == frozenset({near})


def test_parallel_policy_keeps_public_performance_apply_symbol_live() -> None:
    # The fallback installer may already have activated this policy in another
    # test. Reinstallation is intentionally idempotent.
    parallel.install_paved_junction_parallel_planning_policy()
    assert paved._apply_plans is parallel.apply_paved_junctions_parallel
    assert performance.apply_paved_junctions_fast is parallel.apply_paved_junctions_parallel
