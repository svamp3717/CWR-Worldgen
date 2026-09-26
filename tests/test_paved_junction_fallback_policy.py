import math
from types import SimpleNamespace
from unittest.mock import patch

import cwr_worldgen.paved_junction_fallback_policy as fallback
import cwr_worldgen.paved_junction_policy as paved
import cwr_worldgen.procedural_infrastructure as infrastructure
import cwr_worldgen.road_quality_policy as road_quality


def _plan(model: str, point: tuple[float, float], axis=(0.0, 1.0)):
    return SimpleNamespace(
        model_path=model,
        point=point,
        axis=axis,
        connectors=(
            SimpleNamespace(direction=(0.0, 1.0)),
            SimpleNamespace(direction=(0.0, -1.0)),
            SimpleNamespace(direction=(1.0, 0.0)),
        ),
    )


def _object(object_id: int, model: str, x: float, z: float):
    return SimpleNamespace(
        object_id=object_id,
        model_path=model,
        x=x,
        z=z,
    )


def _generated_plan(
    point: tuple[float, float] = (0.0, 0.0),
):
    radius = (
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
        + infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
    )
    return SimpleNamespace(
        model_path=r"test_world\i\paved_j3_m091_b091_a270.p3d",
        point=point,
        axis=(0.0, 1.0),
        connectors=(
            SimpleNamespace(
                point=(point[0], point[1] + radius),
                direction=(0.0, 1.0),
            ),
            SimpleNamespace(
                point=(point[0], point[1] - radius),
                direction=(0.0, -1.0),
            ),
            SimpleNamespace(
                point=(point[0] + radius, point[1]),
                direction=(1.0, 0.0),
            ),
        ),
    )


def _generated_approach(
    object_id: int,
    *,
    x: float,
    z: float,
    heading: float,
):
    return SimpleNamespace(
        object_id=object_id,
        model_path=r"test_world\i\paved_w091_l0062.p3d",
        x=x,
        y=0.0,
        z=z,
        heading_degrees=heading,
        pitch_degrees=0.0,
    )


def test_generated_plan_uses_compact_quality_reserve_not_stock_approach_reserve() -> None:
    key = (100, 100)
    plan = _generated_plan((100.0, 100.0))
    base = {
        key: road_quality._Junction(
            point=(100.0, 100.0),
            axis=(1.0, 0.0),
            half_length=4.0,
            half_width=3.0,
            directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
        )
    }

    token = paved._PLANS.set({key: plan})
    try:
        with patch.object(
            paved, "_ORIGINAL_GEOMETRY", lambda *_args: base
        ), patch.object(
            paved, "_plans", lambda *_args: {key: plan}
        ):
            geometry = fallback._junction_geometry(None, None, None)
    finally:
        paved._PLANS.reset(token)

    expected = (
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
        + infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
        + road_quality._JUNCTION_OVERLAP
    )
    assert math.isclose(geometry[key].half_length, expected, abs_tol=1.0e-9)
    assert math.isclose(geometry[key].half_width, expected, abs_tol=1.0e-9)
    assert geometry[key].half_length < paved._APPROACH_RESERVE


def test_generated_hub_presence_does_not_hide_disconnected_approaches() -> None:
    key = (1, 2)
    plan = _generated_plan()
    hub = SimpleNamespace(
        object_id=1,
        model_path=plan.model_path,
        x=0.0,
        y=0.0,
        z=0.0,
        heading_degrees=0.0,
        pitch_degrees=0.0,
    )
    spec = SimpleNamespace(road_segment_length=25.0)
    hub_only = SimpleNamespace(
        objects=(hub,),
        junction_cap_objects=1,
    )

    assert fallback._successful_plan_keys(
        hub_only,
        {key: plan},
        spec=spec,
    ) == frozenset()

    radius = (
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
        + infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
    )
    half_piece = 6.2 * 0.5
    connected = SimpleNamespace(
        objects=(
            hub,
            _generated_approach(
                2,
                x=0.0,
                z=radius + half_piece,
                heading=0.0,
            ),
            _generated_approach(
                3,
                x=0.0,
                z=-(radius + half_piece),
                heading=180.0,
            ),
            _generated_approach(
                4,
                x=radius + half_piece,
                z=0.0,
                heading=90.0,
            ),
        ),
        junction_cap_objects=1,
    )

    assert fallback._successful_plan_keys(
        connected,
        {key: plan},
        spec=spec,
    ) == frozenset({key})


def test_stitcher_replaces_stock_sil6_that_straddles_generated_connector() -> None:
    connector_radius = (
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
        + infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
    )
    source_heading = 6.0
    source_direction = paved._direction(source_heading)
    connector = paved._Connector(
        "sil",
        (0.0, connector_radius),
        (0.0, 1.0),
    )
    plan = paved._Plan(
        r"seam_world\i\paved_j3_m091_b091_a265.p3d",
        (0.0, 0.0),
        (0.0, 1.0),
        (paved._Arm("sil", source_direction, connector),),
    )

    # This mirrors terrtest41 object 1050: a 6.25 m stock slab whose inner
    # endpoint is well inside the hub and whose outer endpoint is outside it.
    # Neither endpoint is close enough to the connector for the old 3 m search,
    # although the slab itself crosses the seam.
    centre_radius = 6.407942
    straddler = SimpleNamespace(
        object_id=2,
        model_path=r"o\road\sil6.p3d",
        x=math.sin(math.radians(source_heading)) * centre_radius,
        y=0.035,
        z=math.cos(math.radians(source_heading)) * centre_radius,
        heading_degrees=source_heading,
        pitch_degrees=0.0,
    )
    hub = SimpleNamespace(
        object_id=1,
        model_path=plan.model_path,
        x=0.0,
        y=0.010,
        z=0.0,
        heading_degrees=0.0,
        pitch_degrees=0.0,
    )
    report = SimpleNamespace(
        objects=(hub, straddler),
        junction_cap_objects=1,
    )
    spec = SimpleNamespace(
        name="seam_world",
        road_segment_length=25.0,
        cells=8,
        cell_size=25.0,
    )

    stitched = fallback._stitch_generated_hub_approaches(
        report,
        {(0, 0): plan},
        (0.0,) * 64,
        spec,
    )

    assert len(stitched.objects) == 2
    repaired = stitched.objects[1]
    assert repaired.object_id == straddler.object_id
    assert infrastructure.is_generated_paved_road_model(repaired.model_path)
    axis = fallback._generated_paved_axis(repaired, spec)
    assert axis is not None
    assert min(
        math.dist(connector.point, endpoint)
        for endpoint in axis
    ) <= 0.02
    assert all(
        math.dist(plan.point, endpoint)
        >= connector_radius - 0.05
        for endpoint in axis
    )


def test_stitcher_rebuilds_endpoint_matched_but_angle_mismatched_approach() -> None:
    connector_radius = (
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
        + infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
    )
    connector = paved._Connector(
        "sil",
        (0.0, connector_radius),
        (0.0, 1.0),
    )
    source_heading = 10.7
    source_direction = paved._direction(source_heading)
    plan = paved._Plan(
        r"seam_world\i\paved_j3_m091_b091_a265.p3d",
        (0.0, 0.0),
        (0.0, 1.0),
        (paved._Arm("sil", source_direction, connector),),
    )

    length = 8.1
    direction = source_direction
    old = SimpleNamespace(
        object_id=2,
        model_path=r"seam_world\i\paved_w091_l0081.p3d",
        x=connector.point[0] + direction[0] * length * 0.5,
        y=0.010,
        z=connector.point[1] + direction[1] * length * 0.5,
        heading_degrees=source_heading,
        pitch_degrees=0.0,
    )
    hub = SimpleNamespace(
        object_id=1,
        model_path=plan.model_path,
        x=0.0,
        y=0.010,
        z=0.0,
        heading_degrees=0.0,
        pitch_degrees=0.0,
    )
    report = SimpleNamespace(
        objects=(hub, old),
        junction_cap_objects=1,
    )
    spec = SimpleNamespace(
        name="seam_world",
        road_segment_length=25.0,
        cells=8,
        cell_size=25.0,
    )

    old_axis = fallback._generated_paved_axis(old, spec)
    assert old_axis is not None
    assert min(
        math.dist(connector.point, endpoint)
        for endpoint in old_axis
    ) <= 1.0e-9

    stitched = fallback._stitch_generated_hub_approaches(
        report,
        {(0, 0): plan},
        (0.0,) * 64,
        spec,
    )
    repaired = stitched.objects[1]
    assert repaired.object_id == old.object_id
    assert repaired.model_path != old.model_path
    assert infrastructure.is_generated_paved_road_model(repaired.model_path)


def test_successful_plan_keys_preserves_boundary_match_across_buckets() -> None:
    model = r"o\road\kr_new_sil_sil_t.p3d"
    key = (1, 2)
    plans = {key: _plan(model, (0.0, 0.0))}
    report = SimpleNamespace(
        objects=(
            _object(1, model, fallback._SUCCESS_DISTANCE_METRES, 0.0),
            _object(2, r"o\road\other.p3d", 0.0, 0.0),
        )
    )

    assert fallback._successful_plan_keys(report, plans) == frozenset({key})


def test_successful_plan_keys_prunes_distant_same_model_positions() -> None:
    model = r"o\road\kr_new_sil_sil_t.p3d"
    key = (5, 6)
    plans = {key: _plan(model, (0.0, 0.0))}
    objects = [
        _object(index + 1, model, 1000.0 + index * 2.0, 1000.0)
        for index in range(5000)
    ]
    objects.append(_object(6000, model, 0.10, 0.10))
    report = SimpleNamespace(objects=tuple(objects))
    original_dist = math.dist

    with patch.object(fallback.math, "dist", wraps=original_dist) as distance:
        result = fallback._successful_plan_keys(report, plans)

    assert result == frozenset({key})
    assert distance.call_count <= 4


def test_failed_stock_plan_uses_ordinary_junction_geometry_on_refit() -> None:
    active_key = (100, 100)
    failed_key = (200, 200)
    active_plan = _plan(r"o\road\kr_new_sil_sil_t.p3d", (100.0, 100.0))
    failed_plan = _plan(r"o\road\kr_new_sil_sil_t.p3d", (200.0, 200.0))
    plans = {active_key: active_plan, failed_key: failed_plan}
    base = {
        active_key: road_quality._Junction(
            point=(100.0, 100.0),
            axis=(1.0, 0.0),
            half_length=4.0,
            half_width=3.0,
            directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
        ),
        failed_key: road_quality._Junction(
            point=(200.0, 200.0),
            axis=(1.0, 0.0),
            half_length=5.0,
            half_width=3.5,
            directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
        ),
    }

    token = paved._PLANS.set({active_key: active_plan})
    try:
        with patch.object(paved, "_ORIGINAL_GEOMETRY", lambda *_args: base), patch.object(
            paved, "_plans", lambda *_args: plans
        ):
            geometry = fallback._junction_geometry(None, None, None)
    finally:
        paved._PLANS.reset(token)

    assert geometry[active_key].half_length == paved._APPROACH_RESERVE
    assert geometry[active_key].half_width == paved._APPROACH_RESERVE
    assert geometry[failed_key] == base[failed_key]


def test_failed_stock_junction_retries_stock_before_generated_fallback() -> None:
    key = (100, 100)
    model = r"o\road\kr_new_sil_sil_t.p3d"
    plan = _plan(model, (100.0, 100.0))
    plans = {key: plan}
    trimmed_report = SimpleNamespace(
        objects=(
            _object(1, r"o\road\sil6.p3d", 100.0, 100.0),
        ),
        junction_cap_objects=1,
    )
    ordinary_report = SimpleNamespace(
        objects=(
            _object(10, r"o\road\sil6.p3d", 100.0, 100.0),
        ),
        junction_cap_objects=1,
    )
    recovered_report = SimpleNamespace(
        objects=(
            _object(10, model, 100.0, 100.0),
        ),
        junction_cap_objects=1,
    )

    spec = SimpleNamespace(
        stock_road_piece_fitting=True,
        procedural_paved_road_fallback=True,
    )
    with patch.object(paved, "_plans", lambda *_args: plans), patch.object(
        fallback, "_ORIGINAL_FIT", lambda *_args, **_kwargs: trimmed_report
    ), patch.object(
        paved, "_ORIGINAL_FIT", lambda *_args, **_kwargs: ordinary_report
    ), patch.object(
        paved, "_apply_plans", lambda report, *_args: recovered_report
    ), patch.object(
        paved,
        "_generated_plan",
        side_effect=AssertionError(
            "generated fallback must not run after stock recovery succeeds"
        ),
    ):
        result = fallback._fit(
            None,
            None,
            (),
            spec,
            starting_id=1,
            progress_callback=None,
        )

    assert result is recovered_report


def test_failed_stock_junction_still_returns_ordinary_roads_when_recovery_fails() -> None:
    key = (100, 100)
    plan = _plan(r"o\road\kr_new_sil_sil_t.p3d", (100.0, 100.0))
    plans = {key: plan}
    trimmed_report = SimpleNamespace(
        objects=(
            _object(1, r"o\road\sil6.p3d", 100.0, 100.0),
        ),
        junction_cap_objects=1,
    )
    ordinary_report = SimpleNamespace(
        objects=(
            _object(10, r"o\road\sil12.p3d", 100.0, 95.0),
            _object(11, r"o\road\sil12.p3d", 100.0, 105.0),
        ),
        junction_cap_objects=1,
    )
    active_plan_sets = []

    def ordinary_fit(*_args, **_kwargs):
        active_plan_sets.append(dict(paved._PLANS.get() or {}))
        return ordinary_report

    spec = SimpleNamespace(
        stock_road_piece_fitting=True,
        procedural_paved_road_fallback=False,
    )
    with patch.object(paved, "_plans", lambda *_args: plans), patch.object(
        fallback, "_ORIGINAL_FIT", lambda *_args, **_kwargs: trimmed_report
    ), patch.object(paved, "_ORIGINAL_FIT", ordinary_fit), patch.object(
        paved, "_apply_plans", lambda report, *_args: report
    ):
        result = fallback._fit(
            None,
            None,
            (),
            spec,
            starting_id=1,
            progress_callback=None,
        )

    assert result is ordinary_report
    assert active_plan_sets
    assert all(value == {} for value in active_plan_sets)
