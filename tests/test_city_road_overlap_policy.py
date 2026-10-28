from types import SimpleNamespace

from cwr_worldgen import city_road_overlap_policy as city
from cwr_worldgen import playability
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport


def _spec():
    return SimpleNamespace(
        name="city_overlap",
        include_minor_roads=True,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        gravel_road_model="",
        procedural_gravel_roads=False,
        custom_road_shapes=False,
        road_segment_length=25.0,
    )


ROAD = {"highway": "residential", "surface": "asphalt"}
DIRT = {"highway": "service", "surface": "dirt"}


def _dataset(*tags):
    return SimpleNamespace(
        roads=tuple(SimpleNamespace(tags=t) for t in tags),
    )


def _straight(x=0.0, *, z=0.0):
    return ((x, z), (x + 50.0, z), (x + 100.0, z))


def test_city_duplicate_straight_removed_before_p3d_fitting():
    first, duplicate = _straight(), _straight(z=0.35)
    changed = city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD), (first, duplicate), _spec(),
    )
    assert changed == (first, ())


def test_reversed_duplicate_street_removed():
    first, duplicate = _straight(), tuple(reversed(_straight(z=0.20)))
    assert city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD), (first, duplicate), _spec(),
    ) == (first, ())


def test_city_junction_vertices_keep_richer_duplicate_tracing():
    simple = ((0.0, 0.0), (100.0, 0.0))
    rich = ((0.0, 0.10), (25.0, 0.10), (50.0, 0.10),
            (75.0, 0.10), (100.0, 0.10))
    result = city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD), (simple, rich), _spec(),
    )
    assert result == ((), rich)


def test_not_merge_different_road_surface_or_class():
    first, second = _straight(), _straight(z=0.2)
    for different in (
        DIRT,
        {**ROAD, "highway": "primary"},
        {**ROAD, "lanes": "2"},
        {**ROAD, "oneway": "yes"},
        {**ROAD, "bridge": "yes"},
        {**ROAD, "layer": "1"},
    ):
        assert city.suppress_coincident_street_strokes(
            _dataset(ROAD, different), (first, second), _spec(),
        ) == (first, second)


def test_parallel_two_carriageways_are_not_removed():
    lines = (_straight(), _straight(z=3.25))
    assert city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD), lines, _spec(),
    ) == lines


def test_connected_separate_blocks_not_removed():
    lines = (((0.0, 0.0), (50.0, 0.0)),
             ((50.0, 0.0), (100.0, 0.0)))
    assert city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD), lines, _spec(),
    ) == lines


def test_identical_endpoints_different_city_routes_not_removed():
    lines = (((0.0, 0.0), (50.0, 0.0), (100.0, 0.0)),
             ((0.0, 0.0), (50.0, 10.0), (100.0, 0.0)))
    assert city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD), lines, _spec(),
    ) == lines


def test_partial_overlaps_do_not_remove_street_sections():
    lines = (((0.0, 0.0), (100.0, 0.0)),
             ((50.0, 0.05), (150.0, 0.05)))
    assert city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD), lines, _spec(),
    ) == lines


def test_extra_branch_node_on_losing_way_is_protected():
    first = ((0.0, 0.0), (100.0, 0.0))
    duplicate = ((0.0, 0.2), (50.0, 0.2), (100.0, 0.2))
    branch = ((50.0, 0.2), (50.0, 40.0))
    # Richer duplicate is kept; the explicit branch continues to meet it.
    assert city.suppress_coincident_street_strokes(
        _dataset(ROAD, ROAD, ROAD), (first, duplicate, branch), _spec(),
    ) == ((), duplicate, branch)


def _obj(num, model, x, z, *, y=0.0, heading=0.0):
    return WorldObject(num, model, x, y, z, heading, 0.0)


def _report(objects, *, cap_count=1):
    return RoadFitReport(
        objects=tuple(objects), chain_count=1, connection_count=0,
        failed_connections=0, maximum_connection_gap=0.0,
        maximum_chain_gap=0.0, truncated=False,
        junction_cap_objects=cap_count,
    )


def test_repeated_short_dirt_piece_at_town_junction_removed():
    cap = _obj(1, r"o\road\ces6.p3d", 100, 100)
    duplicate = _obj(2, r"o\road\ces6.p3d", 100, 100)
    result = city.remove_duplicate_town_cap_slabs(
        _report((cap, duplicate)), _spec(),
    )
    assert tuple(obj.object_id for obj in result.objects) == (1,)
    assert result.junction_cap_objects == 1


def test_genuine_partial_town_junction_approach_remains():
    cap = _obj(1, r"o\road\ces6.p3d", 100, 100)
    approach = _obj(2, r"o\road\ces6.p3d", 100, 103.5)
    report = _report((cap, approach))
    assert city.remove_duplicate_town_cap_slabs(report, _spec()) is report


def test_parallel_and_perpendicular_town_roads_remain():
    cap = _obj(1, r"o\road\ces6.p3d", 100, 100)
    for road in (
        _obj(2, r"o\road\ces6.p3d", 102.5, 100),
        _obj(2, r"o\road\ces6.p3d", 100, 100, heading=90),
        _obj(2, r"o\road\ces6.p3d", 100, 100, y=2.0),
        _obj(2, r"o\road\sil6.p3d", 100, 100),
    ):
        report = _report((cap, road))
        assert city.remove_duplicate_town_cap_slabs(report, _spec()) is report


def test_preserves_town_junction_caps_and_metadata():
    cap = _obj(1, r"o\road\ces6.p3d", 100, 100)
    candidate = _obj(2, r"o\road\ces6.p3d", 100, 100)
    unrelated = _obj(3, r"o\road\ces25.p3d", 200, 200)
    result = city.remove_duplicate_town_cap_slabs(
        _report((cap, candidate, unrelated)), _spec(),
    )
    assert [obj.object_id for obj in result.objects] == [1, 3]
    assert result.junction_cap_objects == 1


def test_disabled_experiment_does_not_remove_town_slabs(monkeypatch):
    cap = _obj(1, r"o\road\ces6.p3d", 100, 100)
    candidate = _obj(2, r"o\road\ces6.p3d", 100, 100)
    report = _report((cap, candidate))
    monkeypatch.setattr(city, "_ORIGINAL_FIT", lambda *args, **kwargs: report)
    monkeypatch.delenv("CWR_ROAD_PAINT_SNAP_METRES", raising=False)
    assert playability.fit_road_objects(None, None, None, _spec()) is report
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    assert len(playability.fit_road_objects(None, None, None, _spec()).objects) == 1


def test_sparse_city_network_is_not_suppressed():
    lines = tuple(_straight(float(i * 300.0)) for i in range(180))
    result = city.suppress_coincident_street_strokes(
        _dataset(*(ROAD for _ in lines)), lines, _spec(),
    )
    assert result == lines



def test_opposite_direction_oneway_carriageways_are_preserved():
    road = {**ROAD, "oneway": "yes"}
    first, reversed_second = _straight(), tuple(reversed(_straight(z=0.20)))
    assert city.suppress_coincident_street_strokes(
        _dataset(road, road), (first, reversed_second), _spec(),
    ) == (first, reversed_second)
