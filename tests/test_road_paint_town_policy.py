from types import SimpleNamespace

from cwr_worldgen import road_paint_town_policy as town
from cwr_worldgen import road_paint_snap_policy as painting


def _spec():
    return SimpleNamespace(
        name="town_roads",
        include_minor_roads=True,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        gravel_road_model="",
        procedural_gravel_roads=True,
        road_segment_length=25.0,
        custom_road_shapes=False,
    )


def _data(*tags):
    return SimpleNamespace(roads=tuple(SimpleNamespace(tags=item) for item in tags))


_DIRT = {"highway": "service", "surface": "dirt"}
_GRAVEL = {"highway": "track", "surface": "gravel"}
_PAVED = {"highway": "residential", "surface": "asphalt"}


def _junction():
    return (((0.0, 20.0), (60.0, 20.0)),
            ((30.0, 0.0), (30.0, 18.0)))


def test_dirt_town_service_road_joins_main_street_at_exact_shared_node():
    source = _junction()
    result = town.snap_town_road_mouths(
        _data(_DIRT, _DIRT), source, _spec(), reach_metres=2.5,
    )
    assert result[0] == ((0.0, 20.0), (30.0, 20.0), (60.0, 20.0))
    assert result[1] == ((30.0, 0.0), (30.0, 20.0))
    assert town.snap_town_road_mouths(
        _data(_DIRT, _DIRT), result, _spec(), reach_metres=2.5,
    ) == result


def test_gravel_town_track_joins_main_track():
    result = town.snap_town_road_mouths(
        _data(_GRAVEL, _GRAVEL), _junction(), _spec(), reach_metres=2.5,
    )
    assert result[0][1] == result[1][-1] == (30.0, 20.0)


def test_incomplete_same_surface_town_crossing_remains_isolated_if_ambiguous():
    lines = (
        ((0.0, 20.0), (60.0, 20.0)),
        ((30.0, 0.0), (30.0, 18.0)),
        ((32.0, 0.0), (32.0, 18.0)),
    )
    result = town.snap_town_road_mouths(
        _data(_DIRT, _DIRT, _DIRT), lines, _spec(), reach_metres=2.5,
    )
    assert result == lines


def test_town_paved_streets_keep_existing_paved_intersection_handler():
    lines = _junction()
    assert town.snap_town_road_mouths(
        _data(_PAVED, _PAVED), lines, _spec(), reach_metres=2.5,
    ) == lines


def test_different_town_road_surfaces_do_not_merge():
    lines = _junction()
    for surface in (_PAVED, _GRAVEL):
        assert town.snap_town_road_mouths(
            _data(_DIRT, surface), lines, _spec(), reach_metres=2.5,
        ) == lines


def test_no_sideways_snap_to_parallel_town_street():
    lines = (
        ((0.0, 20.0), (60.0, 20.0)),
        ((0.0, 18.0), (30.0, 18.0)),
    )
    assert town.snap_town_road_mouths(
        _data(_DIRT, _DIRT), lines, _spec(), reach_metres=2.5,
    ) == lines


def test_grade_separated_or_distant_town_street_stays_detached():
    main, branch = _junction()
    for tags in ({**_DIRT, "bridge": "yes"}, {**_DIRT, "tunnel": "yes"},
                 {**_DIRT, "layer": "1"}):
        lines = (main, branch)
        assert town.snap_town_road_mouths(
            _data(_DIRT, tags), lines, _spec(), reach_metres=2.5,
        ) == lines
    far = (main, ((30.0, 0.0), (30.0, 16.0)))
    assert town.snap_town_road_mouths(
        _data(_DIRT, _DIRT), far, _spec(), reach_metres=2.5,
    ) == far


def test_already_connected_t_junction_keeps_existing_vertex_positions():
    main = ((0.0, 20.0), (30.0, 20.0), (60.0, 20.0))
    branch = ((30.0, 0.0), (30.0, 20.0))
    assert town.snap_town_road_mouths(
        _data(_DIRT, _DIRT), (main, branch), _spec(), reach_metres=2.5,
    ) == (main, branch)


def test_short_town_spur_cannot_be_stretched_beyond_one_fifth():
    main, _ = _junction()
    short = ((30.0, 10.0), (30.0, 18.0))
    lines = (main, short)
    assert town.snap_town_road_mouths(
        _data(_DIRT, _DIRT), lines, _spec(), reach_metres=2.5,
    ) == lines


def test_full_experiment_gate_also_connects_town_service_mouth(monkeypatch):
    lines = _junction()
    monkeypatch.setattr(painting, "_ORIGINAL_PROJECT",
                        lambda data, projection, spec: lines)
    data, spec = _data(_DIRT, _DIRT), _spec()
    monkeypatch.delenv("CWR_ROAD_PAINT_SNAP_METRES", raising=False)
    assert painting._project_with_joined_strokes(data, None, spec) is lines
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    assert painting._project_with_joined_strokes(
        data, None, spec,
    )[0][1] == (30.0, 20.0)


def test_no_spurious_connections_across_sparse_town_grid():
    lines = tuple(
        ((float(i * 100), 0.0), (float(i * 100), 60.0))
        for i in range(120)
    )
    tags = (_DIRT,) * len(lines)
    assert town.snap_town_road_mouths(
        _data(*tags), lines, _spec(), reach_metres=2.5,
    ) == lines
