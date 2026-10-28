from types import SimpleNamespace

from cwr_worldgen import road_paint_snap_policy as painting


def _spec():
    return SimpleNamespace(
        name="road_paint_experiment",
        include_minor_roads=True,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        gravel_road_model="",
        procedural_gravel_roads=False,
        road_segment_length=25.0,
        custom_road_shapes=False,
    )


def _dataset(*tags):
    return SimpleNamespace(roads=tuple(SimpleNamespace(tags=item) for item in tags))


PAVED = {"highway": "residential", "surface": "asphalt"}
GRAVEL = {"highway": "track", "surface": "gravel"}


def _pair(gap=2.0):
    return (((0.0, 0.0), (20.0, 0.0)),
            ((20.0 + gap, 0.0), (40.0 + gap, 0.0)))


def test_aligned_road_paint_join_eliminates_short_gap_and_preserves_interiors():
    lines = _pair()
    joined = painting.snap_loose_road_ends(
        _dataset(PAVED, PAVED), lines, _spec(), reach_metres=3.0,
    )
    assert joined[0][-1] == joined[1][0] == (21.0, 0.0)
    assert joined[0][0] == lines[0][0]
    assert joined[1][-1] == lines[1][-1]
    assert painting.snap_loose_road_ends(
        _dataset(PAVED, PAVED), joined, _spec(), reach_metres=3.0,
    ) == joined


def test_refuses_distant_or_disabled_join():
    data = _dataset(PAVED, PAVED)
    assert painting.snap_loose_road_ends(data, _pair(4.0), _spec(), reach_metres=3.0) == _pair(4.0)
    assert painting.snap_loose_road_ends(data, _pair(), _spec(), reach_metres=0.0) == _pair()


def test_refuses_misaligned_and_parallel_road_ends():
    data = _dataset(PAVED, PAVED)
    for other in (
        ((22.0, 0.0), (22.0, 20.0)),
        ((22.0, 2.0), (42.0, 2.0)),
        ((22.0, 0.0), (2.0, 0.0)),
    ):
        lines = (((0.0, 0.0), (20.0, 0.0)), other)
        assert painting.snap_loose_road_ends(data, lines, _spec(), reach_metres=3.0) == lines


def test_refuses_mixed_surface_and_grade_separation():
    lines = _pair()
    for tags in (GRAVEL, {**PAVED, "bridge": "yes"}, {**PAVED, "layer": "1"}):
        assert painting.snap_loose_road_ends(
            _dataset(PAVED, tags), lines, _spec(), reach_metres=3.0,
        ) == lines


def test_refuses_ambiguous_multiple_endpoints():
    lines = (
        ((0.0, 0.0), (20.0, 0.0)),
        ((22.0, 0.0), (40.0, 0.0)),
        ((22.2, 0.0), (42.0, 0.0)),
    )
    assert painting.snap_loose_road_ends(
        _dataset(PAVED, PAVED, PAVED), lines, _spec(), reach_metres=3.0,
    ) == lines


def test_keeps_short_segments_from_being_distorted():
    lines = (((17.0, 0.0), (20.0, 0.0)), ((22.0, 0.0), (42.0, 0.0)))
    assert painting.snap_loose_road_ends(
        _dataset(PAVED, PAVED), lines, _spec(), reach_metres=3.0,
    ) == lines


def test_environment_gate_leaves_default_projector_unchanged(monkeypatch):
    lines = _pair()
    monkeypatch.setattr(
        painting, "_ORIGINAL_PROJECT",
        lambda dataset, projection, spec: lines,
    )
    monkeypatch.delenv("CWR_ROAD_PAINT_SNAP_METRES", raising=False)
    data, spec = _dataset(PAVED, PAVED), _spec()
    assert painting._project_with_joined_strokes(data, None, spec) is lines
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    assert painting._project_with_joined_strokes(data, None, spec)[0][-1] == (21.0, 0.0)
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "invalid")
    assert painting._project_with_joined_strokes(data, None, spec) is lines
