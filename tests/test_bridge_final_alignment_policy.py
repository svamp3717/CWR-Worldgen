from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_final_alignment_policy as policy
from cwr_worldgen import bridge_render_policy as bridge
from cwr_worldgen import bridge_underlay_cleanup_policy as cleanup
from cwr_worldgen import bridge_water_deck_clamp_policy as clamp
from cwr_worldgen.model import WorldObject


def _spec():
    return SimpleNamespace(
        sea_level=0.0,
        cells=64,
        cell_size=50.0,
        world_size=3200.0,
    )


def test_tinybridgetest10_parallel_sil_inside_stock_width_is_underlay() -> None:
    span = cleanup._BridgeSpan(
        points=((591.407, 666.322), (928.224, 884.896)),
        road_width=7.0,
    )
    # PBO10 has surviving sil pieces around 5.4-5.5 m from the bridge chord.
    duplicate = WorldObject(
        1072,
        r"o\road\sil25.p3d",
        881.958,
        7.105,
        861.477,
        236.793,
        -0.248,
    )
    separate_parallel = WorldObject(
        2000,
        r"o\road\sil25.p3d",
        881.958,
        7.105,
        869.0,
        236.793,
        0.0,
    )

    assert policy._inside_emitted_stock_footprint(duplicate, span)
    assert not policy._inside_emitted_stock_footprint(separate_parallel, span)


def test_crossing_road_inside_bridge_width_is_preserved_by_heading() -> None:
    span = cleanup._BridgeSpan(
        points=((0.0, 0.0), (100.0, 0.0)),
        road_width=7.0,
    )
    crossing = WorldObject(
        1,
        r"o\road\sil25.p3d",
        50.0,
        0.0,
        5.5,
        0.0,
        0.0,
    )
    assert not policy._inside_emitted_stock_footprint(crossing, span)


def test_final_bridge_deck_matches_high_road_surface_after_legacy_tuning() -> None:
    spec = _spec()
    raw_road_surface = 7.035

    with patch.object(
        clamp,
        "_ORIGINAL_DRY_APPROACH_HEIGHT",
        return_value=raw_road_surface,
    ):
        pre_tuning = policy._final_road_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    final_deck = pre_tuning - bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - raw_road_surface) < 1e-12


def test_low_bank_still_finishes_at_tide_safe_deck() -> None:
    spec = _spec()

    with patch.object(
        clamp,
        "_ORIGINAL_DRY_APPROACH_HEIGHT",
        return_value=0.05,
    ):
        pre_tuning = policy._final_road_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    final_deck = pre_tuning - bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - 5.5) < 1e-12


def test_tinybridgetest11_prefers_continuing_road_chain_over_isolated_cap() -> None:
    bridge_point = (591.4072485, 666.3218953)
    bridge_heading = 57.01882435

    # Object 12 is visually closer to the PBO11 bridge, but its opposite endpoint
    # does not continue into another fitted road object. Object 853 does continue
    # into 852, so it is the real approach chain the connector should overlap.
    cap = WorldObject(
        12,
        r"o\road\sil6.p3d",
        588.7501831,
        7.06,
        664.4996338,
        231.6882537,
        0.0,
    )
    approach = WorldObject(
        853,
        r"o\road\sil6.p3d",
        586.0164185,
        7.035,
        662.0222168,
        47.8162586,
        0.0,
    )
    continuation = WorldObject(
        852,
        r"o\road\sil6.p3d",
        581.6593628,
        7.035,
        658.0737305,
        47.8162586,
        0.0,
    )

    selected = policy._approach_candidate(
        bridge_point,
        bridge_heading,
        (cap, approach, continuation),
    )
    assert selected is not None
    candidate, near, gap = selected
    assert candidate.object_id == 853
    assert 3.8 < gap < 4.1
    assert math.dist((near[0], near[2]), bridge_point) == gap


def test_tinybridgetest11_connector_starts_exactly_on_bridge_abutment() -> None:
    bridge_point = (928.2240006, 884.8959707)
    bridge_deck_y = 7.1565831
    bridge_heading = 57.01882435
    approach = WorldObject(
        1068,
        r"o\road\sil6.p3d",
        932.2178345,
        7.1381631,
        886.4214478,
        249.0762802,
        0.2536515,
    )
    continuation = WorldObject(
        1067,
        r"o\road\sil6.p3d",
        937.7100830,
        7.1102862,
        888.5213623,
        249.0762803,
        0.2896126,
    )

    selected = policy._approach_candidate(
        bridge_point,
        bridge_heading,
        (approach, continuation),
    )
    assert selected is not None
    candidate, near, gap = selected
    assert candidate.object_id == 1068
    assert 1.1 < gap < 1.5

    filler = policy._approach_filler(
        2000,
        bridge_point,
        (0.0, 1.0),
        bridge_heading,
        candidate,
        near,
        gap,
        bridge_deck_y,
    )
    assert filler is not None
    endpoints = policy._road_endpoints(filler)
    assert endpoints is not None
    bridge_end = min(
        endpoints,
        key=lambda point: math.dist(
            (point[0], point[2]),
            bridge_point,
        ),
    )
    assert math.dist(
        (bridge_end[0], bridge_end[2]),
        bridge_point,
    ) < 1.0e-6
    assert abs(bridge_end[1] - bridge_deck_y) < 1.0e-6
    assert filler.model_path.casefold().endswith(r"\sil6.p3d")


def test_connector_is_buried_at_existing_road_join() -> None:
    bridge_point = (928.2240006, 884.8959707)
    bridge_deck_y = 7.1565831
    bridge_heading = 57.01882435
    approach = WorldObject(
        1068,
        r"o\road\sil6.p3d",
        932.2178345,
        7.1381631,
        886.4214478,
        249.0762802,
        0.2536515,
    )
    continuation = WorldObject(
        1067,
        r"o\road\sil6.p3d",
        937.7100830,
        7.1102862,
        888.5213623,
        249.0762803,
        0.2896126,
    )

    selected = policy._approach_candidate(
        bridge_point,
        bridge_heading,
        (approach, continuation),
    )
    assert selected is not None
    candidate, near, gap = selected
    filler = policy._approach_filler(
        2000,
        bridge_point,
        (0.0, 1.0),
        bridge_heading,
        candidate,
        near,
        gap,
        bridge_deck_y,
    )
    assert filler is not None

    # The filler is a straight plane. Evaluate its height at the horizontal
    # location where it first overlaps the existing approach road.
    filler_pitch = math.radians(filler.pitch_degrees)
    join_y = bridge_deck_y + gap * math.tan(filler_pitch)
    assert abs(
        join_y
        - (near[1] - policy._APPROACH_FILL_OVERLAP_BURY_METRES)
    ) < 1.0e-6


def test_connector_is_not_added_when_road_already_meets_abutment() -> None:
    bridge_point = (100.0, 100.0)
    road = WorldObject(
        1,
        r"o\road\sil6.p3d",
        100.0,
        7.0,
        97.0,
        0.0,
        0.0,
    )
    # Local +z endpoint is exactly (100,100), so no filler should be needed.
    selected = policy._approach_candidate(
        bridge_point,
        0.0,
        (road,),
    )
    assert selected is not None
    candidate, near, gap = selected
    assert candidate is road
    assert gap < policy._APPROACH_FILL_MIN_GAP_METRES
    assert policy._approach_filler(
        2,
        bridge_point,
        (0.0, -1.0),
        0.0,
        candidate,
        near,
        gap,
        7.0,
    ) is None
