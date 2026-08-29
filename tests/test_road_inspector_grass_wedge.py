# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_grass_wedge as _grass


def _endpoint(
    object_id: int,
    *,
    family: str = "sil",
    kind: str = "straight",
    endpoint_index: int,
    tangent: float,
    outward: float,
    point=(100.0, 200.0),
):
    return _core._endpoint(
        object_id=object_id,
        model_path=rf"o\road\{family}6.p3d",
        family=family,
        kind=kind,
        endpoint_index=endpoint_index,
        point=point,
        tangent_axis_degrees=tangent,
        outward_heading_degrees=outward,
    )


def _road(object_id: int, first, second, *, family="sil", kind="straight"):
    return _core.RoadObject(
        object_id=object_id,
        model_path=rf"o\road\{family}6.p3d",
        x=100.0,
        y=10.0,
        z=200.0,
        heading_degrees=float(first.tangent_axis_degrees),
        pitch_degrees=0.0,
        family=family,
        kind=kind,
        nominal_length_metres=6.25,
        logical_center=(100.0, 200.0),
        endpoints=(first, second),
    )


def _issue(first_id=1, second_id=2, *, category="straight_miter"):
    return _core.RoadIssue(
        issue_id="RI-00001",
        severity="high",
        score=70.0,
        category=category,
        x=100.0,
        z=200.0,
        object_ids=(first_id, second_id),
        models=(r"o\road\sil6.p3d", r"o\road\sil6.p3d"),
        message="legacy seam diagnostic",
        candidate_fix="legacy candidate",
        metrics={
            "center_gap_metres": 0.0,
            "tangent_error_degrees": 20.0,
            "edge_gap_max_metres": 1.5,
        },
    )


def test_forward_edge_rays_find_the_outer_grass_triangle() -> None:
    incoming = _endpoint(1, endpoint_index=1, tangent=0.0, outward=0.0)
    outgoing = _endpoint(2, endpoint_index=0, tangent=20.0, outward=200.0)

    geometry = _grass._grass_wedge_geometry(incoming, outgoing)

    assert geometry is not None
    area, depth, opening, _apex, _centroid, first_extension, second_extension, turn, center_gap = geometry
    assert area > 0.10
    assert depth > 0.10
    assert opening > 1.5
    assert 0.79 < first_extension < 0.82
    assert 0.79 < second_extension < 0.82
    assert turn == 20.0
    assert center_gap == 0.0


def test_straight_tangent_continuation_is_not_a_grass_wedge() -> None:
    incoming = _endpoint(1, endpoint_index=1, tangent=0.0, outward=0.0)
    outgoing = _endpoint(2, endpoint_index=0, tangent=0.0, outward=180.0)

    assert _grass._grass_wedge_geometry(incoming, outgoing) is None


def test_tiny_edge_sliver_stays_below_visible_wedge_threshold() -> None:
    incoming = _endpoint(1, endpoint_index=1, tangent=0.0, outward=0.0)
    outgoing = _endpoint(2, endpoint_index=0, tangent=4.0, outward=184.0)

    assert _grass._grass_wedge_geometry(incoming, outgoing) is None


def test_paved_miter_is_reclassified_with_explicit_wedge_metrics() -> None:
    incoming = _endpoint(1, endpoint_index=1, tangent=0.0, outward=0.0)
    outgoing = _endpoint(2, endpoint_index=0, tangent=20.0, outward=200.0)
    first_road = _road(1, _endpoint(1, endpoint_index=0, tangent=0.0, outward=180.0, point=(100.0, 193.75)), incoming)
    second_road = _road(2, outgoing, _endpoint(2, endpoint_index=1, tangent=20.0, outward=20.0, point=(102.1376, 205.8731)))

    classified = _grass._classify_grass_wedge(
        _issue(),
        (first_road, second_road),
        (),
        0.75,
    )

    assert classified.category == "grass_wedge"
    assert classified.metrics["grass_wedge_detector"] == "forward_edge_ray_miter_triangle"
    assert float(classified.metrics["grass_wedge_area_square_metres"]) > 0.10
    assert float(classified.metrics["grass_wedge_depth_metres"]) > 0.10
    assert "Do not hide" in classified.candidate_fix


def test_dirt_miter_is_not_promoted_to_grass_wedge() -> None:
    incoming = _endpoint(1, family="ces", endpoint_index=1, tangent=0.0, outward=0.0)
    outgoing = _endpoint(2, family="ces", endpoint_index=0, tangent=20.0, outward=200.0)
    first_road = _road(1, _endpoint(1, family="ces", endpoint_index=0, tangent=0.0, outward=180.0, point=(100.0, 193.75)), incoming, family="ces")
    second_road = _road(2, outgoing, _endpoint(2, family="ces", endpoint_index=1, tangent=20.0, outward=20.0, point=(102.1376, 205.8731)), family="ces")
    issue = _issue()

    classified = _grass._classify_grass_wedge(issue, (first_road, second_road), (), 0.75)

    assert classified is issue


def test_source_junction_neighbourhood_is_left_to_intersection_diagnostics() -> None:
    incoming = _endpoint(1, endpoint_index=1, tangent=0.0, outward=0.0)
    outgoing = _endpoint(2, endpoint_index=0, tangent=20.0, outward=200.0)
    first_road = _road(1, _endpoint(1, endpoint_index=0, tangent=0.0, outward=180.0, point=(100.0, 193.75)), incoming)
    second_road = _road(2, outgoing, _endpoint(2, endpoint_index=1, tangent=20.0, outward=20.0, point=(102.1376, 205.8731)))
    junction = _core.SourceJunction((100.0, 200.0), (0.0, 90.0, 180.0))
    issue = _issue()

    classified = _grass._classify_grass_wedge(
        issue,
        (first_road, second_road),
        (junction,),
        0.75,
    )

    assert classified is issue
