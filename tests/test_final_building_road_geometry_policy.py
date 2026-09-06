import math

from cwr_worldgen import final_building_road_geometry_policy as geometry


def test_gravel_curve_keeps_nominal_endpoints_on_local_z_axis() -> None:
    points = geometry.gravel_curve_centreline_points(25.0, "r", 30.0)

    assert points[0] == (0.0, -12.5)
    assert points[-1] == (0.0, 12.5)
    assert max(point[0] for point in points) > 0.0


def test_left_and_right_gravel_curves_are_mirrors() -> None:
    right = geometry.gravel_curve_centreline_points(12.5, "r", 45.0)
    left = geometry.gravel_curve_centreline_points(12.5, "l", 45.0)

    assert len(right) == len(left)
    for right_point, left_point in zip(right, left):
        assert math.isclose(right_point[0], -left_point[0], abs_tol=1.0e-9)
        assert math.isclose(right_point[1], left_point[1], abs_tol=1.0e-9)


def test_zero_degree_gravel_curve_is_straight() -> None:
    assert geometry.gravel_curve_centreline_points(6.25, "r", 0.0) == (
        (0.0, -3.125),
        (0.0, 3.125),
    )
