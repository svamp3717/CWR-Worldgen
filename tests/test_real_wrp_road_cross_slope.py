"""Ground the real wg_belowtest23 gravel crossing using its source WRP facts.

Original supplied PBO: wg_belowtest23.pbo, player setPos [3302.524,3167.709,0].
Its 4WVR elevation samples and actual generated road WorldObject transforms
showed uphill gravel verges 7-12 cm below ground despite pavement being higher.
These are minimal numeric fixtures, not redistributed original P3D/WRP assets.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from cwr_worldgen.model import WorldObject, TerrainShearedWorldObject
from cwr_worldgen.osm import _sample_elevation
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen.road_cross_slope_grounding import ground_generated_approaches


def _scene():
    spec = SimpleNamespace(
        name="wg_belowtest23", cells=128, cell_size=50.0,
        road_segment_length=25.0, custom_road_shapes=True,
    )
    heights = [11.3] * (128 * 128)
    # Actual quantized RVW4 cells surrounding X3302 Z3170:
    # map row 63/64, column 66/67, height scale 0.05m.
    for row, col, val in (
        (63,66,11.45), (63,67,10.55),
        (64,66,11.15), (64,67,10.45),
    ):
        heights[row * 128 + col] = val
    roads = (
        WorldObject(
            2101, r"wg_belowtest23\i\road_gravel_w070_l0061.p3d",
            3303.0139, 11.2201, 3173.2576, 8.3, 0.05,
        ),
        WorldObject(
            2100, r"wg_belowtest23\i\road_gravel_w070_l0005.p3d",
            3303.1316, 11.2407, 3174.0676, 8.3, -0.46,
        ),
        WorldObject(
            12316, r"wg_belowtest23\i\road_paved_w091_l0063_l004.p3d",
            3306.1106, 11.3139, 3167.4092, 282.3, 0.86,
        ),
        WorldObject(
            12317, r"wg_belowtest23\i\road_paved_w091_l0063_l005.p3d",
            3300.0564, 11.3963, 3168.9473, 286.2, 0.65,
        ),
    )
    report = RoadFitReport(
        objects=roads, chain_count=2, connection_count=0,
        failed_connections=0, maximum_connection_gap=0.0,
        maximum_chain_gap=0.0, truncated=False, junction_cap_objects=0,
    )
    return spec, heights, report


def _surface_y(obj, x: float, z: float) -> float:
    if isinstance(obj, TerrainShearedWorldObject):
        return (
            obj.y + 0.025
            + obj.terrain_shear_x * (x - obj.x)
            + obj.terrain_shear_z * (z - obj.z)
        )
    heading = math.radians(obj.heading_degrees)
    along = (x - obj.x) * math.sin(heading) + (z - obj.z) * math.cos(heading)
    return (
        obj.y + 0.025 * math.cos(math.radians(obj.pitch_degrees))
        + math.sin(math.radians(obj.pitch_degrees)) * along
    )


def test_uploaded_wrp_reproduces_terrain_burying_gravel_high_side() -> None:
    spec, elevations, report = _scene()
    gravel = report.objects[0]
    assert (
        _surface_y(gravel, 3301.0, 3175.0)
        < _sample_elevation(elevations, spec.cells, spec.cell_size, 3301, 3175)
        - 0.01
    )


def test_real_cross_slope_is_banked_so_gravel_reaches_asphalt_without_burial():
    spec, elevations, report = _scene()
    result = ground_generated_approaches(report, spec, elevations)
    assert result is not report
    gravel = result.objects[0]
    assert isinstance(gravel, TerrainShearedWorldObject)
    assert gravel.terrain_shear_x < -0.01
    assert gravel.pitch_degrees == 0.0
    for x,z in ((3301.0,3175.0),(3302.5,3175.0),(3305.0,3175.0)):
        # Both sides of the 7m-wide gravel approach must be visible even
        # where the terrain slopes laterally across its width.
        terrain = _sample_elevation(elevations, spec.cells, spec.cell_size, x,z)
        assert _surface_y(gravel,x,z) >= terrain + 0.015
    # Pavement remains above the buried approach within their overlap.
    asphalt = report.objects[-1]
    for x,z in ((3302.5,3170.0),(3303.3,3171.0)):
        assert _surface_y(gravel,x,z) < _surface_y(asphalt,x,z) - 0.03
    assert result.objects[2:] == report.objects[2:]


def test_road_bank_uses_native_rvw4_terrain_shear_matrix() -> None:
    spec, elevations, report = _scene()
    result = ground_generated_approaches(report, spec, elevations)
    obj = result.objects[0]
    matrix = obj.matrix_4x3()
    assert matrix[1] != 0.0 and matrix[7] != 0.0
    for local_x,local_z in ((-3.5,0.0),(3.5,0.0)):
        x = matrix[0]*local_x + matrix[6]*local_z + matrix[9]
        z = matrix[2]*local_x + matrix[8]*local_z + matrix[11]
        y = matrix[1]*local_x + matrix[7]*local_z + matrix[10] + 0.025
        assert y == pytest.approx(_surface_y(obj,x,z),abs=1e-5)


def test_flat_ground_does_not_change_deterministic_plain_road_placements():
    spec, _, report = _scene()
    elevations = [11.3] * (128*128)
    result = ground_generated_approaches(report, spec, elevations)
    assert tuple(obj.model_path for obj in result.objects) == tuple(
        obj.model_path for obj in report.objects
    )
    assert tuple(obj.object_id for obj in result.objects) == tuple(
        obj.object_id for obj in report.objects
    )


@pytest.mark.parametrize(
    ("paved_path", "paved_top"),
    (
        (r"wg_belowtest23\i\road_paved_w020_l0020.p3d", 10.025),
        (r"o\road\sil6.p3d", 10.0),
    ),
)
def test_small_paved_overlap_between_road_samples_still_limits_gravel_lift(
    paved_path: str, paved_top: float,
) -> None:
    """An asphalt patch between all nine sample sites still owns its footprint.

    The older nine-point-only clamp lifted the gravel to 10.02 m on this
    flat terrain, leaving just 5 mm below generated asphalt and actually
    sitting 20 mm *above* the stock sil slab. Both must retain 35 mm clearance.
    """
    spec, _, report = _scene()
    elevations = [10.0] * (spec.cells * spec.cells)
    gravel = WorldObject(
        311, r"wg_belowtest23\i\road_gravel_w070_l0250.p3d",
        100.0, 9.9, 100.0, 0.0, 0.0,
    )
    paved = WorldObject(
        312, paved_path, 102.0, 10.0, 104.0, 0.0, 0.0,
    )
    report = replace(report, objects=(gravel, paved))
    result = ground_generated_approaches(report, spec, elevations)
    assert isinstance(result.objects[0], TerrainShearedWorldObject)
    assert result.objects[1] == paved
    assert result.objects[0].y > gravel.y
    # x=102,z=104 is within both surfaces, but is not one of the gravel
    # fitter's three-by-three sample locations.
    assert _surface_y(result.objects[0], 102.0, 104.0) <= (
        paved_top - 0.035 + 1.0e-6
    )
