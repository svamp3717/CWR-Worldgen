from pathlib import Path
import math

import cwr_worldgen.playability as playability
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


def _gravel_feature(projection, osm_key: str, points):
    return OsmLineFeature(
        osm_key,
        {"highway": "unclassified", "surface": "gravel"},
        tuple(projection.to_latlon(point) for point in points),
    )


def _spec(bbox):
    return _Milestone9PlayabilitySpec(
        name="gravel_corner",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=128,
        cell_size=50.0,
        max_road_objects=10000,
        strict_assets=False,
        procedural_gravel_roads=True,
    )


def test_uploaded_lundby_54_degree_gravel_corner_gets_patch() -> None:
    # Geometry copied from the user's normalized Lundby source around the
    # reported player position [3950, 35, 2292].
    bbox = (59.4012347, 16.8211928, 59.4587912, 16.9343614)
    projection = BboxProjection.create(bbox, 6400.0)
    centre = (3943.500087480133, 2295.750436527429)
    southwest = (3840.0, 2096.249771962806)
    northeast = (4098.0000459490875, 2595.7503322819402)
    west_branch = (3898.4999019163347, 2288.9997828219803)
    dataset = OsmDataset(
        source_generator="uploaded-lundby-regression",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            _gravel_feature(projection, "way/main", (southwest, centre, northeast)),
            _gravel_feature(projection, "way/branch", (centre, west_branch)),
        ),
    )
    spec = _spec(bbox)
    report = playability.fit_road_objects(
        dataset,
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )

    patches = [
        obj for obj in report.objects
        if obj.model_path.casefold().endswith(r"\gravel3.p3d")
        and obj.y > -0.020
        and math.dist((obj.x, obj.z), centre) < 5.0
    ]
    assert len(patches) == 1
    patch = patches[0]
    assert abs(((patch.heading_degrees - 234.44 + 180.0) % 360.0) - 180.0) < 1.0
    assert 2.5 <= math.dist((patch.x, patch.z), centre) <= 3.4


def test_right_angle_gravel_t_needs_no_corner_patch() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 6400.0)
    centre = (3200.0, 3200.0)
    dataset = OsmDataset(
        source_generator="right-angle-gravel-t",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            _gravel_feature(
                projection,
                "way/main",
                ((3200.0, 2900.0), centre, (3200.0, 3500.0)),
            ),
            _gravel_feature(projection, "way/branch", (centre, (3500.0, 3200.0))),
        ),
    )
    spec = _spec(bbox)
    report = playability.fit_road_objects(
        dataset,
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )
    # Normal flat generated gravel is centred at y=-0.025 so its visual top is
    # exactly on terrain. Only the corner patch is raised to y=-0.017.
    assert not any(obj.y > -0.020 for obj in report.objects)
