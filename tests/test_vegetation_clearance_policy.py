from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen.model import WorldObject
from cwr_worldgen.vegetation_clearance_policy import filter_vegetation_objects


class _IdentityProjection:
    world_size = 200.0

    @staticmethod
    def to_world(point):
        return float(point[0]), float(point[1])


def _dataset():
    runway = SimpleNamespace(
        osm_key="way/runway",
        tags={"aeroway": "runway", "width": "30"},
        points=((10.0, 30.0), (190.0, 30.0)),
    )
    pitch_polygon = SimpleNamespace(
        outer=((60.0, 90.0), (140.0, 90.0), (140.0, 140.0), (60.0, 140.0), (60.0, 90.0)),
        holes=(),
    )
    pitch = SimpleNamespace(
        osm_key="way/pitch",
        tags={"site": "sports_pitch", "sport": "soccer"},
        polygons=(pitch_polygon,),
    )
    return SimpleNamespace(aeroway_lines=(runway,), sites=(pitch,))


def _spec():
    return SimpleNamespace(
        name="clearworld",
        ground_texture_profile="nogova",
        forest_tree_model=r"data3d\str smrk.p3d",
        forest_single_tree_model=r"data3d\str dub.p3d",
        steep_hill_bush_models=(r"data3d\Krovi2.p3d",),
    )


def test_trees_and_bushes_are_removed_from_runway_and_pitch_only() -> None:
    objects = (
        WorldObject(1, r"data3d\str smrk.p3d", 80.0, 0.0, 30.0),
        WorldObject(2, r"data3d\Krovi2.p3d", 100.0, 0.0, 110.0),
        WorldObject(3, r"data3d\str dub.p3d", 180.0, 0.0, 180.0),
        WorldObject(4, r"O\Hous\dum01.p3d", 100.0, 0.0, 110.0),
    )

    filtered, report = filter_vegetation_objects(
        objects, _dataset(), _IdentityProjection(), _spec()
    )

    assert tuple(obj.object_id for obj in filtered) == (3, 4)
    assert report["removed"] == 2
    assert report["runway"] == 1
    assert report["sports_pitch"] == 1


def test_nonvegetation_on_clear_zone_is_not_removed() -> None:
    objects = (
        WorldObject(1, r"O\Road\sil25.p3d", 80.0, 0.0, 30.0),
        WorldObject(2, r"O\Hous\dum01.p3d", 100.0, 0.0, 110.0),
    )
    filtered, report = filter_vegetation_objects(
        objects, _dataset(), _IdentityProjection(), _spec()
    )
    assert filtered == objects
    assert report["removed"] == 0
