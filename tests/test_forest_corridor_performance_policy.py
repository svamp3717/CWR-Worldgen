from cwr_worldgen import forest_corridor_performance_policy as policy
from cwr_worldgen import osm


def test_rebucketed_corridors_preserve_exact_rectangle_hits() -> None:
    corridors = (
        ((10.0, 10.0), (90.0, 10.0), 3.0),
        ((50.0, 40.0), (50.0, 90.0), 6.0),
        ((70.0, 70.0), (90.0, 90.0), 1.0),
    )
    source = osm.IndexedRoadCorridors(
        corridors=corridors,
        bucket_size=100.0,
        buckets={(0, 0): (0, 1, 2)},
    )
    fine = policy._rebucket_index(source, 20.0)

    assert fine.corridors is source.corridors
    assert fine.bucket_size == 20.0
    assert len(fine.buckets) > len(source.buckets)

    cases = (
        (20.0, 10.0, 2.0),
        (20.0, 20.0, 2.0),
        (50.0, 55.0, 4.0),
        (61.0, 55.0, 4.0),
        (80.0, 80.0, 3.0),
        (95.0, 95.0, 2.0),
    )
    brute = tuple(corridors)
    for x, z, size in cases:
        expected = osm.forest_block_intersects_road_corridors(
            brute, x, z, block_size=size
        )
        assert (
            osm.forest_block_intersects_road_corridors(
                source, x, z, block_size=size
            )
            is expected
        )
        assert (
            osm.forest_block_intersects_road_corridors(
                fine, x, z, block_size=size
            )
            is expected
        )


def test_rebucket_policy_never_coarsens_source_index(monkeypatch) -> None:
    monkeypatch.setenv("CWR_WORLDGEN_FOREST_ROAD_BUCKET", "500")
    assert policy._configured_bucket(100.0) == 100.0
