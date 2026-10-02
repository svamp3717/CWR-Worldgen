from __future__ import annotations

import math

from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from cwr_worldgen.normalization import (
    _coastline_ocean,
    _nearest_coastline_segment_indexes,
)


def test_indexed_coastline_nearest_segments_match_bruteforce_order() -> None:
    segments = (
        LineString(((0.0, 0.0), (10.0, 0.0))),
        LineString(((0.0, 10.0), (10.0, 10.0))),
        LineString(((20.0, 0.0), (20.0, 10.0))),
    )
    points = (
        Point(4.0, 1.0),
        Point(4.0, 9.0),
        Point(19.0, 6.0),
        # Exactly equidistant from the first two segments. Python min historically
        # picked the first segment, so the indexed path must preserve that tie.
        Point(5.0, 5.0),
    )
    expected = tuple(
        min(
            range(len(segments)),
            key=lambda index: segments[index].distance(point),
        )
        for point in points
    )

    assert _nearest_coastline_segment_indexes(segments, points) == expected


def test_archipelago_coastline_reconstruction_keeps_islands_dry() -> None:
    boundary = box(0.0, 0.0, 1000.0, 1000.0)
    side = 12.0
    lines = []
    centres = []
    for row in range(12):
        for column in range(12):
            x = 25.0 + column * 75.0
            y = 25.0 + row * 75.0
            # Counter-clockwise coastline: OSM semantics put land on the left.
            lines.append(
                LineString(
                    (
                        (x, y),
                        (x + side, y),
                        (x + side, y + side),
                        (x, y + side),
                        (x, y),
                    )
                )
            )
            centres.append(Point(x + side * 0.5, y + side * 0.5))

    progress = []
    ocean = _coastline_ocean(
        lines,
        boundary,
        progress_callback=lambda percent, message: progress.append(
            (percent, message)
        ),
    )
    merged = unary_union(ocean)

    assert merged.covers(Point(5.0, 5.0))
    assert all(not merged.covers(point) for point in centres)
    expected_area = boundary.area - len(lines) * side * side
    assert math.isclose(merged.area, expected_area, rel_tol=0.0, abs_tol=1.0e-6)
    assert any("Indexing" in message for _percent, message in progress)
    assert progress[-1][0] == 100
