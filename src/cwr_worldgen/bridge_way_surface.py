# SPDX-License-Identifier: GPL-3.0-or-later
"""Short stock road covers for buried bridge abutments in the native-ways profile.

The stock bridge has its own Roadway LOD, but the solved terrain can cover the
first few metres of its deck. Native .ways geometry follows that terrain, so
its texture cannot restore a drivable visible surface above the embankment.
Cover *only* occluded ends with real, elevated stock road P3Ds; preserve the
stock bridge over the open crossing and the .ways everywhere else.
"""
from __future__ import annotations

import math
from typing import Sequence

from .bridge_render_policy import (
    _bridge_components, _ordered_component, _visible_deck_point,
    _STOCK_ROADWAY_HALF_LENGTH_METRES,
)
from .model import WorldObject
from .osm import _sample_elevation

# These are installed stock CWA road models, not generated P3Ds in the PBO.
_STOCK_COVER_PIECES = (
    (25.0, r"O\road\sil25.p3d"),
    (12.0, r"O\road\sil12.p3d"),
    (6.0, r"O\road\sil6.p3d"),
)
_OUTER_DECK_OVERLAP_METRES = 6.0
_CLEAR_DECK_OVERLAP_METRES = 4.0
_MIN_BURIED_DEPTH_METRES = -0.08
_VISIBLE_SURFACE_CLEARANCE_METRES = 0.09
_SAMPLE_STEP_METRES = 2.0
_MAX_PITCH_DEGREES = 13.0
_ROAD_HALF_WIDTH_METRES = 3.0


def _deck_component_ends(bridges, indices):
    first = _visible_deck_point(bridges[indices[0]], -_STOCK_ROADWAY_HALF_LENGTH_METRES)
    last = _visible_deck_point(bridges[indices[-1]], _STOCK_ROADWAY_HALF_LENGTH_METRES)
    return first, last


def _cover_lengths(distance: float) -> list[tuple[float, str]]:
    """Minimize bank/deck overhang using real stock straight-piece lengths.

    Prefer covering to within 0.5 m of the target with few large pieces.
    Avoid the naive 25+12 choice for a 31 m bank, where 25+6 fits.
    """
    best = None
    for long_count in range(int(distance / 25.0) + 2):
        for medium_count in range(int(distance / 12.0) + 2):
            for short_count in range(int(distance / 6.0) + 2):
                count = long_count + medium_count + short_count
                if count == 0:
                    continue
                coverage = 25 * long_count + 12 * medium_count + 6 * short_count
                shortfall = max(0.0, distance - coverage)
                excess = max(0.0, coverage - distance)
                if shortfall > 0.5 or excess > 6.0:
                    continue
                ranking = (round(excess + shortfall * 2.0, 4),
                           count, -long_count)
                if best is None or ranking < best[0]:
                    best = (ranking, long_count, medium_count, short_count)
    if best is None:
        return []
    _, long_count, medium_count, short_count = best
    return ([ _STOCK_COVER_PIECES[0] ] * long_count
            + [ _STOCK_COVER_PIECES[1] ] * medium_count
            + [ _STOCK_COVER_PIECES[2] ] * short_count)


def _road_shoulder_ground(
    x: float, z: float, outward: tuple[float, float],
    elevations: Sequence[float], spec,
) -> float:
    """Sample the full paved width, not only the grass at its centre."""
    nx, nz = -outward[1], outward[0]
    return max(
        _sample_elevation(
            elevations, spec.cells, spec.cell_size,
            x + nx * side, z + nz * side,
        )
        for side in (-_ROAD_HALF_WIDTH_METRES, 0.0, _ROAD_HALF_WIDTH_METRES)
    )


def bridge_road_surface_covers(
    bridge_objects: Sequence[WorldObject],
    elevations: Sequence[float],
    spec,
    *,
    starting_id: int = 1,
) -> tuple[WorldObject, ...]:
    """Make asphalt visible/drivable over stock bridge ends buried by terrain.

    Only a few straight models are created at the two abutments. Detect burial
    from final solved elevations versus the *visible* deck model coordinates.
    All output placements are deterministic and separate from legacy road_fit.
    """
    bridges = tuple(bridge_objects)
    if not bridges or len(elevations) == 0:
        return ()
    covers: list[WorldObject] = []
    next_id = starting_id
    for component in _bridge_components(bridges):
        state = _ordered_component(bridges, component)
        if state is None:
            continue
        ordered, _axis, _midpoint = state
        deck_start, deck_end = _deck_component_ends(bridges, ordered)
        length = math.hypot(
            deck_end[0] - deck_start[0], deck_end[2] - deck_start[2]
        )
        if length < 8.0:
            continue
        direction = (
            (deck_end[0] - deck_start[0]) / length,
            (deck_end[2] - deck_start[2]) / length,
        )

        for side in (1, -1):
            if side > 0:
                end = deck_start
                inward = direction
                height_slope = (deck_end[1] - deck_start[1]) / length
            else:
                end = deck_end
                inward = (-direction[0], -direction[1])
                height_slope = (deck_start[1] - deck_end[1]) / length

            def position(distance: float) -> tuple[float, float]:
                return (
                    end[0] + inward[0] * distance,
                    end[2] + inward[1] * distance,
                )

            def terrain(distance: float) -> float:
                x, z = position(distance)
                return _road_shoulder_ground(x, z, inward, elevations, spec)

            def deck(distance: float) -> float:
                return end[1] + height_slope * distance

            # Search no farther than the component midpoint on either side:
            # opposing covers must not overlap across a long bridge.
            scan_length = length * 0.5
            buried_end = -1.0
            first_buried = None
            clear_distance = 0.0
            count = max(1, int(math.ceil(scan_length / _SAMPLE_STEP_METRES)))
            for index in range(count + 1):
                d = min(scan_length, index * _SAMPLE_STEP_METRES)
                x, z = position(d)
                if not (3.0 <= x < spec.world_size - 3.0
                        and 3.0 <= z < spec.world_size - 3.0):
                    break
                if terrain(d) >= deck(d) + _MIN_BURIED_DEPTH_METRES:
                    if first_buried is None:
                        first_buried = d
                    buried_end = d
                    clear_distance = 0.0
                elif first_buried is not None:
                    clear_distance += _SAMPLE_STEP_METRES
                    if clear_distance >= 8.0:
                        break
            if first_buried is None or first_buried > 15.0:
                continue

            # Cover six metres of the dry approach as well; otherwise a gap
            # can remain between a terrain-following .ways ribbon and the
            # elevated road model. Run four metres into clear stock deck.
            target = min(
                scan_length,
                buried_end + _CLEAR_DECK_OVERLAP_METRES,
            )
            start_distance = -_OUTER_DECK_OVERLAP_METRES
            remaining = target - start_distance
            cursor = start_distance
            for piece_length, model in _cover_lengths(remaining):
                # Skip any model overhanging outside the WRP.
                x0, z0 = position(cursor)
                x1, z1 = position(cursor + piece_length)
                if not all(
                    3.5 < coord < spec.world_size - 3.5
                    for coord in (x0, z0, x1, z1)
                ):
                    cursor += piece_length
                    continue

                def top(distance: float) -> float:
                    ground = terrain(distance) + _VISIBLE_SURFACE_CLEARANCE_METRES
                    return max(
                        ground,
                        deck(distance) + _VISIBLE_SURFACE_CLEARANCE_METRES,
                    ) if distance >= 0 else ground

                y0, y1 = top(cursor), top(cursor + piece_length)
                # If a narrow peak would pierce the middle of the slab, lift
                # it slightly at both ends. Avoid huge vertical jumps.
                lift = max(
                    0.0,
                    *(
                        top(cursor + piece_length * t / 4.0)
                        - (y0 + (y1-y0) * t / 4.0)
                        for t in range(1, 4)
                    ),
                )
                if lift > 0.4:
                    cursor += piece_length
                    continue
                y0 += lift
                y1 += lift
                pitch = math.degrees(
                    math.atan2(y1 - y0, piece_length)
                )
                if abs(pitch) > _MAX_PITCH_DEGREES:
                    cursor += piece_length
                    continue
                x = (x0 + x1) * 0.5
                z = (z0 + z1) * 0.5
                covers.append(WorldObject(
                    next_id, model, x, (y0 + y1) * 0.5, z,
                    math.degrees(math.atan2(
                        inward[0], inward[1]
                    )) % 360.0, pitch,
                ))
                next_id += 1
                cursor += piece_length
                if cursor >= target - 0.5:
                    break

    return tuple(covers)
