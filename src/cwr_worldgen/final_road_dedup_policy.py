# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservatively remove redundant overlapping straight road slabs.

This pass runs after the complete road fitting/junction policy chain.  It only
considers ordinary straight road pieces whose model dimensions are known.  Curves,
junctions, bridges and unknown road models are deliberately left alone: reducing
those safely requires richer geometry/provenance than a final WorldObject carries.

The implementation is spatially indexed.  A candidate is compared only with kept
pieces whose expanded axis bounds share a 25 m bucket, avoiding the O(N^2) scan
that would be rather unkind on worlds with tens of thousands of road objects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Callable

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _quality

_BUCKET_METRES = 25.0
_MAXIMUM_AXIS_ANGLE_DEGREES = 6.0
_ALIGNMENT_COSINE = math.cos(math.radians(_MAXIMUM_AXIS_ANGLE_DEGREES))
_MINIMUM_SHORTER_AXIS_OVERLAP = 0.70
_MAXIMUM_VERTICAL_SEPARATION_METRES = 0.75
_PROGRESS_BUCKET_PERCENT = 2
_RAW_PROGRESS_PERCENT = 99

# These values match the effective half-widths used by the post-build road
# inspector.  They are used only to bound the conservative centre-line offset;
# surface overlap itself is not approximated by polygon clipping here.
_HALF_WIDTH_METRES = {
    "sil": 4.55,
    "kos": 4.55,
    "asf": 3.50,
    "ces": 1.75,
    "gravel": 2.30,
}

_STOCK_STRAIGHT = re.compile(r"^(?P<family>sil|kos|asf|ces)(?P<nominal>25|12|6)\.p3d$", re.I)
_GRAVEL_STRAIGHT = re.compile(r"^gravel(?P<nominal>25|12|6|3)\.p3d$", re.I)

_INSTALLED = False
_ORIGINAL_FIT = None


@dataclass(frozen=True, slots=True)
class _RoadAxis:
    object_id: int
    object_index: int
    family: str
    start: tuple[float, float]
    end: tuple[float, float]
    ux: float
    uz: float
    length: float
    half_width: float
    elevation: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            min(self.start[0], self.end[0]),
            min(self.start[1], self.end[1]),
            max(self.start[0], self.end[0]),
            max(self.start[1], self.end[1]),
        )


def _filename(path: str) -> str:
    return path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()


def _family_and_length(model_path: str, configured_long_length: float) -> tuple[str, float] | None:
    filename = _filename(model_path)
    match = _STOCK_STRAIGHT.fullmatch(filename)
    if match is not None:
        family = match.group("family").casefold()
        nominal = int(match.group("nominal"))
        return family, float(configured_long_length) * nominal / 25.0
    match = _GRAVEL_STRAIGHT.fullmatch(filename)
    if match is not None:
        nominal = int(match.group("nominal"))
        return "gravel", float(configured_long_length) * nominal / 25.0
    return None


def _road_axis(obj, object_index: int, spec) -> _RoadAxis | None:
    dimensions = _family_and_length(obj.model_path, float(spec.road_segment_length))
    if dimensions is None:
        return None
    family, expected_length = dimensions
    if expected_length <= 1.0e-6:
        return None
    start, end = _p._model_axis(obj, expected_length)
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length = math.hypot(dx, dz)
    if length <= 1.0e-6:
        return None
    return _RoadAxis(
        object_id=int(obj.object_id),
        object_index=int(object_index),
        family=family,
        start=start,
        end=end,
        ux=dx / length,
        uz=dz / length,
        length=length,
        half_width=float(_HALF_WIDTH_METRES[family]),
        elevation=float(obj.y),
    )


def _surface_priority(family: str) -> int:
    # Final WorldObjects do not retain OSM highway class provenance.  Preserve
    # the strongest information still available: paved beats generated gravel,
    # which beats the stock dirt/earth family.
    if family in {"sil", "kos", "asf"}:
        return 3
    if family == "gravel":
        return 2
    return 1


def _priority(axis: _RoadAxis) -> tuple[int, float, float, int]:
    # Higher surface/width/length wins.  Lower object id is the deterministic
    # tie-break, hence the negation while sorting in reverse.
    return (
        _surface_priority(axis.family),
        axis.half_width,
        axis.length,
        -axis.object_id,
    )


def _bucket_range(minimum: float, maximum: float) -> range:
    return range(
        math.floor(minimum / _BUCKET_METRES),
        math.floor(maximum / _BUCKET_METRES) + 1,
    )


def _buckets_for(axis: _RoadAxis) -> tuple[tuple[int, int], ...]:
    min_x, min_z, max_x, max_z = axis.bounds
    # No candidate can be accepted beyond this conservative centre-line offset.
    padding = min(2.0, axis.half_width * 0.55)
    return tuple(
        (bx, bz)
        for bz in _bucket_range(min_z - padding, max_z + padding)
        for bx in _bucket_range(min_x - padding, max_x + padding)
    )


def _axis_angle_is_close(first: _RoadAxis, second: _RoadAxis) -> bool:
    return abs(first.ux * second.ux + first.uz * second.uz) >= _ALIGNMENT_COSINE


def _mean_lateral_offset(reference: _RoadAxis, other: _RoadAxis) -> float:
    # Signed distance to the infinite reference line, averaged over the other
    # axis endpoints.  The angle gate above keeps this meaningful for long pieces.
    nx, nz = -reference.uz, reference.ux
    first = abs((other.start[0] - reference.start[0]) * nx + (other.start[1] - reference.start[1]) * nz)
    second = abs((other.end[0] - reference.start[0]) * nx + (other.end[1] - reference.start[1]) * nz)
    return (first + second) * 0.5


def _longitudinal_overlap(reference: _RoadAxis, other: _RoadAxis) -> float:
    first = (
        (other.start[0] - reference.start[0]) * reference.ux
        + (other.start[1] - reference.start[1]) * reference.uz
    )
    second = (
        (other.end[0] - reference.start[0]) * reference.ux
        + (other.end[1] - reference.start[1]) * reference.uz
    )
    other_min, other_max = sorted((first, second))
    return max(0.0, min(reference.length, other_max) - max(0.0, other_min))


def _is_redundant(candidate: _RoadAxis, kept: _RoadAxis) -> bool:
    if abs(candidate.elevation - kept.elevation) > _MAXIMUM_VERTICAL_SEPARATION_METRES:
        return False
    if not _axis_angle_is_close(candidate, kept):
        return False

    lateral_limit = min(2.0, min(candidate.half_width, kept.half_width) * 0.55)
    # Use the smaller of the two line-reference measurements so small heading
    # noise does not turn a coincident 25 m slab into a false negative.
    lateral = min(
        _mean_lateral_offset(candidate, kept),
        _mean_lateral_offset(kept, candidate),
    )
    if lateral > lateral_limit:
        return False

    overlap = max(
        _longitudinal_overlap(candidate, kept),
        _longitudinal_overlap(kept, candidate),
    )
    shorter = min(candidate.length, kept.length)
    return shorter > 1.0e-6 and overlap / shorter >= _MINIMUM_SHORTER_AXIS_OVERLAP


def deduplicate_final_road_objects(
    report,
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Return ``report`` with redundant ordinary straight road pieces removed."""
    if not report.objects:
        return report

    protected_prefix = max(0, min(int(report.junction_cap_objects), len(report.objects)))
    axes = []
    for index, obj in enumerate(report.objects):
        # Junction-cap slots are intentionally not compared.  They can overlap a
        # short approach by design, and their prefix count also carries report
        # semantics used by earlier road policies.
        if index < protected_prefix:
            continue
        axis = _road_axis(obj, index, spec)
        if axis is not None:
            axes.append(axis)

    if len(axes) < 2:
        return report

    ordered = sorted(axes, key=_priority, reverse=True)
    total = len(ordered)
    bucket_members: dict[tuple[int, int], list[int]] = {}
    kept_axes: list[_RoadAxis] = []
    removed_ids: set[int] = set()
    comparisons = 0
    last_progress_bucket = -1

    if progress_callback is not None:
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            f"Deduplicating final road pieces (0/{total:,}; 0 removed)",
        )

    for completed, candidate in enumerate(ordered, start=1):
        candidate_buckets = _buckets_for(candidate)
        candidate_indices: set[int] = set()
        for bucket in candidate_buckets:
            candidate_indices.update(bucket_members.get(bucket, ()))

        redundant = False
        for kept_index in sorted(candidate_indices):
            kept = kept_axes[kept_index]
            comparisons += 1
            if _is_redundant(candidate, kept):
                removed_ids.add(candidate.object_id)
                redundant = True
                break

        if not redundant:
            kept_index = len(kept_axes)
            kept_axes.append(candidate)
            for bucket in candidate_buckets:
                bucket_members.setdefault(bucket, []).append(kept_index)

        if progress_callback is not None:
            percent = min(100, int(completed * 100 / total))
            progress_bucket = percent // _PROGRESS_BUCKET_PERCENT
            if completed == total or progress_bucket > last_progress_bucket:
                last_progress_bucket = progress_bucket
                progress_callback(
                    _RAW_PROGRESS_PERCENT,
                    f"Deduplicating final road pieces ({completed:,}/{total:,}, {percent}%; "
                    f"{len(removed_ids):,} removed; {comparisons:,} nearby comparisons)",
                )

    if not removed_ids:
        return report

    objects = tuple(obj for obj in report.objects if int(obj.object_id) not in removed_ids)
    return replace(report, objects=objects)


def install_final_road_dedup_policy() -> None:
    """Install the dedupe wrapper after every existing road fitting policy."""
    global _INSTALLED, _ORIGINAL_FIT
    if _INSTALLED:
        return

    _ORIGINAL_FIT = _p.fit_road_objects

    def deduplicating_fit(
        dataset,
        projection,
        elevations,
        spec,
        *,
        starting_id: int = 1,
        progress_callback=None,
    ):
        report = _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        return deduplicate_final_road_objects(
            report,
            spec,
            progress_callback=progress_callback,
        )

    _p.fit_road_objects = deduplicating_fit
    _generator.fit_road_objects = deduplicating_fit
    _INSTALLED = True
