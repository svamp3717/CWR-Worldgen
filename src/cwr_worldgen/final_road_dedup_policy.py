# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservatively remove redundant overlapping road slabs.

This pass runs after the complete road fitting/junction policy chain. It handles
ordinary stock straights, stock 10-degree curves, generated paved fallback
ribbons, and generated gravel straights whose dimensions are encoded or known.
Junction hubs, bridges and unknown road models remain deliberately excluded.

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
_MAXIMUM_PAVED_COVERAGE_ANGLE_DEGREES = 12.0
_PAVED_COVERAGE_ALIGNMENT_COSINE = math.cos(
    math.radians(_MAXIMUM_PAVED_COVERAGE_ANGLE_DEGREES)
)
_MINIMUM_PAVED_CANDIDATE_COVERAGE = 0.90
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

_STOCK_STRAIGHT = re.compile(
    r"^(?P<family>sil|kos|asf|ces)(?P<nominal>25|12|6)\.p3d$", re.I
)
_STOCK_CURVE = re.compile(
    r"^(?P<family>sil|kos|asf)10 (?P<radius>25|50|75|100)\.p3d$", re.I
)
_GENERATED_PAVED = re.compile(
    r"^paved_w(?P<width>\d{3})_l(?P<length>\d{4})"
    r"(?:_[lr](?:05|10|15|20|25|30|35|40|45))?\.p3d$",
    re.I,
)
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
    stock_model: bool

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


def _world_point(
    local: tuple[float, float],
    obj,
) -> tuple[float, float]:
    angle = math.radians(float(obj.heading_degrees))
    return (
        float(obj.x) + local[0] * math.cos(angle) + local[1] * math.sin(angle),
        float(obj.z) - local[0] * math.sin(angle) + local[1] * math.cos(angle),
    )


def _stock_curve_axis(
    obj,
    family: str,
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    # Match the stock 10-degree connector geometry used by paved-junction
    # fitting. The stock curve origin is not the chord midpoint.
    angle = math.radians(10.0)
    half = angle * 0.5
    chord = 2.0 * float(radius) * math.sin(half)
    half_width = float(_HALF_WIDTH_METRES[family])
    midpoint = (
        half_width * (1.0 - math.cos(angle)) * 0.5,
        -half_width * math.sin(angle) * 0.5,
    )
    unit = (math.sin(half), math.cos(half))
    start_local = (
        midpoint[0] - unit[0] * chord * 0.5,
        midpoint[1] - unit[1] * chord * 0.5,
    )
    end_local = (
        midpoint[0] + unit[0] * chord * 0.5,
        midpoint[1] + unit[1] * chord * 0.5,
    )
    return _world_point(start_local, obj), _world_point(end_local, obj)


def _road_axis(obj, object_index: int, spec) -> _RoadAxis | None:
    filename = _filename(obj.model_path)
    stock_model = True

    match = _STOCK_STRAIGHT.fullmatch(filename)
    if match is not None:
        family = match.group("family").casefold()
        nominal = int(match.group("nominal"))
        expected_length = (
            float(spec.road_segment_length) * nominal / 25.0
        )
        half_width = float(_HALF_WIDTH_METRES[family])
        start, end = _p._model_axis(obj, expected_length)
    else:
        match = _STOCK_CURVE.fullmatch(filename)
        if match is not None:
            family = match.group("family").casefold()
            half_width = float(_HALF_WIDTH_METRES[family])
            start, end = _stock_curve_axis(
                obj,
                family,
                float(match.group("radius")),
            )
        else:
            match = _GENERATED_PAVED.fullmatch(filename)
            if match is not None:
                family = "paved"
                half_width = int(match.group("width")) / 20.0
                expected_length = int(match.group("length")) / 10.0
                start, end = _p._model_axis(obj, expected_length)
                stock_model = False
            else:
                match = _GRAVEL_STRAIGHT.fullmatch(filename)
                if match is None:
                    return None
                family = "gravel"
                nominal = int(match.group("nominal"))
                expected_length = (
                    float(spec.road_segment_length) * nominal / 25.0
                )
                half_width = float(_HALF_WIDTH_METRES[family])
                start, end = _p._model_axis(obj, expected_length)
                stock_model = False
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
        half_width=half_width,
        elevation=float(obj.y),
        stock_model=stock_model,
    )


def _is_paved_family(family: str) -> bool:
    return family in {"sil", "kos", "asf", "paved"}


def _surface_priority(family: str) -> int:
    # Final WorldObjects do not retain OSM highway class provenance.  Preserve
    # the strongest information still available: paved beats generated gravel,
    # which beats the stock dirt/earth family.
    if _is_paved_family(family):
        return 3
    if family == "gravel":
        return 2
    return 1


def _priority(axis: _RoadAxis) -> tuple[int, float, float, int, int]:
    # Higher surface/width/length wins.  Lower object id is the deterministic
    # tie-break, hence the negation while sorting in reverse.
    return (
        _surface_priority(axis.family),
        axis.half_width,
        axis.length,
        1 if axis.stock_model else 0,
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


def _maximum_lateral_offset(reference: _RoadAxis, other: _RoadAxis) -> float:
    # The paved-only second pass is intentionally stricter laterally than the
    # ordinary overlap rule: every endpoint of the candidate axis must stay
    # inside the same narrow centre-line envelope of the kept piece.
    nx, nz = -reference.uz, reference.ux
    return max(
        abs(
            (point[0] - reference.start[0]) * nx
            + (point[1] - reference.start[1]) * nz
        )
        for point in (other.start, other.end)
    )


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

    lateral_limit = min(2.0, min(candidate.half_width, kept.half_width) * 0.55)
    if _axis_angle_is_close(candidate, kept):
        # Use the smaller of the two line-reference measurements so small heading
        # noise does not turn a coincident 25 m slab into a false negative.
        lateral = min(
            _mean_lateral_offset(candidate, kept),
            _mean_lateral_offset(kept, candidate),
        )
        if lateral <= lateral_limit:
            overlap = max(
                _longitudinal_overlap(candidate, kept),
                _longitudinal_overlap(kept, candidate),
            )
            shorter = min(candidate.length, kept.length)
            if (
                shorter > 1.0e-6
                and overlap / shorter >= _MINIMUM_SHORTER_AXIS_OVERLAP
            ):
                return True

    # The PBO audit found a second, distinct paved failure mode: short pieces can
    # be almost completely painted under a stronger paved piece while differing
    # by slightly more than the conservative 6-degree alignment gate.  Do not
    # widen the ordinary rule.  Instead remove only the lower-priority candidate
    # when *its own* axis is nearly fully covered and remains tightly inside the
    # kept centre-line envelope.  This leaves partial seams, divided roads, dirt,
    # and longer through-pieces outside the aggressive second pass.
    if not (
        _is_paved_family(candidate.family)
        and _is_paved_family(kept.family)
    ):
        return False
    alignment = abs(candidate.ux * kept.ux + candidate.uz * kept.uz)
    if alignment < _PAVED_COVERAGE_ALIGNMENT_COSINE:
        return False
    if _maximum_lateral_offset(kept, candidate) > lateral_limit:
        return False
    candidate_overlap = _longitudinal_overlap(candidate, kept)
    return (
        candidate.length > 1.0e-6
        and candidate_overlap / candidate.length
        >= _MINIMUM_PAVED_CANDIDATE_COVERAGE
    )


def deduplicate_final_road_objects(
    report,
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Return ``report`` with redundant overlapping road pieces removed."""
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
