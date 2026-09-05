# SPDX-License-Identifier: GPL-3.0-or-later
"""Speed up the post-fit road-junction audit and expose live progress."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
from typing import Callable

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _quality

_AUDIT_RAW_PERCENT = 99
_PROGRESS_BUCKET_PERCENT = 2
_PROGRESS_CALLBACK: ContextVar[Callable[[int, str], None] | None] = ContextVar(
    "cwr_road_audit_progress", default=None
)
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _AuditAxis:
    ux: float
    uz: float
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(slots=True)
class _ArmAudit:
    dx: float
    dz: float
    cover: float
    bx0: int
    bx1: int
    bz0: int
    bz1: int
    best: float = math.inf

    @property
    def covered(self) -> bool:
        return self.best <= self.cover


class _AuditReporter:
    def __init__(
        self,
        callback: Callable[[int, str], None] | None,
        junction_total: int,
    ) -> None:
        self.callback = callback
        self.junction_total = max(0, int(junction_total))
        self._last_index_bucket = -1
        self._last_junction_bucket = -1

    @staticmethod
    def _percent(completed: int, total: int) -> int:
        if total <= 0:
            return 100
        return min(100, max(0, int(completed * 100 / total)))

    def index(self, completed: int, total: int, *, force: bool = False) -> None:
        if self.callback is None:
            return
        percent = self._percent(completed, total)
        bucket = percent // _PROGRESS_BUCKET_PERCENT
        if (
            not force
            and completed not in {1, total}
            and bucket <= self._last_index_bucket
        ):
            return
        self._last_index_bucket = bucket
        self.callback(
            _AUDIT_RAW_PERCENT,
            f"Auditing {self.junction_total:,} road junctions: indexing road pieces "
            f"({completed:,}/{total:,}, {percent}%)",
        )

    def junction(
        self,
        completed: int,
        *,
        failed: int,
        indexed_axes: int,
        force: bool = False,
    ) -> None:
        if self.callback is None:
            return
        percent = self._percent(completed, self.junction_total)
        bucket = percent // _PROGRESS_BUCKET_PERCENT
        if (
            not force
            and completed not in {1, self.junction_total}
            and bucket <= self._last_junction_bucket
        ):
            return
        self._last_junction_bucket = bucket
        self.callback(
            _AUDIT_RAW_PERCENT,
            f"Auditing road junctions "
            f"({completed:,}/{self.junction_total:,}, {percent}%; "
            f"{failed:,} failed; {indexed_axes:,} nearby road axes)",
        )


def _audit_axis_distance(
    point_x: float,
    point_z: float,
    axis: _AuditAxis,
) -> float:
    """Distance from a junction point to one already-materialized road axis."""
    return _p._point_segment_distance(
        (point_x, point_z),
        axis.start,
        axis.end,
    )


def _potential_audit_buckets(
    context,
    *,
    bucket_size: float,
    maximum_half_length: float,
) -> set[tuple[int, int]]:
    """Return every midpoint bucket that could matter to any junction.

    ``_piece_length`` never exceeds ``spec.road_segment_length``. Using that
    configured maximum lets the index skip axis/trigonometry work for road pieces
    that are nowhere near an audited junction, while later per-arm bounds still
    use the exact maximum length observed in the report.
    """
    tolerance = float(context.spec.road_connection_tolerance)
    result: set[tuple[int, int]] = set()
    for junction in context.junctions.values():
        maximum_cover = max(
            (
                _quality._exit_distance(junction, direction)
                + _quality._JUNCTION_MARGIN
                for direction in junction.directions
            ),
            default=0.0,
        )
        radius = maximum_cover + tolerance + maximum_half_length + 0.05
        point_x, point_z = junction.point
        bx0 = math.floor((point_x - radius) / bucket_size)
        bx1 = math.floor((point_x + radius) / bucket_size)
        bz0 = math.floor((point_z - radius) / bucket_size)
        bz1 = math.floor((point_z + radius) / bucket_size)
        for bx in range(bx0, bx1 + 1):
            for bz in range(bz0, bz1 + 1):
                result.add((bx, bz))
    return result


def audit_road_junctions(
    report,
    context,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Audit junction coverage with one nearby-axis distance pass per junction.

    The previous spatial audit was already much better than scanning the whole
    road network for every arm, but a T/X junction still walked the same nearby
    bucket set three or four times and recomputed the same point-to-axis distance
    on every arm. Here the nearby buckets are gathered once per junction and each
    candidate axis distance is evaluated at most once, then shared by every
    direction whose alignment/bounds admit it.
    """
    if not context.junctions or not report.objects:
        return report

    spec = context.spec
    bucket_size = float(_quality._AUDIT_BUCKET_METRES)
    tolerance = float(spec.road_connection_tolerance)
    junction_total = len(context.junctions)
    reporter = _AuditReporter(progress_callback, junction_total)

    objects = report.objects[report.junction_cap_objects :]
    object_total = len(objects)
    configured_half_length = max(0.0, float(spec.road_segment_length)) * 0.5
    potential_buckets = _potential_audit_buckets(
        context,
        bucket_size=bucket_size,
        maximum_half_length=configured_half_length,
    )

    length_cache: dict[str, float] = {}
    buckets: dict[tuple[int, int], list[_AuditAxis]] = {}
    maximum_half_length = 0.0
    indexed_axes = 0

    reporter.index(0, object_total, force=True)
    for completed, obj in enumerate(objects, start=1):
        model_key = obj.model_path.casefold()
        length = length_cache.get(model_key)
        if length is None:
            length = _quality._piece_length(
                obj.model_path, float(spec.road_segment_length)
            )
            length_cache[model_key] = length
        half_length = max(0.0, float(length)) * 0.5
        maximum_half_length = max(maximum_half_length, half_length)

        key = (
            math.floor(float(obj.x) / bucket_size),
            math.floor(float(obj.z) / bucket_size),
        )
        if half_length > 1.0e-9 and key in potential_buckets:
            axis = _p._model_axis(obj, length)
            dx = axis[1][0] - axis[0][0]
            dz = axis[1][1] - axis[0][1]
            axis_length = math.hypot(dx, dz)
            if axis_length > 1.0e-9:
                buckets.setdefault(key, []).append(
                    _AuditAxis(
                        dx / axis_length,
                        dz / axis_length,
                        axis[0],
                        axis[1],
                    )
                )
                indexed_axes += 1
        reporter.index(completed, object_total)

    reporter.index(object_total, object_total, force=True)

    failed = 0
    maximum_gap = report.maximum_connection_gap
    maximum_cover = 0.0
    reporter.junction(0, failed=0, indexed_axes=indexed_axes, force=True)

    for completed, junction in enumerate(context.junctions.values(), start=1):
        point_x, point_z = junction.point
        arms: list[_ArmAudit] = []
        for direction in junction.directions:
            cover = _quality._exit_distance(junction, direction) + _quality._JUNCTION_MARGIN
            maximum_cover = max(maximum_cover, cover)
            radius = cover + tolerance + maximum_half_length + 0.05
            arms.append(
                _ArmAudit(
                    float(direction[0]),
                    float(direction[1]),
                    cover,
                    math.floor((point_x - radius) / bucket_size),
                    math.floor((point_x + radius) / bucket_size),
                    math.floor((point_z - radius) / bucket_size),
                    math.floor((point_z + radius) / bucket_size),
                )
            )

        if arms:
            bx0 = min(arm.bx0 for arm in arms)
            bx1 = max(arm.bx1 for arm in arms)
            bz0 = min(arm.bz0 for arm in arms)
            bz1 = max(arm.bz1 for arm in arms)

            all_covered = False
            for bx in range(bx0, bx1 + 1):
                if all_covered:
                    break
                for bz in range(bz0, bz1 + 1):
                    if all_covered:
                        break
                    candidates = buckets.get((bx, bz), ())
                    if not candidates:
                        continue
                    for axis in candidates:
                        distance: float | None = None
                        for arm in arms:
                            if (
                                arm.covered
                                or not (arm.bx0 <= bx <= arm.bx1)
                                or not (arm.bz0 <= bz <= arm.bz1)
                                or abs(axis.ux * arm.dx + axis.uz * arm.dz)
                                < _quality._AUDIT_ALIGNMENT_COSINE
                            ):
                                continue
                            if distance is None:
                                distance = _audit_axis_distance(
                                    point_x, point_z, axis
                                )
                            if distance < arm.best:
                                arm.best = distance
                        if distance is not None and all(
                            arm.covered for arm in arms
                        ):
                            all_covered = True
                            break

        for arm in arms:
            if not math.isfinite(arm.best):
                failed += 1
                maximum_gap = max(maximum_gap, tolerance + 1.0e-6)
                continue
            uncovered = max(0.0, arm.best - arm.cover)
            maximum_gap = max(maximum_gap, uncovered)
            if uncovered > tolerance:
                failed += 1

        reporter.junction(
            completed,
            failed=failed,
            indexed_axes=indexed_axes,
        )

    reporter.junction(
        junction_total,
        failed=failed,
        indexed_axes=indexed_axes,
        force=True,
    )
    return replace(
        report,
        failed_connections=failed,
        maximum_connection_gap=maximum_gap,
        maximum_junction_clearance_metres=max(
            report.maximum_junction_clearance_metres,
            maximum_cover,
        ),
    )


def _installed_audit(report, context):
    return audit_road_junctions(
        report,
        context,
        progress_callback=_PROGRESS_CALLBACK.get(),
    )


def install_road_audit_performance_policy() -> None:
    """Install the optimized audit after the base road-quality policy."""
    global _INSTALLED
    if _INSTALLED:
        return

    # This policy installs after the paved/gravel/raceway wrappers. Preserve that
    # complete final chain and only add a ContextVar around it so the inner
    # road-quality audit can see the existing progress callback.
    original_fit = _p.fit_road_objects

    def progressive_fit(
        dataset,
        projection,
        elevations,
        spec,
        *,
        starting_id: int = 1,
        progress_callback=None,
    ):
        token = _PROGRESS_CALLBACK.set(progress_callback)
        try:
            return original_fit(
                dataset,
                projection,
                elevations,
                spec,
                starting_id=starting_id,
                progress_callback=progress_callback,
            )
        finally:
            _PROGRESS_CALLBACK.reset(token)

    # road_quality_policy._fit resolves its module-global ``_audit`` at call time.
    # Replace only that audit target; do not bypass any later road fit wrappers.
    _quality._audit = _installed_audit
    _p.fit_road_objects = progressive_fit
    _generator.fit_road_objects = progressive_fit
    _INSTALLED = True
