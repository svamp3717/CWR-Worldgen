# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallelize paved-junction planning and reuse unchanged fallback solutions.

Paved junction application has two very different costs.  Building the road-axis
spatial index and applying chosen objects are comparatively cheap, while testing
stock curve/straight templates for every arm is CPU-heavy.  Fallback refits used
to repeat that expensive search for every still-active paved junction, even when
only one distant junction had fallen back to ordinary connected roads.

This policy keeps the existing indexed apply implementation authoritative while
precomputing its per-plan choices:

* candidate road endpoints are selected in the parent from the existing spatial
  buckets, so workers receive only a tiny junction-local job rather than the full
  road-object index;
* independent junction searches run in a bounded process pool, with deterministic
  parent-side cap assignment still performed by the existing apply function;
* fallback passes reuse a previous solution only when the complete ordered target
  geometry for every arm is unchanged; object ids may be rebound safely;
* plans near a newly failed junction are proactively re-planned, while the target
  fingerprint check also catches long-chain changes outside that radius;
* the already-built spatial state and precomputed choices are injected into the
  indexed apply pass so neither indexing nor template planning is repeated.

The process pool is intentionally limited to the expensive planning phase.  This
avoids the Windows spawn/pickling regression seen when tiny road-transform work
was distributed across another pool.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import multiprocessing
import os
from typing import Any, Mapping, Sequence

from . import paved_junction_policy as _paved
from . import paved_junction_performance_policy as _perf
from . import road_audit_performance_policy as _audit_progress

_INSTALLED = False
_BASE_APPLY: Any = None
_BASE_PLAN_APPLICATION: Any = None
_BASE_BUILD_STATE: Any = None

Key = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _PlanningJob:
    order: int
    key: Key
    plan: Any
    targets_by_arm: tuple[tuple[Any, ...], ...]
    tolerance: float


@dataclass(frozen=True, slots=True)
class _PlanSnapshot:
    target_fingerprint: tuple[tuple[tuple[float, float, float, float], ...], ...]
    choices: Any


@dataclass(frozen=True, slots=True)
class _PlanningSnapshot:
    plans: Mapping[Key, _PlanSnapshot]


@dataclass(slots=True)
class _PlanningSession:
    label: str = "Planning paved-junction approaches"
    reuse: _PlanningSnapshot | None = None
    replan_keys: frozenset[Key] = frozenset()
    latest: _PlanningSnapshot | None = None


_SESSION: ContextVar[_PlanningSession | None] = ContextVar(
    "cwr_paved_junction_planning_session", default=None
)
_ACTIVE_STATE: ContextVar[tuple[Any, Any] | None] = ContextVar(
    "cwr_paved_junction_prebuilt_state", default=None
)
# The indexed apply callback receives only the plan object, not the dictionary key
# used by _plans().  Key precomputed choices by the exact live plan object identity
# rather than trying to reconstruct that key from coordinates through a helper on
# the wrong module.
_ACTIVE_CHOICES: ContextVar[dict[int, Any] | None] = ContextVar(
    "cwr_paved_junction_precomputed_choices", default=None
)


def begin_planning_session(
    label: str = "Planning paved-junction approaches, initial pass",
):
    session = _PlanningSession(label=label)
    return session, _SESSION.set(session)


def end_planning_session(token) -> None:
    _SESSION.reset(token)


def configure_planning_session(
    session: _PlanningSession,
    *,
    reuse: _PlanningSnapshot | None,
    replan_keys: Sequence[Key] = (),
    label: str,
) -> None:
    session.reuse = reuse
    session.replan_keys = frozenset(replan_keys)
    session.label = str(label)
    session.latest = None


def _workers(job_count: int) -> int:
    if job_count < 32 or multiprocessing.current_process().daemon:
        return 1
    raw = os.environ.get("CWR_WORLDGEN_JUNCTION_WORKERS", "").strip()
    if not raw:
        raw = os.environ.get("CWR_WORLDGEN_ROAD_WORKERS", "").strip()
    if not raw:
        raw = os.environ.get("CWR_WORLDGEN_OBJECT_WORKERS", "").strip()
    if raw:
        try:
            requested = max(1, int(raw))
        except ValueError:
            requested = 1
    else:
        cpu = os.cpu_count() or 1
        requested = max(1, min(8, cpu - 1 if cpu > 2 else cpu))
    return min(requested, max(1, job_count // 8))


def _tolerance(spec) -> float:
    return max(
        0.20,
        min(
            0.40,
            float(getattr(spec, "road_connection_tolerance", 0.35)),
        ),
    )


def _target_signature(target) -> tuple[float, float, float, float]:
    return (
        float(target.point[0]),
        float(target.point[1]),
        float(target.continuation[0]),
        float(target.continuation[1]),
    )


def _fingerprint(targets_by_arm) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
    return tuple(
        tuple(_target_signature(target) for target in targets)
        for targets in targets_by_arm
    )


def _arm_options_from_targets(plan, arm, targets, tolerance: float):
    result = []
    for target in targets:
        match = _perf._approach_choice_to_target(
            plan, arm, target, tolerance
        )
        if match is not None:
            score, choice = match
            result.append((score, target, choice))
    result.sort(key=lambda item: (item[0], item[1].object_id))
    return tuple(result[:8])


def _best_combination(options):
    """Exact branch-and-bound equivalent of ``product(*options)``.

    Options are already sorted by score/object id.  Depth-first traversal keeps
    the historical lexicographic first-best tie behaviour, while the suffix lower
    bound avoids enumerating combinations that cannot beat the best score found.
    """
    if any(not values for values in options):
        return None
    count = len(options)
    suffix_minimum = [0.0] * (count + 1)
    for index in range(count - 1, -1, -1):
        suffix_minimum[index] = suffix_minimum[index + 1] + float(options[index][0][0])

    best_score = math.inf
    best = None
    chosen: list[Any] = []
    used_ids: set[int] = set()

    def visit(index: int, score: float) -> None:
        nonlocal best_score, best
        if index == count:
            if score < best_score:
                best_score = score
                best = tuple(chosen)
            return
        if score + suffix_minimum[index] >= best_score:
            return
        for value in options[index]:
            object_id = int(value[1].object_id)
            if object_id in used_ids:
                continue
            next_score = score + float(value[0])
            if next_score + suffix_minimum[index + 1] >= best_score:
                continue
            used_ids.add(object_id)
            chosen.append(value)
            visit(index + 1, next_score)
            chosen.pop()
            used_ids.remove(object_id)

    visit(0, 0.0)
    return best


def _solve_job(job: _PlanningJob):
    options = tuple(
        _arm_options_from_targets(job.plan, arm, targets, job.tolerance)
        for arm, targets in zip(job.plan.arms, job.targets_by_arm)
    )
    return job.order, job.key, _best_combination(options)


def _solve_batch(batch: tuple[_PlanningJob, ...]):
    return tuple(_solve_job(job) for job in batch)


def _worker_init() -> None:
    # Build the two immutable template indexes once in each spawned worker rather
    # than making its first real junction pay the complete catalogue setup cost.
    _perf._path_template_index(1)
    _perf._path_template_index(-1)


def _batches(jobs: Sequence[_PlanningJob], workers: int):
    if not jobs:
        return ()
    size = max(4, min(32, math.ceil(len(jobs) / max(1, workers * 10))))
    return tuple(
        tuple(jobs[offset : offset + size])
        for offset in range(0, len(jobs), size)
    )


def _rebind_choices(targets_by_arm, cached_choices):
    if cached_choices is None:
        return None
    if len(cached_choices) != len(targets_by_arm):
        return None
    rebound = []
    for targets, cached in zip(targets_by_arm, cached_choices):
        score, old_target, choice = cached
        signature = _target_signature(old_target)
        matches = [
            target for target in targets
            if _target_signature(target) == signature
        ]
        # Ambiguous duplicate geometry can make object-id tie breaking relevant.
        # Re-plan rather than guessing which duplicate the historical search wins.
        if len(matches) != 1:
            return None
        rebound.append((score, matches[0], choice))
    return tuple(rebound)


def _emit_progress(
    callback,
    label: str,
    completed: int,
    total: int,
    *,
    applicable: int,
    reused: int,
    workers: int,
    force: bool = False,
    previous_bucket: list[int] | None = None,
) -> None:
    if callback is None:
        return
    percent = 100 if total <= 0 else min(100, max(0, int(completed * 100 / total)))
    bucket = percent // 2
    if previous_bucket is not None:
        if (
            not force
            and completed not in {0, 1, total}
            and bucket <= previous_bucket[0]
        ):
            return
        previous_bucket[0] = bucket
    detail = f"{applicable:,} applicable"
    if reused:
        detail += f"; {reused:,} cached"
    if workers > 1:
        detail += f"; {workers} workers"
    callback(
        _perf._RAW_PERCENT,
        f"{label} ({completed:,}/{total:,}, {percent}%; {detail})",
    )


def _cached_build_spatial_state(report, spec, reporter):
    active = _ACTIVE_STATE.get()
    if active is not None and active[0] is report:
        return active[1]
    return _BASE_BUILD_STATE(report, spec, reporter)


def _cached_plan_application(state, plan, spec):
    choices = _ACTIVE_CHOICES.get()
    if choices is not None:
        plan_identity = id(plan)
        if plan_identity in choices:
            return choices[plan_identity]
    return _BASE_PLAN_APPLICATION(state, plan, spec)


def _filtered_callback(callback):
    if callback is None:
        return None

    def emit(percent, message):
        if str(message).startswith("Planning paved-junction approaches"):
            return
        callback(percent, message)

    return emit


def apply_paved_junctions_parallel(report, plans, elevations, spec):
    """Pre-plan junction choices in parallel, then delegate exact application."""
    if not plans or report.junction_cap_objects <= 0:
        return _BASE_APPLY(report, plans, elevations, spec)

    callback = _audit_progress._PROGRESS_CALLBACK.get()
    reporter = _perf._Reporter(callback)
    state = _BASE_BUILD_STATE(report, spec, reporter)
    ordered = tuple(sorted(plans))
    session = _SESSION.get()
    reuse = session.reuse if session is not None else None
    explicit_replan = session.replan_keys if session is not None else frozenset()
    label = (
        session.label
        if session is not None
        else "Planning paved-junction approaches"
    )

    fingerprints: dict[Key, tuple[tuple[tuple[float, float, float, float], ...], ...]] = {}
    choices: dict[Key, Any] = {}
    jobs: list[_PlanningJob] = []
    reused = 0
    tolerance = _tolerance(spec)

    for order, key in enumerate(ordered):
        plan = plans[key]
        targets_by_arm = tuple(
            _perf._target_candidates(state, plan, arm)
            for arm in plan.arms
        )
        fingerprint = _fingerprint(targets_by_arm)
        fingerprints[key] = fingerprint

        previous = reuse.plans.get(key) if reuse is not None else None
        if (
            previous is not None
            and key not in explicit_replan
            and previous.target_fingerprint == fingerprint
        ):
            if previous.choices is None:
                choices[key] = None
                reused += 1
                continue
            rebound = _rebind_choices(targets_by_arm, previous.choices)
            if rebound is not None:
                choices[key] = rebound
                reused += 1
                continue

        if any(not targets for targets in targets_by_arm):
            choices[key] = None
            continue
        jobs.append(
            _PlanningJob(
                order,
                key,
                plan,
                targets_by_arm,
                tolerance,
            )
        )

    workers = _workers(len(jobs))
    # In fallback passes the progress denominator is only the plans that actually
    # need solving. Initial planning still reports the full junction count.
    planning_total = len(ordered) if reuse is None else len(ordered) - reused
    immediate = planning_total - len(jobs)
    applicable = sum(value is not None for value in choices.values())
    progress_bucket = [-1]
    _emit_progress(
        callback,
        label,
        immediate,
        planning_total,
        applicable=applicable,
        reused=reused,
        workers=workers,
        force=True,
        previous_bucket=progress_bucket,
    )

    def store(values) -> None:
        nonlocal applicable
        for _order, key, value in values:
            choices[key] = value
            if value is not None:
                applicable += 1

    completed = immediate
    if workers <= 1:
        for job in jobs:
            store((_solve_job(job),))
            completed += 1
            _emit_progress(
                callback,
                label,
                completed,
                planning_total,
                applicable=applicable,
                reused=reused,
                workers=workers,
                previous_bucket=progress_bucket,
            )
    else:
        try:
            batches = _batches(jobs, workers)
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
            ) as executor:
                futures = {
                    executor.submit(_solve_batch, batch): len(batch)
                    for batch in batches
                }
                for future in as_completed(futures):
                    values = future.result()
                    store(values)
                    completed += len(values)
                    _emit_progress(
                        callback,
                        label,
                        completed,
                        planning_total,
                        applicable=applicable,
                        reused=reused,
                        workers=workers,
                        previous_bucket=progress_bucket,
                    )
        except (OSError, RuntimeError, BrokenPipeError, EOFError, TypeError):
            # Frozen/embedded Python can reject a new pool.  Finish only the keys
            # not already returned, serially, preserving deterministic results.
            pending = {job.key: job for job in jobs if job.key not in choices}
            for job in jobs:
                if job.key not in pending:
                    continue
                store((_solve_job(job),))
                completed += 1
                _emit_progress(
                    callback,
                    label,
                    completed,
                    planning_total,
                    applicable=applicable,
                    reused=reused,
                    workers=1,
                    previous_bucket=progress_bucket,
                )

    # The completion event is normally emitted by the last serial job/future.  Do
    # not print a second 100% line after ProcessPoolExecutor waits for worker
    # shutdown, which previously made the phase look like it had run twice.
    if progress_bucket[0] < 50:
        _emit_progress(
            callback,
            label,
            planning_total,
            planning_total,
            applicable=applicable,
            reused=reused,
            workers=workers,
            force=True,
            previous_bucket=progress_bucket,
        )

    snapshot = _PlanningSnapshot(
        {
            key: _PlanSnapshot(fingerprints[key], choices.get(key))
            for key in ordered
        }
    )
    if session is not None:
        session.latest = snapshot

    # _BASE_APPLY iterates the exact plan instances supplied in this mapping.
    # Identity is therefore a collision-free lookup key and avoids duplicating the
    # road-node quantisation implementation in this performance layer.
    active_choices = {
        id(plans[key]): choices.get(key)
        for key in ordered
    }
    state_token = _ACTIVE_STATE.set((report, state))
    choices_token = _ACTIVE_CHOICES.set(active_choices)
    callback_token = None
    filtered = _filtered_callback(callback)
    if filtered is not None:
        callback_token = _audit_progress._PROGRESS_CALLBACK.set(filtered)
    try:
        return _BASE_APPLY(report, plans, elevations, spec)
    finally:
        if callback_token is not None:
            _audit_progress._PROGRESS_CALLBACK.reset(callback_token)
        _ACTIVE_CHOICES.reset(choices_token)
        _ACTIVE_STATE.reset(state_token)


def install_paved_junction_parallel_planning_policy() -> None:
    """Layer parallel/cached planning outside the indexed paved apply policy."""
    global _INSTALLED, _BASE_APPLY, _BASE_PLAN_APPLICATION, _BASE_BUILD_STATE
    if _INSTALLED:
        return

    _perf.install_paved_junction_performance_policy()
    _BASE_APPLY = _paved._apply_plans
    _BASE_PLAN_APPLICATION = _perf._plan_application
    _BASE_BUILD_STATE = _perf._build_spatial_state

    _perf._plan_application = _cached_plan_application
    _perf._build_spatial_state = _cached_build_spatial_state
    # Keep the historical public performance symbol synchronized so existing
    # wrapper-order tests and any callers resolving it by name see the live target.
    _perf.apply_paved_junctions_fast = apply_paved_junctions_parallel
    _paved._apply_plans = apply_paved_junctions_parallel
    _INSTALLED = True
