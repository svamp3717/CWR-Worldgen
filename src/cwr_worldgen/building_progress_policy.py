# SPDX-License-Identifier: GPL-3.0-or-later
"""Stream progress from expensive procedural building asset workers.

The generic asset pool historically returned one whole worker chunk at a time.
For modeler-backed buildings that can mean minutes with no visible movement even
though child processes are steadily completing textures or P3Ds. This policy
keeps the existing spawned-process lifecycle and deterministic ordering, but
lets the two expensive building phases send one result at a time so the parent
can emit useful counters.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection, wait as wait_connections
import os
import traceback
from typing import Callable, Iterable, TypeVar, cast

from . import parallel_assets as _parallel

T = TypeVar("T")
R = TypeVar("R")

_TEXTURE_WORKER = (
    "cwr_worldgen.building_asset_budget_policy",
    "_write_modeler_texture_cache_task",
)
_BUILDING_WORKER = (
    "cwr_worldgen.procedural_buildings",
    "_write_building_asset_task",
)
_INSTALLED = False


class _ProgressAssetWorkerFailure(RuntimeError):
    """Marker used to fall back to deterministic serial generation."""


def _worker_identity(worker: Callable[..., object]) -> tuple[str, str]:
    return (
        str(getattr(worker, "__module__", "")),
        str(getattr(worker, "__name__", "")),
    )


def _progress_worker_chunk(
    worker: Callable[[T], R],
    indexed_tasks: list[tuple[int, T]],
    sender: Connection,
) -> None:
    """Run one worker chunk while returning each completed asset immediately."""
    try:
        for index, task in indexed_tasks:
            sender.send(("result", index, worker(task)))
        sender.send(("done",))
    except BaseException as exc:  # pragma: no cover - parent reruns serially
        try:
            sender.send((
                "error",
                f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}",
                traceback.format_exc(),
            ))
        except BaseException:
            pass
    finally:
        try:
            sender.close()
        except OSError:
            pass


def _process_map_with_progress(
    worker: Callable[[T], R],
    tasks: list[T],
    workers: int,
    progress_callback: Callable[[int, int], None] | None,
) -> list[R]:
    """Parallel map that streams task completion back to the parent process."""
    if workers <= 1 or len(tasks) < 2:
        results: list[R] = []
        for completed, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if progress_callback is not None:
                progress_callback(completed, len(tasks))
        return results

    worker_count = min(max(1, int(workers)), len(tasks))
    indexed_tasks = list(enumerate(tasks))
    chunks: list[list[tuple[int, T]]] = [
        indexed_tasks[worker_index::worker_count]
        for worker_index in range(worker_count)
    ]

    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    receivers: dict[Connection, mp.Process] = {}
    output: list[R | None] = [None] * len(tasks)
    completed = 0

    try:
        for worker_index, chunk in enumerate(chunks):
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_progress_worker_chunk,
                args=(worker, chunk, sender),
                name=f"cwr-asset-progress-{worker_index + 1}",
            )
            process.daemon = True
            try:
                process.start()
            except BaseException:
                receiver.close()
                sender.close()
                raise
            sender.close()
            processes.append(process)
            receivers[receiver] = process

        pending = set(receivers)
        while pending:
            ready = wait_connections(
                tuple(pending), timeout=_parallel._WORKER_POLL_SECONDS
            )
            if not ready:
                for receiver in tuple(pending):
                    process = receivers[receiver]
                    if process.exitcode is not None and not receiver.poll():
                        raise _ProgressAssetWorkerFailure(
                            f"asset worker {process.name} exited with code {process.exitcode}"
                        )
                continue

            for receiver in ready:
                process = receivers[receiver]
                try:
                    message = receiver.recv()
                except (EOFError, OSError) as exc:
                    receiver.close()
                    pending.discard(receiver)
                    raise _ProgressAssetWorkerFailure(
                        f"asset worker {process.name} closed its result pipe"
                    ) from exc

                if not isinstance(message, tuple) or not message:
                    raise _ProgressAssetWorkerFailure(
                        f"asset worker {process.name} returned an invalid result"
                    )
                kind = message[0]
                if kind == "result":
                    index = int(message[1])
                    output[index] = cast(R, message[2])
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, len(tasks))
                    continue
                if kind == "done":
                    receiver.close()
                    pending.discard(receiver)
                    continue
                if kind == "error":
                    detail = str(message[1]) if len(message) > 1 else "unknown worker error"
                    trace = str(message[2]) if len(message) > 2 else ""
                    raise _ProgressAssetWorkerFailure(
                        f"asset worker {process.name} failed: {detail}\n{trace}"
                    )
                raise _ProgressAssetWorkerFailure(
                    f"asset worker {process.name} returned unknown message {kind!r}"
                )

        _parallel._finish_processes(
            processes, grace_seconds=_parallel._WORKER_EXIT_GRACE_SECONDS
        )
    except BaseException:
        for receiver in list(receivers):
            try:
                receiver.close()
            except OSError:
                pass
        _parallel._abort_processes(processes)
        raise

    if any(result is None for result in output):
        raise _ProgressAssetWorkerFailure(
            "one or more asset workers returned no result"
        )
    return [cast(R, result) for result in output]


def process_asset_tasks_with_progress(
    worker: Callable[[T], R],
    tasks: Iterable[T],
    *,
    minimum_parallel_tasks: int = _parallel._DEFAULT_PARALLEL_MINIMUM,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Run asset tasks and report completed/total after every finished task."""
    task_list = list(tasks)
    if not task_list:
        return []

    results: list[R | None] = [None] * len(task_list)
    miss_indices: list[int] = []
    completed_hits = 0
    for index, task in enumerate(task_list):
        if _parallel._cache_hit_without_refresh(task):
            results[index] = worker(task)
            completed_hits += 1
            if progress_callback is not None:
                progress_callback(completed_hits, len(task_list))
        else:
            miss_indices.append(index)

    workers = _parallel.asset_worker_count(len(miss_indices))
    explicit_workers = bool(
        os.environ.get("CWR_WORLDGEN_ASSET_WORKERS", "").strip()
    )
    threshold = (
        2
        if explicit_workers
        else _parallel._parallel_minimum_for_worker(worker, minimum_parallel_tasks)
    )
    miss_tasks = [task_list[index] for index in miss_indices]

    def miss_progress(done: int, _total: int) -> None:
        if progress_callback is not None:
            progress_callback(completed_hits + done, len(task_list))

    if workers > 1 and len(miss_tasks) >= threshold:
        try:
            miss_results = _process_map_with_progress(
                worker, miss_tasks, workers, miss_progress
            )
        except (_ProgressAssetWorkerFailure, OSError, RuntimeError):
            # Match the mature generic pool's safety behavior. Any already-written
            # deterministic cache files are harmlessly reused or replaced here.
            miss_results = []
            for done, task in enumerate(miss_tasks, start=1):
                miss_results.append(worker(task))
                miss_progress(done, len(miss_tasks))
    else:
        miss_results = []
        for done, task in enumerate(miss_tasks, start=1):
            miss_results.append(worker(task))
            miss_progress(done, len(miss_tasks))

    for index, result in zip(miss_indices, miss_results):
        results[index] = result
    return [cast(R, result) for result in results]


class _BuildingPhaseReporter:
    """Throttle task counters to roughly two-percent status increments."""

    def __init__(self, noun: str, raw_percent: int) -> None:
        self.noun = noun
        self.raw_percent = raw_percent
        self._last_bucket = -1

    def __call__(self, completed: int, total: int) -> None:
        if total <= 0:
            return
        percent = min(100, max(0, int(completed * 100 / total)))
        bucket = percent // 2
        if completed not in {1, total} and bucket <= self._last_bucket:
            return
        self._last_bucket = bucket
        from .progress import report_progress

        report_progress(
            self.raw_percent,
            f"Generating procedural building {self.noun} "
            f"({completed}/{total}, {percent}%)",
        )


def install_building_progress_policy() -> None:
    """Route only modeler texture/P3D batches through streaming progress."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import procedural_buildings as buildings

    original_process_asset_tasks = _parallel.process_asset_tasks

    def progressive_process_asset_tasks(
        worker,
        tasks,
        *,
        minimum_parallel_tasks: int = _parallel._DEFAULT_PARALLEL_MINIMUM,
    ):
        identity = _worker_identity(worker)
        if identity == _TEXTURE_WORKER:
            return process_asset_tasks_with_progress(
                worker,
                tasks,
                minimum_parallel_tasks=minimum_parallel_tasks,
                progress_callback=_BuildingPhaseReporter("textures", 73),
            )
        if identity == _BUILDING_WORKER:
            return process_asset_tasks_with_progress(
                worker,
                tasks,
                minimum_parallel_tasks=minimum_parallel_tasks,
                progress_callback=_BuildingPhaseReporter("P3Ds", 74),
            )
        return original_process_asset_tasks(
            worker,
            tasks,
            minimum_parallel_tasks=minimum_parallel_tasks,
        )

    # building_asset_budget_policy keeps the module object and therefore sees
    # this replacement dynamically. procedural_buildings imported the function
    # directly, so update that module-level reference as well.
    _parallel.process_asset_tasks = progressive_process_asset_tasks
    buildings.process_asset_tasks = progressive_process_asset_tasks
    _INSTALLED = True
