# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection, wait as wait_connections
import os
from pathlib import Path
import time
import traceback
from typing import Callable, Iterable, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")

_DEFAULT_PARALLEL_MINIMUM = 768
# Detailed modeler-backed P3Ds are substantially more expensive than the older
# compact building assets for which the 768-task cutoff was chosen. Keep the
# generic threshold conservative for forests/infrastructure/site assets, but let
# a medium fresh batch of building models use the existing bounded worker pool.
_BUILDING_PARALLEL_MINIMUM = 64
_BUILDING_ASSET_WORKER = (
    "cwr_worldgen.procedural_buildings",
    "_write_building_asset_task",
)
_WORKER_POLL_SECONDS = 0.20
_WORKER_EXIT_GRACE_SECONDS = 5.0
_WORKER_TERMINATE_GRACE_SECONDS = 2.0


class _ParallelAssetWorkerFailure(RuntimeError):
    """Internal marker used to fall back to deterministic serial generation."""


def asset_worker_count(task_count: int) -> int:
    """Return a conservative worker count for independent procedural assets."""
    if task_count < 2:
        return 1
    override = os.environ.get("CWR_WORLDGEN_ASSET_WORKERS", "").strip()
    if override:
        try:
            requested = max(1, int(override))
        except ValueError:
            requested = 1
        return min(task_count, 16, requested)

    # Spawned Python workers have noticeable startup/import cost because they
    # load Pillow, GEOS/Shapely and the P3D writer. Four workers is a better
    # default than blindly matching a 16/32-core workstation; users can override
    # it through CWR_WORLDGEN_ASSET_WORKERS when model generation is genuinely
    # heavy enough to benefit from a wider pool.
    requested = max(1, min(4, (os.cpu_count() or 2) - 1))
    return min(task_count, requested)


def _parallel_minimum_for_worker(worker: Callable[..., object], requested: int) -> int:
    """Return a worker-specific threshold without changing other asset stages."""
    threshold = max(2, int(requested))
    identity = (
        str(getattr(worker, "__module__", "")),
        str(getattr(worker, "__name__", "")),
    )
    if identity == _BUILDING_ASSET_WORKER:
        return min(threshold, _BUILDING_PARALLEL_MINIMUM)
    return threshold


def _cache_hit_without_refresh(task: object) -> bool:
    enabled = bool(getattr(task, "cache_enabled", False))
    refresh = bool(getattr(task, "cache_refresh", False))
    path = getattr(task, "cache_path", None)
    return enabled and not refresh and isinstance(path, Path) and path.is_file()


def _asset_worker_chunk(
    worker: Callable[[T], R],
    indexed_tasks: list[tuple[int, T]],
    sender: Connection,
) -> None:
    """Run one deterministic task chunk and return it through a one-way pipe.

    This deliberately avoids multiprocessing Queue/Pool/ProcessPoolExecutor.
    Those abstractions allocate synchronization semaphores and an executor
    management thread, which can leave Python's resource_tracker waiting during
    GUI/frozen shutdown. Plain processes plus one-way pipes need neither.
    """
    try:
        payload = [(index, worker(task)) for index, task in indexed_tasks]
        sender.send(("ok", payload))
    except BaseException as exc:  # pragma: no cover - exercised through parent fallback
        # Do not try to pickle the original exception: third-party exceptions
        # are not guaranteed to be picklable. The parent reruns serially after
        # cleanup, which exposes the real exception in the normal process.
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


def _finish_processes(
    processes: Iterable[mp.Process], *, grace_seconds: float
) -> None:
    """Give all completed workers one shared exit grace, then reap them."""
    process_list = list(processes)
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    for process in process_list:
        try:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(remaining)
        except ValueError:
            continue

    # Do not let four stuck workers turn a five-second shutdown allowance into
    # twenty seconds. They share one grace window, then are terminated together.
    for process in process_list:
        try:
            if process.is_alive():
                process.terminate()
        except ValueError:
            continue
    for process in process_list:
        try:
            process.join(_WORKER_TERMINATE_GRACE_SECONDS)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(_WORKER_TERMINATE_GRACE_SECONDS)
            process.close()
        except (OSError, ValueError):
            pass


def _abort_processes(processes: Iterable[mp.Process]) -> None:
    """Best-effort synchronous cleanup for every started worker."""
    process_list = list(processes)
    for process in process_list:
        try:
            if process.is_alive():
                process.terminate()
        except ValueError:
            continue
    for process in process_list:
        try:
            process.join(_WORKER_TERMINATE_GRACE_SECONDS)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(_WORKER_TERMINATE_GRACE_SECONDS)
            process.close()
        except (OSError, ValueError):
            pass


def _process_map(worker: Callable[[T], R], tasks: list[T], workers: int) -> list[R]:
    """Map *worker* over *tasks* with explicitly managed spawned processes."""
    if workers <= 1 or len(tasks) < 2:
        return [worker(task) for task in tasks]

    worker_count = min(max(1, int(workers)), len(tasks))
    # Striding gives complex/cheap procedural variants to each worker instead
    # of accidentally putting all expensive polygon buildings in one tail chunk.
    indexed_tasks = list(enumerate(tasks))
    chunks: list[list[tuple[int, T]]] = [
        indexed_tasks[worker_index::worker_count]
        for worker_index in range(worker_count)
    ]

    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    receivers: dict[Connection, mp.Process] = {}
    output: list[R | None] = [None] * len(tasks)

    try:
        for worker_index, chunk in enumerate(chunks):
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_asset_worker_chunk,
                args=(worker, chunk, sender),
                name=f"cwr-asset-{worker_index + 1}",
            )
            # A daemon flag is only a final safety net. Normal completion always
            # performs explicit join/terminate/close below.
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
            ready = wait_connections(tuple(pending), timeout=_WORKER_POLL_SECONDS)
            if not ready:
                # A crashed worker can close its pipe before a useful payload is
                # available. Detect that instead of waiting forever for recv().
                for receiver in tuple(pending):
                    process = receivers[receiver]
                    if process.exitcode is not None and not receiver.poll():
                        raise _ParallelAssetWorkerFailure(
                            f"asset worker {process.name} exited with code {process.exitcode}"
                        )
                continue

            for receiver in ready:
                process = receivers[receiver]
                try:
                    message = receiver.recv()
                except (EOFError, OSError) as exc:
                    raise _ParallelAssetWorkerFailure(
                        f"asset worker {process.name} closed its result pipe"
                    ) from exc
                finally:
                    receiver.close()
                    pending.discard(receiver)

                if not isinstance(message, tuple) or not message:
                    raise _ParallelAssetWorkerFailure(
                        f"asset worker {process.name} returned an invalid result"
                    )
                if message[0] != "ok":
                    detail = str(message[1]) if len(message) > 1 else "unknown worker error"
                    trace = str(message[2]) if len(message) > 2 else ""
                    raise _ParallelAssetWorkerFailure(
                        f"asset worker {process.name} failed: {detail}\n{trace}"
                    )

                for index, result in message[1]:
                    output[int(index)] = result

        # The result payloads have crossed their pipes and every output file was
        # already closed. Give all children one shared grace period for ordinary
        # library teardown, then synchronously terminate any stragglers.
        _finish_processes(processes, grace_seconds=_WORKER_EXIT_GRACE_SECONDS)

    except BaseException:
        for receiver in list(receivers):
            try:
                receiver.close()
            except OSError:
                pass
        _abort_processes(processes)
        raise

    if any(result is None for result in output):
        raise _ParallelAssetWorkerFailure("one or more asset workers returned no result")
    return [cast(R, result) for result in output]


def process_asset_tasks(
    worker: Callable[[T], R],
    tasks: Iterable[T],
    *,
    minimum_parallel_tasks: int = _DEFAULT_PARALLEL_MINIMUM,
) -> list[R]:
    """Run large batches of independent deterministic asset jobs in processes.

    Cache hits stay in the parent process; spawning workers just to copy and
    inspect already-cached files is slower. Only sufficiently large cache-miss
    batches use subprocesses, because small P3D batches are faster serially after
    accounting for Python/Pillow/Shapely import cost. Detailed building P3Ds have
    a lower worker-specific cutoff; other asset families retain the conservative
    generic threshold. Set ``CWR_WORLDGEN_ASSET_WORKERS`` to explicitly tune
    worker count.

    Spawned workers use explicit one-way pipes rather than ProcessPoolExecutor.
    This avoids leaked multiprocessing semaphores and gives the parent a bounded
    cleanup path if a third-party library stalls during child interpreter exit.
    If process startup/communication fails, all children are reaped first and
    the deterministic serial path is used as a safe fallback.
    """
    task_list = list(tasks)
    if not task_list:
        return []

    results: list[R | None] = [None] * len(task_list)
    miss_indices: list[int] = []
    for index, task in enumerate(task_list):
        if _cache_hit_without_refresh(task):
            results[index] = worker(task)
        else:
            miss_indices.append(index)

    workers = asset_worker_count(len(miss_indices))
    explicit_workers = bool(os.environ.get("CWR_WORLDGEN_ASSET_WORKERS", "").strip())
    threshold = (
        2
        if explicit_workers
        else _parallel_minimum_for_worker(worker, minimum_parallel_tasks)
    )
    miss_tasks = [task_list[index] for index in miss_indices]

    if workers > 1 and len(miss_tasks) >= threshold:
        try:
            miss_results = _process_map(worker, miss_tasks, workers)
        except (_ParallelAssetWorkerFailure, OSError, RuntimeError):
            # Re-run only after _process_map has synchronously reaped every child.
            # This keeps a process-lifecycle failure from aborting world creation.
            miss_results = [worker(task) for task in miss_tasks]
    else:
        miss_results = [worker(task) for task in miss_tasks]

    for index, result in zip(miss_indices, miss_results):
        results[index] = result
    return [cast(R, result) for result in results]
