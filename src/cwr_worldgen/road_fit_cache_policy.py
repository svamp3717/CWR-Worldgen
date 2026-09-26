# SPDX-License-Identifier: GPL-3.0-or-later
"""Persist the expensive deterministic core road-fit stage between builds.

The public generator deliberately keeps final road cleanup/building-clearance
wrappers outside this cache. Those lightweight wrappers still execute on every
build, preserving their per-build ContextVar side effects, while stock chain
fitting and paved-junction fallback planning can be restored from disk.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import os
from pathlib import Path
import pickle
from typing import Any

from . import generator as _generator
from . import playability as _p
from .cache import (
    CACHE_SCHEMA_VERSION,
    cache_key,
    float_sequence_sha256,
    streaming_hash,
)

_STAGE_SCHEMA = 1
_CACHE_NAMESPACE = "road-fit-core-v1-final-quantized-terrain"
_PROGRESS_PERCENT = 99
_INSTALLED = False
_ORIGINAL_FIT = None
_ORIGINAL_WRITE_JSON = None
_CODE_FINGERPRINT: str | None = None
_LAST_CACHE_INFO: ContextVar[dict[str, Any] | None] = ContextVar(
    "cwr_road_fit_cache_info", default=None
)

# Only source that can affect the cached *inner* road-fit result belongs here.
# Final deduplication/bridge-underlay/building-clearance wrappers are installed
# later and therefore still run after a cache hit.
_CODE_FILES = (
    "__init__.py",
    "osm.py",
    "playability.py",
    "procedural_infrastructure.py",
    "road_quality_policy.py",
    "paved_junction_policy.py",
    "paved_junction_fallback_policy.py",
    "paved_junction_performance_policy.py",
    "paved_junction_parallel_planning_policy.py",
    "gravel_junction_policy.py",
    "gravel_gap_policy.py",
    "gravel_family_policy.py",
    "paved_road_generated_fallback_policy.py",
    "road_chain_parallel_policy.py",
    "road_quality_parallel_compat_policy.py",
    "bridge_runtime_policy.py",
    "bridge_source_water_policy.py",
    "bridge_source_water_performance_policy.py",
    "object_stage_parallel_policy.py",
    "road_finish_parallel_policy.py",
    "road_audit_performance_policy.py",
)


def _runtime_code_fingerprint() -> str:
    """Hash road-fit implementation files so code changes invalidate old fits."""
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is not None:
        return _CODE_FINGERPRINT

    root = Path(__file__).resolve().parent
    digest = sha256()
    digest.update(_CACHE_NAMESPACE.encode("utf-8"))
    for name in _CODE_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            data = path.read_bytes()
        except OSError:
            digest.update(b"<missing>")
        else:
            digest.update(sha256(data).digest())
    _CODE_FINGERPRINT = digest.hexdigest()
    return _CODE_FINGERPRINT


def _dataset_fingerprint(dataset: Any) -> str:
    normalized = getattr(dataset, "normalized_fingerprint", None)
    if normalized:
        return str(normalized)
    return streaming_hash("road-fit-dataset-fallback-v1", dataset)


def _spec_payload(spec: Any) -> Any:
    if is_dataclass(spec) and not isinstance(spec, type):
        payload = asdict(spec)
    else:
        try:
            payload = dict(vars(spec))
        except TypeError:
            return repr(spec)
    # Runtime cache controls and output packaging do not alter fitted road objects.
    for field in (
        "cache_dir",
        "cache_enabled",
        "cache_refresh",
        "pbo_backend",
        "poseidon_tools_path",
        "verify_regeneration",
    ):
        payload.pop(field, None)
    return payload


def _road_fit_key(dataset, projection, elevations, spec, starting_id: int) -> str:
    return cache_key(
        _CACHE_NAMESPACE,
        {
            "dataset": _dataset_fingerprint(dataset),
            "projection": streaming_hash("road-fit-projection-v1", projection),
            "elevations": float_sequence_sha256(elevations),
            "spec": _spec_payload(spec),
            "starting_id": int(starting_id),
            "code": _runtime_code_fingerprint(),
        },
    )


def _cache_path(spec, key: str) -> Path | None:
    root = getattr(spec, "cache_dir", None)
    if root is None:
        return None
    return Path(root) / "roads" / f"road-fit-{key}.pickle"


def _valid_report(value: Any) -> bool:
    return isinstance(value, _p.RoadFitReport)


def _load(path: Path) -> Any | None:
    try:
        with path.open("rb") as stream:
            envelope = pickle.load(stream)
    except (OSError, EOFError, ValueError, TypeError, AttributeError, pickle.PickleError):
        return None
    if (
        not isinstance(envelope, dict)
        or envelope.get("cache_schema") != CACHE_SCHEMA_VERSION
        or envelope.get("stage_schema") != _STAGE_SCHEMA
        or "value" not in envelope
    ):
        return None
    value = envelope["value"]
    return value if _valid_report(value) else None


def _store(path: Path, report: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(
                {
                    "cache_schema": CACHE_SCHEMA_VERSION,
                    "stage_schema": _STAGE_SCHEMA,
                    "value": report,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except (OSError, pickle.PickleError):
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _cache_info(*, hit: bool, key: str | None, path: Path | None, report: Any) -> dict[str, Any]:
    return {
        "hit": bool(hit),
        "key": key,
        "path": str(path) if path is not None else None,
        "objects": len(getattr(report, "objects", ())),
        "chains": int(getattr(report, "chain_count", 0)),
        "stage_schema": _STAGE_SCHEMA,
        "code_fingerprint": _runtime_code_fingerprint(),
    }


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    # The deterministic-verification pass intentionally calls fit_road_objects
    # without a progress callback. Bypass this stage cache there so verification
    # remains a real regeneration rather than a comparison against cached bytes.
    if progress_callback is None:
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=None,
        )

    enabled = bool(getattr(spec, "cache_enabled", True))
    refresh = bool(getattr(spec, "cache_refresh", False))
    if not enabled or getattr(spec, "cache_dir", None) is None:
        report = _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        _LAST_CACHE_INFO.set(_cache_info(hit=False, key=None, path=None, report=report))
        return report

    progress_callback(0, "Checking road-fit cache")
    key = _road_fit_key(dataset, projection, elevations, spec, starting_id)
    path = _cache_path(spec, key)

    if not refresh and path is not None and path.is_file():
        report = _load(path)
        if report is not None:
            _LAST_CACHE_INFO.set(_cache_info(hit=True, key=key, path=path, report=report))
            progress_callback(
                100,
                "Road fitting ready from cache: "
                f"{len(report.objects):,} core objects in {report.chain_count:,} chains",
            )
            return report

    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )

    wrote = False
    if path is not None:
        progress_callback(
            _PROGRESS_PERCENT,
            "Writing road-fit cache: "
            f"{len(report.objects):,} core objects in {report.chain_count:,} chains",
        )
        wrote = _store(path, report)

    _LAST_CACHE_INFO.set(_cache_info(hit=False, key=key, path=path, report=report))
    if wrote:
        progress_callback(
            100,
            "Road-fit cache stored: "
            f"{len(report.objects):,} core objects in {report.chain_count:,} chains",
        )
    elif path is not None:
        progress_callback(
            _PROGRESS_PERCENT,
            "Road-fit cache write failed; continuing with in-memory result",
        )
    return report


def _write_json(path, value) -> None:
    if Path(path).name == "cache-report.json" and isinstance(value, dict):
        info = _LAST_CACHE_INFO.get()
        if info is not None:
            value = dict(value)
            value["road_fit"] = dict(info)
    _ORIGINAL_WRITE_JSON(path, value)


def install_road_fit_cache_policy() -> None:
    """Cache the expensive inner road-fit stage without skipping late side effects."""
    global _INSTALLED, _ORIGINAL_FIT, _ORIGINAL_WRITE_JSON
    if _INSTALLED:
        return

    _ORIGINAL_FIT = _p.fit_road_objects
    _ORIGINAL_WRITE_JSON = _generator._write_json
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _generator._write_json = _write_json
    _INSTALLED = True
