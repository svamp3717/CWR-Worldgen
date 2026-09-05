# SPDX-License-Identifier: GPL-3.0-or-later
"""Expose live progress while procedural building variants are prepared.

Preparing modeler-backed variants is mostly a serial style-resolution pass over
mapped buildings followed by bounded reuse matching for variants beyond the
configured model cap.  Both used to sit behind one static GUI status line.  Keep
the established deterministic prepare implementation authoritative and wrap only
its iterator/reuse boundaries so users can see useful completed/total counters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_PREPARE_RAW_PERCENT = 23
_STATE_ATTRIBUTE = "_cwr_building_prepare_progress_state"
_INSTALLED = False


def _dataset_building_candidate_total(dataset: Any) -> int:
    """Count the building records that ``_iter_dataset_keys`` can resolve."""
    polygon_total = 0
    for feature in getattr(dataset, "building_polygons", ()):
        for polygon in getattr(feature, "polygons", ()):
            outer = getattr(polygon, "outer", ())
            # The production iterator projects outer[:-1] and requires at least
            # three resulting vertices. Match that cheap eligibility test here.
            if len(outer[:-1]) >= 3:
                polygon_total += 1
    return polygon_total + len(getattr(dataset, "building_points", ()))


def _emit(stage: str) -> None:
    from .progress import report_progress

    report_progress(_PREPARE_RAW_PERCENT, stage)


@dataclass
class _PrepareProgressState:
    candidate_total: int
    maximum_variants: int
    resolved: int = 0
    unique_keys: set[Any] = field(default_factory=set)
    match_total: int = 0
    matched: int = 0
    _last_resolve_bucket: int = -1
    _last_match_bucket: int = -1
    scan_finished: bool = False

    def _percent(self, completed: int, total: int) -> int:
        if total <= 0:
            return 100
        return min(100, max(0, int(completed * 100 / total)))

    def report_resolve(self, *, force: bool = False) -> None:
        total = self.candidate_total
        percent = self._percent(self.resolved, total)
        bucket = percent // 2
        if (
            not force
            and self.resolved not in {1, total}
            and bucket <= self._last_resolve_bucket
        ):
            return
        self._last_resolve_bucket = bucket
        _emit(
            "Preparing procedural building variants: resolving building styles "
            f"({self.resolved}/{total}, {percent}%; {len(self.unique_keys)} unique)"
        )

    def finish_scan(self) -> None:
        self.scan_finished = True
        selected_total = min(len(self.unique_keys), max(0, self.maximum_variants))
        self.match_total = max(0, len(self.unique_keys) - selected_total)
        self.report_resolve(force=True)
        if self.match_total:
            self.report_match(force=True)

    def report_match(self, *, force: bool = False) -> None:
        if self.match_total <= 0:
            return
        percent = self._percent(self.matched, self.match_total)
        bucket = percent // 2
        if (
            not force
            and self.matched not in {1, self.match_total}
            and bucket <= self._last_match_bucket
        ):
            return
        self._last_match_bucket = bucket
        _emit(
            "Preparing procedural building variants: matching reusable variants "
            f"({self.matched}/{self.match_total}, {percent}%; "
            f"{len(self.unique_keys)} unique requests)"
        )

    def report_complete(self, selected_total: int) -> None:
        _emit(
            "Preparing procedural building variants complete "
            f"({self.resolved}/{self.candidate_total} buildings; "
            f"{len(self.unique_keys)} unique; {selected_total} selected)"
        )


def install_building_prepare_progress_policy() -> None:
    """Add prepare-time counters without changing variant selection semantics."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import procedural_buildings as buildings

    original_prepare = buildings.ProceduralBuildingLibrary.prepare
    original_iter_dataset_keys = buildings.ProceduralBuildingLibrary._iter_dataset_keys
    original_reuse_candidates = buildings.ProceduralBuildingLibrary._reuse_candidates

    def progressive_iter_dataset_keys(self, dataset, projection, point_footprint):
        state = getattr(self, _STATE_ATTRIBUTE, None)
        if state is None:
            yield from original_iter_dataset_keys(
                self, dataset, projection, point_footprint
            )
            return

        for key in original_iter_dataset_keys(
            self, dataset, projection, point_footprint
        ):
            state.resolved += 1
            state.unique_keys.add(key)
            state.report_resolve()
            yield key
        state.finish_scan()

    def progressive_reuse_candidates(self, requested, candidates):
        result = original_reuse_candidates(self, requested, candidates)
        state = getattr(self, _STATE_ATTRIBUTE, None)
        if state is not None and state.scan_finished and state.match_total:
            state.matched += 1
            state.report_match()
        return result

    def progressive_prepare(self, dataset, projection, point_footprint):
        state = _PrepareProgressState(
            candidate_total=_dataset_building_candidate_total(dataset),
            maximum_variants=max(0, int(getattr(self, "maximum_variants", 0) or 0)),
        )
        setattr(self, _STATE_ATTRIBUTE, state)
        state.report_resolve(force=True)
        succeeded = False
        try:
            result = original_prepare(self, dataset, projection, point_footprint)
            succeeded = True
            return result
        finally:
            try:
                if succeeded:
                    selected_total = len(set(getattr(self, "_mapping", {}).values()))
                    # A complete prepare should invoke reuse exactly once for each
                    # unique request beyond the selected cap. Do not fabricate a
                    # counter if a future prepare algorithm legitimately differs.
                    if state.match_total and state.matched != state.match_total:
                        _emit(
                            "Preparing procedural building variants: reuse matching "
                            f"completed ({state.matched} matches observed; "
                            f"{state.match_total} expected from current cap)"
                        )
                    state.report_complete(selected_total)
            finally:
                try:
                    delattr(self, _STATE_ATTRIBUTE)
                except AttributeError:
                    pass

    buildings.ProceduralBuildingLibrary._iter_dataset_keys = progressive_iter_dataset_keys
    buildings.ProceduralBuildingLibrary._reuse_candidates = progressive_reuse_candidates
    buildings.ProceduralBuildingLibrary.prepare = progressive_prepare
    _INSTALLED = True

    # This module is installed late in package startup, after the paved/gravel/
    # raceway road wrappers. That makes it a safe place to add the audit wrapper
    # without bypassing any of those policies.
    from .road_audit_performance_policy import install_road_audit_performance_policy

    install_road_audit_performance_policy()

    # The audit counter exposed a second post-fit hotspot: paved-junction target
    # selection and cleanup. Install its indexed replacement after the progress
    # ContextVar exists so all sub-phases can reuse the same GUI callback.
    from .paved_junction_performance_policy import (
        install_paved_junction_performance_policy,
    )

    install_paved_junction_performance_policy()

    # Final straight-road deduplication must be outermost: only after the complete
    # junction/gravel/raceway chain has produced its final object list do we know
    # which slabs would actually overlap in the WRP.
    from .final_road_dedup_policy import install_final_road_dedup_policy

    install_final_road_dedup_policy()
