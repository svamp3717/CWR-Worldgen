# SPDX-License-Identifier: GPL-3.0-or-later
"""Commit local road deviations only when the complete junction fit is safe.

The local-fit policy deliberately allows a small amount of source-line freedom
so shallow paved dog-legs and skewed T nodes can use cleaner stock geometry.
Planning that geometry is necessarily more permissive than final placement: a
native T may be considered while deciding where its approaches would need to
move.  That wider matcher must not leak into the final cap-selection pass.

This policy makes that distinction explicit.  Relaxed matching is enabled only
inside the approach-planning call.  Planned edits are grouped by junction node,
all changed arms must pass the source-backed obstacle corridor together, and the
resulting geometry must then satisfy the ordinary strict junction matcher.  If
any part fails, the whole node keeps its original geometry.  Humans call this a
transaction; road meshes call it finally not being asked to improvise.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Mapping

from . import playability as _p
from . import stock_road_connector_policy as _connector
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local
from . import stock_road_relaxation_policy as _relax
from . import stock_road_skew_policy as _skew

_PLANNING_RELAXED_JUNCTION: ContextVar[bool] = ContextVar(
    "cwr_planning_relaxed_stock_junction", default=False
)

_ORIGINAL_COLLECT_RELAXATIONS = None
_INSTALLED = False


def _strict_native_junction_for_incidents(incidents):
    """Return the non-relaxed measured junction choice used for final placement."""

    # stock_road_skew_policy captured this function before it installed its
    # temporary relaxed matcher.  The function still sees the later measured
    # geometry and gravel/asphalt surface policy, but it obeys the normal native
    # heading tolerance.  In particular it does not inherit the local-fit
    # policy's same-family paved relaxation eligibility.
    strict = _skew._ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS
    if strict is None:
        raise RuntimeError("stock road skew policy is not installed")
    return strict(incidents)


def _native_junction_for_incidents(incidents):
    """Use the wider matcher only while connector-aligned edits are being planned."""

    if _PLANNING_RELAXED_JUNCTION.get():
        return _local._native_junction_for_incidents(incidents)
    return _strict_native_junction_for_incidents(incidents)


def _group_relaxations(projected, relaxations):
    grouped: dict[
        tuple[int, int],
        dict[tuple[int, int, int], tuple[float, float]],
    ] = {}
    for key, point in relaxations.items():
        feature_index, node_index, _neighbour_index = key
        node = tuple(projected[feature_index][node_index])
        grouped.setdefault(_p._road_node_key(node), {})[key] = point
    return grouped


def _group_is_obstacle_safe(projected, plans, obstacles) -> bool:
    """Require every changed arm at one junction to survive the same safety pass."""

    for key, point in plans.items():
        feature_index, node_index, neighbour_index = key
        node = tuple(projected[feature_index][node_index])
        neighbour = tuple(projected[feature_index][neighbour_index])
        if not _relax._shortcut_clear(obstacles, node, point):
            return False
        if not _relax._shortcut_clear(obstacles, point, neighbour):
            return False
    return True


def _flatten(groups: Mapping[tuple[int, int], Mapping]) -> dict:
    result = {}
    for plans in groups.values():
        result.update(plans)
    return result


def _strict_match_keys(dataset, projection, projected, spec) -> set[tuple[int, int]]:
    """Return nodes whose *actual candidate geometry* satisfies strict matching."""

    token = _connector._RELAXED_PROJECTED_ROADS.set(projected)
    try:
        return set(_junction._junction_incidents(dataset, projection, spec))
    finally:
        _connector._RELAXED_PROJECTED_ROADS.reset(token)


def _collect_relaxations(dataset, projection, projected, spec):
    if _ORIGINAL_COLLECT_RELAXATIONS is None:
        raise RuntimeError("stock road relaxation transaction policy is not installed")

    # First plan with the wider matcher.  Nothing from this block is committed
    # yet, and the ContextVar prevents the permissive matcher from leaking into
    # the later junction-cap replacement pass.
    planning_token = _PLANNING_RELAXED_JUNCTION.set(True)
    try:
        planned = _ORIGINAL_COLLECT_RELAXATIONS(
            dataset, projection, projected, spec
        )
    finally:
        _PLANNING_RELAXED_JUNCTION.reset(planning_token)

    if not planned:
        return planned

    context = _relax._CONTEXT.get()
    if context is None:
        # Production fitting installs the obstacle context outside this call.
        # Without it, fail closed rather than moving source geometry blindly.
        return {}

    groups = _group_relaxations(projected, planned)
    groups = {
        key: plans
        for key, plans in groups.items()
        if _group_is_obstacle_safe(projected, plans, context.obstacles)
    }
    if not groups:
        return {}

    # A planner can legitimately be unable to move one short arm far enough to
    # reach the candidate connector.  Validate the geometry after all surviving
    # edits are applied and discard a whole node unless the ordinary strict
    # matcher now accepts it.  Iterate because two very close junctions can
    # share one road feature; removing one edit may change the local segment
    # seen by its neighbour.
    for _ in range(4):
        candidate = _connector._apply_relaxations(projected, _flatten(groups))
        strict_keys = _strict_match_keys(dataset, projection, candidate, spec)
        retained = {
            key: plans for key, plans in groups.items() if key in strict_keys
        }
        if retained.keys() == groups.keys():
            break
        groups = retained
        if not groups:
            return {}

    return _flatten(groups)


def install_stock_road_relaxation_transaction_policy() -> None:
    global _ORIGINAL_COLLECT_RELAXATIONS, _INSTALLED
    if _INSTALLED:
        return

    # Bypass the local-fit wrapper's per-arm filtering.  This policy performs
    # the same obstacle checks as one all-or-nothing junction transaction and
    # then validates the resulting node against the strict matcher.
    _ORIGINAL_COLLECT_RELAXATIONS = _local._ORIGINAL_COLLECT_RELAXATIONS
    if _ORIGINAL_COLLECT_RELAXATIONS is None:
        raise RuntimeError("stock road local fit policy must be installed first")

    _junction._native_junction_for_incidents = _native_junction_for_incidents
    _connector._collect_relaxations = _collect_relaxations
    _INSTALLED = True
