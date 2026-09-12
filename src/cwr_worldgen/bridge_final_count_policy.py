# SPDX-License-Identifier: GPL-3.0-or-later
"""Make the final wet bridge plan authoritative over emitted stock objects.

Several bridge-generation paths can emit stock objects before the final runtime
planner is installed. Overlapping source features can also emit interleaved
objects that later become one physical stock-bridge component. The seam repair
pass historically preserved however many objects already existed, so a correct
three-module wet plan could still become a perfectly joined ten-module bridge.

Normalize each physical bridge component against the final terrain-aware stock
plan immediately before the existing 3-D seam anchoring pass. The final plan
controls both module count and horizontal placement. If the final terrain has
no wet span, leave the component alone so source-water fallback behavior remains
available to the earlier source-aware generation path.

ObjectGenerationResult also carries category counters and an optional model-usage
aggregate beside the object tuple. Keep those in lockstep when bridge objects are
removed or cloned; downstream ordering deliberately rejects inconsistent counts.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import bridge_render_policy as _bridge

_INSTALLED = False
_ORIGINAL_ANCHOR = None


def _component_corridor(objects, ordered, axis, midpoint):
    """Return the full nominal stock corridor represented by one component."""
    count = len(ordered)
    half_span = float(_bridge._STOCK_MODULE_SPACING_METRES) * count * 0.5
    return (
        (
            float(midpoint[0]) - float(axis[0]) * half_span,
            float(midpoint[1]) - float(axis[1]) * half_span,
        ),
        (
            float(midpoint[0]) + float(axis[0]) * half_span,
            float(midpoint[1]) + float(axis[1]) * half_span,
        ),
    )


def _planned_centres(plan):
    """Return exact visible-deck centres for the final fixed-stock plan."""
    count = max(0, int(plan.module_count))
    if count <= 0:
        return ()
    start, end = plan.points
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    length = math.hypot(dx, dz)
    if length <= 0.1:
        return ()
    ux, uz = dx / length, dz / length
    step = float(_bridge._STOCK_MODULE_SPACING_METRES)
    return tuple(
        (
            float(start[0]) + ux * step * (index + 0.5),
            float(start[1]) + uz * step * (index + 0.5),
        )
        for index in range(count)
    )


def _updated_model_usage(result, bridge_delta: int):
    """Apply a stock-bridge population delta to an existing model aggregate."""
    usage = tuple(getattr(result, "model_usage", ()) or ())
    if not usage or bridge_delta == 0:
        return usage

    stock_key = None
    values: dict[str, int] = {}
    for model, count in usage:
        key = str(model)
        values[key] = int(count)
        if key.replace("/", "\\").casefold() == _bridge._STOCK_MODEL:
            stock_key = key

    if stock_key is None:
        # A non-empty aggregate should normally already contain the bridge model,
        # but preserve correctness if an older cache omitted it.
        stock_key = str(_bridge._osm.NOGOVA_BRIDGE_MODEL)
        values.setdefault(stock_key, 0)

    values[stock_key] = max(0, values.get(stock_key, 0) + int(bridge_delta))
    if values[stock_key] <= 0:
        values.pop(stock_key, None)
    return tuple(sorted(values.items(), key=lambda item: item[0].casefold()))


def _reconcile_stock_bridge_components(result, elevations, spec):
    """Resize and recenter physical stock chains to the final wet-span plan."""
    if result is None or elevations is None or spec is None:
        return result
    objects = list(tuple(getattr(result, "objects", ()) or ()))
    if not objects:
        return result

    stock_before = sum(1 for obj in objects if _bridge._is_stock_bridge(obj))
    components = _bridge._bridge_components(objects)
    if not components:
        return result

    replacements: dict[int, object] = {}
    removals: set[int] = set()
    additions: list[object] = []
    next_id = max((int(getattr(obj, "object_id", 0)) for obj in objects), default=0) + 1
    changed = False

    for component in components:
        ordered_state = _bridge._ordered_component(objects, component)
        if ordered_state is None:
            continue
        ordered, axis, midpoint = ordered_state
        if not ordered:
            continue

        corridor = _component_corridor(objects, ordered, axis, midpoint)
        # The runtime installs its final wet-only planner into this module-level
        # binding. Calling it here therefore uses the same plan the renderer is
        # supposed to honor, rather than trusting the number of objects emitted
        # by an earlier source feature or cache entry.
        plan = _bridge.stock_bridge_span_plan(corridor, elevations, spec)
        if plan is None or int(plan.module_count) <= 0:
            continue

        centres = _planned_centres(plan)
        required = len(centres)
        if required <= 0:
            continue

        start, end = plan.points
        dx = float(end[0]) - float(start[0])
        dz = float(end[1]) - float(start[1])
        heading = math.degrees(math.atan2(dx, dz)) % 360.0
        deck_heights = [
            float(_bridge._visible_deck_point(objects[index], 0.0)[1])
            for index in ordered
        ]
        placeholder_deck_y = sum(deck_heights) / len(deck_heights)

        # Keep stable IDs for as many existing objects as possible. Excess
        # objects are deleted; missing objects are cloned with fresh IDs. The
        # normal anchor pass immediately recomputes final Y/pitch from the banks.
        templates = list(ordered[:required])
        if len(ordered) > required:
            removals.update(ordered[required:])
        while len(templates) < required:
            templates.append(ordered[-1])

        for position, (template_index, centre) in enumerate(zip(templates, centres)):
            template = objects[template_index]
            object_id = int(template.object_id)
            is_clone = position >= len(ordered)
            if is_clone:
                object_id = next_id
                next_id += 1
            origin_x, origin_y, origin_z = _bridge._model_origin_for_visible_deck_center(
                float(centre[0]),
                placeholder_deck_y,
                float(centre[1]),
                heading,
                0.0,
            )
            rewritten = replace(
                template,
                object_id=object_id,
                x=origin_x,
                y=origin_y,
                z=origin_z,
                heading_degrees=heading,
                pitch_degrees=0.0,
            )
            if is_clone:
                additions.append(rewritten)
            else:
                replacements[template_index] = rewritten
        changed = changed or required != len(ordered) or any(
            math.dist(
                (
                    _bridge._visible_deck_point(objects[index], 0.0)[0],
                    _bridge._visible_deck_point(objects[index], 0.0)[2],
                ),
                centres[position],
            ) > 0.05
            for position, index in enumerate(ordered[:required])
        )

    if not changed:
        return result

    reconciled = [
        replacements.get(index, obj)
        for index, obj in enumerate(objects)
        if index not in removals
    ]
    reconciled.extend(additions)

    stock_after = sum(1 for obj in reconciled if _bridge._is_stock_bridge(obj))
    bridge_delta = stock_after - stock_before
    updates: dict[str, object] = {"objects": tuple(reconciled)}
    if hasattr(result, "bridge_objects"):
        updates["bridge_objects"] = max(
            0,
            int(getattr(result, "bridge_objects", 0)) + bridge_delta,
        )
    if hasattr(result, "model_usage"):
        updates["model_usage"] = _updated_model_usage(result, bridge_delta)
    return replace(result, **updates)


def install_bridge_final_count_policy() -> None:
    """Normalize bridge count/placement immediately before final seam anchoring."""
    global _INSTALLED, _ORIGINAL_ANCHOR
    if _INSTALLED:
        return

    _ORIGINAL_ANCHOR = _bridge._anchor_stock_bridge_chains

    def authoritative_anchor(result, raster, elevations, spec):
        reconciled = _reconcile_stock_bridge_components(result, elevations, spec)
        return _ORIGINAL_ANCHOR(reconciled, raster, elevations, spec)

    _bridge._anchor_stock_bridge_chains = authoritative_anchor
    _INSTALLED = True
