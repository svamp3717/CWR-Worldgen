# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, read-only WRP comparison across 4WVR and OFP/CWA OPRW v2/v3.

Objects are matched by *unique object IDs*, not file order. Duplicate IDs
are reported as ambiguous rather than confidently attributing moved objects.
Actual model names and height values are compared across binary formats.
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Callable

import numpy as np

_MAX_MARKERS = 300
_ELEVATION_TOLERANCE = 0.01
_POSITION_TOLERANCE = 0.01
_TRANSFORM_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class DifferencePoint:
    category: str
    x: float
    z: float
    description: str


@dataclass(frozen=True, slots=True)
class WorldComparison:
    reference: str
    candidate: str
    reference_dimensions: tuple[int, int]
    candidate_dimensions: tuple[int, int]
    reference_cell_metres: float
    candidate_cell_metres: float
    reference_objects: int
    candidate_objects: int
    terrain_changes: int | None
    max_height_difference: float | None
    texture_changes: int | None
    texture_table_changes: int
    unique_ids_reference_only: int
    unique_ids_candidate_only: int
    ambiguous_ids: int
    moved_objects: int
    rotated_or_scaled_objects: int
    changed_models: int
    positions: tuple[DifferencePoint, ...]

    def report(self) -> str:
        def quantity(value: int | None) -> str:
            return "N/A (grid dimensions differ)" if value is None else f"{value:,}"
        return (
            f"REFERENCE: {self.reference}\n"
            f"CANDIDATE: {self.candidate}\n\n"
            f"Terrain grids: {self.reference_dimensions[0]} × {self.reference_dimensions[1]}"
            f" vs {self.candidate_dimensions[0]} × {self.candidate_dimensions[1]}\n"
            f"Grid spacing: {self.reference_cell_metres:g} m vs "
            f"{self.candidate_cell_metres:g} m\n"
            f"Elevation samples changed (>1 cm): {quantity(self.terrain_changes)}\n"
            f"Largest height difference: "
            + ("N/A" if self.max_height_difference is None
               else f"{self.max_height_difference:.4f} m")
            + "\n"
            f"Texture cells changed: {quantity(self.texture_changes)}\n"
            f"Texture table slots changed: {self.texture_table_changes:,}\n\n"
            f"Placed objects: {self.reference_objects:,} vs {self.candidate_objects:,}\n"
            f"Object IDs only in reference: {self.unique_ids_reference_only:,}\n"
            f"Object IDs only in candidate: {self.unique_ids_candidate_only:,}\n"
            f"Ambiguous duplicate-ID placements: {self.ambiguous_ids:,}\n"
            f"Objects moved (>1 cm): {self.moved_objects:,}\n"
            f"Rotated/scaled objects: {self.rotated_or_scaled_objects:,}\n"
            f"Objects using different P3D paths: {self.changed_models:,}\n\n"
            "Differences are read-only. Object matches require unique IDs; "
            "duplicate IDs are counted but not position-matched. "
            "Terrain compares loaded WRP samples (not unsaved edits). "
            "Metadata such as OPRW geography/sound flags is not compared."
        )


def _models(document) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(document, "model_names"):
        names = tuple(str(name).replace("/", "\\").casefold() for name in document.model_names)
        return np.asarray(names, dtype=object), document._records["model"].astype(np.int32, copy=False)
    unique, inverse = np.unique(document._records["model"], return_inverse=True)
    names = np.asarray([
        raw.split(b"\0", 1)[0].decode("latin-1", "replace").replace("/", "\\").casefold()
        for raw in unique
    ], dtype=object)
    return names, inverse


def compare_worlds(
    reference, candidate, *,
    cancel: Callable[[], bool] | None = None,
    max_markers: int = _MAX_MARKERS,
) -> WorldComparison:
    """Compare opened documents without indexing, modifying or closing either.

    The function is suitable for the editor's background executor. Row blocks
    bound temporary terrain buffers; model names are decoded once per unique
    model, not once per placement.
    """
    markers: list[DifferencePoint] = []
    max_markers = max(0, min(_MAX_MARKERS, max_markers))

    def cancelled() -> None:
        if cancel is not None and cancel():
            raise CancelledError()

    def mark(kind: str, x: float, z: float, description: str) -> None:
        if len(markers) < max_markers:
            markers.append(DifferencePoint(kind, float(x), float(z), description))

    cancelled()
    dims_a = (reference.width, reference.height)
    dims_b = (candidate.width, candidate.height)
    terrain_changes: int | None = None
    maximum_height: float | None = None
    texture_changes: int | None = None

    if dims_a == dims_b:
        terrain_changes = 0
        maximum_height = 0.0
        for row in range(0, reference.height, 256):
            cancelled()
            end = min(reference.height, row + 256)
            # OPRW stores physical metre heights directly; its GUI height-scale
            # entry must not accidentally rescale those floats. 4WVR stores
            # int16 samples whose scale is supplied by the user.
            scale_a = 1.0 if hasattr(reference, "oprw_version") else reference.height_scale
            scale_b = 1.0 if hasattr(candidate, "oprw_version") else candidate.height_scale
            a = np.asarray(reference.heights[row:end], dtype=np.float64) * scale_a
            b = np.asarray(candidate.heights[row:end], dtype=np.float64) * scale_b
            diff = np.abs(a - b)
            changed = diff > _ELEVATION_TOLERANCE
            terrain_changes += int(np.count_nonzero(changed))
            if diff.size:
                maximum_height = max(maximum_height, float(np.max(diff)))
            if bool(np.any(changed)) and len(markers) < max_markers:
                ys, xs = np.where(changed)
                for y, x in zip(ys[:max_markers-len(markers)], xs[:max_markers-len(markers)]):
                    mark("Terrain", x * reference.cell_size,
                         (row + y) * reference.cell_size,
                         f"Elevation Δ {diff[y,x]:.3f} m (cell {x}, {row+y})")

    ta = reference.textures
    tb = candidate.textures
    if ta.shape == tb.shape:
        texture_changes = 0
        layer_height, layer_width = ta.shape
        for row in range(0, layer_height, 256):
            cancelled()
            end = min(layer_height, row + 256)
            differences = np.asarray(ta[row:end]) != np.asarray(tb[row:end])
            texture_changes += int(np.count_nonzero(differences))
            if len(markers) < max_markers and bool(np.any(differences)):
                ys, xs = np.where(differences)
                for y, x in zip(ys[:max_markers-len(markers)], xs[:max_markers-len(markers)]):
                    wx = x * reference.world_width / layer_width
                    wz = (row + y) * reference.world_height / layer_height
                    mark("Texture", wx, wz, f"Texture index {int(ta[row+y,x])} → {int(tb[row+y,x])}")

    # Compare the effective non-empty texture slot name, not table padding.
    tex_a = tuple(name.replace("/", "\\").casefold() for name in reference.texture_paths)
    tex_b = tuple(name.replace("/", "\\").casefold() for name in candidate.texture_paths)
    texture_table_changes = sum(
        a != b for a, b in zip(
            tex_a + ("",) * max(0, len(tex_b) - len(tex_a)),
            tex_b + ("",) * max(0, len(tex_a) - len(tex_b)),
        )
    )

    cancelled()
    ids_a = reference._records["object_id"]
    ids_b = candidate._records["object_id"]
    unique_a, counts_a = np.unique(ids_a, return_counts=True)
    unique_b, counts_b = np.unique(ids_b, return_counts=True)
    ambiguous = int(np.sum(counts_a[counts_a > 1]) + np.sum(counts_b[counts_b > 1]))
    unique_ids_a = unique_a[counts_a == 1]
    unique_ids_b = unique_b[counts_b == 1]
    missing_a = np.setdiff1d(unique_a, unique_b, assume_unique=True)
    missing_b = np.setdiff1d(unique_b, unique_a, assume_unique=True)
    only_a = len(missing_a)
    only_b = len(missing_b)
    cancelled()

    ids_common = np.intersect1d(unique_ids_a, unique_ids_b, assume_unique=True)
    pos_a = np.argsort(ids_a, kind="stable")
    pos_b = np.argsort(ids_b, kind="stable")
    indexes_a = pos_a[np.searchsorted(ids_a[pos_a], ids_common)]
    indexes_b = pos_b[np.searchsorted(ids_b[pos_b], ids_common)]
    # Report sample locations for objects present in only one file too.
    for label, missing, ids, order, document in (
        ("Reference only", missing_a, ids_a, pos_a, reference),
        ("Candidate only", missing_b, ids_b, pos_b, candidate),
    ):
        if len(markers) >= max_markers:
            break
        for object_id in missing[:max_markers-len(markers)]:
            record_index = order[np.searchsorted(ids[order], object_id)]
            record = document._records["transform"][record_index]
            mark("Missing object", record[9], record[11],
                 f"{label}: object ID {int(object_id)}")
    cancelled()

    trans_a = reference._records["transform"]
    trans_b = candidate._records["transform"]
    moved = rotated = changed_models = 0
    names_a, model_a = _models(reference)
    names_b, model_b = _models(candidate)
    common_names = {
        name: idx for idx, name in enumerate(sorted(set(names_a) | set(names_b)))
    }
    model_map_a = np.asarray([common_names[name] for name in names_a], dtype=np.int32)
    model_map_b = np.asarray([common_names[name] for name in names_b], dtype=np.int32)

    for start in range(0, len(ids_common), 100000):
        cancelled()
        end = min(len(ids_common), start + 100000)
        aa = indexes_a[start:end]
        bb = indexes_b[start:end]
        left = trans_a[aa]
        right = trans_b[bb]
        d = np.abs(left.astype(np.float64) - right.astype(np.float64))
        displacement = np.any(d[:, (9, 10, 11)] > _POSITION_TOLERANCE, axis=1)
        orientation = np.any(d[:, :9] > _TRANSFORM_TOLERANCE, axis=1)
        models_differ = model_map_a[model_a[aa]] != model_map_b[model_b[bb]]
        moved += int(np.count_nonzero(displacement))
        rotated += int(np.count_nonzero(orientation))
        changed_models += int(np.count_nonzero(models_differ))
        if len(markers) < max_markers:
            mask = displacement | orientation | models_differ
            for j in np.flatnonzero(mask)[:max_markers-len(markers)]:
                label = []
                if displacement[j]:
                    label.append("moved")
                if orientation[j]:
                    label.append("rotated/scaled")
                if models_differ[j]:
                    label.append("model changed")
                mark("Object", left[j, 9], left[j, 11],
                     f"Object ID {int(ids_common[start+j])}: {', '.join(label)}")

    return WorldComparison(
        reference=getattr(reference, "source_label", reference.path.name),
        candidate=getattr(candidate, "source_label", candidate.path.name),
        reference_dimensions=dims_a, candidate_dimensions=dims_b,
        reference_cell_metres=reference.cell_size,
        candidate_cell_metres=candidate.cell_size,
        reference_objects=reference.object_count,
        candidate_objects=candidate.object_count,
        terrain_changes=terrain_changes, max_height_difference=maximum_height,
        texture_changes=texture_changes,
        texture_table_changes=texture_table_changes,
        unique_ids_reference_only=only_a, unique_ids_candidate_only=only_b,
        ambiguous_ids=ambiguous, moved_objects=moved,
        rotated_or_scaled_objects=rotated, changed_models=changed_models,
        positions=tuple(markers),
    )
