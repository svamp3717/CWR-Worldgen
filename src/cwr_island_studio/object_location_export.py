"""Portable, read-only ZIP export of every Island Studio object placement.

Works with 4WVR and binarized OPRW documents, including sparse unsaved 4WVR
object edits. Streams the records so large worlds do not need a Python list of
all their placements or an object spatial index.
"""
from __future__ import annotations

from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .core import IslandDocument


def export_object_location_atlas(
    doc: IslandDocument,
    destination: Path | str,
    *,
    progress=None,
) -> Path:
    """Export all active objects, a model inventory, and a north-up density PNG.

    Coordinates are original WRP world metres: X east, Z north, Y elevation.
    A record's four horizontal basis values preserve scale/shear information.
    Record indices remain original and stable even after sparse edits/deletes.
    """
    destination = Path(destination)
    if destination.suffix.casefold() != ".zip":
        raise ValueError("Object atlas destination must be a .zip file")
    world_width = float(doc.world_width)
    world_height = float(doc.world_height)
    if (not math.isfinite(world_width) or not math.isfinite(world_height)
            or world_width <= 0 or world_height <= 0):
        raise ValueError("Invalid world dimensions")

    # Capped image memory, irrespective of how many millions of objects a
    # map contains. The CSV is the lossless representation of every placement.
    image_width = min(2048, max(1, int(doc.width)))
    image_height = min(2048, max(1, int(doc.height)))
    density = np.zeros((image_height, image_width), dtype=np.uint32)
    models: Counter[str] = Counter()
    extents: dict[str, list[float]] = {}
    record_count = int(doc.object_count) + len(getattr(doc, "added", ()))
    deleted = getattr(doc, "deleted", set())
    emitted = 0
    outside = 0

    if progress is not None:
        progress("Exporting exact WRP object positions…")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
            allowZip64=True,
        ) as archive:
            # Stream UTF-8 CSV direct into the archive. Do not materialize one
            # MapObject per world object in memory at the same time.
            with archive.open("object-placements.csv", "w", force_zip64=True) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow((
                        "record_index", "object_id", "model_path",
                        "x_metres", "y_metres", "z_metres", "heading_degrees",
                        "basis_xx", "basis_xz", "basis_zx", "basis_zz",
                    ))
                    for index in range(record_count):
                        if index in deleted:
                            continue
                        obj = doc.object_at(index)
                        writer.writerow((
                            obj.index, obj.object_id, obj.model_path,
                            obj.x, obj.y, obj.z, obj.heading,
                            obj.basis_xx, obj.basis_xz, obj.basis_zx, obj.basis_zz,
                        ))
                        emitted += 1
                        models[obj.model_path] += 1
                        if math.isfinite(obj.x) and math.isfinite(obj.z):
                            box = extents.setdefault(
                                obj.model_path, [obj.x, obj.x, obj.z, obj.z],
                            )
                            box[0] = min(box[0], obj.x)
                            box[1] = max(box[1], obj.x)
                            box[2] = min(box[2], obj.z)
                            box[3] = max(box[3], obj.z)
                            if (0 <= obj.x < world_width
                                    and 0 <= obj.z < world_height):
                                col = min(image_width - 1, int(obj.x / world_width * image_width))
                                # Pixel zero is geographical north.
                                row = min(image_height - 1, int(
                                    (world_height - obj.z) / world_height * image_height
                                ))
                                density[row, col] += 1
                            else:
                                outside += 1
                        else:
                            outside += 1
                        if progress is not None and emitted % 25000 == 0:
                            progress(f"Exported {emitted:,} object placements…")

            with archive.open("object-models.csv", "w", force_zip64=True) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow((
                        "model_path", "object_count",
                        "x_min_metres", "x_max_metres",
                        "z_min_metres", "z_max_metres",
                    ))
                    for model, count in sorted(models.items(), key=lambda v: (-v[1], v[0])):
                        writer.writerow((model, count, *extents.get(model, ("", "", "", ""))))

            # Image is a *density overview*, not a lossy substitute for the
            # placement CSV. Log scaling reveals sparse and crowded regions.
            intensity = np.zeros(density.shape, dtype=np.uint8)
            peak = int(density.max())
            if peak:
                intensity = np.rint(
                    255 * np.log1p(density.astype(np.float64)) / math.log1p(peak)
                ).astype(np.uint8)
            pixels = np.empty((image_height, image_width, 3), dtype=np.uint8)
            pixels[:, :, 0] = 15 + (intensity.astype(np.uint16) * 225 // 255).astype(np.uint8)
            pixels[:, :, 1] = 28 + (intensity.astype(np.uint16) * 186 // 255).astype(np.uint8)
            pixels[:, :, 2] = 24 + (intensity.astype(np.uint16) * 85 // 255).astype(np.uint8)
            stream = io.BytesIO()
            Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
            archive.writestr("object-density.png", stream.getvalue())
            archive.writestr("metadata.json", json.dumps({
                "schema": "island-studio-object-atlas-v1",
                "source": doc.source_label,
                "world_width_metres": world_width,
                "world_height_metres": world_height,
                "object_count": emitted,
                "distinct_models": len(models),
                "objects_outside_world_bounds": outside,
                "original_record_count": int(doc.object_count),
                "deleted_object_records_omitted": sum(
                    index in deleted for index in range(int(doc.object_count))
                ),
                "pending_object_edits_included": bool(
                    getattr(doc, "record_edits", {})
                    or getattr(doc, "added", ())
                    or deleted
                ),
                "orientation": "north-up: pixel (0,0) is NW, X east, Z north",
                "density_image_width": image_width,
                "density_image_height": image_height,
                "density_pixels_are_aggregated": True,
                "all_placements_are_in_csv": True,
            }, indent=2, ensure_ascii=False))
            archive.writestr("README.txt", (
                "Island Studio WRP object location export (4WVR / OPRW).\n"
                "object-placements.csv: EVERY active object with stable record index,\n"
                "object ID, exact model path, world X/Y/Z metres, heading (degrees),\n"
                "and horizontal transform basis values. Includes pending unsaved\n"
                "4WVR object edits, added records, and excludes deleted records.\n"
                "object-models.csv: model counts and world-coordinate extents.\n"
                "object-density.png: north-up aggregate object density overview.\n"
                "metadata.json: dimensions, counts, orientation and source.\n"
                "Pixel X=0 is west; pixel Y=0 is north. World X increases east,\n"
                "world Z increases north, world Y is elevation.\n"
                "Density pixels aggregate objects; CSV positions are exact.\n"
                "Original source WRP/PBO data are never changed.\n"
            ))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if progress is not None:
        progress(f"Exported {emitted:,} objects from {len(models):,} model types.")
    return destination
