# SPDX-License-Identifier: GPL-3.0-or-later
"""Companion debug layers for Worldgen's WRP outputs.

This sidecar is deliberately outside the game PBO: it is for Island Studio,
not engine consumption. All rasters are WRP south-to-north row order.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# Layer names and stable RGB palette are shared by generator and viewer.
LAYERS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "osm_water": ("OSM water", (36, 134, 238)),
    "osm_forest": ("OSM forest", (38, 188, 77)),
    "osm_roads": ("OSM roads", (249, 160, 56)),
    "osm_farmland": ("OSM farmland", (234, 211, 94)),
    "osm_urban": ("OSM urban", (216, 103, 155)),
    "osm_buildings": ("OSM buildings", (247, 92, 91)),
    "semantic_base": ("Base terrain materials", (255, 255, 255)),
    "semantic_final": ("Final terrain materials", (255, 255, 255)),
    "wrp_slots": ("Final WRP texture slots", (255, 255, 255)),
    "vegetation_placements": ("Placed forest objects", (104, 243, 129)),
}
BOOL_LAYERS = frozenset(k for k in LAYERS if k.startswith("osm_"))
SEMANTIC_LAYERS = frozenset({"semantic_base", "semantic_final"})
MAX_DEBUG_CELLS = 16_777_216


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _square(values: Sequence, cells: int, *, dtype) -> np.ndarray:
    if len(values) != cells * cells:
        raise ValueError("debug layer does not match terrain dimensions")
    return np.asarray(values, dtype=dtype).reshape(cells, cells)


def write_worldgen_debug_layers(
    destination: Path, *, wrp_path: Path, world_name: str,
    cells: int, cell_size: float, raster: object,
    base_material: Sequence[int], final_material: Sequence[int],
    wrp_slots: Sequence[int], materials: Sequence[object],
    forest_objects: Sequence[object],
) -> Path:
    """Write an opt-in compressed diagnostic file, never into the game PBO."""
    if not (0 < cells * cells <= MAX_DEBUG_CELLS and cell_size > 0):
        raise ValueError("unsupported debug layer world size")
    layers: dict[str, np.ndarray] = {}
    for name in BOOL_LAYERS:
        source = getattr(raster, name.removeprefix("osm_"))
        layers[name] = _square(source, cells, dtype=np.uint8)
    for name, source in (
        ("semantic_base", base_material),
        ("semantic_final", final_material),
        ("wrp_slots", wrp_slots),
    ):
        layers[name] = _square(source, cells, dtype=np.uint16)
    placements = np.zeros((cells, cells), dtype=np.uint16)
    for obj in forest_objects:
        x, z = float(obj.x), float(obj.z)
        if 0 <= x < cells * cell_size and 0 <= z < cells * cell_size:
            col, row = int(x / cell_size), int(z / cell_size)
            if placements[row, col] != 65535:
                placements[row, col] += 1
    layers["vegetation_placements"] = placements
    manifest = {
        "format": "cwr-worldgen-debug-layers-v1",
        "world_name": world_name,
        "width": cells, "height": cells,
        "cell_size": cell_size,
        "wrp_sha256": file_sha256(wrp_path),
        "materials": [
            {"code": str(mat.code), "name": str(mat.name),
             "colour": [int(c) for c in mat.colour]}
            for mat in materials
        ],
        "layer_names": list(LAYERS),
        "row_order": "south-to-north",
        "description": "Original OSM raster masks and actual Worldgen stage outputs; not reconstructed from the final WRP.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, metadata=np.frombuffer(
                    json.dumps(manifest, sort_keys=True).encode("utf-8"),
                    dtype=np.uint8,
                ), **layers,
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@dataclass
class WorldgenDebugLayers:
    path: Path
    metadata: dict
    rasters: dict[str, np.ndarray]

    @property
    def choices(self) -> tuple[str, ...]:
        return tuple(LAYERS[k][0] for k in LAYERS if k in self.rasters)

    def name_key(self, display_name: str) -> str | None:
        return next(
            (key for key, (label, _) in LAYERS.items() if label == display_name
             and key in self.rasters), None,
        )

    def sample(self, key: str, x: float, z: float) -> int | None:
        if key not in self.rasters:
            return None
        col = int(x / self.metadata["cell_size"])
        row = int(z / self.metadata["cell_size"])
        if not (0 <= col < self.metadata["width"]
                and 0 <= row < self.metadata["height"]):
            return None
        return int(self.rasters[key][row, col])

    def composite(
        self, background: np.ndarray, key: str, *,
        left: float, top: float, right: float, bottom: float,
        material_colours: Sequence[Sequence[int]] = (),
    ) -> np.ndarray:
        """Blend a sampled debug raster, using geographic north-up sampling."""
        if key not in self.rasters:
            raise ValueError("unknown debug layer")
        h, w = background.shape[:2]
        cell = float(self.metadata["cell_size"])
        x = np.linspace(left, right, w, endpoint=False)
        z = np.linspace(top, bottom, h, endpoint=False)
        if top > bottom:
            z -= (top - bottom) / (2 * h)
        cols = np.clip((x / cell).astype(np.intp), 0, self.metadata["width"] - 1)
        rows = np.clip((z / cell).astype(np.intp), 0, self.metadata["height"] - 1)
        sampled = self.rasters[key][np.ix_(rows, cols)]
        if key in BOOL_LAYERS or key == "vegetation_placements":
            present = sampled != 0
            if not np.any(present):
                return background
            tint = np.asarray(LAYERS[key][1], dtype=np.float32)
            result = background.copy()
            original = result[present].astype(np.float32)
            result[present] = np.rint(original * .28 + tint * .72).astype(np.uint8)
            return result
        if key in SEMANTIC_LAYERS:
            palette = np.asarray(self.metadata["materials_colours"], dtype=np.uint8)
            colour = palette[np.clip(sampled, 0, len(palette) - 1)]
        else:
            # Stable per-slot colour, not dependent on the image assets.
            slot = sampled.astype(np.uint32)
            colour = np.stack((
                (47 + (slot * 53) % 199).astype(np.uint8),
                (36 + (slot * 89) % 201).astype(np.uint8),
                (38 + (slot * 131) % 195).astype(np.uint8),
            ), axis=-1)
        return np.rint(background.astype(np.float32) * .20
                       + colour.astype(np.float32) * .80).astype(np.uint8)


def load_worldgen_debug_layers(
    path: Path, *, world_width: float, world_height: float,
    wrp_path: Path | None = None,
) -> WorldgenDebugLayers:
    """Load a strict, dimension-matched, non-pickled sidecar."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        raw = data["metadata"]
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.size > 262144:
            raise ValueError("invalid debug metadata")
        metadata = json.loads(raw.tobytes().decode("utf-8"))
        if metadata.get("format") != "cwr-worldgen-debug-layers-v1":
            raise ValueError("unsupported Worldgen debug data format")
        width, height, cell = (int(metadata["width"]), int(metadata["height"]),
                               float(metadata["cell_size"]))
        if (width < 1 or height < 1 or width * height > MAX_DEBUG_CELLS
                or cell <= 0 or not np.isfinite(cell)):
            raise ValueError("invalid debug world dimensions")
        if (abs(width * cell - world_width) > .001
                or abs(height * cell - world_height) > .001):
            raise ValueError("debug layers belong to a different world size")
        if wrp_path is not None and file_sha256(Path(wrp_path)) != metadata["wrp_sha256"]:
            raise ValueError("debug layers do not match this WRP's content")
        materials = metadata.get("materials", [])
        colours = [entry["colour"] for entry in materials]
        if not colours:
            raise ValueError("missing debug materials")
        metadata["materials_colours"] = colours
        rasters = {}
        for key in LAYERS:
            if key not in data:
                raise ValueError(f"missing debug raster {key}")
            array = data[key]
            if array.shape != (height, width) or array.dtype.kind not in "uib":
                raise ValueError(f"invalid debug raster {key}")
            rasters[key] = array.copy()
    return WorldgenDebugLayers(path, metadata, rasters)


def find_worldgen_debug_layers(source_path: Path, world_name: str) -> Path | None:
    """Look for a generated sidecar beside either a source WRP or world PBO."""
    source_path = Path(source_path)
    filename = f"{world_name}-worldgen-debug.npz"
    for parent in (source_path.parent, *list(source_path.parents)[:3]):
        candidate = parent / filename
        if candidate.is_file():
            return candidate
    return None
