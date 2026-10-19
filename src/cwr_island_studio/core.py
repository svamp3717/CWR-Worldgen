# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast, lossless RVW4 document access for the large-world island editor.

Terrain and original object records remain in a read-only memory map. Edits
are sparse and never mutate the source. Export streams untouched bytes and
patches only edited records, preserving unknown matrix components verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import mmap
import os
import shutil
import struct
import tempfile
from typing import Callable

import numpy as np

from .roads import is_road_model
from .object_layers import is_forest_model

_HEADER = struct.Struct("<4sii")
_OBJECT = struct.Struct("<12fi76s")
_TEXTURE_SLOTS = 512
_TEXTURE_BYTES = 32
_TABLE_BYTES = _TEXTURE_SLOTS * _TEXTURE_BYTES
_RECORD_DTYPE = np.dtype([
    ("transform", "<f4", (12,)),
    ("object_id", "<i4"),
    ("model", "S76"),
])
assert _RECORD_DTYPE.itemsize == _OBJECT.size == 128

# Avoid allocating a second object stream, including when editing million-
# object worlds. This controls the size of the sorting buffer, not map size.
_BUCKET_METRES = 256.0


@dataclass(frozen=True, slots=True)
class MapObject:
    index: int
    object_id: int
    model_path: str
    x: float
    y: float
    z: float
    heading: float
    # Horizontal components of the original WRP 3x3 transform matrix.
    # These preserve model scale, rotation, and local shear in map previews.
    basis_xx: float = 1.0
    basis_xz: float = 0.0
    basis_zx: float = 0.0
    basis_zz: float = 1.0


class IslandDocument:
    """An RVW4 document with cheap random access and sparse undoable edits.

    This reader deliberately supports unbinarized 4WVR maps. Other world
    formats require different parsers and are rejected rather than corrupted.
    The vertical terrain scale and horizontal grid spacing are user-supplied:
    neither is encoded in the 4WVR header.
    """

    def __init__(
        self, path: Path | str, *, cell_size: float = 50.0,
        height_scale: float = 1.0,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path).resolve()
        self.read_only = bool(read_only)
        self.source_label = self.path.name
        self._archive_owner = None
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell size must be positive and finite")
        if not math.isfinite(height_scale) or height_scale <= 0:
            raise ValueError("height scale must be positive and finite")
        self.cell_size = float(cell_size)
        self.height_scale = float(height_scale)
        self._file = self.path.open("rb")
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            if len(self._map) < _HEADER.size:
                raise ValueError("truncated RVW4 header")
            magic, self.width, self.height = _HEADER.unpack_from(self._map)
            if magic != b"4WVR":
                raise ValueError("only unbinarized 4WVR worlds are supported")
            if self.width <= 0 or self.height <= 0:
                raise ValueError("invalid world dimensions")
            self.cell_count = self.width * self.height
            self.texture_offset = _HEADER.size + self.cell_count * 2
            self.table_offset = self.texture_offset + self.cell_count * 2
            self.object_offset = self.table_offset + _TABLE_BYTES
            if self.object_offset > len(self._map):
                raise ValueError("truncated terrain or texture table")
            remainder = len(self._map) - self.object_offset
            if remainder % _OBJECT.size:
                raise ValueError("truncated RVW4 object record")
            record_count = remainder // _OBJECT.size
            self._raw_records = np.ndarray(
                (record_count,), dtype=_RECORD_DTYPE,
                buffer=self._map, offset=self.object_offset,
            )
            self.has_terminator = bool(
                record_count
                and not bytes(self._raw_records["model"][-1]).split(b"\0", 1)[0]
            )
            self.object_count = record_count - int(self.has_terminator)
            self._records = self._raw_records[:self.object_count]
            self.heights = np.ndarray(
                (self.height, self.width), dtype="<i2",
                buffer=self._map, offset=_HEADER.size,
            )
            self.textures = np.ndarray(
                (self.height, self.width), dtype="<i2",
                buffer=self._map, offset=self.texture_offset,
            )
            self.texture_paths = tuple(
                bytes(self._map[self.table_offset + i * _TEXTURE_BYTES:
                                self.table_offset + (i + 1) * _TEXTURE_BYTES])
                .split(b"\0", 1)[0].decode("ascii", errors="replace")
                for i in range(_TEXTURE_SLOTS)
            )
        except BaseException:
            if hasattr(self, "_map"):
                self._map.close()
            self._file.close()
            raise

        self.height_edits: dict[int, int] = {}
        self.texture_edits: dict[int, int] = {}
        self.record_edits: dict[int, bytes] = {}
        self.deleted: set[int] = set()
        self.added: list[bytes] = []
        self._index_order: np.ndarray | None = None
        self._index_keys: np.ndarray | None = None
        self._index_columns: int = 0
        self._road_mask: np.ndarray | None = None
        self._forest_mask: np.ndarray | None = None
        self._last_object_id: int | None = None

    @property
    def world_width(self) -> float:
        return self.width * self.cell_size

    @property
    def world_height(self) -> float:
        return self.height * self.cell_size

    @property
    def changed(self) -> bool:
        return bool(
            self.height_edits or self.texture_edits or self.record_edits
            or self.deleted or self.added
        )

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("Archived WRP is read-only; no editing or exporting is allowed")

    def close(self) -> None:
        # NumPy memory-map views must be dropped before closing the mapping.
        self._index_order = self._index_keys = self._road_mask = self._forest_mask = None
        self._records = self._raw_records = None
        self.heights = self.textures = None
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None
        if self._archive_owner is not None:
            owner, self._archive_owner = self._archive_owner, None
            owner.close()

    def __enter__(self) -> "IslandDocument":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _record(self, index: int) -> bytes:
        if not 0 <= index < self.object_count + len(self.added):
            raise IndexError(index)
        if index in self.deleted:
            raise ValueError("object was removed")
        if index >= self.object_count:
            return self.added[index - self.object_count]
        return self.record_edits.get(
            index,
            self._map[self.object_offset + index * _OBJECT.size:
                      self.object_offset + (index + 1) * _OBJECT.size],
        )

    def object_at(self, index: int) -> MapObject:
        record = self._record(index)
        values = _OBJECT.unpack(record)
        model = values[13].split(b"\0", 1)[0].decode("ascii", "replace")
        heading = math.degrees(math.atan2(-values[2], values[0])) % 360.0
        return MapObject(
            index, values[12], model,
            values[9], values[10], values[11], heading,
            values[0], values[6], values[2], values[8],
        )

    def move_object(
        self, index: int, *, x: float, y: float, z: float,
        heading: float | None = None,
    ) -> None:
        self._require_writable()
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError("object coordinates must be finite")
        old = self.object_at(index)
        raw = bytearray(self._record(index))
        struct.pack_into("<fff", raw, 9 * 4, x, y, z)
        if heading is not None:
            if not math.isfinite(heading):
                raise ValueError("object heading must be finite")
            delta = math.radians((float(heading) - old.heading) % 360.0)
            if abs(delta) > 1.0e-9 and abs(delta - math.tau) > 1.0e-9:
                sine, cosine = math.sin(delta), math.cos(delta)
                # Rotate basis vectors around the world up axis. Preserve
                # all Y components, including terrain shear and object tilt.
                for start in (0, 3, 6):
                    # Unpack the original horizontal components before
                    # replacing them; preserve the vertical components.
                    vx = struct.unpack_from("<f", raw, start * 4)[0]
                    vz = struct.unpack_from("<f", raw, (start + 2) * 4)[0]
                    struct.pack_into("<f", raw, start * 4, cosine * vx + sine * vz)
                    struct.pack_into("<f", raw, (start + 2) * 4, -sine * vx + cosine * vz)
        if index >= self.object_count:
            self.added[index - self.object_count] = bytes(raw)
        else:
            self.record_edits[index] = bytes(raw)

    def set_model(self, index: int, model_path: str) -> None:
        self._require_writable()
        data = model_path.encode("ascii", errors="strict")
        if not data or len(data) > 75 or b"\0" in data:
            raise ValueError("model path must be 1 to 75 ASCII bytes")
        raw = bytearray(self._record(index))
        raw[52:128] = data.ljust(76, b"\0")
        if index >= self.object_count:
            self.added[index - self.object_count] = bytes(raw)
        else:
            self.record_edits[index] = bytes(raw)

    def add_object(
        self, model_path: str, *, x: float, y: float, z: float,
        heading: float = 0.0,
    ) -> int:
        self._require_writable()
        encoded = model_path.encode("ascii", "strict")
        if not encoded or len(encoded) > 75 or b"\0" in encoded:
            raise ValueError("model path must be 1 to 75 ASCII bytes")
        if not all(math.isfinite(v) for v in (x, y, z, heading)):
            raise ValueError("object coordinates and heading must be finite")
        if self._last_object_id is None:
            self._last_object_id = int(self._records["object_id"].max()) if self.object_count else 0
        self._last_object_id += 1
        if self._last_object_id > 2_147_483_647:
            raise ValueError("no object IDs available")
        angle = math.radians(heading)
        c, s = math.cos(angle), math.sin(angle)
        raw = _OBJECT.pack(
            c, 0.0, -s, 0.0, 1.0, 0.0,
            s, 0.0, c, x, y, z, self._last_object_id,
            encoded.ljust(76, b"\0"),
        )
        index = self.object_count + len(self.added)
        self.added.append(raw)
        return index

    def delete_object(self, index: int) -> None:
        self._require_writable()
        self._record(index)
        self.deleted.add(index)
        self.record_edits.pop(index, None)

    def terrain_at(self, column: int, row: int) -> float:
        if not (0 <= column < self.width and 0 <= row < self.height):
            raise IndexError((column, row))
        index = row * self.width + column
        return self.height_edits.get(index, int(self.heights[row, column])) * self.height_scale

    def texture_at(self, column: int, row: int) -> int:
        if not (0 <= column < self.width and 0 <= row < self.height):
            raise IndexError((column, row))
        index = row * self.width + column
        return self.texture_edits.get(index, int(self.textures[row, column]))

    def paint(
        self, x: float, z: float, *, radius: float,
        raise_metres: float = 0.0, texture: int | None = None,
    ) -> int:
        self._require_writable()
        if radius < 0 or not math.isfinite(radius) or not math.isfinite(raise_metres):
            raise ValueError("invalid terrain brush")
        if texture is not None and not 0 <= texture < _TEXTURE_SLOTS:
            raise ValueError("texture slot must be within 0..511")
        cx, cz = x / self.cell_size, z / self.cell_size
        step = radius / self.cell_size
        left = max(0, math.ceil(cx - step))
        right = min(self.width - 1, math.floor(cx + step))
        top = max(0, math.ceil(cz - step))
        bottom = min(self.height - 1, math.floor(cz + step))
        changed = 0
        # Preserve the precise input grid, which is important for round trips.
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                if (col - cx) ** 2 + (row - cz) ** 2 > step ** 2 + 1e-9:
                    continue
                key = row * self.width + col
                if texture is not None:
                    self.texture_edits[key] = texture
                if raise_metres:
                    before = self.height_edits.get(key, int(self.heights[row, col]))
                    after = before + int(round(raise_metres / self.height_scale))
                    if not -32768 <= after <= 32767:
                        raise ValueError("brush exceeds RVW4 elevation limits")
                    self.height_edits[key] = after
                changed += 1
        return changed

    def build_object_index(self) -> None:
        """Build a compact, vectorized spatial index without decoding models.

        This can run on a worker thread after the terrain preview is visible.
        """
        cols = max(1, math.ceil(self.world_width / _BUCKET_METRES))
        transforms = self._records["transform"]
        x = transforms[:, 9]
        z = transforms[:, 11]
        valid = np.isfinite(x) & np.isfinite(z)
        ix = np.maximum(0, np.floor(x / _BUCKET_METRES)).astype(np.int64)
        iz = np.maximum(0, np.floor(z / _BUCKET_METRES)).astype(np.int64)
        keys = np.where(valid, iz * cols + ix, np.iinfo(np.int64).max)
        order = np.argsort(keys, kind="stable")
        self._index_keys = keys[order]
        self._index_order = order.astype(np.int32 if self.object_count < 2_147_483_647 else np.int64)
        self._index_columns = cols
        # Decoding object paths is deferred until background indexing. Keep
        # only one boolean per original object, not Python road objects.
        self._road_mask = np.fromiter(
            (is_road_model(value) for value in self._records["model"]),
            dtype=np.bool_, count=self.object_count,
        )
        # Classify just once in the background rather than on every pan.
        self._forest_mask = np.fromiter(
            (is_forest_model(value) for value in self._records["model"]),
            dtype=np.bool_, count=self.object_count,
        )

    def visible_objects(
        self, left: float, top: float, right: float, bottom: float,
        *, limit: int = 3500, roads_only: bool = False,
        forest_only: bool = False,
    ) -> tuple[MapObject, ...]:
        if roads_only and forest_only:
            raise ValueError("choose either roads_only or forest_only")
        if limit <= 0:
            return ()
        transforms = self._records["transform"]
        # Wide overview views can span thousands of buckets. A vectorized
        # full scan is cheaper than issuing tens of thousands of searches.
        bucket_span = (
            max(0, math.floor(bottom / _BUCKET_METRES) - max(0, math.floor(top / _BUCKET_METRES)) + 1)
            * max(0, math.floor(right / _BUCKET_METRES) - max(0, math.floor(left / _BUCKET_METRES)) + 1)
        )
        if self._index_order is None or self._index_keys is None or bucket_span > 4096:
            # Before the background index finishes, avoid a million-object
            # Python loop. This masked scan is vectorized and has bounded output.
            hits = np.flatnonzero(
                (transforms[:, 9] >= left) & (transforms[:, 9] <= right)
                & (transforms[:, 11] >= top) & (transforms[:, 11] <= bottom)
            )
        else:
            ids: list[np.ndarray] = []
            for row in range(max(0, math.floor(top / _BUCKET_METRES)),
                             max(0, math.floor(bottom / _BUCKET_METRES)) + 1):
                for col in range(max(0, math.floor(left / _BUCKET_METRES)),
                                 max(0, math.floor(right / _BUCKET_METRES)) + 1):
                    key = row * self._index_columns + col
                    first = int(np.searchsorted(self._index_keys, key, side="left"))
                    last = int(np.searchsorted(self._index_keys, key, side="right"))
                    if last > first:
                        ids.append(self._index_order[first:last])
            hits = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
            if hits.size:
                hits = hits[
                    (transforms[hits, 9] >= left) & (transforms[hits, 9] <= right)
                    & (transforms[hits, 11] >= top) & (transforms[hits, 11] <= bottom)
                ]
        if roads_only and self._road_mask is not None and len(hits):
            hits = hits[self._road_mask[hits]]
        if forest_only and self._forest_mask is not None and len(hits):
            hits = hits[self._forest_mask[hits]]
        # Avoid rendering tens of thousands of Tk objects per frame.
        if len(hits) > limit:
            hits = hits[::max(1, math.ceil(len(hits) / limit))][:limit]
        result: list[MapObject] = []
        overrides = set(self.record_edits)
        for index in hits:
            i = int(index)
            if i not in self.deleted and i not in overrides:
                obj = self.object_at(i)
                if (not roads_only or is_road_model(obj.model_path)) and (
                    not forest_only or is_forest_model(obj.model_path)
                ):
                    result.append(obj)
        for i in (*sorted(overrides), *range(self.object_count, self.object_count + len(self.added))):
            if i in self.deleted:
                continue
            obj = self.object_at(i)
            if left <= obj.x <= right and top <= obj.z <= bottom:
                if (not roads_only or is_road_model(obj.model_path)) and (
                    not forest_only or is_forest_model(obj.model_path)
                ):
                    result.append(obj)
        return tuple(result[:limit])

    def sample_viewport(
        self, left: float, top: float, right: float, bottom: float,
        width: int, height: int, *, textures: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample only current viewport pixels, applying sparse edits."""
        width = max(1, min(1600, int(width)))
        height = max(1, min(1200, int(height)))
        world_x = np.linspace(left, right, width, endpoint=False)
        world_z = np.linspace(top, bottom, height, endpoint=False)
        columns = np.clip((world_x / self.cell_size).astype(np.int64), 0, self.width - 1)
        rows = np.clip((world_z / self.cell_size).astype(np.int64), 0, self.height - 1)
        source = self.textures if textures else self.heights
        sampled = source[np.ix_(rows, columns)].astype(np.int32)
        patches = self.texture_edits if textures else self.height_edits
        if patches:
            keys = np.fromiter(patches, dtype=np.int64)
            values = np.fromiter(patches.values(), dtype=np.int32)
            order = np.argsort(keys)
            keys, values = keys[order], values[order]
            pixels = (rows[:, None] * self.width + columns[None, :]).ravel()
            locations = np.searchsorted(keys, pixels)
            present = locations < len(keys)
            present[present] = keys[locations[present]] == pixels[present]
            sampled.ravel()[present] = values[locations[present]]
        return sampled, world_x, world_z

    def viewport(
        self, left: float, top: float, right: float, bottom: float,
        width: int, height: int, *, textures: bool = False,
        water_level: float = 0.0,
    ) -> np.ndarray:
        """Render height-shaded terrain (with water) or texture-slot colours."""
        sampled, _, _ = self.sample_viewport(
            left, top, right, bottom, width, height, textures=textures,
        )
        if textures:
            ids = sampled.astype(np.uint32)
            return np.stack((
                (52 + (ids * 57 + 32) % 150).astype(np.uint8),
                (72 + (ids * 41 + 77) % 135).astype(np.uint8),
                (48 + (ids * 71 + 11) % 140).astype(np.uint8),
            ), axis=-1)
        if not math.isfinite(water_level):
            raise ValueError("water level must be finite")
        terrain = sampled.astype(np.float32) * self.height_scale
        slope_x = np.gradient(terrain, axis=1) if terrain.shape[1] > 1 else np.zeros_like(terrain)
        slope_z = np.gradient(terrain, axis=0) if terrain.shape[0] > 1 else np.zeros_like(terrain)
        shadow = np.clip(145 + (slope_x - slope_z) * 1.25, 45, 230)
        green = np.clip(105 + terrain * 0.18 + shadow * 0.37, 32, 235)
        rgb = np.stack((
            np.clip(green * 0.64, 0, 255).astype(np.uint8),
            np.clip(green * 1.04, 0, 255).astype(np.uint8),
            np.clip(green * 0.67, 0, 255).astype(np.uint8),
        ), axis=-1)
        water = terrain <= water_level
        if np.any(water):
            depth = np.clip(water_level - terrain, 0.0, 100.0)
            rgb[water, 0] = (27 + 0.13 * depth[water]).astype(np.uint8)
            rgb[water, 1] = (85 + 0.13 * depth[water]).astype(np.uint8)
            rgb[water, 2] = (147 + 0.33 * depth[water]).astype(np.uint8)
        return rgb

    def export(self, destination: Path | str, progress: Callable[[str], None] | None = None) -> Path:
        """Write a new file atomically, retaining original object transforms.

        Most edits use a fast copy-and-patch path. Deletions use sequential
        copying of surviving record spans, not per-object Python packing.
        """
        self._require_writable()
        destination = Path(destination).resolve()
        if destination == self.path:
            raise ValueError("the source map is read-only; use a different output filename")
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w+b", dir=destination.parent, prefix="." + destination.name + ".",
                suffix=".tmp", delete=False,
            ) as target:
                tmp_name = target.name
                if progress:
                    progress("Copying original map data")
                with self.path.open("rb") as source:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                for index, value in self.height_edits.items():
                    target.seek(_HEADER.size + index * 2)
                    target.write(struct.pack("<h", value))
                for index, value in self.texture_edits.items():
                    target.seek(self.texture_offset + index * 2)
                    target.write(struct.pack("<h", value))
                if not self.deleted:
                    for index, value in self.record_edits.items():
                        target.seek(self.object_offset + index * _OBJECT.size)
                        target.write(value)
                    if self.added:
                        target.seek(
                            -_OBJECT.size if self.has_terminator else 0,
                            os.SEEK_END,
                        )
                        for entry in self.added:
                            target.write(entry)
                        target.write(bytes(_OBJECT.size))
                else:
                    if progress:
                        progress("Compacting removed objects")
                    target.truncate(self.object_offset)
                    target.seek(self.object_offset)
                    kept = 0
                    for i in range(self.object_count):
                        if i in self.deleted:
                            continue
                        record = self.record_edits.get(i)
                        if record is None:
                            offset = self.object_offset + i * _OBJECT.size
                            record = self._map[offset:offset + _OBJECT.size]
                        target.write(record)
                        kept += 1
                    for i, record in enumerate(self.added, self.object_count):
                        if i not in self.deleted:
                            target.write(record)
                    target.write(bytes(_OBJECT.size))
                    target.truncate()
                target.flush()
                os.fsync(target.fileno())
            os.replace(tmp_name, destination)
            tmp_name = None
            return destination
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
