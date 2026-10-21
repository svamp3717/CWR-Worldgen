# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, bounded RAP (config.bin) metadata reader for Island Studio.

Only CfgWorlds/<world>/LandGrid and Names are decoded. Other classes and
values are skipped without constructing an entire game configuration.
This supports the OFP/CWA and common RV RAP hierarchy, not arbitrary future
engine-specific config formats.
"""
from __future__ import annotations

import math
import struct

_MAGIC = b"\x00raP"
_MAX_CONFIG = 16 * 1024 * 1024
_MAX_ENTRIES = 100000
_MAX_DEPTH = 24
_MAX_STRING = 4096
_MAX_LOCATIONS = 5000


class _Rap:
    def __init__(self, data: bytes) -> None:
        if not 16 <= len(data) <= _MAX_CONFIG or data[:4] != _MAGIC:
            raise ValueError("not a supported RAP configuration")
        self.data = data
        self.pos = 16
        self.enum_offset = struct.unpack_from("<I", data, 12)[0]
        # Known RAP signature variants, including OFP/CWA style headers.
        # Do not assume an enum list is present.
        self._active: set[int] = set()

    def _take(self, count: int) -> bytes:
        end = self.pos + count
        if count < 0 or end > len(self.data):
            raise ValueError("truncated RAP value")
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def byte(self) -> int:
        return self._take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def cstring(self) -> str:
        end = self.data.find(b"\0", self.pos, min(len(self.data), self.pos + _MAX_STRING + 1))
        if end < 0:
            raise ValueError("unterminated or oversized RAP string")
        text = self.data[self.pos:end].decode("latin-1", "replace")
        self.pos = end + 1
        return text

    def count(self) -> int:
        # BIS compressed integers retain the high bit in the initial byte.
        value = self.byte()
        current = value
        shift = 7
        while current & 0x80:
            current = self.byte()
            value += (current - 1) << shift
            shift += 7
            if shift > 35:
                raise ValueError("invalid compressed RAP integer")
        if not 0 <= value <= _MAX_ENTRIES:
            raise ValueError("implausible RAP entry count")
        return value

    def scalar(self, subtype: int) -> str | float | int:
        if subtype in (0, 4):
            return self.cstring()
        if subtype == 1:
            return struct.unpack("<f", self._take(4))[0]
        if subtype == 2:
            return struct.unpack("<i", self._take(4))[0]
        if subtype == 6:
            return struct.unpack("<q", self._take(8))[0]
        raise ValueError("unknown RAP value type")

    def array(self, depth: int = 0) -> tuple[object, ...]:
        if depth > _MAX_DEPTH:
            raise ValueError("deeply nested RAP array")
        result: list[object] = []
        for _ in range(self.count()):
            subtype = self.byte()
            value = self.array(depth + 1) if subtype == 3 else self.scalar(subtype)
            result.append(value)
        return tuple(result)

    def body(self, offset: int) -> tuple[str, dict[str, object]]:
        if not 16 <= offset < len(self.data) or offset in self._active:
            raise ValueError("invalid cyclic RAP class offset")
        self._active.add(offset)
        try:
            self.pos = offset
            parent = self.cstring()
            values: dict[str, object] = {}
            for _ in range(self.count()):
                tag = self.byte()
                if tag == 0:
                    name = self.cstring()
                    values[name.casefold()] = ("class", self.u32())
                elif tag == 1:
                    subtype = self.byte()
                    name = self.cstring()
                    values[name.casefold()] = self.scalar(subtype)
                elif tag == 2:
                    name = self.cstring()
                    values[name.casefold()] = self.array()
                elif tag in (3, 4):
                    self.cstring()  # extern / delete class
                elif tag == 5:
                    # Array append flags precede the field name in RAP.
                    self.u32()
                    name = self.cstring()
                    values[name.casefold()] = self.array()
                else:
                    raise ValueError("unknown RAP class entry type")
            return parent, values
        finally:
            self._active.discard(offset)


class _OFP:
    """Original CWA RAP: version 4, indexed strings, contiguous class bodies.

    Reference: PMC Editing Wiki 'OFP raP File Format'. New-style RAP uses
    offsets instead and is handled by _Rap above.
    """

    def __init__(self, data: bytes) -> None:
        if not 8 <= len(data) <= _MAX_CONFIG or data[:7] != b"\x00raP\x04\x00\x00":
            raise ValueError("not an OFP/CWA RAP configuration")
        self.data = data
        self.pos = 7
        self.names: dict[int, str] = {}
        self.processed = 0

    def take(self, count: int) -> bytes:
        end = self.pos + count
        if count < 0 or end > len(self.data):
            raise ValueError("truncated OFP RAP value")
        result = self.data[self.pos:end]
        self.pos = end
        return result

    def byte(self) -> int:
        return self.take(1)[0]

    def count(self) -> int:
        result = self.byte()
        previous = result
        shift = 7
        while previous & 0x80:
            previous = self.byte()
            result += (previous - 1) << shift
            shift += 7
            if shift > 35:
                raise ValueError("invalid OFP RAP integer")
        if not 0 <= result <= _MAX_ENTRIES:
            raise ValueError("implausible OFP RAP packet count")
        return result

    def string(self) -> str:
        stop = self.data.find(b"\0", self.pos, min(len(self.data), self.pos + _MAX_STRING + 1))
        if stop < 0:
            raise ValueError("invalid OFP RAP string")
        value = self.data[self.pos:stop].decode("latin-1", "replace")
        self.pos = stop + 1
        return value

    def indexed(self) -> str:
        index = struct.unpack("<H", self.take(2))[0]
        value = self.string()
        if value:
            self.names[index] = value
            return value
        if index not in self.names:
            raise ValueError("undefined OFP RAP indexed string")
        return self.names[index]

    def scalar(self, tag: int) -> str | float | int:
        if tag == 0:
            return self.indexed()
        if tag == 1:
            return struct.unpack("<f", self.take(4))[0]
        if tag == 2:
            return struct.unpack("<i", self.take(4))[0]
        raise ValueError("unknown OFP RAP scalar type")

    def array(self, depth: int) -> tuple[object, ...]:
        if depth > _MAX_DEPTH:
            raise ValueError("deeply nested OFP RAP array")
        result: list[object] = []
        for _ in range(self.count()):
            tag = self.byte()
            result.append(self.array(depth + 1) if tag == 3 else self.scalar(tag))
        return tuple(result)

    def body(self, depth: int) -> dict[str, object]:
        if depth > _MAX_DEPTH:
            raise ValueError("deeply nested OFP RAP class")
        result: dict[str, object] = {}
        for _ in range(self.count()):
            self.processed += 1
            if self.processed > 200000:
                raise ValueError("too many OFP RAP entries")
            tag = self.byte()
            if tag == 0:
                name = self.indexed().casefold()
                base = self.string()
                children = self.body(depth + 1)
                result[name] = ("class", base, children)
            elif tag == 1:
                subtype = self.byte()
                name = self.indexed().casefold()
                result[name] = self.scalar(subtype)
            elif tag == 2:
                name = self.indexed().casefold()
                result[name] = self.array(depth + 1)
            else:
                raise ValueError("unsupported OFP RAP packet type")
        return result


class _CwaNative(_OFP):
    """Exact CWR-CE ParamFile v4 wire layout, with shared varint string pool."""

    def __init__(self, source: bytes) -> None:
        super().__init__(source)
        if source[:8] != b"\x00raP\x04\x00\x00\x00":
            raise ValueError("not CWA ParamFile binary v4")
        self.pos = 8

    def count(self) -> int:
        result = 0
        shift = 0
        for _ in range(5):
            byte = self.byte()
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                if result > _MAX_ENTRIES:
                    raise ValueError("CWA RAP entry count too large")
                return result
            shift += 7
        raise ValueError("invalid CWA RAP varint")

    def indexed(self) -> str:
        index = self.count()
        if index < len(self.names):
            return self.names[index]
        if index != len(self.names):
            raise ValueError("invalid CWA RAP string table index")
        name = self.string()
        self.names[index] = name
        return name


def _parse_ofp_world(
    source: bytes, world_name: str, *, legacy: bool = False,
) -> tuple[float | None, tuple[tuple[str, float, float], ...]] | None:
    rap = (
        _CwaNative(source)
        if not legacy and source[:8] == b"\x00raP\x04\x00\x00\x00"
        else _OFP(source)
    )
    if isinstance(rap, _CwaNative):
        # Real CWA root has name(index) + base(C-string) directly.
        rap.indexed()
        rap.string()
    elif rap.byte() != 0:
        raise ValueError("missing OFP RAP root class")
    if not isinstance(rap, _CwaNative):
        rap.indexed()  # legacy synthetic root class token
        rap.string()
    root = rap.body(0)
    cfg = root.get("cfgworlds")
    if not isinstance(cfg, tuple) or cfg[0] != "class":
        return None
    worlds = cfg[2]
    entry = worlds.get(world_name.casefold())
    if not isinstance(entry, tuple) or entry[0] != "class":
        return None
    all_fields: dict[str, object] = {}
    seen: set[str] = set()
    for _ in range(_MAX_DEPTH):
        if entry[1].casefold() in seen:
            raise ValueError("cyclic OFP RAP inheritance")
        all_fields = {**entry[2], **all_fields}
        parent = entry[1].casefold()
        if not parent:
            break
        seen.add(parent)
        entry = worlds.get(parent)
        if not isinstance(entry, tuple) or entry[0] != "class":
            break
    grid = all_fields.get("landgrid")
    try:
        land_grid = float(grid) if grid is not None else None
        if land_grid is not None and not (math.isfinite(land_grid) and 0 < land_grid <= 10000):
            land_grid = None
    except (ValueError, TypeError, OverflowError):
        land_grid = None
    names = all_fields.get("names")
    locations: list[tuple[str, float, float]] = []
    if isinstance(names, tuple) and names[0] == "class":
        for item in names[2].values():
            if not isinstance(item, tuple) or item[0] != "class":
                continue
            name = item[2].get("name")
            position = item[2].get("position")
            if not isinstance(name, str) or not 0 < len(name) <= 160:
                continue
            if not isinstance(position, tuple) or len(position) < 2:
                continue
            try:
                x, z = float(position[0]), float(position[1])
            except (ValueError, TypeError, OverflowError):
                continue
            if math.isfinite(x) and math.isfinite(z):
                locations.append((name, x, z))
            if len(locations) >= _MAX_LOCATIONS:
                break
    return land_grid, tuple(locations)


def parse_rap_world(
    source: bytes, world_name: str,
) -> tuple[float | None, tuple[tuple[str, float, float], ...]] | None:
    """Return LandGrid and labels, or None if no matching world class exists.

    A corrupt config is rejected by ValueError, not silently treated as text.
    """
    if source[:7] == b"\x00raP\x04\x00\x00":
        # The legacy synthetic index-u16 layout starts with an implicit
        # root-class token at byte 7, also zero. Try the real CWA v4 reader
        # first and fall back only for compatibility with those files.
        if source[:8] == b"\x00raP\x04\x00\x00\x00":
            try:
                actual = _parse_ofp_world(source, world_name)
                if actual is not None:
                    return actual
            except ValueError:
                pass
        return _parse_ofp_world(source, world_name, legacy=True)
    rap = _Rap(source)
    _base, root = rap.body(16)
    worlds = root.get("cfgworlds")
    if not isinstance(worlds, tuple) or worlds[0] != "class":
        return None
    _base, all_worlds = rap.body(worlds[1])
    world = all_worlds.get(world_name.casefold())
    if not isinstance(world, tuple) or world[0] != "class":
        return None

    # Config inheritance is local to CfgWorlds here. The actual engine can
    # merge classes across addons, which cannot be resolved from one PBO.
    def inherited(offset: int) -> dict[str, object]:
        chain: list[dict[str, object]] = []
        seen: set[int] = set()
        for _ in range(_MAX_DEPTH):
            if offset in seen:
                raise ValueError("cyclic RAP world inheritance")
            seen.add(offset)
            parent, fields = rap.body(offset)
            chain.append(fields)
            previous = all_worlds.get(parent.casefold())
            if not parent or not isinstance(previous, tuple) or previous[0] != "class":
                break
            offset = previous[1]
        combined: dict[str, object] = {}
        for item in reversed(chain):
            combined.update(item)
        return combined

    data = inherited(world[1])
    grid = data.get("landgrid")
    if isinstance(grid, (int, float, str)):
        try:
            grid = float(grid)
            if not math.isfinite(grid) or not 0 < grid <= 10000:
                grid = None
        except (ValueError, OverflowError):
            grid = None
    else:
        grid = None
    names = data.get("names")
    places: list[tuple[str, float, float]] = []
    if isinstance(names, tuple) and names[0] == "class":
        _parent, entries = rap.body(names[1])
        for value in entries.values():
            if not isinstance(value, tuple) or value[0] != "class":
                continue
            _parent, marker = rap.body(value[1])
            name = marker.get("name")
            position = marker.get("position")
            if isinstance(name, str) and name and len(name) <= 160 and isinstance(position, tuple) and len(position) >= 2:
                try:
                    x, z = float(position[0]), float(position[1])
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(x) and math.isfinite(z):
                    places.append((name, x, z))
            if len(places) >= _MAX_LOCATIONS:
                break
    return grid, tuple(places)
