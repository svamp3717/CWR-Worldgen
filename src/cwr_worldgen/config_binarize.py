# SPDX-License-Identifier: GPL-3.0-or-later
"""CWA/Resistance-compatible RAP v4 config.cpp binarization.

Implements Poseidon ParamFile::SerializeBin's *contiguous* class layout and
indexed string pool, not the later Arma offset-based RAP dialect. The parser
is intentionally strict: it rejects unsupported config expressions rather
than silently shipping an incomplete game configuration.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

_MAX_CONFIG = 16 * 1024 * 1024
_MAX_DEPTH = 64
_MAX_ENTRIES = 100000
_TOKEN = re.compile(
    r'\s+|//[^\n]*|/\*[\s\S]*?\*/|"(?:[^"]|"")*"'
    r'|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
    r'|[A-Za-z_][A-Za-z_0-9]*|[{}[\];:=,]'
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?\Z")


class ConfigClass:
    def __init__(self, name: str, base: str = "", entries: list | None = None):
        self.name = name
        self.base = base
        self.entries = [] if entries is None else entries


class _Parser:
    def __init__(self, source: str):
        if len(source.encode("utf-8")) > _MAX_CONFIG:
            raise ValueError("config.cpp is too large to binarize")
        self.tokens: list[str] = []
        offset = 0
        while offset < len(source):
            match = _TOKEN.match(source, offset)
            if match is None:
                raise ValueError(
                    f"unsupported config.cpp syntax near offset {offset}: "
                    f"{source[offset:offset+55]!r}"
                )
            token = match.group()
            if not token.isspace() and not token.startswith(("//", "/*")):
                self.tokens.append(token)
            offset = match.end()
        self.pos = 0

    def _peek(self) -> str:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def _take(self) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError("unexpected end of config.cpp")
        value = self.tokens[self.pos]
        self.pos += 1
        return value

    def _expect(self, value: str) -> None:
        actual = self._take()
        if actual != value:
            raise ValueError(f"expected {value!r} in config.cpp, got {actual!r}")

    def _ident(self) -> str:
        value = self._take()
        if _IDENT.fullmatch(value) is None:
            raise ValueError(f"expected config identifier, got {value!r}")
        return value

    def _scalar(self) -> str | float | int:
        token = self._take()
        if token.startswith('"') and token.endswith('"'):
            return token[1:-1].replace('""', '"')
        if _NUMBER.fullmatch(token):
            if any(char in token.casefold() for char in ".e"):
                return float(token)
            return int(token)
        raise ValueError(f"unsupported config value {token!r}")

    def _array(self, depth: int) -> list:
        if depth > _MAX_DEPTH:
            raise ValueError("config array nesting is too deep")
        self._expect("{")
        elements: list = []
        if self._peek() != "}":
            while True:
                elements.append(
                    self._array(depth + 1) if self._peek() == "{"
                    else self._scalar()
                )
                if self._peek() == "}":
                    break
                self._expect(",")
                if self._peek() == "}":
                    break
        self._expect("}")
        return elements

    def _statements(self, depth: int, *, nested: bool = False) -> list:
        if depth > _MAX_DEPTH:
            raise ValueError("config class nesting is too deep")
        entries: list[ConfigClass | tuple[str, str, object]] = []
        names: set[str] = set()
        while self._peek() and self._peek() != "}":
            if self._peek().casefold() == "class":
                self._take()
                name = self._ident()
                base = ""
                if self._peek() == ":":
                    self._take()
                    base = self._ident()
                self._expect("{")
                children = self._statements(depth + 1, nested=True)
                self._expect("}")
                self._expect(";")
                entry: ConfigClass | tuple[str, str, object] = ConfigClass(name, base, children)
            else:
                name = self._ident()
                is_array = False
                if self._peek() == "[":
                    self._take()
                    self._expect("]")
                    is_array = True
                self._expect("=")
                if is_array:
                    value = self._array(depth + 1)
                else:
                    value = self._scalar()
                self._expect(";")
                entry = ("array" if is_array else "value", name, value)
            key = (entry.name if isinstance(entry, ConfigClass) else entry[1]).casefold()
            if key in names:
                raise ValueError(f"duplicate config member {key!r}")
            names.add(key)
            entries.append(entry)
            if len(entries) > _MAX_ENTRIES:
                raise ValueError("too many config members")
        if nested and not self._peek():
            raise ValueError("unterminated config class")
        return entries

    def parse(self) -> list:
        entries = self._statements(0)
        if self._peek():
            raise ValueError("unexpected trailing config.cpp tokens")
        return entries


def parse_cpp_config(source: str) -> list:
    """Parse the subset emitted by Worldgen; reject unsupported constructs."""
    return _Parser(source).parse()


class _CwaEncoder:
    def __init__(self):
        self.data = bytearray(b"\x00raP\x04\x00\x00\x00")
        self.pool: dict[str, int] = {}

    def _byte(self, value: int) -> None:
        self.data.append(value)

    def _cstring(self, value: str) -> None:
        if "\0" in value:
            raise ValueError("NUL in config string")
        self.data.extend(value.encode("utf-8") + b"\0")

    def _int(self, value: int) -> None:
        self.data.extend(struct.pack("<i", value))

    def _varint(self, value: int) -> None:
        if not 0 <= value <= 0x7FFFFFFF:
            raise ValueError("config index out of range")
        while value >= 128:
            self._byte((value & 127) | 128)
            value >>= 7
        self._byte(value)

    def _indexed(self, value: str) -> None:
        if value in self.pool:
            self._varint(self.pool[value])
        else:
            self._varint(len(self.pool))
            self.pool[value] = len(self.pool)
            self._cstring(value)

    def _value(self, item: str | int | float) -> None:
        if isinstance(item, str):
            self._byte(0)
            self._indexed(item)  # strings are interned in CWA RAP v4
        elif isinstance(item, int):
            self._byte(2)
            self._int(item)
        elif isinstance(item, float):
            self._byte(1)
            self.data.extend(struct.pack("<f", item))
        else:
            raise ValueError(f"unsupported config value: {item!r}")

    def _array(self, elements: list, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise ValueError("config array nesting too deep")
        self._varint(len(elements))
        for element in elements:
            if isinstance(element, list):
                self._byte(3)  # SVArray nested value
                self._array(element, depth + 1)
            else:
                self._value(element)

    def _class(self, name: str, base: str, entries: list, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise ValueError("config class nesting too deep")
        self._indexed(name)
        self._cstring(base)
        self._varint(len(entries))
        defined: set[str] = set()
        for entry in entries:
            if isinstance(entry, ConfigClass):
                self._byte(0)
                # A base must be visible within the parent scope as CWR-CE
                # resolves it at read time. Reject rather than creating a
                # bin that the CWA engine cannot load.
                if entry.base and entry.base.casefold() not in defined:
                    raise ValueError(
                        f"class {entry.name} inherits {entry.base}, "
                        "but the base class is not defined earlier in its parent scope"
                    )
                self._class(entry.name, entry.base, entry.entries, depth + 1)
                defined.add(entry.name.casefold())
            else:
                kind, key, item = entry
                if kind == "value":
                    self._byte(1)
                    self._value_type_and_name(key, item)
                else:
                    self._byte(2)
                    self._indexed(key)
                    self._array(item, depth + 1)

    def _value_type_and_name(self, key: str, item: object) -> None:
        # ParamValueSpec::SerializeBin writes type *before* property name.
        if isinstance(item, str):
            self._byte(0)
            self._indexed(key)
            self._indexed(item)
        elif isinstance(item, int):
            self._byte(2)
            self._indexed(key)
            self._int(item)
        elif isinstance(item, float):
            self._byte(1)
            self._indexed(key)
            self.data.extend(struct.pack("<f", item))
        else:
            raise ValueError(f"unsupported config property {key}")

    def encode(self, entries: list) -> bytes:
        self._class("", "", entries, 0)
        if len(self.data) > _MAX_CONFIG:
            raise ValueError("binarized config is too large")
        return bytes(self.data)


def binarize_config_text(source: str) -> bytes:
    """Convert a generated config.cpp to CWA RAP v4; never mutate input."""
    return _CwaEncoder().encode(parse_cpp_config(source))


def binarize_config(source: Path, destination: Path) -> Path:
    """Write and validate a standalone config.bin without touching config.cpp."""
    from cwr_island_studio.rap_config import parse_rap_world

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("config.bin must not replace config.cpp")
    original = source.read_text(encoding="utf-8")
    binary = binarize_config_text(original)
    # Require that the binary really loads and preserves CfgWorlds landmarks.
    # This also verifies compatibility with Island Studio's RAP reader.
    tree = parse_cpp_config(original)
    cfgworlds = next(
        (entry for entry in tree
         if isinstance(entry, ConfigClass) and entry.name.casefold() == "cfgworlds"),
        None,
    )
    if cfgworlds is None:
        raise ValueError("config.cpp has no CfgWorlds class")
    # Verify *every* world with a worldName assignment, even if the config
    # also defines Land_* classes under CfgVehicles.
    for item in cfgworlds.entries:
        if not isinstance(item, ConfigClass):
            continue
        fields = {
            entry[1].casefold(): entry[2] for entry in item.entries
            if isinstance(entry, tuple)
        }
        if "worldname" not in fields:
            continue
        try:
            metadata = parse_rap_world(binary, item.name)
        except ValueError as exc:
            raise ValueError("RAP output could not be decoded after binarization") from exc
        if metadata is None:
            raise ValueError(f"RAP output is missing world {item.name}")
        grid = fields.get("landgrid")
        if grid is not None and (metadata[0] is None or
                                 abs(float(grid) - metadata[0]) > 0.0001):
            raise ValueError(f"RAP output changed LandGrid for {item.name}")
        names = next(
            (entry for entry in item.entries
             if isinstance(entry, ConfigClass) and entry.name.casefold() == "names"),
            None,
        )
        if names is not None and len(metadata[1]) != len(names.entries):
            raise ValueError(f"RAP output lost location names for {item.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".tmp", delete=False,
        ) as out:
            tmp_path = out.name
            out.write(binary)
        os.replace(tmp_path, destination)
        tmp_path = None
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)
    return destination


def pack_binarized_runtime_pbo(
    source_dir: Path, source_config: Path, binary: Path, pbo_path: Path,
    *, binarized_wrp: Path | None = None,
    backend: str = "auto", poseidon_tools_path: Path | None = None,
) -> None:
    """Atomically pack config.bin and optional OPRW with editable sources intact."""
    from .pbo import pack_directory
    source_dir = Path(source_dir)
    source_config = Path(source_config)
    pbo_path = Path(pbo_path)
    if source_config.parent.resolve() != source_dir.resolve():
        raise ValueError("config.cpp must live inside the source archive directory")
    with Path(binary).open("rb") as reader:
        if reader.read(8) != b"\x00raP\x04\x00\x00\x00":
            raise ValueError("expected CWA RAP v4 config.bin")
    if binarized_wrp is not None:
        binarized_wrp = Path(binarized_wrp)
        with binarized_wrp.open("rb") as reader:
            if reader.read(8) != b"OPRW\x03\x00\x00\x00":
                raise ValueError("expected OPRW v3 WRP")
    source_wrp = (
        source_dir / binarized_wrp.name.replace(".binarized.wrp", ".wrp")
        if binarized_wrp is not None else None
    )
    if binarized_wrp and (source_wrp is None or not source_wrp.is_file()):
        raise ValueError("generated 4WVR source WRP is missing")
    if (source_dir / "config.bin").exists():
        raise ValueError("config.bin already exists in the source directory; refusing to overwrite it")
    with tempfile.TemporaryDirectory(dir=source_dir.parent, prefix=".cfg-binarize-") as scratch:
        work = Path(scratch)
        saved_config = work / "editable-config.cpp"
        pending_pbo = work / pbo_path.name
        saved_wrp = work / "editable-world.wrp"
        os.replace(source_config, saved_config)
        wrp_staged = False
        try:
            shutil.copyfile(binary, source_dir / "config.bin")
            if binarized_wrp:
                os.replace(source_wrp, saved_wrp)
                wrp_staged = True
                shutil.copyfile(binarized_wrp, source_wrp)
            pack_directory(
                source_dir, pending_pbo, backend=backend,
                poseidon_tools_path=poseidon_tools_path,
            )
        finally:
            if wrp_staged:
                os.replace(saved_wrp, source_wrp)
            os.replace(saved_config, source_config)
            (source_dir / "config.bin").unlink(missing_ok=True)
        os.replace(pending_pbo, pbo_path)
