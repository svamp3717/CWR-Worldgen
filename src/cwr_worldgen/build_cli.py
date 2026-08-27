# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import sys
from collections.abc import Sequence

from .cli import main as _cli_main


def _translate_args(argv: Sequence[str]) -> list[str]:
    """Translate the stable user-facing build command onto Milestone 9.

    ``worldgen build`` is intentionally a very small compatibility layer over the
    current production pipeline.  Keeping it here means road/terrain fixes on a
    development branch can be tested against a real frozen source bundle without
    teaching users which internal milestone happens to be current.
    """

    args = list(argv)
    if not args or args[0] != "build":
        return args

    tail = args[1:]

    # Convenience form: ``worldgen build SOURCE_DIR --output ...``.
    if tail and not tail[0].startswith("-"):
        tail = ["--source-dir", tail[0], *tail[1:]]

    # Also accept the shorter spelling when callers prefer explicit options.
    translated: list[str] = []
    for token in tail:
        if token == "--source":
            token = "--source-dir"
        elif token.startswith("--source="):
            token = "--source-dir=" + token.split("=", 1)[1]
        translated.append(token)

    return ["milestone9", *translated]


def main(argv: list[str] | None = None) -> int:
    """Run the normal CLI, exposing ``build`` as the production source build."""

    effective_argv = sys.argv[1:] if argv is None else argv
    return _cli_main(_translate_args(effective_argv))
