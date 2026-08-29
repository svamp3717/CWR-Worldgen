# SPDX-License-Identifier: GPL-3.0-or-later
"""Give isolated paved source corners a connector-locked curve fit.

The sharp-turn fitter originally required at least two significant source
vertices before it even attempted the stock-curve beam. Lundby28 exposes the
consequence of that gate: many paved roads contain one isolated 8-35 degree
corner surrounded by otherwise quiet geometry, so they fall straight through to
rotated ``sil6``/``sil12`` pieces and leave a visible miter.

Do not cover those miters with another road object. Extend only the span detector
so an isolated paved corner gets the same exact-pose beam search as sustained
bends. The beam still owns acceptance: every replacement must stay inside the
existing 0.60 m source corridor and use exact stock connectors. Corners close to
a run boundary are left alone because they belong to junction/endpoint fitting.
"""
from __future__ import annotations

import math

from . import stock_road_sharp_turn_policy as _sharp

MINIMUM_SINGLE_VERTEX_TURN_DEGREES = 7.5
MAXIMUM_SINGLE_VERTEX_TURN_DEGREES = 35.0
MAXIMUM_ADJACENT_SIGNIFICANT_TURN_DEGREES = 0.70

_ORIGINAL_SPANS = None
_INSTALLED = False


def _isolated_single_vertex_spans(points, existing=()):
    """Return expanded spans for isolated corners not handled by the base pass."""

    cleaned = tuple(points)
    if len(cleaned) < 5:
        return ()

    covered = []
    for start, end, _sign in existing:
        covered.append((int(start), int(end)))

    turns = [0.0] * len(cleaned)
    for index in range(1, len(cleaned) - 1):
        turns[index] = _sharp._signed_turn(
            cleaned[index - 1], cleaned[index], cleaned[index + 1]
        )

    result = []
    # Two quiet boundary segments are required so _locked_measure can derive
    # stable entry and exit tangents around the three-point corner span.
    for index in range(2, len(cleaned) - 2):
        turn = float(turns[index])
        magnitude = abs(turn)
        if not (
            MINIMUM_SINGLE_VERTEX_TURN_DEGREES
            <= magnitude
            <= MAXIMUM_SINGLE_VERTEX_TURN_DEGREES
        ):
            continue
        if any(start <= index <= end + 1 for start, end in covered):
            continue
        if (
            abs(float(turns[index - 1]))
            >= MAXIMUM_ADJACENT_SIGNIFICANT_TURN_DEGREES
            or abs(float(turns[index + 1]))
            >= MAXIMUM_ADJACENT_SIGNIFICANT_TURN_DEGREES
        ):
            continue

        start = index - 1
        end = index + 1
        length = sum(
            math.dist(a, b)
            for a, b in zip(cleaned[start:end], cleaned[start + 1 : end + 1])
        )
        if length <= 1.0 or length > _sharp._MAXIMUM_SPAN_METRES:
            continue
        result.append((start, end, 1 if turn > 0.0 else -1))
    return tuple(result)


def _single_vertex_sharp_turn_spans(points):
    if _ORIGINAL_SPANS is None:
        raise RuntimeError("single-vertex bend policy is not installed")

    existing = tuple(_ORIGINAL_SPANS(points))
    additions = _isolated_single_vertex_spans(points, existing)
    if not additions:
        return existing
    return tuple(sorted((*existing, *additions), key=lambda item: (item[0], item[1])))


def install_stock_road_single_vertex_bend_policy() -> None:
    """Install isolated-corner discovery after the micro-bend beam is active."""

    global _ORIGINAL_SPANS, _INSTALLED
    if _INSTALLED:
        return
    if not _sharp._INSTALLED:
        raise RuntimeError("stock road sharp-turn policy must install first")

    _ORIGINAL_SPANS = _sharp._sharp_turn_spans
    _sharp._sharp_turn_spans = _single_vertex_sharp_turn_spans
    _INSTALLED = True
