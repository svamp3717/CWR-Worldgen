# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility shim for the removed polygon-native church tower experiment.

Churches once again use the established rectangular church renderer and its
existing model cache.  Keep the installer name because foundation_visual_policy
already imports it, but deliberately do not wrap cache_key or alter generated
church P3Ds.
"""
from __future__ import annotations

_INSTALLED = False


def install_church_native_tower_cache_policy() -> None:
    """Retain compatibility without changing the mature legacy church cache."""
    global _INSTALLED
    _INSTALLED = True
