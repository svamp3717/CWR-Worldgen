"""Backward-compatible DayZ helper imports.

Game-neutral detection now lives in p3d_game_profiles so the asset browser
and model categoriser can recognize Take On Helicopters as well.
"""
from __future__ import annotations

from p3d_game_profiles import is_dayz_model_path, model_format_label

__all__ = ("is_dayz_model_path", "model_format_label")
