"""Worldgen should dismiss its bootloader splash only after Tk is prepared."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from cwr_worldgen import gui


def test_frozen_worldgen_closes_splash_before_mainloop(monkeypatch, tmp_path) -> None:
    events: list[str] = []
    app = Mock()
    app.update_idletasks.side_effect = lambda: events.append("tk-ready")
    app.mainloop.side_effect = lambda: events.append("mainloop")
    monkeypatch.setattr(gui, "WorldgenGui", lambda: app)
    monkeypatch.setattr(gui, "application_base_dir", lambda: tmp_path)
    monkeypatch.setattr(gui.os, "chdir", lambda *_: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "pyi_splash",
        SimpleNamespace(close=lambda: events.append("splash-close")),
    )

    assert gui.main([]) == 0
    assert events == ["tk-ready", "splash-close", "mainloop"]


def test_source_worldgen_does_not_access_frozen_splash(monkeypatch) -> None:
    app = Mock()
    monkeypatch.setattr(gui, "WorldgenGui", lambda: app)
    monkeypatch.delattr(sys, "frozen", raising=False)
    splash = Mock()
    monkeypatch.setitem(sys.modules, "pyi_splash", splash)

    assert gui.main([]) == 0
    splash.close.assert_not_called()
    app.update_idletasks.assert_not_called()
    app.mainloop.assert_called_once_with()
