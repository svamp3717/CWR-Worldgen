"""Island Studio should dismiss the onefile bootloader splash on GUI startup."""
from __future__ import annotations

from types import SimpleNamespace
import sys
from unittest.mock import Mock

from cwr_island_studio import app


def _fake_startup(monkeypatch):
    events = []
    root = Mock()
    root.update_idletasks.side_effect = lambda: events.append("tk-ready")
    root.mainloop.side_effect = lambda: events.append("mainloop")
    monkeypatch.setattr(app.tk, "Tk", lambda: root)
    monkeypatch.setattr(
        app, "IslandStudio",
        lambda created_root, path: events.append("studio-created")
    )
    return root, events


def test_frozen_studio_closes_splash_after_window_setup(monkeypatch) -> None:
    root, events = _fake_startup(monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "pyi_splash", SimpleNamespace(
        close=lambda: events.append("splash-close")
    ))
    assert app.main([]) == 0
    assert events == [
        "studio-created", "tk-ready", "splash-close", "mainloop",
    ]
    root.update_idletasks.assert_called_once_with()


def test_source_studio_does_not_dismiss_splash(monkeypatch) -> None:
    root, events = _fake_startup(monkeypatch)
    monkeypatch.delattr(sys, "frozen", raising=False)
    splash = Mock()
    monkeypatch.setitem(sys.modules, "pyi_splash", splash)
    assert app.main([]) == 0
    assert events == ["studio-created", "mainloop"]
    splash.close.assert_not_called()
    root.update_idletasks.assert_not_called()


def test_missing_or_dead_splash_does_not_block_startup(monkeypatch) -> None:
    root, events = _fake_startup(monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "pyi_splash", None)
    assert app.main([]) == 0
    assert events == ["studio-created", "tk-ready", "mainloop"]

    events.clear()
    broken = Mock()
    broken.close.side_effect = OSError("bootloader already exited")
    monkeypatch.setitem(sys.modules, "pyi_splash", broken)
    assert app.main([]) == 0
    broken.close.assert_called_once_with()
    assert events == ["studio-created", "tk-ready", "mainloop"]
