from types import SimpleNamespace

import cwr_worldgen.road_postfit_watchdog_policy as watchdog


def test_postfit_watchdog_arms_after_stock_completion_and_cancels(monkeypatch) -> None:
    calls = []
    progress = []

    def fit(*_args, **kwargs):
        callback = kwargs["progress_callback"]
        callback(100, "Stock road fitting complete: 123 objects in 45 chains")
        calls.append("returned")
        return "report"

    monkeypatch.setattr(watchdog, "_ORIGINAL_FIT", fit)
    monkeypatch.setattr(
        watchdog.faulthandler,
        "dump_traceback_later",
        lambda seconds, **kwargs: calls.append(("arm", seconds, kwargs.get("repeat"))),
    )
    monkeypatch.setattr(
        watchdog.faulthandler,
        "cancel_dump_traceback_later",
        lambda: calls.append("cancel"),
    )

    result = watchdog._fit_with_postfit_watchdog(
        None,
        None,
        (),
        SimpleNamespace(),
        progress_callback=lambda value, message: progress.append((value, message)),
    )

    assert result == "report"
    assert progress == [(100, "Stock road fitting complete: 123 objects in 45 chains")]
    assert ("arm", 300.0, False) in calls
    assert calls[-1] == "cancel"


def test_postfit_watchdog_rearms_on_second_stock_completion(monkeypatch) -> None:
    arms = []
    cancels = []

    def fit(*_args, **kwargs):
        callback = kwargs["progress_callback"]
        callback(100, "Stock road fitting complete: 100 objects in 10 chains")
        callback(100, "Stock road fitting complete: 120 objects in 12 chains")
        return "report"

    monkeypatch.setattr(watchdog, "_ORIGINAL_FIT", fit)
    monkeypatch.setattr(
        watchdog.faulthandler,
        "dump_traceback_later",
        lambda seconds, **kwargs: arms.append(seconds),
    )
    monkeypatch.setattr(
        watchdog.faulthandler,
        "cancel_dump_traceback_later",
        lambda: cancels.append(1),
    )

    result = watchdog._fit_with_postfit_watchdog(
        None,
        None,
        (),
        SimpleNamespace(),
        progress_callback=lambda *_args: None,
    )

    assert result == "report"
    assert arms == [300.0, 300.0]
    assert len(cancels) >= 3
