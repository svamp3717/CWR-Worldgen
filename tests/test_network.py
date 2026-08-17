from __future__ import annotations

from io import BytesIO
import json
import re
from urllib import error, request

from cwr_worldgen import network


class _Response:
    def __init__(self, payload: bytes = b"ok") -> None:
        self._payload = payload
        self.headers = {}

    def read(self, *_args):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_https_uses_unverified_ssl_context(monkeypatch) -> None:
    calls = []

    def fake(target, *args, **kwargs):
        calls.append((target, args, kwargs))
        return _Response()

    monkeypatch.setattr(network, "_ORIGINAL_URLOPEN", fake)
    network._urlopen_compat("https://example.invalid/file")
    assert calls[0][2]["context"] is network.UNVERIFIED_SSL_CONTEXT
    assert network.UNVERIFIED_SSL_CONTEXT.check_hostname is False


def test_opentopomap_retries_a_b_c(monkeypatch) -> None:
    hosts = []

    def fixed_fake(target, *_args, **_kwargs):
        host = network.parse.urlparse(network._request_url(target)).hostname
        hosts.append(host)
        if host != "c.tile.opentopomap.org":
            raise OSError("mirror unavailable")
        return _Response(b"png")

    monkeypatch.setattr(network, "_ORIGINAL_URLOPEN", fixed_fake)
    response = network._urlopen_compat("https://a.tile.opentopomap.org/12/1/2.png")
    assert response.read() == b"png"
    assert hosts == ["a.tile.opentopomap.org", "b.tile.opentopomap.org", "c.tile.opentopomap.org"]


def test_overpass_identifies_client_and_waits_30_seconds_on_429(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake(target, *_args, **_kwargs):
        calls.append(target)
        if len(calls) == 1:
            raise error.HTTPError(network._request_url(target), 429, "rate limited", {}, BytesIO())
        return _Response(b"{}")

    monkeypatch.setattr(network, "_ORIGINAL_URLOPEN", fake)
    monkeypatch.setattr(network.time, "sleep", lambda seconds: sleeps.append(seconds))
    req = request.Request(
        "https://overpass-api.de/api/interpreter",
        data=b"data=x",
        headers={"User-Agent": "old-client"},
        method="POST",
    )
    network._urlopen_compat(req)
    assert sleeps == [30.0]
    sent = calls[-1]
    headers = {key.casefold(): value for key, value in sent.header_items()}
    assert headers["user-agent"] == network.OVERPASS_USER_AGENT
    assert headers["referer"] == network.OVERPASS_REFERER
    assert headers["accept"] == "application/json"


def test_latest_overture_release_is_read_from_live_catalog(monkeypatch) -> None:
    payload = json.dumps({"latest": "2026-07-22.0"}).encode("utf-8")
    monkeypatch.setattr(network.request, "urlopen", lambda *_args, **_kwargs: _Response(payload))
    release = network._fetch_latest_overture_release(re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$"))
    assert release == "2026-07-22.0"
