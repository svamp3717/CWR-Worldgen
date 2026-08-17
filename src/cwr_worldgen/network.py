# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import os
import ssl
import sys
import time
from typing import Any
from urllib import error, parse, request

from ._version import __version__


# Compatibility mode for systems whose HTTPS certificate store or intercepted
# TLS chain is broken. Every urllib HTTPS request made by cwr-worldgen uses
# this context, so hostname and certificate validation are intentionally off.
UNVERIFIED_SSL_CONTEXT = ssl.create_default_context()
UNVERIFIED_SSL_CONTEXT.check_hostname = False
UNVERIFIED_SSL_CONTEXT.verify_mode = ssl.CERT_NONE

APP_URL = "https://github.com/svamp3717/CWR-Worldgen"
APP_USER_AGENT = f"CWR-Worldgen/{__version__} (+{APP_URL})"
OVERPASS_USER_AGENT = f"CWR-Worldgen/{__version__} (+{APP_URL}; Overpass client)"
OVERPASS_REFERER = APP_URL
OVERPASS_RATE_LIMIT_BACKOFF_SECONDS = 30.0
OVERTURE_CATALOG_URL = "https://stac.overturemaps.org/catalog.json"
OVERTURE_FALLBACK_RELEASE = "2026-07-22.0"

_ORIGINAL_URLOPEN = request.urlopen


def _request_url(target: object) -> str:
    if isinstance(target, request.Request):
        return str(target.full_url)
    return str(target)


def _is_overpass_url(url: str) -> bool:
    return "overpass" in url.casefold()


def _prepare_request(target: object) -> object:
    if not isinstance(target, request.Request):
        return target
    if _is_overpass_url(str(target.full_url)):
        target.add_header("User-Agent", OVERPASS_USER_AGENT)
        target.add_header("Referer", OVERPASS_REFERER)
        target.add_header("Accept", "application/json")
    return target


def _clone_request_with_url(source: request.Request, url: str) -> request.Request:
    headers = dict(source.headers)
    headers.update(source.unredirected_hdrs)
    return request.Request(
        url,
        data=source.data,
        headers=headers,
        origin_req_host=source.origin_req_host,
        unverifiable=source.unverifiable,
        method=source.get_method(),
    )


def _opentopomap_candidates(target: object) -> tuple[object, ...]:
    url = _request_url(target)
    parsed = parse.urlparse(url)
    if parsed.hostname not in {
        "a.tile.opentopomap.org",
        "b.tile.opentopomap.org",
        "c.tile.opentopomap.org",
    }:
        return (target,)

    candidates: list[object] = []
    for host in ("a.tile.opentopomap.org", "b.tile.opentopomap.org", "c.tile.opentopomap.org"):
        replacement = parsed._replace(netloc=host).geturl()
        candidate = _clone_request_with_url(target, replacement) if isinstance(target, request.Request) else replacement
        candidates.append(candidate)
    return tuple(candidates)


def _call_urlopen(target: object, args: tuple[Any, ...], kwargs: dict[str, Any]):
    call_kwargs = dict(kwargs)
    if _request_url(target).casefold().startswith("https://"):
        call_kwargs["context"] = UNVERIFIED_SSL_CONTEXT
    return _ORIGINAL_URLOPEN(target, *args, **call_kwargs)


def _urlopen_compat(target: object, *args: Any, **kwargs: Any):
    """Open URLs with CWR's compatibility SSL, identity and retry policy."""
    last_error: Exception | None = None
    candidates = _opentopomap_candidates(target)
    for candidate in candidates:
        prepared = _prepare_request(candidate)
        overpass = _is_overpass_url(_request_url(prepared))
        for attempt in range(2 if overpass else 1):
            try:
                return _call_urlopen(prepared, args, kwargs)
            except error.HTTPError as exc:
                last_error = exc
                if overpass and exc.code == 429:
                    # Public Overpass policy asks clients to wait before making
                    # another request. Sleep even after the final failed attempt
                    # so an outer mirror/retry loop also observes the backoff.
                    time.sleep(OVERPASS_RATE_LIMIT_BACKOFF_SECONDS)
                    if attempt == 0:
                        continue
                break
            except Exception as exc:  # noqa: BLE001 - preserve urllib behavior after mirror fallback.
                last_error = exc
                break
        if len(candidates) == 1:
            break
    assert last_error is not None
    raise last_error


def install_network_compatibility() -> None:
    """Install process-wide urllib compatibility behavior once."""
    current = request.urlopen
    if bool(getattr(current, "_cwr_network_compatibility", False)):
        return
    _urlopen_compat._cwr_network_compatibility = True  # type: ignore[attr-defined]
    request.urlopen = _urlopen_compat


def _fetch_latest_overture_release(release_pattern) -> str:
    req = request.Request(
        OVERTURE_CATALOG_URL,
        headers={"User-Agent": APP_USER_AGENT, "Accept": "application/json"},
    )
    with request.urlopen(req, timeout=15) as response:
        document = json.loads(response.read().decode("utf-8"))
    latest = str(document.get("latest") or "").strip()
    if not release_pattern.fullmatch(latest):
        raise RuntimeError(f"Overture STAC catalog returned an invalid latest release: {latest!r}")
    return latest


def install_overture_release_resolution() -> None:
    """Make Overture use the live STAC latest release unless explicitly pinned."""
    from . import overture

    original = overture.selected_overture_release
    if bool(getattr(original, "_cwr_live_release", False)):
        return
    cached: list[str] = []

    def selected_overture_release() -> str:
        if os.environ.get("CWR_OVERTURE_RELEASE", "").strip():
            return original()
        if cached:
            return cached[0]
        try:
            release = _fetch_latest_overture_release(overture._OVERTURE_RELEASE_RE)
        except Exception as exc:  # noqa: BLE001 - Overture remains optional; retain its existing fallback.
            print(
                f"CWR Overture worker: live STAC release lookup failed ({exc}); "
                f"falling back to last-known release {OVERTURE_FALLBACK_RELEASE}",
                file=sys.stderr,
                flush=True,
            )
            return OVERTURE_FALLBACK_RELEASE
        cached.append(release)
        print(f"CWR Overture worker: live STAC latest release={release}", file=sys.stderr, flush=True)
        return release

    selected_overture_release._cwr_live_release = True  # type: ignore[attr-defined]
    overture.selected_overture_release = selected_overture_release
