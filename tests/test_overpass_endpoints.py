# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from cwr_worldgen.location_example import _fetch_overpass
from cwr_worldgen.overpass_endpoints import (
    GLOBAL_OVERPASS_URLS,
    overpass_urls_for_bbox,
)


class _Response:
    def __init__(self, data: bytes) -> None:
        self._stream = BytesIO(data)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self._stream.close()

    def read(self) -> bytes:
        return self._stream.read()


class OverpassEndpointTests(unittest.TestCase):
    def test_global_defaults_use_current_public_instances(self) -> None:
        self.assertEqual(
            GLOBAL_OVERPASS_URLS,
            (
                "https://overpass-api.de/api/interpreter",
                "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                "https://overpass.private.coffee/api/interpreter",
            ),
        )
        self.assertNotIn("https://overpass.kumi.systems/api/interpreter", GLOBAL_OVERPASS_URLS)
        self.assertNotIn("https://overpass.nchc.org.tw/api/interpreter", GLOBAL_OVERPASS_URLS)

    def test_swiss_bbox_prepends_swiss_regional_mirror(self) -> None:
        urls = overpass_urls_for_bbox(GLOBAL_OVERPASS_URLS, (46.8, 7.2, 47.0, 7.5))
        self.assertEqual(urls[0], "https://overpass.osm.ch/api/interpreter")
        self.assertEqual(urls[1:], GLOBAL_OVERPASS_URLS)
        self.assertNotIn("https://overpass.atownsend.org.uk/api/", urls)

    def test_custom_endpoint_list_is_not_expanded(self) -> None:
        custom = ("https://example.invalid/overpass", "https://example.invalid/overpass")
        self.assertEqual(
            overpass_urls_for_bbox(custom, (46.8, 7.2, 47.0, 7.5)),
            ("https://example.invalid/overpass",),
        )

    def test_fetch_falls_through_http_failure_to_next_mirror(self) -> None:
        first = "https://one.invalid/api/interpreter"
        second = "https://two.invalid/api/interpreter"
        payload = b'{"version":0.6,"elements":[]}'
        failure = HTTPError(first, 503, "busy", {"Retry-After": "2"}, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "raw-overpass.json"
            query = root / "overpass-query.txt"
            with patch(
                "cwr_worldgen.location_example.request.urlopen",
                side_effect=(failure, _Response(payload)),
            ) as urlopen, patch("cwr_worldgen.location_example.time.sleep") as sleep:
                selected = _fetch_overpass(
                    target,
                    query,
                    (0.0, 0.0, 0.01, 0.01),
                    (first, second),
                    refresh=True,
                    timeout=30,
                )
            self.assertEqual(selected, second)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(0.5)


if __name__ == "__main__":
    unittest.main()
