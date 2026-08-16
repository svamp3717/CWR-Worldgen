# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from cwr_worldgen.location_example import _fetch_overpass


class _Response:
    def __init__(self, data: bytes) -> None:
        self._stream = BytesIO(data)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self._stream.close()

    def read(self, *_args: object) -> bytes:
        return self._stream.read()


class OverpassRetryTests(unittest.TestCase):
    def test_transient_504_retries_after_all_endpoints_fail(self) -> None:
        endpoint = "https://one.invalid/api/interpreter"
        payload = b'{"version":0.6,"elements":[]}'
        failure = HTTPError(endpoint, 504, "gateway timeout", {}, None)
        stages: list[str] = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch(
                "cwr_worldgen.location_example.request.urlopen",
                side_effect=(failure, _Response(payload)),
            ) as urlopen, patch("cwr_worldgen.overpass_retry.time.sleep") as sleep:
                selected = _fetch_overpass(
                    root / "raw-overpass.json",
                    root / "overpass-query.txt",
                    (0.0, 0.0, 0.01, 0.01),
                    (endpoint,),
                    refresh=True,
                    timeout=30,
                    retry_rounds=3,
                    retry_delay_seconds=120,
                    retry_countdown_seconds=30,
                    progress_callback=lambda _percent, stage: stages.append(stage),
                )
        self.assertEqual(selected, endpoint)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(sleep.call_count, 4)
        self.assertTrue(any("Automatic retry 2/3 in 2:00" in stage for stage in stages))

    def test_non_transient_http_error_does_not_wait_and_retry(self) -> None:
        endpoint = "https://one.invalid/api/interpreter"
        failure = HTTPError(endpoint, 400, "bad request", {}, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch(
                "cwr_worldgen.location_example.request.urlopen",
                side_effect=failure,
            ) as urlopen, patch("cwr_worldgen.overpass_retry.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "all Overpass endpoints failed"):
                    _fetch_overpass(
                        root / "raw-overpass.json",
                        root / "overpass-query.txt",
                        (0.0, 0.0, 0.01, 0.01),
                        (endpoint,),
                        refresh=True,
                        timeout=30,
                    )
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_exhausted_transient_retries_explain_that_outage_is_temporary(self) -> None:
        endpoint = "https://one.invalid/api/interpreter"
        failures = tuple(
            HTTPError(endpoint, 504, "gateway timeout", {}, None)
            for _ in range(3)
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch(
                "cwr_worldgen.location_example.request.urlopen",
                side_effect=failures,
            ) as urlopen, patch("cwr_worldgen.overpass_retry.time.sleep"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "still unavailable after 3 automatic attempt round",
                ) as raised:
                    _fetch_overpass(
                        root / "raw-overpass.json",
                        root / "overpass-query.txt",
                        (0.0, 0.0, 0.01, 0.01),
                        (endpoint,),
                        refresh=True,
                        timeout=30,
                        retry_delay_seconds=0,
                    )
        self.assertEqual(urlopen.call_count, 3)
        self.assertIn("Existing source files are safe", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
