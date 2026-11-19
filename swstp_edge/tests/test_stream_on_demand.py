"""
test_stream_on_demand.py — Checks the live camera streamer in on-demand mode
(config.STREAM_ON_DEMAND = True) and continuous mode (False).
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

_EDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDGE_DIR not in sys.path:
    sys.path.insert(0, _EDGE_DIR)

import numpy as np

import config as _cfg
import network.uploader as up

FRAME = np.zeros((36, 64, 3), dtype=np.uint8)


def _resp(status=200, json_body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body if json_body is not None else {}
    return r


class TestStreamOnDemand(unittest.TestCase):
    def setUp(self):
        from telemetry import latest_sensor
        self.patches = [
            patch.dict(up.dynamic_session_info, {"authenticated": True, "deviceId": "DEV"}),
            patch.dict(latest_sensor, {"deviceId": None}),
            patch.multiple(_cfg, STREAM_DEMAND_POLL_SEC=0.5, create=True),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _run(self, session, seconds=1.3, **kwargs):
        """Runs live_frame_streamer briefly with a mocked HTTP session; returns the session."""
        stop = threading.Event()
        get_frame = lambda: (FRAME, time.monotonic())   # always a fresh frame
        with patch.object(up.requests, "Session", return_value=session):
            t = threading.Thread(target=up.live_frame_streamer,
                                 args=("http://mock", "DEV", 10.0, stop, threading.Lock(), get_frame),
                                 kwargs=kwargs, daemon=True)
            t.start()
            time.sleep(seconds)
            stop.set()
            t.join(timeout=3.0)
        self.assertFalse(t.is_alive(), "streamer must stop when stop_event is set")
        return session

    def _session(self, get_result):
        s = MagicMock()
        s.post.return_value = _resp(200)
        if isinstance(get_result, Exception):
            s.get.side_effect = get_result
        else:
            s.get.return_value = get_result
        return s

    def test_continuous_mode_streams_without_asking_backend(self):
        s = self._run(self._session(_resp(200, {"active": False})), on_demand=False)
        self.assertGreater(s.post.call_count, 3)
        s.get.assert_not_called()

    def test_config_false_means_continuous(self):
        with patch.object(_cfg, "STREAM_ON_DEMAND", False):
            s = self._run(self._session(_resp(200, {"active": False})))
        self.assertGreater(s.post.call_count, 3)
        s.get.assert_not_called()

    def test_on_demand_no_viewer_sends_nothing(self):
        s = self._run(self._session(_resp(200, {"active": False})), on_demand=True)
        s.post.assert_not_called()
        self.assertGreaterEqual(s.get.call_count, 1)
        self.assertTrue(s.get.call_args.args[0].endswith("/api/camera/DEV/stream-status"))

    def test_on_demand_viewer_streams(self):
        s = self._run(self._session(_resp(200, {"active": True})), on_demand=True)
        self.assertGreater(s.post.call_count, 3)
        self.assertTrue(s.post.call_args.args[0].endswith("/api/camera/DEV/frame"))

    def test_on_demand_endpoint_missing_sends_nothing(self):
        s = self._run(self._session(_resp(404)), on_demand=True)
        s.post.assert_not_called()

    def test_on_demand_network_error_sends_nothing(self):
        s = self._run(self._session(up.requests.exceptions.ConnectionError("down")), on_demand=True)
        s.post.assert_not_called()

    def test_on_demand_stops_when_viewer_leaves(self):
        s = self._session(None)
        s.get.side_effect = [_resp(200, {"active": True})] + [_resp(200, {"active": False})] * 20
        self._run(s, seconds=1.6, on_demand=True)
        posts_after_first_poll = s.post.call_count
        self.assertGreater(posts_after_first_poll, 0)
        self.assertLess(posts_after_first_poll, 8, "frames must stop after active turns false")

    def test_poll_stream_demand_parsing(self):
        s = MagicMock()
        s.get.return_value = _resp(200, {"active": True})
        self.assertTrue(up.poll_stream_demand(s, "http://m/", "D", "/api/camera/{device_id}/stream-status"))
        s.get.return_value = _resp(200, {})
        self.assertFalse(up.poll_stream_demand(s, "http://m", "D", "/x/{device_id}"))
        s.get.return_value = _resp(500)
        self.assertIsNone(up.poll_stream_demand(s, "http://m", "D", "/x/{device_id}"))


if __name__ == "__main__":
    unittest.main()
