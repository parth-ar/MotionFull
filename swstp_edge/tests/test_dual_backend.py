"""
test_dual_backend.py — every transmitted packet goes to BOTH backends:
  primary   = config.DEFAULT_BACKEND_URL    (https://solidwasteapi.scipl.info.in)
  secondary = config.SECONDARY_BACKEND_URL  (https://solidwastenewapi.scipl.info.in)

Telemetry batches, raw GNSS live location and evidence are mirrored to the secondary through
background queues with the secondary's own session; the camera stays primary-only; evidence
is deleted locally when the primary confirms; a dead secondary never affects the primary.
"""

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

for mod in ["cv2", "smbus2", "gpiozero", "gpsdclient", "imutils", "numpy"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

_EDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDGE_DIR not in sys.path:
    sys.path.insert(0, _EDGE_DIR)

import config as _cfg
import network.uploader as up

SECONDARY = "https://solidwastenewapi.scipl.info.in"


def _resp(code=200, body=None):
    r = MagicMock()
    r.status_code = code
    r.json.return_value = body if body is not None else {}
    r.text = ""
    return r


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


class DualBackendBase(unittest.TestCase):
    def setUp(self):
        self.mirror_http = MagicMock()                      # session used by mirror workers
        self.mirror_http.post.return_value = _resp(200)
        self.login = MagicMock(return_value=_resp(200, {"sessionId": 777, "ulbId": "ULB_NEW"}))
        self.stream_worker = up._MirrorWorker("stream-test", maxsize=5)
        self.evidence_worker = up._MirrorWorker("evidence-test", maxsize=5)
        self.patches = [
            patch.object(_cfg, "SECONDARY_BACKEND_URL", SECONDARY),
            patch.object(up, "_mirror_stream", self.stream_worker),
            patch.object(up, "_mirror_evidence", self.evidence_worker),
            patch.object(up.requests, "Session", return_value=self.mirror_http),
            patch.object(up.requests, "post", self.login),
            patch.dict(up.dynamic_session_info, {"deviceId": "SWSTP-PI-001", "sessionId": 111,
                                                  "authenticated": True}),
            patch.dict(up.secondary_session_info, {"sessionId": 0, "authenticated": False, "ulbId": None}),
            patch.object(up, "_secondary_last_login", 0.0),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def mirror_calls(self, path_part):
        return [c for c in self.mirror_http.post.call_args_list if path_part in c.args[0]]


class TestMirroring(DualBackendBase):
    def test_config_defaults(self):
        self.assertEqual(_cfg.DEFAULT_BACKEND_URL.rstrip("/"), "https://solidwasteapi.scipl.info.in")
        self.assertEqual(up.secondary_backend_url(), SECONDARY)

    def test_telemetry_batch_mirrored_with_secondary_session(self):
        pkts = [{"seq": 1}, {"seq": 2}]
        up.mirror_telemetry_batch(pkts)
        self.assertTrue(_wait(lambda: self.mirror_calls("/api/telemetry/ingest-batch")))
        call = self.mirror_calls("/api/telemetry/ingest-batch")[0]
        self.assertEqual(call.args[0], SECONDARY + "/api/telemetry/ingest-batch")
        self.assertEqual(call.kwargs["json"]["sessionId"], 777, "secondary's own session, not primary's 111")
        self.assertEqual(call.kwargs["json"]["packets"], pkts)
        self.login.assert_called_once()
        self.assertEqual(self.login.call_args.args[0], SECONDARY + "/api/sessions/start-hardware-session")
        self.assertEqual(self.login.call_args.kwargs["json"]["deviceCode"], "SWSTP-PI-001")

    def test_live_location_mirrored_unchanged(self):
        payload = {"latitude": 20.93, "longitude": 77.75, "fix": True}
        up.mirror_live_location("SWSTP-AMRMC-001", payload)
        self.assertTrue(_wait(lambda: self.mirror_calls("/live_location")))
        call = self.mirror_calls("/live_location")[0]
        self.assertEqual(call.args[0], SECONDARY + "/api/SWSTP-AMRMC-001/live_location")
        self.assertEqual(call.kwargs["json"], payload)

    def test_telemetry_dropped_while_secondary_login_rejected(self):
        self.login.return_value = _resp(403)
        up.mirror_telemetry_batch([{"seq": 1}])
        self.assertTrue(_wait(lambda: self.login.called))
        time.sleep(0.1)
        self.assertEqual(self.mirror_calls("/api/telemetry/ingest-batch"), [])

    def test_401_forces_relogin(self):
        up.secondary_session_info.update(sessionId=5, authenticated=True)
        self.mirror_http.post.return_value = _resp(401)
        up.mirror_telemetry_batch([{"seq": 1}])
        self.assertTrue(_wait(lambda: not up.secondary_session_info["authenticated"]))

    def test_disabled_when_url_empty(self):
        with patch.object(_cfg, "SECONDARY_BACKEND_URL", ""):
            up.mirror_telemetry_batch([{"seq": 1}])
            up.mirror_live_location("V", {"a": 1})
        time.sleep(0.1)
        self.mirror_http.post.assert_not_called()
        self.assertIsNone(self.stream_worker._thread)

    def test_full_queue_drops_oldest_and_never_blocks(self):
        gate = threading.Event()
        self.mirror_http.post.side_effect = lambda *a, **k: (gate.wait(2.0), _resp(200))[1]
        t0 = time.perf_counter()
        for i in range(50):
            up.mirror_live_location("V", {"i": i})
        self.assertLess(time.perf_counter() - t0, 0.2, "submit must never block the primary path")
        self.assertGreater(self.stream_worker.dropped, 0)
        gate.set()

    def test_secondary_down_does_not_raise(self):
        self.mirror_http.post.side_effect = ConnectionError("down")
        up.mirror_live_location("V", {"i": 1})
        self.assertTrue(_wait(lambda: self.stream_worker.failed >= 1))


class TestEvidenceMirroring(DualBackendBase):
    def setUp(self):
        super().setUp()
        self.dir = tempfile.mkdtemp(prefix="swstp_dual_")
        self.jpg = os.path.join(self.dir, "motion_0001.jpg")
        self.meta = os.path.join(self.dir, "motion_0001.json")
        for p, content in ((self.jpg, b"JPEGDATA"), (self.meta, b"{}")):
            with open(p, "wb") as f:
                f.write(content)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        super().tearDown()

    def _upload(self, primary_status):
        primary = MagicMock()
        primary.post.return_value = _resp(primary_status, {"evidenceImageId": 1})
        ok = up._upload_and_cleanup_evidence(
            session=primary, url="https://solidwasteapi.scipl.info.in/api/evidence/upload",
            backend_url="https://solidwasteapi.scipl.info.in", device_id="SWSTP-PI-001",
            ulb_id="ULB_MH_AMRAVATI", active_sid=111,
            item={"local_path": self.jpg, "meta_path": self.meta, "jpeg_bytes": b"JPEGDATA",
                  "latitude": 20.93, "longitude": 77.75, "idempotency_key": "key-1",
                  "detection_type": "LITTER"})
        return ok, primary

    def test_mirrored_after_primary_confirms(self):
        ok, primary = self._upload(200)
        self.assertTrue(ok)
        self.assertEqual(primary.post.call_args.kwargs["data"]["sessionId"], 111)
        self.assertFalse(os.path.exists(self.jpg), "deleted on primary confirmation")
        self.assertTrue(_wait(lambda: self.mirror_calls("/api/evidence/upload")))
        call = self.mirror_calls("/api/evidence/upload")[0]
        self.assertEqual(call.args[0], SECONDARY + "/api/evidence/upload")
        self.assertEqual(call.kwargs["data"]["sessionId"], 777)
        self.assertEqual(call.kwargs["data"]["ulbId"], "ULB_NEW")
        self.assertEqual(call.kwargs["data"]["idempotencyKey"], "key-1")
        self.assertEqual(call.kwargs["files"]["file"][1], b"JPEGDATA", "bytes survive local delete")

    def test_not_mirrored_when_primary_fails(self):
        ok, _ = self._upload(500)
        self.assertFalse(ok)
        self.assertTrue(os.path.exists(self.jpg), "kept for primary retry")
        time.sleep(0.1)
        self.assertEqual(self.mirror_calls("/api/evidence/upload"), [])


class TestCameraPrimaryOnly(unittest.TestCase):
    def test_frame_streamer_has_no_mirror(self):
        import inspect
        src = inspect.getsource(up.live_frame_streamer)
        self.assertNotIn("mirror_", src)


if __name__ == "__main__":
    unittest.main()
