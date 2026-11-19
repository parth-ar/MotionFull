"""
test_evidence_packets.py — Checks the multipart form sent to POST /api/evidence/upload
for motion and litter items, on both the live path and offline backlog replay.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

for mod in ["cv2", "smbus2", "gpiozero", "gpsdclient", "imutils", "numpy"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

_EDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDGE_DIR not in sys.path:
    sys.path.insert(0, _EDGE_DIR)

from telemetry import latest_sensor
from network.uploader import _upload_and_cleanup_evidence, sync_offline_captures_backlog

# The vehicle's position "now" (at sync time) — must never be used for backlog items.
CURRENT_LAT, CURRENT_LON = 21.1111, 78.2222


def _ok_session():
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"evidenceImageId": 1, "relativePath": "/x.jpg"}
    session.post.return_value = resp
    return session


def _sent_forms(session):
    return [c.kwargs["data"] for c in session.post.call_args_list]


class TestEvidencePackets(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="swstp_evidence_")
        self._saved = {k: latest_sensor.get(k) for k in ("gps_valid", "lat", "lon")}
        latest_sensor.update(gps_valid=True, lat=CURRENT_LAT, lon=CURRENT_LON)

    def tearDown(self):
        latest_sensor.update(self._saved)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write_capture(self, name, sidecar):
        jpg = os.path.join(self.dir, f"{name}.jpg")
        with open(jpg, "wb") as f:
            f.write(b"JPEG")
        with open(os.path.join(self.dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f)
        return jpg

    def _sync(self, session):
        return sync_offline_captures_backlog(
            backend_url="http://mock", device_id="DEV", ulb_id="ULB",
            active_sid=1, session=session, save_dir=self.dir,
        )

    def _upload(self, session, item):
        return _upload_and_cleanup_evidence(
            session=session, url="http://mock/api/evidence/upload",
            backend_url="http://mock", device_id="DEV", ulb_id="ULB",
            active_sid=1, item=item,
        )

    # ── Task 1: backlog location ──────────────────────────────────────────
    def test_new_litter_sidecar_uses_capture_location(self):
        self._write_capture("litter_new", {
            "latitude": 20.5, "longitude": 77.5,
            "gnss": {"latitude": 20.5, "longitude": 77.5},
            "detection_type": "LITTER", "idempotency_key": "k-new",
        })
        session = _ok_session()
        self.assertEqual(self._sync(session), 1)
        form = _sent_forms(session)[0]
        self.assertEqual((form["latitude"], form["longitude"]), (20.5, 77.5))

    def test_old_litter_sidecar_reads_nested_gnss(self):
        self._write_capture("litter_old", {
            "gnss": {"latitude": 20.6, "longitude": 77.6},
            "detection_type": "LITTER",
        })
        session = _ok_session()
        self.assertEqual(self._sync(session), 1)
        form = _sent_forms(session)[0]
        self.assertEqual((form["latitude"], form["longitude"]), (20.6, 77.6))

    def test_backlog_without_fix_sends_no_location_and_still_cleans_up(self):
        jpg = self._write_capture("litter_nofix", {
            "latitude": None, "longitude": None,
            "gnss": {"latitude": None, "longitude": None},
            "detection_type": "LITTER",
        })
        session = _ok_session()
        self.assertEqual(self._sync(session), 1)
        form = _sent_forms(session)[0]
        self.assertIsNone(form["latitude"])
        self.assertIsNone(form["longitude"])
        self.assertFalse(os.path.exists(jpg), "confirmed upload must delete the local copy")

    def test_live_item_still_falls_back_to_current_gps(self):
        session = _ok_session()
        self.assertTrue(self._upload(session, {
            "jpeg_bytes": b"JPEG", "latitude": None, "longitude": None,
            "detection_type": "LITTER",
        }))
        form = _sent_forms(session)[0]
        self.assertEqual((form["latitude"], form["longitude"]), (CURRENT_LAT, CURRENT_LON))

    def test_live_item_without_any_location_does_not_crash(self):
        latest_sensor.update(gps_valid=False)
        for dtype in ("MOTION", "LITTER"):
            session = _ok_session()
            self.assertTrue(self._upload(session, {
                "jpeg_bytes": b"JPEG", "latitude": None, "longitude": None,
                "detection_type": dtype,
            }))

    # ── Task 2: idempotency key ───────────────────────────────────────────
    def test_idempotency_key_is_stable_across_retries(self):
        item = {"jpeg_bytes": b"JPEG", "latitude": 20.5, "longitude": 77.5,
                "detection_type": "LITTER", "idempotency_key": "fixed-key"}
        s1, s2 = _ok_session(), _ok_session()
        self._upload(s1, dict(item))
        self._upload(s2, dict(item))
        self.assertEqual(_sent_forms(s1)[0]["idempotencyKey"], "fixed-key")
        self.assertEqual(_sent_forms(s2)[0]["idempotencyKey"], "fixed-key")

    # ── Task 5: zero confidence ───────────────────────────────────────────
    def test_zero_detection_confidence_is_sent(self):
        session = _ok_session()
        self._upload(session, {"jpeg_bytes": b"JPEG", "latitude": 20.5, "longitude": 77.5,
                               "detection_type": "LITTER", "detection_confidence": 0.0})
        self.assertEqual(_sent_forms(session)[0]["detectionConfidence"], 0.0)

    # ── Task 6: motionConfidence dropped ──────────────────────────────────
    def test_motion_confidence_not_sent(self):
        session = _ok_session()
        self._upload(session, {"jpeg_bytes": b"JPEG", "latitude": 20.5, "longitude": 77.5,
                               "motion_confidence": 0.95})
        self.assertNotIn("motionConfidence", _sent_forms(session)[0])

    # ── Task 7: detectionType tag ─────────────────────────────────────────
    def test_motion_items_tagged_motion(self):
        session = _ok_session()
        self._upload(session, {"jpeg_bytes": b"JPEG", "latitude": 20.5, "longitude": 77.5,
                               "detection_type": "MOTION"})
        # Old motion sidecar on disk with no tag at all
        self._write_capture("motion_old", {"latitude": 20.5, "longitude": 77.5})
        self._sync(session)
        forms = _sent_forms(session)
        self.assertEqual([f["detectionType"] for f in forms], ["MOTION", "MOTION"])

    def test_litter_items_tagged_litter(self):
        session = _ok_session()
        self._upload(session, {"jpeg_bytes": b"JPEG", "latitude": 20.5, "longitude": 77.5,
                               "detection_type": "LITTER"})
        self.assertEqual(_sent_forms(session)[0]["detectionType"], "LITTER")


if __name__ == "__main__":
    unittest.main()
