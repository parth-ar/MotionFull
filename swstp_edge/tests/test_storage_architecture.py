"""
test_storage_architecture.py — Automated verification test for the store-and-forward
offline backup and upload confirmation deletion architecture.
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure hardware/GUI/networking dependencies can be mocked in non-Pi dev environments
for mod in ["cv2", "smbus2", "gpiozero", "gpsdclient", "imutils", "numpy"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

if "requests" not in sys.modules:
    try:
        import requests
    except ImportError:
        import types
        mock_req = MagicMock()
        class MockReqError(Exception): pass
        class MockConnError(MockReqError): pass
        class MockTimeout(MockReqError): pass
        mock_exceptions = types.SimpleNamespace(
            RequestException=MockReqError,
            ConnectionError=MockConnError,
            Timeout=MockTimeout,
        )
        mock_req.exceptions = mock_exceptions
        sys.modules["requests"] = mock_req
        sys.modules["requests.adapters"] = MagicMock()

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import motion
from motion import cleanup_old_local_captures
from network.uploader import (
    _upload_and_cleanup_evidence,
    sync_offline_captures_backlog,
    dynamic_session_info,
)


class TestStorageArchitecture(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="swstp_test_captures_")
        dynamic_session_info["sessionId"] = 999
        dynamic_session_info["authenticated"] = True
        dynamic_session_info["deviceId"] = "TEST_DEV_01"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_motion_simulation_removed(self):
        """Verify that generate_virtual_video_frame was removed from motion.py."""
        self.assertFalse(
            hasattr(motion, "generate_virtual_video_frame"),
            "generate_virtual_video_frame should be removed from motion.py",
        )

    def test_reset_tracking_exists_and_callable(self):
        """Verify that reset_tracking exists on motion module and does not raise AttributeError."""
        self.assertTrue(hasattr(motion, "reset_tracking"), "motion module must provide reset_tracking()")
        motion.drawn_polygon_pts = [[10, 10], [20, 20]]
        motion.is_drawing_polygon = True
        motion.reset_tracking()
        self.assertFalse(motion.is_drawing_polygon)
        self.assertEqual(motion.drawn_polygon_pts, [])

    def test_confirmed_upload_deletes_local_files(self):
        """Verify that HTTP 200/201 response immediately deletes both .jpg and .json sidecar."""
        jpg_path = os.path.join(self.test_dir, "motion_RTC_2026_09_08_0001.jpg")
        json_path = os.path.join(self.test_dir, "motion_RTC_2026_09_08_0001.json")

        dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb"
        with open(jpg_path, "wb") as f:
            f.write(dummy_jpeg)

        metadata = {
            "image_file": "motion_RTC_2026_09_08_0001.jpg",
            "captured_at": "2026-09-08T12:00:00Z",
            "latitude": 20.9320,
            "longitude": 77.7523,
            "speed_kph": 0.0,
            "idempotency_key": "test-key-1",
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        self.assertTrue(os.path.exists(jpg_path))
        self.assertTrue(os.path.exists(json_path))

        # Mock successful backend response
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"evidenceImageId": 12345, "relativePath": "/uploads/test.jpg"}
        mock_session.post.return_value = mock_resp

        item = dict(metadata)
        item["local_path"] = jpg_path
        item["meta_path"] = json_path
        item["jpeg_bytes"] = dummy_jpeg

        success = _upload_and_cleanup_evidence(
            session=mock_session,
            url="http://mock-backend/api/evidence/upload",
            backend_url="http://mock-backend",
            device_id="TEST_DEV_01",
            ulb_id="ULB_MH_AMRAVATI",
            active_sid=999,
            item=item,
        )

        self.assertTrue(success, "Upload should return True on HTTP 200")
        # Local files MUST BE DELETED!
        self.assertFalse(os.path.exists(jpg_path), "Local JPG should be deleted after confirmed upload")
        self.assertFalse(os.path.exists(json_path), "Local JSON sidecar should be deleted after confirmed upload")

    def test_failed_upload_retains_local_files(self):
        """Verify that network failure or non-200 retains both .jpg and .json on disk as offline backup."""
        jpg_path = os.path.join(self.test_dir, "motion_RTC_2026_09_08_0002.jpg")
        json_path = os.path.join(self.test_dir, "motion_RTC_2026_09_08_0002.json")

        dummy_jpeg = b"FAKE_JPEG_BYTES"
        with open(jpg_path, "wb") as f:
            f.write(dummy_jpeg)

        metadata = {
            "image_file": "motion_RTC_2026_09_08_0002.jpg",
            "captured_at": "2026-09-08T12:05:00Z",
            "latitude": 20.9325,
            "longitude": 77.7530,
            "speed_kph": 1.2,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # Mock network failure
        mock_session = MagicMock()
        import requests
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Offline")

        item = dict(metadata)
        item["local_path"] = jpg_path
        item["meta_path"] = json_path
        item["jpeg_bytes"] = dummy_jpeg

        success = _upload_and_cleanup_evidence(
            session=mock_session,
            url="http://mock-backend/api/evidence/upload",
            backend_url="http://mock-backend",
            device_id="TEST_DEV_01",
            ulb_id="ULB_MH_AMRAVATI",
            active_sid=999,
            item=item,
        )

        self.assertFalse(success, "Upload should return False on network error")
        # Local files MUST REMAIN ON DISK as offline backup!
        self.assertTrue(os.path.exists(jpg_path), "Local JPG should be retained when offline")
        self.assertTrue(os.path.exists(json_path), "Local JSON should be retained when offline")

    def test_backlog_synchronization(self):
        """Verify that sync_offline_captures_backlog uploads and deletes all queued offline captures."""
        # Create 3 backlog captures on disk
        for i in range(1, 4):
            jpg = os.path.join(self.test_dir, f"motion_RTC_2026_09_08_{i:04d}.jpg")
            meta = os.path.join(self.test_dir, f"motion_RTC_2026_09_08_{i:04d}.json")
            with open(jpg, "wb") as f:
                f.write(b"JPEG_DATA")
            with open(meta, "w", encoding="utf-8") as f:
                json.dump({
                    "captured_at": f"2026-09-08T12:0{i}:00Z",
                    "latitude": 20.9320 + (i * 0.001),
                    "longitude": 77.7520 + (i * 0.001),
                    "speed_kph": 0.0,
                }, f)

        # Mock successful backend response for all 3
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"evidenceImageId": 9999, "relativePath": "/uploads/backlog.jpg"}
        mock_session.post.return_value = mock_resp

        synced = sync_offline_captures_backlog(
            backend_url="http://mock-backend",
            device_id="TEST_DEV_01",
            ulb_id="ULB_MH_AMRAVATI",
            active_sid=999,
            session=mock_session,
            save_dir=self.test_dir,
        )

        self.assertEqual(synced, 3, "All 3 backlog captures should be synchronized")
        # Ensure all files in test_dir are now deleted
        remaining_files = [f for f in os.listdir(self.test_dir) if not f.startswith(".")]
        self.assertEqual(remaining_files, [], f"Local storage should be completely clean, found: {remaining_files}")


if __name__ == "__main__":
    unittest.main()
