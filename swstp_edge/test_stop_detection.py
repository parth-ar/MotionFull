"""
test_stop_detection.py — Comprehensive automated test suite for vehicle stop detection.

Verifies:
  1. Vehicle comes to rest (dual GNSS and IMU fusion).
  2. Stop timer starts at rest detection, tracking elapsed duration continuously.
  3. No timeout: absence of camera motion does NOT end the stop.
  4. Logging of motion frames captured during the stop with exact offsets and metadata.
  5. Vehicle gets back in motion (speed > 5.0 km/h via GNSS and IMU).
  6. Stop ends upon motion resume, outputting full historical summary report.
  7. Fallback handling when IMU is unavailable (GNSS-only) or GNSS is lost (IMU-only).
  8. Debouncing against transient sensor noise.
"""

import math
import os
import sys
import time
import unittest
from unittest.mock import MagicMock

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

import main as edge_main
from stop_detector import (
    VehicleStopDetector,
    StopEventRecord,
    StopCaptureItem,
)


class TestVehicleStopDetection(unittest.TestCase):
    def setUp(self):
        # Instantiate detector with short debounce for rapid test execution
        self.detector = VehicleStopDetector(
            stop_speed_gate=5.0,
            rest_speed_threshold=3.0,
            imu_accel_tolerance=0.45,
            imu_gyro_tolerance=4.0,
            rest_debounce_sec=0.5,
            motion_debounce_sec=0.3,
        )

    def _make_imu(self, accel_mag=9.80665, gx=0.0, gy=0.0, gz=0.0, valid=True):
        return {
            "valid": valid,
            "accel_magnitude_ms2": accel_mag,
            "gyro_dps": {"x": gx, "y": gy, "z": gz},
        }

    # -----------------------------------------------------------------------
    # Condition 1: Vehicle comes to rest (GNSS & IMU)
    # -----------------------------------------------------------------------
    def test_sensor_evaluation_rest_and_motion(self):
        """Test instantaneous sensor classification logic."""
        # Rest: speed 0.0 km/h, IMU 9.80665 m/s² and 0 gyro
        imu_rest = self._make_imu(accel_mag=9.81, gx=0.5, gy=0.2, gz=0.3)
        is_rest, is_motion, metrics = self.detector.evaluate_sensors(0.0, imu_rest, gps_valid=True)
        self.assertTrue(is_rest)
        self.assertFalse(is_motion)
        self.assertAlmostEqual(metrics["dyn_accel"], abs(9.81 - 9.80665), places=3)

        # In motion: speed 12.0 km/h, dynamic accel 1.5, gyro 10 deg/s
        imu_moving = self._make_imu(accel_mag=11.5, gx=5.0, gy=8.0, gz=6.0)
        is_rest, is_motion, metrics = self.detector.evaluate_sensors(12.0, imu_moving, gps_valid=True)
        self.assertFalse(is_rest)
        self.assertTrue(is_motion)

    def test_vehicle_comes_to_rest_starts_timer(self):
        """Condition 1 & 2: Rest confirmed after debounce -> starts timer at 0.0s."""
        t0 = 1000.0
        imu_rest = self._make_imu(accel_mag=9.80665, gx=0.0, gy=0.0, gz=0.0)

        # Initial tick at rest
        res = self.detector.update(
            vehicle_speed=0.0,
            imu_data=imu_rest,
            latitude=20.9320,
            longitude=77.7523,
            rtc_timestamp="2026-09-08 14:00:00",
            now_mono=t0,
        )
        self.assertEqual(res["state"], "MOVING")
        self.assertFalse(res["is_stopped"])
        self.assertFalse(res["event_just_started"])

        # Tick at t0 + 0.2s (within debounce window)
        res = self.detector.update(
            vehicle_speed=0.0,
            imu_data=imu_rest,
            latitude=20.9320,
            longitude=77.7523,
            rtc_timestamp="2026-09-08 14:00:00",
            now_mono=t0 + 0.2,
        )
        self.assertEqual(res["state"], "MOVING")

        # Tick at t0 + 0.5s (debounce threshold met -> STOPPED)
        res = self.detector.update(
            vehicle_speed=0.0,
            imu_data=imu_rest,
            latitude=20.9320,
            longitude=77.7523,
            rtc_timestamp="2026-09-08 14:00:01",
            now_mono=t0 + 0.5,
        )
        self.assertEqual(res["state"], "STOPPED")
        self.assertTrue(res["is_stopped"])
        self.assertTrue(res["event_just_started"])
        self.assertEqual(res["stop_event_id"], 1)
        self.assertEqual(res["capture_count"], 0)
        self.assertIsNotNone(self.detector.active_stop)

        # Verify timer progression
        res = self.detector.update(
            vehicle_speed=0.0,
            imu_data=imu_rest,
            now_mono=t0 + 25.5,
        )
        self.assertEqual(res["state"], "STOPPED")
        self.assertAlmostEqual(res["duration_sec"], 25.0, places=1)

    # -----------------------------------------------------------------------
    # Condition 2: No timeout — absence of motion does NOT end stop
    # -----------------------------------------------------------------------
    def test_no_camera_motion_timeout(self):
        """Verify that absence of camera motion does NOT close the stop event."""
        t0 = 1000.0
        imu_rest = self._make_imu()

        # Enter STOPPED state
        self.detector.update(0.0, imu_rest, now_mono=t0)
        self.detector.update(0.0, imu_rest, now_mono=t0 + 0.6)
        self.assertTrue(self.detector.is_stopped)

        # Simulate 10 seconds of no camera motion (the old STOP_IDLE_TIMEOUT)
        res_10s = self.detector.update(0.0, imu_rest, now_mono=t0 + 10.6)
        self.assertTrue(res_10s["is_stopped"])
        self.assertFalse(res_10s["event_just_ended"])
        self.assertAlmostEqual(res_10s["duration_sec"], 10.0, places=1)

        # Simulate 60 seconds of no camera motion
        res_60s = self.detector.update(0.0, imu_rest, now_mono=t0 + 60.6)
        self.assertTrue(res_60s["is_stopped"])
        self.assertAlmostEqual(res_60s["duration_sec"], 60.0, places=1)

        # Simulate 300 seconds of no camera motion
        res_300s = self.detector.update(0.0, imu_rest, now_mono=t0 + 300.6)
        self.assertTrue(res_300s["is_stopped"])
        self.assertAlmostEqual(res_300s["duration_sec"], 300.0, places=1)

    # -----------------------------------------------------------------------
    # Condition 2: Keep logs of motion frames captured during stop
    # -----------------------------------------------------------------------
    def test_record_motion_captures_during_stop(self):
        """Verify motion frames captured during the stop are tracked and logged."""
        t0 = 2000.0
        imu_rest = self._make_imu()

        # Confirm stop
        self.detector.update(0.0, imu_rest, now_mono=t0)
        self.detector.update(0.0, imu_rest, now_mono=t0 + 0.5)

        # Capture #1 at +10s
        cap1 = self.detector.record_capture({
            "saved_count": 1,
            "image_file": "motion_RTC_2026_09_08_0001.jpg",
            "local_path": "/captures/motion_RTC_2026_09_08_0001.jpg",
            "rtc_timestamp": "2026-09-08 14:00:10",
            "latitude": 20.932014,
            "longitude": 77.752319,
        }, now_mono=t0 + 10.5)
        self.assertIsNotNone(cap1)
        self.assertEqual(cap1.stop_frame_idx, 1)
        self.assertEqual(cap1.filename, "motion_RTC_2026_09_08_0001.jpg")
        self.assertAlmostEqual(cap1.stop_offset_sec, 10.0, places=1)

        # Capture #2 at +25s
        cap2 = self.detector.record_capture({
            "saved_count": 2,
            "image_file": "motion_RTC_2026_09_08_0002.jpg",
            "local_path": "/captures/motion_RTC_2026_09_08_0002.jpg",
            "rtc_timestamp": "2026-09-08 14:00:25",
            "latitude": 20.932014,
            "longitude": 77.752319,
        }, now_mono=t0 + 25.5)
        self.assertIsNotNone(cap2)
        self.assertEqual(cap2.stop_frame_idx, 2)
        self.assertAlmostEqual(cap2.stop_offset_sec, 25.0, places=1)

        # Verify active stop event capture tracking
        active = self.detector.get_active_stop()
        self.assertEqual(len(active.captures), 2)
        self.assertEqual(active.captures[0].filename, "motion_RTC_2026_09_08_0001.jpg")
        self.assertEqual(active.captures[1].filename, "motion_RTC_2026_09_08_0002.jpg")

    # -----------------------------------------------------------------------
    # Condition 3: Vehicle back in motion (speed > 5 km/h via GNSS & IMU)
    # -----------------------------------------------------------------------
    def test_vehicle_back_in_motion_ends_stop_and_mentions_logs(self):
        """Condition 3: Speed > 5.0 km/h ends stop and outputs complete summary of previous logs."""
        t0 = 3000.0
        imu_rest = self._make_imu()

        # 1. Enter STOPPED state
        self.detector.update(0.0, imu_rest, latitude=20.9320, longitude=77.7523, rtc_timestamp="2026-09-08 14:10:00", now_mono=t0)
        self.detector.update(0.0, imu_rest, latitude=20.9320, longitude=77.7523, rtc_timestamp="2026-09-08 14:10:01", now_mono=t0 + 0.5)
        self.assertTrue(self.detector.is_stopped)

        # 2. Record 2 motion frame captures during stop
        self.detector.record_capture({
            "saved_count": 1,
            "image_file": "motion_RTC_141012_0001.jpg",
            "local_path": "/tmp/1.jpg",
            "rtc_timestamp": "2026-09-08 14:10:12",
            "latitude": 20.9320,
            "longitude": 77.7523,
        }, now_mono=t0 + 12.5)
        self.detector.record_capture({
            "saved_count": 2,
            "image_file": "motion_RTC_141028_0002.jpg",
            "local_path": "/tmp/2.jpg",
            "rtc_timestamp": "2026-09-08 14:10:28",
            "latitude": 20.9320,
            "longitude": 77.7523,
        }, now_mono=t0 + 28.5)

        # 3. Vehicle resumes motion: speed = 7.5 km/h (> 5 km/h) with IMU motion
        imu_moving = self._make_imu(accel_mag=11.2, gx=4.5, gy=6.0, gz=2.0)
        t_resume = t0 + 45.0

        # Motion tick 1 (within debounce)
        res = self.detector.update(
            vehicle_speed=7.5,
            imu_data=imu_moving,
            latitude=20.9325,
            longitude=77.7528,
            rtc_timestamp="2026-09-08 14:10:45",
            now_mono=t_resume,
        )
        self.assertTrue(res["is_stopped"])  # Still stopped during debounce

        # Motion tick 2 (debounce met: 0.3s)
        res = self.detector.update(
            vehicle_speed=8.2,
            imu_data=imu_moving,
            latitude=20.9326,
            longitude=77.7529,
            rtc_timestamp="2026-09-08 14:10:45",
            now_mono=t_resume + 0.35,
        )
        self.assertEqual(res["state"], "MOVING")
        self.assertFalse(res["is_stopped"])
        self.assertTrue(res["event_just_ended"])
        self.assertIsNotNone(res["last_completed_event"])

        completed = res["last_completed_event"]
        self.assertEqual(completed.stop_id, 1)
        self.assertAlmostEqual(completed.duration_sec, 44.85, delta=0.5)
        self.assertEqual(len(completed.captures), 2)
        self.assertIn("Vehicle back in motion", completed.end_reason)
        self.assertIn("8.2 km/h", completed.end_reason)

        # 4. Verify that all previous logs are mentioned in the summary report
        summary = completed.format_summary()
        self.assertIn("Stop #1 Summary Report", summary)
        self.assertIn("2026-09-08 14:10:01", summary)
        self.assertIn("2026-09-08 14:10:45", summary)
        self.assertIn("motion_RTC_141012_0001.jpg", summary)
        self.assertIn("motion_RTC_141028_0002.jpg", summary)
        self.assertIn("Total Stopped Duration", summary)
        self.assertIn("Motion Frames Captured: 2 frame(s)", summary)

        # Verify history preservation
        self.assertEqual(len(self.detector.get_history()), 1)

    # -----------------------------------------------------------------------
    # Sensor Fallbacks & Edge Cases
    # -----------------------------------------------------------------------
    def test_gnss_only_fallback_when_imu_unavailable(self):
        """Verify fallback when IMU is not present or invalid."""
        t0 = 4000.0
        # IMU is invalid
        invalid_imu = {"valid": False}

        # Rest via GNSS speed = 1.0 km/h
        self.detector.update(1.0, invalid_imu, now_mono=t0)
        res = self.detector.update(1.0, invalid_imu, now_mono=t0 + 0.6)
        self.assertTrue(res["is_stopped"])

        # Resume motion via GNSS speed = 7.0 km/h (> 5 km/h)
        self.detector.update(7.0, invalid_imu, now_mono=t0 + 10.0)
        res = self.detector.update(7.0, invalid_imu, now_mono=t0 + 10.4)
        self.assertFalse(res["is_stopped"])
        self.assertEqual(res["state"], "MOVING")

    def test_imu_only_fallback_when_gnss_lost(self):
        """Verify inertial fallback when GNSS fix is lost."""
        t0 = 5000.0
        imu_rest = self._make_imu(accel_mag=9.80665)
        imu_moving = self._make_imu(accel_mag=12.0, gx=6.0, gy=8.0, gz=5.0)

        # Rest via IMU while GPS has no fix
        self.detector.update(0.0, imu_rest, gps_valid=False, now_mono=t0)
        res = self.detector.update(0.0, imu_rest, gps_valid=False, now_mono=t0 + 0.6)
        self.assertTrue(res["is_stopped"])

        # Motion via IMU while GPS has no fix
        self.detector.update(0.0, imu_moving, gps_valid=False, now_mono=t0 + 10.0)
        res = self.detector.update(0.0, imu_moving, gps_valid=False, now_mono=t0 + 10.4)
        self.assertFalse(res["is_stopped"])
        self.assertEqual(res["state"], "MOVING")

    def test_transient_noise_debounce(self):
        """Verify that single-sample sensor glitches do not cause false stops or false resumes."""
        t0 = 6000.0
        imu_moving = self._make_imu(accel_mag=11.5, gx=5.0)
        imu_rest = self._make_imu()

        # Moving normally
        self.detector.update(25.0, imu_moving, now_mono=t0)
        self.assertEqual(self.detector.state, "MOVING")

        # Instantaneous 0 km/h glitch for 0.1s (< 0.5s debounce)
        self.detector.update(0.0, imu_rest, now_mono=t0 + 0.1)
        self.assertEqual(self.detector.state, "MOVING")

        # Returns to normal driving speed
        self.detector.update(25.0, imu_moving, now_mono=t0 + 0.2)
        self.assertEqual(self.detector.state, "MOVING")
        self.assertEqual(self.detector._current_stop_id, 0)

    def test_main_module_integration(self):
        """Verify that main.py integrates VehicleStopDetector cleanly without symbol errors."""
        self.assertTrue(hasattr(edge_main, "VehicleStopDetector"))
        self.assertTrue(hasattr(edge_main, "overlay_metadata"))
        self.assertTrue(hasattr(edge_main, "get_rtc_timestamp"))


if __name__ == "__main__":
    unittest.main()
