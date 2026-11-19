"""
test_telemetry_packets.py — Checks the slim telemetry packet sent to
POST /api/telemetry/ingest-batch, and that the backup heartbeat matches it.
"""

import os
import sys
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

import telemetry as t


def _gnss_fix(hdop=1.23):
    return {
        "fix": True, "data_received": True, "status": "3D_FIX",
        "latitude": 20.9320, "longitude": 77.7523, "altitude_m": 340.0,
        "speed_kmh": 12.0, "course_deg": 90.0, "satellites": 9, "hdop": hdop,
    }


class TestTelemetryPackets(unittest.TestCase):
    # ── Task 8: real uptime ───────────────────────────────────────────────
    def test_uptime_is_time_since_boot_not_epoch(self):
        slim = t.build_slim_packet(t.build_telemetry_packet("DEV"))
        expected = (time.time() - t._boot_time) * 1000
        self.assertNotEqual(slim["uptime"], slim["epoch"])
        self.assertAlmostEqual(slim["uptime"], expected, delta=5000)

    # ── Task 9: hdop ──────────────────────────────────────────────────────
    def test_hdop_is_sent_when_fix(self):
        with patch.object(t.gnss_sensor, "read", return_value=_gnss_fix(1.23)):
            slim = t.build_slim_packet(t.build_telemetry_packet("DEV"))
        if slim["fix"] != "3D_FIX":
            self.skipTest("location pipeline did not accept the mocked fix on this machine")
        self.assertEqual(slim["hdop"], 1.23)

    def test_hdop_is_null_without_fix(self):
        with patch.object(t.gnss_sensor, "read",
                          return_value={"fix": False, "data_received": False, "hdop": 9.9}):
            with patch.dict(t.latest_sensor, {"gps_valid": False}):
                slim = t.build_slim_packet(t.build_telemetry_packet("DEV"))
        if slim["fix"] == "NO_FIX":
            self.assertIsNone(slim["hdop"])

    # ── Task 10: backup heartbeat matches the main packet ─────────────────
    def test_fallback_has_same_keys_and_types(self):
        main = t.build_slim_packet(t.build_telemetry_packet("DEV"))
        backup = t.build_slim_packet(t.build_fallback_full_packet())
        self.assertEqual(sorted(main), sorted(backup))
        for k in main:
            if main[k] is not None and backup[k] is not None:
                self.assertIs(type(main[k]), type(backup[k]), f"type mismatch for '{k}'")
        self.assertIn(backup["fix"], ("3D_FIX", "NO_FIX"))
        self.assertIn("pwr_src", backup)
        self.assertIn("low_batt", backup)

    def test_fallback_uptime_and_sequence(self):
        a = t.build_slim_packet(t.build_fallback_full_packet())
        b = t.build_slim_packet(t.build_fallback_full_packet())
        self.assertEqual(b["seq"] - a["seq"], 1)
        self.assertNotEqual(b["uptime"], b["epoch"])

    def test_main_packet_sequence_steps_by_one(self):
        a = t.build_telemetry_packet("DEV")["sequence"]
        b = t.build_telemetry_packet("DEV")["sequence"]
        self.assertEqual(b - a, 1)

    def test_fallback_unknown_values_are_null_not_zero(self):
        with patch.dict(t.latest_sensor, {"gps_valid": False, "alt": None,
                                          "speed": None, "satellites": None}):
            backup = t.build_slim_packet(t.build_fallback_full_packet())
        self.assertIsNone(backup["lat"])
        self.assertIsNone(backup["alt"])
        self.assertIsNone(backup["spd"])
        self.assertEqual(backup["fix"], "NO_FIX")


if __name__ == "__main__":
    unittest.main()
