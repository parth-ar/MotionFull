"""
test_gnss_parsing.py — NMEA parsing pipeline of sensors/gnss.py after the redundancy cleanup:
  - position/fix from GGA only, speed/course from RMC only, VTG/GLL/TXT skipped,
  - GSV → satellites-in-view count only (per-satellite records only with GNSS_SATELLITE_DETAIL),
  - a serial port that keeps erroring is reopened instead of looping silently.
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

_EDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDGE_DIR not in sys.path:
    sys.path.insert(0, _EDGE_DIR)

import config as _cfg
import sensors.gnss as g


def mk(body):
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return "$%s*%02X" % (body, cs)


GGA = mk("GNGGA,104207.00,2055.92074,N,07745.13874,E,1,11,0.90,340.2,M,-63.1,M,,")
RMC = mk("GNRMC,104207.00,A,2055.92074,N,07745.13874,E,6.696,91.30,191126,,,A,V")
VTG = mk("GNVTG,45.00,T,,M,1.000,N,99.999,K,A")          # deliberately different values
GLL = mk("GNGLL,2055.92074,N,07745.13874,E,104207.00,V,N")  # status V during a valid fix
GSA = [mk("GNGSA,A,3,02,05,07,,,,,,,,,,1.60,0.90,1.32,1")]


def gsv_sequence(talker, count):
    n = (count + 3) // 4
    out = []
    for m in range(1, n + 1):
        sats = ",".join("%02d,40,120,35" % p for p in range(m * 4 - 3, min(m * 4, count) + 1))
        out.append(mk("%sGSV,%d,%d,%d,%s,1" % (talker, n, m, count, sats)))
    return out


GSV = gsv_sequence("GP", 12) + gsv_sequence("GL", 8) + gsv_sequence("GA", 9) + gsv_sequence("GB", 10)


class TestGnssParsing(unittest.TestCase):
    def setUp(self):
        g._reset_state_to_disconnected()

    def feed(self, sentences):
        return [g._dispatch(s) for s in sentences]

    def test_full_epoch(self):
        handled = self.feed([RMC, VTG, GGA, GLL] + GSA + GSV)
        d = g.read()
        self.assertEqual(d["status"], "FIX")
        self.assertAlmostEqual(d["latitude"], 20.932012, places=5)
        self.assertAlmostEqual(d["longitude"], 77.752312, places=5)
        self.assertAlmostEqual(d["speed_kmh"], 6.696 * 1.852, places=2, msg="speed from RMC, not VTG")
        self.assertEqual(d["course_deg"], 91.3, "course from RMC, not VTG")
        self.assertEqual(d["satellites"], 11)
        self.assertEqual(d["hdop"], 0.9)
        self.assertEqual(d["satellites_in_view"], 12 + 8 + 9 + 10)
        self.assertIsNone(d["satellites_detail"], "no per-satellite records in production")
        self.assertNotIn("primary_module", d)
        self.assertFalse(handled[1] or handled[3], "VTG and GLL are skipped")
        self.assertFalse(handled[4], "GSA skipped without detail")

    def test_rmc_alone_does_not_create_a_fix(self):
        self.feed([RMC])
        d = g.read()
        self.assertFalse(d["fix"])
        self.assertIsNone(d["latitude"])
        self.assertIsNotNone(d["speed_kmh"])

    def test_gll_status_v_cannot_clear_fix(self):
        self.feed([GGA, GLL])
        self.assertTrue(g.read()["fix"])

    def test_bad_checksum_rejected(self):
        self.assertFalse(g._dispatch(GGA[:-2] + "00"))
        self.assertFalse(g.read()["data_received"])

    def test_detail_mode_builds_records(self):
        with patch.object(_cfg, "GNSS_SATELLITE_DETAIL", True):
            self.feed(GSA + [GGA] + GSV)
            d = g.read()
        self.assertEqual(len(d["satellites_detail"]), 39)
        used = {s["prn"] for s in d["satellites_detail"] if s["used_in_fix"]}
        self.assertTrue({2, 5, 7} <= used, "GSA parsed in detail mode")

    def test_skipped_sentences_are_cheaper(self):
        n = 3000
        t0 = time.perf_counter()
        for _ in range(n):
            g._dispatch(VTG)
        skipped = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            g._dispatch(GGA)
        parsed = time.perf_counter() - t0
        self.assertLess(skipped, parsed / 3)


import types as _types

if g.HAVE_PYSERIAL:
    _serial_mod = g.serial
else:   # dev PC without pyserial: minimal stand-in exposing SerialException
    _serial_mod = _types.SimpleNamespace(SerialException=type("SerialException", (OSError,), {}))


class FakeFailingSerial:
    def __init__(self):
        self.closed = False

    def readline(self):
        time.sleep(0.005)
        raise _serial_mod.SerialException("device reports readiness to read but returned no data")

    @property
    def in_waiting(self):
        return 0

    def close(self):
        self.closed = True


class TestStuckPortReconnect(unittest.TestCase):
    def test_reopens_port_after_continuous_errors(self):
        opened = []

        def fake_connect(port, verbose=True):
            s = FakeFailingSerial()
            opened.append(s)
            return s

        with patch.object(g, "_connect_and_configure_baud", side_effect=fake_connect), \
             patch.object(g, "configure_receiver", return_value={"rate_hz": 5.0}), \
             patch.object(g, "RECONNECT_DELAY", 0.05), \
             patch.object(_cfg, "GNSS_SERIAL_ERROR_RECONNECT_SEC", 0.2, create=True), \
             patch.object(g.os.path, "exists", return_value=True), \
             patch.object(g, "_target_port", "/dev/fake-gnss"), \
             patch.object(g, "_set_thread_priority_high"), \
             patch.object(g, "HAVE_PYSERIAL", True), \
             patch.object(g, "serial", _serial_mod, create=True):
            g._stop_event.clear()
            t = threading.Thread(target=g._serial_thread, daemon=True)
            t.start()
            time.sleep(1.0)
            g._stop_event.set()
            t.join(timeout=2.0)

        self.assertGreaterEqual(len(opened), 2, "port must be reopened after continuous errors")
        self.assertTrue(opened[0].closed)


if __name__ == "__main__":
    unittest.main()
