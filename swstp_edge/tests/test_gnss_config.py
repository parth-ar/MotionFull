"""
test_gnss_config.py — u-blox receiver configuration (rate + NMEA sentence filter) against a
simulated receiver that ACKs / NAKs UBX commands.

Background: on the deployed MAX-M10S the legacy UBX-CFG-RATE command was ignored
(GGA measured at 1.0/s), so configuration now uses UBX-CFG-VALSET with ACK checking.
"""

import os
import sys
import unittest
from unittest.mock import patch

_EDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDGE_DIR not in sys.path:
    sys.path.insert(0, _EDGE_DIR)

import config as _cfg
import sensors.gnss as g


def _mon_ver_payload():
    pad = lambda s, n: s.encode().ljust(n, b"\x00")
    return (pad("ROM SPG 5.10 (7b202e)", 30) + pad("000A0000", 10)
            + pad("FWVER=SPG 5.10", 30) + pad("PROTVER=34.10", 30) + pad("MOD=MAX-M10S", 30))


class FakeReceiver:
    """Minimal serial port: answers UBX frames according to `policy(cls, id, payload)`."""
    def __init__(self, policy):
        self.policy = policy
        self.rx = bytearray(b"$GNTXT,01,01,02,noise*00\r\n")   # some NMEA noise in the buffer
        self.sent = []

    @property
    def in_waiting(self):
        return len(self.rx)

    def write(self, data):
        data = bytes(data)
        self.sent.append(data)
        cls, mid = data[2], data[3]
        plen = data[4] | (data[5] << 8)
        payload = data[6:6 + plen]
        if (cls, mid) == (0x0A, 0x04):
            self.rx += g._ubx_frame(0x0A, 0x04, _mon_ver_payload())
            return
        answer = self.policy(cls, mid, payload)
        if answer in ("ack", "nak"):
            self.rx += b"$GNGGA,,,,,,0,00,99.99,,,,,,*56\r\n"   # NMEA interleaved before the ACK
            self.rx += g._ubx_frame(0x05, 0x01 if answer == "ack" else 0x00, bytes([cls, mid]))

    def read(self, n):
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def flush(self):
        pass

    def cfg_frames(self):
        return [f for f in self.sent if f[2] == 0x06]


def _decode_valset(frame):
    plen = frame[4] | (frame[5] << 8)
    p = frame[6:6 + plen]
    items, i = {}, 4
    sizes = {0x3: 2, 0x2: 1}   # key size nibble: 0x3 = 2 bytes, 0x2 = 1 byte
    while i < len(p):
        key = int.from_bytes(p[i:i + 4], "little")
        size = sizes[(key >> 28) & 0x7]
        items[key] = int.from_bytes(p[i + 4:i + 4 + size], "little")
        i += 4 + size
    return p[1], items      # layers, {key: value}


class TestGnssConfig(unittest.TestCase):
    def test_valset_ack_sets_rate_and_filter(self):
        rx = FakeReceiver(lambda c, m, p: "ack")
        res = g.configure_receiver(rx)
        self.assertEqual((res["rate_ok"], res["filter_ok"], res["method"]), (True, True, "VALSET"))
        frame = rx.cfg_frames()[0]
        self.assertEqual(frame[2:4], b"\x06\x8a")
        self.assertEqual(g._ubx_checksum(frame[2:-2]), (frame[-2], frame[-1]), "valid checksum")
        layers, items = _decode_valset(frame)
        self.assertEqual(layers, 0x01, "RAM only — never written to flash/BBR")
        self.assertEqual(items[g._CFG_RATE_MEAS], 200)
        self.assertEqual(items[g._CFG_RATE_NAV], 1)
        out = {name: items[key] for name, key in g._CFG_MSGOUT_UART1.items()}
        self.assertEqual(out, {"GGA": 1, "RMC": 1, "GSV": 5, "VTG": 0, "GLL": 0, "GSA": 0})

    def test_filter_rejected_falls_back_to_rate_only(self):
        def policy(c, m, p):
            return "nak" if len(p) > 4 + 2 * 6 else "ack"    # NAK the long (rate+filter) VALSET
        rx = FakeReceiver(policy)
        res = g.configure_receiver(rx)
        self.assertEqual((res["rate_ok"], res["filter_ok"], res["method"]), (True, False, "VALSET"))
        self.assertEqual(len(rx.cfg_frames()), 2)

    def test_valset_rejected_falls_back_to_legacy(self):
        rx = FakeReceiver(lambda c, m, p: "ack" if m == 0x08 else "nak")
        res = g.configure_receiver(rx)
        self.assertEqual((res["rate_ok"], res["method"]), (True, "CFG-RATE"))
        legacy = rx.cfg_frames()[-1]
        self.assertEqual(legacy, bytes([0xB5, 0x62, 0x06, 0x08, 0x06, 0x00, 0xC8, 0x00,
                                        0x01, 0x00, 0x01, 0x00, 0xDE, 0x6A]),
                         "same bytes as the previous hard-coded command")

    def test_no_answer_reports_failure(self):
        rx = FakeReceiver(lambda c, m, p: None)
        res = g.configure_receiver(rx)
        self.assertFalse(res["rate_ok"])
        self.assertEqual(len(rx.cfg_frames()), 3, "tried VALSET+filter, VALSET, legacy")

    def test_filter_disabled_in_config(self):
        rx = FakeReceiver(lambda c, m, p: "ack")
        with patch.object(_cfg, "GNSS_NMEA_FILTER", False):
            res = g.configure_receiver(rx)
        self.assertFalse(res["filter_ok"])
        _, items = _decode_valset(rx.cfg_frames()[0])
        self.assertEqual(set(items), {g._CFG_RATE_MEAS, g._CFG_RATE_NAV})

    def test_rate_from_config(self):
        rx = FakeReceiver(lambda c, m, p: "ack")
        with patch.object(_cfg, "GNSS_RATE_HZ", 2.0):
            g.configure_receiver(rx)
        _, items = _decode_valset(rx.cfg_frames()[0])
        self.assertEqual(items[g._CFG_RATE_MEAS], 500)
        self.assertEqual(items[g._CFG_MSGOUT_UART1["GSV"]], 2, "GSV stays at ~1 Hz")

    def test_firmware_version_parsed(self):
        res = g.configure_receiver(FakeReceiver(lambda c, m, p: "ack"))
        self.assertEqual(res["firmware"], "SPG 5.10 / PROTVER=34.10 / MOD=MAX-M10S")

    def test_measured_rate_check(self):
        g._start_rate_check(5.0)
        g._rate_check_start -= g._RATE_CHECK_SEC
        g._gga_count = int(5 * g._RATE_CHECK_SEC)
        g._maybe_report_rate()
        self.assertAlmostEqual(g.measured_rate_hz, 5.0, delta=0.1)

        g._start_rate_check(5.0)                   # receiver stuck at 1 Hz
        g._rate_check_start -= g._RATE_CHECK_SEC
        g._gga_count = int(1 * g._RATE_CHECK_SEC)
        with patch("builtins.print") as p:
            g._maybe_report_rate()
        self.assertIn("WARNING", p.call_args.args[0])

    def test_gga_dispatch_counts_epochs(self):
        g._start_rate_check(5.0)
        def mk(body):
            cs = 0
            for ch in body:
                cs ^= ord(ch)
            return "$%s*%02X" % (body, cs)
        for _ in range(3):
            g._dispatch(mk("GNGGA,104207.00,2055.92074,N,07745.13874,E,1,11,0.90,340.2,M,-63.1,M,,"))
        g._dispatch(mk("GNRMC,104207.00,A,2055.92074,N,07745.13874,E,6.696,91.30,191126,,,A,V"))
        self.assertEqual(g._gga_count, 3)


if __name__ == "__main__":
    unittest.main()
