"""
test_imu_recovery.py — IMU driver resilience and calibration with a simulated MPU-6500
on a fake smbus2.

Recovery:
  - a failed re-init no longer disables the IMU permanently,
  - an IMU missing at boot is picked up (and calibrated) when it appears,
  - read() never blocks the 20 Hz telemetry loop while a re-init is running.
Calibration (boot may happen while the vehicle is moving):
  - option 1: moving / turning boots are rejected,
  - option 2: saved offsets are loaded and used at once,
  - option 3: re-calibration at GNSS-confirmed stops (blended into a trusted baseline),
  - option 4: implausible offsets are rejected.
"""

import importlib
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import time
import types
import unittest

_EDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDGE_DIR not in sys.path:
    sys.path.insert(0, _EDGE_DIR)

ACCEL_LSB, GYRO_LSB = 16384.0, 131.0


class FakeChip:
    """Simulated sensor. Values in physical units; converted to raw registers on read."""
    def __init__(self, present=True):
        self.present = present
        self.gyro_bias  = (0.2, -0.1, 0.05)   # °/s — true chip bias
        self.rotation   = (0.0, 0.0, 0.0)     # °/s — real rotation (turning)
        self.accel_extra = (0.0, 0.0, 0.0)    # g   — real acceleration (braking)
        self.noise_g    = 0.0                 # accel noise std (vibration / driving)
        self.noise_dps  = 0.0                 # gyro noise std
        self.rng = random.Random(1)

    def sample(self):
        r = self.rng
        a = [self.accel_extra[0], self.accel_extra[1], 1.0 + self.accel_extra[2]]
        g = [self.gyro_bias[i] + self.rotation[i] for i in range(3)]
        a = [v + r.gauss(0, self.noise_g) for v in a] if self.noise_g else a
        g = [v + r.gauss(0, self.noise_dps) for v in g] if self.noise_dps else g
        clamp = lambda v: max(-32768, min(32767, int(round(v))))
        raw = [clamp(v * ACCEL_LSB) for v in a] + [0] + [clamp(v * GYRO_LSB) for v in g]
        return list(struct.pack(">hhhhhhh", *raw))


class FakeBus:
    def __init__(self, chip):
        self.chip = chip
        self.closed = False

    def _check(self):
        if self.closed or not self.chip.present:
            raise OSError(121, "Remote I/O error")

    def write_byte_data(self, addr, reg, val):
        self._check()

    def read_byte_data(self, addr, reg):
        self._check()
        return 0x70 if reg == 0x75 else 0

    def read_i2c_block_data(self, addr, reg, n):
        self._check()
        return self.chip.sample()

    def close(self):
        self.closed = True


class ImuTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="swstp_imu_")
        self.cal_file = os.path.join(self.tmp, "imu_calibration.json")

    def tearDown(self):
        sys.modules.pop("smbus2", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load_imu(self, chip):
        fake = types.ModuleType("smbus2")
        fake.SMBus = lambda n: FakeBus(chip)
        sys.modules["smbus2"] = fake
        import sensors.imu as imu
        imu = importlib.reload(imu)
        imu._RECOVERY_INTERVAL_SEC = 0.05
        imu._CALIB_DELAY = 0.0
        imu._CAL_FILE = self.cal_file
        return imu

    def write_saved(self, gyro=(0.2, -0.1, 0.05), trusted=True):
        with open(self.cal_file, "w", encoding="utf-8") as f:
            json.dump({"offsets": {"ax": 0.0, "ay": 0.0, "az": 0.0,
                                   "gx": gyro[0], "gy": gyro[1], "gz": gyro[2]},
                       "source": "stop", "trusted": trusted}, f)

    @staticmethod
    def wait_for(pred, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            time.sleep(0.01)
        return False


class TestImuRecovery(ImuTestBase):
    def test_normal_read(self):
        imu = self.load_imu(FakeChip())
        self.assertTrue(imu.init())
        data = imu.read()
        self.assertTrue(data["valid"])
        self.assertAlmostEqual(data["accel_magnitude_ms2"], 9.80665, places=2)
        self.assertAlmostEqual(data["gyro_dps"]["x"], 0.0, places=1)   # bias removed
        self.assertIn("quaternion", data)   # still used by test_all_sensors.py

    def test_recovers_after_a_failed_reinit(self):
        chip = FakeChip()
        imu = self.load_imu(chip)
        self.assertTrue(imu.init())
        chip.present = False
        for _ in range(25):
            imu.read()
        self.assertTrue(self.wait_for(lambda: not imu._recovery_running() and not imu.imu_ok))
        self.assertGreaterEqual(imu._recovery_attempts, 1, "first re-init attempt should have failed")
        chip.present = True
        self.assertTrue(self.wait_for(lambda: imu.read().get("valid") is True),
                        "IMU must recover after a failed re-init")

    def test_missing_at_boot_is_picked_up_later(self):
        chip = FakeChip(present=False)
        imu = self.load_imu(chip)
        self.assertFalse(imu.init())
        self.assertFalse(imu.read()["valid"])
        chip.present = True
        self.assertTrue(self.wait_for(lambda: imu.read().get("valid") is True))
        self.assertTrue(imu._calibrated)

    def test_read_never_blocks_during_reinit(self):
        chip = FakeChip()
        imu = self.load_imu(chip)
        self.assertTrue(imu.init())
        chip.present = False
        for _ in range(25):
            imu.read()
        chip.present = True
        worst = 0.0
        end = time.monotonic() + 0.4
        while time.monotonic() < end:
            t0 = time.perf_counter()
            imu.read()
            worst = max(worst, time.perf_counter() - t0)
            time.sleep(0.01)
        self.assertLess(worst, 0.05, f"read() blocked for {worst*1000:.0f} ms")

    def test_no_retry_without_smbus(self):
        sys.modules["smbus2"] = None          # import smbus2 → ImportError (bench machine)
        import sensors.imu as imu
        imu = importlib.reload(imu)
        imu._CAL_FILE = self.cal_file
        self.assertFalse(imu.init())
        imu.read()
        self.assertIsNone(imu._recovery_thread)


class TestImuCalibration(ImuTestBase):
    # ── Option 1: stillness check ──────────────────────────────────────────
    def test_still_boot_is_accepted_and_saved(self):
        chip = FakeChip()
        chip.noise_g, chip.noise_dps = 0.005, 0.1        # idling engine vibration
        imu = self.load_imu(chip)
        self.assertTrue(imu.init())
        self.assertTrue(imu._calibrated)
        self.assertEqual(imu._cal_source, "boot")
        self.assertFalse(imu._cal_trusted, "first boot calibration is provisional")
        self.assertAlmostEqual(imu._offsets["gx"], 0.2, places=1)
        self.assertTrue(os.path.exists(self.cal_file))

    def test_bumpy_boot_without_saved_file_stays_invalid(self):
        chip = FakeChip()
        chip.noise_g, chip.noise_dps = 0.15, 6.0          # driving over a rough road
        imu = self.load_imu(chip)
        self.assertTrue(imu.init(), "chip is present")
        self.assertFalse(imu._calibrated)
        self.assertFalse(imu.read()["valid"], "no bad baseline: readings invalid instead")

    def test_uncalibrated_retries_and_calibrates_when_still(self):
        chip = FakeChip()
        chip.noise_g, chip.noise_dps = 0.15, 6.0
        imu = self.load_imu(chip)
        imu.init()
        chip.noise_g = chip.noise_dps = 0.0               # truck stops
        imu._last_cal_attempt_ts = 0.0                    # skip the 30 s retry wait
        imu.read()
        self.assertTrue(self.wait_for(lambda: imu._calibrated))
        self.assertTrue(self.wait_for(lambda: imu.read().get("valid") is True))

    def test_steady_turn_rejected_against_trusted_baseline(self):
        self.write_saved(trusted=True)
        chip = FakeChip()
        chip.rotation = (0.0, 0.0, 5.0)                   # smooth 5 °/s turn: low spread
        imu = self.load_imu(chip)
        imu.init()
        self.assertEqual(imu._cal_source, "saved", "turn must not overwrite trusted offsets")
        self.assertAlmostEqual(imu._offsets["gz"], 0.05, places=3)

    # ── Option 2: saved calibration ────────────────────────────────────────
    def test_moving_boot_uses_saved_calibration(self):
        self.write_saved(trusted=True)
        chip = FakeChip()
        chip.noise_g, chip.noise_dps = 0.15, 6.0
        imu = self.load_imu(chip)
        imu.init()
        self.assertEqual(imu._cal_source, "saved")
        self.assertTrue(imu.read()["valid"], "usable immediately from saved offsets")

    def test_corrupt_saved_file_is_ignored(self):
        with open(self.cal_file, "w") as f:
            f.write("{not json")
        imu = self.load_imu(FakeChip())
        imu.init()
        self.assertEqual(imu._cal_source, "boot")

    # ── Option 4: sanity limits ────────────────────────────────────────────
    def test_implausible_gyro_offset_rejected(self):
        chip = FakeChip()
        chip.gyro_bias = (25.0, 0.0, 0.0)                 # no healthy chip does this
        imu = self.load_imu(chip)
        imu.init()
        self.assertFalse(imu._calibrated)

    def test_implausible_saved_file_ignored(self):
        self.write_saved(gyro=(40.0, 0.0, 0.0))
        imu = self.load_imu(FakeChip())
        imu.init()
        self.assertEqual(imu._cal_source, "boot")

    # ── Option 3: re-calibration at stops ──────────────────────────────────
    def _stop_for(self, imu, seconds=0.2):
        imu.note_vehicle_state(0.2, True)                 # start of stop
        imu._stop_since -= 10.0                           # pretend it began 10 s ago
        imu.note_vehicle_state(0.2, True)

    def test_stop_recal_replaces_provisional(self):
        chip = FakeChip()
        chip.rotation = (0.0, 0.0, 3.0)                   # booted in a gentle turn → bad gz
        imu = self.load_imu(chip)
        imu.init()
        self.assertAlmostEqual(imu._offsets["gz"], 3.05, places=1)
        chip.rotation = (0.0, 0.0, 0.0)                   # now genuinely stopped
        self._stop_for(imu)
        self.assertTrue(self.wait_for(lambda: imu._cal_source == "stop"))
        self.assertAlmostEqual(imu._offsets["gz"], 0.05, places=2, msg="bad boot offset corrected")
        self.assertTrue(imu._cal_trusted)

    def test_stop_recal_blends_into_trusted(self):
        self.write_saved(gyro=(0.2, -0.1, 0.05), trusted=True)
        chip = FakeChip()
        chip.gyro_bias = (0.6, -0.1, 0.05)                # bias drifted with temperature
        chip.rotation = (0.0, 0.0, 0.0)
        imu = self.load_imu(chip)
        imu.init()                                        # boot: shift 0.4 < 1.5 → accepted, trusted
        imu._offsets = dict(imu._offsets, gx=0.2)         # reset to old baseline for the check
        self._stop_for(imu)
        self.assertTrue(self.wait_for(lambda: imu._cal_source == "stop"))
        self.assertAlmostEqual(imu._offsets["gx"], 0.2 + 0.3 * 0.4, places=2)

    def test_no_stop_recal_while_moving_or_without_fix(self):
        imu = self.load_imu(FakeChip())
        imu.init()
        for speed, fix in ((12.0, True), (0.2, False), (None, True)):
            imu.note_vehicle_state(speed, fix)
            self.assertIsNone(imu._stop_since)
        self.assertIsNone(imu._cal_thread)

    def test_stop_recal_rate_limited(self):
        imu = self.load_imu(FakeChip())
        imu.init()
        self._stop_for(imu)
        self.assertTrue(self.wait_for(lambda: not imu._calibration_running()))
        first = imu._cal_thread
        self._stop_for(imu)
        self.assertIs(imu._cal_thread, first, "second stop within 10 min must not recalibrate")


class TestStaleImuData(unittest.TestCase):
    """latest_sensor['imu'] must reflect a dead IMU instead of holding the last good sample."""
    def test_dead_imu_reported_invalid(self):
        from unittest.mock import patch
        import telemetry as t
        good = {"valid": True, "accel_magnitude_ms2": 9.81,
                "orientation": {"roll": 1.0, "pitch": 2.0, "yaw": 3.0}, "temperature_c": 30.0}
        with patch.object(t.imu_sensor, "read", return_value=good):
            self.assertTrue(t.build_slim_packet(t.build_telemetry_packet("DEV"))["imu_ok"])
        with patch.object(t.imu_sensor, "read", return_value={"valid": False}):
            slim = t.build_slim_packet(t.build_telemetry_packet("DEV"))
        self.assertFalse(slim["imu_ok"])
        self.assertEqual(slim["accel_mag"], 0.0)
        self.assertFalse(t.latest_sensor["imu"]["valid"], "stop detector / HUD must see the failure")


if __name__ == "__main__":
    unittest.main()
