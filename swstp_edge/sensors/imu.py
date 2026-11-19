"""
sensors/imu.py — MPU-6500 / MPU-6050 6-Axis IMU Hardware Driver for Raspberry Pi via smbus2.

DATA ORIGINS & HARDWARE SPECIFICATION
-------------------------------------
Communicates with the InvenSense MPU-6500 (or MPU-6050) over the Raspberry Pi hardware
I2C bus (I2C-1, physical pins 3/5: SDA=BCM 2, SCL=BCM 3).
Auto-probes primary I2C address 0x69 (AD0 pin pulled HIGH) with automatic fallback
to 0x68 (AD0 pulled LOW/GND) to guarantee rock-solid initialization across all breakout boards.

Register Map (MPU-6500 / MPU-6050):
  0x6B  PWR_MGMT_1         — Power management (Device reset: 0x80, Wake + PLL Clock: 0x01)
  0x68  SIGNAL_PATH_RESET  — Reset gyro/accel/temp digital signal paths (0x07)
  0x1A  CONFIG             — Digital Low Pass Filter (DLPF = 42 Hz: 0x03)
  0x1B  GYRO_CONFIG        — Full-scale range ±250 °/s (0x00)
  0x1C  ACCEL_CONFIG       — Full-scale range ±2 g (0x00)
  0x3B  ACCEL_XOUT_H       — Start of 14-byte continuous burst readout:
                             [0:1]   AX_H AX_L  (16-bit signed, big-endian)
                             [2:3]   AY_H AY_L  (16-bit signed, big-endian)
                             [4:5]   AZ_H AZ_L  (16-bit signed, big-endian)
                             [6:7]   T_H  T_L   (16-bit signed, die temperature)
                             [8:9]   GX_H GX_L  (16-bit signed, big-endian)
                             [10:11] GY_H GY_L  (16-bit signed, big-endian)
                             [12:13] GZ_H GZ_L  (16-bit signed, big-endian)
  0x75  WHO_AM_I           — Returns 0x70 (MPU-6500), 0x68 (MPU-6050), 0x71 (MPU-9250), 0x19 (ICM-20600/20602)

Physical Scale Conversions:
  - Accelerometer: ±2 g full scale → 16384 LSB/g (raw / 16384.0 = g; g * 9.80665 = m/s²)
  - Gyroscope:     ±250 °/s scale   → 131.0 LSB/(°/s) (raw / 131.0 = °/s)
  - Temperature:   T_C = raw / 333.87 + 21.0 °C (per InvenSense MPU-6500 Register Map §4.18)

Orientation Estimation:
  - Complementary Filter (α = 0.98, matches firmware updateOrientation):
      roll  = α * (roll  + gx * dt) + (1 - α) * accel_roll
      pitch = α * (pitch + gy * dt) + (1 - α) * accel_pitch
      yaw   = normalise(yaw + gz * dt)  [gyro integration]
  - Tait-Bryan Z-Y-X rotation transformed to unit quaternion [w, x, y, z].
"""

import json
import math
import os
import struct
import threading
import time
from typing import Optional, Tuple

import config as _cfg
from config import COMPLEMENTARY_ALPHA, TELEMETRY_INTERVAL_SEC


def _c(name: str, default):
    return getattr(_cfg, name, default)

# ---------------------------------------------------------------------------
# MPU-6500 / MPU-6050 Registers
# ---------------------------------------------------------------------------
_PRIMARY_ADDR = 0x69      # AD0 pulled HIGH (default for SWSTP)
_BACKUP_ADDR  = 0x68      # AD0 pulled LOW / GND

_SMPLRT_DIV   = 0x19
_CONFIG       = 0x1A      # DLPF config (0x03 = 42 Hz filter)
_GYRO_CFG     = 0x1B      # ±250 °/s (0x00)
_ACCEL_CFG    = 0x1C      # ±2 g (0x00)
_BURST_START  = 0x3B      # 14-byte block: accel(6) + temp(2) + gyro(6)
_SIG_RESET    = 0x68      # Signal path reset
_PWR_MGMT_1   = 0x6B      # Power management 1: reset (0x80), wake+PLL (0x01)
_WHO_AM_I     = 0x75

# Valid WHO_AM_I signatures:
#   0x70 = MPU-6500
#   0x68 = MPU-6050
#   0x71 = MPU-9250
#   0x73 = MPU-9255
#   0x19 = ICM-20600
#   0x11 = ICM-20602
_VALID_WHO_AM_I = frozenset({0x70, 0x68, 0x71, 0x73, 0x19, 0x11})

# Scale Factors
_ACCEL_SCALE = 16384.0    # LSB/g for ±2 g
_GYRO_SCALE  = 131.0      # LSB/(°/s) for ±250 °/s
_G_MS2       = 9.80665    # m/s² per g

# Retries & Timing
_INIT_RETRIES     = 3
_INIT_RETRY_DELAY = 0.050 # s
_READ_RETRIES     = 3
_READ_RETRY_DELAY = 0.005 # s
_CALIB_SAMPLES    = int(_c("IMU_CAL_SAMPLES", 100))
_CALIB_DELAY      = float(_c("IMU_CAL_SAMPLE_DELAY_SEC", 0.010))  # s
_CAL_FILE         = _c("IMU_CALIBRATION_FILE", None)

# ---------------------------------------------------------------------------
# Module State (Thread-Safe)
# ---------------------------------------------------------------------------
imu_ok: bool = False

_lock            = threading.Lock()
_i2c_lock        = threading.Lock()
_bus             = None   # smbus2.SMBus instance
_active_addr: int = _PRIMARY_ADDR

_roll_deg: float  = 0.0
_pitch_deg: float = 0.0
_yaw_deg: float   = 0.0
_last_ts: float   = 0.0

_last_valid_imu: Optional[dict] = None
_last_valid_ts: float = 0.0
_last_error_log_ts: float = 0.0
_last_reinit_ts: float = 0.0
_read_fail_count: int = 0

# Background recovery (never blocks the 20 Hz telemetry loop)
_RECOVERY_INTERVAL_SEC  = 5.0     # min seconds between re-init attempts
_RECOVERY_LOG_EVERY_SEC = 60.0    # repeat "still unavailable" log at most this often
_smbus_available: bool  = False   # False on bench machines without smbus2 → never retry
_calibrated: bool       = False   # usable offsets present (saved, boot or stop)
_cal_source: Optional[str] = None # "saved" | "boot" | "stop"
_cal_trusted: bool      = False   # True = learned at a GNSS-confirmed stop (or checked against one)
_cal_thread: Optional[threading.Thread] = None
_last_cal_attempt_ts: float = 0.0
_stop_since: Optional[float] = None
_last_stop_recal_ts: float  = float("-inf")
_recovery_thread: Optional[threading.Thread] = None
_recovery_lock          = threading.Lock()
_recovery_attempts: int = 0
_last_recovery_log_ts: float = 0.0

# Calibration offsets (in physical units: g for accel, °/s for gyro)
_offsets = {
    "ax": 0.0, "ay": 0.0, "az": 0.0,
    "gx": 0.0, "gy": 0.0, "gz": 0.0,
}


# ---------------------------------------------------------------------------
# Low-Level I2C Helpers
# ---------------------------------------------------------------------------

def _write_reg(bus, addr: int, reg: int, val: int, retries: int = 3) -> bool:
    """Safely write an 8-bit register with retries and lock."""
    with _i2c_lock:
        for attempt in range(1, retries + 1):
            try:
                bus.write_byte_data(addr, reg, val)
                return True
            except OSError:
                if attempt < retries:
                    time.sleep(0.005)
        return False


def _read_reg(bus, addr: int, reg: int, retries: int = 3) -> Optional[int]:
    """Safely read an 8-bit register with retries and lock."""
    with _i2c_lock:
        for attempt in range(1, retries + 1):
            try:
                return bus.read_byte_data(addr, reg)
            except OSError:
                if attempt < retries:
                    time.sleep(0.005)
        return None


def _burst_read_14(bus, addr: int) -> Tuple[int, int, int, int, int, int, int]:
    """
    Read 14 contiguous bytes starting at register 0x3B.
    Returns: (ax_raw, ay_raw, az_raw, t_raw, gx_raw, gy_raw, gz_raw)
    Protected by _i2c_lock against concurrent I2C bus contention.
    """
    with _i2c_lock:
        raw = bus.read_i2c_block_data(addr, _BURST_START, 14)
    return struct.unpack(">hhhhhhh", bytes(raw))


# ---------------------------------------------------------------------------
# Chip Reset & Configuration Sequence
# ---------------------------------------------------------------------------

def _probe_and_configure_chip(bus) -> Optional[Tuple[int, int]]:
    """
    Probe candidate I2C addresses (0x69, then 0x68).
    Issues a hardware DEVICE_RESET (0x80 to PWR_MGMT_1), wakes the chip with
    gyro PLL clock source (0x01), and configures DLPF, accel (±2g), and gyro (±250°/s).
    Returns (active_address, who_am_i) on success, or None on failure.
    """
    candidate_addrs = (_PRIMARY_ADDR, _BACKUP_ADDR)

    for addr in candidate_addrs:
        try:
            # 1. Hardware device reset: write 0x80 to PWR_MGMT_1 (bit 7 = H_RESET)
            # Clears any bad internal state from power hiccups or previous unclean script shutdowns.
            if not _write_reg(bus, addr, _PWR_MGMT_1, 0x80, retries=2):
                continue
            time.sleep(0.100)  # Wait 100 ms for reset sequence to settle

            # 2. Wake chip & select PLL clock: write 0x01 to PWR_MGMT_1
            # (CLKSEL=1: Auto-selects gyro PLL clock for maximum timing stability per datasheet §4.3)
            if not _write_reg(bus, addr, _PWR_MGMT_1, 0x01, retries=3):
                continue
            time.sleep(0.030)  # Wait for PLL clock to lock

            # 3. Read WHO_AM_I register (0x75)
            who = _read_reg(bus, addr, _WHO_AM_I, retries=3)
            if who is None or who not in _VALID_WHO_AM_I:
                continue

            # 4. Configure Digital Low-Pass Filter (DLPF): 0x03 (~42 Hz cutoff for vehicle vibration rejection)
            _write_reg(bus, addr, _CONFIG, 0x03, retries=2)

            # 5. Configure Accelerometer: ±2 g (16384 LSB/g)
            _write_reg(bus, addr, _ACCEL_CFG, 0x00, retries=2)

            # 6. Configure Gyroscope: ±250 °/s (131 LSB/(°/s))
            _write_reg(bus, addr, _GYRO_CFG, 0x00, retries=2)
            time.sleep(0.020)

            # 7. Test burst read to verify data pipeline
            _burst_read_14(bus, addr)

            return addr, who
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Calibration — Gravity-Direction-Aware Zeroing
# ---------------------------------------------------------------------------

_AXES = ("ax", "ay", "az", "gx", "gy", "gz")


def _collect_samples(bus, addr: int) -> Optional[dict]:
    """Read _CALIB_SAMPLES samples; return per-axis mean / std (g, °/s) and mean temperature."""
    sums  = dict.fromkeys(_AXES, 0.0)
    sqs   = dict.fromkeys(_AXES, 0.0)
    t_sum = 0.0
    n = 0
    for _ in range(_CALIB_SAMPLES):
        try:
            ax, ay, az, t, gx, gy, gz = _burst_read_14(bus, addr)
            vals = (ax / _ACCEL_SCALE, ay / _ACCEL_SCALE, az / _ACCEL_SCALE,
                    gx / _GYRO_SCALE,  gy / _GYRO_SCALE,  gz / _GYRO_SCALE)
            for k, v in zip(_AXES, vals):
                sums[k] += v
                sqs[k]  += v * v
            t_sum += (t / 333.87) + 21.0
            n += 1
        except Exception:
            pass
        if _CALIB_DELAY > 0:
            time.sleep(_CALIB_DELAY)

    if n < (_CALIB_SAMPLES // 2):
        return None
    mean = {k: sums[k] / n for k in _AXES}
    std  = {k: math.sqrt(max(0.0, sqs[k] / n - mean[k] ** 2)) for k in _AXES}
    return {"n": n, "mean": mean, "std": std, "temp_c": t_sum / n}


def _offsets_from_stats(stats: dict) -> dict:
    """Gravity-axis zeroing (unchanged method): the axis with the largest mean carries 1 g."""
    m = stats["mean"]
    accel = [m["ax"], m["ay"], m["az"]]
    g_axis = max(range(3), key=lambda i: abs(accel[i]))
    g_sign = 1.0 if accel[g_axis] > 0 else -1.0
    return {
        "ax": m["ax"] - (g_sign if g_axis == 0 else 0.0),
        "ay": m["ay"] - (g_sign if g_axis == 1 else 0.0),
        "az": m["az"] - (g_sign if g_axis == 2 else 0.0),
        "gx": m["gx"], "gy": m["gy"], "gz": m["gz"],
    }


def _sane(offsets: dict, gravity_g: Optional[float] = None) -> Optional[str]:
    """Option 4 — sanity limits. Returns a rejection reason, or None if plausible."""
    max_g = float(_c("IMU_CAL_MAX_GYRO_OFFSET_DPS", 10.0))
    if any(abs(offsets[k]) > max_g for k in ("gx", "gy", "gz")):
        return f"gyro offset beyond ±{max_g:g} °/s"
    if gravity_g is not None:
        lo = float(_c("IMU_CAL_GRAVITY_MIN_G", 0.85))
        hi = float(_c("IMU_CAL_GRAVITY_MAX_G", 1.15))
        if not (lo <= gravity_g <= hi):
            return f"gravity {gravity_g:.3f} g outside {lo:g}–{hi:g} g"
    return None


def _evaluate(stats: dict, offsets: dict, mode: str) -> Optional[str]:
    """Options 1 + 4. Returns a rejection reason, or None if the attempt is acceptable."""
    m, s = stats["mean"], stats["std"]
    gravity = math.sqrt(m["ax"] ** 2 + m["ay"] ** 2 + m["az"] ** 2)
    reason = _sane(offsets, gravity)
    if reason:
        return reason

    gyro_spread  = max(s["gx"], s["gy"], s["gz"])
    accel_spread = max(s["ax"], s["ay"], s["az"])
    if gyro_spread > float(_c("IMU_CAL_MAX_GYRO_STD_DPS", 0.5)):
        return "moving (gyro varies)"
    if accel_spread > float(_c("IMU_CAL_MAX_ACCEL_STD_G", 0.03)):
        return "moving (accel varies)"

    # A smooth steady turn passes the spread test; compare with a trusted baseline.
    if mode == "boot" and _calibrated and _cal_trusted:
        g_shift = max(abs(offsets[k] - _offsets[k]) for k in ("gx", "gy", "gz"))
        a_shift = max(abs(offsets[k] - _offsets[k]) for k in ("ax", "ay", "az"))
        if g_shift > float(_c("IMU_CAL_MAX_GYRO_SHIFT_DPS", 1.5)):
            return f"gyro differs {g_shift:.2f} °/s from trusted baseline (turning?)"
        if a_shift > float(_c("IMU_CAL_MAX_ACCEL_SHIFT_G", 0.05)):
            return f"accel differs {a_shift:.3f} g from trusted baseline (braking/accelerating?)"
    return None


def _save_calibration(temp_c: Optional[float]) -> None:
    """Option 2 — persist offsets atomically (tmp file + rename)."""
    if not _CAL_FILE:
        return
    payload = {
        "offsets":  _offsets,
        "source":   _cal_source,
        "trusted":  _cal_trusted,
        "temp_c":   None if temp_c is None else round(temp_c, 2),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        tmp = _CAL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, _CAL_FILE)
    except Exception as exc:
        print(f"[IMU] Note: could not save calibration ({exc})")


def _load_calibration() -> bool:
    """Option 2 — load the last good offsets so a device booting while moving is usable at once."""
    global _offsets, _calibrated, _cal_source, _cal_trusted
    if not _CAL_FILE or not os.path.exists(_CAL_FILE):
        return False
    try:
        with open(_CAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        offs = {k: float(data["offsets"][k]) for k in _AXES}
    except Exception as exc:
        print(f"[IMU] Ignoring unreadable calibration file ({exc})")
        return False
    reason = _sane(offs)
    if reason:
        print(f"[IMU] Ignoring saved calibration: {reason}")
        return False
    _offsets     = offs
    _calibrated  = True
    _cal_source  = "saved"
    _cal_trusted = bool(data.get("trusted", False))
    print(f"[IMU] Loaded saved calibration ({'trusted' if _cal_trusted else 'provisional'}, "
          f"saved {data.get('saved_at', '?')})")
    return True


def _calibrate(bus, addr: int, mode: str = "boot") -> bool:
    """
    One calibration attempt. mode="boot": at start-up / while uncalibrated (vehicle may be
    moving). mode="stop": at a GNSS-confirmed stop.
    Only accepted if the readings show the vehicle was still (option 1) and the offsets are
    plausible (option 4). Accepted results are saved (option 2). At stops the result is
    blended into a trusted baseline instead of replacing it (option 3).
    """
    global _offsets, _calibrated, _cal_source, _cal_trusted
    stats = _collect_samples(bus, addr)
    if stats is None:
        print(f"[IMU] Calibration ({mode}) REJECTED: too few valid samples")
        return False

    measured = _offsets_from_stats(stats)
    reason   = _evaluate(stats, measured, mode)
    s = stats["std"]
    spread = (f"spread gyro={max(s['gx'], s['gy'], s['gz']):.3f} °/s "
              f"accel={max(s['ax'], s['ay'], s['az']):.4f} g, temp={stats['temp_c']:.1f} °C")
    if reason:
        keep = f"keeping {_cal_source} offsets" if _calibrated else "IMU stays invalid until calibrated"
        print(f"[IMU] Calibration ({mode}) REJECTED: {reason} | {spread} | {keep}")
        return False

    blend = float(_c("IMU_STOP_RECAL_BLEND", 0.3))
    if mode == "stop" and _calibrated and _cal_trusted:
        new = {k: _offsets[k] + blend * (measured[k] - _offsets[k]) for k in _AXES}
        how = f"blended {blend:.0%}"
    else:
        new = measured
        how = "replaced"

    trusted = (mode == "stop") or (mode == "boot" and _calibrated and _cal_trusted)
    _offsets     = new          # rebind: read() never sees a half-updated dict
    _calibrated  = True
    _cal_source  = mode
    _cal_trusted = trusted
    _save_calibration(stats["temp_c"])
    print(f"[IMU] Calibration ({mode}) ACCEPTED ({how}, {'trusted' if trusted else 'provisional'}) | {spread} | "
          f"gyro bias=({new['gx']:+.3f}, {new['gy']:+.3f}, {new['gz']:+.3f}) °/s")
    return True


def _calibration_running() -> bool:
    t = _cal_thread
    return t is not None and t.is_alive()


def _request_calibration(mode: str) -> None:
    """Run one calibration attempt in the background (never blocks the caller)."""
    global _cal_thread
    if _bus is None or not imu_ok:
        return
    with _recovery_lock:
        if _calibration_running() or _recovery_running():
            return
        bus, addr = _bus, _active_addr
        _cal_thread = threading.Thread(target=_calibrate, args=(bus, addr, mode),
                                       name=f"imu-calibrate-{mode}", daemon=True)
        _cal_thread.start()


def note_vehicle_state(speed_kmh: Optional[float], gps_fix: bool) -> None:
    """
    Option 3 — call at telemetry rate with live GNSS speed and fix. After the vehicle has
    been stopped (fix + speed ≤ IMU_STOP_RECAL_MAX_SPEED_KMH) for IMU_STOP_RECAL_MIN_STOP_SEC,
    starts a background re-calibration (at most once per IMU_STOP_RECAL_INTERVAL_SEC).
    Cheap: a few comparisons per call.
    """
    global _stop_since, _last_stop_recal_ts
    if not _c("IMU_STOP_RECAL_ENABLED", True) or not imu_ok or _bus is None:
        _stop_since = None
        return
    stopped = bool(gps_fix) and speed_kmh is not None and \
        float(speed_kmh) <= float(_c("IMU_STOP_RECAL_MAX_SPEED_KMH", 1.0))
    now = time.monotonic()
    if not stopped:
        _stop_since = None
        return
    if _stop_since is None:
        _stop_since = now
        return
    if (now - _stop_since) < float(_c("IMU_STOP_RECAL_MIN_STOP_SEC", 5.0)):
        return
    if (now - _last_stop_recal_ts) < float(_c("IMU_STOP_RECAL_INTERVAL_SEC", 600.0)):
        return
    _last_stop_recal_ts = now
    _request_calibration("stop")


def calibration_status() -> dict:
    """For diagnostics / HUD."""
    return {"calibrated": _calibrated, "source": _cal_source, "trusted": _cal_trusted,
            "offsets": dict(_offsets)}


# ---------------------------------------------------------------------------
# Complementary Filter & Quaternion Math
# ---------------------------------------------------------------------------

def _normalise_angle(a: float) -> float:
    """Keeps angle in (-180, +180]."""
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def _update_orientation(ax: float, ay: float, az: float,
                        gx: float, gy: float, gz: float,
                        dt: float) -> None:
    """Complementary filter: fuses accelerometer gravity vector with gyro integration."""
    global _roll_deg, _pitch_deg, _yaw_deg

    if dt <= 0.0 or dt > 0.5:
        dt = TELEMETRY_INTERVAL_SEC

    accel_roll  = math.degrees(math.atan2(ay, az))
    h_mag       = math.sqrt(ay * ay + az * az)
    accel_pitch = math.degrees(math.atan2(-ax, h_mag))

    alpha = COMPLEMENTARY_ALPHA
    _roll_deg  = alpha * (_roll_deg  + gx * dt) + (1.0 - alpha) * accel_roll
    _pitch_deg = alpha * (_pitch_deg + gy * dt) + (1.0 - alpha) * accel_pitch
    _yaw_deg   = _normalise_angle(_yaw_deg + gz * dt)


def _euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> dict:
    """Tait-Bryan Z-Y-X Euler angles to unit quaternion."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    return {
        "w": round(cr * cp * cy + sr * sp * sy, 6),
        "x": round(sr * cp * cy - cr * sp * sy, 6),
        "y": round(cr * sp * cy - sr * sp * sy, 6),
        "z": round(cr * cp * sy - sr * sp * cy, 6),
    }


# ---------------------------------------------------------------------------
# Recovery & Dynamic Bus Re-initialisation
# ---------------------------------------------------------------------------

def _recovery_running() -> bool:
    t = _recovery_thread
    return t is not None and t.is_alive()


def _request_recovery() -> None:
    """Start a background re-init if none is running and the retry interval has passed.
    Never blocks the caller (the 20 Hz telemetry loop)."""
    global _recovery_thread, _last_reinit_ts
    if not _smbus_available:
        return
    with _recovery_lock:
        if _recovery_running():
            return
        now = time.monotonic()
        if (now - _last_reinit_ts) < _RECOVERY_INTERVAL_SEC:
            return
        _last_reinit_ts = now
        _recovery_thread = threading.Thread(target=_reinit_bus, name="imu-recovery", daemon=True)
        _recovery_thread.start()


def _log_recovery_failure(reason: str) -> None:
    """Log the first failure, then at most once per _RECOVERY_LOG_EVERY_SEC."""
    global _last_recovery_log_ts
    now = time.monotonic()
    if _recovery_attempts == 1 or (now - _last_recovery_log_ts) >= _RECOVERY_LOG_EVERY_SEC:
        _last_recovery_log_ts = now
        print(f"[IMU] Unavailable ({reason}) — retrying every {_RECOVERY_INTERVAL_SEC:.0f}s "
              f"(attempt {_recovery_attempts})")


def _reinit_bus() -> bool:
    """
    Background recovery (runs in the 'imu-recovery' thread, never in the telemetry loop):
    opens a fresh SMBus handle, resets/wakes/configures the chip, calibrates if it was never
    calibrated (IMU missing at boot), then swaps the new handle in. Keeps retrying on
    later read() calls until it succeeds — a failed attempt no longer disables the IMU for good.
    Preserves existing calibration offsets and orientation state.
    """
    global _bus, _active_addr, imu_ok, _last_ts, _read_fail_count, _recovery_attempts
    _recovery_attempts += 1

    try:
        import smbus2  # type: ignore
        new_bus = smbus2.SMBus(1)
    except Exception as exc:
        imu_ok = False
        _log_recovery_failure(f"cannot open SMBus(1): {exc}")
        return False

    def _discard(bus) -> None:
        try:
            bus.close()
        except Exception:
            pass

    try:
        res = _probe_and_configure_chip(new_bus)
        if res is None:
            _discard(new_bus)
            imu_ok = False
            _log_recovery_failure("no responsive MPU sensor")
            return False

        addr, who = res
        if not _calibrated:
            # IMU missing at boot and no saved file: try now. If the vehicle is moving this is
            # rejected; the chip still comes up and read() keeps retrying calibration.
            _calibrate(new_bus, addr, "boot")

        with _i2c_lock:
            old_bus      = _bus
            _bus         = new_bus
            _active_addr = addr
        with _lock:
            _last_ts = time.monotonic()
        _read_fail_count = 0
        imu_ok = True
        if old_bus is not None and old_bus is not new_bus:
            _discard(old_bus)

        print(f"[IMU] Recovered: chip at 0x{addr:02X} (WHO_AM_I=0x{who:02X}) "
              f"after {_recovery_attempts} attempt(s)")
        _recovery_attempts = 0
        return True
    except Exception as exc:
        _discard(new_bus)
        imu_ok = False
        _log_recovery_failure(f"error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Public API — init()
# ---------------------------------------------------------------------------

def init() -> bool:
    """
    Initialise the MPU-6500 / MPU-6050 IMU on Raspberry Pi hardware I2C bus 1.
    Performs staged device reset, PLL clock wake-up, sensor configuration,
    and automatic gravity-direction calibration.
    Returns True on success, False otherwise.
    """
    global imu_ok, _bus, _active_addr, _last_ts, _read_fail_count, _smbus_available, _last_cal_attempt_ts

    try:
        import smbus2  # type: ignore
    except (ImportError, Exception) as exc:
        imu_ok = False
        _smbus_available = False
        print(f"[IMU] smbus2 unavailable ({exc}) — operating in bench/offline mode")
        return False

    # From here on a failed init is retried in the background by read().
    _smbus_available = True

    # Option 2: last good offsets first, so the IMU is usable even if the vehicle is moving now.
    _load_calibration()

    try:
        bus = smbus2.SMBus(1)
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — cannot open SMBus(1): {exc} (will keep retrying)")
        return False

    res = _probe_and_configure_chip(bus)
    if res is None:
        imu_ok = False
        print(f"[IMU] INIT FAILED — no responsive MPU sensor at candidate addresses "
              f"(0x{_PRIMARY_ADDR:02X}, 0x{_BACKUP_ADDR:02X}) (will keep retrying)")
        try:
            bus.close()
        except Exception:
            pass
        return False

    _active_addr, who = res
    print(f"[IMU] Detected sensor at 0x{_active_addr:02X} (WHO_AM_I=0x{who:02X}) — configured ±2g, ±250°/s, DLPF 42Hz")

    # Boot calibration attempt (~1 s). Accepted only if the vehicle is verifiably still;
    # otherwise the saved offsets (if any) are kept.
    _calibrate(bus, _active_addr, "boot")
    _last_cal_attempt_ts = time.monotonic()

    _bus = bus
    _last_ts = time.monotonic()
    _read_fail_count = 0
    imu_ok = True
    if _calibrated:
        print(f"[IMU] READY — MPU sensor at 0x{_active_addr:02X} on SMBus(1) ({_cal_source} calibration).")
    else:
        print(f"[IMU] Chip ready at 0x{_active_addr:02X} but NOT calibrated (vehicle moving, no saved "
              f"calibration) — readings invalid until a still moment or the first stop.")
    return True


# ---------------------------------------------------------------------------
# Public API — read()
# ---------------------------------------------------------------------------

def read() -> dict:
    """
    Read one sample from the IMU via single 14-byte I2C burst.
    Returns a dictionary formatted for vehicle telemetry, HUD, and stop detection.

    Fault tolerance:
      Transient I2C read errors are retried up to _READ_RETRIES times.
      Holds over the last valid sample (up to 1.0s) across brief glitches to
      prevent spurious drops in the 20 Hz telemetry loop and stop detector.
      If failure persists for >= 20 ticks (or the IMU was never initialised), a re-init
      is started in a background thread; this call never waits for it.
    """
    global _last_ts, _read_fail_count, _last_valid_imu, _last_valid_ts, _last_error_log_ts
    global _last_cal_attempt_ts

    if _recovery_running():
        # Re-init in progress on another thread — don't touch the bus.
        if _last_valid_imu is not None and (time.monotonic() - _last_valid_ts) < 1.0:
            return _last_valid_imu
        return {"valid": False}

    if not imu_ok or _bus is None:
        _request_recovery()
        return {"valid": False}

    if not _calibrated:
        # No usable offsets yet: retry a (background) calibration periodically; the
        # first GNSS-confirmed stop also calibrates via note_vehicle_state().
        now_c = time.monotonic()
        if now_c - _last_cal_attempt_ts >= float(_c("IMU_CAL_RETRY_SEC", 30.0)):
            _last_cal_attempt_ts = now_c
            _request_calibration("boot")
        return {"valid": False}

    off = _offsets   # one consistent snapshot even if a calibration swaps it mid-read
    last_exc = None
    for attempt in range(1, _READ_RETRIES + 1):
        try:
            ax_r, ay_r, az_r, t_r, gx_r, gy_r, gz_r = _burst_read_14(_bus, _active_addr)

            # Scale and apply calibration offsets
            ax = (ax_r / _ACCEL_SCALE) - off["ax"]
            ay = (ay_r / _ACCEL_SCALE) - off["ay"]
            az = (az_r / _ACCEL_SCALE) - off["az"]
            gx = (gx_r / _GYRO_SCALE)  - off["gx"]
            gy = (gy_r / _GYRO_SCALE)  - off["gy"]
            gz = (gz_r / _GYRO_SCALE)  - off["gz"]
            temperature_c = (t_r / 333.87) + 21.0

            # Complementary filter update (thread-safe)
            now_ts = time.monotonic()
            with _lock:
                dt = now_ts - _last_ts
                _last_ts = now_ts
                _update_orientation(ax, ay, az, gx, gy, gz, dt)
                roll  = _roll_deg
                pitch = _pitch_deg
                yaw   = _yaw_deg

            mag  = math.sqrt(ax * ax + ay * ay + az * az) * _G_MS2
            quat = _euler_to_quaternion(roll, pitch, yaw)

            _read_fail_count = 0

            result = {
                "valid":   True,
                "accel_g": {"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)},
                "accel_ms2": {
                    "x": round(ax * _G_MS2, 4),
                    "y": round(ay * _G_MS2, 4),
                    "z": round(az * _G_MS2, 4),
                },
                "accel_magnitude_ms2": round(mag, 4),
                "gyro_dps": {"x": round(gx, 4), "y": round(gy, 4), "z": round(gz, 4)},
                "temperature_c": round(temperature_c, 2),
                "orientation": {
                    "roll":       round(roll, 3),
                    "pitch":      round(pitch, 3),
                    "yaw":        round(yaw, 3),
                    "yaw_source": "GYRO_INTEGRATED",
                },
                "quaternion": quat,
            }
            _last_valid_imu = result
            _last_valid_ts  = now_ts
            return result

        except Exception as exc:
            last_exc = exc
            if attempt < _READ_RETRIES:
                time.sleep(_READ_RETRY_DELAY)

    # All retries exhausted on this tick
    _read_fail_count += 1
    now_mono = time.monotonic()

    # Rate-limit persistent failure logging to once every 5 seconds
    if _read_fail_count >= 20 and (now_mono - _last_error_log_ts) >= 5.0:
        _last_error_log_ts = now_mono
        print(f"[IMU] Warning: I2C read failed continuously ({_read_fail_count} samples): {last_exc}")

    # After 20 consecutive fails (~1 s) start a background re-init (non-blocking)
    if _read_fail_count >= 20:
        _request_recovery()

    # Hold over last valid sample for up to 1.0s to smooth transient bus glitches
    if _last_valid_imu is not None and (now_mono - _last_valid_ts) < 1.0:
        return _last_valid_imu

    return {"valid": False}
