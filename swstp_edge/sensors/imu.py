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

import math
import struct
import threading
import time
from typing import Optional, Tuple, Dict, Any

from config import COMPLEMENTARY_ALPHA, TELEMETRY_INTERVAL_SEC

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
_CALIB_SAMPLES    = 100
_CALIB_DELAY      = 0.010 # s

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

def _calibrate(bus, addr: int) -> bool:
    """
    Collect _CALIB_SAMPLES at rest to determine stationary zero offsets.
    Identifies the gravity axis automatically (axis with largest absolute mean)
    and removes 1.0 g, zeroing all other axes.
    """
    global _offsets
    print(f"[IMU] Calibrating ({_CALIB_SAMPLES} samples) at address 0x{addr:02X} — hold still…")

    ax_acc = ay_acc = az_acc = 0.0
    gx_acc = gy_acc = gz_acc = 0.0
    valid_samples = 0

    for _ in range(_CALIB_SAMPLES):
        try:
            ax, ay, az, _t, gx, gy, gz = _burst_read_14(bus, addr)
            ax_acc += ax; ay_acc += ay; az_acc += az
            gx_acc += gx; gy_acc += gy; gz_acc += gz
            valid_samples += 1
        except Exception:
            pass
        time.sleep(_CALIB_DELAY)

    if valid_samples < (_CALIB_SAMPLES // 2):
        print(f"[IMU] Calibration failed: insufficient valid samples ({valid_samples}/{_CALIB_SAMPLES})")
        return False

    n = float(valid_samples)
    ax_mean = (ax_acc / n) / _ACCEL_SCALE
    ay_mean = (ay_acc / n) / _ACCEL_SCALE
    az_mean = (az_acc / n) / _ACCEL_SCALE

    # Gravity axis: whichever has largest absolute mean
    magnitudes = [abs(ax_mean), abs(ay_mean), abs(az_mean)]
    grav_axis = magnitudes.index(max(magnitudes))  # 0=X, 1=Y, 2=Z
    means = [ax_mean, ay_mean, az_mean]
    grav_sign = 1.0 if means[grav_axis] > 0 else -1.0

    _offsets["ax"] = ax_mean - (grav_sign if grav_axis == 0 else 0.0)
    _offsets["ay"] = ay_mean - (grav_sign if grav_axis == 1 else 0.0)
    _offsets["az"] = az_mean - (grav_sign if grav_axis == 2 else 0.0)
    _offsets["gx"] = (gx_acc / n) / _GYRO_SCALE
    _offsets["gy"] = (gy_acc / n) / _GYRO_SCALE
    _offsets["gz"] = (gz_acc / n) / _GYRO_SCALE

    axis_names = ("X", "Y", "Z")
    print(
        f"[IMU] Calibration complete ({valid_samples} samples).\n"
        f"      Gravity axis : {axis_names[grav_axis]} ({grav_sign:+.0f} g, mean={means[grav_axis]:+.4f} g)\n"
        f"      Accel offsets: ax={_offsets['ax']:+.4f}  ay={_offsets['ay']:+.4f}  az={_offsets['az']:+.4f} g\n"
        f"      Gyro  offsets: gx={_offsets['gx']:+.4f}  gy={_offsets['gy']:+.4f}  gz={_offsets['gz']:+.4f} °/s"
    )
    return True


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

def _reinit_bus() -> bool:
    """
    Recover from persistent I2C bus errors: closes the stale smbus handle,
    opens a fresh connection, wakes chip with PLL clock, and restores communication.
    Rate-limited to once every 5.0 seconds. Preserves calibrated offsets.
    """
    global _bus, _active_addr, imu_ok, _last_ts, _last_reinit_ts
    now = time.monotonic()
    if (now - _last_reinit_ts) < 5.0:
        return False
    _last_reinit_ts = now

    try:
        import smbus2  # type: ignore
        # Safely close old bus handle to release Linux I2C adapter state
        if _bus is not None:
            try:
                _bus.close()
            except Exception:
                pass
            _bus = None

        new_bus = smbus2.SMBus(1)
        res = _probe_and_configure_chip(new_bus)
        if res is None:
            try:
                new_bus.close()
            except Exception:
                pass
            imu_ok = False
            return False

        _active_addr, who = res
        _bus = new_bus
        _last_ts = time.monotonic()
        imu_ok = True
        print(f"[IMU] Re-initialisation successful: chip at 0x{_active_addr:02X} (WHO_AM_I=0x{who:02X})")
        return True
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] Re-initialisation error: {exc}")
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
    global imu_ok, _bus, _active_addr, _last_ts, _read_fail_count

    try:
        import smbus2  # type: ignore
    except (ImportError, Exception) as exc:
        imu_ok = False
        print(f"[IMU] smbus2 unavailable ({exc}) — operating in bench/offline mode")
        return False

    try:
        bus = smbus2.SMBus(1)
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — cannot open SMBus(1): {exc}")
        return False

    res = _probe_and_configure_chip(bus)
    if res is None:
        imu_ok = False
        print(f"[IMU] INIT FAILED — no responsive MPU sensor at candidate addresses (0x{_PRIMARY_ADDR:02X}, 0x{_BACKUP_ADDR:02X})")
        try:
            bus.close()
        except Exception:
            pass
        return False

    _active_addr, who = res
    print(f"[IMU] Detected sensor at 0x{_active_addr:02X} (WHO_AM_I=0x{who:02X}) — configured ±2g, ±250°/s, DLPF 42Hz")

    # Run calibration
    if not _calibrate(bus, _active_addr):
        imu_ok = False
        print(f"[IMU] INIT FAILED — sensor calibration uncompleted")
        try:
            bus.close()
        except Exception:
            pass
        return False

    _bus = bus
    _last_ts = time.monotonic()
    _read_fail_count = 0
    imu_ok = True
    print(f"[IMU] READY — MPU sensor at 0x{_active_addr:02X} on SMBus(1) initialised successfully.")
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
      Attempts non-blocking dynamic re-init if failure persists for >= 20 ticks.
    """
    global _last_ts, _read_fail_count, _last_valid_imu, _last_valid_ts, _last_error_log_ts

    if not imu_ok or _bus is None:
        return {"valid": False}

    last_exc = None
    for attempt in range(1, _READ_RETRIES + 1):
        try:
            ax_r, ay_r, az_r, t_r, gx_r, gy_r, gz_r = _burst_read_14(_bus, _active_addr)

            # Scale and apply calibration offsets
            ax = (ax_r / _ACCEL_SCALE) - _offsets["ax"]
            ay = (ay_r / _ACCEL_SCALE) - _offsets["ay"]
            az = (az_r / _ACCEL_SCALE) - _offsets["az"]
            gx = (gx_r / _GYRO_SCALE)  - _offsets["gx"]
            gy = (gy_r / _GYRO_SCALE)  - _offsets["gy"]
            gz = (gz_r / _GYRO_SCALE)  - _offsets["gz"]
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

    # Trigger automatic fresh bus re-init after 20 consecutive fails (1.0s)
    if _read_fail_count >= 20:
        if _reinit_bus():
            _read_fail_count = 0

    # Hold over last valid sample for up to 1.0s to smooth transient bus glitches
    if _last_valid_imu is not None and (now_mono - _last_valid_ts) < 1.0:
        return _last_valid_imu

    return {"valid": False}
