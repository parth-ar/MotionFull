"""
sensors/imu.py — MPU-6500 IMU driver for Raspberry Pi via smbus2.

DESIGN
------
Reads the MPU-6500 at I2C address 0x69 (AD0 pin pulled HIGH) using smbus2
for direct register access.  Ports the complementary filter and Euler →
quaternion conversion from the Arduino firmware (updateOrientation /
printQuaternion) as closely as possible.

Register map (MPU-6500 matches MPU-6050 for the used registers):
  0x6B  PWR_MGMT_1     — wake from sleep
  0x1B  GYRO_CONFIG    — ±250 °/s  → value 0x00
  0x1C  ACCEL_CONFIG   — ±2 g      → value 0x00
  0x3B  ACCEL_XOUT_H   — 6 bytes accel (X H/L, Y H/L, Z H/L)
  0x41  TEMP_OUT_H     — 2 bytes temperature
  0x43  GYRO_XOUT_H    — 6 bytes gyro  (X H/L, Y H/L, Z H/L)

Scale factors:
  Accel ±2 g   → 16384 LSB/g
  Gyro  ±250 °/s → 131 LSB/(°/s)
  Temp  → T_C = raw / 333.87 + 21.0  (MPU-6500 datasheet §4.18)

Complementary filter (identical to .ino):
  α = 0.98
  roll  = α * (roll  + gx * dt) + (1-α) * accel_roll
  pitch = α * (pitch + gy * dt) + (1-α) * accel_pitch
  yaw   = normalise(yaw + gz * dt)   ← gyro-only, no magnetometer

VOLTAGE CAUTION ⚠️
------------------
Most GY-6500 / GY-521 breakout boards have an onboard 3.3 V regulator so
the chip itself runs at 3.3 V.  The I2C logic levels on those boards are
therefore 3.3 V — safe for direct connection to Pi GPIO 2/3 (SDA/SCL).
If you have a bare MPU-6500 module wired to 5 V, insert a bidirectional
level-shifter before connecting to the Pi.  Check your specific module's
schematic.

I2C address: 0x69 (AD0 pulled HIGH — mirrors .ino MPU6500_WE mpu(0x69))
"""

import math
import struct
import threading
import time
from config import COMPLEMENTARY_ALPHA, TELEMETRY_INTERVAL_SEC

# ---------------------------------------------------------------------------
# MPU-6500 register addresses
# ---------------------------------------------------------------------------
_ADDR       = 0x69
_PWR_MGMT_1 = 0x6B
_ACCEL_CFG  = 0x1C
_GYRO_CFG   = 0x1B
_ACCEL_OUT  = 0x3B   # 6 bytes: AX_H AX_L AY_H AY_L AZ_H AZ_L
_TEMP_OUT   = 0x41   # 2 bytes: T_H T_L
_GYRO_OUT   = 0x43   # 6 bytes: GX_H GX_L GY_H GY_L GZ_H GZ_L
_WHO_AM_I   = 0x75

# Scale factors
_ACCEL_SCALE = 16384.0   # LSB/g  for ±2 g
_GYRO_SCALE  = 131.0     # LSB/(°/s) for ±250 °/s
_G_MS2       = 9.80665   # m/s² per g

# ---------------------------------------------------------------------------
# Module state (thread-safe via lock)
# ---------------------------------------------------------------------------
imu_ok: bool = False

_lock          = threading.Lock()
_roll_deg      = 0.0
_pitch_deg     = 0.0
_yaw_deg       = 0.0
_last_us: float = 0.0
_bus           = None   # smbus2.SMBus instance


# ---------------------------------------------------------------------------
# I2C helpers
# ---------------------------------------------------------------------------
def _read_word_signed(bus, reg: int) -> int:
    """Read a big-endian signed 16-bit word from two consecutive registers."""
    high = bus.read_byte_data(_ADDR, reg)
    low  = bus.read_byte_data(_ADDR, reg + 1)
    val  = (high << 8) | low
    if val >= 0x8000:
        val -= 0x10000
    return val


def _read_block(bus, reg: int, length: int) -> bytes:
    """Read `length` bytes starting at register `reg`."""
    return bytes(bus.read_i2c_block_data(_ADDR, reg, length))


# ---------------------------------------------------------------------------
# Calibration offsets (computed by autoOffsets() equivalent)
# ---------------------------------------------------------------------------
_ax_off = _ay_off = _az_off = 0.0
_gx_off = _gy_off = _gz_off = 0.0
_CALIB_SAMPLES = 200


def _calibrate(bus) -> None:
    """
    Equivalent to mpu.autoOffsets():  average N samples at rest and store
    offsets.  The device must be held still during startup.
    """
    global _ax_off, _ay_off, _az_off, _gx_off, _gy_off, _gz_off
    print("[IMU] Calibrating — hold still…")
    ax_acc = ay_acc = az_acc = 0.0
    gx_acc = gy_acc = gz_acc = 0.0
    for _ in range(_CALIB_SAMPLES):
        raw = _read_block(bus, _ACCEL_OUT, 6)
        ax, ay, az = struct.unpack(">hhh", raw)
        raw_g = _read_block(bus, _GYRO_OUT, 6)
        gx, gy, gz = struct.unpack(">hhh", raw_g)
        ax_acc += ax; ay_acc += ay; az_acc += az
        gx_acc += gx; gy_acc += gy; gz_acc += gz
        time.sleep(0.005)
    n = float(_CALIB_SAMPLES)
    _ax_off = ax_acc / n / _ACCEL_SCALE
    _ay_off = ay_acc / n / _ACCEL_SCALE
    _az_off = az_acc / n / _ACCEL_SCALE - 1.0   # subtract 1 g on Z
    _gx_off = gx_acc / n / _GYRO_SCALE
    _gy_off = gy_acc / n / _GYRO_SCALE
    _gz_off = gz_acc / n / _GYRO_SCALE
    print(f"[IMU] Calibration done. Offsets: ax={_ax_off:.4f} ay={_ay_off:.4f} az={_az_off:.4f} "
          f"gx={_gx_off:.4f} gy={_gy_off:.4f} gz={_gz_off:.4f}")


# ---------------------------------------------------------------------------
# Complementary filter — verbatim port from .ino updateOrientation()
# ---------------------------------------------------------------------------
def _update_orientation(ax: float, ay: float, az: float,
                         gx: float, gy: float, gz: float,
                         dt: float) -> None:
    """Port of Arduino updateOrientation().  Updates module globals."""
    global _roll_deg, _pitch_deg, _yaw_deg

    # Guard: clamp dt to valid range (same as .ino `if (dt<=0 || dt>0.5) dt=0.05`)
    if dt <= 0.0 or dt > 0.5:
        dt = TELEMETRY_INTERVAL_SEC

    accel_roll  = math.degrees(math.atan2(ay, az))
    h_mag       = math.sqrt(ay * ay + az * az)
    accel_pitch = math.degrees(math.atan2(-ax, h_mag))

    alpha = COMPLEMENTARY_ALPHA
    _roll_deg  = alpha * (_roll_deg  + gx * dt) + (1.0 - alpha) * accel_roll
    _pitch_deg = alpha * (_pitch_deg + gy * dt) + (1.0 - alpha) * accel_pitch
    _yaw_deg   = _normalise_angle(_yaw_deg + gz * dt)


def _normalise_angle(a: float) -> float:
    """Port of .ino normalizeAngle() — keeps angle in (-180, +180]."""
    while a >  180.0: a -= 360.0
    while a < -180.0: a += 360.0
    return a


def _euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> dict:
    """Port of .ino printQuaternion() — Tait-Bryan Z-Y-X → quaternion."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    return {
        "w": round(cr*cp*cy + sr*sp*sy, 6),
        "x": round(sr*cp*cy - cr*sp*sy, 6),
        "y": round(cr*sp*cy + sr*cp*sy, 6),
        "z": round(cr*cp*sy - sr*sp*cy, 6),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init() -> bool:
    """
    Open the SMBus, wake the MPU-6500, configure ranges, and run calibration.
    Returns True on success.
    """
    global imu_ok, _bus, _last_us
    try:
        import smbus2  # type: ignore
        bus = smbus2.SMBus(1)

        # Verify WHO_AM_I (0x70 for MPU-6500, 0x68 for MPU-6050)
        who = bus.read_byte_data(_ADDR, _WHO_AM_I)
        if who not in (0x70, 0x68, 0x71, 0x19):
            print(f"[IMU] Unexpected WHO_AM_I: 0x{who:02X} — continuing anyway")

        # Wake from sleep: PWR_MGMT_1 bit[6] = 0
        bus.write_byte_data(_ADDR, _PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Accel ±2 g (ACCEL_CONFIG = 0x00)
        bus.write_byte_data(_ADDR, _ACCEL_CFG, 0x00)
        # Gyro ±250 °/s (GYRO_CONFIG = 0x00)
        bus.write_byte_data(_ADDR, _GYRO_CFG,  0x00)
        time.sleep(0.05)

        _calibrate(bus)
        _bus = bus
        _last_us = time.monotonic()
        imu_ok = True
        print(f"[IMU] MPU-6500 OK at 0x{_ADDR:02X} (WHO_AM_I=0x{who:02X})")
        return True

    except Exception as exc:
        imu_ok = False
        print(f"[IMU] FAULT — could not initialise MPU-6500 at 0x{_ADDR:02X}: {exc}")
        return False


def read() -> dict:
    """
    Read one sample from the MPU-6500 and return a dict structurally identical
    to the 'imu' sub-object emitted by the .ino emitTelemetry().

    Output shape:
    {
        "valid": bool,
        "accel_g":          {"x": float, "y": float, "z": float},
        "accel_ms2":        {"x": float, "y": float, "z": float},
        "accel_magnitude_ms2": float,
        "gyro_dps":         {"x": float, "y": float, "z": float},
        "temperature_c":    float,
        "orientation":      {"roll": float, "pitch": float, "yaw": float,
                             "yaw_source": "GYRO_INTEGRATED"},
        "quaternion":       {"w": float, "x": float, "y": float, "z": float},
    }
    If imu_ok is False returns a minimal valid=False dict.
    """
    global _last_us

    if not imu_ok or _bus is None:
        return {"valid": False}

    try:
        # --- Read accel (6 bytes) ---
        raw_a = _read_block(_bus, _ACCEL_OUT, 6)
        ax_raw, ay_raw, az_raw = struct.unpack(">hhh", raw_a)

        # --- Read temperature (2 bytes) ---
        raw_t = _read_block(_bus, _TEMP_OUT, 2)
        t_raw, = struct.unpack(">h", raw_t)

        # --- Read gyro (6 bytes) ---
        raw_g = _read_block(_bus, _GYRO_OUT, 6)
        gx_raw, gy_raw, gz_raw = struct.unpack(">hhh", raw_g)

        # --- Scale + apply calibration offsets ---
        ax = ax_raw / _ACCEL_SCALE - _ax_off
        ay = ay_raw / _ACCEL_SCALE - _ay_off
        az = az_raw / _ACCEL_SCALE - _az_off
        gx = gx_raw / _GYRO_SCALE  - _gx_off
        gy = gy_raw / _GYRO_SCALE  - _gy_off
        gz = gz_raw / _GYRO_SCALE  - _gz_off
        temperature_c = t_raw / 333.87 + 21.0

        # --- Complementary filter ---
        now_us = time.monotonic()
        with _lock:
            dt = now_us - _last_us
            _last_us = now_us
            _update_orientation(ax, ay, az, gx, gy, gz, dt)
            roll  = _roll_deg
            pitch = _pitch_deg
            yaw   = _yaw_deg

        # --- Derived values ---
        mag = math.sqrt(ax*ax + ay*ay + az*az) * _G_MS2
        quat = _euler_to_quaternion(roll, pitch, yaw)

        return {
            "valid": True,
            "accel_g":   {"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)},
            "accel_ms2": {"x": round(ax * _G_MS2, 4),
                          "y": round(ay * _G_MS2, 4),
                          "z": round(az * _G_MS2, 4)},
            "accel_magnitude_ms2": round(mag, 4),
            "gyro_dps":  {"x": round(gx, 4), "y": round(gy, 4), "z": round(gz, 4)},
            "temperature_c": round(temperature_c, 2),
            "orientation": {
                "roll":       round(roll, 3),
                "pitch":      round(pitch, 3),
                "yaw":        round(yaw, 3),
                "yaw_source": "GYRO_INTEGRATED",
            },
            "quaternion": quat,
        }

    except Exception as exc:
        print(f"[IMU] Read error: {exc}")
        return {"valid": False}
