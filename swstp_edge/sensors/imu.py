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
  0x3B  ACCEL_XOUT_H   — start of 14-byte burst block:
                          [0:5]   AX_H AX_L AY_H AY_L AZ_H AZ_L  (accel)
                          [6:7]   T_H  T_L                         (temp)
                          [8:13]  GX_H GX_L GY_H GY_L GZ_H GZ_L  (gyro)
  0x75  WHO_AM_I       — 0x70 for MPU-6500

Scale factors:
  Accel ±2 g   → 16384 LSB/g
  Gyro  ±250 °/s → 131 LSB/(°/s)
  Temp  → T_C = raw / 333.87 + 21.0  (MPU-6500 datasheet §4.18)

Complementary filter (identical to .ino):
  α = 0.98
  roll  = α * (roll  + gx * dt) + (1-α) * accel_roll
  pitch = α * (pitch + gy * dt) + (1-α) * accel_pitch
  yaw   = normalise(yaw + gz * dt)   ← gyro-only, no magnetometer

Gravity-axis calibration:
  During calibration the axis with the largest absolute mean is inferred as
  the gravity axis.  Only that axis gets a 1 g offset removed; all other
  axes are zeroed on their measured mean.  This handles arbitrary board
  orientation without hard-coding "Z is always +1 g".

Initialization sequence (staged, with per-stage retries):
  BUS_OPEN → WHO_AM_I → WAKE → CONFIG_ACCEL → CONFIG_GYRO →
  FIRST_READ → CALIBRATION → READY

Retry policy:
  Init stages   : 3 attempts, 50 ms between attempts
  Normal reads  : 3 attempts,  8 ms between attempts; triggers re-init on
                  repeated failure

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
_ADDR        = 0x69
_PWR_MGMT_1  = 0x6B
_ACCEL_CFG   = 0x1C
_GYRO_CFG    = 0x1B
_BURST_START = 0x3B   # start of 14-byte block: accel(6) + temp(2) + gyro(6)
_WHO_AM_I    = 0x75

# Known valid WHO_AM_I values
_VALID_WHO_AM_I = frozenset({0x70, 0x68, 0x71, 0x19})

# Scale factors
_ACCEL_SCALE = 16384.0   # LSB/g  for ±2 g
_GYRO_SCALE  = 131.0     # LSB/(°/s) for ±250 °/s
_G_MS2       = 9.80665   # m/s² per g

# Retry / timing parameters
_INIT_RETRIES     = 3      # attempts per init stage
_INIT_RETRY_DELAY = 0.050  # s between init retries
_READ_RETRIES     = 3      # attempts per normal read
_READ_RETRY_DELAY = 0.008  # s between read retries

# Calibration
_CALIB_SAMPLES     = 100
_CALIB_DELAY       = 0.010  # s between calibration samples

# ---------------------------------------------------------------------------
# Module state (thread-safe via lock)
# ---------------------------------------------------------------------------
imu_ok: bool = False

_lock           = threading.Lock()
_roll_deg       = 0.0
_pitch_deg      = 0.0
_yaw_deg        = 0.0
_last_ts: float = 0.0
_bus            = None   # smbus2.SMBus — kept open after init

# Calibration offsets (in scaled units: g for accel, °/s for gyro)
_ax_off = _ay_off = _az_off = 0.0
_gx_off = _gy_off = _gz_off = 0.0

# Consecutive read-failure counter — triggers re-init when it reaches _READ_RETRIES
_read_fail_count: int = 0


# ---------------------------------------------------------------------------
# Low-level 14-byte burst read
# ---------------------------------------------------------------------------
def _burst_read_14(bus) -> tuple[int, int, int, int, int, int, int]:
    """
    Single 14-byte I²C burst starting at register 0x3B.
    Returns (ax_raw, ay_raw, az_raw, t_raw, gx_raw, gy_raw, gz_raw) as
    signed 16-bit integers.
    """
    raw = bus.read_i2c_block_data(_ADDR, _BURST_START, 14)
    ax, ay, az, t, gx, gy, gz = struct.unpack(">hhhhhhh", bytes(raw))
    return ax, ay, az, t, gx, gy, gz


# ---------------------------------------------------------------------------
# Staged init helper — retry wrapper
# ---------------------------------------------------------------------------
def _stage(name: str, fn, retries: int = _INIT_RETRIES, delay: float = _INIT_RETRY_DELAY):
    """
    Call fn() up to `retries` times.  Prints stage-level diagnostics.
    Returns the return value of fn() on success, raises the last exception
    on total failure.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = fn()
            print(f"[IMU] {name} — OK")
            return result
        except Exception as exc:
            last_exc = exc
            print(
                f"[IMU] INIT FAILED\n"
                f"      Stage  : {name}\n"
                f"      Attempt: {attempt}/{retries}\n"
                f"      Error  : {type(exc).__name__} {exc}"
            )
            if attempt < retries:
                time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Calibration — 14-byte burst, gravity-direction-aware
# ---------------------------------------------------------------------------
def _calibrate(bus) -> None:
    """
    Collect _CALIB_SAMPLES burst reads at rest and compute per-axis offsets.

    Gravity axis determination:
      The axis (X, Y, or Z) with the largest absolute mean accelerometer
      value is inferred as the gravity axis.  A 1 g offset is removed from
      that axis in the direction of the mean (handles ±Z, ±X, ±Y mounting).
      All other accel axes are zeroed to their measured mean.
      Gyroscope offsets are the raw means (gyro should read 0 at rest).

    This correctly handles the confirmed mounting where Z ≈ -1.11 g.
    """
    global _ax_off, _ay_off, _az_off, _gx_off, _gy_off, _gz_off
    print(f"[IMU] Calibrating ({_CALIB_SAMPLES} samples) — hold still…")
    ax_acc = ay_acc = az_acc = 0.0
    gx_acc = gy_acc = gz_acc = 0.0
    for _ in range(_CALIB_SAMPLES):
        ax, ay, az, _t, gx, gy, gz = _burst_read_14(bus)
        ax_acc += ax; ay_acc += ay; az_acc += az
        gx_acc += gx; gy_acc += gy; gz_acc += gz
        time.sleep(_CALIB_DELAY)

    n = float(_CALIB_SAMPLES)
    ax_mean = ax_acc / n / _ACCEL_SCALE
    ay_mean = ay_acc / n / _ACCEL_SCALE
    az_mean = az_acc / n / _ACCEL_SCALE

    # Determine gravity axis: whichever has the largest |mean|
    magnitudes = [abs(ax_mean), abs(ay_mean), abs(az_mean)]
    grav_axis  = magnitudes.index(max(magnitudes))   # 0=X, 1=Y, 2=Z
    grav_sign  = 1.0 if [ax_mean, ay_mean, az_mean][grav_axis] > 0 else -1.0

    # Build offsets: gravity axis → remove 1 g in measured direction; others → zero mean
    ax_off_new = ax_mean - (grav_sign if grav_axis == 0 else 0.0)
    ay_off_new = ay_mean - (grav_sign if grav_axis == 1 else 0.0)
    az_off_new = az_mean - (grav_sign if grav_axis == 2 else 0.0)

    _ax_off = ax_off_new
    _ay_off = ay_off_new
    _az_off = az_off_new
    _gx_off = gx_acc / n / _GYRO_SCALE
    _gy_off = gy_acc / n / _GYRO_SCALE
    _gz_off = gz_acc / n / _GYRO_SCALE

    axis_names = ("X", "Y", "Z")
    print(
        f"[IMU] Calibration complete.\n"
        f"      Gravity axis : {axis_names[grav_axis]} ({grav_sign:+.0f} g, mean={[ax_mean,ay_mean,az_mean][grav_axis]:+.4f} g)\n"
        f"      Accel offsets: ax={_ax_off:+.4f}  ay={_ay_off:+.4f}  az={_az_off:+.4f}  g\n"
        f"      Gyro  offsets: gx={_gx_off:+.4f}  gy={_gy_off:+.4f}  gz={_gz_off:+.4f}  °/s"
    )


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
# Internal re-initialisation (used by read() recovery path)
# ---------------------------------------------------------------------------
def _reinit_bus() -> bool:
    """
    Attempt a lightweight re-initialisation: verify WHO_AM_I and re-apply
    wake + config registers.  Called when repeated read failures occur.
    Does NOT re-run full calibration — keeps existing offsets.
    Returns True if the bus is responsive again.
    """
    global _bus, imu_ok, _last_ts
    try:
        bus = _bus
        if bus is None:
            return False

        who = bus.read_byte_data(_ADDR, _WHO_AM_I)
        if who not in _VALID_WHO_AM_I:
            print(f"[IMU] Re-init: unexpected WHO_AM_I 0x{who:02X} — aborting")
            return False

        bus.write_byte_data(_ADDR, _PWR_MGMT_1, 0x00)
        time.sleep(0.15)
        bus.write_byte_data(_ADDR, _ACCEL_CFG, 0x00)
        bus.write_byte_data(_ADDR, _GYRO_CFG,  0x00)
        time.sleep(0.05)

        # Verify a burst read succeeds
        _burst_read_14(bus)

        _last_ts = time.monotonic()
        imu_ok   = True
        print(f"[IMU] Re-initialisation successful (WHO_AM_I=0x{who:02X})")
        return True

    except Exception as exc:
        print(f"[IMU] Re-initialisation failed: {exc}")
        imu_ok = False
        return False


# ---------------------------------------------------------------------------
# Public API — init()
# ---------------------------------------------------------------------------
def init() -> bool:
    """
    Open SMBus(1), run staged initialisation with per-stage retries, calibrate.

    Staged sequence:
      BUS_OPEN → WHO_AM_I → WAKE → CONFIG_ACCEL → CONFIG_GYRO →
      FIRST_READ → CALIBRATION

    Returns True on success, False on any unrecoverable failure.
    """
    global imu_ok, _bus, _last_ts, _read_fail_count

    import smbus2  # type: ignore

    bus = None

    # ── Stage: BUS_OPEN ───────────────────────────────────────────────────
    try:
        bus = _stage("BUS_OPEN", lambda: smbus2.SMBus(1))
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — cannot open SMBus(1): {exc}")
        return False

    # ── Stage: WHO_AM_I ───────────────────────────────────────────────────
    try:
        who = _stage("WHO_AM_I",
                     lambda: bus.read_byte_data(_ADDR, _WHO_AM_I))
        if who not in _VALID_WHO_AM_I:
            print(f"[IMU]   Note: WHO_AM_I=0x{who:02X} not in known-good set "
                  f"{[hex(v) for v in sorted(_VALID_WHO_AM_I)]} — continuing")
        else:
            print(f"[IMU]   WHO_AM_I = 0x{who:02X}  ✔")
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — WHO_AM_I unreadable: {exc}")
        try: bus.close()
        except Exception: pass
        return False

    # ── Stage: WAKE ──────────────────────────────────────────────────────
    try:
        _stage("WAKE",
               lambda: bus.write_byte_data(_ADDR, _PWR_MGMT_1, 0x00))
        time.sleep(0.15)   # give oscillator time to stabilise
        print("[IMU]   Wake successful")
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — cannot wake MPU-6500: {exc}")
        try: bus.close()
        except Exception: pass
        return False

    # ── Stage: CONFIG_ACCEL ───────────────────────────────────────────────
    try:
        _stage("CONFIG_ACCEL",
               lambda: bus.write_byte_data(_ADDR, _ACCEL_CFG, 0x00))
        print("[IMU]   Accelerometer ±2 g configured")
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — accelerometer config: {exc}")
        try: bus.close()
        except Exception: pass
        return False

    # ── Stage: CONFIG_GYRO ────────────────────────────────────────────────
    try:
        _stage("CONFIG_GYRO",
               lambda: bus.write_byte_data(_ADDR, _GYRO_CFG, 0x00))
        time.sleep(0.05)
        print("[IMU]   Gyroscope ±250 °/s configured")
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — gyroscope config: {exc}")
        try: bus.close()
        except Exception: pass
        return False

    # ── Stage: WHO_AM_I verification (post-config) ────────────────────────
    try:
        who2 = _stage("WHO_AM_I_VERIFY",
                      lambda: bus.read_byte_data(_ADDR, _WHO_AM_I))
        print(f"[IMU]   Post-config WHO_AM_I = 0x{who2:02X}  ✔")
    except Exception as exc:
        # Non-fatal — log but don't abort; hardware responded fine up to here
        print(f"[IMU]   Post-config WHO_AM_I verify warning: {exc} — continuing")

    # ── Stage: FIRST_READ ─────────────────────────────────────────────────
    try:
        _stage("FIRST_READ",
               lambda: _burst_read_14(bus))
        print("[IMU]   First 14-byte burst read successful")
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — first sensor read failed: {exc}")
        try: bus.close()
        except Exception: pass
        return False

    # ── Stage: CALIBRATION ────────────────────────────────────────────────
    try:
        _stage("CALIBRATION",
               lambda: _calibrate(bus),
               retries=2, delay=0.1)
    except Exception as exc:
        imu_ok = False
        print(f"[IMU] INIT FAILED — calibration: {exc}")
        try: bus.close()
        except Exception: pass
        return False

    # ── READY ─────────────────────────────────────────────────────────────
    _bus            = bus
    _last_ts        = time.monotonic()
    _read_fail_count = 0
    imu_ok          = True
    print(f"[IMU] READY — MPU-6500 at 0x{_ADDR:02X} on SMBus(1)  (WHO_AM_I=0x{who:02X})")
    return True


# ---------------------------------------------------------------------------
# Public API — read()
# ---------------------------------------------------------------------------
def read() -> dict:
    """
    Read one sample from the MPU-6500 and return a dict structurally identical
    to the 'imu' sub-object emitted by the .ino emitTelemetry().

    Uses a single 14-byte I²C burst read (register 0x3B, 14 bytes):
      bytes [0:5]  → accel  X/Y/Z  (signed 16-bit, big-endian)
      bytes [6:7]  → temperature    (signed 16-bit, big-endian)
      bytes [8:13] → gyro   X/Y/Z  (signed 16-bit, big-endian)

    Output shape:
    {
        "valid": bool,
        "accel_g":             {"x": float, "y": float, "z": float},
        "accel_ms2":           {"x": float, "y": float, "z": float},
        "accel_magnitude_ms2": float,
        "gyro_dps":            {"x": float, "y": float, "z": float},
        "temperature_c":       float,
        "orientation":         {"roll": float, "pitch": float, "yaw": float,
                                "yaw_source": "GYRO_INTEGRATED"},
        "quaternion":          {"w": float, "x": float, "y": float, "z": float},
    }
    Returns {"valid": False} if imu_ok is False.

    Recovery:
      Transient I²C errors are retried up to _READ_RETRIES times.
      After _READ_RETRIES consecutive failures across calls, a lightweight
      re-initialisation is attempted automatically.
    """
    global _last_ts, _read_fail_count

    if not imu_ok or _bus is None:
        return {"valid": False}

    last_exc = None
    for attempt in range(1, _READ_RETRIES + 1):
        try:
            ax_r, ay_r, az_r, t_r, gx_r, gy_r, gz_r = _burst_read_14(_bus)

            # Scale + apply calibration offsets
            ax = ax_r / _ACCEL_SCALE - _ax_off
            ay = ay_r / _ACCEL_SCALE - _ay_off
            az = az_r / _ACCEL_SCALE - _az_off
            gx = gx_r / _GYRO_SCALE  - _gx_off
            gy = gy_r / _GYRO_SCALE  - _gy_off
            gz = gz_r / _GYRO_SCALE  - _gz_off
            temperature_c = t_r / 333.87 + 21.0

            # Complementary filter (under lock for thread safety)
            now_ts = time.monotonic()
            with _lock:
                dt      = now_ts - _last_ts
                _last_ts = now_ts
                _update_orientation(ax, ay, az, gx, gy, gz, dt)
                roll  = _roll_deg
                pitch = _pitch_deg
                yaw   = _yaw_deg

            # Derived values
            mag  = math.sqrt(ax*ax + ay*ay + az*az) * _G_MS2
            quat = _euler_to_quaternion(roll, pitch, yaw)

            # Successful read — reset failure counter
            _read_fail_count = 0

            return {
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

        except Exception as exc:
            last_exc = exc
            print(
                f"[IMU] NORMAL_READ attempt {attempt}/{_READ_RETRIES} failed: "
                f"{type(exc).__name__} {exc}"
            )
            if attempt < _READ_RETRIES:
                time.sleep(_READ_RETRY_DELAY)

    # All retries exhausted — increment persistent failure counter
    _read_fail_count += 1
    print(f"[IMU] Read failed after {_READ_RETRIES} attempts "
          f"(consecutive failures: {_read_fail_count})")

    # Attempt lightweight re-initialisation after sustained failures
    if _read_fail_count >= _READ_RETRIES:
        print("[IMU] Attempting automatic re-initialisation…")
        if _reinit_bus():
            _read_fail_count = 0
        else:
            print("[IMU] Re-initialisation unsuccessful — IMU marked not OK")

    return {"valid": False}
