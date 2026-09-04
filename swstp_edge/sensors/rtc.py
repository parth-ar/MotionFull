"""
sensors/rtc.py — DS3231 RTC sensor module for Raspberry Pi.

Timezone:
  Asia/Kolkata (IST, UTC+05:30) for Maharashtra, India.

Design & Synchronization Flow:
  1. At boot: `init()` calls `sensors.rtc_sync.sync(try_internet=True)`:
     - Checks for internet access immediately after boot.
     - If online: Calibrates system clock and DS3231 RTC module using NTP/HTTP.
     - If offline: Refers to the Raspberry Pi's local machine time, setting the
       DS3231 hardware RTC module to match it.
  2. Hourly recalibration: A background worker in main.py calls
     `sensors.rtc_sync.recalibrate_from_internet()` once every hour.
  3. `read()` returns wall-clock timestamps in Indian Standard Time (IST).

I2C Address: 0x68 (DS3231 default)
"""

import datetime
import os
import threading
import time

# ---------------------------------------------------------------------------
# Timezone helper (Maharashtra, India: Asia/Kolkata / IST, UTC+05:30)
# ---------------------------------------------------------------------------
try:
    from config import get_timezone_obj
    IST_TZ = get_timezone_obj()
except Exception:
    try:
        import zoneinfo
        IST_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")
    except Exception:
        IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")

# ---------------------------------------------------------------------------
# Public state (read by telemetry.py)
# ---------------------------------------------------------------------------
rtc_ok:      bool = False          # True once first successful read is done
rtc_error:   str | None = None     # Human-readable error string if not OK
sync_source: str = "unknown"       # "internet_ntp" | "internet_http" | "pi_local" | "ds3231" | "system"


def _probe_rtc_i2c() -> bool:
    """
    Non-invasively confirm the DS3231 is present on the I2C bus by attempting
    to open /dev/i2c-1 (and /dev/i2c-0 as fallback) and reading address 0x68.

    Logs a specific, actionable error message on failure so the cause is
    immediately visible in the Pi console / systemd journal.

    Common failure reasons:
      - smbus2 not installed          → pip install smbus2
      - I2C not enabled               → sudo raspi-config (Interface Options)
      - dtoverlay=i2c-rtc,ds3231 active → 0x68 shows as 'UU' in i2cdetect;
                                          run: sudo i2cdetect -y 1
      - Wiring fault / no VCC to RTC  → check SDA/SCL/3.3 V/GND
      - Address 0x68 conflict with IMU (MPU-6500 default) — IMU here uses 0x69
    """
    for bus_num in (1, 0):  # try bus 1 first (standard Pi), then bus 0
        try:
            import smbus2  # type: ignore
            bus = smbus2.SMBus(bus_num)
            # Read DS3231 register 0x00 (seconds) — valid BCD range 0x00–0x59
            val = bus.read_byte_data(0x68, 0x00)
            bus.close()
            if bus_num != 1:
                print(f"[RTC] DS3231 found on I2C bus {bus_num} (not bus 1 — check dtoverlay config)")
            return True
        except ImportError:
            print("[RTC] smbus2 library not installed — run: pip install smbus2>=0.4.3")
            return False
        except FileNotFoundError:
            # /dev/i2c-N does not exist — try next bus number
            continue
        except OSError as exc:
            errno_val = getattr(exc, 'errno', None)
            if errno_val == 121:  # EREMOTEIO / Remote I/O error — device not on bus
                print(f"[RTC] DS3231 NOT present on I2C bus {bus_num} (0x68): Remote I/O error — "
                      "check SDA/SCL wiring and VCC (3.3 V) to RTC module")
            elif errno_val == 16:  # EBUSY — kernel driver claimed the device
                print(f"[RTC] I2C bus {bus_num} addr 0x68 is BUSY (kernel rtc driver holds it). "
                      "This is expected when dtoverlay=i2c-rtc,ds3231 is active. "
                      "Run 'sudo i2cdetect -y 1' — if 0x68 shows 'UU' the hardware IS present.")
            else:
                print(f"[RTC] I2C bus {bus_num} OSError probing 0x68: {exc}")
            # Don't break — try the next bus number only for FileNotFoundError
            return False
        except Exception as exc:
            print(f"[RTC] Unexpected error probing DS3231 on bus {bus_num}: {exc}")
            return False
    print("[RTC] /dev/i2c-1 and /dev/i2c-0 not found — enable I2C via raspi-config or add 'dtparam=i2c_arm=on' to /boot/config.txt")
    return False


def read() -> dict:
    """
    Return an RTC data dict for telemetry and overlay HUD.
    Timestamps are localized to Maharashtra, India (Asia/Kolkata, IST).

    Output shape:
    {
        "valid": bool,
        "timestamp": "YYYY-MM-DDTHH:MM:SS",   # ISO-8601 IST wall-clock
        "epoch": int,                           # milliseconds since Unix epoch
        "timezone": "Asia/Kolkata"
    }
    """
    global rtc_ok, rtc_error
    now = datetime.datetime.now(IST_TZ)
    epoch_ms = int(now.timestamp() * 1000)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    rtc_ok = True
    rtc_error = None
    return {
        "valid": True,
        "timestamp": ts,
        "epoch": epoch_ms,
        "timezone": "Asia/Kolkata",
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init() -> bool:
    """
    Boot initialization:
      1. Check internet access; if available, calibrate system clock and DS3231.
         If offline, set DS3231 module from Raspberry Pi local machine time.
      2. Probe I2C bus to verify physical DS3231 presence (0x68).
      3. Set module status flags.

    Returns True if DS3231 is detected on I2C bus.
    """
    global rtc_ok, rtc_error, sync_source

    # ── Step 1: Time synchronisation ─────────────────────────────────────
    try:
        from sensors.rtc_sync import sync as _sync, configure_os_timezone
        configure_os_timezone("Asia/Kolkata")
    except ImportError:
        try:
            from rtc_sync import sync as _sync, configure_os_timezone  # type: ignore
            configure_os_timezone("Asia/Kolkata")
        except ImportError:
            _sync = None

    if _sync is not None:
        result = _sync(verbose=True, try_internet=True)
        sync_source = result.source
        if not result.success:
            print(f"[RTC] Clock sync warning: {result.error}")
    else:
        sync_source = "pi_local"
        print("[RTC] rtc_sync module not found — using Raspberry Pi local machine time.")

    # ── Step 2: I2C bus probe ────────────────────────────────────────────
    present = _probe_rtc_i2c()
    if present:
        rtc_ok    = True
        rtc_error = None
        print(f"[RTC] DS3231 detected on I2C bus (0x68). Clock source: {sync_source.upper()} (Timezone: Asia/Kolkata).")
    else:
        rtc_ok    = False
        rtc_error = (
            "DS3231 not detected on I2C bus — "
            "verify: (1) sudo i2cdetect -y 1 shows '68' or 'UU' at address 0x68, "
            "(2) dtparam=i2c_arm=on in /boot/config.txt, "
            "(3) VCC 3.3 V and GND connected, "
            "(4) SDA→GPIO2 / SCL→GPIO3 wiring correct"
        )
        print(f"[RTC] WARNING: DS3231 hardware not detected. Reason logged above.")
        print(f"[RTC] Diagnostics: run 'sudo i2cdetect -y 1' on the Pi to confirm hardware presence.")
        print(f"[RTC] Clock source: {sync_source.upper()} (Timezone: Asia/Kolkata) — software timestamps remain valid.")

    return present
