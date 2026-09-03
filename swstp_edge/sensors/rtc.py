"""
sensors/rtc.py — DS3231 RTC sensor module for Raspberry Pi.

DESIGN
------
The DS3231 is driven by the Linux kernel's i2c-rtc driver, loaded via:

    # /boot/config.txt (or /boot/firmware/config.txt on newer Pi OS)
    dtoverlay=i2c-rtc,ds3231

At startup, `init()` calls `sensors.rtc_sync.sync()` which:
  1. Queries public NTP servers → writes accurate time to DS3231 and system clock.
  2. If internet is unavailable → reads DS3231 → sets system clock from it.
  3. As a last resort, uses the existing system clock unchanged.

After init, `read()` simply calls `datetime.now()` — the system clock IS the
RTC (disciplined either by NTP or by the DS3231 kernel driver).

VOLTAGE CAUTION ⚠️
------------------
Pi I2C (GPIO 2/SDA, GPIO 3/SCL) operates at 3.3 V.
If your DS3231 breakout board's pull-up resistors connect to VCC = 5 V, the
SDA/SCL lines will be pulled to 5 V — which exceeds the Pi's 3.3 V GPIO
absolute maximum.  In that case insert a bidirectional 3.3 V ↔ 5 V I2C
level-shifter (e.g. TXB0102, BSS138-based module) between the Pi and the
breakout board.  DS3231 breakout boards whose pull-ups connect to 3.3 V
(e.g. Adafruit #3013) are safe to connect directly.

I2C address: 0x68  (fixed in DS3231; matches the .ino RTC_DS3231 object)
"""

import datetime
import threading
import time

# ---------------------------------------------------------------------------
# Public state (read by telemetry.py)
# ---------------------------------------------------------------------------
rtc_ok:     bool = False          # True once the first successful read is done
rtc_error:  str | None = None     # Human-readable error string if not OK
sync_source: str = "unknown"      # "ntp" | "ds3231" | "system" | "unknown"


def _probe_rtc_i2c() -> bool:
    """
    Non-invasively confirm the DS3231 is present on the I2C bus by attempting
    to open /dev/i2c-1 and address 0x68.  Falls back gracefully — if smbus2
    is unavailable or the bus is absent the system clock is still used.
    """
    try:
        import smbus2  # type: ignore
        bus = smbus2.SMBus(1)
        # Read DS3231 register 0x00 (seconds) — should return 0x00–0x59 BCD
        val = bus.read_byte_data(0x68, 0x00)
        bus.close()
        return True
    except Exception:
        return False


def read() -> dict:
    """
    Return an RTC data dict that is structurally identical to what the .ino
    emitTelemetry() wrote into the 'rtc' JSON key.

    Output shape (mirrors emitTelemetry RTC block):
    {
        "valid": bool,
        "timestamp": "YYYY-MM-DDTHH:MM:SS",   # ISO-8601 local wall-clock
        "epoch": int                            # milliseconds since Unix epoch
    }
    """
    global rtc_ok, rtc_error
    now = datetime.datetime.now()
    epoch_ms = int(now.timestamp() * 1000)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    rtc_ok = True
    rtc_error = None
    return {
        "valid": True,
        "timestamp": ts,
        "epoch": epoch_ms,
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init() -> bool:
    """
    1. Run the NTP → DS3231 → system-clock sync cascade (via rtc_sync.sync()).
    2. Probe the I2C bus to confirm the DS3231 chip is present.
    3. Set module-level status flags.

    Returns True if the DS3231 is detected on the I2C bus.
    Even if the chip is absent, the system clock is still available and
    `read()` will return valid timestamps.
    """
    global rtc_ok, rtc_error, sync_source

    # ── Step 1: time synchronisation ─────────────────────────────────────
    try:
        from sensors.rtc_sync import sync as _sync
    except ImportError:
        # Fallback if run from inside the sensors/ directory
        try:
            from rtc_sync import sync as _sync  # type: ignore
        except ImportError:
            _sync = None

    if _sync is not None:
        result = _sync(verbose=True)
        sync_source = result.source
        if not result.success:
            print(f"[RTC] Clock sync warning: {result.error}")
    else:
        sync_source = "system"
        print("[RTC] rtc_sync module not found — using existing system clock.")

    # ── Step 2: I2C bus probe ────────────────────────────────────────────
    present = _probe_rtc_i2c()
    if present:
        rtc_ok    = True
        rtc_error = None
        print(f"[RTC] DS3231 detected on I2C bus 1 (0x68). Clock source: {sync_source.upper()}.")
    else:
        rtc_ok    = False
        rtc_error = "DS3231 not detected on I2C bus (check dtoverlay=i2c-rtc,ds3231 and wiring)"
        print(f"[RTC] WARNING: {rtc_error}")
        print(f"[RTC] System clock source: {sync_source.upper()} — timestamps still valid.")

    return present

