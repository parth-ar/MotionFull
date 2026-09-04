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

Detection Priority:
  1. Linux kernel RTC subsystem (/sys/class/rtc/rtcX/name containing "1-0068")
     → Hardware present = True, access via /sys/class/rtc/rtcX/date+time
  2. Direct SMBus probe at bus 1 / bus 0, address 0x68
     → Hardware present = True, access via smbus2
  3. Both fail → Hardware present = False, software timestamp fallback
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
rtc_ok:      bool = False           # True once hardware is confirmed present
rtc_error:   str | None = None      # Human-readable error string if not OK
sync_source: str = "unknown"        # "internet_ntp"|"internet_http"|"pi_local"|"ds3231"|"system"

# Internal: set by init(), used by read() for kernel-path time reads
_kernel_rtc_dev:  str | None = None  # e.g. "rtc0"
_kernel_rtc_name: str | None = None  # e.g. "rtc-ds1307 1-0068"
_access_mode:     str = "none"       # "kernel" | "smbus" | "software"

# I2C address of the DS3231
_DS3231_ADDR = 0x68
# The bus+address identifier the kernel exposes for the DS3231 on bus 1
_DS3231_KERNEL_ID = "1-0068"


# ---------------------------------------------------------------------------
# Detection — Method 1: Linux kernel RTC subsystem
# ---------------------------------------------------------------------------
def _probe_kernel_rtc() -> tuple[bool, str | None, str | None]:
    """
    Scan /sys/class/rtc/ for an RTC device whose 'name' file contains
    the bus+address identifier '1-0068' (I²C bus 1, address 0x68).

    This is the correct way to detect a DS3231 (or compatible, e.g. ds1307
    driver) that has been claimed by the Linux kernel via dtoverlay.

    When the kernel owns the device:
      - i2cdetect shows 'UU' at address 0x68 (not '--')
      - /sys/class/rtc/rtc0/name reads e.g.  "rtc-ds1307 1-0068"
      - /sys/class/rtc/rtc0/date  and  /sys/class/rtc/rtc0/time  are readable
      - Direct smbus2 access to 0x68 will raise OSError errno=16 (EBUSY)

    Returns:
        (found: bool, rtc_dev: str|None, rtc_name: str|None)
        e.g. (True, "rtc0", "rtc-ds1307 1-0068")
    """
    rtc_base = "/sys/class/rtc"
    try:
        if not os.path.isdir(rtc_base):
            return False, None, None

        for entry in sorted(os.listdir(rtc_base)):          # rtc0, rtc1, …
            name_path = os.path.join(rtc_base, entry, "name")
            try:
                with open(name_path, "r") as f:
                    name_str = f.read().strip()
            except OSError:
                continue

            # Match on the bus+address token "1-0068" anywhere in the name line
            if _DS3231_KERNEL_ID in name_str:
                return True, entry, name_str

    except Exception as exc:
        print(f"[RTC] Error scanning {rtc_base}: {exc}")

    return False, None, None


def _read_kernel_rtc_time(rtc_dev: str) -> datetime.datetime | None:
    """
    Read UTC time from /sys/class/rtc/<rtc_dev>/date and .../time.
    Returns a UTC datetime or None on error.
    """
    rtc_dir = f"/sys/class/rtc/{rtc_dev}"
    try:
        with open(os.path.join(rtc_dir, "date"), "r") as f:
            date_str = f.read().strip()   # e.g. "2026-09-04"
        with open(os.path.join(rtc_dir, "time"), "r") as f:
            time_str = f.read().strip()   # e.g. "11:42:00"
        dt_utc = datetime.datetime.strptime(
            f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)
        return dt_utc
    except Exception as exc:
        print(f"[RTC] Could not read time from {rtc_dir}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Detection — Method 2: Direct SMBus probe (fallback)
# ---------------------------------------------------------------------------
def _probe_rtc_smbus() -> tuple[bool, int | None]:
    """
    Attempt direct SMBus communication with the DS3231 at address 0x68.
    Only called when no kernel-managed RTC is found.

    Returns (success: bool, bus_num: int|None).

    Common failure reasons when this also fails:
      - smbus2 not installed      → pip install smbus2>=0.4.3
      - I2C not enabled           → sudo raspi-config (Interface Options)
      - Wiring fault / no VCC     → check SDA/SCL/3.3 V/GND
      - errno=16 (EBUSY)          → kernel driver holds 0x68 but /sys/class/rtc
                                    scan missed it — re-check /sys/class/rtc
    """
    for bus_num in (1, 0):
        try:
            import smbus2  # type: ignore
            bus = smbus2.SMBus(bus_num)
            # Read DS3231 register 0x00 (seconds) — valid BCD range 0x00–0x59
            bus.read_byte_data(_DS3231_ADDR, 0x00)
            bus.close()
            return True, bus_num
        except ImportError:
            print("[RTC] smbus2 library not installed — run: pip install smbus2>=0.4.3")
            return False, None
        except FileNotFoundError:
            continue   # /dev/i2c-N absent — try next bus
        except OSError as exc:
            errno_val = getattr(exc, "errno", None)
            if errno_val == 16:   # EBUSY — kernel owns the device
                print(
                    f"[RTC] SMBus {bus_num}: 0x68 is BUSY (errno=16) — kernel RTC driver holds it.\n"
                    f"[RTC]   This is unexpected at this stage; check /sys/class/rtc/ manually."
                )
            elif errno_val == 121:  # EREMOTEIO — device not on bus
                print(
                    f"[RTC] SMBus {bus_num}: No response from 0x68 (errno=121) — "
                    "check SDA/SCL wiring and VCC (3.3 V) to RTC module."
                )
            else:
                print(f"[RTC] SMBus {bus_num} OSError probing 0x68: {exc}")
            return False, None
        except Exception as exc:
            print(f"[RTC] Unexpected error probing DS3231 on SMBus {bus_num}: {exc}")
            return False, None

    print(
        "[RTC] /dev/i2c-1 and /dev/i2c-0 not found — "
        "enable I2C via raspi-config or add 'dtparam=i2c_arm=on' to /boot/config.txt"
    )
    return False, None


# ---------------------------------------------------------------------------
# Public: read()
# ---------------------------------------------------------------------------
def read() -> dict:
    """
    Return an RTC data dict for telemetry and overlay HUD.
    Timestamps are localized to Maharashtra, India (Asia/Kolkata, IST).

    When a kernel-managed RTC is present, the hardware time is read directly
    from /sys/class/rtc/<dev>/date + time (UTC) and converted to IST.
    Otherwise, the system wall-clock (already NTP/RTC-synced) is used.

    Output shape:
    {
        "valid":     bool,
        "timestamp": "YYYY-MM-DDTHH:MM:SS",   # ISO-8601 IST wall-clock
        "epoch":     int,                       # milliseconds since Unix epoch
        "timezone":  "Asia/Kolkata",
        "hw_source": "kernel" | "smbus" | "software"
    }
    """
    global rtc_ok, rtc_error

    # Prefer kernel RTC hardware read when available
    if _access_mode == "kernel" and _kernel_rtc_dev is not None:
        dt_utc = _read_kernel_rtc_time(_kernel_rtc_dev)
        if dt_utc is not None:
            now = dt_utc.astimezone(IST_TZ)
            rtc_ok    = True
            rtc_error = None
            return {
                "valid":     True,
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "epoch":     int(now.timestamp() * 1000),
                "timezone":  "Asia/Kolkata",
                "hw_source": "kernel",
            }

    # Software fallback (NTP-synced system clock)
    now = datetime.datetime.now(IST_TZ)
    rtc_ok    = True
    rtc_error = None
    return {
        "valid":     True,
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch":     int(now.timestamp() * 1000),
        "timezone":  "Asia/Kolkata",
        "hw_source": _access_mode,
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init() -> bool:
    """
    Boot initialization:
      1. Run NTP/internet clock sync (or fall back to Pi local time).
      2. Detect DS3231 hardware using the priority order:
           a. Linux kernel RTC subsystem (/sys/class/rtc — handles UU/claimed devices)
           b. Direct SMBus probe at bus 1 / bus 0, address 0x68
      3. Set public status flags (rtc_ok, rtc_error, sync_source).

    Returns True if DS3231 hardware is confirmed present by either method.
    """
    global rtc_ok, rtc_error, sync_source
    global _kernel_rtc_dev, _kernel_rtc_name, _access_mode

    # ── Step 1: Time synchronisation ─────────────────────────────────────
    _sync = None
    try:
        from sensors.rtc_sync import sync as _sync, configure_os_timezone
        configure_os_timezone("Asia/Kolkata")
    except ImportError:
        try:
            from rtc_sync import sync as _sync, configure_os_timezone  # type: ignore
            configure_os_timezone("Asia/Kolkata")
        except ImportError:
            pass

    if _sync is not None:
        result = _sync(verbose=True, try_internet=True)
        sync_source = result.source
        if not result.success:
            print(f"[RTC] Clock sync warning: {result.error}")
    else:
        sync_source = "pi_local"
        print("[RTC] rtc_sync module not found — using Raspberry Pi local machine time.")

    # ── Step 2a: Kernel RTC subsystem detection (primary) ────────────────
    print("[RTC] Scanning Linux kernel RTC subsystem (/sys/class/rtc)...")
    kernel_found, rtc_dev, rtc_name = _probe_kernel_rtc()

    if kernel_found:
        _kernel_rtc_dev  = rtc_dev
        _kernel_rtc_name = rtc_name
        _access_mode     = "kernel"
        rtc_ok           = True
        rtc_error        = None

        # Verify the kernel RTC can actually be read right now
        dt_utc = _read_kernel_rtc_time(rtc_dev)
        time_readable = dt_utc is not None
        time_note = (
            dt_utc.astimezone(IST_TZ).strftime("%Y-%m-%d %H:%M:%S IST")
            if dt_utc else "time read failed (kernel RTC still present)"
        )

        print(
            f"[RTC] ✔ DS3231 detected via Linux kernel RTC subsystem.\n"
            f"[RTC]   Device       : /dev/{rtc_dev}  (/sys/class/rtc/{rtc_dev})\n"
            f"[RTC]   Kernel name  : {rtc_name}\n"
            f"[RTC]   I²C address  : 0x{_DS3231_ADDR:02X}  (bus 1)\n"
            f"[RTC]   Access mode  : Kernel-managed (direct SMBus not required)\n"
            f"[RTC]   Hardware time: {time_note}\n"
            f"[RTC]   Clock source : {sync_source.upper()}  (Timezone: Asia/Kolkata)"
        )
        return True

    print(
        f"[RTC] No kernel-managed RTC found matching '{_DS3231_KERNEL_ID}' in "
        f"/sys/class/rtc — falling back to direct SMBus probe."
    )

    # ── Step 2b: Direct SMBus probe (fallback) ───────────────────────────
    smbus_found, smbus_bus = _probe_rtc_smbus()

    if smbus_found:
        _access_mode = "smbus"
        rtc_ok       = True
        rtc_error    = None
        print(
            f"[RTC] ✔ DS3231 detected via direct SMBus access.\n"
            f"[RTC]   I²C bus      : {smbus_bus}\n"
            f"[RTC]   I²C address  : 0x{_DS3231_ADDR:02X}\n"
            f"[RTC]   Access mode  : Direct SMBus\n"
            f"[RTC]   Clock source : {sync_source.upper()}  (Timezone: Asia/Kolkata)"
        )
        return True

    # ── Step 2c: Both probes failed ───────────────────────────────────────
    _access_mode = "software"
    rtc_ok       = False
    rtc_error    = (
        f"DS3231 not detected — "
        f"neither kernel RTC subsystem (sought '{_DS3231_KERNEL_ID}' in /sys/class/rtc) "
        f"nor direct SMBus probe (bus 1/0, addr 0x{_DS3231_ADDR:02X}) succeeded. "
        f"Verify: (1) sudo i2cdetect -y 1 — '68'=direct OK, 'UU'=kernel owned, '--'=wiring fault; "
        f"(2) dtparam=i2c_arm=on in /boot/config.txt; "
        f"(3) VCC 3.3 V and GND connected; "
        f"(4) SDA→GPIO2 / SCL→GPIO3."
    )
    print(
        f"[RTC] ✘ DS3231 hardware NOT detected by any method.\n"
        f"[RTC]   Run 'sudo i2cdetect -y 1' on the Pi:\n"
        f"[RTC]     '68' → direct SMBus accessible (probe issue)\n"
        f"[RTC]     'UU' → kernel holds device (check /sys/class/rtc)\n"
        f"[RTC]     '--' → wiring fault or power missing\n"
        f"[RTC]   Clock source: {sync_source.upper()} (Timezone: Asia/Kolkata) — "
        f"software timestamps remain valid."
    )
    return False
