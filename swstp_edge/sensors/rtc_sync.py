"""
sensors/rtc_sync.py — RTC / NTP synchronisation utility for the SWSTP Pi node.

Timezone
--------
Configured for Maharashtra, India: Asia/Kolkata (IST, UTC+05:30).

Boot & Recalibration Strategy
-----------------------------
1. Boot Initialization:
   - Checks if internet access is available immediately after boot.
   - If internet is online: queries NTP (or HTTP Date header fallback) to ensure
     maximum time accuracy, setting both the Linux system clock and the DS3231 RTC module.
   - If offline / no internet: immediately refers to the Raspberry Pi's local machine time
     and sets the DS3231 hardware RTC module to match it without stalling boot.

2. Hourly Internet Recalibration:
   - Background daemon thread runs every hour (`periodic_sync_loop(interval_hours=1.0)`).
   - Recalibrates both the system clock and the DS3231 RTC module from internet time.

Register map (DS3231, I2C 0x68)
    0x00  seconds    BCD, bits[6:0]
    0x01  minutes    BCD, bits[6:0]
    0x02  hours      BCD, bits[5:0] (24 h when bit6=0)
    0x03  day-of-week 1-7
    0x04  date       BCD, bits[5:0]
    0x05  month      BCD, bits[4:0] (bit7 = century flag)
    0x06  year       BCD (00-99, relative to century)
    0x0F  status     bit7 = OSF (oscillator stop flag — 1 = time invalid)
"""

import argparse
import datetime
import os
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Ensure safe terminal encoding across all platforms / serial lines
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# DS3231 constants
# ---------------------------------------------------------------------------
_DS3231_ADDR   = 0x68
_REG_SECONDS   = 0x00   # first register; read 7 bytes for full time
_REG_STATUS    = 0x0F
_OSF_BIT       = 0x80   # bit 7 of status register

# ---------------------------------------------------------------------------
# NTP & HTTP constants
# ---------------------------------------------------------------------------
_NTP_EPOCH_DELTA = 2208988800   # seconds between 1900-01-01 and 1970-01-01
_NTP_PACKET_FMT  = "!12I"       # 12 unsigned 32-bit integers, big-endian
_NTP_PORT        = 123
_NTP_TIMEOUT_SEC = 2.5          # fast timeout to avoid delaying boot

# Prioritize Indian NTP pool servers for optimal latency in Maharashtra
_NTP_SERVERS = [
    "0.in.pool.ntp.org",
    "1.in.pool.ntp.org",
    "2.in.pool.ntp.org",
    "3.in.pool.ntp.org",
    "time.google.com",
    "time.cloudflare.com",
    "pool.ntp.org",
    "time.windows.com",
]

_HTTP_TIME_URLS = [
    "https://clients3.google.com/generate_204",
    "https://www.cloudflare.com",
    "https://www.google.com",
]


# ---------------------------------------------------------------------------
# Timezone helpers (Maharashtra, India: Asia/Kolkata / IST, UTC+05:30)
# ---------------------------------------------------------------------------
def get_local_tz() -> datetime.tzinfo:
    """Return tzinfo for India/Maharashtra (Asia/Kolkata, UTC+05:30)."""
    try:
        from config import get_timezone_obj
        return get_timezone_obj()
    except Exception:
        pass
    try:
        import zoneinfo
        return zoneinfo.ZoneInfo("Asia/Kolkata")
    except Exception:
        return datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")


def get_local_now() -> datetime.datetime:
    """Return current wall-clock datetime in India/Maharashtra timezone (Asia/Kolkata / IST)."""
    return datetime.datetime.now(get_local_tz())


def configure_os_timezone(tz_name: str = "Asia/Kolkata") -> None:
    """Set process timezone environment variable (TZ) and call time.tzset() if supported."""
    try:
        os.environ["TZ"] = tz_name
        if hasattr(time, "tzset"):
            time.tzset()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class SyncResult:
    success:      bool
    source:       str           # "internet_ntp" | "internet_http" | "pi_local" | "ds3231" | "system"
    utc_time:     Optional[datetime.datetime] = None
    local_time:   Optional[datetime.datetime] = None
    ntp_server:   Optional[str] = None
    rtc_written:  bool          = False
    sysclock_set: bool          = False
    error:        Optional[str] = None


# ---------------------------------------------------------------------------
# BCD helpers
# ---------------------------------------------------------------------------
def _dec_to_bcd(n: int) -> int:
    return ((n // 10) << 4) | (n % 10)


def _bcd_to_dec(b: int) -> int:
    return ((b >> 4) & 0x0F) * 10 + (b & 0x0F)


# ---------------------------------------------------------------------------
# NTP query (raw UDP socket — zero external dependencies)
# ---------------------------------------------------------------------------
def _query_ntp(host: str, timeout: float = _NTP_TIMEOUT_SEC) -> Optional[datetime.datetime]:
    """
    Send a SNTPv4 request and return the server's transmit timestamp as a
    UTC datetime, or None on failure.
    """
    try:
        data = bytearray(48)
        data[0] = 0x1B       # LI=0, VN=3, Mode=3 (client)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(timeout)
            s.sendto(bytes(data), (host, _NTP_PORT))
            raw, _ = s.recvfrom(1024)

        if len(raw) < 48:
            return None

        unpacked = struct.unpack(_NTP_PACKET_FMT, raw[:48])
        tx_secs  = unpacked[10] - _NTP_EPOCH_DELTA
        tx_frac  = unpacked[11]
        unix_ts  = tx_secs + tx_frac / 2**32

        return datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc)

    except Exception:
        return None


def query_ntp(servers=None, timeout=_NTP_TIMEOUT_SEC):
    """Try each NTP server in order; return (datetime_utc, server_name) or (None, None)."""
    for host in (servers or _NTP_SERVERS):
        dt = _query_ntp(host, timeout)
        if dt is not None:
            return dt, host
    return None, None


# ---------------------------------------------------------------------------
# HTTP Date header fallback
# ---------------------------------------------------------------------------
def query_http_time(urls=None, timeout: float = 2.0) -> Optional[datetime.datetime]:
    """
    Fallback method to fetch internet time via HTTP Date header.
    Particularly effective on cellular / 4G connections where UDP port 123 may be blocked.
    """
    import email.utils
    import urllib.request

    for url in (urls or _HTTP_TIME_URLS):
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "SWSTP-Edge-RTC/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                date_hdr = resp.headers.get("Date")
                if date_hdr:
                    parsed = email.utils.parsedate_to_datetime(date_hdr)
                    if parsed is not None:
                        return parsed.astimezone(datetime.timezone.utc)
        except Exception:
            continue
    return None


def query_internet_time(servers=None, timeout: float = _NTP_TIMEOUT_SEC):
    """
    Query internet time using NTP first, falling back to HTTP Date headers.
    Returns (datetime_utc, source_label, server_name) or (None, None, None).
    """
    dt, host = query_ntp(servers=servers, timeout=timeout)
    if dt is not None:
        return dt, "internet_ntp", host

    dt_http = query_http_time(timeout=2.0)
    if dt_http is not None:
        return dt_http, "internet_http", "HTTP Date header"

    return None, None, None


# ---------------------------------------------------------------------------
# System clock setter
# ---------------------------------------------------------------------------
def _set_system_clock(dt: datetime.datetime, dry_run: bool = False) -> bool:
    """
    Set the Linux system clock to dt (converted to UTC).

    Tries three methods in order:
      1. `sudo date -u -s` (works if sudo is passwordless for date)
      2. `date -u -s` directly (works if running as root)
      3. Python ctypes clock_settime (requires CAP_SYS_TIME)
    """
    dt_utc = dt.astimezone(datetime.timezone.utc) if dt.tzinfo else dt
    iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    if dry_run:
        print(f"  [DRY-RUN] Would set system clock to: {iso} UTC")
        return True

    # Method 1: sudo / direct date
    for cmd in (
        ["sudo", "date", "-u", "-s", iso],
        ["date", "-u", "-s", iso],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except Exception:
            pass

    # Method 2: Python ctypes (requires root / CAP_SYS_TIME)
    try:
        import ctypes
        import ctypes.util
        CLOCK_REALTIME = 0
        class Timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        ts = Timespec()
        ts.tv_sec  = int(dt_utc.timestamp())
        ts.tv_nsec = 0
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.clock_settime(CLOCK_REALTIME, ctypes.byref(ts)) == 0:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# DS3231 direct smbus2 access
# ---------------------------------------------------------------------------
def _ds3231_read(bus) -> Optional[datetime.datetime]:
    """
    Read the DS3231 time registers.  Returns None if the oscillator-stop flag
    (OSF) is set (power was lost, time is invalid).
    """
    try:
        status = bus.read_byte_data(_DS3231_ADDR, _REG_STATUS)
        if status & _OSF_BIT:
            print("  [RTC] OSF flag set - DS3231 lost power; time unreliable.")
            return None

        raw = bus.read_i2c_block_data(_DS3231_ADDR, _REG_SECONDS, 7)
        sec  = _bcd_to_dec(raw[0] & 0x7F)
        mn   = _bcd_to_dec(raw[1] & 0x7F)
        hr   = _bcd_to_dec(raw[2] & 0x3F)   # 24-h mode
        day  = _bcd_to_dec(raw[4] & 0x3F)
        mon  = _bcd_to_dec(raw[5] & 0x1F)
        yr   = _bcd_to_dec(raw[6]) + 2000   # DS3231 stores 00-99

        return datetime.datetime(yr, mon, day, hr, mn, sec,
                                  tzinfo=datetime.timezone.utc)
    except Exception as exc:
        print(f"  [RTC] smbus2 read error: {exc}")
        return None


def _ds3231_write(bus, dt: datetime.datetime, dry_run: bool = False) -> bool:
    """
    Write dt (converted to UTC) to the DS3231 time registers and clear OSF flag.
    """
    dt_utc = dt.astimezone(datetime.timezone.utc) if dt.tzinfo else dt
    if dry_run:
        print(f"  [DRY-RUN] Would write {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC to DS3231.")
        return True
    try:
        dow = dt_utc.isoweekday()   # Monday=1 ... Sunday=7
        yr  = dt_utc.year - 2000
        regs = [
            _dec_to_bcd(dt_utc.second),
            _dec_to_bcd(dt_utc.minute),
            _dec_to_bcd(dt_utc.hour),   # 24-h, bit6=0
            dow,
            _dec_to_bcd(dt_utc.day),
            _dec_to_bcd(dt_utc.month),
            _dec_to_bcd(yr),
        ]
        bus.write_i2c_block_data(_DS3231_ADDR, _REG_SECONDS, regs)

        # Clear OSF flag in status register (bit 7)
        status = bus.read_byte_data(_DS3231_ADDR, _REG_STATUS)
        bus.write_byte_data(_DS3231_ADDR, _REG_STATUS, status & ~_OSF_BIT)
        return True
    except Exception as exc:
        print(f"  [RTC] smbus2 write error: {exc}")
        return False


def _hwclock_systohc(dry_run: bool = False) -> bool:
    """Write system clock -> DS3231 using the kernel hwclock tool."""
    if dry_run:
        print("  [DRY-RUN] Would run: sudo hwclock --systohc")
        return True
    for cmd in (["sudo", "hwclock", "--systohc"], ["hwclock", "--systohc"]):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _hwclock_hctosys(dry_run: bool = False) -> bool:
    """Read DS3231 -> system clock using the kernel hwclock tool."""
    if dry_run:
        print("  [DRY-RUN] Would run: sudo hwclock --hctosys")
        return True
    for cmd in (["sudo", "hwclock", "--hctosys"], ["hwclock", "--hctosys"]):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _write_rtc_module(bus, dt_target: datetime.datetime, dry_run: bool = False) -> bool:
    """Helper to write time to DS3231 using direct smbus2 or kernel hwclock."""
    written = False
    if bus is not None:
        written = _ds3231_write(bus, dt_target, dry_run=dry_run)
    if not written:
        written = _hwclock_systohc(dry_run=dry_run)
    return written


# ---------------------------------------------------------------------------
# Core Synchronization Logic
# ---------------------------------------------------------------------------
def sync(ntp_servers=None, dry_run: bool = False, verbose: bool = True,
         try_internet: bool = True) -> SyncResult:
    """
    Main clock synchronisation entry point called at boot.

    Logic:
      1. Configure process timezone for Maharashtra, India (Asia/Kolkata / IST).
      2. If internet access is available after boot (try_internet=True):
         Use internet time (NTP / HTTP) to set the system clock and DS3231 RTC module.
      3. If internet is NOT available:
         Refer to the Raspberry Pi's local machine time, setting the DS3231 RTC module
         to match it without blocking boot.
    """
    def log(msg):
        if verbose:
            print(msg)

    configure_os_timezone("Asia/Kolkata")
    local_tz = get_local_tz()

    log("\n[RTC-SYNC] Initialising clock synchronisation (Timezone: Asia/Kolkata, IST)...")

    # Open smbus2 if available
    bus = None
    try:
        import smbus2  # type: ignore
        bus = smbus2.SMBus(1)
    except Exception as exc:
        log(f"  [RTC-SYNC] smbus2 unavailable ({exc}) - using hwclock commands if needed.")

    # ── Step 1: Internet Check (if enabled for boot) ────────────────────
    if try_internet:
        log("  [1/2] Checking internet access for accurate network time...")
        net_time, src_label, srv_name = query_internet_time(servers=ntp_servers, timeout=_NTP_TIMEOUT_SEC)

        if net_time is not None:
            local_time = net_time.astimezone(local_tz)
            log(f"  [1/2] [OK] Internet access available: {local_time.strftime('%Y-%m-%d %H:%M:%S')} IST ({srv_name})")

            result = SyncResult(
                success=True,
                source=src_label,
                utc_time=net_time,
                local_time=local_time,
                ntp_server=srv_name,
            )

            # Update system clock
            result.sysclock_set = _set_system_clock(net_time, dry_run=dry_run)
            if result.sysclock_set:
                log("  [1/2] [OK] Linux system clock calibrated from internet.")
            else:
                log("  [1/2] [WARN] System clock set skipped (requires sudo/root privileges).")

            # Write accurate time to DS3231 hardware RTC
            result.rtc_written = _write_rtc_module(bus, net_time, dry_run=dry_run)
            if result.rtc_written:
                log(f"  [1/2] [OK] DS3231 RTC module calibrated with internet time.")
            else:
                log("  [1/2] [WARN] DS3231 RTC write unsuccessful.")

            if bus is not None:
                try: bus.close()
                except Exception: pass

            log(f"[RTC-SYNC] Synchronisation complete. Source: {src_label.upper()}.\n")
            return result

        log("  [1/2] [FAIL] Internet unavailable at boot.")

    # ── Step 2: Fallback to Raspberry Pi Local Machine Time ──────────────
    log("  [2/2] Referring to Raspberry Pi local machine time...")
    local_now = get_local_now()
    utc_now   = local_now.astimezone(datetime.timezone.utc)

    log(f"  [2/2] [OK] Local machine time: {local_now.strftime('%Y-%m-%d %H:%M:%S')} IST")

    result = SyncResult(
        success=True,
        source="pi_local",
        utc_time=utc_now,
        local_time=local_now,
        sysclock_set=True,
    )

    # Set the DS3231 hardware RTC module using the local machine time
    result.rtc_written = _write_rtc_module(bus, utc_now, dry_run=dry_run)
    if result.rtc_written:
        log("  [2/2] [OK] DS3231 RTC module set from Raspberry Pi local machine time.")
    else:
        log("  [2/2] [WARN] Could not write to DS3231 (using local system clock).")

    if bus is not None:
        try: bus.close()
        except Exception: pass

    log("[RTC-SYNC] Synchronisation complete. Source: PI_LOCAL (Timezone: Asia/Kolkata).\n")
    return result


# ---------------------------------------------------------------------------
# Hourly Internet Recalibration
# ---------------------------------------------------------------------------
def recalibrate_from_internet(ntp_servers=None, dry_run: bool = False,
                              verbose: bool = True) -> SyncResult:
    """
    Recalibrate the system clock and DS3231 RTC module using internet time (NTP/HTTP).
    Invoked every hour by periodic_sync_loop.
    """
    def log(msg):
        if verbose:
            print(msg)

    local_tz = get_local_tz()
    log("\n[RTC-RECALIBRATE] Starting hourly internet recalibration...")

    bus = None
    try:
        import smbus2  # type: ignore
        bus = smbus2.SMBus(1)
    except Exception:
        pass

    net_time, src_label, srv_name = query_internet_time(servers=ntp_servers, timeout=3.0)

    if net_time is not None:
        local_time = net_time.astimezone(local_tz)
        log(f"  [RECALIBRATE] [OK] Internet time verified: {local_time.strftime('%Y-%m-%d %H:%M:%S')} IST ({srv_name})")

        sysclock_set = _set_system_clock(net_time, dry_run=dry_run)
        rtc_written  = _write_rtc_module(bus, net_time, dry_run=dry_run)

        # Update rtc.sync_source if sensors.rtc module is loaded
        try:
            import sensors.rtc as _rtc
            _rtc.sync_source = src_label
        except Exception:
            pass

        if bus is not None:
            try: bus.close()
            except Exception: pass

        log(f"  [RECALIBRATE] [OK] Clock & DS3231 calibrated successfully (sysclock={sysclock_set}, rtc={rtc_written}).\n")
        return SyncResult(
            success=True,
            source=src_label,
            utc_time=net_time,
            local_time=local_time,
            ntp_server=srv_name,
            rtc_written=rtc_written,
            sysclock_set=sysclock_set,
        )

    if bus is not None:
        try: bus.close()
        except Exception: pass

    log("  [RECALIBRATE] [WARN] Internet unreachable during hourly recalibration; current timing maintained.\n")
    local_now = get_local_now()
    return SyncResult(
        success=False,
        source="system",
        utc_time=local_now.astimezone(datetime.timezone.utc),
        local_time=local_now,
        error="Internet unreachable for recalibration",
    )


# ---------------------------------------------------------------------------
# Periodic Sync Loop (Hourly)
# ---------------------------------------------------------------------------
def periodic_sync_loop(interval_hours: float = 1.0, stop_event=None) -> None:
    """
    Recalibrate the system clock and DS3231 from internet every `interval_hours` (default: 1.0 h = hourly).
    Designed to run as a background daemon thread from main.py.
    """
    import threading
    _stop = stop_event or threading.Event()
    while not _stop.is_set():
        _stop.wait(interval_hours * 3600)
        if not _stop.is_set():
            recalibrate_from_internet(verbose=True)


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="SWSTP RTC/NTP synchronisation utility (Maharashtra / Asia/Kolkata)."
    )
    parser.add_argument("--ntp-host", nargs="+", default=None, metavar="HOST",
                        help="NTP server(s) to query")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without modifying hardware or clock")
    parser.add_argument("--interval", type=float, default=0,
                        help="If > 0, run recalibration loop every N hours")
    parser.add_argument("--no-internet", action="store_true",
                        help="Force local machine time without checking internet")
    args = parser.parse_args()

    result = sync(ntp_servers=args.ntp_host, dry_run=args.dry_run, verbose=True,
                  try_internet=not args.no_internet)

    print("-" * 55)
    print(f"  Timezone      : Asia/Kolkata (IST, UTC+05:30)")
    print(f"  Sync source   : {result.source.upper()}")
    print(f"  Local time    : {result.local_time.strftime('%Y-%m-%d %H:%M:%S') if result.local_time else 'unknown'} IST")
    print(f"  UTC time      : {result.utc_time.strftime('%Y-%m-%d %H:%M:%S') if result.utc_time else 'unknown'} UTC")
    print(f"  System clock  : {'updated' if result.sysclock_set else 'not updated'}")
    print(f"  DS3231 written: {'yes' if result.rtc_written else 'no'}")
    if result.ntp_server:
        print(f"  Server        : {result.ntp_server}")
    if result.error:
        print(f"  Warning       : {result.error}")
    print("-" * 55)

    if args.interval > 0:
        print(f"\n[RTC-SYNC] Entering periodic recalibration loop (every {args.interval} h). Ctrl-C to stop.\n")
        try:
            while True:
                time.sleep(args.interval * 3600)
                recalibrate_from_internet(ntp_servers=args.ntp_host, dry_run=args.dry_run, verbose=True)
        except KeyboardInterrupt:
            print("\n[RTC-SYNC] Stopped.")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    _cli()
