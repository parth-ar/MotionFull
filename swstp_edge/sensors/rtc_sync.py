"""
sensors/rtc_sync.py — RTC / NTP synchronisation utility for the SWSTP Pi node.

Purpose
-------
Ensures the system clock is accurate at startup using the following priority
cascade:

  Priority 1 — Internet (NTP)
    Query public NTP servers.  On success:
      a) Set the Linux system clock (requires `sudo date -s` or CAP_SYS_TIME).
      b) Write the accurate time to the DS3231 RTC via smbus2 direct register
         access OR via `hwclock --systohc` if the kernel i2c-rtc overlay owns
         the chip.
      The DS3231 now acts as a battery-backed backup for future offline starts.

  Priority 2 — DS3231 hardware RTC (internet unavailable)
    Read the DS3231 time registers via smbus2 (or `hwclock --hctosys`).
    The oscillator-stop flag (OSF) in register 0x0F is checked first; if set
    the DS3231 has lost power and its time is unreliable (falls through to
    Priority 3).

  Priority 3 — Existing system clock (last resort)
    If neither NTP nor the RTC is available the system clock is used unchanged
    and a warning is printed.

Standalone usage:
    python3 sensors/rtc_sync.py [--dry-run] [--ntp-host pool.ntp.org]

Module usage (called automatically from sensors/rtc.py init):
    from sensors.rtc_sync import sync
    result = sync()     # returns a SyncResult namedtuple

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
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# DS3231 constants
# ---------------------------------------------------------------------------
_DS3231_ADDR   = 0x68
_REG_SECONDS   = 0x00   # first register; read 7 bytes for full time
_REG_STATUS    = 0x0F
_OSF_BIT       = 0x80   # bit 7 of status register

# ---------------------------------------------------------------------------
# NTP constants (RFC 4330 SNTPv4 over UDP)
# ---------------------------------------------------------------------------
_NTP_EPOCH_DELTA = 2208988800   # seconds between 1900-01-01 and 1970-01-01
_NTP_PACKET_FMT  = "!12I"       # 12 unsigned 32-bit integers, big-endian
_NTP_PORT        = 123
_NTP_TIMEOUT_SEC = 3.0
_NTP_SERVERS     = [
    "time.cloudflare.com",
    "pool.ntp.org",
    "time.google.com",
    "time.windows.com",
    "0.pool.ntp.org",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class SyncResult:
    success:      bool
    source:       str           # "ntp" | "ds3231" | "system"
    utc_time:     Optional[datetime.datetime] = None
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
# NTP query (raw socket — no external dependency)
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

        # Transmit timestamp is at offset 40 (words 10 & 11)
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
# System clock setter
# ---------------------------------------------------------------------------
def _set_system_clock(dt_utc: datetime.datetime, dry_run: bool = False) -> bool:
    """
    Set the Linux system clock to dt_utc.

    Tries three methods in order:
      1. `sudo date -s` (works if sudo is passwordless for date)
      2. `date` directly (works if running as root)
      3. Python ctypes clock_settime (requires CAP_SYS_TIME)
    """
    iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    if dry_run:
        print(f"  [DRY-RUN] Would set system clock to: {iso} UTC")
        return True

    # Method 1: sudo date
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
            print("  [RTC] OSF flag set — DS3231 lost power; time unreliable.")
            return None

        raw = bus.read_i2c_block_data(_DS3231_ADDR, _REG_SECONDS, 7)
        sec  = _bcd_to_dec(raw[0] & 0x7F)
        mn   = _bcd_to_dec(raw[1] & 0x7F)
        hr   = _bcd_to_dec(raw[2] & 0x3F)   # 24-h mode
        # raw[3] = day-of-week (1-7) — not needed for datetime
        day  = _bcd_to_dec(raw[4] & 0x3F)
        mon  = _bcd_to_dec(raw[5] & 0x1F)
        yr   = _bcd_to_dec(raw[6]) + 2000   # DS3231 stores 00-99

        return datetime.datetime(yr, mon, day, hr, mn, sec,
                                  tzinfo=datetime.timezone.utc)
    except Exception as exc:
        print(f"  [RTC] smbus2 read error: {exc}")
        return None


def _ds3231_write(bus, dt_utc: datetime.datetime, dry_run: bool = False) -> bool:
    """
    Write dt_utc to the DS3231 time registers and clear the OSF flag.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would write {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC to DS3231.")
        return True
    try:
        dow = dt_utc.isoweekday()   # Monday=1 … Sunday=7
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
    """Write system clock → DS3231 using the kernel hwclock tool."""
    if dry_run:
        print("  [DRY-RUN] Would run: sudo hwclock --systohc")
        return True
    try:
        r = subprocess.run(["sudo", "hwclock", "--systohc"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _hwclock_hctosys(dry_run: bool = False) -> bool:
    """Read DS3231 → system clock using the kernel hwclock tool."""
    if dry_run:
        print("  [DRY-RUN] Would run: sudo hwclock --hctosys")
        return True
    try:
        r = subprocess.run(["sudo", "hwclock", "--hctosys"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main sync function
# ---------------------------------------------------------------------------
def sync(ntp_servers=None, dry_run: bool = False, verbose: bool = True) -> SyncResult:
    """
    Run the full NTP → DS3231 → system synchronisation cascade.

    Returns a SyncResult describing what happened.
    """
    def log(msg):
        if verbose:
            print(msg)

    log("\n[RTC-SYNC] Starting clock synchronisation…")

    # ── Open smbus2 once (used for both read and write if available) ──────
    bus = None
    try:
        import smbus2  # type: ignore
        bus = smbus2.SMBus(1)
    except Exception as exc:
        log(f"  [RTC-SYNC] smbus2 unavailable ({exc}) — will use hwclock commands only.")

    # ─────────────────────────────────────────────────────────────────────
    # Priority 1: NTP
    # ─────────────────────────────────────────────────────────────────────
    log("  [1/3] Querying NTP servers…")
    ntp_time, ntp_server = query_ntp(servers=ntp_servers)

    if ntp_time is not None:
        log(f"  [1/3] ✓ NTP OK: {ntp_time.strftime('%Y-%m-%d %H:%M:%S')} UTC  (server: {ntp_server})")
        result = SyncResult(success=True, source="ntp", utc_time=ntp_time, ntp_server=ntp_server)

        # Set system clock
        result.sysclock_set = _set_system_clock(ntp_time, dry_run=dry_run)
        if result.sysclock_set:
            log("  [1/3] ✓ System clock updated from NTP.")
        else:
            log("  [1/3] ⚠ Could not set system clock (need sudo / root).")

        # Write to DS3231 — try smbus2 first, fall back to hwclock
        if bus is not None:
            result.rtc_written = _ds3231_write(bus, ntp_time, dry_run=dry_run)
            if result.rtc_written:
                log(f"  [1/3] ✓ DS3231 RTC written via smbus2 (0x{_DS3231_ADDR:02X}).")
        if not result.rtc_written:
            # If system clock was set, hwclock --systohc copies it to the RTC
            if result.sysclock_set:
                result.rtc_written = _hwclock_systohc(dry_run=dry_run)
                if result.rtc_written:
                    log("  [1/3] ✓ DS3231 RTC written via hwclock --systohc.")
                else:
                    log("  [1/3] ⚠ hwclock --systohc failed; RTC not written.")

        if bus is not None:
            try: bus.close()
            except Exception: pass
        log("[RTC-SYNC] Done. Source: NTP\n")
        return result

    log("  [1/3] ✗ NTP unavailable (no internet or all servers timed out).")

    # ─────────────────────────────────────────────────────────────────────
    # Priority 2: DS3231 hardware
    # ─────────────────────────────────────────────────────────────────────
    log("  [2/3] Reading DS3231 hardware RTC…")
    rtc_time: Optional[datetime.datetime] = None

    if bus is not None:
        rtc_time = _ds3231_read(bus)
        if rtc_time is not None:
            log(f"  [2/3] ✓ DS3231 time: {rtc_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # If smbus2 unavailable or read failed, try hwclock --hctosys
    if rtc_time is None:
        ok = _hwclock_hctosys(dry_run=dry_run)
        if ok:
            # hwclock already updated the system clock; read it back
            rtc_time = datetime.datetime.now(datetime.timezone.utc)
            log(f"  [2/3] ✓ hwclock --hctosys OK: {rtc_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    if rtc_time is not None:
        result = SyncResult(success=True, source="ds3231", utc_time=rtc_time)

        # Set system clock from DS3231 if not done by hwclock above
        if bus is not None:   # smbus2 path — need to set clock manually
            result.sysclock_set = _set_system_clock(rtc_time, dry_run=dry_run)
            if result.sysclock_set:
                log("  [2/3] ✓ System clock set from DS3231.")
            else:
                log("  [2/3] ⚠ Could not set system clock (need sudo / root).")
        else:
            result.sysclock_set = True   # hwclock already did it

        if bus is not None:
            try: bus.close()
            except Exception: pass
        log("[RTC-SYNC] Done. Source: DS3231 hardware backup\n")
        return result

    log("  [2/3] ✗ DS3231 unavailable or OSF flag set.")

    # ─────────────────────────────────────────────────────────────────────
    # Priority 3: System clock as-is
    # ─────────────────────────────────────────────────────────────────────
    log("  [3/3] ⚠ Using existing system clock (may be inaccurate).")
    sys_time = datetime.datetime.now(datetime.timezone.utc)
    log(f"  [3/3] System clock: {sys_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    if bus is not None:
        try: bus.close()
        except Exception: pass

    log("[RTC-SYNC] Done. Source: system clock (fallback)\n")
    return SyncResult(
        success=False,
        source="system",
        utc_time=sys_time,
        error="NTP unreachable and DS3231 unavailable or invalid",
    )


# ---------------------------------------------------------------------------
# Periodic re-sync (background thread helper)
# ---------------------------------------------------------------------------
def periodic_sync_loop(interval_hours: float = 6.0, stop_event=None) -> None:
    """
    Re-sync the system clock and DS3231 from NTP every `interval_hours`.
    Designed to run as a daemon thread from main.py.

    If NTP is unavailable at a re-sync attempt the DS3231 is used, keeping the
    system clock accurate even over long offline periods.
    """
    import threading
    _stop = stop_event or threading.Event()
    while not _stop.is_set():
        _stop.wait(interval_hours * 3600)
        if not _stop.is_set():
            print("\n[RTC-SYNC] Scheduled re-sync…")
            sync(verbose=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="SWSTP RTC/NTP synchronisation utility — "
                    "sets DS3231 from internet time, falls back to DS3231 if offline."
    )
    parser.add_argument("--ntp-host", nargs="+", default=None,
                        metavar="HOST",
                        help="NTP server(s) to query (default: cloudflare, pool.ntp.org, google)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing anything")
    parser.add_argument("--interval", type=float, default=0,
                        help="If > 0, re-sync every N hours in a loop (Ctrl-C to stop)")
    args = parser.parse_args()

    result = sync(ntp_servers=args.ntp_host, dry_run=args.dry_run, verbose=True)

    print("─" * 50)
    print(f"  Sync source   : {result.source.upper()}")
    print(f"  UTC time      : {result.utc_time.strftime('%Y-%m-%d %H:%M:%S') if result.utc_time else 'unknown'}")
    print(f"  System clock  : {'updated' if result.sysclock_set else 'not updated'}")
    print(f"  DS3231 written: {'yes' if result.rtc_written else 'no'}")
    if result.ntp_server:
        print(f"  NTP server    : {result.ntp_server}")
    if result.error:
        print(f"  Warning       : {result.error}")
    print("─" * 50)

    if args.interval > 0:
        print(f"\n[RTC-SYNC] Entering periodic re-sync loop (every {args.interval} h). Ctrl-C to stop.\n")
        try:
            while True:
                time.sleep(args.interval * 3600)
                sync(ntp_servers=args.ntp_host, dry_run=args.dry_run, verbose=True)
        except KeyboardInterrupt:
            print("\n[RTC-SYNC] Stopped.")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    _cli()
