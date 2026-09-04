#!/usr/bin/env python3
"""
test_all_sensors.py — Standalone diagnostic script for SWSTP Edge sensors.

Initialises and reads all three sensor modules in order:
  1. RTC  — DS3231 via I2C (with internet/NTP sync attempt)
  2. IMU  — MPU-6500 via I2C (with calibration)
  3. GNSS — NavCast TCP NMEA stream (background thread)

Prints clearly formatted output for every module.
This file is 100% independent — it makes NO changes to any other file.

Run from the swstp_edge/ directory:
    python test_all_sensors.py [--imu-samples N] [--gnss-wait N]

Optional flags:
  --imu-samples N   Number of IMU read iterations to display   (default: 5)
  --gnss-wait   N   Seconds to wait for a GNSS fix/data        (default: 30)
  --no-rtc          Skip RTC initialisation (useful if no DS3231 is wired)
  --no-imu          Skip IMU initialisation
  --no-gnss         Skip GNSS initialisation
"""

import argparse
import datetime
import json
import sys
import time

# ── Pretty-print helpers ─────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
DIM    = "\033[2m"


def _hr(char="─", width=70, color=DIM):
    print(f"{color}{char * width}{RESET}")


def _section(title: str, color=CYAN):
    _hr("═", color=color)
    print(f"{color}{BOLD}  {title}{RESET}")
    _hr("═", color=color)


def _ok(msg: str):
    print(f"  {GREEN}✔  {msg}{RESET}")


def _warn(msg: str):
    print(f"  {YELLOW}⚠  {msg}{RESET}")


def _err(msg: str):
    print(f"  {RED}✘  {msg}{RESET}")


def _field(label: str, value, unit: str = "", indent: int = 4):
    pad = " " * indent
    label_str = f"{BOLD}{label:<28}{RESET}"
    value_str = f"{GREEN}{value}{RESET}" if value is not None else f"{RED}None{RESET}"
    unit_str  = f"{DIM} {unit}{RESET}" if unit else ""
    print(f"{pad}{label_str}{value_str}{unit_str}")


def _sub(title: str):
    print(f"\n  {BLUE}{BOLD}▶ {title}{RESET}")


# ── Path bootstrap (run from any CWD) ────────────────────────────────────────

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ════════════════════════════════════════════════════════════════════════════
# 1. RTC
# ════════════════════════════════════════════════════════════════════════════

def test_rtc():
    _section("RTC — DS3231 Hardware Clock  (I2C 0x68)", color=CYAN)

    try:
        from sensors import rtc as rtc_mod
    except ImportError as exc:
        _err(f"Cannot import sensors.rtc: {exc}")
        return False

    # ── Init (runs sync + I2C probe) ─────────────────────────────────────
    print(f"\n  {DIM}Initialising RTC (NTP sync attempt will run now)…{RESET}")
    hw_present = rtc_mod.init()

    print()
    if hw_present:
        _ok(f"DS3231 detected on I2C bus (0x68)")
    else:
        _warn("DS3231 NOT detected on I2C bus — timestamps still valid (software RTC)")
        if rtc_mod.rtc_error:
            _err(f"Detail : {rtc_mod.rtc_error}")
        _warn("Pi diagnostic  : sudo i2cdetect -y 1")
        _warn("  '68' = device present, smbus access OK")
        _warn("  'UU' = device present but claimed by kernel rtc driver (dtoverlay=i2c-rtc,ds3231)")
        _warn("  '--' = device absent / wiring fault")

    _field("Hardware present",  hw_present)
    _field("Module status OK",  rtc_mod.rtc_ok)
    _field("Sync source",       rtc_mod.sync_source)
    _field("Error",             rtc_mod.rtc_error or "None")

    # ── Read ─────────────────────────────────────────────────────────────
    _sub("RTC read() output")
    data = rtc_mod.read()
    _field("valid",             data.get("valid"))
    _field("timestamp (IST)",   data.get("timestamp"))
    _field("epoch",             data.get("epoch"),  "ms")
    _field("timezone",          data.get("timezone"))

    # Cross-check with system time
    now_sys = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    _sub("System clock cross-check")
    _field("System time (local)", now_sys)

    _hr()
    return True


# ════════════════════════════════════════════════════════════════════════════
# 2. IMU
# ════════════════════════════════════════════════════════════════════════════

def test_imu(num_samples: int = 5):
    _section("IMU — MPU-6500  (I2C 0x69)", color=MAGENTA)

    try:
        from sensors import imu as imu_mod
    except ImportError as exc:
        _err(f"Cannot import sensors.imu: {exc}")
        return False

    # ── Init ─────────────────────────────────────────────────────────────
    print(f"\n  {DIM}Initialising IMU (calibration takes ~1 s — keep still)…{RESET}")
    ok = imu_mod.init()
    print()

    if ok:
        _ok("MPU-6500 initialised and calibrated")
    else:
        _err("MPU-6500 init FAILED — check I2C wiring and address (AD0 HIGH → 0x69)")

    _field("imu_ok", imu_mod.imu_ok)

    if not ok:
        _hr()
        return False

    # ── Read loop ─────────────────────────────────────────────────────────
    print(f"\n  {DIM}Reading {num_samples} samples (0.5 s interval)…{RESET}\n")

    for i in range(1, num_samples + 1):
        data = imu_mod.read()
        print(f"  {BOLD}{BLUE}── Sample {i}/{num_samples} ──────────────────────────────────────{RESET}")

        if not data.get("valid"):
            _err("Read returned valid=False")
            time.sleep(0.5)
            continue

        # Accelerometer
        _sub("Accelerometer")
        ag = data.get("accel_g", {})
        am = data.get("accel_ms2", {})
        _field("accel_g   x / y / z",
               f"{ag.get('x'):+.4f}  {ag.get('y'):+.4f}  {ag.get('z'):+.4f}", "g")
        _field("accel_ms2 x / y / z",
               f"{am.get('x'):+.4f}  {am.get('y'):+.4f}  {am.get('z'):+.4f}", "m/s²")
        _field("magnitude",
               f"{data.get('accel_magnitude_ms2'):+.4f}", "m/s²")

        # Gyroscope
        _sub("Gyroscope")
        gd = data.get("gyro_dps", {})
        _field("gyro_dps  x / y / z",
               f"{gd.get('x'):+.4f}  {gd.get('y'):+.4f}  {gd.get('z'):+.4f}", "°/s")

        # Temperature
        _sub("Temperature")
        _field("temperature_c", f"{data.get('temperature_c'):.2f}", "°C")

        # Orientation (complementary filter)
        _sub("Orientation  (complementary filter)")
        ori = data.get("orientation", {})
        _field("roll",         f"{ori.get('roll'):+.3f}",  "°")
        _field("pitch",        f"{ori.get('pitch'):+.3f}", "°")
        _field("yaw",          f"{ori.get('yaw'):+.3f}",   "°  (gyro-integrated)")
        _field("yaw_source",   ori.get("yaw_source"))

        # Quaternion
        _sub("Quaternion  (Tait-Bryan Z-Y-X)")
        q = data.get("quaternion", {})
        _field("w / x / y / z",
               f"{q.get('w'):+.6f}  {q.get('x'):+.6f}  {q.get('y'):+.6f}  {q.get('z'):+.6f}")

        if i < num_samples:
            time.sleep(0.5)

    _hr()
    return True


# ════════════════════════════════════════════════════════════════════════════
# 3. GNSS
# ════════════════════════════════════════════════════════════════════════════

def test_gnss(wait_sec: int = 30):
    _section("GNSS — NavCast TCP NMEA  (USB tethering)", color=GREEN)

    try:
        from sensors import gnss as gnss_mod
    except ImportError as exc:
        _err(f"Cannot import sensors.gnss: {exc}")
        return False

    try:
        from config import NAVCAST_HOST, NAVCAST_PORT
    except ImportError:
        NAVCAST_HOST = "10.208.43.190"
        NAVCAST_PORT = 10110

    print(f"\n  {DIM}Starting NavCast TCP reader → {NAVCAST_HOST}:{NAVCAST_PORT}{RESET}")
    gnss_mod.init()

    # ── Wait for data ─────────────────────────────────────────────────────
    print(f"  {DIM}Waiting up to {wait_sec} s for GNSS data …{RESET}\n")
    deadline = time.time() + wait_sec
    got_data = False
    got_fix  = False
    spinner  = ["|", "/", "─", "\\"]
    idx      = 0

    while time.time() < deadline:
        snap = gnss_mod.read()
        status = snap.get("status", "NO_DATA")

        # Spinner progress
        spin = spinner[idx % len(spinner)]
        idx += 1
        remaining = max(0, int(deadline - time.time()))
        print(f"\r  {spin}  Status: {BOLD}{status:<10}{RESET}  "
              f"sats={snap.get('satellites', 0):2d}  "
              f"sats_in_view={snap.get('satellites_in_view', 0):2d}  "
              f"[{remaining:2d}s left]   ",
              end="", flush=True)

        if status == "FIX":
            got_fix  = True
            got_data = True
            break
        if status == "NO_FIX":
            got_data = True

        time.sleep(1.0)

    print()   # newline after spinner

    # ── Final snapshot ────────────────────────────────────────────────────
    data = gnss_mod.read()

    print()
    status = data.get("status")
    if status == "FIX":
        _ok(f"GNSS FIX acquired")
    elif status == "NO_FIX":
        _warn("GNSS data flowing — no positional fix yet")
    else:
        _err(f"No GNSS data received (status={status}). Check NavCast app & USB tethering.")

    _sub("GNSS read() snapshot")
    _field("data_received",       data.get("data_received"))
    _field("fix",                 data.get("fix"))
    _field("status",              data.get("status"))
    _field("latitude",            data.get("latitude"),           "°")
    _field("longitude",           data.get("longitude"),          "°")
    _field("altitude_m",          data.get("altitude_m"),         "m")
    _field("speed_kmh",           data.get("speed_kmh"),          "km/h")
    _field("course_deg",          data.get("course_deg"),         "°")
    _field("satellites (in fix)", data.get("satellites"),         "")
    _field("satellites_in_view",  data.get("satellites_in_view"), "")
    _field("satellite_records",   data.get("satellite_records"),  "")
    _field("hdop",                data.get("hdop"),               "")
    _field("gnss_ok flag",        gnss_mod.gnss_ok)

    # ── Satellite detail ──────────────────────────────────────────────────
    detail = data.get("satellites_detail")
    if detail:
        _sub(f"Satellite detail ({len(detail)} records)")
        header = (f"    {'PRN':>4}  {'Const':<6}  {'Elev°':>5}  "
                  f"{'Azim°':>5}  {'SNR dB':>6}  {'Used':>4}")
        print(f"  {DIM}{header}{RESET}")
        _hr(".", width=60, color=DIM)
        for s in detail[:20]:    # cap at 20 rows
            used_str = f"{GREEN}yes{RESET}" if s.get("used_in_fix") else f"{DIM}no{RESET}"
            snr = s.get("snr_db")
            snr_str = f"{snr:.1f}" if snr is not None else "  -"
            print(f"    {s.get('prn', '?'):>4}  "
                  f"{s.get('constellation', '?'):<6}  "
                  f"{s.get('elevation_deg', 0):>5}  "
                  f"{s.get('azimuth_deg', 0):>5}  "
                  f"{snr_str:>6}  "
                  f"{used_str}")
        if len(detail) > 20:
            print(f"    {DIM}... and {len(detail) - 20} more{RESET}")
        _hr(".", width=60, color=DIM)
    else:
        _warn("No satellite detail available (GSV not yet received or snapshot expired)")

    # ── Stop background thread ────────────────────────────────────────────
    gnss_mod.stop()

    _hr()
    return True


# ════════════════════════════════════════════════════════════════════════════
# Entry-point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SWSTP Edge -- full sensor diagnostic (RTC + IMU + GNSS)"
    )
    parser.add_argument("--imu-samples", type=int,  default=5,
                        help="Number of IMU read samples to display (default: 5)")
    parser.add_argument("--gnss-wait",   type=int,  default=30,
                        help="Seconds to wait for GNSS data (default: 30)")
    parser.add_argument("--no-rtc",  action="store_true", help="Skip RTC test")
    parser.add_argument("--no-imu",  action="store_true", help="Skip IMU test")
    parser.add_argument("--no-gnss", action="store_true", help="Skip GNSS test")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────
    print()
    _hr("=", color=BOLD + CYAN)
    print(f"{BOLD}{CYAN}{'SWSTP EDGE -- SENSOR DIAGNOSTIC':^70}{RESET}")
    print(f"{DIM}{'RTC  .  IMU  .  GNSS':^70}{RESET}")
    print(f"{DIM}{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'):^70}{RESET}")
    _hr("=", color=BOLD + CYAN)
    print()

    results = {}

    if not args.no_rtc:
        results["RTC"]  = test_rtc()
    else:
        print(f"  {DIM}[RTC]  skipped (--no-rtc){RESET}\n")

    if not args.no_imu:
        results["IMU"]  = test_imu(args.imu_samples)
    else:
        print(f"  {DIM}[IMU]  skipped (--no-imu){RESET}\n")

    if not args.no_gnss:
        results["GNSS"] = test_gnss(args.gnss_wait)
    else:
        print(f"  {DIM}[GNSS] skipped (--no-gnss){RESET}\n")

    # ── Summary ───────────────────────────────────────────────────────────
    _section("SUMMARY", color=BOLD + YELLOW)
    print()
    for module, passed in results.items():
        if passed is True:
            _ok(f"{module:<6} -- PASSED")
        elif passed is False:
            _err(f"{module:<6} -- FAILED / not detected")
        else:
            _warn(f"{module:<6} -- SKIPPED")
    print()
    _hr("=", color=BOLD + CYAN)
    print()


if __name__ == "__main__":
    main()
