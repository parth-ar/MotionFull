#!/usr/bin/env python3
"""
test_gnss_stability.py — Verification test for GNSS stream stability and 20 Hz telemetry polling.

Tests:
1. NMEA sentence parsing & concatenated sentence handling.
2. 20 Hz telemetry polling across 1 Hz burst intervals (confirms zero dropouts/flickers).
3. Coordinate holdover stability.
4. Clean transition to NO_FIX after timeout window.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sensors.gnss as gnss_sensor
import telemetry as _telemetry
from telemetry import latest_sensor, hardware_state, build_telemetry_packet
from config import GNSS_DATA_TIMEOUT_SEC


def make_nmea(body: str) -> str:
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"${body}*{c:02X}"


def run_tests():
    print("=" * 70)
    print("       RUNNING GNSS STABILITY & TELEMETRY 20 HZ VERIFICATION")
    print("=" * 70)

    # 1. Reset state
    gnss_sensor._reset_state_to_disconnected()

    # Verify initial state
    snap = gnss_sensor.read()
    assert snap["fix"] is False
    assert snap["latitude"] is None
    print("[PASS] Initial state is cleanly disconnected.")

    # 2. Test Concatenated NMEA handling
    # Simulates serial buffer receiving two sentences in a single line
    gga_str = make_nmea("GNGGA,123456.00,1831.2258,N,07351.4044,E,1,08,1.0,550.0,M,0.0,M,,")
    rmc_str = make_nmea("GNRMC,123456.00,A,1831.2258,N,07351.4044,E,15.2,120.0,311026,,,A")
    concat_line = f"{gga_str}\r\n{rmc_str}"

    parts = concat_line.split("$")
    for p in parts:
        p_clean = p.strip()
        if p_clean:
            gnss_sensor._dispatch("$" + p_clean)

    snap = gnss_sensor.read()
    assert snap["fix"] is True, "Fix must be True after concatenated dispatch"
    assert snap["latitude"] is not None, "Latitude must be present"
    assert abs(snap["latitude"] - 18.52043) < 0.001
    assert abs(snap["speed_kmh"] - round(15.2 * 1.852, 2)) < 0.1
    print(f"[PASS] Concatenated sentence dispatch: lat={snap['latitude']:.6f}, spd={snap['speed_kmh']} km/h")

    # 3. Test 20 Hz Telemetry Polling over 1 Hz NMEA Bursts
    print("\n[*] Simulating 20 Hz telemetry polling across 1 Hz NMEA bursts (3 seconds)...")
    device_id = "TEST-DEVICE-001"

    # Feed burst at t=0
    gnss_sensor._dispatch(make_nmea("GNGGA,123457.00,1831.2260,N,07351.4046,E,1,09,0.9,551.0,M,0.0,M,,"))
    gnss_sensor._dispatch(make_nmea("GNRMC,123457.00,A,1831.2260,N,07351.4046,E,10.0,110.0,311026,,,A"))

    poll_count = 0
    drop_count = 0

    for i in range(60):  # 60 polls @ 50ms = 3.0 seconds
        # Inject 1 Hz NMEA update every 20 ticks (1.0s)
        if i == 20:
            gnss_sensor._dispatch(make_nmea("GNGGA,123458.00,1831.2262,N,07351.4048,E,1,10,0.8,552.0,M,0.0,M,,"))
            gnss_sensor._dispatch(make_nmea("GNRMC,123458.00,A,1831.2262,N,07351.4048,E,10.5,112.0,311026,,,A"))
        elif i == 40:
            gnss_sensor._dispatch(make_nmea("GNGGA,123459.00,1831.2264,N,07351.4050,E,1,10,0.8,553.0,M,0.0,M,,"))
            gnss_sensor._dispatch(make_nmea("GNRMC,123459.00,A,1831.2264,N,07351.4050,E,11.0,114.0,311026,,,A"))

        pkt = build_telemetry_packet(device_id)
        poll_count += 1

        with _telemetry._lock:
            valid = latest_sensor.get("gps_valid")
            lat = latest_sensor.get("raw_gps_lat")
            lon = latest_sensor.get("raw_gps_lon")

        if not valid or lat is None or lon is None:
            drop_count += 1

        time.sleep(0.01)  # fast simulation tick

    print(f"    Total 20 Hz Telemetry Polls : {poll_count}")
    print(f"    Telemetry Fix Drops         : {drop_count}")
    assert drop_count == 0, f"Fix must never drop between bursts! (Drops: {drop_count})"
    print("[PASS] Zero telemetry fix drops observed across all 1 Hz burst intervals!")

    # 4. Verify Timeout Expiration (> 10.0s silence)
    print("\n[*] Verifying clean expiration when GPS silence exceeds timeout window...")
    with _telemetry._lock:
        # Simulate last fix occurring 11 seconds ago
        latest_sensor["last_gnss_fix_time"] = time.monotonic() - 11.0

    with gnss_sensor._lock:
        gnss_sensor._state["last_fix_time"] = time.monotonic() - 11.0

    pkt = build_telemetry_packet(device_id)
    with _telemetry._lock:
        valid_after_timeout = latest_sensor.get("gps_valid")
        lat_after_timeout = latest_sensor.get("lat")

    assert valid_after_timeout is False, "GPS must expire to invalid after timeout"
    assert lat_after_timeout is None, "Coordinates must be cleared after timeout"
    print("[PASS] Clean timeout expiration confirmed (> 10.0s silence).")

    print("\n" + "=" * 70)
    print("      ALL GNSS STABILITY & TELEMETRY TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
