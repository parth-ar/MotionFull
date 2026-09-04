#!/usr/bin/env python3
"""
test_navcast_gnss.py — Standalone GNSS connection & accuracy test script.

Designed for testing phone GNSS data sent via NavCast (USB Serial or USB Tethering Network)
without modifying any core application code.

Features:
- Dual Connection Modes:
    1) Serial / COM Port (at 115200 baud or custom baud)
    2) Network Socket over USB Tethering (TCP client, TCP server, or UDP listener)
- Auto-detection of available COM ports and USB tethering gateway
- NMEA 0183 Parser ($GGA, $RMC, $GSA, $GSV, $VTG) with checksum validation
- Accuracy & Precision Analysis:
    - HDOP / VDOP / PDOP rating (Ideal / Good / Moderate / Poor)
    - Fix quality (SPS, DGPS, RTK Float, RTK Fixed)
    - Satellite count (used vs. in view) & average satellite SNR (dB-Hz)
    - Real-time stationary position drift (std dev & max distance spread in meters)
    - Update frequency (Hz) & sentence rate
- Interactive clean terminal dashboard + optional CSV / NMEA logging

Usage examples:
    # 1. Serial mode (specify COM port):
    python test_navcast_gnss.py --port COM3 --baud 115200

    # 2. Serial mode (auto-scan available COM ports):
    python test_navcast_gnss.py --scan

    # 3. USB Tethering TCP Client (NavCast TCP server on phone):
    # Running directly with your phone's address:
    python test_navcast_gnss.py --mode tcp --host 10.208.43.190 --net-port 10110
    # Or simply:
    python test_navcast_gnss.py --mode tcp

    # 4. USB Tethering UDP listener (NavCast broadcasting to laptop):
    python test_navcast_gnss.py --mode udp --net-port 10110

    # 5. Log raw NMEA and parsed coordinates to files:
    python test_navcast_gnss.py --port COM3 --log-nmea raw_gps.nmea --log-csv fixes.csv
"""

import argparse
import csv
import math
import os
import re
import socket
import sys
import time
from collections import deque
from datetime import datetime

# Optional dependency: pyserial for COM port access
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================================
# Math & Geodesy Utilities
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def dec_to_dms(deg: float, is_lat: bool) -> str:
    """Convert decimal degrees to Degrees, Minutes, Seconds format."""
    if deg is None:
        return "N/A"
    direction = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    deg_abs = abs(deg)
    d = int(deg_abs)
    m = int((deg_abs - d) * 60)
    s = (deg_abs - d - m / 60) * 3600
    return f"{d}d {m:02d}' {s:05.2f}\" {direction}"


def rate_hdop(hdop: float | None) -> tuple[str, str]:
    """Return rating and color tag for HDOP."""
    if hdop is None:
        return "UNKNOWN", ""
    if hdop <= 1.0:
        return "IDEAL (<1.0)", "\033[92m"       # Bright green
    if hdop <= 2.0:
        return "EXCELLENT (1-2)", "\033[92m"
    if hdop <= 5.0:
        return "GOOD (2-5)", "\033[96m"         # Cyan
    if hdop <= 10.0:
        return "MODERATE (5-10)", "\033[93m"    # Yellow
    return "POOR (>10)", "\033[91m"             # Red


# ============================================================================
# NMEA 0183 Parser
# ============================================================================

def verify_nmea_checksum(sentence: str) -> bool:
    """Verify NMEA XOR checksum."""
    if not sentence.startswith("$"):
        return False
    if "*" not in sentence:
        return False
    try:
        content, csum_str = sentence[1:].rsplit("*", 1)
        expected_csum = int(csum_str[:2], 16)
        calc_csum = 0
        for char in content:
            calc_csum ^= ord(char)
        return calc_csum == expected_csum
    except (ValueError, IndexError):
        return False


def parse_nmea_lat_lon(val: str, direction: str) -> float | None:
    """Parse NMEA format ddmm.mmmm or dddmm.mmmm to decimal degrees."""
    if not val or not direction:
        return None
    try:
        dot_idx = val.find(".")
        if dot_idx < 0:
            return None
        deg_len = dot_idx - 2
        deg = float(val[:deg_len])
        minutes = float(val[deg_len:])
        decimal = deg + (minutes / 60.0)
        if direction in ("S", "W"):
            decimal = -decimal
        return decimal
    except (ValueError, IndexError):
        return None


class GNSSState:
    """Accumulates and updates GNSS tracking status and metrics."""

    def __init__(self, history_len: int = 60):
        self.last_update = 0.0
        self.sentence_count = 0
        self.valid_sentence_count = 0
        self.checksum_errors = 0        # Position & motion
        self.utc_time: str | None = None
        self.latitude: float | None = None
        self.longitude: float | None = None
        self.altitude_m: float | None = None
        self.speed_kmh: float | None = None
        self.max_speed_kmh: float = 0.0
        self.speed_knots: float | None = None
        self.course_deg: float | None = None

        # Precision & Quality
        self.fix_status: str = "NO_FIX"
        self.fix_mode_dimension: str = "Unknown"  # 2D, 3D
        self.satellites_used: int = 0
        self.satellites_in_view: int = 0
        self.hdop: float | None = None
        self.vdop: float | None = None
        self.pdop: float | None = None
        self.sat_snr_list: list[int] = []

        # Rolling statistics for stationary accuracy assessment
        self.history_len = history_len
        self.lat_history = deque(maxlen=history_len)
        self.lon_history = deque(maxlen=history_len)
        self.alt_history = deque(maxlen=history_len)
        self.update_rates = deque(maxlen=20)
        self.last_fix_time = None
        self.last_fix_wall_time = 0.0
        self.fix_rates = deque(maxlen=20)

    def update_from_nmea(self, raw_line: str) -> bool:
        """Parse NMEA sentence and return True if new position coordinates were acquired."""
        self.sentence_count += 1
        line = raw_line.strip()
        if not line:
            return False

        if not verify_nmea_checksum(line):
            self.checksum_errors += 1
            return False

        self.valid_sentence_count += 1
        self.last_update = time.time()

        # Strip $ and checksum
        body = line[1:].split("*")[0]
        fields = body.split(",")
        talker = fields[0]

        had_new_fix = False
        # Handle sentence types regardless of talker prefix (GP, GN, GL, GA, etc.)
        if talker.endswith("GGA"):
            had_new_fix = self._parse_gga(fields)
        elif talker.endswith("RMC"):
            had_new_fix = self._parse_rmc(fields)
        elif talker.endswith("GSA"):
            self._parse_gsa(fields)
        elif talker.endswith("GSV"):
            self._parse_gsv(fields)
        elif talker.endswith("VTG"):
            self._parse_vtg(fields)

        return bool(had_new_fix)

    def _parse_gga(self, f: list[str]) -> bool:
        # $--GGA,time,lat,NS,lon,EW,quality,numSV,HDOP,alt,M,sep,M,diffAge,diffStation
        new_fix = False
        if len(f) > 1 and f[1]:
            raw_t = f[1]
            if "." in raw_t:
                hms, frac = raw_t.split(".", 1)
                frac_str = f".{frac[:2]}"
            else:
                hms = raw_t
                frac_str = ""
            if len(hms) >= 6:
                self.utc_time = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}{frac_str} UTC"
            else:
                self.utc_time = raw_t

        if len(f) > 5:
            lat = parse_nmea_lat_lon(f[2], f[3])
            lon = parse_nmea_lat_lon(f[4], f[5])
            if lat is not None and lon is not None:
                self.latitude = lat
                self.longitude = lon
                self.lat_history.append(lat)
                self.lon_history.append(lon)
                self.last_fix_time = time.time()
                new_fix = True

                now = time.time()
                if self.last_fix_wall_time > 0:
                    dt = now - self.last_fix_wall_time
                    if 0.01 <= dt <= 5.0:
                        self.fix_rates.append(1.0 / dt)
                self.last_fix_wall_time = now

        if len(f) > 6 and f[6]:
            q = f[6]
            q_map = {
                "0": "NO_FIX",
                "1": "GPS_SPS_FIX",
                "2": "DGPS_FIX",
                "3": "PPS_FIX",
                "4": "RTK_FIXED",
                "5": "RTK_FLOAT",
                "6": "ESTIMATED",
            }
            self.fix_status = q_map.get(q, f"TYPE_{q}")

        if len(f) > 7 and f[7]:
            try:
                self.satellites_used = int(f[7])
            except ValueError:
                pass

        if len(f) > 8 and f[8]:
            try:
                self.hdop = float(f[8])
            except ValueError:
                pass

        if len(f) > 9 and f[9]:
            try:
                self.altitude_m = float(f[9])
                self.alt_history.append(self.altitude_m)
            except ValueError:
                pass

        return new_fix

    def _parse_rmc(self, f: list[str]) -> bool:
        # $--RMC,time,status,lat,NS,lon,EW,spd,cog,date,mv,mvEW,posMode
        new_fix = False
        if len(f) > 1 and f[1]:
            raw_t = f[1]
            if "." in raw_t:
                hms, frac = raw_t.split(".", 1)
                frac_str = f".{frac[:2]}"
            else:
                hms = raw_t
                frac_str = ""
            if len(hms) >= 6:
                self.utc_time = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}{frac_str} UTC"
            else:
                self.utc_time = raw_t

        if len(f) > 2 and f[2]:
            if f[2] == "A" and self.fix_status == "NO_FIX":
                self.fix_status = "2D/3D_FIX"
            elif f[2] == "V" and self.fix_status not in ("GPS_SPS_FIX", "DGPS_FIX", "RTK_FIXED", "RTK_FLOAT"):
                self.fix_status = "NO_FIX"

        if len(f) > 6:
            lat = parse_nmea_lat_lon(f[3], f[4])
            lon = parse_nmea_lat_lon(f[5], f[6])
            if lat is not None and lon is not None:
                self.latitude = lat
                self.longitude = lon
                new_fix = True

        if len(f) > 7 and f[7]:
            try:
                self.speed_knots = float(f[7])
                self.speed_kmh = self.speed_knots * 1.852
                if self.speed_kmh > self.max_speed_kmh:
                    self.max_speed_kmh = self.speed_kmh
            except ValueError:
                pass

        if len(f) > 8 and f[8]:
            try:
                self.course_deg = float(f[8])
            except ValueError:
                pass

        return new_fix

    def _parse_gsa(self, f: list[str]):
        # $--GSA,mode1,mode2,sv1..sv12,PDOP,HDOP,VDOP
        if len(f) > 2 and f[2]:
            mode_map = {"1": "No Fix", "2": "2D Fix", "3": "3D Fix"}
            self.fix_mode_dimension = mode_map.get(f[2], f[2])

        if len(f) > 15 and f[15]:
            try:
                self.pdop = float(f[15])
            except ValueError:
                pass
        if len(f) > 16 and f[16]:
            try:
                self.hdop = float(f[16])
            except ValueError:
                pass
        if len(f) > 17 and f[17]:
            try:
                self.vdop = float(f[17])
            except ValueError:
                pass

    def _parse_gsv(self, f: list[str]):
        # $--GSV,total_msgs,msg_num,total_sats, [prn, elev, az, snr] * 4
        if len(f) > 3 and f[3]:
            try:
                self.satellites_in_view = int(f[3])
            except ValueError:
                pass

        # Extract SNR values to compute signal strength
        msg_num = int(f[2]) if len(f) > 2 and f[2].isdigit() else 1
        if msg_num == 1:
            self.sat_snr_list.clear()

        idx = 4
        while idx + 3 < len(f):
            snr_str = f[idx + 3]
            if snr_str and snr_str.isdigit():
                self.sat_snr_list.append(int(snr_str))
            idx += 4

    def _parse_vtg(self, f: list[str]):
        # $--VTG,cog_t,T,cog_m,M,sog_n,N,sog_k,K
        if len(f) > 1 and f[1]:
            try:
                self.course_deg = float(f[1])
            except ValueError:
                pass
        if len(f) > 7 and f[7]:
            try:
                self.speed_kmh = float(f[7])
                if self.speed_kmh > self.max_speed_kmh:
                    self.max_speed_kmh = self.speed_kmh
            except ValueError:
                pass

    def calculate_accuracy_metrics(self) -> dict:
        """Calculate stationary jitter, standard deviation, and estimated accuracy."""
        n = len(self.lat_history)
        if n < 3:
            return {
                "samples": n,
                "lat_std_m": 0.0,
                "lon_std_m": 0.0,
                "max_drift_m": 0.0,
                "estimated_accuracy_m": (self.hdop * 2.5) if self.hdop else 0.0,
                "avg_snr": sum(self.sat_snr_list) / len(self.sat_snr_list) if self.sat_snr_list else 0.0,
                "rate_hz": (sum(self.update_rates) / len(self.update_rates)) if self.update_rates else 0.0,
            }

        mean_lat = sum(self.lat_history) / n
        mean_lon = sum(self.lon_history) / n

        # Distance from mean for each point
        dists = [haversine_distance(mean_lat, mean_lon, lat, lon)
                 for lat, lon in zip(self.lat_history, self.lon_history)]

        # Lat / Lon std in meters (1 deg lat ~= 111,320m)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))

        lat_var = sum((lat - mean_lat) ** 2 for lat in self.lat_history) / (n - 1)
        lon_var = sum((lon - mean_lon) ** 2 for lon in self.lon_history) / (n - 1)

        lat_std_m = math.sqrt(lat_var) * m_per_deg_lat
        lon_std_m = math.sqrt(lon_var) * m_per_deg_lon
        max_drift_m = max(dists) if dists else 0.0

        # Estimated 95% 2DRMS horizontal accuracy derived from HDOP
        # Typical standalone GPS UERE ~= 2.5 to 3.5m
        uere = 2.5
        est_acc_m = (self.hdop * uere * 2.0) if self.hdop is not None else 0.0

        avg_snr = sum(self.sat_snr_list) / len(self.sat_snr_list) if self.sat_snr_list else 0.0
        rate_hz = sum(self.fix_rates) / len(self.fix_rates) if self.fix_rates else 0.0

        return {
            "samples": n,
            "lat_std_m": lat_std_m,
            "lon_std_m": lon_std_m,
            "max_drift_m": max_drift_m,
            "estimated_accuracy_m": est_acc_m,
            "avg_snr": avg_snr,
            "rate_hz": rate_hz,
        }


# ============================================================================
# Terminal UI & Output Display
# ============================================================================

_first_render = True

def render_dashboard(state: GNSSState, connection_info: str):
    """Render a clean ANSI dashboard focused on Lat, Lon, Altitude, and Travel Speed with zero flicker."""
    global _first_render
    acc = state.calculate_accuracy_metrics()
    hdop_label, hdop_color = rate_hdop(state.hdop)
    reset = "\033[0m"
    bold = "\033[1m"
    cyan = "\033[96m"
    green = "\033[92m"
    yellow = "\033[93m"
    gray = "\033[90m"
    clr = "\033[K"  # Clear from cursor to end of line to prevent ghost characters

    has_fix = state.latitude is not None and state.longitude is not None
    fix_color = green if (has_fix and "FIX" in state.fix_status) else yellow

    # Clear terminal screen once, then overwrite in-place for flicker-free 10Hz updates
    if _first_render:
        sys.stdout.write("\033[2J\033[H")
        _first_render = False
    else:
        sys.stdout.write("\033[H")

    out = [
        f"{bold}{cyan}+==============================================================================+{reset}{clr}",
        f"{bold}{cyan}|                  NAVCAST GNSS REAL-TIME POSITION MONITOR                     |{reset}{clr}",
        f"{bold}{cyan}+==============================================================================+{reset}{clr}",
        f" {bold}Connection:{reset} {connection_info} | {bold}Status:{reset} {fix_color}{state.fix_status} ({state.fix_mode_dimension}){reset}{clr}",
        f" {bold}Satellites:{reset} {state.satellites_used} used / {state.satellites_in_view} in view | {bold}Fix Rate:{reset} {bold}{green}{acc['rate_hz']:.1f} Hz{reset}{clr}",
        f"{gray}{'-' * 78}{reset}{clr}",
        f"{bold}[POSITION & TRAVEL SPEED]{reset}{clr}"
    ]

    if has_fix:
        lat_dms = dec_to_dms(state.latitude, True)
        lon_dms = dec_to_dms(state.longitude, False)
        alt_str = f"{state.altitude_m:.2f} m" if state.altitude_m is not None else "Acquiring..."
        spd_str = f"{state.speed_kmh:.2f} km/hr" if state.speed_kmh is not None else "0.00 km/hr"
        max_spd_str = f"{state.max_speed_kmh:.2f} km/hr"

        out.append(f"  >> {bold}LATITUDE     :{reset}  {green}{state.latitude:12.6f} deg{reset}   ({lat_dms}){clr}")
        out.append(f"  >> {bold}LONGITUDE    :{reset}  {green}{state.longitude:12.6f} deg{reset}   ({lon_dms}){clr}")
        out.append(f"  >> {bold}ALTITUDE     :{reset}  {cyan}{alt_str:<16}{reset}{clr}")
        out.append(f"  >> {bold}TRAVEL SPEED :{reset}  {bold}{spd_str:<16}{reset} (Peak: {max_spd_str}){clr}")
        if state.course_deg is not None:
            out.append(f"  >> {bold}HEADING      :{reset}  {state.course_deg:.1f} deg{clr}")
        if state.utc_time:
            out.append(f"  >> {bold}TIMESTAMP    :{reset}  {state.utc_time}{clr}")
    else:
        out.append(f"  {yellow}Waiting for valid GNSS fix from phone... (Check GPS / Location permissions in NavCast){reset}{clr}")
        out.append(f"  >> {bold}LATITUDE     :{reset}  --{clr}")
        out.append(f"  >> {bold}LONGITUDE    :{reset}  --{clr}")
        out.append(f"  >> {bold}ALTITUDE     :{reset}  --{clr}")
        out.append(f"  >> {bold}TRAVEL SPEED :{reset}  0.00 km/hr{clr}")

    out.append(f"{gray}{'-' * 78}{reset}{clr}")
    out.append(f"{bold}[ACCURACY & DILUTION OF PRECISION]{reset}{clr}")
    hdop_str = f"{state.hdop:.2f}" if state.hdop is not None else "N/A"
    vdop_str = f"{state.vdop:.2f}" if state.vdop is not None else "N/A"
    pdop_str = f"{state.pdop:.2f}" if state.pdop is not None else "N/A"

    out.append(f"  * HDOP (Horizontal): {hdop_color}{hdop_str:<6} [{hdop_label}]{reset}{clr}")
    out.append(f"  * VDOP (Vertical)  : {vdop_str:<6}   * PDOP: {pdop_str}{clr}")

    est_acc = acc["estimated_accuracy_m"]
    est_acc_str = f"+/- {est_acc:.2f} meters (95% confidence)" if est_acc > 0 else "Calculating..."
    out.append(f"  * Estimated Accuracy: {bold}{est_acc_str}{reset}{clr}")

    snr_rating = "Strong" if acc["avg_snr"] >= 35 else ("Moderate" if acc["avg_snr"] >= 25 else "Weak")
    out.append(f"  * Carrier C/N0 SNR : {acc['avg_snr']:.1f} dB-Hz ({snr_rating}){clr}")

    if acc["samples"] >= 3:
        out.append(f"{gray}{'-' * 78}{reset}{clr}")
        out.append(f"{bold}[STATIONARY JITTER & STABILITY ANALYSIS ({acc['samples']} Fixes)]{reset}{clr}")
        out.append(f"  * Lat Spread StdDev : +/- {acc['lat_std_m']:.3f} m{clr}")
        out.append(f"  * Lon Spread StdDev : +/- {acc['lon_std_m']:.3f} m{clr}")
        out.append(f"  * Max Drift Spread  : {acc['max_drift_m']:.3f} m from centroid{clr}")

    out.append(f"{gray}{'-' * 78}{reset}{clr}")
    out.append(f" {bold}Press Ctrl+C to stop test and save summary report.{reset}{clr}\n")
    sys.stdout.write("\n".join(out))
    sys.stdout.flush()


def print_stream_line(state: GNSSState):
    """Print a single clean, formatted line whenever a new fix arrives."""
    if state.latitude is None or state.longitude is None:
        return
    t_str = state.utc_time or datetime.now().strftime("%H:%M:%S")
    alt_str = f"{state.altitude_m:.1f}m" if state.altitude_m is not None else "--"
    spd_str = f"{state.speed_kmh:.1f} km/hr" if state.speed_kmh is not None else "0.0 km/hr"
    hdop_str = f"{state.hdop:.1f}" if state.hdop is not None else "--"
    print(f"[{t_str}] Lat: {state.latitude:11.6f} deg | Lon: {state.longitude:11.6f} deg | Alt: {alt_str:>8} | Speed: {spd_str:>12} | HDOP: {hdop_str:>4} | Sats: {state.satellites_used}")
    sys.stdout.flush()


# ============================================================================
# Connection Handlers
# ============================================================================

def scan_available_ports():
    """Scan and print available serial ports."""
    if not SERIAL_AVAILABLE:
        print("\n[!] 'pyserial' is not available. Install it with: pip install pyserial")
        return []
    ports = list(serial.tools.list_ports.comports())
    print("\n--- Available COM / Serial Ports ---")
    if not ports:
        print("  No COM ports found. If using USB tethering, check Network mode (--mode tcp / udp).")
    for p in ports:
        print(f"  Port: {p.device:<8} | Description: {p.description}")
    print("-----------------------------------\n")
    return [p.device for p in ports]


def run_serial_stream(port: str, baud: int, state: GNSSState, log_nmea_fp, log_csv_writer, stream_mode: bool = False):
    """Read NMEA from serial port."""
    if not SERIAL_AVAILABLE:
        print("ERROR: pyserial is required for serial mode. Run: pip install pyserial")
        sys.exit(1)

    print(f"[*] Opening Serial Port {port} at {baud} baud (8-N-1)...")
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.02)
    except Exception as e:
        print(f"\n[!] Failed to open serial port {port}: {e}")
        scan_available_ports()
        sys.exit(1)

    connection_info = f"SERIAL {port} @ {baud} bps"
    last_render = 0.0

    try:
        while True:
            raw_line = ser.readline()
            if raw_line:
                try:
                    line = raw_line.decode("ascii", errors="replace").strip()
                except Exception:
                    continue

                if line.startswith("$"):
                    has_new_fix = state.update_from_nmea(line)
                    if log_nmea_fp:
                        log_nmea_fp.write(line + "\n")
                        log_nmea_fp.flush()
                    if log_csv_writer and state.latitude is not None:
                        log_csv_writer.writerow([
                            datetime.now().isoformat(), state.utc_time, state.latitude,
                            state.longitude, state.altitude_m, state.speed_kmh,
                            state.course_deg, state.hdop, state.satellites_used, state.fix_status
                        ])
                    if stream_mode and has_new_fix:
                        print_stream_line(state)

            if not stream_mode:
                now = time.time()
                if now - last_render >= 0.08:  # 12.5 Hz refresh capability
                    render_dashboard(state, connection_info)
                    last_render = now

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


def run_tcp_client_stream(host: str, port: int, state: GNSSState, log_nmea_fp, log_csv_writer, stream_mode: bool = False):
    """Connect to NavCast TCP server running on the phone."""
    print(f"[*] Connecting to NavCast TCP server at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        sock.connect((host, port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(0.02)  # Low-latency polling
        print(f"[+] Connected to {host}:{port} successfully!")
    except Exception as e:
        print(f"\n[!] Connection failed to {host}:{port}: {e}")
        print("\nTroubleshooting tips for USB Tethering:")
        print(" 1. Ensure USB Tethering is turned ON in Android Settings.")
        print(" 2. In NavCast, check the output settings: ensure TCP Server is enabled and note the Port (e.g. 10110).")
        print(" 3. Check Phone IP (usually 192.168.42.129 or 10.208.43.190).")
        sys.exit(1)

    connection_info = f"TCP CLIENT -> {host}:{port}"
    last_render = 0.0
    buffer = ""

    try:
        while True:
            try:
                data = sock.recv(4096).decode("ascii", errors="replace")
                if not data:
                    print("\n[!] Connection closed by remote server.")
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("$"):
                        has_new_fix = state.update_from_nmea(line)
                        if log_nmea_fp:
                            log_nmea_fp.write(line + "\n")
                            log_nmea_fp.flush()
                        if log_csv_writer and state.latitude is not None:
                            log_csv_writer.writerow([
                                datetime.now().isoformat(), state.utc_time, state.latitude,
                                state.longitude, state.altitude_m, state.speed_kmh,
                                state.course_deg, state.hdop, state.satellites_used, state.fix_status
                            ])
                        if stream_mode and has_new_fix:
                            print_stream_line(state)
            except socket.timeout:
                pass

            if not stream_mode:
                now = time.time()
                if now - last_render >= 0.08:  # 12.5 Hz refresh capability
                    render_dashboard(state, connection_info)
                    last_render = now

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def run_udp_stream(bind_host: str, port: int, state: GNSSState, log_nmea_fp, log_csv_writer, stream_mode: bool = False):
    """Listen for NavCast UDP broadcast stream."""
    print(f"[*] Binding UDP listener on {bind_host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_host, port))
    sock.settimeout(2.0)

    connection_info = f"UDP LISTENER @ {bind_host}:{port}"
    last_render = 0.0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(2048)
                lines = data.decode("ascii", errors="replace").splitlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("$"):
                        has_new_fix = state.update_from_nmea(line)
                        if log_nmea_fp:
                            log_nmea_fp.write(line + "\n")
                            log_nmea_fp.flush()
                        if log_csv_writer and state.latitude is not None:
                            log_csv_writer.writerow([
                                datetime.now().isoformat(), state.utc_time, state.latitude,
                                state.longitude, state.altitude_m, state.speed_kmh,
                                state.course_deg, state.hdop, state.satellites_used, state.fix_status
                            ])
                        if stream_mode and has_new_fix:
                            print_stream_line(state)
            except socket.timeout:
                pass

            if not stream_mode:
                now = time.time()
                if now - last_render >= 0.08:
                    render_dashboard(state, f"{connection_info} (Active)")
                    last_render = now

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NavCast GNSS Connection & Accuracy Test Tool for Phone to Laptop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run directly with phone defaults (TCP 10.208.43.190:10110):
  python test_navcast_gnss.py

  # Stream scrolling fixes line-by-line:
  python test_navcast_gnss.py --stream

  # Test via Serial COM port (Virtual COM at 115200 baud):
  python test_navcast_gnss.py --mode serial --port COM3 --baud 115200

  # Auto scan available COM ports:
  python test_navcast_gnss.py --scan

  # Test via USB Tethering UDP listener:
  python test_navcast_gnss.py --mode udp --net-port 10110
        """
    )
    parser.add_argument("--mode", choices=["serial", "tcp", "udp"], default="tcp",
                        help="Connection mode: 'tcp' (phone TCP server, default), 'serial' (COM port), or 'udp' (listener)")
    parser.add_argument("--port", default=None,
                        help="Serial COM port (e.g. COM3, COM4, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Serial baud rate (default: 115200)")
    parser.add_argument("--host", default="10.208.43.190",
                        help="NavCast TCP server host IP (default: 10.208.43.190)")
    parser.add_argument("--net-port", type=int, default=10110,
                        help="Network port for TCP or UDP (default: 10110)")
    parser.add_argument("--stream", action="store_true",
                        help="Print scrolling single-line logs of Lat/Lon/Alt/Speed instead of dashboard")
    parser.add_argument("--scan", action="store_true",
                        help="Scan and list detected COM ports, then exit")
    parser.add_argument("--log-nmea", default=None,
                        help="Save raw NMEA sentences to file (e.g., output.nmea)")
    parser.add_argument("--log-csv", default=None,
                        help="Save parsed GPS fixes to CSV file (e.g., fixes.csv)")

    args = parser.parse_args()

    if args.scan:
        scan_available_ports()
        return

    state = GNSSState(history_len=100)

    log_nmea_fp = None
    if args.log_nmea:
        log_nmea_fp = open(args.log_nmea, "w", encoding="utf-8")
        print(f"[+] Logging raw NMEA sentences to: {args.log_nmea}")

    log_csv_fp = None
    log_csv_writer = None
    if args.log_csv:
        log_csv_fp = open(args.log_csv, "w", newline="", encoding="utf-8")
        log_csv_writer = csv.writer(log_csv_fp)
        log_csv_writer.writerow([
            "timestamp", "utc_time", "latitude", "longitude", "altitude_m",
            "speed_kmh", "course_deg", "hdop", "satellites_used", "fix_status"
        ])
        print(f"[+] Logging parsed fixes to CSV: {args.log_csv}")

    print("\nStarting NavCast GNSS Test...")

    try:
        if args.mode == "serial":
            port = args.port
            if not port:
                available = scan_available_ports()
                if not available:
                    print("\n[!] No serial COM port specified and none detected automatically.")
                    print("    If your phone is sending data via USB Tethering over TCP, run:")
                    print("      python test_navcast_gnss.py --mode tcp --host 10.208.43.190 --net-port 10110")
                    sys.exit(1)
                port = available[0]
                print(f"[*] Auto-selected first available port: {port}")
            run_serial_stream(port, args.baud, state, log_nmea_fp, log_csv_writer, stream_mode=args.stream)

        elif args.mode == "tcp":
            run_tcp_client_stream(args.host, args.net_port, state, log_nmea_fp, log_csv_writer, stream_mode=args.stream)

        elif args.mode == "udp":
            run_udp_stream("0.0.0.0", args.net_port, state, log_nmea_fp, log_csv_writer, stream_mode=args.stream)

    finally:
        if log_nmea_fp:
            log_nmea_fp.close()
        if log_csv_fp:
            log_csv_fp.close()

        # Final accuracy report on exit
        print("\n" + "=" * 60)
        print("            NAVCAST GNSS TEST SUMMARY REPORT")
        print("=" * 60)
        acc = state.calculate_accuracy_metrics()
        print(f"Total Sentences Processed : {state.sentence_count}")
        print(f"Valid Checksum Sentences  : {state.valid_sentence_count}")
        print(f"Checksum Errors           : {state.checksum_errors}")
        if state.latitude is not None and state.longitude is not None:
            lat_dms = dec_to_dms(state.latitude, True)
            lon_dms = dec_to_dms(state.longitude, False)
            print(f"Final Latitude            : {state.latitude:.6f} deg ({lat_dms})")
            print(f"Final Longitude           : {state.longitude:.6f} deg ({lon_dms})")
            alt_str = f"{state.altitude_m:.2f} m" if state.altitude_m is not None else "N/A"
            spd_str = f"{state.speed_kmh:.2f} km/hr" if state.speed_kmh is not None else "0.00 km/hr"
            print(f"Final Altitude            : {alt_str}")
            print(f"Final Travel Speed        : {spd_str}")
            print(f"Peak Travel Speed         : {state.max_speed_kmh:.2f} km/hr")
            print(f"Average HDOP              : {state.hdop}")
            print(f"Estimated Error Margin    : +/- {acc['estimated_accuracy_m']:.2f} m")
            print(f"Stationary Spread StdDev  : Lat +/- {acc['lat_std_m']:.3f} m, Lon +/- {acc['lon_std_m']:.3f} m")
            print(f"Max Recorded Jitter Drift : {acc['max_drift_m']:.3f} m")
            print(f"Satellites Tracked        : {state.satellites_used} used, {state.satellites_in_view} in view")
        else:
            print("No valid position fix was acquired during this session.")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
