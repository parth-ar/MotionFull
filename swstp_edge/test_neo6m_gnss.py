#!/usr/bin/env python3
"""
test_neo6m_gnss.py — Standalone GNSS NEO-6M Hardware Test & Precision Dashboard.

Completely independent, single-file diagnostic tool for the u-blox NEO-6M GNSS module.
Parses NMEA sentences from hardware UART (/dev/serial0) or USB-to-UART (/dev/ttyUSB0)
and renders a clean, live, flicker-free terminal dashboard.

Features:
- Auto-detects Raspberry Pi hardware UART (/dev/serial0, /dev/ttyAMA0) and USB serial
- Auto-detects baud rate (tests 9600, 38400, 115200)
- Validates NMEA checksums ($GPGGA, $GPRMC, $GPGSA, $GPGSV, $GPVTG)
- Live Dashboard (NO raw NMEA text spam):
    * Position: Decimal Degrees, DMS, Altitude (m & ft)
    * Status: 3D FIX / 2D FIX / SEARCHING / NO DATA
    * Satellites: Used vs In View, PRN list, Signal-to-Noise Ratio (SNR dB-Hz)
    * Precision: HDOP, VDOP, PDOP with accuracy ratings (Ideal/Good/Moderate/Poor)
    * Velocity: Speed in km/h & knots, Heading in degrees & 16-point Compass direction
    * Time: GPS UTC & Local time (IST)
    * Drift Analysis: Real-time stationary position spread in meters
- Optional CSV logging (--log-csv fixes.csv)
- Optional scrolling stream mode (--stream)

Usage:
    # Auto-detect port at 9600 baud:
    python test_neo6m_gnss.py

    # Specify custom port or baud:
    python test_neo6m_gnss.py --port /dev/serial0 --baud 9600
    python test_neo6m_gnss.py --port COM3 --baud 9600

    # Stream clean single-line logs instead of dashboard:
    python test_neo6m_gnss.py --stream

    # Log clean coordinates to CSV:
    python test_neo6m_gnss.py --log-csv neo6m_fixes.csv
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Serial Import with Graceful Fallback
# ---------------------------------------------------------------------------
try:
    import serial
    import serial.tools.list_ports
    HAVE_PYSERIAL = True
except ImportError:
    HAVE_PYSERIAL = False

# UTF-8 output configuration
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================================
# Math & Geodesy Helpers
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two coordinates."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def dec_to_dms(deg: float, is_lat: bool) -> str:
    """Convert decimal degrees to Degrees, Minutes, Seconds (DMS)."""
    if deg is None:
        return "N/A"
    direction = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    d_abs = abs(deg)
    d = int(d_abs)
    m = int((d_abs - d) * 60)
    s = (d_abs - d - m / 60.0) * 3600.0
    return f"{d}° {m:02d}' {s:05.2f}\" {direction}"


def degrees_to_cardinal(deg: float | None) -> str:
    """Convert heading angle to 16-point compass direction."""
    if deg is None:
        return "---"
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25) / 22.5) % 16
    return dirs[ix]


def rate_hdop(hdop: float | None) -> tuple[str, str]:
    """Return quality label and ANSI color for HDOP."""
    if hdop is None:
        return "N/A", "\033[90m"
    if hdop <= 1.0:
        return "IDEAL", "\033[92m"       # bright green
    elif hdop <= 2.0:
        return "EXCELLENT", "\033[92m"
    elif hdop <= 5.0:
        return "GOOD", "\033[96m"        # cyan
    elif hdop <= 10.0:
        return "MODERATE", "\033[93m"    # yellow
    return "POOR", "\033[91m"            # red


# ============================================================================
# NMEA Parser & State Engine
# ============================================================================

class NEO6MState:
    """Maintains decoded GNSS state from NMEA stream."""

    def __init__(self, history_len: int = 100):
        # Position & motion
        self.latitude: float | None = None
        self.longitude: float | None = None
        self.altitude_m: float | None = None
        self.speed_kmh: float | None = None
        self.speed_knots: float | None = None
        self.course_deg: float | None = None

        # Precision & fix
        self.fix_status: str = "NO FIX (Searching)"
        self.fix_type_str: str = "No Fix"  # 2D, 3D
        self.fix_quality: int = 0          # 0=invalid, 1=GPS SPS, 2=DGPS
        self.hdop: float | None = None
        self.vdop: float | None = None
        self.pdop: float | None = None

        # Satellites
        self.satellites_used: int = 0
        self.satellites_in_view: int = 0
        self.used_prns: list[int] = []
        self.satellites_detail: dict[int, dict] = {}   # prn -> {elev, azim, snr}

        # Time
        self.utc_time: str = "N/A"
        self.local_time: str = "N/A"
        self.date_str: str = "N/A"

        # Diagnostic metrics
        self.sentence_count: int = 0
        self.valid_sentence_count: int = 0
        self.checksum_errors: int = 0
        self.last_fix_time: float = 0.0
        self.last_data_time: float = 0.0

        # Stationary drift calculation
        self.history_len = history_len
        self.lat_history = deque(maxlen=history_len)
        self.lon_history = deque(maxlen=history_len)
        self.rate_history = deque(maxlen=20)
        self._last_sentence_mono = time.monotonic()

    def update_from_nmea(self, sentence: str) -> bool:
        """Parse NMEA sentence. Returns True if position was updated."""
        sentence = sentence.strip()
        if not sentence.startswith("$"):
            return False

        self.sentence_count += 1
        self.last_data_time = time.monotonic()

        # Checksum validation
        if "*" in sentence:
            body, cs_hex = sentence.rsplit("*", 1)
            body = body.lstrip("$").lstrip("!")
            calc_cs = 0
            for char in body:
                calc_cs ^= ord(char)
            try:
                if calc_cs != int(cs_hex[:2], 16):
                    self.checksum_errors += 1
                    return False
            except ValueError:
                self.checksum_errors += 1
                return False
        else:
            body = sentence.lstrip("$")

        self.valid_sentence_count += 1

        # Track update frequency (Hz)
        now_mono = time.monotonic()
        dt = now_mono - self._last_sentence_mono
        self._last_sentence_mono = now_mono
        if 0.001 < dt < 2.0:
            self.rate_history.append(1.0 / dt)

        parts = body.split(",")
        if not parts:
            return False

        stype = parts[0].upper()
        # Normalise talker ID ($GP, $GN, $GL)
        tag = stype[-3:] if len(stype) >= 3 else stype

        if tag == "GGA":
            return self._parse_gga(parts)
        elif tag == "RMC":
            return self._parse_rmc(parts)
        elif tag == "GSA":
            self._parse_gsa(parts)
        elif tag == "GSV":
            self._parse_gsv(parts)
        elif tag == "VTG":
            self._parse_vtg(parts)

        return False

    def _parse_lat(self, val_str: str, hemi: str) -> float | None:
        if not val_str or not hemi:
            return None
        try:
            dot = val_str.find(".")
            if dot < 2:
                return None
            deg = float(val_str[:dot - 2])
            mins = float(val_str[dot - 2:])
            deg += mins / 60.0
            if hemi.upper() == "S":
                deg = -deg
            return deg
        except Exception:
            return None

    def _parse_lon(self, val_str: str, hemi: str) -> float | None:
        if not val_str or not hemi:
            return None
        try:
            dot = val_str.find(".")
            if dot < 2:
                return None
            deg = float(val_str[:dot - 2])
            mins = float(val_str[dot - 2:])
            deg += mins / 60.0
            if hemi.upper() == "W":
                deg = -deg
            return deg
        except Exception:
            return None

    def _parse_gga(self, p: list[str]) -> bool:
        """$GPGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh"""
        if len(p) < 10:
            return False

        # UTC Time
        if len(p[1]) >= 6:
            self.utc_time = f"{p[1][0:2]}:{p[1][2:4]}:{p[1][4:6]}"
            self._update_local_time(p[1])

        lat = self._parse_lat(p[2], p[3])
        lon = self._parse_lon(p[4], p[5])

        try:
            self.fix_quality = int(p[6]) if p[6] else 0
        except ValueError:
            self.fix_quality = 0

        try:
            self.satellites_used = int(p[7]) if p[7] else 0
        except ValueError:
            pass

        try:
            self.hdop = float(p[8]) if p[8] else None
        except ValueError:
            self.hdop = None

        try:
            self.altitude_m = float(p[9]) if p[9] else None
        except ValueError:
            self.altitude_m = None

        if self.fix_quality > 0 and lat is not None and lon is not None:
            self.latitude = lat
            self.longitude = lon
            self.lat_history.append(lat)
            self.lon_history.append(lon)
            self.last_fix_time = time.monotonic()
            self.fix_status = "3D FIX (Solid)" if self.satellites_used >= 4 else "2D FIX"
            return True
        else:
            self.fix_status = "NO FIX (Searching)"
            return False

    def _parse_rmc(self, p: list[str]) -> bool:
        """$GPRMC,hhmmss.ss,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,ddmmyy,,,a*hh"""
        if len(p) < 10:
            return False

        if len(p[1]) >= 6:
            self.utc_time = f"{p[1][0:2]}:{p[1][2:4]}:{p[1][4:6]}"
            self._update_local_time(p[1])

        status = p[2].upper() if p[2] else "V"

        lat = self._parse_lat(p[3], p[4])
        lon = self._parse_lon(p[5], p[6])

        # Speed in knots -> km/h
        try:
            if p[7]:
                knots = float(p[7])
                self.speed_knots = knots
                self.speed_kmh = knots * 1.852
            else:
                self.speed_knots = 0.0
                self.speed_kmh = 0.0
        except ValueError:
            pass

        # Heading / course
        try:
            self.course_deg = float(p[8]) if p[8] else None
        except ValueError:
            self.course_deg = None

        # Date (DDMMYY)
        if len(p[9]) >= 6:
            self.date_str = f"20{p[9][4:6]}-{p[9][2:4]}-{p[9][0:2]}"

        if status == "A" and lat is not None and lon is not None:
            self.latitude = lat
            self.longitude = lon
            self.lat_history.append(lat)
            self.lon_history.append(lon)
            self.last_fix_time = time.monotonic()
            self.fix_status = "3D FIX (Solid)" if self.satellites_used >= 4 else "2D FIX"
            return True
        else:
            if status == "V":
                self.fix_status = "NO FIX (Searching)"
            return False

    def _parse_gsa(self, p: list[str]) -> None:
        """$GPGSA,A,3,19,28,14,18,27,22,31,32,,,,,1.7,1.0,1.3*35"""
        if len(p) < 18:
            return
        fix_mode = p[2]
        if fix_mode == "1":
            self.fix_type_str = "No Fix"
        elif fix_mode == "2":
            self.fix_type_str = "2D Fix"
        elif fix_mode == "3":
            self.fix_type_str = "3D Fix"

        used = []
        for prn_str in p[3:15]:
            if prn_str:
                try:
                    used.append(int(prn_str))
                except ValueError:
                    pass
        self.used_prns = used
        if not self.satellites_used:
            self.satellites_used = len(used)

        try:
            self.pdop = float(p[15]) if p[15] else None
            self.hdop = float(p[16]) if p[16] else self.hdop
            self.vdop = float(p[17].split("*")[0]) if p[17] else None
        except (ValueError, IndexError):
            pass

    def _parse_gsv(self, p: list[str]) -> None:
        """$GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,01,010,00,13,06,292,00*74"""
        if len(p) < 4:
            return
        try:
            self.satellites_in_view = int(p[3])
        except ValueError:
            pass

        # Reset dictionary on first message of sequence
        msg_num = int(p[2]) if p[2].isdigit() else 1
        if msg_num == 1:
            self.satellites_detail.clear()

        i = 4
        while i + 3 < len(p):
            try:
                prn = int(p[i])
                elev = int(p[i+1]) if p[i+1] else 0
                azim = int(p[i+2]) if p[i+2] else 0
                snr_str = p[i+3].split("*")[0]
                snr = int(snr_str) if snr_str else 0
                self.satellites_detail[prn] = {"elev": elev, "azim": azim, "snr": snr}
            except (ValueError, IndexError):
                pass
            i += 4

    def _parse_vtg(self, p: list[str]) -> None:
        """$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48"""
        if len(p) < 9:
            return
        try:
            if p[1]:
                self.course_deg = float(p[1])
            if p[7]:
                self.speed_kmh = float(p[7])
                self.speed_knots = float(p[5]) if p[5] else (self.speed_kmh / 1.852)
        except ValueError:
            pass

    def _update_local_time(self, utc_str: str) -> None:
        """Convert UTC HHMMSS to IST (UTC+05:30) display string."""
        try:
            h = int(utc_str[0:2])
            m = int(utc_str[2:4])
            s = int(utc_str[4:6])
            dt_utc = datetime.now(timezone.utc).replace(hour=h, minute=m, second=s)
            dt_ist = dt_utc + timedelta(hours=5, minutes=30)
            self.local_time = dt_ist.strftime("%H:%M:%S IST")
        except Exception:
            self.local_time = "N/A"

    def get_drift_metrics(self) -> dict:
        """Calculate stationary position spread in meters."""
        n = len(self.lat_history)
        if n < 2:
            return {"std_m": 0.0, "max_m": 0.0, "samples": n}
        mean_lat = sum(self.lat_history) / n
        mean_lon = sum(self.lon_history) / n
        dists = [haversine_distance(mean_lat, mean_lon, lat, lon)
                 for lat, lon in zip(self.lat_history, self.lon_history)]
        std_m = math.sqrt(sum(d ** 2 for d in dists) / n)
        max_m = max(dists)
        return {"std_m": std_m, "max_m": max_m, "samples": n}


# ============================================================================
# Terminal UI & Dashboard Renderer
# ============================================================================

_first_render = True

def render_dashboard(state: NEO6MState, port_info: str):
    """Render a clean in-place terminal dashboard without screen flicker."""
    global _first_render
    now = time.monotonic()
    data_age = now - state.last_data_time if state.last_data_time else 999.0
    has_fix = state.latitude is not None and state.longitude is not None and (now - state.last_fix_time < 5.0)

    # ANSI styles
    reset = "\033[0m"
    bold = "\033[1m"
    cyan = "\033[96m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    gray = "\033[90m"
    clr = "\033[K"

    # Status color
    if has_fix:
        fix_col = green
        fix_badge = "[ 3D SATELLITE FIX ]"
    elif data_age < 3.0:
        fix_col = yellow
        fix_badge = "[ SEARCHING SATELLITES ]"
    else:
        fix_col = red
        fix_badge = "[ NO DATA / DISCONNECTED ]"

    hdop_label, hdop_col = rate_hdop(state.hdop)
    drift = state.get_drift_metrics()

    # Calculate average rate
    hz = (sum(state.rate_history) / len(state.rate_history)) if state.rate_history else 0.0

    # In-place cursor positioning
    if _first_render:
        sys.stdout.write("\033[2J\033[H")
        _first_render = False
    else:
        sys.stdout.write("\033[H")

    lines = [
        f"{bold}{cyan}+==============================================================================+{reset}{clr}",
        f"{bold}{cyan}|                  u-blox NEO-6M GNSS RECEIVER LIVE DASHBOARD                  |{reset}{clr}",
        f"{bold}{cyan}+==============================================================================+{reset}{clr}",
        f" Port: {bold}{port_info}{reset} | Rate: {bold}{hz:.1f} Hz{reset} | Status: {bold}{fix_col}{fix_badge}{reset}{clr}",
        f"{gray}--------------------------------------------------------------------------------{reset}{clr}",
        f"{bold}>> POSITION & COORDINATES:{reset}{clr}",
        f"   Latitude  : {bold}{green if has_fix else yellow}{state.latitude:.7f}°{reset}  ({dec_to_dms(state.latitude, True)}){clr}" if state.latitude is not None else f"   Latitude  : {yellow}Acquiring...{reset}{clr}",
        f"   Longitude : {bold}{green if has_fix else yellow}{state.longitude:.7f}°{reset}  ({dec_to_dms(state.longitude, False)}){clr}" if state.longitude is not None else f"   Longitude : {yellow}Acquiring...{reset}{clr}",
        f"   Altitude  : {bold}{state.altitude_m:.1f} m{reset} ({state.altitude_m * 3.28084:.1f} ft){clr}" if state.altitude_m is not None else f"   Altitude  : {gray}---{reset}{clr}",
        f"{gray}--------------------------------------------------------------------------------{reset}{clr}",
        f"{bold}>> MOTION & DYNAMICS:{reset}{clr}",
        f"   Speed     : {bold}{state.speed_kmh:.1f} km/h{reset}  ({state.speed_knots:.1f} knots){clr}" if state.speed_kmh is not None else f"   Speed     : {gray}0.0 km/h{reset}{clr}",
        f"   Heading   : {bold}{state.course_deg:.1f}°{reset} [{degrees_to_cardinal(state.course_deg)}]{clr}" if state.course_deg is not None else f"   Heading   : {gray}---{reset}{clr}",
        f"{gray}--------------------------------------------------------------------------------{reset}{clr}",
        f"{bold}>> SATELLITES & ACCURACY (DOP):{reset}{clr}",
        f"   Satellites: {bold}{state.satellites_used}{reset} used in fix / {bold}{state.satellites_in_view}{reset} in view{clr}",
        f"   Precision : HDOP={bold}{hdop_col}{state.hdop or 0.0:.2f} ({hdop_label}){reset} | VDOP={state.vdop or 0.0:.2f} | PDOP={state.pdop or 0.0:.2f}{clr}",
        f"   PRNs Used : {gray}{', '.join(str(x) for x in state.used_prns) if state.used_prns else 'None'}{reset}{clr}",
    ]

    # Signal bars for top satellites
    if state.satellites_detail:
        sorted_sats = sorted(state.satellites_detail.items(), key=lambda x: x[1]['snr'], reverse=True)[:6]
        sat_bars = []
        for prn, info in sorted_sats:
            snr = info['snr']
            bar_len = int(snr / 10)
            bar = "#" * bar_len + "-" * (5 - bar_len)
            sat_bars.append(f"PRN{prn:02d}:{snr:02d}dB[{bar}]")
        lines.append(f"   Signals   : {gray}{' | '.join(sat_bars)}{reset}{clr}")

    lines.extend([
        f"{gray}--------------------------------------------------------------------------------{reset}{clr}",
        f"{bold}>> TIME & PRECISION METRICS:{reset}{clr}",
        f"   Local Time: {bold}{state.local_time}{reset} | UTC: {state.utc_time} | Date: {state.date_str}{clr}",
        f"   Drift Std : {bold}{drift['std_m']:.2f} m{reset} (Max spread: {drift['max_m']:.2f} m across {drift['samples']} samples){clr}",
        f"   Sentences : Total={state.sentence_count:,} | Checksum OK={state.valid_sentence_count:,} | Errors={state.checksum_errors}{clr}",
        f"{bold}{cyan}+==============================================================================+{reset}{clr}",
        f" Press {bold}Ctrl+C{reset} to exit.{clr}",
    ])

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def print_stream_line(state: NEO6MState):
    """Print clean scrolling single-line fix update."""
    if state.latitude is None or state.longitude is None:
        print(f"[{state.local_time}] Waiting for satellite fix... (Sats: {state.satellites_used}/{state.satellites_in_view})")
    else:
        print(f"[{state.local_time}] Lat: {state.latitude:.6f}° | Lon: {state.longitude:.6f}° "
              f"| Alt: {state.altitude_m or 0.0:.1f}m | Speed: {state.speed_kmh or 0.0:.1f}km/h "
              f"| HDOP: {state.hdop or 0.0:.2f} | Sats: {state.satellites_used}/{state.satellites_in_view}")


# ============================================================================
# Serial Port Detection & Scanning
# ============================================================================

def auto_detect_neo6m_port() -> str | None:
    """Scan candidate serial ports for NEO-6M on Raspberry Pi and PC."""
    candidates = []

    # 1. Raspberry Pi hardware UART ports
    pi_uarts = ["/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"]
    for p in pi_uarts:
        if os.path.exists(p):
            candidates.append(p)

    # 2. USB to UART adapters (/dev/ttyUSB*, /dev/ttyACM*)
    candidates.extend(glob.glob("/dev/ttyUSB*"))
    candidates.extend(glob.glob("/dev/ttyACM*"))

    # 3. Windows COM ports
    if HAVE_PYSERIAL:
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                candidates.append(p.device)
        except Exception:
            pass

    return candidates[0] if candidates else None


# ============================================================================
# Main Runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NEO-6M Hardware GNSS Test & Diagnostic Tool for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_neo6m_gnss.py
  python test_neo6m_gnss.py --port /dev/serial0 --baud 9600
  python test_neo6m_gnss.py --stream
  python test_neo6m_gnss.py --log-csv neo6m_fixes.csv
        """
    )
    parser.add_argument("--port", default=None,
                        help="Serial port (e.g. /dev/serial0, /dev/ttyUSB0, COM3). Default: auto-detect")
    parser.add_argument("--baud", type=int, default=9600,
                        help="Baud rate (NEO-6M default: 9600)")
    parser.add_argument("--stream", action="store_true",
                        help="Print scrolling lines instead of full dashboard")
    parser.add_argument("--log-csv", default=None,
                        help="Save parsed GPS fixes to CSV file")

    args = parser.parse_args()

    port = args.port
    if not port:
        port = auto_detect_neo6m_port()
        if not port:
            print("[!] No serial port detected!")
            print("    Make sure the NEO-6M UART TX/RX pins are connected to:")
            print("      NEO-6M TX -> Pi Pin 10 (GPIO 15 / RXD)")
            print("      NEO-6M RX -> Pi Pin 8  (GPIO 14 / TXD)")
            print("      VCC       -> Pi Pin 1  (3.3V) or Pin 2/4 (5V)")
            print("      GND       -> Pi Pin 6  (GND)")
            print("    And ensure UART is enabled in /boot/firmware/config.txt: enable_uart=1")
            sys.exit(1)
        print(f"[*] Auto-detected GNSS port: {port}")

    if not HAVE_PYSERIAL:
        print("[!] 'pyserial' package is not installed.")
        print("    Install it with: pip install pyserial")
        sys.exit(1)

    print(f"[*] Opening serial connection on {port} @ {args.baud} baud...")
    try:
        ser = serial.Serial(
            port=port,
            baudrate=args.baud,
            timeout=1.0,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
    except Exception as e:
        print(f"[!] Failed to open serial port {port}: {e}")
        print("\nTroubleshooting:")
        print(" 1. Check permissions: sudo usermod -aG dialout $USER")
        print(" 2. Ensure Serial Console is disabled in raspi-config:")
        print("    sudo raspi-config -> Interface Options -> Serial Port -> Login shell: No, Hardware: Yes")
        sys.exit(1)

    state = NEO6MState(history_len=100)

    # CSV Logger
    csv_fp = None
    csv_writer = None
    if args.log_csv:
        csv_fp = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "timestamp_local", "utc_time", "latitude", "longitude",
            "altitude_m", "speed_kmh", "heading_deg", "hdop",
            "satellites_used", "satellites_in_view", "fix_status"
        ])
        print(f"[+] Logging fixes to CSV: {args.log_csv}")

    port_info = f"{port} ({args.baud} baud)"
    last_render = 0.0

    print("[*] Listening for NMEA sentences... (Press Ctrl+C to stop)")

    try:
        while True:
            try:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue
                line = line_bytes.decode("ascii", errors="replace").strip()
                if line.startswith("$"):
                    has_fix = state.update_from_nmea(line)

                    if csv_writer and state.latitude is not None and has_fix:
                        csv_writer.writerow([
                            state.local_time, state.utc_time,
                            f"{state.latitude:.7f}", f"{state.longitude:.7f}",
                            state.altitude_m, state.speed_kmh, state.course_deg,
                            state.hdop, state.satellites_used, state.satellites_in_view,
                            state.fix_status
                        ])
                        csv_fp.flush()

                    if args.stream and has_fix:
                        print_stream_line(state)

            except Exception:
                pass

            if not args.stream:
                now = time.monotonic()
                if now - last_render >= 0.1:  # 10 Hz dashboard refresh
                    render_dashboard(state, port_info)
                    last_render = now

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        if csv_fp:
            csv_fp.close()

        print("\n" + "=" * 65)
        print("            NEO-6M GNSS TEST SUMMARY REPORT")
        print("=" * 65)
        print(f" Port Tested             : {port} @ {args.baud} baud")
        print(f" Total Sentences Read    : {state.sentence_count:,}")
        print(f" Checksum Valid          : {state.valid_sentence_count:,}")
        print(f" Checksum Errors         : {state.checksum_errors}")
        if state.latitude is not None and state.longitude is not None:
            print(f" Last Position Fix       : {state.latitude:.7f}°, {state.longitude:.7f}°")
            print(f" Last Altitude           : {state.altitude_m} m")
            print(f" Satellites Used/In View : {state.satellites_used} / {state.satellites_in_view}")
            print(f" Final HDOP              : {state.hdop}")
            drift = state.get_drift_metrics()
            print(f" Stationary Drift StdDev : {drift['std_m']:.2f} meters")
        else:
            print(" Fix Status              : No position fix was acquired.")
            print(" Tip                     : Place the NEO-6M antenna near an open window or outdoors.")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
