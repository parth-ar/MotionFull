"""
sensors/gnss.py — u-blox MAX-M10S GNSS Hardware Driver (Primary GNSS, UART NMEA Serial).

DESIGN
------
Reads standard NMEA 0183 sentences from the u-blox MAX-M10S GNSS module via
hardware UART (/dev/serial0 on Raspberry Pi 4B) or USB-to-UART serial adapters.

The u-blox MAX-M10S is the primary GNSS receiver for the SWSTP edge node.
Hardware Pin Connections:
  - TXD (GNSS) -> Raspberry Pi Physical 10 / BCM 15 (RXD0 / /dev/serial0)
  - RXD (GNSS) -> Raspberry Pi Physical 8  / BCM 14 (TXD0)
  - VCC        -> Raspberry Pi Physical 1  (3.3V) or Physical 2/4 (5V)
  - GND        -> Raspberry Pi Physical 6  (GND)
  - RST (Opt)  -> Raspberry Pi Physical 36 / BCM 16 (GNSS_RESET_PIN, active-low)

NMEA Sentences Parsed:
  - $GGA / $GNGGA / $GPGGA: Fix quality, latitude, longitude, altitude, satellite count, HDOP
  - $RMC / $GNRMC / $GPRMC: Recommended minimum navigation (speed km/h, course, date/time)
  - $VTG / $GNVTG / $GPVTG: Speed over ground and true course
  - $GSA / $GNGSA / $GPGSA: DOP metrics (PDOP, HDOP, VDOP) and active satellite PRNs
  - $GSV / $GNGSV / $GPGSV / $GLGSV / $GAGSV / $GBGSV: Multi-constellation satellites
    in view (GPS, GLONASS, Galileo, BeiDou, QZSS) with SNR and elevation.

Output shape (identical to the application contract):
{
    "data_received":      bool,
    "fix":                bool,
    "status":             "NO_DATA" | "NO_FIX" | "FIX",
    "latitude":           float | None,
    "longitude":          float | None,
    "altitude_m":         float | None,
    "speed_kmh":          float | None,
    "course_deg":         float | None,
    "satellites":         int,
    "satellites_in_view": int,
    "satellite_records":  int,
    "satellites_detail":  [...] | None,
    "hdop":               float | None,
    "location_source":    "gnss",
    "primary_module":     "u-blox MAX-M10S",
}
"""

import glob
import math
import os
import threading
import time

try:
    import serial
    import serial.tools.list_ports
    HAVE_PYSERIAL = True
except ImportError:
    HAVE_PYSERIAL = False

import config as _cfg
from config import (
    GNSS_DATA_TIMEOUT_SEC,
    GNSS_SATELLITE_SNAP_SEC,
    GNSS_SERIAL_PORT,
    GNSS_BAUD_RATE,
)
try:
    from config import GNSS_AUTO_DETECT
except ImportError:
    GNSS_AUTO_DETECT = True

try:
    from config import GNSS_RESET_PIN
except ImportError:
    GNSS_RESET_PIN = 16

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
RECONNECT_DELAY   = 2.0    # seconds between serial reconnect attempts
_READ_TIMEOUT_SEC = 1.0    # serial read timeout so thread can inspect _stop_event
_PROBE_IDLE_SEC   = 4.0    # seconds of silence on a port before testing candidate ports

_TALKER_CONSTELLATION = {
    "GP": "GPS",
    "GL": "GLO",
    "GA": "GAL",
    "GB": "BDS",
    "BD": "BDS",
    "GQ": "QZS",
    "GN": "GNSS",
}

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
gnss_ok: bool = False   # True once valid NMEA sentences are actively flowing

_lock = threading.Lock()
_state = {
    "data_received":         False,
    "fix":                   False,
    "status":                "NO_DATA",
    "latitude":              None,
    "longitude":             None,
    "altitude_m":            None,
    "speed_kmh":             None,
    "course_deg":            None,
    "satellites":            0,
    "satellites_in_view":    0,
    "satellite_records":     0,
    "satellites_detail":     None,
    "hdop":                  None,
    "last_data_time":        None,
    "last_fix_time":         None,
    "last_detail_snap_time": None,
    "location_source":       "gnss",
    "primary_module":        "u-blox MAX-M10S",
}

# Tracking satellites grouped by talker to avoid multi-constellation GSV collisions
_gsv_satellites_by_talker: dict[str, list[dict]] = {}
_used_prn_set: set[int] = set()

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_active_port: str | None = None
_active_baud: int = GNSS_BAUD_RATE
_target_port: str | None = None
_target_baud: int | None = None


# ---------------------------------------------------------------------------
# Port Auto-Detection & Hardware Reset
# ---------------------------------------------------------------------------

def get_candidate_ports(preferred: str | None = None) -> list[str]:
    """Return an ordered list of candidate serial ports to probe for MAX-M10S."""
    candidates = []

    def _add(p):
        if p and p not in candidates and p != "Pi-native":
            candidates.append(p)

    if preferred:
        _add(preferred)

    if _target_port:
        _add(_target_port)

    cfg_port = getattr(_cfg, "GNSS_SERIAL_PORT", "/dev/serial0")
    if cfg_port:
        _add(cfg_port)

    # 1. USB-to-UART adapters (/dev/ttyUSB*, /dev/ttyACM*)
    for p in sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")):
        if os.path.exists(p):
            _add(p)

    # 2. Raspberry Pi hardware UART ports
    for p in ("/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"):
        if os.path.exists(p):
            _add(p)

    # 3. Windows COM ports (development/bench testing)
    if HAVE_PYSERIAL:
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                _add(p.device)
        except Exception:
            pass

    return candidates


def find_gnss_port() -> str | None:
    """Locate the u-blox MAX-M10S UART or USB serial port."""
    candidates = get_candidate_ports()
    return candidates[0] if candidates else GNSS_SERIAL_PORT


def hardware_reset() -> bool:
    """Pulse the active-low RST pin on MAX-M10S (Physical 36 / BCM 16) if present.
    Safely no-ops if not running on Raspberry Pi GPIO.
    """
    if GNSS_RESET_PIN is None:
        return False
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GNSS_RESET_PIN, GPIO.OUT, initial=GPIO.HIGH)
        time.sleep(0.01)
        GPIO.output(GNSS_RESET_PIN, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(GNSS_RESET_PIN, GPIO.HIGH)
        return True
    except Exception:
        pass

    try:
        from gpiozero import OutputDevice
        dev = OutputDevice(GNSS_RESET_PIN, active_high=True, initial_value=True)
        dev.off()  # Pull LOW
        time.sleep(0.1)
        dev.on()   # Release HIGH
        dev.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NMEA helpers
# ---------------------------------------------------------------------------

def _nmea_checksum_ok(sentence: str) -> bool:
    """Return True if the NMEA sentence has a valid XOR checksum."""
    try:
        if "*" not in sentence:
            return True   # accept sentences without checksum
        body, cs = sentence.rsplit("*", 1)
        body = body.lstrip("$").lstrip("!")
        computed = 0
        for ch in body:
            computed ^= ord(ch)
        return computed == int(cs[:2], 16)
    except Exception:
        return False


def _parse_lat(raw: str, hemi: str) -> float | None:
    """Convert NMEA ddmm.mmmm + N/S to signed decimal degrees."""
    if not raw or not hemi:
        return None
    try:
        raw = raw.strip()
        dot = raw.find(".")
        if dot < 2:
            return None
        deg = float(raw[:dot - 2])
        mins = float(raw[dot - 2:])
        if mins >= 60.0:
            return None
        val = deg + mins / 60.0
        if hemi.upper() == "S":
            val = -val
        if abs(val) > 90.0:
            return None
        if abs(val) < 0.0001:  # reject 0.0 (Null Island)
            return None
        return round(val, 8)
    except Exception:
        return None


def _parse_lon(raw: str, hemi: str) -> float | None:
    """Convert NMEA dddmm.mmmm + E/W to signed decimal degrees."""
    if not raw or not hemi:
        return None
    try:
        raw = raw.strip()
        dot = raw.find(".")
        if dot < 2:
            return None
        deg = float(raw[:dot - 2])
        mins = float(raw[dot - 2:])
        if mins >= 60.0:
            return None
        val = deg + mins / 60.0
        if hemi.upper() == "W":
            val = -val
        if abs(val) > 180.0:
            return None
        if abs(val) < 0.0001:  # reject 0.0 (Null Island)
            return None
        return round(val, 8)
    except Exception:
        return None


def _safe_float(s: str) -> float | None:
    try:
        return float(s.strip()) if s and s.strip() else None
    except Exception:
        return None


def _safe_int(s: str) -> int:
    try:
        return int(s.strip()) if s and s.strip() else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# NMEA sentence parsers
# ---------------------------------------------------------------------------

def _parse_gga(fields: list) -> None:
    """Parse $GGA / $GNGGA / $GPGGA — Time, position, fix quality, altitude."""
    if len(fields) < 10:
        return

    quality = _safe_int(fields[6])   # 0=no fix, 1=SPS, 2=DGPS, 4=RTK, 5=Float RTK
    has_fix = quality > 0

    lat = _parse_lat(fields[2], fields[3])
    lon = _parse_lon(fields[4], fields[5])
    alt = _safe_float(fields[9])
    hdop = _safe_float(fields[8]) if len(fields) > 8 else None
    num_sats = _safe_int(fields[7])

    now = time.monotonic()
    with _lock:
        _state["data_received"]  = True
        _state["last_data_time"] = now
        if num_sats > 0:
            _state["satellites"] = num_sats
        if hdop is not None:
            _state["hdop"]       = round(hdop, 2)
        if alt is not None:
            _state["altitude_m"] = round(alt, 2)

        if has_fix and lat is not None and lon is not None:
            _state["fix"]           = True
            _state["status"]        = "FIX"
            _state["latitude"]      = lat
            _state["longitude"]     = lon
            _state["last_fix_time"] = now
        elif (now - (_state["last_fix_time"] or 0)) >= GNSS_DATA_TIMEOUT_SEC:
            _state["fix"]       = False
            _state["status"]    = "NO_FIX"
            _state["latitude"]  = None
            _state["longitude"] = None


def _parse_rmc(fields: list) -> None:
    """Parse $RMC / $GNRMC / $GPRMC — Speed, course over ground, coordinates, status."""
    if len(fields) < 4:
        return

    now = time.monotonic()
    with _lock:
        _state["data_received"]  = True
        _state["last_data_time"] = now

    active = fields[2].strip().upper() == "A"

    lat = _parse_lat(fields[3], fields[4]) if len(fields) > 4 else None
    lon = _parse_lon(fields[5], fields[6]) if len(fields) > 6 else None
    speed_kn = _safe_float(fields[7]) if len(fields) > 7 else None
    course   = _safe_float(fields[8]) if len(fields) > 8 else None

    # Convert speed from knots to km/h (1 knot = 1.852 km/h)
    speed_kmh = round(speed_kn * 1.852, 2) if speed_kn is not None else None

    with _lock:
        if speed_kmh is not None:
            _state["speed_kmh"] = speed_kmh
        if course is not None:
            _state["course_deg"] = round(course, 2)

        if active and lat is not None and lon is not None:
            _state["fix"]           = True
            _state["status"]        = "FIX"
            _state["latitude"]      = lat
            _state["longitude"]     = lon
            _state["last_fix_time"] = now
        elif (now - (_state["last_fix_time"] or 0)) >= GNSS_DATA_TIMEOUT_SEC:
            _state["fix"]       = False
            _state["status"]    = "NO_FIX"
            _state["latitude"]  = None
            _state["longitude"] = None


def _parse_vtg(fields: list) -> None:
    """Parse $VTG / $GNVTG / $GPVTG — Ground speed and heading.
    fields: [id, cog_t, T, cog_m, M, sog_knots, N, sog_kmh, K, mode]
    """
    if len(fields) < 8:
        return
    now = time.monotonic()
    speed_kmh = _safe_float(fields[7])
    course    = _safe_float(fields[1]) if len(fields) > 1 else None

    with _lock:
        _state["data_received"]  = True
        _state["last_data_time"] = now
        if speed_kmh is not None:
            _state["speed_kmh"]  = round(speed_kmh, 2)
        if course is not None:
            _state["course_deg"] = round(course, 2)


def _parse_gsa(fields: list) -> None:
    """Parse $GSA / $GNGSA / $GPGSA — DOP and active satellite PRNs.
    fields: [id, auto/manual, fix_type, sv1..sv12, pdop, hdop, vdop]
    """
    if len(fields) < 17:
        return
    hdop = _safe_float(fields[16]) if len(fields) > 16 else None

    used_prns: set[int] = set()
    for sv_str in fields[3:15]:
        p = _safe_int(sv_str)
        if p > 0:
            used_prns.add(p)

    global _used_prn_set
    _used_prn_set = used_prns

    now = time.monotonic()
    with _lock:
        _state["data_received"]  = True
        _state["last_data_time"] = now
        if hdop is not None:
            _state["hdop"] = round(hdop, 2)
        if used_prns and not _state["satellites"]:
            _state["satellites"] = len(used_prns)


def _parse_gsv(fields: list, talker: str) -> None:
    """Parse $GSV / $GNGSV / $GPGSV / $GLGSV / $GAGSV / $GBGSV — Satellites in view.
    fields: [id, num_msgs, msg_num, total_sats, (sv, elev, az, snr)*n]

    Handles multi-constellation transmissions from MAX-M10S by isolating sequences
    per talker ID (GP=GPS, GL=GLONASS, GA=Galileo, GB/BD=BeiDou, GQ=QZSS).
    """
    if len(fields) < 4:
        return

    msg_num    = _safe_int(fields[2])
    constellation = _TALKER_CONSTELLATION.get(talker, "GNSS")

    sat_records = []
    i = 4
    while i + 3 < len(fields):
        snr_raw = fields[i + 3].split("*")[0]
        prn  = _safe_int(fields[i])
        elev = _safe_int(fields[i + 1])
        az   = _safe_int(fields[i + 2])
        snr  = _safe_float(snr_raw)
        if prn > 0:
            sat_records.append({
                "prn":           prn,
                "constellation": constellation,
                "elevation_deg": elev,
                "azimuth_deg":   az,
                "snr_db":        round(snr, 1) if snr is not None else None,
                "used_in_fix":   prn in _used_prn_set,
            })
        i += 4

    now = time.monotonic()
    with _lock:
        if msg_num <= 1 or talker not in _gsv_satellites_by_talker:
            _gsv_satellites_by_talker[talker] = sat_records
        else:
            _gsv_satellites_by_talker[talker].extend(sat_records)

        # Merge all active constellations into a single unified satellite list
        merged_sats: list[dict] = []
        for sats in _gsv_satellites_by_talker.values():
            merged_sats.extend(sats)

        _state["satellites_detail"]     = merged_sats
        _state["satellite_records"]    = len(merged_sats)
        _state["satellites_in_view"]    = len(merged_sats)
        _state["data_received"]         = True
        _state["last_data_time"]        = now
        _state["last_detail_snap_time"] = now


# ---------------------------------------------------------------------------
# Sentence dispatcher
# ---------------------------------------------------------------------------

def _dispatch(line: str) -> bool:
    """Validate checksum and dispatch a single NMEA sentence. Returns True on valid sentence."""
    line = line.strip()
    idx = line.find("$")
    if idx < 0:
        return False
    line = line[idx:]

    if not _nmea_checksum_ok(line):
        return False

    body = line.lstrip("$").split("*")[0]
    fields = body.split(",")
    if not fields or not fields[0]:
        return False

    sentence_id = fields[0].upper()
    talker = sentence_id[:2] if len(sentence_id) >= 2 else "GN"

    if sentence_id.endswith("GGA"):
        _parse_gga(fields)
        return True
    elif sentence_id.endswith("RMC"):
        _parse_rmc(fields)
        return True
    elif sentence_id.endswith("VTG"):
        _parse_vtg(fields)
        return True
    elif sentence_id.endswith("GSA"):
        _parse_gsa(fields)
        return True
    elif sentence_id.endswith("GSV"):
        _parse_gsv(fields, talker)
        return True

    return False


# ---------------------------------------------------------------------------
# Background Serial Reader Thread with Auto-Port & Auto-Baud Scanning
# ---------------------------------------------------------------------------

def _serial_thread() -> None:
    """Background worker that connects to the MAX-M10S hardware UART serial port,
    reads NMEA sentences line-by-line, and updates module state.
    Auto-detects ports and bauds if data is not actively flowing.
    """
    global gnss_ok, _active_port, _active_baud

    # Candidate bauds to test (MAX-M10S default is 9600; high-speed is 38400 or 115200)
    base_baud = _target_baud or GNSS_BAUD_RATE
    candidate_bauds = [base_baud]
    for b in (9600, 38400, 115200):
        if b not in candidate_bauds:
            candidate_bauds.append(b)

    while not _stop_event.is_set():
        if not HAVE_PYSERIAL:
            print("[GNSS] CRITICAL: 'pyserial' package is not installed.")
            print("[GNSS] Please install with: pip install pyserial")
            _stop_event.wait(RECONNECT_DELAY)
            continue

        candidate_ports = get_candidate_ports(_target_port)
        if not candidate_ports:
            candidate_ports = [GNSS_SERIAL_PORT]

        connected_and_streamed = False

        for port in candidate_ports:
            if _stop_event.is_set():
                break

            for baud in candidate_bauds:
                if _stop_event.is_set():
                    break

                ser = None
                try:
                    ser = serial.Serial(
                        port=port,
                        baudrate=baud,
                        timeout=_READ_TIMEOUT_SEC,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                    )

                    # Probe port for valid NMEA sentences
                    probe_start = time.monotonic()
                    port_has_nmea = False

                    while (time.monotonic() - probe_start) < _PROBE_IDLE_SEC and not _stop_event.is_set():
                        try:
                            line_bytes = ser.readline()
                        except Exception:
                            break

                        if not line_bytes:
                            continue

                        line = line_bytes.decode("ascii", errors="replace").strip()
                        if "$" in line:
                            if _dispatch(line):
                                port_has_nmea = True
                                break

                    if not port_has_nmea:
                        # Try next baud or port
                        ser.close()
                        ser = None
                        continue

                    # Confirmed NMEA stream! Lock onto this port and baud
                    _active_port = port
                    _active_baud = baud
                    gnss_ok = True
                    print(f"[GNSS] Primary GNSS active: u-blox MAX-M10S on {port} @ {baud} baud")

                    last_valid_sentence_mono = time.monotonic()
                    while not _stop_event.is_set():
                        try:
                            line_bytes = ser.readline()
                        except Exception as read_err:
                            print(f"[GNSS] Read error on {port}: {read_err}")
                            break

                        if not line_bytes:
                            # 1.0s read timeout elapsed with no newline.
                            # Standard 1 Hz GPS bursts can have inter-epoch timing jitter.
                            # Only treat as disconnected if completely silent for >= 12 seconds.
                            if (time.monotonic() - last_valid_sentence_mono) >= 12.0:
                                print(f"[GNSS] No NMEA data on {port} for 12s — stream lost, re-establishing...")
                                break
                            continue

                        line = line_bytes.decode("ascii", errors="replace").strip()
                        if "$" in line:
                            if _dispatch(line):
                                last_valid_sentence_mono = time.monotonic()

                    connected_and_streamed = True

                except Exception as exc:
                    pass
                finally:
                    if ser:
                        try:
                            ser.close()
                        except Exception:
                            pass

                if connected_and_streamed:
                    break

            if connected_and_streamed:
                break

        gnss_ok = False
        _active_port = None
        with _lock:
            _state["data_received"]         = False
            _state["fix"]                   = False
            _state["status"]                = "NO_DATA"
            _state["latitude"]              = None
            _state["longitude"]             = None
            _state["altitude_m"]            = None
            _state["speed_kmh"]             = None
            _state["course_deg"]            = None
            _state["satellites"]            = 0
            _state["satellites_in_view"]    = 0
            _state["satellite_records"]     = 0
            _state["satellites_detail"]     = None
            _state["hdop"]                  = None
            _state["last_data_time"]        = None
            _state["last_fix_time"]         = None
            _state["last_detail_snap_time"] = None

        _stop_event.wait(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init(port: str | None = None, baud: int | None = None) -> bool:
    """Start the background u-blox MAX-M10S serial reader thread.
    Args:
        port: Serial port override (e.g. /dev/serial0, /dev/ttyUSB0, COM3).
        baud: Baud rate override (default: 9600).
    Returns True immediately.
    """
    global _thread, _target_port, _target_baud
    if port and port != "Pi-native":
        _target_port = port
    if baud and baud > 0:
        _target_baud = baud

    if _thread and _thread.is_alive():
        target_p = _target_port or find_gnss_port() or GNSS_SERIAL_PORT
        target_b = _target_baud or GNSS_BAUD_RATE
        print(f"[GNSS] Primary u-blox MAX-M10S reader already running ({target_p} @ {target_b} baud)")
        return True

    _stop_event.clear()
    _thread = threading.Thread(target=_serial_thread, name="gnss-max-m10s", daemon=True)
    _thread.start()
    target_p = _target_port or find_gnss_port() or GNSS_SERIAL_PORT
    target_b = _target_baud or GNSS_BAUD_RATE
    print(f"[GNSS] Primary u-blox MAX-M10S reader initiated (probed: {target_p} @ {target_b} baud)")
    return True


def stop() -> None:
    """Signal the background thread to stop and clean up."""
    _stop_event.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=2.0)


def get_active_port() -> str | None:
    """Return the currently open serial port device name, or None."""
    return _active_port


def get_active_baud() -> int:
    """Return the currently active baud rate."""
    return _active_baud


def read() -> dict:
    """Return a snapshot of the current GNSS state.

    Applies GNSS_DATA_TIMEOUT: if no NMEA data has arrived within the timeout
    window, reports status as 'NO_DATA'. Fix status is stabilized across GGA and RMC
    bursts to eliminate false NO_FIX flickers.

    The 'satellites_detail' field is included for up to GNSS_SATELLITE_SNAP_SEC
    seconds after the last GSV message, then set to None.
    """
    with _lock:
        snap = dict(_state)

    now       = time.monotonic()
    last_data = snap.get("last_data_time")
    last_fix  = snap.get("last_fix_time")
    last_snap = snap.get("last_detail_snap_time")

    data_received = (
        snap["data_received"] and
        last_data is not None and
        (now - last_data) < GNSS_DATA_TIMEOUT_SEC
    )

    has_fix = (
        data_received and
        snap["fix"] and
        last_fix is not None and
        (now - last_fix) < GNSS_DATA_TIMEOUT_SEC and
        snap["latitude"] is not None and
        snap["longitude"] is not None
    )

    if not data_received:
        status = "NO_DATA"
    elif not has_fix:
        status = "NO_FIX"
    else:
        status = "FIX"

    detail = None
    if last_snap is not None and (now - last_snap) < GNSS_SATELLITE_SNAP_SEC:
        detail = snap["satellites_detail"]

    return {
        "data_received":      data_received,
        "fix":                has_fix,
        "status":             status,
        "latitude":           snap["latitude"]    if has_fix else None,
        "longitude":          snap["longitude"]   if has_fix else None,
        "altitude_m":         snap["altitude_m"]  if has_fix else None,
        "speed_kmh":          snap["speed_kmh"]   if has_fix else None,
        "course_deg":         snap["course_deg"]  if has_fix else None,
        "satellites":         snap["satellites"]  if data_received else 0,
        "satellites_in_view": snap["satellites_in_view"] if data_received else 0,
        "satellite_records":  snap["satellite_records"] if data_received else 0,
        "satellites_detail":  detail if data_received else None,
        "hdop":               snap["hdop"] if has_fix else None,
        "location_source":    "gnss" if has_fix else None,
        "primary_module":     "u-blox MAX-M10S",
    }


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS coordinates in meters.
    Uses the Haversine formula with mean Earth radius R = 6,371,000 m.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    R = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
