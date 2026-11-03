"""
sensors/gnss.py — u-blox MAX-M10S GNSS Hardware Driver (Primary GNSS, UART NMEA Serial).

DESIGN & CONNECTION
-------------------
Reads standard NMEA 0183 sentences from the u-blox MAX-M10S GNSS module via
hardware UART (/dev/serial0 on Raspberry Pi 4B) or USB-to-UART serial adapters.
Architecture directly mirrors test_max_m10s_gnss.py (and test_neo6m_gnss.py) for
seamless, flicker-free streaming without baud-hunting or port cycling.

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



import os
import threading
import time

try:
    import serial

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

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
RECONNECT_DELAY   = 2.0    # seconds between serial reconnect attempts
_READ_TIMEOUT_SEC = 0.05   # serial read timeout (50 ms) for 5 Hz low-latency buffer draining

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
# Per-system PRN sets keyed by GSA system-ID field (field[18]).
# MAX-M10S emits one $GNGSA per constellation; naively overwriting _used_prn_set
# with each sentence loses all but the last constellation's PRNs.
# _used_by_sys accumulates all per-system sets so _used_prn_set is their union.
_used_by_sys: dict[str, set[int]] = {}

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_init_done_event = threading.Event()
_active_port: str | None = None
_active_baud: int = GNSS_BAUD_RATE
_target_port: str | None = None
_target_baud: int | None = None


# ---------------------------------------------------------------------------
# Port Auto-Detection & Hardware Reset (Directly mirrored from test_max_m10s_gnss)
# ---------------------------------------------------------------------------

def auto_detect_gnss_port() -> str | None:
    """Scan Pi hardware UARTs for MAX-M10S. Only hardware UARTs are considered
    to avoid accidentally grabbing the EC200U LTE modem's /dev/ttyUSB* ports.
    An explicitly passed port (init(port=...)) still bypasses this function.
    """
    # Only Raspberry Pi hardware UART paths, in priority order
    for p in ("/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"):
        if os.path.exists(p):
            return p
    return None


def _nmea_checksum_ok(sentence: str) -> bool:
    """Return True if the NMEA sentence has a valid XOR checksum."""
    try:
        if "*" not in sentence:
            return False  # reject sentences without checksum or truncated
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
    has_fix = (quality > 0)

    lat = _parse_lat(fields[2], fields[3])
    lon = _parse_lon(fields[4], fields[5])
    num_sats = _safe_int(fields[7])
    hdop = _safe_float(fields[8]) if len(fields) > 8 else None
    alt = _safe_float(fields[9]) if len(fields) > 9 else None

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


def _parse_rmc(fields: list) -> None:
    """Parse $RMC / $GNRMC / $GPRMC — Speed, course over ground, coordinates, status."""
    if len(fields) < 4:
        return

    now = time.monotonic()
    status = fields[2].strip().upper() if fields[2] else "V"
    active = (status == "A")

    lat = _parse_lat(fields[3], fields[4]) if len(fields) > 4 else None
    lon = _parse_lon(fields[5], fields[6]) if len(fields) > 6 else None
    speed_kn = _safe_float(fields[7]) if len(fields) > 7 else None
    course   = _safe_float(fields[8]) if len(fields) > 8 else None

    # Convert speed from knots to km/h (1 knot = 1.852 km/h)
    speed_kmh = round(speed_kn * 1.852, 2) if speed_kn is not None else None

    with _lock:
        _state["data_received"]  = True
        _state["last_data_time"] = now
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

    # Key PRNs by the GSA system-ID (field[18]) so each constellation's $GNGSA
    # sentence accumulates into the union instead of overwriting the previous one.
    global _used_prn_set, _used_by_sys
    sys_id = fields[18].split("*")[0].strip() if len(fields) > 18 else "0"
    _used_by_sys[sys_id] = used_prns
    _used_prn_set = set().union(*_used_by_sys.values()) if _used_by_sys else set()

    now = time.monotonic()
    with _lock:
        _state["data_received"]  = True
        _state["last_data_time"] = now
        if hdop is not None:
            _state["hdop"] = round(hdop, 2)
        if used_prns and not _state["satellites"]:
            _state["satellites"] = len(_used_prn_set)


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


def _dispatch(line: str) -> bool:
    """Validate checksum and dispatch a single NMEA sentence. Returns True on valid sentence."""
    try:
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
        # GLL omitted: u-blox MAX-M10S emits $GNGLL with status V even during an active
        # 3D fix, racing against GGA/RMC and falsely clearing fix=True. Authoritative
        # position comes from GGA + RMC only (mirrors test_max_m10s_gnss.py design).
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
    except Exception:
        return False


# ---------------------------------------------------------------------------
# State Reset Helper
# ---------------------------------------------------------------------------

def _reset_state_to_disconnected() -> None:
    """Strictly reset all GNSS state when serial port closes or NMEA stream is disconnected.
    Ensures zero stale coordinates or false fixes are retained.
    """
    global gnss_ok, _active_port, _used_prn_set, _used_by_sys
    gnss_ok = False
    _active_port = None
    _used_by_sys.clear()
    _used_prn_set = set()
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


# ---------------------------------------------------------------------------
# Background Serial Reader Thread (Mirrors test_max_m10s_gnss.py connection)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# u-blox MAX-M10S UBX Protocol & Baud Rate Auto-Configuration (115200 Baud)
# ---------------------------------------------------------------------------
# UBX-CFG-VALSET command configuring UART1 to 115200 baud (CFG-UART1-BAUDRATE = 0x40520001).
# Targets RAM + BBR layers (0x03) for clean runtime speed and battery-backed persistence.
UBX_CFG_VALSET_115200 = bytes.fromhex(
    "B5 62 06 8A 0C 00 "
    "00 03 00 00 "
    "01 00 52 40 "
    "00 C2 01 00 "
    "F5 BB"
)

# UBX-MON-VER poll (Class 0x0A, ID 0x04) - harmless read query to elicit immediate response
UBX_POLL_MON_VER = bytes.fromhex("B5 62 0A 04 00 00 0E 34")


def _ubx_checksum(data: bytes) -> tuple[int, int]:
    """Calculate 8-bit Fletcher checksum over UBX data (class, id, len, payload)."""
    ck_a = 0
    ck_b = 0
    for b in data:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b


def _read_ubx_ack(ser, expected_cls: int = 0x06, expected_msg: int = 0x8A, timeout_sec: float = 0.4) -> bool | None:
    """Read incoming bytes from serial, looking for a valid UBX ACK-ACK or ACK-NAK response.
    Validates Fletcher checksum over the response header and payload.
    Returns:
        True  if ACK-ACK (0x05, 0x01) received with valid checksum for expected message.
        False if ACK-NAK (0x05, 0x00) received with valid checksum for expected message.
        None  if timed out without receiving a valid ACK/NAK frame.
    """
    deadline = time.monotonic() + timeout_sec
    buf = bytearray()
    while time.monotonic() < deadline:
        try:
            n = getattr(ser, "in_waiting", 0) or 1
            chunk = ser.read(min(n, 128))
        except Exception:
            break
        if not chunk:
            time.sleep(0.01)
            continue
        buf.extend(chunk)

        while len(buf) >= 10:
            idx = buf.find(b"\xb5\x62")
            if idx < 0:
                buf = buf[-1:]
                break
            if idx > 0:
                del buf[:idx]
            if len(buf) < 10:
                break

            msg_cls = buf[2]
            msg_id = buf[3]
            payload_len = buf[4] | (buf[5] << 8)

            if msg_cls == 0x05 and (msg_id in (0x00, 0x01)) and payload_len == 2:
                ck_a, ck_b = _ubx_checksum(buf[2:8])
                if ck_a == buf[8] and ck_b == buf[9]:
                    acked_cls = buf[6]
                    acked_msg = buf[7]
                    if acked_cls == expected_cls and acked_msg == expected_msg:
                        return (msg_id == 0x01)
                del buf[:2]
            else:
                del buf[:2]
    return None


def _check_communication(ser, timeout_sec: float = 0.9) -> bool:
    """Check whether valid GNSS data (NMEA sentence with valid checksum or UBX frame)
    is actively received from the receiver at the current baud rate.
    """
    try:
        ser.write(UBX_POLL_MON_VER)
        if hasattr(ser, "flush"):
            ser.flush()
    except Exception:
        pass

    deadline = time.monotonic() + timeout_sec
    buf = bytearray()
    while time.monotonic() < deadline:
        try:
            n = getattr(ser, "in_waiting", 0) or 1
            chunk = ser.read(min(n, 256))
        except Exception:
            return False

        if not chunk:
            time.sleep(0.01)
            continue
        buf.extend(chunk)

        # Check for valid UBX response header and frame
        if b"\xb5\x62" in buf:
            idx = buf.find(b"\xb5\x62")
            if len(buf) >= idx + 8:
                payload_len = buf[idx + 4] | (buf[idx + 5] << 8)
                total_len = idx + 6 + payload_len + 2
                if len(buf) >= total_len:
                    ck_a, ck_b = _ubx_checksum(buf[idx + 2 : idx + 6 + payload_len])
                    if ck_a == buf[total_len - 2] and ck_b == buf[total_len - 1]:
                        return True

        # Check for valid NMEA sentence with verified checksum
        while b"\n" in buf:
            nl_idx = buf.find(b"\n")
            raw_line = bytes(buf[:nl_idx + 1])
            del buf[:nl_idx + 1]
            try:
                line_str = raw_line.decode("ascii", errors="replace").strip()
            except Exception:
                continue
            if "$" in line_str:
                line_str = line_str[line_str.find("$"):]
                if _nmea_checksum_ok(line_str) and len(line_str) > 6:
                    return True

    return False


def _open_serial_port(port: str, baud: int, timeout: float = _READ_TIMEOUT_SEC):
    """Safely open serial port at specified baud rate without raising exceptions."""
    try:
        return serial.Serial(
            port=port,
            baudrate=baud,
            timeout=timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
    except Exception:
        return None


def _connect_and_configure_baud(target_port: str):
    """Automatically detect receiver baud rate, configure MAX-M10S to 115200 baud
    if operating at 9600, verify communication, and return the open serial connection.
    Preserves connection at 115200 baud for normal operation without double-opening.
    """
    print("[GNSS] Initializing receiver...")
    print("[GNSS] Checking communication at 115200 baud...")

    # Step 1: Check communication at 115200 baud
    ser = _open_serial_port(target_port, 115200, timeout=0.1)
    if ser is not None:
        if _check_communication(ser, timeout_sec=0.8):
            ser.timeout = _READ_TIMEOUT_SEC
            print("[GNSS] Receiver detected at 115200 baud.")
            print("[GNSS] Baud rate configuration already correct.")
            print("[GNSS] GNSS initialization successful.")
            return ser
        try:
            ser.close()
        except Exception:
            pass

    print("[GNSS] No response at 115200 baud.")

    # Step 2: Check communication at 9600 baud
    print("[GNSS] Checking communication at 9600 baud...")
    ser = _open_serial_port(target_port, 9600, timeout=0.1)
    if ser is not None:
        if _check_communication(ser, timeout_sec=0.9):
            print("[GNSS] Receiver detected at 9600 baud.")
            print("[GNSS] Configuring UART1 to 115200 baud...")

            # Send UBX-CFG-VALSET command for 115200 baud (RAM + BBR)
            try:
                ser.write(UBX_CFG_VALSET_115200)
                if hasattr(ser, "flush"):
                    ser.flush()
            except Exception as write_err:
                print(f"[GNSS] Failed to send configuration command: {write_err}")
                try:
                    ser.close()
                except Exception:
                    pass
                return None

            # Read UBX ACK-ACK or ACK-NAK response at 9600 baud if emitted before switch
            ack = _read_ubx_ack(ser, expected_cls=0x06, expected_msg=0x8A, timeout_sec=0.3)

            # Wait briefly for receiver to process configuration and update UART hardware
            time.sleep(0.1)
            try:
                ser.close()
            except Exception:
                pass

            # Step 3: Reopen at 115200 baud and verify communication
            print("[GNSS] Verifying new baud rate...")
            ser_new = _open_serial_port(target_port, 115200, timeout=0.1)
            if ser_new is not None:
                # If ACK was not received on 9600, check if receiver sent it on 115200
                if ack is None:
                    ack = _read_ubx_ack(ser_new, expected_cls=0x06, expected_msg=0x8A, timeout_sec=0.2)

                if _check_communication(ser_new, timeout_sec=1.2) or (ack is True):
                    ser_new.timeout = _READ_TIMEOUT_SEC
                    print("[GNSS] Receiver responding at 115200 baud.")
                    print("[GNSS] GNSS initialization successful.")
                    return ser_new
                try:
                    ser_new.close()
                except Exception:
                    pass
            print(f"[GNSS] Receiver failed verification at 115200 baud on {target_port}.")
            return None
        try:
            ser.close()
        except Exception:
            pass

    print("[GNSS] No response at 9600 baud.")
    print(f"[GNSS] GNSS receiver not detected on {target_port}.")
    return None


def configure_ubx_5hz(ser) -> bool:
    """Send UBX-CFG-RATE command to set u-blox measurement rate to 5 Hz (200 ms).
    UBX-CFG-RATE: msg class 0x06, id 0x08, length 6
    measRate = 200 ms (0x00C8) -> 5 Hz
    navRate = 1 cycle
    timeRef = 1 (GPS time)
    Checksum = 0xDE, 0x6A
    """
    try:
        cmd = bytes([0xB5, 0x62, 0x06, 0x08, 0x06, 0x00, 0xC8, 0x00, 0x01, 0x00, 0x01, 0x00, 0xDE, 0x6A])
        ser.write(cmd)
        ser.flush()
        print("[GNSS] Configured u-blox receiver measurement rate: 5 Hz (200 ms)")
        return True
    except Exception as exc:
        print(f"[GNSS] Note: 5 Hz UBX config skipped ({exc})")
        return False

def _serial_thread() -> None:
    """Background worker that opens the MAX-M10S hardware UART serial port directly,
    reads NMEA sentences continuously, and updates module state.
    
    Adopts the proven architecture of test_max_m10s_gnss.py:
      - Uses auto_detect_gnss_port() to resolve /dev/serial0 first
      - Connects directly at 115200 baud without hopping or cycling
      - Avoids invasive multi-baud probing loops that interrupt the stream
      - Stays locked onto the open stream indefinitely without closing the UART port
    
    Thread priority is raised at startup so GIL contention from OpenCV/YOLO
    inference never starves the UART buffer reader.
    """
    global gnss_ok, _active_port, _active_baud

    # Elevate OS scheduling priority — GNSS/location streaming is higher priority
    # than camera capture, motion detection, or frame streaming threads.
    _set_thread_priority_high()

    while not _stop_event.is_set():
        if not HAVE_PYSERIAL:
            print("[GNSS] CRITICAL: 'pyserial' package is not installed.")
            print("[GNSS] Please install with: pip install pyserial")
            _init_done_event.set()
            _stop_event.wait(RECONNECT_DELAY)
            continue

        # Resolve serial port: if user/configured port is missing on this machine, auto-detect
        target_port = _target_port
        if not target_port or (target_port.startswith("/dev") and not os.path.exists(target_port)):
            target_port = auto_detect_gnss_port() or getattr(_cfg, "GNSS_SERIAL_PORT", "/dev/serial0")

        ser = _connect_and_configure_baud(target_port)
        _init_done_event.set()

        if ser is None:
            _active_port = None
            if not _stop_event.is_set():
                _stop_event.wait(RECONNECT_DELAY)
            continue

        _active_port = target_port
        _active_baud = 115200
        configure_ubx_5hz(ser)

        try:
            while not _stop_event.is_set():
                try:
                    line_bytes = ser.readline()
                    if not line_bytes:
                        # Serial read timeout with no bytes received.
                        # Stays open indefinitely without tearing down port.
                        continue

                    # Drain any additional lines already waiting in the serial buffer
                    # so sentences in the 1 Hz burst are processed with zero latency
                    # (directly matches test_max_m10s_gnss.py snappy update rate).
                    lines_to_dispatch = [line_bytes]
                    try:
                        while ser.in_waiting > 0 and len(lines_to_dispatch) < 20:
                            extra = ser.readline()
                            if extra:
                                lines_to_dispatch.append(extra)
                    except Exception:
                        pass

                    for raw_line in lines_to_dispatch:
                        line = raw_line.decode("ascii", errors="replace").strip()
                        if "$" in line:
                            # Split concatenated NMEA sentences if multiple arrived in buffer
                            parts = line.split("$")
                            for p in parts:
                                p_clean = p.strip()
                                if not p_clean:
                                    continue
                                sentence = "$" + p_clean
                                if _dispatch(sentence):
                                    if not gnss_ok:
                                        gnss_ok = True
                                        print(f"[GNSS] Primary GNSS stream locked: u-blox MAX-M10S on {target_port} @ {target_baud} baud")

                except (serial.SerialException, OSError) as port_err:
                    # Never tear down the hardware UART port on transient read glitches
                    # (directly mirrors test_max_m10s_gnss.py resilient reader design).
                    # Only break if the physical device node disappeared from the OS.
                    if target_port.startswith("/dev") and not os.path.exists(target_port):
                        print(f"[GNSS] Hardware port {target_port} disconnected from OS: {port_err}")
                        break
                    time.sleep(0.01)
                except Exception:
                    # Individual line decode/format glitches are safely skipped without dropping the stream
                    time.sleep(0.005)

        except Exception as loop_err:
            print(f"[GNSS] Exception in serial reader loop: {loop_err}")
        finally:
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
            _reset_state_to_disconnected()
            if not _stop_event.is_set():
                _stop_event.wait(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _set_thread_priority_high() -> None:
    """Attempt to raise the calling thread's OS scheduling priority.
    GNSS serial reader is higher priority than motion capture / camera threads
    so GIL contention from OpenCV/YOLO never starves the UART buffer reader.
    No-ops cleanly on Windows or where privileges are insufficient.
    """
    try:
        import os as _os
        # Linux: nice value -5 (higher than default 0, lower than -20 max)
        _os.nice(-5)
    except Exception:
        pass
    try:
        # Linux pthreads SCHED_RR with moderate priority for I/O-bound serial reader
        import ctypes, ctypes.util
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if hasattr(_libc, "pthread_setschedparam"):
            # SCHED_RR = 2, priority 10 (1-99 range; 99 = highest)
            class _SchedParam(ctypes.Structure):
                _fields_ = [("sched_priority", ctypes.c_int)]
            p = _SchedParam(10)
            _libc.pthread_setschedparam(ctypes.c_ulong(threading.current_thread().ident or 0), 2, ctypes.byref(p))
    except Exception:
        pass


def init(port: str | None = None, baud: int | None = None) -> bool:
    """Start the background u-blox MAX-M10S serial reader thread.
    Args:
        port: Serial port override (e.g. /dev/serial0, /dev/ttyUSB0, COM3).
        baud: Baud rate override (default: 115200).
    Returns True immediately.
    """
    global _thread, _target_port, _target_baud
    if port and port != "Pi-native":
        _target_port = port
    if baud and baud > 0:
        _target_baud = baud

    if _thread and _thread.is_alive():
        target_p = _target_port or auto_detect_gnss_port() or GNSS_SERIAL_PORT
        target_b = _target_baud or GNSS_BAUD_RATE
        print(f"[GNSS] Primary u-blox MAX-M10S reader already running ({target_p} @ {target_b} baud)")
        return True

    _stop_event.clear()
    _init_done_event.clear()
    _thread = threading.Thread(target=_serial_thread, name="gnss-max-m10s", daemon=True)
    _thread.start()
    # Wait briefly for receiver auto-baud detection & configuration sequence to run
    # Avoids blocking main gateway startup indefinitely (times out after 3.5s if hardware missing)
    _init_done_event.wait(timeout=3.5)
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

    # Position coordinates: preserved continuously while data is streaming
    # (directly mirrors test_max_m10s_gnss.py where coordinates never flicker to None
    # on momentary fix status transitions). 'fix' indicates if lock is active.
    # Live position coordinates: strictly returned ONLY when active fix is valid.
    # Prevents broadcasting stale identical coordinates back-to-back when satellite lock drops.
    lat = snap["latitude"] if has_fix else None
    lon = snap["longitude"] if has_fix else None
    loc_source = "gnss" if has_fix else None

    return {
        "data_received":      data_received,
        "fix":                has_fix,
        "status":             status,
        "latitude":           lat,
        "longitude":          lon,
        "altitude_m":         snap["altitude_m"],
        # Speed and course are kept even without a position fix so the HUD
        # always shows live speed (including 0.0 km/h at stops) and the stop
        # detector is never starved of velocity data during brief fix blips.
        "speed_kmh":          snap["speed_kmh"]   if data_received else None,
        "course_deg":         snap["course_deg"]  if data_received else None,
        "satellites":         snap["satellites"]  if data_received else 0,
        "satellites_in_view": snap["satellites_in_view"] if data_received else 0,
        "satellite_records":  snap["satellite_records"] if data_received else 0,
        "satellites_detail":  detail if data_received else None,
        "hdop":               snap["hdop"] if has_fix else None,
        "location_source":    loc_source,
        "primary_module":     "u-blox MAX-M10S",
    }
