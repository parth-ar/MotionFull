"""
sensors/gnss.py — u-blox MAX-M10S GNSS Hardware Driver (Primary GNSS, UART NMEA Serial).

DESIGN & CONNECTION
-------------------
Reads standard NMEA 0183 sentences from the u-blox MAX-M10S GNSS module via
hardware UART (/dev/serial0 on Raspberry Pi 4B) or USB-to-UART serial adapters.

The u-blox MAX-M10S is the primary GNSS receiver for the SWSTP edge node.
Hardware Pin Connections:
  - TXD (GNSS) -> Raspberry Pi Physical 10 / BCM 15 (RXD0 / /dev/serial0)
  - RXD (GNSS) -> Raspberry Pi Physical 8  / BCM 14 (TXD0)
  - VCC        -> Raspberry Pi Physical 1  (3.3V) or Physical 2/4 (5V)
  - GND        -> Raspberry Pi Physical 6  (GND)
  - RST (Opt)  -> Raspberry Pi Physical 36 / BCM 16 (GNSS_RESET_PIN, active-low)

Receiver configuration (applied at every connect, RAM only — see configure_receiver()):
  - Measurement rate GNSS_RATE_HZ (5 Hz) via UBX-CFG-VALSET, ACK-checked, real rate measured.
  - With GNSS_NMEA_FILTER: GGA + RMC every epoch, GSV at GNSS_GSV_RATE_HZ, VTG/GLL/GSA off.

NMEA Sentences Parsed:
  - $xxGGA: position, fix quality, altitude, satellites used, HDOP   (authoritative position/fix)
  - $xxRMC: speed (knots → km/h) and course over ground
  - $xxGSV: satellites in view (count only, unless GNSS_SATELLITE_DETAIL)
  - $xxGSA: used-satellite PRNs — only when GNSS_SATELLITE_DETAIL (diagnostics)
  Everything else (VTG duplicates RMC, GLL races GGA, TXT, ...) is skipped before checksumming.

Output shape:
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
    "satellites_detail":  [...] | None,     # only with GNSS_SATELLITE_DETAIL
    "hdop":               float | None,
    "location_source":    "gnss" | None,
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
}

# Satellites in view per talker (GP/GL/GA/GB...), from GSV field 3 — the only GSV data
# production code needs. Summed into satellites_in_view.
_gsv_count_by_talker: dict[str, int] = {}
# Full per-satellite records per talker — only built when GNSS_SATELLITE_DETAIL is on
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


def _detail_enabled() -> bool:
    return bool(getattr(_cfg, "GNSS_SATELLITE_DETAIL", False))


# ---------------------------------------------------------------------------
# Port Auto-Detection & Helper Utilities
# ---------------------------------------------------------------------------

def auto_detect_gnss_port() -> str | None:
    """Scan Pi hardware UARTs for MAX-M10S. Only hardware UARTs are considered
    to avoid accidentally grabbing the EC200U LTE modem's /dev/ttyUSB* ports.
    An explicitly passed port (init(port=...)) still bypasses this function.
    """
    for p in ("/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"):
        if os.path.exists(p):
            return p
    return None


def _nmea_checksum_ok(sentence: str) -> bool:
    """Return True if the NMEA sentence has a valid XOR checksum."""
    try:
        if "*" not in sentence:
            return False
        body, cs = sentence.rsplit("*", 1)
        body = body.lstrip("$").lstrip("!")
        computed = 0
        for ch in body:
            computed ^= ord(ch)
        return computed == int(cs[:2], 16)
    except Exception:
        return False


def _parse_coord(raw: str, hemi: str, max_val: float, neg_hemi: str) -> float | None:
    """Helper to convert NMEA ddmm.mmmm/dddmm.mmmm + hemisphere to signed decimal degrees."""
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
        if hemi.strip().upper() == neg_hemi:
            val = -val
        if abs(val) > max_val or abs(val) < 0.0001:  # reject out-of-bounds or Null Island
            return None
        return round(val, 8)
    except Exception:
        return None


def _parse_lat(raw: str, hemi: str) -> float | None:
    """Convert NMEA ddmm.mmmm + N/S to signed decimal degrees."""
    return _parse_coord(raw, hemi, 90.0, "S")


def _parse_lon(raw: str, hemi: str) -> float | None:
    """Convert NMEA dddmm.mmmm + E/W to signed decimal degrees."""
    return _parse_coord(raw, hemi, 180.0, "W")


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


def _update_fix_state(now: float, has_fix: bool, lat: float | None, lon: float | None) -> None:
    """Update position fix state under _lock."""
    if has_fix and lat is not None and lon is not None:
        _state["fix"]           = True
        _state["status"]        = "FIX"
        _state["latitude"]      = lat
        _state["longitude"]     = lon
        _state["last_fix_time"] = now
    elif (now - (_state["last_fix_time"] or 0)) >= GNSS_DATA_TIMEOUT_SEC:
        _state["fix"]       = False
        _state["status"]    = "NO_FIX"


# ---------------------------------------------------------------------------
# NMEA sentence parsers
# ---------------------------------------------------------------------------

def _parse_gga(fields: list) -> None:
    """Parse $GGA / $GNGGA / $GPGGA — Time, position, fix quality, altitude."""
    if len(fields) < 10:
        return
    global _gga_count
    _gga_count += 1     # one GGA per navigation epoch → measured output rate

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
        _update_fix_state(now, has_fix, lat, lon)


def _parse_rmc(fields: list) -> None:
    """Parse $xxRMC — speed over ground and course only.

    Position and fix come from GGA alone: RMC carries the same coordinates for the same
    epoch, so parsing them twice per epoch was redundant work (and two writers of the
    same fix state). VTG is no longer parsed either — it repeats RMC's speed/course.
    """
    if len(fields) < 9:
        return
    speed_kn = _safe_float(fields[7])
    course   = _safe_float(fields[8])

    now = time.monotonic()
    with _lock:
        _state["data_received"]  = True
        _state["last_data_time"] = now
        if speed_kn is not None:
            _state["speed_kmh"] = round(speed_kn * 1.852, 2)   # knots → km/h
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
    sys_id = fields[18].strip() if len(fields) > 18 else "0"
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

    msg_num = _safe_int(fields[2])
    now = time.monotonic()

    if not _detail_enabled():
        # Production path: field 3 = total satellites in view for this talker. Every message
        # of the sequence repeats it, so only the first message is read.
        if msg_num > 1:
            return
        with _lock:
            _gsv_count_by_talker[talker] = _safe_int(fields[3])
            total = sum(_gsv_count_by_talker.values())
            _state["satellites_in_view"] = total
            _state["satellite_records"]  = total
            _state["data_received"]      = True
            _state["last_data_time"]     = now
        return

    constellation = _TALKER_CONSTELLATION.get(talker, "GNSS")
    sat_records = []
    i = 4
    while i + 3 < len(fields):
        prn  = _safe_int(fields[i])
        elev = _safe_int(fields[i + 1])
        az   = _safe_int(fields[i + 2])
        snr  = _safe_float(fields[i + 3])
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

    with _lock:
        if msg_num <= 1 or talker not in _gsv_satellites_by_talker:
            _gsv_satellites_by_talker[talker] = sat_records
        else:
            _gsv_satellites_by_talker[talker].extend(sat_records)

        # Merge all active constellations into a single unified satellite list
        merged_sats = [s for sats in _gsv_satellites_by_talker.values() for s in sats]

        _state["satellites_detail"]     = merged_sats
        _state["satellite_records"]    = len(merged_sats)
        _state["satellites_in_view"]    = len(merged_sats)
        _state["data_received"]         = True
        _state["last_data_time"]        = now
        _state["last_detail_snap_time"] = now


def _dispatch(line: str) -> bool:
    """Validate checksum and dispatch a single NMEA sentence. Returns True on valid sentence."""
    try:
        idx = line.find("$")
        if idx < 0:
            return False
        clean_line = line[idx:].strip()

        # Identify the sentence type from the header ("$GNGGA,..." → "GGA") BEFORE the
        # checksum, so unused sentences cost a slice instead of a full XOR pass.
        comma = clean_line.find(",")
        if comma < 4:
            return False
        sentence_type = clean_line[comma - 3:comma].upper()
        if sentence_type == "GSA":
            if not _detail_enabled():
                return False
        elif sentence_type not in _HANDLED_TYPES:
            # VTG (duplicate of RMC), GLL (races GGA with status V during a valid fix), TXT, ...
            return False
        elif sentence_type == "GSV" and not _detail_enabled():
            # Count-only mode needs just message 1 of each GSV sequence (it carries the total).
            # "$GPGSV,3,2,12,..." → message number is the 3rd field.
            parts = clean_line.split(",", 3)
            if len(parts) < 4 or parts[2] != "1":
                return False

        if not _nmea_checksum_ok(clean_line):
            return False

        fields = clean_line[1:].split("*", 1)[0].split(",")
        talker = fields[0][:2].upper() if len(fields[0]) >= 2 else "GN"

        if sentence_type == "GGA":
            _parse_gga(fields)
        elif sentence_type == "RMC":
            _parse_rmc(fields)
        elif sentence_type == "GSV":
            _parse_gsv(fields, talker)
        else:
            _parse_gsa(fields)
        return True
    except Exception:
        return False


_HANDLED_TYPES = frozenset({"GGA", "RMC", "GSV"})


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
    _gsv_satellites_by_talker.clear()
    _gsv_count_by_talker.clear()
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
# u-blox MAX-M10S UBX Protocol & Configuration (115200 Baud & 5 Hz)
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
        )
    except Exception:
        return None


def _connect_and_configure_baud(target_port: str, verbose: bool = True):
    """Automatically detect receiver baud rate, configure MAX-M10S to 115200 baud
    if operating at 9600, verify communication, and return the open serial connection.
    """
    if verbose:
        print("[GNSS] Initializing receiver...", flush=True)
        print("[GNSS] Checking communication at 115200 baud...", flush=True)

    # Step 1: Check communication at 115200 baud
    ser = _open_serial_port(target_port, 115200, timeout=0.1)
    if ser is not None:
        if _check_communication(ser, timeout_sec=0.8):
            ser.timeout = _READ_TIMEOUT_SEC
            print("[GNSS] Receiver detected at 115200 baud.", flush=True)
            print("[GNSS] Baud rate configuration already correct.", flush=True)
            print("[GNSS] GNSS initialization successful.", flush=True)
            return ser
        try:
            ser.close()
        except Exception:
            pass

    if verbose:
        print("[GNSS] No response at 115200 baud.", flush=True)
        print("[GNSS] Checking communication at 9600 baud...", flush=True)
    ser = _open_serial_port(target_port, 9600, timeout=0.1)
    if ser is not None:
        if _check_communication(ser, timeout_sec=0.9):
            print("[GNSS] Receiver detected at 9600 baud.", flush=True)
            print("[GNSS] Configuring UART1 to 115200 baud...", flush=True)

            # Send UBX-CFG-VALSET command for 115200 baud (RAM + BBR)
            try:
                ser.write(UBX_CFG_VALSET_115200)
                if hasattr(ser, "flush"):
                    ser.flush()
            except Exception as write_err:
                print(f"[GNSS] Failed to send configuration command: {write_err}", flush=True)
                try:
                    ser.close()
                except Exception:
                    pass
                return None

            # Read UBX ACK response if emitted before switch
            ack = _read_ubx_ack(ser, expected_cls=0x06, expected_msg=0x8A, timeout_sec=0.3)

            # Wait briefly for receiver to process configuration and update UART hardware
            time.sleep(0.1)
            try:
                ser.close()
            except Exception:
                pass

            # Step 3: Reopen at 115200 baud and verify communication
            print("[GNSS] Verifying new baud rate...", flush=True)
            ser_new = _open_serial_port(target_port, 115200, timeout=0.1)
            if ser_new is not None:
                if ack is None:
                    ack = _read_ubx_ack(ser_new, expected_cls=0x06, expected_msg=0x8A, timeout_sec=0.2)

                if _check_communication(ser_new, timeout_sec=1.2) or (ack is True):
                    ser_new.timeout = _READ_TIMEOUT_SEC
                    print("[GNSS] Receiver responding at 115200 baud.", flush=True)
                    print("[GNSS] GNSS initialization successful.", flush=True)
                    return ser_new
                try:
                    ser_new.close()
                except Exception:
                    pass
            print(f"[GNSS] Receiver failed verification at 115200 baud on {target_port}.", flush=True)
            return None
        try:
            ser.close()
        except Exception:
            pass

    if verbose:
        print("[GNSS] No response at 9600 baud.", flush=True)
        print(f"[GNSS] GNSS receiver not detected on {target_port}.", flush=True)
    return None


# ── UBX configuration (u-blox M10: CFG-VALSET; legacy UBX-CFG-RATE was confirmed ignored
#    on the deployed MAX-M10S — GGA stayed at 1.0/s after sending it) ─────────────────────
_CFG_RATE_MEAS = 0x30210001   # U2, ms between measurements
_CFG_RATE_NAV  = 0x30210002   # U2, measurements per navigation solution
_CFG_MSGOUT_UART1 = {         # U1, output every N navigation epochs (0 = off)
    "GGA": 0x209100BB,
    "RMC": 0x209100AC,
    "VTG": 0x209100B1,
    "GLL": 0x209100CA,
    "GSA": 0x209100C0,
    "GSV": 0x209100C5,
}
_LAYER_RAM = 0x01             # runtime only — re-applied at every connect, never written to flash/BBR


def _ubx_frame(cls: int, msg_id: int, payload: bytes = b"") -> bytes:
    body = bytes([cls, msg_id, len(payload) & 0xFF, (len(payload) >> 8) & 0xFF]) + payload
    ck_a, ck_b = _ubx_checksum(body)
    return b"\xb5\x62" + body + bytes([ck_a, ck_b])


def _ubx_valset(items: list[tuple[int, int, int]], layers: int = _LAYER_RAM) -> bytes:
    """UBX-CFG-VALSET. items = [(key_id, size_bytes, value), ...]."""
    payload = bytes([0x00, layers, 0x00, 0x00])
    for key, size, value in items:
        payload += key.to_bytes(4, "little") + int(value).to_bytes(size, "little")
    return _ubx_frame(0x06, 0x8A, payload)


def _read_ubx_frame(ser, cls: int, msg_id: int, timeout_sec: float = 0.6) -> bytes | None:
    """Return the payload of the first valid UBX frame with this class/id, or None."""
    deadline = time.monotonic() + timeout_sec
    buf = bytearray()
    while time.monotonic() < deadline:
        try:
            n = getattr(ser, "in_waiting", 0) or 1
            chunk = ser.read(min(n, 256))
        except Exception:
            return None
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.01)
        while True:
            idx = buf.find(b"\xb5\x62")
            if idx < 0:
                del buf[:-1]
                break
            if idx:
                del buf[:idx]
            if len(buf) < 8:
                break
            plen  = buf[4] | (buf[5] << 8)
            total = 6 + plen + 2
            if len(buf) < total:
                break
            ck_a, ck_b = _ubx_checksum(buf[2:6 + plen])
            ok = (ck_a == buf[6 + plen] and ck_b == buf[7 + plen])
            if ok and buf[2] == cls and buf[3] == msg_id:
                return bytes(buf[6:6 + plen])
            del buf[:2 if not ok else total]
    return None


def _send_and_ack(ser, frame: bytes) -> bool | None:
    """Send a UBX-CFG frame; True = ACK, False = NAK, None = no answer."""
    try:
        ser.write(frame)
        if hasattr(ser, "flush"):
            ser.flush()
    except Exception:
        return None
    return _read_ubx_ack(ser, expected_cls=frame[2], expected_msg=frame[3], timeout_sec=0.6)


def _query_firmware(ser) -> str | None:
    """UBX-MON-VER → e.g. 'SPG 5.10 / PROTVER=34.10 / MOD=MAX-M10S'."""
    try:
        ser.write(UBX_POLL_MON_VER)
        if hasattr(ser, "flush"):
            ser.flush()
    except Exception:
        return None
    payload = _read_ubx_frame(ser, 0x0A, 0x04, timeout_sec=0.6)
    if not payload or len(payload) < 40:
        return None
    text = lambda b: b.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    parts = [text(payload[i:i + 30]) for i in range(40, len(payload), 30)]
    wanted = [p for p in parts if p.startswith(("FWVER=", "PROTVER=", "MOD="))]
    return " / ".join(p.replace("FWVER=", "") for p in wanted) or text(payload[0:30])


def configure_receiver(ser) -> dict:
    """
    Apply measurement rate (config.GNSS_RATE_HZ) and, if config.GNSS_NMEA_FILTER, the NMEA
    sentence set, verifying the receiver's ACK. Fallback order:
      1. VALSET rate + sentence filter (one message)
      2. VALSET rate only            (if the receiver rejects a sentence key)
      3. legacy UBX-CFG-RATE         (older firmware)
    Returns {"rate_hz", "rate_ok", "filter_ok", "method", "firmware"}.
    """
    rate_hz = max(1.0, min(10.0, float(getattr(_cfg, "GNSS_RATE_HZ", 5.0))))
    meas_ms = int(round(1000.0 / rate_hz))
    want_filter = bool(getattr(_cfg, "GNSS_NMEA_FILTER", True))
    gsv_every = max(1, int(round(rate_hz / max(0.1, float(getattr(_cfg, "GNSS_GSV_RATE_HZ", 1.0))))))

    result = {"rate_hz": rate_hz, "rate_ok": False, "filter_ok": False,
              "method": None, "firmware": _query_firmware(ser)}
    if result["firmware"]:
        print(f"[GNSS] Receiver firmware: {result['firmware']}", flush=True)

    rate_items = [(_CFG_RATE_MEAS, 2, meas_ms), (_CFG_RATE_NAV, 2, 1)]
    filter_items = [
        (_CFG_MSGOUT_UART1["GGA"], 1, 1),
        (_CFG_MSGOUT_UART1["RMC"], 1, 1),
        (_CFG_MSGOUT_UART1["GSV"], 1, gsv_every),
        (_CFG_MSGOUT_UART1["VTG"], 1, 0),
        (_CFG_MSGOUT_UART1["GLL"], 1, 0),
        # GSA only feeds the per-satellite "used in fix" flag → on (at GSV rate) in detail mode only
        (_CFG_MSGOUT_UART1["GSA"], 1, gsv_every if _detail_enabled() else 0),
    ]

    if want_filter and _send_and_ack(ser, _ubx_valset(rate_items + filter_items)) is True:
        result.update(rate_ok=True, filter_ok=True, method="VALSET")
    elif _send_and_ack(ser, _ubx_valset(rate_items)) is True:
        result.update(rate_ok=True, method="VALSET")
    else:
        legacy = _ubx_frame(0x06, 0x08, meas_ms.to_bytes(2, "little") + (1).to_bytes(2, "little")
                            + (1).to_bytes(2, "little"))
        if _send_and_ack(ser, legacy) is True:
            result.update(rate_ok=True, method="CFG-RATE")

    if result["rate_ok"]:
        extra = (f", sentences GGA+RMC @ {rate_hz:g} Hz, GSV every {gsv_every} epochs, VTG/GLL off, "
                 f"GSA {'every ' + str(gsv_every) + ' epochs' if _detail_enabled() else 'off'}"
                 if result["filter_ok"] else
                 (" (sentence filter REJECTED — receiver output unchanged)" if want_filter else ""))
        print(f"[GNSS] Measurement rate {rate_hz:g} Hz ({meas_ms} ms) ACK via {result['method']}{extra}",
              flush=True)
    else:
        print(f"[GNSS] WARNING: receiver did not acknowledge {rate_hz:g} Hz rate "
              f"(VALSET and CFG-RATE both unanswered/rejected) — staying at its default rate.",
              flush=True)
    return result


def _set_thread_priority_high() -> None:
    """Attempt to raise the calling thread's OS scheduling priority.
    GNSS serial reader is higher priority than motion capture / camera threads
    so GIL contention from OpenCV/YOLO never starves the UART buffer reader.
    No-ops cleanly on Windows or where privileges are insufficient.
    """
    try:
        os.nice(-5)
    except Exception:
        pass
    try:
        import ctypes, ctypes.util
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if hasattr(_libc, "pthread_setschedparam"):
            class _SchedParam(ctypes.Structure):
                _fields_ = [("sched_priority", ctypes.c_int)]
            p = _SchedParam(10)
            _libc.pthread_setschedparam(ctypes.c_ulong(threading.current_thread().ident or 0), 2, ctypes.byref(p))
    except Exception:
        pass


# ── Measured output rate (verifies the configuration actually took effect) ──────────────
_RATE_CHECK_SEC = 5.0
_gga_count: int = 0
_rate_check_start: float | None = None
_rate_check_target: float = 1.0
measured_rate_hz: float | None = None    # last measured GGA rate (diagnostics)


def _start_rate_check(target_hz: float) -> None:
    global _gga_count, _rate_check_start, _rate_check_target
    _gga_count = 0
    _rate_check_target = float(target_hz)
    _rate_check_start = time.monotonic()


def _maybe_report_rate() -> None:
    """Once, ~5 s after connect: log the real GGA rate and warn if it misses the target."""
    global _rate_check_start, measured_rate_hz
    if _rate_check_start is None:
        return
    elapsed = time.monotonic() - _rate_check_start
    if elapsed < _RATE_CHECK_SEC:
        return
    _rate_check_start = None
    measured_rate_hz = round(_gga_count / elapsed, 2)
    if measured_rate_hz >= 0.8 * _rate_check_target:
        print(f"[GNSS] Measured output rate: {measured_rate_hz:.1f} Hz (target {_rate_check_target:g} Hz) ✔",
              flush=True)
    else:
        print(f"[GNSS] WARNING: measured output rate {measured_rate_hz:.1f} Hz is below target "
              f"{_rate_check_target:g} Hz — receiver configuration did not take effect.", flush=True)


# ---------------------------------------------------------------------------
# Background Serial Reader Thread
# ---------------------------------------------------------------------------

def _serial_thread() -> None:
    """Background worker that opens the MAX-M10S hardware UART serial port directly,
    reads NMEA sentences continuously, and updates module state.
    """
    global gnss_ok, _active_port, _active_baud

    _set_thread_priority_high()
    first_run = True

    while not _stop_event.is_set():
        if not HAVE_PYSERIAL:
            print("[GNSS] CRITICAL: 'pyserial' package is not installed.")
            print("[GNSS] Please install with: pip install pyserial")
            _init_done_event.set()
            _stop_event.wait(RECONNECT_DELAY)
            continue

        target_port = _target_port
        if not target_port or (target_port.startswith("/dev") and not os.path.exists(target_port)):
            target_port = auto_detect_gnss_port() or getattr(_cfg, "GNSS_SERIAL_PORT", "/dev/serial0")

        ser = _connect_and_configure_baud(target_port, verbose=first_run)
        first_run = False
        _init_done_event.set()

        if ser is None:
            _active_port = None
            if not _stop_event.is_set():
                _stop_event.wait(RECONNECT_DELAY)
            continue

        _active_port = target_port
        _active_baud = 115200
        cfg_result = configure_receiver(ser)
        _start_rate_check(cfg_result.get("rate_hz", 1.0))

        err_since: float | None = None      # start of the current run of serial errors
        reconnect_after = float(getattr(_cfg, "GNSS_SERIAL_ERROR_RECONNECT_SEC", 2.0))
        try:
            while not _stop_event.is_set():
                _maybe_report_rate()
                try:
                    line_bytes = ser.readline()
                    err_since = None
                    if not line_bytes:
                        continue

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
                            for p in line.split("$"):
                                p_clean = p.strip()
                                if not p_clean:
                                    continue
                                sentence = "$" + p_clean
                                if _dispatch(sentence):
                                    if not gnss_ok:
                                        gnss_ok = True
                                        print(f"[GNSS] Primary GNSS stream locked: u-blox MAX-M10S on {target_port} @ {_active_baud} baud")

                except (serial.SerialException, OSError) as port_err:
                    if target_port.startswith("/dev") and not os.path.exists(target_port):
                        print(f"[GNSS] Hardware port {target_port} disconnected from OS: {port_err}")
                        break
                    # Port still exists but keeps failing (e.g. another process holds it, driver
                    # wedged): previously retried every 10 ms forever without a word.
                    now_e = time.monotonic()
                    if err_since is None:
                        err_since = now_e
                    elif now_e - err_since >= reconnect_after:
                        print(f"[GNSS] Serial errors on {target_port} for {now_e - err_since:.1f}s "
                              f"({port_err}) — reopening port")
                        break
                    time.sleep(0.01)
                except Exception:
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

def init(port: str | None = None, baud: int | None = None) -> bool:
    """Start the background u-blox MAX-M10S serial reader thread.
    Args:
        port: Serial port override (e.g. /dev/serial0, /dev/ttyUSB0, COM3).
        baud: Ignored — kept for call compatibility. The baud rate is auto-detected
              (115200, else 9600 → receiver switched to 115200).
    Returns True immediately.
    """
    global _thread, _target_port
    if port and port != "Pi-native":
        _target_port = port

    if _thread and _thread.is_alive():
        target_p = _target_port or auto_detect_gnss_port() or GNSS_SERIAL_PORT
        print(f"[GNSS] Primary u-blox MAX-M10S reader already running ({target_p} @ {_active_baud} baud)")
        return True

    _stop_event.clear()
    _init_done_event.clear()
    _thread = threading.Thread(target=_serial_thread, name="gnss-max-m10s", daemon=True)
    _thread.start()
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

    # Live position coordinates: strictly returned ONLY when active fix is valid.
    # Prevents broadcasting stale identical coordinates when satellite lock drops.
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
        # Speed and course are kept even without a position fix so HUD and stop detector
        # always receive velocity data during brief satellite fix blips.
        "speed_kmh":          snap["speed_kmh"]   if data_received else None,
        "course_deg":         snap["course_deg"]  if data_received else None,
        "satellites":         snap["satellites"]  if data_received else 0,
        "satellites_in_view": snap["satellites_in_view"] if data_received else 0,
        "satellite_records":  snap["satellite_records"] if data_received else 0,
        "satellites_detail":  detail if data_received else None,
        "hdop":               snap["hdop"] if has_fix else None,
        "location_source":    loc_source,
    }
