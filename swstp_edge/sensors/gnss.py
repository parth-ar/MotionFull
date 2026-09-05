"""
sensors/gnss.py — NavCast GNSS driver (TCP NMEA over USB tethering).

DESIGN
------
The NavCast Android app streams standard NMEA 0183 sentences over a TCP
connection made available via USB tethering.  This module connects to that
TCP server in a background thread, parses incoming NMEA sentences
($GGA/$GNGGA, $RMC/$GNRMC, $GSA/$GNGSA, $GSV/$GNGSV), and maintains an
up-to-date _state dict — the same shape the rest of the application expects.

Connection parameters are read from config.py:
    NAVCAST_HOST  (default "10.208.43.190")
    NAVCAST_PORT  (default 10110)

No gpsd, no gpsdclient, no pyserial required.

Auto-reconnect:
    On TCP disconnect or error the thread waits RECONNECT_DELAY seconds
    and retries indefinitely until _stop_event is set.

Module output shape (unchanged from the NEO-6M/gpsd driver):
{
    "data_received": bool,
    "fix": bool,
    "status": "NO_DATA" | "NO_FIX" | "FIX",
    "latitude": float | null,
    "longitude": float | null,
    "altitude_m": float | null,
    "speed_kmh": float | null,
    "course_deg": float | null,
    "satellites": int,
    "satellites_in_view": int,
    "satellite_records": int,
    "satellites_detail": [...] | null,
    "hdop": float | null,
}
"""

import math
import socket
import threading
import time

from config import (
    GNSS_DATA_TIMEOUT_SEC,
    GNSS_SATELLITE_SNAP_SEC,
    NAVCAST_HOST,
    NAVCAST_PORT,
)
try:
    from config import NAVCAST_AUTO_DETECT
except ImportError:
    NAVCAST_AUTO_DETECT = True

from sensors.navcast_discovery import discover_navcast

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
RECONNECT_DELAY   = 5.0    # seconds between reconnect attempts
_SOCKET_TIMEOUT   = 10.0   # recv timeout so the thread can check _stop_event
_RECV_BUF         = 4096   # bytes per recv() call

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
gnss_ok: bool = False   # True once NMEA sentences are flowing

_lock = threading.Lock()
_state = {
    "data_received":      False,
    "fix":                False,
    "status":             "NO_DATA",
    "latitude":           None,
    "longitude":          None,
    "altitude_m":         None,
    "speed_kmh":          None,
    "course_deg":         None,
    "satellites":         0,
    "satellites_in_view": 0,
    "satellite_records":  0,
    "satellites_detail":  None,
    "hdop":               None,
    "last_data_time":     None,
    "last_fix_time":      None,
    "last_detail_snap_time": None,
}

_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# NMEA helpers
# ---------------------------------------------------------------------------

def _nmea_checksum_ok(sentence: str) -> bool:
    """Return True if the NMEA sentence has a valid checksum."""
    try:
        if "*" not in sentence:
            return True   # no checksum field — accept as-is
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
        dot = raw.index(".")
        deg = float(raw[:dot - 2])
        mins = float(raw[dot - 2:])
        val = deg + mins / 60.0
        if hemi.upper() == "S":
            val = -val
        return round(val, 8)
    except Exception:
        return None


def _parse_lon(raw: str, hemi: str) -> float | None:
    """Convert NMEA dddmm.mmmm + E/W to signed decimal degrees."""
    if not raw or not hemi:
        return None
    try:
        raw = raw.strip()
        dot = raw.index(".")
        deg = float(raw[:dot - 2])
        mins = float(raw[dot - 2:])
        val = deg + mins / 60.0
        if hemi.upper() == "W":
            val = -val
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
    """
    Parse $GGA / $GNGGA — Time, position, fix quality.
    fields[0] is the talker+sentence id (e.g. GPGGA or GNGGA).
    """
    # fields: [id, time, lat, N/S, lon, E/W, quality, num_sats, hdop, alt, M, geoid, M, age, ref_id]
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
        if has_fix and lat is not None and lon is not None:
            _state["fix"]        = True
            _state["status"]     = "FIX"
            _state["latitude"]   = lat
            _state["longitude"]  = lon
            _state["altitude_m"] = round(alt, 2) if alt is not None else None
            _state["satellites"] = num_sats
            _state["hdop"]       = round(hdop, 2) if hdop is not None else None
            _state["last_fix_time"] = now
        elif not _state["fix"]:
            _state["status"] = "NO_FIX"


def _parse_rmc(fields: list) -> None:
    """
    Parse $RMC / $GNRMC — Recommended minimum (includes speed + course).
    """
    # fields: [id, time, status, lat, N/S, lon, E/W, speed_knots, course, date, ...]
    if len(fields) < 9:
        return

    active = fields[2].strip().upper() == "A"
    if not active:
        return

    lat = _parse_lat(fields[3], fields[4])
    lon = _parse_lon(fields[5], fields[6])
    speed_kn = _safe_float(fields[7])
    course   = _safe_float(fields[8])

    speed_kmh = round(speed_kn * 1.852, 2) if speed_kn is not None else None

    now = time.monotonic()
    with _lock:
        _state["data_received"] = True
        _state["last_data_time"] = now
        if lat is not None and lon is not None:
            _state["fix"]        = True
            _state["status"]     = "FIX"
            _state["latitude"]   = lat
            _state["longitude"]  = lon
            _state["speed_kmh"]  = speed_kmh
            _state["course_deg"] = round(course, 2) if course is not None else None
            _state["last_fix_time"] = now


def _parse_gsa(fields: list) -> None:
    """
    Parse $GSA / $GNGSA — DOP and active satellites.
    fields: [id, auto/manual, fix_type, sv1..sv12, pdop, hdop, vdop]
    """
    if len(fields) < 17:
        return
    hdop = _safe_float(fields[16]) if len(fields) > 16 else None
    if hdop is not None:
        with _lock:
            _state["hdop"] = round(hdop, 2)


def _parse_gsv(fields: list) -> None:
    """
    Parse $GSV / $GNGSV — Satellites in view.
    fields: [id, num_msgs, msg_num, total_sats, (sv, elev, az, snr)*n]
    """
    if len(fields) < 4:
        return

    total_sats = _safe_int(fields[3])
    msg_num    = _safe_int(fields[2])
    num_msgs   = _safe_int(fields[1])

    # Build satellite records from groups of 4
    sat_records = []
    i = 4
    while i + 3 < len(fields):
        # strip checksum from last field
        snr_raw = fields[i + 3].split("*")[0]
        prn  = _safe_int(fields[i])
        elev = _safe_int(fields[i + 1])
        az   = _safe_int(fields[i + 2])
        snr  = _safe_float(snr_raw)
        if prn > 0:
            sat_records.append({
                "prn":           prn,
                "constellation": "GNSS",
                "elevation_deg": elev,
                "azimuth_deg":   az,
                "snr_db":        round(snr, 1) if snr is not None else None,
                "used_in_fix":   False,   # will be updated by GSA if needed
            })
        i += 4

    now = time.monotonic()
    with _lock:
        _state["satellites_in_view"]    = total_sats
        _state["data_received"]         = True
        _state["last_data_time"]        = now
        _state["last_detail_snap_time"] = now
        # Accumulate satellite records across multi-message GSV sequence
        if msg_num == 1:
            _state["satellites_detail"]  = sat_records
        else:
            existing = _state["satellites_detail"] or []
            _state["satellites_detail"]  = existing + sat_records
        _state["satellite_records"] = len(_state["satellites_detail"] or [])


# ---------------------------------------------------------------------------
# Sentence dispatcher
# ---------------------------------------------------------------------------
_PARSERS = {
    "GPGGA":  _parse_gga,
    "GNGGA":  _parse_gga,
    "GLGGA":  _parse_gga,
    "GPRMC":  _parse_rmc,
    "GNRMC":  _parse_rmc,
    "GLRMC":  _parse_rmc,
    "GPGSA":  _parse_gsa,
    "GNGSA":  _parse_gsa,
    "GLGSA":  _parse_gsa,
    "GPGSV":  _parse_gsv,
    "GNGSV":  _parse_gsv,
    "GLGSV":  _parse_gsv,
}


def _dispatch(line: str) -> None:
    """Validate checksum and dispatch a single NMEA sentence."""
    line = line.strip()
    if not line.startswith("$"):
        return
    if not _nmea_checksum_ok(line):
        return
    body = line.lstrip("$").split("*")[0]
    fields = body.split(",")
    if not fields:
        return
    sentence_id = fields[0].upper()
    parser = _PARSERS.get(sentence_id)
    if parser:
        try:
            parser(fields)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background NavCast TCP reader thread
# ---------------------------------------------------------------------------
def _navcast_thread() -> None:
    """
    Connects to NavCast TCP server, reads NMEA sentences line-by-line,
    and dispatches them. Automatically discovers phone tethering IP and port,
    and auto-reconnects on any error or disconnection.
    """
    global gnss_ok

    host = NAVCAST_HOST
    port = NAVCAST_PORT

    while not _stop_event.is_set():
        # Auto-detect tethering IP & port when enabled
        if NAVCAST_AUTO_DETECT:
            try:
                disc_host, disc_port = discover_navcast(
                    preferred_host=host, preferred_port=port, timeout=0.25
                )
                if disc_host and disc_port:
                    if disc_host != host or disc_port != port:
                        print(f"[GNSS] Auto-detected NavCast @ {disc_host}:{disc_port}")
                    host, port = disc_host, disc_port
            except Exception:
                pass

        sock = None
        try:
            print(f"[GNSS] Connecting to NavCast @ {host}:{port} …")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(_SOCKET_TIMEOUT)
            sock.connect((host, port))
            print(f"[GNSS] NavCast connected: {host}:{port}")
            gnss_ok = True

            buf = ""
            while not _stop_event.is_set():
                try:
                    chunk = sock.recv(_RECV_BUF)
                except socket.timeout:
                    continue
                if not chunk:
                    print("[GNSS] NavCast connection closed by remote.")
                    break
                buf += chunk.decode("ascii", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    _dispatch(line)

        except OSError as exc:
            gnss_ok = False
            with _lock:
                _state["data_received"] = False
                _state["fix"]           = False
                _state["status"]        = "NO_DATA"
            print(f"[GNSS] NavCast connection error ({exc}). Retrying in {RECONNECT_DELAY:.0f}s…")
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

        _stop_event.wait(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Public API  (unchanged interface — drop-in replacement for gpsd driver)
# ---------------------------------------------------------------------------
def init() -> bool:
    """Start the background NavCast TCP reader thread.  Returns True immediately."""
    global _thread
    _stop_event.clear()
    _thread = threading.Thread(target=_navcast_thread, name="gnss-navcast", daemon=True)
    _thread.start()
    print(f"[GNSS] NavCast TCP reader started (target: {NAVCAST_HOST}:{NAVCAST_PORT})")
    return True


def stop() -> None:
    """Signal the background thread to stop."""
    _stop_event.set()


def read() -> dict:
    """
    Return a snapshot of the current GNSS state in the application-compatible format.

    Applies GNSS_DATA_TIMEOUT: if no NMEA data has arrived within the timeout
    window, reports status as 'NO_DATA' — mirrors original gpsd driver behaviour.

    The 'satellites_detail' field is included for up to GNSS_SATELLITE_SNAP_SEC
    seconds after the last GSV message, then set to null.
    """
    with _lock:
        snap = dict(_state)

    now       = time.monotonic()
    last_data = snap.get("last_data_time")
    last_snap = snap.get("last_detail_snap_time")

    data_received = (
        snap["data_received"] and
        last_data is not None and
        (now - last_data) < GNSS_DATA_TIMEOUT_SEC
    )

    has_fix = snap["fix"] and data_received
    if not data_received:
        status = "NO_DATA"
    elif not has_fix:
        status = "NO_FIX"
    else:
        status = "FIX"

    # Satellite detail validity window (mirrors original 5-second check)
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
        "satellites":         snap["satellites"],
        "satellites_in_view": snap["satellites_in_view"],
        "satellite_records":  snap["satellite_records"],
        "satellites_detail":  detail,
        "hdop":               snap["hdop"],
    }
