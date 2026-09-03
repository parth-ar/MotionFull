"""
sensors/gnss.py — NEO-6M GNSS driver for Raspberry Pi via gpsd + gpsdclient.

DESIGN
------
The NEO-6M module is wired to the Pi's hardware UART (/dev/serial0, GPIO
14 TX / GPIO 15 RX) at 9600 baud.  The `gpsd` daemon consumes that port and
exposes a structured JSON stream on a Unix socket.  This module uses the
`gpsdclient` Python library to subscribe to that stream in a background
thread.

No hand-rolled NMEA parsing (no checksum logic, no GSV/GSA sentence parser)
is done in application code — that responsibility belongs to gpsd, per Hard
Constraint #6.

Required system-level setup (one-time, on the Pi):
    sudo apt install gpsd gpsd-clients
    sudo systemctl enable gpsd
    # Edit /etc/default/gpsd:
    #   DEVICES="/dev/serial0"
    #   GPSD_OPTIONS="-n"
    sudo systemctl restart gpsd
    cgps -s    # verify satellite data

UART configuration (/boot/config.txt or /boot/firmware/config.txt):
    enable_uart=1
    dtoverlay=disable-bt    # frees /dev/serial0 from the Bluetooth chip

VOLTAGE CAUTION ⚠️
------------------
NEO-6M TX → Pi RX (GPIO15):  NEO-6M output is 2.8 V logic — safe for Pi.
NEO-6M RX ← Pi TX (GPIO14):  Pi TX is 3.3 V.  Most NEO-6M modules accept
  3.3 V on their RX pin, but some inexpensive modules marked "5 V" on the
  module housing may have the RX pin connected to 5 V-level circuitry.
  Check your module's schematic.  If in doubt, use a 1 kΩ / 2 kΩ voltage
  divider on the Pi TX line, or a logic-level shifter.

Module output shape (mirrors the .ino 'gnss' JSON sub-object):
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

import threading
import time

from config import GNSS_DATA_TIMEOUT_SEC, GNSS_DETAIL_INTERVAL_SEC, GNSS_SATELLITE_SNAP_SEC

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
gnss_ok: bool = False           # True once data flows from gpsd

_lock = threading.Lock()
_state = {
    "data_received":     False,
    "fix":               False,
    "status":            "NO_DATA",
    "latitude":          None,
    "longitude":         None,
    "altitude_m":        None,
    "speed_kmh":         None,
    "course_deg":        None,
    "satellites":        0,
    "satellites_in_view": 0,
    "satellite_records": 0,
    "satellites_detail": None,
    "hdop":              None,
    "last_data_time":    None,
    "last_fix_time":     None,
    "last_detail_time":  None,
    "last_detail_snap_time": None,
}

_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# gpsdclient helper — converts SKY satellites to the same format the .ino
# wrote in printGnssSatelliteDetails()
# ---------------------------------------------------------------------------
_TALKER_TO_CONSTELLATION = {
    "GP": "GPS",
    "GL": "GLO",
    "GA": "GAL",
    "GB": "BDS",
    "GQ": "QZS",
    "GN": "GNSS",
}


def _satellites_from_sky(sky_satellites: list) -> list:
    """Convert gpsdclient SKY satellite list to the .ino JSON format."""
    result = []
    for sat in sky_satellites:
        # gpsdclient satellite dict keys: PRN/gnssid/svid, el, az, ss, used
        prn  = sat.get("PRN") or sat.get("svid") or 0
        elev = sat.get("el")
        az   = sat.get("az")
        snr  = sat.get("ss")
        used = sat.get("used", False)
        gnssid = sat.get("gnssid", -1)

        # Derive constellation string
        gnssid_map = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS",
                      4: "IMES", 5: "QZS", 6: "GLO"}
        constellation = gnssid_map.get(gnssid, "UNK")

        result.append({
            "prn":           prn,
            "constellation": constellation,
            "elevation_deg": int(elev) if elev is not None else 0,
            "azimuth_deg":   int(az)   if az   is not None else 0,
            "snr_db":        round(float(snr), 1) if snr is not None else None,
            "used_in_fix":   bool(used),
        })
    return result


# ---------------------------------------------------------------------------
# Background gpsd reader thread
# ---------------------------------------------------------------------------
def _gpsd_thread() -> None:
    """
    Continuously reads from gpsd via gpsdclient and updates _state.
    Handles reconnects silently — gpsd may not be running at startup.
    """
    global gnss_ok
    reconnect_delay = 5.0

    while not _stop_event.is_set():
        try:
            from gpsdclient import GPSDClient  # type: ignore
            with GPSDClient(host="127.0.0.1", port=2947) as client:
                for result in client.dict_stream(convert_datetime=False):
                    if _stop_event.is_set():
                        break

                    cls = result.get("class", "")
                    now = time.monotonic()

                    if cls == "TPV":
                        # TPV = Time-Position-Velocity report
                        mode  = result.get("mode", 0)   # 0=no data, 1=no fix, 2=2D, 3=3D
                        lat   = result.get("lat")
                        lon   = result.get("lon")
                        alt   = result.get("alt") or result.get("altHAE")
                        speed = result.get("speed")    # m/s
                        track = result.get("track")    # degrees, true north
                        # Convert speed m/s → km/h (same as .ino gps.speed.kmph())
                        speed_kmh = round(float(speed) * 3.6, 2) if speed is not None else None

                        has_fix = (mode >= 2) and (lat is not None) and (lon is not None)

                        with _lock:
                            _state["data_received"]  = True
                            _state["last_data_time"] = now
                            _state["fix"]            = has_fix
                            _state["status"]         = ("FIX" if has_fix else "NO_FIX")
                            if has_fix:
                                _state["latitude"]       = round(float(lat), 6)
                                _state["longitude"]      = round(float(lon), 6)
                                _state["altitude_m"]     = round(float(alt), 2) if alt is not None else None
                                _state["speed_kmh"]      = speed_kmh
                                _state["course_deg"]     = round(float(track), 2) if track is not None else None
                                _state["last_fix_time"]  = now
                            else:
                                # Keep last coordinates but mark no fix
                                _state["fix"]            = False
                                _state["status"]         = "NO_FIX"

                        gnss_ok = True

                    elif cls == "SKY":
                        # SKY = satellite-in-view report
                        sats      = result.get("satellites", [])
                        used_sats = [s for s in sats if s.get("used")]
                        hdop      = result.get("hdop")

                        detail = _satellites_from_sky(sats)
                        now = time.monotonic()

                        with _lock:
                            _state["satellites_in_view"]   = len(sats)
                            _state["satellite_records"]    = len(detail)
                            _state["satellites"]           = len(used_sats)
                            _state["hdop"]                 = round(float(hdop), 2) if hdop is not None else None
                            _state["satellites_detail"]    = detail
                            _state["last_detail_snap_time"] = now
                            _state["data_received"]        = True
                            _state["last_data_time"]       = now

                        gnss_ok = True

        except ImportError:
            print("[GNSS] gpsdclient not installed — run: pip3 install gpsdclient")
            _stop_event.wait(reconnect_delay)
        except Exception as exc:
            gnss_ok = False
            print(f"[GNSS] gpsd connection lost ({exc}). Retrying in {reconnect_delay:.0f}s…")
            _stop_event.wait(reconnect_delay)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init() -> bool:
    """Start the background gpsd reader thread.  Returns True immediately."""
    global _thread
    _stop_event.clear()
    _thread = threading.Thread(target=_gpsd_thread, name="gnss-reader", daemon=True)
    _thread.start()
    print("[GNSS] NEO-6M reader started (gpsdclient → gpsd on /dev/serial0)")
    return True


def stop() -> None:
    """Signal the background thread to stop."""
    _stop_event.set()


def read() -> dict:
    """
    Return a snapshot of the current GNSS state in the .ino-compatible format.
    Handles the GNSS_DATA_TIMEOUT: if no data has arrived within the timeout
    window, reports status as 'NO_DATA'.

    The 'satellites_detail' field is included for up to GNSS_SATELLITE_SNAP_SEC
    seconds after the last SKY message (same logic as the .ino 5-second window),
    then set to null — matching the .ino behavior exactly.
    """
    with _lock:
        snap = dict(_state)

    now = time.monotonic()
    last_data = snap.get("last_data_time")
    last_snap = snap.get("last_detail_snap_time")

    # Apply data-timeout (same as .ino gnssComm check)
    data_received = (
        snap["data_received"] and
        last_data is not None and
        (now - last_data) < GNSS_DATA_TIMEOUT_SEC
    )

    # Determine fix validity considering timeout
    has_fix = snap["fix"] and data_received
    if not data_received:
        status = "NO_DATA"
    elif not has_fix:
        status = "NO_FIX"
    else:
        status = "FIX"

    # Satellite detail validity window (mirrors .ino 5-second check)
    detail = None
    if last_snap is not None and (now - last_snap) < GNSS_SATELLITE_SNAP_SEC:
        detail = snap["satellites_detail"]

    return {
        "data_received":    data_received,
        "fix":              has_fix,
        "status":           status,
        "latitude":         snap["latitude"]   if has_fix else None,
        "longitude":        snap["longitude"]  if has_fix else None,
        "altitude_m":       snap["altitude_m"] if has_fix else None,
        "speed_kmh":        snap["speed_kmh"]  if has_fix else None,
        "course_deg":       snap["course_deg"] if has_fix else None,
        "satellites":       snap["satellites"],
        "satellites_in_view": snap["satellites_in_view"],
        "satellite_records": snap["satellite_records"],
        "satellites_detail": detail,
        "hdop":             snap["hdop"],
    }
