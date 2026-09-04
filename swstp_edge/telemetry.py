"""
telemetry.py — JSON telemetry packet builder & shared sensor state.

This module owns:
  1. latest_sensor — the shared mutable state dict used by all local
     processing (geofencing, motion detection, LED control, HUD overlay).
  2. hardware_state — hardware detection tracker.
  3. build_telemetry_packet() — reads all three sensors, runs road-snap,
     complementary filter, heading computation, and geofencing at 20 Hz.
     All derived state is stored in latest_sensor.  Returns a FULL packet
     used only for internal diagnostics / rawJson logging — NOT sent to
     the backend.
  4. build_slim_packet() — extracts ~20 pre-processed fields from the full
     packet and latest_sensor.  This is the ONLY shape that goes to the
     backend endpoint.  Raw sensor registers, quaternions, satellite detail
     lists, and duplicate coordinate representations are all dropped here
     on the Pi before any bytes leave the device.
  5. telemetry_loop() — 20 Hz background thread that calls build_telemetry_
     packet() (for all local processing) then enqueues a slim packet.

SLIM packet shape sent to POST /api/telemetry/ingest-batch:
{
    "seq":        int,          // sequence number
    "ts":         str,          // ISO-8601 UTC timestamp from RTC
    "epoch":      int,          // ms since Unix epoch
    "uptime":     int,          // device uptime in ms
    // --- Location (already road-snapped on Pi) ---
    "lat":        float|null,   // snapped latitude  (8 dp)
    "lon":        float|null,   // snapped longitude (8 dp)
    "alt":        float,        // altitude metres
    "spd":        float,        // speed km/h
    "hdg":        float|null,   // heading degrees (0-360)
    "sats":       int,          // satellites used in fix
    "hdop":       float|null,   // HDOP
    "fix":        bool,         // true = valid GPS fix
    "snapped":    bool,         // road-snap was applied
    "safe_zone":  bool,         // inside depot/safe-zone
    "loc_src":    str|null,     // "gnss" | "fallback" | "last_known"
    // --- IMU (complementary filter already applied) ---
    "accel_mag":  float,        // |accel| in m/s² — motion magnitude
    "roll":       float,        // degrees
    "pitch":      float,        // degrees
    "yaw":        float,        // degrees (gyro-integrated)
    "temp_c":     float,        // IMU die temperature
    "imu_ok":     bool,
    // --- Source / health ---
    "clk_src":    str           // "ntp" | "ds3231" | "system"
}
"""

import datetime
import json
import queue
import threading
import time

import sensors.rtc  as rtc_sensor
import sensors.imu  as imu_sensor
import sensors.gnss as gnss_sensor
import sensors.leds as leds

from config import TELEMETRY_RATE_HZ, TELEMETRY_INTERVAL_SEC, get_timezone_obj
from geofence import (
    snap_coordinates_to_road, compute_heading_from_gps_history,
    track_field_events, haversine_dist_meters, is_in_safe_zone,
)

# ---------------------------------------------------------------------------
# Shared state (mirrors webcam_motion_detect.py globals)
# ---------------------------------------------------------------------------
latest_sensor: dict = {
    "timestamp":               None,
    "lat":                     None,
    "lon":                     None,
    "raw_gps_lat":             None,
    "raw_gps_lon":             None,
    "alt":                     0.0,
    "speed":                   0.0,
    "heading":                 None,
    "satellites":              0,
    "gps_valid":               False,
    "location_source":         None,
    "last_gnss_fix_time":      None,
    "last_known_valid_lat":    None,
    "last_known_valid_lon":    None,
    "last_known_valid_heading": None,
    "last_known_valid_timestamp": None,
    "imu":                     None,
    "serial_connected":        False,   # always True on Pi — kept for overlay_metadata compat
    "serial_port":             "Pi-native",
    "rtc_error":               None,
    "available_ports":         [],
    "last_raw_line":           None,
    "sequence":                0,
    "active_session_id":       0,
    "deviceId":                None,
    "is_inside_safe_zone":     False,
    "is_snapped":              False,
    "road_bearing":            None,
}

hardware_state: dict = {
    "camera":  {"detected": False, "source": None, "resolution": None, "fps": None},
    "serial":  {"connected": True, "port": "Pi-native", "baud": None, "desc": "Raspberry Pi native sensors"},
    "rtc":     {"detected": False, "module": "DS3231 (I2C 0x68)", "last_ts": None, "logged_online": False, "logged_error": False},
    "gps":     {"detected": False, "module": "NavCast (TCP USB tethering)", "fix": False, "coords": None, "logged_detected": False, "logged_fix": False, "logged_fallback": False},
    "backend": {"connected": False, "last_ping": 0, "upload_count": 0, "telemetry_count": 0, "active_session_id": 0},
}

# Thread-safe queues (same names and semantics as the original script)
upload_queue    = queue.Queue(maxsize=50)
telemetry_queue = queue.Queue(maxsize=3000)

# Boot time for uptime_ms calculation
_boot_time = time.time()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Packet builder
# ---------------------------------------------------------------------------
def build_telemetry_packet(device_id: str) -> dict:
    """
    Read all sensors and build a telemetry packet whose shape is identical to
    what parse_serial_line() produced in webcam_motion_detect.py.
    """
    # --- 1. RTC ---
    rtc_data  = rtc_sensor.read()
    iso_time  = rtc_data.get("timestamp") or datetime.datetime.now(get_timezone_obj()).strftime("%Y-%m-%d %H:%M:%S")
    epoch_ms  = rtc_data.get("epoch") or int(time.time() * 1000)

    # --- 2. IMU ---
    imu_data = imu_sensor.read()

    # --- 3. GNSS ---
    gnss_data = gnss_sensor.read()
    gnss_fix  = gnss_data.get("fix", False)
    lat_raw   = gnss_data.get("latitude")
    lon_raw   = gnss_data.get("longitude")

    # --- 4. Update latest_sensor state (GNSS path) ---
    if gnss_data.get("data_received") and not hardware_state["gps"]["logged_detected"]:
        hardware_state["gps"]["detected"]        = True
        hardware_state["gps"]["logged_detected"] = True
        _log_hardware("GPS MODULE (NavCast)", "STREAM DETECTED", "NavCast NMEA TCP stream active.")

    if gnss_fix and lat_raw is not None and lon_raw is not None:
        if abs(lat_raw) > 0.001 and abs(lon_raw) > 0.001:
            snapped_lat, snapped_lon, in_sz, is_snapped, road_bearing = snap_coordinates_to_road(lat_raw, lon_raw)
            drift_m = haversine_dist_meters(lat_raw, lon_raw, snapped_lat, snapped_lon)

            if not hardware_state["gps"]["logged_fix"]:
                hardware_state["gps"]["fix"]        = True
                hardware_state["gps"]["logged_fix"] = True
                _log_hardware(
                    "GPS MODULE (NavCast)", "SATELLITE FIX & ROAD SNAP",
                    f"Raw GPS: ({lat_raw:.6f}, {lon_raw:.6f}) -> Road: ({snapped_lat:.6f}, {snapped_lon:.6f}) "
                    f"[Drift: {drift_m:.1f}m | SafeZone: {in_sz} | Snapped: {is_snapped}]"
                )

            with _lock:
                latest_sensor["raw_gps_lat"]          = lat_raw
                latest_sensor["raw_gps_lon"]          = lon_raw
                latest_sensor["lat"]                  = snapped_lat
                latest_sensor["lon"]                  = snapped_lon
                latest_sensor["is_inside_safe_zone"]  = in_sz
                latest_sensor["is_snapped"]           = is_snapped
                latest_sensor["road_bearing"]         = road_bearing
                latest_sensor["alt"]                  = gnss_data.get("altitude_m") or 0.0
                latest_sensor["speed"]                = gnss_data.get("speed_kmh")  or 0.0
                latest_sensor["satellites"]           = gnss_data.get("satellites", 0)
                latest_sensor["gps_valid"]            = True
                latest_sensor["location_source"]      = "gnss"
                latest_sensor["last_gnss_fix_time"]   = time.monotonic()
                latest_sensor["last_known_valid_lat"]       = snapped_lat
                latest_sensor["last_known_valid_lon"]       = snapped_lon
                latest_sensor["last_known_valid_timestamp"] = time.time()

            # Compute heading
            heading = compute_heading_from_gps_history(
                lat_raw, lon_raw,
                in_safe_zone=in_sz,
                is_snapped=is_snapped,
                road_bearing=road_bearing,
            )
            if heading is not None:
                with _lock:
                    latest_sensor["heading"]                      = heading
                    latest_sensor["last_known_valid_heading"]     = heading

            track_field_events(snapped_lat, snapped_lon,
                               latest_sensor["speed"], latest_sensor.get("heading") or 0.0)

    # --- 5. Update RTC state in latest_sensor ---
    with _lock:
        if rtc_data.get("valid"):
            latest_sensor["timestamp"] = iso_time
            latest_sensor["rtc_error"] = None
            if not hardware_state["rtc"]["logged_online"]:
                hardware_state["rtc"]["detected"]     = True
                hardware_state["rtc"]["logged_online"] = True
                hardware_state["rtc"]["last_ts"]      = iso_time
                _log_hardware("RTC MODULE (DS3231)", "ONLINE & SYNCHRONIZED", f"System clock: {iso_time}")

    # --- 6. Update IMU state in latest_sensor ---
    if imu_data.get("valid"):
        with _lock:
            latest_sensor["imu"] = imu_data

    # --- 7. Increment sequence & compute uptime ---
    with _lock:
        latest_sensor["sequence"] += 1
        seq = latest_sensor["sequence"]

    uptime_ms = int((time.time() - _boot_time) * 1000)

    # --- 8. Update LED state ---
    leds.update(
        rtc  = bool(rtc_data.get("valid")),
        imu  = bool(imu_data.get("valid")),
        gnss = bool(gnss_data.get("fix")),
    )

    # --- 9. Snapshot latest_sensor values for packet (avoids race) ---
    with _lock:
        s = dict(latest_sensor)

    # --- 10. Build the GNSS sub-object (same field names as parse_serial_line output) ---
    gnss_valid = s["gps_valid"] and s["lat"] is not None
    gnss_pkt = {
        "lat":             round(float(s["lat"]), 8)            if gnss_valid else None,
        "lon":             round(float(s["lon"]), 8)            if gnss_valid else None,
        "rawLat":          round(float(s["raw_gps_lat"]), 8)    if (gnss_valid and s.get("raw_gps_lat") is not None) else None,
        "rawLon":          round(float(s["raw_gps_lon"]), 8)    if (gnss_valid and s.get("raw_gps_lon") is not None) else None,
        "displayLat":      round(float(s["lat"]), 8)            if gnss_valid else None,
        "displayLon":      round(float(s["lon"]), 8)            if gnss_valid else None,
        "isInsideSafeZone": s.get("is_inside_safe_zone", False),
        "isSnapped":       s.get("is_snapped", False),
        "alt":             s["alt"],
        "speed":           s["speed"],
        "heading":         s["heading"],
        "satellites":      s["satellites"],
        "fix":             "3D_FIX" if gnss_valid else "NO_FIX",
        "valid":           gnss_valid,
        "source":          s.get("location_source"),
    }

    # --- 11. Build the IMU sub-object ---
    imu_pkt = s.get("imu") or {
        "accel": {"x": 0.0, "y": 0.0, "z": 1.0},
        "gyro":  {"x": 0.0, "y": 0.0, "z": 0.0},
        "valid": True,
    }

    # --- 12. Assemble full packet ---
    # rawJson is the full hardware telemetry line; here we serialise the
    # sensor data dict so the backend's rawJson field is populated
    raw_line = json.dumps({
        "type": "telemetry",
        "device_id": device_id,
        "seq": seq,
        "uptime_ms": uptime_ms,
        "rtc": rtc_data,
        "imu": imu_data,
        "gnss": gnss_data,
        "status": {
            "rtc":    "ACTIVE" if rtc_data.get("valid") else "FAULT",
            "imu":    "ACTIVE" if imu_data.get("valid") else "FAULT",
            "gnss":   gnss_data.get("status", "NO_DATA"),
            "serial": "CONNECTED",
            "baud":   0,
        }
    })
    with _lock:
        latest_sensor["last_raw_line"] = raw_line

    pkt = {
        "sequence":   seq,
        "timestamp":  epoch_ms,
        "uptime_ms":  epoch_ms,
        "rawJson":    raw_line,
        "rtc": {
            "iso":    iso_time,
            "epoch":  epoch_ms,
            "synced": bool(rtc_data.get("valid")),
        },
        "gnss": gnss_pkt,
        "imu":  imu_pkt,
    }
    return pkt


# ---------------------------------------------------------------------------
# Slim packet builder — the ONLY shape that leaves the device over HTTP
# ---------------------------------------------------------------------------
def build_slim_packet(full_pkt: dict) -> dict:
    """
    Distil a full internal telemetry packet into the minimal set of
    pre-processed fields needed by the backend.  All heavy lifting
    (road-snap, complementary filter, heading, geofencing) has already
    been done on the Pi inside build_telemetry_packet(); this function
    simply picks the results.

    Dropped vs full packet:
      - rawJson            (multi-KB debug blob)
      - gnss.rawLat/rawLon (pre-snap coordinates — backend sees snapped only)
      - gnss.displayLat/displayLon (duplicate of lat/lon)
      - gnss.satellites_detail (full per-satellite list)
      - imu.accel_g        (raw g-unit axes — magnitude already computed)
      - imu.accel_ms2      (individual axes — magnitude already computed)
      - imu.gyro_dps       (raw gyro axes — orientation already derived)
      - imu.quaternion     (verbose, orientation expressed as roll/pitch/yaw)
      - rtc block          (collapsed to ts + epoch + clk_src)
    """
    # --- RTC ---
    rtc_blk  = full_pkt.get("rtc") or {}
    gnss_blk = full_pkt.get("gnss") or {}
    imu_blk  = full_pkt.get("imu") or {}

    # IMU derived fields
    imu_ok   = bool(imu_blk.get("valid", False))
    orient   = imu_blk.get("orientation") or {}
    accel_m  = imu_blk.get("accel_magnitude_ms2") or 0.0
    temp_c   = imu_blk.get("temperature_c")    or 0.0
    roll     = orient.get("roll")  or 0.0
    pitch    = orient.get("pitch") or 0.0
    yaw      = orient.get("yaw")   or 0.0

    # Clock source
    try:
        import sensors.rtc as _rtc_mod
        clk_src = _rtc_mod.sync_source
    except Exception:
        clk_src = "unknown"

    # Power status
    try:
        from sensors.power import power_state as _pwr_state
        pwr_src = _pwr_state.get("power_source", "MAIN")
        low_batt = bool(_pwr_state.get("low_battery_triggered", False))
    except Exception:
        pwr_src = "MAIN"
        low_batt = False

    return {
        # Identification & timing
        "seq":       full_pkt.get("sequence"),
        "ts":        rtc_blk.get("iso"),
        "epoch":     rtc_blk.get("epoch") or full_pkt.get("timestamp"),
        "uptime":    full_pkt.get("uptime_ms"),
        # Location
        "lat":       gnss_blk.get("lat"),
        "lon":       gnss_blk.get("lon"),
        "alt":       gnss_blk.get("alt"),
        "spd":       gnss_blk.get("speed"),
        "hdg":       gnss_blk.get("heading"),
        "sats":      gnss_blk.get("satellites"),
        "hdop":      gnss_blk.get("hdop"),
        "fix":       gnss_blk.get("fix", False),
        "snapped":   gnss_blk.get("isSnapped", False),
        "safe_zone": gnss_blk.get("isInsideSafeZone", False),
        "loc_src":   gnss_blk.get("source"),
        # IMU — processed outputs only
        "accel_mag": round(float(accel_m), 3),
        "roll":      round(float(roll),    2),
        "pitch":     round(float(pitch),   2),
        "yaw":       round(float(yaw),     2),
        "temp_c":    round(float(temp_c),  2),
        "imu_ok":    imu_ok,
        # Health & Power
        "clk_src":   clk_src,
        "pwr_src":   pwr_src,
        "low_batt":  low_batt,
    }



# ---------------------------------------------------------------------------
# 20 Hz telemetry loop (background thread)
# ---------------------------------------------------------------------------
_stop_event = threading.Event()


def telemetry_loop(device_id: str) -> None:
    """
    Background thread: reads sensors at TELEMETRY_RATE_HZ.

    For each tick:
      1. build_telemetry_packet() — runs all local processing (road-snap,
         complementary filter, heading, geofencing, LED update, latest_sensor
         update).  The full packet is kept in-process only.
      2. build_slim_packet() — extracts ~20 processed fields from the full
         packet and enqueues them into telemetry_queue for the HTTP worker.
    """
    interval  = TELEMETRY_INTERVAL_SEC
    next_tick = time.monotonic()

    while not _stop_event.is_set():
        now = time.monotonic()
        if now >= next_tick:
            next_tick = now + interval
            try:
                full_pkt = build_telemetry_packet(device_id)
                slim_pkt = build_slim_packet(full_pkt)
                try:
                    telemetry_queue.put_nowait(slim_pkt)
                except queue.Full:
                    try:
                        telemetry_queue.get_nowait()
                        telemetry_queue.put_nowait(slim_pkt)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[TELEMETRY] Loop error: {exc}")

        sleep_dur = max(0.001, next_tick - time.monotonic())
        _stop_event.wait(sleep_dur)


def start(device_id: str) -> threading.Thread:
    """Start the 20 Hz telemetry loop thread and return it."""
    _stop_event.clear()
    t = threading.Thread(target=telemetry_loop, args=(device_id,),
                         name="telemetry-loop", daemon=True)
    t.start()
    return t


def stop() -> None:
    _stop_event.set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log_hardware(component: str, status: str, details: str = "") -> None:
    print("\n" + "=" * 65)
    print(f" [HARDWARE] {component.upper()} -> {status.upper()}")
    if details:
        print(f"            {details}")
    print("=" * 65 + "\n")
