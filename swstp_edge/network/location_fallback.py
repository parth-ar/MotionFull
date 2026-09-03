"""
network/location_fallback.py — IP-geolocation fallback for GNSS outages.

Port of gps_fallback_worker() and get_laptop_location() from
webcam_motion_detect.py, adapted for Linux/Raspberry Pi:

  - The Windows-only branch (get_windows_gps_location / win_location.ps1)
    has been removed entirely — this is a Linux deployment.
  - IP-geolocation falls back to the same two providers in the same order:
      1. ipwho.is
      2. ip-api.com
  - All geofencing / road-snap calls are unchanged.
"""

import time
import threading

import requests

from config import (
    ENABLE_GPS_FALLBACK,
    GNSS_FALLBACK_TIMEOUT_SEC,
    GNSS_FALLBACK_REFRESH_SEC,
)
from geofence import (
    snap_coordinates_to_road, compute_heading_from_gps_history,
    track_field_events, haversine_dist_meters, is_in_safe_zone,
)


# ---------------------------------------------------------------------------
# IP-geolocation  (Linux-only; Windows branch removed)
# ---------------------------------------------------------------------------
def get_laptop_location():
    """Fetches approximate location for this machine via public IP-geolocation APIs.

    Returns: (lat, lon, city_or_label, ip_str)  — any element may be None on failure.

    Two providers are tried in priority order (same as the original script):
      1. ipwho.is
      2. ip-api.com
    """
    sources = [
        (
            "ipwho.is",
            "http://ipwho.is/",
            lambda d: (
                (float(d["latitude"]), float(d["longitude"]), d.get("city"), d.get("ip"))
                if d.get("success") else None
            ),
        ),
        (
            "ip-api",
            "http://ip-api.com/json/?fields=status,message,country,regionName,city,lat,lon,timezone,query",
            lambda d: (
                (float(d["lat"]), float(d["lon"]), d.get("city"), d.get("query"))
                if d.get("status") == "success" else None
            ),
        ),
    ]

    for name, url, parser in sources:
        try:
            resp = requests.get(url, timeout=4.0)
            if resp.status_code == 200:
                res = parser(resp.json())
                if res and res[0] is not None and res[1] is not None:
                    lat, lon, city, ip = res
                    return round(float(lat), 8), round(float(lon), 8), city, ip
        except Exception:
            continue

    return None, None, None, None


# ---------------------------------------------------------------------------
# GPS fallback worker thread
# ---------------------------------------------------------------------------
def gps_fallback_worker(stop_event: threading.Event,
                         enable: bool = ENABLE_GPS_FALLBACK,
                         timeout_sec: float = GNSS_FALLBACK_TIMEOUT_SEC,
                         refresh_sec: float = GNSS_FALLBACK_REFRESH_SEC) -> None:
    """Background worker: engages IP-based geolocation whenever the hardware
    GNSS module hasn't produced a valid fix for `timeout_sec` seconds.
    Hands control straight back to GNSS the moment a fresh hardware fix arrives.

    Mirrors gps_fallback_worker() from webcam_motion_detect.py exactly, with
    the Windows PowerShell branch removed.
    """
    from telemetry import latest_sensor, hardware_state

    last_fallback_fetch = 0.0

    while not stop_event.is_set():
        if enable:
            last_fix  = latest_sensor.get("last_gnss_fix_time")
            gnss_stale = (last_fix is None) or (time.monotonic() - last_fix > timeout_sec)

            if gnss_stale:
                now = time.monotonic()
                need_refresh = (
                    latest_sensor.get("location_source") != "fallback"
                    or (now - last_fallback_fetch) >= refresh_sec
                )
                if need_refresh:
                    lat, lon, city, ip = get_laptop_location()
                    last_fallback_fetch = now

                    # Re-check staleness — GNSS may have recovered while we fetched
                    last_fix_now  = latest_sensor.get("last_gnss_fix_time")
                    still_stale   = (last_fix_now is None) or (time.monotonic() - last_fix_now > timeout_sec)

                    if lat is not None and lon is not None and still_stale:
                        raw_lat, raw_lon = lat, lon
                        latest_sensor["raw_gps_lat"] = raw_lat
                        latest_sensor["raw_gps_lon"] = raw_lon

                        snapped_lat, snapped_lon, in_sz, is_snapped, road_bearing = \
                            snap_coordinates_to_road(raw_lat, raw_lon)
                        lat, lon    = snapped_lat, snapped_lon
                        dist_drift  = haversine_dist_meters(raw_lat, raw_lon, snapped_lat, snapped_lon)

                        latest_sensor["is_inside_safe_zone"] = in_sz
                        latest_sensor["is_snapped"]          = is_snapped
                        latest_sensor["road_bearing"]        = road_bearing
                        latest_sensor["lat"]                 = lat
                        latest_sensor["lon"]                 = lon
                        latest_sensor["heading"]             = compute_heading_from_gps_history(lat, lon)
                        latest_sensor["last_known_valid_lat"]       = lat
                        latest_sensor["last_known_valid_lon"]       = lon
                        latest_sensor["last_known_valid_heading"]   = latest_sensor["heading"]
                        latest_sensor["last_known_valid_timestamp"] = time.time()
                        latest_sensor["gps_valid"]           = True
                        latest_sensor["location_source"]     = "fallback"
                        track_field_events(lat, lon,
                                           latest_sensor.get("speed", 0.0),
                                           latest_sensor["heading"])

                        if not hardware_state["gps"]["logged_fallback"]:
                            hardware_state["gps"]["logged_fallback"] = True
                            _log_fallback(raw_lat, raw_lon, snapped_lat, snapped_lon,
                                          dist_drift, in_sz, city, ip)

                    elif still_stale and latest_sensor.get("last_known_valid_lat") is not None:
                        # Signal lost — hold last known position
                        latest_sensor["lat"]     = latest_sensor["last_known_valid_lat"]
                        latest_sensor["lon"]     = latest_sensor["last_known_valid_lon"]
                        latest_sensor["heading"] = latest_sensor["last_known_valid_heading"]
                        latest_sensor["gps_valid"]       = True
                        latest_sensor["location_source"] = "last_known"

            else:
                # GNSS recovered — disengage fallback
                if latest_sensor.get("location_source") in ("fallback", "last_known"):
                    latest_sensor["location_source"] = "gnss"
                    hardware_state["gps"]["logged_fallback"] = False
                    print("\n" + "=" * 65)
                    print(" [HARDWARE] GPS FALLBACK -> GNSS RECOVERED")
                    print("            Hardware GNSS fix restored — IP fallback disengaged.")
                    print("=" * 65 + "\n")

        stop_event.wait(5.0)


def _log_fallback(raw_lat, raw_lon, snapped_lat, snapped_lon, dist_drift, in_sz, city, ip):
    label = city or ip or "IP-geolocation"
    print("\n" + "=" * 65)
    if in_sz:
        print(" [HARDWARE] GPS FALLBACK -> SAFE ZONE ACTIVE (NO SNAP)")
        print(f"            IP Location ({label}): ({raw_lat:.8f}, {raw_lon:.8f}) [🛡 Municipal Depot Safe Zone]")
    else:
        print(" [HARDWARE] GPS FALLBACK -> ROAD SNAP APPLIED")
        print(f"            Raw IP GPS:  ({raw_lat:.8f}, {raw_lon:.8f})")
        print(f"            Road Snap:   ({snapped_lat:.8f}, {snapped_lon:.8f}) [Drift: {dist_drift:.1f}m]")
    print("=" * 65 + "\n")
