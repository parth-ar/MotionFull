"""
geofence.py — Geofencing and field-tracking logic.

Verbatim port of all geofencing functions from webcam_motion_detect.py.
No logic has been changed — functions, thresholds, and variable names are
preserved byte-for-byte.  Only the module structure has changed (moved from
the monolithic script into this dedicated module).

Exported:
  haversine_dist_meters(lat1, lon1, lat2, lon2) → float
  snap_coordinates_to_road(lat, lon) → (lat, lon, in_safe_zone, is_snapped, road_bearing)
  is_in_safe_zone(lat, lon) → bool
  track_field_events(lat, lon, speed, heading)
  compute_heading_from_gps_history(raw_lat, raw_lon, in_safe_zone, is_snapped, road_bearing) → float|None
  angular_distance(a, b) → float
  get_nearest_house(lat, lon) → (house, dist)

Mutable state shared with main.py / uploader.py:
  dynamic_houses   : list[dict]  — populated by sync_backend_metadata()
  dynamic_roads    : list[dict]  — populated by sync_backend_metadata()
  dynamic_safe_zones: list[dict] — populated by sync_backend_metadata()
  field_state      : dict
"""

import math
import time

# ---------------------------------------------------------------------------
# Shared mutable GIS state (populated by network/uploader.sync_backend_metadata)
# ---------------------------------------------------------------------------
dynamic_houses: list = []
dynamic_roads:  list = []
dynamic_safe_zones: list = [
    {"ulbId": "ULB_MH_AMRAVATI", "lat": 20.928816, "lon": 77.7514375, "radius": 1000.0}
]

field_state: dict = {
    "in_depot":       None,
    "in_corridor":    None,
    "is_stopped":     None,
    "marked_houses":  set(),
    "last_house_near": None,
    "last_dist_log_time": 0.0,
}

# GPS heading history
gps_sliding_window: list = []   # [(lat, lon, timestamp), ...]
last_known_gps_heading = None


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def haversine_dist_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


haversine_distance_m = haversine_dist_meters


def angular_distance(a, b) -> float:
    """Calculates the minimal circular distance between two angles in degrees (0 - 180)."""
    if a is None or b is None:
        return 0.0
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def compute_heading_from_gps_history(raw_lat, raw_lon,
                                      in_safe_zone=False, is_snapped=False,
                                      road_bearing=None,
                                      hardware_course_deg=None):
    """Calculates context-aware vehicle motion heading.

    Priority order:
    1. Hardware Doppler course from $RMC/$VTG (hardware_course_deg) — available at
       1 Hz even when the vehicle is slow or the position barely changes. This is
       preferred over geometric displacement which fails at low speeds because the
       GPS sliding window covers < 0.8 m at < 10 km/h at 1 Hz update rate.
    2. Outside Safe Zone & Snapped to Road: align hardware course parallel to the
       winning road segment (forward vs. reverse).
    3. Inside Safe Zone or Off-Road: use raw GPS displacement vector (geometric).
    4. Stationary (displacement < 0.8 m and no hardware course): hold previous
       valid heading without jitter.

    Args:
        raw_lat, raw_lon:      Current raw GPS coordinates.
        in_safe_zone:          True when inside a depot/safe zone.
        is_snapped:            True when road-snapping was applied.
        road_bearing:          Bearing of the matched road segment (degrees).
        hardware_course_deg:   True-north course from $RMC / $VTG (degrees). Pass
                               gnss_data.get("course_deg") — None if unavailable.
    """
    global gps_sliding_window, last_known_gps_heading
    if raw_lat is None or raw_lon is None or (abs(raw_lat) < 0.001 and abs(raw_lon) < 0.001):
        return last_known_gps_heading

    now = time.monotonic()
    gps_sliding_window.append((float(raw_lat), float(raw_lon), now))
    # Retain fixes across 2-4 second sliding window (3-4 points)
    gps_sliding_window = [p for p in gps_sliding_window if now - p[2] <= 4.5][-8:]

    # --- Priority 1: Hardware Doppler course from $RMC / $VTG ---
    # Only use it when the vehicle is clearly moving (course > ~1 km/h generates
    # a valid non-zero Doppler reading). Below ~0.5 km/h the module outputs 0.0
    # which is indistinguishable from true North — hold the last known heading.
    if hardware_course_deg is not None:
        candidate_course = float(hardware_course_deg)
        if candidate_course >= 0.0:  # 0.0 could be true North or stopped; prefer if > 0
            travel_angle = candidate_course
            if is_snapped and not in_safe_zone and road_bearing is not None:
                # Align to road segment direction
                rev_bearing = (road_bearing + 180.0) % 360.0
                diff_fwd = angular_distance(travel_angle, road_bearing)
                diff_rev = angular_distance(travel_angle, rev_bearing)
                travel_angle = road_bearing if diff_fwd <= diff_rev else rev_bearing
            last_known_gps_heading = travel_angle
            return travel_angle

    # --- Priority 2 / 3: Geometric displacement fallback ---
    if len(gps_sliding_window) >= 2:
        old_lat, old_lon, _ = gps_sliding_window[0]
        lat_scale = 111320.0
        lon_scale = 111320.0 * math.cos(math.radians(raw_lat))

        dx = (raw_lon - old_lon) * lon_scale
        dy = (raw_lat - old_lat) * lat_scale
        dist = math.hypot(dx, dy)

        if dist >= 0.8:
            travel_angle = float(math.degrees(math.atan2(dx, dy)))
            if travel_angle < 0:
                travel_angle += 360.0

            # Inside Safe Zone or Unsnapped Off-Road → Direct GPS Travel Vector
            if in_safe_zone or (not is_snapped) or road_bearing is None:
                last_known_gps_heading = travel_angle
                return travel_angle

            # Outside Safe Zone along Road Corridor → Align parallel to Road Segment
            rev_bearing = (road_bearing + 180.0) % 360.0
            diff_fwd = angular_distance(travel_angle, road_bearing)
            diff_rev = angular_distance(travel_angle, rev_bearing)

            aligned_heading = road_bearing if diff_fwd <= diff_rev else rev_bearing
            last_known_gps_heading = aligned_heading
            return aligned_heading

    return last_known_gps_heading


def get_nearest_house(lat: float, lon: float):
    best_h, best_dist = None, 999999.0
    for h in dynamic_houses:
        d = haversine_dist_meters(lat, lon, h["lat"], h["lon"])
        if d < best_dist:
            best_dist = d
            best_h = h
    return best_h, best_dist


def is_in_safe_zone(lat, lon) -> bool:
    """Checks if coordinates fall inside any configured ULB safe zone (depot / garage)."""
    if lat is None or lon is None:
        return False
    for sz in dynamic_safe_zones:
        sz_lat = sz.get("lat") or sz.get("latitude")
        sz_lon = sz.get("lon") or sz.get("longitude")
        radius = sz.get("radius") or sz.get("radiusMeters") or 1000.0
        if sz_lat is not None and sz_lon is not None:
            d = haversine_dist_meters(lat, lon, sz_lat, sz_lon)
            if d <= radius:
                return True
    return False


def snap_coordinates_to_road(lat, lon):
    """Road snapping permanently disabled: raw GNSS coordinates returned untouched."""
    if lat is None or lon is None:
        return lat, lon, False, False, None
    in_sz = is_in_safe_zone(lat, lon)
    return round(float(lat), 8), round(float(lon), 8), in_sz, False, None


def track_field_events(lat, lon, speed, heading) -> None:
    if lat is None or lon is None or (abs(lat) < 0.001 and abs(lon) < 0.001):
        return

    # Dynamic Vehicle Stop detection
    is_stopped = (speed <= 3.5)

    if field_state["is_stopped"] != is_stopped:
        field_state["is_stopped"] = is_stopped
        if is_stopped:
            print(f"\n[FIELD LOG] 🛑 VEHICLE STOPPED at ({lat:.6f}, {lon:.6f}) | Speed: {speed:.1f} km/h\n")
        else:
            print(f"\n[FIELD LOG] 🚚 LEAVING STOP / MOVING: Resumed motion at {speed:.1f} km/h (Heading: {heading:.0f}°)\n")
