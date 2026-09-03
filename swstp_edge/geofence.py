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


def angular_distance(a, b) -> float:
    """Calculates the minimal circular distance between two angles in degrees (0 - 180)."""
    if a is None or b is None:
        return 0.0
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def compute_heading_from_gps_history(raw_lat, raw_lon,
                                      in_safe_zone=False, is_snapped=False,
                                      road_bearing=None):
    """Calculates context-aware vehicle motion heading:
    - Outside Safe Zone & Snapped to Road: Aligns parallel to winning road segment (forward vs reverse).
    - Inside Safe Zone or Off-Road: Follows raw GPS displacement vector.
    - Stationary (displacement < 0.8m): Holds previous valid heading without jitter.
    """
    global gps_sliding_window, last_known_gps_heading
    if raw_lat is None or raw_lon is None or (abs(raw_lat) < 0.001 and abs(raw_lon) < 0.001):
        return last_known_gps_heading

    now = time.monotonic()
    gps_sliding_window.append((float(raw_lat), float(raw_lon), now))
    # Retain fixes across 2-4 second sliding window (3-4 points)
    gps_sliding_window = [p for p in gps_sliding_window if now - p[2] <= 4.5][-8:]

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

            # 1. Inside Safe Zone or Unsnapped Off-Road -> Direct GPS Travel Vector
            if in_safe_zone or (not is_snapped) or road_bearing is None:
                last_known_gps_heading = travel_angle
                return travel_angle

            # 2. Outside Safe Zone along Road Corridor -> Align parallel to Road Segment
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
    """Projects raw or drifted GPS coordinates directly onto the road centerline so all stored
    telemetry and collection circles are 100% on the road, UNLESS the vehicle is inside a safe zone.
    Returns: (display_lat, display_lon, is_inside_safe_zone, is_snapped, road_bearing)
    """
    if lat is None or lon is None or not dynamic_roads:
        return lat, lon, False, False, None

    # Safe Zone Exemption: Do NOT snap to road if vehicle is inside a safe zone (depot / yard)
    in_sz = is_in_safe_zone(lat, lon)
    if in_sz:
        return lat, lon, True, False, None

    best_lat, best_lon = lat, lon
    best_road_bearing = None
    min_dist = float("inf")

    ref_lat = dynamic_roads[0]["coordinates"][0][0]
    ref_lon = dynamic_roads[0]["coordinates"][0][1]
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians(ref_lat))

    px = (lon - ref_lon) * lon_scale
    py = (lat - ref_lat) * lat_scale

    for road in dynamic_roads:
        coords = road.get("coordinates") or []
        for i in range(len(coords) - 1):
            a_lat, a_lon = coords[i]
            b_lat, b_lon = coords[i + 1]

            ax = (a_lon - ref_lon) * lon_scale
            ay = (a_lat - ref_lat) * lat_scale
            bx = (b_lon - ref_lon) * lon_scale
            by = (b_lat - ref_lat) * lat_scale

            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            if l2 < 1e-6:
                t = 0.0
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))

            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)

            if dist < min_dist:
                min_dist = dist
                best_lat = round(ref_lat + proj_y / lat_scale, 8)
                best_lon = round(ref_lon + proj_x / lon_scale, 8)
                seg_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                best_road_bearing = seg_bearing

    # If within 80m corridor of the road, snap permanently to centerline
    if min_dist <= 80.0:
        is_snapped = (abs(best_lat - lat) > 1e-8 or abs(best_lon - lon) > 1e-8)
        return best_lat, best_lon, False, is_snapped, best_road_bearing

    return round(lat, 8), round(lon, 8), False, False, None


def track_field_events(lat, lon, speed, heading) -> None:
    if lat is None or lon is None or (abs(lat) < 0.001 and abs(lon) < 0.001):
        return

    now = time.monotonic()

    # Dynamic Vehicle Stop / House Proximity & 10m Collection Zone
    is_stopped = (speed <= 3.5)
    near_house, house_dist = get_nearest_house(lat, lon)

    # Import here to avoid circular import; latest_sensor lives in telemetry.py
    from telemetry import latest_sensor
    active_sid = latest_sensor.get("active_session_id", 0)

    if field_state["is_stopped"] != is_stopped:
        field_state["is_stopped"] = is_stopped
        if is_stopped:
            if near_house and house_dist <= 15.0:
                h_id = near_house["id"]
                field_state["marked_houses"].add(h_id)
                print("\n" + "=" * 65)
                print(f"[FIELD LOG] 🛑 VEHICLE STOP AT HOUSE: {h_id} - {near_house['name']}")
                print(f"            Proximity Dist: {house_dist:.1f}m (Threshold <= 15m) | Speed: {speed:.1f} km/h")
                print(f"            🏠 HOUSE MARKED AS COLLECTED (GIS Map Status -> SOLID GREEN)")
                print(f"            Total Houses Collected: {len(field_state['marked_houses'])} / {len(dynamic_houses)}")
                print(f"            Active Session: #{active_sid}")
                print("=" * 65 + "\n")
            else:
                print(f"\n[FIELD LOG] 🛑 VEHICLE STOPPED at ({lat:.6f}, {lon:.6f}) | Speed: {speed:.1f} km/h\n")
        else:
            print(f"\n[FIELD LOG] 🚚 LEAVING STOP / MOVING: Resumed motion at {speed:.1f} km/h (Heading: {heading:.0f}°)\n")

    # Periodic proximity heartbeat when near registered building (5s when moving, 30s when stopped)
    elif near_house and house_dist <= 25.0:
        heartbeat_interval = 30.0 if is_stopped else 5.0
        if (now - field_state["last_dist_log_time"]) >= heartbeat_interval:
            field_state["last_dist_log_time"] = now
            print(f"[FIELD PROXIMITY] Nearest House: {near_house['id']} ({near_house['name']}) @ {house_dist:.1f}m | Speed: {speed:.1f} km/h")
