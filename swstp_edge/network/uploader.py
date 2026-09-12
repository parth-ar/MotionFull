"""
network/uploader.py — Backend HTTP worker threads.

Verbatim port of all three HTTP upload workers from webcam_motion_detect.py:
  live_frame_streamer     → POST /api/camera/{deviceId}/frame
  telemetry_streamer      → POST /api/telemetry/ingest-batch
  evidence_upload_worker  → POST /api/evidence/upload

Also contains:
  auto_detect_backend_url()
  sync_backend_metadata()
  ensure_hardware_session()
  log_hardware()

No changes to payload shapes, endpoint paths, or HTTP behavior.
"""

import datetime
import json
import os
import queue
import time
import threading
import uuid

import requests
import requests.adapters

from config import DEFAULT_BACKEND_URL, DEFAULT_ULB_ID, DEFAULT_VEHICLE_ID
import geofence

# Dynamic session metadata (shared mutable — mirrors the original globals)
dynamic_session_info: dict = {
    "sessionId":     0,
    "ulbId":         DEFAULT_ULB_ID,
    "vehicleReg":    DEFAULT_VEHICLE_ID,
    "deviceId":      "UNASSIGNED",
    "authenticated": False,
}


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log_hardware(component: str, status: str, details: str = "") -> None:
    """Prints a prominent hardware status log with clean formatting."""
    print("\n" + "=" * 65)
    print(f" [HARDWARE] {component.upper()} -> {status.upper()}")
    if details:
        print(f"            {details}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------
def auto_detect_backend_url(candidate: str | None = None) -> str:
    """Automatically discovers and connects to the active SWSTP backend API."""
    candidates = []
    if candidate:
        candidates.append(candidate)
    import os
    if "SWSTP_BACKEND_URL" in os.environ:
        candidates.append(os.environ["SWSTP_BACKEND_URL"])
    candidates.extend([
        "https://solidwasteapi.scipl.info.in",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:5244",
    ])
    for u in candidates:
        try:
            r = requests.get(f"{u.rstrip('/')}/api/gis/roads?ulbId={DEFAULT_ULB_ID}", timeout=3.0)
            if r.status_code in (200, 401, 403, 404):
                return u
        except Exception:
            continue
    return candidate or "https://solidwasteapi.scipl.info.in"


def sync_backend_metadata(backend_url: str, ulb_id: str) -> None:
    """Dynamically synchronizes registered ULBs, houses, roads, and active sessions from Backend API."""
    base = backend_url.rstrip('/')

    # 1. Resolve ULB dynamically
    resolved_ulb = ulb_id if (ulb_id and ulb_id != "AUTO") else "ULB_MH_AMRAVATI"
    try:
        r_ulbs = requests.get(f"{base}/api/admin/ulbs", timeout=2.0)
        if r_ulbs.status_code == 200:
            ulbs_data = r_ulbs.json()
            if isinstance(ulbs_data, list) and len(ulbs_data) > 0:
                first_ulb = ulbs_data[0].get("ulbId") or ulbs_data[0].get("id")
                if first_ulb and (not ulb_id or ulb_id == "AUTO"):
                    resolved_ulb = first_ulb
    except Exception:
        pass
    dynamic_session_info["ulbId"] = resolved_ulb

    # 2. Fetch dynamic houses from API
    try:
        r_houses = requests.get(f"{base}/api/gis/houses?ulbId={resolved_ulb}", timeout=3.0)
        if r_houses.status_code == 200:
            data = r_houses.json()
            if isinstance(data, list) and len(data) > 0:
                geofence.dynamic_houses = [
                    {
                        "id":   h.get("houseId") or f"H-{i+1}",
                        "name": h.get("address") or h.get("houseId") or f"House {i+1}",
                        "lat":  float(h.get("latitude", 0)),
                        "lon":  float(h.get("longitude", 0)),
                    }
                    for i, h in enumerate(data)
                    if h.get("latitude") and h.get("longitude")
                ]
                print(f"[API METADATA] Synchronized {len(geofence.dynamic_houses)} registered houses from DB (ULB: {resolved_ulb}).")
    except Exception as ex:
        print(f"[API METADATA NOTE] Houses sync note: {ex}")

    # 3. Fetch dynamic roads from API
    try:
        r_roads = requests.get(f"{base}/api/gis/roads?ulbId={resolved_ulb}", timeout=3.0)
        if r_roads.status_code == 200:
            data = r_roads.json()
            if isinstance(data, list):
                geofence.dynamic_roads = data
                print(f"[API METADATA] Synchronized {len(geofence.dynamic_roads)} road geometry layers from DB.")
    except Exception as ex:
        print(f"[API METADATA NOTE] Roads sync note: {ex}")

    # 4. Fetch dynamic safe zones from API
    try:
        r_safe = requests.get(f"{base}/api/gis/safe-zones?ulbId={resolved_ulb}", timeout=3.0)
        if r_safe.status_code == 200:
            data = r_safe.json()
            if isinstance(data, list) and len(data) > 0:
                geofence.dynamic_safe_zones = data
                print(f"[API METADATA] Synchronized {len(geofence.dynamic_safe_zones)} safe zones from DB (ULB: {resolved_ulb}).")
    except Exception as ex:
        print(f"[API METADATA NOTE] Safe zones sync note: {ex}")

    # Note: Sessions are strictly bound per device identity via ensure_hardware_session().
    # Never hijack arbitrary sessions from other devices.


def ensure_hardware_session(backend_url: str, device_code: str) -> int:
    """Dynamically activates or binds the hardware session in the DB for the identified device.
    
    Device ID is the primary source of authentication. If registration fails or the device
    code is invalid, session creation is blocked and no telemetry/evidence is uploaded.
    """
    if not device_code or device_code in ("AUTO", "UNASSIGNED", "DISCONNECTED", "UNPROVISIONED"):
        log_hardware("AUTH REJECTED", "BLOCKED", f"Invalid or unprovisioned device ID: '{device_code}'. All portal uploads disabled.")
        dynamic_session_info["sessionId"]     = 0
        dynamic_session_info["authenticated"] = False
        return 0

    base = backend_url.rstrip('/')
    try:
        r = requests.post(
            f"{base}/api/sessions/start-hardware-session",
            json={"deviceCode": device_code, "mode": "REAL_HARDWARE"},
            timeout=3.0,
        )
        if r.status_code == 200:
            sess = r.json()
            sid  = sess.get("sessionId") or 0
            if sid > 0:
                from telemetry import latest_sensor, hardware_state
                dynamic_session_info["sessionId"]     = sid
                dynamic_session_info["vehicleReg"]    = (
                    sess.get("vehicleRegistrationNumber") or
                    (sess.get("vehicle") or {}).get("registrationNumber") or
                    dynamic_session_info["vehicleReg"]
                )
                dynamic_session_info["ulbId"]         = sess.get("ulbId") or (sess.get("ulb") or {}).get("ulbId") or dynamic_session_info["ulbId"]
                dynamic_session_info["deviceId"]      = device_code
                dynamic_session_info["authenticated"] = True
                latest_sensor["active_session_id"]             = sid
                hardware_state["backend"]["active_session_id"] = sid

                last_seq = sess.get("lastSequenceNumber") or sess.get("maxSequenceNumber") or 0
                if last_seq > latest_sensor.get("sequence", 0):
                    latest_sensor["sequence"] = int(last_seq)

                print(f"[SESSION BIND] Telemetry Session #{sid} AUTHENTICATED for Vehicle '{dynamic_session_info['vehicleReg']}' (Device: {device_code})")
                return sid
        else:
            log_hardware("AUTH FAILED", "REJECTED", f"Backend rejected device '{device_code}': HTTP {r.status_code} - {r.text}")
            dynamic_session_info["sessionId"]     = 0
            dynamic_session_info["authenticated"] = False
            return 0
    except Exception as ex:
        log_hardware("AUTH ERROR", "EXCEPTION", f"Failed to authenticate device '{device_code}': {ex}")
        dynamic_session_info["sessionId"]     = 0
        dynamic_session_info["authenticated"] = False
        return 0
    return 0


# ---------------------------------------------------------------------------
# Live Frame Streamer (POST /api/camera/{deviceId}/frame)
# ---------------------------------------------------------------------------
def live_frame_streamer(backend_url: str, device_id: str, fps: float,
                         stop_event: threading.Event,
                         frame_lock: threading.Lock | None = None,
                         get_frame=None) -> None:
    """Pushes live JPEG frames to Backend asynchronously, encoding frames on-demand without loading the main loop.

    frame_lock  — threading.Lock protecting the shared frame buffer (passed from main.py)
    get_frame   — callable() → numpy array | None; returns the latest BGR frame
                  (injected by main.py to avoid a circular import)

    Resilience: throttled error logging + exponential back-off reconnect so a transient
    network drop doesn't spam the console or stall frame delivery.
    """
    from telemetry import latest_sensor, hardware_state
    import cv2

    target_fps = max(1.0, min(30.0, fps))
    interval   = 1.0 / target_fps

    # Provide a no-op fallback if the caller doesn't inject frame accessors
    _lock      = frame_lock or threading.Lock()
    _get_frame = get_frame or (lambda: None)

    def _make_session():
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=0)
        s.mount("http://", a)
        s.mount("https://", a)
        return s

    session          = _make_session()
    logged_first_ok  = False
    last_sent_time   = 0.0   # monotonic write-timestamp of the last frame we sent

    # Error back-off state
    _err_count         = 0
    _last_err_log      = 0.0
    _ERR_LOG_INTERVAL  = 10.0   # seconds between repeated error messages
    _backoff_until     = 0.0    # epoch: don't attempt to send until this time

    while not stop_event.is_set():
        if not dynamic_session_info.get("authenticated", False):
            # Block frame streaming if device is not authenticated
            stop_event.wait(2.0)
            continue

        loop_start = time.time()

        # Honour back-off window (reconnect delay after repeated failures)
        if loop_start < _backoff_until:
            stop_event.wait(min(0.5, _backoff_until - loop_start))
            continue

        frame_to_send = None
        with _lock:
            # _get_frame() returns a (ndarray, write_time) tuple or a bare ndarray
            # (bare ndarray kept for backward compat with tests)
            candidate = _get_frame()
            if candidate is not None:
                if isinstance(candidate, tuple):
                    arr, write_time = candidate
                else:
                    arr, write_time = candidate, time.monotonic()
                # Send whenever the write_time is strictly newer than last send
                if arr is not None and write_time > last_sent_time:
                    frame_to_send  = arr
                    last_sent_time = write_time

        if frame_to_send is not None:
            try:
                _, enc = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_bytes = enc.tobytes()
                cur_dev    = latest_sensor.get("deviceId") or dynamic_session_info.get("deviceId") or device_id
                stream_url = f"{backend_url.rstrip('/')}/api/camera/{cur_dev}/frame"
                resp = session.post(
                    stream_url,
                    data=frame_bytes,
                    headers={"Content-Type": "image/jpeg", "Connection": "keep-alive"},
                    timeout=1.5,
                )
                if resp.status_code == 200:
                    hardware_state["backend"]["connected"]  = True
                    hardware_state["backend"]["last_ping"]  = time.time()
                    if not logged_first_ok:
                        logged_first_ok = True
                        print(f"\n[CAMERA STREAM] Active -> Streaming frames to {stream_url} @ {target_fps:.1f} FPS (HTTP 200 OK)\n")
                    # Reset error counters on success
                    _err_count     = 0
                    _backoff_until = 0.0
                else:
                    # Non-200 but server responded: count as a soft error, no back-off
                    hardware_state["backend"]["connected"] = False
                    now = time.time()
                    if now - _last_err_log > _ERR_LOG_INTERVAL:
                        print(f"[CAMERA STREAM] Backend returned HTTP {resp.status_code} — will retry.")
                        _last_err_log = now

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as net_err:
                hardware_state["backend"]["connected"] = False
                _err_count += 1
                now = time.time()
                if now - _last_err_log > _ERR_LOG_INTERVAL:
                    print(f"[CAMERA STREAM] Network error (attempt {_err_count}): {net_err}")
                    _last_err_log = now
                # Exponential back-off: 0.5 s, 1 s, 2 s … up to 8 s
                backoff = min(8.0, 0.5 * (2 ** min(_err_count - 1, 4)))
                _backoff_until = now + backoff
                # Rebuild session to clear any stale connections
                try:
                    session.close()
                except Exception:
                    pass
                session = _make_session()

            except Exception as exc:
                hardware_state["backend"]["connected"] = False
                now = time.time()
                if now - _last_err_log > _ERR_LOG_INTERVAL:
                    print(f"[CAMERA STREAM] Unexpected error: {exc}")
                    _last_err_log = now

        elapsed   = time.time() - loop_start
        sleep_dur = max(0.005, interval - elapsed)
        stop_event.wait(sleep_dur)

    try:
        session.close()
    except Exception:
        pass



# ---------------------------------------------------------------------------
# Telemetry Batch Ingestion Worker (POST /api/telemetry/ingest-batch)
# ---------------------------------------------------------------------------
def telemetry_streamer(backend_url: str, session_id_arg: int, ulb_id: str,
                        stop_event: threading.Event) -> None:
    """Sends sequential telemetry batches to backend with HTTP keep-alive connection pooling."""
    from telemetry import latest_sensor, hardware_state, telemetry_queue
    import datetime

    url = f"{backend_url.rstrip('/')}/api/telemetry/ingest-batch"
    active_session_query_url = f"{backend_url.rstrip('/')}/api/officer/sessions?ulbId={ulb_id}&status=ACTIVE"

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    current_session_id = session_id_arg or dynamic_session_info.get("sessionId") or 0
    last_session_check = 0.0

    while not stop_event.is_set():
        active_dyn_sid = dynamic_session_info.get("sessionId", 0)
        is_auth = dynamic_session_info.get("authenticated", False)

        if not is_auth or active_dyn_sid <= 0:
            # Drop telemetry queue while unauthenticated
            while not telemetry_queue.empty():
                try:
                    telemetry_queue.get_nowait()
                except queue.Empty:
                    break
            stop_event.wait(2.0)
            continue

        current_session_id = active_dyn_sid

        packets = []
        while not telemetry_queue.empty() and len(packets) < 50:
            try:
                packets.append(telemetry_queue.get_nowait())
            except queue.Empty:
                break

        if not packets and (latest_sensor["timestamp"] or latest_sensor["gps_valid"] or latest_sensor["imu"]):
            # Heartbeat slim packet — same shape as what the 20 Hz loop sends
            from telemetry import build_slim_packet, build_telemetry_packet
            latest_sensor["sequence"] += 1
            try:
                # Build a fresh full packet then slim it down (reuses all local state)
                # We pass device_id from dynamic_session_info since we don't have it locally
                full_hb = build_telemetry_packet(
                    dynamic_session_info.get("deviceId") or "UNPROVISIONED"
                )
                packets.append(build_slim_packet(full_hb))
            except Exception as hb_err:
                # Absolute fallback: hand-craft a minimal slim packet from latest_sensor
                import datetime as _dt
                s = latest_sensor
                now_ms = int(time.time() * 1000)
                imu_s  = s.get("imu") or {}
                orient = (imu_s.get("orientation") or {})
                packets.append({
                    "seq":       s.get("sequence"),
                    "ts":        s.get("timestamp") or _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    "epoch":     now_ms,
                    "uptime":    now_ms,
                    "lat":       round(float(s["lat"]), 8)  if (s.get("gps_valid") and s.get("lat")  is not None) else None,
                    "lon":       round(float(s["lon"]), 8)  if (s.get("gps_valid") and s.get("lon")  is not None) else None,
                    "alt":       s.get("alt")    or 0.0,
                    "spd":       s.get("speed")  or 0.0,
                    "hdg":       s.get("heading"),
                    "sats":      s.get("satellites") or 0,
                    "hdop":      None,
                    "fix":       bool(s.get("gps_valid")),
                    "snapped":   bool(s.get("is_snapped")),
                    "safe_zone": bool(s.get("is_inside_safe_zone")),
                    "loc_src":   s.get("location_source"),
                    "accel_mag": round(float(imu_s.get("accel_magnitude_ms2") or 0.0), 3),
                    "roll":      round(float(orient.get("roll")  or 0.0), 2),
                    "pitch":     round(float(orient.get("pitch") or 0.0), 2),
                    "yaw":       round(float(orient.get("yaw")   or 0.0), 2),
                    "temp_c":    round(float(imu_s.get("temperature_c") or 0.0), 2),
                    "imu_ok":    bool(imu_s.get("valid")),
                    "clk_src":   "unknown",
                })


        if packets:
            batch = {
                "sessionId": current_session_id,
                "mode":      "REAL_HARDWARE",
                "packets":   packets,
            }
            try:
                resp = session.post(url, json=batch, timeout=2.0)
                if resp.status_code == 200:
                    hardware_state["backend"]["connected"]        = True
                    hardware_state["backend"]["telemetry_count"] += len(packets)
            except Exception:
                hardware_state["backend"]["connected"] = False

        stop_event.wait(0.1)  # 10 Hz batch flush for ultra-smooth sequential streaming

    session.close()


# ---------------------------------------------------------------------------
# Evidence Upload Worker & Offline Backlog Synchronizer (POST /api/evidence/upload)
# ---------------------------------------------------------------------------
_in_flight_evidence_files: set = set()
_in_flight_evidence_lock = threading.Lock()


def _upload_and_cleanup_evidence(session: requests.Session,
                                 url: str,
                                 backend_url: str,
                                 device_id: str,
                                 ulb_id: str,
                                 active_sid: int,
                                 item: dict) -> bool:
    """Uploads one capture to POST /api/evidence/upload and deletes local backup upon confirmed receipt (HTTP 200/201).
    
    Returns True if confirmed and local files deleted; False if upload failed (files retained as backup).
    """
    from telemetry import latest_sensor, hardware_state

    local_path = item.get("local_path")
    meta_path  = item.get("meta_path")
    jpeg_bytes = item.get("jpeg_bytes")

    # If bytes not in memory, read from local backup on disk
    if not jpeg_bytes and local_path and os.path.exists(local_path):
        try:
            with open(local_path, "rb") as f:
                jpeg_bytes = f.read()
        except Exception as read_err:
            print(f"[EVIDENCE UPLOAD] Error reading local capture {local_path}: {read_err}")
            return False

    if not jpeg_bytes:
        print(f"[EVIDENCE UPLOAD] No JPEG bytes available for item: {local_path}")
        return False

    captured_at         = item.get("captured_at") or datetime.datetime.now(datetime.timezone.utc).isoformat()
    collection_event_id = item.get("collection_event_id", 0)
    idempotency_key     = item.get("idempotency_key") or str(uuid.uuid4())
    width               = item.get("width",  1280)
    height              = item.get("height", 720)
    compression_quality = item.get("compression_quality", 80)

    lat = item.get("latitude")
    lon = item.get("longitude")
    if lat is None or lon is None or (abs(lat) < 0.001 and abs(lon) < 0.001):
        lat = latest_sensor.get("lat")
        lon = latest_sensor.get("lon")
        if lat is None or lon is None or (abs(lat) < 0.001 and abs(lon) < 0.001):
            lat = latest_sensor.get("last_known_valid_lat") or 0.0
            lon = latest_sensor.get("last_known_valid_lon") or 0.0

    speed = item.get("speed_kph", item.get("vehicle_speed_kmh", 0.0))
    if speed is None:
        speed = latest_sensor.get("speed") or 0.0

    data = {
        "collectionEventId":  collection_event_id,
        "capturedAt":         captured_at,
        "width":              width,
        "height":             height,
        "compressionQuality": compression_quality,
        "idempotencyKey":     idempotency_key,
        "latitude":           round(float(lat), 8) if lat is not None else 0.0,
        "longitude":          round(float(lon), 8) if lon is not None else 0.0,
        "speedKph":           speed,
        "motionConfidence":   item.get("motionConfidence", 0.95),
        "sessionId":          active_sid,
        "deviceId":           device_id,
        "ulbId":              ulb_id,
    }

    # ── Optional enrichment fields (litter / stop duration) ───────────────
    stop_duration_sec    = item.get("stop_duration_sec", item.get("stop_offset_sec"))
    detection_type       = item.get("detection_type") or item.get("detectionType")
    detection_confidence = item.get("detection_confidence") or item.get("detectionConfidence")

    if stop_duration_sec is not None:
        data["stopDurationSec"] = round(float(stop_duration_sec), 2)
    if detection_type:
        data["detectionType"] = detection_type
    if detection_confidence is not None:
        data["detectionConfidence"] = round(float(detection_confidence), 4)

    file_display_name = os.path.basename(local_path) if local_path else f"evidence_{idempotency_key[:8]}.jpg"
    resp = None
    for attempt in range(2):
        try:
            resp = session.post(
                url,
                files={"file": (file_display_name, jpeg_bytes, "image/jpeg")},
                data=data,
                timeout=15.0,
            )
            if resp.status_code in (200, 201):
                break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as req_err:
            if attempt == 0:
                time.sleep(1.0)
                continue
            print(f"[EVIDENCE UPLOAD] Network/timeout error connecting to backend: {req_err}")
            print(f"[OFFLINE BACKUP] Retaining local backup on disk: {file_display_name}")
            return False
        except Exception as exc:
            print(f"[EVIDENCE UPLOAD] Unexpected error: {exc}")
            return False

    if resp is not None and resp.status_code in (200, 201):
        try:
            res_json = resp.json()
        except Exception:
            res_json = {}

        hardware_state["backend"]["upload_count"] += 1
        img_id  = res_json.get("evidenceImageId") or res_json.get("id") or "N/A"
        img_url = res_json.get("imageUrl") or f"{backend_url.rstrip('/')}/api/evidence/images/{img_id}"

        if detection_type == "LITTER":
            house_tag = f"🟡 Road Litter Marker @ ({lat:.8f}, {lon:.8f})"
        else:
            near_h, h_dist = geofence.get_nearest_house(lat, lon)
            house_tag = (
                f"{near_h['id']} - {near_h['name']} (@ {h_dist:.1f}m)"
                if (near_h and h_dist <= 25.0)
                else "Road Corridor (Auto-allocated)"
            )
            # ── Gate: mark house green ONLY on confirmed backend receipt ────────
            if near_h and h_dist <= 25.0:
                h_id = near_h["id"]
                geofence.field_state["marked_houses"].add(h_id)
                print(f"[FIELD LOG] 🏠 HOUSE MARKED AS COLLECTED (EVIDENCE CONFIRMED by backend)")
                print(f"            House: {h_id} - {near_h['name']} (@ {h_dist:.1f}m)")
                print(f"            Total collected: {len(geofence.field_state['marked_houses'])} / {len(geofence.dynamic_houses)}")

        raw_lat_val = latest_sensor.get("raw_gps_lat") or lat
        raw_lon_val = latest_sensor.get("raw_gps_lon") or lon

        print("\n" + "=" * 65)
        print(f"[FIELD LOG] 📸 EVIDENCE UPLOADED & CONFIRMED BY BACKEND")
        print(f"            EvidenceImageId: #{img_id}")
        print(f"            Server Path:     {res_json.get('relativePath', 'N/A')}")
        if geofence.is_in_safe_zone(raw_lat_val, raw_lon_val):
            print(f"            GPS Coordinates: ({lat:.8f}, {lon:.8f}) [🛡 SAFE ZONE - NO SNAP]")
        elif abs(raw_lat_val - lat) > 0.00000001 or abs(raw_lon_val - lon) > 0.00000001:
            drift_val = geofence.haversine_dist_meters(raw_lat_val, raw_lon_val, lat, lon)
            print(f"            Real Raw GPS:    ({raw_lat_val:.8f}, {raw_lon_val:.8f})")
            print(f"            Road Snapped GPS:({lat:.8f}, {lon:.8f}) [Correction: {drift_val:.1f}m]")
        else:
            print(f"            GPS Coordinates: ({lat:.8f}, {lon:.8f})")
        print(f"            Associated House:{house_tag}")
        print(f"            Access URL:      {img_url}")
        print(f"            Session ID:      #{active_sid}")
        print("=" * 65)

        # ── IMMEDIATE LOCAL CLEANUP ──────────────────────────────────────────
        # Upload is confirmed by backend -> Delete local staging/backup files immediately
        deleted_list = []
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                deleted_list.append(os.path.basename(local_path))
            except Exception as e:
                print(f"[LOCAL STORAGE] Warning removing local image file {local_path}: {e}")
        if meta_path and os.path.exists(meta_path):
            try:
                os.remove(meta_path)
                deleted_list.append(os.path.basename(meta_path))
            except Exception:
                pass
        if deleted_list:
            print(f"[LOCAL STORAGE CLEANUP] ✔ Upload confirmed -> Deleted local backup: {', '.join(deleted_list)}\n")
        return True
    elif resp is not None:
        print(f"[EVIDENCE UPLOAD FAILED] HTTP {resp.status_code}: {resp.text}")
        print(f"[OFFLINE BACKUP] Retaining local backup on disk: {file_display_name}")
        return False

    return False


def sync_offline_captures_backlog(backend_url: str,
                                  device_id: str,
                                  ulb_id: str,
                                  active_sid: int,
                                  session: requests.Session,
                                  save_dir: str) -> int:
    """Scans save_dir for offline backed-up .jpg captures and their .json sidecars.
    Uploads each to backend in chronological order and deletes the local backup
    upon confirmed HTTP 200/201 response.
    Returns the count of successfully synchronized captures.
    """
    if not os.path.exists(save_dir):
        return 0

    candidates = []
    try:
        with os.scandir(save_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    name_lower = entry.name.lower()
                    if name_lower.endswith(".jpg") or name_lower.endswith(".jpeg"):
                        candidates.append((entry.stat().st_mtime, entry.path))
    except Exception as ex:
        print(f"[BACKLOG SYNC] Error scanning {save_dir}: {ex}")
        return 0

    if not candidates:
        return 0

    # Sort oldest first (chronological replay)
    candidates.sort(key=lambda x: x[0])
    url = f"{backend_url.rstrip('/')}/api/evidence/upload"
    uploaded_count = 0

    for mtime, img_path in candidates:
        abs_path = os.path.abspath(img_path)
        with _in_flight_evidence_lock:
            if abs_path in _in_flight_evidence_files:
                continue
            _in_flight_evidence_files.add(abs_path)

        try:
            meta_path = os.path.splitext(img_path)[0] + ".json"
            meta_dict = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f_m:
                        meta_dict = json.load(f_m)
                except Exception as je:
                    print(f"[BACKLOG SYNC] Note reading sidecar {meta_path}: {je}")

            # Fallback metadata if sidecar missing
            if not meta_dict:
                utc_from_mtime = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()
                meta_dict = {
                    "captured_at":         utc_from_mtime,
                    "collection_event_id": 0,
                    "idempotency_key":     str(uuid.uuid4()),
                    "width":               1280,
                    "height":              720,
                    "compression_quality": 80,
                    "latitude":            0.0,
                    "longitude":           0.0,
                    "speed_kph":           0.0,
                    "motion_confidence":   0.95,
                }

            meta_dict["local_path"] = img_path
            meta_dict["meta_path"]  = meta_path if os.path.exists(meta_path) else None

            print(f"[BACKLOG SYNC] Uploading offline backup: {os.path.basename(img_path)}...")
            success = _upload_and_cleanup_evidence(
                session=session,
                url=url,
                backend_url=backend_url,
                device_id=device_id,
                ulb_id=ulb_id,
                active_sid=active_sid,
                item=meta_dict,
            )
            if success:
                uploaded_count += 1
            else:
                # Backend unavailable or failed — abort remaining backlog to avoid spamming
                print(f"[BACKLOG SYNC] Pausing sync; remaining offline captures preserved locally.")
                break
        finally:
            with _in_flight_evidence_lock:
                _in_flight_evidence_files.discard(abs_path)

    if uploaded_count > 0:
        print(f"[BACKLOG SYNC] ✔ Synchronized and deleted {uploaded_count} offline capture(s) from local storage.")
    return uploaded_count


def evidence_upload_worker(backend_url: str, device_id: str, ulb_id: str,
                            stop_event: threading.Event,
                            save_dir: str | None = None) -> None:
    """Consumes motion detection captures from upload_queue and uploads to backend.
    
    Image Storage Architecture:
      - All captures are initially staged on disk as a backup.
      - Upon confirmed backend upload (HTTP 200/201), the local files are deleted immediately.
      - If offline or unauthenticated, captures remain safely stored locally.
      - As soon as connectivity & authentication are confirmed, the offline backlog
        is uploaded in chronological order and deleted from local storage upon receipt.
    """
    from telemetry import upload_queue
    from config import CAPTURES_DIR

    target_save_dir = save_dir or CAPTURES_DIR
    url = f"{backend_url.rstrip('/')}/api/evidence/upload"

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    _last_auth_warn   = 0.0
    _last_backlog_chk = 0.0
    _BACKLOG_INTERVAL = 10.0   # seconds between offline backlog scans

    # Network back-off state (exponential, resets on any successful upload)
    _net_fail_count  = 0
    _backoff_until   = 0.0
    _MAX_BACKOFF_SEC = 30.0

    while not stop_event.is_set():
        active_sid = dynamic_session_info.get("sessionId", 0)
        is_auth    = dynamic_session_info.get("authenticated", False)

        # If not authenticated, do not consume items from upload_queue; let them stay in queue & disk
        if not is_auth or active_sid <= 0:
            now_t = time.time()
            if now_t - _last_auth_warn > 10.0:
                _last_auth_warn = now_t
                print(f"[AUTH HOLD] Evidence upload waiting: Device '{device_id}' is not authenticated in backend. Captures remain safely in local backup.")
            stop_event.wait(2.0)
            continue

        # 1. Process live queue items
        # Honour back-off window before attempting any upload
        now_t = time.time()
        if now_t < _backoff_until:
            stop_event.wait(min(1.0, _backoff_until - now_t))
            continue

        try:
            item = upload_queue.get(timeout=1.0)
        except queue.Empty:
            item = None

        if item is not None:
            local_path = item.get("local_path")
            abs_path   = os.path.abspath(local_path) if local_path else None
            if abs_path:
                with _in_flight_evidence_lock:
                    _in_flight_evidence_files.add(abs_path)

            try:
                success = _upload_and_cleanup_evidence(
                    session=session,
                    url=url,
                    backend_url=backend_url,
                    device_id=device_id,
                    ulb_id=ulb_id,
                    active_sid=active_sid,
                    item=item,
                )
                if success:
                    # Reset back-off on confirmed upload
                    _net_fail_count = 0
                    _backoff_until  = 0.0
                else:
                    # Upload failed (network or server error) — apply back-off
                    _net_fail_count += 1
                    backoff = min(_MAX_BACKOFF_SEC, 0.5 * (2 ** min(_net_fail_count - 1, 6)))
                    _backoff_until = time.time() + backoff
                    print(f"[EVIDENCE UPLOAD] Back-off #{_net_fail_count}: waiting {backoff:.0f}s before retry.")
                    # Rebuild session to clear stale connections
                    try:
                        session.close()
                    except Exception:
                        pass
                    session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=1)
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
            except Exception as e:
                print(f"[EVIDENCE UPLOAD ERROR] {e}")
            finally:
                if abs_path:
                    with _in_flight_evidence_lock:
                        _in_flight_evidence_files.discard(abs_path)
                upload_queue.task_done()

        # 2. Check offline backlog if queue is clear or interval elapsed
        now_t = time.time()
        if (upload_queue.empty() and (now_t - _last_backlog_chk >= _BACKLOG_INTERVAL)) or (_last_backlog_chk == 0.0):
            _last_backlog_chk = now_t
            try:
                sync_offline_captures_backlog(
                    backend_url=backend_url,
                    device_id=device_id,
                    ulb_id=ulb_id,
                    active_sid=active_sid,
                    session=session,
                    save_dir=target_save_dir,
                )
            except Exception as b_err:
                print(f"[BACKLOG SYNC ERROR] {b_err}")

            # Also sync offline litter captures backlog
            try:
                from config import LITTER_CAPTURES_DIR
                if os.path.exists(LITTER_CAPTURES_DIR):
                    sync_offline_captures_backlog(
                        backend_url=backend_url,
                        device_id=device_id,
                        ulb_id=ulb_id,
                        active_sid=active_sid,
                        session=session,
                        save_dir=LITTER_CAPTURES_DIR,
                    )
            except Exception as lb_err:
                print(f"[LITTER BACKLOG SYNC ERROR] {lb_err}")

    session.close()

