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
    "sessionId":  0,
    "ulbId":      DEFAULT_ULB_ID,
    "vehicleReg": DEFAULT_VEHICLE_ID,
    "deviceId":   "UNASSIGNED",
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

    # 5. Discover Active Session & Vehicle from API if exists
    try:
        r_sess = requests.get(f"{base}/api/officer/sessions?ulbId={resolved_ulb}&status=ACTIVE", timeout=3.0)
        if r_sess.status_code == 200:
            sessions = r_sess.json()
            if isinstance(sessions, list) and len(sessions) > 0:
                from telemetry import latest_sensor, hardware_state
                active_s = sessions[0]
                sid = active_s.get("operationalSessionId") or active_s.get("sessionId") or 0
                dynamic_session_info["sessionId"]  = sid
                dynamic_session_info["vehicleReg"] = active_s.get("vehicleRegistrationNumber") or dynamic_session_info["vehicleReg"]
                dynamic_session_info["ulbId"]      = active_s.get("ulbId") or resolved_ulb
                dynamic_session_info["deviceId"]   = active_s.get("deviceCode") or active_s.get("deviceId") or dynamic_session_info["deviceId"]
                latest_sensor["active_session_id"] = sid
                hardware_state["backend"]["active_session_id"] = sid
                print(f"[API METADATA] Existing Active Session detected: #{sid} (Vehicle: {dynamic_session_info['vehicleReg']})")
    except Exception as ex:
        print(f"[API METADATA NOTE] Active session sync note: {ex}")


def ensure_hardware_session(backend_url: str, device_code: str) -> int:
    """Dynamically activates or binds the hardware session in the DB for the identified device."""
    if not device_code or device_code in ("AUTO", "UNASSIGNED", "DISCONNECTED"):
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
                dynamic_session_info["sessionId"]  = sid
                dynamic_session_info["vehicleReg"] = (
                    sess.get("vehicleRegistrationNumber") or
                    (sess.get("vehicle") or {}).get("registrationNumber") or
                    dynamic_session_info["vehicleReg"]
                )
                dynamic_session_info["ulbId"]    = sess.get("ulbId") or (sess.get("ulb") or {}).get("ulbId") or dynamic_session_info["ulbId"]
                dynamic_session_info["deviceId"] = device_code
                latest_sensor["active_session_id"]             = sid
                hardware_state["backend"]["active_session_id"] = sid

                last_seq = sess.get("lastSequenceNumber") or sess.get("maxSequenceNumber") or 0
                if last_seq > latest_sensor.get("sequence", 0):
                    latest_sensor["sequence"] = int(last_seq)

                print(f"[SESSION BIND] Dynamic Telemetry Session #{sid} active for Vehicle '{dynamic_session_info['vehicleReg']}' (Device: {device_code}) [LastSeq: {latest_sensor['sequence']}]")
                return sid
    except Exception as ex:
        print(f"[SESSION BIND NOTE] Session creation note: {ex}")
    return dynamic_session_info.get("sessionId", 0)


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

    session            = _make_session()
    logged_first_ok    = False
    last_sent_frame_id = None

    # Error back-off state
    _err_count         = 0
    _last_err_log      = 0.0
    _ERR_LOG_INTERVAL  = 10.0   # seconds between repeated error messages
    _backoff_until     = 0.0    # epoch: don't attempt to send until this time

    while not stop_event.is_set():
        loop_start = time.time()

        # Honour back-off window (reconnect delay after repeated failures)
        if loop_start < _backoff_until:
            stop_event.wait(min(0.5, _backoff_until - loop_start))
            continue

        frame_to_send = None
        with _lock:
            candidate = _get_frame()
            if candidate is not None and id(candidate) != last_sent_frame_id:
                frame_to_send      = candidate
                last_sent_frame_id = id(candidate)

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
        now_time = time.time()
        active_dyn_sid = dynamic_session_info.get("sessionId") or latest_sensor.get("active_session_id") or 0
        if active_dyn_sid > 0:
            current_session_id = active_dyn_sid

        if not session_id_arg and not active_dyn_sid and (now_time - last_session_check) > 3.0:
            last_session_check = now_time
            try:
                s_resp = session.get(active_session_query_url, timeout=2.0)
                if s_resp.status_code == 200:
                    sessions_list = s_resp.json()
                    if sessions_list and isinstance(sessions_list, list) and len(sessions_list) > 0:
                        first_active = sessions_list[0]
                        sid = first_active.get("operationalSessionId") or first_active.get("sessionId") or 0
                        if sid != current_session_id and sid > 0:
                            current_session_id = sid
                            latest_sensor["active_session_id"] = sid
                            hardware_state["backend"]["active_session_id"] = sid
                            print(f"[EDGE SESSION] Automatically locked to active operational session #{sid}")
            except Exception:
                pass

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
# Evidence Upload Worker (POST /api/evidence/upload)
# ---------------------------------------------------------------------------
def evidence_upload_worker(backend_url: str, device_id: str, ulb_id: str,
                            stop_event: threading.Event) -> None:
    """Consumes motion detection captures from queue and uploads to backend with full GPS, session, and device metadata."""
    from telemetry import latest_sensor, hardware_state, upload_queue

    url = f"{backend_url.rstrip('/')}/api/evidence/upload"

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    while not stop_event.is_set():
        try:
            item = upload_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            jpeg_bytes           = item["jpeg_bytes"]
            captured_at          = item.get("captured_at") or datetime.datetime.now(datetime.timezone.utc).isoformat()
            collection_event_id  = item.get("collection_event_id", 0)
            idempotency_key      = item.get("idempotency_key") or str(uuid.uuid4())
            width                = item.get("width",  1280)
            height               = item.get("height", 720)
            compression_quality  = item.get("compression_quality", 80)

            lat = latest_sensor.get("lat")
            lon = latest_sensor.get("lon")
            if lat is None or lon is None or (abs(lat) < 0.001 and abs(lon) < 0.001):
                lat = latest_sensor.get("last_known_valid_lat") or 0.0
                lon = latest_sensor.get("last_known_valid_lon") or 0.0

            speed      = latest_sensor.get("speed") or 0.0
            active_sid = latest_sensor.get("active_session_id") or dynamic_session_info.get("sessionId") or 0

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
                "motionConfidence":   0.95,
                "sessionId":          active_sid,
                "deviceId":           device_id,
                "ulbId":              ulb_id,
            }

            resp = None
            for attempt in range(2):
                try:
                    resp = session.post(
                        url,
                        files={"file": ("evidence.jpg", jpeg_bytes, "image/jpeg")},
                        data=data,
                        timeout=15.0,
                    )
                    if resp.status_code in (200, 201):
                        break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as req_err:
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                    raise req_err

            if resp is not None and resp.status_code in (200, 201):
                res_json = resp.json()
                hardware_state["backend"]["upload_count"] += 1
                img_id  = res_json.get("evidenceImageId") or res_json.get("id") or "N/A"
                img_url = res_json.get("imageUrl") or f"{backend_url.rstrip('/')}/api/evidence/images/{img_id}"

                near_h, h_dist = geofence.get_nearest_house(lat, lon)
                house_tag = (
                    f"{near_h['id']} - {near_h['name']} (@ {h_dist:.1f}m)"
                    if (near_h and h_dist <= 25.0)
                    else "Road Corridor (Auto-allocated)"
                )

                raw_lat_val = latest_sensor.get("raw_gps_lat") or lat
                raw_lon_val = latest_sensor.get("raw_gps_lon") or lon

                print("\n" + "=" * 65)
                print(f"[FIELD LOG] 📸 EVIDENCE UPLOADED & STORED IN BACKEND")
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
                print("=" * 65 + "\n")
            elif resp is not None:
                print(f"[EVIDENCE UPLOAD FAILED] HTTP {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"[EVIDENCE UPLOAD ERROR] {e}")
        finally:
            upload_queue.task_done()

    session.close()
