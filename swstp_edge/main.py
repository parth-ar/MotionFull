"""
main.py — SWSTP Edge Gateway for Raspberry Pi 4B.

Single-device replacement for the original Arduino Uno + PC two-device stack.
Sensors (DS3231 RTC, MPU-6500 IMU, NavCast GNSS) are read natively from the Pi.

Run from a terminal:
    python3 main.py [--headless] [--backend-url URL] [--device-id ID] ...

All original CLI flags from webcam_motion_detect.py are preserved.
No systemd unit / auto-start — errors print live to stdout (testing phase).
"""

import argparse
import datetime
import os
import re
import sys
import threading
import time
import uuid

# Headless / systemd safety: prevent OpenCV Qt plugin from aborting when no display is present
if "--headless" in sys.argv or "DISPLAY" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import imutils
import numpy as np
import requests

# Ensure UTF-8 terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Project imports — add swstp_edge/ to path so sub-modules resolve correctly
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as _cfg
from config import (
    DIFF_THRESHOLD, MIN_CONTOUR_AREA, BG_ALPHA, WARMUP_FRAMES,
    SAVE_COOLDOWN_SEC, FRAME_SIZE, CAPTURES_DIR,
    DEFAULT_BACKEND_URL, DEFAULT_ULB_ID, DEFAULT_STREAM_FPS,
    ENABLE_GPS_FALLBACK, GNSS_FALLBACK_TIMEOUT_SEC,
)

import sensors.rtc  as rtc_sensor
import sensors.imu  as imu_sensor
import sensors.gnss as gnss_sensor
import sensors.leds as leds
import sensors.power as power_sensor
from sensors.rtc_sync import periodic_sync_loop

import telemetry as _telemetry
from telemetry import (
    latest_sensor, hardware_state,
    upload_queue, telemetry_queue,
)

from motion import (
    load_roi_polygon, save_roi_polygon,
    apply_polygon_roi_mask, on_mouse_roi,
    overlay_metadata, generate_virtual_video_frame,
    cleanup_old_local_captures, get_rtc_timestamp,
    active_polygon_roi, is_drawing_polygon, drawn_polygon_pts, camera_feed_active,
)
import motion as _motion

from network.uploader import (
    auto_detect_backend_url, sync_backend_metadata, ensure_hardware_session,
    live_frame_streamer, telemetry_streamer, evidence_upload_worker,
    dynamic_session_info,
)
from network.location_fallback import gps_fallback_worker

# ---------------------------------------------------------------------------
# Shared live stream frame (read by live_frame_streamer in a separate thread)
# ---------------------------------------------------------------------------
latest_frame_lock:  threading.Lock = threading.Lock()
# latest_stream_frame is a (BGR ndarray, monotonic write_time) tuple.
# The write_time lets live_frame_streamer detect new frames even when the
# numpy array is mutated in-place, avoiding the streaming-stall bug.
latest_stream_frame: tuple | None = None

RE_CLEAN_FILENAME = re.compile(r"[^\w]")

# ---------------------------------------------------------------------------
# Argument parser — preserves all flags from webcam_motion_detect.py
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SWSTP Edge Gateway (Raspberry Pi): Motion Detection, Telemetry & Live Video Streamer."
    )
    parser.add_argument("--source", "-s", default=None,
                        help="Video source (0 for default webcam, or path to MP4)")
    # --port and --baud are kept for CLI compatibility but have no effect on Pi
    # (the Arduino serial link no longer exists)
    parser.add_argument("--port", "-p", default="Pi-native",
                        help="[IGNORED on Pi] Legacy serial port flag — kept for CLI compatibility")
    parser.add_argument("--baud", "-b", type=int, default=115200,
                        help="[IGNORED on Pi] Legacy baud rate flag — kept for CLI compatibility")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL,
                        help="SWSTP Backend URL")
    parser.add_argument("--device-id", default=None,
                        help="Hardware Device Code (overrides device_config.json)")
    parser.add_argument("--ulb-id", default=DEFAULT_ULB_ID,
                        help="ULB ID")
    parser.add_argument("--session-id", type=int, default=0,
                        help="Operational Session ID (0 = auto-detect)")
    parser.add_argument("--fps-stream", type=float, default=DEFAULT_STREAM_FPS,
                        help="Live frame streaming FPS to backend")
    parser.add_argument("--save-dir", default=os.path.join(_HERE, "captures"),
                        help="Local directory to save motion captures")
    parser.add_argument("--headless", action="store_true",
                        help="Run without cv2.imshow GUI window")
    parser.add_argument("--no-gps-fallback", action="store_true",
                        help="Disable IP-based location fallback when GNSS has no fix")
    parser.add_argument("--gps-fallback-timeout", type=float, default=GNSS_FALLBACK_TIMEOUT_SEC,
                        help="Seconds without a GNSS fix before engaging IP fallback")
    parser.add_argument("--no-power-monitor", action="store_true",
                        help="Disable background power & low-battery monitoring thread")
    parser.add_argument("--simulate-low-batt-sec", type=float, default=0,
                        help="Simulate low battery alert after N seconds for testing")
    return parser



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global latest_stream_frame

    # Allow motion.py globals to be written via the module reference
    import motion as _motion_mod

    parser = build_parser()
    args, _ = parser.parse_known_args()

    # ── Resolve device identity ───────────────────────────────────────────
    device_id = (args.device_id or _cfg.load_device_id()).strip()
    if not device_id or device_id in ("", "UNPROVISIONED", "UNASSIGNED"):
        print("[INIT] WARNING: Device ID not set. Edit swstp_edge/device_config.json to provision.")
        print("[INIT]          Continuing with 'UNPROVISIONED' — backend session will not bind.")

    effective_ulb_id = args.ulb_id if (args.ulb_id and args.ulb_id != "AUTO") else DEFAULT_ULB_ID
    enable_fallback  = not args.no_gps_fallback

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Backend discovery & metadata sync ────────────────────────────────
    effective_backend_url = auto_detect_backend_url(args.backend_url)
    sync_backend_metadata(effective_backend_url, effective_ulb_id)

    # ── Session bind ─────────────────────────────────────────────────────
    effective_session_id = args.session_id
    if device_id and device_id not in ("UNPROVISIONED", "UNASSIGNED"):
        sid = ensure_hardware_session(effective_backend_url, device_id)
        if sid > 0:
            effective_session_id = sid
    dynamic_session_info["deviceId"] = device_id
    latest_sensor["deviceId"]        = device_id

    # ── Sensor initialisation ────────────────────────────────────────────
    print("\n=== SWSTP Pi EDGE SENSOR INIT ===")
    rtc_ok   = rtc_sensor.init()
    imu_ok   = imu_sensor.init()
    gnss_ok  = gnss_sensor.init()   # starts background NavCast TCP reader thread

    # LEDs
    try:
        leds.init()
        if args.headless:
            leds.set_headless_mode(True)
            # Yellow will go solid once the main loop is ready (after camera init)
    except Exception as exc:
        print(f"[LEDS] init skipped: {exc}")

    # ── Periodic RTC re-sync thread (every hour) ────────────────────────
    resync_interval = getattr(_cfg, "RTC_RESYNC_INTERVAL_HOURS", 1.0)
    t_rtc_resync = threading.Thread(
        target=periodic_sync_loop,
        args=(resync_interval,),   # recalibrate timing every hour
        daemon=True, name="rtc-resync",
    )
    t_rtc_resync.start()

    tz_name = getattr(_cfg, "load_timezone", lambda: "Asia/Kolkata")()
    print("\n========================================================")
    print(" SWSTP EDGE GATEWAY INITIALIZING (Raspberry Pi 4B)")
    print(f" Backend Endpoint: {effective_backend_url}")
    print(f" Device ID:        {device_id}")
    print(f" ULB ID:           {effective_ulb_id}")
    print(f" Session ID:       {effective_session_id if effective_session_id else 'DYNAMIC HARDWARE BIND'}")
    print(f" Timezone:         {tz_name} (IST, UTC+05:30)")
    print(f" Motion Captures:  {os.path.abspath(args.save_dir)}")
    print(f" RTC:              {'OK (kernel + DS3231)' if rtc_ok else 'FALLBACK (system clock)'} [Source: {rtc_sensor.sync_source.upper()}]")
    print(f" IMU:              {'OK (MPU-6500 @ 0x69)' if imu_ok else 'FAULT'}")
    print(f" GNSS:             NavCast TCP reader started ({_cfg.NAVCAST_HOST}:{_cfg.NAVCAST_PORT})")
    print("========================================================\n")

    # ── Camera / video source ────────────────────────────────────────────
    VIDEO_SOURCE = 0
    source = (
        int(args.source)
        if (args.source and args.source.isdigit())
        else (args.source if args.source else VIDEO_SOURCE)
    )
    cap = None
    use_synthetic_video = False

    if isinstance(source, int):
        for idx in [source, 0, 1, 2]:
            try:
                # Linux/Pi: V4L2 backend (no CAP_DSHOW)
                c = cv2.VideoCapture(idx)
                if c.isOpened():
                    cap    = c
                    source = idx
                    break
                c.release()
            except Exception:
                pass
    elif isinstance(source, str):
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print("[CAMERA] No hardware webcam found — operating in synthetic video simulation mode.")
        use_synthetic_video = True
        cam_w, cam_h, fps = 640, 360, 30.0
        hardware_state["camera"] = {"detected": True, "source": "Virtual Edge Stream",
                                     "resolution": f"{cam_w}x{cam_h}", "fps": fps}
    else:
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        hardware_state["camera"] = {"detected": True, "source": source,
                                     "resolution": f"{cam_w}x{cam_h}", "fps": fps}
        print(f"[CAMERA] Active: {cam_w}x{cam_h} @ {fps:.1f} FPS")

    stop_event = threading.Event()

    # ── Power & Battery Supervisor ────────────────────────────────────────
    if not args.no_power_monitor:
        power_sensor.init(main_shutdown_callback=lambda: stop_event.set())

    if args.simulate_low_batt_sec > 0:
        print(f"[POWER TEST] Will simulate low battery shutdown in {args.simulate_low_batt_sec:.1f}s...")
        def _sim_low_batt():
            time.sleep(args.simulate_low_batt_sec)
            power_sensor.initiate_emergency_shutdown("SIMULATED_TEST_TIMER")
        threading.Thread(target=_sim_low_batt, daemon=True, name="sim-low-batt").start()

    # ── Worker threads ────────────────────────────────────────────────────
    # 1. 20 Hz sensor → telemetry_queue thread
    t_telemetry_loop = _telemetry.start(device_id)

    # 2. Live frame streamer  (POST /api/camera/{deviceId}/frame)
    # Pass the lock and a lambda getter so uploader.py never needs to import main.
    def _get_latest_frame():
        return latest_stream_frame


    t_frame_streamer = threading.Thread(
        target=live_frame_streamer,
        args=(effective_backend_url, device_id, args.fps_stream, stop_event,
              latest_frame_lock, _get_latest_frame),
        daemon=True, name="frame-streamer",
    )
    t_frame_streamer.start()

    # 3. Telemetry batch ingestion  (POST /api/telemetry/ingest-batch)
    t_telemetry_http = threading.Thread(
        target=telemetry_streamer,
        args=(effective_backend_url, effective_session_id, effective_ulb_id, stop_event),
        daemon=True, name="telemetry-http",
    )
    t_telemetry_http.start()

    # 4. Evidence upload worker  (POST /api/evidence/upload)
    t_evidence = threading.Thread(
        target=evidence_upload_worker,
        args=(effective_backend_url, device_id, effective_ulb_id, stop_event),
        daemon=True, name="evidence-upload",
    )
    t_evidence.start()

    # 5. GNSS → IP geolocation fallback worker
    t_gps_fallback = threading.Thread(
        target=gps_fallback_worker,
        args=(stop_event, enable_fallback, args.gps_fallback_timeout),
        daemon=True, name="gps-fallback",
    )
    t_gps_fallback.start()

    # ── ROI polygon ───────────────────────────────────────────────────────
    load_roi_polygon()

    # ── GUI window ────────────────────────────────────────────────────────
    WIN_TITLE = "SWSTP Motion & Telemetry Gateway (Pi)"
    if not args.headless:
        cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TITLE, 960, 540)
        cv2.setMouseCallback(WIN_TITLE, on_mouse_roi, {"width": cam_w, "height": cam_h})

    delay       = max(1, int(1000 / (fps if (fps and 0 < fps < 120) else 30)))
    is_file     = isinstance(source, str)
    LOOP_VIDEO  = True
    bg_model    = None
    frame_count = 0
    last_save_time = 0.0
    saved_count    = 0
    last_known_frame = None

    # ── Stop-collection event state ───────────────────────────────────────
    # A stop event begins when motion is detected AND vehicle speed < 5 km/h.
    # It ends when the vehicle speeds up or motion is absent for STOP_IDLE_TIMEOUT s.
    STOP_IDLE_TIMEOUT    = 10.0      # seconds of no-motion before auto-closing event
    MOTION_SPEED_GATE    = 5.0       # km/h threshold
    _stop_ev_active      = False
    _stop_ev_start       = 0.0       # monotonic timestamp of event start
    _stop_ev_lat         = None
    _stop_ev_lon         = None
    _stop_last_motion    = 0.0       # monotonic time of last frame with motion

    print("[INIT] Edge Gateway running.")
    print("  Controls: [t] Toggle Camera | [r] Plot Area of Interest | [c] Clear Pointers | [q] Quit\n")

    # Signal LED driver that system is operational and ready to capture
    try:
        leds.set_system_ready()
    except Exception:
        pass

    try:
        while not stop_event.is_set():

            # ── Camera feed paused state ──────────────────────────────────
            if not _motion_mod.camera_feed_active:
                if last_known_frame is not None:
                    paused_frame = cv2.convertScaleAbs(last_known_frame.copy(), alpha=0.35, beta=0)
                else:
                    paused_frame = np.zeros((cam_h or 360, cam_w or 640, 3), dtype=np.uint8)

                ph, pw = paused_frame.shape[:2]
                banner_y = ph // 2
                cv2.rectangle(paused_frame, (0, max(0, banner_y - 45)), (pw, min(ph, banner_y + 45)), (20, 20, 20), -1)
                cv2.rectangle(paused_frame, (0, max(0, banner_y - 45)), (pw, min(ph, banner_y + 45)), (0, 165, 255), 2)
                cv2.putText(paused_frame, "CAMERA FEED PAUSED",
                            (max(10, pw // 2 - 170), banner_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(paused_frame, "Press 't' on keyboard to toggle camera feed ON",
                            (max(10, pw // 2 - 210), banner_y + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                display_frame = overlay_metadata(paused_frame)

                with latest_frame_lock:
                    latest_stream_frame = (display_frame, time.monotonic())

                if not args.headless:
                    cv2.imshow(WIN_TITLE, display_frame)
                    k = cv2.waitKey(delay) & 0xFF
                    if k == ord('q'):
                        break
                    elif k == ord('t'):
                        _motion_mod.camera_feed_active = True
                        print("\n[CAMERA] Camera feed ACTIVATED / ON.")
                else:
                    time.sleep(0.1)
                continue

            # ── Frame acquisition ─────────────────────────────────────────
            if use_synthetic_video:
                frame   = generate_virtual_video_frame(cam_w, cam_h, frame_count)
                success = True
                time.sleep(1.0 / fps)
            else:
                success, frame = cap.read()
                if not success:
                    if is_file and LOOP_VIDEO:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        bg_model    = None
                        frame_count = 0
                        continue
                    else:
                        break

            orig_frame   = frame.copy()
            last_known_frame = orig_frame.copy()
            orig_h, orig_w   = orig_frame.shape[:2]

            # ── Downsample for motion processing (320×180) ────────────────
            if FRAME_SIZE is not None:
                proc_w, proc_h = FRAME_SIZE
                proc_frame = cv2.resize(frame, (proc_w, proc_h))
                scale_x    = orig_w / float(proc_w)
                scale_y    = orig_h / float(proc_h)
            else:
                proc_frame   = frame
                proc_w, proc_h = orig_w, orig_h
                scale_x, scale_y = 1.0, 1.0

            gray_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            gray_blur  = cv2.GaussianBlur(gray_frame, (9, 9), 0)

            if bg_model is None:
                bg_model    = gray_blur.astype(np.float32)
                frame_count = 1
                continue

            cv2.accumulateWeighted(gray_blur, bg_model, BG_ALPHA)
            frame_count += 1

            # ── Warm-up ───────────────────────────────────────────────────
            if frame_count <= WARMUP_FRAMES:
                display_frame = overlay_metadata(orig_frame.copy())
                with latest_frame_lock:
                    latest_stream_frame = (display_frame, time.monotonic())
                if not args.headless:
                    cv2.imshow(WIN_TITLE, display_frame)
                    k = cv2.waitKey(delay) & 0xFF
                    if k == ord('q'):
                        break
                    elif k == ord('t'):
                        _motion_mod.camera_feed_active = not _motion_mod.camera_feed_active
                        print(f"\n[CAMERA] Camera feed {'ON' if _motion_mod.camera_feed_active else 'PAUSED'}.")
                continue

            # ── Background subtraction ────────────────────────────────────
            bg_uint8 = cv2.convertScaleAbs(bg_model)
            diff     = cv2.absdiff(bg_uint8, gray_blur)
            _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Apply Polygon Area of Interest mask
            roi_pts = _motion_mod.active_polygon_roi
            thresh  = apply_polygon_roi_mask(thresh, roi_pts, FRAME_SIZE[0], FRAME_SIZE[1])
            dilated = cv2.dilate(thresh, None, iterations=1)

            cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            motion_detected = False
            box_thickness   = max(2, int(min(scale_x, scale_y)))

            for c in cnts:
                if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(c)
                orig_x, orig_y   = int(x * scale_x), int(y * scale_y)
                orig_bw, orig_bh = int(w * scale_x), int(h * scale_y)
                cv2.rectangle(orig_frame,
                              (orig_x, orig_y),
                              (orig_x + orig_bw, orig_y + orig_bh),
                              (0, 255, 0), box_thickness)

            # ── Draw Area of Interest polygon ─────────────────────────────
            disp_pts = []
            for p in roi_pts:
                px = int(p[0] * orig_w) if p[0] <= 1.0 else int(p[0])
                py = int(p[1] * orig_h) if p[1] <= 1.0 else int(p[1])
                disp_pts.append([px, py])

            if len(disp_pts) >= 3 and not _motion_mod.is_drawing_polygon:
                poly_color = (0, 0, 255) if motion_detected else (0, 255, 200)
                cv2.polylines(orig_frame, [np.array(disp_pts, dtype=np.int32)],
                              isClosed=True, color=poly_color, thickness=2)
                for pt in disp_pts:
                    cv2.circle(orig_frame, tuple(pt), 4, (0, 255, 255), -1)
                cv2.putText(orig_frame,
                            f"AREA OF INTEREST ({len(disp_pts)} pts, press 'r' to re-plot)",
                            (disp_pts[0][0] + 5, max(20, disp_pts[0][1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, poly_color, 1, cv2.LINE_AA)

            # ── In-progress polygon drawing overlay ───────────────────────
            if _motion_mod.is_drawing_polygon:
                prog_pts = [
                    [int(p[0] * orig_w), int(p[1] * orig_h)]
                    for p in _motion_mod.drawn_polygon_pts
                ]
                for idx, pt in enumerate(prog_pts):
                    pt_color = (0, 255, 0) if (idx == 0 and len(prog_pts) >= 3) else (0, 165, 255)
                    cv2.circle(orig_frame, tuple(pt), 6, pt_color, -1)
                    cv2.circle(orig_frame, tuple(pt), 10, (255, 255, 255), 1)
                    cv2.line(orig_frame, (pt[0] - 12, pt[1]), (pt[0] + 12, pt[1]), (0, 255, 255), 1)
                    cv2.line(orig_frame, (pt[0], pt[1] - 12), (pt[0], pt[1] + 12), (0, 255, 255), 1)
                    cv2.putText(orig_frame, f"P{idx+1}", (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if len(prog_pts) >= 2:
                    cv2.polylines(orig_frame, [np.array(prog_pts, dtype=np.int32)],
                                  isClosed=False, color=(0, 255, 255), thickness=2)
                    if len(prog_pts) >= 3:
                        cv2.line(orig_frame, tuple(prog_pts[-1]), tuple(prog_pts[0]), (0, 200, 100), 1, cv2.LINE_AA)
                        cv2.circle(orig_frame, tuple(prog_pts[0]), 14, (0, 255, 0), 2)
                        cv2.putText(orig_frame, "P1 (Snap/Close)", (prog_pts[0][0] + 16, prog_pts[0][1] + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(orig_frame,
                            f"PLOTTING AREA OF INTEREST: Click to place pointers ({len(prog_pts)} set) | Press 'r' again when done to connect & save",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

            display_frame = overlay_metadata(orig_frame.copy())

            # Update live stream buffer (tuple with write timestamp for stall-free streaming)
            with latest_frame_lock:
                latest_stream_frame = (display_frame, time.monotonic())

            # ── Stop-collection event state machine ───────────────────────
            now_mono       = time.monotonic()
            vehicle_speed  = latest_sensor.get("speed") or 0.0
            is_vehicle_stopped = vehicle_speed < MOTION_SPEED_GATE

            if motion_detected and is_vehicle_stopped:
                # Keep motion timestamp alive
                _stop_last_motion = now_mono
                # Open a new stop event if not already active
                if not _stop_ev_active:
                    _stop_ev_active = True
                    _stop_ev_start  = now_mono
                    _stop_ev_lat    = latest_sensor.get("lat")
                    _stop_ev_lon    = latest_sensor.get("lon")
                    print(f"[STOP EVENT] Collection event started "
                          f"@ ({_stop_ev_lat}, {_stop_ev_lon}) "
                          f"| Speed: {vehicle_speed:.1f} km/h")

            if _stop_ev_active:
                # Close event if vehicle sped up OR motion idle too long
                idle_secs = now_mono - _stop_last_motion
                if (not is_vehicle_stopped) or (idle_secs > STOP_IDLE_TIMEOUT):
                    duration_sec = now_mono - _stop_ev_start
                    reason = "speed > 5 km/h" if not is_vehicle_stopped else "motion idle timeout"
                    print(f"[STOP EVENT] Collection event ended — "
                          f"Duration: {duration_sec:.1f}s | Reason: {reason}")
                    _stop_ev_active = False

            # Current event duration (0 if no active event)
            current_event_duration = (now_mono - _stop_ev_start) if _stop_ev_active else 0.0

            # ── Motion capture & upload ───────────────────────────────────────
            if motion_detected:
                text_scale = max(0.5, (orig_h / 480.0) * 0.5)
                cv2.putText(display_frame, "MOTION DETECTED",
                            (10, int(55 * text_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                            (0, 0, 255), max(1, int(orig_h / 360)), cv2.LINE_AA)

                # ── Speed gate: only capture when vehicle is stopped ─────────
                if not is_vehicle_stopped:
                    # Vehicle moving too fast — skip this capture
                    pass
                elif now_mono - last_save_time >= SAVE_COOLDOWN_SEC:
                    saved_count    += 1
                    last_save_time  = now_mono

                    _, enc_evidence = cv2.imencode('.jpg', display_frame,
                                                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    evidence_bytes = enc_evidence.tobytes()

                    utc_iso  = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    rtc_ts   = get_rtc_timestamp() or utc_iso
                    clean_ts = RE_CLEAN_FILENAME.sub("_", rtc_ts)
                    local_path = os.path.join(args.save_dir,
                                              f"motion_RTC_{clean_ts}_{saved_count:04d}.jpg")
                    with open(local_path, "wb") as f:
                        f.write(evidence_bytes)

                    try:
                        upload_queue.put_nowait({
                            "jpeg_bytes":           evidence_bytes,
                            "captured_at":          utc_iso,
                            "collection_event_id":  0,
                            "idempotency_key":      str(uuid.uuid4()),
                            "width":                orig_w,
                            "height":               orig_h,
                            "compression_quality":  80,
                            "stop_duration_sec":    round(current_event_duration, 2),
                            "vehicle_speed_kmh":    round(vehicle_speed, 2),
                        })
                    except Exception:
                        pass

                    print(f"[CAPTURE #{saved_count}] Saved: {local_path} "
                          f"| Event duration: {current_event_duration:.1f}s "
                          f"| Speed: {vehicle_speed:.1f} km/h "
                          f"| Queued for backend upload.")
                    cleanup_old_local_captures(args.save_dir, retention_days=3)

                    # ── Headless LED: blink yellow 3× to confirm snap ────────
                    try:
                        leds.notify_motion_snap()
                    except Exception:
                        pass

            # ── Display & keyboard handling ───────────────────────────────
            if not args.headless:
                cv2.imshow(WIN_TITLE, display_frame)
                k = cv2.waitKey(delay) & 0xFF

                if k == ord('q'):
                    break
                elif k == ord('t'):
                    _motion_mod.camera_feed_active = not _motion_mod.camera_feed_active
                    print(f"\n[CAMERA] Camera feed {'ACTIVATED / ON' if _motion_mod.camera_feed_active else 'PAUSED / OFF'}.")
                elif k == ord('r'):
                    if not _motion_mod.is_drawing_polygon:
                        _motion_mod.is_drawing_polygon = True
                        _motion_mod.drawn_polygon_pts  = []
                        print("\n[ROI PLOT] Pointer selection mode ACTIVE:")
                        print("  1. Left-click on video to place pointers (P1, P2, P3...).")
                        print("  2. When all points are placed, press 'r' again to connect points and save polygon.\n")
                    else:
                        if len(_motion_mod.drawn_polygon_pts) >= 3:
                            _motion_mod.active_polygon_roi = _motion_mod.drawn_polygon_pts.copy()
                            save_roi_polygon(_motion_mod.active_polygon_roi)
                            _motion_mod.is_drawing_polygon = False
                            _motion_mod.drawn_polygon_pts  = []
                            print(f"\n[ROI PLOT] Connected {len(_motion_mod.active_polygon_roi)} pointers! Area of interest set up and saved persistently.\n")
                        else:
                            print(f"\n[ROI PLOT] Need at least 3 pointers (currently {len(_motion_mod.drawn_polygon_pts)}). Click on video to add more, or press 'c' to clear.\n")
                elif k == ord('c'):
                    if _motion_mod.is_drawing_polygon:
                        _motion_mod.drawn_polygon_pts = []
                        print("\n[ROI PLOT] Cleared temporary pointers.")
                    else:
                        print("\n[ROI PLOT] Not currently plotting.")

    finally:
        stop_event.set()
        # Signal fault LED before closing (yellow off, red blink briefly)
        try:
            leds.set_fault("Edge gateway shutting down")
            time.sleep(0.5)   # brief visible fault indication
        except Exception:
            pass
        power_sensor.stop()
        _telemetry.stop()
        gnss_sensor.stop()
        try:
            leds.close()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"\n[EDGE GATEWAY STOPPED] Total captures: {saved_count}")



if __name__ == "__main__":
    main()
