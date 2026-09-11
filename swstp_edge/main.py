"""
main.py — SWSTP Unified Edge Gateway for Raspberry Pi 4B.

Single executable entry point that runs BOTH:
  1. Motion Detection  — background-subtraction based, captures at vehicle stops.
  2. Litter Detection  — YOLO inference triggered every 10 m of GNSS travel.

Camera access: ONE VideoCapture object opened here; the litter engine receives
               frame copies directly — no second camera handle is ever opened.

Run as a service:
    python3 main.py --headless

All original motion-detection CLI flags are preserved.
"""

import argparse
import datetime
import json
import os
import re
import sys
import threading
import time
import uuid

# Headless / systemd safety
if "--headless" in sys.argv or "DISPLAY" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

os.environ["OPENCV_LOG_LEVEL"]              = "ERROR"
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "V4L2"

import cv2
if hasattr(cv2, "setLogLevel"):
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass
import imutils
import numpy as np
import requests

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
# Project path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as _cfg
from config import (
    DIFF_THRESHOLD, MIN_CONTOUR_AREA, BG_ALPHA, WARMUP_FRAMES,
    SAVE_COOLDOWN_SEC, FRAME_SIZE, CAPTURES_DIR, LITTER_CAPTURES_DIR,
    DEFAULT_BACKEND_URL, DEFAULT_ULB_ID, DEFAULT_STREAM_FPS,
    ENABLE_GPS_FALLBACK, GNSS_FALLBACK_TIMEOUT_SEC,
    LITTER_WEIGHTS, LITTER_DISTANCE_INTERVAL_M,
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
    overlay_metadata, run_roi_setup_phase,
    cleanup_old_local_captures, get_rtc_timestamp,
    active_polygon_roi, is_drawing_polygon, drawn_polygon_pts, camera_feed_active,
)
import motion as _motion

from litter_engine import LitterEngine, cleanup_old_litter_captures

from network.uploader import (
    auto_detect_backend_url, sync_backend_metadata, ensure_hardware_session,
    live_frame_streamer, telemetry_streamer, evidence_upload_worker,
    dynamic_session_info,
)
from stop_detector import VehicleStopDetector
from network.location_fallback import gps_fallback_worker

# ---------------------------------------------------------------------------
# Shared live stream frame
# ---------------------------------------------------------------------------
latest_frame_lock:  threading.Lock = threading.Lock()
latest_stream_frame: tuple | None = None

RE_CLEAN_FILENAME = re.compile(r"[^\w]")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SWSTP Unified Edge Gateway (Raspberry Pi 4B): "
                    "Motion Detection + Litter Detection + Telemetry + Live Stream."
    )
    parser.add_argument("--source", "-s", default=None,
                        help="Video source (0 for default webcam, or path to MP4)")
    parser.add_argument("--port", "-p", default="Pi-native",
                        help="[IGNORED on Pi] Legacy serial port flag")
    parser.add_argument("--baud", "-b", type=int, default=115200,
                        help="[IGNORED on Pi] Legacy baud rate flag")
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
    parser.add_argument("--save-dir", default=os.path.join(_HERE, "captures", "motion"),
                        help="Local directory for motion captures")
    parser.add_argument("--headless", action="store_true",
                        help="Run without cv2.imshow GUI window")
    parser.add_argument("--no-gps-fallback", action="store_true",
                        help="Disable IP-based location fallback when GNSS has no fix")
    parser.add_argument("--gps-fallback-timeout", type=float, default=GNSS_FALLBACK_TIMEOUT_SEC,
                        help="Seconds without GNSS fix before engaging IP fallback")
    parser.add_argument("--no-power-monitor", action="store_true",
                        help="Disable background power & low-battery monitoring")
    parser.add_argument("--simulate-low-batt-sec", type=float, default=0,
                        help="Simulate low battery alert after N seconds (testing)")
    # Litter detection overrides
    parser.add_argument("--litter-weights", default=LITTER_WEIGHTS,
                        help=f"Path to YOLO ONNX weights for litter detection (default: {LITTER_WEIGHTS})")
    parser.add_argument("--litter-distance", type=float, default=LITTER_DISTANCE_INTERVAL_M,
                        help=f"GNSS distance (m) between litter detection triggers (default: {LITTER_DISTANCE_INTERVAL_M}m)")
    parser.add_argument("--litter-conf", type=float, default=_cfg.LITTER_CONF_THRESHOLD,
                        help=f"YOLO confidence threshold for litter (default: {_cfg.LITTER_CONF_THRESHOLD})")
    parser.add_argument("--no-litter", action="store_true",
                        help="Disable litter detection (run motion-only mode)")
    return parser


# ---------------------------------------------------------------------------
# Camera & Video Port Scanning Helpers  (unchanged from swstp_edge)
# ---------------------------------------------------------------------------
def is_v4l2_capture_device(dev_idx: int) -> bool:
    sys_path = f"/sys/class/video4linux/video{dev_idx}"
    if not os.path.exists(sys_path):
        return False
    name_file = os.path.join(sys_path, "name")
    if os.path.exists(name_file):
        try:
            with open(name_file, "r", encoding="utf-8", errors="ignore") as f:
                name = f.read().strip().lower()
            if "metadata" in name or "bcm2835-codec" in name or "bcm2835-isp" in name:
                return False
        except Exception:
            pass
    return True


def probe_video_source(src):
    try:
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            dev_idx = int(src)
            if sys.platform.startswith("linux"):
                dev_node = f"/dev/video{dev_idx}"
                if not os.path.exists(dev_node) or not os.access(dev_node, os.R_OK):
                    return None, 0, 0, 0
                if not is_v4l2_capture_device(dev_idx):
                    return None, 0, 0, 0
                c = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            else:
                c = cv2.VideoCapture(dev_idx)
        elif isinstance(src, str) and src.startswith("/dev/video"):
            dev_name = os.path.basename(src)
            idx_str  = dev_name.replace("video", "")
            if idx_str.isdigit() and sys.platform.startswith("linux"):
                dev_idx = int(idx_str)
                if not os.path.exists(src) or not os.access(src, os.R_OK):
                    return None, 0, 0, 0
                if not is_v4l2_capture_device(dev_idx):
                    return None, 0, 0, 0
                c = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            else:
                c = cv2.VideoCapture(src)
        else:
            c = cv2.VideoCapture(src)

        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            try:
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ret, test_frame = c.read()
            if ret and test_frame is not None and test_frame.size > 0:
                w      = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))  or test_frame.shape[1]
                h      = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) or test_frame.shape[0]
                cam_fps = c.get(cv2.CAP_PROP_FPS) or 30.0
                if cam_fps <= 0 or cam_fps > 120:
                    cam_fps = 30.0
                return c, w, h, cam_fps
            c.release()
    except Exception:
        pass
    return None, 0, 0, 0


def get_candidate_video_ports(preferred_source=None):
    candidates = []
    if preferred_source is not None and preferred_source != "":
        try:
            candidates.append(int(preferred_source) if str(preferred_source).isdigit() else preferred_source)
        except Exception:
            candidates.append(preferred_source)
    if sys.platform.startswith("linux"):
        import glob
        found_devs = []
        for p in sorted(glob.glob("/dev/video*")):
            dev_name = os.path.basename(p)
            idx_str  = dev_name.replace("video", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                if is_v4l2_capture_device(idx):
                    found_devs.append(idx)
        for d in found_devs:
            if d not in candidates:
                candidates.append(d)
        if not candidates:
            candidates = [0, 1]
    else:
        for idx in [0, 1, 2]:
            if idx not in candidates:
                candidates.append(idx)
    return candidates


def scan_for_camera(candidates):
    for cand in candidates:
        c, w, h, cam_fps = probe_video_source(cand)
        if c is not None:
            return c, cand, w, h, cam_fps
    return None, None, 0, 0, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global latest_stream_frame

    import motion as _motion_mod

    parser = build_parser()
    args, _ = parser.parse_known_args()

    # ── Resolve device identity ──────────────────────────────────────────
    device_id = (args.device_id or _cfg.load_device_id()).strip()
    if not device_id or device_id in ("", "UNPROVISIONED", "UNASSIGNED"):
        print("[INIT] WARNING: Device ID not set. Edit device_config.json to provision.")

    effective_ulb_id = args.ulb_id if (args.ulb_id and args.ulb_id != "AUTO") else DEFAULT_ULB_ID
    enable_fallback  = not args.no_gps_fallback

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(LITTER_CAPTURES_DIR, exist_ok=True)

    # ── Backend discovery & metadata sync ───────────────────────────────
    effective_backend_url = auto_detect_backend_url(args.backend_url)
    sync_backend_metadata(effective_backend_url, effective_ulb_id)

    # ── Session bind ─────────────────────────────────────────────────────
    effective_session_id = args.session_id
    if device_id and device_id not in ("UNPROVISIONED", "UNASSIGNED"):
        sid = ensure_hardware_session(effective_backend_url, device_id)
        if sid > 0:
            effective_session_id = sid
        else:
            print(f"\n[CRITICAL AUTH] Device '{device_id}' failed backend authentication!")
            print("[CRITICAL AUTH] Telemetry, evidence images, and live stream uploads are BLOCKED.\n")
    else:
        print(f"\n[CRITICAL AUTH] Device ID '{device_id}' is unprovisioned. Backend uploads are BLOCKED.\n")
    dynamic_session_info["deviceId"] = device_id
    latest_sensor["deviceId"]        = device_id

    # ── Sensor initialisation ─────────────────────────────────────────────
    print("\n=== SWSTP UNIFIED SENSOR INIT ===")
    rtc_ok  = rtc_sensor.init()
    imu_ok  = imu_sensor.init()
    gnss_ok = gnss_sensor.init()

    try:
        leds.init()
        if args.headless:
            leds.set_headless_mode(True)
    except Exception as exc:
        print(f"[LEDS] init skipped: {exc}")

    # ── Periodic RTC re-sync thread ──────────────────────────────────────
    resync_interval = getattr(_cfg, "RTC_RESYNC_INTERVAL_HOURS", 1.0)
    t_rtc_resync = threading.Thread(
        target=periodic_sync_loop,
        args=(resync_interval,),
        daemon=True, name="rtc-resync",
    )
    t_rtc_resync.start()

    tz_name = getattr(_cfg, "load_timezone", lambda: "Asia/Kolkata")()
    print("\n========================================================")
    print(" SWSTP UNIFIED EDGE GATEWAY INITIALIZING (Raspberry Pi 4B)")
    print(f" Backend:          {effective_backend_url}")
    print(f" Device ID:        {device_id}")
    print(f" ULB ID:           {effective_ulb_id}")
    print(f" Session ID:       {effective_session_id if effective_session_id else 'DYNAMIC HARDWARE BIND'}")
    print(f" Timezone:         {tz_name} (IST, UTC+05:30)")
    print(f" Motion Captures:  {os.path.abspath(args.save_dir)}")
    print(f" Litter Captures:  {os.path.abspath(LITTER_CAPTURES_DIR)}")
    print(f" Litter Weights:   {args.litter_weights}")
    print(f" Litter Trigger:   every {args.litter_distance:.1f}m (GNSS) | 30s fallback")
    print(f" Litter Detection: {'ENABLED' if not args.no_litter else 'DISABLED (--no-litter)'}")
    print(f" RTC:              {'OK (DS3231)' if rtc_ok else 'FALLBACK (system clock)'} [{rtc_sensor.sync_source.upper()}]")
    print(f" IMU:              {'OK (MPU-6500 @ 0x69)' if imu_ok else 'FAULT'}")
    print(f" GNSS:             NavCast TCP reader started ({_cfg.NAVCAST_HOST}:{_cfg.NAVCAST_PORT})")
    print("========================================================\n")

    # ── Camera / video source ─────────────────────────────────────────────
    VIDEO_SOURCE = 0
    initial_source = (
        int(args.source)
        if (args.source and args.source.isdigit())
        else (args.source if args.source else VIDEO_SOURCE)
    )
    candidate_ports = get_candidate_video_ports(initial_source)
    cap, source, cam_w, cam_h, fps = scan_for_camera(candidate_ports)

    if cap is not None:
        hardware_state["camera"] = {
            "detected": True, "source": source,
            "resolution": f"{cam_w}x{cam_h}", "fps": fps
        }
        print(f"[CAMERA] Active: Port {source} ({cam_w}x{cam_h} @ {fps:.1f} FPS)")
    else:
        source = initial_source
        cam_w, cam_h, fps = 640, 360, 30.0
        hardware_state["camera"] = {
            "detected": False, "source": None, "resolution": "N/A", "fps": 0
        }
        print(f"[CAMERA] No webcam found on {candidate_ports}. Motion on hold. Triple LED blinks...")
        try:
            leds.set_camera_scanning(True)
        except Exception:
            pass

    stop_event = threading.Event()

    # ── Power & Battery Supervisor ─────────────────────────────────────────
    if not args.no_power_monitor:
        power_sensor.init(main_shutdown_callback=lambda: stop_event.set())

    if args.simulate_low_batt_sec > 0:
        print(f"[POWER TEST] Simulating low battery in {args.simulate_low_batt_sec:.1f}s...")
        def _sim_low_batt():
            time.sleep(args.simulate_low_batt_sec)
            power_sensor.initiate_emergency_shutdown("SIMULATED_TEST_TIMER")
        threading.Thread(target=_sim_low_batt, daemon=True, name="sim-low-batt").start()

    # ── Worker threads ────────────────────────────────────────────────────
    t_telemetry_loop = _telemetry.start(device_id)

    def _get_latest_frame():
        return latest_stream_frame

    t_frame_streamer = threading.Thread(
        target=live_frame_streamer,
        args=(effective_backend_url, device_id, args.fps_stream, stop_event,
              latest_frame_lock, _get_latest_frame),
        daemon=True, name="frame-streamer",
    )
    t_frame_streamer.start()

    t_telemetry_http = threading.Thread(
        target=telemetry_streamer,
        args=(effective_backend_url, effective_session_id, effective_ulb_id, stop_event),
        daemon=True, name="telemetry-http",
    )
    t_telemetry_http.start()

    t_evidence = threading.Thread(
        target=evidence_upload_worker,
        args=(effective_backend_url, device_id, effective_ulb_id, stop_event, args.save_dir),
        daemon=True, name="evidence-upload",
    )
    t_evidence.start()

    t_gps_fallback = threading.Thread(
        target=gps_fallback_worker,
        args=(stop_event, enable_fallback, args.gps_fallback_timeout),
        daemon=True, name="gps-fallback",
    )
    t_gps_fallback.start()

    # ── ROI polygon (also AoD for litter) ─────────────────────────────────
    load_roi_polygon()

    # NOTE: LitterEngine is initialized AFTER the setup phase (below) so it
    # picks up any polygon the operator just drew in the setup screen.
    litter_engine: LitterEngine | None = None

    # ── GUI window ─────────────────────────────────────────────────────────
    WIN_TITLE = "SWSTP Unified Gateway (Motion + Litter)"
    if not args.headless:
        try:
            cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_TITLE, 960, 540)
            cv2.setMouseCallback(WIN_TITLE, on_mouse_roi, {"width": cam_w, "height": cam_h})
        except Exception:
            print("[GUI] Graphical display not available. Falling back to HEADLESS mode.")
            args.headless = True

    # ── Startup ROI / AoD setup phase (GUI mode only) ─────────────────────
    # In headless mode: saved polygon is loaded and used silently — no setup screen.
    # In GUI (desktop) mode: shows live camera feed with the saved polygon overlaid
    #   so the operator can confirm, redraw, or skip before detection starts.
    if not args.headless and cap is not None:
        print("[SETUP] Entering startup ROI/AoD configuration screen.")
        print("[SETUP] Press Enter/Space to use saved polygon, or draw a new one.")
        updated_roi = run_roi_setup_phase(cap, WIN_TITLE, cam_w, cam_h)
        # Propagate any change made in setup phase to the motion module globals
        _motion.active_polygon_roi = updated_roi
        print(f"[SETUP] ROI/AoD confirmed: {len(updated_roi)} pts. Starting detection loop.\n")
    else:
        if args.headless:
            print(f"[HEADLESS] Skipping setup screen — using saved polygon "
                  f"({len(_motion.active_polygon_roi)} pts) from roi_polygon.json.")

    # ── Litter Detection Engine (init after setup phase — uses final polygon) ─
    if not args.no_litter:
        litter_engine = LitterEngine(
            weights_path          = args.litter_weights,
            output_dir            = LITTER_CAPTURES_DIR,
            conf                  = args.litter_conf,
            overlap_threshold     = _cfg.LITTER_OVERLAP_THRESHOLD,
            aod_polygon_normalized = _motion.active_polygon_roi,   # ROI = AoD (post-setup)
            distance_interval_m   = args.litter_distance,
            gnss_lost_timeout_sec = _cfg.LITTER_GNSS_LOST_TIMEOUT_SEC,
            time_fallback_sec     = _cfg.LITTER_TIME_FALLBACK_SEC,
            device_id             = device_id,
        )

    delay       = max(1, int(1000 / (fps if (fps and 0 < fps < 120) else 30)))
    is_file     = isinstance(source, str)
    LOOP_VIDEO  = True
    bg_model    = None
    frame_count = 0
    last_save_time = 0.0
    saved_count    = 0
    last_known_frame = None

    stop_detector = VehicleStopDetector()

    print("[INIT] Unified Edge Gateway running.")
    print("  Controls: [t] Toggle Camera | [r] Plot ROI/AoD | [c] Clear | [q] Quit\n")

    if cap is not None:
        try:
            leds.set_system_ready()
        except Exception:
            pass
    else:
        try:
            leds.set_camera_scanning(True)
        except Exception:
            pass

    try:
        while not stop_event.is_set():

            # ── 1. Camera Scanning / Hold ─────────────────────────────────
            if cap is None or not cap.isOpened():
                try:
                    leds.set_camera_scanning(True)
                except Exception:
                    pass
                candidate_ports = get_candidate_video_ports(initial_source)
                new_cap, new_src, new_w, new_h, new_fps = scan_for_camera(candidate_ports)
                if new_cap is not None:
                    cap    = new_cap
                    source = new_src
                    cam_w, cam_h, fps = new_w, new_h, new_fps
                    delay  = max(1, int(1000 / (fps if (fps and 0 < fps < 120) else 30)))
                    is_file = isinstance(source, str) and not source.isdigit() and not str(source).startswith("/dev/video")
                    bg_model    = None
                    frame_count = 0
                    if hasattr(_motion_mod, "reset_tracking"):
                        try:
                            _motion_mod.reset_tracking()
                        except Exception:
                            pass
                    hardware_state["camera"] = {
                        "detected": True, "source": source,
                        "resolution": f"{cam_w}x{cam_h}", "fps": fps
                    }
                    print(f"\n[CAMERA] ✔ Webcam connected on port {source} ({cam_w}x{cam_h} @ {fps:.1f} FPS).")
                    try:
                        leds.set_camera_scanning(False)
                        leds.set_system_ready()
                    except Exception:
                        pass
                    if not args.headless:
                        try:
                            cv2.setMouseCallback(WIN_TITLE, on_mouse_roi, {"width": cam_w, "height": cam_h})
                        except Exception:
                            pass
                    continue

                # Standby screen
                standby_frame = np.zeros((cam_h or 360, cam_w or 640, 3), dtype=np.uint8)
                sh, sw = standby_frame.shape[:2]
                cv2.rectangle(standby_frame, (0, 0), (sw, sh), (20, 20, 24), -1)
                banner_y = sh // 2
                cv2.rectangle(standby_frame, (0, max(0, banner_y - 45)), (sw, min(sh, banner_y + 45)), (0, 165, 255), 2)
                cv2.putText(standby_frame, "NO WEBCAM DETECTED - SCANNING PORTS...",
                            (max(10, sw // 2 - 240), banner_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.putText(standby_frame, f"Probing {candidate_ports} | Algorithm ON HOLD | Triple LED blinks",
                            (max(10, sw // 2 - 260), banner_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
                display_frame = overlay_metadata(standby_frame, litter_engine)
                with latest_frame_lock:
                    latest_stream_frame = (display_frame, time.monotonic())
                if not args.headless:
                    cv2.imshow(WIN_TITLE, display_frame)
                    k = cv2.waitKey(250) & 0xFF
                    if k == ord('q'):
                        break
                else:
                    stop_event.wait(0.5)
                continue

            # ── 2. Camera feed paused ────────────────────────────────────
            if not _motion_mod.camera_feed_active:
                if last_known_frame is not None:
                    paused_frame = cv2.convertScaleAbs(last_known_frame.copy(), alpha=0.35, beta=0)
                else:
                    paused_frame = np.zeros((cam_h or 360, cam_w or 640, 3), dtype=np.uint8)
                ph, pw = paused_frame.shape[:2]
                banner_y = ph // 2
                cv2.rectangle(paused_frame, (0, max(0, banner_y - 45)), (pw, min(ph, banner_y + 45)), (0, 165, 255), 2)
                cv2.putText(paused_frame, "CAMERA FEED PAUSED",
                            (max(10, pw // 2 - 170), banner_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(paused_frame, "Press 't' to toggle camera feed ON",
                            (max(10, pw // 2 - 210), banner_y + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                display_frame = overlay_metadata(paused_frame, litter_engine)
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

            # ── 3. Frame acquisition ─────────────────────────────────────
            success, frame = cap.read()
            if not success:
                if is_file and LOOP_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    bg_model    = None
                    frame_count = 0
                    continue
                elif is_file:
                    print(f"[CAMERA] End of video file '{source}'.")
                    break
                else:
                    print(f"\n[CAMERA] Feed dropped on port {source} (device disconnected).")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    hardware_state["camera"] = {"detected": False, "source": None, "resolution": "N/A", "fps": 0}
                    bg_model    = None
                    frame_count = 0
                    if hasattr(_motion_mod, "reset_tracking"):
                        try:
                            _motion_mod.reset_tracking()
                        except Exception:
                            pass
                    try:
                        leds.set_camera_scanning(True)
                    except Exception:
                        pass
                    continue

            orig_frame       = frame.copy()
            last_known_frame = orig_frame.copy()
            orig_h, orig_w   = orig_frame.shape[:2]

            # ── 4. Downsample for motion processing ───────────────────────
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

            # ── 5. Warm-up ────────────────────────────────────────────────
            if frame_count <= WARMUP_FRAMES:
                display_frame = overlay_metadata(orig_frame.copy(), litter_engine)
                with latest_frame_lock:
                    latest_stream_frame = (display_frame, time.monotonic())
                if not args.headless:
                    cv2.imshow(WIN_TITLE, display_frame)
                    k = cv2.waitKey(delay) & 0xFF
                    if k == ord('q'):
                        break
                    elif k == ord('t'):
                        _motion_mod.camera_feed_active = not _motion_mod.camera_feed_active
                continue

            # ── 6. Background subtraction ─────────────────────────────────
            bg_uint8 = cv2.convertScaleAbs(bg_model)
            diff     = cv2.absdiff(bg_uint8, gray_blur)
            _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

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

            # ── 7. Draw ROI polygon overlay (= AoD for litter) ────────────
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
                            f"ROI (Motion) / AoD (Litter) | {len(disp_pts)} pts | 'r' to re-plot",
                            (disp_pts[0][0] + 5, max(20, disp_pts[0][1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, poly_color, 1, cv2.LINE_AA)

            # In-progress polygon drawing overlay
            if _motion_mod.is_drawing_polygon:
                prog_pts = [
                    [int(p[0] * orig_w), int(p[1] * orig_h)]
                    for p in _motion_mod.drawn_polygon_pts
                ]
                for idx, pt in enumerate(prog_pts):
                    pt_color = (0, 255, 0) if (idx == 0 and len(prog_pts) >= 3) else (0, 165, 255)
                    cv2.circle(orig_frame, tuple(pt), 6, pt_color, -1)
                    cv2.circle(orig_frame, tuple(pt), 10, (255, 255, 255), 1)
                    cv2.putText(orig_frame, f"P{idx+1}", (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if len(prog_pts) >= 2:
                    cv2.polylines(orig_frame, [np.array(prog_pts, dtype=np.int32)],
                                  isClosed=False, color=(0, 255, 255), thickness=2)
                    if len(prog_pts) >= 3:
                        cv2.line(orig_frame, tuple(prog_pts[-1]), tuple(prog_pts[0]), (0, 200, 100), 1, cv2.LINE_AA)
                cv2.putText(orig_frame,
                            f"PLOTTING ROI/AoD: Click pointers ({len(prog_pts)} set) | Press 'r' to finish",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

            # ── 8. Vehicle Stop & Motion Detection ────────────────────────
            now_mono      = time.monotonic()
            vehicle_speed = float(latest_sensor.get("speed") or 0.0)
            imu_data      = latest_sensor.get("imu")
            cap_lat       = latest_sensor.get("lat")
            cap_lon       = latest_sensor.get("lon")
            gps_valid     = bool(latest_sensor.get("gps_valid"))
            rtc_ts        = get_rtc_timestamp()

            stop_status = stop_detector.update(
                vehicle_speed=vehicle_speed,
                imu_data=imu_data,
                latitude=cap_lat,
                longitude=cap_lon,
                rtc_timestamp=rtc_ts,
                gps_valid=gps_valid,
                now_mono=now_mono,
            )

            is_vehicle_stopped    = stop_status["is_stopped"]
            current_event_duration = stop_status["duration_sec"]
            current_stop_id       = stop_status["stop_event_id"]
            current_capture_count = stop_status["capture_count"]

            latest_sensor["is_vehicle_stopped"]   = is_vehicle_stopped
            latest_sensor["vehicle_motion_state"] = stop_status["state"]
            latest_sensor["stop_duration_sec"]    = current_event_duration
            latest_sensor["stop_event_id"]        = current_stop_id
            latest_sensor["stop_capture_count"]   = current_capture_count

            display_frame = overlay_metadata(orig_frame.copy(), litter_engine)

            with latest_frame_lock:
                latest_stream_frame = (display_frame, time.monotonic())

            # ── 9. Litter Detection — 10 m GNSS trigger ──────────────────
            # Note: update_gnss() is called every frame (lightweight haversine check).
            # try_trigger() only posts to queue when 10 m threshold is crossed.
            if litter_engine is not None:
                if litter_engine.update_gnss(cap_lat, cap_lon):
                    # Grab a clean copy of the original (unprocessed) frame
                    litter_frame = orig_frame.copy()
                    litter_engine.try_trigger(
                        frame    = litter_frame,
                        lat      = cap_lat,
                        lon      = cap_lon,
                        gnss_data = latest_sensor,
                        rtc_ts   = rtc_ts,
                    )
                    # Cleanup old litter captures (rate-limited to once/minute)
                    cleanup_old_litter_captures(LITTER_CAPTURES_DIR, retention_days=3)

            # ── 10. Motion capture & upload ───────────────────────────────
            if motion_detected:
                text_scale = max(0.5, (orig_h / 480.0) * 0.5)
                cv2.putText(display_frame, "MOTION DETECTED",
                            (10, int(55 * text_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                            (0, 0, 255), max(1, int(orig_h / 360)), cv2.LINE_AA)

                if not is_vehicle_stopped:
                    pass  # Vehicle moving — skip capture
                elif now_mono - last_save_time >= SAVE_COOLDOWN_SEC:
                    saved_count   += 1
                    last_save_time = now_mono

                    _, enc_evidence = cv2.imencode(
                        '.jpg', display_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    evidence_bytes = enc_evidence.tobytes()

                    utc_iso    = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    cur_rtc_ts = get_rtc_timestamp() or utc_iso
                    clean_ts   = RE_CLEAN_FILENAME.sub("_", cur_rtc_ts)
                    base_name  = f"motion_RTC_{clean_ts}_{saved_count:04d}"
                    local_path = os.path.join(args.save_dir, f"{base_name}.jpg")
                    meta_path  = os.path.join(args.save_dir, f"{base_name}.json")

                    with open(local_path, "wb") as f:
                        f.write(evidence_bytes)

                    if cap_lat is None or cap_lon is None or (abs(cap_lat) < 0.001 and abs(cap_lon) < 0.001):
                        cap_lat = latest_sensor.get("last_known_valid_lat") or 0.0
                        cap_lon = latest_sensor.get("last_known_valid_lon") or 0.0

                    item_meta = {
                        "image_file":          f"{base_name}.jpg",
                        "captured_at":         utc_iso,
                        "rtc_timestamp":       cur_rtc_ts,
                        "collection_event_id": current_stop_id,
                        "idempotency_key":     str(uuid.uuid4()),
                        "width":               orig_w,
                        "height":              orig_h,
                        "compression_quality": 80,
                        "latitude":            round(float(cap_lat), 8) if cap_lat is not None else 0.0,
                        "longitude":           round(float(cap_lon), 8) if cap_lon is not None else 0.0,
                        "speed_kph":           round(float(vehicle_speed), 2),
                        "vehicle_speed_kmh":   round(float(vehicle_speed), 2),
                        "stop_duration_sec":   round(float(current_event_duration), 2),
                        "motion_confidence":   0.95,
                        "saved_count":         saved_count,
                        "local_path":          local_path,
                    }

                    stop_detector.record_capture(item_meta)
                    if stop_detector.active_stop is not None:
                        latest_sensor["stop_capture_count"] = len(stop_detector.active_stop.captures)

                    try:
                        with open(meta_path, "w", encoding="utf-8") as f_meta:
                            json.dump(item_meta, f_meta, indent=2)
                    except Exception as meta_err:
                        print(f"[CAPTURE] Warning saving metadata: {meta_err}")

                    queue_item = dict(item_meta)
                    queue_item["local_path"] = local_path
                    queue_item["meta_path"]  = meta_path
                    queue_item["jpeg_bytes"] = evidence_bytes

                    try:
                        upload_queue.put_nowait(queue_item)
                    except Exception:
                        print(f"[OFFLINE BACKUP] Upload queue full; {base_name}.jpg retained locally.")

                    print(f"[MOTION #{saved_count}] {local_path} | "
                          f"Stop #{current_stop_id} {current_event_duration:.1f}s | "
                          f"Speed: {vehicle_speed:.1f} km/h")
                    cleanup_old_local_captures(args.save_dir, retention_days=3)

                    # LED: triple yellow blink on motion snap
                    try:
                        leds.notify_motion_snap()
                    except Exception:
                        pass

            # ── 11. Display & keyboard ────────────────────────────────────
            if not args.headless:
                cv2.imshow(WIN_TITLE, display_frame)
                k = cv2.waitKey(delay) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('t'):
                    _motion_mod.camera_feed_active = not _motion_mod.camera_feed_active
                    print(f"\n[CAMERA] Camera feed {'ACTIVATED' if _motion_mod.camera_feed_active else 'PAUSED'}.")
                elif k == ord('r'):
                    if not _motion_mod.is_drawing_polygon:
                        _motion_mod.is_drawing_polygon = True
                        _motion_mod.drawn_polygon_pts  = []
                        print("\n[ROI/AoD PLOT] Click to place pointers. Press 'r' again when done.")
                    else:
                        if len(_motion_mod.drawn_polygon_pts) >= 3:
                            _motion_mod.active_polygon_roi = _motion_mod.drawn_polygon_pts.copy()
                            save_roi_polygon(_motion_mod.active_polygon_roi)
                            _motion_mod.is_drawing_polygon = False
                            _motion_mod.drawn_polygon_pts  = []
                            # Update litter engine AoD with new polygon
                            if litter_engine is not None:
                                litter_engine.aod_polygon_norm = _motion_mod.active_polygon_roi
                            print(f"\n[ROI/AoD] Saved {len(_motion_mod.active_polygon_roi)} pts. "
                                  f"Active for both Motion ROI and Litter AoD.")
                        else:
                            print(f"\n[ROI/AoD] Need ≥3 pts (have {len(_motion_mod.drawn_polygon_pts)}).")
                elif k == ord('c'):
                    if _motion_mod.is_drawing_polygon:
                        _motion_mod.drawn_polygon_pts = []
                        print("\n[ROI/AoD] Cleared temporary pointers.")

    finally:
        stop_event.set()
        if stop_detector.active_stop is not None:
            stop_detector.active_stop.end_mono     = time.monotonic()
            stop_detector.active_stop.end_rtc      = get_rtc_timestamp()
            stop_detector.active_stop.duration_sec = max(0.0, time.monotonic() - stop_detector.active_stop.start_mono)
            stop_detector.active_stop.end_reason   = "Edge gateway shutdown"
            print("\n" + stop_detector.active_stop.format_summary() + "\n")

        # Shutdown litter engine gracefully
        if litter_engine is not None:
            litter_engine.shutdown()

        try:
            leds.set_fault("Edge gateway shutting down")
            time.sleep(0.5)
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
        print(f"\n[UNIFIED GATEWAY STOPPED] Motion captures: {saved_count} | "
              f"Litter events: {litter_engine.event_count if litter_engine else 0}")


if __name__ == "__main__":
    main()
