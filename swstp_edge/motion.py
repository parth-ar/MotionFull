"""
motion.py — Motion detection, ROI polygon management, and frame overlay.

Verbatim port of all motion detection code from webcam_motion_detect.py.
No algorithm parameters have been changed:
  DIFF_THRESHOLD   = 22
  MIN_CONTOUR_AREA = 350
  BG_ALPHA         = 0.04
  WARMUP_FRAMES    = 30
  SAVE_COOLDOWN_SEC = 2.0

Linux/Pi note: cv2.VideoCapture() uses V4L2 on Linux.  The Windows-specific
  cv2.CAP_DSHOW backend flag is guarded and omitted on non-Windows platforms.
"""

import json
import math
import os
import time

import cv2
import imutils
import numpy as np

from config import (
    ROI_CONFIG_FILE, CAPTURES_DIR,
    DIFF_THRESHOLD, MIN_CONTOUR_AREA, BG_ALPHA, WARMUP_FRAMES,
    SAVE_COOLDOWN_SEC, FRAME_SIZE,
)

# ---------------------------------------------------------------------------
# ROI polygon state
# ---------------------------------------------------------------------------
DEFAULT_ROI_POLYGON = [[0.35, 0.10], [0.65, 0.10], [0.65, 0.50], [0.35, 0.50]]
active_polygon_roi: list = DEFAULT_ROI_POLYGON.copy()
is_drawing_polygon: bool = False
drawn_polygon_pts:  list = []
camera_feed_active: bool = True


# ---------------------------------------------------------------------------
# ROI persistence
# ---------------------------------------------------------------------------
def load_roi_polygon(config_file: str = ROI_CONFIG_FILE) -> list:
    """Loads saved polygon ROI vertices from JSON config file so previous setup is remembered on boot."""
    global active_polygon_roi
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) >= 3:
                    active_polygon_roi = data
                    print(f"[ROI] Restored previous Area of Interest ({len(data)} points) from {config_file}")
                    return active_polygon_roi
        except Exception as e:
            print(f"[ROI] Warning loading {config_file}: {e}")
    active_polygon_roi = DEFAULT_ROI_POLYGON.copy()
    save_roi_polygon(active_polygon_roi, config_file)
    print(f"[ROI] Initialized default Area of Interest ({len(active_polygon_roi)} points) to {config_file}")
    return active_polygon_roi


def save_roi_polygon(pts: list, config_file: str = ROI_CONFIG_FILE) -> None:
    """Saves polygon ROI vertices to JSON config file for persistent memory across boots."""
    try:
        with open(config_file, "w") as f:
            json.dump(pts, f, indent=2)
        print(f"[ROI] Persisted Area of Interest ({len(pts)} points) to {config_file}")
    except Exception as e:
        print(f"[ROI] Warning saving {config_file}: {e}")


# ---------------------------------------------------------------------------
# ROI mask application
# ---------------------------------------------------------------------------
def apply_polygon_roi_mask(thresh, polygon_pts: list, frame_w: int, frame_h: int):
    """Creates a binary mask from polygon vertices and applies bitwise AND to threshold."""
    if not polygon_pts or len(polygon_pts) < 3:
        return thresh

    pts = []
    for p in polygon_pts:
        # Support normalized (0.0 - 1.0) and pixel coordinates
        px = int(p[0] * frame_w) if p[0] <= 1.0 else int(p[0])
        py = int(p[1] * frame_h) if p[1] <= 1.0 else int(p[1])
        pts.append([px, py])

    pts_array = np.array(pts, dtype=np.int32)
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_array], 255)
    return cv2.bitwise_and(thresh, mask)


# ---------------------------------------------------------------------------
# Mouse callback for interactive ROI plotting
# ---------------------------------------------------------------------------
def on_mouse_roi(event, x: int, y: int, flags, param) -> None:
    """Interactive mouse callback to place pointers for the Area of Interest in OpenCV GUI window."""
    global drawn_polygon_pts, active_polygon_roi, is_drawing_polygon
    if not is_drawing_polygon:
        return

    frame_w, frame_h = param.get("width", 640), param.get("height", 360)

    if event == cv2.EVENT_LBUTTONDOWN:
        norm_x = round(float(x) / frame_w, 4)
        norm_y = round(float(y) / frame_h, 4)

        # If >= 3 points and clicking near P1, auto-complete & save
        if len(drawn_polygon_pts) >= 3:
            p1_x, p1_y = drawn_polygon_pts[0]
            dist = math.hypot(norm_x - p1_x, norm_y - p1_y)
            if dist < 0.04:
                active_polygon_roi = drawn_polygon_pts.copy()
                save_roi_polygon(active_polygon_roi)
                is_drawing_polygon = False
                drawn_polygon_pts = []
                print(f"[ROI PLOT] Auto-snapped to P1! Connected {len(active_polygon_roi)} pointers & saved Area of Interest.")
                return

        drawn_polygon_pts.append([norm_x, norm_y])
        print(f"[ROI PLOT] Added pointer P{len(drawn_polygon_pts)} at ({x}, {y}) -> [{norm_x}, {norm_y}]. Press 'r' when done to connect.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(drawn_polygon_pts) >= 3:
            active_polygon_roi = drawn_polygon_pts.copy()
            save_roi_polygon(active_polygon_roi)
            is_drawing_polygon = False
            drawn_polygon_pts = []
            print(f"[ROI PLOT] Right-click completed! Connected {len(active_polygon_roi)} pointers & saved Area of Interest.")
        else:
            print(f"[ROI PLOT] Need at least 3 pointers (currently {len(drawn_polygon_pts)}). Click on video to add more points or press 'c' to clear.")


# ---------------------------------------------------------------------------
# Frame overlay HUD
# ---------------------------------------------------------------------------
def get_rtc_timestamp() -> str | None:
    from telemetry import latest_sensor
    ts = latest_sensor.get("timestamp")
    if ts and str(ts).strip() not in ("Waiting for RTC...", "", "None", "0"):
        return str(ts).strip()
    return None


def overlay_metadata(frame):
    """Draws real-time hardware RTC time, GNSS, and status HUD on frame with zero memory allocation."""
    from telemetry import latest_sensor, hardware_state

    rtc_ts    = get_rtc_timestamp()
    rtc_error = latest_sensor.get("rtc_error")
    # On Pi there is no serial port in the Arduino sense; always show "Pi-native"
    serial_connected = True
    active_port      = latest_sensor.get("serial_port", "Pi-native")
    backend_ok       = hardware_state["backend"]["connected"]
    active_sid       = latest_sensor.get("active_session_id") or hardware_state["backend"].get("active_session_id") or 0

    if rtc_ts:
        ts_text  = f"RTC: {rtc_ts}"
        ts_color = (0, 255, 255)
    elif rtc_error:
        ts_text  = f"RTC: {rtc_error}"
        ts_color = (0, 140, 255)
    else:
        ts_text  = "RTC: Awaiting DS3231 sync…"
        ts_color = (0, 200, 255)

    is_fallback = latest_sensor.get("location_source") == "fallback"
    if latest_sensor["gps_valid"] and latest_sensor["lat"] is not None and is_fallback:
        loc_text  = f"GPS (fallback): {latest_sensor['lat']:.8f}, {latest_sensor['lon']:.8f}"
        loc_color = (0, 200, 255)
    elif latest_sensor["gps_valid"] and latest_sensor["lat"] is not None:
        loc_text  = f"GNSS: {latest_sensor['lat']:.8f}, {latest_sensor['lon']:.8f}"
        loc_color = (0, 255, 0)
    else:
        loc_text  = "GNSS: Searching for fix… (gpsd)"
        loc_color = (0, 165, 255)

    h, w = frame.shape[:2]
    font_scale  = max(0.4, (h / 480.0) * 0.45)
    thickness   = max(1, int(h / 360))
    line_spacing = int(font_scale * 24)
    strip_height = line_spacing * 2 + 15

    # Top Status HUD
    cam_badge = "CAM:OK" if hardware_state["camera"]["detected"] else "CAM:OFF"
    rtc_badge = "RTC:ONLINE" if rtc_ts else "RTC:WAIT"
    gps_badge = (
        "GPS:FALLBACK" if latest_sensor.get("location_source") == "fallback"
        else "GPS:FIX" if latest_sensor.get("gps_valid")
        else "GPS:SEARCH"
    )
    imu_badge = "IMU:OK" if (latest_sensor.get("imu") and latest_sensor["imu"].get("valid")) else "IMU:WAIT"
    net_badge = (
        f"NET:ONLINE (SESS #{active_sid})" if (backend_ok and active_sid > 0)
        else ("NET:ONLINE" if backend_ok else "NET:OFFLINE")
    )
    try:
        from sensors.power import power_state as _ps
        if _ps.get("shutdown_in_progress"):
            pwr_badge = "PWR:SHUTDOWN"
        elif _ps.get("low_battery_triggered"):
            pwr_badge = "BATT:LOW"
        elif _ps.get("on_backup_battery"):
            pwr_badge = "PWR:BATT"
        else:
            pwr_badge = "PWR:MAIN"
    except Exception:
        pwr_badge = "PWR:OK"

    hud_text = f"[SWSTP-PI] {cam_badge} | {rtc_badge} | {imu_badge} | {gps_badge} | {pwr_badge} | {net_badge}"
    cv2.putText(frame, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


    # Bottom Metadata Strip
    banner_slice = frame[h - strip_height:h, 0:w]
    cv2.convertScaleAbs(banner_slice, banner_slice, alpha=0.25, beta=0)

    y_pos = h - strip_height + line_spacing
    cv2.putText(frame, ts_text,  (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ts_color,  thickness, cv2.LINE_AA)
    y_pos += line_spacing
    cv2.putText(frame, loc_text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, loc_color, thickness, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Synthetic video frame generator
# ---------------------------------------------------------------------------
def generate_virtual_video_frame(width: int, height: int, count: int):
    """Generates a dynamic real-time simulated roadside camera frame with simulated motion."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[0:height // 2, :] = (55, 45, 35)
    img[height // 2:, :]  = (30, 30, 30)
    dash_offset = (count * 6) % 60
    for x in range(-dash_offset, width, 60):
        cv2.line(img, (x, height * 3 // 4), (min(width, x + 30), height * 3 // 4), (200, 200, 200), 2)
    motion_x = int(width  * (0.42 + 0.12 * math.sin(count * 0.08)))
    motion_y = int(height * (0.22 + 0.08 * math.cos(count * 0.08)))
    cv2.rectangle(img, (motion_x - 22, motion_y - 16), (motion_x + 22, motion_y + 16), (0, 140, 255), -1)
    cv2.putText(img, "SWSTP-PI EDGE VIDEO FEED", (width // 2 - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Local capture cleanup
# ---------------------------------------------------------------------------
_last_cleanup_time = 0.0


def cleanup_old_local_captures(save_dir: str = CAPTURES_DIR,
                                retention_days: int = 3,
                                force: bool = False) -> None:
    """Deletes local capture frames older than retention_days with rate-limiting."""
    global _last_cleanup_time
    now = time.time()
    if not force and (now - _last_cleanup_time) < 60.0:
        return
    _last_cleanup_time = now
    if not os.path.exists(save_dir):
        return
    cutoff = now - (retention_days * 86400)
    try:
        with os.scandir(save_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    try:
                        if entry.stat().st_mtime < cutoff:
                            os.remove(entry.path)
                    except Exception:
                        pass
    except Exception:
        pass
