"""
motion.py — Motion detection, ROI polygon management, and frame overlay.
(SWSTP Unified — adapted from swstp_edge/motion.py)

No algorithm parameters have been changed:
  DIFF_THRESHOLD   = 22
  MIN_CONTOUR_AREA = 350
  BG_ALPHA         = 0.04
  WARMUP_FRAMES    = 30
  SAVE_COOLDOWN_SEC = 5.0

The ROI polygon stored in roi_polygon.json is ALSO used as the Area of
Disinterest (AoD) for litter detection — same file, inverted usage.
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


def reset_tracking() -> None:
    global is_drawing_polygon, drawn_polygon_pts
    is_drawing_polygon = False
    drawn_polygon_pts.clear()


# ---------------------------------------------------------------------------
# ROI persistence
# ---------------------------------------------------------------------------
def load_roi_polygon(config_file: str = ROI_CONFIG_FILE) -> list:
    """Load saved polygon ROI vertices from JSON. Also used as AoD for litter."""
    global active_polygon_roi
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) >= 3:
                    active_polygon_roi = data
                    print(f"[ROI] Restored Area of Interest ({len(data)} pts) from {config_file}")
                    print(f"[ROI] Same polygon used as Area of Disinterest for litter detection.")
                    return active_polygon_roi
        except Exception as e:
            print(f"[ROI] Warning loading {config_file}: {e}")
    active_polygon_roi = DEFAULT_ROI_POLYGON.copy()
    save_roi_polygon(active_polygon_roi, config_file)
    print(f"[ROI] Initialized default Area of Interest ({len(active_polygon_roi)} pts) to {config_file}")
    return active_polygon_roi


def save_roi_polygon(pts: list, config_file: str = ROI_CONFIG_FILE) -> None:
    """Save polygon ROI vertices to JSON for persistent memory across boots."""
    try:
        with open(config_file, "w") as f:
            json.dump(pts, f, indent=2)
        print(f"[ROI] Persisted Area of Interest ({len(pts)} pts) to {config_file}")
    except Exception as e:
        print(f"[ROI] Warning saving {config_file}: {e}")


# ---------------------------------------------------------------------------
# ROI mask application
# ---------------------------------------------------------------------------
def apply_polygon_roi_mask(thresh, polygon_pts: list, frame_w: int, frame_h: int):
    """Create binary mask from polygon and apply bitwise AND to threshold image."""
    if not polygon_pts or len(polygon_pts) < 3:
        return thresh
    pts = []
    for p in polygon_pts:
        px = int(p[0] * frame_w) if p[0] <= 1.0 else int(p[0])
        py = int(p[1] * frame_h) if p[1] <= 1.0 else int(p[1])
        pts.append([px, py])
    pts_array = np.array(pts, dtype=np.int32)
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_array], 255)
    return cv2.bitwise_and(thresh, mask)


# ---------------------------------------------------------------------------
# Startup ROI / AoD Setup Phase  (GUI mode only)
# ---------------------------------------------------------------------------
def run_roi_setup_phase(cap, win_title: str, frame_w: int, frame_h: int) -> list:
    """
    Interactive startup phase shown in GUI (desktop) mode ONLY.
    Displays a live camera feed with the currently-saved ROI/AoD polygon overlaid
    and lets the operator confirm, redraw, or skip before detection begins.

    This is the same polygon used for:
      • Motion detection  → Area of Interest  (only motion INSIDE counts)
      • Litter detection  → Area of Disinterest (detections INSIDE are ignored)

    Controls
    --------
    Left-click        Add a new vertex to the polygon being drawn
    Right-click       Remove the last vertex placed
    Enter / Space     Confirm polygon (≥ 3 points required to confirm a new one)
                      If no new points were placed, confirms the saved polygon as-is
    Esc               Skip — use the currently-saved polygon unchanged
    C                 Clear all placed points and start drawing fresh

    Returns the (possibly updated) normalised polygon points list.
    """
    global active_polygon_roi

    # Points being drawn in this session (pixel coords during draw, converted to
    # normalised [0-1] on confirm)
    new_points_px = []   # list of (x, y) pixel tuples placed this session
    mouse_pos     = [0, 0]
    warn_until    = 0.0

    def _mouse_cb(event, x, y, _flags, _param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            new_points_px.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and new_points_px:
            new_points_px.pop()

    try:
        cv2.setMouseCallback(win_title, _mouse_cb)
    except Exception:
        # Window not yet created or headless — skip phase
        return active_polygon_roi

    print("\n[ROI/AoD SETUP] Startup polygon configuration:")
    print("  The same polygon is used as Motion ROI and Litter AoD.")
    print("  • Left-click  : place a new point")
    print("  • Right-click : undo last point")
    print("  • C key       : clear and start fresh")
    print("  • Enter/Space : confirm new polygon (need ≥3 pts), or keep saved polygon")
    print("  • Esc         : skip — use current saved polygon as-is\n")

    # Build pixel-coord version of the currently saved polygon for display
    def _saved_poly_px():
        pts = []
        for p in active_polygon_roi:
            px = int(p[0] * frame_w) if p[0] <= 1.0 else int(p[0])
            py = int(p[1] * frame_h) if p[1] <= 1.0 else int(p[1])
            pts.append((px, py))
        return pts

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        h, w = display.shape[:2]
        now = time.time()

        saved_px = _saved_poly_px()

        # ── Draw currently saved polygon (dimmed) ──
        if len(saved_px) >= 3 and not new_points_px:
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(saved_px, dtype=np.int32)], (0, 80, 180))
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.polylines(display, [np.array(saved_px, dtype=np.int32)],
                          isClosed=True, color=(0, 200, 255), thickness=2)
            for i, pt in enumerate(saved_px):
                cv2.circle(display, pt, 5, (0, 200, 255), -1)
                cv2.putText(display, f"P{i+1}", (pt[0]+7, pt[1]-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            cx = int(np.mean([p[0] for p in saved_px]))
            cy = int(np.mean([p[1] for p in saved_px]))
            cv2.putText(display, "SAVED ROI / AoD", (cx - 70, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # ── Draw new polygon being placed ──
        if new_points_px:
            if len(new_points_px) >= 3:
                overlay = display.copy()
                cv2.fillPoly(overlay, [np.array(new_points_px, dtype=np.int32)], (0, 0, 160))
                cv2.addWeighted(overlay, 0.30, display, 0.70, 0, display)
            if len(new_points_px) >= 2:
                cv2.polylines(display, [np.array(new_points_px, dtype=np.int32)],
                              isClosed=False, color=(0, 255, 255), thickness=2)
                # Closing edge preview
                cv2.line(display, new_points_px[-1], new_points_px[0], (0, 200, 100), 1, cv2.LINE_AA)
            # Preview line to cursor
            cv2.line(display, new_points_px[-1], tuple(mouse_pos), (0, 200, 200), 1)
            for i, pt in enumerate(new_points_px):
                col = (0, 255, 0) if i == 0 else (0, 255, 255)
                cv2.circle(display, pt, 7, col, -1)
                cv2.circle(display, pt, 7, (0, 0, 0), 1)
                cv2.putText(display, str(i + 1), (pt[0] + 9, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            if len(new_points_px) >= 3:
                cv2.circle(display, new_points_px[0], 14, (0, 255, 0), 2)

        # ── Top banner ──
        cv2.rectangle(display, (0, 0), (w, 50), (15, 15, 18), -1)
        cv2.putText(display, "SETUP  —  Region of Interest / Area of Disinterest",
                    (10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 1, cv2.LINE_AA)

        # ── Bottom instruction strip ──
        cv2.rectangle(display, (0, h - 62), (w, h), (15, 15, 18), -1)
        cv2.line(display, (0, h - 62), (w, h - 62), (50, 50, 50), 1)

        if new_points_px:
            tip1 = (f"Drawing: {len(new_points_px)} pts placed  |  "
                    f"Right-click: undo  |  C: clear  |  Enter/Space: confirm (need ≥3)")
        else:
            tip1 = ("Showing SAVED polygon  |  Left-click to draw a NEW one  |  "
                    "Enter/Space: keep saved  |  Esc: skip")
        tip2 = "Left-click: add point   Right-click: undo   C: clear   Enter/Space: confirm   Esc: use saved"
        cv2.putText(display, tip1, (10, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display, tip2, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

        # ── "Need ≥3 points" warning flash ──
        if now < warn_until:
            cv2.putText(display, "Need at least 3 points to confirm!",
                        (w // 2 - 190, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

        try:
            cv2.imshow(win_title, display)
        except Exception:
            # Display failed mid-session — fall back gracefully
            break

        key = cv2.waitKey(1) & 0xFF

        if key in (13, 32):  # Enter or Space
            if new_points_px:
                if len(new_points_px) >= 3:
                    # Convert pixel coords → normalised [0-1]
                    normalised = [
                        [round(px / frame_w, 4), round(py / frame_h, 4)]
                        for px, py in new_points_px
                    ]
                    active_polygon_roi = normalised
                    save_roi_polygon(active_polygon_roi)
                    print(f"[ROI/AoD SETUP] New polygon confirmed & saved ({len(normalised)} pts).")
                    break
                else:
                    warn_until = now + 1.8
            else:
                # No new drawing — use saved polygon as-is
                print(f"[ROI/AoD SETUP] Using saved polygon ({len(active_polygon_roi)} pts). Proceeding.")
                break

        elif key == 27:  # Esc
            print(f"[ROI/AoD SETUP] Skipped. Using saved polygon ({len(active_polygon_roi)} pts).")
            break

        elif key in (ord('c'), ord('C')):
            new_points_px.clear()
            print("[ROI/AoD SETUP] Cleared new points — start drawing fresh.")

    # Restore mouse callback to the runtime ROI handler
    try:
        cv2.setMouseCallback(win_title, on_mouse_roi, {"width": frame_w, "height": frame_h})
    except Exception:
        pass

    return active_polygon_roi


# ---------------------------------------------------------------------------
# Mouse callback for interactive ROI plotting (runtime, during detection)
# ---------------------------------------------------------------------------
def on_mouse_roi(event, x: int, y: int, flags, param) -> None:
    """Interactive mouse callback to place pointers for the Area of Interest."""
    global drawn_polygon_pts, active_polygon_roi, is_drawing_polygon
    if not is_drawing_polygon:
        return
    frame_w, frame_h = param.get("width", 640), param.get("height", 360)
    if event == cv2.EVENT_LBUTTONDOWN:
        norm_x = round(float(x) / frame_w, 4)
        norm_y = round(float(y) / frame_h, 4)
        if len(drawn_polygon_pts) >= 3:
            p1_x, p1_y = drawn_polygon_pts[0]
            dist = math.hypot(norm_x - p1_x, norm_y - p1_y)
            if dist < 0.04:
                active_polygon_roi = drawn_polygon_pts.copy()
                save_roi_polygon(active_polygon_roi)
                is_drawing_polygon = False
                drawn_polygon_pts  = []
                print(f"[ROI PLOT] Auto-snapped to P1! Connected {len(active_polygon_roi)} pts & saved.")
                return
        drawn_polygon_pts.append([norm_x, norm_y])
        print(f"[ROI PLOT] Added P{len(drawn_polygon_pts)} at ({x},{y}) -> [{norm_x},{norm_y}].")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(drawn_polygon_pts) >= 3:
            active_polygon_roi = drawn_polygon_pts.copy()
            save_roi_polygon(active_polygon_roi)
            is_drawing_polygon = False
            drawn_polygon_pts  = []
            print(f"[ROI PLOT] Right-click completed! Saved {len(active_polygon_roi)} pts.")
        else:
            print(f"[ROI PLOT] Need ≥3 pts (have {len(drawn_polygon_pts)}).")


# ---------------------------------------------------------------------------
# Frame overlay HUD
# ---------------------------------------------------------------------------
def get_rtc_timestamp() -> str | None:
    from telemetry import latest_sensor
    ts = latest_sensor.get("timestamp")
    if ts and str(ts).strip() not in ("Waiting for RTC...", "", "None", "0"):
        return str(ts).strip()
    return None


def overlay_metadata(frame, litter_engine=None):
    """
    Draws real-time hardware RTC time, GNSS, status HUD, and litter status on frame.
    Accepts an optional litter_engine reference to show distance/trigger HUD.
    """
    from telemetry import latest_sensor, hardware_state

    rtc_ts           = get_rtc_timestamp()
    rtc_error        = latest_sensor.get("rtc_error")
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
        loc_text  = "GNSS: Searching for fix…"
        loc_color = (0, 165, 255)

    h, w = frame.shape[:2]
    font_scale   = max(0.4, (h / 480.0) * 0.45)
    thickness    = max(1, int(h / 360))
    line_spacing = int(font_scale * 24)
    hud_lines    = 3 if litter_engine is not None else 2
    strip_height = line_spacing * hud_lines + 15

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

    is_veh_stopped = bool(latest_sensor.get("is_vehicle_stopped"))
    stop_sec = int(latest_sensor.get("stop_duration_sec") or 0)
    stop_caps = int(latest_sensor.get("stop_capture_count") or 0)
    mins, s = divmod(stop_sec, 60)
    if is_veh_stopped:
        veh_badge = f"VEH:STOP({mins:02d}:{s:02d}|{stop_caps}c)"
    else:
        curr_spd = float(latest_sensor.get("speed") or 0.0)
        veh_badge = f"VEH:MOV({curr_spd:.1f}kph)"

    # Litter engine badge
    if litter_engine is not None:
        if litter_engine.model_ready:
            if litter_engine.op_mode == "GNSS_DISTANCE":
                lit_badge = (f"LITTER:{litter_engine.distance_since_trigger:.1f}/"
                             f"{litter_engine.distance_interval_m:.0f}m "
                             f"T{litter_engine.trigger_count} E{litter_engine.event_count}")
            else:
                lit_badge = f"LITTER:CLK T{litter_engine.trigger_count} E{litter_engine.event_count}"
        else:
            lit_badge = "LITTER:LOADING"
    else:
        lit_badge = ""

    hud_text = (f"[SWSTP-UNIFIED] {cam_badge} | {rtc_badge} | {imu_badge} | "
                f"{gps_badge} | {veh_badge} | {pwr_badge} | {net_badge}")
    cv2.putText(frame, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    if lit_badge:
        cv2.putText(frame, lit_badge, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)

    # Bottom Metadata Strip
    banner_slice = frame[h - strip_height:h, 0:w]
    cv2.convertScaleAbs(banner_slice, banner_slice, alpha=0.25, beta=0)

    if is_veh_stopped:
        stop_info_text = f" | [STOP #{latest_sensor.get('stop_event_id', 1)}: {mins:02d}m {s:02d}s | Caps: {stop_caps}]"
        full_ts_text = f"{ts_text}{stop_info_text}"
    else:
        full_ts_text = ts_text

    y_pos = h - strip_height + line_spacing
    cv2.putText(frame, full_ts_text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ts_color, thickness, cv2.LINE_AA)
    y_pos += line_spacing
    cv2.putText(frame, loc_text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, loc_color, thickness, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Local capture cleanup
# ---------------------------------------------------------------------------
_last_cleanup_time = 0.0


def cleanup_old_local_captures(save_dir: str = CAPTURES_DIR,
                                retention_days: int = 3,
                                force: bool = False) -> None:
    """Safety guard: delete local capture frames older than retention_days."""
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
                            if entry.name.lower().endswith(".jpg"):
                                json_path = os.path.splitext(entry.path)[0] + ".json"
                                if os.path.exists(json_path):
                                    os.remove(json_path)
                    except Exception:
                        pass
    except Exception:
        pass
