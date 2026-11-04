#!/usr/bin/env python3
"""
test_camera_location_stream.py — GNSS Fix Loss Diagnostic & Stream Isolation Test.

PURPOSE & HYPOTHESIS:
---------------------
Determines whether GNSS fix losses observed during operation are caused by:
  A) High CPU / thread starvation / GIL contention from the OpenCV motion detection
     and YOLO litter detection pipeline, OR
  B) A genuine GNSS hardware, antenna, RF interference, or serial connection fault.

TEST METHODOLOGY:
-----------------
Runs ONLY the two live network transmission streams, exactly matching the main program:
  1. Camera Video Stream    -> POST /api/camera/{deviceId}/frame
                             (via live_frame_streamer in network.uploader)
  2. Live Location Stream   -> POST /api/telemetry/ingest-batch
                             (via telemetry_streamer in network.uploader & 20 Hz telemetry loop)

EXCLUDED FROM THIS TEST (Zero motion capture overhead):
  - NO background subtraction (cv2.createBackgroundSubtractorMOG2 / cv2.absdiff)
  - NO contour finding, bounding box calculation, or ROI filtering
  - NO YOLO litter detection engine (LitterEngine / ONNX Runtime CPU inference)
  - NO vehicle stop detection state machine
  - NO local evidence frame captures or disk writes
  - NO evidence upload worker
  - NO new /api/{vehicle_id}/live_location API (which is not yet implemented backend-side)

METRICS TRACKED:
  - GNSS Fix Health: 3D Fix vs No-Fix vs No-Data status, satellite count, HDOP, coordinates
  - Fix Stability: Total fix time %, number of fix drops, maximum outage duration
  - Resource Usage: System CPU %, Process CPU %, RAM usage, Raspberry Pi CPU Temperature
  - Network Stream Health: Video FPS, Ingest batches, HTTP 200 responses vs errors

USAGE EXAMPLES:
  # Standard test on Raspberry Pi (Headless, auto-detects camera and GNSS):
  python3 test_camera_location_stream.py --headless

  # Test with live terminal dashboard (interactive SSH session):
  python3 test_camera_location_stream.py --headless

  # Run with specific Camera Module 3 / IMX708 and GNSS UART:
  python3 test_camera_location_stream.py --source imx708 --af-mode continuous --gnss-port /dev/serial0

  # Run for exactly 10 minutes and produce a summary report:
  python3 test_camera_location_stream.py --headless --duration 600

  # Log second-by-second GNSS stability and CPU metrics to CSV:
  python3 test_camera_location_stream.py --headless --log-csv gnss_isolation_test.csv

  # Run automated self-test (mock backend and simulated GNSS):
  python3 test_camera_location_stream.py --self-test
"""

import argparse
import csv
import datetime
import glob
import http.server
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time

# Ensure swstp_edge is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Headless / systemd safety
if "--headless" in sys.argv or "DISPLAY" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "V4L2"

import cv2
if hasattr(cv2, "setLogLevel"):
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass
import numpy as np
import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config as _cfg
from config import (
    DEFAULT_BACKEND_URL,
    DEFAULT_ULB_ID,
    DEFAULT_STREAM_FPS,
    GNSS_DATA_TIMEOUT_SEC,
)

import sensors.rtc as rtc_sensor
import sensors.imu as imu_sensor
import sensors.gnss as gnss_sensor
import sensors.leds as leds

import telemetry as _telemetry
from telemetry import (
    latest_sensor,
    hardware_state,
    telemetry_queue,
)

from network.uploader import (
    auto_detect_backend_url,
    sync_backend_metadata,
    ensure_hardware_session,
    live_frame_streamer,
    telemetry_streamer,
    dynamic_session_info,
)

from camera import (
    get_candidate_video_ports,
    scan_for_camera,
)

# ---------------------------------------------------------------------------
# Globals & Thread Synchronization
# ---------------------------------------------------------------------------
latest_frame_lock = threading.Lock()
latest_stream_frame = None  # tuple (ndarray, monotonic_timestamp)

stop_event = threading.Event()


# ---------------------------------------------------------------------------
# System Resource Monitor (Linux / Raspberry Pi / Windows fallback)
# ---------------------------------------------------------------------------
_prev_proc_stat_total = None
_prev_proc_stat_idle = None
_proc_start_cpu_time = None
_proc_start_mono = None

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


def get_system_metrics() -> dict:
    """Return system CPU %, process CPU %, RAM used MB & %, and CPU temperature.
    Completely zero external dependencies: reads /proc and /sys directly on Linux.
    """
    global _prev_proc_stat_total, _prev_proc_stat_idle

    cpu_pct = 0.0
    proc_cpu_pct = 0.0
    ram_mb = 0.0
    ram_pct = 0.0
    temp_c = None

    # 1. PSUtil if available
    if _HAVE_PSUTIL:
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_mb = mem.used / (1024.0 * 1024.0)
            ram_pct = mem.percent
            proc = psutil.Process()
            proc_cpu_pct = proc.cpu_percent(interval=None)
        except Exception:
            pass

    # 2. Linux /proc/stat CPU fallback
    if not _HAVE_PSUTIL and os.path.exists("/proc/stat"):
        try:
            with open("/proc/stat", "r") as f:
                first_line = f.readline()
            parts = [float(x) for x in first_line.split()[1:]]
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
            total = sum(parts)
            if _prev_proc_stat_total is not None and total > _prev_proc_stat_total:
                delta_total = total - _prev_proc_stat_total
                delta_idle = idle - _prev_proc_stat_idle
                cpu_pct = max(0.0, min(100.0, (1.0 - (delta_idle / delta_total)) * 100.0))
            _prev_proc_stat_total = total
            _prev_proc_stat_idle = idle
        except Exception:
            pass

    # 3. Linux /proc/meminfo fallback
    if not _HAVE_PSUTIL and os.path.exists("/proc/meminfo"):
        try:
            mem_vals = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    p = line.split(":")
                    if len(p) == 2:
                        mem_vals[p[0].strip()] = float(p[1].split()[0])
            total_kb = mem_vals.get("MemTotal", 0.0)
            avail_kb = mem_vals.get("MemAvailable", mem_vals.get("MemFree", 0.0))
            used_kb = total_kb - avail_kb
            ram_mb = used_kb / 1024.0
            ram_pct = (used_kb / total_kb * 100.0) if total_kb > 0 else 0.0
        except Exception:
            pass

    # 4. Raspberry Pi Thermal Zone
    try:
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            with open(thermal_path, "r") as f:
                temp_c = float(f.read().strip()) / 1000.0
    except Exception:
        temp_c = None

    return {
        "cpu_percent": round(cpu_pct, 1),
        "proc_cpu_percent": round(proc_cpu_pct, 1),
        "ram_used_mb": round(ram_mb, 1),
        "ram_percent": round(ram_pct, 1),
        "cpu_temp_c": round(temp_c, 1) if temp_c is not None else None,
    }


# ---------------------------------------------------------------------------
# Fix Stability Tracker
# ---------------------------------------------------------------------------
class FixStabilityTracker:
    """Tracks GNSS fix continuity, dropout events, outage durations, and correlations with CPU."""

    def __init__(self, log_csv_path: str | None = None):
        self.start_mono = time.monotonic()
        self.last_tick_mono = self.start_mono

        self.total_seconds = 0.0
        self.fix_seconds = 0.0
        self.no_fix_seconds = 0.0
        self.no_data_seconds = 0.0

        self.prev_has_fix = False
        self.fix_drop_count = 0
        self.fix_recovery_count = 0

        self.current_fix_streak = 0.0
        self.current_loss_streak = 0.0
        self.max_outage_sec = 0.0

        self.drop_events = []  # dict records of each dropout

        self.cpu_samples = []
        self.ram_samples = []
        self.temp_samples = []

        self.frames_streamed = 0
        self.last_frames_streamed = 0

        self.csv_file = None
        self.csv_writer = None
        if log_csv_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(log_csv_path)), exist_ok=True)
            except Exception:
                pass
            self.csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "timestamp_iso",
                "elapsed_sec",
                "gnss_fix",
                "gnss_status",
                "latitude",
                "longitude",
                "altitude_m",
                "speed_kmh",
                "satellites",
                "hdop",
                "cpu_percent",
                "ram_used_mb",
                "cpu_temp_c",
                "telemetry_packets_sent",
                "fix_drop_count",
            ])
            self.csv_file.flush()

    def update(self, gnss_snap: dict, metrics: dict, telemetry_count: int) -> dict:
        now_mono = time.monotonic()
        dt = max(0.0, now_mono - self.last_tick_mono)
        self.last_tick_mono = now_mono
        self.total_seconds = now_mono - self.start_mono

        has_fix = bool(gnss_snap.get("fix", False))
        data_received = bool(gnss_snap.get("data_received", False))
        status = gnss_snap.get("status", "NO_DATA")

        # Track durations
        if has_fix:
            self.fix_seconds += dt
            self.current_fix_streak += dt
            self.current_loss_streak = 0.0
        elif data_received:
            self.no_fix_seconds += dt
            self.current_loss_streak += dt
            self.current_fix_streak = 0.0
        else:
            self.no_data_seconds += dt
            self.current_loss_streak += dt
            self.current_fix_streak = 0.0

        if self.current_loss_streak > self.max_outage_sec:
            self.max_outage_sec = self.current_loss_streak

        # Track state transitions
        if self.prev_has_fix and not has_fix:
            # Drop event occurred!
            self.fix_drop_count += 1
            drop_rec = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "runtime_sec": round(self.total_seconds, 1),
                "lat": gnss_snap.get("latitude"),
                "lon": gnss_snap.get("longitude"),
                "sats": gnss_snap.get("satellites", 0),
                "cpu_pct": metrics.get("cpu_percent", 0.0),
                "reason": "NO_FIX (Searching satellites)" if data_received else "NO_DATA (UART timeout)",
            }
            self.drop_events.append(drop_rec)
        elif not self.prev_has_fix and has_fix:
            self.fix_recovery_count += 1

        self.prev_has_fix = has_fix

        # Resource samples
        if metrics.get("cpu_percent") is not None:
            self.cpu_samples.append(metrics["cpu_percent"])
        if metrics.get("ram_used_mb") is not None:
            self.ram_samples.append(metrics["ram_used_mb"])
        if metrics.get("cpu_temp_c") is not None:
            self.temp_samples.append(metrics["cpu_temp_c"])

        # CSV Logging
        if self.csv_writer and self.csv_file:
            try:
                self.csv_writer.writerow([
                    datetime.datetime.utcnow().isoformat() + "Z",
                    round(self.total_seconds, 2),
                    1 if has_fix else 0,
                    status,
                    gnss_snap.get("latitude") or "",
                    gnss_snap.get("longitude") or "",
                    gnss_snap.get("altitude_m") or "",
                    gnss_snap.get("speed_kmh") or "",
                    gnss_snap.get("satellites", 0),
                    gnss_snap.get("hdop") or "",
                    metrics.get("cpu_percent", ""),
                    metrics.get("ram_used_mb", ""),
                    metrics.get("cpu_temp_c", ""),
                    telemetry_count,
                    self.fix_drop_count,
                ])
                self.csv_file.flush()
            except Exception:
                pass

        fix_pct = (self.fix_seconds / self.total_seconds * 100.0) if self.total_seconds > 0 else 0.0
        return {
            "runtime_sec": self.total_seconds,
            "fix_pct": fix_pct,
            "has_fix": has_fix,
            "status": status,
            "drop_count": self.fix_drop_count,
            "recovery_count": self.fix_recovery_count,
            "current_streak_sec": self.current_fix_streak if has_fix else self.current_loss_streak,
            "max_outage_sec": self.max_outage_sec,
        }

    def close(self):
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None


# ---------------------------------------------------------------------------
# HUD Overlay Generator (Lightweight Diagnostic Badge on Camera Stream)
# ---------------------------------------------------------------------------
def draw_diagnostic_overlay(frame: np.ndarray, gnss_snap: dict, metrics: dict,
                            stream_fps: float, telemetry_count: int,
                            tracker_summary: dict) -> np.ndarray:
    """Renders a low-overhead diagnostic overlay onto the camera frame before streaming."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Semi-transparent top banner
    banner_h = min(110, int(h * 0.28))
    overlay_bar = out[:banner_h, :]
    cv2.rectangle(out, (0, 0), (w, banner_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay_bar, 0.25, out[:banner_h, :], 0.75, 0, out[:banner_h, :])

    # Status badge color
    has_fix = gnss_snap.get("fix", False)
    data_rcv = gnss_snap.get("data_received", False)
    if has_fix:
        badge_text = "GNSS: 3D FIX"
        badge_color = (0, 255, 120)  # Bright Green
    elif data_rcv:
        badge_text = "GNSS: SEARCHING (NO FIX)"
        badge_color = (0, 215, 255)  # Amber / Yellow
    else:
        badge_text = "GNSS: DISCONNECTED (NO DATA)"
        badge_color = (60, 60, 255)  # Red

    # Header
    cv2.putText(out, "SWSTP STREAM TEST | CAMERA + LOCATION ONLY (ZERO MOTION OVERHEAD)",
                (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

    # Line 1: GNSS Status & Fix Ratio
    lat = gnss_snap.get("latitude")
    lon = gnss_snap.get("longitude")
    sats = gnss_snap.get("satellites", 0)
    hdop = gnss_snap.get("hdop")
    hdop_str = f"{hdop:.1f}" if hdop is not None else "---"
    fix_pct = tracker_summary.get("fix_pct", 0.0)
    drops = tracker_summary.get("drop_count", 0)

    l1 = f"[{badge_text}] Sats: {sats} | HDOP: {hdop_str} | Fix Ratio: {fix_pct:.1f}% (Drops: {drops})"
    cv2.putText(out, l1, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, badge_color, 1, cv2.LINE_AA)

    # Line 2: Coordinates & Speed
    if lat is not None and lon is not None:
        spd = gnss_snap.get("speed_kmh") or 0.0
        alt = gnss_snap.get("altitude_m") or 0.0
        l2 = f"Coord: {lat:.6f}, {lon:.6f} | Alt: {alt:.1f}m | Speed: {spd:.1f} km/h"
    else:
        l2 = "Coord: Acquiring satellite lock... (No coordinate output)"
    cv2.putText(out, l2, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Line 3: System CPU & Streaming Delivery
    cpu_pct = metrics.get("cpu_percent", 0.0)
    temp_c = metrics.get("cpu_temp_c")
    temp_str = f" | Temp: {temp_c:.1f}C" if temp_c is not None else ""
    ram_mb = metrics.get("ram_used_mb", 0.0)
    l3 = f"CPU: {cpu_pct:.1f}%{temp_str} | RAM: {ram_mb:.0f}MB | Cam: {stream_fps:.1f} FPS | Ingest Pkts: {telemetry_count}"
    cv2.putText(out, l3, (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 255), 1, cv2.LINE_AA)

    # Dotted bottom border of banner
    cv2.line(out, (0, banner_h), (w, banner_h), (60, 60, 70), 1)
    return out


# ---------------------------------------------------------------------------
# Terminal Dashboard Renderer (Interactive SSH / Console)
# ---------------------------------------------------------------------------
_dash_first_render = True


def render_terminal_dashboard(gnss_snap: dict, metrics: dict, tracker_summary: dict,
                              telemetry_count: int, cam_info: dict, backend_url: str,
                              device_id: str, stream_fps: float) -> None:
    global _dash_first_render

    bold = "\033[1m"
    cyan = "\033[96m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    gray = "\033[90m"
    magenta = "\033[95m"
    reset = "\033[0m"
    clr = "\033[K"

    has_fix = tracker_summary.get("has_fix", False)
    status = tracker_summary.get("status", "NO_DATA")
    if has_fix:
        badge = f"{green}{bold}[ 3D FIX ACQUIRED ]{reset}"
    elif status == "NO_FIX":
        badge = f"{yellow}{bold}[ SEARCHING / NO FIX ]{reset}"
    else:
        badge = f"{red}{bold}[ DISCONNECTED / NO DATA ]{reset}"

    run_sec = tracker_summary.get("runtime_sec", 0.0)
    m, s = divmod(int(run_sec), 60)
    h, m = divmod(m, 60)
    dur_str = f"{h:02d}:{m:02d}:{s:02d}"

    lat = gnss_snap.get("latitude")
    lon = gnss_snap.get("longitude")
    coord_str = f"{lat:.7f}°, {lon:.7f}°" if (lat and lon) else "Waiting for fix..."
    sats = gnss_snap.get("satellites", 0)
    sats_view = gnss_snap.get("satellites_in_view", 0)
    hdop = gnss_snap.get("hdop")
    hdop_str = f"{hdop:.2f}" if hdop is not None else "---"

    cpu = metrics.get("cpu_percent", 0.0)
    ram = metrics.get("ram_used_mb", 0.0)
    temp = metrics.get("cpu_temp_c")
    temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"

    fix_pct = tracker_summary.get("fix_pct", 0.0)
    drops = tracker_summary.get("drop_count", 0)
    max_outage = tracker_summary.get("max_outage_sec", 0.0)
    streak = tracker_summary.get("current_streak_sec", 0.0)

    if _dash_first_render:
        sys.stdout.write("\033[2J\033[H")
        _dash_first_render = False
    else:
        sys.stdout.write("\033[H")

    out_lines = [
        f"{bold}{cyan}╔══════════════════════════════════════════════════════════════════════════════╗{reset}{clr}",
        f"{bold}{cyan}║    SWSTP STREAM ISOLATION TEST: CAMERA + LOCATION ONLY (ZERO MOTION)        ║{reset}{clr}",
        f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════════════╝{reset}{clr}",
        f" Runtime: {bold}{dur_str}{reset} | Device: {bold}{device_id}{reset} | Status: {badge}{clr}",
        f"{gray}──────────────────────────────────────────────────────────────────────────────{reset}{clr}",
        f"{bold}>> GNSS FIX STABILITY ANALYSIS:{reset}{clr}",
        f"   Fix Ratio            : {green if fix_pct > 95 else yellow}{bold}{fix_pct:.1f}%{reset} of test time in continuous fix{clr}",
        f"   Fix Losses (Drops)   : {red if drops > 0 else green}{bold}{drops} event(s){reset}  (transitions from FIX -> NO_FIX/NO_DATA){clr}",
        f"   Longest Outage       : {yellow if max_outage > 0 else green}{max_outage:.1f} s{reset} | Current State Streak: {streak:.1f} s{clr}",
        f"   Coordinates          : {bold}{green if has_fix else gray}{coord_str}{reset}{clr}",
        f"   Satellites (Used/View): {bold}{sats}{reset} in fix / {sats_view} in view | HDOP: {bold}{hdop_str}{reset}{clr}",
        f"   Speed / Altitude     : {gnss_snap.get('speed_kmh', 0.0) or 0.0:.1f} km/h | {gnss_snap.get('altitude_m', 0.0) or 0.0:.1f} m{clr}",
        f"{gray}──────────────────────────────────────────────────────────────────────────────{reset}{clr}",
        f"{bold}>> RESOURCE CONSUMPTION (Without Motion Capture):{reset}{clr}",
        f"   System CPU Load      : {green if cpu < 30 else (yellow if cpu < 60 else red)}{bold}{cpu:.1f}%{reset} (Contrast with 80-100% in main.py){clr}",
        f"   System RAM Used      : {ram:.1f} MB | CPU Temperature: {temp_str}{clr}",
        f"{gray}──────────────────────────────────────────────────────────────────────────────{reset}{clr}",
        f"{bold}>> ACTIVE NETWORK STREAM TRANSMISSIONS:{reset}{clr}",
        f"   1. Video Stream API  : {cyan}POST /api/camera/{device_id}/frame{reset}{clr}",
        f"      Camera Source     : {cam_info.get('model', 'Camera')} ({cam_info.get('resolution', 'N/A')}) @ {stream_fps:.1f} FPS{clr}",
        f"   2. Location Stream   : {cyan}POST /api/telemetry/ingest-batch{reset} (Batched 1 Hz){clr}",
        f"      Telemetry Packets : {bold}{telemetry_count}{reset} packets transmitted{clr}",
        f"      Backend Endpoint  : {backend_url}{clr}",
        f"{gray}──────────────────────────────────────────────────────────────────────────────{reset}{clr}",
        f" [Ctrl+C] to stop test and generate comprehensive diagnostic verdict{clr}",
    ]

    sys.stdout.write("\n".join(out_lines) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SWSTP Stream Isolation Diagnostic Tool (Raspberry Pi 4B): "
                    "Runs Camera Stream + Ingest Batch Location Stream with ZERO motion capture overhead."
    )
    parser.add_argument("--source", "-s", default=None,
                        help="Video source: 'imx708' / 'picam' (Camera Module 3), V4L2 device (/dev/video0), or index (default: auto)")
    parser.add_argument("--af-mode", default="continuous", choices=["continuous", "auto", "manual"],
                        help="Autofocus mode for IMX708 Camera Module 3 (default: continuous)")
    parser.add_argument("--awb-mode", default=getattr(_cfg, "CAMERA_AWB_MODE", "auto"),
                        choices=["auto", "daylight", "cloudy", "indoor", "tungsten", "fluorescent"],
                        help="White balance mode (default: auto)")
    parser.add_argument("--hdr", action="store_true",
                        help="Enable hardware HDR mode on Sony IMX708 sensor")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL,
                        help="SWSTP Backend Base URL")
    parser.add_argument("--device-id", default=None,
                        help="Device Hardware Code (overrides device_config.json)")
    parser.add_argument("--vehicle-id", default=None,
                        help="Vehicle Registration / ID (overrides device_config.json)")
    parser.add_argument("--ulb-id", default=DEFAULT_ULB_ID,
                        help="ULB Identifier")
    parser.add_argument("--session-id", type=int, default=0,
                        help="Operational Session ID (0 = auto-authenticate with backend)")
    parser.add_argument("--gnss-port", "-p", default=getattr(_cfg, "GNSS_SERIAL_PORT", "/dev/serial0"),
                        help="Serial port for u-blox MAX-M10S primary GNSS (default: /dev/serial0 or auto-detect)")
    parser.add_argument("--gnss-baud", "-b", type=int, default=getattr(_cfg, "GNSS_BAUD_RATE", 9600),
                        help="Baud rate for GNSS (default: 9600)")
    parser.add_argument("--fps-stream", type=float, default=DEFAULT_STREAM_FPS,
                        help="Live camera streaming FPS to backend (default: 3.0)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Run test for N seconds, then stop and print report (0 = run indefinitely until Ctrl+C)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without cv2.imshow GUI window (auto-enabled if no display)")
    parser.add_argument("--clean-stream", action="store_true",
                        help="Stream pure raw camera frames without diagnostic HUD overlay")
    parser.add_argument("--stream-logs", action="store_true",
                        help="Output single-line periodic logs instead of interactive ANSI dashboard (ideal for systemd/journalctl)")
    parser.add_argument("--log-csv", default=None,
                        help="Path to CSV file to record second-by-second GNSS stability and CPU metrics")
    parser.add_argument("--force-auth", action="store_true",
                        help="Force session authentication for bench testing without reaching backend")
    parser.add_argument("--self-test", action="store_true",
                        help="Execute internal automated test with mock backend & simulated GNSS feed")
    return parser


# ---------------------------------------------------------------------------
# Automated Self-Test (Mock Verification)
# ---------------------------------------------------------------------------
class MockBackendHandler(http.server.BaseHTTPRequestHandler):
    received_frames = 0
    received_telemetry = 0
    lock = threading.Lock()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""
        with self.lock:
            if "/api/camera/" in self.path:
                MockBackendHandler.received_frames += 1
            elif "/api/telemetry/ingest-batch" in self.path:
                MockBackendHandler.received_telemetry += 1

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok","sessionId":999}')

    def log_message(self, format, *args):
        return  # Silence server logs during self-test


def run_self_test() -> int:
    """Automated verification test using simulated camera, mock HTTP endpoints, and NMEA dispatch."""
    print("=" * 70)
    print("  RUNNING SELF-TEST: CAMERA STREAM + INGEST BATCH LOCATION ISOLATION")
    print("=" * 70)

    # 1. Start Mock HTTP Server (with graceful in-memory fallback for sandboxed environments)
    server = None
    server_url = "http://127.0.0.1:58999"
    orig_session_post = None

    try:
        server = http.server.HTTPServer(("127.0.0.1", 0), MockBackendHandler)
        server_port = server.server_port
        server_url = f"http://127.0.0.1:{server_port}"
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        # Probe socket connectivity (handles environments where socket listen succeeds but connect is sandboxed)
        requests.post(f"{server_url}/probe", timeout=0.3)
        print(f"[SELF-TEST] Mock Backend socket listening on: {server_url}")
    except (OSError, PermissionError, requests.exceptions.RequestException) as sock_err:
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
            server = None
        print(f"[SELF-TEST] Socket listen restricted ({sock_err}). Using in-memory mock transport.")
        orig_session_post = requests.Session.post

        def _mock_post(self, url, **kwargs):
            with MockBackendHandler.lock:
                if "/api/camera/" in url:
                    MockBackendHandler.received_frames += 1
                elif "/api/telemetry/ingest-batch" in url:
                    MockBackendHandler.received_telemetry += 1
            resp = requests.Response()
            resp.status_code = 200
            resp._content = b'{"status":"ok","sessionId":999}'
            return resp

        requests.Session.post = _mock_post

    try:
        # Set authentication
        dynamic_session_info["authenticated"] = True
        dynamic_session_info["sessionId"] = 999
        dynamic_session_info["deviceId"] = "SELFTEST-DEV-01"

        # 2. Feed simulated GNSS fix
        def make_nmea(body: str) -> str:
            c = 0
            for ch in body:
                c ^= ord(ch)
            return f"${body}*{c:02X}"

        gga = make_nmea("GNGGA,120000.00,2055.9200,N,07745.1380,E,1,08,0.9,340.0,M,0.0,M,,")
        rmc = make_nmea("GNRMC,120000.00,A,2055.9200,N,07745.1380,E,0.0,0.0,031126,,,A")
        gnss_sensor._dispatch(gga)
        gnss_sensor._dispatch(rmc)

        snap = gnss_sensor.read()
        assert snap["fix"] is True, "GNSS simulated fix must be True"
        print(f"[SELF-TEST] GNSS mock fix: {snap['latitude']:.4f}, {snap['longitude']:.4f}")

        # 3. Start Telemetry Loop and Telemetry Streamer
        test_stop = threading.Event()
        t_telemetry = _telemetry.start("SELFTEST-DEV-01")
        t_http = threading.Thread(
            target=telemetry_streamer,
            args=(server_url, 999, DEFAULT_ULB_ID, test_stop),
            daemon=True,
        )
        t_http.start()

        # 4. Start Frame Streamer with simulated frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "SELF-TEST SYNTHETIC FRAME", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        shared_frame_lock = threading.Lock()
        shared_frame = (dummy_frame, time.monotonic())

        def _get_test_frame():
            return shared_frame

        t_frame = threading.Thread(
            target=live_frame_streamer,
            args=(server_url, "SELFTEST-DEV-01", 10.0, test_stop, shared_frame_lock, _get_test_frame),
            daemon=True,
        )
        t_frame.start()

        # 5. Let it run for 2.5 seconds to transmit frames and batch telemetry
        time.sleep(2.5)

        # 6. Test Tracker & Metrics
        tracker = FixStabilityTracker()
        metrics = get_system_metrics()
        summary = tracker.update(snap, metrics, MockBackendHandler.received_telemetry)
        assert summary["has_fix"] is True

        # 7. Simulate a GNSS drop to verify tracker
        gnss_sensor._reset_state_to_disconnected()
        drop_snap = gnss_sensor.read()
        summary_drop = tracker.update(drop_snap, metrics, MockBackendHandler.received_telemetry)
        assert summary_drop["drop_count"] == 1, "Tracker must record drop event"
        print(f"[SELF-TEST] Drop event correctly tracked: {tracker.drop_events[0]}")

        # Cleanup worker threads
        test_stop.set()
        _telemetry.stop()
        tracker.close()

        print(f"[SELF-TEST] Frames received by Mock Backend      : {MockBackendHandler.received_frames}")
        print(f"[SELF-TEST] Ingest batches received by Mock Backend: {MockBackendHandler.received_telemetry}")
        assert MockBackendHandler.received_frames > 0, "Frames must be successfully posted"
        assert MockBackendHandler.received_telemetry > 0, "Telemetry batches must be successfully posted"

    finally:
        if orig_session_post is not None:
            requests.Session.post = orig_session_post
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass

    print("=" * 70)
    print("  SELF-TEST COMPLETED: ALL STREAM APIS & FIX TRACKERS VERIFIED [PASS]")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------
def main() -> None:
    global latest_stream_frame, stop_event

    parser = build_parser()
    args, _ = parser.parse_known_args()

    if args.self_test:
        sys.exit(run_self_test())

    # Signal handlers for clean exit
    def _sig_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Resolve device and vehicle identity
    device_id = (args.device_id or _cfg.load_device_id()).strip()
    if not device_id or device_id in ("", "UNPROVISIONED", "UNASSIGNED"):
        device_id = "SWSTP-TEST-DEV"

    vehicle_id = (args.vehicle_id or getattr(_cfg, "load_vehicle_id", lambda: device_id)()).strip()
    if not vehicle_id:
        vehicle_id = device_id

    effective_ulb_id = args.ulb_id if (args.ulb_id and args.ulb_id != "AUTO") else DEFAULT_ULB_ID
    effective_backend_url = args.backend_url or DEFAULT_BACKEND_URL
    effective_session_id = args.session_id

    dynamic_session_info["deviceId"] = device_id
    dynamic_session_info["vehicleId"] = vehicle_id
    dynamic_session_info["backendUrl"] = effective_backend_url
    dynamic_session_info["ulbId"] = effective_ulb_id
    latest_sensor["deviceId"] = device_id
    latest_sensor["vehicleId"] = vehicle_id

    # ── Non-blocking asynchronous backend session handshake ─────────────
    def _async_backend_init():
        nonlocal effective_session_id, effective_backend_url
        if args.force_auth:
            dynamic_session_info["authenticated"] = True
            dynamic_session_info["sessionId"] = effective_session_id or 1
            latest_sensor["active_session_id"] = dynamic_session_info["sessionId"]
            hardware_state["backend"]["active_session_id"] = dynamic_session_info["sessionId"]
            print("[AUTH] --force-auth enabled: Local authentication forced for bench testing.")
            return

        try:
            detected_url = auto_detect_backend_url(effective_backend_url)
            if detected_url and detected_url != effective_backend_url:
                effective_backend_url = detected_url
                dynamic_session_info["backendUrl"] = detected_url
            sync_backend_metadata(effective_backend_url, effective_ulb_id)
            if device_id and device_id not in ("UNPROVISIONED", "UNASSIGNED"):
                sid = ensure_hardware_session(effective_backend_url, device_id)
                if sid > 0:
                    effective_session_id = sid
                    dynamic_session_info["active_session_id"] = sid
                    hardware_state["backend"]["active_session_id"] = sid
                    latest_sensor["active_session_id"] = sid
                else:
                    if effective_session_id > 0:
                        dynamic_session_info["sessionId"] = effective_session_id
                        dynamic_session_info["authenticated"] = True
            elif effective_session_id > 0:
                dynamic_session_info["sessionId"] = effective_session_id
                dynamic_session_info["authenticated"] = True
        except Exception as b_err:
            print(f"[BACKEND INIT NOTE] Async sync: {b_err}")
            if effective_session_id > 0:
                dynamic_session_info["sessionId"] = effective_session_id
                dynamic_session_info["authenticated"] = True

    threading.Thread(target=_async_backend_init, daemon=True, name="backend-init").start()

    # ── Sensor Initialisation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SWSTP STREAM ISOLATION TEST: CAMERA + LOCATION ONLY")
    print("=" * 70)
    print("  Hypothesis: Testing if GNSS fix loss occurs WITHOUT motion capture.")
    print("  Motion Detection: DISABLED")
    print("  YOLO Inference  : DISABLED")
    print("  Stop Captures   : DISABLED")
    print("  Active Streams  : Camera Stream + Location Ingest Batch Stream")
    print("=" * 70)

    rtc_ok = rtc_sensor.init()
    imu_ok = imu_sensor.init()
    target_gnss_p = args.gnss_port
    if target_gnss_p and target_gnss_p.startswith("/dev") and not os.path.exists(target_gnss_p):
        target_gnss_p = None
    target_gnss_b = args.gnss_baud
    gnss_ok = gnss_sensor.init(port=target_gnss_p, baud=target_gnss_b)

    try:
        leds.init()
        if args.headless:
            leds.set_headless_mode(True)
    except Exception:
        pass

    # ── Camera Initialisation ─────────────────────────────────────────────
    initial_source = args.source if args.source is not None else getattr(_cfg, "CAMERA_SOURCE_DEFAULT", "auto")
    if str(initial_source).isdigit():
        initial_source = int(initial_source)
    elif str(initial_source).strip() in ("", "auto", "None"):
        initial_source = None

    target_cam_w = getattr(_cfg, "CAMERA_WIDTH", 640)
    target_cam_h = getattr(_cfg, "CAMERA_HEIGHT", 480)
    target_cam_fps = getattr(_cfg, "CAMERA_FPS", 30.0)
    prefer_imx = getattr(_cfg, "CAMERA_PREFER_IMX708", True)

    candidate_ports = get_candidate_video_ports(initial_source, prefer_imx708=prefer_imx)
    cap, source, cam_w, cam_h, fps, cam_info = scan_for_camera(
        candidate_ports,
        target_w=target_cam_w,
        target_h=target_cam_h,
        target_fps=target_cam_fps,
        af_mode=args.af_mode,
        hdr=args.hdr,
        awb_mode=args.awb_mode,
    )

    if cap is not None:
        cam_model = cam_info.get("model", "Camera Module")
        print(f"[CAMERA] Active: {cam_model} on '{source}' ({cam_w}x{cam_h} @ {fps:.1f} FPS)")
    else:
        cam_model = "None"
        source = initial_source or 0
        cam_w, cam_h, fps = target_cam_w, target_cam_h, target_cam_fps
        print(f"[CAMERA] Warning: No camera found on {candidate_ports}. Video stream will show standby frame.")

    # ── Worker Threads ────────────────────────────────────────────────────
    # 1. 20 Hz Telemetry Loop
    t_telemetry_loop = _telemetry.start(device_id)

    # 2. Camera Frame Streamer (POST /api/camera/{deviceId}/frame)
    def _get_latest_frame():
        return latest_stream_frame

    t_frame_streamer = threading.Thread(
        target=live_frame_streamer,
        args=(effective_backend_url, device_id, args.fps_stream, stop_event,
              latest_frame_lock, _get_latest_frame),
        daemon=True, name="frame-streamer",
    )
    t_frame_streamer.start()

    # 3. Telemetry Ingest Batch Streamer (POST /api/telemetry/ingest-batch)
    t_telemetry_http = threading.Thread(
        target=telemetry_streamer,
        args=(effective_backend_url, effective_session_id, effective_ulb_id, stop_event),
        daemon=True, name="telemetry-http",
    )
    t_telemetry_http.start()

    # ── Stability Tracker & Dashboard Setup ──────────────────────────────
    tracker = FixStabilityTracker(log_csv_path=args.log_csv)
    WIN_TITLE = "SWSTP Stream Isolation Diagnostic (Camera + Location)"
    if not args.headless:
        try:
            cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_TITLE, 800, 480)
        except Exception:
            args.headless = True

    test_start_time = time.monotonic()
    last_dash_time = 0.0
    frames_captured = 0

    print("\n[TEST RUNNING] Press Ctrl+C or 'q' in window to finish test.\n")

    try:
        while not stop_event.is_set():
            loop_now = time.monotonic()

            # Duration check
            if args.duration > 0 and (loop_now - test_start_time) >= args.duration:
                print(f"\n[TEST] Requested duration ({args.duration:.1f}s) reached.")
                break

            # ── 1. Acquire Camera Frame ──────────────────────────────────
            frame = None
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    frame = None
                else:
                    frames_captured += 1

            if frame is None:
                # Standby test frame
                frame = np.zeros((cam_h or 480, cam_w or 640, 3), dtype=np.uint8)
                sh, sw = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (sw, sh), (20, 22, 26), -1)
                cv2.putText(frame, "STANDBY / SCANNING CAMERA", (max(10, sw // 2 - 160), sh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

            # ── 2. Query Sensors & Metrics ───────────────────────────────
            gnss_snap = gnss_sensor.read()
            metrics = get_system_metrics()
            telemetry_count = hardware_state.get("backend", {}).get("telemetry_count", 0)

            # ── 3. Update Fix Stability Tracker ──────────────────────────
            tracker_summary = tracker.update(gnss_snap, metrics, telemetry_count)

            # ── 4. Prepare Stream Frame (With or without Diagnostic HUD) ──
            if args.clean_stream:
                stream_frame = frame
            else:
                stream_frame = draw_diagnostic_overlay(
                    frame, gnss_snap, metrics, args.fps_stream, telemetry_count, tracker_summary
                )

            with latest_frame_lock:
                latest_stream_frame = (stream_frame, time.monotonic())

            # ── 5. Render Dashboard or Streaming Logs ────────────────────
            if loop_now - last_dash_time >= 1.0:
                last_dash_time = loop_now
                if args.stream_logs or not sys.stdout.isatty():
                    # Compact one-line periodic log (perfect for SSH/systemd/journalctl)
                    lat_s = f"{gnss_snap['latitude']:.5f}" if gnss_snap.get("latitude") else "None"
                    lon_s = f"{gnss_snap['longitude']:.5f}" if gnss_snap.get("longitude") else "None"
                    fix_s = "FIX" if tracker_summary["has_fix"] else gnss_snap.get("status", "NO_FIX")
                    print(
                        f"[DIAG] T+{tracker_summary['runtime_sec']:05.1f}s | GNSS: {fix_s:<7} "
                        f"| Sats: {gnss_snap.get('satellites', 0):02d} | Coord: {lat_s},{lon_s} "
                        f"| Fix: {tracker_summary['fix_pct']:5.1f}% (Drops: {tracker_summary['drop_count']}) "
                        f"| CPU: {metrics['cpu_percent']:4.1f}% | Ingest Pkts: {telemetry_count}"
                    )
                    sys.stdout.flush()
                else:
                    render_terminal_dashboard(
                        gnss_snap, metrics, tracker_summary, telemetry_count,
                        cam_info, effective_backend_url, device_id, args.fps_stream,
                    )

            # ── 6. GUI Window Handling ───────────────────────────────────
            if not args.headless:
                cv2.imshow(WIN_TITLE, stream_frame)
                k = cv2.waitKey(max(1, int(1000.0 / max(1.0, fps)))) & 0xFF
                if k == ord('q'):
                    print("\n[USER] 'q' pressed. Exiting...")
                    break
            else:
                # Regulate loop cadence to camera FPS
                time.sleep(max(0.01, 1.0 / max(5.0, fps)))

    except KeyboardInterrupt:
        print("\n[USER] KeyboardInterrupt (Ctrl+C). Terminating...")
    finally:
        stop_event.set()
        _telemetry.stop()
        gnss_sensor.stop()
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        tracker.close()

        # ── 7. Generate Comprehensive Diagnostic Verdict Report ───────────
        print_diagnostic_report(
            tracker, metrics, device_id, vehicle_id, effective_backend_url,
            cam_model, source, cam_w, cam_h, fps, args.fps_stream,
            target_gnss_p, target_gnss_b,
        )


def print_diagnostic_report(tracker: FixStabilityTracker, last_metrics: dict,
                            device_id: str, vehicle_id: str, backend_url: str,
                            cam_model: str, cam_source, cam_w: int, cam_h: int,
                            cam_fps: float, stream_fps: float,
                            gnss_port: str, gnss_baud: int) -> None:
    """Prints a clear, rigorous summary report answering the user's hypothesis."""
    total_sec = tracker.total_seconds
    fix_sec = tracker.fix_seconds
    no_fix_sec = tracker.no_fix_seconds
    no_data_sec = tracker.no_data_seconds
    drops = tracker.fix_drop_count
    max_outage = tracker.max_outage_sec

    fix_pct = (fix_sec / total_sec * 100.0) if total_sec > 0 else 0.0
    no_fix_pct = (no_fix_sec / total_sec * 100.0) if total_sec > 0 else 0.0
    no_data_pct = (no_data_sec / total_sec * 100.0) if total_sec > 0 else 0.0

    avg_cpu = (sum(tracker.cpu_samples) / len(tracker.cpu_samples)) if tracker.cpu_samples else 0.0
    peak_cpu = max(tracker.cpu_samples) if tracker.cpu_samples else 0.0
    avg_ram = (sum(tracker.ram_samples) / len(tracker.ram_samples)) if tracker.ram_samples else 0.0
    peak_temp = max(tracker.temp_samples) if tracker.temp_samples else None

    snap = gnss_sensor.read()
    final_sats = snap.get("satellites", 0)
    final_sats_view = snap.get("satellites_in_view", 0)
    final_hdop = snap.get("hdop")
    hdop_str = f"{final_hdop:.2f}" if final_hdop is not None else "N/A"
    telemetry_count = hardware_state.get("backend", {}).get("telemetry_count", 0)

    m, s = divmod(int(total_sec), 60)
    h, m = divmod(m, 60)
    dur_str = f"{h:02d}:{m:02d}:{s:02d}"

    print("\n" + "=" * 80)
    print("           SWSTP STREAM ISOLATION TEST: DIAGNOSTIC SUMMARY REPORT")
    print("=" * 80)
    print(" TEST OBJECTIVE:")
    print("   Validate if GNSS fix loss is caused by heavy motion capture / YOLO CPU load,")
    print("   or if it is an independent GNSS hardware / antenna fault.")
    print("-" * 80)
    print(" TEST RUN CONFIGURATION:")
    print(f"   • Device ID              : {device_id} (Vehicle: {vehicle_id})")
    print(f"   • Video Stream API       : POST {backend_url.rstrip('/')}/api/camera/{device_id}/frame")
    print(f"   • Location Ingest API    : POST {backend_url.rstrip('/')}/api/telemetry/ingest-batch")
    print(f"   • Motion Capture Pipeline: EXCLUDED (Disabled)")
    print(f"   • YOLO Litter Engine     : EXCLUDED (Disabled)")
    print(f"   • Local Image Writes     : EXCLUDED (Disabled)")
    print(f"   • Camera Sensor          : {cam_model} on '{cam_source}' ({cam_w}x{cam_h} @ {cam_fps:.1f} FPS)")
    print(f"   • GNSS Module            : u-blox MAX-M10S on {gnss_port} @ {gnss_baud} baud")
    print(f"   • Stream Frame Rate      : {stream_fps:.1f} FPS")
    print("-" * 80)
    print(" RUNTIME & DATA TRANSMISSION:")
    print(f"   • Total Test Duration    : {dur_str} ({total_sec:.1f} seconds)")
    print(f"   • Telemetry Packets Sent : {telemetry_count} packets")
    print("-" * 80)
    print(" SYSTEM RESOURCE CONSUMPTION (Baseline without motion capture):")
    print(f"   • Average System CPU     : {avg_cpu:.1f}%")
    print(f"   • Peak System CPU        : {peak_cpu:.1f}%")
    print(f"   • Average RAM Used       : {avg_ram:.1f} MB")
    if peak_temp is not None:
        print(f"   • Peak CPU Temperature   : {peak_temp:.1f}°C")
    print("-" * 80)
    print(" GNSS FIX PERFORMANCE:")
    print(f"   • Total Time in 3D FIX   : {fix_sec:.1f}s ({fix_pct:.1f}%)")
    print(f"   • Time in NO_FIX (Search): {no_fix_sec:.1f}s ({no_fix_pct:.1f}%)")
    print(f"   • Time in NO_DATA (Silent: {no_data_sec:.1f}s ({no_data_pct:.1f}%)")
    print(f"   • Fix Loss Events (Drops): {drops} event(s)")
    print(f"   • Longest Outage Duration: {max_outage:.1f} seconds")
    print(f"   • Final Satellites       : {final_sats} in fix / {final_sats_view} in view (HDOP: {hdop_str})")

    if tracker.drop_events:
        print("\n FIX DROP LOG:")
        for idx, d in enumerate(tracker.drop_events, 1):
            coord_info = f"at ({d['lat']:.5f}, {d['lon']:.5f})" if (d['lat'] and d['lon']) else "No valid coords"
            print(f"   [{idx}] at T+{d['runtime_sec']:.1f}s ({d['timestamp']}) - {d['reason']} | Sats: {d['sats']} | CPU: {d['cpu_pct']:.1f}% | {coord_info}")

    print("=" * 80)
    print(" DIAGNOSTIC VERDICT & ROOT-CAUSE CONCLUSION:")
    print("-" * 80)

    if total_sec < 15.0:
        print(" [!] TEST DURATION TOO SHORT: Ran for less than 15 seconds.")
        print("     Please run the test for at least 5-10 minutes (--duration 300) to allow reliable")
        print("     satellite tracking and statistical isolation.")
    elif drops == 0 and fix_pct >= 90.0:
        print(" [✔] RESULT: ZERO FIX LOSSES OBSERVED WITH MOTION CAPTURE DISABLED.")
        print(f"     Fix remained solid for {fix_pct:.1f}% of the test ({fix_sec:.1f}s) with 0 dropouts.")
        print(f"     System CPU stayed very low (average {avg_cpu:.1f}%, peak {peak_cpu:.1f}%).")
        print("")
        print("     CONCLUSION:")
        print("     The u-blox MAX-M10S GNSS hardware, antenna, and serial UART connection are FUNCTIONAL.")
        print("     The fix losses experienced in main.py are almost certainly caused by thread starvation")
        print("     or CPU / GIL contention from the high resource-consuming motion capture and YOLO")
        print("     litter inference pipeline.")
    elif drops > 0 and avg_cpu < 35.0:
        print(f" [✘] RESULT: {drops} FIX LOSS EVENT(S) OCCURRED EVEN WITHOUT MOTION CAPTURE.")
        print(f"     Total outage duration: {no_fix_sec + no_data_sec:.1f} seconds.")
        print(f"     System CPU load was completely relaxed (average {avg_cpu:.1f}%, peak {peak_cpu:.1f}%).")
        print("")
        print("     CONCLUSION:")
        print("     Since motion capture was completely disabled and CPU load was well below saturation,")
        print("     the fix loss is NOT caused by CPU resource starvation.")
        print("     The root cause is a GNSS hardware or environment fault:")
        if no_data_sec > 5.0:
            print("       - Serial connection dropouts (loose TX/RX wiring, UART buffer overruns, or baud mismatch).")
        else:
            print("       - Antenna obstruction, poor sky view, or multipath interference.")
            print("       - Insufficient satellite signal / antenna cable attenuation.")
            print("       - Power supply dips / ripple on the 3.3V/5V rail affecting the GNSS receiver.")
    elif no_data_sec > 10.0:
        print(" [✘] RESULT: GNSS SERIAL DATA FLOW WAS TIMED OUT OR DISCONNECTED.")
        print(f"     NO_DATA received for {no_data_sec:.1f} seconds on {gnss_port}.")
        print("     Check hardware UART connection (Pin 8/10), GNSS module VCC/GND, and serial port permissions.")
    else:
        print(f" [?] RESULT: FIX RATIO WAS {fix_pct:.1f}% WITH {drops} DROP(S).")
        print("     Review satellite view counts and environment.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
