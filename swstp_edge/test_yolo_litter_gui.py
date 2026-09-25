#!/usr/bin/env python3
"""
test_yolo_litter_gui.py  --  Standalone GUI Test for YOLO Litter Detection Engine
===================================================================================

Tests ONLY the YOLO litter detection model.
No motion detection, no cloud uploads, no sensors/* imports required.

DEFAULT GNSS SOURCE: NavCast TCP (phone NMEA stream via USB tethering).
  Host: 10.208.43.190  Port: 10110  (matches config.py NAVCAST_HOST/PORT)
  Pass --host / --port to override. Use --sim to fall back to simulation.

KEY FEATURES:
  - Live NavCast NMEA parsed in background: GGA, RMC, GSA, GSV with checksum.
  - Auto-reconnect on TCP disconnect.
  - Real GNSS coordinates drive the 5 m distance trigger and TIME_FALLBACK.
  - Saves detected frames locally: JPEG + JSON sidecar with production banner overlay.
  - Software LED panel mirrors all real LED patterns from sensors/leds.py.
  - Live GUI: distance bar, mode badge, satellite count, HDOP, speed, log.

USAGE:
  python test_yolo_litter_gui.py                         # NavCast default host
  python test_yolo_litter_gui.py --host 192.168.x.x     # custom NavCast IP
  python test_yolo_litter_gui.py --sim                   # simulation fallback
  python test_yolo_litter_gui.py --camera 0              # laptop webcam
  python test_yolo_litter_gui.py --mode fallback         # start in clock fallback

REQUIREMENTS:
  pip install opencv-python numpy Pillow
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import scrolledtext, ttk
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure local detector.py is found first
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Ensure local modules (detector.py, sensors/navcast_discovery.py) are found
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from detector import YOLO
    _DETECTOR_AVAILABLE = True
    _DETECTOR_IMPORT_ERROR = ""
except ImportError as _det_err:
    _DETECTOR_AVAILABLE = False
    _DETECTOR_IMPORT_ERROR = str(_det_err)

try:
    from sensors.navcast_discovery import discover_navcast, get_candidate_ips, auto_detect_navcast
except ImportError:
    try:
        sys.path.insert(0, os.path.join(_HERE, "sensors"))
        from navcast_discovery import discover_navcast, get_candidate_ips, auto_detect_navcast
    except ImportError:
        discover_navcast = None
        get_candidate_ips = None
        auto_detect_navcast = None

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# ===========================================================================
# CONFIG CONSTANTS  (mirrors config.py, no import)
# ===========================================================================
LITTER_DISTANCE_INTERVAL_M    = 5.0
LITTER_GNSS_LOST_TIMEOUT_SEC  = 5.0
LITTER_TIME_FALLBACK_SEC      = 30.0
LITTER_CONF_THRESHOLD         = 0.25
GNSS_DATA_TIMEOUT_SEC         = 3.0    # mirrors config.py

# NavCast defaults — "auto" enables automatic network/tether discovery
NAVCAST_HOST_DEFAULT = "auto"
NAVCAST_PORT_DEFAULT = 10110

SIM_SPEED_KMH_DEFAULT     = 20.0
SIM_UPDATE_HZ             = 10
GUI_REFRESH_MS            = 80

DEFAULT_LAT = 20.9374
DEFAULT_LON = 77.7796

_DEFAULT_SAVE_DIR = os.path.join(_HERE, "captures", "litter_test")

# ===========================================================================
# HAVERSINE / COORDINATE UTILS
# ===========================================================================

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def move_coords(lat, lon, bearing_deg, distance_m):
    R = 6_371_000.0
    d = distance_m / R
    b = math.radians(bearing_deg)
    lat1, lon1_ = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
    lon2 = lon1_ + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# ===========================================================================
# SELF-CONTAINED NAVCAST GNSS READER
# Mirrors sensors/gnss.py but requires NO project imports.
# Parses GGA + RMC + GSA + GSV with XOR checksum validation.
# Auto-reconnects on TCP error.
# ===========================================================================

class NavCastGNSSReader:
    """
    Self-contained NavCast TCP NMEA reader.
    Connects to the NavCast Android app over USB tethering TCP and parses
    $GGA / $GNGGA, $RMC / $GNRMC, $GSA / $GNGSA, $GSV / $GNGSV sentences.

    Call get_position() from any thread to get (lat, lon, has_fix).
    Call get_state() for the full parsed state dict.
    """

    _RECONNECT_DELAY = 5.0
    _SOCKET_TIMEOUT  = 10.0
    _RECV_BUF        = 4096

    def __init__(self, host: str = "auto", port: int = NAVCAST_PORT_DEFAULT,
                 log_fn=None, auto_scan: bool = True):
        self._host   = host
        self._port   = port
        self._log    = log_fn or print
        self._auto_scan = auto_scan
        self._needs_scan = (host == "auto" or not host)
        self._rescan_requested = threading.Event()
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        # Internal state (mirrors sensors/gnss.py _state shape)
        self._state = {
            "data_received":      False,
            "fix":                False,
            "status":             "NO_DATA",
            "latitude":           None,
            "longitude":          None,
            "altitude_m":         None,
            "speed_kmh":          None,
            "course_deg":         None,
            "satellites":         0,
            "satellites_in_view": 0,
            "hdop":               None,
            "last_data_time":     None,
            "last_fix_time":      None,
        }
        self.connected  = False
        self.last_error = ""

        self._thread = threading.Thread(
            target=self._reader_thread, daemon=True, name="navcast-gnss"
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def request_rescan(self):
        """Signal the reader thread to perform a fresh discovery scan."""
        self._needs_scan = True
        self._rescan_requested.set()

    @property
    def endpoint(self) -> str:
        return f"{self._host}:{self._port}"

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------
    def get_position(self) -> tuple[float, float, bool]:
        """Returns (lat, lon, has_fix). Thread-safe."""
        with self._lock:
            snap = dict(self._state)
        now       = time.monotonic()
        last_data = snap.get("last_data_time")
        data_ok   = last_data is not None and (now - last_data) < GNSS_DATA_TIMEOUT_SEC
        has_fix   = snap["fix"] and data_ok
        lat = snap["latitude"]  if has_fix else None
        lon = snap["longitude"] if has_fix else None
        return lat, lon, has_fix

    def get_state(self) -> dict:
        """Returns full snapshot dict. Thread-safe."""
        with self._lock:
            snap = dict(self._state)
        now       = time.monotonic()
        last_data = snap.get("last_data_time")
        data_ok   = last_data is not None and (now - last_data) < GNSS_DATA_TIMEOUT_SEC
        has_fix   = snap["fix"] and data_ok
        status    = "FIX" if has_fix else ("NO_FIX" if data_ok else "NO_DATA")
        return {
            "data_received":      data_ok,
            "fix":                has_fix,
            "status":             status,
            "latitude":           snap["latitude"]  if has_fix else None,
            "longitude":          snap["longitude"] if has_fix else None,
            "altitude_m":         snap["altitude_m"],
            "speed_kmh":          snap["speed_kmh"] if has_fix else None,
            "course_deg":         snap["course_deg"],
            "satellites":         snap["satellites"],
            "satellites_in_view": snap["satellites_in_view"],
            "hdop":               snap["hdop"],
        }

    # -----------------------------------------------------------------------
    # NMEA checksum validation
    # -----------------------------------------------------------------------
    @staticmethod
    def _checksum_ok(sentence: str) -> bool:
        if "*" not in sentence:
            return True  # no checksum field — accept
        try:
            body, cs = sentence.rsplit("*", 1)
            body = body.lstrip("$").lstrip("!")
            computed = 0
            for ch in body:
                computed ^= ord(ch)
            return computed == int(cs[:2], 16)
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Coordinate parsing
    # -----------------------------------------------------------------------
    @staticmethod
    def _parse_lat(raw: str, hemi: str) -> Optional[float]:
        if not raw or not hemi:
            return None
        try:
            raw = raw.strip()
            dot = raw.index(".")
            val = float(raw[:dot - 2]) + float(raw[dot - 2:]) / 60.0
            return round(-val if hemi.upper() == "S" else val, 8)
        except Exception:
            return None

    @staticmethod
    def _parse_lon(raw: str, hemi: str) -> Optional[float]:
        if not raw or not hemi:
            return None
        try:
            raw = raw.strip()
            dot = raw.index(".")
            val = float(raw[:dot - 2]) + float(raw[dot - 2:]) / 60.0
            return round(-val if hemi.upper() == "W" else val, 8)
        except Exception:
            return None

    @staticmethod
    def _sf(s: str) -> Optional[float]:
        try:
            return float(s.strip()) if s and s.strip() else None
        except Exception:
            return None

    @staticmethod
    def _si(s: str) -> int:
        try:
            return int(s.strip()) if s and s.strip() else 0
        except Exception:
            return 0

    # -----------------------------------------------------------------------
    # NMEA sentence parsers  (mirrors sensors/gnss.py parsers exactly)
    # -----------------------------------------------------------------------
    def _parse_gga(self, f: list):
        # $--GGA,time,lat,N/S,lon,E/W,quality,numSV,HDOP,alt,...
        if len(f) < 10:
            return
        quality = self._si(f[6])
        has_fix = quality > 0
        lat = self._parse_lat(f[2], f[3])
        lon = self._parse_lon(f[4], f[5])
        alt  = self._sf(f[9])
        hdop = self._sf(f[8]) if len(f) > 8 else None
        nsats = self._si(f[7])
        now = time.monotonic()
        with self._lock:
            self._state["data_received"]  = True
            self._state["last_data_time"] = now
            if has_fix and lat is not None and lon is not None:
                self._state["fix"]        = True
                self._state["status"]     = "FIX"
                self._state["latitude"]   = lat
                self._state["longitude"]  = lon
                self._state["altitude_m"] = round(alt, 2) if alt is not None else None
                self._state["satellites"] = nsats
                self._state["hdop"]       = round(hdop, 2) if hdop is not None else None
                self._state["last_fix_time"] = now
            elif not self._state["fix"]:
                self._state["status"] = "NO_FIX"

    def _parse_rmc(self, f: list):
        # $--RMC,time,status,lat,N/S,lon,E/W,speed_kn,course,...
        if len(f) < 3:
            return
        now = time.monotonic()
        with self._lock:
            self._state["data_received"]  = True
            self._state["last_data_time"] = now
        active = f[2].strip().upper() == "A"
        if not active or len(f) < 9:
            return
        lat      = self._parse_lat(f[3], f[4])
        lon      = self._parse_lon(f[5], f[6])
        speed_kn = self._sf(f[7])
        course   = self._sf(f[8])
        speed_kmh = round(speed_kn * 1.852, 2) if speed_kn is not None else None
        with self._lock:
            if lat is not None and lon is not None:
                self._state["fix"]        = True
                self._state["status"]     = "FIX"
                self._state["latitude"]   = lat
                self._state["longitude"]  = lon
                self._state["speed_kmh"]  = speed_kmh
                self._state["course_deg"] = round(course, 2) if course is not None else None
                self._state["last_fix_time"] = now

    def _parse_gsa(self, f: list):
        # $--GSA,...,PDOP,HDOP,VDOP
        if len(f) < 17:
            return
        hdop = self._sf(f[16]) if len(f) > 16 else None
        if hdop is not None:
            with self._lock:
                self._state["hdop"] = round(hdop, 2)

    def _parse_gsv(self, f: list):
        # $--GSV,num_msgs,msg_num,total_sats,...
        if len(f) < 4:
            return
        total = self._si(f[3])
        with self._lock:
            self._state["satellites_in_view"] = total

    # -----------------------------------------------------------------------
    # NMEA dispatcher
    # -----------------------------------------------------------------------
    _PARSERS_SUFFIXES = {
        "GGA": "_parse_gga",
        "RMC": "_parse_rmc",
        "GSA": "_parse_gsa",
        "GSV": "_parse_gsv",
    }

    def _dispatch(self, line: str):
        line = line.strip()
        if not line.startswith("$"):
            return
        if not self._checksum_ok(line):
            return
        body   = line.lstrip("$").split("*")[0]
        fields = body.split(",")
        if not fields:
            return
        sid = fields[0].upper()
        for suffix, method_name in self._PARSERS_SUFFIXES.items():
            if sid.endswith(suffix):
                try:
                    getattr(self, method_name)(fields)
                except Exception:
                    pass
                break

    # -----------------------------------------------------------------------
    # Background reader thread  (mirrors sensors/gnss.py _navcast_thread)
    # -----------------------------------------------------------------------
    def _reader_thread(self):
        while not self._stop.is_set():
            sock = None
            try:
                # If auto-scan is enabled and needed, or rescan was requested
                if (self._auto_scan and self._needs_scan) or self._rescan_requested.is_set():
                    self._rescan_requested.clear()
                    self._log("[GNSS] Auto-scanning network interfaces for NavCast data stream...")
                    pref_h = None if self._host == "auto" else self._host
                    if discover_navcast is not None:
                        found_h, found_p = discover_navcast(
                            preferred_host=pref_h, preferred_port=self._port
                        )
                        if found_h and found_p:
                            self._host = found_h
                            self._port = found_p
                            self._needs_scan = False
                            self._log(f"[GNSS] SUCCESS: Found NavCast stream @ {self._host}:{self._port}")
                        else:
                            self._log(f"[GNSS] Auto-scan found no active NavCast. Trying {self._host}:{self._port}...")
                    else:
                        self._log("[GNSS] Discovery module not available; skipping auto-scan.")

                target_host = "10.208.43.190" if self._host == "auto" else self._host

                self._log(f"[GNSS] Connecting to NavCast @ {target_host}:{self._port} ...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._SOCKET_TIMEOUT)
                sock.connect((target_host, self._port))
                self._host = target_host
                self.connected  = True
                self.last_error = ""
                self._needs_scan = False
                self._log(f"[GNSS] NavCast connected: {self._host}:{self._port}")

                buf = ""
                while not self._stop.is_set():
                    try:
                        chunk = sock.recv(self._RECV_BUF)
                    except socket.timeout:
                        continue
                    if not chunk:
                        self._log("[GNSS] NavCast connection closed by remote.")
                        break
                    buf += chunk.decode("ascii", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        self._dispatch(line)

            except OSError as exc:
                self.connected  = False
                self.last_error = str(exc)
                with self._lock:
                    self._state["data_received"] = False
                    self._state["fix"]           = False
                    self._state["status"]        = "NO_DATA"
                if self._auto_scan:
                    self._needs_scan = True
                    self._log(f"[GNSS] NavCast error ({exc}). Auto-scanning on retry in {self._RECONNECT_DELAY:.0f}s...")
                else:
                    self._log(f"[GNSS] NavCast error ({exc}). Retry in {self._RECONNECT_DELAY:.0f}s...")
            finally:
                self.connected = False
                if sock:
                    try: sock.close()
                    except Exception: pass

            self._stop.wait(self._RECONNECT_DELAY)


# ===========================================================================
# GNSS SIMULATOR (fallback, --sim flag)
# ===========================================================================

class GNSSSimulator:
    """Walks a virtual GPS at configurable speed along a fixed bearing."""

    def __init__(self, start_lat=DEFAULT_LAT, start_lon=DEFAULT_LON,
                 speed_kmh=SIM_SPEED_KMH_DEFAULT, bearing_deg=45.0,
                 update_hz=SIM_UPDATE_HZ):
        self.lat         = start_lat
        self.lon         = start_lon
        self.speed_kmh   = speed_kmh
        self.bearing_deg = bearing_deg
        self.update_hz   = update_hz
        self.has_fix     = True
        self._lock       = threading.Lock()
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True, name="gnss-sim")

    def start(self): self._thread.start()

    def _run(self):
        interval = 1.0 / self.update_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._lock:
                if self.has_fix and self.speed_kmh > 0:
                    d = (self.speed_kmh * 1000.0 / 3600.0) * interval
                    self.lat, self.lon = move_coords(self.lat, self.lon, self.bearing_deg, d)
            s = interval - (time.monotonic() - t0)
            if s > 0: time.sleep(s)

    def get_position(self):
        with self._lock:
            return self.lat, self.lon, self.has_fix

    def get_state(self):
        with self._lock:
            return {
                "data_received": True,
                "fix": self.has_fix,
                "status": "FIX" if self.has_fix else "NO_FIX",
                "latitude": self.lat if self.has_fix else None,
                "longitude": self.lon if self.has_fix else None,
                "speed_kmh": self.speed_kmh if self.has_fix else None,
                "satellites": 12, "satellites_in_view": 14, "hdop": 0.9,
            }

    def set_speed(self, kmh):
        with self._lock: self.speed_kmh = max(0.0, float(kmh))

    def drop_fix(self):
        with self._lock: self.has_fix = False

    def restore_fix(self):
        with self._lock: self.has_fix = True

    def stop(self): self._stop.set()


# ===========================================================================
# SOFTWARE LED STUB (LED display removed per user request)
# ===========================================================================

class SoftwareLEDs:
    """Stub retained for backwards compatibility."""
    def __init__(self, *a, **k): pass
    def update(self, *a, **k): pass
    def notify_litter_capture(self): pass
    def notify_litter_snap(self): pass
    def shutdown(self): pass


# ===========================================================================
# LOCAL SAVE  (no cloud upload, same format as litter_engine.py)
# ===========================================================================

_save_queue: queue.Queue = queue.Queue()
_save_thread_ref: list = []


def _save_worker():
    while True:
        item = _save_queue.get()
        if item is None:
            _save_queue.task_done()
            break
        img_path, frame, json_path, sidecar = item
        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                with open(img_path, "wb") as f:
                    f.write(buf.tobytes())
            else:
                print(f"[SAVE] ERROR: cv2.imencode failed for {os.path.basename(img_path)}")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)
            print(f"[SAVE] {os.path.basename(img_path)}")
        except Exception as exc:
            print(f"[SAVE] ERROR: {exc}")
        finally:
            _save_queue.task_done()


def _ensure_save_thread():
    if not _save_thread_ref or not _save_thread_ref[0].is_alive():
        t = threading.Thread(target=_save_worker, daemon=True, name="litter-save")
        t.start()
        if _save_thread_ref:
            _save_thread_ref[0] = t
        else:
            _save_thread_ref.append(t)


def save_litter_result(output_dir, annotated_frame, detections, trigger_n, event_n,
                       lat, lon, has_fix, op_mode, infer_ms, gnss_state):
    """Save JPEG + JSON sidecar with banner overlay matching production litter_engine.py."""
    os.makedirs(output_dir, exist_ok=True)
    captured_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clean_ts    = captured_at[:19].replace(":", "-").replace("T", "_")
    img_path    = os.path.join(output_dir, f"litter_T{trigger_n:04d}_E{event_n:04d}_{clean_ts}.jpg")
    json_path   = os.path.splitext(img_path)[0] + ".json"

    h, w = annotated_frame.shape[:2]
    strip_h = 70
    banner  = np.zeros((strip_h, w, 3), dtype=np.uint8)
    cv2.line(banner, (0, 0), (w, 0), (50, 50, 50), 1)

    ts_display = captured_at[:19].replace("T", " ")
    cv2.putText(banner, f"RTC: {ts_display}",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    if has_fix:
        gps_text = f"GNSS: {float(lat):.8f}, {float(lon):.8f}"
        gps_col  = (0, 255, 0)
    elif op_mode == "TIME_FALLBACK":
        gps_text = "GNSS: NO FIX (CLOCK TRIGGER)"
        gps_col  = (0, 165, 255)
    else:
        gps_text = "GNSS: NO FIX"
        gps_col  = (0, 0, 255)
    cv2.putText(banner, gps_text,
                (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gps_col, 2, cv2.LINE_AA)

    final_frame = np.vstack([annotated_frame, banner])

    sidecar = {
        "event_number":    event_n,
        "trigger_number":  trigger_n,
        "image_file":      os.path.basename(img_path),
        "device_id":       "TEST_LAPTOP",
        "captured_at":     captured_at,
        "trigger_mode":    op_mode,
        "inference_ms":    infer_ms,
        "gnss": {
            "fix":                has_fix,
            "latitude":           float(lat) if lat is not None else None,
            "longitude":          float(lon) if lon is not None else None,
            "speed_kmh":          gnss_state.get("speed_kmh"),
            "satellites":         gnss_state.get("satellites", 0),
            "satellites_in_view": gnss_state.get("satellites_in_view", 0),
            "hdop":               gnss_state.get("hdop"),
        },
        "detections":             detections,
        "detection_type":         "LITTER",
        "detection_confidence":   round(
            max((d["conf"] for d in detections), default=0.0), 4),
    }

    _ensure_save_thread()
    _save_queue.put((img_path, final_frame, json_path, sidecar))
    return img_path


# ===========================================================================
# STANDALONE LITTER ENGINE
# ===========================================================================

class StandaloneLitterEngine:
    """
    Trigger logic + YOLO inference + local save.
    Mirrors LitterEngine from litter_engine.py with no external imports.
    """

    def __init__(self, model, output_dir,
                 distance_interval_m=LITTER_DISTANCE_INTERVAL_M,
                 gnss_lost_timeout_sec=LITTER_GNSS_LOST_TIMEOUT_SEC,
                 time_fallback_sec=LITTER_TIME_FALLBACK_SEC,
                 conf=LITTER_CONF_THRESHOLD, log_fn=None, leds=None):
        self._model               = model
        self._output_dir          = output_dir
        self.distance_interval_m  = distance_interval_m
        self.gnss_lost_timeout_sec = gnss_lost_timeout_sec
        self.time_fallback_sec    = time_fallback_sec
        self.conf                 = conf
        self._log                 = log_fn or print
        self._leds: Optional[SoftwareLEDs] = leds

        self._op_mode             = "GNSS_DISTANCE"
        self._last_trigger_lat    = None
        self._last_trigger_lon    = None
        self._gnss_loss_start     = None
        self._last_fallback_time  = time.monotonic()
        self._total_distance_m    = 0.0
        self._distance_since_trig = 0.0
        self._trigger_count       = 0
        self._event_count         = 0
        self._gnss_initialized    = False
        self._last_infer_ms       = 0
        self._last_detections     = []
        self._last_annotated_frame = None
        self._last_gnss_state     = {}

        self._infer_queue = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._infer_thread = threading.Thread(
            target=self._inference_worker, daemon=True, name="yolo-infer")
        self._infer_thread.start()

        self.on_trigger   = None
        self.on_detection = None

    def update_gnss(self, lat, lon, has_hardware_fix=True):
        now = time.monotonic()
        coord_valid = (lat is not None and lon is not None
                       and not (abs(lat) < 0.001 and abs(lon) < 0.001))
        has_fix = coord_valid and has_hardware_fix

        if has_fix:
            if not self._gnss_initialized:
                self._last_trigger_lat = lat
                self._last_trigger_lon = lon
                self._op_mode          = "GNSS_DISTANCE"
                self._gnss_initialized = True
                self._gnss_loss_start  = None
                self._log(f"[ENGINE] GNSS anchored @ ({lat:.6f}, {lon:.6f}) -- "
                          f"trigger every {self.distance_interval_m:.1f} m")
            elif self._op_mode == "TIME_FALLBACK":
                self._last_trigger_lat = lat
                self._last_trigger_lon = lon
                self._op_mode          = "GNSS_DISTANCE"
                self._gnss_loss_start  = None
                self._log("[ENGINE] GNSS REGAINED -- re-anchoring, resuming distance mode")
            elif self._gnss_loss_start is not None:
                elapsed = now - self._gnss_loss_start
                self._gnss_loss_start = None
                self._log(f"[ENGINE] GNSS recovered within grace period ({elapsed:.1f}s)")
        else:
            if self._op_mode == "GNSS_DISTANCE":
                if self._gnss_loss_start is None:
                    self._gnss_loss_start = now
                    self._log(f"[ENGINE] GNSS fix LOST -- grace {self.gnss_lost_timeout_sec:.0f}s "
                              f"before clock fallback...")
                elif (now - self._gnss_loss_start) >= self.gnss_lost_timeout_sec:
                    _FIRST = 5.0
                    self._op_mode = "TIME_FALLBACK"
                    self._last_fallback_time = now - (self.time_fallback_sec - _FIRST)
                    self._gnss_loss_start = None
                    self._log(f"[ENGINE] GNSS lost >{self.gnss_lost_timeout_sec:.0f}s -- "
                              f"clock fallback every {self.time_fallback_sec:.0f}s "
                              f"(first in ~{_FIRST:.0f}s)")

        trigger = False
        if self._op_mode == "GNSS_DISTANCE" and has_fix and self._last_trigger_lat is not None:
            dist = haversine_m(self._last_trigger_lat, self._last_trigger_lon, lat, lon)
            self._distance_since_trig = dist
            if dist >= self.distance_interval_m:
                self._total_distance_m   += dist
                self._last_trigger_lat    = lat
                self._last_trigger_lon    = lon
                self._distance_since_trig = 0.0
                trigger = True
                self._log(f"[ENGINE] DISTANCE TRIGGER #{self._trigger_count + 1} -- "
                          f"{dist:.2f} m >= {self.distance_interval_m:.1f} m")
        elif self._op_mode == "TIME_FALLBACK":
            elapsed = now - self._last_fallback_time
            if elapsed >= self.time_fallback_sec:
                self._last_fallback_time = now
                trigger = True
                self._log(f"[ENGINE] TIME FALLBACK TRIGGER #{self._trigger_count + 1}")

        return trigger

    def try_trigger(self, frame, lat, lon, gnss_state=None):
        if self._model is None:
            self._log("[ENGINE] No model -- skipping inference")
            return
        self._last_gnss_state = gnss_state or {}
        job = {"frame": frame.copy(), "lat": lat, "lon": lon,
               "trigger_n": self._trigger_count + 1,
               "op_mode": self._op_mode,
               "gnss_state": gnss_state or {}}
        try:
            self._infer_queue.put_nowait(job)
            self._trigger_count += 1
            if self.on_trigger:
                self.on_trigger(self._trigger_count)
            if self._leds:
                self._leds.notify_litter_capture()
        except queue.Full:
            self._log("[ENGINE] Trigger DROPPED -- inference still busy")

    def _inference_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self._infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run_inference(job)
            except Exception as exc:
                self._log(f"[ENGINE] Inference error: {exc}")
            finally:
                self._infer_queue.task_done()

    def _run_inference(self, job):
        frame       = job["frame"]
        trigger_n   = job["trigger_n"]
        lat         = job["lat"]
        lon         = job["lon"]
        op_mode     = job["op_mode"]
        gnss_state  = job["gnss_state"]
        coord_valid = (lat is not None and lon is not None
                       and not (abs(lat) < 0.001 and abs(lon) < 0.001))

        t0 = time.monotonic()
        try:
            results = self._model.track(frame, persist=True, conf=self.conf, verbose=False)
        except Exception as exc:
            self._log(f"[YOLO] Model error: {exc}")
            return
        infer_ms = int((time.monotonic() - t0) * 1000)
        self._last_infer_ms = infer_ms

        result = results[0]
        canvas = frame.copy()
        detections = []
        has_litter = False

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_v   = float(box.conf[0])
                cls_id   = int(box.cls[0])
                label    = f"litter #{track_id}  {conf_v:.2f}"
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 50), 3)
                cv2.putText(canvas, label, (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 50), 2, cv2.LINE_AA)
                detections.append({"track_id": track_id, "cls_id": cls_id,
                                   "conf": round(conf_v, 4), "bbox": [x1, y1, x2, y2]})
                has_litter = True

        self._last_detections = detections
        self._last_annotated_frame = canvas

        if has_litter:
            self._event_count += len(detections)
            self._log(f"[YOLO] Trigger #{trigger_n}: {len(detections)} litter ({infer_ms} ms)")
            if self._leds:
                self._leds.notify_litter_snap()
            save_litter_result(
                output_dir=self._output_dir,
                annotated_frame=canvas,
                detections=detections,
                trigger_n=trigger_n, event_n=self._event_count,
                lat=lat, lon=lon, has_fix=coord_valid,
                op_mode=op_mode, infer_ms=infer_ms,
                gnss_state=gnss_state,
            )
        else:
            self._log(f"[YOLO] Trigger #{trigger_n}: clean -- 0 litter ({infer_ms} ms)")

        if self.on_detection:
            self.on_detection(canvas, detections, infer_ms)

    @property
    def op_mode(self): return self._op_mode
    @property
    def total_distance_m(self): return self._total_distance_m
    @property
    def distance_since_trigger(self): return self._distance_since_trig
    @property
    def trigger_count(self): return self._trigger_count
    @property
    def event_count(self): return self._event_count
    @property
    def last_infer_ms(self): return self._last_infer_ms

    @property
    def gnss_grace_remaining(self):
        if self._gnss_loss_start is None:
            return None
        return max(0.0, self.gnss_lost_timeout_sec - (time.monotonic() - self._gnss_loss_start))

    def shutdown(self): self._stop_event.set()


# ===========================================================================
# WEBCAM
# ===========================================================================

class CameraCapture:
    def __init__(self, index=0):
        self._cap = None; self._frame = None
        self._lock = threading.Lock(); self._stop = threading.Event()
        self._index = index; self._ok = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="cam")

    def start(self):
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(self._index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap = cap; self._ok = True; self._thread.start(); return True
        cap.release(); return False

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock: self._frame = frame
            else: time.sleep(0.05)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_ok(self): return self._ok

    def stop(self):
        self._stop.set()
        if self._cap: self._cap.release()


def make_synthetic_frame(width=640, height=480):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / height
        frame[y, :] = [int(20 + 40 * t), int(30 + 60 * t), int(60 + 80 * t)]
    rng = np.random.RandomState(int(time.time()) % 10000)
    for _ in range(rng.randint(0, 4)):
        cx, cy = rng.randint(80, width - 80), rng.randint(80, height - 80)
        w2, h2 = rng.randint(30, 110), rng.randint(20, 70)
        c = tuple(int(x) for x in rng.randint(80, 255, 3).tolist())
        cv2.rectangle(frame, (cx-w2//2, cy-h2//2), (cx+w2//2, cy+h2//2), c, -1)
    cv2.putText(frame, f"SYNTHETIC  {time.strftime('%H:%M:%S')}",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
    cv2.putText(frame, "No camera -- generated test frame",
                (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1, cv2.LINE_AA)
    return frame


# ===========================================================================
# COLOUR PALETTE
# ===========================================================================
P = {
    "bg":      "#0d1117", "panel":  "#161b22", "border": "#30363d",
    "accent":  "#58a6ff", "green":  "#3fb950", "orange": "#f0883e",
    "red":     "#f85149", "purple": "#bc8cff", "text":   "#e6edf3",
    "dim":     "#8b949e", "yellow": "#e3b341",
}


# ===========================================================================
# GUI
# ===========================================================================

class YOLOTestGUI:

    def __init__(self, root, engine, gnss_source, camera,
                 weights_label, distance_interval_m, output_dir, use_sim, leds=None):
        self._root        = root
        self._engine      = engine
        self._gnss        = gnss_source   # NavCastGNSSReader or GNSSSimulator
        self._camera      = camera
        self._running     = True
        self._dist_m      = distance_interval_m
        self._out_dir     = output_dir
        self._use_sim     = use_sim
        self._last_frame  = None
        self._frame_lock  = threading.Lock()
        self._log_queue   = queue.Queue()

        engine.on_trigger   = self._cb_trigger
        engine.on_detection = self._cb_detection

        root.title("YOLO Litter Engine Test  --  SWSTP Edge")
        root.configure(bg=P["bg"])
        root.minsize(1180, 800)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui(weights_label)
        self._root.after(GUI_REFRESH_MS, self._tick)

    # -----------------------------------------------------------------------
    # UI BUILD
    # -----------------------------------------------------------------------
    def _build_ui(self, weights_label):
        try:
            FM = tkfont.Font(family="Consolas", size=10)
            FS = tkfont.Font(family="Segoe UI", size=9)
            FB = tkfont.Font(family="Segoe UI", size=11)
            FT = tkfont.Font(family="Segoe UI", size=10, weight="bold")
            FH = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        except Exception:
            FM = tkfont.Font(family="Courier New", size=10)
            FS = tkfont.Font(size=9); FB = tkfont.Font(size=11)
            FT = tkfont.Font(size=10, weight="bold"); FH = tkfont.Font(size=13, weight="bold")

        tb = tk.Frame(self._root, bg=P["panel"], pady=8)
        tb.pack(fill="x")
        tk.Label(tb, text="  YOLO Litter Engine Test  --  SWSTP Edge",
                 bg=P["panel"], fg=P["accent"], font=FH).pack(side="left", padx=12)
        src_txt = "SIM" if self._use_sim else "NavCast GNSS"
        tk.Label(tb, text=f"GNSS: {src_txt} | Model: {weights_label}",
                 bg=P["panel"], fg=P["dim"], font=FS).pack(side="right", padx=12)

        content = tk.Frame(self._root, bg=P["bg"])
        content.pack(fill="both", expand=True, padx=6, pady=4)

        left = tk.Frame(content, bg=P["bg"], width=370)
        left.pack(side="left", fill="y", padx=(0, 4))
        left.pack_propagate(False)

        right = tk.Frame(content, bg=P["bg"])
        right.pack(side="left", fill="both", expand=True)

        self._build_gnss_panel(left, FB, FS, FT)
        self._build_status(left, FB, FS, FT)
        self._build_controls(left, FT, FB, FS)
        self._build_camera(right, FT)
        self._build_log(right, FM, FT)

    def _box(self, parent, title, font):
        wrap = tk.Frame(parent, bg=P["border"], padx=1, pady=1)
        wrap.pack(fill="x", padx=4, pady=4)
        hdr = tk.Frame(wrap, bg=P["panel"])
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, bg=P["panel"], fg=P["accent"],
                 font=font, pady=5, padx=10).pack(side="left")
        body = tk.Frame(wrap, bg=P["panel"], padx=12, pady=8)
        body.pack(fill="x")
        return body

    # ── GNSS Source Panel ──────────────────────────────────────────────────
    def _build_gnss_panel(self, parent, FB, FS, FT):
        p = self._box(parent, "GNSS SOURCE (NAVCAST)", FT)

        row = tk.Frame(p, bg=P["panel"])
        row.pack(fill="x", pady=(0, 4))
        src = "SIMULATION (--sim)" if self._use_sim else "NavCast TCP (auto-scan)"
        tk.Label(row, text="Source:", bg=P["panel"], fg=P["dim"], font=FS).pack(side="left")
        self._lbl_gnss_src = tk.Label(
            row, text=src,
            bg="#1a3a2a" if not self._use_sim else "#2a1a3a",
            fg=P["green"] if not self._use_sim else P["purple"],
            font=FT, padx=8, pady=2)
        self._lbl_gnss_src.pack(side="left", padx=6)

        stats = tk.Frame(p, bg=P["panel"])
        stats.pack(fill="x", pady=(2, 0))

        def row(label, init, fg=P["text"]):
            r = tk.Frame(stats, bg=P["panel"])
            r.pack(fill="x", pady=1)
            tk.Label(r, text=label, bg=P["panel"], fg=P["dim"],
                     font=FS, width=16, anchor="w").pack(side="left")
            lbl = tk.Label(r, text=init, bg=P["panel"], fg=fg, font=FB, anchor="w")
            lbl.pack(side="left")
            return lbl

        self._L_endpoint = row("Target:",       "Auto-scanning..." if not self._use_sim else "N/A", P["accent"])
        self._L_conn     = row("Connection:",   "Connecting..." if not self._use_sim else "N/A", P["orange"])
        self._L_gnss_st  = row("GNSS status:",  "--", P["dim"])
        self._L_sats     = row("Satellites:",   "--", P["text"])
        self._L_hdop     = row("HDOP:",         "--", P["text"])
        self._L_gnss_sp  = row("Speed (GNSS):", "--", P["text"])

    # ── Engine Status ─────────────────────────────────────────────────────────
    def _build_status(self, parent, FB, FS, FT):
        p = self._box(parent, "ENGINE STATUS", FT)

        row = tk.Frame(p, bg=P["panel"])
        row.pack(fill="x", pady=(0, 6))
        tk.Label(row, text="Mode:", bg=P["panel"], fg=P["dim"], font=FS).pack(side="left")
        self._lbl_mode = tk.Label(row, text="GNSS_DISTANCE",
                                  bg="#1a3a2a", fg=P["green"], font=FT, padx=8, pady=2)
        self._lbl_mode.pack(side="left", padx=6)

        tk.Label(p, text="Distance to next trigger:",
                 bg=P["panel"], fg=P["dim"], font=FS).pack(anchor="w", pady=(4, 0))
        bw = tk.Frame(p, bg=P["border"], height=24, padx=1, pady=1)
        bw.pack(fill="x", pady=(2, 4))
        bw.pack_propagate(False)
        self._bar_bg   = tk.Frame(bw, bg=P["bg"])
        self._bar_bg.place(relwidth=1.0, relheight=1.0)
        self._bar_fill = tk.Frame(self._bar_bg, bg=P["accent"])
        self._bar_fill.place(x=0, y=0, relheight=1.0, width=0)
        self._lbl_bar  = tk.Label(bw, text="0.00 / 5.00 m  (0%)", bg=P["bg"], fg=P["text"], font=FS)
        self._lbl_bar.place(relx=0.5, rely=0.5, anchor="center")

        stats = tk.Frame(p, bg=P["panel"])
        stats.pack(fill="x", pady=(6, 0))

        def stat(label, init, fg=P["text"]):
            r = tk.Frame(stats, bg=P["panel"])
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, bg=P["panel"], fg=P["dim"],
                     font=FS, width=22, anchor="w").pack(side="left")
            lbl = tk.Label(r, text=init, bg=P["panel"], fg=fg, font=FB, anchor="w")
            lbl.pack(side="left")
            return lbl

        self._L_total  = stat("Total distance:",   "0.00 m",  P["accent"])
        self._L_trig   = stat("Triggers fired:",   "0",       P["purple"])
        self._L_det    = stat("Litter detected:",  "0",       P["orange"])
        self._L_ms     = stat("Last infer time:",  "--",      P["dim"])
        self._L_grace  = stat("Grace / fallback:", "--",      P["orange"])
        self._L_pos    = stat("Position:",         "--",      P["dim"])
        self._L_saves  = stat("Saves (local):",    "0",       P["green"])

    # ── Controls ─────────────────────────────────────────────────────────────
    def _build_controls(self, parent, FT, FB, FS):
        p = self._box(parent, "CONTROLS", FT)

        # Speed slider — only relevant in sim mode
        if self._use_sim:
            tk.Label(p, text="Simulated speed (km/h):",
                     bg=P["panel"], fg=P["dim"], font=FS).pack(anchor="w")
            self._spd_var = tk.DoubleVar(value=SIM_SPEED_KMH_DEFAULT)
            sf = tk.Frame(p, bg=P["panel"])
            sf.pack(fill="x", pady=(2, 8))
            tk.Scale(sf, from_=0, to=80, orient="horizontal",
                     variable=self._spd_var, command=self._on_speed,
                     bg=P["panel"], fg=P["text"], activebackground=P["accent"],
                     troughcolor=P["border"], highlightthickness=0, bd=0,
                     showvalue=False).pack(side="left", fill="x", expand=True)
            self._lbl_spd = tk.Label(sf, text="20", bg=P["panel"],
                                     fg=P["accent"], font=FB, width=4)
            self._lbl_spd.pack(side="left")

            self._fix_var = tk.BooleanVar(value=True)
            tk.Checkbutton(p, text="GNSS Fix Active (sim)", variable=self._fix_var,
                           command=self._on_fix, bg=P["panel"], fg=P["text"],
                           selectcolor=P["bg"], activebackground=P["panel"],
                           activeforeground=P["text"], font=FB).pack(anchor="w", pady=4)
        else:
            # Live mode: show NavCast info
            tk.Label(p, text="NavCast live stream controls:",
                     bg=P["panel"], fg=P["dim"], font=FS).pack(anchor="w", pady=(0, 6))

        def btn(txt, cmd, bg):
            tk.Button(p, text=txt, command=cmd, bg=bg, fg=P["text"],
                      font=FB, relief="flat", padx=10, pady=5, cursor="hand2",
                      activebackground=bg, activeforeground=P["text"]).pack(fill="x", pady=3)

        btn("Force Trigger NOW",           self._force_trigger,  "#1f4e79")
        if not self._use_sim:
            btn("Auto-Scan NavCast Stream", self._rescan_navcast, "#0e3a40")
        if self._use_sim:
            btn("Force GNSS Loss / Fallback",  self._force_fallback, "#4a2c0a")
            btn("Restore GNSS + Re-anchor",    self._restore_gnss,   "#0a3321")
        btn("Reset All Stats",             self._reset_stats,    "#2a1a3a")

        iv = tk.Frame(p, bg=P["panel"])
        iv.pack(fill="x", pady=(8, 0))
        tk.Label(iv, text="Trigger interval:",
                 bg=P["panel"], fg=P["dim"], font=FS).pack(side="left")
        tk.Label(iv, text=f"{self._dist_m:.1f} m",
                 bg=P["panel"], fg=P["accent"], font=FB).pack(side="left", padx=6)

    def _build_camera(self, parent, FT):
        wrap = tk.Frame(parent, bg=P["border"], padx=1, pady=1)
        wrap.pack(fill="both", expand=True, padx=4, pady=(4, 2))
        hdr = tk.Frame(wrap, bg=P["panel"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="INFERENCE VIEW",
                 bg=P["panel"], fg=P["accent"], font=FT, pady=5, padx=10).pack(side="left")
        self._lbl_cam_status = tk.Label(hdr, text="Waiting for first trigger...",
                                        bg=P["panel"], fg=P["dim"], font=FT, padx=10)
        self._lbl_cam_status.pack(side="right")
        self._cam_lbl = tk.Label(wrap, bg="#0a0a0a")
        self._cam_lbl.pack(fill="both", expand=True)
        ph = np.zeros((300, 640, 3), dtype=np.uint8)
        cv2.putText(ph, "Waiting for first inference trigger...",
                    (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 80, 80), 2)
        self._show_frame(ph)

    def _build_log(self, parent, FM, FT):
        wrap = tk.Frame(parent, bg=P["border"], padx=1, pady=1)
        wrap.pack(fill="x", padx=4, pady=(2, 4))
        hdr = tk.Frame(wrap, bg=P["panel"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="LIVE LOG",
                 bg=P["panel"], fg=P["accent"], font=FT, pady=4, padx=10).pack(side="left")
        tk.Button(hdr, text="Clear", command=lambda: self._log_txt.delete("1.0", "end"),
                  bg=P["border"], fg=P["dim"], relief="flat",
                  padx=6, cursor="hand2").pack(side="right", padx=8, pady=2)
        self._log_txt = scrolledtext.ScrolledText(
            wrap, height=9, bg="#0d1117", fg=P["text"],
            font=FM, wrap="word", relief="flat", insertbackground=P["text"])
        self._log_txt.pack(fill="x")
        self._log_txt.tag_config("trig",  foreground=P["orange"])
        self._log_txt.tag_config("det",   foreground=P["green"])
        self._log_txt.tag_config("fall",  foreground=P["purple"])
        self._log_txt.tag_config("warn",  foreground=P["orange"])
        self._log_txt.tag_config("err",   foreground=P["red"])
        self._log_txt.tag_config("ok",    foreground=P["accent"])
        self._log_txt.tag_config("led",   foreground=P["yellow"])
        self._log_txt.tag_config("save",  foreground=P["green"])
        self._log_txt.tag_config("dim",   foreground=P["dim"])

    # -----------------------------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------------------------
    def _log(self, msg):
        self._log_queue.put(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _flush_log(self):
        cnt = 0
        while not self._log_queue.empty() and cnt < 60:
            try: msg = self._log_queue.get_nowait()
            except queue.Empty: break
            cnt += 1
            mu = msg.upper()
            tag = "dim"
            if "[LED]" in msg: tag = "led"
            elif "[SAVE]" in msg: tag = "save"
            elif "TRIGGER" in mu or "DISTANCE" in mu: tag = "trig"
            elif "LITTER DETECTED" in mu or "LITTER DETECT" in mu: tag = "det"
            elif "FALLBACK" in mu or "CLOCK" in mu: tag = "fall"
            elif "GNSS LOST" in mu or "WARN" in mu: tag = "warn"
            elif "ERROR" in mu or "FATAL" in mu: tag = "err"
            elif "REGAINED" in mu or "ANCHOR" in mu or "READY" in mu or "RESTOR" in mu or "CONNECT" in mu: tag = "ok"
            self._log_txt.insert("end", msg + "\n", tag)
        if cnt: self._log_txt.see("end")

    def _cb_trigger(self, n):
        self._log(f"[ENGINE] Trigger #{n} queued for inference")

    def _cb_detection(self, annotated, dets, ms):
        with self._frame_lock:
            self._last_frame = annotated
        n = len(dets)
        if n:
            self._log(f"[YOLO] {n} litter detected -- {ms} ms  --> saved locally")
        else:
            self._log(f"[YOLO] Clean frame -- 0 litter ({ms} ms) [not saved]")

    # -----------------------------------------------------------------------
    # CONTROLS
    # -----------------------------------------------------------------------
    def _rescan_navcast(self):
        self._log("[GNSS] Manual auto-scan requested -- scanning network interfaces...")
        if hasattr(self._gnss, "request_rescan"):
            self._gnss.request_rescan()

    def _on_speed(self, val=None):
        kmh = self._spd_var.get()
        if hasattr(self._gnss, "set_speed"):
            self._gnss.set_speed(kmh)
        self._lbl_spd.config(text=f"{kmh:.0f}")

    def _on_fix(self):
        if not hasattr(self._gnss, "drop_fix"): return
        if self._fix_var.get():
            self._gnss.restore_fix()
            self._log("[SIM] GNSS fix RESTORED")
        else:
            self._gnss.drop_fix()
            self._log("[SIM] GNSS fix DROPPED -- grace period starting")

    def _force_trigger(self):
        lat, lon, _ = self._gnss.get_position()
        gnss_state  = self._gnss.get_state()
        self._engine.try_trigger(self._get_frame(), lat, lon, gnss_state)
        self._log("[SIM] Manual trigger forced")

    def _force_fallback(self):
        if hasattr(self._gnss, "drop_fix"):
            self._gnss.drop_fix()
            self._fix_var.set(False)
            self._log("[SIM] GNSS dropped -- TIME_FALLBACK will engage after grace period")

    def _restore_gnss(self):
        if hasattr(self._gnss, "restore_fix"):
            self._gnss.restore_fix()
            self._fix_var.set(True)
        self._engine._gnss_initialized    = False
        self._engine._distance_since_trig = 0.0
        self._log("[SIM] GNSS restored -- engine will re-anchor on next update")

    def _reset_stats(self):
        e = self._engine
        e._total_distance_m = 0.0; e._trigger_count = 0
        e._event_count = 0; e._distance_since_trig = 0.0; e._last_infer_ms = 0
        self._log("[SIM] All stats reset")

    # -----------------------------------------------------------------------
    # FRAME HELPERS
    # -----------------------------------------------------------------------
    def _get_frame(self):
        if self._camera and self._camera.is_ok():
            f = self._camera.read()
            if f is not None: return f
        return make_synthetic_frame()

    def _show_frame(self, frame):
        try:
            from PIL import Image, ImageTk
            w = max(10, self._cam_lbl.winfo_width())
            h = max(10, self._cam_lbl.winfo_height())
            fh, fw = frame.shape[:2]
            scale = min(w / fw, h / fh, 1.0)
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            rgb = cv2.cvtColor(
                cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._cam_lbl.config(image=img)
            self._cam_lbl.image = img
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # MAIN TICK
    # -----------------------------------------------------------------------
    def _tick(self):
        if not self._running: return

        # Pull position from either NavCast or sim
        lat, lon, has_fix = self._gnss.get_position()
        gnss_state        = self._gnss.get_state()

        # Update NavCast GNSS panel
        if not self._use_sim:
            reader: NavCastGNSSReader = self._gnss
            connected = reader.connected
            status = gnss_state.get("status", "NO_DATA")
            if hasattr(reader, "endpoint"):
                self._L_endpoint.config(text=reader.endpoint)
            self._L_conn.config(
                text="Connected" if connected else f"Reconnecting... ({reader.last_error[:30]})",
                fg=P["green"] if connected else P["orange"])
            self._L_gnss_st.config(
                text=status,
                fg=P["green"] if status == "FIX" else (P["orange"] if status == "NO_FIX" else P["red"]))
        else:
            status = "FIX" if has_fix else "NO_FIX"
            self._L_endpoint.config(text="Simulation")
            self._L_conn.config(text="Simulation", fg=P["purple"])
            self._L_gnss_st.config(
                text=status,
                fg=P["green"] if has_fix else P["orange"])

        sats = gnss_state.get("satellites", 0) or 0
        sats_v = gnss_state.get("satellites_in_view", 0) or 0
        hdop = gnss_state.get("hdop")
        spd  = gnss_state.get("speed_kmh")
        self._L_sats.config(text=f"{sats} used / {sats_v} in view")
        self._L_hdop.config(
            text=f"{hdop:.2f}" if hdop is not None else "--",
            fg=P["green"] if (hdop and hdop <= 2.0) else (P["orange"] if (hdop and hdop <= 5.0) else P["red"] if hdop else P["dim"]))
        self._L_gnss_sp.config(text=f"{spd:.1f} km/h" if spd is not None else "--")

        # Run engine trigger logic
        triggered = self._engine.update_gnss(lat, lon, has_hardware_fix=has_fix)
        if triggered:
            self._engine.try_trigger(self._get_frame(), lat, lon, gnss_state)

        # Inference view
        with self._frame_lock:
            infer_frame = self._last_frame
            self._last_frame = None

        if infer_frame is not None:
            self._show_frame(infer_frame)
            n = len(self._engine._last_detections)
            ms = self._engine.last_infer_ms
            status_txt = f"Clean ({ms} ms)" if n == 0 else f"{n} litter detected ({ms} ms)"
            self._lbl_cam_status.config(
                text=f"Last inference: {status_txt}",
                fg=P["green"] if n == 0 else P["orange"])
        else:
            live = self._get_frame()
            cv2.putText(live, "LIVE -- waiting for trigger",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,120), 1)
            self._show_frame(live)

        # Distance bar
        e    = self._engine
        dist = e.distance_since_trigger
        pct  = min(dist / self._dist_m, 1.0) if self._dist_m > 0 else 0.0
        bw   = max(1, self._bar_bg.winfo_width())
        self._bar_fill.place(width=int(bw * pct))
        self._bar_fill.config(bg=P["accent"] if pct < 0.85 else P["orange"])
        self._lbl_bar.config(text=f"{dist:.2f} / {self._dist_m:.1f} m  ({pct*100:.0f}%)")

        mode = e.op_mode
        if mode == "GNSS_DISTANCE":
            self._lbl_mode.config(text="GNSS_DISTANCE", bg="#1a3a2a", fg=P["green"])
        else:
            self._lbl_mode.config(text="TIME_FALLBACK", bg="#3a2a10", fg=P["orange"])

        self._L_total.config(text=f"{e.total_distance_m:.2f} m")
        self._L_trig.config(text=str(e.trigger_count))
        det_n = e.event_count
        self._L_det.config(text=str(det_n), fg=P["orange"] if det_n > 0 else P["dim"])
        self._L_ms.config(text=f"{e.last_infer_ms} ms" if e.last_infer_ms else "--")

        grace = e.gnss_grace_remaining
        if grace is not None:
            self._L_grace.config(text=f"{grace:.1f}s until fallback", fg=P["orange"])
        elif mode == "TIME_FALLBACK":
            elapsed = time.monotonic() - e._last_fallback_time
            rem = max(0.0, e.time_fallback_sec - elapsed)
            self._L_grace.config(text=f"next clock trigger in {rem:.0f}s", fg=P["purple"])
        else:
            self._L_grace.config(text="--", fg=P["dim"])

        pos_txt = f"{lat:.6f}, {lon:.6f}" if (lat is not None and lon is not None) else "No fix"
        self._L_pos.config(text=pos_txt)

        try:
            saves = len([f for f in os.listdir(self._out_dir) if f.endswith(".jpg")])
            self._L_saves.config(text=str(saves))
        except Exception:
            pass

        self._flush_log()
        self._root.after(GUI_REFRESH_MS, self._tick)

    def _on_close(self):
        self._running = False
        self._engine.shutdown()
        if hasattr(self._gnss, "stop"): self._gnss.stop()
        if self._camera: self._camera.stop()
        self._root.destroy()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="GUI test for YOLO litter detection engine (SWSTP Edge). "
                    "NavCast TCP with auto-scan is the default GNSS source."
    )
    ap.add_argument("--host",      default=NAVCAST_HOST_DEFAULT,
                    help=f"NavCast TCP host (default: '{NAVCAST_HOST_DEFAULT}' to auto-scan phone IP across tether/hotspot interfaces)")
    ap.add_argument("--port",      type=int, default=NAVCAST_PORT_DEFAULT,
                    help=f"NavCast TCP port (default: {NAVCAST_PORT_DEFAULT})")
    ap.add_argument("--no-scan",   action="store_true",
                    help="Disable automatic network scan; connect only to specified --host")
    ap.add_argument("--sim",       action="store_true",
                    help="Use simulated GNSS instead of NavCast (offline testing)")
    ap.add_argument("--sim-speed", type=float, default=SIM_SPEED_KMH_DEFAULT, dest="sim_speed",
                    help=f"Sim speed km/h when --sim is used (default {SIM_SPEED_KMH_DEFAULT})")
    ap.add_argument("--weights",   default=None,
                    help="Path to ONNX weights file.")
    ap.add_argument("--camera",    type=int, default=0, metavar="INDEX",
                    help="Webcam index (default: 0). Set to -1 or use --no-cam for synthetic frames.")
    ap.add_argument("--no-cam",    action="store_true",
                    help="Disable webcam and use synthetic test frames.")
    ap.add_argument("--interval",  type=float, default=LITTER_DISTANCE_INTERVAL_M,
                    help=f"Distance trigger metres (default {LITTER_DISTANCE_INTERVAL_M})")
    ap.add_argument("--mode",      choices=["gnss", "fallback"], default="gnss",
                    help="Start mode: gnss or fallback (only relevant with --sim)")
    ap.add_argument("--conf",      type=float, default=LITTER_CONF_THRESHOLD)
    ap.add_argument("--gnss-timeout", type=float, default=LITTER_GNSS_LOST_TIMEOUT_SEC,
                    dest="gnss_timeout")
    ap.add_argument("--fallback-interval", type=float, default=LITTER_TIME_FALLBACK_SEC,
                    dest="fallback_interval")
    ap.add_argument("--save-dir",  default=_DEFAULT_SAVE_DIR, dest="save_dir")
    return ap.parse_args()


def resolve_weights(user_path):
    if user_path and os.path.isfile(user_path):
        return user_path, os.path.basename(user_path)
    for c in [os.path.join(_HERE, "weights", "best_int8.onnx"),
              os.path.join(_HERE, "weights", "best.onnx")]:
        if os.path.isfile(c): return c, os.path.basename(c)
    return None, "NOT FOUND"


def main():
    args = parse_args()
    print("=" * 70)
    print("  SWSTP Edge -- YOLO Litter Detection Test (no motion, no cloud)")
    if args.sim:
        print("  GNSS: SIMULATION")
    else:
        scan_desc = "auto-scan enabled" if not args.no_scan else "auto-scan disabled"
        print(f"  GNSS: NavCast TCP ({args.host}:{args.port}, {scan_desc})")
    print("=" * 70)

    if not _DETECTOR_AVAILABLE:
        print(f"[FATAL] Cannot import detector.py: {_DETECTOR_IMPORT_ERROR}")
        sys.exit(1)

    try:
        from PIL import Image, ImageTk  # noqa
    except ImportError:
        print("[WARN] Pillow not installed -- camera view blank. pip install Pillow")

    weights_path, weights_label = resolve_weights(args.weights)
    if weights_path is None:
        print("[WARN] No ONNX weights -- triggers fire but no inference.")
        model = None; weights_label = "NO MODEL"
    else:
        print(f"[INFO] Loading: {weights_path}")
        try:
            model = YOLO(weights_path); print("[INFO] Model loaded OK.")
        except Exception as exc:
            print(f"[WARN] Model load failed: {exc}")
            model = None; weights_label = "LOAD FAILED"

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[INFO] Save directory: {args.save_dir}")

    camera = None
    if not args.no_cam and args.camera is not None and args.camera >= 0:
        print(f"[INFO] Opening webcam {args.camera}...")
        camera = CameraCapture(args.camera)
        if camera.start():
            print(f"[INFO] Webcam {args.camera} ready.")
        else:
            print(f"[WARN] Camera {args.camera} open failed -- falling back to synthetic frames.")
            camera = None
    else:
        print("[INFO] Webcam disabled -- using synthetic frames.")

    # GNSS source
    if args.sim:
        gnss_source = GNSSSimulator(speed_kmh=args.sim_speed)
        if args.mode == "fallback":
            gnss_source.drop_fix()
            print("[INFO] Sim starting with no fix (fallback mode).")
        gnss_source.start()
        print(f"[INFO] GNSS: Simulation @ {args.sim_speed:.0f} km/h")
    else:
        auto_scan = not args.no_scan
        gnss_source = NavCastGNSSReader(
            host=args.host, port=args.port, log_fn=print, auto_scan=auto_scan
        )
        gnss_source.start()
        print(f"[INFO] GNSS: NavCast TCP starting (auto_scan={auto_scan})...")

    def _console_log(msg): print(msg)

    engine = StandaloneLitterEngine(
        model=model,
        output_dir=args.save_dir,
        distance_interval_m=args.interval,
        gnss_lost_timeout_sec=args.gnss_timeout,
        time_fallback_sec=args.fallback_interval,
        conf=args.conf,
        log_fn=_console_log,
    )

    root = tk.Tk()
    root.geometry("1280x820")
    app = YOLOTestGUI(
        root=root, engine=engine, gnss_source=gnss_source,
        camera=camera, weights_label=weights_label,
        distance_interval_m=args.interval, output_dir=args.save_dir,
        use_sim=args.sim,
    )
    engine._log = app._log
    if hasattr(gnss_source, "_log"):
        gnss_source._log = app._log

    print(f"[INFO] GUI ready | Trigger every {args.interval:.1f} m | "
          f"GNSS timeout {args.gnss_timeout:.0f}s | "
          f"Clock fallback {args.fallback_interval:.0f}s")
    root.mainloop()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
