"""
session_logger.py — Detailed per-session operational logger for SWSTP Edge Gateway.

Creates one JSON log file per gateway session in logs/sessions/:
  YYYY-MM-DD_HH-MM-SS_<device_id>.json

The log captures:
  - Session metadata  : device_id, session_id, firmware version, start/end timestamps
  - Sensor init status: RTC, IMU, GNSS, Camera
  - Configuration     : ROI polygon, litter distance interval, conf threshold, etc.
  - Stop events       : each vehicle stop with duration, coordinates, capture count
  - Motion captures   : per-capture metadata (path, lat/lon, speed, stop_id)
  - Litter events     : per-inference result with class, confidence, coordinates
  - Session summary   : totals for stops, motion captures, litter events, runtime

A human-readable summary is also appended to logs/device.log for live monitoring.

Usage (in main.py):
    import session_logger
    log = session_logger.SessionLogger(device_id, session_id)
    log.record_sensor_init(rtc_ok, imu_ok, gnss_ok, cam_ok)
    log.record_stop_begin(stop_id, lat, lon, timestamp)
    log.record_stop_end(stop_id, end_ts, duration_sec, n_captures)
    log.record_motion_capture(meta_dict)
    log.record_litter_event(lat, lon, detections, rtc_ts)
    log.finalize()
"""

import datetime
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

# ── Config paths ──────────────────────────────────────────────────────────────
try:
    from config import SESSION_LOG_DIR, DEVICE_LOG_FILE
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    SESSION_LOG_DIR = os.path.join(_HERE, "logs", "sessions")
    DEVICE_LOG_FILE = os.path.join(_HERE, "logs", "device.log")

_VERSION = "1.0.0"
_DEVICE_LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB rolling cap


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _local_ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _write_device_log(line: str) -> None:
    """Append a line to the rolling device.log with timestamp prefix."""
    try:
        os.makedirs(os.path.dirname(DEVICE_LOG_FILE), exist_ok=True)
        # Rolling cap: truncate oldest half if file is too large
        if os.path.isfile(DEVICE_LOG_FILE) and os.path.getsize(DEVICE_LOG_FILE) > _DEVICE_LOG_MAX_BYTES:
            with open(DEVICE_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            with open(DEVICE_LOG_FILE, "w", encoding="utf-8") as f:
                f.write(data[len(data) // 2:])  # keep second half
        with open(DEVICE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{_local_ts()}] {line}\n")
    except Exception:
        pass


# =============================================================================
# SessionLogger
# =============================================================================

class SessionLogger:
    """
    Thread-safe per-session structured logger.
    Call finalize() at program exit to flush and write the JSON log.
    """

    def __init__(self, device_id: str, session_id: int = 0,
                 backend_url: str = "", ulb_id: str = "") -> None:
        self._lock = threading.Lock()
        self._device_id  = device_id or "UNKNOWN"
        self._session_id = session_id
        self._start_mono = time.monotonic()
        self._start_ts   = _now_iso()
        self._start_local = _local_ts()

        os.makedirs(SESSION_LOG_DIR, exist_ok=True)

        # Build log filename
        safe_ts  = self._start_local.replace(":", "-").replace("T", "_")
        safe_dev = "".join(c if (c.isalnum() or c in "-_") else "_" for c in self._device_id)
        self._log_path = os.path.join(SESSION_LOG_DIR, f"{safe_ts}_{safe_dev}.json")

        # ── Internal data structures ──────────────────────────────────────
        self._data: Dict[str, Any] = {
            "schema_version":   "1.0",
            "logger_version":   _VERSION,
            "session_metadata": {
                "device_id":        self._device_id,
                "session_id":       session_id,
                "backend_url":      backend_url,
                "ulb_id":           ulb_id,
                "start_utc":        self._start_ts,
                "start_local":      self._start_local,
                "end_utc":          None,
                "end_local":        None,
                "runtime_sec":      None,
                "exit_reason":      "running",
            },
            "sensor_init": {
                "rtc":    None,
                "imu":    None,
                "gnss":   None,
                "camera": None,
            },
            "configuration": {},
            "stop_events":     [],
            "motion_captures": [],
            "litter_events":   [],
            "session_summary": {
                "total_stops":           0,
                "total_motion_captures": 0,
                "total_litter_events":   0,
                "total_litter_detected": 0,
                "total_frames_captured": 0,
                "runtime_sec":           0.0,
            },
        }

        _write_device_log(
            f"SESSION START | device={self._device_id} | session_id={session_id} | log={self._log_path}"
        )
        print(f"[SESSION LOG] Logging to: {self._log_path}")

    # ── Configuration ─────────────────────────────────────────────────────────

    def record_configuration(self, **kwargs: Any) -> None:
        """Store any configuration key-value pairs (ROI polygon, thresholds, etc.)."""
        with self._lock:
            self._data["configuration"].update(kwargs)

    # ── Sensor init ───────────────────────────────────────────────────────────

    def record_sensor_init(
        self,
        rtc_ok: bool,
        imu_ok: bool,
        gnss_ok: bool,
        camera_ok: bool,
        camera_model: str = "",
        camera_source: str = "",
        camera_resolution: str = "",
    ) -> None:
        """Record hardware sensor initialisation results."""
        with self._lock:
            self._data["sensor_init"] = {
                "rtc":    {"ok": bool(rtc_ok)},
                "imu":    {"ok": bool(imu_ok)},
                "gnss":   {"ok": bool(gnss_ok)},
                "camera": {
                    "ok":         bool(camera_ok),
                    "model":      camera_model,
                    "source":     camera_source,
                    "resolution": camera_resolution,
                },
            }
        status = (
            f"RTC={'OK' if rtc_ok else 'FAIL'} | "
            f"IMU={'OK' if imu_ok else 'FAIL'} | "
            f"GNSS={'OK' if gnss_ok else 'FAIL'} | "
            f"Cam={'OK' if camera_ok else 'FAIL'}"
            + (f" ({camera_model})" if camera_model else "")
        )
        _write_device_log(f"SENSOR INIT | {status}")

    # ── Stop events ───────────────────────────────────────────────────────────

    def record_stop_begin(
        self,
        stop_id: str,
        lat: Optional[float],
        lon: Optional[float],
        timestamp: str = "",
    ) -> None:
        """Record the start of a vehicle stop event."""
        with self._lock:
            ev = {
                "stop_id":        stop_id,
                "begin_utc":      timestamp or _now_iso(),
                "begin_local":    _local_ts(),
                "end_utc":        None,
                "end_local":      None,
                "duration_sec":   None,
                "start_lat":      lat,
                "start_lon":      lon,
                "end_lat":        None,
                "end_lon":        None,
                "motion_capture_count": 0,
                "end_reason":     None,
            }
            self._data["stop_events"].append(ev)
            self._data["session_summary"]["total_stops"] = len(self._data["stop_events"])
        _write_device_log(
            f"STOP BEGIN   | stop_id={stop_id} | lat={lat} lon={lon}"
        )

    def record_stop_end(
        self,
        stop_id: str,
        end_ts: str = "",
        duration_sec: float = 0.0,
        n_captures: int = 0,
        end_lat: Optional[float] = None,
        end_lon: Optional[float] = None,
        end_reason: str = "vehicle_moved",
    ) -> None:
        """Update the stop event record when the vehicle resumes movement."""
        with self._lock:
            for ev in reversed(self._data["stop_events"]):
                if ev["stop_id"] == stop_id:
                    ev["end_utc"]      = end_ts or _now_iso()
                    ev["end_local"]    = _local_ts()
                    ev["duration_sec"] = round(float(duration_sec), 2)
                    ev["motion_capture_count"] = n_captures
                    ev["end_lat"]      = end_lat
                    ev["end_lon"]      = end_lon
                    ev["end_reason"]   = end_reason
                    break
        _write_device_log(
            f"STOP END     | stop_id={stop_id} | duration={duration_sec:.1f}s "
            f"| captures={n_captures} | reason={end_reason}"
        )

    # ── Motion captures ───────────────────────────────────────────────────────

    def record_motion_capture(self, meta: Dict[str, Any]) -> None:
        """Record a motion capture event (called after each saved frame)."""
        with self._lock:
            entry = {
                "capture_index":       meta.get("saved_count", 0),
                "stop_id":             meta.get("collection_event_id", ""),
                "captured_at_utc":     meta.get("captured_at", _now_iso()),
                "rtc_timestamp":       meta.get("rtc_timestamp", ""),
                "latitude":            meta.get("latitude"),
                "longitude":           meta.get("longitude"),
                "speed_kmh":           meta.get("vehicle_speed_kmh"),
                "stop_duration_sec":   meta.get("stop_duration_sec"),
                "drift_from_origin_m": meta.get("drift_from_origin_m"),
                "image_file":          meta.get("image_file", ""),
                "resolution":          f"{meta.get('width', 0)}x{meta.get('height', 0)}",
            }
            self._data["motion_captures"].append(entry)
            self._data["session_summary"]["total_motion_captures"] = len(
                self._data["motion_captures"]
            )
            self._data["session_summary"]["total_frames_captured"] = len(
                self._data["motion_captures"]
            )
        _write_device_log(
            f"MOTION CAPTURE #{entry['capture_index']} | "
            f"stop={entry['stop_id']} | "
            f"lat={entry['latitude']} lon={entry['longitude']} | "
            f"speed={entry['speed_kmh']} km/h | "
            f"file={entry['image_file']}"
        )

    # ── Litter events ─────────────────────────────────────────────────────────

    def record_litter_event(
        self,
        lat: Optional[float],
        lon: Optional[float],
        detections: List[Dict[str, Any]],
        rtc_ts: str = "",
        image_file: str = "",
        gnss_source: str = "",
    ) -> None:
        """
        Record a litter inference event.
        detections: list of dicts with keys: class_name, confidence, bbox [x1,y1,x2,y2]
        An empty detections list means no litter was detected in this inference.
        """
        litter_found = len(detections) > 0
        with self._lock:
            entry = {
                "inference_utc":     _now_iso(),
                "rtc_timestamp":     rtc_ts,
                "latitude":          lat,
                "longitude":         lon,
                "gnss_source":       gnss_source,
                "litter_detected":   litter_found,
                "detection_count":   len(detections),
                "detections":        [
                    {
                        "class":      d.get("class_name", d.get("label", "litter")),
                        "confidence": round(float(d.get("confidence", d.get("conf", 0))), 4),
                        "bbox":       d.get("bbox", d.get("box", [])),
                    }
                    for d in detections
                ],
                "image_file":        image_file,
            }
            self._data["litter_events"].append(entry)
            self._data["session_summary"]["total_litter_events"] = len(
                self._data["litter_events"]
            )
            if litter_found:
                self._data["session_summary"]["total_litter_detected"] += 1
        if litter_found:
            classes = ", ".join(d.get("class_name", "litter") for d in detections)
            _write_device_log(
                f"LITTER DETECTED | {len(detections)} object(s): [{classes}] | "
                f"lat={lat} lon={lon} | file={image_file}"
            )

    # ── Finalize ──────────────────────────────────────────────────────────────

    def finalize(self, exit_reason: str = "normal_shutdown") -> None:
        """Flush all data to the session JSON log and write a summary to device.log."""
        runtime = time.monotonic() - self._start_mono
        with self._lock:
            self._data["session_metadata"]["end_utc"]     = _now_iso()
            self._data["session_metadata"]["end_local"]   = _local_ts()
            self._data["session_metadata"]["runtime_sec"] = round(runtime, 1)
            self._data["session_metadata"]["exit_reason"] = exit_reason

            summary = self._data["session_summary"]
            summary["runtime_sec"] = round(runtime, 1)

            try:
                with open(self._log_path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
            except Exception as write_err:
                print(f"[SESSION LOG] Failed to write log: {write_err}")
                return

        _write_device_log(
            f"SESSION END  | device={self._device_id} | "
            f"runtime={runtime:.0f}s | "
            f"stops={summary['total_stops']} | "
            f"motion_captures={summary['total_motion_captures']} | "
            f"litter_inferences={summary['total_litter_events']} | "
            f"litter_detected_frames={summary['total_litter_detected']} | "
            f"exit={exit_reason}"
        )
        print(
            f"\n[SESSION LOG] Session summary written -> {self._log_path}\n"
            f"  Runtime          : {runtime:.0f} s\n"
            f"  Stops observed   : {summary['total_stops']}\n"
            f"  Motion captures  : {summary['total_motion_captures']}\n"
            f"  Litter inferences: {summary['total_litter_events']}\n"
            f"  Litter detections: {summary['total_litter_detected']} frames with >= 1 object\n"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def log_path(self) -> str:
        return self._log_path

    @property
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data["session_summary"])
