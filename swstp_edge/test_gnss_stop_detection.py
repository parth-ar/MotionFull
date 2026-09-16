#!/usr/bin/env python3
"""
test_gnss_stop_detection.py — Standalone GNSS & IMU Stop Detection Test Harness.

OVERVIEW:
---------
A completely self-contained test file for vehicle stop detection.
Designed to be run independently with ZERO external module requirements:
  - Does NOT import config.py, stop_detector.py, sensors.gnss, telemetry.py, main.py, cv2, torch, etc.
  - Requires only Python 3.8+ standard library (pyserial is strictly optional for COM port mode).
  - Keeps all other repository files completely untouched.

KEY CAPABILITIES:
-----------------
1. HEAVY GNSS LOGIC WITH OPTIONAL IMU FUSION:
   - Primary driver: GNSS speed, coordinates, fix status, and stationary drift tracking.
   - IMU contribution: Supports real hardware IMU (if present) OR built-in Synthetic IMU
     (to test fused GNSS + IMU cross-validation without physical hardware on hand).
   - Seamless GNSS-only fallback when IMU is not connected.
2. STANDALONE STOP DETECTOR STATE MACHINE:
   - Rest detection with configurable threshold (< 3.0 km/h) and debouncing (1.0s).
   - Continuous stop timer tracking (no idle camera timeout).
   - Real-time GPS stationary drift measurement (distance in meters from stop origin).
   - Motion resume detection with speed gate (> 5.0 km/h) and debouncing (0.6s).
   - Frame capture logging during active stops with timestamps and offsets.
   - Multi-line comprehensive stop summary reports upon resuming motion.
3. MULTIPLE TESTING MODES:
   - 'sim'     : Automated simulation running realistic driving, deceleration, stop, drift,
                 camera capture, transient noise spikes, and acceleration.
   - 'tcp'     : Connect to NavCast Android app streaming NMEA over USB tethering / Wi-Fi.
   - 'udp'     : Listen for UDP NMEA broadcasts.
   - 'serial'  : Read from physical GPS USB / COM port (if pyserial installed).
   - 'manual'  : Interactive terminal to type speeds and inject events in real time.
   - 'file'    : Replay a recorded NMEA text file or log.

USAGE EXAMPLES:
---------------
  # 1. Run automated simulation (instant, no hardware needed):
  python test_gnss_stop_detection.py --mode sim

  # 2. Run simulation with synthetic IMU enabled (testing dual fusion):
  python test_gnss_stop_detection.py --mode sim --imu synthetic

  # 3. Connect to live NavCast GNSS on phone (USB tethering):
  python test_gnss_stop_detection.py --mode tcp --host 10.208.43.190 --port 10110

  # 4. Interactive manual testing (type speeds live):


  # 5. Replay recorded NMEA file:
  python test_gnss_stop_detection.py --mode file --file my_drive.nmea
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import datetime
import math
import os
import random
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Optional dependency: pyserial for COM port access
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


# ============================================================================
# TUNABLE PARAMETERS — Adjust these to calibrate stop detection
# ============================================================================
# Speed thresholds in km/h
REST_SPEED_THRESHOLD_KMH: float = 3.0     # Speed below which vehicle is a rest candidate
STOP_SPEED_GATE: float          = 5.0     # Speed above which vehicle is in motion

# Debounce durations in seconds
REST_DEBOUNCE_SEC: float        = 1.0     # Sustained rest required to confirm STOPPED state
MOTION_DEBOUNCE_SEC: float      = 0.6     # Sustained motion required to confirm MOVING state

# IMU tolerances (used when IMU is connected or synthetic IMU is enabled)
IMU_REST_ACCEL_TOLERANCE: float = 0.45    # m/s² max dynamic acceleration deviation (|a| - 9.80665)
IMU_REST_GYRO_TOLERANCE: float  = 4.0     # deg/s max total angular rate sqrt(gx² + gy² + gz²)

# Advanced GNSS filtering & drift
SPEED_SMOOTHING_WINDOW: int     = 1       # Number of readings to average (1 = raw instantaneous speed)
REQUIRE_GPS_FIX: bool           = True    # Require active GPS fix for stop detection
MAX_STATIONARY_DRIFT_M: float   = 25.0    # Alert threshold for GPS drift radius while stopped

# Default NavCast connection parameters
DEFAULT_NAVCAST_HOST: str       = "10.208.43.190"
DEFAULT_NAVCAST_PORT: int       = 10110


# ============================================================================
# Geodesy & Math Utilities (Self-Contained)
# ============================================================================
def haversine_distance_m(
    lat1: Optional[float], lon1: Optional[float],
    lat2: Optional[float], lon2: Optional[float]
) -> float:
    """Calculate great-circle distance between two GPS coordinates in meters."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    r = 6371000.0  # Earth radius in meters
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def format_duration(duration_sec: float) -> str:
    """Formats seconds into readable mm:ss or hh:mm:ss string."""
    sec = int(max(0.0, duration_sec))
    mins, s = divmod(sec, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}h {mins:02d}m {s:02d}s ({duration_sec:.1f}s)"
    return f"{mins:02d}m {s:02d}s ({duration_sec:.1f}s)"


# ============================================================================
# Data Structures
# ============================================================================
@dataclasses.dataclass
class StopCaptureItem:
    """Represents a single motion frame captured during an active stop event."""
    capture_seq: int
    stop_frame_idx: int
    filename: str
    local_path: str
    rtc_timestamp: str
    stop_offset_sec: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capture_seq": self.capture_seq,
            "stop_frame_idx": self.stop_frame_idx,
            "filename": self.filename,
            "local_path": self.local_path,
            "rtc_timestamp": self.rtc_timestamp,
            "stop_offset_sec": round(self.stop_offset_sec, 2),
            "latitude": round(self.latitude, 8) if self.latitude is not None else None,
            "longitude": round(self.longitude, 8) if self.longitude is not None else None,
        }


@dataclasses.dataclass
class StopEventRecord:
    """Record of an individual vehicle stop session."""
    stop_id: int
    start_mono: float
    start_rtc: str
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    end_mono: Optional[float] = None
    end_rtc: Optional[str] = None
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    captures: List[StopCaptureItem] = dataclasses.field(default_factory=list)
    duration_sec: float = 0.0
    end_reason: str = ""
    trigger_speed_kmh: Optional[float] = None
    max_drift_m: float = 0.0

    def get_duration_formatted(self) -> str:
        return format_duration(self.duration_sec)

    def format_summary(self) -> str:
        """Generates a structured, multi-line summary report of the concluded stop."""
        lines = [
            "=" * 70,
            f"[STOP EVENT CONCLUDED] Vehicle Resumed Motion -- Stop #{self.stop_id} Summary Report",
            "-" * 70,
            f"  Stop Index:             Stop #{self.stop_id}",
            f"  Start Time (RTC):       {self.start_rtc}",
            f"  End Time (RTC):         {self.end_rtc or 'N/A'}",
            f"  Total Stopped Duration: {self.get_duration_formatted()}",
            (
                f"  Start Coordinates:      ({self.start_lat:.8f}, {self.start_lon:.8f})"
                if (self.start_lat is not None and self.start_lon is not None)
                else "  Start Coordinates:      N/A"
            ),
            (
                f"  End Coordinates:        ({self.end_lat:.8f}, {self.end_lon:.8f})"
                if (self.end_lat is not None and self.end_lon is not None)
                else "  End Coordinates:        N/A"
            ),
            f"  Stationary GNSS Drift:  Max displacement {self.max_drift_m:.2f} m from stop origin",
            f"  Conclusion Trigger:     {self.end_reason}",
            f"  Motion Frames Captured: {len(self.captures)} frame(s)",
        ]

        if self.captures:
            lines.append("  --------------------------------------------------------------------")
            lines.append("  Motion Frames Captured During This Stop:")
            for item in self.captures:
                coords = (
                    f"({item.latitude:.6f}, {item.longitude:.6f})"
                    if item.latitude is not None else "(N/A)"
                )
                lines.append(
                    f"    [{item.stop_frame_idx:02d}] {item.filename} | "
                    f"Offset: +{item.stop_offset_sec:.1f}s | RTC: {item.rtc_timestamp} | Coords: {coords}"
                )
        else:
            lines.append("  No motion frames triggered camera capture during this stop.")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# Vehicle Stop Detector (GNSS-Heavy with Optional IMU Fusion)
# ============================================================================
class VehicleStopDetector:
    """
    State machine managing MOVING vs STOPPED states.

    Logic:
      1. REST DETECTION:
         - GNSS speed < REST_SPEED_THRESHOLD_KMH (default 3.0 km/h).
         - If IMU is present: cross-validate that dynamic acceleration < IMU_REST_ACCEL_TOLERANCE
           and gyro rate < IMU_REST_GYRO_TOLERANCE.
         - If IMU is not present (or disabled): GNSS speed is the sole rest detector.
         - Debounced for REST_DEBOUNCE_SEC (default 1.0s) -> Enters STOPPED state.

      2. STOPPED STATE:
         - Stop timer runs continuously without idle timeout.
         - Monitors GPS coordinates to track stationary GPS drift radius.
         - Records any camera captures or events.

      3. MOTION RESUME:
         - GNSS speed > STOP_SPEED_GATE (default 5.0 km/h).
         - If IMU is present: confirms dynamic motion OR high GPS speed (> gate + 1.5 km/h).
         - If IMU is not present: GNSS speed > STOP_SPEED_GATE alone confirms candidate motion.
         - Debounced for MOTION_DEBOUNCE_SEC (default 0.6s) -> Enters MOVING state.
         - Concludes stop session and emits full summary report.
    """

    def __init__(
        self,
        stop_speed_gate: float = STOP_SPEED_GATE,
        rest_speed_threshold: float = REST_SPEED_THRESHOLD_KMH,
        imu_accel_tolerance: float = IMU_REST_ACCEL_TOLERANCE,
        imu_gyro_tolerance: float = IMU_REST_GYRO_TOLERANCE,
        rest_debounce_sec: float = REST_DEBOUNCE_SEC,
        motion_debounce_sec: float = MOTION_DEBOUNCE_SEC,
        speed_smoothing_window: int = SPEED_SMOOTHING_WINDOW,
        require_gps_fix: bool = REQUIRE_GPS_FIX,
    ):
        self.stop_speed_gate = float(stop_speed_gate)
        self.rest_speed_threshold = float(rest_speed_threshold)
        self.imu_accel_tolerance = float(imu_accel_tolerance)
        self.imu_gyro_tolerance = float(imu_gyro_tolerance)
        self.rest_debounce_sec = float(rest_debounce_sec)
        self.motion_debounce_sec = float(motion_debounce_sec)
        self.speed_smoothing_window = max(1, int(speed_smoothing_window))
        self.require_gps_fix = bool(require_gps_fix)

        self.state: str = "MOVING"  # "MOVING" | "STOPPED"
        self._current_stop_id: int = 0
        self.active_stop: Optional[StopEventRecord] = None
        self.history: List[StopEventRecord] = []

        # Debounce tracking
        self._rest_candidate_start: Optional[float] = None
        self._motion_candidate_start: Optional[float] = None
        self._last_completed_event: Optional[StopEventRecord] = None

        # Speed filter buffer
        self._speed_buffer: collections.deque[float] = collections.deque(
            maxlen=self.speed_smoothing_window
        )

    @property
    def is_stopped(self) -> bool:
        return self.state == "STOPPED"

    def get_stop_duration(self, now_mono: Optional[float] = None) -> float:
        if self.state == "STOPPED" and self.active_stop is not None:
            if now_mono is None:
                now_mono = time.monotonic()
            return max(0.0, now_mono - self.active_stop.start_mono)
        return 0.0

    @property
    def current_stop_duration(self) -> float:
        return self.get_stop_duration()

    def get_smoothed_speed(self, raw_speed: float) -> float:
        """Applies rolling average to speed readings if smoothing window > 1."""
        self._speed_buffer.append(raw_speed)
        return sum(self._speed_buffer) / len(self._speed_buffer)

    def evaluate_sensors(
        self,
        speed_kmh: Optional[float],
        imu_data: Optional[Dict[str, Any]],
        gps_valid: bool = True,
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Evaluate instantaneous sensor state for candidate rest and candidate motion.

        Returns:
          (is_rest_candidate, is_motion_candidate, metrics_dict)
        """
        raw_speed = float(speed_kmh or 0.0)
        eff_speed = self.get_smoothed_speed(raw_speed)

        # GPS validity check
        effective_gps_valid = gps_valid if self.require_gps_fix else True

        gnss_at_rest = eff_speed < self.rest_speed_threshold
        gnss_in_motion = eff_speed > self.stop_speed_gate

        # IMU evaluation
        imu_valid = bool(imu_data and imu_data.get("valid"))
        dyn_accel = 0.0
        gyro_mag = 0.0
        imu_at_rest = False
        imu_in_motion = False

        if imu_valid:
            # Dynamic acceleration magnitude deviation from 1G (9.80665 m/s²)
            accel_mag = float(imu_data.get("accel_magnitude_ms2") or 9.80665)
            dyn_accel = abs(accel_mag - 9.80665)

            # Gyroscope total angular rate sqrt(gx² + gy² + gz²) in °/s
            gyro = imu_data.get("gyro_dps") or {}
            gx = float(gyro.get("x") or 0.0)
            gy = float(gyro.get("y") or 0.0)
            gz = float(gyro.get("z") or 0.0)
            gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)

            imu_at_rest = (
                dyn_accel < self.imu_accel_tolerance
                and gyro_mag < self.imu_gyro_tolerance
            )
            imu_in_motion = (
                dyn_accel >= (self.imu_accel_tolerance * 1.2)
                or gyro_mag >= (self.imu_gyro_tolerance * 1.1)
            )

        # Fused / Fallback Decision
        if effective_gps_valid and imu_valid:
            # Dual cross-validation: Both GNSS & IMU agree
            is_rest_candidate = gnss_at_rest and imu_at_rest
            is_motion_candidate = (
                (gnss_in_motion and imu_in_motion)
                or (eff_speed > (self.stop_speed_gate + 1.5))
            )
        elif effective_gps_valid and not imu_valid:
            # Pure GNSS operation (primary mode when IMU hardware is absent)
            is_rest_candidate = gnss_at_rest
            is_motion_candidate = gnss_in_motion
        elif not effective_gps_valid and imu_valid:
            # Inertial-only fallback (e.g. tunnel / basement)
            is_rest_candidate = imu_at_rest
            is_motion_candidate = imu_in_motion
        else:
            # Neither valid
            is_rest_candidate = False
            is_motion_candidate = False

        metrics = {
            "raw_speed_kmh": raw_speed,
            "eff_speed_kmh": eff_speed,
            "gps_valid": gps_valid,
            "imu_valid": imu_valid,
            "dyn_accel": round(dyn_accel, 4),
            "gyro_mag": round(gyro_mag, 4),
            "gnss_at_rest": gnss_at_rest,
            "gnss_in_motion": gnss_in_motion,
            "imu_at_rest": imu_at_rest,
            "imu_in_motion": imu_in_motion,
            "is_rest_candidate": is_rest_candidate,
            "is_motion_candidate": is_motion_candidate,
        }
        return is_rest_candidate, is_motion_candidate, metrics

    def update(
        self,
        vehicle_speed: Optional[float],
        imu_data: Optional[Dict[str, Any]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        rtc_timestamp: Optional[str] = None,
        gps_valid: bool = True,
        now_mono: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Main update method called on every sensor tick / incoming NMEA sentence.
        Handles debounced state transitions, drift tracking, and timer updates.
        """
        if now_mono is None:
            now_mono = time.monotonic()

        rtc_str = rtc_timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        is_rest_cand, is_motion_cand, metrics = self.evaluate_sensors(
            speed_kmh=vehicle_speed,
            imu_data=imu_data,
            gps_valid=gps_valid,
        )

        event_just_started = False
        event_just_ended = False
        self._last_completed_event = None
        drift_from_origin_m = 0.0

        if self.state == "MOVING":
            if is_rest_cand:
                if self._rest_candidate_start is None:
                    self._rest_candidate_start = now_mono
                elif (now_mono - self._rest_candidate_start) >= self.rest_debounce_sec:
                    # Vehicle confirmed at rest -> Enter STOPPED state
                    self.state = "STOPPED"
                    self._rest_candidate_start = None
                    self._motion_candidate_start = None
                    self._current_stop_id += 1

                    self.active_stop = StopEventRecord(
                        stop_id=self._current_stop_id,
                        start_mono=now_mono,
                        start_rtc=rtc_str,
                        start_lat=latitude,
                        start_lon=longitude,
                    )
                    event_just_started = True

                    print("\n" + "=" * 70)
                    print(f"[VEHICLE REST DETECTED] Stop #{self._current_stop_id} Started")
                    print("-" * 70)
                    print(f"  Timestamp (RTC): {rtc_str}")
                    if latitude is not None and longitude is not None:
                        print(f"  Start Location:  ({latitude:.8f}, {longitude:.8f})")
                    print(
                        f"  GNSS Speed:      {metrics['eff_speed_kmh']:.1f} km/h "
                        f"(Rest Threshold: < {self.rest_speed_threshold:.1f} km/h)"
                    )
                    if metrics["imu_valid"]:
                        print(
                            f"  IMU Status:      Active & Stationary "
                            f"(dyn_accel: {metrics['dyn_accel']:.3f} m/s2, gyro: {metrics['gyro_mag']:.2f} deg/s)"
                        )
                    else:
                        print("  IMU Status:      Not Active (Solely relying on GNSS)")
                    print("  Stop Timer:      Running. Monitoring for captures & motion resume...")
                    print("=" * 70 + "\n")
            else:
                self._rest_candidate_start = None

        elif self.state == "STOPPED":
            if self.active_stop is not None:
                # Update continuous duration
                self.active_stop.duration_sec = max(0.0, now_mono - self.active_stop.start_mono)

                # Compute drift from stop origin
                if (
                    latitude is not None and longitude is not None
                    and self.active_stop.start_lat is not None
                    and self.active_stop.start_lon is not None
                ):
                    drift_from_origin_m = haversine_distance_m(
                        self.active_stop.start_lat, self.active_stop.start_lon,
                        latitude, longitude
                    )
                    if drift_from_origin_m > self.active_stop.max_drift_m:
                        self.active_stop.max_drift_m = drift_from_origin_m

            # Check for vehicle resuming motion
            if is_motion_cand:
                if self._motion_candidate_start is None:
                    self._motion_candidate_start = now_mono
                elif (now_mono - self._motion_candidate_start) >= self.motion_debounce_sec:
                    # Vehicle confirmed back in motion -> Conclude STOPPED state
                    self.state = "MOVING"
                    self._motion_candidate_start = None
                    self._rest_candidate_start = None

                    if self.active_stop is not None:
                        self.active_stop.end_mono = now_mono
                        self.active_stop.end_rtc = rtc_str
                        self.active_stop.end_lat = latitude
                        self.active_stop.end_lon = longitude
                        self.active_stop.duration_sec = max(
                            0.0, now_mono - self.active_stop.start_mono
                        )
                        self.active_stop.trigger_speed_kmh = metrics["eff_speed_kmh"]

                        if metrics["imu_valid"]:
                            imu_info = (
                                f" | GNSS & IMU confirmed "
                                f"[dyn_accel: {metrics['dyn_accel']:.2f} m/s2, gyro: {metrics['gyro_mag']:.1f} deg/s]"
                            )
                        else:
                            imu_info = " | GNSS speed confirmed (IMU inactive)"

                        self.active_stop.end_reason = (
                            f"Vehicle back in motion (Speed: {metrics['eff_speed_kmh']:.1f} km/h > "
                            f"{self.stop_speed_gate:.1f} km/h{imu_info})"
                        )

                        # Output comprehensive summary report
                        print("\n" + self.active_stop.format_summary() + "\n")

                        self._last_completed_event = self.active_stop
                        self.history.append(self.active_stop)
                        self.active_stop = None
                        event_just_ended = True
            else:
                self._motion_candidate_start = None

        # Debounce progress calculations for status display
        rest_debounce_progress = 0.0
        if self.state == "MOVING" and self._rest_candidate_start is not None:
            rest_debounce_progress = min(
                1.0, (now_mono - self._rest_candidate_start) / max(0.001, self.rest_debounce_sec)
            )

        motion_debounce_progress = 0.0
        if self.state == "STOPPED" and self._motion_candidate_start is not None:
            motion_debounce_progress = min(
                1.0, (now_mono - self._motion_candidate_start) / max(0.001, self.motion_debounce_sec)
            )

        current_duration = self.active_stop.duration_sec if self.active_stop else 0.0
        current_captures = len(self.active_stop.captures) if self.active_stop else 0
        current_id = self.active_stop.stop_id if self.active_stop else (
            self._current_stop_id if self.state == "STOPPED" else 0
        )

        return {
            "state": self.state,
            "is_stopped": self.is_stopped,
            "duration_sec": current_duration,
            "duration_formatted": format_duration(current_duration),
            "stop_event_id": current_id,
            "capture_count": current_captures,
            "drift_from_origin_m": round(drift_from_origin_m, 2),
            "max_drift_m": round(self.active_stop.max_drift_m, 2) if self.active_stop else 0.0,
            "event_just_started": event_just_started,
            "event_just_ended": event_just_ended,
            "last_completed_event": self._last_completed_event,
            "active_stop": self.active_stop,
            "rest_debounce_progress": rest_debounce_progress,
            "motion_debounce_progress": motion_debounce_progress,
            "metrics": metrics,
        }

    def record_capture(
        self,
        meta: Optional[Dict[str, Any]] = None,
        now_mono: Optional[float] = None,
    ) -> Optional[StopCaptureItem]:
        """Registers a motion capture item during an active stop."""
        if self.state != "STOPPED" or self.active_stop is None:
            print("[CAPTURE IGNORED] Cannot record capture while vehicle is MOVING.")
            return None

        if now_mono is None:
            now_mono = time.monotonic()
        if meta is None:
            meta = {}

        offset = max(0.0, now_mono - self.active_stop.start_mono)
        stop_frame_idx = len(self.active_stop.captures) + 1
        filename = meta.get("filename") or f"motion_frame_stop{self.active_stop.stop_id:03d}_{stop_frame_idx:02d}.jpg"
        rtc_ts = meta.get("rtc_timestamp") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        item = StopCaptureItem(
            capture_seq=meta.get("capture_seq", stop_frame_idx),
            stop_frame_idx=stop_frame_idx,
            filename=filename,
            local_path=meta.get("local_path", f"./captures/{filename}"),
            rtc_timestamp=rtc_ts,
            stop_offset_sec=offset,
            latitude=meta.get("latitude"),
            longitude=meta.get("longitude"),
        )
        self.active_stop.captures.append(item)

        print(
            f"[STOP #{self.active_stop.stop_id} | CAPTURE #{stop_frame_idx}] "
            f"Logged frame '{item.filename}' @ +{offset:.1f}s into stop "
            f"(Total captures in this stop: {stop_frame_idx})"
        )
        return item

    def get_active_stop(self) -> Optional[StopEventRecord]:
        return self.active_stop

    def get_history(self) -> List[StopEventRecord]:
        return list(self.history)


# ============================================================================
# Synthetic IMU Provider (For Testing Dual Fusion Without Physical IMU)
# ============================================================================
class SyntheticIMUProvider:
    """
    Generates realistic synthetic IMU readings correlated with vehicle speed.
    Allows testing dual GNSS + IMU fusion algorithms when physical IMU is not present.
    """

    def __init__(self, mode: str = "none"):
        # mode: "none" | "synthetic" | "hardware"
        self.mode = mode.lower()

    def get_imu_reading(self, vehicle_speed_kmh: float) -> Optional[Dict[str, Any]]:
        if self.mode == "none":
            return None

        if self.mode == "synthetic":
            is_stopped = vehicle_speed_kmh < 1.0
            if is_stopped:
                # Stationary vehicle: minor engine idle / vibration
                dyn_noise = random.uniform(-0.08, 0.08)
                accel_mag = 9.80665 + dyn_noise
                gx = random.uniform(-0.4, 0.4)
                gy = random.uniform(-0.4, 0.4)
                gz = random.uniform(-0.3, 0.3)
            else:
                # Vehicle in motion: road bumps, turns, acceleration
                speed_factor = min(3.0, vehicle_speed_kmh / 20.0)
                dyn_noise = random.uniform(0.6, 1.8) * speed_factor
                accel_mag = 9.80665 + (dyn_noise if random.random() > 0.4 else -dyn_noise * 0.5)
                gx = random.uniform(-5.0, 5.0) * speed_factor
                gy = random.uniform(-6.0, 6.0) * speed_factor
                gz = random.uniform(-4.0, 4.0) * speed_factor

            return {
                "valid": True,
                "accel_magnitude_ms2": round(accel_mag, 4),
                "gyro_dps": {
                    "x": round(gx, 2),
                    "y": round(gy, 2),
                    "z": round(gz, 2),
                },
                "source": "synthetic",
            }

        return None


# ============================================================================
# Self-Contained NMEA Parser
# ============================================================================
class NMEAParser:
    """Lightweight, self-contained NMEA 0183 sentence parser."""

    @staticmethod
    def verify_checksum(sentence: str) -> bool:
        if not sentence.startswith("$"):
            return False
        if "*" not in sentence:
            return True
        try:
            body, csum_hex = sentence.lstrip("$").rsplit("*", 1)
            computed = 0
            for char in body:
                computed ^= ord(char)
            return computed == int(csum_hex[:2], 16)
        except Exception:
            return False

    @staticmethod
    def parse_coordinate(raw: str, direction: str) -> Optional[float]:
        """Converts NMEA ddmm.mmmm (lat) or dddmm.mmmm (lon) to decimal degrees."""
        if not raw or not direction:
            return None
        try:
            raw = raw.strip()
            dot_idx = raw.index(".")
            deg_digits = dot_idx - 2
            deg = float(raw[:deg_digits])
            mins = float(raw[deg_digits:])
            decimal = deg + mins / 60.0
            if direction.upper() in ("S", "W"):
                decimal = -decimal
            return round(decimal, 8)
        except Exception:
            return None

    def parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Parses a single NMEA sentence ($RMC, $GGA, $VTG).
        Returns a dict of extracted fields.
        """
        result: Dict[str, Any] = {"valid_sentence": False}
        sentence = sentence.strip()
        if not sentence.startswith("$"):
            return result
        if not self.verify_checksum(sentence):
            return result

        result["valid_sentence"] = True
        body = sentence.lstrip("$").split("*")[0]
        fields = body.split(",")
        if not fields:
            return result

        talker_type = fields[0].upper()

        # ── RMC: Recommended Minimum Navigation Information ────────────────
        if talker_type.endswith("RMC"):
            result["type"] = "RMC"
            if len(fields) >= 9:
                status = fields[2].upper()
                result["status"] = status
                result["fix"] = (status == "A")
                # Always extract coordinates and speed, even when fix is void ('V').
                # Many phones (NavCast included) report valid speed in void RMC sentences.
                if fields[3] and fields[4]:
                    result["latitude"] = self.parse_coordinate(fields[3], fields[4])
                if fields[5] and fields[6]:
                    result["longitude"] = self.parse_coordinate(fields[5], fields[6])
                try:
                    speed_knots = float(fields[7]) if fields[7] else 0.0
                    result["speed_kmh"] = round(speed_knots * 1.852, 2)
                except ValueError:
                    result["speed_kmh"] = 0.0
                try:
                    result["course_deg"] = float(fields[8]) if fields[8] else 0.0
                except ValueError:
                    result["course_deg"] = 0.0

        # ── GGA: Global Positioning System Fix Data ────────────────────────
        elif talker_type.endswith("GGA"):
            result["type"] = "GGA"
            if len(fields) >= 10:
                try:
                    fix_quality = int(fields[6]) if fields[6] else 0
                except ValueError:
                    fix_quality = 0
                result["fix"] = (fix_quality > 0)
                result["fix_quality"] = fix_quality
                result["latitude"] = self.parse_coordinate(fields[2], fields[3])
                result["longitude"] = self.parse_coordinate(fields[4], fields[5])
                try:
                    result["satellites"] = int(fields[7]) if fields[7] else 0
                except ValueError:
                    result["satellites"] = 0
                try:
                    result["hdop"] = float(fields[8]) if fields[8] else None
                except ValueError:
                    result["hdop"] = None
                try:
                    result["altitude_m"] = float(fields[9]) if fields[9] else None
                except ValueError:
                    result["altitude_m"] = None

        # ── VTG: Track Made Good and Ground Speed ───────────────────────────
        elif talker_type.endswith("VTG"):
            result["type"] = "VTG"
            if len(fields) >= 8:
                try:
                    # Field 7 is speed in km/h
                    result["speed_kmh"] = float(fields[7]) if fields[7] else 0.0
                except ValueError:
                    pass

        return result


# ============================================================================
# Terminal UI / Dashboard Helper
# ============================================================================
def render_progress_bar(val: float, total: float, width: int = 10) -> str:
    """Renders a simple ASCII progress bar [####......]."""
    if total <= 0:
        return "[" + " " * width + "]"
    ratio = min(1.0, max(0.0, val / total))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_status_line(
    timestamp_str: str,
    speed: float,
    state: str,
    duration: float,
    drift_m: float,
    captures: int,
    rest_prog: float,
    motion_prog: float,
    imu_status: str,
    fix_tag: str = "FIX",
) -> None:
    """Prints a structured one-line telemetry & stop status update."""
    state_badge = f"\033[91m[{state:7s}]\033[0m" if state == "STOPPED" else f"\033[92m[{state:7s}]\033[0m"
    # Dim the fix tag when void so the user can immediately see GPS quality
    fix_display = f"\033[93m{fix_tag:4s}\033[0m" if fix_tag != "FIX" else f"\033[92m{fix_tag:3s}\033[0m"

    debounce_info = ""
    if state == "MOVING" and rest_prog > 0:
        bar = render_progress_bar(rest_prog, 1.0, 8)
        debounce_info = f" | Rest Debounce: {bar} {rest_prog * REST_DEBOUNCE_SEC:.1f}/{REST_DEBOUNCE_SEC:.1f}s"
    elif state == "STOPPED":
        bar = render_progress_bar(motion_prog, 1.0, 8)
        motion_str = f" | Motion Debounce: {bar} {motion_prog * MOTION_DEBOUNCE_SEC:.1f}/{MOTION_DEBOUNCE_SEC:.1f}s" if motion_prog > 0 else ""
        debounce_info = f" | Duration: {format_duration(duration)} | Drift: {drift_m:.1f}m | Captures: {captures}{motion_str}"

    print(
        f"[{timestamp_str}] {state_badge} GPS:{fix_display} Speed: {speed:5.1f} km/h | IMU: {imu_status:9s}{debounce_info}",
        flush=True,
    )


# ============================================================================
# Mode 1: Automated Realistic Simulation Suite
# ============================================================================
def run_simulation(detector: VehicleStopDetector, imu_provider: SyntheticIMUProvider) -> None:
    """
    Runs a realistic simulated driving session demonstrating:
      1. Cruising at 35 km/h (MOVING state).
      2. Decelerating to rest (triggers rest debouncer).
      3. STOPPED state confirmed -> stop timer starts at 0.0s.
      4. Stationary GPS drift jitter (0.0 to 1.5 km/h noise + coordinate wander).
      5. Frame captures recorded during the stop.
      6. Transient noise spike (brief 6.5 km/h spike that is correctly ignored by motion debouncer).
      7. Sustained acceleration (> 5.0 km/h for > 0.6s) -> enters MOVING state.
      8. Concludes stop event and outputs comprehensive summary report.
    """
    print("\n" + "=" * 75)
    print("  RUNNING AUTOMATED GNSS VEHICLE STOP DETECTION SIMULATION")
    print("=" * 75)
    print("Configuration:")
    print(f"  Rest Speed Threshold:   < {detector.rest_speed_threshold:.1f} km/h")
    print(f"  Stop Speed Gate:        > {detector.stop_speed_gate:.1f} km/h")
    print(f"  Rest Debounce Time:     {detector.rest_debounce_sec:.1f} s")
    print(f"  Motion Debounce Time:   {detector.motion_debounce_sec:.1f} s")
    print(f"  IMU Mode:               {imu_provider.mode.upper()}")
    print("=" * 75 + "\n")

    base_lat = 20.932000
    base_lon = 77.752300

    # Simulation timeline profile: (duration_sec, start_speed, end_speed, label)
    profile = [
        (3.0, 35.0, 35.0, "Cruising on main road"),
        (2.0, 35.0, 10.0, "Approaching intersection & braking"),
        (1.0, 10.0, 1.5,  "Coming to a halt (< 3.0 km/h rest candidate)"),
        (4.0, 0.0, 0.0,   "Vehicle stopped at red light (rest confirmed, stop timer running)"),
        (3.0, 0.0, 1.2,   "Stationary GPS drift jitter (0.0 - 1.2 km/h drift noise)"),
        (2.0, 6.2, 6.2,   "Transient GPS speed glitch for 0.3s (debouncer should ignore)"),
        (3.0, 0.0, 0.0,   "Vehicle remains stopped"),
        (1.0, 2.0, 7.5,   "Vehicle resumes motion (> 5.0 km/h motion candidate)"),
        (3.0, 12.0, 30.0, "Driving away (motion confirmed -> stop ends & summary printed)"),
    ]

    tick_rate_hz = 5.0
    tick_dt = 1.0 / tick_rate_hz
    sim_mono = 1000.0
    sim_rtc_base = datetime.datetime(2026, 9, 15, 12, 0, 0)
    elapsed_total = 0.0
    capture_index = 0

    for seg_duration, s_start, s_end, seg_desc in profile:
        steps = int(round(seg_duration * tick_rate_hz))
        print(f"\n>>> [PHASE] {seg_desc} ({seg_duration:.1f}s)")

        for step in range(steps):
            frac = step / max(1, steps)
            speed = s_start + (s_end - s_start) * frac

            # Add subtle GPS jitter
            jitter_lat = base_lat + random.uniform(-0.00001, 0.00001)
            jitter_lon = base_lon + random.uniform(-0.00001, 0.00001)

            sim_mono += tick_dt
            elapsed_total += tick_dt
            current_rtc = (sim_rtc_base + datetime.timedelta(seconds=elapsed_total)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # IMU reading
            imu_reading = imu_provider.get_imu_reading(speed)

            # Special case for transient spike in phase 6: make it short (0.3s)
            if "Transient GPS speed glitch" in seg_desc:
                if step < 2:
                    current_speed = 6.2
                else:
                    current_speed = 0.0
            else:
                current_speed = speed

            status = detector.update(
                vehicle_speed=current_speed,
                imu_data=imu_reading,
                latitude=jitter_lat,
                longitude=jitter_lon,
                rtc_timestamp=current_rtc,
                gps_valid=True,
                now_mono=sim_mono,
            )

            # Trigger sample captures during stopped state
            if detector.is_stopped:
                stop_dur = status["duration_sec"]
                if stop_dur >= 2.0 and capture_index == 0:
                    capture_index += 1
                    detector.record_capture(
                        {
                            "filename": f"trash_pile_{capture_index}.jpg",
                            "latitude": jitter_lat,
                            "longitude": jitter_lon,
                            "rtc_timestamp": current_rtc,
                        },
                        now_mono=sim_mono,
                    )
                elif stop_dur >= 6.0 and capture_index == 1:
                    capture_index += 1
                    detector.record_capture(
                        {
                            "filename": f"bin_overflow_{capture_index}.jpg",
                            "latitude": jitter_lat,
                            "longitude": jitter_lon,
                            "rtc_timestamp": current_rtc,
                        },
                        now_mono=sim_mono,
                    )

            # Print status update
            imu_tag = "ACTIVE" if (imu_reading and imu_reading.get("valid")) else "NONE"
            print_status_line(
                timestamp_str=current_rtc.split(" ")[1],
                speed=current_speed,
                state=status["state"],
                duration=status["duration_sec"],
                drift_m=status["drift_from_origin_m"],
                captures=status["capture_count"],
                rest_prog=status["rest_debounce_progress"],
                motion_prog=status["motion_debounce_progress"],
                imu_status=imu_tag,
            )

            time.sleep(0.05)  # Fast-forward playback for rapid verification

    print("\n" + "=" * 75)
    print("  SIMULATION COMPLETE")
    print(f"  Total Stop Events Recorded: {len(detector.get_history())}")
    print("=" * 75 + "\n")


# ============================================================================
# Automatic Network & Wi-Fi Discovery for NavCast
# ============================================================================
RE_IPV4 = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
DEFAULT_CANDIDATE_PORTS = [10110, 2947, 11123, 50000, 8080]
COMMON_TETHER_GATEWAYS = [
    "192.168.42.129",   # Android USB tethering default
    "192.168.42.1",     # Android USB tethering alternate
    "192.168.43.1",     # Android Wi-Fi hotspot default
    "192.168.44.1",     # Android USB tethering variant
    "172.20.10.1",      # iOS USB / Wi-Fi tethering gateway
]


def get_candidate_network_ips(fallback_host: Optional[str] = None) -> List[str]:
    """
    Assembles an ordered list of candidate phone IP addresses on Wi-Fi or USB tethering:
      1. Active ARP neighbors on the current Wi-Fi subnet (e.g. 10.32.250.155)
      2. Routing table default gateways
      3. Derived subnet gateways from local interface IPs (.1, .129, .254)
      4. User-provided fallback host (e.g. 10.208.43.190)
      5. Common Android/iOS tethering defaults
    """
    candidates: List[str] = []
    seen = set()

    def _add(ip: Optional[str]):
        if (
            ip
            and ip not in seen
            and ip != "0.0.0.0"
            and not ip.startswith("127.")
            and not ip.endswith(".255")
            and not ip.startswith("224.")
            and not ip.startswith("239.")
            and ip != "255.255.255.255"
        ):
            seen.add(ip)
            candidates.append(ip)

    # 1. Probe preferred / fallback host first if given
    if fallback_host:
        _add(fallback_host)

    # 2. ARP table scan (covers Wi-Fi peers and phone IP)
    try:
        cmd = ["arp", "-a"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=1.0).decode("ascii", errors="ignore")
        for match in RE_IPV4.findall(out):
            _add(match)
    except Exception:
        pass

    # 3. Windows route print 0.0.0.0 / Linux ip route
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(["route", "print", "0.0.0.0"], stderr=subprocess.DEVNULL, timeout=1.0).decode("ascii", errors="ignore")
            for match in RE_IPV4.findall(out):
                _add(match)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        for cmd in (["ip", "route"], ["ip", "neighbor"]):
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=1.0).decode("ascii", errors="ignore")
                for match in RE_IPV4.findall(out):
                    _add(match)
            except Exception:
                pass

    # 4. Local network interfaces -> derive local subnet gateways (.1, .129, .254)
    try:
        host_info = socket.gethostbyname_ex(socket.gethostname())
        for iface_ip in host_info[2]:
            parts = iface_ip.split(".")
            if len(parts) == 4 and not iface_ip.startswith("127."):
                prefix = ".".join(parts[:3])
                _add(f"{prefix}.1")
                _add(f"{prefix}.129")
                _add(f"{prefix}.254")
    except Exception:
        pass

    # 5. Common tethering defaults
    for ip in COMMON_TETHER_GATEWAYS:
        _add(ip)

    return candidates


def probe_navcast_server(host: str, port: int, timeout: float = 0.4) -> Tuple[bool, bool]:
    """
    Checks if TCP (host, port) is open and streaming NMEA sentences ($G, $P, GGA, RMC, GSV, VTG).
    Returns (is_open: bool, is_nmea: bool).
    """
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(0.6)
        try:
            chunk = sock.recv(256).decode("ascii", errors="replace")
            is_nmea = any(sig in chunk for sig in ("$G", "$P", "GGA", "RMC", "GSA", "GSV", "VTG"))
            return (True, is_nmea)
        except Exception:
            return (True, False)
    except Exception:
        return (False, False)
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


def discover_navcast(
    preferred_host: Optional[str] = None,
    preferred_port: int = DEFAULT_NAVCAST_PORT,
    timeout: float = 0.4,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Scans candidate IPs concurrently to auto-detect NavCast on Wi-Fi or USB tethering.
    """
    # 1. Quick probe preferred_host first if responsive
    if preferred_host:
        is_open, is_nmea = probe_navcast_server(preferred_host, preferred_port, timeout=0.3)
        if is_open and is_nmea:
            return (preferred_host, preferred_port)

    # 2. Scan all candidate IPs concurrently
    candidates = get_candidate_network_ips(fallback_host=preferred_host)
    if not candidates:
        return (None, None)

    ports = [preferred_port]
    for p in DEFAULT_CANDIDATE_PORTS:
        if p not in ports:
            ports.append(p)

    targets = [(ip, p) for ip in candidates for p in ports]
    best_candidate: Optional[Tuple[str, int]] = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
        future_map = {
            pool.submit(probe_navcast_server, ip, p, timeout): (ip, p)
            for ip, p in targets
        }
        for future in concurrent.futures.as_completed(future_map):
            ip, p = future_map[future]
            try:
                is_open, is_nmea = future.result()
                if is_open:
                    if is_nmea:
                        return (ip, p)
                    elif best_candidate is None:
                        best_candidate = (ip, p)
            except Exception:
                pass

    return best_candidate if best_candidate else (None, None)


# ============================================================================
# Mode 2: Live NavCast TCP Connection (Wi-Fi & USB Tethering)
# ============================================================================
def run_tcp_mode(
    detector: VehicleStopDetector,
    imu_provider: SyntheticIMUProvider,
    host: str,
    port: int,
    auto_detect: bool = True,
    debug: bool = False,
) -> None:
    """Connects to phone NavCast app streaming NMEA sentences over TCP (Wi-Fi or USB tethering)."""
    parser = NMEAParser()
    active_host = host
    active_port = port

    print("\n" + "=" * 70)
    print("  LIVE TCP GNSS STOP DETECTION (Wi-Fi & USB Tethering)")
    if debug:
        print("  [DEBUG MODE] Raw NMEA sentences will be printed.")
    print("=" * 70)

    while True:
        if auto_detect:
            print(f"[TCP GNSS] Auto-detecting NavCast server across Wi-Fi & network interfaces...")
            disc_host, disc_port = discover_navcast(preferred_host=active_host, preferred_port=active_port)
            if disc_host and disc_port:
                print(f"[AUTO-DETECT] ✔ Successfully found NavCast server @ {disc_host}:{disc_port}")
                active_host, active_port = disc_host, disc_port
            else:
                print(f"[AUTO-DETECT] Could not discover active server; will try {active_host}:{active_port}...")

        print(f"[TCP GNSS] Connecting to NavCast at {active_host}:{active_port} ...")
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(6.0)
            sock.connect((active_host, active_port))
            print(f"[TCP GNSS] ✔ Connected! Receiving live NMEA data stream from phone...")
            print("Press Ctrl+C to stop.\n")

            buffer = ""
            current_lat: Optional[float] = None
            current_lon: Optional[float] = None
            current_speed: float = 0.0
            current_fix: bool = False
            # Diagnostic: track fix-loss events and last diagnostic print time
            _last_diag_mono: float = 0.0
            _diag_interval_sec: float = 5.0
            _total_sentences: int = 0
            _fix_lost_count: int = 0
            _last_fix_state: Optional[bool] = None

            while True:
                try:
                    chunk = sock.recv(2048)
                except socket.timeout:
                    # Print a periodic heartbeat so the user can see the connection is alive
                    now = time.monotonic()
                    if (now - _last_diag_mono) >= _diag_interval_sec:
                        fix_str = "ACTIVE" if current_fix else "VOID/NO-FIX"
                        print(
                            f"[TCP GNSS | HEARTBEAT] Speed={current_speed:.1f} km/h "
                            f"Fix={fix_str} | {_total_sentences} sentences received",
                            flush=True,
                        )
                        _last_diag_mono = now
                    continue
                if not chunk:
                    print("\n[TCP GNSS] NavCast connection closed by phone. Auto-reconnecting in 3s...")
                    break
                buffer += chunk.decode("ascii", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if debug and line.startswith("$"):
                        print(f"[NMEA] {line}", flush=True)

                    parsed = parser.parse_sentence(line)
                    if not parsed.get("valid_sentence"):
                        continue

                    _total_sentences += 1

                    if "latitude" in parsed and parsed["latitude"] is not None:
                        current_lat = parsed["latitude"]
                    if "longitude" in parsed and parsed["longitude"] is not None:
                        current_lon = parsed["longitude"]
                    if "fix" in parsed:
                        # Log fix state transitions so the user knows when signal drops
                        new_fix = parsed["fix"]
                        if _last_fix_state is not None and _last_fix_state != new_fix:
                            if not new_fix:
                                _fix_lost_count += 1
                                print(
                                    f"\n[TCP GNSS | WARNING] GPS fix LOST (void RMC). "
                                    f"Speed from VTG still used. Fix losses: {_fix_lost_count}",
                                    flush=True,
                                )
                            else:
                                print("[TCP GNSS | INFO] GPS fix RESTORED.", flush=True)
                        _last_fix_state = new_fix
                        current_fix = new_fix
                    if "speed_kmh" in parsed:
                        current_speed = parsed["speed_kmh"]

                    # Process stop detection update on RMC or GGA or VTG updates.
                    # Use gps_valid=True when we have a speed signal (even from a void RMC
                    # or VTG), since the detector ignores speed entirely when gps_valid=False
                    # with no IMU — causing a frozen/dead state in poor indoor conditions.
                    if parsed.get("type") in ("RMC", "GGA", "VTG"):
                        # Consider speed trustworthy if we have current_fix OR a non-zero
                        # speed from VTG (VTG always carries valid Doppler speed).
                        speed_source_valid = (
                            current_fix
                            or (parsed.get("type") == "VTG" and current_speed > 0.0)
                            or (parsed.get("type") == "RMC" and "speed_kmh" in parsed)
                        )

                        now_mono = time.monotonic()
                        rtc_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        imu_data = imu_provider.get_imu_reading(current_speed)

                        status = detector.update(
                            vehicle_speed=current_speed,
                            imu_data=imu_data,
                            latitude=current_lat,
                            longitude=current_lon,
                            rtc_timestamp=rtc_now,
                            gps_valid=speed_source_valid,
                            now_mono=now_mono,
                        )

                        imu_tag = "ACTIVE" if (imu_data and imu_data.get("valid")) else "NONE"
                        fix_tag = "FIX" if current_fix else "VOID"
                        print_status_line(
                            timestamp_str=rtc_now.split(" ")[1],
                            speed=current_speed,
                            state=status["state"],
                            duration=status["duration_sec"],
                            drift_m=status["drift_from_origin_m"],
                            captures=status["capture_count"],
                            rest_prog=status["rest_debounce_progress"],
                            motion_prog=status["motion_debounce_progress"],
                            imu_status=imu_tag,
                            fix_tag=fix_tag,
                        )

        except (socket.error, OSError) as exc:
            print(f"\n[TCP GNSS ERROR] Could not connect to {active_host}:{active_port} ({exc}).")
            if auto_detect:
                print("[TCP GNSS] Auto-redetecting phone on Wi-Fi in 3 seconds... (Press Ctrl+C to cancel)")
            else:
                print(f"[TCP GNSS] Retrying {active_host}:{active_port} in 3 seconds... (Press Ctrl+C to cancel)")
        except KeyboardInterrupt:
            print("\n[TCP GNSS] Stopped by user.")
            break
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

        try:
            time.sleep(3.0)
        except KeyboardInterrupt:
            print("\n[TCP GNSS] Stopped by user.")
            break


# ============================================================================
# Mode 3: Live Serial / COM Port
# ============================================================================
def run_serial_mode(
    detector: VehicleStopDetector,
    imu_provider: SyntheticIMUProvider,
    port_name: str,
    baudrate: int,
) -> None:
    """Reads NMEA sentences from physical GPS receiver on a COM/Serial port."""
    if not SERIAL_AVAILABLE:
        print("[ERROR] pyserial is required for serial mode. Install with: pip install pyserial")
        return

    print(f"\n[SERIAL GNSS] Opening port {port_name} at {baudrate} baud ...")
    parser = NMEAParser()

    try:
        ser = serial.Serial(port_name, baudrate=baudrate, timeout=1.0)
        print(f"[SERIAL GNSS] Connected to {port_name}! Reading live NMEA...")
        print("Press Ctrl+C to stop.\n")

        current_lat: Optional[float] = None
        current_lon: Optional[float] = None
        current_speed: float = 0.0
        current_fix: bool = False

        while True:
            raw_line = ser.readline()
            if not raw_line:
                continue
            line = raw_line.decode("ascii", errors="replace").strip()
            parsed = parser.parse_sentence(line)
            if not parsed.get("valid_sentence"):
                continue

            if "latitude" in parsed and parsed["latitude"] is not None:
                current_lat = parsed["latitude"]
            if "longitude" in parsed and parsed["longitude"] is not None:
                current_lon = parsed["longitude"]
            if "fix" in parsed:
                current_fix = parsed["fix"]
            if "speed_kmh" in parsed:
                current_speed = parsed["speed_kmh"]

            if parsed.get("type") in ("RMC", "GGA", "VTG"):
                now_mono = time.monotonic()
                rtc_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                imu_data = imu_provider.get_imu_reading(current_speed)

                status = detector.update(
                    vehicle_speed=current_speed,
                    imu_data=imu_data,
                    latitude=current_lat,
                    longitude=current_lon,
                    rtc_timestamp=rtc_now,
                    gps_valid=current_fix,
                    now_mono=now_mono,
                )

                imu_tag = "ACTIVE" if (imu_data and imu_data.get("valid")) else "NONE"
                print_status_line(
                    timestamp_str=rtc_now.split(" ")[1],
                    speed=current_speed,
                    state=status["state"],
                    duration=status["duration_sec"],
                    drift_m=status["drift_from_origin_m"],
                    captures=status["capture_count"],
                    rest_prog=status["rest_debounce_progress"],
                    motion_prog=status["motion_debounce_progress"],
                    imu_status=imu_tag,
                )

    except KeyboardInterrupt:
        print("\n[SERIAL GNSS] Stopped by user.")
    except Exception as exc:
        print(f"[SERIAL GNSS ERROR] {exc}")


# ============================================================================
# Mode 4: Interactive Manual Control
# ============================================================================
def run_manual_mode(
    detector: VehicleStopDetector,
    imu_provider: SyntheticIMUProvider,
) -> None:
    """
    Interactive console runner where the user can enter speeds and commands:
      'speed <val>' or just '<val>' : Set vehicle speed in km/h (e.g. 0, 2.5, 15)
      'cap'                         : Record a simulated motion frame capture
      'drift'                       : Inject 5 meters of GPS stationary drift
      'status'                      : Show detector state and active stop metrics
      'help'                        : Show available commands
      'quit' or 'q'                 : Exit
    """
    print("\n" + "=" * 70)
    print("  INTERACTIVE MANUAL STOP DETECTION TEST CONSOLE")
    print("=" * 70)
    print("Commands:")
    print("  <number>        : Set vehicle speed in km/h (e.g., '0' to stop, '25' to drive)")
    print("  cap             : Record a simulated motion frame capture during an active stop")
    print("  drift <meters>  : Inject stationary GPS coordinate drift")
    print("  status          : Print full current detector status")
    print("  q               : Quit console")
    print("=" * 70 + "\n")

    current_lat = 20.932000
    current_lon = 77.752300
    current_speed = 30.0

    # Background tick thread to keep debouncers and duration updating smoothly
    running = True

    def tick_loop() -> None:
        nonlocal current_speed, current_lat, current_lon
        while running:
            now_mono = time.monotonic()
            rtc_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            imu_data = imu_provider.get_imu_reading(current_speed)

            detector.update(
                vehicle_speed=current_speed,
                imu_data=imu_data,
                latitude=current_lat,
                longitude=current_lon,
                rtc_timestamp=rtc_str,
                gps_valid=True,
                now_mono=now_mono,
            )
            time.sleep(0.2)

    bg_thread = threading.Thread(target=tick_loop, daemon=True)
    bg_thread.start()

    try:
        while True:
            cmd = input("cmd [speed/cap/drift/status/q] > ").strip().lower()
            if not cmd:
                continue

            if cmd in ("q", "quit", "exit"):
                break

            if cmd == "cap":
                now_mono = time.monotonic()
                detector.record_capture(
                    {
                        "filename": f"interactive_capture_{len(detector.active_stop.captures) + 1 if detector.active_stop else 0}.jpg",
                        "latitude": current_lat,
                        "longitude": current_lon,
                    },
                    now_mono=now_mono,
                )
                continue

            if cmd.startswith("drift"):
                parts = cmd.split()
                meters = float(parts[1]) if len(parts) > 1 else 3.0
                # 1 meter ~ 0.000009 degrees
                current_lat += (meters * 0.000009)
                print(f"[DRIFT] Shifted GPS position by +{meters:.1f} m -> ({current_lat:.8f}, {current_lon:.8f})")
                continue

            if cmd == "status":
                dur = detector.current_stop_duration
                print("-" * 50)
                print(f"  Detector State:    {detector.state}")
                print(f"  Vehicle Speed:     {current_speed:.1f} km/h")
                print(f"  Active Stop ID:    {detector.active_stop.stop_id if detector.active_stop else 'None'}")
                print(f"  Stop Duration:     {format_duration(dur)}")
                if detector.active_stop:
                    print(f"  Captures Recorded: {len(detector.active_stop.captures)}")
                    print(f"  Max GPS Drift:     {detector.active_stop.max_drift_m:.2f} m")
                print("-" * 50)
                continue

            # Try parsing as a numeric speed value
            try:
                parts = cmd.replace("speed", "").strip()
                val = float(parts)
                current_speed = max(0.0, val)
                print(f"[SPEED SET] Vehicle speed changed to {current_speed:.1f} km/h.")
                if current_speed < detector.rest_speed_threshold:
                    print(f"  -> Below rest threshold (< {detector.rest_speed_threshold:.1f} km/h). Rest debounce started...")
                elif current_speed > detector.stop_speed_gate:
                    print(f"  -> Above motion gate (> {detector.stop_speed_gate:.1f} km/h). Motion debounce started...")
            except ValueError:
                print(f"Unknown command: '{cmd}'. Enter a speed number (e.g. '0' or '25') or 'cap', 'drift', 'q'.")

    finally:
        running = False


# ============================================================================
# Mode 5: Replay NMEA Log File
# ============================================================================
def run_file_mode(
    detector: VehicleStopDetector,
    imu_provider: SyntheticIMUProvider,
    filepath: str,
    playback_speed: float = 1.0,
) -> None:
    """Replays NMEA sentences from a saved text or log file."""
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return

    print(f"\n[FILE REPLAY] Replaying NMEA log: {filepath} (playback: {playback_speed:.1f}x) ...\n")
    parser = NMEAParser()

    current_lat: Optional[float] = None
    current_lon: Optional[float] = None
    current_speed: float = 0.0
    current_fix: bool = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parser.parse_sentence(line)
            if not parsed.get("valid_sentence"):
                continue

            if "latitude" in parsed and parsed["latitude"] is not None:
                current_lat = parsed["latitude"]
            if "longitude" in parsed and parsed["longitude"] is not None:
                current_lon = parsed["longitude"]
            if "fix" in parsed:
                current_fix = parsed["fix"]
            if "speed_kmh" in parsed:
                current_speed = parsed["speed_kmh"]

            if parsed.get("type") in ("RMC", "GGA", "VTG"):
                now_mono = time.monotonic()
                rtc_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                imu_data = imu_provider.get_imu_reading(current_speed)

                status = detector.update(
                    vehicle_speed=current_speed,
                    imu_data=imu_data,
                    latitude=current_lat,
                    longitude=current_lon,
                    rtc_timestamp=rtc_now,
                    gps_valid=current_fix,
                    now_mono=now_mono,
                )

                imu_tag = "ACTIVE" if (imu_data and imu_data.get("valid")) else "NONE"
                print_status_line(
                    timestamp_str=rtc_now.split(" ")[1],
                    speed=current_speed,
                    state=status["state"],
                    duration=status["duration_sec"],
                    drift_m=status["drift_from_origin_m"],
                    captures=status["capture_count"],
                    rest_prog=status["rest_debounce_progress"],
                    motion_prog=status["motion_debounce_progress"],
                    imu_status=imu_tag,
                )

                time.sleep(0.1 / max(0.1, playback_speed))

    print("\n[FILE REPLAY] Replay complete.")


# ============================================================================
# Main CLI Entrypoint
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone GNSS & IMU Stop Detection Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Run automated simulation (no hardware needed):
  python test_gnss_stop_detection.py --mode sim

  # 2. Run simulation with synthetic IMU fusion enabled:
  python test_gnss_stop_detection.py --mode sim --imu synthetic

  # 3. Connect to live NavCast phone app:
  python test_gnss_stop_detection.py --mode tcp --host 10.208.43.190 --port 10110

  # 4. Interactive manual testing:
  python test_gnss_stop_detection.py --mode manual

  # 5. Read physical GPS receiver via COM port:
  python test_gnss_stop_detection.py --mode serial --port COM3 --baud 9600
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["sim", "tcp", "serial", "manual", "file"],
        default="tcp",
        help="Execution mode (default: 'tcp')",
    )
    parser.add_argument(
        "--imu",
        choices=["none", "synthetic"],
        default="none",
        help="IMU mode: 'none' (rely solely on GNSS) or 'synthetic' (simulate IMU dynamics) (default: none)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_NAVCAST_HOST,
        help=f"NavCast TCP host IP (default: {DEFAULT_NAVCAST_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_NAVCAST_PORT,
        help=f"NavCast TCP port (default: {DEFAULT_NAVCAST_PORT})",
    )
    parser.add_argument(
        "--serial-port",
        default="COM3",
        help="Serial/COM port name for GPS receiver (default: COM3)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Serial baud rate (default: 9600)",
    )
    parser.add_argument(
        "--file",
        help="Path to NMEA file to replay (for --mode file)",
    )
    parser.add_argument(
        "--rest-thresh",
        type=float,
        default=REST_SPEED_THRESHOLD_KMH,
        help=f"Speed threshold for rest candidate in km/h (default: {REST_SPEED_THRESHOLD_KMH})",
    )
    parser.add_argument(
        "--motion-thresh",
        type=float,
        default=STOP_SPEED_GATE,
        help=f"Speed threshold for motion resume in km/h (default: {STOP_SPEED_GATE})",
    )
    parser.add_argument(
        "--rest-debounce",
        type=float,
        default=REST_DEBOUNCE_SEC,
        help=f"Debounce seconds to confirm rest (default: {REST_DEBOUNCE_SEC})",
    )
    parser.add_argument(
        "--motion-debounce",
        type=float,
        default=MOTION_DEBOUNCE_SEC,
        help=f"Debounce seconds to confirm motion (default: {MOTION_DEBOUNCE_SEC})",
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        default=True,
        help="Auto-detect NavCast phone IP across Wi-Fi and tethering interfaces (default: enabled)",
    )
    parser.add_argument(
        "--no-auto-detect",
        dest="auto_detect",
        action="store_false",
        help="Disable auto-detection and connect strictly to specified --host",
    )
    parser.add_argument(
        "--speed-smoothing",
        type=int,
        default=SPEED_SMOOTHING_WINDOW,
        help=f"Moving average window for speed (default: {SPEED_SMOOTHING_WINDOW})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print raw NMEA sentences as they arrive (useful for diagnosing live feed issues)",
    )

    args = parser.parse_args()

    # Instantiate detector with requested configuration
    detector = VehicleStopDetector(
        stop_speed_gate=args.motion_thresh,
        rest_speed_threshold=args.rest_thresh,
        imu_accel_tolerance=IMU_REST_ACCEL_TOLERANCE,
        imu_gyro_tolerance=IMU_REST_GYRO_TOLERANCE,
        rest_debounce_sec=args.rest_debounce,
        motion_debounce_sec=args.motion_debounce,
        speed_smoothing_window=args.speed_smoothing,
    )

    imu_provider = SyntheticIMUProvider(mode=args.imu)

    # Route to requested mode
    if args.mode == "sim":
        run_simulation(detector, imu_provider)
    elif args.mode == "tcp":
        run_tcp_mode(
            detector,
            imu_provider,
            host=args.host,
            port=args.port,
            auto_detect=args.auto_detect,
            debug=args.debug,
        )
    elif args.mode == "serial":
        run_serial_mode(detector, imu_provider, port_name=args.serial_port, baudrate=args.baud)
    elif args.mode == "manual":
        run_manual_mode(detector, imu_provider)
    elif args.mode == "file":
        if not args.file:
            print("[ERROR] Please provide --file <path_to_nmea_file> when using --mode file.")
            sys.exit(1)
        run_file_mode(detector, imu_provider, filepath=args.file)


if __name__ == "__main__":
    main()
