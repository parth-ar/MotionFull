#!/usr/bin/env python3
"""
test_gnss_stop_detection.py — Hardware-in-the-Loop & Diagnostic Test Harness
for Vehicle Stop Detection with u-blox MAX-M10S / NEO-6M GNSS and Arduino IMU.

Designed for standalone testing on a laptop with an Arduino Uno R3 running both
a u-blox MAX-M10S GNSS module and an MPU-6050 (or MPU-6500) 6-axis IMU over a
single USB COM port (or separate ports / Wi-Fi TCP streams / simulation).

==============================================================================
HARDWARE SETUP ON LAPTOP (Arduino Uno R3 + MAX-M10S + MPU-6050):
==============================================================================
  1. Arduino Uno R3 connects to Laptop via USB (e.g. COM3, COM4, /dev/ttyUSB0).
  2. MPU-6050 IMU connects to Arduino I2C:
       - SDA -> Pin A4 (or dedicated SDA pin)
       - SCL -> Pin A5 (or dedicated SCL pin)
       - VCC -> 5V / 3.3V, GND -> GND
  3. u-blox MAX-M10S GNSS connects to Arduino SoftwareSerial (or UART):
       - GNSS TXD -> Arduino Pin D8 (or D2 / RX)
       - GNSS RXD -> Arduino Pin D9 (or D3 / TX)
       - VCC      -> 3.3V or 5V, GND -> GND
  4. Arduino Uno sends data over USB to laptop in one of the following formats:
       Format A (Recommended — Multiplexed Stream):
         Arduino outputs raw NMEA sentences from MAX-M10S ($GNRMC, $GNGGA, $GNVTG)
         interleaved with IMU lines:
           AX:0.12,AY:-0.04,AZ:9.83,GX:0.21,GY:-0.13,GZ:0.05
           or: IMU: 0.12,-0.04,9.83,0.21,-0.13,0.05
           or: $PIMU,0.12,-0.04,9.83,0.21,-0.13,0.05*2A
       Format B (Combined Telemetry Line):
         DATA: 20.932000, 77.752300, 15.4, 1, 0.12, -0.04, 9.83, 0.21, -0.13, 0.05
         or JSON: {"lat": 20.932, "lon": 77.7523, "speed": 15.4, "fix": true, "ax": 0.12, ...}
       Format C (Pure GNSS NMEA):
         Arduino acts as transparent serial passthrough for MAX-M10S ($GN*).

==============================================================================
USAGE EXAMPLES:
==============================================================================
  # Auto-detect Arduino Uno R3 (GNSS + IMU) on laptop and run:
  python test_gnss_stop_detection.py --mode arduino

  # Explicit COM port and baud rate:
  python test_gnss_stop_detection.py --mode arduino --serial-port COM3 --baud 115200

  # Run automated simulation (no hardware needed):
  python test_gnss_stop_detection.py --mode sim

  # Connect to TCP NMEA stream (e.g. mobile phone GPS):
  python test_gnss_stop_detection.py --mode tcp --host 10.208.43.190 --port 10110
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import math
import os
import random
import re
import socket
import sys
import threading
import time
import json
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
# TUNABLE PARAMETERS — Aligned with swstp_edge/config.py
# ============================================================================
# Single symmetric speed threshold in km/h: < 5.0 km/h = candidate rest, > 5.0 km/h = motion
REST_SPEED_THRESHOLD_KMH: float = 5.0     # Speed below which vehicle is a rest candidate
STOP_SPEED_GATE: float          = 5.0     # Speed above which vehicle is in motion

# Debounce durations in seconds
REST_DEBOUNCE_SEC: float        = 1.0     # Sustained rest required to confirm STOPPED state (config.py: 1.0s)
MOTION_DEBOUNCE_SEC: float      = 0.6     # Sustained motion required to confirm MOVING state (config.py: 0.6s)

# IMU tolerances (used when IMU is connected or synthetic IMU is enabled)
IMU_REST_ACCEL_TOLERANCE: float = 0.45    # m/s² max dynamic acceleration deviation (|a| - 9.80665)
IMU_REST_GYRO_TOLERANCE: float  = 4.0     # deg/s max total angular rate sqrt(gx² + gy² + gz²)

# Advanced GNSS filtering & drift
SPEED_SMOOTHING_WINDOW: int     = 1       # Number of readings to average (1 = raw instantaneous speed)
REQUIRE_GPS_FIX: bool           = True    # Require active GPS fix for stop detection
MAX_STATIONARY_DRIFT_M: float   = 25.0    # Alert threshold for GPS drift radius while stopped

# Default TCP connection parameters
DEFAULT_TCP_HOST: str           = "127.0.0.1"
DEFAULT_TCP_PORT: int           = 10110


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
         - GNSS speed < REST_SPEED_THRESHOLD_KMH (default 5.0 km/h).
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
        self._rest_candidate_rtc: Optional[str] = None
        self._rest_candidate_lat: Optional[float] = None
        self._rest_candidate_lon: Optional[float] = None
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
        # GNSS speed is the authoritative gate — IMU agreement/disagreement
        # controls adaptive debounce in update(), never blocks detection outright.
        if effective_gps_valid:
            is_rest_candidate   = gnss_at_rest
            is_motion_candidate = gnss_in_motion
            imu_rest_agreement   = imu_at_rest   if imu_valid else None
            imu_motion_agreement = imu_in_motion if imu_valid else None
        elif imu_valid:
            # GNSS unavailable — IMU-only fallback (tunnel / GNSS outage)
            is_rest_candidate    = imu_at_rest
            is_motion_candidate  = imu_in_motion
            imu_rest_agreement   = imu_at_rest
            imu_motion_agreement = imu_in_motion
        else:
            # Neither sensor valid
            is_rest_candidate    = False
            is_motion_candidate  = False
            imu_rest_agreement   = None
            imu_motion_agreement = None

        metrics = {
            "raw_speed_kmh":        raw_speed,
            "eff_speed_kmh":        eff_speed,
            "gps_valid":            gps_valid,
            "imu_valid":            imu_valid,
            "dyn_accel":            round(dyn_accel, 4),
            "gyro_mag":             round(gyro_mag, 4),
            "gnss_at_rest":         gnss_at_rest,
            "gnss_in_motion":       gnss_in_motion,
            "imu_at_rest":          imu_at_rest,
            "imu_in_motion":        imu_in_motion,
            "imu_rest_agreement":   imu_rest_agreement,
            "imu_motion_agreement": imu_motion_agreement,
            "is_rest_candidate":    is_rest_candidate,
            "is_motion_candidate":  is_motion_candidate,
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
                    self._rest_candidate_rtc = rtc_str
                    self._rest_candidate_lat = latitude
                    self._rest_candidate_lon = longitude

                # Adaptive debounce based on IMU agreement
                effective_rest_debounce = self.rest_debounce_sec
                if metrics["imu_rest_agreement"] is True:
                    effective_rest_debounce *= 0.6   # Accelerated confirmation when IMU agrees
                elif metrics["imu_rest_agreement"] is False:
                    effective_rest_debounce *= 1.5   # Extra debounce if IMU indicates vibration/movement

                elapsed_cand = now_mono - self._rest_candidate_start
                if elapsed_cand >= effective_rest_debounce:
                    # Transition to STOPPED
                    self.state = "STOPPED"
                    self._current_stop_id += 1
                    event_just_started = True

                    self.active_stop = StopEventRecord(
                        stop_id=self._current_stop_id,
                        start_mono=self._rest_candidate_start,
                        start_rtc=self._rest_candidate_rtc or rtc_str,
                        start_lat=self._rest_candidate_lat if self._rest_candidate_lat is not None else latitude,
                        start_lon=self._rest_candidate_lon if self._rest_candidate_lon is not None else longitude,
                    )
                    self._rest_candidate_start = None
                    self._rest_candidate_rtc = None
                    self._rest_candidate_lat = None
                    self._rest_candidate_lon = None
                    self._motion_candidate_start = None

                    fusion_note = (
                        "Dual GNSS+IMU rest confirmed"
                        if metrics["imu_rest_agreement"] is True
                        else "GNSS-only (IMU unavailable)"
                        if not metrics["imu_valid"]
                        else "GNSS rest override"
                    )
                    dur_str = format_duration(self.get_stop_duration(now_mono))
                    imu_stat_str = ('At Rest (|a|=' + str(metrics['dyn_accel']) + ')') if metrics['imu_at_rest'] else 'Active/Vibrating' if metrics['imu_valid'] else 'Not Active (GNSS-only mode)'
                    print(
                        f"\n{'=' * 70}\n"
                        f"[VEHICLE REST DETECTED] Stop #{self.active_stop.stop_id} Started\n"
                        f"{'-' * 70}\n"
                        f"  Start Time (RTC): {self.active_stop.start_rtc} (confirmed at {rtc_str})\n"
                        f"  Start Location:  ({self.active_stop.start_lat or 0.0:.8f}, {self.active_stop.start_lon or 0.0:.8f})\n"
                        f"  GNSS Speed:      {metrics['raw_speed_kmh']:.1f} km/h (Rest Threshold: < {self.rest_speed_threshold:.1f} km/h)\n"
                        f"  IMU Status:      {imu_stat_str}\n"
                        f"  Sensor Fusion:   {fusion_note} (debounce: {self.rest_debounce_sec:.1f}s, confirmed in {elapsed_cand:.1f}s)\n"
                        f"  Stop Timer:      Running ({dur_str} elapsed since rest began). Monitoring for captures & motion resume...\n"
                        f"{'=' * 70}\n"
                    )
            else:
                self._rest_candidate_start = None
                self._rest_candidate_rtc = None
                self._rest_candidate_lat = None
                self._rest_candidate_lon = None

        elif self.state == "STOPPED":
            if self.active_stop is not None:
                # Track drift from initial stop location
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

                # Motion resume candidate evaluation
                if is_motion_cand:
                    if self._motion_candidate_start is None:
                        self._motion_candidate_start = now_mono

                    effective_motion_debounce = self.motion_debounce_sec
                    if metrics["imu_motion_agreement"] is True:
                        effective_motion_debounce *= 0.6  # Rapid exit when IMU confirms vehicle acceleration
                    elif metrics["imu_motion_agreement"] is False:
                        effective_motion_debounce *= 1.4  # Ignore transient GPS speed glitch if IMU is still stationary

                    elapsed_motion = now_mono - self._motion_candidate_start
                    if elapsed_motion >= effective_motion_debounce:
                        # Confirm motion resume
                        self.state = "MOVING"
                        event_just_ended = True

                        self.active_stop.end_mono = now_mono
                        self.active_stop.end_rtc = rtc_str
                        self.active_stop.end_lat = latitude
                        self.active_stop.end_lon = longitude
                        self.active_stop.duration_sec = max(
                            0.0, now_mono - self.active_stop.start_mono
                        )
                        self.active_stop.trigger_speed_kmh = metrics["raw_speed_kmh"]

                        fusion_reason = (
                            "GNSS speed + IMU motion confirmed"
                            if metrics["imu_motion_agreement"] is True
                            else "GNSS speed confirmed (IMU inactive)"
                            if not metrics["imu_valid"]
                            else "GNSS speed override"
                        )
                        self.active_stop.end_reason = (
                            f"Vehicle back in motion (Speed: {metrics['raw_speed_kmh']:.1f} km/h > {self.stop_speed_gate:.1f} km/h | {fusion_reason})"
                        )

                        self.history.append(self.active_stop)
                        self._last_completed_event = self.active_stop
                        print("\n" + self.active_stop.format_summary() + "\n")

                        self.active_stop = None
                        self._motion_candidate_start = None
                        self._rest_candidate_start = None
                else:
                    self._motion_candidate_start = None

        # Calculate progress for display
        rest_debounce_progress = 0.0
        if self._rest_candidate_start is not None and self.state == "MOVING":
            rest_debounce_progress = min(
                1.0, (now_mono - self._rest_candidate_start) / max(0.01, self.rest_debounce_sec)
            )

        motion_debounce_progress = 0.0
        if self._motion_candidate_start is not None and self.state == "STOPPED":
            motion_debounce_progress = min(
                1.0, (now_mono - self._motion_candidate_start) / max(0.01, self.motion_debounce_sec)
            )

        return {
            "state": self.state,
            "is_stopped": self.is_stopped,
            "duration_sec": self.get_stop_duration(now_mono),
            "duration_formatted": format_duration(self.get_stop_duration(now_mono)),
            "active_stop_id": self.active_stop.stop_id if self.active_stop else None,
            "event_just_started": event_just_started,
            "event_just_ended": event_just_ended,
            "drift_from_origin_m": round(drift_from_origin_m, 2),
            "rest_debounce_progress": rest_debounce_progress,
            "motion_debounce_progress": motion_debounce_progress,
            "capture_count": len(self.active_stop.captures) if self.active_stop else 0,
            "metrics": metrics,
        }

    def record_capture(
        self,
        frame_metadata: Dict[str, Any],
        now_mono: Optional[float] = None,
    ) -> Optional[StopCaptureItem]:
        """Logs a camera motion frame capture associated with the current stop."""
        if not self.is_stopped or self.active_stop is None:
            return None

        if now_mono is None:
            now_mono = time.monotonic()

        offset_sec = max(0.0, now_mono - self.active_stop.start_mono)
        frame_idx = len(self.active_stop.captures) + 1

        item = StopCaptureItem(
            capture_seq=int(frame_metadata.get("capture_seq", frame_idx)),
            stop_frame_idx=frame_idx,
            filename=str(frame_metadata.get("filename", f"frame_{frame_idx}.jpg")),
            local_path=str(frame_metadata.get("local_path", "")),
            rtc_timestamp=str(
                frame_metadata.get(
                    "rtc_timestamp",
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            ),
            stop_offset_sec=offset_sec,
            latitude=frame_metadata.get("latitude"),
            longitude=frame_metadata.get("longitude"),
        )
        self.active_stop.captures.append(item)
        print(
            f"[STOP #{self.active_stop.stop_id} | CAPTURE #{frame_idx}] "
            f"Logged frame '{item.filename}' @ +{offset_sec:.1f}s into stop "
            f"(Total captures in this stop: {len(self.active_stop.captures)})"
        )
        return item

    def get_history(self) -> List[StopEventRecord]:
        return list(self.history)


# ============================================================================
# Arduino IMU & Custom Telemetry Parsers
# ============================================================================
_LABELED_IMU_RE = re.compile(
    r"[Aa][cC]?[Xx]\s*[=:]\s*([\-\d\.]+).*?"
    r"[Aa][cC]?[Yy]\s*[=:]\s*([\-\d\.]+).*?"
    r"[Aa][cC]?[Zz]\s*[=:]\s*([\-\d\.]+).*?"
    r"[Gg][yY]?[Xx]\s*[=:]\s*([\-\d\.]+).*?"
    r"[Gg][yY]?[Yy]\s*[=:]\s*([\-\d\.]+).*?"
    r"[Gg][yY]?[Zz]\s*[=:]\s*([\-\d\.]+)"
)

def parse_imu_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parses a raw serial line containing IMU measurements.
    Supports labeled format (AX:..., AY:..., AZ:..., GX:..., GY:..., GZ:...),
    key-value format, bare 6-float CSV, and $PIMU sentences.
    Automatically auto-scales from 'g' to m/s² if magnitude ~ 1.0.
    """
    if not line:
        return None
    line = line.strip()

    # Proprietary NMEA IMU format ($PIMU,ax,ay,az,gx,gy,gz*hh)
    if line.startswith("$PIMU") or line.startswith("$IMU") or line.startswith("PIMU,") or line.startswith("IMU,"):
        parts = line.split("*")[0].split(",")
        if len(parts) >= 7:
            line = ",".join(parts[1:])
    elif line.upper().startswith("IMU:"):
        line = line[4:].strip()

    # Labeled format (AX:0.12, AY:-0.04, ...)
    m = _LABELED_IMU_RE.search(line)
    if m:
        try:
            ax, ay, az = float(m.group(1)), float(m.group(2)), float(m.group(3))
            gx, gy, gz = float(m.group(4)), float(m.group(5)), float(m.group(6))
            accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
            # Auto-scale if reported in g units (0.5g to 1.5g)
            if 0.5 <= accel_mag <= 1.5:
                ax *= 9.80665
                ay *= 9.80665
                az *= 9.80665
                accel_mag *= 9.80665
            return {
                "valid": True,
                "accel_magnitude_ms2": round(accel_mag, 4),
                "gyro_dps": {
                    "x": round(gx, 3),
                    "y": round(gy, 3),
                    "z": round(gz, 3),
                },
                "raw": {"ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4)},
                "source": "arduino",
            }
        except (ValueError, TypeError):
            pass

    # Bare CSV or whitespace separated (6 floats)
    clean = line.replace("	", ",").replace(";", ",")
    parts = [p.strip() for p in clean.split(",") if p.strip()] if "," in clean else line.split()
    if len(parts) == 6:
        try:
            vals = [float(p) for p in parts]
            ax, ay, az, gx, gy, gz = vals
            accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
            if 0.5 <= accel_mag <= 1.5:
                ax *= 9.80665
                ay *= 9.80665
                az *= 9.80665
                accel_mag *= 9.80665
            return {
                "valid": True,
                "accel_magnitude_ms2": round(accel_mag, 4),
                "gyro_dps": {
                    "x": round(gx, 3),
                    "y": round(gy, 3),
                    "z": round(gz, 3),
                },
                "raw": {"ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4)},
                "source": "arduino",
            }
        except (ValueError, TypeError):
            pass

    return None


def parse_custom_telemetry(line: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Parses custom Arduino telemetry formats (JSON, GNSS:, DATA:, COMBINED:).
    Returns (gnss_dict, imu_dict).
    """
    line = line.strip()
    if not line:
        return None, None

    # JSON telemetry format
    if line.startswith("{") and line.endswith("}"):
        try:
            d = json.loads(line)
            gnss_data = None
            imu_data = None
            if "lat" in d or "latitude" in d or "speed" in d:
                gnss_data = {
                    "latitude": float(d.get("lat") or d.get("latitude") or 0.0),
                    "longitude": float(d.get("lon") or d.get("longitude") or 0.0),
                    "speed_kmh": float(d.get("speed") or d.get("speed_kmh") or 0.0),
                    "fix": bool(d.get("fix", True)),
                    "satellites": int(d.get("satellites") or d.get("sats") or 0),
                    "hdop": float(d.get("hdop")) if d.get("hdop") is not None else None,
                }
            if "ax" in d and "ay" in d and "az" in d:
                ax, ay, az = float(d["ax"]), float(d["ay"]), float(d["az"])
                gx = float(d.get("gx", 0.0))
                gy = float(d.get("gy", 0.0))
                gz = float(d.get("gz", 0.0))
                mag = math.sqrt(ax * ax + ay * ay + az * az)
                if 0.5 <= mag <= 1.5:
                    ax *= 9.80665; ay *= 9.80665; az *= 9.80665; mag *= 9.80665
                imu_data = {
                    "valid": True,
                    "accel_magnitude_ms2": round(mag, 4),
                    "gyro_dps": {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
                    "raw": {"ax": ax, "ay": ay, "az": az},
                    "source": "arduino",
                }
            return gnss_data, imu_data
        except Exception:
            return None, None

    upper = line.upper()
    # GNSS / GPS custom line: GNSS: lat, lon, speed, [fix], [satellites]
    if upper.startswith("GNSS:") or upper.startswith("GPS:"):
        payload = line.split(":", 1)[1].strip()
        parts = [p.strip() for p in payload.split(",") if p.strip()]
        if len(parts) >= 3:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                spd = float(parts[2])
                fix = True
                sats = 0
                if len(parts) >= 4:
                    fix = parts[3].strip().lower() in ("1", "true", "fix", "a")
                if len(parts) >= 5:
                    sats = int(parts[4])
                return {"latitude": lat, "longitude": lon, "speed_kmh": spd, "fix": fix, "satellites": sats}, None
            except Exception:
                pass

    # COMBINED / DATA telemetry: DATA: lat, lon, speed, [fix], ax, ay, az, gx, gy, gz
    if upper.startswith("DATA:") or upper.startswith("COMBINED:") or upper.startswith("TELEMETRY:"):
        payload = line.split(":", 1)[1].strip()
        parts = [p.strip() for p in payload.split(",") if p.strip()]
        if len(parts) >= 9:
            try:
                lat, lon, spd = float(parts[0]), float(parts[1]), float(parts[2])
                fix = True
                offset = 0
                if len(parts) >= 10:
                    fix = parts[3].strip().lower() in ("1", "true", "fix", "a")
                    offset = 1
                ax, ay, az = float(parts[3 + offset]), float(parts[4 + offset]), float(parts[5 + offset])
                gx, gy, gz = float(parts[6 + offset]), float(parts[7 + offset]), float(parts[8 + offset])
                mag = math.sqrt(ax * ax + ay * ay + az * az)
                if 0.5 <= mag <= 1.5:
                    ax *= 9.80665; ay *= 9.80665; az *= 9.80665; mag *= 9.80665
                gnss = {"latitude": lat, "longitude": lon, "speed_kmh": spd, "fix": fix}
                imu = {
                    "valid": True,
                    "accel_magnitude_ms2": round(mag, 4),
                    "gyro_dps": {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
                    "raw": {"ax": ax, "ay": ay, "az": az},
                    "source": "arduino",
                }
                return gnss, imu
            except Exception:
                pass

    return None, None


class ArduinoIMUReader:
    """
    Dedicated background reader for Arduino Uno R3 IMU over a standalone serial port.
    Used when GNSS and IMU are on separate COM ports or in TCP mode.
    """
    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        debug: bool = False,
        verbose: bool = False,
    ):
        self.port = port
        self.baudrate = baudrate
        self.debug = debug
        self.verbose = verbose

        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ser = None

        self._lines_received: int = 0
        self._lines_parsed: int = 0
        self._lines_failed: int = 0
        self._parse_errors: List[str] = []
        self._last_raw: str = ""
        self._connect_error: str = ""
        self._connected: bool = False

    def start(self) -> bool:
        if not SERIAL_AVAILABLE:
            print("[IMU-ARDUINO] ERROR: pyserial not installed. Run: pip install pyserial")
            return False

        print(f"[IMU-ARDUINO] Opening {self.port} @ {self.baudrate} baud ...")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self._ser = serial.Serial()
                self._ser.port = self.port
                self._ser.baudrate = self.baudrate
                self._ser.timeout = 1.0
                self._ser.write_timeout = 1.0
                self._ser.dtr = False
                self._ser.rts = False
                self._ser.open()
                self._connected = True
                print(f"[IMU-ARDUINO] Connected to {self.port}. Waiting for data...")
                break
            except serial.SerialException as e:
                err_str = str(e)
                if "Access is denied" in err_str or "PermissionError" in err_str:
                    print(f"[IMU-ARDUINO] {self.port} is busy / access denied (attempt {attempt}/{max_attempts}).")
                    print(f"[IMU-ARDUINO] >>> Make sure to CLOSE the Arduino IDE Serial Monitor! <<<")
                    if attempt < max_attempts:
                        time.sleep(1.5)
                        continue
                self._connect_error = err_str
                print(f"[IMU-ARDUINO] FAILED to open {self.port}: {e}")
                return False
            except Exception as e:
                self._connect_error = str(e)
                print(f"[IMU-ARDUINO] FAILED to open {self.port}: {e}")
                return False

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="ArduinoIMUReader",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"[IMU-ARDUINO] Reader stopped. Stats — Lines rx:{self._lines_received} parsed:{self._lines_parsed}")

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def _reader_loop(self) -> None:
        while self._running:
            try:
                if not self._ser or not self._ser.is_open:
                    time.sleep(0.5)
                    continue

                raw = self._ser.readline()
                if not raw:
                    continue

                line = raw.decode("ascii", errors="replace").strip()
                self._lines_received += 1
                self._last_raw = line

                if self.debug:
                    print(f"[IMU-ARDUINO RAW] {line!r}")

                parsed = parse_imu_line(line)
                if parsed:
                    self._lines_parsed += 1
                    with self._lock:
                        self._latest = parsed
                    if self.verbose:
                        print(f"[IMU-ARDUINO PARSED] {parsed}")
                else:
                    if line and not line.startswith("//") and not line.startswith("#"):
                        self._lines_failed += 1

            except Exception as exc:
                exc_str = str(exc)
                if "ClearCommError" in exc_str or "PermissionError" in exc_str:
                    time.sleep(0.05)
                    continue
                time.sleep(0.2)


# ============================================================================
# Synthetic & Hardware IMU Provider
# ============================================================================
class SyntheticIMUProvider:
    """
    Unified IMU provider supporting:
      'none'      -- IMU disabled; GNSS is the sole sensor.
      'synthetic' -- Generates realistic synthetic readings correlated with speed.
      'arduino'   -- Reads live IMU data from an Arduino Uno R3 via serial port.
    """
    def __init__(self, mode: str = "none", arduino_reader: Optional[ArduinoIMUReader] = None):
        self.mode = mode.lower()
        self._arduino = arduino_reader

    def get_imu_reading(self, vehicle_speed_kmh: float) -> Optional[Dict[str, Any]]:
        if self.mode == "none":
            return None

        if self.mode == "arduino":
            if self._arduino is None:
                return None
            return self._arduino.get_latest()

        if self.mode == "synthetic":
            is_stopped = vehicle_speed_kmh < 1.0
            if is_stopped:
                dyn_noise = random.uniform(-0.08, 0.08)
                accel_mag = 9.80665 + dyn_noise
                gx = random.uniform(-0.4, 0.4)
                gy = random.uniform(-0.4, 0.4)
                gz = random.uniform(-0.3, 0.3)
            else:
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
# u-blox MAX-M10S Multi-Constellation NMEA Parser
# ============================================================================
class NMEAParser:
    """
    Lightweight, high-performance NMEA 0183 sentence parser supporting
    multi-constellation u-blox MAX-M10S and NEO-6M GNSS receivers.
    Supports $GN (Multi-GNSS), $GP (GPS), $GL (GLONASS), $GA (Galileo), $GB (BeiDou).
    """
    @staticmethod
    def verify_checksum(sentence: str) -> bool:
        if not sentence.startswith("$") and not sentence.startswith("!"):
            return False
        if "*" not in sentence:
            return True
        try:
            body, csum_hex = sentence.lstrip("$!").rsplit("*", 1)
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
        Parses a single NMEA sentence ($RMC, $GGA, $VTG, $GSA, $GLL).
        Returns a dict of extracted navigation fields.
        """
        result: Dict[str, Any] = {"valid_sentence": False}
        sentence = sentence.strip()
        if not sentence.startswith("$") and not sentence.startswith("!"):
            return result
        if not self.verify_checksum(sentence):
            return result

        result["valid_sentence"] = True
        body = sentence.lstrip("$!").split("*")[0]
        fields = body.split(",")
        if not fields:
            return result

        talker_type = fields[0].upper()
        result["talker"] = talker_type[:2] if len(talker_type) >= 2 else ""

        # ── RMC: Recommended Minimum Navigation Information ────────────────
        if talker_type.endswith("RMC"):
            result["type"] = "RMC"
            if len(fields) >= 9:
                status = fields[2].upper()
                result["status"] = status
                result["fix"] = (status == "A")
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

        # ── GSA: Active Satellites & Dilution of Precision ─────────────────
        elif talker_type.endswith("GSA"):
            result["type"] = "GSA"
            if len(fields) >= 18:
                try:
                    result["fix_mode"] = int(fields[2]) if fields[2] else 1
                    result["pdop"] = float(fields[15]) if fields[15] else None
                    result["hdop"] = float(fields[16]) if fields[16] else None
                    result["vdop"] = float(fields[17]) if fields[17] else None
                except Exception:
                    pass

        # ── GLL: Geographic Position (Lat/Lon/Fix) ──────────────────────────
        elif talker_type.endswith("GLL"):
            result["type"] = "GLL"
            if len(fields) >= 7:
                result["latitude"] = self.parse_coordinate(fields[1], fields[2])
                result["longitude"] = self.parse_coordinate(fields[3], fields[4])
                result["fix"] = (fields[6].upper() == "A")

        return result


# ============================================================================
# Smart Serial Port Auto-Detection
# ============================================================================
def auto_detect_serial_source(
    candidate_ports: Optional[List[str]] = None,
    candidate_bauds: Optional[List[int]] = None,
    probe_timeout_sec: float = 1.5,
) -> Tuple[Optional[str], Optional[int], str]:
    """
    Scans available COM / serial ports on laptop to automatically detect:
      - Arduino Uno R3 with MAX-M10S GNSS + MPU-6050 IMU ("multiplexed")
      - Direct MAX-M10S or GPS receiver ("gnss")
      - Standalone Arduino IMU ("imu")

    Returns:
      (port, baud, stream_type)
    """
    if not SERIAL_AVAILABLE:
        print("[SERIAL AUTO-DETECT] pyserial not available.")
        return None, None, "none"

    bauds = candidate_bauds or [115200, 9600, 57600, 38400]

    # Enumerate ports
    if candidate_ports:
        ports = candidate_ports
    else:
        try:
            detected = serial.tools.list_ports.comports()
            # Prioritize ports with Arduino / USB-SERIAL / Silicon Labs / FTDI
            ports = []
            for p in detected:
                desc = f"{p.device} {p.description or ''} {p.manufacturer or ''}".lower()
                if any(k in desc for k in ["arduino", "ch340", "cp210", "ftdi", "usb", "serial"]):
                    ports.insert(0, p.device)
                else:
                    ports.append(p.device)
        except Exception as e:
            print(f"[SERIAL AUTO-DETECT] Port enumeration error: {e}")
            return None, None, "none"

    if not ports:
        return None, None, "none"

    parser = NMEAParser()

    for port in ports:
        for baud in bauds:
            ser = None
            try:
                ser = serial.Serial()
                ser.port = port
                ser.baudrate = baud
                ser.timeout = 0.5
                ser.write_timeout = 0.5
                ser.dtr = False
                ser.rts = False
                ser.open()
                ser.reset_input_buffer()

                nmea_count = 0
                imu_count = 0
                custom_count = 0
                deadline = time.monotonic() + probe_timeout_sec

                while time.monotonic() < deadline:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("ascii", errors="replace").strip()
                    if not line:
                        continue

                    if line.startswith("$PIMU") or line.startswith("$IMU"):
                        imu_count += 1
                    elif line.startswith("$G") or line.startswith("$P") or line.startswith("!A"):
                        parsed_nmea = parser.parse_sentence(line)
                        if parsed_nmea.get("valid_sentence"):
                            nmea_count += 1
                    else:
                        parsed_imu = parse_imu_line(line)
                        if parsed_imu:
                            imu_count += 1
                        else:
                            g, im = parse_custom_telemetry(line)
                            if g or im:
                                custom_count += 1
                                if g: nmea_count += 1
                                if im: imu_count += 1

                if nmea_count > 0 and imu_count > 0:
                    return port, baud, "multiplexed"
                elif nmea_count > 0 or custom_count > 0:
                    return port, baud, "gnss"
                elif imu_count > 0:
                    return port, baud, "imu"

            except Exception:
                pass
            finally:
                if ser and ser.is_open:
                    try:
                        ser.close()
                    except Exception:
                        pass

    return None, None, "none"


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
    state_badge = f"[91m[{state:7s}][0m" if state == "STOPPED" else f"[92m[{state:7s}][0m"
    fix_display = f"[93m{fix_tag:4s}[0m" if fix_tag not in ("FIX", "3D FIX", "2D FIX") else f"[92m{fix_tag:4s}[0m"

    debounce_info = ""
    if state == "MOVING" and rest_prog > 0:
        bar = render_progress_bar(rest_prog, 1.0, 8)
        debounce_info = f" | Rest Debounce: {bar} {rest_prog * REST_DEBOUNCE_SEC:.1f}/{REST_DEBOUNCE_SEC:.1f}s"
    elif state == "STOPPED":
        bar = render_progress_bar(motion_prog, 1.0, 8)
        motion_str = f" | Motion Debounce: {bar} {motion_prog * MOTION_DEBOUNCE_SEC:.1f}/{MOTION_DEBOUNCE_SEC:.1f}s" if motion_prog > 0 else ""
        debounce_info = f" | Duration: {format_duration(duration)} | Drift: {drift_m:.1f}m | Captures: {captures}{motion_str}"

    print(
        f"[{timestamp_str}] {state_badge} GPS:{fix_display} Speed: {speed:5.1f} km/h | IMU: {imu_status:10s}{debounce_info}",
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
      6. Transient noise spike (brief 6.5 km/h spike ignored by motion debouncer).
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

    profile = [
        (3.0, 35.0, 35.0, "Cruising on main road"),
        (2.0, 35.0, 10.0, "Approaching intersection & braking"),
        (1.0, 10.0, 1.5,  "Coming to a halt (< 5.0 km/h rest candidate)"),
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

            jitter_lat = base_lat + random.uniform(-0.00001, 0.00001)
            jitter_lon = base_lon + random.uniform(-0.00001, 0.00001)

            sim_mono += tick_dt
            elapsed_total += tick_dt
            current_rtc = (sim_rtc_base + datetime.timedelta(seconds=elapsed_total)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            imu_reading = imu_provider.get_imu_reading(speed)

            if "Transient GPS speed glitch" in seg_desc:
                current_speed = 6.2 if step < 2 else 0.0
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

            time.sleep(0.05)

    print("\n" + "=" * 75)
    print("  SIMULATION COMPLETE")
    print(f"  Total Stop Events Recorded: {len(detector.get_history())}")
    print("=" * 75 + "\n")


# ============================================================================
# Mode 2: TCP NMEA Stream (Wi-Fi / Phone GPS)
# ============================================================================
def discover_tcp_nmea_server(preferred_host: str, preferred_port: int, scan_subnet: bool = True) -> Tuple[str, int]:
    """Helper to auto-detect TCP NMEA server."""
    return preferred_host, preferred_port


def run_tcp_mode(
    detector: VehicleStopDetector,
    imu_provider: SyntheticIMUProvider,
    host: str = DEFAULT_TCP_HOST,
    port: int = DEFAULT_TCP_PORT,
    auto_detect: bool = True,
    debug: bool = False,
) -> None:
    """Connects to live TCP NMEA stream."""
    parser = NMEAParser()
    active_host, active_port = host, port

    print(f"\n[TCP GNSS] Connecting to TCP NMEA stream at {active_host}:{active_port} ...")
    sock = None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((active_host, active_port))
        sock.settimeout(2.0)
        print(f"[TCP GNSS] Connected to {active_host}:{active_port}! Reading stream...\n")

        buffer = ""
        current_lat: Optional[float] = None
        current_lon: Optional[float] = None
        current_speed: float = 0.0
        current_fix: bool = False

        while True:
            try:
                data = sock.recv(1024).decode("ascii", errors="replace")
                if not data:
                    print("[TCP GNSS] Connection closed by server.")
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if debug:
                        print(f"[TCP RAW] {line}")

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

                    if parsed.get("type") in ("RMC", "GGA", "VTG", "GLL"):
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

                        fix_tag = "FIX" if current_fix else "VOID"
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
                            fix_tag=fix_tag,
                        )
            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("\n[TCP GNSS] Stopped by user.")
    except Exception as exc:
        print(f"[TCP GNSS ERROR] {exc}")
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


# ============================================================================
# Mode 3: Live Serial / Arduino Uno R3 Multiplexed Mode
# ============================================================================
def run_serial_mode(
    detector: VehicleStopDetector,
    imu_provider: SyntheticIMUProvider,
    port_name: str,
    baudrate: int,
    debug: bool = False,
    is_multiplexed: bool = True,
) -> None:
    """
    Reads live MAX-M10S NMEA sentences and/or interleaved IMU telemetry
    from an Arduino Uno R3 connected over a single COM/Serial port.
    """
    if not SERIAL_AVAILABLE:
        print("[ERROR] pyserial is required for serial mode. Install with: pip install pyserial")
        return

    # Auto-detect serial port if requested or if configured port doesn't exist
    if port_name in ("auto", None, "") or (not os.path.exists(port_name) and not port_name.upper().startswith("COM")):
        print("[SERIAL] Probing serial ports for Arduino Uno R3 / MAX-M10S GNSS ...")
        found_port, found_baud, found_type = auto_detect_serial_source()
        if found_port:
            port_name = found_port
            if baudrate == 9600 and found_baud:
                baudrate = found_baud
            print(f"[SERIAL] Auto-detected: {port_name} @ {baudrate} baud (Stream Type: {found_type.upper()})")
        else:
            # Fallback scan
            try:
                ports = [cp.device for cp in serial.tools.list_ports.comports()]
                if ports:
                    port_name = ports[0]
                    print(f"[SERIAL] Defaulting to available COM port: {port_name}")
                else:
                    print("[SERIAL ERROR] No serial/COM ports found on this laptop.")
                    return
            except Exception:
                port_name = "COM3"

    print("\n" + "=" * 70)
    print(f"  CONNECTING TO SERIAL DEVICE ON {port_name} @ {baudrate} BAUD")
    print(f"  Target: Arduino Uno R3 (u-blox MAX-M10S GNSS + MPU-6050 IMU)")
    print("=" * 70)
    print("Press Ctrl+C to stop.\n")

    parser = NMEAParser()
    ser = None

    try:
        ser = serial.Serial()
        ser.port = port_name
        ser.baudrate = baudrate
        ser.timeout = 0.5
        ser.write_timeout = 1.0
        # Disable DTR/RTS to prevent continuously rebooting Arduino Uno R3 on connect
        ser.dtr = False
        ser.rts = False
        ser.open()
        ser.reset_input_buffer()
        print(f"[SERIAL] Connected to {port_name}! Reading telemetry stream...\n")

        current_lat: Optional[float] = None
        current_lon: Optional[float] = None
        current_speed: float = 0.0
        current_fix: bool = False
        current_sats: int = 0
        current_hdop: Optional[float] = None
        current_imu: Optional[Dict[str, Any]] = None

        last_tick_time = time.monotonic()
        tick_interval = 0.2  # 5 Hz periodic detector tick for smooth UI updates

        while True:
            raw_line = ser.readline()
            now_mono = time.monotonic()
            updated_gnss = False

            if raw_line:
                line = raw_line.decode("ascii", errors="replace").strip()
                if line:
                    if debug:
                        print(f"[SERIAL RAW] {line}")

                    # 1. Proprietary NMEA IMU Sentence ($PIMU or $IMU)
                    if line.startswith("$PIMU") or line.startswith("$IMU"):
                        parsed_imu = parse_imu_line(line)
                        if parsed_imu:
                            current_imu = parsed_imu

                    # 2. Standard NMEA sentences from MAX-M10S ($GN, $GP, $GL, $GA, $GB)
                    elif line.startswith("$") or line.startswith("!"):
                        parsed = parser.parse_sentence(line)
                        if parsed.get("valid_sentence"):
                            if "latitude" in parsed and parsed["latitude"] is not None:
                                current_lat = parsed["latitude"]
                            if "longitude" in parsed and parsed["longitude"] is not None:
                                current_lon = parsed["longitude"]
                            if "fix" in parsed:
                                current_fix = parsed["fix"]
                            if "speed_kmh" in parsed:
                                current_speed = parsed["speed_kmh"]
                            if "satellites" in parsed:
                                current_sats = parsed["satellites"]
                            if "hdop" in parsed:
                                current_hdop = parsed["hdop"]

                            if parsed.get("type") in ("RMC", "GGA", "VTG", "GLL"):
                                updated_gnss = True

                    # 3. Non-NMEA lines: IMU telemetry or Custom Telemetry
                    else:
                        parsed_imu = parse_imu_line(line)
                        if parsed_imu:
                            current_imu = parsed_imu
                        else:
                            c_gnss, c_imu = parse_custom_telemetry(line)
                            if c_imu:
                                current_imu = c_imu
                            if c_gnss:
                                if c_gnss.get("latitude") is not None:
                                    current_lat = c_gnss["latitude"]
                                if c_gnss.get("longitude") is not None:
                                    current_lon = c_gnss["longitude"]
                                if "speed_kmh" in c_gnss:
                                    current_speed = c_gnss["speed_kmh"]
                                if "fix" in c_gnss:
                                    current_fix = c_gnss["fix"]
                                if "satellites" in c_gnss:
                                    current_sats = c_gnss["satellites"]
                                updated_gnss = True

            # Trigger detector update whenever GNSS fixes arrive or periodic timer fires
            if updated_gnss or (now_mono - last_tick_time >= tick_interval):
                last_tick_time = now_mono
                rtc_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Fused IMU: prefer live Arduino IMU reading from this COM port if available,
                # else fallback to imu_provider (which may be external or synthetic)
                fused_imu = current_imu if current_imu is not None else imu_provider.get_imu_reading(current_speed)

                status = detector.update(
                    vehicle_speed=current_speed,
                    imu_data=fused_imu,
                    latitude=current_lat,
                    longitude=current_lon,
                    rtc_timestamp=rtc_now,
                    gps_valid=current_fix,
                    now_mono=now_mono,
                )

                fix_tag = "3D FIX" if (current_fix and current_sats >= 4) else "FIX" if current_fix else "SEARCH"
                if fused_imu and fused_imu.get("valid"):
                    accel_str = f"|a|={fused_imu.get('accel_magnitude_ms2', 9.8):.2f}"
                    imu_tag = f"OK ({accel_str})"
                else:
                    imu_tag = "NONE"

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

    except KeyboardInterrupt:
        print("\n[SERIAL] Stopped by user.")
    except Exception as exc:
        print(f"[SERIAL ERROR] {exc}")
    finally:
        if ser and ser.is_open:
            try:
                ser.close()
            except Exception:
                pass


# ============================================================================
# Mode 4: Interactive Manual Control
# ============================================================================
def run_manual_mode(
    detector: VehicleStopDetector,
    imu_provider: SyntheticIMUProvider,
) -> None:
    """Interactive console runner where user can input speeds and test stop detection."""
    print("\n" + "=" * 70)
    print("  INTERACTIVE MANUAL STOP DETECTION TEST CONSOLE")
    print("=" * 70)
    print("Commands:")
    print("  <number>        : Set vehicle speed in km/h (e.g. '0' to stop, '25' to drive)")
    print("  cap             : Record a simulated motion frame capture during an active stop")
    print("  drift <meters>  : Inject stationary GPS coordinate drift")
    print("  status          : Print full current detector status")
    print("  q               : Quit console")
    print("=" * 70 + "\n")

    current_lat = 20.932000
    current_lon = 77.752300
    current_speed = 30.0
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
            time.sleep(0.5)

    tick_thread = threading.Thread(target=tick_loop, daemon=True)
    tick_thread.start()

    try:
        while True:
            cmd = input("stop-test> ").strip().lower()
            if not cmd:
                continue
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "status":
                dur = detector.get_stop_duration()
                print(
                    f"State: {detector.state} | Duration: {format_duration(dur)} | "
                    f"Speed: {current_speed:.1f} km/h | Rest Thresh: {detector.rest_speed_threshold:.1f} km/h | "
                    f"Motion Gate: {detector.stop_speed_gate:.1f} km/h"
                )
            elif cmd == "cap":
                if detector.is_stopped:
                    detector.record_capture({
                        "filename": f"manual_cap_{int(time.time())}.jpg",
                        "latitude": current_lat,
                        "longitude": current_lon,
                    })
                else:
                    print("Vehicle is MOVING — captures are logged only when STOPPED.")
            elif cmd.startswith("drift"):
                parts = cmd.split()
                dist = float(parts[1]) if len(parts) > 1 else 5.0
                current_lat += (dist / 111111.0)
                print(f"[DRIFT] Injected {dist:.1f}m coordinate drift.")
            else:
                try:
                    val = float(cmd.replace("speed", "").strip())
                    current_speed = max(0.0, val)
                    print(f"[SPEED SET] Vehicle speed changed to {current_speed:.1f} km/h.")
                except ValueError:
                    print("Unknown command. Enter a speed number, 'cap', 'drift', 'status', or 'q'.")
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

            if parsed.get("type") in ("RMC", "GGA", "VTG", "GLL"):
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

                fix_tag = "FIX" if current_fix else "VOID"
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
                    fix_tag=fix_tag,
                )

                time.sleep(0.1 / max(0.1, playback_speed))

    print("\n[FILE REPLAY] Replay complete.")


# ============================================================================
# Main CLI Entrypoint
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone GNSS & IMU Stop Detection Test Harness (u-blox MAX-M10S + Arduino Uno R3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Test on laptop with Arduino Uno R3 (GNSS + IMU connected):
  python test_gnss_stop_detection.py --mode arduino

  # 2. Specify exact COM port and baud rate:
  python test_gnss_stop_detection.py --mode arduino --serial-port COM3 --baud 115200

  # 3. Run automated simulation (no hardware needed):
  python test_gnss_stop_detection.py --mode sim

  # 4. Run simulation with synthetic IMU fusion enabled:
  python test_gnss_stop_detection.py --mode sim --imu synthetic

  # 5. Connect to live TCP NMEA stream (mobile phone GPS):
  python test_gnss_stop_detection.py --mode tcp --host 10.208.43.190 --port 10110

  # 6. Interactive manual testing:
  python test_gnss_stop_detection.py --mode manual
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["arduino", "serial", "sim", "tcp", "manual", "file"],
        default="arduino",
        help="Execution mode (default: 'arduino')",
    )
    parser.add_argument(
        "--imu",
        choices=["none", "synthetic", "arduino", "auto"],
        default="auto",
        help="IMU mode: 'auto' (detect Arduino IMU), 'arduino', 'synthetic', or 'none' (default: auto)",
    )
    parser.add_argument(
        "--serial-port", "--port",
        dest="serial_port",
        default="auto",
        help="Serial port for Arduino / GNSS receiver (e.g. COM3, COM4). Default: 'auto'",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate (Arduino default: 115200, MAX-M10S raw default: 9600)",
    )
    parser.add_argument(
        "--imu-port",
        default="auto",
        help="Separate serial port for IMU if not multiplexed on same Arduino COM port (default: auto)",
    )
    parser.add_argument(
        "--imu-baud",
        default="auto",
        help="Baud rate for separate IMU port (default: auto)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_TCP_HOST,
        help=f"TCP NMEA host IP (default: {DEFAULT_TCP_HOST})",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=DEFAULT_TCP_PORT,
        help=f"TCP NMEA port (default: {DEFAULT_TCP_PORT})",
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
        "--speed-smoothing",
        type=int,
        default=SPEED_SMOOTHING_WINDOW,
        help=f"Moving average window for speed (default: {SPEED_SMOOTHING_WINDOW})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print raw NMEA and serial telemetry lines as they arrive",
    )

    args = parser.parse_args()

    detector = VehicleStopDetector(
        stop_speed_gate=args.motion_thresh,
        rest_speed_threshold=args.rest_thresh,
        imu_accel_tolerance=IMU_REST_ACCEL_TOLERANCE,
        imu_gyro_tolerance=IMU_REST_GYRO_TOLERANCE,
        rest_debounce_sec=args.rest_debounce,
        motion_debounce_sec=args.motion_debounce,
        speed_smoothing_window=args.speed_smoothing,
    )

    # Separate IMU reader only needed if mode is TCP/sim with physical IMU,
    # or if user explicitly requested two different COM ports for GNSS and IMU
    arduino_reader: Optional[ArduinoIMUReader] = None
    use_separate_imu_port = (
        args.imu in ("arduino", "auto")
        and args.imu_port.strip().lower() != "auto"
        and args.imu_port.strip().upper() != args.serial_port.strip().upper()
    )

    if use_separate_imu_port and SERIAL_AVAILABLE:
        imu_baud_int = int(args.imu_baud) if str(args.imu_baud).isdigit() else 115200
        arduino_reader = ArduinoIMUReader(port=args.imu_port.strip(), baudrate=imu_baud_int, debug=args.debug)
        arduino_reader.start()

    imu_provider = SyntheticIMUProvider(mode=args.imu, arduino_reader=arduino_reader)

    # Dispatch to requested mode
    if args.mode in ("arduino", "serial"):
        run_serial_mode(
            detector,
            imu_provider,
            port_name=args.serial_port,
            baudrate=args.baud,
            debug=args.debug,
            is_multiplexed=True,
        )
    elif args.mode == "sim":
        run_simulation(detector, imu_provider)
    elif args.mode == "tcp":
        run_tcp_mode(
            detector,
            imu_provider,
            host=args.host,
            port=args.tcp_port,
            debug=args.debug,
        )
    elif args.mode == "manual":
        run_manual_mode(detector, imu_provider)
    elif args.mode == "file":
        if not args.file:
            print("[ERROR] Please provide --file <path_to_nmea_file> when using --mode file.")
            sys.exit(1)
        run_file_mode(detector, imu_provider, filepath=args.file)


if __name__ == "__main__":
    main()
