"""
stop_detector.py — Vehicle stop detection via GNSS + IMU complementary fusion.

This module owns:
  1. StopCaptureItem — metadata for a motion frame captured during an active stop.
  2. StopEventRecord — full record of a vehicle stop session.
  3. VehicleStopDetector — MOVING / STOPPED state machine:

     Fusion strategy
     ---------------
     GNSS speed is the authoritative gate:
       - speed < 5.0 km/h  →  stop candidate
       - speed > 5.0 km/h  →  motion candidate

     IMU momentum dynamics modulate confidence via adaptive debounce:
       - GNSS + IMU both agree  →  short debounce (fast detection)
       - GNSS says stop, IMU still active (e.g. engine vibration)
                                →  extended debounce (tolerant, still detects)
       - IMU unavailable        →  normal debounce (GNSS-only)
       - GNSS unavailable       →  IMU-only mode (tunnels / GNSS loss)

     House collection status is NOT updated here — that gate lives in
     uploader.py and fires only after a confirmed backend upload (HTTP 200/201).
     A stop near a house with no frames captured does NOT change house status.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import time
from typing import Any, Dict, List, Optional

try:
    from config import (
        STOP_SPEED_GATE,
        REST_SPEED_THRESHOLD_KMH,
        IMU_REST_ACCEL_TOLERANCE,
        IMU_REST_GYRO_TOLERANCE,
        REST_DEBOUNCE_SEC,
        MOTION_DEBOUNCE_SEC,
    )
except ImportError:
    STOP_SPEED_GATE          = 5.0
    REST_SPEED_THRESHOLD_KMH = 3.0
    IMU_REST_ACCEL_TOLERANCE = 0.45
    IMU_REST_GYRO_TOLERANCE  = 4.0
    REST_DEBOUNCE_SEC        = 1.0
    MOTION_DEBOUNCE_SEC      = 0.6


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
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
    trigger_imu_summary: str = ""

    def get_duration_formatted(self) -> str:
        """Returns mm:ss or hh:mm:ss formatted string of duration."""
        sec = int(self.duration_sec)
        mins, s = divmod(sec, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs:02d}h {mins:02d}m {s:02d}s ({self.duration_sec:.1f}s)"
        return f"{mins:02d}m {s:02d}s ({self.duration_sec:.1f}s)"

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
            f"  Start Coordinates:      ({self.start_lat:.8f}, {self.start_lon:.8f})" if (self.start_lat is not None and self.start_lon is not None) else "  Start Coordinates:      N/A",
            f"  End Coordinates:        ({self.end_lat:.8f}, {self.end_lon:.8f})" if (self.end_lat is not None and self.end_lon is not None) else "  End Coordinates:        N/A",
            f"  Conclusion Trigger:     {self.end_reason}",
            f"  Motion Frames Captured: {len(self.captures)} frame(s)",
        ]

        if self.captures:
            lines.append("  --------------------------------------------------------------------")
            lines.append("  Previous Motion Frames Captured During This Stop:")
            for item in self.captures:
                coords = f"({item.latitude:.6f}, {item.longitude:.6f})" if item.latitude is not None else "(N/A)"
                lines.append(
                    f"    [{item.stop_frame_idx:02d}] {item.filename} | "
                    f"Offset: +{item.stop_offset_sec:.1f}s | RTC: {item.rtc_timestamp} | Coords: {coords}"
                )
        else:
            lines.append("  No motion frames triggered camera capture during this stop.")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vehicle Stop & Motion Detector State Machine
# ---------------------------------------------------------------------------
class VehicleStopDetector:
    """
    MOVING / STOPPED state machine using GNSS + IMU complementary fusion.

    GNSS speed is the required gate:
      < 5.0 km/h sustained → STOPPED (timer starts, house status unchanged)
      > 5.0 km/h sustained → MOVING  (stop concludes, summary printed)

    IMU momentum dynamics modulate debounce:
      Both agree    → 0.6× rest debounce  / 0.5× motion debounce  (fast)
      GNSS ok, IMU active (vibration) → 1.5× rest / 1.2× motion (tolerant)
      IMU unavailable → standard debounce (GNSS-only)
      GNSS unavailable → IMU-only mode
    """

    def __init__(
        self,
        stop_speed_gate: float = STOP_SPEED_GATE,
        rest_speed_threshold: float = REST_SPEED_THRESHOLD_KMH,
        imu_accel_tolerance: float = IMU_REST_ACCEL_TOLERANCE,
        imu_gyro_tolerance: float = IMU_REST_GYRO_TOLERANCE,
        rest_debounce_sec: float = REST_DEBOUNCE_SEC,
        motion_debounce_sec: float = MOTION_DEBOUNCE_SEC,
    ):
        self.stop_speed_gate = stop_speed_gate
        self.rest_speed_threshold = rest_speed_threshold
        self.imu_accel_tolerance = imu_accel_tolerance
        self.imu_gyro_tolerance = imu_gyro_tolerance
        self.rest_debounce_sec = rest_debounce_sec
        self.motion_debounce_sec = motion_debounce_sec

        self.state: str = "MOVING"  # "MOVING" | "STOPPED"
        self._current_stop_id: int = 0
        self.active_stop: Optional[StopEventRecord] = None
        self.history: List[StopEventRecord] = []

        # Debouncing state
        self._rest_candidate_start: Optional[float] = None
        self._motion_candidate_start: Optional[float] = None
        self._last_completed_event: Optional[StopEventRecord] = None

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

    def evaluate_sensors(
        self,
        speed_kmh: Optional[float],
        imu_data: Optional[Dict[str, Any]],
        gps_valid: bool = True,
    ) -> tuple[bool, bool, Dict[str, Any]]:
        """
        Evaluate instantaneous sensor state using GNSS + IMU complementary fusion.

        GNSS speed is the required gate:
          speed < rest_speed_threshold (5.0 km/h) → rest candidate
          speed > stop_speed_gate      (5.0 km/h) → motion candidate

        IMU agreement flags (imu_rest_agreement, imu_motion_agreement) are
        returned in metrics and used by update() to compute adaptive debounce.

        Returns:
          (is_rest_candidate, is_motion_candidate, metrics_dict)
        """
        speed = float(speed_kmh or 0.0)

        # ── IMU momentum dynamics ──────────────────────────────────────────
        # Computed from raw sensor regardless of GNSS state.
        # dyn_accel = deviation of |accel| from gravitational rest (9.80665 m/s²).
        # gyro_mag  = total angular rate |ω| = sqrt(gx² + gy² + gz²) in °/s.
        imu_valid     = bool(imu_data and imu_data.get("valid"))
        dyn_accel     = 0.0
        gyro_mag      = 0.0
        imu_at_rest   = False
        imu_in_motion = False

        if imu_valid:
            accel_mag = float(imu_data.get("accel_magnitude_ms2") or 9.80665)
            dyn_accel = abs(accel_mag - 9.80665)
            gyro      = imu_data.get("gyro_dps") or {}
            gx = float(gyro.get("x") or 0.0)
            gy = float(gyro.get("y") or 0.0)
            gz = float(gyro.get("z") or 0.0)
            gyro_mag      = math.sqrt(gx * gx + gy * gy + gz * gz)
            imu_at_rest   = (dyn_accel < self.imu_accel_tolerance) and \
                            (gyro_mag  < self.imu_gyro_tolerance)
            imu_in_motion = (dyn_accel >= self.imu_accel_tolerance * 1.2) or \
                            (gyro_mag  >= self.imu_gyro_tolerance  * 1.1)

        # ── Fusion decision ──────────────────────────────────────────────────
        # GNSS is the gate; IMU agreement/disagreement is surfaced in metrics
        # so update() can apply an adaptive debounce.
        if gps_valid:
            gnss_at_rest    = speed < self.rest_speed_threshold   # < 5.0
            gnss_in_motion  = speed > self.stop_speed_gate        # > 5.0
            is_rest_candidate   = gnss_at_rest
            is_motion_candidate = gnss_in_motion
            # imu_rest_agreement:   True  = IMU also at rest   (fast debounce)
            #                       False = IMU shows activity (tolerant debounce)
            #                       None  = IMU unavailable   (normal debounce)
            imu_rest_agreement   = imu_at_rest   if imu_valid else None
            imu_motion_agreement = imu_in_motion if imu_valid else None

        elif imu_valid:
            # GNSS unavailable — IMU-only mode (tunnel / GNSS outage)
            is_rest_candidate    = imu_at_rest
            is_motion_candidate  = imu_in_motion
            imu_rest_agreement   = imu_at_rest
            imu_motion_agreement = imu_in_motion
            gnss_at_rest = gnss_in_motion = False

        else:
            # Neither sensor valid (bench testing or sensor initialization phase)
            is_rest_candidate   = (speed < self.rest_speed_threshold)
            is_motion_candidate = (speed > self.stop_speed_gate)
            imu_rest_agreement = imu_motion_agreement = None
            gnss_at_rest = gnss_in_motion = False

        metrics = {
            "speed_kmh":            speed,
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
        imu_data: Optional[Dict[str, Any]],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        rtc_timestamp: Optional[str] = None,
        gps_valid: bool = True,
        now_mono: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Main update method called on every frame / sensor tick.
        Handles debounced state transitions and timer updates.
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

        if self.state == "MOVING":
            if is_rest_cand:
                if self._rest_candidate_start is None:
                    self._rest_candidate_start = now_mono

                # Adaptive debounce based on IMU momentum agreement:
                #   Both GNSS + IMU confirm rest  → short   (0.6× base = 0.6 s)
                #   GNSS stop, IMU still active   → tolerant (1.5× base = 1.5 s)
                #   IMU unavailable               → normal  (1.0× base = 1.0 s)
                imu_agree = metrics.get("imu_rest_agreement")
                if imu_agree is True:
                    eff_debounce = self.rest_debounce_sec * 0.6
                    agree_label  = "GNSS+IMU agree → fast confirm"
                elif imu_agree is False:
                    eff_debounce = self.rest_debounce_sec * 1.5
                    agree_label  = "GNSS stop, IMU active → extended confirm"
                else:
                    eff_debounce = self.rest_debounce_sec
                    agree_label  = "GNSS-only (IMU unavailable)"

                if (now_mono - self._rest_candidate_start) >= eff_debounce:
                    # Vehicle confirmed at rest → start stop event
                    self.state = "STOPPED"
                    self._rest_candidate_start  = None
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

                    print("=" * 70)
                    print(f"[VEHICLE STOP DETECTED] Stop #{self._current_stop_id} Started")
                    print("-" * 70)
                    print(f"  Timestamp (RTC):  {rtc_str}")
                    if latitude is not None and longitude is not None:
                        print(f"  Location:         ({latitude:.8f}, {longitude:.8f})")
                    print(f"  GNSS Speed:       {metrics['speed_kmh']:.1f} km/h (< {self.rest_speed_threshold:.1f} km/h)")
                    if metrics["imu_valid"]:
                        print(f"  IMU Dynamics:     dyn_accel: {metrics['dyn_accel']:.3f} m/s² | "
                              f"gyro_mag: {metrics['gyro_mag']:.2f} °/s | {'at rest' if metrics['imu_at_rest'] else 'active'}")
                    print(f"  Sensor Fusion:    {agree_label} (debounce: {eff_debounce:.1f}s)")
                    print(f"  Status:           Stop timer started. House status unchanged until camera frame confirmed.")
                    print("=" * 70)
            else:
                self._rest_candidate_start = None

        elif self.state == "STOPPED":
            # Check for vehicle resuming motion
            if is_motion_cand:
                if self._motion_candidate_start is None:
                    self._motion_candidate_start = now_mono

                # Adaptive debounce for motion detection:
                #   Both GNSS + IMU confirm motion → short   (0.5× = 0.3 s)
                #   GNSS motion, IMU still quiet   → tolerant (1.2× = 0.72 s)
                #   IMU unavailable               → normal  (1.0× = 0.6 s)
                imu_agree_m = metrics.get("imu_motion_agreement")
                if imu_agree_m is True:
                    eff_motion_debounce = self.motion_debounce_sec * 0.5
                    m_agree_label       = "GNSS+IMU agree → fast confirm"
                elif imu_agree_m is False:
                    eff_motion_debounce = self.motion_debounce_sec * 1.2
                    m_agree_label       = "GNSS motion, IMU quiet → extended confirm"
                else:
                    eff_motion_debounce = self.motion_debounce_sec
                    m_agree_label       = "GNSS-only (IMU unavailable)"

                if (now_mono - self._motion_candidate_start) >= eff_motion_debounce:
                    # Vehicle confirmed back in motion → end stop event
                    self.state = "MOVING"
                    self._motion_candidate_start = None
                    self._rest_candidate_start   = None

                    if self.active_stop is not None:
                        self.active_stop.end_mono         = now_mono
                        self.active_stop.end_rtc          = rtc_str
                        self.active_stop.end_lat          = latitude
                        self.active_stop.end_lon          = longitude
                        self.active_stop.duration_sec     = max(0.0, now_mono - self.active_stop.start_mono)
                        self.active_stop.trigger_speed_kmh = metrics["speed_kmh"]
                        self.active_stop.end_reason = (
                            f"Vehicle back in motion: Speed rose above {self.stop_speed_gate:.1f} km/h "
                            f"(current: {metrics['speed_kmh']:.1f} km/h) | {m_agree_label} | "
                            f"{'No frames captured — house status unchanged' if not self.active_stop.captures else f'{len(self.active_stop.captures)} frame(s) captured'}"
                        )

                        summary_report = self.active_stop.format_summary()
                        print("\n" + summary_report + "\n")

                        self._last_completed_event = self.active_stop
                        self.history.append(self.active_stop)
                        self.active_stop = None
                        event_just_ended = True
            else:
                self._motion_candidate_start = None

        # Use the single authoritative monotonic counter for the returned duration.
        # active_stop.duration_sec is only written at stop-end (for the summary report).
        current_duration = self.get_stop_duration(now_mono)
        current_capture_count = len(self.active_stop.captures) if self.active_stop else 0
        current_stop_id = self.active_stop.stop_id if self.active_stop else (self._current_stop_id if self.state == "STOPPED" else 0)

        return {
            "state": self.state,
            "is_stopped": self.is_stopped,
            "duration_sec": current_duration,
            "stop_event_id": current_stop_id,
            "capture_count": current_capture_count,
            "event_just_started": event_just_started,
            "event_just_ended": event_just_ended,
            "last_completed_event": self._last_completed_event,
            "active_stop": self.active_stop,
            "metrics": metrics,
        }

    def record_capture(self, meta: Dict[str, Any], now_mono: Optional[float] = None) -> Optional[StopCaptureItem]:
        """
        Registers a motion frame captured during an active stop event.
        """
        if self.state != "STOPPED" or self.active_stop is None:
            return None

        if now_mono is None:
            now_mono = time.monotonic()
        offset = max(0.0, now_mono - self.active_stop.start_mono)
        stop_frame_idx = len(self.active_stop.captures) + 1

        item = StopCaptureItem(
            capture_seq=meta.get("saved_count", stop_frame_idx),
            stop_frame_idx=stop_frame_idx,
            filename=meta.get("image_file", "unknown.jpg"),
            local_path=meta.get("local_path", ""),
            rtc_timestamp=meta.get("rtc_timestamp") or meta.get("captured_at") or "",
            stop_offset_sec=offset,
            latitude=meta.get("latitude"),
            longitude=meta.get("longitude"),
        )
        self.active_stop.captures.append(item)

        print(
            f"[STOP #{self.active_stop.stop_id} | CAPTURE #{stop_frame_idx}] "
            f"Logged motion frame '{item.filename}' @ +{offset:.1f}s into stop "
            f"(Total captures in this stop: {stop_frame_idx})"
        )
        return item

    def get_active_stop(self) -> Optional[StopEventRecord]:
        return self.active_stop

    def get_history(self) -> List[StopEventRecord]:
        return list(self.history)
