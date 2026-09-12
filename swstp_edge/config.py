"""
config.py — Centralised runtime configuration for SWSTP Unified Edge Node.

Merges swstp_edge (motion detection) and litter_event_logger (litter detection)
into a single config. Device identity is read from device_config.json.
All sensor, telemetry, motion, and litter constants are defined here.
"""

import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_CONFIG_FILE = os.path.join(_HERE, "device_config.json")

# Motion detection
ROI_CONFIG_FILE = os.path.join(_HERE, "roi_polygon.json")
CAPTURES_DIR    = os.path.join(_HERE, "captures", "motion")   # Motion captures sub-dir

# Litter detection
LITTER_CAPTURES_DIR = os.path.join(_HERE, "captures", "litter")  # Litter captures sub-dir

# ---------------------------------------------------------------------------
# Device identity  (loaded from device_config.json)
# ---------------------------------------------------------------------------
_DEFAULT_DEVICE_ID = "UNPROVISIONED"
DEFAULT_TIMEZONE   = "Asia/Kolkata"    # Indian Standard Time (IST, UTC+05:30)
RTC_RESYNC_INTERVAL_HOURS = 1.0       # Recalibrate timing every hour using the internet


def load_device_id() -> str:
    """Read device ID from device_config.json.  Falls back to UNPROVISIONED."""
    try:
        with open(DEVICE_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dev_id = str(cfg.get("device_id") or cfg.get("deviceId") or "").strip()
        return dev_id if dev_id else _DEFAULT_DEVICE_ID
    except FileNotFoundError:
        return _DEFAULT_DEVICE_ID
    except Exception as exc:
        print(f"[CONFIG] Warning reading {DEVICE_CONFIG_FILE}: {exc}")
        return _DEFAULT_DEVICE_ID


def load_timezone() -> str:
    """Read timezone string from device_config.json. Falls back to 'Asia/Kolkata'."""
    try:
        with open(DEVICE_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tz_name = str(cfg.get("timezone") or "").strip()
        return tz_name if tz_name else DEFAULT_TIMEZONE
    except Exception:
        return DEFAULT_TIMEZONE


def get_timezone_obj():
    """Return a datetime.tzinfo object for the configured timezone."""
    import datetime
    tz_name = load_timezone()
    try:
        import zoneinfo
        return zoneinfo.ZoneInfo(tz_name)
    except Exception:
        return datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")


# ---------------------------------------------------------------------------
# Backend / session defaults
# ---------------------------------------------------------------------------
DEFAULT_BACKEND_URL = os.environ.get(
    "SWSTP_BACKEND_URL", "https://solidwasteapi.scipl.info.in"
)
DEFAULT_ULB_ID    = "ULB_MH_AMRAVATI"
DEFAULT_VEHICLE_ID = "UNASSIGNED"
DEFAULT_STREAM_FPS = 30.0

# ---------------------------------------------------------------------------
# Telemetry rate
# ---------------------------------------------------------------------------
TELEMETRY_RATE_HZ      = 20           # packets per second
TELEMETRY_INTERVAL_SEC = 1.0 / TELEMETRY_RATE_HZ

# ---------------------------------------------------------------------------
# GNSS / timing thresholds
# ---------------------------------------------------------------------------
GNSS_DATA_TIMEOUT_SEC    = 3.0
GNSS_DETAIL_INTERVAL_SEC = 1.0
GNSS_SATELLITE_SNAP_SEC  = 5.0

# ---------------------------------------------------------------------------
# NavCast GNSS — phone app streaming NMEA over USB tethering TCP
# ---------------------------------------------------------------------------
NAVCAST_HOST        = "10.208.43.190"
NAVCAST_PORT        = 10110
NAVCAST_AUTO_DETECT = True

# ---------------------------------------------------------------------------
# IMU complementary filter
# ---------------------------------------------------------------------------
COMPLEMENTARY_ALPHA = 0.98

# ---------------------------------------------------------------------------
# LED BCM GPIO pin assignments (Pi 40-pin header)
# ---------------------------------------------------------------------------
LED_RTC_GREEN  = 17   # BCM 17 — RTC status  (green)
LED_IMU_GREEN  = 27   # BCM 27 — IMU status  (green)
LED_GNSS_GREEN = 22   # BCM 22 — GNSS status (green)
LED_YELLOW     = 23   # BCM 23 — heartbeat / snap (yellow)
LED_RED        = 24   # BCM 24 — fault / litter alert (red)

LED_FAULT_BLINK_INTERVAL  = 0.400    # seconds
LED_YELLOW_BLINK_INTERVAL = 1.000    # seconds

# ---------------------------------------------------------------------------
# GNSS fallback / IP-geolocation
# ---------------------------------------------------------------------------
ENABLE_GPS_FALLBACK        = True
GNSS_FALLBACK_TIMEOUT_SEC  = 20
GNSS_FALLBACK_REFRESH_SEC  = 300

# ---------------------------------------------------------------------------
# Motion detection parameters
# ---------------------------------------------------------------------------
DIFF_THRESHOLD    = 22
MIN_CONTOUR_AREA  = 350
BG_ALPHA          = 0.04
WARMUP_FRAMES     = 30
SAVE_COOLDOWN_SEC = 5.0        # 5-second buffer between motion captures
FRAME_SIZE        = (320, 180) # Downsampled resolution for motion processing

# ---------------------------------------------------------------------------
# Vehicle Stop Detection Parameters
# ---------------------------------------------------------------------------
# Single symmetric threshold: < 5 km/h = STOPPED, > 5 km/h = MOVING.
# Both STOP_SPEED_GATE and REST_SPEED_THRESHOLD_KMH are set to 5.0 so there
# is no hysteresis gap where the vehicle state is ambiguous.
STOP_SPEED_GATE            = 5.0    # km/h: speed > 5.0 km/h ends a stop event
REST_SPEED_THRESHOLD_KMH   = 5.0    # km/h: speed < 5.0 km/h starts a stop event
IMU_REST_ACCEL_TOLERANCE   = 0.45   # m/s² — logged in metrics, not a gate condition
IMU_REST_GYRO_TOLERANCE    = 4.0    # deg/s — logged in metrics, not a gate condition
REST_DEBOUNCE_SEC          = 1.0    # seconds of sustained < 5 km/h to confirm stop
MOTION_DEBOUNCE_SEC        = 0.6    # seconds of sustained > 5 km/h to confirm motion

# ---------------------------------------------------------------------------
# Litter Detection Parameters
# ---------------------------------------------------------------------------
# Distance interval — trigger litter inference every N meters of GNSS travel
LITTER_DISTANCE_INTERVAL_M   = 10.0

# Grace period (seconds) on GNSS loss before falling back to clock-based trigger
# Kept short so the clock fallback engages quickly and no litter frames are missed
LITTER_GNSS_LOST_TIMEOUT_SEC = 5.0

# Clock fallback interval (seconds) when no GNSS fix available
LITTER_TIME_FALLBACK_SEC     = 30.0

# YOLO inference confidence threshold
LITTER_CONF_THRESHOLD        = 0.25

# Fraction of detection bounding box that must overlap the AoD polygon to be ignored
LITTER_OVERLAP_THRESHOLD     = 0.50

# ONNX weights resolution — prefer int8 quantised for Pi 4B performance
_WEIGHT_CANDIDATES = [
    os.path.join(_HERE, "weights", "best_int8.onnx"),
    os.path.join(_HERE, "weights", "best.onnx"),
]
LITTER_WEIGHTS = next((p for p in _WEIGHT_CANDIDATES if os.path.isfile(p)), _WEIGHT_CANDIDATES[-1])

# ---------------------------------------------------------------------------
# Power Management & Backup Battery Settings
# ---------------------------------------------------------------------------
POWER_MANAGEMENT_ENABLED = True

# GPIO pin monitored for Low Battery Alert from UPS / secondary battery BMS
GPIO_LOW_BATT_PIN = 25   # BCM 25 (Pin 22 on 40-pin header). Set to None if not wired.

# Low battery alert logic level: True = active LOW (0 V = warning), False = active HIGH
LOW_BATT_ACTIVE_LOW = True

# Minimum duration (s) the low battery signal must stay active before triggering shutdown
LOW_BATT_DEBOUNCE_SEC = 2.0

# Optional GPIO pin for ignition / 12 V supply detection. None if unused.
GPIO_MAIN_POWER_PIN = None

# Grace period (s) to remain operating on secondary battery after ignition off.
VEHICLE_OFF_SHUTDOWN_DELAY_SEC = 0

# Maximum time (s) to wait for pending uploads before forcing shutdown
SHUTDOWN_FLUSH_TIMEOUT_SEC = 30.0
