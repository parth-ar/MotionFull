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

# Session / device logs
SESSION_LOG_DIR = os.path.join(_HERE, "logs", "sessions")    # Detailed per-session JSON logs
DEVICE_LOG_FILE = os.path.join(_HERE, "logs", "device.log")  # Rolling terminal/device log

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
# Camera Configuration (Raspberry Pi Camera Module 3 / IMX708 Wide NoIR)
# ---------------------------------------------------------------------------
# IMX708 native sensor: 4608 × 2592 px  →  aspect ratio 16:9 (1.777:1)
# Operational resolution must be 16:9 to preserve the full 120° diagonal FOV.
#
# Resolution options (all 16:9, all achievable at 30 fps on Pi 4B):
#   High   : 1280 × 720   — sharper captures, heavier motion diff CPU
#   Default: 960  × 540   — good balance for Pi 4B (motion + YOLO 30 fps)
#   Light  : 640  × 360   — minimum for very slow hardware
#
# DO NOT use 4:3 resolutions (e.g. 640×480) — they crop the sensor width
# and silently discard the wide-angle edges, reducing effective FOV to ~90°.
CAMERA_SOURCE_DEFAULT    = "auto"      # "auto" scans IMX708 first, then USB
CAMERA_WIDTH             = 960         # 16:9 — preserves full 120° FOV
CAMERA_HEIGHT            = 540         # 16:9 — matches IMX708 native aspect
CAMERA_FPS               = 30.0
CAMERA_AUTOFOCUS         = True        # Continuous PDAF autofocus for IMX708
CAMERA_AF_MODE           = "continuous"# Options: "continuous", "auto", "manual"
CAMERA_PREFER_IMX708     = True        # Prioritise IMX708 / Picamera2 over standard V4L2
CAMERA_IMX708_HDR        = False       # Hardware HDR mode for IMX708 sensor
CAMERA_SWAP_RB           = True        # Convert Picamera2 [R,G,B] array to OpenCV native [B,G,R]
CAMERA_AWB_MODE          = "auto"      # Default White Balance: "auto" (standard dynamic ISP, matches rpicam-hello).
                                       # Options: "auto", "daylight", "cloudy", "indoor", "tungsten", "fluorescent"
CAMERA_TUNING_FILE       = None        # Optional path to libcamera tuning JSON file (None = native auto-detection like rpicam-hello)

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
# u-blox MAX-M10S GNSS Hardware Configuration (UART Serial)
# Pin connections (identical to NEO-6M):
#   TXD (GNSS) -> Raspberry Pi Physical 10 / BCM 15 (RXD0 / /dev/serial0)
#   RXD (GNSS) -> Raspberry Pi Physical 8  / BCM 14 (TXD0)
#   VCC        -> Raspberry Pi Physical 1  (3.3V) or Physical 2/4 (5V)
#   GND        -> Raspberry Pi Physical 6  (GND)
#   RST (Opt)  -> Raspberry Pi Physical 36 / BCM 16 (GNSS_RESET_PIN)
# ---------------------------------------------------------------------------
GNSS_SERIAL_PORT    = "/dev/serial0"   # Default hardware UART on Raspberry Pi
GNSS_BAUD_RATE      = 9600             # Standard NMEA default baud rate for MAX-M10S / NEO-6M
GNSS_AUTO_DETECT    = True             # Auto-scan candidate serial ports if default fails

# ---------------------------------------------------------------------------
# IMU complementary filter
# ---------------------------------------------------------------------------
COMPLEMENTARY_ALPHA = 0.98

# ---------------------------------------------------------------------------
# LED BCM GPIO pin assignments (Pi 40-pin header)
# Updated per hardware revision — Raspberry Pi 4B physical pin → BCM mapping:
#   Physical 11 → BCM 17  RTC status LED   (green)
#   Physical 13 → BCM 27  IMU status LED   (green)
#   Physical 15 → BCM 22  GNSS status LED  (green)
#   Physical 16 → BCM 23  Red LED          (error indicator only — blinks on fault, OFF when healthy)
#   Physical 18 → BCM 24  Yellow LED       (system ready & snap — solid ON, triple blink on capture)
#   Physical 19 → BCM 10  LTE status LED   (green)
#   Physical 36 → BCM 16  GNSS module RST  (MAX-M10S active-low reset)
# ---------------------------------------------------------------------------
LED_RTC_GREEN  = 17   # BCM 17 — Physical 11 — RTC status  (green)
LED_IMU_GREEN  = 27   # BCM 27 — Physical 13 — IMU status  (green)
LED_GNSS_GREEN = 22   # BCM 22 — Physical 15 — GNSS status (green)
LED_RED        = 23   # BCM 23 — Physical 16 — error indicator only (red, blinks on fault, OFF when healthy)
LED_YELLOW     = 24   # BCM 24 — Physical 18 — system ready / snap (yellow, solid ON, triple blink on capture)
LED_LTE_GREEN  = 10   # BCM 10 — Physical 19 — LTE/modem status (green)

GNSS_RESET_PIN = 16   # BCM 16 — Physical 36 — MAX-M10S RST (active-low, drive LOW to reset)

LED_FAULT_BLINK_INTERVAL  = 0.400    # seconds
LED_YELLOW_BLINK_INTERVAL = 1.000    # seconds
REQUIRE_LTE_FOR_HEALTH    = False    # If True, LTE disconnection also triggers Red error blink

# ---------------------------------------------------------------------------
# GNSS fallback / IP-geolocation (Completely disabled — MAX-M10S hardware GNSS only)
# ---------------------------------------------------------------------------
ENABLE_GPS_FALLBACK        = False
GNSS_FALLBACK_TIMEOUT_SEC  = 20
GNSS_FALLBACK_REFRESH_SEC  = 300

# ---------------------------------------------------------------------------
# Motion detection parameters (Robust against shadows, lighting & reflections)
# ---------------------------------------------------------------------------
DIFF_THRESHOLD             = 30     # Pixel difference threshold (higher = filters soft shadows/ambient drift)
MIN_CONTOUR_AREA           = 700    # Min contour pixel area at downsampled FRAME_SIZE (filters small reflection glints)
MAX_MOTION_AREA_RATIO      = 0.55   # If motion covers > 55% of ROI, treat as global lighting shock (headlights/sun/auto-exposure)
BG_ALPHA                   = 0.05   # Background model learning rate under normal steady state
FAST_BG_ALPHA              = 0.25   # Accelerated adaptation rate when global lighting shock is detected
MOTION_CONSECUTIVE_FRAMES  = 2      # Consecutive frames of confirmed motion required before capture (rejects 1-frame flashes)
WARMUP_FRAMES              = 30
SAVE_COOLDOWN_SEC          = 10.0    # 10-second buffer between motion captures
FRAME_SIZE                 = (320, 180) # Downsampled resolution for motion processing

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
REST_CONFIRM_SEC           = 1.0    # Alias: rest confirmation threshold in seconds
MOTION_DEBOUNCE_SEC        = 0.6    # seconds of sustained > 5 km/h to confirm motion

# ---------------------------------------------------------------------------
# Litter Detection Parameters
# ---------------------------------------------------------------------------
# Distance interval — trigger litter inference every N meters of GNSS travel
LITTER_DISTANCE_INTERVAL_M   = 5.0

# Grace period (seconds) on GNSS loss before falling back to clock-based trigger
# Kept short so the clock fallback engages quickly and no litter frames are missed
LITTER_GNSS_LOST_TIMEOUT_SEC = 5.0

# Clock fallback interval (seconds) when no GNSS fix available
LITTER_TIME_FALLBACK_SEC     = 30.0

# YOLO inference confidence threshold
LITTER_CONF_THRESHOLD        = 0.60

# Fraction of detection bounding box that must overlap the AoD polygon to be ignored
LITTER_OVERLAP_THRESHOLD     = 0.50

# ---------------------------------------------------------------------------
# ONNX Weights Resolution & Active-Learning Model Replacement
# ---------------------------------------------------------------------------
# Note: The live edge node exclusively executes .onnx models (best_int8.onnx, best.onnx)
# for fast, lightweight inference on Raspberry Pi 4B CPU (via ONNX Runtime / OpenCV DNN).
# Base PyTorch weights (best.pt) are maintained in weights/ for the active-learning
# retraining pipeline (swstp_edge/active_learning/).
#
# REPLACEMENT POLICY:
# Newly trained models produced by retrain.py / export_onnx.py are exported to ONNX
# and configured to REPLACE the old production ONNX files in this weights directory.
# The edge detector automatically resolves and loads the newly replaced ONNX model.
WEIGHTS_DIR = os.path.join(_HERE, "weights")
BASE_PT_WEIGHTS = os.path.join(WEIGHTS_DIR, "best.pt")
PROD_ONNX_PATH = os.path.join(WEIGHTS_DIR, "best.onnx")
PROD_INT8_ONNX_PATH = os.path.join(WEIGHTS_DIR, "best_int8.onnx")

_WEIGHT_CANDIDATES = [
    PROD_INT8_ONNX_PATH,
    PROD_ONNX_PATH,
]
LITTER_WEIGHTS = next((p for p in _WEIGHT_CANDIDATES if os.path.isfile(p)), _WEIGHT_CANDIDATES[-1])

# ---------------------------------------------------------------------------
# Power Management & Backup Battery Settings
# ---------------------------------------------------------------------------
POWER_MANAGEMENT_ENABLED = True

# GPIO pin monitored for Low Battery Alert from UPS / secondary battery BMS.
# Set to None if NO UPS HAT / BMS hardware is physically wired to this pin.
# WARNING: Leaving a pin number here without wired hardware causes the floating
# pin to randomly read LOW (especially with NativeFactory GPIO backend), which
# triggers a false EMERGENCY SHUTDOWN after LOW_BATT_DEBOUNCE_SEC seconds.
# Re-enable by setting: GPIO_LOW_BATT_PIN = 25  (BCM 25 = physical Pin 22)
GPIO_LOW_BATT_PIN = None   # Set to 25 when UPS/BMS low-battery wire is connected

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
