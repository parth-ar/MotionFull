"""
config.py — Centralised runtime configuration for the SWSTP Edge Pi node.

Device identity is read from device_config.json (replaces Arduino EEPROM).
All sensor / telemetry constants are defined here so they can be imported
by any module without circular dependencies.
"""

import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_CONFIG_FILE = os.path.join(_HERE, "device_config.json")
ROI_CONFIG_FILE    = os.path.join(_HERE, "roi_polygon.json")
CAPTURES_DIR       = os.path.join(_HERE, "captures")

# ---------------------------------------------------------------------------
# Device identity  (loaded from device_config.json)
# ---------------------------------------------------------------------------
_DEFAULT_DEVICE_ID = "UNPROVISIONED"

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

# ---------------------------------------------------------------------------
# Backend / session defaults (mirrors webcam_motion_detect.py globals)
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
TELEMETRY_RATE_HZ      = 20           # packets per second (matches .ino 20 Hz)
TELEMETRY_INTERVAL_SEC = 1.0 / TELEMETRY_RATE_HZ

# ---------------------------------------------------------------------------
# GNSS / timing thresholds (ported from .ino #defines, units converted)
# ---------------------------------------------------------------------------
GNSS_DATA_TIMEOUT_SEC   = 3.0         # was GNSS_DATA_TIMEOUT_MS / 1000
GNSS_DETAIL_INTERVAL_SEC = 1.0        # emit satellite detail once per second
GNSS_SATELLITE_SNAP_SEC  = 5.0        # drop satellite detail if snapshot > 5 s old

# ---------------------------------------------------------------------------
# IMU complementary filter
# ---------------------------------------------------------------------------
COMPLEMENTARY_ALPHA = 0.98            # matches .ino #define COMPLEMENTARY_ALPHA

# ---------------------------------------------------------------------------
# LED BCM GPIO pin assignments (Pi 40-pin header)
# Mirrors the Arduino pin assignments from the .ino where possible.
# Adjust to your physical wiring.
# ---------------------------------------------------------------------------
LED_RTC_GREEN  = 17   # BCM 17 — RTC status  (green)
LED_IMU_GREEN  = 27   # BCM 27 — IMU status  (green)
LED_GNSS_GREEN = 22   # BCM 22 — GNSS status (green)
LED_YELLOW     = 23   # BCM 23 — heartbeat   (yellow)
LED_RED        = 24   # BCM 24 — fault       (red)

LED_FAULT_BLINK_INTERVAL  = 0.400    # seconds — matches .ino 400 ms
LED_YELLOW_BLINK_INTERVAL = 1.000    # seconds — matches .ino 1000 ms

# ---------------------------------------------------------------------------
# GNSS fallback / IP-geolocation
# ---------------------------------------------------------------------------
ENABLE_GPS_FALLBACK        = True
GNSS_FALLBACK_TIMEOUT_SEC  = 20
GNSS_FALLBACK_REFRESH_SEC  = 300

# ---------------------------------------------------------------------------
# Motion detection parameters  (verbatim from webcam_motion_detect.py)
# ---------------------------------------------------------------------------
DIFF_THRESHOLD   = 22
MIN_CONTOUR_AREA = 350
BG_ALPHA         = 0.04
WARMUP_FRAMES    = 30
SAVE_COOLDOWN_SEC = 2.0
FRAME_SIZE       = (320, 180)

# ---------------------------------------------------------------------------
# Power Management & Backup Battery Settings
# ---------------------------------------------------------------------------
POWER_MANAGEMENT_ENABLED = True

# GPIO pin monitored for Low Battery Alert from UPS / secondary battery BMS (BCM numbering).
# Default: BCM 25 (Pin 22 on 40-pin header). Set to None if no hardware alert pin is wired.
GPIO_LOW_BATT_PIN = 25

# Low battery alert logic level: True if active LOW (0V on pin = low battery warning), False if active HIGH
LOW_BATT_ACTIVE_LOW = True

# Minimum duration the low battery signal must remain continuously active before triggering shutdown (debounce)
LOW_BATT_DEBOUNCE_SEC = 2.0

# Optional GPIO pin monitoring if main vehicle ignition / 12V supply is present.
# Set to None if not using an ignition detection pin.
GPIO_MAIN_POWER_PIN = None

# Grace period (seconds) to remain operating on secondary battery after vehicle ignition is turned off.
# If 0 or negative, runs on secondary battery until low-battery pin fires.
VEHICLE_OFF_SHUTDOWN_DELAY_SEC = 0

# Maximum time (seconds) to wait for pending motion evidence and telemetry batches to upload before forcing shutdown
SHUTDOWN_FLUSH_TIMEOUT_SEC = 30.0

