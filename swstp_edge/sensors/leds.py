"""
sensors/leds.py — Status LED driver for Raspberry Pi (SWSTP Unified).

LED Hardware Mapping (BCM GPIO pins — config.py):
  RTC_GREEN  (BCM 17): RTC module status:
                       - Solid ON if functional / working
                       - OFF on error / not detected
  IMU_GREEN  (BCM 27): IMU module status:
                       - Solid ON if functional / working
                       - OFF on error / not detected
  GNSS_GREEN (BCM 22): GNSS module status:
                       - Solid ON if GNSS fix acquired
                       - Blinking (500 ms) if looking for GPS fix
                       - OFF if disconnected / no data received
  LTE_GREEN  (BCM 10): LTE modem connectivity status:
                       - Solid ON if online / connected
                       - OFF if offline / disconnected
  RED        (BCM 23): Error indicator ONLY:
                       - Blinking (400 ms) if ANY module is not detected or not working as intended
                       - OFF when all conditions and circumstances are proper
                       - Strictly reserved for error indication (never used for litter detection)
  YELLOW     (BCM 24): System ready & image capture indicator:
                       - Solid ON when system is ready and no issues found
                       - Triple blink (3×: 100 ms OFF / 100 ms ON) whenever an image is captured
                       - Repeating 3× blinks while scanning for webcam
                       - OFF if ANY module fails or system has an issue

All LEDs are active-HIGH with current-limiting resistors (220-470 Ω).
"""

import threading
import time

from config import (
    LED_RTC_GREEN, LED_IMU_GREEN, LED_GNSS_GREEN, LED_YELLOW, LED_RED,
    LED_LTE_GREEN,
    LED_FAULT_BLINK_INTERVAL,
)

try:
    from config import REQUIRE_LTE_FOR_HEALTH
except ImportError:
    REQUIRE_LTE_FOR_HEALTH = False

GNSS_SEARCH_BLINK_INTERVAL = 0.500  # 1 Hz blink while searching for GNSS fix

# ---------------------------------------------------------------------------
# gpiozero import — gracefully degrade if not on a Pi
# ---------------------------------------------------------------------------
try:
    from gpiozero import LED as _GpioLED  # type: ignore
    _GPIO_AVAILABLE = True
except Exception:
    _GPIO_AVAILABLE = False
    print("[LEDS] gpiozero not available — LED control disabled (not running on Pi?)")


class _DummyLED:
    """No-op LED for non-Pi or testing environments."""
    def __init__(self, pin):
        self.pin = pin
    def on(self): pass
    def off(self): pass
    def close(self): pass


def _make_led(pin: int):
    if _GPIO_AVAILABLE:
        try:
            return _GpioLED(pin)
        except Exception as exc:
            print(f"[LEDS] Could not open GPIO {pin}: {exc}")
    return _DummyLED(pin)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_leds: dict = {}
_stop_event = threading.Event()

_system_ready: bool = False
_program_fault: bool = False
_fault_reason: str = ""

_last_rtc_ok: bool = False
_last_imu_ok: bool = False
_last_gnss_fix: bool = False
_last_gnss_connected: bool = False
_last_lte_ok: bool = False
_last_camera_ok: bool = True

# Snap blink state (Yellow triple blink on image capture)
_snap_blinking: bool = False
_snap_lock = threading.Lock()

# Fault blink timers (Red error indicator)
_fault_blink_state: bool = False
_last_fault_toggle: float = 0.0

# GNSS search blink timers (Green GNSS searching indicator)
_gnss_blink_state: bool = False
_last_gnss_toggle: float = 0.0

# Camera scanning state
_camera_scanning: bool = False
_camera_scan_thread: threading.Thread | None = None
_camera_scan_stop_event = threading.Event()
_camera_scan_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Init / Shutdown
# ---------------------------------------------------------------------------
def init() -> None:
    """Open GPIO handles and run the 3-flash boot splash (≈ 1.5 s)."""
    global _leds, _system_ready, _program_fault
    _system_ready = False
    _program_fault = False

    _leds = {
        "rtc_green":  _make_led(LED_RTC_GREEN),
        "imu_green":  _make_led(LED_IMU_GREEN),
        "gnss_green": _make_led(LED_GNSS_GREEN),
        "yellow":     _make_led(LED_YELLOW),
        "red":        _make_led(LED_RED),
        "lte_green":  _make_led(LED_LTE_GREEN),
    }
    _boot_splash()
    print("[LEDS] GPIO LED driver initialised.")


def close() -> None:
    """Release GPIO handles on shutdown."""
    global _system_ready, _camera_scanning
    _system_ready = False
    with _camera_scan_lock:
        _camera_scanning = False
        _camera_scan_stop_event.set()
    _stop_event.set()
    _all_off()
    for led in _leds.values():
        try:
            led.close()
        except Exception:
            pass


def _all_on() -> None:
    for led in _leds.values():
        led.on()


def _all_off() -> None:
    for led in _leds.values():
        led.off()


def _boot_splash() -> None:
    """3 × (200 ms ON + 250 ms OFF) to verify all LED hardware connections."""
    for _ in range(3):
        _all_on()
        time.sleep(0.200)
        _all_off()
        time.sleep(0.250)
    print("[LEDS] Boot splash complete.")


# ---------------------------------------------------------------------------
# Public System Control API
# ---------------------------------------------------------------------------
def set_headless_mode(enabled: bool) -> None:
    """Kept for backward compatibility."""
    pass


def _camera_scan_worker() -> None:
    """Background worker: repeating triple blinks on Yellow LED while scanning for webcam."""
    global _camera_scanning
    while not _camera_scan_stop_event.is_set() and not _stop_event.is_set():
        with _camera_scan_lock:
            if not _camera_scanning or _program_fault:
                break
        for _ in range(3):
            if _camera_scan_stop_event.is_set() or _stop_event.is_set() or _program_fault:
                break
            with _camera_scan_lock:
                if not _camera_scanning:
                    break
            if _leds and "yellow" in _leds:
                _leds["yellow"].on()
            time.sleep(0.12)
            if _leds and "yellow" in _leds:
                _leds["yellow"].off()
            time.sleep(0.12)
        for _ in range(6):
            if _camera_scan_stop_event.is_set() or _stop_event.is_set() or _program_fault:
                break
            with _camera_scan_lock:
                if not _camera_scanning:
                    break
            time.sleep(0.10)
    if _leds and "yellow" in _leds and not _system_ready:
        _leds["yellow"].off()


def set_camera_scanning(enabled: bool = True) -> None:
    """Signal whether the system is actively scanning for a webcam / video source."""
    global _camera_scanning, _camera_scan_thread, _system_ready
    with _camera_scan_lock:
        if enabled:
            _system_ready = False
            if not _camera_scanning:
                _camera_scanning = True
                _camera_scan_stop_event.clear()
                _camera_scan_thread = threading.Thread(
                    target=_camera_scan_worker,
                    name="led-cam-scan-blink",
                    daemon=True,
                )
                _camera_scan_thread.start()
                print("[LEDS] Camera SCANNING active — repeating triple LED blinks.")
        else:
            if _camera_scanning:
                _camera_scanning = False
                _camera_scan_stop_event.set()
                print("[LEDS] Camera scanning stopped.")


def set_system_ready() -> None:
    """Signal that the edge application and camera loop are ready."""
    global _system_ready, _program_fault, _fault_reason, _camera_scanning
    with _camera_scan_lock:
        _camera_scanning = False
        _camera_scan_stop_event.set()
    _system_ready = True
    _program_fault = False
    _fault_reason = ""
    print("[LEDS] System READY: Camera & motion engine running.")


def set_fault(reason: str = "") -> None:
    """Signal a program or system-level fault. Yellow OFF, Red blinks."""
    global _program_fault, _fault_reason, _camera_scanning
    _program_fault = True
    _fault_reason = reason
    with _camera_scan_lock:
        _camera_scanning = False
        _camera_scan_stop_event.set()
    if _leds and "yellow" in _leds:
        _leds["yellow"].off()
    if reason:
        print(f"[LEDS] System FAULT: {reason} — Yellow OFF, Red blinking.")
    else:
        print("[LEDS] System FAULT — Yellow OFF, Red blinking.")


def _trigger_snap_blink() -> None:
    """
    Blink Yellow LED 3× (100 ms OFF / 100 ms ON) whenever an image is captured.
    Non-blocking — runs in a daemon background thread.
    Pattern: 3 cycles of OFF -> ON -> restore steady state.
    """
    global _snap_blinking
    if not _leds or "yellow" not in _leds:
        return

    with _snap_lock:
        if _snap_blinking:
            return
        _snap_blinking = True

    def _worker():
        global _snap_blinking
        try:
            for _ in range(3):
                if _leds and "yellow" in _leds:
                    _leds["yellow"].off()
                time.sleep(0.10)
                if _leds and "yellow" in _leds:
                    _leds["yellow"].on()
                time.sleep(0.10)
        finally:
            with _snap_lock:
                _snap_blinking = False

            has_error = (
                (not _last_rtc_ok)
                or (not _last_imu_ok)
                or (not _last_gnss_connected)
                or (not _last_camera_ok)
                or _program_fault
            )
            if REQUIRE_LTE_FOR_HEALTH:
                has_error = has_error or (not _last_lte_ok)

            all_healthy = _system_ready and (not has_error)
            if _leds and "yellow" in _leds:
                if all_healthy:
                    _leds["yellow"].on()
                else:
                    _leds["yellow"].off()

    t = threading.Thread(target=_worker, name="led-snap-blink", daemon=True)
    t.start()


def notify_motion_snap() -> None:
    """
    Blink Yellow LED 3× (100 ms OFF / 100 ms ON) when a motion frame is captured.
    Non-blocking — runs in a daemon background thread.
    """
    _trigger_snap_blink()


def notify_litter_capture() -> None:
    """
    Blink Yellow LED 3× (100 ms OFF / 100 ms ON) when a frame is captured
    and queued for YOLO litter inference.
    Non-blocking — runs in a daemon background thread.
    """
    _trigger_snap_blink()


def notify_litter_snap() -> None:
    """
    Litter detection indicator:
    Red LED is strictly reserved for error indications only.
    Image capture triggers Yellow LED triple blink.
    """
    _trigger_snap_blink()


# ---------------------------------------------------------------------------
# 20 Hz State Machine (called from telemetry.py)
# ---------------------------------------------------------------------------
def update(
    rtc: bool,
    imu: bool,
    gnss: bool = False,
    gnss_fix: bool | None = None,
    gnss_connected: bool | None = None,
    lte: bool | None = None,
    camera: bool | None = None,
    require_lte: bool | None = None,
    **kwargs,
) -> None:
    """
    Called at ~20 Hz from the telemetry loop with current sensor module states.

    Operational Logic:
      1. Green Status LEDs (RTC, IMU, LTE):
         - Solid ON if functional / working
         - OFF if not detected or error
      2. GNSS Green LED:
         - Solid ON once GPS fix is acquired
         - Blinking (500 ms) while looking for GPS fix
         - OFF if disconnected / no serial data
      3. Red LED (Error indicator ONLY):
         - Blinking (400 ms) if ANY module is not detected or not working as intended
         - Stays OFF if all conditions and circumstances are proper
      4. Yellow LED:
         - Stays ON (solid) if no issues found and system is ready
         - Triple blink (3×) when an image is captured
         - Stays OFF if any issues found
    """
    global _last_rtc_ok, _last_imu_ok, _last_gnss_fix, _last_gnss_connected
    global _last_lte_ok, _last_camera_ok
    global _fault_blink_state, _last_fault_toggle
    global _gnss_blink_state, _last_gnss_toggle

    if not _leds:
        return

    # Evaluate GNSS states
    has_fix = bool(gnss if gnss_fix is None else gnss_fix)
    if gnss_connected is None:
        if "gnss_data" in kwargs:
            is_connected = bool(kwargs["gnss_data"])
        elif "gnss_active" in kwargs:
            is_connected = bool(kwargs["gnss_active"])
        else:
            is_connected = True if has_fix else bool(gnss)
    else:
        is_connected = bool(gnss_connected)

    # Evaluate LTE status
    if lte is not None:
        lte_ok = bool(lte)
    elif "lte_ok" in kwargs:
        lte_ok = bool(kwargs["lte_ok"])
    elif "backend" in kwargs:
        lte_ok = bool(kwargs["backend"])
    else:
        lte_ok = _last_lte_ok

    # Evaluate Camera status
    if camera is not None:
        cam_ok = bool(camera)
    elif "cam_ok" in kwargs:
        cam_ok = bool(kwargs["cam_ok"])
    else:
        cam_ok = _last_camera_ok

    # Update state records
    _last_rtc_ok         = bool(rtc)
    _last_imu_ok         = bool(imu)
    _last_gnss_fix       = bool(has_fix)
    _last_gnss_connected = bool(is_connected)
    _last_lte_ok         = bool(lte_ok)
    _last_camera_ok      = bool(cam_ok)

    now = time.monotonic()

    # Toggle fault blink state at configured interval (400 ms)
    if (now - _last_fault_toggle) > LED_FAULT_BLINK_INTERVAL:
        _fault_blink_state = not _fault_blink_state
        _last_fault_toggle = now

    # Toggle GNSS search blink state at 500 ms (1 Hz)
    if (now - _last_gnss_toggle) > GNSS_SEARCH_BLINK_INTERVAL:
        _gnss_blink_state = not _gnss_blink_state
        _last_gnss_toggle = now

    # 1. RTC Green (BCM 17): Solid ON if functional, OFF on error
    if "rtc_green" in _leds:
        _leds["rtc_green"].on() if rtc else _leds["rtc_green"].off()

    # 2. IMU Green (BCM 27): Solid ON if functional, OFF on error
    if "imu_green" in _leds:
        _leds["imu_green"].on() if imu else _leds["imu_green"].off()

    # 3. LTE Green (BCM 10): Solid ON if functional, OFF on error
    if "lte_green" in _leds:
        _leds["lte_green"].on() if lte_ok else _leds["lte_green"].off()

    # 4. GNSS Green (BCM 22):
    #    - Solid ON if GPS fix acquired
    #    - Blinking (500 ms) if looking for GPS fix
    #    - OFF if disconnected / no data received
    if "gnss_green" in _leds:
        if has_fix:
            _leds["gnss_green"].on()
        elif is_connected:
            _leds["gnss_green"].on() if _gnss_blink_state else _leds["gnss_green"].off()
        else:
            _leds["gnss_green"].off()

    # Determine whether any module is not detected or not working as intended
    module_failure = (not rtc) or (not imu) or (not is_connected) or (not cam_ok)
    if require_lte is True:
        module_failure = module_failure or (not lte_ok)
    elif require_lte is None and REQUIRE_LTE_FOR_HEALTH:
        module_failure = module_failure or (not lte_ok)

    has_fault = module_failure or _program_fault

    # Red LED (BCM 23): Error indicator ONLY
    # - Blinks if ANY module is not detected or not working as intended
    # - Stays OFF when all conditions and circumstances are proper
    if "red" in _leds:
        if has_fault:
            _leds["red"].on() if _fault_blink_state else _leds["red"].off()
        else:
            _leds["red"].off()

    # Yellow LED (BCM 24):
    # - Solid ON if no issues found and system is ready
    # - OFF if any module fails or program fault
    # - (Triple blink on image capture handled by _snap_blinking worker)
    if "yellow" in _leds and not _snap_blinking and not _camera_scanning:
        system_working_and_ready = _system_ready and (not has_fault)
        _leds["yellow"].on() if system_working_and_ready else _leds["yellow"].off()
