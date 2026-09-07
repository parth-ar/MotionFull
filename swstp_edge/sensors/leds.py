"""
sensors/leds.py — Status LED driver for Raspberry Pi using gpiozero.

LED Hardware Mapping (BCM GPIO pins — config.py):
  RTC_GREEN  (BCM 17): RTC module status:
                       - Solid ON if RTC working & valid
                       - OFF on RTC error / offline
  IMU_GREEN  (BCM 27): IMU module status:
                       - Solid ON if IMU working & valid
                       - OFF on IMU error / offline
  GNSS_GREEN (BCM 22): GNSS module status:
                       - Solid ON if GNSS fix is acquired
                       - Blinking (500 ms) if GNSS is looking for fix (connected, no fix yet)
                       - OFF if GNSS is disconnected / no data / module error
  YELLOW     (BCM 23): System ready & snap indicator:
                       - Solid ON when system is healthy and ready to click motion frames
                       - Blinks 3× when a motion frame is captured, then returns to solid ON
                       - Repeating 3× blinks when scanning for webcam / video source (algorithm on hold)
                       - OFF if ANY module fails or program error occurs
  RED        (BCM 24): Fault indicator:
                       - Blinking (400 ms) if ANY module fails or program error is encountered
                       - OFF when all modules and system are working normally

All LEDs are active-HIGH (logic 1 = LED on) with current-limiting resistors (220-470 Ω).
"""

import threading
import time

from config import (
    LED_RTC_GREEN, LED_IMU_GREEN, LED_GNSS_GREEN, LED_YELLOW, LED_RED,
    LED_FAULT_BLINK_INTERVAL,
)

# Blinking interval for GNSS searching for fix (500 ms = 1 Hz blink)
GNSS_SEARCH_BLINK_INTERVAL = 0.500

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

# System / Program health states
_system_ready: bool = False       # True once camera and main loop are running
_program_fault: bool = False      # True if an unhandled error or shutdown occurred
_fault_reason: str = ""

# Module health cache (updated at 20 Hz from telemetry loop)
_last_rtc_ok: bool = False
_last_imu_ok: bool = False
_last_gnss_fix: bool = False
_last_gnss_connected: bool = False

# Snap blink state (Yellow LED 3× blink on capture)
_snap_blinking: bool = False
_snap_lock = threading.Lock()

# Blink timers
_fault_blink_state: bool = False
_last_fault_toggle: float = 0.0

_gnss_blink_state: bool = False
_last_gnss_toggle: float = 0.0

# Camera scanning state (repeating triple blink on Yellow LED)
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
    """Background worker that continuously outputs repeating triple blinks
    on the status LED (Yellow / BCM 23) while scanning for a webcam.

    Pattern:
      3 rapid pulses:
        120 ms ON / 120 ms OFF x 3
      Pause:
        600 ms OFF
      Repeats continuously until _camera_scanning is disabled or stop_event is set.
    """
    global _camera_scanning
    while not _camera_scan_stop_event.is_set() and not _stop_event.is_set():
        with _camera_scan_lock:
            if not _camera_scanning or _program_fault:
                break

        # 3 rapid blinks: ON -> OFF -> ON -> OFF -> ON -> OFF
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

        # Pause interval between bursts (≈ 600 ms)
        for _ in range(6):
            if _camera_scan_stop_event.is_set() or _stop_event.is_set() or _program_fault:
                break
            with _camera_scan_lock:
                if not _camera_scanning:
                    break
            time.sleep(0.10)

    # Ensure Yellow LED is turned off when scanning ends if system is not yet ready
    if _leds and "yellow" in _leds and not _system_ready:
        _leds["yellow"].off()


def set_camera_scanning(enabled: bool = True) -> None:
    """
    Signal whether the system is actively scanning for a webcam / video source.
    When enabled, triggers repeating triple LED blinks on the Yellow LED
    to visually showcase that video ports are being scanned and the motion
    algorithm is on hold.
    """
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
    """
    Signal that the edge application and camera loop have initialized and are
    actively ready to capture motion frames.
    """
    global _system_ready, _program_fault, _fault_reason, _camera_scanning
    with _camera_scan_lock:
        _camera_scanning = False
        _camera_scan_stop_event.set()
    _system_ready = True
    _program_fault = False
    _fault_reason = ""
    print("[LEDS] System READY: Camera & motion engine running.")


def set_fault(reason: str = "") -> None:
    """
    Signal a program or system-level fault.
    Yellow LED turns OFF immediately. Red LED starts blinking.
    """
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


def notify_motion_snap() -> None:
    """
    Blink yellow LED 3 times (100 ms off / 100 ms on) when a motion frame
    is captured, then return to solid ON (if system is healthy).
    Non-blocking: runs in a background thread.
    """
    global _snap_blinking
    if not _leds:
        return

    def _snap_worker():
        global _snap_blinking
        with _snap_lock:
            _snap_blinking = True

        try:
            # 3 rapid blinks: OFF -> ON -> OFF -> ON -> OFF -> ON
            for _ in range(3):
                _leds["yellow"].off()
                time.sleep(0.10)
                _leds["yellow"].on()
                time.sleep(0.10)
        finally:
            with _snap_lock:
                _snap_blinking = False

            # Restore correct steady state
            has_error = (not _last_rtc_ok) or (not _last_imu_ok) or (not _last_gnss_connected) or _program_fault
            all_healthy = _system_ready and (not has_error)
            if all_healthy:
                _leds["yellow"].on()
            else:
                _leds["yellow"].off()

    t = threading.Thread(target=_snap_worker, name="led-snap-blink", daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# 20 Hz State Machine (called from telemetry.py)
# ---------------------------------------------------------------------------
def update(
    rtc: bool,
    imu: bool,
    gnss: bool = False,
    gnss_fix: bool | None = None,
    gnss_connected: bool | None = None,
    **kwargs,
) -> None:
    """
    Called at ~20 Hz from the telemetry loop with current sensor module states.

    Rules:
      1. RTC Green:
         - Solid ON if RTC is working, OFF if error
      2. IMU Green:
         - Solid ON if IMU is working, OFF if error
      3. GNSS Green:
         - Solid ON if GNSS has fix
         - Blinking (500 ms) if GNSS is connected and looking for fix
         - OFF if GNSS module error / disconnected / no data
      4. Red LED:
         - Blinking (400 ms) if ANY module fails (RTC/IMU/GNSS offline) OR program fault
         - OFF if all modules and program are healthy
      5. Yellow LED:
         - Solid ON if system is ready and all modules healthy
         - Blinks 3× on motion snap (managed by notify_motion_snap)
         - OFF if ANY module fails or program error occurs
    """
    global _last_rtc_ok, _last_imu_ok, _last_gnss_fix, _last_gnss_connected
    global _fault_blink_state, _last_fault_toggle
    global _gnss_blink_state, _last_gnss_toggle

    if not _leds:
        return

    # Normalize GNSS arguments
    has_fix = bool(gnss if gnss_fix is None else gnss_fix)
    if gnss_connected is None:
        if "gnss_data" in kwargs:
            is_connected = bool(kwargs["gnss_data"])
        elif "gnss_active" in kwargs:
            is_connected = bool(kwargs["gnss_active"])
        else:
            # Fallback: if has_fix is True, it is definitely connected; otherwise assume True if gnss passed
            is_connected = True if has_fix else bool(gnss)
    else:
        is_connected = bool(gnss_connected)

    _last_rtc_ok         = rtc
    _last_imu_ok         = imu
    _last_gnss_fix       = has_fix
    _last_gnss_connected = is_connected

    now = time.monotonic()

    # Toggle fault blink tick for Red LED (400 ms)
    if (now - _last_fault_toggle) > LED_FAULT_BLINK_INTERVAL:
        _fault_blink_state = not _fault_blink_state
        _last_fault_toggle = now

    # Toggle GNSS search blink tick for GNSS Green LED (500 ms)
    if (now - _last_gnss_toggle) > GNSS_SEARCH_BLINK_INTERVAL:
        _gnss_blink_state = not _gnss_blink_state
        _last_gnss_toggle = now

    # ── 1. RTC Green LED ───────────────────────────────────────────────
    _leds["rtc_green"].on() if rtc else _leds["rtc_green"].off()

    # ── 2. IMU Green LED ───────────────────────────────────────────────
    _leds["imu_green"].on() if imu else _leds["imu_green"].off()

    # ── 3. GNSS Green LED ──────────────────────────────────────────────
    if has_fix:
        _leds["gnss_green"].on()                  # Solid ON: Fix acquired
    elif is_connected:
        if _gnss_blink_state:                     # Blinking: Looking for fix
            _leds["gnss_green"].on()
        else:
            _leds["gnss_green"].off()
    else:
        _leds["gnss_green"].off()                 # OFF: Disconnected / error

    # ── 4. Health Evaluation ───────────────────────────────────────────
    # Module failure = any sensor completely disconnected or erroring
    module_failure = (not rtc) or (not imu) or (not is_connected)
    has_fault = module_failure or _program_fault

    # ── 5. Red LED (Blinks on ANY module failure or program fault) ─────
    if has_fault:
        if _fault_blink_state:
            _leds["red"].on()
        else:
            _leds["red"].off()
    else:
        _leds["red"].off()

    # ── 6. Yellow LED (Solid ON when ready & healthy, OFF on any error) ─
    if not _snap_blinking and not _camera_scanning:
        system_working_and_ready = _system_ready and (not has_fault)
        if system_working_and_ready:
            _leds["yellow"].on()
        else:
            _leds["yellow"].off()
