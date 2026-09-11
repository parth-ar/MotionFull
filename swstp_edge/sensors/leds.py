"""
sensors/leds.py — Status LED driver for Raspberry Pi (SWSTP Unified).

LED Hardware Mapping (BCM GPIO pins — config.py):
  RTC_GREEN  (BCM 17): RTC module status — Solid ON if working, OFF on error.
  IMU_GREEN  (BCM 27): IMU module status — Solid ON if working, OFF on error.
  GNSS_GREEN (BCM 22): GNSS module status:
                       - Solid ON if GNSS fix acquired
                       - Blinking (500 ms) if looking for fix
                       - OFF if disconnected / no data
  YELLOW     (BCM 23): System ready & snap indicator:
                       - Solid ON when system is healthy and ready
                       - Blinks 3× on a motion frame capture
                       - Blinks 2× double-blink (Red-Yellow-Red-Yellow) NOT handled here;
                         the litter pattern blends Red+Yellow together — see notify_litter_snap()
                       - Repeating 3× blinks while scanning for webcam
                       - OFF if ANY module fails
  RED        (BCM 24): Fault & litter detection indicator:
                       - Blinking (400 ms) if ANY module fails or program fault
                       - Alternating Red-Yellow-Red-Yellow blink when litter is DETECTED
                       - OFF when all modules and system are healthy

All LEDs are active-HIGH with current-limiting resistors (220-470 Ω).
"""

import threading
import time

from config import (
    LED_RTC_GREEN, LED_IMU_GREEN, LED_GNSS_GREEN, LED_YELLOW, LED_RED,
    LED_FAULT_BLINK_INTERVAL,
)

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

# Snap blink state
_snap_blinking: bool = False
_snap_lock = threading.Lock()

# Litter detection blink state (Red-Yellow-Red-Yellow)
_litter_blinking: bool = False
_litter_lock = threading.Lock()

# Fault blink timers
_fault_blink_state: bool = False
_last_fault_toggle: float = 0.0

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


def notify_motion_snap() -> None:
    """
    Blink Yellow LED 3× (100 ms off / 100 ms on) when a motion frame is captured.
    Non-blocking — runs in a daemon background thread.
    Pattern: triple yellow blink → restore steady state.
    """
    global _snap_blinking
    if not _leds:
        return

    def _snap_worker():
        global _snap_blinking
        with _snap_lock:
            _snap_blinking = True
        try:
            for _ in range(3):
                _leds["yellow"].off()
                time.sleep(0.10)
                _leds["yellow"].on()
                time.sleep(0.10)
        finally:
            with _snap_lock:
                _snap_blinking = False
            has_error = (not _last_rtc_ok) or (not _last_imu_ok) or (not _last_gnss_connected) or _program_fault
            all_healthy = _system_ready and (not has_error)
            if all_healthy:
                _leds["yellow"].on()
            else:
                _leds["yellow"].off()

    t = threading.Thread(target=_snap_worker, name="led-snap-blink", daemon=True)
    t.start()


def notify_litter_capture() -> None:
    """
    Blink Yellow LED 2x (double-blink: 90 ms off / 90 ms on) when a frame
    is captured and queued for YOLO litter inference.
    Non-blocking — runs in a daemon background thread.
    Pattern: double yellow blink -> restore steady state.
    """
    global _snap_blinking
    if not _leds:
        return

    def _worker():
        global _snap_blinking
        with _snap_lock:
            _snap_blinking = True
        try:
            for _ in range(2):
                if _leds and "yellow" in _leds:
                    _leds["yellow"].off()
                time.sleep(0.09)
                if _leds and "yellow" in _leds:
                    _leds["yellow"].on()
                time.sleep(0.09)
        finally:
            with _snap_lock:
                _snap_blinking = False
            has_error = (not _last_rtc_ok) or (not _last_imu_ok) or (not _last_gnss_connected) or _program_fault
            all_healthy = _system_ready and (not has_error)
            if _leds and "yellow" in _leds:
                if all_healthy:
                    _leds["yellow"].on()
                else:
                    _leds["yellow"].off()

    t = threading.Thread(target=_worker, name="led-litter-capture-blink", daemon=True)
    t.start()


def notify_litter_snap() -> None:
    """
    Litter DETECTED indicator: Red-Yellow-Red-Yellow alternating blink (2 cycles).
    Fires only when the YOLO model detects litter in a triggered frame.
    Non-blocking — runs in a daemon background thread.

    Pattern (per cycle): Red ON + Yellow OFF → Red OFF + Yellow ON
      Each phase: 120 ms. Total: 4 phases × 120 ms = ~480 ms per cycle × 2 cycles ≈ 960 ms.
    After completion: both LEDs restore to their correct steady state.
    """
    global _litter_blinking
    if not _leds:
        return

    def _litter_worker():
        global _litter_blinking
        with _litter_lock:
            _litter_blinking = True
        try:
            for _ in range(2):
                # Phase 1: Red ON, Yellow OFF
                if _leds and "red" in _leds:
                    _leds["red"].on()
                if _leds and "yellow" in _leds:
                    _leds["yellow"].off()
                time.sleep(0.12)
                # Phase 2: Red OFF, Yellow ON
                if _leds and "red" in _leds:
                    _leds["red"].off()
                if _leds and "yellow" in _leds:
                    _leds["yellow"].on()
                time.sleep(0.12)
            # Final phase: both off briefly then restore
            if _leds and "red" in _leds:
                _leds["red"].off()
            if _leds and "yellow" in _leds:
                _leds["yellow"].off()
            time.sleep(0.05)
        finally:
            with _litter_lock:
                _litter_blinking = False
            # Restore correct LED steady state
            has_error = (not _last_rtc_ok) or (not _last_imu_ok) or (not _last_gnss_connected) or _program_fault
            all_healthy = _system_ready and (not has_error)
            if _leds and "yellow" in _leds:
                _leds["yellow"].on() if all_healthy else _leds["yellow"].off()
            # Red returns to off (fault blinking handled by update() 20 Hz loop)
            if _leds and "red" in _leds and not has_error and not _program_fault:
                _leds["red"].off()

    t = threading.Thread(target=_litter_worker, name="led-litter-blink", daemon=True)
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
      1. RTC Green:  Solid ON if working, OFF on error.
      2. IMU Green:  Solid ON if working, OFF on error.
      3. GNSS Green: Solid ON (fix), Blinking (no fix, connected), OFF (disconnected).
      4. Red LED:    Blinking (400 ms) on ANY module failure or program fault.
      5. Yellow LED: Solid ON when all healthy & ready; 3× on motion snap;
                     Red-Yellow alternating on litter detection (managed by notify_litter_snap).
    """
    global _last_rtc_ok, _last_imu_ok, _last_gnss_fix, _last_gnss_connected
    global _fault_blink_state, _last_fault_toggle
    global _gnss_blink_state, _last_gnss_toggle

    if not _leds:
        return

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

    _last_rtc_ok         = rtc
    _last_imu_ok         = imu
    _last_gnss_fix       = has_fix
    _last_gnss_connected = is_connected

    now = time.monotonic()

    if (now - _last_fault_toggle) > LED_FAULT_BLINK_INTERVAL:
        _fault_blink_state = not _fault_blink_state
        _last_fault_toggle = now

    if (now - _last_gnss_toggle) > GNSS_SEARCH_BLINK_INTERVAL:
        _gnss_blink_state = not _gnss_blink_state
        _last_gnss_toggle = now

    # RTC Green
    _leds["rtc_green"].on() if rtc else _leds["rtc_green"].off()

    # IMU Green
    _leds["imu_green"].on() if imu else _leds["imu_green"].off()

    # GNSS Green
    if has_fix:
        _leds["gnss_green"].on()
    elif is_connected:
        _leds["gnss_green"].on() if _gnss_blink_state else _leds["gnss_green"].off()
    else:
        _leds["gnss_green"].off()

    module_failure = (not rtc) or (not imu) or (not is_connected)
    has_fault = module_failure or _program_fault

    # Red LED — skip if litter blink is active (it controls red directly)
    if not _litter_blinking:
        if has_fault:
            _leds["red"].on() if _fault_blink_state else _leds["red"].off()
        else:
            _leds["red"].off()

    # Yellow LED — skip if snap or litter blink is active
    if not _snap_blinking and not _camera_scanning and not _litter_blinking:
        system_working_and_ready = _system_ready and (not has_fault)
        _leds["yellow"].on() if system_working_and_ready else _leds["yellow"].off()
