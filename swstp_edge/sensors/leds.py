"""
sensors/leds.py — Status LED driver for Raspberry Pi using gpiozero.

Replicates the LED logic from the .ino updateStatusLEDs() / bootSplash()
using gpiozero instead of digitalWrite(), per Hard Constraint #6.

LED mapping (BCM GPIO pins — configurable in config.py):
  RTC_GREEN  : RTC module online  → solid ON | fault blink
  IMU_GREEN  : IMU module online  → solid ON | fault blink
  GNSS_GREEN : GNSS fix active    → solid ON | fault blink
  YELLOW     : Heartbeat (normal) | Solid ON (headless ready) | 3×blink (snap)
  RED        : Fault indicator    → blinks when any module is faulting
               In headless mode   → blinks whenever set_fault() is called

All LEDs are active-HIGH (logic 1 = LED on) — wire LED + through a current-
limiting resistor (220 Ω–470 Ω recommended) to the BCM GPIO pin, and LED −
to GND.

VOLTAGE NOTE:
Pi GPIO outputs are 3.3 V at up to ~16 mA per pin.  Do NOT connect LEDs
directly without a current-limiting resistor.  Do NOT drive 5 V LEDs without
additional circuitry.

Headless mode LED behaviour
---------------------------
Call set_headless_mode(True) once at startup (from main.py when --headless).
Then call the following helpers as needed:

  set_system_ready()    → Yellow solid ON, Red OFF  (system is live & ready)
  notify_motion_snap()  → Yellow blinks 3× (100 ms each), then back to solid ON
  set_fault(msg)        → Yellow OFF, Red blinks until set_system_ready() called

The 20 Hz update() call continues to manage the green module-status LEDs
regardless of headless mode.
"""

import threading
import time

from config import (
    LED_RTC_GREEN, LED_IMU_GREEN, LED_GNSS_GREEN, LED_YELLOW, LED_RED,
    LED_FAULT_BLINK_INTERVAL, LED_YELLOW_BLINK_INTERVAL,
)

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
    """No-op LED for non-Pi environments."""
    def __init__(self, pin): self.pin = pin
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


# LED objects (initialised in init())
_leds: dict = {}
_stop_event = threading.Event()
_thread: threading.Thread | None = None

# ---------------------------------------------------------------------------
# Legacy (non-headless) blink state — used by update()
# ---------------------------------------------------------------------------
_fault_blink_state  = False
_yellow_state       = False
_last_fault_toggle  = 0.0
_last_yellow_toggle = 0.0

# ---------------------------------------------------------------------------
# Headless mode state
# ---------------------------------------------------------------------------
_headless_mode: bool = False

# Possible headless yellow/red states
_HL_READY   = "ready"    # yellow solid ON, red OFF
_HL_SNAP    = "snap"     # yellow blinking 3× (managed by thread)
_HL_FAULT   = "fault"    # yellow OFF, red blinking
_hl_state   = _HL_FAULT  # start in fault until set_system_ready() called

_hl_lock          = threading.Lock()
_snap_thread: threading.Thread | None = None   # short-lived 3-blink thread

# Fault blink state (headless red)
_hl_red_blink_state  = False
_hl_last_red_toggle  = 0.0


# ---------------------------------------------------------------------------
# Public state (set by telemetry.py / main.py after sensor reads)
# ---------------------------------------------------------------------------
rtc_ok:   bool = False
imu_ok:   bool = False
gnss_fix: bool = False


# ---------------------------------------------------------------------------
# Init / cleanup
# ---------------------------------------------------------------------------
def init() -> None:
    """Open GPIO handles and run the 3-flash boot splash (≈ 1.5 s)."""
    global _leds
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
    _stop_event.set()
    _all_off()
    for led in _leds.values():
        try:
            led.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Boot splash — 3-cycle flash (≈ 1.5 s), mirrors .ino bootSplash()
# ---------------------------------------------------------------------------
def _all_on() -> None:
    for led in _leds.values(): led.on()

def _all_off() -> None:
    for led in _leds.values(): led.off()


def _boot_splash() -> None:
    """Port of .ino bootSplash(): 3 × (200 ms ON + 250 ms OFF)."""
    for _ in range(3):
        _all_on()
        time.sleep(0.200)
        _all_off()
        time.sleep(0.250)
    print("[LEDS] Boot splash complete.")


# ---------------------------------------------------------------------------
# Headless mode public API
# ---------------------------------------------------------------------------
def set_headless_mode(enabled: bool) -> None:
    """
    Enable or disable headless LED mode.  Call with enabled=True from main.py
    when --headless flag is active, immediately after leds.init().
    """
    global _headless_mode
    _headless_mode = enabled
    if enabled:
        print("[LEDS] Headless mode enabled — yellow/red managed by headless state machine.")


def set_system_ready() -> None:
    """
    Signal that the system is operational and ready to capture.
    Yellow → solid ON.  Red → OFF.
    Call once after all sensors initialise successfully (headless mode only).
    """
    global _hl_state
    if not _leds:
        return
    with _hl_lock:
        _hl_state = _HL_READY
    # Apply immediately (outside lock for thread safety with _DummyLED)
    _leds["yellow"].on()
    _leds["red"].off()
    print("[LEDS] Status: READY — Yellow solid ON.")


def notify_motion_snap() -> None:
    """
    Blink yellow 3 times (100 ms each) in a background thread to signal that
    a motion snap was taken, then restore solid ON.
    Non-blocking — returns immediately.
    """
    if not _leds or not _headless_mode:
        return

    def _blink_worker():
        global _hl_state
        with _hl_lock:
            # Only blink if we are currently in ready state
            if _hl_state != _HL_READY:
                return
            _hl_state = _HL_SNAP

        try:
            for _ in range(3):
                _leds["yellow"].off()
                time.sleep(0.10)
                _leds["yellow"].on()
                time.sleep(0.10)
        finally:
            # Restore ready state (solid ON)
            with _hl_lock:
                if _hl_state == _HL_SNAP:
                    _hl_state = _HL_READY
            _leds["yellow"].on()

    t = threading.Thread(target=_blink_worker, name="led-snap-blink", daemon=True)
    t.start()


def set_fault(reason: str = "") -> None:
    """
    Signal that a fault has occurred.
    Yellow → OFF.  Red → blink at LED_FAULT_BLINK_INTERVAL until
    set_system_ready() is called.
    """
    global _hl_state
    if not _leds:
        return
    with _hl_lock:
        _hl_state = _HL_FAULT
    _leds["yellow"].off()
    # Red blinking is handled by _tick_headless_red() called from update()
    if reason:
        print(f"[LEDS] Status: FAULT — Yellow OFF, Red blinking. Reason: {reason}")
    else:
        print("[LEDS] Status: FAULT — Yellow OFF, Red blinking.")


def _tick_headless_red() -> None:
    """
    Called from update() at ~20 Hz.  Manages the red LED blink when in
    headless fault mode.  No-op otherwise.
    """
    global _hl_red_blink_state, _hl_last_red_toggle
    now = time.monotonic()
    if (now - _hl_last_red_toggle) > LED_FAULT_BLINK_INTERVAL:
        _hl_red_blink_state  = not _hl_red_blink_state
        _hl_last_red_toggle  = now
    if _hl_red_blink_state:
        _leds["red"].on()
    else:
        _leds["red"].off()


# ---------------------------------------------------------------------------
# LED update — port of .ino updateStatusLEDs()
# ---------------------------------------------------------------------------
def update(rtc: bool, imu: bool, gnss: bool) -> None:
    """
    Call this at ~20 Hz from the telemetry loop.

    Green module-status LEDs behave identically in both modes:
      - RTC_GREEN  : solid ON if rtcOK, else blink at FAULT_BLINK_INTERVAL
      - IMU_GREEN  : solid ON if imuOK, else blink at FAULT_BLINK_INTERVAL
      - GNSS_GREEN : solid ON if gnssFix, else blink at FAULT_BLINK_INTERVAL

    In NORMAL (non-headless) mode (original behaviour):
      - RED        : blinks when any module is faulting; OFF when all OK
      - YELLOW     : 1 Hz heartbeat blink when all OK; OFF otherwise

    In HEADLESS mode:
      - RED / YELLOW are controlled exclusively by the headless state machine
        (set_system_ready / notify_motion_snap / set_fault).
        update() only drives red blinking in the FAULT state.
    """
    global _fault_blink_state, _yellow_state
    global _last_fault_toggle, _last_yellow_toggle

    if not _leds:
        return

    now = time.monotonic()

    # Fault blink tick (shared between modes for green LEDs)
    if (now - _last_fault_toggle) > LED_FAULT_BLINK_INTERVAL:
        _fault_blink_state  = not _fault_blink_state
        _last_fault_toggle  = now

    all_ok = rtc and imu and gnss

    # --- Green module-status LEDs (same in both modes) ---
    _leds["rtc_green"].on()  if rtc  else (_leds["rtc_green"].on()  if _fault_blink_state else _leds["rtc_green"].off())
    _leds["imu_green"].on()  if imu  else (_leds["imu_green"].on()  if _fault_blink_state else _leds["imu_green"].off())
    _leds["gnss_green"].on() if gnss else (_leds["gnss_green"].on() if _fault_blink_state else _leds["gnss_green"].off())

    if _headless_mode:
        # --- Headless: yellow managed by state machine; tick red if in fault ---
        with _hl_lock:
            current_state = _hl_state
        if current_state == _HL_FAULT:
            _tick_headless_red()
        elif current_state == _HL_READY:
            # Ensure red is off (guard against stale state)
            _leds["red"].off()
        # _HL_SNAP: snap-blink thread owns yellow; don't touch red
    else:
        # --- Normal mode: legacy red + yellow heartbeat ---
        # RED — on (blinking) when any fault
        if all_ok:
            _leds["red"].off()
        else:
            _leds["red"].on() if _fault_blink_state else _leds["red"].off()

        # YELLOW — heartbeat blink when all OK
        if all_ok:
            if (now - _last_yellow_toggle) > LED_YELLOW_BLINK_INTERVAL:
                _yellow_state        = not _yellow_state
                _last_yellow_toggle  = now
            _leds["yellow"].on() if _yellow_state else _leds["yellow"].off()
        else:
            _yellow_state = False
            _leds["yellow"].off()
