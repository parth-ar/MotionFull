"""
sensors/leds.py — Status LED driver for Raspberry Pi using gpiozero.

Replicates the LED logic from the .ino updateStatusLEDs() / bootSplash()
using gpiozero instead of digitalWrite(), per Hard Constraint #6.

LED mapping (BCM GPIO pins — configurable in config.py):
  RTC_GREEN  : RTC module online  → solid ON | fault blink
  IMU_GREEN  : IMU module online  → solid ON | fault blink
  GNSS_GREEN : GNSS fix active    → solid ON | fault blink
  YELLOW     : Heartbeat          → 1 Hz blink when all OK | OFF otherwise
  RED        : Fault indicator    → blinks when any module is faulting

All LEDs are active-HIGH (logic 1 = LED on) — wire LED + through a current-
limiting resistor (220 Ω–470 Ω recommended) to the BCM GPIO pin, and LED −
to GND.

VOLTAGE NOTE:
Pi GPIO outputs are 3.3 V at up to ~16 mA per pin.  Do NOT connect LEDs
directly without a current-limiting resistor.  Do NOT drive 5 V LEDs without
additional circuitry.
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

# Blink state
_fault_blink_state = False
_yellow_state = False
_last_fault_toggle = 0.0
_last_yellow_toggle = 0.0


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
# LED update — port of .ino updateStatusLEDs()
# ---------------------------------------------------------------------------
def update(rtc: bool, imu: bool, gnss: bool) -> None:
    """
    Call this at ~20 Hz from the telemetry loop.  Mirrors .ino updateStatusLEDs():

    - RTC_GREEN  : solid ON if rtcOK, else blink at FAULT_BLINK_INTERVAL
    - IMU_GREEN  : solid ON if imuOK, else blink at FAULT_BLINK_INTERVAL
    - GNSS_GREEN : solid ON if gnssFix, else blink at FAULT_BLINK_INTERVAL
    - RED        : blinks when any module is faulting; OFF when all OK
    - YELLOW     : blinks at YELLOW_BLINK_INTERVAL when all OK; OFF otherwise
    """
    global _fault_blink_state, _yellow_state
    global _last_fault_toggle, _last_yellow_toggle

    if not _leds:
        return

    now = time.monotonic()

    # Fault blink tick
    if (now - _last_fault_toggle) > LED_FAULT_BLINK_INTERVAL:
        _fault_blink_state   = not _fault_blink_state
        _last_fault_toggle   = now

    all_ok = rtc and imu and gnss

    # Individual module LEDs
    _leds["rtc_green"].on()  if rtc  else (_leds["rtc_green"].on()  if _fault_blink_state else _leds["rtc_green"].off())
    _leds["imu_green"].on()  if imu  else (_leds["imu_green"].on()  if _fault_blink_state else _leds["imu_green"].off())
    _leds["gnss_green"].on() if gnss else (_leds["gnss_green"].on() if _fault_blink_state else _leds["gnss_green"].off())

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
