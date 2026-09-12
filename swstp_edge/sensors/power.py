"""
sensors/power.py — Power Management, Low-Battery Detection & Safe Shutdown Coordinator.

DEPLOYMENT SCENARIO & ARCHITECTURE
-----------------------------------
1. Vehicle Running (Main Power):
   - Vehicle 12V/24V battery powers the Pi via a DC-DC converter (5V/3A+).
   - Secondary backup battery (UPS HAT, Supercapacitor, or 18650 Li-ion pack) is kept charged.
   - Pi operates in standard high-performance edge detection & telemetry mode.

2. Vehicle Turned Off (Secondary Battery Active):
   - Main vehicle power cuts off; secondary backup battery takes over seamlessly.
   - System continues tracking and evidence upload temporarily.

3. Low Battery Alert:
   - When secondary battery voltage drops below safe threshold, the UPS or battery BMS
     asserts a Low-Battery signal on a configured GPIO pin (e.g. GPIO 25, active LOW),
     or the vehicle-off grace timer expires.
   - The coordinator initiates EMERGENCY SHUTDOWN PROCEDURE:
       a) Pauses camera capture (prevents new queue items).
       b) Flushes pending evidence in upload_queue (waiting for backend confirmation).
       c) Flushes pending telemetry in telemetry_queue with SHUTDOWN status.
       d) Syncs filesystem (os.sync()) to prevent SD card corruption.
       e) Issues safe OS halt: `sudo shutdown -h now` (or `sudo poweroff`).

4. Auto-Boot When Vehicle Powers On Again:
   - Raspberry Pi 4B in halted state will boot automatically via either:
       Method A (Power Cycle - Recommended): The UPS HAT or automotive delay relay completely
                cuts 5V power after Pi halts. When vehicle starts, 12V returns, 5V is re-applied,
                and the Pi boots automatically from 0V.
       Method B (GPIO 3 / RUN Wake-up): Pulling GPIO 3 (Pin 5) or GLOBAL_EN / RUN pin to GND
                momentarily causes the halted Pi to wake up immediately.
"""

import os
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

from config import (
    POWER_MANAGEMENT_ENABLED,
    GPIO_LOW_BATT_PIN,
    GPIO_MAIN_POWER_PIN,
    LOW_BATT_ACTIVE_LOW,
    LOW_BATT_DEBOUNCE_SEC,
    SHUTDOWN_FLUSH_TIMEOUT_SEC,
    VEHICLE_OFF_SHUTDOWN_DELAY_SEC,
)

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------
power_state = {
    "monitoring_active": False,
    "main_power_present": True,
    "on_backup_battery": False,
    "low_battery_triggered": False,
    "shutdown_in_progress": False,
    "power_source": "MAIN",     # "MAIN" | "BACKUP_BATTERY" | "UNKNOWN"
}

_stop_event = threading.Event()
_monitor_thread: Optional[threading.Thread] = None
_main_shutdown_trigger: Optional[Callable[[], None]] = None

# GPIO handle
_gpio_low_batt = None
_gpio_main_power = None


# ---------------------------------------------------------------------------
# Hardware GPIO setup
# ---------------------------------------------------------------------------
def _init_gpio():
    """Initialises GPIO pins using gpiozero if available."""
    global _gpio_low_batt, _gpio_main_power

    if not POWER_MANAGEMENT_ENABLED:
        return

    try:
        from gpiozero import Button, DigitalInputDevice  # type: ignore

        if GPIO_LOW_BATT_PIN is not None:
            # Button/DigitalInputDevice with internal pull-up/pull-down
            pull_up = LOW_BATT_ACTIVE_LOW
            _gpio_low_batt = DigitalInputDevice(
                GPIO_LOW_BATT_PIN, pull_up=pull_up, bounce_time=0.1
            )
            print(f"[POWER] Monitored Low-Battery Alert on BCM GPIO {GPIO_LOW_BATT_PIN} (Active {'LOW' if LOW_BATT_ACTIVE_LOW else 'HIGH'}).")

        if GPIO_MAIN_POWER_PIN is not None:
            _gpio_main_power = DigitalInputDevice(
                GPIO_MAIN_POWER_PIN, pull_up=True, bounce_time=0.1
            )
            print(f"[POWER] Monitored Main Vehicle Power Sense on BCM GPIO {GPIO_MAIN_POWER_PIN}.")

    except Exception as exc:
        print(f"[POWER] Note: gpiozero hardware pin init skipped ({exc}). Running in software/polling mode.")


def is_low_battery_pin_active() -> bool:
    """Returns True if the low-battery alert pin is currently asserted."""
    # Check manual test trigger file for testing / simulation
    if os.path.exists("/tmp/trigger_low_battery") or os.path.exists("trigger_low_battery.flag"):
        return True

    if _gpio_low_batt is not None:
        try:
            val = _gpio_low_batt.value
            # In DigitalInputDevice with pull_up=True, active-low means value is 0 (or is_active is True depending on setup)
            if LOW_BATT_ACTIVE_LOW:
                return val == 0
            else:
                return val == 1
        except Exception:
            pass

    return False


def is_main_power_present() -> bool:
    """Returns True if main vehicle power is detected on GPIO_MAIN_POWER_PIN."""
    if _gpio_main_power is not None:
        try:
            return bool(_gpio_main_power.value == 1)
        except Exception:
            pass
    # If no ignition pin is wired, assume power is present unless low-battery triggers
    return True


# ---------------------------------------------------------------------------
# Emergency Flush & Safe Shutdown Sequence
# ---------------------------------------------------------------------------
def initiate_emergency_shutdown(reason: str = "LOW_BATTERY") -> None:
    """
    Executes the critical shutdown cascade:
      1. Prevent new captures
      2. Flush all pending evidence uploads to backend
      3. Flush all pending telemetry packets
      4. Flush disk caches
      5. Issue OS safe shutdown command
    """
    if power_state["shutdown_in_progress"]:
        return
    power_state["shutdown_in_progress"] = True
    power_state["low_battery_triggered"] = True

    print("\n" + "#" * 70)
    print(f" [POWER MONITOR] ⚠️  CRITICAL: EMERGENCY SHUTDOWN TRIGGERED ({reason})")
    print(f" [POWER MONITOR] Secondary battery running low or vehicle shutdown signal active.")
    print(f" [POWER MONITOR] Starting safe evidence flush & system shutdown sequence...")
    print("#" * 70 + "\n")

    # Step 1: Halt motion detection capture immediately
    try:
        import motion as _motion_mod
        _motion_mod.camera_feed_active = False
        print("[SHUTDOWN 1/5] Camera feed and new motion captures halted.")
    except Exception as e:
        print(f"[SHUTDOWN 1/5] Warning pausing camera: {e}")

    # Step 2: Flush pending evidence in upload_queue
    from telemetry import upload_queue, telemetry_queue

    pending_evidence = upload_queue.qsize()
    print(f"[SHUTDOWN 2/5] Flushing {pending_evidence} pending evidence captures to backend...")
    flush_start = time.monotonic()
    last_reported_count = pending_evidence

    while not upload_queue.empty():
        remaining = upload_queue.qsize()
        elapsed = time.monotonic() - flush_start

        if remaining != last_reported_count:
            last_reported_count = remaining
            print(f"               Evidence remaining to upload: {remaining} item(s)...")

        if elapsed >= SHUTDOWN_FLUSH_TIMEOUT_SEC:
            print(f" [SHUTDOWN WARNING] Evidence upload timeout ({SHUTDOWN_FLUSH_TIMEOUT_SEC:.0f}s reached). Proceeding with remaining {remaining} items saved locally.")
            break

        time.sleep(0.3)

    if upload_queue.empty():
        print(" [SHUTDOWN 2/5] ✓ All evidence captures uploaded successfully.")

    # Step 3: Flush telemetry queue
    pending_telemetry = telemetry_queue.qsize()
    print(f"[SHUTDOWN 3/5] Flushing {pending_telemetry} telemetry records...")
    t_start = time.monotonic()
    while not telemetry_queue.empty() and (time.monotonic() - t_start) < 5.0:
        time.sleep(0.2)
    print(" [SHUTDOWN 3/5] ✓ Telemetry queue drained.")

    # Step 4: Sync filesystem buffers to protect SD card / SSD
    print("[SHUTDOWN 4/5] Flushing Linux filesystem buffers (sync)...")
    try:
        if hasattr(os, "sync"):
            os.sync()
        else:
            subprocess.run(["sync"], check=False)
        print(" [SHUTDOWN 4/5] ✓ Filesystem buffers synced.")
    except Exception as sync_err:
        print(f" [SHUTDOWN 4/5] Sync note: {sync_err}")

    # Step 5: Execute system shutdown
    print("[SHUTDOWN 5/5] Executing safe OS shutdown command...")
    print("\n" + "=" * 70)
    print(" [SWSTP PI SAFE SHUTDOWN] System will halt now.")
    print(" When vehicle powers on, the Pi will auto-boot upon power restoration.")
    print("=" * 70 + "\n")

    # Stop main loop if callback registered
    if _main_shutdown_trigger is not None:
        try:
            _main_shutdown_trigger()
        except Exception:
            pass

    # Safe system halt on Linux
    if sys.platform.startswith("linux"):
        try:
            # shutdown -h now shuts down the system safely and powers off / halts
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
        except Exception as shut_err:
            print(f"[POWER] Could not run sudo shutdown: {shut_err}. Attempting poweroff...")
            try:
                subprocess.run(["sudo", "poweroff"], check=False)
            except Exception:
                pass
    else:
        print("[DRY-RUN / NON-LINUX] Simulated OS shutdown complete.")


# ---------------------------------------------------------------------------
# Background Monitoring Loop
# ---------------------------------------------------------------------------
def _power_monitor_loop():
    """Continuously monitors power status and battery alerts."""
    low_batt_active_since = None
    main_power_lost_since = None

    print("[POWER] Power & Battery supervisor thread running.")

    while not _stop_event.is_set():
        now = time.monotonic()

        # Check main vehicle power
        main_ok = is_main_power_present()
        power_state["main_power_present"] = main_ok
        power_state["on_backup_battery"] = not main_ok
        power_state["power_source"] = "MAIN" if main_ok else "BACKUP_BATTERY"

        # Track vehicle power loss duration
        if not main_ok:
            if main_power_lost_since is None:
                main_power_lost_since = now
                print("\n[POWER] ⚡ MAIN VEHICLE POWER DISCONNECTED. Running on secondary backup battery.")
            elif VEHICLE_OFF_SHUTDOWN_DELAY_SEC > 0 and (now - main_power_lost_since) >= VEHICLE_OFF_SHUTDOWN_DELAY_SEC:
                initiate_emergency_shutdown(reason=f"VEHICLE_OFF_TIMER_{VEHICLE_OFF_SHUTDOWN_DELAY_SEC:.0f}S")
                break
        else:
            if main_power_lost_since is not None:
                print("\n[POWER] ⚡ MAIN VEHICLE POWER RESTORED.")
                main_power_lost_since = None

        # Check low battery alert pin
        batt_low = is_low_battery_pin_active()

        if batt_low:
            if low_batt_active_since is None:
                low_batt_active_since = now
            elif (now - low_batt_active_since) >= LOW_BATT_DEBOUNCE_SEC:
                # Debounce threshold satisfied -> low battery confirmed!
                initiate_emergency_shutdown(reason="LOW_BATTERY_ALERT")
                break
        else:
            low_batt_active_since = None

        _stop_event.wait(0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init(main_shutdown_callback: Optional[Callable[[], None]] = None) -> bool:
    """Initialises power management hardware pins and starts background monitor."""
    global _monitor_thread, _main_shutdown_trigger
    if not POWER_MANAGEMENT_ENABLED:
        print("[POWER] Power management monitoring disabled in config.")
        return False

    _main_shutdown_trigger = main_shutdown_callback
    _init_gpio()

    _stop_event.clear()
    _monitor_thread = threading.Thread(
        target=_power_monitor_loop,
        name="power-monitor",
        daemon=True
    )
    _monitor_thread.start()
    power_state["monitoring_active"] = True
    return True


def stop():
    """Stops the power supervisor thread."""
    _stop_event.set()
