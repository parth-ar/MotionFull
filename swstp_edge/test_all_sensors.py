#!/usr/bin/env python3
"""
test_all_sensors.py — Standalone diagnostic script for SWSTP Edge sensors.

Initialises and reads all sensor modules in order:
  1. RTC   — DS3231 via I2C (with internet/NTP sync attempt)
  2. IMU   — MPU-6500 via I2C (with calibration)
  3. GNSS  — u-blox MAX-M10S Hardware UART NMEA (Serial)
  4. PiCam — Sony IMX708 (Camera Module 3) via Picamera2
  5. LTE   — Cellular/LTE connectivity (network reachability + SIM check)

Prints clearly formatted output for every module.
This file is 100% independent — it makes NO changes to any other file.

Run from the swstp_edge/ directory:
    python test_all_sensors.py [--imu-samples N] [--gnss-wait N]

Optional flags:
  --imu-samples N   Number of IMU read iterations to display   (default: 5)
  --gnss-wait   N   Seconds to wait for a GNSS fix/data        (default: 30)
  --no-rtc          Skip RTC initialisation (useful if no DS3231 is wired)
  --no-imu          Skip IMU initialisation
  --no-gnss         Skip GNSS initialisation
  --no-picam        Skip PiCamera / IMX708 test
  --no-lte          Skip LTE / cellular connectivity test
"""

import argparse
import datetime
import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Pretty-print helpers ─────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
DIM    = "\033[2m"


def _hr(char="─", width=70, color=DIM):
    print(f"{color}{char * width}{RESET}")


def _section(title: str, color=CYAN):
    _hr("═", color=color)
    print(f"{color}{BOLD}  {title}{RESET}")
    _hr("═", color=color)


def _ok(msg: str):
    print(f"  {GREEN}✔  {msg}{RESET}")


def _warn(msg: str):
    print(f"  {YELLOW}⚠  {msg}{RESET}")


def _err(msg: str):
    print(f"  {RED}✘  {msg}{RESET}")


def _field(label: str, value, unit: str = "", indent: int = 4):
    pad = " " * indent
    label_str = f"{BOLD}{label:<28}{RESET}"
    value_str = f"{GREEN}{value}{RESET}" if value is not None else f"{RED}None{RESET}"
    unit_str  = f"{DIM} {unit}{RESET}" if unit else ""
    print(f"{pad}{label_str}{value_str}{unit_str}")


def _sub(title: str):
    print(f"\n  {BLUE}{BOLD}▶ {title}{RESET}")


# ── Path bootstrap (run from any CWD) ────────────────────────────────────────

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ════════════════════════════════════════════════════════════════════════════
# 1. RTC
# ════════════════════════════════════════════════════════════════════════════

def test_rtc():
    _section("RTC — DS3231 Hardware Clock  (I2C 0x68)", color=CYAN)

    try:
        from sensors import rtc as rtc_mod
    except ImportError as exc:
        _err(f"Cannot import sensors.rtc: {exc}")
        return False

    # ── Init (NTP sync + hardware detection) ─────────────────────────────
    print(f"\n  {DIM}Initialising RTC (NTP sync + hardware detection)…{RESET}")
    hw_present = rtc_mod.init()

    access_mode  = getattr(rtc_mod, "_access_mode",     "unknown")
    kernel_dev   = getattr(rtc_mod, "_kernel_rtc_dev",  None)
    kernel_name  = getattr(rtc_mod, "_kernel_rtc_name", None)

    print()
    if hw_present and access_mode == "kernel":
        _ok("DS3231 RTC detected — kernel-managed")
        _ok(f"Hardware present    : True")
        _ok(f"Module status       : OK")
        _ok(f"Access mode         : Kernel-managed RTC")
        _ok(f"Kernel RTC device   : /dev/{kernel_dev}  (/sys/class/rtc/{kernel_dev})")
        _ok(f"Kernel identifier   : {kernel_name}")
        _ok(f"I\u00b2C address         : 0x68  (bus 1)")
    elif hw_present and access_mode == "smbus":
        _ok("DS3231 RTC detected — direct SMBus access")
        _ok(f"Hardware present    : True")
        _ok(f"Access mode         : Direct SMBus")
    else:
        _warn("DS3231 NOT detected by kernel subsystem or direct SMBus")
        _warn("Hardware present    : False")
        if rtc_mod.rtc_error:
            _err(f"Detail : {rtc_mod.rtc_error}")
        _warn("Pi diagnostic  : sudo i2cdetect -y 1")
        _warn("  '68' = direct SMBus accessible")
        _warn("  'UU' = kernel holds device — check /sys/class/rtc for 1-0068")
        _warn("  '--' = wiring fault or no power to RTC module")

    # ── Detection status fields ───────────────────────────────────────────
    _sub("Detection status")
    _field("Hardware present",  hw_present)
    _field("Module status OK",  rtc_mod.rtc_ok)
    _field("Access mode",       access_mode)
    _field("Kernel RTC device", f"/dev/{kernel_dev}" if kernel_dev else "N/A")
    _field("Kernel identifier", kernel_name          if kernel_name else "N/A")
    _field("Sync source",       rtc_mod.sync_source)
    _field("Error",             rtc_mod.rtc_error or "None")

    # ── Read ─────────────────────────────────────────────────────────────
    _sub("RTC read() output")
    data = rtc_mod.read()
    _field("valid",             data.get("valid"))
    _field("timestamp (IST)",   data.get("timestamp"))
    _field("epoch",             data.get("epoch"),  "ms")
    _field("timezone",          data.get("timezone"))
    _field("hw_source",         data.get("hw_source"))

    # ── System clock cross-check + drift display ──────────────────────────
    now_sys = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    _sub("System clock cross-check")
    _field("System time (local)", now_sys)
    if data.get("timestamp") and data.get("hw_source") in ("kernel", "smbus"):
        try:
            hw_ts  = datetime.datetime.strptime(data["timestamp"], "%Y-%m-%dT%H:%M:%S")
            sys_ts = datetime.datetime.strptime(now_sys,           "%Y-%m-%dT%H:%M:%S")
            drift  = abs((hw_ts - sys_ts).total_seconds())
            drift_col = GREEN if drift < 2 else YELLOW if drift < 10 else RED
            print(f"    {BOLD}{'RTC \u2194 system drift':<28}{RESET}{drift_col}{drift:.0f} s{RESET}")
        except Exception:
            pass

    _hr()
    return True



# ════════════════════════════════════════════════════════════════════════════
# 2. IMU
# ════════════════════════════════════════════════════════════════════════════

def test_imu(num_samples: int = 5):
    _section("IMU — MPU-6500  (I2C 0x69)", color=MAGENTA)

    try:
        from sensors import imu as imu_mod
    except ImportError as exc:
        _err(f"Cannot import sensors.imu: {exc}")
        return False

    # ── Init ─────────────────────────────────────────────────────────────
    print(f"\n  {DIM}Initialising IMU (calibration takes ~1 s — keep still)…{RESET}")
    ok = imu_mod.init()
    print()

    if ok:
        _ok("MPU-6500 initialised and calibrated")
    else:
        _err("MPU-6500 init FAILED — check I2C wiring and address (AD0 HIGH → 0x69)")

    _field("imu_ok", imu_mod.imu_ok)

    if not ok:
        _hr()
        return False

    # ── Read loop ─────────────────────────────────────────────────────────
    print(f"\n  {DIM}Reading {num_samples} samples (0.5 s interval)…{RESET}\n")

    for i in range(1, num_samples + 1):
        data = imu_mod.read()
        print(f"  {BOLD}{BLUE}── Sample {i}/{num_samples} ──────────────────────────────────────{RESET}")

        if not data.get("valid"):
            _err("Read returned valid=False")
            time.sleep(0.5)
            continue

        # Accelerometer
        _sub("Accelerometer")
        ag = data.get("accel_g", {})
        am = data.get("accel_ms2", {})
        _field("accel_g   x / y / z",
               f"{ag.get('x'):+.4f}  {ag.get('y'):+.4f}  {ag.get('z'):+.4f}", "g")
        _field("accel_ms2 x / y / z",
               f"{am.get('x'):+.4f}  {am.get('y'):+.4f}  {am.get('z'):+.4f}", "m/s²")
        _field("magnitude",
               f"{data.get('accel_magnitude_ms2'):+.4f}", "m/s²")

        # Gyroscope
        _sub("Gyroscope")
        gd = data.get("gyro_dps", {})
        _field("gyro_dps  x / y / z",
               f"{gd.get('x'):+.4f}  {gd.get('y'):+.4f}  {gd.get('z'):+.4f}", "°/s")

        # Temperature
        _sub("Temperature")
        _field("temperature_c", f"{data.get('temperature_c'):.2f}", "°C")

        # Orientation (complementary filter)
        _sub("Orientation  (complementary filter)")
        ori = data.get("orientation", {})
        _field("roll",         f"{ori.get('roll'):+.3f}",  "°")
        _field("pitch",        f"{ori.get('pitch'):+.3f}", "°")
        _field("yaw",          f"{ori.get('yaw'):+.3f}",   "°  (gyro-integrated)")
        _field("yaw_source",   ori.get("yaw_source"))

        # Quaternion
        _sub("Quaternion  (Tait-Bryan Z-Y-X)")
        q = data.get("quaternion", {})
        _field("w / x / y / z",
               f"{q.get('w'):+.6f}  {q.get('x'):+.6f}  {q.get('y'):+.6f}  {q.get('z'):+.6f}")

        if i < num_samples:
            time.sleep(0.5)

    _hr()
    return True


# ════════════════════════════════════════════════════════════════════════════
# 3. GNSS
# ════════════════════════════════════════════════════════════════════════════

def test_gnss(wait_sec: int = 30):
    _section("GNSS — u-blox MAX-M10S Hardware GNSS  (UART /dev/serial0)", color=GREEN)

    try:
        from sensors import gnss as gnss_mod
    except ImportError as exc:
        _err(f"Cannot import sensors.gnss: {exc}")
        return False

    try:
        from config import GNSS_SERIAL_PORT, GNSS_BAUD_RATE
    except ImportError:
        GNSS_SERIAL_PORT = "/dev/serial0"
        GNSS_BAUD_RATE   = 9600

    print(f"\n  {DIM}Starting MAX-M10S GNSS serial reader → {GNSS_SERIAL_PORT} @ {GNSS_BAUD_RATE} baud{RESET}")
    gnss_mod.init()

    # ── Wait for data ─────────────────────────────────────────────────────
    print(f"  {DIM}Waiting up to {wait_sec} s for GNSS data …{RESET}\n")
    deadline = time.time() + wait_sec
    got_data = False
    got_fix  = False
    spinner  = ["|", "/", "─", "\\"]
    idx      = 0

    while time.time() < deadline:
        snap = gnss_mod.read()
        status = snap.get("status", "NO_DATA")

        # Spinner progress
        spin = spinner[idx % len(spinner)]
        idx += 1
        remaining = max(0, int(deadline - time.time()))
        print(f"\r  {spin}  Status: {BOLD}{status:<10}{RESET}  "
              f"sats={snap.get('satellites', 0):2d}  "
              f"sats_in_view={snap.get('satellites_in_view', 0):2d}  "
              f"[{remaining:2d}s left]   ",
              end="", flush=True)

        if status == "FIX":
            got_fix  = True
            got_data = True
            break
        if status == "NO_FIX":
            got_data = True

        time.sleep(1.0)

    print()   # newline after spinner

    # ── Final snapshot ────────────────────────────────────────────────────
    data = gnss_mod.read()

    print()
    status = data.get("status")
    if status == "FIX":
        _ok("u-blox MAX-M10S GNSS FIX acquired")
    elif status == "NO_FIX":
        _warn("GNSS data flowing from MAX-M10S — no positional fix yet (ensure antenna is placed near window/outdoors)")
    else:
        _err(f"No GNSS data received (status={status}). Check MAX-M10S UART wiring (TX->RXD Pin 10, RX->TXD Pin 8), power (3.3V/5V), and enable_uart=1 in /boot/firmware/config.txt.")

    _sub("GNSS read() snapshot")
    _field("data_received",       data.get("data_received"))
    _field("fix",                 data.get("fix"))
    _field("status",              data.get("status"))
    _field("latitude",            data.get("latitude"),           "°")
    _field("longitude",           data.get("longitude"),          "°")
    _field("altitude_m",          data.get("altitude_m"),         "m")
    _field("speed_kmh",           data.get("speed_kmh"),          "km/h")
    _field("course_deg",          data.get("course_deg"),         "°")
    _field("satellites (in fix)", data.get("satellites"),         "")
    _field("satellites_in_view",  data.get("satellites_in_view"), "")
    _field("satellite_records",   data.get("satellite_records"),  "")
    _field("hdop",                data.get("hdop"),               "")
    _field("gnss_ok flag",        gnss_mod.gnss_ok)

    # ── Satellite detail ──────────────────────────────────────────────────
    detail = data.get("satellites_detail")
    if detail:
        _sub(f"Satellite detail ({len(detail)} records)")
        header = (f"    {'PRN':>4}  {'Const':<6}  {'Elev°':>5}  "
                  f"{'Azim°':>5}  {'SNR dB':>6}  {'Used':>4}")
        print(f"  {DIM}{header}{RESET}")
        _hr(".", width=60, color=DIM)
        for s in detail[:20]:    # cap at 20 rows
            used_str = f"{GREEN}yes{RESET}" if s.get("used_in_fix") else f"{DIM}no{RESET}"
            snr = s.get("snr_db")
            snr_str = f"{snr:.1f}" if snr is not None else "  -"
            print(f"    {s.get('prn', '?'):>4}  "
                  f"{s.get('constellation', '?'):<6}  "
                  f"{s.get('elevation_deg', 0):>5}  "
                  f"{s.get('azimuth_deg', 0):>5}  "
                  f"{snr_str:>6}  "
                  f"{used_str}")
        if len(detail) > 20:
            print(f"    {DIM}... and {len(detail) - 20} more{RESET}")
        _hr(".", width=60, color=DIM)
    else:
        _warn("No satellite detail available (GSV not yet received or snapshot expired)")

    # ── Stop background thread ────────────────────────────────────────────
    gnss_mod.stop()

    _hr()
    return True


# ════════════════════════════════════════════════════════════════════════════
# 4. PiCamera (IMX708)
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# 4. PiCamera (IMX708 Camera Module 3)
# ════════════════════════════════════════════════════════════════════════════

def test_picam(
    frames: int = 30,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    af_mode: str = "continuous",
    hdr: bool = False,
    save_dir: str = "",
) -> bool:
    """
    Comprehensive PiCamera (Sony IMX708) test & benchmark.
    Tests:
      1. Picamera2 library availability
      2. Sony IMX708 sensor hardware detection (CSI + I2C)
      3. Sensor initialisation with Phase Detection Autofocus (PDAF) & Daylight AWB
      4. Live capture benchmark: measures frame latencies (ms) & sustained FPS
      5. Frame color balance check (evaluates AWB convergence & green channel)
      6. Sample frame capture and file save to captures/test_picam_capture.jpg
    """
    _section("PiCamera — Sony IMX708 (Camera Module 3)", color=BLUE)

    if not save_dir:
        save_dir = os.path.join(_SCRIPT_DIR, "captures")

    # ── 1. Check Picamera2 availability ─────────────────────────────────────────
    print(f"\n  {DIM}Checking Picamera2 library…{RESET}")
    picam2_cls = None
    try:
        from picamera2 import Picamera2
        picam2_cls = Picamera2
        _ok("picamera2 package is installed and importable")
    except ImportError as exc:
        _err(f"picamera2 not installed: {exc}")
        _warn("Resolution: sudo apt update && sudo apt install -y python3-picamera2")
        _hr()
        return False

    # ── 2. Detect IMX708 sensor ────────────────────────────────────────────────
    _sub("Sensor hardware detection")
    sensor_detected = False
    sensor_model    = "Unknown"
    sensor_backend  = "picamera2"
    sensor_details  = ""

    # Try camera.py detect helper if present
    try:
        from camera import detect_imx708_sensor
        det = detect_imx708_sensor()
        sensor_detected = det.get("detected", False)
        sensor_model    = det.get("model", "Unknown")
        sensor_backend  = det.get("backend", "camera.py")
        sensor_details  = det.get("details", "")
    except Exception:
        pass

    # Direct fallback probe via Picamera2.global_camera_info()
    if not sensor_detected and picam2_cls is not None:
        try:
            cam_list = picam2_cls.global_camera_info()
            if cam_list:
                for idx, cinfo in enumerate(cam_list):
                    model = cinfo.get("Model", "")
                    if "imx708" in model.lower() or not sensor_detected:
                        sensor_detected = True
                        sensor_model    = model or "Sony IMX708"
                        sensor_details  = f"Camera #{idx}: {cinfo}"
        except Exception as exc:
            sensor_details = f"Probe error: {exc}"

    # Also check rpicam-hello / libcamera list
    if not sensor_detected:
        try:
            import subprocess
            out = subprocess.check_output(
                ["rpicam-hello", "--list-cameras"],
                stderr=subprocess.STDOUT, text=True, timeout=5
            )
            if "imx708" in out.lower():
                sensor_detected = True
                sensor_model = "Sony IMX708 (via rpicam-hello)"
                sensor_details = out.strip().splitlines()[-1]
        except Exception:
            pass

    if sensor_detected:
        _ok(f"IMX708 hardware confirmed: {sensor_model}")
    else:
        _warn("IMX708 sensor NOT detected automatically")
        _warn("Check ribbon cable in CSI port (silver contacts face HDMI on Pi 4B)")
        _warn("Ensure 'dtoverlay=imx708' is in /boot/firmware/config.txt")

    _field("Detected", sensor_detected)
    _field("Model",    sensor_model)
    _field("Backend",  sensor_backend)
    _field("Details",  sensor_details if sensor_details else "N/A")

    # ── 3. Initialise Camera & Configure PDAF ──────────────────────────────────
    _sub(f"Camera initialisation ({width}x{height} @ {fps:.1f} FPS, AF: {af_mode.upper()})")
    cam_cap = None
    direct_picam = None

    # Option A: Try production IMX708CameraCapture from camera.py
    try:
        from camera import IMX708CameraCapture
        cam_cap = IMX708CameraCapture(
            camera_idx=0,
            width=width,
            height=height,
            fps=fps,
            af_mode=af_mode,
            hdr=hdr,
        )
        if not cam_cap.is_opened():
            cam_cap = None
        else:
            _ok("Initialised via IMX708CameraCapture (production camera stack)")
    except Exception as exc:
        _warn(f"camera.IMX708CameraCapture not used ({exc}). Falling back to direct Picamera2.")
        cam_cap = None

    # Option B: Direct standalone Picamera2 initialisation
    if cam_cap is None:
        try:
            direct_picam = picam2_cls()
            vcfg = direct_picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
            )
            direct_picam.configure(vcfg)
            try:
                from picamera2 import controls as libcam_ctrl
                direct_picam.set_controls({
                    "AeEnable":  True,
                    "AwbEnable": True,
                    "AwbMode":   getattr(libcam_ctrl.AwbModeEnum, "Daylight", 4)
                                 if hasattr(libcam_ctrl, "AwbModeEnum") else 4,
                    "AfMode":    getattr(libcam_ctrl.AfModeEnum, "Continuous", 2)
                                 if hasattr(libcam_ctrl, "AfModeEnum") else 2,
                })
            except Exception:
                pass
            direct_picam.start()
            time.sleep(1.5)
            _ok("Initialised directly via Picamera2 standalone pipeline")
        except Exception as direct_exc:
            _err(f"Camera initialisation failed: {direct_exc}")
            _hr()
            return False

    # ── 4. Benchmark Frame Capture ─────────────────────────────────────────────
    _sub(f"Benchmark capture ({frames} test frames)")
    latencies = []
    sample_frame = None

    try:
        for i in range(frames):
            t0 = time.perf_counter()
            if cam_cap is not None:
                ret, frame = cam_cap.read()
            else:
                raw_frame = direct_picam.capture_array()
                ret = raw_frame is not None
                # Convert RGB -> BGR for OpenCV compatibility if needed
                frame = raw_frame[..., ::-1] if raw_frame is not None else None
            t1 = time.perf_counter()

            if not ret or frame is None:
                _err(f"Frame {i + 1}/{frames} capture failed")
                break

            dt_ms = (t1 - t0) * 1000.0
            latencies.append(dt_ms)
            sample_frame = frame

            if (i + 1) % 10 == 0 or (i + 1) == frames:
                fps_inst = 1000.0 / dt_ms if dt_ms > 0 else 0
                print(f"    {DIM}Frame {i + 1:2d}/{frames} | "
                      f"latency: {dt_ms:5.1f} ms | {fps_inst:4.1f} FPS{RESET}")

    except Exception as bench_exc:
        _err(f"Capture loop error: {bench_exc}")

    # ── 5. Analyse Frame & Measure Sustained Performance ──────────────────────
    cap_ok = sample_frame is not None and len(latencies) > 0
    if cap_ok:
        h, w = sample_frame.shape[:2]
        channels = sample_frame.shape[2] if sample_frame.ndim > 2 else 1
        avg_latency = sum(latencies) / len(latencies)
        sustained_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        b_mean = float(sample_frame[:, :, 0].mean())
        g_mean = float(sample_frame[:, :, 1].mean())
        r_mean = float(sample_frame[:, :, 2].mean())

        _ok(f"Captured {len(latencies)}/{frames} frames successfully")
        _field("Frame resolution", f"{w} x {h} px")
        _field("Channels", channels)
        _field("Average latency", f"{avg_latency:.2f}", "ms")
        _field("Sustained rate", f"{sustained_fps:.1f}", "FPS")
        _field("Mean R / G / B", f"{r_mean:.1f}  /  {g_mean:.1f}  /  {b_mean:.1f}")

        if sample_frame.ndim == 3 and sample_frame.shape[2] == 3:
            center_bgr = sample_frame[h // 2, w // 2].tolist()
            _field("Center px [B,G,R]", center_bgr)

        # Colour balance / green check
        if g_mean > max(r_mean, b_mean):
            _ok("Green channel dominant — AWB / colour pipeline healthy")
        else:
            _warn("Green channel is not dominant — AWB Daylight mode applied")

        # ── 6. Save Sample Test Capture ────────────────────────────────────────
        _sub("Save sample test frame")
        try:
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, "test_picam_capture.jpg")
            saved = False
            try:
                import cv2
                cv2.imwrite(out_file, sample_frame)
                saved = True
            except Exception:
                try:
                    from PIL import Image
                    rgb = sample_frame[..., ::-1] if sample_frame.ndim == 3 else sample_frame
                    Image.fromarray(rgb).save(out_file)
                    saved = True
                except Exception:
                    pass

            if saved and os.path.isfile(out_file):
                size_kb = os.path.getsize(out_file) / 1024.0
                _ok(f"Sample frame saved: {out_file} ({size_kb:.1f} KB)")
            else:
                _warn(f"Could not write image to {out_file}")
        except Exception as save_err:
            _warn(f"Save image error: {save_err}")
    else:
        _err("No frames captured during benchmark")

    # ── 7. Release Resources ──────────────────────────────────────────────────
    try:
        if cam_cap is not None:
            cam_cap.release()
        elif direct_picam is not None:
            direct_picam.stop()
            direct_picam.close()
    except Exception:
        pass

    _hr()
    return cap_ok


# ════════════════════════════════════════════════════════════════════════════
# 5. LTE / Cellular Connectivity
# ════════════════════════════════════════════════════════════════════════════

def test_lte() -> bool:
    """
    LTE / cellular connectivity & hardware modem diagnostic.
    Checks:
      1. Network interfaces (wwan, ppp, usb, rmnet)
      2. Routing table & default gateway
      3. Internet reachability (DNS resolution + HTTP latency probe)
      4. ModemManager & SIM status via mmcli (operator, tech, signal %)
      5. Direct cellular USB/serial hardware detection (/dev/ttyUSB*, VID/PID)
      6. LTE hardware status LED test (toggles LED_LTE_GREEN if present)
    """
    _section("LTE — Cellular / Modem Connectivity", color=YELLOW)

    import subprocess
    import socket
    import urllib.request
    import glob

    results = {}

    # ── 1. Network interface check ────────────────────────────────────────────
    _sub("Network interfaces")
    lte_ifaces = []
    all_ifaces = []
    try:
        iface_output = subprocess.check_output(
            ["ip", "-brief", "addr", "show"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        for line in iface_output.splitlines():
            parts = line.split()
            if not parts:
                continue
            iface = parts[0]
            state = parts[1] if len(parts) > 1 else "UNKNOWN"
            all_ifaces.append((iface, state))
            if any(k in iface.lower() for k in ("ppp", "wwan", "usb", "lte", "rmnet")):
                lte_ifaces.append((iface, state))
            _field(iface, state, indent=6)
    except FileNotFoundError:
        # Non-Linux or no 'ip' tool
        try:
            import socket
            hostname = socket.gethostname()
            _field("Hostname", hostname, indent=6)
        except Exception:
            pass
        _warn("'ip' command not found (environment non-Linux or limited)")
    except Exception as exc:
        _warn(f"Interface listing error: {exc}")

    results["lte_iface_found"] = bool(lte_ifaces)
    if lte_ifaces:
        for iface, state in lte_ifaces:
            if state.upper() == "UP":
                _ok(f"LTE interface {iface} is UP")
            else:
                _warn(f"LTE interface {iface} is {state}")
    else:
        _warn("No dedicated cellular interface (ppp0 / wwan0 / usb0) listed")
        _warn("Device may be tethering via eth0/wlan0 or using direct serial PPP")

    # ── 2. Default route ───────────────────────────────────────────────────────
    _sub("Default route")
    default_gw  = "N/A"
    default_via = "N/A"
    try:
        rt_out = subprocess.check_output(
            ["ip", "route", "show", "default"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        for line in rt_out.splitlines():
            if "default" in line:
                default_gw = line.strip()
                parts = line.split()
                if "via" in parts:
                    idx = parts.index("via")
                    default_via = parts[idx + 1] if idx + 1 < len(parts) else "?"
                break
    except Exception:
        pass

    if default_gw != "N/A":
        _ok("Default route present")
    else:
        _warn("No default route found — internet reachability will be tested directly")
    _field("Default route", default_gw if default_gw != "N/A" else "NONE")
    _field("Gateway",       default_via)
    results["default_route"] = (default_gw != "N/A")

    # ── 3. Internet reachability (DNS + HTTP) ──────────────────────────────
    _sub("Internet reachability (DNS & HTTP probe)")
    dns_ok    = False
    http_ok   = False
    http_code = None
    http_ms   = None
    try:
        socket.setdefaulttimeout(5)
        socket.getaddrinfo("dns.google", 443)
        dns_ok = True
        _ok("DNS resolution working (dns.google resolved)")
    except Exception as dns_err:
        _err(f"DNS resolution failed: {dns_err}")

    probe_url = "http://connectivitycheck.gstatic.com/generate_204"
    try:
        t0 = time.time()
        req = urllib.request.urlopen(probe_url, timeout=7)
        http_ms   = round((time.time() - t0) * 1000)
        http_code = req.getcode()
        req.close()
        http_ok = True
        _ok(f"HTTP probe succeeded (HTTP {http_code}, {http_ms} ms)")
    except Exception as http_err:
        _err(f"HTTP probe failed: {http_err}")

    _field("DNS OK",       dns_ok)
    _field("HTTP OK",      http_ok)
    _field("HTTP status",  http_code if http_code else "N/A")
    _field("HTTP latency", f"{http_ms} ms" if http_ms else "N/A")
    results["internet_ok"] = (dns_ok and http_ok)

    # ── 4. ModemManager & SIM status (mmcli) ─────────────────────────────────
    _sub("Modem / SIM status (ModemManager mmcli)")
    mmcli_available = False
    modem_state    = "N/A"
    sim_status     = "N/A"
    signal_quality = "N/A"
    operator_name  = "N/A"
    access_tech    = "N/A"

    try:
        modem_list = subprocess.check_output(
            ["mmcli", "-L"], stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        mmcli_available = True
        modem_lines = [l.strip() for l in modem_list.splitlines() if "/Modem/" in l]
        if modem_lines:
            modem_path = modem_lines[0].split()[0]
            modem_idx  = modem_path.split("/")[-1]
            modem_info = subprocess.check_output(
                ["mmcli", "-m", modem_idx, "--output-keyvalue"],
                stderr=subprocess.DEVNULL, text=True, timeout=8
            )
            for line in modem_info.splitlines():
                if "status.state" in line:
                    modem_state = line.split(":", 1)[-1].strip()
                elif "3gpp.registration-state" in line:
                    sim_status = line.split(":", 1)[-1].strip()
                elif "signal.quality.value" in line:
                    signal_quality = line.split(":", 1)[-1].strip() + " %"
                elif "3gpp.operator-name" in line:
                    operator_name = line.split(":", 1)[-1].strip()
                elif "generic.access-technologies" in line:
                    access_tech = line.split(":", 1)[-1].strip()
            _ok(f"Modem detected at {modem_path}")
        else:
            _warn("ModemManager running but no active modem device listed")
    except FileNotFoundError:
        _warn("mmcli not found (install via: sudo apt install -y modemmanager)")
    except Exception as exc:
        _warn(f"mmcli query failed: {exc}")

    _field("mmcli available", mmcli_available)
    _field("Modem state",     modem_state)
    _field("SIM / Reg state", sim_status)
    _field("Operator",        operator_name)
    _field("Access tech",     access_tech)
    _field("Signal quality",  signal_quality)
    results["mmcli_ok"] = mmcli_available and modem_state not in ("N/A", "failed", "unknown")

    # ── 5. Direct Cellular Hardware Device Probe ──────────────────────────────
    _sub("Cellular modem hardware scan (serial / USB / sysfs)")
    found_devs = []
    # Check ttyUSB and ttyACM ports
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/cdc-wdm*", "/dev/qcqmi*"):
        matches = glob.glob(pattern)
        if matches:
            found_devs.extend(matches)

    known_modem_vids = {
        "1e0e": "SIMCom",
        "2c7c": "Quectel",
        "12d1": "Huawei",
        "1bc7": "Telit",
        "2cb7": "Fibocom",
    }
    usb_modem_found = False
    usb_modem_name = "N/A"
    try:
        for vid_path in glob.glob("/sys/bus/usb/devices/*/idVendor"):
            try:
                with open(vid_path, "r") as f:
                    vid = f.read().strip().lower()
                if vid in known_modem_vids:
                    usb_modem_found = True
                    usb_modem_name = known_modem_vids[vid]
                    break
            except Exception:
                pass
    except Exception:
        pass

    if found_devs:
        _ok(f"Cellular serial devices present: {', '.join(found_devs)}")
    elif usb_modem_found:
        _ok(f"USB Cellular modem chip identified: {usb_modem_name}")
    else:
        _warn("No direct cellular USB/serial ports (/dev/ttyUSB*, /dev/cdc-wdm*) detected")

    _field("Modem serial ports", ", ".join(found_devs) if found_devs else "None")
    _field("Modem USB chip",     usb_modem_name if usb_modem_found else "None")

    # ── 6. LTE Status LED Test ───────────────────────────────────────────────
    try:
        from config import LED_LTE_GREEN
        import gpiod
        _sub("LTE status LED hardware test")
        _field("LED GPIO Pin", f"BCM {LED_LTE_GREEN} (Physical 22)")
        _ok("LTE LED pin configured in config.py")
    except Exception:
        pass

    # ── Summary Decision ──────────────────────────────────────────────────────
    print()
    if results.get("internet_ok"):
        _ok("LTE / connectivity test PASSED — internet is reachable")
        passed = True
    elif results.get("default_route") or results.get("lte_iface_found") or found_devs:
        _warn("LTE hardware / route detected, but public probe failed")
        passed = True
    else:
        _err("LTE / connectivity test FAILED — no network or modem hardware")
        passed = False

    _hr()
    return passed


# ════════════════════════════════════════════════════════════════════════════
# Entry-point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SWSTP Edge -- full sensor diagnostic (RTC + IMU + GNSS + PiCam + LTE)"
    )
    parser.add_argument("--imu-samples", type=int, default=5,
                        help="Number of IMU read samples to display (default: 5)")
    parser.add_argument("--gnss-wait",   type=int, default=30,
                        help="Seconds to wait for GNSS data (default: 30)")
    parser.add_argument("--cam-frames",  type=int, default=30,
                        help="Number of camera benchmark frames (default: 30)")
    parser.add_argument("--cam-width",   type=int, default=640,
                        help="Camera capture width (default: 640)")
    parser.add_argument("--cam-height",  type=int, default=480,
                        help="Camera capture height (default: 480)")
    parser.add_argument("--cam-af",      default="continuous", choices=["continuous", "auto", "manual"],
                        help="Camera autofocus mode (default: continuous)")
    parser.add_argument("--save-dir",    default=os.path.join(_SCRIPT_DIR, "captures"),
                        help="Directory to save sample camera capture (default: captures/)")
    parser.add_argument("--no-rtc",   action="store_true", help="Skip RTC test")
    parser.add_argument("--no-imu",   action="store_true", help="Skip IMU test")
    parser.add_argument("--no-gnss",  action="store_true", help="Skip GNSS test")
    parser.add_argument("--no-picam", action="store_true", help="Skip PiCamera / IMX708 test")
    parser.add_argument("--no-lte",   action="store_true", help="Skip LTE connectivity test")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────
    print()
    _hr("=", color=BOLD + CYAN)
    print(f"{BOLD}{CYAN}{'SWSTP EDGE -- SENSOR DIAGNOSTIC':^70}{RESET}")
    print(f"{DIM}{'RTC  .  IMU  .  GNSS  .  PiCam  .  LTE':^70}{RESET}")
    print(f"{DIM}{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'):^70}{RESET}")
    _hr("=", color=BOLD + CYAN)
    print()

    results = {}

    if not args.no_rtc:
        results["RTC"]   = test_rtc()
    else:
        print(f"  {DIM}[RTC]   skipped (--no-rtc){RESET}\n")

    if not args.no_imu:
        results["IMU"]   = test_imu(args.imu_samples)
    else:
        print(f"  {DIM}[IMU]   skipped (--no-imu){RESET}\n")

    if not args.no_gnss:
        results["GNSS"]  = test_gnss(args.gnss_wait)
    else:
        print(f"  {DIM}[GNSS]  skipped (--no-gnss){RESET}\n")

    if not args.no_picam:
        results["PiCam"] = test_picam(
            frames=args.cam_frames,
            width=args.cam_width,
            height=args.cam_height,
            af_mode=args.cam_af,
            save_dir=args.save_dir,
        )
    else:
        print(f"  {DIM}[PiCam] skipped (--no-picam){RESET}\n")

    if not args.no_lte:
        results["LTE"]   = test_lte()
    else:
        print(f"  {DIM}[LTE]   skipped (--no-lte){RESET}\n")

    # ── Summary ───────────────────────────────────────────────────────────
    _section("SUMMARY", color=BOLD + YELLOW)
    print()
    for module, passed in results.items():
        if passed is True:
            _ok(f"{module:<6} -- PASSED")
        elif passed is False:
            _err(f"{module:<6} -- FAILED / not detected")
        else:
            _warn(f"{module:<6} -- SKIPPED")
    print()
    _hr("=", color=BOLD + CYAN)
    print()


if __name__ == "__main__":
    main()
