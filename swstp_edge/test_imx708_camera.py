#!/usr/bin/env python3
"""
test_imx708_camera.py — Interactive Calibration & Colour-Match Tool for the Sony IMX708
                         (Raspberry Pi Camera Module 3 Wide NoIR)

═══════════════════════════════════════════════════════════════════
  WHY RPICAM-HELLO LOOKS CORRECT BUT PYTHON/PICAMERA2 DOES NOT
═══════════════════════════════════════════════════════════════════
  'rpicam-hello' auto-detects the sensor model from the camera
  EEPROM and loads the MATCHING libcamera tuning JSON file, e.g.:

      imx708_wide_noir.json  (NoIR — no IR-cut filter)
      imx708_wide.json       (standard, with IR-cut filter)
      imx708_noir.json       (telephoto NoIR)

  This tuning file contains a CCM (Color Correction Matrix)
  calibrated specifically for that sensor variant.  Without it,
  libcamera's ISP applies a generic CCM, and you get a green/
  purple/magenta cast — especially noticeable outdoors.

  This tool:
    1. Sniffs which tuning file rpicam-hello actually loads.
    2. Captures a reference frame using rpicam-still (the same
       pipeline rpicam-hello uses) and shows it side-by-side with
       the live picamera2 frame so you can see the delta.
    3. Lets you interactively tweak AWB, ColourGains, CCM-bypass,
       sharpness, saturation etc. in real time.
    4. Exports the winning settings as a config.py snippet and as
       a JSON report with per-channel mean deltas.

Usage:
    python3 test_imx708_camera.py                 # GUI live preview
    python3 test_imx708_camera.py --compare        # Side-by-side vs rpicam-still
    python3 test_imx708_camera.py --headless       # No display — frame report
    python3 test_imx708_camera.py --list-tuning    # List tuning files on disk
    python3 test_imx708_camera.py --help

Keyboard controls (GUI mode):
    ─────────────────────────────────────────────────────────────────
    COLOUR MATCH  (the important ones for your use-case)
      Z       Clone rpicam-hello colours — sniffs the tuning file
              rpicam-hello actually loaded, applies it to picamera2,
              then captures a reference frame so you can compare.
      M       Open side-by-side comparison window (rpicam-still vs
              current picamera2 frame) and print colour delta report.

    AWB / COLOUR
      w / W   Cycle AWB mode (Auto→Incandescent→Tungsten→Fluorescent
                              →Indoor→Daylight→Cloudy→Custom)
      r / R   Increase / Decrease Red  colour gain (Custom AWB only)
      b / B   Increase / Decrease Blue colour gain (Custom AWB only)

    AUTOFOCUS
      a       Cycle AF mode  (Continuous PDAF → Auto → Manual)
      f / t   Trigger single-shot AF
      [ / ]   Manual lens position ±0.5 diopters (0=infinity, 15=close)

    EXPOSURE
      e       Toggle AEC  (Auto ↔ Manual)
      i / I   Shutter speed step up / down  (manual mode)
      g / G   Analogue gain step up / down  (manual mode)

    IMAGE QUALITY
      s / S   Sharpness  ±0.25
      c / C   Contrast   ±0.25
      u / U   Saturation ±0.25

    RESOLUTION / FPS
      p / P   Cycle resolution preset forward / backward
      + / -   FPS ±1  (requires camera restart)

    FEATURES
      h       Toggle RGB histogram overlay
      d       Toggle HUD overlay
      n       Toggle NoIR tuning file  (requires restart)
      H       Toggle hardware HDR      (requires restart)

    CAPTURE / EXPORT
      Space   Save PNG snapshot + JSON sidecar
      x       Export config.py snippet
      q / ESC Quit
    ─────────────────────────────────────────────────────────────────
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import glob
import threading

# ---------------------------------------------------------------------------
# cv2 import
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("[ERROR] OpenCV (cv2) not found.  Install: pip install opencv-python")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Picamera2 — try venv first, then system site-packages (apt install puts it there)
# ---------------------------------------------------------------------------
try:
    from picamera2 import Picamera2
    try:
        from libcamera import controls as libcam_controls
    except ImportError:
        libcam_controls = None
    _HAS_PICAMERA2 = True
except (ImportError, Exception):
    Picamera2 = None
    libcam_controls = None
    _HAS_PICAMERA2 = False
    try:
        _site_candidates = (
            glob.glob("/usr/lib/python3*/dist-packages")
            + glob.glob("/usr/local/lib/python3*/dist-packages")
        )
        for _sp in _site_candidates:
            if _sp not in sys.path:
                sys.path.insert(0, _sp)
        from picamera2 import Picamera2          # type: ignore
        try:
            from libcamera import controls as libcam_controls  # type: ignore
        except ImportError:
            libcam_controls = None
        _HAS_PICAMERA2 = True
        print("[CAMERA] picamera2 loaded from system site-packages.")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_OUT_DIR = os.path.join(_HERE, "captures", "calibration")
os.makedirs(CALIB_OUT_DIR, exist_ok=True)


# ===========================================================================
# rpicam-hello colour-matching helpers
# ===========================================================================

def sniff_rpicam_tuning_file(camera_idx: int = 0) -> dict:
    """
    Run 'rpicam-hello --list-cameras' (and optionally a 1-frame capture
    with LIBCAMERA_LOG_LEVELS=*:DEBUG) to discover:
      - Which tuning JSON file libcamera actually loaded for this sensor.
      - The sensor model string.
      - The pipeline (vc4 / pisp).

    Returns a dict:
      {
        "tuning_file":  "/path/to/imx708_wide_noir.json" or None,
        "sensor_model": "imx708",
        "pipeline":     "vc4",
        "source":       "env" | "debug_log" | "inferred" | "not_found",
        "raw_output":   "...",
      }
    """
    result = {
        "tuning_file":  None,
        "sensor_model": None,
        "pipeline":     None,
        "source":       "not_found",
        "raw_output":   "",
    }

    # ── Step 1: check env override (the most direct answer) ───────────────
    env_tf = os.environ.get("LIBCAMERA_RPI_TUNING_FILE", "")
    if env_tf and os.path.isfile(env_tf):
        result["tuning_file"] = env_tf
        result["source"]      = "env"
        print(f"[RPICAM-SNIFF] Tuning file from env: {env_tf}")

    # ── Step 2: run rpicam-hello --list-cameras to get sensor model ────────
    for cmd in ["rpicam-hello", "libcamera-hello"]:
        if not shutil.which(cmd):
            continue
        try:
            r = subprocess.run(
                [cmd, "--list-cameras"],
                capture_output=True, text=True, timeout=5.0
            )
            output = r.stdout + r.stderr
            result["raw_output"] += f"\n--- {cmd} --list-cameras ---\n" + output

            # Parse sensor model
            m = re.search(r"(imx\d+[\w_]*)", output, re.IGNORECASE)
            if m:
                result["sensor_model"] = m.group(1).lower()

            # Parse pipeline
            if "pisp" in output.lower():
                result["pipeline"] = "pisp"
            elif "vc4" in output.lower():
                result["pipeline"] = "vc4"
            break
        except Exception as e:
            print(f"[RPICAM-SNIFF] {cmd} error: {e}")

    # ── Step 3: run a 1-frame capture with debug logging to see tuning load ─
    if result["tuning_file"] is None:
        for cmd in ["rpicam-still", "libcamera-still", "rpicam-hello", "libcamera-hello"]:
            if not shutil.which(cmd):
                continue
            try:
                env = os.environ.copy()
                env["LIBCAMERA_LOG_LEVELS"] = "*:DEBUG"
                extra = ["-t", "1", "--nopreview", "-o", "/dev/null"]
                if "still" not in cmd:
                    extra = ["-t", "1", "--nopreview"]
                r = subprocess.run(
                    [cmd] + extra,
                    capture_output=True, text=True, timeout=10.0,
                    env=env,
                )
                dbg_out = r.stdout + r.stderr
                result["raw_output"] += f"\n--- {cmd} DEBUG ---\n" + dbg_out[:4000]

                # libcamera debug log line looks like:
                #   Loaded tuning data file '/usr/share/.../imx708_wide_noir.json'
                m = re.search(
                    r"(?:Loaded tuning data file|tuning file)[^'\"]*['\"]([^'\"]+\.json)['\"]" ,
                    dbg_out, re.IGNORECASE
                )
                if m:
                    tf = m.group(1)
                    if os.path.isfile(tf):
                        result["tuning_file"] = tf
                        result["source"]      = "debug_log"
                        print(f"[RPICAM-SNIFF] Tuning file from debug log: {tf}")
                        break
            except Exception as e:
                print(f"[RPICAM-SNIFF] debug capture error ({cmd}): {e}")

    # ── Step 4: infer from sensor model if still not found ─────────────────
    if result["tuning_file"] is None and result["sensor_model"]:
        search_dirs = [
            "/usr/share/libcamera/ipa/rpi/vc4",
            "/usr/share/libcamera/ipa/rpi/pisp",
            "/usr/share/libcamera/ipa/raspberrypi",
            "/usr/share/rpi-camera-assets",
        ]
        model = result["sensor_model"]
        # Priority order: wide_noir > noir > wide > plain
        candidates = [
            f"{model}_wide_noir.json",
            f"{model}_noir.json",
            f"{model}_wide.json",
            f"{model}.json",
        ]
        for d in search_dirs:
            for fname in candidates:
                p = os.path.join(d, fname)
                if os.path.isfile(p):
                    result["tuning_file"] = p
                    result["source"]      = "inferred"
                    print(f"[RPICAM-SNIFF] Tuning file inferred: {p}")
                    break
            if result["tuning_file"]:
                break

    return result


def capture_rpicam_reference_frame(
    width: int = 960,
    height: int = 540,
    timeout_ms: int = 3000,
) -> tuple:
    """
    Capture one frame using rpicam-still (same libcamera ISP pipeline as
    rpicam-hello) and return it as a BGR numpy array.

    Returns (frame_bgr, cmd_used, error_str).
    Frame is None on failure.
    """
    for cmd in ["rpicam-still", "libcamera-still"]:
        if not shutil.which(cmd):
            continue
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = f.name
        try:
            r = subprocess.run(
                [
                    cmd,
                    "--nopreview",
                    "-t", str(timeout_ms),
                    "--width",  str(width),
                    "--height", str(height),
                    "--encoding", "jpg",
                    "-o", tmp,
                    "--immediate",
                ],
                capture_output=True, text=True, timeout=timeout_ms / 1000 + 10,
            )
            if os.path.isfile(tmp) and os.path.getsize(tmp) > 1000:
                frame = cv2.imread(tmp)
                os.unlink(tmp)
                if frame is not None:
                    frame = cv2.resize(frame, (width, height))
                    return frame, cmd, ""
                return None, cmd, "cv2.imread returned None"
            os.unlink(tmp)
            return None, cmd, r.stderr[:500]
        except Exception as e:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            return None, cmd, str(e)
    return None, "none", "Neither rpicam-still nor libcamera-still found on PATH"


def analyse_colour_delta(ref_frame: np.ndarray, cur_frame: np.ndarray) -> dict:
    """
    Compute per-channel mean difference between reference (rpicam-still)
    and current (picamera2) frames.  Both must be BGR, same size.

    Returns dict with per-channel means and delta, plus a diagnostic
    string describing what the mismatch looks like.
    """
    if ref_frame.shape[:2] != cur_frame.shape[:2]:
        cur_frame = cv2.resize(cur_frame, (ref_frame.shape[1], ref_frame.shape[0]))

    ref_b, ref_g, ref_r = [float(np.mean(ref_frame[:, :, c])) for c in range(3)]
    cur_b, cur_g, cur_r = [float(np.mean(cur_frame[:, :, c])) for c in range(3)]

    db = cur_b - ref_b
    dg = cur_g - ref_g
    dr = cur_r - ref_r
    total = abs(db) + abs(dg) + abs(dr)

    # Human-readable diagnosis
    symptoms = []
    if dg > 15:
        symptoms.append("green cast (excess green)")
    elif dg < -15:
        symptoms.append("green deficit")
    if db > 15:
        symptoms.append("blue / cool cast")
    elif db < -15:
        symptoms.append("blue deficit / warm")
    if dr > 15:
        symptoms.append("red / warm cast")
    elif dr < -15:
        symptoms.append("red deficit / cool")
    if abs(dg) > 10 and abs(db) > 10 and dg * db > 0:
        symptoms.append("purple / magenta cast (green+blue both shifted)")
    if not symptoms:
        symptoms.append("colours match closely — delta OK")

    return {
        "ref_means":    {"R": round(ref_r, 1), "G": round(ref_g, 1), "B": round(ref_b, 1)},
        "cur_means":    {"R": round(cur_r, 1), "G": round(cur_g, 1), "B": round(cur_b, 1)},
        "delta":        {"dR": round(dr, 1),   "dG": round(dg, 1),   "dB": round(db, 1)},
        "total_delta":  round(total, 1),
        "diagnosis":    ", ".join(symptoms),
    }


def run_compare_mode(
    state,
    ctrl,
    width: int = 960,
    height: int = 540,
) -> dict:
    """
    Full colour-match comparison workflow:
      1. Sniff which tuning file rpicam-hello would use.
      2. Capture a reference frame via rpicam-still.
      3. Capture the current picamera2 frame.
      4. Show side-by-side and print a colour delta report.
      5. Return the sniff + delta info dict.
    """
    print("\n" + "=" * 65)
    print("  COLOUR MATCH COMPARISON  vs  rpicam-hello / rpicam-still")
    print("=" * 65)

    # 1. Sniff tuning file
    print("[COMPARE] Sniffing tuning file used by rpicam-hello...")
    sniff = sniff_rpicam_tuning_file()
    print(f"[COMPARE] Sensor model : {sniff['sensor_model']}")
    print(f"[COMPARE] Pipeline     : {sniff['pipeline']}")
    print(f"[COMPARE] Tuning file  : {sniff['tuning_file']}")
    print(f"[COMPARE] Source       : {sniff['source']}")

    # 2. Capture reference frame via rpicam-still
    print("[COMPARE] Capturing reference frame via rpicam-still...")
    ref_frame, cmd_used, err = capture_rpicam_reference_frame(width, height)
    if ref_frame is None:
        print(f"[COMPARE] WARNING: rpicam-still capture failed ({cmd_used}): {err}")
        print("[COMPARE] Skipping side-by-side — showing sniff info only.")
    else:
        print(f"[COMPARE] Reference frame captured via {cmd_used}.")

    # 3. Capture current picamera2 frame
    ok, cur_frame, meta = ctrl.read()
    if not ok or cur_frame is None:
        print("[COMPARE] Could not read current picamera2 frame.")
        cur_frame = None

    # 4. Analyse delta and show side-by-side
    delta_info = {}
    if ref_frame is not None and cur_frame is not None:
        delta_info = analyse_colour_delta(ref_frame, cur_frame)
        print("\n" + "-" * 55)
        print("  COLOUR DELTA  (picamera2 MINUS rpicam-still reference)")
        print("-" * 55)
        print(f"  rpicam-still : R={delta_info['ref_means']['R']:.1f}  "
              f"G={delta_info['ref_means']['G']:.1f}  "
              f"B={delta_info['ref_means']['B']:.1f}")
        print(f"  picamera2    : R={delta_info['cur_means']['R']:.1f}  "
              f"G={delta_info['cur_means']['G']:.1f}  "
              f"B={delta_info['cur_means']['B']:.1f}")
        print(f"  Delta        : dR={delta_info['delta']['dR']:+.1f}  "
              f"dG={delta_info['delta']['dG']:+.1f}  "
              f"dB={delta_info['delta']['dB']:+.1f}  "
              f"(total={delta_info['total_delta']:.1f})")
        print(f"  Diagnosis    : {delta_info['diagnosis']}")
        print("-" * 55 + "\n")

        # Build side-by-side image
        sep   = np.zeros((height, 4, 3), dtype=np.uint8)
        side  = np.hstack([ref_frame, sep, cur_frame])
        sw    = side.shape[1]
        cv2.putText(side, "rpicam-still  (REFERENCE — correct colours)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(side, "picamera2  (CURRENT — your Python output)",
                    (width + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        # Delta banner
        diag_short = delta_info['diagnosis'][:70]
        banner = f"Delta  dR={delta_info['delta']['dR']:+.1f}  dG={delta_info['delta']['dG']:+.1f}  dB={delta_info['delta']['dB']:+.1f}   {diag_short}"
        cv2.putText(side, banner,
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 80), 1, cv2.LINE_AA)

        win = "Colour Comparison  |  LEFT=rpicam-still  RIGHT=picamera2  |  any key to close"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(sw, 1920), min(height + 10, 1080))
        cv2.imshow(win, side)

        # Save
        ts   = _ts()
        cmp_path = os.path.join(CALIB_OUT_DIR, f"compare_{ts}.png")
        cv2.imwrite(cmp_path, side)
        print(f"[COMPARE] Side-by-side saved: {cmp_path}")

        cv2.waitKey(0)
        cv2.destroyWindow(win)

    # 5. Build report
    report = {
        "timestamp":    _ts(),
        "sniff":        sniff,
        "delta":        delta_info,
        "current_settings": state.to_dict() if state else {},
    }
    rpt_path = os.path.join(CALIB_OUT_DIR, f"colour_match_report_{_ts()}.json")
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[COMPARE] Report saved: {rpt_path}")
    return report


def clone_rpicam_colours(state, ctrl) -> str:
    """
    The key action for 'Z':
    1. Sniff the tuning file rpicam-hello uses.
    2. Set LIBCAMERA_RPI_TUNING_FILE to that path.
    3. Restart the camera so libcamera re-reads the tuning.
    4. Reset AWB to Auto (the tuning file handles CCM correctly).
    5. Clear any ColourGains override that might conflict.

    Returns a status string.
    """
    print("\n[CLONE] Sniffing rpicam-hello tuning file...")
    sniff = sniff_rpicam_tuning_file()
    tf    = sniff.get("tuning_file")

    if not tf:
        msg = ("[CLONE] Could not determine rpicam-hello tuning file.\n"
               "        Try running:  rpicam-hello --list-cameras\n"
               "        and then:    python3 test_imx708_camera.py --list-tuning")
        print(msg)
        return msg

    print(f"[CLONE] Applying tuning file: {tf}")
    os.environ["LIBCAMERA_RPI_TUNING_FILE"] = tf
    state.noir_tuning = True       # keep the flag consistent
    # Reset AWB to Auto — let the tuning file's CCM do the work
    state.awb_idx = 0              # Auto
    # IMPORTANT: do NOT lock ColourGains when using the tuning file
    # The tuning file contains the correct CCM + AWB calibration tables.
    # Forcing ColourGains on top of the CCM will shift colours WRONG.

    ctrl.restart()

    status = (f"[CLONE] Done.  Tuning: {os.path.basename(tf)}\n"
              f"        AWB = Auto  (tuning file CCM active)\n"
              f"        Press M to compare vs rpicam-still reference.")
    print(status)
    return status

# ---------------------------------------------------------------------------
# Resolution presets  (all 16:9 — preserves full 120 deg FOV)
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS = [
    (640,   360,  "640x360   Lite (360p)"),
    (960,   540,  "960x540   Standard (540p) <-- Production default"),
    (1280,  720,  "1280x720  HD (720p)"),
    (1920, 1080,  "1920x1080 Full HD (1080p)"),
    (2304, 1296,  "2304x1296 2K (binned)"),
    (2560, 1440,  "2560x1440 2.5K"),
    (3200, 1800,  "3200x1800 3.2K"),
    (4608, 2592,  "4608x2592 FULL SENSOR (max, ~5 fps)"),
]

# ---------------------------------------------------------------------------
# AWB mode presets
# ---------------------------------------------------------------------------
AWB_MODES = [
    ("Auto",         0),
    ("Incandescent", 1),
    ("Tungsten",     2),
    ("Fluorescent",  3),
    ("Indoor",       4),
    ("Daylight",     5),
    ("Cloudy",       6),
    ("Custom",       None),  # Manual ColourGains
]

_AWB_INT = {
    "Auto": 0, "Incandescent": 1, "Tungsten": 2,
    "Fluorescent": 3, "Indoor": 4, "Daylight": 5, "Cloudy": 6,
}

# ---------------------------------------------------------------------------
# Shutter speed presets (microseconds)
# ---------------------------------------------------------------------------
SHUTTER_PRESETS_US = [
    100, 200, 500, 1000, 2000, 4000, 8000, 15000,
    30000, 60000, 120000, 240000,
]

# ---------------------------------------------------------------------------
# Analogue gain presets  (x multiplier)
# ---------------------------------------------------------------------------
GAIN_PRESETS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _cycle(idx, lst, step=1):
    return (idx + step) % len(lst)


# ===========================================================================
# Calibration Session State
# ===========================================================================

class CalibState:
    """All tuneable parameters for the current calibration session."""

    def __init__(self):
        # Resolution / FPS
        self.res_idx     = 1          # default: 960x540
        self.fps         = 30.0
        # AWB
        self.awb_idx     = 5          # Daylight (best NoIR default)
        self.rg          = 2.1        # red  gain for Custom AWB
        self.bg          = 1.45       # blue gain for Custom AWB
        # Autofocus
        self.af_idx      = 0          # 0=Continuous, 1=Auto, 2=Manual
        self.lens_pos    = 0.0        # manual focus diopters (0 = infinity, 15 = very close)
        # Exposure
        self.aec         = True       # True = AEC on
        self.shutter_idx = 5          # index into SHUTTER_PRESETS_US (4000 us)
        self.gain_idx    = 0          # index into GAIN_PRESETS (1.0x)
        # Image quality
        self.sharpness   = 1.0
        self.contrast    = 1.0
        self.saturation  = 1.0
        # Features
        self.hdr         = False
        self.noir_tuning = True
        # Overlay
        self.show_hist   = True
        self.show_debug  = True

    @property
    def resolution(self):
        return RESOLUTION_PRESETS[self.res_idx][:2]

    @property
    def awb_name(self):
        return AWB_MODES[self.awb_idx][0]

    @property
    def af_name(self):
        return ["Continuous PDAF", "Auto (single-shot)", "Manual"][self.af_idx]

    def to_dict(self) -> dict:
        w, h = self.resolution
        return {
            "resolution":        f"{w}x{h}",
            "fps":               self.fps,
            "awb_mode":          self.awb_name,
            "colour_gains":      {"rg": round(self.rg, 3), "bg": round(self.bg, 3)},
            "af_mode":           self.af_name,
            "lens_pos_diopters": round(self.lens_pos, 2),
            "aec":               self.aec,
            "shutter_us":        SHUTTER_PRESETS_US[self.shutter_idx] if not self.aec else "auto",
            "analogue_gain":     GAIN_PRESETS[self.gain_idx] if not self.aec else "auto",
            "sharpness":         round(self.sharpness, 2),
            "contrast":          round(self.contrast, 2),
            "saturation":        round(self.saturation, 2),
            "hdr":               self.hdr,
            "noir_tuning":       self.noir_tuning,
        }

    def config_snippet(self) -> str:
        w, h   = self.resolution
        af_map = {"Continuous PDAF": "continuous", "Auto (single-shot)": "auto", "Manual": "manual"}
        af_str = af_map.get(self.af_name, "continuous")
        awb_int = _AWB_INT.get(self.awb_name, 0)
        lines = [
            "# Generated by test_imx708_camera.py  ----------------------------",
            f"CAMERA_WIDTH         = {w}",
            f"CAMERA_HEIGHT        = {h}",
            f"CAMERA_FPS           = {self.fps}",
            f"CAMERA_AF_MODE       = \"{af_str}\"",
            f"# AWB mode: {self.awb_name} (libcamera int enum {awb_int})",
            f"# ColourGains: Rg={self.rg:.3f}, Bg={self.bg:.3f}  (Custom AWB only)",
            f"# Sharpness={self.sharpness:.2f}  Contrast={self.contrast:.2f}  Saturation={self.saturation:.2f}",
            f"# HDR={self.hdr}",
            "# -----------------------------------------------------------------",
        ]
        return "\n".join(lines)


# ===========================================================================
# Camera Controller
# ===========================================================================

class IMX708CalibController:
    """
    Opens the Sony IMX708 via Picamera2 and provides live-update methods
    that apply CalibState parameters to the running camera without restarting.
    """

    TUNING_SEARCH_PATHS = [
        "/usr/share/libcamera/ipa/rpi/vc4",
        "/usr/share/libcamera/ipa/rpi/pisp",
        "/usr/share/libcamera/ipa/raspberrypi",
        "/usr/share/rpi-camera-assets",
    ]

    def __init__(self, state: CalibState, camera_idx: int = 0):
        self.state        = state
        self.camera_idx   = camera_idx
        self._picam2      = None
        self._frame_times = []

    # ── Tuning file discovery ────────────────────────────────────────────
    def _find_noir_tuning(self):
        for d in self.TUNING_SEARCH_PATHS:
            for fname in ("imx708_wide_noir.json", "imx708_noir.json"):
                p = os.path.join(d, fname)
                if os.path.isfile(p):
                    return p
        return None

    def find_all_tuning_files(self):
        found = []
        for d in self.TUNING_SEARCH_PATHS:
            found += sorted(glob.glob(os.path.join(d, "imx708*.json")))
        return found

    # ── Lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if not _HAS_PICAMERA2:
            raise RuntimeError("picamera2 not found. sudo apt install -y python3-picamera2")

        # HDR must be set before Picamera2 is constructed
        if self.state.hdr:
            try:
                from picamera2.devices.imx708 import IMX708 as _Dev
                with _Dev(self.camera_idx) as dev:
                    dev.set_sensor_hdr_mode(True)
                print("[IMX708] Hardware HDR enabled.")
            except Exception as e:
                print(f"[IMX708] HDR unavailable: {e}")

        # NoIR tuning file
        if self.state.noir_tuning:
            tf = self._find_noir_tuning()
            if tf:
                os.environ["LIBCAMERA_RPI_TUNING_FILE"] = tf
                print(f"[IMX708] NoIR tuning: {tf}")
            else:
                os.environ.pop("LIBCAMERA_RPI_TUNING_FILE", None)
                print("[IMX708] NoIR tuning file not found — libcamera auto-detect.")
        else:
            os.environ.pop("LIBCAMERA_RPI_TUNING_FILE", None)
            print("[IMX708] NoIR tuning override disabled.")

        w, h = self.state.resolution
        self._picam2 = Picamera2(self.camera_idx)

        print("\n[IMX708] Sensor properties:")
        props = self._picam2.camera_properties or {}
        for k, v in props.items():
            print(f"         {k}: {v}")

        cfg = self._picam2.create_video_configuration(
            main={"format": "BGR888", "size": (w, h)},
            raw={"size": (4608, 2592)},
            controls={"FrameRate": self.state.fps},
            queue=False,
        )
        self._picam2.configure(cfg)
        self._picam2.start()

        # Full-sensor ScalerCrop  ->  preserve 120 deg FOV
        try:
            pas = props.get("PixelArraySize", (4608, 2592))
            self._picam2.set_controls({"ScalerCrop": (0, 0, pas[0], pas[1])})
        except Exception:
            pass

        self.apply_all_controls()
        time.sleep(2.0)   # let AWB/AEC converge
        print(f"[IMX708] Camera ready: {w}x{h} @ {self.state.fps:.0f} fps\n")

    def stop(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None

    def restart(self) -> None:
        print("[IMX708] Restarting with new settings...")
        self.stop()
        time.sleep(0.5)
        self.start()

    # ── Live control push ────────────────────────────────────────────────
    def apply_all_controls(self) -> None:
        if self._picam2 is None:
            return
        ctrls = {}

        # AWB
        if self.state.awb_name == "Custom":
            ctrls["AwbEnable"]   = False
            ctrls["ColourGains"] = (float(self.state.rg), float(self.state.bg))
        else:
            awb_int = _AWB_INT.get(self.state.awb_name, 0)
            if libcam_controls and hasattr(libcam_controls, "AwbModeEnum"):
                _enum_map = {
                    0: libcam_controls.AwbModeEnum.Auto,
                    1: getattr(libcam_controls.AwbModeEnum, "Incandescent",
                               libcam_controls.AwbModeEnum.Tungsten),
                    2: libcam_controls.AwbModeEnum.Tungsten,
                    3: libcam_controls.AwbModeEnum.Fluorescent,
                    4: libcam_controls.AwbModeEnum.Indoor,
                    5: libcam_controls.AwbModeEnum.Daylight,
                    6: libcam_controls.AwbModeEnum.Cloudy,
                }
                ctrls["AwbMode"] = _enum_map.get(awb_int, libcam_controls.AwbModeEnum.Auto)
            else:
                ctrls["AwbMode"] = awb_int
            ctrls["AwbEnable"] = True

        # AEC / Exposure
        if self.state.aec:
            ctrls["AeEnable"] = True
        else:
            ctrls["AeEnable"]     = False
            ctrls["ExposureTime"] = SHUTTER_PRESETS_US[self.state.shutter_idx]
            ctrls["AnalogueGain"] = float(GAIN_PRESETS[self.state.gain_idx])

        # Autofocus
        af_idx = self.state.af_idx
        if libcam_controls and hasattr(libcam_controls, "AfModeEnum"):
            _af_map = [
                libcam_controls.AfModeEnum.Continuous,
                libcam_controls.AfModeEnum.Auto,
                libcam_controls.AfModeEnum.Manual,
            ]
            ctrls["AfMode"] = _af_map[af_idx]
        else:
            ctrls["AfMode"] = af_idx
        if af_idx == 2:
            ctrls["LensPosition"] = float(self.state.lens_pos)

        # Image quality
        ctrls["Sharpness"]  = float(self.state.sharpness)
        ctrls["Contrast"]   = float(self.state.contrast)
        ctrls["Saturation"] = float(self.state.saturation)

        try:
            self._picam2.set_controls(ctrls)
        except Exception as exc:
            print(f"[CALIB] set_controls error: {exc}")

    def trigger_af(self) -> None:
        if self._picam2 is None:
            return
        try:
            if libcam_controls and hasattr(libcam_controls, "AfTriggerEnum"):
                self._picam2.set_controls({"AfTrigger": libcam_controls.AfTriggerEnum.Start})
            else:
                self._picam2.set_controls({"AfTrigger": 0})
            print("[CALIB] AF trigger fired.")
        except Exception as exc:
            print(f"[CALIB] AF trigger error: {exc}")

    # ── Frame capture ────────────────────────────────────────────────────
    def read(self):
        """Return (ok, bgr_frame, metadata_dict)."""
        if self._picam2 is None:
            return False, None, {}
        try:
            frame = self._picam2.capture_array("main")
            meta  = {}
            try:
                req  = self._picam2.capture_request()
                meta = req.get_metadata() if req else {}
                req.release()
            except Exception:
                pass
            if frame is None or frame.size == 0:
                return False, None, {}
            now = time.monotonic()
            self._frame_times.append(now)
            self._frame_times = [t for t in self._frame_times if now - t <= 2.0]
            return True, frame, meta
        except Exception:
            return False, None, {}

    @property
    def measured_fps(self) -> float:
        n = len(self._frame_times)
        if n < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (n - 1) / span if span > 0 else 0.0


# ===========================================================================
# Histogram overlay
# ===========================================================================

def draw_histogram(frame: np.ndarray, x: int, y: int, w: int = 200, h: int = 80) -> None:
    """Draw an inset RGB histogram in the corner of `frame` (in-place)."""
    panel = frame.copy()
    cv2.rectangle(panel, (x - 3, y - 12), (x + w + 3, y + h + 3), (15, 15, 20), -1)
    cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "RGB Histogram", (x, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA)
    for ch, color in enumerate([(255, 60, 60), (60, 255, 60), (60, 120, 255)]):
        hist = cv2.calcHist([frame], [ch], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist, 0, h - 2, cv2.NORM_MINMAX)
        for i in range(1, 64):
            x1 = x + int((i - 1) * w / 64)
            x2 = x + int(i * w / 64)
            y1 = y + h - int(hist[i - 1])
            y2 = y + h - int(hist[i])
            cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


# ===========================================================================
# HUD Overlay
# ===========================================================================

def draw_hud(frame: np.ndarray, state: CalibState,
             ctrl: IMX708CalibController, meta: dict, save_count: int) -> None:
    fh, fw = frame.shape[:2]
    panel_w = 330

    # Semi-transparent left panel
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (panel_w, fh), (8, 10, 16), -1)
    cv2.addWeighted(panel, 0.58, frame, 0.42, 0, frame)
    cv2.line(frame, (panel_w, 0), (panel_w, fh), (40, 40, 50), 1)

    ROW_H  = 17
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    AA     = cv2.LINE_AA
    GRAY   = (150, 150, 150)
    DIVID  = (45, 45, 55)
    WHITE  = (220, 220, 220)

    def put(txt, row, color=WHITE, scale=0.40, bold=False):
        cv2.putText(frame, txt, (8, 20 + row * ROW_H),
                    FONT, scale, color, 2 if bold else 1, AA)

    def divider(row):
        y = 20 + row * ROW_H - 4
        cv2.line(frame, (4, y), (panel_w - 4, y), DIVID, 1)

    # ── Header ──────────────────────────────────────────────────────────
    put("IMX708  CALIBRATION  TOOL", 0, (0, 240, 190), 0.46, bold=True)
    divider(1)

    # ── Resolution / FPS ────────────────────────────────────────────────
    rw, rh = state.resolution
    put(f"[p/P]  Resolution : {rw}x{rh} @ {state.fps:.0f} fps", 1, (255, 220, 80))
    put(f"       Measured   : {ctrl.measured_fps:.1f} fps", 2, GRAY, 0.36)
    put(f"       Preset     : {RESOLUTION_PRESETS[state.res_idx][2]}", 3, GRAY, 0.33)
    divider(4)

    # ── AWB ─────────────────────────────────────────────────────────────
    awb_c = (100, 255, 140) if state.awb_name != "Custom" else (255, 190, 60)
    put(f"[w/W]  AWB Mode   : {state.awb_name}", 4, awb_c)
    rg_c = (120, 180, 255) if state.awb_name == "Custom" else GRAY
    bg_c = (120, 180, 255) if state.awb_name == "Custom" else GRAY
    put(f"[r/R]  Rg (red)   : {state.rg:.3f}", 5, rg_c)
    put(f"[b/B]  Bg (blue)  : {state.bg:.3f}", 6, bg_c)
    # Actual AWB gains from metadata
    col_gains = meta.get("ColourGains")
    if col_gains and len(col_gains) == 2:
        put(f"       Actual Rg/Bg: {col_gains[0]:.3f} / {col_gains[1]:.3f}", 7, GRAY, 0.33)
    divider(8)

    # ── Autofocus ────────────────────────────────────────────────────────
    af_c = (80, 240, 255)
    put(f"[a]    AF Mode    : {state.af_name}", 8, af_c)
    if state.af_idx == 2:
        put(f"[[ ]]  Lens Pos   : {state.lens_pos:.2f} diopters", 9, (220, 210, 100))
    else:
        put(f"       Lens Pos   : auto", 9, GRAY)
    af_state   = meta.get("AfState",  "-")
    af_paused  = meta.get("AfPauseState", "-")
    lens_actual = meta.get("LensPosition", "-")
    put(f"       AfState    : {af_state}  Lens: {lens_actual}", 10, GRAY, 0.33)
    divider(11)

    # ── Exposure ─────────────────────────────────────────────────────────
    aec_c = (160, 255, 160)
    put(f"[e]    AEC        : {'AUTO' if state.aec else 'MANUAL'}", 11, aec_c)
    if state.aec:
        exp_us = meta.get("ExposureTime", 0)
        again  = meta.get("AnalogueGain", 0.0)
        dg     = meta.get("DigitalGain", 1.0)
        put(f"       Shutter   : {exp_us/1000:.2f} ms (actual)", 12, GRAY, 0.35)
        put(f"       A.Gain    : x{again:.2f}  D.Gain: x{dg:.2f} (actual)", 13, GRAY, 0.33)
    else:
        put(f"[i/I]  Shutter   : {SHUTTER_PRESETS_US[state.shutter_idx]/1000:.2f} ms", 12, (255, 200, 90))
        put(f"[g/G]  A.Gain    : x{GAIN_PRESETS[state.gain_idx]:.1f}", 13, (255, 200, 90))
    divider(14)

    # ── Image Quality ────────────────────────────────────────────────────
    put(f"[s/S]  Sharpness  : {state.sharpness:.2f}", 14, (200, 170, 255))
    put(f"[c/C]  Contrast   : {state.contrast:.2f}", 15, (200, 170, 255))
    put(f"[u/U]  Saturation : {state.saturation:.2f}", 16, (200, 170, 255))
    divider(17)

    # ── Features ─────────────────────────────────────────────────────────
    def feat_c(on): return (60, 255, 100) if on else GRAY
    put(f"[H]    HDR        : {'ON' if state.hdr else 'OFF'}   (needs restart)", 17, feat_c(state.hdr))
    put(f"[n]    NoIR Tuning: {'ON' if state.noir_tuning else 'OFF'}   (needs restart)", 18, feat_c(state.noir_tuning))
    put(f"[h]    Histogram  : {'ON' if state.show_hist else 'OFF'}", 19, feat_c(state.show_hist))
    divider(20)

    # ── Frame Stats ──────────────────────────────────────────────────────
    b_m = float(np.mean(frame[:, :, 0]))
    g_m = float(np.mean(frame[:, :, 1]))
    r_m = float(np.mean(frame[:, :, 2]))
    bri = 0.114 * b_m + 0.587 * g_m + 0.299 * r_m
    put(f"       Brightness : {bri:.1f}", 20, WHITE)
    put(f"       R/G/B mean : {r_m:.0f} / {g_m:.0f} / {b_m:.0f}", 21, GRAY, 0.34)
    temp = meta.get("SensorTemperature")
    if temp is not None:
        put(f"       Sensor Temp: {temp:.1f} C", 22, GRAY, 0.34)
    divider(23)

    # ── Capture / controls ───────────────────────────────────────────────
    put(f"[Spc]  Saves      : {save_count}", 23, (80, 255, 200))
    put("[x]    Export config snippet", 24, (80, 255, 200))
    put("[q]    Quit", 25, (255, 90, 90))


# ===========================================================================
# Snapshot save
# ===========================================================================

def save_snapshot(frame: np.ndarray, state: CalibState,
                  meta: dict, ctrl: IMX708CalibController) -> str:
    ts   = _ts()
    base = os.path.join(CALIB_OUT_DIR, f"calib_{ts}")
    png  = base + ".png"
    jsn  = base + ".json"

    cv2.imwrite(png, frame)

    safe_meta = {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in meta.items()
    }
    info = {
        "timestamp":    ts,
        "calibration":  state.to_dict(),
        "camera_meta":  safe_meta,
        "measured_fps": round(ctrl.measured_fps, 2),
        "image_file":   os.path.basename(png),
    }
    with open(jsn, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"[SAVE]  PNG:     {png}")
    print(f"[SAVE]  Sidecar: {jsn}")
    return png


# ===========================================================================
# Argument parser
# ===========================================================================

def build_args():
    p = argparse.ArgumentParser(
        description="IMX708 Camera Module 3 — Interactive Calibration & Test Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--headless", action="store_true",
                   help="Run without GUI — capture N frames and save a report")
    p.add_argument("--frames", type=int, default=30,
                   help="Number of frames to capture in headless mode (default: 30)")
    p.add_argument("--camera-idx", type=int, default=0,
                   help="Picamera2 camera index (default: 0)")
    p.add_argument("--res", default="960x540",
                   help="Starting resolution WxH (default: 960x540)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Target frame rate (default: 30.0)")
    p.add_argument("--awb", default="Daylight",
                   choices=[m[0] for m in AWB_MODES],
                   help="Starting AWB mode (default: Daylight)")
    p.add_argument("--af-mode", default="continuous",
                   choices=["continuous", "auto", "manual"],
                   help="Starting autofocus mode (default: continuous)")
    p.add_argument("--no-noir-tuning", action="store_true",
                   help="Disable NoIR tuning file override")
    p.add_argument("--hdr", action="store_true",
                   help="Enable hardware HDR at startup")
    p.add_argument("--list-tuning", action="store_true",
                   help="List all available IMX708 tuning files and exit")
    p.add_argument("--compare", action="store_true",
                   help="Run the colour-match comparison workflow and exit "
                        "(captures reference via rpicam-still, shows side-by-side, "
                        "prints colour delta report)")
    p.add_argument("--sniff-tuning", action="store_true",
                   help="Sniff the tuning file rpicam-hello would use and exit")
    return p.parse_args()


# ===========================================================================
# GUI run loop
# ===========================================================================

def run_gui(args) -> None:
    state = CalibState()

    # Apply CLI args
    if "x" in args.res.lower():
        try:
            rw, rh = map(int, args.res.lower().split("x"))
            for i, (pw, ph, _) in enumerate(RESOLUTION_PRESETS):
                if pw == rw and ph == rh:
                    state.res_idx = i
                    break
        except Exception:
            pass
    state.fps         = args.fps
    state.awb_idx     = next((i for i, (n, _) in enumerate(AWB_MODES) if n == args.awb), 5)
    state.af_idx      = {"continuous": 0, "auto": 1, "manual": 2}.get(args.af_mode, 0)
    state.noir_tuning = not args.no_noir_tuning
    state.hdr         = args.hdr

    ctrl        = IMX708CalibController(state, camera_idx=args.camera_idx)
    save_count  = 0
    _clone_msg  = ""    # status text shown after Z action
    WIN         = ("IMX708 Calibration  |  [q] Quit  [Space] Save  "
                   "[Z] Clone rpicam colours  [M] Compare  [w] AWB  [a] AF  [e] Exp  [p] Res")

    print("\n" + "=" * 65)
    print("  IMX708 Camera Module 3  --  Calibration & Test Tool")
    print("=" * 65)

    try:
        ctrl.start()
    except Exception as exc:
        print(f"\n[ERROR] Could not open camera: {exc}")
        print("  Ensure picamera2 is installed:  sudo apt install -y python3-picamera2")
        sys.exit(1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    need_restart = False

    try:
        while True:
            if need_restart:
                ctrl.restart()
                need_restart = False

            ok, frame, meta = ctrl.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            disp = frame.copy()

            if state.show_hist:
                fh, fw = disp.shape[:2]
                draw_histogram(disp, fw - 224, fh - 96)

            if state.show_debug:
                draw_hud(disp, state, ctrl, meta, save_count)
                # Show clone status message for 4 s
                if _clone_msg:
                    lines = _clone_msg.strip().split("\n")
                    fh2, fw2 = disp.shape[:2]
                    for li, ln in enumerate(lines[:4]):
                        cv2.putText(disp, ln, (340, fh2 - 60 + li * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                    (0, 255, 180), 1, cv2.LINE_AA)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF

            # ── Quit ───────────────────────────────────────────────────────
            if key in (ord('q'), 27):
                break

            # ── AWB ────────────────────────────────────────────────────────
            elif key == ord('w'):
                state.awb_idx = _cycle(state.awb_idx, AWB_MODES)
                ctrl.apply_all_controls()
                print(f"[AWB] -> {state.awb_name}")
            elif key == ord('W'):
                state.awb_idx = _cycle(state.awb_idx, AWB_MODES, -1)
                ctrl.apply_all_controls()
                print(f"[AWB] -> {state.awb_name}")
            elif key == ord('r'):
                state.rg = _clamp(round(state.rg + 0.05, 3), 0.1, 8.0)
                if state.awb_name == "Custom":
                    ctrl.apply_all_controls()
                print(f"[Rg] -> {state.rg:.3f}")
            elif key == ord('R'):
                state.rg = _clamp(round(state.rg - 0.05, 3), 0.1, 8.0)
                if state.awb_name == "Custom":
                    ctrl.apply_all_controls()
                print(f"[Rg] -> {state.rg:.3f}")
            elif key == ord('b'):
                state.bg = _clamp(round(state.bg + 0.05, 3), 0.1, 8.0)
                if state.awb_name == "Custom":
                    ctrl.apply_all_controls()
                print(f"[Bg] -> {state.bg:.3f}")
            elif key == ord('B'):
                state.bg = _clamp(round(state.bg - 0.05, 3), 0.1, 8.0)
                if state.awb_name == "Custom":
                    ctrl.apply_all_controls()
                print(f"[Bg] -> {state.bg:.3f}")

            # ── Autofocus ──────────────────────────────────────────────────
            elif key == ord('a'):
                state.af_idx = _cycle(state.af_idx, [0, 1, 2])
                ctrl.apply_all_controls()
                print(f"[AF] -> {state.af_name}")
            elif key in (ord('f'), ord('t')):
                ctrl.trigger_af()
            elif key == ord('['):
                state.lens_pos = _clamp(round(state.lens_pos - 0.5, 2), 0.0, 15.0)
                if state.af_idx == 2:
                    ctrl.apply_all_controls()
                print(f"[Lens] -> {state.lens_pos:.2f} diopters")
            elif key == ord(']'):
                state.lens_pos = _clamp(round(state.lens_pos + 0.5, 2), 0.0, 15.0)
                if state.af_idx == 2:
                    ctrl.apply_all_controls()
                print(f"[Lens] -> {state.lens_pos:.2f} diopters")

            # ── Exposure ───────────────────────────────────────────────────
            elif key == ord('e'):
                state.aec = not state.aec
                ctrl.apply_all_controls()
                print(f"[AEC] -> {'ON' if state.aec else 'OFF'}")
            elif key == ord('i'):
                state.shutter_idx = min(state.shutter_idx + 1, len(SHUTTER_PRESETS_US) - 1)
                if not state.aec:
                    ctrl.apply_all_controls()
                print(f"[Shutter] -> {SHUTTER_PRESETS_US[state.shutter_idx]} us")
            elif key == ord('I'):
                state.shutter_idx = max(state.shutter_idx - 1, 0)
                if not state.aec:
                    ctrl.apply_all_controls()
                print(f"[Shutter] -> {SHUTTER_PRESETS_US[state.shutter_idx]} us")
            elif key == ord('g'):
                state.gain_idx = min(state.gain_idx + 1, len(GAIN_PRESETS) - 1)
                if not state.aec:
                    ctrl.apply_all_controls()
                print(f"[Gain] -> x{GAIN_PRESETS[state.gain_idx]:.1f}")
            elif key == ord('G'):
                state.gain_idx = max(state.gain_idx - 1, 0)
                if not state.aec:
                    ctrl.apply_all_controls()
                print(f"[Gain] -> x{GAIN_PRESETS[state.gain_idx]:.1f}")

            # ── Image Quality ──────────────────────────────────────────────
            elif key == ord('s'):
                state.sharpness = _clamp(round(state.sharpness + 0.25, 2), 0.0, 16.0)
                ctrl.apply_all_controls()
                print(f"[Sharpness] -> {state.sharpness:.2f}")
            elif key == ord('S'):
                state.sharpness = _clamp(round(state.sharpness - 0.25, 2), 0.0, 16.0)
                ctrl.apply_all_controls()
                print(f"[Sharpness] -> {state.sharpness:.2f}")
            elif key == ord('c'):
                state.contrast = _clamp(round(state.contrast + 0.25, 2), 0.0, 32.0)
                ctrl.apply_all_controls()
                print(f"[Contrast] -> {state.contrast:.2f}")
            elif key == ord('C'):
                state.contrast = _clamp(round(state.contrast - 0.25, 2), 0.0, 32.0)
                ctrl.apply_all_controls()
                print(f"[Contrast] -> {state.contrast:.2f}")
            elif key == ord('u'):
                state.saturation = _clamp(round(state.saturation + 0.25, 2), 0.0, 32.0)
                ctrl.apply_all_controls()
                print(f"[Saturation] -> {state.saturation:.2f}")
            elif key == ord('U'):
                state.saturation = _clamp(round(state.saturation - 0.25, 2), 0.0, 32.0)
                ctrl.apply_all_controls()
                print(f"[Saturation] -> {state.saturation:.2f}")

            # ── Resolution / FPS ───────────────────────────────────────────
            elif key == ord('p'):
                state.res_idx = _cycle(state.res_idx, RESOLUTION_PRESETS)
                rw2, rh2 = state.resolution
                print(f"[RES] -> {rw2}x{rh2}  (restarting...)")
                need_restart = True
            elif key == ord('P'):
                state.res_idx = _cycle(state.res_idx, RESOLUTION_PRESETS, -1)
                rw2, rh2 = state.resolution
                print(f"[RES] -> {rw2}x{rh2}  (restarting...)")
                need_restart = True
            elif key in (ord('+'), ord('=')):
                state.fps = _clamp(state.fps + 1.0, 1.0, 120.0)
                print(f"[FPS] -> {state.fps:.0f}  (restarting...)")
                need_restart = True
            elif key == ord('-'):
                state.fps = _clamp(state.fps - 1.0, 1.0, 120.0)
                print(f"[FPS] -> {state.fps:.0f}  (restarting...)")
                need_restart = True

            # ── Features ───────────────────────────────────────────────────
            elif key == ord('h'):
                state.show_hist = not state.show_hist
            elif key == ord('d'):
                state.show_debug = not state.show_debug
            elif key == ord('H'):
                state.hdr = not state.hdr
                print(f"[HDR] -> {'ON' if state.hdr else 'OFF'}  (restarting...)")
                need_restart = True
            elif key == ord('n'):
                state.noir_tuning = not state.noir_tuning
                print(f"[NoIR] -> {'ON' if state.noir_tuning else 'OFF'}  (restarting...)")
                need_restart = True

            # ── Colour-match actions ────────────────────────────────────────
            elif key == ord('Z'):
                # Clone rpicam-hello colours into the live picamera2 session
                _clone_msg = clone_rpicam_colours(state, ctrl)
                need_restart = False  # restart already done inside clone fn

            elif key == ord('M'):
                # Side-by-side comparison vs rpicam-still
                ok2, cur_f, meta2 = ctrl.read()
                if ok2 and cur_f is not None:
                    run_compare_mode(state, ctrl, *state.resolution)
                else:
                    print("[COMPARE] Could not read current frame.")

            # ── Capture / Export ───────────────────────────────────────────
            elif key == ord(' '):
                save_snapshot(disp, state, meta, ctrl)
                save_count += 1
            elif key == ord('x'):
                snippet = state.config_snippet()
                print("\n" + "=" * 60)
                print("  config.py SNIPPET:")
                print("=" * 60)
                print(snippet)
                print("=" * 60 + "\n")
                sp = os.path.join(CALIB_OUT_DIR, f"config_snippet_{_ts()}.txt")
                with open(sp, "w") as f:
                    f.write(snippet)
                print(f"[EXPORT] Saved: {sp}")

    finally:
        ctrl.stop()
        cv2.destroyAllWindows()
        print(f"\n[DONE]  {save_count} snapshot(s) saved to: {CALIB_OUT_DIR}")


# ===========================================================================
# Headless run
# ===========================================================================

def run_headless(args) -> None:
    state = CalibState()
    state.noir_tuning = not args.no_noir_tuning
    state.hdr         = args.hdr
    state.fps         = args.fps
    state.awb_idx     = next((i for i, (n, _) in enumerate(AWB_MODES) if n == args.awb), 5)
    state.af_idx      = {"continuous": 0, "auto": 1, "manual": 2}.get(args.af_mode, 0)

    if "x" in args.res.lower():
        try:
            rw, rh = map(int, args.res.lower().split("x"))
            for i, (pw, ph, _) in enumerate(RESOLUTION_PRESETS):
                if pw == rw and ph == rh:
                    state.res_idx = i
                    break
        except Exception:
            pass

    ctrl = IMX708CalibController(state, camera_idx=args.camera_idx)
    print("\n" + "=" * 65)
    print("  IMX708 Headless Calibration Run")
    print("=" * 65)

    try:
        ctrl.start()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    captured     = 0
    frame_data   = []
    last_frame   = None
    last_meta    = {}
    t0           = time.monotonic()

    print(f"[RUN] Capturing {args.frames} frames...")
    while captured < args.frames:
        ok, frame, meta = ctrl.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        captured  += 1
        last_frame = frame
        last_meta  = meta
        b = float(np.mean(frame[:, :, 0]))
        g = float(np.mean(frame[:, :, 1]))
        r = float(np.mean(frame[:, :, 2]))
        bri = 0.114 * b + 0.587 * g + 0.299 * r
        frame_data.append({
            "frame":          captured,
            "brightness":     round(bri, 2),
            "r_mean":         round(r, 2),
            "g_mean":         round(g, 2),
            "b_mean":         round(b, 2),
            "exposure_us":    meta.get("ExposureTime"),
            "analogue_gain":  meta.get("AnalogueGain"),
        })
        if captured % 5 == 0 or captured == args.frames:
            print(f"  [{captured:3d}/{args.frames}]  bri={bri:.1f}  "
                  f"R={r:.0f} G={g:.0f} B={b:.0f}  fps={ctrl.measured_fps:.1f}")

    elapsed = time.monotonic() - t0
    snap_path = "N/A"
    if last_frame is not None:
        snap_path = save_snapshot(last_frame, state, last_meta, ctrl)

    avg_bri = sum(f["brightness"] for f in frame_data) / len(frame_data) if frame_data else 0
    report = {
        "timestamp":      _ts(),
        "settings":       state.to_dict(),
        "total_frames":   captured,
        "elapsed_sec":    round(elapsed, 2),
        "avg_fps":        round(captured / elapsed, 2),
        "avg_brightness": round(avg_bri, 2),
        "final_snapshot": os.path.basename(snap_path),
        "frames":         frame_data,
    }
    rpt = os.path.join(CALIB_OUT_DIR, f"headless_report_{_ts()}.json")
    with open(rpt, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    ctrl.stop()
    print(f"\n[DONE]  {captured} frames in {elapsed:.1f}s  ({report['avg_fps']:.1f} fps)")
    print(f"[DONE]  Report: {rpt}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    if not _HAS_PICAMERA2:
        print("\n[ERROR] picamera2 is not importable.")
        print("  Fix:  sudo apt install -y python3-picamera2")
        print("        python3 -m venv venv --system-site-packages")
        sys.exit(1)

    args = build_args()

    # ── --list-tuning ───────────────────────────────────────────────────────
    if args.list_tuning:
        ctrl = IMX708CalibController(CalibState())
        files = ctrl.find_all_tuning_files()
        print("\nAvailable IMX708 libcamera tuning files:")
        if files:
            for f in files:
                print(f"  {f}")
        else:
            print("  None found in standard search paths.")
        sys.exit(0)

    # ── --sniff-tuning ──────────────────────────────────────────────────────
    if args.sniff_tuning:
        info = sniff_rpicam_tuning_file()
        print("\nrpicam-hello tuning file sniff result:")
        print(f"  Sensor model : {info['sensor_model']}")
        print(f"  Pipeline     : {info['pipeline']}")
        print(f"  Tuning file  : {info['tuning_file']}")
        print(f"  Source       : {info['source']}")
        print()
        if info["tuning_file"]:
            print("  To use this in your Python app, add to config.py / camera.py:")
            print(f"    os.environ['LIBCAMERA_RPI_TUNING_FILE'] = '{info['tuning_file']}'")
            print("  Or run the main app with:")
            print(f"    LIBCAMERA_RPI_TUNING_FILE='{info['tuning_file']}' python3 main.py --headless")
        sys.exit(0)

    # ── --compare ───────────────────────────────────────────────────────────
    if args.compare:
        state = CalibState()
        ctrl  = IMX708CalibController(state)
        try:
            ctrl.start()
            time.sleep(1.0)
            run_compare_mode(state, ctrl)
        finally:
            ctrl.stop()
        sys.exit(0)

    # ── normal modes ────────────────────────────────────────────────────────
    if args.headless:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()

