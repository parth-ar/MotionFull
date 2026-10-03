"""
camera.py — Unified Camera Abstraction for SWSTP Edge Gateway.

Native support for:
  1. Raspberry Pi Camera Module 3 (Sony IMX708) via Picamera2:
     - Phase Detection Autofocus (PDAF) continuous auto-focus
     - Hardware ISP debayering, auto-exposure, and AWB
     - Zero-overhead BGR888 NumPy frames directly into OpenCV
     - Hardware HDR mode option
  2. Fallback to GStreamer libcamerasrc pipeline (OpenCV libcamera bridge)
  3. Standard USB Webcams via OpenCV V4L2 / DirectShow
  4. Prerecorded MP4/AVI video files for simulation/testing

Exposes an interface 100% compatible with cv2.VideoCapture so the rest of the
application (motion detection, YOLO litter engine, live streamer) operates seamlessly.
"""

import os
import sys
import time
import glob
import subprocess
from typing import Optional, Tuple, Dict, Any, Union

# Attempt to import OpenCV; on minimal headless test environments it may be absent
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    np = None
    _HAS_CV2 = False

# Attempt to import Picamera2 and libcamera
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


# ===========================================================================
# Hardware & Sensor Detection Helpers
# ===========================================================================

def is_picamera2_available() -> bool:
    """Check if the picamera2 Python package is installed and importable."""
    return _HAS_PICAMERA2


def detect_imx708_sensor() -> Dict[str, Any]:
    """
    Detect whether a Sony IMX708 (Raspberry Pi Camera Module 3) is connected.
    Checks Picamera2 global camera info, Linux I2C drivers, and rpicam/libcamera CLI.
    """
    detection = {
        "detected": False,
        "model": "Unknown",
        "camera_num": 0,
        "details": "",
        "backend": None,
    }

    # 1. Picamera2 global camera enumeration (most reliable)
    if _HAS_PICAMERA2:
        try:
            cam_list = Picamera2.global_camera_info()
            if cam_list:
                for idx, cam in enumerate(cam_list):
                    model_str = str(cam.get("Model", "")).lower()
                    if "imx708" in model_str:
                        variant = "Camera Module 3"
                        if "wide" in model_str:
                            variant += " (Wide FOV)"
                        if "noir" in model_str:
                            variant += " (NoIR)"
                        detection.update({
                            "detected": True,
                            "model": f"Sony IMX708 [{variant}]",
                            "camera_num": idx,
                            "details": f"Picamera2 ID: {cam.get('Id', idx)}, Model: {cam.get('Model')}",
                            "backend": "picamera2",
                        })
                        return detection
                # If cameras exist but none explicitly called imx708, use camera 0 if available
                first_cam = cam_list[0]
                detection.update({
                    "detected": True,
                    "model": f"CSI Camera ({first_cam.get('Model', 'Unknown')})",
                    "camera_num": 0,
                    "details": str(first_cam),
                    "backend": "picamera2",
                })
                return detection
        except Exception as exc:
            detection["details"] = f"Picamera2 global_camera_info exception: {exc}"

    # 2. Check Linux kernel I2C sysfs for the imx708 driver
    if sys.platform.startswith("linux"):
        imx_driver_paths = glob.glob("/sys/bus/i2c/drivers/imx708/*-*")
        if imx_driver_paths:
            detection.update({
                "detected": True,
                "model": "Sony IMX708 (Camera Module 3)",
                "camera_num": 0,
                "details": f"Kernel driver active: {imx_driver_paths[0]}",
                "backend": "picamera2" if _HAS_PICAMERA2 else "libcamerasrc",
            })
            return detection

        # 3. Quick check via rpicam-hello / libcamera-hello CLI tool
        for cmd in ["rpicam-hello", "libcamera-hello"]:
            try:
                res = subprocess.run(
                    [cmd, "--list-cameras"],
                    capture_output=True,
                    text=True,
                    timeout=2.0
                )
                output = (res.stdout + res.stderr).lower()
                if "imx708" in output:
                    detection.update({
                        "detected": True,
                        "model": "Sony IMX708 (Camera Module 3)",
                        "camera_num": 0,
                        "details": f"Detected via {cmd} --list-cameras",
                        "backend": "picamera2" if _HAS_PICAMERA2 else "libcamerasrc",
                    })
                    return detection
            except Exception:
                pass

    return detection


def is_unicam_device(dev_idx: int) -> bool:
    """Check if /dev/video{idx} belongs to Unicam (CSI interface) rather than USB."""
    if not sys.platform.startswith("linux"):
        return False
    name_file = f"/sys/class/video4linux/video{dev_idx}/name"
    if os.path.exists(name_file):
        try:
            with open(name_file, "r", encoding="utf-8", errors="ignore") as f:
                name = f.read().strip().lower()
            return "unicam" in name or "bcm2835-unicam" in name or "rp1-cfe" in name
        except Exception:
            pass
    return False


def is_v4l2_capture_device(dev_idx: int) -> bool:
    """Filter out non-capture nodes (codecs, ISPs, metadata)."""
    sys_path = f"/sys/class/video4linux/video{dev_idx}"
    if not os.path.exists(sys_path):
        return False
    name_file = os.path.join(sys_path, "name")
    if os.path.exists(name_file):
        try:
            with open(name_file, "r", encoding="utf-8", errors="ignore") as f:
                name = f.read().strip().lower()
            if "metadata" in name or "bcm2835-codec" in name or "bcm2835-isp" in name:
                return False
        except Exception:
            pass
    return True


# ===========================================================================
# Picamera2 / IMX708 Capture Wrapper (cv2.VideoCapture drop-in)
# ===========================================================================

class IMX708CameraCapture:
    """
    High-performance drop-in wrapper around Picamera2 providing full
    cv2.VideoCapture compatibility for the Sony IMX708 (Camera Module 3).
    """

    def __init__(
        self,
        camera_idx: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        af_mode: str = "continuous",
        hdr: bool = False,
    ):
        self.camera_idx = camera_idx
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.af_mode = af_mode
        self.hdr = bool(hdr)

        self._picam2: Optional[Any] = None
        self._is_opened = False
        self._sensor_model = "Sony IMX708"
        self._props: Dict[str, Any] = {}

        if not _HAS_PICAMERA2:
            raise RuntimeError(
                "picamera2 is not installed. To use the Raspberry Pi Camera Module 3 (IMX708), "
                "run: sudo apt install -y python3-picamera2"
            )

        self._init_camera()

    # Known libcamera IPA tuning file search paths (Raspberry Pi OS Bookworm / Bullseye)
    _TUNING_SEARCH_PATHS = [
        "/usr/share/libcamera/ipa/rpi/vc4",       # Pi 4 / Bullseye + Bookworm
        "/usr/share/libcamera/ipa/rpi/pisp",      # Pi 5 (PISP ISP)
        "/usr/share/rpi-camera-assets",           # Legacy path
    ]

    # Tuning file names in preference order for IMX708 Wide NoIR
    _TUNING_CANDIDATES_NOIR_WIDE = [
        "imx708_wide_noir.json",   # Exact match — Wide + NoIR
        "imx708_noir.json",        # NoIR without explicit wide label
        "imx708_wide.json",        # Wide with IR-cut (better than generic)
        "imx708.json",             # Generic fallback
    ]

    def _find_tuning_file(self, model_str: str) -> Optional[str]:
        """
        Locate the best-matching libcamera IPA tuning JSON for the detected
        sensor model string.  Returns the full path or None if not found.
        """
        model_lower = model_str.lower()
        is_wide = "wide" in model_lower
        is_noir = "noir" in model_lower

        if "imx708" in model_lower:
            if is_wide and is_noir:
                candidates = self._TUNING_CANDIDATES_NOIR_WIDE
            elif is_noir:
                candidates = ["imx708_noir.json", "imx708.json"]
            elif is_wide:
                candidates = ["imx708_wide_noir.json", "imx708_wide.json", "imx708.json"]
            else:
                candidates = ["imx708.json"]
        else:
            return None

        for search_dir in self._TUNING_SEARCH_PATHS:
            for fname in candidates:
                full = os.path.join(search_dir, fname)
                if os.path.isfile(full):
                    return full
        return None

    def _init_camera(self) -> None:
        try:
            # Enable hardware HDR if requested and supported
            if self.hdr:
                try:
                    from picamera2.devices.imx708 import IMX708 as _IMX708Device
                    with _IMX708Device(self.camera_idx) as dev:
                        dev.set_sensor_hdr_mode(True)
                    print("[IMX708] Hardware HDR enabled.")
                except Exception as hdr_err:
                    print(f"[IMX708] HDR setting skipped: {hdr_err}")

            # ── Tuning file — IMX708 Wide NoIR colour correction ──────────────
            # The NoIR camera has no infrared-cut filter.  Without its specific
            # tuning file (imx708_wide_noir.json), libcamera applies the standard
            # imx708.json CCM which is calibrated for the IR-cut variant.
            # Result: red objects appear purple because the CCM over-corrects.
            #
            # Strategy — two independent layers, both applied:
            #   1. LIBCAMERA_RPI_TUNING_FILE env var (libcamera reads this
            #      before Picamera2 is even created — cannot be ignored).
            #   2. ColourGains manual override (applied after start) as a
            #      belt-and-suspenders correction if tuning still drifts.

            tuning_path: Optional[str] = None
            _is_noir = False
            try:
                cam_list = Picamera2.global_camera_info()
                if cam_list and self.camera_idx < len(cam_list):
                    raw_model = cam_list[self.camera_idx].get("Model", "")
                    print(f"[IMX708] Detected sensor model: '{raw_model}'")
                    _is_noir = "noir" in raw_model.lower()
                    tuning_path = self._find_tuning_file(raw_model)
            except Exception as probe_err:
                print(f"[IMX708] Sensor model probe failed: {probe_err}")

            if tuning_path:
                # Layer 1: set env var BEFORE constructing Picamera2.
                # libcamera reads this at driver init time — guaranteed to load.
                os.environ["LIBCAMERA_RPI_TUNING_FILE"] = tuning_path
                print(f"[IMX708] Tuning file set via env: {tuning_path}")
            else:
                print("[IMX708] WARN: tuning file not found — colours may be incorrect.")
                print("[IMX708]       Run: sudo apt install -y rpicam-apps  (ships tuning files)")

            # Construct Picamera2 (libcamera now uses the tuning file from env)
            self._picam2 = Picamera2(self.camera_idx)

            # Query hardware properties
            try:
                self._props = self._picam2.camera_properties or {}
                model = self._props.get("Model")
                if model:
                    self._sensor_model = f"{model} (Camera Module 3)"
            except Exception:
                pass

            # ── Colour format ────────────────────────────────────────────
            # picamera2 format names are from OpenCV’s perspective, NOT raw
            # memory order.  ‘BGR888’ means “deliver bytes as B,G,R so OpenCV
            # can use the array directly without any conversion”.
            # DO NOT add cvtColor(COLOR_RGB2BGR) after this — the data is
            # already in OpenCV-native BGR order and any channel swap will
            # corrupt colours (reds become blue/purple).
            config = self._picam2.create_video_configuration(
                main={"format": "BGR888", "size": (self.width, self.height)},
                raw={"size": (4608, 2592)},   # Force full-sensor readout → 120° FOV
                controls={"FrameRate": self.fps},
                queue=False,
            )
            self._picam2.configure(config)
            self._picam2.start()

            # Reset ScalerCrop to full pixel array
            try:
                pixel_array_size = self._picam2.camera_properties.get(
                    "PixelArraySize", (4608, 2592)
                )
                self._picam2.set_controls({
                    "ScalerCrop": (0, 0, pixel_array_size[0], pixel_array_size[1])
                })
            except Exception as crop_err:
                print(f"[IMX708] ScalerCrop reset skipped: {crop_err}")

            # Configure autofocus (PDAF is supported by IMX708)
            self.set_autofocus(self.af_mode)

            # Warm-up: allow AWB and AEC to converge before we apply corrections
            time.sleep(1.5)

            # ── Layer 2: ColourGains correction for NoIR ─────────────────────
            # Applied AFTER AWB converges. If the tuning file loaded correctly
            # this won't be needed, but it acts as a guaranteed safety net.
            #
            # ColourGains = (red_gain, blue_gain).
            # NoIR cameras absorb IR into red channel → red appears over-boosted
            # → CCM compensates by pushing red toward blue → purple cast.
            # Reducing red gain slightly and keeping blue gain corrects this.
            if _is_noir:
                try:
                    # Read what AWB converged to
                    md = self._picam2.capture_metadata()
                    awb_r = md.get("ColourGains", (2.0, 1.8))[0]
                    awb_b = md.get("ColourGains", (2.0, 1.8))[1]
                    # Apply NoIR correction: reduce red, boost blue slightly
                    corrected_r = round(awb_r * 0.80, 3)   # -20% red to remove purple
                    corrected_b = round(awb_b * 1.10, 3)   # +10% blue
                    self._picam2.set_controls({
                        "AwbEnable": False,              # Lock gains so AWB doesn't undo fix
                        "ColourGains": (corrected_r, corrected_b),
                    })
                    print(f"[IMX708] NoIR ColourGains correction applied: "
                          f"R={corrected_r} (was {awb_r:.3f}), "
                          f"B={corrected_b} (was {awb_b:.3f})")
                except Exception as gains_err:
                    print(f"[IMX708] ColourGains correction skipped: {gains_err}")

            self._is_opened = True

        except Exception as exc:
            self._is_opened = False
            if self._picam2 is not None:
                try:
                    self._picam2.close()
                except Exception:
                    pass
                self._picam2 = None
            raise RuntimeError(f"Failed to initialize IMX708 camera: {exc}") from exc

    def set_autofocus(self, mode: Union[str, int]) -> bool:
        """
        Configure autofocus mode on the IMX708 VCM lens:
          - 'continuous' / 2: Continuous Phase Detection AF (PDAF) as scene changes
          - 'auto' / 1: Single-shot autofocus
          - 'manual' / 0: Fixed focus position
        """
        if self._picam2 is None:
            return False

        mode_str = str(mode).lower()
        try:
            if libcam_controls is not None and hasattr(libcam_controls, "AfModeEnum"):
                if mode_str in ("continuous", "2"):
                    val = libcam_controls.AfModeEnum.Continuous
                elif mode_str in ("auto", "1"):
                    val = libcam_controls.AfModeEnum.Auto
                else:
                    val = libcam_controls.AfModeEnum.Manual
            else:
                # Raw libcamera enum integers fallback
                val = 2 if mode_str in ("continuous", "2") else (1 if mode_str in ("auto", "1") else 0)

            self._picam2.set_controls({"AfMode": val})
            self.af_mode = mode_str
            return True
        except Exception as exc:
            print(f"[IMX708] Warning setting AfMode: {exc}")
            return False

    def isOpened(self) -> bool:
        return self._is_opened and (self._picam2 is not None)

    def read(self) -> Tuple[bool, Optional[Any]]:
        """
        Capture a frame as a BGR NumPy array, identical to cv2.VideoCapture.read().
        Returns: (success: bool, frame: np.ndarray)
        """
        if not self.isOpened():
            return False, None

        try:
            frame = self._picam2.capture_array("main")
            if frame is None or frame.size == 0:
                return False, None

            # BGR888 delivers B,G,R bytes — already OpenCV-native.
            # Only handle the rare 4-channel (XBGR/BGRA) output from some
            # libcamera builds by stripping the alpha channel.
            if _HAS_CV2 and frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            return True, frame
        except Exception:
            return False, None

    def get(self, prop_id: int) -> float:
        """Emulate cv2.VideoCapture.get()."""
        if _HAS_CV2:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
            if prop_id == cv2.CAP_PROP_FPS:
                return float(self.fps)
            if prop_id == getattr(cv2, "CAP_PROP_AUTOFOCUS", -1):
                return 1.0 if self.af_mode == "continuous" else 0.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        """Emulate cv2.VideoCapture.set()."""
        if _HAS_CV2:
            if prop_id == getattr(cv2, "CAP_PROP_AUTOFOCUS", -1):
                return self.set_autofocus("continuous" if value > 0.5 else "manual")
            if prop_id == cv2.CAP_PROP_BUFFERSIZE:
                return True
        return False

    def release(self) -> None:
        """Release camera and free system resources."""
        self._is_opened = False
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None

    def get_info(self) -> Dict[str, Any]:
        return {
            "model": self._sensor_model,
            "backend": "Picamera2 (libcamera)",
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "autofocus": f"{self.af_mode.capitalize()} (PDAF)",
            "hdr": self.hdr,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Alias for versatility
Picamera2Capture = IMX708CameraCapture


# ===========================================================================
# Unified Camera Probing & Scanning
# ===========================================================================

def probe_video_source(
    src: Any,
    target_w: int = 640,
    target_h: int = 480,
    target_fps: float = 30.0,
    af_mode: str = "continuous",
    hdr: bool = False,
) -> Tuple[Optional[Any], int, int, float, Dict[str, Any]]:
    """
    Probe and open a video source.
    Supports:
      - "imx708", "picam", "csi" -> Native IMX708 Picamera2
      - Integer index or "/dev/videoX" -> IMX708 on Unicam or V4L2 USB camera
      - Video file path -> cv2.VideoCapture file playback
      - GStreamer libcamerasrc pipeline

    Returns:
      (cap, width, height, fps, camera_info_dict)
    """
    cam_info: Dict[str, Any] = {
        "model": "Generic",
        "backend": "unknown",
        "is_imx708": False,
        "autofocus": "N/A",
    }

    if not _HAS_CV2:
        return None, 0, 0, 0.0, cam_info

    src_str = str(src).lower().strip()

    # ── 1. Explicit request for IMX708 / Picamera2 / CSI ──────────────────
    is_explicit_imx = src_str in ("imx708", "imx", "picam", "picamera", "picamera2", "csi", "cam3", "camera3")
    if is_explicit_imx:
        if not _HAS_PICAMERA2:
            print("[CAMERA] Picamera2 is not available. Install with: sudo apt install -y python3-picamera2")
            return None, 0, 0, 0.0, cam_info
        try:
            cap = IMX708CameraCapture(
                camera_idx=0,
                width=target_w,
                height=target_h,
                fps=target_fps,
                af_mode=af_mode,
                hdr=hdr,
            )
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                cam_info.update({
                    "model": cap.get_info()["model"],
                    "backend": "picamera2",
                    "is_imx708": True,
                    "autofocus": cap.get_info()["autofocus"],
                })
                return cap, w, h, target_fps, cam_info
            cap.release()
        except Exception as exc:
            print(f"[CAMERA] Native Picamera2 IMX708 probe failed: {exc}")
        return None, 0, 0, 0.0, cam_info

    # ── 2. Device Index or /dev/video* Path ────────────────────────────────
    dev_idx: Optional[int] = None
    if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
        dev_idx = int(src)
    elif isinstance(src, str) and src.startswith("/dev/video"):
        dev_name = os.path.basename(src)
        idx_str = dev_name.replace("video", "")
        if idx_str.isdigit():
            dev_idx = int(idx_str)

    if dev_idx is not None:
        # Check if this dev_idx is a CSI Unicam node on Raspberry Pi
        if is_unicam_device(dev_idx):
            # Unicam cannot be decoded directly by OpenCV V4L2; route to Picamera2!
            print(f"[CAMERA] /dev/video{dev_idx} is a CSI Unicam device. Routing to Picamera2 (IMX708)...")
            if _HAS_PICAMERA2:
                try:
                    cap = IMX708CameraCapture(
                        camera_idx=0,
                        width=target_w,
                        height=target_h,
                        fps=target_fps,
                        af_mode=af_mode,
                        hdr=hdr,
                    )
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        h, w = frame.shape[:2]
                        cam_info.update({
                            "model": cap.get_info()["model"],
                            "backend": "picamera2",
                            "is_imx708": True,
                            "autofocus": cap.get_info()["autofocus"],
                        })
                        return cap, w, h, target_fps, cam_info
                    cap.release()
                except Exception as exc:
                    print(f"[CAMERA] Picamera2 Unicam route failed: {exc}")

        # Standard V4L2 device probe
        if sys.platform.startswith("linux"):
            dev_node = f"/dev/video{dev_idx}"
            if not os.path.exists(dev_node) or not os.access(dev_node, os.R_OK):
                return None, 0, 0, 0.0, cam_info
            if not is_v4l2_capture_device(dev_idx):
                return None, 0, 0, 0.0, cam_info
            c = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
        else:
            c = cv2.VideoCapture(dev_idx)

        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            try:
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ret, test_frame = c.read()
            if ret and test_frame is not None and test_frame.size > 0:
                w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) or test_frame.shape[1]
                h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) or test_frame.shape[0]
                cam_fps = c.get(cv2.CAP_PROP_FPS) or target_fps
                if cam_fps <= 0 or cam_fps > 120:
                    cam_fps = target_fps
                cam_info.update({
                    "model": f"V4L2 Video Device {dev_idx}",
                    "backend": "v4l2",
                    "is_imx708": False,
                    "autofocus": "Fixed / USB",
                })
                return c, w, h, cam_fps, cam_info
            c.release()

    # ── 3. Prerecorded File / Stream URL ──────────────────────────────────
    try:
        c = cv2.VideoCapture(src)
        if c.isOpened():
            ret, test_frame = c.read()
            if ret and test_frame is not None and test_frame.size > 0:
                w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) or test_frame.shape[1]
                h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) or test_frame.shape[0]
                cam_fps = c.get(cv2.CAP_PROP_FPS) or target_fps
                if cam_fps <= 0 or cam_fps > 120:
                    cam_fps = target_fps
                cam_info.update({
                    "model": f"Video Source: {os.path.basename(str(src))}",
                    "backend": "file_or_stream",
                    "is_imx708": False,
                    "autofocus": "N/A",
                })
                return c, w, h, cam_fps, cam_info
            c.release()
    except Exception:
        pass

    return None, 0, 0, 0.0, cam_info


def get_candidate_video_ports(preferred_source: Any = None, prefer_imx708: bool = True) -> list:
    """
    Enumerate candidate video capture sources in priority order.
    Prioritizes IMX708 (Camera Module 3) when connected, followed by USB webcams.
    """
    candidates = []

    # Priority 1: User explicit preference
    if preferred_source is not None and str(preferred_source).strip() not in ("", "auto", "None"):
        src = preferred_source
        if isinstance(src, str) and src.isdigit():
            candidates.append(int(src))
        else:
            candidates.append(src)

    # Priority 2: Check for Raspberry Pi Camera Module 3 (IMX708)
    if prefer_imx708:
        imx_check = detect_imx708_sensor()
        if imx_check["detected"] or _HAS_PICAMERA2:
            if "imx708" not in candidates:
                candidates.append("imx708")

    # Priority 3: Scan Linux /dev/video* V4L2 devices
    if sys.platform.startswith("linux"):
        found_devs = []
        for p in sorted(glob.glob("/dev/video*")):
            dev_name = os.path.basename(p)
            idx_str = dev_name.replace("video", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                # Skip metadata and ISP nodes
                if is_v4l2_capture_device(idx):
                    found_devs.append(idx)
        for d in found_devs:
            if d not in candidates:
                candidates.append(d)

        # Always ensure standard indices are present as final fallback
        for idx in [0, 1]:
            if idx not in candidates:
                candidates.append(idx)
    else:
        # Windows / macOS development fallback
        for idx in [0, 1, 2]:
            if idx not in candidates:
                candidates.append(idx)

    return candidates


def scan_for_camera(
    candidates: list,
    target_w: int = 640,
    target_h: int = 480,
    target_fps: float = 30.0,
    af_mode: str = "continuous",
    hdr: bool = False,
) -> Tuple[Optional[Any], Any, int, int, float, Dict[str, Any]]:
    """
    Scan candidate video sources and return the first successfully opened camera.
    Returns:
      (cap, source_identifier, width, height, fps, camera_info)
    """
    for cand in candidates:
        cap, w, h, fps, info = probe_video_source(
            src=cand,
            target_w=target_w,
            target_h=target_h,
            target_fps=target_fps,
            af_mode=af_mode,
            hdr=hdr,
        )
        if cap is not None:
            return cap, cand, w, h, fps, info

    return None, None, 0, 0, 0.0, {
        "model": "None", "backend": "none", "is_imx708": False, "autofocus": "N/A"
    }
